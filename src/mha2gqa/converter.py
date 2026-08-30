from transformers import AutoModelForCausalLM


def extract_lora_targets(config):
    # placeholder helper: extract target module names for LoRA from config
    return getattr(config, "lora_target", [])


class GQAConverter:
    def __init__(self, model, model_config, user_config):
        self.state_dict = model.state_dict()
        self.num_att_heads = model_config.num_attention_heads
        
        # 1. Берем РЕАЛЬНОЕ количество KV-голов из конфига модели. 
        # Если параметра нет (как у старых моделей), значит это MHA и оно равно num_att_heads
        self.source_kv_heads = getattr(model_config, "num_key_value_heads", None) or self.num_att_heads
        
        self.num_kv_groups = user_config.target_kv_groups
        
        # 2. ИСПРАВЛЕНО: проверяем именно ИСХОДНУЮ модель (source_kv_heads), а не целевые группы
        if self.source_kv_heads != self.num_att_heads and not getattr(user_config, "allow_non_mha", False):
            raise ValueError(
                f"Source model has num_key_value_heads={self.source_kv_heads} != "
                f"num_attention_heads={self.num_att_heads} - it's already GQA/MQA, not MHA. "
                "Set allow_non_mha=True on GQAUserConfig to compress it further anyway."
            )
            
        self.model_output_path = getattr(user_config, "model_save_path", None) or getattr(user_config, "save_path", "./gqa_model_output")
        self.model_dtype = model.dtype
        self.hidden_size = model_config.hidden_size


    def reconfig(self, model_weight_path):
        for layer_prefix, paths in model_weight_path.items():
            k_weights = self.state_dict[paths["k_weight"]]
            self.state_dict[paths["k_weight"]] = self.mha_to_gqa_converter(k_weights)

            if "k_bias" in paths:
                k_bias = self.state_dict[paths["k_bias"]]
                self.state_dict[paths["k_bias"]] = self.mha_to_gqa_bias_converter(k_bias)

            v_weights = self.state_dict[paths["v_weight"]]
            self.state_dict[paths["v_weight"]] = self.mha_to_gqa_converter(v_weights)

            if "v_bias" in paths:
                v_bias = self.state_dict[paths["v_bias"]]
                self.state_dict[paths["v_bias"]] = self.mha_to_gqa_bias_converter(v_bias)

        return self.state_dict

    def mha_to_gqa_converter(self, mha_weights):
        if self.source_kv_heads % self.num_kv_groups != 0:
            raise ValueError(f"Source KV heads ({self.source_kv_heads}) must be divisible by target KV groups ({self.num_kv_groups})")

        head_dim = self.hidden_size // self.num_att_heads
        heads_per_group = self.source_kv_heads // self.num_kv_groups

        mha_weights_splitted = mha_weights.reshape(self.source_kv_heads, head_dim, self.hidden_size)
        preprocces_gqa_weights = mha_weights_splitted.reshape(self.num_kv_groups, heads_per_group, head_dim, self.hidden_size)

        gqa_weights = preprocces_gqa_weights.mean(dim=1).reshape(-1, self.hidden_size)
        return gqa_weights.clone()

    def mha_to_gqa_bias_converter(self, mha_bias):
        head_dim = self.hidden_size // self.num_att_heads
        heads_per_group = self.source_kv_heads // self.num_kv_groups

        bias_splitted = mha_bias.reshape(self.source_kv_heads, head_dim)
        bias_grouped = bias_splitted.reshape(self.num_kv_groups, heads_per_group, head_dim)

        gqa_bias = bias_grouped.mean(dim=1).reshape(-1)
        return gqa_bias.clone()

    def save_gqa_model(self,config):
        config.num_key_value_heads = int(self.num_kv_groups)
        # Instantiate model from config (avoid passing backend-specific kwargs)
        gqa_model = AutoModelForCausalLM.from_config(config)
        # Cast dtype if possible
        try:
            gqa_model.to(dtype=self.model_dtype)
        except Exception:
            pass
        gqa_model.load_state_dict(self.state_dict, strict=True)

        # Ensure generation/pad token ids are valid to avoid save errors
        try:
            pad_id = getattr(config, "pad_token_id", None)
            if pad_id is None or (isinstance(pad_id, int) and pad_id < 0):
                # fall back to 0 or eos_token_id if present
                pad_id = getattr(config, "eos_token_id", 0) or 0
                config.pad_token_id = int(pad_id)
            if hasattr(gqa_model, "generation_config"):
                gen = gqa_model.generation_config
                if getattr(gen, "pad_token_id", None) is None or (isinstance(gen.pad_token_id, int) and gen.pad_token_id < 0):
                    gen.pad_token_id = int(config.pad_token_id)
        except Exception:
            pass

        gqa_model.save_pretrained(self.model_output_path)

