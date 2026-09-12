import torch
from transformers import AutoModelForCausalLM


def extract_lora_targets(config):
    # placeholder helper: extract target module names for LoRA from config
    return getattr(config, "lora_target", [])


class GQAConverter:
    def __init__(self, model, model_config, user_config):
        self.state_dict = model.state_dict()
        self.num_att_heads = model_config.num_attention_heads

        # Real number of KV heads from model config (or default to num_att_heads for MHA)
        self.source_kv_heads = getattr(model_config, "num_key_value_heads", None) or self.num_att_heads
        self.num_kv_groups = user_config.target_kv_groups

        if self.source_kv_heads != self.num_att_heads and not getattr(user_config, "allow_non_mha", False):
            raise ValueError(
                f"Source model has num_key_value_heads={self.source_kv_heads} != "
                f"num_attention_heads={self.num_att_heads} - it's already GQA/MQA, not MHA. "
                "Set allow_non_mha=True on GQAUserConfig to compress it further anyway."
            )

        self.model_output_path = getattr(user_config, "model_save_path", None) or getattr(user_config, "save_path", "./gqa_model_output")
        self.model_dtype = model.dtype
        self.hidden_size = model_config.hidden_size
        self.head_dim = self.hidden_size // self.num_att_heads

    def _permutate_heads(self, weights, permutation, dim=0):
        num_heads = len(permutation)
        if dim == 0:
            blocks = weights.view(num_heads, self.head_dim, -1)
            return blocks[permutation].reshape(num_heads * self.head_dim, -1)
        else:
            blocks = weights.view(-1, num_heads, self.head_dim)
            return blocks[:, permutation, :].reshape(-1, num_heads * self.head_dim)
    
    def _permute_bias(self, bias, permutation):
        blocks = bias.view(len(permutation), self.head_dim)
        return blocks[permutation].reshape(-1)

    def reconfig(self, model_weight_path, per_layer_grouping: dict | None = None):
        """Convert all layers using per-layer activation-similarity groupings.

        Args:
            model_weight_path: layer-index -> weight key mapping from the detector.
            per_layer_grouping: dict mapping layer_idx -> list of head-index tensors
                (one tensor per KV group). Produced by Calibration.calibrate().
                If None, falls back to contiguous equal-sized grouping.
        """
        for layer_idx, paths in model_weight_path.items():
            layer_grouping = per_layer_grouping.get(layer_idx) if per_layer_grouping is not None else None

            # Derive 1-D permutation from grouping: concat group tensors in order
            # [group0_heads, group1_heads, ...] -> flat head-index permutation
            layer_1d_perm = (
                torch.cat([g if isinstance(g, torch.Tensor) else torch.tensor(g) for g in layer_grouping]).flatten()
                if layer_grouping is not None
                else None
            )

            k_weights = self.state_dict[paths["k_weight"]]
            self.state_dict[paths["k_weight"]] = self.mha_to_gqa_converter(k_weights, grouping=layer_grouping)

            if "k_bias" in paths:
                self.state_dict[paths["k_bias"]] = self.mha_to_gqa_bias_converter(self.state_dict[paths["k_bias"]], grouping=layer_grouping)

            v_weights = self.state_dict[paths["v_weight"]]
            self.state_dict[paths["v_weight"]] = self.mha_to_gqa_converter(v_weights, grouping=layer_grouping)

            if "v_bias" in paths:
                self.state_dict[paths["v_bias"]] = self.mha_to_gqa_bias_converter(self.state_dict[paths["v_bias"]], grouping=layer_grouping)

            if layer_1d_perm is not None:
                if "q_weight" in paths and paths["q_weight"] in self.state_dict:
                    self.state_dict[paths["q_weight"]] = self._permutate_heads(
                        self.state_dict[paths["q_weight"]], layer_1d_perm, dim=0
                    )
                if "q_bias" in paths and paths["q_bias"] in self.state_dict:
                    self.state_dict[paths["q_bias"]] = self._permute_bias(
                        self.state_dict[paths["q_bias"]], layer_1d_perm
                    )
                if "o_weight" in paths and paths["o_weight"] in self.state_dict:
                    self.state_dict[paths["o_weight"]] = self._permutate_heads(
                        self.state_dict[paths["o_weight"]], layer_1d_perm, dim=1
                    )

        return self.state_dict


    def mha_to_gqa_converter(self, mha_weights, grouping=None):
        source_kv_heads = getattr(self, "source_kv_heads", None) or self.num_att_heads
        head_dim = self.hidden_size // self.num_att_heads

        mha_weights_splitted = mha_weights.reshape(source_kv_heads, head_dim, self.hidden_size)

        if grouping is not None:
            group_weights = []
            for head_indices in grouping:
                g_weight = mha_weights_splitted[head_indices].mean(dim=0)
                group_weights.append(g_weight)
            gqa_weights = torch.stack(group_weights, dim=0).reshape(-1, self.hidden_size)
        else:
            if source_kv_heads % self.num_kv_groups != 0:
                raise ValueError(f"Source KV heads ({source_kv_heads}) must be divisible by target KV groups ({self.num_kv_groups})")
            heads_per_group = source_kv_heads // self.num_kv_groups
            preprocces_gqa_weights = mha_weights_splitted.reshape(self.num_kv_groups, heads_per_group, head_dim, self.hidden_size)
            gqa_weights = preprocces_gqa_weights.mean(dim=1).reshape(-1, self.hidden_size)

        return gqa_weights.clone()

    def mha_to_gqa_bias_converter(self, mha_bias, grouping=None):
        source_kv_heads = getattr(self, "source_kv_heads", None) or self.num_att_heads
        head_dim = self.hidden_size // self.num_att_heads

        bias_splitted = mha_bias.reshape(source_kv_heads, head_dim)

        if grouping is not None:
            group_biases = []
            for head_indices in grouping:
                g_bias = bias_splitted[head_indices].mean(dim=0)
                group_biases.append(g_bias)
            gqa_bias = torch.stack(group_biases, dim=0).reshape(-1)
        else:
            if source_kv_heads % self.num_kv_groups != 0:
                raise ValueError(f"Source KV heads ({source_kv_heads}) must be divisible by target KV groups ({self.num_kv_groups})")
            heads_per_group = source_kv_heads // self.num_kv_groups
            bias_grouped = bias_splitted.reshape(self.num_kv_groups, heads_per_group, head_dim)
            gqa_bias = bias_grouped.mean(dim=1).reshape(-1)

        return gqa_bias.clone()

    def save_gqa_model(self, config):
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
