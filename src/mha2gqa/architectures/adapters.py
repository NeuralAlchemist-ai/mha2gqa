from .base import StandardAdapter


class LlamaAdapter(StandardAdapter):
    def __init__(self):
        super().__init__(
            supported_types={"llama"},
            layer_prefix="model.layers.{i}",
            k_weight="self_attn.k_proj.weight",
            v_weight="self_attn.v_proj.weight",
            k_bias="self_attn.k_proj.bias",
            v_bias="self_attn.v_proj.bias"
        )

class MistralAdapter(StandardAdapter):
    def __init__(self):
        super().__init__(
            supported_types={"mistral"},
            layer_prefix="model.layers.{i}",
            k_weight="self_attn.k_proj.weight",
            v_weight="self_attn.v_proj.weight",
            k_bias="self_attn.k_proj.bias",
            v_bias="self_attn.v_proj.bias"
        )

class MixtralAdapter(StandardAdapter):
    def __init__(self):
        super().__init__(
            supported_types={"mixtral"},
            layer_prefix="model.layers.{i}",
            k_weight="self_attn.k_proj.weight",
            v_weight="self_attn.v_proj.weight",
            k_bias="self_attn.k_proj.bias",
            v_bias="self_attn.v_proj.bias"
        )

class Qwen2Adapter(StandardAdapter):
    def __init__(self):
        super().__init__(
            supported_types={"qwen2"},
            layer_prefix="model.layers.{i}",
            k_weight="self_attn.k_proj.weight",
            v_weight="self_attn.v_proj.weight",
            k_bias="self_attn.k_proj.bias",
            v_bias="self_attn.v_proj.bias"
        )

class GemmaAdapter(StandardAdapter):
    def __init__(self):
        super().__init__(
            supported_types={"gemma"},
            layer_prefix="model.layers.{i}",
            k_weight="self_attn.k_proj.weight",
            v_weight="self_attn.v_proj.weight",
            k_bias="self_attn.k_proj.bias",
            v_bias="self_attn.v_proj.bias"
        )

class Gemma2Adapter(StandardAdapter):
    def __init__(self):
        super().__init__(
            supported_types={"gemma2"},
            layer_prefix="model.layers.{i}",
            k_weight="self_attn.k_proj.weight",
            v_weight="self_attn.v_proj.weight",
            k_bias="self_attn.k_proj.bias",
            v_bias="self_attn.v_proj.bias"
        )

class StableLMAdapter(StandardAdapter):
    def __init__(self):
        super().__init__(
            supported_types={"stablelm"},
            layer_prefix="model.layers.{i}",
            k_weight="self_attn.k_proj.weight",
            v_weight="self_attn.v_proj.weight",
            k_bias="self_attn.k_proj.bias",
            v_bias="self_attn.v_proj.bias"
        )

class GPTJAdapter(StandardAdapter):
    def __init__(self):
        super().__init__(
            supported_types={"gptj"},
            layer_prefix="transformer.h.{i}",
            k_weight="attn.k_proj.weight",
            v_weight="attn.v_proj.weight",
            k_bias="attn.k_proj.bias",
            v_bias="attn.v_proj.bias"
        )
