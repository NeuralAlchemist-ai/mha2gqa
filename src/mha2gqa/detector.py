import logging
import re

from .architectures import get_adapter

logger = logging.getLogger(__name__)

class AutoArchitectureDetector:
    def __init__(self, model_config, model):
        self.state_dict = model.state_dict()
        self.model_type = model_config.model_type
        self.num_layers = model_config.num_hidden_layers
        self.model_weight_path = {}

    def dynamic_search(self):
        self.model_weight_path = {}

        adapter = get_adapter(self.model_type)
        if adapter:
            logger.info("Architecture %s found in registry. Using adapter.", self.model_type)
            mapping_template = adapter.layer_mapping(self.num_layers)
        else:
            logger.info("Architecture %s not found in registry. Falling back to heuristic search...", self.model_type)
            mapping_template = self._fallback_search()

        for i in range(self.num_layers):
            current_layer_path = {}
            layer_info = mapping_template[i]
            
            k_w_key = layer_info['k_weight']
            v_w_key = layer_info['v_weight']
            k_b_key = layer_info['k_bias']
            v_b_key = layer_info['v_bias']

            if k_w_key in self.state_dict:
                current_layer_path["k_weight"] = k_w_key
                if k_b_key in self.state_dict:
                    current_layer_path["k_bias"] = k_b_key

            if v_w_key in self.state_dict:
                current_layer_path["v_weight"] = v_w_key
                if v_b_key in self.state_dict:
                    current_layer_path["v_bias"] = v_b_key

            if current_layer_path:
                self.model_weight_path[i] = current_layer_path

        return self.model_weight_path


    def _fallback_search(self):
        k_patterns = [r'\.k\.', r'key', r'k_proj']
        v_patterns = [r'\.v\.', r'value', r'v_proj']
        qkv_patterns = [r'qkv', r'query_key_value', r'in_proj']

        found_k_w = None
        found_v_w = None

        for key in self.state_dict.keys():
            if not key.endswith('.weight') or not re.search(r'\b0\b', key):
                continue

            if any(re.search(p, key, re.IGNORECASE) for p in qkv_patterns):
                raise NotImplementedError(
                    f"Fused-QKV GQA conversion for '{self.model_type}' isn't supported. "
                    "Some fused-QKV architectures (e.g. Phi3) lay out Q/K/V as flat "
                    "contiguous blocks; others (e.g. Falcon) interleave them per KV-group; "
                    "some (e.g. GPT-2, GPT-NeoX) don't support GQA at all. This tool "
                    "currently only supports separate q_proj/k_proj/v_proj architectures."
                )

            if any(re.search(p, key, re.IGNORECASE) for p in k_patterns):
                found_k_w = key
            elif any(re.search(p, key, re.IGNORECASE) for p in v_patterns):
                found_v_w = key

        if not (found_k_w and found_v_w):
            raise ValueError(f"Could not determine weight structure for {self.model_type}")

        logger.info("Detected separate Q/K/V architecture.")
        match = re.search(r'\.0\.', found_k_w)
        if not match:
            raise ValueError("Atypical layer index format found.")

        zero_start, zero_end = match.span()
        layer_prefix = f"{found_k_w[:zero_start]}.{{i}}"
        k_suffix = found_k_w[zero_end:].replace('.weight', '')
        v_suffix = found_v_w[zero_end:].replace('.weight', '')

        k_bias = "bias" if found_k_w.replace('.weight', '.bias') in self.state_dict else "nonexistent_bias_key"
        v_bias = "bias" if found_v_w.replace('.weight', '.bias') in self.state_dict else "nonexistent_bias_key"

        mapping = {}
        for i in range(self.num_layers):
            prefix = layer_prefix.format(i=i)
            mapping[i] = {
                "k_weight": f"{prefix}.{k_suffix}.weight",
                "v_weight": f"{prefix}.{v_suffix}.weight",
                "k_bias": f"{prefix}.{k_suffix}.{k_bias}",
                "v_bias": f"{prefix}.{v_suffix}.{v_bias}",
            }
        return mapping
