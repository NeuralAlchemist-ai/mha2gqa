from typing import Protocol


class ArchitectureAdapter(Protocol):
    def matches(self, model_type: str) -> bool:
        ...

    def layer_mapping(self, num_layers: int) -> dict[int, dict[str, str]]:
        ...

class StandardAdapter:
    def __init__(self, supported_types, layer_prefix, k_weight, v_weight, k_bias, v_bias):
        self.supported_types = supported_types
        self.layer_prefix = layer_prefix
        self.k_weight = k_weight
        self.v_weight = v_weight
        self.k_bias = k_bias
        self.v_bias = v_bias

    def matches(self, model_type: str) -> bool:
        return model_type in self.supported_types

    def layer_mapping(self, num_layers: int) -> dict[int, dict[str, str]]:
        mapping = {}
        for i in range(num_layers):
            prefix = self.layer_prefix.format(i=i)
            mapping[i] = {
                "k_weight": f"{prefix}.{self.k_weight}",
                "v_weight": f"{prefix}.{self.v_weight}",
                "k_bias": f"{prefix}.{self.k_bias}",
                "v_bias": f"{prefix}.{self.v_bias}",
            }
        return mapping
