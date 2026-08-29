from typing import List

from .adapters import (
    Gemma2Adapter,
    GemmaAdapter,
    GPTJAdapter,
    LlamaAdapter,
    MistralAdapter,
    MixtralAdapter,
    Qwen2Adapter,
    StableLMAdapter,
)
from .base import ArchitectureAdapter

ARCHITECTURE_REGISTRY: list[ArchitectureAdapter] = [
    LlamaAdapter(),
    MistralAdapter(),
    MixtralAdapter(),
    Qwen2Adapter(),
    GemmaAdapter(),
    Gemma2Adapter(),
    StableLMAdapter(),
    GPTJAdapter(),
]

def get_adapter(model_type: str) -> ArchitectureAdapter:
    for adapter in ARCHITECTURE_REGISTRY:
        if adapter.matches(model_type):
            return adapter
    return None
