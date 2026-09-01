from dataclasses import dataclass, field

import torch


@dataclass
class GQAUserConfig:
    model_id: str
    data_path: str | None = None
    dataset_name: str | None = None
    allow_non_mha: bool = False

    target_kv_groups: int = 8
    save_path: str = "./gqa_model_output"
    
    # Настройки LoRA / QLoRA
    lora_rank: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_output_dir: str = "./gqa_lora_output"
    lora_target: list[str] = field(default_factory=list) # Сюда запишем auto-targets
    
    # Настройки обучения
    batch_size: int = 32
    accumulation_steps: int = 1
    lr: float = 2e-4
    epochs: int = 3
    model_dtype: torch.dtype = torch.bfloat16

    def __post_init__(self):
        if self.target_kv_groups <= 0:
            raise ValueError("target_kv_groups must be positive")
