import pytest
import torch

from src.mha2gqa import AutoArchitectureDetector


def test_detector_rejects_fused_qkv():
    fake_state_dict = {"transformer.h.0.attn.query_key_value.weight": torch.randn(12, 4)}
    class FakeModel:
        def state_dict(self): return fake_state_dict
    class FakeConfig:
        model_type = "totally_unknown_arch"
        num_hidden_layers = 1

    detector = AutoArchitectureDetector(FakeConfig(), FakeModel())
    with pytest.raises(NotImplementedError):
        detector.dynamic_search()