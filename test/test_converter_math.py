import torch

from src.mha2gqa import GQAConverter


def test_mha_to_gqa_converter_mean_pooling():
    hidden_size, num_heads, num_groups = 4, 2, 1
    head0 = torch.full((2, hidden_size), 1.0)
    head1 = torch.full((2, hidden_size), 3.0)
    mha_weights = torch.cat([head0, head1], dim=0)  # shape (4, 4)

    converter = GQAConverter.__new__(GQAConverter)  # bypass __init__, no model/config needed
    converter.hidden_size = hidden_size
    converter.num_att_heads = num_heads
    converter.num_kv_groups = num_groups

    result = converter.mha_to_gqa_converter(mha_weights)
    expected = torch.full((2, hidden_size), 2.0)  # mean of 1.0 and 3.0
    assert torch.allclose(result, expected)