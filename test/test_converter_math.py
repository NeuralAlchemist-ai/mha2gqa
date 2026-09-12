import torch
from src.mha2gqa import GQAConverter


def test_mha_to_gqa_converter_mean_pooling():
    hidden_size, num_heads, num_groups = 4, 2, 1
    head0 = torch.full((2, hidden_size), 1.0)
    head1 = torch.full((2, hidden_size), 3.0)
    mha_weights = torch.cat([head0, head1], dim=0)  # shape (4, 4)

    converter = GQAConverter.__new__(GQAConverter)  # bypass __init__
    converter.hidden_size = hidden_size
    converter.num_att_heads = num_heads
    converter.num_kv_groups = num_groups
    converter.head_dim = 2
    converter.grouping = [torch.arange(num_heads)]

    result = converter.mha_to_gqa_converter(mha_weights)
    expected = torch.full((2, hidden_size), 2.0)  # mean of 1.0 and 3.0
    assert torch.allclose(result, expected)


def test_q_o_proj_permutation():
    # 4 heads, head_dim=2, hidden_size=4
    # heads: 0, 1, 2, 3
    # permutation: [2, 0, 3, 1]
    num_heads = 4
    head_dim = 2
    hidden_size = 4
    permutation = [2, 0, 3, 1]

    converter = GQAConverter.__new__(GQAConverter)
    converter.hidden_size = hidden_size
    converter.num_att_heads = num_heads
    converter.head_dim = head_dim
    converter.num_kv_groups = 2
    converter.grouping = None

    # Dummy q_weight of shape (8, 4) -> (4 heads, 2 head_dim, 4 hidden)
    # Head 0: all 0s, Head 1: all 1s, Head 2: all 2s, Head 3: all 3s
    q_blocks = [torch.full((head_dim, hidden_size), float(h)) for h in range(num_heads)]
    q_weight = torch.cat(q_blocks, dim=0)  # shape (8, 4)

    q_permuted = converter._permutate_heads(q_weight, permutation, dim=0)
    # Expected order after permutation [2, 0, 3, 1]:
    # Head 0 slot -> Head 2 (2s)
    # Head 1 slot -> Head 0 (0s)
    # Head 2 slot -> Head 3 (3s)
    # Head 3 slot -> Head 1 (1s)
    expected_q_blocks = [torch.full((head_dim, hidden_size), float(h)) for h in permutation]
    expected_q = torch.cat(expected_q_blocks, dim=0)
    assert torch.allclose(q_permuted, expected_q)

    # Dummy o_weight of shape (4, 8) -> (4 hidden, 4 heads, 2 head_dim)
    o_weight = torch.zeros((hidden_size, num_heads * head_dim))
    for idx, h in enumerate(range(num_heads)):
        o_weight[:, h * head_dim : (h + 1) * head_dim] = float(h)

    o_permuted = converter._permutate_heads(o_weight, permutation, dim=1)
    for slot_idx, original_head in enumerate(permutation):
        slice_vals = o_permuted[:, slot_idx * head_dim : (slot_idx + 1) * head_dim]
        assert torch.allclose(slice_vals, torch.full((hidden_size, head_dim), float(original_head)))