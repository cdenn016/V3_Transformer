import torch

from vfe3.attention_prior import attention_log_prior


def test_uniform_is_zero_bias():
    B = attention_log_prior("uniform", 4, 4)
    assert torch.allclose(B, torch.zeros(4, 4))


def test_causal_masks_future_keys():
    B = attention_log_prior("causal", 3, 3)
    # j > i masked (-inf), j <= i allowed (0)
    assert torch.isneginf(B[0, 1]) and torch.isneginf(B[0, 2]) and torch.isneginf(B[1, 2])
    assert B[2, 0] == 0.0 and B[1, 1] == 0.0 and B[2, 2] == 0.0


def test_alibi_is_linear_in_distance():
    B = attention_log_prior("alibi", 4, 4, slope=0.5)
    # B_ij = -slope * |i - j|
    for i in range(4):
        for j in range(4):
            assert torch.isclose(B[i, j], torch.tensor(-0.5 * abs(i - j)), atol=1e-6)
