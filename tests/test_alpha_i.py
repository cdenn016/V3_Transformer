import torch

from vfe3.alpha_i import alpha_regularizer, self_coupling_alpha


def test_constant_alpha_is_value_zero_reg():
    kl = torch.rand(3, 5)
    a, r = self_coupling_alpha(kl, mode="constant", value=1.0)
    assert torch.allclose(a, torch.ones(3, 5))
    assert torch.allclose(r, torch.zeros(3, 5))


def test_state_dependent_alpha_formula_and_minimizes_objective():
    # alpha* = c0/(b0 + KL) is the stationary point of  alpha*KL + b0*alpha - c0*log(alpha).
    kl = torch.tensor([0.0, 1.0, 4.0])
    b0, c0 = 0.5, 2.0
    a, r = self_coupling_alpha(kl, mode="state_dependent", b0=b0, c0=c0)
    assert torch.allclose(a, c0 / (b0 + kl), atol=1e-6)
    # d/d alpha [alpha*KL + b0*alpha - c0*log alpha] = KL + b0 - c0/alpha == 0 at alpha*
    grad = kl + b0 - c0 / a
    assert torch.allclose(grad, torch.zeros_like(grad), atol=1e-5)


def test_per_coord_alpha_uses_per_dimension_kl():
    kl = torch.rand(2, 4, 3) + 0.1                       # (..., N, K) per-coordinate KL
    b0 = torch.full((3,), 0.5)
    c0 = torch.full((3,), 2.0)
    a, r = self_coupling_alpha(kl, mode="state_dependent_per_coord", b0=b0, c0=c0)
    assert torch.allclose(a, c0 / (b0 + kl), atol=1e-6)
