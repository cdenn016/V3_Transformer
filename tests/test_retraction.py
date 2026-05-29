import torch

from vfe3.geometry.retraction import (
    natural_gradient,
    retract_spd_diagonal,
    retract_spd_full,
)


def test_diagonal_retraction_positive_and_bounded():
    g = torch.Generator().manual_seed(0)
    sigma = torch.rand(4, 6, generator=g) + 0.1
    delta = 5.0 * torch.randn(4, 6, generator=g)
    out = retract_spd_diagonal(sigma, delta, sigma_max=5.0)
    assert (out >= 1e-6).all()
    assert (out <= 5.0 + 1e-6).all()


def test_full_retraction_stays_spd():
    g = torch.Generator().manual_seed(1)
    A = torch.randn(3, 4, 4, generator=g)
    sigma = A @ A.transpose(-1, -2) + torch.eye(4)
    D = torch.randn(3, 4, 4, generator=g)
    delta = 0.5 * (D + D.transpose(-1, -2))
    out = retract_spd_full(sigma, delta)
    assert torch.allclose(out, out.transpose(-1, -2), atol=1e-4)
    assert (torch.linalg.eigvalsh(out) > 0).all()


def test_full_retraction_identity_tangent_is_identity():
    g = torch.Generator().manual_seed(2)
    A = torch.randn(3, 4, 4, generator=g)
    sigma = A @ A.transpose(-1, -2) + torch.eye(4)
    zero = torch.zeros(3, 4, 4)
    out = retract_spd_full(sigma, zero)
    assert torch.allclose(out, sigma, atol=1e-3)


def test_natural_gradient_diagonal_formula():
    g = torch.Generator().manual_seed(3)
    sigma = torch.rand(4, 5, generator=g) + 0.1
    gmu = torch.randn(4, 5, generator=g)
    gsig = torch.randn(4, 5, generator=g)
    nmu, nsig = natural_gradient(gmu, gsig, sigma)
    assert torch.allclose(nmu, sigma * gmu, atol=1e-6)
    assert torch.allclose(nsig, 2.0 * sigma * sigma * gsig, atol=1e-6)


def test_full_retraction_K1_matches_diagonal_formula():
    # For K=1 the affine-invariant SPD exp map reduces to the diagonal rule
    # sigma_new = sigma * exp(tau * delta/sigma).
    sigma = torch.tensor([[[2.0]]])          # (1,1,1) as (B,K,K) with K=1
    delta = torch.tensor([[[0.6]]])
    out = retract_spd_full(sigma, delta, trust_region=0.0)   # disable TR for exact check
    expected = 2.0 * torch.exp(torch.tensor(0.6 / 2.0))
    assert torch.allclose(out.reshape(()), expected, atol=1e-4)
