import pytest
import torch


def test_retract_spd_diagonal_matches_vfe2(vfe2_retract):
    from vfe3.geometry.retraction import retract_spd_diagonal
    g = torch.Generator(device="cpu").manual_seed(0)
    sigma = torch.rand(2, 3, 5, generator=g) + 0.1
    delta = 0.5 * torch.randn(2, 3, 5, generator=g)
    ref = vfe2_retract["vfe_utils"].retract_spd_diagonal_torch(
        sigma, delta, step_size=1.0, trust_region=5.0, eps=1e-6, sigma_max=5.0
    )
    got = retract_spd_diagonal(sigma, delta)
    assert torch.allclose(got, ref, atol=1e-5, rtol=1e-5)


def test_retract_spd_full_matches_vfe2(vfe2_retract):
    from vfe3.geometry.retraction import retract_spd_full
    g = torch.Generator(device="cpu").manual_seed(1)
    A = torch.randn(2, 3, 4, 4, generator=g)
    sigma = A @ A.transpose(-1, -2) + torch.eye(4)
    D = torch.randn(2, 3, 4, 4, generator=g)
    delta = 0.3 * (D + D.transpose(-1, -2))
    ref = vfe2_retract["vfe_utils"].retract_spd_torch(
        sigma, delta, step_size=1.0, trust_region=2.0, eps=1e-6, sigma_max=5.0
    )
    got = retract_spd_full(sigma, delta)
    assert torch.allclose(got, ref, atol=1e-3, rtol=1e-3)


def test_natural_gradient_diag_matches_vfe2(vfe2_retract):
    from vfe3.geometry.retraction import natural_gradient
    g = torch.Generator(device="cpu").manual_seed(2)
    sigma = torch.rand(2, 3, 5, generator=g) + 0.1
    gmu = torch.randn(2, 3, 5, generator=g)
    gsig = torch.randn(2, 3, 5, generator=g)
    rmu, rsig = vfe2_retract["vfe_gradients"].compute_natural_gradient_gpu(gmu, gsig, sigma)
    nmu, nsig = natural_gradient(gmu, gsig, sigma)
    assert torch.allclose(nmu, rmu, atol=1e-5, rtol=1e-5)
    assert torch.allclose(nsig, rsig, atol=1e-5, rtol=1e-5)


def test_natural_gradient_full_matches_vfe2(vfe2_retract):
    from vfe3.geometry.retraction import natural_gradient
    g = torch.Generator(device="cpu").manual_seed(3)
    A = torch.randn(2, 3, 4, 4, generator=g)
    sigma = A @ A.transpose(-1, -2) + torch.eye(4)
    gmu = torch.randn(2, 3, 4, generator=g)
    Gs = torch.randn(2, 3, 4, 4, generator=g)
    gsig = 0.5 * (Gs + Gs.transpose(-1, -2))
    rmu, rsig = vfe2_retract["vfe_gradients"].compute_natural_gradient_gpu(gmu, gsig, sigma)
    nmu, nsig = natural_gradient(gmu, gsig, sigma)
    assert torch.allclose(nmu, rmu, atol=1e-4, rtol=1e-4)
    assert torch.allclose(nsig, rsig, atol=1e-4, rtol=1e-4)
