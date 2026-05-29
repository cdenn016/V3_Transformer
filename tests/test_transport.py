import pytest
import torch

from vfe3.geometry.groups import get_group
from vfe3.geometry.transport import (
    compute_transport_operators,
    transport_covariance,
    transport_mean,
)


def _omega(seed, K=4):
    grp = get_group("so_k")(K=K)
    g = torch.Generator().manual_seed(seed)
    phi = 0.3 * torch.randn(2, 3, grp.generators.shape[0], generator=g)
    return compute_transport_operators(phi, grp, gauge_mode="learned")["Omega"], g


def test_transport_mean_identity_at_phi_zero():
    grp = get_group("so_k")(K=4)
    phi = torch.zeros(2, 3, grp.generators.shape[0])
    omega = compute_transport_operators(phi, grp, gauge_mode="learned")["Omega"]
    g = torch.Generator().manual_seed(0)
    mu = torch.randn(2, 3, 4, generator=g)
    mu_t = transport_mean(omega, mu)
    assert torch.allclose(mu_t, mu.unsqueeze(1).expand(2, 3, 3, 4), atol=1e-5)


def test_transport_covariance_full_is_spd():
    omega, g = _omega(1)
    A = torch.randn(2, 3, 4, 4, generator=g)
    sigma = A @ A.transpose(-1, -2) + torch.eye(4)
    sigma_t = transport_covariance(omega, sigma)
    assert torch.allclose(sigma_t, sigma_t.transpose(-1, -2), atol=1e-4)
    assert (torch.linalg.eigvalsh(sigma_t) > 0).all()


def test_transport_covariance_diag_matches_full_diagonal():
    omega, g = _omega(2)
    sigma_diag = torch.rand(2, 3, 4, generator=g) + 0.1
    full = transport_covariance(omega, torch.diag_embed(sigma_diag))
    approx = transport_covariance(omega, sigma_diag)
    assert torch.allclose(approx, torch.diagonal(full, dim1=-2, dim2=-1), atol=1e-5)


def test_transport_covariance_diag_matches_vfe2_formula():
    omega, g = _omega(3)
    sigma_diag = torch.rand(2, 3, 4, generator=g) + 0.1
    approx = transport_covariance(omega, sigma_diag)
    ref = torch.einsum("bijkl,bijkl,bjl->bijk", omega, omega, sigma_diag)
    assert torch.allclose(approx, ref, atol=1e-6)


def test_transported_kl_is_gauge_consistent():
    from vfe3.divergence import kl
    grp = get_group("so_k")(K=4)
    g = torch.Generator().manual_seed(9)
    phi = 0.3 * torch.randn(2, 3, grp.generators.shape[0], generator=g)
    omega = compute_transport_operators(phi, grp, gauge_mode="learned")["Omega"]

    mu_q = torch.randn(2, 3, 4, generator=g)
    mu_k = torch.randn(2, 3, 4, generator=g)
    Aq = torch.randn(2, 3, 4, 4, generator=g)
    Ak = torch.randn(2, 3, 4, 4, generator=g)
    S_q = Aq @ Aq.transpose(-1, -2) + torch.eye(4)
    S_k = Ak @ Ak.transpose(-1, -2) + torch.eye(4)

    mu_kt = transport_mean(omega, mu_k)
    S_kt = transport_covariance(omega, S_k)
    mu_qb = mu_q.unsqueeze(2).expand(2, 3, 3, 4)
    S_qb = S_q.unsqueeze(2).expand(2, 3, 3, 4, 4)
    base = kl(mu_qb, S_qb, mu_kt, S_kt, family="gaussian_full")

    coeff = 0.25 * torch.randn(grp.generators.shape[0], generator=g)
    h = torch.linalg.matrix_exp(torch.einsum("a,aij->ij", coeff, grp.generators))
    mu_qb2 = torch.einsum("kl,bijl->bijk", h, mu_qb)
    mu_kt2 = torch.einsum("kl,bijl->bijk", h, mu_kt)
    S_qb2 = torch.einsum("kl,bijlm,nm->bijkn", h, S_qb, h)
    S_kt2 = torch.einsum("kl,bijlm,nm->bijkn", h, S_kt, h)
    moved = kl(mu_qb2, S_qb2, mu_kt2, S_kt2, family="gaussian_full")
    assert torch.allclose(base, moved, atol=1e-3, rtol=1e-3)


def test_direct_omega_represents_reflection():
    from vfe3.geometry.transport import compute_transport_operators_direct
    refl = torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0]))
    omega = refl.expand(1, 2, 4, 4).contiguous()
    out = compute_transport_operators_direct(omega, gauge_mode="learned")
    assert torch.det(out["omega_i"][0, 0]) < 0
