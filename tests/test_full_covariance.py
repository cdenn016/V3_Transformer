r"""Full-covariance (gaussian_full) pure path: end-to-end runnability + golden equivalence.

GL(K) audit finding #2: the GL(K)-invariant covariance sandwich Omega Sigma Omega^T must be
runnable end-to-end through VFEModel / the E-step under appropriate toggles, not only as
isolated kernels. The toggles are family='gaussian_full' + diagonal_covariance=False +
decode_mode='full' (the three are cross-validated for consistency).
"""

import pytest
import torch

from vfe3.belief import BeliefState
from vfe3.config import VFE3Config
from vfe3.geometry.groups import get_group
from vfe3.inference.e_step import e_step_iteration
from vfe3.model.model import VFEModel


def test_full_covariance_config_requires_consistent_flags():
    """diagonal_covariance is a live field cross-validated against family (kept, not collapsed)."""
    VFE3Config(family="gaussian_diagonal", diagonal_covariance=True)
    VFE3Config(family="gaussian_full", diagonal_covariance=False, decode_mode="full")
    with pytest.raises(ValueError):
        VFE3Config(family="gaussian_full", diagonal_covariance=True)
    with pytest.raises(ValueError):
        VFE3Config(family="gaussian_diagonal", diagonal_covariance=False)


def test_full_covariance_model_runs_end_to_end():
    """The full-covariance pure path runs encode -> E-step -> full decode -> CE, forward+backward."""
    cfg = VFE3Config(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=1,
                     n_e_steps=2, e_mu_lr=0.05, e_sigma_lr=0.01, e_phi_lr=0.0,
                     family="gaussian_full", diagonal_covariance=False, decode_mode="full")
    model = VFEModel(cfg)
    tokens = torch.randint(0, 20, (2, 5)); targets = torch.randint(0, 20, (2, 5))
    beliefs = model.prior_bank.encode(tokens)
    assert beliefs.sigma.shape == (2, 5, 4, 4)              # full SPD covariance encode
    logits, loss, _ = model(tokens, targets)
    assert logits.shape == (2, 5, 20) and torch.isfinite(loss)
    loss.backward()
    assert model.prior_bank.mu_embed.grad is not None and model.prior_bank.mu_embed.grad.abs().sum() > 0


def _spd(N, K, gen):
    r"""A batch of genuinely non-diagonal SPD matrices A Aᵀ + K I."""
    A = torch.randn(N, K, K, generator=gen)
    return A @ A.transpose(-1, -2) + K * torch.eye(K)


def test_full_covariance_e_step_keeps_sigma_spd_and_symmetric():
    """One full-covariance E-step iteration on a NON-diagonal SPD belief stays SPD + symmetric
    (the affine-invariant retract_spd_full, not the elementwise diagonal retraction)."""
    grp = get_group("glk")(3)
    g = torch.Generator().manual_seed(0)
    N, K = 4, 3
    b = BeliefState(
        mu=torch.randn(N, K, generator=g),
        sigma=_spd(N, K, g),
        phi=0.1 * torch.randn(N, grp.generators.shape[0], generator=g),
    )
    mu_p = torch.randn(N, K, generator=g)
    sigma_p = _spd(N, K, g)
    out = e_step_iteration(b, mu_p, sigma_p, grp, tau=1.5, e_mu_lr=0.05, e_sigma_lr=0.05,
                           e_phi_lr=0.0, family="gaussian_full")
    assert out.sigma.shape == (N, K, K)
    assert torch.allclose(out.sigma, out.sigma.transpose(-1, -2), atol=1e-5)   # symmetric
    assert (torch.linalg.eigvalsh(out.sigma) > 0).all()                        # stays SPD


def test_full_covariance_reduces_to_diagonal_at_identity_transport():
    """Golden gate: with Omega=I (phi=0) and a diagonal-initialised covariance, the full-cov
    E-step's diagonal matches the diagonal-cov E-step (full generalises, not replaces, diagonal)."""
    grp = get_group("glk")(3)
    g = torch.Generator().manual_seed(1)
    N, K = 4, 3
    n_gen = grp.generators.shape[0]
    mu = torch.randn(N, K, generator=g)
    sigma_diag = torch.rand(N, K, generator=g) + 0.5
    phi = torch.zeros(N, n_gen)                             # Omega = I
    mu_p = torch.randn(N, K, generator=g)
    sigma_p_diag = torch.rand(N, K, generator=g) + 0.5

    out_diag = e_step_iteration(
        BeliefState(mu=mu, sigma=sigma_diag, phi=phi), mu_p, sigma_p_diag, grp,
        tau=1.5, e_mu_lr=0.05, e_sigma_lr=0.01, e_phi_lr=0.0, family="gaussian_diagonal",
    )
    out_full = e_step_iteration(
        BeliefState(mu=mu, sigma=torch.diag_embed(sigma_diag), phi=phi),
        mu_p, torch.diag_embed(sigma_p_diag), grp,
        tau=1.5, e_mu_lr=0.05, e_sigma_lr=0.01, e_phi_lr=0.0, family="gaussian_full",
    )
    assert torch.allclose(out_full.mu, out_diag.mu, atol=1e-4)
    diag_of_full = torch.diagonal(out_full.sigma, dim1=-2, dim2=-1)
    assert torch.allclose(diag_of_full, out_diag.sigma, atol=1e-3)
    assert (out_full.sigma - torch.diag_embed(diag_of_full)).abs().max() < 1e-4
