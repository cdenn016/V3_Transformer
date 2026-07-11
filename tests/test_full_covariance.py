r"""Full-covariance (gaussian_full) pure path: end-to-end runnability + golden equivalence.

GL(K) audit finding #2: the GL(K)-invariant covariance sandwich Omega Sigma Omega^T must be
runnable end-to-end through VFEModel / the E-step under appropriate toggles, not only as
isolated kernels. The toggles are family='gaussian_full' + decode_mode='full'
(diagonal_covariance is a derived read-only property of family).
"""

import pytest
import torch

from vfe3.belief import BeliefState
from vfe3.config import VFE3Config
from vfe3.geometry.groups import get_group
from vfe3.inference.e_step import e_step_iteration
from vfe3.model.model import VFEModel


def test_full_covariance_config_derives_diagonal_covariance_flag():
    """diagonal_covariance is a derived read-only property of family (single source of truth)."""
    assert VFE3Config(family="gaussian_diagonal").diagonal_covariance is True
    assert VFE3Config(family="gaussian_full", decode_mode="full").diagonal_covariance is False
    with pytest.raises(TypeError):
        VFE3Config(family="gaussian_full", diagonal_covariance=True)    # no longer a settable field


def test_full_covariance_model_runs_end_to_end():
    """The full-covariance pure path runs encode -> E-step -> full decode -> CE, forward+backward."""
    cfg = VFE3Config(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=1,
                     n_e_steps=2, e_q_mu_lr=0.05, e_q_sigma_lr=0.01, e_phi_lr=0.0,
                     family="gaussian_full", decode_mode="full", use_prior_bank=True)
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
    out = e_step_iteration(b, mu_p, sigma_p, grp, tau=1.5, e_q_mu_lr=0.05, e_q_sigma_lr=0.05,
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
        tau=1.5, e_q_mu_lr=0.05, e_q_sigma_lr=0.01, e_phi_lr=0.0, family="gaussian_diagonal",
    )
    out_full = e_step_iteration(
        BeliefState(mu=mu, sigma=torch.diag_embed(sigma_diag), phi=phi),
        mu_p, torch.diag_embed(sigma_p_diag), grp,
        tau=1.5, e_q_mu_lr=0.05, e_q_sigma_lr=0.01, e_phi_lr=0.0, family="gaussian_full",
    )
    assert torch.allclose(out_full.mu, out_diag.mu, atol=1e-4)
    diag_of_full = torch.diagonal(out_full.sigma, dim1=-2, dim2=-1)
    assert torch.allclose(diag_of_full, out_diag.sigma, atol=1e-3)
    assert (out_full.sigma - torch.diag_embed(diag_of_full)).abs().max() < 1e-4


def test_full_kl_survives_non_pd_covariance():
    # The alpha=1 full-covariance KL must CLAMP (not raise) on a numerically non-PD covariance.
    # Such a prior covariance can arise after training shifts it, and full-cov configs route the
    # belief KL through this closed form via the E-step oracle (e.g. decode_mode='full'); a raw
    # torch.linalg.cholesky there raises and kills the run. The alpha != 1 branch was already
    # hardened with safe_cholesky; this pins the same robustness for alpha = 1.
    from vfe3.families.gaussian import FullGaussian
    K = 4
    mu = torch.zeros(2, K)
    sigma_q = torch.eye(K).expand(2, K, K).contiguous()
    bad = torch.eye(K).clone(); bad[0, 0] = -1.0                 # a negative eigenvalue -> not PD
    sigma_p = bad.expand(2, K, K).contiguous()
    kl = FullGaussian(mu, sigma_q).renyi_closed_form(            # must NOT raise
        FullGaussian(mu, sigma_p), alpha=1.0, kl_max=100.0, eps=1e-6)
    assert torch.isfinite(kl).all()
    assert (kl <= 100.0 + 1e-3).all()


def test_full_renyi_alpha_gt1_nonpd_blend_clamps_to_kl_max():
    # alpha>1 leaves the convex regime: the blend (1-alpha)Sigma_q + alpha*Sigma_t can be indefinite,
    # making the Renyi divergence undefined -> it must clamp to kl_max. A jitter-rescued Cholesky on
    # the blend would silently report it PD and (with the fp64 logdet dropping the sign) collapse the
    # divergence to ~0. The mask must gate on the blend's eigenvalue SIGN. (audit 2026-06-17 id 38.)
    from vfe3.families.gaussian import FullGaussian
    K = 4
    q = FullGaussian(torch.zeros(K), torch.eye(K))
    t = FullGaussian(torch.zeros(K), 1e-4 * torch.eye(K))         # blend ~ -0.0049 I (neg-definite)
    div = q.renyi_closed_form(t, alpha=1.005, kl_max=100.0, eps=1e-6)
    assert torch.isfinite(div).all()
    assert div.item() > 50.0                                       # ~kl_max, NOT the spurious ~0


def test_full_entropy_survives_non_pd_covariance():
    # FullGaussian.entropy must use the same safe (jittered, never-raising) Cholesky as the full-cov
    # KL: a raw torch.linalg.cholesky on a numerically non-PD Sigma raises and kills a diagnostics
    # / entropy call. Mirrors test_full_kl_survives_non_pd_covariance for the entropy path.
    from vfe3.families.gaussian import FullGaussian
    K = 4
    mu = torch.zeros(2, K)
    bad = torch.eye(K).clone(); bad[0, 0] = -1.0                 # negative eigenvalue -> not PD
    h = FullGaussian(mu, bad.expand(2, K, K).contiguous()).entropy()   # must NOT raise
    assert torch.isfinite(h).all()
