"""Audit Group 5: the full-covariance (gaussian_full) pure path, end to end.

Restores CLAUDE.md's "a theoretically pure path must always exist under toggles" for the
covariance sandwich: full-Sigma encode, full SPD retraction in the E-step, and the
Cholesky full-covariance decode. The equivalence test pins that the full path REDUCES to
the diagonal path when the transport is trivial (Omega = I) and the covariances are
diagonal-initialized -- the golden gate the implementation was written against.
"""

import torch

from vfe3.belief import BeliefState
from vfe3.config import VFE3Config
from vfe3.geometry.groups import get_group
from vfe3.inference.e_step import e_step_iteration
from vfe3.model.model import VFEModel


def test_full_cov_reduces_to_diagonal_when_omega_identity():
    # With phi = 0 (Omega = I) and a diagonal-initialized covariance, ONE full-mode E-step
    # iteration must reproduce the diagonal-mode mean exactly and the diagonal-mode variances
    # on its diagonal, with the off-diagonal covariance staying ~0.
    torch.manual_seed(0)
    N, K = 4, 3
    grp = get_group("glk")(K)
    n_gen = grp.generators.shape[0]
    mu = torch.randn(N, K)
    sig = torch.rand(N, K) + 0.5
    phi = torch.zeros(N, n_gen)                       # Omega = I
    mu_p = torch.randn(N, K)
    sig_p = torch.rand(N, K) + 0.5

    out_d = e_step_iteration(
        BeliefState(mu.clone(), sig.clone(), phi.clone()), mu_p, sig_p, grp,
        tau=1.0, e_mu_lr=0.1, e_sigma_lr=0.1, e_phi_lr=0.0, family="gaussian_diagonal",
    )
    out_f = e_step_iteration(
        BeliefState(mu.clone(), torch.diag_embed(sig.clone()), phi.clone()),
        mu_p, torch.diag_embed(sig_p), grp,
        tau=1.0, e_mu_lr=0.1, e_sigma_lr=0.1, e_phi_lr=0.0, family="gaussian_full",
    )

    assert out_f.sigma.shape == (N, K, K)
    assert torch.allclose(out_d.mu, out_f.mu, atol=1e-4)
    diag_f = torch.diagonal(out_f.sigma, dim1=-2, dim2=-1)
    assert torch.allclose(out_d.sigma, diag_f, atol=1e-3)
    off = out_f.sigma - torch.diag_embed(diag_f)
    assert off.abs().max() < 1e-4


def test_full_cov_retraction_keeps_sigma_spd():
    torch.manual_seed(1)
    N, K = 3, 3
    grp = get_group("glk")(K)
    n_gen = grp.generators.shape[0]
    b = BeliefState(
        mu=torch.randn(N, K),
        sigma=torch.diag_embed(torch.rand(N, K) + 0.5),
        phi=0.05 * torch.randn(N, n_gen),
    )
    out = e_step_iteration(
        b, torch.randn(N, K), torch.diag_embed(torch.rand(N, K) + 0.5), grp,
        tau=1.0, e_mu_lr=0.1, e_sigma_lr=0.1, e_phi_lr=0.05, family="gaussian_full",
    )
    eig = torch.linalg.eigvalsh(out.sigma)
    assert (eig > 0).all()                            # stays on the SPD cone
    assert torch.allclose(out.sigma, out.sigma.transpose(-1, -2), atol=1e-5)


def test_full_cov_model_forward_end_to_end():
    # The full pure path is reachable through config (encode full Sigma -> full E-step ->
    # full Cholesky decode) and produces finite logits/loss -- the CLAUDE.md pure-path mandate.
    torch.manual_seed(0)
    cfg = VFE3Config(
        vocab_size=12, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=1, n_e_steps=1,
        e_phi_lr=0.0,
        family="gaussian_full", divergence_family="gaussian_full",
        diagonal_covariance=False, decode_mode="full",
    )
    model = VFEModel(cfg)
    tokens = torch.randint(0, 12, (2, 5))
    targets = torch.randint(0, 12, (2, 5))
    logits, loss, ce = model(tokens, targets)
    assert logits.shape == (2, 5, 12)
    assert torch.isfinite(loss) and torch.isfinite(ce)


def test_full_cov_encode_emits_full_covariance():
    cfg = VFE3Config(
        vocab_size=8, embed_dim=4, n_heads=2, max_seq_len=4,
        family="gaussian_full", divergence_family="gaussian_full",
        diagonal_covariance=False, decode_mode="full",
    )
    model = VFEModel(cfg)
    beliefs = model.prior_bank.encode(torch.randint(0, 8, (2, 4)))
    assert beliefs.sigma.shape == (2, 4, 4, 4)        # (B, N, K, K) full covariance
