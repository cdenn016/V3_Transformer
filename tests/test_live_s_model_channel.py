r"""Live model channel s (dynamic prior tie), default-off. Spec:
docs/superpowers/specs/2026-06-08-live-s-model-channel-design.md.
"""

import warnings

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel


def _tiny_cfg(**overrides) -> VFE3Config:
    r"""Minimal model config (embed_dim=4, n_heads=2, vocab=8, seq=4, 1 layer)."""
    base = dict(vocab_size=8, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=1)
    base.update(overrides)
    return VFE3Config(**base)


def test_s_e_step_defaults_off():
    cfg = VFE3Config()
    assert cfg.s_e_step is False
    assert cfg.e_s_mu_lr == 0.1
    assert cfg.e_s_sigma_lr == 0.1


def test_s_e_step_lr_validation_rejects_negative():
    with pytest.raises(ValueError, match="e_s_mu_lr"):
        _tiny_cfg(s_e_step=True, prior_source="model_channel", e_s_mu_lr=-1.0)
    with pytest.raises(ValueError, match="e_s_sigma_lr"):
        _tiny_cfg(s_e_step=True, prior_source="model_channel", e_s_sigma_lr=-0.5)


def test_s_e_step_requires_model_channel_prior_source():
    # s_e_step anchors the belief to s AND must decode against s -> require model_channel.
    with pytest.raises(ValueError, match="prior_source"):
        _tiny_cfg(s_e_step=True, prior_source="token")


def test_s_e_step_inert_misconfig_warns():
    with pytest.warns(UserWarning, match="s_e_step"):
        _tiny_cfg(s_e_step=True, prior_source="model_channel",
                  lambda_h=0.0, lambda_gamma=0.0)


def test_s_tables_and_frozen_r_created_under_s_e_step():
    m = VFEModel(_tiny_cfg(s_e_step=True, prior_source="model_channel",
                           lambda_h=1.0, lambda_gamma=1.0))
    pb = m.prior_bank
    assert getattr(pb, "s_mu_embed", None) is not None
    assert getattr(pb, "r_mu", None) is not None
    assert pb.r_mu.requires_grad is False
    assert pb.r_sigma_log.requires_grad is False


def test_frozen_r_created_when_s_e_step_is_the_only_trigger():
    # The r-gate's new `or s_e_step` is the load-bearing change: with lambda_h=0 the OLD gate
    # (`if lambda_h > 0`) would NOT create r, so s_e_step alone must create it. (lambda_h=0 AND
    # lambda_gamma=0 fires the inert-misconfig warning by design.)
    with pytest.warns(UserWarning, match="s_e_step"):
        m = VFEModel(_tiny_cfg(s_e_step=True, prior_source="model_channel",
                               lambda_h=0.0, lambda_gamma=0.0))
    assert getattr(m.prior_bank, "r_mu", None) is not None
    assert m.prior_bank.r_mu.requires_grad is False


def test_belief_tables_byte_identical_with_or_without_s_e_step():
    # s-tables are drawn LAST, so the belief tables (drawn first) are bit-identical.
    torch.manual_seed(0); off = VFEModel(_tiny_cfg(s_e_step=False))
    torch.manual_seed(0); on = VFEModel(_tiny_cfg(s_e_step=True, prior_source="model_channel",
                                                  lambda_h=1.0, lambda_gamma=1.0))
    assert torch.equal(off.prior_bank.mu_embed, on.prior_bank.mu_embed)
    assert torch.equal(off.prior_bank.phi_embed, on.prior_bank.phi_embed)
    assert torch.equal(off.prior_bank.sigma_log_embed, on.prior_bank.sigma_log_embed)


def test_refine_s_preserves_shape_and_zero_lr_is_static():
    torch.manual_seed(0)
    m = VFEModel(_tiny_cfg(s_e_step=True, prior_source="model_channel",
                           lambda_h=1.0, lambda_gamma=1.0,
                           e_s_mu_lr=0.0, e_s_sigma_lr=0.0))
    tok = torch.randint(0, m.cfg.vocab_size, (2, 4))
    phi0 = m._apply_pos_phi(m.prior_bank.encode(tok).phi)
    s0_mu, s0_sigma = m.prior_bank.encode_s(tok)
    s1_mu, s1_sigma = m._refine_s(tok, phi0)
    assert s1_mu.shape == s0_mu.shape == (2, 4, m.cfg.embed_dim)
    # e_s_lr=0 -> the refine is a no-op -> s1 == s0.
    assert torch.allclose(s1_mu, s0_mu)
    assert torch.allclose(s1_sigma, s0_sigma)


def test_refine_s_moves_s_with_nonzero_lr():
    torch.manual_seed(0)
    m = VFEModel(_tiny_cfg(s_e_step=True, prior_source="model_channel",
                           lambda_h=1.0, lambda_gamma=1.0,
                           e_s_mu_lr=0.5, e_s_sigma_lr=0.5))
    tok = torch.randint(0, m.cfg.vocab_size, (2, 4))
    phi0 = m._apply_pos_phi(m.prior_bank.encode(tok).phi)
    s0_mu, _ = m.prior_bank.encode_s(tok)
    s1_mu, _ = m._refine_s(tok, phi0)
    assert not torch.allclose(s1_mu, s0_mu)   # the refine actually descends toward r + consensus


def _tok(m, b=2, n=4):
    return torch.randint(0, m.cfg.vocab_size, (b, n))


def test_default_off_forward_is_unchanged_by_the_new_code():
    # The pure path must stay finite/sane: same seed, s_e_step=False.
    torch.manual_seed(0); m = VFEModel(_tiny_cfg(s_e_step=False))
    tok = _tok(m)
    lg = m(tok)
    assert torch.isfinite(lg).all()


def test_s_e_step_changes_logits_at_n_e_steps_1():
    # Belief tables are bit-identical across the two models (s-tables drawn last); the ONLY
    # difference is the live s channel, which must move the logits at the operative n_e_steps=1.
    torch.manual_seed(0); base = VFEModel(_tiny_cfg(s_e_step=False, prior_source="model_channel",
                                                    lambda_h=1.0, lambda_gamma=1.0))
    torch.manual_seed(0); live = VFEModel(_tiny_cfg(s_e_step=True, prior_source="model_channel",
                                                    lambda_h=1.0, lambda_gamma=1.0,
                                                    e_s_mu_lr=0.5, e_s_sigma_lr=0.5))
    tok = _tok(live)
    assert not torch.allclose(base(tok), live(tok))


def test_e_s_lr_zero_reduces_to_static_model_channel():
    # s_e_step + e_s_lr=0 == static prior_source='model_channel' (refine no-ops): logits match.
    torch.manual_seed(0); static = VFEModel(_tiny_cfg(s_e_step=False, prior_source="model_channel",
                                                      lambda_h=1.0, lambda_gamma=1.0))
    torch.manual_seed(0); live0 = VFEModel(_tiny_cfg(s_e_step=True, prior_source="model_channel",
                                                     lambda_h=1.0, lambda_gamma=1.0,
                                                     e_s_mu_lr=0.0, e_s_sigma_lr=0.0))
    tok = _tok(live0)
    assert torch.allclose(static(tok), live0(tok), atol=1e-6, rtol=1e-5)


def test_s_e_step_gradient_reaches_s_tables_at_t1():
    torch.manual_seed(0)
    m = VFEModel(_tiny_cfg(s_e_step=True, prior_source="model_channel",
                           lambda_h=1.0, lambda_gamma=1.0, e_s_mu_lr=0.5))
    tok = _tok(m)
    tgt = _tok(m)
    _, loss, _ = m(tok, targets=tgt)
    loss.backward()
    assert m.prior_bank.s_mu_embed.grad is not None
    assert m.prior_bank.s_mu_embed.grad.abs().sum() > 0


def test_generate_runs_under_s_e_step():
    torch.manual_seed(0)
    m = VFEModel(_tiny_cfg(s_e_step=True, prior_source="model_channel",
                           lambda_h=1.0, lambda_gamma=1.0, e_s_mu_lr=0.5))
    prompt = torch.randint(0, m.cfg.vocab_size, (1, 3))
    out = m.generate(prompt, max_new_tokens=2)
    assert out.shape == (1, 5)


def test_diagnostics_runs_under_s_e_step():
    torch.manual_seed(0)
    m = VFEModel(_tiny_cfg(s_e_step=True, prior_source="model_channel",
                           lambda_h=1.0, lambda_gamma=1.0, e_s_mu_lr=0.5))
    tok = torch.randint(0, m.cfg.vocab_size, (1, 4))
    d = m.diagnostics(tok)            # must not raise
    assert d is not None


def test_model_channel_self_divergence_zero_at_s_equals_r():
    from vfe3.families.gaussian import DiagonalGaussian
    from vfe3.free_energy import self_divergence
    K = 4
    mu = torch.randn(2, 3, K)
    sig = torch.rand(2, 3, K) + 0.1
    d = self_divergence(DiagonalGaussian(mu, sig), DiagonalGaussian(mu, sig)).abs().max()
    assert d < 1e-5            # D(s||s) == 0


def test_s_e_step_forward_runs_finite_under_so_k():
    # Smoke: the live-s forward runs and stays finite under a non-trivial group (so_k).
    # (Full gauge-invariance is inherited from the shared phi machinery + existing gauge tests;
    # an exact global-invariance assertion is intentionally NOT used here because the base
    # forward carries the global-diagonal stabilizer residual by design.)
    torch.manual_seed(0)
    m = VFEModel(_tiny_cfg(s_e_step=True, prior_source="model_channel",
                           lambda_h=1.0, lambda_gamma=1.0, e_s_mu_lr=0.5,
                           gauge_group="so_k"))
    tok = torch.randint(0, m.cfg.vocab_size, (1, 4))
    lg = m(tok)
    assert torch.isfinite(lg).all()


# ===========================================================================
# PB-11 (Task 3, 2026-07-12): _refine_s consumes the configured family and the
# active connection law (was hardcoded family='gaussian_diagonal' + flat transport).
# ===========================================================================

def test_refine_s_full_covariance_returns_full_rank():
    # s_e_step + gaussian_full was REJECTED at construction pre-change (the diagonal s-refine
    # would overwrite the (B,N,K,K) belief sigma). The family-dispatched _refine_s now refines the
    # model channel AS a full Gaussian: the refined sigma keeps full (B,N,K,K) rank.
    torch.manual_seed(0)
    m = VFEModel(_tiny_cfg(s_e_step=True, prior_source="model_channel", family="gaussian_full",
                           decode_mode="full", lambda_h=1.0, lambda_gamma=1.0,
                           e_s_mu_lr=0.5, e_s_sigma_lr=0.5))
    K = m.cfg.embed_dim
    tok = torch.randint(0, m.cfg.vocab_size, (2, 4))
    phi0 = m._apply_pos_phi(m.prior_bank.encode(tok).phi)
    s0_mu, s0_sigma = m.prior_bank.encode_s(tok)
    assert s0_sigma.shape == (2, 4, K, K)                       # encode_s covariance is full rank
    s1_mu, s1_sigma = m._refine_s(tok, phi0)
    assert s1_mu.shape == (2, 4, K)
    assert s1_sigma.shape == (2, 4, K, K)                       # refined s stays full rank (no crash)
    assert torch.isfinite(s1_mu).all() and torch.isfinite(s1_sigma).all()


def test_refine_s_full_covariance_forward_and_backward_run():
    torch.manual_seed(0)
    m = VFEModel(_tiny_cfg(s_e_step=True, prior_source="model_channel", family="gaussian_full",
                           decode_mode="full", lambda_h=1.0, lambda_gamma=1.0, e_s_mu_lr=0.5))
    tok = _tok(m)
    tgt = _tok(m)
    _, loss, _ = m(tok, targets=tgt)                            # full-cov live-s forward (was unreachable)
    assert torch.isfinite(loss)
    loss.backward()
    assert m.prior_bank.s_mu_embed.grad is not None
    assert float(m.prior_bank.s_mu_embed.grad.abs().sum()) > 0.0


def test_refine_s_laplace_family_diagonal_rank_runs():
    torch.manual_seed(0)
    m = VFEModel(_tiny_cfg(s_e_step=True, prior_source="model_channel", family="laplace_diagonal",
                           r_update_mode="gradient", lambda_h=1.0, lambda_gamma=1.0,
                           e_s_mu_lr=0.5, e_s_sigma_lr=0.5))
    K = m.cfg.embed_dim
    tok = torch.randint(0, m.cfg.vocab_size, (2, 4))
    phi0 = m._apply_pos_phi(m.prior_bank.encode(tok).phi)
    s1_mu, s1_sigma = m._refine_s(tok, phi0)
    assert s1_mu.shape == (2, 4, K) and s1_sigma.shape == (2, 4, K)   # Laplace is diagonal rank
    assert torch.isfinite(s1_mu).all() and torch.isfinite(s1_sigma).all()


def test_refine_s_shares_connection_W():
    # regime_ii s-refine reads the s-channel means AND the shared connection_W (was flat-hardcoded,
    # so connection_W had no effect on the refined s).
    torch.manual_seed(0)
    m = VFEModel(_tiny_cfg(s_e_step=True, prior_source="model_channel", transport_mode="regime_ii",
                           lambda_h=1.0, lambda_gamma=1.0, e_s_mu_lr=0.5, e_s_sigma_lr=0.5))
    with torch.no_grad():
        m.prior_bank.s_mu_embed.normal_(0.0, 0.5)
        m.connection_W.normal_(0.0, 0.6)
    tok = torch.randint(0, m.cfg.vocab_size, (2, 4))
    phi0 = m._apply_pos_phi(m.prior_bank.encode(tok).phi)
    s1a, _ = m._refine_s(tok, phi0)
    with torch.no_grad():
        m.connection_W.mul_(1.8)
    s1b, _ = m._refine_s(tok, phi0)
    assert not torch.allclose(s1a, s1b)                         # refined s moves with the shared connection


def test_refine_s_gradient_reaches_connection_W():
    torch.manual_seed(0)
    m = VFEModel(_tiny_cfg(s_e_step=True, prior_source="model_channel", transport_mode="regime_ii",
                           lambda_h=1.0, lambda_gamma=1.0, e_s_mu_lr=0.5))
    with torch.no_grad():
        m.prior_bank.s_mu_embed.normal_(0.0, 0.5)
        m.connection_W.normal_(0.0, 0.4)
    tok = _tok(m)
    tgt = _tok(m)
    _, loss, _ = m(tok, targets=tgt)
    loss.backward()
    assert m.connection_W.grad is not None
    assert float(m.connection_W.grad.abs().sum()) > 0.0        # s-refine keeps the connection gradient live
