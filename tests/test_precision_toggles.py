r"""A default-OFF diagnostic toggle probing whether the belief covariance Sigma carries
predictive signal, while keeping the discriminative linear decode (no NN added beyond the existing
linear-decode exception; the pure path preserved when off).

(1) precision_weighted_attention: fold a detached per-key reliability bias -log(b0 + tr Sigma_j)
    into the attention log_prior, so attention down-weights high-variance (unreliable) keys before
    the softmax. A uniform-over-keys Sigma is softmax-absorbed (no effect); only key-to-key variance
    in Sigma changes attention. Detached -> the closed-form belief kernel stays exact.

Defaults OFF and byte-identical to the current path when off.
"""

import warnings

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel


def _model(seed: int = 0, **over) -> VFEModel:
    cfg = VFE3Config(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=1,
                     n_e_steps=1, e_q_mu_lr=0.5, e_phi_lr=0.0, mass_phi=0.0,
                     mstep_self_coupling_weight=0.0, seed=seed, **over)
    torch.manual_seed(seed)
    return VFEModel(cfg)


# =========================================================================== feature 2: attention

def test_precision_weighted_attention_defaults_off():
    assert VFE3Config().precision_weighted_attention is False
    assert VFE3Config().precision_attention_b0 == 1.0


def test_precision_key_bias_monotonic_in_sigma():
    # The per-key reliability bias -log(b0 + tr Sigma_j) is strictly decreasing in the key's
    # total variance: a more-uncertain key gets a MORE NEGATIVE bias (down-weighted).
    from vfe3.model.model import _precision_key_bias
    sigma_lo = torch.full((1, 3, 4), 0.1)
    sigma_hi = torch.full((1, 3, 4), 2.0)
    b_lo = _precision_key_bias(sigma_lo, b0=1.0)
    b_hi = _precision_key_bias(sigma_hi, b0=1.0)
    assert (b_hi < b_lo).all()
    assert torch.allclose(b_lo, -torch.log(1.0 + sigma_lo.sum(-1)))


def test_precision_weighted_attention_off_byte_identical():
    # OFF == the baseline model (no toggle) bit-for-bit on the forward.
    m_off = _model(seed=0, precision_weighted_attention=False)
    m_base = _model(seed=0)
    tokens = torch.randint(0, 20, (3, 5)); targets = torch.randint(0, 20, (3, 5))
    _, l_off, _ = m_off(tokens, targets)
    _, l_base, _ = m_base(tokens, targets)
    assert torch.equal(l_off, l_base)


def test_precision_weighted_attention_no_effect_when_sigma_uniform():
    # At init all vocab share sigma_init -> every key has equal tr Sigma_j -> the per-key bias is
    # constant over keys -> softmax-absorbed -> ON == OFF. (s_e_step=False so the block sees the
    # encoded, still-uniform sigma.)
    m_on = _model(seed=0, precision_weighted_attention=True)
    m_off = _model(seed=0, precision_weighted_attention=False)
    tokens = torch.randint(0, 20, (3, 5)); targets = torch.randint(0, 20, (3, 5))
    _, l_on, _ = m_on(tokens, targets)
    _, l_off, _ = m_off(tokens, targets)
    assert torch.allclose(l_on, l_off, atol=1e-6)


def test_precision_weighted_attention_changes_forward_when_sigma_varies():
    # Make per-vocab sigma non-uniform -> key reliability varies -> the bias is non-constant over
    # keys -> attention (and the forward logits) change when the toggle is on.
    m_on = _model(seed=0, precision_weighted_attention=True)
    m_off = _model(seed=0, precision_weighted_attention=False)
    with torch.no_grad():
        for m in (m_on, m_off):
            m.prior_bank.sigma_log_embed.copy_(torch.randn_like(m.prior_bank.sigma_log_embed))
    tokens = torch.randint(0, 20, (3, 5)); targets = torch.randint(0, 20, (3, 5))
    _, l_on, _ = m_on(tokens, targets)
    _, l_off, _ = m_off(tokens, targets)
    assert not torch.allclose(l_on, l_off, atol=1e-4)


def test_precision_weighted_attention_b0_must_be_positive():
    with pytest.raises(ValueError, match="precision_attention_b0"):
        VFE3Config(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5,
                   precision_weighted_attention=True, precision_attention_b0=0.0)


def test_precision_weighted_attention_grad_flows():
    m = _model(seed=0, precision_weighted_attention=True)
    with torch.no_grad():
        m.prior_bank.sigma_log_embed.copy_(torch.randn_like(m.prior_bank.sigma_log_embed))
    tokens = torch.randint(0, 20, (3, 5)); targets = torch.randint(0, 20, (3, 5))
    _, loss, _ = m(tokens, targets)
    loss.backward()
    g = m.prior_bank.mu_embed.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


# ================================================== feature 2b: per-head reliability

def test_precision_attention_per_head_defaults_off():
    assert VFE3Config().precision_attention_per_head is False


def test_precision_key_bias_per_head_splits_into_blocks():
    # With irrep_dims the reliability is computed PER gauge block (head): trace over that block's
    # coords only, shape (B, N, H) instead of the global (B, N).
    from vfe3.model.model import _precision_key_bias
    sigma = torch.tensor([[[0.1, 0.1, 0.5, 0.5],          # token 0: block0 tr=0.2, block1 tr=1.0
                           [0.2, 0.2, 0.2, 0.2]]])         # token 1: both blocks tr=0.4   -> (1,2,4)
    b = _precision_key_bias(sigma, b0=1.0, irrep_dims=[2, 2])
    assert b.shape == (1, 2, 2)
    expect = -torch.log(1.0 + torch.tensor([[[0.2, 1.0], [0.4, 0.4]]]))
    assert torch.allclose(b, expect, atol=1e-6)


def test_precision_attention_per_head_no_effect_at_uniform_sigma():
    # At init every coord shares sigma_init -> every per-head block trace is equal across heads and
    # keys -> the bias is constant -> softmax-absorbed -> per-head ON == baseline (no pwa).
    m_ph = _model(seed=0, precision_weighted_attention=True, precision_attention_per_head=True)
    m_base = _model(seed=0)
    tokens = torch.randint(0, 20, (3, 5)); targets = torch.randint(0, 20, (3, 5))
    _, l_ph, _ = m_ph(tokens, targets)
    _, l_base, _ = m_base(tokens, targets)
    assert torch.allclose(l_ph, l_base, atol=1e-6)


def test_precision_attention_per_head_differs_from_global_when_coords_vary():
    # When sigma varies ACROSS coordinates, the per-head block traces differ across heads, so the
    # per-head bias is NOT the global bias -> the two produce different attention / forward logits.
    m_ph = _model(seed=0, precision_weighted_attention=True, precision_attention_per_head=True)
    m_gl = _model(seed=0, precision_weighted_attention=True, precision_attention_per_head=False)
    with torch.no_grad():
        for m in (m_ph, m_gl):
            m.prior_bank.sigma_log_embed.copy_(torch.randn_like(m.prior_bank.sigma_log_embed))
    tokens = torch.randint(0, 20, (3, 5)); targets = torch.randint(0, 20, (3, 5))
    _, l_ph, _ = m_ph(tokens, targets)
    _, l_gl, _ = m_gl(tokens, targets)
    assert not torch.allclose(l_ph, l_gl, atol=1e-4)


def test_precision_attention_per_head_inert_warning_without_pwa():
    with pytest.warns(UserWarning, match="precision_attention_per_head"):
        VFE3Config(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5,
                   precision_weighted_attention=False, precision_attention_per_head=True)


def test_precision_attention_per_head_grad_flows():
    m = _model(seed=0, precision_weighted_attention=True, precision_attention_per_head=True)
    with torch.no_grad():
        m.prior_bank.sigma_log_embed.copy_(torch.randn_like(m.prior_bank.sigma_log_embed))
    tokens = torch.randint(0, 20, (3, 5)); targets = torch.randint(0, 20, (3, 5))
    _, loss, _ = m(tokens, targets)
    loss.backward()
    g = m.prior_bank.mu_embed.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


# ============================ feature 2c: single-block gauge groups (headless energy)
# A single-block group (glk/so_k/sp, or block_glk collapsed to n_heads==1) produces a HEADLESS
# (B,N,N) coupling energy -- no head axis. The precision bias must therefore be folded WITHOUT a
# head axis, or the forward broadcasts wrong and raises. (audit 2026-06-17 finding 45/46.)

def _single_block_model(seed: int = 0, **over) -> VFEModel:
    cfg = VFE3Config(vocab_size=20, embed_dim=4, n_heads=1, gauge_group="glk", max_seq_len=5,
                     n_layers=1, n_e_steps=1, e_q_mu_lr=0.5, e_phi_lr=0.0, mass_phi=0.0,
                     mstep_self_coupling_weight=0.0, seed=seed, **over)
    torch.manual_seed(seed)
    return VFEModel(cfg)


def test_precision_weighted_attention_single_block_forward_runs():
    # Global pwa on a single-block (glk) group: the energy is headless (B,N,N); the bias must fold
    # to (B,1,N), not (B,1,1,N). Previously this raised at the forward.
    m = _single_block_model(seed=0, precision_weighted_attention=True)
    assert len(m.group.irrep_dims) == 1                       # genuinely single-block
    with torch.no_grad():
        m.prior_bank.sigma_log_embed.copy_(torch.randn_like(m.prior_bank.sigma_log_embed))
    tokens = torch.randint(0, 20, (3, 5)); targets = torch.randint(0, 20, (3, 5))
    _, loss, _ = m(tokens, targets)
    assert torch.isfinite(loss).all()


def test_precision_attention_per_head_single_block_forward_runs():
    # Per-head pwa on a single-block group: one block == global, must also fold headless and run.
    m = _single_block_model(seed=0, precision_weighted_attention=True,
                            precision_attention_per_head=True)
    with torch.no_grad():
        m.prior_bank.sigma_log_embed.copy_(torch.randn_like(m.prior_bank.sigma_log_embed))
    tokens = torch.randint(0, 20, (3, 5)); targets = torch.randint(0, 20, (3, 5))
    _, loss, _ = m(tokens, targets)
    assert torch.isfinite(loss).all()


def test_precision_weighted_attention_single_block_changes_forward_when_sigma_varies():
    # And it is not inert: non-uniform key reliability moves the single-block forward too.
    m_on = _single_block_model(seed=0, precision_weighted_attention=True)
    m_off = _single_block_model(seed=0, precision_weighted_attention=False)
    with torch.no_grad():
        for m in (m_on, m_off):
            m.prior_bank.sigma_log_embed.copy_(torch.randn_like(m.prior_bank.sigma_log_embed))
    tokens = torch.randint(0, 20, (3, 5)); targets = torch.randint(0, 20, (3, 5))
    _, l_on, _ = m_on(tokens, targets)
    _, l_off, _ = m_off(tokens, targets)
    assert not torch.allclose(l_on, l_off, atol=1e-4)


# ============================ feature 2d: full-covariance family (sigma is (.., N, K, K))
# family='gaussian_full' carries the belief covariance as (.., N, K, K), not the diagonal
# (.., N, K). The per-key reliability is the TRACE tr Sigma_j (= sum of the diagonal variances),
# so the bias must reduce the covariance to its diagonal first; the old sigma.sum(-1) summed a
# covariance ROW, leaving a spurious K axis that broadcast-crashed the log_prior fold. This is the
# gauge_equivariance / cg_coupling sweep crash (full cov + block_glk + pwa, 2026-06-23).

def _fullcov_model(seed: int = 0, **over) -> VFEModel:
    # Mirrors the gauge_equivariance sweep arm: full covariance + multi-block block_glk + head
    # mixer + the KL-to-prior full-chunked decode, with precision-weighted attention ON. Tiny.
    cfg = VFE3Config(vocab_size=20, embed_dim=4, n_heads=2, gauge_group="block_glk",
                     max_seq_len=5, n_layers=1, n_e_steps=1, e_q_mu_lr=0.5, e_phi_lr=0.0,
                     mass_phi=0.0, mstep_self_coupling_weight=0.0,
                     family="gaussian_full", use_head_mixer=True, use_prior_bank=True,
                     decode_mode="full_chunked", phi_precond_mode="killing", s_e_step=False,
                     precision_weighted_attention=True, precision_attention_b0=2.0,
                     seed=seed, **over)
    torch.manual_seed(seed)
    return VFEModel(cfg)


def test_precision_weighted_attention_full_cov_forward_runs():
    # Regression: the multi-block (head-axis) global-bias branch must run with a full covariance.
    m = _fullcov_model(seed=0)
    assert len(m.group.irrep_dims) > 1                        # genuinely multi-block (crash branch)
    assert m.cfg.diagonal_covariance is False                 # full cov: sigma is (.., N, K, K)
    with torch.no_grad():
        m.prior_bank.sigma_log_embed.copy_(torch.randn_like(m.prior_bank.sigma_log_embed))
    tokens = torch.randint(0, 20, (3, 5)); targets = torch.randint(0, 20, (3, 5))
    _, loss, _ = m(tokens, targets)
    assert torch.isfinite(loss).all()


def test_precision_attention_per_head_full_cov_forward_runs():
    # Regression: the per-head branch must also run with a full covariance.
    m = _fullcov_model(seed=0, precision_attention_per_head=True)
    with torch.no_grad():
        m.prior_bank.sigma_log_embed.copy_(torch.randn_like(m.prior_bank.sigma_log_embed))
    tokens = torch.randint(0, 20, (3, 5)); targets = torch.randint(0, 20, (3, 5))
    _, loss, _ = m(tokens, targets)
    assert torch.isfinite(loss).all()


def test_precision_bias_full_cov_uses_matrix_trace():
    # The full-cov global bias must equal -log(b0 + tr Sigma_j): the TRACE (sum of diagonal
    # variances), invariant to off-diagonal covariance, with NO spurious K axis.
    m = _fullcov_model(seed=0)
    k = m.cfg.embed_dim
    torch.manual_seed(1)
    diag = torch.rand(2, 4, k) + 0.5                                       # (.., N, K) variances
    off  = 0.3 * torch.randn(2, 4, k, k)
    off  = off + off.transpose(-1, -2)                                     # symmetric ...
    off  = off - torch.diag_embed(off.diagonal(dim1=-2, dim2=-1))          # ... with zero diagonal
    full = torch.diag_embed(diag) + off                                    # diagonal == diag exactly
    b_full = m._fold_precision_bias(None, full)                            # (.., 1, 1, N)
    expect = (-torch.log(2.0 + diag.sum(-1))).unsqueeze(-2).unsqueeze(-2)  # trace, then query/head-broadcast
    assert b_full.shape == expect.shape
    assert torch.allclose(b_full, expect, atol=1e-6)


def test_precision_bias_per_head_full_cov_uses_block_trace():
    # The full-cov per-head bias must use each block's diagonal trace, shape (.., H, 1, N).
    m = _fullcov_model(seed=0, precision_attention_per_head=True)
    k = m.cfg.embed_dim
    torch.manual_seed(2)
    diag = torch.rand(2, 4, k) + 0.5
    off  = 0.3 * torch.randn(2, 4, k, k)
    off  = off + off.transpose(-1, -2)
    off  = off - torch.diag_embed(off.diagonal(dim1=-2, dim2=-1))
    full = torch.diag_embed(diag) + off
    b_ph = m._fold_precision_bias(None, full)                              # (.., H, 1, N)
    dims = list(m.group.irrep_dims)
    tr   = torch.stack([blk.sum(-1) for blk in diag.split(dims, dim=-1)], dim=-1)  # (.., N, H)
    expect = (-torch.log(2.0 + tr)).transpose(-1, -2).unsqueeze(-2)        # (.., H, 1, N)
    assert b_ph.shape == expect.shape
    assert torch.allclose(b_ph, expect, atol=1e-6)
