r"""Two default-OFF diagnostic toggles probing whether the belief covariance Sigma carries
predictive signal, while keeping the discriminative linear decode (no NN added beyond the existing
linear-decode exception; both pure paths preserved when off).

(1) decode_precision_scaled (use_prior_bank=False only): feed the precision-weighted mean -- the
    diagonal natural parameter eta = Sigma^-1 mu = mu/sigma -- to the linear head instead of mu, so
    Sigma_q enters the DISCRIMINATIVE readout. Isolates "does Sigma help at the readout" from the
    generative/capacity confound of the prior-bank KL decode (which the user finds underperforms).

(2) precision_weighted_attention: fold a detached per-key reliability bias -log(b0 + tr Sigma_j)
    into the attention log_prior, so attention down-weights high-variance (unreliable) keys before
    the softmax. A uniform-over-keys Sigma is softmax-absorbed (no effect); only key-to-key variance
    in Sigma changes attention. Detached -> the closed-form belief kernel stays exact.

Both default OFF and byte-identical to the current path when off.
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


# =========================================================================== feature 1: decode

def test_decode_precision_scaled_defaults_off():
    assert VFE3Config().decode_precision_scaled is False


def test_decode_precision_scaled_off_is_plain_linear():
    # OFF (default): linear decode is exactly mu_q @ W^T (+ b), sigma_q ignored.
    m = _model(use_prior_bank=False)
    pb = m.prior_bank
    mu = torch.randn(3, 5, 4)
    sigma = torch.rand(3, 5, 4) + 0.5
    logits = pb.decode(mu, sigma)
    expect = mu @ pb.output_proj_weight.transpose(-1, -2)
    if pb.output_proj_bias is not None:
        expect = expect + pb.output_proj_bias
    assert torch.allclose(logits, expect, atol=1e-6)


def test_decode_precision_scaled_on_uses_natural_parameter():
    # ON: the linear head reads eta = mu / (sigma + eps) instead of mu.
    m = _model(use_prior_bank=False, decode_precision_scaled=True)
    pb = m.prior_bank
    mu = torch.randn(3, 5, 4)
    sigma = torch.rand(3, 5, 4) + 0.5
    logits = pb.decode(mu, sigma)
    eta = mu / (sigma + pb.eps)
    expect = eta @ pb.output_proj_weight.transpose(-1, -2)
    if pb.output_proj_bias is not None:
        expect = expect + pb.output_proj_bias
    assert torch.allclose(logits, expect, atol=1e-6)
    # and it genuinely differs from the mu-only readout (sigma is not constant here)
    assert not torch.allclose(logits, mu @ pb.output_proj_weight.transpose(-1, -2)
                              + (pb.output_proj_bias if pb.output_proj_bias is not None else 0.0),
                              atol=1e-4)


def test_decode_precision_scaled_warns_inert_under_prior_bank():
    # Like decode_bias: meaningful only on the linear path; warn when use_prior_bank=True.
    with pytest.warns(UserWarning, match="decode_precision_scaled"):
        VFE3Config(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5,
                   use_prior_bank=True, decode_precision_scaled=True)


def test_decode_precision_scaled_grad_flows():
    m = _model(use_prior_bank=False, decode_precision_scaled=True)
    tokens = torch.randint(0, 20, (3, 5)); targets = torch.randint(0, 20, (3, 5))
    _, loss, _ = m(tokens, targets)
    loss.backward()
    g = m.prior_bank.output_proj_weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


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
