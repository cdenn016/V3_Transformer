r"""Tier-1/Tier-2 decode toggles (2026-07-05 ideas doc): unigram log-prior decode,
expected-likelihood decode kernel, untied decode bank, and the fused-CE z-loss kwarg.

All four default OFF; these tests pin (a) the unigram shift is exactly kappa * log pi_v on the
dense path and enters every chunk of the fused CE, (b) expected_likelihood_chunked matches a
naive per-vocab dense build and its fused CE equals the dense -log-softmax gather, (c) the
untied decode tables start byte-identical to the encode tables and split cleanly after a
perturbation, (d) z_loss_weight adds exactly w * mean(logsumexp^2) and is bit-identical at 0.0.
Device-agnostic (CPU default; honors VFE3_TEST_DEVICE).
"""

import os
import warnings

import pytest
import torch
import torch.nn.functional as F

import vfe3.model.prior_bank as prior_bank_module
from vfe3.config import VFE3Config
from vfe3.model.prior_bank import PriorBank, _CHUNKED_DECODERS, _FULL_DECODERS

DEVICE = torch.device(os.environ.get("VFE3_TEST_DEVICE", "cpu"))

V, K, N_GEN = 50, 8, 4


def _bank(seed=0, **kw):
    """Seeded PriorBank; identical seeds give identical tables across toggle variants."""
    torch.manual_seed(seed)
    return PriorBank(V, K, N_GEN, mu_init_std=0.5, **kw).to(DEVICE)


def _q(B=2, N=5, seed=1):
    torch.manual_seed(seed)
    mu_q = 0.3 * torch.randn(B, N, K)
    sigma_q = 0.5 + 0.5 * torch.rand(B, N, K)
    return mu_q.to(DEVICE), sigma_q.to(DEVICE)


def _targets(B=2, N=5, seed=2):
    torch.manual_seed(seed)
    return torch.randint(0, V, (B, N)).to(DEVICE)


def _counts(seed=3):
    torch.manual_seed(seed)
    return torch.randint(0, 100, (V,)).float().to(DEVICE)


def _dense_ce(logits, targets):
    return F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), ignore_index=-100)


# ---------------------------------------------------------------------------
# (a) unigram log-prior decode
# ---------------------------------------------------------------------------

def test_unigram_dense_shift_is_exactly_kappa_log_prior():
    off = _bank()
    on = _bank(decode_unigram_prior=True, unigram_kappa=0.7)
    counts = _counts()
    on.set_unigram_log_prior(counts)
    log_prior = torch.log((counts + 1.0) / (counts.sum() + float(V)))    # add-one smoothing
    mu_q, sigma_q = _q()
    shift = on.decode(mu_q, sigma_q) - off.decode(mu_q, sigma_q)         # (B, N, V)
    assert torch.allclose(shift, (0.7 * log_prior).expand_as(shift), atol=1e-5)


def test_unigram_fused_ce_equals_dense_ce_with_shifted_logits():
    on = _bank(decode_unigram_prior=True, unigram_kappa=1.1)
    on.set_unigram_log_prior(_counts())
    mu_q, sigma_q = _q()
    targets = _targets()
    dense = on.decode(mu_q, sigma_q)                                     # shifted logits
    for chunk in (7, V, 100):
        ce = on.decode_ce_diagonal_chunked(mu_q, sigma_q, targets, chunk_size=chunk)
        assert torch.allclose(ce, _dense_ce(dense, targets), atol=1e-5), f"chunk={chunk}"


def test_unigram_linear_path_and_linear_fused_ce():
    off = _bank(use_prior_bank=False)
    on = _bank(use_prior_bank=False, decode_unigram_prior=True, unigram_kappa=1.3)
    counts = _counts()
    on.set_unigram_log_prior(counts)
    log_prior = torch.log((counts + 1.0) / (counts.sum() + float(V)))
    mu_q, sigma_q = _q()
    shift = on.decode(mu_q, sigma_q) - off.decode(mu_q, sigma_q)
    assert torch.allclose(shift, (1.3 * log_prior).expand_as(shift), atol=1e-5)
    targets = _targets()
    dense = on.decode(mu_q, sigma_q)
    ce = on.decode_ce_linear_chunked(mu_q, targets, chunk_size=7)
    assert torch.allclose(ce, _dense_ce(dense, targets), atol=1e-5)


def test_unigram_unset_table_warns_once_and_is_a_value_noop():
    prior_bank_module._WARNED_UNIGRAM_UNSET = False                      # deterministic once-per-process
    off = _bank()
    on = _bank(decode_unigram_prior=True)                                # table left all-zero
    mu_q, sigma_q = _q()
    with pytest.warns(UserWarning, match="unigram_log_prior"):
        logits_on = on.decode(mu_q, sigma_q)
    assert torch.equal(logits_on, off.decode(mu_q, sigma_q))             # kappa * 0 = uniform-prior no-op
    with warnings.catch_warnings():                                      # second call: silent
        warnings.simplefilter("error")
        on.decode(mu_q, sigma_q)


def test_set_unigram_log_prior_requires_toggle_and_shape():
    pb = _bank()
    with pytest.raises(RuntimeError, match="decode_unigram_prior=True"):
        pb.set_unigram_log_prior(torch.ones(V, device=DEVICE))
    on = _bank(decode_unigram_prior=True)
    with pytest.raises(ValueError, match="shape"):
        on.set_unigram_log_prior(torch.ones(V + 1, device=DEVICE))


# ---------------------------------------------------------------------------
# (b) expected-likelihood decode kernel
# ---------------------------------------------------------------------------

def _naive_el_logits(pb, mu_q, sigma_q):
    """Naive per-vocab dense build of log N(mu_q; mu_v, Sigma_q + Sigma_v) / tau_eff (no constant)."""
    mu_v = pb._decode_mu_table()
    sigma_v = torch.exp(pb._decode_sigma_log_table()).clamp(min=pb.eps)
    tau_eff = pb._tau_eff(None)
    cols = []
    for v in range(pb.vocab_size):
        s = sigma_q + sigma_v[v]                                          # (B, N, K)
        d = mu_q - mu_v[v]                                                # (B, N, K)
        cols.append(-0.5 * (d ** 2 / s + torch.log(s)).sum(-1) / tau_eff)
    return torch.stack(cols, dim=-1)                                      # (B, N, V)


def test_expected_likelihood_registered_at_diagonal_rank():
    # Config's rank cross-check reads _FULL_DECODERS; the new kernel must land diagonal.
    assert "expected_likelihood_chunked" in _CHUNKED_DECODERS
    assert "expected_likelihood_chunked" not in _FULL_DECODERS


def test_expected_likelihood_config_validates_diagonal_only():
    VFE3Config(decode_mode="expected_likelihood_chunked", use_prior_bank=True)   # diagonal family: OK
    with pytest.raises(ValueError, match="rank-incompatible"):
        VFE3Config(decode_mode="expected_likelihood_chunked", use_prior_bank=True,
                   family="gaussian_full")


def test_expected_likelihood_decode_matches_naive_dense():
    pb = _bank(decode_mode="expected_likelihood_chunked", decode_chunk_size=7)
    mu_q, sigma_q = _q()
    dense = _naive_el_logits(pb, mu_q, sigma_q)
    out = pb.decode(mu_q, sigma_q)
    assert out.shape == dense.shape
    assert torch.allclose(out, dense, atol=1e-5)


def test_expected_likelihood_fused_ce_matches_dense_log_softmax():
    pb = _bank(decode_mode="expected_likelihood_chunked")
    mu_q, sigma_q = _q()
    targets = _targets()
    dense = _naive_el_logits(pb, mu_q, sigma_q)
    ref = -F.log_softmax(dense, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1).mean()
    for chunk in (7, V, 100):
        ce = pb.decode_ce_expected_likelihood_chunked(mu_q, sigma_q, targets, chunk_size=chunk)
        assert torch.allclose(ce, ref, atol=1e-5), f"chunk={chunk}"


def test_expected_likelihood_composes_with_unigram():
    pb = _bank(decode_mode="expected_likelihood_chunked", decode_unigram_prior=True,
               unigram_kappa=0.9)
    pb.set_unigram_log_prior(_counts())
    mu_q, sigma_q = _q()
    targets = _targets()
    dense = pb.decode(mu_q, sigma_q)                                     # includes the unigram bias
    ce = pb.decode_ce_expected_likelihood_chunked(mu_q, sigma_q, targets, chunk_size=7)
    assert torch.allclose(ce, _dense_ce(dense, targets), atol=1e-5)


# ---------------------------------------------------------------------------
# (c) untied decode bank
# ---------------------------------------------------------------------------

def test_untied_tables_equal_encode_tables_at_init_and_decode_is_byte_identical():
    tied = _bank()
    un = _bank(untie_decode_bank=True)
    assert torch.equal(un.decode_mu_embed, un.mu_embed)
    assert torch.equal(un.decode_sigma_log_embed, un.sigma_log_embed)
    assert un.decode_mu_embed is not un.mu_embed                         # separate Parameter objects
    mu_q, sigma_q = _q()
    assert torch.equal(un.decode(mu_q, sigma_q), tied.decode(mu_q, sigma_q))  # step-0 byte-identical


def test_untied_perturbation_moves_decode_but_not_encode():
    un = _bank(untie_decode_bank=True)
    mu_q, sigma_q = _q()
    targets = _targets()
    tokens = _targets(seed=4)
    base_logits = un.decode(mu_q, sigma_q)
    base_ce = un.decode_ce_diagonal_chunked(mu_q, sigma_q, targets, chunk_size=7)
    mu_embed_before = un.mu_embed.detach().clone()
    enc_before = un.encode(tokens)
    with torch.no_grad():
        un.decode_mu_embed.add_(0.05)
    assert not torch.allclose(un.decode(mu_q, sigma_q), base_logits, atol=1e-6)
    assert not torch.allclose(
        un.decode_ce_diagonal_chunked(mu_q, sigma_q, targets, chunk_size=7), base_ce, atol=1e-6)
    assert torch.equal(un.mu_embed, mu_embed_before)                     # encode tables untouched
    assert torch.equal(un.encode(tokens).mu, enc_before.mu)


def test_untied_gradient_lands_on_decode_tables_only():
    un = _bank(decode_mode="expected_likelihood_chunked", untie_decode_bank=True,
               decode_unigram_prior=True)
    un.set_unigram_log_prior(_counts())
    mu_q, sigma_q = _q()
    ce = un.decode_ce_expected_likelihood_chunked(mu_q, sigma_q, _targets(),
                                                  chunk_size=7, z_loss_weight=0.1)
    ce.backward()
    assert un.decode_mu_embed.grad is not None
    assert un.decode_sigma_log_embed.grad is not None
    assert un.mu_embed.grad is None                                      # decode no longer reads it
    assert un.sigma_log_embed.grad is None


# ---------------------------------------------------------------------------
# (d) z-loss kwarg on the fused CE paths
# ---------------------------------------------------------------------------

def test_z_loss_diagonal_matches_dense_lse_and_is_bit_identical_at_zero():
    pb = _bank()
    mu_q, sigma_q = _q()
    targets = _targets()
    targets[0, 1] = -100                                                 # pin the masked-mean semantics
    valid = targets != -100
    ce0 = pb.decode_ce_diagonal_chunked(mu_q, sigma_q, targets, chunk_size=7)
    ce0_explicit = pb.decode_ce_diagonal_chunked(mu_q, sigma_q, targets, chunk_size=7,
                                                 z_loss_weight=0.0)
    assert torch.equal(ce0, ce0_explicit)                                # 0.0 is the guarded no-op
    w = 0.37
    cew = pb.decode_ce_diagonal_chunked(mu_q, sigma_q, targets, chunk_size=7, z_loss_weight=w)
    lse = pb.decode(mu_q, sigma_q).logsumexp(-1)                         # (B, N) dense logsumexp
    expected = ce0 + w * (lse ** 2)[valid].mean()
    assert torch.allclose(cew, expected, atol=1e-5)


def test_z_loss_linear_matches_dense_lse():
    pb = _bank(use_prior_bank=False)
    mu_q, sigma_q = _q()
    targets = _targets()
    w = 0.21
    ce0 = pb.decode_ce_linear_chunked(mu_q, targets, chunk_size=7)
    cew = pb.decode_ce_linear_chunked(mu_q, targets, chunk_size=7, z_loss_weight=w)
    lse = pb.decode(mu_q, sigma_q).logsumexp(-1)
    assert torch.allclose(cew, ce0 + w * (lse ** 2).mean(), atol=1e-5)


def test_z_loss_full_chunked_matches_dense_lse():
    pb = _bank(decode_mode="full_chunked", diagonal_covariance=False)
    mu_q, sigma_q = _q()
    sigma_full = torch.diag_embed(sigma_q)                               # (B, N, K, K) SPD
    targets = _targets()
    w = 0.15
    ce0 = pb.decode_ce_full_chunked(mu_q, sigma_full, targets, chunk_size=7)
    cew = pb.decode_ce_full_chunked(mu_q, sigma_full, targets, chunk_size=7, z_loss_weight=w)
    lse = pb.decode(mu_q, sigma_full).logsumexp(-1)                      # full_chunked logits kernel
    assert torch.allclose(cew, ce0 + w * (lse ** 2).mean(), atol=1e-5)


def test_z_loss_expected_likelihood_matches_dense_lse():
    pb = _bank(decode_mode="expected_likelihood_chunked")
    mu_q, sigma_q = _q()
    targets = _targets()
    w = 0.5
    ce0 = pb.decode_ce_expected_likelihood_chunked(mu_q, sigma_q, targets, chunk_size=7)
    cew = pb.decode_ce_expected_likelihood_chunked(mu_q, sigma_q, targets, chunk_size=7,
                                                   z_loss_weight=w)
    lse = pb.decode(mu_q, sigma_q).logsumexp(-1)
    assert torch.allclose(cew, ce0 + w * (lse ** 2).mean(), atol=1e-5)


def test_z_loss_applied_on_dense_decode():
    # m20: z_loss must add the logsumexp^2 penalty on the DENSE (non-chunked) decode path too, not
    # only the four fused chunked kernels. decode_mode="diagonal" routes through VFEModel's dense branch.
    from vfe3.model.model import VFEModel
    torch.manual_seed(0)
    cfg = VFE3Config(vocab_size=12, embed_dim=4, n_heads=2, max_seq_len=6, n_layers=1,
                     decode_mode="diagonal", use_prior_bank=True, mass_phi=0.0, z_loss_weight=0.0)
    model = VFEModel(cfg).to(DEVICE)
    x = torch.randint(0, 12, (2, 6), device=DEVICE)
    y = torch.randint(0, 12, (2, 6), device=DEVICE)
    logits, loss0, _ = model(x, y)
    assert logits is not None                                    # dense path returns (B, N, V) logits
    V = cfg.vocab_size
    lse = torch.logsumexp(logits.reshape(-1, V).float(), dim=-1)
    valid = (y.reshape(-1) != -100).to(lse.dtype)
    w = 0.5
    expected = w * (lse ** 2 * valid).sum() / valid.sum().clamp(min=1)
    model.cfg.z_loss_weight = w
    _, loss_w, _ = model(x, y)
    assert loss_w > loss0                                        # RED pre-fix: dense branch ignored z-loss
    assert torch.allclose(loss_w - loss0, expected, atol=1e-5)
