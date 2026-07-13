r"""Tier-1/Tier-2 attention toggles (2026-07-05 ideas doc): no-self attention priors,
per-query adaptive temperature, the gamma-as-beta-prior hierarchical fold, and the
two-hop coupling F-term. Tiny shapes; device-agnostic (set VFE3_TEST_DEVICE=cuda).
"""
import os

import pytest
import torch

from vfe3.attention_prior import attention_log_prior
from vfe3.config import VFE3Config
from vfe3.free_energy import (attention_tau, attention_weights, free_energy,
                              query_adaptive_tau, reduced_free_energy)
from vfe3.model.model import VFEModel

DEVICE = torch.device(os.environ.get("VFE3_TEST_DEVICE", "cpu"))


def _cfg(**over):
    base = dict(vocab_size=32, embed_dim=4, n_heads=2, max_seq_len=8, n_e_steps=2)
    base.update(over)
    return VFE3Config(**base)


# ---------------------------------------------------------------- (a) no-self priors

def test_causal_noself_diagonal_masked_except_00():
    B = attention_log_prior("causal_noself", 6, 6, device=DEVICE)
    diag = B.diagonal()
    assert diag[0] == 0.0                                    # (0,0) stays allowed (row 0's only key)
    assert torch.isinf(diag[1:]).all() and (diag[1:] < 0).all()
    beta = torch.softmax(B, dim=-1)                          # every row keeps >= 1 finite logit
    assert torch.isfinite(beta).all()
    assert torch.allclose(beta.sum(-1), torch.ones(6, device=DEVICE))


def test_causal_noself_equals_causal_off_diagonal():
    Bn = attention_log_prior("causal_noself", 7, 7, device=DEVICE)
    Bc = attention_log_prior("causal", 7, 7, device=DEVICE)
    off = ~torch.eye(7, dtype=torch.bool, device=DEVICE)
    assert torch.equal(Bn[off], Bc[off])


def test_causal_alibi_noself_shape_mask_and_offdiagonal():
    B  = attention_log_prior("causal_alibi_noself", 5, 5, n_heads=2, device=DEVICE)
    Ba = attention_log_prior("causal_alibi",        5, 5, n_heads=2, device=DEVICE)
    assert B.shape == (2, 5, 5)
    diag = B.diagonal(dim1=-2, dim2=-1)                      # (2, 5)
    assert (diag[:, 0] == 0.0).all()                         # (0,0) allowed on every head
    assert torch.isinf(diag[:, 1:]).all()
    off = ~torch.eye(5, dtype=torch.bool, device=DEVICE)
    assert torch.equal(B[:, off], Ba[:, off])                # identical off-diagonal per head
    assert torch.isfinite(torch.softmax(B, dim=-1)).all()


# ------------------------------------------------------- (b) per-query adaptive tau

def test_query_adaptive_tau_monotone_detached_and_c0_inert():
    tr = torch.tensor([0.1, 0.5, 1.0, 2.0], device=DEVICE).view(4, 1)
    sigma = tr.expand(4, 8).clone().requires_grad_(True)     # rows of increasing trace
    t = query_adaptive_tau(sigma, 2.0, [4, 4], c=1.0)
    assert t.shape == (2, 4, 1)                              # (H, N, 1) per-query, per-head
    assert not t.requires_grad                               # DETACHED: no grad path into sigma
    rows = t[0, :, 0]
    assert (rows[1:] > rows[:-1]).all()                      # monotone in tr Sigma_i
    t0 = query_adaptive_tau(sigma, 2.0, [4, 4], c=0.0)       # c=0 -> the base tau on every row
    assert torch.equal(t0, torch.full_like(t0, 2.0))


def test_attention_weights_per_row_tau_rows_normalize():
    torch.manual_seed(0)
    energy = torch.rand(2, 5, 5, device=DEVICE)              # (H, N, N)
    sigma  = torch.rand(5, 8, device=DEVICE) + 0.1
    tau    = query_adaptive_tau(sigma, attention_tau(1.0, [4, 4]), [4, 4], c=0.7)
    lp     = attention_log_prior("causal_noself", 5, 5, device=DEVICE)
    beta   = attention_weights(energy, tau=tau, log_prior=lp)
    assert beta.shape == (2, 5, 5)
    assert torch.isfinite(beta).all()
    assert torch.allclose(beta.sum(-1), torch.ones(2, 5, device=DEVICE), atol=1e-6)


def test_reduced_free_energy_per_row_tau():
    torch.manual_seed(0)
    energy = torch.rand(2, 4, 4, device=DEVICE)
    sigma  = torch.rand(4, 8, device=DEVICE) + 0.1
    tau    = query_adaptive_tau(sigma, 2.0, [4, 4], c=0.5)   # (2, 4, 1)
    out    = reduced_free_energy(energy, tau=tau)
    assert out.shape == (2, 4)
    lz = torch.logsumexp(-energy / tau - torch.log(torch.tensor(4.0, device=DEVICE)), dim=-1)
    assert torch.allclose(out, -tau.squeeze(-1) * lz, atol=1e-6)


def test_query_tau_off_path_identical_and_on_path_load_bearing():
    def _logits(**over):
        torch.manual_seed(0)
        model = VFEModel(_cfg(**over)).to(DEVICE)
        torch.manual_seed(1)
        tok = torch.randint(0, 32, (1, 6), device=DEVICE)
        return model(tok)
    l_off = _logits()                                        # toggle OFF: the scalar-tau path
    l_c0  = _logits(query_adaptive_tau=True, query_tau_c=0.0)
    l_on  = _logits(query_adaptive_tau=True, query_tau_c=5.0)
    # c=0 makes the per-row tau EXACTLY the base tau on every row, so the toggle-on forward is
    # bitwise the scalar-tau computation -- the OFF path's byte-identity proxy.
    assert torch.equal(l_off, l_c0)
    assert not torch.allclose(l_off, l_on)                   # the adaptive tau is load-bearing


@pytest.mark.parametrize("c", [-1.0, float("nan"), float("inf")])
def test_query_adaptive_tau_rejects_nonfinite_or_negative_c(c):
    sigma = torch.ones(2, 3, 4)
    with pytest.raises(ValueError, match="c must be finite and >= 0"):
        query_adaptive_tau(sigma, 1.0, [4], c=c)


# ------------------------------------------------ (c) gamma-as-beta-prior fold

def test_gamma_fold_rows_normalize_and_preserve_inf():
    torch.manual_seed(0)
    model = VFEModel(_cfg(lambda_gamma=0.5, gamma_as_beta_prior=True,
                          gamma_prior_weight=0.3)).to(DEVICE)
    tok = torch.randint(0, 32, (2, 6), device=DEVICE)
    enc = model.prior_bank.encode(tok)
    phi = model._apply_pos_phi(enc.phi)
    lp  = model._attention_log_prior(6, tok.device)          # (N, N) causal
    folded = model._fold_gamma_prior(lp, tok, phi)           # (B, [H,] N, N)
    support = torch.isfinite(lp)
    masked = folded[..., ~support]
    assert torch.isinf(masked).all() and (masked < 0).all()  # EXACT -inf causal structure kept
    rowsum = folded.exp().sum(dim=-1)                        # exp(log pi) sums to 1 over the support
    assert torch.allclose(rowsum, torch.ones_like(rowsum), atol=1e-5)


def test_gamma_fold_no_grad_to_s_tables():
    torch.manual_seed(0)
    model = VFEModel(_cfg(lambda_gamma=0.5, gamma_as_beta_prior=True)).to(DEVICE)
    tok = torch.randint(0, 32, (1, 5), device=DEVICE)
    enc = model.prior_bank.encode(tok)
    phi = model._apply_pos_phi(enc.phi)
    lp  = model._attention_log_prior(5, tok.device).clone().requires_grad_(True)
    folded = model._fold_gamma_prior(lp, tok, phi)
    folded[torch.isfinite(folded)].sum().backward()
    assert model.prior_bank.s_mu_embed.grad is None          # gamma is detached: nothing reaches s
    assert model.prior_bank.s_sigma_log_embed.grad is None
    assert lp.grad is not None                               # the belief prior's own graph stays live


def test_gamma_fold_off_path_not_invoked():
    torch.manual_seed(0)
    model = VFEModel(_cfg(lambda_gamma=0.5)).to(DEVICE)      # toggle OFF (default)

    def _boom(*args, **kwargs):
        raise AssertionError("_fold_gamma_prior must not run on the OFF path")
    model._fold_gamma_prior = _boom
    tok = torch.randint(0, 32, (1, 5), device=DEVICE)
    model.forward_beliefs(tok)                               # must not raise (byte-identical guard)


def test_gamma_fold_changes_forward():
    def _logits(**over):
        torch.manual_seed(0)
        model = VFEModel(_cfg(lambda_gamma=0.5, **over)).to(DEVICE)
        torch.manual_seed(1)
        tok = torch.randint(0, 32, (1, 6), device=DEVICE)
        return model(tok)
    l_off = _logits()
    l_on  = _logits(gamma_as_beta_prior=True, gamma_prior_weight=0.5)
    assert l_off.shape == l_on.shape
    assert not torch.allclose(l_off, l_on)                   # the folded prior is load-bearing


def test_gamma_prior_folded_in_diagnostic_replays():
    # m4: under gamma_as_beta_prior the forward folds the model-channel gamma into the belief prior, but
    # the diagnostic/figure replays folded only the precision bias -> they scored a DIFFERENT prior than
    # the forward. Assert every replay now invokes the gamma fold (off-path stays byte-identical).
    model = VFEModel(_cfg(n_layers=1, lambda_gamma=0.5, gamma_as_beta_prior=True,
                          gamma_prior_weight=0.3)).to(DEVICE)
    tok = torch.randint(0, 32, (1, 5), device=DEVICE)
    orig = model._fold_gamma_prior
    for name in ("diagnostics", "attention_maps", "gamma_attention_maps", "diagnostics_per_layer"):
        calls = {"n": 0}
        def spy(*a, _orig=orig, _c=calls, **k):
            _c["n"] += 1
            return _orig(*a, **k)
        model._fold_gamma_prior = spy
        try:
            getattr(model, name)(tok)
        finally:
            model._fold_gamma_prior = orig
        assert calls["n"] > 0, f"{name} did not fold the gamma prior under gamma_as_beta_prior"


# ------------------------------------------------------------ (d) two-hop F-term

def test_twohop_term_matches_hand_computation():
    torch.manual_seed(0)
    N, H = 5, 2
    sd     = torch.rand(N, device=DEVICE)
    alpha  = torch.ones(N, device=DEVICE)
    energy = torch.rand(H, N, N, device=DEVICE)
    lp     = attention_log_prior("causal", N, N, device=DEVICE)
    tau    = 1.7
    F0 = free_energy(sd, energy, alpha, tau=tau, log_prior=lp)
    F2 = free_energy(sd, energy, alpha, tau=tau, log_prior=lp, lambda_twohop=0.6)
    beta = attention_weights(energy, tau=tau, log_prior=lp)
    expected = 0.6 * ((beta @ beta) * energy).sum()          # lambda * sum_ik W2_ik E_ik, per head
    assert torch.allclose(F2 - F0, expected, atol=1e-5)


def test_twohop_zero_is_byte_identical():
    torch.manual_seed(1)
    sd     = torch.rand(4, device=DEVICE)
    alpha  = torch.ones(4, device=DEVICE)
    energy = torch.rand(4, 4, device=DEVICE)                 # single-block (N, N) grid
    F0  = free_energy(sd, energy, alpha, tau=2.0)
    F0b = free_energy(sd, energy, alpha, tau=2.0, lambda_twohop=0.0)
    assert torch.equal(F0, F0b)                              # lambda=0 skips the guarded block
