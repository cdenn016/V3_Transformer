r"""Pins for the 2026-07-05 deep-audit fixes (docs/audit-results.md).

M1  _fold_precision_bias no longer @torch.no_grad: t5_bias trains under
    precision_weighted_attention=True + t5_learnable_bias=True.
m1  FullGaussian.renyi_closed_form drops the unconditional eps*I ridge (pure full-cov KL).
m4  the free_energy_terms metric wrapper forwards lambda_beta / include_attention_entropy / alpha_reg.
m6  pos_rotation='rope' warns that gauge-RoPE breaks global gauge equivariance (every group).
m7  transport_covariance rejects an ambiguous sigma rank instead of mis-dispatching.
m8  train-loop warn-once when a gauge-frame table's embedded norm exceeds the transport clamp.
m9  the oracle-route freeze warning covers t5_bias.
m11 _refine_s threads e_step_mu_precond (source-wiring pin).
m12 lambda_h_mode='state_dependent' warns that the lambda_h VALUE is only a gate.
m13 the oracle_unroll_grad auto-enable coercions warn.
m10 fp32 islands key autocast-disable to the tensor's device (CPU-AMP safe).
"""

import inspect
import warnings as _warnings
from types import SimpleNamespace

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.families.gaussian import FullGaussian
from vfe3.geometry.retraction import natural_gradient
from vfe3.geometry.transport import transport_covariance
from vfe3.metrics import compute_metrics, free_energy_terms
from vfe3.model.model import VFEModel


def _t5_precision_cfg(**kw):
    base = dict(vocab_size=12, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                n_e_steps=1, e_phi_lr=0.0, m_phi_lr=0.0,
                beta_attention_prior="t5_relative_bias", t5_learnable_bias=True,
                precision_weighted_attention=True)
    base.update(kw)
    return VFE3Config(**base)


# --- M1: precision-bias fold must not sever the learnable T5 prior's graph -------------

def test_t5_bias_grad_flows_under_precision_weighted_attention():
    # End-to-end: before the fix, the @torch.no_grad on _fold_precision_bias made the folded
    # log_prior requires_grad=False, so t5_bias.grad stayed None under this double opt-in.
    torch.manual_seed(0)
    m = VFEModel(_t5_precision_cfg())
    x = torch.randint(0, 12, (2, 8)); y = torch.randint(0, 12, (2, 8))
    _, loss, _ = m(x, y); loss.backward()
    assert torch.isfinite(loss)
    assert m.t5_bias.grad is not None
    assert torch.isfinite(m.t5_bias.grad).all() and m.t5_bias.grad.abs().sum() > 0


def test_fold_precision_bias_keeps_log_prior_graph_and_detaches_kb():
    m = VFEModel(_t5_precision_cfg())
    n = 4
    log_prior = torch.zeros(n, n, requires_grad=True)
    sigma = torch.rand(n, m.cfg.embed_dim) + 0.5
    sigma.requires_grad_(True)
    out = m._fold_precision_bias(log_prior, sigma)
    assert out.requires_grad                       # the prior's graph survives the fold (M1)
    out.sum().backward()
    assert log_prior.grad is not None and torch.allclose(log_prior.grad, torch.ones(n, n))
    assert sigma.grad is None                      # kb stays detached: no gradient into the key sigma


# --- m1: no unconditional eps ridge on the full-covariance KL --------------------------

def test_full_cov_kl_matches_analytic_without_ridge():
    # Small variances make the old +1e-6*I ridge a ~1e-3-relative bias; the fixed kernel must
    # match the exact diagonal-as-full KL far tighter than that.
    K = 3
    sq = 1e-3 * torch.ones(K)
    st = 2e-3 * torch.ones(K)
    mu_q = torch.zeros(1, K)
    mu_t = 0.01 * torch.ones(1, K)
    q = FullGaussian(mu_q, torch.diag_embed(sq).unsqueeze(0))
    p = FullGaussian(mu_t, torch.diag_embed(st).unsqueeze(0))
    got = q.renyi_closed_form(p, alpha=1.0, kl_max=float("inf"))
    # exact diagonal KL in float64
    sq64, st64 = sq.double(), st.double()
    d64 = (mu_t - mu_q).double()
    expect = 0.5 * ((sq64 / st64).sum() + (d64 ** 2 / st64).sum() - K
                    + (torch.log(st64) - torch.log(sq64)).sum())
    assert torch.allclose(got.double(), expect.reshape_as(got.double()), rtol=1e-5, atol=1e-7), \
        f"got {got.item()}, expect {expect.item()}"


def test_full_cov_self_kl_is_zero():
    K = 3
    sigma = torch.diag_embed(1e-3 * torch.ones(K)).unsqueeze(0)
    q = FullGaussian(torch.zeros(1, K), sigma)
    # With the ridge removed, KL(q||q) is exactly the analytic 0 (up to fp32 rounding of
    # identical terms cancelling, which is 0 here since both arguments are the same tensor).
    assert q.renyi_closed_form(q, alpha=1.0).abs().item() < 1e-6


# --- m7: transport_covariance rank-gap hardening ----------------------------------------

def test_transport_covariance_ambiguous_sigma_raises():
    N, K, B = 3, 4, 2
    omega = torch.eye(K).expand(N, N, K, K)               # (N, N, K, K), batch-independent
    sigma_batched_diag = torch.rand(B, N, K) + 0.5        # dim == omega.dim()-1: the m7 trap
    with pytest.raises(ValueError, match="diagonal_out"):
        transport_covariance(omega, sigma_batched_diag)
    # explicit disambiguation keeps working
    out = transport_covariance(omega, sigma_batched_diag[0], diagonal_out=True)
    assert out.shape == (N, N, K)


def test_transport_covariance_valid_full_still_passes():
    N, K = 3, 4
    omega = torch.eye(K).expand(N, N, K, K)
    sigma_full = torch.diag_embed(torch.rand(N, K) + 0.5)  # (N, K, K): genuine full covariance
    out = transport_covariance(omega, sigma_full)          # identity transport -> unchanged
    assert out.shape == (N, N, K, K)
    assert torch.allclose(out[0, 1], sigma_full[1], atol=1e-6)


# --- m4: metric wrapper forwards the scaling/entropy/regularizer knobs ------------------

def test_free_energy_terms_metric_forwards_all_knobs():
    g = torch.Generator().manual_seed(0)
    N = 4
    self_div = torch.rand(N, generator=g)
    energy = torch.rand(N, N, generator=g)
    beta = torch.softmax(-energy, dim=-1)
    alpha = torch.ones(N)
    alpha_reg = 0.3 * torch.ones(N)
    direct = free_energy_terms(self_div, energy, beta, alpha, tau=2.0, lambda_beta=0.5,
                               include_attention_entropy=False, alpha_reg=alpha_reg)
    via_registry = compute_metrics(["free_energy_terms"], self_div=self_div, energy=energy,
                                   beta=beta, alpha=alpha, tau=2.0, lambda_beta=0.5,
                                   include_attention_entropy=False, alpha_reg=alpha_reg)
    assert via_registry["free_energy_terms"] == direct    # dropped before the fix (lambda_beta=1 etc.)


# --- m6 / m9 / m12 / m13: config warning coverage ----------------------------------------

def test_rope_warns_gauge_fixing_for_every_group():
    with pytest.warns(UserWarning, match="gauge-FIXING"):
        VFE3Config(embed_dim=4, n_heads=2, pos_rotation="rope")


def test_state_dependent_lambda_h_nontrivial_value_warns():
    with pytest.warns(UserWarning, match="ignores the lambda_h VALUE"):
        VFE3Config(embed_dim=4, n_heads=2, lambda_h_mode="state_dependent", lambda_h=5.0)
    # gate values (0=off, 1=bare gate) stay silent
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        try:
            VFE3Config(embed_dim=4, n_heads=2, lambda_h_mode="state_dependent", lambda_h=1.0)
        except UserWarning as w:                           # unrelated warnings may exist; only
            assert "ignores the lambda_h VALUE" not in str(w)   # THIS one must not fire


def test_regime_ii_auto_enable_oracle_unroll_warns():
    with pytest.warns(UserWarning, match="oracle_unroll_grad auto-enabled"):
        cfg = VFE3Config(embed_dim=4, n_heads=2, transport_mode="regime_ii")
    assert cfg.oracle_unroll_grad is True                  # the coercion itself is unchanged


def test_oracle_freeze_warning_names_t5_bias():
    # Non-kernel route (renyi_order != 1) + detached oracle + active learnable t5 channel:
    # the freeze warning must now name t5_bias (it listed every other E-step-only param).
    with pytest.warns(UserWarning, match="t5_bias"):
        VFE3Config(embed_dim=4, n_heads=2, renyi_order=0.5,
                   beta_attention_prior="t5_relative_bias", t5_learnable_bias=True,
                   e_step_gradient="unroll", oracle_unroll_grad=False)


# --- m8: warn-once when the gauge frame exceeds the transport clamp ---------------------

def test_phi_transport_clamp_warns_once():
    import vfe3.train as vtrain
    K, n_gen = 3, 4
    gen = torch.randn(n_gen, K, K)
    big_phi = torch.zeros(5, n_gen); big_phi[2, 1] = 1e3   # embedded norm far past max_norm=15
    stub = SimpleNamespace(group=SimpleNamespace(generators=gen),
                           prior_bank=SimpleNamespace(phi_embed=big_phi))
    old_flag = vtrain._PHI_CLAMP_WARNED
    try:
        vtrain._PHI_CLAMP_WARNED = False
        with pytest.warns(RuntimeWarning, match="transport clamp"):
            vtrain._warn_phi_transport_clamp(stub)
        assert vtrain._PHI_CLAMP_WARNED is True
        with _warnings.catch_warnings():
            _warnings.simplefilter("error")                 # second call: warn-once, so silent
            vtrain._warn_phi_transport_clamp(stub)
    finally:
        vtrain._PHI_CLAMP_WARNED = old_flag


def test_phi_transport_clamp_silent_below_threshold():
    import vfe3.train as vtrain
    K, n_gen = 3, 4
    gen = torch.randn(n_gen, K, K)
    small_phi = 0.01 * torch.randn(5, n_gen)
    stub = SimpleNamespace(group=SimpleNamespace(generators=gen),
                           prior_bank=SimpleNamespace(phi_embed=small_phi))
    old_flag = vtrain._PHI_CLAMP_WARNED
    try:
        vtrain._PHI_CLAMP_WARNED = False
        with _warnings.catch_warnings():
            _warnings.simplefilter("error")
            vtrain._warn_phi_transport_clamp(stub)
        assert vtrain._PHI_CLAMP_WARNED is False
    finally:
        vtrain._PHI_CLAMP_WARNED = old_flag


# --- m11: the s-channel E-step threads the mean-arm preconditioner ----------------------

def test_refine_s_threads_e_step_mu_precond():
    # Wiring pin: _refine_s previously omitted e_step_mu_precond, silently running the default
    # 'fisher' s-refine under an e_step_mu_precond='raw' ablation. A behavioral probe would need
    # the full s_e_step harness; the wiring is what regressed, so the wiring is what is pinned.
    src = inspect.getsource(VFEModel._refine_s)
    assert "e_step_mu_precond=cfg.e_step_mu_precond" in src


# --- m10: fp32 islands hold under CPU autocast ------------------------------------------

def test_natural_gradient_full_cov_immune_to_cpu_autocast():
    g = torch.Generator().manual_seed(0)
    N, K = 4, 6
    A = torch.randn(N, K, K, generator=g)
    sigma = A @ A.transpose(-1, -2) + 0.5 * torch.eye(K)
    grad_mu = torch.randn(N, K, generator=g)
    grad_sigma = torch.randn(N, K, K, generator=g)
    ref_mu, ref_sigma = natural_gradient(grad_mu, grad_sigma, sigma)
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        amp_mu, amp_sigma = natural_gradient(grad_mu, grad_sigma, sigma)
    # the island now disables the ACTIVE (cpu) autocast, so the einsums run fp32: bit-identical
    assert torch.equal(ref_mu, amp_mu)
    assert torch.equal(ref_sigma, amp_sigma)
