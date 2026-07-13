r"""Authoritative typed hierarchical free-energy evaluator (PB-10, 2026-07-12).

Before this task the q/p/s/h hierarchy had no single typed evaluator: ``free_energy()`` scored
only the q self/beta blocks, the hyper-prior lambda_h KL(s||h) and gamma model-coupling blocks were
added independently in the model, and diagnostics reconstructed them by hand. These tests pin the
new boundary:

  * ``hierarchical_free_energy_terms`` -- the exact scalar evaluator over eight already-signed,
    already-weighted per-query rows; the first four rows plus the observation row reduce with
    ``q_reduction``, the three s-channel rows with ``model_reduction``; the total is assembled in the
    fixed field order and nothing is detached or reweighted;
  * ``_belief_free_energy_rows`` -- the per-query (..., N) decomposition of the belief channel whose
    summed rows reproduce ``free_energy()`` numerically (float64), with an exact zero observation
    slot and a ``beta_override`` seam so diagnostics does not recompute attention weights;
  * the legacy scalar ``free_energy()`` keeps its byte-for-byte reduction order (float32 ``torch.equal``
    for values AND gradients against a copied pre-change oracle) -- it does NOT delegate to the rows;
  * the compatibility wrapper ``metrics.free_energy_terms`` keeps its raw public field values while its
    ``total`` alone routes through the evaluator (weights, entropy gating, negative-likelihood sign);
  * the model channel (``_gamma_coupling_rows``) carries gradient into the s tables (no hidden detach)
    and matches the pre-change ``_hyper_prior_weighted().mean() + lambda_gamma*_gamma_coupling_term``
    outer-loss assembly for H=1/H=2 and both attention-entropy gates.
"""

import math

import pytest
import torch

from vfe3.free_energy import (
    BeliefFreeEnergyRows,
    HierarchicalFreeEnergyTerms,
    _belief_free_energy_rows,
    _broadcast_tau,
    attention_weights,
    free_energy,
    hierarchical_free_energy_terms,
)
from vfe3.metrics import free_energy_terms


# ---------------------------------------------------------------------------
# 1. The evaluator sums each row exactly once, in the fixed field order.
# ---------------------------------------------------------------------------

def test_hierarchical_terms_sum_exact_components():
    torch.manual_seed(0)
    N = 3
    names = ("self_coupling", "belief_coupling", "attention_entropy", "twohop_coupling",
             "hyper_prior", "model_coupling", "meta_entropy", "observation_nll")
    rows = {n: torch.randn(N, dtype=torch.float64) for n in names}

    terms = hierarchical_free_energy_terms(
        rows["self_coupling"], rows["belief_coupling"], rows["attention_entropy"],
        rows["twohop_coupling"], rows["hyper_prior"], rows["model_coupling"],
        rows["meta_entropy"], rows["observation_nll"],
        q_reduction="sum", model_reduction="sum",
    )
    assert isinstance(terms, HierarchicalFreeEnergyTerms)

    # q_reduction (sum) rows: self, belief, attention entropy, two-hop, observation.
    for n in ("self_coupling", "belief_coupling", "attention_entropy",
              "twohop_coupling", "observation_nll"):
        assert torch.allclose(getattr(terms, n), rows[n].sum())
    # model_reduction (sum here) rows: hyper-prior, model coupling, meta-entropy.
    for n in ("hyper_prior", "model_coupling", "meta_entropy"):
        assert torch.allclose(getattr(terms, n), rows[n].sum())

    # total is the field-order sum of the eight reduced components.
    expected = sum(rows[n].sum() for n in names)
    assert torch.allclose(terms.total, expected)


def test_hierarchical_terms_validate_shape_and_reduction():
    N = 3
    z = torch.zeros(N, dtype=torch.float64)
    bad = torch.zeros(N + 1, dtype=torch.float64)
    with pytest.raises(ValueError):                                    # mismatched (..., N) shape
        hierarchical_free_energy_terms(z, z, z, z, z, z, z, bad)
    with pytest.raises(ValueError):                                    # unknown reduction
        hierarchical_free_energy_terms(z, z, z, z, z, z, z, z, q_reduction="max")
    with pytest.raises(ValueError):
        hierarchical_free_energy_terms(z, z, z, z, z, z, z, z, model_reduction="median")
    with pytest.raises(TypeError):                                     # non-tensor input
        hierarchical_free_energy_terms(z, z, z, z, z, z, z, 0.0)


# ---------------------------------------------------------------------------
# 2. The belief-row specialization reproduces free_energy() numerically.
# ---------------------------------------------------------------------------

def _q_total(rows: BeliefFreeEnergyRows) -> torch.Tensor:
    zero = torch.zeros_like(rows.self_coupling)
    return hierarchical_free_energy_terms(
        rows.self_coupling, rows.belief_coupling, rows.attention_entropy,
        rows.twohop_coupling, zero, zero, zero, rows.observation_nll,
        q_reduction="sum", model_reduction="sum",
    ).total


@pytest.mark.parametrize("include_entropy", [True, False])
@pytest.mark.parametrize("with_ll", [False, True])
def test_q_row_specialization_matches_free_energy_numerically(include_entropy, with_ll):
    torch.manual_seed(1)
    N = 3
    self_div = torch.rand(N, dtype=torch.float64) + 0.1
    energy = torch.rand(N, N, dtype=torch.float64)
    alpha = torch.rand(N, dtype=torch.float64) + 0.5
    log_prior = torch.randn(N, N, dtype=torch.float64)
    ll = torch.rand(N, dtype=torch.float64) if with_ll else None
    tau, lambda_beta, lambda_twohop = 1.3, 0.7, 0.2

    F = free_energy(
        self_div, energy, alpha, tau=tau, lambda_beta=lambda_beta,
        lambda_twohop=lambda_twohop, include_attention_entropy=include_entropy,
        log_prior=log_prior, log_likelihood=ll,
    )
    rows = _belief_free_energy_rows(
        self_div, energy, alpha, tau=tau, lambda_beta=lambda_beta,
        lambda_twohop=lambda_twohop, include_attention_entropy=include_entropy,
        log_prior=log_prior, log_likelihood=ll,
    )
    assert isinstance(rows, BeliefFreeEnergyRows)
    assert rows.self_coupling.shape == (N,)
    assert torch.allclose(_q_total(rows), F, atol=1e-10, rtol=0.0)


def test_q_row_specialization_per_coord_self_div_with_head_energy():
    r"""Regression (full-suite catch): a PER-COORDINATE self divergence (N, K) meeting a PER-HEAD
    energy (H, N, N) -- the state_dependent_per_coord + block_glk diagnostics layout -- has the same
    rank gap as the plain unheaded case; the rows must still come out (N,) and sum to free_energy().
    N == K here so the shape disambiguation (not just K != N) is what resolves it."""
    torch.manual_seed(5)
    H, N, K = 2, 4, 4
    self_div = torch.rand(N, K, dtype=torch.float64) + 0.1            # per-coordinate D^(k)
    energy = torch.rand(H, N, N, dtype=torch.float64)                 # per-head E_ij^(h)
    alpha = torch.rand(N, K, dtype=torch.float64) + 0.5
    alpha_reg = torch.rand(N, K, dtype=torch.float64) + 0.2
    tau = 1.2

    F = free_energy(self_div, energy, alpha, tau=tau, lambda_beta=0.8, alpha_reg=alpha_reg)
    for hint in (None, True):                                         # inferred and explicit agree
        rows = _belief_free_energy_rows(self_div, energy, alpha, tau=tau, lambda_beta=0.8,
                                        alpha_reg=alpha_reg, per_coord=hint)
        assert rows.self_coupling.shape == (N,)
        assert rows.belief_coupling.shape == (N,)
        assert torch.allclose(_q_total(rows), F, atol=1e-10, rtol=0.0)


def test_q_row_specialization_batched_and_beta_override():
    torch.manual_seed(7)
    B, N = 2, 3
    self_div = torch.rand(B, N, dtype=torch.float64) + 0.1
    energy = torch.rand(B, N, N, dtype=torch.float64)
    alpha = torch.rand(B, N, dtype=torch.float64) + 0.5
    tau = 1.1
    F = free_energy(self_div, energy, alpha, tau=tau, lambda_beta=0.9)
    beta = attention_weights(energy, tau=tau)
    rows = _belief_free_energy_rows(self_div, energy, alpha, tau=tau,
                                    lambda_beta=0.9, beta_override=beta)
    assert rows.self_coupling.shape == (B, N)
    assert torch.allclose(_q_total(rows), F, atol=1e-10, rtol=0.0)


# ---------------------------------------------------------------------------
# 3. The legacy scalar keeps its exact float32 reduction order (values + grads).
# ---------------------------------------------------------------------------

def _legacy_scalar_oracle(self_div, energy, alpha, *, tau, lambda_beta, log_eps=1e-12,
                          lambda_twohop, include_attention_entropy, log_prior=None,
                          alpha_reg=None, coupling_energy=None, log_likelihood=None):
    r"""Verbatim copy of the pre-change ``free_energy`` body (one-shot ``.sum()`` reductions).

    Regrouping these sums row-then-query changes the float32 result, so ``torch.equal`` against the
    live ``free_energy`` is a real guard that the scalar path was not rewired to the row evaluator."""
    beta = attention_weights(energy, tau=tau, log_prior=log_prior)
    self_term = alpha * self_div
    if alpha_reg is not None:
        self_term = self_term + alpha_reg
    self_total = self_term.sum()
    coupling = (beta * (energy if coupling_energy is None else coupling_energy)).sum()
    F = self_total + lambda_beta * coupling
    if include_attention_entropy:
        if log_prior is not None:
            log_pi = torch.log_softmax(log_prior, dim=-1)
            log_pi = torch.where(torch.isfinite(log_pi), log_pi, torch.zeros_like(log_pi))
        else:
            log_pi = math.log(max(1.0 / beta.shape[-1], log_eps))
        _tau_e = _broadcast_tau(tau, energy)
        entropy = (_tau_e * (beta * (torch.log(beta.clamp(min=log_eps)) - log_pi))).sum()
        F = F + lambda_beta * entropy
    if lambda_twohop != 0.0:
        w2 = beta.detach() @ beta.detach()
        F = F + lambda_twohop * (w2 * (energy if coupling_energy is None else coupling_energy)).sum()
    if log_likelihood is not None:
        F = F - log_likelihood.sum()
    return F


def test_legacy_free_energy_reduction_order_is_bitwise_unchanged():
    torch.manual_seed(2)
    N = 6
    # A WIDE dynamic range so float32 sum regrouping is not associative: a row-then-query reduction
    # gives a different float32 value than the one-shot .sum(), making torch.equal a meaningful guard.
    scale = torch.tensor([1.0e3, 1.0e-2, 5.0, 1.0e-3, 2.0e2, 7.0], dtype=torch.float32)
    base_self = torch.rand(N, dtype=torch.float32) * scale
    base_energy = torch.rand(N, N, dtype=torch.float32) * scale
    base_alpha = torch.rand(N, dtype=torch.float32) + 0.5
    base_prior = torch.randn(N, N, dtype=torch.float32)
    base_ll = torch.rand(N, dtype=torch.float32) * scale

    def leaves():
        return (base_self.clone().requires_grad_(True),
                base_energy.clone().requires_grad_(True),
                base_alpha.clone().requires_grad_(True))

    kw = dict(tau=1.3, lambda_beta=0.9, lambda_twohop=0.4,
              include_attention_entropy=True, log_prior=base_prior,
              log_likelihood=base_ll)

    s1, e1, a1 = leaves()
    f_impl = free_energy(s1, e1, a1, **kw)
    s2, e2, a2 = leaves()
    f_oracle = _legacy_scalar_oracle(s2, e2, a2, **kw)

    assert torch.equal(f_impl, f_oracle), "free_energy value drifted from the pre-change oracle"

    f_impl.backward()
    f_oracle.backward()
    assert torch.equal(s1.grad, s2.grad)
    assert torch.equal(e1.grad, e2.grad)
    assert torch.equal(a1.grad, a2.grad)


# ---------------------------------------------------------------------------
# 4. model_reduction sum vs mean is explicit and independent of q_reduction.
# ---------------------------------------------------------------------------

def test_model_mean_vs_sum_reduction_is_explicit():
    N = 4
    hyper = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    zero = torch.zeros(N, dtype=torch.float64)

    t_sum = hierarchical_free_energy_terms(zero, zero, zero, zero, hyper, zero, zero, zero,
                                           q_reduction="sum", model_reduction="sum")
    t_mean = hierarchical_free_energy_terms(zero, zero, zero, zero, hyper, zero, zero, zero,
                                            q_reduction="sum", model_reduction="mean")
    assert torch.allclose(t_sum.hyper_prior, hyper.sum())
    assert torch.allclose(t_mean.hyper_prior, hyper.mean())
    assert torch.allclose(t_sum.total, hyper.sum())
    assert torch.allclose(t_mean.total, hyper.mean())
    assert not torch.allclose(t_sum.total, t_mean.total)              # N=4 -> sum != mean

    # q_reduction acts on the q rows independently of model_reduction.
    q = torch.tensor([2.0, 4.0, 6.0, 8.0], dtype=torch.float64)
    q_sum = hierarchical_free_energy_terms(q, zero, zero, zero, zero, zero, zero, zero,
                                           q_reduction="sum", model_reduction="mean")
    q_mean = hierarchical_free_energy_terms(q, zero, zero, zero, zero, zero, zero, zero,
                                            q_reduction="mean", model_reduction="mean")
    assert torch.allclose(q_sum.self_coupling, q.sum())
    assert torch.allclose(q_mean.self_coupling, q.mean())


# ---------------------------------------------------------------------------
# 5. The model-channel gamma energy is not hiddenly detached.
# ---------------------------------------------------------------------------

def test_evaluator_passes_model_gradient_through():
    N = 3
    x = torch.randn(N, dtype=torch.float64, requires_grad=True)
    zero = torch.zeros(N, dtype=torch.float64)
    terms = hierarchical_free_energy_terms(zero, zero, zero, zero, zero, x, zero, zero,
                                           q_reduction="sum", model_reduction="sum")
    terms.total.backward()
    assert x.grad is not None
    assert torch.allclose(x.grad, torch.ones(N, dtype=torch.float64))


def test_gamma_energy_gradient_is_not_hiddenly_detached():
    from vfe3.config import VFE3Config
    from vfe3.model.model import VFEModel

    torch.manual_seed(0)
    cfg = VFE3Config(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                     n_e_steps=1, lambda_h=0.0, lambda_gamma=0.5,
                     prior_source="model_channel")
    m = VFEModel(cfg)
    with torch.no_grad():
        m.prior_bank.s_mu_embed.normal_(0.0, 0.5)
        m.prior_bank.s_sigma_log_embed.normal_(0.0, 0.3)
    tok = torch.randint(0, 6, (1, 6))
    phi = m.prior_bank.encode(tok).phi.detach()                       # frame held fixed (loss detaches it)
    c_rows, me_rows = m._gamma_coupling_rows(tok, phi, head_reduction="mean")
    assert c_rows.shape == (1, 6) and me_rows.shape == (1, 6)
    (c_rows + me_rows).sum().backward()
    assert m.prior_bank.s_mu_embed.grad is not None
    assert float(m.prior_bank.s_mu_embed.grad.abs().sum()) > 0.0      # gamma energy reaches the s tables


# ---------------------------------------------------------------------------
# 6. The observation slot defaults to an exact zero.
# ---------------------------------------------------------------------------

def test_observation_slot_defaults_to_zero():
    N = 3
    self_div = torch.rand(N, dtype=torch.float64) + 0.1
    energy = torch.rand(N, N, dtype=torch.float64)
    alpha = torch.ones(N, dtype=torch.float64)

    rows = _belief_free_energy_rows(self_div, energy, alpha, tau=1.0)
    assert torch.equal(rows.observation_nll, torch.zeros_like(rows.self_coupling))

    ll = torch.rand(N, dtype=torch.float64)
    rows_ll = _belief_free_energy_rows(self_div, energy, alpha, tau=1.0, log_likelihood=ll)
    assert torch.allclose(rows_ll.observation_nll, -ll)               # observation_nll = -log p(o|x)

    zeros = [torch.zeros(N, dtype=torch.float64) for _ in range(8)]
    terms = hierarchical_free_energy_terms(*zeros)
    assert torch.equal(terms.observation_nll, torch.zeros((), dtype=torch.float64))
    assert torch.equal(terms.total, torch.zeros((), dtype=torch.float64))


def test_two_hop_weights_are_detached_but_rows_carry_gradient():
    r"""The only detach inside ``_belief_free_energy_rows`` is the fixed two-hop weight
    W2 = beta.detach() @ beta.detach(); the coupling row still carries gradient into the energy."""
    torch.manual_seed(4)
    N = 3
    self_div = torch.rand(N, dtype=torch.float64) + 0.1
    energy = (torch.rand(N, N, dtype=torch.float64)).requires_grad_(True)
    alpha = torch.ones(N, dtype=torch.float64)
    rows = _belief_free_energy_rows(self_div, energy, alpha, tau=1.0,
                                    lambda_beta=1.0, lambda_twohop=0.3)
    rows.twohop_coupling.sum().backward()
    assert energy.grad is not None and float(energy.grad.abs().sum()) > 0.0


# ---------------------------------------------------------------------------
# Compatibility wrapper: raw public fields preserved, total via the evaluator.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("include_entropy", [True, False])
def test_metrics_wrapper_keeps_raw_fields_and_routes_total(include_entropy):
    torch.manual_seed(3)
    N = 3
    self_div = torch.rand(N, dtype=torch.float64) + 0.1
    energy = torch.rand(N, N, dtype=torch.float64)
    beta = attention_weights(energy, tau=1.0)
    alpha = torch.ones(N, dtype=torch.float64)
    ll = torch.rand(N, dtype=torch.float64)
    lambda_beta, lambda_twohop = 0.25, 0.3

    terms = free_energy_terms(
        self_div, energy, beta, alpha, tau=1.0, lambda_beta=lambda_beta,
        lambda_twohop=lambda_twohop, include_attention_entropy=include_entropy,
        log_likelihood=ll,
    )

    # RAW public fields keep their current unweighted, positive-likelihood values.
    assert terms["belief_coupling"] == pytest.approx(float((beta * energy).sum()))
    w2 = beta.detach() @ beta.detach()
    assert terms["twohop_coupling"] == pytest.approx(float((w2 * energy).sum()))
    assert terms["observation_likelihood"] == pytest.approx(float(ll.sum()))
    log_pi = math.log(1.0 / N)
    raw_entropy = float((beta * (torch.log(beta.clamp(min=1e-12)) - log_pi)).sum())
    assert terms["attention_entropy"] == pytest.approx(raw_entropy)   # reported regardless of the gate

    # total alone applies weights, the entropy gate, and the negative-likelihood sign; it equals
    # the scalar free_energy for the same inputs.
    F = free_energy(
        self_div, energy, alpha, tau=1.0, lambda_beta=lambda_beta,
        lambda_twohop=lambda_twohop, include_attention_entropy=include_entropy,
        log_likelihood=ll,
    )
    assert terms["total"] == pytest.approx(float(F), abs=1e-10)


# ---------------------------------------------------------------------------
# Diagnostics + scored-outer-loss integration (H=1 and H=2).
# ---------------------------------------------------------------------------

def _active_model(n_heads, **kw):
    from vfe3.config import VFE3Config
    from vfe3.model.model import VFEModel

    base = dict(vocab_size=6, embed_dim=4, n_heads=n_heads, max_seq_len=8, n_layers=1,
                n_e_steps=2, s_e_step=True, prior_source="model_channel",
                lambda_h=0.25, lambda_gamma=0.75)
    base.update(kw)
    torch.manual_seed(0)
    m = VFEModel(VFE3Config(**base))
    torch.manual_seed(123)
    with torch.no_grad():
        m.prior_bank.s_mu_embed.normal_(0.0, 0.5)
        m.prior_bank.s_sigma_log_embed.normal_(0.0, 0.3)
        m.prior_bank.phi_embed.normal_(0.0, 0.2)
    return m


@pytest.mark.parametrize("n_heads", [1, 2])
def test_diagnostics_model_channel_total_is_finite(n_heads):
    m = _active_model(n_heads)
    tok = torch.randint(0, 6, (2, 8))
    d = m.diagnostics(tok)
    assert math.isfinite(d["total"])
    for k in ("hyper_prior", "gamma_coupling", "gamma_meta_entropy"):
        assert k in d and math.isfinite(d[k])
    # the reported blocks reconstruct total at ONE (sum) scale, as the F-decomposition figure expects.
    lb = m.cfg.lambda_beta
    recon = (d["self_coupling"] + lb * d["belief_coupling"] + lb * d["attention_entropy"]
             + d["hyper_prior_weighted"]
             + m.cfg.lambda_gamma * (d["gamma_coupling"] + d["gamma_meta_entropy"]))
    assert abs(d["total"] - recon) < 1e-3


@pytest.mark.parametrize("include_entropy", [True, False])
@pytest.mark.parametrize("n_heads", [1, 2])
def test_scored_outer_loss_matches_pre_change_assembly(include_entropy, n_heads):
    r"""The row+evaluator outer-loss assembly (model_reduction='mean') equals the pre-change
    ``_hyper_prior_weighted().mean() + lambda_gamma*_gamma_coupling_term`` in value and s/gamma
    gradients (no bitwise claim -- the canonical branch swaps the reduced envelope for the equal
    coupling-plus-entropy decomposition)."""
    m = _active_model(n_heads, s_e_step=False, include_attention_entropy=include_entropy)
    cfg = m.cfg
    tok = torch.randint(0, 6, (2, 8))
    belief = m.forward_beliefs(tok, return_logits=False)[0]
    model_phi = m._resolve_model_frame(tok, belief.phi).detach()
    tied = cfg.s_frame_mode == "tied"
    omega = belief.omega.detach() if tied and belief.omega is not None else None
    reflection = belief.reflection if tied else None

    pre = (m._hyper_prior_weighted(tok).mean()
           + cfg.lambda_gamma * m._gamma_coupling_term(tok, model_phi, omega=omega,
                                                        reflection=reflection))

    hp_rows = m._hyper_prior_weighted(tok)
    c_rows, me_rows = m._gamma_coupling_rows(tok, model_phi, head_reduction="mean",
                                             omega=omega, reflection=reflection)
    assert hp_rows.shape == c_rows.shape == me_rows.shape == (2, 8)
    mc = cfg.lambda_gamma * c_rows
    me = cfg.lambda_gamma * me_rows if include_entropy else torch.zeros_like(c_rows)
    z = torch.zeros_like(hp_rows)
    new = hierarchical_free_energy_terms(z, z, z, z, hp_rows, mc, me, z,
                                         q_reduction="sum", model_reduction="mean").total

    torch.testing.assert_close(new, pre, rtol=1e-4, atol=1e-6)

    params = [m.prior_bank.s_mu_embed, m.prior_bank.s_sigma_log_embed]
    g_pre = torch.autograd.grad(pre, params, retain_graph=True, allow_unused=True)
    g_new = torch.autograd.grad(new, params, allow_unused=True)
    for a, b in zip(g_pre, g_new):
        if a is None and b is None:
            continue
        assert a is not None and b is not None
        torch.testing.assert_close(b, a, rtol=2e-3, atol=1e-4)


# ===========================================================================
# PB-11 (Task 2, 2026-07-12): family-owned model-channel + full-SPD prior storage.
#
# Under family='gaussian_full' the model-channel s/r tables carry a packed strict-lower
# Cholesky (K*(K-1)//2) alongside the log-variance diagonal, so encode_s/r_parameters return
# a full (..., K, K) covariance; diagonal and Laplace channels create NO packed keys and stay
# byte-identical. _hyper_prior_kl dispatches KL(s||r) through the configured family, and
# barycenter_r_ moment-matches full Gaussians. Vocabulary-prior and decode variance tables stay
# diagonal in every family.
# ===========================================================================

from vfe3.families.covariance_tables import (                    # noqa: E402
    covariance_from_packed,
    packed_from_covariance,
    packed_strict_lower_size,
)
from vfe3.numerics import bounded_variance_from_log              # noqa: E402


def _pb11_model(**kw):
    from vfe3.config import VFE3Config
    from vfe3.model.model import VFEModel

    base = dict(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                n_e_steps=1, lambda_h=0.5)
    base.update(kw)
    torch.manual_seed(0)
    return VFEModel(VFE3Config(**base))


def _full_kw(**kw):
    base = dict(family="gaussian_full", decode_mode="full")
    base.update(kw)
    return base


def _param_names(model):
    return {name for name, _ in model.named_parameters()}


# --- helper-level: bounded-diagonal SPD assembly (K=3, log_diag=100) -------------------------

def test_covariance_from_packed_bounded_diagonal_stays_finite():
    log_diag = torch.full((3,), 100.0)
    packed = torch.zeros(packed_strict_lower_size(3))             # zero off-diagonal -> pure diagonal
    cov = covariance_from_packed(log_diag, packed)
    assert cov.shape == (3, 3)
    assert torch.isfinite(cov).all()                             # 100 is bounded BEFORE the sqrt
    diag = torch.diagonal(cov, dim1=-2, dim2=-1)
    assert torch.equal(diag, bounded_variance_from_log(log_diag))  # exact diagonal == bounded variance
    off = cov - torch.diag_embed(diag)
    assert torch.equal(off, torch.zeros_like(off))               # off-diagonal is exactly zero


def test_covariance_from_packed_is_spd_and_round_trips():
    torch.manual_seed(3)
    log_diag = torch.randn(2, 3) * 0.3
    packed = torch.randn(2, packed_strict_lower_size(3)) * 0.5
    cov = covariance_from_packed(log_diag, packed)
    assert cov.shape == (2, 3, 3)
    assert torch.allclose(cov, cov.transpose(-1, -2), atol=1e-6)  # symmetric
    eigs = torch.linalg.eigvalsh(cov)
    assert (eigs > 0).all()                                      # positive definite
    log_diag_rt, packed_rt = packed_from_covariance(cov)
    cov_rt = covariance_from_packed(log_diag_rt, packed_rt)
    torch.testing.assert_close(cov_rt, cov, rtol=1e-5, atol=1e-6)  # exact inverse


# --- pure path: diagonal / Laplace channels create NO packed keys ---------------------------

def test_diagonal_model_channel_has_no_packed_keys():
    m = _pb11_model()                                            # gaussian_diagonal (default) + lambda_h
    names = _param_names(m)
    assert "prior_bank.s_mu_embed" in names                      # the model channel IS built
    assert not any("sigma_lower" in n for n in names)           # ...but with no packed Cholesky keys
    s_mu, s_sigma = m.prior_bank.encode_s(torch.zeros(1, 4, dtype=torch.long))
    assert s_sigma.shape == (1, 4, m.cfg.embed_dim)             # (B, N, K) diagonal rank
    r_mu, r_sigma = m.prior_bank.r_parameters()
    assert r_sigma.shape == (m.cfg.embed_dim,)                  # (K,) diagonal rank


def test_laplace_model_channel_constructs_with_gradient_r_update():
    m = _pb11_model(**{"family": "laplace_diagonal", "r_update_mode": "gradient",
                       "prior_source": "model_channel"})
    names = _param_names(m)
    assert not any("sigma_lower" in n for n in names)           # Laplace (diagonal) creates no packed keys
    s_mu, s_sigma = m.prior_bank.encode_s(torch.zeros(1, 4, dtype=torch.long))
    assert s_sigma.shape == (1, 4, m.cfg.embed_dim)            # diagonal rank
    r_mu, r_sigma = m.prior_bank.r_parameters()
    assert r_sigma.shape == (m.cfg.embed_dim,)


def test_barycenter_rejected_for_family_without_registered_barycenter():
    from vfe3.config import VFE3Config
    with pytest.raises(ValueError, match="barycenter"):
        VFE3Config(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8,
                   family="laplace_diagonal", lambda_h=0.5, learnable_r=True,
                   prior_source="model_channel", r_update_mode="barycenter")


# --- full path: packed shapes, rank, SPD reconstruction -------------------------------------

def test_full_model_channel_packed_shapes_zero_init():
    m = _pb11_model(**_full_kw())
    pb = m.prior_bank
    V, K = m.cfg.vocab_size, m.cfg.embed_dim
    n_lower = packed_strict_lower_size(K)
    assert pb.s_sigma_lower_embed.shape == (V, n_lower)
    assert pb.r_sigma_lower.shape == (n_lower,)
    assert torch.equal(pb.s_sigma_lower_embed, torch.zeros(V, n_lower))   # zero-init -> diagonal at start
    assert torch.equal(pb.r_sigma_lower, torch.zeros(n_lower))


def test_full_encode_s_and_r_parameters_return_full_rank():
    m = _pb11_model(**_full_kw())
    pb = m.prior_bank
    K = m.cfg.embed_dim
    tok = torch.zeros(2, 4, dtype=torch.long)
    s_mu, s_sigma = pb.encode_s(tok)
    assert s_mu.shape == (2, 4, K)
    assert s_sigma.shape == (2, 4, K, K)                        # full covariance rank
    r_mu, r_sigma = pb.r_parameters()
    assert r_mu.shape == (K,)
    assert r_sigma.shape == (K, K)
    # zero-init packed lower -> the encoded covariance is exactly the diagonal bounded variance.
    diag = bounded_variance_from_log(pb.s_sigma_log_embed[tok], eps=m.cfg.eps)
    torch.testing.assert_close(s_sigma, torch.diag_embed(diag), rtol=0.0, atol=0.0)


def test_full_encode_s_covariance_is_spd_with_offdiagonal():
    m = _pb11_model(**_full_kw())
    pb = m.prior_bank
    with torch.no_grad():
        pb.s_sigma_lower_embed.normal_(0.0, 0.4)               # nonzero off-diagonal Cholesky
    tok = torch.arange(4, dtype=torch.long).reshape(1, 4)
    _, s_sigma = pb.encode_s(tok)
    eigs = torch.linalg.eigvalsh(s_sigma)
    assert (eigs > 0).all()                                     # SPD everywhere
    off = s_sigma - torch.diag_embed(torch.diagonal(s_sigma, dim1=-2, dim2=-1))
    assert float(off.abs().max()) > 0.0                        # genuinely off-diagonal


# --- vocabulary / decode tables stay diagonal in every family -------------------------------

def test_vocab_and_decode_tables_have_no_lower_triangle_keys():
    m = _pb11_model(**_full_kw(untie_decode_bank=True, use_prior_bank=True))
    names = _param_names(m)
    # Only the model-channel s/r tables carry packed lower keys; the vocabulary prior
    # (sigma_log_embed) and the untied decode variance table stay diagonal.
    lower_keys = {n for n in names if "sigma_lower" in n}
    assert lower_keys == {"prior_bank.s_sigma_lower_embed", "prior_bank.r_sigma_lower"}
    assert m.prior_bank.sigma_log_embed.shape == (m.cfg.vocab_size, m.cfg.embed_dim)
    assert m.prior_bank.decode_sigma_log_embed.shape == (m.cfg.vocab_size, m.cfg.embed_dim)


# --- gradients flow to the off-diagonal s/r Cholesky ----------------------------------------

def test_full_hyper_prior_kl_dispatches_full_family_and_grads_offdiagonal():
    m = _pb11_model(**_full_kw(learnable_r=True, r_update_mode="gradient",
                               prior_source="model_channel"))
    pb = m.prior_bank
    assert pb.r_sigma_lower.requires_grad                       # learnable_r un-freezes the packed centroid
    with torch.no_grad():
        pb.s_mu_embed.normal_(0.0, 0.5)
        pb.s_sigma_lower_embed.normal_(0.0, 0.3)               # nonzero -> off-diagonal KL gradient
        pb.r_sigma_lower.normal_(0.0, 0.3)
    tok = torch.arange(4, dtype=torch.long).reshape(1, 4)
    kl = m._hyper_prior_kl(tok)                                 # dispatched through get_family(gaussian_full)
    assert kl.shape == (1, 4)
    kl.sum().backward()
    assert pb.s_sigma_lower_embed.grad is not None
    assert float(pb.s_sigma_lower_embed.grad.abs().sum()) > 0.0
    assert pb.r_sigma_lower.grad is not None
    assert float(pb.r_sigma_lower.grad.abs().sum()) > 0.0


# --- full-Gaussian barycenter moment matching -----------------------------------------------

def test_full_barycenter_matches_full_gaussian_moments():
    m = _pb11_model(**_full_kw())
    pb = m.prior_bank
    torch.manual_seed(11)
    with torch.no_grad():
        pb.s_mu_embed.normal_(0.0, 0.7)
        pb.s_sigma_log_embed.normal_(0.0, 0.3)
        pb.s_sigma_lower_embed.normal_(0.0, 0.4)

    s_sigma = covariance_from_packed(pb.s_sigma_log_embed, pb.s_sigma_lower_embed,
                                     eps=m.cfg.eps)             # (V, K, K)
    r_mu_expected = pb.s_mu_embed.mean(dim=0)                   # (K,)
    centered = pb.s_mu_embed - r_mu_expected                    # (V, K)
    outer = centered.unsqueeze(-1) * centered.unsqueeze(-2)     # (V, K, K)
    r_sigma_expected = (s_sigma + outer).mean(dim=0)            # (K, K) within + between

    pb.barycenter_r_()
    torch.testing.assert_close(pb.r_mu, r_mu_expected, rtol=1e-5, atol=1e-6)
    r_sigma_actual = covariance_from_packed(pb.r_sigma_log, pb.r_sigma_lower, eps=m.cfg.eps)
    torch.testing.assert_close(r_sigma_actual, r_sigma_expected, rtol=1e-4, atol=1e-6)


# --- optimizer coverage: packed tables are grouped ------------------------------------------

def test_full_packed_tables_are_covered_by_the_optimizer():
    from vfe3.train import build_optimizer
    m = _pb11_model(**_full_kw(learnable_r=True, r_update_mode="gradient",
                               prior_source="model_channel"))
    opt = build_optimizer(m, m.cfg)                             # raises if any trainable param is ungrouped
    grouped = {p for g in opt.param_groups for p in g["params"]}
    assert m.prior_bank.s_sigma_lower_embed in grouped
    assert m.prior_bank.r_sigma_lower in grouped


# ===========================================================================
# PB-11 (Task 3, 2026-07-12): model-channel family + nonflat transport parity.
#
# _gamma_energy now dispatches through get_family(cfg.family) and builds the
# s-channel transport through the SAME connection-regime registry the belief
# channel uses, sharing the active connection_W/M/L and gating the belief tensors
# it feeds the builder on the transport-registration metadata (needs_mu/needs_sigma)
# rather than a mode-name conditional. The stateful regime_ii/covariant s-channel
# transport reads the s-channel means/covariances (channel-local), not the belief q.
# ===========================================================================

from vfe3.families.base import get_family as _get_family                        # noqa: E402
from vfe3.free_energy import pairwise_energy as _pairwise_energy                 # noqa: E402
from vfe3.geometry.transport import (                                           # noqa: E402
    _TRANSPORT_NEEDS_MU,
    _TRANSPORT_NEEDS_SIGMA,
    transport_covariance as _transport_covariance,
    transport_mean as _transport_mean,
)
from vfe3.inference.e_step import build_belief_transport as _build_belief_transport  # noqa: E402


def _gamma_model(**kw):
    from vfe3.config import VFE3Config
    from vfe3.model.model import VFEModel

    # prior_source='token' (default) keeps the belief mu_embed table SEPARATE from the s tables, so the
    # channel-local "transport reads the s state, not the belief q state" assertions are meaningful
    # (model_channel would tie encode() to encode_s and collapse the distinction). lambda_gamma>0 still
    # creates the s tables the gamma model coupling reads.
    base = dict(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                n_e_steps=1, lambda_h=0.0, lambda_gamma=0.5)
    base.update(kw)
    torch.manual_seed(0)
    m = VFEModel(VFE3Config(**base))
    torch.manual_seed(11)
    with torch.no_grad():                                       # non-degenerate s / belief / frame tables
        m.prior_bank.s_mu_embed.normal_(0.0, 0.5)
        m.prior_bank.s_sigma_log_embed.normal_(0.0, 0.3)
        m.prior_bank.mu_embed.normal_(0.0, 0.5)
        m.prior_bank.phi_embed.normal_(0.0, 0.2)
    return m


def _transport_from_state(m, phi, *, mu_state, sigma_state):
    r"""Build the s-channel transport exactly as the family/registry-driven ``_gamma_energy`` must:
    forward the active connections and gate the state tensors on the transport-registration metadata."""
    cfg = m.cfg
    tm = cfg.transport_mode
    return _build_belief_transport(
        phi, m.group, transport_mode=tm, gauge_parameterization="phi",
        mu=(mu_state if tm in _TRANSPORT_NEEDS_MU else None),
        sigma=(sigma_state if tm in _TRANSPORT_NEEDS_SIGMA else None),
        connection_W=getattr(m, "connection_W", None),
        connection_M=getattr(m, "connection_M", None),
        connection_L=getattr(m, "connection_L", None),
        link_alpha=cfg.link_alpha, link_soft_cap=cfg.link_soft_cap,
        cocycle_relaxation=cfg.cocycle_relaxation,
    )


def _gamma_e_s_reference(m, tok, phi, *, mu_state, sigma_state):
    r"""The configured-family s-channel pairwise energy under a transport built from ``mu_state`` /
    ``sigma_state`` -- the golden the metadata-driven ``_gamma_energy`` must reproduce."""
    cfg = m.cfg
    fam = _get_family(cfg.family)
    s_mu, s_sigma = m.prior_bank.encode_s(tok)
    omega = _transport_from_state(m, phi, mu_state=mu_state, sigma_state=sigma_state)
    s_mu_t = _transport_mean(omega, s_mu)
    s_sigma_t = _transport_covariance(omega, s_sigma, diagonal_out=(s_sigma.dim() == s_mu.dim()))
    return _pairwise_energy(fam(s_mu, s_sigma), fam(s_mu_t, s_sigma_t),
                            alpha=cfg.renyi_order, kl_max=cfg.kl_max, eps=cfg.eps,
                            divergence_family=cfg.divergence_family,
                            irrep_dims=m.group.irrep_dims)


def _belief_channel_energy(m, tok, phi):
    r"""The belief (q) channel pairwise energy under the SAME active connection, transport fed the
    belief means/covariances -- so it responds to the shared connection independently of the s path."""
    cfg = m.cfg
    fam = _get_family(cfg.family)
    enc = m.prior_bank.encode(tok)
    omega = _transport_from_state(m, phi, mu_state=enc.mu, sigma_state=enc.sigma)
    q_mu_t = _transport_mean(omega, enc.mu)
    q_sigma_t = _transport_covariance(omega, enc.sigma, diagonal_out=(enc.sigma.dim() == enc.mu.dim()))
    return _pairwise_energy(fam(enc.mu, enc.sigma), fam(q_mu_t, q_sigma_t),
                            alpha=cfg.renyi_order, kl_max=cfg.kl_max, eps=cfg.eps,
                            divergence_family=cfg.divergence_family,
                            irrep_dims=m.group.irrep_dims)


# --- family parity: _gamma_energy dispatches the configured family --------------------------

@pytest.mark.parametrize("family,extra,cov_rank", [
    ("gaussian_diagonal", {}, 3),
    ("gaussian_full", {"decode_mode": "full"}, 4),
    ("laplace_diagonal", {"r_update_mode": "gradient"}, 3),
])
def test_gamma_energy_dispatches_configured_family(family, extra, cov_rank):
    m = _gamma_model(family=family, **extra)
    tok = torch.randint(0, 6, (2, 5))
    phi = m.prior_bank.encode(tok).phi
    e_s, tau, log_prior = m._gamma_energy(tok, phi)             # gaussian_full: crashed pre-change
    assert torch.isfinite(e_s).all()
    _, s_sigma = m.prior_bank.encode_s(tok)
    assert s_sigma.dim() == cov_rank                            # covariance rank follows the family
    # e_s is the configured-family energy, not a hardcoded DiagonalGaussian one.
    ref = _gamma_e_s_reference(m, tok, phi, mu_state=None, sigma_state=None)
    torch.testing.assert_close(e_s, ref, rtol=1e-4, atol=1e-5)


def test_hyper_prior_kl_dispatches_configured_family_full_rank():
    m = _gamma_model(family="gaussian_full", decode_mode="full", lambda_h=0.5)
    tok = torch.randint(0, 6, (2, 5))
    kl = m._hyper_prior_kl(tok)
    assert kl.shape == (2, 5) and torch.isfinite(kl).all()
    _, s_sigma = m.prior_bank.encode_s(tok)
    assert s_sigma.dim() == 4                                   # full-rank s covariance scored


# --- nonflat parity: both q and s energies respond to the shared connection -----------------

def test_gamma_energy_shares_connection_W_both_channels_change():
    m = _gamma_model(transport_mode="regime_ii")
    with torch.no_grad():
        m.connection_W.normal_(0.0, 0.5)
    tok = torch.randint(0, 6, (2, 5))
    phi = m.prior_bank.encode(tok).phi
    e_s0 = m._gamma_energy(tok, phi)[0].clone()
    q_e0 = _belief_channel_energy(m, tok, phi).clone()
    with torch.no_grad():
        m.connection_W.mul_(1.7)                               # perturb ONLY the shared connection
    e_s1 = m._gamma_energy(tok, phi)[0]
    q_e1 = _belief_channel_energy(m, tok, phi)
    assert not torch.allclose(e_s0, e_s1)                      # s-channel now reads connection_W (was flat)
    assert not torch.allclose(q_e0, q_e1)                      # belief channel reads the same connection


def test_gamma_energy_regime_ii_transport_reads_s_state_not_q_state():
    m = _gamma_model(transport_mode="regime_ii")
    with torch.no_grad():
        m.connection_W.normal_(0.0, 0.6)
    tok = torch.randint(0, 6, (2, 5))
    phi = m.prior_bank.encode(tok).phi
    e_s = m._gamma_energy(tok, phi)[0]
    s_mu, s_sigma = m.prior_bank.encode_s(tok)
    enc = m.prior_bank.encode(tok)
    assert not torch.allclose(s_mu, enc.mu)                    # s and belief means genuinely differ
    ref_s = _gamma_e_s_reference(m, tok, phi, mu_state=s_mu, sigma_state=s_sigma)
    torch.testing.assert_close(e_s, ref_s, rtol=1e-4, atol=1e-5)   # transport reads the s-channel state
    ref_q = _gamma_e_s_reference(m, tok, phi, mu_state=enc.mu, sigma_state=enc.sigma)
    assert not torch.allclose(e_s, ref_q)                     # ...not the belief-channel state


def test_gamma_energy_gradient_reaches_connection_W():
    m = _gamma_model(transport_mode="regime_ii")
    with torch.no_grad():
        m.connection_W.normal_(0.0, 0.4)
    tok = torch.randint(0, 6, (2, 5))
    phi = m.prior_bank.encode(tok).phi
    e_s = m._gamma_energy(tok, phi)[0]
    e_s.sum().backward()
    assert m.connection_W.grad is not None
    assert float(m.connection_W.grad.abs().sum()) > 0.0       # shared connection is live in the s block


# --- config: a valid nonflat model channel constructs without the flat-island warning -------

def test_nonflat_model_channel_constructs_without_flat_island_warning():
    import warnings

    from vfe3.config import VFE3Config

    for tm in ("regime_ii", "regime_ii_covariant", "regime_ii_link_charted"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            VFE3Config(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                       transport_mode=tm, lambda_gamma=0.5, prior_source="model_channel")
        flat_island = [w for w in caught
                       if "FLAT phi-cocycle" in str(w.message) or "no non-flat transport law" in str(w.message)]
        assert not flat_island, f"{tm} still warns the model channel is a flat island: {[str(w.message) for w in flat_island]}"


# ===========================================================================
# PB-13 (Task 4, 2026-07-12): probabilistic CG moment closure + q-only moment
# regularizer. The CG coupling now exposes an exact analytic Jacobian and a
# delta-method covariance pushforward; cg_energy_weight>0 adds
# cg_energy_weight * mean_layers(mean_tokens(D(q_post||q_pre))) to the outer
# objective EXACTLY ONCE, leaving the canonical q/p/s/h hierarchy total untouched.
# ===========================================================================

import warnings                                                              # noqa: E402


def _cg_cfg(grad_mode="unroll", cg_energy_weight=0.0, **kw):
    from vfe3.config import VFE3Config
    base = dict(vocab_size=8, embed_dim=4, n_heads=2, max_seq_len=6, n_layers=1,
                n_e_steps=1, e_q_mu_lr=0.05, e_phi_lr=0.0,
                gauge_group="so_n", group_n=3, irrep_spec=[("l0", 1), ("l1", 1)],
                phi_precond_mode="none", use_cg_coupling=True,
                cg_energy_weight=cg_energy_weight, e_step_gradient=grad_mode)
    base.update(kw)
    return VFE3Config(**base)


def _cg_model(grad_mode="unroll", cg_energy_weight=0.0, weight_shift=0.3, **kw):
    from vfe3.model.model import VFEModel
    torch.manual_seed(0)
    with warnings.catch_warnings():                              # use_cg_coupling + detach warns; not under test here
        warnings.simplefilter("ignore")
        m = VFEModel(_cg_cfg(grad_mode, cg_energy_weight, **kw))
    torch.manual_seed(5)
    with torch.no_grad():                                        # non-degenerate beliefs -> a sizable CG delta
        m.prior_bank.mu_embed.normal_(0.0, 0.6)
        m.prior_bank.phi_embed.normal_(0.0, 0.2)
        if weight_shift is not None:
            m.cg_coupling.path_weights.add_(weight_shift)        # nonzero CG paths
    return m


def _cg_tokens():
    torch.manual_seed(1)
    return torch.randint(0, 8, (2, 6)), torch.randint(0, 8, (2, 6))


# --- the moment-energy rows equal a direct active-family divergence oracle ------------------

def test_cg_moment_energy_rows_match_divergence_oracle():
    from vfe3.model.cg_coupling import CGCoupling, cg_moment_energy_rows
    from vfe3.families.base import get_family
    from vfe3.free_energy import self_divergence
    from vfe3.geometry.groups import get_group

    grp = get_group("so_n")(4, group_n=3, irrep_spec=[("l0", 1), ("l1", 1)], dtype=torch.float64)
    cpl = CGCoupling(3, "so", grp.irrep_dims, grp.irrep_labels).double()
    with torch.no_grad():
        cpl.path_weights.copy_(0.3 * torch.randn(cpl.path_weights.shape[0], dtype=torch.float64))
    mu = torch.randn(2, 5, 4, dtype=torch.float64)
    sigma = torch.rand(2, 5, 4, dtype=torch.float64) + 0.1               # diagonal
    res = cpl.forward_moments(mu, sigma)                                 # passthrough -> post_sigma == sigma
    rows = cg_moment_energy_rows(mu, sigma, res.mu, res.sigma,
                                 renyi_order=1.0, family="gaussian_diagonal",
                                 divergence_family="renyi")
    fam = get_family("gaussian_diagonal")
    oracle = self_divergence(fam(res.mu, res.sigma), fam(mu, sigma),
                             alpha=1.0, divergence_family="renyi")
    assert rows.shape == (2, 5)
    assert torch.allclose(rows, oracle, atol=1e-12, rtol=0.0)


# --- the outer loss and path_weights gradient change with the energy, both estimators -------

@pytest.mark.parametrize("grad_mode", ["unroll", "detach"])
def test_cg_energy_changes_loss_and_path_weight_grad(grad_mode):
    tok, tgt = _cg_tokens()
    m_w = _cg_model(grad_mode, cg_energy_weight=0.5)
    m_0 = _cg_model(grad_mode, cg_energy_weight=0.0)                     # same seed + same weight_shift
    _, loss_w, _ = m_w(tok, tgt)
    _, loss_0, _ = m_0(tok, tgt)
    assert not torch.allclose(loss_w, loss_0)                           # energy term moved the objective
    loss_w.backward()
    loss_0.backward()
    g_w = m_w.cg_coupling.path_weights.grad
    assert g_w is not None and torch.isfinite(g_w).all() and float(g_w.abs().sum()) > 0.0
    g_0 = m_0.cg_coupling.path_weights.grad
    if grad_mode == "detach":
        # without the energy, detach wraps the CG mean coupling in no_grad -> path_weights frozen.
        assert g_0 is None or float(g_0.abs().sum()) == 0.0
    else:
        assert g_0 is not None and not torch.allclose(g_w, g_0)


def test_cg_energy_is_added_exactly_once():
    r"""loss(weight>0) - weight * cg_moment_energy == loss(weight=0): the regularizer enters once."""
    tok, tgt = _cg_tokens()
    m_w = _cg_model("unroll", cg_energy_weight=0.5)
    m_0 = _cg_model("unroll", cg_energy_weight=0.0)
    _, loss_w, _ = m_w(tok, tgt)
    _, loss_0, _ = m_0(tok, tgt)
    diag = m_w._cg_energy_diagnostics
    recovered = loss_w - m_w.cfg.cg_energy_weight * diag["cg_moment_energy"]
    assert torch.allclose(recovered, loss_0, atol=1e-6)
    assert diag["objective_total_with_cg"] == pytest.approx(float(loss_w.detach()))


def test_cg_energy_weight_zero_is_exactly_pre_change_loss():
    r"""cg_energy_weight=0 leaves the objective byte-identical across two identical builds AND never
    populates the diagnostic side channel."""
    tok, tgt = _cg_tokens()
    m_a = _cg_model("unroll", cg_energy_weight=0.0)
    m_b = _cg_model("unroll", cg_energy_weight=0.0)
    _, la, _ = m_a(tok, tgt)
    _, lb, _ = m_b(tok, tgt)
    assert torch.equal(la, lb)
    assert not hasattr(m_a, "_cg_energy_diagnostics")


# --- an M-step-only capture never reads or stacks a CG list ---------------------------------

def test_mstep_capture_without_cg_energy_never_touches_cg_lists():
    from vfe3.model.model import VFEModel
    torch.manual_seed(0)
    cfg = _cg_cfg("unroll", cg_energy_weight=0.0, use_cg_coupling=False,
                  mstep_self_coupling_weight=0.5)
    m = VFEModel(cfg)
    tok, tgt = _cg_tokens()
    _, loss, _ = m(tok, tgt)                                            # must not raise (no empty stack / missing key)
    assert torch.isfinite(loss)
    loss.backward()
    assert not hasattr(m, "_cg_energy_diagnostics")


# --- the q-only regularizer cannot reweight the independent hyper-prior (h/s) block ---------

def test_cg_energy_cannot_reweight_hyper_prior_block():
    tok, tgt = _cg_tokens()
    kw = dict(prior_source="token", s_e_step=False, lambda_h=0.25)
    m_0 = _cg_model("unroll", cg_energy_weight=0.0, **kw)
    m_1 = _cg_model("unroll", cg_energy_weight=0.5, **kw)
    torch.manual_seed(7)
    for m in (m_0, m_1):
        with torch.no_grad():                                          # identical non-degenerate s/r tables
            torch.manual_seed(7)
            m.prior_bank.s_mu_embed.normal_(0.0, 0.5)
            torch.manual_seed(8)
            m.prior_bank.s_sigma_log_embed.normal_(0.0, 0.3)
    _, l0, _ = m_0(tok, tgt)
    _, l1, _ = m_1(tok, tgt)
    l0.backward()
    l1.backward()
    # the CG moment energy is q-only: it can touch NEITHER the s tables NOR the global r centroid.
    assert torch.equal(m_0.prior_bank.s_mu_embed.grad, m_1.prior_bank.s_mu_embed.grad)
    assert torch.equal(m_0.prior_bank.s_sigma_log_embed.grad, m_1.prior_bank.s_sigma_log_embed.grad)


# --- two layers: ordered per-layer captures and the token-then-layer mean -------------------

def test_cg_energy_two_layers_ordered_capture_and_layer_mean(monkeypatch):
    m = _cg_model("unroll", cg_energy_weight=0.5, n_layers=2)
    tok, tgt = _cg_tokens()

    seen = {"n": 0}

    def spy(pre_mu, pre_sigma, post_mu, post_sigma, **kw):
        val = 1.0 if seen["n"] == 0 else 3.0                            # layer-indexed constant rows
        seen["n"] += 1
        return pre_mu.new_full(pre_mu.shape[:-1], val)                  # (..., N)

    monkeypatch.setattr("vfe3.model.block.cg_moment_energy_rows", spy)
    _, loss, _ = m(tok, tgt)
    diag = m._cg_energy_diagnostics
    assert seen["n"] == 2                                               # both applications ran, none omitted
    assert len(diag["cg_moment_energy_layers"]) == 2
    assert diag["cg_moment_energy_layers"][0] == pytest.approx(1.0)     # ordered
    assert diag["cg_moment_energy_layers"][1] == pytest.approx(3.0)
    assert diag["cg_moment_energy"] == pytest.approx(2.0)               # mean_layers(mean_tokens) = mean(1, 3)


# --- the detach freeze warning is accurate under the energy re-evaluation -------------------

def test_detach_freeze_warning_gated_off_when_cg_energy_trains_path_weights():
    r"""Under detach, cg_energy_weight>0 trains path_weights through the post-stack enable_grad
    re-evaluation, so the 'frozen at zero init' warning would be false -- it must NOT fire. At
    weight 0 the CG module genuinely freezes and the warning stays (pinned separately by
    test_audit_fixes_2026_06_10::test_detach_with_mixer_or_cg_warns)."""
    from vfe3.model.model import VFEModel
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        VFEModel(_cg_cfg("detach", cg_energy_weight=0.5))
    assert not any("freezes mixer_deltas" in str(w.message) for w in rec)
    with warnings.catch_warnings(record=True) as rec0:
        warnings.simplefilter("always")
        VFEModel(_cg_cfg("detach", cg_energy_weight=0.0))
    assert any("freezes mixer_deltas" in str(w.message) for w in rec0)


# --- construction guards --------------------------------------------------------------------

def test_cg_energy_weight_requires_coupling():
    with pytest.raises(ValueError, match="use_cg_coupling"):
        _cg_cfg("unroll", cg_energy_weight=0.5, use_cg_coupling=False)


def test_negative_or_nonfinite_cg_energy_weight_rejected():
    with pytest.raises(ValueError, match="cg_energy_weight"):
        _cg_cfg("unroll", cg_energy_weight=-1.0)
    with pytest.raises(ValueError, match="cg_energy_weight"):
        _cg_cfg("unroll", cg_energy_weight=float("nan"))


def test_delta_full_requires_gaussian_full():
    with pytest.raises(ValueError, match="delta_full"):
        _cg_cfg("unroll", cg_covariance_mode="delta_full", family="gaussian_diagonal")


def test_cg_covariance_mode_validated():
    with pytest.raises(ValueError, match="cg_covariance_mode"):
        _cg_cfg("unroll", cg_covariance_mode="bogus")


# ===========================================================================
# PB-14 (Task 5, 2026-07-12): family/divergence-consistent chunked decode parity.
#
# `family_chunked` streams the vocabulary through the SAME registered functional and the
# existing fused log-sum-exp/gather reduction, materializing only a per-chunk workspace inside
# a gradient checkpoint (for a full family, (B,N,Vc,K,K)). Its CE and gradients (to mu_q, the
# diagonal prior variances, the untied decode tables, and with the unigram bias active) must
# equal the dense `family` decode -> cross-entropy for every supported family.
# ===========================================================================

import torch.nn.functional as F                                        # noqa: E402
from vfe3.model.prior_bank import PriorBank, get_decode                # noqa: E402


def _pb14_chunked_bank(family, order, *, untie, V=7, K=3, n_gen=4):
    torch.manual_seed(0)
    pb = PriorBank(V, K, n_gen, decode_tau=1.2, family=family,
                   divergence_family="renyi", renyi_order=order, decode_mode="family_chunked",
                   diagonal_covariance=(family != "gaussian_full"),
                   untie_decode_bank=untie, decode_unigram_prior=True, decode_chunk_size=8192)
    with torch.no_grad():
        pb.mu_embed.normal_(0.0, 0.5)
        pb.sigma_log_embed.normal_(0.0, 0.3)
        if untie:
            pb.decode_mu_embed.normal_(0.0, 0.5)
            pb.decode_sigma_log_embed.normal_(0.0, 0.3)
    pb.set_unigram_log_prior(torch.arange(1, V + 1, dtype=torch.float32))   # nonzero unigram bias
    return pb


@pytest.mark.parametrize("untie", [False, True])
@pytest.mark.parametrize("family", ["gaussian_diagonal", "gaussian_full", "laplace_diagonal"])
@pytest.mark.parametrize("order", [1.0, 0.5])
def test_family_chunked_matches_dense_value_and_grads(family, order, untie):
    V, K, B, N = 7, 3, 2, 4
    pb = _pb14_chunked_bank(family, order, untie=untie, V=V, K=K)
    g = torch.Generator().manual_seed(1)
    mu_q = torch.randn(B, N, K, generator=g, requires_grad=True)
    if family == "gaussian_full":
        A = torch.randn(B, N, K, K, generator=g)
        sigma_q = A @ A.transpose(-1, -2) + K * torch.eye(K)
    else:
        sigma_q = torch.rand(B, N, K, generator=g) + 0.3
    targets = torch.randint(0, V, (B, N), generator=g)
    targets[0, 0] = -100                                               # ignore_index honored identically

    dense_logits = pb.decode(mu_q, sigma_q)                            # (B,N,V) inference == _decode_family + unigram
    dense_ce = F.cross_entropy(dense_logits.reshape(-1, V), targets.reshape(-1), ignore_index=-100)
    for chunk in (3, V, 8192):                                         # divides / single-window / >= V
        chunked_ce = pb.decode_ce_family_chunked(mu_q, sigma_q, targets, chunk_size=chunk)
        assert torch.allclose(chunked_ce, dense_ce, atol=1e-4), f"{family} order={order} chunk={chunk}"

    leaves = ([mu_q, pb.decode_mu_embed, pb.decode_sigma_log_embed] if untie
              else [mu_q, pb.mu_embed, pb.sigma_log_embed])
    chunked_ce = pb.decode_ce_family_chunked(mu_q, sigma_q, targets, chunk_size=3)
    g_dense = torch.autograd.grad(dense_ce, leaves, retain_graph=True, allow_unused=True)
    g_chunk = torch.autograd.grad(chunked_ce, leaves, allow_unused=True)
    for a, b in zip(g_dense, g_chunk):
        if a is None and b is None:
            continue
        assert a is not None and b is not None
        assert torch.allclose(a, b, atol=1e-3, rtol=0.0)


def test_family_chunked_all_ignore_is_finite_zero():
    pb = _pb14_chunked_bank("gaussian_diagonal", 1.0, untie=False)
    mu_q = torch.randn(2, 4, 3, requires_grad=True)
    sigma_q = torch.rand(2, 4, 3) + 0.3
    targets = torch.full((2, 4), -100)
    ce = pb.decode_ce_family_chunked(mu_q, sigma_q, targets, chunk_size=3)
    assert torch.isfinite(ce) and ce.item() == 0.0
    ce.backward()                                                      # grad-connected (no autograd error)


# ===========================================================================
# Task 6 (2026-07-12): end-to-end CPU matrix over the completed hierarchy.
#
# One forward/backward per family/transport/decode cell, each V<=9 / K<6: finite loss and
# gradients, optimizer coverage (build_optimizer asserts its groups cover model.parameters()
# exactly), a state-dict save/load round trip that reproduces the forward, and typed-term
# equality -- the reported per-token hierarchy blocks reconstruct diagnostics['total'] through
# the single typed evaluator (PB-10). The unit-level shape/gradient tests above already pin the
# packed storage, family dispatch, nonflat parity, and CG closure internals; this matrix is the
# integrated cell coverage on top of them.
# ===========================================================================

import os                                                              # noqa: E402


_MATRIX_CELLS = {
    # canonical diagonal / flat -- the pure route (no model channel, flat transport).
    "canonical_diagonal_flat": dict(
        vocab_size=9, embed_dim=4, n_heads=2, max_seq_len=6, n_layers=1, n_e_steps=2,
        family="gaussian_diagonal", transport_mode="flat"),
    # full-SPD model channel -- packed strict-lower Cholesky s/r tables (PB-11), gradient centroid.
    "full_spd_model_channel": dict(
        vocab_size=9, embed_dim=4, n_heads=2, max_seq_len=6, n_layers=1, n_e_steps=2,
        family="gaussian_full", decode_mode="full", prior_source="model_channel",
        lambda_h=0.5, lambda_gamma=0.5, learnable_r=True, r_update_mode="gradient"),
    # Laplace model channel decoded through the family-consistent KL-to-prior readout (PB-14).
    "laplace_family_decode": dict(
        vocab_size=9, embed_dim=4, n_heads=2, max_seq_len=6, n_layers=1, n_e_steps=2,
        family="laplace_diagonal", prior_source="model_channel", r_update_mode="gradient",
        use_prior_bank=True, decode_mode="family", lambda_h=0.5),
    # nonflat full model channel -- the covariant Regime-II connection shared by both channels (PB-11).
    "nonflat_full_model_channel": dict(
        vocab_size=9, embed_dim=4, n_heads=2, max_seq_len=6, n_layers=1, n_e_steps=2,
        family="gaussian_full", decode_mode="full", prior_source="model_channel",
        transport_mode="regime_ii_covariant", lambda_gamma=0.5, oracle_unroll_grad=True),
    # full-CG moment closure -- delta_full covariance pushforward + moment-energy regularizer (PB-13).
    "full_cg_moment_closure": dict(
        vocab_size=9, embed_dim=4, n_heads=2, max_seq_len=6, n_layers=1, n_e_steps=2,
        family="gaussian_full", decode_mode="full",
        gauge_group="so_n", group_n=3, irrep_spec=[("l0", 1), ("l1", 1)],
        phi_precond_mode="none", use_cg_coupling=True, cg_energy_weight=0.5,
        cg_covariance_mode="delta_full", e_step_gradient="unroll"),
}


def _build_matrix_model(kw):
    r"""Build a matrix-cell model and perturb every trainable table off its (often zero) init so the
    hierarchy energy, transport, and CG paths are all genuinely exercised. Warnings (the benign
    full-covariance linear-decode note, the regime_ii oracle auto-enable) are silenced here."""
    from vfe3.config import VFE3Config
    from vfe3.model.model import VFEModel

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.manual_seed(0)
        cfg = VFE3Config(**kw)
        m = VFEModel(cfg)
    torch.manual_seed(3)
    with torch.no_grad():
        for p in m.parameters():
            if p.requires_grad and p.dim() >= 1:
                p.normal_(0.0, 0.3)
    return m, cfg


def _reconstruct_total(d, cfg):
    r"""Reassemble diagnostics['total'] from the reported (raw) hierarchy blocks, applying the same
    weights and gates the typed evaluator uses inside ``diagnostics``: lambda_beta on the belief
    coupling and (gated) attention entropy, lambda_twohop on the two-hop block, the pre-weighted
    hyper-prior contribution, and lambda_gamma on the gamma coupling + meta-entropy."""
    lb = cfg.lambda_beta
    recon = d["self_coupling"] + lb * d["belief_coupling"]
    if cfg.include_attention_entropy:
        recon += lb * d["attention_entropy"]
    if cfg.lambda_twohop != 0.0:
        recon += cfg.lambda_twohop * d.get("twohop_coupling", 0.0)
    recon += d.get("hyper_prior_weighted", 0.0)
    recon += cfg.lambda_gamma * (d.get("gamma_coupling", 0.0) + d.get("gamma_meta_entropy", 0.0))
    return recon


@pytest.mark.parametrize("cell", sorted(_MATRIX_CELLS))
def test_end_to_end_hierarchy_matrix(cell):
    kw = _MATRIX_CELLS[cell]
    m, cfg = _build_matrix_model(kw)
    torch.manual_seed(1)
    tok = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    tgt = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))

    # one forward / backward
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loss = m(tok, tgt)[1]
    assert torch.isfinite(loss), f"{cell}: non-finite loss"
    loss.backward()

    # finite gradients (at least one live parameter, all finite)
    grads = [(name, p.grad) for name, p in m.named_parameters() if p.grad is not None]
    assert grads, f"{cell}: no parameter received a gradient"
    for name, g in grads:
        assert torch.isfinite(g).all(), f"{cell}: non-finite gradient in {name}"

    # optimizer coverage: build_optimizer raises if any trainable parameter is ungrouped.
    from vfe3.train import build_optimizer
    build_optimizer(m, cfg)

    # save/load round trip: a fresh model reloaded from the state dict reproduces the forward loss.
    from vfe3.model.model import VFEModel
    sd = m.state_dict()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.manual_seed(0)
        fresh = VFEModel(cfg)
        fresh.load_state_dict(sd)
        with torch.no_grad():
            loss_fresh = fresh(tok, tgt)[1]
    assert torch.allclose(loss.detach(), loss_fresh, atol=1e-6, rtol=0.0), \
        f"{cell}: state-dict reload changed the forward loss"

    # typed-term equality: the reported blocks reconstruct the typed hierarchy total.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = m.diagnostics(tok)
    assert math.isfinite(d["total"]), f"{cell}: non-finite diagnostics total"
    recon = _reconstruct_total(d, cfg)
    assert abs(d["total"] - recon) < 1e-3, \
        f"{cell}: typed total {d['total']} != reconstruction {recon}"


# ---------------------------------------------------------------------------
# RTX 5090 CUDA smoke (Task 6 Step 5). Guarded by VFE3_TEST_DEVICE; SKIPS cleanly on CPU and
# awaits an explicit GPU run. Exercises the full-covariance model channel under the live s E-step
# and the covariant Regime-II connection -- the heaviest completed-hierarchy path.
# ---------------------------------------------------------------------------

_DEVICE = torch.device(os.environ.get("VFE3_TEST_DEVICE", "cpu"))


@pytest.mark.skipif(_DEVICE.type != "cuda",
                    reason="RTX 5090 CUDA smoke; set VFE3_TEST_DEVICE=cuda to run")
def test_hierarchy_full_covariant_cuda_smoke():
    from vfe3.config import VFE3Config
    from vfe3.model.model import VFEModel
    from vfe3.families.covariance_tables import covariance_from_packed
    from vfe3.train import build_optimizer

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.manual_seed(0)
        cfg = VFE3Config(
            vocab_size=9, embed_dim=4, n_heads=2, max_seq_len=6, n_layers=1, n_e_steps=2,
            family="gaussian_full", decode_mode="full", prior_source="model_channel",
            s_e_step=True, transport_mode="regime_ii_covariant",
            lambda_h=0.5, lambda_gamma=0.5, r_update_mode="gradient", oracle_unroll_grad=True)
        m = VFEModel(cfg).to(_DEVICE)
    torch.manual_seed(3)
    with torch.no_grad():
        for p in m.parameters():
            if p.requires_grad and p.dim() >= 1:
                p.normal_(0.0, 0.3)

    tok = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len), device=_DEVICE)
    tgt = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len), device=_DEVICE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loss = m(tok, tgt)[1]
    assert torch.isfinite(loss)                                        # finite loss
    assert loss.is_cuda                                                # CUDA residency
    loss.backward()

    grads = [(name, p.grad) for name, p in m.named_parameters() if p.grad is not None]
    assert grads
    for name, g in grads:
        assert g.is_cuda, f"gradient {name} left CUDA"
        assert torch.isfinite(g).all(), f"non-finite gradient in {name}"

    build_optimizer(m, cfg).step()                                     # one optimizer step

    pb = m.prior_bank
    s_cov = covariance_from_packed(pb.s_sigma_log_embed, pb.s_sigma_lower_embed, eps=cfg.eps)
    assert s_cov.is_cuda                                               # packed s covariance on CUDA
    eigs = torch.linalg.eigvalsh(s_cov)
    assert (eigs > 0).all()                                           # SPD model covariances


# ---------------------------------------------------------------------------
# Pure-route identity probe (Task 6 Step 3): the feature branch and its merge base must produce a
# byte-identical fingerprint of one deterministic forward/backward/optimizer step of the default
# (diagonal / flat) route. tests/hierarchy_identity_probe.py writes each bundle; this test recursively
# compares the two named by VFE3_BASELINE_BUNDLE / VFE3_FEATURE_BUNDLE with torch.equal. Any difference
# is BLOCKING -- the completed hierarchy is meant to leave the pure path untouched.
# ---------------------------------------------------------------------------

def _assert_bundle_equal(a, b, path="bundle"):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor), f"{path}: tensor vs {type(b).__name__}"
        assert a.shape == b.shape, f"{path}: shape {tuple(a.shape)} != {tuple(b.shape)}"
        assert a.dtype == b.dtype, f"{path}: dtype {a.dtype} != {b.dtype}"
        assert torch.equal(a, b), f"{path}: tensor values differ"
    elif isinstance(a, dict):
        assert isinstance(b, dict), f"{path}: dict vs {type(b).__name__}"
        assert set(a) == set(b), f"{path}: key set differs ({set(a) ^ set(b)})"
        for k in a:
            _assert_bundle_equal(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        assert isinstance(b, (list, tuple)), f"{path}: sequence vs {type(b).__name__}"
        assert len(a) == len(b), f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_bundle_equal(x, y, f"{path}[{i}]")
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def test_pure_route_bundle_is_byte_identical_to_branch_base():
    baseline = os.environ.get("VFE3_BASELINE_BUNDLE")
    feature = os.environ.get("VFE3_FEATURE_BUNDLE")
    if not baseline or not feature:
        pytest.skip("set VFE3_BASELINE_BUNDLE and VFE3_FEATURE_BUNDLE (see hierarchy_identity_probe.py)")
    base_bundle = torch.load(baseline, weights_only=False)
    feat_bundle = torch.load(feature, weights_only=False)
    assert base_bundle["state_dict_keys"] == feat_bundle["state_dict_keys"]
    _assert_bundle_equal(base_bundle, feat_bundle)
