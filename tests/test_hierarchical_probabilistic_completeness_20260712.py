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
