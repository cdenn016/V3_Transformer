r"""Regression pins for the 2026-07-27 deep-audit punch list.

One test per surviving finding, each asserting the CORRECTED behavior and, where the defect was
silent, also asserting that the pre-fix behavior would have failed. See
``docs/audits/audit-2026-07-27.md`` for the full findings, verifier verdicts and duel adjudications.

Every model here is tiny (K < 6, single-digit everything) per the project's CPU-test rule.
"""

import math
import warnings

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.emission import bohning_curvature_diagonal, bohning_emission_terms
from vfe3.gradients.kernels import mm_exact_update
from vfe3.model.model import VFEModel
from vfe3.numerics import apply_mu_trust_region

BASE = dict(vocab_size=16, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1, seed=0)


# --------------------------------------------------------------- the emission cluster (6 findings)

def _emission_case(**kw):
    cfg = VFE3Config(**{**BASE, "use_prior_bank": False, "e_step_update": "mm_exact",
                        "emission_mode": "separate", "emission_weight": 2.0, **kw})
    torch.manual_seed(0)
    return VFEModel(cfg), torch.randint(0, cfg.vocab_size, (2, 8))


def test_emission_survives_autocast():
    r"""``mm_exact_update``'s fp32 island dropped ``emission``/``emission_weight`` from its explicit
    keyword list, so the recursion defaulted them and the Bohning block never ran. Both triggers are
    live together by construction: config REQUIRES ``mm_exact`` whenever emission is on, and the
    belief pipeline runs under autocast whenever ``amp_dtype`` is set. Measured pre-fix:
    ``torch.equal(mu*_on, mu*_off) == True`` under bf16 on both CPU and CUDA."""
    torch.manual_seed(0)
    n, k = 3, 4
    omega = torch.eye(k).expand(n, n, k, k).contiguous()          # identity transport
    args = (torch.randn(n, k), torch.rand(n, k) + 0.5, torch.randn(n, k), torch.rand(n, k) + 0.5, omega)
    d = torch.rand(k) + 0.5
    g = torch.randn(n, k)
    z0 = torch.randn(n, k)

    def run(emission):
        mu, _ = mm_exact_update(*args, irrep_dims=[2, 2], emission_weight=2.0, emission=emission)
        return mu

    off = run(None)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        amp_on, amp_off = run((d, g, z0)), run(None)
    assert not torch.equal(amp_on, amp_off), "emission must survive the autocast fp32 island"
    assert torch.allclose(amp_off, off, atol=1e-5)


def test_emission_anchors_on_its_own_expansion_point_not_the_layer_prior():
    r"""``(d, g)`` is built ONCE before the stack while ``vfe_stack`` advances ``mu_p`` every layer,
    so a fusion that re-centered on its own ``mu_p`` minimized a quadratic that majorizes nothing at
    ``n_layers > 1``. The expansion point now travels in the tuple."""
    torch.manual_seed(0)
    n, k = 3, 4
    omega = torch.eye(k).expand(n, n, k, k).contiguous()          # identity transport
    args = (torch.randn(n, k), torch.rand(n, k) + 0.5, torch.randn(n, k), torch.rand(n, k) + 0.5, omega)
    d, g, z0 = torch.rand(k) + 0.5, torch.randn(n, k), torch.randn(n, k)

    # The fusion adds w*(d*z_0 + g) to the numerator, so it depends on the pair (d, d*z_0 + g) ONLY.
    # Build a second tuple with the same d and the same d*z_0 + g but a different split between the
    # two: the corrected code must return an identical mu*, while code that anchored on the seam's
    # own mu_p would add w*(d*mu_p + g) and see the two different g's.
    z1 = z0 + 3.0
    g1 = g + d * (z0 - z1)                                        # keeps d*z + g invariant
    assert not torch.allclose(g, g1, atol=1e-3), "the two tuples must actually differ"

    mu_a, _ = mm_exact_update(*args, irrep_dims=[2, 2], emission_weight=1.0, emission=(d, g, z0))
    mu_b, _ = mm_exact_update(*args, irrep_dims=[2, 2], emission_weight=1.0, emission=(d, g1, z1))
    assert torch.allclose(mu_a, mu_b, atol=1e-5), \
        "the fusion must anchor on the returned z_0, not on the layer prior it is fused against"

    # and the anchor is load-bearing: a genuinely different d*z_0 + g must move the result
    mu_c, _ = mm_exact_update(*args, irrep_dims=[2, 2], emission_weight=1.0,
                              emission=(d, g, z0 + 3.0))
    assert not torch.allclose(mu_a, mu_c, atol=1e-4)


def test_expansion_point_probabilities_are_frozen_but_the_table_still_trains():
    r"""The module's stated MM contract is that ``p_0`` is detached; the code kept ``W`` live through
    every softmax pass, which broke the contract AND retained all three vocabulary loops for backward
    (measured 1.6x the memory of the dense decode it exists to avoid). ``W`` must still receive a
    gradient through the CONTRACTION and the observed row."""
    torch.manual_seed(0)
    vocab, k = 12, 4
    weight = torch.randn(vocab, k, requires_grad=True)
    mu_p = torch.randn(2, 3, k)
    tokens = torch.randint(0, vocab, (2, 3))

    _, g, _ = bohning_emission_terms(mu_p, weight, tokens)
    g.sum().backward()
    assert weight.grad is not None and torch.isfinite(weight.grad).all()
    assert float(weight.grad.abs().max()) > 0.0, "the readout table must still get an E-step gradient"

    # frozen-p_0 reference: identical to differentiating with the probabilities held constant
    w2 = weight.detach().clone().requires_grad_(True)
    probs = torch.softmax(mu_p @ w2.detach().T, dim=-1).detach()
    ref = (-(probs @ w2) + w2[tokens]).sum()
    ref.backward()
    assert torch.allclose(weight.grad, w2.grad, atol=1e-5), "p_0 must be frozen w.r.t. W"


def test_free_energy_value_scores_the_emission_block():
    r"""``free_energy_value`` accepted-and-ignored the emission, so every logged F measured a
    functional the belief does not descend -- the D-08 class, one file over. Measured pre-fix: one
    E-step lowered the true objective 4.825 -> 3.803 while the reported F ROSE 96x."""
    from vfe3.inference.e_step import free_energy_value
    from vfe3.belief import BeliefState
    from vfe3.geometry.groups import get_group

    torch.manual_seed(0)
    grp = get_group("block_glk")(4, 2)
    n, k = 4, 4
    belief = BeliefState(mu=torch.randn(n, k), sigma=torch.rand(n, k) + 0.5,
                         phi=torch.zeros(n, grp.generators.shape[0]))
    mu_p, sigma_p = torch.randn(n, k), torch.rand(n, k) + 0.5
    d, g, z0 = torch.rand(k) + 0.5, torch.randn(n, k), belief.mu.clone()

    f_off = free_energy_value(belief, mu_p, sigma_p, grp)
    f_on = free_energy_value(belief, mu_p, sigma_p, grp, emission_weight=1.5, emission=(d, g, z0))
    assert not torch.isclose(f_off, f_on), "the reported F must include the emission block"

    # at mu == z0 the quadratic and linear parts vanish; only the trace term survives
    expected = f_off + 1.5 * 0.5 * (d * belief.sigma).sum()
    assert torch.allclose(f_on, expected, atol=1e-5)


def test_bohning_curvature_uses_the_cancellation_free_form():
    r"""The one-pass ``sum W^2 - (sum W)^2 / V`` identity cancels two O(V) quantities; at the live
    V it lost 3+ digits on a column with nonzero mean. The centered form is exact and non-negative
    by construction, which also makes the eps clamp a real SPD floor."""
    torch.manual_seed(0)
    vocab, k = 4096, 4
    weight = (torch.randn(vocab, k) * 0.02 + 1.0)                 # unit column mean: the bad case
    got = bohning_curvature_diagonal(weight)
    ref = 0.5 * ((weight.double() - weight.double().mean(0)) ** 2).sum(0)
    assert torch.allclose(got.double(), ref, rtol=1e-5)

    # A genuinely constant column has d_k == 0 exactly. The centered form leaves only fp32
    # accumulation residue, orders below any real curvature, and never negative -- which is what the
    # SPD fusion needs. (The one-pass form's residue grew with the column MEAN; this one does not.)
    constant = torch.full((512, 3), 0.7)
    d_const = bohning_curvature_diagonal(constant, eps=1e-12)
    assert bool((d_const > 0).all()) and float(d_const.max()) < 1e-9
    assert float(bohning_curvature_diagonal(torch.full((512, 3), 70.0), eps=1e-12).max()) < 1e-6


# ------------------------------------------------------------------- numerics / geometry guards

@pytest.mark.parametrize("mode", ["box", "ball"])
@pytest.mark.parametrize("is_diagonal", [True, False])
def test_mu_trust_region_contains_non_finite_updates(mode, is_diagonal):
    r"""This guard exists to bound an exploding update, yet a non-finite entry defeated it:
    ``solve_triangular`` spread one inf across UNRELATED coordinates (all-NaN out), and 'ball' mode
    zeroed the finite coordinates via ``inf * 0``. Both were reproduced before the fix."""
    delta = torch.tensor([1.0, float("inf"), 2.0])
    sigma = torch.ones(3) if is_diagonal else torch.eye(3)
    out = apply_mu_trust_region(delta, sigma, trust=5.0, mode=mode, is_diagonal=is_diagonal)
    assert torch.isfinite(out).all(), "a non-finite update must not produce a non-finite step"
    assert float(out.norm()) <= 5.0 * math.sqrt(3) + 1e-4

    nan_in = torch.tensor([1.0, float("nan"), 2.0])
    nan_out = apply_mu_trust_region(nan_in, sigma, trust=5.0, mode=mode, is_diagonal=is_diagonal)
    assert torch.isfinite(nan_out).all()

    finite = torch.tensor([0.1, 0.2, 0.3])                        # untouched when nothing explodes
    kept = apply_mu_trust_region(finite, sigma, trust=5.0, mode=mode, is_diagonal=is_diagonal)
    assert torch.allclose(kept, finite, atol=1e-6)


# -------------------------------------------------------------------------- diagnostic honesty

def test_attention_entropy_min_is_not_structurally_constant():
    r"""Query row 0 has exactly one admissible key under any causal prior, so its entropy is 0 for
    every head, layer, seed and parameter value -- the min over rows could never exceed the eps
    artifact. Measured pre-fix: attn_entropy_min == 8.28931e-11 and collapsed_heads == 2.0,
    byte-identical across two seeds AND three distinct causal priors."""
    seen = set()
    for seed in (0, 7):
        for prior in ("causal", "causal_noself", "causal_alibi_noself"):
            cfg = VFE3Config(**{**BASE, "seed": seed, "beta_attention_prior": prior})
            torch.manual_seed(seed)
            model = VFEModel(cfg)
            with torch.no_grad():
                d = model.diagnostics(torch.randint(0, cfg.vocab_size, (2, 8)))
            seen.add(round(d["attn_entropy_min"], 9))
            assert d["attn_entropy_min"] > 1e-6, "row 0 must not pin the minimum at the eps artifact"
    assert len(seen) > 1, "the metric must vary with seed/prior rather than being structurally fixed"


def test_per_head_gauge_invariant_is_conjugation_invariant():
    r"""``per_head_gauge_invariants`` returned a singular-value ratio, invariant only under
    ORTHOGONAL conjugation, while the gauge action is GL(K) conjugation. The project had already
    made exactly this correction for ``sp``; the per-head twin was left on ``block_glk``, the live
    group. See tests/test_metrics.py for the numeric invariance pin."""
    from vfe3.metrics import per_head_gauge_invariants
    torch.manual_seed(0)
    dims = [2, 2]

    def blockdiag(mats):
        out = torch.zeros(4, 4, dtype=mats[0].dtype)
        start = 0
        for m in mats:
            d = m.shape[-1]
            out[start:start + d, start:start + d] = m
            start += d
        return out

    exp_phi = torch.stack([blockdiag([torch.matrix_exp(0.3 * torch.randn(d, d)) for d in dims])
                           for _ in range(4)])
    g = blockdiag([torch.matrix_exp(0.4 * torch.randn(d, d)) for d in dims])
    moved = g @ exp_phi @ torch.linalg.inv(g)
    base, conj = per_head_gauge_invariants(exp_phi, dims), per_head_gauge_invariants(moved, dims)
    for key in ("logdet", "anisotropy"):
        rel = ((conj[key] - base[key]).abs() / base[key].abs().clamp(min=1e-9)).max()
        assert float(rel) < 1e-4, f"{key} is not conjugation-invariant ({float(rel):.2e})"


def test_gamma_fold_escapes_the_beta_channel_ablation():
    r"""``collect_beta_channel_decomposition`` patches ``attention_weights`` on free_energy /
    kernels / e_step to ablate the BELIEF channel. ``_fold_gamma_prior`` resolved its binding at
    call time from free_energy, so the MODEL channel's softmax was intercepted too and the reported
    content/positional split mixed the two channels."""
    from vfe3 import free_energy as fe
    from vfe3.gradients import kernels as km
    from vfe3.inference import e_step as em

    cfg = VFE3Config(**{**BASE, "gauge_group": "block_glk", "gamma_as_beta_prior": True,
                        "lambda_gamma": 1.0, "gamma_prior_weight": 0.5})
    torch.manual_seed(0)
    model = VFEModel(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, 8))

    calls = {"n": 0}
    original = fe.attention_weights

    def spy(*a, **k):
        calls["n"] += 1
        return original(*a, **k)

    binders = [m for m in (fe, km, em) if getattr(m, "attention_weights", None) is original]
    for m in binders:
        m.attention_weights = spy
    try:
        with torch.no_grad():
            model._fold_gamma_prior(None, tokens, model.prior_bank.encode(tokens).phi)
    finally:
        for m in binders:
            m.attention_weights = original
    assert binders, "the ablation must actually have something to patch"
    assert calls["n"] == 0, "the gamma softmax must not be intercepted by the beta-channel ablation"


def test_gamma_fold_is_composed_in_log_space_without_a_floor():
    r"""The fold composed the mixture in probability space and then took ``log(pi.clamp(1e-12))``,
    flooring the deep tail flat at -27.63 nats and destroying ALiBi's monotone decay on the steepest
    heads (72 of 126 outward steps became ties). It is now an exact log-space mixture."""
    from vfe3.attention_prior import attention_log_prior

    cfg = VFE3Config(**{**BASE, "gauge_group": "block_glk", "gamma_as_beta_prior": True,
                        "lambda_gamma": 0.75, "gamma_prior_weight": 0.5,
                        "beta_attention_prior": "causal_alibi_noself",
                        "gamma_attention_prior": "causal_alibi_noself", "alibi_slope": 1})
    torch.manual_seed(0)
    model = VFEModel(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, 8))
    log_prior = attention_log_prior("causal_alibi_noself", 8, 8, n_heads=2, alibi_slope=1,
                                    device=tokens.device, dtype=torch.float32)
    with torch.no_grad():
        out = model._fold_gamma_prior(log_prior, tokens, model.prior_bank.encode(tokens).phi)

    finite = torch.isfinite(out)
    rows = out.masked_fill(~finite, float("-inf")).logsumexp(dim=-1)
    assert float(rows.abs().max()) < 1e-5, "rows must stay exactly normalized on the causal support"
    assert bool((~finite).any()), "the -inf causal structure must survive"
    floor = math.log(1e-12)
    assert not bool((((out - floor).abs() < 1e-5) & finite).any()), \
        "no finite entry may sit on the old log_eps floor"


# ------------------------------------------------------------------------ config / experiment wiring

def test_mm_exact_reports_its_unread_step_knobs():
    r"""The mm_exact branch reads only mm_damping/eps/sigma_max/skip_belief_sigma_update; every
    step-size, preconditioner and trust knob lives in the gradient branch, on both channels. Only
    the FORWARD rule existed (mm_damping inert when NOT mm_exact), so a run that set them got no
    warning and seven ablation sweeps varied them against an mm_exact baseline."""
    common = dict(vocab_size=16, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                  s_e_step=True, prior_source="model_channel")
    knobs = dict(e_q_mu_lr=0.9, e_q_sigma_lr=0.001, e_sigma_q_trust=10.0, e_s_mu_lr=0.85)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        VFE3Config(**common, e_step_update="mm_exact", **knobs)
    inert = [str(w.message) for w in caught if "unread under e_step_update='mm_exact'" in str(w.message)]
    assert inert, "mm_exact must report the step knobs it never reads"
    for name in knobs:
        assert name in inert[0]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        VFE3Config(**common, e_step_update="gradient", **knobs)
    assert not [w for w in caught if "unread under" in str(w.message)], \
        "the gradient route reads all of them; it must not warn"

    with warnings.catch_warnings(record=True) as caught:   # the project's zero-warning invariant
        warnings.simplefilter("always")
        VFE3Config(vocab_size=16, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1)
    assert not caught


def test_ablation_sweeps_of_gradient_only_knobs_pin_the_gradient_route():
    r"""Seven sweeps varied knobs the mm_exact baseline never reads, so every cell was byte-identical
    and any spread plotted as a curve was seed noise. The B3/EXP-14 Fisher-vs-Euclidean ablation
    (6 cells x 3 seeds) compared an arm against itself."""
    import ablation
    for name in ("e_q_mu_lr", "e_q_sigma_lr", "e_s_mu_lr", "e_s_sigma_lr",
                 "e_mu_q_trust", "e_mu_q_trust_ball", "e_sigma_q_trust", "fisher_mu_precond"):
        sweep = ablation.SWEEPS[name]
        assert sweep.get("requires", {}).get("e_step_update") == "gradient", \
            f"sweep {name!r} varies a knob the mm_exact baseline does not read"


def test_scaling_route_a_scales_kl_max_with_width():
    r"""BASELINE freezes kl_max at 8*train_K, so route A inherited one ceiling at every width while
    its three siblings set 8*k per cell -- route A's divergence ceiling fell from 8.0 to 1.33
    nats/coordinate across the very axis it measures."""
    import scaling
    cells = scaling.route_grow_k([20, 40, 80], n_heads=2)
    for cell in cells:
        k = cell["overrides"]["embed_dim"]
        assert cell["overrides"]["kl_max"] == 8 * k, "route A must scale kl_max with width"
