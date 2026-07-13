r"""Objective-parity tests for the centralized effective beta-prior builder (audit PB-12, plan4 task 1).

``VFEModel._effective_beta_log_prior(belief, context)`` is the single authoritative constructor of the
belief-channel attention log-prior the E-step actually descends: it folds the detached
precision-weighted reliability bias (``_fold_precision_bias``) and, under ``gamma_as_beta_prior``, the
detached hierarchical gamma prior (``_fold_gamma_prior``) onto the captured RAW ``_attention_log_prior``.
The forward pass builds one ``EffectiveBetaPriorContext`` at the fixed pre-``vfe_stack`` seam and calls
the helper; later reflection/two-hop scorers reuse the SAME helper so they score the SAME objective.

Pins:
  * value parity -- the helper reproduces the pre-refactor inline fold sequence for no folds, precision
    only, gamma only, and both folds, and the forward stores exactly the helper's output;
  * the learnable T5 relative-position bias graph survives the helper while the precision reliability
    bias is detached from the (captured) belief covariance and the gamma fold is detached from the s
    tables;
  * the fixed pre-stack ``precision_sigma`` -- changing a CANDIDATE belief's covariance leaves the
    precision-only prior EXACTLY unchanged, while changing ``context.precision_sigma`` changes it;
  * candidate-frame dependence of the tied gamma fold -- flipping one belief-frame reflection changes
    the prior under ``s_frame_mode='tied'`` but NOT under the independent ``phi_tilde`` model frame.

Device-agnostic (CPU default; set VFE3_TEST_DEVICE=cuda for the GPU). Tiny models (K < 6).
"""
import math
import os
import warnings

import pytest
import torch

from vfe3.belief import BeliefState
from vfe3.config import VFE3Config
from vfe3.contracts import EffectiveBetaPriorContext, MetropolisObjectiveContext
from vfe3.families.base import get_family
from vfe3.free_energy import attention_tau, free_energy, pairwise_energy, query_adaptive_tau, reduced_free_energy
from vfe3.geometry.groups import get_group
from vfe3.geometry.rope import build_rope_rotation
from vfe3.geometry.transport import RopeTransport, transport_covariance, transport_mean
from vfe3.inference.e_step import build_belief_transport, free_energy_value, phi_alignment_loss
from vfe3.model.block import _as_coeff
from vfe3.model.model import VFEModel

DEVICE = torch.device(os.environ.get("VFE3_TEST_DEVICE", "cpu"))


# --------------------------------------------------------------------------------------------------
# builders / capture helpers
# --------------------------------------------------------------------------------------------------
def _model(*, seed: int = 0, perturb: bool = True, **over) -> VFEModel:
    base = dict(vocab_size=10, embed_dim=4, n_heads=2, max_seq_len=6, n_layers=1,
                n_e_steps=1, e_q_mu_lr=0.4, e_phi_lr=0.0, mass_phi=0.0,
                mstep_self_coupling_weight=0.0, pos_phi="none")
    base.update(over)
    cfg = VFE3Config(**base)
    torch.manual_seed(seed)
    model = VFEModel(cfg).to(DEVICE)
    model.eval()
    if perturb:
        with torch.no_grad():
            # non-constant tr Sigma_j (precision bias non-trivial) + non-identity exp(phi) (transport
            # non-trivial), so neither fold is a hidden no-op.
            model.prior_bank.sigma_log_embed.add_(
                0.4 * torch.randn_like(model.prior_bank.sigma_log_embed))
            model.prior_bank.phi_embed.add_(
                0.3 * torch.randn_like(model.prior_bank.phi_embed))
            if hasattr(model.prior_bank, "s_mu_embed"):
                model.prior_bank.s_mu_embed.add_(
                    0.5 * torch.randn_like(model.prior_bank.s_mu_embed))
                model.prior_bank.s_sigma_log_embed.add_(
                    0.3 * torch.randn_like(model.prior_bank.s_sigma_log_embed))
    return model


def _tokens(model: VFEModel, n: int = 4, b: int = 2, seed: int = 7) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, model.cfg.vocab_size, (b, n), generator=g).to(DEVICE)


def _forward_capture(model: VFEModel, tokens: torch.Tensor, *, grad: bool = False):
    r"""Run one belief forward and return (context, initial_belief, folded_log_prior).

    ``initial_belief`` is the belief entering ``vfe_stack`` (the candidate the forward feeds the
    helper); ``folded_log_prior`` is the prior the forward actually descended.
    """
    diag: dict = {}
    cap: dict = {"diagnostic": diag}
    ctxmgr = torch.no_grad() if not grad else _null()
    with ctxmgr:
        model.forward_beliefs(tokens, capture=cap)
    return cap["beta_prior_context"], diag["initial_belief"], diag["log_prior"]


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def _reference_fold(model: VFEModel, ctx: EffectiveBetaPriorContext, belief) -> torch.Tensor:
    r"""The pre-refactor inline forward fold sequence, replicated verbatim from model.forward_beliefs.

    This is the independent 'current forward construction' the helper must reproduce exactly.
    """
    log_prior = model._fold_precision_bias(ctx.base_log_prior, ctx.precision_sigma)
    if model.cfg.gamma_as_beta_prior:
        tied_model_frame = model.cfg.s_frame_mode == "tied"
        s_belief = None if ctx.s_mu is None else (ctx.s_mu, ctx.s_sigma)
        log_prior = model._fold_gamma_prior(
            log_prior, ctx.token_ids, ctx.model_phi,
            omega=belief.omega if tied_model_frame else None,
            reflection=(belief.reflection if tied_model_frame else None),
            s_belief=s_belief,
        )
    return log_prior


def _assert_priors_equal(a, b, *, msg: str = ""):
    if a is None or b is None:
        assert a is None and b is None, f"one prior is None, the other is not {msg}"
        return
    assert a.shape == b.shape, f"prior shape mismatch {a.shape} vs {b.shape} {msg}"
    fin = torch.isfinite(a)
    assert torch.equal(fin, torch.isfinite(b)), f"finite/-inf mask mismatch {msg}"
    assert torch.equal(a[fin], b[fin]), f"finite entries differ {msg}"


# --------------------------------------------------------------------------------------------------
# config presets for the four fold combinations
# --------------------------------------------------------------------------------------------------
_GAMMA = dict(gamma_as_beta_prior=True, lambda_gamma=0.5, kappa_gamma=1.0, gamma_prior_weight=0.5)

_FOLDS = {
    "none":      dict(),
    "precision": dict(precision_weighted_attention=True, precision_attention_b0=1.5),
    "gamma":     dict(**_GAMMA),
    "both":      dict(precision_weighted_attention=True, precision_attention_b0=1.5, **_GAMMA),
}


# --------------------------------------------------------------------------------------------------
# Step 1: value parity -- helper == pre-refactor inline fold sequence, and forward stores the helper.
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("combo", list(_FOLDS))
def test_helper_matches_reference_fold(combo):
    model = _model(**_FOLDS[combo])
    tokens = _tokens(model)
    ctx, initial, folded = _forward_capture(model, tokens)

    helper = model._effective_beta_log_prior(initial, ctx)
    reference = _reference_fold(model, ctx, initial)
    _assert_priors_equal(helper, reference, msg=f"[{combo}] helper vs reference")


@pytest.mark.parametrize("combo", list(_FOLDS))
def test_forward_descends_the_helper_output(combo):
    # The forward's stored/descended log_prior must BE the helper's output for the candidate belief it
    # fed (wiring: the inline fold was replaced by exactly one helper call).
    model = _model(**_FOLDS[combo])
    tokens = _tokens(model)
    ctx, initial, folded = _forward_capture(model, tokens)

    helper = model._effective_beta_log_prior(initial, ctx)
    _assert_priors_equal(helper, folded, msg=f"[{combo}] helper vs forward-descended prior")


def test_context_fields_captured_at_pre_stack_seam():
    # The captured context is the fixed pre-stack state: token ids, the RAW attention prior (BEFORE any
    # fold), the pre-stack belief covariance, and the resolved model frame.
    model = _model(**_FOLDS["both"])
    tokens = _tokens(model)
    ctx, initial, folded = _forward_capture(model, tokens)

    assert isinstance(ctx, EffectiveBetaPriorContext)
    assert torch.equal(ctx.token_ids, tokens)
    raw = model._attention_log_prior(tokens.shape[1], tokens.device)
    _assert_priors_equal(ctx.base_log_prior, raw, msg="base_log_prior is the RAW prior")
    assert torch.equal(ctx.precision_sigma, initial.sigma)  # pre-stack belief covariance
    assert ctx.model_phi.shape[:2] == tokens.shape


# --------------------------------------------------------------------------------------------------
# Step 1 (cont.): gradient contracts -- T5 graph preserved, precision + gamma folds detached.
# --------------------------------------------------------------------------------------------------
def _t5_model(**over) -> VFEModel:
    return _model(beta_attention_prior="t5_relative_bias", t5_learnable_bias=True, **over)


def test_t5_bias_graph_survives_the_helper():
    model = _t5_model(precision_weighted_attention=True, precision_attention_b0=1.5)
    tokens = _tokens(model)
    ctx, initial, _ = _forward_capture(model, tokens, grad=True)
    assert ctx.base_log_prior.requires_grad          # learnable T5 table feeds the raw prior

    out = model._effective_beta_log_prior(initial, ctx)
    assert out.requires_grad
    out[torch.isfinite(out)].sum().backward()
    assert model.t5_bias.grad is not None
    assert model.t5_bias.grad.abs().sum() > 0


def test_precision_bias_detached_from_covariance():
    # The precision reliability bias -log(b0 + tr Sigma_j) is detached: even a grad-leaf precision_sigma
    # receives NO gradient, while the T5 graph in the base prior still flows.
    model = _t5_model(precision_weighted_attention=True, precision_attention_b0=1.5)
    tokens = _tokens(model)
    ctx, initial, _ = _forward_capture(model, tokens, grad=True)

    ps = ctx.precision_sigma.detach().clone().requires_grad_(True)
    out = model._effective_beta_log_prior(initial, ctx._replace(precision_sigma=ps))
    out[torch.isfinite(out)].sum().backward()
    assert ps.grad is None                            # covariance path severed
    assert model.t5_bias.grad is not None             # T5 path preserved


def test_gamma_fold_detached_from_s_tables():
    # gamma_as_beta_prior computes gamma under no_grad, so the s tables get NO gradient through the belief
    # prior; the T5 base graph still flows.
    model = _t5_model(**_GAMMA)
    tokens = _tokens(model)
    ctx, initial, _ = _forward_capture(model, tokens, grad=True)

    out = model._effective_beta_log_prior(initial, ctx)
    out[torch.isfinite(out)].sum().backward()
    assert model.prior_bank.s_mu_embed.grad is None
    assert model.prior_bank.s_sigma_log_embed.grad is None
    assert model.t5_bias.grad is not None


# --------------------------------------------------------------------------------------------------
# Step 2: fixed precision -- the helper never reads belief.sigma for the precision fold.
# --------------------------------------------------------------------------------------------------
def test_precision_prior_ignores_candidate_covariance():
    model = _model(precision_weighted_attention=True, precision_attention_b0=1.5)
    tokens = _tokens(model)
    ctx, initial, _ = _forward_capture(model, tokens)

    other = initial._replace(sigma=initial.sigma * 3.0 + 0.7)   # deliberately different candidate cov
    a = model._effective_beta_log_prior(initial, ctx)
    b = model._effective_beta_log_prior(other, ctx)
    _assert_priors_equal(a, b, msg="precision prior must ignore candidate covariance")


def test_precision_prior_tracks_context_covariance():
    model = _model(precision_weighted_attention=True, precision_attention_b0=1.5)
    tokens = _tokens(model)
    ctx, initial, _ = _forward_capture(model, tokens)

    ctx2 = ctx._replace(precision_sigma=ctx.precision_sigma * 3.0 + 0.7)
    a = model._effective_beta_log_prior(initial, ctx)
    b = model._effective_beta_log_prior(initial, ctx2)
    fin = torch.isfinite(a)
    assert not torch.allclose(a[fin], b[fin])          # the FIXED precision_sigma is what the fold reads


# --------------------------------------------------------------------------------------------------
# Step 2 (cont.): tied vs independent (phi_tilde) frame -- candidate-reflection dependence.
# --------------------------------------------------------------------------------------------------
def test_tied_gamma_changes_with_candidate_reflection():
    model = _model(gauge_parameterization="phi", phi_reflection="init_seed",
                   s_frame_mode="tied", **_GAMMA)
    tokens = _tokens(model)
    ctx, initial, _ = _forward_capture(model, tokens)
    assert initial.reflection is not None

    flipped = initial.reflection.clone()
    flipped[:, 1] *= -1.0                                # flip one position's det-sign
    cand = initial._replace(reflection=flipped)

    base = model._effective_beta_log_prior(initial, ctx)
    moved = model._effective_beta_log_prior(cand, ctx)
    fin = torch.isfinite(base)
    assert not torch.allclose(base[fin], moved[fin])    # tied: the belief frame enters the gamma fold


def _phi_tilde_model() -> VFEModel:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _model(
            gauge_parameterization="phi", s_frame_mode="phi_tilde", s_e_step=True,
            prior_source="model_channel", share_refine_s_transport=False,
            phi_reflection="off", pos_rotation="none",
            e_s_mu_lr=0.5, e_s_sigma_lr=0.2, m_s_phi_lr=0.1, lambda_h=0.0, **_GAMMA)


def test_phi_tilde_gamma_invariant_to_candidate_reflection():
    # Under the INDEPENDENT phi_tilde model frame, the tied belief frame is not consumed: a belief-frame
    # reflection leaves the gamma fold EXACTLY unchanged.
    model = _phi_tilde_model()
    tokens = _tokens(model)
    ctx, initial, _ = _forward_capture(model, tokens)
    assert model.cfg.s_frame_mode == "phi_tilde"

    injected = initial._replace(
        reflection=torch.where(
            torch.arange(initial.mu.shape[-2], device=DEVICE) % 2 == 0, -1.0, 1.0
        ).expand(initial.mu.shape[:-1]).clone())
    base = model._effective_beta_log_prior(initial, ctx)
    same = model._effective_beta_log_prior(injected, ctx)
    _assert_priors_equal(base, same, msg="phi_tilde model frame must ignore the belief reflection")


# --------------------------------------------------------------------------------------------------
# Task 2 (audit m9 / PB-12): two-hop coupling in the phi objective.
#
# The mean/covariance E-step kernels honor lambda_twohop but phi_alignment_loss did not, so under
# lambda_twohop>0 with e_phi_lr>0 phi descended a DIFFERENT objective. phi_alignment_loss must now
# add the SAME detached-weight two-hop block free_energy already carries:
#   F_2 = lambda_twohop * sum_ik (beta beta)_ik E_ik,   beta = softmax_j(log pi - E/tau),
# with W2 = beta.detach() @ beta.detach() (no independent entropy term), on the value-gauge energy
# grid. Its phi-gradient must match autograd of free_energy_value (which already honors
# lambda_twohop) under both the coupled (flat) and decoupled-RoPE value gauges, and the block must
# be an EXACT no-op at lambda_twohop=0.0.
# --------------------------------------------------------------------------------------------------
def _phi_beliefs(seed: int, *, n: int = 3, k: int = 3):
    r"""Tiny (N=3, K=3) glk beliefs + prior + phi leaf for the phi-objective parity checks."""
    torch.manual_seed(seed)
    group = get_group("glk")(k)
    mu = torch.randn(n, k, dtype=torch.float32, device=DEVICE)
    sigma = torch.rand(n, k, dtype=torch.float32, device=DEVICE) + 0.6
    mu_p = torch.randn(n, k, dtype=torch.float32, device=DEVICE)
    sigma_p = torch.rand(n, k, dtype=torch.float32, device=DEVICE) + 0.6
    phi = 0.2 * torch.randn(n, group.generators.shape[0], dtype=torch.float32, device=DEVICE)
    log_prior = torch.randn(n, n, dtype=torch.float32, device=DEVICE)
    return group, mu, sigma, mu_p, sigma_p, phi, log_prior


def _rope_for(group, n: int, phi: torch.Tensor) -> torch.Tensor:
    return build_rope_rotation(
        torch.arange(n, device=DEVICE), group.irrep_dims,
        base=10.0, device=phi.device, dtype=phi.dtype,
    )


def _phi_loss_pre_twohop(mu, sigma, phi, group, *, tau, lambda_beta, log_prior,
                         rope=None, rope_on_value=True):
    r"""Faithful copy of phi_alignment_loss's PRE-two-hop body (the flat-entropy and decoupled-RoPE
    branches), rebuilt from the same public primitives. The independent 'old form' the extended loss
    must reproduce EXACTLY at lambda_twohop=0.0 (defaults mirror phi_alignment_loss's)."""
    omega = build_belief_transport(phi, group, transport_mode="flat", mu=mu, sigma=sigma,
                                   rope=rope, rope_on_value=rope_on_value)
    mu_t = transport_mean(omega, mu)
    sigma_t = transport_covariance(omega, sigma)
    fam = get_family("gaussian_diagonal")
    score_energy = pairwise_energy(fam(mu, sigma), fam(mu_t, sigma_t), alpha=1.0,
                                   kl_max=100.0, eps=1e-6, divergence_family="renyi",
                                   irrep_dims=group.irrep_dims)
    mass = 0.0
    if isinstance(omega, RopeTransport) and not omega.on_value:
        mu_tv = transport_mean(omega.base, mu)
        sigma_tv = transport_covariance(omega.base, sigma)
        value_energy = pairwise_energy(fam(mu, sigma), fam(mu_tv, sigma_tv), alpha=1.0,
                                       kl_max=100.0, eps=1e-6, divergence_family="renyi",
                                       irrep_dims=group.irrep_dims)
        zero = score_energy.new_zeros(score_energy.shape[:-1])
        return free_energy(
            zero, score_energy, zero,
            tau=tau, lambda_beta=lambda_beta,
            include_attention_entropy=True,
            log_prior=log_prior, coupling_energy=value_energy,
        ) + mass
    return lambda_beta * reduced_free_energy(score_energy, tau=tau, log_prior=log_prior).sum() + mass


@pytest.mark.parametrize("decoupled_rope", [False, True])
def test_phi_twohop_gradient_matches_scalar_free_energy(decoupled_rope):
    # Red oracle: the extended phi loss's two-hop phi-gradient must equal autograd of
    # free_energy_value's (which already honors lambda_twohop) -- the self-coupling term is
    # phi-independent, so only the coupled + two-hop blocks contribute to the phi gradient.
    group, mu, sigma, mu_p, sigma_p, phi, log_prior = _phi_beliefs(seed=41)
    n = mu.shape[0]
    kw = dict(tau=1.3, lambda_beta=0.7, lambda_twohop=0.2, log_prior=log_prior)
    if decoupled_rope:
        kw.update(rope=_rope_for(group, n, phi), rope_on_value=False)

    phi_loss = phi.clone().requires_grad_(True)
    loss = phi_alignment_loss(mu, sigma, phi_loss, group, **kw)
    grad_loss, = torch.autograd.grad(loss, phi_loss)

    phi_scalar = phi.clone().requires_grad_(True)
    scalar = free_energy_value(
        BeliefState(mu=mu, sigma=sigma, phi=phi_scalar), mu_p, sigma_p, group, **kw)
    grad_scalar, = torch.autograd.grad(scalar, phi_scalar)

    torch.testing.assert_close(grad_loss, grad_scalar, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("decoupled_rope", [False, True])
def test_phi_twohop_zero_weight_is_exact_identity(decoupled_rope):
    # Zero-weight identity: at lambda_twohop=0.0 the extended loss must be BYTE-identical (torch.equal
    # scalar AND gradient) to the pre-two-hop form -- the guarded block is a strict no-op.
    group, mu, sigma, mu_p, sigma_p, phi, log_prior = _phi_beliefs(seed=43)
    n = mu.shape[0]
    tau, lambda_beta = 1.1, 0.8
    rope = _rope_for(group, n, phi) if decoupled_rope else None
    rope_on_value = not decoupled_rope

    phi_ext = phi.clone().requires_grad_(True)
    ext = phi_alignment_loss(mu, sigma, phi_ext, group, tau=tau, lambda_beta=lambda_beta,
                             lambda_twohop=0.0, log_prior=log_prior,
                             rope=rope, rope_on_value=rope_on_value)
    grad_ext, = torch.autograd.grad(ext, phi_ext)

    phi_old = phi.clone().requires_grad_(True)
    old = _phi_loss_pre_twohop(mu, sigma, phi_old, group, tau=tau, lambda_beta=lambda_beta,
                               log_prior=log_prior, rope=rope, rope_on_value=rope_on_value)
    grad_old, = torch.autograd.grad(old, phi_old)

    assert torch.equal(ext.detach(), old.detach())
    assert torch.equal(grad_ext, grad_old)


# ==================================================================================================
# Task 3 (audit PB-12): the Metropolis reflection scorer evaluates the EXACT active fixed-belief
# objective. The current scorer omits the precision/tied-gamma folds, the two-hop block, the
# query-adaptive tau, the handoff-adjusted final-block prior, and the active transport/RoPE numerics.
# After the fix, ``_metropolis_prepare`` returns one ``MetropolisObjectiveContext`` and
# ``_metropolis_free_energy(belief, context)`` calls ``_effective_beta_log_prior(belief, context.prior)``
# per candidate and ``free_energy_value`` with the fixed captured tau/prior/rope + every active
# transport control, so the fixed-belief DeltaF is the exact change in F the E-step descended.
# All models are TINY (V=6, K=4, N=3), CPU-bound, device-agnostic.
# ==================================================================================================
_GAMMA_OVER = dict(gamma_as_beta_prior=True, lambda_gamma=0.5, kappa_gamma=1.0, gamma_prior_weight=0.5)


def _metro_model(mode, *, seed=0, perturb=True, **over) -> VFEModel:
    r"""Tiny reflection-Metropolis model. ``mode='omega'`` -> omega_direct + omega_reflection;
    ``mode='phi'`` -> phi + phi_reflection. Tables are perturbed so the transport and every fold bite
    (nonzero phi -> Omega != I; varying tr Sigma -> nontrivial precision/adaptive-tau; s tables ->
    nontrivial gamma)."""
    base = dict(gauge_group="glk", embed_dim=4, n_heads=1, vocab_size=6, max_seq_len=4,
                n_layers=1, n_e_steps=2, transport_mode="flat", e_phi_lr=0.0,
                use_head_mixer=False, family="gaussian_diagonal", decode_mode="diagonal",
                lambda_gamma=0.0, s_e_step=False, pos_phi="none")
    if mode == "omega":
        base.update(gauge_parameterization="omega_direct", omega_reflection="metropolis")
    else:
        base.update(gauge_parameterization="phi", phi_reflection="metropolis")
    base.update(over)
    torch.manual_seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # diagonal-family 'near-no sheet selection' efficacy warning
        m = VFEModel(VFE3Config(**base)).to(DEVICE)
    m.eval()
    if perturb:
        with torch.no_grad():
            m.prior_bank.sigma_log_embed.add_(
                0.4 * torch.randn_like(m.prior_bank.sigma_log_embed))
            if hasattr(m.prior_bank, "phi_embed"):
                m.prior_bank.phi_embed.add_(0.3 * torch.randn_like(m.prior_bank.phi_embed))
            if hasattr(m.prior_bank, "s_mu_embed"):
                m.prior_bank.s_mu_embed.add_(0.5 * torch.randn_like(m.prior_bank.s_mu_embed))
                m.prior_bank.s_sigma_log_embed.add_(
                    0.3 * torch.randn_like(m.prior_bank.s_sigma_log_embed))
            for a in ("connection_W", "connection_M", "connection_L"):
                p = getattr(m, a, None)
                if p is not None:
                    p.add_(0.15 * torch.randn_like(p))   # nonzero regime-II connection -> non-flat
    return m


def _metro_tokens(n: int = 3, b: int = 1, seed: int = 5) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    # distinct ids per row so the transport Omega_ij != I and the reflection genuinely moves F
    return torch.stack([torch.randperm(6, generator=g)[:n] for _ in range(b)]).to(DEVICE)


def _scorer_F(m, belief, context, *, mode) -> float:
    r"""Independent oracle for one fixed-belief F: rebuild the effective prior for THIS candidate via
    the authoritative ``_effective_beta_log_prior`` and evaluate ``free_energy_value`` with the fixed
    captured tau/prior/rope and every active transport/numerics control (audit PB-12). The scorer
    must reproduce this exactly."""
    cfg, grp = m.cfg, m.group
    gp = "omega_direct" if mode == "omega" else "phi"
    dev = belief.mu.device
    lp = m._effective_beta_log_prior(belief, context.prior)
    with torch.no_grad():
        return free_energy_value(
            belief, context.mu_p, context.sigma_p, grp,
            tau=context.tau, renyi_order=cfg.renyi_order, value=cfg.lambda_alpha,
            b0=_as_coeff(cfg.b0, dev), c0=_as_coeff(cfg.c0, dev),
            lambda_beta=cfg.lambda_beta, kl_max=cfg.kl_max, eps=cfg.eps,
            lambda_twohop=cfg.lambda_twohop, include_attention_entropy=cfg.include_attention_entropy,
            family=cfg.family, divergence_family=cfg.divergence_family,
            lambda_alpha_mode=cfg.lambda_alpha_mode, gauge_parameterization=gp, log_prior=lp,
            transport_mode=cfg.transport_mode,
            connection_W=getattr(m, "connection_W", None),
            connection_M=getattr(m, "connection_M", None),
            connection_L=getattr(m, "connection_L", None),
            cocycle_relaxation=cfg.cocycle_relaxation, link_alpha=cfg.link_alpha,
            link_soft_cap=cfg.link_soft_cap, clamp_monitor=cfg.transport_clamp_monitor,
            transport_mean_per_head=cfg.transport_mean_per_head, rope=context.rope,
            rope_on_cov=cfg.rope_full_gauge, rope_on_value=cfg.rope_on_value,
            exp_fp64_mode=cfg.exp_fp64_mode, exp_fp64_norm_threshold=cfg.exp_fp64_norm_threshold,
        ).item()


def _oracle_delta(m, context, tid, *, mode) -> float:
    cur = context.belief
    trial = m._metropolis_trial_belief(cur, context.token_ids, tid, mode=mode)
    return _scorer_F(m, trial, context, mode=mode) - _scorer_F(m, cur, context, mode=mode)


def _raw_F(m, belief, context, *, mode) -> float:
    r"""The PRE-Task-3 RAW-prior scorer body (scalar tau, ``_attention_log_prior`` with NO folds, no
    two-hop, no active transport controls). The independent reference for the folds-off identity and
    the acceptance-boundary corrected-vs-raw comparison."""
    cfg, grp = m.cfg, m.group
    gp = "omega_direct" if mode == "omega" else "phi"
    dev = belief.mu.device
    tau = attention_tau(m.effective_kappa_beta(dev), grp.irrep_dims)
    log_prior = m._attention_log_prior(belief.mu.shape[-2], dev)
    with torch.no_grad():
        return free_energy_value(
            belief, context.mu_p, context.sigma_p, grp,
            tau=tau, renyi_order=cfg.renyi_order, value=cfg.lambda_alpha,
            b0=_as_coeff(cfg.b0, dev), c0=_as_coeff(cfg.c0, dev),
            lambda_beta=cfg.lambda_beta, kl_max=cfg.kl_max, eps=cfg.eps,
            include_attention_entropy=cfg.include_attention_entropy,
            family=cfg.family, divergence_family=cfg.divergence_family,
            lambda_alpha_mode=cfg.lambda_alpha_mode, gauge_parameterization=gp,
            log_prior=log_prior).item()


def _raw_delta(m, context, tid, *, mode) -> float:
    cur = context.belief
    trial = m._metropolis_trial_belief(cur, context.token_ids, tid, mode=mode)
    return _raw_F(m, trial, context, mode=mode) - _raw_F(m, cur, context, mode=mode)


def _src_sign_state(m, tid, mode) -> float:
    return (m.prior_bank.reflection_sign[tid].item() if mode == "phi"
            else torch.det(m.prior_bank.omega_embed[tid]).item())


# --------------------------------------------------------------------------------------------------
# Step 1: exact-delta parity across the folds (precision / tied gamma / two-hop / adaptive tau / all).
# --------------------------------------------------------------------------------------------------
_FOLD_OVER = {
    "precision":    dict(precision_weighted_attention=True, precision_attention_b0=1.5),
    "gamma":        dict(**_GAMMA_OVER),
    "twohop":       dict(lambda_twohop=0.3),
    "adaptive_tau": dict(query_adaptive_tau=True, query_tau_c=0.9),
    "all":          dict(precision_weighted_attention=True, precision_attention_b0=1.5,
                         lambda_twohop=0.3, query_adaptive_tau=True, query_tau_c=0.9, **_GAMMA_OVER),
}


@pytest.mark.parametrize("mode", ["omega", "phi"])
@pytest.mark.parametrize("fold", list(_FOLD_OVER))
def test_metropolis_delta_matches_active_objective(mode, fold):
    m = _metro_model(mode, **_FOLD_OVER[fold])
    tok = _metro_tokens()
    context = m._metropolis_prepare(tok, mode=mode)
    tid = int(torch.unique(tok)[1])                             # a non-first token, genuinely moved
    move = m._metropolis_delta_f(context, tid, mode=mode)
    oracle = _oracle_delta(m, context, tid, mode=mode)
    assert abs(move) > 0.0, f"[{mode}/{fold}] delta is vacuously zero"
    assert abs(move - oracle) < 1e-8, f"[{mode}/{fold}] move={move} oracle={oracle}"


# --------------------------------------------------------------------------------------------------
# Step 1: exact-delta parity across the active TRANSPORT numerics (flat / RoPE-on-cov /
# RoPE-decoupled-value for both modes; regime_ii variants for phi only -- omega-direct is flat-only).
# --------------------------------------------------------------------------------------------------
_TCFG = {
    "flat":           dict(),
    "rope_decoupled": dict(pos_rotation="rope", rope_on_value=False),
    "rope_on_cov":    dict(pos_rotation="rope", rope_full_gauge=True,
                           family="gaussian_full", decode_mode="full"),
}


@pytest.mark.parametrize("mode", ["omega", "phi"])
@pytest.mark.parametrize("tcfg", list(_TCFG))
def test_metropolis_delta_matches_active_objective_transport(mode, tcfg):
    m = _metro_model(mode, **_TCFG[tcfg])
    tok = _metro_tokens()
    context = m._metropolis_prepare(tok, mode=mode)
    if tcfg.startswith("rope"):
        assert context.rope is not None                        # the positional RoPE tensor is captured
    tid = int(torch.unique(tok)[1])
    move = m._metropolis_delta_f(context, tid, mode=mode)
    oracle = _oracle_delta(m, context, tid, mode=mode)
    assert abs(move - oracle) < 1e-8, f"[{mode}/{tcfg}] move={move} oracle={oracle}"


@pytest.mark.parametrize("tmode", ["regime_ii", "regime_ii_covariant",
                                   "regime_ii_link", "regime_ii_link_charted"])
def test_metropolis_delta_matches_active_objective_regime_ii_phi_only(tmode):
    # omega-direct Metropolis is flat-only, so the non-flat connection regimes are phi-reflection only.
    m = _metro_model("phi", transport_mode=tmode)
    tok = _metro_tokens()
    context = m._metropolis_prepare(tok, mode="phi")
    tid = int(torch.unique(tok)[1])
    move = m._metropolis_delta_f(context, tid, mode="phi")
    oracle = _oracle_delta(m, context, tid, mode="phi")
    assert abs(move - oracle) < 1e-8, f"[{tmode}] move={move} oracle={oracle}"


# --------------------------------------------------------------------------------------------------
# Step 1: the query-adaptive tau is the final-block ENTRY-derived tau (not recomputed from the
# converged sigma), captured in MStepCapture['final_block_tau'].
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["omega", "phi"])
def test_metropolis_tau_is_final_block_entry_derived(mode):
    m = _metro_model(mode, query_adaptive_tau=True, query_tau_c=0.9, n_e_steps=3, e_q_sigma_lr=0.5)
    tok = _metro_tokens()
    context = m._metropolis_prepare(tok, mode=mode)
    dev = context.belief.mu.device
    base_tau = attention_tau(m.effective_kappa_beta(dev), m.group.irrep_dims)
    diag: dict = {}
    with torch.no_grad():
        m.forward_beliefs(tok, capture={"diagnostic": diag})
    entry_sigma = diag["initial_belief"].sigma                 # belief ENTERING the (final=only) block
    expected = query_adaptive_tau(entry_sigma, base_tau, m.group.irrep_dims, c=m.cfg.query_tau_c)
    torch.testing.assert_close(context.tau, expected)          # tau is entry-derived
    conv_tau = query_adaptive_tau(context.belief.sigma, base_tau, m.group.irrep_dims,
                                  c=m.cfg.query_tau_c)
    assert not torch.allclose(context.belief.sigma, entry_sigma)   # the E-step moved sigma
    assert not torch.allclose(context.tau, conv_tau)           # NOT recomputed from converged sigma


# --------------------------------------------------------------------------------------------------
# Step 1: with a nonzero mean/sigma handoff (n_layers=2), the scorer prior is the FINAL-block
# handoff-adjusted prior, NOT the encode-time prior.
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["omega", "phi"])
def test_metropolis_uses_final_block_handoff_prior(mode):
    m = _metro_model(mode, n_layers=2, prior_handoff_rho=0.6, prior_handoff_sigma=0.4)
    tok = _metro_tokens()
    context = m._metropolis_prepare(tok, mode=mode)
    cap: dict = {}
    with torch.no_grad():
        m.forward_beliefs(tok, capture=cap)
    fbp_mu, fbp_sigma = cap["final_block_prior"]
    torch.testing.assert_close(context.mu_p, fbp_mu)           # == the handoff-adjusted final prior
    torch.testing.assert_close(context.sigma_p, fbp_sigma)
    assert not torch.allclose(context.mu_p, cap["prior"].mu)   # != the encode-time prior
    assert not torch.allclose(context.sigma_p, cap["prior"].sigma)


# --------------------------------------------------------------------------------------------------
# Step 1: the precision fold reads the FIXED pre-stack covariance (context), unequal to both the
# current and the trial belief covariance; the tied gamma fold moves with the proposed frame.
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["omega", "phi"])
def test_metropolis_precision_fold_uses_context_not_belief(mode):
    m = _metro_model(mode, precision_weighted_attention=True, precision_attention_b0=1.5,
                     n_e_steps=3, e_q_sigma_lr=0.5, **_GAMMA_OVER)
    tok = _metro_tokens()
    context = m._metropolis_prepare(tok, mode=mode)
    trial = m._metropolis_trial_belief(context.belief, context.token_ids,
                                       int(torch.unique(tok)[1]), mode=mode)
    # the captured precision covariance is the pre-stack sigma, unequal to current AND trial cov
    assert not torch.allclose(context.prior.precision_sigma, context.belief.sigma)
    assert not torch.allclose(context.prior.precision_sigma, trial.sigma)
    lp_cur = m._effective_beta_log_prior(context.belief, context.prior)
    lp_trial = m._effective_beta_log_prior(trial, context.prior)
    fin = torch.isfinite(lp_cur)
    assert not torch.allclose(lp_cur[fin], lp_trial[fin])      # tied gamma moves with the frame

    # precision ONLY (no gamma): the fold is frame-blind and reads the FIXED context sigma, so a frame
    # flip leaves the prior EXACTLY unchanged even though the trial belief carries a different frame.
    m2 = _metro_model(mode, precision_weighted_attention=True, precision_attention_b0=1.5,
                      n_e_steps=3, e_q_sigma_lr=0.5)
    ctx2 = m2._metropolis_prepare(tok, mode=mode)
    tr2 = m2._metropolis_trial_belief(ctx2.belief, ctx2.token_ids,
                                      int(torch.unique(tok)[1]), mode=mode)
    _assert_priors_equal(m2._effective_beta_log_prior(ctx2.belief, ctx2.prior),
                         m2._effective_beta_log_prior(tr2, ctx2.prior),
                         msg="precision-only fold must be frame-blind")


# --------------------------------------------------------------------------------------------------
# Step 1: free_energy_value forwards exp_fp64_mode / exp_fp64_norm_threshold to _transport (the fp64
# island now triggers identically in the active evaluator and the Metropolis oracle).
# --------------------------------------------------------------------------------------------------
def test_metropolis_fp64_island_forwarded_to_transport(monkeypatch):
    m = _metro_model("phi", exp_fp64_mode="norm", exp_fp64_norm_threshold=0.5)
    with torch.no_grad():
        m.prior_bank.phi_embed.mul_(4.0)                       # large ||M||_F -> above the threshold
    tok = _metro_tokens()
    context = m._metropolis_prepare(tok, mode="phi")

    import vfe3.inference.e_step as es
    real = es._transport
    seen: dict = {}

    def spy(*a, **k):
        seen.setdefault("mode", k.get("exp_fp64_mode"))
        seen.setdefault("thr", k.get("exp_fp64_norm_threshold"))
        return real(*a, **k)

    monkeypatch.setattr(es, "_transport", spy)
    m._metropolis_free_energy(context.belief, context, mode="phi")
    assert seen.get("mode") == "norm"                          # configured island key reaches _transport
    assert seen.get("thr") == 0.5


# --------------------------------------------------------------------------------------------------
# Step 1: phi_tilde model frame -> the belief-frame reflection is not consumed (tested reflection-OFF,
# because live config rejects phi_tilde + either Metropolis mode).
# --------------------------------------------------------------------------------------------------
def test_metropolis_scorer_prior_phi_tilde_invariant_reflection_off():
    m = _phi_tilde_model()
    tokens = _tokens(m, n=3)
    ctx, initial, _ = _forward_capture(m, tokens)
    injected = initial._replace(reflection=torch.where(
        torch.arange(initial.mu.shape[-2], device=DEVICE) % 2 == 0, -1.0, 1.0
    ).expand(initial.mu.shape[:-1]).clone())
    _assert_priors_equal(m._effective_beta_log_prior(initial, ctx),
                         m._effective_beta_log_prior(injected, ctx),
                         msg="phi_tilde model frame must ignore the belief reflection")


# --------------------------------------------------------------------------------------------------
# Step 1: caller-contract -- _metropolis_prepare returns one context whose belief has shape (B,N,K),
# and the sweep's FIRST scorer call receives that same current belief before any proposal is applied.
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["omega", "phi"])
def test_metropolis_prepare_caller_contract(mode, monkeypatch):
    m = _metro_model(mode)
    tok = _metro_tokens()
    context = m._metropolis_prepare(tok, mode=mode)
    assert isinstance(context, MetropolisObjectiveContext)
    B, N = tok.shape
    K = m.cfg.embed_dim
    assert context.belief.mu.shape == (B, N, K)
    assert context.belief.sigma.shape == (B, N, K)

    captured: dict = {}
    real_prep = m._metropolis_prepare

    def prep_spy(token_ids, *, mode=None):
        c = real_prep(token_ids, mode=mode)
        captured["ctx"] = c
        return c

    seen: list = []
    real_fe = m._metropolis_free_energy

    def fe_spy(belief, ctx, *, mode=None):
        seen.append(belief)
        return real_fe(belief, ctx, mode=mode)

    monkeypatch.setattr(m, "_metropolis_prepare", prep_spy)
    monkeypatch.setattr(m, "_metropolis_free_energy", fe_spy)
    m.metropolis_omega_step(tok, generator=torch.Generator().manual_seed(0))
    assert seen and seen[0] is captured["ctx"].belief          # first F call = current, pre-proposal


# --------------------------------------------------------------------------------------------------
# Step 2: acceptance-boundary -- the corrected (folded) delta changes the accept/reject result vs the
# raw-prior scorer. The tied-gamma fold flips token-0's delta sign (verified), so at tiny T the
# corrected objective decides opposite to the raw one; the sweep must follow the CORRECTED decision.
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["omega", "phi"])
def test_metropolis_acceptance_boundary_corrected_vs_raw(mode):
    found = None
    tok = _metro_tokens()
    for seed in range(40):
        m = _metro_model(mode, seed=seed, **_GAMMA_OVER)
        context = m._metropolis_prepare(tok, mode=mode)
        tid0 = int(torch.unique(tok).min())
        dF_corr = m._metropolis_delta_f(context, tid0, mode=mode)
        dF_raw = _raw_delta(m, context, tid0, mode=mode)
        if (dF_corr <= 0.0) != (dF_raw <= 0.0):                # opposite accept at tiny T
            found = (m, context, tid0, dF_corr, dF_raw)
            break
    assert found is not None, f"[{mode}] no seed produced a corrected-vs-raw accept flip within budget"
    m, context, tid0, dF_corr, dF_raw = found

    m.cfg.omega_metropolis_temperature = 1e-6                  # tiny T: accept iff dF <= 0
    before = _src_sign_state(m, tid0, mode)
    gen = torch.Generator().manual_seed(0)
    n_unique = int(torch.unique(tok).numel())
    m.metropolis_omega_step(tok, generator=gen)
    after = _src_sign_state(m, tid0, mode)
    flipped = before * after < 0.0
    assert flipped == (dF_corr <= 0.0)                         # sweep followed the CORRECTED objective
    assert (dF_corr <= 0.0) != (dF_raw <= 0.0)                 # ... and the raw scorer would decide otherwise
    ref = torch.Generator().manual_seed(0)                     # RNG advanced once per proposal
    for _ in range(n_unique):
        torch.rand((), generator=ref)
    assert torch.equal(gen.get_state(), ref.get_state())


# --------------------------------------------------------------------------------------------------
# Step 3: off-path identity -- both reflection modes off -> the scorer is NEVER invoked; folds off ->
# the refactored delta equals the pre-Task-3 raw-prior calculation EXACTLY.
# --------------------------------------------------------------------------------------------------
def test_metropolis_scorer_not_invoked_when_reflection_off(monkeypatch):
    m = _model()                                               # plain model, both reflection modes off
    tok = _tokens(m, n=3)

    def _boom(*a, **k):
        raise AssertionError("the Metropolis scorer must not be invoked with reflection off")

    monkeypatch.setattr(m, "_metropolis_free_energy", _boom)
    monkeypatch.setattr(m, "_metropolis_prepare", _boom)
    assert m.metropolis_omega_step(tok, generator=torch.Generator().manual_seed(0)) == {}


@pytest.mark.parametrize("mode", ["omega", "phi"])
def test_metropolis_folds_off_delta_equals_raw_prior(mode):
    # No folds, n_layers=1, flat, no RoPE, scalar tau -> the effective prior is the RAW attention
    # prior and the final-block prior is the encode prior, so the refactored delta must equal the
    # pre-Task-3 raw-prior calculation to float round-off.
    m = _metro_model(mode)
    tok = _metro_tokens()
    context = m._metropolis_prepare(tok, mode=mode)
    tid = int(torch.unique(tok)[1])
    move = m._metropolis_delta_f(context, tid, mode=mode)
    raw = _raw_delta(m, context, tid, mode=mode)
    assert abs(move - raw) < 1e-9, f"[{mode}] refactored={move} raw={raw}"


# ==================================================================================================
# Task 4 (audit PB-12): the final CPU combination matrix and the scope boundary. Mean, covariance,
# phi, and reflection all descend the ONE objective free_energy_value evaluates; these cells close the
# remaining crossed axes -- the tied-gamma fold folded onto every valid nonflat transport, the private
# accept/reject generator surviving a checkpoint round-trip, and the RTX 5090 CUDA smoke (which skips
# off the GPU). The reflection straight-through estimator stays construction-time rejected (config
# raises on omega_reflection='ste' / phi_reflection='ste'), so it is out of scope here by design.
# ==================================================================================================
# The tied-gamma fold and each nonflat transport are pinned separately above; here they are CROSSED.
# omega-direct Metropolis is flat-only, so the non-flat connection regimes are phi-reflection only.
_NONFLAT_TCFG = {
    "rope_decoupled":         dict(pos_rotation="rope", rope_on_value=False),
    "rope_on_cov":            dict(pos_rotation="rope", rope_full_gauge=True,
                                   family="gaussian_full", decode_mode="full"),
    "regime_ii":              dict(transport_mode="regime_ii"),
    "regime_ii_covariant":    dict(transport_mode="regime_ii_covariant"),
    "regime_ii_link":         dict(transport_mode="regime_ii_link"),
    "regime_ii_link_charted": dict(transport_mode="regime_ii_link_charted"),
}


@pytest.mark.parametrize("tcfg", list(_NONFLAT_TCFG))
def test_metropolis_delta_matches_active_objective_tied_gamma_nonflat(tcfg):
    # phi Metropolis with the tied-gamma fold AND a non-flat transport still scores the exact active
    # DeltaF: the gamma fold reads the proposed frame while the transport bends the energy grid.
    m = _metro_model("phi", **_GAMMA_OVER, **_NONFLAT_TCFG[tcfg])
    assert m.cfg.gamma_as_beta_prior and m.cfg.s_frame_mode == "tied"
    tok = _metro_tokens()
    context = m._metropolis_prepare(tok, mode="phi")
    tid = int(torch.unique(tok)[1])
    move = m._metropolis_delta_f(context, tid, mode="phi")
    oracle = _oracle_delta(m, context, tid, mode="phi")
    assert abs(move) > 0.0, f"[gamma/{tcfg}] delta is vacuously zero"
    assert abs(move - oracle) < 1e-8, f"[gamma/{tcfg}] move={move} oracle={oracle}"


def test_metropolis_checkpoint_private_rng_continuation(tmp_path):
    # The private accept/reject generator is threaded across steps and checkpointed INDEPENDENTLY of the
    # global CPU/CUDA RNG (train.py builds it from cfg.seed before resume). A checkpoint save captures
    # generator.get_state(); load restores it into a fresh, differently-seeded generator, so a resumed
    # sweep continues the proposal draws byte-identically -- and the sweep itself never touches global RNG.
    from vfe3.run_artifacts import RunArtifacts, load_checkpoint
    from vfe3.train import build_optimizer

    m = _metro_model("phi")
    tok = _metro_tokens()
    gen = torch.Generator().manual_seed(11)
    global_before = torch.get_rng_state().clone()
    m.metropolis_omega_step(tok, generator=gen)                    # advance the private stream
    m.metropolis_omega_step(tok, generator=gen)
    assert torch.equal(torch.get_rng_state(), global_before)       # private stream never touched global RNG

    art = RunArtifacts(tmp_path / "run", m.cfg, m)
    opt = build_optimizer(m, m.cfg)
    saved = gen.get_state().clone()
    ckpt = art.save_checkpoint(1, m, opt, m.cfg, metropolis_generator=gen)

    resumed = torch.Generator().manual_seed(999)                   # deliberately different seed
    assert not torch.equal(resumed.get_state(), saved)
    load_checkpoint(ckpt, m, opt, metropolis_generator=resumed)
    assert torch.equal(resumed.get_state(), saved)                 # byte-identical private-RNG restore

    a = [torch.rand((), generator=gen).item() for _ in range(4)]
    b = [torch.rand((), generator=resumed).item() for _ in range(4)]
    assert a == b                                                  # the accept-draw continuation is identical


# --------------------------------------------------------------------------------------------------
# Step 4: RTX 5090 CUDA smoke. Runs ONLY under VFE3_TEST_DEVICE=cuda on a CUDA host; skips on CPU.
# The K=4 omega-Metropolis objective (two-hop + tied-gamma + precision folds) evaluates end-to-end on
# the GPU with no device mismatch, the private accept generator drives a deterministic proposal
# sequence, and the global CPU/CUDA RNG streams are untouched. Awaits a GPU run (this host is CPU-only).
# --------------------------------------------------------------------------------------------------
@pytest.mark.skipif(DEVICE.type != "cuda" or not torch.cuda.is_available(),
                    reason="RTX 5090 CUDA smoke: set VFE3_TEST_DEVICE=cuda on a CUDA host")
def test_phi_reflection_objective_parity_cuda_smoke():
    m = _metro_model("omega", lambda_twohop=0.1, precision_weighted_attention=True,
                     precision_attention_b0=1.5, **_GAMMA_OVER)
    tok = _metro_tokens()
    context = m._metropolis_prepare(tok, mode="omega")

    # CUDA-resident objective tensors (no device mismatch). tau is a scalar temperature, not a tensor.
    for t in (context.belief.mu, context.belief.sigma, context.prior.base_log_prior,
              context.mu_p, context.sigma_p):
        assert t.device.type == "cuda"
    lp = m._effective_beta_log_prior(context.belief, context.prior)
    assert lp.device.type == "cuda"
    assert torch.isfinite(lp[torch.isfinite(lp)]).all()

    # Finite scores for the current belief and every trial flip.
    assert math.isfinite(m._metropolis_free_energy(context.belief, context, mode="omega"))
    for tid in torch.unique(tok).tolist():
        trial = m._metropolis_trial_belief(context.belief, context.token_ids, tid, mode="omega")
        assert math.isfinite(m._metropolis_free_energy(trial, context, mode="omega"))

    # Deterministic private-generator proposal sequence + global CPU/CUDA RNG isolation.
    cpu_before  = torch.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state_all()
    gen = torch.Generator().manual_seed(0)
    m.metropolis_omega_step(tok, generator=gen)
    ref = torch.Generator().manual_seed(0)
    for _ in range(int(torch.unique(tok).numel())):
        torch.rand((), generator=ref)
    assert torch.equal(gen.get_state(), ref.get_state())           # one deterministic draw per proposal
    assert torch.equal(torch.get_rng_state(), cpu_before)          # global CPU RNG untouched
    assert all(torch.equal(a, b)
               for a, b in zip(torch.cuda.get_rng_state_all(), cuda_before))   # global CUDA RNG untouched
