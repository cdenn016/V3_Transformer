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
import os
import warnings

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.contracts import EffectiveBetaPriorContext
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
