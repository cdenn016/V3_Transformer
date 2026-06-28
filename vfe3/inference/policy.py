r"""Active-inference Expected-Free-Energy (EFE) policy-scorer seam for VFE_3.0.

This module is the registry machinery for the opt-in, no-grad, default-off EFE token-continuation
policy scorer specified in
``docs/superpowers/specs/2026-06-28-active-inference-efe-policy-scorer-spec.md``. It follows the
project's mandated add-by-registering pattern (cf. ``vfe3/alpha_i.py``): a module dict, a
``register_*`` decorator, and a ``get_*`` lookup that raises ``KeyError`` with the available keys.

Three orthogonal registries, each a config-selected seam:
  - ``_POLICIES``     -- policy_mode: how a candidate menu is scored (none | logprob_control |
                         efe_one_step | efe_rollout). ``none`` is the default and is NEVER dispatched
                         (``generate`` short-circuits before any lookup); it exists only so config
                         validation accepts the default. The concrete scorers are registered by the
                         later build phases (Phase 1+), each slotting in by registration alone.
  - ``_PREFERENCES``  -- policy_preference: the goal preference p(o | C) (task | held_out_predictive |
                         flat). Registered by Phase 1; read only when policy_mode != 'none'.
  - ``_AMBIGUITIES``  -- the ambiguity / epistemic estimator E_{q(s|pi)} H[p(o|s)] (likelihood_entropy
                         default; the sigma_mc variant is gated behind the pre-registered
                         sigma-validation gate of the spec, Sections 2.7 / 4.5). Registered by Phase 1.

Phase 0 (this commit) ships only the registry machinery plus the ``none`` placeholder, so the seam
exists and the default config validates. No scorer, preference, or ambiguity body, no PolicyScore
return type, and no learned parameter is created here yet; those arrive with the Phase 1 scorer.
"""

import math
from typing import Callable, Dict, NamedTuple, Optional, Tuple

import torch


_POLICIES:    Dict[str, Callable] = {}
_PREFERENCES: Dict[str, Callable] = {}
_AMBIGUITIES: Dict[str, Callable] = {}


def register_policy(name: str) -> Callable:
    """Decorator registering a policy scorer under ``name`` (cf. :func:`vfe3.alpha_i.register_alpha`)."""
    def _wrap(fn: Callable) -> Callable:
        _POLICIES[name] = fn
        return fn
    return _wrap


def get_policy(name: str) -> Callable:
    """Return the registered policy scorer (KeyError with the available keys if absent)."""
    if name not in _POLICIES:
        raise KeyError(f"no policy mode {name!r}; available: {sorted(_POLICIES)}")
    return _POLICIES[name]


def register_preference(name: str) -> Callable:
    """Decorator registering a preference p(o | C) builder under ``name``."""
    def _wrap(fn: Callable) -> Callable:
        _PREFERENCES[name] = fn
        return fn
    return _wrap


def get_preference(name: str) -> Callable:
    """Return the registered preference builder (KeyError with the available keys if absent)."""
    if name not in _PREFERENCES:
        raise KeyError(f"no preference {name!r}; available: {sorted(_PREFERENCES)}")
    return _PREFERENCES[name]


def register_ambiguity(name: str) -> Callable:
    """Decorator registering an ambiguity / epistemic estimator under ``name``."""
    def _wrap(fn: Callable) -> Callable:
        _AMBIGUITIES[name] = fn
        return fn
    return _wrap


def get_ambiguity(name: str) -> Callable:
    """Return the registered ambiguity estimator (KeyError with the available keys if absent)."""
    if name not in _AMBIGUITIES:
        raise KeyError(f"no ambiguity {name!r}; available: {sorted(_AMBIGUITIES)}")
    return _AMBIGUITIES[name]


@register_policy("none")
def _policy_none(*args, **kwargs):
    """The default, never-dispatched policy: ``generate`` short-circuits to its verbatim
    pre-existing body when ``policy_mode == 'none'``, so this is never called. It exists only so
    config validation accepts the default and the pure path stays byte-identical."""
    raise RuntimeError(
        "policy_mode='none' is never dispatched; generate() runs its verbatim pre-existing path. "
        "This callable exists only so config validation accepts the default."
    )


# ======================================================================================
# Phase 1: the one-step EFE scorer (return type, preferences, ambiguity, scorers).
#
# Honest scope (spec Section 2.8): at the v1 operating regime (horizon=1, sigma-free point belief)
# the expected-information-gain term I is identically zero, so the EFE score collapses to the
# pragmatic cross-entropy and the scorer is a preference-matching reranker. `PolicyScore.epistemic`
# is therefore returned as exact zeros at v1; the epistemic machinery is present, logged, and marked
# inert rather than presented as a live signal. Everything here runs under the caller's @torch.no_grad
# scope (generate / rollout_beliefs); no learned parameter is created.
# ======================================================================================


class PolicyScore(NamedTuple):
    """The scorer return: the active score and every diagnostic, kept strictly separate so the raw
    continuation log-prob is never folded into the metric the policy acts on (spec Section 3.2)."""

    score:            torch.Tensor   # (B, Kp) G(pi) = sum of the enabled score_terms
    risk:             torch.Tensor   # (B, Kp) KL[q(o|pi) || p(o|C)]            (pragmatic / preference)
    ambiguity:        torch.Tensor   # (B, Kp) E_{q(s|pi)} H[p(o|s)]            (= predictive entropy at v1)
    epistemic:        torch.Tensor   # (B, Kp) MI bridge I; IDENTICALLY 0 at v1 (logged separately)
    log_prob:         torch.Tensor   # (B, Kp) raw continuation log-prob       (logged SEPARATELY)
    policy_posterior: torch.Tensor   # (B, Kp) softmax(-gamma * score + log_prior)


# ---- preference registry: p(o | C) builders, each returning log-probabilities (V,) or (B, V) -------

@register_preference("flat")
def _pref_flat(
    prior_bank: 'object',                            # PriorBank: vocab handle (V)

    *,
    eps: float = 1e-12,
    **kwargs,
) -> torch.Tensor:                                   # (V,) log p(o|C) = -log V (uniform)
    r"""The uniform preference (limit beta_C -> 0). Carries no goal; the pure-epistemic ablation. By
    spec Section 2.3 the full score is G = log V - I, which at the v1 point belief is the constant
    log V (a uniform policy posterior). Reported, not gated, at v1."""
    V = prior_bank.vocab_size
    return torch.full((V,), -math.log(V))


@register_preference("task")
def _pref_task(
    prior_bank: 'object',                            # PriorBank: vocab handle (V)

    *,
    goal:    'int | torch.Tensor',                   # goal token id (scalar) or (B,) per-episode goals
    beta_C:  float = 5.0,                             # preference precision (spec: 5.0 -> goal mass ~0.90)
    support: Optional[torch.Tensor] = None,          # (S,) allowed (state) token ids; mass ~0 elsewhere
    eps:     float = 1e-12,
    **kwargs,
) -> torch.Tensor:                                   # (V,) or (B, V) log p(o|C) = log softmax(beta_C U_C)
    r"""The explicit, peaked goal preference p(o|C) = softmax(beta_C U_C): utility beta_C on the goal
    symbol, 0 on other ``support`` (state) symbols, and ~0 mass (-inf utility) on non-support tokens.
    The genuine pragmatic-EFE arm (spec Section 2.3). Per-episode: pass a (B,) ``goal`` for a (B, V)
    preference whose peak differs per episode."""
    V = prior_bank.vocab_size
    goal_t = goal if isinstance(goal, torch.Tensor) else torch.tensor(goal)
    if goal_t.dim() == 0:                            # scalar goal -> (V,)
        U = torch.zeros(V) if support is None else torch.full((V,), float("-inf"))
        if support is not None:
            U[support] = 0.0
        U[goal_t] = beta_C
        return torch.log_softmax(U, dim=-1)
    B = goal_t.shape[0]                              # (B,) goals -> (B, V)
    U = torch.zeros(B, V) if support is None else torch.full((B, V), float("-inf"))
    if support is not None:
        U[:, support] = 0.0
    U[torch.arange(B), goal_t] = beta_C
    return torch.log_softmax(U, dim=-1)


@register_preference("held_out_predictive")
def _pref_held_out_predictive(
    prior_bank: 'object',                            # PriorBank (unused; signature parity)

    *,
    p_data: torch.Tensor,                            # (V,) or (B, V) data distribution p_data(o)
    eps:    float = 1e-12,
    **kwargs,
) -> torch.Tensor:                                   # log p_data
    r"""The held-out-predictive preference p(o|C) = p_data(o), making risk reduce to next-observation
    NLL (spec Section 2.3). The control arm the peaked-preference EFE must beat; cannot steer a
    per-episode goal, which is what makes it the control."""
    return p_data.clamp_min(eps).log()


# ---- ambiguity registry: E_{q(s|pi)} H[p(o|s)] estimators -------------------------------------------

@register_ambiguity("likelihood_entropy")
def _amb_likelihood_entropy(
    q_log: torch.Tensor,                             # (B, Kp, V) log p(o | mu_s) at the belief MEAN

    **kwargs,
) -> torch.Tensor:                                   # (B, Kp) H[p(o | mu_s)]
    r"""The default ambiguity: the entropy of the decoded predictive categorical at the belief MEAN,
    using no sigma (spec Section 3.3). At the v1 point belief q(o|pi) = p(o|mu_s), so this equals the
    predictive entropy and the MI bridge I = predictive_entropy - ambiguity is identically 0."""
    return -(q_log.exp() * q_log).sum(dim=-1)


@register_ambiguity("sigma_mc")
def _amb_sigma_mc(
    q_log: torch.Tensor,

    **kwargs,
) -> torch.Tensor:
    r"""The sigma-dependent Monte-Carlo ambiguity E_{mu^(s)~N(mu,Sigma)} H[p(o|mu^(s))] (spec Sections
    2.7, 4.5). GATED: it is unlocked only after the pre-registered sigma-validation gate passes; until
    then it must never be dispatched, so a trained nonzero sigma cannot be called an ambiguity value."""
    raise RuntimeError(
        "ambiguity='sigma_mc' is gated behind the sigma-validation gate (spec Sections 2.7/4.5): set "
        "policy_sigma_ambiguity_validated=True with a matching PASS artifact before it can be used."
    )


# ---- scorer internals ------------------------------------------------------------------------------

def _rollout_predictive(
    context:     torch.Tensor,                       # (B, N) context ids
    candidates:  torch.Tensor,                       # (B, Kp, L) candidate continuation ids

    model:       'object',                            # VFEModel: rollout_beliefs / forward / prior_bank

    *,
    base_logits: Optional[torch.Tensor] = None,      # (B, V) base last-position logits (reused if given)
) -> 'Tuple[torch.Tensor, torch.Tensor]':
    r"""Batched one-step rollout. For each candidate appends its ACTION token(s) to the context and
    rolls the belief forward through the shared seam, returning the predicted outcome distribution
    q(o|pi) = p(o | q*_pi) as log-probs (B, Kp, V) and the raw continuation log-prob (B, Kp) of the
    first action token under the BASE predictive. The environment response is NOT folded into the
    rollout (spec Section 2.2). All Kp candidates run in one batched forward (B*Kp sequences)."""
    B, N = context.shape
    Kp, L = candidates.shape[1], candidates.shape[2]
    max_len = model.cfg.max_seq_len
    if N + L > max_len:                              # keep context+action within the model's built length
        context = context[:, -(max_len - L):]
        N = context.shape[1]
    ctx_exp = context.unsqueeze(1).expand(B, Kp, N)              # (B, Kp, N)
    ext = torch.cat([ctx_exp, candidates], dim=2).reshape(B * Kp, N + L)   # (B*Kp, N+L)
    _belief, logits = model.rollout_beliefs(ext, return_logits=True)       # logits (B*Kp, N+L, V)
    last = logits[:, -1, :]                                      # (B*Kp, V) post-action prediction
    q_log = torch.log_softmax(last, dim=-1).reshape(B, Kp, -1)  # (B, Kp, V) = log q(o|pi)
    if base_logits is None:
        base_logits = model.forward(context)[:, -1, :]          # (B, V) base last-position logits
    base_logp = torch.log_softmax(base_logits, dim=-1)          # (B, V)
    log_prob = torch.gather(base_logp, 1, candidates[:, :, 0])  # (B, Kp) logprob of the first action
    return q_log, log_prob


def _efe_terms(
    q_log:      torch.Tensor,                        # (B, Kp, V) log q(o|pi)
    preference: torch.Tensor,                        # (V,) or (B, V) log p(o|C)
) -> 'Tuple[torch.Tensor, torch.Tensor]':
    r"""risk = KL[q(o|pi) || p(o|C)] and the predictive entropy H[q(o|pi)] (spec Section 2.6)."""
    q = q_log.exp()
    logpC = preference.view(1, 1, -1) if preference.dim() == 1 else preference.unsqueeze(1)
    risk = (q * (q_log - logpC)).sum(dim=-1)                     # (B, Kp) forward KL
    pred_ent = -(q * q_log).sum(dim=-1)                          # (B, Kp) H[q]
    return risk, pred_ent


def _policy_posterior(
    score:     torch.Tensor,                         # (B, Kp) G(pi)
    gamma:     float,
    log_prior: Optional[torch.Tensor],               # (B, Kp) log E(pi); None -> uniform
) -> torch.Tensor:
    r"""Q(pi) = softmax_pi(-gamma G(pi) + log E)."""
    logits = -gamma * score
    if log_prior is not None:
        logits = logits + log_prior
    return torch.softmax(logits, dim=-1)


# ---- policy registry: the scorers ------------------------------------------------------------------

@register_policy("efe_one_step")
def _policy_efe_one_step(
    context:     torch.Tensor,                       # (B, N) context ids                          -> D
    candidates:  torch.Tensor,                       # (B, Kp, L) candidate continuations          -> E
    preference:  torch.Tensor,                       # (V,) or (B, V) log p(o|C)                   -> C

    model:       'object',                            # VFEModel: belief seam + prior_bank          -> A,B

    *,
    gamma:         float               = 1.0,         # policy precision in softmax(-gamma G)
    horizon:       int                 = 1,           # fixed rollout depth H (efe_one_step is H=1)
    score_terms:   Tuple[str, ...]     = ("risk", "ambiguity"),  # which terms enter G(pi)
    ambiguity_mode: str                = "likelihood_entropy",   # ambiguity registry key
    log_prior:     Optional[torch.Tensor] = None,    # (B, Kp) log candidate prior E; None -> uniform
    base_logits:   Optional[torch.Tensor] = None,    # (B, V) reused base logits (avoid a duplicate fwd)
    **kwargs,
) -> 'PolicyScore':
    r"""The v1 one-step EFE scorer. G(pi) = risk(pi) + ambiguity(pi) = KL[q(o|pi)||p(o|C)] +
    E_{q(s|pi)} H[p(o|s)]. At the v1 default (horizon=1, sigma-free point belief) the information-gain
    I is identically 0 (spec Section 2.8): G reduces to the pragmatic cross-entropy and `epistemic` is
    returned as exact zeros. ``score_terms`` selects which terms enter G, so risk-only / ambiguity-only
    / flat-preference reductions are recoverable without a code change."""
    if horizon != 1:
        raise ValueError(
            f"efe_one_step is the H=1 scorer; got horizon={horizon}. Horizon>1 is efe_rollout, which "
            f"is gated on a belief/key-value cache (spec Section 3.5).")
    q_log, log_prob = _rollout_predictive(context, candidates, model, base_logits=base_logits)
    risk, pred_ent = _efe_terms(q_log, preference)
    ambiguity = get_ambiguity(ambiguity_mode)(q_log, model=model)
    epistemic = pred_ent - ambiguity                            # MI bridge; ==0 at v1 (likelihood_entropy)
    terms = {"risk": risk, "ambiguity": ambiguity, "epistemic": -epistemic}
    score = sum(terms[t] for t in score_terms)
    post = _policy_posterior(score, gamma, log_prior)
    return PolicyScore(score, risk, ambiguity, epistemic, log_prob, post)


@register_policy("logprob_control")
def _policy_logprob_control(
    context:     torch.Tensor,                       # (B, N) context ids
    candidates:  torch.Tensor,                       # (B, Kp, L) candidate continuations
    preference:  torch.Tensor,                       # (V,) or (B, V) (unused; signature parity)

    model:       'object',                            # VFEModel

    *,
    gamma:       float                  = 1.0,
    horizon:     int                    = 1,
    score_terms: Tuple[str, ...]        = ("risk", "ambiguity"),  # unused; parity
    log_prior:   Optional[torch.Tensor] = None,
    base_logits: Optional[torch.Tensor] = None,
    **kwargs,
) -> 'PolicyScore':
    r"""The matched-compute control: pays the SAME Kp rollout cost as efe_one_step but scores by raw
    continuation log-prob (score = -log_prob), with risk/ambiguity/epistemic returned as zeros so the
    diagnostic columns line up (spec Section 3.2)."""
    q_log, log_prob = _rollout_predictive(context, candidates, model, base_logits=base_logits)
    zeros = torch.zeros_like(log_prob)
    score = -log_prob                                           # lower G <-> higher continuation logprob
    post = _policy_posterior(score, gamma, log_prior)
    return PolicyScore(score, zeros, zeros, zeros, log_prob, post)


@register_policy("efe_rollout")
def _policy_efe_rollout(*args, **kwargs):
    r"""The staged horizon extension (horizon>1). GATED: hard-errors until a belief/key-value cache
    exists, because without incremental belief reuse the Kp*H recompute makes the compute-matched
    baselines decisive and the wall-clock honesty check unwinnable (spec Sections 3.5, 4.2)."""
    raise NotImplementedError(
        "efe_rollout (horizon>1) is gated on a belief/key-value cache (spec Section 3.5); it is "
        "deferred to the epistemic-live phase. Use efe_one_step (horizon=1) for v1."
    )
