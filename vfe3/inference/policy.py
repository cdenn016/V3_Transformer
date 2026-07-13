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

from vfe3.contracts import AmbiguityEstimate, PolicyRollout
from vfe3.inference.belief_cache import cache_supported, rollout_predictive_cached, rollout_predictive_state_cached
from vfe3.numerics import safe_cholesky


_POLICIES:    Dict[str, Callable] = {}
_PREFERENCES: Dict[str, Callable] = {}
_AMBIGUITIES: Dict[str, Callable] = {}

# Preferences usable in the GENERIC generate() policy path: those needing no per-episode context (no
# goal, no p_data). The config guard (audit F4) is FAIL-CLOSED against this allow-list, so a
# context-requiring preference -- 'task' (needs a goal), 'held_out_predictive' (needs p_data), or any
# future @register_preference whose context arg has no default -- is rejected rather than slipping
# through to a mid-generate TypeError. Those preferences are driven through a harness that calls the
# scorer directly (the ring experiment keeps policy_mode='none'). Adding a new context-free preference
# here is the explicit opt-in.
_GENERATE_SAFE_PREFERENCES: frozenset = frozenset({"flat"})


def register_policy(name: str, *, override: bool = False) -> Callable:
    """Decorator registering a policy scorer under ``name`` (cf. :func:`vfe3.alpha_i.register_alpha`).
    Duplicate keys fail closed (audit F12): re-registering an existing ``name`` raises ``KeyError``
    unless ``override=True``, so a second registration cannot silently shadow the first."""
    def _wrap(fn: Callable) -> Callable:
        if name in _POLICIES and not override:
            raise KeyError(f"policy mode {name!r} already registered; pass override=True to replace")
        _POLICIES[name] = fn
        return fn
    return _wrap


def get_policy(name: str) -> Callable:
    """Return the registered policy scorer (KeyError with the available keys if absent)."""
    if name not in _POLICIES:
        raise KeyError(f"no policy mode {name!r}; available: {sorted(_POLICIES)}")
    return _POLICIES[name]


def register_preference(name: str, *, override: bool = False) -> Callable:
    """Decorator registering a preference p(o | C) builder under ``name``.

    Duplicate keys fail closed (audit 2026-07-01 round-3): a second registration under an
    existing name silently shadowed the first. Pass ``override=True`` to replace deliberately.
    """
    def _wrap(fn: Callable) -> Callable:
        if name in _PREFERENCES and not override:
            raise KeyError(f"preference {name!r} already registered; pass override=True to replace")
        _PREFERENCES[name] = fn
        return fn
    return _wrap


def get_preference(name: str) -> Callable:
    """Return the registered preference builder (KeyError with the available keys if absent)."""
    if name not in _PREFERENCES:
        raise KeyError(f"no preference {name!r}; available: {sorted(_PREFERENCES)}")
    return _PREFERENCES[name]


def register_ambiguity(name: str, *, override: bool = False) -> Callable:
    """Decorator registering an ambiguity / epistemic estimator under ``name``.

    Duplicate keys fail closed (audit 2026-07-01 round-3): a second registration under an
    existing name silently shadowed the first. Pass ``override=True`` to replace deliberately.
    """
    def _wrap(fn: Callable) -> Callable:
        if name in _AMBIGUITIES and not override:
            raise KeyError(f"ambiguity {name!r} already registered; pass override=True to replace")
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
    eps:    float                       = 1e-12,
    device: Optional[torch.device]      = None,      # model device (audit F5); None -> CPU (direct-call default)
    **kwargs,
) -> torch.Tensor:                                   # (V,) log p(o|C) = -log V (uniform)
    r"""The uniform preference (limit beta_C -> 0). Carries no goal; the pure-epistemic ablation. By
    spec Section 2.3 the full score is G = log V - I, which at the v1 point belief is the constant
    log V (a uniform policy posterior). Reported, not gated, at v1. ``device`` honors the model device
    so the generic generate() path does not build a CPU preference for CUDA scorer tensors (audit F5)."""
    V = prior_bank.vocab_size
    return torch.full((V,), -math.log(V), device=device)


@register_preference("task")
def _pref_task(
    prior_bank: 'object',                            # PriorBank: vocab handle (V)

    *,
    goal:          'int | torch.Tensor',             # goal token id (scalar) or (B,) per-episode goals
    beta_C:        float                  = 5.0,     # preference precision (spec: 5.0 -> goal mass ~0.90)
    eps:           float                  = 1e-12,
    support:       Optional[torch.Tensor] = None,    # (S,) allowed (state) token ids
    device:        Optional[torch.device] = None,    # model device (audit F5); None -> CPU
    support_floor: Optional[float]        = None,    # None -> hard -inf; finite -> explicit soft floor
    **kwargs,
) -> torch.Tensor:                                   # (V,) or (B, V) log p(o|C) = log softmax(beta_C U_C)
    r"""The explicit, peaked goal preference p(o|C) = softmax(beta_C U_C): utility beta_C on the goal
    symbol and 0 on other ``support`` (state) symbols. ``support_floor=None`` preserves exact hard
    support with -inf outside it; a supplied finite floor is an explicit finite-floor preference and is
    normalized with the other utilities. The genuine pragmatic-EFE arm (spec Section 2.3). Per-episode:
    pass a (B,) ``goal`` for a (B, V) preference whose peak differs per episode. ``device`` honors the
    model device (audit F5)."""
    if support_floor is not None and not math.isfinite(support_floor):
        raise ValueError(f"support_floor must be finite or None, got {support_floor}")
    V = prior_bank.vocab_size
    goal_t = goal.to(device) if isinstance(goal, torch.Tensor) else torch.tensor(goal, device=device)
    if support is not None and device is not None:
        support = support.to(device)
    floor = float("-inf") if support_floor is None else support_floor
    if goal_t.dim() == 0:                            # scalar goal -> (V,)
        U = torch.zeros(V, device=device) if support is None else torch.full((V,), floor, device=device)
        if support is not None:
            U[support] = 0.0
        U[goal_t] = beta_C
        return torch.log_softmax(U, dim=-1)
    B = goal_t.shape[0]                              # (B,) goals -> (B, V)
    U = torch.zeros(B, V, device=device) if support is None else torch.full((B, V), floor, device=device)
    if support is not None:
        U[:, support] = 0.0
    U[torch.arange(B, device=device), goal_t] = beta_C
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
) -> AmbiguityEstimate:                              # predictive_log_prob = q_log; entropy H[p(o|mu_s)]
    r"""The default ambiguity: the entropy of the decoded predictive categorical at the belief MEAN,
    using no sigma (spec Section 3.3). At the v1 point belief q(o|pi) = p(o|mu_s), so the predictive
    marginal is q_log ITSELF (returned unchanged) and this equals the predictive entropy, so the MI
    bridge I = predictive_entropy - ambiguity is identically 0. Returns an :class:`AmbiguityEstimate`
    so the scorer reads the same two fields for every arm; the four sigma-consumer identities and the
    (mu, sigma, model, num_samples) kwargs are ignored on this sigma-free path (source-compatible with
    every existing direct caller)."""
    q = q_log.exp()
    q_log_term = torch.where(q > 0, q * q_log, torch.zeros_like(q))
    entropy = -q_log_term.sum(dim=-1)                            # (B, Kp) H[p(o | mu_s)]
    return AmbiguityEstimate(predictive_log_prob=q_log, expected_conditional_entropy=entropy)


def _antithetic_shared_state_samples(
    mu:      torch.Tensor,                           # (B, Kp, K) terminal belief means
    sigma:   torch.Tensor,                           # (B, Kp, K)/(B, Kp, K, K) terminal covariance

    eps:         float,                              # covariance floor / Cholesky ridge (model.cfg.eps)

    *,
    num_samples: int,                                # S; MUST be 16 (sealed antithetic_shared_v1)
    mc_seed:     int = 0,                            # sealed local-generator seed
) -> torch.Tensor:                                   # (B, Kp, S, K) reparameterized state samples
    r"""The sealed ``antithetic_shared_v1`` reparameterized sampler (PB-06, spec Sections 2.7/4.5).

    Draws ``S // 2`` standard-normal draws on a LOCAL ``torch.Generator`` (seed ``mc_seed``) on the
    belief device -- SHARED across the candidate axis within each batch element (a size-1 Kp axis
    broadcast) so the global torch RNG is untouched and permuting candidates permutes the per-candidate
    estimate. The antithetic partner of each draw is its negative, concatenated POSITIVE-then-negative
    to give exactly ``S`` samples. Diagonal covariance uses ``sigma.clamp_min(eps).sqrt()``; full
    covariance uses ``safe_cholesky(sigma, eps=eps, rounds=5)`` (the repository jitter policy -- NEVER
    raw ``torch.linalg.cholesky``), reading its per-item ``ok`` mask and falling back to the point mean
    (delta = 0) for any item whose Cholesky is unrecoverable even after the jitter escalation, so a
    zero/near-singular covariance stays finite and converges to the point (likelihood_entropy) decode."""
    if num_samples != 16:
        raise ValueError(
            f"the sealed antithetic_shared_v1 sampler requires num_samples=16, got {num_samples}.")
    half = num_samples // 2
    B, Kp, K = mu.shape
    gen = torch.Generator(device=mu.device).manual_seed(mc_seed)
    z = torch.randn(B, 1, half, K, generator=gen, device=mu.device, dtype=mu.dtype)  # (B,1,half,K) shared
    mu_e = mu.unsqueeze(2)                                       # (B, Kp, 1, K)
    if sigma.dim() == mu.dim():                                  # diagonal (B, Kp, K)
        std = sigma.clamp_min(eps).sqrt().unsqueeze(2)          # (B, Kp, 1, K)
        delta = std * z                                          # (B, Kp, half, K)
    else:                                                        # full (B, Kp, K, K)
        L, ok = safe_cholesky(sigma, eps=eps, rounds=5)         # (B, Kp, K, K), ok (B, Kp)
        delta = (L.unsqueeze(2) @ z.unsqueeze(-1)).squeeze(-1)  # (B,Kp,1,K,K)@(B,1,half,K,1) -> (B,Kp,half,K)
        delta = torch.where(ok[..., None, None], delta, torch.zeros_like(delta))
    return torch.cat([mu_e + delta, mu_e - delta], dim=2)        # (B, Kp, S, K) positive-then-negative


@register_ambiguity("sigma_mc")
def _amb_sigma_mc(
    q_log: torch.Tensor,                             # (B, Kp, V) point predictive at the mean (unused here)

    *,
    mu:          torch.Tensor,                       # (B, Kp, K) terminal belief means
    sigma:       torch.Tensor,                       # (B, Kp, K)/(B, Kp, K, K) terminal covariance
    model:       object,                             # VFEModel: prior_bank (decode_point) + cfg
    num_samples: int,                                # S; MUST be 16
    model_behavior_sha256:      Optional[str] = None,
    spec_identity:              Optional[str] = None,
    code_identity_sha256:       Optional[str] = None,
    measurement_context_sha256: Optional[str] = None,
    **kwargs,
) -> AmbiguityEstimate:
    r"""The sigma-dependent Monte-Carlo ambiguity E_{s~N(mu,Sigma)} H[p(o|s)] with its antithetic
    shared-noise predictive marginal (spec Sections 2.7, 4.5; PB-06).

    GATED, fail-closed. Dispatch is authorized ONLY when all four live consumer-gate identities are
    supplied (a direct registry call with none -- ``get_ambiguity('sigma_mc')(q_log)`` -- raises) AND
    :func:`verify_sigma_consumer_gate` re-verifies the pre-registered artifact against them here, as
    defense in depth: the artifact is re-READ per sigma dispatch so a post-construction replacement
    fails closed, without rehashing the model/source/corpus on every generated token. It samples the
    terminal state through :func:`_antithetic_shared_state_samples`, decodes each sample to its outcome
    distribution via :meth:`PriorBank.decode_point`, and forms (i) the NORMALIZED predictive marginal
    q(o|pi) = E_s p(o|s) stably as ``log_softmax(logsumexp_s(log p) - log S)`` and (ii) the Monte-Carlo
    expected conditional entropy ``mean_s H[p(o|s)]`` with a ``where(prob > 0, ...)`` guard so exact
    ``-inf`` sample-log-prob tails never form ``0 * -inf`` NaNs. The B/Kp/S axes are never flattened."""
    if (model_behavior_sha256 is None or spec_identity is None
            or code_identity_sha256 is None or measurement_context_sha256 is None):
        raise RuntimeError(
            "ambiguity='sigma_mc' is dispatchable only under the fully verified sigma_mc consumer gate: "
            "all four derived identities (model_behavior_sha256, spec_identity, code_identity_sha256, "
            "measurement_context_sha256) are REQUIRED and are supplied only after VFEModel.generate "
            "verifies verify_sigma_consumer_gate (spec Sections 2.7/4.5). A direct registry dispatch "
            "without them fails closed; setting policy_sigma_ambiguity_validated=True alone does NOT "
            "unlock it.")
    from vfe3.inference import sigma_gate
    sigma_gate.verify_sigma_consumer_gate(
        model.cfg.policy_sigma_gate_artifact,
        actual_model_behavior_sha256=model_behavior_sha256,
        actual_spec_identity=spec_identity,
        actual_code_identity_sha256=code_identity_sha256,
        actual_measurement_context_sha256=measurement_context_sha256,
    )
    S = num_samples
    samples = _antithetic_shared_state_samples(
        mu, sigma, model.cfg.eps, num_samples=S)               # (B, Kp, S, K)
    logits = model.prior_bank.decode_point(samples)             # (B, Kp, S, V)
    sample_log_prob = torch.log_softmax(logits, dim=-1)         # (B, Kp, S, V) log p(o | s)
    # Predictive marginal q(o|pi) = mean_s p(o|s): logsumexp over samples minus log S, then a final
    # log_softmax to shed float normalization drift (the mixture is already normalized in exact math).
    predictive_log_prob = torch.logsumexp(sample_log_prob, dim=2) - math.log(S)   # (B, Kp, V)
    predictive_log_prob = torch.log_softmax(predictive_log_prob, dim=-1)
    # Expected conditional entropy mean_s H[p(o|s)]; the where-guard keeps 0*(-inf) tails NaN-free.
    sample_prob = sample_log_prob.exp()                         # (B, Kp, S, V)
    ent_term = torch.where(sample_prob > 0, sample_prob * sample_log_prob, torch.zeros_like(sample_prob))
    sample_entropy = -ent_term.sum(dim=-1)                      # (B, Kp, S) H[p(o|s)]
    expected_conditional_entropy = sample_entropy.mean(dim=2)  # (B, Kp) E_s H[p(o|s)]
    return AmbiguityEstimate(predictive_log_prob=predictive_log_prob,
                             expected_conditional_entropy=expected_conditional_entropy)


# ---- scorer internals ------------------------------------------------------------------------------

def _validate_policy_context(
    context:          torch.Tensor,                   # (B, N) current policy context

    candidate_length: int,                            # L appended action tokens
    max_seq_len:      int,                            # model sequence bound
) -> None:
    """Reject a policy rollout that cannot preserve its complete context and candidate."""
    N = context.shape[1]
    if N <= 0:
        raise ValueError(f"policy context must be nonempty, got N={N}")
    if candidate_length <= 0:
        raise ValueError(f"policy candidate length must be > 0, got L={candidate_length}")
    if N + candidate_length > max_seq_len:
        raise ValueError(
            f"policy context length N={N} plus candidate length L={candidate_length} exceeds "
            f"max_seq_len={max_seq_len}; policy paths do not truncate context."
        )


def _rollout_predictive_state(
    context:     torch.Tensor,                       # (B, N) context ids
    candidates:  torch.Tensor,                       # (B, Kp, L) candidate continuation ids

    model:       'object',                            # VFEModel: rollout_beliefs / forward / prior_bank

    *,
    base_logits: Optional[torch.Tensor] = None,      # (B, V) base last-position logits (reused if given)
) -> 'PolicyRollout':
    r"""Batched one-step rollout carrying the TERMINAL belief state (PB-06). For each candidate appends
    its ACTION token(s) to the context and rolls the belief forward through the shared seam, returning a
    :class:`PolicyRollout` with the predicted outcome distribution q(o|pi) = p(o | q*_pi) as log-probs
    (B, Kp, V), the raw continuation log-prob (B, Kp) of the first action token under the BASE
    predictive, and the terminal belief mean (B, Kp, K) and covariance (B, Kp, K[,K]) read from the LAST
    appended position of the returned ``BeliefState`` (post block_norm/final_norm). The environment
    response is NOT folded into the rollout (spec Section 2.2). All Kp candidates run in one batched
    forward (B*Kp sequences); the cache fast path engages on the supported config."""
    B, N = context.shape
    Kp, L = candidates.shape[1], candidates.shape[2]
    max_len = model.cfg.max_seq_len
    _validate_policy_context(context, L, max_len)
    # Cache fast path (Phase 3a): on the verified causal/filtering/flat/single-block regime, and when
    # context+candidate fits the built length (no sliding-window eviction, which would invalidate the
    # prefix), compute only the appended positions' E-step rows against a shared context. Golden-tested
    # equal to the full recompute below to float tolerance (tests/test_belief_cache.py).
    if cache_supported(model.cfg):
        return rollout_predictive_state_cached(context, candidates, model, base_logits=base_logits)
    ctx_exp = context.unsqueeze(1).expand(B, Kp, N)              # (B, Kp, N)
    ext = torch.cat([ctx_exp, candidates], dim=2).reshape(B * Kp, N + L)   # (B*Kp, N+L)
    belief, logits = model.rollout_beliefs(
        ext, return_logits=True, decode_last=True)                # logits (B*Kp, 1, V)
    last = logits[:, 0, :]                                        # (B*Kp, V) post-action prediction
    q_log = torch.log_softmax(last, dim=-1).reshape(B, Kp, -1)  # (B, Kp, V) = log q(o|pi)
    # Terminal belief moments at the last appended position (the returned belief already carries
    # block_norm/final_norm), reshaped candidate-major: mu (B, Kp, K), sigma (B, Kp, K[,K]).
    mu = belief.mu[:, -1].reshape(B, Kp, *belief.mu.shape[2:])
    sigma = belief.sigma[:, -1].reshape(B, Kp, *belief.sigma.shape[2:])
    if base_logits is None:
        base_logits = model.forward(context)[:, -1, :]          # (B, V) base last-position logits
    base_logp = torch.log_softmax(base_logits, dim=-1)          # (B, V)
    log_prob = torch.gather(base_logp, 1, candidates[:, :, 0])  # (B, Kp) logprob of the first action
    return PolicyRollout(q_log=q_log, log_prob=log_prob, mu=mu, sigma=sigma)


def _rollout_predictive(
    context:     torch.Tensor,                       # (B, N) context ids
    candidates:  torch.Tensor,                       # (B, Kp, L) candidate continuation ids

    model:       'object',                            # VFEModel: rollout_beliefs / forward / prior_bank

    *,
    base_logits: Optional[torch.Tensor] = None,      # (B, V) base last-position logits (reused if given)
) -> 'Tuple[torch.Tensor, torch.Tensor]':
    r"""Compatibility wrapper (PB-06): the historical two-tensor rollout return. Delegates to
    :func:`_rollout_predictive_state` and returns exactly ``(q_log, log_prob)`` so existing external
    unpacking is unchanged."""
    state = _rollout_predictive_state(context, candidates, model, base_logits=base_logits)
    return state.q_log, state.log_prob


def _efe_terms(
    q_log:      torch.Tensor,                        # (B, Kp, V) log q(o|pi)
    preference: torch.Tensor,                        # (V,) or (B, V) log p(o|C)
) -> 'Tuple[torch.Tensor, torch.Tensor]':
    r"""risk = KL[q(o|pi) || p(o|C)] and the predictive entropy H[q(o|pi)] (spec Section 2.6)."""
    q = q_log.exp()
    q_log_term = torch.where(q > 0, q * q_log, torch.zeros_like(q))
    logpC = preference.view(1, 1, -1) if preference.dim() == 1 else preference.unsqueeze(1)
    preference_term = torch.where(q > 0, q * logpC, torch.zeros_like(q))
    risk = (q_log_term - preference_term).sum(dim=-1)            # (B, Kp) forward KL; may be +inf
    pred_ent = -q_log_term.sum(dim=-1)                            # (B, Kp) H[q]
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
    invalid_row = torch.isnan(logits).any(dim=-1) | torch.isposinf(logits).any(dim=-1)
    if bool(invalid_row.any()):
        rows = invalid_row.nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(f"policy posterior logits contain NaN or +inf candidates in rows {rows}")
    finite_row = torch.isfinite(logits).any(dim=-1)
    if not bool(finite_row.all()):
        rows = (~finite_row).nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(f"policy posterior has no finite candidate in rows {rows}")
    return torch.softmax(logits, dim=-1)


def _efe_score(
    context:     torch.Tensor,                       # (B, N) context ids
    candidates:  torch.Tensor,                       # (B, Kp, L) candidate continuations (L = horizon)
    preference:  torch.Tensor,                       # (V,) or (B, V) log p(o|C)

    model:       'object',                            # VFEModel

    *,
    gamma:          float,
    score_terms:    Tuple[str, ...],
    ambiguity_mode: str,
    log_prior:      Optional[torch.Tensor],
    base_logits:    Optional[torch.Tensor],
    model_behavior_sha256:      Optional[str] = None,
    sigma_spec_identity:        Optional[str] = None,
    sigma_code_identity_sha256: Optional[str] = None,
    sigma_measurement_context_sha256: Optional[str] = None,
) -> 'PolicyScore':
    r"""Shared EFE scoring body for ``efe_one_step`` (H=1) and ``efe_rollout`` (H>1): roll the
    candidates forward (the cache fast path engages inside ``_rollout_predictive_state`` when supported),
    form risk + ambiguity, and return the policy posterior. The horizon distinction lives entirely in
    the candidate length L and the per-scorer guards; the scoring algebra is identical because the
    rollout always reads the LAST appended position's predictive q(o|pi). The ``sum`` below reduces
    over the enabled ``score_terms`` (risk/ambiguity/epistemic), NOT over timesteps: even at H > 1
    the terms are evaluated once, on the terminal predictive (audit F3).

    Under the gated ``sigma_mc`` ambiguity (PB-06) all four consumer-gate identities are REQUIRED (fail
    closed): a direct scorer call without them raises before the estimator, so a trained nonzero sigma
    cannot be called an ambiguity value without the validated gate. Every other registered ambiguity
    ignores the identities, so existing modes stay source-compatible."""
    valid_terms = ("risk", "ambiguity", "epistemic")
    if not score_terms:
        raise ValueError("score_terms must be nonempty so the EFE policy score remains a Tensor")
    invalid_terms = tuple(term for term in score_terms if term not in valid_terms)
    if invalid_terms:
        raise ValueError(f"score_terms contains unknown EFE terms {invalid_terms}; expected {valid_terms}")
    if ambiguity_mode == "sigma_mc" and (
            model_behavior_sha256 is None or sigma_spec_identity is None
            or sigma_code_identity_sha256 is None or sigma_measurement_context_sha256 is None):
        raise ValueError(
            "ambiguity_mode='sigma_mc' requires the validated consumer-gate identities "
            "(model_behavior_sha256, sigma_spec_identity, sigma_code_identity_sha256, "
            "sigma_measurement_context_sha256); they are supplied only after VFEModel.generate verifies "
            "verify_sigma_consumer_gate. It fails closed when any is absent (spec Sections 2.7/4.5).")
    state = _rollout_predictive_state(context, candidates, model, base_logits=base_logits)
    log_prob = state.log_prob
    # Dispatch the ambiguity BEFORE risk so the predictive marginal is single-sourced: under sigma_mc it
    # is the antithetic MC marginal q(o|pi) = E_s p(o|s); under likelihood_entropy it is q_log unchanged,
    # so risk/pred_ent stay byte-identical to the pre-PB-06 default path.
    estimate = get_ambiguity(ambiguity_mode)(
        state.q_log,
        mu=state.mu,
        sigma=state.sigma,
        model=model,
        num_samples=model.cfg.policy_sigma_mc_samples,
        model_behavior_sha256=model_behavior_sha256,
        spec_identity=sigma_spec_identity,
        code_identity_sha256=sigma_code_identity_sha256,
        measurement_context_sha256=sigma_measurement_context_sha256,
    )
    risk, pred_ent = _efe_terms(estimate.predictive_log_prob, preference)
    ambiguity = estimate.expected_conditional_entropy
    epistemic = pred_ent - ambiguity                            # MI bridge; ==0 at v1 (likelihood_entropy)
    terms = {"risk": risk, "ambiguity": ambiguity, "epistemic": -epistemic}
    score = torch.stack([terms[term] for term in score_terms], dim=0).sum(dim=0)
    post = _policy_posterior(score, gamma, log_prior)
    return PolicyScore(score, risk, ambiguity, epistemic, log_prob, post)


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
    model_behavior_sha256:      Optional[str] = None,   # PB-06 consumer-gate identities (sigma_mc only)
    sigma_spec_identity:        Optional[str] = None,
    sigma_code_identity_sha256: Optional[str] = None,
    sigma_measurement_context_sha256: Optional[str] = None,
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
    L = candidates.shape[2]
    _validate_policy_context(context, L, model.cfg.max_seq_len)
    if L != horizon:
        raise ValueError(
            f"efe_one_step candidate length L={L} must equal horizon={horizon}.")
    return _efe_score(context, candidates, preference, model, gamma=gamma, score_terms=score_terms,
                      ambiguity_mode=ambiguity_mode, log_prior=log_prior, base_logits=base_logits,
                      model_behavior_sha256=model_behavior_sha256,
                      sigma_spec_identity=sigma_spec_identity,
                      sigma_code_identity_sha256=sigma_code_identity_sha256,
                      sigma_measurement_context_sha256=sigma_measurement_context_sha256)


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
    if horizon != 1:
        raise ValueError(f"logprob_control accepts only horizon=1, got {horizon}")
    L = candidates.shape[2]
    _validate_policy_context(context, L, model.cfg.max_seq_len)
    if L != horizon:
        raise ValueError(
            f"logprob_control candidate length L={L} must equal horizon={horizon}.")
    if score_terms != ("risk", "ambiguity"):
        raise ValueError(
            "logprob_control scores raw continuation log-probability and accepts only the default "
            f"score_terms=('risk', 'ambiguity') metadata, got {score_terms}")
    q_log, log_prob = _rollout_predictive(context, candidates, model, base_logits=base_logits)
    zeros = torch.zeros_like(log_prob)
    score = -log_prob                                           # lower G <-> higher continuation logprob
    post = _policy_posterior(score, gamma, None)                 # log_prob already contains the base prior
    return PolicyScore(score, zeros, zeros, zeros, log_prob, post)


@register_policy("efe_rollout")
def _policy_efe_rollout(
    context:     torch.Tensor,                       # (B, N) context ids                          -> D
    candidates:  torch.Tensor,                       # (B, Kp, H) candidate H-action policies      -> E
    preference:  torch.Tensor,                       # (V,) or (B, V) log p(o|C)                   -> C

    model:       'object',                            # VFEModel: belief seam + prior_bank          -> A,B

    *,
    gamma:          float               = 1.0,        # policy precision in softmax(-gamma G)
    horizon:        int                 = 2,          # fixed rollout depth H (> 1; == candidate length)
    score_terms:    Tuple[str, ...]     = ("risk", "ambiguity"),  # which terms enter G(pi)
    ambiguity_mode: str                 = "likelihood_entropy",   # ambiguity registry key
    log_prior:      Optional[torch.Tensor] = None,    # (B, Kp) log candidate prior E; None -> uniform
    base_logits:    Optional[torch.Tensor] = None,    # (B, V) reused base logits
    model_behavior_sha256:      Optional[str] = None,   # PB-06 consumer-gate identities (sigma_mc only)
    sigma_spec_identity:        Optional[str] = None,
    sigma_code_identity_sha256: Optional[str] = None,
    sigma_measurement_context_sha256: Optional[str] = None,
    **kwargs,
) -> 'PolicyScore':
    r"""The staged horizon extension (H > 1): a TERMINAL-OUTCOME rollout scorer, not a per-step
    horizon sum. A policy pi = (a_1, ..., a_H) is the H-action candidate sequence (``candidates``
    column, length H); the belief is rolled forward over all H appended action tokens and G(pi) is
    evaluated on the SINGLE terminal predictive q(o|pi_H) = p(o | q*_pi) read from the LAST appended
    position -- NOT the active-inference per-step sum sum_{tau=1..H} G_tau (audit F3). The H-step
    rollout only advances the belief, so the terminal predictive is conditioned on all H actions (the
    environment response is never folded in, spec Section 2.2). Unlocked by the Phase-3a belief-prefix
    cache: the spec gates H > 1 on a cache so the Kp*H rollout reuses the shared context instead of
    paying Kp*H full recomputes, which is what makes the matched-compute baselines and the wall-clock
    honesty check fair (spec Sections 3.5, 4.2). It therefore REQUIRES the cache to be active; on a
    config the cache does not support it raises rather than silently falling back to the dishonest
    full recompute."""
    if horizon <= 1:
        raise ValueError(
            f"efe_rollout is the H>1 horizon scorer; got horizon={horizon}. Use efe_one_step for H=1.")
    L = candidates.shape[2]
    if L != horizon:
        raise ValueError(
            f"efe_rollout candidates carry the H-action policy sequence: candidate length L={L} must "
            f"equal horizon={horizon}.")
    _validate_policy_context(context, L, model.cfg.max_seq_len)
    if not cache_supported(model.cfg):
        raise NotImplementedError(
            "efe_rollout (horizon>1) requires the belief-prefix cache, which is exact only for the "
            "verified config (vfe3/inference/belief_cache.py::cache_supported: single block and E-step "
            "iteration, causal filtering flat kernel, frozen gauge frame, no model channel / "
            "precision-bias / gauge-RoPE / head mixer). On this config the rollout would fall back to "
            "the full per-candidate recompute, making the Kp*H cost dishonest (spec Section 3.5). Use a "
            "cache-supported config, or efe_one_step (horizon=1).")
    return _efe_score(context, candidates, preference, model, gamma=gamma, score_terms=score_terms,
                      ambiguity_mode=ambiguity_mode, log_prior=log_prior, base_logits=base_logits,
                      model_behavior_sha256=model_behavior_sha256,
                      sigma_spec_identity=sigma_spec_identity,
                      sigma_code_identity_sha256=sigma_code_identity_sha256,
                      sigma_measurement_context_sha256=sigma_measurement_context_sha256)
