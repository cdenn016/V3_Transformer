"""Shared type contracts for mutable runtime dictionaries."""

from typing import List, NamedTuple, Optional, Tuple, TypedDict

import torch

from vfe3.belief import BeliefState


class EffectiveBetaPriorContext(NamedTuple):
    r"""Fixed pre-``vfe_stack`` state the belief-channel attention log-prior is folded from.

    Captured once in ``VFEModel.forward_beliefs`` at the seam immediately after the optional
    s-refinement and before ``vfe_stack``, then consumed by ``VFEModel._effective_beta_log_prior`` to
    (re)build the effective prior the E-step descends -- the RAW ``_attention_log_prior`` with the
    detached precision-weighted reliability bias and, under ``gamma_as_beta_prior``, the detached
    hierarchical gamma prior folded on. Every field is the FIXED encode-time state: ``precision_sigma``
    is the pre-stack belief covariance (an intentional fixed reliability prior held across the E-step,
    NOT a per-iteration one), ``model_phi`` the resolved model-channel frame, and ``s_mu``/``s_sigma``
    the refined model belief (``None`` when ``s_e_step`` is off, so the gamma fold reads the raw s
    tables). Only the CANDIDATE belief passed alongside supplies the tied-gamma frame; nothing here is
    candidate-dependent, so a reflection/two-hop scorer can reuse one context across every proposal."""

    token_ids:       torch.Tensor            # (B, N) integer token ids
    base_log_prior:  Optional[torch.Tensor]  # RAW _attention_log_prior (BEFORE any fold), or None
    precision_sigma: torch.Tensor            # (B, N, K)/(B, N, K, K) fixed pre-stack belief covariance
    model_phi:       torch.Tensor            # (B, N, n_gen) resolved model-channel frame
    s_mu:            Optional[torch.Tensor]   # refined model-belief mean, or None (raw s tables)
    s_sigma:         Optional[torch.Tensor]   # refined model-belief covariance, or None (raw s tables)


class MetropolisObjectiveContext(NamedTuple):
    r"""Fixed q/p state the reflection-Metropolis scorer evaluates the EXACT active objective from.

    Captured once per sweep by ``VFEModel._metropolis_prepare`` from a single belief forward, then held
    FIXED across every proposal so the fixed-belief ``DeltaF = F(trial) - F(current)`` is the exact
    change in the joint free energy the E-step descended (audit PB-12). ``belief`` is the final block's
    converged/current ``BeliefState`` (carrying the frame the E-step minimized -- ``omega`` under the
    omega-direct move, ``reflection`` under the phi move); the sweep initializes ``f_cur`` and the
    sequential current state from this exact object before constructing any trial. ``mu_p``/``sigma_p``
    are the HANDOFF-ADJUSTED prior moments ENTERING the final block (``MStepCapture['final_block_prior']``,
    equal to the encode prior only at ``n_layers==1``), ``tau`` the exact query-adaptive temperature the
    final block used (``MStepCapture['final_block_tau']`` -- the ENTRY-derived tau that produced the
    converged belief, NOT a tau recomputed from the converged sigma), and ``rope`` the positional RoPE
    rotation for this token length (``None`` when ``pos_rotation=='none'``). ``prior`` is the fixed
    pre-stack :class:`EffectiveBetaPriorContext`; the scorer rebuilds the candidate-dependent effective
    prior per proposal via ``_effective_beta_log_prior(candidate_belief, prior)`` -- the precision fold
    reads the FIXED ``prior.precision_sigma`` (frame-blind), and only the tied-gamma fold varies with the
    proposed frame. Nothing else is candidate-dependent, so one context serves the whole sweep."""

    token_ids: torch.Tensor                     # (B, N) integer token ids
    mu_p:      torch.Tensor                     # (B, N, K) final-block handoff-adjusted prior means
    sigma_p:   torch.Tensor                     # (B, N, K)/(B, N, K, K) final-block prior variances
    belief:    BeliefState                      # final-block converged/current belief (carries the frame)
    tau:       'float | torch.Tensor'           # final-block entry-derived (query-adaptive) softmax tau
    rope:      Optional[torch.Tensor]           # (N, K, K) positional RoPE rotation, or None
    prior:     EffectiveBetaPriorContext         # fixed pre-stack effective-prior context (folds rebuilt per candidate)


class PolicyRollout(NamedTuple):
    r"""The state-carrying result of one EFE candidate rollout (PB-06).

    Extends the historical two-tensor ``(q_log, log_prob)`` return with the TERMINAL belief moments read
    at the last appended position, so a sigma-dependent ambiguity estimator can read the belief
    covariance the rollout actually converged to (the sigma-free ``likelihood_entropy`` arm ignores
    ``mu``/``sigma``). ``mu`` is ``(B, Kp, K)`` and ``sigma`` is ``(B, Kp, K)`` (diagonal family) or
    ``(B, Kp, K, K)`` (full family). The full path reads them from the returned ``BeliefState``; the
    cached path reads them from the appended positions after the same block_norm/final_norm that produced
    ``q_log``. The compatibility wrappers ``_rollout_predictive`` / ``rollout_predictive_cached`` return
    exactly ``(q_log, log_prob)`` so existing two-tensor unpacking is unchanged."""

    q_log:    torch.Tensor   # (B, Kp, V) log q(o|pi) at the terminal predictive
    log_prob: torch.Tensor   # (B, Kp) raw first-action continuation log-prob under the base predictive
    mu:       torch.Tensor   # (B, Kp, K) terminal belief mean
    sigma:    torch.Tensor   # (B, Kp, K) or (B, Kp, K, K) terminal belief covariance


class MStepCapture(TypedDict, total=False):
    """Mutable intermediates captured for the M-step self-coupling term."""

    converged:          BeliefState
    final_block_prior:  Tuple[torch.Tensor, torch.Tensor]
    final_block_tau:    'float | torch.Tensor'
    prior:              BeliefState
    out:                BeliefState
    beta_prior_context: EffectiveBetaPriorContext
    # CG moment-energy participation (PB-13). Present ONLY when cfg.cg_energy_weight>0. Attached
    # E-step estimators append the per-layer D(q_post||q_pre) rows to ``cg_moment_energy_rows``; the
    # 'detach' estimator instead appends the detached pre-CG (mu, sigma) pairs to ``cg_pre_moments``
    # for the post-stack ``torch.enable_grad`` re-evaluation. A capture allocated only for M-step
    # self-coupling never carries either key.
    cg_moment_energy_rows: List[torch.Tensor]
    cg_pre_moments:        List[Tuple[torch.Tensor, torch.Tensor]]


class EStepGradientRecord(TypedDict, total=False):
    """Detached tensor-valued E-step gradient norms before host conversion."""

    mu:    torch.Tensor
    sigma: torch.Tensor
    phi:   torch.Tensor


class EStepGradientOutput(TypedDict, total=False):
    """Host float-valued E-step gradient norms exposed by the model API."""

    mu:    float
    sigma: float
    phi:   float


class DataStateBuffer(TypedDict, total=False):
    """Load-time buffer populated only when a checkpoint carries iterator state."""

    epoch_start_generator_state: torch.Tensor
    batches_consumed:            int
    epoch:                       int


class DataState(TypedDict):
    """Required iterator state written into a resumable checkpoint."""

    epoch_start_generator_state: torch.Tensor
    batches_consumed:            int
    epoch:                       int
