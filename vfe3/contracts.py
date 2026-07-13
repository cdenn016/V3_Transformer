"""Shared type contracts for mutable runtime dictionaries."""

from typing import NamedTuple, Optional, Tuple, TypedDict

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


class MStepCapture(TypedDict, total=False):
    """Mutable intermediates captured for the M-step self-coupling term."""

    converged:          BeliefState
    final_block_prior:  Tuple[torch.Tensor, torch.Tensor]
    prior:              BeliefState
    out:                BeliefState
    beta_prior_context: EffectiveBetaPriorContext


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
