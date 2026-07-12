"""Shared type contracts for mutable runtime dictionaries."""

from typing import Tuple, TypedDict

import torch

from vfe3.belief import BeliefState


class MStepCapture(TypedDict, total=False):
    """Mutable intermediates captured for the M-step self-coupling term."""

    converged:         BeliefState
    final_block_prior: Tuple[torch.Tensor, torch.Tensor]
    prior:             BeliefState
    out:               BeliefState


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
