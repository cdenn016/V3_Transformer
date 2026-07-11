"""Shared runtime-state setup and reporting for click-to-run entry points."""

import os
from typing import Dict

import torch


_INITIAL_CUBLAS_WORKSPACE_CONFIG = os.environ.get("CUBLAS_WORKSPACE_CONFIG")


def seed_everything(
    seed: int,

    *,
    deterministic: bool,
) -> None:
    """Seed PyTorch and configure the requested deterministic execution state."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    elif _INITIAL_CUBLAS_WORKSPACE_CONFIG is None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = _INITIAL_CUBLAS_WORKSPACE_CONFIG


def deterministic_state() -> Dict[str, object]:
    """Return the effective PyTorch, cuDNN, and cuBLAS determinism state."""
    return {
        "algorithms":              torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic":     bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark":         bool(torch.backends.cudnn.benchmark),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
