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


def _fp32_matmul_precision() -> Dict[str, object]:
    r"""Effective fp32 matmul / TF32 policy, best-effort across torch versions.

    ``torch.use_deterministic_algorithms(True)`` does NOT constrain TF32 (audit 2026-08-06 F25), so
    this is a real degree of freedom that production neither pinned nor recorded while
    ``tests/conftest.py:61-99`` pins and restores it. That left every numeric tolerance in the suite
    asserted under a policy the training run never stated, and ``provenance.json`` unable to show
    the drift. RECORDING is unconditional and cannot change numerics; SETTING it is not done here,
    because on the user's card that would change results mid-project.
    """
    state: Dict[str, object] = {}
    try:                                   # torch >= 2.9 typed accessor
        state["matmul_fp32_precision"] = str(torch.backends.cuda.matmul.fp32_precision)
    except AttributeError:
        state["matmul_fp32_precision"] = None
    for name, get in (
        ("matmul_allow_tf32", lambda: torch.backends.cuda.matmul.allow_tf32),
        ("cudnn_allow_tf32",  lambda: torch.backends.cudnn.allow_tf32),
        ("float32_matmul_precision", torch.get_float32_matmul_precision),
    ):
        try:
            state[name] = get()
        except Exception:                  # noqa: BLE001 - a missing backend knob is not an error
            state[name] = None
    return state


def deterministic_state() -> Dict[str, object]:
    """Return the effective PyTorch, cuDNN, cuBLAS, and fp32-matmul determinism state."""
    return {
        "algorithms":              torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic":     bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark":         bool(torch.backends.cudnn.benchmark),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        **_fp32_matmul_precision(),
    }
