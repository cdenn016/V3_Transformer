import os
import sys
from collections.abc import Mapping
from contextlib import AbstractContextManager, contextmanager
from typing import Any, Iterator


_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
_REQUESTED_DEVICE_NAME = os.environ.get("VFE3_TEST_DEVICE", "cpu")


def _requests_cuda(name: str) -> bool:
    return name.partition(":")[0].lower() == "cuda"


def _validate_cuda_preimport(
    requested_device_name: str,
    environment:           Mapping[str, str],
    loaded_modules:        Mapping[str, object],
) -> None:
    if not _requests_cuda(requested_device_name):
        return
    if (
        "torch" in loaded_modules
        and environment.get("CUBLAS_WORKSPACE_CONFIG") != _CUBLAS_WORKSPACE_CONFIG
    ):
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG=:4096:8 must be set before importing torch "
            "for CUDA tests"
        )


_validate_cuda_preimport(_REQUESTED_DEVICE_NAME, os.environ, sys.modules)
if _requests_cuda(_REQUESTED_DEVICE_NAME):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import pytest
import torch


@contextmanager
def _deterministic_cuda_policy(
    torch_module:          Any,
    requested_device_name: str,
) -> Iterator[None]:
    requested_device = torch_module.device(requested_device_name)
    if requested_device.type != "cuda":
        yield
        return

    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != _CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG=:4096:8 must be set before importing torch "
            "for CUDA tests"
        )

    algorithms_enabled = torch_module.are_deterministic_algorithms_enabled()
    warn_only_enabled = torch_module.is_deterministic_algorithms_warn_only_enabled()
    cudnn_deterministic = torch_module.backends.cudnn.deterministic
    cudnn_benchmark = torch_module.backends.cudnn.benchmark
    matmul = torch_module.backends.cuda.matmul
    cudnn = torch_module.backends.cudnn
    modern_tf32 = hasattr(matmul, "fp32_precision") and hasattr(
        cudnn,
        "fp32_precision",
    )
    if modern_tf32:
        matmul_tf32 = matmul.fp32_precision
        cudnn_tf32 = cudnn.fp32_precision
    else:
        matmul_tf32 = matmul.allow_tf32
        cudnn_tf32 = cudnn.allow_tf32

    try:
        torch_module.use_deterministic_algorithms(True, warn_only=False)
        cudnn.deterministic = True
        cudnn.benchmark = False
        if modern_tf32:
            matmul.fp32_precision = "ieee"
            cudnn.fp32_precision = "ieee"
        else:
            matmul.allow_tf32 = False
            cudnn.allow_tf32 = False
        yield
    finally:
        torch_module.use_deterministic_algorithms(
            algorithms_enabled,
            warn_only=warn_only_enabled,
        )
        cudnn.deterministic = cudnn_deterministic
        cudnn.benchmark = cudnn_benchmark
        if modern_tf32:
            matmul.fp32_precision = matmul_tf32
            cudnn.fp32_precision = cudnn_tf32
        else:
            matmul.allow_tf32 = matmul_tf32
            cudnn.allow_tf32 = cudnn_tf32


def _start_deterministic_cuda_policy(
    torch_module:          Any,
    requested_device_name: str,
) -> AbstractContextManager[None]:
    """Enter the process CUDA policy before pytest begins collection."""
    policy_context = _deterministic_cuda_policy(torch_module, requested_device_name)
    policy_context.__enter__()
    return policy_context


_DETERMINISTIC_CUDA_POLICY_CONTEXT = _start_deterministic_cuda_policy(
    torch,
    _REQUESTED_DEVICE_NAME,
)

pytest_plugins = ("tests.pytest_policy",)


@pytest.fixture(scope="session", autouse=True)
def deterministic_cuda_policy() -> Iterator[None]:
    try:
        yield
    finally:
        _DETERMINISTIC_CUDA_POLICY_CONTEXT.__exit__(None, None, None)


@pytest.fixture
def device() -> torch.device:
    # Tests are device-agnostic; default CPU for portability.
    # Set VFE3_TEST_DEVICE=cuda to run on the GPU.
    name = os.environ.get("VFE3_TEST_DEVICE", "cpu")
    resolved_device = torch.device(name)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA requested but not available")
    return resolved_device
