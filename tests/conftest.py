import pytest
import torch


@pytest.fixture
def device():
    # Tests are device-agnostic; default CPU for portability.
    # Set VFE3_TEST_DEVICE=cuda to run on the GPU.
    import os
    name = os.environ.get("VFE3_TEST_DEVICE", "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA requested but not available")
    return torch.device(name)
