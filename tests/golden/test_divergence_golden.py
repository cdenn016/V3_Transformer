import torch


def test_can_import_vfe2_kernels(vfe2_kl):
    # Smoke test: the 2.0 reference kernels are importable.
    assert hasattr(vfe2_kl, "_kl_kernel_diagonal")
    assert hasattr(vfe2_kl, "_kl_kernel_dense")
    assert hasattr(vfe2_kl, "safe_kl_clamp")
