import pytest
import torch


def test_can_import_vfe2_kernels(vfe2_kl):
    # Smoke test: the 2.0 reference kernels are importable.
    assert hasattr(vfe2_kl, "_kl_kernel_diagonal")
    assert hasattr(vfe2_kl, "_kl_kernel_dense")
    assert hasattr(vfe2_kl, "safe_kl_clamp")


def _rand_diag(B, N, K, device, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    mu_q = torch.randn(B, N, K, generator=g).to(device)
    mu_t = torch.randn(B, N, K, generator=g).to(device)
    # variances in [0.1, 1.1], well-conditioned
    sigma_q = (torch.rand(B, N, K, generator=g) + 0.1).to(device)
    sigma_t = (torch.rand(B, N, K, generator=g) + 0.1).to(device)
    return mu_q, sigma_q, mu_t, sigma_t


def _rand_full(B, N, K, device, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    mu_q = torch.randn(B, N, K, generator=g).to(device)
    mu_t = torch.randn(B, N, K, generator=g).to(device)
    Aq = torch.randn(B, N, K, K, generator=g)
    At = torch.randn(B, N, K, K, generator=g)
    eye = torch.eye(K)
    # SPD, well-conditioned: A A^T + I
    sigma_q = (Aq @ Aq.transpose(-1, -2) + eye).to(device)
    sigma_t = (At @ At.transpose(-1, -2) + eye).to(device)
    return mu_q, sigma_q, mu_t, sigma_t


def test_diagonal_kl_matches_vfe2(vfe2_kl, device):
    from vfe3.divergence import get_divergence
    mu_q, sigma_q, mu_t, sigma_t = _rand_diag(2, 4, 5, device, seed=0)
    ref = vfe2_kl._kl_kernel_diagonal(
        mu_q, sigma_q, mu_t, sigma_t, kl_max=100.0, eps=1e-6, alpha_div=1.0
    )
    got = get_divergence("gaussian_diagonal")(
        mu_q, sigma_q, mu_t, sigma_t, alpha=1.0, kl_max=100.0, eps=1e-6
    )
    assert torch.allclose(got, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("alpha", [0.5, 0.9, 1.5, 2.0])
def test_diagonal_renyi_matches_vfe2(vfe2_kl, device, alpha):
    from vfe3.divergence import get_divergence
    mu_q, sigma_q, mu_t, sigma_t = _rand_diag(2, 4, 5, device, seed=1)
    ref = vfe2_kl._kl_kernel_diagonal(
        mu_q, sigma_q, mu_t, sigma_t, kl_max=100.0, eps=1e-6, alpha_div=alpha
    )
    got = get_divergence("gaussian_diagonal")(
        mu_q, sigma_q, mu_t, sigma_t, alpha=alpha, kl_max=100.0, eps=1e-6
    )
    assert torch.allclose(got, ref, atol=1e-5, rtol=1e-5)


def test_full_kl_matches_vfe2(vfe2_kl, device):
    from vfe3.divergence import get_divergence
    mu_q, sigma_q, mu_t, sigma_t = _rand_full(2, 3, 4, device, seed=2)
    ref = vfe2_kl._kl_kernel_dense(
        mu_q, sigma_q, mu_t, sigma_t, kl_max=100.0, eps=1e-6, alpha_div=1.0
    )
    got = get_divergence("gaussian_full")(
        mu_q, sigma_q, mu_t, sigma_t, alpha=1.0, kl_max=100.0, eps=1e-6
    )
    assert torch.allclose(got, ref, atol=1e-4, rtol=1e-4)


# Full-cov Renyi: the blend (1-a)Sig_q + a Sig_t is a convex combination of
# two SPD matrices (hence guaranteed SPD) only for alpha in (0, 1]. For
# alpha > 1 the blend can be indefinite, where 2.0 returns NaN via its
# 5-round Cholesky fallback; replicating that NaN contract is a deferred
# robustness task (see _gaussian_full_renyi in divergence.py).
@pytest.mark.parametrize("alpha", [0.5, 0.9])
def test_full_renyi_matches_vfe2(vfe2_kl, device, alpha):
    from vfe3.divergence import get_divergence
    mu_q, sigma_q, mu_t, sigma_t = _rand_full(2, 3, 4, device, seed=3)
    ref = vfe2_kl._kl_kernel_dense(
        mu_q, sigma_q, mu_t, sigma_t, kl_max=100.0, eps=1e-6, alpha_div=alpha
    )
    got = get_divergence("gaussian_full")(
        mu_q, sigma_q, mu_t, sigma_t, alpha=alpha, kl_max=100.0, eps=1e-6
    )
    assert torch.allclose(got, ref, atol=1e-4, rtol=1e-4)
