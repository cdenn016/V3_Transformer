"""Regression coverage for the full-Gaussian float64 numerical island."""

from __future__ import annotations

import pytest
import torch

from vfe3.families.gaussian import FullGaussian


_DIMENSION = 4
_SELF_VALUE_ATOL = 5.0e-10
_SHARED_GRAD_ATOL = 5.0e-5


def _symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def _audited_covariance(seed: int, *, dtype: torch.dtype) -> torch.Tensor:
    """Reconstruct the exact twenty-seed SPD family used by the July 20 audit."""
    generator = torch.Generator().manual_seed(seed)
    frame, _ = torch.linalg.qr(torch.randn(
        _DIMENSION,
        _DIMENSION,
        generator=generator,
        dtype=torch.float64,
    ))
    eigenvalues = torch.logspace(0.0, -6.0, _DIMENSION, dtype=torch.float64)
    covariance = frame @ torch.diag(eigenvalues) @ frame.transpose(-1, -2)
    return covariance.to(dtype)


def _kl_unclamped(
    mu_q:    torch.Tensor,
    sigma_q: torch.Tensor,
    mu_p:    torch.Tensor,
    sigma_p: torch.Tensor,
) -> torch.Tensor:
    """Independent closed-form KL oracle evaluated in the operands' dtype."""
    sigma_q = _symmetrize(sigma_q)
    sigma_p = _symmetrize(sigma_p)
    factor_p = torch.linalg.cholesky(sigma_p)
    factor_q = torch.linalg.cholesky(sigma_q)
    first_solve = torch.linalg.solve_triangular(factor_p, sigma_q, upper=False)
    precision_sigma_q = torch.linalg.solve_triangular(
        factor_p.transpose(-1, -2),
        first_solve,
        upper=True,
    )
    trace_term = torch.diagonal(precision_sigma_q, dim1=-2, dim2=-1).sum(dim=-1)
    delta_mu = mu_p - mu_q
    whitened_delta = torch.linalg.solve_triangular(
        factor_p,
        delta_mu.unsqueeze(-1),
        upper=False,
    ).squeeze(-1)
    mahal_term = whitened_delta.square().sum(dim=-1)
    logdet_p = 2.0 * torch.log(torch.diagonal(
        factor_p,
        dim1=-2,
        dim2=-1,
    )).sum(dim=-1)
    logdet_q = 2.0 * torch.log(torch.diagonal(
        factor_q,
        dim1=-2,
        dim2=-1,
    )).sum(dim=-1)
    return 0.5 * (trace_term + mahal_term - mu_q.shape[-1] + logdet_p - logdet_q)


def _renyi_unclamped(
    mu_q:    torch.Tensor,
    sigma_q: torch.Tensor,
    mu_p:    torch.Tensor,
    sigma_p: torch.Tensor,

    *,
    alpha:   float,
) -> torch.Tensor:
    if abs(alpha - 1.0) < 1.0e-6:
        return _kl_unclamped(mu_q, sigma_q, mu_p, sigma_p)

    sigma_q = _symmetrize(sigma_q)
    sigma_p = _symmetrize(sigma_p)
    sigma_blend = _symmetrize((1.0 - alpha) * sigma_q + alpha * sigma_p)
    factor_blend = torch.linalg.cholesky(sigma_blend)
    delta_mu = mu_p - mu_q
    whitened_delta = torch.linalg.solve_triangular(
        factor_blend,
        delta_mu.unsqueeze(-1),
        upper=False,
    ).squeeze(-1)
    mahal_term = alpha * whitened_delta.square().sum(dim=-1)
    logdet_q = torch.linalg.slogdet(sigma_q).logabsdet
    logdet_p = torch.linalg.slogdet(sigma_p).logabsdet
    logdet_blend = torch.linalg.slogdet(sigma_blend).logabsdet
    logdet_term = (
        (1.0 - alpha) * logdet_q + alpha * logdet_p - logdet_blend
    ) / (alpha - 1.0)
    return 0.5 * (mahal_term + logdet_term)


def _batched_nonidentical_gaussians(
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(91)
    batch_shape = (2, 3)
    mu_q = 0.2 * torch.randn(*batch_shape, _DIMENSION, generator=generator, dtype=torch.float64)
    mu_p = 0.2 * torch.randn(*batch_shape, _DIMENSION, generator=generator, dtype=torch.float64)
    root_q = torch.randn(
        *batch_shape,
        _DIMENSION,
        _DIMENSION,
        generator=generator,
        dtype=torch.float64,
    )
    root_p = torch.randn(
        *batch_shape,
        _DIMENSION,
        _DIMENSION,
        generator=generator,
        dtype=torch.float64,
    )
    eye = torch.eye(_DIMENSION, dtype=torch.float64)
    sigma_q = root_q @ root_q.transpose(-1, -2) + 0.75 * eye
    sigma_p = root_p @ root_p.transpose(-1, -2) + 0.50 * eye
    return mu_q.to(dtype), sigma_q.to(dtype), mu_p.to(dtype), sigma_p.to(dtype)


def _self_kl_metrics(
    seed:   int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    sigma = _audited_covariance(seed, dtype=torch.float32).to(device).requires_grad_(True)
    mu = torch.zeros(_DIMENSION, dtype=torch.float32, device=device)
    divergence = FullGaussian(mu, sigma).renyi_closed_form(
        FullGaussian(mu, sigma),
        alpha=1.0,
        kl_max=100.0,
        eps=1.0e-6,
    )
    shared_gradient, = torch.autograd.grad(divergence, sigma)
    return divergence, shared_gradient


def test_audited_seed_17_reproduces_legacy_failure_and_float64_oracle() -> None:
    sigma32 = _audited_covariance(17, dtype=torch.float32).requires_grad_(True)
    mu32 = torch.zeros(_DIMENSION, dtype=torch.float32)
    legacy_value = _kl_unclamped(mu32, sigma32, mu32, sigma32)
    legacy_gradient, = torch.autograd.grad(legacy_value, sigma32)

    sigma64 = sigma32.detach().double().requires_grad_(True)
    mu64 = mu32.double()
    oracle_value = _kl_unclamped(mu64, sigma64, mu64, sigma64)
    oracle_gradient, = torch.autograd.grad(oracle_value, sigma64)

    assert float(torch.linalg.cond(sigma32.detach())) == pytest.approx(950_196.8125, abs=0.5)
    assert float(legacy_value.detach()) == pytest.approx(0.012682914733886719, abs=1.0e-8)
    assert float(legacy_gradient.abs().max()) == pytest.approx(8_645.646484375, abs=0.1)
    assert float(oracle_value.detach().abs()) <= 2.0e-10
    assert float(oracle_gradient.abs().max()) <= 2.0e-5


@pytest.mark.parametrize("seed", range(20))
def test_full_gaussian_self_kl_and_shared_derivative_are_near_zero(seed: int) -> None:
    r"""For q=p=N(mu,Sigma),

    KL(q||q) = 0.5 * (tr(I) + 0 - K + logdet(Sigma) - logdet(Sigma)) = 0.
    For a shared covariance variable, d tr(Sigma^-1 Sigma) vanishes because its inverse and
    direct differentials cancel, and the two log-determinant differentials cancel. The total
    derivative is therefore zero. At exact equality the two independent partial derivatives also
    vanish separately; away from equality they generally differ.
    """
    divergence, shared_gradient = _self_kl_metrics(seed, torch.device("cpu"))

    assert divergence.dtype == torch.float32
    assert float(divergence.detach().abs()) <= _SELF_VALUE_ATOL
    assert float(shared_gradient.abs().max()) <= _SHARED_GRAD_ATOL


def test_full_gaussian_self_kl_cuda_mirror(device: torch.device) -> None:
    max_abs_value = 0.0
    max_abs_gradient = 0.0
    for seed in range(20):
        divergence, shared_gradient = _self_kl_metrics(seed, device)
        assert divergence.dtype == torch.float32
        max_abs_value = max(max_abs_value, float(divergence.detach().abs()))
        max_abs_gradient = max(max_abs_gradient, float(shared_gradient.abs().max()))

    assert max_abs_value <= _SELF_VALUE_ATOL
    assert max_abs_gradient <= _SHARED_GRAD_ATOL


def test_separate_covariance_gradients_match_float64_oracle() -> None:
    mu_q, sigma_q_base, mu_p, sigma_p_base = _batched_nonidentical_gaussians(dtype=torch.float32)
    sigma_q = sigma_q_base.requires_grad_(True)
    sigma_p = sigma_p_base.requires_grad_(True)
    actual = FullGaussian(mu_q, sigma_q).renyi_closed_form(
        FullGaussian(mu_p, sigma_p),
        alpha=1.0,
        kl_max=100.0,
        eps=1.0e-6,
    )
    actual_grad_q, actual_grad_p = torch.autograd.grad(actual.sum(), (sigma_q, sigma_p))

    oracle_sigma_q = sigma_q.detach().double().requires_grad_(True)
    oracle_sigma_p = sigma_p.detach().double().requires_grad_(True)
    oracle = _kl_unclamped(
        mu_q.double(),
        oracle_sigma_q,
        mu_p.double(),
        oracle_sigma_p,
    )
    oracle_grad_q, oracle_grad_p = torch.autograd.grad(
        oracle.sum(),
        (oracle_sigma_q, oracle_sigma_p),
    )

    assert actual.shape == (2, 3)
    assert actual.dtype == torch.float32
    assert torch.allclose(actual.double(), oracle.detach(), atol=2.0e-6, rtol=2.0e-6)
    assert torch.allclose(actual_grad_q.double(), oracle_grad_q, atol=2.0e-6, rtol=2.0e-6)
    assert torch.allclose(actual_grad_p.double(), oracle_grad_p, atol=2.0e-6, rtol=2.0e-6)
    assert not torch.allclose(actual_grad_q, actual_grad_p, atol=1.0e-6, rtol=1.0e-6)


@pytest.mark.parametrize("alpha", [0.5, 1.0, 1.0001])
def test_nonidentical_batched_values_preserve_renyi_and_alpha_limit(alpha: float) -> None:
    mu_q, sigma_q, mu_p, sigma_p = _batched_nonidentical_gaussians(dtype=torch.float32)
    actual = FullGaussian(mu_q, sigma_q).renyi_closed_form(
        FullGaussian(mu_p, sigma_p),
        alpha=alpha,
        kl_max=100.0,
        eps=1.0e-6,
    )
    oracle = _renyi_unclamped(
        mu_q.double(),
        sigma_q.double(),
        mu_p.double(),
        sigma_p.double(),
        alpha=alpha,
    )

    assert actual.shape == (2, 3)
    assert torch.allclose(actual.double(), oracle, atol=3.0e-6, rtol=3.0e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_full_gaussian_public_result_dtype_policy_is_preserved(dtype: torch.dtype) -> None:
    mu_q, sigma_q, mu_p, sigma_p = _batched_nonidentical_gaussians(dtype=dtype)
    actual = FullGaussian(mu_q, sigma_q).renyi_closed_form(
        FullGaussian(mu_p, sigma_p),
        alpha=1.0,
        kl_max=100.0,
        eps=1.0e-6,
    )
    oracle = _kl_unclamped(
        mu_q.double(),
        sigma_q.double(),
        mu_p.double(),
        sigma_p.double(),
    )
    expected_dtype = torch.float64 if dtype == torch.float64 else torch.float32

    assert actual.dtype == expected_dtype
    assert torch.allclose(actual.double(), oracle, atol=3.0e-4, rtol=3.0e-4)
