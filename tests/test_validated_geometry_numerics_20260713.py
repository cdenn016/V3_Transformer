r"""Regressions for the validated July 13 geometry and numerical audit findings."""

import warnings

import pytest
import torch

from vfe3.attention_prior import attention_log_prior
from vfe3.config import VFE3Config
from vfe3.families.base import get_family
from vfe3.families.covariance_tables import packed_from_covariance
from vfe3.geometry.groups import get_group
from vfe3.geometry.norms import MahalanobisNorm
from vfe3.geometry.transport import (
    FactoredTransport,
    _factored_full_covariance,
    build_factored_transport,
    gauge_invariant_edge_features,
    get_transport,
    transport_covariance,
)


def test_query_adaptive_tau_warns_that_noncompact_gl_gauge_is_broken() -> None:
    with pytest.warns(UserWarning, match=r"query_adaptive_tau.*GL\(K\).*(break|breaking)"):
        VFE3Config(embed_dim=4, n_heads=2, query_adaptive_tau=True, query_tau_c=1.0)


def test_query_adaptive_tau_does_not_warn_when_strength_is_zero() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        VFE3Config(embed_dim=4, n_heads=2, query_adaptive_tau=True, query_tau_c=0.0)
    assert not any("query_adaptive_tau" in str(item.message) for item in caught)


def test_regime_ii_trivial_gauge_preserves_bilinear_edge_connection() -> None:
    torch.manual_seed(3)
    group = get_group("block_glk")(4, 2)
    phi = torch.randn(1, 3, group.generators.shape[0])
    mu = torch.randn(1, 3, 4)
    connection = 0.2 * torch.randn(group.generators.shape[0], 4, 4)

    omega = get_transport("regime_ii")(
        phi,
        group,
        gauge_mode="trivial",
        mu=mu,
        connection_W=connection,
        cocycle_relaxation=1.0,
    )["Omega"]

    eye = torch.eye(4)
    assert not torch.allclose(omega[0, 0, 1], eye, atol=1e-6, rtol=0.0)


def test_regime_ii_covariant_trivial_gauge_preserves_invariant_edge_connection() -> None:
    torch.manual_seed(4)
    group = get_group("block_glk")(4, 2)
    phi = torch.randn(1, 3, group.generators.shape[0])
    mu = torch.randn(1, 3, 4)
    sigma = torch.rand(1, 3, 4) + 0.5
    connection = 0.2 * torch.randn(group.generators.shape[0], 3)

    omega = get_transport("regime_ii_covariant")(
        phi,
        group,
        gauge_mode="trivial",
        mu=mu,
        sigma=sigma,
        connection_M=connection,
        cocycle_relaxation=1.0,
    )["Omega"]

    eye = torch.eye(4)
    assert not torch.allclose(omega[0, 0, 1], eye, atol=1e-6, rtol=0.0)


@pytest.mark.parametrize("per_coordinate", [False, True])
def test_diagonal_gaussian_renyi_preserves_float64(per_coordinate: bool) -> None:
    family = get_family("gaussian_diagonal")
    mu_q = torch.tensor([[0.1, -0.2]], dtype=torch.float64)
    sigma_q = torch.tensor([[1.0 / 3.0, 1.0 / 7.0]], dtype=torch.float64)
    mu_p = torch.tensor([[0.25, 0.05]], dtype=torch.float64)
    sigma_p = torch.tensor([[1.0 / 9.0, 3.0]], dtype=torch.float64)
    q = family(mu_q, sigma_q)
    p = family(mu_p, sigma_p)

    if per_coordinate:
        out = q.renyi_per_coord(p, alpha=1.0, kl_max=1e12)
        expected = 0.5 * (
            sigma_q / sigma_p + (mu_p - mu_q).square() / sigma_p - 1.0
            + torch.log(sigma_p) - torch.log(sigma_q)
        )
    else:
        out = q.renyi_closed_form(p, alpha=1.0, kl_max=1e12)
        expected = 0.5 * (
            sigma_q / sigma_p + (mu_p - mu_q).square() / sigma_p - 1.0
            + torch.log(sigma_p) - torch.log(sigma_q)
        ).sum(dim=-1)

    assert out.dtype == torch.float64
    torch.testing.assert_close(out, expected, rtol=0.0, atol=1e-14)


def test_full_gaussian_entropy_masks_failed_cholesky() -> None:
    family = get_family("gaussian_full")
    covariance = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
    entropy = family(torch.zeros(2), covariance).entropy()
    assert torch.isnan(entropy)


def test_packed_from_covariance_masks_failed_cholesky() -> None:
    covariance = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
    log_diag, packed = packed_from_covariance(covariance)
    assert torch.isnan(log_diag).all()
    assert torch.isnan(packed).all()


def test_covariant_edge_features_route_logdet_through_guarded_helper(monkeypatch) -> None:
    import vfe3.geometry.transport as transport_module
    from vfe3.families.base import _logdet_chol as original

    calls = []

    def _record(factor: torch.Tensor) -> torch.Tensor:
        calls.append(tuple(factor.shape))
        return original(factor)

    monkeypatch.setattr(transport_module, "_logdet_chol", _record, raising=False)
    mu_q = torch.tensor([[0.1, -0.2]])
    mu_k = torch.tensor([[0.3, 0.4]])
    covariance = torch.tensor([[[1.2, 0.1], [0.1, 0.9]]])
    gauge_invariant_edge_features(mu_q, covariance, mu_k, covariance)
    assert calls == [(1, 2, 2), (1, 2, 2)]


def test_alibi_non_power_of_two_heads_matches_press_reference() -> None:
    bias = attention_log_prior("alibi", 2, 2, n_heads=3, alibi_slope=1.0)
    slopes = -bias[:, 0, 1]
    expected = torch.tensor([1.0 / 16.0, 1.0 / 256.0, 1.0 / 4.0])
    torch.testing.assert_close(slopes, expected, rtol=0.0, atol=1e-8)


def _spd(batch: int, tokens: int, dimension: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(8)
    matrix = torch.randn(batch, tokens, dimension, dimension, generator=generator)
    return matrix @ matrix.transpose(-1, -2) + dimension * torch.eye(dimension)


def test_equal_block_full_covariance_transport_uses_one_batched_einsum(monkeypatch) -> None:
    group = get_group("block_glk")(4, 2)
    phi = 0.2 * torch.randn(1, 3, group.generators.shape[0], generator=torch.Generator().manual_seed(9))
    factored = build_factored_transport(phi, group)
    sigma = _spd(1, 3, 4)
    expected = transport_covariance(factored.to_dense_omega(), sigma)
    original = torch.einsum
    calls = []

    def _record(equation: str, *operands: torch.Tensor) -> torch.Tensor:
        calls.append(equation)
        return original(equation, *operands)

    monkeypatch.setattr(torch, "einsum", _record)
    actual = _factored_full_covariance(factored, sigma)

    assert len(calls) == 1
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_heterogeneous_block_full_covariance_transport_keeps_dense_parity() -> None:
    exp_phi = torch.stack(
        [torch.diag(torch.tensor([1.1, 0.9, 1.2])), torch.diag(torch.tensor([0.8, 1.3, 0.7]))]
    ).unsqueeze(0)
    exp_neg_phi = torch.linalg.inv(exp_phi)
    factored = FactoredTransport(exp_phi=exp_phi, exp_neg_phi=exp_neg_phi, irrep_dims=[1, 2])
    sigma = _spd(1, 2, 3)

    actual = _factored_full_covariance(factored, sigma)
    expected = transport_covariance(factored.to_dense_omega(), sigma)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_mahalanobis_full_covariance_uses_exact_float64_solve() -> None:
    generator = torch.Generator().manual_seed(22)
    basis, _ = torch.linalg.qr(torch.randn(4, 4, generator=generator))
    sigma = basis @ torch.diag(torch.tensor([1e-5, 0.2, 1.3, 4.0])) @ basis.transpose(-1, -2)
    mu = torch.tensor([0.7, -0.4, 0.2, 1.1], dtype=torch.float32)
    norm = MahalanobisNorm(4, eps=0.0)

    actual = norm(mu, sigma)
    solution = torch.linalg.solve(sigma.double(), mu.double())
    s2 = (mu.double() * solution).sum()
    expected = mu * torch.sqrt(torch.tensor(4.0, dtype=torch.float64) / s2).to(mu.dtype)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-7)


def test_mahalanobis_singular_fallback_is_finite_and_documented_as_approximate() -> None:
    sigma = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    mu = torch.tensor([1.0, -0.5])
    norm = MahalanobisNorm(2, eps=1e-6)

    assert torch.isfinite(norm(mu, sigma)).all()
    assert "not gauge-invariant" in (MahalanobisNorm.__doc__ or "")


def test_laplace_family_selects_degree_one_scale_transport() -> None:
    family = get_family("laplace_diagonal")
    scale = torch.tensor([[2.0, 3.0]])
    omega = torch.diag(torch.tensor([4.0, -0.5])).reshape(1, 1, 2, 2)

    transported = family.transport_dispersion(scale, omega, diagonal_out=True)
    expected = torch.tensor([[[8.0, 1.5]]])
    torch.testing.assert_close(transported, expected, rtol=0.0, atol=1e-7)


def test_laplace_off_diagonal_transport_is_variance_matching_projection() -> None:
    family = get_family("laplace_diagonal")
    scale = torch.tensor([[2.0, 3.0]])
    omega = torch.tensor([[1.0, 1.0], [0.0, 1.0]]).reshape(1, 1, 2, 2)

    transported = family.transport_dispersion(scale, omega, diagonal_out=True)
    expected = torch.tensor([[[13.0 ** 0.5, 3.0]]])
    torch.testing.assert_close(transported, expected, rtol=0.0, atol=1e-7)
