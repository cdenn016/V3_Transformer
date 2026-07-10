r"""Focused regressions for family-aware full-covariance inference geometry."""

import math

import pytest
import torch
import torch.nn.functional as F

from vfe3.belief import BeliefState
from vfe3.config import VFE3Config
from vfe3.families.laplace import DiagonalLaplace
from vfe3.geometry.groups import get_group
from vfe3.geometry.retraction import (
    _eigh_damped,
    _rel_gap_eps,
    get_retraction,
    retract_logeuclidean_full,
)
from vfe3.geometry.transport import get_transport, stable_matrix_exp_pair
from vfe3.model.model import VFEModel
from vfe3.model.prior_bank import PriorBank
from vfe3.numerics import apply_mu_trust_region


def _full_prior_bank(*, eps: float = 1e-3) -> PriorBank:
    cfg = VFE3Config(
        vocab_size=3,
        embed_dim=2,
        n_heads=1,
        max_seq_len=2,
        n_layers=1,
        n_e_steps=1,
        e_step_gradient="straight_through",
        pos_phi="none",
        family="gaussian_full",
        decode_mode="full",
        use_prior_bank=True,
        eps=eps,
    )
    return VFEModel(cfg).prior_bank


def test_halt_tol_full_covariance_uses_full_gaussian_kl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vfe3.inference.e_step as e_step_module

    group = get_group("glk")(2)
    n_gen = group.generators.shape[0]
    sigma_0 = torch.eye(2).reshape(1, 2, 2)
    sigma_1 = torch.tensor([[[1.0, -0.5], [-0.5, 1.0]]])
    belief = BeliefState(
        mu=torch.zeros(1, 2),
        sigma=sigma_0,
        phi=torch.zeros(1, n_gen),
    )
    calls = 0

    def _fixed_iteration(
        current: BeliefState,
        *args: object,
        **kwargs: object,
    ) -> BeliefState:
        nonlocal calls
        calls += 1
        return current._replace(sigma=sigma_1)

    monkeypatch.setattr(e_step_module, "e_step_iteration", _fixed_iteration)
    with torch.no_grad():
        e_step_module.e_step(
            belief,
            torch.zeros(1, 2),
            sigma_0,
            group,
            n_iter=3,
            e_phi_lr=0.1,
            e_step_halt_tol=0.05,
            family="gaussian_full",
        )

    assert calls == 2


def test_full_chunked_matches_dense_at_variance_floor_without_double_ridge() -> None:
    pb = _full_prior_bank(eps=1e-3)
    with torch.no_grad():
        pb.mu_embed.copy_(torch.tensor([[0.0, 0.0], [0.02, -0.01], [-0.03, 0.04]]))
        pb.sigma_log_embed.fill_(math.log(pb.eps))
    mu_q = torch.tensor([[[0.01, -0.02]]])
    sigma_q = torch.diag_embed(torch.full((1, 1, 2), pb.eps))

    pb.decode_mode = "full"
    dense = pb.decode(mu_q, sigma_q)
    pb.decode_mode = "full_chunked"
    chunked = pb.decode(mu_q, sigma_q)

    assert torch.allclose(chunked, dense, rtol=1e-5, atol=1e-6)

    targets = torch.tensor([[2]])
    dense_ce = F.cross_entropy(dense.reshape(-1, pb.vocab_size), targets.reshape(-1))
    chunked_ce = pb.decode_ce_full_chunked(mu_q, sigma_q, targets, chunk_size=2)
    assert torch.allclose(chunked_ce, dense_ce, rtol=1e-5, atol=1e-6)

    sigma_retry = torch.diag_embed(torch.tensor([[[-0.1, 1.0]]]))
    pb.decode_mode = "full"
    dense_retry = pb.decode(mu_q, sigma_retry)
    pb.decode_mode = "full_chunked"
    chunked_retry = pb.decode(mu_q, sigma_retry)
    assert torch.equal(torch.isneginf(chunked_retry), torch.isneginf(dense_retry))
    assert torch.allclose(chunked_retry, dense_retry, rtol=1e-5, atol=1e-6)


def test_full_cov_query_invariants_use_raw_spd_on_round_zero() -> None:
    pb = _full_prior_bank(eps=1e-3)
    sigma_q = torch.tensor([[[[0.004, 0.001], [0.001, 0.003]]]])

    diag_q, logdet_q = pb._full_cov_query_invariants(sigma_q)
    factor = torch.linalg.cholesky(sigma_q)
    expected_logdet = 2.0 * torch.log(torch.diagonal(factor, dim1=-2, dim2=-1)).sum(-1)

    assert torch.equal(diag_q, torch.diagonal(sigma_q, dim1=-2, dim2=-1))
    assert torch.equal(logdet_q, expected_logdet)


def test_full_cov_box_binds_in_mahalanobis_units() -> None:
    sigma_q = torch.tensor([[[1.0, 0.9], [0.9, 1.0]]])
    delta_mu = torch.tensor([[0.0, 1.0]])
    factor = torch.linalg.cholesky(sigma_q)

    out = apply_mu_trust_region(
        delta_mu,
        sigma_q,
        trust=1.0,
        mode="box",
        is_diagonal=False,
    )
    whitened = torch.linalg.solve_triangular(factor, out.unsqueeze(-1), upper=False).squeeze(-1)

    assert whitened.abs().max() <= 1.0 + 1e-6
    assert torch.allclose(whitened, torch.tensor([[0.0, 1.0]]), atol=1e-6)


def test_full_cov_ball_bounds_mahalanobis_norm() -> None:
    sigma_q = torch.tensor([[[1.0, 0.9], [0.9, 1.0]]])
    delta_mu = torch.tensor([[0.0, 1.0]])
    factor = torch.linalg.cholesky(sigma_q)

    out = apply_mu_trust_region(
        delta_mu,
        sigma_q,
        trust=1.0,
        mode="ball",
        is_diagonal=False,
    )
    whitened = torch.linalg.solve_triangular(factor, out.unsqueeze(-1), upper=False).squeeze(-1)

    assert torch.allclose(whitened.norm(dim=-1), torch.ones(1), atol=1e-6)


def test_full_cov_failed_cholesky_falls_back_per_element() -> None:
    sigma_q = torch.tensor(
        [
            [[1.0, 0.9], [0.9, 1.0]],
            [[1.0, 2.0], [2.0, 1.0]],
        ]
    )
    delta_mu = torch.tensor([[0.0, 3.0], [0.0, 3.0]])

    out = apply_mu_trust_region(
        delta_mu,
        sigma_q,
        trust=1.0,
        mode="box",
        is_diagonal=False,
    )

    good_factor = torch.linalg.cholesky(sigma_q[:1])
    good_white = torch.linalg.solve_triangular(
        good_factor,
        delta_mu[:1].unsqueeze(-1),
        upper=False,
    ).squeeze(-1)
    expected_good = (good_factor @ good_white.clamp(-1.0, 1.0).unsqueeze(-1)).squeeze(-1)
    expected_bad = torch.tensor([[0.0, 1.0]])

    assert torch.allclose(out[:1], expected_good, atol=1e-6)
    assert torch.allclose(out[1:], expected_bad, atol=1e-6)


def test_rel_gap_eps_is_per_matrix() -> None:
    matrices = torch.tensor(
        [
            [[2.0, -0.5], [-0.5, 1.0]],
            [[2.0e6, -5.0e5], [-5.0e5, 1.0e6]],
        ],
        dtype=torch.float64,
    )

    gap_eps = _rel_gap_eps(matrices, rel=1e-6, floor=0.0)
    expected = (1e-6 * matrices.abs().amax(dim=(-2, -1), keepdim=True)).pow(2)

    assert gap_eps.shape == (2, 1, 1)
    assert torch.equal(gap_eps, expected)


def test_eigh_damped_gradient_is_batch_separable_across_scales() -> None:
    small = torch.tensor([[2.0, 0.3], [0.3, 1.0]], dtype=torch.float64)
    large = 1.0e8 * torch.tensor([[1.5, -0.2], [-0.2, 0.8]], dtype=torch.float64)
    weight = torch.tensor([[1.0, 0.4], [0.4, -0.7]], dtype=torch.float64)

    single = small.clone().requires_grad_(True)
    single_eig, single_vec = _eigh_damped(single, _rel_gap_eps(single, floor=0.0))
    single_sqrt = (
        single_vec * single_eig.sqrt().unsqueeze(-2)
    ) @ single_vec.transpose(-1, -2)
    single_grad = torch.autograd.grad((single_sqrt * weight).sum(), single)[0]

    batch = torch.stack((small, large)).requires_grad_(True)
    batch_eig, batch_vec = _eigh_damped(batch, _rel_gap_eps(batch, floor=0.0))
    batch_sqrt = (
        batch_vec * batch_eig.sqrt().unsqueeze(-2)
    ) @ batch_vec.transpose(-1, -2)
    batch_grad = torch.autograd.grad((batch_sqrt[0] * weight).sum(), batch)[0][0]

    assert torch.allclose(batch_grad, single_grad, rtol=1e-10, atol=1e-12)


def test_log_euclidean_full_retraction_has_identity_first_derivative() -> None:
    sigma = torch.tensor(
        [[2.0, 0.3], [0.3, 0.8]],
        dtype=torch.float64,
    )
    tangent = torch.tensor(
        [[0.2, -0.1], [-0.1, 0.05]],
        dtype=torch.float64,
    )
    fd_step = 1e-6

    out_plus = retract_logeuclidean_full(
        sigma,
        fd_step * tangent,
        trust_region=0.0,
        eps=1e-12,
        sigma_max=None,
    )
    out_minus = retract_logeuclidean_full(
        sigma,
        -fd_step * tangent,
        trust_region=0.0,
        eps=1e-12,
        sigma_max=None,
    )
    derivative = (out_plus - out_minus) / (2.0 * fd_step)

    assert torch.allclose(derivative, tangent, rtol=2e-6, atol=2e-7)


def test_log_euclidean_scalar_uses_h_over_sigma_chart_tangent() -> None:
    sigma = torch.tensor([[2.0]])
    tangent = torch.tensor([[0.6]])

    out = get_retraction("log_euclidean")(
        sigma,
        tangent,
        mean_ndim=2,
        trust_region=0.0,
        eps=1e-6,
        sigma_max=None,
    )
    expected = sigma * torch.exp(tangent / sigma)

    assert torch.allclose(out, expected, rtol=1e-6, atol=1e-6)


def test_regime_ii_soft_cap_is_finite_before_squaring() -> None:
    group = get_group("glk")(2)
    n_gen = group.generators.shape[0]
    phi = torch.zeros(1, 2, n_gen)
    mu = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    connection_W = torch.zeros(n_gen, 2, 2)
    connection_W[0, 0, 0] = 1.0e20
    connection_W.requires_grad_(True)

    omega = get_transport("regime_ii")(
        phi,
        group,
        mu=mu,
        connection_W=connection_W,
        cocycle_relaxation=1.0,
        delta_soft_cap=12.0,
    )["Omega"]
    omega.sum().backward()

    assert torch.isfinite(omega).all()
    assert not torch.allclose(omega[0, 0, 1], torch.eye(2))
    assert connection_W.grad is not None
    assert torch.isfinite(connection_W.grad).all()


def test_regime_ii_covariant_soft_cap_is_finite_before_squaring() -> None:
    group = get_group("glk")(2)
    n_gen = group.generators.shape[0]
    phi = torch.zeros(1, 2, n_gen)
    mu = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]])
    sigma = torch.ones(1, 2, 2)
    connection_M = torch.zeros(n_gen, 3)
    connection_M[0, 0] = 1.0e20
    connection_M.requires_grad_(True)

    omega = get_transport("regime_ii_covariant")(
        phi,
        group,
        mu=mu,
        sigma=sigma,
        connection_M=connection_M,
        cocycle_relaxation=1.0,
        delta_soft_cap=12.0,
    )["Omega"]
    omega.sum().backward()

    assert torch.isfinite(omega).all()
    assert not torch.allclose(omega[0, 0, 1], torch.eye(2))
    assert connection_M.grad is not None
    assert torch.isfinite(connection_M.grad).all()


def test_regime_ii_direct_link_soft_cap_is_finite_before_squaring() -> None:
    group = get_group("glk")(2)
    n_gen = group.generators.shape[0]
    phi = torch.zeros(1, 2, n_gen)
    connection_L = torch.zeros(2, 2, n_gen)
    connection_L[0, 1, 0] = 1.0e20
    connection_L[1, 0, 0] = 1.0e20
    connection_L.requires_grad_(True)

    omega = get_transport("regime_ii_link")(
        phi,
        group,
        connection_L=connection_L,
        link_alpha=1.0,
        link_soft_cap=6.0,
    )["Omega"]
    omega.sum().backward()

    assert torch.isfinite(omega).all()
    assert not torch.allclose(omega[0, 1], torch.eye(2))
    assert connection_L.grad is not None
    assert torch.isfinite(connection_L.grad).all()


def test_large_skew_dim_mode_matches_float64_matrix_exp() -> None:
    matrix = torch.tensor([[0.0, 1000.0], [-1000.0, 0.0]])

    exp_pos, exp_neg = stable_matrix_exp_pair(
        matrix,
        skew_symmetric=True,
        max_norm=float("inf"),
        exp_fp64_mode="dim",
        exp_fp64_norm_threshold=5.0,
    )
    reference = torch.linalg.matrix_exp(matrix.double()).float()

    assert torch.equal(exp_pos, reference)
    assert torch.equal(exp_neg, reference.transpose(-1, -2))


def test_small_skew_dim_mode_keeps_float32_identity() -> None:
    matrix = torch.tensor([[0.0, 0.25], [-0.25, 0.0]])

    exp_pos, exp_neg = stable_matrix_exp_pair(
        matrix,
        skew_symmetric=True,
        max_norm=float("inf"),
        exp_fp64_mode="dim",
        exp_fp64_norm_threshold=5.0,
    )
    reference = torch.linalg.matrix_exp(matrix)

    assert torch.equal(exp_pos, reference)
    assert torch.equal(exp_neg, reference.transpose(-1, -2))


def test_laplace_renyi_large_separation_has_finite_gradients() -> None:
    mu_q = torch.tensor([[0.0]], requires_grad=True)
    b_q = torch.tensor([[1.0]], requires_grad=True)
    mu_p = torch.tensor([[2000.0]], requires_grad=True)
    b_p = torch.tensor([[1.0]], requires_grad=True)

    divergence = DiagonalLaplace(mu_q, b_q)._renyi_terms(
        DiagonalLaplace(mu_p, b_p),
        alpha=1.5,
        eps=1e-6,
    )
    gradients = torch.autograd.grad(divergence.sum(), (mu_q, b_q, mu_p, b_p))

    assert torch.isfinite(divergence).all()
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_laplace_renyi_alpha_gt_one_matches_numerical_quadrature() -> None:
    alpha = 1.5
    mu_q = torch.tensor([[0.0]])
    b_q = torch.tensor([[0.75]])
    mu_p = torch.tensor([[2.0]])
    b_p = torch.tensor([[1.0]])
    got = DiagonalLaplace(mu_q, b_q)._renyi_terms(
        DiagonalLaplace(mu_p, b_p),
        alpha=alpha,
        eps=1e-6,
    ).double()

    x = torch.linspace(-40.0, 40.0, 200_001, dtype=torch.float64)
    log_q = -math.log(2.0 * float(b_q)) - (x - float(mu_q)).abs() / float(b_q)
    log_p = -math.log(2.0 * float(b_p)) - (x - float(mu_p)).abs() / float(b_p)
    integral = torch.trapezoid(torch.exp(alpha * log_q + (1.0 - alpha) * log_p), x)
    expected = torch.log(integral) / (alpha - 1.0)

    assert torch.allclose(got.squeeze(), expected, rtol=2e-5, atol=2e-5)


def test_laplace_renyi_divergent_blend_clamps_without_poisoning_gradients() -> None:
    kl_max = 123.0
    mu_q = torch.tensor([[0.0, 0.0]], requires_grad=True)
    b_q = torch.tensor([[1.0, 4.0]], requires_grad=True)
    mu_p = torch.tensor([[2.0, 2000.0]])
    b_p = torch.ones(1, 2)

    got = DiagonalLaplace(mu_q, b_q).renyi_per_coord(
        DiagonalLaplace(mu_p, b_p),
        alpha=1.5,
        kl_max=kl_max,
    )
    good = DiagonalLaplace(mu_q[:, :1], b_q[:, :1]).renyi_per_coord(
        DiagonalLaplace(mu_p[:, :1], b_p[:, :1]),
        alpha=1.5,
        kl_max=kl_max,
    )
    got.sum().backward()

    assert torch.equal(got[:, :1], good)
    assert torch.equal(got[:, 1:], torch.full_like(got[:, 1:], kl_max))
    assert mu_q.grad is not None and torch.isfinite(mu_q.grad).all()
    assert b_q.grad is not None and torch.isfinite(b_q.grad).all()
