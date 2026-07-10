r"""Focused regressions for family-aware full-covariance inference geometry."""

import ast
import gc
import inspect
import math
import warnings
import weakref
from collections import Counter

import pytest
import torch
import torch.nn.functional as F

import vfe3.geometry.retraction as retraction_module
import vfe3.model.model as model_module
import vfe3.model.prior_bank as prior_bank_module
import vfe3.numerics as numerics_module
import vfe3.viz.extract as extract_module
from vfe3.belief import BeliefState
from vfe3.config import VFE3Config
from vfe3.divergence import get_functional
from vfe3.families.gaussian import DiagonalGaussian
from vfe3.families.laplace import DiagonalLaplace
from vfe3.geometry.groups import get_group
from vfe3.geometry.phi_preconditioner import (
    precondition_phi_gradient,
    pullback_metric,
    pullback_metric_per_block,
)
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


def test_bounded_log_variance_exp_is_finite_with_finite_gradient() -> None:
    log_sigma = torch.tensor([100.0], requires_grad=True)

    with pytest.warns(RuntimeWarning, match="trainable log-variance"):
        sigma = numerics_module.bounded_variance_from_log(log_sigma)
    sigma.sum().backward()

    assert torch.isfinite(sigma).all()
    assert log_sigma.grad is not None
    assert torch.isfinite(log_sigma.grad).all()


def test_bounded_log_variance_is_identity_in_normal_range() -> None:
    log_sigma = torch.tensor([-5.0, 0.0, 5.0])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sigma = numerics_module.bounded_variance_from_log(log_sigma)

    assert torch.equal(sigma, torch.exp(log_sigma))


def test_prior_model_and_decode_variance_reads_share_guard() -> None:
    production_modules = {
        "model":      model_module,
        "prior_bank": prior_bank_module,
        "extract":    extract_module,
    }
    retraction_source = inspect.getsource(retraction_module)
    known_trainable_log_variances = (
        "sigma_log_embed",
        "s_sigma_log_embed",
        "decode_sigma_log_embed",
        "r_sigma_log",
        "_prior_sigma_log_table",
        "_decode_sigma_log_table",
    )
    expected_guarded_reads = {
        "model": Counter({"pb.r_sigma_log": 2}),
        "prior_bank": Counter(
            {
                "self.s_sigma_log_embed[token_ids]": 1,
                "self.s_sigma_log_embed": 1,
                "self._decode_sigma_log_table()": 4,
                "pb._prior_sigma_log_table()[token_ids]": 2,
                "pb._decode_sigma_log_table()": 4,
            }
        ),
        "extract": Counter({"pb.r_sigma_log": 2}),
    }

    direct_table_exponentiations = []
    guarded_reads = {}
    for module_name, module in production_modules.items():
        source = inspect.getsource(module)
        tree = ast.parse(source)
        module_guarded_reads = Counter()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and node.func.id == "bounded_variance_from_log":
                argument = ast.unparse(node.args[0])
                if any(name in argument for name in known_trainable_log_variances):
                    module_guarded_reads[argument] += 1
                continue
            is_torch_exp = (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "exp"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "torch"
            )
            is_tensor_exp = isinstance(node.func, ast.Attribute) and node.func.attr == "exp"
            if is_torch_exp and node.args:
                exponent = ast.unparse(node.args[0])
            elif is_tensor_exp:
                exponent = ast.unparse(node.func.value)
            else:
                continue
            if any(name in exponent for name in known_trainable_log_variances):
                direct_table_exponentiations.append(
                    f"{module_name}:{node.lineno}:{ast.unparse(node)}"
                )
        guarded_reads[module_name] = module_guarded_reads

    retraction_tree = ast.parse(retraction_source)
    retraction_guard_calls = [
        node.lineno
        for node in ast.walk(retraction_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "bounded_variance_from_log"
    ]

    assert direct_table_exponentiations == []
    assert guarded_reads == expected_guarded_reads
    assert retraction_guard_calls == []


def _trainable_r_extractor_model() -> VFEModel:
    cfg = VFE3Config(
        vocab_size=6,
        embed_dim=4,
        n_heads=2,
        max_seq_len=4,
        n_layers=1,
        n_e_steps=1,
        e_phi_lr=0.0,
        prior_source="model_channel",
        s_e_step=True,
        lambda_h=1.0,
        learnable_r=True,
        pos_phi="none",
    )
    return VFEModel(cfg)


def test_s_channel_refinement_bounds_large_trainable_centroid_variance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _trainable_r_extractor_model()
    pb = model.prior_bank
    token_ids = torch.tensor([[0, 1, 2, 3]])

    def _identity_refine(
        tokens: torch.Tensor,
        phi:    torch.Tensor,

        *,
        rope: 'torch.Tensor | None' = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del phi, rope
        return pb.encode_s(tokens)

    monkeypatch.setattr(model, "_refine_s", _identity_refine)
    with torch.no_grad():
        pb.r_sigma_log.fill_(0.25)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        normal = extract_module.s_channel_refinement(model, token_ids)
    assert normal is not None

    s_mu, s_sigma = (tensor[0] for tensor in pb.encode_s(token_ids))
    r_mu = pb.r_mu.expand_as(s_mu)
    r_sigma = torch.exp(pb.r_sigma_log).expand_as(s_sigma)
    expected = get_functional("renyi")(
        DiagonalGaussian(s_mu, s_sigma),
        DiagonalGaussian(r_mu, r_sigma),
        alpha=1.0,
        kl_max=model.cfg.kl_max,
        eps=model.cfg.eps,
    ).cpu()
    assert torch.equal(normal["kl_s0_r"], expected)
    assert torch.equal(normal["kl_s1_r"], expected)

    with torch.no_grad():
        pb.r_sigma_log.fill_(100.0)
    with pytest.warns(RuntimeWarning, match="trainable log-variance"):
        bounded = extract_module.s_channel_refinement(model, token_ids)

    assert bounded is not None
    assert all(torch.isfinite(value).all() for value in bounded.values())


def test_hyper_prior_centroid_bounds_large_trainable_centroid_variance() -> None:
    model = _trainable_r_extractor_model()
    pb = model.prior_bank
    token_ids = torch.tensor([[0, 1, 2, 3]])

    with torch.no_grad():
        pb.r_sigma_log.fill_(0.25)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        normal = extract_module.hyper_prior_centroid(model, token_ids)
    assert normal is not None
    assert torch.equal(normal["r_sigma"], torch.exp(pb.r_sigma_log).detach().cpu())

    with torch.no_grad():
        pb.r_sigma_log.fill_(100.0)
    with pytest.warns(RuntimeWarning, match="trainable log-variance"):
        bounded = extract_module.hyper_prior_centroid(model, token_ids)

    assert bounded is not None
    assert torch.isfinite(bounded["r_sigma"]).all()


def _matrix_exp_jacobian_pullback_metric(
    phi:        torch.Tensor,             # (n_gen,) frame coordinates
    generators: torch.Tensor,             # (n_gen, K, K) basis
) -> torch.Tensor:                        # (n_gen, n_gen) float64 pullback metric
    """Independent float64 pullback metric from the Jacobian of ``matrix_exp``."""
    phi_64 = phi.double().requires_grad_(True)
    generators_64 = generators.double()

    def _exp_from_coordinates(coordinates: torch.Tensor) -> torch.Tensor:
        algebra = torch.einsum("a,aij->ij", coordinates, generators_64)
        return torch.linalg.matrix_exp(algebra)

    jacobian = torch.autograd.functional.jacobian(
        _exp_from_coordinates,
        phi_64,
        create_graph=False,
    )
    dexp = jacobian.movedim(-1, 0)
    return torch.einsum("aij,bij->ab", dexp, dexp)


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


def test_pullback_preconditioner_matches_float64_ill_conditioned_reference() -> None:
    group = get_group("glk")(2)
    phi = torch.tensor([4.4922614, -0.8553943, -6.3515353, 1.6570722])
    grad_phi = torch.tensor([-1.0845224, -1.3985955, 0.40334684, 0.83802634])

    got = precondition_phi_gradient(
        grad_phi,
        phi,
        group.generators,
        mode="pullback",
    )
    metric_64 = _matrix_exp_jacobian_pullback_metric(phi, group.generators)
    eye_64 = torch.eye(metric_64.shape[-1], dtype=torch.float64)
    reference_64 = torch.linalg.solve(
        metric_64 + 1e-6 * eye_64,
        grad_phi.double().unsqueeze(-1),
    ).squeeze(-1)

    assert pullback_metric(phi, group.generators).dtype == torch.float64
    assert got.dtype == grad_phi.dtype
    assert torch.allclose(got.double(), reference_64, rtol=1e-5, atol=1e-6)


def test_pullback_per_block_solve_stays_float64_until_final_cast() -> None:
    group = get_group("block_glk")(4, 2)
    irrep_dims = [2, 2]
    phi_block = torch.tensor([4.4922614, -0.8553943, -6.3515353, 1.6570722])
    grad_block = torch.tensor([-1.0845224, -1.3985955, 0.40334684, 0.83802634])
    phi = phi_block.repeat(2)
    grad_phi = grad_block.repeat(2)

    got = precondition_phi_gradient(
        grad_phi,
        phi,
        group.generators,
        mode="pullback_per_block",
        irrep_dims=irrep_dims,
    )
    metric_64 = _matrix_exp_jacobian_pullback_metric(phi, group.generators)
    eye_64 = torch.eye(metric_64.shape[-1], dtype=torch.float64)
    reference_64 = torch.linalg.solve(
        metric_64 + 1e-6 * eye_64,
        grad_phi.double().unsqueeze(-1),
    ).squeeze(-1)

    assert pullback_metric_per_block(phi, group.generators, irrep_dims).dtype == torch.float64
    assert got.dtype == grad_phi.dtype
    assert torch.allclose(got.double(), reference_64, rtol=1e-5, atol=1e-6)


def test_pullback_closure_cache_is_bounded_and_releases_cuda_basis() -> None:
    import vfe3.geometry.lie_ops as lie_ops

    cache_max = getattr(lie_ops, "_BRACKET_CLOSURE_CACHE_MAXSIZE", 32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lie_ops._BRACKET_CLOSURE_RES.clear()
    lie_ops._BRACKET_CLOSURE_WARNED.clear()
    try:
        basis = torch.tensor([[[1.0]]], device=device)
        basis_ref = weakref.ref(basis)
        lie_ops.warn_if_basis_not_closed(basis, where="curated-cache-release")
        lie_ops.warn_if_basis_not_closed(basis.clone(), where="curated-cache-release")

        assert len(lie_ops._BRACKET_CLOSURE_RES) == 1
        first_key = next(iter(lie_ops._BRACKET_CLOSURE_RES))
        second_basis = torch.tensor([[[2.0]]], device=device)
        lie_ops.warn_if_basis_not_closed(second_basis, where="curated-cache-bound")
        second_key = next(reversed(lie_ops._BRACKET_CLOSURE_RES))
        lie_ops.warn_if_basis_not_closed(basis, where="curated-cache-release")
        del basis
        del second_basis
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        assert basis_ref() is None

        for value in range(3, cache_max + 2):
            candidate = torch.tensor([[[float(value)]]], device=device)
            lie_ops.warn_if_basis_not_closed(candidate, where="curated-cache-bound")

        assert len(lie_ops._BRACKET_CLOSURE_RES) == cache_max
        assert first_key in lie_ops._BRACKET_CLOSURE_RES
        assert second_key not in lie_ops._BRACKET_CLOSURE_RES
        assert all(
            not isinstance(entry, torch.Tensor)
            for entry in lie_ops._BRACKET_CLOSURE_RES.values()
        )
    finally:
        lie_ops._BRACKET_CLOSURE_RES.clear()
        lie_ops._BRACKET_CLOSURE_WARNED.clear()


def test_config_rejects_nonclosed_cross_couplings_for_phi_bch() -> None:
    with pytest.raises(ValueError, match="close_basis=True"):
        VFE3Config(
            embed_dim=6,
            n_heads=3,
            gauge_group="block_glk",
            cross_couplings=[(0, 1), (1, 2)],
            close_basis=False,
            family="gaussian_full",
            beta_attention_prior="uniform",
            gamma_attention_prior="uniform",
            e_phi_lr=0.1,
            phi_retract_mode="bch",
            pos_phi="none",
        )


def test_config_rejects_nonclosed_cross_couplings_for_positional_bch() -> None:
    with pytest.raises(ValueError, match="close_basis=True"):
        VFE3Config(
            embed_dim=6,
            n_heads=3,
            gauge_group="block_glk",
            cross_couplings=[(0, 1), (1, 2)],
            close_basis=False,
            family="gaussian_full",
            beta_attention_prior="uniform",
            gamma_attention_prior="uniform",
            phi_retract_mode="euclidean",
            pos_phi="learned",
            pos_phi_compose="bch",
        )


def test_config_allows_nonclosed_basis_when_bch_inactive() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = VFE3Config(
            embed_dim=6,
            n_heads=3,
            gauge_group="block_glk",
            cross_couplings=[(0, 1), (1, 2)],
            close_basis=False,
            family="gaussian_full",
            beta_attention_prior="uniform",
            gamma_attention_prior="uniform",
            phi_retract_mode="euclidean",
            pos_phi="none",
            pos_phi_compose="bch",
        )

    assert cfg.close_basis is False


def test_config_allows_bracket_closed_disjoint_cross_couplings_with_bch() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = VFE3Config(
            embed_dim=8,
            n_heads=4,
            gauge_group="block_glk",
            cross_couplings=[(0, 1), (2, 3)],
            close_basis=False,
            family="gaussian_full",
            beta_attention_prior="uniform",
            gamma_attention_prior="uniform",
            e_phi_lr=0.1,
            phi_retract_mode="bch",
            pos_phi="none",
        )

    assert cfg.close_basis is False


def test_config_allows_bracket_closed_fanout_cross_couplings_with_bch() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = VFE3Config(
            embed_dim=6,
            n_heads=3,
            gauge_group="block_glk",
            cross_couplings=[(0, 1), (0, 2)],
            close_basis=False,
            family="gaussian_full",
            beta_attention_prior="uniform",
            gamma_attention_prior="uniform",
            e_phi_lr=0.1,
            phi_retract_mode="bch",
            pos_phi="none",
        )

    assert cfg.close_basis is False


def test_config_allows_bracket_closed_transitive_cross_couplings_with_bch() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = VFE3Config(
            embed_dim=6,
            n_heads=3,
            gauge_group="block_glk",
            cross_couplings=[(0, 1), (1, 2), (0, 2)],
            close_basis=False,
            family="gaussian_full",
            beta_attention_prior="uniform",
            gamma_attention_prior="uniform",
            e_phi_lr=0.1,
            phi_retract_mode="bch",
            pos_phi="none",
        )

    assert cfg.close_basis is False


def test_config_allows_phi_bch_when_e_phi_route_inactive() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = VFE3Config(
            embed_dim=6,
            n_heads=3,
            gauge_group="block_glk",
            cross_couplings=[(0, 1), (1, 2)],
            close_basis=False,
            family="gaussian_full",
            beta_attention_prior="uniform",
            gamma_attention_prior="uniform",
            e_phi_lr=0.0,
            phi_retract_mode="bch",
            pos_phi="none",
        )

    assert cfg.e_phi_lr == 0.0
