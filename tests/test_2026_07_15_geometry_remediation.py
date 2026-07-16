from pathlib import Path

import pytest
import torch

from vfe3.belief import BeliefState
from vfe3.config import VFE3Config
from vfe3.geometry import lie_ops
from vfe3.geometry.groups import get_group
from vfe3.geometry.lie_ops import compose_phi, embed_phi, retract_omega
from vfe3.geometry.retraction import _eigh_damped, _rel_gap_eps, retract_spd_full
from vfe3.geometry.transport import (
    _stable_compact_glk_exp_pair,
    gauge_invariant_edge_features,
    get_transport,
    group_element_inverse,
    stable_matrix_exp_pair,
)
from vfe3.gradients.pairwise_stats import diagonal_kl_pair_stats
from vfe3.inference.e_step import e_step_iteration, free_energy_value, phi_alignment_loss
from vfe3.model.model import VFEModel
from vfe3.run_artifacts import RunArtifacts, _pure_path_report


def _congruence(
    frame:      torch.Tensor,
    covariance: torch.Tensor,
) -> torch.Tensor:
    return frame @ covariance @ frame.transpose(-1, -2)


def test_valid_spd_edge_features_are_exact_under_nonorthogonal_congruence() -> None:
    dtype = torch.float64
    mu_q = torch.tensor([0.2, -0.3], dtype=dtype)
    mu_t = torch.tensor([-0.1, 0.4], dtype=dtype)
    cov_q = torch.diag(torch.tensor([1.0e-6, 2.0], dtype=dtype))
    cov_t = torch.diag(torch.tensor([2.0e-6, 0.7], dtype=dtype))
    frame = torch.tensor([[1000.0, 0.4], [0.0, 0.5]], dtype=dtype)

    features = gauge_invariant_edge_features(mu_q, cov_q, mu_t, cov_t)
    transformed = gauge_invariant_edge_features(
        frame @ mu_q,
        _congruence(frame, cov_q),
        frame @ mu_t,
        _congruence(frame, cov_t),
    )

    assert torch.allclose(features, transformed, atol=1.0e-10, rtol=1.0e-10)


def test_jitter_recovery_returns_false_exactness_status() -> None:
    mu = torch.zeros(2, dtype=torch.float64)
    covariance = torch.diag(torch.tensor([0.0, 1.0], dtype=torch.float64))

    features, exact = gauge_invariant_edge_features(
        mu,
        covariance,
        mu,
        covariance,
        return_exactness=True,
    )

    assert torch.isfinite(features).all()
    assert exact.shape == torch.Size([])
    assert not bool(exact)


def test_covariant_builder_exposes_jitter_exactness_status() -> None:
    group = get_group("glk")(K=2)
    phi = torch.zeros(1, 2, group.generators.shape[0])
    mu = torch.zeros(1, 2, 2)
    covariance = torch.diag(torch.tensor([0.0, 1.0])).expand(1, 2, 2, 2).clone()
    connection = torch.zeros(group.generators.shape[0], 3, requires_grad=True)
    exactness: dict[str, torch.Tensor] = {}

    transport = get_transport("regime_ii_covariant")(
        phi,
        group,
        mu=mu,
        sigma=covariance,
        connection_M=connection,
        exactness_out=exactness,
    )

    assert set(transport) == {"exp_phi", "exp_neg_phi", "Omega"}
    assert not bool(exactness["regime_ii_covariant_feature_exact"])


def test_production_jitter_status_reaches_logged_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path:    Path,
) -> None:
    cfg = VFE3Config(
        vocab_size=5,
        embed_dim=2,
        n_heads=1,
        max_seq_len=2,
        n_layers=1,
        n_e_steps=1,
        family="gaussian_full",
        transport_mode="regime_ii_covariant",
    )
    model = VFEModel(cfg)
    original_encode = model.prior_bank.encode

    def _singular_encode(token_ids: torch.Tensor) -> BeliefState:
        belief = original_encode(token_ids)
        singular = torch.zeros_like(belief.sigma)
        singular[..., 1, 1] = 1.0
        return belief._replace(sigma=singular)

    monkeypatch.setattr(model.prior_bank, "encode", _singular_encode)
    tokens = torch.tensor([[0, 1]])
    model.forward_beliefs(tokens)
    metrics = model.diagnostics(tokens)

    monkeypatch.setattr(model.prior_bank, "encode", original_encode)
    model.forward_beliefs(tokens)
    recovered_metrics = model.diagnostics(tokens)

    artifacts = RunArtifacts(tmp_path / "run", cfg, model)
    artifacts.log_metrics({"step": 1.0, **recovered_metrics})
    report = _pure_path_report(cfg, artifacts.history)

    assert metrics["regime_ii_covariant_feature_exact"] == 0.0
    assert recovered_metrics["regime_ii_covariant_feature_exact"] == 0.0
    assert report["config_toggles"]["regime_ii_covariant_exact"] is False
    assert report["config_toggles"]["regime_ii_covariant_exactness"] == "jitter_recovered_approximation"


def test_artifact_exactness_is_false_after_feature_jitter_recovery() -> None:
    cfg = VFE3Config(transport_mode="regime_ii_covariant", family="gaussian_full")
    report = _pure_path_report(cfg, [{"regime_ii_covariant_feature_exact": 0.0}])

    assert report["config_toggles"]["regime_ii_covariant_exact"] is False
    assert report["config_toggles"]["regime_ii_covariant_exactness"] == "jitter_recovered_approximation"


def test_capped_and_uncapped_airm_routes_have_distinct_artifact_labels() -> None:
    projected = _pure_path_report(VFE3Config(spd_retract_mode="spd_affine", sigma_max=10.0), [])
    exact = _pure_path_report(VFE3Config(spd_retract_mode="spd_affine", sigma_max=None), [])

    assert projected["config_toggles"]["spd_retraction_route"] == "airm_projected_spectral_cap"
    assert projected["config_toggles"]["spd_retraction_exact"] is False
    assert exact["config_toggles"]["spd_retraction_route"] == "airm_exact"
    assert exact["config_toggles"]["spd_retraction_exact"] is True


def test_transport_chart_bound_fails_before_exponential_clamp() -> None:
    matrix = 30.0 * torch.eye(2)

    with pytest.raises(ValueError, match="transport chart validity bound"):
        stable_matrix_exp_pair(matrix, validity_max_norm=20.0)


def test_configured_transport_chart_bound_reaches_flat_model_vertex_exponential() -> None:
    cfg = VFE3Config(
        vocab_size=5,
        embed_dim=2,
        n_heads=1,
        max_seq_len=2,
        n_layers=1,
        n_e_steps=1,
        transport_chart_max_norm=0.5,
    )
    model = VFEModel(cfg)
    with torch.no_grad():
        model.prior_bank.phi_embed.fill_(2.0)

    with pytest.raises(ValueError, match="transport chart validity bound"):
        model.forward_beliefs(torch.tensor([[0, 1]]))


def test_configured_transport_chart_bound_reaches_covariant_connection_exponential() -> None:
    cfg = VFE3Config(
        vocab_size=5,
        embed_dim=2,
        n_heads=1,
        max_seq_len=2,
        n_layers=1,
        n_e_steps=1,
        family="gaussian_full",
        transport_mode="regime_ii_covariant",
        transport_chart_max_norm=1.0,
    )
    model = VFEModel(cfg)
    with torch.no_grad():
        model.prior_bank.mu_embed.copy_(
            torch.arange(model.prior_bank.mu_embed.numel()).reshape_as(model.prior_bank.mu_embed)
        )
        model.connection_M.fill_(1000.0)

    with pytest.raises(ValueError, match="transport chart validity bound"):
        model.forward_beliefs(torch.tensor([[0, 1]]))


@pytest.mark.parametrize("bound", [0.0, -1.0, float("inf"), float("nan")])
def test_transport_chart_bound_config_requires_positive_finite_value(bound: float) -> None:
    with pytest.raises(ValueError, match="transport_chart_max_norm"):
        VFE3Config(transport_chart_max_norm=bound)


def test_phi_substep_rejects_retracted_frame_outside_chart_bound() -> None:
    group = get_group("glk")(K=2)
    mu = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    sigma = torch.ones(2, 2)
    belief = BeliefState(
        mu=mu,
        sigma=sigma,
        phi=torch.zeros(2, group.generators.shape[0]),
    )

    with pytest.raises(ValueError, match="transport chart validity bound"):
        e_step_iteration(
            belief,
            mu.clone(),
            sigma.clone(),
            group,
            e_q_mu_lr=0.0,
            e_q_sigma_lr=0.0,
            e_phi_lr=0.1,
            transport_chart_max_norm=0.01,
        )


@pytest.mark.parametrize("evaluator", ("alignment", "free-energy"))
def test_objective_evaluators_honor_transport_chart_bound(evaluator: str) -> None:
    group = get_group("glk")(K=2)
    mu = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    sigma = torch.ones(2, 2)
    phi = torch.ones(2, group.generators.shape[0])

    with pytest.raises(ValueError, match="transport chart validity bound"):
        if evaluator == "alignment":
            phi_alignment_loss(
                mu,
                sigma,
                phi,
                group,
                transport_chart_max_norm=0.1,
            )
        else:
            free_energy_value(
                BeliefState(mu=mu, sigma=sigma, phi=phi),
                mu,
                sigma,
                group,
                transport_chart_max_norm=0.1,
            )


def test_bch_residual_bound_fails_closed() -> None:
    group = get_group("so_k")(K=3)
    left = torch.tensor([1.0, 0.5, -0.3])
    right = torch.tensor([-0.4, 0.7, 0.9])

    with pytest.raises(ValueError, match="BCH residual validity bound"):
        compose_phi(
            left,
            right,
            group.generators,
            mode="bch",
            order=4,
            residual_max=1.0e-6,
        )


def test_bch_residual_bound_rejects_nonfinite_group_product() -> None:
    group = get_group("glk")(K=1)

    with pytest.raises(ValueError, match="nonfinite"):
        compose_phi(
            torch.tensor([1000.0]),
            torch.tensor([1000.0]),
            group.generators,
            mode="bch",
            order=4,
            residual_max=1.0e-4,
        )


def test_euclidean_positional_composition_ignores_bch_residual_bound() -> None:
    cfg = VFE3Config(
        vocab_size=8,
        embed_dim=2,
        n_heads=1,
        max_seq_len=4,
        n_layers=1,
        n_e_steps=1,
        pos_phi="frozen",
        pos_phi_compose="euclidean",
        bch_residual_max=0.1,
    )

    logits = VFEModel(cfg)(torch.tensor([[0, 1, 2, 3]]))

    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
@pytest.mark.parametrize("compact", (False, True))
def test_transport_chart_bound_rejects_nonfinite_matrix_norm(
    value:   float,
    compact: bool,
) -> None:
    matrix = torch.tensor([[value, 0.0], [0.0, 0.0]])

    with pytest.raises(ValueError, match="nonfinite"):
        if compact:
            _stable_compact_glk_exp_pair(
                matrix.reshape(1, 2, 2),
                validity_max_norm=1.0,
            )
        else:
            stable_matrix_exp_pair(matrix, validity_max_norm=1.0)


def test_near_orthogonal_frame_uses_true_inverse_for_exact_cocycle() -> None:
    group = get_group("so_k")(K=3)
    omega = torch.eye(3)
    omega[0, 1] = 2.0e-5

    inverse = group_element_inverse(omega, group)
    residual = omega @ inverse - torch.eye(3)

    assert torch.linalg.matrix_norm(residual) < 2.0e-7
    assert torch.allclose(inverse, torch.linalg.inv(omega.double()).float(), atol=1.0e-7, rtol=0.0)


def test_high_dimensional_near_orthogonal_frame_uses_true_inverse() -> None:
    dimension = 100
    group = get_group("so_k")(K=dimension)
    omega = torch.eye(dimension)
    omega[0, 0] = 1.0 + 2.1458e-5

    inverse = group_element_inverse(omega, group)
    reference = torch.linalg.inv(omega.double()).float()

    torch.testing.assert_close(inverse, reference, atol=0.0, rtol=0.0)
    torch.testing.assert_close(omega @ inverse, torch.eye(dimension), atol=0.0, rtol=0.0)


def test_omega_retraction_dispatches_through_registry() -> None:
    register = getattr(lie_ops, "register_omega_retraction", None)
    registry = getattr(lie_ops, "_OMEGA_RETRACTIONS", None)
    assert callable(register)
    assert isinstance(registry, dict)

    name = "_test_registry_selected"
    calls = {"count": 0}

    @register(name)
    def _registry_retraction(algebra_step: torch.Tensor) -> torch.Tensor:
        calls["count"] += 1
        dimension = algebra_step.shape[-1]
        return 2.0 * torch.eye(dimension, dtype=algebra_step.dtype, device=algebra_step.device)

    try:
        generators = torch.eye(2).reshape(1, 2, 2)
        result = retract_omega(torch.eye(2), torch.zeros(1), generators, mode=name)
        assert calls["count"] == 1
        assert torch.equal(result, 2.0 * torch.eye(2))
    finally:
        registry.pop(name, None)


def test_small_float32_diagonal_kl_matches_float64_reference_before_mask() -> None:
    K = 20
    mu_q = torch.zeros(1, 1, K, dtype=torch.float32)
    sigma_q = torch.ones(1, 1, K, dtype=torch.float32)
    mu_t = torch.zeros(1, 1, 1, K, dtype=torch.float32)
    sigma_t = torch.full((1, 1, 1, K), 1.0001, dtype=torch.float32)

    stats = diagonal_kl_pair_stats(mu_q, sigma_q, mu_t, sigma_t)
    ratio = sigma_q.double().unsqueeze(-2) / sigma_t.double()
    reference = 0.5 * (
        ratio - 1.0 + torch.log(sigma_t.double()) - torch.log(sigma_q.double()).unsqueeze(-2)
    ).sum(dim=-1)

    assert stats.energy.dtype == torch.float32
    assert stats.energy.item() > 0.0
    assert torch.allclose(stats.energy.double(), reference, atol=1.0e-12, rtol=5.0e-5)
    assert stats.pair_mask.item() == 1.0


def test_near_floor_spectral_gradient_matches_float64_oracle() -> None:
    angle = torch.tensor(0.37, dtype=torch.float64)
    cosine, sine = torch.cos(angle), torch.sin(angle)
    rotation = torch.stack((
        torch.stack((cosine, -sine, torch.tensor(0.0, dtype=torch.float64))),
        torch.stack((sine, cosine, torch.tensor(0.0, dtype=torch.float64))),
        torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
    ))
    eigenvalues = torch.tensor([1.0e-6, 1.01e-6, 2.0e-6], dtype=torch.float64)
    matrix64 = rotation @ torch.diag(eigenvalues) @ rotation.transpose(-1, -2)
    weight64 = torch.tensor(
        [[0.2, -0.7, 0.3], [-0.7, 1.1, 0.4], [0.3, 0.4, -0.6]],
        dtype=torch.float64,
    )

    oracle_input = matrix64.clone().requires_grad_(True)
    oracle_eigenvalues, oracle_eigenvectors = torch.linalg.eigh(oracle_input)
    oracle_sqrt = (
        oracle_eigenvectors * oracle_eigenvalues.sqrt().unsqueeze(-2)
    ) @ oracle_eigenvectors.transpose(-1, -2)
    oracle_gradient = torch.autograd.grad((oracle_sqrt * weight64).sum(), oracle_input)[0]

    test_input = matrix64.float().requires_grad_(True)
    test_eigenvalues, test_eigenvectors = _eigh_damped(test_input, _rel_gap_eps(test_input))
    test_sqrt = (
        test_eigenvectors * test_eigenvalues.sqrt().unsqueeze(-2)
    ) @ test_eigenvectors.transpose(-1, -2)
    test_gradient = torch.autograd.grad((test_sqrt * weight64.float()).sum(), test_input)[0]

    relative_error = torch.linalg.matrix_norm(test_gradient.double() - oracle_gradient) / torch.linalg.matrix_norm(
        oracle_gradient
    )
    assert relative_error < 2.0e-3


def test_repeated_spectrum_uses_analytic_spectral_derivative_limit() -> None:
    sigma = torch.eye(3, requires_grad=True)
    tangent = torch.zeros(3, 3)
    weight = torch.tensor(
        [[0.2, -0.7, 0.3], [-0.7, 1.1, 0.4], [0.3, 0.4, -0.6]],
        dtype=torch.float32,
    )

    output = retract_spd_full(
        sigma,
        tangent,
        trust_region=0.0,
        sigma_max=None,
    )
    gradient = torch.autograd.grad((output * weight).sum(), sigma)[0]

    torch.testing.assert_close(gradient, weight, atol=2.0e-6, rtol=2.0e-6)


def test_exact_positional_group_product_remains_reachable() -> None:
    cfg = VFE3Config(
        vocab_size=9,
        embed_dim=4,
        n_heads=2,
        max_seq_len=5,
        n_layers=1,
        n_e_steps=1,
        pos_phi="learned",
        pos_phi_compose="group_product",
    )
    model = VFEModel(cfg)
    phi = torch.randn(1, 5, model.group.generators.shape[0])

    left = model._apply_pos_phi(phi)
    right = model._pos_phi_right(phi)

    assert torch.equal(left, phi)
    assert right is not None
    expected = torch.linalg.matrix_exp(embed_phi(phi, model.group.generators)) @ torch.linalg.matrix_exp(
        embed_phi(right, model.group.generators)
    )
    assert expected.shape == (1, 5, cfg.embed_dim, cfg.embed_dim)
