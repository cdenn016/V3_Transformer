"""Regression pins for validity-aware SPD and GaugeGate precision escalation."""

from __future__ import annotations

import pytest
import torch

from vfe3.geometry.transport import (
    CompactFactoredTransport,
    DirectLinkTransport,
    FactoredTransport,
    RopeTransport,
    full_cov_congruence_precision,
    set_full_cov_congruence_precision,
    transport_covariance,
)
from vfe3.model.block_mlp import GaugeGateBlockMLP


_FINITE_INDEFINITE_OMEGA = torch.tensor(
    [
        [0.057695649564266205, 0.016874510794878006,
         0.20264112949371338, 0.07412116974592209],
        [-0.49864041805267334, -0.11252593994140625,
         -0.2768476605415344, 0.1497870236635208],
        [-0.4640336036682129, -0.11606039106845856,
         -0.08934598416090012, 0.28132152557373047],
        [0.4184989333152771, 0.15523938834667206,
         0.12013706564903259, -0.3998108506202698],
    ],
    dtype=torch.float32,
).reshape(1, 1, 4, 4)


@pytest.fixture(autouse=True)
def _restore_congruence_policy():
    previous = full_cov_congruence_precision()
    yield
    set_full_cov_congruence_precision(previous)


def test_fp32_escalate_rejects_finite_indefinite_congruence_and_keeps_fp64_backward():
    """A finite fp32 sandwich outside SPD must not be accepted or cast back after recovery."""
    set_full_cov_congruence_precision("fp32_escalate")
    omega = _FINITE_INDEFINITE_OMEGA.clone().requires_grad_(True)
    sigma = torch.eye(4, dtype=torch.float32).reshape(1, 4, 4).requires_grad_(True)

    raw_fast = torch.einsum(
        "...ijkl,...jlm,...ijnm->...ijkn", omega, sigma, omega)
    _, raw_info = torch.linalg.cholesky_ex(
        0.5 * (raw_fast + raw_fast.transpose(-1, -2)))
    assert torch.isfinite(raw_fast).all()
    assert bool((raw_info != 0).all()), "fixture no longer isolates finite-but-non-SPD fp32"

    result = transport_covariance(omega, sigma, diagonal_out=False)
    result_grads = torch.autograd.grad(result.square().sum(), (omega, sigma))

    omega_reference = _FINITE_INDEFINITE_OMEGA.clone().requires_grad_(True)
    sigma_reference = torch.eye(4, dtype=torch.float32).reshape(1, 4, 4).requires_grad_(True)
    reference = torch.einsum(
        "...ijkl,...jlm,...ijnm->...ijkn",
        omega_reference.double(), sigma_reference.double(), omega_reference.double(),
    )
    reference_grads = torch.autograd.grad(
        reference.square().sum(), (omega_reference, sigma_reference))

    assert result.dtype is torch.float64
    assert torch.equal(result, reference)
    assert bool((torch.linalg.cholesky_ex(result)[1] == 0).all())
    for actual, expected in zip(result_grads, reference_grads):
        assert torch.isfinite(actual).all()
        assert torch.equal(actual, expected)


_STRUCTURAL_ROUTES = (
    "dense",
    "direct",
    "compact",
    "factored_equal",
    "factored_heterogeneous",
    "rope_dense_means_only",
    "rope_dense_full",
    "rope_direct_full",
    "rope_compact_full",
    "rope_factored_full",
)


def _structural_transport(route: str, block: torch.Tensor) -> 'tuple[object, torch.Tensor]':
    if route in ("factored_equal", "rope_factored_full"):
        operator = torch.block_diag(block, block)
        dims = [4, 4]
    elif route == "factored_heterogeneous":
        operator = torch.block_diag(torch.ones(1, 1, dtype=block.dtype), block)
        dims = [1, 4]
    else:
        operator = block
        dims = [operator.shape[-1]]
    K = operator.shape[-1]
    identity = torch.eye(K, dtype=operator.dtype).reshape(1, K, K)
    dense = operator.reshape(1, 1, K, K)

    if route == "dense":
        return dense, dense
    if route == "direct":
        return DirectLinkTransport(dense), dense
    if route == "compact":
        return CompactFactoredTransport(
            operator.reshape(1, 1, K, K),
            identity.reshape(1, 1, K, K),
            K,
        ), dense
    if route in ("factored_equal", "factored_heterogeneous"):
        return FactoredTransport(operator.reshape(1, K, K), identity, dims), dense

    rope = torch.eye(K, dtype=operator.dtype).reshape(1, K, K)
    if route in ("rope_dense_means_only", "rope_dense_full"):
        return RopeTransport(
            dense, rope, on_cov=(route == "rope_dense_full"), insertion="left",
        ), dense
    if route == "rope_direct_full":
        return RopeTransport(
            DirectLinkTransport(dense), rope, on_cov=True, insertion="left",
        ), dense
    if route == "rope_compact_full":
        compact = CompactFactoredTransport(
            operator.reshape(1, 1, K, K),
            identity.reshape(1, 1, K, K),
            K,
        )
        return RopeTransport(compact, rope, on_cov=True, insertion="left"), dense
    if route == "rope_factored_full":
        factored = FactoredTransport(operator.reshape(1, K, K), identity, dims)
        return RopeTransport(factored, rope, on_cov=True, insertion="left"), dense
    raise AssertionError(f"unhandled structural route {route!r}")


@pytest.mark.parametrize("policy", ("fp64", "fp32_escalate"))
@pytest.mark.parametrize("route", _STRUCTURAL_ROUTES)
def test_every_public_structural_route_retains_certified_float64_when_fp32_is_indefinite(
    route: str, policy: str,
):
    set_full_cov_congruence_precision(policy)
    container, dense_operator = _structural_transport(route, _FINITE_INDEFINITE_OMEGA[0, 0])
    K = dense_operator.shape[-1]
    sigma = torch.eye(K, dtype=torch.float32).reshape(1, K, K)
    raw_fast = torch.einsum(
        "...ijkl,...jlm,...ijnm->...ijkn", dense_operator, sigma, dense_operator)
    assert torch.isfinite(raw_fast).all()
    assert bool((torch.linalg.cholesky_ex(raw_fast)[1] != 0).all())

    actual = transport_covariance(container, sigma, diagonal_out=False)
    reference = torch.einsum(
        "...ijkl,...jlm,...ijnm->...ijkn",
        dense_operator.double(), sigma.double(), dense_operator.double(),
    )

    assert actual.dtype is torch.float64
    assert bool((torch.linalg.cholesky_ex(actual)[1] == 0).all())
    torch.testing.assert_close(actual, reference, rtol=2.0e-14, atol=2.0e-15)


@pytest.mark.parametrize("policy", ("fp64", "fp32_escalate"))
@pytest.mark.parametrize("route", _STRUCTURAL_ROUTES)
def test_every_public_structural_route_returns_benign_certified_fp32_cast(
    route: str, policy: str,
):
    set_full_cov_congruence_precision(policy)
    identity_block = torch.eye(4, dtype=torch.float32)
    container, dense_operator = _structural_transport(route, identity_block)
    K = dense_operator.shape[-1]
    sigma = torch.eye(K, dtype=torch.float32).reshape(1, K, K)

    actual = transport_covariance(container, sigma, diagonal_out=False)

    assert actual.dtype is torch.float32
    assert torch.equal(actual, torch.eye(K).reshape(1, 1, K, K))


def test_public_full_covariance_route_fails_when_float64_is_uncertifiable():
    set_full_cov_congruence_precision("fp32_escalate")
    singular = torch.zeros(1, 1, 3, 3, dtype=torch.float32)
    sigma = torch.eye(3, dtype=torch.float32).reshape(1, 3, 3)

    with pytest.raises(FloatingPointError, match="float64 recomputation could not be certified"):
        transport_covariance(singular, sigma, diagonal_out=False)


def _fixed_gauge_gate() -> GaugeGateBlockMLP:
    gate = GaugeGateBlockMLP(
        irrep_dims=[2], expansion=1, activation="silu", dropout=0.0,
        covariance_contract="passthrough",
    ).float().eval()
    with torch.no_grad():
        gate.fc1.weight.fill_(1.0e-4)
        gate.fc1.bias.fill_(0.1)
        gate.fc2.weight.fill_(0.2)
        gate.fc2.bias.fill_(0.03)
    return gate


def test_gauge_gate_escalates_ill_conditioned_gauge_pair_through_learned_gate():
    """The checked invariant and learned gate match their direct float64 semantics."""
    gate = _fixed_gauge_gate()
    gauge = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float32)
    mu = torch.tensor([0.75, -0.25], dtype=torch.float32)
    sigma = torch.diag(torch.tensor([1.0, 2.0 ** -18], dtype=torch.float32))
    transformed_mu = gauge @ mu
    transformed_sigma = gauge @ sigma @ gauge.transpose(-1, -2)

    eigmins = torch.stack((
        torch.linalg.eigvalsh(sigma.double())[0],
        torch.linalg.eigvalsh(transformed_sigma.double())[0],
    ))
    assert bool((eigmins > 0.0).all()), "fixture must remain strictly SPD after storage"

    base_invariant, _, _ = gate._invariants(mu, sigma)
    transformed_invariant, _, _ = gate._invariants(transformed_mu, transformed_sigma)
    base_gate, _ = gate._gate_values(base_invariant, need_jacobian=False)
    transformed_gate, _ = gate._gate_values(transformed_invariant, need_jacobian=False)

    def direct_reference(mean: torch.Tensor, covariance: torch.Tensor) -> torch.Tensor:
        mean64 = mean.double()
        covariance64 = covariance.double()
        solved64 = torch.linalg.solve(covariance64, mean64.unsqueeze(-1)).squeeze(-1)
        invariant64 = (mean64 * solved64).sum().reshape(1)
        hidden64 = torch.nn.functional.linear(
            invariant64, gate.fc1.weight.double(), gate.fc1.bias.double())
        return torch.nn.functional.linear(
            torch.nn.functional.silu(hidden64),
            gate.fc2.weight.double(), gate.fc2.bias.double(),
        )

    base_reference_gate = direct_reference(mu, sigma)
    transformed_reference_gate = direct_reference(transformed_mu, transformed_sigma)

    assert base_invariant.dtype is torch.float64
    assert transformed_invariant.dtype is torch.float64
    assert base_gate.dtype is torch.float64
    assert transformed_gate.dtype is torch.float64
    assert torch.allclose(base_gate, base_reference_gate, atol=5.0e-12, rtol=0.0)
    assert torch.allclose(
        transformed_gate, transformed_reference_gate, atol=5.0e-12, rtol=0.0)
    assert torch.equal(base_reference_gate, transformed_reference_gate)
    assert torch.allclose(base_gate, transformed_gate, atol=5.0e-12, rtol=0.0)

    base_output = gate.mean_update(mu, sigma)
    transformed_output = gate.mean_update(transformed_mu, transformed_sigma)
    expected_transformed = gauge.double() @ base_output
    assert torch.allclose(
        transformed_output, expected_transformed, atol=5.0e-12, rtol=0.0)


def test_gauge_gate_well_conditioned_fp32_stays_on_fast_path():
    gate = _fixed_gauge_gate()
    mu = torch.tensor([0.25, -0.5], dtype=torch.float32)
    sigma = torch.tensor([[2.0, 0.125], [0.125, 1.0]], dtype=torch.float32)

    invariants, _, solutions = gate._invariants(mu, sigma)
    gates, _ = gate._gate_values(invariants, need_jacobian=False)
    output = gate.mean_update(mu, sigma)

    assert invariants.dtype is torch.float32
    assert all(solution.dtype is torch.float32 for solution in solutions)
    assert gates.dtype is torch.float32
    assert output.dtype is torch.float32


def test_gauge_gate_fails_visibly_when_float64_cannot_certify_spd():
    gate = _fixed_gauge_gate()
    mu = torch.tensor([0.25, -0.5], dtype=torch.float32)
    indefinite = torch.diag(torch.tensor([1.0, -0.25], dtype=torch.float32))

    with pytest.raises(FloatingPointError, match="certif"):
        gate.mean_update(mu, indefinite)


def test_gauge_gate_mixed_fast_and_escalated_rows_preserve_row_semantics_and_backward():
    gate = GaugeGateBlockMLP(
        irrep_dims=[2], expansion=1, activation="silu", dropout=0.0,
        covariance_contract="delta_full",
    ).float().eval()
    with torch.no_grad():
        gate.fc1.weight.fill_(1.0e-4)
        gate.fc1.bias.fill_(0.1)
        gate.fc2.weight.fill_(0.2)
        gate.fc2.bias.fill_(0.03)
    mu = torch.tensor([[0.25, -0.5], [0.75, -0.25]], dtype=torch.float32,
                      requires_grad=True)
    sigma = torch.stack((
        torch.tensor([[2.0, 0.125], [0.125, 1.0]], dtype=torch.float32),
        torch.diag(torch.tensor([1.0, 2.0 ** -18], dtype=torch.float32)),
    )).requires_grad_(True)

    invariants, _, solutions = gate._invariants(mu, sigma)
    result = gate.forward_moments(mu, sigma)

    fast_factor = torch.linalg.cholesky(sigma[0].detach())
    expected_fast = torch.cholesky_solve(
        mu[0].detach().unsqueeze(-1), fast_factor).squeeze(-1).double()
    high_factor = torch.linalg.cholesky(sigma[1].detach().double())
    expected_high = torch.cholesky_solve(
        mu[1].detach().double().unsqueeze(-1), high_factor).squeeze(-1)

    assert invariants.dtype is torch.float64
    assert solutions[0].dtype is torch.float64
    assert torch.equal(solutions[0][0], expected_fast)
    assert torch.equal(solutions[0][1], expected_high)
    assert result.mu.dtype is torch.float64
    assert result.sigma.dtype is torch.float64
    assert result.jacobian is not None and result.jacobian.dtype is torch.float64

    loss = result.mu.square().sum() + result.sigma.square().sum()
    gradients = torch.autograd.grad(loss, (mu, sigma, *gate.parameters()))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
