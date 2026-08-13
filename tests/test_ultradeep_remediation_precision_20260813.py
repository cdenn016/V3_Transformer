"""Regression pins for validity-aware SPD and GaugeGate precision escalation."""

from __future__ import annotations

import pytest
import torch

from vfe3.geometry.transport import (
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
