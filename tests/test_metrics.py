import math

import torch

from vfe3.geometry.groups import get_group
from vfe3.geometry.transport import compute_transport_operators
from vfe3.metrics import (
    attention_entropy,
    compute_metrics,
    effective_rank,
    free_energy_terms,
    gauge_trace_spread,
    holonomy_deviation,
)


def test_effective_rank_flat_and_peaked():
    assert torch.allclose(effective_rank(torch.ones(4)), torch.tensor(4.0), atol=1e-5)
    assert torch.allclose(effective_rank(torch.ones(2)), torch.tensor(2.0), atol=1e-5)
    peaked = torch.tensor([1.0, 1e-9, 1e-9, 1e-9])
    assert abs(float(effective_rank(peaked)) - 1.0) < 1e-3        # one dominant mode -> ~1


def test_attention_entropy_uniform_and_onehot():
    N = 5
    uniform = torch.full((3, N, N), 1.0 / N)
    assert abs(float(attention_entropy(uniform)) - math.log(N)) < 1e-4
    onehot = torch.zeros(2, N, N); onehot[..., 0] = 1.0
    assert float(attention_entropy(onehot)) < 1e-5


def test_free_energy_terms_is_registered_metric():
    """free_energy_terms is selectable through the metrics registry (compute_metrics), not only
    as a bare function."""
    N = 4
    self_div = torch.rand(N)
    energy = torch.rand(N, N)
    beta = torch.softmax(-energy, dim=-1)
    alpha = torch.ones(N)
    out = compute_metrics(["free_energy_terms"], self_div=self_div, energy=energy,
                          beta=beta, alpha=alpha, tau=1.0)
    assert "free_energy_terms" in out and "total" in out["free_energy_terms"]


def test_holonomy_deviation_zero_for_flat_cocycle():
    grp = get_group("glk")(3)
    phi = 0.2 * torch.randn(1, 4, grp.generators.shape[0])
    omega = compute_transport_operators(phi, grp)["Omega"][0]      # (4,4,3,3) flat cocycle
    assert float(holonomy_deviation(omega)) < 1e-4                 # every triangle closes (H=I)


def test_holonomy_deviation_positive_for_non_cocycle():
    g = torch.Generator().manual_seed(0)
    N, K = 4, 3
    omega = torch.eye(K).expand(N, N, K, K) + 0.3 * torch.randn(N, N, K, K, generator=g)
    assert float(holonomy_deviation(omega)) > 1e-2                 # random transport does not close


def test_gauge_trace_spread_zero_at_phi_zero():
    grp = get_group("glk")(3)
    G = grp.generators
    assert float(gauge_trace_spread(torch.zeros(5, G.shape[0]), G)) < 1e-7
    assert float(gauge_trace_spread(torch.randn(5, G.shape[0]), G)) > 0.0


def test_free_energy_terms_decomposition():
    N = 3
    self_div = torch.zeros(N)                                      # q == p -> self term 0
    energy = torch.rand(N, N)
    beta = torch.softmax(-energy, dim=-1)
    alpha = torch.ones(N)
    terms = free_energy_terms(self_div, energy, beta, alpha, tau=1.0)
    assert abs(terms["self_coupling"]) < 1e-6
    assert abs(terms["total"] - (terms["self_coupling"] + terms["belief_coupling"] + terms["attention_entropy"])) < 1e-5


def test_compute_metrics_registry_record():
    grp = get_group("glk")(3)
    phi = 0.1 * torch.randn(1, 4, grp.generators.shape[0])
    omega = compute_transport_operators(phi, grp)["Omega"][0]
    rec = compute_metrics(
        ["effective_rank", "attention_entropy", "holonomy_deviation", "gauge_trace_spread"],
        sigma=torch.rand(4, 3) + 0.5,
        beta=torch.softmax(torch.randn(4, 4), dim=-1),
        omega=omega,
        phi=phi[0],
        generators=grp.generators,
    )
    assert set(rec) == {"effective_rank", "attention_entropy", "holonomy_deviation", "gauge_trace_spread"}
    assert all(isinstance(v, float) for v in rec.values())
