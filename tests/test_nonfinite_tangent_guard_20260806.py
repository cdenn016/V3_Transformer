r"""Non-finite tangent guard on the SPD retractions (audit 2026-08-06).

An ``e_q_mu_lr=0.01`` sweep cell died at step 4253 of 10000 with
``_LinAlgError: linalg.eigh: (Batch element 2607): The algorithm failed to converge``. The tangent
carried a non-finite entry and the Frobenius trust region -- the only guard on that path -- does not
stop one: ``||R||_F`` of a matrix holding a NaN is NaN, so ``clamp(trust/(NaN + eps), max=1.0)`` is
NaN and the clamp multiplies the poison through. The full arm then handed it to ``eigh`` and raised;
the diagonal arm propagated it silently to a NaN covariance.

Pins: (a) the trust region really does bound a large FINITE tangent, so the guard is not papering
over a missing norm bound; (b) it provably does NOT bound a non-finite one; (c) both arms now
survive, leaving the poisoned element's covariance unchanged (``exp(0) = I``) while its neighbours
retract normally; (d) the neutralization is COUNTED, so a masked non-finite gradient cannot pass as
a healthy run; (e) an all-finite tangent is bitwise unchanged, forward and backward.
"""

import pytest
import torch

import vfe3.geometry.retraction as R
from vfe3.geometry.retraction import (
    nonfinite_tangent_elements,
    reset_nonfinite_tangent_elements,
    retract_logeuclidean_full,
    retract_spd_diagonal,
    retract_spd_full,
)

K, Bn = 20, 4
KW = dict(step_size=1.0, trust_region=5.0, eps=1e-6, sigma_max=100.0)


def _sigma(seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(Bn, K, K, generator=g)
    return (a @ a.transpose(-1, -2) + K * torch.eye(K)) * 0.1


def _sigma_diag(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(Bn, K, generator=g) + 0.5


@pytest.fixture(autouse=True)
def _reset():
    reset_nonfinite_tangent_elements()
    yield
    reset_nonfinite_tangent_elements()


# -- (a)/(b) what the trust region does and does not bound --------------------------------------


def test_trust_region_bounds_a_large_finite_tangent():
    delta = torch.zeros(Bn, K, K)
    delta[1] = 1e30 * torch.eye(K)
    out = retract_spd_full(_sigma(), delta, **KW)
    assert torch.isfinite(out).all()
    assert nonfinite_tangent_elements() == 0, "a finite tangent must not trip the guard"


def test_trust_region_does_not_bound_a_nonfinite_tangent():
    r"""The premise of the guard: the clamp factor on a NaN-bearing tangent is itself NaN."""
    poisoned = torch.full((K, K), float("nan"))
    norm = torch.linalg.norm(poisoned, ord="fro")
    assert torch.isnan(torch.clamp(5.0 / (norm + 1e-6), max=1.0))


# -- (c)/(d) both arms survive, conservatively, and say so --------------------------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_full_arm_survives_and_freezes_only_the_poisoned_element(bad):
    sigma = _sigma()
    delta = torch.zeros(Bn, K, K)
    delta[2, 0, 0] = bad

    out = retract_spd_full(sigma, delta, **KW)

    assert torch.isfinite(out).all()
    # exp(0) = I  =>  Sigma^{1/2} I Sigma^{1/2} = Sigma, that element is simply not stepped.
    assert torch.allclose(out[2], sigma[2], atol=1e-5)
    assert nonfinite_tangent_elements() == 1


def test_full_arm_neighbours_still_retract():
    r"""Guards against over-correcting: the whole batch must not be frozen by one bad element."""
    sigma = _sigma()
    g = torch.Generator().manual_seed(7)
    delta = 0.2 * torch.randn(Bn, K, K, generator=g)
    delta = 0.5 * (delta + delta.transpose(-1, -2))
    delta[2] = float("nan")

    out = retract_spd_full(sigma, delta, **KW)
    assert torch.allclose(out[2], sigma[2], atol=1e-5)
    for i in (0, 1, 3):
        assert not torch.allclose(out[i], sigma[i], atol=1e-4)


def test_diagonal_arm_no_longer_propagates_silent_nan():
    sigma = _sigma_diag()
    delta = torch.zeros(Bn, K)
    delta[3, 2] = float("nan")

    out = retract_spd_diagonal(sigma, delta, **KW)

    assert torch.isfinite(out).all()
    assert torch.allclose(out[3], sigma[3], atol=1e-6)
    assert nonfinite_tangent_elements() == 1


def test_counter_accumulates_and_resets():
    delta = torch.zeros(Bn, K, K)
    delta[0, 1, 1] = float("nan")
    delta[3] = float("inf")
    retract_spd_full(_sigma(), delta, **KW)
    assert nonfinite_tangent_elements() == 2
    retract_spd_full(_sigma(), delta, **KW)
    assert nonfinite_tangent_elements() == 4, "counts accumulate across calls"
    reset_nonfinite_tangent_elements()
    assert nonfinite_tangent_elements() == 0


def test_logeuclidean_nonfinite_tangent_freezes_only_its_row_and_counts_once():
    sigma = _sigma()
    g = torch.Generator().manual_seed(11)
    finite_delta = 0.1 * torch.randn(Bn, K, K, generator=g)
    finite_delta = 0.5 * (finite_delta + finite_delta.transpose(-1, -2))
    finite_delta[2].zero_()
    poisoned = finite_delta.clone()
    poisoned[2, 0, 0] = float("nan")
    poisoned[2, 1, 1] = float("inf")
    poisoned[2, 2, 2] = float("-inf")

    expected = retract_logeuclidean_full(sigma, finite_delta, **KW)
    out = retract_logeuclidean_full(sigma, poisoned, **KW)

    assert torch.allclose(out[2], sigma[2], atol=1e-5)
    assert torch.equal(out[[0, 1, 3]], expected[[0, 1, 3]])
    assert nonfinite_tangent_elements() == 1


@pytest.mark.parametrize(
    "bad",
    [float("nan"), float("inf"), float("-inf"), torch.finfo(torch.float32).max],
    ids=["nan", "posinf", "neginf", "finite_chart_overflow"],
)
def test_logeuclidean_frozen_rows_have_identity_sigma_gradient_and_zero_delta_gradient(bad):
    sigma_value = _sigma()
    # This makes the finite largest-fp32 tangent overflow only after the log-chart derivative
    # multiplies it by Sigma^{-1}; the three literal nonfinites exercise the raw-input path.
    sigma_value[2] = 0.5 * torch.eye(K)
    finite_delta = torch.zeros(Bn, K, K)
    finite_delta[0, 0, 0] = 0.1
    finite_delta[1, 1, 1] = -0.05
    finite_delta[3, 2, 2] = 0.08
    poisoned_delta = finite_delta.clone()
    poisoned_delta[2, 0, 0] = bad
    cotangent = torch.eye(K).expand(Bn, K, K).clone()

    reference_sigma = sigma_value.clone().requires_grad_(True)
    reference_delta = finite_delta.clone().requires_grad_(True)
    reference = retract_logeuclidean_full(reference_sigma, reference_delta, **KW)
    reference_sigma_grad, reference_delta_grad = torch.autograd.grad(
        (reference * cotangent).sum(),
        (reference_sigma, reference_delta),
    )

    sigma = sigma_value.clone().requires_grad_(True)
    delta = poisoned_delta.requires_grad_(True)
    out = retract_logeuclidean_full(sigma, delta, **KW)
    sigma_grad, delta_grad = torch.autograd.grad((out * cotangent).sum(), (sigma, delta))

    assert torch.equal(out[[0, 1, 3]], reference[[0, 1, 3]])
    assert torch.equal(sigma_grad[[0, 1, 3]], reference_sigma_grad[[0, 1, 3]])
    assert torch.equal(delta_grad[[0, 1, 3]], reference_delta_grad[[0, 1, 3]])
    assert torch.allclose(out[2], sigma[2], atol=1e-5)
    assert torch.isfinite(sigma_grad[2]).all()
    assert torch.equal(sigma_grad[2], cotangent[2])
    assert torch.equal(delta_grad[2], torch.zeros_like(delta_grad[2]))
    assert nonfinite_tangent_elements() == 1


# -- (e) the finite path is untouched -----------------------------------------------------------


def test_finite_tangent_is_bitwise_unchanged(monkeypatch):
    sigma, sigma_d = _sigma(3), _sigma_diag(3)
    g = torch.Generator().manual_seed(4)
    delta = 0.05 * torch.randn(Bn, K, K, generator=g)
    delta = 0.5 * (delta + delta.transpose(-1, -2))
    delta_d = 0.05 * torch.randn(Bn, K, generator=g)

    guarded_full = retract_spd_full(sigma, delta, **KW)
    guarded_diag = retract_spd_diagonal(sigma_d, delta_d, **KW)

    monkeypatch.setattr(R, "_neutralize_nonfinite_tangent", lambda t, event_ndim: t)
    assert torch.equal(retract_spd_full(sigma, delta, **KW), guarded_full)
    assert torch.equal(retract_spd_diagonal(sigma_d, delta_d, **KW), guarded_diag)


def test_finite_backward_is_bitwise_unchanged(monkeypatch):
    sigma = _sigma(3).requires_grad_(True)
    g = torch.Generator().manual_seed(4)
    delta = 0.05 * torch.randn(Bn, K, K, generator=g)
    delta = 0.5 * (delta + delta.transpose(-1, -2))

    guarded = torch.autograd.grad(retract_spd_full(sigma, delta, **KW).sum(), sigma)[0]
    monkeypatch.setattr(R, "_neutralize_nonfinite_tangent", lambda t, event_ndim: t)
    bare = torch.autograd.grad(retract_spd_full(sigma, delta, **KW).sum(), sigma)[0]
    assert torch.equal(guarded, bare)
