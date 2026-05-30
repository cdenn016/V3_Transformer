import pytest
import torch

from vfe3.numerics import (
    check_finite,
    condition_number,
    floor_eigenvalues,
    nan_inf_fraction,
    run_monitors,
    safe_spd_inverse,
)


def test_safe_spd_inverse_matches_inv_on_well_conditioned():
    g = torch.Generator().manual_seed(0)
    A = torch.randn(3, 4, 4, generator=g)
    M = A @ A.transpose(-1, -2) + torch.eye(4)              # SPD, well-conditioned
    out = safe_spd_inverse(M)
    assert torch.allclose(out, torch.linalg.inv(M), atol=1e-3)


def test_safe_spd_inverse_is_finite_on_singular():
    M = torch.zeros(4, 4)                                   # singular; pure Cholesky fails
    out = safe_spd_inverse(M)
    assert torch.isfinite(out).all()                       # jitter/pinv fallback keeps it finite


def test_floor_eigenvalues_clamps_spectrum():
    M = torch.diag(torch.tensor([5.0, 1e-9, -0.3]))        # one tiny, one negative
    out = floor_eigenvalues(M, floor=1e-3)
    evals = torch.linalg.eigvalsh(out)
    assert (evals >= 1e-3 - 1e-6).all()


def test_condition_number_known_values():
    assert torch.allclose(condition_number(torch.eye(4)), torch.tensor(1.0), atol=1e-5)
    M = torch.diag(torch.tensor([1.0, 100.0]))
    assert torch.allclose(condition_number(M), torch.tensor(100.0), atol=1e-3)


def test_nan_inf_fraction_counts_nonfinite():
    t = torch.tensor([1.0, float("nan"), float("inf"), 2.0])
    assert abs(nan_inf_fraction(t) - 0.5) < 1e-6
    assert nan_inf_fraction(torch.ones(10)) == 0.0


def test_check_finite_warns_and_can_raise():
    bad = torch.tensor([1.0, float("nan")])
    assert check_finite(torch.ones(3)) is True
    with pytest.warns(RuntimeWarning):
        assert check_finite(bad) is False
    with pytest.raises(FloatingPointError):
        check_finite(bad, raise_on_nonfinite=True)


def test_run_monitors_record():
    rec = run_monitors(torch.tensor([1.0, 2.0, float("nan")]))
    assert set(rec) == {"nan_fraction", "abs_max"}
    assert abs(rec["nan_fraction"] - 1.0 / 3.0) < 1e-6
    assert rec["abs_max"] == 2.0
    # matrix probe on request
    M = torch.diag(torch.tensor([1.0, 9.0]))
    rec2 = run_monitors(M, ["condition_number"])
    assert abs(rec2["condition_number"] - 9.0) < 1e-3
