import pytest
import torch

from vfe3.numerics import (
    check_finite,
    condition_number,
    floor_eigenvalues,
    nan_inf_fraction,
    run_monitors,
    safe_cholesky,
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


def test_condition_number_non_pd_returns_inf():
    # A symmetric matrix with a negative eigenvalue has no condition number: the monitor must surface
    # +inf, not a large positive value from clamping lambda_min up to eps (which would read as a
    # merely ill-conditioned SPD matrix). (audit 2026-06-17 id 39)
    torch.manual_seed(0)
    K = 3
    Q, _ = torch.linalg.qr(torch.randn(K, K))
    M = Q @ torch.diag(torch.tensor([-1.0, 1.0, 3.0])) @ Q.transpose(-1, -2)   # spectrum {-1,1,3}
    cond = condition_number(M)
    assert torch.isinf(cond) and cond > 0


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


def test_safe_cholesky_factors_spd_byte_identical():
    r"""On SPD inputs safe_cholesky's round-0 (zero added jitter) factor is byte-identical
    to torch.linalg.cholesky, and the ok-mask is all-True."""
    g = torch.Generator().manual_seed(5)
    A = torch.randn(3, 4, 4, generator=g)
    M = A @ A.transpose(-1, -2) + torch.eye(4)              # SPD, well-conditioned
    L, ok = safe_cholesky(M)
    assert ok.all()
    assert torch.equal(L, torch.linalg.cholesky(M))         # byte-identical, not approx


def test_safe_cholesky_indefinite_marks_failed_no_raise():
    r"""An indefinite matrix yields ok=False for that element WITHOUT raising."""
    bad = torch.diag(torch.tensor([1.0, -1.0]))             # indefinite
    L, ok = safe_cholesky(bad)                              # must not raise
    assert ok.item() is False


def test_safe_cholesky_mixed_batch_isolates_failure():
    r"""In a mixed batch the failed (indefinite) element is masked while the good
    element keeps the exact factor it would get alone."""
    good = torch.eye(3)
    bad = torch.diag(torch.tensor([1.0, -2.0, 1.0]))
    M = torch.stack([good, bad])
    L, ok = safe_cholesky(M)                               # must not raise
    assert ok[0].item() is True
    assert ok[1].item() is False
    assert torch.equal(L[0], torch.linalg.cholesky(good))  # good element unperturbed


def test_run_monitors_record():
    rec = run_monitors(torch.tensor([1.0, 2.0, float("nan")]))
    assert set(rec) == {"nan_fraction", "abs_max"}
    assert abs(rec["nan_fraction"] - 1.0 / 3.0) < 1e-6
    assert rec["abs_max"] == 2.0
    # matrix probe on request
    M = torch.diag(torch.tensor([1.0, 9.0]))
    rec2 = run_monitors(M, ["condition_number"])
    assert abs(rec2["condition_number"] - 9.0) < 1e-3
