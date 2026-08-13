# Task 4 report: SPD and GaugeGate precision policies

## Status

Implemented at base `88880691cd5a704eb12c1f7d2cf68e9b9e1f907a` in implementation commit
`dc51ac4` (`fix: validate covariance and gauge solves`). Scope is limited to M13/M14 numerical
validation, the GaugeGate solve, and their regression coverage.

## RED evidence

CPU-only command environment: `C:/Python314/python.exe`, `CUDA_VISIBLE_DEVICES=-1`, and
`VFE3_TEST_DEVICE=cpu`.

- `.verification/remediation-2026-08-13/task-04-red.xml`: 4 selected, 3 failures, 0 errors,
  0 skipped. The well-conditioned fp32 fast-path test passed; failures were the missing fp64
  congruence retention, missing GaugeGate escalation, and missing fail-visible behavior.
- The dimension-aware condition-bound fixture was separately proved RED: a `K=2`, condition
  `100,000` fp32 system was incorrectly accepted by the initial dimension-free threshold.

## Implementation

- `vfe3/numerics.py` now owns one zero-semantic-jitter `validated_cholesky_solve` abstraction.
  Its default residual tolerance is `64 K eps` and condition limit is `1 / (64 K eps)`, both
  evaluated in the validation dtype. Symmetry and factorization residuals are max-entry errors
  normalized by `max(abs(A))`; the solve certificate uses the normwise backward error
  `max(abs(Ax-b)) / (||A||_inf max(abs(x)) + max(abs(b)))`.
- Full-covariance fast congruences are accepted only when finite, symmetric within the shared
  tolerance, unjittered-Cholesky valid, below the condition limit, and below the factor residual
  bound. Failed fp32 results are recomputed and retained in float64. Uncertified float64 results
  raise `FloatingPointError`.
- GaugeGate replaces generic solves with checked Cholesky solves. Well-conditioned rows remain
  fp32. Any uncertain row triggers float64 recomputation, per-row selection retains certified fast
  rows, and gate/Jacobian arithmetic stays in the promoted dtype. An uncertified high-precision row
  raises `FloatingPointError`.

## GREEN evidence

- `.verification/remediation-2026-08-13/task-04-focused-green.xml`: 41 tests, 0 failures,
  0 errors, 0 skipped.
- `.verification/remediation-2026-08-13/task-04-green.xml`: 62 tests, 0 failures, 0 errors,
  0 skipped across the four requested Task 4 modules plus `tests/test_transport.py`.
- `.verification/remediation-2026-08-13/task-04-policy-regression.xml`: 9 tests, 0 failures,
  0 errors, 0 skipped in the pre-existing congruence precision-policy module.
- Changed Python files passed `C:/Python314/python.exe -m compileall -q`; `git diff --check`
  reported no whitespace errors.

The audited finite-but-indefinite fp32 sandwich now returns the exact direct-float64 congruence as
float64. Its backward gradients are finite and exactly equal to the direct float64 oracle. The
GaugeGate ill-conditioned gauge pair retains float64 through the learned gate and agrees with an
independent direct-float64 `solve`/linear/SILU oracle within literal absolute tolerance `5e-12`.
Direct certificate tests pin fast, escalation, and fail-visible routing. The ordinary
well-conditioned GaugeGate fixture remains fp32 throughout.

## Warnings and concerns

No pytest warnings were emitted by the focused or final Task 4 lanes. The correctness policy adds
an unjittered Cholesky and spectral condition check to every full-covariance congruence accepted by
the guarded routes; this is required by the binding validity contract and is a deliberate cost.
The final evidence is CPU-only as required; no GPU lane was run.

## Self-review

- No Task 2 registration/common-signature changes.
- No Task 3 certificate/provenance changes.
- No decoder, diagnostic, training, or configuration changes.
- Tests use direct float64 mathematical oracles rather than the helper under test.
- The only source files changed are `vfe3/numerics.py`, `vfe3/geometry/transport.py`, and
  `vfe3/model/block_mlp.py`; regression changes are limited to the two Task 4 test modules.
