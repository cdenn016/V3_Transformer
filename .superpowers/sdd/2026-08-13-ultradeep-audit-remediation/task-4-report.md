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

## Review remediation

Review findings C1, I1, and M1 were remediated in implementation commit
`b4145eafeea5757c95d818e3a5a93a7a8bebed3a` (`fix: certify all covariance transport routes`).

### Review RED evidence

CPU-only command environment remained `C:/Python314/python.exe`, `CUDA_VISIBLE_DEVICES=-1`, and
`VFE3_TEST_DEVICE=cpu`.

- `.verification/remediation-2026-08-13/task-04-review-red.xml`: 29 tests, 10 failures, 0 errors,
  0 skipped. The failures isolated the unchecked DirectLinkTransport, equal-block FactoredTransport,
  heterogeneous FactoredTransport, and RoPE-over-direct full-covariance routes, plus all six NaN or
  infinite custom-bound cases. Dense, compact, other wrapper controls, mixed GaugeGate rows, and the
  float64 fail-visible branch passed before the production repair.

### Review implementation

- `_certify_full_congruence` is now the single authoritative post-congruence acceptance and dtype
  boundary for dense, DirectLink, compact head-block, compact full, equal-block factored, and
  heterogeneous factored routes. RoPE wrappers reach the same boundary through their base route.
- Direct and factored contractions now use the selected fast working dtype first. They no longer
  cast unchecked structural results back to the source dtype. A certified benign fp32 result stays
  fp32; failed fp32 recomputes and remains float64; failed float64 raises `FloatingPointError`.
- Public container output shapes and existing wrapper semantics are preserved. Parameterized tests
  cover ten dense/container/wrapper routes for both benign fast and finite-but-indefinite inputs.
- `validated_cholesky_solve` now rejects NaN, positive infinity, and negative infinity for both
  custom bounds before applying the existing range checks. A non-identity SPD matrix pins a real,
  positive normalized factor residual violation at zero tolerance.
- The stale `fp32_escalate` comment now documents the finite, symmetric, zero-jitter Cholesky,
  residual, and conditioning certificate rather than finiteness-only escalation.

### Review GREEN evidence

- `.verification/remediation-2026-08-13/task-04-review-focused-green.xml`: 29 tests, 0 failures,
  0 errors, 0 skipped.
- `.verification/remediation-2026-08-13/task-04-review-green.xml`: 186 tests, 0 failures, 0 errors,
  0 skipped across the requested Task 4 modules and dense/direct/compact/factored/RoPE structural
  wrapper regressions.
- `C:/Python314/python.exe -m compileall -q` passed for all four changed Python files, and
  `git diff --check` reported no whitespace errors.

The full structural lane emitted 16 pre-existing configuration/oracle warnings from
`test_exact_congruence_family.py`, `test_p1_compact_phi_block_transport_20260711.py`, and
`test_audit_transport_registry_20260720.py`; none concern numerical certification or this diff.
No GPU lane was run, as required.

## Post-cast certification remediation

The remaining Critical review finding was remediated in implementation commit
`f0da488e8a6de450af4e5e0c0546599b9bc03cd7` (`fix: certify covariance casts before return`).

### Post-cast RED evidence

- `.verification/remediation-2026-08-13/task-04-post-cast-red.xml`: 40 selected tests,
  10 failures, 0 errors, 0 skipped. Every failure was the `fp64` policy returning an uncertified
  fp32 cast for one of the ten dense/DirectLink/compact/equal-factored/heterogeneous-factored/RoPE
  structural routes. All `fp32_escalate` retained-float64 cases and all 20 benign controls passed.

### Post-cast implementation

- `_certify_full_congruence` now treats the source-dtype cast as a separate candidate requiring the
  same finite, symmetry, zero-jitter Cholesky, normalized-residual, and conditioning certificate.
- A certified cast is returned at source dtype. If the cast fails but the working float64 result is
  certified, that exact float64 tensor is retained, preserving its direct high-precision autograd
  graph. The existing uncertified-float64 exception remains fail-visible.
- The audited finite-but-indefinite fixture now covers both `fp64` and `fp32_escalate` policies for
  every public structural route. A separate identity matrix covers benign certified fp32 casts for
  both policies.
- The older full-Gaussian regression was aligned with the binding policy: it independently proves
  that the fp32 cast is indefinite, then requires the public result to equal the certified direct
  float64 oracle.

### Post-cast GREEN evidence

- `.verification/remediation-2026-08-13/task-04-post-cast-focused-green.xml`: 40 tests,
  0 failures, 0 errors, 0 skipped.
- `.verification/remediation-2026-08-13/task-04-post-cast-green.xml`: 206 tests, 0 failures,
  0 errors, 0 skipped across the prior 186-test Task 4 structural/full suite plus 20 new
  policy-route cases.
- Changed files passed `C:/Python314/python.exe -m compileall -q`, and `git diff --check` reported
  no whitespace errors.

The full lane emitted the same 16 pre-existing configuration/oracle warnings documented above.
No GPU lane was run. The final diff does not change registrations, signatures, provenance,
certificates, configuration, decoder, diagnostics, or training behavior.

## Reduced-dtype cast remediation

The final Important review finding was remediated in implementation commit
`b42376b773643252369442257d13df766c205ba4` (`fix: retain certified reduced-dtype casts`).

- `.verification/remediation-2026-08-13/task-04-reduced-cast-red.xml`: 4 tests, 4 failures,
  0 errors, 0 skipped. Both bf16 and fp16 cast candidates were incorrectly passed to
  `validated_cholesky_solve` under both `fp64` and `fp32_escalate`.
- The authoritative boundary now treats source dtypes outside float32/float64 as uncertifiable and
  retains the already certified working tensor without calling or weakening the shared validator.
  The returned working dtype is float64 under `fp64` and float32 under `fp32_escalate`.
- `.verification/remediation-2026-08-13/task-04-reduced-cast-focused-green.xml`: 4 tests,
  0 failures, 0 errors, 0 skipped. CPU execution supports both bf16 and fp16 fixtures.
- `.verification/remediation-2026-08-13/task-04-reduced-cast-green.xml`: 210 tests, 0 failures,
  0 errors, 0 skipped across the prior 206-test Task 4 structural/full lane plus the four new
  reduced-dtype policy cases.

The reduced-dtype change is confined to cast-candidate routing and its regression coverage. The
same 16 unrelated warnings remain; no GPU lane was run.
