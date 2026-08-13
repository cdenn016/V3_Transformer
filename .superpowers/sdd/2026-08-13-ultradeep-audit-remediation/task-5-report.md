# Task 5 report: remaining mathematical domains and decoder parity

## Status and revision

Implemented from remediation program base
`59a6ca255a067a71ec58e536e03f44d32f8134d7` in Task 5 implementation commit
`3a7ba7a831bf3de240ae8016fd1651f2b360a660` (`fix: harden divergence and decoder domains`).
Work was performed only in `C:\tmp\V3_Transformer_ultradeep_remediation_20260813`; the live
checkout was not touched. All test execution was CPU-only with `C:/Python314/python.exe`,
`CUDA_VISIBLE_DEVICES=-1`, and `VFE3_TEST_DEVICE=cpu`.

## RED evidence

Command:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:VFE3_TEST_DEVICE='cpu'
C:/Python314/python.exe -m pytest `
  tests/test_ultradeep_remediation_domains_20260813.py -q `
  --junitxml=.verification/remediation-2026-08-13/task-05-red.xml
```

Machine-readable result: 12 selected, 12 failures, 0 errors, 0 skipped. Each failure was the
audited behavior: missing covariant floor and floor validation, non-atomic reflection loading,
unscoped alpha-greater-than-one acceptance, NaN all-invalid reduction/no exclusion count,
successful-but-inaccurate fp32 KL, near-tie rank reversal, Laplace downcast, and non-identical
last-position decode.

The literal fixtures include four separate invalid-floor cases (`0`, negative, NaN, infinity), so
the nine conceptual regressions collect as twelve pytest cases.

## Implementation and mathematical bounds

### CG covariance

`CGCoupling.forward_moments` now implements exactly

`Sigma_out = sym(J Sigma J^T) + cg_covariance_floor * Sigma`.

The floor is a finite, strictly positive scalar, defaulting to `1e-6`, validated both by
`VFE3Config` and direct `CGCoupling` construction. It is covariance-relative rather than
coordinate-identity noise, so it is gauge-congruent and preserves every input covariance
direction when `J` is singular. At zero path weights the delta-full result is
`(1 + floor) Sigma`; passthrough remains exactly unchanged.

### Reflection checkpoint domain

Every `reflection_sign` tensor is checked for exact shape, dtype, layout, finiteness, and membership
in `{-1,+1}` before either direct `PriorBank.load_state_dict` or root `VFEModel.load_state_dict`
can invoke PyTorch mutation. The strict-load fixture also changes an earlier mean-table key and
proves the complete module state remains byte-equal after the invalid reflection raises.

### Renyi decoder domain and excluded rows

Config rejection is restricted to the active family-consistent prior-bank decoder when
`divergence_family='renyi'` and `renyi_order > 1`. The linear/no-prior path continues to accept the
same order; no global divergence restriction was added.

The family-chunked reducer no longer forms an all-invalid `logsumexp` or subtracts two nonfinite
quantities. It records whether each row has any finite vocabulary score and whether its target is
finite, zeros excluded rows before CE subtraction, and reduces over the final validity mask.
`DecodeCEResult` now carries device-side int64 `scored_tokens` and `excluded_tokens`; production
decoder/model paths populate both explicitly. `excluded_tokens` is the total row count minus the
final valid count, including ignore-index, degenerate covariance, and runtime all-invalid rows.
The literal two-row all-invalid fixture returns grad-connected CE `0`, scored `0`, excluded `2`,
and finite mean/covariance gradients. The field has an optional default only to preserve existing
third-party/custom test constructors; built-in return paths never omit it.

### Full-KL and expanded-decoder precision

The fp32 full-Gaussian KL policies reuse Task 4's `validated_cholesky_solve` rather than defining a
parallel validator. A row is accepted only when its zero-jitter symmetry and factor residuals are
at most `64 K eps` and its spectral condition number is at most `1 / (64 K eps)`. The
`fp32_escalate_cond` policy additionally retains its stricter legacy pivot-ratio trigger. Any
uncertified row promotes the complete comparison grid to direct float64 arithmetic and the public
result remains float64. In the literal `K=2`, condition-`2e7` case, the limit is `65,536`; the
returned value is bit-equal to the direct float64 oracle (approximately `100000.1975`).

For expanded diagonal decoding, let `n` be the promoted matmul width. The fp32 absolute score
uncertainty is conservatively bounded by

`(2 n + 4) eps * (abs(lhs) @ abs(rhs)^T + abs(bias))`.

If the gap between the two smallest energies is no larger than the sum of their bounds, the whole
tile is recomputed in float64 and retained. This prevents a mixed-precision vocabulary ranking.
The audited near-tie fixture reverses under plain fp32 and now matches the promoted token-0 oracle.

### Laplace dtype and decode parity

`DiagonalLaplace` records the promoted public dtype of its location and scale. Entropy, divergence,
and natural-gradient APIs compute at float64 whenever any public/input operand is float64 and do
not silently cast the result back to the lower-precision operand.

`decode_last` now calls the same full belief/context decoder and then takes `[:, -1:]`. There is no
second last-position algebra path; the real full-covariance model fixture proves `torch.equal`
against the last slice of a full decode.

## GREEN evidence

Authoritative command from the final implementation diff:

```powershell
C:/Python314/python.exe -m pytest tests/test_cg.py tests/test_prior_bank.py `
  tests/test_families.py tests/test_decode_nonpd_fallback_20260806.py `
  tests/test_ultradeep_remediation_domains_20260813.py tests/test_tier12_decode.py -q `
  --junitxml=.verification/remediation-2026-08-13/task-05-green.xml
```

Result: 102 tests, 0 failures, 0 errors, 0 skipped.

Task 4 precision/family-workspace preservation command:

```powershell
C:/Python314/python.exe -m pytest tests/test_precision_policies_20260806.py `
  tests/test_family_chunked_canonical_dispatch_20260808.py `
  tests/test_family_chunked_workspace_20260807.py -q `
  --junitxml=.verification/remediation-2026-08-13/task-05-numerical-green.xml
```

Result: 59 tests, 0 failures, 0 errors, 0 skipped. The six changed/new Python modules also passed
`C:/Python314/python.exe -m compileall -q`, and `git diff --check` reported no errors.

A broader Task 1/config and legacy decoder-stat consumer lane collected 321 tests: 320 passed and
one unrelated scaling fixture failed because its own `scale_knob` test input is integer `1`, which
reaches `getattr(cfg, name, None)` and raises `TypeError: attribute name must be string, not 'int'`.
Task 5 does not touch `scaling.py` or that fixture; no out-of-scope repair was made.
