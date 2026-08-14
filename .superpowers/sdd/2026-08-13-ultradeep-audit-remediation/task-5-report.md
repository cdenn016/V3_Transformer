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

## Review remediation addendum (supersedes the original expanded-decoder closure)

Task 5 review identified two Important decoder defects at report revision `d57cf024`: a wholly
invalid family-decoder chunk contributed `log(Vc)` mass to a partially valid row, and expanded
decoder promotion was decided from local raw-energy tiles rather than the global final-score
ranking. The original Task 5 ledger and the earlier expanded-decoder paragraph are superseded by
this addendum. Both findings are fixed in implementation commit
`c07f4c1fe2ef36dadfec185cd550358e72cc1716`.

### Review RED evidence

Command:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:VFE3_TEST_DEVICE='cpu'
C:/Python314/python.exe -m pytest `
  tests/test_ultradeep_remediation_domains_20260813.py `
  -k "partial_invalid_vocab or global_cross_chunk or global_final_ranking" -q `
  --junitxml=.verification/remediation-2026-08-13/task-05-review-red.xml
```

Machine-readable result: 10 tests, 10 expected failures, 0 errors, 0 skipped. The four
partial-invalid cases returned a larger CE than the dense finite-mask oracle under both
`decode_ce_checkpoint='never'` and `'always'` at chunk widths 2 and 3. The six ranking cases
remained float32 for cross-chunk competitors at widths 1 and 2 and for final-score cancellations
introduced by the unigram and head-evidence terms.

### Invalid-chunk semantics

`decode_ce_family_chunked` retains the graph-safe local finite placeholder, computes its local
`logsumexp`, and then replaces the summary for a wholly invalid chunk with constant negative
infinity before the cross-chunk reduction. The final row is zeroed only after global validity is
known. Thus an invalid chunk contributes no vocabulary mass, while a valid target in another chunk
remains scored. The literal value-and-gradient oracle cases report `scored_tokens=1` and
`excluded_tokens=0`; parameters belonging only to the invalid chunk receive exactly zero gradient.
Completely invalid rows retain the original explicit exclusion contract: finite grad-connected
zero CE with `scored_tokens=0` and `excluded_tokens` incremented.

### Global final-score interval policy

The fp32 expanded matmul begins with the existing absolute bound

`b_a = (2 n + 4) eps * (abs(lhs) @ abs(rhs)^T + abs(bias))`.

The final-score certificate then propagates this interval through every supported transformation:
the half-difference with the query-only term, the 1-Lipschitz zero clamp, the head-evidence delta,
division by the effective temperature (including a temperature-rounding allowance), and the
unigram-bias addition. Each arithmetic seam includes an explicit fp32 roundoff term. Dense
decoding compares the nominal winner's lower endpoint against every other token's upper endpoint.
Fused decoding emits four constant-size summaries per chunk: nominal winner score and bound,
maximum upper endpoint, and maximum upper endpoint excluding the nominal winner. These summaries
are combined across all chunks, so chunk width 1 cannot bypass the decision and a lower nominal
candidate with a wider interval cannot be missed.

Endpoint addition/subtraction and the overlap comparison run in float64, preventing the decision
itself from rounding away uncertainty. If any row overlaps, the complete batch/vocabulary
expanded decode is recomputed differentiably in float64, reusing the canonical full-covariance
invariant kernel where applicable. The promoted result retains float64 through CE; query and
prior-table gradients flow back to the original fp32 leaves and therefore are stored with expected
fp32 accumulation rounding. If intervals are separated, no promoted decode is performed. A
monkeypatched sentinel proves this non-overlap fast path for diagonal and full covariance at chunk
widths 1 and 2; both dense logits and fused CE remain float32.

### Review GREEN and preservation evidence

Focused review command (the final focused artifact includes the four fast-path cases):

```powershell
C:/Python314/python.exe -m pytest `
  tests/test_ultradeep_remediation_domains_20260813.py `
  -k "partial_invalid_vocab or global_cross_chunk or global_final_ranking or global_interval_nonoverlap" -q `
  --junitxml=.verification/remediation-2026-08-13/task-05-review-green-attempt4.xml
```

Result: 14 tests, 0 failures, 0 errors, 0 skipped.

Final full Task 5 command:

```powershell
C:/Python314/python.exe -m pytest tests/test_cg.py tests/test_prior_bank.py `
  tests/test_families.py tests/test_decode_nonpd_fallback_20260806.py `
  tests/test_ultradeep_remediation_domains_20260813.py tests/test_tier12_decode.py -q `
  --junitxml=.verification/remediation-2026-08-13/task-05-review-full-green-final.xml
```

Result: 116 tests, 0 failures, 0 errors, 0 skipped.

Final Task 4 numerical-preservation command:

```powershell
C:/Python314/python.exe -m pytest tests/test_precision_policies_20260806.py `
  tests/test_family_chunked_canonical_dispatch_20260808.py `
  tests/test_family_chunked_workspace_20260807.py -q `
  --junitxml=.verification/remediation-2026-08-13/task-05-review-numerical-green-final.xml
```

Result: 59 tests, 0 failures, 0 errors, 0 skipped. All commands used
`C:/Python314/python.exe`, `CUDA_VISIBLE_DEVICES=-1`, and `VFE3_TEST_DEVICE=cpu`. The six Task 5
Python modules plus the review fixture passed `compileall -q`; `git diff --check` was clean.

Warnings were expected and unchanged in kind: the literal Renyi-alpha-1.5 fixtures emit the
documented convex-regime warning, and adjacent model fixtures emit existing detached-oracle and
full-covariance/linear-decode configuration warnings. No CUDA path, diagnostic pipeline, training
run, coverage percentage, or unrelated scaling repair is claimed.

### Self-review

- The partial-invalid fix changes only the inter-chunk summary sentinel; local graph construction,
  target gathering, global validity, and scored/excluded accounting remain unchanged.
- The ranking gate uses final logits after clamp, temperature, head evidence, and unigram bias; it
  is global across chunks and conservative for candidates whose upper bound is large.
- The promoted branch is conditional, differentiable, mask/count identical, and shares the
  canonical covariance invariant algebra rather than introducing a parallel full-KL policy.
- The non-overlap sentinel tests rule out unconditional float64 recomputation.
- No files outside `vfe3/model/prior_bank.py`, the Task 5 fixture, this report, and ignored
  revision-bound verification artifacts were changed by the review remediation.

## Performance-contract addendum (supersedes the review interval-scan closure)

A subsequent performance review found that the interval logic fixed in `c07f4c1` cast complete
`(B,N,V)` dense logits/bounds and `(B,N,Vc)` fused chunk tensors to float64 before reducing them.
That preserved correctness but violated the non-overlap workspace contract. The performance defect
is fixed in implementation commit `e0e1ecd`.

### Performance RED

```powershell
C:/Python314/python.exe -m pytest `
  tests/test_ultradeep_remediation_domains_20260813.py `
  -k "never_casts_vocab_or_chunk or bound_matmul_is_included" -q `
  --junitxml=.verification/remediation-2026-08-13/task-05-performance-red.xml
```

Machine-readable result: 6 tests, 6 expected failures, 0 errors, 0 skipped. Four execution
sentinels observed forbidden float64 casts in dense/fused diagonal/full non-overlap scans. Two
workspace fixtures observed that the checkpoint estimate counted one `(B,N,Vc)` tensor instead of
both the expanded score and absolute-bound matmul.

### Outward-rounded reduction and workspace contract

The final propagated fp32 bound is rounded upward by one representable value with
`nextafter(bound, +inf)`. Interval endpoints are then formed in fp32 and rounded outward again:
`nextafter(logit - bound, -inf)` for lower endpoints and
`nextafter(logit + bound, +inf)` for upper endpoints. This preserves the conservative overlap
decision without a vocabulary-sized promotion.

For dense decoding, the nominal winner and its bound are gathered first, producing only `(B,N)`
values; competitor upper endpoints reduce across `V` in fp32. Only the resulting winner-lower and
competitor-upper `(B,N)` summaries are cast to float64 for the stable comparison. For fused
decoding, each chunk reduces in fp32 to nominal winner/bound plus maximum and runner-up upper
endpoints. The global chunk dimension then reduces in fp32, and again only the two final `(B,N)`
summaries promote. The conditional complete float64 recompute remains unchanged and occurs only
when these conservatively outward-rounded intervals overlap.

The execution sentinel wraps both `Tensor.double()` and `Tensor.to(dtype=float64)` and rejects any
cast whose shape contains the literal vocabulary width `V=7` or chunk width `Vc=2`. It exercises
dense and fused diagonal/full decoders on clearly separated logits, checks the fused backward pass,
and confirms outputs remain fp32. Workspace fixtures intercept the real checkpoint gate and pin
each chunk estimate to `B*N*Vc*2*4` bytes: two fp32 expanded tensors (score and absolute bound).

### Final evidence

Focused correctness/performance lane:

```powershell
C:/Python314/python.exe -m pytest `
  tests/test_ultradeep_remediation_domains_20260813.py `
  -k "partial_invalid_vocab or global_cross_chunk or global_final_ranking or global_interval_nonoverlap or never_casts_vocab_or_chunk or bound_matmul_is_included" -q `
  --junitxml=.verification/remediation-2026-08-13/task-05-performance-focused-final.xml
```

Result: 20 tests, 0 failures, 0 errors, 0 skipped.

Full Task 5 lane:

```powershell
C:/Python314/python.exe -m pytest tests/test_cg.py tests/test_prior_bank.py `
  tests/test_families.py tests/test_decode_nonpd_fallback_20260806.py `
  tests/test_ultradeep_remediation_domains_20260813.py tests/test_tier12_decode.py -q `
  --junitxml=.verification/remediation-2026-08-13/task-05-performance-full-green.xml
```

Result: 122 tests, 0 failures, 0 errors, 0 skipped. This is the prior 116-test lane plus the six
new performance-contract cases.

Numerical-preservation lane:

```powershell
C:/Python314/python.exe -m pytest tests/test_precision_policies_20260806.py `
  tests/test_family_chunked_canonical_dispatch_20260808.py `
  tests/test_family_chunked_workspace_20260807.py -q `
  --junitxml=.verification/remediation-2026-08-13/task-05-performance-numerical-green.xml
```

Result: 59 tests, 0 failures, 0 errors, 0 skipped. All runs were CPU-only with
`C:/Python314/python.exe`, `CUDA_VISIBLE_DEVICES=-1`, and `VFE3_TEST_DEVICE=cpu`.

The fixtures are small rather than stress-scale: across the Task 5 lane the maximum `K/embed_dim`
is 9, maximum sequence/context length is 6, and maximum vocabulary is 50. The largest batch value
is 5 in small CG tensor algebra; decoder/model fixtures use at most batch 3 here and ordinarily
batch 2. No scale, training, CUDA, diagnostic, or unrelated brute-force workload was launched.

### Performance self-review

- No non-overlap path casts a `V`- or `Vc`-shaped tensor to float64; promotion is limited to final
  `(B,N)` comparison summaries or the already-required complete recompute after overlap.
- Outward rounding is applied to the propagated bound and both endpoint directions, so removing
  the large float64 scan does not weaken uncertainty safety.
- Dense and fused diagonal/full value and gradient fixtures remain green, including cross-chunk,
  unigram-cancellation, and head-evidence-cancellation cases.
- The expanded checkpoint budget now includes the absolute-bound matmul in addition to the score
  tensor; the test pins every chunk width, including the final short chunk.
- Expected Renyi convex-regime and preexisting detached-oracle/configuration warnings remain; no
  new warning class appeared.
