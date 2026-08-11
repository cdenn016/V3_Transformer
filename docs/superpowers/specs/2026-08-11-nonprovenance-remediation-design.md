# Non-Provenance Audit Remediation

**Date:** 2026-08-11
**Status:** Approved design
**Source revision:** `b033a19d4a75630a88924e57efc25581364fbc6e`
**Scope:** Mechanically verified runtime, numerical, checkpoint, and analysis defects from the
three-wave full-Gaussian audit, excluding provenance integrity and prospective architecture work

## 1. Goal

Repair the audit's verified non-provenance defects without changing the user's active experiment
configuration, disturbing the live GPU process, or folding an MLP into a correctness patch. Every
repair must begin with a regression test that fails for the audited behavior and end with fresh,
revision-bound machine-readable evidence.

## 2. Considered approaches

1. **Repair the verified runtime defects in dependency-ordered groups (selected).** This keeps each
   behavioral contract reviewable, allows focused red/green testing, and separates mathematical
   kernels from checkpoint and reporting interfaces.
2. **Repair only the currently active full-Gaussian forward path.** This would be smaller, but it
   would knowingly leave verified resume, log-Euclidean, and analysis failures in supported paths.
3. **Repair every audit candidate at once.** This would mix verified defects with inconclusive
   hypotheses, provenance policy, experimental-method changes, and the prospective MLP design.
   Those categories need different evidence and approval contracts.

## 3. Included findings

The batch includes these closed non-provenance findings:

- `W3-SPECTRAL-EXP-GRAD`: preserve the Fréchet derivative of the bounded matrix exponential when
  distinct fp32 eigenvalues round to the same exponential value.
- `W3-LOGEUC-REPEATED-GRAD`: preserve off-diagonal derivatives in repeated eigenspaces on the
  log-Euclidean retraction.
- `W2-LOGEUC-NONFINITE`: neutralize and count nonfinite log-Euclidean tangents before trust-region
  scaling and certification.
- `W1-GLOBAL-POLICY`: make full-covariance precision policy instance-owned on all model paths.
- `W1-NATURAL-DTYPE`: promote mixed public dtypes before the full-Gaussian natural-coordinate
  solve.
- `W3-EMISSION-TRACE`: compute `tr(diag(d) Sigma)` from the covariance diagonal.
- `W3-MM-PARTIAL-CHOLESKY`: retain the prior state when any MM Cholesky factorization fails and
  report the fallback.
- `W1-DECODE-NONFINITE`: exclude nonfinite full-covariance rows before decoder arithmetic and
  count the fallback.
- `W3-EXCLUDED-TOKEN-DENOM`: carry the decoder's exact scored-token count through evaluation and
  gradient accumulation.
- `W2-CUDA-RESUME`: validate optimizer snapshots portably before loading, then validate realized
  devices after optimizer state is materialized.
- `W2-OPT-NAMES`: bind optimizer slots to an ordered parameter-name/shape manifest.
- `W2-NEGATIVE-MOMENT`: reject negative Adam second-moment state.
- `W2-MULTISEED-EXTRAS`: withhold requested-seed aggregates when unexpected, duplicate, or
  unidentified runs are present and report the discrepancy explicitly.
- `W2-UNPAIRED-PARETO`: compute validation-loss/time summaries from the same seed cohort.
- `W1-PARAM-GRID`: restore the documented two-width, two-percent parameter-match grid with real
  model dimensions rather than relaxing the tolerance.

## 4. Geometry and information-geometry contract

### Stable spectral divided differences

`_loewner_adjoint` keeps its current repeated-eigenvalue and clamp-flat behavior. For the bounded
exponential only, when a nonzero eigenvalue gap lies inside the active interval but the output gap
rounds to zero, it evaluates the divided difference with `exp(lambda_j) * expm1(gap) / gap`.
Forward values are unchanged; saturated eigenvalues continue to have zero derivative.

The log-Euclidean path uses the same spectral-map adjoint for the floored logarithm, bounded
exponential, and final projection. It reuses the existing eigendecomposition and preserves the
current three-eigendecomposition performance contract. A nonfinite chart tangent is neutralized
before its norm is computed, freezing only affected batch elements and incrementing the existing
nonfinite-tangent counter.

### Instance-owned precision policy

`VFEModel` and its `PriorBank` own an immutable validated full-covariance precision policy.
Model-owned divergence, transport, decoder, and workspace paths receive that policy explicitly;
constructing another model cannot mutate their behavior. The existing module-level setter may
remain for standalone compatibility, but no model-owned path may depend on it.

`FullGaussian.natural()` promotes `mu` and `sigma` to the family's public compute dtype before
constructing the identity and running the existing LU solve. Matching-dtype behavior and the
existing solve algorithm remain unchanged.

## 5. Full-Gaussian update, decoder, and accounting contract

The emission trace uses only `Sigma.diagonal(...)` for a dense covariance and retains the existing
diagonal-family calculation. MM kernels retain and combine every Cholesky success mask, sanitize
failed factors before solves, and select the incoming mean/covariance exactly for failed rows. A
per-device asynchronous counter reports MM Cholesky fallbacks.

Before full-covariance decoding, each nonfinite covariance row is replaced by an identity only for
safe downstream arithmetic. Final row validity is the conjunction of original finiteness and
Cholesky success. Invalid rows contribute neither cross-entropy nor gradient, and they increment
the existing decoder fallback counter. Finite valid rows preserve current values.

Built-in decoder kernels gain an opt-in result object containing scalar cross-entropy and an
unclamped int64 `scored_tokens`. Scalar returns remain the default for direct callers. Registered
custom kernels retain their legacy call signature unless their registration metadata explicitly
declares scored-token support. `VFEModel.forward()` likewise preserves its existing tuple by
default and exposes the count only when requested.

Evaluation weights batch losses by exact scored-token counts and rejects a corpus with no scored
targets. Gradient accumulation represents the same token-level objective: each microbatch loss is
weighted by its scored count, gradients are divided by the total count after unscaling, and a
zero-count accumulation is skipped explicitly. The common one-microbatch valid-input path must not
gain an unconditional device synchronization.

## 6. Checkpoint contract

New checkpoints include a versioned optimizer parameter manifest. Each parameter group records its
ordered parameter names and shapes plus a canonical JSON SHA-256 digest. Resume verifies manifest
integrity and requires an exact match to the live optimizer grouping before mutating model,
optimizer, scaler, scheduler, or RNG state.

Optimizer validation has two phases:

1. Portable preflight validates structure, shape, dtype, finiteness, clock semantics, and
   nonnegative second moments on the CPU-loaded snapshot without requiring snapshot tensors to
   match the live parameter device.
2. After `optimizer.load_state_dict()` performs its normal placement, realized validation checks
   every moment and fused/capturable step tensor against live parameter/device requirements. On
   failure, the optimizer's prior state is restored before any other training state is mutated.

Legacy model-only checkpoints remain loadable. A legacy optimizer checkpoint without a manifest
may resume only when the existing source/config identity checks establish an unchanged execution
contract; if drift is allowed or detected, positional optimizer rebinding fails closed. This is a
compatibility gate, not a repair of the excluded provenance-reporting findings.

## 7. Analysis contract

Requested-seed analysis classifies observed directories as accepted, unexpected, duplicate, or
unidentified. A requested panel is complete only when every requested seed appears exactly once
and no other category is populated. Public summaries report accepted and observed cohorts
separately, and aggregates/figures are withheld for an invalid panel.

Pareto and capacity summaries join validation loss and wall time by exact seed before aggregation.
They report paired and missing seed sets; both coordinates of each published point use the paired
cohort.

The default parameter-match grid retains the existing two-percent scientific tolerance and adds
realizable widths so at least two candidates satisfy it. Existing candidates and public result
schema remain stable.

## 8. Compatibility and non-goals

- Do not modify `train_vfe3.py`, `ablation.py`, `vfe3/config.py`, or `zzzzz.py` in the user's live
  checkout; all work occurs in the isolated remediation worktree.
- Do not alter active configuration values, terminate processes, or run CUDA tests while the live
  GPU job is resident.
- Do not repair `W1-PROCESS-IDENTITY`, `W1-MULTISEED-DRIFT`, or `W3-START-PROVENANCE`.
- Do not promote any of the audit's 34 inconclusive claims to established defects.
- Do not change experimental comparison protocols, reinterpret block-SPD as unrestricted SPD,
  revise the FLOP model, or add an MLP in this batch.
- Preserve default public return types and serialized model/config compatibility wherever the
  corrected semantics do not require new opt-in metadata.

## 9. Verification strategy

Each task follows red/green TDD with the audit reproducer inverted into a regression. Focused CPU
tests run under `C:/Python314/python.exe` with `CUDA_VISIBLE_DEVICES=-1` and
`VFE3_TEST_DEVICE=cpu`. The final branch receives focused and aggregate JUnit XML, static checks,
an independent code review, and a fresh closure-mode verification ledger bound to the final Git
revision. CUDA-specific closure is attempted only after an idle-GPU check; otherwise the CPU-side
resume correction is reported separately from the still-open device integration obligation.
