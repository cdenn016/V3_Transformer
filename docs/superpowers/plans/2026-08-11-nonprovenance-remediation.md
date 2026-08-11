# Non-Provenance Audit Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to
> implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the mechanically verified non-provenance defects from the 2026-08-10 three-wave
audit while preserving public compatibility, the live experiment configuration, and active GPU
work.

**Architecture:** Correct the lowest-level spectral and Gaussian contracts first, then propagate
exact validity/count information through MM, decoding, training, and evaluation. Harden optimizer
checkpoint binding with portable and realized validation phases. Finish with cohort-safe analysis
and the minimal real parameter grid. Every task is a separate red/green commit and receives an
independent review before the next task begins.

**Tech Stack:** Python 3.14 CPU interpreter, PyTorch, pytest, JUnit XML, Ruff, Git, JSON/SHA-256.

## Global constraints

- Work only in the isolated `codex/nonprovenance-remediation-20260811` worktree.
- Do not touch the user's live checkout, configuration edits, untracked files, GPU process, or run
  artifacts.
- Use `C:/Python314/python.exe` with `CUDA_VISIBLE_DEVICES=-1` and `VFE3_TEST_DEVICE=cpu` for CPU
  tests. Do not make a CUDA-closure claim without `C:/anaconda/python.exe`, an idle-GPU check, and
  `VFE3_TEST_DEVICE=cuda`.
- Use `apply_patch` for edits. Do not revert or rewrite another task's commits.
- Preserve default public return types and matching-dtype/finite-valid behavior.
- Do not implement provenance fixes, inconclusive audit candidates, experiment-policy changes, or
  the prospective MLP.
- Record final evidence in JUnit XML and a fresh closure ledger bound to the final source revision.

---

### Task 1: Stable SPD spectral and log-Euclidean derivatives

**Findings:** `W3-SPECTRAL-EXP-GRAD`, `W3-LOGEUC-REPEATED-GRAD`, `W2-LOGEUC-NONFINITE`

**Files:**

- Modify: `vfe3/geometry/retraction.py`
- Test: `tests/test_retraction.py`
- Test: `tests/test_nonfinite_tangent_guard_20260806.py`
- Test: `tests/test_curated_geometry_math_20260709.py`

**Interfaces:**

- Preserve `retract_spd_full(...) -> Tensor` and `retract_logeuclidean_full(...) -> Tensor`.
- Extend the internal spectral-map operation set with a floored logarithm and use the existing
  Loewner adjoint for its backward.
- Preserve the log-Euclidean three-eigendecomposition contract.

- [ ] **Step 1: Add RED regressions**

Add tests that assert:

1. An fp32 tangent with eigenvalues `0` and `1e-8` has the analytic off-diagonal exponential
   derivative rather than zero, while two values in one flat clamp region still have zero
   derivative.
2. `retract_logeuclidean_full(I, 0)` has identity backward for a symmetric off-diagonal
   cotangent at a repeated spectrum.
3. NaN and both infinities in one log-Euclidean tangent freeze only that batch row, increment the
   nonfinite-tangent counter, leave its neighbor unchanged, and leave finite inputs bitwise inert.

- [ ] **Step 2: Prove RED with machine-readable output**

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:VFE3_TEST_DEVICE='cpu'
& 'C:/Python314/python.exe' -m pytest -q -p no:cacheprovider `
  tests/test_retraction.py tests/test_nonfinite_tangent_guard_20260806.py `
  tests/test_curated_geometry_math_20260709.py `
  -k 'near_degenerate or repeated_spectrum or logeuclidean_nonfinite' `
  --junitxml=.verification/task-01-red.xml
```

Expected: the new assertions fail for the audited derivative/fallback behavior.

- [ ] **Step 3: Implement the minimal spectral repairs**

Use a stable `expm1` divided difference only for nonzero, active-interval exponential gaps whose
rounded output gap is zero. Add the floored-log spectral values/derivatives, route log-Euclidean
reconstructions through the cached spectral-map adjoint, and neutralize the chart tangent before
the trust-region norm.

- [ ] **Step 4: Prove GREEN and adjacent invariants**

Run the three files above without `-k`, write `.verification/task-01-green.xml`, and explicitly
confirm the three-eigendecomposition, float64-island, finite-backward, and affine retraction tests.

- [ ] **Step 5: Commit**

```powershell
git add vfe3/geometry/retraction.py tests/test_retraction.py `
  tests/test_nonfinite_tangent_guard_20260806.py tests/test_curated_geometry_math_20260709.py
git commit -m "fix: stabilize SPD spectral derivatives"
```

---

### Task 2: Instance-owned full-Gaussian policy and mixed dtypes

**Findings:** `W1-GLOBAL-POLICY`, `W1-NATURAL-DTYPE`

**Files:**

- Modify: `vfe3/families/gaussian.py`
- Modify: `vfe3/families/base.py`
- Modify: `vfe3/model/model.py`
- Modify: `vfe3/model/prior_bank.py`
- Modify: `vfe3/inference/e_step.py`
- Modify: `vfe3/gradients/oracle.py`
- Modify: `vfe3/viz/extract.py`
- Test: `tests/test_precision_policies_20260806.py`
- Test: `tests/test_family_chunked_canonical_dispatch_20260808.py`
- Test: `tests/test_families.py`
- Test: `tests/test_fix_numerics_audit.py`

**Interfaces:**

- Add an explicit optional precision-policy argument through model-owned full-Gaussian divergence,
  transport, decoder, and workspace paths.
- Preserve the module-level policy only as the default for standalone callers.
- Make `FullGaussian.natural()` compute in `self._public_dtype`.

- [ ] **Step 1: Add RED regressions**

Add a two-model test proving construction of model B cannot change model A's stored policy or an
actual model-A kernel result. Add `(mu32, sigma64)` and `(mu64, sigma32)` natural-coordinate cases
against an explicit promoted-dtype solve, retaining the same-dtype bitwise regression.

- [ ] **Step 2: Prove RED**

Run the four test files with a focused expression and `.verification/task-02-red.xml`. Expect the
cross-model result to change under the global mutation and mixed-dtype solves to fail.

- [ ] **Step 3: Implement explicit ownership**

Validate and store the policy on `VFEModel` and `PriorBank`; remove model-construction mutation;
thread the value through all model-owned coupling, self-divergence, transport, decoder dispatch,
and workspace helpers. Promote `mu`, `sigma`, and the identity to the public compute dtype before
the unchanged LU solve.

- [ ] **Step 4: Prove GREEN and call-site completeness**

Run the four files without the focused expression plus:

```powershell
& 'C:/Python314/python.exe' -m pytest -q -p no:cacheprovider `
  tests/test_model.py tests/test_prior_bank.py tests/test_e_step.py `
  --junitxml=.verification/task-02-green.xml
```

Use `rg` to verify every model-owned full-covariance consumer passes the instance policy.

- [ ] **Step 5: Commit**

```powershell
git add vfe3/families/gaussian.py vfe3/families/base.py vfe3/model/model.py `
  vfe3/model/prior_bank.py vfe3/inference/e_step.py vfe3/gradients/oracle.py `
  vfe3/viz/extract.py tests/test_precision_policies_20260806.py `
  tests/test_family_chunked_canonical_dispatch_20260808.py tests/test_families.py `
  tests/test_fix_numerics_audit.py
git commit -m "fix: isolate full Gaussian precision policy"
```

---

### Task 3: Correct emission trace and MM Cholesky failures

**Findings:** `W3-EMISSION-TRACE`, `W3-MM-PARTIAL-CHOLESKY`

**Files:**

- Modify: `vfe3/inference/e_step.py`
- Modify: `vfe3/gradients/kernels.py`
- Modify: `vfe3/numerics.py`
- Modify: `vfe3/train.py`
- Test: `tests/test_e_step.py`
- Test: `tests/test_numerics.py`
- Test: `tests/test_run_artifacts.py`

**Interfaces:**

- Preserve the MM public `(mu, sigma)` result.
- Add reset/read/increment helpers for an asynchronous per-device MM Cholesky fallback counter and
  publish it beside existing numerical fallback metrics.

- [ ] **Step 1: Add RED regressions**

Assert the dense emission fixture contributes `4.0`, has exactly zero off-diagonal covariance
gradient, and retains diagonal-family behavior. Add MM tests in which failed `sigma_star` and a
monkeypatched partial factor retain the old mean/covariance exactly, while a mixed valid/invalid
batch updates only its valid row and increments the new counter.

- [ ] **Step 2: Prove RED**

Run the focused tests with `.verification/task-03-red.xml`; current emission should contribute
`4.375`, and current MM should consume the failed partial factor.

- [ ] **Step 3: Implement masked MM and reporting**

Use `belief.sigma.diagonal(...)` for the dense emission variance. Retain every Cholesky success
mask, replace failed factors with identity before solve/inversion, combine masks, certify candidate
rows, and finally select the old state for failures. Add/reset/publish the counter without a
per-step device synchronization.

- [ ] **Step 4: Prove GREEN**

Run the three listed files and the focused full-Gaussian E-step/MM suites to
`.verification/task-03-green.xml`.

- [ ] **Step 5: Commit**

```powershell
git add vfe3/inference/e_step.py vfe3/gradients/kernels.py vfe3/numerics.py `
  vfe3/train.py tests/test_e_step.py tests/test_numerics.py tests/test_run_artifacts.py
git commit -m "fix: reject failed full Gaussian MM rows"
```

---

### Task 4: Exclude invalid decoder rows with exact token accounting

**Findings:** `W1-DECODE-NONFINITE`, `W3-EXCLUDED-TOKEN-DENOM`

**Files:**

- Modify: `vfe3/model/prior_bank.py`
- Modify: `vfe3/model/model.py`
- Modify: `vfe3/train.py`
- Modify: decoder registration module if registration metadata is defined elsewhere
- Test: `tests/test_decode_nonpd_fallback_20260806.py`
- Test: `tests/test_family_chunked_canonical_dispatch_20260808.py`
- Test: `tests/test_train.py`
- Test: `tests/test_grad_accum.py`

**Interfaces:**

- Add immutable `DecodeCEResult(ce, scored_tokens)`.
- Add `return_stats: bool = False` to built-in decoder CE methods.
- Add `return_decode_stats: bool = False` to `VFEModel.forward()`.
- Add registration metadata so legacy custom decoders keep their existing signature.
- Keep default scalar decoder returns, default model tuple, and `evaluate()` result keys unchanged.

- [ ] **Step 1: Add RED regressions**

Cover NaN and infinity covariance rows, manual-ignore loss equivalence, exact surviving counts,
fallback counts, finite backward, and zero excluded-row gradient. Add evaluation batches with
`(CE,count)=(2,1),(4,3)` expecting `3.5`; require a no-scored-token corpus to raise. Add a
two-microbatch gradient objective expecting `(1*1 + 3*3)/4 = 2.5` and a combined ignore/non-PD
count case.

- [ ] **Step 2: Prove RED**

Run the focused decoder/evaluation/accumulation cases to `.verification/task-04-red.xml`.

- [ ] **Step 3: Implement sanitized decoding and count propagation**

Sanitize nonfinite rows before all decoder arithmetic, combine finiteness and Cholesky masks, and
count every exclusion. Return exact int64 counts from built-ins when opted in. Propagate the count
through model forward, evaluation, and training. Weight accumulated losses by counts; after
GradScaler unscale, divide gradients by the total scored count. Explicitly skip a zero-count
training accumulation.

- [ ] **Step 4: Prove GREEN and compatibility**

Run all four listed files plus dense/diagonal/family decoder tests to
`.verification/task-04-green.xml`. Assert legacy custom decoder callables and the exact four-key
evaluation result remain unchanged. Inspect the common `grad_accum_steps=1` path for no new
unconditional scalar synchronization.

- [ ] **Step 5: Commit**

```powershell
git add vfe3/model/prior_bank.py vfe3/model/model.py vfe3/train.py `
  tests/test_decode_nonpd_fallback_20260806.py `
  tests/test_family_chunked_canonical_dispatch_20260808.py tests/test_train.py `
  tests/test_grad_accum.py
git commit -m "fix: account for decoder-excluded tokens"
```

---

### Task 5: Bind and portably validate optimizer checkpoint state

**Findings:** `W2-CUDA-RESUME`, `W2-OPT-NAMES`, `W2-NEGATIVE-MOMENT`

**Files:**

- Modify: `vfe3/run_artifacts.py`
- Test: `tests/test_checkpoint_resume.py`
- Test: `tests/test_run_artifacts.py`

**Interfaces:**

- Add checkpoint field `optimizer_parameter_manifest` with schema version, ordered named/shape
  groups, and canonical SHA-256 digest.
- Split internal optimizer validation into portable snapshot and realized live-device phases.
- Preserve model-only resume and unchanged-contract legacy optimizer resume.

- [ ] **Step 1: Add RED regressions**

Assert a CPU-mapped populated fused optimizer snapshot passes portable preflight, then validates
after normal load placement. Assert same-shaped parameter reordering is rejected by name, a
tampered manifest digest is rejected, negative `exp_avg_sq` and `max_exp_avg_sq` are rejected, and
signed `exp_avg` remains valid. Assert model/optimizer/scaler/scheduler/RNG state is unchanged after
any pre-mutation failure. Cover legacy unchanged-contract acceptance and drift-enabled rejection.

- [ ] **Step 2: Prove RED**

Run only the new cases to `.verification/task-05-red.xml` with the CPU interpreter.

- [ ] **Step 3: Implement manifest and two-phase validation**

Build the manifest from `model.named_parameters()` and exact optimizer group order at save time.
Validate schema, digest, names, and shapes before mutation. Remove live-device equality from CPU
preflight; load optimizer state; validate realized moments and fused/capturable step devices; and
restore the optimizer snapshot if realized validation fails. Require nonnegative second moments.

- [ ] **Step 4: Prove CPU GREEN and record CUDA obligation**

Run both full files in bounded groups, writing `.verification/task-05-green-part*.xml`. If the GPU
is still resident, do not run CUDA tests and leave only the real-device integration claim open. If
it is idle, verify `C:/anaconda/python.exe` reports CUDA and run the targeted resume test with
`VFE3_TEST_DEVICE=cuda` into `.verification/task-05-cuda.xml`.

- [ ] **Step 5: Commit**

```powershell
git add vfe3/run_artifacts.py tests/test_checkpoint_resume.py tests/test_run_artifacts.py
git commit -m "fix: bind optimizer checkpoints by parameter name"
```

---

### Task 6: Make cross-run analysis cohorts internally consistent

**Findings:** `W2-MULTISEED-EXTRAS`, `W2-UNPAIRED-PARETO`, `W1-PARAM-GRID`

**Files:**

- Modify: `multiseed_analysis.py`
- Modify: `vfe3/viz/sweep_adapters.py`
- Modify: `ablation.py`
- Test: `tests/test_multiseed.py`
- Test: `tests/test_2026_07_15_driver_reliability_remediation.py`
- Test: `tests/test_figures_tail.py`
- Test: `tests/test_scaling_mup.py`

**Interfaces:**

- Extend requested-seed design/summary diagnostics with accepted, observed, unexpected,
  duplicate, and unidentified seeds without removing existing keys.
- Extend sweep-adapter points with paired/missing cohort metadata while preserving existing scalar
  fields.
- Keep the parameter-match target at `30_000_000` and tolerance at `0.02`.

- [ ] **Step 1: Add RED regressions**

Create a requested `[8]` fixture with observed `s8,s6,s6`; require the panel to be incomplete,
aggregates/figures withheld, and all discrepancy categories reported. Create validation/time data
whose only paired seed has `(bits=6,time=100)` and require both coordinates to use that seed.
Update the existing parameter-grid test to expect grids containing `K=45`, `H=5`, 90 Cartesian
candidates, and exactly `(45,5,29_452_186)` plus `(60,10,30_200_281)`.

- [ ] **Step 2: Prove RED**

Run the three focused tests to `.verification/task-06-red.xml`.

- [ ] **Step 3: Implement cohort joins and minimal grid expansion**

Classify every observed directory before completeness is decided; expose accepted and observed
counts separately; withhold publication on extras/duplicates/unidentified runs. Join validation
and time by seed before means. Add only `45` to `embed_dim` and `5` to `n_heads`; do not relax the
tolerance.

- [ ] **Step 4: Prove GREEN**

Run `tests/test_multiseed.py`, the driver-reliability module, and all sweep-adapter/scaling tests to
`.verification/task-06-green.xml`.

- [ ] **Step 5: Commit**

```powershell
git add multiseed_analysis.py vfe3/viz/sweep_adapters.py ablation.py `
  tests/test_multiseed.py tests/test_2026_07_15_driver_reliability_remediation.py `
  tests/test_figures_tail.py tests/test_scaling_mup.py
git commit -m "fix: align cross-run analysis cohorts"
```

---

### Task 7: Independent review and revision-bound closure

**Files:**

- Add: `.verification/nonprovenance-remediation-20260811-ledger.json` (ignored evidence artifact)
- Add: JUnit XML under `.verification/` (ignored evidence artifacts)
- Modify: `docs/audits/deep-audit-three-wave-2026-08-10.md` only to append remediation status and
  exact evidence links; do not rewrite the historical audit verdict.

- [ ] **Step 1: Review every task commit**

After each implementation commit, generate an SDD review package and dispatch an independent
reviewer. Resolve all correctness findings before beginning the next implementation task.

- [ ] **Step 2: Run static and aggregate CPU verification**

Run Ruff on modified Python files, the focused full-Gaussian 231-test lane, the CPU-fast lane in
bounded partitions, and every task regression into fresh JUnit XML. Derive totals only from XML.

- [ ] **Step 3: Start and adjudicate the closure ledger**

After the final source commit, start a new closure-mode ledger. Record one claim per finding,
source/config/environment identity, at least two independent views, eligible mechanical evidence,
and an adjudicator. Mark any unavailable CUDA integration obligation `INCONCLUSIVE`, never
verified by inference from CPU tests.

- [ ] **Step 4: Perform final code review**

Dispatch a clean-context reviewer over the complete base-to-head diff, the design, the plan, and
the ledger. Repair and re-run affected evidence for every accepted finding.

- [ ] **Step 5: Append remediation status and commit**

Append a dated remediation section to the audit report listing fixed, refuted, and still-open
claims with exact final revision and JUnit totals. Commit the report and plan checkbox updates.

- [ ] **Step 6: Handoff without publishing**

Report the isolated worktree, branch, commits, test totals, ledger states, and remaining
obligations. Do not merge or push unless the user asks.
