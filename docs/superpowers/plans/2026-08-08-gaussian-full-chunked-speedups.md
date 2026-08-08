# Gaussian-Full Chunked Decoder Speedups Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development for each task. Work only in the dedicated feature worktree, preserve unrelated changes, and commit each completed task independently.

**Goal:** Eliminate the canonical `family_chunked` full-Gaussian performance cliff and apply the existing family workspace limit to registered chunked logits without changing generic family semantics.

**Architecture:** Add a narrowly guarded dispatch from the exact built-in FullGaussian/Renyi/alpha=1 active-policy case to the established analytic full-covariance scorer. Keep `reference_decode`, dense `family`, custom registrations, mixed dtypes, noncanonical alpha values, and other functionals on the generic family route. Extend the existing family workspace estimators with an optional scalar-byte override and use a conservative FullGaussian value at each call site.

**Tech Stack:** Python 3, PyTorch, pytest, CUDA test lane through `C:/anaconda/python.exe`, Git.

## Global constraints

- Worktree: `C:/Users/chris and christine/.codex/visualizations/2026/08/08/019fe1e2-29dd-7c31-a5ad-a12b3c77bc59/V3_Transformer_gaussian_full_speedups_20260808`.
- Never edit `train_vfe3.py`, `ablation.py`, or `zzzzz.py`; those are user WIP in the live checkout.
- Use `C:/anaconda/python.exe` for every command that imports torch or makes a CUDA claim.
- Set `VFE3_TEST_DEVICE=cuda` for GPU-lane tests.
- Run the named RED test before each production edit and record the observed failure.
- Do not broaden the specialization by registry name. Compare resolved object identities.
- Do not change configuration values, dense `family`, or `reference_decode`.
- Preserve American English in code, comments, and documentation.

---

## Task 1: Dispatch the exact canonical family route to the analytic scorer

**Files:**

- Create: `tests/test_family_chunked_canonical_dispatch_20260808.py`
- Modify: `tests/test_family_chunked_decode_20260807.py`
- Modify: `tests/test_family_chunked_workspace_20260807.py`
- Modify: `vfe3/model/prior_bank.py`

### Step 1: Keep legacy family tests on the generic route

Change the local bank factories in the two existing family test modules to default to `renyi_order=0.5`. Allow an explicit keyword to override it. This prevents the new canonical dispatch from turning existing generic-workspace tests into tests of the analytic path.

Run the two modules to confirm their generic expectations still pass:

```powershell
$env:VFE3_TEST_DEVICE='cuda'; C:/anaconda/python.exe -m pytest -q tests/test_family_chunked_decode_20260807.py tests/test_family_chunked_workspace_20260807.py
```

### Step 2: Write canonical-dispatch RED tests

Create a small untied full-Gaussian bank (`B=2`, `N=3`, `K=4`, `V=17`, `chunk=5`) with off-diagonal SPD query/prior covariances. Cover:

- fused CE parity between canonical `family_chunked` and `full_chunked`, including nonzero z-loss, nonuniform unigram bias, learned temperature, an ignore index, and the remainder chunk;
- gradients for query mean/covariance, decoder mean/variance tables, and decode log-scale;
- registered-logits parity and a weighted scalar backward, while proving unigram bias is applied once;
- the real `torch.linalg.solve_triangular` call count: canonical `family_chunked` must match analytic `full_chunked` at zero pair-grid solves, while alpha `0.5` must record a positive count;
- generic fallbacks for alpha `0.5`, `squared_hellinger`, a same-name runtime `renyi` override, mixed public dtypes, and non-active precision policies;
- non-PD query positions retain the public exclusion/uniform-logit behavior.

Use `PriorBank.reference_decode` as the generic oracle. Use `atol=5e-6, rtol=2e-5` only when comparing the pre-existing analytic and generic implementations; once the family path dispatches to the analytic implementation, require exact equality to `full_chunked` where inputs and reduction order are identical. Restore every process-global registry or precision override in `finally` blocks/fixtures.

Run the new tests and confirm the operation-count/canonical-dispatch assertions fail on the base implementation:

```powershell
$env:VFE3_TEST_DEVICE='cuda'; C:/anaconda/python.exe -m pytest -q tests/test_family_chunked_canonical_dispatch_20260808.py
```

### Step 3: Implement the narrow predicate and delegation

In `vfe3/model/prior_bank.py`:

- import the built-in `renyi` callable and `FullGaussian`/`full_cov_kl_precision` definitions without replacing the existing registry lookups;
- add a private predicate that resolves `get_family(pb.family)` and `get_functional(pb.divergence_family)` and requires identity with those original built-ins;
- require `type(pb.renyi_order) in (int, float)` and `pb.renyi_order == 1.0`;
- require `full_cov_kl_precision() == "fp32_escalate"` and `decode_av_precision() == "fp32"`;
- require one homogeneous public dtype in `{torch.float32, torch.float64}` across `mu_q`, `sigma_q`, `decode_mu`, the selected raw full-covariance decode parameter tables, and `decode_log_scale`; inspect the raw tables rather than calling `_decode_sigma_log_table()` in the predicate, because the model-channel full path would otherwise materialize the vocabulary table twice;
- in `decode_ce_family_chunked`, delegate to `decode_ce_full_chunked` before constructing generic family state, forwarding all public options exactly;
- in `_decode_family_chunked`, delegate to `_decode_full_chunked` under the same predicate;
- leave `_decode_family` and `reference_decode` untouched.

Do not select by the strings `gaussian_full` or `renyi`; runtime registrations permit overrides under those names.

### Step 4: Run GREEN and regression tests

```powershell
$env:VFE3_TEST_DEVICE='cuda'; C:/anaconda/python.exe -m pytest -q tests/test_family_chunked_canonical_dispatch_20260808.py tests/test_family_chunked_decode_20260807.py tests/test_family_chunked_workspace_20260807.py tests/test_decode_ce_checkpoint_gate_20260807.py
```

### Step 5: Commit Task 1

```powershell
git add vfe3/model/prior_bank.py tests/test_family_chunked_canonical_dispatch_20260808.py tests/test_family_chunked_decode_20260807.py tests/test_family_chunked_workspace_20260807.py
git commit -m "perf: dispatch canonical family decode analytically"
```

---

## Task 2: Enforce dtype-aware family workspaces for CE and registered logits

**Files:**

- Modify: `tests/test_family_chunked_workspace_20260807.py`
- Modify: `vfe3/model/prior_bank.py`

### Step 1: Write workspace RED tests

Extend the workspace module with tests that:

- call `_decode_ce_family_effective_chunk` with an explicit eight-byte scalar cost and prove the width is half the four-byte result for a budget chosen to avoid integer-rounding ambiguity;
- force the workspace budget to one full-family entry and call registered `pb.decode`, spying on the real functional to require a complete vocabulary tiling of width one; current code makes one raw-width call;
- prove wide and forced-width-one logits agree and that the recorded slices cover the vocabulary exactly;
- prove dense `decode_mode="family"` still makes one width-`V` call under the same forced budget;
- under `full_cov_kl_precision="fp64"`, prove both fused CE and registered logits use the eight-byte sizing override even when the public tensors are fp32;
- choose a checkpoint threshold between the four-byte and eight-byte whole-vocabulary estimates and prove fp64 FullGaussian selects checkpointing;
- prove diagonal-family sizing remains unchanged.

Restore the process-global precision policy after every test.

Run the affected tests and record the registered-logits and fp64-sizing failures:

```powershell
$env:VFE3_TEST_DEVICE='cuda'; C:/anaconda/python.exe -m pytest -q tests/test_family_chunked_workspace_20260807.py
```

### Step 2: Add compatible scalar-byte overrides

In `vfe3/model/prior_bank.py`:

- add keyword-only `workspace_bytes_per_scalar: Optional[int] = None` to `_decode_ce_chunk_activation_bytes` and `_decode_ce_family_effective_chunk`;
- default `None` to `ref.element_size()` so all existing callers remain behavior-compatible;
- validate/use the explicit positive byte count in the per-entry arithmetic;
- add a private call-time resolver scoped to the resolved built-in `FullGaussian`: return eight bytes when a public operand is float64 or the active full-covariance policy may use fp64, otherwise use the reference tensor element size;
- pass the resolved scalar bytes into the fused family CE effective-width calculation and whole-vocabulary checkpoint estimate;
- in `_family_logits`, apply `_decode_ce_family_effective_chunk` only when `chunk is not None`, using the same scalar-byte resolver and full-family inner size;
- keep `chunk=None` dense behavior unchanged;
- do not claim or invent a universal cap for custom functionals because the registration has no workset metadata.

### Step 3: Run GREEN and targeted regressions

```powershell
$env:VFE3_TEST_DEVICE='cuda'; C:/anaconda/python.exe -m pytest -q tests/test_family_chunked_workspace_20260807.py tests/test_family_chunked_decode_20260807.py tests/test_family_chunked_canonical_dispatch_20260808.py tests/test_precision_policies_20260806.py tests/test_decode_ce_checkpoint_gate_20260807.py
```

### Step 4: Commit Task 2

```powershell
git add vfe3/model/prior_bank.py tests/test_family_chunked_workspace_20260807.py
git commit -m "fix: cap registered family decode workspaces"
```

---

## Task 3: Verify semantics, memory routing, and GPU speed

**Files:**

- Create/update task-owned artifacts under `.verification/results/`
- Do not modify production configuration.

### Step 1: Run the complete targeted CUDA lane with JUnit

Verify CUDA first, then run the same 96-test baseline selection plus the new tests, writing fresh machine-readable output:

```powershell
C:/anaconda/python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
$env:VFE3_TEST_DEVICE='cuda'; C:/anaconda/python.exe -m pytest -q --junitxml=.verification/results/post-gaussian-full-fixes-20260808.xml tests/test_family_chunked_canonical_dispatch_20260808.py tests/test_family_chunked_workspace_20260807.py tests/test_family_chunked_decode_20260807.py tests/test_decode_ce_checkpoint_gate_20260807.py tests/test_decode_nonpd_20260807.py tests/test_precision_policies_20260806.py tests/test_tier12_full_covariance_20260807.py tests/test_prior_bank.py
```

Parse the JUnit XML for tests/failures/errors/skips. Do not report counts from terminal progress text.

### Step 2: Run synchronized RTX 5090 benchmarks

Use the existing task benchmark harness against this worktree and write JSON results under `.verification/results/`. Measure warmed, synchronized forward and forward/backward calls for both `full_chunked` and canonical `family_chunked` with identical inputs. Include a small correctness shape and the largest safe active-like shape the GPU can execute without changing user processes or configuration.

Success requires canonical `family_chunked` to land in the same performance class as `full_chunked`, and the operation-count tests to show the pair-grid solve path is absent.

### Step 3: Validate an evidence ledger

Create a fresh task ledger bound to the final commit, CUDA environment, exact config/shape, JUnit XML, and benchmark JSON. Record separate claims for semantic parity, generic-route preservation, registered-logits tiling, and measured speedup. Run the repository verification gate and leave every claim either `EVIDENCE_VERIFIED` or explicitly `INCONCLUSIVE` with its open obligation.

### Step 4: Review the complete branch

Generate a full branch diff from `origin/main`, dispatch a fresh read-only code reviewer, resolve any confirmed issue through a new RED test, rerun affected verification, and keep the branch unpushed unless the user asks to publish it.
