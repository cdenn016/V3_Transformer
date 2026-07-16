# Phi Projection Hot-Path Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the exact global embedded-matrix norm bound imposed by `phi_mstep_max_matrix_norm` while removing the dense vocabulary-by-generator-by-matrix contraction and per-chunk CUDA synchronization from accepted optimizer steps.

**Architecture:** Gauge-group builders explicitly certify Frobenius-orthogonal generator bases. A shared norm kernel uses the certified diagonal Gram form and falls back to exact dense embedding for uncertified bases. The projector continues to scan every eligible row after each accepted optimizer step, but uses memory-budgeted coordinate chunks, keeps reductions on device, and extracts Python diagnostics only on logged steps. The training call site, transport-clamp warning, run summaries, and a standalone CUDA benchmark consume the same route metadata and norm definition.

**Tech Stack:** Python 3, PyTorch, CUDA events, pytest, JUnit XML, JSON, Git worktrees.

## Global Constraints

- Do not change `train_vfe3.py`, `ablation.py`, any ablation sweep value, or `SWEEP_ORDER`.
- `phi_mstep_max_matrix_norm=None` must remain a true no-op.
- The hard bound remains global across token, positional, model-token, and model-positional phi tables after every accepted optimizer step.
- Optimizer moments remain untouched by projection.
- Frobenius orthogonality is an explicit builder capability, never inferred from a group name at the training call site.
- Uncertified bases use an exact fallback and emit a one-time performance warning.
- Silent steps cannot perform diagnostic device-to-host extraction or timing synchronization.
- Pass counts come from JUnit XML; pytest commands do not add `-q` because repository configuration already supplies it.
- The known baseline failure `tests/test_runnable_tail_buildout.py::test_runnable_cluster_sweeps_build[sigma_max]` remains out of scope because the preserved `sigma_max` sweep contains one cell.
- Update `docs/2026-07-15-edits.md`; do not create a second dated edit record.

### Task 1: Certify generator Gram structure

**Files:**
- Modify: `vfe3/geometry/groups.py`
- Create: `tests/test_phi_projection_optimization_20260715.py`

**Interfaces:**
- Add `GaugeGroup.frobenius_gram_diagonal: bool = False`.
- Add `GaugeGroup.gram_diagonal() -> Optional[torch.Tensor]`, cached by generator object, device, dtype, shape, and tensor version.
- Add `GaugeGroup.gram_diagonal_uniform() -> Optional[float]`, cached once for the unit or constant-diagonal fast route.
- Add `GaugeGroup.phi_norm_route() -> str`, returning `"diagonal_gram"` or `"dense_fallback"`.
- Certify `glk`, non-closed `block_glk`, `tied_block_glk`, `so_k`, and `sp`; leave `so_n`, `sp_n`, and closed custom bases uncertified unless construction itself proves the property.

- [x] **Step 1: Write failing capability and Gram-invariant tests**

```python
@pytest.mark.parametrize("name,kwargs", [
    ("glk", {}),
    ("block_glk", {"n_heads": 2}),
    ("block_glk", {"n_heads": 2, "cross_couplings": [(0, 1)]}),
    ("tied_block_glk", {"n_heads": 2}),
    ("so_k", {}),
    ("sp", {}),
])
def test_certified_gram_diagonal_matches_dense_gram(name, kwargs):
    group = get_group(name)(4, dtype=torch.float64, **kwargs)
    gram = torch.einsum("aij,bij->ab", group.generators, group.generators)
    torch.testing.assert_close(group.gram_diagonal(), gram.diagonal())
    torch.testing.assert_close(gram - torch.diag(gram.diagonal()), torch.zeros_like(gram))
    assert group.phi_norm_route() == "diagonal_gram"


def test_closed_basis_fails_closed_to_dense_route():
    group = get_group("block_glk")(
        4, 2, cross_couplings=[(0, 1)], close_basis=True)
    assert group.gram_diagonal() is None
    assert group.phi_norm_route() == "dense_fallback"
```

- [x] **Step 2: Witness the missing metadata**

Run: `python -m pytest tests/test_phi_projection_optimization_20260715.py --junitxml=C:\tmp\vfe3-phi-projection-groups-red-20260715.xml`

Expected: collection or attribute failures naming the absent `GaugeGroup` capability and methods.

- [x] **Step 3: Implement the minimum explicit capability and cache**

Compute diagonal weights as `generators.square().sum(dim=(-2, -1)).detach()` only for certified groups. Cache the tensor and the optional uniform scalar without constructing a dense generator Gram. Mark `block_glk` certified when `close_basis=False`, including its unique elementary cross-coupling basis. Do not certify bracket-closed output.

- [x] **Step 4: Run the group tests**

Run: `python -m pytest tests/test_phi_projection_optimization_20260715.py tests/test_gauge_groups.py --junitxml=C:\tmp\vfe3-phi-projection-groups-green-20260715.xml`

Expected: zero failures and zero errors.

- [x] **Step 5: Commit**

```powershell
git add vfe3/geometry/groups.py tests/test_phi_projection_optimization_20260715.py
git commit -m "feat: certify diagonal generator Gram bases"
```

### Task 2: Replace dense projection with the shared exact norm kernel

**Files:**
- Modify: `vfe3/gauge_optim.py`
- Modify: `tests/test_phi_projection_optimization_20260715.py`
- Modify: `tests/test_phi_numerics_buildout_20260715.py`

**Interfaces:**
- Add `embedded_phi_frobenius_norm(phi: torch.Tensor, group: GaugeGroup) -> torch.Tensor`.
- Retain `project_phi_parameter_rows_(model, max_matrix_norm, *, chunk_rows=None, temporary_bytes=67108864, collect_stats=True) -> Dict[str, float]`.
- `chunk_rows=None` selects a coordinate-width-aware chunk; an explicit positive integer remains available for deterministic tests and compatibility.
- The returned dictionary is empty when `collect_stats=False`; with `True`, the five historical numeric keys retain their meanings.

- [x] **Step 1: Write failing dense-oracle and dispatch tests**

```python
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_diagonal_norm_kernel_matches_dense_embedding(dtype):
    group = get_group("block_glk")(8, 2, dtype=dtype)
    phi = torch.randn(13, group.generators.shape[0], dtype=dtype)
    expected = torch.linalg.matrix_norm(
        torch.einsum("ra,aij->rij", phi, group.generators),
        ord="fro", dim=(-2, -1))
    torch.testing.assert_close(embedded_phi_frobenius_norm(phi, group), expected)


def test_uncertified_nonorthogonal_group_uses_exact_dense_fallback():
    group = nonorthogonal_custom_group()
    phi = torch.randn(7, group.generators.shape[0])
    assert group.phi_norm_route() == "dense_fallback"
    torch.testing.assert_close(
        embedded_phi_frobenius_norm(phi, group), dense_norm(phi, group.generators))


def test_silent_projection_matches_dense_oracle_and_returns_no_stats():
    model = four_phi_table_model()
    expected = dense_projected_state(model, radius=2.0)
    assert project_phi_parameter_rows_(model, 2.0, collect_stats=False) == {}
    assert_model_phi_tables_close(model, expected)
```

- [x] **Step 2: Witness dense-only behavior or missing interfaces**

Run: `python -m pytest tests/test_phi_projection_optimization_20260715.py tests/test_phi_numerics_buildout_20260715.py --junitxml=C:\tmp\vfe3-phi-projection-kernel-red-20260715.xml`

Expected: failures for the missing norm helper, missing silent mode, and missing automatic chunking.

- [x] **Step 3: Implement the exact fast and fallback routes**

For a uniform certified diagonal, calculate `torch.linalg.vector_norm(phi, dim=-1) * sqrt(weight)`. For a nonuniform diagonal, calculate `sqrt(sum(phi.square() * weights))`. For uncertified bases, retain `einsum("ra,aij->rij")` and matrix Frobenius norm. Warn once per group instance on the fallback route.

The projector selects a chunk size bounded by `temporary_bytes`, accumulates count and extrema in zero-dimensional device tensors, and calls no `int(tensor)`, `float(tensor)`, `.item()`, `nonzero()`, or CPU transfer inside the chunk loop. Extract the five historic metrics once after all tables only when `collect_stats=True`.

- [x] **Step 4: Add coverage for all tables, below-bound identity, invalid arguments, deduplication, and exact metric values**

Use dense embedding only in test oracles. Exercise explicit `chunk_rows=3`, automatic chunks, float32, float64, certified `sp` nonuniform weights, and the nonorthogonal fallback.

- [x] **Step 5: Run projector tests**

Run: `python -m pytest tests/test_phi_projection_optimization_20260715.py tests/test_phi_numerics_buildout_20260715.py --junitxml=C:\tmp\vfe3-phi-projection-kernel-green-20260715.xml`

Expected: zero failures and zero errors.

- [x] **Step 6: Commit**

```powershell
git add vfe3/gauge_optim.py tests/test_phi_projection_optimization_20260715.py tests/test_phi_numerics_buildout_20260715.py
git commit -m "perf: eliminate dense phi projection embedding"
```

### Task 3: Remove the second dense Gram path from clamp diagnostics

**Files:**
- Modify: `vfe3/train.py`
- Modify: `tests/test_phi_projection_optimization_20260715.py`
- Modify: `tests/test_phi_numerics_buildout_20260715.py`

**Interfaces:**
- `_warn_phi_transport_clamp` consumes `embedded_phi_frobenius_norm` and the same chunk-sizing helper as the projector.
- Remove `_PHI_CLAMP_GRAM_CACHE`, `_phi_clamp_gram_key`, and `_cached_phi_clamp_gram` after their final consumer is migrated.

- [x] **Step 1: Write a failing shared-kernel routing test**

Patch `vfe3.train.embedded_phi_frobenius_norm` with a recording wrapper, run `_warn_phi_transport_clamp` on a certified small model, and assert that the helper receives coordinate chunks and that no dense `(n_gen, n_gen)` Gram is created.

- [x] **Step 2: Witness the current private Gram route**

Run: `python -m pytest tests/test_phi_projection_optimization_20260715.py -k transport_clamp --junitxml=C:\tmp\vfe3-phi-clamp-red-20260715.xml`

Expected: failure because `_warn_phi_transport_clamp` still calls `_cached_phi_clamp_gram`.

- [x] **Step 3: Migrate the warning to the shared norm definition**

Chunk each eligible table, reduce the maximum on device, and retain the existing log-cadence-only host comparison and warning text. The warning path may synchronize once because it already runs only on metrics cadence; it may not allocate a dense generator Gram for certified K240 `block_glk`.

- [x] **Step 4: Run clamp and train diagnostics tests**

Run: `python -m pytest tests/test_phi_projection_optimization_20260715.py tests/test_phi_numerics_buildout_20260715.py tests/test_train.py --junitxml=C:\tmp\vfe3-phi-clamp-green-20260715.xml`

Expected: zero failures and zero errors.

- [x] **Step 5: Commit**

```powershell
git add vfe3/train.py tests/test_phi_projection_optimization_20260715.py tests/test_phi_numerics_buildout_20260715.py
git commit -m "perf: share phi norm kernel with clamp diagnostics"
```

### Task 4: Gate statistics and timing at the training boundary

**Files:**
- Modify: `vfe3/train.py`
- Modify: `tests/test_phi_projection_optimization_20260715.py`
- Modify: `tests/test_phi_numerics_buildout_20260715.py`

**Interfaces:**
- `train_step` passes `collect_stats=metrics_out is not None`.
- Logged accepted steps add `phi_chart_projection_ms` and `phi_chart_projection_stats_collected=1.0`.
- Silent accepted steps perform projection without CUDA-event creation, synchronization, or Python scalar extraction.
- Skipped and disabled steps call neither the projector nor the timer.

- [x] **Step 1: Write failing logged, silent, skipped, and disabled integration tests**

Use a projector spy to assert the `collect_stats` value. Patch CUDA event construction in a CPU test to remain unreachable on silent steps. Preserve the existing disabled-path test and add an optimizer-step skip test.

- [x] **Step 2: Witness the current unconditional stats request**

Run: `python -m pytest tests/test_phi_projection_optimization_20260715.py tests/test_phi_numerics_buildout_20260715.py -k 'train_step or projection' --junitxml=C:\tmp\vfe3-phi-training-red-20260715.xml`

Expected: failure because `train_step` does not pass `collect_stats` and records no timing fields.

- [x] **Step 3: Implement metrics-cadence-only timing**

On CUDA, create start and end events only when `metrics_out` is present, record around projection, synchronize the end event once, and store elapsed milliseconds. On CPU, use `time.perf_counter()` only on logged steps. Do not move projection relative to the accepted optimizer step, barycenter update, or scheduler step.

- [x] **Step 4: Run training integration tests**

Run: `python -m pytest tests/test_phi_projection_optimization_20260715.py tests/test_phi_numerics_buildout_20260715.py tests/test_train.py tests/test_fp16_gradscaler.py tests/test_round3_train_sync.py --junitxml=C:\tmp\vfe3-phi-training-green-20260715.xml`

Expected: zero failures and zero errors.

- [x] **Step 5: Commit**

```powershell
git add vfe3/train.py tests/test_phi_projection_optimization_20260715.py tests/test_phi_numerics_buildout_20260715.py
git commit -m "perf: gate phi projection diagnostics to logged steps"
```

### Task 5: Persist route provenance and expose timing in geometry reporting

**Files:**
- Modify: `vfe3/run_artifacts.py`
- Modify: `vfe3/viz/figures.py`
- Modify: `tests/test_phi_projection_optimization_20260715.py`
- Modify: `tests/test_viz.py`

**Interfaces:**
- Both summary-writing finalizers add `phi_chart_norm_route`, set to `None` when the bound is disabled and to the group route otherwise.
- The geometry-health history subset includes `phi_chart_projection_ms`.
- `plot_geometry_health` shows timing only when the column is present; existing histories remain valid.

- [x] **Step 1: Write failing summary and optional-plot tests**

Verify disabled and enabled summary provenance without running long training. Verify that geometry-health plotting accepts old history and adds a labeled timing trace when the new field exists.

- [x] **Step 2: Witness missing provenance and plot routing**

Run: `python -m pytest tests/test_phi_projection_optimization_20260715.py tests/test_viz.py -k 'phi_chart or geometry_health' --junitxml=C:\tmp\vfe3-phi-reporting-red-20260715.xml`

Expected: failures for missing `phi_chart_norm_route` and omitted timing history.

- [x] **Step 3: Implement summary and optional figure fields**

Read route metadata from `model.group`; do not serialize a string into numeric `metrics.csv`. Keep timing absent from silent rows and absent from plots for historical runs.

- [x] **Step 4: Run reporting tests**

Run: `python -m pytest tests/test_phi_projection_optimization_20260715.py tests/test_viz.py tests/test_reporting_additions.py tests/test_run_artifacts.py tests/test_round3_artifacts.py --junitxml=C:\tmp\vfe3-phi-reporting-green-20260715.xml`

Expected: zero failures and zero errors.

- [x] **Step 5: Commit**

```powershell
git add vfe3/run_artifacts.py vfe3/viz/figures.py tests/test_phi_projection_optimization_20260715.py tests/test_viz.py
git commit -m "feat: report phi projection route and timing"
```

### Task 6: Add and run the CUDA acceptance benchmark

**Files:**
- Create: `benchmarks/benchmark_phi_projection.py`
- Modify: `tests/test_phi_projection_optimization_20260715.py`
- Modify: `docs/2026-07-15-edits.md`

**Interfaces:**
- Click-to-run benchmark constants at module top; no CLI parser.
- JSON output records device, torch version, seed, K, heads, vocabulary, number of tables and rows, route, warmups, repeats, projected median and p95 milliseconds, disabled-control median and p95 milliseconds where measured, overhead ratio, and maximum post-projection norm.
- A small CPU smoke helper validates schema without enforcing timing.

- [x] **Step 1: Write a failing import/schema smoke test**

Import the benchmark module, run its tiny CPU projection-only case, and validate finite timings, exact route, row count, and post-bound correctness. Timing thresholds stay out of pytest.

- [x] **Step 2: Witness the missing benchmark module**

Run: `python -m pytest tests/test_phi_projection_optimization_20260715.py -k benchmark --junitxml=C:\tmp\vfe3-phi-benchmark-red-20260715.xml`

Expected: import failure for `benchmarks.benchmark_phi_projection`.

- [x] **Step 3: Implement the standalone benchmark**

Use seeded tensors and warmed CUDA events. The K240 route must allocate phi coordinates but never a vocabulary-by-240-by-240 embedded tensor. Preserve the observed pre-fix K20 evidence, `340.7574` versus `83.5599` milliseconds per step, as historical context rather than fabricating a new pre-fix run.

- [x] **Step 4: Run CPU smoke and inspect GPU availability/load**

Run: `python -m pytest tests/test_phi_projection_optimization_20260715.py -k benchmark --junitxml=C:\tmp\vfe3-phi-benchmark-green-20260715.xml`

Expected: zero failures and zero errors.

Before a CUDA run, inspect active GPU processes. If the user's training job is still active, do not contend with it; record the benchmark as deferred for resource isolation. Otherwise run the script and save JSON under `C:\tmp`, not the repository.

- [x] **Step 5: Evaluate acceptance evidence**

The CPU harness and full K20-shape sanity case passed. The CUDA acceptance measurement was deferred because `nvidia-smi` showed 87 percent GPU utilization and an active `C:\anaconda\python.exe` training process; contending with that process would perturb the user's run and invalidate timing evidence.

The K20 projected training-step median overhead target is at most 10 percent, with 5 percent preferred. K240 must complete without OOM, report `diagonal_gram`, and satisfy the configured radius. If a full training-step control is too expensive at K240, report projection-only K240 evidence honestly rather than substituting an invented ratio.

- [x] **Step 6: Update the dated edit record and commit**

```powershell
git add benchmarks/benchmark_phi_projection.py tests/test_phi_projection_optimization_20260715.py docs/2026-07-15-edits.md
git commit -m "bench: verify phi projection scaling"
```

### Task 7: Consolidated verification and repository lifecycle

**Files:**
- Modify: `docs/2026-07-15-edits.md`

- [ ] **Step 1: Run focused verification**

Run: `python -m pytest tests/test_phi_projection_optimization_20260715.py tests/test_phi_numerics_buildout_20260715.py tests/test_train.py tests/test_viz.py tests/test_run_artifacts.py tests/test_gauge_groups.py --junitxml=C:\tmp\vfe3-phi-projection-focused-final-20260715.xml`

Read `tests`, `failures`, `errors`, `skipped`, and `time` from JUnit before recording results.

- [ ] **Step 2: Run syntax and whitespace verification**

Run: `python -m compileall -q vfe3 tests benchmarks`

Run: `git diff --check origin/main...HEAD`

Expected: both exit zero.

- [ ] **Step 3: Run the full suite and compare with baseline**

Run: `python -m pytest --junitxml=C:\tmp\vfe3-phi-projection-full-final-20260715.xml`

Expected: no new failures. The preserved single-cell `sigma_max` mismatch may remain as the same sole failure. Record the exact machine-readable result and failure identity; do not call the suite fully passing if that failure remains.

- [ ] **Step 4: Review the complete diff and configuration invariants**

Run `git diff --stat origin/main...HEAD`, `git diff --check origin/main...HEAD`, and `git diff origin/main...HEAD -- ablation.py train_vfe3.py`. The final command must be empty.

- [ ] **Step 5: Complete the dated record and final implementation commit**

Record exact JUnit counts, benchmark results or the explicit resource blocker, files changed, and the preserved baseline failure in `docs/2026-07-15-edits.md`. Commit any final documentation-only change.

- [ ] **Step 6: Push, merge, and update the live checkout safely**

Push `codex/phi-projection-optimization-20260715`, fast-forward `main` to the verified task tip, push `origin/main`, fetch, and confirm the remote SHA. Fast-forward the user's live `main` checkout only if Git can do so without altering its pre-existing WIP. Never stash, restore, reset, clean, or rewrite the user's files.

- [ ] **Step 7: Clean task-owned artifacts and report final state**

Remove only task-owned JUnit and benchmark files from `C:\tmp`, remove the temporary worktree, delete the local task branch, and report the task commits, pushed branch, resulting `origin/main` SHA, machine-readable verification, safe live fast-forward result, and the live checkout's final `git status --short` with remaining files identified as user-owned.
