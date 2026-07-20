# Parameter-Matched Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in `ablation.py` sweep that selects one closest valid realized-parameter configuration per embedding width under a user-configurable target and tolerance.

**Architecture:** Extend the sweep registry with a third, parameter-grid shape. Pure helpers expand and validate the grid, while an exact CPU `VFEModel` construction supplies the realized count used for selection. The runner resolves a budget-specific output scope and attaches target/deviation metadata without changing ordinary sweep behavior.

**Tech Stack:** Python 3.14, PyTorch, `VFE3Config`, `VFEModel`, standard-library `unittest`, JSON/CSV run artifacts, and the repository verification ledger.

## Global Constraints

The target parameter count is an exact positive integer and is not hard-coded to 30 million.

The maximum relative deviation is a finite float in `[0, 1)` and defaults to `0.02`.

Selection retains at most one closest valid candidate per `embed_dim` and requires at least two retained widths.

Realized `sum(parameter.numel() for parameter in model.parameters())` is the acceptance count; the approximate scaling predictor is not an authority.

When `embed_dim` changes without an explicit `kl_max`, the generated override uses `8 * embed_dim`.

Ordinary single-field and multi-arm sweeps retain their current expansion, paths, contracts, resume semantics, and reporting.

No pytest command may be run. RED/GREEN checks use the standalone standard-library unittest file directly.

The task remains in the isolated `codex/parameter-matched-ablation-20260720` branch and must not touch the live checkout WIP.

---

### Task 1: Parameter-grid expansion and exact selector

**Files:**
- Modify: `ablation.py:540-585`
- Modify: `ablation.py:1695-1835`
- Create: `tests/test_parameter_matched_ablation_20260720.py`

**Interfaces:**
- Consumes: `BASELINE_CONFIG`, `_VFE3_FIELDS`, `_cell_cfg_dict`, `VFE3Config`, and `VFEModel`.
- Produces: `_validated_target_n_params(value: object) -> int`, `_validated_param_relative_deviation(value: object) -> float`, `_parameter_grid_overrides(sweep: Mapping[str, object]) -> List[Dict[str, Any]]`, `_realized_n_params_for_overrides(overrides: Mapping[str, object]) -> int`, and `_parameter_match_selection(sweep_name: str) -> Dict[str, Any]`.

- [ ] **Step 1: Write standalone failing selector checks**

Create a standard-library `unittest` module that imports `ablation`, patches `_realized_n_params_for_overrides` for deterministic selection cases, and directly exercises the wished-for interfaces. Include these concrete cases:

```python
class ParameterMatchedSelectionTests(unittest.TestCase):
    def test_grid_derives_kl_max_and_filters_invalid_pairs(self) -> None:
        sweep = {
            "description": "fixture",
            "match_by": "embed_dim",
            "parameter_grid": {"embed_dim": [20, 24], "n_heads": [3, 4]},
        }
        candidates = ablation._parameter_grid_overrides(sweep)
        self.assertIn({"embed_dim": 20, "n_heads": 4, "kl_max": 160}, candidates)
        self.assertIn({"embed_dim": 24, "n_heads": 3, "kl_max": 192}, candidates)

    def test_selector_keeps_one_closest_candidate_per_width(self) -> None:
        sweep_name = "parameter_fixture"
        sweep = {
            "description": "fixture",
            "match_by": "embed_dim",
            "parameter_grid": {"embed_dim": [20, 40], "n_heads": [2, 4]},
        }
        counts = {(20, 2): 98, (20, 4): 101, (40, 2): 105, (40, 4): 99}
        with patch.dict(ablation.SWEEPS, {sweep_name: sweep}), \
             patch.dict(ablation.CONFIG, {
                 "target_n_params": 100,
                 "max_param_relative_deviation": 0.05,
             }), \
             patch.object(
                 ablation,
                 "_realized_n_params_for_overrides",
                 side_effect=lambda ov: counts[(ov["embed_dim"], ov["n_heads"])],
             ):
            selected = ablation._parameter_match_selection(sweep_name)
        self.assertEqual(
            [(row["overrides"]["embed_dim"], row["overrides"]["n_heads"])
             for row in selected["selected"]],
            [(20, 4), (40, 4)],
        )
```

Add checks for exact target/tolerance validation, deterministic declared-order tie-breaking, fewer-than-two-width failure text, unknown/empty grid fields, and invalid configurations appearing in the rejected summary.

- [ ] **Step 2: Run the selector checks and confirm RED**

Run: `python tests\test_parameter_matched_ablation_20260720.py`

Expected: nonzero exit with `AttributeError` for `_parameter_grid_overrides` or `_parameter_match_selection`.

- [ ] **Step 3: Add the registry shape and selection helpers**

Import `itertools.product`. Register `SWEEPS["parameter_matched"]` with an `embed_dim`/`n_heads` grid broad enough to produce several feasible widths around common budgets, but keep it outside `SWEEP_ORDER`.

Implement strict target and tolerance validators. Expand the grid in declared dictionary/list order, merge `requires`, derive `kl_max`, and retain invalid cross-product combinations as rejected records rather than aborting the whole search. Validate `match_by == "embed_dim"` for this first implementation.

Implement exact counting as:

```python
def _realized_n_params_for_overrides(overrides: Mapping[str, object]) -> int:
    cfg = VFE3Config(**_cell_cfg_dict(dict(overrides), seed=0))
    model = VFEModel(cfg)
    try:
        return int(sum(parameter.numel() for parameter in model.parameters()))
    finally:
        del model
        gc.collect()
```

Implement selection records with `label`, `overrides`, `n_params`, signed `param_difference`, and absolute `param_relative_deviation`. Sort within each width by `(relative_deviation, candidate_index)`, retain only rows within tolerance, and raise a `ValueError` containing each width's closest rejected count/deviation when fewer than two widths remain.

- [ ] **Step 4: Run the standalone selector checks and confirm GREEN**

Run: `python tests\test_parameter_matched_ablation_20260720.py`

Expected: exit code 0 with all selector test cases reported as `ok`.

- [ ] **Step 5: Commit the selector increment**

Stage `ablation.py` and `tests/test_parameter_matched_ablation_20260720.py`, inspect the staged diff, and commit with `feat: select parameter-matched ablation cells`.

### Task 2: Budget-specific runner scope and persisted metadata

**Files:**
- Modify: `ablation.py:1660-1692`
- Modify: `ablation.py:2228-2250`
- Modify: `ablation.py:3326-3730`
- Modify: `ablation.py:4708-4829`
- Modify: `tests/test_parameter_matched_ablation_20260720.py`

**Interfaces:**
- Consumes: `_parameter_match_selection(sweep_name)` from Task 1 and existing `run_sweep`, `_write_sweep_csv`, and figure-worker seams.
- Produces: `_sweep_output_scope(sweep_name: str) -> str` and budget metadata in cell markers, CSV rows, and sweep metadata.

- [ ] **Step 1: Add failing output-scope and metadata checks**

Add direct unittest cases asserting:

```python
with patch.dict(ablation.CONFIG, {
    "target_n_params": 30_000_000,
    "max_param_relative_deviation": 0.02,
}):
    self.assertEqual(
        ablation._sweep_output_scope("parameter_matched"),
        "parameter_matched_N30000000_rtol0p02",
    )
self.assertEqual(ablation._sweep_output_scope("n_heads"), "n_heads")
```

Patch `_parameter_match_selection`, `run_single`, source/code identity helpers, checkpoint validators, cleanup, and figure launch boundaries to run a two-cell temporary-directory sweep without data or training. Assert that `sweep_meta.json`, both `ablation_result.json` markers, and `sweep_results.csv` carry the exact target, realized count, signed difference, and absolute relative deviation.

- [ ] **Step 2: Run the expanded checks and confirm RED**

Run: `python tests\test_parameter_matched_ablation_20260720.py`

Expected: nonzero exit because `_sweep_output_scope` and persisted budget fields do not exist.

- [ ] **Step 3: Integrate the selected cells into the runner**

Add `target_n_params`, `param_difference`, and `param_relative_deviation` to `_CSV_COLUMNS`. Resolve selection exactly once per parameter-matched `run_sweep` call and build a label-to-budget-record mapping. Attach the record to cached and newly computed results; if a successful cell's `run_single` count differs from the selection count, mark it failed with `error_kind="parameter_count_drift"` rather than publishing a mismatched row.

Add the complete parameter-match record to `report_metadata`, including target, tolerance, grid, selected rows, and rejected summary. Use `_sweep_output_scope` for creation, invalidation, completion metadata reads, analysis, and figure rendering in `main`. Ordinary sweep names must return byte-for-byte identical scopes.

Keep the exact target and normalized tolerance in the directory name so aggregation cannot cross budgets. Do not bump the existing cell or aggregation contract schemas; effective configuration remains cell-bound, and the budget-specific scope plus `sweep_meta.json` forms the invocation namespace.

- [ ] **Step 4: Run the standalone checks and confirm GREEN**

Run: `python tests\test_parameter_matched_ablation_20260720.py`

Expected: exit code 0 with selector, scope, and persistence cases reported as `ok`.

- [ ] **Step 5: Commit the runner increment**

Stage `ablation.py` and `tests/test_parameter_matched_ablation_20260720.py`, inspect the staged diff, and commit with `feat: persist parameter-matched ablation budgets`.

### Task 3: Direct mechanical verification and documentation closeout

**Files:**
- Modify: `docs/2026-07-20-edits.md`
- Create: `.verification/ledger.json` only if the verification tool does not treat it as ignored runtime state.

**Interfaces:**
- Consumes: final `ablation.py`, the standalone unittest module, and the approved design.
- Produces: a validated closure ledger and the final dated change record.

- [ ] **Step 1: Start the verification control plane at the final code revision**

Run the repository verification start command from the isolated worktree. Add claims for ordinary-sweep compatibility, exact realized counting, one-per-width tolerance enforcement, and artifact budget separation. Use current direct mechanical outputs as evidence; unresolved baseline pytest failures remain disclosed and do not become supporting evidence.

- [ ] **Step 2: Run direct final checks without pytest**

Run:

```powershell
python tests\test_parameter_matched_ablation_20260720.py
python -m py_compile ablation.py tests\test_parameter_matched_ablation_20260720.py
python -c "import ablation; print(ablation._sweep_output_scope('parameter_matched')); print(ablation.make_run_overrides('parameter_matched'))"
```

Expected: each command exits 0; the final command prints a budget-specific scope and at least two selected width configurations whose recorded deviations do not exceed `CONFIG["max_param_relative_deviation"]`.

- [ ] **Step 3: Validate the claim ledger**

Run the verification validator against `.verification/ledger.json` from the isolated worktree. Expected: exit code 0 and no open `CANDIDATE` or `LLM_SUPPORTED` claims.

- [ ] **Step 4: Update the dated edit record**

Append the implemented registry shape, exact counting behavior, selected default grid results, direct unittest count, compilation/import results, validated ledger path, and the explicit no-pytest verification boundary to `docs/2026-07-20-edits.md`. Do not claim a full pytest pass.

- [ ] **Step 5: Commit the verified implementation**

Stage every task-owned source, test, documentation, and ledger file; inspect `git diff --cached --check`, the complete staged diff, and `git status --short`; commit with `feat: add parameter-matched ablation sweep`.

- [ ] **Step 6: Complete the repository Git lifecycle**

Push `codex/parameter-matched-ablation-20260720`, merge it into `main`, push `main`, fetch and inspect `origin/main`, safely fast-forward the live checkout only if its WIP is not touched, remove the temporary worktree, delete the local task branch, and report all resulting SHAs and final statuses. Preserve every pre-existing live-checkout modification and deletion exactly.
