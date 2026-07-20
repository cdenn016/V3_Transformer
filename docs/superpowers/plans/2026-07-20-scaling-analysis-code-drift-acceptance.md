# Scaling Analysis Code-Drift Acceptance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Include all nine current blocksize runs through a narrow, auditable code-identity override and correct distinct-size reporting without weakening other artifact-integrity checks.

**Architecture:** Extend the existing schema-2 run validator with one keyword-only policy switch and retain both the cell-bound and observed code identities on accepted rows. Let requested-design completion override only the exact terminal code-drift condition when every declared cell otherwise joins successfully, then propagate forced-acceptance metadata into CSV, JSON, Markdown, console warnings, and fit confounds.

**Tech Stack:** Python 3.14, pytest, NumPy, JSON/CSV artifacts, existing click-to-run module configuration.

## Global Constraints

- `force_accept_code_identity_drift` is enabled in the checked-in analysis `CONFIG` for the requested current blocksize analysis.
- The override may bypass only `git_dirty` or `git_dirty_fingerprint` disagreement when Git SHA agrees and both identities are structurally valid.
- Dataset/source identities, schema, reuse digest, configuration digest, seed, and metric checks remain fail-closed.
- Scaling generation, caching, resume logic, and training behavior remain unchanged.
- Forced rows and their paired identities remain visible in persisted artifacts and human-readable output.
- Tests follow red-green-refactor, and reported counts come from JUnit XML.

---

### Task 1: Narrow run-level code-identity override

**Files:**
- Modify: `tests/test_2026_07_15_driver_reliability_remediation.py`
- Modify: `scaling_analysis.py:91-121,189-382`

**Interfaces:**
- Consumes: schema-2 `scaling_cell.json`, `summary.json`, `config.json`, and `provenance.json`.
- Produces: `_validated_bound_scaling_run(..., *, force_accept_code_identity_drift: bool = False) -> Optional[Dict[str, object]]` and `harvest(input_dir: Path, *, force_accept_code_identity_drift: bool = False) -> List[Dict[str, Any]]`.

- [ ] **Step 1: Write failing strict/forced validator tests**

Create otherwise-valid synthetic runs, change only `provenance.json["git_dirty_fingerprint"]`, and assert strict harvest returns no row while forced harvest returns one row with `code_identity_forced is True`, `cell_git_dirty_fingerprint`, and `provenance_git_dirty_fingerprint`. Parametrize Git-SHA, contract-digest, source, seed, and metric corruption and assert forced harvest still rejects each case.

- [ ] **Step 2: Run the new tests and verify RED**

Run `python -m pytest tests/test_2026_07_15_driver_reliability_remediation.py -k "force_accept_code_identity" --junitxml=C:\tmp\vfe3-scaling-force-red.xml`.

Expected: failures because the keyword arguments and audit fields do not exist.

- [ ] **Step 3: Implement the minimal run-level policy**

Add the checked-in setting:

```python
CONFIG: Dict[str, Any] = {
    "input_dir": "vfe3_scaling_results",
    "with_offset": True,
    "n_bootstrap": 2000,
    "min_points": 2,
    "force_accept_code_identity_drift": True,
}
```

Validate both code-identity mappings structurally and require equal nonempty Git SHAs. In strict mode require exact `git_dirty` and fingerprint equality. In forced mode permit only those two fields to disagree and return both identities plus `code_identity_forced=True`. Pass the keyword from `harvest` and add stable audit columns to `_CSV_COLUMNS`.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Step 2 command again. Read the JUnit `tests`, `failures`, and `errors` attributes and require zero failures/errors.

- [ ] **Step 5: Commit the run-level validator**

```powershell
git add scaling_analysis.py tests/test_2026_07_15_driver_reliability_remediation.py
git commit -m "feat: narrowly accept scaling code identity drift"
```

### Task 2: Effective forced completion and honest reporting

**Files:**
- Modify: `tests/test_2026_07_15_driver_reliability_remediation.py`
- Modify: `scaling_analysis.py:385-584,612-675,889-1192`

**Interfaces:**
- Consumes: forced-row metadata from Task 1 and a schema-1 `scaling_design.json`.
- Produces: `_requested_design(..., *, force_accept_code_identity_drift: bool = False)` with `forced_code_identity_acceptance`, `forced_row_count`, and original manifest status/error; summary fields `n_harvested_param_sizes` and `n_fit_param_sizes`.

- [ ] **Step 1: Write failing design and message tests**

Assert that an incomplete manifest whose exact error is `code identity drifted during the scaling invocation`, whose declared cells are complete, and whose rows all join becomes `complete=True` only in forced mode. Assert unrelated errors remain incomplete. Capture stdout for incomplete strict mode and require `2 harvested parameter sizes; fitting withheld (incomplete_design)` rather than `only 0 distinct parameter size`.

- [ ] **Step 2: Run the new tests and verify RED**

Run `python -m pytest tests/test_2026_07_15_driver_reliability_remediation.py -k "forced_code_identity_design or reports_harvested_sizes" --junitxml=C:\tmp\vfe3-scaling-report-red.xml`.

Expected: failures because effective forced completion and separate size counts do not exist.

- [ ] **Step 3: Implement effective completion and reporting**

Pass `CONFIG["force_accept_code_identity_drift"]` through `analyze()` to `harvest()` and `_requested_design()`. Permit effective completion only when the manifest is schema-valid, every declared cell joins as complete, the top status is `incomplete`, the exact manifest error is code-identity drift, and at least one accepted row is forced. Record forced row identities in the design/summary, add `code_identity_forced` to provenance and pooled confounds, print a warning before fitting, and render the warning in `SCALING_ANALYSIS.md`.

Compute both counts:

```python
harvested_param_sizes = {float(point["n_params"]) for point in param_points}
fit_param_sizes = {float(point["n_params"]) for point in fit_param_points}
```

When fitting is withheld, report `len(harvested_param_sizes)` and the withholding reason. Preserve `n_distinct_param_sizes` as the fit count for compatibility and add explicit harvested/fit fields.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Step 2 command again and require zero JUnit failures/errors.

- [ ] **Step 5: Commit effective completion and reporting**

```powershell
git add scaling_analysis.py tests/test_2026_07_15_driver_reliability_remediation.py
git commit -m "fix: report forced scaling cohorts honestly"
```

### Task 3: Current artifact reproduction, documentation, and closure

**Files:**
- Modify: `docs/2026-07-20-edits.md`
- Create: `.verification/ledger.json` through the installed verification control plane (ignored task metadata)

**Interfaces:**
- Consumes: the completed implementation and the live checkout's `vfe3_scaling_results` after merge.
- Produces: focused/full JUnit records, validated claim ledger, regenerated nine-run scaling analysis, and completed Git lifecycle.

- [ ] **Step 1: Run focused regression verification**

Run `python -m pytest tests/test_2026_07_15_driver_reliability_remediation.py tests/test_scaling_mup.py --junitxml=C:\tmp\vfe3-scaling-force-focused.xml` with no extra `-q`. Read the XML totals and require zero failures/errors.

- [ ] **Step 2: Run the full suite**

Run `python -m pytest --junitxml=C:\tmp\vfe3-scaling-force-full.xml` with no extra `-q`. Read machine-readable totals and compare any failures to the current baseline; do not claim closure if new failures remain.

- [ ] **Step 3: Update the dated edit record**

Record the narrow override, retained fail-closed boundaries, distinct-size correction, exact focused/full JUnit totals, and current-artifact reproduction outcome in `docs/2026-07-20-edits.md`.

- [ ] **Step 4: Validate the evidence ledger**

Start/update `.verification/ledger.json`, record one claim per implementation and experiment-output check, and run the installed `verification_gate.py validate .verification/ledger.json --cwd .` until it exits zero.

- [ ] **Step 5: Commit and publish**

Inspect `git status --short`, `git diff --cached`, and `git diff --check`; commit all intended tracked changes, push `codex/fix-scaling-analysis-force-20260720`, fast-forward or merge it into `main`, push `main`, and fetch to verify `origin/main`.

- [ ] **Step 6: Safely update the live checkout and regenerate analysis**

Fast-forward the live `main` only if it preserves all pre-existing WIP. Run the merged `scaling_analysis.py` against the live current results, then verify `scaling_summary.json` reports nine runs, three parameter sizes, forced code-identity acceptance, and no missing declared cells. Do not stage or commit generated ignored run artifacts.

- [ ] **Step 7: Clean up the isolated worktree**

After remote/local SHA verification, remove the temporary worktree and local task branch. Report final `git status --short`, including ownership of remaining live WIP.

