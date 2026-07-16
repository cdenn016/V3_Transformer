# Test-Suite Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce full-suite elapsed time by eliminating repeated expensive setup and enabling safe CPU process parallelism while retaining the complete semantic coverage union.

**Architecture:** Explicit pytest markers define the lane union. Pure report planning replaces expensive renders in routing-only tests, while narrow module fixtures reuse immutable artifact trees. Xdist and branch coverage are opt-in verification tools; CUDA, real UMAP, and external-bundle cases remain serial.

**Tech Stack:** Python 3.10+, pytest, pytest-xdist, pytest-cov, coverage.py, PyTorch, matplotlib.

## Global Constraints

- Preserve every independent finite-difference, oracle, golden, gauge, Regime-II, two-hop, retraction, and learnability contract named in the design.
- Do not reduce seeds, step counts, thresholds, tolerances, dtypes, exact equality checks, or negative controls.
- Keep mutable models, registries, RNG state, and artifact writers function-local except for narrowly controlled module fixture setup.
- Keep CUDA, real UMAP, and external-bundle execution serial.
- Use fixed xdist worker counts; do not add `-n auto` to repository defaults.
- Obtain pass counts only from JUnit XML or another machine-readable result.

### Task 1: Explicit marker semantics and registry isolation

**Files:**
- Modify: `tests/conftest.py`
- Create: `tests/test_test_harness.py`
- Modify: the test modules containing the 18 current slow nodes and the six dedicated CUDA nodes
- Modify: `tests/test_alpha_i.py`
- Modify: `tests/test_attention_prior.py`
- Modify: `tests/test_config.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: pytest item markers and the existing `--runslow` option.
- Produces: intrinsic `slow`, `serial`, `cuda`, and `external_bundle` marks; a collection hook that only conditionally skips `slow` items.

- [ ] **Step 1: Write the failing hook regression.** Add fake config and item objects to `tests/test_test_harness.py`. The `--runslow=True` case must begin with an intrinsic slow marker and assert that the hook leaves `slow` present without adding `skip`; the default case must assert that `skip` is added.

```python
def test_runslow_preserves_intrinsic_slow_marker_without_skip():
    item = _Item("tests/test_probe.py::test_probe", markers=(pytest.mark.slow,))
    pytest_collection_modifyitems(_Config(runslow=True), [item])
    assert item.marker_names == ["slow"]
```

- [ ] **Step 2: Run the new hook regression and confirm it fails against the allowlist hook.** Run `python -m pytest tests/test_test_harness.py --junitxml=artifacts/test-harness-red.xml`. Expected result: failure because the existing hook does not consume intrinsic slow markers as the source of truth.
- [ ] **Step 3: Replace `_SLOW_TESTS` with intrinsic decorators.** Add `@pytest.mark.slow` to each current slow node. Add `@pytest.mark.serial` to real UMAP process tests, all dedicated CUDA tests, and the external-bundle case; also add `cuda` or `external_bundle` as applicable. Change the hook to:

```python
def pytest_collection_modifyitems(config, items):
    run_slow = config.getoption("--runslow")
    skip_slow = pytest.mark.skip(reason="slow integration test; pass --runslow to run it")
    for item in items:
        if item.get_closest_marker("slow") is not None and not run_slow:
            item.add_marker(skip_slow)
```

- [ ] **Step 4: Register markers and strict validation.** Set `addopts = "-q --strict-markers"` and define the four marker descriptions under `[tool.pytest.ini_options]`.
- [ ] **Step 5: Restore registry mutations.** Capture the previous entry with a sentinel, perform the registration, and restore or delete the key in `finally` so worker order cannot leak state.
- [ ] **Step 6: Run the harness and registry modules.** Run `python -m pytest tests/test_test_harness.py tests/test_alpha_i.py tests/test_attention_prior.py tests/test_config.py --junitxml=artifacts/task1.xml` and read the XML counts.
- [ ] **Step 7: Commit.** Commit with `test: make execution lanes explicit`.

### Task 2: Pure report planning and routing tests

**Files:**
- Modify: `vfe3/viz/report.py`
- Modify: `tests/test_report.py`
- Modify: `tests/test_model_channel_diagnostics_2026_06_13.py`

**Interfaces:**
- Produces: `plan_single_run_figures(dataset: str, availability: Mapping[str, bool]) -> tuple[str, ...]`.
- The tuple contains `<figure>.png` names in mapping order after input and language-route filtering.

- [ ] **Step 1: Write failing plan tests.** Replace the Japanese full-render routing test with a pure call using true availability for `belief_category_separation`, `vocab_confusion`, `vocab_probability_heatmap`, `vocab_calibration`, and `decode_readout`. Add active/off availability cases for `s_channel_refinement`, `model_channel_belief`, `hyper_prior_centroid`, and `hyper_prior_coupling`.

```python
planned = set(plan_single_run_figures("wiki-ja", availability))
assert "belief_category_separation.png" not in planned
assert "vocab_confusion.png" not in planned
assert {"vocab_probability_heatmap.png", "vocab_calibration.png", "decode_readout.png"} <= planned
```

- [ ] **Step 2: Run those exact nodes and confirm import or attribute failure.** Record the failing JUnit XML.
- [ ] **Step 3: Implement the pure planner.** Define the two English-only figure names once, filter only unavailable or route-ineligible names, and return a tuple. Do not call a model, loader, renderer, UMAP worker, or filesystem API.
- [ ] **Step 4: Make `generate_figures` consume the plan.** Build the availability mapping from the existing extractor results and use membership in the planned set for fixed figure emissions. Preserve the current thunks, best-effort exception handling, and dynamic UMAP channel loops.
- [ ] **Step 5: Retain integration boundaries.** Keep the live-model render, reload render, finalize autorun, and real UMAP lifecycle tests marked slow/serial as designed.
- [ ] **Step 6: Run `tests/test_report.py` and `tests/test_model_channel_diagnostics_2026_06_13.py` with `--runslow`, save JUnit XML, and read counts.**
- [ ] **Step 7: Commit.** Commit with `perf: separate report routing from rendering`.

### Task 3: Narrow immutable artifact fixtures

**Files:**
- Modify: `tests/test_run_artifacts.py`
- Modify: `tests/test_model_channel_diagnostics_2026_06_13.py`

**Interfaces:**
- Produces: frozen module-local records containing run directories and read-only result mappings.
- Fixture setup may retain private model/config/artifact references only so a dependent finalization fixture can run once; test functions receive paths/results and do not mutate those references.

- [ ] **Step 1: Add call-count regressions around fixture setup.** Use module-scoped fixtures backed by `tmp_path_factory`; separate tests continue to assert their original files and keys.
- [ ] **Step 2: Convert the default artifact pair.** One trained run supplies `metrics.csv`, `best_model.pt`, checkpoints, and the two gauge-geometry CSV columns.
- [ ] **Step 3: Convert the slow finalization trio.** One dependent finalized run supplies `test_results.json`, `summary.json`, loss/validation plots, `holonomy.png`, and the `reloaded_best` result while retaining three test nodes.
- [ ] **Step 4: Convert active/off model-channel setup.** One active and one pure-path training setup supplies both CSV-column and attention-file assertions. A dependent fixture finalizes each route once for the root figure presence/absence contract.
- [ ] **Step 5: Run the two affected modules with `--runslow` and machine-readable XML.** Confirm no assertion migrated behind the slow gate unless it was already slow.
- [ ] **Step 6: Commit.** Commit with `perf: reuse immutable test artifacts`.

### Task 4: Low-risk duplicate consolidation

**Files:**
- Modify: `tests/test_retraction.py`
- Modify: `tests/test_ultradeep_fixes_2026_06_13.py`
- Modify: `tests/test_numerics.py`
- Modify: `tests/test_audit_fixes_2026_06_14.py`

**Interfaces:**
- Retains: the domain-owned retraction zero-tangent finite-backward node.
- Produces: `test_safe_spd_inverse_matches_linalg_inverse[ridge]` with explicit ridge IDs for both previous regimes.

- [ ] **Step 1: Run the two retraction nodes before deletion and record that both pass.** Apply a temporary local mutation that breaks zero-tangent gradient propagation, rerun both nodes, and require both to fail for the same missing-gradient defect; restore the production file immediately and verify both pass again.
- [ ] **Step 2: Delete only the dated duplicate retraction node.** Keep formula, finite-difference, degeneracy, and domain-owned zero-tangent coverage unchanged.
- [ ] **Step 3: Parameterize the well-conditioned inverse contract.** Use `@pytest.mark.parametrize("ridge", [pytest.param(0.0, id="zero"), pytest.param(1e-7, id="regularized")])` and remove only the redundant dated body.
- [ ] **Step 4: Run the four affected modules and inspect JUnit counts.**
- [ ] **Step 5: Commit.** Commit with `test: consolidate proven duplicate contracts`.

### Task 5: Parallel and branch-coverage tooling

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `docs/testing/test-lanes.md`
- Modify: `docs/2026-07-15-edits.md`

**Interfaces:**
- Adds: `pytest-xdist` and `pytest-cov` to the `dev` extra.
- Coverage configuration: `branch = true`, `source = ["vfe3"]`, `parallel = true`, `show_missing = true`.

- [ ] **Step 1: Add dependencies and coverage configuration.** Regenerate `uv.lock` with the repository's `uv` workflow.
- [ ] **Step 2: Document the semantic lane union.** Include the serial default command, fixed two/four-worker CPU commands excluding `slow`, `serial`, and `cuda`, the serial slow/UMAP command, the RTX 5090 command, and the two-environment-variable external-bundle command. State that absent prerequisites are skips, not passes.
- [ ] **Step 3: Document branch-coverage and JUnit commands.** Keep generated XML and coverage files task-owned and outside the final commit.
- [ ] **Step 4: Update the dated edit document with the implemented changes and verification evidence.**
- [ ] **Step 5: Commit.** Commit with `build: add parallel coverage test tooling`.

### Task 6: Verification, timing selection, and closeout

**Files:**
- Modify: `docs/2026-07-15-edits.md`

**Interfaces:**
- Produces: machine-readable serial, slow-inclusive, coverage, CUDA/external eligibility, and xdist timing evidence.

- [ ] **Step 1: Capture collection identities.** Compare pre-change and post-change node lists, accounting only for the approved duplicate and parameter-ID changes.
- [ ] **Step 2: Run the serial default suite with JUnit XML and `--durations=100`.** Read `tests`, `failures`, `errors`, `skipped`, and wall time from the artifact.
- [ ] **Step 3: Run the serial slow-inclusive suite with JUnit XML.** Report CUDA and external-bundle skips separately.
- [ ] **Step 4: Run branch coverage and record the total.** Compare against the baseline established on the rebased source.
- [ ] **Step 5: Run fixed two- and four-worker CPU trials with identical marker selection and node sets.** Select the faster repeatable fixed count; do not change default `addopts` to enable xdist.
- [ ] **Step 6: Run the dedicated CUDA lane on the RTX 5090 if CUDA is available.** Run the external-bundle node only if both supplied paths exist; otherwise report the exact prerequisite blocker.
- [ ] **Step 7: Dispatch per-task and whole-branch reviews, resolve all substantive findings, and rerun affected verification.**
- [ ] **Step 8: Update the dated edit document, inspect `git diff --check`, status, staged diff, and commit. Push the task branch, merge to `main`, push `main`, preserve dirty live WIP, and clean up temporary worktrees only after remote verification.**
