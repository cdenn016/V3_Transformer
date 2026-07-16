# CPU/CUDA Hybrid Test-Lanes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete the eleven approved slow UMAP tests, run the retained CPU suite across all twelve physical Ryzen 9 9900X cores where useful, and add a serial RTX 5090 device matrix without replacing CPU coverage.

**Architecture:** A click-to-run CPU driver launches isolated xdist lanes with explicit worker counts and one native thread per worker. One curated pytest policy separates CUDA-only tests from ordinary CPU contracts mirrored onto CUDA only in the CUDA process. Native UMAP integration coverage is intentionally removed.

**Tech Stack:** Python 3.10+, pytest, pytest-xdist, pytest-cov, PyTorch, PowerShell, JUnit XML.

## Global Constraints

- Delete exactly the eleven node IDs named in the approved design and do not replace their native UMAP, report, finalization, reload, or cleanup assertions.
- Keep production UMAP support, the `umap-learn` visualization dependency, mocked worker protocol tests, and pure report-planning tests.
- Keep CPU goldens and host-side contracts mandatory; CUDA adds a matrix and never replaces the CPU lane.
- Use twelve explicit xdist workers for the fast CPU lane and at most three for the retained three-node slow lane; never use `-n auto`.
- Set every supported native numerical-library thread variable to one in child processes before imports.
- Run CUDA serially with deterministic algorithms and TF32 disabled; do not run CUDA verification while another training process owns the RTX 5090.
- Seed numerical matrix inputs on CPU and move the same samples to the selected device.
- Obtain every pass, skip, failure, error, coverage, and timing claim from fresh machine-readable output.

### Task 1: Remove the slow UMAP cohort

**Files:**
- Modify: `tests/pytest_policy.py`
- Modify: `tests/test_pytest_policy.py`
- Modify: `tests/test_viz.py`
- Modify: `tests/test_july13_root_fixes.py`
- Modify: `tests/test_round3_artifacts.py`
- Modify: `tests/test_report.py`
- Modify: `tests/test_run_artifacts.py`
- Modify: `check_audit_fixes.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: a three-node `SLOW_TESTS` set and CUDA-only resource groups; no UMAP policy table or marker remains.

- [ ] **Step 1: Write the failing policy expectation.** Change `tests/test_pytest_policy.py` to require no UMAP table or marker behavior, require representative slow items to receive only `slow` plus the optional skip, require `RESOURCE_GROUPS` to contain only CUDA nodes, and require exactly the three retained slow node IDs.
- [ ] **Step 2: Run `python -m pytest tests/test_pytest_policy.py --junitxml=C:\tmp\vfe3-hybrid-task1-red.xml` and confirm failure against the existing eleven-node UMAP policy.**
- [ ] **Step 3: Remove the eleven test functions named in the design.** Remove only helpers, imports, the finalized evidence record, and the finalized evidence fixture proven unused after those deletions.
- [ ] **Step 4: Remove UMAP policy and stale script references.** Delete `UMAP_TESTS`, UMAP resource grouping, marker assignment, marker registration, help text, and the deleted `m26` audit-runner node. Keep the optional production dependency.
- [ ] **Step 5: Run the policy module and every edited surviving test module with JUnit XML.** Confirm the deleted symbols are absent by source search and the remaining modules collect and pass.
- [ ] **Step 6: Commit with `test: remove slow UMAP integration cohort`.**

### Task 2: Add the twelve-core CPU runner

**Files:**
- Create: `run_cpu_tests.py`
- Create: `tests/test_run_cpu_tests.py`
- Modify: `docs/testing/test-lanes.md`

**Interfaces:**
- Produces: `resolve_cpu_workers`, `build_cpu_environment`, `build_cpu_lane_command`, `run_lane`, and `main`.
- Configuration: `FAST_WORKERS = 12`, `SLOW_WORKERS = 3`, and `RUN_LANES = ("fast", "slow")`.

```python
FAST_WORKERS = 12
SLOW_WORKERS = 3
RUN_LANES = ("fast", "slow")

CPU_ENVIRONMENT = {
    "VFE3_TEST_DEVICE": "cpu",
    "CUDA_VISIBLE_DEVICES": "-1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
```

- [ ] **Step 1: Write failing unit tests.** Assert that worker validation accepts 12 on a 24-logical-CPU host, rejects bool/zero/negative/excess values, fast commands contain explicit `-n 12 --dist loadscope`, slow commands contain `--runslow -n 3`, neither contains `auto`, and every child thread variable is one without mutating the parent mapping.
- [ ] **Step 2: Add a failing driver test.** Inject subprocess and JUnit readers, require one fresh child per lane, require immediate stop after a nonzero exit, and require the exact child environment.
- [ ] **Step 3: Run `python -m pytest tests/test_run_cpu_tests.py --junitxml=C:\tmp\vfe3-hybrid-task2-red.xml` and confirm import failure because the runner does not exist.**
- [ ] **Step 4: Implement the click-to-run driver with standard-library subprocesses and temporary JUnit paths.** Do not parse CLI arguments and do not mutate `os.environ`.
- [ ] **Step 5: Run `tests/test_run_cpu_tests.py` and `tests/test_pytest_policy.py` with JUnit XML.**
- [ ] **Step 6: Update `docs/testing/test-lanes.md` so the driver is the ordinary CPU entry point and manual commands use explicit 12/3 worker counts.**
- [ ] **Step 7: Commit with `test: use all physical CPU cores`.**

### Task 3: Canonicalize CUDA selection and mirror policy

**Files:**
- Modify: `tests/pytest_policy.py`
- Modify: `tests/test_pytest_policy.py`
- Modify: `tests/conftest.py`
- Modify: `check_gpu_tests.py`

**Interfaces:**
- Produces: the existing six-node `CUDA_TESTS` table plus a curated `CUDA_MIRROR_TESTS` table in one policy module.
- Marker behavior: CUDA-only node IDs always receive `cuda`; mirror nodes receive `cuda` and the CUDA group only when the requested test device has type `cuda`; mirror nodes remain ordinary CPU items otherwise.

```python
CUDA_MIRROR_TESTS: frozenset[str] = frozenset({
    "test_tier12_transport.py::test_per_head_transport_mean_matches_dense",
    "test_tier12_transport.py::test_per_head_transport_mean_rope_wrapped_matches_dense",
    "test_tier12_transport.py::test_stable_exp_norm_mode_small_norm_takes_fp32_path_exactly",
    "test_tier12_transport.py::test_stable_exp_norm_mode_large_norm_reenters_fp64_island",
    "test_tier12_estep.py::test_mm_exact_stationarity_folds_twohop",
    "test_tier12_estep.py::test_mm_exact_monotone_filtered_f_descent",
    "test_tier12_estep.py::test_twohop_zero_is_byte_identical",
    "test_tier12_estep.py::test_backprop_last_truncates_transport_gradient_to_phi",
    "test_omega_tilde_model_frame.py::test_phi_tilde_mm_exact_device_smoke",
    "test_tier12_attention.py::test_query_adaptive_tau_monotone_detached_and_c0_inert",
    "test_tier12_attention.py::test_twohop_term_matches_hand_computation",
    "test_tier12_decode.py::test_expected_likelihood_decode_matches_naive_dense",
    "test_tier12_decode.py::test_z_loss_full_chunked_matches_dense_lse",
    "test_divergence.py::test_safe_kl_clamp_bounds_and_nan",
    "test_free_energy.py::test_free_energy_entropy_exact_for_deep_finite_prior",
    "test_retraction.py::test_full_retraction_stays_spd",
})
```

- [ ] **Step 1: Write failing policy tests.** Require a mirror node to remain unmarked under `cpu`, require the same node to receive `cuda` plus the CUDA group under `cuda:0`, and require CUDA-only nodes to receive `cuda` under either environment.
- [ ] **Step 2: Write a failing canonical-runner test.** Require `check_gpu_tests.py` to invoke `-m cuda` and forbid a private literal node list.
- [ ] **Step 3: Run the new policy/runner tests and confirm the missing mirror-policy symbol or marker assertions fail.**
- [ ] **Step 4: Implement `CUDA_MIRROR_TESTS` and device-dependent collection policy.** Keep the six existing CUDA-only nodes unchanged and add the sixteen exact mirror nodes defined by the design domains.
- [ ] **Step 5: Configure the CUDA test process before collection.** Set cuBLAS workspace configuration before importing Torch, enable deterministic algorithms, disable cuDNN benchmarking, disable CUDA/cuDNN TF32 only for CUDA requests, and snapshot/restore global state.
- [ ] **Step 6: Resolve the shared fixture by `torch.device(name).type` so `cuda:0` works, make `check_gpu_tests.py` select `-m cuda` serially, and delete its stale private t6 list.**
- [ ] **Step 7: Run the policy/runner tests with JUnit XML.**
- [ ] **Step 8: Commit with `test: add canonical CUDA mirror lane`.**

### Task 4: Add representative numerical contracts to the matrix

**Files:**
- Modify: `tests/test_divergence.py`
- Modify: `tests/test_free_energy.py`
- Modify: `tests/test_retraction.py`

**Interfaces:**
- Converts: `test_safe_kl_clamp_bounds_and_nan`, `test_free_energy_entropy_exact_for_deep_finite_prior`, and `test_full_retraction_stays_spd` to the shared `device` fixture.

- [ ] **Step 1: Extend the policy meta-test to require the three exact nodes in `CUDA_MIRROR_TESTS`; run it and confirm failure before the mirror table exists.**
- [ ] **Step 2: Convert the three tests.** Construct deterministic source inputs on CPU, move inputs and expected tensors to `device`, preserve exact comparisons and existing tolerances, and introduce no global tolerance changes.
- [ ] **Step 3: Run the three CPU nodes and their complete source modules with JUnit XML.**
- [ ] **Step 4: Run the CUDA marker collection statically and confirm the mirror includes attention, decode, E-step, transport, MM, model-frame, divergence, free-energy, and retraction coverage without importing test modules in the meta-test.**
- [ ] **Step 5: Commit with `test: matrix core numerics across devices`.**

### Task 5: Documentation, verification, and closeout

**Files:**
- Modify: `docs/testing/test-lanes.md`
- Create or modify: `docs/2026-07-16-edits.md`

**Interfaces:**
- Produces: machine-readable retained CPU, coverage, CUDA, and external-prerequisite evidence with the eleven intentional deletions stated plainly.

- [ ] **Step 1: Run focused policy, runner, and edited-module verification and parse JUnit counts.**
- [ ] **Step 2: Run the click-to-run twelve-core fast CPU lane and three-worker slow lane.** Compare node counts and elapsed time with the 2,915-node/41.413-second fast baseline and 14-node/138.774-second slow baseline.
- [ ] **Step 3: Run twelve-core branch coverage over the retained CPU union.** Record the exact line/branch delta and attribute no loss beyond the removed nodes without evidence.
- [ ] **Step 4: When the RTX 5090 is idle, run the serial `-m cuda` lane with the CUDA-enabled interpreter and require zero skips, failures, and errors.** If training remains active, report CUDA verification as blocked rather than running competitively.
- [ ] **Step 5: Run the external node only when both bundle paths exist; otherwise record the prerequisite skip.**
- [ ] **Step 6: Update the dated edit record with exact commands, XML counts, timings, coverage, hardware, and intentional coverage loss.** Run `python -m compileall -q` on changed Python paths and `git diff --check`.
- [ ] **Step 7: Dispatch task and whole-branch review, resolve all substantive findings, and rerun affected verification.**
- [ ] **Step 8: Inspect status and staged diff, commit every intended file, push the task branch, merge to `main`, push `main`, fetch and verify `origin/main`, safely fast-forward the live checkout only if its WIP permits, and remove task-owned artifacts/worktree/local branch.**
