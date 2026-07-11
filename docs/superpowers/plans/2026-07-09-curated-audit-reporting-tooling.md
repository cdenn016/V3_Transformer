# Curated Audit Reporting and Tooling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make scaling estimates, run reports, diagnostic figures, statistical summaries, helper scripts, and public metadata accurate and reproducible without altering experiment choices.

**Architecture:** Reporting consumes explicit runtime contracts from the other subsystem plans. Estimator configuration, provenance flags, purity predicates, figure memory gates, and statistical degeneracy are represented in saved schemas rather than transient warnings.

**Tech Stack:** Python, NumPy/SciPy, PyTorch, matplotlib, pytest, JUnit XML, JSON/CSV/Markdown artifacts.

## Global Constraints

- Follow the master plan and approved design; do not change route grids, live config values, or defaults.
- Scope owner: Findings 2-3, 5, 12-15, 24, 28-29, 31-32, 37-38, 48, 60, 88-89, and 101-105, 107. Finding 14's pullback-cache addendum is implemented by the core plan; this plan closes the full row after its direct-link work.
- Finding 30 is closed with the state/data generation task. Finding 91 is closed with the omega ablation task.
- Labels must distinguish exact mathematics, fixed-prior surrogates, and diagnostics.
- Saved reports must retain every warning needed to interpret a fit after console output is gone.
- No extra pytest `-q`; helper scripts read counts from JUnit.

---

## File Structure

- Modify `scaling_analysis.py`, `vfe3/viz/figures.py`: one scaling estimator and saved provenance/confound state.
- Modify `vfe3/run_artifacts.py`, `vfe3/geometry/groups.py`, `vfe3/geometry/transport.py`: cost and pure-path truth.
- Modify `vfe3/model/model.py`, `vfe3/train.py`, `vfe3/viz/extract.py`: reusable diagnostic snapshots and figure gates.
- Modify `vfe3/viz/report.py`, `make_figures.py`, `compare_vocab_figures.py`: config migration, memory guards, and figure cleanup.
- Modify `vfe3/metrics.py`, `vfe3/numerics.py`: average ranks, zero-token handling, and explicit matrix kind.
- Create `check_junit.py`; modify `check_audit_fixes.py`, `check_gpu_tests.py`.
- Create `tests/test_curated_audit_reporting_20260709.py`; extend reporting, metrics, config, transport, and visualization tests.

### Task 1: One scaling estimator and persistent confound metadata

**Files:** Modify `scaling_analysis.py`, `vfe3/viz/figures.py`; modify `tests/test_reporting_additions.py`, `tests/test_viz.py`.

**Interfaces:** `plot_scaling_routes(..., *, with_offset: bool, weights_by_route: Optional[Mapping[str, ndarray]] = None)`; analysis summary includes `provenance` and `pooled_fit_status`.

- [ ] **Step 1: Add failing tests** `test_all_scaling_tables_and_overlays_share_estimator`, `test_offset_fit_requires_four_distinct_sizes`, `test_scaling_summary_persists_code_and_data_drift`, and `test_divergent_routes_mark_pooled_fit_confounded`.
- [ ] **Step 2: Run** reporting/viz tests; expect different exponents and absent metadata.
- [ ] **Step 3: Implement.** Compute SEM weights once and pass them plus `CONFIG["with_offset"]` to headline, per-route, and overlay fits. In `_fit_power_law`, enter the three-parameter branch only for four distinct sizes; otherwise return log-log with `form="power_law_fallback_underdetermined"`. Persist SHA sets, token-budget variation, ANCOVA result, and `pooled_fit_status` in JSON/Markdown. Keep the pooled value visible but label it confounded when routes diverge.
- [ ] **Step 4: Relabel Finding 15.** Call the tied route a structural ablation in report text; do not alter its family, decode mode, or grid.
- [ ] **Step 5: Run** `python -m pytest tests/test_reporting_additions.py tests/test_viz.py`; expect PASS.
- [ ] **Step 6: Commit** `fix(scaling): unify estimators and persist confounds`.

### Task 2: Correct cost accounting and pure-path predicates

**Files:** Modify `vfe3/run_artifacts.py`, `vfe3/geometry/groups.py`; modify `tests/test_reporting_additions.py`, `tests/test_run_artifacts.py`.

**Interfaces:** `_cost_model_fields` distinguishes linear and prior-bank reads and includes s-channel E-step work. `_pure_path_report` derives family/group invariance from registry metadata.

- [ ] **Step 1: Add failing tests** `test_cost_model_linear_decode_does_not_count_prior_bank_readout`, `test_cost_model_counts_s_channel_estep`, `test_pure_path_marks_sigma_twohop_reflection_and_surrogates`, and `test_diagonal_gl_route_reports_not_exactly_gauge_invariant`.
- [ ] **Step 2: Run** the focused tests; expect overstated active parameters and permissive purity.
- [ ] **Step 3: Implement.** For linear decode count `V*K` plus optional bias; for prior-bank KL decode count `2*V*K`. Add one s-channel E-step term when enabled. Expand pure flags for full sigma update, zero two-hop coupling, no reflection sampling, gauge parameterization, fixed-prior surrogate, and `group.invariant_for(family)`. Do not reconfigure the model.
- [ ] **Step 4: Run** reporting and artifact tests; expect PASS.
- [ ] **Step 5: Commit** `fix(report): account for active work and exactness`.

### Task 3: Reuse one diagnostic snapshot per evaluation

**Files:** Modify `vfe3/model/model.py`, `vfe3/train.py`, `vfe3/viz/extract.py`; modify `tests/test_run_diagnostics_2026_06_13.py`, `tests/test_viz.py`.

**Interfaces:** `DiagnosticSnapshot` contains encoded belief, per-layer priors/outputs, final belief, logits, beta/gamma maps, and E-step trace. `model.build_diagnostic_snapshot(tokens) -> DiagnosticSnapshot`; diagnostic consumers accept `snapshot=None`.

- [ ] **Step 1: Add failing tests** `test_eval_diagnostics_builds_one_snapshot`, `test_snapshot_and_independent_diagnostics_are_value_equal`, and `test_attention_and_trace_reuse_snapshot_without_forward_replay` using forward-call spies.
- [ ] **Step 2: Run** them; expect four or more forwards.
- [ ] **Step 3: Implement.** Extend the existing capture dictionaries to retain per-layer priors/outputs and maps during one no-grad forward. Build an immutable snapshot and make diagnostics, attention maps, gamma maps, trace, per-position loss, and converged-state extractors read it when supplied. `_val_diagnostics` builds once and passes it to all consumers.
- [ ] **Step 4: Run** diagnostics and viz tests; expect one forward and equal values.
- [ ] **Step 5: Commit** `perf(diagnostics): reuse one converged evaluation snapshot`.

### Task 4: Direct-link factored transport and cheap clamp monitoring

**Files:** Modify `vfe3/geometry/transport.py`, `vfe3/train.py`; modify `tests/test_regime_ii_link.py`, `tests/test_train.py`.

**Interfaces:** `DirectLinkTransport(exp_phi, exp_link, exp_neg_phi)` is consumed by mean/covariance transport without allocating `(B,N,N,K,K)`. Bare links construct only edge exponentials. Clamp Gram is cached per generator value/device.

- [ ] **Step 1: Add failing tests** `test_charted_direct_link_does_not_materialize_dense_pair_transport`, `test_charted_direct_link_factored_matches_dense_reference`, `test_bare_direct_link_skips_vertex_exponentials`, and `test_phi_clamp_monitor_reuses_cached_gram`.
- [ ] **Step 2: Run** them; expect dense allocation, unused exponents, and repeated Gram work.
- [ ] **Step 3: Implement.** Add the factored container and dispatch in `transport_mean`/`transport_covariance`. For the bare link return an edge-only container and identity vertex semantics without calling `build_factored_transport`. Cache the detached Gram using a weak/value-stable generator signature and invalidate on device/dtype/value change.
- [ ] **Step 4: Confirm** the core plan's bounded closure-cache and no-host-sync tests also pass, closing Finding 14 and its addendum together.
- [ ] **Step 5: Run** link and train tests; expect PASS.
- [ ] **Step 6: Commit** `perf(transport): keep direct links factored`.

### Task 5: Figure memory gates, config migration, and cleanup

**Files:** Modify `vfe3/config.py`, `vfe3/viz/report.py`, `vfe3/train.py`, `vfe3/run_artifacts.py`; modify `tests/test_config.py`, `tests/test_report.py`, `tests/test_round3_artifacts.py`, `tests/test_reporting_additions.py`.

**Interfaces:** `config_from_serialized(payload: Mapping[str, Any], *, source: str) -> VFE3Config` filters unknown fields with a warning. `generate_figures(..., allow_large: bool = False)` owns the memory guard.

- [ ] **Step 1: Add failing tests** for strict `force_large_figures` bool, legacy unknown config fields, comparison-figure closure after thunk/save failure, standalone memory guard, and periodic attention figures disabled by `generate_figures=False`.
- [ ] **Step 2: Run** them; expect bypass/crash/leak/replay.
- [ ] **Step 3: Implement.** Add strict bool validation. Move the shared field-filter helper to `vfe3/config.py` and use it in report/generation loaders. Snapshot matplotlib figure numbers before each comparison thunk and close new figures in `except/finally`. Compute the full-vocab memory estimate inside `generate_figures` and gate large extractors. Gate periodic attention/gamma PNG creation on the existing `generate_figures` field.
- [ ] **Step 4: Run** config/report/figure tests; expect PASS.
- [ ] **Step 5: Commit** `fix(figures): guard reusable generation and close failures`.

### Task 6: Explicit condition kind and honest degenerate statistics

**Files:** Modify `vfe3/numerics.py`, `vfe3/metrics.py`, `vfe3/viz/extract.py`; modify `tests/test_numerics.py`, `tests/test_metrics.py`, `tests/test_sigma_gate.py`.

**Interfaces:** `condition_number(matrix, *, kind: Literal["auto","full","diagonal"] = "auto", eps=...)`; Spearman uses paired-finite average ranks; zero-token bootstrap returns NaNs.

- [ ] **Step 1: Add failing tests** for square `(N,K)` diagonal tables with `N==K`, full override, invalid kind, average ties, nonfinite filtering, fewer than two pairs, zero variance, zero-token aggregate, and unchanged nondegenerate results.
- [ ] **Step 2: Run** numerics/metrics tests; expect eigensolve misclassification and finite-looking degenerate results.
- [ ] **Step 3: Implement.** Honor explicit kind before shape inference and pass it from known-family callers. Build fractional average ranks by tie groups after filtering paired finite entries. Return NaN when fewer than two pairs remain; preserve zero for finite zero-variance inputs. Return `{ce,lo,hi}=NaN` when total tokens are zero.
- [ ] **Step 4: Run** numerics, metrics, and sigma-gate tests; expect PASS.
- [ ] **Step 5: Commit** `fix(metrics): make shape and degeneracy explicit`.

### Task 7: Machine-readable check scripts and API conventions

**Files:** Create `check_junit.py`; modify `check_audit_fixes.py`, `check_gpu_tests.py`, `vfe3/config.py`, `vfe3/inference/e_step.py`, `vfe3/viz/report.py`; create `tests/test_fixes_20260709_scripts.py`; modify `tests/test_config.py`.

**Interfaces:** `read_junit_counts(path: Path) -> Dict[str, int]`; `run_pytest_junit(args: Sequence[str], *, prefix: str) -> tuple[int, Dict[str,int]]`.

- [ ] **Step 1: Add failing tests** for both JUnit root forms, parametrized/skip counts, randomized E-step integer bounds, and the public signature ordering. Add a source-text assertion that touched report text uses `gray`/`color`.
- [ ] **Step 2: Run** script/config tests; expect missing parser and fail-late floats.
- [ ] **Step 3: Implement.** Parse every `testsuite` element and sum tests/failures/errors/skipped. Both scripts create/delete a temporary XML and print derived passes. Validate `type(e_steps_min) is int` and `type(e_steps_max) is int` when randomization is active. Mechanically reorder the keyword-only `e_step` signature to tensor/float-or-tensor, undefined/defined scalar groups, booleans, then optionals; call sites remain keyword based. Replace UK spellings in touched warning/comments.
- [ ] **Step 4: Run** `python -m pytest tests/test_fixes_20260709_scripts.py tests/test_config.py tests/test_e_step.py`; expect PASS.
- [ ] **Step 5: Commit** `fix(tooling): derive test counts and enforce public contracts`.

### Task 8: Transport registry owns covariance-class metadata

**Files:** Modify `vfe3/geometry/transport.py`, `vfe3/run_artifacts.py`; modify `tests/test_run_artifacts.py`, registry tests.

**Interfaces:** `register_transport(name, fn, *, needs_mu, needs_sigma, batch_independent, covariance_class)`; metadata lookup returns the registered covariance class.

- [ ] **Step 1: Add failing tests** `test_every_transport_registers_covariance_class`, `test_pure_path_report_reads_transport_registry_metadata`, and override metadata replacement.
- [ ] **Step 2: Run** them; expect hard-coded report mapping.
- [ ] **Step 3: Implement.** Extend the transport registration record, register the existing five exact strings, and replace the literal report dictionary with registry lookup. Override replaces the complete record.
- [ ] **Step 4: Run** artifact and registry tests; expect PASS.
- [ ] **Step 5: Commit** `refactor(transport): register covariance metadata at the seam`.

## Reporting/Tooling Plan Verification

- [ ] Run `python -m pytest tests/test_curated_audit_reporting_20260709.py tests/test_reporting_additions.py tests/test_report.py tests/test_metrics.py tests/test_numerics.py tests/test_viz.py tests/test_run_artifacts.py tests/test_fixes_20260709_scripts.py tests/test_run_naming.py tests/test_phase0_forward_beliefs.py --junitxml=C:\tmp\vfe3-curated-reporting-tooling.xml`.
- [ ] Read JUnit attributes and update each assigned ledger row.
- [ ] Run `git diff --check` and a banned-language scan on touched prose.
