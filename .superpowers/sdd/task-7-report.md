# Task 7 Runtime Performance, Reuse, and Synchronization Report

## Outcome

Task 7 repairs P1, P2, P4, P5, P6, P7, S2-C2, S2-C3, and S2-C4 without changing the configured/default mathematical routes. Compact block diagnostics and diagnostic free-energy traces retain factored transports. Direct-omega updates cache the converted generator basis and Gram pseudo-inverse by source tensor identity/version, device, and dtype; finite/nonsingular status remains on device until one aggregate host decision, and float64 determinant validation runs only when gauge diagnostics are explicitly collected. Candidate omega updates are staged and applied only after validation succeeds.

Phi projection keeps the exact full-table norm bound while using its memory-budget-derived chunk size and one aggregate statistics transfer. Evaluation accumulates token counts and weighted nats in device float64 scalars and transfers one two-scalar vector. EMA transfers one finiteness vector per update and retains the existing per-parameter skip behavior. The compatibility-only gamma objective helpers now delegate to `_gamma_coupling_rows`; production objective authority remains `_model_channel_free_energy` and the Task 1 Metropolis evaluator.

Fixed-point diagnostics reuse captured E-step states and execute only the next map evaluation. Snapshot reuse accepts the realized terminal depth from early halting. Report population extraction runs `forward_beliefs` once per batch, shares its records across the belief, covariance/CE, model-channel, and vocabulary consumers, and CPU-offloads each full-vocabulary logit batch so accelerator memory remains bounded to one logit/probability working batch. The diagnostic snapshot remains the single detailed pass for same-sequence consumers.

The skipped-step phi-projection regression still expected a scheduler-before-optimizer warning. Task 1 intentionally changed the scheduler clock to advance only when `did_step` is true. The test now asserts that a nonfinite skipped step neither emits that warning nor advances `scheduler.last_epoch`; no Task 7 production change was made to this path.

## RED evidence

The initial focused suite was written before production edits and run as follows:

```text
python -m pytest tests/test_2026_07_15_performance_remediation.py --junitxml=C:\tmp\task7-red.xml
Exit code: 1
FFFFFFFFF                                                                [100%]
9 failed, 1 warning in 0.99s
```

The literal machine-readable RED summary was:

```xml
<testsuite name="pytest" errors="0" failures="9" skipped="0" tests="9" time="0.988" timestamp="2026-07-16T13:08:16.860280-05:00" />
```

The nine failures measured four dense `_transport` calls on the compact diagnostic path, two full-basis Gram factorizations across two steps, seven scalar tensor truth decisions in one omega update, nonaggregated projection statistics, three per-batch evaluation tensor-to-Python conversions, the absent snapshot keyword, the absent shared inference bank, independent gamma helper logic, and two EMA scalar truth decisions.

Independent review then identified three uncovered edge paths. Their tests were added before the fixes and produced this second literal RED result:

```text
python -m pytest tests/test_2026_07_15_performance_remediation.py::test_compact_trace_fallback_keeps_free_energy_transport_factored tests/test_2026_07_15_performance_remediation.py::test_fixed_point_snapshot_reuses_realized_early_halt_terminal tests/test_2026_07_15_performance_remediation.py::test_shared_report_inference_bank_serves_all_population_consumers_once --junitxml=C:\tmp\task7-review-red.xml
Exit code: 1
FFF                                                                      [100%]
3 failed, 1 warning in 0.22s
```

```xml
<testsuite name="pytest" errors="0" failures="3" skipped="0" tests="3" time="0.222" timestamp="2026-07-16T13:29:37.534538-05:00" />
```

Those failures showed three dense fallback builds in a two-iteration trace, rejection of a valid early-halted snapshot, and zero logit offloads across a two-batch report bank.

## GREEN evidence

The final focused command was:

```text
python -m pytest tests/test_2026_07_15_performance_remediation.py --junitxml=C:\tmp\task7-green-focused-final.xml
Exit code: 0
...........                                                              [100%]
11 passed, 1 warning in 0.86s
```

```xml
<testsuite name="pytest" errors="0" failures="0" skipped="0" tests="11" time="0.862" timestamp="2026-07-16T13:30:14.305593-05:00" />
```

The authorized neighboring set covered model diagnostics, model-channel diagnostics, compact transport, gauge optimization, phi projection, evaluation/training, epoch metrics, EMA, extraction fidelity, fixed-point reporting, and figure reporting. No full or slow suite was run.

```text
python -m pytest tests/test_2026_07_15_performance_remediation.py tests/test_diagnostics.py tests/test_model_channel_diagnostics_2026_06_13.py tests/test_p1_compact_phi_block_transport_20260711.py tests/test_gauge_optim.py tests/test_phi_projection_optimization_20260715.py tests/test_train.py tests/test_training_epoch_metrics_20260715.py tests/test_ema.py tests/test_extract.py tests/test_extract_forward_fidelity.py tests/test_estep_fixed_point_reporting_20260715.py tests/test_report.py -m "not slow" --junitxml=C:\tmp\task7-green-neighbor-reviewed.xml
Exit code: 0
223 passed, 8 deselected, 23 warnings in 68.22s
```

```xml
<testsuite name="pytest" errors="0" failures="0" skipped="0" tests="223" time="68.213" timestamp="2026-07-16T13:30:29.983848-05:00" />
```

Targeted static verification also passed:

```text
ruff check --no-cache vfe3/ema.py vfe3/gauge_optim.py vfe3/inference/e_step.py vfe3/model/model.py vfe3/train.py vfe3/viz/extract.py vfe3/viz/report.py tests/test_2026_07_15_performance_remediation.py tests/test_phi_projection_optimization_20260715.py
All checks passed!
```

## Review and ownership

The independent read-only review reported no Critical issue. Its three Important findings were the compact no-snapshot trace fallback, early-halt snapshot depth, and multi-batch accelerator logit retention. Each received a RED regression and was closed before the final 11-test and 223-test GREEN runs. Task 7 did not edit the Research vault or daily ledger, did not run the full/slow suite, and did not push or merge.

## Second blocking review closure

The second blocking review found four remaining boundaries. The report-facing `converged_state`, `diagnostics_per_layer`, `numerical_health`, and no-snapshot `attention_maps` paths now use the model's compact-aware diagnostic transport builder. `converged_state` returns the compact transport instead of explicitly materializing a dense pairwise tensor, and the curvature report consumes that representation directly. The report driver continues to pass one diagnostic snapshot to same-sequence consumers and now moves only that current sequence to the model device after the shared inference bank is CPU-hosted.

The inference bank keeps its list-of-records API but CPU-offloads tokens, targets, every `BeliefState` tensor including phi and compact omega blocks, logits, model-channel means and covariances, and the independent model frame. Its exact memory counter sums unique retained tensor storages across every batch. The full-vocabulary preflight guard scales with the entire bounded batch count, and the post-capture guard checks the exact aggregate bank bytes. Population consumers either remain on CPU or move only their current record to the model device.

Eval-only cadence now enables sparse gauge diagnostics even when `artifacts=None` and logging and CSV output are disabled, so diagnostic determinant validation runs on evaluation steps. Direct-omega retractions are staged across every omega parameter group, validated with one aggregate decision, and only then committed. A singular later candidate leaves all earlier frame tables, dirty masks, gradients, and serialized optimizer state unchanged.

The five second-review regressions were first run before these production changes and failed for their intended reasons:

```text
python -m pytest tests/test_2026_07_15_performance_remediation.py::test_compact_report_fallbacks_keep_transport_factored_end_to_end tests/test_2026_07_15_performance_remediation.py::test_shared_report_inference_bank_serves_all_population_consumers_once tests/test_2026_07_15_performance_remediation.py::test_full_vocab_memory_guard_scales_with_all_retained_batches tests/test_2026_07_15_performance_remediation.py::test_eval_only_step_runs_sparse_omega_determinant_validation tests/test_2026_07_15_performance_remediation.py::test_direct_omega_retraction_is_atomic_across_parameter_groups --junitxml=C:\tmp\task7-review2-red.xml
Exit code: 1
FFFFF                                                                    [100%]
5 failed, 2 warnings in 1.10s
```

```xml
<testsuite name="pytest" errors="0" failures="5" skipped="0" tests="5" time="1.101" timestamp="2026-07-16T13:49:28.126661-05:00" />
```

The literal failures reported four dense transport-builder calls, missing complete CPU transfer, an absent aggregate memory helper, zero eval-only determinant calls, and mutation of the first omega table before the later singular candidate failed.

The final focused module passed from JUnit evidence:

```text
python -m pytest tests/test_2026_07_15_performance_remediation.py --junitxml=C:\tmp\task7-review2-focused-final.xml
Exit code: 0
...............                                                          [100%]
15 passed, 2 warnings in 1.56s
```

```xml
<testsuite name="pytest" errors="0" failures="0" skipped="0" tests="15" time="1.562" timestamp="2026-07-16T13:55:19.137820-05:00" />
```

Only directly affected report, diagnostics, gauge optimizer, eval-cadence, and compact-transport neighbors were run. The slow report integration node remained skipped by its repository marker.

```text
python -m pytest tests/test_report.py::test_converged_state_shapes_and_finite tests/test_report.py::test_generate_figures_drives_live_model tests/test_report.py::test_generate_figures_reuses_one_same_token_snapshot tests/test_diagnostics.py tests/test_gauge_optim.py tests/test_train.py::test_train_caps_periodic_eval_at_eval_max_batches tests/test_omega_direct.py::test_compact_free_energy_and_diagnostics_never_dense_materialize --junitxml=C:\tmp\task7-review2-neighbors.xml
Exit code: 0
.s...........................                                            [100%]
28 passed, 1 skipped, 1 warning in 6.07s
```

```xml
<testsuite name="pytest" errors="0" failures="0" skipped="1" tests="29" time="6.072" timestamp="2026-07-16T13:53:26.261634-05:00" />
```

Targeted static verification passed with `ruff check --no-cache` over the seven production/test files changed by this review closure. The review-owned XML files are outside the repository and are deleted before commit. No full or slow suite was run, and this closure does not push or merge.
