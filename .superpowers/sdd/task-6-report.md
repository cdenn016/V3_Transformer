# Task 6 RED/GREEN Report

Task 6 repaired Q1 through Q3 and Q5 through Q9 together with D1 through D4. Q4 was already repaired by Task 5 and was not duplicated. The changes reserve same-second run directories atomically, use uniquely reserved same-directory temporary files for artifact publication, apply and persist the shared deterministic runtime contract in the ring driver, and publish explicit requested-design records for scaling and multi-seed runs. Each multi-seed launch now atomically reserves its own invocation directory containing the request manifest and only that invocation's run directories, so sequential and concurrent launches cannot overwrite or cross-join one another. Scaling and multi-seed analyses retain failed, missing, unreadable, duplicate, and nonfinite request statuses; an incomplete declared scaling design cannot produce pooled, per-route, validation-frontier, or inference-correlation analyses from survivors. Controlled figures and their JSON sidecars publish independently, malformed multi-seed requests fail before training, stochastic generation uses and persists an explicit seed, the visualization extra declares NetworkX directly, and duplicate-OpenMP suppression is available only through the explicit `VFE3_ALLOW_DUPLICATE_OPENMP=1` compatibility opt-in.

The experiment-construction pass repaired only invalid arm-local prerequisites. The ablation baseline remains `e_step_update="mm_exact"` with `phi_precond_mode="pullback_per_block"`, the scaling baseline remains `phi_precond_mode="pullback_per_block"`, and `SWEEP_ORDER` remains `['gamma_prior_weight', 'lambda_twohop']`. The 17 MM-incompatible ablation cells use the gradient E-step without changing their valid controls. The tied GL arms use the ambient Killing preconditioner, `so3_spin2x4` uses four heads, and the coupled-head arm uses causal no-self priors for both beta and gamma. All six tied scaling cells use the Killing preconditioner. Both drivers now construct every declared arm before launching training.

## RED evidence

The initial focused command was:

```text
python -m pytest tests\test_2026_07_15_driver_reliability_remediation.py --junitxml=C:\tmp\vfe3-task6-red-20260716.xml
```

The machine-readable JUnit attributes were:

```text
tests=19 failures=18 errors=0 skipped=0 time=5.884
```

Pytest reported `18 failed, 1 passed in 5.89s`. The sole passing probe was the positive control showing that an explicitly requested duplicate-OpenMP compatibility setting was already reachable. The 18 failures covered same-second directory aliasing, shared writer temporary names, the missing ring deterministic-runtime contract and persisted state, survivor-only scaling and multi-seed aggregation, destructive figure/sidecar coupling, malformed run-count and duplicate-seed acceptance, ambient stochastic-generation RNG, absent generation records, the missing NetworkX extra, default OpenMP environment mutation, scaling's false success exit, and the invalid ablation/scaling arms.

## GREEN evidence

The focused gate after implementation was:

```text
python -m pytest tests\test_2026_07_15_driver_reliability_remediation.py --junitxml=C:\tmp\vfe3-task6-green2-20260716.xml
```

JUnit recorded:

```text
tests=19 failures=0 errors=0 skipped=0 time=4.493
```

Pytest reported `19 passed, 2 warnings in 4.50s`. An earlier GREEN attempt reached 18 passes and exposed a Windows-only simultaneous `os.replace` race caused by the test forcing a single replacement attempt. The test retained the concurrent-writer and unique-temporary assertions while using the production retry contract; the rerun above was clean. The focused controlled-figure probe was also isolated from an unrelated Python 3.14/PyArrow extension teardown fault by replacing its unnecessary scikit-learn computations with deterministic test doubles.

The final focused-plus-neighbor command covered the Task 6 probes, multi-seed analysis, ring bundle/resume publication, and stochastic generation:

```text
python -m pytest tests\test_2026_07_15_driver_reliability_remediation.py tests\test_multiseed.py tests\test_efe_ring_experiment.py tests\test_generate.py --junitxml=C:\tmp\vfe3-task6-final-focused2-20260716.xml
```

JUnit recorded `tests=82 failures=0 errors=0 skipped=1 time=6.194`; pytest reported `81 passed, 1 skipped, 2 warnings in 6.20s`.

The first neighboring command covered run naming, artifact persistence, ring resume, generation, multi-seed analysis, scaling routes, and the recommended E-step/phi ablations:

```text
python -m pytest tests\test_run_naming.py tests\test_run_artifacts.py tests\test_efe_ring_experiment.py tests\test_generate.py tests\test_multiseed.py tests\test_scaling_mup.py tests\test_blocks_k48_followup_routes.py tests\test_estep_phi_ablation_routes_20260715.py --junitxml=C:\tmp\vfe3-task6-neighbor1-20260716.xml
```

JUnit recorded `tests=159 failures=0 errors=0 skipped=5 time=7.672`; pytest reported `154 passed, 5 skipped, 26 warnings in 7.67s`.

The second neighboring command covered controlled UMAP comparison, scaling-data integrity, audit artifact tooling, and ablation route/resume/reporting behavior:

```text
python -m pytest tests\test_controlled_umap_comparison_20260714.py tests\test_2026_07_15_data_integrity_remediation.py tests\test_audit_artifact_tooling_20260713.py tests\test_ablation_sweep_route_compatibility_20260711.py tests\test_ablation_artifact_resume_20260712.py tests\test_ablation_tackon.py tests\test_ablation_reporting.py --junitxml=C:\tmp\vfe3-task6-neighbor2-20260716.xml
```

JUnit recorded `tests=128 failures=0 errors=0 skipped=0 time=12.854`; pytest reported `128 passed, 35 warnings in 12.86s` and the command returned zero. After the pytest summary, this host printed its known Python 3.14/PyArrow native-extension access-violation teardown trace. No test failed and the JUnit document was complete, but that host diagnostic is recorded rather than hidden.

## Final review-gap evidence

The final read-only diff review found that the first multi-seed manifest lived at the shared `RUN_ROOT` and that incomplete scaling designs still supplied survivor validation points to supplementary figures. Two existing focused tests were strengthened, without increasing the 19-test count, to require invocation-owned run grouping and to require empty parameter, inference, and validation figure inputs whenever the requested design is incomplete. Before production repair, the narrow command reported `2 failed in 0.35s`; its JUnit attributes were `tests=2 failures=2 errors=0 skipped=0 time=0.354`.

After reserving a unique multi-seed group per invocation and gating all scaling analysis inputs, the same narrow command reported `2 passed in 0.31s`; JUnit recorded `tests=2 failures=0 errors=0 skipped=0 time=0.304`. The final complete focused module then reported `19 passed, 2 warnings in 4.55s`; JUnit recorded `tests=19 failures=0 errors=0 skipped=0 time=4.550`.

These were selected neighboring runs, not the full or slow suite. `git diff --check` passed. A targeted Ruff unused-import check found only the pre-existing unused `coverage_lines` import in `scaling.py`; it was left untouched because Task 6 did not create it and the task requires surgical edits.

## Q3 follow-up review repair

The Q3 follow-up made both analysis paths fail closed against their invocation manifests. `scaling_analysis._requested_design` now requires a readable schema-version-1 manifest, explicit `complete` or `success` top-level status, nonempty unique route and seed declarations, exact typed cell identities, explicit recognized cell statuses, no duplicate cells, complete seed coverage for every declared route-label pair, exactly one harvested result per requested cell, and finite positive test cross-entropy. Missing, pending, running, incomplete, failed, malformed, and unverifiable designs now set `complete=false`; their parameter fits, validation points, frontier test, per-route estimates, inference correlations, fit weights, and figure inputs are empty. A completed design also filters fit inputs to declared cells, so unrelated stale summaries cannot join the analysis.

Multi-seed launch manifests now begin with pending top-level and per-seed states, transition through running, finish as complete after every requested seed returns, and atomically record a failed cell plus error if a run raises. Scalar, curve, and per-layer aggregation use the same requested-seed design join. A declared failed seed overrides stale artifacts; a missing or duplicate requested run, unreadable CSV, or requested seed with no finite curve or per-layer values withholds the corresponding aggregate. `main()` publishes no survivor scalar table, curve summary, per-layer summary, or figures unless the requested design, headline scalar, curve artifacts, and per-layer artifacts are complete. Its JSON output records the incomplete design and explicit withheld flags.

The strict RED command was:

```text
python -m pytest tests/test_2026_07_15_driver_reliability_remediation.py tests/test_multiseed.py --junitxml=C:\tmp\vfe3-task6-review2-red-20260716.xml
```

JUnit recorded `tests=62 failures=19 errors=0 skipped=0 time=5.972`; pytest reported `19 failed, 43 passed, 2 warnings in 5.98s`. The failures covered each unfinished or malformed scaling-manifest class, missing launcher status transitions, and survivor curve/per-layer publication for missing, failed, unreadable, and nonfinite requested seeds.

The final named GREEN gate was:

```text
python -m pytest tests/test_2026_07_15_driver_reliability_remediation.py tests/test_multiseed.py tests/test_reporting_additions.py tests/test_scaling_mup.py --junitxml=C:\tmp\vfe3-task6-review2-green-final-20260716.xml
```

JUnit recorded `tests=113 failures=0 errors=0 skipped=0 time=12.284`; pytest reported `113 passed, 70 warnings in 12.29s`. The two pre-existing scaling-analysis fixture helpers now create explicit completed request manifests, preserving their positive-fit and figure-publication coverage under the fail-closed contract. The run covered only the focused Task 6 module and the directly affected multiseed and scaling-analysis modules, as requested.
