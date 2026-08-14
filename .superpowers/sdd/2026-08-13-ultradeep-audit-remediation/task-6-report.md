# Task 6 report: diagnostics, accounting, split discipline, and labels

## Status and scope

Implemented from remediation base `50540b5384791d1c76a464c7cd7c5fe7e3acd2f7` in the isolated
worktree `C:\tmp\V3_Transformer_ultradeep_remediation_20260813`. The live checkout was not
touched. All execution was CPU-only with `C:/Python314/python.exe`,
`CUDA_VISIBLE_DEVICES=-1`, and `VFE3_TEST_DEVICE=cpu`; no GPU work ran.

Task 6 changes only the listed reporting, analysis, click-to-run naming/comment boundary, and test
files. Tasks 1 through 5 and the A01/A02 waiver remain intact.

## RED evidence

Command:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:VFE3_TEST_DEVICE='cpu'
C:/Python314/python.exe -m pytest tests/test_ultradeep_remediation_reporting_20260813.py -q `
  --junitxml=.verification/remediation-2026-08-13/task-06-red.xml
```

Machine-readable result: **10 selected, 10 failures, 0 errors, 0 skipped**. The failures were the
missing exact/overflow PPL rule, count partition and device-count propagation, evidence-neutral
`final_iterate` label, target-blind separation wording, registered objective label, nonshared
`data_seed=None` status, scheduler metadata, four policy diagnostics, and state-specific totals.

## Implementation

Diagnostics now recompute complete pre- and post-BlockMLP belief-channel component sets using the
active family, divergence functional, transport, temperature, attention prior, entropy gate, and
alpha policy at each named belief state. The unqualified compatibility total aliases the coherent
post-BlockMLP total. A same-object pre/post input is rejected unless equality is explicitly marked
as mechanically established. Raw pre/post self divergences remain available.

E-step artifacts use `final_iterate` unconditionally. Descent requires at least two mechanically
monotone objective values; fixed-point/convergence evidence additionally requires an explicit halt
and residual within tolerance. Current no-halt finalization therefore makes no convergence claim.
Target blindness is described only as structural objective separation; correlation values remain
descriptive and carry no expected sign.

Free-energy plots source objective names and units from the active registered functional metadata,
with generic registered-objective fallbacks rather than hard-coded KL/nats labels. Scheduler
metadata records the executable `max(absolute_floor, fractional_floor * group_base_lr)` rule, and
the click-to-run comment correctly names the active one-percent fractional floor.

Evaluation, training aggregation, periodic validation CSV rows, validation/test result JSON,
summary JSON, and scaling-point records propagate `expected_targets`, `scored_targets`, and
`excluded_targets`. Every partition is validated as
`expected_targets == scored_targets + excluded_targets`. Task 5 device-side decoder counts are
transferred together after aggregation. Exact PPL is `exp(CE)`; Python overflow is serialized as
`inf`, never capped at `exp(20)`.

Exploratory run directory naming and multiseed headline selection use validation PPL. Test PPL
remains a held-out final artifact endpoint. The old `_rename_run_by_ppl` symbol remains as a
compatibility alias whose argument now denotes the selected validation PPL. A `data_seed=None`
panel reports `nonshared_unspecified`; only one explicit common integer seed reports shared order.

The exact diagnostic policy dictionary now includes:

- `m_phi_group_trust_radius`
- `phi_mstep_max_matrix_norm`
- `transport_chart_max_norm`
- `exp_fp64_norm_threshold`

## GREEN evidence and invariants

Authoritative command:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:VFE3_TEST_DEVICE='cpu'
C:/Python314/python.exe -m pytest tests/test_train.py tests/test_run_artifacts.py `
  tests/test_run_diagnostics_2026_06_13.py tests/test_multiseed.py tests/test_viz.py `
  tests/test_ultradeep_remediation_reporting_20260813.py tests/test_run_naming.py `
  tests/test_scaling_mup.py -q `
  --junitxml=.verification/remediation-2026-08-13/task-06-green.xml
```

Final machine-readable result: **283 tests, 0 failures, 0 errors, 2 expected skips** in 96.537 s.
The skips are the pre-existing slow integration tests that require `--runslow`.

Independent known-padding invariant:

```json
{"evaluate_counts":{"excluded_targets":3,"expected_targets":8,"scored_targets":5},"ppl_exact":true}
```

This proves the serialized count closure `8 = 5 + 3` and exact `exp(CE)` on a two-row padded
fixture. `compileall -q` over every changed Python file and `git diff --check` both passed.

## Compatibility and warnings

Compatibility aliases retained: `_rename_run_by_ppl` and `estep_final_f_per_token`. The former now
receives validation PPL; the latter aliases `estep_final_iterate_f_per_token` and is accompanied by
explicit iterate-evidence metadata. Existing unqualified diagnostic terms all refer to the coherent
post-BlockMLP state, while explicit pre/post keys provide the state split.

Warnings were pre-existing configuration notices, scheduler-test ordering warnings, parameter-motion
warnings on deliberately short runs, and active-path semantic notices. No Task 6 warning indicates a
failure or unresolved accounting invariant. No GPU claims were made.

## Commit

Implementation commit: `1448106484e8236c23bdec4b23a3927cd9ba674e` (`fix: make diagnostics and evaluation accounting honest`).`r`n`r`nThe report-finalization commit is docs-only and is identified in the Task 6 handoff.
