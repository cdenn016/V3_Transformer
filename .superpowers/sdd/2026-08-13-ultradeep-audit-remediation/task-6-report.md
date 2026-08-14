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

Implementation commit: `1448106484e8236c23bdec4b23a3927cd9ba674e` (`fix: make diagnostics and evaluation accounting honest`).

The report-finalization commit is docs-only and is identified in the Task 6 handoff.

## Review-remediation closure

The six Important review findings and one Minor report defect were addressed under strict TDD in implementation commit `45477b5` (`fix: close task 6 review gaps`). The focused RED fixture is `.verification/remediation-2026-08-13/task-06-review-red.xml`: **14 tests, 14 failures, 0 errors, 0 skips**. The post-fix focused contract modules passed **24 tests, 0 failures, 0 errors, 0 skips** before the final integration lane.

The shared CE/PPL contract now drives evaluation plus both scaling consumers. CE=25 remains exact `exp(25)` and CE=1000 is positive `inf` only because exponentiation overflows. Training derives expected counts independently from every target tensor/microbatch; decoder scored/excluded counts must partition that value. Evaluation transfers nats as float64 and counts as int64. The `2**53 + 1` boundary is preserved exactly, and non-int64 device counts fail visibly.

Scheduler metadata serializes every group base LR, role, frozen state, absolute/fractional floors, and executable effective floor. All built-in divergence functionals carry typed display label/units metadata; real overrides preserve metadata unless explicitly replaced. Every objective plot receives the active divergence family, with a unit-free `divergence` fallback.

Default public artifacts are evidence-neutral: `iterate_trajectory.png`, `estep_endpoint_delta.png`, `free_energy_relationship.png`, and `f_ce_relationship.png`. Compatibility function/registry aliases remain. No endpoint delta is promoted to descent evidence. Target blindness is described only as structural separation, with no correlation-sign claim.

The diagnostic capture now records the immediate BlockMLP input after mixer/CG/norm and immediate output. Each pre/post state serializes all weighted contributors (`self_coupling`, `belief_coupling`, `attention_entropy`, `twohop_coupling`, `hyper_prior`, `model_coupling`, `meta_entropy`, `observation_nll`) plus state-qualified raw self, observation, hyper-prior, and gamma components. Actual belief tensors establish equality; only proven-equal states share evaluation. Regression coverage includes mixer, CG, norm, two-hop, observation likelihood, reflection, model channel, and BlockMLP off/on cases, and reconciles both state totals.

The final authoritative CPU lane is `.verification/remediation-2026-08-13/task-06-review-final.xml`: **377 tests, 0 failures, 0 errors, 3 expected skips**. The three skips are pre-existing opt-in slow integration cases. A broader exploratory lane also collected `tests/test_extract_forward_fidelity.py`; seven projected-encoder fixtures failed before Task 6 execution because their current configuration violates the existing canonical-content-projected admissibility contract. Those Task 5/config fixtures were outside Task 6 ownership and were not weakened. The two genuine compatibility failures in that exploratory lane were corrected and are green in the authoritative result.

Serialized compatibility invariants:

```json
{
  "ce_ppl": {"ce25_exact": true, "overflow_positive_inf": true},
  "counts": {"dtype": "int64", "boundary": 9007199254740993, "partition_enforced": true},
  "artifacts": {
    "iterate": "iterate_trajectory.png",
    "endpoint": "estep_endpoint_delta.png",
    "f_ce": "f_ce_relationship.png",
    "legacy_callable_aliases": true
  },
  "state_totals": {"pre_reconciles": true, "post_reconciles": true, "shared_only_if_equal": true}
}
```

Warnings in the authoritative lane are existing configuration notices, deliberately short-run parameter-motion notices, scheduler-test call-order warnings, and known empty-artist plotting warnings. No warning indicates a Task 6 accounting, labeling, state-boundary, or PPL-contract failure. CPU-only execution was used as required; no GPU claim is made.
