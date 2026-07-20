# Parameter-Matched Ablation Sweep Design

**Date:** 2026-07-20

**Status:** Approved

## Purpose

`ablation.py` can vary structural fields such as `embed_dim` and `n_heads`, but its current sweeps allow the realized model size to change freely. The requested addition is a parameter-matched structural ablation: the user chooses a total learned-parameter budget and a relative tolerance, the runner searches a declared architecture grid, and the sweep trains several structurally different models whose realized parameter counts remain close to that budget.

The target is a click-to-run setting rather than a hard-coded 30 million parameters. A user may select 15 million, 30 million, 100 million, or another feasible budget without changing the selection algorithm.

## Scientific Comparison

The matched sweep retains one architecture for each requested `embed_dim`. Within that width, it selects the valid candidate with the closest realized parameter count to the target. `n_heads` is the primary compensating field because, on the baseline `block_glk` route, it changes the gauge-generator table size while preserving the requested width. The candidate grid may include additional real `VFE3Config` structural fields, but every selected row remains grouped by `embed_dim` so a dense cluster around one width cannot displace the other requested widths.

This is a fixed-budget structural ablation, not a single-variable head-count ablation. Because `embed_dim`, `n_heads`, gauge-block size, and associated compute can co-vary, reporting must describe the full selected configuration and must not attribute an observed difference to one field alone.

## Click-to-Run Configuration

Add an opt-in `parameter_matched` entry to `SWEEPS`. Its `parameter_grid` maps real `VFE3Config` fields to explicit candidate lists, and its `match_by` field is `embed_dim`. The initial grid covers multiple widths and head counts; cross-product pairs that violate the divisor constraint are filtered during validation. The existing baseline and ordinary sweeps remain unchanged.

Add two settings to the module-level `CONFIG` dictionary:

```python
"target_n_params":                 30_000_000,
"max_param_relative_deviation":   0.02,
```

The target must be an exact positive integer. The tolerance must be a finite float in `[0, 1)`. These settings are consulted only for a parameter-matched sweep. The 30 million default is an editable example, not a restriction in the implementation.

When an arm changes `embed_dim` without explicitly setting `kl_max`, its generated override sets `kl_max = 8 * embed_dim`, preserving the baseline numerical-safety convention across widths.

## Candidate Expansion and Exact Counting

The selector forms the Cartesian product of the declared candidate lists, merges the sweep's `requires` values, derives width-dependent fields, and validates every field name against the `VFE3Config` dataclass. Candidate ordering follows the declared grid order and is deterministic.

Each candidate is then passed through `VFE3Config`. Invalid cross-field combinations, including head counts that do not divide `embed_dim`, are recorded as rejected candidates and are never trained. Each valid configuration is used to construct a temporary CPU `VFEModel`; the selector counts `sum(parameter.numel() for parameter in model.parameters())`, then releases the model before considering the next candidate. This is the same realized count recorded by `run_single`, so optional head-mixer, positional, and other live parameter groups are included. The approximate predictor in `scaling.py` is not an acceptance authority.

Within each `embed_dim` group, candidates are ordered by absolute relative deviation from the target, then by declared candidate order. The first candidate is retained only when its relative deviation is at most the configured tolerance. The selector requires at least two retained widths. If fewer qualify, startup fails before loading data or starting training and prints the closest rejected candidate for every requested width, including its realized count and deviation.

## Artifact Identity and Reporting

A parameter-matched invocation uses a budget-specific output scope derived from the logical sweep name, exact target, and normalized tolerance, for example `parameter_matched_N30000000_rtol0p02`. This prevents a later 50 million parameter run from being aggregated with an earlier 30 million parameter run while preserving ordinary sweep directory names.

Every selected cell records these fields in `ablation_result.json` and `sweep_results.csv`:

* `target_n_params`
* `n_params`
* `param_difference`
* `param_relative_deviation`

`sweep_meta.json` records the target, tolerance, complete candidate-grid definition, grouping field, selected configurations, and rejected-candidate summary. The budget specification is part of the sweep aggregation identity, while the effective `VFE3Config` remains part of each existing cell reuse contract. Console analysis prints target and deviation beside the realized parameter count.

Ordinary single-field and multi-arm sweep schemas, tack-on behavior, resume semantics, result collection, and figures retain their existing behavior.

## Failure Handling

Malformed targets, tolerances, grids, unknown fields, empty candidate lists, duplicate selected labels, and insufficient matched widths are setup errors. They fail before training rather than becoming `primary_val_ppl = inf` cells. Individual invalid architecture combinations are expected search rejections and are summarized; they do not abort expansion when enough valid matched widths remain.

The temporary counting models are built one at a time so peak host memory is bounded by the largest declared candidate rather than the sum of the grid. Model construction occurs before data loaders are opened and does not change the training stream because `run_single` reseeds model construction and the train loader for every actual cell.

## Verification

Implementation will follow a test-first sequence using focused direct Python assertion scripts rather than pytest, per the user's instruction. The checks will first fail against the unimplemented schema and then pass after the minimum code is added. They will cover Cartesian expansion, invalid-pair rejection, exact realized counting, one-per-width selection, deterministic tie-breaking, tolerance rejection, the two-width minimum, derived `kl_max`, budget-specific output scope, and persisted budget metadata.

Final mechanical verification will include `py_compile`, direct import and selection probes, a dry run that performs candidate selection without loading a dataset or training, and validation of `.verification/ledger.json`. No additional pytest command will be run. The pre-edit isolated `origin/main` baseline had 75 ablation-focused tests with 73 passes and two failures in `tests/test_ablation_artifact_resume_20260712.py`; those failures predate this task and are not included in the feature scope.
