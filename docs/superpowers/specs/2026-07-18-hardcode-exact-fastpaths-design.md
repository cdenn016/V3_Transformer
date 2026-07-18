# Hardcode Exact Fast Paths Design

**Date:** 2026-07-18
**Status:** Approved
**Base:** `origin/main` at `865b474`

## Objective

Make the exactness-preserving compact-phi transport, pairwise diagonal-KL
statistics reuse, and per-head mean transport mandatory production behavior.
The three choices cease to be model or experiment configuration. Unsupported
shapes, routes, and dtypes continue to use their existing automatic fallback
implementations.

## Decisions

The fields `compact_phi_block_transport`, `reuse_pairwise_kl_stats`, and
`transport_mean_per_head` are removed from `VFE3Config`. They are also removed
from the click-to-run dictionaries in `train_vfe3.py`, `scaling.py`, and
`ablation.py`. Historical serialized configurations containing these names are
not supported and receive no retired-key migration rule.

Production model paths request all three optimizations unconditionally. The
low-level Boolean parameters remain available only as internal differential
oracles for parity and fallback tests. They are not exposed through the model
configuration or executable drivers.

## Runtime behavior

Compact phi transport is mandatory when the existing eligibility predicate is
satisfied: phi parameterization, flat transport, reflections off, and the
canonical `block_head_row_major` coordinate layout. The predicate no longer
consults a user toggle. Ineligible configurations continue to use dense
transport because applying the packed layout outside that domain would be
incorrect.

Pairwise KL statistics reuse is requested on every production belief E-step.
The kernel's existing guards remain authoritative: unsupported families,
divergences, routes, value gauges, and non-float32 tensors recompute through
the established path. Hardcoding the request therefore does not broaden the
mathematical domain of the reuse kernel.

Per-head mean transport is requested on every production transport build.
Factored block transports contract each block separately; layouts without the
supported factorization retain the established transport implementation. The
operation is algebraically the same block-diagonal contraction, subject only
to the already tested float32 reassociation tolerance.

## Internal API boundary

The lower-level parameters on transport, E-step, and gradient helpers remain
temporarily available. Existing tests use their false values to compare dense
and optimized results, VJPs, dtype behavior, eligibility rejection, and
fallback routing. Removing these internal oracles would enlarge the change
without improving the production contract. No production caller may derive
their value from configuration after this change.

## Testing

A new architectural regression first fails against the current public
configuration and production forwarding. It then pins that the three dataclass
fields and three driver dictionary keys are absent, shared E-step arguments
request pairwise reuse, eligible model routes select compact phi, ineligible
routes fall back, and production transport call sites request per-head mean
contraction.

Existing compact-phi, pairwise-reuse, transport, AMP, reflection, objective,
and visualization tests remain the numerical oracles. Tests whose sole purpose
is to prove the old fields defaulted off or accepted explicit values are
removed or rewritten around the hardcoded production contract.

## Verification gate

The untouched baseline JUnit records 3,659 tests, 10 failures, zero errors,
and 16 skips in 322.093 seconds. The ten accepted baseline node IDs are:

- `tests.test_2026_07_15_data_integrity_remediation::test_training_log_always_names_bits_per_token_when_bpc_is_available`
- `tests.test_ablation_artifact_resume_20260712::test_missing_requested_diagnostics_output_forbids_contract_publication`
- `tests.test_ablation_artifact_resume_20260712::test_run_single_terminal_merge_preserves_metadata_and_primary_val_ppl`
- `tests.test_curated_geometry_math_20260709::test_prior_model_and_decode_variance_reads_share_guard`
- `tests.test_diagnostics::test_attention_and_trace_reuse_snapshot_without_forward_replay`
- `tests.test_estep_fixed_point_reporting_20260715::test_one_step_ahead_residual_is_distinct_from_configured_last_step`
- `tests.test_report::test_generate_figures_reuses_one_same_token_snapshot`
- `tests.test_report::test_generate_figures_memory_guard_uses_materialized_batch_peak`
- `tests.test_run_diagnostics_2026_06_13::test_val_diagnostics_passes_explicit_diagonal_covariance_for_square_trace`
- `tests.test_train::test_validation_finalizer_appends_to_existing_metrics_schema`

Acceptance requires zero new failure node IDs, zero errors, and focused parity
coverage for all three hardcoded paths. Pass and skip counts must come from
JUnit, not console inference.

## Scope boundaries

This change does not alter the mathematical formulas, expand compact transport
eligibility, remove dense/recompute fallback implementations, change AMP or
fp64-island rules, or modify other performance toggles. The Research wiki was
consulted for the gauge and transport domain constraints but is not modified.
The user's current local checkout changes remain outside the isolated task
worktree.
