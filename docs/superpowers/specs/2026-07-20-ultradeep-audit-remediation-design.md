# Ultra-Deep Audit Remediation Design

Date: 2026-07-20

Approved scope: repair all twenty-one sustained findings in
`docs/audits/ultradeep-audit-codebase-2026-07-20.md` and resolve all ten
failures in the clean CPU baseline. Refuted audit candidates remain outside
the repair scope and must not be converted into changes by preference.

Audited and remediation base:
`aa5aceab7844d48c800d72397c0ce4550c567ba1`, the fetched `origin/main`
revision when remediation began.

## Goal

Close every sustained audit finding with the smallest executable correction,
one regression that fails before the correction, and current machine-readable
evidence. Restore a green complete CPU suite and a green RTX 5090 CUDA-marked
lane without deleting tests, weakening assertions, changing the user's chosen
training regime, or removing the exact mathematical paths preserved by the
repository.

## Non-goals

This remediation does not redesign the VFE objective, change active
hyperparameters, add neural-network capacity, retune experiments, alter
manuscript claims, or repair any of the nine refuted candidates. It does not
silently update the Research wiki. Findings worth retaining there will be
offered for ingest only after code closure.

## Design principles

Each patch addresses the source of one contract failure. No production edit
lands before its regression has been observed failing. Existing tests may be
updated only when current executable behavior and the documented public
contract show that the assertion is stale; the dated edit record must name
that decision. No test is deleted.

The pure path remains available under the existing toggles. Full-covariance
Gaussian computations retain full-SPD semantics, diagonal-family
approximations remain labeled as approximations, and gauge-breaking controls
remain explicit. Numerical safety must not replace exact mathematics with a
different objective.

The implementation is divided by file ownership so parallel work cannot
silently overwrite another lane. Integration occurs only after each lane's
focused tests are green and its diff passes an independent source review.

## Remediation architecture

### Lane A: data and public type contracts

Lane A owns `vfe3/data/datasets.py`, `vfe3/config.py`,
`vfe3/viz/figure_worker.py`, `vfe3/process_utils.py`, and their focused tests.
It closes M1, M2, M12, M13, M17, L1, L2, and L3, and owns the configuration
guard required by M8.

For M1, an uncapped binary cache will no longer expose a live memmap-backed
tensor as identity-bound data. The loader will copy the mapped bytes into an
owned tensor in the cache's native integer dtype after validating metadata.
The existing identity-before and identity-after comparison will surround that
copy, so a concurrent mutation during snapshot creation fails. This incurs no
corpus-sized int64 expansion and preserves capped-load behavior, while later
external file writes cannot change the loaded dataset.

For M2, binary-cache payload, metadata, and optional provenance paths will be
derived from one resolved payload path. Identity calculation and loading must
read the same sidecar bytes. A symlink test will place the link and target in
different directories with conflicting metadata and prove that split
interpretation is rejected or consistently resolved.

For M12 and M13, configuration validation will use centralized plain-type
predicates. Every dataclass field annotated exactly `bool` must satisfy
`type(value) is bool`; every field annotated exactly `int` must satisfy
`type(value) is int`, excluding booleans and fractional numbers. Existing
range and cross-field validation remains responsible for semantic bounds.
Optional integers and structured fields keep their existing explicit
validators. Serialized compatibility migration continues to normalize old
schemas before construction; it may not coerce an invalid current payload.

For M17, `make_dataloader` will require exact booleans for `shuffle` and
`drop_last` before constructing either `TokenWindows` or `DataLoader`. The
literal string `"false"` must fail rather than select a random sampler or omit
the padded evaluation tail.

For L1, finalize-worker JSON will be schema-checked before coercion. Boolean,
integer, string, path, and nullable fields accept only their declared JSON
types. For L2, process commands reject `str` and `bytes` as outer sequences
and require a nonempty sequence of nonempty strings. For L3, irrep
multiplicity must be a positive plain integer.

M8 is closed at configuration time. `encode_mode="per_token_additive"`
represents ordinary additive code and is incompatible with
`m_phi_update_mode="pullback_group"`. The combination will raise a targeted
configuration error rather than routing additive code through a group-chart
optimizer. Existing Euclidean AdamW behavior remains selectable.

### Lane B: artifact and experiment integrity

Lane B owns `vfe3/run_artifacts.py`, `ablation.py`, and their focused tests. It
closes M3, M4, M11, M14, and M16 and diagnoses the two existing ablation
failures.

For M3, selected-model loading will have one nonmutating validation boundary.
It will validate bundle schema, configuration fingerprint, selected-state
tensor contract, executable-code identity, and validation-data identity
before any live state is copied. Finalization will call this same boundary
instead of duplicating the weaker configuration-only path. Legacy bundles
that lack current identity fields remain nonportable and cannot supply a
headline test result without an explicit pre-existing legacy policy.

For M4, a shared model-state tensor validator will reject nonfinite floating
or complex tensors after type, key, shape, dtype, and layout checks and before
`load_state_dict`. Raw resume and best-model selection will use the same
predicate. Integer and boolean state tensors remain subject to their existing
structural checks.

For M11, growing-sequence extrapolation is complete only when the configured
largest requested sequence length succeeds, at least two distinct lengths
exist, and every mandatory point has a finite metric. Failures remain in the
aggregate with their reason. Batch scaling may reduce memory with increasing
sequence length, but the requested domain may not silently shrink to a nearby
two-point curve.

For M14, ablation sweep flags are checked before any truthiness conversion.
Raw strings and numeric surrogates fail with the field name. Persisted
diagnostic flags must equal the exact validated booleans used by execution.

For M16, the zero-E-step held-out comparison becomes an explicit opt-in
configuration field with a default of `False`. When disabled, finalization
performs exactly one full test evaluation. When enabled, it performs the
second comparison, restores `n_e_steps` in a `finally` boundary, and labels
the result as a diagnostic counterfactual rather than another headline test
metric. Selection semantics and configuration fingerprinting include the new
field through ordinary dataclass serialization.

The two failing ablation tests will first be reproduced in isolation. Their
fixtures and expected cell contracts will be compared against current
parameter-matched artifact semantics. Production code changes only if a
missing requested diagnostic or terminal metadata can still publish an
invalid contract; otherwise the stale test contract is corrected and the
dated edit record explains why.

### Lane C: model, inference, geometry, and runtime semantics

Lane C owns `vfe3/train.py`, `vfe3/model/model.py`,
`vfe3/model/prior_bank.py`, `vfe3/inference/e_step.py`,
`vfe3/families/gaussian.py`, `vfe3/geometry/transport.py`,
`vfe3/gauge_optim.py`, and their focused tests. It closes H1, M5, M6, M7, M9,
M10, and M15. It also owns the runtime half of M8 and the non-ablation
baseline failures.

For H1, periodic held-out structural diagnostics will operate on exactly the
first validation sequence. The ordinary validation pass remains unchanged
and still scores the configured validation population. Only the diagnostic
snapshot input is bounded before full-vocabulary decode. At the active
`N=128`, `V=50257` configuration, the logits original-plus-copy lower bound
therefore falls from 12.2698 GiB to about 49 MiB. The regression will assert
the bounded shape at the snapshot boundary and prove that the diagnostic path
does not replay the full batch. No active config toggle is added for this
internal diagnostic safety bound.

For M5, every fused chunked decoder will validate targets once before chunk
reduction. Each target must equal `ignore_index` or lie in `[0,V)`. Invalid
negative and upper-bound IDs raise the same class of error as dense
cross-entropy. Valid and ignored-target numerical parity remains pinned.

For M6, evaluation with zero valid targets is undefined. `evaluate` will
raise a targeted error before constructing CE, PPL, bits-per-token, or BPC.
Callers that intentionally allow an empty diagnostic subset must handle the
exception explicitly; no perfect metric may enter selection or publication.

For M7, `e_steps_backprop_last` receives an exact plain-integer guard in
configuration and a defensive runtime guard at the public E-step boundary.
The integer values zero and one retain their current gradient semantics.
Fractional and boolean values fail before iteration policy is computed.

For M9, `TransportRegistration` will own optional trainable-state
construction and state-key declarations. The model will ask the selected
registration to create its parameter mapping and register those parameters
generically. Transport calls receive that mapping through one generic state
argument. Existing `connection_W`, `connection_M`, and `connection_L` state
keys and checkpoint names remain stable for compatibility, but a new
registered stateful transport can be added without a new literal branch in
model construction or call sites. A synthetic registered transport test will
prove this extension contract.

For M10, the full-Gaussian closed-form KL branch will factor, solve, trace,
and subtract log determinants in float64 whenever caller tensors are float32
or lower precision. The returned divergence follows the existing public dtype
policy after the stable computation. A derivation-backed regression requires
identical valid SPD operands near the admitted conditioning limit to have
near-zero self-KL and near-zero shared-covariance gradient. Separate-operand
gradient tests retain the expected nonzero geometry. The diagonal kernel and
active diagonal route remain byte-unchanged.

For M15, `PriorBank` will construct base mean and variance vocabulary tables
only when the selected encode, prior, or decode route consumes them. Under the
active model-channel prior plus linear decode, those dormant tables become
registered `None` parameters: they consume no storage, enter no optimizer
group, and do not contribute to realized parameter counts. Attribute names
remain present for introspection and checkpoint compatibility logic. Any
route that consumes token priors still constructs the same tables with the
same initialization order.

The remaining eight non-ablation CPU failures will be treated as debugging
obligations, not automatically as production defects. Each failing node will
be run alone, its stack and inputs traced to the first incorrect boundary,
and its closest working peer compared. The BPC log test may be updated only if
the current `Inner alignment energy` record already carries correctly labeled
BPT and BPC. Snapshot, Cholesky, covariance-rank, memory-guard, and metrics-
schema failures require production fixes when the current executable contract
is unsafe or internally inconsistent.

## Cross-lane compatibility

Checkpoint key compatibility is preserved wherever the corresponding state
still exists. Removing dormant parameters is an intentional semantic change:
strict loading of an older checkpoint into a configuration that now omits
those parameters must either discard only the proven dormant keys through an
explicit migration record or fail with a targeted compatibility message. It
must not silently ignore arbitrary unexpected keys.

New configuration fields receive defaults through the existing serialized-
config migration path. Unknown current fields continue to fail in strict
artifact paths. Exact-type validation happens after legitimate migration, so
old schema defaults remain loadable while malformed current values fail.

The transport registry change preserves current parameter names, callable
behavior, pure flat transport, and configuration selection. It introduces no
new trainable path by default and no neural layer.

## Test strategy

Every sustained finding receives at least one focused regression. A combined
test may cover two findings only when one execution proves both contracts and
the assertions name both obligations. The RED command and failure reason are
recorded before the production patch; the same command must pass afterward.

Focused tests run after each lane. Integration then runs the ten original
failing node IDs together, the complete CPU suite without an extra `-q`, and
the CUDA-marked lane with `VFE3_TEST_DEVICE=cuda` and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. Counts and failures come only from JUnit
XML. A repository-wide AST parse and `git diff --check` close syntax and
whitespace obligations.

The clean starting JUnit record is
`C:\tmp\vfe3-fixall-baseline-20260720.xml`, SHA-256
`189A1ED681DFF172AD78E948CED7B1A02DF07FB8D19B2A56447FBAEFDED58192`.
It records 3,904 cases, 10 failures, 0 errors, 37 skips, and 343.956 seconds.
The ten baseline failures are:

1. `tests.test_2026_07_15_data_integrity_remediation::test_training_log_always_names_bits_per_token_when_bpc_is_available`
2. `tests.test_ablation_artifact_resume_20260712::test_missing_requested_diagnostics_output_forbids_contract_publication`
3. `tests.test_ablation_artifact_resume_20260712::test_run_single_terminal_merge_preserves_metadata_and_primary_val_ppl`
4. `tests.test_curated_geometry_math_20260709::test_prior_model_and_decode_variance_reads_share_guard`
5. `tests.test_diagnostics::test_attention_and_trace_reuse_snapshot_without_forward_replay`
6. `tests.test_estep_fixed_point_reporting_20260715::test_one_step_ahead_residual_is_distinct_from_configured_last_step`
7. `tests.test_report::test_generate_figures_reuses_one_same_token_snapshot`
8. `tests.test_report::test_generate_figures_memory_guard_uses_materialized_batch_peak`
9. `tests.test_run_diagnostics_2026_06_13::test_val_diagnostics_passes_explicit_diagonal_covariance_for_square_trace`
10. `tests.test_train::test_validation_finalizer_appends_to_existing_metrics_schema`

## Verification and evidence closure

The final claim ledger contains one claim for each of the twenty-one audit
findings and one aggregate claim that the ten-case baseline failure set is
closed. Code claims require current mechanical or reproduced-output evidence.
M10 also carries the analytic self-KL derivation; numerical agreement alone
does not prove the identity. H1 receives four views, including a skeptic and
an adjudicator, because it remains High until the bounded path is mechanically
verified.

An independent reviewer must inspect each lane's focused diff against this
specification before integration. Final closure requires no open
`CANDIDATE` or `LLM_SUPPORTED` state, a passing deterministic ledger
validator, exact current JUnit artifacts, and a clean task worktree. The
user's live WIP remains outside the task diff and is preserved during the
final fast-forward.

## Documentation and publication

All production and test changes are summarized in the existing
`docs/2026-07-20-edits.md`; there is only one dated edit document for the day.
The audit report remains the historical finding record and is not rewritten
to pretend the original defects never existed. A separate verification
appendix or remediation report may record closure if needed by the final
ledger.

The task branch is committed in reviewable increments, pushed, merged into
`main` only after verification, and pushed to `origin/main`. The live checkout
is fast-forwarded only after proving that incoming task paths do not overlap
the user's existing dirty paths. Task-owned JUnit, ledger, and temporary probe
artifacts are removed after their hashes and results are durably recorded.
