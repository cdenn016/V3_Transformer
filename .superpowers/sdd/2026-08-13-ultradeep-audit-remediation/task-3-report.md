# Task 3 Report: Truthful Scientific Certificates

Date: 2026-08-13
Base revision: `383f17ecb52397b16abe9da383e9b6c8281b9a9b`
Implementation commit: `beaaf7b32ea8667789b8dd41daa829a6141de9fb`
Status: implemented and focused CPU suites green

## Scope and Result

Task 3 makes the persisted certificate facets describe the active executable path without rejecting diagnostic configurations. It adds independent causality, transport-exactness, and conjunctive theory-purity facets; expands gauge purity across the registered active GL-breaking seams; and persists runtime-owned reflection scope and accessible orientation-component metadata. All preexisting report keys remain present.

The implementation consumes Task 2's immutable `ExecutableBuildMetadata` for BlockMLP. Transport, norm, attention-prior, clipping, adaptive-temperature, encoder, emission, and reflection facts come from active registrations, runtime evidence, or the live execution-selector config where no immutable runtime record exists. The report does not reconstruct BlockMLP from mutable config when executable metadata is supplied.

Diagnostic GL-breaking and noncausal configurations remain executable. They now report false on the corresponding facet rather than being rejected.

## TDD RED Evidence

Initial command:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'; $env:VFE3_TEST_DEVICE='cpu'; & 'C:/Python314/python.exe' -m pytest -q tests/test_ultradeep_remediation_certificates_20260813.py tests/test_run_artifacts.py tests/test_validated_geometry_numerics_20260713.py tests/test_tier12_attention.py --junitxml=.verification/remediation-2026-08-13/task-03-red.xml
```

Machine-readable result from `task-03-red.xml`: 109 tests, 21 failures, 0 errors, 1 skipped, 87 passed, 23.053 seconds. The failures were the intended missing-contract failures: absent additive report keys, incomplete gauge-purity predicates, absent causality registration metadata, missing reflection scope/count, and the old affirmative exactness behavior for empty covariant history.

Self-review added a second RED edge for a negative exactness observation followed by a missing row:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'; $env:VFE3_TEST_DEVICE='cpu'; & 'C:/Python314/python.exe' -m pytest -q tests/test_ultradeep_remediation_certificates_20260813.py -k certificate_truth_table --junitxml=.verification/remediation-2026-08-13/task-03-negative-red.xml
```

Machine-readable result: 16 tests, 1 failure, 0 errors, 0 skipped. The failure proved that a known negative observation was incorrectly classified as `unknown` when another history row lacked evidence. The minimal correction makes any finite negative observation `approximate`; a wholly affirmative complete history is `exact`; missing evidence without a negative observation is `unknown`.

## GREEN Evidence

Required focused suite:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'; $env:VFE3_TEST_DEVICE='cpu'; & 'C:/Python314/python.exe' -m pytest -q tests/test_run_artifacts.py tests/test_validated_geometry_numerics_20260713.py tests/test_tier12_attention.py tests/test_ultradeep_remediation_certificates_20260813.py tests/test_ablation_reporting.py --junitxml=.verification/remediation-2026-08-13/task-03-green.xml
```

Machine-readable result from `task-03-green.xml`: 121 tests, 0 failures, 0 errors, 1 skipped, 120 passed, 21.948 seconds. The skip is the preexisting slow `test_train_with_artifacts_writes_attention_pngs`, which requires `--runslow` and is not part of this focused CPU lane.

Registry/schema compatibility lane:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'; $env:VFE3_TEST_DEVICE='cpu'; & 'C:/Python314/python.exe' -m pytest -q tests/test_attention_prior.py tests/test_alpha_i.py tests/test_round3_registry_guards.py tests/test_round3_geometry_wiring.py tests/test_norms.py tests/test_transport.py --junitxml=.verification/remediation-2026-08-13/task-03-registry-green.xml
```

Machine-readable result: 89 tests, 0 failures, 0 errors, 0 skipped, 89 passed, 0.326 seconds. This independently covers duplicate-key fail-closed behavior, override/restore behavior, complete transport-registration restoration, and the preexisting norm/prior/transport public seams.

The self-review edge was rerun after the fix into `task-03-negative-green.xml`: 16 tests, 0 failures, 0 errors, 0 skipped.

## Implemented Certificate Semantics

- `on_causal_lm_path` is the conjunction of the beta and gamma active attention-prior registrations. Uniform, ALiBi-only, windowed, and bidirectional T5 paths remain executable and report false. Causality does not alter `on_gauge_pure_path`.
- `transport_exactness_status` emits only `exact`, `approximate`, `not_applicable`, or `unknown`. Flat is `not_applicable`; gauge-fixed or structurally noninvariant transport is `approximate`; a complete affirmative covariant history is `exact`; a finite negative observation is `approximate`; missing evidence is `unknown`. The covariant charted-link registration has no runtime exactness record and therefore reports `unknown`, never an inferred affirmative.
- `on_theory_pure_path` is the explicit conjunction of `gauge_pure_path`, `causal_lm_path`, and `transport_exact_when_applicable` in the new `theory_flags` map.
- `on_gauge_pure_path` retains its key and boolean type, but now includes active transport covariance class, both norm registrations including LayerNorm, positional rotation, model-channel coupling, parameterization, reflection, family/group invariance, HeadMixer, immutable BlockMLP class, fixed-coordinate covariance spectral cap, post-M-step chart cap, pullback trust region, fail-closed exponential validity bound, E-step phi retraction clipping, query-adaptive trace temperature, fixed-basis emission, and additive encoder control.
- Reflection persistence records configured and effective scope, effective subgroup, accessible blocks, accessible component count, and total represented component count. Multi-block `block_glk` with the existing reflection proposal reports `block_zero_only`, block 0, two accessible components, and `2**H` represented product-group components.

## Schema Compatibility

`test_certificate_schema_is_additive` asserts that the preexisting top-level keys remain present:

- `on_pure_path`
- `pure_flags`
- `gauge_flags`
- `on_gauge_pure_path`
- `config_toggles`
- `converged_stress`

It also asserts the additive keys:

- `on_causal_lm_path`
- `transport_exactness_status`
- `on_theory_pure_path`
- `causal_flags`
- `theory_flags`
- `reflection`

Ablation rows and `_CSV_COLUMNS` add the three certificate facets plus `reflection_effective_scope` and `reflection_accessible_component_count`; existing columns and result fields are retained. Failure rows use conservative false/`unknown` values.

The A01/A02 provenance waiver is preserved. No provenance-drift rejection behavior was added or changed.

## Files

- `vfe3/run_artifacts.py`: derives and persists the additive certificate facets and runtime reflection metadata.
- `vfe3/geometry/transport.py`: adds gauge and runtime-exactness metadata to complete active transport registrations.
- `vfe3/geometry/norms.py`: adds active norm gauge-class metadata with override/restore-safe callable identity.
- `vfe3/attention_prior.py`: adds active next-token causality metadata, including T5's live bidirectionality selector.
- `ablation.py`: propagates immutable BlockMLP metadata, runtime reflection scope, additive result keys, CSV columns, and conservative failure values.
- `tests/test_run_artifacts.py`: updates legacy exactness expectations and makes the gauge-pure fixture genuinely unclipped/uncapped.
- `tests/test_ultradeep_remediation_certificates_20260813.py`: adds the certificate truth table, causality independence, reflection scope/count, immutable BlockMLP, and additive-schema assertions.

The brief's `vfe3/model/free_energy.py` path was a typo; the live file is `vfe3/free_energy.py`. No change there was necessary because the active query-adaptive temperature selector already reaches the reporting seam. `tests/test_validated_geometry_numerics_20260713.py` and `tests/test_tier12_attention.py` were exercised unchanged.

## Warnings

The required green suite emitted only expected, preexisting, or intentionally exercised warnings:

- two parameter-motion warnings in `tests/test_run_artifacts.py` for `pos_phi_free` and `prior_bank.phi_embed` under zero-motion tiny fixtures;
- one intentional query-adaptive-temperature GL-breaking warning;
- two `decode_bias=True` inert `ConfigNotice` instances in ablation metadata tests;
- two grouped inert-setting `ConfigNotice` instances in the same ablation tests;
- two tiny-ablation parameter-motion warnings.

No test failure or error was hidden by these warnings.

## Self-Review

- Confirmed Task 3 touches only the seven implementation/test paths listed above plus this report; no numerical-precision, diagnostic, or training-loop remediation was implemented.
- Confirmed all status outputs are from the four-value exactness vocabulary.
- Confirmed empty/incomplete covariant evidence cannot publish `exact` and a known negative cannot be softened to `unknown`.
- Confirmed BlockMLP uses immutable executable metadata whenever present.
- Confirmed active prior and norm registry metadata follows the currently active callable across override/restore operations.
- Confirmed reflection scope comes from the instantiated PriorBank at both run finalizers and in actual ablation execution.
- Confirmed old report keys and diagnostic execution remain intact.
- Confirmed `git diff --check` and `git diff --cached --check` were clean before the implementation commit.

## Review Correction Pass

Review-fix date: 2026-08-13

Correction commit: `5b53186b4741abc79a2b4b1148e323dc15192fa8`

The Task 3 review identified one critical and three important gaps. All four are corrected:

- `theory_flags` now transparently contains every named legacy `pure_flags` requirement plus `gauge_pure_path`, `causal_lm_path`, and `transport_exact_when_applicable`. `on_theory_pure_path` is their full conjunction; no opaque legacy aggregate substitutes for the individual keys.
- Successful ablation rows pass `artifacts.history` into certificate derivation. Fresh-result and cached-result merges fill only absent defaults and preserve persisted execution-derived evidence. Aggregation validates the persisted exactness vocabulary and runtime-derived field types instead of comparing them to empty-history defaults.
- The existing `tests/test_reporting_additions.py` consumer fixture is genuinely gauge-pure: no fixed spectral cap, a fail-closed pre-clamp transport bound, inert clipping/adaptive/emission/encoder seams, and explicit compatible metadata. It asserts the old schema as a subset plus the additive certificate keys.
- The gauge truth table now pins the complete 19-key `gauge_flags` set. One active seam is flipped per row, including both norms, spectral and M-step caps, pullback trust, missing/over-limit exponential bounds, E-step phi retraction, adaptive trace temperature, fixed-basis emission, additive encoding, transport metadata, immutable BlockMLP metadata, and every preexisting gauge facet. Transport and norm override/restore behavior is exercised against the active registration/callable.

Theory tests flip each enumerated legacy requirement and assert the exact failed `theory_flags` key set. The three additive facets also have exact failed-key cases. Transport exactness cannot be made inapplicable-false while retaining the legacy `flat_transport` requirement, so the incomplete covariant case transparently reports both failed keys rather than concealing that logical coupling.

### Review RED

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'; $env:VFE3_TEST_DEVICE='cpu'; & 'C:/Python314/python.exe' -m pytest -q tests/test_ultradeep_remediation_certificates_20260813.py tests/test_reporting_additions.py::test_pure_path_report_structure_and_flags tests/test_ablation_tackon.py::test_success_and_cached_resume_preserve_runtime_certificate_history --junitxml=.verification/remediation-2026-08-13/task-03-review-red.xml
```

Machine-readable result: 67 tests, 18 failures, 0 errors, 0 skipped. The expected failures were 11 legacy theory-requirement rows, one transparent theory-key-set assertion, three successful history cases, and three fresh/cache-resume cases.

### Review GREEN

Focused correction lane: 67 tests, 67 passed, 0 failed/errors/skipped (`task-03-review-focused-green.xml`).

Expanded reporting and resume consumers: 184 tests, 184 passed, 0 failed/errors/skipped (`task-03-review-consumers-green.xml`).

Full expanded Task 3 lane:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'; $env:VFE3_TEST_DEVICE='cpu'; & 'C:/Python314/python.exe' -m pytest -q tests/test_run_artifacts.py tests/test_validated_geometry_numerics_20260713.py tests/test_tier12_attention.py tests/test_ultradeep_remediation_certificates_20260813.py tests/test_ablation_reporting.py tests/test_reporting_additions.py tests/test_ablation_tackon.py tests/test_ablation_artifact_resume_20260712.py --junitxml=.verification/remediation-2026-08-13/task-03-review-full-green.xml
```

Machine-readable result: 274 tests, 273 passed, 0 failures, 0 errors, 1 expected slow skip. The skipped test remains `test_train_with_artifacts_writes_attention_pngs`, which requires `--runslow`.

Active registry compatibility: 89 tests, 89 passed, 0 failures/errors/skips (`task-03-review-registry-green.xml`).

Warnings remained expected and disclosed: intentional query-adaptive GL-breaking; inert-setting `ConfigNotice` instances in fixture sweeps; tiny-run parameter-motion warnings; the deliberate covariant/full-family numerical warning in the three certificate-history fixtures; and the preexisting one-step resume config-drift warning. No warning hid a failure or error.

### Review Self-Check

- Negative runtime exactness evidence still dominates missing rows (`approximate`); incomplete evidence without a negative remains `unknown`; complete affirmative history is `exact`.
- Cache reuse preserves persisted runtime-derived certificate values and refuses malformed/missing runtime-derived fields during aggregation.
- Pre-run and failure rows remain conservatively reconstructed without runtime history.
- Causality remains independent of gauge purity; reflection metadata and the four-value exactness enum are unchanged.
- No diagnostic configuration was rejected, and the A01/A02 provenance waiver remains unchanged.
