# Ultra-Deep Audit Remediation Implementation Plan

> **Required sub-skill:** Use `superpowers:subagent-driven-development` to execute this plan task by task, with one implementation agent and one review pass per task. If subagent execution is unavailable, use `superpowers:executing-plans` and preserve the same RED/GREEN/review gates.

**Goal:** Repair all twenty-one sustained findings in the 2026-07-20 ultra-deep audit, resolve all ten clean CPU baseline failures, and close the result with current CPU, RTX 5090 CUDA, syntax, source-review, and validated claim-ledger evidence.

**Architecture:** Keep the existing registries and public mathematical paths intact while strengthening each failing boundary at its narrowest owner. Data/configuration contracts are repaired before runtime construction; artifact identity and experiment completeness are validated before publication; model, inference, transport, and Gaussian changes preserve existing public names and exact paths. Every production patch begins with a focused regression observed failing at revision `aa5aceab7844d48c800d72397c0ce4550c567ba1` and ends with an independent diff review.

**Tech Stack:** Python 3.14 CPU test environment at `C:\Python314\python.exe`; PyTorch and pytest; Python 3.14 AST/compile checks; RTX 5090 CUDA test environment at `C:\anaconda\python.exe`; JUnit XML for machine-readable counts; Git worktree branch `codex/fix-ultradeep-audit-20260720`; verification ledger at `.verification/ledger.json`.

**Global Constraints:** Work only in `C:\tmp\V3_Transformer_fix_all_audit_20260720`. Do not touch the user's dirty live checkout, delete tests, weaken assertions to obtain green output, change active configuration values, add neural layers, add CLI parsing, or remove a theoretically pure path. Preserve float32 public behavior unless a named numerical kernel uses an internal float64 island. Do not add another dated edit document; append all implementation and verification facts to `docs/2026-07-20-edits.md`. Every test count must come from current JUnit XML. Every task must record its RED failure before changing production code.

---

## Task 1: Bind binary token data and sidecars to one immutable source

**Finding coverage:** M1 and M2.

**Files:**

- Create: `tests/test_audit_data_contracts_20260720.py`
- Modify: `vfe3/data/datasets.py`
- Modify: `docs/2026-07-20-edits.md`

- [ ] **Step 1: Write the mutable-memmap regression.**

  Add `test_uncapped_binary_cache_is_owned_after_identity_binding`. Construct a small valid binary cache and metadata file, load it through `TokenWindows`, mutate the payload with a separate writable mapping, and assert that the already-loaded dataset returns its original window and still reports the recorded identity. Assert the loaded tensor is in the binary cache's native integer dtype rather than eagerly expanded to `torch.int64`.

- [ ] **Step 2: Write the resolved-sidecar regression.**

  Add `test_symlinked_binary_cache_uses_one_resolved_sidecar_family`. Put a payload target and valid metadata in one directory, a symlink and conflicting metadata in another, and assert that `cache_source_identity` and `load_cached_tokens` either use the target-side metadata consistently or reject the split. Skip only when the platform denies symlink creation.

- [ ] **Step 3: Run and record RED.**

  Run:

  ```powershell
  C:\Python314\python.exe -m pytest tests/test_audit_data_contracts_20260720.py --junitxml=C:\tmp\vfe3-fixall-task01-red.xml
  ```

  Expected: both tests fail against the current live memmap/unresolved-sidecar behavior. Record the exact failure names and XML fields in the dated edit document.

- [ ] **Step 4: Resolve one source family.**

  Introduce a private helper that returns the resolved payload and all sidecar paths from the resolved parent. Both `cache_source_identity` and `load_cached_tokens` must call this helper. The contract should be equivalent to:

  ```python
  def _resolved_binary_cache_paths(source: Path) -> Tuple[Path, Path, Path]:
      payload = source.resolve(strict=True)
      return payload, payload.with_suffix(".json"), payload.with_suffix(".provenance.json")
  ```

  Match the repository's actual metadata suffixes and signature formatting; do not invent alternate sidecars.

- [ ] **Step 5: Make identity-bound uncapped data owned.**

  In `_load_identity_bound_tokens`, retain the identity-before/load/identity-after guard. When the resolved source is a binary cache and no token limit was requested, clone the loaded native-dtype tensor before returning it. Do not expand the corpus to int64, and do not change capped or `.pt` behavior.

- [ ] **Step 6: Run GREEN and adjacent cache tests.**

  ```powershell
  C:\Python314\python.exe -m pytest tests/test_audit_data_contracts_20260720.py tests/test_data.py tests/test_source_identity_snapshot_performance_20260716.py --junitxml=C:\tmp\vfe3-fixall-task01-green.xml
  ```

  Expected: zero failures and zero errors. Read the JUnit attributes rather than the terminal progress line.

- [ ] **Step 7: Review and commit.**

  Inspect `git diff -- vfe3/data/datasets.py tests/test_audit_data_contracts_20260720.py docs/2026-07-20-edits.md`, run `git diff --check`, obtain an independent source review for M1/M2, then commit:

  ```powershell
  git add vfe3/data/datasets.py tests/test_audit_data_contracts_20260720.py docs/2026-07-20-edits.md
  git commit -m "fix: bind cached tokens to immutable source identity"
  ```

## Task 2: Enforce exact configuration and loader types

**Finding coverage:** M7 configuration half, M8 configuration half, M12, M13, M17, and L3. This task also introduces the M16 opt-in field, whose runtime use is Task 11.

**Files:**

- Create: `tests/test_audit_config_contracts_20260720.py`
- Modify: `vfe3/config.py`
- Modify: `vfe3/data/datasets.py`
- Modify: `docs/2026-07-20-edits.md`

- [ ] **Step 1: Write exact-type parameterized regressions.**

  Add tests that inspect `dataclasses.fields(VFE3Config)` and, for every field annotated exactly `bool` or `int`, replace the default with a wrong plain type that would otherwise survive truthiness or comparison. Require `TypeError` naming the field. Include explicit cases for `include_attention_entropy="false"`, `n_layers=True`, `n_layers=1.5`, and `e_steps_backprop_last=0.5`. Keep optional and structured fields out of this generic loop and cover them only through their existing validators.

- [ ] **Step 2: Write cross-field and irrep regressions.**

  Add `test_additive_encoder_rejects_pullback_group_update` and `test_irrep_multiplicity_rejects_bool`. The former constructs the smallest otherwise-valid configuration with `encode_mode="per_token_additive"` and `m_phi_update_mode="pullback_group"`; the latter uses a supported irrep group with multiplicity `True`.

- [ ] **Step 3: Write public loader exact-boolean regressions.**

  Add parameterized checks that `make_dataloader(..., shuffle="false")` and `make_dataloader(..., drop_last=1)` raise before `TokenWindows` or `DataLoader` is constructed.

- [ ] **Step 4: Run and record RED.**

  ```powershell
  C:\Python314\python.exe -m pytest tests/test_audit_config_contracts_20260720.py --junitxml=C:\tmp\vfe3-fixall-task02-red.xml
  ```

  Expected: current permissive values and incompatible additive/group-chart combination are accepted, so the new assertions fail.

- [ ] **Step 5: Add centralized plain-type validators.**

  At the beginning of `VFE3Config.__post_init__`, before branches use configuration values, iterate over dataclass fields and require exact types for annotations exactly `bool` and `int`. Use `type(value) is bool` and `type(value) is int`, not `isinstance`. Preserve all existing range and semantic validators. Serialized compatibility migration remains responsible for explicit old-schema normalization before construction.

- [ ] **Step 6: Add semantic guards and the opt-in field.**

  Reject `per_token_additive` plus `pullback_group` with a targeted error. Change the irrep multiplicity predicate to a positive plain integer. Add `evaluate_zero_e_steps_counterfactual: bool = False` near other finalization controls; let ordinary dataclass serialization and exact-type validation own it.

- [ ] **Step 7: Guard loader booleans before side effects.**

  At the top of `make_dataloader`, reject nonexact `shuffle` and `drop_last` values before loading token data or selecting a sampler.

- [ ] **Step 8: Run GREEN and configuration compatibility tests.**

  ```powershell
  C:\Python314\python.exe -m pytest tests/test_audit_config_contracts_20260720.py tests/test_config.py tests/test_checkpoint_resume.py tests/test_data.py --junitxml=C:\tmp\vfe3-fixall-task02-green.xml
  ```

  Expected: zero failures/errors, including serialized checkpoint configuration migration and data-loader behavior.

- [ ] **Step 9: Review and commit.**

  Verify the generic loop does not accidentally constrain `Optional[int]`, enum-like strings, or migrated legacy values. Obtain independent review, then commit:

  ```powershell
  git add vfe3/config.py vfe3/data/datasets.py tests/test_audit_config_contracts_20260720.py docs/2026-07-20-edits.md
  git commit -m "fix: enforce exact public configuration types"
  ```

## Task 3: Validate worker JSON and process commands before coercion

**Finding coverage:** L1 and L2.

**Files:**

- Create: `tests/test_audit_process_contracts_20260720.py`
- Modify: `vfe3/viz/figure_worker.py`
- Modify: `vfe3/process_utils.py`
- Modify: `docs/2026-07-20-edits.md`

- [ ] **Step 1: Add request-schema and command regressions.**

  Test finalize-worker request values such as `allow_large="false"`, numeric booleans, nonstring paths/devices, and wrong nullable values. Require rejection before any render function runs. Test `run_process_tree("python")`, `run_process_tree(b"python")`, an empty sequence, and a sequence containing `""`; mock process creation and assert it is never called.

- [ ] **Step 2: Run and record RED.**

  ```powershell
  C:\Python314\python.exe -m pytest tests/test_audit_process_contracts_20260720.py --junitxml=C:\tmp\vfe3-fixall-task03-red.xml
  ```

- [ ] **Step 3: Implement narrow validators.**

  Add exact JSON-field helpers in `figure_worker.py` and use them in finalize mode before conversion. In `run_process_tree`, reject `str`/`bytes`, require a nonempty `Sequence[str]`, and require every element to be nonempty before job/process creation.

- [ ] **Step 4: Run GREEN and neighboring tests.**

  ```powershell
  C:\Python314\python.exe -m pytest tests/test_audit_process_contracts_20260720.py tests/test_figures_tail.py tests/test_process_utils.py --junitxml=C:\tmp\vfe3-fixall-task03-green.xml
  ```

- [ ] **Step 5: Review and commit.**

  Confirm valid existing JSON produced by `run_artifacts.py` still passes, then independently review and commit:

  ```powershell
  git add vfe3/viz/figure_worker.py vfe3/process_utils.py tests/test_audit_process_contracts_20260720.py docs/2026-07-20-edits.md
  git commit -m "fix: validate worker and process boundaries"
  ```

## Task 4: Unify selected-bundle and checkpoint tensor integrity

**Finding coverage:** M3 and M4.

**Files:**

- Create: `tests/test_audit_artifact_integrity_20260720.py`
- Modify: `vfe3/run_artifacts.py`
- Modify: `docs/2026-07-20-edits.md`

- [ ] **Step 1: Add identity-bypass regressions.**

  Build a syntactically valid best-model bundle with the expected configuration fingerprint but mismatched executable-code identity, then a second with mismatched validation-data identity. Call the finalization preflight path and assert that neither reaches `load_state_dict` or test evaluation.

- [ ] **Step 2: Add nonfinite state regressions.**

  Parameterize NaN, positive infinity, and negative infinity in an otherwise shape/dtype/layout-correct floating state tensor. Exercise raw checkpoint resume and best-model bundle validation. Require rejection before any live state copy. Add a finite control and retain integer/bool structural acceptance where applicable.

- [ ] **Step 3: Run and record RED.**

  ```powershell
  C:\Python314\python.exe -m pytest tests/test_audit_artifact_integrity_20260720.py --junitxml=C:\tmp\vfe3-fixall-task04-red.xml
  ```

- [ ] **Step 4: Create one nonmutating selected-bundle boundary.**

  Extend or wrap `_validate_best_model_mapping` so schema, semantic configuration, tensor contract, executable-code identity, and validation-data identity are all checked before returning an owned candidate state. Make `finalize_run`, `_restore_best_selection`, `_preflight_best_selection`, publication round-trips, and direct bundle reads use that same boundary rather than local subsets.

- [ ] **Step 5: Add shared model-state finiteness validation.**

  In `_validate_checkpoint_model_state`, after structural checks and before copying, require `torch.isfinite(tensor).all()` for floating and complex tensors. Ensure every selected-bundle path uses this function. Error text must name the offending state key and artifact class without dumping values.

- [ ] **Step 6: Run GREEN and artifact/resume suites.**

  ```powershell
  C:\Python314\python.exe -m pytest tests/test_audit_artifact_integrity_20260720.py tests/test_checkpoint_resume.py tests/test_run_artifacts.py --junitxml=C:\tmp\vfe3-fixall-task04-green.xml
  ```

- [ ] **Step 7: Review and commit.**

  Review mutation ordering and legacy-bundle policy. No arbitrary unexpected key may be silently discarded. Obtain independent review and commit:

  ```powershell
  git add vfe3/run_artifacts.py tests/test_audit_artifact_integrity_20260720.py docs/2026-07-20-edits.md
  git commit -m "fix: fail closed on stale or nonfinite model artifacts"
  ```

## Task 5: Make ablation flags and growing-sequence completion exact

**Finding coverage:** M11 and M14, plus baseline failures 2 and 3.

**Files:**

- Create: `tests/test_audit_ablation_contracts_20260720.py`
- Modify: `ablation.py`
- Modify only if the existing assertion is proven stale: `tests/test_ablation_artifact_resume_20260712.py`
- Modify: `docs/2026-07-20-edits.md`

- [ ] **Step 1: Reproduce the two baseline nodes alone.**

  ```powershell
  C:\Python314\python.exe -m pytest tests/test_ablation_artifact_resume_20260712.py::test_missing_requested_diagnostics_output_forbids_contract_publication tests/test_ablation_artifact_resume_20260712.py::test_run_single_terminal_merge_preserves_metadata_and_primary_val_ppl --junitxml=C:\tmp\vfe3-fixall-task05-baseline-red.xml
  ```

  Record each stack's first incorrect boundary. Do not edit either assertion until the current parameter-matched artifact contract and persisted metadata are inspected.

- [ ] **Step 2: Add raw-flag regressions.**

  Test every field in the diagnostic flag set with raw `"false"`, `0`, and `1`. Require a field-named error before `_validated_diagnostic_flags`, `_sweep_diagnostic_request`, `run_single`, or persisted metadata converts the value.

- [ ] **Step 3: Add extrapolation-completeness regressions.**

  Supply requested lengths such as `[128, 256, 512]` with finite successes only at 128 and 256 and a recorded failure at 512. Assert that the aggregate is incomplete, retains the 512 failure reason, and cannot publish a completed fitted domain. Add controls for two distinct finite points including the largest requested N and for duplicate N values.

- [ ] **Step 4: Run and record audit RED.**

  ```powershell
  C:\Python314\python.exe -m pytest tests/test_audit_ablation_contracts_20260720.py --junitxml=C:\tmp\vfe3-fixall-task05-audit-red.xml
  ```

- [ ] **Step 5: Validate booleans before construction.**

  Replace truthiness conversion with one exact-boolean helper used by the sweep validator, diagnostic request builder, `run_single` call assembly, and persisted flag record. Execution and metadata must receive the same validated value.

- [ ] **Step 6: Strengthen extrapolation aggregation.**

  Preserve a record for every requested N, including status and failure reason. Mark complete only when the largest requested length succeeds, at least two distinct lengths have finite metrics, and all configured mandatory lengths succeed. If batch size is reduced with N, persist the effective batch size per point.

- [ ] **Step 7: Repair the baseline root causes.**

  Make requested-but-missing diagnostics prevent contract publication. Preserve terminal metadata and primary validation PPL when `run_single` overlays diagnostic outputs. If either existing test encodes an obsolete parameter-matched scope rather than the executable public contract, update only that assertion and explain the exact stale expectation in the dated edit document.

- [ ] **Step 8: Run GREEN.**

  ```powershell
  C:\Python314\python.exe -m pytest tests/test_audit_ablation_contracts_20260720.py tests/test_ablation_artifact_resume_20260712.py --junitxml=C:\tmp\vfe3-fixall-task05-green.xml
  ```

- [ ] **Step 9: Review and commit.**

  Independently review publication fail-closed behavior, raw versus persisted flags, and mandatory tail semantics. Commit:

  ```powershell
  git add ablation.py tests/test_audit_ablation_contracts_20260720.py tests/test_ablation_artifact_resume_20260712.py docs/2026-07-20-edits.md
  git commit -m "fix: harden ablation completion contracts"
  ```

  If the existing test file was not changed, omit it from `git add`.

## Task 6: Bound validation diagnostics before dense decode

**Finding coverage:** H1, plus baseline failures 5, 7, 8, and 9.

**Files:**

- Create: `tests/test_audit_diagnostic_memory_20260720.py`
- Modify: `vfe3/train.py`
- Modify only if snapshot ownership requires it: `vfe3/model/model.py`
- Modify as root-cause evidence requires: `vfe3/run_artifacts.py`
- Modify only when an expectation is proven stale: `tests/test_diagnostics.py`, `tests/test_report.py`, `tests/test_run_diagnostics_2026_06_13.py`
- Modify: `docs/2026-07-20-edits.md`

- [ ] **Step 1: Reproduce the four diagnostic baseline failures.**

  ```powershell
  C:\Python314\python.exe -m pytest tests/test_diagnostics.py::test_attention_and_trace_reuse_snapshot_without_forward_replay tests/test_report.py::test_generate_figures_reuses_one_same_token_snapshot tests/test_report.py::test_generate_figures_memory_guard_uses_materialized_batch_peak tests/test_run_diagnostics_2026_06_13.py::test_val_diagnostics_passes_explicit_diagonal_covariance_for_square_trace --junitxml=C:\tmp\vfe3-fixall-task06-baseline-red.xml
  ```

  Trace call counts, token shapes, tensor storage, and covariance rank at the first divergent boundary.

- [ ] **Step 2: Add the active-shape boundary regression.**

  Use a spy model and a first validation batch shaped `(256, 128)`. Call `_val_diagnostics` and assert `build_diagnostic_snapshot` receives exactly `(1, 128)`. Assert ordinary `evaluate` still receives and scores the configured validation population. Assert all diagnostic consumers reuse the returned snapshot without replaying `forward_beliefs`.

- [ ] **Step 3: Add allocation arithmetic and CUDA peak regressions.**

  On CPU, assert the bounded logits copy pair formula is `2 * 1 * 128 * 50257 * 4 = 51,463,168` bytes and the former full-batch formula is `13,174,571,008` bytes. Mark a CUDA test that resets peak memory, runs a bounded surrogate through the actual snapshot boundary, and requires peak growth below a declared multiple of the bounded formula. It must not attempt the historical 12.2698-GiB allocation.

- [ ] **Step 4: Run and record audit RED.**

  ```powershell
  C:\Python314\python.exe -m pytest tests/test_audit_diagnostic_memory_20260720.py --junitxml=C:\tmp\vfe3-fixall-task06-audit-red.xml
  ```

- [ ] **Step 5: Slice before decode and preserve one snapshot.**

  In `_val_diagnostics`, derive `diagnostic_tokens = val_tok[:1]` before `build_diagnostic_snapshot`. Pass the one snapshot through attention, trace, covariance, and figure/report consumers. Do not slice the ordinary validation loader or change headline CE/PPL selection. Preserve explicit diagonal covariance for square-trace diagnostics.

- [ ] **Step 6: Repair snapshot memory accounting.**

  Ensure report/figure guards calculate peak memory from the materialized diagnostic batch and do not trigger a second forward replay. Only modify `DiagnosticSnapshot` if one durable dense logits field is proven unnecessary to all consumers; if removed or made optional, cover serialization/consumer compatibility explicitly.

- [ ] **Step 7: Run GREEN on all diagnostic tests.**

  ```powershell
  C:\Python314\python.exe -m pytest tests/test_audit_diagnostic_memory_20260720.py tests/test_diagnostics.py tests/test_report.py tests/test_run_diagnostics_2026_06_13.py --junitxml=C:\tmp\vfe3-fixall-task06-green.xml
  ```

- [ ] **Step 8: Obtain four-view H1 review and commit.**

  Require implementation review, performance/memory review, an adversarial skeptic, and an adjudicator to inspect the exact boundary and test evidence. Record what would falsify closure. Commit:

  ```powershell
  git add vfe3/train.py vfe3/model/model.py vfe3/run_artifacts.py tests/test_audit_diagnostic_memory_20260720.py tests/test_diagnostics.py tests/test_report.py tests/test_run_diagnostics_2026_06_13.py docs/2026-07-20-edits.md
  git commit -m "fix: bound held-out diagnostic decode memory"
  ```

  Omit unchanged files from `git add`.

## Task 7: Fail closed on invalid targets, empty evaluation, and fractional E-step depth

**Finding coverage:** M5, M6, and M7 runtime half.

**Files:**

- Create: `tests/test_audit_runtime_semantics_20260720.py`
- Modify: `vfe3/model/prior_bank.py`
- Modify: `vfe3/train.py`
- Modify: `vfe3/inference/e_step.py`
- Modify: `docs/2026-07-20-edits.md`

- [ ] **Step 1: Add invalid-target parity tests.**

  Parameterize every registered fused chunked decoder and both invalid bounds: `-1` and `V`. Require the same failure class as the corresponding dense PyTorch cross-entropy boundary. Add valid class `0`, valid class `V-1`, and `ignore_index=-100` controls with dense/chunked loss parity.

- [ ] **Step 2: Add empty-evaluation tests.**

  Exercise an empty loader and an all-ignored loader. Require a targeted exception before CE, PPL, BPT, or BPC is returned. Assert the exception cannot be confused with a perfect metric record.

- [ ] **Step 3: Add the defensive E-step boundary test.**

  Call `e_step` directly with `e_steps_backprop_last=0.5` and `True`, bypassing `VFE3Config`, and require exact-type rejection before iteration policy is computed. Retain gradient controls for zero and one.

- [ ] **Step 4: Run and record RED.**

  ```powershell
  C:\Python314\python.exe -m pytest tests/test_audit_runtime_semantics_20260720.py --junitxml=C:\tmp\vfe3-fixall-task07-red.xml
  ```

- [ ] **Step 5: Add one target validator and use it in all fused kernels.**

  Add a private `PriorBank` helper or module function that checks once per fused CE call that every nonignored target satisfies `0 <= target < vocab_size`. Call it before chunk reduction in diagonal, full, linear, expected-likelihood, and family fused kernels.

- [ ] **Step 6: Reject undefined evaluation and malformed E-step depth.**

  In `evaluate`, raise after aggregation when `total_tok == 0`, before metric construction. At the public `e_step` boundary, require `type(e_steps_backprop_last) is int` and the existing nonnegative constraint.

- [ ] **Step 7: Run GREEN and adjacent decode/evaluation tests.**

  ```powershell
  C:\Python314\python.exe -m pytest tests/test_audit_runtime_semantics_20260720.py tests/test_prior_bank.py tests/test_train.py tests/test_e_step.py --junitxml=C:\tmp\vfe3-fixall-task07-green.xml
  ```

- [ ] **Step 8: Review and commit.**

  Review ignored-target normalization and caller handling of empty diagnostics. Commit after independent review:

  ```powershell
  git add vfe3/model/prior_bank.py vfe3/train.py vfe3/inference/e_step.py tests/test_audit_runtime_semantics_20260720.py docs/2026-07-20-edits.md
  git commit -m "fix: reject undefined runtime metric inputs"
  ```

## Task 8: Move trainable transport state behind the registry

**Finding coverage:** M9.

**Files:**

- Create: `tests/test_audit_transport_registry_20260720.py`
- Modify: `vfe3/geometry/transport.py`
- Modify: `vfe3/model/model.py`
- Modify: `vfe3/inference/e_step.py`
- Modify only if optimizer grouping consumes literal connection names: `vfe3/train.py`, `vfe3/gauge_optim.py`
- Modify: `docs/2026-07-20-edits.md`

- [ ] **Step 1: Write a synthetic stateful registration test.**

  Register a uniquely named test transport whose registration declares a small trainable state mapping such as `{"connection_probe": parameter}`. Construct `VFEModel` using only the public registration seam. Assert the parameter is registered under the declared stable key, appears in `state_dict` and the optimizer exactly once, and reaches the transport callable without adding its name to model or E-step source.

- [ ] **Step 2: Add compatibility tests for current transports.**

  For existing stateful modes, assert `connection_W`, `connection_M`, and `connection_L` state-dict names remain unchanged. For `flat`, assert no trainable transport state is constructed and the pure path output is unchanged.

- [ ] **Step 3: Run and record RED.**

  ```powershell
  C:\Python314\python.exe -m pytest tests/test_audit_transport_registry_20260720.py --junitxml=C:\tmp\vfe3-fixall-task08-red.xml
  ```

- [ ] **Step 4: Extend `TransportRegistration`.**

  Add optional metadata for a trainable-state builder and declared serialization keys. Define a typed transport-state mapping. Existing registrations supply builders that reproduce current parameter shapes, initialization, and names; stateless registrations use no builder.

- [ ] **Step 5: Make construction and calls generic.**

  `VFEModel` obtains the selected registration, calls its state builder, and registers each returned parameter generically. Model/E-step transport calls pass one `transport_state` mapping. Remove literal mode/name branches only where the new registration replaces them; preserve unrelated transport behavior.

- [ ] **Step 6: Preserve checkpoint and optimizer contracts.**

  Ensure state dict keys and parameter groups for current transports are byte-for-byte named as before. A new registration must require no central source edit. No default route gains trainable state.

- [ ] **Step 7: Run GREEN and transport/gauge tests.**

  ```powershell
  C:\Python314\python.exe -m pytest tests/test_audit_transport_registry_20260720.py tests/test_transport.py tests/test_gauge_optim.py tests/test_checkpoint_resume.py --junitxml=C:\tmp\vfe3-fixall-task08-green.xml
  ```

- [ ] **Step 8: Review and commit.**

  Obtain implementation and gauge-theory reviews. The reviewer must verify that flat transport remains state-free and current state names are stable. Commit:

  ```powershell
  git add vfe3/geometry/transport.py vfe3/model/model.py vfe3/inference/e_step.py vfe3/train.py vfe3/gauge_optim.py tests/test_audit_transport_registry_20260720.py docs/2026-07-20-edits.md
  git commit -m "refactor: register trainable transport state"
  ```

  Omit unchanged files from `git add`.

## Task 9: Stabilize full-Gaussian self-KL in a float64 island

**Finding coverage:** M10.

**Files:**

- Create: `tests/test_audit_full_gaussian_numerics_20260720.py`
- Modify: `vfe3/families/gaussian.py`
- Modify: `docs/2026-07-20-edits.md`

- [ ] **Step 1: Write the analytic obligation in the test.**

  For identical `q=p=N(mu,Sigma)`, record the exact identity
  `KL(q||q) = 0.5 * (tr(I) + 0 - K + logdet(Sigma) - logdet(Sigma)) = 0`.
  Under a shared covariance variable, its total derivative is zero because the trace and log-determinant differentials cancel.

- [ ] **Step 2: Reproduce the conditioned float32 failure.**

  Recreate the audited seed-17 4-by-4 SPD operand near condition number 950,197. Assert the pre-fix float32 route violates tight self-KL and shared-gradient tolerances while a float64 oracle satisfies them. Also include twenty deterministic seeds to prevent overfitting to one matrix.

- [ ] **Step 3: Run and record RED.**

  ```powershell
  C:\Python314\python.exe -m pytest tests/test_audit_full_gaussian_numerics_20260720.py --junitxml=C:\tmp\vfe3-fixall-task09-red.xml
  ```

- [ ] **Step 4: Compute the full-SPD kernel in float64.**

  In `FullGaussian.renyi_closed_form`, promote float32 and lower operands together to float64 before Cholesky/factorization, solve, trace, mean quadratic, and log-determinant subtraction. Cast the returned divergence to the existing public result dtype only after the stable expression. Leave `DiagonalGaussian` byte-unchanged.

- [ ] **Step 5: Test values and both gradient interpretations.**

  Require near-zero self-KL and shared-variable covariance gradient. Also test separate q/p covariance variables, where gradients need not individually vanish, against a float64 autograd oracle. Retain alpha-limit and nonidentical-distribution behavior.

- [ ] **Step 6: Run GREEN and geometry numerical tests.**

  ```powershell
  C:\Python314\python.exe -m pytest tests/test_audit_full_gaussian_numerics_20260720.py tests/test_families.py tests/test_curated_geometry_math_20260709.py --junitxml=C:\tmp\vfe3-fixall-task09-green.xml
  ```

- [ ] **Step 7: Obtain derivation and numerical reviews, then commit.**

  A mathematical reviewer must check the derivative argument; a numerical reviewer must inspect tolerances and condition numbers. Commit:

  ```powershell
  git add vfe3/families/gaussian.py tests/test_audit_full_gaussian_numerics_20260720.py docs/2026-07-20-edits.md
  git commit -m "fix: stabilize full Gaussian KL cancellation"
  ```

## Task 10: Remove dormant prior tables from active capacity accounting

**Finding coverage:** M15, plus baseline failure 4.

**Files:**

- Create: `tests/test_audit_prior_bank_routing_20260720.py`
- Modify: `vfe3/model/prior_bank.py`
- Modify: `vfe3/model/model.py`
- Modify: `vfe3/train.py`
- Modify only if the existing expectation is proven stale: `tests/test_curated_geometry_math_20260709.py`
- Modify: `docs/2026-07-20-edits.md`

- [ ] **Step 1: Reproduce the variance-guard baseline failure.**

  ```powershell
  C:\Python314\python.exe -m pytest tests/test_curated_geometry_math_20260709.py::test_prior_model_and_decode_variance_reads_share_guard --junitxml=C:\tmp\vfe3-fixall-task10-baseline-red.xml
  ```

  Trace the source guard shared by prior/model/decode variance reads before deciding whether implementation or assertion drifted.

- [ ] **Step 2: Add route-consumption regressions.**

  Under model-channel prior plus linear decode, assert `mu_embed` and the base variance table attributes exist as registered `None`, contribute zero storage, do not occur in `named_parameters`, optimizer groups, or realized parameter counts, and do not affect outputs. For every route that consumes token priors, assert both tables exist and initialization remains deterministic under the same seed.

- [ ] **Step 3: Add checkpoint migration tests.**

  Load a legacy state dict containing the now-dormant keys into the exact route that omits them. Require an explicit migration that discards only the named, proven-dormant keys or a targeted compatibility error. Arbitrary unexpected keys must still fail. A route that consumes the tables must continue strict loading.

- [ ] **Step 4: Run and record audit RED.**

  ```powershell
  C:\Python314\python.exe -m pytest tests/test_audit_prior_bank_routing_20260720.py --junitxml=C:\tmp\vfe3-fixall-task10-audit-red.xml
  ```

- [ ] **Step 5: Gate construction on actual consumers.**

  Compute explicit booleans for base-mean and base-variance consumption from encode/prior/decode registrations. Register `None` for an unconsumed table and construct an `nn.Parameter` only when consumed. Preserve attribute names and the initialization order for routes that still allocate them.

- [ ] **Step 6: Make optimizer and reporting generic over present parameters.**

  Ensure optimizer grouping, parameter reporting, and realized-capacity counting consume `named_parameters()` or explicit non-`None` checks. Remove no unrelated dormant/pre-existing code.

- [ ] **Step 7: Repair the shared variance guard.**

  Make prior, model, and decode reads use the same executable guard. If the baseline assertion was tied to an obsolete comment or literal source string, update it to test behavior and record why.

- [ ] **Step 8: Run GREEN.**

  ```powershell
  C:\Python314\python.exe -m pytest tests/test_audit_prior_bank_routing_20260720.py tests/test_curated_geometry_math_20260709.py tests/test_prior_bank.py tests/test_train.py --junitxml=C:\tmp\vfe3-fixall-task10-green.xml
  ```

- [ ] **Step 9: Review and commit.**

  Review parameter counts, optimizer membership, checkpoint compatibility, and seed/init order. Commit:

  ```powershell
  git add vfe3/model/prior_bank.py vfe3/model/model.py vfe3/train.py tests/test_audit_prior_bank_routing_20260720.py tests/test_curated_geometry_math_20260709.py docs/2026-07-20-edits.md
  git commit -m "fix: exclude unconsumed prior tables"
  ```

  Omit unchanged files from `git add`.

## Task 11: Make the zero-E-step held-out counterfactual opt in

**Finding coverage:** M16 and baseline failure 10.

**Files:**

- Create: `tests/test_audit_finalization_contracts_20260720.py`
- Modify: `vfe3/run_artifacts.py`
- Modify only if the existing expectation is proven stale: `tests/test_train.py`
- Modify: `docs/2026-07-20-edits.md`

- [ ] **Step 1: Reproduce the metrics-schema baseline failure.**

  ```powershell
  C:\Python314\python.exe -m pytest tests/test_train.py::test_validation_finalizer_appends_to_existing_metrics_schema --junitxml=C:\tmp\vfe3-fixall-task11-baseline-red.xml
  ```

  Inspect the existing metrics header and finalizer append record before editing either side.

- [ ] **Step 2: Add default and opt-in evaluation-count tests.**

  With a test loader and default configuration, spy on `evaluate` and require exactly one full held-out call. With `evaluate_zero_e_steps_counterfactual=True`, require two calls, the second with `n_e_steps=0`, a diagnostic-counterfactual label, and no replacement of the headline metric. Make the second call raise and assert `n_e_steps` is restored in all cases.

- [ ] **Step 3: Run and record audit RED.**

  ```powershell
  C:\Python314\python.exe -m pytest tests/test_audit_finalization_contracts_20260720.py --junitxml=C:\tmp\vfe3-fixall-task11-audit-red.xml
  ```

- [ ] **Step 4: Guard the counterfactual branch.**

  In `finalize_run`, execute the zero-E-step evaluation only when the new exact-boolean configuration field is true. Save the original depth, set zero only inside `try`, restore in `finally`, and persist results under a diagnostic namespace that cannot be selected as the headline test metric.

- [ ] **Step 5: Repair append-schema behavior.**

  Ensure validation finalization appends a record compatible with an existing metrics schema. If the baseline fixture is stale, update it only after proving the current writer's documented schema and explain the change.

- [ ] **Step 6: Run GREEN.**

  ```powershell
  C:\Python314\python.exe -m pytest tests/test_audit_finalization_contracts_20260720.py tests/test_train.py tests/test_run_artifacts.py --junitxml=C:\tmp\vfe3-fixall-task11-green.xml
  ```

- [ ] **Step 7: Review and commit.**

  Review call count, restoration on exception, metric names, and selection isolation. Commit:

  ```powershell
  git add vfe3/run_artifacts.py tests/test_audit_finalization_contracts_20260720.py tests/test_train.py docs/2026-07-20-edits.md
  git commit -m "fix: make zero-step held-out evaluation opt in"
  ```

  Omit unchanged test files from `git add`.

## Task 12: Resolve the remaining baseline failures by root cause

**Failure coverage:** baseline failures 1 and 6 after Tasks 5, 6, 10, and 11 own the other eight.

**Files:**

- Modify as root cause requires: `vfe3/train.py`, `vfe3/inference/e_step.py`, and the nearest owning helper
- Modify only if proven stale: `tests/test_2026_07_15_data_integrity_remediation.py`, `tests/test_estep_fixed_point_reporting_20260715.py`
- Modify: `docs/2026-07-20-edits.md`

- [ ] **Step 1: Reproduce both nodes in isolation.**

  ```powershell
  C:\Python314\python.exe -m pytest tests/test_2026_07_15_data_integrity_remediation.py::test_training_log_always_names_bits_per_token_when_bpc_is_available tests/test_estep_fixed_point_reporting_20260715.py::test_one_step_ahead_residual_is_distinct_from_configured_last_step --junitxml=C:\tmp\vfe3-fixall-task12-red.xml
  ```

- [ ] **Step 2: Diagnose the training-log label contract.**

  Capture the exact log record. If the executable output carries both BPT and BPC under unambiguous labels, update the stale expected string. If BPC is emitted under an ambiguous energy/unit label, change production logging so every available bits-per-character value is named `bits_per_character` or `bpc`. Do not rename a value without checking its formula.

- [ ] **Step 3: Diagnose fixed-point residual timing.**

  Trace the configured last E-step state and the one-step-ahead probe state. Ensure the reported one-step-ahead residual is computed from an additional update applied to the configured terminal state and is stored separately. Do not make both columns equal merely to satisfy a fixture.

- [ ] **Step 4: Run GREEN and nearby reporting tests.**

  ```powershell
  C:\Python314\python.exe -m pytest tests/test_2026_07_15_data_integrity_remediation.py tests/test_estep_fixed_point_reporting_20260715.py --junitxml=C:\tmp\vfe3-fixall-task12-green.xml
  ```

- [ ] **Step 5: Review and commit.**

  Independently inspect unit labels and fixed-point state timing. Commit only the root-cause corrections:

  ```powershell
  git add vfe3/train.py vfe3/inference/e_step.py tests/test_2026_07_15_data_integrity_remediation.py tests/test_estep_fixed_point_reporting_20260715.py docs/2026-07-20-edits.md
  git commit -m "fix: restore diagnostic reporting contracts"
  ```

  Omit unchanged files from `git add`.

## Task 13: Integrate focused lanes and prove the original failure set is closed

**Files:**

- Modify as required by integration defects: only the task-owned files above
- Modify: `docs/2026-07-20-edits.md`

- [ ] **Step 1: Run all new audit regressions together.**

  ```powershell
  C:\Python314\python.exe -m pytest tests/test_audit_data_contracts_20260720.py tests/test_audit_config_contracts_20260720.py tests/test_audit_process_contracts_20260720.py tests/test_audit_artifact_integrity_20260720.py tests/test_audit_ablation_contracts_20260720.py tests/test_audit_diagnostic_memory_20260720.py tests/test_audit_runtime_semantics_20260720.py tests/test_audit_transport_registry_20260720.py tests/test_audit_full_gaussian_numerics_20260720.py tests/test_audit_prior_bank_routing_20260720.py tests/test_audit_finalization_contracts_20260720.py --junitxml=C:\tmp\vfe3-fixall-audit-focused.xml
  ```

  Expected: zero failures/errors. Record XML counts, duration, and SHA-256.

- [ ] **Step 2: Re-run the exact ten baseline node IDs.**

  Use the ten fully qualified node IDs recorded in the design specification in one pytest command with `--junitxml=C:\tmp\vfe3-fixall-original-ten.xml`. Expected: ten cases, zero failures, zero errors. A changed parameterization count must be explained from the XML rather than assumed.

- [ ] **Step 3: Run syntax and diff checks.**

  ```powershell
  C:\Python314\python.exe -m compileall -q vfe3 ablation.py train_vfe3.py scaling.py scaling_analysis.py tests
  git diff --check origin/main...
  ```

  Also run a repository AST parse over every tracked `.py` file and record the parsed-file count.

- [ ] **Step 4: Repair only demonstrated integration failures.**

  For each failure, use `superpowers:systematic-debugging`: reproduce the smallest node, identify the first wrong boundary, add or strengthen a regression, patch minimally, rerun the focused module and both aggregate commands. Do not opportunistically refactor adjacent code.

- [ ] **Step 5: Obtain an independent integrated source review.**

  Review the complete diff against the approved specification and all finding IDs. The review must confirm no test deletion, no active-config change, no neural layer, preserved exact paths, stable current transport state keys, and explicit checkpoint handling for omitted dormant keys.

- [ ] **Step 6: Commit integration corrections.**

  ```powershell
  git add -u
  git add tests/test_audit_data_contracts_20260720.py tests/test_audit_config_contracts_20260720.py tests/test_audit_process_contracts_20260720.py tests/test_audit_artifact_integrity_20260720.py tests/test_audit_ablation_contracts_20260720.py tests/test_audit_diagnostic_memory_20260720.py tests/test_audit_runtime_semantics_20260720.py tests/test_audit_transport_registry_20260720.py tests/test_audit_full_gaussian_numerics_20260720.py tests/test_audit_prior_bank_routing_20260720.py tests/test_audit_finalization_contracts_20260720.py docs/2026-07-20-edits.md
  git commit -m "test: integrate ultradeep audit remediation"
  ```

  If no post-task integration edits exist, do not create an empty commit.

## Task 14: Run complete CPU and RTX 5090 verification

**Files:**

- Create temporarily outside the repository: `C:\tmp\vfe3-fixall-full-cpu-20260720.xml`
- Create temporarily outside the repository: `C:\tmp\vfe3-fixall-cuda-20260720.xml`
- Modify: `docs/2026-07-20-edits.md`

- [ ] **Step 1: Run the complete CPU suite without an extra `-q`.**

  ```powershell
  C:\Python314\python.exe -m pytest --junitxml=C:\tmp\vfe3-fixall-full-cpu-20260720.xml
  ```

  Expected: zero failures and zero errors. Read `tests`, `failures`, `errors`, `skipped`, and `time` from XML, derive passes as `tests - failures - errors - skipped`, and record the file SHA-256. Do not use the progress line as evidence.

- [ ] **Step 2: If CPU fails, return to the owning task.**

  Reproduce each failing node, diagnose before editing, add a regression when coverage is absent, run the owning focused lane, then rerun the complete CPU suite. Do not declare closure while any failure remains.

- [ ] **Step 3: Run the CUDA-marked lane on the RTX 5090.**

  Use the repository's documented serial `-m cuda` lane and restore both environment variables in `finally`:

  ```powershell
  $priorTestDevice = $env:VFE3_TEST_DEVICE
  $priorCublasConfig = $env:CUBLAS_WORKSPACE_CONFIG
  try {
      $env:VFE3_TEST_DEVICE = 'cuda'
      $env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
      & C:\anaconda\python.exe -m pytest -m cuda --junitxml=C:\tmp\vfe3-fixall-cuda-20260720.xml --durations=100
  }
  finally {
      if ($null -eq $priorTestDevice) { Remove-Item Env:VFE3_TEST_DEVICE -ErrorAction SilentlyContinue } else { $env:VFE3_TEST_DEVICE = $priorTestDevice }
      if ($null -eq $priorCublasConfig) { Remove-Item Env:CUBLAS_WORKSPACE_CONFIG -ErrorAction SilentlyContinue } else { $env:CUBLAS_WORKSPACE_CONFIG = $priorCublasConfig }
  }
  ```

  Include the new H1 bounded-memory CUDA regression and M10 CUDA-compatible numerical regression in that marker lane.

- [ ] **Step 4: Record machine-readable CUDA evidence.**

  Require zero failures and zero errors. Record JUnit fields, SHA-256, `torch.__version__`, device name, capability, and the exact selection expression. Never report the prior audit's 22-case count as current evidence.

- [ ] **Step 5: Commit the verification record.**

  Append exact current results to the dated edit document and commit:

  ```powershell
  git add docs/2026-07-20-edits.md
  git commit -m "docs: record audit remediation verification"
  ```

## Task 15: Build and validate the closure ledger

**Files:**

- Create: `.verification/ledger.json`
- Create or modify: the verification skill's required source-record/evidence files only if its schema requires them
- Modify: `docs/2026-07-20-edits.md`

- [ ] **Step 1: Create one claim per sustained finding.**

  Add H1, M1 through M17, and L1 through L3 as separate claims. Add one aggregate claim for closure of the ten baseline failures. Bind every code claim to the final source revision, focused test command, JUnit file/hash, and relevant full-suite record.

- [ ] **Step 2: Add the mathematical evidence for M10.**

  Record the self-KL derivation and shared-variable derivative cancellation as proof evidence, with the conditioned regression and float64 oracle as supporting numerical evidence. Do not classify numerical agreement alone as mathematical proof.

- [ ] **Step 3: Add four views for H1.**

  Record implementation/source reachability, mechanical shape/allocation evidence, skeptic analysis, and adjudication. The closure statement must be limited to bounded allocation and diagnostic preservation; it must not claim that the historical code produced a measured OOM.

- [ ] **Step 4: Validate in closure mode.**

  Run the exact deterministic validator command from the installed `verification` skill against `.verification/ledger.json`. Expected: exit status zero and no `CANDIDATE` or `LLM_SUPPORTED` claims. Any evidence invalidated by a later source edit must be rerun before validation.

- [ ] **Step 5: Commit the ledger.**

  ```powershell
  git add .verification/ledger.json docs/2026-07-20-edits.md
  git commit -m "docs: validate audit remediation ledger"
  ```

## Task 16: Complete the mandatory Git lifecycle without touching live WIP

**Files:** no new task-owned files after this point.

- [ ] **Step 1: Inspect final task state and provenance.**

  Run `git status --short`, `git diff --stat origin/main...HEAD`, `git log --oneline origin/main..HEAD`, and `git diff --check origin/main...HEAD`. Confirm every created file is tracked and no JUnit XML, synthetic cache, or temporary probe is inside the repository.

- [ ] **Step 2: Fetch and rebase only if safe.**

  Run `git fetch origin` and inspect `git log -5 --oneline origin/main`. If remote main advanced, integrate it in the isolated worktree, resolve conflicts without changing the user's live checkout, and rerun all evidence invalidated by the new revision.

- [ ] **Step 3: Push the task branch.**

  ```powershell
  git push -u origin codex/fix-ultradeep-audit-20260720
  ```

- [ ] **Step 4: Merge to main and push.**

  From an isolated clean integration worktree, fast-forward or merge the verified task branch into current `origin/main`, then push `main`. Fetch again and prove `origin/main` points to the resulting commit.

- [ ] **Step 5: Check whether the user's live checkout can fast-forward.**

  In `C:\Users\chris and christine\Desktop\V3_Transformer`, record `git status --short`. Compare every incoming task path with every dirty live path. Because current WIP includes `scaling.py`, `scaling_analysis.py`, `train_vfe3.py`, and `vfe3/config.py`, fast-forward only if Git can update without altering, overwriting, or masking any WIP. If any incoming path overlaps, leave live main untouched and report the exact overlap.

- [ ] **Step 6: Remove task-owned temporary evidence.**

  After hashes and results are recorded, enumerate the nonrecursive files matching `C:\tmp\vfe3-fixall-*.xml`, compare their resolved full names against the exact RED/GREEN/baseline/final evidence paths declared in Tasks 1-14, print that checked list, and remove only exact matches. Inspect and remove task-created synthetic directories one by one after the same ownership and resolved-path check. Never delete a pre-existing artifact or use recursive wildcard deletion.

- [ ] **Step 7: Clean up the temporary worktree and local branch.**

  After confirming remote merge/push and no uncommitted task files, remove `C:\tmp\V3_Transformer_fix_all_audit_20260720` with `git worktree remove`, delete the local task branch, and prune worktree metadata. Do not remove the remote task branch unless the user requests it.

- [ ] **Step 8: Report exact closure facts.**

  Report the task branch, every final commit SHA needed to identify the patch series, pushed remote branch, resulting `origin/main` SHA, CPU and CUDA JUnit fields/hashes, validated ledger path and validator exit, worktree removal, local branch deletion, live fast-forward result, and final `git status --short` with remaining dirty files attributed to the user.

## Plan self-review checklist

- [ ] Every sustained finding is assigned exactly once: H1; M1-M17; L1-L3.
- [ ] Every clean-baseline failure is assigned to Tasks 5, 6, 10, 11, or 12 and rerun together in Task 13.
- [ ] Every production task begins with an observed RED regression and ends with focused GREEN evidence plus independent review.
- [ ] The full CPU and CUDA counts come from current JUnit XML rather than memory or progress output.
- [ ] M10 includes analytic proof evidence; H1 includes four independent views.
- [ ] No placeholders, `TBD`, vague test commands, or unspecified file ownership remain.
- [ ] The one dated edit document, isolated worktree, pure-path, and full Git-lifecycle requirements are preserved.
