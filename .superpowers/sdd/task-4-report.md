# Task 4 RED/GREEN Report

Task 4 repaired second-panel findings S2-D1 through S2-D7, first-panel P3, and the duplicated first-panel P8/second-panel S2-D6 finding. The implementation binds resumable data cursors to an exact dataset, split, tokenizer, vocabulary, cap, cache metadata, and content identity before any model, optimizer, cursor, or RNG restoration. It validates binary cache sidecars and byte lengths before memory mapping, covers every held-out transition once through ignored-target padding, separates always-defined bits per token from nullable bits per character across training, ablation, and scaling publishers, keys character normalization to source content, bounds character and unigram working chunks, and reuses an immutable loader digest across ablation finalizations. Existing corpus, tokenizer, cap, and training-loader defaults were not changed.

## Root cause and implementation

The saved cursor previously carried only epoch and generator position, so an otherwise valid checkpoint could splice its cursor into different data. Binary loading trusted the sidecar shape, evaluation retained only full nonoverlapping windows, unavailable character normalization was replaced with `1.0` and mislabeled as BPC, character normalization memoized only a path-like key and decoded the whole split, unigram construction created a corpus-wide host-int64 copy, and provenance rehashed the same shared loader on each ablation cell.

`make_dataloader` now attaches a JSON-compatible identity contract to its `TokenWindows` dataset. `RunArtifacts.save_checkpoint` owns and persists that contract, while `load_checkpoint` validates the saved and live contracts before loading any mutable state. Direct tensor-backed loaders receive a bounded canonical in-memory identity; opaque shuffled loaders still fail closed for exact resume. Binary cache metadata validation is shared by identity, count, and load paths and rejects missing sidecars, nonpositive logical lengths, unsupported dtypes, and any byte-length mismatch before `np.memmap`.

Evaluation loaders now use nonoverlapping padded final windows when `shuffle=False` and `drop_last=False`; real targets remain ordered exactly once and padding targets are `-100`. `evaluate` always emits `bits_per_token` and emits `bpc=None` when no character normalization exists. Tokens-per-character uses bounded token chunks, byte decoding, and incremental UTF-8 state so chunk boundaries preserve whole-stream codepoint semantics. Training-token counts use bounded host-int64 chunks, and provenance caches the canonical content digest on the immutable shared dataset instance.

## RED evidence

The initial focused command was:

```text
python -m pytest tests/test_2026_07_15_data_integrity_remediation.py --junitxml=C:\tmp\vfe3-task4-red-20260716.xml
```

Its literal pytest summary was:

```text
22 failed, 1 passed, 1 warning in 0.99s
```

The JUnit attributes were read directly as `tests=23 failures=22 errors=0 skipped=0 time=0.990`. Representative failures were `invalid binary metadata reached np.memmap`, `DID NOT RAISE <class 'ValueError'>` for an extended cache, `TokenWindows.__init__() got an unexpected keyword argument 'pad_final'`, `unsupported operand type(s) for *: 'float' and 'NoneType'` for unavailable normalization, `tokens_per_char() got an unexpected keyword argument 'chunk_tokens'`, absence of `_bincount_token_chunks`, two digest calls instead of one, and `load_checkpoint() got an unexpected keyword argument 'expected_data_identity'`. The only pre-edit pass was the already fail-closed missing-sidecar case.

## GREEN evidence

The focused GREEN command was:

```text
python -m pytest tests/test_2026_07_15_data_integrity_remediation.py --junitxml=C:\tmp\vfe3-task4-green-focused-20260716.xml
```

Pytest reported `23 passed in 0.92s`. The JUnit attributes were read directly as:

```text
tests=23 failures=0 errors=0 skipped=0 time=0.918
```

The first neighboring command covered dataset, BPC, checkpoint-resume, contract, run-artifact, and ablation-artifact behavior:

```text
python -m pytest tests/test_data.py tests/test_fixes_20260709_data.py tests/test_bpc.py tests/test_checkpoint_resume.py tests/test_audit_contract_types_20260711.py tests/test_run_artifacts.py tests/test_ablation_artifact_resume_20260712.py tests/test_ablation_tackon.py --maxfail=30 --junitxml=C:\tmp\vfe3-task4-neighbors-iteration2-20260716.xml
```

Pytest reported `180 passed, 4 skipped, 28 warnings in 11.98s`; JUnit recorded:

```text
tests=184 failures=0 errors=0 skipped=4 time=11.974
```

The entrypoint and reporting neighbor command was:

```text
python -m pytest tests/test_train.py tests/test_training_timing_20260711.py tests/test_training_epoch_metrics_20260715.py tests/test_ablation_reporting.py tests/test_audit_artifact_tooling_20260713.py tests/test_round3_artifacts.py tests/test_round3_train_sync.py --maxfail=30 --junitxml=C:\tmp\vfe3-task4-train-neighbors-green-20260716.xml
```

An intermediate run identified two pinned expectations that did not yet include the newly named bits-per-token fields. After updating those neighboring contract assertions, the identical command reported `78 passed, 2 skipped, 14 warnings in 28.95s`; JUnit recorded:

```text
tests=80 failures=0 errors=0 skipped=2 time=28.948
```

These were selected focused and neighboring runs, not the full or slow suite. All Task 4 JUnit XML files were task-owned transient evidence; their machine-readable totals were recorded here before cleanup.

## Scaling publisher integration

A final publisher search found the same unavailable-normalization coercion in `scaling.py`. A first isolated test attempt stopped in the test fixture because its configuration stub omitted `deterministic`; no production code had changed, and the fixture was corrected before using the test as contract evidence. The genuine RED command was:

```text
python -m pytest tests/test_2026_07_15_data_integrity_remediation.py::test_scaling_preserves_unavailable_character_normalization --junitxml=C:\tmp\vfe3-task4-scaling-red-contract-20260716.xml
```

It failed on `assert 1.0 is None`, showing that both scaling validation and test normalization still coerced unavailable character normalization. Pytest reported `1 failed in 0.09s`; JUnit recorded `tests=1 failures=1 errors=0 skipped=0 time=0.088`.

After removing only those two `or 1.0` coercions, the focused regression and directly affected scaling module command was:

```text
python -m pytest tests/test_2026_07_15_data_integrity_remediation.py::test_scaling_preserves_unavailable_character_normalization tests/test_scaling_mup.py --junitxml=C:\tmp\vfe3-task4-scaling-green-20260716.xml
```

Pytest reported `16 passed, 22 warnings in 2.07s`; JUnit recorded `tests=16 failures=0 errors=0 skipped=0 time=2.067`. The regression executes `scaling.run_cell`, forces character normalization to be unavailable, verifies that `None` reaches both validation training and test finalization, and verifies the downstream metric mapping contains named `bits_per_token` with nullable `bpc`.

## Final review gap

Read-only final review found that a legacy checkpoint with no `data_state` could still restore model, optimizer, and RNG state when the caller supplied a live expected data identity; training rejected the missing cursor only after that mutation. The strict no-mutation RED command was:

```text
python -m pytest tests/test_2026_07_15_data_integrity_remediation.py::test_exact_resume_rejects_missing_data_state_before_any_mutation --junitxml=C:\tmp\vfe3-task4-legacy-resume-red-20260716.xml
```

It reported `1 failed in 0.72s` with `DID NOT RAISE <class 'RuntimeError'>`; JUnit recorded `tests=1 failures=1 errors=0 skipped=0 time=0.721`. `load_checkpoint` now rejects that legacy bundle immediately after the safe load and before any mutable restoration whenever a live expected data identity makes the request an exact data resume.

The reviewer also found that the existing fixed-order generator fixture mocked token loading but not the newly required source-identity seam. Its helper now supplies a deterministic fixture identity without weakening production behavior. The narrow GREEN command was:

```text
python -m pytest tests/test_2026_07_15_data_integrity_remediation.py::test_exact_resume_rejects_missing_data_state_before_any_mutation tests/test_data_generator.py --junitxml=C:\tmp\vfe3-task4-review-blockers-green-20260716.xml
```

Pytest reported `4 passed in 0.78s`; JUnit recorded `tests=4 failures=0 errors=0 skipped=0 time=0.777`. The regression asserts that the rejected legacy resume leaves every model tensor, optimizer state, global CPU RNG, and caller cursor unchanged.

The same review found that the versioned identity validator required the top-level `source` mapping but accepted an empty or partial nested mapping. The parameterized RED command was:

```text
python -m pytest tests/test_2026_07_15_data_integrity_remediation.py::test_data_identity_schema_rejects_malformed_source_before_any_restore --junitxml=C:\tmp\vfe3-task4-identity-schema-red-20260716.xml
```

All eleven cases failed because malformed identities were accepted: missing source format, tokenizer tag, byte size, content digest, binary metadata, metadata digest, token count, or dtype; empty format or content digest; and a binary byte count inconsistent with `n_tokens * dtype.itemsize`. Pytest reported `11 failed in 0.86s`; JUnit recorded `tests=11 failures=11 errors=0 skipped=0 time=0.860`.

The validator now enforces the complete schema-version-1 identity at both save and load boundaries. File-backed identities require a tokenizer contract, known source format, positive byte size, SHA-256 content digest, and the format-appropriate metadata contract. Binary identities additionally require a positive token count, supported integer dtype, SHA-256 sidecar digest, and exact byte equality. Tensor identities carry the corresponding positive count, supported tensor dtype, exact byte equality, and canonical content digest. The identical parameterized GREEN target reported `11 passed in 0.89s`; JUnit recorded `tests=11 failures=0 errors=0 skipped=0 time=0.891`.

The final fixture audit found one ablation loader test that mocked loader construction but still called the new source-identity seam. It happened to pass on this machine by hashing a real local cache, making the unit test environment-dependent. The fixture now supplies a deterministic source identity, and the only requested verification target was:

```text
python -m pytest tests/test_ablation_tackon.py::test_get_loader_threads_split_aware_shuffle_drop_last --junitxml=C:\tmp\vfe3-task4-ablation-fixture-green-20260716.xml
```

Pytest reported `1 passed in 0.25s`; JUnit recorded `tests=1 failures=0 errors=0 skipped=0 time=0.248`.

## Scope and handoff

The task changes are limited to the data loader/cache path, data-state contracts, checkpoint and artifact handling, training/evaluation metric semantics, click-to-run, ablation, and scaling integration, focused regressions, and directly affected neighboring expectations. No Research-vault file, live checkout, default data choice, default tokenizer, configured cap, daily edit ledger, push, merge, or full/slow test run is part of this task. There is no unresolved public identity choice: cache-backed loaders use their exact source contract, direct tensor-backed loaders use an exact bounded content identity, and opaque resumable loaders fail closed.
