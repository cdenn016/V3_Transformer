# Task 5: Belief Cache, Checkpoint Schema, Serialization, and Comparison Contracts

Task 5 repairs first-panel findings T1, T2, T3, T4, and Q4 without changing any configuration default or removing any full-inference or mathematically pure route. Cache-ineligible configurations fall back to the existing full rollout. The sigma-gate loader supports one explicit schema, `vfe3config-exact-v1`, defined by the exact current `VFE3Config` field set; every older, newer, or partial field set is rejected until a named migration is implemented.

## RED evidence

The regression tests were written before production changes. This command reproduced the five audit probes:

```text
python -m pytest tests\test_2026_07_15_cache_serialization_remediation.py --junitxml=C:\tmp\vfe3-task5-red-20260716.xml
```

Literal RED output:

```text
FFFFFFFFFFFFFFFF.                                                        [100%]
16 failed, 1 passed in 0.18s
```

The JUnit record reported `tests=17 failures=16 errors=0 skipped=0`. The failures established that the exact-MM alias and active positional group product were both admitted to a cache whose cached/full terminal beliefs differed, recognized true/false strings remained strings, ambiguous boolean-like strings were accepted, equal-token comparison records with `(sequence_count, sequence_length)=(10,10)` and `(11,5)` were accepted, and a checkpoint missing a current behavior field loaded through the current default. The one passing control was an exact-current-schema checkpoint.

## Implementation

`cache_supported` now resolves `e_step_update` through the same `canonical_e_step_update` function used by executable E-step inference, then rejects the effective `mm_exact` route. It also rejects an active `pos_phi_compose='group_product'` route because the optimized cache does not carry `right_phi`; the exact positional route remains reachable through the full rollout.

`config_from_serialized` now identifies every `bool` and `Optional[bool]` dataclass field before construction. Actual booleans remain unchanged, case-insensitive exact spellings `true` and `false` become real booleans, and every other string or non-boolean representation fails closed.

Controlled embedding comparison identity now includes `sample.sequence_count` and `sample.sequence_length` alongside token count and token fingerprint. Sigma-gate checkpoint loading now rejects missing or unknown config fields before model construction and routes accepted current-schema values through the centralized deserializer.

## GREEN evidence

Focused GREEN command:

```text
python -m pytest tests\test_2026_07_15_cache_serialization_remediation.py --junitxml=C:\tmp\vfe3-task5-green-focused-20260716.xml
```

The machine-readable JUnit record reported `tests=17 failures=0 errors=0 skipped=0`.

Neighboring cache, E-step, config, controlled-comparison, checkpoint, sigma-gate, and run-artifact command:

```text
python -m pytest tests\test_belief_cache.py tests\test_e_step.py tests\test_config.py tests\test_fix_config_audit.py tests\test_controlled_umap_comparison_20260714.py tests\test_checkpoint_resume.py tests\test_sigma_gate.py tests\test_run_artifacts.py --junitxml=C:\tmp\vfe3-task5-green-neighboring-20260716.xml
```

The machine-readable JUnit record reported `tests=369 failures=0 errors=0 skipped=4`; pytest summarized `365 passed, 4 skipped`. After the successful summary and exit status 0, the Python 3.14 process emitted an optional sklearn/pandas/pyarrow access-violation diagnostic during shutdown. It did not alter the JUnit result, but it is recorded here rather than hidden.

Static verification used:

```text
ruff check --no-cache sigma_gate_measure.py vfe3\config.py vfe3\inference\belief_cache.py vfe3\viz\embedding_comparison.py tests\test_belief_cache.py tests\test_2026_07_15_cache_serialization_remediation.py
```

Ruff reported `All checks passed!`. The full and slow suites were not run, as reserved by the controller. The Research vault, daily edit ledger, live checkout, remote refs, and push/merge lifecycle were not touched.
