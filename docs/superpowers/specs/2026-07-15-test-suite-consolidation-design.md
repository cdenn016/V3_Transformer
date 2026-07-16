# Test-Suite Consolidation Design

## Status and scope

This design implements the recommendation approved after the read-only investigation in `docs/audits/test-suite-consolidation-investigation-2026-07-15.md`. It targets elapsed time and repeated expensive work without weakening the suite's mathematical, geometric, numerical, device, or pure-route coverage. It does not reduce seeds, training steps, tolerances, golden values, finite-difference grids, or negative controls.

## Architecture

The suite will use explicit semantic markers instead of a central node-ID allowlist. A slow test will always carry `slow`; `--runslow` will control only whether those tests receive a skip marker. Dedicated `serial`, `cuda`, and `external_bundle` markers will make the CPU-parallel, serial integration, RTX 5090, and pure-route bundle lanes an explicit union rather than an ambiguous single command. Strict marker validation will prevent misspelled tiers from silently changing that union.

Repeated expensive work will be removed only where outputs are immutable after setup. `test_run_artifacts.py` will share one module-scoped trained run between read-only file assertions and one module-scoped finalized run between separate finalization assertions. `test_model_channel_diagnostics_2026_06_13.py` will similarly share one active and one pure-path trained run, with finalization performed once per route only when the slow finalization contract is selected. Tests will receive paths and frozen result records; they will not receive a shared mutable model as a general-purpose fixture.

`vfe3.viz.report` will expose a pure figure-planning function. It will accept a dataset name and an ordered availability mapping and return the output filenames that are eligible for publication. The function will enforce the English-taxonomy route and preserve input-availability gating. The live renderer will consume this plan, so routing tests can verify Japanese and model-channel filename behavior without running extractors, UMAP, matplotlib rendering, or filesystem publication. One live-model render, one reload/finalize render, and one real UMAP lifecycle test will remain as independent integration boundaries.

The development dependency set will add `pytest-xdist` and `pytest-cov`. Coverage will use branch measurement over `vfe3`, and the repository will document fixed-worker CPU commands rather than enabling `-n auto` globally. CUDA, real UMAP, and external-bundle cases remain outside the parallel CPU lane.

## Coverage-retention contract

The retained case identity is `source symbol x invariant x configuration or geometry route x oracle type x dtype or device lane`. Finite-difference versus autograd, kernel versus oracle, golden regression, positive and negative gauge controls, Regime-II route distinctions, scalar versus gradient two-hop identities, and retraction formula, derivative, and degeneracy tests remain separate nodes. The 800-step learnability structure remains unchanged.

Every pre-change collected node will be compared with the post-change collection. Intentional node changes are limited to explicit parameter IDs for a near-duplicate inverse contract and deletion of the demonstrated retraction duplicate. The replacement contract must retain both ridge values. Statement and branch coverage must not regress after the baseline is established. CUDA and external-bundle cases may be skipped when their prerequisites are absent, but their skips must be reported separately and cannot be described as executed coverage.

## Failure behavior and isolation

Registry-mutating tests will restore the exact prior registry entry in `finally` blocks or through `monkeypatch`, including the case where a key existed before the test. Module fixtures will create their own `tmp_path_factory` directories. Tests consume those outputs read-only; any test that mutates a model, RNG state, registry, or artifact tree remains function-scoped.

The report plan is a pure function. Unknown figure names are not invented, false availability excludes a figure, and non-English datasets exclude only the two English-taxonomy outputs. Rendering remains best-effort and keeps the existing per-figure failure isolation.

## Verification

Development follows targeted red-green cycles for marker semantics and report planning. Then the affected artifact, model-channel, report, registry, inverse, and retraction modules run. Final verification records machine-readable JUnit results for the serial default suite and the slow-inclusive suite, branch-coverage output, dedicated CUDA results when available, the external-bundle result when both paths are available, and fixed two- and four-worker timing trials. Worker selection is based on those measurements; no speedup is claimed from static reasoning or one unrecorded run.
