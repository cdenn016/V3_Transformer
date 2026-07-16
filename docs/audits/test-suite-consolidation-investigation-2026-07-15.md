# Test-Suite Consolidation Investigation — 2026-07-15

## Scope and method

This investigation evaluates how to reduce the V3 test suite's wall time while preserving its behavioral, numerical, geometric, and pure-path coverage. The source baseline is `origin/main` at `2ebf0c84b6f49a1edfa855c9a750f858e9ee5aef` in an isolated worktree. The user's live checkout and its pre-existing configuration and documentation changes were not modified.

No test command was run. Three independent investigators analyzed duplication, runtime structure, and coverage risk using source and AST inspection. A fourth agent independently re-read the load-bearing findings. Existing committed JUnit summaries and one pre-existing JUnit XML file were used as timing evidence. The Research wiki pages `[[VFE Transformer Program]]`, `[[Gauge equivariance and geometric deep learning]]`, and `[[GL(K) gauge group]]` were consulted to preserve the distinctions among full-Gaussian congruence, diagonal-family projection, flat Regime I, and covariant and noncovariant Regime-II routes.

## Conclusion

Broadly merging or deleting tests is not the right primary optimization. The suite has many test definitions, but almost all are individually cheap. A surviving default-run JUnit file contains 2,927 cases; 2,321 cases, or 79.3 percent, completed below 10 milliseconds, and 2,756 cases, or 94.2 percent, completed below 50 milliseconds. Only 23 cases took at least one second, while the slowest 25 cases accounted for 63.9 percent of measured testcase time. Collection and other non-testcase overhead was only 1.576 seconds in that run.

The recommended design combines two approaches. First, remove repeated expensive work by sharing immutable trained/finalized artifacts within narrow modules and by separating report-routing decisions from full figure rendering. Second, repair test-tier metadata and benchmark capped process parallelism on the CPU suite. Parameterizing registration checks, consolidating exact duplicates, and extracting shared helpers should follow for maintainability, but those changes will not materially shorten the suite by themselves.

## Evidence baseline

The static inventory found 163 `test_*.py` files, 47,813 lines in those files, 2,419 syntactic test functions, 185 `pytest.mark.parametrize` decorators, and one repository-defined fixture, `device` in `tests/conftest.py:7-14`. There are no module-scoped or session-scoped repository fixtures. Test files contain 659 `VFE3Config` constructor calls across 119 files and 560 `VFEModel` constructor calls across 105 files. AST fingerprinting found no identical complete test bodies, although source-level review found one behaviorally exact regression duplicate and several near duplicates. Twenty-two exact helper-body clone groups cover 50 helper definitions, which is a maintenance issue rather than a runtime driver.

The most recent committed comparison records the following CPU results in `docs/audits/ultradeep-audit-codebase-2026-07-15.md:120-124`:

| Lane | Collected | Passed | Skipped | Wall time |
|---|---:|---:|---:|---:|
| Default | 2,927 | 2,896 | 31 | 127.53 s |
| `--runslow` | 2,927 | 2,914 | 13 | 419.91 s |

The two runs differ by exactly 18 executed tests and 292.38 seconds of observed wall time. This is an observed increment across separate runs, not a stable causal estimate for each slow test. It nevertheless localizes the dominant optimization surface: the 18-node slow allowlist in `tests/conftest.py:33-52` corresponds to 69.6 percent of the slow-inclusive measurement.

A second pre-existing JUnit file, `C:\tmp\vfe3-second-audit-baseline-20260715.xml`, has SHA-256 `C9ED7AF4222D83644632118CF25D7E78E169C94A3FCB3C43880576D01CB4D6D2`. It records 2,927 tests in 130.315 seconds at `2026-07-15T21:32:59-05:00`. No production or test source changed between the audited source commit and this investigation's `origin/main`. Its slowest default-lane cases were:

| Test | Time |
|---|---:|
| `test_report.py::test_generate_figures_skips_english_taxonomies_for_japanese` | 17.126 s |
| `test_train.py::test_training_decreases_loss_on_structured_stream` | 16.386 s |
| `test_train.py::test_random_stream_does_not_clear_cutover_anchor` | 5.401 s |
| `test_forward_kl_uniqueness.py::test_geometric_mean_gap_is_kl_specific` | 4.901 s |
| `test_report.py::test_generate_figures_reuses_one_same_token_snapshot` | 3.793 s |

Static call mapping attributes 43.562 measured seconds to default cases that call `train` and 23.442 measured seconds to default cases that call `generate_figures`; these categories overlap other setup and should be treated as prioritization evidence rather than additive totals.

## Verified findings

### F1. Slow-tier selection is internally inconsistent

`pytest_collection_modifyitems` returns immediately when `--runslow` is present at `tests/conftest.py:68-70`. The hook therefore never adds `pytest.mark.slow` at lines 76-77 in the mode that executes the slow tests. A command such as `pytest --runslow -m slow` cannot reliably select the slow tier. The central node-ID allowlist is also fail-open: a renamed or newly added expensive test runs in the default lane until the string table is updated.

The first harness change should mark every slow node unconditionally and add only the skip marker conditionally. Explicit decorators are preferable over filename strings, but retaining the central allowlist is acceptable if a static contract test proves that every entry resolves and every designated slow test is present. Add `--strict-markers` after the marker set is explicit.

### F2. The slow-inclusive command is not the complete semantic coverage set

The `--runslow` measurement still skipped 13 cases. Six belong to `tests/test_curated_audit_reporting_20260709.py`, whose entire module is skipped because its closure-ledger document was intentionally removed (`tests/test_curated_audit_reporting_20260709.py:6-19`). Six are CUDA-only smokes. The remaining pure-route byte-equality case explicitly skips unless `VFE3_BASELINE_BUNDLE` and `VFE3_FEATURE_BUNDLE` are supplied (`tests/test_hierarchical_probabilistic_completeness_20260712.py:1356-1364`).

"Full coverage" should therefore mean the union of explicit lanes: CPU contracts and mathematics, slow artifact/render integrations, serial CUDA smokes on the RTX 5090, and the external-bundle pure-route identity probe. A green CPU `--runslow` result alone must not be reported as proof that the pure-route bundle identity or CUDA behavior ran.

### F3. Artifact tests repeat train/finalize work that can be shared without losing assertions

`tests/test_run_artifacts.py` repeats a four-step training path at lines 362-370, 373-385, 437-453, 628-638, 641-649, and 652-662. The three slow finalization tests at lines 437-453, 641-649, and 652-662 use compatible default configurations and can consume one module-local immutable finalized-run fixture. Their separate test nodes and assertions can remain, while one training/finalization result supplies `test_results.json`, `summary.json`, the figures, and the `reloaded_best` result.

The default-lane metrics-column assertions at lines 628-638 can consume the output already produced at lines 362-370, or be folded into that default test. They must not be moved behind the slow gate. The attention-PNG test at lines 373-385 uses a different two-layer configuration and should remain separate.

This narrow reuse removes up to three repeated four-step training calls and two repeated finalizers while preserving every existing assertion. A broad shared `VFEModel` fixture is unsafe because training, caches, gradients, RNG, and registry state are mutable.

### F4. Model-channel tests repeat active/off training and render pairs

`tests/test_model_channel_diagnostics_2026_06_13.py:286-332` trains active and inactive configurations once for CSV assertions and then again for attention-file assertions. One module-local active/off trained-artifact fixture can support the three read-only assertion groups while retaining separate test cases. This removes two four-step training executions from the default lane.

The same module calls full `generate_figures` for active and inactive models at lines 337-349. `tests/test_report.py:385-392` performs another active/off pair to check `s_channel_refinement.png`. These are routing assertions wrapped around four complete report renders. A shared report-plan contract can retain the active/inactive filename expectations, while one end-to-end render remains as the production integration proof.

### F5. A default-lane routing test pays for the complete figure driver

`test_generate_figures_skips_english_taxonomies_for_japanese` at `tests/test_report.py:90-107` verifies that English-only outputs are suppressed and language-independent outputs remain. It invokes the full `generate_figures` driver at lines 97-102 and is absent from the slow allowlist. The pre-existing JUnit file records 17.126 seconds for this test, making it the single slowest default-lane case in that artifact.

The production driver currently combines input collection, route availability, UMAP lifecycle, rendering, and filesystem publication in `vfe3/viz/report.py:124-440`. Extract a small report-plan or figure-spec stage that determines enabled figure names from dataset and available inputs. Test Japanese routing against that stage without rendering the full figure set. Retain one live-model full render, one reload/finalize render, and one real UMAP worker lifecycle case as integration boundaries.

### F6. Process parallelism is available but not configured

The development extra contains only `pytest` (`pyproject.toml:15-16`), pytest configuration contains only `testpaths` and `-q` (`pyproject.toml:32-34`), and neither `pyproject.toml` nor `uv.lock` contains xdist or coverage tooling. There is no repository CI configuration.

Process-based xdist is a plausible wall-time reduction because most filesystem cases use worker-local `tmp_path`, while process isolation contains global registries, RNG state, and matplotlib state. It must be benchmarked rather than assumed. Risks include un-restored registry additions in `tests/test_alpha_i.py:33-64`, `tests/test_attention_prior.py:30-41`, and `tests/test_config.py:587-597`; nested UMAP child processes; native Torch/BLAS thread multiplication; and multiple workers targeting the same GPU. Fix registry cleanup first, keep CUDA serial, keep real UMAP serial or tightly capped, and begin with fixed two- and four-worker CPU trials rather than `-n auto`.

The [pytest-xdist distribution documentation](https://pytest-xdist.readthedocs.io/en/stable/distribution.html) defines `loadscope`, `loadfile`, `loadgroup`, and `worksteal`. Benchmark `loadscope` after module fixtures are introduced, and compare it with `worksteal` plus explicit groups for UMAP and shared expensive fixtures. Do not select a scheduler from intuition alone.

### F7. Exact and structural duplication exists, but most of it is not a speed lever

The strongest safe deletion is the behaviorally exact `retract_spd_full` identity/zero-tangent backward-finiteness duplicate in `tests/test_ultradeep_fixes_2026_06_13.py:323-331` and `tests/test_retraction.py:345-351`; retain the domain-owned retraction case. The well-conditioned `safe_spd_inverse` checks in `tests/test_audit_fixes_2026_06_14.py:94-99` and `tests/test_numerics.py:16-21` are near duplicates and can become one explicit parameterized contract covering both ridge regimes.

Registration-presence checks for `regime_ii`, `regime_ii_covariant`, `regime_ii_link`, and `regime_ii_link_charted` can become one expected-key table. Repeated configuration-domain tests and exact helper clones can be consolidated gradually. These changes reduce source size and drift, but their runtime benefit is expected to be small and remains unmeasured.

## Tests that must remain independent

The following test families are superficially similar but protect different executable seams. Shared setup is safe; collapsing their test cases is not.

1. Finite-difference versus autograd-oracle checks in `tests/test_gradients_oracle.py:47-186`, analytic-kernel versus autograd-oracle checks in `tests/test_gradients_kernels.py:19-149`, and the objective-level finite-difference and canonical/surrogate identity checks in `tests/test_free_energy.py:334-407` detect different classes of common-mode error.
2. Frozen goldens in `tests/test_e_step.py:254-260` and `tests/test_perf_equivalence.py:52-105` detect value drift that algebraic property tests can miss.
3. Full-Gaussian pushforward invariance, transported-KL invariance, cocycle flatness, and full-model invariance in `tests/test_gauge_groups.py:149-235` and `tests/test_transport.py:58-111` are distinct invariants.
4. Noncovariant `connection_W`, covariant feature-based Regime II, bare-link noncovariance, and charted-link covariance are separate routes in `tests/test_regime_ii.py`, `tests/test_regime_ii_covariant.py`, and `tests/test_regime_ii_link.py`. A positive gauge test must retain a nonvacuity or negative control.
5. Scalar two-hop zero identity and gradient-kernel zero identity cover separate guarded implementations in `vfe3/free_energy.py:456-466` and `vfe3/gradients/kernels.py:163-190`.
6. Retraction formula correctness, finite-difference derivatives, and degenerate-spectrum finiteness in `tests/test_retraction.py:305-358` must remain separate.

The learnability tests at `tests/test_train.py:287-318` request 800 training steps across their seed/control structure and are expensive. They certify a semantic training claim rather than a smoke path. Do not reduce seeds, steps, or thresholds without a separate statistical recalibration and repeated flakiness study. Process parallelism can shorten their wall time without weakening the claim.

## Design options

| Option | What changes | Runtime effect | Coverage risk | Recommendation |
|---|---|---|---|---|
| Harness and process parallelism | Repair markers, isolate registries, add fixed-worker xdist lanes | Reduces wall time without reducing executed cases; exact gain requires benchmark | Low after isolation; nested UMAP and GPU need serial groups | Adopt |
| Expensive-fixture reuse and plan/render separation | Reuse immutable trained/finalized outputs; test route plans without repeated full renders | Reduces actual CPU, rendering, subprocess, and I/O work | Low to medium; retain explicit end-to-end boundaries | Adopt |
| Broad file merging and parameterization | Merge dated/domain files, centralize helpers, combine similar assertions | Mainly source-size and maintenance improvement | Medium if independent invariants are collapsed | Defer; apply only to verified clones/contracts |

The recommended design is the first two options together. Parallelism attacks elapsed time while fixture and report-plan changes reduce the amount of work. The third option is worthwhile only as a controlled cleanup after coverage identities are explicit.

## Coverage-retention contract

Before deleting or merging test cases, record a stable manifest keyed by:

`source symbol × invariant × configuration/geometry route × oracle type × dtype/device lane`.

Acceptance requires all of the following:

1. The union of all lane manifests equals the pre-change manifest, with no missing or duplicated case IDs.
2. Golden values, tolerances, seeds, dtypes, exact `torch.equal` checks, and negative controls remain unchanged unless separately justified.
3. Oracle-versus-finite-difference, kernel-versus-oracle, algebraic property, golden regression, and end-to-end routing remain independent nodes.
4. Registry matrices pin an explicit expected key and capability set before iterating dynamic registries, preventing vacuous shrinkage.
5. Each gauge-positive contract retains a nonvacuity or gauge-negative control.
6. A dedicated job supplies both pure-route bundles and records that the byte-equality test passed rather than skipped.
7. CUDA cases run serially on the RTX 5090 and are reported separately from CPU coverage.
8. Any deleted historical regression first passes a failure-equivalence check: a targeted mutation or probe must make both the old case and its proposed replacement fail for the same defect.
9. Statement and branch totals do not regress. The repository currently has no coverage configuration, so a baseline must be established before consolidation. Coverage.py tracks branch destinations in addition to executed statements, and pytest-cov can combine results from xdist workers; see the [Coverage.py branch documentation](https://coverage.readthedocs.io/en/latest/branch.html) and [pytest-cov xdist documentation](https://pytest-cov.readthedocs.io/en/latest/xdist.html).

## Proposed implementation sequence

Phase 0 establishes trustworthy measurement. Assign `slow` unconditionally, conditionally skip it, enforce strict marker names, restore every registry mutation, define the lane manifest, and capture serial JUnit plus `--durations=100`. Pytest documents `--durations` and `--durations-min` in its [execution profiling guidance](https://docs.pytest.org/en/stable/how-to/usage.html#profiling-test-execution-duration). Establish statement/branch coverage and test contexts before removing any case.

Phase 1 reduces repeated work without changing case identities. Add narrow module-local immutable fixtures for the artifact and model-channel cohorts. Extract report availability/spec planning from full render execution. Convert the Japanese taxonomy case and active/off filename contracts to plan-level tests, while retaining the named end-to-end render boundaries.

Phase 2 benchmarks CPU process parallelism. Compare serial, two-worker, and four-worker runs under `loadscope` and a grouped/work-stealing alternative. Record node-set equality, passes/skips, wall time, testcase time, peak RAM, process count, disk writes, and native thread settings. Keep CUDA and real UMAP lanes serial during the first experiment.

Phase 3 performs low-risk consolidation. Remove the exact retraction duplicate, parameterize the near-duplicate safe-inverse and registration-presence contracts, and extract only domain-local helper clones. Do not perform a repository-wide helper migration because the maintenance gain is small and broad mutable-fixture reuse is unsafe.

Phase 4 runs the coverage-retention gate and repeated timing trials. No speedup should be claimed from a single post-change run. The resulting default, slow CPU, CUDA, and pure-route bundle results should be reported independently and as a manifest union.

## Verification and limitations

No tests, pytest collection, test imports, or test modules were executed during this investigation, per user instruction. Verification consisted of source and AST inspection, four-agent cross-checking, existing JUnit summaries, the pre-existing XML artifact described above, official pytest/xdist/coverage documentation, and repository-state checks. Proposed speedups are unmeasured until implementation and benchmarking; only the current timing distribution and repeated executable call sites are established here.
