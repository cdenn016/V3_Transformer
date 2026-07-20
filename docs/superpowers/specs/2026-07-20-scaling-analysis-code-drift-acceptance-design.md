# Scaling Analysis Code-Drift Acceptance Design

**Date:** 2026-07-20

**Status:** Approved

## Purpose

`scaling_analysis.py` currently rejects a completed scaling artifact when the schema-2 cell contract and the run provenance record disagree about the dirty-worktree fingerprint. That strict behavior is the correct default integrity policy, but the current blocksize sweep contains nine completed cells split across two dirty-worktree fingerprints. The analyzer accepts five rows, classifies four present results as missing, and then prints that zero parameter sizes are present because it counts the disabled fit set rather than the harvested points.

The requested change adds an explicit analysis-only override that accepts a code-identity mismatch while retaining every other artifact-integrity check. The current click-to-run configuration enables that override so all nine blocksize cells enter the analysis.

## Configuration

Add `force_accept_code_identity_drift` to the module-level `CONFIG` dictionary and set it to `True` for the requested current analysis. Setting it to `False` restores strict fail-closed behavior.

The option affects analysis only. It does not change scaling-run caching, resume classification, training, provenance capture, or the terminal status written by `scaling.py`.

## Validation Boundary

Strict mode continues to require exact agreement between the scaling-cell code identity and the run provenance identity.

Forced mode may override only disagreement in `git_dirty` or `git_dirty_fingerprint` when both records otherwise contain well-formed code identities and the `git_sha` values agree. It must continue rejecting failures in the schema version, dataset binding, serialized configuration digest, reuse-contract digest, summary metrics, route/label/seed identity, Git SHA, and train/validation/test source identities. A forced row must also match at least one strictly accepted peer on all three serialized source identities and all three provenance data digests, so the override cannot create its own unanchored source cohort.

An accepted row records whether code identity was forced, the cell-bound identity, and the provenance identity. These fields propagate to persisted analysis outputs so the override is auditable.

## Design Completion and Fitting

The requested-design join treats a declared cell as complete when a corresponding row passed either strict validation or the narrow forced-code-identity validation. A manifest whose cells all join successfully may become analysis-complete under forced mode only when its original terminal error is exactly code-identity drift. Any cell failure, missing artifact, malformed manifest, data-source drift, or unrelated invocation error keeps the design incomplete and withholds fitting.

The analysis summary records the original manifest status, the effective forced-completion decision, the number and identities of forced rows, and a code-drift confound. Console and Markdown output display a warning that the fit spans forced code-identity cohorts. The fit is descriptive and must not be presented as a provenance-clean scaling law.

## Distinct-Size Reporting

Track harvested parameter sizes separately from fit-eligible sizes. When fitting is disabled, the console must report the harvested count and the reason fitting was withheld instead of printing that zero sizes are present. When forced completion enables fitting, the normal fit-size count is used and the forced-acceptance warning remains visible.

## Tests

Tests will establish the behavior in this order:

1. Strict mode rejects a dirty-worktree fingerprint mismatch.
2. Forced mode accepts that mismatch and records both identities.
3. Forced mode still rejects Git-SHA, source-identity, contract-digest, configuration, seed, and metric failures.
4. A design containing only otherwise-valid rows plus the exact code-drift terminal error becomes effectively complete in forced mode.
5. Unrelated incomplete-design errors remain incomplete under forced mode.
6. The current nine-cell blocksize artifact shape yields nine accepted rows and three parameter sizes when force mode is enabled.
7. Disabled fitting reports harvested sizes instead of a false zero.

Focused tests will run before the full repository suite. Test counts will come from JUnit XML.
