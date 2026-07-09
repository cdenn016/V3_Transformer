# Design: Curated Salvage of the 2026-07-09 Audit Findings

Date: 2026-07-09
Status: approved design, awaiting written-spec review
Branch: `fix/curated-audit-salvage-20260709`
Base: `origin/main` at `e504f1c5ad5d277f653534cfc7fb63fd3b1bee61`
Worktree: `C:\tmp\V3_Transformer_curated_audit_salvage_20260709`

## Purpose

This repair closes the verified correctness, numerical, reliability, API-contract, resume, and reporting findings in `docs/audits/ultradeep-audit-findings-investigation-2026-07-09.md` and `docs/audits/deep-audit-and-wikitext103-performance-investigation-2026-07-09.md`. The second report will be imported into this branch as a source artifact before implementation begins. The user approved a curated salvage: work proceeds immediately in a fresh worktree, without waiting for the live Fable session and without modifying its checkout.

The performance bottlenecks P1 through P6 are deferred to a separate performance branch. Hypotheses H1 through H7 remain research proposals and are not implementation scope. This branch does not increase E-step depth, layer count, model capacity, or experiment-toggle values.

## Source ledger and baseline

The prior audit contains 107 numbered rows. Findings 1, 22, 26, 56, 75, 81, and 93 remain intentional-design rulings. Findings 62 and 92 were fixed before this branch. The other 98 rows require a tested repair, a fail-closed guard, a truthful reporting correction, or a documented and tested approximation ruling. The new audit adds M1 through M11 and L1 through L8. Its addenda extend Findings 14 and 87 rather than creating new identifiers. The closure ledger therefore tracks 117 nonexcluded identifiers, with the two addenda nested under their original rows.

The untouched worktree baseline was measured with `python -m pytest --junitxml=C:\tmp\vfe3-curated-salvage-baseline-20260709.xml`. The JUnit suite reports `tests=1692`, `failures=9`, `errors=0`, `skipped=1`, and `time=692.408`, hence 1,682 passing tests. The failures are two deterministic-contract tests, three seed-sensitive Phase-0 golden tests, one checkpoint-interval default test, and three run-label tests. They are baseline defects in this repair, not evidence produced by a source change.

## Curated salvage boundary

The live checkout is a donor, not a merge base. Each donor hunk is independently checked against executable source and its audit row before it enters this branch. The current donor has 20 clearly relevant repairs and seven partial repairs, but none is accepted merely because a test file or comment names a finding. A salvaged change must reproduce the intended counterexample, make the new regression pass, and preserve unrelated behavior.

The branch will not import the live configuration changes in `ablation.py` or `train_vfe3.py`, the deletions under `vfe3_policy_results`, or unrelated generated artifacts. Audit-related changes inside files that also contain user configuration edits will be reconstructed surgically. The five donor regression modules may be copied only after their assertions are reviewed for mathematical and contract fidelity. The live checkout will never be stashed, reset, reverted, or committed by this work.

## Closure rules

A finding is closed only when at least one of four conditions is met. First, its executable counterexample becomes a regression test and no longer reproduces. Second, an unsupported combination is rejected before computation with a targeted test and an accurate error. Third, a reporting field is corrected, renamed, or omitted and its statistic is pinned by a schema test. Fourth, an approximation is explicitly identified as such, the pure path remains available, and a test proves that the approximation is not mislabeled as exact. Comments alone do not close a row.

Existing configuration values and defaults remain unchanged. Where the audit concerns a broken generated sweep cell, the repair may make the cell structurally constructible without changing the baseline experiment. Where several mathematically non-equivalent remedies exist, the branch uses the smallest remedy that follows the repository's stated contract and retains the theoretically pure path.

## Mathematical and behavioral decisions

### Inference and free-energy consistency

The truncation boundary will create fresh gradient-bearing leaves for the retained last-k E-step and rebuild any transport whose graph must not cross that boundary. Both internally hoisted and caller-supplied transports will obey the same rule. Oracle-routed last-k inference must restore nonzero gradients into prior and connection parameters while `k=T` remains equivalent to full unrolling.

`lambda_h=0` is an absolute channel-off gate before state-dependent coefficient dispatch. Refined model beliefs will be used consistently by every gamma consumer. The model-channel E-step will receive the same global update policy as the belief channel, while each channel evaluates its own halting criterion and randomized iteration draw. The actual prior entering the final layer will be captured during the forward recurrence and used by multilayer M-step self-coupling.

Two-hop coupling will use raw detached attention products in the scalar functional, with the destination clamp mask applied to the derivative of the destination energy. The closed-form kernel, autograd oracle, line-search value, Metropolis value, and diagnostic decomposition will share that convention. The decoupled RoPE phi objective will distinguish score-gauge attention energy from value-gauge coupling energy in the same manner as the scalar functional. Phi reflection will be threaded through every differentiated transport route.

Query-adaptive temperature remains an explicitly detached surrogate rather than silently acquiring a new derivative term. Reports and configuration diagnostics will state that convention. Finding 50's state-dependent-alpha MM update remains a valid tangent-majorization step; it receives a contract test and accurate wording rather than a behavior change.

### Geometry and numerical behavior

Spectral eigengap damping will be computed per matrix over the final two axes. The Log-Euclidean full-covariance route will convert the ambient covariance tangent with the Frechet derivative of the matrix logarithm before exponentiation, restoring the first-order retraction condition. The affine-invariant SPD route remains the pure default.

Full-covariance trust regions will use Cholesky Mahalanobis whitening, with an explicit marginal fallback only for failed factorizations. Full-chunked Gaussian decoding will match the unridged full Gaussian KL, using jitter only on failed factorization. Regime-II soft caps will use an overflow-safe scaled or float64 norm. Pullback solves and their convergence checks will avoid repeated host synchronization and retain sufficient precision through the solve. The bracket-closure cache will use a bounded, value-stable key and will not retain fresh CUDA basis tensors by identity. Laplace Renyi evaluation above order one will remain in log space.

Nonclosed cross-coupled BCH configurations will fail at construction rather than project silently. Large skew exponentials may enter an automatic float64 island without changing the represented group element. Trainable log-variance overflow protection will use a representable exponential bound and a diagnostic warning; it will not reuse `sigma_max` as a hidden hard model prior.

### Omega, reflection, cache, and registry behavior

Omega-direct lower APIs will reject phi optimization, additive encoders that provide no omega frame, and incompatible off/frozen/reflection combinations before forward computation. Existing reflection modes for product groups will be labeled as block-0 probes rather than expanded into a new proposal distribution. The omega optimizer's retraction-SGD semantics will be reported accurately; this branch will not invent tangent momentum or symplectic projection algorithms.

Skew omega transport will check orthogonality on a bounded diagnostic cadence and use a true inverse when the transpose contract is not met. Symplectic routes will receive residual monitoring and truthful reporting, while projection remains deferred because it requires a separate mathematical design. Cache inverse conventions will match the live transport, and shared context inverses will be reused across candidates. Compact omega blocks will remain compact through their inverse and transport operations to close Finding 97, and reorthogonalization will visit only rows changed since the preceding cadence to close Finding 99. The broader block-GL belief representation in P1 remains deferred.

The belief cache will fail closed for phi-reflection configurations until reflection is represented in the cached state and transport. Visualization and diagnostic replay will use the trained gauge parameterization. Decoder callable, capabilities, and fused-CE support will live in one registry record. Kernel override will invalidate its compiled cache. RoPE cache keys will include all semantic fields that determine the cached tensor.

### Checkpoint, determinism, data, and generation contracts

Deterministic setup will be centralized and reversible across entry points without changing the configured default. The effective deterministic state will be recorded. Optimizer group metadata will be re-stamped from the current construction after state loading. Checkpoints will persist the reflection generator, omega cadence, and data-iterator state required for exact continuation. The data iterator design will save the generator state from the start of the current epoch and the consumed-batch cursor, then reconstruct and fast-forward that epoch on resume.

Best weights will be bundled with a normalized semantic configuration fingerprint. An external `config_from` file must match it. Missing checkpoint files will retain `FileNotFoundError` semantics rather than being described as unsafe serialization.

Token limits will be applied before memmap conversion and released from full backing storage. Dataset and split names must be safe path components, and token IDs must fit the configured vocabulary. Provenance will record train, validation, and test hashes, token counts, tokenizer identity, data-order seed, and train cap. Scaling and ablation cache hits will require matching code identity, matching requested artifacts, and a successful terminal state.

`generate_efe.py` will honor its dataset tokenizer, require an explicit existing checkpoint, verify vocabulary compatibility, and compare stochastic arms with paired CPU/CUDA RNG state. Non-policy generation will decode only the last position and reject nonfinite sampling rows. Policy scorers will fail closed when context truncation would make the prior and rollout use different windows. `logprob_control` will use the continuation log probability once rather than multiply by the same base prior twice.

### Reporting and analysis contracts

Scaling fits shown side by side will share one weighting and offset specification. An offset fit requires at least four distinct sizes. Mixed-code, mixed-data, and route-divergence states will be saved in the analysis artifact; pooled estimates may remain visible but must be labeled as confounded. Frequency strata will use corpus-level training counts rather than the evaluation sample. Existing route configurations will be relabeled when necessary, not silently changed.

Pure-path reports will include sigma-update suppression, two-hop coupling, reflection, gauge parameterization, family/group invariance, and fixed-prior surrogate status. Capacity and FLOP reports will distinguish linear decoding from prior-bank decoding and include the live model-channel E-step. Figure generation will enforce its memory guard at the reusable entry point, close failed figures, use dataset-specific tokenizers, and honor the existing `generate_figures` gate for periodic attention images.

E-step gradient diagnostics under accumulation will report the arithmetic mean across contributing microbatches under a name that states the aggregation. EMA will advance only after an accepted optimizer update, including GradScaler overflow handling. Test helper scripts will derive pass/fail/skip counts from JUnit XML. Public signatures and touched text will be brought into the repository's ordering and American-English conventions.

## Implementation phases

| Phase | Work | Exit condition |
| --- | --- | --- |
| 0 | Import the second audit artifact, build the 117-row closure ledger, and salvage reviewed donor tests and hunks | Every imported hunk has a finding ID and a failing-or-baseline regression |
| 1 | Repair free-energy, E-step, hierarchy, SPD, divergence, and transport mathematics | Focused CPU gradient/value/property tests pass |
| 2 | Repair checkpoint, deterministic, optimizer-loop, data, provenance, and resume contracts | Split-run continuation and artifact-schema tests pass |
| 3 | Repair omega/cache/policy/registry and generation contracts | Gauge/cache/policy differential tests pass, including fail-closed routes |
| 4 | Repair scaling, reporting, figures, statistics, tooling, and baseline test drift | Reporting/schema tests pass and no user config value changes appear in the diff |
| 5 | Run targeted CUDA validation, the complete suite, and an independent closure-ledger review | JUnit reports zero failures/errors and every ledger row has evidence |

## Test and verification policy

Each behavioral edit begins with a failing regression or an executable reproduction of the audit probe. Tests will use small dimensions and deterministic seeds. Mathematical paths receive value-equivalence, gradient, first-order, or equivariance assertions as appropriate; reporting paths receive schema and provenance assertions. Existing tests will not be weakened to accommodate a patch.

The current command-line `python` reports `torch=2.11.0+cpu`. Before Phase 5, the implementation plan must locate the project's CUDA-enabled interpreter or record that it is unavailable. CUDA-sensitive geometry, cache, and optimizer tests will run on the RTX 5090 environment when available. No long training run is required for this correctness branch.

The final suite will run without an extra quiet flag and will write JUnit XML. Pass counts will be read from XML attributes. Verification also includes `git diff --check`, import/compile checks, focused check scripts, a clean status review, and confirmation that the live checkout was never changed by this branch.

## Documentation and commit structure

The branch will maintain `docs/2026-07-09-edits.md` as the single dated post-edit note. The closure ledger will live under `docs/audits/` and will distinguish `FIXED`, `FAIL_CLOSED`, `RELABELED`, `INTENTIONAL`, and `DEFERRED_PERFORMANCE`. Commits will be phase-scoped so a numerical or semantic cluster can be reviewed without unrelated reporting changes.

Implementation does not begin until this written specification has been committed, self-reviewed, and accepted by the user. A detailed TDD plan follows that acceptance.
