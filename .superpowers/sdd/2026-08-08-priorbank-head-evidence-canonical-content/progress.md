# SDD ledger — plan: docs/superpowers/plans/2026-08-08-priorbank-head-evidence-canonical-content.md

Setup: isolated linked worktree verified on codex/priorbank-head-evidence-canonical-content-20260808.
Setup: focused baseline passed before implementation; plan pre-flight found no internal conflicts.
Task 1: fix round 1/5 (2 addressed, 0 open — registry identity gate; disabled irrep metadata parity; commits be40b46..8653435)
Task 1: complete (commits f9c0ab5..8653435, review clean)
Task 2: complete (commits 8653435..7cee3e3, review clean)
Task 3: minor (deferred): add head-evidence-enabled non-PD full-query fused parity and gradient-reduction coverage; initial report overclaimed this boundary.
Task 3: fix round 1/5 (2 addressed, 0 open — hoisted evidence LHS; restored fp64 temperature island; commits f1d7b88..e056abc)
Task 3: complete (commits 7cee3e3..e056abc, review clean; 1 deferred minor)
Task 4: review conflict awaiting human ruling — approved constraints allow the existing optional unigram decoder bias, while reviewer recommends rejecting it to make total logits pure intrinsic Gaussian KL.
Task 4: human ruling — allow decode_unigram_prior; exactness applies to the intrinsic Gaussian divergence, while unigram bias remains a separate additive vocabulary base-rate term. Pin the separation with tests and provenance text.
Task 4: minor (deferred): add explicit diagonal_chunked parity coverage for the approved post-divergence unigram-bias separation.
Task 4: fix round 1/5 (1 addressed, 0 open — human-approved unigram separation pinned by independent oracle and provenance; commits 3e518c8..cede686)
Task 4: complete (commits e056abc..cede686, review clean; 1 deferred minor)
Task 5: minor (deferred): move CanonicalFrameContext ownership into contracts.py to avoid contracts importing the model helper module before Task 6 broadens consumers.
Task 5: fix round 1/5 (3 addressed, 0 open — authoritative truncated transport; single-head factored frames; AMP dtype preservation; commits a4d3ba3..553aece)
Task 5: complete (commits cede686..553aece, review clean; 1 deferred minor)
Task 3: deferred minor resolved by Task 6 commit 5a465c3 - non-PD head-evidence full fused exclusion and gradient parity are pinned.
Task 5: deferred minor resolved by Task 6 commit 5a465c3 - CanonicalFrameContext ownership moved to contracts.py for shared consumers.
Task 6: fix round 1/5 (2 addressed, 0 open - projected decoder identity gate; nonvacuous N>1 decode_last and multistep generation coverage; commits 5a465c3..197ab31)
Task 6: complete (commits 553aece..197ab31, re-review clean)
Integrated review: fix round complete (PriorBank evidence-role provenance; canonical encoder/table-family/full-family fail-closed gates; production-faithful projected extraction replay; commits 8f81cab..dae4bbe, final review clean)
Task 7: documentation complete (README/design distinguish exact phi cancellation, projected realized-frame pullback, post-divergence unigram base rate, fail-closed identities, diagnostics, and run-artifact provenance)
Task 7: final-source verification complete at dae4bbe (CUDA focused JUnit: 230 passed; broader JUnit: 137 passed, 1 skipped; zero failures/errors; broader lane used MKL_THREADING_LAYER=SEQUENTIAL to avoid the reproduced Torch/SciPy duplicate-OpenMP abort)
Task 7: static/ownership checks complete (feature Ruff clean; git diff --check clean; protected-path diff empty; deferred source marker occurs once; known unrelated config-default assertion unchanged)
Task 4: minor (still deferred): explicit diagonal_chunked parity coverage for the approved post-divergence unigram-bias separation was not added during docs-only Task 7 because the already-implemented behavior cannot supply a genuine RED phase.
Task 7: pre-ledger report/artifacts ready; custom closure ledger will be activated only after the scoped Task 7 commit and will remain uncommitted to preserve the activated revision.
Integrated review fix round 1: findings 1, 2, 3, and 5 closed by shared projected preparation, import-time identity gates, and direct PriorBank contract validation; 139 focused and 291 regression tests pass. Finding 4 is owned separately.
Integrated review fix round 2: projected `gaussian_diagonal` canonical-table identity is now pinned at config and direct PriorBank seams; RED was 2 expected DID-NOT-RAISE failures, GREEN is 314 affected tests passing.
