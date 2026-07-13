# Validated July 13 Audit Remediation Design

## Scope

This remediation covers the findings in the two July 13 audits that survived direct source inspection or runtime probing. It does not treat every proposed audit remedy as authoritative. Where the investigation found a narrower defect than the audit described, the implementation will correct that narrower defect and preserve valid behavior.

The work starts from fetched `origin/main` in an isolated worktree. The user's dirty audit checkout and its configuration changes remain untouched.

## Design

The artifact and analysis changes will isolate figure failures per registry entry, stream provenance hashes, bind scaling caches to the corpus, make per-cell loader seeding explicit, remove the superseded free-energy-descent figure, log diagnostic failures, filter nonfinite Pareto timing, use a single filesystem-safe slug policy, reuse an isolated UMAP worker within one report, and make packaging and result-ignore rules exact. Scaling bootstrap intervals will use the same estimator family as the reported point estimate.

The runtime changes will keep CG diagnostics on the accelerator until logging, reuse model-channel encodings within a forward, enforce tensor-valued EFE score accumulation, restore metric annotations, share only semantically identical post-evaluation snapshots, validate inert or invalid toggles, and make EMA restoration and barycenter evaluation consistent with the live model. The pre-step diagnostic snapshot will remain separate from post-step and EMA evaluation because those states are intentionally different.

The mathematical and numerical changes will retain supported input precision, propagate failed-Cholesky masks, use the guarded Cholesky log determinant, adopt the reference ALiBi slope construction, remove the trivial-gauge Regime-II bypass, document the two trust-region geometries, harden Mahalanobis solves with an exact higher-precision path plus an explicitly approximate fallback, transport Laplace scales with degree-one homogeneity through a family-selected seam, expose nonpositive belief eigenvalues, require enough sigma-gate samples for finite statistics, and use exact log-softmax priors in gamma entropy. Query-adaptive temperature will remain an opt-in gauge-breaking baseline because a covariance trace is not invariant under general `GL(K)` transformations.

The performance change for full-covariance transport will add a batched equal-block fast path and retain the heterogeneous-block fallback. This avoids changing representation semantics merely to remove Python loops. The testing-policy repair will reduce model-building CPU tests to `K < 6`; tests whose subject is a dimension value but which do not build expensive models may retain the value.

## Finding disposition

Pass 1 findings 0 through 18 receive code, test, or explicit-contract changes. Finding 4 receives explicit loader-generator seeding even though the global seed currently supplies a fallback. Finding 11 is documented as an opt-in non-equivariant baseline rather than represented as gauge-pure. Finding 13 is clarified as two configuration-selected geometric constraints rather than collapsed into one formula. Finding 18 receives an inert-toggle warning on the linear decoder.

Pass 2 findings 0 through 21 receive code, test, tooling, packaging, or policy changes. Finding 9 shares only the two post-evaluation attention snapshots; the earlier raw diagnostic snapshot is not merged with them. Finding 13 does not claim that float64 cancels the covariance perturbation introduced by regularization. Finding 14 is implemented through the existing family registry boundary. Finding 17 recomputes the derived barycenter after EMA copies and restores it after temporary evaluation so the raw state is not lost.

## Verification

Each behavior change begins with a focused regression test or an existing failing policy check. Focused CPU tests use dimensions below six. CUDA-sensitive numerical, transport, and training paths receive an RTX 5090 run through `VFE3_TEST_DEVICE=cuda`. The final suite runs once with JUnit XML, and counts are reported only from that machine-readable artifact. Packaging, static checks, and the click-to-run audit verifier are checked separately.

## Completion

The branch will be reviewed as a complete diff, committed, pushed, merged into `main`, and pushed again. The live dirty checkout will be fast-forwarded only if Git can do so without touching user-owned changes; otherwise it will remain unchanged and the blocker will be reported. The temporary worktree and task-owned validation artifacts will be removed after remote verification.
