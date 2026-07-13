# Validated July 13 Audit Remediation Plan

## Objective

Correct the source-verified findings from `audit-2026-07-13.md` and `audit-2026-07-13-pass2.md` while preserving the pure path, existing registry boundaries, the distinct semantics of raw and EMA snapshots, and the user's live configuration work.

## Phase 1: Artifact integrity, analysis, and tooling

Add focused regression tests for per-figure failure isolation, streaming provenance, corpus-bound scaling caches, deterministic cell loaders, same-estimator bootstrap intervals, finite Pareto inputs, exact package discovery, result-directory ignore policy, shared slugging, manual large-figure propagation, and the audit-check node list. Update the implementation only after each test or static policy check demonstrates the defect. Remove the superseded free-energy-descent registry entry and its now-orphaned implementation, and remove only imports orphaned by these changes.

## Phase 2: Runtime state and diagnostic consistency

Add regression tests for tensor-valued EFE scores, one-per-forward model-channel encoding, device-resident CG diagnostics, diagnostic logging, metric annotations, EMA key reconciliation, barycenter consistency after EMA copy and temporary restore, supported `sigma_max=None`, exact gamma log priors, positive finite trust and Metropolis parameters, and cache parity for every allowlisted causal prior. Reuse a post-evaluation snapshot only where the model state and requested weights are identical.

## Phase 3: Geometry and numerical correctness

Add precision, equivariance-contract, Cholesky-mask, guarded-logdet, ALiBi, Regime-II trivial-gauge, Mahalanobis, Laplace transport, belief-spectrum, and sigma-gate regressions. Implement a batched equal-block full-covariance transport path and compare it numerically with the retained reference fallback. Keep family-specific scale semantics behind a selected transport policy rather than a call-site family conditional.

## Phase 4: Test-policy and visualization performance

Identify tests that instantiate models or execute dense geometry at `K >= 6`, reduce those fixtures to dimensions below six, and leave pure configuration-parser tests unchanged when their stated subject requires a larger number. Replace per-call UMAP interpreter startup with one crash-isolated worker per report and preserve cleanup on success and failure. Gate the two long UMAP tests under the existing slow-test policy.

## Phase 5: Consolidated verification

Run the focused tests for all changed clusters on CPU, then the CUDA-sensitive subset with `VFE3_TEST_DEVICE=cuda`. Run the audit verifier, compilation and static checks, package discovery check, and one complete pytest invocation with `--junitxml` and no extra quiet flag. Read the final counts and failures from XML. Delete task-owned XML and scratch output after recording the results in `docs/2026-07-13-edits.md`.

## Phase 6: Review and Git lifecycle

Inspect the complete diff, run an independent review pass, resolve actionable findings, and repeat affected verification. Force-add ignored documentation, commit every intended change, push `codex/fix-validated-7-13-audits`, merge it into a clean `main` integration worktree, push `main`, fetch, and verify the resulting `origin/main` log. Do not modify the user's dirty audit checkout. Remove the temporary worktrees and local task branch after confirming the remote state.
