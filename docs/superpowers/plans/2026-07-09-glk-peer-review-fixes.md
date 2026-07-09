# GL(K) Peer-Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:subagent-driven-development` to execute this plan task by task, with a separate review gate for every task.

**Goal:** Correct every defensible mathematical, empirical, bibliographic, reproducibility, and knowledge-base finding from the 2026-07-09 referee review while preserving unsupported historical results only when they are labeled as historical and unverified against the current package.

**Architecture:** The canonical LaTeX sources and research wiki are revised in the clean Research worktree. The detailed referee report, execution plan, and daily edit record live in the clean V3 worktree. Mathematical claims are corrected at their definitions and propagated to summaries; empirical claims are partitioned into current-supported, legacy-only, and unsupported-current evidence classes; the wiki receives a new immutable revision record and explicit supersessions rather than rewriting its historical source note.

**Tech Stack:** LaTeX, BibTeX, Markdown, Obsidian wikilinks, PowerShell, Python, SymPy, Git.

## Global Constraints

The canonical manuscript files are `C:\tmp\Research-glk-fixes-20260709\manuscripts\GL(K)_attention.tex`, `GL(K)_supplementary.tex`, and `references.bib`. Do not edit the older `V3_Transformer\Manuscripts-Theory` mirrors. Preserve the user's live WIP in both original checkouts. Use American English. In manuscript prose, do not introduce horizontal rules, the phrases “key insight,” “crucially,” “critically,” “notably,” “importantly,” “it’s worth noting,” “fundamentally,” “leverages,” or “underscores,” or the LaTeX spacing macros `\;`, `\,`, and `\!`. Add punctuation to display equations. Make surgical changes: do not invent replacement numbers, evidence, code-release artifacts, or theoretical guarantees. Keep the ambient full-Gaussian `GL(K)` invariance theorem, Gibbs/KKT attention result, Gaussian Fisher formulas, SPD retraction, and exact covariance-gap identity. Distinguish an ambient theorem from the implemented diagonal and single-exponential realization. Preserve historical empirical tables only with explicit provenance and verification limits. Do not change V3 source code or configuration. Update the single V3 daily record `docs/2026-07-09-edits.md`. Do not modify the old immutable wiki source `sources/manuscripts/gl-k-attention.md`. Every wiki claim changed because of this review must cite `[[gl-k-attention-2026-07-09-review-revision]]`. Do not claim completion without actual LaTeX, BibTeX, asset, reference, symbolic, and wiki-lint outputs.

## Task 1: Materialize the review package and execution record

**Files:**

- Add: `C:\tmp\V3_Transformer-glk-fixes-20260709\docs\reviews\2026-07-09-glk-manuscript-peer-review.md`
- Add: `C:\tmp\V3_Transformer-glk-fixes-20260709\docs\superpowers\plans\2026-07-09-glk-peer-review-fixes.md`
- Modify: `C:\tmp\V3_Transformer-glk-fixes-20260709\docs\2026-07-09-edits.md`

Copy the already verified review artifact from the live V3 checkout without changing its bytes. Record its SHA-256 `5054879310967C48C094EE6BEB8D7DC10E6726F70039D57E25EEB963DD69418B`, both canonical pre-revision manuscript hashes, the reviewed V3 `origin/main` commit `e504f1c5ad5d277f653534cfc7fb63fd3b1bee61`, and the two fresh worktree branches in the daily edit document. State that this is documentation and manuscript work only and that no V3 code or configuration changed.

Verify with `Get-FileHash -Algorithm SHA256`, `git diff --check`, and `git status --short` in the V3 worktree. Commit only these V3 documentation files.

## Task 2: Correct the mathematical core in the main manuscript

**Files:**

- Modify: `C:\tmp\Research-glk-fixes-20260709\manuscripts\GL(K)_attention.tex`

Make the following linked corrections throughout the abstract, introduction, derivations, status tables, limitations, and conclusion.

First, in Regime I use `\Omega_{ij}=U_iU_j^{-1}` and state that a constant value on all attended pairs must be the identity when self-edges or a transitive triple are present. Derive the strict identity-transport isotropic score. Treat a general head-space bilinear `M=W_QW_K^\top` only as a structural compatibility morphism unless a separate transformation law is supplied; do not identify it with nonidentity constant Regime-I transport. Reclassify the affected correspondence claims as structural rather than exactly derived.

Second, define the canonical and entropy-suppressed objectives with the same non-attention term `P`, retain the exact covariance-gap equation, and state that their vector fields coincide if and only if that gap vanishes. Remove the claim that canonical joint stationarity is sufficient. Describe the implemented one-step row update as a natural-gradient filtering step, not exact coordinate ascent.

Third, preserve the full-Gaussian ambient invariance theorem but state that a single real matrix-exponential chart is not a global action on `GL^+(K)` and that diagonal SPD covariances are not closed under general congruence. Identify the exact covariance-preserving subgroup as the monomial group and label diagonal projection as an approximation.

Fourth, replace the gauge-frame natural-gradient overclaim with accurate conditioning language. Distinguish the regularized Cartan/Killing preconditioner from the extrinsic Frobenius pullback; do not call either a demonstrated full-`GL(K)`-covariant natural gradient. Mention a direct group-element, left-invariant metric only as an alternative, not as the implemented method.

Fifth, replace the Regime-I renormalization claim by the vertex-fluctuation result `g_2=O_{RMS}(n^{-1/2})`, `y_2=-1/2`, and exact `H_{ijk}=I`, hence `g_3=0` with no Regime-I holonomy exponent. Reserve `n^{-1}` behavior for an explicitly independent-edge ensemble.

Also narrow LayerNorm language to a conditional norm-cancellation mechanism, label the identity-backward sign estimator as a biased straight-through derivative, delete the unsupported compute crossover formula, and state that the single-step filter is not an argmin. Update the abstract and conclusion so they no longer claim that historical BERT, mechanism, or full-gauge-realization evidence validates the ambient theory.

Verify all algebraic counterexamples and identities with an exact SymPy script run from standard input; record its printed assertions in the task report. Search the main source for every stale claim named above and run `git diff --check`.

## Task 3: Correct the supplement, forward-KL theorem, and empirical scope

**Files:**

- Modify: `C:\tmp\Research-glk-fixes-20260709\manuscripts\GL(K)_supplementary.tex`

Reconcile all supplement statements with Task 2. Replace the entropy identity by `D_KL(q||p)=<H>_q-S(q)`. Distinguish dot-product evaluation cost `O(d_k)` from the `O(sqrt(d_k))` root-mean-square magnitude under centered unit-variance independence. Correct the frame-metric section to the implemented regularized preconditioner and state the scope of the extrinsic Frobenius pullback. Split the Regime-I vertex ensemble from the independent-edge synthetic check. Replace a pure perplexity power law by `L(D)=L_\infty+AD^{-\beta}` and `PPL(D)=\exp L(D)`, and remove the unsupported crossover estimate.

Strengthen the forward-KL necessity theorem: require one fixed admissible witness configuration whose ratio has a nonempty open essential range, or explicit overlap conditions equating configuration-specific constants. State that closure selects the positive KL ray `f_c(t)=c(t\log t-t+1)`, while `f''(1)=1` or the exact half-exponent target fixes `c=1`. State that the witness assumption is not automatic for diagonal Gaussians under general transport.

For BERT and legacy analyses, state the selection reuse, head/passages clustering, posterior-interpretation, and provenance limits. Describe the reported 105-passage result as historical exploratory evidence that has not been reproduced against the current package. Do not treat a posterior interval containing the finite-dimensional values as resolution of the finite-dimension question. Describe the width curves and single-seed ablations as descriptive rather than universal scaling laws or causal mechanism tests. Correct statements that the reported runs used a gauge-frame natural gradient when their configuration did not.

Use the same exact SymPy verification record as Task 2, search for stale theorem and scaling claims, and run `git diff --check`.

## Task 4: Repair literature positioning, bibliography, and reproducibility statements

**Files:**

- Modify: `C:\tmp\Research-glk-fixes-20260709\manuscripts\GL(K)_attention.tex`
- Modify: `C:\tmp\Research-glk-fixes-20260709\manuscripts\GL(K)_supplementary.tex`
- Modify: `C:\tmp\Research-glk-fixes-20260709\manuscripts\references.bib`

Narrow the novelty statement using direct precursors: structured and variational attention, attention as implicit structural inference, Gauge Equivariant Transformer, Energy Transformer, and later inference/free-energy interpretations. State the remaining claim precisely as the manuscript's combination of transported Gaussian posteriors, token-local frames, and a `GL(K)` variational construction. Correct the Sengupta characterization: the paper includes geometric parallel transport, but not the manuscript's token-indexed closed-form matrix transport and transformer reduction. Add verified primary BibTeX records, including the missing `Neal1998` entry, without inventing publication metadata.

Remove the fourteen references to missing legacy graphics because the associated BERT, mechanism, clustering, and historical-model numerical claims are not supported by the current release and the recovered files do not repair selection reuse, clustered dependence, or commit provenance. Do not import those legacy binaries or scripts. Retain the two existing fixed-`GL^+(10)` development-sweep images only if their displayed values agree with the audited fits; otherwise remove their figure references and report the exact values in text and tables. Do not delete preexisting image files merely because they become unreferenced.

Replace Code Availability with the current verified state: name the actual public V3 repository URL and reviewed commit, state that the tracked package does not yet contain the manuscript figures, experiment configurations, test/provenance manifests, or a one-command reproduction path, and identify the recovered historical artifacts separately. Do not claim that `requirements.txt` or a release archive exists.

Verify that every citation key resolves, every remaining `\includegraphics` target exists, no binary was added, and `git diff --check` passes.

## Task 5: Ingest the revision into the Research wiki

**Files:**

- Add: `C:\tmp\Research-glk-fixes-20260709\sources\manuscripts\gl-k-attention-2026-07-09-review-revision.md`
- Modify: `C:\tmp\Research-glk-fixes-20260709\manuscripts\verified-ledger.md`
- Modify: `C:\tmp\Research-glk-fixes-20260709\index.md`
- Modify: `C:\tmp\Research-glk-fixes-20260709\log.md`
- Modify only the wiki pages containing stale versions of the six corrected claims.

Create a new immutable manuscript source note with the title “Attention as Gauge-Theoretic Variational Inference: 2026-07-09 Review and Revision Record,” author Robert C. Dennis, year 2026, status “in preparation (major revision),” `created` and `updated` dates 2026-07-09, and the appropriate gauge, attention, VFE, information-geometry, SPD-geometry, transformer, multi-agent, CS/ML, mathematics, statistics, and physics tags. Record the review hash, all pre-revision hashes, the reviewed V3 commit, final manuscript hashes, and the Research branch commit available at ingest time. Summarize the six scope corrections and distinguish surviving results from superseded claims. Preserve `sources/manuscripts/gl-k-attention.md` byte-for-byte.

Annotate rather than delete superseded ledger entries, then add `## 5. 2026-07-09 counterexample-backed supersessions` before Provenance. Correct the constant-transport status, stationarity claim, Regime-I exponents and holonomy, forward-KL range hypothesis, realized-family status, and former “review-exhausted” or submission-ready language. Do not demote the results that remain proven.

Update the central pages for GL(K) attention and group structure, renormalization, holonomy, Killing/Frobenius metrics, the VFE Transformer program, attention precursors, geometric deep learning, and opinion pooling. Use an `rg` sweep to find repeated stale language elsewhere, but edit only actual contradictions. Preserve belief-side Fisher natural-gradient statements. Add the new source to `index.md`, update the manuscript-source count, append an `INGEST` entry to `log.md`, and append a `LINT` entry only after an actual clean lint run.

Verify the old source hash is unchanged, all changed claims backlink to the new source, and `python docs/_lint.py` reports zero broken wikilinks, grey graph nodes, empty files, case collisions, and identity collisions.

## Task 6: Build, cross-check, and adversarially review the integrated revision

**Files:**

- Modify only files needed to repair defects found by the verification and review gates.

Run an exact symbolic verification for the covariance-gap counterexample, diagonal-family counterexample, entropy sign, metric mismatch/invariance examples, Regime-I cocycle cancellation, and forward-KL counterexample. Compile both manuscripts through LaTeX and BibTeX from clean auxiliary directories. If cross-document references require `xr-hyper`, add the symmetric external-document declarations and confirm the `.aux` sequence resolves them; do not hide unresolved references. Verify all graphics and citation keys with independent scripts. Run prohibited-pattern and stale-claim searches over both TeX files. Run the Research wiki linter and inspect every nonzero category. Run `git diff --check` in both worktrees.

Dispatch one theory reviewer and one empirical/reproducibility reviewer over the complete Research diff. Resolve every Critical or Important finding, rerun the covering checks, and have a final whole-branch reviewer confirm both specification compliance and manuscript quality.

Commit the verified Research revision on `fix/glk-peer-review-findings-20260709`. Record final hashes and commit IDs in the V3 daily edit record and the new wiki source; if recording the commit ID would create a self-referential commit problem, record the parent/base and final file hashes instead and describe that choice explicitly.

## Task 7: Copy verified deliverables to the canonical live paths

Copy only the changed GL(K) manuscript, bibliography, new source note, ledger, index, log, and surgically changed wiki pages from the clean Research worktree to `C:\Users\chris and christine\Desktop\Research`. Do not touch `.obsidian\graph.json`, PIFB build outputs, or any other unrelated live file. Copy the verified review, plan, and final daily-edit update from the clean V3 worktree into `C:\Users\chris and christine\Desktop\V3_Transformer\docs`, preserving unrelated live documentation changes.

Recompute canonical-live file hashes after copying and compare them with the worktrees. Show `git status --short` for both clean branches and both live checkouts. Do not merge, push, discard, or clean the user's live branches without a separate request.
