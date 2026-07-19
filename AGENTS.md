# V3_Transformer (VFE_3.0)

Clean-room rebuild of the gauge-theoretic VFE transformer. No neural networks:
all capacity comes from iterative VFE minimization over Gaussian belief tuples
`(mu, Sigma, phi)`. Built bottom-up, every layer pinned by golden regression
tests. See `docs/superpowers/specs/2026-05-29-vfe3-clean-room-design.md`.

## Hard constraints
- NO neural networks (no nn.Linear, no MLP, no activations).
- NO CLI arg parsing; entry points are click-to-run (edit config dicts, then run).
- float32 throughout; CUDA where applicable (user has an RTX 5090).
- High modularity: a config-selected registry behind every seam (divergence,
  alpha_i, family, transport/gauge, retraction, decode). Add a variant by
  writing-and-registering it, never by editing call sites.
- Always preserve a theoretically pure path under appropriate toggles.

## Function signature convention (MANDATORY)
Argument order: all torch.Tensor first, then 'float | torch.Tensor', then
undefined floats, undefined ints, undefined bools, then defined floats,
defined ints, defined bools, then Optional, then **kwargs last.

Vertical alignment: names, type annotations, `=` signs, and trailing `#`
comments are each aligned to a common column. Blank lines separate type
groups. Tensor shape comments at critical points. Type hints on every
signature. Docstrings carry the LaTeX/math form for non-trivial formulas.
Variable names match paper notation (mu_q, sigma_q, alpha, kappa).

Example:

    def kernel(
        mu_q:    torch.Tensor,             # (..., K) query means
        sigma_q: torch.Tensor,             # (..., K) query variances

        *,
        alpha:   float = 1.0,
        kl_max:  float = 100.0,
        eps:     float = 1e-6,
    ) -> torch.Tensor:

## Testing
Golden regression tests pin every kernel to its reference values;
finite-difference gradient checks against the autograd-of-F oracle (later
phases); property tests (non-negativity, self-divergence zero, gauge
equivariance). Tests are device-agnostic (default CPU; set
VFE3_TEST_DEVICE=cuda for the GPU).

### Tooling & verification discipline (MANDATORY)
- **Pass counts come from a machine-readable source, never from memory.** `pyproject.toml`
  already sets `addopts = "-q"`. Adding `-q` again on the command line makes `-qq`, which
  SILENTLY SUPPRESSES pytest's `N passed` summary line (stdout ends at `[100%]`, exit code
  still 0). Either run pytest with no extra `-q`, or `--junitxml=out.xml` and read
  `testsuite tests=/failures=/errors=`. Do NOT add `-q`. Do NOT report a pass count you did
  not read from that line or the XML.
- **Never assert a fact that is not in an actual tool result.** During the 2026-05-30 audit
  fixes, fabricated pass counts ("188/189 passed") and a phantom `XYZZY_AUDIT_PROBE` diff were
  written into commits/docs though no tool ever returned them — model hallucination, not a
  tooling fault. Before claiming a test result, a file's contents, or a diff: quote the tool
  output that shows it. If you cannot, you do not know it.
- **A test edit can silently no-op.** An `Edit` against stale/wrong content fails to apply;
  grep that the new test name is actually in the file before claiming "+N tests."
- **PowerShell `>` redirection writes UTF-16LE+BOM** (reads back as `\xff\xfe d a ...`). Use
  `-Encoding utf8`, or the Bash tool, when another tool must read the file.

**Post Edit Policy**:  Always write a brief post-edit description of changes made to the codebase as a .md.  The date the edits were made should be in the naming convention of the document.  there should be only one document per day.  you should update the same document as edits are made

**There should ALWAYS exist a theoretically/mathematically "pure" path under appropriate toggles.**  Computationally extreme paths should be 'opt in' toggles and clearly documented.


**CODE FOCUS** when investigating and/or auditing the codebase do NOT rely on code comments....focus on the actual code and paths

**user has RTX5090 GPU** - use cuda and code accordingly where applicable


## Mandatory Git Lifecycle

For every task that changes files, the following workflow is the default definition of done. The
user does not need to repeat it.

1. Before editing, run `git fetch origin` and inspect the current `origin/main` log.
2. Never work directly in the user's existing checkout. Create a separate temporary git worktree
   and a fresh task branch from `origin/main`.
3. Never stash, reset, clean, restore, overwrite, or otherwise alter pre-existing user changes.
4. Make all task changes only inside the temporary worktree.
5. Track every file created by the task, including reports, test artifacts, patches, logs, and
   temporary files. Delete task-owned temporary artifacts before completion, but never delete files
   that predated the task.
6. Run verification appropriate to the change and inspect the actual exit status and
   machine-readable results. Update the required dated post-edit document.
7. Inspect `git status --short` and the staged diff, then commit every intended task change.
8. Push the task branch to `origin`. After verification succeeds, merge the task branch into `main`
   and push `main`.
9. Fast-forward the user's local `main` checkout only when doing so cannot alter or overwrite user
   WIP. If WIP prevents a safe fast-forward, leave it untouched and report that explicitly.
10. After confirming the merge and push, remove the temporary worktree and delete the local task
    branch. Run final cleanliness checks and report the task branch, commit SHA, pushed remote
    branch, resulting `origin/main` SHA, verification result, worktree removal, and final
    `git status --short`, including the owner of any remaining dirty files.

A task is not complete merely because a file was edited. It is complete only after verified changes
are committed, pushed, merged into `main`, `origin/main` is updated, safe local fast-forwarding is
performed, and task-owned temporary artifacts are removed. If permissions, conflicts, failing
verification, remote changes, or user WIP prevent any step, stop and report the exact blocker. Never
silently substitute "changes are ready" for the required completed lifecycle.

Do not claim that a repository is clean without showing the final `git status --short`. Do not claim
that the remote is updated without fetching and inspecting `origin/main`.

## Mathematical Reference

Minimal equations for code review — see manuscripts for full derivations.

**VFE hierarchy**: `h → s → p → q → observations` (hyper-prior → models → priors → beliefs → data)

**Free energy** (canonical form, manuscript `\label{eq:free_energy_functional_final}`):
```
F = alpha * KL(q_i || p_i)                                          # self-coupling: beliefs to priors
  + lambda_h * KL(s_i || h)                                         # hyper-prior: models to centroid
  + sum_ij [ beta_ij  * KL(q_i || Omega_ij * q_j)
             + tau * beta_ij  * log(beta_ij  / pi_ij) ]             # belief coupling + attention entropy
  + sum_ij [ gamma_ij * KL(s_i || Omega_ij * s_j)
             + tau * gamma_ij * log(gamma_ij / pi^(s)_ij) ]         # model coupling + meta entropy
  - E_q[log p(o | x)]                                               # observation likelihood
```
tau = kappa * sqrt(K) is the effective softmax temperature. The tau * beta_ij * log(beta_ij/pi_ij) term is the attention-distribution entropy with uniform prior pi_ij = 1/N; it is required for the softmax β to be a stationary point of F (without it the row-Lagrangian gives a delta, not softmax). The canonical F vs "entropy-suppressed surrogate" sum β KL distinction is made explicitly in Manuscripts-Theory/GL(K)_attention.tex  (the surrogate is acknowledged again in Manuscripts-Theory/GL(K)_supplementary.tex ) — their gradients differ by -tau^{-1} Cov_β(KL, ∇KL). See Manuscripts-Theory/PIFB.tex for the FULL GENERAL theory

## Communication

**Humility.** Say "I don't know" when unsure. Honest uncertainty beats confident speculation — acknowledge what needs verification.

**Be direct.** State errors and concerns plainly: "This is wrong because X," not "this might be slightly off." Ultra-think and double-check.

**Push back.** Challenge gaps in derivations; ask for justification and proof. Maintain position under pushback — ask "What am I missing?" rather than capitulating.

**No bullshit.** If a correspondence is interpretive rather than mathematically exact, say so. Admit gaps; never dress up hand-waving as theorem. When asked "what does X have to do with anything?" and the answer is "not much," say that.

**Verify with citations** for theoretical and mathematical claims. use /literature-review skill.

**Skip praise preambles.** No "Great question!" or "Excellent point!" — engage with the substance. no sycophancy
## Before Coding

**Plan first.** State assumptions explicitly; if uncertain, ask. If multiple interpretations exist, present them — don't pick silently. If something is unclear, stop, name it, and ask.

**Simplicity first.** Minimum code that solves the problem, nothing speculative: no features beyond what was asked, no abstractions for single-use code, no unrequested configurability, no error handling for impossible scenarios. If you write 200 lines and it could be 50, rewrite it. Flag over-engineering and ask what the complexity buys. The test: "Would a senior engineer say this is overcomplicated?"

**Surgical changes.** Touch only what you must. Don't improve or refactor adjacent code that isn't broken; match existing style even if you'd do it differently. Remove imports/variables/functions your changes orphaned, but leave pre-existing dead code — mention it, don't delete it. Every changed line should trace directly to the request.

## Writing Style

Write in academic prose — flowing paragraphs with clear logical progression, not bullet points. Minimize itemizations and enumerations; if content can be a paragraph, make it one. Use /literature-review, /scientific-writing, /sympy, and other relevant skills.

**Scientific writing rules.** Do not use LaTeX spacing macros (`\;`, `\,`, `\!`) — banned in this project's docs. Apply standard equation punctuation (comma/period at end of display equations) in any doc cleanup pass.

**Banned patterns** (Codex-isms, never in manuscripts): horizontal rules (`---` and `--`), "key insight," "crucially," "critically," "notably," "importantly," "it's worth noting,"  "fundamentally,"  "leverages," "underscores."

<!-- BEGIN VERIFICATION PROJECT POLICY -->
## Project verification control plane

For this repository, invoke the installed `verification` skill for audits, proofs, correctness
claims, experimental results, and source-backed factual claims. Use its validated claim ledger as
the closure record. Every claim is exactly one of `CANDIDATE` (queued), `LLM_SUPPORTED` (reasoned
triage without closure evidence), `EVIDENCE_VERIFIED` (closed by current supporting evidence),
`REFUTED` (closed by current contradicting evidence), or `INCONCLUSIVE` (an open obligation remains).
Closure mode rejects `CANDIDATE` and `LLM_SUPPORTED`.

Read executable paths and active configuration rather than relying on comments. Code and
experiment closure requires current mechanical or reproduced-output evidence, with commands,
exit status, configuration, and artifact revision recorded. Mathematical closure requires a
derivation or formal proof; numerical checks are supporting evidence only. Research, source, and
general factual closure requires a current primary source or reproduced-source record. LLM output,
agent consensus, comments, and remembered results cannot independently close a claim.

Evidence is revision-bound. Any relevant edit, configuration or dependency change, new run, input
change, or artifact-revision mismatch invalidates the affected evidence and requires re-running the
verification. Pytest totals and failures must be read from current machine-readable output such as
JUnit XML. Missing eligible evidence or unresolved disagreement is `INCONCLUSIVE` with a specific
open obligation. Before reporting closure, run the verification skill's deterministic ledger
validator and name the validated ledger in the final response.
<!-- END VERIFICATION PROJECT POLICY -->
