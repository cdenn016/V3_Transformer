# V3_Transformer (VFE_3.0)

Clean-room rebuild of the gauge-theoretic VFE transformer. No neural networks (backprop is allowed):
all capacity comes from iterative VFE minimization over Gaussian belief tuples
`(mu, Sigma, phi)`. Built bottom-up, every layer pinned by golden regression
tests. See `docs/superpowers/specs/2026-05-29-vfe3-clean-room-design.md`.

This V3 is a production quality gauge-theoretic VFE transformer that allows clean code, clear math, and future expandability

## Git Workflow

- Before making any changes, create a dedicated fresh branch off `main` (see "ALWAYS BRANCH FRESH FROM MAIN" below) and keep a tidy worktree.
- NEVER stash, revert, discard, or modify the user's live/WIP (uncommitted) config files or config toggles during git operations. They are intentional working state, not noise to clean up.
- Treat the remote as the source of truth. Before making any sync claim, run `git fetch` and inspect the ACTUAL remote state — show `git log origin/main` — rather than trusting stale local state.
- When asked to make the local match `origin/main`: run `git fetch` first, show `git log origin/main`, treat the remote as authoritative — but still do not touch uncommitted config files.

## Hard constraints
- NO neural networks (no nn.Linear, no MLP, no activations. backprop is allowed).
  **Documented exceptions (opt-in toggles, both default OFF; the pure no-NN path is the
  default and always exists):** (1) `use_prior_bank=False` decodes via a single learned
  linear output projection `logits = mu @ W^T` (`W` a raw `(V, K)` nn.Parameter, not an
  nn.Linear module; sigma discarded) — the linear-decode ablation the user compares
  against the KL-to-prior decode. Encode and the free-energy self-coupling stay on the
  PriorBank. (2) `use_head_mixer=True` applies a learned Schur-commutant per-irrep-block
  head mixer; under `block_glk`'s untied per-block gauge it breaks strict gauge
  equivariance (exact at identity init, deviates as the mixer drifts) — user-accepted. Its irrep-tower siblings (so_n/sp_n) are the isotypic per-type mixer (exactly equivariant under the tied gauge) and, under use_cg_coupling=True, learned scalar Clebsch-Gordan path weights (exactly equivariant for any weights; means-only sigma) -- both zero-init, default OFF.
  (3) `transport_mode='regime_ii'` consumes a learned bilinear connection `connection_W`
  (`(n_gen, K, K)` nn.Parameter, default OFF; flat Regime-I is the pure path). Its edge factor
  `exp(delta_ij·G)`, `delta_ij^a = mu_i^T W^a mu_j`, is gauge-INVARIANT only at `W=0` (the only
  constant W with `g^T W^a g = W^a` for all group elements g is zero), so a trained nonzero
  `connection_W` breaks strict gauge equivariance — exact at zero init, deviates as W drifts (the
  same footprint as the head mixer) — user-accepted. (Pinned by
  `tests/test_regime_ii.py::test_regime_ii_edge_factor_breaks_gauge_invariance_for_nonzero_W`.)
  (4) `t5_learnable_bias=True` learns the per-bucket T5 relative-position attention-bias table
  `b_{i-j}` (a raw `(t5_num_buckets,)` nn.Parameter read by the `t5_relative_bias` attention prior;
  default OFF, created only when a T5 channel is active, init to the fixed `-log1p(bucket)` table so
  step 0 is byte-identical). Unlike (2)/(3) this bias is a scalar function of position OFFSET only and
  touches no gauge transport, so it does NOT break gauge equivariance — the cleanest exception. Like
  the E-step-coupled params above it carries a `detach_e_step` freeze footgun. (5)
  `learnable_kappa_beta=True` / `learnable_kappa_gamma=True` learn the per-irrep-block softmax
  temperatures as `kappa = exp(log_kappa_beta/gamma)` (raw `(len(irrep_dims),)` nn.Parameters, init
  `log(cfg.kappa_*)` so step 0 is byte-identical; `exp` keeps tau > 0). Like (4) a per-block scalar
  temperature multiplies the gauge-invariant per-block energy and touches no gauge transport, so
  equivariance is preserved; `log_kappa_beta` (and `log_kappa_gamma` under `s_e_step`) carries the
  same `detach_e_step`/`straight_through` freeze footgun (`kappa_gamma` on the scored
  `lambda_gamma>0` path trains under any estimator). (6) `layernorm_affine=True` adds a learned
  per-feature affine `gamma`/`beta` to any `"layernorm"` norm seam (`mu_norm = gamma*LN(mu) + beta`,
  raw `(K,)` nn.Parameters carried by an `AffineLayerNorm` nn.Module, init `gamma=1`/`beta=0` so
  step 0 is byte-identical; default OFF, inert unless a seam is `"layernorm"`). Unlike (4)/(5) it is
  NOT gauge-preserving — the per-coordinate affine in the fixed basis does not commute with `g` in
  GL(K); it sits on the same non-gauge-pure path `"layernorm"` already occupies, adding a diagonal
  scale/shift on top of its centering (an opt-in non-equivariant baseline; the gauge-pure norms
  remain `"none"`/`"mahalanobis"`). As the BLOCK norm it is applied to the belief VALUE inside the
  stack, so (unlike (5), which enters only the E-step tangent) it trains under `unroll` AND
  `straight_through` and is frozen ONLY by the fully-detached E-step (effective `detach`); as the
  FINAL norm it trains under any estimator. (`learnable_r`, `pos_phi='learned'` are the other
  default-OFF learned-scalar/table toggles in the same family.)
- NO CLI arg parsing; entry points are click-to-run (edit config dicts, then run).
- float32 throughout; CUDA where applicable (user has an RTX 5090).
- High modularity: a config-selected registry behind every seam (divergence,
  alpha_i, family, transport/gauge, retraction, decode). Add a variant by
  writing-and-registering it, never by editing call sites.
- Always preserve a theoretically pure path under appropriate toggles.
- Codebase should be modular:  e.g. we should be able to slot in different exponential/mixture families, different f-divergences, different groups, etc)


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

### Tests are CPU-bound and MUST be SMALL — use K < 6 (MANDATORY)
- **Every test runs CPU-bound by default.** Tests instantiate TINY models. Cost scales
  steeply in the belief dimension `K` (dense `(K,K)` transports, `K x K` gauge generators,
  SPD `Sigma` ops), so a large-`K` model that trains fine on the RTX 5090 will HANG a
  CPU test run for effectively forever.
- **ALWAYS use small `K < 6` for CPU-bound tests** (`K = 2`, `3`, or `4` is the norm; the
  golden kernels are pinned at these sizes). Keep every other dim tiny too: sequence length,
  batch, `n_heads`, `n_layers`, `n_gen`, vocab — all single digits. A unit test should
  finish in well under a second.
- **NEVER run a production-scale model inside a test.** A `K = 200` (or any K in the tens or
  hundreds) model under a CPU test does NOT "run slowly" — it will take millions of years and
  never return. Production/scaling configs (`train_vfe3.py`, `scaling.py`) belong on the GPU,
  NOT in the test suite. If you need to exercise a big-`K` path, do it as an explicit GPU run,
  never as a pytest.
- **Before running any test that builds a model, check the `K` (and every other dim) in its
  config.** If `K >= 6`, it is not a CPU test — shrink it or move it to the GPU.

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

**Post Edit Policy**:  Always write a brief (BRIEF DAMMIT!) post-edit description of changes made to the codebase as a .md.  The date the edits were made should be in the naming convention of the document.  there should be only one document per day.  you should update the same document as edits are made

**There should ALWAYS exist a theoretically/mathematically "pure" path under appropriate toggles.**  Computationally extreme paths should be 'opt in' toggles and clearly documented.

**Audit Instructions** - when auditing the code base dont concern yourself whether default config toggles are theoretically pure.  rather, concern yourself with whether the theoretically pure paths exist.  i am constantly changing toggles.


**CODE FOCUS** when investigating and/or auditing the codebase do NOT rely on code comments....focus on the actual code and paths

**user has RTX5090 GPU** - use cuda and code accordingly where applicable

**ALWAYS BRANCH FRESH FROM MAIN** - each session should be a fresh branch from main and you should maintain a tidy worktree!

**DONT LEAVE MESSES!!** ALWAYS CLEAN UP temp FILES FROM ATOMIC EDITS AND SUCH WHEN FINISHED!


## Project Conventions

- Do NOT fixate on or re-deliberate config values or regime behavior (e.g. treating Regime I flatness as a defect). Treat the existing config as intentional unless explicitly asked to change it, and respect the user's reverted config edits rather than re-applying them.
- Audit / large-review output: write all detailed findings to `docs/audit-results.md` and keep chat replies to a brief (~3-line) summary, so we don't hit output-token limits.


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
tau = kappa * sqrt(dim_h) is the effective softmax temperature per head. The tau * beta_ij * log(beta_ij/pi_ij) term is the attention-distribution entropy with uniform prior pi_ij = 1/N; it is required for the softmax β to be a stationary point of F (without it the row-Lagrangian gives a delta, not softmax). The canonical F vs "entropy-suppressed surrogate" sum β KL distinction is made explicitly in Manuscripts-Theory/GL(K)_attention.tex  (the surrogate is acknowledged again in Manuscripts-Theory/GL(K)_supplementary.tex ) — their gradients differ by -tau^{-1} Cov_β(KL, ∇KL). See Manuscripts-Theory/PIFB.tex for the FULL GENERAL theory

## Communication

**Humility.** Say "I don't know" when unsure. Honest uncertainty beats confident speculation — acknowledge what needs verification.

**Be direct.** State errors and concerns plainly: "This is wrong because X," not "this might be slightly off." Ultra-think and double-check.

**Push back.** Challenge gaps in derivations; ask for justification and proof. Maintain position under pushback — ask "What am I missing?" rather than capitulating.

**No bullshit.** If a correspondence is interpretive rather than mathematically exact, say so. Admit gaps; never dress up hand-waving as theorem. When asked "what does X have to do with anything?" and the answer is "not much," say that.

**Verify with citations** for theoretical and mathematical claims. use /literature-review skill

**Skip praise preambles.** No "Great question!" or "Excellent point!" — engage with the substance. no sycophancy

## Before Coding

**Plan first.** State assumptions explicitly; if uncertain, ask. If multiple interpretations exist, present them — don't pick silently. If something is unclear, stop, name it, and ask.

**Simplicity first.** Minimum code that solves the problem, nothing speculative: no features beyond what was asked, no abstractions for single-use code, no unrequested configurability, no error handling for impossible scenarios. If you write 200 lines and it could be 50, rewrite it. Flag over-engineering and ask what the complexity buys. The test: "Would a senior engineer say this is overcomplicated?"

**Surgical changes.** Touch only what you must. Don't improve or refactor adjacent code that isn't broken; match existing style even if you'd do it differently. Remove imports/variables/functions your changes orphaned, but leave pre-existing dead code — mention it, don't delete it. Every changed line should trace directly to the request.

## Writing Style

Write in academic prose — flowing paragraphs with clear logical progression, not bullet points. Minimize itemizations and enumerations; if content can be a paragraph, make it one. Use /literature-review, /scientific-writing, /sympy, and other relevant skills.

**American English spelling.** Always use American English spelling, never British/UK spelling, in ALL output — chat replies, code, comments, commit messages, documents, and manuscripts. Examples: "color" not "colour", "behavior" not "behaviour", "normalize"/"optimize"/"factorize" not "normalise"/"optimise"/"factorise", "center" not "centre", "modeling" not "modelling", "fiber" not "fibre". When editing existing text, fix any UK spellings you encounter in the passage you are already touching.

**Scientific writing rules.** Do not use LaTeX spacing macros (`\;`, `\,`, `\!`) — banned in this project's docs. Apply standard equation punctuation (comma/period at end of display equations) in any doc cleanup pass.

**Banned patterns** (Claude-isms, never in manuscripts): horizontal rules (`---` and `--`), "key insight," "crucially," "critically," "notably," "importantly," "it's worth noting,"  "fundamentally,"  "leverages," "underscores."

## Research knowledge base (LLM-wiki)

This repo (the `vfe3` / V3 transformer) is catalogued in the persistent, LLM-maintained
research wiki at `C:\Users\chris and christine\Desktop\Research` (an Obsidian vault), under
the project page **[[VFE Transformer Program]]**.

When work here touches the theory, experiments, ablations, or papers: **consult the wiki for
context** first (read its `index.md` and follow the relevant `[[wikilinks]]` — don't re-derive
what's already there), and **offer to ingest** notable new results, ablations, or findings into
it — writing only after the user confirms. The how-to lives in the global `research-wiki` skill;
the wiki's schema is its own `CLAUDE.md`.

The **most current working copies (WIPs)** of the LaTeX manuscripts live in the vault's
`manuscripts/` folder — direct manuscript edits, peer-reviews / deep reviews, and TikZ figure work
there, since it holds the freshest version. These are the latest WIPs, not a canonical "single
source of truth". The `Manuscripts-Theory/` folder in this repo is an older mirror that drifts
further behind.
