# Overnight buildout handoff — 2026-06-02

Branch: `vfe3-roadmap-overnight-2026-06-02` (cut from the families/M2 HEAD `5f2ceb7`, which is itself
unmerged on `vfe3-buildout-roadmap-2026-06-01`; NOT merged to main). Everything below is committed and
pushed to `origin/vfe3-roadmap-overnight-2026-06-02`.

Triage rule used (per the advisor): build overnight only what has a verification oracle that does not
depend on me (byte-identity refactor, default-OFF no-op equivalence, additive-isolation, or an
independent reference). Everything whose correctness would rest on a test written to match the
implementation was SPEC'd for your decision, not built.

## A. BUILT, GREEN, REVIEWED, PUSHED

Full suite after each: `failures=0 errors=0`; the count climbed 274 -> 280 -> 284 -> 293 -> 300 as
tests were added (includes `test_viz.py`; the FD-gradient-oracle, E-step-descent, gauge-equivariance,
and frozen-oracle tests are the byte-identity gate). Final state: 300 passed, 0 failures, 0 errors.

1. **M-step self-coupling regularizer** `alpha_hat * sum_i KL(q_i*||p_i)` (commit `73e6b42`). Manuscript
   Algorithm 1 loss term (`GL(K)_attention.tex:2083`) that was on no path. New config field
   `mstep_self_coupling_weight: float = 0.0` (OFF by default = byte-identical pure path). When > 0,
   `model.forward` adds `weight * mean self-divergence(converged belief vs per-block prior)`,
   grad-connected, reusing `diagnostics()`'s prior-fold (exact at `n_layers=1`). Oracle: no-op at
   weight 0 + `loss(w) - loss(0) == w * self_div` (both pinned by new tests).

2. **`register_retraction` seam** over the SPD covariance retraction (commit `5dfa819`). Byte-identity
   refactor: `register_retraction`/`get_retraction` (mirroring `_PRECOND`/`_DECODERS`), the current
   affine-invariant SPD retraction registered as `"spd_affine"` owning the diagonal-vs-full rank
   dispatch the E-step used to do inline; new config `spd_retract_mode = "spd_affine"` validated against
   the registry. No new variants (log-Euclidean / Bures-Wasserstein are spec'd — section B). Oracle:
   `spd_affine` bit-identical to the bare `retract_spd_{diagonal,full}` + the full suite green.

3. **cross_couplings reachable from config** (commit `e581f08`). The off-block GL(K) head-coupling
   machinery (`generate_glk_cross_head` + Lie-bracket `close_under_brackets`) existed in geometry but
   was unreachable; a verify-first check confirmed it forwards+backwards green today. New config field
   `cross_couplings: Optional[List[Tuple[int,int]]] = None`, validated (index range, a!=b, only the
   `block_glk` builder accepts it — checked via `inspect.signature`, so `glk`/`so_k`/`tied_block_glk`
   are rejected when set). `build_group` forwards it only when set + accepted, so the default-None path
   is byte-identical (`torch.equal` on generators + frozen-oracle unchanged). NOTE: only the un-closed
   cross-head basis is config-reachable; the bracket-CLOSED subalgebra (`close_basis=True`) stays
   builder-only by design (scope) — say if you want it exposed too.

4. **Autoregressive `generate()`** (commit `ef5ebbe`). The model could only do teacher-forced CE
   training; it had no generation path. Added `VFEModel.generate(token_ids, max_new_tokens, *,
   temperature, top_k, top_p, greedy)`, `@torch.no_grad`, which REUSES the existing `forward` (no
   reimplementation of encode/E-step/decode), reads the last-position logits, samples (greedy / top-k /
   top-p / temperature), appends, repeats (truncating context to `max_seq_len`). Oracle: greedy is
   deterministic and equals `argmax(forward(prompt))` on the first token; training-isolated (a param is
   byte-identical before/after a generate call; additive, changes no existing test). Per-step
   full-forward is the correct-but-slow first version; incremental belief reuse is a future optimization.

5. **`register_transport` seam** over the gauge transport (commit `310bdf2`). Byte-identity refactor
   mirroring A.2: `register_transport`/`get_transport`, the flat Regime-I phi-cocycle registered as
   `"flat"` (a kwargs-tolerant adapter over `compute_transport_operators`), new config
   `transport_mode = "flat"` validated against the registry (orthogonal to `gauge_parameterization`).
   The E-step's primary belief-transport build routes through `get_transport(cfg.transport_mode)`; the
   flat default is `torch.equal`-identical to the direct call (+ frozen-oracle unchanged). This is the
   home the spec'd Regime-II builder (section B) drops into once you decide the `delta_ij`
   parameterization. No Regime-II built.

## B. SPEC-ONLY — AWAITING YOUR DECISION (committed `88ddaac`, in docs/superpowers/specs/2026-06-01-*)

These have design latitude and no overnight oracle; each spec has a "DECISION NEEDED" section. The
decisions, in one place:

- **learnable / Bayesian alpha** (`...-learnable-alpha-design.md`): DECISION — do you sanction a
  learnable-alpha `nn.Parameter` as a blessed no-NN exception (like the PriorBank tables / linear
  decode / head mixer)? It is a THIRD learned-parameter path; I would not introduce it without your
  yes/no. The fully-Bayesian (Gamma-posterior) variant is also designed.
- **Regime-II edge-relaxed connection + `register_transport` seam** (`...-regime-ii-connection-design.md`):
  the `register_transport` SEAM itself is a safe byte-identity refactor I can build (flat cocycle as
  default) — say the word. The Regime-II physics has a load-bearing DECISION: how is `delta_ij`
  parameterized? The only pure no-NN option (a variational edge field) is functionally vacuous on the
  causal default; the USEFUL options (bilinear `mu_i^T W^a mu_j`, etc.) need a learned weight = another
  no-NN exception. Your call on the parameterization + the purity tradeoff.
- **f-divergence beyond Renyi** (`...-f-divergence-functional-design.md`): DECISION — which f-divergence
  first (squared Hellinger and Jensen-Shannon are the closed-form-friendly candidates for Gaussians),
  and the functional-signature generalization (the registry signature currently bakes in an `alpha`
  arg; optionalize it vs pass functional-specific params via `**kwargs`).
- **SPD retraction variants** (`...-spd-retraction-variants-design.md`): the `register_retraction` seam
  (section A.2) is built; DECISION — which variant to add first (log-Euclidean is the cheaper, Bures-
  Wasserstein the richer) and whether the natural-gradient preconditioner gets a per-retraction hook.
- **hyper-prior + model-coupling two-tier channel (XL)** (`...-hyperprior-model-coupling-design.md`):
  the big one (gates the Regime-II meta-bundle hierarchy). DECISION — the `s_i`/`r_i` representation
  (new PriorBank tables vs derived), and whether to build incrementally (hyper-prior `lambda_h KL(s||r)`
  term alone first, model-coupling `gamma` second). Designed to reuse the divergence/family/group seams
  via a second `BeliefParams` channel (couples to the M3 BeliefState-extensibility finding).

## C. COMPLETED THIS SESSION (both now built — see A.4, A.5)
- Autoregressive generation path — BUILT (A.4, commit `ef5ebbe`).
- `register_transport` seam — BUILT (A.5, commit `310bdf2`).

Nothing further was built blind. The remaining roadmap items all fall in section B (they need a
decision from you) or are the XL hyper-prior channel; building them overnight without an oracle would
risk shipping a plausible-but-wrong implementation, so they were spec'd instead.

## D. Notes
- The user's `train_vfe3.py` toggle experiment and `docs/edits/2026-05-30-diagnostics-tier.md` were left
  uncommitted/untouched throughout. Untracked `.claude/`, `.codex/`, `AGENTS.md`, `Manuscripts-Theory/`
  were never staged.
- A spec-extraction script briefly created mis-named files (a heredoc collapsed `"\\20"` into an octal
  escape); cleaned up, git status clean. Tooling lesson: use pathlib, never manual `"\\"` path
  concatenation in a heredoc'd script.
