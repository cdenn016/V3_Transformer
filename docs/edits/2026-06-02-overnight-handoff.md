# Overnight buildout handoff — 2026-06-02

Branch: `vfe3-roadmap-overnight-2026-06-02`. Everything below is committed and pushed to
`origin/vfe3-roadmap-overnight-2026-06-02`.

### MERGE ORDER (read this first — it is a dependency CHAIN, not one branch)
Four levels of unmerged work stack; merge/PR bottom-up, NOT the overnight branch straight to main:
1. `main`
2. `vfe3-artifacts-priorbank-2026-05-31` — 6 commits (run-artifacts, use_prior_bank, head mixer, tied gauge, per-coord alpha)
3. `vfe3-buildout-roadmap-2026-06-01` — the roadmap doc + M1+M4 cov_kind seam + the families/M2 parameter-object refactor (built on level 2)
4. `vfe3-roadmap-overnight-2026-06-02` (this branch) — tonight's 5 items (built on level 3)
A PR of this branch against `main` will show ALL cumulative commits from levels 2-4. Either merge the
chain in order, or open the PR against the level-3 branch to see only tonight's diff.

Triage rule used (per the advisor): build overnight only what has a verification oracle that does not
depend on me (byte-identity refactor, default-OFF no-op equivalence, additive-isolation, or an
independent reference). Everything whose correctness would rest on a test written to match the
implementation was SPEC'd for your decision, not built.

## A. BUILT — oracle-verified + self-reviewed, GREEN, PUSHED

Honesty note on "reviewed": unlike the families/M2 work (which got independent two-stage spec+quality
review), tonight's 5 items were oracle-verified (the byte-identity / no-op / additive oracle in each)
plus a self-review of the diff by me (the controller). The `register_retraction` subagent's own review
was lost to a session limit, so I read that diff myself. No independent reviewer signed off on these 5;
the oracles + the 300-green suite are the correctness evidence.

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

These specs are workflow-generated DRAFTS I have not deeply reviewed for quality — read them
critically. Each does carry a concrete recommendation in its "DECISION NEEDED" section (I confirmed
this), so your morning can be yes/no confirmations, not open research. Headline recommendations:
- **f-divergence**: build **squared Hellinger** first — `H^2 = 1 - exp(-D_{1/2}/2)` over the
  already-pinned Renyi-1/2 kernel (sympy-verified `diff=0`), so it HAS an oracle and is the cleanest
  next build the moment you bless it; optionalize the functional signature's `alpha` arg via `**kwargs`.
- **SPD retraction variant**: **log-Euclidean first** (pure retraction 2a, SPD-exact, trivial); Frechet
  natural-gradient (2b) behind a sub-flag; single `spd_retract_mode` field; warn-not-error on log-Euclidean+diagonal.
- **hyper-prior/model-coupling**: **build incrementally, hyper-prior `lambda_h KL(s||r)` first** (it has a
  closed-form oracle; the gamma block does not); gamma gets its own attention-prior field reusing the seam.
- **learnable alpha**: needs your **no-NN-exception yes/no** first; if yes, posterior-mean point summary (b1)
  is the shippable form, full digamma-marginalized F (b2) opt-in after MC-verification.
- **Regime-II**: the `register_transport` seam is already built (A.5); the `regime_ii` builder needs your
  `delta_ij` decision (bilinear `mu_i^T W^a mu_j` is useful but a learned weight = no-NN exception; the pure
  variational form is vacuous on the causal default). Expose `cocycle_relaxation` alpha (default 1.0).

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
- **Training still works.** All 5 new config fields (`mstep_self_coupling_weight`, `spd_retract_mode`,
  `cross_couplings`, `transport_mode`) default to no-op / current behavior, and `test_train.py` is green,
  so `train_vfe3.py` launches and trains exactly as before. To try a new feature, flip one field (e.g.
  `mstep_self_coupling_weight=0.1`). `generate()` is available on the trained model for sampling.
- Your `train_vfe3.py` toggle experiment is STILL UNCOMMITTED (carried untouched in this branch's working
  tree); commit it whenever you like. `docs/edits/2026-05-30-diagnostics-tier.md` likewise left as-is.
  Untracked `.claude/`, `.codex/`, `AGENTS.md`, `Manuscripts-Theory/`
  were never staged.
- A spec-extraction script briefly created mis-named files (a heredoc collapsed `"\\20"` into an octal
  escape); cleaned up, git status clean. Tooling lesson: use pathlib, never manual `"\\"` path
  concatenation in a heredoc'd script.

## E. Directed session 2026-06-02 (continued, user awake) — items 1-8 + speedups

All committed + pushed to `origin/vfe3-roadmap-overnight-2026-06-02`; suite climbed 313 -> 401, 0 failures throughout. Each is opt-in/default-off or a byte-identity/equivalence-gated refactor; the user steered and sanctioned the two NN exceptions (#5, #6) with the required NN comments at the function AND the config toggle.

BUILT (oracle-verified + self-reviewed diff, green, pushed):
1. Squared-Hellinger f-divergence (`8d52194`) — `H^2=1-exp(-D_{1/2}/2)`; oracle = analytic Gaussian H^2 + SYMMETRY + self-zero. First non-Renyi functional; signature generalized via `**kwargs`.
2. log-Euclidean SPD retraction (`09ffb7b`) — into `register_retraction`; SPD-preservation + independent matrix-exp oracle. NOTE: the spec's "LE==affine on diagonal" claim was FALSE under the pre-whitened tangent; the implementer pinned the real relationship + truthful warning (code-truth over unvetted spec).
3. Extensible `BeliefState` (`56aa955`, roadmap M3) — NamedTuple + optional `s`/`r` fields, 3-field byte-identical; precondition for the hyper-prior channel.
4. Batched per-head `pairwise_energy` (`db7645b`) — stack equal blocks, one functional call; `torch.equal` bit-identical (diagonal + full).
5. Learnable alpha (`fe0bfce`,`9caa785`) — SANCTIONED NN exception; `alpha=exp(log_alpha)`, init 0 => alpha=1.0 (init==constant-1.0 oracle, `torch.equal`); NN comments at function/config/param; detach_e_step footgun warning.
6. Regime-II edge-relaxed transport (`a4ba488`) — SANCTIONED NN exception; bilinear `delta_ij=mu_i^T W^a mu_j`, init W=0 => flat (byte-identical); holonomy_deviation>0 when W!=0; per-edge O(N^2) matrix-exps; breaks strict gauge covariance (head-mixer-analogous, documented).
7. Hyper-prior channel, increment 1 (`d23211d`) — `lambda_h*KL(s||r)`, default lambda_h=0 (no-op); s/r tables; linear-in-lambda_h oracle. DEFERRED to increment 2: gamma model-coupling, s-channel E-step update, s->q coupling.
8. Straight-through E-step (`e8a847c`) — `e_step_gradient=unroll|straight_through|detach` (Alg-1:2050); straight_through forward==unroll forward (`torch.equal`), grad differs (no second-order); reconciled with `detach_e_step`.

SPEEDUPS:
- Dense-Omega fusion / P0 #2 (`d729e6c`) — `FactoredTransport` skips the dense `(B,N,N,K,K)` Omega on the flat block-diagonal path; mean diff 9.5e-7 (1 ULP), diagonal cov exactly 0.0, full cov byte-identical; regime_ii/full/single-block/cross-coupled stay dense (guarded).
- Mixed-precision opt-in (`dc4c255`) — `amp_dtype in {None,bf16,fp16}`, default None byte-identical (autocast tripwire test); fp32 islands for matrix_exp/SPD + decode/CE; bf16 recommended for the 5090, fp16-GradScaler is a follow-up.
- Chunked-vocab decode (`5c4618a`) — `decode_mode=diagonal_chunked`, fused chunked decode+CE avoids the `(B,N,V)` logit tensor in training (peak `O(V*K+B*N*K)`, verified via saved_tensors_hooks); chunked CE vs full 4.8e-7 (gate 1e-3), grad ~1e-9; inference still materializes full logits.

REMAINING (flagged, not built):
- Causal-packed transport (#12) — N-factor win, but now entangled with `FactoredTransport`; do as a focused follow-up.
- Observation-likelihood seam (#9) — a decide-and-document item (the current CE-external design is the defensible vacuum-plus-source split); needs your call before building.
- Hyper-prior increment 2 (gamma model-coupling + s-channel E-step + s->q coupling); torch.compile/CUDA-graph (after a 5090 re-profile); the spec-only items still awaiting decisions (Bures-Wasserstein retraction, additional groups U/SU/Sp, f-divergence beyond Hellinger).
