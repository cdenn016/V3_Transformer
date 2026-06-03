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

## D. POSITIONAL ENCODING CONFIG (Task 1.2 — 2026-06-02)

Added BCH positional-encoding config fields (`pos_phi`, `pos_phi_compose`, `bch_pe_order`, `pos_phi_scale`, `pos_phi_project_slk`) to `VFE3Config`, along with `_VALID_POS_PHI_COMPOSE` tuple and `__post_init__` validation against the `_POS_PHI` registry. Default `"none"` preserves the pure path byte-identically. Three tests added to `tests/test_config.py` (32 total pass).

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

## F. Gamma model-coupling block — hyper-prior increment 2 (WORKING TREE, not yet committed)

The retry of the gamma item that the prior session's subagent dropped on an API socket error
(working tree was unchanged). Built by hand (the seam is tightly interdependent — config field ↔
prior_bank table gate ↔ model.py loss assembly ↔ tests — exactly the shape that lost edits in a
fan-out before; see `[[workflow-shared-tree-lost-edits]]`), TDD red→green, advisor-steered on the
one load-bearing design decision (the detach). Suite: **437 tests, 0 failures, 0 errors** (was 426
before this increment; +11 new `tests/test_gamma_coupling.py`). The lone `xpassed`
(`test_training_decreases_loss_on_structured_stream`) is a pre-existing non-strict xfail, unrelated.

**What it is.** `L += gamma_coupling * mean_i F_red^s_i`, the reduced (envelope) form of the
model-coupling block `sum_ij [ gamma_ij KL(s_i||Omega_tilde_ij s_j) + tau_g gamma_ij
log(gamma_ij/pi^s_ij) ]` (manuscript `Participatory_it_from_bit.tex` eq:pointwise_free_energy,
1241-1249). The s-channel is the SAME softmax-over-KL object as the belief beta block, so it REUSES
`pairwise_energy` + `reduced_free_energy` with `(q,p,beta,pi,tau) -> (s,Omega s,gamma,pi^s,tau_g)` —
no new energy/softmax code. New config: `gamma_coupling=0.0` (scale, OFF default), `kappa_gamma=1.0`
(→ `tau_gamma = kappa_gamma*sqrt(d_head)` property mirroring `tau`), `gamma_attention_prior="causal"`
(own pi^s seam). The `s` tables are now created on `lambda_h>0 OR gamma_coupling>0` (s drawn before r,
so the existing `lambda_h`-only RNG order — and hence byte-identity — is preserved); `r` stays
`lambda_h`-only.

**The load-bearing decision — TIED + DETACHED transport (advisor-confirmed, spec-forced).**
`Omega_tilde` is the flat phi-cocycle `exp(phi_i)exp(-phi_j)` from the CONVERGED belief frame
`out.phi`, **detached**. So the gamma gradient flows ONLY to the `s` tables: the forward (logits/ce)
is **byte-identical** to the gamma=0 path and the model channel stays **predictively INERT** — `s`
does not feed `q`. This is forced by the user's own framing ("s stays predictively inert until
s→q"): a live Omega would backprop into `phi_embed`, which feeds the forward, changing predictions.
The detach deliberately SEVERS the `phi <- gamma` coupling that full tied transport carries in the
canonical E-step F; restoring it (or keeping it severed) is part of the deferred s→q design, NOT
this term. (Theory note for whoever builds s→q: "tied transport" here means *evaluated at the
current phi, frozen* — the coupling is not yet wired.)

**Oracles (all green).** (1) default-off → no s/r tables, `loss==ce`; (2) gamma>0 alone creates `s`
but not `r`; (3) GOLD: `loss_w - loss_0 == w * gamma_term` against an INDEPENDENT recomputation of
the term from the s tables + `Omega(encode().phi.detach())` at `atol=1e-6` (only recomputation
catches a wrong-tensor bug — linearity/envelope are necessary-not-sufficient; the `e_phi_lr=0`
shortcut makes `out.phi == encode().phi` so the oracle skips re-running `vfe_stack`, guarded by an
assert); (4) predictive inertness — mutating `s` leaves logits/ce byte-identical; (5) detach contract
on the REAL forward — `phi_embed.grad` and `mu_embed.grad` are EQUAL across gamma=0/gamma=w, while
`s` trains only at gamma>0; (6) envelope identity for the gamma channel; (7) self-zero under identity
transport; (8) config validation + `tau_gamma`.

**Documented simplifications (parity, not new approximations).** `gamma_coupling=1` is a
per-token-per-head *mean* weight (the reduction is `.mean()` over (B,H,N)), not the canonical
sum-over-ij — the scale is a free coupling. The diagonal `transport_covariance` keeps only
`diag(Omega Sigma Omega^T)`, the same approximation the belief diagonal family already uses. The
tied transport is the flat cocycle (exact under the default flat regime; a documented tie under
regime_ii). Dense Omega is materialized once per forward at the loss level (like `diagnostics()`),
not the hot-path `FactoredTransport`.

**Files.** `vfe3/config.py` (3 fields + validation + `tau_gamma` property), `vfe3/model/prior_bank.py`
(s-table gate split, `gamma_coupling` param), `vfe3/model/model.py` (PriorBank wiring + the gamma
block in `forward`), `tests/test_gamma_coupling.py` (new, 12 tests), `docs/verified.md` (new).

**Adversarial verification (5-lens workflow, each verifier told to falsify).** math-envelope CONFIRMED
(sympy envelope identity → 0; hand-matmul falsified Omega^T/Omega_ji; from-scratch per-head loop matched
to 5.96e-8). detach-inertness CONFIRMED (`autograd.grad(gamma_term,[phi_embed,mu_embed,sigma_log_embed])`
→ None,None,None; no-detach mutation probe proves the test discriminates). byte-identity CONFIRMED
(exact-equality forward, pure-path RNG identical, no `'s implies r'` dep). code-quality CONFIRMED (no-NN
sanctioned; mutation testing confirms the oracle catches missing-detach/wrong-reduction/per-head/forgot-
transport). manuscript-fidelity CONCERN — NOT a defect in what was built: the term is correct and the
simplifications honestly documented; the concern IS the deferred s→q design (static-s vs inferred-field;
detach severs phi←gamma), which the manuscript itself zeroes in its sims (line 1296). Recorded in
`docs/verified.md`. No FLAWs across all five lenses.

**Post-verification fixes (low-severity, surfaced by the verifiers).** (a) Updated stale comments my
change rendered false — the `encode_s` docstring and the `lambda_h` comments that said "gamma block
DEFERRED to increment 2" (it is now built); `config.py`, `model.py`, `prior_bank.py`, `test_hyperprior.py`.
(b) Corrected my own test docstring that overstated the linearity oracle's independence (it shares
primitives with the impl, so it pins wiring/isolation/reduction/linearity/per-head, not the math). (c)
Added `test_gamma_energy_equals_analytic_kl_at_nonzero_phi` — a formula-independent analytic diagonal-KL
check at NONZERO phi (Omega != I), closing the verifier-flagged coverage gap (every other fixture uses
e_phi_lr=0 so Omega≈I) and giving the one genuinely-independent `E_s == KL(s_i||Omega s_j)` check.
Suite after fixes: 438 tests, 0 failures, 0 errors.

**Still deferred (the valuable part — wants your design input):** s→q coupling (the model channel
actually driving predictions, via `p_i(k_i|m_i)` or making `s` an E-step-iterated state). New
behavior with no clean oracle — building it blind is the plausible-but-wrong trap. The gamma assembly
here is reusable infrastructure for it: only the *source* of `s_i` changes (static table → iterated
state), not the energy/softmax machinery. NOT committed yet (awaiting your go).

## G. s→q coupling — `prior_source="model_channel"` (Realization A; WORKING TREE, not committed)

The user supplied the design input (3 decisions: drive predictions *through the belief prior*; *static*
s; *replace* `p_i = s_i`) and then chose **Realization A** when examining the decode revealed a
realization fork. Built by hand, TDD, advisor-grounded; **5-lens adversarial verification**.
Suite: **446 / 0 / 0** (+8 `tests/test_prior_source.py`).

**What it is.** A default-off config toggle `prior_source ∈ {"token","model_channel"}`. `"token"`
(default) = the belief tables `mu_embed/sigma_log_embed`, byte-identical to before. `"model_channel"`
REPLACES the belief prior with the model channel: `p_i = s_i`, routed through one accessor pair
`PriorBank._prior_mu_table()/_prior_sigma_log_table()` at **every** place the prior is consumed —
encode (`q_i(0)=p_i`), the E-step self-coupling target `α·KL(q_i‖p_i)`, and **all four decode kernels**
(diagonal/full/chunked/reference). The model-channel `s` tables (coupled by the γ/λ_h I shipped this
session) thus become the belief prior and drive predictions; φ stays the belief table (tied,
`B_state=B_model`). `mu_embed` is dead on this path.

**Trainability fix (caught by grep, not by the unit tests).** `build_optimizer`'s exact-coverage guard
would have *raised* under `model_channel` (the `s` tables weren't grouped) — the model couldn't train
its prior. Now the `s` tables are grouped (mean@`m_mu_lr`, log-scale@`m_sigma_lr`), with an end-to-end
"optimizer steps the prior" test (loss 2.996→2.418 over 20 steps in verification). The hyper-prior
centroid `r` (lambda_h>0) is **FROZEN** (`requires_grad=False`; your decision) — a fixed centroid per the
manuscript's "higher, slower meta-level" (`supp:1081`); free-training it would collapse `KL(s‖r)→0`. The
coverage guard now exempts non-trainable params, so `build_optimizer` works for lambda_h>0 (s grouped,
frozen r skipped) — the hyper-prior channel now trains end-to-end. Still ungrouped & RAISES
(**pre-existing**, genuinely trainable): `log_alpha` (`alpha_mode='learnable'`) and `connection_W`
(`regime_ii`).

**Oracles.** default `token` byte-identical (accessor returns the *literal same object*); **copy-equivalence**
(s := belief tables → byte-identical, now also through the M-step self-coupling rebuild); directional
(s live / `mu_embed` dead); grad (trains s, not `mu_embed`); config validation.

**Verification verdicts.** prior-consistency CONFIRMED (no issues — whole-tree grep + empirical directional
checks on every path), byte-identity CONFIRMED, trainability CONFIRMED. manuscript-fidelity **CONCERN with
a HIGH-severity finding I fixed**: my citation was WRONG — I attributed `p_i(k_i|m_i)` to
`Participatory_it_from_bit.tex`, but that conditional is in `GL(K)_supplementary.tex:1083-1085`, and
Participatory:1440 states the *opposite* ("s_i does not act through p_i at the same scale"; its `p` is a
*cross-scale* shadow). Verified against the actual `.tex` myself. **The two manuscripts carry different
s→p mechanisms; this increment realizes the supplementary's same-scale hierarchical-Bayes reading, NOT the
main manuscript's cross-scale one.** Corrected the citations and disclosed the tension in `config.py` +
the test docstring + `docs/verified.md`. The realization is mathematically faithful to the supplementary's
Eq. 1085 (not a code/math flaw); whether the same-scale reading matches your intent (vs the cross-scale
`p`, which needs a meta-agent/scale-(s+1) object that does not exist yet) is **your call** — see §H.

**Files.** `vfe3/config.py` (field + `_VALID_PRIOR_SOURCES` + validation), `vfe3/model/prior_bank.py`
(accessor pair + s-table gate + 5 rerouted reads), `vfe3/model/model.py` (wiring), `vfe3/train.py`
(s optimizer group + NOTE), `tests/test_prior_source.py` (new, 8 tests), `docs/verified.md`.

## H. OPEN QUESTION FOR YOU — I gave you a WRONG premise for this choice; do you still want it?

This is the important one. **Before your Q3 decision I told you "the manuscript route is unambiguous:
s drives predictions through the belief prior (h→s→p→q — `p_i(k_i|m_i)`, Participatory:1083)." BOTH
halves were wrong** — the citation (that conditional is in `GL(K)_supplementary.tex:1083`, not
Participatory) and "unambiguous" (the **main** `Participatory_it_from_bit.tex:1440` states the OPPOSITE
mechanism: `p_i` is the **cross-scale shadow** of the meta-agent's belief `q^(s+1)` transported down, and
"`s_i` does not act through `p_i` at the same scale"). You partly chose **replace `p_i = s_i`** because I
presented it as the settled manuscript route. It is not.

Precisely what is and isn't contaminated: **Q1 ("drive predictions through the prior") still holds** in
*both* readings — in the cross-scale reading `p_i` is still the prior `q` aligns to, just sourced from
`q^(s+1)` instead of `s_i`. It is **Q3 (the same-scale identity `p_i = s_i`)** that commits to the
**supplementary's** reading, which the main manuscript's text contradicts. The code is a correct,
tractable realization of the supplementary's Eq. 1085 — but the *mechanism choice* was made on a wrong
premise, so it's genuinely yours to remake with the correct picture:
- (a) **Keep** the same-scale `p_i = s_i` (the supplementary is explicit; it's what trains today, default
  `token` unaffected);
- (b) **Cross-scale instead** — `p_i = Ω·q^(s+1)` (the main manuscript's mechanism; a much larger build,
  needs a meta-agent / scale-(s+1) hierarchy that does not exist yet);
- (c) **Both**, behind the toggle.

Smaller related decision — **RESOLVED**: you chose to **freeze** the hyper-prior centroid `r` (fixed
centroid, `requires_grad=False`). Done — the coverage guard now exempts non-trainable params, so
`build_optimizer` works for `lambda_h>0` (s trains, r fixed). And note `model_channel` with
`gamma=lambda_h=0` is a *pure rename* of `mu_embed` (zero added capacity); the model channel changes
predictions only once `gamma>0`/`lambda_h>0` shape `s` beyond CE.

## Per-eval attention plots + sample text (2026-06-02, separate from the model-channel work above)

Two reporting features were added to the periodic-eval path. Both are opt-in and leave the silent
training path bitwise-identical when not configured. Full suite green at 457 tests (was 447), 0
failures, 0 errors, via `--junitxml`.

**Per-layer, per-head attention heatmaps every eval interval.** A new `VFEModel.attention_maps(token_ids)`
(no_grad, off the training graph) returns `(L, H, N, N)` for sequence 0: it replays the `vfe_stack`
block loop one block at a time, mirroring the `mu_p`/`sigma_p` handoff in `stack.py` line for line, and
at the converged output belief of each block recomputes the attention pattern the same way
`diagnostics()` does at the final belief (transport `Omega_ij(phi)` then `pairwise_energy` then
`attention_weights`). The per-irrep-block energy supplies the head axis `H = len(group.irrep_dims)` (1 for
glk/so_k, 2 for the default block_glk at K=20). The canonical definition of "per-layer attention" is
pinned by construction: the last layer's map equals the attention `diagnostics()` reads, byte-identical at
`n_layers=1` (where the stack is a single block and the handoff loop is empty), which is asserted as a
test (`attention_entropy(maps[-1]) == diagnostics()["attn_entropy"]`); for `n_layers>1` the replay uses
each block's own output, the exact trajectory the model ran, while diagnostics folds the final belief, so
the two diverge by design. A new figure `plot_attention_grid` renders an `L x H` grid of `beta` heatmaps
(`squeeze=False`, shared colour scale; rows query `i`, cols key `j`) and a best-effort
`RunArtifacts.save_attention_maps` writes `attention/step_<N>.png` each periodic eval. The save is wired
inside the eval block's `if artifacts is not None` guard (so it needs a run directory, matching all other
persistence) and runs on the same live-batch sequence 0 that the logged diagnostics consume. A viz error
is logged and swallowed, never fatal, and every figure is closed.

**Sample text directly below the BPC value every eval interval.** `train()` gained three keyword
arguments (`sample_decode`, `sample_new_tokens=40`, `sample_prompt_len=16`). The decoder is the explicit
`sample_decode` if given, otherwise an AUTO-DEFAULT (`_default_sample_decoder(cfg)`) chosen from
`cfg.vocab_size`: gpt2 for a vocab in roughly `[40k, 60k]`, cl100k for `[90k, 110k]`, and `None`
otherwise. This vocab gate is what preserves the pure path without a new toggle — a real click-to-run on
wikitext-* (vocab 50257) prints samples with zero wiring and no entry-file edit, while a tiny
synthetic/test vocab (e.g. 6) gets no decoder and the eval stays silent. When a decoder exists, the eval
block greedily continues sequence 0 of the live batch by `sample_new_tokens` (`model.generate`, already
no_grad) and logs `Sample: <prompt> -> <continuation>` immediately under the BPC line, best-effort (a
generation/decode error is logged, never fatal). This choice (auto-default in `train()`) was the user's,
made because the click-to-run entry `train_vfe3.py` is user-owned and was to stay untouched. The pure
silent path stays reachable under a toggle: `train(..., generate_samples=False)` forces no generation and
no `Sample:` line even at a real vocab, satisfying the project's pure-path constraint without an entry-file
edit (default `True` preserves the requested auto-on). A companion `datasets.get_tiktoken_decoder(dataset)`
remains available for an explicit, dataset-named decoder (gpt2 / cl100k by the cache tag, `None` for the
synthetic anchor / absent tiktoken) should a caller want to pass one explicitly. Caveat on verification:
the decode round-trip is verified (gpt2 ids `[15496, 995]` -> `"Hello world"`); end-to-end smoke runs used
the synthetic period-3 stream, so real-corpus continuation quality has not yet been observed in this work.

**Scheduler note (asked in passing).** Yes — `train()` already builds a warmup-then-cosine schedule:
`lr_lambda(step, cfg)` does a linear warmup to 1.0 over `warmup_steps`, then a half-cosine decay to 0.0 at
`max_steps` (argument clamped to `[0, pi]` so steps past `max_steps` stay at 0), wired through
`torch.optim.lr_scheduler.LambdaLR` and advanced once per optimizer step in `train_step`. No change was
made to it.

Files touched: `vfe3/model/model.py` (attention_maps + the `vfe_block` import), `vfe3/viz/figures.py`
(plot_attention_grid + registry), `vfe3/run_artifacts.py` (save_attention_maps), `vfe3/train.py`
(sample-text args + block, attention-save wire-in), `vfe3/data/datasets.py` (get_tiktoken_decoder). Tests
added across `test_model.py`, `test_viz.py`, `test_run_artifacts.py`, `test_train.py`, `test_data.py`.

## Task 1.1 — BCH-PE pos_phi registry (2026-06-02, branch vfe3-positional-encodings-2026-06-02)

Created the standalone `pos_phi` module and its unit tests as the first task of the BCH positional-encoding feature. The module is default-off and adds no wiring to the model yet; later tasks compose it into the gauge frame.

**`vfe3/model/positional_phi.py`** — a registry of per-position Lie-algebra coordinate builders:
- `register_pos_phi` / `get_pos_phi` — decorator registry (mirrors the existing retraction/transport seam idiom).
- `"none"` builder — returns `None`; the pure default-off path.
- `"frozen"` builder — parameter-free Lie-algebra ALiBi: `pos_phi_i = (i * scale)` on one generator axis.
- `"learned"` builder — slices the first `n` rows of a model-owned `(max_seq_len, n_gen)` parameter table.
- `positional_phi_coords` — dispatch shim.
- `apply_positional_phi` — composes the (N, n_gen) coords into `phi` via `compose_phi` (BCH by default); `"none"` returns `phi` byte-identical; optional `project_slk` removes the per-block trace to preserve `det(Omega_h)=1`.

**`tests/test_positional_phi.py`** — 4 tests: `none` returns `None`; `frozen` shape + values; `learned` slices the table; `apply` with `mode="none"` is identity.

TDD: test written and confirmed failing (ModuleNotFoundError) before the module was created. Final run: **4 passed** (read from the `N passed` line). Commit: `4df355b`.

## Task 1.1 code-review fixes — `**kwargs` forwarding + `get_pos_phi` test (2026-06-02, branch vfe3-positional-encodings-2026-06-02)

Two small non-behavior-changing fixes applied to `vfe3/model/positional_phi.py` and `tests/test_positional_phi.py`.

**Fix 1 (extensibility).** `positional_phi_coords` and `apply_positional_phi` previously enumerated `scale`/`frozen_axis`/`pos_phi_free` explicitly with no `**kwargs`, meaning a future builder with a novel param would require editing both dispatchers. Added `**kwargs` at the end of both signatures and forwarded it into the builder call (the individual builders already accepted `**kwargs`). This mirrors the `attention_prior.py` registry idiom and satisfies the CLAUDE.md hard constraint: add a variant by writing-and-registering it, never by editing call sites.

**Fix 2 (coverage).** `tests/test_positional_phi.py` imported `get_pos_phi` but never exercised it. Added `import pytest` at the top and a new test `test_get_pos_phi_unknown_raises_keyerror` that confirms `KeyError` is raised for an unregistered mode name.

Final run: **5 passed** (read from `5 passed in 0.02s`). Commit: `accb330`.

## Task 1.3 — wire BCH-PE into forward / diagnostics / attention_maps (2026-06-02, branch vfe3-positional-encodings-2026-06-02)

Wired the `apply_positional_phi` call into `VFEModel` so the configured positional gauge element is composed into `beliefs.phi` before the E-step and before any diagnostic/map replay.

Changes to `vfe3/model/model.py`: (1) imported `apply_positional_phi` from `vfe3.model.positional_phi`; (2) added `pos_phi_free = nn.Parameter(randn(max_seq_len, n_gen) * pos_phi_scale)` in `__init__` ONLY for `pos_phi='learned'`, mirroring the `log_alpha`/`connection_W` detach-warning idiom (pure "none"/"frozen" paths add no parameter); (3) added private `_apply_pos_phi(phi)` helper that is a no-op for `'none'`; (4) called `beliefs._replace(phi=self._apply_pos_phi(beliefs.phi))` immediately after `prior_bank.encode` in `forward`; (5) wrapped `enc.phi[0]` with `self._apply_pos_phi(...)` in both `diagnostics` and `attention_maps`. Three tests added to `tests/test_positional_phi.py`: determinism + no-param-on-pure-path; parameter shape + logit divergence; gradient flow to `pos_phi_free`. Final runs: **8 passed** (`tests/test_positional_phi.py`), **24 passed** (`tests/test_model.py`), both read from the `N passed` lines.

## Task 1.5 — BCH-PE property tests: DONE_WITH_CONCERNS (2026-06-02, branch vfe3-positional-encodings-2026-06-02)

Three property tests appended to `tests/test_positional_phi.py`: BCH vs euclidean divergence when the bracket is nonzero; `project_slk` makes composed blocks trace-free; and the self-coupling isolation test.

**Two tests pass cleanly.** `test_bch_differs_from_euclidean_when_bracket_nonzero`: verified that `gl(4)` generator 0 is `E_{00}` (not the identity/trace direction), so a random `phi` (all 16 generators active) generically has `[embed(phi), coords] != 0`; BCH and euclidean differ by 3.3e-4 at `atol=1e-4`. `test_project_slk_makes_blocks_traceless`: with `phi=0` the compose reduces to the projected coords, and both 2x2 diagonal block traces are zero to `atol=1e-5`.

**Real finding: `test_pos_phi_does_not_change_self_coupling_diagnostic` fails at 3.3e-4 >> 1e-6.** The task said "if this test FAILS, it means the diagnostic folds phi into the prior — report as DONE_WITH_CONCERNS." The actual mechanism is more precise: the *prior* `(mu_p, sigma_p)` is genuinely phi-independent (confirmed: `encode(x).mu/sigma/phi` are byte-identical between base and learned models at the same prior tables). However, `self_coupling = KL(q_converged || p)` and `q_converged = vfe_stack(belief)` depends on `belief.phi` (which carries the positional element). With one E-step at `e_mu_lr=0.1`, the phi-augmented transport `Omega_ij(phi_pos)` pushes `out.mu` away from the base trajectory, so the converged `q_i` moves even though `p_i` stays fixed. The self_coupling legitimately changes because the converged belief is phi-dependent — the prior is untouched, but the belief is not. The assertion `< 1e-6` is over-strong: the correct isolation is "the prior is phi-independent," not "the self_coupling KL is phi-independent." Tolerance was NOT loosened (per task instructions); the test is committed as-written with this documented finding.

## Task 1.4 — group `pos_phi_free` in `build_optimizer` (2026-06-02, branch vfe3-positional-encodings-2026-06-02)

`build_optimizer`'s exact-coverage guard would raise for any model instantiated with `pos_phi="learned"` because `pos_phi_free` is a trainable `nn.Parameter` not yet assigned to any optimizer group. Added a conditional group block in `vfe3/train.py` (after the head-mixer block, mirroring its idiom): `if getattr(model, "pos_phi_free", None) is not None: groups.append({"params": [model.pos_phi_free], "lr": cfg.m_phi_lr})`. Learning rate `m_phi_lr` is the gauge-frame scale, matching how `phi_embed` is grouped. One test appended to `tests/test_train.py`: `test_build_optimizer_groups_pos_phi_free` (TDD: confirmed failing with "build_optimizer left 1 model parameter(s) ungrouped" before the fix, passing after). Final runs: **1 passed** (single test), **13 passed, 1 xpassed** (full `test_train.py`), both read from the `N passed` lines.
