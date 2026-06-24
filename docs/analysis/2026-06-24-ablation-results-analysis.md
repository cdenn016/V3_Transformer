# Ablation results analysis — `vfe3_ablation_results/`

Date: 2026-06-24
Source: 9 completed sweeps in `vfe3_ablation_results/` against the pre-registered
`docs/hypotheses/2026-06-21-hypotheses.md`.
Method: per-sweep analyst + adversarial verifier (config-diff verification, seed-noise and
confound stress-test). Workflow `wf_f5bc6a3a-961`, 18 agents.

## Universal caveats (apply to every result below)

1. **Single seed.** Every run is `seed=6`. The pre-registered decision rule (non-overlapping
   ±1 SD bands at ≥3 seeds) cannot be satisfied by any of these. EXP-1 / I1 (the multi-seed
   variance floor — the discipline gate the doc calls ★★★★★) was **not run** in this batch.
2. **Noise anchor is imported, not measured here.** The verifier found older multi-seed runs in
   `vfe3_runs/` with across-seed `test_ppl` CV ≈ 0.47–0.69% (1 SD ≈ 0.4–0.9 PPL). That is *tighter*
   than the doc's 1–1.5% estimate, but it comes from a better-converged regime (45k steps, linear
   decode, gaussian_diagonal); under this batch's 15k-step underfit regime the true CV is plausibly
   2–3× larger. Throughout I treat **≈2 SD ≈ 3% ≈ 4.3 PPL** at the 144.5 baseline as the
   "credible-signal" bar, scaled proportionally for higher-PPL arms.
3. **One validation eval per run.** `eval_interval=15000` ⇒ `best_val_ppl == final_val_ppl`; there
   are no validation curves, and `train_ppl` is still falling at step 15000 in every arm. Good for
   *relative* ranking; not an absolute-PPL claim.
4. **Small / underfit scale.** `embed_dim=20`, `n_layers=1`, 15k steps. Effects established here are
   directional at this scale.
5. The recurring **144.47 PPL** operating point (block_glk, head_mixer on, AdamW-φ, n_e_steps=1,
   κ=1, canonical entropy, unroll) reappears identically as the "good" arm of five different sweeps
   — a clean internal consistency check (fixed data order ⇒ deterministic).

## Verdict summary

| Hyp | Sweep | Headline | Δ vs control | Robustness | Verdict |
|---|---|---|---|---|---|
| A1 | gauge_transport | ON 154 vs OFF 267–270 vs FROZEN 277–281 | −43–44% (≈25–30×SD) | **strong** | **CONFIRMED** (dir.) |
| D1 | gauge_mstep_optim | AdamW 144 vs pullback 252 vs killing 272 | +75% / +88% (≈50–59×SD) | **strong** | **REFUTED** (NG loses) |
| D1 | m_phi_lr_natgrad | NG @0.0005 → 271.9 ≈ killing | no LR rescues | strong | **NULL confirmed** |
| C2 | n_e_steps_em | unroll flat 144–145; ST 300→527 | extra E-steps don't help | strong (dir.) | **CONFIRMED** (qual.) |
| H1 | pos_extrapolation | learned-table +76.5% @4×N; offset flat | ≈25×SD (within-model) | **strong** | **PARTIAL** (rope regroup) |
| B3 | fisher_mu_precond | raw_T5 diverges 231; fisher flat 145 | +59% at T5 only | medium (1 draw) | **CONFIRMED** (fragile) |
| C1 | attention_entropy | surr +14.5% @κ0.25; +0.7% @κ1 | 4.8×SD @low-κ only | medium | **PARTIAL** |
| A3 | cg_coupling | cg_on −1.73% (val only; train ≈) | 1.1–1.7×SD | **weak** | **INCONCLUSIVE** |
| A2 | gauge_equivariance | tied +7.7% worse than untied | confounded by +55.6% params | confounded | **REFUTED w/ caveat** |

Not in this tree: EXP-3 (Σ_q calibration B1), EXP-6 (μP F1), EXP-7/F2 (rank-collapse —
`rho_handoff/anchor_rho1` is a single crashed arm, no `sweep_results.csv`), the cluster-G
thermodynamic experiments. EXP-1 (multi-seed) absent.

---

## A1 — gauge_transport (the central causal claim) — CONFIRMED, directionally

ON `≈154.1` (L1/L2) ≪ OFF `267.0 / 270.4` ≪ FROZEN `281.4 / 277.3`. Learned GL(K) transport
delivers a **43–44% PPL reduction**, ≈25–30× the seed-noise floor, and the gap is mirrored in
`train_ppl` (ON ≈132 vs OFF ≈241 vs FROZEN ≈250), so it is not a single-val fluke. `omega_identity_dev`
confirms the regime cleanly: OFF = 0 exactly (Ω=I to machine eps), FROZEN = 0.64 (small fixed random),
ON = 432–1432 (trained transport).

The clean isolation is **ON vs FROZEN**, not ON vs OFF: the verifier confirmed ON-vs-FROZEN differ in
only `gauge_transport` and `m_phi_lr`, with byte-identical `n_params=15152998` and identical
`pos_phi`/`phi_scale`; ON-vs-OFF additionally moves `phi_scale` and `pos_phi` (OFF drops 25.6k params).
ON-vs-FROZEN still gives a **123 PPL (44%)** advantage. **Interpretation:** *having* a non-trivial frame
(FROZEN) buys essentially nothing over flat Ω=I — all of the advantage comes from **learning** the
frame in the M-step. This is the program's strongest single result.

Two qualifications. (a) The secondary prediction `FROZEN ≤ OFF` is mildly **refuted**: FROZEN is
marginally *worse* than identity at both depths (+2.5% L2, +5.4% L1) — a random fixed rotation actively
hurts vs flat — but that 2.5–5% sub-ordering is within single-seed fragility. (b) The depth arm shows
no L1↔L2 difference in any cell (rank-collapse does not manifest at this depth/scale), so the
"transport prevents collapse at depth" story is not yet tested here. (c) `use_head_mixer=False` was
forced, so this certifies *flat-GL(K)-transport vs none*, not the shipped head-mixer model. The giant
ON effect needs no more seeds; the OFF-vs-FROZEN ordering needs ≥2.

## D1 — gauge_mstep_optim + m_phi_lr_natgrad — natural gradient REFUTED

AdamW-on-φ `144.5` ≪ pullback `252.3` (+74.6%) < killing `272.0` (+88.1%), at ≈50–59 SD with
byte-identical params and exactly the two intended differing config keys. Corroborated four ways
(val PPL, train_ppl trajectories that never cross AdamW's, `free_energy_total` on the objective itself
22.4 vs 43.8 vs 47.1, and finite-throughout). The LR sub-sweep does **not** rescue it: NG @ `m_phi_lr=0.0005`
→ 271.86, essentially identical to the killing conformal control (271.80) and *worse* than NG @0.015 —
lowering the LR moved PPL the wrong way. The pre-registered **null condition `killing = pullback` is
independently met**, and pullback also costs ≈6× wall-clock. This **refutes** the pullback-natural-gradient
hypothesis and matches the earlier mechanistic diagnosis (Fisher ≈ identity, `cos_nat_phi`≈0.99,
near-isotropic `pullback_cond`≈2.7, frozen weight norms) — the preconditioner is a near-no-op that
discards AdamW's per-coordinate adaptation. (This is what motivated the new `m_gauge_update_rule`
Adam-on-φ work; that variant is not in this sweep.) Only the fine pullback<killing ordering (≈5 SD,
n=1) stays tentative. Caveat: pre-registered K=64/n_heads=8 was run at embed_dim=20/n_heads=2, reducing
external validity to the intended regime. The LR sweep is also incomplete: 4 of 6 points missing
(0.0015 crashed at step 10000; 0.005/0.05/0.15 never created), so a high-LR optimum is not *formally*
excluded — but the crashed 0.0015 arm was on the same losing trajectory (train_ppl ≈320 at step 10000,
rising free energy), so the null is well-supported.

## C2 — n_e_steps_em (structural EM) — CONFIRMED qualitatively

**Unroll arm flat** at 144–145 PPL across n_e_steps {1,2,3,5,8}, best marginally at n=1. Extra E-steps
do **not** lower PPL, which is exactly the non-Neal–Hinton prediction: the E-step descends a target-blind
functional, not the likelihood, so iterating it does not serve CE. The `straight_through` control
degrades **monotonically** 300.6 → 526.8 (+75%), far outside any noise band and visible in training loss.
Reading: the E-step trajectory **must be differentiated through** (unroll) — detaching it (straight-through)
costs 2–3.6× PPL even at n=1, which is where a large share of the model's effective capacity is coming from.
Fragilities: the unroll flatness is "within noise" on one seed (≈0.5% spread, 0.17× the 2-SD bar) — but
flatness *is* the predicted outcome of a null-of-improvement, and train loss is flat too. The secondary
F-vs-CE decorrelation (Pearson) is **uncomputable** — the engineered diagnostic columns are empty in
`sweep_results.csv`. The straight-through degradation is also partly numerical (`guard_energy_klmax_frac`
grows 0→0.0137 with n_e_steps), i.e. a biased target-blind estimator going unstable, not the clean
"bias compounds" story — same conclusion, messier mechanism.

## H1 — pos_extrapolation — PARTIAL (offset vs absolute, with a rope regrouping)

The held-out-CE-vs-N curve (the actual test) is decisive: the **learned absolute position table**
degrades catastrophically, CE 4.99 → 5.56 nats (≈+76.5% PPL) at 4× train length. Because the same fixed
checkpoint is evaluated at every N, this is a within-model slope with seed variance differenced out —
≈25× the noise floor and robust. All offset-*in-effect* schemes preserve length generalization: **alibi**
improves slightly with N (5.090→5.071), **t5** is flat then rises mildly (5.061→5.102), **rope** is flat
(≈5.149). The strict 4-way prediction (alibi/t5 good, learned/rope bad) is **refuted on rope**: RoPE's
attention score depends only on i−j, so it is offset-only *in effect* despite per-absolute-index
application — its "absolute" label is a mislabel, not a counterexample. Regrouped, the clean statement
holds: **a learned absolute table fails to extrapolate; everything offset-in-effect generalizes.**
Confound (disclosed): the `gamma`/model-coupling channel uses `causal_alibi` for *all* arms including
learned/rope, so the "absolute" arms still receive offset help on that channel — the learned table
collapsing *despite* this only strengthens the result. In-distribution (N=128) the ranking inverts
(learned 146.6 best, rope 172.2 worst): the absolute table wins at train length and loses everywhere
beyond it.

## B3 — fisher_mu_precond — CONFIRMED but fragile

The headline is **stability, not accuracy**: Fisher μ-preconditioning (`nat_μ = Σ·grad_μ`) stays flat
at 144.5–145.0 PPL across n_e_steps {1,3,5}, while **raw Euclidean grad_μ diverges at T5** (231.2,
+59%). The divergence is a genuine intrinsic inner-loop instability — `grad_norm_mu` 0.06→185→987,
free energy *increases* (`estep_f_drop=+0.034`, `nondecreasing_frac=0.60`), finite throughout (not a
NaN/logging artifact). At the production n_e_steps=1, raw is only +4.1% worse (150.4 vs 144.5) and at
T3 it is marginally *better* (144.6 vs 145.0) — so the preconditioner barely matters for PPL at shallow
E-steps; its value is robustness headroom as the E-step iterates. The verdict rests on a single divergent
draw at one seed (whether raw always diverges at T5 or only at this seed/LR is untested), but the
mechanism — a raw Euclidean step is mis-scaled in the σ-whitened E-step geometry — is intrinsic.
Config verified clean (only `e_step_mu_precond` and `n_e_steps` vary).

## C1 — attention_entropy (canonical-F vs surrogate) — PARTIAL

The canonical attention-entropy term `τ·β·log(β/π)` is **load-bearing only when attention is diffuse**.
Surrogate (entropy off) vs canonical: **+21.14 PPL (+14.5%) at κ=0.25** (≈4.8× the noise bar, echoed
in train loss → real optimization effect), but only **+1.04 PPL (+0.7%, below 1 SD) at κ=1**. So at the
production κ=1 sharp-attention operating point the entropy term is empirically a derivation nicety the
M-step washes out; it becomes consequential as κ falls and the softmax diffuses. **Tension to flag:** the
`cov_gap` diagnostic that was supposed to mechanistically explain this (the −τ⁻¹Cov_β(E,∇E) gap) *rises*
with κ (0.246 → 0.354), the **opposite** of the "larger Cov_β at low κ" rationale — so the figure's
`cov_gap` column does not quantitatively tie the PPL gap to the covariance gap. The PPL interaction is
solid; the named mechanism diagnostic is not yet doing its job. Config verified clean (only
`include_attention_entropy` and `kappa_beta` vary; both arms on the unrolled oracle).

## A3 — cg_coupling — equivariance CONFIRMED, capacity INCONCLUSIVE

Equivariance is preserved (gauge residuals ≈1.5e-7 both arms; +40 params for the CG scalar path
weights — 0.0008% of the model). The PPL "win" is **not** established: cg_on is −1.73% on validation
(268.7 vs 273.5) but the two arms are **identical on training** (cg_on marginally *worse*), the result
rests on a single val eval, and −1.73% is 1.1–1.7 SD — below the credible bar. A generalization-only
effect with identical training is indistinguishable from seed/eval noise here. Pre-registration scope
gap: A3 specified *both* decodes, but only `use_prior_bank=false` (KL-to-prior) ran — the linear-decode
arm was never run, so the capacity half of A3 is essentially untested. The exactly-equivariant
cross-irrep channel works; whether it buys capacity needs 3–5 seeds and both decodes.

## A2 — gauge_equivariance (tied vs untied) — REFUTED with a hard confound

On its face, the exactly-equivariant **tied** arm is +26.4 PPL (+7.7%) *worse* than the
equivariance-broken **untied** arm (367.3 vs 340.9), a gap ≈10× the imported SD. The manipulation
landed (`builder_resid` 1.3e-7 tied vs 0.92 untied), and the only swept config key is `gauge_group`.
**But the untied arm carries +55.6% parameters** (14.10M vs 9.06M — the head mixer adds parameters under
the untied gauge). In this capacity-starved underfit regime a +55.6% capacity gain could plausibly
explain the entire 7.7% gap, leaving the equivariance tax ≈0. The 26 PPL is therefore an **upper bound**
on the cost of strict equivariance, not its value. No param-matched control exists. A param-matched,
multi-seed control is required before "strict equivariance costs PPL" can stand.

## Recommended next steps (priority order)

1. **Run EXP-1 (multi-seed variance floor) at this 15k/embed_dim=20 regime** — without it none of the
   sub-3% effects (A3, the A1 FROZEN-vs-OFF ordering, the C2 unroll flatness, the C1 κ=1 null) are
   formally interpretable. ≥3 seeds, ideally with `eval_interval` < `max_steps` for val curves.
2. **A2 param-matched control** — add a tied arm with width raised to match untied's 14.1M params (or an
   untied arm shrunk to 9.06M). This is the single fix that turns A2 from confounded to publishable.
3. **A1 reseed ×2** to settle OFF-vs-FROZEN and add an L=4 depth cell where rank-collapse can manifest.
4. **A3** — rerun with both decodes and 3–5 seeds; report train-side and a Σ-conditioned eval.
5. **C1** — fix or re-derive the `cov_gap` diagnostic so it tracks the PPL interaction (currently
   anti-correlated with κ).
6. **B3** — reseed raw_T5 ×2 to confirm the divergence is generic, not a single unlucky draw.
7. **D1** — the natural-gradient null is solid; the actionable follow-up is the new Adam-on-φ
   `m_gauge_update_rule` arm (not in this sweep) vs AdamW-on-φ.
