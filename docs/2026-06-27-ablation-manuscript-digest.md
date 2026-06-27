# vfe3 ablation/scaling results: digest and GL(K) manuscript-inclusion recommendation

Date: 2026-06-27. Scope: digest the completed experiments in `vfe3_ablation_results/`
and `vfe3_scaling_results/grow_K_GL10/` (from `docs/hypotheses/2026-06-21-hypotheses.md`)
and decide what, if anything, belongs in `GL(K)_attention.tex` / `GL(K)_supplementary.tex`.
Manuscript line numbers refer to the freshest WIP in the Research vault
(`Research/manuscripts/GL(K)_*.tex`), which is byte-identical to the repo mirror for these
two files. All numbers below were read from the sweep CSVs / per-run JSONs and verified by a
3-agent workflow (run `wf_560f7477-93f`).

## Bottom line

Eight of the fourteen experiments are manuscript-relevant; three are the strongest. The single
highest-value result (gauge transport on/frozen/off) fills a gap the paper names *verbatim* as
"not yet run." The new runs are best framed as a **mechanistic-ablation** package that
substantiates the manuscript's existing predictions, NOT as replacements for the headline tables:
they live at a different (smaller, `embed_dim=20`) scale and report relative contrasts, not
comparable absolute perplexities.

## Two experimental generations (read this first)

The manuscript's existing empirical section (attention.tex lines ~2052–2308) reports an **older**
run generation: `train_vfe.py`, GL+(10)/GL+(15), K=90–120, 60k steps, batch 16, n=2-seed sweep,
prior-bank decode. Headline GL+(15) K=90 = test PPL 71.6; sweep best 64.9 at K=120; Japanese
GL+(15) = 24.2.

The **new** `vfe3` results (`train_vfe3.py`) split internally:
- **Ablations** — `embed_dim=20`, 15k steps, `use_head_mixer=False`, **single seed=6**, linear
  decode (`use_prior_bank=False`). Operating point ≈ 144.5 PPL. These are relative contrasts.
- **Scaling** (`grow_K_GL10`) — 60k steps, `use_head_mixer=True`, **3 seeds (6/23/64)**, axis is
  `embed_dim` (= the cell label "K"), spanning 10–70.

Consequence: the new scaling curve (embed_dim 10–70, floor ≈ 70 PPL) is a **disjoint regime**
from the paper's K=90–120 sweep and must be plotted/reported separately, never spliced into one
law. The ablation perplexities (~144) are not comparable to the 71.6 headline; report them as
percentage contrasts at their own operating point.

## Inclusion recommendation table

| ID | Experiment | Effect | Verdict | Home |
|----|-----------|--------|---------|------|
| R1 | gauge_transport on/frozen/off | ON 154 vs FROZEN ~279 (~45%) | **INCLUDE — headline** | main Experiments + line 2287 |
| R3 | attention_entropy canonical vs surrogate | 21 PPL (~14%) at κ=0.25 (sharp); ~1 PPL at κ=1.0 (diffuse) | **INCLUDE — high** | main ablations + supp App. B |
| R2 | grow_K scaling (3-seed) | b≈−1.15, floor≈70, R²=0.999 | **INCLUDE — supp** | supp App. F (separate regime) |
| R4 | renyi_order divergence sweep | KL optimal; α≥1.5 catastrophic | **INCLUDE — supp** | supp App. H |
| R11 | pos_extrapolation (corrected) | absolute +11.4%; relative flat | **INCLUDE — supp** | supp App. E §E.9 |
| R5 | n_e_steps EM (unroll vs ST) | unroll flat; ST 300→527 | INCLUDE — supp (medium) | supp App. D |
| R6 | fisher_mu_precond | raw blows up at T5 (231); fisher stable | INCLUDE — supp (medium) | supp App. D §D.1 |
| R10 | gauge_mstep_optim | AdamW 144 vs Killing 272 / pullback 252 | **FLAG only** (LR-confounded) | see "Tension" below |
| R8 | gauge_equivariance tied/untied | tied 367 vs untied 341 | EXCLUDE unless param-matched | supp App. C (confounded) |
| R7 | cg_coupling | 273.5 → 268.7 (1.7%) | EXCLUDE (within noise) | — |
| R9 | kappa_beta_per_head | all 145.6–147.3 (null) | EXCLUDE (within noise) | — |
| R12 | rho_handoff rank-collapse | no collapse / no anchor effect @ L=4 | EXCLUDE (null at shallow depth) | — |
| — | m_p_sigma_lr, phi_weight_decay, m_phi_lr_natgrad | tuning | EXCLUDE | — |

Decision rule (from the measured noise floor, below): single-seed effects below ~2% are inside
seed noise and must not be reported as wins; effects above ~3% are trustworthy point estimates.

## Measured noise floor and scaling law (R2, the only multi-seed run)

Per-K test PPL (mean of seeds 6/23/64), with SD / CV:

| K=embed_dim | n_params | mean PPL | SD | CV% |
|---|---|---|---|---|
| 10 | 7.60M | 219.01 | 1.32 | 0.61 |
| 20 | 15.15M | 135.65 | 1.08 | 0.80 |
| 30 | 22.70M | 113.14 | 0.65 | 0.57 |
| 40 | 30.26M | 101.40 | 1.10 | 1.09 |
| 50 | 37.81M | 94.06 | 1.02 | 1.08 |
| 60 | 45.36M | 88.92 | 0.53 | 0.59 |
| 70 | 52.91M | 83.94 | — | (s6 only; s23/s64 incomplete) |

Noise floor: SD-based CV 0.57–1.09%; max–min spread up to ~2.1% (K=50). Offset power law
`PPL = a·K^b + c` fits with b ≈ −1.14 to −1.19, floor c ≈ 69–72, R² = 0.999 (a pure power law
without offset is markedly worse, R² = 0.977). The axis is `embed_dim`; n_params is linear in it.
This is the only result that carries error bars, and it partially discharges the manuscript's own
promise (line 2419) that "companion-paper experiments will report n≥5 seed runs."

## Detailed per-result findings

### R1 — gauge_transport (HEADLINE; fills a named gap)
Param-matched, architecture-identical contrast (both `pos_phi='learned'`, identical 15,152,998
params; only `m_phi_lr` differs): **ON 154.1 vs FROZEN ~279.4 (~45% PPL reduction)** at both L=1
and L=2. FROZEN (~279) > OFF (~268.7) shows a random frame is worse than exact identity. The
manuscript states at line 2287: *"Whether it stems specifically from the gauge-transport geometry
... awaits the controlled gauge on/off ablation, which is not yet run."* This result answers that
sentence directly.
- Confound: the ON-vs-OFF contrast is NOT clean — `gauge_transport='off'` also sets
  `pos_phi='none'` (drops all positional gauge), conflating transport with positional encoding.
  Report ON-vs-FROZEN as the clean causal contrast; report OFF only as the secondary
  "random frame worse than identity" point.
- Caveat: single seed; embed_dim=20 (not comparable to the 71.6 headline).

### R3 — attention_entropy (first empirical test of the canonical-vs-surrogate gap)
canon κ0.25 = 146.11, canon κ1.0 = 144.47; surr κ0.25 = 167.24, surr κ1.0 = 145.51. Note κ
sets the softmax temperature τ = κ·√d, so κ=0.25 is the **lower-temperature, sharper-attention**
setting (attn_entropy ≈ 1.4) and κ=1.0 is the **higher-temperature, more diffuse** one
(attn_entropy ≈ 2.5). The canonical attention-entropy term is load-bearing at the sharper setting:
a ~21 PPL (~14%) gap at κ=0.25 collapses to ~1 PPL (within noise) at κ=1.0, consistent with the
−τ⁻¹Cov_β gap vanishing in the τ→∞ (diffuse) limit. This matches the
manuscript's own prediction that the two gradients differ by `−τ⁻¹ Cov_β(E, ∂E/∂x)` (attention.tex
lines 887–893; supp App. B lines 189, 277), a covariance that scales with attention diffuseness
and vanishes as τ→∞. The paper already names "the envelope (canonical) attention-gradient
convention" (line 2064) but has had no empirical confirmation. Both arms ran on the autograd
oracle, isolating the entropy term from the kernel-vs-oracle route.

### R2 — grow_K scaling
See the table/law above. Place in supp App. F (§F.3 testable predictions / §F.5 numerical), as a
separate regime from the paper's K-sweep. Note it has no standard-transformer baseline, so it does
NOT measure the conjectured cross-architecture sample-efficiency ratio R(K) — it is the VFE PPL(K)
curve only.

### R4 — renyi_order
α ∈ {0.5,0.8,1.0,1.2,1.5,2.0} → 147.5/145.4/**144.5**/148.7/222.1/267.2. Minimum at α=1 (KL);
α≥1.5 catastrophic with rising kl_max saturation. Corroborates the *spirit* of supp App. H
(KL is the unique convex f-divergence giving the closed-form geometric-mean update). Caveats:
Rényi-α is not itself an f-divergence (a monotone transform of the α-divergence), so this supports
rather than tests the theorem's exact hypotheses; on WikiText, not the Japanese Rényi-½ setting;
the α∈{0.5,0.8,1.2} differences straddle the noise floor (report "flat near α=1, sharply worse for
α≥1.5", not a fine ranking).

### R11 — pos_extrapolation (digest corrected)
Eval CE at N=128/192/256/384/512 (train length 128), from `extrap_ce`:
- LEARNED (absolute): 4.99→5.56, **+11.4%** (PPL 146.6→258.8) — collapses.
- ALIBI (offset): 5.090→5.071, −0.37% — improves slightly.
- T5 (offset): 5.061→5.102, +0.83% — near-flat.
- ROPE (relative rotation): 5.149→5.147, −0.02% — flat.
Clean story: **absolute position fails to extrapolate; offset/relative (alibi, T5, RoPE) all hold.**
(My initial digest wrongly grouped RoPE with the degraders; RoPE is flat.) In-distribution ordering
is the opposite of robustness (learned best at N=128, worst extrapolator). Natural home: supp
App. E §E.9 (sequence-length sensitivity, line 879).

### R5 / R6 — EM structure and natural-gradient stability (supplement, medium)
R5: under unroll, PPL is flat across n_e_steps {1,2,3,5,8} (144.5–145.2, within noise); straight-
through degrades monotonically (300.6→526.8). The non-Neal-Hinton EM signature — q already serves
descent at T=1; the gradient estimator, not the E-step count, is the lever. Fits supp App. D and
the structural-EM note (attention.tex line 2121, n_E-step=1 at line 2129).
R6: Fisher (natural-gradient) μ-preconditioning is stable across n_e_steps (144.5/145.0/145.0)
while raw Euclidean μ-grad blows up at T=5 (231.2). Numerical justification for the natural-gradient
μ update (supp App. D §D.1). Report as a stability result, not a PPL win.

## Tension worth your attention (R10)

`gauge_mstep_optim`: AdamW-on-φ (144.5) beats both natural-gradient M-step variants
(Killing 271.8, pullback 252.3; pullback ~6× slower). This **apparently tensions with** the
supplement's stated claim (lines 560 / 578 / 673) that "the reported runs use the block-diagonal
Killing-form variant." BUT it is **inconclusive**: the LR-rescue sweep (`m_phi_lr_natgrad`) is
incomplete (1 cell), so the natural-gradient arms are very likely just LR-mistuned (they bypass
Adam's normalization). Do not include as a result and do not let it overturn the stated optimizer
choice until a matched-LR sweep is run. Flagging because, if it survives a proper LR sweep, it
would require editing a factual claim in the supplement.

## Excluded, with reasons
- R7 cg_coupling (1.7% < noise floor; the strict-equivariance property is real but the PPL gain
  is not established). R9 kappa_beta_per_head (null within noise). R12 rho_handoff (no rank
  collapse and no anchor effect at depth 4 — a shallow-depth null that does not resolve the
  manuscript's deep-stack question). Tuning sweeps (m_p_sigma_lr, phi_weight_decay,
  m_phi_lr_natgrad). R8 gauge_equivariance is confounded by a 55% parameter gap (untied 14.1M vs
  tied 9.06M), so the 26-PPL "cost of equivariance" cannot be attributed to equivariance alone;
  quote it only against the param-matched A2 control noted in the git log, not this pair.

## Open data gaps
- All ablations are single-seed (seed=6). A 3-seed repeat of at least R1 and R3 would convert
  point estimates into error-barred claims at the ablation operating point.
- B1 (Σ_q calibration) produced no artifact — not run.
- Scaling K=70 is s6-only (s23/s64 incomplete); report K=70 as single-seed or omit.
- The manuscript's hyperparameter table (attention.tex lines 2086–2117) is an all-"TBD"
  placeholder; the new runs supply concrete values for the `train_vfe3` regime, but those differ
  from the `train_vfe` headline runs the table describes, so do not fill it blindly.

## Concrete edits (APPLIED 2026-06-27 to the vault WIPs, high-value set)
The user approved the high-value set (R1+R3 main text, R2/R4/R11 supplement). Applied to
`Research/manuscripts/GL(K)_*.tex`:
1. `GL(K)_attention.tex` (Results): the "...awaits the controlled gauge on/off ablation, which is
   not yet run" sentence now points to a new paragraph "Direct gauge-transport and attention-entropy
   ablations," reporting R1 (clean param-matched ON-vs-FROZEN ~45% reduction; OFF/identity as the
   secondary, confound disclosed) and R3 (canonical-vs-surrogate, framed by the τ→∞ vanishing of
   `eq:autograd_envelope_gap`), both flagged single-seed at embed_dim=20 and not commensurable with
   the K=90 table. References the new Supplementary Appendix~J for the noise floor.
2. `GL(K)_supplementary.tex`: a new last appendix, "Empirical Ablations of the Gauge VFE Language
   Model" (App. J, label `app:vfe3_ablations`), with three subsections — embedding-dimension scaling
   (R2, 3-seed, offset power law, Table `tab:vfe3_scaling`), divergence order (R4, cross-ref the
   forward-KL uniqueness App. H), and positional-prior extrapolation (R11, cross-ref main-text
   positional priors). Placed last so existing hard-coded appendix letters (A–I) do not shift.
Verified: all new labels unique, all cross-references resolve, no banned terms. Not applied:
R5/R6/R8/R10 (left for a later supplement pass or pending the R10 matched-LR sweep).
