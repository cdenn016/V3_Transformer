---
title: V3_Transformer Research Directions
date: 2026-06-21
method: /hypothesis-generation + multi-agent Workflow (wf_0b06790b-144)
provenance: 69 seeds mapped (GL(K)_attention/supplementary, meta_entropy, belief_inertia, codebase config.py, wiki, verified-ledger §3/§4) -> 40 hypotheses across 8 expert lenses -> 37 survived adversarial filtering (falsifiable / novel-vs-ledger / feasible-on-code) -> 3 killed. 54 agents, ~4.7M subagent tokens.
scope: experimental + extension directions; the analytic math core is review-exhausted (see verified-ledger.md).
---

# V3_Transformer Research Directions

## Executive summary

The gauge-theoretic VFE transformer has reached a state in which its mathematical core is review-exhausted but its central empirical claims remain almost entirely unmeasured, and the highest-value work is now experimental rather than further derivation. The single most consequential experiment is the unrun gauge ON/OFF/frozen-random ablation: the program asserts that the VFE advantage stems from GL(K) transport geometry, yet no run has ever compared trained transport against the exact Ω=I control, and that comparison is reachable today by config alone on one RTX 5090 across roughly nine runs. Beyond it, a cluster of cheap, decisive probes is ready: whether the belief covariance Σ_q carries calibrated decode-time uncertainty (a "free" channel standard transformers lack), whether the canonical attention-entropy term is empirically load-bearing or a derivation nicety the M-step washes out, and whether the headline single-seed perplexities even survive an honest across-seed error bar. A second tier tests structural predictions the manuscripts make but never measured — μP width-stability of the inverse-K scaling exponent, prior-anchoring as the anti-rank-collapse brake that substitutes for the FFN, and the thermodynamic-limit and belief-inertia predictions from the meta-entropy and belief-inertia papers. The recurring discipline across all of it is to pin the confounds the code actually imposes (parameter-count matching for Ω=I, the kernel-vs-oracle route for entropy ablations, n_e_steps convergence for any fixed-point or covariance claim, and per-K kl_max for scaling), and to report band-overlap decisions rather than speculative point margins.

## Thematic clusters

The 37 surviving hypotheses collapse into eleven unified entries after merging near-duplicates across the gauge, info-geometry, ml-engineer, transformer, geometry, variational, statmech-dyn, and numerical lenses. The merges are noted inline; several hypotheses that three or four lenses arrived at independently (gauge ON/OFF, Σ_q calibration, the canonical-entropy gradient gap) are the strongest signal that they belong at the top.

### Cluster A — Gauge and equivariance

This cluster owns the program's central causal claim and the representation-theoretic capacity channels.

**A1. Gauge ON vs OFF (Ω=I) vs frozen-random.** Merges S1 [gauge] and S2 [ml-engineer], which are the same experiment reached by two lenses: realize gauge-OFF by `phi_scale=0` + `e_phi_lr=0` + `m_phi_lr=0` + `pos_phi='none'` so Ω=exp(0)exp(0)=I exactly, a frozen-random control at `phi_scale=0.06` with the learning rates zeroed, and the trained baseline. S2 contributes the depth arm (L=2/3, since at L=1 rank-collapse cannot manifest) and the parameter-count and `use_head_mixer=False` confound controls. The decision rule is non-overlapping ±1 SD bands at ≥3 seeds, not a speculative ≥5% margin.

**A2. Exact-equivariant tied vs strictly-broken untied gauge.** S16 [gauge] — the residual-vs-step drift of `block_glk`+`use_head_mixer` (broken as the mixer drifts) against `tied_block_glk` (exact), with the PPL cost of strict equivariance as the open empirical question. Requires `family='gaussian_full'` + `use_prior_bank=True` and a per-eval residual hook (the metric currently fires only once at end-of-run). *Built (2026-06-22): the per-eval builder-break residual (`val_builder_resid`) is logged in `_val_diagnostics`, and `gauge_residual_drift` auto-plots the tied-vs-untied drift vs step.*

**A3. Clebsch-Gordan cross-irrep coupling.** S5 [gauge] — `use_cg_coupling` is the only exactly-equivariant inequivalent-irrep channel; a Schur-commutant head mixer is structurally forbidden across irrep types. Test on an SO(3) two-type tower; the equivariance half is near-certain, the capacity (PPL-delta) half is the genuine gamble, capped because CG is means-only. *Built (2026-06-22): `ppl_equivariance_bars` auto-plots per-arm PPL with the median equivariance residual for the `cg_coupling` sweep.*

**A4. Trained Regime-II connection: holonomy and route covariance.** Merges S26 [gauge], S35 [gauge], and the holonomy half of S34/S36 [statmech-dyn]. The settled half (telescoping, Route-A-vs-B asymmetry) is pinned by existing tests; the open empirical contribution is the trainability-at-scale curve (does CE training drive `connection_W`/`connection_M` to a stable nonzero optimum, and does holonomy track ‖connection‖) plus a new training-time rebuilt-equivariance probe. S26's span/holonomy-locality probe rides on this as a cheap follow-up, conditioned on `pos_rotation='rope'`.

### Cluster B — Information geometry and divergences

**B1. Σ_q as calibrated Fisher uncertainty.** Merges S3 [info-geo] and S6 [transformer], the two strongest formulations of the same calibration question (open thread f). S3 is the cheap post-hoc correlation/recalibration on one checkpoint; S6 is the three-decode-arm training comparison (bare-mean vs precision-scaled η=μ/(σ+ε) vs KL-to-prior). Both flag the same load-bearing risk: at `n_e_steps=1` from `sigma_init=1.0` the per-token tr(Σ_q) spread is only ~4%, so a pre-registered spread gate is mandatory and a rescue arm at `n_e_steps≥4`/`gaussian_full` is required.

**B2. Alpha-divergence (Rényi) attention sweep.** Merges S8 [info-geo] and S27 [numerical]. S8 sweeps `renyi_order` for mode-seeking/mass-covering attention-entropy ordering; S27 is the same mechanism's pathology — for α>1 the non-PD blend saturates to `kl_max` with zero gradient, predicting a non-monotone H(β)-vs-α curve in the tail. They must run together: S27 is the saturation diagnostic that explains an S8 non-monotonicity, and both demand all arms on the oracle with mu-separation/τ controlled.

**B3. Fisher natural-gradient E-step preconditioning.** S17 [info-geo] — the μ-arm ablation (nat_μ→raw grad_μ) is clean; the σ-arm mechanism in the original is wrong because `retract_spd_diagonal` already whitens by 1/σ, so a genuine Euclidean σ test requires registering a new additive-Euclidean σ retraction, not feeding raw grad into the affine retraction.

### Cluster C — Variational and free-energy structure

**C1. Canonical-F vs entropy-suppressed surrogate.** Merges S4 [ml-engineer] and S18 [variational], the two formulations of the −τ⁻¹Cov_β gradient gap. S4 establishes that the closed-form kernel never computes the entropy term (so production "canonical" has trained on the surrogate gradient all along) and mandates a CANON_ORACLE control; S18 adds the n_e_steps-scaling prediction and the two confounds that must be equalized (`oracle_unroll_grad=True` in both arms; force the trajectory F logger to canonical F in both).

**C2. Structural non-Neal-Hinton EM: n_e_steps and the F-vs-CE decorrelation.** Merges S28 [variational] and S31 [numerical]. The theory is settled (the E-step is target-blind, descends a distinct functional); the open empirical object is whether n_e_steps>1 moves PPL and whether final E-step F correlates with CE across the sweep. S31 contributes that the σ-sector is the rate-limiting channel. Pin `e_step_gradient` across the sweep (unroll deepens the graph, a confound).

**C3. lambda_h hyper-prior empirical-Bayes shrinkage.** S10 [variational] — Arm A (`prior_source='token'`) is a code-guaranteed inert wiring check; Arm B (`prior_source='model_channel'`, `use_prior_bank=True`) tests rare-token shrinkage toward the barycenter centroid.

**C4. Mean-field full-vs-diagonal F-gap.** S37 [variational], reframed: drop the untestable MCMC/posterior-bias claim (the E-step has no likelihood channel) and measure whether `gaussian_full` reduces the model's own F_red vs `gaussian_diagonal` on a dense-Ω stack, as a function of off-diagonal mass of Ω.

### Cluster D — Optimizer geometry on the gauge M-step

A tightly converged cluster: S9 [info-geo], S12 [geometry], and S14 [ml-engineer] are three readings of "does the pullback natural-gradient M-step beat AdamW-on-φ, and in what regime."

**D1. Pullback natural-gradient gauge M-step.** Merges S9, S12, S14. S9/S12 ask whether pullback descends F faster than AdamW and predict the gap scales with frame norm (killing is conformal, a no-op); S12's critical fix is that the regime knob is `mass_phi` (default 0.0), not `phi_weight_decay` (hard-zeroed under natural-grad). S14 is the LR-mis-scaling sub-experiment: GaugeNaturalGradAdamW bypasses Adam normalization so the AdamW-tuned `m_phi_lr=0.015` mis-scales, and a log-spaced LR sweep should place the natural-grad optimum ≥2x away. All three require `e_phi_lr=0` to keep the preconditioner off the E-step.

### Cluster E — SPD retraction geometry and numerics

**E1. SPD chart and the sigma_max congruence break.** Merges S13 [geometry], S29 [numerical], and S33 [geometry]. S13 measures whether E-step σ-trajectories are AIRM-geodesically short but Euclidean-long (largely definitional on the diagonal/flat cone; the diagonal-vs-full gap is the real signal). S29/S33 quantify the known `sigma_max` clamp break of congruence-equivariance, correctly restated at the active `sigma_max=10.0` (not the dead default 5.0) and decided inert-vs-load-bearing from the already-logged `guard_sigma_ceil_frac`. S33's mechanism correction: on the diagonal family `spd_affine` whitens by 1/σ while `log_euclidean` does not, so the charts diverge as |log σ| everywhere, independent of the ceiling.

**E2. bf16 autocast transport-matmul exposure.** S30 [numerical], reframed: the diagonal-KL energy is fp32-islanded, so the only genuine bf16 exposure is the upstream transport einsums; the predicted null (PPL/entropy within noise) is the likely and useful outcome certifying bf16 as a safe throughput default.

**E3. Mean trust region as stability guard.** S32 [numerical] — `e_mu_q_trust` box mode is a near-no-op at production `embed_dim≤64`/`e_q_mu_lr=0.5`; it binds only at large `embed_dim` or raised LR, with NaN/loss-spike rate (not PPL) as the endpoint, and the ball radius needs sqrt(K)-scaling.

### Cluster F — Scaling, depth, and RG

**F1. μP width-stability of the inverse-K exponent.** S7 [ml-engineer] — re-run grow_K width-fixed vs μP-corrected (scale `e_q_mu_lr`/`m_p_mu_lr`~1/K, `mu_init_std`~1/√K), with the critical fix that `kl_max` is frozen at 160 from `embed_dim=20` and must be recomputed per-K in both arms, and the fit re-done against `embed_dim` not `n_params`.

**F2. Prior-anchoring resists rank collapse (the FFN-brake substitute).** S11 [transformer] — sweep `n_layers` on the pure no-MLP path, measuring the Dong rank-one residual r(X)=‖X−1xᵀ‖_F/‖X‖_F (a new metric; existing `effective_rank` is the wrong object). Directional null confirmed in a probe; drop absolute thresholds for a per-layer decay-rate comparison; control `e_phi_lr` and the ρ=0-vs-ρ=1 handoff.

**F3. Prefactor-only gauge advantage and same-exponent universality.** S19 [transformer], scoped: the VFE-internal gauge-on-vs-off data-budget scaling is runnable now; the cross-architecture exponent claim is fenced behind a standard-transformer baseline that does not exist in the repo.

**F4. Empirical y3 RG exponent and fit-window stability.** Merges S36 [statmech-dyn] and the RG half of S26. The flat-path g2 (inter-cluster β) depth-scaling is the runnable headline; the g3/holonomy arm requires the opt-in `regime_ii` path and is fenced. The open object is strictly fit-window stability on trained reps, not the settled nominal y3=−1.

### Cluster G — Thermodynamic limit and belief dynamics

These test the meta_entropy.tex and belief_inertia.tex predictions (open thread d, e), the freshest unmeasured theory.

**G1. Susceptibility χ=1/lambda_alpha transfer.** S23 [statmech-dyn] — does the toy Gaussian's mean-consensus response law survive in the trained population? Must train and extract at converged `n_e_steps` (at `n_e_steps=1`, q=p, α drops out of the first-order response — a trivial-failure trap).

**G2. Sanov LDP rate function.** S24 [statmech-dyn] — fit the empirical rate from sub-sampled per-token F-density; foreground the Fisher-Rao (det Σ)^{−(K+2)/2} volume-tracking as the discriminating endpoint and run uniform vs causal prior as the explicit homogeneous-vs-heterogeneous contrast the manuscript predicts.

**G3. Symplectic momentum E-step: √(M/K) overshoot law.** S25 [statmech-dyn] — add a leapfrog E-step variant with Fisher-trace mass; sanity checks (overdamped→first-order, γ→∞→DeGroot) plus the slope-½ overshoot-vs-mass law against the slope-1 ballistic null, scoped to the near-converged frozen-attention regime.

**G4. Consensus-Laplacian frustration gapping.** The frustration half of S34 [statmech-dyn] — a codebase-invented proxy (algebraic connectivity of the consensus Laplacian, validated on a known-gapped toy first), explicitly demoted below the manuscript's actual thermodynamic-integration test, which requires a meta-ensemble sampler that does not exist.

### Cluster H — Positional structure and head specialization

**H1. Offset-only priors extrapolate; absolute/RoPE do not.** S21 [transformer] — train at `max_seq_len=128`, eval at growing N; ALiBi/T5 (functions of |i−j|) extrapolate, `pos_phi='learned'` and RoPE do not. Fix the two code traps: `pos_phi` truncates (shape error, not IndexError — test with a shorter table) and `t5_max_distance` must be ≥ max eval-N to isolate offset from bucket saturation.

**H2. Per-head temperature dispersion.** S20 [transformer] — a hand-set per-head `kappa_beta` list (zero added params) vs a tied-tau baseline retuned to the geometric-mean tau (the mandatory confound control). The learnable arm is unbuilt; scope to the hand-grid. *Built (2026-06-22): the `kappa_beta_per_head` sweep gained the geo-mean-τ confound-control arms, and `kappa_dispersion` auto-plots PPL vs std(κ_β).*

### Cluster I — Empirical rigor and ablations

**I1. Multi-seed variance floor.** S15 [ml-engineer] — the cheapest, highest-leverage deliverable: 5-seed K=20 baseline error bar overlaid on the live narrow ablation grids to flag seed-noise-dominated "wins." The phenomenon is disclosed in the ledger; the measured error bar and overlay are not.

### Cluster J — Structural metric probes

**J1. Learned gauge frames as a syntactic Mahalanobis metric.** S22 [geometry] — S(Ω)/d_AI vs dependency-tree distance. Requires a real isometry projector (not `project_phi_to_slk`, which removes only the trace), a coded S(Ω) functional, a UD/CoNLL-U data pipeline (absent), and a `gaussian_full` retrain — substantial new infrastructure, hence low feasibility.

## Prioritization table

Ranked by impact × feasibility, ties broken by novelty. Stars: ★★★★★ = run this week.

| # | Hypothesis (cluster) | Impact | Feas. | Novelty | Priority | First experiment |
|---|---|---|---|---|---|---|
| 1 | Multi-seed variance floor (I1) | 3 | 5 | 3 | ★★★★★ | 5-seed K=20 baseline; SD overlay on live grids |
| 2 | Gauge ON/OFF/frozen-random (A1) | 5 | 4 | 5 | ★★★★★ | 3-cell × 3-seed `gauge_transport` sweep |
| 3 | Σ_q calibration (B1) | 4 | 5 | 4 | ★★★★★ | Post-hoc tr(Σ_q)–nats Spearman + ECE recalibration |
| 4 | Canonical-F vs surrogate (C1) | 4 | 5 | 4 | ★★★★★ | SURROGATE vs CANON_ORACLE, both on oracle |
| 5 | Structural EM / n_e_steps (C2) | 3 | 5 | 2 | ★★★★ | Registered n_e_steps sweep {1,2,4,8}, F-vs-CE corr |
| 6 | μP width-stability (F1) | 4 | 4 | 4 | ★★★★ | grow_K vs μP route, per-K kl_max, fit vs embed_dim |
| 7 | Rank-collapse / FFN brake (F2) | 3 | 4 | 4 | ★★★★ | n_layers sweep, Dong r(X) metric, anchor on/off |
| 8 | Pullback NG M-step + LR (D1) | 3 | 4 | 4 | ★★★★ | mass_phi sweep, pullback vs killing vs AdamW |
| 9 | Tied vs untied equivariance (A2) | 3 | 4 | 3 | ★★★ | block_glk vs tied_block_glk, residual-vs-step hook |
| 10 | CG cross-irrep coupling (A3) | 3 | 5 | 4 | ★★★ | SO(3) two-type tower, coupling on/off, both decodes |
| 11 | Per-head temperature (H2) | 3 | 4 | 3 | ★★★ | Hand-set kappa_beta list vs geo-mean-tau baseline |
| 12 | Rényi α-attention + saturation (B2) | 3 | 5 | 3 | ★★★ | α∈{0.5,…,1.5} all-oracle, H(β) + non-PD mask frac |
| 13 | Offset-prior extrapolation (H1) | 3 | 4 | 3 | ★★★ | Train @128, eval @{192…512}, CE-vs-N |
| 14 | Fisher NG E-step (μ-arm) (B3) | 3 | 4 | 3 | ★★★ | nat_μ→grad_μ bypass, n_e_steps {1,3,5} |
| 15 | Regime-II trainability/holonomy (A4) | 2 | 4 | 2 | ★★ | Train W/M from flat, holonomy vs ‖connection‖ |
| 16 | VFE-internal gauge scaling (F3) | 4 | 2 | 4 | ★★ | route_data_budget, gauge-on vs off, fit β |
| 17 | χ=1/α transfer (G1) | 3 | 3 | 4 | ★★ | α sweep, field on μ, χ=dm/db @ converged E-step |
| 18 | Symplectic E-step (G3) | 3 | 3 | 4 | ★★ | Leapfrog E-step, DeGroot/overshoot sanity + law |
| 19 | Sanov LDP rate (G2) | 3 | 3 | 4 | ★★ | Per-token F-density rate, FR-volume tracking |
| 20 | SPD chart / sigma_max (E1) | 2 | 5 | 2 | ★★ | Congruence probe + read guard_sigma_ceil_frac |
| 21 | Mean-field full-vs-diag F-gap (C4) | 2 | 3 | 3 | ★★ | K=4 dense-Ω E-step, ΔF_red vs off-diag mass |
| 22 | lambda_h shrinkage (C3) | 3 | 4 | 3 | ★★ | Arm A/B, rare-token PPL stratified |
| 23 | bf16 transport exposure (E2) | 2 | 4 | 2 | ★ | None/bf16/fp16 matched runs @K∈{32,160} |
| 24 | Mean trust region (E3) | 2 | 4 | 3 | ★ | Whitened-step log, NaN rate @large embed_dim |
| 25 | y3 fit-window stability (F4) | 3 | 2 | 3 | ★ | Flat-path g2 community depth-scaling |
| 26 | Frustration Laplacian gap (G4) | 3 | 3 | 3 | ★ | Validated consensus-Laplacian on frustrated input |
| 27 | Holonomy-span locality (A4b) | 2 | 4 | 3 | ★ | Spearman(per_triple, span) on regime_ii ckpt |
| 28 | Syntactic gauge metric (J1) | 4 | 2 | 4 | ★ | S(Ω)/d_AI vs parse distance (needs data pipeline) |

## Experiment plans (top 8)

### EXP-1. Multi-seed variance floor (I1) — ★★★★★

The discipline gate for every other ablation: nothing else's "win" is interpretable without it.

**Status (2026-06-22): built.** Across-seed aggregator (`multiseed_analysis.aggregate_seed_metric`), the fixed data-order generator (`DATA_SEED`), and the `ppl_noise_band` figure (per-seed PPL + mean±SD band, auto-emitted by `multiseed_analysis.py`) are in. See `docs/2026-06-22-edits.md`.

- **Config.** Baseline `train_vfe3.py`: `embed_dim=20`, `n_heads=2`, `n_layers=1`, `max_steps=15000`, `gauge_group='block_glk'`, `use_head_mixer=True`, `lambda_h=0.25`. Set `NUM_RUNS=5`, `SEEDS=[6,64,23,3,17]`.
- **Baseline/control.** The five seeds are the experiment; no separate control.
- **Primary metric.** Across-seed mean and SD of `test_ppl` from the five `test_results.json`. **Secondary.** Read the existing single-seed `m_p_mu_lr`/`m_phi_lr`/`lambda_h`/`kappa_beta` grids and flag every cell whose between-cell spread < 2 SD.
- **Test/thresholds.** Primary claim: across-seed SD ≥ 1.0% of mean (1.5% pre-registered threshold; report measured value regardless). Null: SD < 0.5% of mean. Mandatory disclosure: the post-build reseed fixes data order, so this SD is init+optimization only and a lower bound on deployment variance — add a 2-seed longer-budget spot check.
- **Seeds.** 5 + 2 spot-check.
- **Deliverable.** A noise-band overlay figure a referee can read, plus the corrected error bar on the headline K=20 PPL.
- **Compute.** ~1–2 GPU-hours.

### EXP-2. Gauge ON vs OFF vs frozen-random (A1) — ★★★★★

The single experiment that converts the program's central causal claim from hedged conjecture to measured result, in either direction.

**Status (2026-06-22): built + run.** The `gauge_transport` sweep arm (3 cells × L{1,2}, `collect_diagnostics`) is registered in `ablation.py` (commit 31cdae5 + the readiness build); the user has run the ON/OFF/frozen ablation; the `gauge_transport_bars` grouped-bar figure (val PPL by mode×depth, max|Ω−I| annotated) now auto-emits per sweep. See `docs/2026-06-22-edits.md`.

- **Config.** New `ablation.py` SWEEPS `'gauge_transport'` multi-arm entry, all sharing matched `phi_weight_decay`, `lambda_alpha_mode`, `precision_weighted_attention`, and `use_head_mixer=False`:
  - **ON:** `{use_head_mixer:False}` (baseline `phi_scale=0.06`, `m_phi_lr=0.015`).
  - **OFF (Ω=I):** `{phi_scale:0.0, pos_phi:'none', e_phi_lr:0.0, m_phi_lr:0.0, use_head_mixer:False}`.
  - **FROZEN-random:** `{e_phi_lr:0.0, m_phi_lr:0.0, use_head_mixer:False}` at baseline `phi_scale=0.06`.
  - Run each at `n_layers∈{1,2}` (depth arm: rank-collapse cannot manifest at L=1).
- **Baseline/control.** OFF is the null control; FROZEN isolates "having a frame" from "learning a frame." Match parameter counts: φ_embed is allocated and frozen-at-zero (identical param count); the only asymmetry is `pos_phi='none'` dropping `pos_phi_free`, so set `pos_phi='none'` in the ON cell too (or keep `pos_phi='learned'`+`m_phi_lr=0` in OFF).
- **Primary metric.** `primary_val_ppl` from `sweep_results.csv`. **Secondary.** `metrics.gauge_equivariance_residual` (confirm OFF Ω is flat to float-eps) and matched param counts.
- **Test/thresholds.** Decision rule: non-overlapping ±1 SD bands at ≥3 seeds. Predict ON < FROZEN ≤ OFF at L=2. Null: ON and OFF ±1 SD bands overlap once `phi_weight_decay` matched — then GL(K) transport is inert for single-layer LM perplexity and the advantage is owned by diagonal-Gaussian KL attention + PriorBank.
- **Seeds.** [6,64,23] × 3 cells × 2 depths = 18 runs.
- **Caveat to report.** Forcing `use_head_mixer=False` moves ON off the production operating point, so the result certifies flat-GL(K)-transport-vs-none, not the shipped model.
- **Deliverable.** The gauge-ablation bar figure (val PPL, three cells × two depths, error bars) + residual table.
- **Compute.** ~4–8 GPU-hours.

### EXP-3. Σ_q as calibrated Fisher uncertainty (B1) — ★★★★★

Tests open thread (f) and the manuscript's flagged "uncertainty channel" over-read, nearly for free.

**Status (2026-06-22): built.** `belief_ce_bank` (the Σ↔CE join, replaying forward's belief path incl. the s-refine anchor + precision fold), the reliability / Σ-stratified-error / Σ-CE-scatter figures (wired into `generate_figures`), and `sigma_ce_spearman` / `sigma_trace_cv` / the CV>0.10 gate (in `research.json`) are all in (commit 3acd7ad). See `docs/2026-06-22-edits.md`.

- **Config.** Default trained checkpoint: `family='gaussian_diagonal'`, `n_e_steps=1`, `e_q_sigma_lr=0.015`, `sigma_init=1.0`, `sigma_max=10.0`, `use_prior_bank=False` (clean isolation — σ does not touch decode), `decode_precision_scaled=False`. Rescue arm: retrain at `n_e_steps≥4` and/or `family='gaussian_full'`.
- **Baseline/control.** Uncalibrated decode-softmax confidence from `run_artifacts._calibration_and_strata`.
- **Primary metric.** Spearman ρ(tr(Σ_q), per-token CE nats) on held-out tokens (valid mask `targets!=-100`); relative ECE change from a 1-parameter Σ-conditioned temperature recalibration (15-bin). **Pre-check (near-free):** `fisher_trace_mean/median` are already in the per-run CSV — screen whether σ varies before any new code.
- **Pre-registered gate.** Require across-token CV of tr(Σ_q) > 0.10; if not, report "covariance inert (CV<0.10)" and STOP — do not miscode as "decode doesn't matter."
- **Test/thresholds.** Pass the gate, then predict ρ ≥ 0.2 and ≥20% relative ECE reduction. Null: IQR spread < 10% of median, OR |ρ| < 0.05, OR < 5% relative ECE change.
- **Seeds.** Post-hoc on one checkpoint; 1 retrain for the rescue arm.
- **Deliverable.** Reliability diagram + tr(Σ_q)-stratified error curve + the ρ scatter.
- **Compute.** < 3 GPU-hours.

### EXP-4. Canonical-F vs entropy-suppressed surrogate (C1) — ★★★★★

The production kernel never computes the entropy term, so CANON_ORACLE is the only path that has ever exercised the canonical gradient — this raises the stakes above a second-order correction.

**Status (2026-06-22): built.** `attention_entropy_cov_gap` (`vfe3/viz/extract.py`) measures the −τ⁻¹Cov_β(E,∇E) gap by differencing the autograd oracle gradient (entropy ON vs OFF) on the converged belief; the `attention_entropy` sweep is the 2×2 entropy×κ grid with a `cov_gap` CSV column; the `entropy_ppl_gap` + `cov_gap_vs_kappa` figures are wired. See `docs/2026-06-22-edits.md`.

- **Config.** `ablation.py` `configs` multi-arm, K=20, 15k steps:
  - **SURROGATE:** `{include_attention_entropy:False, oracle_unroll_grad:True}`.
  - **CANON_ORACLE:** `{include_attention_entropy:True, oracle_unroll_grad:True}` (the clean control; both on the oracle, isolating the entropy term from the kernel-vs-oracle route).
  - Companion low-κ arm `kappa_beta=0.25` for both (Cov_β scales with attention diffuseness; guards against a sharp-attention null that does not generalize).
- **Baseline/control.** CANON_ORACLE is the control; SURROGATE vs CANON_ORACLE is the single-variable contrast. Force `oracle_unroll_grad=True` on every arm (`__post_init__` does not auto-enable it for entropy=False).
- **Primary metric.** `primary_val_ppl`. **Secondary.** mean H(β); directly measure the −τ⁻¹Cov_β gap by differencing kernel grad and autograd-of-(β·E) grad on the same belief snapshot.
- **Test/thresholds.** Predict SURROGATE degrades PPL relative to CANON_ORACLE at κ=1; null: same `primary_val_ppl` within ±1 SD across 3 seeds. Report at iso-optimizer-step (oracle is ~2x slower); wall-clock separately.
- **Seeds.** [6,64,23] × 2 arms × 2 κ = 12 runs.
- **Deliverable.** PPL-gap table (entropy on/off × κ) + the measured covariance-gap magnitude.
- **Compute.** ~6–10 GPU-hours.

### EXP-5. Structural non-Neal-Hinton EM (C2) — ★★★★

Pre-built registered sweep; the theory is settled so the value is the unrun PPL-vs-n_e_steps curve and the F-vs-CE decorrelation the manuscript does not itself assert.

**Status (2026-06-22): built.** The converged final E-step F/token is persisted by `finalize_run` (`estep_final_f_per_token`); `scaling_analysis.py` emits the `f_ce_decorrelation` + `estep_capacity` figures over the `infer_T` n_e_steps arms and prints Pearson(n_e_steps, F) and Pearson(F, CE). See `docs/2026-06-22-edits.md`.

- **Config.** K=64, single layer, `e_step_gradient` PINNED to `'unroll'` (and a second arm at `'straight_through'` to remove the deepening-graph confound), `e_phi_lr=0.0`, `gradient_mode='filtering'`. Sweep `n_e_steps∈{1,2,3,5,8}`. `KMP_DUPLICATE_LIB_OK` already set.
- **Baseline/control.** The `n_e_steps=1` cell is the operating point; the straight_through arm controls trajectory-graph depth.
- **Primary metric.** test PPL vs n_e_steps (monotonicity test). **Secondary.** Pearson(n_e_steps, final F/token) and Pearson(final F/token, CE) across the sweep, from `return_trajectory=True`.
- **Test/thresholds.** Predict non-monotone PPL, best at {1,2}, PPL(5) ≥ PPL(1) within ±15%, Pearson(n_e_steps, F) < −0.8 while Pearson(F, CE) ≥ 0. Null: PPL decreases monotonically AND Pearson(F, CE) < −0.5 (the bound serves the likelihood, contradicting the structural-EM separation).
- **Seeds.** 3 seeds × 5 settings × 2 gradient modes.
- **Deliverable.** PPL-vs-n_e_steps curve + the F-CE decorrelation scatter.
- **Compute.** ~15–25 GPU-hours.

### EXP-6. μP width-stability of the inverse-K exponent (F1) — ★★★★

Decides whether the headline scaling exponent is a capacity law or partly optimization mis-tuning.

**Status (2026-06-22): built.** The `grow_K_mup` route (per-K kl_max + μP LR/init scaling) exists; `scaling_analysis` now carries `embed_dim` + test PPL and auto-emits `kmup_stability` (grow_K vs grow_K_mup on the K axis, offset power-law fit, b annotated). See `docs/2026-06-22-edits.md`.

- **Config.** Add `route_grow_k_mup` to `scaling.py` mirroring `route_grow_k`, anchored at K=20: scale `e_q_mu_lr`, `m_p_mu_lr` ~ (20/K) and `mu_init_std` ~ √(20/K). **Mandatory in BOTH arms:** recompute `kl_max = 8*embed_dim` per cell (the baseline freezes it at 160 from embed_dim=20 — a width confound that zeros the hyper-prior self-gradient near K≈126, i.e. at K=120). K∈{20,40,80,120}, iso-token.
- **Baseline/control.** Width-fixed `route_grow_k` (with kl_max-per-K corrected) is the control. Secondary: a direct LR grid at K=20 and K=120 to separate "μP rule mis-specified for the preconditioned E-step" from "no width effect."
- **Primary metric.** Fitted exponent b from PPL=aK^b+c, fit against `embed_dim` (not `n_params`, which is K²-dominated). **Secondary.** per-K test-CE curve flatness at K≥80.
- **Test/thresholds.** Predict |Δb| ≥ 0.15 between arms and large-K flattening under μP. Null: b agrees within bootstrap CI with kl_max matched — then the Σ-preconditioned E-step + AdamW already deliver width-stability and b is a genuine capacity law.
- **Seeds.** [6,64] × 4 K × 2 routes = 16 runs.
- **Deliverable.** Two scaling curves (b_fixed vs b_μP) with bootstrap CIs.
- **Compute.** ~15–30 GPU-hours.

### EXP-7. Prior-anchoring resists rank collapse (F2) — ★★★★

Tests whether α·KL(q‖p) is the FFN-brake substitute the no-MLP claim relies on; the directional null is already confirmed in a probe.

**Status (2026-06-22): built.** The `rho_handoff` 5-arm sweep (lambda_alpha × rho × e_phi_lr, n_layers=4, lambda_alpha_mode='constant'), the per-cell `rank_resid` / `rank_resid_by_layer` diagnostics, and the `rank_residual_by_depth` depth-overlay figure (`_plot_rank_collapse` driver) are in (commit 3acd7ad). See `docs/2026-06-22-edits.md`.

- **Config.** Trained prior tables. Sweep `n_layers∈{1,2,4,8}`, pure no-MLP path. Arms: (a) anchored `lambda_alpha=1` with both `prior_handoff_rho=1` (previous-layer handoff) and `rho=0` (embedding anchor) sub-arms; (b) no-anchor `lambda_alpha≈1e-3, rho=0`. Run a second pair with `e_phi_lr>0` so transport is genuinely per-layer-independent (default `e_phi_lr=0` freezes Ω across layers). `KMP_DUPLICATE_LIB_OK` for n_e_steps>1.
- **Baseline/control.** No-anchor arm is the collapse control; the e_phi_lr>0 pair controls fixed-vs-per-layer geometry.
- **Primary metric.** New Dong rank-one residual r(X)=‖X−1xᵀ‖_F/‖X‖_F on converged per-layer μ (~3-line add; existing `effective_rank` is the wrong object — spectral rank of per-token Σ). **Secondary.** per-layer decay rate.
- **Test/thresholds.** Directional: no-anchor decays faster per layer than anchored at matched depth, statistically separated across 2 seeds. Drop the absolute ≥0.3/<0.05 thresholds (the no-anchor control plateaus at r~0.055, it does not collapse to rank-1 — Dong's theorem assumes symmetric-softmax self-attention). Null: indistinguishable per-layer decay rates, OR both stay full-rank. Frame the Dong mapping as qualitative.
- **Seeds.** 2 seeds × 4 depths × ~3–4 arms.
- **Deliverable.** r(X)-vs-depth curves per arm.
- **Compute.** ~25–40 GPU-hours.

### EXP-8. Pullback natural-gradient gauge M-step + LR scaling (D1) — ★★★★

Three lenses (S9/S12/S14) converged here. Confirms or refutes the geometric justification for the `m_phi_natural_grad` lever and tests the natural-gradient LR-mis-scaling pitfall in one panel.

**Status (2026-06-22): built.** The gauge M-step sweeps exist; `GaugeNaturalGradAdamW` now stashes gated (log/eval-only, read-only) `cos_nat_phi` + pullback-metric `pullback_cond_*`, `train.py` logs a cumulative `wall_clock_s`, and `wallclock_convergence` (driven by `_plot_wallclock_convergence`) plots val PPL vs wall time with steps/wall-to-target annotated; the LR bowl falls out of `_plot_one_sweep` on `m_phi_lr_natgrad`. See `docs/2026-06-22-edits.md`.

- **Config.** `block_glk`, K=64, `n_heads=8` (d_head=8 ≤ max_k=12, so `pullback_per_block` builds), `m_gauge_momentum=0.9`, `e_phi_lr=0` (keep the preconditioner off the E-step). Arms: AdamW-on-φ (`m_phi_natural_grad=False`), `pullback_per_block`, and `killing_per_block` (conformal control). **Regime knob: sweep `mass_phi∈{0.0, positive}`** — NOT `phi_weight_decay` (hard-zeroed under natural-grad and pinned by `test_phi_weight_decay.py`). LR sub-experiment: log-spaced `m_phi_lr∈{0.0005,0.0015,0.005,0.015,0.05,0.15}` on the pullback arm (add `requires:{m_phi_natural_grad:True, phi_precond_mode:'pullback_per_block'}` to the SWEEPS entry).
- **Baseline/control.** AdamW-on-φ; killing is the conformal LR-rescale control (cos(nat,grad)=1, must NOT serve as the geometric arm).
- **Primary metric.** steps-to-target-val-PPL and final PPL, both per-step and per-wall-clock (the per-token matrix_exp solve is the dominant added cost). **Secondary.** per-token pullback condition number (logged), cos(nat,grad).
- **Test/thresholds.** Predict pullback reaches target PPL in ≥15% fewer steps at `mass_phi=0` (frame grows, ad_φ spreads, Ψ departs from I), gap closes at positive `mass_phi`; killing matches AdamW within 3%. LR sub-experiment: natural-grad optimum ≥2x from 0.015, matching AdamW within ±3% at its own optimum. Null: pullback ≤ AdamW per-step, OR no per-wall-clock advantage once the solve is counted, OR killing = pullback.
- **Seeds.** 2 seeds × 3 arms + 6-point LR sweep.
- **Deliverable.** Per-step and per-wall-clock convergence curves + the LR-vs-PPL bowl.
- **Compute.** ~10–15 GPU-hours.

## Do-NOT-pursue appendix

These were killed in adversarial review; preserved so the reasoning is not re-litigated.

**Envelope-cancellation broken off-equilibrium at n_e_steps=1 under state-dependent α (kernel coef ≠ autograd-of-F).** The prediction was crisp and the experiment was seconds of CPU, but the central empirical claim — a >2x kernel-vs-oracle gradient gap on the self-coupling leg at the default operating point — does not survive contact with the code: under the default `n_e_steps=1` with q initialized to p, the self-coupling gradient is identically zero, so there is no gap to measure. The "gap" is an artifact of evaluating off the operating point the model actually runs.

**kl_max=8·K hard-clamp moves but does not remove the gradient cliff; residual binding from pairwise/hyper-prior/full-cov saturation.** Every mechanical claim was verified correct against `families/base.py` (`safe_kl_clamp` is a hard `clamp(0,kl_max).nan_to_num(kl_max)`), but the finding was already applied and recorded in the verified ledger (the kl_max=8·embed_dim fix). Re-raising it re-proposes settled, applied work rather than opening anything new.

**Regime-II curvature is the only source of belief-transport path-dependence.** The implementation-lens reading of the code, tests, ledger, and manuscript shows the central claim is settled, not open: vertex-frame holonomy telescoping to I is a proven identity with passing unit tests, and the Route-A-vs-Route-B covariance asymmetry is likewise pinned. Only the empirical trainability curve survives, and it is captured under A4 (S35) — the path-dependence framing itself is closed.

**S37's MCMC posterior-bias comparison (mean-field variance collapse vs NUTS ground truth).** Untestable against the actual runtime: vfe3's E-step has no live observation-likelihood channel (`log_likelihood` is a dead stub off the kernel/oracle descent path), so q* is the fixed point of a target-blind functional, not a Bayesian posterior p(k,z|data). Comparing its tr(Σ_q) to a NUTS posterior of a generative model the code never ingests compares two different objects — a category error. Only the likelihood-free full-vs-diagonal F_red gap (retained as C4) is defensible.

**S22's `project_phi_to_slk` isometry control and S33's first-order chart-equivalence mechanism.** Both shipped with a wrong operator/mechanism: `project_phi_to_slk` removes only the per-block trace (det Ω=1), leaving symmetric-traceless directions that keep S(Ω)>0, so it is not the orthogonal projector the syntactic-metric probe needs; and on the diagonal cone `spd_affine` whitens by 1/σ while `log_euclidean` does not, so the two charts diverge as |log σ| everywhere, independent of the `sigma_max` ceiling. The experiments survive only with corrected mechanisms (a real polar/symmetric-part projector; the gaussian_full cone to isolate clamp-driven from whitening-driven divergence), as noted in J1 and E1.
