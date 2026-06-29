# Metrics and Figures Documentation-Gap Investigation

Date: 2026-06-28.

Scope: read-only investigation of the current `origin/main` codebase at
`5e88afc941e4b08f96b2b057ad6c1cb41a0fdfa1`, using the clean worktree
`C:\tmp\V3_Transformer_metrics_figures_20260628`. The live checkout was dirty,
so the source investigation was performed in the clean worktree and this report
was copied back as a docs-only artifact. No production source files were edited,
no experiments were run, and no pytest run was executed.

Five read-only investigators examined independent evidence domains: the training
artifact path, ablation and scaling machinery, the visualization registry, the
mathematical diagnostics, and documentation/manuscript coverage. I then checked
the conflicting claims directly against source. The most important correction is
that the attention-entropy covariance-gap metric is not missing: it is produced
by `vfe3/viz/extract.py:530` and wired into the `attention_entropy` ablation at
`ablation.py:1402`. The gap is that this metric is sweep-local and not treated
as a routine reported diagnostic.

## Executive Finding

The codebase already computes far more scientifically useful diagnostics than
the current README, docs, digest, and manuscript-facing materials document. The
underreported surface is not another PPL chart. It is the set of diagnostics
that make the VFE claim falsifiable: whether covariance uncertainty is
calibrated, whether free-energy terms move coherently with held-out CE, whether
the gauge/SPD/Fisher geometry remains numerically meaningful, whether inference
iterations actually buy predictive accuracy, whether positional and attention
sanity checks remain stable over validation, and whether reported ablations are
aggregated with seed uncertainty rather than read as single-seed point wins.

## Include First

### 1. Belief-Uncertainty and Calibration Package

This should be a first-class report and manuscript artifact. The model's
distinctive object is a Gaussian belief tuple `(mu, Sigma, phi)`, so the
covariance must be shown to carry predictive uncertainty rather than acting as
decorative state.

The code already saves the scalar and binned evidence in `research.json`.
`vfe3/run_artifacts.py:367` computes decode calibration and frequency-stratified
CE, returning `ece`, `reliability`, `overall_ce`, and `freq_strata_ce`.
`vfe3/run_artifacts.py:471` writes research artifacts, including
`fd_gradient_worst_rel_error`, `sigma_ce_spearman`, `sigma_trace_cv`, and
`sigma_trace_cv_gate_pass` at `vfe3/run_artifacts.py:490` and
`vfe3/run_artifacts.py:503`. `vfe3/viz/report.py:303` emits
`reliability_diagram`, `sigma_stratified_error`, and `sigma_ce_scatter`.

The documentation gap is that these are treated as generated extras rather than
standard evidence. The 2026-06-27 digest explicitly says B1 Sigma calibration
produced no artifact, while the current code path can produce the B1 figure
family and scalar summaries. Future run reports should include at least ECE,
rare/mid/frequent CE, `sigma_ce_spearman`, `sigma_trace_cv`, the reliability
diagram, the Sigma-stratified CE curve, and the Sigma-CE scatter. For manuscript
use, report them only after the relevant run actually produces `research.json`;
do not infer them from code availability.

### 2. Gauge, SPD, and Fisher Geometry Health

The code records a rich geometry-health layer, but the standard artifact set
plots only a narrow slice of it. `vfe3/model/model.py:1417` records holonomy
deviation, bootstrap confidence interval endpoints, and Wilson holonomy.
`vfe3/model/model.py:1429` records group-correct gauge invariants.
`vfe3/model/model.py:1477` records effective-rank quantiles.
`vfe3/model/model.py:1480` records Fisher trace. `vfe3/model/model.py:1484`
records guard saturation and nonfinite fractions. `vfe3/model/model.py:1502`
records attention entropy collapse. These values are copied to `metrics.csv` by
the logging path at `vfe3/train.py:848`.

The current automatic history plots in `vfe3/run_artifacts.py:763` cover
`holonomy_deviation` and `gauge_trace_spread`, and the single-run report emits
final-state gauge and numerical-trust figures through `vfe3/viz/report.py:223`
and `vfe3/viz/report.py:248`. That is not enough for a paper figure set. Add
or standardize history plots for `holonomy_wilson`, `cocycle_residual`,
`gauge_invariant_spread`, `phi_norm_mean/std`, `belief_cond_p95/max`,
`fisher_trace_mean`, `guard_sigma_floor_frac`, `guard_sigma_ceil_frac`,
`guard_energy_klmax_frac`, `nonfinite_frac`, `renyi_band_frac`,
`eff_rank_p5/median/p95`, and `attn_entropy_collapsed_heads`.

The more important implementation gap is held-out geometry. `_val_diagnostics`
currently emits validation attention, E-step, and positional probes at
`vfe3/train.py:494`, but not the full gauge/SPD/Fisher set. To support claims
about learned geometry rather than training-batch artifacts, add validation
columns for the same geometry-health quantities, for example
`val_holonomy_wilson`, `val_cocycle_residual`, `val_gauge_invariant_spread`,
`val_fisher_trace_mean`, `val_belief_cond_p95`, guard fractions, and
`val_phi_norm_mean/std`.

### 3. E-Step Quality and Inference-Capacity Readouts

The model's forward pass is iterative inference, so every report should include
direct evidence for what the E-step contributes. The code already logs more than
the currently documented artifact set surfaces. `vfe3/train.py:553` logs
`estep_f_nondecreasing_frac`, and `vfe3/train.py:555` logs final E-step
residuals for `mu`, `sigma`, and `phi`. Only `estep_f_drop` receives a standard
history plot at `vfe3/run_artifacts.py:830`. End-of-run capacity metrics are
saved in JSON: `test_ce_no_estep` and `estep_capacity_gain` at
`vfe3/run_artifacts.py:643`, and `estep_final_f_per_token` at
`vfe3/run_artifacts.py:663`.

Include an E-step panel with `estep_f_drop`, `estep_f_nondecreasing_frac`,
`estep_r_mu_last`, `estep_r_sigma_last`, `estep_r_phi_last`, and the
end-of-run `test_ce_no_estep` versus `test_ce` capacity gain. For sweep-level
analysis, the existing `plot_estep_capacity` and `plot_f_ce_decorrelation`
paths in `scaling_analysis.py` should be named in reports, not left as implied
side effects.

### 4. Free-Energy Decomposition and Closure Notes

The free-energy figures are among the best conceptual artifacts, but they are
not consistently documented as required outputs. `vfe3/run_artifacts.py:849`
selects free-energy history columns and emits `free_energy_decomposition.png`
and `free_energy_codescent.png` at `vfe3/run_artifacts.py:869` and
`vfe3/run_artifacts.py:873`. The figure functions are registered at
`vfe3/viz/figures.py:508`, `vfe3/viz/figures.py:602`, and
`vfe3/viz/figures.py:655`.

These should be included in every serious run report, with one explicit caption
qualification: the plotted free-energy blocks and `val_ce` are not all the same
closed functional unless the data term and model-channel weighting are stated.
The caption should say whether `free_energy_total` excludes the observation
term, whether `val_ce` is an external held-out observation-loss proxy, and
whether model-channel terms such as `hyper_prior_weighted`, `gamma_coupling`,
and `gamma_meta_entropy` are active.

### 5. Generalization, Positional Loss, and Attention Sanity Time Series

The validation path computes a useful group of sanity metrics that are largely
CSV-only. `vfe3/train.py:530` derives attention maps on validation batches.
`vfe3/train.py:535` logs causal leakage and row-sum error. `vfe3/train.py:537`
logs positional-content score. `vfe3/train.py:538` records previous-token mass,
period-match mass, and head redundancy. `vfe3/train.py:569` records
position-stratified CE and `pos_loss_ratio`. `vfe3/train.py:843` logs
`generalization_gap`.

These should be a compact "validation sanity" figure family. Include
`generalization_gap`, `pos_loss_first_q`, `pos_loss_last_q`, `pos_loss_ratio`,
`val_future_leakage`, `val_row_sum_error`, `val_pos_content_r2`,
`val_prev_token_mass`, `val_period_match_mass`, and
`val_head_redundancy_js`. They are particularly relevant because the manuscript
and digest discuss positional extrapolation, causal priors, and head behavior,
but the routine training artifact path does not surface their time evolution.

### 6. Natural-Gradient and Optimizer-Geometry Diagnostics

The code already records raw gradient decomposition and some natural-gradient
geometry, but reporting focuses mostly on PPL. `vfe3/train.py:382` records
global and per-role M-step gradient norms. `vfe3/train.py:392` records E-step
gradient norms. `vfe3/run_artifacts.py:805` and `vfe3/run_artifacts.py:820`
emit M-step and E-step gradient-decomposition figures when those columns exist.
Natural-gradient optimizer diagnostics are recorded at `vfe3/train.py:883`:
`cos_nat_phi`, `pullback_cond_median`, and `pullback_cond_max`.

For reports, include the two gradient-decomposition figures plus an
optimizer-geometry table or plot for `cos_nat_phi`, pullback condition, role
weight norms, and update-to-weight ratios derivable from `grad_norm_*`,
`weight_norm_*`, and the learning rates. The missing metric family is actual
preconditioned step size: add `estep_nat_grad_norm_mu/sigma`, raw-to-natural norm
ratios, trust-region clamp fractions, and actual retraction/update norms if the
goal is to argue that optimization is information-geometric rather than merely
Adam-like.

### 7. Scaling and Compute-Frontier Artifacts

The scaling path emits and harvests compute data, but the documentation does not
name several core files and metric keys. `vfe3/run_artifacts.py:514` creates
cost-model fields. `vfe3/run_artifacts.py:572` records analytic FLOP proxies,
active parameters per token, and decode/E-step FLOPs per token.
`vfe3/run_artifacts.py:581` records wall time per token and per step.
`scaling_analysis.py:405` emits `scaling_ce_vs_params.png`;
`scaling_analysis.py:409` emits `scaling_routes_overlay.png`;
`scaling_analysis.py:415` emits `scaling_ce_vs_flops.png`;
`scaling_analysis.py:421` emits `scaling_ce_vs_tokens.png`;
`scaling_analysis.py:442` emits `inference_capacity.png`.

Two additions should be made. First, `scaling_analysis.py` should persist a
`scaling_summary.json` and `SCALING_ANALYSIS.md`; the pooled fits, bootstrap
confidence intervals, per-route exponents, and route-comparison tests are too
important to live only in console output. Second, because the June 27 result is
reported as an offset law in PPL against `embed_dim`, add a
`ppl_vs_embed_dim_offset.png` figure. The current analyzer fits CE versus
parameter count; `embed_dim`, `test_ppl`, and per-seed PPL aggregates are
already harvested, so this is cheap and would align the generated figure with
the result people actually cite.

### 8. Multi-Seed Ablation Aggregation

The ablation machinery supports per-sweep seed lists, but the report layer is
not yet a statistical endpoint. Per-seed rows are written to `sweep_results.csv`
through `ablation.py:1600`, and seed-aware cell names are created in the sweep
runner. However, `analyze_sweep` still reads and prints individual rows, and
some special plotters treat labels as if no `__s<seed>` suffix exists.

Before interpreting any sub-3 percent effect, group rows by base label, then
report `n`, mean, SD, CV, and an error-bar figure. This matters most for
`gauge_transport`, `attention_entropy`, `fisher_mu_precond`, and any future
optimizer-geometry sweep. The June 27 digest correctly uses the scaling run's
noise floor to discount small single-seed effects; the next ablation report
should make this aggregation native rather than advisory.

### 9. Vocabulary Probability and Decode-Readout Figures

The single-run and cross-run vocabulary figures are implemented but largely
undocumented. `vfe3/viz/report.py:288` emits `vocab_probability_heatmap`,
`vocab_calibration`, `vocab_confusion`, and `decode_readout`.
`vfe3/viz/report.py:407` emits the comparison versions, and
`compare_vocab_figures.py` is the click-to-run wrapper.

These figures are worth documenting because they connect the abstract belief
state to actual next-token behavior, expose frequency/category failures, and
separate decode calibration from covariance calibration. Include them in reports
for any claim about prior-bank decode, linear decode, vocabulary-tail behavior,
or category-level separation. They are less central than the geometry and
calibration packages, but much more informative than another raw loss curve.

### 10. Pure-Path Certificate

The project requires a theoretically pure path under appropriate toggles, but
there is no single artifact that certifies whether a run is on that path. Add a
small `pure_path_report.json` or table that states the relevant config toggles
and stress metrics: canonical attention entropy enabled, `lambda_beta` status,
transport mode, head mixer status, non-flat connection status, prior-bank versus
linear decode, precision-weighted attention status, guard saturation fractions,
nonfinite fraction, cocycle residual, and gauge residual. This should be
reported as a path certificate, not as a judgment that the current defaults are
wrong.

## Document, But Do Not Overweight

The visualization registry contains low-level helpers that are tested but do
not automatically deserve manuscript space: `plot_embedding`,
`plot_attention_graph`, `plot_attention_grid`, `plot_covariance_ellipses`, and
generic `plot_trajectory`. They should be listed in a figure manifest or API
reference, not promoted as paper evidence by default. The hard documentation
issue is that `vfe3/viz/figures.py:443` references
`docs/superpowers/specs/2026-06-04-publication-figures-design.md`, which is not
present in the clean worktree. Either restore that spec or update the pointer.

Several ablation figures are already documented in current docs and should not
be reclassified as undocumented: `pos_extrapolation`, `renyi_saturation`,
`mu_precond`, `holonomy_trainability`, `estep_capacity`,
`f_ce_decorrelation`, `kmup_stability`, `ppl_noise_band`,
`reliability_diagram`, `sigma_stratified_error`, and `sigma_ce_scatter`.
The gap is not their existence; it is that the report layer does not yet provide
a unified "expected artifacts" manifest and does not consistently carry these
figures into manuscript-facing summaries.

## Recommended Reporting Bundle

For a completed serious run, include:

1. Performance: train CE, validation PPL, test CE/PPL/BPC, generalization gap,
   tokens/s, peak memory, wall time per token, and scaling-point metadata.
2. Free energy: decomposition, co-descent with held-out CE, model-channel terms
   when active, and a caption that states the functional closure.
3. Inference: E-step drop, nondecreasing fraction, residuals, no-E-step test CE,
   E-step capacity gain, and final E-step F/token.
4. Geometry: holonomy deviation and Wilson holonomy, cocycle residual, gauge
   invariant spread, phi norm, SPD condition quantiles, Fisher trace, guard
   saturation, nonfinite fraction, and effective-rank quantiles.
5. Calibration: ECE, reliability diagram, frequency-stratified CE,
   Sigma-CE Spearman, Sigma trace CV, Sigma-stratified CE, and Sigma-CE scatter.
6. Attention and position: causal leakage, row-sum error, positional-content
   score, previous-token mass, period-match mass, head redundancy, positional
   loss ratio, attention entropy minimum, and collapsed-head count.
7. Optimizer geometry: M-step and E-step gradient decompositions, role weight
   norms, `cos_nat_phi`, pullback condition, update-to-weight ratios, and
   preconditioned step norms once logged.
8. Ablations and scaling: seed-grouped mean/SD/CV tables, error-bar plots,
   PPL-vs-embed-dim offset scaling, CE-vs-params/compute/data scaling, and a
   saved summary artifact rather than console-only fit results.
9. Decode behavior: vocabulary probability heatmap, vocabulary calibration,
   vocabulary confusion, decode-readout plots, and cross-run comparison plots
   when comparing decode regimes.

This bundle is the shortest defensible answer to "what should we include but
currently are not documenting?" It treats perplexity as the headline outcome,
but it treats geometry, calibration, inference, and compute as the evidence that
the headline means what the theory says it means.
