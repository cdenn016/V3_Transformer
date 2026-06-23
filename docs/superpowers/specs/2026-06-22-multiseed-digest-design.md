# Multi-seed run digest: full cross-seed aggregation + figure set

Date: 2026-06-22
Branch: `feat/multiseed-digest` (from `main`)

## Problem

`multiseed_analysis.py` was built for the EXP-1 variance floor: it aggregates ONE scalar
(`test_ppl`) from per-seed `summary.json` and emits ONE figure. A completed 5-seed identical-config
run (`vfe3_runs/K=20_GL(10)`) has far more per-seed data than that tool digests, and three concrete
defects block even the headline use:

1. **Wrong seed source.** It reads the seed from `config.json`, which stores `seed: null`; the real
   seed lives in `provenance.json` (`"seed": 54`). Every seed label comes back `None`.
2. **`run_root` does not resolve the run folder.** The default `"vfe3_runs"` rglobs *every* run
   (K20, K160, ...) into one mixed aggregate; the user's WIP sets `"K=20_GL(10)"`, a bare name that
   does not exist relative to the repo root.
3. **One metric, one figure.** Nothing aggregates the per-seed training curves
   (`metrics.csv`, 450 steps x ~95 cols), per-layer diagnostics (`metrics_per_layer.csv`), or
   research scalars (`research.json`: ECE, frequency-stratified CE, sigma-trace CV, ...).

## Goal

`python multiseed_analysis.py` with `CONFIG["run_root"]="K=20_GL(10)"` resolves the run folder,
aggregates every per-seed artifact across seeds, and writes the full figure + data set in one pass.

## Design

### `multiseed_analysis.py` (aggregation + driver)

- `_resolve_run_root(run_root)`: return it if it exists, else `vfe3_runs/<run_root>` if that exists,
  else the literal (downstream reports "nothing found"). Honors the user's bare `"K=20_GL(10)"`.
- `_seed_for(run_dir)`: `provenance.json["seed"]` -> `config.json["seed"]` -> regex `_s(\d+)$` on the
  dir name. Used everywhere a seed is labeled.
- Keep `aggregate_seed_metric` (summary.json scalar; backward-compatible) but route its seed read
  through `_seed_for`.
- `aggregate_scalar(run_root, key, sources=(summary, test_results, research))`: searches the sources
  in order for a (possibly dotted, e.g. `freq_strata_ce.rare`) key; returns the standard
  `{n, mean, sd, two_sd, cv, values, seeds}` (unbiased ddof=1 SD).
- `aggregate_seed_curves(run_root, columns=None)`: read each seed's `metrics.csv`, align on `step`
  (union grid, NaN-pad the sparse `val_*` columns), return `{col: {steps, mean, sd, n}}` with
  NaN-aware mean / ddof=1 SD across seeds per step. `columns=None` -> all numeric columns.
- `aggregate_per_layer(run_root)`: `metrics_per_layer.csv` grouped by layer -> `{layer: {col: stats}}`.
- `aggregate_research(run_root)`: scalars (ece, overall_ce, sigma_ce_spearman, sigma_trace_cv,
  fd_gradient_worst_rel_error), `freq_strata_ce.{rare,mid,frequent}`, and the reliability curve
  aggregated per bin index.

### `vfe3/viz/figures.py` (new plot functions, beside `plot_ppl_noise_band`)

- `plot_curve_band(steps, mean, sd, *, label, ylabel, n=None, logy=False, path=None)`: mean line +
  +/-1 SD (and +/-2 SD) ribbon over training steps. One generic function, reused for every curve.
- `plot_curve_band_grid(curves, *, ncols=3, path=None)`: multi-panel overview of the key curves.
- `plot_scalar_cv_summary(aggs, *, path=None)`: horizontal CV% bar per scalar metric (seed-stability
  at a glance), with per-seed dots.
- `plot_per_layer_band(per_layer_agg, metric, *, path=None)`: per-layer bars with across-seed SD
  error bars.

All reuse `_CB`, `_save`, the `Agg` backend, and the existing `set_publication_style`. Not registered
in the single-run `FIGURE_REGISTRY` (they are multi-run, off the `generate_figures` driver), matching
how `plot_ppl_noise_band` already lives there unregistered.

### Outputs (into the git-ignored run dir)

- `figures/`: `ppl_noise_band_<root>.png` + a per-headline-scalar noise band, `scalar_cv_summary.png`,
  `curve_band__<col>.png` for the curated curve set, `curve_band_grid.png`, `per_layer__<metric>.png`.
- `multiseed_summary.json`: every scalar's n/mean/sd/2sd/cv, per-curve final-step stats, per-layer,
  research scalars, the seed list, and the shared-data-order lower-bound caveat.
- `multiseed_summary.csv`: flat scalar table.
- `MULTISEED_ANALYSIS.md`: summary table + key cross-seed findings + the caveat.

### Curated curve set (band figures)

train/val CE & PPL; `free_energy_total` + components (self_coupling, belief_coupling,
attention_entropy, self_divergence, hyper_prior, gamma_coupling); `grad_norm` (+ mu/sigma/phi);
`holonomy_deviation`; `gauge_trace_spread`; `effective_rank`; `attn_entropy`; `belief_cond_median`;
`fisher_trace_mean`; `estep_grad_norm_*`; `generalization_gap`. (The grid shows the headline subset.)

## Caveat (carried into outputs)

The per-run reseed shares the data-shuffle order across seeds, so every SD here is the
init+optimization spread only -- a LOWER BOUND on deployment variance. (Companion fix: fixed
data-order generator, `docs/experiments/2026-06-21-experiment-readiness.md` S6.)

## Testing (TDD)

Extend `tests/test_multiseed.py`: `_resolve_run_root` bare-name resolution; `_seed_for` provenance
precedence; `aggregate_seed_curves` step alignment + NaN handling; `aggregate_per_layer`;
`aggregate_scalar` dotted research key. Figure smoke tests (write a PNG to `tmp_path`) for the four
new plot functions, in the `tests/test_viz.py` style. Device-agnostic, no GPU.

## Non-goals

No change to training, the figure registry, or the single-run `report.py` driver. No new runtime
dependency (stdlib `csv` + `numpy`, matplotlib `Agg`). The user's `train_vfe3.py` WIP is untouched.
