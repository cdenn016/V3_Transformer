# V3_Transformer edits — 2026-06-14 (parameter-scaling tooling)

Branch `vfe3-scaling-experiments-2026-06-14`, off `vfe3-audit-fixes-2026-06-14` @ `bec80db` (a
separate edit doc from `2026-06-14-edits.md`, which carries concurrent audit-branch work). New
click-to-run tooling to RUN, ANALYZE, and PLOT parameter-scaling experiments. Designed from a
5-expert investigation; the scaling axis is the recorded `n_params`, and the gauge group / block
structure is a first-class parameter lever because `phi_embed = V*n_gen` dominates N
(`N = 2*V*K + V*n_gen + 1`; e.g. K=20 block_glk GL(5)x4 phi=5.0M vs GL(10)x2 phi=10.0M). Verified
end-to-end on synthetic loaders and a wikitext-2 smoke.

## Added
- `scaling.py` — click-to-run runner. Loops a size grid x seeds, delegating each cell to a replicated
  `_run_once` body (build -> train -> val -> `finalize_run` held-out TEST eval), reusing a memoised
  split-aware loader and the data-order reseed-after-build. Route builders: `grow_K`, `blocksize`,
  `group`, `model_channel`, `infer_T`, `infer_L` (the last two are flat-N inference-compute axes).
  `predict_n_params(cfg)` sizes a grid (builds only the cheap group) before any training; resumable
  via a config-equality check; config-error cells isolated, never crash the suite. Imports the
  baseline from `train_vfe3.config` (no edits to `train_vfe3.py`).
- `scaling_analysis.py` — click-to-run aggregator. Harvests run dirs -> `scaling_points.csv`,
  aggregates seeds (mean / SEM / t-CI), fits `L(N) = A*N^-alpha` (SEM-weighted log-log; optional
  Chinchilla `E + A*N^-alpha`) with a nested points-x-seeds bootstrap CI on alpha, runs the
  multi-route frontier-collapse F-test (do different routes to N share one curve?), and drives the
  figures. Provenance guards warn on `data_sha256` / `git_sha` drift; null / non-positive CE dropped.

## Changed
- `vfe3/run_artifacts.py` — `finalize_run` enriches `scaling_point` with a faithful cost model
  (helper `_cost_model_fields`): structural axes (embed_dim / n_heads / n_gen / n_blocks / n_layers /
  n_e_steps / ...), `active_params_per_token` (decode-bound ~K, NOT n_gen-bound — the mirror of
  n_params being n_gen-dominated), `est_flops_analytic` + per-token breakdown, `n_learnable_params`,
  normalized wall-clock, device. Keeps `est_flops_6ND` (the loose proxy). Best-effort: a failure logs
  and the saved numbers survive with the 6ND proxy intact.
- `vfe3/viz/figures.py` — added `_fit_power_law` (shared log-log / offset-power-law fit; scipy
  fallback to numpy), `_t95`, `_scaling_point_stats`, and registered figures `plot_scaling_law`
  (headline CE-vs-N log-log: per-seed cloud + cross-seed CI + fit + residual subpanel),
  `plot_scaling_routes` (route-colored+markered overlay; per-route and pooled fits),
  `plot_inference_capacity` (flat-N CE vs n_e_steps / n_layers). Best-effort, never-fatal.

## Verified
- `finalize_run` writes every cost-model field; param math exact (predicted N matches actual minus the
  small head-mixer commutant m^2).
- Aggregator on 14 synthetic-loader runs: CSV + pooled/per-route power-law fits + ANCOVA F-test + 4
  figures. wikitext-2 smoke through `scaling.py`'s real loader: 2 sizes, predicted vs actual N matches.

Note: `n_layers`, `n_e_steps`, and full covariance add ZERO parameters (inference-compute axes at
flat N) — plotted on the inference-capacity figure, never on the `L(N)` curve.
