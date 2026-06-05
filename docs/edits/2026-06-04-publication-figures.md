# 2026-06-04 — Publication-quality figures and metrics

Branch: `feat-publication-figures-2026-06-04` (fresh from the contiguous-flow + lambda_beta state,
merged to main first). Design spec:
`docs/superpowers/specs/2026-06-04-publication-figures-design.md`.

## What changed

The visualization surface was thin (single-marker trajectories, a one-bar free-energy snapshot
that omitted the data term, single-seed PPL bars with no spread, raw attention heatmaps with no
structural statistic). This change adds the claim-linked publication figure set and the metrics it
needs, all registry-backed and tested. The run protocol stays single-seed, so every figure leans on
within-run distributions (per-token, per-head, bootstrap over tokens/sequences) rather than
cross-seed CIs, and no figure claims backprop has been eliminated (the M-step still uses AdamW).

### `vfe3/metrics.py` (new pure-measurement functions, ~24)

Spectral/SPD: `effective_rank_per_token`, `belief_spectrum`, `fisher_trace`,
`spd_geodesic_distance`. Attention: `attention_entropy_rows`, `head_redundancy_js`,
`attention_distance_decay`, `positional_content_score`, `structured_head_scores`,
`causal_sanity`. Gauge: `group_gauge_invariant` (group-dispatched, fixing `gauge_trace_spread`'s
det-blindness on SO/Sp), `per_head_gauge_invariants`, `transport_asymmetry`, `energy_directedness`,
`gauge_equivariance_residual`. Holonomy: `holonomy_deviation_sampled` (random/stratified triples,
fixing the first-512 row-major low-index bias), `curvature_field`. Free energy / bootstrap:
`free_energy_full_decomposition` (closes the stack with the data term + lambda_beta scaling, guards
lambda_h/gamma_coupling != 0), `self_coupling_profile`, `estep_residuals`, `bootstrap_ce_band`,
`bootstrap_token_ce_band`, plus `guard_saturation`. All are pure (no gradients, no side effects),
keeping the module's contract.

### `vfe3/viz/extract.py` (new module — checkpoint/model runners)

The side-effecting runners that drive the model live here, not in `metrics.py`:
`per_unit_eval_nats` (per-sequence and per-token nats — the prerequisite the existing aggregate
`evaluate` cannot supply, unblocking every single-seed bootstrap band), `belief_bank` (collects
converged mu/Sigma/phi across many sequences for the UMAP triptych), `e_step_belief_trace` (loops
the inner E-step capturing the full belief, since `return_trajectory` yields F floats only),
`across_layer_belief_trace`, and `numerical_health`.

### `vfe3/viz/figures.py` (17 new registered figures)

`free_energy_descent` (F1), `estep_convergence` (F2), `ln3_symmetry_breaking` (F3),
`belief_trajectories` (F4), `belief_umap` (F5, the mu/Sigma/phi triptych — each channel embedded in
its own geometry with native-space silhouette/CH), `gauge_equivariance` (F6),
`gauge_head_specialization` (F7), `attention_structure` (F8), `belief_spectrum` and `spd_ellipses`
(F9; the latter eigendecomposes the 2x2 sub-block, replacing the diagonal-only ellipse),
`holonomy_curvature` (F10), `capacity_scaling`, `estep_capacity`, `pareto_frontier` (F11),
`ablation_forest`, `lr_grid_heatmap` (F12), `numerical_trust` (F13). Colourblind-safe palette,
multi-panel composites, units in nats/PPL/BPC.

### `vfe3/run_artifacts.py` (wiring)

`finalize_run` now writes `free_energy_descent.png` (the full F stack over training, closing to the
runtime total) alongside the existing bar; `RunArtifacts` keeps `self.cfg` so the figure can apply
the `lambda_beta` scaling. The existing one-bar `free_energy_terms.png` is retained for continuity;
the descent supersedes it.

## Tests

`tests/test_metrics.py` +24 property tests (e.g. `spd_geodesic_distance(S,S)==0`,
`attention_entropy_rows.mean()==attention_entropy`, in-group gauge-equivariance residual at machine
tolerance vs large out-of-group, flat-cocycle holonomy ~0). `tests/test_extract.py` (new, 5) builds
a tiny real model and runs each runner. `tests/test_viz.py` +17 smoke tests (each figure saves a
nonempty PNG). Full suite: 568 passed, 0 failures, 0 errors (read from `--junitxml`).

## Deliberately NOT done (deferred)

`ablation.py` was left untouched on purpose: it had live, uncommitted config-toggle edits during
this session, so editing it risked clobbering them. Consequently the F11/F12 ablation figures are
implemented and tested as figure functions in `figures.py` but are NOT yet wired into `ablation.py`
— specifically (1) the `grid` sweep form (Cartesian product of two value lists) that would feed
`plot_lr_grid_heatmap`, and (2) replacing `_plot_one_sweep`/`_plot_sensitivity` with
`plot_pareto_frontier` / `plot_capacity_scaling` / `plot_ablation_forest`. To render those, assemble
the inputs from `sweep_results.csv` (and `per_unit_eval_nats` for the bands) and call the plotters,
or add the wiring when `ablation.py` is not being edited. The `numerical_health` fallback-activation
counters (safe_cholesky jitter rounds, pinv fallbacks) are also deferred; the finiteness map and
conditioning are implemented.

The user's working-tree changes to `ablation.py` and `train_vfe3.py` were left as uncommitted WIP
(not part of this commit).
