# Experiment Readiness — Consolidated (2026-06-21)

Synthesis of verified per-experiment readiness audits for the top eleven experiments
(EXP-1 … EXP-11) of the gauge-theoretic VFE transformer program. Every capability and
build item below is traced to a `path:line` citation found by the per-experiment auditors;
nothing is invented. This is an engineering readiness document, so tables are used freely.

## 1. Executive summary

Of the eleven audited experiments, **zero are READY to run as-is with no new code at the
full deliverable level**, **one is PARTIAL** (EXP-11, H2 per-head temperature dispersion —
its core run + an auto PPL figure are runnable today, only the geo-mean-tau baseline arm and
a dispersion-scalar x-axis figure are unbuilt), and **ten are NEEDS_BUILD** (EXP-1 through
EXP-10). The single most important takeaway is that the *model and training core are not the
blocker* — almost every gap is **harness plumbing**: registering a multi-arm `configs` sweep
arm, capturing one extra scalar per ablation cell into `sweep_results.csv`, persisting one
converged-state quantity into the per-run headline JSON, and writing a new `@register_figure`
plotter plus its driver. The deepest shared blocker is that `ablation.run_single`
(`ablation.py:1069-1081`) returns **only PPL-family headline fields and never extracts the
converged belief state** (`mu/sigma/omega`), so four separate experiments (EXP-2, EXP-4,
EXP-9, EXP-10) that need a per-cell residual/diagnostic are each blocked on the *same*
missing extraction. Building that one capability, plus the `configs`-arm-with-`requires`
template (already battle-tested at `ablation.py:393-407`), unblocks the largest cluster of
experiments for the least code.

A second cross-cutting hazard the synthesizer must adjudicate is **confound control**: EXP-1's
"init+optimization only" SD claim is *false today* because the train loader has no fixed-seed
generator (`datasets.py:196-197`); EXP-6's inverse-K exponent is corrupted in *both* arms by a
`kl_max` frozen at `8*160=1280` (`scaling.py` deepcopies BASELINE and never recomputes it);
and EXP-9's existing `gauge_equivariance_residual` is the *wrong instrument* — it co-transforms
the supplied `omega` (`metrics.py:943`) so it reads float-eps for both tied and untied arms.
These are wiring gaps, not doc caveats, and each must be fixed before the corresponding
headline number is truthful.

## 2. Readiness matrix

| Exp | Title | Verdict | #EXISTS | #PARTIAL | #MISSING | Blocking gaps (short) |
|-----|-------|---------|:------:|:------:|:------:|-----------------------|
| EXP-1 | Multi-seed variance floor (I1) | NEEDS_BUILD | 0 | 2 | 3 | No across-seed mean±SD aggregator; data order co-varies with seed (no fixed generator) |
| EXP-2 | Gauge ON/OFF/frozen (A1) | NEEDS_BUILD | 1 | 4 | 0 | No `gauge_transport` sweep arm; `run_single` extracts no converged state for residual |
| EXP-3 | Sigma_q as calibrated Fisher (B1) | NEEDS_BUILD | 0 | 2 | 5 | No aligned (tr Σ_q, CE) join; no `sigma_trace`/Spearman/CV helpers; no reliability plot |
| EXP-4 | Canonical-F vs surrogate (C1) | NEEDS_BUILD | 2 | 2 | 2 | No `include_attention_entropy` arm; no per-snapshot Cov-gap metric; `run_single` drops H(β) |
| EXP-5 | Non-Neal-Hinton EM / n_e_steps (C2) | NEEDS_BUILD | 0 | 5 | 3 | Final E-step F/token not persisted to headline JSON; no cross-arm Pearson; no decorrelation plot |
| EXP-6 | muP width-stability of inverse-K (F1) | NEEDS_BUILD | 0 | 3 | 1 | No muP route; `kl_max` frozen at 1280 (both arms); embed_dim dropped at aggregation |
| EXP-7 | Prior-anchoring vs rank collapse (F2) | NEEDS_BUILD | 0 | 2 | 2 | No r(X) rank-one residual on mu; no rho-handoff multi-arm sweep; no per-arm depth-curve plot |
| EXP-8 | Pullback nat-grad gauge M-step + LR (D1) | NEEDS_BUILD | 0 | 2 | 4 | m_phi_lr entry not log-spaced/gated; no steps-to-target-PPL; no cos(nat,grad)/cond/wall-clock |
| EXP-9 | Tied vs untied gauge equivariance (A2) | NEEDS_BUILD | 0 | 2 | 1 | Existing residual blind to tied/untied (co-transforms ω); no per-eval series; no drift plot |
| EXP-10 | Clebsch-Gordan cross-irrep coupling (A3) | NEEDS_BUILD | 4 | 1 | 1 | No `use_cg_coupling` sweep arm; no combined equivariance+PPL bar (residual not in sweep CSV) |
| EXP-11 | Per-head temperature dispersion (H2) | PARTIAL | 2 | 1 | 2 | (non-blocking) geo-mean-tau baseline arm not auto-derived; PPL-vs-dispersion-scalar x-axis figure |

Totals: **READY 0 · PARTIAL 1 · NEEDS_BUILD 10**.

## 3. Consolidated build list (deduplicated)

The per-experiment build lists collapse into a small number of *shared infrastructure*
primitives plus a tail of *experiment-specific* plotters. Items are sorted so the
highest-leverage shared builds (those that unblock the most experiments) come first.

### 3a. Shared infrastructure (highest leverage first)

| # | What | Where (file / function) | Est lines | Effort | Unblocks |
|---|------|--------------------------|:--------:|:------:|----------|
| S1 | **Multi-arm `configs` SWEEPS entries** — register categorical arms via the existing `make_run_overrides` `configs`+`requires` machinery (template: `transport_mode` `ablation.py:393-400`, `cocycle_relaxation` `:406`). One entry per experiment: `gauge_transport` (3×2 cells×depths), `attention_entropy` (canon/surrogate, both `oracle_unroll_grad=True`), `n_e_steps × e_step_gradient` cross, EXP-7 `(lambda_alpha × rho × e_phi_lr)` arms, EXP-8 log-spaced `m_phi_lr` with `requires:{m_phi_natural_grad:True, phi_precond_mode:'pullback_per_block'}`, `cg_coupling` on/off on a labeled so_n tower. | `ablation.py` SWEEPS dict (`:336-718`) + active-sweep list (`:742-790`) | 12–22 each (~80 total) | small | **EXP-2, EXP-4, EXP-5, EXP-7, EXP-8, EXP-10** (6) |
| S2 | **Converged-state extraction + per-cell residual in `run_single`** — extract `(mu, sigma, omega/exp_phi)` from one forward pass at the end of `run_single` (mirror the cstate that `report.py:213` reads), call the relevant residual/diagnostic metric, reduce to a scalar (e.g. median in-group residual / per-cell H(β) / per-cell builder-defect), add to the return dict. `run_single` today returns only PPL-family fields and extracts **no** cstate (grep `cstate|omega|exp_phi` over `ablation.py` → zero hits). | `ablation.py` `run_single` (`:1062-1081`) | 18–20 | medium | **EXP-2, EXP-4, EXP-9, EXP-10** (4) |
| S3 | **New per-cell columns in `_CSV_COLUMNS`** — persist the S2 scalars to `sweep_results.csv`: `gauge_resid_in_group` (EXP-2/9), `attention_entropy` mean (EXP-4), `equiv_resid` (EXP-10), plus the EXP-4 Cov-gap scalar. | `ablation.py` `_CSV_COLUMNS` (`:1098-1102`) | 2 each (~8 total) | trivial | **EXP-2, EXP-4, EXP-9, EXP-10** (4) |
| S4 | **Cross-arm / cross-seed post-hoc correlation+aggregation reader** — a small harvester that scans per-run `summary.json`/`test_results.json` or per-cell `ablation_result.json`, gathers the relevant scalar tuples, and computes the experiment's reduction: across-seed mean±SD of `test_ppl` (EXP-1), Pearson(`n_e_steps`,`final_F`)/Pearson(`final_F`,`CE`) (EXP-5), per-arm K-axis exponent (EXP-6). Model on `scaling_analysis.aggregate_points` (`scaling_analysis.py:127-156`) but key to the correct directory/JSON schema. **Do not reuse** `figures.py:644 np.corrcoef` — it is a within-run training co-descent, wrong semantics. | new small entry points / `scaling_analysis.py` helpers | 30–45 each | small | **EXP-1, EXP-3, EXP-5, EXP-6** (4) |
| S5 | **Multi-seed keying for error bars** — key ablation cells by `label+seed` (not label-only at `ablation.py:1199`) so re-running with different `CONFIG['seed']` does not overwrite the same dir; required for any grouped-bar/curve that shows cross-seed spread. (EXP-1 needs the analogous fixed-seed generator on the *data* loader; see S6.) | `ablation.py` `run_sweep` (`:1198-1226`) + CONFIG seed (`:820`) | 30 | medium | **EXP-1, EXP-2** (2) |
| S6 | **Fixed data-order generator** — add a `generator=` kwarg to `make_dataloader` (`datasets.py:175-197`, shuffle=True/train path) and thread a `DATA_SEED` constant through `_select_loader`→`make_dataloader` so all seeds share one shuffle order while model-init RNG still varies with `cfg.seed`. Default `generator=None` keeps existing callers unchanged. Without this, EXP-1's "init+optimization only" SD claim is false (data-shuffle variance is folded in). | `vfe3/data/datasets.py` `make_dataloader`; `train_vfe3.py` `_select_loader`/`_run_once` | 20 | small | **EXP-1** (1; truthfulness gate) |
| S7 | **Power-law fit generalization (K-axis + PPL + parameterized bootstrap CI)** — the `+c` offset fit form already exists (`figures.py:1977-1989`), but `x` is hard-wired to `n_params` and `y` to CE. Add an `x_key` param to `bootstrap_exponent_ci` (`scaling_analysis.py:163`, default `'n_params'`, line `:179` reads `p[x_key]`) and propagate `embed_dim` through `aggregate_points` (`:143-155`; harvested at `:84`, currently dropped) so per-arm exponents can be fit vs K on PPL with a bootstrap CI. | `scaling_analysis.py` `aggregate_points` + `bootstrap_exponent_ci` | ~28 | medium | **EXP-6** (1; reused pattern across scaling work) |

### 3b. Experiment-specific builds

| # | What | Where | Est lines | Effort | Exp |
|---|------|-------|:--------:|:------:|:---:|
| B1-a | Across-seed aggregator `aggregate_seed_ppl(run_root)`: scan seed-labelled `vfe3_runs/` dirs, load each `test_results.json`, report n / mean / SD (ddof=1) / 2·SD. | new `multiseed_analysis.py` | 45 | small | EXP-1 |
| B1-b | Noise-floor reader `flag_noise_dominated_cells(sweep_dir, sd)`: load `sweep_results.csv` (reuse `_read_sweep_csv` `:1268-1273`), compute between-cell spread, flag cells with spread < 2·SD; reconcile val-PPL grid vs test-PPL SD units. | `ablation.py` near `analyze_sweep` (`:1283`) | 35 | small | EXP-1 |
| B1-c | `@register_figure('ppl_noise_band')`: horizontal cross-seed SD band (axhspan/fill_between) over the K=20 grid/headline PPL. Existing band callers are within-run/scaling-route only. | `vfe3/viz/figures.py` near `plot_ablation_forest` (`:1847`) + caller | 40 | medium | EXP-1 |
| B3-a | `sigma_trace(sigma, *, diagonal, eps)` → per-token tr(Σ_q)=Σ_k σ_k. NB `fisher_trace` (`metrics.py:244-265`) is **precision** tr(Σ⁻¹)/2, wrong sign for B1; the raw tr(Σ_j) already exists internally at `model.py:1230`. | `vfe3/metrics.py` beside `fisher_trace` | 18 | trivial | EXP-3 |
| B3-b | `belief_ce_bank(...)`: the missing Σ↔CE **join** — replay the trained stack (as `belief_bank` `extract.py:182-195`) for `out.sigma`, compute per-position CE with `ignore_index=-100`, mask `valid=targets!=-100`, return aligned `{tr_sigma, ce, token_ids}`. | `vfe3/viz/extract.py` after `per_unit_eval_nats` (`:151`) | 45 | medium | EXP-3 |
| B3-c | `spearman_rho(x,y)` (rank via `argsort().argsort()`, Pearson on ranks) + `cv(x)=std/|mean|`; persist ρ(tr Σ,CE) and CV(tr Σ) into `test_results.json` + CSV. No Spearman/CV helper exists anywhere (both greps empty). | `vfe3/metrics.py` + `run_artifacts.py` beside `_calibration_and_strata` (`:404`) | 40 | small | EXP-3 |
| B3-d | `plot_reliability_diagram(reliability)` consuming the existing `run_artifacts.py:395` reliability bins (conf/acc/frac); acc-vs-conf with y=x diagonal, annotate ECE. Bins are computed but never plotted. | `vfe3/viz/figures.py` (`~:2262`) + `report.py` driver | 35 | small | EXP-3 |
| B3-e | `plot_sigma_stratified_error(bank)`: bin `ce` by tr(Σ_q) deciles, mean CE ± bootstrap band per bin. | `vfe3/viz/figures.py` (`~:2262`) | 40 | small | EXP-3 |
| B3-f | `plot_sigma_ce_scatter(bank)`: per-token tr(Σ_q) vs CE hexbin, Spearman ρ in title. | `vfe3/viz/figures.py` (`~:2262`) | 30 | small | EXP-3 |
| B3-g | *(optional, only if Σ-conditioned recalibration in scope)* 1-param temperature fit T over held-out logits + a tr(Σ)-binned variant, recompute 15-bin ECE per Σ-bin. Extend `_calibration_and_strata` (`:350-404`). | `vfe3/run_artifacts.py` | 55 | medium | EXP-3 |
| B4-a | Cov-gap metric: given a model belief snapshot, call `belief_gradients` twice (`include_attention_entropy` True vs False) on the **same** snapshot and return per-snapshot ‖g_surrogate − g_canonical‖ (= ‖−τ⁻¹ Cov_β(E,dE)‖). Both routes already return both grads; identity proven only synthetically at `test_free_energy.py:346-386`. Register as a metric. | `vfe3/metrics.py` near `attention_entropy` (`:39/:1001`) + call sites | 45 | medium | EXP-4 |
| B4-b | 2-row (canon/surrogate) × |κ| PPL-gap table/heatmap, reusing the `plot_lr_grid_heatmap` layout (`figures.py:1876`) with a new caller. | `vfe3/viz/figures.py` new `@register_figure` + caller (`~:1399` / make_figures) | 50 | medium | EXP-4 |
| B4-c | Cov-gap-magnitude figure (‖−τ⁻¹ Cov_β‖ vs κ), fed by B4-a. | `vfe3/viz/figures.py` + caller | 40 | medium | EXP-4 |
| B5-a | Add value `8` to the `n_e_steps` sweep values → `[1,2,3,5,8]`. | `ablation.py:359` | 1 | trivial | EXP-5 |
| B5-b | Persist converged **final E-step F/token** per run: after test eval run one `e_step_belief_trace`/`free_energy_value` on a fixed batch, record `results['estep_final_f_per_token']` into `test_results.json` + `summary.json`. Load-bearing — unblocks the two Pearsons and the decorrelation scatter. | `vfe3/run_artifacts.py` `finalize_run` (`:591-598`, `:641-655`) | 25 | medium | EXP-5 |
| B5-c | `plot_f_ce_decorrelation(arms)`: per-arm (final F/token) vs (test CE) scatter, one point per `n_e_steps`, annotate Pearson r; follow `plot_inference_capacity` (`:2130`). | `vfe3/viz/figures.py` | 35 | small | EXP-5 |
| B5-d | Driver feeding n_e_steps arms' (n_e_steps, BPC/PPL, F) into `plot_estep_capacity` (registered+tested but **uncalled** by any driver) and the new decorrelation plot. | `scaling_analysis.py` (`~:370-378`) or new entry point | 30 | small | EXP-5 |
| B6-a | `route_grow_k_mup(embed_dims, n_heads, anchor_K=20)`: per-cell muP-scale `e_q_mu_lr`/`m_p_mu_lr` by `20/K` and `mu_init_std` by `sqrt(20/K)`; tag `route='grow_K_mup'`. The three knobs already exist and are live (`config.py:337/438/59`). Register in `ROUTES`. | `scaling.py` near `route_grow_k` (`:70`) + `ROUTES` (`:135`) | 18 | small | EXP-6 |
| B6-b | Recompute `kl_max` per cell: after `d.update(overrides)`, `d["kl_max"] = 8*int(d["embed_dim"])` so every cell gets 8·K not the frozen 1280. Confound that corrupts **both** arms. | `scaling.py` `_cell_cfg_dict` after `:246` | 2 | trivial | EXP-6 |
| B6-c | `plot_kmup_stability(points)`: grow_K vs grow_K_mup on shared axes, x=embed_dim(K), test_ppl/CE per K with cross-seed 95% CI bars, offset-power-law fit overlaid, each arm's b annotated with bootstrap CI. | `vfe3/viz/figures.py` after `plot_scaling_routes` (`:2127`) + caller in `scaling_analysis._make_figures` (`:348+`) | 55 | medium | EXP-6 |
| B7-a | `rank_one_residual(X)=‖X−1·mean(X)ᵀ‖_F/‖X‖_F` on (N,K) means + per-layer wrapper over (L,N,K). The existing `effective_rank` is on **Σ**, the wrong object. Per-layer mu stack already materialized (`extract.py:288`). | `vfe3/metrics.py` near `effective_rank` (`:21`/`:204`) | 12 | small | EXP-7 |
| B7-b | Register r(X) and surface on the per-layer trace (`across_layer_belief_trace` return `extract.py:288-293`; `diagnostics_per_layer` `model.py:1654-1721`). | `extract.py` + `model.py` | 8 | small | EXP-7 |
| B7-c | `depth_decay_rate(curve)`: log-linear slope of r(X) over depth (`np.polyfit`). | `vfe3/metrics.py` or EXP-7 driver | 6 | trivial | EXP-7 |
| B7-d | `plot_rank_residual_by_depth({arm: (L,) r(X)})`: one line per arm, r(X) on y, layer on x — no multi-arm depth-overlay driver exists (all are single-model). NB EXP-7 arms must pin `lambda_alpha_mode='constant'` (baseline is `'state_dependent'`) and set `rho` explicitly per arm. | `vfe3/viz/figures.py` near `plot_per_layer_diagnostics` (`:1554`) + EXP-7 driver | 35 | medium | EXP-7 |
| B8-a | Log `cos(nat,grad)` at train time: in `GaugeNaturalGradAdamW.step` compute `cosine_similarity(nat[active], g[active]).mean()`, stash, emit `cos_nat_phi` via `train.py:836-842`. Distinguishes pullback (rotates step) from killing/none (cos=1). | `vfe3/gauge_optim.py` `step` (`~:111-127`) + step_metrics | 12 | small | EXP-8 |
| B8-b | Log per-token pullback condition number: `condition_number(G_metric+eps·I)` inside the pullback preconditioner (`phi_preconditioner.py:334-337`,`:402-406`), stash, emit `pullback_cond_*`. | `phi_preconditioner.py` + `gauge_optim.py` + `train.py:817-832` | 25 | small | EXP-8 |
| B8-c | Per-row cumulative wall-clock column `wall_clock_s` in metrics.csv (running sum of the per-window elapsed already measured for `tokens_per_s`). | `vfe3/train.py:802-804` | 6 | trivial | EXP-8 |
| B8-d | Steps-to-target-val-PPL reducer: post-run scan metrics.csv `val_ppl` vs step, record first step ≤ target into summary.json. Derivable today but computed nowhere (grep empty). | `vfe3/run_artifacts.py` (`~:547-552`) or end-of-run reducer | 20 | small | EXP-8 |
| B8-e | `@register_figure('wallclock_convergence')`: val_ppl vs cumulative wall time, one line per arm (log-y). Depends on B8-c. | `vfe3/viz/figures.py` near `plot_pareto_frontier` (`~:1804`) | 30 | medium | EXP-8 |
| B9-a | **Builder-break equivariance metric** `gauge_builder_residual`: for sampled g=exp(Σ c_a G_a), compare `mix(g·μ, g·Σ·gᵀ)` against `g·mix(μ,Σ)·gᵀ` — float-eps under `tied_block_glk`, grows under `block_glk` as the mixer drifts. The existing `gauge_equivariance_residual` co-transforms supplied ω (`metrics.py:943`) so it is **blind** to tied/untied. Lift the recipe from `tests/test_head_mixer.py::test_head_mixer_equivariant_under_tied_gauge_full_cov`. | `vfe3/metrics.py` near `:878` | 55 | medium | EXP-9 |
| B9-b | Log `val_gauge_builder_residual` per eval: add to `_VAL_DIAG_KEYS` (`train.py:470-478`) and compute in `_val_diagnostics` (`:482-559`) → metrics.csv series. | `vfe3/train.py` | 14 | small | EXP-9 |
| B9-c | `@register_figure('gauge_residual_drift')`: residual vs step (log-y), overlay tied vs untied arms. Existing `plot_gauge_equivariance` (`:1437`) is a single-state ECDF, no step axis. | `vfe3/viz/figures.py` near `:1437` | 35 | small | EXP-9 |
| B9-d | Driver assembling the drift figure across the `gauge_group` sweep arms (`ablation.py:374-389`). NB mixer must be ON and drifted (>0 steps); step 0 is byte-identical for both arms. | `make_figures.py` (`:47-56`) | 30 | small | EXP-9 |
| B10-a | Combined per-arm bar (val PPL + median equivariance residual) reading the new `equiv_resid` column; model on `_plot_one_sweep` (`:1353-1396`). NB the CG on-arm must run with `e_step_gradient` `unroll`/`straight_through` (or `detach_e_step=False`) or path_weights freeze and on/off become identical (`model.py:179-198`). | `ablation.py` near `_plot_one_sweep` (`~:1397`) | 30 | small | EXP-10 |
| B11-a | Geo-mean-τ confound-control baseline arm: scalar κ* = (∏ κ_h)^(1/H) (equal-block) added to the `kappa_beta_per_head` `configs` list. *(EXP-11; non-blocking.)* | `ablation.py:614-620` | 8 | trivial | EXP-11 |
| B11-b | `geomean_tau_kappa(kappa_list, irrep_dims)` helper to derive κ* in-code. | `vfe3/free_energy.py` near `attention_tau` (`:41`) | 12 | trivial | EXP-11 |
| B11-c | `_plot_kappa_dispersion(sweep_dir, fig_dir)`: read per-cell `ablation_result.json` `overrides` (the κ list is **not** a flat CSV column), scatter PPL vs dispersion scalar (std or max/min). | `ablation.py` near `_plot_one_sweep` (`:1353`); dispatch `:1484` | 35 | small | EXP-11 |

## 4. Per-experiment detail

### EXP-1 — Multi-seed variance floor (I1) — NEEDS_BUILD

| Capability | Kind | Status | Evidence | Note |
|---|---|---|---|---|
| EXP-1 baseline knobs (K=20, n_heads=2, n_layers=1, max_steps=15000, block_glk, head_mixer, λ_h=0.25) | config | PARTIAL | all knobs exist+consumed: `train_vfe3.py` config dict lines 69/70/72/75/77/104/119/133/195; coerced `config.py:546-547`; EXP-1 values match `ablation.py` BASELINE_CONFIG `:81/83/87/89`; LIVE values are the K=160 op-point | reaching K=20 is a config edit (click-to-run), not a build; do not silently re-apply (live WIP) |
| across-seed mean and SD of test_ppl | metric | MISSING | `run_artifacts.py:593/619` writes per-run test_ppl; `train_vfe3.py:513` writes one seed dir per seed; nothing aggregates the 5; `scaling_analysis.aggregate_points` (`:127-156`) is wrong dir/schema and reports SEM on test_ce | the 5 values are on disk, nothing reduces them |
| read single-seed grids + flag cells with spread < 2 SD | metric/infra | MISSING | `analyze_sweep` (`:1283-1306`) prints per-cell val_ppl; no SD-aware flagging; grep `2 SD/between-cell/noise-band` → only the hypotheses doc | entirely manual; grids are val-PPL vs EXP-1's test_ppl SD (units mismatch) |
| noise-band (SD) overlay on K=20 PPL/ablation grids | figure | PARTIAL | band primitives exist (`figures.py:459`, `:1848`, `:2046-2058`, `:823-868`) but every caller is within-run or scaling-route; `ablation._plot_one_sweep:1353` has no band | new plotting fn + caller required |
| reseed fixes data order so SD is init+optimization only | infra | MISSING | post-build reseed (`train_vfe3.py:484`) uses per-run `cfg.seed`; `make_dataloader` (`datasets.py:196-197`) shuffle=True, no `generator=` → batch order co-varies with seed; no `data_seed`/`generator=` anywhere | the "init+optimization only" disclosure is **false** until S6 is built |

Blocking gaps: (1) no across-seed aggregator; (2) data order co-varies with seed, so the reported SD confounds init+optimization with data-shuffle variance.

### EXP-2 — Gauge ON/OFF/frozen (A1) — NEEDS_BUILD

| Capability | Kind | Status | Evidence | Note |
|---|---|---|---|---|
| `gauge_transport` on/off/frozen knob | config | EXISTS | `config.py:78` declared, consumed `:546-583` (off→Ω=I via phi_scale=0/pos_phi='none'/e_phi_lr=0/m_phi_lr=0; frozen→lr=0, phi_scale>0); real field assignments | the knob half EXISTS (commit 31cdae5 + 7 tests) |
| ablation SWEEPS `gauge_transport` multi-arm entry | sweep_route | MISSING | grep over `ablation.py` → zero hits; not in SWEEPS (`:336-718`) or SWEEP_ORDER (`:742-790`); `main()` raises "unknown sweep" `:1469` | the hypotheses doc calls it "New … entry" — not yet written |
| primary_val_ppl in CSV; gauge residual per-run | metric | PARTIAL | `primary_val_ppl` is a CSV col (`:1099`); `gauge_equivariance_residual` exists (`metrics.py:878-963`) but consumed only in the single-run figure path (`report.py:211-216`); `run_single` returns no cstate (grep zero) and residual not in `_CSV_COLUMNS` | residual needs converged state `run_single` doesn't extract (S2) |
| depth arm n_layers∈{1,2}; matched param counts | infra | PARTIAL | `n_params` logged per cell (`:1025/1078/1101`); standalone n_layers sweep `:353-356`; no 2-D cross product; pos_phi asymmetry real (`model.py:346-347` vs `:336-338`); phi_embed unconditional (`prior_bank.py:143`) | arms must pin `pos_phi` uniformly to match counts |
| gauge-ablation grouped-bar + residual-table driver | figure | PARTIAL | `_plot_one_sweep` (`:1353-1396`) single-axis, no depth grouping/error bars/seed agg; single-seed (`:820`, `:1199` keys by label only); `plot_gauge_equivariance` is one-run ECDF | error bars need ≥3 seeds; multi-seed keying absent (S5) |

Blocking gaps: no `gauge_transport` sweep arm → the ON/OFF/FROZEN experiment cannot launch via the ablation runner at all.

### EXP-3 — Sigma_q as calibrated Fisher uncertainty (B1) — NEEDS_BUILD

| Capability | Kind | Status | Evidence | Note |
|---|---|---|---|---|
| per-token tr(Σ_q) on held-out tokens (mask ≠ −100) | metric | PARTIAL | `fisher_trace` (`metrics.py:244-265`) returns **precision** tr(Σ⁻¹)/2; `belief_bank` (`extract.py:154-214`) banks σ but no target/CE, no mask; raw tr(Σ_j) on the live path at `model.py:1220-1238/:1230` but detached | no helper returns Σ_k σ_k; `per_unit_eval_nats` masks but carries no σ |
| Spearman ρ(tr Σ_q, CE) | metric | MISSING | grep spearman/rankdata/corrcoef → only `figures.py:644` (F-vs-CE, unrelated); no aligned (tr Σ, CE) pairs | headline statistic has neither data nor helper |
| across-token CV of tr(Σ_q) (gate CV>0.10) | metric | MISSING | grep CV → only a test comment; only `fisher_trace_mean/median` (`model.py:1466-1468`), on precision | gate has no implementation |
| Σ-conditioned temperature recalibration (15-bin ECE) | metric/infra | PARTIAL | 15-bin ECE + reliability + frequency strata at `run_artifacts.py:350-404`; no post-hoc T fit, no Σ covariate | strata are frequency-based, not Σ-based |
| reliability diagram | figure | MISSING | only `plot_vocab_calibration` (`:2226-2261`, hexbin), no acc-vs-conf reliability plot; reliability bins (`:395`) unplotted | — |
| tr(Σ_q)-stratified error curve | figure | MISSING | grep `strat` in figures.py → none | needs the Σ↔CE join + plotter |
| ρ scatter (tr Σ_q vs CE) | figure | MISSING | generic scatter exists, no Σ-vs-CE caller | — |

Blocking gaps: (1) no aligned (tr Σ_q, CE) per-token data (the join); (2) no `sigma_trace` helper (fisher_trace is precision, wrong sign); (3) no Spearman and no CV helper anywhere.

### EXP-4 — Canonical-F vs entropy-suppressed surrogate (C1) — NEEDS_BUILD

| Capability | Kind | Status | Evidence | Note |
|---|---|---|---|---|
| `include_attention_entropy` knob, live E-step + oracle | config | EXISTS | `config.py:310`; consumed `kernels.py:202/259/267`, `oracle.py:68/139`, free_energy gate, `metrics.py:158-159`; pinned `test_surrogate_end_to_end.py` | the load-bearing toggle is solid and live in descent |
| attention_entropy H(β) computed and logged | metric | EXISTS | `metrics.py:39`, registered `:1001-1004`, per-row `:297`; logged per training step `run_artifacts.py:800/819-826` | captured per-run, **not** per ablation cell |
| ablation multi-arm SURROGATE vs CANON_ORACLE on oracle route | sweep_route | PARTIAL | configs machinery proven (`:393-400`, `:410-418`); κ_beta sweep exists (`:603-606`); grep `include_attention_entropy` → no arm; `run_single` returns only PPL fields | drop-in template; force both onto oracle route to isolate entropy term |
| −τ⁻¹ Cov_β gradient gap measurable | metric | PARTIAL | identity proven only synthetically `test_free_energy.py:346-386`; `belief_gradients` (`kernels.py:209-271`) returns both grads; grep cov_gap → none | only the differencer + registered metric are absent |
| PPL-gap table (entropy × κ) + Cov-gap plotter | figure | MISSING | `_plot_one_sweep`/`_plot_sensitivity` only generic PPL; full figures.py grep → no surrogate/gap/Cov-gap plotter; `plot_lr_grid_heatmap` (`:1876`) reusable with new caller | two new plotters needed |

Blocking gaps: no `include_attention_entropy` arm; `run_single` drops H(β) per cell; no per-snapshot Cov-gap function (only a synthetic proof in tests).

### EXP-5 — Structural non-Neal-Hinton EM / n_e_steps (C2) — NEEDS_BUILD

| Capability | Kind | Status | Evidence | Note |
|---|---|---|---|---|
| n_e_steps sweep {1,2,3,5,8} | sweep_route | PARTIAL | `ablation.py:357-359` values `[1,2,3,5]`; consumed `model.py:570`, `block.py:62`, `extract.py:234/315` | one-char fix: add 8 |
| e_step_gradient pinned 'unroll' + 'straight_through' arm | config | PARTIAL | both live in `e_step.py` (unroll `:473`, straight_through `:498`); grep over ablation → no arm; BASELINE doesn't set it (falls to 'unroll') | needs a crossing arm; 2nd arm must override explicitly |
| test PPL vs n_e_steps | metric | PARTIAL | ablation returns val PPL only (`:1069-1081`); test split deliberately not scored per cell (`:23-26`); test only via `finalize_run` (`:592-594`) / scaling route | test PPL needs scaling `route_inference_t` or per-arm train_vfe3 |
| final E-step F/token logged & saved per run | metric | PARTIAL | computable (`e_step.py:164`, `extract.py:243-254` fs[-1]); only `estep_f_drop` (a delta) persisted to metrics.csv (`train.py:476/536-537`); grep `final_f` → none | no converged-F scalar in headline JSON |
| Pearson(n_e_steps, F) and Pearson(F, CE) across sweep | metric | MISSING | only `figures.py:644` within-run co-descent; no cross-sweep Pearson | different semantics — do not reuse |
| final F/token saved per run for post-hoc correlation | infra | MISSING | `finalize_run` (`:591-598`, `:641-655`) carries no converged-F field; grep `final_f` → none | load-bearing gap |
| PPL-vs-n_e_steps curve | figure | PARTIAL | `plot_estep_capacity` (`:1769-1801`) registered+tested but **uncalled** by any driver; `plot_inference_capacity` (`:2130`) wired only into scaling (test_ce, no F) | clean PPL+F-vs-n_e_steps from ablation needs a new caller |
| F-CE decorrelation scatter | figure | MISSING | enumerated all scatter sites; none plot final-F vs CE across arms | new figure + driver |

Blocking gaps: each sweep run must persist a converged final E-step F/token scalar next to test_ce; without it the two cross-arm Pearsons and the decorrelation scatter are uncomputable.

### EXP-6 — muP width-stability of inverse-K exponent (F1) — NEEDS_BUILD

| Capability | Kind | Status | Evidence | Note |
|---|---|---|---|---|
| route_grow_k_mup (muP LRs + init) | sweep_route | MISSING | `route_grow_k` (`scaling.py:64-70`) scales only embed_dim/n_heads/gauge_group; grep mup/muP → only the pre-existing `mu_init_std` knob | the 3 target knobs exist+live (`config.py:337/438/59`); build is a route builder over existing overrides |
| kl_max recomputed per cell as 8·embed_dim in BOTH arms | config | PARTIAL | `train_vfe3.py:311` runs once at import with embed_dim=160 → static 1280; `scaling.py:244` deepcopies BASELINE, no recompute; `config.py:540-541` only validates >0 | frozen 1280 over-relaxes every small-K cell — width-dependent confound in both arms |
| fit exponent b from PPL=aK^b+c vs embed_dim | metric | PARTIAL | `_fit_power_law` (`figures.py:1954`) has +c offset (`:1977-1989`) but x hard-wired to n_params (`scaling_analysis.py:292/179`), embed_dim dropped at aggregation (`:143-155`; harvested `:84`), y is CE not PPL; per-K test_ppl/test_ce/embed_dim all recorded (`run_artifacts.py:646/635/526`) | underlying per-K data intact; only aggregation + K-axis/PPL fit absent |
| two scaling curves (b_fixed vs b_muP) with bootstrap CIs | figure | PARTIAL | `bootstrap_exponent_ci` (`:163-198`) keyed on n_params (`:179`); `plot_scaling_routes` (`figures.py:2086`) x_key default n_params, no per-route CI band; make_figures is single-run | new caller for K-axis two-arm + CI band |

Blocking gaps: (1) no muP route; (2) kl_max frozen at 1280 for every cell — must be recomputed per cell before any EXP-6 run is valid.

### EXP-7 — Prior-anchoring resists rank collapse (F2) — NEEDS_BUILD

| Capability | Kind | Status | Evidence | Note |
|---|---|---|---|---|
| prior_handoff_rho (1 vs 0) | config | PARTIAL | `config.py:392`, validated `:1425-1426`; live `stack.py:62/72`; mirrored `model.py:836/1302/1548/1650`, `extract.py:272/401`; **no sweep**: only baseline assignment `ablation.py:286` | knob real+consumed; only the sweep route missing; needs (λ_alpha × rho × e_phi_lr) arms |
| Dong rank-one residual r(X)=‖X−1xᵀ‖_F/‖X‖_F on per-layer mu | metric | MISSING | grep rank_one/rank1/outer → none on the mean stream; only `effective_rank` (`metrics.py:21/991-998`) on **Σ**; per-layer mu stack exists (`extract.py:288`) | wrong object confirmed; add ~3-5 lines + registration |
| per-layer decay-rate | metric | MISSING | grep decay_rate/polyfit → only scaling-law/attention polyfit; nothing slopes a rank quantity over depth | depends on r(X) |
| r(X)-vs-depth curves per arm | figure | PARTIAL | `plot_belief_trajectories` (`:1056`) and `plot_per_layer_diagnostics` (`:1554`) plot effective_rank-on-Σ, single-run; no r(X) y-axis, no multi-arm overlay | new plotter + multi-arm caller |

Blocking gaps: (1) no r(X) metric on per-layer mu (only effective_rank on Σ); (2) no multi-arm rho-handoff sweep route; (3) no per-arm r(X)-vs-depth plotter. NB EXP-7 arms must pin `lambda_alpha_mode='constant'` and set `rho` explicitly per arm.

### EXP-8 — Pullback natural-gradient gauge M-step + LR (D1) — NEEDS_BUILD

| Capability | Kind | Status | Evidence | Note |
|---|---|---|---|---|
| mass_phi regime knob (NOT phi_weight_decay) — exists, wired, sweepable | config | PARTIAL | `config.py:211`, validated `:865-866`; E-step `e_step.py:340`; M-step `model.py:799/805`; sweepable (generic param machinery); phi_weight_decay hard-zeroed under nat-grad `train.py:90-94`; mass_phi NOT in preconditioner (grep empty) | EXISTS as the swept regime knob; PARTIAL only because it acts via loss/E-step shrinkage, not inside G(phi) |
| log-spaced m_phi_lr sweep with requires gate | sweep_route | PARTIAL | requires machinery live (`:912/918/922`); existing entry `:694-697` is 2 linear ungated points | the gated log-spaced entry must be authored (trivial) |
| steps-to-target-val-PPL | metric | MISSING | val_ppl logged per eval (`train.py:790`); grep steps_to/target_ppl → empty | derivable post-hoc, computed nowhere |
| per-token pullback condition number | metric | MISSING | `condition_number` (`numerics.py:146`) applied only to belief Σ (`extract.py:343-344`); G(phi) solved+discarded (`phi_preconditioner.py:334-337/402-406`); grep pullback-cond → empty | reuse condition_number on G_metric |
| cos(nat,grad) | metric | MISSING | only docstring `gauge_optim.py:13` + test `:53-54`; step (`:111-127`) never computes it | distinguishes pullback (rotate) from killing/none (cos=1) |
| per-wall-clock convergence curve | figure | MISSING | wall_time only in cost-frontier plots (`figures.py:1795-1798/1818/1841`); no per-row cumulative-elapsed column (only per-run `wall_time_s` `:548-550/652`) | needs both a metrics.csv column and a plotter |

Blocking gaps: (1) the gated log-spaced m_phi_lr entry not authored; (2) no steps-to-target convergence-speed metric on the live path. (cos_nat / pullback cond / wall-clock are headline diagnostics but do not block the run.)

### EXP-9 (A2) — Tied vs untied gauge equivariance — NEEDS_BUILD

| Capability | Kind | Status | Evidence | Note |
|---|---|---|---|---|
| gauge_equivariance_residual per-eval (drift curve) | metric | PARTIAL | defined `metrics.py:878-963`; only call site `report.py:212-215` (post-hoc, end-of-run); not registered; not in `_VAL_DIAG_KEYS` (`train.py:470-478`); cstate exists `extract.py:366/439-444` | fires once per run, never per eval |
| does residual DISTINGUISH tied vs untied? | metric | PARTIAL | `_residuals` (`:939-947`) applies one global g and **co-transforms supplied omega** (`:943`) → joint-congruence identity, float-eps for both arms; head_mixer applied in forward (`extract.py:190/277/394`), not in the recompute → blind to the mixer break | genuine confound; needs a NEW builder-break metric |
| residual-drift-vs-step plotter | figure | MISSING | only `plot_gauge_equivariance` (`:1437`) single-state ECDF, no step axis; grep drift/vs_step → empty | new plotter + the new per-step series |

Blocking gaps: (1) existing residual is blind to tied/untied (co-transforms ω; blind to the mixer break) — author a builder-break metric (lift `tests/test_head_mixer.py::test_head_mixer_equivariant_under_tied_gauge_full_cov`); (2) no per-eval residual series; (3) no drift-vs-step plotter. NB step 0 is byte-identical for both arms (identity init); only the untied arm should climb.

### EXP-10 (A3) — Clebsch-Gordan cross-irrep coupling — NEEDS_BUILD

| Capability | Kind | Status | Evidence | Note |
|---|---|---|---|---|
| use_cg_coupling knob | config | EXISTS | `ablation.py:161` default; live `model.py:172-178`; threaded `:719/1142/1292/1563/1662` | requires a labeled so_n/sp_n tower |
| CGCoupling bilinear module | infra | EXISTS | `cg_coupling.py:26-162` (zero-init path_weights `:102`, means-only `:139-162`) | sigma passes through (means-only phase) |
| gauge equivariance residual | metric | EXISTS | `metrics.py:878`; consumed `report.py:211-212` | per-run, not per-sweep |
| CG zero-init/equivariance/e2e tests | infra | EXISTS | `tests/test_cg.py:82/93/109/124/148/155` | none exercise an ablation arm |
| use_cg_coupling on/off arm in SWEEPS | sweep_route | MISSING | grep → only `:161` default; SWEEPS (`:336-718`) has no key; gauge_group arms (`:374-390`) are single-type towers, none set use_cg_coupling | template ready (`:393-400`); a ≥2-type tower is the honest test |
| equivariance + PPL bar (combined) | figure | PARTIAL | PPL bar `_plot_one_sweep` (`:1353-1396`); equivariance is a per-run ECDF (`figures.py:1437/report.py:211`); residual not in `_CSV_COLUMNS` | needs residual CSV column + combined plotter |

Blocking gaps: no ablation arm toggles use_cg_coupling on/off (the module + knob exist; nothing sweeps them on a labeled tower). NB the on-arm must run with `e_step_gradient` `unroll`/`straight_through` (or `detach_e_step=False`) or path_weights freeze and on/off become identical (`model.py:179-198`).

### EXP-11 — H2 Per-head temperature dispersion — PARTIAL (runnable today)

| Capability | Kind | Status | Evidence | Note |
|---|---|---|---|---|
| per-head kappa_beta list arm (live knob) | config | EXISTS | `config.py:195`; live `block.py:62/21-23` → `free_energy.attention_tau` (`:41-76`, per-head `:76`); validated `:842-853`; pinned `test_son_irreps.py:193-196`, `test_cheap_ledger_wins.py:133-160` | works end-to-end |
| kappa_beta_per_head multi-arm sweep | sweep_route | EXISTS | `ablation.py:608-621` (5 arms); runs via `run_sweep` (`:1176`); in SWEEP_ORDER (commented) `:746` | mean held at 1.0 isolates per-head asymmetry |
| learnable per-head kappa arm | config | MISSING | grep log_kappa/learnable_kappa/Parameter.*kappa → none | correctly absent + out of scope |
| geometric-mean-tau baseline (confound control) | metric | MISSING | grep geomean/gmean/prod → only unrelated hits; only arithmetic-mean logging `config.py:1622/1634` | not auto-derived (non-blocking) |
| PPL vs per-head dispersion plotter | figure | PARTIAL | `_plot_one_sweep` (`:1353`, wired `:1484`) auto-generates a per-arm PPL bar for this sweep, but x-axis is the categorical label, not a dispersion scalar; κ list lives in per-cell JSON, not `_CSV_COLUMNS` (`:1098-1102`) | spec figure needs a small new caller |

Blocking gaps: **none**. The core experiment (live per-head knob + multi-arm sweep + an auto PPL figure) is runnable today; only the geo-mean-τ baseline arm and a dispersion-scalar x-axis figure are unbuilt, both trivial/small.

## 5. What you can run TODAY with zero new code

- **EXP-11 (H2 per-head temperature dispersion) — fully runnable as a sweep.** The
  `kappa_beta_per_head` sweep (`ablation.py:608-621`, 5 arms, mean pinned at 1.0) runs through
  the standard ablation machinery and auto-emits a per-arm PPL bar (`_plot_one_sweep`, wired at
  `:1484`). You only lose the geo-mean-τ baseline arm and the PPL-vs-dispersion-scalar x-axis —
  both small adds, neither blocks the run.
- **EXP-5's n_e_steps PPL curve (validation) — runnable as the existing `n_e_steps` sweep**
  (`ablation.py:357-359`, values `[1,2,3,5]`). You can launch it today for *validation* PPL vs
  n_e_steps; what you cannot get without a build is the value-8 cell, the straight_through arm,
  the persisted final-F, and the decorrelation analysis.
- **Any single arm of EXP-1/EXP-2/EXP-7/EXP-8** can be *executed* today as a one-off
  `train_vfe3`/ablation run (all underlying knobs are live and consumed). What is missing is the
  multi-arm *driving* and the cross-arm/cross-seed *analysis+figures* — i.e. you can produce the
  raw runs but not the headline aggregate number or its plot.

Everything else (EXP-1 headline SD, EXP-2/4/9/10 per-cell residual sweeps, EXP-3 entirely,
EXP-6 muP contrast, EXP-7 r(X) curves, EXP-8 convergence diagnostics) needs at least one build
item first.

## 6. Needs a build sprint first

- **EXP-3 (Sigma_q calibration, B1)** — the heaviest: 5 MISSING capabilities, blocked on the
  Σ↔CE join (`belief_ce_bank`), `sigma_trace`, Spearman/CV helpers, and three new plotters.
- **EXP-8 (pullback nat-grad, D1)** — 4 MISSING (steps-to-target, cos(nat,grad), pullback cond,
  wall-clock convergence) plus the gated log-spaced sweep entry.
- **EXP-9 (tied vs untied, A2)** — must author a *new* builder-break metric before it measures
  anything; the existing residual is structurally blind to the effect.
- **EXP-6 (muP, F1)** — gated on the `kl_max` confound fix *and* a new muP route before any run
  is valid.
- **EXP-1 (variance floor, I1)** — needs the across-seed aggregator and (for a truthful
  "init+optimization only" SD) the fixed data-order generator.
- **EXP-2, EXP-4, EXP-5, EXP-7, EXP-10** — each needs its `configs` sweep arm (S1) plus, for
  EXP-2/4/9/10, the shared per-cell converged-state extraction (S2) and CSV columns (S3). These
  cluster: build S1+S2+S3 once and four experiments move from NEEDS_BUILD to runnable.

The cheapest path to the most experiments runnable is the S1→S2→S3 shared cluster (multi-arm
arms + per-cell converged-state residual + CSV columns), which unblocks EXP-2, EXP-4, EXP-9, and
EXP-10 together, followed by S4 (the cross-arm/seed correlation+aggregation reader) for EXP-1,
EXP-5, and EXP-6.

## 7. Build status update (2026-06-21, overnight)

Seven commits on `feat/gauge-transport-toggle` (3eca386, c0558e1, 487c3f1, 0671215, b79f818,
eb36ce2, 558d77f) moved the matrix from READY 0 to nine experiments runnable as-is and two PARTIAL.
Full test suite green throughout (1180 passed / 0 failed); each change is additive and leaves the
user's config WIP intact. Per-experiment how-to-run and what-remains:

| Exp | Status now | Run it | Remaining (deferred) |
|-----|-----------|--------|----------------------|
| EXP-1 | DONE (2026-06-22) | `train_vfe3.py`: NUM_RUNS>1 + SEEDS + DATA_SEED, then `python multiseed_analysis.py` | — (`ppl_noise_band` auto-emits) |
| EXP-2 | DONE (2026-06-22) | `ablation.py` CONFIG["sweep"]="gauge_transport" | — (`gauge_transport_bars` auto-emits) |
| EXP-3 | DONE (2026-06-22) | metrics + `belief_ce_bank` join + reliability/stratified/scatter figures + rho/CV in research.json | — (figures auto-emit via `generate_figures`) |
| EXP-4 | DONE (2026-06-22) | `attention_entropy` 2×2 entropy×κ sweep + `attention_entropy_cov_gap` metric (cov_gap CSV col) + PPL-gap / cov-gap-vs-κ figures | — (per-sweep figures) |
| EXP-5 | DONE (2026-06-22) | `infer_T` route + persisted `estep_final_f_per_token` + `f_ce_decorrelation`/`estep_capacity` figures + the two Pearsons | — (`scaling_analysis.py` after the `infer_T` run) |
| EXP-6 | DONE (2026-06-22) | `scaling.py` CONFIG["routes"]=["grow_K","grow_K_mup"], then `scaling_analysis.py` | — (`kmup_stability` auto-emits) |
| EXP-7 | DONE (2026-06-22) | `rho_handoff` sweep + per-cell `rank_resid`/`rank_resid_by_layer` + `rank_residual_by_depth` figure | — (per-sweep `rho_handoff_rank_collapse.png`) |
| EXP-8 | DONE (2026-06-22) | gauge M-step sweeps + gated `cos_nat_phi`/`pullback_cond_*`/`wall_clock_s` in metrics.csv + `wallclock_convergence` figure (steps/wall-to-target) + LR bowl | — (per-sweep figures) |
| EXP-9 | DONE (2026-06-22) | CONFIG["sweep"]="gauge_equivariance" (per-eval val_builder_resid) | — (`gauge_residual_drift` auto-emits) |
| EXP-10 | DONE (2026-06-22) | CONFIG["sweep"]="cg_coupling" (equiv residual in CSV) | — (`ppl_equivariance_bars` auto-emits) |
| EXP-11 | DONE (2026-06-22) | CONFIG["sweep"]="kappa_beta_per_head" (geo-mean-τ arms added) | — (`kappa_dispersion` auto-emits) |

The recurring deferred theme is FIGURES (a post-hoc plotting pass over `vfe3/viz/figures.py` once
the runs produce CSVs) plus three involved infra items that each warrant their own careful pass:
the EXP-3 Sigma<->CE join, the EXP-5 final-F persistence (free_energy_value threading), and the
EXP-8 training-time diagnostics (train.py metrics hot path). All are catalogued in section 3.
See `docs/2026-06-21-edits.md` for the per-commit detail.

## 8. Build status update (2026-06-22)

The PARTIAL and RUNNABLE-with-deferred-infra experiments from section 7 (EXP-3, EXP-4, EXP-5, EXP-7,
EXP-8) are now fully built (additive only; default path unchanged; full suite green at each step). All
three "involved infra items" flagged above — the EXP-3 Sigma<->CE join, the EXP-5 final-F persistence,
and the EXP-8 training-time diagnostics — are closed.

- **EXP-3 (Sigma_q calibration, B1) — DONE.** `vfe3.viz.extract.belief_ce_bank` is the Sigma_q<->CE
  join (replays forward's belief path EXACTLY -- including the s-refine anchor and precision-bias fold
  -- so the traced covariance is the one the decode used). Three registered figures
  (`reliability_diagram`, `sigma_stratified_error`, `sigma_ce_scatter`) wired into
  `generate_figures`; `research.json` carries `sigma_ce_spearman`, `sigma_trace_cv`, and the CV>0.10
  `sigma_trace_cv_gate_pass`. (commit 3acd7ad; the s-refine/precision faithfulness fix was caught by
  an adversarial review and pinned by a `converged_state` cross-check test.)

- **EXP-7 (prior-anchoring vs rank collapse, F2) — DONE.** `ablation.py` `rho_handoff` 5-arm sweep
  (lambda_alpha x rho x e_phi_lr, n_layers=4, lambda_alpha_mode='constant'); `_cell_diagnostics`
  emits `rank_resid` (final-layer Dong r(X), a CSV column) and `rank_resid_by_layer` (per-layer curve
  in the cell JSON); `rank_residual_by_depth` figure + `_plot_rank_collapse` driver dispatched per
  sweep (no-op when absent). (commit 3acd7ad.)

- **EXP-5 (structural non-Neal-Hinton EM, C2) — DONE.** `finalize_run` persists
  `estep_final_f_per_token` (the E-step's converged target-blind functional value) to
  `test_results.json` + `summary.json`; `scaling_analysis.py` harvests it across the `infer_T`
  n_e_steps arms and emits `plot_estep_capacity` (was registered but uncalled) + the new
  `f_ce_decorrelation` figure, printing `Pearson(n_e_steps, F)` and `Pearson(F, CE)` (the structural-EM
  prediction: the former strongly negative, the latter ~0/positive).

- **EXP-4 (canonical-F vs entropy-suppressed surrogate, C1) — DONE.** `attention_entropy_cov_gap`
  (`vfe3/viz/extract.py`) measures the −τ⁻¹Cov_β(E,∇E) attention-entropy gradient gap by differencing
  the autograd oracle gradient (entropy ON vs OFF) on the converged belief; `_cell_diagnostics` emits
  it as the `cov_gap` CSV column; the `attention_entropy` sweep is the 2×2 entropy×κ grid; the
  `entropy_ppl_gap` + `cov_gap_vs_kappa` figures are driven by `_plot_attention_entropy`.

- **EXP-8 (pullback nat-grad gauge M-step + LR, D1) — DONE.** `GaugeNaturalGradAdamW` stashes GATED
  (log/eval-only, read-only) training-time diagnostics `cos_nat_phi` and the pullback-metric
  `pullback_cond_*`; `train.py` adds the cumulative `wall_clock_s` column; `wallclock_convergence`
  (`vfe3/viz/figures.py`, driven by `_plot_wallclock_convergence`) plots val PPL vs wall time with the
  steps/wall-to-target annotated; the LR-vs-PPL bowl falls out of `_plot_one_sweep` on the
  `m_phi_lr_natgrad` numeric sweep.

- **Figures tail (EXP-1/2/6/9/10/11) — DONE.** Six auto-dispatched figures wired to the standard
  entry points (no manual step): `gauge_transport_bars` (EXP-2), `gauge_residual_drift` (EXP-9, off a
  new per-eval `val_builder_resid` series), `ppl_equivariance_bars` (EXP-10), `kappa_dispersion`
  (EXP-11, + geo-mean-τ confound-control arms), `kmup_stability` (EXP-6, via `scaling_analysis`),
  `ppl_noise_band` (EXP-1, via `multiseed_analysis`). The EXP-2/9/10/11 figures emit per-sweep from
  `ablation.main()`; EXP-6 from `scaling_analysis`; EXP-1 from `multiseed_analysis`.

The entire EXP-1…EXP-11 experiment harness — config toggles, metrics, sweeps/routes, per-run
diagnostics, and the auto-generating figure set — is now built and green (1218 passed / 0 failed). See
`docs/2026-06-22-edits.md` for the per-commit detail.

- **H1 / #13 (offset-only positional extrapolation) — DONE (2026-06-22).** Beyond the top-11: the
  buildability audit's top remaining pick. The `pos_extrapolation` sweep (alibi/t5/learned/rope) +
  `_eval_at_growing_n` (CE at N up to 4× max_seq_len) auto-emit the `pos_extrapolation` CE-vs-N figure;
  both code traps fixed (the `pos_phi='learned'` table now clamps past `max_seq_len`; `t5_max_distance`
  raised in the T5 arm). The remaining hypotheses (#14-#28) are surveyed in `docs/2026-06-22-edits.md`
  / the audit: a few small builds (B2 Rényi-saturation, B3 Fisher-NG μ-arm), several runnable-now
  diagnostics (A4/A4b/E1/E2/E3), and the fenced items needing absent infra (J1 parse pipeline, F3
  standard-transformer baseline, G4 meta-ensemble sampler).
