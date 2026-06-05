# 2026-06-05 — Rényi divergence investigation: ablation confound + latent alpha>1 fix

Triggered by the user's report that `alpha_div < 1` trains ~2.5x faster than `alpha_div = 1`
and that the `vfe3_ablation_results/alpha_div` sweep shows Rényi at much worse validation
perplexity (~259-273 vs ~159 for KL). Four expert agents investigated (implementation,
information-geometry, numerical-analysis, variational). Full findings in
`docs/audits/audit-2026-06-05.md`; math verification logged in `docs/verified.md`.

## Diagnosis (no code defect in the divergence)

The Rényi math is correct (diagonal, full-cov, per-coord all verified against van Erven &
Harremoës 2014 / Gil 2013). The `alpha_div` ablation is confounded by a gradient-routing
asymmetry: `alpha_div = 1` uses the always-live analytic hand kernel
(`vfe3/gradients/kernels.py:179-194`), while `alpha_div != 1` falls back to the autograd
oracle, which under the default `oracle_unroll_grad=False` returns a detached belief
gradient (`vfe3/gradients/oracle.py:118`, gated by `vfe3/inference/e_step.py:357` +
`vfe3/config.py:247`). With the ablation's `e_phi_lr=0`, that detachment severs the prior
and gauge-frame tables from the loss, producing both the speedup (dropped backward graph)
and the degraded perplexity (untrained priors/frames). The cleanest proof is the sweep's
own `alpha_div=0.99` row, which jumps discontinuously to PPL 273 / 695s exactly at the
kernel-gate boundary — impossible for a continuous divergence-order effect. The pure
live-gradient path already exists via `oracle_unroll_grad=True`, so this is a
dangerous-default / confounded-ablation issue, not a missing-pure-path one.

## Changes

`ablation.py` — the `alpha_div` sweep entry now carries
`"requires": {"oracle_unroll_grad": True}` so every `alpha_div != 1` cell runs the live
oracle, making the sweep a clean single-variable divergence-order comparison rather than a
gradient-truncation comparison. This is a no-op at `alpha_div = 1` (the kernel ignores the
toggle), so the KL baseline is unchanged. (The file also carries the user's concurrent
sweep operating-point tuning.)

`vfe3/families/gaussian.py` — latent-bug fix (found by the numerical agent, not the user's
α<1 regime). The diagonal Rényi closed form and per-coordinate form previously did
`sigma_blend = (...).clamp(min=eps)`, which for `alpha > 1` silently turns a non-positive
(indefinite) blend into a tiny positive variance and emits a wrong finite divergence with
a nonzero gradient, escaping the `kl_max` sentinel that the full-covariance path correctly
uses via `safe_cholesky`. Both branches now build the raw blend, clamp only to guard
log/division on the in-bounds coordinates, then map non-positive-blend elements to
`NaN -> kl_max` (summed: any non-PD coordinate masks the whole pair; per-coord: only the
bad coordinate is masked). For `alpha in (0,1)` the blend is a convex combination and
always positive, so the mask is inert and the path is byte-identical (the user's regime is
unaffected).

`tests/test_divergence.py` — two new tests (TDD, written failing first):
`test_diagonal_renyi_alpha_gt_one_negative_blend_masks_to_kl_max` (summed) and
`test_diagonal_renyi_per_coord_alpha_gt_one_masks_only_bad_coord` (per-coordinate), mirroring
the existing full-cov `alpha > 1` mask tests.

`docs/verified.md`, `docs/audits/audit-2026-06-05.md` — verification log and full audit.

## Verification

Full suite: 575 tests, 574 passed + 1 xpassed, 0 failures, 0 errors (junit XML, CPU box).
The new tests fail against the old `clamp(min=eps)` code (returned 4.73 / 4.61 instead of
kl_max) and pass after the fix. The ablation override is wired:
`make_run_overrides('alpha_div')` yields `{'oracle_unroll_grad': True, 'alpha_div': 0.99}`.

## Not done (needs the GPU)

The corrected sweep must be RERUN on the user's RTX 5090 (this box is CPU-only) to get the
valid divergence-order numbers: `python ablation.py` with the `alpha_div` sweep. Expect
Rényi to be slower than KL (autograd vs analytic kernel) and the perplexity gap to shrink.
Two real non-bug effects mean exact parity is not guaranteed even after the fix: the
state-dependent self-coupling `alpha^(k) = c0/(b0 + D)` is larger for `alpha < 1`
(Rényi D <= KL), and attention softens at fixed `tau = kappa*sqrt(K)` (retune kappa). For
`alpha != 1`, F is a heuristic consensus functional, not an evidence bound (author-disclosed,
`GL(K)_attention.tex:771`).

---

# 2026-06-05 — Wire up the publication figures/metrics expansion (driver) + free_energy_descent fix

Triggered by the user's report that "the recent expansion of figures/plots/metrics was never
wired up." Investigation confirmed it: the 2026-06-04 expansion (PR #26) added ~24 pure metrics
(`vfe3/metrics.py`), 5 model-replay runners (`vfe3/viz/extract.py`), and 17 figures
(`vfe3/viz/figures.py`), but tracing every caller outside `tests/` showed only ONE figure
(`free_energy_descent`) was ever driven (from `run_artifacts._save_figures`). The other 16
figures, all 5 extract runners, and all ~24 new metrics were reachable only from the unit tests —
a tested library with no driver. The figures split by data dependency: ~10 are single-run
(model + a few eval batches), 5 are sweep-level (belong to `ablation.py`), and 1
(`ln3_symmetry_breaking`) is a two-arm frozen-vs-learned experiment. The user chose to ship the
single-run driver only (ablation F11/F12 wiring stays deferred) and to fix the
`free_energy_descent` correctness bug from `audit-2026-06-05-new.md` (Findings 2/3).

## Changes — the missing driver

`vfe3/viz/extract.py` — new `converged_state(model, token_ids)` runner. The gauge/spectrum/
numerical figures need the converged-belief TENSORS (mu, sigma, phi, the per-token vertex factor
`exp_phi`, the pre-rope pairwise transport `omega`, energy, beta, self_div), which the scalar
`model.diagnostics` computes internally but discards. This mirrors `diagnostics` EXACTLY (same
active config: transport mode, connection_W, rope, family, divergence, alpha) and returns the
tensors. It is the single source feeding `gauge_equivariance_residual`,
`per_head_gauge_invariants`, `belief_spectrum`/`spd_ellipses`, and `guard_saturation`.

`vfe3/viz/report.py` (NEW) — `generate_figures(run_dir, ...)`, the single-run driver. Rebuilds
the trained model from `config.json` + `best_model.pt` (or drives a live model passed in), builds
a stable unshuffled loader (synthetic fallback when the cache is absent), runs the extract runners
once, and writes the 10 single-run figures to `run_dir/figures/`: estep_convergence,
belief_trajectories, belief_umap, attention_structure, gauge_equivariance,
gauge_head_specialization, belief_spectrum, spd_ellipses, holonomy_curvature, numerical_trust.
OPT-IN and OFF the training hot path (the runners are expensive: UMAP, E-step replay, holonomy
sampling, a belief bank over many sequences), so it is a separate step, never auto-run by
`train`/`finalize_run`. Each figure is best-effort (a plotting/dependency/shape error is logged
and skipped, mirroring `RunArtifacts._save_figures`), and each expensive input is guarded so a
failure skips only its dependent figures.

`make_figures.py` (NEW, repo root) — click-to-run entry point (edit `CONFIG`, run). Points at a
run directory (None -> newest under `vfe3_runs`), rebuilds the model, and calls `generate_figures`.

## Changes — free_energy_descent correctness fix (audit-2026-06-05 Findings 2/3)

`vfe3/train.py` — the diagnostics free-energy terms are per-sequence SUMS over seq 0, but
`val_ce` is a token-weighted MEAN (nats/token); stacking them made the data term an invisible
sliver and the "closes to F" claim dimensionally false. The four F-stack terms
(self_coupling, belief_coupling, attention_entropy, free_energy_total) are now normalized to
PER TOKEN (divided by the sequence length) at the logging site, so the CSV and the stack are
commensurate with `val_ce`. (Finding 2 reduction mismatch.)

`vfe3/viz/figures.py::plot_free_energy_descent` — panel B previously plotted
`history["free_energy_total"]` (the coupling-only runtime total, which EXCLUDES the CE data term)
while panel A stacked the data term in, so the two panels disagreed. Panel B now plots the SAME
data-term-inclusive stacked total as panel A. Units relabeled to nats/token; docstring made honest
(descriptive snapshot-vs-aggregate, literal closure only at lambda_h = gamma = 0). A per-row
`lambda_beta` vector is now accepted (it already broadcast in the stack). (Finding 2 A/B
inconsistency.)

`vfe3/run_artifacts.py::_save_figures` — on a `learnable_lambda_beta` run the figure was scaled
by the static `cfg.lambda_beta` scalar, ignoring the logged learned trajectory. It now passes the
row-wise `lambda_beta` vector when every free-energy row carries it, falling back to the config
scalar on constant-coupling runs. The now-unused `free_energy_total` injection is dropped.
(Finding 3.)

## Tests

`tests/test_report.py` (NEW, 3 tests): `converged_state` shapes/finiteness; the driver against a
live model writes the 10-figure set (PNGs nonempty on disk); the driver reload path
(config.json + best_model.pt -> rebuilt model -> figures). The existing smoke tests for the
figures themselves (`tests/test_viz.py`) and extract runners (`tests/test_extract.py`) are
unchanged and still pass (the figure-side change is backward compatible: `plot_free_energy_descent`
still accepts a `free_energy_total` key, it just no longer uses it).

## Verification

Full suite: junit XML `tests=583 failures=0 errors=0 skipped=0` (582 passed + 1 xpassed, CPU box).
End-to-end real run (throwaway, cleaned up): a tiny `learnable_lambda_beta=True` training ->
`finalize_run` -> `generate_figures` produced all 10 driver figures (116-192 KB each) plus the
regenerated `free_energy_descent.png`; `metrics.csv` confirmed per-token F-terms
(belief_coupling ~0.0025/token vs val_ce ~1.78/token, now commensurate) and a varying
`lambda_beta` column (1.015 -> 1.043) exercising the per-row-vector branch. Visual check of
`free_energy_descent.png` (per-token units, data term visible and dominant, panels A/B agree) and
`numerical_trust.png` (real 3-panel plot) confirmed genuine output.

## Not done (by design / user decision)

Sweep-level figures (capacity_scaling, estep_capacity, pareto_frontier, ablation_forest,
lr_grid_heatmap) into `ablation.py` — DEFERRED per user (single-run driver only this pass).
The two-arm `ln3_symmetry_breaking` needs a frozen-gauge + learned-gauge experiment, not glue.
Most of the new metrics are now driven through the 10 wired figures, but a subset remains DARK
until the deferred figures land: `structured_head_scores` / `transport_asymmetry` (only the ln3
figure consumes them), the bootstrap CE bands `bootstrap_ce_band` / `bootstrap_token_ce_band`
(sweep figures), and `curvature_field` (the optional `holonomy_curvature` panel the single-run
driver does not pass). None is surfaced in the periodic per-step CSV diagnostics.

Driver robustness was spot-checked across the non-kernel configs (the autograd-oracle-under-no_grad
path, historically fragile): `alpha_div=0.5` diagonal, full-cov `alpha_div=0.5` (linear decode)
both produced 10/10 figures. The per-row `lambda_beta`-vector branch is covered only by the
end-to-end run above, not a retained unit test (low priority).

## Follow-up (user feedback same day): autorun, CSV cadence, loader-bug fix

The user reported the figures "still aren't wired up", pointed at a real run dir, and asked for
(1) autorun at end of training and (2) metrics.csv every `log_interval` (their typical cadence is
log 100-200, eval 1000-3000).

- **Loader bug (the real "still not wired up")**: `report._build_loader` imported
  `synthetic_period3_loader` from `vfe3.data.datasets`, but that function lives in the click-to-run
  `train_vfe3.py`, not the package — so `generate_figures` crashed with `ImportError` whenever it
  had to build its own loader (the tests/smoke always passed `loader=` explicitly, hiding it).
  Replaced with an inline `_synthetic_loader` fallback (a tiny period-3 `TokenWindows` loader) and a
  `make_dataloader(..., shuffle=False, drop_last=False)` primary path. Verified by producing 10/10
  figures on the user's real `vfe3_runs/...wikitext-103_K20_block_glk_linear_mix` run dir.
- **Autorun**: `finalize_run` now calls `generate_figures(run_dir, model=best, loader=test_loader)`
  at the end of training, gated by a new `VFE3Config.generate_figures` toggle (default True;
  opt-out for fast/ablation runs). Best-effort, off the hot path, drives the reloaded best-val
  model. `ablation.py` does not call `finalize_run`, so sweeps are unaffected.
- **metrics.csv cadence**: the row write moved from the eval block to a `do_csv = do_log or do_eval`
  block — a row every `log_interval` (and every eval), with the dense per-step diagnostics. The
  validation columns carry the MOST RECENT eval forward (NaN until the first eval; fresh on a step
  where the eval just ran). Diagnostics are now computed ONCE per logged step and reused for both
  the console line and the CSV (previously computed twice when both fired). `run_artifacts` figure
  helpers made NaN-robust (the descent figure and val_ppl curve skip pre-first-eval NaN rows).
- Tests: +3 in `tests/test_report.py` (`test_finalize_autoruns_figures`,
  `test_finalize_skips_figures_when_disabled`, `test_metrics_csv_logs_at_log_cadence`). Full suite
  583 passed / 0 failures; end-to-end smoke confirmed CSV rows at steps [2,4,6,8] with val NaN ->
  fresh -> carried, and `figures/` (10) + `free_energy_descent.png` auto-written at finalize.

# 2026-06-05 — Codex deep-audit triage + fixes (F1, F4, F6, F7, F8)

Triggered by `audit-2026-06-05-new.md` (Codex's 8-finding deep audit, branch
`codex/deep-audit-20260605`). Four expert agents verified every finding against current source;
full triage in `docs/audits/audit-2026-06-05-codex-triage.md` (two findings overstated, two carry
false-positive sub-claims, none a training-loop correctness bug). The user selected F1, F4, F6, F7,
F8 for repair; F2/F3 (the descent figure) were handled separately in the figures-wiring work above.
Branch `fix-codex-audit-2026-06-05`; full suite 583 passed / 0 failures (junit XML, CPU box).

`vfe3/data/datasets.py`, `train_vfe3.py` (F1) — `make_dataloader` gained a `drop_last` parameter
(default True = train regime); `_select_loader` now requests `shuffle=False, drop_last=False` for
validation/test (a stable, whole-split corpus metric) and `True/True` only for train. Previously all
splits inherited `shuffle=True, drop_last=True`, so the held-out test PPL was a randomly-varying ~97%
subset and the `evaluate` docstring's "partial last batch" was contradicted. Tests:
`test_data.py::test_make_dataloader_eval_keeps_tail_and_is_sequential`,
`test_train.py::test_select_loader_is_split_aware`.

`vfe3/config.py` (F4) — the four exactly-matching static tuples (gauge_group, alpha_mode, attention
priors, norms) now validate against `tuple(sorted(_REGISTRY))`, matching the transport/retraction
siblings, so a newly registered variant is a valid config value without editing the validator
(add-by-registering). The orphaned `_VALID_*` constants were removed; decode/encode stay explicit
second-gates (`linear`/`gauge_fixed` are reached via use_prior_bank / are NotImplementedError stubs,
not registry oversights — Codex's "decode_mode=linear blocked" was a false positive). Tests:
`test_config.py::test_gauge_group_validation_reads_registry_not_static_list`,
`test_decode_mode_linear_stays_a_rejected_second_gate`.

`CLAUDE.md`, `tests/test_regime_ii.py` (F6) — documented the trained-`connection_W` gauge-equivariance
break (only W=0 is gauge-invariant; g_i^T W^a g_j = W^a forces W=0), parallel to the head-mixer caveat,
with a characterization test pinning that the edge factor is invariant at W=0 and deviates monotonically
with ||W||. Theory entry in `docs/verified.md`.

`vfe3/model/model.py` (F7) — the periodic holonomy diagnostic switched from the row-major low-index
triangle prefix to `holonomy_deviation_sampled(...)["mean"]` (seeded random distinct triples); the
dict key is unchanged so existing diagnostics tests hold.

`train_vfe3.py` (F8) — the click-to-run banner now prints `attention_tau(cfg.kappa,
model.group.irrep_dims)` (group-aware) instead of the per-head `cfg.tau`, which understated the active
temperature by sqrt(n_heads) for single-block groups (glk/so_k/sp). Matches the in-package banner;
reporting-only. Distinct from the PR #27 gamma-tau (tau_gamma) fix.

# 2026-06-05 — Belief UMAP / semantic-clustering figure redesign

Branch `vfe3-belief-umap-figure-2026-06-05` (fresh from main). The old `belief_umap` was a single
mu/Sigma/phi triptych colored by raw `token_id` via `cmap="tab10"` (so colors were arbitrary
token-id magnitude, no legend). Brainstormed design (user-approved): separate per-channel figures,
a real legend, labeled tokens, and quantitative semantic-clustering data, colored by LINGUISTIC
CATEGORY (both BPE-structure and function/content taxonomies).

`vfe3/viz/figures.py`:
- `clustering_metrics` gained a `sample_size` kwarg (silhouette is O(N^2); the new figures call it
  per channel x taxonomy, so it subsamples; CH stays full). Backward compatible (default None).
- New helpers: `_bpe_category` (punctuation / number / word-start lc / word-start Cap / continuation
  subword / whitespace, from the decoded gpt2 token's structure), `_funccontent_category`
  (punctuation / number / function-word via a built-in ~140-word stopword set / content-word / other),
  `_token_category_labels` (decode each unique id once -> category index), `_scatter_by_category`
  (per-category colored scatter + legend with counts; greys out when no decoder), and
  `_annotate_frequent_tokens` (mark + label the N most frequent tokens at their occurrence-centroid).
- `plot_belief_umap` rewritten: now ONE channel per call (`channel="mu"/"sigma"/"phi"`), a 1x2 panel
  (BPE | function/content), colored by category with a legend, the ~10 most frequent decoded tokens
  annotated, and each panel title reporting silhouette + CH of the CATEGORY labels in the channel's
  NATIVE space (so the number measures linguistic separation, not arbitrary token id). `decode=None`
  -> single-colour fallback. No external deps (tiktoken already present for gpt2; sklearn for silhouette).
- New `plot_belief_category_separation`: grouped silhouette bars per channel x taxonomy -- the
  quantitative companion ("how strongly is each belief channel organized by each taxonomy").

`vfe3/viz/report.py`: derives `decode = datasets.get_tiktoken_decoder(dataset)` (gpt2 for wikitext;
None on synthetic/absent -> greys out), emits `belief_umap_mu/sigma/phi.png` (3 files) +
`belief_category_separation.png` instead of the single triptych; the final count log is no longer
hardcoded to 10 (now 13 single-run figures).

Tests: `test_viz.py` belief_umap test rewritten to the per-channel signature; added category-helper
unit tests, a no-decode fallback test, and a category-separation test (`test_viz.py` 32 passing with
`test_report.py`). Verified on the real `vfe3_runs/...wikitext-103_K20_block_glk_linear_mix` run:
13 figures, with `mu` showing the strongest function/content separation (~0.08 silhouette) and
frequent function words (`the`, `to`, `of`, `a`, `in`, `and`) forming tight peripheral clusters.
