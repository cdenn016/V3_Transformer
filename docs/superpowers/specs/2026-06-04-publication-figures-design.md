# Publication-Quality Figures and Metrics for VFE_3.0 — Design

Date: 2026-06-04
Status: design (awaiting approval before implementation)
Authoring method: 8-expert fan-out workflow (info-geometry, gauge theory, differential
geometry, variational, transformer-ML, numerical analysis, scientific visualization,
ML-experimental) → synthesis → adversarial completeness critique → finalization.

## Purpose and scope

The current visualization surface is thin: single-marker scalar trajectories
(`plot_trajectory`), a one-bar free-energy snapshot taken at the last eval that omits the
data term, single-seed perplexity line/bar ablations with no spread, and raw attention
heatmaps with no aggregate or structural statistic. This design replaces that surface with a
small set of decisive, claim-linked, publication-quality figures and the metrics they need.

The user's decisions fix the scope. The deliverable is **design plus implementation** of the
figures and metrics, wired into the existing registries (`register_metric`, `register_figure`),
`RunArtifacts`, and `ablation.py` — ready to generate, with rendering of final PNGs left to a
later run. Figures serve all four narratives: the free-energy-minimization inference mechanism,
the gauge/holonomy geometry, the belief and information geometry, and language-model rigor.
The run protocol stays **single-seed** (seed 6), so figures must be defensible without
cross-seed confidence intervals; within-run distributions (per-token, per-head, bootstrap over
tokens or validation sequences) carry all uncertainty.

Two honesty constraints govern every figure. First, the current implementation **uses
backprop** (the M-step trains the prior bank and gauge parameters with AdamW; the E-step
gradient flows via unroll or straight-through). The no-neural-network constraint is real —
there are no weight-matrix layers, and representational capacity comes from iterative
free-energy minimization over Gaussian belief tuples — but backprop is present and
acknowledged, with backprop-free inference a future goal. No figure may narrate "backprop
replaced"; the mechanism story is about the minimization dynamics. Second, the hierarchical
h→s→p→q free energy closes to the runtime total only because the model-channel couplings are
inert at the defaults (`lambda_h = 0.0`, `gamma_coupling = 0.0`); figures that sum the free
energy must state this precondition and guard against nonzero defaults.

## Architecture

New code is placed to preserve the existing contracts and the registry-behind-every-seam
pattern.

`vfe3/metrics.py` keeps its stated contract: pure measurements, no gradients, no side effects,
each registered via `register_metric` and reading its inputs from the keyword context. All new
*pure* metrics (spectra, entropies, divergences, asymmetries, invariants, bootstrap bands over
arrays already in hand) live here.

`vfe3/viz/extract.py` is a **new module** for the runners and collectors that reload a
checkpoint, rebuild beliefs, loop the E-step or the block stack, score a loader per unit, or
tally numerical-fallback events. These have side effects and/or drive the model, so they do not
belong in `metrics.py`. This is where `belief_bank`, `e_step_belief_trace`,
`across_layer_belief_trace`, `per_unit_eval_nats`, and the numerical-health fallback counters
live. They return plain arrays/dicts that the pure metrics and the figure functions consume.

`vfe3/viz/figures.py` gains the new plotters, each registered via `register_figure`, accepting
torch or numpy and detaching to numpy, returning a matplotlib `Figure` and optionally saving.
The publication style (`set_publication_style`) is extended only as needed (perceptually-uniform
and diverging colormaps, a shared ridgeline/violin helper).

`vfe3/run_artifacts.py` and `ablation.py` are the wiring seams. `_save_free_energy_bar` is
replaced by the stacked-area free-energy descent; the gauge-geometry trajectory writers are
extended; `ablation.py` gains a `grid` sweep form (Cartesian product of two value lists) and its
weak `_plot_one_sweep` / `_plot_sensitivity` are replaced by the Pareto, capacity-scaling, and
baseline-anchored ablation/sensitivity plotters.

Implementation order follows the dependency graph: the cross-shared pure metrics and the
extract-module runners are built and tested first, then the figures that compose them. The
cross-shared items are `spd_geodesic_distance` (F2, F4), `attention_entropy_rows` (F7, F8),
`guard_saturation` (F9, F13), `effective_rank_per_token` (F5, F9), `belief_spectrum` (F9, F13),
and `per_unit_eval_nats` (F11, F12).

## The figure set

Thirteen figures. Tier is honest: `must_have` means the paper's claim fails without it.

### F1 — Free-energy decomposition and co-descent (must_have, vfe_inference_mechanism)

Panel A: stacked area of the full free energy over training step (nats) — self-coupling
`alpha·KL(q||p)`, then `lambda_beta`-scaled belief-coupling and attention-entropy, then the data
term `-E_q[log p(o|x)] = val_ce` on top, so the stack closes to the runtime-realized total F (the
current single bar omits this term). Panel B: twin-axis co-descent of `free_energy_total` and
`val_ce` with the decouple-step marked; the Spearman/Pearson r is annotated only as a
co-monotonicity descriptor, captioned as unable to establish causation since both decrease
monotonically (causation is carried by F3 and F11B). Panel C: per-token violin of self-divergence
`D(q_i||p_i)` at the converged best model over held-out sequences. Mandatory closure caption: the
stack closes only because `lambda_h = gamma_coupling = 0`; `free_energy_full_decomposition` warns
if either is nonzero. New metrics: `free_energy_full_decomposition`, `self_coupling_profile`. New
figure fn: `plot_free_energy_descent`. Data: bespoke dense-eval run (`eval_interval << max_steps`).

### F2 — E-step convergence in the SPD metric (must_have, vfe_inference_mechanism)

Panel A: global `F(t)` vs inner E-step iteration with the term decomposition recomputed at each
iterate, a dashed converged-F line, and a marker at the trained `n_e_steps`. Panel B (sole owner of
per-iteration covariance motion): residuals on log-y — `r_mu(t)`, the affine-invariant SPD step
`r_sigma(t)`, and `r_phi(t)`, each with a 10–90 token band. Panel C: per-token `F_i` ridgeline
contracting as t grows. Caption discloses that `free_energy_value` uses flat transport, so under
regime_ii the overlay is a flat diagnostic. New metrics (extract): `e_step_belief_trace`; (pure)
`spd_geodesic_distance`, `estep_residuals`, `e_step_convergence_trace`. New figure fn:
`plot_estep_convergence`. Data: bespoke off-path E-step loop on the reloaded model.

### F3 — The gauge does the work: ln(3) symmetry-breaking (must_have, vfe_inference_mechanism)

The mechanism intervention on the synthetic period-3 stream. Panel A: `val_ce` vs step for two
arms differing only in the gauge — frozen (`e_phi_lr = 0`, the plain-averaging symmetry) vs
learned (`e_phi_lr > 0` **and** `m_phi_lr > 0` together, since `e_phi_lr` gates the inner update
and the baseline `m_phi_lr` alone is insufficient) — with the analytic floor `CE = ln 3 = 1.0986`
nats drawn as a dashed reference. The frozen arm pins at the deterministic floor (robust under a
single seed by construction); the learned arm reaches lower CE, the gap annotated honestly as one
single-seed observation (do not dramatize). Panel B: signed transport asymmetry
`A_ij = ||Omega_ij − Omega_ji||_F` over the converged grid (near-zero for frozen) on a centered
diverging colormap. Panel C: per-head period-3-match and prev-token scores, grouped frozen vs
learned, with the winning learned head's beta heatmap inset. New metrics: `transport_asymmetry`,
`structured_head_scores`, `energy_directedness`. New figure fn: `plot_ln3_symmetry_breaking`. Data:
two `dataset=synthetic-period3` cells, dense eval.

### F4 — Belief trajectories across E-step iterations and layers (must_have, belief_infogeom)

User-required. Panel A (path view): shared-PCA 2D quiver of belief means `mu_i` across E-step
iterations (arrows `mu_t → mu_{t+1}` colored by token position), priors `p_i` marked as targets,
the PCA basis fit on all captured states so arrows are comparable across frames. Panel B
(across-layers): cumulative affine-invariant geodesic distance `d_AI(Sigma^(0), Sigma^(l))` and
mean effective rank vs layer index, median plus 10–90 token band. Mandatory coverage caption: the
user requested iterations, layers, and across-training; this covers iterations (A) and layers (B);
across-training is out of scope here (it needs per-checkpoint belief replay) and is stated, not
overclaimed. The per-iteration cumulative SPD-length panel is cut as the cumsum of F2 Panel B. New
metrics (extract): `e_step_belief_trace`, `across_layer_belief_trace`; (pure)
`spd_geodesic_distance`. New figure fn: `plot_belief_trajectories`.

### F5 — UMAP semantic clustering of beliefs: the mu / Sigma / phi triptych (must_have, belief_infogeom)

User-required, **amended to all three belief components**. A three-channel triptych, each channel a
shared 2D UMAP over many held-out token-beliefs aggregated into one bank (thousands of tokens, not
one length-N sequence), each colored by token identity (top-K frequent plus "other"), corpus
frequency decile, and optional POS tag, with silhouette and Calinski-Harabasz computed in that
channel's **native** space (not the 2D projection). The channels embed faithfully to their
geometry: `mu` (N, K) directly under Euclidean UMAP — the content channel; `Sigma` via the
log-Euclidean chart, the log-variances for the diagonal default and `vech(log Sigma)` for full
covariance — the uncertainty-geometry channel; `phi` (N, n_gen) directly — the gauge/positional
channel, expected to surface positional structure distinct from `mu`'s content structure. The `mu`
panel additionally renders marker size by Fisher trace so confident beliefs read solid; a
per-cluster effective-rank ridgeline shows beliefs are not collapsed. The three-channel layout lets
a reader see what content, uncertainty, and gauge each encode. New metrics (extract): `belief_bank`
(collects mu, Sigma, **and** phi with token ids and sequence index); (pure) `fisher_trace`,
`effective_rank_per_token`, and a `belief_log_chart` helper that maps Sigma/phi to their UMAP
feature space. Reuses the currently-orphaned `umap_embed` and `clustering_metrics`. New figure fn:
`plot_belief_umap` (renders the triptych; one call per channel via a `channel` argument, composed
into the multi-panel figure).

### F6 — Gauge-equivariance certificate (must_have, gauge_holonomy)

Panel A: log-scale ECDF of the relative residual of `E_ij` and `beta_ij` after applying random
in-group elements `g = exp(sum_a c_a G_a)` (per head for block_glk) to the converged beliefs and
recomputing — in-group residuals cluster near float32 eps, a matched out-of-group control sits far
right, a vertical line at machine eps. Panel B: median residual vs gauge-step magnitude `||log g||`
(flat across the group), plus, only when `use_head_mixer` or the Killing preconditioner is on, a
second series showing the documented equivariance drift growing with the mixer norm (shown
explicitly, never silently included). New metric: `gauge_equivariance_residual`. New figure fn:
`plot_gauge_equivariance`.

### F7 — Per-head gauge specialization and its link to attention (strong, gauge_holonomy)

Panel A: per-head ridgelines over tokens of the group-correct gauge invariant — volume
`log|det exp(phi^(h))|` for glk/block_glk, anisotropy `s_max/s_min`, frame displacement — with the
invariant dispatched on the group (total rotation angle for so_k, symplectic squeeze for sp), since
det-based metrics are identically one for the unimodular groups (the reason the existing
`gauge_trace_spread` is blind there). Panel B: scatter of the L·H heads, gauge magnitude vs
attention row entropy `H(beta^(h))` (from the shared `attention_entropy_rows`), colored by layer,
with the uniform-entropy ceiling and per-token bootstrap bars. New metrics: `group_gauge_invariant`,
`per_head_gauge_invariants`, `attention_entropy_rows`. New figure fn: `plot_gauge_head_specialization`.

### F8 — Attention structure: entropy, head divergence, distance decay (strong, vfe_inference_mechanism)

Panel A: per-head row-entropy ridgelines vs the causal-uniform reference (does attention sharpen
below uniform — the honest mass-concentration signal). Panel B: per-layer head-redundancy heatmap of
mean row-wise Jensen-Shannon divergence between heads. Panel C: per-head attention-vs-offset profile
`beta_bar(d)` on log-y with a 68% bootstrap-over-tokens band, plus a per-head positional-vs-content
`R^2` placing each head on a positional↔content axis. New metrics: `attention_entropy_rows`
(shared with F7), `head_redundancy_js`, `attention_distance_decay`, `positional_content_score`. New
figure fn: `plot_attention_structure`.

### F9 — Belief covariance geometry (strong, belief_infogeom)

Panel A: per-token effective-rank violin over the N tokens (the distribution the current `.mean()`
collapse discards), the logged mean overlaid. Panel B (sole owner of the guarded eigenvalue scree):
scree ridgeline of normalized eigenvalues on log-y with a 10/90 band, the `eps`/`sigma_max`
retraction guard lines, and the saturated-zone shading from `guard_saturation`. Panel C: per-token
spectral condition number histogram with a well-conditioned reference band. Panel D: SPD covariance
ellipses for a chosen 2D plane from the eigen-decomposed 2×2 sub-block (orientation and axes from
eigenvectors, not the current diagonal-only width/height), colored by effective rank or
self-divergence. New metrics: `effective_rank_per_token`, `belief_spectrum`, `guard_saturation`. New
figure fns: `plot_belief_spectrum`, `plot_spd_ellipses` (replaces `plot_covariance_ellipses`).

### F10 — Holonomy and curvature with corrected sampling (strong, gauge_holonomy)

Panel A: per-triangle holonomy `||Omega_ij Omega_jk Omega_ki − I||_F` over random/stratified triples
(fixing the first-512 row-major low-index bias), a split violin — the flat default path spikes at eps
(the cocycle closes exactly, shown as the expected flatness certificate, not a failure), the opt-in
regime_ii path spreads to genuine curvature — with a bootstrap-over-triples CI and the old biased
estimator's single value overlaid. Panel B: holonomy vs triangle span `max|i-j|`. Panel C: regime_ii
spatial curvature field for a fixed anchor. Labeled throughout as a Regime-II quantity. New metrics:
`holonomy_deviation_sampled`, `curvature_field`. New figure fn: `plot_holonomy_curvature`.

### F11 — Capacity and efficiency (must_have, lm_rigor)

Panel A: capacity scaling — `final_val_bpc` (PPL secondary axis) vs `embed_dim`, `n_heads`,
`n_layers`, each with a bootstrap-over-validation-sequences band (the legitimate single-seed
uncertainty, captioned as not a cross-seed CI). Panel B (the at-scale causal claim for the
mechanism): val BPC and converged `free_energy_total` vs `n_e_steps` on a dual axis with a flat-params
annotation and a wall-time inset — more inner free-energy minimization lowers loss at near-constant
parameter count, capacity from inference. Panel C: Pareto frontier of BPC vs `n_params` and vs
wall-time, dominated cells faded. New metrics (extract): `per_unit_eval_nats`; (pure)
`bootstrap_ce_band`. New figure fns: `plot_capacity_scaling`, `plot_estep_capacity`,
`plot_pareto_frontier`. Data: existing capacity and `n_e_steps` sweeps plus the new per-unit eval.

### F12 — Single-seed ablation panel and joint LR sensitivity (strong, lm_rigor)

Panel A: baseline-ladder ablation forest of delta-BPC from the full model for each disabling ablation
(frozen vs learned gauge, surrogate vs canonical F, uniform vs causal prior, flat vs learned
transport, the gauge groups), dot plus paired bootstrap-over-tokens interval, sorted by effect size.
Panel B: 2D joint LR sweep heatmaps for `(m_mu_lr × m_sigma_lr)` and `(e_mu_lr × e_sigma_lr)`, val PPL
color, basin minimum starred, operating point marked — exposing ridge interactions the independent 1D
sweeps hide. New metrics (extract): `per_unit_eval_nats`; (pure) `bootstrap_token_ce_band`. New figure
fns: `plot_ablation_forest`, `plot_lr_grid_heatmap`. Requires a `grid` sweep form in `ablation.py`.

### F13 — Numerical-trust panel (nice, belief_infogeom)

Panel A: guard saturation — references F9 Panel B for the Sigma-eigenvalue scree and adds the
complementary boundaries (histograms of `E_ij` and `D(q_i||p_i)` with the `kl_max` line, the fraction
pinned at each boundary). Panel B: incidence map of non-finite intermediates and numerical-fallback
activations (safe_cholesky jitter rounds, pinv fallbacks, eigenvalue-floor hits) over checkpoints.
Panel C: causal-mask and row-stochastic sanity — future-attention leakage (must be zero), row-sum
deviation, active-set growth slope, per head. New metrics: `guard_saturation` (shared with F9),
`numerical_health` (extract), `causal_sanity` (pure). New figure fn: `plot_numerical_trust`.

## New metrics master

Pure measurements (in `metrics.py`): `spd_geodesic_distance`, `estep_residuals`,
`e_step_convergence_trace`, `free_energy_full_decomposition`, `self_coupling_profile`,
`transport_asymmetry`, `energy_directedness`, `structured_head_scores`, `fisher_trace`,
`effective_rank_per_token`, `belief_spectrum`, `gauge_equivariance_residual`,
`group_gauge_invariant`, `per_head_gauge_invariants`, `attention_entropy_rows`,
`head_redundancy_js`, `attention_distance_decay`, `positional_content_score`,
`holonomy_deviation_sampled`, `curvature_field`, `bootstrap_ce_band`, `bootstrap_token_ce_band`,
`guard_saturation`, `causal_sanity`, plus the `belief_log_chart` helper for F5.

Runners and collectors (in `vfe3/viz/extract.py`): `per_unit_eval_nats`, `e_step_belief_trace`,
`across_layer_belief_trace`, `belief_bank`, `numerical_health`.

Formulas are recorded per metric in the workflow output and will be carried into the docstrings
(the project convention places the LaTeX/math form in the docstring of every non-trivial kernel).

## Defects fixed as part of this work

The work fixes nine real defects, each verified against the code by the experts:
`evaluate()` retaining no per-unit nats (blocks every single-seed bootstrap band — the largest
feasibility gap); `holonomy_deviation`'s first-512 row-major triple bias and bare-mean collapse;
`effective_rank(...).mean()` discarding the per-token distribution; `diagnostics` computing then
discarding per-token `self_div`/`alpha`; the free-energy bar omitting the data term, the time axis,
the `lambda_beta` scaling, and within-run spread; `attention_entropy` collapsing across heads, layers,
and rows; `plot_covariance_ellipses` building diagonal-only axis-aligned ellipses with no correlation;
`_plot_one_sweep`/`_plot_sensitivity` discarding params/wall-time/BPC and using a diverged-arm-dominated
absolute range; and `gauge_trace_spread` being identically zero (det-blind) for the unimodular groups.
The pre-existing `gauge_trace_spread` column in `metrics.csv` is left in place (not deleted);
`group_gauge_invariant` supersedes it in figures.

## Data preconditions for rendering

Because the deliverable is implement-not-render, these are documented as what a later run supplies.
Every over-training panel (F1, F11B) needs a dense-eval run with `eval_interval << max_steps`, since
the default `eval_interval == max_steps` yields a single `metrics.csv` row. F3 needs two
`synthetic-period3` cells (frozen and learned gauge). The capacity and ablation figures (F11, F12)
need the existing sweeps plus the new per-unit eval pass on each cell's reloaded best model. The
belief-geometry figures (F2, F4, F5, F9) reload `best_model.pt` and recompute converged beliefs
off-path; F4 Panel A benefits from a larger `n_e_steps` than the baseline 1.

## Testing

Every new metric and figure gets a unit test on synthetic tensors in `tests/test_metrics.py` and
`tests/test_viz.py`. Property tests double as correctness proofs: `spd_geodesic_distance(S, S) == 0`
and symmetry/triangle behavior; `gauge_equivariance_residual` near eps in-group and large
out-of-group; `holonomy_deviation_sampled` near eps on a flat cocycle; `attention_entropy_rows`
reducing to the existing `attention_entropy` under mean; `bootstrap_*_band` covering the point
estimate; figure functions returning a `Figure` and saving without error on small inputs. Pass counts
are read from `--junitxml` or the `N passed` line, never from memory, and no extra `-q` is added.

## Scope tiers for implementation

The full set is 13 figures, roughly 30 new metrics/runners, and roughly 17 figure functions — a
multi-day implementation. Three reasonable scopes:

Minimum honors the two explicit requests plus one must_have per claim and the defect fixes:
F4 (trajectories), F5 (UMAP triptych), F1 (mechanism), F3 (ln 3), F6 (gauge certificate),
F11 (capacity), and the metrics/defects those need.

Must-have tier is the seven `must_have` figures: F1, F2, F3, F4, F5, F6, F11.

Full is all thirteen (F1–F13), the complete plan.

## Cut list (scope boundaries)

The synthesis deliberately cut, with reasons recorded: the model-channel / hierarchical s figure
(scoped out — the paper is positioned on the q-channel and the s-couplings are inert at defaults;
reinstate if the hierarchy becomes a claimed contribution); belief trajectories across training
(needs per-checkpoint replay; F4 covers iterations and layers and captions the gap); the redundant
F4 cumulative-SPD-length panel; the Renyi-alpha attention-energy geometry; the mean-vs-covariance
energy decomposition; the state-dependent alpha hyperbola (conditional on a non-default alpha mode);
the standalone `E_ij` substrate heatmap; the canonical-vs-surrogate entropy-gradient gap (a theory
supplement); attention-as-token-graph centrality; the per-head free-energy budget; attention
evolution over training; the retraction-geometry comparison; Fisher natural-gradient anisotropy; the
geodesic-distance matrix; and the standalone sensitivity tornado (merged into F12).
