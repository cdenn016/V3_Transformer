# Lightweight UMAP Animation Design

## Status and decision

This design adds opt-in UMAP animations for two distinct trajectories: representation change over training and belief/model-state change across E-step iterations. Each active feature channel receives a separate GIF. The user approved lightweight native-feature snapshots, a joint balanced UMAP fit, temporally matched HDBSCAN colors, and persistent semantic-probe markers.

The feature is intended for future runs. It does not attempt to reconstruct trajectories from a run that retained only its terminal or sparse model checkpoints. It extends the current controlled UMAP and semantic-probe reporting contracts without changing the model, free-energy objective, optimizer, training data order, or E-step equations.

## Scientific contract

An animation is a diagnostic view, not evidence by itself that semantic structure emerged. UMAP supplies the display coordinates. HDBSCAN assignments and semantic-probe measurements are computed through the existing controlled native-chart/PCA path and never from the two-dimensional display coordinates.

Every frame in one animation uses the same token occurrences, native chart, feature scaler, UMAP reducer, display limits, and semantic manifest. Independently fitted per-frame axes are forbidden because projection drift would be visually indistinguishable from representation drift. Belief `phi` remains a gauge-fixed coordinate diagnostic rather than a gauge-invariant observable, and every associated sidecar and frame footer states that limitation.

The five requested channels are belief `mu`, belief `sigma`, belief `phi`, model `mu`, and model `sigma`. Model channels are emitted only when the model channel is active. Model `phi` is outside this version: it is held fixed during the model-channel E-step and was not part of the requested model-state outputs.

## Configuration

The feature is controlled independently of the heavier static publication-figure suite:

```python
generate_umap_gifs        = False   # master opt-in; no snapshot or rendering work when false
umap_snapshot_every_evals = 1       # capture every N validation evaluations
umap_snapshot_max_tokens  = 2048    # cap for the fixed diagnostic population
```

`generate_umap_gifs=False` is the default and preserves the existing path. `umap_snapshot_every_evals` must be positive, and `umap_snapshot_max_tokens` must be large enough for the controlled UMAP/HDBSCAN path. The master toggle is independent of `generate_figures`, so a run can request lightweight animations without requesting the full publication figure set.

## Fixed diagnostic population

When animation capture is enabled, training freezes one deterministic population from the stable validation loader before the first optimizer step. The population contains at most `umap_snapshot_max_tokens` token occurrences. Token IDs, sequence indices, within-sequence positions, original batch shapes, and a SHA-256 fingerprint are persisted before any feature frame.

The fixed token tensors, rather than a fresh validation iterator, drive every later capture. This prevents changes in shuffle state, batch boundaries, or finite-loader exhaustion from changing the population. Resume loads the persisted population and never resamples it. A mismatched population or config fingerprint disables further animation capture for that run and records the exact reason; it does not terminate training.

## Training snapshots

Training captures step zero, every configured evaluation multiple, and the terminal step. Duplicate step requests are idempotent. At evaluation steps the capture uses the same temporary model weights that validation and best-model selection use, including EMA weights when EMA selection is active. The manifest records the weight source for every frame.

Capture reuses the existing belief and model bank semantics under `torch.no_grad()` and evaluation mode. Each frame stores raw CPU float32 values for the active channels plus aligned token metadata. Covariance chart conversion is deferred to rendering so the snapshots remain faithful to the model state and can be reprocessed under the versioned chart contract.

No UMAP fit, HDBSCAN call, semantic evaluation, PNG rendering, or GIF encoding occurs inside the training loop. Snapshot persistence is the only enabled-path training overhead beyond replaying the fixed diagnostic population.

## E-step snapshots

After training, the selected best checkpoint replays the same fixed token population. Belief snapshots capture the actual initial state and every executed E-step iterate. The existing diagnostic state recorder in `vfe3.inference.e_step.e_step` is the source of truth; interpolated or repeated artificial frames are forbidden.

When the model channel and `s_e_step` are active, the same diagnostic-only state recorder is threaded through `VFEModel._refine_s` to capture the initial and refined model-state iterates. Its optional argument defaults to `None`, leaving the executable inference path and gradients unchanged. When the model channel is inactive or static, the corresponding E-step animations are unavailable with a recorded reason.

Belief E-step GIFs require at least one executed transition. A two-frame GIF from `n_e_steps=1` is allowed but its sidecar records the low temporal depth. Model E-step GIFs apply the same rule to the model-state refinement trajectory.

## Snapshot and manifest format

Snapshots live under `run_dir/umap_snapshots/`. Each frame is an atomically replaced compressed `.npz` file containing non-executable arrays. The directory contains a strict JSON manifest with schema version, trajectory kind, step or iteration index, model/config/code fingerprints, weight source, token-population fingerprint, active channels, array shapes and dtypes, file checksum, and status.

Training snapshots are append-only across resume. If a file already exists with the expected checksum and metadata, the write is skipped. If the same logical frame has conflicting content, the manifest becomes invalid for animation generation and records the conflict. JSON contains no NaN or infinity.

The native snapshots remain after GIF generation. A dedicated Python API can regenerate animations from a completed run directory without rerunning training.

## Shared UMAP projection

Each trajectory/channel pair receives one shared scaler and one UMAP reducer. Native channel features use the current contracts: Euclidean coordinates for `mu`, log-Euclidean half-vectorized coordinates for `sigma`, and stored gauge coordinates for `phi`.

The fit population is balanced equally across frames. If all frame points exceed a fixed total fit budget, deterministic token-occurrence sampling selects the same count from each frame. The reducer is fit once on the balanced union and then transforms every complete frame. All transformed coordinates determine one fixed plot extent with a small constant margin.

The reducer configuration, fit-population hashes, frame order, scaler parameters, UMAP parameters, dependency versions, and random seed are stored in the sidecar. Raw UMAP coordinates are not treated as cross-run metrics.

## HDBSCAN identity tracking

HDBSCAN runs independently on every frame through the existing controlled native-chart/PCA cluster space. Noise always uses label `-1` and remains gray. Since raw HDBSCAN integers have no temporal identity, adjacent frames are compared by their sets of fixed token-occurrence identifiers.

A maximum-weight one-to-one assignment maximizes cluster membership overlap between adjacent frames. A matched cluster retains its persistent color only when it satisfies the documented overlap threshold. Unmatched clusters receive new persistent IDs. Additional many-to-one and one-to-many overlap edges identify merges and splits without forcing them into the one-to-one color assignment.

The sidecar records raw and persistent labels, cluster sizes, overlap matrices, accepted assignments, births, deaths, splits, merges, and noise fractions. Rendering uses persistent IDs only for color selection.

## Semantic-probe overlays and metrics

Every frame runs the existing preregistered semantic evaluator in native feature space. Resolved concepts retain stable semantic-field outline colors. All occurrences are outlined faintly, and one labeled representative is chosen as the occurrence nearest that concept's native-space centroid. This places a real observed occurrence in the UMAP rather than plotting an artificial average of nonlinear display coordinates.

Per-frame JSON retains concept resolution, pair distance percentiles, reciprocal ranks, hit-at-five, HDBSCAN co-membership, semantic-field separation, and semantic-label agreement. It also stores compact time series for the preregistered close, control, and descriptive pairs, including `king`-`queen` and the descriptive `king`-`father` comparison when those concepts resolve.

Semantic markers are overlays only. Their quantitative values always come from native features and controlled cluster labels.

## Output contract

GIFs and same-stem JSON sidecars are written under `run_dir/figures/animations/`:

```text
belief_umap_mu_training.gif
belief_umap_sigma_training.gif
belief_umap_phi_training.gif
model_umap_mu_training.gif
model_umap_sigma_training.gif
belief_umap_mu_estep.gif
belief_umap_sigma_estep.gif
belief_umap_phi_estep.gif
model_umap_mu_estep.gif
model_umap_sigma_estep.gif
```

Only available channels are written. Temporary per-frame raster files are task-owned and removed after successful GIF encoding. The final frame is held longer than intermediate frames. Every frame displays trajectory position, token count, persistent cluster count, noise fraction, sample fingerprint prefix, and the native-chart/display distinction.

The run-level manifest lists every requested output with `written`, `unavailable`, or `failed` status and an exact reason. Missing optional visualization dependencies or one channel's rendering failure do not invalidate the trained model or suppress other channels.

## Components and interfaces

`vfe3/viz/umap_animation.py` owns snapshot serialization, manifest validation, joint projection, temporal cluster matching, semantic overlay preparation, GIF rendering, and deterministic regeneration from a run directory. It reuses native-feature and controlled-clustering helpers rather than duplicating their mathematics.

`vfe3/viz/extract.py` gains focused fixed-token bank and model-state trace extractors where the existing batch-oriented interfaces cannot preserve the exact persisted population. Existing extractors retain their behavior.

`vfe3/model/model.py` threads an optional diagnostic state recorder through `_refine_s`. `vfe3/train.py` freezes the diagnostic population and invokes snapshot capture at step zero, selected evaluation points, and the terminal step. `vfe3/run_artifacts.py` owns the run-directory seam and invokes post-training E-step capture and GIF generation. `vfe3/config.py`, `train_vfe3.py`, and `ablation.py` expose the three explicit click-to-run settings without changing their current regime choices.

## Failure behavior

Snapshot and rendering failures are diagnostic failures, not training failures. They are logged and persisted with sufficient provenance to distinguish unavailable data, optional-dependency failures, corrupt snapshots, and implementation errors. A manifest fingerprint conflict fails closed: no GIF is presented as controlled when its frames do not share the required population and feature contract.

Training animation requires at least two distinct steps. E-step animation requires at least two actual states. Missing semantic concepts produce the existing null-bearing semantic records and do not block rendering. Insufficient points for UMAP or HDBSCAN make that channel unavailable rather than substituting a different scientific method without disclosure.

## Testing and acceptance criteria

Tests follow red-green cycles and use deterministic small arrays or tiny synthetic models. They pin configuration defaults and validation, fixed-token freezing, atomic `.npz` and JSON writes, dtype and alignment checks, population fingerprints, idempotent resume behavior, conflicting-frame invalidation, and channel availability.

Projection tests use an injected reducer to prove one fit per trajectory/channel, balanced per-frame sampling, full-frame transformation, fixed extents, and deterministic frame ordering. Cluster tests use hand-constructed partitions to pin persistent identity assignment, births, deaths, splits, merges, and noise handling. Semantic tests pin real-occurrence medoid selection and verify that quantitative values are native-space results rather than UMAP-coordinate calculations.

Integration tests pin the ten requested filenames under fully active synthetic settings, belief-only and static-model gating, two-frame E-step behavior, manifest reasons, temporary-frame cleanup, and regeneration from saved snapshots. Heavy native UMAP/HDBSCAN execution remains in the existing slow-test tier.

Focused configuration, extraction, artifact, animation, semantic-probe, and controlled-UMAP tests must pass. The full pytest suite must pass with totals read from JUnit XML. `git diff --check` must pass. The required `docs/2026-07-15-edits.md` record is updated during implementation.
