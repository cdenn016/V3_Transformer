# Lightweight UMAP Animation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in, reproducible, separate-channel UMAP GIFs for training progression and actual E-step trajectories, backed by lightweight native-feature snapshots, temporally matched native-space clusters, and preregistered semantic probes.

**Architecture:** A new `vfe3.viz.umap_animation` module owns the fixed token population, atomic snapshot manifest, training/E-step frame capture, shared projection, temporal cluster identities, semantic overlays, and GIF/JSON output. Training only replays and persists fixed native features; all UMAP, HDBSCAN, semantic evaluation, and rendering occur after model selection. The existing isolated UMAP worker gains a one-fit/many-transform operation, while both finalizers invoke the animation pipeline on selected weights.

**Tech Stack:** Python 3.14, PyTorch float32, NumPy compressed NPZ, strict JSON, matplotlib, Pillow, umap-learn through the existing subprocess worker, scikit-learn HDBSCAN/PCA, SciPy linear assignment, pytest, JUnit XML.

## Global Constraints

`generate_umap_gifs=False` must leave the current training and finalization paths unchanged.

Every frame in one animation must use the same ordered token occurrences, native feature chart, scaler, UMAP reducer, plot limits, and semantic manifest.

HDBSCAN and semantic metrics must consume native-chart/PCA features, never two-dimensional UMAP coordinates.

Training snapshots must contain CPU float32 feature arrays and aligned integer token metadata in non-executable compressed NPZ files.

Belief `phi` must be labeled as a gauge-fixed coordinate diagnostic, not a gauge-invariant observable.

No interpolation, repeated artificial frame, independent per-frame UMAP fit, neural layer, CLI parser, or default-on expensive path may be added.

The live checkout and its config WIP must remain untouched; implementation stays in the isolated task worktree until the final safe integration step.

---

### Task 1: Configuration and selection-provenance contract

**Files:**
- Modify: `vfe3/config.py:650-700,2490-2530`
- Modify: `vfe3/run_artifacts.py:157-193`
- Modify: `train_vfe3.py:338-353`
- Modify: `ablation.py:377-397,1450-1465`
- Modify: `tests/test_config.py`
- Modify: `tests/test_checkpoint_resume.py`
- Modify: `tests/test_ablation_reporting.py`

**Interfaces:**
- Produces: `VFE3Config.generate_umap_gifs: bool`
- Produces: `VFE3Config.umap_snapshot_every_evals: int`
- Produces: `VFE3Config.umap_snapshot_max_tokens: int`
- Preserves: `_selection_semantic_config()` equality across output-only animation changes

- [ ] **Step 1: Write failing configuration and selection-projection tests**

```python
def test_umap_animation_config_defaults_and_validation():
    cfg = VFE3Config()
    assert cfg.generate_umap_gifs is False
    assert cfg.umap_snapshot_every_evals == 1
    assert cfg.umap_snapshot_max_tokens == 2048
    with pytest.raises(ValueError, match="generate_umap_gifs must be a bool"):
        VFE3Config(generate_umap_gifs=1)
    for value in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="umap_snapshot_every_evals"):
            VFE3Config(umap_snapshot_every_evals=value)
    for value in (0, 1, 2, 3.5, True):
        with pytest.raises(ValueError, match="umap_snapshot_max_tokens"):
            VFE3Config(umap_snapshot_max_tokens=value)


def test_selection_projection_ignores_umap_animation_output_controls():
    base = VFE3Config()
    changed = VFE3Config(
        generate_umap_gifs=True,
        umap_snapshot_every_evals=3,
        umap_snapshot_max_tokens=4096,
    )
    assert _selection_semantic_config(base) == _selection_semantic_config(changed)
```

Add an ablation assertion that the three names appear in `NON_SWEPT_FIELDS`, and add a serialized best-model compatibility assertion showing that the animation settings can differ without rejecting selected weights.

- [ ] **Step 2: Run the focused tests and verify the new fields are absent**

Run:

```powershell
python -m pytest tests/test_config.py::test_umap_animation_config_defaults_and_validation tests/test_checkpoint_resume.py::test_selection_projection_ignores_umap_animation_output_controls tests/test_ablation_reporting.py -x
```

Expected: FAIL because `VFE3Config` does not accept the animation fields and the output-only projection does not exclude them.

- [ ] **Step 3: Add the config fields and exact validation**

Add beside `generate_figures`:

```python
generate_umap_gifs:          bool = False
umap_snapshot_every_evals:   int  = 1
umap_snapshot_max_tokens:    int  = 2048
```

Extend the security/behavior bool loop with `generate_umap_gifs`. Validate exact integers rather than truthy coercions:

```python
if type(self.umap_snapshot_every_evals) is not int or self.umap_snapshot_every_evals < 1:
    raise ValueError(
        "umap_snapshot_every_evals must be an int >= 1, got "
        f"{self.umap_snapshot_every_evals!r}"
    )
if type(self.umap_snapshot_max_tokens) is not int or self.umap_snapshot_max_tokens < 3:
    raise ValueError(
        "umap_snapshot_max_tokens must be an int >= 3, got "
        f"{self.umap_snapshot_max_tokens!r}"
    )
```

Exclude all three fields in `_selection_semantic_config` because they cannot change selected weights. Add default-off values to both click-to-run config dictionaries without altering any existing regime choice, and add all three names to `NON_SWEPT_FIELDS`.

- [ ] **Step 4: Run configuration and compatibility tests**

Run:

```powershell
python -m pytest tests/test_config.py tests/test_checkpoint_resume.py tests/test_ablation_reporting.py --junitxml=C:\tmp\vfe3-umap-config-20260715.xml
```

Expected: JUnit records zero failures and zero errors.

- [ ] **Step 5: Commit the configuration contract**

```powershell
git add vfe3/config.py vfe3/run_artifacts.py train_vfe3.py ablation.py tests/test_config.py tests/test_checkpoint_resume.py tests/test_ablation_reporting.py
git commit -m "feat: configure lightweight UMAP animations"
```

### Task 2: Fixed population and atomic snapshot store

**Files:**
- Create: `vfe3/viz/umap_animation.py`
- Create: `tests/test_umap_animation_20260715.py`

**Interfaces:**
- Produces: `FixedPopulation`
- Produces: `freeze_population(loader: Iterable, max_tokens: int) -> FixedPopulation`
- Produces: `UMAPSnapshotStore.initialize(run_dir, cfg, loader, resume_from=None) -> UMAPSnapshotStore`
- Produces: `UMAPSnapshotStore.token_batches(device, batch_size) -> list[torch.Tensor]`
- Produces: `UMAPSnapshotStore.write_frame(trajectory, index, arrays, metadata) -> Path`
- Produces: `UMAPSnapshotStore.mark_unavailable(output, reason) -> None`

- [ ] **Step 1: Write failing fixed-population, strict-serialization, and resume tests**

Use a two-batch loader with known `(B,N)` token matrices. Pin that the frozen population keeps complete context rows while the aligned occurrence arrays stop exactly at `max_tokens`:

```python
def test_freeze_population_preserves_context_and_exact_occurrences():
    batches = [
        (torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]), torch.zeros(2, 4)),
        (torch.tensor([[9, 10, 11, 12]]), torch.zeros(1, 4)),
    ]
    population = freeze_population(batches, max_tokens=10)
    assert population.token_matrix.tolist() == [
        [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]
    ]
    assert population.token_ids.tolist() == list(range(1, 11))
    assert population.seq_idx.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2]
    assert population.pos_idx.tolist() == [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]
    assert len(population.sha256) == 64
```

Add tests that initialization writes `population.npz` plus `manifest.json`, all feature arrays remain `float32`, metadata arrays remain `int64`, duplicate identical frame writes are idempotent, conflicting content marks the manifest invalid, JSON rejects nonfinite values, and resume imports the source run's population and prior frame records rather than resampling a different loader.

- [ ] **Step 2: Run the new test module and verify import failure**

Run:

```powershell
python -m pytest tests/test_umap_animation_20260715.py -x
```

Expected: FAIL during collection because `vfe3.viz.umap_animation` does not exist.

- [ ] **Step 3: Implement the population and store primitives**

Start the module with versioned constants and immutable population data:

```python
SNAPSHOT_SCHEMA_VERSION = 1
FIT_MAX_POINTS = 50_000
CLUSTER_MATCH_MIN_JACCARD = 0.25


@dataclass(frozen=True)
class FixedPopulation:
    token_matrix: torch.Tensor
    token_ids:    torch.Tensor
    seq_idx:      torch.Tensor
    pos_idx:      torch.Tensor
    sha256:       str
```

`freeze_population` must detach tokens to CPU `int64`, require a consistent positive sequence length, collect enough complete sequence rows to cover the requested occurrence cap, derive aligned metadata, and fingerprint the context matrix shape plus canonical little-endian bytes.

Implement `_write_npz_atomic(path, arrays)` with a same-directory named temporary file, `np.savez_compressed`, `os.replace`, SHA-256 calculation, and cleanup in `finally`. Reuse `embedding_comparison.write_json_atomic` for strict manifests.

`UMAPSnapshotStore.initialize` must create:

```text
umap_snapshots/
  population.npz
  manifest.json
  training/
  estep/
```

The manifest must contain schema version, status, output configuration, source-run provenance, population metadata, ordered frame records, and requested-output statuses. If `resume_from` points to `.../checkpoints/step_N.pt`, inspect `Path(resume_from).parent.parent / "umap_snapshots"`; import its validated population and training frames before appending. If no compatible source snapshots exist, freeze the supplied loader and record that the historical trajectory is unavailable before the resume step.

`write_frame` must validate a common first dimension, finite float arrays, exact metadata equality with the fixed population, and the requested trajectory/index. It must write one NPZ and append a checksum-bearing manifest record atomically. On a same-index conflict, set `status="invalid"`, record the reason, and raise `ValueError` so the caller can stop animation capture while training continues.

- [ ] **Step 4: Run snapshot-store tests**

Run:

```powershell
python -m pytest tests/test_umap_animation_20260715.py -k "population or snapshot or resume or manifest" --junitxml=C:\tmp\vfe3-umap-snapshots-20260715.xml
```

Expected: JUnit records zero failures and zero errors.

- [ ] **Step 5: Commit the snapshot contract**

```powershell
git add vfe3/viz/umap_animation.py tests/test_umap_animation_20260715.py
git commit -m "feat: persist lightweight UMAP snapshots"
```

### Task 3: Actual belief/model trajectories and training cadence

**Files:**
- Modify: `vfe3/model/model.py:792-910`
- Modify: `vfe3/viz/extract.py:423-472,1100-1165`
- Modify: `vfe3/viz/umap_animation.py`
- Modify: `vfe3/train.py:1000-1535`
- Modify: `tests/test_umap_animation_20260715.py`
- Modify: `tests/test_train.py`
- Modify: `tests/test_model_channel_diagnostics_2026_06_13.py`

**Interfaces:**
- Produces: `_refine_s(..., state_record: Optional[dict] = None)`
- Produces: `e_step_belief_bank_trace(model, token_batches, max_tokens) -> dict[str, torch.Tensor]`
- Produces: `e_step_model_bank_trace(model, token_batches, max_tokens) -> Optional[dict[str, torch.Tensor]]`
- Produces: `capture_training_frame(store, model, step, weight_source, device) -> Path`
- Produces: `capture_estep_frames(store, model, device) -> list[Path]`

- [ ] **Step 1: Write failing trajectory and cadence tests**

Pin the model-channel recorder without changing the return value:

```python
def test_refine_s_optional_state_record_captures_real_iterates():
    model = _model(s_e_step=True, n_e_steps=2, lambda_h=0.2, lambda_gamma=0.2)
    tokens = torch.randint(0, model.cfg.vocab_size, (1, model.cfg.max_seq_len))
    belief = model.prior_bank.encode(tokens)
    phi = model._resolve_model_frame(tokens, model._apply_pos_phi(belief.phi))
    record = {}
    mu_s, sigma_s = model._refine_s(tokens, phi, state_record=record)
    assert len(record["beliefs"]) == 3
    assert torch.equal(record["beliefs"][-1].mu, mu_s)
    assert torch.equal(record["beliefs"][-1].sigma, sigma_s)
```

Add an extractor test with two token batches proving `(T+1,M,...)` alignment for belief `mu/sigma/phi` and model `mu/sigma`. Add a train test with a recording snapshot session proving capture steps `[0, 2, 4]` for `eval_interval=1`, `umap_snapshot_every_evals=2`, and `n_steps=4`; add an `eval_interval=0` case proving `[0, 4]`; add a default-off test proving no population iterator or capture import occurs.

- [ ] **Step 2: Run trajectory and train tests and verify failures**

Run:

```powershell
python -m pytest tests/test_umap_animation_20260715.py tests/test_train.py -k "umap or state_record or e_step_bank_trace" -x
```

Expected: FAIL because `_refine_s` rejects `state_record` and training has no animation cadence.

- [ ] **Step 3: Thread the diagnostic recorder through the model channel**

Add the optional argument at the end of `_refine_s`'s optional group:

```python
state_record: Optional[dict] = None,
```

Pass `state_record=state_record` into the existing `e_step` call. Do not alter `n_iter`, gradients, damping, transports, or the returned `(out.mu, out.sigma)`.

- [ ] **Step 4: Implement population-wide actual E-step extractors**

For belief traces, call `model.build_diagnostic_snapshot(tokens)` for each fixed batch and concatenate `snapshot.trace_states[t].mu/sigma/phi` across batch rows and positions at each actual recorded iteration. Reject inconsistent trajectory lengths across batches.

For model traces, return `None` unless both `model._model_channel_active` and `model.cfg.s_e_step` are true. Rebuild the exact model frame and RoPE used by `_refine_s`, pass a fresh recorder, and concatenate its actual beliefs per iteration. Slice every field to `max_tokens` only after complete-context inference.

The returned shapes are:

```python
{
    "mu":    torch.Tensor,  # (T+1, M, K)
    "sigma": torch.Tensor,  # (T+1, M, K) or (T+1, M, K, K)
    "phi":   torch.Tensor,  # belief only: (T+1, M, n_gen)
}
```

- [ ] **Step 5: Implement training and post-training capture helpers**

`capture_training_frame` must call the existing `belief_bank` and `model_channel_bank` over `store.token_batches(...)`, write the five requested raw channels when available, and attach aligned metadata.

`capture_estep_frames` must call the new extractors once, write one `estep/iter_NNNN.npz` per actual state, omit unavailable model arrays, and record a low-depth note when only two states exist.

In `train`, create the snapshot store only when `artifacts is not None`, `cfg.generate_umap_gifs`, and `val_loader is not None`. Capture the starting step after resume restoration. At periodic validation, capture after the EMA swap and best-save when the one-based evaluation count is divisible by `umap_snapshot_every_evals`. Capture the terminal step before `terminal_callback`, temporarily swapping EMA weights in exactly as validation does. Wrap diagnostic capture in a focused `try/except` that marks the manifest failed and logs the exception without terminating training.

- [ ] **Step 6: Run trajectory, model-channel, and train regressions**

Run:

```powershell
python -m pytest tests/test_umap_animation_20260715.py tests/test_train.py tests/test_model_channel_diagnostics_2026_06_13.py --junitxml=C:\tmp\vfe3-umap-capture-20260715.xml
```

Expected: JUnit records zero failures and zero errors.

- [ ] **Step 7: Commit real trajectory capture**

```powershell
git add vfe3/model/model.py vfe3/viz/extract.py vfe3/viz/umap_animation.py vfe3/train.py tests/test_umap_animation_20260715.py tests/test_train.py tests/test_model_channel_diagnostics_2026_06_13.py
git commit -m "feat: capture UMAP training and E-step trajectories"
```

### Task 4: One-fit projection and temporal cluster identities

**Files:**
- Modify: `vfe3/viz/figures.py:90-290`
- Modify: `vfe3/viz/umap_animation.py`
- Modify: `tests/test_july13_root_fixes.py`
- Modify: `tests/test_umap_animation_20260715.py`

**Interfaces:**
- Produces: `UMAPWorker.project_frames(fit_features, frame_features, ...) -> np.ndarray`
- Produces: `balanced_joint_projection(frames, projector, max_fit_points=50_000) -> ProjectionResult`
- Produces: `track_cluster_identities(labels_by_frame, min_jaccard=0.25) -> ClusterTimeline`

- [ ] **Step 1: Write failing one-fit and cluster-event tests**

Use an injected recording projector:

```python
def test_balanced_projection_fits_once_and_transforms_every_full_frame():
    frames = np.stack([
        np.arange(60, dtype=float).reshape(20, 3) + offset
        for offset in (0.0, 10.0, 20.0)
    ])
    calls = []

    def projector(fit_features, frame_features):
        calls.append((fit_features.copy(), frame_features.copy()))
        return frame_features[..., :2]

    result = balanced_joint_projection(frames, projector=projector, max_fit_points=12)
    assert len(calls) == 1
    assert calls[0][0].shape == (12, 3)
    assert calls[0][1].shape == (3, 20, 3)
    assert result.coordinates.shape == (3, 20, 2)
    assert len(np.unique(result.fit_frame_indices, return_counts=True)[1]) == 1
```

Use partitions with stable clusters, one birth, one death, one split, one merge, and noise. Assert exact persistent IDs and event records. Assert that passing UMAP coordinates to the clusterer is impossible because `analyze_frames` calls the injected clusterer with the standardized native/PCA matrix before the projector result is consumed.

- [ ] **Step 2: Run the projection tests and verify missing interfaces**

Run:

```powershell
python -m pytest tests/test_umap_animation_20260715.py tests/test_july13_root_fixes.py -k "project_frames or balanced_projection or cluster_identit" -x
```

Expected: FAIL because the worker and timeline functions do not exist.

- [ ] **Step 3: Extend the isolated UMAP worker with one-fit/many-transform**

Add a `project` request to `_UMAP_WORKER_SRC`. It must load one `(P,D)` fit array and one `(F,M,D)` frame array, construct one seeded reducer, call `fit(fit_features)` once, call `transform(frame)` for each frame, stack `(F,M,2)`, and save one numeric NPY. Do not enable pickle.

Add:

```python
def project_frames(
    self,
    fit_features:   np.ndarray,
    frame_features: np.ndarray,
    *,
    n_neighbors:    int,
    min_dist:       float,
    n_components:   int,
    seed:           int,
) -> np.ndarray:
```

Reuse the worker's timeout, crash isolation, temporary cleanup, and missing-UMAP error translation. Preserve the existing `embed` protocol and tests.

- [ ] **Step 4: Implement balanced scaling and shared projection**

Select `floor(max_fit_points / frame_count)` deterministic occurrence indices per frame, with at least one and at most `M`. Fit one mean and population standard deviation on the balanced union; replace zero scales with `1.0`; standardize all frames with those same parameters. Return coordinates, fixed global extent, scaler values, selected frame/occurrence indices, and hashes. A fully collapsed trajectory returns finite zero coordinates without starting UMAP.

- [ ] **Step 5: Implement persistent cluster matching**

For every frame, call `embedding_comparison.cluster_coordinates(native_features)` and `figures._cluster_embedding` with the controlled HDBSCAN constants. Reject any fallback method that does not begin with `HDBSCAN`.

Build Jaccard overlaps over fixed row identities, apply `scipy.optimize.linear_sum_assignment` to the negative overlap matrix, retain matches at or above `0.25`, and allocate monotonic persistent IDs to births. Record nonzero overlap edges separately; classify one previous cluster with two accepted edges as a split and two previous clusters with one current target as a merge. Noise remains `-1` and never participates in matching.

- [ ] **Step 6: Run projection and timeline tests**

Run:

```powershell
python -m pytest tests/test_umap_animation_20260715.py tests/test_july13_root_fixes.py tests/test_controlled_umap_comparison_20260714.py --junitxml=C:\tmp\vfe3-umap-projection-20260715.xml
```

Expected: JUnit records zero failures and zero errors.

- [ ] **Step 7: Commit shared projection and cluster tracking**

```powershell
git add vfe3/viz/figures.py vfe3/viz/umap_animation.py tests/test_july13_root_fixes.py tests/test_umap_animation_20260715.py
git commit -m "feat: stabilize UMAP animation geometry"
```

### Task 5: Semantic overlays, separate GIFs, and finalizer wiring

**Files:**
- Modify: `vfe3/viz/umap_animation.py`
- Modify: `vfe3/run_artifacts.py:1170-1410,1564-1750`
- Modify: `tests/test_umap_animation_20260715.py`
- Modify: `tests/test_run_artifacts.py`
- Modify: `tests/test_ablation_artifact_resume_20260712.py`

**Interfaces:**
- Produces: `semantic_marker_indices(features, semantic_record) -> dict[str, int]`
- Produces: `generate_umap_animations(run_dir, decode=None, projector=None, clusterer=None, logger=None) -> list[Path]`
- Produces: `finalize_umap_animations(run_dir, model, device, logger=None) -> list[Path]`

- [ ] **Step 1: Write failing semantic-marker, output-gating, and cleanup tests**

Pin that a concept label uses an actual occurrence nearest its native centroid:

```python
def test_semantic_marker_is_native_centroid_medoid_occurrence():
    features = np.array([[0.0], [2.0], [9.0]], dtype=float)
    semantic = {
        "concepts": {
            "king": {"resolved": True, "occurrence_indices": [0, 1]},
        }
    }
    assert semantic_marker_indices(features, semantic) == {"king": 0}
```

Create two synthetic training frames and three E-step frames with all five channels. Inject deterministic projection and cluster functions, then assert exactly ten GIF names and ten same-stem JSON files under `figures/animations`. Open each GIF with Pillow and assert its frame count. Assert no temporary frame directory remains.

Add gating cases for belief-only models, inactive `s_e_step`, one-state E-step traces, missing decoder, missing semantic concepts, insufficient training frames, and one channel's injected renderer failure. Assert explicit manifest statuses and that other outputs still write.

- [ ] **Step 2: Run rendering tests and verify missing output API**

Run:

```powershell
python -m pytest tests/test_umap_animation_20260715.py -k "semantic_marker or gif or animation_output or cleanup" -x
```

Expected: FAIL because rendering and finalizer APIs do not exist.

- [ ] **Step 3: Implement semantic evaluation and overlay preparation**

For each frame, call `semantic_probes.evaluate_semantic_probes(native_features, token_ids, raw_cluster_labels, decode)` when English diagnostics and a decoder are available; otherwise use `unavailable_record` with the exact policy reason. Resolve marker medoids from the semantic record's occurrence indices and native vectors. Keep semantic-field outline colors fixed across all frames.

- [ ] **Step 4: Implement separate GIF and JSON rendering**

Loop over:

```python
CHANNELS = (
    ("belief", "mu"),
    ("belief", "sigma"),
    ("belief", "phi"),
    ("model", "mu"),
    ("model", "sigma"),
)
TRAJECTORIES = ("training", "estep")
```

For each available pair, convert raw snapshots through `figures._belief_channel_features`, run native/PCA HDBSCAN, semantic evaluation, persistent matching, and one shared projection. Render all frames with fixed limits, persistent cluster colors, gray noise, faint semantic occurrence outlines, and medoid labels. Frame text must include trajectory position, population size, persistent cluster count, noise fraction, sample hash prefix, and the native-chart/display distinction. `phi` frames and sidecars must include the gauge-fixed warning.

Write temporary PNG frames inside `TemporaryDirectory(dir=animation_dir)`. Use Pillow `save(..., save_all=True, append_images=..., loop=0, disposal=2)` with 500 ms intermediate frames and a 1500 ms final frame. Close all images before the temporary directory exits.

The same-stem strict JSON must contain the projection/scaler contract, population and snapshot fingerprints, raw and persistent cluster labels, events, per-frame semantic records, compact semantic-pair time series, frame durations, dependency versions, and output checksum. Update the run manifest after every output attempt.

- [ ] **Step 5: Wire selected-weight E-step capture and rendering into both finalizers**

In `finalize_run`, call `finalize_umap_animations` after the best-validation model is reloaded and numeric/research artifacts are safe, independently of `generate_figures`.

In `finalize_validation_run`, finish validation/provenance/history figures on the deployed final EMA as before, then safe-load `best_model.pt`, temporarily load its selected state, and call `finalize_umap_animations` before the existing `finally` restores raw terminal weights and RNG. This ensures E-step GIFs use the selected best checkpoint even when an earlier periodic checkpoint beat the terminal validation.

Both calls must be gated by `cfg.generate_umap_gifs`, log failures, and leave the trained/selected artifact contract intact. Add animation output names and statuses to the run manifest; do not misclassify GIFs as the existing root-level PNG dashboard list.

- [ ] **Step 6: Run rendering and artifact-finalizer regressions**

Run:

```powershell
python -m pytest tests/test_umap_animation_20260715.py tests/test_run_artifacts.py tests/test_ablation_artifact_resume_20260712.py tests/test_reporting_additions.py --junitxml=C:\tmp\vfe3-umap-finalizers-20260715.xml
```

Expected: JUnit records zero failures and zero errors.

- [ ] **Step 7: Commit output generation and wiring**

```powershell
git add vfe3/viz/umap_animation.py vfe3/run_artifacts.py tests/test_umap_animation_20260715.py tests/test_run_artifacts.py tests/test_ablation_artifact_resume_20260712.py
git commit -m "feat: render semantic UMAP trajectory GIFs"
```

### Task 6: Documentation and verification

**Files:**
- Modify: `README.md`
- Modify: `docs/2026-07-15-edits.md`
- Verify: all files changed by Tasks 1-5

**Interfaces:**
- Documents: exact click-to-run enablement and output paths
- Verifies: focused and full repository behavior with machine-readable totals

- [ ] **Step 1: Document opt-in usage and scientific limitations**

Add a concise README section showing:

```python
generate_umap_gifs        = True
umap_snapshot_every_evals = 1
umap_snapshot_max_tokens  = 2048
```

State that the feature is future-run only, training snapshots are required for training progression, UMAP coordinates are display-only, clustering and semantic metrics use native/PCA features, and `phi` is gauge-fixed. Name `umap_snapshots/` and `figures/animations/`.

Update the existing `docs/2026-07-15-edits.md` section with implemented components, tests, exact JUnit totals, and any unavailable optional-tool checks. Do not claim a GIF was generated for the previously inspected run.

- [ ] **Step 2: Run syntax and focused verification**

Run:

```powershell
python -m compileall -q vfe3 tests
python -m pytest tests/test_umap_animation_20260715.py tests/test_config.py tests/test_train.py tests/test_run_artifacts.py tests/test_report.py tests/test_controlled_umap_comparison_20260714.py tests/test_semantic_probes_20260715.py tests/test_july13_root_fixes.py --junitxml=C:\tmp\vfe3-umap-focused-final-20260715.xml
```

Read `tests`, `failures`, `errors`, `skipped`, and `time` from the JUnit XML. Expected: zero failures and zero errors.

- [ ] **Step 3: Run the full test suite and repository checks**

Run:

```powershell
python -m pytest --junitxml=C:\tmp\vfe3-umap-full-final-20260715.xml
git diff --check
git status --short
```

Read totals from the JUnit XML. Expected: zero failures and zero errors; `git diff --check` exits zero; only intended source, test, README, plan/spec, and dated-edit files are modified.

- [ ] **Step 4: Inspect and commit the completed implementation**

```powershell
git diff --stat origin/main...HEAD
git diff --cached --check
git add README.md docs/2026-07-15-edits.md
git commit -m "docs: explain UMAP trajectory artifacts"
git status --short
```

Expected: the final worktree is clean after the documentation commit.

- [ ] **Step 5: Complete the mandatory repository lifecycle**

Fetch and inspect the current remote again. Rebase or merge only if remote movement requires it and verification remains valid. Push `codex/umap-gif-lightweight-snapshots`, merge it into `main`, push `main`, and fetch to verify `origin/main` contains the task commits. Fast-forward the user's local checkout only if doing so cannot overwrite its WIP; otherwise preserve it and report the exact blocking files. Remove the temporary worktree and local task branch after confirming the merge.
