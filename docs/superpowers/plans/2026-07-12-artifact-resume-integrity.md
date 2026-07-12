# Artifact and Resume Integrity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete PB-01 through PB-04 by binding ablation reuse to the exact code and corpus, guaranteeing terminal ablation artifacts under the default cadence, carrying the selected best weights across run-directory resumes, and making each EFE ring seed independently durable and resumable.

**Architecture:** Every reusable artifact receives a versioned semantic contract. Ablation cells compare a precomputed code/data/config contract before reuse; training checkpoints carry a validated best-model bundle rather than detached scalar metadata; and the EFE ring experiment advances atomic per-seed bundles through `trained` and `complete` states. Validation-only ablations use a dedicated finalizer, so completing their artifact set never evaluates a test split or changes the experiment's model-selection rule.

**Tech Stack:** Python 3, PyTorch, NumPy, JSON, CSV, SHA-256, pathlib, pytest, JUnit XML.

## Global Constraints

- Implement PB-01, PB-02, PB-03, and PB-04 only. Do not change ablation grids, model equations, optimizer choices, E-step behavior, policy scoring, data order, or live experiment values.
- Work in a fresh task worktree and branch from the fetched `origin/main`; preserve every file and config toggle in the user's live checkout.
- Preserve the mathematically pure path and all current default model behavior. New fields are artifact bookkeeping and resume integrity only.
- Fail closed on an absent, malformed, stale, or semantically incompatible reuse artifact. Recompute the affected cell or seed; never silently trust legacy metadata.
- Do not evaluate an ablation test split. The ablation finalizer selects and reports with validation data only, and labels the split explicitly.
- Atomic publication means a same-directory temporary file followed by `os.replace`; no reader may observe a partial JSON or PyTorch bundle.
- Keep unit tests on CPU with `embed_dim=4`, `n_heads=1`, `n_layers=1`, and `n_e_steps=1` whenever a model is required. No CUDA benchmark is needed for artifact-integrity work.
- Run pytest without an extra `-q`. Derive all counts from JUnit XML rather than terminal recollection.
- Update the existing `docs/2026-07-12-edits.md` once with the implemented changes and machine-read verification result; do not create a second edit log for the date.

---

## File Structure

- Modify `vfe3/data/datasets.py` to expose a streaming, cache-file source identity used before an ablation cell is loaded.
- Modify `ablation.py` to build, persist, and compare `cell_contract.json`, and to invoke the validation-only terminal finalizer.
- Modify `vfe3/run_artifacts.py` to add validation-only finalization and portable best-model checkpoint state.
- Modify `efe_ring_experiment.py` to publish and resume atomic per-seed bundles plus an atomic aggregate result.
- Create `tests/test_ablation_artifact_resume_20260712.py` for PB-01 and PB-02 regressions.
- Modify `tests/test_checkpoint_resume.py` for PB-03 cross-run best-model continuity.
- Modify `tests/test_efe_ring_experiment.py` for PB-04 seed-bundle state transitions.
- Modify `docs/2026-07-12-edits.md` with the implementation and verified JUnit attributes.

### Task 1: Bind ablation reuse to code, data, and semantic configuration

**Files:** Modify `vfe3/data/datasets.py`, `ablation.py`; create `tests/test_ablation_artifact_resume_20260712.py`.

**Interfaces:**

```python
def cache_source_identity(
    dataset:   str,
    split:     str = "validation",

    *,
    cache_dir: Optional[Path] = None,
) -> Dict[str, object]:
    """Return the tokenizer, format, byte size, and SHA-256 identity of one cache source."""


def _cell_contract(
    cfg:              VFE3Config,
    dataset:         str,
    diagnostic_flags: Mapping[str, bool],

    *,
    data_seed:  int,
    max_tokens: Optional[int]  = None,
    cache_dir:  Optional[Path] = None,
) -> Dict[str, object]:
    """Build the versioned contract that authorizes reuse of one ablation cell."""


def _cell_is_current(
    run_dir:           Path,
    expected_contract: Mapping[str, object],
) -> bool:
    """Return true only for a successful cell with an exactly matching contract."""


def _expected_cell_contract_or_none(
    overrides: Mapping[str, object],
    dataset: str,
    diagnostic_flags: Mapping[str, bool],
    *,
    seed: int,
    max_steps: Optional[int] = None,
    max_tokens: Optional[int] = None,
) -> Optional[Dict[str, object]]:
    """Build the reuse contract inside the per-cell failure boundary, else forbid reuse."""
```

The persisted `cell_contract.json` schema is exact and versioned:

```python
contract = {
    "schema_version": 1,
    "semantic_config_fingerprint": semantic_config_fingerprint(asdict(cfg)),
    "dataset": dataset,
    "data_seed": int(data_seed),
    "max_tokens": int(max_tokens) if max_tokens is not None else None,
    "tokenizer_tag": _tokenizer_tag(dataset),
    "train_source": cache_source_identity(dataset, "train", cache_dir=cache_dir),
    "validation_source": cache_source_identity(dataset, "validation", cache_dir=cache_dir),
    "code_identity": _git_code_identity(),
    "diagnostic_flags": dict(sorted(diagnostic_flags.items())),
}
```

`cache_source_identity` must stream file bytes in fixed-size blocks. For `.pt`, hash the tensor file. For `.bin`, hash the binary file and its `.meta.json` sidecar separately and retain both digests. Include the resolved cache format, byte count, sidecar metadata, and tokenizer tag in the returned mapping. Memoize identities by resolved path, size, and nanosecond modification time within one process so a sweep hashes each unchanged corpus once without weakening the comparison.

- [ ] **Step 1: Add failing contract tests.** Add `test_cell_reuse_requires_contract`, `test_cell_reuse_rejects_code_identity_drift`, `test_cell_reuse_rejects_train_or_validation_source_drift`, `test_cell_reuse_rejects_semantic_config_drift`, `test_cache_source_identity_changes_when_cache_bytes_change`, `test_invalid_config_contract_forbids_reuse_without_aborting_sweep`, and `test_missing_or_corrupt_source_contract_forbids_reuse_without_aborting_sweep`. Construct temporary `.pt` and `.bin` caches; monkeypatch `_git_code_identity` to deterministic mappings; write a success marker plus contract; then mutate one contract axis at a time. A legacy directory with the old marker and no `cell_contract.json` must return `False`. Invalid config and source fixtures must produce the existing isolated per-cell error row while a following valid cell still runs.
- [ ] **Step 2: Run the red test.** Run `python -m pytest tests/test_ablation_artifact_resume_20260712.py --junitxml=C:\tmp\vfe3-ablation-contract-red.xml`. Expect a nonzero exit because the new interfaces and strict contract do not exist.
- [ ] **Step 3: Implement source identity.** Reuse `cache_path()` and `_tokenizer_tag()` in `vfe3/data/datasets.py`. Use a helper with this concrete digest loop:

```python
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
```

- [ ] **Step 4: Implement the ablation contract.** `_expected_cell_contract_or_none` wraps both `VFE3Config(**_cell_cfg_dict(...))` and `_cell_contract` source hashing in the per-cell try boundary. Any construction/source exception is logged and returns `None`, which only forbids cache reuse; `run_single` then executes inside its existing isolation and records the actual config/data error without aborting later cells. Call `_cell_is_current` only for a non-None contract. Compare loaded JSON with ordinary mapping equality; reject parse errors, wrong schema versions, non-mappings, missing success markers, and any unequal nested field. If the pre-run contract was None but `run_single` succeeds, rebuild it after the run to handle a transient source race. Publish neither `cell_contract.json` nor the completion marker unless that post-run contract exists and terminal artifacts are complete; otherwise convert the cell to a failed result. Publish a valid contract atomically before its completion marker. Do not write either file after an exception.
- [ ] **Step 5: Run the green test.** Re-run `python -m pytest tests/test_ablation_artifact_resume_20260712.py --junitxml=C:\tmp\vfe3-ablation-contract-green.xml`. Read the XML and require `failures="0"` and `errors="0"`.
- [ ] **Step 6: Commit.** Stage `vfe3/data/datasets.py ablation.py tests/test_ablation_artifact_resume_20260712.py` and commit as `fix(ablation): bind resume to code and corpus identity`.

### Task 2: Guarantee terminal artifacts for default ablation cells

**Files:** Modify `vfe3/train.py`, `vfe3/run_artifacts.py`, `ablation.py`, `tests/test_train.py`, `tests/test_ablation_artifact_resume_20260712.py`.

**Interfaces:**

```python
class TrainingTerminalState(NamedTuple):
    step:                 int
    optimizer:            torch.optim.Optimizer
    scaler:               Optional["torch.amp.GradScaler"]
    ema:                  Optional[object]
    metropolis_generator: Optional[torch.Generator]
    data_state:           Optional[DataState]
    raw_model_state:      Dict[str, torch.Tensor]
    rng_state:            Dict[str, object]


@torch.no_grad()
def finalize_validation_run(
    model:       torch.nn.Module,
    artifacts:   RunArtifacts,
    cfg:         VFE3Config,
    val_loader:  Iterable,

    *,
    tokens_per_char: float                    = 1.0,
    train_loader:    Optional[Iterable]       = None,
    losses:          Optional[List[float]]    = None,
    data_seed:       Optional[int]            = None,
    max_tokens:      Optional[int]            = None,
    tokenizer_tag:   Optional[str]            = None,
    device:          Optional[torch.device]   = None,
    wall_time:       Optional[float]          = None,
    logger:          Optional[logging.Logger] = None,
    terminal_state:  Optional[TrainingTerminalState] = None,
) -> Dict[str, object]:
    """Score validation, save terminal artifacts, and never open a test split."""
```

Add `terminal_callback: Optional[Callable[[TrainingTerminalState, List[float]], None]] = None` to the Optional keyword group of `train()`. Immediately after the final optimizer step and before the existing `ema.copy_to(model)`, clone the raw training `state_dict`, CPU RNG state, and all CUDA RNG states, then build `TrainingTerminalState(step=n_steps, ...)` with the live optimizer/scaler/EMA/private RNG/data cursor and invoke the callback exactly once. `None` takes the existing direct `ema.copy_to(model)`/return path without cloning or new RNG work.

The finalizer evaluates `val_loader` exactly once, saves the final weights if their validation perplexity improves the run-wide best, appends a terminal metrics row, writes `validation_results.json`, `summary.json`, `provenance.json`, and the pure-path report, then invokes the existing history-only `_save_figures`. Its terminal row is compatible with either an empty history or the established training schema:

```python
terminal_row = {
    "step":       int(cfg.max_steps),
    "train_loss": float(losses[-1]) if losses else float("nan"),
    "val_ce":     float(metrics["ce"]),
    "val_ppl":    float(metrics["ppl"]),
    "val_bpc":    float(metrics["bpc"]),
}
artifacts.log_metrics(terminal_row)
artifacts.maybe_save_best(cfg.max_steps, model, float(metrics["ppl"]))
```

The finalizer uses this exact successful sequence. The model enters the callback with raw training weights. If EMA exists, copy the EMA shadow into the model, then evaluate validation, publish the selected best weights, collect provenance, and render figures against that EMA model. In a `finally` block, load `terminal_state.raw_model_state` strictly and restore the captured CPU/CUDA RNG states. After successful validation, write the resumable checkpoint from the restored raw model plus its matching optimizer/scaler/EMA/private RNG/data cursor; then copy EMA back into the live model so `train()` retains its existing returned-model behavior. Publish `summary.json` only after the checkpoint exists. A validation/checkpoint failure restores raw weights/RNG, raises, and publishes no success contract. The checkpoint therefore never pairs EMA weights with raw optimizer moments, while `best_model.pt` and reported validation remain EMA-based exactly as deployed.

The terminal checkpoint captures the final optimizer/scaler/EMA/RNG/data cursor and the Task 3 portable best-selection bundle. Scheduler continuation remains reconstructed from the saved completed step and configured base learning rates by the existing `load_checkpoint` path. `summary.json` must contain `selection_split: "validation"`, `primary_val_ppl`, `final_val_ce`, `final_val_ppl`, `final_val_bpc`, `best_val_ppl`, `best_step`, `n_steps`, `n_params`, `final_train_loss`, `wall_time_s`, `terminal_checkpoint`, and `figures_written`. It must not contain `test_ce`, `test_ppl`, or `test_bpc`. The finalizer returns the exact ablation merge mapping `{"primary_val_ppl", "final_val_ppl", "final_val_ce", "final_val_bpc", "best_val_ppl", "best_step", "final_train_loss", "n_params", "terminal_checkpoint"}`. Set `primary_val_ppl` to the minimum of the finite run-wide best and the final validation PPL, or the final value when no earlier best exists; after `maybe_save_best`, it must equal the selected finite best. `run_single()` preserves its label/error/seed/overrides/token-cap metadata, updates it with this mapping, and then runs the requested diagnostics; it no longer relies on periodic log/eval/checkpoint cadence to create the advertised artifact set.

- [ ] **Step 1: Add failing finalization tests.** Add `test_default_ablation_cell_writes_terminal_artifact_set`, `test_validation_finalizer_records_validation_without_test_fields`, `test_validation_finalizer_appends_to_existing_metrics_schema`, `test_train_terminal_callback_receives_resumable_state`, `test_terminal_checkpoint_resumes_optimizer_rng_and_next_step`, `test_terminal_checkpoint_ema_raw_weights_resume_exactly`, `test_terminal_callback_restores_cpu_and_cuda_rng`, `test_run_single_finalizes_before_writing_success_contract`, and `test_run_single_terminal_merge_preserves_metadata_and_primary_val_ppl`. Use a tiny CPU model and loaders with `embed_dim=4`, `n_heads=1`, `n_layers=1`, `n_e_steps=1`, and two batches. The artifact-set assertion requires `checkpoints/step_<max_steps>.pt`, safe-loads it, and checks model, optimizer, scaler/EMA slots, RNG, data cursor, step, and best-selection metadata; it then resumes one additional step. The merge regression sets an earlier periodic best below final validation and requires the returned `primary_val_ppl` to equal that best while label, seed, overrides, max_tokens, and parameter count survive. The EMA test compares the next raw optimizer step to an uninterrupted control and separately requires the returned model/best validation to use EMA weights. Capture CPU and available CUDA global RNG immediately after the last training step and require equality after terminal finalization. Patch expensive model work only in tests that do not validate resume state.
- [ ] **Step 2: Run the red test.** Run `python -m pytest tests/test_ablation_artifact_resume_20260712.py --junitxml=C:\tmp\vfe3-ablation-terminal-red.xml`. Expect a nonzero exit because default cells currently omit terminal metrics, best weights, and summaries.
- [ ] **Step 3: Implement validation-only finalization.** Factor shared summary/provenance/pure-path/figure operations from `finalize_run()` only where doing so prevents duplication. Do not call `finalize_run()` with a validation loader masquerading as test data. Keep figure failures best-effort and record only paths that exist after the figure pass.
- [ ] **Step 4: Wire `run_single()`.** Define a terminal callback closure that calls `finalize_validation_run(..., terminal_state=state)` with the real train and validation loaders, `DATA_SEED`, token cap, tokenizer tag, wall time, and callback loss history, and stores its returned mapping. Pass that closure to `train()`; do not call the finalizer a second time after `train()` returns. Publish the result mapping and success contract only after the callback completed and the returned terminal checkpoint path exists. Preserve `checkpoint_interval=0` and `generate_figures=False` during the loop; terminal generation/checkpointing is one opt-in operation after the final step.
- [ ] **Step 5: Run the green tests.** Run `python -m pytest tests/test_ablation_artifact_resume_20260712.py tests/test_ablation_tackon.py tests/test_run_artifacts.py --junitxml=C:\tmp\vfe3-ablation-terminal-green.xml`. Read the XML and require `failures="0"` and `errors="0"`.
- [ ] **Step 6: Commit.** Stage `vfe3/train.py vfe3/run_artifacts.py ablation.py tests/test_train.py tests/test_ablation_artifact_resume_20260712.py` and commit as `fix(ablation): finalize default cell artifacts`.

### Task 3: Carry selected best weights across run-directory resumes

**Files:** Modify `vfe3/run_artifacts.py`, `tests/test_checkpoint_resume.py`.

**Interfaces:** Checkpoint bundles add `best_model_bundle: Optional[Dict[str, object]]`. Add a model/selection projection plus two internal helpers with strict semantic validation:

```python
def _selection_semantic_config(
    config: "VFE3Config | Mapping[str, object]",
) -> Dict[str, object]:
    """Remove resume/log/checkpoint/figure bookkeeping, retaining architecture and objective fields."""


def _read_best_model_bundle(
    path:                 Path,
    cfg:                  VFE3Config,
    expected_model_state: Mapping[str, torch.Tensor],
    map_location:         "str | torch.device",
) -> Dict[str, object]:
    """Load and validate config plus model-state keys, shapes, and dtypes without mutation."""


def _publish_best_model_bundle(
    bundle:               Mapping[str, object],
    expected_model_state: Mapping[str, torch.Tensor],
    artifacts:            RunArtifacts,
) -> None:
    """Revalidate and atomically publish a best-model bundle without loading it into the model."""
```

`_selection_semantic_config` starts from `asdict(cfg)` for a live `VFE3Config`. For a serialized mapping, first require every supplied key to name a field in the current `VFE3Config`; reject unknown newer fields instead of allowing `config_from_serialized()` to warn and ignore them. After that check, call `config_from_serialized(mapping, source="best-model selection compatibility")` and convert the resulting current config with `asdict`, so genuinely older mappings acquire current defaults for missing fields. Only then remove exactly `resume_from`, `log_interval`, `checkpoint_interval`, and `generate_figures`. Retain `eval_interval`, `max_steps`, optimizer/schedule fields, every architecture/family/transport/decode field, and every objective weight. Bundle compatibility compares `semantic_config_fingerprint(_selection_semantic_config(saved_config))` to the same projection of the live config; ordinary resume-path and output-cadence changes cannot invalidate otherwise identical selected weights. The raw saved mapping's full `config_fingerprint` is still verified before normalization, so default migration never hides artifact tampering.

`RunArtifacts.save_checkpoint()` must validate and embed the current `best_model.pt` whenever `best_val_ppl` is finite. A finite best scalar without a readable, matching best bundle is an integrity error and must prevent checkpoint publication. When no validation best exists, write `best_model_bundle=None`, `best_val_ppl=inf`, and `best_step=None`.

`load_checkpoint(..., artifacts=new_artifacts)` restores best state by this precedence:

1. Validate and publish the embedded `best_model_bundle`.
2. For a legacy checkpoint without that field, validate `<old_run>/best_model.pt`, where `<old_run>` is `checkpoint_path.parent.parent`, and publish it into the new run.
3. If neither validated bundle exists, set `new_artifacts.best_val_ppl=inf` and `new_artifacts.best_step=None` and emit one warning. Never retain unreachable best scalars.

After bundle publication, set the scalar metadata and require the file to exist. At the start of both finalizers, finite best metadata requires a file, but an old file with `best_val_ppl=inf` is ignored rather than treated as selected state:

```python
has_best_metadata = math.isfinite(float(artifacts.best_val_ppl))
if has_best_metadata and not artifacts.best_path.is_file():
    raise RuntimeError("finite best-model metadata has no reachable weights")
```

Reachability validation and model loading are separate. The ordinary `finalize_run` reloads `best_model.pt` only when `has_best_metadata` is true, because its held-out test evaluation intentionally scores the selected validation checkpoint. `finalize_validation_run` never loads a preexisting best into the live model before terminal validation: it scores the current terminal EMA exactly once, calls `maybe_save_best` with that result, reports the prior best through `primary_val_ppl` when it remains better, and keeps the terminal EMA in memory for provenance/figures until the raw-state restore. A recomputed ablation cell begins with infinite in-memory best metadata, so terminal validation atomically replaces any stale file before setting finite metadata; after publication, require both finite metadata and a readable compatible file. This prevents stale-contract reruns from either failing prematurely or silently selecting the previous cell's weights.

- [ ] **Step 1: Add failing portability tests.** Add `test_cross_run_resume_restores_embedded_best_bundle`, `test_legacy_cross_run_resume_imports_sibling_best_bundle`, `test_resume_without_best_weights_drops_unreachable_best_metadata`, `test_checkpoint_rejects_finite_best_without_weights`, `test_finalize_rejects_best_metadata_without_best_weights`, `test_validation_finalizer_scores_terminal_ema_before_best_selection`, `test_selection_projection_migrates_missing_defaults_and_rejects_unknown_fields`, `test_resume_from_only_difference_survives_finalization`, and `test_stale_contract_rerun_replaces_old_unselected_best`. The terminal-EMA test installs a distinct prior best whose PPL is lower than the final EMA result, spies on evaluation input, and requires exactly one validation call on terminal EMA weights, unchanged prior best-file bytes, `primary_val_ppl` equal to the prior best, and final fields equal to the terminal score. The migration test removes one known defaulted behavior field from a serialized mapping and requires equality with the live default projection, then adds an unknown key and requires fail-closed rejection; in both cases first verify that the stored full fingerprint matches the raw mapping. Add a cross-run case where the only config difference is `resume_from` and require successful import plus finalization, and a behavior-field control such as changed `decode_tau` that must reject. Add parameterized corrupt-bundle cases for a stale full `config_fingerprint`, missing key, extra key, wrong tensor shape, wrong tensor dtype, and non-tensor value. Save distinct final and best parameter values so equality checks prove that the selected weights, rather than the latest weights, moved to the new run. The stale-cell fixture starts with an old best file but infinite new-run metadata and requires terminal validation to replace its hash before success publication.
- [ ] **Step 2: Run the red test.** Run `python -m pytest tests/test_checkpoint_resume.py --junitxml=C:\tmp\vfe3-best-resume-red.xml`. Expect the cross-run assertions to fail because current checkpoints carry only best scalars.
- [ ] **Step 3: Implement portable checkpoint writes.** Load `best_model.pt` with `weights_only=True`. First require its stored full `config_fingerprint` to equal `semantic_config_fingerprint(saved_config)` so excluded-field tampering is still detected. Then compare `semantic_config_fingerprint(_selection_semantic_config(saved_config))` to the live projected config and compare `model_state` nonmutatingly against `model.state_dict()`: exact key set, tensor type, shape, and dtype for every key. Place only the validated mapping under `best_model_bundle` before the checkpoint's existing atomic `torch.save` and replace.
- [ ] **Step 4: Implement portable checkpoint loads and finalizer compatibility.** Validate each bundle's full internal fingerprint, then its projected saved-vs-live compatibility and state schema before publication. Revalidate in `_publish_best_model_bundle` and write through a same-directory `.pt.tmp` path under the new artifact directory. Replace both finalizers' full saved-vs-live config comparison with `_selection_semantic_config`, while retaining the full internal fingerprint check. Preserve safe loading and current config-drift warnings. Do not call `load_state_dict` on the best bundle or otherwise mutate the live training model during resume; only make the selection checkpoint reachable for later finalization.
- [ ] **Step 5: Run the green tests.** Run `python -m pytest tests/test_checkpoint_resume.py tests/test_run_artifacts.py --junitxml=C:\tmp\vfe3-best-resume-green.xml`. Read the XML and require `failures="0"` and `errors="0"`.
- [ ] **Step 6: Commit.** Stage `vfe3/run_artifacts.py tests/test_checkpoint_resume.py` and commit as `fix(checkpoint): make best-model state portable`.

### Task 4: Add atomic per-seed EFE ring bundles and resume states

**Files:** Modify `efe_ring_experiment.py`, `tests/test_efe_ring_experiment.py`. Add `asdict` from `dataclasses`, `hashlib`, `Path` from `pathlib`, `Real` from `numbers`, and `Dict`, `Mapping`, `Optional`, and `Tuple` from `typing`. Import `_atomic_replace` and `semantic_config_fingerprint` exactly from `vfe3.run_artifacts`; keep model/config construction qualified through the existing `rt` import.

**Interfaces:**

```python
def _semantic_experiment_config(cfg: Mapping[str, object]) -> Dict[str, object]:
    """Return every training/evaluation field that can change one seed's result."""


def _efe_ring_code_identity(root: Optional[Path] = None) -> str:
    """Hash the executable ring entry point and package Python sources, never result files."""


def _save_seed_bundle(
    path:              Path,
    model:             rt.VFEModel,
    experiment_config: Mapping[str, object],
    result:            Optional[Mapping[str, object]],

    *,
    seed:     int,
    adequacy: float,
    status:   str,
) -> Path:
    """Atomically publish a `trained` or `complete` seed bundle."""


def _load_seed_bundle_if_current(
    path:              Path,
    experiment_config: Mapping[str, object],
    device:            torch.device,

    *,
    seed: int,
) -> Optional[Tuple[rt.VFEModel, Dict[str, object]]]:
    """Rebuild the model and return a current validated seed bundle, else return None."""


def _validated_complete_result(
    result:   object,
    adequacy: float,
) -> Optional[Dict[str, object]]:
    """Return a safe aggregate-ready copy of one complete result, else None."""
```

Add `resume=True` to `CONFIG`; it changes only whether current bundles may be reused. Exclude `resume`, `out_dir`, and `log_every` from `_semantic_experiment_config` because they do not change model weights or measurements. Include `steps`, `batch_size`, `lr`, `n_dev`, `n_episodes`, `budget`, `candidate_mode`, `top_k`, `beta_C`, `gamma_grid`, `adequacy_threshold`, `delta_min`, `alpha`, `TEMP_GRID`, `NUCLEUS_TOP_P`, `TYPICAL_P`, and `FDR_Q`.

Each `<out_dir>/seeds/seed_<seed>.pt` bundle contains `schema_version=1`, `status`, `seed`, the semantic experiment mapping and SHA-256 fingerprint, `code_identity_sha256=_efe_ring_code_identity()`, `model_config`, its `semantic_config_fingerprint`, `model_state`, `adequacy`, and `result`. `_efe_ring_code_identity` hashes relative path plus bytes for `efe_ring_experiment.py` and every sorted `vfe3/**/*.py`, excluding `__pycache__`; it never inspects git status, docs, `out_dir`, seed bundles, or aggregate results. For either status, top-level `adequacy` must be a finite `Real` and not a bool. Accept only `status in {"trained", "complete"}`. A trained bundle requires `result is None`. A complete bundle requires a result mapping whose finite numeric `adequacy` exactly equals the top-level adequacy and whose `admitted` value is a real `bool`. When `admitted` is true, require finite numeric `gamma` and `temp`, a `metrics` mapping whose key set is exactly `{"full_efe_tuned", "full_efe_g1", "risk_only", "ambiguity_only", "flat_pref", "p_data_control", "temp_tuned_logprob", "logprob_baseline", "nucleus", "typical", "greedy_ref", "random"}` and in which every arm has a finite numeric `success`, and a `gates` mapping whose `go` member is a real `bool`. Reject any incomplete or inconsistent result and recompute the seed. Rebuild with `rt.VFEModel(rt.VFE3Config(**model_config))`, load strictly, move to the selected device, and set eval mode.

The complete-result validator is explicit so no restored mapping is indexed before validation:

```python
def _validated_complete_result(result: object, adequacy: float) -> Optional[Dict[str, object]]:
    if not isinstance(result, Mapping):
        return None
    copied = dict(result)
    result_adequacy = copied.get("adequacy")
    if (not isinstance(result_adequacy, Real) or isinstance(result_adequacy, bool)
            or not math.isfinite(float(result_adequacy))
            or float(result_adequacy) != float(adequacy)):
        return None
    admitted_value = copied.get("admitted")
    if type(admitted_value) is not bool:
        return None
    if admitted_value:
        for name in ("gamma", "temp"):
            value = copied.get(name)
            if (not isinstance(value, Real) or isinstance(value, bool)
                    or not math.isfinite(float(value))):
                return None
        metrics = copied.get("metrics")
        gates = copied.get("gates")
        expected_arms = {
            "full_efe_tuned", "full_efe_g1", "risk_only", "ambiguity_only",
            "flat_pref", "p_data_control", "temp_tuned_logprob", "logprob_baseline",
            "nucleus", "typical", "greedy_ref", "random",
        }
        if not isinstance(metrics, Mapping) or set(metrics) != expected_arms:
            return None
        if not isinstance(gates, Mapping) or type(gates.get("go")) is not bool:
            return None
        for arm in metrics.values():
            if not isinstance(arm, Mapping):
                return None
            success = arm.get("success")
            if (not isinstance(success, Real) or isinstance(success, bool)
                    or not math.isfinite(float(success))):
                return None
    return copied
```

Initialize the paths and semantic identity before the loop; these names are not implicit placeholders:

```python
out_dir = Path(cfg["out_dir"])
seed_dir = out_dir / "seeds"
out_dir.mkdir(parents=True, exist_ok=True)
seed_dir.mkdir(parents=True, exist_ok=True)
semantic_cfg = _semantic_experiment_config(cfg)
admitted = []

for seed in cfg["seeds"]:
    seed_path = seed_dir / f"seed_{int(seed)}.pt"
    # state machine below
```

The loop body becomes this explicit state machine:

```python
bundle = (_load_seed_bundle_if_current(seed_path, semantic_cfg, device, seed=seed)
          if cfg["resume"] else None)
if bundle is not None and bundle[1]["status"] == "complete":
    entry = dict(bundle[1]["result"])
else:
    if bundle is None:
        model, adequacy = rt.train_ring_checkpoint(
            seed=seed,
            steps=cfg["steps"],
            batch_size=cfg["batch_size"],
            lr=cfg["lr"],
            log_every=cfg["log_every"],
            device=str(device),
        )
        _save_seed_bundle(seed_path, model, semantic_cfg, None,
                          seed=seed, adequacy=adequacy, status="trained")
    else:
        model, saved = bundle
        adequacy = float(saved["adequacy"])
    is_admitted = adequacy >= cfg["adequacy_threshold"]
    entry = {"adequacy": adequacy, "admitted": is_admitted}
    if is_admitted:
        entry.update(run_checkpoint(model, cfg, str(device), seed))
    _save_seed_bundle(seed_path, model, semantic_cfg, entry,
                      seed=seed, adequacy=adequacy, status="complete")
if bool(entry["admitted"]):
    admitted.append(seed)
results["checkpoints"][str(seed)] = entry
```

`_load_seed_bundle_if_current` must run `_validated_complete_result(saved["result"], adequacy)` before returning a `complete` bundle and replace its stored `result` with the validated copy. A failed schema check returns `None`; the state machine then retrains and reevaluates instead of indexing malformed data. `_save_seed_bundle` writes through `seed_path.with_suffix(".pt.tmp")` and the shared atomic replace helper, so the trained and complete states are each independently publishable.

Write `ring_v1_results.json` through `<name>.tmp` and `os.replace` only after all requested seeds have complete entries. A crash after training resumes from `trained` without retraining; a crash after evaluation resumes from `complete` without retraining or reevaluating. In-step optimizer resume is outside PB-04 and must not be added.

- [ ] **Step 1: Add failing seed-state tests.** Add `test_seed_bundle_round_trip_restores_exact_weights`, `test_trained_seed_bundle_skips_training_but_runs_evaluation`, `test_complete_seed_bundle_skips_training_and_evaluation`, `test_seed_bundle_rejects_code_or_experiment_drift`, `test_malformed_seed_bundle_retrains`, and `test_aggregate_result_is_atomically_replaced`. Add `test_efe_ring_code_identity_ignores_seed_and_aggregate_publication_but_changes_with_source`: point the helper at a copied minimal source tree, publish seed/JSON results and require identity equality, then edit the copied entry source and require inequality. Parameterize malformed bundles over trained top-level NaN/bool adequacy and complete results with missing/non-boolean `admitted`, missing or extra metric arm, missing `gates`, non-boolean `gates["go"]`, non-finite arm `success`, and top-level/result adequacy mismatch; every case must return `None` and execute training/evaluation once. Monkeypatch the expensive training and arm matrix with counters; use a tiny CPU `rt.VFE3Config` with `embed_dim=4`.
- [ ] **Step 2: Run the red test.** Run `python -m pytest tests/test_efe_ring_experiment.py --junitxml=C:\tmp\vfe3-efe-seed-resume-red.xml`. Expect a nonzero exit because no per-seed bundle exists.
- [ ] **Step 3: Implement bundle validation and publication.** Reuse `semantic_config_fingerprint` and the atomic replacement helper from `vfe3.run_artifacts`; use `_efe_ring_code_identity` for the declared executable surface instead of whole-tree dirty identity. Catch expected file, safe-load, schema, and state-dict validation failures at the reuse boundary, log the reason, and return `None`; allow programming errors to surface.
- [ ] **Step 4: Implement the state machine.** Keep seed order, adequacy admission, paired episodes, statistics, gates, and final verdict byte-for-byte equivalent for a fresh run. Populate the aggregate from complete entries regardless of whether each entry was computed now or restored.
- [ ] **Step 5: Run the green tests.** Run `python -m pytest tests/test_efe_ring_experiment.py --junitxml=C:\tmp\vfe3-efe-seed-resume-green.xml`. Read the XML and require `failures="0"` and `errors="0"`.
- [ ] **Step 6: Commit.** Stage `efe_ring_experiment.py tests/test_efe_ring_experiment.py` and commit as `fix(efe): persist and resume each ring seed`.

### Task 5: Integrate, document, and verify PB-01 through PB-04

**Files:** Modify `docs/2026-07-12-edits.md`; inspect every file changed in Tasks 1 through 4.

- [ ] **Step 1: Run the combined focused suite.** Run `python -m pytest tests/test_ablation_artifact_resume_20260712.py tests/test_ablation_tackon.py tests/test_checkpoint_resume.py tests/test_run_artifacts.py tests/test_efe_ring_experiment.py --junitxml=C:\tmp\vfe3-artifact-resume-integrity.xml`.
- [ ] **Step 2: Read machine-readable results.** Parse the root `testsuite` or nested `testsuites` attributes and require `failures="0"` and `errors="0"`. Record the observed `tests`, `failures`, `errors`, and `skipped` values in `docs/2026-07-12-edits.md`; do not infer the pass count.
- [ ] **Step 3: Exercise a default ablation cell on CPU.** Patch only the test harness values to a two-step, `embed_dim=4` cell and run the production `run_single()` path. Require `config.json`, `metrics.csv`, `best_model.pt`, `validation_results.json`, `summary.json`, `provenance.json`, at least one history figure, `cell_contract.json`, and the completion marker. Confirm the summary contains validation fields and no test fields.
- [ ] **Step 4: Inspect artifact integrity.** Load every emitted PyTorch file with `weights_only=True`; parse every emitted JSON file; verify the completion marker is newer than the contract and terminal artifacts; and verify no `.tmp` file remains after successful publication.
- [ ] **Step 5: Inspect repository changes.** Run `git diff --check`, `git status --short`, and `git diff --stat origin/main...HEAD`. Confirm that no live config value, ablation grid, mathematical kernel, or policy statistic changed.
- [ ] **Step 6: Complete the repository lifecycle.** Run `git add vfe3/data/datasets.py vfe3/train.py vfe3/run_artifacts.py ablation.py efe_ring_experiment.py tests/test_ablation_artifact_resume_20260712.py tests/test_train.py tests/test_checkpoint_resume.py tests/test_efe_ring_experiment.py docs/2026-07-12-edits.md` as the exact final union, then inspect the staged diff and commit any remaining dated-log or integration change. Push the task branch, merge it into `main`, push `main`, fetch, and inspect `origin/main`. Fast-forward the user's local `main` only if that cannot alter WIP, then remove the temporary worktree and local task branch. Report the task commit, resulting `origin/main` SHA, JUnit attributes, worktree removal, and the final `git status --short` without claiming cleanliness unless the command shows it.
