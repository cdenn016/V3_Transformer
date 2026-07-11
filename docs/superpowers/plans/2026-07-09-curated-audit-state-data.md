# Curated Audit State and Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make deterministic execution, checkpoint continuation, data loading, provenance, cache reuse, artifact writes, and optimizer-loop reporting exact and auditable.

**Architecture:** One shared runtime seeding helper defines reversible process state. Checkpoints carry every auxiliary state required for continuation, data loaders expose enough state to resume the current epoch, and provenance/caches use explicit semantic identities rather than incidental paths.

**Tech Stack:** Python, PyTorch, pytest, JUnit XML, pathlib, hashlib, XML/JSON, Windows PowerShell.

## Global Constraints

- Follow the master plan and design spec; preserve production defaults and live experiment values.
- Scope: Findings 25, 27, 30, 39-46, 57, 76-80, 85-87, M5, M8, M10, L4, L6, and L8, plus the nine known baseline test failures.
- Reconstruct donor changes; do not patch the live checkout or import configuration hunks.
- `train_step` remains backward compatible for callers that consume only its scalar loss.
- Exact resume claims require exact batch equivalence; otherwise fail or label the limitation.
- Use `mmap=True` before limiting `.pt` caches; slicing after a normal `torch.load` does not close Finding 42.
- No extra pytest `-q`; machine counts come from JUnit.

---

## File Structure

- Create `vfe3/runtime.py`: reversible seeding and effective deterministic-state reporting.
- Modify `vfe3/run_artifacts.py`: checkpoint schema, fingerprints, provenance, safe artifact names, and missing-file handling.
- Modify `vfe3/train.py`: exact iterator resume, authoritative update status, EMA gating, and accumulated diagnostics.
- Modify `vfe3/gauge_optim.py`: persist `_omega_step` in optimizer extra state.
- Modify `vfe3/data/datasets.py`: safe components, limited mmap loading, token counts, and vocabulary validation.
- Modify `train_vfe3.py`, `scaling.py`, `ablation.py`: call shared runtime/data/provenance interfaces without changing config values.
- Modify `multiseed_analysis.py`, `vfe3/inference/sigma_gate.py`: identity checks and atomic publication.
- Test in the existing domain files named below plus new `tests/test_fixes_20260709_data.py` and `tests/test_fixes_20260709_scripts.py`.

### Task 1: Reversible deterministic setup and seed precedence

**Files:** Create `vfe3/runtime.py`; modify `ablation.py`, `scaling.py`, `train_vfe3.py`, `vfe3/run_artifacts.py`; modify `tests/test_deterministic.py`, `tests/test_run_naming.py`.

**Interfaces:** Produces `seed_everything(seed: int, *, deterministic: bool) -> None` and `deterministic_state() -> Dict[str, object]`.

- [ ] **Step 1: Write failing tests.** Add:

```python
def test_seed_everything_true_then_false_is_reversible():
    from vfe3.runtime import seed_everything
    seed_everything(1, deterministic=True)
    assert torch.are_deterministic_algorithms_enabled()
    seed_everything(1, deterministic=False)
    assert not torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.benchmark is True


def test_single_run_rejects_seed_precedence_mismatch():
    import train_vfe3
    with pytest.raises(ValueError, match="SEEDS.*config.*seed"):
        train_vfe3._resolve_seeds({"seed": 6}, seeds=(54,), num_runs=1)
```

- [ ] **Step 2: Run** `python -m pytest tests/test_deterministic.py tests/test_run_naming.py`; expect failures from one-way state and missing `_resolve_seeds`.
- [ ] **Step 3: Implement the shared helper.** In `vfe3/runtime.py`:

```python
_INITIAL_CUBLAS_WORKSPACE_CONFIG = os.environ.get("CUBLAS_WORKSPACE_CONFIG")


def seed_everything(seed: int, *, deterministic: bool) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    elif _INITIAL_CUBLAS_WORKSPACE_CONFIG is None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = _INITIAL_CUBLAS_WORKSPACE_CONFIG


def deterministic_state() -> Dict[str, object]:
    return {
        "algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
```

Replace entry-point-local seeding with this helper and always pass `cfg.deterministic`. Add `_resolve_seeds`; when one run has a nonempty `SEEDS`, require `SEEDS[0] == config["seed"]`. Persist `deterministic_state()` in provenance.
- [ ] **Step 4: Run** the two files again; expect PASS.
- [ ] **Step 5: Commit** `feat(runtime): centralize reversible deterministic setup`.

### Task 2: Correct the nine stale baseline tests without changing defaults

**Files:** Modify `tests/test_deterministic.py`, `tests/test_phase0_forward_beliefs.py`, `tests/test_run_artifacts.py`, `tests/test_run_naming.py`.

**Interfaces:** Test fixtures explicitly own their seeds; production remains `deterministic=True`, seed 6, and checkpoint interval 25000.

- [ ] **Step 1: Update the assertions.** Set `seed=0` in the Phase-0 `_base()` fixture and run-naming `_cfg()` fixture. Assert `VFE3Config().deterministic is True` and `checkpoint_interval == 25000`. Do not rewrite the frozen Phase-0 numeric values.
- [ ] **Step 2: Run:**

```powershell
python -m pytest tests/test_deterministic.py tests/test_phase0_forward_beliefs.py tests/test_run_naming.py tests/test_run_artifacts.py::test_config_checkpoint_interval_default_and_validated
```

Expected: all nine baseline failures are gone.
- [ ] **Step 3: Commit** `test(defaults): pin golden fixtures independently of click configs`.

### Task 3: Checkpoint optimizer, auxiliary RNG, and omega cadence state

**Files:** Modify `vfe3/run_artifacts.py`, `vfe3/train.py`, `vfe3/gauge_optim.py`; modify `tests/test_checkpoint_resume.py`, `tests/test_gauge_optim.py`, `tests/test_omega_metropolis.py`.

**Interfaces:** Checkpoint bundles add `metropolis_rng_state` and optimizer extra state. `GaugeNaturalGradAdamW.state_dict()` returns `omega_step`; `load_state_dict()` accepts its absence for old checkpoints with a non-exact-resume warning.

- [ ] **Step 1: Add failing tests** named `test_load_checkpoint_restamps_current_optimizer_group_metadata`, `test_metropolis_generator_state_roundtrips`, `test_state_dict_roundtrips_omega_reorth_cadence`, and `test_missing_checkpoint_preserves_file_not_found`.
- [ ] **Step 2: Run** `python -m pytest tests/test_checkpoint_resume.py tests/test_gauge_optim.py tests/test_omega_metropolis.py`; expect the new tests to fail.
- [ ] **Step 3: Implement.** Capture fresh non-parameter group metadata before `optimizer.load_state_dict`, then reapply it afterward:

```python
fresh = [{k: v for k, v in group.items() if k != "params"}
         for group in optimizer.param_groups]
optimizer.load_state_dict(bundle["optimizer_state"])
for group, metadata in zip(optimizer.param_groups, fresh):
    params = group["params"]
    group.clear()
    group.update(metadata)
    group["params"] = params
```

Persist the private Metropolis generator with `generator.get_state()` / `set_state()`. Override optimizer `state_dict`/`load_state_dict` to round-trip `_omega_step` under an `optimizer_extra` mapping. Preflight `Path(path).is_file()` before the safe-load exception wrapper.
- [ ] **Step 4: Rerun** the focused files; expect PASS.
- [ ] **Step 5: Commit** `fix(resume): restore optimizer and auxiliary continuation state`.

### Task 4: Exact shuffled-data iterator continuation

**Files:** Modify `vfe3/train.py`, `vfe3/run_artifacts.py`; modify `tests/test_checkpoint_resume.py`.

**Interfaces:** Checkpoint `data_state` contains `epoch_start_generator_state: Tensor`, `batches_consumed: int`, and `epoch: int`. Resume recreates the epoch permutation from its start state and consumes exactly `batches_consumed` batches before the next update.

- [ ] **Step 1: Replace the warning-only test** with `test_shuffled_resume_matches_uninterrupted_run`, comparing loss sequence and final parameters for an uninterrupted six-step run versus three steps plus resume plus three steps.
- [ ] **Step 2: Run** that test; expect a mismatch.
- [ ] **Step 3: Implement iterator state.** Immediately before `iter(loader)`, clone the loader generator state as `epoch_start_generator_state`. Increment `batches_consumed` after every successful `next(it)`. On resume:

```python
loader.generator.set_state(data_state["epoch_start_generator_state"])
it = iter(loader)
for _ in range(int(data_state["batches_consumed"])):
    next(it)
```

At epoch exhaustion, increment `epoch`, reset `batches_consumed`, and capture the next epoch's start state. Reject exact resume when a shuffled loader exposes no generator.
- [ ] **Step 4: Run** `python -m pytest tests/test_checkpoint_resume.py`; expect exact equivalence.
- [ ] **Step 5: Commit** `fix(resume): continue the exact shuffled batch stream`.

### Task 5: Limited cache loading, safe paths, and vocabulary bounds

**Files:** Modify `vfe3/data/datasets.py`, `train_vfe3.py`, `scaling.py`, `ablation.py`; create `tests/test_fixes_20260709_data.py`; modify `tests/test_run_naming.py`.

**Interfaces:** `load_cached_tokens(..., limit: Optional[int] = None)`, `cached_token_count(...) -> int`, `tokenizer_vocab_size(dataset: str) -> int`, `validate_token_range(tokens, vocab_size, *, dataset) -> None`, and `make_dataloader(..., vocab_size: Optional[int] = None)`.

- [ ] **Step 1: Recreate the donor tests** for `.pt`/`.bin` limiting, metadata counts, unsafe components, and vocabulary overflow. Add `test_load_pt_limit_uses_mmap` by spying that `torch.load(..., mmap=True)` was used, and `test_run_label_rejects_unsafe_dataset`.
- [ ] **Step 2: Run** `python -m pytest tests/test_fixes_20260709_data.py tests/test_run_naming.py::test_run_label_rejects_unsafe_dataset`; expect failures.
- [ ] **Step 3: Implement.** Validate components with a full-match safe-name expression and reject `"."`, `".."`, separators, drive-relative forms, and empty names. The limited `.pt` branch must be:

```python
tokens = torch.load(pt, weights_only=True, mmap=(limit is not None)).reshape(-1)
if limit is not None:
    tokens = tokens[:limit].clone()
return tokens.to(torch.long)
```

Slice `.bin` memmaps before `np.asarray` and int64 conversion. Thread `cfg.vocab_size` through every loader. Reuse the path-component validator in `_run_label`. Use `cached_token_count` for the uncapped banner count.
- [ ] **Step 4: Run** `python -m pytest tests/test_data.py tests/test_fixes_20260709_data.py tests/test_run_naming.py`; expect PASS.
- [ ] **Step 5: Commit** `fix(data): bound cache loads and validate token contracts`.

### Task 6: Per-split provenance and scaling code identity

**Files:** Modify `vfe3/run_artifacts.py`, `train_vfe3.py`, `scaling.py`, `ablation.py`; modify `tests/test_run_artifacts.py`, `tests/test_scaling_mup.py`.

**Interfaces:** `finalize_run(..., train_loader=None, val_loader=None, data_seed=None, max_tokens=None, tokenizer_tag=None)` forwards every value to `_write_provenance`. Provenance contains per-split SHA-256/count fields and `git_error` on failure.

- [ ] **Step 1: Add failing tests** `test_provenance_records_all_split_hashes_and_data_knobs`, `test_provenance_git_probe_timeout_records_error`, and `test_cell_is_current_rejects_missing_or_mismatched_code_identity`.
- [ ] **Step 2: Run** `python -m pytest tests/test_run_artifacts.py tests/test_scaling_mup.py`; expect failures.
- [ ] **Step 3: Implement.** Thread the new public arguments through every caller. Resolve Git with `shutil.which`, use a five-second timeout and constrained environment, and record `repr(exc)`. Require scaling cache provenance SHA to match current HEAD; fail closed on missing provenance. Dirty reuse is allowed only when both saved and current dirty fingerprints match exactly.
- [ ] **Step 4: Rerun** the focused files; expect PASS.
- [ ] **Step 5: Commit** `fix(provenance): bind runs and caches to code and data identity`.

### Task 7: Corpus frequency and homogeneous multiseed identity

**Files:** Modify `vfe3/run_artifacts.py`, `vfe3/viz/report.py`, `multiseed_analysis.py`; modify `tests/test_run_artifacts.py`, `tests/test_report.py`, `tests/test_multiseed.py`.

**Interfaces:** `_calibration_and_strata(..., corpus_counts: Tensor)` uses training counts for bucket membership while retaining sampled evaluation CE. Multiseed analysis computes a normalized config fingerprint excluding only `seed` and aborts on mixed fingerprints.

- [ ] **Step 1: Add failing tests** `test_frequency_strata_use_training_corpus_counts`, `test_vocab_comparison_rejects_mixed_tokenizers`, `test_seed_dirs_reject_mixed_semantic_configs`, and homogeneous-root acceptance.
- [ ] **Step 2: Run** `python -m pytest tests/test_run_artifacts.py tests/test_report.py tests/test_multiseed.py`; expect failures.
- [ ] **Step 3: Implement.** Build training `torch.bincount` once, pass it into calibration, and name the emitted field `corpus_freq_strata_ce`. Derive each comparison arm's tokenizer tag and reject mixed tags before plotting. Normalize serialized config by removing only `seed`; group or abort before aggregating. This plan chooses abort with a list of fingerprints and paths.
- [ ] **Step 4: Rerun** the focused files; expect PASS.
- [ ] **Step 5: Commit** `fix(analysis): require corpus and config identity`.

### Task 8: Atomic artifacts and successful ablation cache state

**Files:** Modify `vfe3/inference/sigma_gate.py`, `vfe3/run_artifacts.py`, `ablation.py`; modify `tests/test_sigma_gate.py`, `tests/test_run_artifacts.py`, `tests/test_ablation_tackon.py`.

**Interfaces:** Artifact names are regular bare filenames. Ablation markers persist requested diagnostic flags and a successful terminal state.

- [ ] **Step 1: Add failing tests** for atomic replace, `save_json` names `"."`, `".."`, `"a/b"`, `"a\\b"`, `"C:evil"`, missing requested artifacts, and failed markers.
- [ ] **Step 2: Run** the three focused files; expect failures.
- [ ] **Step 3: Implement.** Publish sigma-gate JSON through a same-directory temporary file and `os.replace`. Require `Path(name).name == name`, `not Path(name).is_absolute()`, and `name not in {".", ".."}`. Persist `collect_diagnostics`, `collect_extrapolation`, `error_kind`, and a finite terminal metric; `_cell_is_current` requires requested flags and success.
- [ ] **Step 4: Rerun**; expect PASS.
- [ ] **Step 5: Commit** `fix(artifacts): publish atomically and reject failed cache hits`.

### Task 9: Semantic best-model bundles, dataset-aware EFE generation, and paired RNG

**Files:** Modify `vfe3/run_artifacts.py`, `generate_efe.py`; create `tests/test_fixes_20260709_scripts.py`; modify `tests/test_run_artifacts.py`, `tests/test_generate.py`.

**Interfaces:** Best bundles are `{model_state, config, config_fingerprint}`. `semantic_config_fingerprint(config: Mapping[str, Any]) -> str` is stable JSON SHA-256. Legacy pure state dictionaries require an explicitly matching bound config and otherwise fail closed.

- [ ] **Step 1: Add failing tests** for embedded fingerprint, empty checkpoint error, dataset tokenizer selection, vocab mismatch, config mismatch, and identical pre-arm RNG state.
- [ ] **Step 2: Run** `python -m pytest tests/test_run_artifacts.py tests/test_fixes_20260709_scripts.py tests/test_generate.py`; expect failures.
- [ ] **Step 3: Implement.** Serialize normalized `asdict(cfg)` with sorted keys and compact separators before hashing. Save the fingerprint with best weights. Require `config_from` fingerprint equality. Select GPT-2 versus cl100k from the dataset and check `enc.n_vocab`. Construct both model instances before taking the RNG snapshot, then capture CPU and all CUDA RNG states before the base arm and restore them before the policy arm:

```python
cpu_state = torch.random.get_rng_state()
cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
base = model.generate(...)
torch.random.set_rng_state(cpu_state)
if cuda_state is not None:
    torch.cuda.set_rng_state_all(cuda_state)
policy = policy_model.generate(...)
```

- [ ] **Step 4: Rerun** the focused files; expect PASS.
- [ ] **Step 5: Commit** `fix(checkpoints): bind best weights and pair generation RNG`.

### Task 10: Authoritative optimizer-step status, EMA, and accumulated diagnostics

**Files:** Modify `vfe3/train.py`, `vfe3/run_artifacts.py`, `vfe3/viz/figures.py`; modify `tests/test_ema.py`, `tests/test_fp16_gradscaler.py`, `tests/test_grad_accum.py`, `tests/test_run_diagnostics_2026_06_13.py`.

**Interfaces:** `train_step(..., status_out: Optional[dict] = None) -> float` writes `status_out["did_step"]`. Accumulated E-step fields are named `estep_grad_norm_mu_microbatch_mean` and `estep_grad_norm_sigma_microbatch_mean`.

- [ ] **Step 1: Add failing tests** `test_ema_does_not_advance_when_train_step_skips`, `test_ema_does_not_advance_on_gradscaler_overflow`, and `test_estep_grad_metrics_are_microbatch_mean_not_last`.
- [ ] **Step 2: Run** `python -m pytest tests/test_ema.py tests/test_fp16_gradscaler.py tests/test_grad_accum.py`; expect failures.
- [ ] **Step 3: Implement.** Set `did_step=False` on nonfinite skip. For GradScaler, compare scale before and after `scaler.update()` and treat a decrease as overflow/no accepted update. Gate `ema.update(model)` on `did_step`. Collect a fresh diagnostic mapping per microbatch, sum each numeric field, divide by contributing microbatches, and emit the explicit mean names; update report consumers.
- [ ] **Step 4: Run** `python -m pytest tests/test_ema.py tests/test_fp16_gradscaler.py tests/test_grad_accum.py tests/test_run_diagnostics_2026_06_13.py tests/test_viz.py`; expect PASS.
- [ ] **Step 5: Commit** `fix(train): gate EMA and aggregate E-step diagnostics`.

## State/Data Plan Verification

- [ ] Run `python -m pytest tests/test_deterministic.py tests/test_checkpoint_resume.py tests/test_gauge_optim.py tests/test_fixes_20260709_data.py tests/test_fixes_20260709_scripts.py tests/test_run_artifacts.py tests/test_scaling_mup.py tests/test_multiseed.py tests/test_sigma_gate.py tests/test_ema.py tests/test_grad_accum.py --junitxml=C:\tmp\vfe3-curated-state-data.xml`.
- [ ] Read the XML attributes and update every assigned ledger row with the exact test and commit.
- [ ] Run `git diff --check`; verify no configured value in `train_vfe3.py` or `ablation.py` changed.
