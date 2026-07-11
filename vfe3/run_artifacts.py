r"""Run artifacts for VFE_3.0 training: the persistence + reporting layer.

A training run produces a self-contained directory::

    run_dir/
      config.json        full VFE3Config + run metadata (n_params, dataset, device, timestamp)
      metrics.csv        one row per periodic eval (step, train_loss, lr, val_ce/ppl/bpc, diagnostics)
      checkpoints/
        step_<N>.pt      resumable {step, model_state, optimizer_state, config}
      best_model.pt      {model_state, config, config_fingerprint} at the lowest validation PPL
      test_results.json  end-of-run TEST-split eval on the reloaded best checkpoint
      summary.json       headline numbers (best_val_ppl, test_ppl, wall_time, ...)
      loss_curve.png     training cross-entropy trajectory
      val_ppl.png        validation perplexity trajectory (log-y, best marked)
      holonomy.png / gauge_trace_spread.png   gauge-geometry diagnostics
      free_energy_decomposition.png   per-token F budget snapshot + early/mid/late evolution
      free_energy_codescent.png       F-vs-validation-CE co-descent (twin axis)

``RunArtifacts`` is OPT-IN: ``train`` only touches it when an instance is passed, so the silent
path (``artifacts=None``) writes nothing and is unchanged. ``finalize_run`` reloads the best-val
checkpoint, scores the held-out test split, and writes the summary + figures. Figure generation
is best-effort (a plotting/dependency error is logged, never fatal) so the numeric results
survive a viz problem.
"""

import csv
import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, Iterable, List, Mapping, Optional

import torch

from vfe3.config import VFE3Config
from vfe3.ema import EMA
from vfe3.runtime import deterministic_state


def _require_nonnegative_int(value: object, field: str) -> int:
    """Return an exact nonnegative integer cursor; reject coercible lookalikes."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"data_state {field} must be a non-negative integer")
    return value


def semantic_config_fingerprint(
    config: Mapping[str, Any],
) -> str:
    """Return the stable SHA-256 fingerprint of a normalized semantic config mapping."""
    normalized = json.dumps(dict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _atomic_replace(
    final: Path,                         # destination (the artifact name readers load)
    tmp:   Path,                         # same-directory temp file, already fully written

    *,
    delay:   float = 0.2,
    retries: int   = 5,
) -> None:
    r"""Atomically publish ``tmp`` over ``final`` via ``os.replace`` (same-volume rename).

    Same-directory temp + ``os.replace`` makes the publish an atomic rename on one volume, so a
    crash or power loss mid-write can never leave a truncated JSON or corrupt ``.pt`` at the final
    name (audit 2026-07-01 C11). Retries with backoff on ``PermissionError`` -- Windows can hold a
    transient open-handle lock on the destination (this host has hit it on ``best_model.pt``) --
    and re-raises any other error (and the last ``PermissionError``) so a real failure is never
    swallowed. On the raising paths the orphaned ``tmp`` is best-effort deleted first (audit
    2026-07-01 round-3); between retries it must survive (it is the source of the next replace)."""
    def _cleanup_tmp() -> None:
        try:                                             # cleanup failure must not mask the original error
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
    for i in range(retries):
        try:
            os.replace(tmp, final)
            return
        except PermissionError:
            if i == retries - 1:
                _cleanup_tmp()
                raise
            time.sleep(delay)
        except Exception:
            _cleanup_tmp()
            raise


class RunArtifacts:
    r"""Owns a run directory and the incremental writes (CSV rows, checkpoints, best model).

    Contract (m25): each instance owns a FRESH run_dir. ``__init__`` (re)writes config.json and the
    first ``log_metrics`` opens metrics.csv with ``"w"`` (truncate), so aiming a new instance at a
    populated dir would clobber it -- but no path does: resume builds a new timestamped run_dir and
    restores state from a checkpoint FILE via ``load_checkpoint``, never reusing a dir in place."""

    def __init__(
        self,
        run_dir:   'str | Path',
        cfg:       VFE3Config,
        model:     torch.nn.Module,

        *,
        dataset:   str                  = "",
        device:    'str | torch.device' = "cpu",
        timestamp: Optional[str]        = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.run_dir / "metrics.csv"
        self.best_path = self.run_dir / "best_model.pt"
        self.cfg = cfg                                       # kept for figure scaling (lambda_beta) + guards

        self.best_val_ppl: float = float("inf")
        self.best_step: Optional[int] = None
        self.history: List[Dict[str, float]] = []          # in-memory copy of the CSV rows (for figures)
        self._fieldnames: Optional[List[str]] = None

        self.save_json("config.json", {
            "config":    asdict(cfg),
            "n_params":  int(sum(p.numel() for p in model.parameters())),
            "dataset":   dataset,
            "device":    str(device),
            "timestamp": timestamp,
        })

    def save_json(self, name: str, obj: dict) -> Path:
        r"""Write ``obj`` as pretty JSON to ``run_dir/name`` (non-serializable -> str).

        Atomic: written to a same-directory ``.tmp`` then published via ``os.replace``, so a crash
        mid-write can never leave a truncated/partial JSON at the final name."""
        candidate         = Path(name)
        windows_candidate = PureWindowsPath(name)
        if (not name or name in {".", ".."} or "/" in name or "\\" in name
                or candidate.name != name or candidate.is_absolute() or windows_candidate.drive):
            raise ValueError(f"artifact name must be a regular bare filename, got {name!r}")
        path = self.run_dir / name
        tmp  = self.run_dir / (name + ".tmp")
        tmp.write_text(json.dumps(obj, indent=2, default=str))
        _atomic_replace(path, tmp)
        return path

    def log_metrics(self, row: Dict[str, float]) -> None:
        r"""Append one metrics row to ``metrics.csv`` (header written on the first call).

        The column set is fixed by the first row; later rows must share those keys so the CSV
        stays rectangular (the training loop emits a homogeneous row each periodic eval).

        NaN cells are written to the file as BLANK (empty string), so an eval-cadence column
        (val_*, generalization_gap, the held-out probes) -- NaN on the denser log-interval rows
        between evals -- shows an empty cell rather than a repeated value or a literal "nan".
        The IN-MEMORY ``self.history`` keeps the raw NaN float so
        the figure pass (which filters on ``math.isfinite``) is unaffected."""
        self.history.append(dict(row))                          # raw floats (incl. NaN) for the figure pass
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
            with open(self.csv_path, "w", newline="") as fh:
                csv.DictWriter(fh, fieldnames=self._fieldnames).writeheader()
        csv_row = {k: ("" if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items()}
        with open(self.csv_path, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=self._fieldnames).writerow(csv_row)

    def maybe_save_best(self, step: int, model: torch.nn.Module, val_ppl: float) -> bool:
        r"""Save weights bound to their semantic config iff ``val_ppl`` is a new minimum.

        Atomic (same-dir tmp + ``os.replace``): a crash or Windows lock mid-save can never leave a
        corrupt/unreadable ``best_model.pt`` where a good one stood."""
        if val_ppl < self.best_val_ppl:
            self.best_val_ppl = float(val_ppl)
            self.best_step = int(step)
            config = asdict(self.cfg)
            bundle = {
                "model_state":        model.state_dict(),
                "config":             config,
                "config_fingerprint": semantic_config_fingerprint(config),
            }
            tmp = self.best_path.with_suffix(".pt.tmp")
            torch.save(bundle, tmp)
            _atomic_replace(self.best_path, tmp)
            return True
        return False

    def save_attention_maps(
        self,
        step:   int,
        maps:   torch.Tensor,                 # (L, H, N, N) per-layer per-head attention
        logger: Optional[logging.Logger] = None,
    ) -> Optional[list]:
        r"""Best-effort attention heatmaps for one periodic eval: one figure per (layer, head).

        Writes ``attention/step_<N>_layer<l>_head<h>.png`` per (layer, head) -- a LOG-scaled beta
        heatmap (see :func:`vfe3.viz.figures.plot_attention_heatmap`) on a colour scale shared
        across panels so heads/layers stay comparable. Mirrors ``_save_figures``: a plotting or
        dependency error is logged and swallowed (never fatal to the run), and each figure is
        closed so ~30 evals do not leak figures. Returns the paths written, or None on failure.
        """
        try:
            from vfe3.viz import figures as figs
            figs.set_publication_style()
            m = maps.detach().cpu() if hasattr(maps, "detach") else torch.as_tensor(maps)
            if m.dim() == 2:                                        # (N, N) -> one layer, one head
                m = m[None, None]
            elif m.dim() == 3:                                      # (H, N, N) -> one layer
                m = m[None]
            if m.dim() != 4:
                raise ValueError(f"attention maps must be (L, H, N, N); got {tuple(m.shape)}")
            n_layers, n_heads = m.shape[0], m.shape[1]
            pos  = m[m > 0]                                         # shared log scale across all panels
            vmax = float(pos.max()) if pos.numel() else 1.0
            vmin = float(pos.min()) if pos.numel() else vmax * 1e-3
            attn_dir = self.run_dir / "attention"
            attn_dir.mkdir(exist_ok=True)
            paths = []
            for li in range(n_layers):
                for hi in range(n_heads):
                    path = attn_dir / f"step_{step}_layer{li}_head{hi}.png"
                    fig = figs.plot_attention_heatmap(
                        m[li, hi], log=True, vmin=vmin, vmax=vmax,
                        title=f"Attention (step {step}) - layer {li} head {hi}", path=str(path))
                    figs.plt.close(fig)
                    paths.append(path)
            return paths
        except Exception as exc:                                    # a viz error must never kill training
            (logger or logging.getLogger(__name__)).warning(
                "attention-map figure at step %d failed (%s); training continues", step, exc)
            return None

    def save_gamma_attention_maps(
        self,
        step:   int,
        maps:   'Optional[torch.Tensor]',     # (H, N, N) per-head model-coupling gamma, or None (channel off)
        logger: Optional[logging.Logger] = None,
    ) -> Optional[list]:
        r"""Best-effort model-coupling (gamma) heatmaps for one periodic eval: one figure per head.

        The s-channel sibling of :meth:`save_attention_maps`. Writes
        ``attention/step_<N>_gamma_head<h>.png`` per head -- a LOG-scaled gamma_ij heatmap on the
        VIRIDIS colour map (the belief beta channel uses magma) so the two channels read apart, on a
        scale shared across heads. ``maps`` is None when the model channel is inactive
        (``gamma_attention_maps`` returns None) -> no-op. A plotting error is logged and swallowed.
        """
        if maps is None:                                            # model channel inactive -> nothing to plot
            return None
        try:
            from vfe3.viz import figures as figs
            figs.set_publication_style()
            m = maps.detach().cpu() if hasattr(maps, "detach") else torch.as_tensor(maps)
            if m.dim() == 2:                                        # (N, N) -> one head
                m = m[None]
            if m.dim() != 3:
                raise ValueError(f"gamma maps must be (H, N, N); got {tuple(m.shape)}")
            n_heads = m.shape[0]
            pos  = m[m > 0]                                         # shared log scale across heads
            vmax = float(pos.max()) if pos.numel() else 1.0
            vmin = float(pos.min()) if pos.numel() else vmax * 1e-3
            attn_dir = self.run_dir / "attention"
            attn_dir.mkdir(exist_ok=True)
            paths = []
            for hi in range(n_heads):
                path = attn_dir / f"step_{step}_gamma_head{hi}.png"
                fig = figs.plot_attention_heatmap(
                    m[hi], log=True, vmin=vmin, vmax=vmax, cmap="viridis", symbol=r"\gamma",
                    title=f"Model-coupling attention (step {step}) - head {hi}", path=str(path))
                figs.plt.close(fig)
                paths.append(path)
            return paths
        except Exception as exc:                                    # a viz error must never kill training
            (logger or logging.getLogger(__name__)).warning(
                "gamma-map figure at step %d failed (%s); training continues", step, exc)
            return None

    def save_checkpoint(
        self,
        step:      int,
        model:     torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        cfg:       VFE3Config,

        *,
        scaler:               Optional['torch.amp.GradScaler'] = None,
        ema:                  Optional[EMA]                     = None,
        metropolis_generator: Optional[torch.Generator]         = None,
        data_state:            Optional[Dict[str, object]]       = None,
    ) -> Path:
        r"""Write a resumable ``checkpoints/step_<N>.pt`` (model + optimizer + RNG + config + step).

        ``load_checkpoint`` reads this back to continue training: ``model_state`` and
        ``optimizer_state`` restore the weights and AdamW momentum, ``rng_state`` restores the
        CPU (and CUDA) generators for reproducible continuation, and ``step`` is the number of
        completed M-steps so the resumed run rebuilds the cosine ``LambdaLR`` at the right point.
        ``scaler`` (audit 2026-06-09 IE3): an ENABLED fp16 GradScaler's state (current scale +
        growth counters) is bundled so a resumed fp16 run does not restart at the init scale
        65536 and re-converge by skipped steps; a disabled/None scaler stores None.
        ``best_val_ppl``/``best_step`` (audit 2026-07-01 C2): the model-selection state is bundled
        so a resumed run reports the run-wide best, not just the continuation's best. The write is
        atomic (same-dir tmp + ``os.replace``) so a crash never leaves a corrupt ``step_<N>.pt``.
        ``metropolis_generator`` carries the private accept/reject stream independently of the
        global CPU/CUDA RNG so a resumed discrete-reflection sweep continues at the next draw.
        ``data_state`` records the current epoch's starting loader-generator state and cursor;
        the state tensor is cloned into the bundle so later generator advances cannot mutate it.
        """
        path = self.ckpt_dir / f"step_{step}.pt"
        tmp  = path.with_suffix(".pt.tmp")
        rng_state = {
            "cpu":  torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        saved_data_state = None
        if data_state is not None:
            batches_consumed = _require_nonnegative_int(
                data_state["batches_consumed"], "batches_consumed")
            epoch = _require_nonnegative_int(data_state["epoch"], "epoch")
            saved_data_state = {
                "epoch_start_generator_state": data_state["epoch_start_generator_state"].clone(),
                "batches_consumed":            batches_consumed,
                "epoch":                       epoch,
            }
        torch.save({
            "step":            int(step),
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "rng_state":       rng_state,
            "metropolis_rng_state": (metropolis_generator.get_state()
                                      if metropolis_generator is not None else None),
            "config":          asdict(cfg),
            "scaler_state":    (scaler.state_dict()
                                if scaler is not None and scaler.is_enabled() else None),
            "ema_state":       (ema.state_dict() if ema is not None else None),
            "best_val_ppl":    float(self.best_val_ppl),
            "best_step":       self.best_step,
            "data_state":      saved_data_state,
        }, tmp)
        _atomic_replace(path, tmp)
        return path


def load_checkpoint(
    path:      'str | Path',
    model:     torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,

    *,
    map_location:         'Optional[str | torch.device]'   = None,
    restore_rng:          bool                             = True,
    scaler:               Optional['torch.amp.GradScaler'] = None,
    cfg:                  Optional[VFE3Config]             = None,
    ema:                  Optional[EMA]                    = None,
    artifacts:            'Optional[RunArtifacts]'         = None,
    metropolis_generator: Optional[torch.Generator]        = None,
    data_state:            Optional[Dict[str, object]]      = None,
) -> int:
    r"""Restore a ``save_checkpoint`` bundle into ``model`` (and optionally ``optimizer``); return the saved step.

    This is the LOAD half of the resumable checkpoint. It always restores the model weights;
    it restores the AdamW optimizer state (momentum buffers + per-parameter step counts) when an
    ``optimizer`` is supplied, then reapplies that optimizer's current non-parameter group metadata
    so the current config remains authoritative. The CPU/CUDA RNG and the optional private
    ``metropolis_generator`` are restored when ``restore_rng`` is set and the bundle carries their
    states (older checkpoints simply skip absent RNG fields). The returned integer is the number of
    completed M-steps; ``train(resume_from=...)`` uses it to rebuild the cosine ``LambdaLR`` at the
    saved step and to start the loop from there.

    ``scaler`` (audit 2026-06-09 IE3): when given AND the bundle carries a saved scaler state,
    the fp16 GradScaler's scale/growth counters are restored (bundles written before the scaler
    was persisted, or written from a non-fp16 run, simply skip the step). ``cfg`` (audit IE4):
    when given, the CURRENT config is compared against the bundle's saved config and any
    differing fields are warned about -- strict ``load_state_dict`` already catches
    shape-changing divergence, but shape-preserving semantic drift (LR schedule, n_e_steps,
    e_*_lr, ...) would otherwise pass silently. ``artifacts`` (audit 2026-07-01 C2): when given,
    the bundled ``best_val_ppl``/``best_step`` model-selection state is restored into it (bundles
    without those fields skip the restore). When a mutable ``data_state`` mapping is supplied, it
    is filled from the bundled iterator cursor; older checkpoints leave it empty.

    The bundle is loaded with ``weights_only=True`` by default, which refuses to execute arbitrary
    pickle reductions: our bundle carries only tensors, an ``asdict`` config dict, and RNG tensors
    (no custom classes), so it loads safely under it (matching
    ``test_run_artifacts.py::test_save_checkpoint_is_loadable``). A bundle that fails the safe load
    (e.g. an older format) RAISES unless ``cfg.trust_resume_checkpoint`` is set, which falls back to
    the legacy ``weights_only=False`` load -- only use that for a checkpoint you trust, since that
    path can execute arbitrary code embedded in the pickle.
    """
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint file not found: {checkpoint_path}")
    if map_location is None:
        map_location = next(model.parameters()).device
    trust = bool(getattr(cfg, "trust_resume_checkpoint", False))
    try:
        ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except Exception as exc:                                    # safe load rejected a non-tensor object
        if not trust:
            raise RuntimeError(
                f"checkpoint {Path(path).name} could not be loaded under the safe weights_only=True "
                f"path ({type(exc).__name__}: {exc}). If you trust this file, set "
                f"trust_resume_checkpoint=True to allow the legacy weights_only=False load (which can "
                f"execute arbitrary code embedded in the pickle)."
            ) from exc
        ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    saved_data_state = ckpt.get("data_state")
    if saved_data_state is not None:
        saved_batches_consumed = _require_nonnegative_int(
            saved_data_state["batches_consumed"], "batches_consumed")
        saved_epoch = _require_nonnegative_int(saved_data_state["epoch"], "epoch")
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        fresh = [{k: v for k, v in group.items() if k != "params"}
                 for group in optimizer.param_groups]
        optimizer.load_state_dict(ckpt["optimizer_state"])
        for group, metadata in zip(optimizer.param_groups, fresh):
            params = group["params"]
            group.clear()
            group.update(metadata)
            group["params"] = params
    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])
    # EMA shadow: restore it so a resumed run continues the SAME running average instead of re-seeding
    # from the resumed iterate. When the bundle carries no ema_state (a use_ema=False or legacy
    # checkpoint), the shadow was constructed from the PRE-load fresh init (EMA is built before this
    # load overwrites the model), so reseed it from the just-loaded weights -- otherwise the running
    # average blends real weights into random-init noise (audit 2026-07-01 C3).
    if ema is not None:
        if ckpt.get("ema_state") is not None:
            ema.load_state_dict(ckpt["ema_state"])
        else:
            ema.reset(model)   # no bundled shadow: reseed from the just-loaded weights, not the pre-load init
    # Best-val model-selection state (audit 2026-07-01 C2): restore best_val_ppl/best_step into the
    # resumed run's RunArtifacts so a continuation with no post-resume improvement still reports the
    # run-wide best. Only the scalar metadata is bundled; best_model.pt itself lives in the run_dir
    # (correct for a same-run_dir resume; a cross-run_dir resume still lacks the weights file).
    # Bundles written before these fields existed simply skip the restore (backward compatible).
    if artifacts is not None and ckpt.get("best_val_ppl") is not None:
        artifacts.best_val_ppl = float(ckpt["best_val_ppl"])
        artifacts.best_step    = ckpt.get("best_step")
    if cfg is not None and ckpt.get("config") is not None:
        saved = ckpt["config"]
        current = asdict(cfg)
        # resume_from is run bookkeeping (the resumed run necessarily sets it; the saved run
        # rarely did) -- not semantic drift.
        drift = sorted(k for k in (saved.keys() | current.keys())
                       if k not in ("resume_from", "trust_resume_checkpoint")
                       and saved.get(k) != current.get(k))
        if drift:
            import warnings
            warnings.warn(
                f"resume config drift: the checkpoint at {Path(path).name} was written under a "
                f"different config for field(s) {drift}; the resumed run uses the CURRENT values "
                f"(weights/optimizer load strictly, but semantic knobs are not restored from the "
                f"bundle).",
                UserWarning,
                stacklevel=2,
            )
    if restore_rng and ckpt.get("rng_state") is not None:
        rng = ckpt["rng_state"]
        # RNG tensors must be CPU ByteTensors regardless of map_location (set_rng_state asserts this).
        torch.set_rng_state(rng["cpu"].cpu() if hasattr(rng["cpu"], "cpu") else rng["cpu"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
    if restore_rng and metropolis_generator is not None and ckpt.get("metropolis_rng_state") is not None:
        metro_state = ckpt["metropolis_rng_state"]
        metropolis_generator.set_state(
            metro_state.cpu() if hasattr(metro_state, "cpu") else metro_state)
    if data_state is not None:
        data_state.clear()
        if saved_data_state is not None:
            data_state.update({
                "epoch_start_generator_state": saved_data_state["epoch_start_generator_state"],
                "batches_consumed":            saved_batches_consumed,
                "epoch":                       saved_epoch,
            })
    return int(ckpt["step"])


def _git_environment(
    git_executable: str,
) -> Dict[str, str]:
    r"""Minimal noninteractive environment for bounded Git provenance probes."""
    env = {
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": str(Path(git_executable).resolve().parent),
    }
    for name in ("COMSPEC", "PATHEXT", "SYSTEMROOT", "WINDIR"):
        value = os.environ.get(name)
        if value is not None:
            env[name] = value
    return env


def _git_code_identity(
    root: Optional[Path] = None,
) -> Dict[str, object]:
    r"""Return HEAD plus an exact dirty-tree fingerprint, or a persisted probe error."""
    repo = Path(__file__).resolve().parent.parent if root is None else Path(root).resolve()
    identity: Dict[str, object] = {
        "git_sha":               None,
        "git_dirty":             None,
        "git_dirty_fingerprint": None,
    }
    try:
        git_executable = shutil.which("git")
        if git_executable is None:
            raise FileNotFoundError("git executable was not found on PATH")
        env = _git_environment(git_executable)

        def _git(*args: str) -> bytes:
            return subprocess.check_output(
                [git_executable,
                 "-c", "core.fsmonitor=false",
                 "-c", f"safe.directory={repo.as_posix()}",
                 *args],
                cwd=str(repo),
                env=env,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )

        identity["git_sha"] = _git("rev-parse", "HEAD").decode("ascii").strip()
        status = _git("status", "--porcelain=v1", "-z", "--untracked-files=all")
        identity["git_dirty"] = bool(status)
        if status:
            diff = _git("diff", "--binary", "--no-ext-diff", "--no-textconv", "HEAD", "--")
            untracked = _git("ls-files", "--others", "--exclude-standard", "-z")
            digest = hashlib.sha256()
            digest.update(b"status\0")
            digest.update(status)
            digest.update(b"\0diff\0")
            digest.update(diff)
            digest.update(b"\0untracked\0")
            for raw_name in (name for name in untracked.split(b"\0") if name):
                path = repo / os.fsdecode(raw_name)
                digest.update(raw_name)
                digest.update(b"\0")
                digest.update(hashlib.sha256(path.read_bytes()).digest())
            identity["git_dirty_fingerprint"] = digest.hexdigest()
    except Exception as exc:
        identity["git_sha"] = None
        identity["git_dirty"] = None
        identity["git_dirty_fingerprint"] = None
        identity["git_error"] = repr(exc)
    return identity


def _write_provenance(
    artifacts: RunArtifacts,
    cfg:       VFE3Config,
    model:     torch.nn.Module,
    logger:    logging.Logger,

    *,
    train_loader:  Optional[Iterable] = None,
    val_loader:    Optional[Iterable] = None,
    test_loader:   Optional[Iterable] = None,
    data_seed:     Optional[int]      = None,
    max_tokens:    Optional[int]      = None,
    tokenizer_tag: Optional[str]      = None,
) -> None:
    r"""Write code, environment, per-split data, and data-order provenance best-effort."""

    prov: Dict[str, object] = {
        "seed":                cfg.seed,
        "deterministic_state": deterministic_state(),
        "n_params":            int(sum(p.numel() for p in model.parameters())),
        "torch_version":       torch.__version__,
        "cuda_version":        torch.version.cuda,
        "device_name":         (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
        "data_seed":           (int(data_seed) if data_seed is not None else None),
        "max_tokens":          (int(max_tokens) if max_tokens is not None else None),
        "tokenizer_tag":       tokenizer_tag,
    }
    prov.update(_git_code_identity())
    for split, loader in (("train", train_loader), ("val", val_loader), ("test", test_loader)):
        sha_key = f"{split}_data_sha256"
        n_key = f"{split}_data_n_tokens"
        prov[sha_key], prov[n_key] = None, None
        try:
            dataset = getattr(loader, "dataset", None)
            tokens = getattr(dataset, "tokens", None)
            if tokens is not None:
                raw = tokens.detach().cpu().numpy().tobytes()
                prov[sha_key] = hashlib.sha256(raw).hexdigest()
                prov[n_key] = int(tokens.numel())
        except Exception:
            pass
    # Backward-compatible held-out aliases consumed by existing scaling-analysis artifacts.
    prov["data_sha256"] = prov["test_data_sha256"]
    prov["data_n_tokens"] = prov["test_data_n_tokens"]
    artifacts.save_json("provenance.json", prov)
    logger.info("wrote provenance.json (git_sha=%s dirty=%s)", prov.get("git_sha"), prov.get("git_dirty"))


@torch.no_grad()
def _calibration_and_strata(
    corpus_counts: torch.Tensor,             # (V,) training-corpus unigram counts

    model:         torch.nn.Module,
    test_loader:   Iterable,
    device:        torch.device,

    *,
    max_batches: int = 20,
    n_bins:      int = 15,
) -> Dict[str, object]:
    r"""Decode calibration (ECE + reliability curve) and corpus-frequency-stratified CE over the test
    split. The decode is non-standard (KL-to-prior Mahalanobis or mu @ W^T with Sigma feeding the
    logit scale), so a mis-scaled ``decode_log_scale`` can leave PPL acceptable while the probability
    mass is wrong -- PPL alone cannot catch it. Bucket cutoffs are quantiles over the positive-count
    token types in the complete training corpus; sampled target duplication cannot move them, and
    evaluation targets unseen in training are rare. The aggregated values remain sampled held-out CE.
    The strata expose prior-table tail stagnation. Off-graph; capped at ``max_batches``."""
    import torch.nn.functional as F

    confs, corrects, nats, tgts = [], [], [], []
    for i, (tok, tgt) in enumerate(test_loader):
        tok, tgt = tok.to(device), tgt.to(device)
        logits = model(tok)                                     # (B, N, V) inference path
        lp = logits.reshape(-1, logits.shape[-1]).float()
        t = tgt.reshape(-1)
        valid = t != -100
        lp, t = lp[valid], t[valid]
        prob = torch.softmax(lp, dim=-1)
        p_max, pred = prob.max(dim=-1)
        confs.append(p_max)
        corrects.append((pred == t).float())
        nats.append(F.cross_entropy(lp, t, reduction="none"))
        tgts.append(t)
        if i + 1 >= max_batches:
            break
    if not confs:
        return {}
    conf, corr = torch.cat(confs), torch.cat(corrects)
    nat, tg = torch.cat(nats), torch.cat(tgts)

    edges = torch.linspace(0.0, 1.0, n_bins + 1, device=conf.device)
    ece, rel = 0.0, []
    for b in range(n_bins):                                     # expected calibration error (15-bin)
        m = (conf > edges[b]) & (conf <= edges[b + 1])
        if m.any():
            acc, cf, w = corr[m].mean(), conf[m].mean(), m.float().mean()
            ece += float(w * (acc - cf).abs())
            rel.append({"conf": float(cf), "acc": float(acc), "frac": float(w)})
    if corpus_counts.ndim != 1:
        raise ValueError("corpus_counts must be a one-dimensional training-corpus bincount")
    counts = corpus_counts.to(device=tg.device)
    if int(tg.max()) >= counts.numel():
        raise ValueError("corpus_counts does not cover every sampled evaluation target")
    positive_counts = counts[counts > 0].float()
    if positive_counts.numel() == 0:
        q1, q2 = 0.0, 0.0
    else:
        quantiles = positive_counts.new_tensor([1 / 3, 2 / 3])
        q1, q2 = torch.quantile(positive_counts, quantiles).tolist()
    tok_count = counts[tg].float()                              # training-corpus count of each target
    seen = tok_count > 0
    strata = {}
    for name, mask in (("rare", (~seen) | (tok_count <= q1)),
                       ("mid", seen & (tok_count > q1) & (tok_count <= q2)),
                       ("frequent", seen & (tok_count > q2))):
        strata[name] = float(nat[mask].mean()) if mask.any() else float("nan")
    return {"ece": ece, "reliability": rel, "overall_ce": float(nat.mean()),
            "corpus_freq_strata_ce": strata}


def _fd_gradient_check(
    model:       torch.nn.Module,
    test_loader: Iterable,
    device:      torch.device,

    *,
    n_coords:    int   = 4,
    fd_eps:      float = 1e-3,
) -> float:
    r"""Worst relative error between autograd-of-CE and a central finite difference on a few DECODE
    coordinates (``output_proj_weight``, else the decode log-scale) -- a parameter whose gradient does
    NOT pass through the E-step belief adjoint (which the default kernel/oracle route detaches), so a
    healthy model reads ~1e-4 and a broken decode adjoint spikes far above it. (Probing ``mu_embed``
    instead would sit at the detached-oracle's ~10-25% plateau with no headroom to flag a real bug.)
    The CLAUDE.md-mandated FD-vs-autograd check. Best-effort on one tiny batch; restores every coord."""
    batch = next(iter(test_loader))
    tok, tgt = (batch if isinstance(batch, (tuple, list)) else (batch, None))
    tok = tok[:2].to(device)
    tgt = tgt[:2].to(device)
    pb = model.prior_bank
    p = pb.output_proj_weight if getattr(pb, "output_proj_weight", None) is not None else pb.decode_log_scale
    model.zero_grad(set_to_none=True)
    _, loss, _ = model(tok, tgt)
    loss.backward()
    if p.grad is None:                                          # decode param severed under this config
        model.zero_grad(set_to_none=True)
        return float("nan")
    flat, gflat = p.detach().view(-1), p.grad.detach().view(-1).clone()
    # Probe the LARGEST-gradient coords, not random ones: a random coord usually has near-zero
    # gradient where the central difference is dominated by fp rounding (a spurious large rel error),
    # so this checks the gradient where its signal actually dominates -- where a real adjoint bug shows.
    idx = torch.topk(gflat.abs(), min(n_coords, gflat.numel())).indices.tolist()
    worst = 0.0
    with torch.no_grad():
        for j in idx:
            orig = float(flat[j])
            flat[j] = orig + fd_eps
            _, lp, _ = model(tok, tgt)
            flat[j] = orig - fd_eps
            _, lm, _ = model(tok, tgt)
            flat[j] = orig
            fd = (float(lp) - float(lm)) / (2.0 * fd_eps)
            ana = float(gflat[j])
            worst = max(worst, abs(fd - ana) / max(abs(fd), abs(ana), 1e-8))
    model.zero_grad(set_to_none=True)
    return worst


def _write_research_artifacts(
    model:       torch.nn.Module,
    artifacts:   RunArtifacts,
    cfg:         VFE3Config,
    train_loader: Optional[Iterable],
    test_loader: Optional[Iterable],
    device:      torch.device,
    logger:      logging.Logger,
) -> None:
    r"""Best-effort ``research.json``: decode calibration (ECE) + frequency-stratified loss + the FD
    gradient-check residual. Each probe is independently guarded so one failure never blocks the
    others or the saved numeric results."""
    if test_loader is None:
        return
    out: Dict[str, object] = {}
    try:
        train_dataset = getattr(train_loader, "dataset", None)
        train_tokens = getattr(train_dataset, "tokens", None)
        if train_tokens is None:
            raise ValueError("training loader dataset does not expose corpus tokens")
        corpus_counts = torch.bincount(
            train_tokens.detach().reshape(-1).to(device="cpu", dtype=torch.long),
            minlength=int(cfg.vocab_size),
        )
        out.update(_calibration_and_strata(corpus_counts, model, test_loader, device))
    except Exception as exc:
        logger.warning("calibration/strata probe failed (%s); skipped", exc)
    try:
        out["fd_gradient_worst_rel_error"] = _fd_gradient_check(model, test_loader, device)
        logger.info("FD gradient-check worst rel error: %.2e", out["fd_gradient_worst_rel_error"])
    except Exception as exc:
        logger.warning("FD gradient-check failed (%s); skipped", exc)
    # B1/EXP-3 Sigma_q calibration headline: Spearman rho(tr Sigma_q, CE) and the across-token
    # spread gate CV(tr Sigma_q) > 0.10 (below it the covariance channel is inert -- reported as
    # such, NOT miscoded as "decode doesn't matter"). Off-graph; capped at a few batches.
    try:
        from vfe3.viz.extract import belief_ce_bank
        from vfe3 import metrics as _cal_metrics
        bank = belief_ce_bank(model, test_loader, device=device, max_batches=10)
        tr = bank["tr_sigma"]
        if tr.numel() >= 2:
            out["sigma_ce_spearman"] = _cal_metrics.spearman_rho(tr, bank["ce"])
            out["sigma_trace_cv"] = _cal_metrics.cv(tr)
            out["sigma_trace_cv_gate_pass"] = bool(out["sigma_trace_cv"] > 0.10)
            logger.info("Sigma_q calibration: rho(trSigma,CE)=%.3f CV(trSigma)=%.3f (gate>0.10: %s)",
                        out["sigma_ce_spearman"], out["sigma_trace_cv"], out["sigma_trace_cv_gate_pass"])
    except Exception as exc:
        logger.warning("Sigma_q calibration probe failed (%s); skipped", exc)
    if out:
        artifacts.save_json("research.json", out)


def _cost_model_fields(
    model:       torch.nn.Module,
    cfg:         VFE3Config,

    n_params:    int,
    tokens_seen: int,

    *,
    wall_time:   Optional[float] = None,
) -> Dict[str, object]:
    r"""Structural axes + a faithful compute proxy for the scaling frontier (extends scaling_point).

    The ``6ND`` rule (``6 * n_params * tokens_seen``) is LOOSE here: ``n_params`` is dominated by the
    vocab-size gauge/prior tables (``phi_embed`` is ``V * n_gen``), but only the active tokens' rows
    participate per step, while the decode reads all ``V`` rows every forward. So this records
    (a) the structural axes that set the real per-token work, (b) ``active_params_per_token`` -- the
    honest working set (decode-bound, ~``K``, NOT ``phi``/``n_gen``-bound, the mirror image of
    ``n_params`` being ``n_gen``-dominated), and (c) a transparent analytic FLOP proxy assembled from
    those drivers (order-1 constants; for a calibrated frontier use ``wall_time`` or a profiler).
    ``wall_time`` on a fixed GPU is the empirical ground truth the analytic constants calibrate
    against. ``n_gen`` / ``n_blocks`` are read from the GROUP OBJECT so they track ``cross_couplings``,
    bracket closure, and the ``so_n``/``sp_n`` decoupling of ``n_gen`` from ``K``.
    """
    V, K = int(cfg.vocab_size), int(cfg.embed_dim)
    n_gen = int(model.group.generators.shape[0])
    n_blocks = max(1, len(model.group.irrep_dims))
    d_head = K / n_blocks                                        # representative block dim
    model_channel = (cfg.lambda_h > 0.0 or cfg.lambda_gamma > 0.0
                     or cfg.prior_source == "model_channel" or cfg.s_e_step)
    # ACTIVE params per token: the single looked-up belief row is always 2K+n_gen. The decoder then
    # reads EITHER the prior-bank mean/variance rows (2VK) OR the linear output matrix (VK) and its
    # optional V-vector bias. The V*n_gen phi bulk is not touched by either full-vocabulary readout.
    token_row = 2 * K + n_gen
    if cfg.use_prior_bank:
        decode_readout = 2 * V * K
    else:
        decode_readout = V * K + (V if cfg.decode_bias else 0)
    active = token_row + decode_readout
    if model_channel:
        active += 2 * V * K                                     # s tables enter encode/decode
    # Transparent analytic FLOP proxy. Per token: decode over all V (2VK), L*T belief E-step
    # iterations, and one T-iteration model-channel refinement when s_e_step is enabled. Each E-step
    # iteration has O(N) attention energy (2NK) plus O(N) transport application (2N*d_head^2).
    # Constants are O(1); this is a proxy, not a calibrated count.
    L, T, N = int(cfg.n_layers), int(cfg.n_e_steps), int(cfg.max_seq_len)
    fpt_decode         = 2.0 * V * K
    estep_kernel       = 2.0 * N * K + 2.0 * N * d_head * d_head
    belief_estep       = L * T * estep_kernel
    model_estep        = T * estep_kernel if cfg.s_e_step else 0.0
    fpt_estep          = belief_estep + model_estep
    est_flops_analytic = (fpt_decode + fpt_estep) * float(tokens_seen)
    out: Dict[str, object] = {
        "embed_dim":               K,
        "n_heads":                 int(cfg.n_heads),
        "n_blocks":                n_blocks,
        "n_gen":                   n_gen,
        "n_layers":                L,
        "n_e_steps":               T,
        "max_seq_len":             N,
        "batch_size":              int(cfg.batch_size),
        "diagonal_covariance":     bool(cfg.diagonal_covariance),
        "gauge_group":             cfg.gauge_group,
        "use_prior_bank":          bool(cfg.use_prior_bank),
        "model_channel_active":    bool(model_channel),
        "vocab_size":              V,
        "n_learnable_params":      int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "active_params_per_token": int(active),
        "est_flops_analytic":      est_flops_analytic,
        "flops_per_token_decode":  fpt_decode,
        "flops_per_token_estep":   fpt_estep,
        "device_name":             (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
        "amp_dtype":               cfg.amp_dtype,
    }
    if wall_time is not None and tokens_seen > 0:
        out["wall_time_s"]         = float(wall_time)
        out["wall_time_per_token"] = float(wall_time) / float(tokens_seen)
        out["wall_time_per_step"]  = float(wall_time) / max(1, int(cfg.max_steps))
    return out


def finalize_run(
    model:       torch.nn.Module,
    artifacts:   RunArtifacts,
    cfg:         VFE3Config,

    *,
    tokens_per_char: float                    = 1.0,   # test BPC char-correction (1.0 = bits/token)
    train_loader:    Optional[Iterable]       = None,
    val_loader:      Optional[Iterable]       = None,
    test_loader:     Optional[Iterable]       = None,
    losses:          Optional[List[float]]    = None,
    data_seed:       Optional[int]            = None,
    max_tokens:      Optional[int]            = None,
    tokenizer_tag:   Optional[str]            = None,
    device:          Optional[torch.device]   = None,
    wall_time:       Optional[float]          = None,
    logger:          Optional[logging.Logger] = None,
) -> Dict[str, object]:
    r"""Reload the best-val checkpoint, score the TEST split, and write summary + figures.

    The headline metric is the test perplexity of the BEST-validation model (the periodic eval
    saved ``best_model.pt`` at the lowest val PPL); we reload it so the reported test number is
    not the final, possibly-overfit live weights. If no checkpoint was written (no validation
    configured), the live model is scored. Returns the test-results dict.
    """
    from vfe3.train import evaluate                              # local import avoids an import cycle

    logger = logger or logging.getLogger(__name__)
    if device is None:
        device = next(model.parameters()).device

    reloaded_best = False
    if artifacts.best_path.exists():
        bundle = torch.load(artifacts.best_path, map_location=device, weights_only=True)
        if not isinstance(bundle, Mapping) or not {
                "model_state", "config", "config_fingerprint"}.issubset(bundle):
            raise ValueError(
                f"best checkpoint {artifacts.best_path} is not a semantic best-model bundle")
        saved_config = bundle["config"]
        if not isinstance(saved_config, Mapping):
            raise ValueError(
                f"best checkpoint {artifacts.best_path} has a non-mapping config")
        saved_fingerprint = semantic_config_fingerprint(saved_config)
        if bundle["config_fingerprint"] != saved_fingerprint:
            raise ValueError(
                f"best checkpoint {artifacts.best_path} has a config fingerprint mismatch")
        if saved_fingerprint != semantic_config_fingerprint(asdict(cfg)):
            raise ValueError(
                f"best checkpoint {artifacts.best_path} does not match the active config")
        model.load_state_dict(bundle["model_state"])
        reloaded_best = True
        logger.info("Reloaded best-val checkpoint (step %s, val PPL %.3f) for test eval",
                    artifacts.best_step, artifacts.best_val_ppl)

    results: Dict[str, object] = {}                             # mixes float / Optional[float|int] / bool
    if test_loader is not None:
        m = evaluate(model, test_loader, tokens_per_char=tokens_per_char, device=device)
        results = {"test_ce": m["ce"], "test_ppl": m["ppl"], "test_bpc": m["bpc"]}
        logger.info("Test (held-out) | CE: %.4f | PPL: %.2f | BPC: %.4f",
                    m["ce"], m["ppl"], m["bpc"])
    best_val_ppl = artifacts.best_val_ppl if artifacts.best_val_ppl != float("inf") else None
    results.update({"best_val_ppl": best_val_ppl, "best_step": artifacts.best_step,
                    "reloaded_best": reloaded_best})

    # E-step inference-time value: test CE with the inner E-step DISABLED (n_e_steps=0 -> belief =
    # prior, the loop runs zero iterations) minus the configured-budget test CE. NOTE this is the
    # INFERENCE-TIME marginal value of the E-step under tables that were TRAINED with it (the M-step
    # co-adapts the priors to the refinement) -- NOT a clean capacity split into table vs E-step, which
    # would need a second model trained at n_e_steps=0. A near-zero value still flags an E-step that
    # buys little at inference. Off-graph, best-effort; n_e_steps is restored in the finally.
    if test_loader is not None and results.get("test_ce") is not None:
        _saved_ne = model.cfg.n_e_steps
        try:
            model.cfg.n_e_steps = 0
            m0 = evaluate(model, test_loader, tokens_per_char=tokens_per_char, device=device)
            results["test_ce_no_estep"]    = m0["ce"]
            results["estep_capacity_gain"] = m0["ce"] - results["test_ce"]
            logger.info("E-step capacity gain (CE@n_e_steps=0 - CE@%d): %.4f",
                        _saved_ne, results["estep_capacity_gain"])
        except Exception as exc:
            logger.warning("estep capacity-gain probe failed (%s); skipped", exc)
        finally:
            model.cfg.n_e_steps = _saved_ne

    # EXP-5 (C2): the converged final E-step free energy PER TOKEN -- the E-step's OWN target-blind
    # functional value (free_energy_value sums F over the N tokens; divide by N). Persisted so a
    # cross-arm reader (scaling_analysis) can test whether final F DECORRELATES from CE across an
    # n_e_steps sweep -- the structural non-Neal-Hinton EM prediction (the E-step serves a distinct
    # functional, not the likelihood). Off-graph, best-effort, on a fixed test batch (sequence 0).
    if test_loader is not None:
        try:
            from vfe3.viz.extract import e_step_belief_trace
            _b = next(iter(test_loader))
            _tok = (_b[0] if isinstance(_b, (tuple, list)) else _b).to(device)
            _tr = e_step_belief_trace(model, _tok)              # n_iter defaults to cfg.n_e_steps
            results["estep_final_f_per_token"] = float(_tr["free_energy"][-1]) / max(1, int(_tok.shape[1]))
            logger.info("Converged final E-step F/token: %.4f", results["estep_final_f_per_token"])
        except Exception as exc:
            logger.warning("estep final-F probe failed (%s); skipped", exc)
    artifacts.save_json("test_results.json", results)

    # Reproducibility provenance (git SHA / data hash / versions) + a scaling-law data point -- the
    # externally-grounded records a config-only artifact omits (identical config.json can come from
    # different code and data, and a single run carries no (N, tokens, FLOPs, loss) frontier point).
    _write_provenance(
        artifacts,
        cfg,
        model,
        logger,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        data_seed=data_seed,
        max_tokens=max_tokens,
        tokenizer_tag=tokenizer_tag,
    )
    n_params = int(sum(p.numel() for p in model.parameters()))
    tokens_seen = int(cfg.max_steps) * int(cfg.batch_size) * int(cfg.max_seq_len)
    # scaling-law data point: the 6ND FLOP proxy is LOOSE for a no-NN E-step model, so record the
    # inputs too (a cross-run frontier can be re-fit offline with the right cost model). The
    # _cost_model_fields block adds the structural axes + active-params-per-token + a faithful
    # analytic proxy so each point is standalone; best-effort, never blocks the saved numbers.
    scaling_point: Dict[str, object] = {
        "n_params":      n_params,
        "tokens_seen":   tokens_seen,
        "est_flops_6ND": 6 * n_params * tokens_seen,
        "test_ce":       results.get("test_ce"),
    }
    try:
        scaling_point.update(_cost_model_fields(model, cfg, n_params, tokens_seen, wall_time=wall_time))
    except Exception as exc:
        logger.warning("cost-model fields failed (%s); scaling_point keeps the 6ND proxy only", exc)
    artifacts.save_json("summary.json", {
        "n_steps":      cfg.max_steps,
        "n_params":     n_params,
        "best_val_ppl": best_val_ppl,
        "best_step":    artifacts.best_step,
        "reloaded_best": results.get("reloaded_best"),   # m26: False on a cross-run-dir resume whose best_model.pt is elsewhere
        "test_ppl":     results.get("test_ppl"),
        "test_ce":      results.get("test_ce"),
        "test_bpc":     results.get("test_bpc"),
        "test_ce_no_estep":    results.get("test_ce_no_estep"),
        "estep_capacity_gain": results.get("estep_capacity_gain"),
        "estep_final_f_per_token": results.get("estep_final_f_per_token"),
        "final_train_loss": (losses[-1] if losses else None),
        "wall_time_s":  wall_time,
        "use_prior_bank":  cfg.use_prior_bank,
        "use_head_mixer":  cfg.use_head_mixer,
        "scaling_point":   scaling_point,
    })

    # Pure-path certificate: the config toggles for the principal gauge / decode / free-energy purity
    # axes (flat gauge, canonical F, prior-bank decode, no head mixer, ...), plus the converged-state
    # stress metrics that say whether the numerical guards stayed inert. A REPORT of where the run sits,
    # not a judgment that any toggle is wrong (toggles are changed intentionally). Best-effort.
    try:
        artifacts.save_json("pure_path_report.json", _pure_path_report(cfg, artifacts.history))
    except Exception as exc:
        logger.warning("pure-path report failed (%s); skipped", exc)

    # Research artifacts (decode calibration / corpus-frequency-stratified loss / FD gradient check) --
    # externally-grounded probes that do NOT presuppose the gauge framework. Best-effort, AFTER the
    # test-eval n_e_steps restore so the model is in its trained state. Run before the figure pass.
    _write_research_artifacts(model, artifacts, cfg, train_loader, test_loader, device, logger)

    _save_figures(artifacts, losses, logger)
    # Single-run publication figure set (model-replay), auto-run at the end of training unless
    # cfg.generate_figures is False. Best-effort and off the hot path -- the runners are expensive
    # (UMAP, E-step replay, holonomy sampling, a belief bank over many sequences), so a failure is
    # logged and never disturbs the saved numeric results. Drives the BEST-val model reloaded above.
    if getattr(cfg, "generate_figures", True):
        # Memory-budget guard (audit 2026-07-01 F9): the figure extractors materialize dense
        # full-vocab (B, N, V) logits + probabilities, which on a large run (e.g. V=50257, N=1024,
        # B=16) can make finalization the memory peak. Skip the figure pass over an ~8 GB estimate
        # unless the run opts in via force_large_figures; small runs stay below the threshold, so
        # the default-on behavior is byte-identical there.
        approx_gb = 8.0 * int(cfg.vocab_size) * int(cfg.max_seq_len) * int(cfg.batch_size) / 1e9   # fp32 logits+probs (B, N, V) peak
        if approx_gb > 8.0 and not getattr(cfg, "force_large_figures", False):
            logger.warning("skipping publication figures: est full-vocab peak ~%.1f GB exceeds "
                           "8 GB guard; set force_large_figures=True to override", approx_gb)
        else:
            try:
                from vfe3.viz.report import generate_figures
                generate_figures(artifacts.run_dir, model=model, loader=test_loader,
                                 device=device, logger=logger)
            except Exception as exc:
                logger.warning("publication figure generation failed (%s); numeric results are saved", exc)
    return results


def _pure_path_report(cfg: VFE3Config, history: List[Dict]) -> Dict:
    r"""Where a run sits relative to the theoretically pure path: the toggle states that define it plus
    the converged-state stress metrics that say whether the numerical guards stayed inert.

    A REPORT, not a verdict. The pure path must EXIST under appropriate toggles, but the user changes
    toggles intentionally, so a non-pure run is recorded (``on_pure_path=False`` with the offending
    flags), never flagged as wrong. ``pure_flags`` covers the principal gauge / decode / free-energy
    purity axes (canonical attention entropy, flat transport, constant/static coupling weights,
    prior-bank decode, full sigma updates, no two-hop/fixed-prior surrogate, no head mixer,
    unweighted attention); it does NOT enumerate every default-OFF learned-scalar toggle
    (pos_phi, learnable_r, t5_learnable_bias, use_cg_coupling),
    so ``on_pure_path`` certifies these axes rather than a full no-learned-parameter audit.
    ``gauge_flags``/``on_gauge_pure_path`` is a SECOND, independent axis (audit 2026-07-01 F8): the
    gauge / model-channel path (learned gauge transport, phi parameterization, no reflection or
    positional rotation, family/group invariance, no model-channel coupling) -- a run can be pure on
    the free-energy/decode axis while a gauge setting alters the executed belief path, and vice versa.
    ``converged_stress`` reads the last finite value of each guard / flatness column (None if absent)."""
    def _last(key: str) -> Optional[float]:
        for r in reversed(history):
            v = r.get(key)
            if isinstance(v, (int, float)) and math.isfinite(v):
                return float(v)
        return None
    from vfe3.geometry.groups import get_group

    group_builder = get_group(cfg.gauge_group)
    invariant_families = tuple(getattr(group_builder, "invariant_families", ()))
    family_group_invariant = cfg.family in invariant_families
    fixed_prior_surrogate = bool(cfg.precision_weighted_attention)

    pure_flags = {
        "canonical_attention_entropy": bool(cfg.include_attention_entropy),
        "flat_transport":              cfg.transport_mode == "flat",
        "constant_lambda_alpha":       cfg.lambda_alpha_mode == "constant",
        "prior_bank_decode":           bool(cfg.use_prior_bank),
        "no_head_mixer":               not cfg.use_head_mixer,
        "unweighted_attention":        not cfg.precision_weighted_attention,
        "full_sigma_update":           not cfg.skip_belief_sigma_update,
        "no_twohop_coupling":          cfg.lambda_twohop == 0.0,
        "no_fixed_prior_surrogate":    not fixed_prior_surrogate,
    }
    # Second, INDEPENDENT purity axis (audit 2026-07-01 F8): the gauge / model-channel path. Keyed
    # on pos_rotation itself rather than the RoPE sub-toggles (rope_full_gauge / rope_on_value),
    # which are inert while RoPE is off -- those are reported in config_toggles for transparency.
    gauge_flags = {
        "learned_gauge_transport":   cfg.gauge_transport == "on",
        "no_positional_rotation":    cfg.pos_rotation == "none",
        "no_model_channel_coupling": cfg.lambda_gamma == 0.0 and not cfg.s_e_step,
        "phi_parameterization":      cfg.gauge_parameterization == "phi",
        "no_reflection_sampling":    cfg.omega_reflection == "off" and cfg.phi_reflection == "off",
        "family_group_invariant":    family_group_invariant,
    }
    return {
        "on_pure_path":       all(pure_flags.values()),
        "pure_flags":         pure_flags,
        "gauge_flags":        gauge_flags,
        "on_gauge_pure_path": all(gauge_flags.values()),
        "config_toggles": {
            "include_attention_entropy":    bool(cfg.include_attention_entropy),
            "transport_mode":               cfg.transport_mode,
            "lambda_alpha_mode":            cfg.lambda_alpha_mode,
            "lambda_beta":                  float(cfg.lambda_beta),
            "use_prior_bank":               bool(cfg.use_prior_bank),
            "use_head_mixer":               bool(cfg.use_head_mixer),
            "precision_weighted_attention": bool(cfg.precision_weighted_attention),
            "gauge_transport":              cfg.gauge_transport,
            "pos_rotation":                 cfg.pos_rotation,
            "rope_full_gauge":              bool(cfg.rope_full_gauge),
            "rope_on_value":                bool(cfg.rope_on_value),
            "lambda_gamma":                 float(cfg.lambda_gamma),
            "s_e_step":                     bool(cfg.s_e_step),
            "skip_belief_sigma_update":      bool(cfg.skip_belief_sigma_update),
            "lambda_twohop":                 float(cfg.lambda_twohop),
            "gauge_parameterization":        cfg.gauge_parameterization,
            "omega_reflection":              cfg.omega_reflection,
            "phi_reflection":                cfg.phi_reflection,
            "gauge_group":                   cfg.gauge_group,
            "family":                        cfg.family,
            "group_invariant_families":      list(invariant_families),
            # Truthful fixed-surrogate ledger (C6): these derived booleans expose when the run
            # intentionally freezes a state-dependent quantity rather than following its full
            # joint objective. Defaults are False, preserving the pure path.
            "fixed_covariance_surrogate":   bool(getattr(cfg, "skip_belief_sigma_update", False)),
            "detached_precision_prior":     fixed_prior_surrogate,
            "detached_query_adaptive_tau":  bool(getattr(cfg, "query_adaptive_tau", False)),
            "state_dependent_alpha_majorizer": (
                getattr(cfg, "e_step_update", "gradient") == "mm_exact"
                and cfg.lambda_alpha_mode in ("state_dependent", "state_dependent_per_coord")
            ),
            # regime_ii_covariant under gaussian_diagonal is a CONTROLLED APPROXIMATION (the
            # diagonal cone is not closed under GL congruence Omega Sigma Omega^T -- audit C5),
            # so a diagonal covariant run is never reported as exact Route B.
            "regime_ii_covariant_exact":    (cfg.transport_mode != "regime_ii_covariant")
                                            or family_group_invariant,
            # Covariance class of the ACTIVE transport (audit C7): plain regime_ii's bilinear edge
            # delta_ij = mu_i^T W mu_j is gauge-FIXED (invariant only at W=0), never covariant.
            # .get default: a newly registered mode reports its own name rather than KeyError-ing.
            "transport_covariance_class":   {"flat":                   "covariant (flat)",
                                             "regime_ii":              "gauge-fixed (non-covariant)",
                                             "regime_ii_covariant":    "covariant",
                                             "regime_ii_link":         "gauge-fixed",
                                             "regime_ii_link_charted": "covariant",
                                             }.get(cfg.transport_mode, cfg.transport_mode),
        },
        "converged_stress": {k: _last(k) for k in (
            "guard_sigma_floor_frac", "guard_sigma_ceil_frac", "guard_energy_klmax_frac",
            "guard_selfdiv_klmax_frac", "nonfinite_frac", "renyi_band_frac",
            "cocycle_residual", "holonomy_wilson", "gauge_invariant_spread")},
    }


def _save_figures(
    artifacts: RunArtifacts,
    losses:    Optional[List[float]],
    logger:    logging.Logger,
) -> None:
    r"""Best-effort publication figures from the logged history (no model re-run)."""
    try:
        from vfe3.viz import figures as figs
        figs.set_publication_style()
        run = artifacts.run_dir

        def _aligned(key: str) -> tuple:
            r"""Aligned (step, value) for a history column, dropping pre-first-eval NaN rows."""
            xs, ys = [], []
            for i, r in enumerate(artifacts.history):
                if key in r and math.isfinite(r[key]):
                    xs.append(r.get("step", i))
                    ys.append(r[key])
            return xs, ys

        def _hist_subset(keys: tuple) -> Optional[Dict]:
            r"""A ``{step, key: [...]}`` history dict over ``keys`` present (finite on >= 1 row), each a
            full-length column with NaN where missing so an eval-cadence key keeps its step alignment and
            the dashboard masks it per series. Returns None when no key is present (caller skips)."""
            present = [k for k in keys
                       if any(k in r and isinstance(r[k], (int, float)) and math.isfinite(r[k])
                              for r in artifacts.history)]
            if not present:
                return None
            cols: Dict = {"step": [r.get("step", i) for i, r in enumerate(artifacts.history)]}
            for k in present:
                cols[k] = [float(r[k]) if (k in r and isinstance(r[k], (int, float)) and math.isfinite(r[k]))
                           else float("nan") for r in artifacts.history]
            return cols

        if losses:
            # losses is one entry per optimizer step, so the 1-based step index IS the x-axis.
            n = len(losses)
            fig = figs.plot_trajectory(
                losses, list(range(1, n + 1)), ylabel="train CE (nats/token)",
                title="Training cross-entropy", color=figs._CB[0],
                smooth=max(25, n // 240), annotate_final=True,
                path=str(run / "loss_curve.png"))
            figs.plt.close(fig)
        sx, sy = _aligned("val_ppl")
        if sy:
            fig = figs.plot_trajectory(
                sy, sx, ylabel="validation perplexity", title="Validation perplexity",
                color=figs._CB[1], logy=True, smooth=max(5, len(sy) // 80), annotate="min",
                path=str(run / "val_ppl.png"))
            figs.plt.close(fig)
        # Gauge-geometry trajectories (diagnostics tier): curvature proxy + gauge-trace spread.
        hx, hy = _aligned("holonomy_deviation")
        if hy:
            # Heavy-tailed (median ~1e-3, rare spikes ~1e3): log y + a median reference; NOT smoothed,
            # so the curvature spikes survive.
            fig = figs.plot_trajectory(
                hy, hx, ylabel=r"$\langle\|H_{ijk}-I\|_F\rangle$",
                title="Holonomy deviation (frame-dependent Frobenius)", color=figs._CB[2],
                logy=True, median_line=True, annotate="max",
                path=str(run / "holonomy.png"))
            figs.plt.close(fig)
        gx, gy = _aligned("gauge_trace_spread")
        if gy:
            fig = figs.plot_trajectory(
                gy, gx, ylabel=r"std $\log|\det\Omega|$", title="Gauge trace spread",
                color=figs._CB[3], smooth=max(5, len(gy) // 60), annotate_final=True,
                path=str(run / "gauge_trace_spread.png"))
            figs.plt.close(fig)
        # Learnable softmax-temperature trajectories: present exactly when train() logged the live
        # per-block kappa statistics for the default-off learnable_kappa_beta/gamma toggles.
        for _ch in ("beta", "gamma"):
            _hist_kappa = _hist_subset((f"kappa_{_ch}_mean", f"kappa_{_ch}_var"))
            if _hist_kappa and f"kappa_{_ch}_mean" in _hist_kappa:
                fig = figs.plot_kappa_history(
                    _hist_kappa, channel=_ch, path=str(run / f"kappa_{_ch}_history.png"))
                figs.plt.close(fig)
        # Per-irrep-block companion to the aggregate kappa_<ch>_history above: one line per block for
        # kappa AND the effective temperature tau, across the beta/gamma channels (a 2x2 grid when
        # both toggles are on). Present exactly when train() logged the per-block kappa_*/tau_* columns.
        _kb_keys = tuple(sorted({k for r in artifacts.history for k in r
                                 if k.startswith(("kappa_beta_b", "kappa_gamma_b",
                                                  "tau_beta_b", "tau_gamma_b"))}))
        if _kb_keys:
            _hist_kb = _hist_subset(_kb_keys)
            if _hist_kb:
                fig = figs.plot_kappa_block_trajectory(
                    _hist_kb, path=str(run / "kappa_block_trajectory.png"))
                figs.plt.close(fig)
        # Optimization + convergence trends (history-only; no model re-run): the pre-clip gradient
        # norm (THE optimization-health curve, previously discarded), the belief-covariance
        # conditioning, and the per-eval E-step F-descent (negative = the inner loop reduced F).
        nx, ny = _aligned("grad_norm")
        if ny:
            fig = figs.plot_trajectory(
                ny, nx, ylabel=r"$\|\nabla\|_2$ (pre-clip)", title="Gradient norm",
                color=figs._CB[5 % len(figs._CB)], logy=True, smooth=max(5, len(ny) // 80),
                annotate="max", path=str(run / "grad_norm.png"))
            figs.plt.close(fig)
        # M-step per-role gradient-norm decomposition (mu / sigma / phi): the parameter-learning
        # channels the aggregate grad_norm.png folds together. Columns logged by train_step (aggregated
        # by each optimizer group's "role" tag, so the live tables are attributed correctly under any
        # config); present only on a run that captured step_metrics, so gate on their presence (the CSV
        # stays rectangular per run). A SEPARATE figure from the aggregate, same pre-clip magnitudes.
        gd_keys = ("grad_norm_mu", "grad_norm_sigma", "grad_norm_phi")
        gd_present = [k for k in gd_keys
                      if any(k in r and math.isfinite(r[k]) for r in artifacts.history)]
        if gd_present:
            gd_rows = [r for r in artifacts.history
                       if all(k in r and math.isfinite(r[k]) for k in gd_present)]
            if gd_rows:
                hist_gd = {"step": [r.get("step", i) for i, r in enumerate(gd_rows)],
                           **{k: [r[k] for r in gd_rows] for k in gd_present}}
                fig = figs.plot_grad_norm_decomposition(hist_gd, path=str(run / "grad_norm_decomposition.png"))
                figs.plt.close(fig)
        # E-step belief-gradient decomposition (mu / sigma / phi): the INFERENCE analogue of the M-step
        # figure above -- ||grad F|| over the belief tuple per inner-loop component, logged by train_step
        # from model.forward's estep_grad_out. Accumulated runs prefer the explicitly named arithmetic
        # microbatch means; single-batch runs retain the historical column names. Same presence gate;
        # independent of the M-step columns, so build its own row set.
        eg_mean_keys = (
            "estep_grad_norm_mu_microbatch_mean",
            "estep_grad_norm_sigma_microbatch_mean",
            "estep_grad_norm_phi_microbatch_mean",
        )
        eg_keys = (eg_mean_keys if any(any(k in r for k in eg_mean_keys)
                                      for r in artifacts.history)
                   else ("estep_grad_norm_mu", "estep_grad_norm_sigma", "estep_grad_norm_phi"))
        eg_present = [k for k in eg_keys
                      if any(k in r and math.isfinite(r[k]) for r in artifacts.history)]
        if eg_present:
            eg_rows = [r for r in artifacts.history
                       if all(k in r and math.isfinite(r[k]) for k in eg_present)]
            if eg_rows:
                hist_eg = {"step": [r.get("step", i) for i, r in enumerate(eg_rows)],
                           **{k: [r[k] for r in eg_rows] for k in eg_present}}
                fig = figs.plot_estep_grad_norm_decomposition(hist_eg, path=str(run / "estep_grad_norm_decomposition.png"))
                figs.plt.close(fig)
        cx, cy = _aligned("belief_cond_median")
        if cy:
            fig = figs.plot_trajectory(
                cy, cx, ylabel=r"median $\lambda_{\max}/\lambda_{\min}$",
                title="Belief covariance conditioning", color=figs._CB[6 % len(figs._CB)],
                logy=True, smooth=max(5, len(cy) // 80), annotate="max",
                path=str(run / "belief_condition.png"))
            figs.plt.close(fig)
        ex, ey = _aligned("estep_f_drop")
        if ey:
            fig = figs.plot_trajectory(
                ey, ex, ylabel=r"$F_{\mathrm{end}}-F_{\mathrm{start}}$ (inner E-step)",
                title="E-step free-energy descent", color=figs._CB[2 % len(figs._CB)],
                median_line=True, path=str(run / "estep_convergence_trend.png"))
            figs.plt.close(fig)
        # Free-energy figures: the per-token budget DECOMPOSITION (snapshot + early/mid/late evolution)
        # and, as a SEPARATE figure, the F-vs-CE CO-DESCENT over training. Both need every plotted term
        # finite, so rows before the first eval (NaN val_*) are dropped.
        fe_keys = ("self_coupling", "belief_coupling", "attention_entropy", "val_ce")
        fe_rows = [r for r in artifacts.history
                   if all(k in r and math.isfinite(r[k]) for k in fe_keys)]
        if fe_rows:
            cfg = getattr(artifacts, "cfg", None)
            # Model-channel F components fold into the complexity-F total when the channel is live;
            # included only when present on EVERY plotted row (model-channel run). hyper_prior_weighted
            # is the EXACT weighted hyper-prior (state_dependent lambda_h != cfg.lambda_h*raw,
            # so it is read directly); the gamma blocks are scaled by cfg.lambda_gamma in the figure,
            # exactly as the belief block is scaled by lambda_beta.
            mc_fe_keys = [k for k in ("hyper_prior_weighted", "gamma_coupling", "gamma_meta_entropy")
                          if all(k in r and math.isfinite(r[k]) for r in fe_rows)]
            hist = {"step": [r.get("step", i) for i, r in enumerate(fe_rows)],
                    **{k: [r[k] for r in fe_rows] for k in (*fe_keys, *mc_fe_keys)}}
            # Scale the coupling terms by the static config lambda_beta scalar.
            lam = getattr(cfg, "lambda_beta", 1.0)
            gam = getattr(cfg, "lambda_gamma", 0.0)
            iae = getattr(cfg, "include_attention_entropy", True)
            fig = figs.plot_free_energy_decomposition(
                hist, lambda_beta=lam, lambda_gamma=gam, include_attention_entropy=iae,
                path=str(run / "free_energy_decomposition.png"))
            figs.plt.close(fig)
            fig = figs.plot_free_energy_codescent(
                hist, lambda_beta=lam, lambda_gamma=gam, include_attention_entropy=iae,
                path=str(run / "free_energy_codescent.png"))
            figs.plt.close(fig)
        # Model-channel free-energy blocks (s-channel): the hyper-prior KL(s||r), the gamma
        # model-coupling, and its meta-entropy over training. Present only when the model channel
        # is active (diagnostics logs these columns, gated on STATIC config), so the figure appears
        # exactly on the runs that have a model channel. RAW per-token blocks, a SEPARATE figure
        # since the model channel is a distinct hierarchical tier (h -> s -> p -> q).
        mc_keys = ("hyper_prior", "gamma_coupling", "gamma_meta_entropy")
        mc_present = [k for k in mc_keys
                      if any(k in r and math.isfinite(r[k]) for r in artifacts.history)]
        if mc_present:
            mc_rows = [r for r in artifacts.history
                       if all(k in r and math.isfinite(r[k]) for k in mc_present)]
            if mc_rows:
                hist_mc = {"step": [r.get("step", i) for i, r in enumerate(mc_rows)],
                           **{k: [r[k] for r in mc_rows] for k in mc_present}}
                fig = figs.plot_model_channel_terms(hist_mc, path=str(run / "model_channel_terms.png"))
                figs.plt.close(fig)
        # Geometry / SPD / Fisher health dashboard (history-only): the gauge, belief-spectrum, guard,
        # and numerical-safety scalars diagnostics() logs to metrics.csv but that no standard figure
        # surfaced. The panels self-gate, so a run missing a column simply drops that panel.
        hist_geom = _hist_subset((
            "holonomy_wilson", "cocycle_residual", "holonomy_deviation",
            "gauge_invariant_spread", "gauge_head_logdet_spread", "phi_norm_mean", "phi_norm_std",
            "belief_cond_p95", "belief_cond_max", "eff_rank_p5", "eff_rank_median", "eff_rank_p95",
            "fisher_trace_mean", "guard_sigma_floor_frac", "guard_sigma_ceil_frac",
            "guard_energy_klmax_frac", "guard_selfdiv_klmax_frac", "nonfinite_frac", "renyi_band_frac",
            "attn_entropy_min", "attn_entropy_collapsed_heads"))
        if hist_geom:
            fig = figs.plot_geometry_health(hist_geom, path=str(run / "geometry_health.png"))
            figs.plt.close(fig)
        # E-step inference-quality dashboard: the inner-loop F-drop, the nondecreasing fraction, and the
        # last-iter belief residuals -- the E-step evidence the single estep_f_drop curve does not show.
        hist_estep = _hist_subset((
            "estep_f_drop", "estep_f_nondecreasing_frac",
            "estep_r_mu_last", "estep_r_sigma_last", "estep_r_phi_last"))
        if hist_estep:
            fig = figs.plot_estep_quality(hist_estep, path=str(run / "estep_quality.png"))
            figs.plt.close(fig)
        # Held-out validation-sanity dashboard: the per-eval probes (_val_diagnostics) that were CSV-only
        # -- generalization gap, positional loss, causal/attention sanity, and the held-out geometry.
        hist_val = _hist_subset((
            "generalization_gap", "pos_loss_first_q", "pos_loss_last_q", "pos_loss_ratio",
            "val_future_leakage", "val_row_sum_error", "val_pos_content_r2",
            "val_prev_token_mass", "val_period_match_mass", "val_head_redundancy_js",
            "val_holonomy_wilson", "val_cocycle_residual", "val_gauge_invariant_spread",
            "val_belief_cond_p95", "val_fisher_trace_mean", "val_guard_sigma_floor_frac",
            "val_guard_sigma_ceil_frac", "val_guard_energy_klmax_frac",
            "val_phi_norm_mean", "val_phi_norm_std"))
        if hist_val:
            fig = figs.plot_validation_sanity(hist_val, path=str(run / "validation_sanity.png"))
            figs.plt.close(fig)
        # Optimizer information-geometry dashboard: natural-gradient alignment, pullback conditioning,
        # per-role weight norms, and the synthesized update-to-weight ratio. Present only on a gauge
        # natural-grad run (cos_nat_phi / pullback) and when step_metrics captured the norms.
        hist_opt = _hist_subset((
            "cos_nat_phi", "pullback_cond_median", "pullback_cond_max",
            "weight_norm_mu", "weight_norm_sigma", "weight_norm_phi",
            "grad_norm_mu", "grad_norm_sigma", "grad_norm_phi"))
        if hist_opt:
            fig = figs.plot_optimizer_geometry(hist_opt, path=str(run / "optimizer_geometry.png"))
            figs.plt.close(fig)
    except Exception as exc:                                    # never let a plot kill a finished run
        logger.warning("figure generation failed (%s); numeric results are still saved", exc)
