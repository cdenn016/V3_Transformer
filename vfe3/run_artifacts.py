r"""Run artifacts for VFE_3.0 training: the persistence + reporting layer.

A training run produces a self-contained directory::

    run_dir/
      config.json        full VFE3Config + run metadata (n_params, dataset, device, timestamp)
      metrics.csv        one row per periodic eval (step, train_loss, lr, val_ce/ppl/bpc, diagnostics)
      checkpoints/
        step_<N>.pt      resumable {step, model_state, optimizer_state, config}
      best_model.pt      model.state_dict() at the lowest validation PPL seen so far
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
import json
import logging
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch

from vfe3.config import VFE3Config


class RunArtifacts:
    r"""Owns a run directory and the incremental writes (CSV rows, checkpoints, best model)."""

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
        r"""Write ``obj`` as pretty JSON to ``run_dir/name`` (non-serializable -> str)."""
        path = self.run_dir / name
        path.write_text(json.dumps(obj, indent=2, default=str))
        return path

    def log_metrics(self, row: Dict[str, float]) -> None:
        r"""Append one metrics row to ``metrics.csv`` (header written on the first call).

        The column set is fixed by the first row; later rows must share those keys so the CSV
        stays rectangular (the training loop emits a homogeneous row each periodic eval)."""
        self.history.append(dict(row))
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
            with open(self.csv_path, "w", newline="") as fh:
                csv.DictWriter(fh, fieldnames=self._fieldnames).writeheader()
        with open(self.csv_path, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=self._fieldnames).writerow(row)

    def maybe_save_best(self, step: int, model: torch.nn.Module, val_ppl: float) -> bool:
        r"""Save ``model.state_dict()`` to ``best_model.pt`` iff ``val_ppl`` is a new minimum."""
        if val_ppl < self.best_val_ppl:
            self.best_val_ppl = float(val_ppl)
            self.best_step = int(step)
            torch.save(model.state_dict(), self.best_path)
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

    def save_checkpoint(
        self,
        step:      int,
        model:     torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        cfg:       VFE3Config,

        *,
        scaler:    Optional['torch.amp.GradScaler'] = None,
    ) -> Path:
        r"""Write a resumable ``checkpoints/step_<N>.pt`` (model + optimizer + RNG + config + step).

        ``load_checkpoint`` reads this back to continue training: ``model_state`` and
        ``optimizer_state`` restore the weights and AdamW momentum, ``rng_state`` restores the
        CPU (and CUDA) generators for reproducible continuation, and ``step`` is the number of
        completed M-steps so the resumed run rebuilds the cosine ``LambdaLR`` at the right point.
        ``scaler`` (audit 2026-06-09 IE3): an ENABLED fp16 GradScaler's state (current scale +
        growth counters) is bundled so a resumed fp16 run does not restart at the init scale
        65536 and re-converge by skipped steps; a disabled/None scaler stores None.
        """
        path = self.ckpt_dir / f"step_{step}.pt"
        rng_state = {
            "cpu":  torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        torch.save({
            "step":            int(step),
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "rng_state":       rng_state,
            "config":          asdict(cfg),
            "scaler_state":    (scaler.state_dict()
                                if scaler is not None and scaler.is_enabled() else None),
        }, path)
        return path


def load_checkpoint(
    path:      'str | Path',
    model:     torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,

    *,
    map_location: 'Optional[str | torch.device]'        = None,
    restore_rng:  bool                                   = True,
    scaler:       Optional['torch.amp.GradScaler']       = None,
    cfg:          Optional[VFE3Config]                   = None,
) -> int:
    r"""Restore a ``save_checkpoint`` bundle into ``model`` (and optionally ``optimizer``); return the saved step.

    This is the LOAD half of the resumable checkpoint. It always restores the model weights;
    it restores the AdamW optimizer state (momentum buffers + per-parameter step counts) when an
    ``optimizer`` is supplied, and the CPU/CUDA RNG when ``restore_rng`` is set and the bundle
    carries it (checkpoints written before RNG was persisted simply skip that step). The returned
    integer is the number of completed M-steps; ``train(resume_from=...)`` uses it to rebuild the
    cosine ``LambdaLR`` at the saved step and to start the loop from there.

    ``scaler`` (audit 2026-06-09 IE3): when given AND the bundle carries a saved scaler state,
    the fp16 GradScaler's scale/growth counters are restored (bundles written before the scaler
    was persisted, or written from a non-fp16 run, simply skip the step). ``cfg`` (audit IE4):
    when given, the CURRENT config is compared against the bundle's saved config and any
    differing fields are warned about -- strict ``load_state_dict`` already catches
    shape-changing divergence, but shape-preserving semantic drift (LR schedule, n_e_steps,
    e_*_lr, ...) would otherwise pass silently.

    ``weights_only=False`` is required because the bundle carries the optimizer state and the RNG
    tensors (not a pure ``state_dict``); the file is a trusted run artifact this process wrote
    (matching ``test_run_artifacts.py::test_save_checkpoint_is_loadable``).
    """
    if map_location is None:
        map_location = next(model.parameters()).device
    ckpt = torch.load(Path(path), map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])
    if cfg is not None and ckpt.get("config") is not None:
        saved = ckpt["config"]
        current = asdict(cfg)
        # resume_from is run bookkeeping (the resumed run necessarily sets it; the saved run
        # rarely did) -- not semantic drift.
        drift = sorted(k for k in (saved.keys() | current.keys())
                       if k != "resume_from" and saved.get(k) != current.get(k))
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
    return int(ckpt["step"])


def finalize_run(
    model:       torch.nn.Module,
    artifacts:   RunArtifacts,
    cfg:         VFE3Config,

    *,
    test_loader:     Optional[Iterable] = None,
    losses:          Optional[List[float]] = None,
    tokens_per_char: float = 1.0,           # test BPC char-correction (1.0 = bits/token)
    device:          Optional[torch.device] = None,
    wall_time:       Optional[float] = None,
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
        # best_model.pt is a pure state_dict (torch.save(model.state_dict(), ...)), so weights_only=True
        # loads it identically while refusing arbitrary pickle execution on a tampered checkpoint
        # (matches the datasets.py precedent).
        model.load_state_dict(torch.load(artifacts.best_path, map_location=device, weights_only=True))
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
    artifacts.save_json("test_results.json", results)

    artifacts.save_json("summary.json", {
        "n_steps":      cfg.max_steps,
        "n_params":     int(sum(p.numel() for p in model.parameters())),
        "best_val_ppl": best_val_ppl,
        "best_step":    artifacts.best_step,
        "test_ppl":     results.get("test_ppl"),
        "test_ce":      results.get("test_ce"),
        "test_bpc":     results.get("test_bpc"),
        "final_train_loss": (losses[-1] if losses else None),
        "wall_time_s":  wall_time,
        "use_prior_bank":  cfg.use_prior_bank,
        "use_head_mixer":  cfg.use_head_mixer,
    })

    _save_figures(artifacts, losses, logger)
    # Single-run publication figure set (model-replay), auto-run at the end of training unless
    # cfg.generate_figures is False. Best-effort and off the hot path -- the runners are expensive
    # (UMAP, E-step replay, holonomy sampling, a belief bank over many sequences), so a failure is
    # logged and never disturbs the saved numeric results. Drives the BEST-val model reloaded above.
    if getattr(cfg, "generate_figures", True):
        try:
            from vfe3.viz.report import generate_figures
            generate_figures(artifacts.run_dir, model=model, loader=test_loader,
                             device=device, logger=logger)
        except Exception as exc:
            logger.warning("publication figure generation failed (%s); numeric results are saved", exc)
    return results


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
                title="Holonomy deviation (curvature proxy)", color=figs._CB[2],
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
        # Learnable belief-coupling weight: present in history only on a learnable_lambda_beta run.
        lx, ly = _aligned("lambda_beta")
        if ly:
            fig = figs.plot_trajectory(
                ly, lx, ylabel=r"$\lambda_\beta = e^{\log\lambda_\beta}$",
                title="Learned belief-coupling weight", color=figs._CB[4],
                smooth=max(5, len(ly) // 60), annotate_final=True,
                path=str(run / "lambda_beta.png"))
            figs.plt.close(fig)
        # Free-energy figures: the per-token budget DECOMPOSITION (snapshot + early/mid/late evolution)
        # and, as a SEPARATE figure, the F-vs-CE CO-DESCENT over training. Both need every plotted term
        # finite, so rows before the first eval (NaN val_*) are dropped.
        fe_keys = ("self_coupling", "belief_coupling", "attention_entropy", "val_ce")
        fe_rows = [r for r in artifacts.history
                   if all(k in r and math.isfinite(r[k]) for k in fe_keys)]
        if fe_rows:
            cfg = getattr(artifacts, "cfg", None)
            hist = {"step": [r.get("step", i) for i, r in enumerate(fe_rows)],
                    **{k: [r[k] for r in fe_rows] for k in fe_keys}}
            # Scale the coupling terms by the LEARNED lambda_beta trajectory when every row carries it
            # (a learnable_lambda_beta run); else the static config scalar.
            lam = ([r["lambda_beta"] for r in fe_rows] if all("lambda_beta" in r for r in fe_rows)
                   else getattr(cfg, "lambda_beta", 1.0))
            fig = figs.plot_free_energy_decomposition(
                hist, lambda_beta=lam, path=str(run / "free_energy_decomposition.png"))
            figs.plt.close(fig)
            fig = figs.plot_free_energy_codescent(
                hist, lambda_beta=lam, path=str(run / "free_energy_codescent.png"))
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
    except Exception as exc:                                    # never let a plot kill a finished run
        logger.warning("figure generation failed (%s); numeric results are still saved", exc)
