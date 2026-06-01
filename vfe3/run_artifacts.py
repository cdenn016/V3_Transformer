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
      val_ppl.png        validation perplexity trajectory
      free_energy_terms.png   the per-term free-energy decomposition at the last eval

``RunArtifacts`` is OPT-IN: ``train`` only touches it when an instance is passed, so the silent
path (``artifacts=None``) writes nothing and is unchanged. ``finalize_run`` reloads the best-val
checkpoint, scores the held-out test split, and writes the summary + figures. Figure generation
is best-effort (a plotting/dependency error is logged, never fatal) so the numeric results
survive a viz problem.
"""

import csv
import json
import logging
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
        dataset:   str               = "",
        device:    'str | torch.device' = "cpu",
        timestamp: Optional[str]      = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.run_dir / "metrics.csv"
        self.best_path = self.run_dir / "best_model.pt"

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

    def save_checkpoint(
        self,
        step:      int,
        model:     torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        cfg:       VFE3Config,
    ) -> Path:
        r"""Write a resumable ``checkpoints/step_<N>.pt`` (model + optimizer + config + step)."""
        path = self.ckpt_dir / f"step_{step}.pt"
        torch.save({
            "step":            int(step),
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config":          asdict(cfg),
        }, path)
        return path


def finalize_run(
    model:       torch.nn.Module,
    artifacts:   RunArtifacts,
    cfg:         VFE3Config,

    *,
    test_loader: Optional[Iterable] = None,
    losses:      Optional[List[float]] = None,
    device:      Optional[torch.device] = None,
    wall_time:   Optional[float] = None,
    logger:      Optional[logging.Logger] = None,
) -> Dict[str, float]:
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
        model.load_state_dict(torch.load(artifacts.best_path, map_location=device))
        reloaded_best = True
        logger.info("Reloaded best-val checkpoint (step %s, val PPL %.3f) for test eval",
                    artifacts.best_step, artifacts.best_val_ppl)

    results: Dict[str, float] = {}
    if test_loader is not None:
        m = evaluate(model, test_loader, device=device)
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
        if losses:
            fig = figs.plot_trajectory(losses, ylabel="train CE (nats)", title="Training loss",
                                       path=str(artifacts.run_dir / "loss_curve.png"))
            figs.plt.close(fig)
        val_ppl = [r["val_ppl"] for r in artifacts.history if "val_ppl" in r]
        if val_ppl:
            fig = figs.plot_trajectory(val_ppl, ylabel="val PPL", title="Validation perplexity",
                                       path=str(artifacts.run_dir / "val_ppl.png"))
            figs.plt.close(fig)
        if artifacts.history and "self_coupling" in artifacts.history[-1]:
            _save_free_energy_bar(artifacts, figs)
    except Exception as exc:                                    # never let a plot kill a finished run
        logger.warning("figure generation failed (%s); numeric results are still saved", exc)


def _save_free_energy_bar(artifacts: RunArtifacts, figs) -> None:
    r"""Bar of the per-term free-energy decomposition at the last eval."""
    last = artifacts.history[-1]
    terms = {k: last[k] for k in ("self_coupling", "belief_coupling", "attention_entropy")
             if k in last}
    fig, ax = figs.plt.subplots(figsize=(4.5, 3.2))
    ax.bar(range(len(terms)), list(terms.values()), color="#4C72B0")
    ax.set_xticks(range(len(terms)))
    ax.set_xticklabels(list(terms.keys()), rotation=20, ha="right")
    ax.set(title="Free-energy decomposition (last eval)", ylabel="nats")
    fig.savefig(str(artifacts.run_dir / "free_energy_terms.png"))
    figs.plt.close(fig)
