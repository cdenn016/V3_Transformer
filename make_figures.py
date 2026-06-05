r"""Click-to-run: generate the single-run publication figures for a trained run.

Edit ``CONFIG`` below and run. Points at a run directory written by ``train_vfe3.py``
(``config.json`` + ``best_model.pt``), rebuilds the trained model, drives the
:mod:`vfe3.viz.extract` runners + :mod:`vfe3.metrics` measurements, and writes the figure set
to ``<run_dir>/figures/``. With ``run_dir=None`` the most recent run under ``RUN_ROOT`` is used.

This is a SEPARATE, opt-in step from training: the figure runners are expensive (UMAP, E-step
replay, holonomy sampling, a belief bank over many sequences), so they are not produced on the
training hot path. The free-energy descent + trajectory figures are still written at training
``finalize_run``; this driver adds the model-replay figures that were never wired up.
"""

import logging
from pathlib import Path

import torch

from vfe3.viz.report import generate_figures

RUN_ROOT = "vfe3_runs"

CONFIG = {
    "run_dir":       None,                                  # None -> newest run under RUN_ROOT
    "device":        "cuda" if torch.cuda.is_available() else "cpu",
    "split":         "validation",                          # split the representative batch is drawn from
    "max_sequences": 256,                                   # belief-bank size for the UMAP triptych
    "n_e_steps":     None,                                  # E-step trace length (None -> trained cfg.n_e_steps)
}


def _newest_run(root: str) -> Path:
    r"""The most recently modified run directory under ``root`` (a config.json marks a run)."""
    runs = [p for p in Path(root).glob("*") if (p / "config.json").exists()]
    if not runs:
        raise FileNotFoundError(f"no runs (with config.json) under {root!r}; train one first")
    return max(runs, key=lambda p: p.stat().st_mtime)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_dir = Path(CONFIG["run_dir"]) if CONFIG["run_dir"] else _newest_run(RUN_ROOT)
    print(f"\nVFE_3.0 figure generation\n  run_dir: {run_dir}\n  device:  {CONFIG['device']}")
    paths = generate_figures(
        run_dir,
        device=torch.device(CONFIG["device"]),
        split=CONFIG["split"],
        max_sequences=CONFIG["max_sequences"],
        n_e_steps=CONFIG["n_e_steps"],
    )
    print(f"\nwrote {len(paths)} figures to {run_dir / 'figures'}")


if __name__ == "__main__":
    main()
