r"""Tests for the 2026-06-22 runnable-diagnostics tail (E1/E2/E3/A4 sweeps + A4 figure):

  * the five lower-priority sweeps (amp_dtype, spd_retract_mode, sigma_max, e_mu_q_trust, regime_ii)
    validate and build every cell;
  * the A4 holonomy-vs-||connection|| trainability figure renders and its driver reads the regime_ii
    cell's metrics.csv (correctly excluding the flat cell, which logs no connection_w_norm).

Device-agnostic (CPU). Figures use the Agg backend.
"""
import csv as _csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
import torch

import ablation
from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel
from vfe3.viz import figures as figs

DEVICE = torch.device(os.environ.get("VFE3_TEST_DEVICE", "cpu"))


@pytest.mark.parametrize("sweep", ["amp_dtype", "spd_retract_mode", "sigma_max",
                                   "e_mu_q_trust", "regime_ii"])
def test_runnable_cluster_sweeps_build(sweep):
    ablation.validate_sweeps([sweep])
    runs = ablation.make_run_overrides(sweep)
    assert runs
    for _label, ov in runs:
        cfg_dict = ablation._cell_cfg_dict({**ov, "vocab_size": 48, "max_seq_len": 16}, seed=0, max_steps=1)
        assert VFEModel(VFE3Config(**cfg_dict)) is not None


def test_holonomy_trainability_figure_renders():
    arms = [{"label": "regime_ii", "step": [10, 20, 30],
             "connection_norm": [0.01, 0.1, 0.3], "holonomy": [1e-3, 5e-3, 2e-2]}]
    fig = figs.plot_holonomy_trainability(arms)
    assert fig is not None
    plt.close(fig)


def test_plot_holonomy_trainability_driver(tmp_path):
    sweep = tmp_path / "regime_ii"; figdir = tmp_path / "figures"
    d = sweep / "regime_ii"; d.mkdir(parents=True)
    with open(d / "metrics.csv", "w", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=["step", "connection_w_norm", "holonomy_deviation"])
        w.writeheader()
        for st, c, h in [(10, 0.01, 1e-3), (20, 0.1, 5e-3), (30, 0.3, 2e-2)]:
            w.writerow({"step": st, "connection_w_norm": c, "holonomy_deviation": h})
    d2 = sweep / "flat"; d2.mkdir(parents=True)                     # flat cell: no connection_w_norm column
    with open(d2 / "metrics.csv", "w", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=["step", "holonomy_deviation"])
        w.writeheader()
        w.writerow({"step": 10, "holonomy_deviation": 0.0})
    ablation._plot_holonomy_trainability(sweep, figdir)            # must not crash on the flat cell
    assert (figdir / "regime_ii_holonomy_trainability.png").exists()
