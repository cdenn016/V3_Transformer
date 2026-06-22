r"""Tests for the 2026-06-22 EXP-5 build-out (structural non-Neal-Hinton EM, C2):

  * the F-vs-CE decorrelation figure ``plot_f_ce_decorrelation`` renders.
  * ``scaling_analysis.aggregate_points`` carries the per-arm converged final E-step F/token
    (``f_mean``) and test BPC (``bpc_mean``) harvested from the n_e_steps cells -- the inputs the
    decorrelation/estep-capacity figures and the headline Pearsons read.

(The load-bearing persistence of ``estep_final_f_per_token`` into test_results.json / summary.json
is pinned end-to-end by tests/test_run_diagnostics_2026_06_13.py::
test_finalize_writes_tier3_research_and_provenance.)

Device-agnostic (CPU). Figures use the Agg backend.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scaling_analysis
from vfe3.viz import figures as figs


def test_f_ce_decorrelation_figure_renders():
    arms = [{"n_e_steps": t, "final_f": 60.0 - 6.0 * t, "ce": 4.0 - 0.01 * t}
            for t in (1, 2, 4, 8)]                                   # F falls steeply, CE nearly flat
    fig = figs.plot_f_ce_decorrelation(arms)
    assert fig is not None
    plt.close(fig)


def _row(label, seed, t, f, ce, bpc):
    return {"route": "inference", "scale_knob": "n_e_steps", "label": label, "seed": seed,
            "n_params": 1000, "n_gen": 6, "tokens_seen": 1000, "est_flops_6ND": 1.0,
            "est_flops_analytic": 1.0, "n_e_steps": t, "n_layers": 1,
            "test_ce": ce, "test_bpc": bpc, "estep_final_f_per_token": f}


def test_aggregate_points_carries_final_f_and_bpc():
    rows = [_row("T2", 0, 2, 40.0, 3.0, 4.3),
            _row("T2", 1, 2, 42.0, 3.2, 4.5)]                       # two seeds of one n_e_steps cell
    pts = scaling_analysis.aggregate_points(rows)
    assert len(pts) == 1
    p = pts[0]
    assert abs(p["f_mean"] - 41.0) < 1e-9                           # mean of 40, 42
    assert abs(p["bpc_mean"] - 4.4) < 1e-9                          # mean of 4.3, 4.5
    assert p["n_e_steps"] == 2


def test_aggregate_points_final_f_nan_when_absent():
    rows = [{"route": "inference", "scale_knob": "n_e_steps", "label": "T1", "seed": 0,
             "n_params": 1000, "n_gen": 6, "tokens_seen": 1000, "est_flops_6ND": 1.0,
             "est_flops_analytic": 1.0, "n_e_steps": 1, "n_layers": 1, "test_ce": 3.0}]
    pts = scaling_analysis.aggregate_points(rows)
    assert len(pts) == 1
    import math
    assert math.isnan(pts[0]["f_mean"]) and math.isnan(pts[0]["bpc_mean"])
