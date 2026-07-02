r"""Tests for the 2026-07-01 round-3 viz fixes (vfe3/viz/figures.py):

  * the dead category-scatter helpers superseded by the clustering path are gone;
  * register_figure fails closed on duplicate names and replaces under override=True;
  * geometry_health gains the learned-connection trainability panel and still self-gates
    on column presence when the connection/mixer columns are absent.

Device-agnostic (CPU). Figures use the Agg backend.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from vfe3.viz import figures as figs


def _hist(keys, n=30):
    r"""Synthetic {step, key: [...]} history: positive decreasing series (log-axis safe)."""
    h = {"step": list(range(n))}
    base = np.linspace(1.0, 0.1, n) + 0.05
    for k in keys:
        h[k] = base.tolist()
    return h


def test_dead_category_helpers_removed():
    # punch item 10: superseded by the clustering path (_cluster_embedding / _cluster_lift_labels).
    assert not hasattr(figs, "_scatter_by_category")
    assert not hasattr(figs, "_annotate_frequent_tokens")


def test_register_figure_duplicate_fails_closed_and_override_replaces():
    orig = figs.get_figure("geometry_health")
    try:
        with pytest.raises(KeyError, match="already registered"):
            @figs.register_figure("geometry_health")
            def _dup(*a, **k):
                pass
        assert figs.get_figure("geometry_health") is orig      # fail-closed: original untouched

        @figs.register_figure("geometry_health", override=True)
        def _replacement(*a, **k):
            pass
        assert figs.get_figure("geometry_health") is _replacement
    finally:
        figs.register_figure("geometry_health", override=True)(orig)
    assert figs.get_figure("geometry_health") is orig


def test_geometry_health_renders_learned_connection_panel():
    h = _hist(("connection_w_norm", "connection_m_norm", "connection_l_norm",
               "connection_l_offdiag_norm", "head_mixer_drift"))
    fig = figs.plot_geometry_health(h)
    assert fig is not None
    assert "Learned-connection trainability" in [ax.get_title() for ax in fig.axes]
    plt.close(fig)


def test_geometry_health_self_gates_without_connection_columns():
    fig = figs.plot_geometry_health(_hist(("holonomy_wilson",)))
    assert fig is not None
    assert "Learned-connection trainability" not in [ax.get_title() for ax in fig.axes]
    plt.close(fig)
