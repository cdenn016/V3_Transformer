r"""Tests for the across-seed aggregator (multiseed_analysis).

Original EXP-1 scalar aggregator added 2026-06-21; full cross-seed digest (curves, per-layer,
research scalars, figure set) added 2026-06-22.
"""
import json
import math

import numpy as np

import multiseed_analysis as ms


def _write_run(d, ppl, seed):
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps({"test_ppl": ppl}))
    (d / "config.json").write_text(json.dumps({"seed": seed}))


def test_aggregate_seed_metric_mean_sd_and_seeds(tmp_path):
    vals = [10.0, 12.0, 11.0, 13.0, 9.0]                     # mean 11, ddof=1 var 2.5
    for i, v in enumerate(vals):
        _write_run(tmp_path / f"seed{i}", v, seed=i)
    out = ms.aggregate_seed_metric(tmp_path)
    assert out["n"] == 5
    assert abs(out["mean"] - 11.0) < 1e-9
    assert abs(out["sd"] - math.sqrt(2.5)) < 1e-9
    assert abs(out["two_sd"] - 2.0 * math.sqrt(2.5)) < 1e-9
    assert abs(out["cv"] - math.sqrt(2.5) / 11.0) < 1e-9
    assert sorted(out["seeds"]) == [0, 1, 2, 3, 4]


def test_aggregate_skips_nonfinite_and_unreadable(tmp_path):
    _write_run(tmp_path / "a", 10.0, seed=1)
    _write_run(tmp_path / "b", float("inf"), seed=2)         # inf serializes to "Infinity" -> skipped
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "summary.json").write_text("{ not json")
    out = ms.aggregate_seed_metric(tmp_path)
    assert out["n"] == 1 and abs(out["mean"] - 10.0) < 1e-9
    assert math.isnan(out["sd"])                            # n<2 -> SD undefined


def test_aggregate_empty_root(tmp_path):
    out = ms.aggregate_seed_metric(tmp_path)
    assert out["n"] == 0 and math.isnan(out["mean"])


def test_flag_noise_dominated():
    cells = {"a": 10.0, "b": 10.5, "c": 15.0, "d": None}
    flagged = ms.flag_noise_dominated(cells, sd=1.0, k=2.0)   # threshold 2.0 above best (10.0)
    assert set(flagged) == {"a", "b"}                         # c is 5 > 2 away; d skipped
    assert ms.flag_noise_dominated(cells, sd=float("nan")) == []


# =============================================================================
# Full cross-seed digest (2026-06-22): run-root resolution, seed source, curves,
# per-layer, research scalars, and the new figure functions.
# =============================================================================
def _write_full_run(d, *, ppl, seed, metrics=None, per_layer=None, research=None):
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps({"test_ppl": ppl}))
    (d / "config.json").write_text(json.dumps({"seed": None}))           # real seed lives in provenance
    (d / "provenance.json").write_text(json.dumps({"seed": seed}))
    if metrics is not None:
        (d / "metrics.csv").write_text(metrics)
    if per_layer is not None:
        (d / "metrics_per_layer.csv").write_text(per_layer)
    if research is not None:
        (d / "research.json").write_text(json.dumps(research))


def test_resolve_run_root_bare_name_under_vfe3_runs(tmp_path, monkeypatch):
    (tmp_path / "vfe3_runs" / "K=20_GL(10)").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    resolved = ms._resolve_run_root("K=20_GL(10)")
    assert resolved.resolve() == (tmp_path / "vfe3_runs" / "K=20_GL(10)").resolve()


def test_resolve_run_root_prefers_existing_literal(tmp_path, monkeypatch):
    (tmp_path / "explicit").mkdir()
    monkeypatch.chdir(tmp_path)
    assert ms._resolve_run_root("explicit").resolve() == (tmp_path / "explicit").resolve()


def test_seed_for_prefers_provenance_over_config(tmp_path):
    d = tmp_path / "run_s3"
    d.mkdir()
    (d / "provenance.json").write_text(json.dumps({"seed": 99}))
    (d / "config.json").write_text(json.dumps({"seed": None}))
    assert ms._seed_for(d) == 99


def test_seed_for_falls_back_to_config_then_dirname(tmp_path):
    cfg = tmp_path / "run_a"
    cfg.mkdir()
    (cfg / "config.json").write_text(json.dumps({"seed": 5}))
    assert ms._seed_for(cfg) == 5
    name = tmp_path / "wikitext_K20_s42"
    name.mkdir()
    assert ms._seed_for(name) == 42


def test_aggregate_seed_metric_uses_provenance_seed(tmp_path):
    for v, s in [(10.0, 23), (12.0, 54)]:
        _write_full_run(tmp_path / f"run_s{s}", ppl=v, seed=s)
    out = ms.aggregate_seed_metric(tmp_path)
    assert out["n"] == 2 and sorted(out["seeds"]) == [23, 54]


def test_aggregate_scalar_reads_dotted_research_key(tmp_path):
    _write_full_run(tmp_path / "a", ppl=1.0, seed=1, research={"freq_strata_ce": {"rare": 7.0}})
    _write_full_run(tmp_path / "b", ppl=1.0, seed=2, research={"freq_strata_ce": {"rare": 9.0}})
    out = ms.aggregate_scalar(tmp_path, "freq_strata_ce.rare")
    assert out["n"] == 2 and abs(out["mean"] - 8.0) < 1e-9


def test_aggregate_seed_curves_aligns_and_handles_nan(tmp_path):
    _write_full_run(tmp_path / "a", ppl=1.0, seed=1, metrics="step,x,y\n100,1,\n200,3,9\n")
    _write_full_run(tmp_path / "b", ppl=1.0, seed=2, metrics="step,x,y\n100,3,7\n200,5,11\n")
    curves = ms.aggregate_seed_curves(tmp_path, columns=["x", "y"])
    np.testing.assert_allclose(curves["x"]["steps"], [100, 200])
    np.testing.assert_allclose(curves["x"]["mean"], [2.0, 4.0])
    np.testing.assert_allclose(curves["x"]["sd"][0], math.sqrt(2.0))      # ddof=1 over {1,3}
    np.testing.assert_allclose(curves["x"]["n"], [2, 2])
    # y is empty in seed a at step 100 -> NaN-aware mean uses seed b only, n=1, sd undefined.
    np.testing.assert_allclose(curves["y"]["mean"], [7.0, 10.0])
    np.testing.assert_allclose(curves["y"]["n"], [1, 2])
    assert math.isnan(curves["y"]["sd"][0])


def test_aggregate_per_layer(tmp_path):
    _write_full_run(tmp_path / "a", ppl=1.0, seed=1, per_layer="layer,self_coupling\n0,10\n")
    _write_full_run(tmp_path / "b", ppl=1.0, seed=2, per_layer="layer,self_coupling\n0,20\n")
    out = ms.aggregate_per_layer(tmp_path)
    assert abs(out[0]["self_coupling"]["mean"] - 15.0) < 1e-9
    assert abs(out[0]["self_coupling"]["sd"] - math.sqrt(50.0)) < 1e-9   # ddof=1 over {10,20}


# --- new figure functions (smoke tests, test_viz style) ---
def _saved_nonempty(path):
    return path.exists() and path.stat().st_size > 0


def test_plot_curve_band_writes(tmp_path):
    from vfe3.viz import figures as figs
    figs.set_publication_style()
    steps = np.array([1.0, 2.0, 3.0])
    fig = figs.plot_curve_band(steps, np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]),
                               label="x", ylabel="v", path=str(tmp_path / "band.png"))
    figs.plt.close(fig)
    assert _saved_nonempty(tmp_path / "band.png")


def test_plot_curve_band_grid_writes(tmp_path):
    from vfe3.viz import figures as figs
    figs.set_publication_style()
    curves = [
        {"steps": np.array([1.0, 2.0]), "mean": np.array([1.0, 2.0]), "sd": np.array([0.1, 0.1]),
         "title": "a", "logy": False},
        {"steps": np.array([1.0, 2.0]), "mean": np.array([3.0, 4.0]), "sd": np.array([0.2, 0.2]),
         "title": "b", "logy": True},
    ]
    fig = figs.plot_curve_band_grid(curves, path=str(tmp_path / "grid.png"))
    figs.plt.close(fig)
    assert _saved_nonempty(tmp_path / "grid.png")


def test_plot_scalar_cv_summary_writes(tmp_path):
    from vfe3.viz import figures as figs
    figs.set_publication_style()
    aggs = {
        "test_ppl": {"mean": 137.0, "sd": 0.9, "cv": 0.0065, "values": [136.0, 137.0, 138.0]},
        "test_ce":  {"mean": 4.9, "sd": 0.007, "cv": 0.0014, "values": [4.89, 4.9, 4.91]},
    }
    fig = figs.plot_scalar_cv_summary(aggs, path=str(tmp_path / "cv.png"))
    figs.plt.close(fig)
    assert _saved_nonempty(tmp_path / "cv.png")


def test_plot_per_layer_band_writes(tmp_path):
    from vfe3.viz import figures as figs
    figs.set_publication_style()
    per_layer = {0: {"self_coupling": {"mean": 15.0, "sd": 5.0, "n": 2, "values": [10.0, 20.0]}},
                 1: {"self_coupling": {"mean": 30.0, "sd": 4.0, "n": 2, "values": [28.0, 32.0]}}}
    fig = figs.plot_per_layer_band(per_layer, "self_coupling", path=str(tmp_path / "layer.png"))
    figs.plt.close(fig)
    assert _saved_nonempty(tmp_path / "layer.png")
