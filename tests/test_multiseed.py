r"""Tests for the EXP-1 across-seed aggregator (multiseed_analysis), added 2026-06-21."""
import json
import math

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
