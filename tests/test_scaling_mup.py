r"""Tests for the muP width-stability scaling route (F1/EXP-6) added 2026-06-21.

route_grow_k_mup emits a matched fixed-LR vs muP pair per width, each with the per-cell kl_max=8*K
confound fix, and every cell builds a valid VFE3Config/VFEModel."""
import json
from dataclasses import asdict

import pytest
import torch

import scaling
import scaling_analysis
from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel


_SOURCE_IDENTITIES = {
    split: {"split": split, "sha256": split * 8}
    for split in ("train", "validation", "test")
}


def _patch_source_identity(monkeypatch):
    monkeypatch.setattr(
        scaling,
        "cache_source_identity",
        lambda dataset, split: dict(_SOURCE_IDENTITIES[split]),
    )


def test_grow_k_mup_registered():
    assert "grow_K_mup" in scaling.ROUTES


def test_grow_k_mup_kl_max_and_lr_scaling():
    cells = scaling.route_grow_k_mup([20, 40, 80], n_heads=4, anchor_k=20)
    ov = {c["label"]: c["overrides"] for c in cells}

    # per-cell kl_max = 8*K on BOTH arms (the confound fix), every width present as a fixed/mup pair
    for c in cells:
        assert c["overrides"]["kl_max"] == 8 * c["overrides"]["embed_dim"]
    assert set(ov) == {"K20_fixed", "K20_mup", "K40_fixed", "K40_mup", "K80_fixed", "K80_mup"}

    base_eqmu = scaling._baseline_value("e_q_mu_lr")
    base_init = scaling._baseline_value("mu_init_std")

    # anchor K=20: muP factor is 1 -> mup arm LR equals baseline (coincides with fixed)
    assert ov["K20_mup"]["e_q_mu_lr"] == pytest.approx(base_eqmu)
    # K=80: LR ~ anchor/K = 0.25, init ~ sqrt(0.25) = 0.5
    assert ov["K80_mup"]["e_q_mu_lr"] == pytest.approx(base_eqmu * 0.25)
    assert ov["K80_mup"]["mu_init_std"] == pytest.approx(base_init * 0.5)
    # the fixed arm carries no LR/init override (stays at the baseline operating point)
    assert "e_q_mu_lr" not in ov["K80_fixed"] and "mu_init_std" not in ov["K80_fixed"]


def test_grow_k_mup_cells_build():
    for c in scaling.route_grow_k_mup([20, 40], n_heads=4, anchor_k=20):
        d = scaling._cell_cfg_dict({**c["overrides"], "vocab_size": 64}, 0, 1)
        assert VFEModel(VFE3Config(**d)) is not None


def test_cell_is_current_checks_max_tokens(tmp_path, monkeypatch):
    r"""max_tokens (the loader train-token cap) is not a VFE3Config field, so config.json alone
    cannot distinguish a capped smoke cell from a full run: _cell_is_current must also compare
    the max_tokens persisted in scaling_cell.json."""
    run_dir = tmp_path / "cell"
    run_dir.mkdir()
    ds = "wikitext-103"
    cfg = VFE3Config(**scaling._cell_cfg_dict({"vocab_size": 64}, 0, 1))
    (run_dir / "summary.json").write_text("{}", encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps({
        "dataset": ds,
        "config": json.loads(json.dumps(asdict(cfg), default=str)),
    }), encoding="utf-8")
    (run_dir / "scaling_cell.json").write_text(json.dumps({
        "label": "cell", "max_tokens": 1000, "data_sources": _SOURCE_IDENTITIES,
    }), encoding="utf-8")
    (run_dir / "provenance.json").write_text(json.dumps({
        "git_sha": "current-head",
        "git_dirty": False,
        "git_dirty_fingerprint": None,
    }), encoding="utf-8")
    monkeypatch.setattr(scaling, "_current_code_identity", lambda: {
        "git_sha": "current-head",
        "git_dirty": False,
        "git_dirty_fingerprint": None,
    })
    _patch_source_identity(monkeypatch)

    assert scaling._cell_is_current(run_dir, cfg, ds, max_tokens=1000) is True
    assert scaling._cell_is_current(run_dir, cfg, ds, max_tokens=None) is False


def test_cell_is_current_rejects_missing_or_mismatched_code_identity(tmp_path, monkeypatch):
    run_dir = tmp_path / "cell"
    run_dir.mkdir()
    ds = "wikitext-103"
    cfg = VFE3Config(**scaling._cell_cfg_dict({"vocab_size": 64}, 0, 1))
    (run_dir / "summary.json").write_text("{}", encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps({
        "dataset": ds,
        "config": json.loads(json.dumps(asdict(cfg), default=str)),
    }), encoding="utf-8")
    (run_dir / "scaling_cell.json").write_text(json.dumps({
        "label": "cell", "max_tokens": 1000, "data_sources": _SOURCE_IDENTITIES,
    }), encoding="utf-8")
    monkeypatch.setattr(scaling, "_current_code_identity", lambda: {
        "git_sha": "current-head",
        "git_dirty": False,
        "git_dirty_fingerprint": None,
    })
    _patch_source_identity(monkeypatch)

    assert scaling._cell_is_current(run_dir, cfg, ds, max_tokens=1000) is False
    (run_dir / "provenance.json").write_text(json.dumps({
        "git_sha": "other-head",
        "git_dirty": False,
        "git_dirty_fingerprint": None,
    }), encoding="utf-8")
    assert scaling._cell_is_current(run_dir, cfg, ds, max_tokens=1000) is False

    (run_dir / "provenance.json").write_text(json.dumps({
        "git_sha": "current-head",
        "git_dirty": False,
        "git_dirty_fingerprint": None,
    }), encoding="utf-8")
    assert scaling._cell_is_current(run_dir, cfg, ds, max_tokens=1000) is True

    drifted_sources = dict(_SOURCE_IDENTITIES)
    drifted_sources["train"] = {"split": "train", "sha256": "changed"}
    monkeypatch.setattr(
        scaling,
        "cache_source_identity",
        lambda dataset, split: dict(drifted_sources[split]),
    )
    assert scaling._cell_is_current(run_dir, cfg, ds, max_tokens=1000) is False
    _patch_source_identity(monkeypatch)

    monkeypatch.setattr(scaling, "_current_code_identity", lambda: {
        "git_sha": "current-head",
        "git_dirty": True,
        "git_dirty_fingerprint": "current-dirty",
    })
    (run_dir / "provenance.json").write_text(json.dumps({
        "git_sha": "current-head",
        "git_dirty": True,
        "git_dirty_fingerprint": "saved-dirty",
    }), encoding="utf-8")
    assert scaling._cell_is_current(run_dir, cfg, ds, max_tokens=1000) is False
    (run_dir / "provenance.json").write_text(json.dumps({
        "git_sha": "current-head",
        "git_dirty": True,
        "git_dirty_fingerprint": "current-dirty",
    }), encoding="utf-8")
    assert scaling._cell_is_current(run_dir, cfg, ds, max_tokens=1000) is True


@pytest.mark.parametrize(
    "malformed",
    [None, [], "not-an-object", 7],
    ids=["null", "list", "string", "number"],
)
def test_cell_is_current_rejects_non_object_provenance(tmp_path, malformed):
    run_dir = tmp_path / "cell"
    run_dir.mkdir()
    ds = "wikitext-103"
    cfg = VFE3Config(**scaling._cell_cfg_dict({"vocab_size": 64}, 0, 1))
    (run_dir / "summary.json").write_text("{}", encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps({
        "dataset": ds,
        "config": json.loads(json.dumps(asdict(cfg), default=str)),
    }), encoding="utf-8")
    (run_dir / "scaling_cell.json").write_text(
        json.dumps({"label": "cell", "max_tokens": 1000}), encoding="utf-8")
    (run_dir / "provenance.json").write_text(json.dumps(malformed), encoding="utf-8")

    assert scaling._cell_is_current(run_dir, cfg, ds, max_tokens=1000) is False


@pytest.mark.parametrize(
    "malformed",
    [None, [], "not-an-object", 7],
    ids=["null", "list", "string", "number"],
)
def test_cell_is_current_rejects_non_object_cell_metadata(tmp_path, malformed):
    run_dir = tmp_path / "cell"
    run_dir.mkdir()
    ds = "wikitext-103"
    cfg = VFE3Config(**scaling._cell_cfg_dict({"vocab_size": 64}, 0, 1))
    (run_dir / "summary.json").write_text("{}", encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps({
        "dataset": ds,
        "config": json.loads(json.dumps(asdict(cfg), default=str)),
    }), encoding="utf-8")
    (run_dir / "scaling_cell.json").write_text(json.dumps(malformed), encoding="utf-8")
    (run_dir / "provenance.json").write_text(json.dumps({
        "git_sha": "current-head",
        "git_dirty": False,
        "git_dirty_fingerprint": None,
    }), encoding="utf-8")

    assert scaling._cell_is_current(run_dir, cfg, ds, max_tokens=1000) is False


# --------------------------------------------------------------------------- PB-07: end-to-end
# registered scaling figures (capacity_scaling.png / pareto_frontier.png) driven by scaling_analysis


def _record_complete_scaling_cell(root, route, label, seed):
    path = root / "scaling_design.json"
    design = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {
        "schema_version": 1,
        "routes": [],
        "seeds": [],
        "status": "complete",
        "cells": [],
    }
    design["routes"] = sorted(set([*design["routes"], route]))
    design["seeds"] = sorted(set([*design["seeds"], seed]))
    design["cells"].append({
        "route": route,
        "label": label,
        "seed": seed,
        "status": "complete",
    })
    path.write_text(json.dumps(design), encoding="utf-8")


def _write_val_run(root, *, route, scale_knob, label, seed, embed_dim, n_heads, n_layers,
                    n_params, best_val_ppl, wall_time_s):
    run = root / f"{route}_{label}_s{seed}"
    run.mkdir()
    (run / "summary.json").write_text(json.dumps({
        "n_params": n_params,
        "best_val_ppl": best_val_ppl,
        "wall_time_s": wall_time_s,
        "scaling_point": {
            "n_params": n_params, "n_learnable_params": n_params,
            "embed_dim": embed_dim, "n_heads": n_heads, "n_gen": 4,
            "gauge_group": "glk", "n_layers": n_layers, "n_e_steps": 1,
            "tokens_seen": 1000, "test_ce": 3.0,
        },
    }), encoding="utf-8")
    (run / "config.json").write_text(json.dumps({"config": {
        "seed": seed, "embed_dim": embed_dim, "n_heads": n_heads, "n_layers": n_layers,
        "n_e_steps": 1, "family": "gaussian_diagonal",
    }}), encoding="utf-8")
    (run / "scaling_cell.json").write_text(json.dumps({
        "route": route, "scale_knob": scale_knob, "label": label,
    }), encoding="utf-8")
    (run / "provenance.json").write_text(json.dumps({
        "seed": seed, "git_sha": "git-a", "train_data_sha256": "train-a",
        "val_data_sha256": "val-a", "test_data_sha256": "test-a", "data_sha256": "test-a",
    }), encoding="utf-8")
    _record_complete_scaling_cell(root, route, label, seed)


def test_scaling_analysis_emits_capacity_scaling_and_pareto_frontier_pngs(tmp_path, monkeypatch):
    r"""End-to-end PB-07 gate: analyze() must dispatch BOTH registered validation-metric figures
    (capacity_scaling.png, pareto_frontier.png), built from persisted best_val_ppl, alongside the
    unchanged test-metric figures."""
    for embed_dim, n_params, best_val_ppl in ((20, 1000, 40.0), (40, 2000, 30.0), (80, 4000, 20.0)):
        for seed, delta in ((0, -1.0), (1, 1.0)):
            _write_val_run(
                tmp_path, route="grow_K", scale_knob="embed_dim", label=f"K{embed_dim}",
                seed=seed, embed_dim=embed_dim, n_heads=4, n_layers=2, n_params=n_params,
                best_val_ppl=best_val_ppl + delta, wall_time_s=10.0 + embed_dim,
            )
    for n_layers in (1, 2, 3):
        for seed, delta in ((0, -0.5), (1, 0.5)):
            _write_val_run(
                tmp_path, route="inference", scale_knob="n_layers", label=f"L{n_layers}",
                seed=seed, embed_dim=20, n_heads=4, n_layers=n_layers, n_params=1000,
                best_val_ppl=25.0 + n_layers + delta, wall_time_s=5.0 * n_layers,
            )

    monkeypatch.setitem(scaling_analysis.CONFIG, "input_dir", str(tmp_path))
    monkeypatch.setitem(scaling_analysis.CONFIG, "n_bootstrap", 0)
    scaling_analysis.analyze()

    fig_dir = tmp_path / "figures"
    capacity_png = fig_dir / "capacity_scaling.png"
    pareto_png = fig_dir / "pareto_frontier.png"
    assert capacity_png.is_file() and capacity_png.stat().st_size > 0
    assert pareto_png.is_file() and pareto_png.stat().st_size > 0


def test_analyze_survives_explicit_null_best_val_ppl_and_skips_validation_figures(tmp_path, monkeypatch, caplog):
    r"""PB-07 robustness: an explicit-null best_val_ppl beside finite sibling seeds is a
    data-integrity fault for the VALIDATION figures only. analyze() must still produce every legacy
    output (CSV harvest, fit summary, markdown report) and withhold just the two registered
    validation figures, logging one warning that names the fault -- the adapter's fail-loud
    ValueError contract stays, it just cannot take down the unrelated test-metric analysis."""
    import logging as _logging

    for embed_dim, n_params, best_val_ppl in ((20, 1000, 40.0), (40, 2000, 30.0), (80, 4000, 20.0)):
        for seed, delta in ((0, -1.0), (1, 1.0)):
            _write_val_run(
                tmp_path, route="grow_K", scale_knob="embed_dim", label=f"K{embed_dim}",
                seed=seed, embed_dim=embed_dim, n_heads=4, n_layers=2, n_params=n_params,
                best_val_ppl=(None if (embed_dim, seed) == (40, 1) else best_val_ppl + delta),
                wall_time_s=10.0 + embed_dim,
            )

    monkeypatch.setitem(scaling_analysis.CONFIG, "input_dir", str(tmp_path))
    monkeypatch.setitem(scaling_analysis.CONFIG, "n_bootstrap", 0)
    with caplog.at_level(_logging.WARNING, logger="scaling_analysis"):
        scaling_analysis.analyze()                              # must NOT raise

    # legacy outputs are all still produced
    assert (tmp_path / "scaling_points.csv").is_file()
    assert (tmp_path / "scaling_summary.json").is_file()
    assert (tmp_path / "SCALING_ANALYSIS.md").is_file()
    summary = json.loads((tmp_path / "scaling_summary.json").read_text(encoding="utf-8"))
    assert summary["n_runs"] == 6

    # the two validation figures are withheld, with one warning naming the fault
    fig_dir = tmp_path / "figures"
    assert not (fig_dir / "capacity_scaling.png").exists()
    assert not (fig_dir / "pareto_frontier.png").exists()
    null_warnings = [r for r in caplog.records if "explicit-null" in r.getMessage()]
    assert len(null_warnings) == 1
