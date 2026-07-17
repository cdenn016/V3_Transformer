r"""Tests for the muP width-stability scaling route (F1/EXP-6) added 2026-06-21.

route_grow_k_mup emits a matched fixed-LR vs muP pair per width, each with the per-cell kl_max=8*K
confound fix, and every cell builds a valid VFE3Config/VFEModel."""
import hashlib
import json
import math
from dataclasses import asdict

import pytest
import torch

import scaling
import scaling_analysis
from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel


_SOURCE_IDENTITIES = {
    split: {
        "format": "pt",
        "tokenizer_tag": "tiktoken",
        "size_bytes": len(split),
        "sha256": split * 8,
        "meta": None,
        "meta_sha256": None,
    }
    for split in ("train", "validation", "test")
}


def _complete_scaling_summary():
    test_ce = 2.0
    metrics = {
        "n_params": 10,
        "test_ce": test_ce,
        "test_ppl": math.exp(test_ce),
        "test_bits_per_token": test_ce / math.log(2.0),
        "test_bpc": None,
    }
    return {**metrics, "scaling_point": dict(metrics)}


def test_scaling_recompute_removes_owned_stale_artifacts_but_preserves_user_notes(tmp_path):
    run_dir = tmp_path / "cell"
    (run_dir / "figures").mkdir(parents=True)
    (run_dir / scaling._SCALING_OWNER_FILENAME).write_text(
        json.dumps(scaling._scaling_owner_payload("grow_K", "K20", 0)),
        encoding="utf-8",
    )
    (run_dir / "checkpoints").mkdir()
    (run_dir / "attention").mkdir()
    for path in (
        run_dir / "figures" / "old.png",
        run_dir / "checkpoints" / "step_1.pt",
        run_dir / "attention" / "old.png",
        run_dir / "loss_curve.png",
        run_dir / "summary.json",
        run_dir / "best_model.pt",
    ):
        path.write_bytes(b"stale")
    note = run_dir / "user_notes.txt"
    note.write_text("preserve me", encoding="utf-8")
    user_plot = run_dir / "user_plot.png"
    user_plot.write_bytes(b"preserve me too")

    scaling._reset_stale_scaling_artifacts(
        run_dir,
        route="grow_K",
        label="K20",
        seed=0,
    )

    assert note.read_text(encoding="utf-8") == "preserve me"
    assert user_plot.read_bytes() == b"preserve me too"
    assert not (run_dir / "figures").exists()
    assert not (run_dir / "checkpoints").exists()
    assert not (run_dir / "attention").exists()
    assert not (run_dir / "loss_curve.png").exists()
    assert not (run_dir / "summary.json").exists()
    assert not (run_dir / "best_model.pt").exists()


def test_scaling_recompute_rejects_and_preserves_an_unowned_collision(tmp_path):
    run_dir = tmp_path / "cell"
    run_dir.mkdir()
    note = run_dir / "user_notes.txt"
    note.write_text("preserve me", encoding="utf-8")

    with pytest.raises(ValueError, match="ownership|promotable"):
        scaling._reset_stale_scaling_artifacts(
            run_dir,
            route="grow_K",
            label="K20",
            seed=0,
        )

    assert note.read_text(encoding="utf-8") == "preserve me"
    assert sorted(path.name for path in run_dir.iterdir()) == ["user_notes.txt"]


def test_scaling_recompute_promotes_one_exact_legacy_cell_marker(tmp_path):
    run_dir = tmp_path / "cell"
    run_dir.mkdir()
    (run_dir / "scaling_cell.json").write_text(
        json.dumps({"route": "grow_K", "label": "K20", "seed": 0}),
        encoding="utf-8",
    )
    stale = run_dir / "summary.json"
    stale.write_text("{}", encoding="utf-8")

    scaling._reset_stale_scaling_artifacts(
        run_dir,
        route="grow_K",
        label="K20",
        seed=0,
    )

    owner = json.loads(
        (run_dir / scaling._SCALING_OWNER_FILENAME).read_text(encoding="utf-8")
    )
    assert owner == scaling._scaling_owner_payload("grow_K", "K20", 0)
    assert not (run_dir / "scaling_cell.json").exists()
    assert not stale.exists()


def test_scaling_cell_path_rejects_reparse_ancestor_and_traversal(tmp_path, monkeypatch):
    output_dir = scaling._trusted_scaling_output_dir(tmp_path / "output")
    route_dir = output_dir / "grow_K"
    route_dir.mkdir()
    real_probe = scaling._path_is_reparse_point
    monkeypatch.setattr(
        scaling,
        "_path_is_reparse_point",
        lambda path: path == route_dir or real_probe(path),
    )

    with pytest.raises(ValueError, match="unsafe scaling cell path"):
        scaling._trusted_scaling_run_dir(output_dir, "grow_K", "K20", 0)

    monkeypatch.setattr(scaling, "_path_is_reparse_point", real_probe)
    with pytest.raises(ValueError):
        scaling._trusted_scaling_run_dir(output_dir, "../escape", "K20", 0)


def _patch_source_identity(monkeypatch):
    monkeypatch.setattr(
        scaling,
        "cache_source_identity",
        lambda dataset, split: dict(_SOURCE_IDENTITIES[split]),
    )


def _bind_scaling_reuse_contract(run_dir, cfg, dataset, code_identity):
    built = json.loads(json.dumps(asdict(cfg), default=str))
    cellmeta = json.loads((run_dir / "scaling_cell.json").read_text(encoding="utf-8"))
    cellmeta.pop("reuse_contract_sha256", None)
    cellmeta.update({
        "schema_version": 2,
        "dataset": dataset,
        "config_sha256": hashlib.sha256(
            json.dumps(built, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "code_identity": code_identity,
    })
    digest = hashlib.sha256(
        json.dumps(cellmeta, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    cellmeta["reuse_contract_sha256"] = digest
    (run_dir / "scaling_cell.json").write_text(json.dumps(cellmeta), encoding="utf-8")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    summary.update(_complete_scaling_summary())
    summary["scaling_reuse_contract_sha256"] = digest
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return digest


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
    code_identity = {
        "git_sha": "current-head",
        "git_dirty": False,
        "git_dirty_fingerprint": None,
    }
    (run_dir / "provenance.json").write_text(json.dumps(code_identity), encoding="utf-8")
    _bind_scaling_reuse_contract(run_dir, cfg, ds, code_identity)
    monkeypatch.setattr(scaling, "_current_code_identity", lambda: dict(code_identity))
    _patch_source_identity(monkeypatch)

    assert scaling._cell_is_current(run_dir, cfg, ds, max_tokens=1000) is True
    assert scaling._cell_is_current(run_dir, cfg, ds, max_tokens=None) is False
    (run_dir / "scaling_failure.json").write_text(
        json.dumps({"status": "failed", "error": "stale failure"}), encoding="utf-8")
    assert scaling._cell_is_current(run_dir, cfg, ds, max_tokens=1000) is False
    (run_dir / "scaling_failure.json").unlink()
    assert scaling._cell_is_current(run_dir, cfg, ds, max_tokens=1000) is True


def test_cell_is_current_rejects_missing_or_mismatched_code_identity(tmp_path, monkeypatch):
    run_dir = tmp_path / "cell"
    run_dir.mkdir()
    ds = "wikitext-103"
    cfg = VFE3Config(**scaling._cell_cfg_dict({"vocab_size": 64}, 0, 1))
    code_identity = {
        "git_sha": "current-head",
        "git_dirty": False,
        "git_dirty_fingerprint": None,
    }
    (run_dir / "summary.json").write_text("{}", encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps({
        "dataset": ds,
        "config": json.loads(json.dumps(asdict(cfg), default=str)),
    }), encoding="utf-8")
    (run_dir / "scaling_cell.json").write_text(json.dumps({
        "label": "cell", "max_tokens": 1000, "data_sources": _SOURCE_IDENTITIES,
    }), encoding="utf-8")
    _bind_scaling_reuse_contract(run_dir, cfg, ds, code_identity)
    monkeypatch.setattr(scaling, "_current_code_identity", lambda: dict(code_identity))
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

    dirty_identity = {
        "git_sha": "current-head",
        "git_dirty": True,
        "git_dirty_fingerprint": "current-dirty",
    }
    _bind_scaling_reuse_contract(run_dir, cfg, ds, dirty_identity)
    monkeypatch.setattr(scaling, "_current_code_identity", lambda: dict(dirty_identity))
    (run_dir / "provenance.json").write_text(json.dumps({
        "git_sha": "current-head",
        "git_dirty": True,
        "git_dirty_fingerprint": "saved-dirty",
    }), encoding="utf-8")
    assert scaling._cell_is_current(run_dir, cfg, ds, max_tokens=1000) is False
    (run_dir / "provenance.json").write_text(json.dumps(dirty_identity), encoding="utf-8")
    assert scaling._cell_is_current(run_dir, cfg, ds, max_tokens=1000) is True


def test_cell_is_current_rejects_summary_from_another_reuse_contract(tmp_path, monkeypatch):
    run_dir = tmp_path / "cell"
    run_dir.mkdir()
    ds = "wikitext-103"
    cfg = VFE3Config(**scaling._cell_cfg_dict({"vocab_size": 64}, 0, 1))
    code_identity = {
        "git_sha": "current-head",
        "git_dirty": False,
        "git_dirty_fingerprint": None,
    }
    (run_dir / "summary.json").write_text("{}", encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps({
        "dataset": ds,
        "config": json.loads(json.dumps(asdict(cfg), default=str)),
    }), encoding="utf-8")
    (run_dir / "scaling_cell.json").write_text(json.dumps({
        "label": "cell", "max_tokens": 1000, "data_sources": _SOURCE_IDENTITIES,
    }), encoding="utf-8")
    (run_dir / "provenance.json").write_text(json.dumps(code_identity), encoding="utf-8")
    digest = _bind_scaling_reuse_contract(run_dir, cfg, ds, code_identity)
    monkeypatch.setattr(scaling, "_current_code_identity", lambda: dict(code_identity))
    _patch_source_identity(monkeypatch)
    valid_summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    (run_dir / "summary.json").write_text(json.dumps({
        "scaling_reuse_contract_sha256": "0" * 64,
    }), encoding="utf-8")
    assert scaling._cell_is_current(run_dir, cfg, ds, max_tokens=1000) is False
    (run_dir / "summary.json").write_text(json.dumps({
        "scaling_reuse_contract_sha256": digest,
    }), encoding="utf-8")
    assert scaling._cell_is_current(run_dir, cfg, ds, max_tokens=1000) is False

    corrupt_summary = dict(valid_summary)
    corrupt_summary["test_ppl"] = 2.0
    corrupt_summary["scaling_point"] = dict(valid_summary["scaling_point"])
    corrupt_summary["scaling_point"]["test_ppl"] = 2.0
    (run_dir / "summary.json").write_text(json.dumps(corrupt_summary), encoding="utf-8")
    assert scaling._cell_is_current(run_dir, cfg, ds, max_tokens=1000) is False

    (run_dir / "summary.json").write_text(json.dumps(valid_summary), encoding="utf-8")
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


def _record_complete_scaling_cell(root, route, label, seed, scale_knob):
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
        "scale_knob": scale_knob,
        "run_dir": f"{route}/{label}/s{seed}",
        "status": "complete",
    })
    path.write_text(json.dumps(design), encoding="utf-8")


def _write_val_run(root, *, route, scale_knob, label, seed, embed_dim, n_heads, n_layers,
                    n_params, best_val_ppl, wall_time_s):
    run = root / route / label / f"s{seed}"
    run.mkdir(parents=True)
    config = {
        "seed": seed, "embed_dim": embed_dim, "n_heads": n_heads, "n_layers": n_layers,
        "n_e_steps": 1, "family": "gaussian_diagonal",
    }
    code_identity = {
        "git_sha": "git-a",
        "git_dirty": False,
        "git_dirty_fingerprint": None,
    }
    cell = {
        "schema_version": 2,
        "route": route,
        "scale_knob": scale_knob,
        "label": label,
        "seed": seed,
        "dataset": "synthetic",
        "config_sha256": hashlib.sha256(json.dumps(
            config, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest(),
        "code_identity": code_identity,
        "data_sources": {
            split: {"format": "pt", "size_bytes": len(split), "sha256": split[0] * 64}
            for split in ("train", "validation", "test")
        },
    }
    digest = hashlib.sha256(json.dumps(
        cell, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    cell["reuse_contract_sha256"] = digest
    test_ce = 3.0
    metrics = {
        "n_params": n_params,
        "test_ce": test_ce,
        "test_ppl": math.exp(test_ce),
        "test_bits_per_token": test_ce / math.log(2.0),
        "test_bpc": None,
    }
    (run / "summary.json").write_text(json.dumps({
        **metrics,
        "best_val_ppl": best_val_ppl,
        "wall_time_s": wall_time_s,
        "scaling_reuse_contract_sha256": digest,
        "scaling_point": {
            **metrics,
            "n_learnable_params": n_params,
            "embed_dim": embed_dim,
            "n_heads": n_heads,
            "n_gen": 4,
            "gauge_group": "glk",
            "n_layers": n_layers,
            "n_e_steps": 1,
            "tokens_seen": 1000,
        },
    }), encoding="utf-8")
    (run / "config.json").write_text(json.dumps({
        "dataset": "synthetic",
        "config": config,
    }), encoding="utf-8")
    (run / "scaling_cell.json").write_text(json.dumps(cell), encoding="utf-8")
    (run / "provenance.json").write_text(json.dumps({
        "seed": seed, **code_identity, "train_data_sha256": "train-a",
        "val_data_sha256": "val-a", "test_data_sha256": "test-a", "data_sha256": "test-a",
    }), encoding="utf-8")
    _record_complete_scaling_cell(root, route, label, seed, scale_knob)


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
