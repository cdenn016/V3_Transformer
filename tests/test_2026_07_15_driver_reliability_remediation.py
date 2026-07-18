"""Regressions for Task 6 driver reliability and experiment construction."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from unittest import mock

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import torch

import ablation
import efe_ring_experiment
import generate_efe
import multiseed_analysis
import scaling
import scaling_analysis
import train_vfe3
from vfe3.config import VFE3Config
from vfe3.inference import ring_task
from vfe3.run_artifacts import RunArtifacts
from vfe3.runtime import deterministic_state, seed_everything
from vfe3.viz import embedding_comparison, figures, report


def _tiny_config(**overrides: object) -> VFE3Config:
    values = {
        "vocab_size": 8,
        "embed_dim": 4,
        "n_heads": 2,
        "max_seq_len": 4,
        "n_layers": 1,
        "n_e_steps": 1,
        "use_prior_bank": False,
        "use_head_mixer": False,
    }
    values.update(overrides)
    return VFE3Config(**values)


def test_same_second_run_directories_are_reserved_atomically(tmp_path, monkeypatch):
    fixed = datetime(2026, 7, 16, 12, 34, 56)
    monkeypatch.setattr(train_vfe3, "RUN_ROOT", str(tmp_path))
    with mock.patch("datetime.datetime") as frozen_datetime:
        frozen_datetime.now.return_value = fixed
        first = Path(train_vfe3._run_dir(_tiny_config(seed=7), "synthetic"))
        second = Path(train_vfe3._run_dir(_tiny_config(seed=7), "synthetic"))

    assert first != second
    assert first.is_dir() and second.is_dir()
    assert first.name == "20260716-123456_synthetic_K4_block_glk_linear_s7"
    assert second.name == first.name + "_2"


def test_concurrent_json_publication_uses_unique_writer_temporaries(tmp_path, monkeypatch):
    cfg = _tiny_config()
    artifacts = RunArtifacts(tmp_path / "run", cfg, torch.nn.Linear(1, 1))
    real_replace = __import__("vfe3.run_artifacts", fromlist=["_atomic_replace"])._atomic_replace
    barrier = threading.Barrier(2)
    temporaries: list[Path] = []
    errors: list[BaseException] = []

    def delayed_replace(final: Path, temporary: Path, **kwargs: object) -> None:
        temporaries.append(Path(temporary))
        barrier.wait(timeout=5)
        real_replace(final, temporary, delay=0.01, retries=20)

    monkeypatch.setattr("vfe3.run_artifacts._atomic_replace", delayed_replace)

    def writer(value: int) -> None:
        try:
            artifacts.save_json("shared.json", {"writer": value})
        except BaseException as exc:  # retained for assertion in the main test thread
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(writer, (1, 2)))

    assert errors == []
    assert len(temporaries) == 2 and len(set(temporaries)) == 2
    assert json.loads((artifacts.run_dir / "shared.json").read_text())["writer"] in {1, 2}
    assert not list(artifacts.run_dir.glob("*.tmp"))


def test_ring_training_applies_shared_deterministic_runtime_contract(monkeypatch):
    monkeypatch.setattr(ring_task, "predictive_adequacy", lambda *args, **kwargs: 1.0)
    seed_everything(0, deterministic=False)
    try:
        model, _ = ring_task.train_ring_checkpoint(
            seed=17,
            steps=0,
            batch_size=2,
            embed_dim=4,
            n_heads=2,
            n_layers=1,
            n_e_steps=1,
            device="cpu",
            cfg_overrides={"deterministic": True},
        )
        state = deterministic_state()
        assert model.cfg.deterministic is True
        assert state == {
            "algorithms": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cublas_workspace_config": ":4096:8",
        }
    finally:
        seed_everything(0, deterministic=False)


def test_ring_seed_bundle_persists_effective_runtime_state(tmp_path):
    model = ring_task.VFEModel(_tiny_config(max_seq_len=ring_task.SEQ_LEN))
    path = tmp_path / "seed_1.pt"
    efe_ring_experiment._save_seed_bundle(
        path,
        model,
        {"seeds": [1], "steps": 1},
        None,
        seed=1,
        adequacy=0.5,
        status="trained",
    )
    bundle = torch.load(path, weights_only=True)
    assert bundle["runtime_state"] == efe_ring_experiment._runtime_identity(torch.device("cpu"))


def _write_scaling_run(
    root: Path,
    label: str,
    seed: int,
    n_params: int,
    test_ce: float,
    *,
    route: str = "route",
) -> None:
    run_dir = root / route / label / f"s{seed}"
    run_dir.mkdir(parents=True)
    config = {"seed": seed, "embed_dim": n_params}
    code_identity = {
        "git_sha": "same",
        "git_dirty": False,
        "git_dirty_fingerprint": None,
    }
    sources = {
        split: {
            "format": "pt",
            "size_bytes": len(split),
            "sha256": split[0] * 64,
        }
        for split in ("train", "validation", "test")
    }
    cell = {
        "schema_version": 2,
        "route": route,
        "label": label,
        "scale_knob": "embed_dim",
        "seed": seed,
        "dataset": "synthetic",
        "config_sha256": hashlib.sha256(json.dumps(
            config, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest(),
        "code_identity": code_identity,
        "data_sources": sources,
    }
    digest = hashlib.sha256(json.dumps(
        cell, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    cell["reuse_contract_sha256"] = digest
    metrics = {
        "n_params": n_params,
        "test_ce": test_ce,
        "test_ppl": math.exp(test_ce),
        "test_bits_per_token": test_ce / math.log(2.0),
        "test_bpc": None,
    }
    (run_dir / "summary.json").write_text(json.dumps({
        **metrics,
        "best_val_ppl": 10.0,
        "scaling_reuse_contract_sha256": digest,
        "scaling_point": {
            **metrics,
            "n_gen": 4,
            "embed_dim": n_params,
            "tokens_seen": 100,
        },
    }), encoding="utf-8")
    (run_dir / "test_results.json").write_text(json.dumps(metrics), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps({
        "dataset": "synthetic",
        "config": config,
    }), encoding="utf-8")
    (run_dir / "provenance.json").write_text(json.dumps({
        "seed": seed,
        **code_identity,
        "train_data_sha256": "train",
        "val_data_sha256": "val",
        "test_data_sha256": "test",
    }), encoding="utf-8")
    (run_dir / "scaling_cell.json").write_text(json.dumps(cell), encoding="utf-8")


def _complete_scaling_design(*, route: str = "route") -> dict[str, object]:
    return {
        "schema_version": 1,
        "routes": [route],
        "seeds": [1],
        "status": "complete",
        "cells": [
            {
                "route": route,
                "label": "small",
                "seed": 1,
                "scale_knob": "embed_dim",
                "run_dir": f"{route}/small/s1",
                "status": "complete",
            },
            {
                "route": route,
                "label": "large",
                "seed": 1,
                "scale_knob": "embed_dim",
                "run_dir": f"{route}/large/s1",
                "status": "complete",
            },
        ],
    }


@pytest.mark.parametrize("routes", ([], ["grow_K", "grow_K"]))
def test_scaling_requires_nonempty_unique_routes(routes: list[str]) -> None:
    with pytest.raises(ValueError, match="routes.*non-empty|routes.*unique"):
        scaling._validated_scaling_routes(routes)


@pytest.mark.parametrize("labels", ([""], ["same", "same"]))
def test_scaling_requires_nonempty_unique_cell_labels(
    labels:      list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        scaling,
        "ROUTES",
        {"probe": [{"label": label} for label in labels]},
    )

    with pytest.raises(ValueError, match="cell labels.*nonempty|cell labels.*unique"):
        scaling._validated_scaling_routes(["probe"])


@pytest.mark.parametrize("component_kind", ("route", "label"))
@pytest.mark.parametrize(
    "bad_component",
    (
        ".",
        "..",
        "../escape",
        r"..\escape",
        "nested/child",
        r"nested\child",
        "/absolute",
        r"\absolute",
        r"C:\absolute",
        "C:drive-relative",
        r"\\server\share",
        "trailing.",
        "trailing ",
        "   ",
        "NUL",
        "CONIN$",
        "CONOUT$",
        "CLOCK$",
        "x" * 256,
        "🙂" * 100,
        "COM¹",
        "LPT³.txt",
    ),
)
def test_scaling_rejects_unsafe_path_components_before_output_creation(
    tmp_path,
    monkeypatch,
    component_kind,
    bad_component,
):
    route_name = bad_component if component_kind == "route" else "safe_route"
    label = bad_component if component_kind == "label" else "safe_label"
    monkeypatch.setattr(scaling, "ROUTES", {route_name: [{"label": label}]})
    monkeypatch.setitem(scaling.CONFIG, "routes", [route_name])
    output_dir = tmp_path / "must-not-exist"
    monkeypatch.setitem(scaling.CONFIG, "output_dir", str(output_dir))

    with pytest.raises(ValueError, match="safe single path component"):
        scaling.main()

    assert not output_dir.exists()


@pytest.mark.parametrize("bad_component", ("CONIN$", "CONOUT$", "CLOCK$", "COM¹", "LPT³.txt"))
def test_scaling_analysis_portably_rejects_windows_devices(bad_component):
    assert not scaling_analysis._safe_manifest_component(bad_component)


@pytest.mark.parametrize("component_kind", ("route", "label"))
def test_scaling_rejects_casefold_path_aliases_before_output_creation(
    tmp_path,
    monkeypatch,
    component_kind,
):
    if component_kind == "route":
        routes = ["Probe", "probe"]
        registry = {
            "Probe": [{"label": "first"}],
            "probe": [{"label": "second"}],
        }
    else:
        routes = ["safe_route"]
        registry = {
            "safe_route": [{"label": "Cell"}, {"label": "cell"}],
        }
    monkeypatch.setattr(scaling, "ROUTES", registry)
    monkeypatch.setitem(scaling.CONFIG, "routes", routes)
    output_dir = tmp_path / "must-not-exist"
    monkeypatch.setitem(scaling.CONFIG, "output_dir", str(output_dir))

    with pytest.raises(ValueError, match="filesystem aliases"):
        scaling.main()

    assert not output_dir.exists()


def test_scaling_declared_route_and_label_components_remain_valid() -> None:
    route_names = list(scaling.ROUTES)

    assert scaling._validated_scaling_routes(route_names) == route_names


def test_scaling_analysis_refuses_survivor_only_fit_for_incomplete_design(tmp_path, monkeypatch):
    _write_scaling_run(tmp_path, "small", 1, 10, 2.0)
    _write_scaling_run(tmp_path, "large", 1, 20, 1.5)
    (tmp_path / "scaling_design.json").write_text(json.dumps({
        "schema_version": 1,
        "routes": ["route"],
        "seeds": [1],
        "status": "incomplete",
        "cells": [
            {"route": "route", "label": "small", "seed": 1, "scale_knob": "embed_dim",
             "run_dir": "route/small/s1", "status": "complete"},
            {"route": "route", "label": "large", "seed": 1, "scale_knob": "embed_dim",
             "run_dir": "route/large/s1", "status": "complete"},
            {"route": "route", "label": "failed", "seed": 1, "scale_knob": "embed_dim",
             "run_dir": "route/failed/s1", "status": "failed",
             "error_kind": "train", "error": "probe failure"},
        ],
    }), encoding="utf-8")
    monkeypatch.setitem(scaling_analysis.CONFIG, "input_dir", str(tmp_path))
    monkeypatch.setitem(scaling_analysis.CONFIG, "with_offset", False)
    monkeypatch.setitem(scaling_analysis.CONFIG, "n_bootstrap", 0)
    figure_inputs = []

    def record_figure_inputs(param_points, infer_points, fig_dir, **kwargs):
        del fig_dir
        figure_inputs.append((param_points, infer_points, kwargs.get("validation_points")))

    monkeypatch.setattr(scaling_analysis, "_make_figures", record_figure_inputs)

    scaling_analysis.analyze()

    summary = json.loads((tmp_path / "scaling_summary.json").read_text(encoding="utf-8"))
    assert summary["design"]["complete"] is False
    assert summary["design"]["cells"][-1]["status"] == "failed"
    assert summary["pooled_fit"] is None
    assert summary["pooled_fit_status"] == "incomplete_design"
    assert figure_inputs == [([], [], None)]


@pytest.mark.parametrize(
    "manifest_case",
    [
        "missing",
        "pending",
        "running",
        "incomplete",
        "failed",
        "unverifiable",
        "missing_status",
        "missing_routes",
        "missing_seeds",
        "missing_cells",
        "missing_cell_status",
        "missing_cell_route",
        "missing_cell_label",
        "missing_cell_seed",
        "duplicate_cell",
        "bad_schema",
    ],
)
def test_scaling_analysis_withholds_all_fit_inputs_for_unfinished_or_malformed_manifest(
    tmp_path, monkeypatch, manifest_case,
):
    _write_scaling_run(tmp_path, "small", 1, 10, 2.0)
    _write_scaling_run(tmp_path, "large", 1, 20, 1.5)
    manifest = _complete_scaling_design()
    if manifest_case in {"pending", "running", "incomplete", "failed", "unverifiable"}:
        manifest["status"] = manifest_case
    elif manifest_case == "missing_status":
        manifest.pop("status")
    elif manifest_case == "missing_routes":
        manifest.pop("routes")
    elif manifest_case == "missing_seeds":
        manifest.pop("seeds")
    elif manifest_case == "missing_cells":
        manifest.pop("cells")
    elif manifest_case.startswith("missing_cell_"):
        field = manifest_case.removeprefix("missing_cell_")
        manifest["cells"][0].pop(field)
    elif manifest_case == "duplicate_cell":
        manifest["cells"].append(dict(manifest["cells"][0]))
    elif manifest_case == "bad_schema":
        manifest["schema_version"] = 2
    if manifest_case != "missing":
        (tmp_path / "scaling_design.json").write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setitem(scaling_analysis.CONFIG, "input_dir", str(tmp_path))
    monkeypatch.setitem(scaling_analysis.CONFIG, "with_offset", False)
    monkeypatch.setitem(scaling_analysis.CONFIG, "n_bootstrap", 0)
    figure_inputs = []
    monkeypatch.setattr(
        scaling_analysis,
        "_make_figures",
        lambda param, infer, fig_dir, **kwargs: figure_inputs.append(
            (param, infer, kwargs.get("validation_points"))
        ),
    )

    scaling_analysis.analyze()

    summary = json.loads((tmp_path / "scaling_summary.json").read_text(encoding="utf-8"))
    assert summary["design"]["complete"] is False
    assert summary["n_fitted_param_points"] == 0
    assert summary["pooled_fit"] is None
    assert summary["pooled_fit_status"] == "incomplete_design"
    assert summary["frontier_collapse"]["reason"] == "incomplete_design"
    assert summary["estep_structural"] is None
    assert figure_inputs == [([], [], None)]


@pytest.mark.parametrize("status", ["complete", "success"])
def test_scaling_design_accepts_only_explicit_success_terminal_statuses(tmp_path, status):
    _write_scaling_run(tmp_path, "small", 1, 10, 2.0)
    manifest = _complete_scaling_design()
    manifest["status"] = status
    manifest["cells"] = [manifest["cells"][0]]
    manifest["cells"][0]["status"] = status
    (tmp_path / "scaling_design.json").write_text(json.dumps(manifest), encoding="utf-8")

    design = scaling_analysis._requested_design(tmp_path, scaling_analysis.harvest(tmp_path))

    assert design["available"] is True
    assert design["complete"] is True
    assert design["status"] == "complete"


@pytest.mark.parametrize("bad_seed", [True, 1.5, "1", -1])
def test_scaling_analysis_rejects_nonexact_or_negative_request_seed(
    tmp_path, monkeypatch, bad_seed,
):
    (tmp_path / "scaling_design.json").write_text("{}", encoding="utf-8")
    raw = _complete_scaling_design()
    raw["seeds"] = [bad_seed]
    raw["cells"] = [
        {"route": "route", "label": "small", "seed": bad_seed, "status": "complete"},
    ]
    monkeypatch.setattr(scaling_analysis, "_read_json", lambda path: raw)

    design = scaling_analysis._requested_design(tmp_path, [])

    assert design["available"] is False
    assert design["status"] == "unverifiable_design"


def test_scaling_analysis_rejects_integer_enum_request_seed(tmp_path, monkeypatch):
    class Seed(IntEnum):
        ONE = 1

    (tmp_path / "scaling_design.json").write_text("{}", encoding="utf-8")
    raw = _complete_scaling_design()
    raw["seeds"] = [Seed.ONE]
    raw["cells"] = [
        {"route": "route", "label": "small", "seed": Seed.ONE, "status": "complete"},
    ]
    monkeypatch.setattr(scaling_analysis, "_read_json", lambda path: raw)

    assert scaling_analysis._requested_design(tmp_path, [])["available"] is False


def test_scaling_analysis_rejects_integer_enum_request_cell_seed(tmp_path, monkeypatch):
    class Seed(IntEnum):
        ONE = 1

    (tmp_path / "scaling_design.json").write_text("{}", encoding="utf-8")
    raw = _complete_scaling_design()
    raw["cells"] = [
        {"route": "route", "label": "small", "seed": Seed.ONE, "status": "complete"},
    ]
    monkeypatch.setattr(scaling_analysis, "_read_json", lambda path: raw)

    assert scaling_analysis._requested_design(tmp_path, [])["available"] is False


class _StringLike:
    def __init__(self, value: str) -> None:
        self.value = value

    def __str__(self) -> str:
        return self.value


class _StringSubclass(str):
    pass


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("route",  _StringLike("route")),
        ("route",  _StringSubclass("route")),
        ("route",  ""),
        ("label",  _StringLike("small")),
        ("label",  _StringSubclass("small")),
        ("label",  ""),
        ("seed",   True),
        ("seed",   1.5),
        ("seed",   "1"),
        ("seed",   -1),
    ],
)
def test_scaling_analysis_does_not_coerce_observed_join_identity(
    tmp_path, field, bad_value,
):
    (tmp_path / "scaling_design.json").write_text(
        json.dumps(_complete_scaling_design()), encoding="utf-8",
    )
    row = {
        "route": "route",
        "label": "small",
        "seed": 1,
        "scale_knob": "embed_dim",
        "run_dir": str((tmp_path / "route" / "small" / "s1").resolve()),
        "artifact_verified": True,
        "test_ce": 2.0,
    }
    row[field] = bad_value

    design = scaling_analysis._requested_design(tmp_path, [row])

    assert design["complete"] is False
    assert design["cells"][0]["status"] == "missing"


def test_scaling_analysis_rejects_integer_enum_observed_seed(tmp_path):
    class Seed(IntEnum):
        ONE = 1

    (tmp_path / "scaling_design.json").write_text(
        json.dumps(_complete_scaling_design()), encoding="utf-8",
    )
    row = {
        "route": "route",
        "label": "small",
        "seed": Seed.ONE,
        "scale_knob": "embed_dim",
        "run_dir": str((tmp_path / "route" / "small" / "s1").resolve()),
        "artifact_verified": True,
        "test_ce": 2.0,
    }

    design = scaling_analysis._requested_design(tmp_path, [row])

    assert design["complete"] is False
    assert design["cells"][0]["status"] == "missing"


def test_scaling_analysis_uses_root_manifest_for_explicit_route_subdirectory(
    tmp_path, monkeypatch,
):
    route = "blocks_K48"
    root = tmp_path / "vfe3_scaling_results"
    route_dir = root / route
    _write_scaling_run(root, "small", 1, 10, 2.0, route=route)
    _write_scaling_run(root, "large", 1, 20, 1.5, route=route)
    (root / "scaling_design.json").write_text(
        json.dumps(_complete_scaling_design(route=route)),
        encoding="utf-8",
    )
    monkeypatch.setitem(scaling_analysis.CONFIG, "input_dir", str(route_dir))
    monkeypatch.setitem(scaling_analysis.CONFIG, "with_offset", False)
    monkeypatch.setitem(scaling_analysis.CONFIG, "n_bootstrap", 0)
    monkeypatch.setattr(scaling_analysis, "_make_figures", lambda *args, **kwargs: None)

    scaling_analysis.analyze()

    summary = json.loads((route_dir / "scaling_summary.json").read_text(encoding="utf-8"))
    assert summary["design"]["available"] is True
    assert summary["design"]["complete"] is True
    assert summary["design"]["status"] == "complete"
    assert summary["n_fitted_param_points"] == 2
    assert summary["pooled_fit"] is not None
    assert summary["pooled_fit_status"] == "clean"


def test_scaling_analysis_parent_manifest_selects_only_requested_route_cells(
    tmp_path, monkeypatch,
):
    selected_route = "blocks_K48"
    other_route = "grow_K"
    root = tmp_path / "vfe3_scaling_results"
    for route in (selected_route, other_route):
        _write_scaling_run(root, "small", 1, 10, 2.0, route=route)
        _write_scaling_run(root, "large", 1, 20, 1.5, route=route)
    (root / "scaling_design.json").write_text(json.dumps({
        "schema_version": 1,
        "routes": [selected_route, other_route],
        "seeds": [1],
        "status": "complete",
        "cells": [
            {
                "route": route,
                "label": label,
                "seed": 1,
                "scale_knob": "embed_dim",
                "run_dir": f"{route}/{label}/s1",
                "status": "complete",
            }
            for route in (selected_route, other_route)
            for label in ("small", "large")
        ],
    }), encoding="utf-8")
    selected_dir = root / selected_route
    monkeypatch.setitem(scaling_analysis.CONFIG, "input_dir", str(selected_dir))
    monkeypatch.setitem(scaling_analysis.CONFIG, "with_offset", False)
    monkeypatch.setitem(scaling_analysis.CONFIG, "n_bootstrap", 0)
    monkeypatch.setattr(scaling_analysis, "_make_figures", lambda *args, **kwargs: None)

    scaling_analysis.analyze()

    summary = json.loads((selected_dir / "scaling_summary.json").read_text(encoding="utf-8"))
    assert summary["design"]["complete"] is True
    assert {cell["route"] for cell in summary["design"]["cells"]} == {selected_route}
    assert summary["n_fitted_param_points"] == 2
    assert summary["pooled_fit"] is not None


def test_scaling_analysis_checked_in_default_reads_scaling_default_output_root(
    tmp_path, monkeypatch,
):
    output_root = Path(scaling.CONFIG["output_dir"])
    analysis_input = Path(scaling_analysis.CONFIG["input_dir"])
    assert analysis_input == output_root

    route = scaling.CONFIG["routes"][0]
    root = tmp_path / output_root
    _write_scaling_run(root, "small", 1, 10, 2.0, route=route)
    _write_scaling_run(root, "large", 1, 20, 1.5, route=route)
    (root / "scaling_design.json").write_text(
        json.dumps(_complete_scaling_design(route=route)),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(scaling_analysis.CONFIG, "with_offset", False)
    monkeypatch.setitem(scaling_analysis.CONFIG, "n_bootstrap", 0)
    monkeypatch.setattr(scaling_analysis, "_make_figures", lambda *args, **kwargs: None)

    scaling_analysis.analyze()

    summary = json.loads((root / "scaling_summary.json").read_text(encoding="utf-8"))
    assert summary["design"]["complete"] is True
    assert summary["n_fitted_param_points"] == 2
    assert summary["pooled_fit"] is not None


def _write_seed_run(root: Path, seed: int, value: float) -> None:
    run_dir = root / f"run_s{seed}"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(json.dumps({"test_ppl": value}), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps({
        "config": {"seed": seed, "embed_dim": 20},
    }), encoding="utf-8")
    (run_dir / "provenance.json").write_text(json.dumps({
        "seed": seed,
        "git_sha": "a" * 40,
        "git_dirty": False,
        "git_dirty_fingerprint": None,
        "train_data_sha256": "b" * 64,
        "train_data_n_tokens": 100,
        "val_data_sha256": "c" * 64,
        "val_data_n_tokens": 20,
        "test_data_sha256": "d" * 64,
        "test_data_n_tokens": 20,
        "data_seed": 3,
        "max_tokens": None,
        "tokenizer_tag": "synthetic",
    }), encoding="utf-8")


def _write_multiseed_manifest(
    root: Path,
    seeds: list[int],
    *,
    status: str = "complete",
    cell_statuses: dict[int, str] | None = None,
) -> None:
    statuses = cell_statuses or {seed: "complete" for seed in seeds}
    (root / "multiseed_request.json").write_text(json.dumps({
        "schema_version": 1,
        "status": status,
        "seeds": seeds,
        "cells": [{"seed": seed, "status": statuses[seed]} for seed in seeds],
    }), encoding="utf-8")


def _write_seed_artifacts(root: Path, seed: int, value: float) -> None:
    _write_seed_run(root, seed, value)
    run_dir = root / f"run_s{seed}"
    (run_dir / "metrics.csv").write_text("step,x\n1,1.0\n", encoding="utf-8")
    (run_dir / "metrics_per_layer.csv").write_text(
        "layer,self_coupling\n0,1.0\n", encoding="utf-8",
    )


def test_multiseed_analysis_exposes_missing_and_nonfinite_requested_seeds(tmp_path):
    _write_multiseed_manifest(tmp_path, [1, 2, 3])
    _write_seed_run(tmp_path, 1, 10.0)
    _write_seed_run(tmp_path, 2, float("inf"))

    result = multiseed_analysis.aggregate_seed_metric(tmp_path)

    assert result["n"] == 1 and result["values"] == [10.0]
    assert result["requested_seeds"] == [1, 2, 3]
    assert result["complete"] is False
    assert {cell["seed"]: cell["status"] for cell in result["cells"]} == {
        1: "complete",
        2: "nonfinite",
        3: "missing",
    }

    launch_root = tmp_path / "launches"
    calls = []

    def record_run(seed, logger, *, run_root=None):
        del logger
        calls.append((seed, run_root))

    with (
        mock.patch.object(train_vfe3, "RUN_ROOT", str(launch_root)),
        mock.patch.object(train_vfe3, "NUM_RUNS", 3),
        mock.patch.object(train_vfe3, "SEEDS", (1, 2, 3)),
        mock.patch.object(train_vfe3, "_run_once", record_run),
    ):
        train_vfe3.main()

    groups = [path for path in launch_root.iterdir() if path.is_dir()]
    assert len(groups) == 1
    assert not (launch_root / "multiseed_request.json").exists()
    assert json.loads((groups[0] / "multiseed_request.json").read_text()) == {
        "schema_version": 1,
        "status": "complete",
        "seeds": [1, 2, 3],
        "cells": [
            {"seed": 1, "status": "complete"},
            {"seed": 2, "status": "complete"},
            {"seed": 3, "status": "complete"},
        ],
    }
    assert calls == [(1, str(groups[0])), (2, str(groups[0])), (3, str(groups[0]))]


def test_multiseed_launch_persists_failed_cell_status(tmp_path):
    launch_root = tmp_path / "launches"

    def fail_second_run(seed, logger, *, run_root=None):
        del logger, run_root
        if seed == 2:
            raise RuntimeError("seed probe")

    with (
        mock.patch.object(train_vfe3, "RUN_ROOT", str(launch_root)),
        mock.patch.object(train_vfe3, "NUM_RUNS", 3),
        mock.patch.object(train_vfe3, "SEEDS", (1, 2, 3)),
        mock.patch.object(train_vfe3, "_run_once", fail_second_run),
        pytest.raises(RuntimeError, match="seed probe"),
    ):
        train_vfe3.main()

    groups = [path for path in launch_root.iterdir() if path.is_dir()]
    manifest = json.loads((groups[0] / "multiseed_request.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["cells"] == [
        {"seed": 1, "status": "complete"},
        {"seed": 2, "status": "failed", "error": "seed probe"},
        {"seed": 3, "status": "pending"},
    ]


def test_multiseed_launch_is_running_before_first_seed_starts(tmp_path):
    launch_root = tmp_path / "launches"
    observed = []

    def inspect_manifest(seed, logger, *, run_root=None):
        del logger
        observed.append((
            seed,
            json.loads((Path(run_root) / "multiseed_request.json").read_text(encoding="utf-8")),
        ))

    with (
        mock.patch.object(train_vfe3, "RUN_ROOT", str(launch_root)),
        mock.patch.object(train_vfe3, "NUM_RUNS", 2),
        mock.patch.object(train_vfe3, "SEEDS", (1, 2)),
        mock.patch.object(train_vfe3, "_run_once", inspect_manifest),
    ):
        train_vfe3.main()

    assert observed[0] == (1, {
        "schema_version": 1,
        "status": "running",
        "seeds": [1, 2],
        "cells": [
            {"seed": 1, "status": "pending"},
            {"seed": 2, "status": "pending"},
        ],
    })
    assert observed[1][0] == 2
    assert observed[1][1]["status"] == "running"
    assert observed[1][1]["cells"][0] == {"seed": 1, "status": "complete"}


@pytest.mark.parametrize("failure", ["missing", "failed", "unreadable", "nonfinite"])
def test_multiseed_withholds_curves_layers_and_main_publication_for_bad_requested_seed(
    tmp_path, monkeypatch, failure,
):
    top_status = "failed" if failure == "failed" else "complete"
    statuses = {1: "complete", 2: "failed" if failure == "failed" else "complete"}
    _write_multiseed_manifest(tmp_path, [1, 2], status=top_status, cell_statuses=statuses)
    _write_seed_artifacts(tmp_path, 1, 10.0)
    if failure == "failed":
        _write_seed_artifacts(tmp_path, 2, 11.0)  # stale survivor must not override the manifest
    elif failure == "unreadable":
        _write_seed_artifacts(tmp_path, 2, 11.0)
        run_dir = tmp_path / "run_s2"
        (run_dir / "summary.json").write_text("{ not json", encoding="utf-8")
        (run_dir / "metrics.csv").write_text("{ not csv", encoding="utf-8")
        (run_dir / "metrics_per_layer.csv").write_text("{ not csv", encoding="utf-8")
    elif failure == "nonfinite":
        _write_seed_artifacts(tmp_path, 2, float("inf"))
        run_dir = tmp_path / "run_s2"
        (run_dir / "metrics.csv").write_text("step,x\n1,inf\n", encoding="utf-8")
        (run_dir / "metrics_per_layer.csv").write_text(
            "layer,self_coupling\n0,inf\n", encoding="utf-8",
        )

    assert multiseed_analysis.aggregate_seed_curves(tmp_path, columns=["x"]) == {}
    assert multiseed_analysis.aggregate_per_layer(tmp_path) == {}

    emitted = []
    monkeypatch.setitem(multiseed_analysis.CONFIG, "run_root", str(tmp_path))
    monkeypatch.setitem(multiseed_analysis.CONFIG, "key", "test_ppl")
    monkeypatch.setattr(multiseed_analysis, "SCALAR_KEYS", ["test_ppl"])
    monkeypatch.setattr(multiseed_analysis, "_emit_figures", lambda *args: emitted.append(args))

    assert multiseed_analysis.main() != 0

    summary = json.loads((tmp_path / "multiseed_summary.json").read_text(encoding="utf-8"))
    assert summary["design"]["complete"] is False
    assert summary["scalars"] == {}
    assert summary["curves_final_step"] == {}
    assert summary["per_layer"] == {}
    assert summary["withheld"] == {
        "scalars": True,
        "curves": True,
        "per_layer": True,
        "figures": True,
    }
    assert emitted == []


def test_multiseed_main_returns_nonzero_for_empty_unrequested_design(tmp_path, monkeypatch):
    monkeypatch.setitem(multiseed_analysis.CONFIG, "run_root", str(tmp_path))

    assert multiseed_analysis.main() != 0


def _controlled_bank() -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(9)
    features = np.vstack([
        rng.normal(-1.0, 0.2, size=(80, 12)),
        rng.normal(+1.0, 0.2, size=(80, 12)),
    ])
    return {
        "mu": torch.tensor(features, dtype=torch.float32),
        "sigma": torch.ones((160, 12), dtype=torch.float32),
        "phi": torch.zeros((160, 2), dtype=torch.float32),
        "token_ids": torch.arange(160) % 8,
        "seq_idx": torch.arange(160) // 8,
        "pos_idx": torch.arange(160) % 8,
    }


class _ProjectionWorker:
    def embed(self, values, *, n_neighbors, min_dist, n_components, seed):
        del n_neighbors, min_dist
        return np.asarray(values, dtype=float)[:, :n_components] + seed * 1e-5


def test_controlled_figure_survives_sidecar_failure_and_reports_both_artifacts(tmp_path, monkeypatch):
    image_path = tmp_path / "belief.png"
    sidecar_path = tmp_path / "belief.json"
    sidecar_path.write_bytes(b"SENTINEL")
    monkeypatch.setattr(
        embedding_comparison,
        "cluster_coordinates",
        lambda values: (np.asarray(values, dtype=float)[:, :2], "PCA 2-D"),
    )
    monkeypatch.setattr(
        figures,
        "_cluster_embedding",
        lambda values, **kwargs: (np.arange(len(values)) % 2, "HDBSCAN(test double)"),
    )
    monkeypatch.setattr(
        embedding_comparison,
        "controlled_embedding_record",
        lambda **kwargs: {
            "sample": {"token_sha256": "0" * 64},
            "display": {"display_seed": embedding_comparison.CONTROLLED_SEEDS[0]},
        },
    )
    monkeypatch.setattr(
        embedding_comparison,
        "write_json_atomic",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("sidecar probe")),
    )

    figure = figures.plot_belief_umap(
        _controlled_bank(),
        "mu",
        controlled=True,
        english_linguistic_diagnostics=False,
        decode=lambda ids: f" token{int(ids[0])}",
        umap_worker=_ProjectionWorker(),
        path=str(image_path),
        sidecar_path=str(sidecar_path),
    )

    outcomes = figure._vfe3_publication_outcomes
    assert image_path.is_file()
    assert sidecar_path.read_bytes() == b"SENTINEL"
    assert outcomes["figure"]["published"] is True
    assert outcomes["sidecar"]["published"] is False
    assert "sidecar probe" in outcomes["sidecar"]["error"]
    figures.plt.close(figure)


def test_controlled_figure_rejects_figure_sidecar_alias_before_publication(tmp_path):
    destination = tmp_path / "controlled.json"
    destination.write_bytes(b"SENTINEL")

    with pytest.raises(ValueError, match="must not alias"):
        figures.plot_belief_umap(
            _controlled_bank(),
            "mu",
            controlled=True,
            english_linguistic_diagnostics=False,
            decode=lambda ids: f" token{int(ids[0])}",
            umap_worker=_ProjectionWorker(),
            path=str(destination),
            sidecar_path=str(destination),
        )

    assert destination.read_bytes() == b"SENTINEL"


def test_atomic_figure_save_preserves_suffixless_default_format(tmp_path):
    figure, axis = figures.plt.subplots()
    axis.plot([0, 1], [0, 1])
    try:
        figures._save(figure, str(tmp_path / "plot"))
    finally:
        figures.plt.close(figure)
    assert (tmp_path / "plot.png").is_file()


def test_cross_run_figure_survives_sidecar_failure_and_reports_both_artifacts(tmp_path, monkeypatch):
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    left.write_text("{}", encoding="utf-8")
    right.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(embedding_comparison, "comparison_summary", lambda *args, **kwargs: {})

    def write_figure(summary, *, path):
        del summary
        figure, axis = figures.plt.subplots()
        axis.plot([0, 1], [0, 1])
        figure.savefig(path)
        return figure

    monkeypatch.setattr(figures, "plot_controlled_embedding_comparison", write_figure)
    monkeypatch.setattr(
        embedding_comparison,
        "write_json_atomic",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("comparison sidecar probe")),
    )
    figure_path = tmp_path / "comparison.png"
    comparison_path = tmp_path / "comparison.json"
    comparison_path.write_bytes(b"SENTINEL")

    result = report.compare_belief_umap_sidecars(
        [left, right],
        ["left", "right"],
        json_path=comparison_path,
        figure_path=figure_path,
    )
    json_path, published_figure = result

    assert json_path is None
    assert comparison_path.read_bytes() == b"SENTINEL"
    assert published_figure == figure_path and figure_path.is_file()
    assert result.outcomes["figure"]["published"] is True
    assert result.outcomes["sidecar"]["published"] is False


@pytest.mark.parametrize("alias_kind", ("outputs", "json_input", "figure_input"))
def test_cross_run_comparison_rejects_destructive_path_aliases(tmp_path, monkeypatch, alias_kind):
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    left.write_text("{}", encoding="utf-8")
    right.write_text("{}", encoding="utf-8")
    output_json = tmp_path / "comparison.json"
    output_figure = tmp_path / "comparison.png"
    if alias_kind == "outputs":
        output_figure = output_json
    elif alias_kind == "json_input":
        output_json = left
    else:
        output_figure = right
    before = {left: left.read_bytes(), right: right.read_bytes()}
    monkeypatch.setattr(
        figures,
        "plot_controlled_embedding_comparison",
        lambda *args, **kwargs: pytest.fail("alias validation ran after rendering"),
    )

    with pytest.raises(ValueError, match="must not alias"):
        report.compare_belief_umap_sidecars(
            [left, right],
            ["left", "right"],
            json_path=output_json,
            figure_path=output_figure,
        )

    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("num_runs", [0, -1])
def test_multiseed_request_count_must_be_positive(num_runs):
    with pytest.raises(ValueError, match="NUM_RUNS.*positive"):
        train_vfe3._resolve_seeds({"seed": 6}, seeds=(6,), num_runs=num_runs)


def test_multiseed_request_seeds_must_be_unique():
    with pytest.raises(ValueError, match="unique"):
        train_vfe3._resolve_seeds({"seed": 6}, seeds=(6, 6), num_runs=2)


@pytest.mark.parametrize("bad_seed", [True, 1.5, "1", -1])
def test_multiseed_request_rejects_coercible_or_negative_seeds(bad_seed):
    with pytest.raises(ValueError, match="exact non-negative"):
        train_vfe3._resolve_seeds({"seed": 6}, seeds=(bad_seed,), num_runs=1)


@pytest.mark.parametrize("bad_seed", [True, 1.5, "1", -1])
def test_single_run_config_rejects_coercible_or_negative_seed(bad_seed):
    with pytest.raises(ValueError, match="config seed.*exact non-negative"):
        train_vfe3._resolve_seeds({"seed": bad_seed}, seeds=(), num_runs=1)


@pytest.mark.parametrize("bad_seed", [True, 1.5, "7", -1])
def test_train_loader_rejects_invalid_data_seed_before_cache_access(monkeypatch, bad_seed):
    monkeypatch.setattr(train_vfe3, "DATA_SEED", bad_seed)
    monkeypatch.setattr(
        train_vfe3,
        "make_dataloader",
        lambda *args, **kwargs: pytest.fail("invalid DATA_SEED reached cache loading"),
    )

    with pytest.raises(ValueError, match="DATA_SEED.*exact non-negative"):
        train_vfe3._select_loader("wikitext-103", _tiny_config(), split="train")


@pytest.mark.parametrize("bad_seed", [True, 1.5, "7", -1])
def test_ablation_rejects_invalid_data_seed_before_output_or_cache(
    tmp_path,
    monkeypatch,
    bad_seed,
):
    sweep_name = "invalid_data_seed"
    monkeypatch.setattr(ablation, "DATA_SEED", bad_seed)
    monkeypatch.setitem(ablation.SWEEPS, sweep_name, {"description": "seed validation"})

    with pytest.raises(ValueError, match="DATA_SEED.*exact non-negative"):
        ablation.run_sweep(
            sweep_name,
            tmp_path,
            dataset="wikitext-103",
            device=torch.device("cpu"),
            seed=6,
            resume=False,
        )

    assert not (tmp_path / sweep_name).exists()


@pytest.mark.parametrize("bad_seed", [True, 1.5, "7", -1])
def test_scaling_loader_rejects_invalid_data_seed_before_cache_access(monkeypatch, bad_seed):
    monkeypatch.setattr(
        scaling,
        "make_dataloader",
        lambda *args, **kwargs: pytest.fail("invalid data_seed reached cache loading"),
    )

    with pytest.raises(ValueError, match="data_seed.*exact nonnegative"):
        scaling.get_loader("wikitext-103", 4, 1, "train", data_seed=bad_seed)


def test_stochastic_generation_is_controlled_by_explicit_generation_seed(monkeypatch):
    monkeypatch.setattr(generate_efe, "_build_model", lambda *args, **kwargs: kwargs["policy_overrides"])
    monkeypatch.setattr(
        generate_efe,
        "_generate",
        lambda prompt_ids, model, cfg: torch.randint(0, 1000, (1, 8)),
    )
    cfg = dict(generate_efe.CONFIG, policy_mode="efe_one_step", generation_seed=314159)
    prompt = torch.tensor([[1]], dtype=torch.long)

    torch.manual_seed(1)
    first = generate_efe._run_generation_arms(prompt, {}, {}, cfg, device="cpu")
    torch.manual_seed(999)
    second = generate_efe._run_generation_arms(prompt, {}, {}, cfg, device="cpu")

    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


def test_generation_main_atomically_persists_generation_seed_and_outputs(tmp_path, monkeypatch):
    output = tmp_path / "efe_generation.json"
    checkpoint = tmp_path / "fake.pt"
    checkpoint.write_bytes(b"checkpoint-bytes")
    cfg = dict(generate_efe.CONFIG)
    cfg.update(checkpoint=str(checkpoint), dataset="synthetic", generation_seed=23,
               output_path=str(output), device="cpu")

    class _Tokenizer:
        def encode(self, value):
            del value
            return [1, 2]

        def decode(self, values):
            return " ".join(str(value) for value in values)

    monkeypatch.setattr(generate_efe, "CONFIG", cfg)
    monkeypatch.setattr(
        generate_efe, "_load_checkpoint",
        lambda config, **kwargs: ({"vocab_size": 8}, {}),
    )
    monkeypatch.setattr(generate_efe, "_generation_code_identity", lambda: "c" * 64)
    monkeypatch.setattr(
        generate_efe, "_runtime_identity",
        lambda device: {"device": {"type": str(device)}},
    )
    monkeypatch.setattr(generate_efe, "_tokenizer_for_dataset", lambda *args, **kwargs: _Tokenizer())
    monkeypatch.setattr(
        generate_efe,
        "_run_generation_arms",
        lambda *args, **kwargs: (torch.tensor([[1, 2, 3]]), torch.tensor([[1, 2, 4]])),
    )

    generate_efe.main()

    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["schema_version"] == 3
    assert record["checkpoint_sha256"] == hashlib.sha256(b"checkpoint-bytes").hexdigest()
    assert record["model_state_sha256"] == hashlib.sha256().hexdigest()
    assert record["generation_seed"] == 23
    assert record["code_identity_sha256"] == "c" * 64
    assert record["runtime_state"] == {"device": {"type": "cpu"}}
    assert record["generation_contract"] == {
        "dataset": "synthetic",
        "prompt": cfg["prompt"],
        "prompt_token_ids": [[1, 2]],
        "max_new_tokens": cfg["max_new_tokens"],
        "generation_seed": 23,
        "greedy": cfg["greedy"],
        "device": "cpu",
        "policy": {
            key: (list(cfg[key]) if key == "policy_score_terms" else cfg[key])
            for key in generate_efe._POLICY_FIELDS
        },
    }
    assert record["outputs"] == {"base_token_ids": [[1, 2, 3]], "policy_token_ids": [[1, 2, 4]]}
    assert not list(tmp_path.glob("*.tmp"))


def test_generation_checkpoint_digest_binds_the_exact_loaded_snapshot(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    original = {
        "config": {"vocab_size": 8},
        "model_state": {"weight": torch.tensor([1.0])},
    }
    replacement = {
        "config": {"vocab_size": 8},
        "model_state": {"weight": torch.tensor([9.0])},
    }
    torch.save(original, checkpoint)
    snapshot = checkpoint.read_bytes()
    digest = hashlib.sha256(snapshot).hexdigest()
    torch.save(replacement, checkpoint)

    config, state = generate_efe._load_checkpoint(
        {"checkpoint": str(checkpoint), "config_from": None},
        checkpoint_snapshot=snapshot,
    )

    assert config == original["config"]
    assert torch.equal(state["weight"], original["model_state"]["weight"])
    assert digest == hashlib.sha256(snapshot).hexdigest()
    assert not torch.equal(state["weight"], replacement["model_state"]["weight"])


@pytest.mark.parametrize("drift_source", ("checkpoint", "config_from"))
def test_generation_refuses_to_publish_if_an_input_changes_mid_run(
    tmp_path,
    monkeypatch,
    drift_source,
):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint-start")
    config_from = tmp_path / "config.json"
    config_from.write_bytes(b"config-start")
    output = tmp_path / "generation.json"
    output.write_bytes(b"SENTINEL")
    cfg = dict(
        generate_efe.CONFIG,
        checkpoint=str(checkpoint),
        config_from=str(config_from),
        output_path=str(output),
        dataset="synthetic",
        device="cpu",
    )

    class _Tokenizer:
        def encode(self, value):
            del value
            return [1]

        def decode(self, values):
            return str(values)

    def mutate_input(*args, **kwargs):
        del args, kwargs
        target = checkpoint if drift_source == "checkpoint" else config_from
        target.write_bytes(b"changed-during-generation")
        return torch.tensor([[1, 2]]), None

    monkeypatch.setattr(generate_efe, "CONFIG", cfg)
    monkeypatch.setattr(
        generate_efe, "_load_checkpoint",
        lambda config, **kwargs: ({"vocab_size": 8}, {}),
    )
    monkeypatch.setattr(generate_efe, "_tokenizer_for_dataset", lambda *a, **k: _Tokenizer())
    monkeypatch.setattr(generate_efe, "_run_generation_arms", mutate_input)

    with pytest.raises(RuntimeError, match="changed during generation"):
        generate_efe.main()

    assert output.read_bytes() == b"SENTINEL"


@pytest.mark.parametrize("alias_field", ("checkpoint", "config_from"))
def test_generation_output_cannot_alias_input_checkpoint(
    tmp_path,
    monkeypatch,
    alias_field,
):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"trained-model")
    cfg = dict(generate_efe.CONFIG, checkpoint=str(checkpoint), output_path=str(checkpoint))
    if alias_field == "config_from":
        other = tmp_path / "legacy.pt"
        other.write_bytes(b"legacy-state")
        cfg.update(checkpoint=str(other), config_from=str(checkpoint))
    monkeypatch.setattr(generate_efe, "CONFIG", cfg)
    monkeypatch.setattr(
        generate_efe,
        "_load_checkpoint",
        lambda config: pytest.fail("alias validation ran after checkpoint loading"),
    )

    with pytest.raises(ValueError, match="must not alias"):
        generate_efe.main()

    assert checkpoint.read_bytes() == b"trained-model"


def test_visualization_extra_declares_direct_networkx_dependency():
    import tomllib

    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    assert "networkx" in project["project"]["optional-dependencies"]["viz"]


_OPENMP_ENTRY_POINTS = (
    "train_vfe3",
    "scaling",
    "ablation",
    "make_figures",
    "scaling_analysis",
    "compare_vocab_figures",
)


def _openmp_probe(*, opt_in: bool) -> subprocess.CompletedProcess[str]:
    modules = repr(_OPENMP_ENTRY_POINTS)
    code = (
        "import importlib, os\n"
        "os.environ.pop('KMP_DUPLICATE_LIB_OK', None)\n"
        + ("os.environ['VFE3_ALLOW_DUPLICATE_OPENMP'] = '1'\n" if opt_in else
           "os.environ.pop('VFE3_ALLOW_DUPLICATE_OPENMP', None)\n")
        + f"[importlib.import_module(name) for name in {modules}]\n"
        "print('KMP_PROBE=' + repr(os.environ.get('KMP_DUPLICATE_LIB_OK')))\n"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def test_entry_points_do_not_mutate_openmp_compatibility_by_default():
    probe = _openmp_probe(opt_in=False)
    assert probe.returncode == 0, probe.stderr
    assert "KMP_PROBE=None" in probe.stdout


def test_entry_points_offer_explicit_openmp_compatibility_opt_in():
    probe = _openmp_probe(opt_in=True)
    assert probe.returncode == 0, probe.stderr
    assert "KMP_PROBE='TRUE'" in probe.stdout


def _fixture_scaling_sources() -> dict[str, dict[str, object]]:
    return {
        split: {
            "format": "pt",
            "tokenizer_tag": "synthetic",
            "size_bytes": len(split),
            "sha256": split[0] * 64,
            "meta": None,
            "meta_sha256": None,
        }
        for split in ("train", "validation", "test")
    }


def test_scaling_failure_is_persisted_and_returns_nonzero_without_success_banner(
    tmp_path, monkeypatch, capsys,
):
    cell = {"label": "probe", "route": "probe_route", "scale_knob": "embed_dim", "overrides": {}}
    monkeypatch.setattr(scaling, "ROUTES", {"probe_route": [cell]})
    monkeypatch.setitem(scaling.CONFIG, "routes", ["probe_route"])
    monkeypatch.setitem(scaling.CONFIG, "seeds", [6])
    monkeypatch.setitem(scaling.CONFIG, "device", "cpu")
    monkeypatch.setitem(scaling.CONFIG, "dataset", "synthetic")
    monkeypatch.setitem(scaling.CONFIG, "max_tokens", None)
    monkeypatch.setitem(scaling.CONFIG, "max_steps", None)
    monkeypatch.setitem(scaling.CONFIG, "output_dir", str(tmp_path))
    monkeypatch.setattr(scaling, "_cleanup", lambda: None)
    monkeypatch.setattr(scaling, "_data_source_identities",
                        lambda dataset: _fixture_scaling_sources())
    monkeypatch.setattr(scaling, "run_cell", lambda *args, **kwargs: {
        "label": "probe",
        "route": "probe_route",
        "scale_knob": "embed_dim",
        "seed": 6,
        "error_kind": "train",
        "error": "audit probe",
        "test_ce": None,
    })

    status = scaling.main()
    output = capsys.readouterr().out

    assert status == 1
    assert "ALL ROUTES COMPLETE" not in output
    failure = tmp_path / "probe_route" / "probe" / "s6" / "scaling_failure.json"
    assert json.loads(failure.read_text(encoding="utf-8"))["error"] == "audit probe"
    design = json.loads((tmp_path / "scaling_design.json").read_text(encoding="utf-8"))
    assert design["status"] == "incomplete"


def _complete_scaling_result() -> dict[str, object]:
    test_ce = 2.0
    return {
        "error_kind": None,
        "n_params": 10,
        "test_ce": test_ce,
        "test_ppl": math.exp(test_ce),
        "test_bits_per_token": test_ce / math.log(2.0),
        "test_bpc": None,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("n_params", True),
        ("n_params", 0),
        ("n_params", 10.0),
        ("test_ppl", None),
        ("test_ppl", float("nan")),
        ("test_ppl", 0.0),
        ("test_bits_per_token", None),
        ("test_bits_per_token", float("inf")),
        ("test_bits_per_token", 0.0),
        ("test_bpc", float("nan")),
        ("test_bpc", 0.0),
    ),
)
def test_scaling_completion_requires_a_complete_semantic_result(field, value):
    result = _complete_scaling_result()
    result[field] = value

    assert scaling._scaling_result_status(result) != "complete"


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("test_ppl", 2.0),
        ("test_bits_per_token", 2.0),
    ),
)
def test_scaling_completion_rejects_internally_inconsistent_metrics(field, value):
    result = _complete_scaling_result()
    result[field] = value

    assert scaling._scaling_result_status(result) != "complete"


def test_scaling_successful_rerun_removes_stale_failure_marker(tmp_path, monkeypatch):
    cell = {"label": "probe", "route": "probe_route", "scale_knob": "embed_dim", "overrides": {}}
    run_dir = tmp_path / "probe_route" / "probe" / "s6"
    run_dir.mkdir(parents=True)
    failure = run_dir / "scaling_failure.json"
    failure.write_text(json.dumps({"error": "old failure"}), encoding="utf-8")
    monkeypatch.setattr(scaling, "ROUTES", {"probe_route": [cell]})
    monkeypatch.setitem(scaling.CONFIG, "routes", ["probe_route"])
    monkeypatch.setitem(scaling.CONFIG, "seeds", [6])
    monkeypatch.setitem(scaling.CONFIG, "device", "cpu")
    monkeypatch.setitem(scaling.CONFIG, "dataset", "synthetic")
    monkeypatch.setitem(scaling.CONFIG, "max_tokens", None)
    monkeypatch.setitem(scaling.CONFIG, "max_steps", None)
    monkeypatch.setitem(scaling.CONFIG, "output_dir", str(tmp_path))
    monkeypatch.setattr(scaling, "_cleanup", lambda: None)
    monkeypatch.setattr(scaling, "_data_source_identities",
                        lambda dataset: _fixture_scaling_sources())
    monkeypatch.setattr(scaling, "run_cell", lambda *args, **kwargs: {
        **_complete_scaling_result(),
        "label": "probe",
        "route": "probe_route",
        "scale_knob": "embed_dim",
        "seed": 6,
        "cached": False,
    })

    assert scaling.main() == 0
    assert not failure.exists()


def test_scaling_rejects_negative_seed_values():
    with pytest.raises(ValueError, match="nonnegative"):
        scaling._validated_scaling_seeds([6, -1])


def test_scaling_rejects_integer_enum_seed_values():
    class Seed(IntEnum):
        ONE = 1

    with pytest.raises(ValueError, match="exact integers"):
        scaling._validated_scaling_seeds([Seed.ONE])


def test_every_ablation_arm_constructs_with_only_invalid_arm_prerequisites_repaired():
    assert ablation.BASELINE_CONFIG["e_step_update"] == "mm_exact"
    assert ablation.BASELINE_CONFIG["phi_precond_mode"] == "pullback_per_block"
    assert ablation.BASELINE_CONFIG["m_phi_update_mode"] == "adamw"
    assert ablation.BASELINE_CONFIG["m_phi_group_trust_radius"] == 0.1
    assert ablation.SWEEP_ORDER == ["pos_extrapolation", "estep_depth_damping"]

    transport = dict(ablation.make_run_overrides("transport_mode"))
    assert "e_step_update" not in transport["flat"]
    assert transport["regime_ii"]["e_step_update"] == "gradient"
    assert transport["regime_ii_covariant"]["e_step_update"] == "gradient"

    position = dict(ablation.make_run_overrides("pos_rotation"))
    assert "e_step_update" not in position["none"]
    assert position["rope"]["e_step_update"] == "gradient"

    covariance = dict(ablation.make_run_overrides("covariance"))
    assert "e_step_update" not in covariance["diagonal"]
    assert covariance["full"]["e_step_update"] == "gradient"

    renyi = dict(ablation.make_run_overrides("renyi_order"))
    assert "e_step_update" not in renyi["renyi_order=1.0"]
    assert all(
        renyi[f"renyi_order={value}"]["e_step_update"] == "gradient"
        for value in (0.5, 0.8, 1.2, 1.5, 2.0)
    )

    gauges = dict(ablation.make_run_overrides("gauge_group"))
    assert gauges["tied_block_glk"]["phi_precond_mode"] == "killing"
    assert gauges["so3_spin2x4"]["n_heads"] == 4
    cross = dict(ablation.make_run_overrides("cross_couplings"))["pair_0_1"]
    assert cross["beta_attention_prior"] == "causal_noself"
    assert cross["gamma_attention_prior"] == "causal_noself"

    gauge_optim = dict(ablation.make_run_overrides("gauge_mstep_optim"))
    assert set(gauge_optim) == {"adamw", "pullback_group"}
    assert gauge_optim["adamw"]["m_phi_update_mode"] == "adamw"
    pullback_group = gauge_optim["pullback_group"]
    assert pullback_group["m_phi_update_mode"] == "pullback_group"
    assert pullback_group["phi_precond_mode"] == "pullback_per_block"
    assert pullback_group["gauge_group"] == "block_glk"
    assert pullback_group["embed_dim"] == 10 and pullback_group["n_heads"] == 2
    assert pullback_group["e_phi_lr"] == 0.0
    assert pullback_group["transport_chart_max_norm"] == 6.0

    errors = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for sweep_name in ablation.SWEEPS:
            for label, overrides in ablation.make_run_overrides(sweep_name):
                config = dict(ablation.BASELINE_CONFIG)
                config.update(overrides)
                try:
                    VFE3Config(**config)
                except Exception as exc:
                    errors.append((sweep_name, label, str(exc)))
    assert errors == []


def test_every_scaling_arm_constructs_and_tied_groups_use_ambient_preconditioner():
    assert scaling.BASELINE["phi_precond_mode"] == "pullback_per_block"
    assert scaling.BASELINE["m_phi_update_mode"] == "adamw"
    assert scaling.BASELINE["m_phi_group_trust_radius"] == 0.1
    tied_cells = scaling.ROUTES["blocks_K48_tied_2x"]
    assert len(tied_cells) == 5
    assert all(cell["overrides"]["phi_precond_mode"] == "killing" for cell in tied_cells)
    group_tied = next(cell for cell in scaling.ROUTES["group"] if cell["label"] == "K64_tied_h8")
    assert group_tied["overrides"]["phi_precond_mode"] == "killing"

    errors = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for route_name, cells in scaling.ROUTES.items():
            for cell in cells:
                try:
                    VFE3Config(**scaling._cell_cfg_dict(cell["overrides"], 6, None))
                except Exception as exc:
                    errors.append((route_name, cell["label"], str(exc)))
    assert errors == []


def test_click_run_phi_controls_use_current_update_policy():
    assert train_vfe3.config["m_phi_update_mode"] == "adamw"
    assert train_vfe3.config["m_phi_group_trust_radius"] == 0.1
