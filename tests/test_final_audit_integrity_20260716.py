"""Final independent-review regressions for artifact identity and exact resume."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

import generate_efe
import multiseed_analysis
import scaling_analysis
from vfe3.config import VFE3Config
from vfe3.gauge_optim import GaugeManifoldAdamW
from vfe3.geometry.groups import get_group
from vfe3.run_artifacts import RunArtifacts, load_checkpoint


def _sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _code_identity() -> dict[str, object]:
    return {
        "git_sha": "a" * 40,
        "git_dirty": False,
        "git_dirty_fingerprint": None,
    }


def _source_identities() -> dict[str, dict[str, object]]:
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


def _write_bound_scaling_run(
    root: Path,

    *,
    route:             str                                       = "route",
    label:             str                                       = "small",
    seed:              int                                       = 1,
    run_path:          Path | None                               = None,
    cell_seed:         int | None                                = None,
    config_seed:       int | None                                = None,
    prov_seed:         int | None                                = None,
    scale_knob:        str                                       = "embed_dim",
    code_identity:     dict[str, object] | None                  = None,
    source_identities: dict[str, dict[str, object]] | None       = None,
) -> Path:
    run = root / route / label / f"s{seed}" if run_path is None else run_path
    run.mkdir(parents=True)
    cell_seed = seed if cell_seed is None else cell_seed
    config_seed = seed if config_seed is None else config_seed
    prov_seed = seed if prov_seed is None else prov_seed
    config = {
        "seed": config_seed,
        "embed_dim": 10,
        "n_heads": 2,
        "n_layers": 1,
        "n_e_steps": 1,
        "family": "gaussian_diagonal",
    }
    code_identity = _code_identity() if code_identity is None else code_identity
    source_identities = (
        _source_identities() if source_identities is None else source_identities)
    cell = {
        "schema_version": 2,
        "route": route,
        "label": label,
        "scale_knob": scale_knob,
        "overrides": {},
        "predicted_n_params": 10,
        "n_gen": 4,
        "seed": cell_seed,
        "max_tokens": None,
        "dataset": "synthetic",
        "config_sha256": _sha256(config),
        "code_identity": code_identity,
        "data_sources": source_identities,
    }
    digest = _sha256(cell)
    cell["reuse_contract_sha256"] = digest
    test_ce = 2.0
    metrics = {
        "n_params": 10,
        "test_ce": test_ce,
        "test_ppl": math.exp(test_ce),
        "test_bits_per_token": test_ce / math.log(2.0),
        "test_bpc": None,
    }
    summary = {
        **metrics,
        "scaling_point": {
            **metrics,
            "embed_dim": 10,
            "n_heads": 2,
            "n_layers": 1,
            "n_e_steps": 1,
            "n_gen": 4,
            "gauge_group": "block_glk",
            "family": "gaussian_diagonal",
            "tokens_seen": 100,
        },
        "scaling_reuse_contract_sha256": digest,
    }
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run / "test_results.json").write_text(json.dumps(metrics), encoding="utf-8")
    (run / "config.json").write_text(json.dumps({
        "dataset": "synthetic",
        "config": config,
    }), encoding="utf-8")
    (run / "provenance.json").write_text(json.dumps({
        "seed": prov_seed,
        **code_identity,
    }), encoding="utf-8")
    (run / "scaling_cell.json").write_text(json.dumps(cell), encoding="utf-8")
    return run


def _write_scaling_design(
    root: Path,

    *,
    run_dir:     str = "route/small/s1",
    scale_knob:  str = "embed_dim",
) -> None:
    (root / "scaling_design.json").write_text(json.dumps({
        "schema_version": 1,
        "routes": ["route"],
        "seeds": [1],
        "status": "complete",
        "cells": [{
            "route": "route",
            "label": "small",
            "seed": 1,
            "scale_knob": scale_knob,
            "run_dir": run_dir,
            "status": "complete",
        }],
    }), encoding="utf-8")


def _write_two_seed_scaling_design(root: Path) -> None:
    (root / "scaling_design.json").write_text(json.dumps({
        "schema_version": 1,
        "routes": ["route"],
        "seeds": [1, 2],
        "status": "complete",
        "cells": [
            {
                "route": "route",
                "label": "small",
                "seed": seed,
                "scale_knob": "embed_dim",
                "run_dir": f"route/small/s{seed}",
                "status": "complete",
            }
            for seed in (1, 2)
        ],
    }), encoding="utf-8")


def test_scaling_join_rejects_conflicting_seed_identities(tmp_path: Path) -> None:
    _write_bound_scaling_run(tmp_path, cell_seed=2, config_seed=3, prov_seed=1)
    _write_scaling_design(tmp_path)

    design = scaling_analysis._requested_design(
        tmp_path, scaling_analysis.harvest(tmp_path))

    assert design["complete"] is False
    assert design["cells"][0]["status"] != "complete"


def test_scaling_join_requires_declared_run_directory(tmp_path: Path) -> None:
    _write_bound_scaling_run(tmp_path, run_path=tmp_path / "rogue")
    _write_scaling_design(tmp_path)

    design = scaling_analysis._requested_design(
        tmp_path, scaling_analysis.harvest(tmp_path))

    assert design["complete"] is False
    assert design["cells"][0]["status"] == "missing"


@pytest.mark.parametrize("run_dir", ["../rogue", "/absolute", "route\\small\\s1", "."])
def test_scaling_join_rejects_unsafe_declared_run_directory(
    tmp_path: Path,
    run_dir:  str,
) -> None:
    _write_bound_scaling_run(tmp_path)
    _write_scaling_design(tmp_path, run_dir=run_dir)

    design = scaling_analysis._requested_design(
        tmp_path, scaling_analysis.harvest(tmp_path))

    assert design["complete"] is False
    assert design["cells"][0]["status"] == "unreadable"


def test_scaling_harvest_rejects_unbound_or_inconsistent_metrics(tmp_path: Path) -> None:
    run = _write_bound_scaling_run(tmp_path)
    summary_path = run / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["test_ce"] = 999.0
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    _write_scaling_design(tmp_path)

    assert scaling_analysis.harvest(tmp_path) == []
    assert scaling_analysis._requested_design(tmp_path, [])["complete"] is False


def test_scaling_join_requires_manifest_scale_knob_to_match_cell(tmp_path: Path) -> None:
    _write_bound_scaling_run(tmp_path, scale_knob="n_heads")
    _write_scaling_design(tmp_path, scale_knob="embed_dim")

    design = scaling_analysis._requested_design(
        tmp_path, scaling_analysis.harvest(tmp_path))

    assert design["complete"] is False
    assert design["cells"][0]["status"] != "complete"


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("n_params", 11),
        ("scale_knob", "n_heads"),
        ("embed_dim", 12),
        ("n_heads", 3),
        ("n_layers", 2),
        ("n_e_steps", 2),
        ("n_gen", 5),
        ("gauge_group", "so_k"),
        ("family", "laplace_diagonal"),
        ("tokens_seen", 101),
    ],
)
def test_scaling_aggregation_withholds_structurally_mixed_seed_cells(
    field:       str,
    replacement: object,
) -> None:
    row = {
        "route": "route",
        "label": "small",
        "seed": 1,
        "scale_knob": "embed_dim",
        "n_params": 10,
        "embed_dim": 10,
        "n_heads": 2,
        "n_layers": 1,
        "n_e_steps": 1,
        "n_gen": 4,
        "gauge_group": "block_glk",
        "family": "gaussian_diagonal",
        "tokens_seen": 100,
        "test_ce": 2.0,
    }
    other = dict(row, seed=2)
    other[field] = replacement

    assert scaling_analysis.aggregate_points([row, other]) == []


def test_scaling_analysis_withholds_seed_runs_from_different_code(tmp_path: Path) -> None:
    _write_bound_scaling_run(tmp_path, seed=1)
    _write_bound_scaling_run(tmp_path, seed=2, code_identity={
        "git_sha": "b" * 40,
        "git_dirty": False,
        "git_dirty_fingerprint": None,
    })
    _write_two_seed_scaling_design(tmp_path)

    rows = scaling_analysis.harvest(tmp_path)
    design = scaling_analysis._requested_design(tmp_path, rows)

    assert len(rows) == 2
    assert design["complete"] is False
    assert {cell["status"] for cell in design["cells"]} == {"inconsistent"}
    assert scaling_analysis.aggregate_points(rows) == []


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_scaling_analysis_withholds_seed_runs_from_different_corpus(
    tmp_path: Path,
    split:    str,
) -> None:
    _write_bound_scaling_run(tmp_path, seed=1)
    changed_sources = _source_identities()
    changed_sources[split]["sha256"] = "e" * 64
    _write_bound_scaling_run(tmp_path, seed=2, source_identities=changed_sources)
    _write_two_seed_scaling_design(tmp_path)

    rows = scaling_analysis.harvest(tmp_path)
    design = scaling_analysis._requested_design(tmp_path, rows)

    assert len(rows) == 2
    assert design["complete"] is False
    assert {cell["status"] for cell in design["cells"]} == {"inconsistent"}
    assert scaling_analysis.aggregate_points(rows) == []


def _multiseed_provenance(seed: int) -> dict[str, object]:
    return {
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
    }


def _write_multiseed_run(
    root: Path,
    seed: int,

    *,
    provenance: dict[str, object] | None = None,
    generate_figures: bool = False,
) -> Path:
    run = root / f"run_s{seed}"
    run.mkdir()
    (run / "summary.json").write_text(json.dumps({"test_ppl": 10.0 + seed}), encoding="utf-8")
    (run / "config.json").write_text(json.dumps({
        "config": {
            "seed": seed,
            "embed_dim": 20,
            "generate_figures": generate_figures,
        },
    }), encoding="utf-8")
    (run / "provenance.json").write_text(json.dumps(
        _multiseed_provenance(seed) if provenance is None else provenance
    ), encoding="utf-8")
    (run / "metrics.csv").write_text("step,train_ce\n1,2.0\n", encoding="utf-8")
    return run


def _write_multiseed_request(root: Path) -> None:
    (root / "multiseed_request.json").write_text(json.dumps({
        "schema_version": 1,
        "status": "complete",
        "seeds": [1, 2],
        "cells": [
            {"seed": 1, "status": "complete"},
            {"seed": 2, "status": "complete"},
        ],
    }), encoding="utf-8")


@pytest.mark.parametrize(
    ("config_seed", "directory_seed"),
    [(2, 1), (1, 2)],
)
def test_multiseed_seed_join_rejects_conflicting_present_identities(
    tmp_path:      Path,
    config_seed:   int,
    directory_seed: int,
) -> None:
    run = tmp_path / f"run_s{directory_seed}"
    run.mkdir()
    (run / "provenance.json").write_text(json.dumps({"seed": 1}), encoding="utf-8")
    (run / "config.json").write_text(json.dumps({
        "config": {"seed": config_seed},
    }), encoding="utf-8")

    assert multiseed_analysis._seed_for(run) is None


@pytest.mark.parametrize("provenance_case", ["missing", "mixed"])
def test_multiseed_design_marks_unverifiable_provenance_incomplete(
    tmp_path:       Path,
    provenance_case: str,
) -> None:
    _write_multiseed_run(tmp_path, 1)
    second = _multiseed_provenance(2)
    if provenance_case == "missing":
        second = {"seed": 2}
    else:
        second["git_sha"] = "e" * 40
        second["train_data_sha256"] = "f" * 64
    _write_multiseed_run(tmp_path, 2, provenance=second)
    _write_multiseed_request(tmp_path)

    design = multiseed_analysis._requested_seed_design(tmp_path)

    assert design["complete"] is False
    assert any(cell["status"] == "unverifiable" for cell in design["cells"])


@pytest.mark.parametrize("root_json", ["[]", "null", "1", '"manifest"'])
def test_multiseed_nonobject_manifest_is_unverifiable_not_an_exception(
    tmp_path:  Path,
    root_json: str,
) -> None:
    (tmp_path / "multiseed_request.json").write_text(root_json, encoding="utf-8")

    manifest = multiseed_analysis._request_manifest(tmp_path, [1])

    assert manifest["request_verified"] is False
    assert manifest["manifest_status"] == "unverifiable"


def test_multiseed_main_derives_missing_per_layer_intent_from_config(
    tmp_path:    Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_multiseed_run(tmp_path, 1, generate_figures=True)
    _write_multiseed_run(tmp_path, 2, generate_figures=True)
    _write_multiseed_request(tmp_path)
    monkeypatch.setitem(multiseed_analysis.CONFIG, "run_root", str(tmp_path))
    monkeypatch.setitem(multiseed_analysis.CONFIG, "key", "test_ppl")
    monkeypatch.setattr(multiseed_analysis, "SCALAR_KEYS", ["test_ppl"])
    monkeypatch.setattr(multiseed_analysis, "_emit_figures", lambda *args: None)

    multiseed_analysis.main()

    summary = json.loads((tmp_path / "multiseed_summary.json").read_text(encoding="utf-8"))
    assert summary["diagnostics"]["per_layer_requested"] is True
    assert summary["diagnostics"]["per_layer_complete"] is False
    assert summary["withheld"]["per_layer"] is True


def test_generation_rejects_unknown_checkpoint_config_fields(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    config = asdict(VFE3Config())
    config["future_behavior_knob"] = "nondefault"
    torch.save({
        "config": config,
        "model_state": {"weight": torch.tensor([1.0])},
    }, checkpoint)

    with pytest.raises(ValueError, match="unknown.*future_behavior_knob"):
        generate_efe._load_checkpoint({
            "checkpoint": str(checkpoint),
            "config_from": None,
        })


def test_generation_verifies_raw_legacy_fingerprint_before_config_migration(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "legacy-checkpoint.pt"
    raw_config = asdict(VFE3Config())
    raw_config.pop("m_phi_update_mode")
    raw_config["m_phi_natural_grad"] = False
    torch.save({
        "config": raw_config,
        "config_fingerprint": generate_efe.semantic_config_fingerprint(raw_config),
        "model_state": {"weight": torch.tensor([1.0])},
    }, checkpoint)

    migrated, state = generate_efe._load_checkpoint({
        "checkpoint": str(checkpoint),
        "config_from": None,
    })

    assert migrated["m_phi_update_mode"] == "adamw"
    assert "m_phi_natural_grad" not in migrated
    assert torch.equal(state["weight"], torch.tensor([1.0]))

    corrupt = torch.load(checkpoint, weights_only=True)
    corrupt["config_fingerprint"] = "0" * 64
    torch.save(corrupt, checkpoint)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        generate_efe._load_checkpoint({
            "checkpoint": str(checkpoint),
            "config_from": None,
        })


def test_visualization_best_model_verifies_raw_legacy_fingerprint_before_migration(
    tmp_path: Path,
) -> None:
    from vfe3.viz.run_loading import load_best_model_state

    cfg = VFE3Config(vocab_size=8, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=1)
    model = torch.nn.Linear(4, 4)
    raw_config = asdict(cfg)
    raw_config.pop("m_phi_update_mode")
    raw_config["m_phi_natural_grad"] = False
    checkpoint = tmp_path / "best_model.pt"
    payload = {
        "model_state": model.state_dict(),
        "config": raw_config,
        "config_fingerprint": generate_efe.semantic_config_fingerprint(raw_config),
    }
    torch.save(payload, checkpoint)

    loaded = load_best_model_state(checkpoint, cfg, map_location="cpu")
    assert set(loaded) == set(model.state_dict())

    payload["config_fingerprint"] = "f" * 64
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_best_model_state(checkpoint, cfg, map_location="cpu")


def test_pullback_group_artifact_reports_fixed_factor_norm_route() -> None:
    from vfe3.model.model import VFEModel
    from vfe3.run_artifacts import _phi_chart_norm_route

    adamw_cfg = VFE3Config(vocab_size=8, embed_dim=4, n_heads=2)
    pullback_cfg = VFE3Config(
        vocab_size=8,
        embed_dim=4,
        n_heads=2,
        gauge_group="block_glk",
        m_phi_update_mode="pullback_group",
        phi_precond_mode="pullback_per_block",
        transport_chart_max_norm=6.0,
        phi_mstep_max_matrix_norm=None,
    )
    bounded_pullback_cfg = VFE3Config(
        vocab_size=8,
        embed_dim=4,
        n_heads=2,
        gauge_group="block_glk",
        m_phi_update_mode="pullback_group",
        phi_precond_mode="pullback_per_block",
        transport_chart_max_norm=4.0,
        phi_mstep_max_matrix_norm=3.0,
    )

    assert _phi_chart_norm_route(VFEModel(adamw_cfg), adamw_cfg) is None
    assert (
        _phi_chart_norm_route(VFEModel(pullback_cfg), pullback_cfg)
        == "diagonal_gram:factor_radius=5.0"
    )
    assert (
        _phi_chart_norm_route(VFEModel(bounded_pullback_cfg), bounded_pullback_cfg)
        == "diagonal_gram:factor_radius=3.0"
    )


def test_retired_phi_optimizer_core_is_confined_to_migration_fixtures() -> None:
    root = Path(__file__).parents[1]
    python_files = [
        *root.joinpath("vfe3").rglob("*.py"),
        *root.joinpath("tests").rglob("*.py"),
        root / "train_vfe3.py",
        root / "ablation.py",
        root / "scaling.py",
    ]
    source = {path.relative_to(root).as_posix(): path.read_text(encoding="utf-8") for path in python_files}
    old_class = "GaugeNatural" + "GradAdamW"
    assert all(old_class not in text for text in source.values())

    production = "\n".join(
        source[name]
        for name in ("vfe3/gauge_optim.py", "vfe3/train.py", "vfe3/run_artifacts.py")
    )
    retired_moment = re.compile(
        r"gauge_" + r"mom|gauge_" + r"m\b|gauge_" + r"v\b|gauge_" + r"step"
    )
    assert retired_moment.search(production) is None

    retired_mstep_claim = re.compile(
        r"natural[-_ ]*gradient.{0,80}m[-_ ]*step"
        r"|m[-_ ]*step.{0,80}natural[-_ ]*gradient",
        re.IGNORECASE,
    )
    assert all(retired_mstep_claim.search(text) is None for text in source.values())

    retired_fields = (
        "m_phi_" + "natural_grad",
        "m_gauge_" + "momentum",
        "m_gauge_" + "update_rule",
    )
    allowed = {
        "vfe3/config.py",
        "tests/test_config.py",
        "tests/test_checkpoint_resume.py",
        "tests/test_run_artifacts.py",
        "tests/test_final_audit_integrity_20260716.py",
        "tests/test_2026_07_15_cache_serialization_remediation.py",
    }
    hits = {
        name
        for name, text in source.items()
        if any(field in text for field in retired_fields)
    }
    assert hits <= allowed


def test_checkpoint_payload_is_loaded_on_cpu_before_device_restore(
    tmp_path:    Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"placeholder")
    model = torch.nn.Linear(1, 1)
    observed: list[object] = []

    def fake_load(*args, **kwargs):
        del args
        observed.append(kwargs.get("map_location"))
        return {}

    monkeypatch.setattr("vfe3.run_artifacts.torch.load", fake_load)
    with pytest.raises(ValueError, match="step"):
        load_checkpoint(checkpoint, model, map_location=torch.device("cuda"))

    assert observed == ["cpu"]


@pytest.mark.cuda
def test_cuda_custom_optimizer_resume_preserves_cpu_control_state(tmp_path: Path) -> None:
    """CPU-only clocks/RNG stay on CPU while parameter slots restore onto the CUDA model."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    cfg = VFE3Config()
    source = torch.nn.Linear(2, 2)
    source_group = get_group("glk")(K=2)
    source_optimizer = GaugeManifoldAdamW(
        source.parameters(),
        source_group,
        phi_group_trust_radius=0.1,
        phi_chart_max_norm=5.0,
        phi_bch_residual_max=1e-6,
        phi_precond_mode="pullback",
        lr=0.01,
    )
    source(torch.ones(1, 2)).sum().backward()
    source_optimizer.step()
    checkpoint = RunArtifacts(tmp_path / "source", cfg, source).save_checkpoint(
        1,
        source,
        source_optimizer,
        cfg,
    )

    target = torch.nn.Linear(2, 2).cuda()
    target_group = get_group("glk")(K=2, device="cuda")
    target_optimizer = GaugeManifoldAdamW(
        target.parameters(),
        target_group,
        phi_group_trust_radius=0.1,
        phi_chart_max_norm=5.0,
        phi_bch_residual_max=1e-6,
        phi_precond_mode="pullback",
        lr=0.01,
    )
    assert load_checkpoint(
        checkpoint,
        target,
        target_optimizer,
        map_location=torch.device("cuda"),
        cfg=cfg,
    ) == 1

    state = next(iter(target_optimizer.state.values()))
    assert state["step"].device.type == "cpu"
    assert state["exp_avg"].device.type == "cuda"
    assert state["exp_avg_sq"].device.type == "cuda"
