"""Regression tests for invocation-scoped corpus identity snapshots."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import torch

import ablation
import scaling


_CODE_IDENTITY = {
    "git_sha": "a" * 40,
    "git_dirty": False,
    "git_dirty_fingerprint": None,
}


def _source_identities(*splits: str) -> dict[str, dict[str, object]]:
    return {
        split: {
            "format": "pt",
            "tokenizer_tag": "tiktoken",
            "size_bytes": len(split),
            "sha256": (split[0] * 64),
            "meta": None,
            "meta_sha256": None,
        }
        for split in splits
    }


def _ablation_success(label: str, sources: object, run_dir: Path) -> dict[str, object]:
    checkpoint = run_dir / "checkpoints" / "terminal.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(f"checkpoint:{label}".encode("utf-8"))
    return {
        "label": label,
        "error_kind": None,
        "primary_val_ppl": 8.0,
        "final_val_ppl": 9.0,
        "seed": 6,
        "terminal_checkpoint": str(checkpoint),
        "_loaded_data_sources": copy.deepcopy(sources),
    }


def test_ablation_code_identity_preflight_publishes_incomplete_view_without_cell_execution(
    tmp_path, monkeypatch,
):
    output_dir = tmp_path / "ablation"
    sweep_name = "code_identity_preflight"
    monkeypatch.setitem(ablation.SWEEPS, sweep_name, {"description": "code preflight"})
    monkeypatch.setattr(ablation, "make_run_overrides", lambda _name: [("cell", {})])
    monkeypatch.setattr(
        ablation,
        "_git_code_identity",
        lambda: {
            "git_sha": None,
            "git_dirty": None,
            "git_dirty_fingerprint": None,
            "git_error": "probe failed",
        },
    )
    monkeypatch.setattr(
        ablation,
        "cache_source_identity",
        lambda dataset, split, cache_dir=None: _source_identities(split)[split],
    )
    executed: list[object] = []
    monkeypatch.setattr(ablation, "run_single", lambda *args, **kwargs: executed.append(None))

    result = ablation.run_sweep(
        sweep_name,
        output_dir,
        dataset="wikitext-103",
        device=torch.device("cpu"),
        seed=6,
        resume=False,
    )

    assert result == []
    assert executed == []
    sweep_dir = output_dir / sweep_name
    meta = json.loads((sweep_dir / "sweep_meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "incomplete"
    assert "code identity" in meta["error"]
    assert (sweep_dir / "sweep_results.csv").read_text(encoding="utf-8").count("\n") == 1


def test_ablation_main_skips_all_analysis_for_incomplete_sweep(
    tmp_path, monkeypatch,
):
    sweep_name = "incomplete_analysis_guard"
    monkeypatch.setitem(ablation.SWEEPS, sweep_name, {"description": "incomplete guard"})
    monkeypatch.setitem(ablation.CONFIG, "output_dir", str(tmp_path))
    monkeypatch.setitem(ablation.CONFIG, "device", "cpu")
    monkeypatch.setitem(ablation.CONFIG, "dataset", "wikitext-103")
    monkeypatch.setitem(ablation.CONFIG, "resume", False)
    monkeypatch.setitem(ablation.CONFIG, "seed", 6)
    monkeypatch.setitem(ablation.CONFIG, "max_tokens", None)
    monkeypatch.setitem(ablation.CONFIG, "max_steps", None)
    monkeypatch.setitem(ablation.CONFIG, "list_only", False)
    monkeypatch.setitem(ablation.CONFIG, "sweep", sweep_name)
    monkeypatch.setattr(ablation, "validate_sweeps", lambda names: None)

    def incomplete_run(name, output_dir, **kwargs):
        del kwargs
        sweep_dir = output_dir / name
        sweep_dir.mkdir(parents=True)
        (sweep_dir / "sweep_meta.json").write_text(
            json.dumps({"status": "incomplete", "error": "code identity drift"}),
            encoding="utf-8",
        )
        return []

    monkeypatch.setattr(ablation, "run_sweep", incomplete_run)
    called: list[str] = []
    for name in (
        "emit_registered_figures", "analyze_sweep", "_plot_one_sweep",
        "_plot_seed_aggregate", "_plot_rank_collapse", "_plot_attention_entropy",
        "_plot_wallclock_convergence", "_plot_gauge_transport", "_plot_cg_coupling",
        "_plot_kappa_dispersion", "_plot_gauge_residual_drift", "_plot_pos_extrapolation",
        "_plot_renyi_saturation", "_plot_mu_precond", "_plot_holonomy_trainability",
        "_plot_sensitivity", "summarize_sweeps",
    ):
        monkeypatch.setattr(
            ablation,
            name,
            lambda *args, _name=name, **kwargs: called.append(_name),
        )
    figure_requests = []
    monkeypatch.setattr(
        ablation,
        "_run_ablation_figures_isolated",
        lambda _output_dir, *, scope, invalidate=False, cohort_identity=None: (
            figure_requests.append((scope, invalidate, cohort_identity)) or True
        ),
    )

    ablation.main()

    assert called == []
    assert figure_requests == [
        ("__sensitivity__", True, None),
        (sweep_name, True, None),
    ]


def test_ablation_cross_sweep_reports_ignore_incomplete_directories(
    tmp_path, monkeypatch,
):
    contract = ablation._sweep_aggregation_contract(
        "wikitext-103",
        {
            "collect_diagnostics": False,
            "collect_extrapolation": False,
            "paired_token_bootstrap": False,
        },
        data_seed_override=3,
        max_tokens=None,
        max_steps=None,
        seed_design=[6],
        source_identities=_source_identities("train", "validation"),
        code_identity=_CODE_IDENTITY,
        device="cpu",
    )
    cohort = ablation._cross_sweep_cohort_identity(contract)
    for name, status in (("complete", "complete"), ("incomplete", "incomplete")):
        sweep_dir = tmp_path / name
        sweep_dir.mkdir()
        (sweep_dir / "sweep_results.csv").write_text("label,primary_val_ppl\n", encoding="utf-8")
        (sweep_dir / "sweep_meta.json").write_text(
            json.dumps({"status": status, "aggregation_contract": contract}),
            encoding="utf-8",
        )

    visited: list[str] = []
    monkeypatch.setattr(
        ablation,
        "_read_sweep_csv",
        lambda sweep_dir: visited.append(sweep_dir.name) or [],
    )

    ablation._plot_sensitivity(
        tmp_path,
        tmp_path / "figures",
        cohort_identity=cohort,
    )
    ablation.summarize_sweeps(tmp_path, cohort_identity=cohort)

    assert visited == ["complete", "complete"]


def test_ablation_hashes_each_source_once_and_reuses_snapshot_across_cells(
    tmp_path, monkeypatch,
):
    sources = _source_identities("train", "validation")
    calls: list[str] = []
    code_calls: list[object] = []

    def source_identity(dataset, split, *, cache_dir=None):
        del dataset, cache_dir
        calls.append(split)
        return copy.deepcopy(sources[split])

    sweep_name = "source_snapshot_call_count"
    monkeypatch.setitem(ablation.SWEEPS, sweep_name, {"description": "source snapshot"})
    monkeypatch.setattr(
        ablation,
        "make_run_overrides",
        lambda _name: [("first", {}), ("second", {})],
    )
    monkeypatch.setattr(ablation, "cache_source_identity", source_identity)
    def code_identity():
        code_calls.append(None)
        return dict(_CODE_IDENTITY)

    monkeypatch.setattr(ablation, "_git_code_identity", code_identity)
    monkeypatch.setattr(ablation, "_cleanup", lambda: None)
    monkeypatch.setattr(
        ablation,
        "run_single",
        lambda label, _overrides, run_dir, **kwargs: _ablation_success(
            label,
            sources,
            run_dir,
        ),
    )

    ablation.run_sweep(
        sweep_name,
        tmp_path,
        dataset="wikitext-103",
        device=torch.device("cpu"),
        seed=6,
        resume=False,
    )

    assert calls == ["train", "validation"]
    assert len(code_calls) == 2
    for label in ("first", "second"):
        run_dir = tmp_path / sweep_name / ablation._sanitize(label)
        marker = json.loads((run_dir / "ablation_result.json").read_text(encoding="utf-8"))
        contract = json.loads((run_dir / "cell_contract.json").read_text(encoding="utf-8"))
        assert marker["status"] == "success"
        assert contract["train_source"] == sources["train"]
        assert contract["validation_source"] == sources["validation"]


def test_ablation_rejects_unavailable_or_mismatched_loaded_source_identity(
    tmp_path, monkeypatch,
):
    sources = _source_identities("train", "validation")
    calls: list[str] = []

    def source_identity(dataset, split, *, cache_dir=None):
        del dataset, cache_dir
        calls.append(split)
        return copy.deepcopy(sources[split])

    sweep_name = "source_snapshot_drift"
    monkeypatch.setitem(ablation.SWEEPS, sweep_name, {"description": "source drift"})
    monkeypatch.setattr(
        ablation,
        "make_run_overrides",
        lambda _name: [("matching", {}), ("drifted", {}), ("missing", {})],
    )
    monkeypatch.setattr(ablation, "cache_source_identity", source_identity)
    monkeypatch.setattr(ablation, "_git_code_identity", lambda: dict(_CODE_IDENTITY))
    monkeypatch.setattr(ablation, "_cleanup", lambda: None)

    def run_single(label, _overrides, run_dir, **kwargs):
        del kwargs
        loaded = copy.deepcopy(sources)
        if label == "drifted":
            loaded["train"]["sha256"] = "f" * 64
        if label == "missing":
            loaded = None
        return _ablation_success(label, loaded, run_dir)

    monkeypatch.setattr(ablation, "run_single", run_single)

    ablation.run_sweep(
        sweep_name,
        tmp_path,
        dataset="wikitext-103",
        device=torch.device("cpu"),
        seed=6,
        resume=False,
    )

    assert calls == ["train", "validation"]
    statuses = {
        label: json.loads(
            (tmp_path / sweep_name / ablation._sanitize(label) / "ablation_result.json").read_text(
                encoding="utf-8")
        )["status"]
        for label in ("matching", "drifted", "missing")
    }
    assert statuses == {"matching": "success", "drifted": "failed", "missing": "failed"}
    assert not (
        tmp_path / sweep_name / ablation._sanitize("drifted") / "cell_contract.json").exists()
    assert not (
        tmp_path / sweep_name / ablation._sanitize("missing") / "cell_contract.json").exists()


def test_ablation_terminal_code_drift_invalidates_every_invocation_cell(
    tmp_path, monkeypatch,
):
    sources = _source_identities("train", "validation")
    before = dict(_CODE_IDENTITY)
    after = {**before, "git_sha": "b" * 40}
    state = {"code": before}
    code_calls: list[dict[str, object]] = []
    sweep_name = "code_identity_terminal_drift"
    monkeypatch.setitem(ablation.SWEEPS, sweep_name, {"description": "code drift"})
    monkeypatch.setattr(
        ablation,
        "make_run_overrides",
        lambda _name: [("first", {}), ("second", {})],
    )
    monkeypatch.setattr(
        ablation,
        "cache_source_identity",
        lambda dataset, split, *, cache_dir=None: copy.deepcopy(sources[split]),
    )

    def code_identity():
        snapshot = dict(state["code"])
        code_calls.append(snapshot)
        return snapshot

    def run_single(label, _overrides, run_dir, **kwargs):
        del kwargs
        result = _ablation_success(label, sources, run_dir)
        if label == "second":
            state["code"] = after
        return result

    monkeypatch.setattr(ablation, "_git_code_identity", code_identity)
    monkeypatch.setattr(ablation, "run_single", run_single)
    monkeypatch.setattr(ablation, "_cleanup", lambda: None)

    result = ablation.run_sweep(
        sweep_name,
        tmp_path,
        dataset="wikitext-103",
        device=torch.device("cpu"),
        seed=6,
        resume=False,
    )

    assert code_calls == [before, after]
    assert result == []
    for label in ("first", "second"):
        run_dir = tmp_path / sweep_name / ablation._sanitize(label)
        marker = json.loads((run_dir / "ablation_result.json").read_text(encoding="utf-8"))
        assert marker["status"] == "failed"
        assert marker["error_kind"] == "code_identity_drift"
        assert marker["cell_contract_fingerprint"] is None
        assert not (run_dir / "cell_contract.json").exists()
    meta = json.loads((tmp_path / sweep_name / "sweep_meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "incomplete"
    assert "code identity" in meta["error"].lower()


def _complete_scaling_result(label: str, route: str, seed: int) -> dict[str, object]:
    test_ce = 2.0
    return {
        "label": label,
        "route": route,
        "scale_knob": "embed_dim",
        "seed": seed,
        "cached": True,
        "error_kind": None,
        "n_params": 10,
        "test_ce": test_ce,
        "test_ppl": math.exp(test_ce),
        "test_bits_per_token": test_ce / math.log(2.0),
        "test_bpc": None,
    }


def _configure_scaling_main(tmp_path, monkeypatch) -> None:
    cells = [
        {"label": "small", "route": "probe", "scale_knob": "embed_dim", "overrides": {}},
        {"label": "large", "route": "probe", "scale_knob": "embed_dim", "overrides": {}},
    ]
    monkeypatch.setattr(scaling, "ROUTES", {"probe": cells})
    monkeypatch.setitem(scaling.CONFIG, "routes", ["probe"])
    monkeypatch.setitem(scaling.CONFIG, "seeds", [3, 5])
    monkeypatch.setitem(scaling.CONFIG, "device", "cpu")
    monkeypatch.setitem(scaling.CONFIG, "dataset", "wikitext-103")
    monkeypatch.setitem(scaling.CONFIG, "max_tokens", None)
    monkeypatch.setitem(scaling.CONFIG, "max_steps", None)
    monkeypatch.setitem(scaling.CONFIG, "output_dir", str(tmp_path))
    monkeypatch.setattr(scaling, "validate_routes", lambda: None)
    monkeypatch.setattr(scaling, "_cleanup", lambda: None)


def test_scaling_main_shares_one_pre_snapshot_and_one_post_snapshot(
    tmp_path, monkeypatch,
):
    _configure_scaling_main(tmp_path, monkeypatch)
    sources = _source_identities("train", "validation", "test")
    calls: list[str] = []
    received: list[object] = []
    received_code: list[object] = []
    code_calls: list[object] = []

    def source_identity(dataset, split):
        assert dataset == "wikitext-103"
        calls.append(split)
        return copy.deepcopy(sources[split])

    def run_cell(cell, run_dir, seed, **kwargs):
        del run_dir
        received.append(kwargs.get("source_identities"))
        received_code.append(kwargs.get("code_identity"))
        return _complete_scaling_result(cell["label"], cell["route"], seed)

    def code_identity():
        code_calls.append(None)
        return dict(_CODE_IDENTITY)

    monkeypatch.setattr(scaling, "cache_source_identity", source_identity)
    monkeypatch.setattr(scaling, "_current_code_identity", code_identity)
    monkeypatch.setattr(scaling, "run_cell", run_cell)

    assert scaling.main() == 0
    assert calls == ["train", "validation", "test", "train", "validation", "test"]
    assert len(received) == 4
    assert all(item == sources for item in received)
    assert len(code_calls) == 2
    assert all(item == _CODE_IDENTITY for item in received_code)


def test_scaling_main_fails_closed_when_terminal_source_snapshot_drifts(
    tmp_path, monkeypatch,
):
    _configure_scaling_main(tmp_path, monkeypatch)
    before = _source_identities("train", "validation", "test")
    after = copy.deepcopy(before)
    after["test"]["sha256"] = "f" * 64
    snapshots = iter((before, after))
    monkeypatch.setattr(
        scaling,
        "_data_source_identities",
        lambda dataset: copy.deepcopy(next(snapshots)),
    )
    monkeypatch.setattr(
        scaling,
        "run_cell",
        lambda cell, run_dir, seed, **kwargs: _complete_scaling_result(
            cell["label"], cell["route"], seed
        ),
    )

    assert scaling.main() == 1
    design = json.loads((tmp_path / "scaling_design.json").read_text(encoding="utf-8"))
    assert design["status"] == "incomplete"
    assert "source" in design["error"].lower()


def test_scaling_main_fails_closed_when_terminal_code_identity_drifts(
    tmp_path, monkeypatch,
):
    _configure_scaling_main(tmp_path, monkeypatch)
    sources = _source_identities("train", "validation", "test")
    before = dict(_CODE_IDENTITY)
    after = {**before, "git_sha": "b" * 40}
    state = {"code": before}
    code_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        scaling,
        "_data_source_identities",
        lambda dataset: copy.deepcopy(sources),
    )

    def code_identity():
        snapshot = dict(state["code"])
        code_calls.append(snapshot)
        return snapshot

    def run_cell(cell, run_dir, seed, **kwargs):
        del run_dir, kwargs
        if cell["label"] == "large" and seed == 5:
            state["code"] = after
        return _complete_scaling_result(cell["label"], cell["route"], seed)

    monkeypatch.setattr(scaling, "_current_code_identity", code_identity)
    monkeypatch.setattr(scaling, "run_cell", run_cell)

    assert scaling.main() == 1
    assert code_calls == [before, after]
    design = json.loads((tmp_path / "scaling_design.json").read_text(encoding="utf-8"))
    assert design["status"] == "incomplete"
    assert "code identity" in design["error"].lower()


def test_scaling_source_preflight_fails_before_output_or_cell_execution(
    tmp_path, monkeypatch,
):
    output_dir = tmp_path / "must-not-exist"
    _configure_scaling_main(output_dir, monkeypatch)
    executed: list[object] = []

    def missing_sources(dataset):
        raise FileNotFoundError(f"missing corpus {dataset}")

    monkeypatch.setattr(scaling, "_data_source_identities", missing_sources)
    monkeypatch.setattr(scaling, "run_cell", lambda *args, **kwargs: executed.append(None))

    assert scaling.main() == 1
    assert executed == []
    assert not output_dir.exists()


def test_scaling_code_identity_preflight_fails_before_output_or_cell_execution(
    tmp_path, monkeypatch,
):
    output_dir = tmp_path / "must-not-exist"
    _configure_scaling_main(output_dir, monkeypatch)
    executed: list[object] = []
    monkeypatch.setattr(
        scaling,
        "_current_code_identity",
        lambda: {
            "git_sha": None,
            "git_dirty": None,
            "git_dirty_fingerprint": None,
            "git_error": "probe failed",
        },
    )
    monkeypatch.setattr(scaling, "run_cell", lambda *args, **kwargs: executed.append(None))

    assert scaling.main() == 1
    assert executed == []
    assert not output_dir.exists()
