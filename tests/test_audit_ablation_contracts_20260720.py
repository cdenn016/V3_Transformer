"""Regression contracts for exact ablation flags and growing-N completion."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import ablation


_FLAG_KEYS = tuple(sorted(ablation._SWEEP_DIAGNOSTIC_FLAG_KEYS))
_RAW_NON_BOOLEANS = ("false", 0, 1)
_FALSE_FLAGS = {key: False for key in _FLAG_KEYS}
_CODE_IDENTITY = {
    "git_sha": "a" * 40,
    "git_dirty": False,
    "git_dirty_fingerprint": None,
}


def _source_identity(split: str) -> dict[str, object]:
    return {
        "format": "pt",
        "size_bytes": len(split),
        "sha256": split[0] * 64,
    }


def _loaded_sources() -> dict[str, dict[str, object]]:
    return {split: _source_identity(split) for split in ("train", "validation")}


def _owned_checkpoint(run_dir: Path) -> str:
    checkpoint = run_dir / "checkpoints" / "step_1.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"task-5-terminal")
    return str(checkpoint)


def _point(
    n: int,
    *,
    success: bool,
    batch_size: int,
) -> dict[str, object]:
    if success:
        return {
            "n": n,
            "status": "success",
            "ce": 3.0 + n / 10_000.0,
            "ppl": 20.0 + n / 1_000.0,
            "effective_batch_size": batch_size,
        }
    return {
        "n": n,
        "status": "failed",
        "failure_reason": "RuntimeError: out of memory",
        "effective_batch_size": batch_size,
    }


@pytest.mark.parametrize("field", _FLAG_KEYS)
@pytest.mark.parametrize("raw", _RAW_NON_BOOLEANS)
def test_validated_diagnostic_flags_names_raw_non_boolean_field(field: str, raw: object) -> None:
    flags = dict(_FALSE_FLAGS)
    flags[field] = raw

    with pytest.raises(TypeError, match=field):
        ablation._validated_diagnostic_flags(flags)


@pytest.mark.parametrize("field", _FLAG_KEYS)
@pytest.mark.parametrize("raw", _RAW_NON_BOOLEANS)
def test_sweep_diagnostic_request_names_raw_non_boolean_field(field: str, raw: object) -> None:
    sweep: dict[str, object] = {"description": "raw flag contract", **_FALSE_FLAGS}
    sweep[field] = raw

    with pytest.raises(TypeError, match=field):
        ablation._sweep_diagnostic_request(sweep)


@pytest.mark.parametrize("field", _FLAG_KEYS)
@pytest.mark.parametrize("raw", _RAW_NON_BOOLEANS)
def test_run_single_rejects_raw_flag_before_config_construction(
    field: str,
    raw: object,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flags = dict(_FALSE_FLAGS)
    flags[field] = raw
    monkeypatch.setattr(
        ablation,
        "_cell_cfg_dict",
        lambda *args, **kwargs: pytest.fail("configuration constructed before flag validation"),
    )

    with pytest.raises(TypeError, match=field):
        ablation.run_single(
            "raw",
            {},
            tmp_path / "raw",
            dataset="wikitext-103",
            device=torch.device("cpu"),
            seed=6,
            **flags,
        )


@pytest.mark.parametrize("field", _FLAG_KEYS)
@pytest.mark.parametrize("raw", _RAW_NON_BOOLEANS)
def test_run_sweep_rejects_raw_flag_before_output_publication(
    field: str,
    raw: object,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sweep_name = f"raw_{field}_{raw!s}"
    sweep: dict[str, object] = {"description": "raw flag contract", **_FALSE_FLAGS}
    sweep[field] = raw
    monkeypatch.setitem(ablation.SWEEPS, sweep_name, sweep)
    monkeypatch.setattr(
        ablation,
        "_prepare_owned_output_child",
        lambda *args, **kwargs: pytest.fail("output prepared before flag validation"),
    )

    with pytest.raises(TypeError, match=field):
        ablation.run_sweep(
            sweep_name,
            tmp_path,
            dataset="wikitext-103",
            device=torch.device("cpu"),
            seed=6,
            resume=False,
        )


def test_growing_n_preserves_failed_tail_and_effective_batch_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested = [128, 256, 512]
    loader_calls: list[tuple[int, int]] = []

    def fake_get_loader(dataset, seq_len, batch_size, split, **kwargs):
        loader_calls.append((seq_len, batch_size))
        if seq_len == 512:
            raise RuntimeError("out of memory")
        return seq_len

    monkeypatch.setattr(ablation, "get_loader", fake_get_loader)
    monkeypatch.setattr(
        ablation,
        "evaluate",
        lambda model, loader, **kwargs: {"ce": loader / 100.0, "ppl": loader / 10.0},
    )
    cfg = SimpleNamespace(max_seq_len=128, batch_size=32, vocab_size=50_257)

    curve = ablation._eval_at_growing_n(
        object(),
        cfg,
        "wikitext-103",
        torch.device("cpu"),
        requested_lengths=requested,
    )

    assert loader_calls == [(128, 32), (256, 16), (512, 8)]
    assert [point["n"] for point in curve] == requested
    assert curve[-1] == {
        "n": 512,
        "status": "failed",
        "failure_reason": "RuntimeError: out of memory",
        "effective_batch_size": 8,
    }


def test_growing_n_records_nonfinite_metrics_as_a_failed_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ablation,
        "get_loader",
        lambda dataset, seq_len, batch_size, split, **kwargs: seq_len,
    )
    monkeypatch.setattr(
        ablation,
        "evaluate",
        lambda model, loader, **kwargs: {"ce": float("nan"), "ppl": 20.0},
    )
    cfg = SimpleNamespace(max_seq_len=128, batch_size=32, vocab_size=50_257)

    curve = ablation._eval_at_growing_n(
        object(),
        cfg,
        "wikitext-103",
        torch.device("cpu"),
        requested_lengths=[128, 256],
    )

    assert [point["status"] for point in curve] == ["failed", "failed"]
    assert all("nonfinite" in point["failure_reason"] for point in curve)


def test_completion_rejects_failed_largest_requested_length_without_dropping_reason() -> None:
    requested = [128, 256, 512]
    curve = [
        _point(128, success=True, batch_size=32),
        _point(256, success=True, batch_size=16),
        _point(512, success=False, batch_size=8),
    ]
    result = {"extrap_ce": copy.deepcopy(curve)}

    assert ablation._requested_outputs_are_complete(
        result,
        required_diagnostic_keys=(),
        min_extrapolation_points=2,
        requested_extrapolation_lengths=requested,
        mandatory_extrapolation_lengths=(128, 512),
    ) is False
    assert result["extrap_ce"] == curve
    assert result["extrap_ce"][-1]["failure_reason"] == "RuntimeError: out of memory"


def test_completion_accepts_two_distinct_finite_points_including_largest_and_mandatory() -> None:
    requested = [128, 256, 512]
    result = {
        "extrap_ce": [
            _point(128, success=True, batch_size=32),
            _point(256, success=False, batch_size=16),
            _point(512, success=True, batch_size=8),
        ]
    }

    assert ablation._requested_outputs_are_complete(
        result,
        required_diagnostic_keys=(),
        min_extrapolation_points=2,
        requested_extrapolation_lengths=requested,
        mandatory_extrapolation_lengths=(128, 512),
    ) is True


def test_completion_rejects_duplicate_n_and_missing_mandatory_point() -> None:
    requested = [128, 256, 512]
    duplicate = {
        "extrap_ce": [
            _point(128, success=True, batch_size=32),
            _point(128, success=True, batch_size=32),
            _point(512, success=True, batch_size=8),
        ]
    }
    mandatory_failure = {
        "extrap_ce": [
            _point(128, success=True, batch_size=32),
            _point(256, success=False, batch_size=16),
            _point(512, success=True, batch_size=8),
        ]
    }

    assert ablation._requested_outputs_are_complete(
        duplicate,
        required_diagnostic_keys=(),
        min_extrapolation_points=2,
        requested_extrapolation_lengths=requested,
        mandatory_extrapolation_lengths=(128, 512),
    ) is False
    assert ablation._requested_outputs_are_complete(
        mandatory_failure,
        required_diagnostic_keys=(),
        min_extrapolation_points=2,
        requested_extrapolation_lengths=requested,
        mandatory_extrapolation_lengths=(256, 512),
    ) is False


@pytest.mark.parametrize("missing", ("status", "effective_batch_size"))
def test_completion_requires_status_and_effective_batch_for_contract_domain(missing: str) -> None:
    requested = [128, 256, 512]
    curve = [
        _point(128, success=True, batch_size=32),
        _point(256, success=True, batch_size=16),
        _point(512, success=True, batch_size=8),
    ]
    del curve[1][missing]

    assert ablation._requested_outputs_are_complete(
        {"extrap_ce": curve},
        required_diagnostic_keys=(),
        min_extrapolation_points=2,
        requested_extrapolation_lengths=requested,
        mandatory_extrapolation_lengths=(128, 512),
    ) is False


def test_failed_tail_cannot_publish_completed_fitted_domain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sweep_name = "audit_failed_tail"
    monkeypatch.setitem(ablation.SWEEPS, sweep_name, {
        "description": "failed tail must remain visible",
        "collect_extrapolation": True,
        "extrapolation_lengths": [128, 256, 512],
        "mandatory_extrapolation_lengths": [128, 512],
    })
    monkeypatch.setattr(ablation, "make_run_overrides", lambda _name: [("tail", {})])
    monkeypatch.setattr(ablation, "_git_code_identity", lambda: dict(_CODE_IDENTITY))
    monkeypatch.setattr(
        ablation,
        "cache_source_identity",
        lambda dataset, split, cache_dir=None: _source_identity(split),
    )
    monkeypatch.setattr(ablation, "_cleanup", lambda: None)

    execution_flags: dict[str, object] = {}

    def fake_run_single(label, overrides, run_dir, **kwargs):
        execution_flags.update({key: kwargs[key] for key in _FLAG_KEYS})
        return {
            "label": label,
            "error_kind": None,
            "primary_val_ppl": 8.0,
            "final_val_ppl": 9.0,
            "seed": 6,
            "terminal_checkpoint": _owned_checkpoint(run_dir),
            "_loaded_data_sources": _loaded_sources(),
            "extrap_ce": [
                _point(128, success=True, batch_size=32),
                _point(256, success=True, batch_size=16),
                _point(512, success=False, batch_size=8),
            ],
        }

    monkeypatch.setattr(ablation, "run_single", fake_run_single)
    ablation.run_sweep(
        sweep_name,
        tmp_path,
        dataset="wikitext-103",
        device=torch.device("cpu"),
        seed=6,
        resume=False,
    )

    sweep_dir = tmp_path / sweep_name
    marker = ablation._sanitize("tail")
    result = json.loads(
        (sweep_dir / marker / "ablation_result.json").read_text(encoding="utf-8")
    )
    meta = json.loads(
        (sweep_dir / "sweep_meta.json").read_text(encoding="utf-8")
    )
    expected_flags = {
        "collect_diagnostics": False,
        "collect_extrapolation": True,
        "paired_token_bootstrap": False,
    }
    assert execution_flags == expected_flags
    assert {key: result[key] for key in _FLAG_KEYS} == expected_flags
    assert meta["aggregation_contract"]["diagnostic_flags"] == expected_flags
    assert result["status"] == "failed"
    assert result["extrap_ce"][-1]["failure_reason"] == "RuntimeError: out of memory"
    assert not (sweep_dir / marker / "cell_contract.json").exists()
    assert meta["status"] == "incomplete"
    assert meta["failed_requested_labels"] == ["tail"]
