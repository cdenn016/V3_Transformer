"""Regression coverage for selected-bundle and checkpoint tensor integrity."""

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Tuple

import pytest
import torch
from torch.utils.data import DataLoader

import make_figures
from vfe3.config import VFE3Config
from vfe3.data.datasets import TokenWindows
from vfe3.model.model import VFEModel
from vfe3.run_artifacts import (
    RunArtifacts,
    _run_figures_isolated,
    _validate_best_model_mapping,
    _validate_checkpoint_model_state,
    finalize_run,
    load_checkpoint,
    semantic_config_fingerprint,
)
from vfe3.train import _loader_data_identity, build_optimizer


def _cfg() -> VFE3Config:
    return VFE3Config(
        vocab_size=6,
        embed_dim=4,
        n_heads=2,
        max_seq_len=8,
        n_layers=1,
        n_e_steps=1,
        e_q_mu_lr=0.1,
        e_phi_lr=0.0,
        m_phi_lr=0.0,
        warmup_steps=1,
        max_steps=4,
        generate_figures=False,
    )


def _loader(
    *,
    seq_len: int = 8,
) -> DataLoader:
    tokens = torch.arange(3).repeat(24)
    dataset = TokenWindows(tokens.long(), seq_len)
    return DataLoader(dataset, batch_size=4, shuffle=False, drop_last=True)


def _model_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }


def _selected_run(
    run_dir: Path,
) -> Tuple[VFE3Config, VFEModel, RunArtifacts]:
    cfg = _cfg()
    model = VFEModel(cfg)
    artifacts = RunArtifacts(run_dir, cfg, model)
    artifacts.bind_selection_data_identity(
        _loader_data_identity(_loader(), cfg.vocab_size))
    assert artifacts.maybe_save_best(1, model, 5.0)
    return cfg, model, artifacts


def _finalized_selected_run(
    run_dir: Path,
) -> Tuple[VFE3Config, VFEModel, RunArtifacts]:
    cfg, model, artifacts = _selected_run(run_dir)
    artifacts.save_json("summary.json", {
        "n_steps":      1,
        "n_params":     sum(parameter.numel() for parameter in model.parameters()),
        "best_val_ppl": artifacts.best_val_ppl,
    })
    artifacts.save_json("provenance.json", {
        "code_identity_sha256":   artifacts.code_identity_sha256,
        "selection_data_identity": artifacts.selection_data_identity,
    })
    return cfg, model, artifacts


def _replace_first_float(
    state: MutableMapping[str, torch.Tensor],
    value: float,
) -> str:
    for key, tensor in state.items():
        if tensor.is_floating_point() and tensor.numel() > 0:
            replacement = tensor.detach().clone()
            replacement.reshape(-1)[0] = value
            state[key] = replacement
            return key
    raise AssertionError("test model has no nonempty floating state tensor")


@pytest.mark.parametrize("drift", ("code", "validation", "missing-validation"))
def test_finalize_rejects_selected_bundle_identity_drift_before_load_or_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    cfg, model, artifacts = _selected_run(tmp_path / drift)
    bundle = torch.load(artifacts.best_path, weights_only=True)
    if drift == "code":
        bundle["code_identity_sha256"] = "0" * 64
    elif drift == "validation":
        bundle["selection_data_identity"] = _loader_data_identity(
            _loader(seq_len=4), cfg.vocab_size)
    else:
        bundle["selection_data_identity"] = None
    torch.save(bundle, artifacts.best_path)

    before = _model_state(model)
    calls = {"load": 0, "test": 0}
    original_load_state_dict = model.load_state_dict

    def tracked_load_state_dict(
        state_dict: Mapping[str, torch.Tensor],
        strict:     bool = True,
        assign:     bool = False,
    ) -> object:
        calls["load"] += 1
        return original_load_state_dict(state_dict, strict=strict, assign=assign)

    def tracked_evaluate(*args: object, **kwargs: object) -> Dict[str, object]:
        calls["test"] += 1
        return {
            "ce": 1.0,
            "ppl": 2.0,
            "bits_per_token": 1.0,
            "bpc": None,
        }

    monkeypatch.setattr(model, "load_state_dict", tracked_load_state_dict)
    monkeypatch.setattr("vfe3.train.evaluate", tracked_evaluate)

    with pytest.raises(RuntimeError, match="executable-code|validation-data"):
        finalize_run(model, artifacts, cfg, test_loader=_loader())

    assert calls == {"load": 0, "test": 0}
    for key, expected in before.items():
        assert torch.equal(model.state_dict()[key], expected)


@pytest.mark.parametrize(
    "nonfinite",
    (float("nan"), float("inf"), float("-inf")),
    ids=("nan", "positive-infinity", "negative-infinity"),
)
def test_raw_checkpoint_rejects_nonfinite_model_state_before_copy(
    tmp_path: Path,
    nonfinite: float,
) -> None:
    cfg = _cfg()
    source = VFEModel(cfg)
    artifacts = RunArtifacts(tmp_path / "source", cfg, source)
    checkpoint = artifacts.save_checkpoint(
        0, source, build_optimizer(source, cfg), cfg)
    payload = torch.load(checkpoint, weights_only=True)
    offending_key = _replace_first_float(payload["model_state"], nonfinite)
    corrupted = tmp_path / f"raw-{offending_key.replace('.', '-')}.pt"
    torch.save(payload, corrupted)

    target = VFEModel(cfg)
    before = _model_state(target)
    with pytest.raises(RuntimeError, match=f"checkpoint.*{offending_key}.*nonfinite"):
        load_checkpoint(corrupted, target, cfg=cfg)

    for key, expected in before.items():
        assert torch.equal(target.state_dict()[key], expected)


@pytest.mark.parametrize(
    "nonfinite",
    (float("nan"), float("inf"), float("-inf")),
    ids=("nan", "positive-infinity", "negative-infinity"),
)
def test_selected_bundle_rejects_nonfinite_model_state_before_copy(
    tmp_path: Path,
    nonfinite: float,
) -> None:
    cfg, model, artifacts = _selected_run(tmp_path / "selected")
    bundle = torch.load(artifacts.best_path, weights_only=True)
    offending_key = _replace_first_float(bundle["model_state"], nonfinite)
    torch.save(bundle, artifacts.best_path)

    before = _model_state(model)
    with pytest.raises(RuntimeError, match=f"best-model bundle.*{offending_key}.*nonfinite"):
        finalize_run(model, artifacts, cfg, test_loader=None)

    for key, expected in before.items():
        assert torch.equal(model.state_dict()[key], expected)


def test_state_validation_accepts_finite_float_integer_and_boolean_tensors() -> None:
    expected = {
        "weight": torch.zeros(2, dtype=torch.float32),
        "count":  torch.zeros((), dtype=torch.int64),
        "active": torch.zeros(2, dtype=torch.bool),
    }
    saved = {
        "weight": torch.tensor([1.0, -2.0], dtype=torch.float32),
        "count":  torch.tensor(7, dtype=torch.int64),
        "active": torch.tensor([True, False], dtype=torch.bool),
    }

    validated_state = _validate_checkpoint_model_state(
        saved, expected, Path("finite-control.pt"))
    assert validated_state is saved

    cfg = _cfg()
    config = asdict(cfg)
    bundle = {
        "model_state":        saved,
        "config":             config,
        "config_fingerprint": semantic_config_fingerprint(config),
    }
    validated_bundle = _validate_best_model_mapping(
        bundle, cfg, expected, "legacy finite-control best-model bundle")
    assert torch.equal(validated_bundle["model_state"]["count"], saved["count"])
    assert torch.equal(validated_bundle["model_state"]["active"], saved["active"])
    saved["weight"][0] = 99.0
    assert torch.equal(
        validated_bundle["model_state"]["weight"],
        torch.tensor([1.0, -2.0], dtype=torch.float32),
    )


@pytest.mark.parametrize(
    "corruption",
    (
        "stale-code",
        "missing-validation",
        "missing-trusted-validation",
        "extra-state-key",
        "nonfinite",
    ),
)
def test_offline_figure_preflight_rejects_invalid_selected_bundle_before_model_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    _cfg_value, _model_value, artifacts = _finalized_selected_run(tmp_path / corruption)
    bundle = torch.load(artifacts.best_path, weights_only=True)
    if corruption == "stale-code":
        bundle["code_identity_sha256"] = "0" * 64
    elif corruption == "missing-validation":
        bundle["selection_data_identity"] = None
    elif corruption == "missing-trusted-validation":
        provenance_path = artifacts.run_dir / "provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        del provenance["selection_data_identity"]
        provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    elif corruption == "extra-state-key":
        bundle["model_state"]["unexpected_state"] = torch.zeros(1)
    else:
        _replace_first_float(bundle["model_state"], float("nan"))
    torch.save(bundle, artifacts.best_path)

    calls = 0
    original_load_state_dict = VFEModel.load_state_dict

    def tracked_load_state_dict(
        model:      VFEModel,
        state_dict: Mapping[str, torch.Tensor],
        strict:     bool = True,
        assign:     bool = False,
    ) -> object:
        nonlocal calls
        calls += 1
        return original_load_state_dict(model, state_dict, strict=strict, assign=assign)

    monkeypatch.setattr(VFEModel, "load_state_dict", tracked_load_state_dict)

    with pytest.raises(ValueError, match="invalid best_model.pt"):
        make_figures._validated_run_dir(artifacts.run_dir)

    assert calls == 0


def test_isolated_publication_uses_nonselected_live_snapshot_without_best_weights(
    tmp_path: Path,
) -> None:
    cfg = _cfg()
    model = VFEModel(cfg)
    artifacts = RunArtifacts(
        tmp_path / "live-publication",
        cfg,
        model,
        dataset="synthetic-period3",
    )
    artifacts.save_json("provenance.json", {
        "code_identity_sha256":    artifacts.code_identity_sha256,
        "selection_data_identity": None,
    })

    completed = _run_figures_isolated(
        model,
        artifacts,
        None,
        logging.getLogger("test-live-publication-snapshot"),
        generate_publication=True,
        report_loader=_loader(),
    )

    assert completed is True
    assert not artifacts.best_path.exists()
    assert any((artifacts.run_dir / "figures").glob("*.png"))


@pytest.mark.parametrize(
    "corruption",
    (
        "extra-field",
        "schema-version",
        "stale-code",
        "full-config",
        "extra-state-key",
        "nonfinite",
    ),
)
def test_live_figure_snapshot_rejects_contract_drift(
    tmp_path: Path,
    corruption: str,
) -> None:
    from vfe3.viz.run_loading import load_figure_model_state

    cfg = _cfg()
    model = VFEModel(cfg)
    config = asdict(cfg)
    code_identity = "c" * 64
    payload = {
        "artifact_kind":        "live_figure_snapshot",
        "schema_version":       1,
        "model_state":          _model_state(model),
        "config":               config,
        "config_fingerprint":   semantic_config_fingerprint(config),
        "code_identity_sha256": code_identity,
    }
    if corruption == "extra-field":
        payload["unexpected"] = True
    elif corruption == "schema-version":
        payload["schema_version"] = 2
    elif corruption == "stale-code":
        payload["code_identity_sha256"] = "0" * 64
    elif corruption == "full-config":
        drifted_config = dict(config)
        drifted_config["generate_figures"] = not cfg.generate_figures
        payload["config"] = drifted_config
        payload["config_fingerprint"] = semantic_config_fingerprint(drifted_config)
    elif corruption == "extra-state-key":
        payload["model_state"]["unexpected_state"] = torch.zeros(1)
    else:
        _replace_first_float(payload["model_state"], float("nan"))
    snapshot = tmp_path / f"{corruption}.pt"
    torch.save(payload, snapshot)

    with pytest.raises(ValueError, match="live figure snapshot"):
        load_figure_model_state(
            snapshot,
            cfg,
            model.state_dict(),
            code_identity,
            None,
            map_location="cpu",
        )


def test_live_figure_snapshot_cannot_enter_headline_selected_finalization(
    tmp_path: Path,
) -> None:
    cfg = _cfg()
    model = VFEModel(cfg)
    artifacts = RunArtifacts(tmp_path / "headline", cfg, model)
    artifacts.bind_selection_data_identity(
        _loader_data_identity(_loader(), cfg.vocab_size))
    config = asdict(cfg)
    torch.save({
        "artifact_kind":        "live_figure_snapshot",
        "schema_version":       1,
        "model_state":          model.state_dict(),
        "config":               config,
        "config_fingerprint":   semantic_config_fingerprint(config),
        "code_identity_sha256": artifacts.code_identity_sha256,
    }, artifacts.best_path)
    artifacts.best_val_ppl = 2.0
    artifacts.best_step = 1

    with pytest.raises(RuntimeError, match="semantic best-model mapping"):
        finalize_run(model, artifacts, cfg, test_loader=_loader())
