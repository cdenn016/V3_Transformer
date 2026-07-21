"""Audit regressions for opt-in held-out finalization diagnostics."""

import logging
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import multiseed_analysis
from vfe3 import run_artifacts


class _Model(torch.nn.Module):
    def __init__(self, cfg: SimpleNamespace) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.cfg = cfg


class _Artifacts:
    def __init__(self, root: Path) -> None:
        self.best_val_ppl = float("inf")
        self.best_step = 7
        self.best_path = root / "missing-best-model.pt"
        self.history = []
        self.saved: dict[str, object] = {}

    def save_json(self, name: str, payload: object) -> None:
        self.saved[name] = payload


def _cfg(
    *,
    enabled: bool,
    decode_mode: str = "diagonal",
    n_e_steps: int = 3,
) -> SimpleNamespace:
    return SimpleNamespace(
        batch_size=2,
        decode_mode=decode_mode,
        evaluate_zero_e_steps_counterfactual=enabled,
        generate_figures=False,
        max_seq_len=4,
        max_steps=11,
        n_e_steps=n_e_steps,
        use_head_mixer=False,
        use_prior_bank=True,
    )


def _patch_unrelated_finalization_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_artifacts, "_write_provenance", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_artifacts, "_cost_model_fields", lambda *args, **kwargs: {})
    monkeypatch.setattr(run_artifacts, "_pure_path_report", lambda *args, **kwargs: {})
    monkeypatch.setattr(run_artifacts, "_phi_chart_norm_route", lambda *args, **kwargs: None)
    from vfe3.viz import extract
    monkeypatch.setattr(
        extract,
        "e_step_belief_trace",
        lambda *args, **kwargs: {"free_energy": [torch.tensor(0.0)]},
    )


def _loader() -> list[tuple[torch.Tensor, torch.Tensor]]:
    tokens = torch.zeros((1, 4), dtype=torch.long)
    return [(tokens, tokens.clone())]


@pytest.mark.parametrize("decode_mode", ["diagonal", "diagonal_chunked"])
def test_zero_estep_counterfactual_defaults_to_one_headline_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decode_mode: str,
) -> None:
    cfg = _cfg(enabled=False, decode_mode=decode_mode)
    model = _Model(cfg)
    artifacts = _Artifacts(tmp_path)
    observed: list[tuple[str, int]] = []

    def _evaluate(*args, **kwargs):
        observed.append((model.cfg.decode_mode, model.cfg.n_e_steps))
        return {"ce": 1.25, "ppl": 3.5, "bits_per_token": 1.8, "bpc": 1.4}

    _patch_unrelated_finalization_probes(monkeypatch)
    monkeypatch.setattr("vfe3.train.evaluate", _evaluate)

    results = run_artifacts.finalize_run(
        model,
        artifacts,
        cfg,
        test_loader=_loader(),
        device=torch.device("cpu"),
    )

    assert observed == [(decode_mode, 3)]
    assert model.cfg.n_e_steps == 3
    assert results["diagnostics"] == {}
    assert results["test_ce"] == 1.25
    assert results["test_ppl"] == 3.5
    assert results["test_bits_per_token"] == 1.8
    assert results["test_bpc"] == 1.4
    assert artifacts.saved["summary.json"]["diagnostics"] == {}
    assert "test_ce_no_estep" not in results
    assert "estep_capacity_gain" not in results


def test_zero_estep_counterfactual_is_nested_and_preserves_headline_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(enabled=True, decode_mode="diagonal_chunked", n_e_steps=5)
    model = _Model(cfg)
    artifacts = _Artifacts(tmp_path)
    observed_depths: list[int] = []
    metrics = [
        {"ce": 1.0, "ppl": 2.0, "bits_per_token": 1.5, "bpc": 1.2},
        {"ce": 1.75, "ppl": 99.0, "bits_per_token": 88.0, "bpc": 77.0},
    ]

    def _evaluate(*args, **kwargs):
        observed_depths.append(model.cfg.n_e_steps)
        return metrics[len(observed_depths) - 1]

    _patch_unrelated_finalization_probes(monkeypatch)
    monkeypatch.setattr("vfe3.train.evaluate", _evaluate)

    results = run_artifacts.finalize_run(
        model,
        artifacts,
        cfg,
        test_loader=_loader(),
        device=torch.device("cpu"),
    )

    assert observed_depths == [5, 0]
    assert model.cfg.n_e_steps == 5
    assert results["test_ce"] == 1.0
    assert results["test_ppl"] == 2.0
    assert results["test_bits_per_token"] == 1.5
    assert results["test_bpc"] == 1.2
    assert results["best_val_ppl"] is None
    assert results["best_step"] == 7
    assert results["reloaded_best"] is False
    assert results["diagnostics"]["zero_e_steps_counterfactual"] == {
        "kind": "held_out_inference_depth_counterfactual",
        "split": "test",
        "configured_depth": 5,
        "counterfactual_depth": 0,
        "counterfactual_ce": 1.75,
        "ce_delta_vs_headline": 0.75,
    }
    for payload_name in ("test_results.json", "summary.json"):
        payload = artifacts.saved[payload_name]
        assert payload["test_ce"] == 1.0
        assert payload["test_ppl"] == 2.0
        assert payload["test_bits_per_token"] == 1.5
        assert payload["test_bpc"] == 1.2
        assert payload["diagnostics"] == results["diagnostics"]
        assert "test_ce_no_estep" not in payload
        assert "estep_capacity_gain" not in payload
    scaling_point = artifacts.saved["summary.json"]["scaling_point"]
    assert scaling_point["test_ce"] == 1.0
    assert scaling_point["test_ppl"] == 2.0


def test_zero_estep_counterfactual_failure_restores_depth_without_partial_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = _cfg(enabled=True, n_e_steps=4)
    model = _Model(cfg)
    artifacts = _Artifacts(tmp_path)
    observed_depths: list[int] = []

    def _evaluate(*args, **kwargs):
        observed_depths.append(model.cfg.n_e_steps)
        if len(observed_depths) == 2:
            raise RuntimeError("counterfactual failed")
        return {"ce": 0.9, "ppl": 2.1, "bits_per_token": 1.3, "bpc": None}

    _patch_unrelated_finalization_probes(monkeypatch)
    monkeypatch.setattr("vfe3.train.evaluate", _evaluate)

    with caplog.at_level(logging.WARNING):
        results = run_artifacts.finalize_run(
            model,
            artifacts,
            cfg,
            test_loader=_loader(),
            device=torch.device("cpu"),
        )

    assert observed_depths == [4, 0]
    assert model.cfg.n_e_steps == 4
    assert results["test_ce"] == 0.9
    assert results["diagnostics"] == {}
    assert artifacts.saved["test_results.json"]["diagnostics"] == {}
    assert artifacts.saved["summary.json"]["diagnostics"] == {}
    assert "counterfactual failed" in caplog.text
    assert "test_ce_no_estep" not in results
    assert "estep_capacity_gain" not in results


def test_multiseed_zero_estep_metrics_use_dotted_diagnostic_paths() -> None:
    ce_key = "diagnostics.zero_e_steps_counterfactual.counterfactual_ce"
    delta_key = "diagnostics.zero_e_steps_counterfactual.ce_delta_vs_headline"
    assert ce_key in multiseed_analysis.SCALAR_KEYS
    assert delta_key in multiseed_analysis.SCALAR_KEYS
    assert "test_ce_no_estep" not in multiseed_analysis.SCALAR_KEYS
    assert "estep_capacity_gain" not in multiseed_analysis.SCALAR_KEYS
    payload = {
        "diagnostics": {
            "zero_e_steps_counterfactual": {
                "counterfactual_ce": 1.75,
                "ce_delta_vs_headline": 0.75,
            }
        }
    }
    assert multiseed_analysis._dig_present(payload, ce_key) == (True, 1.75)
    assert multiseed_analysis._dig_present(payload, delta_key) == (True, 0.75)
