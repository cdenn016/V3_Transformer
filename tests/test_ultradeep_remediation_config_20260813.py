"""Regression tests for the 2026-08-13 reusable-config remediation."""

from dataclasses import asdict

import pytest
import ablation

from vfe3.config import VFE3Config, migrate_serialized_config


def test_reusable_config_defaults_are_safe_across_construction_contexts() -> None:
    """A new caller gets the chart and decode contracts without launcher settings."""
    cfg = VFE3Config()

    assert cfg.pos_phi_compose == "bch"
    assert cfg.decode_tau == 1.0
    assert cfg.decode_mode == "diagonal_chunked"


def test_legacy_serialized_config_uses_historical_missing_field_defaults() -> None:
    """Omitted historical fields retain their serialized-schema behavior on migration."""
    payload = asdict(VFE3Config())
    for field in ("pos_phi_compose", "decode_tau", "decode_ce_checkpoint"):
        payload.pop(field)

    migrated = migrate_serialized_config(payload, source="legacy config")

    assert migrated.config.pos_phi_compose == "bch"
    assert migrated.config.decode_tau == 1.0
    assert migrated.config.decode_ce_checkpoint == "always"


def test_sweep_resolution_uses_one_compatible_baseline_without_arm_repair(monkeypatch) -> None:
    """A transport arm must not silently acquire a positional-composition override."""
    sweep_name = "task_1_positional_baseline_contract"
    monkeypatch.setitem(
        ablation.SWEEPS,
        sweep_name,
        {
            "description": "test-only incompatible baseline arm",
            "param": "transport_mode",
            "values": ["regime_ii"],
            "requires": {"e_step_update": "gradient"},
        },
    )

    label, overrides = ablation.make_run_overrides(sweep_name)[0]

    assert label == "transport_mode=regime_ii"
    assert "pos_phi_compose" not in overrides
    assert VFE3Config(**ablation._cell_cfg_dict(overrides, seed=6)).pos_phi_compose == "bch"


@pytest.mark.parametrize("sweep_name", ("covariance", "renyi_order"))
def test_registered_sweep_arms_declare_family_consistent_decode_mode(sweep_name: str) -> None:
    """Covariance and Renyi comparisons must resolve their decode prerequisite explicitly."""
    runs = ablation.make_run_overrides(sweep_name)

    assert runs
    assert all(overrides["decode_mode"] == "family_chunked" for _label, overrides in runs)
