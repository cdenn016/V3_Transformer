r"""Regressions for first-panel T1--T4 and Q4 cache/serialization contracts."""

import copy

import pytest

from vfe3.config import config_from_serialized, migrate_serialized_config
from vfe3.viz import embedding_comparison


@pytest.mark.parametrize(
    ("serialized", "expected"),
    [("true", True), ("True", True), ("false", False), ("False", False)],
)
def test_config_deserialization_normalizes_boolean_spellings(
    serialized: str,
    expected:   bool,
) -> None:
    cfg = config_from_serialized(
        {"skip_belief_sigma_update": serialized},
        source="audit probe",
    )

    assert cfg.skip_belief_sigma_update is expected


@pytest.mark.parametrize("serialized", ["0", "1", "yes", "no", "on", "off", " false ", ""])
def test_config_deserialization_rejects_ambiguous_boolean_strings(serialized: str) -> None:
    with pytest.raises(ValueError, match="skip_belief_sigma_update"):
        config_from_serialized(
            {"skip_belief_sigma_update": serialized},
            source="audit probe",
        )


def test_retired_phi_boolean_uses_the_same_exact_serialized_boolean_rules() -> None:
    disabled = migrate_serialized_config(
        {"m_phi_natural_grad": "False"},
        source="legacy cache artifact",
    )
    enabled = migrate_serialized_config(
        {"m_phi_natural_grad": "TRUE"},
        source="legacy cache artifact",
    )

    assert disabled.config.m_phi_update_mode == "adamw"
    assert disabled.legacy_stateful_phi_optimizer is False
    assert enabled.config.m_phi_update_mode == "adamw"
    assert enabled.legacy_stateful_phi_optimizer is True
    with pytest.raises(ValueError, match="m_phi_natural_grad"):
        migrate_serialized_config(
            {"m_phi_natural_grad": " false "},
            source="legacy cache artifact",
        )


def _comparison_record() -> dict:
    record = embedding_comparison.controlled_contract(
        kind="Belief",
        channel="mu",
        feature_dim=4,
        feature_chart="Euclidean means",
        clustering_space="native 4-D",
    )
    record["sample"] = {
        "token_count": 100,
        "sequence_count": 10,
        "sequence_length": 10,
        "token_sha256": "same-token-population",
    }
    record["semantic_probes"] = {
        "manifest": {"name": "audit-probe", "schema_version": 1},
    }
    return record


def test_comparison_rejects_incompatible_sequence_partitions() -> None:
    left = _comparison_record()
    right = copy.deepcopy(left)
    right["sample"]["sequence_count"] = 11
    right["sample"]["sequence_length"] = 5

    with pytest.raises(ValueError) as exc:
        embedding_comparison.validate_comparison_records([left, right])

    message = str(exc.value)
    assert "sample.sequence_count" in message
    assert "sample.sequence_length" in message
