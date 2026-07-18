from dataclasses import asdict, fields
from importlib.util import find_spec
from pathlib import Path

import pytest

from vfe3.config import VFE3Config, migrate_serialized_config
from vfe3.model.model import VFEModel


FORMER_POLICY_FIELDS = frozenset({
    "policy_mode",
    "policy_horizon",
    "policy_top_k",
    "policy_precision",
    "policy_preference",
    "policy_score_terms",
    "policy_sigma_ambiguity_validated",
    "policy_sigma_gate_artifact",
    "policy_ambiguity_mode",
    "policy_sigma_mc_samples",
})


def test_public_runtime_has_no_policy_surface() -> None:
    assert FORMER_POLICY_FIELDS.isdisjoint(field.name for field in fields(VFE3Config))
    assert not hasattr(VFEModel, "_policy_select")
    assert not hasattr(VFEModel, "rollout_beliefs")


@pytest.mark.parametrize("module", [
    "vfe3.inference.policy",
    "vfe3.inference.ring_task",
    "vfe3.inference.belief_cache",
    "vfe3.inference.candidate_menu",
    "vfe3.inference.sigma_gate",
])
def test_policy_modules_are_absent(module: str) -> None:
    assert find_spec(module) is None


def test_policy_drivers_and_artifacts_are_absent() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "efe_ring_experiment.py",
        "generate_efe.py",
        "sigma_gate_measure.py",
        "vfe3/inference/sigma_gate_preregistry.json",
        "vfe3_policy_results/sigma_gate/wikitext103_ed20_15k.json",
    ):
        assert not (root / relative).exists(), relative


def test_historical_policy_keys_migrate_only_as_retired_fields() -> None:
    payload = asdict(VFE3Config())
    payload.update({
        "policy_mode": "none",
        "policy_horizon": 1,
        "policy_top_k": 8,
        "policy_precision": 1.0,
        "policy_preference": "task",
        "policy_score_terms": ["risk", "ambiguity"],
        "policy_sigma_ambiguity_validated": False,
        "policy_sigma_gate_artifact": None,
        "policy_ambiguity_mode": "likelihood_entropy",
        "policy_sigma_mc_samples": 16,
    })
    with pytest.warns(UserWarning, match="retired active-inference"):
        migration = migrate_serialized_config(
            payload,
            source="historical checkpoint",
            strict_unknown=True,
        )
    assert migration.consumed_retired_keys == FORMER_POLICY_FIELDS
    assert asdict(migration.config) == asdict(VFE3Config())
