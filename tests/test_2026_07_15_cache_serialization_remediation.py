r"""Regressions for first-panel T1--T4 and Q4 cache/serialization contracts."""

import copy
from dataclasses import asdict

import pytest
import torch

from sigma_gate_measure import load_model_from_checkpoint
from vfe3.config import VFE3Config, config_from_serialized
from vfe3.inference.belief_cache import (
    cache_supported,
    rollout_predictive_state_cached,
)
from vfe3.model.model import VFEModel
from vfe3.viz import embedding_comparison


def _tiny_model(**overrides: object) -> VFEModel:
    values = {
        "vocab_size": 16,
        "embed_dim": 4,
        "n_heads": 2,
        "max_seq_len": 16,
    }
    values.update(overrides)
    torch.manual_seed(0)
    return VFEModel(VFE3Config(**values))


def _cached_full_terminal_difference(model: VFEModel) -> float:
    context = torch.tensor([[0, 1, 2, 3]])
    candidates = torch.tensor([[[4, 5], [6, 7], [8, 9]]])
    batch, n_context = context.shape
    n_candidates, continuation = candidates.shape[1:]
    extended = torch.cat(
        [context.unsqueeze(1).expand(batch, n_candidates, n_context), candidates],
        dim=2,
    ).reshape(batch * n_candidates, n_context + continuation)

    with torch.no_grad():
        base_logits = model.forward(context)[:, -1, :]
        cached = rollout_predictive_state_cached(
            context,
            candidates,
            model,
            base_logits=base_logits,
        )
        full, _ = model.rollout_beliefs(
            extended,
            return_logits=True,
            decode_last=True,
        )
    full_mu = full.mu[:, -1].reshape(batch, n_candidates, -1)
    full_sigma = full.sigma[:, -1].reshape(batch, n_candidates, -1)
    return max(
        float((cached.mu - full_mu).abs().max()),
        float((cached.sigma - full_sigma).abs().max()),
    )


def test_cache_rejects_alias_whose_effective_update_is_exact_mm() -> None:
    model = _tiny_model(e_step_update="frozen_surrogate_exact", mm_damping=0.75)

    difference = _cached_full_terminal_difference(model)

    assert difference > 1e-6, "the audit probe must exercise a real cached/full mismatch"
    assert not cache_supported(model.cfg)


def test_cache_rejects_group_product_state_with_right_positional_factor() -> None:
    model = _tiny_model(
        pos_phi="frozen",
        pos_phi_compose="group_product",
        pos_phi_scale=0.3,
    )
    encoded = model.prior_bank.encode(torch.tensor([[0, 1, 2, 3]]))
    right_phi = model._pos_phi_right(encoded.phi)

    difference = _cached_full_terminal_difference(model)

    assert right_phi is not None and bool((right_phi != 0).any())
    assert difference > 1e-6, "the audit probe must exercise a real cached/full mismatch"
    assert not cache_supported(model.cfg)


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


def test_sigma_gate_rejects_old_config_schema_before_default_reinterpretation(tmp_path) -> None:
    cfg = VFE3Config(
        vocab_size=16,
        embed_dim=4,
        n_heads=2,
        max_seq_len=8,
        skip_belief_sigma_update=True,
    )
    model = VFEModel(cfg)
    old_config = asdict(cfg)
    del old_config["skip_belief_sigma_update"]
    checkpoint = tmp_path / "old-schema.pt"
    torch.save({"config": old_config, "model_state": model.state_dict()}, checkpoint)

    with pytest.raises(ValueError, match="unsupported checkpoint config schema"):
        load_model_from_checkpoint(str(checkpoint), "cpu")


def test_sigma_gate_loads_exact_current_config_schema(tmp_path) -> None:
    cfg = VFE3Config(
        vocab_size=16,
        embed_dim=4,
        n_heads=2,
        max_seq_len=8,
        skip_belief_sigma_update=True,
    )
    model = VFEModel(cfg)
    checkpoint = tmp_path / "current-schema.pt"
    torch.save({"config": asdict(cfg), "model_state": model.state_dict()}, checkpoint)

    loaded_model, loaded_cfg = load_model_from_checkpoint(str(checkpoint), "cpu")

    assert loaded_cfg.skip_belief_sigma_update is True
    assert set(loaded_model.state_dict()) == set(model.state_dict())
