"""Regression tests for the July 9 click-to-run script repairs."""

import sys
import types

import pytest
import torch

import generate_efe
from vfe3.run_artifacts import semantic_config_fingerprint


def _save_bound_bundle(path, config, state_dict):
    torch.save({
        "model_state":        state_dict,
        "config":             config,
        "config_fingerprint": semantic_config_fingerprint(config),
    }, path)


def test_generate_efe_requires_explicit_checkpoint_path():
    with pytest.raises(ValueError, match=r"CONFIG\['checkpoint'\]"):
        generate_efe._load_checkpoint({"checkpoint": "", "config_from": None})


def test_generate_efe_rejects_corrupt_embedded_config_fingerprint(tmp_path):
    checkpoint = tmp_path / "best_model.pt"
    torch.save({
        "model_state":        {"weight": torch.tensor([1.0])},
        "config":             {"vocab_size": 50257, "n_e_steps": 1},
        "config_fingerprint": "not-the-config-fingerprint",
    }, checkpoint)

    with pytest.raises(ValueError, match="fingerprint"):
        generate_efe._load_checkpoint({"checkpoint": checkpoint, "config_from": None})


def test_generate_efe_loads_self_bound_best_bundle(tmp_path):
    checkpoint = tmp_path / "best_model.pt"
    state_dict = {"weight": torch.tensor([1.0])}
    config = {"vocab_size": 50257, "n_e_steps": 1}
    _save_bound_bundle(checkpoint, config, state_dict)

    loaded_config, loaded_state = generate_efe._load_checkpoint({
        "checkpoint": checkpoint,
        "config_from": None,
    })

    assert loaded_config == config
    assert torch.equal(loaded_state["weight"], state_dict["weight"])


def test_generate_efe_rejects_semantically_mismatched_config_from(tmp_path):
    checkpoint = tmp_path / "best_model.pt"
    config_from = tmp_path / "step_10.pt"
    state_dict = {"weight": torch.tensor([1.0])}
    _save_bound_bundle(
        checkpoint, {"vocab_size": 50257, "n_e_steps": 1}, state_dict,
    )
    torch.save({
        "model_state": state_dict,
        "config":      {"vocab_size": 50257, "n_e_steps": 3},
    }, config_from)

    with pytest.raises(ValueError, match="semantic config mismatch"):
        generate_efe._load_checkpoint({
            "checkpoint": checkpoint,
            "config_from": config_from,
        })


def test_generate_efe_legacy_state_requires_matching_bound_weights(tmp_path):
    checkpoint = tmp_path / "legacy_best_model.pt"
    config_from = tmp_path / "step_10.pt"
    torch.save({"weight": torch.tensor([1.0])}, checkpoint)
    torch.save({
        "model_state": {"weight": torch.tensor([2.0])},
        "config":      {"vocab_size": 50257, "n_e_steps": 1},
    }, config_from)

    with pytest.raises(ValueError, match="cannot bind legacy state_dict"):
        generate_efe._load_checkpoint({
            "checkpoint": checkpoint,
            "config_from": config_from,
        })


def test_generate_efe_legacy_state_accepts_identical_bound_weights(tmp_path):
    checkpoint = tmp_path / "legacy_best_model.pt"
    config_from = tmp_path / "step_10.pt"
    state_dict = {"weight": torch.tensor([1.0])}
    config = {"vocab_size": 50257, "n_e_steps": 1}
    torch.save(state_dict, checkpoint)
    torch.save({"model_state": state_dict, "config": config}, config_from)

    loaded_config, loaded_state = generate_efe._load_checkpoint({
        "checkpoint": checkpoint,
        "config_from": config_from,
    })

    assert loaded_config == config
    assert torch.equal(loaded_state["weight"], state_dict["weight"])


def test_generate_efe_selects_dataset_tokenizer(monkeypatch):
    calls = []

    def get_encoding(name):
        calls.append(name)
        vocab_size = 100277 if name == "cl100k_base" else 50257
        return types.SimpleNamespace(n_vocab=vocab_size)

    monkeypatch.setitem(sys.modules, "tiktoken", types.SimpleNamespace(get_encoding=get_encoding))

    generate_efe._tokenizer_for_dataset("wikitext-103", vocab_size=50257)
    generate_efe._tokenizer_for_dataset("wiki-en", vocab_size=100277)
    generate_efe._tokenizer_for_dataset("wiki-ja", vocab_size=100277)

    assert calls == ["gpt2", "cl100k_base", "cl100k_base"]


def test_generate_efe_rejects_tokenizer_vocab_mismatch(monkeypatch):
    enc = types.SimpleNamespace(n_vocab=50257)
    monkeypatch.setitem(
        sys.modules, "tiktoken", types.SimpleNamespace(get_encoding=lambda _name: enc),
    )

    with pytest.raises(ValueError, match="tokenizer vocabulary"):
        generate_efe._tokenizer_for_dataset("wiki-en", vocab_size=100277)
