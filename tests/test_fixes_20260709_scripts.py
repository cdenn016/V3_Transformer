"""Regression tests for the July 9 click-to-run script repairs."""

import inspect
import sys
import types
from pathlib import Path

import pytest
import torch

import check_junit
import generate_efe
from vfe3.inference.e_step import e_step
from vfe3.run_artifacts import semantic_config_fingerprint


def _save_bound_bundle(path, config, state_dict):
    torch.save({
        "model_state":        state_dict,
        "config":             config,
        "config_fingerprint": semantic_config_fingerprint(config),
    }, path)


def test_read_junit_counts_accepts_testsuite_root(tmp_path):
    xml_path = tmp_path / "single.xml"
    xml_path.write_text(
        '<testsuite tests="5" failures="1" errors="1" skipped="1">'
        '<testcase name="test_param[0]"/><testcase name="test_param[1]"/>'
        '</testsuite>',
        encoding="utf-8",
    )

    assert check_junit.read_junit_counts(xml_path) == {
        "tests": 5,
        "failures": 1,
        "errors": 1,
        "skipped": 1,
        "passes": 2,
    }


def test_read_junit_counts_sums_testsuites_root(tmp_path):
    xml_path = tmp_path / "multiple.xml"
    xml_path.write_text(
        '<testsuites>'
        '<testsuite name="parameterized" tests="3" failures="0" errors="0" skipped="1"/>'
        '<testsuite name="ordinary" tests="2" failures="1" errors="0" skipped="0"/>'
        '</testsuites>',
        encoding="utf-8",
    )

    assert check_junit.read_junit_counts(xml_path) == {
        "tests": 5,
        "failures": 1,
        "errors": 0,
        "skipped": 1,
        "passes": 3,
    }


def test_run_pytest_junit_derives_counts_and_cleans_xml(monkeypatch):
    observed = {}

    def _fake_pytest_main(args):
        observed["args"] = list(args)
        junit_arg = next(arg for arg in args if str(arg).startswith("--junitxml="))
        xml_path = Path(str(junit_arg).split("=", 1)[1])
        observed["xml_path"] = xml_path
        xml_path.write_text(
            '<testsuite tests="4" failures="1" errors="0" skipped="1"/>',
            encoding="utf-8",
        )
        return pytest.ExitCode.TESTS_FAILED

    monkeypatch.setattr(pytest, "main", _fake_pytest_main)

    code, counts = check_junit.run_pytest_junit(["tests/test_example.py"], prefix="audit-")

    assert code == int(pytest.ExitCode.TESTS_FAILED)
    assert counts == {"tests": 4, "failures": 1, "errors": 0, "skipped": 1, "passes": 2}
    assert "-q" not in observed["args"]
    assert not observed["xml_path"].exists()


def test_click_run_verifiers_use_machine_read_junit_counts():
    for script_name in ("check_audit_fixes.py", "check_gpu_tests.py"):
        source = Path(script_name).read_text(encoding="utf-8")
        assert "run_pytest_junit" in source
        assert "pytest.main" not in source


def test_public_e_step_keyword_signature_follows_repository_order():
    names = list(inspect.signature(e_step).parameters)
    expected = [
        "belief", "mu_p", "sigma_p", "group",
        "tau",
        "e_q_mu_lr", "e_q_sigma_lr", "e_phi_lr", "exp_fp64_norm_threshold",
        "n_iter", "e_steps_min", "e_steps_max", "e_steps_backprop_last",
        "e_step_gradient", "exp_fp64_mode",
        "return_trajectory", "oracle_unroll_grad", "randomize_e_steps",
        "transport_mean_per_head", "rope_on_cov", "rope_on_value",
        "e_step_halt_tol", "grad_record", "state_record", "rope", "log_prior", "prebuilt_transport",
        "kwargs",
    ]
    assert names == expected


def test_report_source_uses_american_english_color_spelling():
    source = Path("vfe3/viz/report.py").read_text(encoding="utf-8").lower()
    for british in ("colour", "grey"):
        assert british not in source
    assert "color" in source
    assert "gray" in source


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
