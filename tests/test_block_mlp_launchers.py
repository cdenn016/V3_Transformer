"""Behavioral click-to-run coverage for the optional block MLP experiment."""

import ablation
import train_vfe3

from vfe3.config import VFE3Config


_EXPECTED_DEFAULTS = {
    "use_block_mlp": False,
    "block_mlp_expansion": 4,
    "block_mlp_activation": "gelu",
    "block_mlp_dropout": 0.0,
    "m_block_mlp_lr": None,
}


def _label_config(**overrides: object) -> VFE3Config:
    values = {
        "vocab_size": 17,
        "embed_dim": 4,
        "n_heads": 2,
        "max_seq_len": 8,
        "n_layers": 1,
        "gauge_group": "block_glk",
    }
    values.update(overrides)
    return VFE3Config(**values)


def test_clickrun_dictionaries_construct_the_approved_default_off_mlp_controls():
    """Both editable launchers construct the exact approved default-off configuration."""
    assert {key: train_vfe3.config[key] for key in _EXPECTED_DEFAULTS} == _EXPECTED_DEFAULTS
    assert {key: ablation.BASELINE_CONFIG[key] for key in _EXPECTED_DEFAULTS} == _EXPECTED_DEFAULTS


def test_block_mlp_ablation_is_opt_in_and_arms_differ_only_by_enable_toggle():
    """The registered comparison is runnable but remains outside the default sweep schedule."""
    runs = dict(ablation.make_run_overrides("block_mlp"))
    ablation.validate_sweeps(["block_mlp"])
    assert "block_mlp" not in ablation.SWEEP_ORDER
    assert runs == {
        "block_mlp_off": {"use_block_mlp": False},
        "block_mlp_on": {"use_block_mlp": True},
    }


def test_active_block_mlp_run_label_is_tagged_without_changing_off_label():
    """Only enabled topology receives the unambiguous MLP run-directory suffix."""
    assert "_mlp" not in train_vfe3._run_label(_label_config(), "synthetic")
    assert "_mlp" in train_vfe3._run_label(_label_config(use_block_mlp=True), "synthetic")
