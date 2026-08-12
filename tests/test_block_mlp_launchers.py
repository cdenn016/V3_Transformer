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


def test_config_and_ablation_keep_default_off_mlp_controls():
    """Library defaults and the ablation baseline stay off while the training launcher is editable."""
    defaults = _label_config()
    assert {key: getattr(defaults, key) for key in _EXPECTED_DEFAULTS} == _EXPECTED_DEFAULTS
    assert {key: ablation.BASELINE_CONFIG[key] for key in _EXPECTED_DEFAULTS} == _EXPECTED_DEFAULTS
    VFE3Config(**train_vfe3.config)


def test_block_mlp_ablation_is_opt_in_and_arms_differ_only_by_enable_toggle():
    """The registered comparison is runnable but remains outside the default sweep schedule."""
    runs = dict(ablation.make_run_overrides("block_mlp"))
    ablation.validate_sweeps(["block_mlp"])
    assert "block_mlp" not in ablation.SWEEP_ORDER
    assert runs == {
        "block_mlp_off": {"use_block_mlp": False},
        "block_mlp_on": {"use_block_mlp": True},
    }


def test_block_mlp_hyperparameter_sweeps_are_complete_and_inactive():
    """Each MLP control has an independent opt-in sweep that never auto-runs."""
    expected = {
        "block_mlp_expansion": {
            "block_mlp_expansion=1": {"use_block_mlp": True, "block_mlp_expansion": 1},
            "block_mlp_expansion=2": {"use_block_mlp": True, "block_mlp_expansion": 2},
            "block_mlp_expansion=4": {"use_block_mlp": True, "block_mlp_expansion": 4},
            "block_mlp_expansion=8": {"use_block_mlp": True, "block_mlp_expansion": 8},
        },
        "block_mlp_activation": {
            "block_mlp_activation=gelu": {
                "use_block_mlp": True, "block_mlp_activation": "gelu",
            },
            "block_mlp_activation=silu": {
                "use_block_mlp": True, "block_mlp_activation": "silu",
            },
            "block_mlp_activation=relu": {
                "use_block_mlp": True, "block_mlp_activation": "relu",
            },
        },
        "block_mlp_dropout": {
            "block_mlp_dropout=0.0": {"use_block_mlp": True, "block_mlp_dropout": 0.0},
            "block_mlp_dropout=0.01": {"use_block_mlp": True, "block_mlp_dropout": 0.01},
            "block_mlp_dropout=0.05": {"use_block_mlp": True, "block_mlp_dropout": 0.05},
            "block_mlp_dropout=0.1": {"use_block_mlp": True, "block_mlp_dropout": 0.1},
        },
        "m_block_mlp_lr": {
            "m_block_mlp_lr=0.001": {"use_block_mlp": True, "m_block_mlp_lr": 0.001},
            "m_block_mlp_lr=0.002": {"use_block_mlp": True, "m_block_mlp_lr": 0.002},
            "m_block_mlp_lr=0.004": {"use_block_mlp": True, "m_block_mlp_lr": 0.004},
            "m_block_mlp_lr=0.008": {"use_block_mlp": True, "m_block_mlp_lr": 0.008},
        },
    }
    sweep_names = ["block_mlp", *expected]

    ablation.validate_sweeps(sweep_names)

    assert not set(sweep_names).intersection(ablation.SWEEP_ORDER)
    for sweep_name, expected_runs in expected.items():
        assert dict(ablation.make_run_overrides(sweep_name)) == expected_runs


def test_active_block_mlp_run_label_is_tagged_without_changing_off_label():
    """Only enabled topology receives the unambiguous MLP run-directory suffix."""
    assert "_mlp" not in train_vfe3._run_label(_label_config(), "synthetic")
    assert "_mlp" in train_vfe3._run_label(_label_config(use_block_mlp=True), "synthetic")
