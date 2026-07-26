r"""The depth diagnostic must separate the belief E-step from the model-channel refine.

`cfg.n_e_steps` is read by BOTH `model/block.py::vfe_block` and `model/model.py::_refine_s`, and
under `prior_source='model_channel'` the refined s IS the belief's prior. Sweeping the one field
therefore moves the prior and the belief together; measured 2026-07-25, the belief loop supplies
0.012 nats of the CE change over eight iterations and the model channel 3.48.

Every model here is TINY (K < 6) and CPU-bound per CLAUDE.md.
"""

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel
from vfe3.run_artifacts import collect_estep_depth_sensitivity


def _model(**overrides) -> VFEModel:
    cfg = VFE3Config(
        embed_dim=4, n_heads=2, vocab_size=11, max_seq_len=6, batch_size=2,
        n_layers=1, n_e_steps=2, max_steps=1, lambda_alpha_mode="constant",
        **overrides,
    )
    torch.manual_seed(0)
    return VFEModel(cfg)


def test_default_s_depth_follows_n_e_steps_byte_identically() -> None:
    r"""`s_e_step_n_iter=None` is the historical behavior: the s loop tracks `n_e_steps` exactly."""
    tokens = torch.randint(0, 11, (2, 6))
    shared = dict(prior_source="model_channel", s_e_step=True, lambda_gamma=0.5)

    torch.manual_seed(0)
    follow = _model(**shared)
    torch.manual_seed(0)
    pinned = _model(**shared, s_e_step_n_iter=follow.cfg.n_e_steps)
    pinned.load_state_dict(follow.state_dict())

    follow.eval()
    pinned.eval()
    with torch.no_grad():
        assert torch.equal(follow(tokens), pinned(tokens))


def test_s_depth_override_is_actually_honored() -> None:
    r"""Changing only `s_e_step_n_iter` changes the forward, so the seam is live rather than inert."""
    tokens = torch.randint(0, 11, (2, 6))
    shared = dict(prior_source="model_channel", s_e_step=True, lambda_gamma=0.5)

    torch.manual_seed(0)
    base = _model(**shared, s_e_step_n_iter=1)
    torch.manual_seed(0)
    deep = _model(**shared, s_e_step_n_iter=4)
    deep.load_state_dict(base.state_dict())

    base.eval()
    deep.eval()
    with torch.no_grad():
        assert not torch.allclose(base(tokens), deep(tokens))


def test_diagnostic_pins_the_model_channel_while_sweeping_the_belief_loop() -> None:
    r"""Each point records BOTH depths, and the belief series holds the model channel fixed.

    This is the defect the diagnostic used to carry: the swept `depth` was applied to two loops at
    once and the payload recorded only one number, so the model channel's contribution was reported
    as inference-depth sensitivity.
    """
    model = _model(prior_source="model_channel", s_e_step=True, lambda_gamma=0.5)
    tokens = torch.randint(0, 11, (2, 6))
    targets = torch.randint(0, 11, (2, 6))

    record = collect_estep_depth_sensitivity(model, tokens, targets, [1, 2, 3])
    assert record["model_channel_live"] is True
    assert record["n_sequences"] == 2

    for point in record["points"]:                       # belief sweep
        assert point["s_depth"] == record["trained_depth"]
        assert point["belief_depth"] == point["depth"]
    assert [p["belief_depth"] for p in record["points"]] == [1, 2, 3]

    for point in record["model_channel_points"]:         # model-channel sweep
        assert point["belief_depth"] == record["trained_depth"]
        assert point["s_depth"] == point["depth"]
    assert [p["s_depth"] for p in record["model_channel_points"]] == [1, 2, 3]


def test_frozen_model_channel_emits_no_second_series() -> None:
    r"""With `s_e_step=False` the refine never runs, so `n_e_steps` drives the belief loop alone."""
    model = _model(prior_source="token", s_e_step=False)
    tokens = torch.randint(0, 11, (2, 6))
    targets = torch.randint(0, 11, (2, 6))

    record = collect_estep_depth_sensitivity(model, tokens, targets, [1, 2])
    assert record["model_channel_live"] is False
    assert record["model_channel_points"] == []
    assert len(record["points"]) == 2


def test_diagnostic_restores_both_depths_and_training_mode() -> None:
    r"""The probe mutates two config fields now; both must be restored exactly."""
    model = _model(prior_source="model_channel", s_e_step=True, lambda_gamma=0.5,
                   s_e_step_n_iter=3)
    model.train()
    tokens = torch.randint(0, 11, (2, 6))
    targets = torch.randint(0, 11, (2, 6))

    collect_estep_depth_sensitivity(model, tokens, targets, [1, 2])
    assert model.cfg.n_e_steps == 2
    assert model.cfg.s_e_step_n_iter == 3
    assert model.training is True


def test_negative_s_depth_fails_closed() -> None:
    with pytest.raises(ValueError, match="s_e_step_n_iter"):
        VFE3Config(embed_dim=4, n_heads=2, vocab_size=11, max_seq_len=6, batch_size=2,
                   n_layers=1, n_e_steps=2, max_steps=1, s_e_step_n_iter=-1)
