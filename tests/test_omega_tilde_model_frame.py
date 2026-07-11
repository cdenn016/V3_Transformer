"""Independent model-channel gauge-frame contracts (phi-tilde)."""

import copy

import pytest
import torch

import ablation
import train_vfe3
from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel


BASE = dict(
    vocab_size=12,
    embed_dim=4,
    max_seq_len=5,
    n_heads=2,
    n_layers=1,
    n_e_steps=1,
    prior_source="model_channel",
    s_e_step=True,
    lambda_h=1.0,
    lambda_gamma=0.75,
    pos_phi="learned",
    pos_rotation="none",
    phi_reflection="off",
    share_refine_s_transport=False,
)


def _cfg(**overrides: object) -> VFE3Config:
    values = {**BASE, **overrides}
    return VFE3Config(**values)


def _model(seed: int = 13, **overrides: object) -> VFEModel:
    torch.manual_seed(seed)
    return VFEModel(_cfg(seed=seed, **overrides))


def test_model_frame_config_defaults_are_tied_and_separately_clocked() -> None:
    cfg = VFE3Config()
    assert cfg.s_frame_mode == "tied"
    assert cfg.m_s_phi_lr == 0.015


def test_tied_mode_adds_no_state_or_rng_draw() -> None:
    torch.manual_seed(23)
    implicit = VFEModel(VFE3Config(**BASE, seed=23))
    implicit_rng = torch.random.get_rng_state().clone()

    torch.manual_seed(23)
    explicit = VFEModel(VFE3Config(**BASE, seed=23, s_frame_mode="tied", m_s_phi_lr=9.0))
    explicit_rng = torch.random.get_rng_state().clone()

    assert implicit_rng.equal(explicit_rng)
    assert list(implicit.state_dict()) == list(explicit.state_dict())
    for name, value in implicit.state_dict().items():
        assert torch.equal(value, explicit.state_dict()[name]), name
    assert not hasattr(explicit.prior_bank, "s_phi_embed")
    assert not hasattr(explicit, "s_pos_phi_free")


def test_phi_tilde_clones_complete_learned_frame_without_aliasing_or_rng() -> None:
    torch.manual_seed(31)
    tied = VFEModel(_cfg(seed=31, s_frame_mode="tied"))
    tied_rng = torch.random.get_rng_state().clone()

    torch.manual_seed(31)
    independent = VFEModel(_cfg(seed=31, s_frame_mode="phi_tilde"))
    independent_rng = torch.random.get_rng_state().clone()

    assert tied_rng.equal(independent_rng)
    assert torch.equal(independent.prior_bank.s_phi_embed, independent.prior_bank.phi_embed)
    assert independent.prior_bank.s_phi_embed is not independent.prior_bank.phi_embed
    assert independent.prior_bank.s_phi_embed.data_ptr() != independent.prior_bank.phi_embed.data_ptr()
    assert torch.equal(independent.s_pos_phi_free, independent.pos_phi_free)
    assert independent.s_pos_phi_free is not independent.pos_phi_free
    assert independent.s_pos_phi_free.data_ptr() != independent.pos_phi_free.data_ptr()

    token_ids = torch.tensor([[0, 3, 3, 7]])
    assert torch.equal(independent.prior_bank.s_phi(token_ids), independent.prior_bank.s_phi_embed[token_ids])
    assert len(independent.prior_bank.encode_s(token_ids)) == 2


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"gauge_parameterization": "omega_direct"}, "gauge_parameterization"),
        ({"s_e_step": False}, "s_e_step"),
        ({"prior_source": "token"}, "prior_source"),
        ({"share_refine_s_transport": True}, "share_refine_s_transport"),
        ({"phi_reflection": "init_seed"}, "phi_reflection"),
        ({"pos_rotation": "rope"}, "pos_rotation"),
    ],
)
def test_phi_tilde_rejects_unsupported_or_inactive_paths(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _cfg(s_frame_mode="phi_tilde", **overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"e_step_gradient": "detach"},
        {"e_step_gradient": "straight_through"},
        {"m_s_phi_lr": 0.0},
    ],
)
def test_phi_tilde_warns_when_its_training_path_is_severed(overrides: dict[str, object]) -> None:
    with pytest.warns(UserWarning, match="phi_tilde"):
        _cfg(s_frame_mode="phi_tilde", **overrides)


def test_click_run_configs_expose_default_off_model_frame_controls() -> None:
    assert train_vfe3.config["s_frame_mode"] == "tied"
    assert train_vfe3.config["m_s_phi_lr"] == 0.016
    assert ablation.BASELINE_CONFIG["s_frame_mode"] == "tied"
    assert ablation.BASELINE_CONFIG["m_s_phi_lr"] == 0.016


def test_model_frame_sweeps_are_registered_but_inactive() -> None:
    assert ablation.SWEEPS["s_frame_mode"]["values"] == ["tied", "phi_tilde"]
    assert ablation.SWEEPS["s_frame_mode"]["requires"] == {
        "s_e_step": True,
        "prior_source": "model_channel",
        "share_refine_s_transport": False,
    }
    assert len(ablation.SWEEPS["m_s_phi_lr"]["values"]) >= 2
    assert "s_frame_mode" not in ablation.SWEEP_ORDER
    assert "m_s_phi_lr" not in ablation.SWEEP_ORDER

    values = copy.deepcopy(ablation.BASELINE_CONFIG)
    values.update(ablation.SWEEPS["s_frame_mode"]["requires"])
    values["s_frame_mode"] = "phi_tilde"
    VFE3Config(**values)
