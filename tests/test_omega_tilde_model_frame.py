"""Independent model-channel gauge-frame contracts (phi-tilde)."""

import copy

import pytest
import torch

import ablation
import train_vfe3
from vfe3.config import VFE3Config
from vfe3.geometry.transport import compute_transport_operators
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


def test_invalid_model_frame_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="s_frame_mode"):
        _cfg(s_frame_mode="not_a_model_frame")


@pytest.mark.parametrize(
    "overrides",
    [
        {"e_step_gradient": "detach"},
        {"e_step_gradient": "straight_through"},
        {"detach_e_step": True},
        {"m_s_phi_lr": 0.0},
    ],
)
def test_phi_tilde_warns_when_its_training_path_is_severed(overrides: dict[str, object]) -> None:
    with pytest.warns(UserWarning, match="phi_tilde"):
        _cfg(s_frame_mode="phi_tilde", **overrides)


@pytest.mark.parametrize("gauge_transport", ["off", "frozen"])
def test_global_gauge_transport_gate_freezes_phi_tilde(gauge_transport: str) -> None:
    with pytest.warns(UserWarning):
        cfg = _cfg(s_frame_mode="phi_tilde", gauge_transport=gauge_transport)
    assert cfg.m_phi_lr == 0.0
    assert cfg.m_s_phi_lr == 0.0


def test_click_run_configs_expose_default_off_model_frame_controls() -> None:
    assert train_vfe3.config["s_frame_mode"] == "tied"
    assert train_vfe3.config["m_s_phi_lr"] == 0.016
    assert ablation.BASELINE_CONFIG["s_frame_mode"] == "tied"
    assert ablation.BASELINE_CONFIG["m_s_phi_lr"] == 0.016


def test_model_frame_sweeps_are_registered_but_inactive() -> None:
    assert ablation.SWEEPS["s_frame_mode"]["values"] == ["tied", "phi_tilde"]
    assert ablation.SWEEPS["s_frame_mode"]["requires"] == {
        "gauge_parameterization": "phi",
        "phi_reflection": "off",
        "pos_rotation": "none",
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


def test_phi_tilde_effective_frame_and_transport_match_tied_at_clone_init() -> None:
    model = _model(s_frame_mode="phi_tilde")
    token_ids = torch.tensor([[0, 1, 4, 7]])
    belief_phi = model._apply_pos_phi(model.prior_bank.encode(token_ids).phi)
    model_phi = model._resolve_model_frame(token_ids, belief_phi)

    assert torch.equal(model_phi, belief_phi)
    assert model_phi.data_ptr() != belief_phi.data_ptr()

    belief_omega = compute_transport_operators(belief_phi, model.group)["Omega"]
    model_omega = compute_transport_operators(model_phi, model.group)["Omega"]
    assert torch.equal(model_omega, belief_omega)


def test_phi_tilde_transport_is_a_flat_vertex_cocycle() -> None:
    model = _model(s_frame_mode="phi_tilde")
    token_ids = torch.tensor([[0, 2, 5, 8]])
    belief_phi = model._apply_pos_phi(model.prior_bank.encode(token_ids).phi)
    model_phi = model._resolve_model_frame(token_ids, belief_phi)
    omega = compute_transport_operators(model_phi, model.group)["Omega"][0]

    lhs = omega[0, 1] @ omega[1, 3]
    assert torch.allclose(lhs, omega[0, 3], atol=2e-5, rtol=1e-5)


def test_phi_tilde_drives_gamma_and_refine_without_changing_belief_frame() -> None:
    model = _model(
        s_frame_mode="phi_tilde",
        pos_phi="none",
        e_s_mu_lr=0.2,
        e_s_sigma_lr=0.1,
    )
    token_ids = torch.tensor([[0, 1, 2, 3]])
    belief_phi0 = model.prior_bank.encode(token_ids).phi.detach().clone()
    model_phi0 = model._resolve_model_frame(token_ids, belief_phi0)
    energy0 = model._gamma_energy(token_ids, model_phi0)[0]
    refined0 = model._refine_s(token_ids, model_phi0)
    routed0 = model._refined_s_belief(token_ids)
    assert routed0 is not None

    with torch.no_grad():
        model.prior_bank.s_phi_embed[1].add_(0.7)

    belief_phi1 = model.prior_bank.encode(token_ids).phi.detach().clone()
    model_phi1 = model._resolve_model_frame(token_ids, belief_phi1)
    energy1 = model._gamma_energy(token_ids, model_phi1)[0]
    refined1 = model._refine_s(token_ids, model_phi1)
    routed1 = model._refined_s_belief(token_ids)
    assert routed1 is not None

    assert torch.equal(belief_phi1, belief_phi0)
    assert not torch.allclose(energy1, energy0, atol=1e-6, rtol=0.0)
    assert not torch.allclose(refined1[0], refined0[0], atol=1e-6, rtol=0.0)
    assert not torch.allclose(routed1[0], routed0[0], atol=1e-6, rtol=0.0)

    fixed_model_phi = model_phi1.detach().clone()
    fixed_energy = model._gamma_energy(token_ids, fixed_model_phi)[0]
    fixed_refined = model._refine_s(token_ids, fixed_model_phi)
    with torch.no_grad():
        model.prior_bank.phi_embed[2].add_(0.9)
    changed_belief_phi = model.prior_bank.encode(token_ids).phi
    assert not torch.allclose(changed_belief_phi, belief_phi1, atol=1e-6, rtol=0.0)
    assert torch.equal(model._resolve_model_frame(token_ids, changed_belief_phi), fixed_model_phi)
    assert torch.equal(model._gamma_energy(token_ids, fixed_model_phi)[0], fixed_energy)
    after_refined = model._refine_s(token_ids, fixed_model_phi)
    assert torch.equal(after_refined[0], fixed_refined[0])
    assert torch.equal(after_refined[1], fixed_refined[1])


def test_diagnostic_snapshot_freezes_the_effective_model_frame() -> None:
    model = _model(s_frame_mode="phi_tilde", pos_phi="none")
    token_ids = torch.tensor([[0, 1, 2, 3]])
    snapshot = model.build_diagnostic_snapshot(token_ids)
    expected = model._resolve_model_frame(token_ids, snapshot.final_belief.phi)

    assert torch.equal(snapshot.model_phi, expected)
    frozen = snapshot.model_phi.clone()
    with torch.no_grad():
        model.prior_bank.s_phi_embed.add_(0.5)
    assert torch.equal(snapshot.model_phi, frozen)
    assert torch.equal(model.gamma_attention_maps(token_ids, snapshot=snapshot), snapshot.gamma_maps)
