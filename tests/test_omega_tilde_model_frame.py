"""Independent model-channel gauge-frame contracts (phi-tilde)."""

import copy
import csv

import pytest
import torch

import ablation
import train_vfe3
from vfe3.config import VFE3Config
from vfe3.geometry.transport import compute_transport_operators
from vfe3.model.model import VFEModel
from vfe3.model import model_frame
from vfe3.run_artifacts import RunArtifacts, load_checkpoint
from vfe3.train import _banner as training_banner
from vfe3.train import _warn_phi_transport_clamp, build_optimizer, train
from vfe3.viz import extract


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

    token_ids = torch.tensor([[0, 1, 2, 3]])
    targets = torch.tensor([[1, 2, 3, 4]])
    implicit_logits = implicit(token_ids)
    explicit_logits = explicit(token_ids)
    _, implicit_loss, implicit_ce = implicit(token_ids, targets)
    _, explicit_loss, explicit_ce = explicit(token_ids, targets)
    assert torch.equal(implicit_logits, explicit_logits)
    assert torch.equal(implicit_loss, explicit_loss)
    assert torch.equal(implicit_ce, explicit_ce)


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


def test_registered_model_frame_mode_is_config_selectable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        model_frame._MODEL_FRAMES,
        "registered_test_frame",
        lambda belief_phi, **kwargs: belief_phi,
    )
    assert _cfg(s_frame_mode="registered_test_frame").s_frame_mode == "registered_test_frame"


@pytest.mark.parametrize(
    "overrides",
    [
        {"e_step_gradient": "detach"},
        {"e_step_gradient": "straight_through"},
        {"detach_e_step": True},
        {"m_s_phi_lr": 0.0},
        {"lambda_gamma": 0.0},
        {"e_s_mu_lr": 0.0, "e_s_sigma_lr": 0.0},
    ],
)
def test_phi_tilde_warns_when_its_training_path_is_severed(overrides: dict[str, object]) -> None:
    with pytest.warns(UserWarning, match="phi_tilde"):
        _cfg(s_frame_mode="phi_tilde", **overrides)


def test_phi_tilde_warns_when_a_detached_oracle_severs_its_gradient() -> None:
    with pytest.warns(UserWarning, match="s_phi_embed/s_pos_phi_free"):
        _cfg(
            s_frame_mode="phi_tilde",
            pos_phi="none",
            gradient_mode="smoothing",
            e_step_update="gradient",
            oracle_unroll_grad=False,
        )


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


def test_phi_tilde_frame_graph_is_independent_of_the_belief_frame_graph() -> None:
    model = _model(s_frame_mode="phi_tilde")
    token_ids = torch.tensor([[0, 1, 2, 3]])
    belief_phi = model._apply_pos_phi(model.prior_bank.encode(token_ids).phi)
    model_phi = model._resolve_model_frame(token_ids, belief_phi)

    model_phi.square().sum().backward()

    assert model.prior_bank.phi_embed.grad is None
    assert model.pos_phi_free.grad is None
    for parameter in (model.prior_bank.s_phi_embed, model.s_pos_phi_free):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert float(parameter.grad.abs().sum()) > 0.0


def test_phi_tilde_gamma_energy_respects_a_common_coordinate_pushforward() -> None:
    model = _model(
        s_frame_mode="phi_tilde",
        gauge_group="so_k",
        n_heads=1,
        pos_phi="none",
    ).double()
    token_ids = torch.tensor([[0, 1, 2, 3]])
    with torch.no_grad():
        model.prior_bank.phi_embed.zero_()
        model.prior_bank.s_phi_embed.zero_()
        model.prior_bank.s_sigma_log_embed.zero_()
    belief_phi0 = model.prior_bank.encode(token_ids).phi
    model_phi0 = model._resolve_model_frame(token_ids, belief_phi0)
    energy0 = model._gamma_energy(token_ids, model_phi0)[0]
    s_mu0 = model.prior_bank.s_mu_embed.detach().clone()

    eta = 0.2 * torch.randn(
        model.group.generators.shape[0],
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(7),
    )
    gauge = torch.linalg.matrix_exp(torch.einsum("a,aij->ij", eta, model.group.generators))
    with torch.no_grad():
        model.prior_bank.mu_embed.copy_(
            torch.einsum("kl,vl->vk", gauge, model.prior_bank.mu_embed))
        model.prior_bank.s_mu_embed.copy_(
            torch.einsum("kl,vl->vk", gauge, model.prior_bank.s_mu_embed))
        model.prior_bank.r_mu.copy_(gauge @ model.prior_bank.r_mu)
        model.prior_bank.phi_embed.copy_(eta.expand_as(model.prior_bank.phi_embed))
        model.prior_bank.s_phi_embed.copy_(eta.expand_as(model.prior_bank.s_phi_embed))
    belief_phi1 = model.prior_bank.encode(token_ids).phi
    model_phi1 = model._resolve_model_frame(token_ids, belief_phi1)
    energy1 = model._gamma_energy(token_ids, model_phi1)[0]

    assert not torch.equal(model.prior_bank.s_mu_embed, s_mu0)
    assert torch.equal(model_phi1, belief_phi1)
    assert torch.allclose(energy1, energy0, atol=5e-8, rtol=1e-8)


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


def test_batch_snapshot_diagnostics_reduce_the_first_model_frame_only() -> None:
    model = _model(s_frame_mode="phi_tilde", pos_phi="none")
    token_ids = torch.tensor([
        [0, 1, 2, 3],
        [8, 7, 6, 5],
    ])
    batch_snapshot = model.build_diagnostic_snapshot(token_ids)
    first_snapshot = model.build_diagnostic_snapshot(token_ids[:1])

    batch_diagnostics = model.diagnostics(token_ids, snapshot=batch_snapshot)
    first_diagnostics = model.diagnostics(token_ids[:1], snapshot=first_snapshot)

    assert batch_diagnostics["gamma_coupling"] == pytest.approx(
        first_diagnostics["gamma_coupling"], abs=1e-7)
    assert batch_diagnostics["gamma_meta_entropy"] == pytest.approx(
        first_diagnostics["gamma_meta_entropy"], abs=1e-7)


def _optimizer_group_for(
    optimizer: torch.optim.Optimizer,
    parameter: torch.nn.Parameter,
) -> dict[str, object]:
    groups = [group for group in optimizer.param_groups
              if any(candidate is parameter for candidate in group["params"])]
    assert len(groups) == 1
    return groups[0]


def test_phi_tilde_optimizer_uses_its_own_lr_and_exact_coverage() -> None:
    model = _model(
        s_frame_mode="phi_tilde",
        m_s_phi_lr=0.007,
        phi_weight_decay=0.031,
    )
    optimizer = build_optimizer(model, model.cfg)

    token_group = _optimizer_group_for(optimizer, model.prior_bank.s_phi_embed)
    position_group = _optimizer_group_for(optimizer, model.s_pos_phi_free)
    assert token_group is position_group
    assert token_group["lr"] == 0.007
    assert token_group["role"] == "phi"
    assert token_group["weight_decay"] == 0.031

    grouped = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    assert len(grouped) == len({id(parameter) for parameter in grouped})
    assert {parameter for parameter in model.parameters() if parameter.requires_grad} == set(grouped)


def test_phi_tilde_natural_gradient_group_is_geometric_and_decay_free() -> None:
    model = _model(
        s_frame_mode="phi_tilde",
        m_phi_natural_grad=True,
        phi_precond_mode="pullback_per_block",
        m_s_phi_lr=0.006,
        phi_weight_decay=0.2,
    )
    optimizer = build_optimizer(model, model.cfg)
    group = _optimizer_group_for(optimizer, model.prior_bank.s_phi_embed)

    assert group is _optimizer_group_for(optimizer, model.s_pos_phi_free)
    assert group["gauge"] is True
    assert group["weight_decay"] == 0.0
    assert group["lr"] == 0.006


@pytest.mark.parametrize("e_step_update", ["gradient", "mm_exact"])
def test_attached_model_estep_trains_and_moves_phi_tilde(e_step_update: str) -> None:
    model = _model(
        s_frame_mode="phi_tilde",
        pos_phi="none",
        e_step_update=e_step_update,
        m_s_phi_lr=0.01,
        phi_weight_decay=0.0,
    )
    optimizer = build_optimizer(model, model.cfg)
    token_ids = torch.tensor([[0, 1, 2, 3]])
    targets = torch.tensor([[1, 2, 3, 4]])
    before = model.prior_bank.s_phi_embed.detach().clone()

    _, loss, _ = model(token_ids, targets)
    loss.backward()
    gradient = model.prior_bank.s_phi_embed.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert float(gradient.abs().sum()) > 0.0
    optimizer.step()

    assert not torch.equal(model.prior_bank.s_phi_embed, before)


def test_attached_model_estep_trains_and_moves_learned_model_position_frame() -> None:
    model = _model(
        s_frame_mode="phi_tilde",
        pos_phi="learned",
        m_s_phi_lr=0.01,
        phi_weight_decay=0.0,
    )
    optimizer = build_optimizer(model, model.cfg)
    token_ids = torch.tensor([[0, 1, 2, 3]])
    targets = torch.tensor([[1, 2, 3, 4]])
    token_before = model.prior_bank.s_phi_embed.detach().clone()
    position_before = model.s_pos_phi_free.detach().clone()

    _, loss, _ = model(token_ids, targets)
    loss.backward()
    for parameter in (model.prior_bank.s_phi_embed, model.s_pos_phi_free):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert float(parameter.grad.abs().sum()) > 0.0
    optimizer.step()

    assert not torch.equal(model.prior_bank.s_phi_embed, token_before)
    assert not torch.equal(model.s_pos_phi_free, position_before)


def test_phi_tilde_mm_exact_device_smoke(device: torch.device) -> None:
    model = _model(
        s_frame_mode="phi_tilde",
        pos_phi="learned",
        e_step_update="mm_exact",
        m_s_phi_lr=0.01,
        phi_weight_decay=0.0,
    ).to(device)
    optimizer = build_optimizer(model, model.cfg)
    token_ids = torch.tensor([[0, 1, 2, 3]], device=device)
    targets = torch.tensor([[1, 2, 3, 4]], device=device)
    belief_phi = model._apply_pos_phi(model.prior_bank.encode(token_ids).phi)
    model_phi = model._resolve_model_frame(token_ids, belief_phi)
    energy = model._gamma_energy(token_ids, model_phi)[0]
    token_before = model.prior_bank.s_phi_embed.detach().clone()
    position_before = model.s_pos_phi_free.detach().clone()

    assert model_phi.device == token_ids.device
    assert torch.isfinite(energy).all()
    _, loss, _ = model(token_ids, targets)
    loss.backward()
    for parameter in (model.prior_bank.s_phi_embed, model.s_pos_phi_free):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert float(parameter.grad.abs().sum()) > 0.0
    optimizer.step()

    assert not torch.equal(model.prior_bank.s_phi_embed, token_before)
    assert not torch.equal(model.s_pos_phi_free, position_before)


def test_phi_tilde_checkpoint_round_trip_restores_frame_and_config(tmp_path) -> None:
    cfg = _cfg(s_frame_mode="phi_tilde", m_s_phi_lr=0.007)
    model = VFEModel(cfg)
    optimizer = build_optimizer(model, cfg)
    with torch.no_grad():
        model.prior_bank.s_phi_embed.add_(0.25)
        model.s_pos_phi_free.sub_(0.125)
    token_frame = model.prior_bank.s_phi_embed.detach().clone()
    position_frame = model.s_pos_phi_free.detach().clone()

    artifacts = RunArtifacts(tmp_path / "run", cfg, model)
    checkpoint = artifacts.save_checkpoint(3, model, optimizer, cfg)
    bundle = torch.load(checkpoint, weights_only=True)
    assert bundle["config"]["s_frame_mode"] == "phi_tilde"
    assert bundle["config"]["m_s_phi_lr"] == 0.007

    restored = VFEModel(cfg)
    restored_optimizer = build_optimizer(restored, cfg)
    assert load_checkpoint(checkpoint, restored, restored_optimizer, cfg=cfg) == 3
    assert torch.equal(restored.prior_bank.s_phi_embed, token_frame)
    assert torch.equal(restored.s_pos_phi_free, position_frame)


def test_strict_state_loading_rejects_tied_phi_tilde_migration() -> None:
    tied = _model(s_frame_mode="tied")
    independent = _model(s_frame_mode="phi_tilde")

    with pytest.raises(RuntimeError, match="Error.*state_dict"):
        tied.load_state_dict(independent.state_dict(), strict=True)
    with pytest.raises(RuntimeError, match="Error.*state_dict"):
        independent.load_state_dict(tied.state_dict(), strict=True)


def test_phi_tilde_learning_rate_is_visible_in_banner_and_metrics(tmp_path) -> None:
    cfg = _cfg(s_frame_mode="phi_tilde", m_s_phi_lr=0.007)
    model = VFEModel(cfg)
    banner = training_banner(
        model,
        cfg,
        dataset="synthetic",
        device=torch.device("cpu"),
        n_steps=1,
    )
    assert "s_phi=0.007" in banner

    token_ids = torch.tensor([[0, 1, 2, 3]])
    targets = torch.tensor([[1, 2, 3, 4]])
    artifacts = RunArtifacts(tmp_path / "run", cfg, model)
    train(
        model,
        [(token_ids, targets)],
        cfg,
        n_steps=1,
        log_interval=1,
        eval_interval=0,
        artifacts=artifacts,
        generate_samples=False,
    )
    with open(artifacts.csv_path, newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert "lr_s_phi" in row
    assert float(row["lr_s_phi"]) >= 0.0


def test_model_channel_bank_exposes_only_the_independent_model_frame() -> None:
    token_ids = torch.tensor([[0, 1, 2, 3]])
    independent = _model(s_frame_mode="phi_tilde")
    independent_bank = extract.model_channel_bank(independent, [token_ids])
    belief_phi = independent._apply_pos_phi(independent.prior_bank.encode(token_ids).phi)
    expected_phi = independent._resolve_model_frame(token_ids, belief_phi).reshape(-1, belief_phi.shape[-1])

    assert independent_bank is not None
    assert torch.equal(independent_bank["phi"], expected_phi)

    tied = _model(s_frame_mode="tied")
    tied_bank = extract.model_channel_bank(tied, [token_ids])
    assert tied_bank is not None
    assert "phi" not in tied_bank


def test_transport_clamp_monitor_identifies_the_model_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    import vfe3.train as train_module

    model = _model(s_frame_mode="phi_tilde", pos_phi="none")
    with torch.no_grad():
        model.prior_bank.phi_embed.zero_()
        model.prior_bank.s_phi_embed.fill_(100.0)
    monkeypatch.setattr(train_module, "_PHI_CLAMP_WARNED", False)
    monkeypatch.setattr(train_module, "_S_PHI_CLAMP_WARNED", False)

    with pytest.warns(RuntimeWarning, match="model.s_phi_embed.*m_s_phi_lr"):
        _warn_phi_transport_clamp(model)
