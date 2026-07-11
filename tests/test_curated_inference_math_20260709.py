r"""Curated regression tests for E-step and model-channel inference contracts."""

import importlib

import pytest
import torch

from vfe3.belief import BeliefState
from vfe3.config import VFE3Config
from vfe3.families import get_family
from vfe3.free_energy import attention_weights, self_divergence_for_alpha
from vfe3.geometry.groups import GaugeGroup, get_group
from vfe3.inference.e_step import build_belief_transport, e_step
from vfe3.model.block import vfe_block
from vfe3.model.model import VFEModel
from vfe3.model.stack import vfe_stack


e_step_module = importlib.import_module("vfe3.inference.e_step")
stack_module = importlib.import_module("vfe3.model.stack")


def _truncation_case(
    seed: int = 0,
) -> tuple[BeliefState, torch.Tensor, torch.Tensor, GaugeGroup]:
    generator = torch.Generator().manual_seed(seed)
    n_tokens, width = 3, 2
    group = get_group("glk")(width)
    n_gen = group.generators.shape[0]
    belief = BeliefState(
        mu=torch.randn(n_tokens, width, generator=generator),
        sigma=torch.rand(n_tokens, width, generator=generator) + 0.75,
        phi=0.12 * torch.randn(n_tokens, n_gen, generator=generator),
    )
    mu_p = torch.randn(n_tokens, width, generator=generator)
    sigma_p = torch.rand(n_tokens, width, generator=generator) + 0.75
    return belief, mu_p, sigma_p, group


def _model_channel_case(
    **overrides: object,
) -> tuple[VFEModel, torch.Tensor]:
    values = dict(
        vocab_size=9,
        embed_dim=4,
        n_heads=2,
        max_seq_len=5,
        n_layers=1,
        n_e_steps=2,
        e_q_mu_lr=0.2,
        e_q_sigma_lr=0.1,
        e_phi_lr=0.0,
        e_s_mu_lr=0.7,
        e_s_sigma_lr=0.2,
        mass_phi=0.0,
        use_prior_bank=True,
        prior_source="model_channel",
        s_e_step=True,
        lambda_h=1.0,
        lambda_gamma=0.75,
        seed=13,
    )
    values.update(overrides)
    torch.manual_seed(13)
    model = VFEModel(VFE3Config(**values))
    with torch.no_grad():
        model.prior_bank.s_mu_embed.normal_(mean=0.0, std=0.7)
        model.prior_bank.s_sigma_log_embed.normal_(mean=0.0, std=0.25)
        model.prior_bank.phi_embed.normal_(mean=0.0, std=0.15)
        model.prior_bank.r_mu.fill_(0.8)
        model.prior_bank.r_sigma_log.fill_(0.1)
    token_ids = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long)
    return model, token_ids


def test_shared_prebuilt_transport_respects_truncation_boundary() -> None:
    belief, mu_p, sigma_p, group = _truncation_case(seed=31)
    source_phi = belief.phi.detach().clone().requires_grad_(True)
    belief = belief._replace(phi=source_phi)
    prebuilt_transport = build_belief_transport(source_phi, group)

    out = e_step(
        belief,
        mu_p,
        sigma_p,
        group,
        n_iter=3,
        e_phi_lr=0.0,
        e_steps_backprop_last=1,
        prebuilt_transport=prebuilt_transport,
    )
    loss = out.mu.square().sum()
    source_grad, = torch.autograd.grad(loss, source_phi, allow_unused=True, retain_graph=True)

    assert source_grad is None or torch.count_nonzero(source_grad) == 0
    assert out.phi.is_leaf and out.phi.requires_grad
    boundary_grad, = torch.autograd.grad(loss, out.phi)
    assert torch.isfinite(boundary_grad).all()
    assert torch.count_nonzero(boundary_grad) > 0


def test_oracle_last_k_restores_prior_gradient() -> None:
    belief, mu_p, sigma_p, group = _truncation_case(seed=37)
    mu_p = mu_p.requires_grad_(True)

    out = e_step(
        belief,
        mu_p,
        sigma_p,
        group,
        n_iter=3,
        e_phi_lr=0.0,
        e_steps_backprop_last=1,
        renyi_order=0.5,
        oracle_unroll_grad=True,
    )

    assert out.mu.requires_grad and out.mu.grad_fn is not None
    prior_grad, = torch.autograd.grad(out.mu.square().sum(), mu_p)
    assert torch.isfinite(prior_grad).all()
    assert torch.count_nonzero(prior_grad) > 0


def test_backprop_last_equal_total_matches_full_unroll() -> None:
    base, mu_p_value, sigma_p, group = _truncation_case(seed=41)

    def run(backprop_last: int) -> tuple[BeliefState, torch.Tensor]:
        belief = base._replace(
            mu=base.mu.detach().clone().requires_grad_(True),
            sigma=base.sigma.detach().clone().requires_grad_(True),
            phi=base.phi.detach().clone().requires_grad_(True),
        )
        mu_p = mu_p_value.detach().clone().requires_grad_(True)
        out = e_step(
            belief,
            mu_p,
            sigma_p,
            group,
            n_iter=3,
            e_phi_lr=0.0,
            e_steps_backprop_last=backprop_last,
            renyi_order=0.5,
            oracle_unroll_grad=True,
        )
        prior_grad, = torch.autograd.grad(out.mu.square().sum(), mu_p)
        return out, prior_grad

    full, full_prior_grad = run(0)
    equal_total, equal_total_prior_grad = run(3)

    assert torch.equal(equal_total.mu, full.mu)
    assert torch.equal(equal_total.sigma, full.sigma)
    assert torch.equal(equal_total.phi, full.phi)
    assert torch.equal(equal_total_prior_grad, full_prior_grad)


def test_gamma_attention_maps_uses_refined_s_belief() -> None:
    model, token_ids = _model_channel_case()

    with torch.no_grad():
        encoded = model.prior_bank.encode(token_ids)
        phi = model._apply_pos_phi(encoded.phi)
        s_belief = model._refined_s_belief(token_ids)
        assert s_belief is not None
        refined_energy, tau, log_prior = model._gamma_energy(
            token_ids,
            phi,
            s_belief=s_belief,
        )
        raw_energy, raw_tau, raw_log_prior = model._gamma_energy(token_ids, phi)
        expected = attention_weights(refined_energy, tau=tau, log_prior=log_prior)[0]
        raw = attention_weights(raw_energy, tau=raw_tau, log_prior=raw_log_prior)[0]
        actual = model.gamma_attention_maps(token_ids)

    assert actual is not None
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(expected, raw, atol=1e-5, rtol=1e-5)


def test_gamma_fold_uses_refined_s_belief() -> None:
    model, token_ids = _model_channel_case(
        gamma_as_beta_prior=True,
        gamma_prior_weight=1.0,
    )

    with torch.no_grad():
        phi = model._apply_pos_phi(model.prior_bank.encode(token_ids).phi)
        s_belief = model._refined_s_belief(token_ids)
        assert s_belief is not None
        energy, tau, gamma_log_prior = model._gamma_energy(
            token_ids,
            phi,
            s_belief=s_belief,
        )
        expected_gamma = attention_weights(energy, tau=tau, log_prior=gamma_log_prior)
        expected = torch.log(expected_gamma.clamp(min=1e-12))
        actual = model._fold_gamma_prior(
            None,
            token_ids,
            phi,
            s_belief=s_belief,
        )
        raw = model._fold_gamma_prior(None, token_ids, phi)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(actual, raw, atol=1e-5, rtol=1e-5)


def test_lambda_h_zero_state_dependent_s_refine_has_no_hyperprior_force() -> None:
    with pytest.warns(UserWarning):
        model, token_ids = _model_channel_case(
            lambda_h=0.0,
            lambda_h_mode="state_dependent",
            lambda_gamma=0.0,
            n_e_steps=1,
        )
    s_mu0, s_sigma0 = model.prior_bank.encode_s(token_ids)
    phi0 = model._apply_pos_phi(model.prior_bank.encode(token_ids).phi)

    s_mu1, s_sigma1 = model._refine_s(token_ids, phi0)

    assert torch.equal(s_mu1, s_mu0)
    assert torch.equal(s_sigma1, s_sigma0)


def test_refine_s_forwards_global_estep_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, token_ids = _model_channel_case(
        e_step_update="mm_exact",
        mm_damping=0.4,
        randomize_e_steps=True,
        e_steps_min=2,
        e_steps_max=4,
        e_steps_backprop_last=1,
        e_step_halt_tol=0.125,
    )
    captured: dict[str, object] = {}

    def capture_e_step(
        belief: BeliefState,
        *args: object,
        **kwargs: object,
    ) -> BeliefState:
        captured.update(kwargs)
        return belief

    monkeypatch.setattr(e_step_module, "e_step", capture_e_step)
    phi0 = model._apply_pos_phi(model.prior_bank.encode(token_ids).phi)
    rope = torch.eye(model.cfg.embed_dim).expand(token_ids.shape[1], -1, -1).clone()

    model._refine_s(token_ids, phi0, rope=rope)

    assert captured["e_step_update"] == "mm_exact"
    assert captured["mm_damping"] == 0.4
    assert captured["randomize_e_steps"] is True
    assert captured["e_steps_min"] == 2
    assert captured["e_steps_max"] == 4
    assert captured["e_steps_backprop_last"] == 1
    assert captured["e_step_halt_tol"] == 0.125
    assert captured["rope"] is rope
    assert captured["rope_on_cov"] is model.cfg.rope_full_gauge
    assert captured["rope_on_value"] is model.cfg.rope_on_value


def test_refine_s_rope_matches_direct_rotated_estep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.warns(UserWarning, match="gauge"):
        model, token_ids = _model_channel_case(pos_rotation="rope")
    original_e_step = e_step_module.e_step
    captured_args: tuple[object, ...] = ()
    captured_kwargs: dict[str, object] = {}

    def record_and_run(
        *args: object,
        **kwargs: object,
    ) -> BeliefState:
        nonlocal captured_args, captured_kwargs
        captured_args = args
        captured_kwargs = dict(kwargs)
        return original_e_step(*args, **kwargs)

    monkeypatch.setattr(e_step_module, "e_step", record_and_run)
    phi0 = model._apply_pos_phi(model.prior_bank.encode(token_ids).phi)
    rope = model._rope_rotation(token_ids.shape[1], token_ids.device)
    assert rope is not None

    s_mu1, s_sigma1 = model._refine_s(token_ids, phi0, rope=rope)
    direct = original_e_step(*captured_args, **captured_kwargs)
    unrotated_kwargs = dict(captured_kwargs)
    unrotated_kwargs["rope"] = None
    unrotated = original_e_step(*captured_args, **unrotated_kwargs)

    assert torch.equal(s_mu1, direct.mu)
    assert torch.equal(s_sigma1, direct.sigma)
    assert not torch.allclose(s_mu1, unrotated.mu, atol=1e-6, rtol=1e-6)


def test_q_and_s_randomized_depth_draw_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, token_ids = _model_channel_case(
        n_e_steps=4,
        randomize_e_steps=True,
        e_steps_min=1,
        e_steps_max=4,
    )
    original_randint = torch.randint
    draws: list[int] = []

    def tracked_randint(
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        draw = original_randint(*args, **kwargs)
        draws.append(int(draw.item()))
        return draw

    monkeypatch.setattr(torch, "randint", tracked_randint)
    runs: list[list[int]] = []
    for _ in range(2):
        start = len(draws)
        torch.manual_seed(0)
        model.forward_beliefs(token_ids)
        runs.append(draws[start:])

    assert runs[0] == runs[1]
    assert len(runs[0]) == 2
    assert runs[0][0] != runs[0][1]


def _mstep_prior_case(
    n_layers: int,
    **overrides: object,
) -> tuple[VFEModel, torch.Tensor, torch.Tensor]:
    torch.manual_seed(53)
    values = dict(
        vocab_size=11,
        embed_dim=4,
        n_heads=2,
        max_seq_len=4,
        n_layers=n_layers,
        n_e_steps=2,
        e_q_mu_lr=0.2,
        e_q_sigma_lr=0.08,
        e_phi_lr=0.0,
        prior_handoff_rho=0.65,
        prior_handoff_sigma=0.4,
        mass_phi=0.0,
        mstep_self_coupling_weight=0.7,
        seed=53,
    )
    values.update(overrides)
    model = VFEModel(VFE3Config(**values))
    token_ids = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.long)
    targets = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long)
    return model, token_ids, targets


def _manual_final_block_recurrence(
    token_ids: torch.Tensor,
    model:     VFEModel,
) -> tuple[BeliefState, BeliefState, tuple[torch.Tensor, torch.Tensor], BeliefState]:
    r"""Run the stack recurrence directly, retaining the live prior entering its last block."""
    cfg = model.cfg
    initial = model.prior_bank.encode(token_ids)
    initial = initial._replace(phi=model._apply_pos_phi(initial.phi))
    log_prior = model._attention_log_prior(token_ids.shape[1], token_ids.device)
    log_prior = model._fold_precision_bias(log_prior, initial.sigma)
    belief = initial
    mu_p, sigma_p = initial.mu, initial.sigma
    final_capture: dict[str, BeliefState] = {}
    final_prior: tuple[torch.Tensor, torch.Tensor] | None = None

    for layer_index in range(cfg.n_layers):
        is_final = layer_index == cfg.n_layers - 1
        if is_final:
            final_prior = (mu_p.clone(), sigma_p.clone())
        belief = vfe_block(
            belief,
            mu_p,
            sigma_p,
            model.group,
            cfg,
            log_prior=log_prior,
            block_norm=model.block_norm,
            lambda_beta=cfg.lambda_beta,
            capture=final_capture if is_final else None,
        )
        mu_p = (1.0 - cfg.prior_handoff_rho) * mu_p + cfg.prior_handoff_rho * belief.mu
        sigma_p = (
            (1.0 - cfg.prior_handoff_sigma) * sigma_p
            + cfg.prior_handoff_sigma * belief.sigma
        )

    assert final_prior is not None
    return belief, final_capture["converged"], final_prior, initial


def _constant_self_divergence(
    mu_q:    torch.Tensor,
    sigma_q: torch.Tensor,
    mu_p:    torch.Tensor,
    sigma_p: torch.Tensor,
    model:   VFEModel,
) -> torch.Tensor:
    cfg = model.cfg
    family = get_family(cfg.family)
    return self_divergence_for_alpha(
        family(mu_q, sigma_q),
        family(mu_p, sigma_p),
        alpha=cfg.renyi_order,
        kl_max=cfg.kl_max,
        eps=cfg.eps,
        divergence_family=cfg.divergence_family,
        lambda_alpha_mode=cfg.lambda_alpha_mode,
    ).mean()


def test_multilayer_capture_records_actual_final_block_prior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, token_ids, _ = _mstep_prior_case(n_layers=3)
    manual_out, _, expected_prior, initial = _manual_final_block_recurrence(token_ids, model)
    observed_priors: list[tuple[torch.Tensor, torch.Tensor]] = []
    original_vfe_block = stack_module.vfe_block

    def record_and_run(
        belief:  BeliefState,
        mu_p:    torch.Tensor,
        sigma_p: torch.Tensor,
        *args:   object,
        **kwargs: object,
    ) -> BeliefState:
        observed_priors.append((mu_p, sigma_p))
        return original_vfe_block(belief, mu_p, sigma_p, *args, **kwargs)

    monkeypatch.setattr(stack_module, "vfe_block", record_and_run)
    capture: dict[str, object] = {}
    out = vfe_stack(
        initial,
        initial.mu,
        initial.sigma,
        model.group,
        model.cfg,
        log_prior=model._attention_log_prior(token_ids.shape[1], token_ids.device),
        block_norm=model.block_norm,
        capture=capture,
    )

    assert torch.equal(out.mu, manual_out.mu)
    assert torch.equal(out.sigma, manual_out.sigma)
    captured_mu, captured_sigma = capture["final_block_prior"]
    expected_mu, expected_sigma = expected_prior
    actual_mu, actual_sigma = observed_priors[-1]
    assert torch.equal(captured_mu, expected_mu)
    assert torch.equal(captured_sigma, expected_sigma)
    assert torch.equal(captured_mu, actual_mu)
    assert torch.equal(captured_sigma, actual_sigma)
    assert captured_mu.data_ptr() != actual_mu.data_ptr()
    assert captured_sigma.data_ptr() != actual_sigma.data_ptr()


def test_multilayer_mstep_self_coupling_matches_manual_recurrence() -> None:
    model, token_ids, targets = _mstep_prior_case(n_layers=3)
    _, loss, ce = model(token_ids, targets)
    manual_out, q_converged, final_prior, initial = _manual_final_block_recurrence(
        token_ids,
        model,
    )
    mu_p, sigma_p = final_prior
    expected_sc = _constant_self_divergence(
        q_converged.mu,
        q_converged.sigma,
        mu_p,
        sigma_p,
        model,
    )

    pseudo_mu_p, pseudo_sigma_p = initial.mu, initial.sigma
    for _ in range(model.cfg.n_layers - 1):
        pseudo_mu_p = (
            (1.0 - model.cfg.prior_handoff_rho) * pseudo_mu_p
            + model.cfg.prior_handoff_rho * manual_out.mu
        )
        pseudo_sigma_p = (
            (1.0 - model.cfg.prior_handoff_sigma) * pseudo_sigma_p
            + model.cfg.prior_handoff_sigma * manual_out.sigma
        )
    pseudo_sc = _constant_self_divergence(
        q_converged.mu,
        q_converged.sigma,
        pseudo_mu_p,
        pseudo_sigma_p,
        model,
    )

    assert expected_sc > 1e-6
    assert not torch.allclose(expected_sc, pseudo_sc, atol=1e-7, rtol=1e-6)
    expected_loss = ce + model.cfg.mstep_self_coupling_weight * expected_sc
    assert torch.allclose(loss, expected_loss, atol=1e-6, rtol=1e-6)


def test_single_layer_mstep_self_coupling_is_unchanged() -> None:
    model, token_ids, targets = _mstep_prior_case(n_layers=1)
    _, loss, ce = model(token_ids, targets)
    _, q_converged, final_prior, _ = _manual_final_block_recurrence(token_ids, model)
    mu_p, sigma_p = final_prior
    expected_sc = _constant_self_divergence(
        q_converged.mu,
        q_converged.sigma,
        mu_p,
        sigma_p,
        model,
    )

    expected_loss = ce + model.cfg.mstep_self_coupling_weight * expected_sc
    assert torch.allclose(loss, expected_loss, atol=1e-6, rtol=1e-6)


def test_detach_single_layer_capture_preserves_prior_gradient() -> None:
    belief, mu_p_value, sigma_p_value, group = _truncation_case(seed=61)
    mu_p = mu_p_value.detach().clone().requires_grad_(True)
    sigma_p = sigma_p_value.detach().clone().requires_grad_(True)
    cfg = VFE3Config(
        vocab_size=7,
        embed_dim=2,
        n_heads=1,
        max_seq_len=3,
        n_layers=1,
        n_e_steps=2,
        e_q_mu_lr=0.2,
        e_q_sigma_lr=0.08,
        e_phi_lr=0.0,
        e_step_gradient="detach",
        mass_phi=0.0,
        mstep_self_coupling_weight=0.7,
        seed=61,
    )
    capture: dict[str, object] = {}

    with torch.no_grad():
        vfe_stack(
            belief,
            mu_p,
            sigma_p,
            group,
            cfg,
            e_step_gradient="detach",
            capture=capture,
        )

    captured_mu, captured_sigma = capture["final_block_prior"]
    q_converged = capture["converged"]
    assert torch.equal(captured_mu, mu_p)
    assert torch.equal(captured_sigma, sigma_p)
    assert captured_mu.data_ptr() != mu_p.data_ptr()
    assert captured_sigma.data_ptr() != sigma_p.data_ptr()
    assert captured_mu.requires_grad
    assert captured_sigma.requires_grad

    family = get_family(cfg.family)
    captured_sc = self_divergence_for_alpha(
        family(q_converged.mu.detach(), q_converged.sigma.detach()),
        family(captured_mu, captured_sigma),
        alpha=cfg.renyi_order,
        kl_max=cfg.kl_max,
        eps=cfg.eps,
        divergence_family=cfg.divergence_family,
        lambda_alpha_mode=cfg.lambda_alpha_mode,
    ).mean()
    reference_sc = self_divergence_for_alpha(
        family(q_converged.mu.detach(), q_converged.sigma.detach()),
        family(mu_p, sigma_p),
        alpha=cfg.renyi_order,
        kl_max=cfg.kl_max,
        eps=cfg.eps,
        divergence_family=cfg.divergence_family,
        lambda_alpha_mode=cfg.lambda_alpha_mode,
    ).mean()
    captured_grads = torch.autograd.grad(
        captured_sc,
        (mu_p, sigma_p),
        retain_graph=True,
    )
    reference_grads = torch.autograd.grad(reference_sc, (mu_p, sigma_p))

    for captured_grad, reference_grad in zip(captured_grads, reference_grads):
        assert torch.count_nonzero(reference_grad) > 0
        assert torch.equal(captured_grad, reference_grad)


def _manual_detach_final_prior(
    token_ids: torch.Tensor,
    model:     VFEModel,
) -> tuple[BeliefState, tuple[torch.Tensor, torch.Tensor], BeliefState]:
    r"""Run the live no-grad blocks while retaining the exact detached-state prior recurrence."""
    cfg = model.cfg
    initial = model.prior_bank.encode(token_ids)
    initial = initial._replace(phi=model._apply_pos_phi(initial.phi))
    log_prior = model._attention_log_prior(token_ids.shape[1], token_ids.device)
    log_prior = model._fold_precision_bias(log_prior, initial.sigma)
    live_belief = initial
    live_mu_p, live_sigma_p = initial.mu, initial.sigma
    reference_mu_p, reference_sigma_p = initial.mu, initial.sigma
    final_prior: tuple[torch.Tensor, torch.Tensor] | None = None
    final_capture: dict[str, BeliefState] = {}

    for layer_index in range(cfg.n_layers):
        is_final = layer_index == cfg.n_layers - 1
        if is_final:
            final_prior = (reference_mu_p, reference_sigma_p)
        with torch.no_grad():
            live_belief = vfe_block(
                live_belief,
                live_mu_p,
                live_sigma_p,
                model.group,
                cfg,
                log_prior=log_prior,
                block_norm=model.block_norm,
                lambda_beta=cfg.lambda_beta,
                e_step_gradient="detach",
                capture=final_capture if is_final else None,
            )
            live_mu_p = (
                (1.0 - cfg.prior_handoff_rho) * live_mu_p
                + cfg.prior_handoff_rho * live_belief.mu
            )
            live_sigma_p = (
                (1.0 - cfg.prior_handoff_sigma) * live_sigma_p
                + cfg.prior_handoff_sigma * live_belief.sigma
            )
        if not is_final:
            reference_mu_p = (
                (1.0 - cfg.prior_handoff_rho) * reference_mu_p
                + cfg.prior_handoff_rho * live_belief.mu.detach()
            )
            reference_sigma_p = (
                (1.0 - cfg.prior_handoff_sigma) * reference_sigma_p
                + cfg.prior_handoff_sigma * live_belief.sigma.detach()
            )

    assert final_prior is not None
    return final_capture["converged"], final_prior, initial


def test_detach_multilayer_mstep_gradient_matches_exact_recurrence() -> None:
    model, token_ids, _ = _mstep_prior_case(
        n_layers=3,
        e_step_gradient="detach",
        pos_phi="none",
    )
    targets = torch.full_like(token_ids, -100)
    q_converged, final_prior, initial = _manual_detach_final_prior(token_ids, model)
    expected_mu_p, expected_sigma_p = final_prior
    capture: dict[str, object] = {}

    model.forward_beliefs(token_ids, capture=capture)
    captured_mu, captured_sigma = capture["final_block_prior"]
    assert torch.equal(capture["converged"].mu, q_converged.mu)
    assert torch.equal(capture["converged"].sigma, q_converged.sigma)
    assert torch.equal(captured_mu, expected_mu_p)
    assert torch.equal(captured_sigma, expected_sigma_p)

    expected_sc = _constant_self_divergence(
        q_converged.mu.detach(),
        q_converged.sigma.detach(),
        expected_mu_p,
        expected_sigma_p,
        model,
    )
    expected_term = model.cfg.mstep_self_coupling_weight * expected_sc

    pseudo_mu_p, pseudo_sigma_p = initial.mu, initial.sigma
    for _ in range(model.cfg.n_layers - 1):
        pseudo_mu_p = (
            (1.0 - model.cfg.prior_handoff_rho) * pseudo_mu_p
            + model.cfg.prior_handoff_rho * q_converged.mu.detach()
        )
        pseudo_sigma_p = (
            (1.0 - model.cfg.prior_handoff_sigma) * pseudo_sigma_p
            + model.cfg.prior_handoff_sigma * q_converged.sigma.detach()
        )
    pseudo_sc = _constant_self_divergence(
        q_converged.mu.detach(),
        q_converged.sigma.detach(),
        pseudo_mu_p,
        pseudo_sigma_p,
        model,
    )
    pseudo_term = model.cfg.mstep_self_coupling_weight * pseudo_sc

    prior_parameter = model.prior_bank.mu_embed
    expected_grad, = torch.autograd.grad(expected_term, prior_parameter, retain_graph=True)
    pseudo_grad, = torch.autograd.grad(pseudo_term, prior_parameter)

    _, actual_loss, ce = model(token_ids, targets)
    actual_grad, = torch.autograd.grad(actual_loss, prior_parameter)

    assert torch.equal(ce, torch.zeros_like(ce))
    assert torch.allclose(actual_loss, expected_term.detach(), atol=1e-6, rtol=1e-6)
    assert torch.count_nonzero(expected_grad) > 0
    assert not torch.allclose(expected_grad, pseudo_grad, atol=1e-7, rtol=1e-6)
    assert torch.allclose(actual_grad, expected_grad, atol=1e-6, rtol=1e-6)
