r"""Curated regression tests for E-step and model-channel inference contracts."""

import importlib

import pytest
import torch

from vfe3.belief import BeliefState
from vfe3.config import VFE3Config
from vfe3.free_energy import attention_weights
from vfe3.geometry.groups import GaugeGroup, get_group
from vfe3.inference.e_step import build_belief_transport, e_step
from vfe3.model.model import VFEModel


e_step_module = importlib.import_module("vfe3.inference.e_step")


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
