r"""Focused regressions for the 2026-07-15 objective/state-transition audit findings."""

import inspect
import math
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch.utils.data import DataLoader

from vfe3.config import VFE3Config
from vfe3.data.datasets import TokenWindows
from vfe3.free_energy import attention_weights, free_energy
from vfe3.inference.e_step import free_energy_value
from vfe3.model.block import _as_coeff
from vfe3.model.model import VFEModel
from vfe3.run_artifacts import RunArtifacts
from vfe3.train import (
    _loader_data_identity,
    _maybe_metropolis_omega,
    build_optimizer,
    lr_lambda,
    train,
    train_step,
)


def _model(*, seed: int = 0, perturb: bool = True, **overrides) -> VFEModel:
    base = dict(
        vocab_size=6,
        embed_dim=4,
        n_heads=1,
        max_seq_len=4,
        n_layers=1,
        n_e_steps=2,
        e_q_mu_lr=0.4,
        e_phi_lr=0.0,
        mass_phi=0.0,
        pos_phi="none",
        gauge_group="glk",
        use_head_mixer=False,
    )
    base.update(overrides)
    torch.manual_seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(VFE3Config(**base))
    if perturb:
        with torch.no_grad():
            model.prior_bank.sigma_log_embed.add_(
                0.4 * torch.randn_like(model.prior_bank.sigma_log_embed)
            )
            if hasattr(model.prior_bank, "phi_embed"):
                model.prior_bank.phi_embed.add_(
                    0.3 * torch.randn_like(model.prior_bank.phi_embed)
                )
            if hasattr(model.prior_bank, "s_mu_embed"):
                model.prior_bank.s_mu_embed.add_(
                    0.5 * torch.randn_like(model.prior_bank.s_mu_embed)
                )
                model.prior_bank.s_sigma_log_embed.add_(
                    0.3 * torch.randn_like(model.prior_bank.s_sigma_log_embed)
                )
    return model


def _tokens() -> torch.Tensor:
    return torch.randperm(6, generator=torch.Generator().manual_seed(5))[:3].unsqueeze(0)


def _belief_only_metropolis_energy(
    model: VFEModel,
    belief,
    context,
    *,
    mode: str,
) -> float:
    cfg = model.cfg
    device = belief.mu.device
    log_prior = model._effective_beta_log_prior(belief, context.prior)
    gauge_parameterization = "omega_direct" if mode == "omega" else "phi"
    with torch.no_grad():
        return free_energy_value(
            belief,
            context.mu_p,
            context.sigma_p,
            model.group,
            tau=context.tau,
            renyi_order=cfg.renyi_order,
            value=cfg.lambda_alpha,
            b0=_as_coeff(cfg.b0, device),
            c0=_as_coeff(cfg.c0, device),
            lambda_beta=cfg.lambda_beta,
            lambda_twohop=cfg.lambda_twohop,
            kl_max=cfg.kl_max,
            eps=cfg.eps,
            include_attention_entropy=cfg.include_attention_entropy,
            family=cfg.family,
            divergence_family=cfg.divergence_family,
            lambda_alpha_mode=cfg.lambda_alpha_mode,
            gauge_parameterization=gauge_parameterization,
            log_prior=log_prior,
            transport_mode=cfg.transport_mode,
            connection_W=getattr(model, "connection_W", None),
            connection_M=getattr(model, "connection_M", None),
            connection_L=getattr(model, "connection_L", None),
            cocycle_relaxation=cfg.cocycle_relaxation,
            link_alpha=cfg.link_alpha,
            link_soft_cap=cfg.link_soft_cap,
            clamp_monitor=cfg.transport_clamp_monitor,
            transport_mean_per_head=True,
            rope=context.rope,
            rope_on_cov=cfg.rope_full_gauge,
            rope_on_value=cfg.rope_on_value,
            exp_fp64_mode=cfg.exp_fp64_mode,
            exp_fp64_norm_threshold=cfg.exp_fp64_norm_threshold,
        ).item()


def test_metropolis_uses_complete_joint_objective_when_gamma_reverses_belief_decision() -> None:
    model = _model(
        gauge_parameterization="phi",
        phi_reflection="metropolis",
        lambda_h=0.5,
        lambda_gamma=3.0,
        kappa_gamma=0.25,
        s_e_step=False,
    )
    tokens = _tokens()
    context = model._metropolis_prepare(tokens, mode="phi")
    token_id = 5
    trial = model._metropolis_trial_belief(
        context.belief, tokens, token_id, mode="phi"
    )
    belief_delta = (
        _belief_only_metropolis_energy(model, trial, context, mode="phi")
        - _belief_only_metropolis_energy(model, context.belief, context, mode="phi")
    )
    complete_delta = model._metropolis_delta_f(context, token_id, mode="phi")

    assert belief_delta < 0.0
    assert complete_delta > 0.0


def test_gamma_energy_changes_when_rope_transport_is_active() -> None:
    rope = _model(lambda_gamma=1.0, pos_rotation="rope")
    plain = _model(lambda_gamma=1.0, pos_rotation="none")
    tokens = _tokens()
    belief_phi = rope._apply_pos_phi(rope.prior_bank.encode(tokens).phi)
    model_phi = rope._resolve_model_frame(tokens, belief_phi)

    energy_rope = rope._gamma_energy(tokens, model_phi)[0]
    energy_plain = plain._gamma_energy(tokens, model_phi)[0]

    assert not torch.allclose(energy_rope, energy_plain)


def test_gamma_prior_mix_normalizes_after_applying_active_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model(
        lambda_gamma=1.0,
        gamma_as_beta_prior=True,
        gamma_prior_weight=0.5,
    )
    tokens = _tokens()
    n = tokens.shape[-1]
    energy = torch.zeros(1, n, n)
    monkeypatch.setattr(
        model,
        "_gamma_energy",
        lambda *_args, **_kwargs: (energy, 1.0, None),
    )
    log_prior = torch.tensor(
        [[0.0, -math.inf, -math.inf], [0.0, 0.0, -math.inf], [0.0, 0.0, 0.0]]
    )
    phi = model.prior_bank.encode(tokens).phi

    mixed = model._fold_gamma_prior(log_prior, tokens, phi)
    support = torch.isfinite(log_prior).unsqueeze(0)
    row_sums = torch.exp(mixed).masked_fill(~support, 0.0).sum(dim=-1)

    torch.testing.assert_close(row_sums, torch.ones_like(row_sums))


def test_entropy_tail_derivative_matches_analytic_envelope_coefficient() -> None:
    tail_energy = math.log(1.0 / 1.8467e-14)
    energy = torch.tensor(
        [[0.0, tail_energy], [0.0, tail_energy]],
        dtype=torch.float64,
        requires_grad=True,
    )
    self_div = torch.zeros(2, dtype=torch.float64)
    alpha = torch.ones_like(self_div)
    scalar = free_energy(self_div, energy, alpha, tau=1.0)
    differentiated = torch.autograd.grad(scalar, energy)[0]
    expected = attention_weights(energy.detach(), tau=1.0)

    torch.testing.assert_close(differentiated, expected, rtol=1e-8, atol=1e-20)


def test_scheduler_does_not_advance_after_rejected_update() -> None:
    model = _model(max_steps=4, warmup_steps=1)
    optimizer = build_optimizer(model, model.cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_lambda(step, model.cfg)
    )
    model.prior_bank.mu_embed.register_hook(
        lambda grad: torch.full_like(grad, float("nan"))
    )
    tokens = _tokens().repeat(2, 1)
    targets = torch.roll(tokens, shifts=-1, dims=-1)
    before_epoch = scheduler.last_epoch
    status = {}

    train_step(
        model,
        optimizer,
        scheduler,
        tokens,
        targets,
        status_out=status,
    )

    assert status == {"did_step": False}
    assert scheduler.last_epoch == before_epoch


def test_nonfinite_loss_blocks_enabled_gradscaler_step() -> None:
    class FiniteGradientInfiniteLoss(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.cfg = SimpleNamespace(
                learnable_r=False,
                phi_mstep_max_matrix_norm=None,
            )

        def forward(
            self,
            _tokens:         torch.Tensor,
            _targets:        torch.Tensor,
            *,
            estep_grad_out:  Any = None,
        ) -> tuple[None, torch.Tensor, None]:
            del estep_grad_out
            return None, self.weight + float("inf"), None

    model = FiniteGradientInfiniteLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    scaler = torch.amp.GradScaler(device="cpu", enabled=True, init_scale=2.0)
    before_weight = model.weight.detach().clone()
    before_epoch = scheduler.last_epoch
    before_scale = float(scaler.get_scale())
    status: dict[str, bool] = {}

    result = train_step(
        model,
        optimizer,
        scheduler,
        torch.zeros(1, 1, dtype=torch.long),
        torch.zeros(1, 1, dtype=torch.long),
        grad_clip=0.0,
        scaler=scaler,
        status_out=status,
    )

    assert math.isinf(result)
    assert status == {"did_step": False}
    assert torch.equal(model.weight, before_weight)
    assert scheduler.last_epoch == before_epoch
    assert {group["successful_updates"] for group in optimizer.param_groups} == {0}
    assert float(scaler.get_scale()) == before_scale


def test_gradscaler_overflow_with_nonfinite_loss_backs_off_and_reports_gradient() -> None:
    class InfiniteGradientInfiniteLoss(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.cfg = SimpleNamespace(learnable_r=False, phi_mstep_max_matrix_norm=None)

        def forward(self, _tokens, _targets, *, estep_grad_out=None):
            del estep_grad_out
            return None, self.weight * float("inf"), None

    model = InfiniteGradientInfiniteLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    scaler = torch.amp.GradScaler(
        device="cpu", enabled=True, init_scale=8.0, backoff_factor=0.5)
    metrics: dict[str, float] = {}

    train_step(
        model,
        optimizer,
        scheduler,
        torch.zeros(1, 1, dtype=torch.long),
        torch.zeros(1, 1, dtype=torch.long),
        scaler=scaler,
        metrics_out=metrics,
    )

    assert float(scaler.get_scale()) == 4.0
    assert metrics["grad_finite"] == 0.0
    assert metrics["step_skipped"] == 1.0


def test_finite_unlogged_gradscaler_step_does_not_scan_every_parameter_gradient(monkeypatch) -> None:
    class ScalarModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.cfg = SimpleNamespace(learnable_r=False, phi_mstep_max_matrix_norm=None)

        def forward(self, _tokens, _targets, *, estep_grad_out=None):
            del estep_grad_out
            return None, self.weight.square(), None

    model = ScalarModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    scaler = torch.amp.GradScaler(device="cpu", enabled=True, init_scale=8.0)

    def forbidden_explicit_scan(*_args, **_kwargs):
        raise AssertionError("ordinary enabled-GradScaler steps must delegate overflow detection")

    monkeypatch.setattr(torch, "isfinite", forbidden_explicit_scan)
    train_step(
        model,
        optimizer,
        scheduler,
        torch.zeros(1, 1, dtype=torch.long),
        torch.zeros(1, 1, dtype=torch.long),
        grad_clip=0.0,
        scaler=scaler,
    )

    assert model.weight < 1.0


def test_successful_update_clock_persists_in_optimizer_state() -> None:
    model = _model(max_steps=4, warmup_steps=1)
    optimizer = build_optimizer(model, model.cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_lambda(step, model.cfg)
    )
    tokens = _tokens().repeat(2, 1)
    targets = torch.roll(tokens, shifts=-1, dims=-1)

    train_step(model, optimizer, scheduler, tokens, targets)
    model.prior_bank.mu_embed.register_hook(
        lambda grad: torch.full_like(grad, float("nan"))
    )
    train_step(model, optimizer, scheduler, tokens, targets)

    assert {group["successful_updates"] for group in optimizer.param_groups} == {1}
    restored_model = _model(max_steps=4, warmup_steps=1)
    restored_optimizer = build_optimizer(restored_model, restored_model.cfg)
    restored_optimizer.load_state_dict(optimizer.state_dict())
    assert {group["successful_updates"] for group in restored_optimizer.param_groups} == {1}


def test_resumed_scheduler_uses_persisted_successful_update_clock(
    tmp_path:    Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model(max_steps=6, warmup_steps=1)
    optimizer = build_optimizer(model, model.cfg)
    for group in optimizer.param_groups:
        group["successful_updates"] = 2
    loader = DataLoader(
        TokenWindows(torch.tensor([5, 0, 1, 5]), seq_len=3),
        batch_size=1,
        shuffle=False,
        drop_last=True,
    )
    artifacts = RunArtifacts(tmp_path / "source", model.cfg, model)
    checkpoint = artifacts.save_checkpoint(
        5,
        model,
        optimizer,
        model.cfg,
        metropolis_generator=torch.Generator().manual_seed(model.cfg.seed),
        data_state={
            "epoch_start_generator_state": None,
            "batches_consumed":            0,
            "epoch":                       5,
            "data_identity":               _loader_data_identity(loader, model.cfg.vocab_size),
        },
    )

    observed_last_epochs = []
    real_lambda_lr = torch.optim.lr_scheduler.LambdaLR

    def lambda_lr_spy(*args: Any, **kwargs: Any) -> torch.optim.lr_scheduler.LambdaLR:
        observed_last_epochs.append(kwargs.get("last_epoch", -1))
        return real_lambda_lr(*args, **kwargs)

    monkeypatch.setattr(torch.optim.lr_scheduler, "LambdaLR", lambda_lr_spy)
    resumed = _model(max_steps=6, warmup_steps=1)
    train(
        resumed,
        loader,
        resumed.cfg,
        n_steps=6,
        resume_from=checkpoint,
        generate_samples=False,
    )

    assert observed_last_epochs == [1]


def test_outer_estep_autocast_and_inner_oracle_fp32_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vfe3.gradients.oracle as oracle
    import vfe3.model.block as block

    outer_autocast = []
    inner_calls = []
    real_e_step = block.e_step
    real_pairwise_energy = oracle.pairwise_energy

    def e_step_spy(*args: Any, **kwargs: Any) -> Any:
        belief = args[0]
        outer_autocast.append(torch.is_autocast_enabled(belief.mu.device.type))
        return real_e_step(*args, **kwargs)

    def pairwise_spy(
        query:  Any,
        target: Any,
        *args:  Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        inner_calls.append((
            torch.is_autocast_enabled(query.mu.device.type),
            query.mu.dtype,
            query.sigma.dtype,
            target.mu.dtype,
            target.sigma.dtype,
        ))
        return real_pairwise_energy(query, target, *args, **kwargs)

    monkeypatch.setattr(block, "e_step", e_step_spy)
    monkeypatch.setattr(oracle, "pairwise_energy", pairwise_spy)
    model = _model(amp_dtype="bf16", gradient_mode="smoothing")
    tokens = _tokens().repeat(2, 1)
    targets = torch.roll(tokens, shifts=-1, dims=-1)

    _, loss, _ = model(tokens, targets)

    assert torch.isfinite(loss)
    assert outer_autocast and all(outer_autocast)
    assert inner_calls
    assert all(not call[0] for call in inner_calls)
    assert all(dtype == torch.float32 for call in inner_calls for dtype in call[1:])


def test_rejected_update_cannot_mutate_metropolis_state_or_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert "did_step" in inspect.signature(_maybe_metropolis_omega).parameters
    model = _model(
        gauge_parameterization="phi",
        phi_reflection="metropolis",
        omega_metropolis_every=1,
    )
    calls = []

    def metropolis_spy(
        token_ids: torch.Tensor,

        *,
        generator: torch.Generator,
    ) -> dict:
        calls.append(token_ids.clone())
        torch.rand((), generator=generator)
        model.prior_bank.reflection_sign[0] *= -1.0
        return {}

    monkeypatch.setattr(model, "metropolis_omega_step", metropolis_spy)
    tokens = _tokens()
    generator = torch.Generator().manual_seed(11)
    rng_before = generator.get_state().clone()
    signs_before = model.prior_bank.reflection_sign.clone()

    _maybe_metropolis_omega(
        model,
        tokens,
        step=0,
        generator=generator,
        did_step=False,
    )

    assert calls == []
    assert torch.equal(generator.get_state(), rng_before)
    assert torch.equal(model.prior_bank.reflection_sign, signs_before)


def test_diagnostics_report_passive_direct_gamma_frame_gradient() -> None:
    model = _model(lambda_gamma=1.0, s_e_step=False)
    diagnostics = model.diagnostics(_tokens())

    assert diagnostics["gamma_direct_frame_grad_active"] == 0.0
