r"""Round-3 sync fusion (audit 2026-07-01): train_step defers the step-loss host read on the
default grad_accum_steps==1 path and fuses it with the F1 finite-gradient flag into ONE D2H
transfer, restoring exactly one unconditional CPU-GPU sync per training step. These tests pin
that the fusion preserves the F1 gate semantics -- same skip_step decisions, same metrics keys
(step_skipped / grad_finite / loss_finite), a real finite Python float returned -- on every
path (skip, normal, silent, grad accumulation)."""

import math

import torch

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel
from vfe3.train import build_optimizer, lr_lambda, train_step


def _tiny_setup(seed=0, batch=2):
    torch.manual_seed(seed)
    cfg = VFE3Config(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                     n_e_steps=1, e_q_mu_lr=0.1, e_phi_lr=0.0, m_phi_lr=0.0,
                     warmup_steps=1, max_steps=4)
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, cfg))
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, 6, (batch, 8), generator=g)
    targets = torch.randint(0, 6, (batch, 8), generator=g)
    return model, opt, sched, tokens, targets


def test_fused_gate_skips_on_nan_grad_and_leaves_params_unchanged():
    # (a) finite scalar loss + a poisoned NaN parameter gradient: the fused transfer must
    # carry BOTH the loss value and the grad_finite=False flag, skip the optimizer step, and
    # leave every parameter byte-identical.
    model, opt, sched, tokens, targets = _tiny_setup()
    model.prior_bank.mu_embed.register_hook(lambda grad: torch.full_like(grad, float("nan")))
    before = [p.detach().clone() for p in model.parameters()]
    metrics = {}
    loss = train_step(model, opt, sched, tokens, targets, grad_clip=1.0, metrics_out=metrics)
    assert isinstance(loss, float) and math.isfinite(loss)
    assert metrics["loss_finite"] == 1.0                        # the scalar loss WAS finite
    assert metrics["grad_finite"] == 0.0                        # ...but the gradient was not
    assert metrics["step_skipped"] == 1.0                       # so the optimizer step was skipped
    for p, b in zip(model.parameters(), before):                # skipped step touches NO parameter
        assert torch.equal(p.detach(), b)


def test_fused_gate_takes_step_on_finite_batch():
    # (b) normal finite batch: step taken, returned step_loss is a finite Python float.
    model, opt, sched, tokens, targets = _tiny_setup()
    train_step(model, opt, sched, tokens, targets, grad_clip=1.0)   # warmup step: LR ramps off 0
    before = [p.detach().clone() for p in model.parameters()]
    metrics = {}
    loss = train_step(model, opt, sched, tokens, targets, grad_clip=1.0, metrics_out=metrics)
    assert isinstance(loss, float) and math.isfinite(loss)
    assert metrics["loss_finite"] == 1.0
    assert metrics["grad_finite"] == 1.0
    assert metrics["step_skipped"] == 0.0
    assert any(not torch.equal(p.detach(), b)                   # the step actually moved a parameter
               for p, b in zip(model.parameters(), before))


def test_silent_default_path_returns_float():
    # metrics_out=None (the silent default): the deferred read must still resolve step_loss
    # to a real finite float before train_step returns.
    model, opt, sched, tokens, targets = _tiny_setup()
    loss = train_step(model, opt, sched, tokens, targets, grad_clip=1.0)
    assert isinstance(loss, float) and math.isfinite(loss)


def test_grad_accum_path_unchanged():
    # (c) grad_accum_steps=2: the accumulation branch keeps its per-microbatch Python-float
    # step_loss and its plain grad-finite sync; the fused-read fallback must not clobber it.
    model, opt, sched, tokens, targets = _tiny_setup(batch=4)
    metrics = {}
    loss = train_step(model, opt, sched, tokens, targets, grad_clip=1.0,
                      grad_accum_steps=2, metrics_out=metrics)
    assert isinstance(loss, float) and math.isfinite(loss)
    assert metrics["loss_finite"] == 1.0
    assert metrics["grad_finite"] == 1.0
    assert metrics["step_skipped"] == 0.0
