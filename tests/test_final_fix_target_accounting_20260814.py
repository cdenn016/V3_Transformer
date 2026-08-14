from __future__ import annotations

import math

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel
from vfe3.train import build_optimizer, lr_lambda, train_step


def _train_case() -> tuple[
    VFEModel,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LambdaLR,
    torch.Tensor,
    torch.Tensor,
]:
    torch.manual_seed(20260814)
    cfg = VFE3Config(
        vocab_size=8,
        embed_dim=4,
        n_heads=2,
        max_seq_len=4,
        n_layers=1,
        n_e_steps=1,
        e_q_mu_lr=0.1,
        e_phi_lr=0.0,
        m_phi_lr=0.0,
        warmup_steps=1,
        max_steps=4,
    )
    model = VFEModel(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_lambda(step, cfg)
    )
    tokens = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)
    targets = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
    return model, optimizer, scheduler, tokens, targets


def test_silent_train_step_has_one_scalar_host_materialization(monkeypatch) -> None:
    """A metrics-silent step must retain the existing single post-backward barrier."""
    model, optimizer, scheduler, tokens, targets = _train_case()
    original_tolist = torch.Tensor.tolist
    materializations: list[tuple[torch.dtype, tuple[int, ...]]] = []

    def tracked_tolist(tensor: torch.Tensor):
        materializations.append((tensor.dtype, tuple(tensor.shape)))
        return original_tolist(tensor)

    monkeypatch.setattr(torch.Tensor, "tolist", tracked_tolist)

    loss = train_step(
        model,
        optimizer,
        scheduler,
        tokens,
        targets,
        grad_clip=1.0,
        metrics_out=None,
    )

    assert math.isfinite(loss)
    assert materializations == [(torch.float64, (4,))]


def test_metrics_step_reports_exact_expected_scored_excluded_partition() -> None:
    model, optimizer, scheduler, tokens, targets = _train_case()
    targets = targets.clone()
    targets[0, 2] = -100
    metrics: dict[str, float | int] = {}

    train_step(
        model,
        optimizer,
        scheduler,
        tokens,
        targets,
        grad_clip=1.0,
        metrics_out=metrics,
    )

    assert {
        name: metrics[name]
        for name in ("expected_targets", "scored_targets", "excluded_targets")
    } == {
        "expected_targets": 4,
        "scored_targets": 3,
        "excluded_targets": 1,
    }


def test_exact_device_partition_preserves_int64_above_float64_boundary() -> None:
    from vfe3.train import _target_accounting_from_device

    expected = 2**53 + 1
    assert _target_accounting_from_device(
        torch.tensor(expected, dtype=torch.int64),
        torch.tensor(expected - 7, dtype=torch.int64),
        torch.tensor(7, dtype=torch.int64),
    ) == {
        "expected_targets": expected,
        "scored_targets": expected - 7,
        "excluded_targets": 7,
    }


def test_silent_train_step_rejects_invalid_decoder_partition(monkeypatch) -> None:
    from dataclasses import replace

    model, optimizer, scheduler, tokens, targets = _train_case()
    original_forward = model.forward

    def invalid_forward(*args, **kwargs):
        output, loss, stats = original_forward(*args, **kwargs)
        assert stats.excluded_tokens is not None
        return output, loss, replace(
            stats,
            excluded_tokens=stats.excluded_tokens + 1,
        )

    monkeypatch.setattr(model, "forward", invalid_forward)

    with pytest.raises(
        RuntimeError,
        match=r"expected_targets == scored_targets \+ excluded_targets",
    ):
        train_step(
            model,
            optimizer,
            scheduler,
            tokens,
            targets,
            grad_clip=1.0,
            metrics_out=None,
        )
