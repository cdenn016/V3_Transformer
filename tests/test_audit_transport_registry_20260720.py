r"""Audit regressions for registry-owned trainable transport state (M9)."""

from pathlib import Path

import pytest
import torch
from torch import nn

from vfe3.config import VFE3Config
from vfe3.geometry.groups import GaugeGroup, get_group
from vfe3.geometry.transport import (
    _TRANSPORTS,
    _TRANSPORT_BATCH_INDEPENDENT,
    _TRANSPORT_NEEDS_MU,
    _TRANSPORT_NEEDS_SIGMA,
    compute_transport_operators,
    get_transport_registration,
    register_transport,
)
from vfe3.gradients.kernels import uses_kernel_route
from vfe3.inference.e_step import build_belief_transport
from vfe3.model.model import VFEModel
from vfe3.train import build_optimizer
from vfe3.viz.extract import e_step_belief_trace


def _tiny_cfg(transport_mode: str = "flat") -> VFE3Config:
    return VFE3Config(
        vocab_size=17,
        embed_dim=4,
        n_heads=2,
        max_seq_len=4,
        n_layers=1,
        n_e_steps=1,
        e_phi_lr=0.0,
        pos_phi="none",
        transport_mode=transport_mode,
    )


@pytest.mark.parametrize(
    ("mode", "serialization_key", "shape"),
    [
        ("regime_ii", "connection_W", (8, 4, 4)),
        ("regime_ii_covariant", "connection_M", (8, 3)),
        ("regime_ii_link", "connection_L", (4, 4, 8)),
        ("regime_ii_link_charted", "connection_L", (4, 4, 8)),
    ],
)
def test_existing_stateful_transports_preserve_parameter_and_checkpoint_contract(
    mode:              str,
    serialization_key: str,
    shape:             tuple[int, ...],
) -> None:
    registration = get_transport_registration(mode)
    model = VFEModel(_tiny_cfg(mode))

    assert registration.state_builder is not None
    assert registration.serialization_keys == (serialization_key,)
    assert tuple(model.transport_state) == (serialization_key,)

    parameter = model.transport_state[serialization_key]
    assert parameter is getattr(model, serialization_key)
    assert isinstance(parameter, nn.Parameter)
    assert tuple(parameter.shape) == shape
    assert torch.equal(parameter, torch.zeros(shape))

    connection_keys = [key for key in model.state_dict() if key.startswith("connection_")]
    assert connection_keys == [serialization_key]
    assert f"transport_state.{serialization_key}" not in model.state_dict()

    optimizer = build_optimizer(model, model.cfg)
    grouped = [item for group in optimizer.param_groups for item in group["params"]]
    assert sum(item is parameter for item in grouped) == 1

    restored = VFEModel(_tiny_cfg(mode))
    restored.load_state_dict(model.state_dict(), strict=True)
    assert torch.equal(restored.transport_state[serialization_key], parameter)


def test_flat_transport_remains_state_free_and_output_compatible() -> None:
    registration = get_transport_registration("flat")
    model = VFEModel(_tiny_cfg())

    assert registration.state_builder is None
    assert registration.serialization_keys == ()
    assert model.transport_state == {}
    assert not any(key.startswith("connection_") for key in model.state_dict())

    group = get_group("so_k")(K=4)
    generator = torch.Generator().manual_seed(20260720)
    phi = 0.2 * torch.randn(2, 3, group.generators.shape[0], generator=generator)
    expected = compute_transport_operators(phi, group)["Omega"]
    actual = build_belief_transport(
        phi,
        group,
        transport_mode="flat",
        transport_state=model.transport_state,
    )

    assert isinstance(actual, torch.Tensor)
    assert torch.equal(actual, expected)


def test_registration_rejects_state_keys_reserved_by_the_builder_call() -> None:
    mode = "_audit_reserved_transport_state_20260720"

    def _build_invalid_state(
        cfg:   VFE3Config,
        group: GaugeGroup,
    ) -> dict[str, nn.Parameter]:
        del cfg, group
        return {"mu": nn.Parameter(torch.tensor(0.0, dtype=torch.float32))}

    with pytest.raises(ValueError, match="reserved transport-builder keyword"):
        register_transport(
            mode,
            covariance_class="test-only",
            state_builder=_build_invalid_state,
            serialization_keys=("mu",),
        )

    assert mode not in _TRANSPORTS


@pytest.mark.parametrize(
    ("mode", "unexpected_metric"),
    [
        ("regime_ii", "connection_w_offdiag_norm"),
        ("regime_ii_covariant", "connection_m_offdiag_norm"),
    ],
)
def test_square_parameter_shapes_do_not_invent_direct_link_diagnostics(
    mode:              str,
    unexpected_metric: str,
) -> None:
    cfg = VFE3Config(
        vocab_size=17,
        embed_dim=3,
        n_heads=1,
        max_seq_len=3,
        n_layers=1,
        n_e_steps=1,
        e_phi_lr=0.0,
        pos_phi="none",
        gauge_group="so_k",
        transport_mode=mode,
    )
    model = VFEModel(cfg)
    diagnostics = model.diagnostics(torch.tensor([[1, 2, 3]], dtype=torch.long))

    assert unexpected_metric not in diagnostics


def test_registered_belief_dependent_state_constructs_optimizes_and_reaches_callable() -> None:
    mode = "_audit_stateful_transport_20260720"
    serialization_key = "connection_probe"
    received: list[torch.Tensor] = []

    def _build_probe_state(
        cfg:   VFE3Config,
        group: GaugeGroup,
    ) -> dict[str, nn.Parameter]:
        del cfg, group
        return {serialization_key: nn.Parameter(torch.tensor(0.125, dtype=torch.float32))}

    @register_transport(
        mode,
        covariance_class="test-only",
        needs_mu=True,
        state_builder=_build_probe_state,
        serialization_keys=(serialization_key,),
    )
    def _build_probe_transport(
        phi:   torch.Tensor,
        group: GaugeGroup,

        *,
        mu:               torch.Tensor,
        connection_probe: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        received.append(connection_probe)
        built = compute_transport_operators(
            phi,
            group,
            gauge_mode=kwargs.get("gauge_mode", "learned"),
        )
        built["Omega"] = built["Omega"] * (1.0 + connection_probe * mu.square().mean())
        return built

    try:
        model = VFEModel(_tiny_cfg(mode))
        parameter = model.transport_state[serialization_key]

        assert parameter is getattr(model, serialization_key)
        assert model.state_dict()[serialization_key] is not None
        assert mode not in Path("vfe3/model/model.py").read_text(encoding="utf-8")
        assert mode not in Path("vfe3/inference/e_step.py").read_text(encoding="utf-8")
        assert serialization_key not in Path("vfe3/model/model.py").read_text(encoding="utf-8")
        assert serialization_key not in Path("vfe3/inference/e_step.py").read_text(encoding="utf-8")

        optimizer = build_optimizer(model, model.cfg)
        grouped = [item for group in optimizer.param_groups for item in group["params"]]
        assert sum(item is parameter for item in grouped) == 1

        tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        targets = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
        before = parameter.detach().clone()
        optimizer.zero_grad(set_to_none=True)
        _, loss, _ = model(tokens, targets)
        loss.backward()

        assert not uses_kernel_route(
            renyi_order=model.cfg.renyi_order,
            gradient_mode=model.cfg.gradient_mode,
            family=model.cfg.family,
            divergence_family=model.cfg.divergence_family,
            include_attention_entropy=model.cfg.include_attention_entropy,
            transport_mode=mode,
        )
        assert model.cfg.oracle_unroll_grad is True
        assert any(item is parameter for item in received)
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0

        optimizer.step()
        assert not torch.equal(parameter, before)

        trace = e_step_belief_trace(model, tokens, n_iter=1)
        assert torch.isfinite(trace["free_energy"]).all()
    finally:
        _TRANSPORTS.pop(mode, None)
        _TRANSPORT_NEEDS_MU.discard(mode)
        _TRANSPORT_NEEDS_SIGMA.discard(mode)
        _TRANSPORT_BATCH_INDEPENDENT.discard(mode)


def test_registered_sigma_dependent_state_routes_sigma_and_trains() -> None:
    mode = "_audit_sigma_stateful_transport_20260720"
    serialization_key = "connection_probe_sigma"
    received: list[torch.Tensor] = []

    def _build_probe_state(
        cfg:   VFE3Config,
        group: GaugeGroup,
    ) -> dict[str, nn.Parameter]:
        del cfg, group
        return {serialization_key: nn.Parameter(torch.tensor(0.125, dtype=torch.float32))}

    @register_transport(
        mode,
        covariance_class="test-only",
        needs_sigma=True,
        state_builder=_build_probe_state,
        serialization_keys=(serialization_key,),
    )
    def _build_probe_transport(
        phi:   torch.Tensor,
        group: GaugeGroup,

        *,
        sigma:                  torch.Tensor,
        connection_probe_sigma: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        received.append(sigma)
        if sigma is None:
            raise RuntimeError("sigma was not routed")
        built = compute_transport_operators(
            phi,
            group,
            gauge_mode=kwargs.get("gauge_mode", "learned"),
        )
        built["Omega"] = built["Omega"] * (
            1.0 + connection_probe_sigma * sigma.square().mean()
        )
        return built

    try:
        model = VFEModel(_tiny_cfg(mode))
        parameter = model.transport_state[serialization_key]
        tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        targets = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)

        _, loss, _ = model(tokens, targets)
        loss.backward()

        assert model.cfg.oracle_unroll_grad is True
        assert received
        assert all(item is not None for item in received)
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0
    finally:
        _TRANSPORTS.pop(mode, None)
        _TRANSPORT_NEEDS_MU.discard(mode)
        _TRANSPORT_NEEDS_SIGMA.discard(mode)
        _TRANSPORT_BATCH_INDEPENDENT.discard(mode)
