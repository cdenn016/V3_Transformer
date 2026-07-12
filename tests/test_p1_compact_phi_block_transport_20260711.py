"""P1 compact phi block transport: routing, equivalence, and model wiring."""

import importlib

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.geometry.groups import GaugeGroup, get_group
from vfe3.geometry.lie_ops import _COMPOSE, _equal_diag_blocks, compose_phi, register_compose
from vfe3.geometry.retraction import retract_phi
from vfe3.geometry.transport import (
    CompactFactoredTransport,
    FactoredTransport,
    transport_covariance,
    transport_mean,
)
from vfe3.inference.e_step import build_belief_transport
from vfe3.inference.e_step import phi_alignment_loss
from vfe3.model.model import VFEModel
from vfe3.model.model_frame import _MODEL_FRAMES, register_model_frame, resolve_model_frame
from vfe3.model.positional_phi import apply_positional_phi


e_step_module = importlib.import_module("vfe3.inference.e_step")
lie_ops_module = importlib.import_module("vfe3.geometry.lie_ops")
model_frame_module = importlib.import_module("vfe3.model.model_frame")
model_module = importlib.import_module("vfe3.model.model")
transport_module = importlib.import_module("vfe3.geometry.transport")
viz_extract_module = importlib.import_module("vfe3.viz.extract")


def test_compact_phi_block_transport_defaults_off_and_accepts_opt_in() -> None:
    assert VFE3Config().compact_phi_block_transport is False
    assert VFE3Config(compact_phi_block_transport=True).compact_phi_block_transport is True


def test_compact_phi_block_transport_changes_only_opted_in_block_glk_layout(
    device:      torch.device,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = get_group("block_glk")(4, 2, device=device)
    phi = 0.1 * torch.randn(2, 3, group.generators.shape[0], device=device)
    compact_calls: list[torch.Size] = []
    original = transport_module._stable_compact_glk_exp_pair

    def _spy(blocks: torch.Tensor, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        compact_calls.append(blocks.shape)
        return original(blocks, **kwargs)

    monkeypatch.setattr(transport_module, "_stable_compact_glk_exp_pair", _spy)

    legacy = build_belief_transport(phi, group, transport_mode="flat")
    explicit_false = build_belief_transport(
        phi,
        group,
        transport_mode="flat",
        compact_phi_block_transport=False,
    )
    assert compact_calls == []
    compact = build_belief_transport(
        phi,
        group,
        transport_mode="flat",
        compact_phi_block_transport=True,
    )

    assert isinstance(legacy, FactoredTransport)
    assert isinstance(explicit_false, FactoredTransport)
    assert torch.equal(legacy.exp_phi, explicit_false.exp_phi)
    assert torch.equal(legacy.exp_neg_phi, explicit_false.exp_neg_phi)
    assert isinstance(compact, CompactFactoredTransport)
    assert compact_calls == [torch.Size((2, 3, 2, 2, 2))]
    assert compact.exp_blocks.shape == (2, 3, 2, 2, 2)
    assert compact.inv_blocks.shape == (2, 3, 2, 2, 2)


@pytest.mark.parametrize(
    ("exp_fp64_mode", "threshold"),
    [("dim", 5.0), ("norm", 0.0)],
)
def test_compact_phi_block_transport_preserves_global_full_frame_clamp_and_dtype_rule(
    device:         torch.device,
    exp_fp64_mode: str,
    threshold:     float,
) -> None:
    group = get_group("block_glk")(4, 2, device=device)
    phi = torch.zeros(1, 1, group.generators.shape[0], device=device)
    phi[..., 0] = 15.0
    phi[..., 4] = 15.0

    legacy = build_belief_transport(
        phi,
        group,
        transport_mode="flat",
        exp_fp64_mode=exp_fp64_mode,
        exp_fp64_norm_threshold=threshold,
    )
    compact = build_belief_transport(
        phi,
        group,
        transport_mode="flat",
        exp_fp64_mode=exp_fp64_mode,
        exp_fp64_norm_threshold=threshold,
        compact_phi_block_transport=True,
    )

    legacy_pos = _equal_diag_blocks(legacy.exp_phi, 2, 2)
    legacy_neg = _equal_diag_blocks(legacy.exp_neg_phi, 2, 2)
    assert torch.equal(compact.exp_blocks, legacy_pos)
    assert torch.equal(compact.inv_blocks, legacy_neg)


@pytest.mark.parametrize("mean_per_head", [False, True])
def test_compact_phi_block_transport_matches_mean_covariance_and_vjp(
    device:        torch.device,
    mean_per_head: bool,
) -> None:
    torch.manual_seed(7)
    group = get_group("block_glk")(4, 2, device=device)
    phi0 = 0.05 * torch.randn(1, 3, group.generators.shape[0], device=device)
    mu0 = torch.randn(1, 3, 4, device=device)
    sigma0 = torch.rand(1, 3, 4, device=device) + 0.5
    mean_weight = torch.randn(1, 3, 3, 4, device=device)
    cov_weight = torch.randn(1, 3, 3, 4, device=device)

    outputs = []
    gradients = []
    for compact_enabled in (False, True):
        phi = phi0.clone().requires_grad_(True)
        mu = mu0.clone().requires_grad_(True)
        sigma = sigma0.clone().requires_grad_(True)
        transport = build_belief_transport(
            phi,
            group,
            transport_mode="flat",
            transport_mean_per_head=mean_per_head,
            compact_phi_block_transport=compact_enabled,
        )
        mean = transport_mean(transport, mu)
        covariance = transport_covariance(transport, sigma, diagonal_out=True)
        loss = (mean * mean_weight).sum() + 0.1 * (covariance * cov_weight).sum()
        outputs.append((mean, covariance))
        gradients.append(torch.autograd.grad(loss, (phi, mu, sigma)))

    for legacy, compact in zip(outputs[0], outputs[1]):
        assert torch.allclose(compact, legacy, atol=2e-6, rtol=1e-5)
    for legacy, compact in zip(gradients[0], gradients[1]):
        assert torch.allclose(compact, legacy, atol=5e-5, rtol=1e-5)


def test_compact_phi_block_transport_matches_full_covariance_and_vjp(
    device: torch.device,
) -> None:
    torch.manual_seed(17)
    group = get_group("block_glk")(4, 2, device=device)
    phi0 = 0.04 * torch.randn(1, 2, group.generators.shape[0], device=device)
    raw = torch.randn(1, 2, 4, 4, device=device)
    sigma0 = raw @ raw.transpose(-1, -2) + 0.5 * torch.eye(4, device=device)
    weight = torch.randn(1, 2, 2, 4, 4, device=device)

    outputs = []
    gradients = []
    for compact_enabled in (False, True):
        phi = phi0.clone().requires_grad_(True)
        sigma = sigma0.clone().requires_grad_(True)
        transport = build_belief_transport(
            phi,
            group,
            transport_mode="flat",
            compact_phi_block_transport=compact_enabled,
        )
        covariance = transport_covariance(transport, sigma, diagonal_out=False)
        outputs.append(covariance)
        gradients.append(torch.autograd.grad((covariance * weight).sum(), (phi, sigma)))

    assert torch.allclose(outputs[1], outputs[0], atol=2e-5, rtol=1e-5)
    for legacy, compact in zip(gradients[0], gradients[1]):
        assert torch.allclose(compact, legacy, atol=5e-5, rtol=1e-5)


def test_compact_phi_block_transport_matches_legacy_autocast_dtype_and_vjp() -> None:
    torch.manual_seed(23)
    group = get_group("block_glk")(4, 2)
    phi0 = 0.03 * torch.randn(1, 3, group.generators.shape[0])
    mu0 = torch.randn(1, 3, 4)
    sigma0 = torch.rand(1, 3, 4) + 0.5
    weight = torch.randn(1, 3, 3, 4)
    outputs = []
    gradients = []
    factor_dtypes = []
    factor_values = []

    for compact_enabled in (False, True):
        phi = phi0.clone().requires_grad_(True)
        with torch.amp.autocast("cpu", dtype=torch.bfloat16):
            transport = build_belief_transport(
                phi,
                group,
                transport_mode="flat",
                transport_mean_per_head=True,
                compact_phi_block_transport=compact_enabled,
            )
            mean = transport_mean(transport, mu0)
            covariance = transport_covariance(transport, sigma0, diagonal_out=True)
            loss = (mean * weight).sum() + 0.1 * (covariance * weight).sum()
        factors = transport.exp_blocks if compact_enabled else transport.exp_phi
        factor_dtypes.append(factors.dtype)
        factor_values.append(
            factors if compact_enabled else _equal_diag_blocks(factors, 2, 2))
        outputs.append((mean, covariance))
        gradients.append(torch.autograd.grad(loss, phi)[0])

    assert factor_dtypes == [torch.bfloat16, torch.bfloat16]
    assert torch.equal(factor_values[1], factor_values[0])
    for legacy, compact in zip(outputs[0], outputs[1]):
        assert torch.allclose(compact, legacy, atol=8e-3, rtol=1e-2)
    assert torch.allclose(gradients[1], gradients[0], atol=8e-3, rtol=1e-2)


def test_compact_phi_block_transport_keeps_float64_under_autocast() -> None:
    group = get_group("block_glk")(4, 2, dtype=torch.float64)
    phi = 0.03 * torch.randn(1, 3, group.generators.shape[0], dtype=torch.float64)

    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        legacy = build_belief_transport(phi, group, transport_mode="flat")
        compact = build_belief_transport(
            phi,
            group,
            transport_mode="flat",
            compact_phi_block_transport=True,
        )

    assert legacy.exp_phi.dtype == torch.float64
    assert compact.exp_blocks.dtype == torch.float64
    assert torch.equal(compact.exp_blocks, _equal_diag_blocks(legacy.exp_phi, 2, 2))


def test_compact_phi_block_transport_rejects_unmarked_same_shape_basis() -> None:
    canonical = get_group("block_glk")(4, 2)
    custom = GaugeGroup(
        name="block_glk",
        generators=canonical.generators.roll(1, dims=0),
        irrep_dims=list(canonical.irrep_dims),
        skew_symmetric=False,
    )
    phi = 0.03 * torch.randn(1, 3, custom.generators.shape[0])

    transport = build_belief_transport(
        phi,
        custom,
        transport_mode="flat",
        compact_phi_block_transport=True,
    )

    assert isinstance(transport, FactoredTransport)


def test_phi_alignment_loss_accepts_compact_transport_and_matches_legacy_vjp() -> None:
    torch.manual_seed(29)
    group = get_group("block_glk")(4, 2)
    mu = torch.randn(1, 3, 4)
    sigma = torch.rand(1, 3, 4) + 0.5
    phi0 = 0.03 * torch.randn(1, 3, group.generators.shape[0])
    values = []
    gradients = []

    for compact_enabled in (False, True):
        phi = phi0.clone().requires_grad_(True)
        value = phi_alignment_loss(
            mu,
            sigma,
            phi,
            group,
            transport_mean_per_head=True,
            compact_phi_block_transport=compact_enabled,
        )
        values.append(value)
        gradients.append(torch.autograd.grad(value, phi)[0])

    assert torch.allclose(values[1], values[0], atol=2e-5, rtol=1e-5)
    assert torch.allclose(gradients[1], gradients[0], atol=5e-5, rtol=1e-5)


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_compact_phi_block_transport_keeps_positional_bch_packed(
    monkeypatch: pytest.MonkeyPatch,
    order:       int,
) -> None:
    torch.manual_seed(31)
    group = get_group("block_glk")(4, 2)
    phi0 = 0.02 * torch.randn(1, 3, group.generators.shape[0])
    pos0 = 0.02 * torch.randn(3, group.generators.shape[0])
    weight = torch.randn_like(phi0)

    phi_legacy = phi0.clone().requires_grad_(True)
    pos_legacy = pos0.clone().requires_grad_(True)
    legacy = apply_positional_phi(
        phi_legacy,
        group,
        mode="learned",
        compose_mode="bch",
        order=order,
        pos_phi_free=pos_legacy,
    )
    legacy_grads = torch.autograd.grad((legacy * weight).sum(), (phi_legacy, pos_legacy))

    def _dense_embed_forbidden(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("compact positional BCH must not build a dense K x K embedding")

    monkeypatch.setattr(lie_ops_module, "embed_phi", _dense_embed_forbidden)
    phi_compact = phi0.clone().requires_grad_(True)
    pos_compact = pos0.clone().requires_grad_(True)
    compact = apply_positional_phi(
        phi_compact,
        group,
        mode="learned",
        compose_mode="bch",
        order=order,
        pos_phi_free=pos_compact,
        compact_blocks=True,
    )
    compact_grads = torch.autograd.grad((compact * weight).sum(), (phi_compact, pos_compact))

    assert torch.allclose(compact, legacy, atol=2e-6, rtol=1e-5)
    for compact_grad, legacy_grad in zip(compact_grads, legacy_grads):
        assert torch.allclose(compact_grad, legacy_grad, atol=5e-5, rtol=1e-5)


def test_compact_positional_bch_skips_dense_gram_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = get_group("block_glk")(4, 2)
    phi = 0.02 * torch.randn(1, 3, group.generators.shape[0])
    pos = 0.02 * torch.randn(3, group.generators.shape[0])

    def _gram_forbidden() -> torch.Tensor:
        raise AssertionError("canonical packed BCH must not form the dense generator Gram pinv")

    def _closure_scan_forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("canonical packed BCH must not hash or scan the dense generator basis")

    monkeypatch.setattr(group, "gram_pinv", _gram_forbidden)
    monkeypatch.setattr(lie_ops_module, "warn_if_basis_not_closed", _closure_scan_forbidden)
    out = apply_positional_phi(
        phi,
        group,
        mode="learned",
        compose_mode="bch",
        order=4,
        pos_phi_free=pos,
        compact_blocks=True,
    )

    assert out.shape == phi.shape


def test_compose_registry_default_call_does_not_require_compact_keyword() -> None:
    name = "_p1_legacy_compose_signature"

    @register_compose(name)
    def _legacy_compose(
        phi1:       torch.Tensor,
        phi2:       torch.Tensor,
        generators: torch.Tensor,

        *,
        order:      int,
        gram_pinv_: torch.Tensor,
        block_dims: list[int],
    ) -> torch.Tensor:
        del generators, order, gram_pinv_, block_dims
        return phi1 + phi2

    try:
        generators = get_group("block_glk")(4, 2).generators
        phi1 = torch.zeros(2, generators.shape[0])
        phi2 = torch.ones_like(phi1)
        out = compose_phi(
            phi1,
            phi2,
            generators,
            mode=name,
            gram_pinv_=torch.eye(generators.shape[0]),
            block_dims=[2, 2],
        )
        assert torch.equal(out, phi2)
    finally:
        _COMPOSE.pop(name, None)


def test_compact_phi_block_transport_keeps_bch_retraction_packed_and_matches_vjp(
    device:      torch.device,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = get_group("block_glk")(4, 2, device=device)
    phi = (0.02 * torch.randn(1, 3, group.generators.shape[0], device=device)).requires_grad_()
    delta = (0.01 * torch.randn_like(phi)).requires_grad_()
    cotangent = torch.randn_like(phi)

    legacy = retract_phi(
        phi,
        delta,
        group,
        mode="bch",
        compact_blocks=False,
    )
    legacy_grad = torch.autograd.grad(legacy, (phi, delta), cotangent)

    def _dense_embed_forbidden(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("canonical packed BCH retraction must not embed a dense KxK matrix")

    monkeypatch.setattr(lie_ops_module, "embed_phi", _dense_embed_forbidden)
    phi_compact = phi.detach().clone().requires_grad_()
    delta_compact = delta.detach().clone().requires_grad_()
    compact = retract_phi(
        phi_compact,
        delta_compact,
        group,
        mode="bch",
        compact_blocks=True,
    )
    compact_grad = torch.autograd.grad(compact, (phi_compact, delta_compact), cotangent)

    assert torch.allclose(compact, legacy, atol=2e-6, rtol=1e-5)
    assert torch.allclose(compact_grad[0], legacy_grad[0], atol=2e-6, rtol=1e-5)
    assert torch.allclose(compact_grad[1], legacy_grad[1], atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize(
    ("route", "expected"),
    [("eligible", True), ("nonflat", False), ("reflection", False)],
)
def test_model_threads_compact_toggle_into_live_phi_bch_retraction(
    monkeypatch: pytest.MonkeyPatch,
    route:       str,
    expected:    bool,
) -> None:
    values: dict[str, object] = {
        "vocab_size": 9,
        "embed_dim": 4,
        "n_heads": 2,
        "max_seq_len": 3,
        "n_layers": 1,
        "n_e_steps": 1,
        "e_phi_lr": 0.01,
        "phi_retract_mode": "bch",
        "pos_phi": "none",
        "compact_phi_block_transport": True,
    }
    if route == "nonflat":
        values["transport_mode"] = "regime_ii"
    elif route == "reflection":
        values["phi_reflection"] = "init_seed"
    cfg = VFE3Config(**values)
    model = VFEModel(cfg).eval()
    seen: list[bool] = []
    original = e_step_module.retract_phi

    def _spy(*args: object, **kwargs: object) -> torch.Tensor:
        seen.append(bool(kwargs.get("compact_blocks", False)))
        return original(*args, **kwargs)

    monkeypatch.setattr(e_step_module, "retract_phi", _spy)
    model(torch.tensor([[0, 1, 2]]))

    assert seen == [expected]


def test_model_frame_registry_default_call_does_not_require_compact_keyword() -> None:
    name = "_p1_legacy_model_frame_signature"

    @register_model_frame(name)
    def _legacy_model_frame(
        belief_phi: torch.Tensor,

        *,
        bch_order:     int,
        project_slk:   bool,
        pos_phi:       str,
        compose_mode:  str,
        pos_phi_scale: float,
        model_phi:     torch.Tensor,
        group:         GaugeGroup,
        pos_phi_free:  torch.Tensor,
    ) -> torch.Tensor:
        del bch_order, project_slk, pos_phi, compose_mode, pos_phi_scale
        del model_phi, group, pos_phi_free
        return belief_phi

    try:
        group = get_group("block_glk")(4, 2)
        belief_phi = torch.zeros(1, 3, group.generators.shape[0])
        out = resolve_model_frame(
            belief_phi,
            mode=name,
            model_phi=torch.zeros_like(belief_phi),
            group=group,
            pos_phi_free=torch.zeros_like(belief_phi[0]),
        )
        assert out is belief_phi
    finally:
        _MODEL_FRAMES.pop(name, None)


@pytest.mark.parametrize(
    ("route", "expected"),
    [
        ("eligible", True),
        ("omega_direct", False),
        ("nonflat", False),
        ("reflection", False),
        ("single_head", False),
    ],
)
def test_model_gates_packed_bch_to_eligible_transport_route(
    monkeypatch: pytest.MonkeyPatch,
    route:       str,
    expected:    bool,
) -> None:
    values: dict[str, object] = {
        "vocab_size": 9,
        "embed_dim": 4,
        "n_heads": 2,
        "max_seq_len": 3,
        "n_layers": 1,
        "n_e_steps": 1,
        "e_phi_lr": 0.0,
        "compact_phi_block_transport": True,
    }
    if route == "omega_direct":
        values["gauge_parameterization"] = "omega_direct"
        values["pos_phi"] = "none"
    elif route == "nonflat":
        values["transport_mode"] = "regime_ii"
    elif route == "reflection":
        values["phi_reflection"] = "init_seed"
    elif route == "single_head":
        values["n_heads"] = 1
    model = VFEModel(VFE3Config(**values)).eval()
    seen: list[bool] = []
    original = model_module.apply_positional_phi

    def _spy(phi: torch.Tensor, group: GaugeGroup, **kwargs: object) -> torch.Tensor:
        seen.append(bool(kwargs.get("compact_blocks", False)))
        return original(phi, group, **kwargs)

    monkeypatch.setattr(model_module, "apply_positional_phi", _spy)
    encoded = model.prior_bank.encode(torch.tensor([[0, 1, 2]])).phi
    model._apply_pos_phi(encoded)

    assert seen == ([] if route == "omega_direct" else [expected])
    iter_kwargs = viz_extract_module._iter_kwargs(model, torch.zeros(3, 3), None)
    assert iter_kwargs["compact_phi_block_transport"] is expected


def test_phi_tilde_model_frame_receives_packed_bch_toggle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = get_group("block_glk")(4, 2)
    belief_phi = torch.zeros(1, 3, group.generators.shape[0])
    model_phi = 0.02 * torch.randn_like(belief_phi)
    pos_phi = 0.02 * torch.randn(3, group.generators.shape[0])
    seen: list[bool] = []
    original = model_frame_module.apply_positional_phi

    def _spy(phi: torch.Tensor, group: GaugeGroup, **kwargs: object) -> torch.Tensor:
        seen.append(bool(kwargs.get("compact_blocks", False)))
        return original(phi, group, **kwargs)

    monkeypatch.setattr(model_frame_module, "apply_positional_phi", _spy)
    out = resolve_model_frame(
        belief_phi,
        mode="phi_tilde",
        model_phi=model_phi,
        group=group,
        pos_phi_free=pos_phi,
        pos_phi="learned",
        compose_mode="bch",
        compact_blocks=True,
    )

    assert seen == [True]
    assert out.shape == belief_phi.shape


@pytest.mark.parametrize("route", ["glk", "tied", "cross", "reflection", "single_head"])
def test_compact_phi_block_transport_leaves_ineligible_routes_legacy(route: str) -> None:
    if route == "glk":
        group = get_group("glk")(4)
    elif route == "tied":
        group = get_group("tied_block_glk")(4, 2)
    elif route == "cross":
        group = get_group("block_glk")(4, 2, cross_couplings=[(0, 1)])
    elif route == "single_head":
        group = get_group("block_glk")(4, 1)
    else:
        group = get_group("block_glk")(4, 2)
    phi = 0.05 * torch.randn(1, 3, group.generators.shape[0])
    reflection = torch.tensor([[1.0, -1.0, 1.0]]) if route == "reflection" else None

    transport = build_belief_transport(
        phi,
        group,
        transport_mode="flat",
        reflection=reflection,
        compact_phi_block_transport=True,
    )

    assert not isinstance(transport, CompactFactoredTransport)


def _tiny_two_channel_config(compact: bool) -> VFE3Config:
    return VFE3Config(
        vocab_size=9,
        embed_dim=4,
        n_heads=2,
        max_seq_len=5,
        n_layers=1,
        n_e_steps=1,
        e_phi_lr=0.0,
        use_prior_bank=True,
        prior_source="model_channel",
        s_e_step=True,
        lambda_h=1.0,
        lambda_gamma=0.75,
        share_refine_s_transport=True,
        transport_mean_per_head=True,
        compact_phi_block_transport=compact,
    )


@pytest.mark.parametrize("compact", [False, True])
def test_gamma_energy_forwards_transport_mean_per_head_for_both_factor_layouts(
    monkeypatch: pytest.MonkeyPatch,
    compact:     bool,
) -> None:
    model = VFEModel(_tiny_two_channel_config(compact)).eval()
    token_ids = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long)
    phi = model.prior_bank.encode(token_ids).phi
    seen: list[object] = []
    original = e_step_module.build_belief_transport

    def _spy(*args: object, **kwargs: object) -> object:
        result = original(*args, **kwargs)
        seen.append(result)
        return result

    monkeypatch.setattr(e_step_module, "build_belief_transport", _spy)
    model._gamma_energy(token_ids, phi)

    assert len(seen) == 1
    assert isinstance(seen[0], (CompactFactoredTransport, FactoredTransport))
    assert seen[0].mean_per_head is True


def test_compact_phi_block_transport_reaches_shared_and_gamma_model_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(13)
    legacy_model = VFEModel(_tiny_two_channel_config(False)).eval()
    compact_model = VFEModel(_tiny_two_channel_config(True)).eval()
    compact_model.load_state_dict(legacy_model.state_dict())
    token_ids = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long)
    targets = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)

    legacy_logits, legacy_loss, _ = legacy_model(token_ids, targets)
    legacy_grad, = torch.autograd.grad(legacy_loss, legacy_model.prior_bank.phi_embed)

    seen: list[object] = []
    original = e_step_module.build_belief_transport

    def _spy(*args: object, **kwargs: object) -> object:
        result = original(*args, **kwargs)
        seen.append(result)
        return result

    monkeypatch.setattr(e_step_module, "build_belief_transport", _spy)
    compact_logits, compact_loss, _ = compact_model(token_ids, targets)
    compact_grad, = torch.autograd.grad(compact_loss, compact_model.prior_bank.phi_embed)
    encoded_phi = compact_model.prior_bank.encode(token_ids).phi
    compact_model._gamma_energy(token_ids, encoded_phi)

    assert sum(isinstance(item, CompactFactoredTransport) for item in seen) >= 2
    assert torch.allclose(compact_logits, legacy_logits, atol=2e-5, rtol=1e-5)
    assert torch.allclose(compact_loss, legacy_loss, atol=2e-5, rtol=1e-5)
    assert torch.allclose(compact_grad, legacy_grad, atol=5e-5, rtol=1e-5)
