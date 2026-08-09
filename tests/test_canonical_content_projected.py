"""Projected canonical-content materialization and realized-frame contracts."""

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.geometry.lie_ops import CompactBlockElement
from vfe3.geometry.transport import CompactFactoredTransport, FactoredTransport
from vfe3.model.canonical_content import (
    CanonicalFrameContext,
    project_canonical_diagonal,
    pullback_diagonal_query,
)
from vfe3.model.model import VFEModel
from vfe3.model.prior_bank import PriorBank, get_encode


def _projected_cfg(**overrides: object) -> VFE3Config:
    values = dict(
        vocab_size=11,
        embed_dim=4,
        n_heads=2,
        max_seq_len=5,
        batch_size=2,
        n_layers=1,
        n_e_steps=1,
        e_step_update="gradient",
        oracle_unroll_grad=True,
        family="gaussian_diagonal",
        transport_mode="flat",
        gauge_parameterization="phi",
        prior_source="token",
        s_e_step=False,
        e_phi_lr=0.0,
        use_prior_bank=True,
        decode_mode="full",
        encode_mode="canonical_content_projected",
        omega_reflection="off",
        phi_reflection="off",
        lambda_alpha_mode="constant",
        max_steps=1,
    )
    values.update(overrides)
    return VFE3Config(**values)


def _projected_bank(*, untie_decode_bank: bool = False) -> PriorBank:
    torch.manual_seed(71)
    return PriorBank(
        vocab_size=9,
        K=4,
        n_gen=8,
        family="gaussian_diagonal",
        encode_mode="canonical_content_projected",
        decode_mode="full",
        use_prior_bank=True,
        prior_source="token",
        s_e_step=False,
        gauge_parameterization="phi",
        omega_reflection="off",
        phi_reflection="off",
        untie_decode_bank=untie_decode_bank,
    )


def _noncommuting_frames() -> tuple[torch.Tensor, torch.Tensor]:
    token_frame = torch.tensor(
        [[1.5, 0.4], [-0.2, 0.9]], dtype=torch.float64)
    positional_frame = torch.tensor(
        [[1.0, -0.3], [0.25, 1.2]], dtype=torch.float64)
    return token_frame, positional_frame


def test_pushforward_matches_hand_reference_for_batched_nonorthogonal_frames() -> None:
    """Changing the covariance pushforward from squared frame entries breaks this reference."""
    token_frame, positional_frame = _noncommuting_frames()
    frame = (token_frame @ positional_frame).expand(2, 3, 2, 2).clone()
    mu_c = torch.tensor(
        [
            [[0.2, -0.7], [1.1, 0.3], [-0.4, 0.8]],
            [[-0.6, 0.5], [0.9, -1.2], [0.1, 0.4]],
        ],
        dtype=torch.float64,
    )
    var_c = torch.tensor(
        [
            [[0.5, 1.3], [0.7, 0.4], [1.5, 0.2]],
            [[0.8, 0.6], [0.3, 1.7], [1.1, 0.9]],
        ],
        dtype=torch.float64,
    )

    got_mu, got_var = project_canonical_diagonal(mu_c, var_c, frame)

    expected_mu = torch.einsum("...ij,...j->...i", frame, mu_c)
    expected_var = torch.einsum("...ij,...j->...i", frame.square(), var_c)
    assert torch.equal(got_mu, expected_mu)
    assert torch.equal(got_var, expected_var)
    assert got_mu.dtype == mu_c.dtype and got_mu.device == mu_c.device
    assert got_var.dtype == var_c.dtype and got_var.device == var_c.device


def test_positional_order_is_token_frame_times_right_positional_frame() -> None:
    """Reversing the configured right-factor multiplication changes materialized content."""
    token_frame, positional_frame = _noncommuting_frames()
    mu_c = torch.tensor([0.6, -0.35], dtype=torch.float64)
    var_c = torch.tensor([0.45, 1.25], dtype=torch.float64)
    right_order = token_frame @ positional_frame
    reversed_order = positional_frame @ token_frame

    got_mu, got_var = project_canonical_diagonal(mu_c, var_c, right_order)
    wrong_mu = reversed_order @ mu_c
    wrong_var = reversed_order.square() @ var_c

    assert torch.equal(got_mu, right_order @ mu_c)
    assert torch.equal(got_var, right_order.square() @ var_c)
    assert not torch.allclose(got_mu, wrong_mu)
    assert not torch.allclose(got_var, wrong_var)


def test_pullback_matches_hand_reference_and_mean_inverse_consistency() -> None:
    """Using a transpose or forward frame in pullback breaks canonical query coordinates."""
    token_frame, positional_frame = _noncommuting_frames()
    frame = (token_frame @ positional_frame).expand(2, 3, 2, 2).clone()
    frame_inv = torch.linalg.inv(frame)
    mu_c = torch.tensor(
        [
            [[0.4, -0.1], [0.8, 0.7], [-0.3, 1.0]],
            [[-0.5, 0.6], [0.2, -0.9], [1.3, 0.1]],
        ],
        dtype=torch.float64,
    )
    mu_q = torch.einsum("...ij,...j->...i", frame, mu_c)
    var_q = torch.tensor(
        [
            [[0.9, 0.4], [1.2, 0.8], [0.5, 1.1]],
            [[0.6, 1.4], [0.7, 0.3], [1.5, 0.2]],
        ],
        dtype=torch.float64,
    )

    got_mu, got_cov = pullback_diagonal_query(mu_q, var_q, frame_inv)

    expected_mu = torch.einsum("...ij,...j->...i", frame_inv, mu_q)
    expected_cov = frame_inv @ torch.diag_embed(var_q) @ frame_inv.transpose(-1, -2)
    assert torch.allclose(got_mu, mu_c, atol=1e-14, rtol=1e-14)
    assert torch.equal(got_mu, expected_mu)
    assert torch.equal(got_cov, expected_cov)
    assert torch.allclose(
        frame @ frame_inv,
        torch.eye(2, dtype=torch.float64).expand_as(frame),
        atol=1e-14,
        rtol=1e-14,
    )


def test_pushforward_and_pullback_pass_float64_gradcheck() -> None:
    """Detaching or casting any projected moment or frame factor breaks differentiability."""
    generator = torch.Generator().manual_seed(51)
    mu_c = torch.randn(1, 2, 2, dtype=torch.float64, generator=generator, requires_grad=True)
    var_c = (torch.rand(1, 2, 2, dtype=torch.float64, generator=generator) + 0.7).requires_grad_()
    frame = (
        torch.eye(2, dtype=torch.float64).expand(1, 2, 2, 2).clone()
        + 0.12 * torch.randn(1, 2, 2, 2, dtype=torch.float64, generator=generator)
    ).requires_grad_()
    assert torch.autograd.gradcheck(
        project_canonical_diagonal,
        (mu_c, var_c, frame),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-4,
    )

    mu_q = torch.randn(1, 2, 2, dtype=torch.float64, generator=generator, requires_grad=True)
    var_q = (torch.rand(1, 2, 2, dtype=torch.float64, generator=generator) + 0.8).requires_grad_()
    frame_inv = torch.linalg.inv(frame.detach()).requires_grad_()
    assert torch.autograd.gradcheck(
        pullback_diagonal_query,
        (mu_q, var_q, frame_inv),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-4,
    )


def test_coordinate_transforms_preserve_float32_under_bf16_autocast() -> None:
    """Autocast may not change the public moment/frame coordinate dtype."""
    generator = torch.Generator().manual_seed(83)
    mu = torch.randn(2, 3, 4, generator=generator)
    var = torch.rand(2, 3, 4, generator=generator) + 0.4
    frame = (
        torch.eye(4).expand(2, 3, 4, 4).clone()
        + 0.08 * torch.randn(2, 3, 4, 4, generator=generator)
    )
    frame_inv = torch.linalg.inv(frame)
    expected_push = project_canonical_diagonal(mu, var, frame)
    expected_pull = pullback_diagonal_query(mu, var, frame_inv)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        got_push = project_canonical_diagonal(mu, var, frame)
        got_pull = pullback_diagonal_query(mu, var, frame_inv)

    for got, expected in zip((*got_push, *got_pull), (*expected_push, *expected_pull)):
        assert got.dtype == torch.float32
        assert torch.equal(got, expected)


@pytest.mark.parametrize(
    ("which", "match"),
    [
        ("moment", "identical shapes"),
        ("frame", "square"),
        ("leading", "leading shape"),
        ("dtype", "same dtype"),
    ],
)
def test_pushforward_rejects_incompatible_shapes_and_dtypes(which: str, match: str) -> None:
    """Silent broadcasting or promotion would hide a mismatched realized frame."""
    mu = torch.zeros(2, 3, dtype=torch.float64)
    var = torch.ones_like(mu)
    frame = torch.eye(3, dtype=torch.float64).expand(2, 3, 3).clone()
    if which == "moment":
        var = torch.ones(2, 4, dtype=torch.float64)
    elif which == "frame":
        frame = torch.zeros(2, 3, 4, dtype=torch.float64)
    elif which == "leading":
        frame = torch.eye(3, dtype=torch.float64).expand(1, 3, 3).clone()
    else:
        frame = frame.float()

    with pytest.raises(ValueError, match=match):
        project_canonical_diagonal(mu, var, frame)


def test_frame_context_rejects_noninverse_layout_mismatch() -> None:
    """A context with differently shaped factors cannot identify one realized vertex frame."""
    with pytest.raises(ValueError, match="identical shapes"):
        CanonicalFrameContext(
            forward=torch.eye(2, dtype=torch.float64).expand(2, 2, 2).clone(),
            inverse=torch.eye(2, dtype=torch.float64).expand(1, 2, 2).clone(),
        )


def test_projected_registry_returns_existing_canonical_table_coordinates() -> None:
    """A separate projected embedding table would duplicate parameters and alter initialization."""
    get_encode("canonical_content_projected")
    projected = _projected_bank()
    token_ids = torch.tensor([[0, 4, 2], [7, 1, 8]])

    got = projected.encode(token_ids)

    assert torch.equal(got.mu, projected.mu_embed[token_ids])
    assert torch.equal(got.sigma, torch.exp(projected.sigma_log_embed[token_ids]))
    assert torch.equal(got.phi, projected.phi_embed[token_ids])
    assert not any("canonical" in name for name in projected._parameters)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("family", "gaussian_full"),
        ("transport_mode", "regime_ii"),
        ("gauge_parameterization", "omega_direct"),
        ("prior_source", "model_channel"),
        ("s_e_step", True),
        ("e_phi_lr", 0.01),
        ("use_prior_bank", False),
        ("decode_mode", "diagonal"),
        ("omega_reflection", "init_seed"),
        ("phi_reflection", "init_seed"),
    ],
)
def test_projected_config_rejects_each_incompatible_cross_field_pair(
    field: str,
    value: object,
) -> None:
    """Relaxing any projected-mode gate would invalidate its shared-frame decode contract."""
    with pytest.raises(ValueError, match="canonical_content_projected"):
        _projected_cfg(**{field: value})


@pytest.mark.parametrize("decode_mode", ["full", "full_chunked"])
@pytest.mark.parametrize("pos_phi", ["learned", "frozen"])
@pytest.mark.parametrize("untie_decode_bank", [False, True])
def test_projected_config_accepts_supported_decode_position_and_tie_variants(
    decode_mode: str,
    pos_phi: str,
    untie_decode_bank: bool,
) -> None:
    """Narrowing a legal projected variant would block current-frame rematerialization."""
    cfg = _projected_cfg(
        decode_mode=decode_mode,
        pos_phi=pos_phi,
        pos_phi_compose="group_product",
        untie_decode_bank=untie_decode_bank,
        m_phi_lr=0.02,
    )
    assert cfg.decode_mode == decode_mode
    assert cfg.pos_phi == pos_phi
    assert cfg.untie_decode_bank is untie_decode_bank


def test_direct_projected_prior_bank_rejects_locally_incompatible_family() -> None:
    """Direct bank construction must not bypass the projected encoder's family boundary."""
    with pytest.raises(ValueError, match="canonical_content_projected"):
        PriorBank(
            vocab_size=9,
            K=4,
            n_gen=8,
            family="gaussian_full",
            encode_mode="canonical_content_projected",
            decode_mode="full",
        )


def test_projected_tied_and_untied_banks_share_only_the_canonical_initial_values() -> None:
    """Untying decode must clone the canonical tables while encode retains its original leaves."""
    tied = _projected_bank()
    untied = _projected_bank(untie_decode_bank=True)

    assert torch.equal(tied.mu_embed, untied.mu_embed)
    assert torch.equal(tied.sigma_log_embed, untied.sigma_log_embed)
    assert torch.equal(untied.decode_mu_embed, untied.mu_embed)
    assert torch.equal(untied.decode_sigma_log_embed, untied.sigma_log_embed)
    assert untied.decode_mu_embed.data_ptr() != untied.mu_embed.data_ptr()
    assert untied.decode_sigma_log_embed.data_ptr() != untied.sigma_log_embed.data_ptr()


def _set_noncommuting_model_frames(model: VFEModel, token_ids: torch.Tensor) -> None:
    token_blocks = torch.tensor(
        [
            [[[0.10, 0.28], [-0.16, 0.05]], [[-0.08, 0.19], [0.07, 0.12]]],
            [[[0.04, -0.21], [0.13, -0.09]], [[0.11, 0.06], [-0.18, 0.03]]],
        ],
        dtype=model.prior_bank.phi_embed.dtype,
    )
    position_blocks = torch.tensor(
        [
            [[[0.03, -0.17], [0.09, 0.06]], [[-0.12, 0.04], [0.15, 0.02]]],
            [[[-0.07, 0.14], [0.05, 0.08]], [[0.09, -0.13], [0.04, -0.05]]],
        ],
        dtype=model.prior_bank.phi_embed.dtype,
    )
    with torch.no_grad():
        model.prior_bank.phi_embed[token_ids[0]] = token_blocks.flatten(start_dim=1)
        model.pos_phi_free[: token_ids.shape[1]] = position_blocks.flatten(start_dim=1)
        model.prior_bank.mu_embed[token_ids[0]] = torch.tensor(
            [[0.35, -0.25, 0.55, 0.10], [-0.40, 0.65, 0.15, -0.30]])
        model.prior_bank.sigma_log_embed[token_ids[0]] = torch.log(torch.tensor(
            [[0.7, 1.3, 0.5, 1.1], [1.4, 0.6, 0.9, 1.2]]))


def test_projected_materialize_uses_one_shared_factored_frame_and_preserves_q0_equals_p(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rebuilding transport or materializing q0 and p separately breaks their shared frame/value."""
    import vfe3.inference.e_step as e_step_module
    import vfe3.model.model as model_module

    model = VFEModel(_projected_cfg(
        pos_phi="learned",
        pos_phi_compose="group_product",
        e_step_update="mm_exact",
    ))
    monkeypatch.setattr(model, "_compact_phi_blocks_enabled", lambda: False)
    token_ids = torch.tensor([[2, 5]])
    _set_noncommuting_model_frames(model, token_ids)
    canonical = model.prior_bank.encode(token_ids)

    original_builder = e_step_module.build_belief_transport
    built: list[object] = []

    def tracked_builder(*args: object, **kwargs: object) -> object:
        transport = original_builder(*args, **kwargs)
        built.append(transport)
        return transport

    original_stack = model_module.vfe_stack
    entry: dict[str, object] = {}

    def tracked_stack(
        belief: object,
        mu_p: torch.Tensor,
        sigma_p: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> object:
        entry.update(
            belief=belief,
            mu_p=mu_p,
            sigma_p=sigma_p,
            prebuilt=kwargs.get("prebuilt_transport"),
        )
        return original_stack(belief, mu_p, sigma_p, *args, **kwargs)

    monkeypatch.setattr(e_step_module, "build_belief_transport", tracked_builder)
    monkeypatch.setattr(model_module, "vfe_stack", tracked_stack)
    capture: dict = {}

    model.forward_beliefs(token_ids, capture=capture)

    assert len(built) == 1
    transport = built[0]
    assert isinstance(transport, FactoredTransport)
    assert entry["prebuilt"] is transport
    context = capture["canonical_frame"]
    assert context.forward is transport.exp_phi
    assert context.inverse is transport.exp_neg_phi

    token_matrix = torch.einsum(
        "...a,aij->...ij", canonical.phi, model.group.generators)
    position_matrix = torch.einsum(
        "...a,aij->...ij", model.pos_phi_free[: token_ids.shape[1]], model.group.generators)
    expected_frame = torch.matrix_exp(token_matrix) @ torch.matrix_exp(position_matrix)
    reversed_frame = torch.matrix_exp(position_matrix) @ torch.matrix_exp(token_matrix)
    assert torch.allclose(context.forward, expected_frame, atol=2e-6, rtol=2e-6)
    assert not torch.allclose(context.forward, reversed_frame, atol=1e-5, rtol=1e-5)

    expected_mu, expected_var = project_canonical_diagonal(
        canonical.mu, canonical.sigma, context.forward)
    entry_belief = entry["belief"]
    assert entry_belief.mu is entry["mu_p"]
    assert entry_belief.sigma is entry["sigma_p"]
    assert torch.equal(entry_belief.mu, expected_mu)
    assert torch.equal(entry_belief.sigma, expected_var)
    assert capture["prior"].mu is entry_belief.mu
    assert capture["prior"].sigma is entry_belief.sigma

    pulled_mu, pulled_cov = pullback_diagonal_query(
        entry_belief.mu, entry_belief.sigma, context.inverse)
    assert torch.equal(
        pulled_mu,
        torch.einsum("...ij,...j->...i", context.inverse, entry_belief.mu),
    )
    assert torch.equal(
        pulled_cov,
        context.inverse
        @ torch.diag_embed(entry_belief.sigma)
        @ context.inverse.transpose(-1, -2),
    )


def test_projected_materialize_expands_only_compact_vertex_factors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compact transport must supply dense vertices without materializing pairwise Omega."""
    import vfe3.inference.e_step as e_step_module

    model = VFEModel(_projected_cfg(
        pos_phi="learned",
        pos_phi_compose="group_product",
        e_step_update="mm_exact",
    ))
    token_ids = torch.tensor([[2, 5]])
    _set_noncommuting_model_frames(model, token_ids)
    original_builder = e_step_module.build_belief_transport
    built: list[object] = []

    def tracked_builder(*args: object, **kwargs: object) -> object:
        transport = original_builder(*args, **kwargs)
        built.append(transport)
        return transport

    def pairwise_forbidden(_self: object) -> torch.Tensor:
        raise AssertionError("projected materialization built pairwise Omega")

    monkeypatch.setattr(e_step_module, "build_belief_transport", tracked_builder)
    monkeypatch.setattr(CompactFactoredTransport, "to_dense_omega", pairwise_forbidden)
    capture: dict = {}

    model.forward_beliefs(token_ids, capture=capture)

    assert len(built) == 1
    transport = built[0]
    assert isinstance(transport, CompactFactoredTransport)
    context = capture["canonical_frame"]
    assert torch.equal(
        context.forward,
        CompactBlockElement(transport.exp_blocks, transport.K).to_dense(),
    )
    assert torch.equal(
        context.inverse,
        CompactBlockElement(transport.inv_blocks, transport.K).to_dense(),
    )


def test_projected_truncated_backprop_reuses_one_transport_across_all_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The context frame must remain the E-step frame across truncation boundaries."""
    import vfe3.inference.e_step as e_step_module

    model = VFEModel(_projected_cfg(
        n_layers=2,
        n_e_steps=2,
        e_steps_backprop_last=1,
        e_step_update="mm_exact",
    ))
    token_ids = torch.tensor([[2, 5]])
    original_builder = e_step_module.build_belief_transport
    original_iteration = e_step_module.e_step_iteration
    built: list[object] = []
    consumed: list[object] = []

    def tracked_builder(*args: object, **kwargs: object) -> object:
        transport = original_builder(*args, **kwargs)
        built.append(transport)
        return transport

    def tracked_iteration(*args: object, **kwargs: object) -> object:
        consumed.append(kwargs.get("_prebuilt_omega"))
        return original_iteration(*args, **kwargs)

    monkeypatch.setattr(e_step_module, "build_belief_transport", tracked_builder)
    monkeypatch.setattr(e_step_module, "e_step_iteration", tracked_iteration)
    capture: dict = {}

    model.forward_beliefs(token_ids, training=True, capture=capture)

    assert len(built) == 1
    assert len(consumed) == 4
    assert all(transport is built[0] for transport in consumed)
    context = capture["canonical_frame"]
    if isinstance(built[0], CompactFactoredTransport):
        assert torch.equal(
            context.forward,
            CompactBlockElement(built[0].exp_blocks, built[0].K).to_dense(),
        )
    else:
        assert isinstance(built[0], FactoredTransport)
        assert context.forward is built[0].exp_phi


@pytest.mark.parametrize("gauge_group", ["block_glk", "glk"])
def test_projected_single_head_groups_request_factored_vertex_transport(
    gauge_group: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single-block projected groups still require exact vertex factors, never pairwise Omega."""
    import vfe3.inference.e_step as e_step_module

    model = VFEModel(_projected_cfg(
        gauge_group=gauge_group,
        n_heads=1,
        pos_phi="none",
        e_step_update="mm_exact",
    ))
    original_builder = e_step_module.build_belief_transport
    built: list[object] = []

    def tracked_builder(*args: object, **kwargs: object) -> object:
        transport = original_builder(*args, **kwargs)
        built.append(transport)
        return transport

    def pairwise_forbidden(_self: object) -> torch.Tensor:
        raise AssertionError("projected single-head path materialized pairwise Omega")

    monkeypatch.setattr(e_step_module, "build_belief_transport", tracked_builder)
    monkeypatch.setattr(FactoredTransport, "to_dense_omega", pairwise_forbidden)
    capture: dict = {}

    model.forward_beliefs(torch.tensor([[2, 5]]), capture=capture)

    assert len(built) == 1
    assert isinstance(built[0], FactoredTransport)
    assert capture["canonical_frame"].forward is built[0].exp_phi


def test_projected_model_materialization_preserves_float32_under_bf16_autocast() -> None:
    """The model's public projected prior remains float32 inside its AMP region."""
    model = VFEModel(_projected_cfg(
        amp_dtype="bf16",
        pos_phi="none",
        e_step_update="mm_exact",
    ))
    capture: dict = {}

    model.forward_beliefs(torch.tensor([[2, 5]]), capture=capture)

    canonical = model.prior_bank.encode(torch.tensor([[2, 5]]))
    context = capture["canonical_frame"]
    prior = capture["prior"]
    assert canonical.mu.dtype == torch.float32
    assert context.forward.dtype == torch.float32
    assert context.inverse.dtype == torch.float32
    assert prior.mu.dtype == torch.float32
    assert prior.sigma.dtype == torch.float32


def test_projected_phi_gradient_reaches_canonical_tables_and_realized_frames() -> None:
    """Detaching materialization severs at least one supervised mean/variance/frame gradient."""
    model = VFEModel(_projected_cfg(
        pos_phi="learned",
        pos_phi_compose="group_product",
        e_step_update="gradient",
        e_q_mu_lr=0.03,
        e_q_sigma_lr=0.02,
    ))
    token_ids = torch.tensor([[2, 5]])
    _set_noncommuting_model_frames(model, token_ids)
    capture: dict = {}
    model.forward_beliefs(token_ids, capture=capture)
    materialized_prior = capture["prior"]
    target_mu = torch.tensor(
        [[[0.8, -0.1, 0.2, -0.5], [-0.2, 0.3, 0.9, 0.4]]],
        dtype=materialized_prior.mu.dtype,
    )
    target_var = torch.tensor(
        [[[0.6, 1.0, 1.4, 0.8], [1.1, 0.7, 0.5, 1.3]]],
        dtype=materialized_prior.sigma.dtype,
    )
    supervised_scalar = (
        (materialized_prior.mu - target_mu).square().sum()
        + 0.37 * (materialized_prior.sigma - target_var).square().sum()
    )

    supervised_scalar.backward()

    for parameter in (
        model.prior_bank.mu_embed,
        model.prior_bank.sigma_log_embed,
        model.prior_bank.phi_embed,
        model.pos_phi_free,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad).item() > 0
