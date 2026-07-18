import hashlib

import pytest
import torch
from torch import nn

from vfe3.geometry.groups import get_group
from vfe3.geometry.phi_preconditioner import PullbackGroupDirectionResult
from vfe3.gauge_optim import (
    GaugeManifoldAdamW,
    PhiUpdatePolicy,
    _PHI_UPDATE_POLICIES,
    get_phi_update_policy,
    stage_pullback_group_candidate,
)


def _make_optimizer(
    param_groups,
    group,
    *,
    trust_radius=0.1,
    chart_max_norm=5.0,
    bch_residual_max=1e-6,
    phi_precond_mode="pullback",
    **kwargs,
):
    return GaugeManifoldAdamW(
        param_groups,
        group,
        phi_group_trust_radius=trust_radius,
        phi_chart_max_norm=chart_max_norm,
        phi_bch_residual_max=bch_residual_max,
        phi_precond_mode=phi_precond_mode,
        weight_decay=0.0,
        **kwargs,
    )


def _fixed_direction(xi: torch.Tensor) -> PullbackGroupDirectionResult:
    rows = xi.shape[:-1]
    ones = torch.ones(rows, dtype=torch.float64, device=xi.device)
    zeros = torch.zeros(rows, dtype=torch.float64, device=xi.device)
    return PullbackGroupDirectionResult(
        v_phi=xi.double(),
        xi=xi.double(),
        min_undamped_generalized_eigenvalue=ones,
        undamped_generalized_condition=ones,
        damped_generalized_condition=ones,
        scaled_solve_residual=zeros,
        series_order=40,
    )


def test_phi_update_policy_registry_contract_is_exact_and_immutable():
    assert set(_PHI_UPDATE_POLICIES) == {"adamw", "pullback_group"}

    adamw = get_phi_update_policy("adamw")
    pullback = get_phi_update_policy("pullback_group")
    assert isinstance(adamw, PhiUpdatePolicy)
    assert dict(adamw.optimizer_group_metadata) == {}
    assert adamw.requires_manifold_optimizer is False
    assert adamw.requires_pullback_geometry is False
    assert dict(pullback.optimizer_group_metadata) == {
        "pullback_group": True,
        "weight_decay": 0.0,
    }
    assert pullback.requires_manifold_optimizer is True
    assert pullback.requires_pullback_geometry is True
    with pytest.raises(TypeError):
        pullback.optimizer_group_metadata["weight_decay"] = 1.0
    assert dict(get_phi_update_policy("pullback_group").optimizer_group_metadata) == {
        "pullback_group": True,
        "weight_decay": 0.0,
    }


def test_default_adamw_one_step_is_byte_identical_to_golden():
    from vfe3.config import VFE3Config
    from vfe3.model.model import VFEModel
    from vfe3.train import build_optimizer, train_step

    torch.manual_seed(0)
    cfg = VFE3Config(
        vocab_size=8,
        embed_dim=4,
        n_heads=2,
        max_seq_len=4,
        n_layers=1,
        n_e_steps=1,
        pos_phi="none",
    )
    model = VFEModel(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    tokens = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=torch.long)
    targets = torch.tensor([[1, 2, 3, 4], [2, 1, 0, 7]], dtype=torch.long)

    loss = train_step(model, optimizer, scheduler, tokens, targets, grad_clip=1.0)

    assert type(optimizer) is torch.optim.AdamW
    assert [group["role"] for group in optimizer.param_groups] == ["mu", "sigma", "phi", "mu"]
    state = optimizer.state_dict()["state"]
    assert sorted(state) == [0, 1, 3, 4]
    assert all(set(slot) == {"step", "exp_avg", "exp_avg_sq"} for slot in state.values())
    assert 2 not in state
    assert loss == 2.0770487785339355

    def digest(tensor: torch.Tensor) -> str:
        data = tensor.detach().cpu().contiguous().numpy().tobytes()
        return hashlib.sha256(data).hexdigest()

    assert digest(model.prior_bank.mu_embed) == "ffc101ec6c9b0fc34e1089dda4b5b28cafa6e2d53678c257ced04723e6e2a66a"
    assert digest(model.prior_bank.sigma_log_embed) == "b632d4a844923ebb4e8af9e1158ae9eb20ad37a64677bc53186235915d2790fd"
    assert digest(model.prior_bank.phi_embed) == "e31bfece5ed86861d7d32f3e16214c17ddcb780dfda1116c3cffbf0e27a674b9"
    assert digest(model.prior_bank.decode_log_scale) == "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119"
    assert digest(model.prior_bank.output_proj_weight) == "1bc018feb7e7c10fcd644d14bbf24a3ac2fe50073b9a355e2bcca69b907618de"


@pytest.mark.parametrize("value", [-1, 1.5, True, False, "2", None])
def test_direct_optimizer_rejects_invalid_omega_reorth_every(value):
    group = get_group("glk")(K=4)
    omega = nn.Parameter(torch.eye(4).unsqueeze(0))

    with pytest.raises(ValueError, match="omega_reorth_every"):
        _make_optimizer(
            [{"params": [omega], "lr": 0.1, "omega": True, "weight_decay": 0.0}],
            group,
            omega_reorth_every=value,
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("trust_radius", 0.0),
        ("trust_radius", float("nan")),
        ("chart_max_norm", -1.0),
        ("chart_max_norm", float("inf")),
        ("bch_residual_max", 0.0),
        ("bch_residual_max", float("nan")),
    ],
)
def test_direct_optimizer_rejects_invalid_phi_bounds(name, value):
    group = get_group("glk")(K=2)
    phi = nn.Parameter(torch.zeros(1, 4))
    kwargs = {name: value}
    with pytest.raises(ValueError, match="finite and positive"):
        _make_optimizer(
            [{"params": [phi], "lr": 0.1, "pullback_group": True, "weight_decay": 0.0}],
            group,
            **kwargs,
        )


def test_pullback_group_step_moves_only_active_rows_and_creates_no_phi_state():
    group = get_group("glk")(K=2)
    phi = nn.Parameter(torch.zeros(5, 4))
    optimizer = _make_optimizer(
        [{"params": [phi], "lr": 0.05, "pullback_group": True, "weight_decay": 0.0}],
        group,
    )
    before = phi.detach().clone()
    phi.grad = torch.zeros_like(phi)
    phi.grad[1, 0] = 1.0
    phi.grad[3, 3] = -2.0

    optimizer.step()

    assert torch.equal(phi.detach()[[0, 2, 4]], before[[0, 2, 4]])
    assert not torch.equal(phi.detach()[1], before[1])
    assert not torch.equal(phi.detach()[3], before[3])
    assert phi.grad is None
    assert phi not in optimizer.state
    assert optimizer.state_dict()["state"] == {}


def test_build_optimizer_selects_pullback_group_route_and_moves_phi_embed():
    from vfe3.config import VFE3Config
    from vfe3.model.model import VFEModel
    from vfe3.train import build_optimizer, train_step

    cfg = VFE3Config(
        vocab_size=12,
        embed_dim=4,
        n_heads=2,
        max_seq_len=8,
        n_layers=1,
        gauge_group="block_glk",
        pos_phi="none",
        m_phi_update_mode="pullback_group",
        phi_precond_mode="pullback_per_block",
        transport_chart_max_norm=6.0,
        m_phi_lr=0.1,
    )
    torch.manual_seed(0)
    model = VFEModel(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    phi_before = model.prior_bank.phi_embed.detach().clone()
    tokens = torch.randint(0, 12, (4, 8))
    targets = torch.randint(0, 12, (4, 8))

    loss = train_step(model, optimizer, scheduler, tokens, targets, grad_clip=1.0)

    assert isinstance(optimizer, GaugeManifoldAdamW)
    assert torch.isfinite(torch.tensor(float(loss)))
    assert not torch.equal(model.prior_bank.phi_embed.detach(), phi_before)
    assert model.prior_bank.phi_embed not in optimizer.state


def test_pullback_group_trust_scales_the_learning_rate_weighted_right_factor(monkeypatch):
    import vfe3.gauge_optim as gauge_optim

    group = get_group("glk")(K=2, dtype=torch.float64)
    xi = torch.tensor([[4.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    monkeypatch.setattr(
        gauge_optim,
        "pullback_group_direction",
        lambda *args, **kwargs: _fixed_direction(xi),
    )

    candidate = stage_pullback_group_candidate(
        torch.ones_like(xi),
        torch.zeros_like(xi),
        group,
        learning_rate=0.5,
        trust_radius=0.25,
        chart_max_norm=5.0,
        bch_residual_max=1e-6,
        phi_precond_mode="pullback",
    )

    assert candidate.trust_scale.item() == pytest.approx(0.125)
    assert candidate.backtracking_reductions.item() == 0
    assert candidate.candidate_chart_norm.item() == pytest.approx(0.25)
    assert torch.allclose(
        candidate.candidate_phi,
        torch.tensor([[-0.25, 0.0, 0.0, 0.0]], dtype=torch.float64),
        atol=1e-15,
        rtol=0.0,
    )


def test_pullback_group_stage_passes_source_basis_and_registered_coordinate_layout(monkeypatch, device):
    import vfe3.gauge_optim as gauge_optim

    group = get_group("block_glk")(K=4, n_heads=2, dtype=torch.float32, device=device)
    seen = {}

    def record_direction(grad, phi, generators, **kwargs):
        seen["generators"] = generators
        seen["coordinate_layout"] = kwargs.get("coordinate_layout")
        return _fixed_direction(torch.zeros_like(grad))

    monkeypatch.setattr(gauge_optim, "pullback_group_direction", record_direction)
    stage_pullback_group_candidate(
        torch.ones(1, group.generators.shape[0], device=device),
        torch.zeros(1, group.generators.shape[0], device=device),
        group,
        learning_rate=0.0,
        trust_radius=0.25,
        chart_max_norm=5.0,
        bch_residual_max=1e-6,
        phi_precond_mode="pullback_per_block",
    )

    assert seen["generators"] is group.generators
    assert seen["coordinate_layout"] == "block_head_row_major"


def test_pullback_group_stage_normalizes_basis_for_registered_extension(device):
    import vfe3.geometry.phi_preconditioner as phi_preconditioner

    name = "test_stage_normalized_registered_extension"
    group = get_group("glk")(K=2, dtype=torch.float32, device=device)
    seen = {}

    @phi_preconditioner.register_phi_group_direction(name)
    def registered_direction(grad, phi, generators, *, irrep_dims=None, **kwargs):
        del phi, irrep_dims, kwargs
        seen["generators"] = generators
        return _fixed_direction(torch.zeros_like(grad))

    try:
        stage_pullback_group_candidate(
            torch.ones(1, group.generators.shape[0], device=device),
            torch.zeros(1, group.generators.shape[0], device=device),
            group,
            learning_rate=0.0,
            trust_radius=0.25,
            chart_max_norm=5.0,
            bch_residual_max=1e-6,
            phi_precond_mode=name,
        )
    finally:
        for support_name in (
            "_PHI_GROUP_DIRECTIONS",
            "_PHI_GROUP_DIRECTION_ACCEPTS_COORDINATE_LAYOUT",
            "_PHI_GROUP_DIRECTION_REQUIRES_SOURCE_BASIS",
        ):
            support = getattr(phi_preconditioner, support_name, None)
            if support is not None:
                support.pop(name, None)

    assert seen["generators"].dtype == torch.float64
    assert seen["generators"].device == group.generators.device
    assert seen["generators"] is not group.generators


@pytest.mark.parametrize(
    ("chart_norm", "phi_values", "delta_values", "reductions", "right_residual", "reversed_residual"),
    [
        (
            3.0,
            [0.4461370124222645, -0.37866876575372294, 1.1203006449654396, -2.7207532407183694],
            [0.0075751275839514645, -0.002453059541722277, -0.003516904099976695, -0.002529014357447788],
            6,
            5.48315e-7,
            1.083e-4,
        ),
        (
            5.0,
            [0.19176642751290957, -1.411275433700944, 3.477796870171524, -3.297947273280199],
            [0.008133453045635446, 0.0015903479064029997, -0.0025065446684178387, -0.001781968201968022],
            7,
            9.92428e-7,
            8.28e-5,
        ),
    ],
)
def test_pullback_group_bch_backtracks_and_certifies_right_product_order(
    monkeypatch,
    record_property,
    chart_norm,
    phi_values,
    delta_values,
    reductions,
    right_residual,
    reversed_residual,
):
    import vfe3.gauge_optim as gauge_optim

    group = get_group("glk")(K=2, dtype=torch.float64)
    phi = torch.tensor([phi_values], dtype=torch.float64)
    delta = torch.tensor([delta_values], dtype=torch.float64)
    monkeypatch.setattr(
        gauge_optim,
        "pullback_group_direction",
        lambda *args, **kwargs: _fixed_direction(-delta),
    )

    candidate = stage_pullback_group_candidate(
        torch.ones_like(phi),
        phi,
        group,
        learning_rate=1.0,
        trust_radius=1.0,
        chart_max_norm=6.0,
        bch_residual_max=1e-6,
        phi_precond_mode="pullback",
    )

    accepted_delta = delta / (2 ** reductions)
    embedded = lambda value: torch.einsum("...a,aij->...ij", value, group.generators)
    exp_candidate = torch.linalg.matrix_exp(embedded(candidate.candidate_phi))
    exp_phi = torch.linalg.matrix_exp(embedded(phi))
    exp_delta = torch.linalg.matrix_exp(embedded(accepted_delta))
    right = exp_phi @ exp_delta
    reversed_product = exp_delta @ exp_phi
    actual_right = torch.linalg.matrix_norm(exp_candidate - right) / torch.linalg.matrix_norm(right)
    actual_reversed = (
        torch.linalg.matrix_norm(exp_candidate - reversed_product)
        / torch.linalg.matrix_norm(reversed_product)
    )
    assert torch.linalg.matrix_norm(embedded(phi)).item() == pytest.approx(chart_norm, abs=1e-12)
    assert candidate.backtracking_reductions.item() == reductions
    assert candidate.group_product_residual.item() == pytest.approx(right_residual, rel=2e-6)
    assert candidate.group_product_residual.item() <= 1.0e-6
    assert actual_right.item() == pytest.approx(right_residual, rel=2e-6)
    assert actual_reversed.item() == pytest.approx(reversed_residual, rel=5e-4)
    record_property("current_chart_norm", chart_norm)
    record_property("float64_staging_residual", float(actual_right))


_REAL_COMMIT_CASES = {
    3.0: {
        "phi": [
            0.4461370124222645,
            -0.37866876575372294,
            1.1203006449654396,
            -2.7207532407183694,
        ],
        "delta": [
            0.0075751275839514645,
            -0.002453059541722277,
            -0.003516904099976695,
            -0.002529014357447788,
        ],
    },
    5.0: {
        "phi": [
            0.19176642751290957,
            -1.411275433700944,
            3.477796870171524,
            -3.297947273280199,
        ],
        "delta": [
            0.008133453045635446,
            0.0015903479064029997,
            -0.0025065446684178387,
            -0.001781968201968022,
        ],
    },
}


def _real_optimizer_commit_case(monkeypatch, chart_norm):
    import vfe3.gauge_optim as gauge_optim

    case = _REAL_COMMIT_CASES[chart_norm]
    group = get_group("glk")(K=2, dtype=torch.float32)
    initial = torch.zeros(2, 4, dtype=torch.float32)
    initial[0] = torch.tensor(case["phi"], dtype=torch.float32)
    parameter = nn.Parameter(initial.clone())
    inactive_before = parameter.detach()[1].clone()
    delta = torch.tensor([case["delta"]], dtype=torch.float64)
    monkeypatch.setattr(
        gauge_optim,
        "pullback_group_direction",
        lambda *args, **kwargs: _fixed_direction(-delta),
    )
    staged = []
    original_stage = gauge_optim.stage_pullback_group_candidate

    def _stage_spy(*args, **kwargs):
        candidate = original_stage(*args, **kwargs)
        staged.append(candidate)
        return candidate

    monkeypatch.setattr(gauge_optim, "stage_pullback_group_candidate", _stage_spy)
    optimizer = _make_optimizer(
        [{"params": [parameter], "lr": 1.0, "pullback_group": True, "weight_decay": 0.0}],
        group,
        trust_radius=1.0,
        chart_max_norm=6.0,
    )
    parameter.grad = torch.zeros_like(parameter)
    parameter.grad[0] = 1.0
    optimizer.step()

    assert len(staged) == 1
    candidate = staged[0]
    assert candidate.candidate_phi.dtype == torch.float64
    stored = parameter.detach()[0]
    staged_cast = candidate.candidate_phi[0].to(dtype=parameter.dtype)
    initial_active64 = initial[0].double().unsqueeze(0)
    factor_scale = candidate.trust_scale / torch.pow(
        torch.tensor(2.0, dtype=torch.float64),
        candidate.backtracking_reductions,
    )
    right_coordinates = (
        -candidate.direction.xi * factor_scale.unsqueeze(-1)
    )
    generators64 = group.generators.double()
    embedded = lambda value: torch.einsum("...a,aij->...ij", value, generators64)
    stored_exp = torch.linalg.matrix_exp(embedded(stored.double().unsqueeze(0)))
    initial_exp = torch.linalg.matrix_exp(embedded(initial_active64))
    right_factor_exp = torch.linalg.matrix_exp(embedded(right_coordinates))
    right_product = initial_exp @ right_factor_exp
    reversed_product = right_factor_exp @ initial_exp
    right_residual = (
        torch.linalg.matrix_norm(stored_exp - right_product)
        / torch.linalg.matrix_norm(right_product)
    )
    reversed_residual = (
        torch.linalg.matrix_norm(stored_exp - reversed_product)
        / torch.linalg.matrix_norm(reversed_product)
    )
    return {
        "stored_matches_staged_cast": torch.equal(stored, staged_cast),
        "inactive_unchanged": torch.equal(parameter.detach()[1], inactive_before),
        "right_residual": float(right_residual),
        "reversed_residual": float(reversed_residual),
        "observed_chart_norm": float(torch.linalg.vector_norm(initial[0].double())),
    }


@pytest.mark.parametrize("chart_norm", [3.0, 5.0])
def test_optimizer_commit_is_single_float32_cast_with_certified_right_product(
    monkeypatch,
    record_property,
    chart_norm,
):
    committed = _real_optimizer_commit_case(monkeypatch, chart_norm)
    record_property("current_chart_norm", committed["observed_chart_norm"])
    record_property("float32_committed_residual", committed["right_residual"])
    record_property("float32_committed_reversed_residual", committed["reversed_residual"])
    assert committed["observed_chart_norm"] == pytest.approx(chart_norm, abs=2.0e-7)
    assert committed["stored_matches_staged_cast"]
    assert committed["inactive_unchanged"]
    assert committed["right_residual"] <= 5.0e-6
    assert committed["reversed_residual"] >= 5.0e-5


def test_pullback_group_candidate_rejects_chart_bound(monkeypatch):
    import vfe3.gauge_optim as gauge_optim

    group = get_group("glk")(K=2, dtype=torch.float64)
    phi = torch.tensor([[0.99, 0.0, 0.0, 0.0]], dtype=torch.float64)
    xi = torch.tensor([[-0.2, 0.0, 0.0, 0.0]], dtype=torch.float64)
    monkeypatch.setattr(
        gauge_optim,
        "pullback_group_direction",
        lambda *args, **kwargs: _fixed_direction(xi),
    )

    with pytest.raises(FloatingPointError, match="chart norm"):
        stage_pullback_group_candidate(
            torch.ones_like(phi),
            phi,
            group,
            learning_rate=1.0,
            trust_radius=1.0,
            chart_max_norm=1.0,
            bch_residual_max=1e-6,
            phi_precond_mode="pullback",
        )


def test_pullback_group_step_rejects_two_parameter_groups_atomically(monkeypatch):
    import vfe3.gauge_optim as gauge_optim

    group = get_group("glk")(K=2, dtype=torch.float64)
    good = nn.Parameter(torch.zeros(1, 4, dtype=torch.float64))
    bad = nn.Parameter(torch.tensor([[0.99, 0.0, 0.0, 0.0]], dtype=torch.float64))
    xi = torch.tensor([[-0.2, 0.0, 0.0, 0.0]], dtype=torch.float64)
    monkeypatch.setattr(
        gauge_optim,
        "pullback_group_direction",
        lambda *args, **kwargs: _fixed_direction(xi),
    )
    optimizer = _make_optimizer(
        [
            {"params": [good], "lr": 1.0, "pullback_group": True, "weight_decay": 0.0},
            {"params": [bad], "lr": 1.0, "pullback_group": True, "weight_decay": 0.0},
        ],
        group,
        trust_radius=1.0,
        chart_max_norm=1.0,
    )
    good_before = good.detach().clone()
    bad_before = bad.detach().clone()
    good.grad = torch.ones_like(good)
    bad.grad = torch.ones_like(bad)

    with pytest.raises(FloatingPointError, match="chart norm"):
        optimizer.step()

    assert torch.equal(good, good_before)
    assert torch.equal(bad, bad_before)
    assert good.grad is not None and bad.grad is not None
    assert optimizer.state_dict()["state"] == {}


@pytest.mark.parametrize(
    "bad_value",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "positive_inf", "negative_inf"],
)
def test_pullback_group_step_rejects_nonfinite_covector_before_any_staging_or_mutation(
    monkeypatch,
    bad_value,
):
    import vfe3.gauge_optim as gauge_optim

    group = get_group("glk")(K=2, dtype=torch.float64)
    finite = nn.Parameter(torch.zeros(1, 4, dtype=torch.float64))
    mixed = nn.Parameter(torch.zeros(2, 4, dtype=torch.float64))
    optimizer = _make_optimizer(
        [
            {"params": [finite], "lr": 0.1, "pullback_group": True, "weight_decay": 0.0},
            {"params": [mixed], "lr": 0.1, "pullback_group": True, "weight_decay": 0.0},
        ],
        group,
    )
    finite.grad = torch.ones_like(finite)
    mixed.grad = torch.zeros_like(mixed)
    mixed.grad[0] = 1.0
    mixed.grad[1, 0] = bad_value
    parameter_before = [parameter.detach().clone() for parameter in (finite, mixed)]
    gradient_before = [parameter.grad.clone() for parameter in (finite, mixed)]
    staging_calls = []
    original_stage = gauge_optim.stage_pullback_group_candidate

    def _stage_spy(*args, **kwargs):
        staging_calls.append(1)
        return original_stage(*args, **kwargs)

    monkeypatch.setattr(gauge_optim, "stage_pullback_group_candidate", _stage_spy)

    with pytest.raises(FloatingPointError, match="nonfinite phi covector"):
        optimizer.step()

    assert staging_calls == []
    for parameter, expected_parameter, expected_gradient in zip(
        (finite, mixed),
        parameter_before,
        gradient_before,
    ):
        assert torch.equal(parameter, expected_parameter)
        assert parameter.grad is not None
        torch.testing.assert_close(
            parameter.grad,
            expected_gradient,
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )
    assert optimizer.state_dict()["state"] == {}


def test_state_dict_roundtrips_omega_reorth_cadence(monkeypatch):
    import vfe3.gauge_optim as gauge_optim_mod
    from vfe3.geometry.groups import get_group

    group = get_group("so_k")(K=4)

    def _optimizer(U):
        return _make_optimizer(
            [{"params": [U], "lr": 0.05, "omega": True, "weight_decay": 0.0}],
            group,
            omega_reorth_every=3,
        )

    source_U = nn.Parameter(torch.eye(4).expand(2, 4, 4).contiguous())
    source_opt = _optimizer(source_U)
    source_U.grad = torch.zeros_like(source_U)
    source_U.grad[1, 0, 1] = 1.0
    source_opt.step()
    source_opt.step()
    state = source_opt.state_dict()
    assert state["optimizer_extra"]["omega_step"] == 2
    assert state["optimizer_extra"]["omega_dirty_format"] == 1

    calls = []
    original_polar = gauge_optim_mod._polar_orthogonalize

    def _spy(U):
        calls.append(1)
        return original_polar(U)

    monkeypatch.setattr(gauge_optim_mod, "_polar_orthogonalize", _spy)
    resumed_U = nn.Parameter(torch.eye(4).expand(2, 4, 4).contiguous())
    resumed_opt = _optimizer(resumed_U)
    resumed_opt.load_state_dict(state)
    resumed_opt.step()
    assert resumed_opt._omega_step == 3
    assert calls == [1]

    legacy = {k: v for k, v in state.items() if k != "optimizer_extra"}
    legacy_U = nn.Parameter(torch.eye(4).expand(2, 4, 4).contiguous())
    legacy_opt = _optimizer(legacy_U)
    with pytest.warns(UserWarning, match="non-exact resume"):
        legacy_opt.load_state_dict(legacy)
    assert legacy_opt._omega_step == 0


def test_omega_reorthogonalizes_only_dirty_rows(monkeypatch, device):
    import vfe3.gauge_optim as gauge_optim_mod
    from vfe3.geometry.groups import get_group

    group = get_group("so_k")(K=4, device=device)
    U = nn.Parameter(torch.eye(4, device=device).expand(5, 4, 4).clone())
    with torch.no_grad():
        U[0] *= 1.20
        U[4] *= 0.80
    untouched_before = U.detach()[[0, 4]].clone()
    seen_shapes = []
    original_polar = gauge_optim_mod._polar_orthogonalize

    def _spy(rows):
        seen_shapes.append(tuple(rows.shape))
        return original_polar(rows)

    monkeypatch.setattr(gauge_optim_mod, "_polar_orthogonalize", _spy)
    opt = _make_optimizer(
        [{"params": [U], "lr": 0.05, "omega": True, "weight_decay": 0.0}],
        group,
        omega_reorth_every=2,
    )
    U.grad = torch.zeros_like(U)
    U.grad[1, 0, 1] = 1.0
    opt.step()
    U.grad = torch.zeros_like(U)
    U.grad[3, 1, 2] = 1.0
    opt.step()

    assert seen_shapes == [(2, 4, 4)]
    assert opt.state[U]["omega_dirty"].device == U.device
    assert not bool(opt.state[U]["omega_dirty"].any())
    assert torch.equal(U.detach()[[0, 4]], untouched_before)


def test_dirty_rows_clear_after_cadence():
    from vfe3.geometry.groups import get_group

    group = get_group("so_k")(K=4)

    def _optimizer(parameter):
        return _make_optimizer(
            [{"params": [parameter], "lr": 0.05, "omega": True, "weight_decay": 0.0}],
            group,
            omega_reorth_every=2,
        )

    source_U = nn.Parameter(torch.eye(4).expand(4, 4, 4).clone())
    source_opt = _optimizer(source_U)
    source_U.grad = torch.zeros_like(source_U)
    source_U.grad[1, 0, 1] = 1.0
    source_opt.step()
    assert torch.equal(
        source_opt.state[source_U]["omega_dirty"],
        torch.tensor([False, True, False, False]),
    )
    assert source_opt.state[source_U]["omega_dirty"].dtype == torch.bool

    checkpoint = source_opt.state_dict()
    resumed_U = nn.Parameter(source_U.detach().clone())
    resumed_opt = _optimizer(resumed_U)
    resumed_opt.load_state_dict(checkpoint)
    assert torch.equal(
        resumed_opt.state[resumed_U]["omega_dirty"],
        torch.tensor([False, True, False, False]),
    )
    assert resumed_opt.state[resumed_U]["omega_dirty"].dtype == torch.bool
    assert resumed_opt.state[resumed_U]["omega_dirty"].device == resumed_U.device
    resumed_U.grad = torch.zeros_like(resumed_U)
    resumed_U.grad[2, 1, 2] = 1.0
    resumed_opt.step()

    assert not bool(resumed_opt.state[resumed_U]["omega_dirty"].any())


def test_pre_o5_checkpoint_with_omega_step_marks_every_row_dirty():
    from vfe3.geometry.groups import get_group

    group = get_group("so_k")(K=4)

    def _optimizer(parameter):
        return _make_optimizer(
            [{"params": [parameter], "lr": 0.05, "omega": True, "weight_decay": 0.0}],
            group,
            omega_reorth_every=3,
        )

    source_U = nn.Parameter(torch.eye(4).expand(5, 4, 4).clone())
    source_opt = _optimizer(source_U)
    source_U.grad = torch.zeros_like(source_U)
    source_U.grad[2, 0, 1] = 1.0
    source_opt.step()
    checkpoint = source_opt.state_dict()
    checkpoint["optimizer_extra"].pop("omega_dirty_format")     # pre-O5: step exists, marker absent
    for state in checkpoint["state"].values():
        state.pop("omega_dirty", None)                            # pre-O5 carried no dirty mask

    resumed_U = nn.Parameter(source_U.detach().clone())
    resumed_opt = _optimizer(resumed_U)
    with pytest.warns(UserWarning, match="all omega rows were marked dirty"):
        resumed_opt.load_state_dict(checkpoint)

    dirty = resumed_opt.state[resumed_U]["omega_dirty"]
    assert resumed_opt._omega_step == 1                           # cadence position is preserved
    assert dirty.dtype == torch.bool and dirty.device == resumed_U.device
    assert torch.equal(dirty, torch.ones(5, dtype=torch.bool))


def test_current_checkpoint_without_dirty_mask_defaults_clean():
    from vfe3.geometry.groups import get_group

    group = get_group("so_k")(K=4)
    source_U = nn.Parameter(torch.eye(4).expand(3, 4, 4).clone())
    source_opt = _make_optimizer(
        [{"params": [source_U], "lr": 0.0, "omega": True, "weight_decay": 0.0}],
        group,
        omega_reorth_every=2,
    )
    checkpoint = source_opt.state_dict()
    assert checkpoint["optimizer_extra"]["omega_dirty_format"] == 1

    resumed_U = nn.Parameter(source_U.detach().clone())
    resumed_opt = _make_optimizer(
        [{"params": [resumed_U], "lr": 0.0, "omega": True, "weight_decay": 0.0}],
        group,
        omega_reorth_every=2,
    )
    resumed_opt.load_state_dict(checkpoint)

    assert torch.equal(
        resumed_opt.state[resumed_U]["omega_dirty"], torch.zeros(3, dtype=torch.bool))


def test_compact_omega_condition_captures_cross_block_scale():
    from vfe3.geometry.groups import get_group

    group = get_group("block_glk")(K=4, n_heads=2)
    blocks = torch.stack([100.0 * torch.eye(2), 0.01 * torch.eye(2)]).unsqueeze(0)
    U = nn.Parameter(blocks)
    opt = _make_optimizer(
        [{"params": [U], "lr": 0.0, "omega": True, "weight_decay": 0.0}],
        group,
    )
    opt._collect_gauge_diag = True
    U.grad = torch.ones_like(U)

    opt.step()

    assert opt._gauge_diag["omega_condition_max"] == pytest.approx(1.0e4, rel=1e-6)


def test_zero_omega_gradient_is_consumed_without_empty_diagnostics():
    from vfe3.geometry.groups import get_group

    group = get_group("glk")(K=3)
    omega = nn.Parameter(torch.eye(3).repeat(5, 1, 1))
    optimizer = _make_optimizer(
        [{"params": [omega], "lr": 0.05, "omega": True, "weight_decay": 0.0}],
        group,
    )
    optimizer._collect_gauge_diag = True
    optimizer._gauge_diag = {"omega_condition_max": 123.0}
    before = omega.detach().clone()
    omega.grad = torch.zeros_like(omega)

    optimizer.step()

    assert torch.equal(omega, before)
    assert omega.grad is None
    assert optimizer._gauge_diag == {}
    assert not optimizer.state


def test_attempted_diagnostic_step_clears_stale_gauge_health():
    group = get_group("block_glk")(K=4, n_heads=2)
    phi = nn.Parameter(torch.zeros(3, group.generators.shape[0]))
    opt = _make_optimizer(
        [{"params": [phi], "lr": 0.0, "pullback_group": True, "weight_decay": 0.0}],
        group,
        phi_precond_mode="pullback_per_block",
    )
    opt._collect_gauge_diag = True
    opt._gauge_diag = {"phi_pullback_damped_gen_cond_max": 123.0}
    phi.grad = None

    opt.step()

    assert opt._gauge_diag == {}
