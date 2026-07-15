"""Phi-chart, BCH-fidelity, and fp64 flatness-reference regressions."""

import torch

from vfe3.config import VFE3Config
from vfe3.gauge_optim import project_phi_parameter_rows_
from vfe3.geometry.groups import get_group
from vfe3.metrics import (
    bch_fidelity_statistics,
    flatness_reference_statistics,
    phi_chart_statistics,
)
from vfe3.model.model import VFEModel
from vfe3.run_artifacts import collect_phi_numerics
from vfe3.train import build_optimizer, train_step


def _gl2():
    return get_group("glk")(2)


def test_flatness_reference_reports_relative_and_fp64_residuals() -> None:
    group = _gl2()
    phi = torch.tensor(
        [[[7.0, 2.0, 0.0, -6.0],
          [-5.0, 0.0, 3.0, 6.0],
          [4.0, -2.0, 1.0, -7.0],
          [0.0, 5.0, -3.0, 2.0]]],
        dtype=torch.float32,
    )

    stats = flatness_reference_statistics(phi, group.generators)

    assert stats["numerical_holonomy_fp64_rel"] < stats["numerical_holonomy_fp32_rel"]
    assert stats["inverse_consistency_fp64_rel"] < stats["inverse_consistency_fp32_rel"]
    assert stats["numerical_cocycle_fp64_rel"] < stats["numerical_cocycle_fp32_rel"]


def test_bch_fidelity_detects_large_chart_failure() -> None:
    group = _gl2()
    phi_x = torch.tensor([[7.0, 5.0, 0.0, -6.0]])
    phi_y = torch.tensor([[-5.0, 0.0, 6.0, 7.0]])

    stats = bch_fidelity_statistics(phi_x, phi_y, group.generators, order=4)

    assert stats["bch_relative_error_max"] > 0.1
    assert stats["bch_norm_amplification_max"] > 1.0


def test_phi_chart_statistics_exposes_clamp_and_condition_quantiles() -> None:
    group = _gl2()
    phi = torch.tensor([[0.0, 0.0, 0.0, 0.0], [30.0, 0.0, 0.0, -30.0]])

    stats = phi_chart_statistics(phi, group.generators, max_norm=20.0)

    assert stats["phi_matrix_norm_max"] > 20.0
    assert stats["phi_exp_clamp_frac"] == 0.5
    assert 0.0 < stats["phi_exp_scale_min"] < 1.0
    assert stats["vertex_cond_p99"] >= stats["vertex_cond_median"]


def test_final_phi_numerics_collects_flatness_and_bch_without_state_change() -> None:
    cfg = VFE3Config(
        vocab_size=8,
        embed_dim=4,
        n_heads=1,
        max_seq_len=4,
        n_layers=1,
        n_e_steps=1,
        e_phi_lr=0.0,
        pos_phi="learned",
        pos_phi_compose="bch",
    )
    model = VFEModel(cfg)
    model.train()
    tokens = torch.tensor([[0, 1, 2, 3]])
    state_before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    rng_before = torch.get_rng_state().clone()

    record = collect_phi_numerics(model, tokens)

    assert set(record) >= {"chart", "flatness", "bch_fidelity"}
    assert model.training
    assert torch.equal(torch.get_rng_state(), rng_before)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state_before[name], rtol=0.0, atol=0.0)


def test_project_phi_rows_bounds_all_belief_and_model_frame_tables() -> None:
    cfg = VFE3Config(
        vocab_size=8,
        embed_dim=4,
        n_heads=1,
        max_seq_len=4,
        n_layers=1,
        prior_source="model_channel",
        s_frame_mode="phi_tilde",
        s_e_step=True,
        lambda_h=1.0,
        lambda_gamma=1.0,
        pos_phi="learned",
    )
    model = VFEModel(cfg)
    tables = [
        model.prior_bank.phi_embed,
        model.prior_bank.s_phi_embed,
        model.pos_phi_free,
        model.s_pos_phi_free,
    ]
    with torch.no_grad():
        for table in tables:
            table.fill_(5.0)

    stats = project_phi_parameter_rows_(model, 2.0, chunk_rows=3)

    for table in tables:
        embedded = torch.einsum("...a,aij->...ij", table, model.group.generators)
        assert float(
            torch.linalg.matrix_norm(embedded, ord="fro", dim=(-2, -1)).max().detach()
        ) <= 2.0 + 1e-5
    assert stats["phi_chart_projected_fraction"] == 1.0
    assert stats["phi_chart_projected_rows"] == sum(table.shape[0] for table in tables)


def test_disabled_chart_projection_is_not_called(monkeypatch) -> None:
    import vfe3.train as train_module

    cfg = VFE3Config(
        vocab_size=8,
        embed_dim=4,
        n_heads=1,
        max_seq_len=4,
        n_layers=1,
        n_e_steps=1,
        e_phi_lr=0.0,
        pos_phi="none",
        phi_mstep_max_matrix_norm=None,
    )
    model = VFEModel(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    tokens = torch.tensor([[0, 1, 2, 3]])
    targets = torch.tensor([[1, 2, 3, 4]])

    def _unexpected(*args, **kwargs):
        raise AssertionError("disabled projection was called")

    monkeypatch.setattr(train_module, "project_phi_parameter_rows_", _unexpected)
    train_step(model, optimizer, scheduler, tokens, targets)
