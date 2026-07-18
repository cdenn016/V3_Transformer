r"""Tests for the pullback-group M-step diagnostics.

  * GaugeManifoldAdamW stashes the gated ridge-direction, pullback, trust, and chart diagnostics.
  * the diagnostics are silent (no compute, empty _gauge_diag) unless _collect_gauge_diag is set.
  * train() writes the cumulative wall_clock_s column into metrics.csv.
  * the wall-clock convergence figure renders.

Device-agnostic (CPU). Figures use the Agg backend.
"""
import csv as _csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
import torch
from torch import nn

from vfe3.config import VFE3Config
from vfe3.gauge_optim import (
    GaugeManifoldAdamW,
    PullbackGroupCandidate,
)
from vfe3.geometry.groups import get_group
from vfe3.geometry.phi_preconditioner import PullbackGroupDirectionResult
from vfe3.model.model import VFEModel
from vfe3.train import build_optimizer, train
from vfe3.run_artifacts import RunArtifacts
from vfe3.viz import figures as figs

DEVICE = torch.device(os.environ.get("VFE3_TEST_DEVICE", "cpu"))

_PHI_DIAGNOSTIC_KEYS = {
    "phi_ridge_direction_cosine_mean",
    "phi_pullback_damped_gen_cond_median",
    "phi_pullback_damped_gen_cond_max",
    "phi_group_trust_scale_mean",
    "phi_group_trust_scale_min",
    "phi_group_active_rows",
    "phi_group_chart_norm_max",
}


def _pullback_opt_with_grads() -> GaugeManifoldAdamW:
    r"""A GaugeManifoldAdamW with phi_embed.grad populated by one backward pass."""
    cfg = VFE3Config(vocab_size=32, embed_dim=4, n_heads=2, max_seq_len=8,
                     gauge_group="block_glk", m_phi_update_mode="pullback_group",
                     phi_precond_mode="pullback_per_block", transport_chart_max_norm=6.0,
                     m_phi_lr=0.01, e_phi_lr=0.0)
    torch.manual_seed(0)
    model = VFEModel(cfg).to(DEVICE)
    opt = build_optimizer(model, cfg)
    tok = torch.randint(0, 32, (2, 8), device=DEVICE)
    tgt = torch.randint(0, 32, (2, 8), device=DEVICE)
    _, loss, _ = model(tok, tgt)
    loss.backward()
    return opt


def test_gauge_diag_pullback_collects_exact_canonical_keys():
    opt = _pullback_opt_with_grads()
    assert isinstance(opt, GaugeManifoldAdamW)
    opt._collect_gauge_diag = True
    opt.step()
    gd = opt._gauge_diag
    assert set(gd) == _PHI_DIAGNOSTIC_KEYS
    assert -1.0 - 1e-6 <= gd["phi_ridge_direction_cosine_mean"] <= 1.0 + 1e-6
    assert gd["phi_ridge_direction_cosine_mean"] > 0.0
    assert gd["phi_pullback_damped_gen_cond_median"] >= 1.0 - 1e-6
    assert (
        gd["phi_pullback_damped_gen_cond_max"]
        >= gd["phi_pullback_damped_gen_cond_median"] - 1e-6
    )
    assert 0.0 < gd["phi_group_trust_scale_min"] <= gd["phi_group_trust_scale_mean"] <= 1.0
    assert gd["phi_group_active_rows"] > 0.0
    assert gd["phi_group_chart_norm_max"] >= 0.0


def test_gauge_diag_reports_finite_damped_pullback_condition():
    opt = _pullback_opt_with_grads()
    opt._collect_gauge_diag = True
    opt.step()
    condition = opt._gauge_diag["phi_pullback_damped_gen_cond_median"]
    assert torch.isfinite(torch.tensor(condition))
    assert condition >= 1.0 - 1e-6


def _diagnostic_candidate(phi: torch.Tensor) -> PullbackGroupCandidate:
    rows, width = phi.shape
    value = torch.ones(rows, dtype=torch.float64, device=phi.device)
    direction = PullbackGroupDirectionResult(
        v_phi=torch.ones(rows, width, dtype=torch.float64, device=phi.device),
        xi=torch.ones(rows, width, dtype=torch.float64, device=phi.device),
        min_undamped_generalized_eigenvalue=value,
        undamped_generalized_condition=value,
        damped_generalized_condition=torch.linspace(
            2.0, 4.0, rows, dtype=torch.float64, device=phi.device
        ),
        scaled_solve_residual=torch.zeros_like(value),
        series_order=40,
    )
    return PullbackGroupCandidate(
        candidate_phi=phi.double().clone(),
        trust_scale=torch.linspace(0.8, 0.6, rows, dtype=torch.float64, device=phi.device),
        backtracking_reductions=torch.arange(2, 0, -1, dtype=torch.long, device=phi.device),
        candidate_chart_norm=torch.linspace(1.25, 2.5, rows, dtype=torch.float64, device=phi.device),
        group_product_residual=torch.zeros_like(value),
        direction=direction,
    )


def _two_row_pullback_optimizer() -> tuple[GaugeManifoldAdamW, nn.Parameter]:
    group = get_group("glk")(K=2)
    phi = nn.Parameter(torch.zeros(2, group.generators.shape[0]))
    optimizer = GaugeManifoldAdamW(
        [{"params": [phi], "lr": 0.1, "pullback_group": True, "weight_decay": 0.0}],
        group,
        phi_group_trust_radius=0.1,
        phi_chart_max_norm=5.0,
        phi_bch_residual_max=1e-6,
        phi_precond_mode="pullback",
        weight_decay=0.0,
    )
    phi.grad = torch.ones_like(phi)
    return optimizer, phi


def test_gauge_diag_reports_accepted_effective_scale_after_backtracking(monkeypatch):
    import vfe3.gauge_optim as gauge_optim

    optimizer, _ = _two_row_pullback_optimizer()
    monkeypatch.setattr(
        gauge_optim,
        "stage_pullback_group_candidate",
        lambda grad, phi, *args, **kwargs: _diagnostic_candidate(phi),
    )
    optimizer._collect_gauge_diag = True

    optimizer.step()

    assert optimizer._gauge_diag["phi_group_trust_scale_mean"] == pytest.approx(0.25)
    assert optimizer._gauge_diag["phi_group_trust_scale_min"] == pytest.approx(0.2)
    assert optimizer._gauge_diag["phi_group_active_rows"] == 2.0
    assert optimizer._gauge_diag["phi_group_chart_norm_max"] == pytest.approx(2.5)


def test_gauge_diag_ridge_cosine_is_global_row_mean(monkeypatch):
    import vfe3.gauge_optim as gauge_optim

    group = get_group("glk")(K=2)
    one_row = nn.Parameter(torch.zeros(1, 4))
    three_rows = nn.Parameter(torch.zeros(3, 4))
    optimizer = GaugeManifoldAdamW(
        [
            {"params": [one_row], "lr": 0.1, "pullback_group": True, "weight_decay": 0.0},
            {"params": [three_rows], "lr": 0.1, "pullback_group": True, "weight_decay": 0.0},
        ],
        group,
        phi_group_trust_radius=0.1,
        phi_chart_max_norm=5.0,
        phi_bch_residual_max=1e-6,
        phi_precond_mode="pullback",
        weight_decay=0.0,
    )
    one_row.grad = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    three_rows.grad = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(3, -1).clone()

    def _candidate(grad, phi, *args, **kwargs):
        rows = phi.shape[0]
        value = torch.ones(rows, dtype=torch.float64)
        v_phi = grad.double().clone() if rows == 1 else torch.roll(grad.double(), shifts=1, dims=-1)
        return PullbackGroupCandidate(
            candidate_phi=phi.double().clone(),
            trust_scale=value,
            backtracking_reductions=torch.zeros(rows, dtype=torch.long),
            candidate_chart_norm=value,
            group_product_residual=torch.zeros_like(value),
            direction=PullbackGroupDirectionResult(
                v_phi=v_phi,
                xi=v_phi,
                min_undamped_generalized_eigenvalue=value,
                undamped_generalized_condition=value,
                damped_generalized_condition=value,
                scaled_solve_residual=torch.zeros_like(value),
                series_order=40,
            ),
        )

    monkeypatch.setattr(gauge_optim, "stage_pullback_group_candidate", _candidate)
    optimizer._collect_gauge_diag = True

    optimizer.step()

    assert optimizer._gauge_diag["phi_ridge_direction_cosine_mean"] == pytest.approx(0.25)


def test_gauge_diag_ridge_cosine_preserves_tiny_aligned_float64_directions(monkeypatch):
    import vfe3.gauge_optim as gauge_optim

    optimizer, phi = _two_row_pullback_optimizer()
    phi.grad = torch.full_like(phi, 1e-12)

    def _candidate(grad, chart, *args, **kwargs):
        candidate = _diagnostic_candidate(chart)
        tiny = grad.double().clone()
        return PullbackGroupCandidate(
            candidate_phi=candidate.candidate_phi,
            trust_scale=candidate.trust_scale,
            backtracking_reductions=candidate.backtracking_reductions,
            candidate_chart_norm=candidate.candidate_chart_norm,
            group_product_residual=candidate.group_product_residual,
            direction=PullbackGroupDirectionResult(
                v_phi=tiny,
                xi=tiny,
                min_undamped_generalized_eigenvalue=(
                    candidate.direction.min_undamped_generalized_eigenvalue
                ),
                undamped_generalized_condition=candidate.direction.undamped_generalized_condition,
                damped_generalized_condition=candidate.direction.damped_generalized_condition,
                scaled_solve_residual=candidate.direction.scaled_solve_residual,
                series_order=candidate.direction.series_order,
            ),
        )

    monkeypatch.setattr(gauge_optim, "stage_pullback_group_candidate", _candidate)
    optimizer._collect_gauge_diag = True

    optimizer.step()

    assert optimizer._gauge_diag["phi_ridge_direction_cosine_mean"] == pytest.approx(1.0)


def test_historical_csv_diagnostics_migrate_in_figure_loader(tmp_path):
    from vfe3.viz.figure_worker import _history_from_csv

    path = tmp_path / "metrics.csv"
    path.write_text(
        "step,cos_nat_phi,pullback_cond_median,pullback_cond_max\n"
        "1,0.75,2.0,3.0\n",
        encoding="utf-8",
    )

    history = _history_from_csv(path)

    assert history[0]["phi_ridge_direction_cosine_mean"] == pytest.approx(0.75)
    assert history[0]["phi_pullback_damped_gen_cond_median"] == pytest.approx(2.0)
    assert history[0]["phi_pullback_damped_gen_cond_max"] == pytest.approx(3.0)


def test_silent_pullback_step_does_not_host_convert_diagnostic_tensors(monkeypatch):
    import vfe3.gauge_optim as gauge_optim

    optimizer, _ = _two_row_pullback_optimizer()
    monkeypatch.setattr(
        gauge_optim,
        "stage_pullback_group_candidate",
        lambda grad, phi, *args, **kwargs: _diagnostic_candidate(phi),
    )
    host_calls = []
    real_float = torch.Tensor.__float__
    real_int = torch.Tensor.__int__
    real_item = torch.Tensor.item
    real_tolist = torch.Tensor.tolist
    real_cpu = torch.Tensor.cpu

    def _float_spy(value):
        host_calls.append("float")
        return real_float(value)

    def _int_spy(value):
        host_calls.append("int")
        return real_int(value)

    def _item_spy(value, *args, **kwargs):
        host_calls.append("item")
        return real_item(value, *args, **kwargs)

    def _tolist_spy(value, *args, **kwargs):
        host_calls.append("tolist")
        return real_tolist(value, *args, **kwargs)

    def _cpu_spy(value, *args, **kwargs):
        host_calls.append("cpu")
        return real_cpu(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "__float__", _float_spy)
    monkeypatch.setattr(torch.Tensor, "__int__", _int_spy)
    monkeypatch.setattr(torch.Tensor, "item", _item_spy)
    monkeypatch.setattr(torch.Tensor, "tolist", _tolist_spy)
    monkeypatch.setattr(torch.Tensor, "cpu", _cpu_spy)

    optimizer.step()

    assert host_calls == []
    assert optimizer._gauge_diag == {}


def test_gauge_diag_attempt_does_not_create_phi_optimizer_state():
    opt = _pullback_opt_with_grads()
    opt._collect_gauge_diag = True
    opt.step()
    pullback_parameters = [
        parameter
        for group in opt.param_groups if group.get("pullback_group", False)
        for parameter in group["params"]
    ]
    assert all(parameter not in opt.state for parameter in pullback_parameters)


def test_gauge_diag_silent_when_flag_off():
    opt = _pullback_opt_with_grads()
    opt.step()                                                      # _collect_gauge_diag defaults False
    assert opt._gauge_diag == {}


def test_wall_clock_column_in_metrics(tmp_path):
    torch.manual_seed(0)
    cfg = VFE3Config(vocab_size=32, embed_dim=4, n_heads=2, max_seq_len=8, max_steps=4,
                     log_interval=2, eval_interval=2, batch_size=2)
    model = VFEModel(cfg).to(DEVICE)
    batches = [(torch.randint(0, 32, (2, 8), device=DEVICE),
                torch.randint(0, 32, (2, 8), device=DEVICE)) for _ in range(4)]
    art = RunArtifacts(str(tmp_path), cfg, model, dataset="synthetic-period3", device=str(DEVICE))
    train(model, batches, cfg, n_steps=4, log_interval=2, eval_interval=2,
          val_loader=batches, artifacts=art, device=DEVICE, generate_samples=False)
    with open(tmp_path / "metrics.csv", newline="", encoding="utf-8") as fh:
        rows = list(_csv.DictReader(fh))
    assert rows and "wall_clock_s" in rows[0]
    assert float(rows[0]["wall_clock_s"]) >= 0.0


def test_gauge_diag_columns_rectangular_through_train(tmp_path):
    r"""End-to-end: a pullback-group run writes the fixed diagnostic key set into
    metrics.csv, present in EVERY row (the fixed-key-set / NaN-default contract; log_metrics locks
    fieldnames on row 0, so a key first appearing later would break the CSV)."""
    torch.manual_seed(0)
    cfg = VFE3Config(vocab_size=32, embed_dim=4, n_heads=2, max_seq_len=8, max_steps=4,
                     log_interval=2, eval_interval=2, batch_size=2, gauge_group="block_glk",
                     m_phi_update_mode="pullback_group", phi_precond_mode="pullback_per_block",
                     transport_chart_max_norm=6.0,
                     m_phi_lr=0.01, e_phi_lr=0.0)
    model = VFEModel(cfg).to(DEVICE)
    batches = [(torch.randint(0, 32, (2, 8), device=DEVICE),
                torch.randint(0, 32, (2, 8), device=DEVICE)) for _ in range(4)]
    art = RunArtifacts(str(tmp_path), cfg, model, dataset="synthetic-period3", device=str(DEVICE))
    train(model, batches, cfg, n_steps=4, log_interval=2, eval_interval=2,
          val_loader=batches, artifacts=art, device=DEVICE, generate_samples=False)
    with open(tmp_path / "metrics.csv", newline="", encoding="utf-8") as fh:
        rows = list(_csv.DictReader(fh))
    assert rows
    for r in rows:                                                  # rectangular: keys in row 0 and all rows
        assert _PHI_DIAGNOSTIC_KEYS <= set(r)
        assert "wall_clock_s" in r


def test_optimizer_geometry_uses_pullback_and_ridge_direction_labels():
    history = {
        "step": [1, 2],
        "phi_ridge_direction_cosine_mean": [0.8, 0.9],
        "phi_pullback_damped_gen_cond_median": [2.0, 2.5],
        "phi_pullback_damped_gen_cond_max": [3.0, 3.5],
        "phi_group_trust_scale_mean": [0.5, 0.6],
        "phi_group_trust_scale_min": [0.25, 0.3],
        "phi_group_active_rows": [4.0, 5.0],
        "phi_group_chart_norm_max": [1.0, 1.2],
    }

    fig = figs.plot_optimizer_geometry(history)

    text = " ".join(
        [axis.get_title() + " " + axis.get_ylabel() for axis in fig.axes]
        + [line.get_label() for axis in fig.axes for line in axis.get_lines()]
    ).lower()
    assert "ridge" in text and "pullback" in text
    assert "natural gradient" not in text and "natural-gradient" not in text
    plt.close(fig)


def test_wallclock_convergence_figure_renders():
    arms = [{"label": "pullback", "step": [10, 20, 30], "val_ppl": [40, 30, 25],
             "wall_clock_s": [1.0, 2.5, 4.0]},
            {"label": "adamw", "step": [10, 20, 30], "val_ppl": [42, 33, 28],
             "wall_clock_s": [0.8, 1.6, 2.5]}]
    fig = figs.plot_wallclock_convergence(arms)
    assert fig is not None
    plt.close(fig)
