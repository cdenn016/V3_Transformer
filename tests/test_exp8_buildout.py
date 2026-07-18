r"""Tests for the 2026-06-22 EXP-8 build-out (pullback nat-grad gauge M-step + LR, D1):

  * GaugeManifoldAdamW stashes the GATED training-time diagnostics -- cos(nat,grad) (=1 for the
    conformal killing rescale, a valid cosine for pullback) and the pullback metric condition number.
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
import torch

from vfe3.config import VFE3Config
from vfe3.gauge_optim import GaugeManifoldAdamW
from vfe3.model.model import VFEModel
from vfe3.train import build_optimizer, train
from vfe3.run_artifacts import RunArtifacts
from vfe3.viz import figures as figs

DEVICE = torch.device(os.environ.get("VFE3_TEST_DEVICE", "cpu"))


def _natgrad_opt_with_grads(precond_mode: str) -> GaugeManifoldAdamW:
    r"""A GaugeManifoldAdamW with phi_embed.grad populated by one backward pass."""
    cfg = VFE3Config(vocab_size=32, embed_dim=4, n_heads=2, max_seq_len=8,
                     gauge_group="block_glk", m_phi_natural_grad=True,
                     phi_precond_mode=precond_mode, m_phi_lr=0.01, e_phi_lr=0.0)
    torch.manual_seed(0)
    model = VFEModel(cfg).to(DEVICE)
    opt = build_optimizer(model, cfg)
    tok = torch.randint(0, 32, (2, 8), device=DEVICE)
    tgt = torch.randint(0, 32, (2, 8), device=DEVICE)
    _, loss, _ = model(tok, tgt)
    loss.backward()
    return opt


def test_gauge_diag_pullback_collects_cos_and_cond():
    opt = _natgrad_opt_with_grads("pullback_per_block")
    assert isinstance(opt, GaugeManifoldAdamW)
    opt._collect_gauge_diag = True
    opt.step()
    gd = opt._gauge_diag
    assert "cos_nat_phi" in gd and -1.0 - 1e-6 <= gd["cos_nat_phi"] <= 1.0 + 1e-6
    assert gd["cos_nat_phi"] > 0.0                                  # G SPD -> grad . G^-1 grad > 0
    assert "pullback_cond_median" in gd and gd["pullback_cond_median"] >= 1.0 - 1e-6
    assert gd["pullback_cond_max"] >= gd["pullback_cond_median"] - 1e-6


def test_gauge_diag_declares_pullback_metric_as_full(monkeypatch):
    from vfe3 import numerics

    kinds = []

    def _condition_number(matrix, *, eps=1e-12, kind="auto"):
        kinds.append(kind)
        return torch.ones(matrix.shape[:-2], device=matrix.device, dtype=matrix.dtype)

    monkeypatch.setattr(numerics, "condition_number", _condition_number)
    opt = _natgrad_opt_with_grads("pullback_per_block")
    opt._collect_gauge_diag = True
    opt.step()

    assert kinds and set(kinds) == {"full"}


def test_gauge_diag_killing_cos_is_one_and_no_cond():
    opt = _natgrad_opt_with_grads("killing_per_block")
    opt._collect_gauge_diag = True
    opt.step()
    gd = opt._gauge_diag
    assert abs(gd["cos_nat_phi"] - 1.0) < 1e-4                      # conformal: direction-preserving
    assert "pullback_cond_median" not in gd                        # cond only on the pullback modes


def test_gauge_diag_silent_when_flag_off():
    opt = _natgrad_opt_with_grads("pullback_per_block")
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
    r"""End-to-end: a natural-grad pullback gauge run writes cos_nat_phi + pullback_cond_* into
    metrics.csv, present in EVERY row (the fixed-key-set / NaN-default contract; log_metrics locks
    fieldnames on row 0, so a key first appearing later would break the CSV)."""
    torch.manual_seed(0)
    cfg = VFE3Config(vocab_size=32, embed_dim=4, n_heads=2, max_seq_len=8, max_steps=4,
                     log_interval=2, eval_interval=2, batch_size=2, gauge_group="block_glk",
                     m_phi_natural_grad=True, phi_precond_mode="pullback_per_block",
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
        assert "cos_nat_phi" in r and "pullback_cond_median" in r and "wall_clock_s" in r


def test_wallclock_convergence_figure_renders():
    arms = [{"label": "pullback", "step": [10, 20, 30], "val_ppl": [40, 30, 25],
             "wall_clock_s": [1.0, 2.5, 4.0]},
            {"label": "adamw", "step": [10, 20, 30], "val_ppl": [42, 33, 28],
             "wall_clock_s": [0.8, 1.6, 2.5]}]
    fig = figs.plot_wallclock_convergence(arms)
    assert fig is not None
    plt.close(fig)
