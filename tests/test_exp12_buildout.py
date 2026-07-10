r"""Tests for the 2026-06-22 B2/EXP-12 build-out (Rényi α-attention sweep + non-PD saturation):

  * the renyi_order sweep spans α both sides of 1 and collects diagnostics;
  * _cell_diagnostics emits the kl_max energy-saturation fraction (the α>1 diagnostic);
  * the renyi_saturation figure renders and its driver reads per-cell JSON.

Device-agnostic (CPU). Figures use the Agg backend.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import ablation
from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel
from vfe3.viz import figures as figs

DEVICE = torch.device(os.environ.get("VFE3_TEST_DEVICE", "cpu"))


def test_renyi_sweep_spans_alpha_and_collects():
    runs = dict(ablation.make_run_overrides("renyi_order"))
    alphas = sorted(float(lab.split("=")[-1]) for lab in runs)
    assert min(alphas) < 1.0 < max(alphas)                          # mass-covering AND mode-seeking
    assert all(ov["oracle_unroll_grad"] is True for ov in runs.values())
    assert ablation.SWEEPS["renyi_order"].get("collect_diagnostics") is True
    assert "energy_klmax_frac" in ablation._CSV_COLUMNS


def test_cell_diagnostics_emits_energy_klmax_frac():
    torch.manual_seed(0)
    cfg = VFE3Config(vocab_size=48, embed_dim=8, n_heads=2, max_seq_len=12, n_e_steps=1)
    model = VFEModel(cfg).to(DEVICE)
    loader = [(torch.randint(0, 48, (2, 12), device=DEVICE),
               torch.randint(0, 48, (2, 12), device=DEVICE))]
    diag = ablation._cell_diagnostics(model, cfg, loader, DEVICE)
    assert "energy_klmax_frac" in diag and 0.0 <= diag["energy_klmax_frac"] <= 1.0


def test_renyi_saturation_figure_renders():
    cells = [{"alpha": a, "attn_entropy": 2.0 - 0.1 * a, "energy_klmax_frac": max(0.0, (a - 1.0) * 0.3)}
             for a in (0.5, 0.8, 1.0, 1.2, 1.5, 2.0)]
    fig = figs.plot_renyi_saturation(cells)
    assert fig is not None
    plt.close(fig)


def test_plot_renyi_saturation_driver(tmp_path):
    sweep = tmp_path / "renyi_order"; figdir = tmp_path / "figures"
    for a in (0.5, 1.0, 2.0):
        d = sweep / f"renyi_{a}"; d.mkdir(parents=True)
        (d / "ablation_result.json").write_text(json.dumps(
            {"label": f"renyi_order={a}", "primary_val_ppl": 20.0,
             "final_val_ppl": 20.0, "status": "success", "error_kind": None,
             "attn_entropy": 1.5, "energy_klmax_frac": 0.1 * a}))
    ablation._plot_renyi_saturation(sweep, figdir)
    assert (figdir / "renyi_order_renyi_saturation.png").exists()
