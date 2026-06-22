r"""Tests for the 2026-06-22 EXP-4 build-out (canonical-F vs entropy-suppressed surrogate, C1):

  * ``extract.attention_entropy_cov_gap`` -- the per-token -tau^{-1} Cov_beta(E, dE) gradient gap
    (oracle gradient with the entropy term ON minus OFF, on the converged belief).
  * the `attention_entropy` sweep is the 2x2 entropy x kappa grid; `cov_gap` is a CSV column.
  * the PPL-gap and Cov-gap-vs-kappa figures render.

The -tau^{-1} Cov_beta IDENTITY itself is pinned synthetically by
tests/test_free_energy.py::test_gradient_gap_canonical_minus_surrogate_is_neg_cov_over_tau; this file
checks the extractor that lifts it onto a real converged belief, plus the sweep/figure wiring.

Device-agnostic (CPU). Figures use the Agg backend.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import ablation
from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel
from vfe3.viz import figures as figs
from vfe3.viz.extract import attention_entropy_cov_gap

DEVICE = torch.device(os.environ.get("VFE3_TEST_DEVICE", "cpu"))


def test_cov_gap_extractor_runs_and_is_positive():
    torch.manual_seed(0)
    cfg = VFE3Config(vocab_size=48, embed_dim=8, n_heads=2, max_seq_len=12, n_e_steps=1)
    model = VFEModel(cfg).to(DEVICE)
    out = attention_entropy_cov_gap(model, torch.randint(0, 48, (1, 12), device=DEVICE))
    per_tok = out["cov_gap_per_token"]
    assert per_tok.shape == (12,)
    assert torch.isfinite(per_tok).all() and bool((per_tok >= 0).all())
    assert torch.allclose(out["cov_gap"], per_tok.mean())
    assert float(out["cov_gap"]) > 0.0                              # the entropy term IS load-bearing


def test_cov_gap_folds_precision_bias_when_pwa_on():
    r"""Under precision_weighted_attention=True (the baseline), the extractor must fold the
    -log(b0+trSigma) key bias into beta's prior (as forward/belief_ce_bank do) -- here we exercise
    that path (rank-robust (N,K) sigma) and confirm it runs finite, not the random-init no-op default."""
    torch.manual_seed(0)
    cfg = VFE3Config(vocab_size=48, embed_dim=8, n_heads=2, max_seq_len=12, n_e_steps=2,
                     precision_weighted_attention=True, precision_attention_b0=2.0)
    model = VFEModel(cfg).to(DEVICE)
    out = attention_entropy_cov_gap(model, torch.randint(0, 48, (1, 12), device=DEVICE))
    assert torch.isfinite(out["cov_gap_per_token"]).all()
    assert float(out["cov_gap"]) >= 0.0


def test_attention_entropy_sweep_is_2x2_entropy_x_kappa():
    runs = dict(ablation.make_run_overrides("attention_entropy"))
    assert set(runs) == {"canon_k1.0", "surr_k1.0", "canon_k0.25", "surr_k0.25"}
    assert runs["canon_k1.0"]["include_attention_entropy"] is True
    assert runs["surr_k0.25"]["include_attention_entropy"] is False
    assert runs["canon_k0.25"]["kappa_beta"] == 0.25
    assert all(ov["oracle_unroll_grad"] is True for ov in runs.values())


def test_cov_gap_column_registered():
    assert "cov_gap" in ablation._CSV_COLUMNS


def test_entropy_ppl_gap_figure_renders():
    cells = [{"include_attention_entropy": True,  "kappa": 1.0,  "ppl": 30.0},
             {"include_attention_entropy": False, "kappa": 1.0,  "ppl": 31.5},
             {"include_attention_entropy": True,  "kappa": 0.25, "ppl": 29.0},
             {"include_attention_entropy": False, "kappa": 0.25, "ppl": 33.0}]
    fig = figs.plot_entropy_ppl_gap(cells)
    assert fig is not None
    plt.close(fig)


def test_cov_gap_vs_kappa_figure_renders():
    cells = [{"include_attention_entropy": True,  "kappa": 1.0,  "cov_gap": 0.8},
             {"include_attention_entropy": True,  "kappa": 0.25, "cov_gap": 0.3},
             {"include_attention_entropy": False, "kappa": 1.0,  "cov_gap": 0.7},
             {"include_attention_entropy": False, "kappa": 0.25, "cov_gap": 0.25}]
    fig = figs.plot_cov_gap_vs_kappa(cells)
    assert fig is not None
    plt.close(fig)
