r"""Tests for the 2026-06-22 B3/EXP-14 build-out (Fisher nat-grad vs raw-Euclidean E-step mean arm):

  * e_step_mu_precond config field defaults to 'fisher' and validates;
  * 'raw' actually changes the converged belief (the mean preconditioner is load-bearing), while
    'fisher' is the default;
  * the fisher_mu_precond sweep builds; the mu_precond figure + driver work.

Device-agnostic (CPU). Figures use the Agg backend.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
import torch

import ablation
from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel
from vfe3.viz import figures as figs

DEVICE = torch.device(os.environ.get("VFE3_TEST_DEVICE", "cpu"))


def test_e_step_mu_precond_default_and_validation():
    assert VFE3Config(vocab_size=32, embed_dim=4, n_heads=2, max_seq_len=8).e_step_mu_precond == "fisher"
    with pytest.raises(ValueError):
        VFE3Config(vocab_size=32, embed_dim=4, n_heads=2, max_seq_len=8, e_step_mu_precond="bogus")


def test_raw_mean_precond_changes_output():
    r"""nat_mu = Sigma*grad_mu (fisher) vs grad_mu (raw): once sigma departs from 1 over the E-step,
    the two mean steps differ, so the converged belief -> decode logits differ. Same init + same
    tokens; only e_step_mu_precond differs."""
    def _logits(precond):
        torch.manual_seed(0)
        # sigma_init != 1 so nat_mu = sigma*grad_mu departs from the raw grad_mu from the first step
        # (at sigma=1 the two coincide); n_e_steps>1 compounds the difference.
        cfg = VFE3Config(vocab_size=32, embed_dim=4, n_heads=2, max_seq_len=8, n_e_steps=4,
                         e_q_mu_lr=0.5, e_q_sigma_lr=0.1, sigma_init=0.25, e_step_mu_precond=precond)
        model = VFEModel(cfg).to(DEVICE)
        torch.manual_seed(1)
        return model(torch.randint(0, 32, (1, 8), device=DEVICE))
    lf, lr = _logits("fisher"), _logits("raw")
    assert lf.shape == lr.shape
    assert not torch.allclose(lf, lr)                               # the mean Fisher metric is load-bearing


def test_fisher_mu_precond_sweep_builds():
    ablation.validate_sweeps(["fisher_mu_precond"])
    runs = dict(ablation.make_run_overrides("fisher_mu_precond"))
    assert len(runs) == 6
    assert runs["raw_T3"]["e_step_mu_precond"] == "raw" and runs["raw_T3"]["n_e_steps"] == 3
    assert all(ov["e_phi_lr"] == 0.0 for ov in runs.values())
    for ov in runs.values():
        cfg_dict = ablation._cell_cfg_dict({**ov, "vocab_size": 48, "max_seq_len": 16}, seed=0, max_steps=1)
        assert VFEModel(VFE3Config(**cfg_dict)) is not None


def test_mu_precond_figure_renders():
    cells = [{"precond": p, "n_e_steps": t, "ppl": 30.0 + (2.0 if p == "raw" else 0.0) + 0.1 * t}
             for p in ("fisher", "raw") for t in (1, 3, 5)]
    fig = figs.plot_mu_precond(cells)
    assert fig is not None
    plt.close(fig)


def test_plot_mu_precond_driver(tmp_path):
    sweep = tmp_path / "fisher_mu_precond"; figdir = tmp_path / "figures"
    for p in ("fisher", "raw"):
        for t in (1, 3):
            d = sweep / f"{p}_T{t}"; d.mkdir(parents=True)
            (d / "ablation_result.json").write_text(json.dumps(
                {"label": f"{p}_T{t}", "primary_val_ppl": 30.0 + (2.0 if p == "raw" else 0.0),
                 "final_val_ppl": 30.0 + (2.0 if p == "raw" else 0.0),
                 "status": "success", "error_kind": None}))
    ablation._plot_mu_precond(sweep, figdir)
    assert (figdir / "fisher_mu_precond_mu_precond.png").exists()
