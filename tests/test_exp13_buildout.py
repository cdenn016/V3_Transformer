r"""Tests for the 2026-06-22 H1/EXP-13 build-out (offset-only positional extrapolation):

  * the pos_phi='learned' table CLAMPS past max_seq_len instead of silently returning the full table
    (the shape-crash trap), so the absolute arm RUNS at N > train length;
  * model.forward runs at N > max_seq_len for every positional scheme (alibi/t5/learned/rope);
  * the pos_extrapolation sweep arms validate + build (t5_max_distance raised);
  * the CE-vs-N figure renders and its driver reads per-cell extrap_ce JSON.

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
from vfe3.model.positional_phi import positional_phi_coords
from vfe3.viz import figures as figs

DEVICE = torch.device(os.environ.get("VFE3_TEST_DEVICE", "cpu"))


def test_pos_phi_learned_clamps_beyond_table():
    T, n_gen = 8, 3
    table = torch.randn(T, n_gen, device=DEVICE)
    c = positional_phi_coords("learned", 5, n_gen, pos_phi_free=table, device=DEVICE)
    assert c.shape == (5, n_gen) and torch.allclose(c, table[:5])    # n <= T: exact slice (byte-identical)
    c2 = positional_phi_coords("learned", 12, n_gen, pos_phi_free=table, device=DEVICE)
    assert c2.shape == (12, n_gen)                                   # n > T: clamped to (n, n_gen), not crash
    assert torch.allclose(c2[:T], table)
    assert torch.allclose(c2[T:], table[T - 1].expand(12 - T, n_gen))  # boundary row repeats


@pytest.mark.parametrize("over", [
    {"beta_attention_prior": "causal_alibi"},
    {"beta_attention_prior": "t5_relative_bias", "t5_max_distance": 32},
    {"beta_attention_prior": "causal", "pos_phi": "learned"},
    {"beta_attention_prior": "causal", "pos_rotation": "rope"},
])
def test_forward_runs_beyond_max_seq_len(over):
    r"""Every positional scheme must run at N > max_seq_len (the learned arm crashed before the fix)."""
    torch.manual_seed(0)
    cfg = VFE3Config(vocab_size=32, embed_dim=4, n_heads=2, max_seq_len=8, **over)
    model = VFEModel(cfg).to(DEVICE)
    logits = model(torch.randint(0, 32, (1, 16), device=DEVICE))     # N=16 > max_seq_len=8
    assert tuple(logits.shape) == (1, 16, 32) and torch.isfinite(logits).all()


def test_pos_extrapolation_sweep_arms_build():
    ablation.validate_sweeps(["pos_extrapolation"])
    runs = dict(ablation.make_run_overrides("pos_extrapolation"))
    assert set(runs) == {"alibi", "t5", "learned", "rope"}
    assert runs["t5"]["t5_max_distance"] == 512                      # == 4 x train length (no bucket saturation)
    for lab, ov in runs.items():
        # all arms share ONE belief-gradient route + the trained length (no cross-arm route confound)
        assert ov["oracle_unroll_grad"] is True and ov["max_seq_len"] == 128
        cfg_dict = ablation._cell_cfg_dict({**ov, "vocab_size": 48, "max_seq_len": 16}, seed=0, max_steps=1)
        assert VFEModel(VFE3Config(**cfg_dict)) is not None


def test_pos_extrapolation_figure_renders():
    arms = {"alibi":   [{"n": 8, "ce": 3.0}, {"n": 16, "ce": 3.1}, {"n": 32, "ce": 3.2}],
            "learned": [{"n": 8, "ce": 3.0}, {"n": 16, "ce": 4.5}, {"n": 32, "ce": 6.0}]}
    fig = figs.plot_pos_extrapolation(arms, train_n=8)
    assert fig is not None
    plt.close(fig)


def test_plot_pos_extrapolation_driver(tmp_path):
    sweep = tmp_path / "pos_extrapolation"; figdir = tmp_path / "figures"
    for lab, curve in [("alibi", [{"n": 8, "ce": 3.0}, {"n": 16, "ce": 3.1}]),
                       ("learned", [{"n": 8, "ce": 3.0}, {"n": 16, "ce": 5.0}])]:
        d = sweep / lab; d.mkdir(parents=True)
        (d / "ablation_result.json").write_text(
            json.dumps({"label": lab, "primary_val_ppl": 20.0, "final_val_ppl": 20.0,
                        "status": "success", "error_kind": None, "extrap_ce": curve}))
    ablation._plot_pos_extrapolation(sweep, figdir)
    assert (figdir / "pos_extrapolation_extrapolation.png").exists()
