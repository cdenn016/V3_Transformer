r"""Tests for the muP width-stability scaling route (F1/EXP-6) added 2026-06-21.

route_grow_k_mup emits a matched fixed-LR vs muP pair per width, each with the per-cell kl_max=8*K
confound fix, and every cell builds a valid VFE3Config/VFEModel."""
import pytest
import torch

import scaling
from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel


def test_grow_k_mup_registered():
    assert "grow_K_mup" in scaling.ROUTES


def test_grow_k_mup_kl_max_and_lr_scaling():
    cells = scaling.route_grow_k_mup([20, 40, 80], n_heads=4, anchor_k=20)
    ov = {c["label"]: c["overrides"] for c in cells}

    # per-cell kl_max = 8*K on BOTH arms (the confound fix), every width present as a fixed/mup pair
    for c in cells:
        assert c["overrides"]["kl_max"] == 8 * c["overrides"]["embed_dim"]
    assert set(ov) == {"K20_fixed", "K20_mup", "K40_fixed", "K40_mup", "K80_fixed", "K80_mup"}

    base_eqmu = scaling._baseline_value("e_q_mu_lr")
    base_init = scaling._baseline_value("mu_init_std")

    # anchor K=20: muP factor is 1 -> mup arm LR equals baseline (coincides with fixed)
    assert ov["K20_mup"]["e_q_mu_lr"] == pytest.approx(base_eqmu)
    # K=80: LR ~ anchor/K = 0.25, init ~ sqrt(0.25) = 0.5
    assert ov["K80_mup"]["e_q_mu_lr"] == pytest.approx(base_eqmu * 0.25)
    assert ov["K80_mup"]["mu_init_std"] == pytest.approx(base_init * 0.5)
    # the fixed arm carries no LR/init override (stays at the baseline operating point)
    assert "e_q_mu_lr" not in ov["K80_fixed"] and "mu_init_std" not in ov["K80_fixed"]


def test_grow_k_mup_cells_build():
    for c in scaling.route_grow_k_mup([20, 40], n_heads=4, anchor_k=20):
        d = scaling._cell_cfg_dict({**c["overrides"], "vocab_size": 64}, 0, 1)
        assert VFEModel(VFE3Config(**d)) is not None
