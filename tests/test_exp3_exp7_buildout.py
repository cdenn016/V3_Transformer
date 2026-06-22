r"""Tests for the 2026-06-22 EXP-3 / EXP-7 build-out (finishing the partial experiments):

  EXP-3 (Sigma_q calibration, B1):
    * ``extract.belief_ce_bank`` -- the Sigma_q<->CE JOIN (aligned per-token tr(Sigma_q), CE,
      token id; valid-target masked).
    * the three calibration figures render: reliability diagram, Sigma-stratified error, Sigma-CE
      scatter.

  EXP-7 (rank collapse, F2):
    * the ``rho_handoff`` ablation sweep validates + builds every arm.
    * ``_cell_diagnostics`` surfaces the final-layer ``rank_resid`` (CSV) and the per-layer
      ``rank_resid_by_layer`` curve (cell JSON) the depth-overlay figure consumes.
    * ``plot_rank_residual_by_depth`` renders one r(X)-by-depth line per arm.

Device-agnostic (CPU by default; honors VFE3_TEST_DEVICE). Figures use the Agg backend.
"""
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
from vfe3.viz.extract import belief_ce_bank

DEVICE = torch.device(os.environ.get("VFE3_TEST_DEVICE", "cpu"))


def _tiny_model(**over) -> VFEModel:
    cfg = VFE3Config(vocab_size=48, embed_dim=8, n_heads=2, max_seq_len=12, n_e_steps=1, **over)
    return VFEModel(cfg).to(DEVICE)


def _loader(n_batches: int = 2, b: int = 2, n: int = 12, vocab: int = 48):
    r"""A minimal (tokens, targets) loader (list of tuples); one target per batch is masked -100."""
    torch.manual_seed(0)
    out = []
    for _ in range(n_batches):
        tok = torch.randint(0, vocab, (b, n), device=DEVICE)
        tgt = torch.randint(0, vocab, (b, n), device=DEVICE)
        tgt[0, -1] = -100                                            # an ignored position
        out.append((tok, tgt))
    return out


# --------------------------------------------------------------------------- EXP-3 join

def test_belief_ce_bank_aligns_sigma_and_ce():
    model = _tiny_model()
    bank = belief_ce_bank(model, _loader(), device=DEVICE)
    tr, ce, tid = bank["tr_sigma"], bank["ce"], bank["token_ids"]
    assert tr.shape == ce.shape == tid.shape                        # one row per valid token
    assert tr.numel() == 2 * 2 * 12 - 2                             # 2 batches x 2 seqs x 12, 1 masked/batch
    assert torch.isfinite(tr).all() and torch.isfinite(ce).all()
    assert (tr > 0).all()                                           # tr(Sigma_q) is a sum of variances
    assert (ce >= 0).all() and (tid != -100).all()


def test_belief_ce_bank_respects_max_batches():
    model = _tiny_model()
    bank = belief_ce_bank(model, _loader(n_batches=5), device=DEVICE, max_batches=1)
    assert bank["tr_sigma"].numel() == 2 * 12 - 1                   # only the first batch consumed


def test_belief_ce_bank_matches_forward_belief_under_s_e_step():
    r"""Faithfulness under the live s_e_step=True config: belief_ce_bank must trace the SAME converged
    covariance the decode used, i.e. apply the s-refine anchor (and precision fold) forward applies.
    Cross-checked against converged_state, an INDEPENDENT faithful replay that also runs _refine_s --
    they agree only because belief_ce_bank now mirrors forward's belief path (drop the _refine_s line
    and this diverges)."""
    from vfe3.metrics import sigma_trace
    from vfe3.viz.extract import converged_state
    cfg = VFE3Config(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=1,
                     n_e_steps=1, e_q_mu_lr=0.5, e_phi_lr=0.0, mass_phi=0.0,
                     mstep_self_coupling_weight=0.0, use_prior_bank=True,
                     lambda_h=1.0, lambda_gamma=1.0, prior_source="model_channel",
                     s_e_step=True, e_s_mu_lr=0.5, learnable_r=True, seed=0)
    torch.manual_seed(0)
    model = VFEModel(cfg).to(DEVICE)
    tok = torch.randint(0, 20, (1, 5), device=DEVICE)
    tgt = torch.randint(0, 20, (1, 5), device=DEVICE)                # all valid (no -100)
    bank = belief_ce_bank(model, [(tok, tgt)], device=DEVICE)
    cs_tr = sigma_trace(converged_state(model, tok)["sigma"], diagonal=cfg.diagonal_covariance)
    assert torch.allclose(bank["tr_sigma"], cs_tr.reshape(-1), atol=1e-5)


def test_belief_ce_bank_feeds_spearman_and_cv():
    r"""The bank's aligned columns are exactly the inputs the EXP-3 headline statistics take."""
    from vfe3.metrics import cv, spearman_rho
    model = _tiny_model()
    bank = belief_ce_bank(model, _loader(), device=DEVICE)
    rho = spearman_rho(bank["tr_sigma"], bank["ce"])
    spread = cv(bank["tr_sigma"])
    assert -1.0 <= rho <= 1.0
    assert spread >= 0.0


# --------------------------------------------------------------------------- EXP-3 figures

def test_reliability_diagram_renders():
    rel = [{"conf": 0.1 * (i + 0.5), "acc": 0.1 * i, "frac": 0.1} for i in range(10)]
    fig = figs.plot_reliability_diagram(rel)
    assert fig is not None
    plt.close(fig)


def test_sigma_figures_render_from_bank():
    torch.manual_seed(1)
    tr = torch.rand(300) + 0.1
    bank = {"tr_sigma": tr, "ce": 0.5 * tr + torch.rand(300)}        # CE tilts with uncertainty
    f1 = figs.plot_sigma_stratified_error(bank)
    f2 = figs.plot_sigma_ce_scatter(bank)
    assert f1 is not None and f2 is not None
    plt.close(f1)
    plt.close(f2)


# --------------------------------------------------------------------------- EXP-7 sweep + curve

def test_rho_handoff_arms_validate_and_build():
    ablation.validate_sweeps(["rho_handoff"])
    runs = dict(ablation.make_run_overrides("rho_handoff"))
    assert set(runs) == {"anchor_rho1", "anchor_rho0", "noanchor",
                         "anchor_rho1_ephi", "noanchor_ephi"}
    for _label, over in runs.items():
        assert over["lambda_alpha_mode"] == "constant"             # baseline state-dependent alpha pinned off
        assert over["n_layers"] == 4                               # a multi-point depth curve
        cfg_dict = ablation._cell_cfg_dict(
            {**over, "vocab_size": 48, "max_seq_len": 12}, seed=0, max_steps=1)
        assert VFEModel(VFE3Config(**cfg_dict)) is not None


def test_rho_handoff_ephi_pair_has_live_estep_gauge_lr():
    r"""The *_ephi arms must keep e_phi_lr > 0 after config coercion (transport per-layer-independent);
    if gauge_transport coerced it to 0 the pair would silently duplicate the e_phi_lr=0 arms."""
    runs = dict(ablation.make_run_overrides("rho_handoff"))
    over = {**runs["anchor_rho1_ephi"], "vocab_size": 48, "max_seq_len": 12}
    cfg = VFE3Config(**ablation._cell_cfg_dict(over, seed=0, max_steps=1))
    assert cfg.e_phi_lr > 0.0


def test_rank_resid_column_registered():
    assert "rank_resid" in ablation._CSV_COLUMNS


def test_cell_diagnostics_emits_rank_curve():
    r"""S2/EXP-7: a deep collect_diagnostics replay yields the final-layer rank_resid scalar AND the
    per-layer rank_resid_by_layer curve (length n_layers) the depth-overlay figure reads."""
    model = _tiny_model(n_layers=3)
    diag = ablation._cell_diagnostics(model, model.cfg, _loader(), DEVICE)
    assert "rank_resid" in diag and 0.0 <= diag["rank_resid"] <= 1.0 + 1e-5
    curve = diag["rank_resid_by_layer"]
    assert isinstance(curve, list) and len(curve) == 3
    assert all(0.0 <= float(x) <= 1.0 + 1e-5 for x in curve)


def test_rank_residual_by_depth_figure_renders():
    arms = {"anchor_rho1": [0.50, 0.42, 0.36, 0.31],
            "noanchor":    [0.50, 0.30, 0.16, 0.07]}
    fig = figs.plot_rank_residual_by_depth(arms)
    assert fig is not None
    plt.close(fig)
