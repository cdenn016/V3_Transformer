r"""Tests for the 2026-06-28 reporting build-out (surfacing already-logged diagnostics):

  * the four new registered history-dashboard figures render from synthetic history, drop absent
    panels, and mask eval-cadence NaN gaps; the pooled PPL offset figure renders;
  * run_artifacts._save_figures emits the four dashboard PNGs from a synthetic history;
  * run_artifacts._pure_path_report reports the toggle/stress state and the on-pure-path flag;
  * train() surfaces the held-out gauge/SPD/Fisher geometry columns into metrics.csv;
  * ablation._seed_aggregate / _base_label group seeds into n/mean/SD/CV;
  * scaling_analysis._write_scaling_md renders the console-only fits as a markdown report.

Device-agnostic (CPU). Figures use the Agg backend.
"""
import csv as _csv
import logging
import os
import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import ablation
import scaling_analysis
from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel
from vfe3.run_artifacts import RunArtifacts, _pure_path_report, _save_figures, finalize_run
from vfe3.train import train
from vfe3.viz import figures as figs

DEVICE = torch.device(os.environ.get("VFE3_TEST_DEVICE", "cpu"))


def _hist(keys, n=30, nan_every=0):
    r"""Synthetic {step, key: [...]} history: positive decreasing series, optionally NaN-masked on
    non-eval rows (nan_every>0 keeps a finite value every nan_every steps, like an eval-cadence column)."""
    h = {"step": list(range(n))}
    base = np.linspace(1.0, 0.1, n) + 0.05
    for k in keys:
        col = base.copy()
        if nan_every:
            for i in range(n):
                if i % nan_every:
                    col[i] = float("nan")
        h[k] = col.tolist()
    return h


# --------------------------------------------------------------------------- figure renders

def test_geometry_health_renders():
    h = _hist(("holonomy_wilson", "cocycle_residual", "gauge_invariant_spread", "phi_norm_mean",
               "phi_norm_std", "belief_cond_p95", "belief_cond_max", "eff_rank_p5", "eff_rank_median",
               "eff_rank_p95", "fisher_trace_mean", "guard_sigma_floor_frac", "nonfinite_frac",
               "renyi_band_frac", "attn_entropy_min", "attn_entropy_collapsed_heads"))
    fig = figs.plot_geometry_health(h); assert fig is not None; plt.close(fig)


def test_estep_quality_renders():
    h = _hist(("estep_f_drop", "estep_f_nondecreasing_frac", "estep_r_mu_last", "estep_r_sigma_last",
               "estep_r_phi_last"))
    fig = figs.plot_estep_quality(h); assert fig is not None; plt.close(fig)


def test_validation_sanity_renders_with_eval_cadence_gaps():
    # eval-cadence columns are NaN on non-eval rows; the dashboard must mask per series and still render.
    h = _hist(("generalization_gap", "pos_loss_ratio", "val_future_leakage", "val_row_sum_error",
               "val_pos_content_r2", "val_head_redundancy_js", "val_holonomy_wilson",
               "val_cocycle_residual", "val_belief_cond_p95", "val_fisher_trace_mean",
               "val_phi_norm_mean"), nan_every=5)
    fig = figs.plot_validation_sanity(h); assert fig is not None; plt.close(fig)


def test_optimizer_geometry_renders_with_ratio_synthesis():
    h = _hist(("cos_nat_phi", "pullback_cond_median", "pullback_cond_max",
               "weight_norm_mu", "weight_norm_sigma", "weight_norm_phi",
               "grad_norm_mu", "grad_norm_sigma", "grad_norm_phi"))
    fig = figs.plot_optimizer_geometry(h); assert fig is not None; plt.close(fig)

def test_kappa_history_renders_with_variance_band():
    h = {"step": [1, 2, 3],
         "kappa_beta_mean": [1.0, 1.2, 1.4],
         "kappa_beta_var": [0.0, 0.01, 0.04]}
    fig = figs.plot_kappa_history(h, channel="beta"); assert fig is not None; plt.close(fig)


def test_kappa_block_trajectory_renders_per_block_kappa_and_tau():
    # Per-block 2x2 grid (kappa/tau x beta/gamma), one line per irrep block.
    h = {"step": [1, 2, 3],
         "kappa_beta_b0": [1.0, 1.1, 1.2], "kappa_beta_b1": [1.0, 0.9, 0.8],
         "tau_beta_b0":   [1.4, 1.5, 1.7], "tau_beta_b1":   [1.4, 1.3, 1.1],
         "kappa_gamma_b0": [1.0, 1.05, 1.1], "kappa_gamma_b1": [1.0, 0.95, 0.9],
         "tau_gamma_b0":   [1.4, 1.5, 1.6], "tau_gamma_b1":   [1.4, 1.35, 1.25]}
    fig = figs.plot_kappa_block_trajectory(h); assert fig is not None; plt.close(fig)


def test_ppl_offset_renders():
    pts = [{"embed_dim": k, "ppl_mean": 50.0 / k + 12.0, "ppl_sem": 0.3, "n_seeds": 3}
           for k in (8, 16, 32, 64)]
    fig = figs.plot_ppl_offset(pts); assert fig is not None; plt.close(fig)


def test_new_figures_registered():
    for name, fn in (("geometry_health", figs.plot_geometry_health),
                     ("estep_quality", figs.plot_estep_quality),
                     ("validation_sanity", figs.plot_validation_sanity),
                     ("optimizer_geometry", figs.plot_optimizer_geometry),
                     ("ppl_offset", figs.plot_ppl_offset)):
        assert figs.get_figure(name) is fn


def test_dashboard_drops_absent_and_empty():
    # only one column present -> one panel, still renders; no columns -> the no-data fallback renders.
    fig = figs.plot_geometry_health(_hist(("holonomy_wilson",))); assert fig is not None; plt.close(fig)
    fig = figs.plot_geometry_health({"step": [0, 1, 2]}); assert fig is not None; plt.close(fig)


# --------------------------------------------------------------------------- run_artifacts wiring

def test_save_figures_emits_dashboards(tmp_path):
    torch.manual_seed(0)
    cfg = VFE3Config(vocab_size=16, embed_dim=8, n_heads=2, max_seq_len=8, max_steps=2, batch_size=2)
    model = VFEModel(cfg).to(DEVICE)
    art = RunArtifacts(str(tmp_path), cfg, model, dataset="synthetic", device=str(DEVICE))
    keys = ("holonomy_wilson", "cocycle_residual", "fisher_trace_mean", "belief_cond_p95",
            "estep_f_drop", "estep_r_mu_last", "generalization_gap", "val_holonomy_wilson",
            "cos_nat_phi", "weight_norm_mu", "grad_norm_mu")
    art.history = [{"step": s, **{k: 1.0 / (s + 1) + 0.1 for k in keys}} for s in range(12)]
    _save_figures(art, [3.0, 2.5], logging.getLogger("test"))
    for png in ("geometry_health.png", "estep_quality.png", "validation_sanity.png",
                "optimizer_geometry.png"):
        assert (tmp_path / png).exists(), f"{png} not emitted"



def test_save_figures_emits_kappa_histories(tmp_path):
    torch.manual_seed(0)
    cfg = VFE3Config(vocab_size=16, embed_dim=8, n_heads=2, max_seq_len=8, max_steps=2, batch_size=2,
                     learnable_kappa_beta=True, learnable_kappa_gamma=True, lambda_gamma=0.1)
    model = VFEModel(cfg).to(DEVICE)
    art = RunArtifacts(str(tmp_path), cfg, model, dataset="synthetic", device=str(DEVICE))
    art.history = [
        {"step": s,
         "kappa_beta_mean": 1.0 + 0.05 * s, "kappa_beta_var": 0.01 * s,
         "kappa_gamma_mean": 1.2 + 0.04 * s, "kappa_gamma_var": 0.02 * s}
        for s in range(1, 5)
    ]
    _save_figures(art, None, logging.getLogger("test"))
    for png in ("kappa_beta_history.png", "kappa_gamma_history.png"):
        assert (tmp_path / png).exists(), f"{png} not emitted"


def test_save_figures_emits_kappa_block_trajectory(tmp_path):
    torch.manual_seed(0)
    cfg = VFE3Config(vocab_size=16, embed_dim=8, n_heads=2, max_seq_len=8, max_steps=2, batch_size=2,
                     learnable_kappa_beta=True, learnable_kappa_gamma=True, lambda_gamma=0.1)
    model = VFEModel(cfg).to(DEVICE)
    art = RunArtifacts(str(tmp_path), cfg, model, dataset="synthetic", device=str(DEVICE))
    art.history = [
        {"step": s,
         "kappa_beta_b0": 1.0 + 0.05 * s, "kappa_beta_b1": 1.0 - 0.02 * s,
         "tau_beta_b0":   1.4 + 0.05 * s, "tau_beta_b1":   1.4 - 0.02 * s,
         "kappa_gamma_b0": 1.2 + 0.04 * s, "kappa_gamma_b1": 1.2 - 0.01 * s,
         "tau_gamma_b0":   1.6 + 0.04 * s, "tau_gamma_b1":   1.6 - 0.01 * s}
        for s in range(1, 5)
    ]
    _save_figures(art, None, logging.getLogger("test"))
    assert (tmp_path / "kappa_block_trajectory.png").exists()

# --------------------------------------------------------------------------- pure-path certificate

def _pure_ns(**over):
    r"""SimpleNamespace with every attribute _pure_path_report reads, at the pure values."""
    base = dict(
        include_attention_entropy=True, transport_mode="flat", lambda_alpha_mode="constant",
        use_prior_bank=True, use_head_mixer=False,
        lambda_beta=1.0, precision_weighted_attention=False,
        gauge_transport="on", pos_rotation="none", rope_full_gauge=False, rope_on_value=True,
        lambda_gamma=0.0, s_e_step=False)
    base.update(over)
    return types.SimpleNamespace(**base)


def test_pure_path_report_structure_and_flags():
    history = [{"step": 0, "nonfinite_frac": 0.0, "cocycle_residual": 1e-7, "holonomy_wilson": 2e-7,
                "guard_sigma_floor_frac": 0.0}]
    pure = _pure_ns()
    rep = _pure_path_report(pure, history)
    assert rep["on_pure_path"] is True
    assert rep["on_gauge_pure_path"] is True
    assert set(rep) == {"on_pure_path", "pure_flags", "config_toggles", "converged_stress",
                        "gauge_flags", "on_gauge_pure_path"}
    assert rep["converged_stress"]["cocycle_residual"] == 1e-7
    # flipping any defining toggle drops the run off the pure path
    impure = types.SimpleNamespace(**{**pure.__dict__, "transport_mode": "regime_ii"})
    rep2 = _pure_path_report(impure, history)
    assert rep2["on_pure_path"] is False and rep2["pure_flags"]["flat_transport"] is False


def test_gauge_purity_axis_is_independent_of_fe_axis():
    # F8 (audit 2026-07-01): the gauge / model-channel axis is a SECOND, independent purity axis --
    # flipping a gauge toggle must drop on_gauge_pure_path while on_pure_path stays True.
    history = []
    rep = _pure_path_report(_pure_ns(gauge_transport="off"), history)
    assert rep["on_pure_path"] is True and rep["on_gauge_pure_path"] is False
    assert rep["gauge_flags"]["learned_gauge_transport"] is False
    rep2 = _pure_path_report(_pure_ns(s_e_step=True), history)
    assert rep2["on_pure_path"] is True and rep2["on_gauge_pure_path"] is False
    assert rep2["gauge_flags"]["no_model_channel_coupling"] is False
    # the six gauge/model-channel settings are surfaced in config_toggles for transparency
    for key in ("gauge_transport", "pos_rotation", "rope_full_gauge", "rope_on_value",
                "lambda_gamma", "s_e_step"):
        assert key in rep["config_toggles"]


# --------------------------------------------------------------------------- figure memory guard

def _finalize_ns(**over):
    r"""SimpleNamespace cfg for finalize_run: carries every attribute the numeric path touches, so
    the F9 tests do not depend on the concurrently-added ``force_large_figures`` config field."""
    base = dict(
        # figure memory-guard inputs: 8*V*N*B/1e9 ~ 13.2 GB fp32 logits+probs peak > the 8 GB guard
        vocab_size=50257, max_seq_len=1024, batch_size=32,
        generate_figures=True, force_large_figures=False,
        # numeric-path attrs (summary / provenance / cost model / pure-path report)
        seed=0, max_steps=2, use_prior_bank=True, use_head_mixer=False,
        include_attention_entropy=True, transport_mode="flat", lambda_alpha_mode="constant",
        lambda_beta=1.0, precision_weighted_attention=False,
        gauge_transport="on", pos_rotation="none", rope_full_gauge=False, rope_on_value=True,
        lambda_gamma=0.0, s_e_step=False,
        embed_dim=8, n_heads=2, n_layers=1, n_e_steps=1, diagonal_covariance=True,
        gauge_group="block_glk", lambda_h=0.0, prior_source="prior_bank", amp_dtype="fp32")
    base.update(over)
    return types.SimpleNamespace(**base)


def test_finalize_run_skips_figures_over_memory_guard(tmp_path, monkeypatch, caplog):
    # F9 (audit 2026-07-01): the figure extractors materialize dense (B, N, V) logits+probs, so
    # finalize_run must SKIP the figure pass (with a warning) when the estimated full-vocab peak
    # exceeds the 8 GB guard and force_large_figures is off.
    torch.manual_seed(0)
    real_cfg = VFE3Config(vocab_size=16, embed_dim=8, n_heads=2, max_seq_len=8, max_steps=2,
                          batch_size=2)
    model = VFEModel(real_cfg).to(DEVICE)
    art = RunArtifacts(str(tmp_path), real_cfg, model, dataset="synthetic", device=str(DEVICE))
    calls = []
    monkeypatch.setattr("vfe3.viz.report.generate_figures", lambda *a, **k: calls.append(a))
    with caplog.at_level(logging.WARNING):
        finalize_run(model, art, _finalize_ns(), test_loader=None)
    assert calls == []                                          # figure pass skipped by the guard
    assert any("skipping publication figures" in r.getMessage() for r in caplog.records)


def test_finalize_run_force_large_figures_overrides_guard(tmp_path, monkeypatch):
    # F9 counterpart: force_large_figures=True is the explicit large-run opt-in -- the figure pass
    # runs despite the over-budget estimate.
    torch.manual_seed(0)
    real_cfg = VFE3Config(vocab_size=16, embed_dim=8, n_heads=2, max_seq_len=8, max_steps=2,
                          batch_size=2)
    model = VFEModel(real_cfg).to(DEVICE)
    art = RunArtifacts(str(tmp_path), real_cfg, model, dataset="synthetic", device=str(DEVICE))
    calls = []
    monkeypatch.setattr("vfe3.viz.report.generate_figures", lambda *a, **k: calls.append(a))
    finalize_run(model, art, _finalize_ns(force_large_figures=True), test_loader=None)
    assert calls                                                # figure pass attempted despite the estimate


# --------------------------------------------------------------------------- held-out geometry columns

def test_train_logs_held_out_geometry(tmp_path):
    torch.manual_seed(0)
    cfg = VFE3Config(vocab_size=32, embed_dim=8, n_heads=2, max_seq_len=8, max_steps=4,
                     log_interval=2, eval_interval=2, batch_size=2)
    model = VFEModel(cfg).to(DEVICE)
    batches = [(torch.randint(0, 32, (2, 8), device=DEVICE),
                torch.randint(0, 32, (2, 8), device=DEVICE)) for _ in range(4)]
    art = RunArtifacts(str(tmp_path), cfg, model, dataset="synthetic", device=str(DEVICE))
    train(model, batches, cfg, n_steps=4, log_interval=2, eval_interval=2,
          val_loader=batches, artifacts=art, device=DEVICE, generate_samples=False)
    with open(tmp_path / "metrics.csv", newline="", encoding="utf-8") as fh:
        rows = list(_csv.DictReader(fh))
    assert rows and "val_holonomy_wilson" in rows[0]      # rectangular: column defined from row 0
    finite = [r for r in rows if r["val_holonomy_wilson"] not in ("", "nan", None)]
    assert finite, "an eval row must carry a finite held-out Wilson holonomy"


# --------------------------------------------------------------------------- ablation seed aggregation

def test_base_label_strips_seed_suffix():
    assert ablation._base_label("a2_on__s0") == "a2_on"
    assert ablation._base_label("a2_on__s12") == "a2_on"
    assert ablation._base_label("kappa=2.0") == "kappa=2.0"
    assert ablation._base_label("plain") == "plain"


def test_seed_aggregate_groups_seeds():
    rows = [{"label": "a__s0", "primary_val_ppl": "30.0"},
            {"label": "a__s1", "primary_val_ppl": "32.0"},
            {"label": "b", "primary_val_ppl": "40.0"},
            {"label": "c__s0", "primary_val_ppl": "inf"}]      # non-finite cell dropped
    agg = ablation._seed_aggregate(rows)
    by = {a["label"]: a for a in agg}
    assert by["a"]["n"] == 2 and abs(by["a"]["mean"] - 31.0) < 1e-9
    assert abs(by["a"]["sd"] - 2.0 ** 0.5) < 1e-9 and abs(by["a"]["cv"] - (2.0 ** 0.5) / 31.0) < 1e-9
    assert by["b"]["n"] == 1 and by["b"]["sd"] == 0.0
    assert "c" not in by                                       # no finite seed -> dropped
    assert [a["label"] for a in agg] == ["a", "b"]             # sorted by mean PPL


# --------------------------------------------------------------------------- scaling summary report

def test_write_scaling_md(tmp_path):
    summary = {
        "input_dir": str(tmp_path), "n_runs": 6, "with_offset": True,
        "n_param_points": 4, "n_inference_points": 2,
        "pooled_fit": {"form": "E + A N^-alpha", "alpha": 0.12, "alpha_ci": [0.08, 0.16],
                       "A": 3.4, "E": 2.1, "r2": 0.98, "n_points": 4},
        "per_route": {"grow_K": {"alpha": 0.11, "r2": 0.97, "n_sizes": 4}},
        "frontier_collapse": {"testable": True, "collapses": True, "df1": 2, "df2": 8,
                              "F": 0.5, "p_value": 0.62},
        "estep_structural": {"n_arms": 3, "pearson_ne_final_f": -0.95, "pearson_final_f_test_ce": 0.1},
    }
    out = tmp_path / "SCALING_ANALYSIS.md"
    scaling_analysis._write_scaling_md(out, summary)
    text = out.read_text(encoding="utf-8")
    for section in ("Pooled L(N) power law", "Per-route exponents", "Frontier-collapse F-test",
                    "E-step structural-EM check"):
        assert section in text
