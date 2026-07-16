r"""Tests for the 2026-06-28 reporting build-out (surfacing already-logged diagnostics):

  * the four new registered history-dashboard figures render from synthetic history, drop absent
    panels, and mask eval-cadence NaN gaps; the pooled PPL offset figure renders;
  * run_artifacts._save_figures emits the four dashboard PNGs from a synthetic history;
  * run_artifacts._pure_path_report reports the toggle/stress state and the on-pure-path flag;
  * train() surfaces the held-out gauge/SPD/Fisher geometry columns into metrics.csv;
  * ablation._seed_aggregate / _base_label group seeds into n/mean/SD/CV;
  * scaling_analysis._write_scaling_md renders the console-only fits as a markdown report;
  * end-to-end: scaling_analysis.analyze() and ablation.main() dispatch capacity_scaling /
    pareto_frontier / ablation_forest / lr_grid_heatmap through the figure registry by NAME
    (a monkeypatched registration is what actually produces each PNG) while every legacy output
    keeps being written (PB-07 registry-completion Task 5).

Device-agnostic (CPU). Figures use the Agg backend.
"""
import csv as _csv
import hashlib
import json
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
from vfe3.run_artifacts import (
    RunArtifacts,
    _cost_model_fields,
    _pure_path_report,
    _save_figures,
    finalize_run,
)
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
    cfg = VFE3Config(vocab_size=16, embed_dim=4, n_heads=2, max_seq_len=8, max_steps=2, batch_size=2)
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
    cfg = VFE3Config(vocab_size=16, embed_dim=4, n_heads=2, max_seq_len=8, max_steps=2, batch_size=2,
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


# ------------------------------------------------------ M4 (audit 2026-07-06): multiseed ablation figures

def _capture_ablation_figure(monkeypatch, fn_name):
    r"""Patch vfe3.viz.figures.<fn_name> (imported call-time inside the ablation collectors) to capture
    the cells it is called with instead of rendering."""
    cap = {}
    def fake(cells, path=None):
        cap["cells"] = list(cells)
        return plt.figure()
    monkeypatch.setattr(f"vfe3.viz.figures.{fn_name}", fake)
    return cap


def test_gauge_transport_figure_aggregates_seeds(monkeypatch, tmp_path):
    # M4(a): on_L1__s6 must parse (not split into 4 parts and skip); the seed cells aggregate to one
    # bar per (mode, depth) at the across-seed mean PPL.
    rows = [
        {"label": "on_L1__s6",  "primary_val_ppl": "10.0", "omega_identity_dev": "0.0"},
        {"label": "on_L1__s7",  "primary_val_ppl": "12.0", "omega_identity_dev": "0.0"},
        {"label": "off_L1__s6", "primary_val_ppl": "20.0", "omega_identity_dev": "1e-7"},
        {"label": "off_L1__s7", "primary_val_ppl": "22.0", "omega_identity_dev": "3e-7"},
    ]
    monkeypatch.setattr(ablation, "_collect_sweep_results", lambda d: rows)
    cap = _capture_ablation_figure(monkeypatch, "plot_gauge_transport_bars")
    ablation._plot_gauge_transport(tmp_path, tmp_path)
    assert "cells" in cap, "figure skipped: seeded on_L1__s* labels not parsed"
    assert len(cap["cells"]) == 2
    by = {(c["mode"], c["depth"]): c for c in cap["cells"]}
    assert set(by) == {("on", "L1"), ("off", "L1")}
    assert abs(by[("on", "L1")]["ppl"] - 11.0) < 1e-9      # (10 + 12) / 2
    assert abs(by[("off", "L1")]["ppl"] - 21.0) < 1e-9     # (20 + 22) / 2


def test_mu_precond_figure_aggregates_seeds(monkeypatch, tmp_path):
    # M4(b): fisher_T2__s6 must parse (int('2__s6') would ValueError and skip); aggregate to one point
    # per (precond, n_e_steps).
    rows = [
        {"label": "fisher_T2__s6", "primary_val_ppl": "30.0"},
        {"label": "fisher_T2__s7", "primary_val_ppl": "32.0"},
        {"label": "raw_T2__s6",    "primary_val_ppl": "40.0"},
        {"label": "raw_T2__s7",    "primary_val_ppl": "42.0"},
    ]
    monkeypatch.setattr(ablation, "_collect_sweep_results", lambda d: rows)
    cap = _capture_ablation_figure(monkeypatch, "plot_mu_precond")
    ablation._plot_mu_precond(tmp_path, tmp_path)
    assert "cells" in cap, "figure skipped: int('2__s6') ValueError on seeded labels"
    assert len(cap["cells"]) == 2
    by = {(c["precond"], c["n_e_steps"]): c for c in cap["cells"]}
    assert set(by) == {("fisher", 2), ("raw", 2)}
    assert abs(by[("fisher", 2)]["ppl"] - 31.0) < 1e-9
    assert abs(by[("raw", 2)]["ppl"] - 41.0) < 1e-9


def test_attention_entropy_figure_aggregates_seeds(monkeypatch, tmp_path):
    # M4(c): the seed cells sharing (include_attention_entropy, kappa) must collapse to one cell at the
    # mean PPL, not be passed as N per-seed cells (the figure would then plot one arbitrary seed).
    rows = [
        {"label": f"ent{e}__s{s}", "primary_val_ppl": str(p), "cov_gap": "inf",
         "overrides": {"include_attention_entropy": e, "kappa_beta": 1.0}}
        for (e, base) in ((True, 10.0), (False, 14.0)) for (s, p) in ((6, base), (7, base + 2.0))
    ]
    monkeypatch.setattr(ablation, "_collect_sweep_results", lambda d: rows)
    cap = _capture_ablation_figure(monkeypatch, "plot_entropy_ppl_gap")
    ablation._plot_attention_entropy(tmp_path, tmp_path)
    assert "cells" in cap
    assert len(cap["cells"]) == 2                           # one per include_attention_entropy at kappa=1
    by = {bool(c["include_attention_entropy"]): c for c in cap["cells"]}
    assert set(by) == {True, False}
    assert abs(by[True]["ppl"] - 11.0) < 1e-9              # (10 + 12) / 2
    assert abs(by[False]["ppl"] - 15.0) < 1e-9            # (14 + 16) / 2


def test_offset_power_law_honors_weights():
    # m29: the with_offset (Chinchilla) fit must honor the WLS weights callers pass, not silently
    # run unweighted OLS. Down-weighting a corrupted point must move the fitted exponent.
    import pytest
    pytest.importorskip("scipy")   # the offset branch needs scipy.optimize.curve_fit (else falls back to log-log)
    from vfe3.viz.figures import _fit_power_law
    N = np.array([1e2, 1e3, 1e4, 1e5])
    L = (2.0 + 5.0 * N ** -0.3).copy()
    L[0] += 0.5                                            # corrupt the low-N point
    fit_uniform = _fit_power_law(N, L, weights=np.ones_like(N), with_offset=True)
    fit_downwt  = _fit_power_law(N, L, weights=np.array([1e-3, 1.0, 1.0, 1.0]), with_offset=True)
    assert fit_uniform["form"] == "offset_power_law" and fit_downwt["form"] == "offset_power_law"
    assert abs(fit_uniform["alpha"] - fit_downwt["alpha"]) > 1e-4   # weights must change the fit


def test_save_figures_emits_kappa_block_trajectory(tmp_path):
    torch.manual_seed(0)
    cfg = VFE3Config(vocab_size=16, embed_dim=4, n_heads=2, max_seq_len=8, max_steps=2, batch_size=2,
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

def _cost_cfg(**over):
    base = dict(
        vocab_size=11, embed_dim=6, n_heads=2, max_seq_len=7, batch_size=2,
        n_layers=2, n_e_steps=3, max_steps=5,
        lambda_h=0.0, lambda_gamma=0.0, prior_source="prior_bank", s_e_step=False,
        use_prior_bank=False, decode_bias=False,
        diagonal_covariance=True, gauge_group="block_glk", amp_dtype="fp32",
    )
    base.update(over)
    return types.SimpleNamespace(**base)


def _cost_model(*, K=6, n_gen=5, n_blocks=2):
    return types.SimpleNamespace(
        group=types.SimpleNamespace(
            generators=torch.zeros(n_gen, K, K),
            irrep_dims=[K // n_blocks] * n_blocks,
        ),
        parameters=lambda: iter(()),
    )


def test_cost_model_linear_decode_does_not_count_prior_bank_readout():
    V, K, n_gen = 11, 6, 5
    model = _cost_model(K=K, n_gen=n_gen)
    token_row = 2 * K + n_gen

    linear = _cost_model_fields(model, _cost_cfg(), n_params=123, tokens_seen=13)
    biased = _cost_model_fields(
        model,
        _cost_cfg(decode_bias=True),
        n_params=123,
        tokens_seen=13,
    )
    prior_bank = _cost_model_fields(
        model,
        _cost_cfg(use_prior_bank=True),
        n_params=123,
        tokens_seen=13,
    )

    assert linear["active_params_per_token"] == token_row + V * K
    assert biased["active_params_per_token"] == token_row + V * K + V
    assert prior_bank["active_params_per_token"] == token_row + 2 * V * K
    assert linear["flops_per_token_decode"] == prior_bank["flops_per_token_decode"] == 2.0 * V * K


def test_cost_model_counts_s_channel_estep_once_per_forward():
    V, K, n_gen, N, L, T, n_blocks = 11, 6, 5, 7, 2, 3, 2
    tokens_seen = 13
    model = _cost_model(K=K, n_gen=n_gen, n_blocks=n_blocks)
    cfg = _cost_cfg(
        max_seq_len=N,
        n_layers=L,
        n_e_steps=T,
        prior_source="model_channel",
        s_e_step=True,
    )

    out = _cost_model_fields(model, cfg, n_params=123, tokens_seen=tokens_seen)

    d_head = K / n_blocks
    estep_kernel = 2.0 * N * K + 2.0 * N * d_head * d_head
    belief_estep = L * T * estep_kernel
    s_estep = T * estep_kernel
    decode = 2.0 * V * K
    expected_active = (2 * K + n_gen) + V * K + 2 * V * K
    assert out["model_channel_active"] is True
    assert out["active_params_per_token"] == expected_active
    assert out["flops_per_token_estep"] == belief_estep + s_estep
    assert out["est_flops_analytic"] == (decode + belief_estep + s_estep) * tokens_seen


def _pure_ns(**over):
    r"""SimpleNamespace with every attribute _pure_path_report reads, at the pure values."""
    base = dict(
        include_attention_entropy=True, transport_mode="flat", lambda_alpha_mode="constant",
        use_prior_bank=True, use_head_mixer=False,
        lambda_beta=1.0, precision_weighted_attention=False,
        gauge_transport="on", pos_rotation="none", rope_full_gauge=False, rope_on_value=True,
        lambda_gamma=0.0, s_e_step=False,
        skip_belief_sigma_update=False, lambda_twohop=0.0,
        gauge_parameterization="phi", omega_reflection="off", phi_reflection="off",
        gauge_group="glk", family="gaussian_full")
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


def test_pure_path_marks_sigma_twohop_reflection_and_surrogates():
    rep = _pure_path_report(_pure_ns(
        skip_belief_sigma_update=True,
        lambda_twohop=0.25,
        precision_weighted_attention=True,
        gauge_parameterization="omega_direct",
        omega_reflection="metropolis",
    ), [])

    assert rep["on_pure_path"] is False
    assert rep["pure_flags"]["full_sigma_update"] is False
    assert rep["pure_flags"]["no_twohop_coupling"] is False
    assert rep["pure_flags"]["no_fixed_prior_surrogate"] is False
    assert rep["on_gauge_pure_path"] is False
    assert rep["gauge_flags"]["phi_parameterization"] is False
    assert rep["gauge_flags"]["no_reflection_sampling"] is False
    toggles = rep["config_toggles"]
    assert toggles["skip_belief_sigma_update"] is True
    assert toggles["lambda_twohop"] == 0.25
    assert toggles["gauge_parameterization"] == "omega_direct"
    assert toggles["omega_reflection"] == "metropolis"
    assert toggles["phi_reflection"] == "off"
    assert toggles["fixed_covariance_surrogate"] is True
    assert toggles["detached_precision_prior"] is True


# --------------------------------------------------------------------------- figure memory guard

def _finalize_ns(**over):
    r"""SimpleNamespace cfg for finalize_run: carries every attribute the numeric path touches, so
    the F9 tests do not depend on the concurrently-added ``force_large_figures`` config field."""
    base = dict(
        # figure memory-guard inputs: 8*V*N*B/1e9 ~ 13.2 GB fp32 logits+probs peak > the 8 GB guard
        vocab_size=50257, max_seq_len=1024, batch_size=32,
        generate_figures=True, force_large_figures=False,
        # numeric-path attrs (summary / provenance / cost model / pure-path report)
        seed=0, max_steps=2, use_prior_bank=True, decode_bias=False, use_head_mixer=False,
        include_attention_entropy=True, transport_mode="flat", lambda_alpha_mode="constant",
        lambda_beta=1.0, precision_weighted_attention=False,
        gauge_transport="on", pos_rotation="none", rope_full_gauge=False, rope_on_value=True,
        lambda_gamma=0.0, s_e_step=False, skip_belief_sigma_update=False, lambda_twohop=0.0,
        gauge_parameterization="phi", omega_reflection="off", phi_reflection="off",
        embed_dim=4, n_heads=2, n_layers=1, n_e_steps=1, diagonal_covariance=True,
        gauge_group="block_glk", family="gaussian_full",
        lambda_h=0.0, prior_source="prior_bank", amp_dtype="fp32")
    base.update(over)
    return types.SimpleNamespace(**base)


def test_finalize_run_delegates_memory_guard_to_figure_driver(tmp_path, monkeypatch):
    # The reusable figure driver owns the selective full-vocab guard. finalize_run must still run
    # the lighter figure pass and forward the explicit large-extractor policy.
    torch.manual_seed(0)
    real_cfg = VFE3Config(vocab_size=16, embed_dim=4, n_heads=2, max_seq_len=8, max_steps=2,
                          batch_size=2)
    model = VFEModel(real_cfg).to(DEVICE)
    art = RunArtifacts(str(tmp_path), real_cfg, model, dataset="synthetic", device=str(DEVICE))
    calls = []
    monkeypatch.setattr("vfe3.viz.report.generate_figures", lambda *a, **k: calls.append((a, k)))
    finalize_run(model, art, _finalize_ns(), test_loader=None)
    assert len(calls) == 1
    assert calls[0][1]["allow_large"] is False


def test_finalize_run_force_large_figures_overrides_guard(tmp_path, monkeypatch):
    # F9 counterpart: force_large_figures=True is the explicit large-run opt-in -- the figure pass
    # runs despite the over-budget estimate.
    torch.manual_seed(0)
    real_cfg = VFE3Config(vocab_size=16, embed_dim=4, n_heads=2, max_seq_len=8, max_steps=2,
                          batch_size=2)
    model = VFEModel(real_cfg).to(DEVICE)
    art = RunArtifacts(str(tmp_path), real_cfg, model, dataset="synthetic", device=str(DEVICE))
    calls = []
    monkeypatch.setattr("vfe3.viz.report.generate_figures", lambda *a, **k: calls.append((a, k)))
    finalize_run(model, art, _finalize_ns(force_large_figures=True), test_loader=None)
    assert len(calls) == 1
    assert calls[0][1]["allow_large"] is True


# --------------------------------------------------------------------------- held-out geometry columns

def test_train_logs_held_out_geometry(tmp_path):
    torch.manual_seed(0)
    cfg = VFE3Config(vocab_size=32, embed_dim=4, n_heads=2, max_seq_len=8, max_steps=4,
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


def _record_complete_scaling_cell(root, route, label, seed):
    path = root / "scaling_design.json"
    design = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {
        "schema_version": 1,
        "routes": [],
        "seeds": [],
        "status": "complete",
        "cells": [],
    }
    design["routes"] = sorted(set([*design["routes"], route]))
    design["seeds"] = sorted(set([*design["seeds"], seed]))
    design["cells"].append({
        "route": route,
        "label": label,
        "seed": seed,
        "status": "complete",
    })
    path.write_text(json.dumps(design), encoding="utf-8")


def _write_scaling_analysis_run(
    root,
    *,
    route,
    label,
    seed,
    n_params,
    test_ce,
    tokens_seen=1000,
    git_sha="git-a",
    train_sha="train-a",
    val_sha="val-a",
    test_sha="test-a",
):
    run = root / f"{route}_{label}_s{seed}"
    run.mkdir()
    (run / "summary.json").write_text(json.dumps({
        "n_params": n_params,
        "scaling_point": {
            "n_params": n_params,
            "n_learnable_params": n_params,
            "embed_dim": 4,
            "n_heads": 1,
            "n_gen": 16,
            "gauge_group": "glk",
            "n_layers": 1,
            "n_e_steps": 1,
            "tokens_seen": tokens_seen,
            "test_ce": test_ce,
        },
    }), encoding="utf-8")
    (run / "config.json").write_text(json.dumps({"config": {
        "seed": seed,
        "embed_dim": 4,
        "n_heads": 1,
        "n_layers": 1,
        "n_e_steps": 1,
        "family": "gaussian_diagonal",
    }}), encoding="utf-8")
    (run / "scaling_cell.json").write_text(json.dumps({
        "route": route,
        "scale_knob": "n_params",
        "label": label,
    }), encoding="utf-8")
    (run / "provenance.json").write_text(json.dumps({
        "seed": seed,
        "git_sha": git_sha,
        "train_data_sha256": train_sha,
        "val_data_sha256": val_sha,
        "test_data_sha256": test_sha,
        "data_sha256": test_sha,
    }), encoding="utf-8")
    _record_complete_scaling_cell(root, route, label, seed)


def test_all_scaling_tables_and_overlays_share_estimator(tmp_path, monkeypatch):
    # Two routes, four sizes each, and two seeds per cell give every table and
    # overlay enough sizes for the requested offset estimator.  With symmetric
    # seed values mean +/- 0.05, SEM is exactly 0.05 at every point.
    import pytest
    pytest.importorskip("scipy")

    for route_i, route in enumerate(("route_a", "route_b")):
        for n_params in (100, 200, 400, 800):
            mean = 2.0 + (5.0 + route_i) * n_params ** -0.3
            for seed, delta in ((1, -0.05), (2, 0.05)):
                _write_scaling_analysis_run(
                    tmp_path,
                    route=route,
                    label=f"n{n_params}",
                    seed=seed,
                    n_params=n_params,
                    test_ce=mean + delta,
                )

    real_fit = figs._fit_power_law
    calls = []

    def _record_fit(N, L, *, weights=None, with_offset=False):
        calls.append({
            "N": np.asarray(N, dtype=float).copy(),
            "L": np.asarray(L, dtype=float).copy(),
            "weights": None if weights is None else np.asarray(weights, dtype=float).copy(),
            "with_offset": with_offset,
        })
        return real_fit(N, L, weights=weights, with_offset=with_offset)

    monkeypatch.setattr(figs, "_fit_power_law", _record_fit)
    monkeypatch.setitem(scaling_analysis.CONFIG, "input_dir", str(tmp_path))
    monkeypatch.setitem(scaling_analysis.CONFIG, "with_offset", True)
    monkeypatch.setitem(scaling_analysis.CONFIG, "n_bootstrap", 0)

    scaling_analysis.analyze()

    assert len(calls) >= 8
    assert all(call["with_offset"] is True for call in calls)
    assert all(call["weights"] is not None for call in calls)
    for call in calls:
        np.testing.assert_allclose(call["weights"], (call["L"] / 0.05) ** 2, rtol=1e-12)

    summary = json.loads((tmp_path / "scaling_summary.json").read_text(encoding="utf-8"))
    assert summary["pooled_fit"]["form"] == "offset_power_law"
    assert all(fit["form"] == "offset_power_law" for fit in summary["per_route"].values())


def test_scaling_summary_persists_code_and_data_drift(tmp_path, monkeypatch):
    _write_scaling_analysis_run(
        tmp_path,
        route="grow_K",
        label="n100",
        seed=1,
        n_params=100,
        test_ce=4.5,
        tokens_seen=1000,
        git_sha="git-a",
        train_sha="train-a",
    )
    _write_scaling_analysis_run(
        tmp_path,
        route="grow_K",
        label="n200",
        seed=1,
        n_params=200,
        test_ce=4.2,
        tokens_seen=2000,
        git_sha="git-b",
        train_sha="train-b",
    )
    monkeypatch.setitem(scaling_analysis.CONFIG, "input_dir", str(tmp_path))
    monkeypatch.setitem(scaling_analysis.CONFIG, "with_offset", False)
    monkeypatch.setitem(scaling_analysis.CONFIG, "n_bootstrap", 0)
    monkeypatch.setattr(scaling_analysis, "_make_figures", lambda *args, **kwargs: None)

    scaling_analysis.analyze()

    summary = json.loads((tmp_path / "scaling_summary.json").read_text(encoding="utf-8"))
    provenance = summary["provenance"]
    assert provenance["git_sha"] == ["git-a", "git-b"]
    assert provenance["train_data_sha256"] == ["train-a", "train-b"]
    assert provenance["val_data_sha256"] == ["val-a"]
    assert provenance["test_data_sha256"] == ["test-a"]
    assert provenance["code_drift"] is True
    assert provenance["mixed_corpus"] is True
    assert provenance["token_budgets"] == [1000, 2000]
    assert provenance["token_budget_varies"] is True
    assert provenance["missing"] == {
        "git_sha": 0,
        "train_data_sha256": 0,
        "val_data_sha256": 0,
        "test_data_sha256": 0,
    }
    assert summary["pooled_fit_status"] == "confounded"
    assert set(summary["pooled_fit_confounds"]) == {"code_drift", "mixed_corpus", "token_budget_varies"}

    text = (tmp_path / "SCALING_ANALYSIS.md").read_text(encoding="utf-8")
    for value in ("git-a", "git-b", "train-a", "train-b", "1000", "2000", "confounded"):
        assert value in text


def test_divergent_routes_mark_pooled_fit_confounded(tmp_path, monkeypatch):
    for route, ce0 in (("grow_K", 4.6), ("blocks_K48_tied_2x", 4.1)):
        for i, n_params in enumerate((100, 200)):
            _write_scaling_analysis_run(
                tmp_path,
                route=route,
                label=f"n{n_params}",
                seed=1,
                n_params=n_params,
                test_ce=ce0 - 0.1 * i,
            )
    ancova = {
        "routes": {"blocks_K48_tied_2x": 2, "grow_K": 2},
        "testable": True,
        "F": 12.0,
        "df1": 2,
        "df2": 4,
        "p_value": 0.004,
        "rss_pooled": 0.4,
        "rss_full": 0.04,
        "collapses": False,
    }
    monkeypatch.setitem(scaling_analysis.CONFIG, "input_dir", str(tmp_path))
    monkeypatch.setitem(scaling_analysis.CONFIG, "with_offset", False)
    monkeypatch.setitem(scaling_analysis.CONFIG, "n_bootstrap", 0)
    monkeypatch.setattr(scaling_analysis, "ancova_frontier_collapse", lambda *args, **kwargs: ancova)
    monkeypatch.setattr(scaling_analysis, "_make_figures", lambda *args, **kwargs: None)

    scaling_analysis.analyze()

    summary = json.loads((tmp_path / "scaling_summary.json").read_text(encoding="utf-8"))
    assert summary["pooled_fit"] is not None
    assert summary["pooled_fit_status"] == "confounded"
    assert summary["pooled_fit_confounds"] == ["routes_diverge"]
    assert summary["frontier_collapse"] == ancova
    assert summary["route_notes"]["blocks_K48_tied_2x"] == (
        "tied structural ablation; not a strict full-covariance pure control"
    )

    text = (tmp_path / "SCALING_ANALYSIS.md").read_text(encoding="utf-8")
    assert "Pooled L(N) power law" in text
    assert "confounded" in text
    assert "tied structural ablation" in text
    assert "not a strict full-covariance pure control" in text


def test_indeterminate_ancova_is_unassessed_not_divergent(tmp_path, monkeypatch, capsys):
    for route, ce0 in (("route_a", 4.6), ("route_b", 4.4)):
        for i, n_params in enumerate((100, 200)):
            _write_scaling_analysis_run(
                tmp_path,
                route=route,
                label=f"n{n_params}",
                seed=1,
                n_params=n_params,
                test_ce=ce0 - 0.1 * i,
            )
    ancova = {
        "routes": {"route_a": 2, "route_b": 2},
        "testable": True,
        "F": 1.0,
        "df1": 2,
        "df2": 4,
        "p_value": float("nan"),
        "rss_pooled": 0.2,
        "rss_full": 0.1,
        "collapses": None,
    }
    monkeypatch.setitem(scaling_analysis.CONFIG, "input_dir", str(tmp_path))
    monkeypatch.setitem(scaling_analysis.CONFIG, "with_offset", False)
    monkeypatch.setitem(scaling_analysis.CONFIG, "n_bootstrap", 0)
    monkeypatch.setattr(scaling_analysis, "ancova_frontier_collapse", lambda *args, **kwargs: ancova)
    monkeypatch.setattr(scaling_analysis, "_make_figures", lambda *args, **kwargs: None)

    scaling_analysis.analyze()

    summary = json.loads((tmp_path / "scaling_summary.json").read_text(encoding="utf-8"))
    assert summary["pooled_fit"] is not None
    assert summary["pooled_fit_status"] == "unassessed"
    assert summary["pooled_fit_confounds"] == []
    console = capsys.readouterr().out.lower()
    assert "indeterminate" in console
    assert "routes diverge" not in console
    text = (tmp_path / "SCALING_ANALYSIS.md").read_text(encoding="utf-8").lower()
    assert "indeterminate" in text
    assert "routes diverge" not in text


def test_duplicate_single_n_does_not_persist_nan_fit(tmp_path, monkeypatch):
    for label, ce in (("arm_a", 4.5), ("arm_b", 4.3)):
        _write_scaling_analysis_run(
            tmp_path,
            route="grow_K_mup",
            label=label,
            seed=1,
            n_params=100,
            test_ce=ce,
        )
    monkeypatch.setitem(scaling_analysis.CONFIG, "input_dir", str(tmp_path))
    monkeypatch.setitem(scaling_analysis.CONFIG, "with_offset", True)
    monkeypatch.setitem(scaling_analysis.CONFIG, "n_bootstrap", 0)
    monkeypatch.setattr(scaling_analysis, "_make_figures", lambda *args, **kwargs: None)

    scaling_analysis.analyze()

    raw = (tmp_path / "scaling_summary.json").read_text(encoding="utf-8")
    summary = json.loads(raw)
    assert summary["pooled_fit"] is None
    assert summary["pooled_fit_status"] == "not_fitted"
    assert summary["per_route"] == {}

    def _reject_nonfinite_json(value):
        pytest.fail(f"scaling summary persisted non-finite JSON constant {value!r}")

    assert json.loads(raw, parse_constant=_reject_nonfinite_json) == summary
    text = (tmp_path / "SCALING_ANALYSIS.md").read_text(encoding="utf-8").lower()
    assert "status: **not_fitted**" in text
    assert "no pooled estimate is available" in text


def test_ancova_requires_distinct_sizes_per_route(tmp_path, monkeypatch):
    for route, n_params, ce0 in (("route_a", 100, 4.5), ("route_b", 200, 4.2)):
        for arm in range(3):
            _write_scaling_analysis_run(
                tmp_path,
                route=route,
                label=f"arm_{arm}",
                seed=1,
                n_params=n_params,
                test_ce=ce0 - 0.02 * arm,
            )
    monkeypatch.setitem(scaling_analysis.CONFIG, "input_dir", str(tmp_path))
    monkeypatch.setitem(scaling_analysis.CONFIG, "with_offset", False)
    monkeypatch.setitem(scaling_analysis.CONFIG, "n_bootstrap", 0)
    monkeypatch.setattr(scaling_analysis, "_make_figures", lambda *args, **kwargs: None)

    scaling_analysis.analyze()

    summary = json.loads((tmp_path / "scaling_summary.json").read_text(encoding="utf-8"))
    assert summary["pooled_fit"] is not None
    assert summary["frontier_collapse"]["testable"] is False
    assert summary["frontier_collapse"]["routes"] == {}
    assert summary["pooled_fit_status"] == "unassessed"


def test_scaling_md_persists_not_fitted_status(tmp_path):
    out = tmp_path / "SCALING_ANALYSIS.md"
    scaling_analysis._write_scaling_md(out, {
        "input_dir": str(tmp_path),
        "n_runs": 1,
        "n_param_points": 1,
        "n_inference_points": 0,
        "with_offset": True,
        "pooled_fit": None,
        "pooled_fit_status": "not_fitted",
        "pooled_fit_confounds": [],
    })

    text = out.read_text(encoding="utf-8")
    assert text.count("## Pooled L(N) power law") == 1
    assert "status: **not_fitted**" in text
    assert "no pooled estimate is available" in text


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
        "provenance": {"git_sha": ["abc"], "train_data_sha256": ["train"],
                       "val_data_sha256": ["val"], "test_data_sha256": ["test"],
                       "data_sha256": ["test"], "code_drift": False, "mixed_corpus": False,
                       "token_budgets": [1000], "token_budget_varies": False, "missing": {}},
        "pooled_fit_status": "clean", "pooled_fit_confounds": [],
        "route_notes": {},
    }
    out = tmp_path / "SCALING_ANALYSIS.md"
    scaling_analysis._write_scaling_md(out, summary)
    text = out.read_text(encoding="utf-8")
    for section in ("Pooled L(N) power law", "Per-route exponents", "Frontier-collapse F-test",
                    "E-step structural-EM check", "Provenance and confounds"):
        assert section in text


# =============================================================================
# PB-07 registry-completion Task 5: end-to-end production-driver integration.
#
# Both tests below monkeypatch the SPECIFIC registry slot each new figure claims
# (``vfe3.viz.figures._FIGURES[name]``), not a direct-import reference. If either production
# driver ever regressed to calling its plotter by direct import (bypassing ``get_figure``), the
# stub would never run, the asserted PNG would still carry the old real-plot bytes (or be
# missing entirely), and the captured-kwargs assertion below would KeyError -- so this is a
# substitution test for "resolves by name", not just an existence check. Both drivers avoid any
# model replay: ``scaling_analysis.analyze()`` reads only persisted JSON fixtures, and
# ``ablation.main()`` runs with ``run_single`` stubbed to a fast fabricated cell.
# =============================================================================

def _write_scaling_validation_run(root, *, route, label, embed_dim, n_params, test_ce,
                                  best_val_ppl, seed=1, wall_time_s=12.5):
    r"""One persisted ``scaling_analysis`` run directory carrying BOTH the test-metric fields
    ``aggregate_points`` reads (feeding the legacy scaling_ce_vs_params.png fit) and the
    top-level ``best_val_ppl`` / ``wall_time_s`` fields ``aggregate_validation_points`` reads
    (feeding the two PB-07 registered validation figures)."""
    run = root / f"{route}_{label}_s{seed}"
    run.mkdir()
    (run / "summary.json").write_text(json.dumps({
        "n_params":      n_params,
        "best_val_ppl":  best_val_ppl,
        "wall_time_s":   wall_time_s,
        "scaling_point": {
            "n_params": n_params, "n_learnable_params": n_params, "embed_dim": embed_dim,
            "n_heads": 1, "n_gen": 16, "gauge_group": "glk", "n_layers": 1, "n_e_steps": 1,
            "tokens_seen": 1000, "test_ce": test_ce,
        },
    }), encoding="utf-8")
    (run / "config.json").write_text(json.dumps({"config": {
        "seed": seed, "embed_dim": embed_dim, "n_heads": 1, "n_layers": 1, "n_e_steps": 1,
        "family": "gaussian_diagonal",
    }}), encoding="utf-8")
    (run / "scaling_cell.json").write_text(json.dumps({
        "route": route, "scale_knob": "embed_dim", "label": label,
    }), encoding="utf-8")
    (run / "provenance.json").write_text(json.dumps({
        "seed": seed, "git_sha": "git-a", "train_data_sha256": "train-a",
        "val_data_sha256": "val-a", "test_data_sha256": "test-a", "data_sha256": "test-a",
    }), encoding="utf-8")
    _record_complete_scaling_cell(root, route, label, seed)


def _stub_registered_figure(name, cap):
    r"""A registered-figure stub that records the exact kwargs its caller resolved for ``name``
    and writes a real (nonempty) PNG to ``path``, standing in for the true plotter."""
    def _fn(*, path=None, **kwargs):
        cap[name] = kwargs
        fig = plt.figure()
        if path is not None:
            fig.savefig(path)
        return fig
    return _fn


def test_scaling_analyze_dispatches_registered_figures_by_name_and_keeps_legacy_outputs(
    tmp_path, monkeypatch,
):
    # Tiny persisted scaling fixture: 3 embed_dim points on route "grow_K" (AXIS_ROUTES maps the
    # embed_dim axis to this route), each carrying a finite test_ce (legacy fit) and best_val_ppl
    # (the PB-07 validation figures) so both the legacy and the two new figures have enough points.
    _write_scaling_validation_run(tmp_path, route="grow_K", label="K8",  embed_dim=8,
                                  n_params=100, test_ce=4.5, best_val_ppl=30.0)
    _write_scaling_validation_run(tmp_path, route="grow_K", label="K16", embed_dim=16,
                                  n_params=200, test_ce=4.2, best_val_ppl=24.0)
    _write_scaling_validation_run(tmp_path, route="grow_K", label="K32", embed_dim=32,
                                  n_params=400, test_ce=4.0, best_val_ppl=20.0)

    monkeypatch.setitem(scaling_analysis.CONFIG, "input_dir", str(tmp_path))
    monkeypatch.setitem(scaling_analysis.CONFIG, "with_offset", False)
    monkeypatch.setitem(scaling_analysis.CONFIG, "n_bootstrap", 0)

    cap = {}
    monkeypatch.setitem(figs._FIGURES, "capacity_scaling", _stub_registered_figure("capacity_scaling", cap))
    monkeypatch.setitem(figs._FIGURES, "pareto_frontier", _stub_registered_figure("pareto_frontier", cap))

    scaling_analysis.analyze()

    fig_dir = tmp_path / "figures"
    # The two PB-07 figures exist ONLY because the stub registered under their name ran --
    # emit_registered_figures resolved them through get_figure("capacity_scaling" / "pareto_frontier").
    assert (fig_dir / "capacity_scaling.png").exists() and (fig_dir / "capacity_scaling.png").stat().st_size > 0
    assert (fig_dir / "pareto_frontier.png").exists() and (fig_dir / "pareto_frontier.png").stat().st_size > 0
    assert "embed_dim" in cap["capacity_scaling"]["scaling"]
    assert {"bits_per_token", "n_params"} <= set(cap["pareto_frontier"]["points"])

    # Legacy scaling outputs are unaffected: the test-metric report, csv, and headline figure.
    assert (tmp_path / "scaling_points.csv").exists()
    assert (tmp_path / "scaling_summary.json").exists()
    assert (tmp_path / "SCALING_ANALYSIS.md").exists()
    assert (fig_dir / "scaling_ce_vs_params.png").exists()


_FIXED_ABLATION_CODE_IDENTITY = {"git_sha": "a" * 40, "git_dirty": False, "git_dirty_fingerprint": None}


def _fake_ablation_source_ok(dataset, split, *, cache_dir=None):
    return {"format": "pt", "tokenizer_tag": "tiktoken", "size_bytes": len(split),
            "sha256": "0" * 64 + split, "meta": None, "meta_sha256": None}


def test_ablation_main_dispatches_registered_figures_by_name_and_keeps_legacy_outputs(
    tmp_path, monkeypatch,
):
    # Stub the contract-building identity seams exactly as the existing PB-07 ablation-report
    # tests do (tests/test_ablation_reporting.py::_stub_sweep_identity), and stub run_single so no
    # model ever trains: each cell's terminal PPL is fabricated, and forest cells additionally
    # publish a real val_token_nats.pt with a correctly recomputed identity.
    monkeypatch.setattr(ablation, "_git_code_identity", lambda: dict(_FIXED_ABLATION_CODE_IDENTITY))
    monkeypatch.setattr(ablation, "cache_source_identity", _fake_ablation_source_ok)
    monkeypatch.setattr(ablation, "_cleanup", lambda: None)

    forest_offsets = {"baseline": 0.0, "head_mixer_off": 0.4, "precision_attention_off": 0.25}

    def fake_run_single(label, overrides, run_dir, **kwargs):
        run_dir.mkdir(parents=True, exist_ok=True)
        result = {"label": label, "error_kind": None, "seed": 6,
                  "overrides": ablation._jsonable(overrides)}
        if kwargs.get("paired_token_bootstrap"):
            idx = list(forest_offsets).index(label)
            g = torch.Generator().manual_seed(1000 + idx)
            vec = torch.rand(48, generator=g, dtype=torch.float32) * 0.1 + 1.0 + forest_offsets[label]
            tpath = run_dir / "val_token_nats.pt"
            torch.save(vec, tpath)
            ppl = float(vec.mean())
            result.update({
                "primary_val_ppl": ppl, "final_val_ppl": ppl,
                "val_token_nats_path":       "val_token_nats.pt",
                "val_token_nats_sha256":     hashlib.sha256(tpath.read_bytes()).hexdigest(),
                "val_token_nats_size_bytes": tpath.stat().st_size,
                "val_token_nats_numel":      int(vec.numel()),
                "val_token_nats_dtype":      str(vec.dtype),
            })
        else:
            mu, sigma = overrides["e_q_mu_lr"], overrides["e_q_sigma_lr"]
            ppl = 10.0 + (mu - 0.7) ** 2 * 5.0 + (sigma - 0.0005) ** 2 * 2000.0
            result.update({"primary_val_ppl": ppl, "final_val_ppl": ppl})
        return result

    monkeypatch.setattr(ablation, "run_single", fake_run_single)

    cap = {}
    monkeypatch.setitem(figs._FIGURES, "ablation_forest", _stub_registered_figure("ablation_forest", cap))
    monkeypatch.setitem(figs._FIGURES, "lr_grid_heatmap", _stub_registered_figure("lr_grid_heatmap", cap))

    monkeypatch.setitem(ablation.CONFIG, "output_dir", str(tmp_path))
    monkeypatch.setitem(ablation.CONFIG, "device", "cpu")
    monkeypatch.setitem(ablation.CONFIG, "dataset", "wikitext-103")
    monkeypatch.setitem(ablation.CONFIG, "resume", False)
    monkeypatch.setitem(ablation.CONFIG, "seed", 6)
    monkeypatch.setitem(ablation.CONFIG, "max_tokens", None)
    monkeypatch.setitem(ablation.CONFIG, "max_steps", None)
    monkeypatch.setitem(ablation.CONFIG, "list_only", False)

    # Two production main() calls, one per opt-in report sweep (both out of SWEEP_ORDER, so each
    # must be named explicitly); both write into the same output_dir/figures.
    monkeypatch.setitem(ablation.CONFIG, "sweep", "component_ablation_forest")
    ablation.main()
    monkeypatch.setitem(ablation.CONFIG, "sweep", "e_q_mu_sigma_lr_grid")
    ablation.main()

    fig_dir = tmp_path / "figures"
    assert (fig_dir / "ablation_forest.png").exists() and (fig_dir / "ablation_forest.png").stat().st_size > 0
    assert (fig_dir / "lr_grid_heatmap.png").exists() and (fig_dir / "lr_grid_heatmap.png").stat().st_size > 0
    assert {r["label"] for r in cap["ablation_forest"]["rows"]} == set(forest_offsets)
    grid = cap["lr_grid_heatmap"]["grid"]
    assert grid["z"].shape == (len(ablation._GRID_SIGMA_LRS), len(ablation._GRID_MU_LRS))

    # Legacy per-sweep outputs (CSV, metadata, PPL figure) survive for both sweeps.
    for name in ("component_ablation_forest", "e_q_mu_sigma_lr_grid"):
        sweep_dir = tmp_path / name
        assert (sweep_dir / "sweep_results.csv").exists()
        assert (sweep_dir / "sweep_meta.json").exists()
        assert (fig_dir / f"{name}.png").exists()
