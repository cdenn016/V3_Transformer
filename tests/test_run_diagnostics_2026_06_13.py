r"""Per-run observability rollout (2026-06-13): the Tier-1 hot-path scalars.

Pins the new diagnostics() reductions and the metrics.csv columns added by the run-diagnostics
rollout: pre-clip gradient norms (the value clip_grad_norm_ used to discard), guard saturation,
belief conditioning, the group-correct gauge invariant, phi-frame magnitude, the per-token
effective-rank distribution, throughput/peak-memory, and the derived generalization / ELBO-CE
gaps. Also pins that the equivariance-break order parameters (connection_w_norm, head_mixer_drift)
are CONDITIONAL columns -- present only when their toggle is on -- so the CSV stays rectangular.

Device-agnostic (default CPU; VFE3_TEST_DEVICE=cuda for the GPU).
"""

import csv
import math
import os
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from vfe3.config import VFE3Config
from vfe3.data.datasets import TokenWindows
from vfe3.model.model import VFEModel
from vfe3.train import build_optimizer, train, train_step, _floor_lr_lambdas

DEVICE = torch.device(os.environ.get("VFE3_TEST_DEVICE", "cpu"))

# diagnostics() keys the rollout adds (all should be present and finite on the pure path).
_NEW_DIAG_KEYS = [
    "holonomy_ci_lo", "holonomy_ci_hi", "gauge_invariant_mean", "gauge_invariant_spread",
    "phi_norm_mean", "phi_norm_std", "belief_cond_median", "belief_cond_p95", "belief_cond_max",
    "belief_pd_margin", "eff_rank_p5", "eff_rank_median", "eff_rank_p95", "fisher_trace_mean",
    "fisher_trace_median", "guard_sigma_floor_frac", "guard_sigma_ceil_frac",
    "guard_energy_klmax_frac", "guard_selfdiv_klmax_frac", "nonfinite_frac",
    "attn_entropy_min", "attn_entropy_collapsed_heads",
]


def _cfg(**kw) -> VFE3Config:
    base = dict(vocab_size=16, embed_dim=8, n_heads=2, max_seq_len=8, n_layers=1,
                batch_size=4, max_steps=20, warmup_steps=2)
    base.update(kw)
    return VFE3Config(**base)


def _loader(cfg: VFE3Config) -> DataLoader:
    stream = (torch.arange(cfg.vocab_size).repeat(800 // cfg.vocab_size + 2)[:800]).long()
    ds = TokenWindows(stream, cfg.max_seq_len)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, drop_last=True)


def test_diagnostics_has_extended_keys_finite() -> None:
    torch.manual_seed(0)
    cfg = _cfg()
    model = VFEModel(cfg).to(DEVICE)
    tok = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len), device=DEVICE)
    d = model.diagnostics(tok)
    missing = [k for k in _NEW_DIAG_KEYS if k not in d]
    assert not missing, f"diagnostics missing new keys: {missing}"
    assert all(math.isfinite(d[k]) for k in _NEW_DIAG_KEYS)
    # existing invariants untouched: total still present, holonomy mean key preserved
    assert "total" in d and "holonomy_deviation" in d
    # pure path: clamps inert and frames non-collapsed
    assert d["guard_sigma_floor_frac"] == 0.0 and d["guard_sigma_ceil_frac"] == 0.0
    assert d["belief_pd_margin"] > 1.0


def test_train_step_metrics_out_captures_grad_norm() -> None:
    torch.manual_seed(0)
    cfg = _cfg()
    model = VFEModel(cfg).to(DEVICE)
    opt = build_optimizer(model, cfg)
    base_lrs = [g["lr"] for g in opt.param_groups]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _floor_lr_lambdas(base_lrs, cfg))
    tok = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.max_seq_len), device=DEVICE)
    tgt = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.max_seq_len), device=DEVICE)
    out: dict = {}
    loss = train_step(model, opt, sched, tok, tgt, grad_clip=1.0, metrics_out=out)
    assert isinstance(loss, float) and math.isfinite(loss)         # bare-float return preserved
    assert math.isfinite(out["grad_norm"]) and out["grad_norm"] >= 0.0
    for k in ("grad_norm_mu", "grad_norm_sigma", "grad_norm_phi"):
        assert k in out and math.isfinite(out[k])
    assert out["loss_finite"] == 1.0
    # global norm is the L2 combine of the per-group norms (groups 0/1/2 are mu/sigma/phi + any extra)
    assert out["grad_norm"] >= out["grad_norm_mu"] - 1e-5


def test_train_step_without_metrics_out_is_plain_float() -> None:
    torch.manual_seed(0)
    cfg = _cfg()
    model = VFEModel(cfg).to(DEVICE)
    opt = build_optimizer(model, cfg)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _floor_lr_lambdas([g["lr"] for g in opt.param_groups], cfg))
    tok = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.max_seq_len), device=DEVICE)
    tgt = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.max_seq_len), device=DEVICE)
    loss = train_step(model, opt, sched, tok, tgt, grad_clip=1.0)   # metrics_out defaults None
    assert isinstance(loss, float) and math.isfinite(loss)


def test_metrics_csv_has_tier1_columns_and_is_rectangular() -> None:
    from vfe3.run_artifacts import RunArtifacts
    torch.manual_seed(0)
    cfg = _cfg(log_interval=5, eval_interval=10, eval_max_batches=2)
    model = VFEModel(cfg).to(DEVICE)
    loader = _loader(cfg)
    with tempfile.TemporaryDirectory() as tmp:
        art = RunArtifacts(tmp, cfg, model, dataset="synthetic-period3", device=str(DEVICE))
        train(model, loader, cfg, n_steps=20, log_interval=5, eval_interval=10,
              val_loader=loader, artifacts=art, device=DEVICE, generate_samples=False)
        with open(Path(tmp) / "metrics.csv", newline="") as fh:
            rows = list(csv.DictReader(fh))
    assert rows, "no metrics rows written"
    cols = set(rows[0].keys())
    need = (["grad_norm", "grad_norm_mu", "grad_norm_sigma", "grad_norm_phi", "loss_finite",
             "tokens_per_s", "peak_mem_mb", "generalization_gap",
             "self_coupling", "self_divergence"] + _NEW_DIAG_KEYS)
    assert not [c for c in need if c not in cols], f"CSV missing Tier-1 columns: {[c for c in need if c not in cols]}"
    # rectangular: every row carries the identical column set
    assert all(set(r.keys()) == cols for r in rows)
    # the pure path logs no equivariance-break order parameters
    assert "connection_w_norm" not in cols and "head_mixer_drift" not in cols


def test_conditional_break_columns_present_only_under_toggle() -> None:
    torch.manual_seed(0)
    tok = torch.randint(0, 16, (2, 8), device=DEVICE)
    # pure path: neither order parameter
    d0 = VFEModel(_cfg()).to(DEVICE).diagnostics(tok)
    assert "connection_w_norm" not in d0 and "head_mixer_drift" not in d0
    # regime_ii -> connection_w_norm column appears (zero at W=0 init, but present)
    d_r2 = VFEModel(_cfg(transport_mode="regime_ii")).to(DEVICE).diagnostics(tok)
    assert "connection_w_norm" in d_r2 and math.isfinite(d_r2["connection_w_norm"])
    # head mixer -> head_mixer_drift column appears (zero at identity init, but present)
    d_hm = VFEModel(_cfg(use_head_mixer=True)).to(DEVICE).diagnostics(tok)
    assert "head_mixer_drift" in d_hm and math.isfinite(d_hm["head_mixer_drift"])


_TIER2A_KEYS = [
    "cocycle_residual", "vertex_cond_max", "sandwich_absmax", "transport_asymmetry",
    "energy_abs_asymmetry", "energy_rel_asymmetry", "gauge_head_aniso_mean", "gauge_head_logdet_spread",
]


def test_diagnostics_has_tier2a_transport_gauge_keys() -> None:
    torch.manual_seed(0)
    cfg = _cfg()
    model = VFEModel(cfg).to(DEVICE)
    tok = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len), device=DEVICE)
    d = model.diagnostics(tok)
    missing = [k for k in _TIER2A_KEYS if k not in d]
    assert not missing, f"diagnostics missing Tier-2a keys: {missing}"
    assert all(math.isfinite(d[k]) for k in _TIER2A_KEYS)
    # the flat (default) transport is an EXACT phi-cocycle -> the cocycle residual is ~0
    assert d["cocycle_residual"] < 1e-3
    # directed phi-cocycle Omega_ij = exp(phi_i) exp(-phi_j) breaks i<->j symmetry -> asymmetry >= 0
    assert d["transport_asymmetry"] >= 0.0 and d["vertex_cond_max"] >= 1.0


def test_val_diagnostics_columns_rectangular_and_finite() -> None:
    from vfe3.run_artifacts import RunArtifacts
    from vfe3.train import _VAL_DIAG_KEYS
    torch.manual_seed(0)
    cfg = _cfg(log_interval=6, eval_interval=12, eval_max_batches=2)
    model = VFEModel(cfg).to(DEVICE)
    loader = _loader(cfg)
    with tempfile.TemporaryDirectory() as tmp:
        art = RunArtifacts(tmp, cfg, model, dataset="synthetic-period3", device=str(DEVICE))
        train(model, loader, cfg, n_steps=24, log_interval=6, eval_interval=12,
              val_loader=loader, artifacts=art, device=DEVICE, generate_samples=False)
        with open(Path(tmp) / "metrics.csv", newline="") as fh:
            rows = list(csv.DictReader(fh))
    cols = set(rows[0].keys())
    assert not [k for k in _VAL_DIAG_KEYS if k not in cols], "missing val-diag columns"
    assert all(set(r.keys()) == cols for r in rows), "CSV not rectangular"
    # after the step-24 eval the held-out probes are finite
    last = rows[-1]
    for k in ("val_free_energy_total", "estep_f_drop", "val_future_leakage", "pos_loss_ratio",
              "val_head_redundancy_js"):
        assert math.isfinite(float(last[k])), f"{k} not finite: {last[k]!r}"
    # the soft causal prior must not leak future tokens
    assert float(last["val_future_leakage"]) < 1e-3
    # Tier-2a transport/gauge + weight-norm columns present too
    assert {"cocycle_residual", "transport_asymmetry", "weight_norm_mu"} <= cols


def test_diagnostics_has_renyi_band_frac() -> None:
    torch.manual_seed(0)
    model = VFEModel(_cfg()).to(DEVICE)
    d = model.diagnostics(torch.randint(0, 16, (2, 8), device=DEVICE))
    assert "renyi_band_frac" in d and 0.0 <= d["renyi_band_frac"] <= 1.0


def test_self_divergence_is_raw_unregularized_self_coupling() -> None:
    # self_coupling logs the alpha-regularized F self-term sum_i[alpha_i D + R(alpha_i)]; the
    # self_divergence column logs the RAW sum_i D(q_i||p_i). They coincide ONLY at
    # lambda_alpha_mode='constant' (alpha=1, R=0); under the state-dependent per-coordinate envelope the
    # R(alpha*) regularizer floor makes self_coupling distinct from (and uninformative vs) raw D.
    torch.manual_seed(0)
    tok = torch.randint(0, 16, (2, 8), device=DEVICE)

    d_const = VFEModel(_cfg(lambda_alpha_mode="constant")).to(DEVICE).diagnostics(tok)
    assert "self_divergence" in d_const and math.isfinite(d_const["self_divergence"])
    assert d_const["self_divergence"] >= 0.0
    # constant alpha=1 with no regularizer  ->  raw divergence == the logged self-coupling term
    assert math.isclose(d_const["self_coupling"], d_const["self_divergence"], rel_tol=1e-5, abs_tol=1e-5)

    # state-dependent per-coord: the regularizer floor lifts self_coupling off the raw divergence
    d_sd = VFEModel(_cfg(lambda_alpha_mode="state_dependent_per_coord")).to(DEVICE).diagnostics(tok)
    assert "self_divergence" in d_sd and math.isfinite(d_sd["self_divergence"])
    assert not math.isclose(d_sd["self_coupling"], d_sd["self_divergence"], rel_tol=1e-6, abs_tol=1e-6)


def test_group_gauge_invariant_sp_is_conjugation_invariant() -> None:
    # Adversarial-review HIGH fix: the sp/sp_n invariant must be invariant under GL congruence
    # (conjugation of exp(phi)), which the singular-value squeeze was NOT (Sp is non-orthogonal).
    from vfe3.metrics import group_gauge_invariant
    torch.manual_seed(0)

    class _G:
        name = "sp"

    k = 4
    exp_phi = torch.matrix_exp(torch.randn(k, k) * 0.3).unsqueeze(0)        # (1, K, K)
    g = torch.matrix_exp(torch.randn(k, k) * 0.5)                           # generic GL conjugation
    moved = g @ exp_phi @ torch.linalg.inv(g)
    base = group_gauge_invariant(exp_phi, _G())
    conj = group_gauge_invariant(moved, _G())
    assert torch.allclose(base, conj, atol=1e-4), (float(base), float(conj))


def test_guard_saturation_full_cov_floor_binds() -> None:
    # Adversarial-review HIGH fix: on a full covariance whose SPD floor binds, sigma_floor_frac must
    # not read 0 just because eigvalsh noise exceeds the tight relative window.
    from vfe3.metrics import guard_saturation
    eps = 1e-6
    k = 3
    q = torch.linalg.qr(torch.randn(k, k))[0]
    sigma1 = q @ torch.diag(torch.tensor([eps, 1.0, 10.0])) @ q.T           # cond ~1e7, smallest at floor
    sigma1 = 0.5 * (sigma1 + sigma1.T)
    sig = sigma1.unsqueeze(0).expand(4, k, k).contiguous()                  # (4, K, K) full cov
    gs = guard_saturation(sig, torch.zeros(4, 4), torch.zeros(4), eps=eps, sigma_max=100.0)
    # one of three eigenvalues per token sits at the floor -> ~1/3 of spectrum entries; the point is
    # it is NONZERO (the tight relative window read exactly 0 here before the eigensolver-noise fix).
    assert gs["sigma_floor_frac"] > 0.3, gs["sigma_floor_frac"]


def test_finalize_writes_tier3_research_and_provenance() -> None:
    import json
    from vfe3.run_artifacts import RunArtifacts, finalize_run
    torch.manual_seed(0)
    cfg = _cfg(log_interval=6, eval_interval=12, eval_max_batches=2, generate_figures=False)
    model = VFEModel(cfg).to(DEVICE)
    loader = _loader(cfg)
    with tempfile.TemporaryDirectory() as tmp:
        art = RunArtifacts(tmp, cfg, model, dataset="synthetic-period3", device=str(DEVICE))
        losses = train(model, loader, cfg, n_steps=24, log_interval=6, eval_interval=12,
                       val_loader=loader, artifacts=art, device=DEVICE, generate_samples=False)
        res = finalize_run(model, art, cfg, test_loader=loader, losses=losses, device=DEVICE)
        root = Path(tmp)
        # the n_e_steps=0 capacity-gain falsifier is computed and the budget is restored
        assert "estep_capacity_gain" in res and math.isfinite(res["estep_capacity_gain"])
        assert model.cfg.n_e_steps == cfg.n_e_steps                # restored after the probe
        # provenance.json: code/data/env state a config-only record omits
        prov = json.loads((root / "provenance.json").read_text())
        assert {"seed", "torch_version", "git_sha", "git_dirty"} <= set(prov)
        # summary.json scaling-law point
        summ = json.loads((root / "summary.json").read_text())
        assert {"n_params", "tokens_seen", "est_flops_6ND"} <= set(summ["scaling_point"])
        assert summ["scaling_point"]["tokens_seen"] == cfg.max_steps * cfg.batch_size * cfg.max_seq_len
        # research.json: calibration + frequency strata + FD gradient check
        rj = json.loads((root / "research.json").read_text())
        assert math.isfinite(rj["ece"]) and "freq_strata_ce" in rj
        assert math.isfinite(rj["fd_gradient_worst_rel_error"]) and rj["fd_gradient_worst_rel_error"] >= 0.0
        # history-only trend figures
        for fig in ("grad_norm.png", "belief_condition.png", "estep_convergence_trend.png"):
            assert (root / fig).exists(), f"missing trend figure {fig}"
