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
             "tokens_per_s", "peak_mem_mb", "generalization_gap", "elbo_ce_gap"] + _NEW_DIAG_KEYS)
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
