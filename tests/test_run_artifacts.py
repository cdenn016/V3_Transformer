r"""Training run artifacts: run dir, config.json, metrics.csv, checkpoints, best_model.pt,
end-of-run TEST eval, summary.json, and figures.

These pin the persistence plumbing the user found missing (training ran but saved nothing).
The proof is files on disk, so the integration tests assert the actual artifacts appear; the
silent path (no artifacts object) must write nothing and stay bitwise-identical (the latter is
covered by tests/test_train.py::test_silent_and_logging_paths_are_bitwise_identical).
"""

import json
import math
import types

import pytest
import torch
from torch.utils.data import DataLoader

from vfe3.config import VFE3Config
from vfe3.data.datasets import TokenWindows
from vfe3.model.model import VFEModel
from vfe3.run_artifacts import RunArtifacts, _pure_path_report, finalize_run
from vfe3.train import build_optimizer, train


def _loader(seed=0, n=600, seq_len=8, bs=8):
    g = torch.Generator().manual_seed(seed)
    base = torch.arange(3).repeat(n // 3 + 2)               # period-3 stream over {0,1,2}
    ds = TokenWindows(base[:n].long(), seq_len)
    return DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True, generator=g)


def _cfg(**kw):
    base = dict(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                n_e_steps=1, e_q_mu_lr=0.1, e_phi_lr=0.0, m_phi_lr=0.0,
                warmup_steps=1, max_steps=4)
    base.update(kw)
    return VFE3Config(**base)


def test_config_checkpoint_interval_default_and_validated():
    assert VFE3Config().checkpoint_interval == 25000
    assert VFE3Config(checkpoint_interval=1000).checkpoint_interval == 1000
    with pytest.raises(ValueError):
        VFE3Config(checkpoint_interval=-1)


def test_creates_run_dir_and_config_json(tmp_path):
    cfg = _cfg()
    model = VFEModel(cfg)
    RunArtifacts(tmp_path / "run", cfg, model, dataset="synthetic", device="cpu")
    assert (tmp_path / "run").is_dir()
    assert (tmp_path / "run" / "checkpoints").is_dir()
    meta = json.loads((tmp_path / "run" / "config.json").read_text())
    assert meta["dataset"] == "synthetic"
    assert meta["n_params"] == sum(p.numel() for p in model.parameters())
    assert meta["config"]["embed_dim"] == 4


def test_log_metrics_writes_csv_with_header(tmp_path):
    cfg = _cfg()
    model = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "r", cfg, model)
    art.log_metrics({"step": 1, "val_ppl": 3.0})
    art.log_metrics({"step": 2, "val_ppl": 2.5})
    lines = (tmp_path / "r" / "metrics.csv").read_text().strip().splitlines()
    assert lines[0].split(",") == ["step", "val_ppl"]
    assert len(lines) == 3                                     # header + 2 rows


def test_maybe_save_best_only_on_improvement(tmp_path):
    cfg = _cfg()
    model = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "r", cfg, model)
    assert art.maybe_save_best(1, model, 10.0) is True
    assert (tmp_path / "r" / "best_model.pt").exists()
    assert art.maybe_save_best(2, model, 12.0) is False       # worse PPL -> no save
    assert art.maybe_save_best(3, model, 8.0) is True
    assert art.best_val_ppl == 8.0 and art.best_step == 3


def test_save_checkpoint_is_loadable(tmp_path):
    cfg = _cfg()
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)
    art = RunArtifacts(tmp_path / "r", cfg, model)
    p = art.save_checkpoint(4, model, opt, cfg)
    assert p.exists()
    ckpt = torch.load(p, weights_only=False)
    assert ckpt["step"] == 4
    assert "model_state" in ckpt and "optimizer_state" in ckpt
    # model-selection state is bundled so a resumed run reports the run-wide best (audit 2026-07-01 C2)
    assert "best_val_ppl" in ckpt and "best_step" in ckpt


def test_writes_are_atomic_no_temp_left(tmp_path):
    # C11 (audit 2026-07-01): every writer publishes via same-dir .tmp + os.replace, so no temp
    # file survives a successful write and every final artifact loads cleanly.
    cfg = _cfg()
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)
    art = RunArtifacts(tmp_path / "r", cfg, model)
    art.save_json("summary.json", {"a": 1})
    assert art.maybe_save_best(1, model, 5.0) is True
    p = art.save_checkpoint(2, model, opt, cfg)
    assert list((tmp_path / "r").rglob("*.tmp")) == []          # run_dir AND ckpt_dir hold no temps
    assert json.loads((tmp_path / "r" / "summary.json").read_text()) == {"a": 1}
    best = torch.load(tmp_path / "r" / "best_model.pt", weights_only=True)
    assert set(best) == set(model.state_dict())
    ckpt = torch.load(p, weights_only=True)
    assert ckpt["step"] == 2


def test_best_model_overwrite_replaces(tmp_path):
    # C11: os.replace over an EXISTING best_model.pt succeeds (Windows lock retry path aside),
    # and the file reloads to the improved (second) state_dict.
    cfg = _cfg()
    model = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "r", cfg, model)
    assert art.maybe_save_best(1, model, 10.0) is True
    with torch.no_grad():
        model.prior_bank.mu_embed.add_(1.0)                     # make the second save distinguishable
    assert art.maybe_save_best(2, model, 8.0) is True           # improved -> replaces the existing file
    loaded = torch.load(tmp_path / "r" / "best_model.pt", weights_only=True)
    cur = model.state_dict()
    assert all(torch.equal(loaded[k], cur[k]) for k in cur)     # the SECOND state won


# --------------------------------------------------------------------------- pure-path report labels

def _report_cfg(**over):
    r"""SimpleNamespace with every attribute _pure_path_report reads (incl. family for the
    regime_ii_covariant exactness flag)."""
    base = dict(include_attention_entropy=True, transport_mode="flat", lambda_alpha_mode="constant",
                use_prior_bank=True, use_head_mixer=False,
                lambda_beta=1.0, precision_weighted_attention=False,
                gauge_transport="on", pos_rotation="none", rope_full_gauge=False, rope_on_value=True,
                lambda_gamma=0.0, s_e_step=False, family="gaussian_diagonal")
    base.update(over)
    return types.SimpleNamespace(**base)


def test_pure_path_report_regime_ii_covariant_exact_flag():
    # C5 (audit 2026-07-01): the diagonal cone is not closed under GL congruence, so a diagonal
    # regime_ii_covariant run is a CONTROLLED APPROXIMATION -- never reported as exact Route B.
    diag = _pure_path_report(_report_cfg(transport_mode="regime_ii_covariant",
                                         family="gaussian_diagonal"), [])
    assert diag["config_toggles"]["regime_ii_covariant_exact"] is False
    full = _pure_path_report(_report_cfg(transport_mode="regime_ii_covariant",
                                         family="gaussian_full"), [])
    assert full["config_toggles"]["regime_ii_covariant_exact"] is True
    flat = _pure_path_report(_report_cfg(), [])
    assert flat["config_toggles"]["regime_ii_covariant_exact"] is True


def test_pure_path_report_transport_covariance_class():
    # C7 (audit 2026-07-01): plain regime_ii's bilinear edge is gauge-FIXED; the report must never
    # group it with the covariant Route B. Every registered mode maps to a class; an unknown mode
    # falls through to its own name (no KeyError).
    expected = {"flat":                   "covariant (flat)",
                "regime_ii":              "gauge-fixed (non-covariant)",
                "regime_ii_covariant":    "covariant",
                "regime_ii_link":         "gauge-fixed",
                "regime_ii_link_charted": "covariant"}
    for mode, label in expected.items():
        rep = _pure_path_report(_report_cfg(transport_mode=mode), [])
        assert rep["config_toggles"]["transport_covariance_class"] == label, mode
    rep = _pure_path_report(_report_cfg(transport_mode="future_mode"), [])
    assert rep["config_toggles"]["transport_covariance_class"] == "future_mode"


def test_train_with_artifacts_writes_files(tmp_path):
    cfg = _cfg(checkpoint_interval=2)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "run", cfg, model, dataset="synthetic")
    train(model, _loader(), cfg, n_steps=4, eval_interval=2, val_loader=_loader(seed=1), artifacts=art)
    assert (tmp_path / "run" / "metrics.csv").exists()
    assert (tmp_path / "run" / "best_model.pt").exists()
    assert any((tmp_path / "run" / "checkpoints").glob("step_*.pt"))


def test_train_with_artifacts_writes_attention_pngs(tmp_path):
    # Per eval, one LOG-scaled attention/step_<N>_layer<l>_head<h>.png per (layer, head).
    cfg = _cfg(n_layers=2, prior_handoff_rho=0.5)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "run", cfg, model, dataset="synthetic")
    train(model, _loader(), cfg, n_steps=4, eval_interval=2, val_loader=_loader(seed=1), artifacts=art)
    L, H = cfg.n_layers, len(model.group.irrep_dims)
    expected = sorted(f"step_{s}_layer{l}_head{h}.png"
                      for s in (2, 4) for l in range(L) for h in range(H))
    pngs = sorted((tmp_path / "run" / "attention").glob("step_*.png"))
    assert [p.name for p in pngs] == expected
    assert all(p.stat().st_size > 0 for p in pngs)


def test_save_attention_maps_is_best_effort(tmp_path):
    # A viz/plotting error must be swallowed (logged, never raised) so it cannot kill a run.
    cfg = _cfg()
    art = RunArtifacts(tmp_path / "run", cfg, VFEModel(cfg))
    assert art.save_attention_maps(1, object()) is None         # bad maps -> None, no exception


def test_train_without_artifacts_writes_nothing(tmp_path):
    cfg = _cfg()
    torch.manual_seed(0)
    model = VFEModel(cfg)
    train(model, _loader(), cfg, n_steps=4, eval_interval=2, val_loader=_loader(seed=1))
    assert list(tmp_path.iterdir()) == []                     # no artifacts object -> no writes


def test_finalize_run_writes_test_results_and_figures(tmp_path):
    cfg = _cfg()
    torch.manual_seed(0)
    model = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "run", cfg, model, dataset="synthetic")
    losses = train(model, _loader(), cfg, n_steps=4, eval_interval=2,
                   val_loader=_loader(seed=1), artifacts=art)
    res = finalize_run(model, art, cfg, test_loader=_loader(seed=2), losses=losses)
    assert "test_ppl" in res and math.isfinite(res["test_ppl"])
    assert (tmp_path / "run" / "test_results.json").exists()
    assert (tmp_path / "run" / "summary.json").exists()
    assert (tmp_path / "run" / "loss_curve.png").exists()
    assert (tmp_path / "run" / "val_ppl.png").exists()
    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert "test_ppl" in summary and "best_val_ppl" in summary
    assert "reloaded_best" in summary   # m26: surface whether best_model.pt was reloaded (cross-dir resume honesty)


def test_metrics_csv_includes_gauge_geometry_columns(tmp_path):
    # Part 1 (diagnostics tier): the curvature/gauge probes (holonomy deviation + gauge trace
    # spread) must be surfaced in the per-eval CSV, not only the free-energy terms.
    cfg = _cfg()
    torch.manual_seed(0)
    model = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "run", cfg, model)
    train(model, _loader(), cfg, n_steps=4, eval_interval=2, val_loader=_loader(seed=1), artifacts=art)
    header = (tmp_path / "run" / "metrics.csv").read_text().splitlines()[0]
    assert "holonomy_deviation" in header
    assert "gauge_trace_spread" in header


def test_finalize_writes_gauge_geometry_figure(tmp_path):
    cfg = _cfg()
    torch.manual_seed(0)
    model = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "run", cfg, model)
    losses = train(model, _loader(), cfg, n_steps=4, eval_interval=2,
                   val_loader=_loader(seed=1), artifacts=art)
    finalize_run(model, art, cfg, test_loader=_loader(seed=2), losses=losses)
    assert (tmp_path / "run" / "holonomy.png").exists()


def test_finalize_reloads_best_checkpoint(tmp_path):
    # finalize must report the TEST metric on the reloaded best-val checkpoint, not the final
    # (possibly worse) live weights. Pin the reload happened.
    cfg = _cfg()
    torch.manual_seed(0)
    model = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "run", cfg, model, dataset="synthetic")
    losses = train(model, _loader(), cfg, n_steps=4, eval_interval=2,
                   val_loader=_loader(seed=1), artifacts=art)
    res = finalize_run(model, art, cfg, test_loader=_loader(seed=2), losses=losses)
    assert res["reloaded_best"] is True
