r"""Single-run publication-figure driver (vfe3.viz.report) + the converged_state extractor.

These pin the WIRING the user found missing: the figure generators and extract runners existed
and were unit-tested in isolation, but nothing drove them end-to-end against a real model, so a
trained run produced only one of the publication figures. The proof is PNG files on disk, so the
integration test asserts the figure set actually appears when the driver runs the real model.
"""

import torch
from torch.utils.data import DataLoader

from vfe3.config import VFE3Config
from vfe3.data.datasets import TokenWindows
from vfe3.model.model import VFEModel
from vfe3.run_artifacts import RunArtifacts, finalize_run
from vfe3.train import train
from vfe3.viz.extract import converged_state
from vfe3.viz.report import generate_figures


def _loader(seed=0, n=600, seq_len=8, bs=8):
    g = torch.Generator().manual_seed(seed)
    base = torch.arange(3).repeat(n // 3 + 2)                  # period-3 stream over {0,1,2}
    ds = TokenWindows(base[:n].long(), seq_len)
    return DataLoader(ds, batch_size=bs, shuffle=False, drop_last=True, generator=g)


def _cfg(**kw):
    base = dict(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                n_e_steps=2, e_q_mu_lr=0.1, e_phi_lr=0.05)
    base.update(kw)
    return VFE3Config(**base)


def _model(**kw):
    torch.manual_seed(0)
    return VFEModel(_cfg(**kw))


def test_converged_state_shapes_and_finite():
    model = _model(n_layers=2)
    tok = torch.randint(0, 6, (2, 8))                          # only seq 0 is used
    st = converged_state(model, tok)
    n, k = 8, 4
    assert st["mu"].shape == (n, k)
    assert st["phi"].shape[0] == n
    assert st["exp_phi"].shape == (n, k, k)
    assert st["omega"].shape == (n, n, k, k)
    assert st["energy"].shape[-2:] == (n, n)
    assert st["beta"].shape[-2:] == (n, n)
    assert st["self_div"].shape[0] == n
    for key in ("mu", "sigma", "phi", "exp_phi", "omega", "energy", "beta", "self_div"):
        assert torch.isfinite(st[key]).all(), key


def test_generate_figures_drives_live_model(tmp_path):
    # The driver against a live in-memory model writes the single-run figure set to figures/.
    model = _model()
    paths = generate_figures(tmp_path / "run", model=model, loader=_loader(), max_sequences=16)
    figdir = tmp_path / "run" / "figures"
    written = {p.name for p in paths}
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)
    # The figures that need no optional dependency (UMAP is best-effort, so belief_umap is excluded).
    robust = {"estep_convergence.png", "belief_trajectories.png", "attention_structure.png",
              "gauge_equivariance.png", "gauge_head_specialization.png", "belief_spectrum.png",
              "spd_ellipses.png", "holonomy_curvature.png", "numerical_trust.png",
              "belief_category_separation.png",
              # vocab next-token figures (decoder-free; default use_prior_bank=False -> decode_readout
              # present; vocab_confusion needs the optional tokenizer so it is excluded like belief_umap).
              "vocab_probability_heatmap.png", "vocab_calibration.png", "decode_readout.png"}
    missing = robust - written
    assert not missing, f"driver did not produce {missing}"
    assert all((figdir / name).exists() for name in robust)


def test_generate_figures_reloads_from_run_dir(tmp_path):
    # The reload path: config.json + best_model.pt -> rebuilt model -> figures, no live handle.
    cfg = _cfg()
    model = _model()
    art = RunArtifacts(tmp_path / "run", cfg, model, dataset="synthetic-period3")   # writes config.json
    torch.save(model.state_dict(), art.best_path)                                   # the reloaded weights
    paths = generate_figures(tmp_path / "run", loader=_loader(), max_sequences=16)
    assert len(paths) >= 6
    assert (tmp_path / "run" / "figures" / "numerical_trust.png").exists()


def test_finalize_autoruns_figures(tmp_path):
    # generate_figures defaults True -> finalize_run auto-writes run_dir/figures/ (the autorun).
    cfg = _cfg()
    model = _model()
    art = RunArtifacts(tmp_path / "run", cfg, model, dataset="synthetic-period3")
    losses = train(model, _loader(), cfg, n_steps=4, log_interval=2, eval_interval=2,
                   val_loader=_loader(seed=1), artifacts=art)
    finalize_run(model, art, cfg, test_loader=_loader(seed=2), losses=losses)
    figs = list((tmp_path / "run" / "figures").glob("*.png"))
    names = {f.name for f in figs}
    assert len(figs) >= 6, f"autorun produced too few figures: {[f.name for f in figs]}"
    # The vocabulary next-token figures are part of the finalize_run autorun (i.e. produced by
    # train_vfe3.py): default use_prior_bank=False -> decode_readout present; the synthetic-period3
    # dataset has no tokenizer so vocab_confusion is skipped.
    assert {"vocab_probability_heatmap.png", "vocab_calibration.png", "decode_readout.png"} <= names


def test_finalize_skips_figures_when_disabled(tmp_path):
    # generate_figures=False is the opt-out: finalize_run writes no figures/ directory.
    cfg = _cfg(generate_figures=False)
    model = _model()
    art = RunArtifacts(tmp_path / "run", cfg, model, dataset="synthetic-period3")
    losses = train(model, _loader(), cfg, n_steps=4, log_interval=2, eval_interval=2,
                   val_loader=_loader(seed=1), artifacts=art)
    finalize_run(model, art, cfg, test_loader=_loader(seed=2), losses=losses)
    assert not (tmp_path / "run" / "figures").exists()


def test_metrics_csv_logs_at_log_cadence(tmp_path):
    # metrics.csv gets a row every log_interval (denser than eval_interval), but the validation
    # columns are EVAL-CADENCE: a value only on an eval step, a BLANK cell on the log-interval rows
    # in between (NOT carried forward) -- matching VFE_2.0's metrics.csv.
    import csv
    import math
    cfg = _cfg()
    model = _model()
    art = RunArtifacts(tmp_path / "run", cfg, model, dataset="synthetic-period3")
    train(model, _loader(), cfg, n_steps=8, log_interval=2, eval_interval=4,
          val_loader=_loader(seed=1), artifacts=art)
    rows = list(csv.DictReader(open(tmp_path / "run" / "metrics.csv")))
    assert [r["step"] for r in rows] == ["2", "4", "6", "8"]          # a row every log_interval
    assert rows[0]["val_ce"] == ""                                    # blank before the first eval (step 4)
    assert math.isfinite(float(rows[1]["val_ce"]))                    # fresh val at the step-4 eval
    assert rows[2]["val_ce"] == ""                                    # blank between evals (NOT carried forward)
    assert math.isfinite(float(rows[3]["val_ce"]))                    # fresh again at the step-8 eval


def test_s_channel_refinement_extractor_present_iff_s_e_step():
    # s_e_step=True replays encode_s -> _refine_s and returns the per-position refinement diagnostics;
    # s_e_step=False (the model channel never runs) returns None so the figure is skipped downstream.
    from vfe3.viz.extract import s_channel_refinement
    tok = torch.randint(0, 6, (2, 8))
    on = _model(s_e_step=True, prior_source="model_channel", lambda_h=0.25, lambda_gamma=0.75)
    d = s_channel_refinement(on, tok)
    assert d is not None and set(d) == {"mu_delta", "logsigma_delta", "kl_s0_r", "kl_s1_r"}
    for key, v in d.items():
        assert v.shape == (8,) and torch.isfinite(v).all(), key
    off = _model(s_e_step=False)
    assert s_channel_refinement(off, tok) is None


def test_generate_figures_emits_s_channel_under_s_e_step(tmp_path):
    # The s-channel figure is written iff s_e_step=True (guarded by the None extractor).
    on = _model(s_e_step=True, prior_source="model_channel", lambda_h=0.25, lambda_gamma=0.75)
    written = {p.name for p in generate_figures(tmp_path / "on", model=on, loader=_loader(), max_sequences=16)}
    assert "s_channel_refinement.png" in written
    off = _model(s_e_step=False)
    written_off = {p.name for p in generate_figures(tmp_path / "off", model=off, loader=_loader(), max_sequences=16)}
    assert "s_channel_refinement.png" not in written_off
