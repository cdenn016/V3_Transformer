r"""Model-channel (s / r / h) and gamma_ij tracking + figures (2026-06-13).

The model-channel free-energy blocks -- the hyper-prior lambda_h KL(s_i||r), the gamma model-coupling
sum_ij gamma_ij KL(s_i||Omega s_j), and its meta-entropy tau_g sum_ij gamma_ij log(gamma_ij/pi^s_ij) --
were either invisible (gated on ``not s_e_step`` in diagnostics) or fused into a single envelope, and
had no figures. These tests pin:

  (1) diagnostics surfaces the hyper-prior + gamma blocks (gamma SPLIT into coupling vs meta-entropy)
      whenever the model channel is active, INCLUDING under s_e_step=True (previously invisible);
  (2) the gamma split satisfies the envelope identity coupling + meta_entropy == total, and the loss
      term's mean reduction is consistent with the sum-scale diagnostic;
  (3) the reduction-consistent total assembly (the per-token MEAN-into-per-sequence-SUM bug, obs 18497):
      d["total"] is the weighted sum of every reported block at one scale;
  (4) _hyper_prior_term is the mean of the _hyper_prior_kl vector (byte-identical refactor);
  (5) gamma_attention_maps shape + gating;
  (6) the s/r/h/gamma extractors gate correctly (None when their tables are absent);
  (7) every new figure renders (the s/r/h + gamma publication figures and the root model_channel_terms);
  (8) metrics.csv carries the model-channel columns under an active channel and NONE on the pure path;
  (9) generate_figures emits the four publication figures under an active channel, none on the pure path;
 (10) finalize_run's root-folder _save_figures emits model_channel_terms.png iff the channel is active.
"""

import csv
import math

import torch
from torch.utils.data import DataLoader

from vfe3.config import VFE3Config
from vfe3.data.datasets import TokenWindows
from vfe3.model.model import VFEModel
from vfe3.run_artifacts import RunArtifacts, finalize_run
from vfe3.train import train
from vfe3.viz import extract, figures as figs
from vfe3.viz.report import generate_figures


def _loader(seed: int = 0, n: int = 600, seq_len: int = 8, bs: int = 8) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    base = torch.arange(3).repeat(n // 3 + 2)                      # period-3 stream over {0,1,2}
    ds = TokenWindows(base[:n].long(), seq_len)
    return DataLoader(ds, batch_size=bs, shuffle=False, drop_last=True, generator=g)


def _cfg(**kw) -> VFE3Config:
    base = dict(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                n_e_steps=2, e_mu_lr=0.1, e_phi_lr=0.05)
    base.update(kw)
    return VFE3Config(**base)


def _model(**kw) -> VFEModel:
    torch.manual_seed(0)
    return VFEModel(_cfg(**kw))


def _active(**kw) -> VFEModel:
    r"""A model with the full model channel live (s_e_step + hyper-prior + gamma), distinct s tables."""
    m = _model(s_e_step=True, prior_source="model_channel", lambda_h=0.25, gamma_coupling=0.75, **kw)
    torch.manual_seed(123)
    with torch.no_grad():                                          # make the blocks robustly non-vacuous
        m.prior_bank.s_mu_embed.normal_(0.0, 0.5)
        m.prior_bank.s_sigma_log_embed.normal_(0.0, 0.3)
        m.prior_bank.phi_embed.normal_(0.0, 0.2)
    return m


# ---- (1) diagnostics surfaces the blocks under s_e_step (the regression the user hit) --------------

def test_diagnostics_surfaces_model_channel_blocks_under_s_e_step():
    tok = torch.randint(0, 6, (2, 8))
    d = _active().diagnostics(tok)
    for k in ("hyper_prior", "gamma_coupling", "gamma_meta_entropy"):
        assert k in d, f"{k} missing under s_e_step (was gated on `not s_e_step`)"
        assert math.isfinite(d[k])
    # the pure belief path carries none of them
    d0 = _model().diagnostics(tok)
    assert not ({"hyper_prior", "gamma_coupling", "gamma_meta_entropy"} & set(d0))


# ---- (2) gamma split: envelope identity + mean/sum reduction consistency ---------------------------

def test_gamma_split_envelope_identity_and_reduction():
    m = _active()
    tok = torch.randint(0, 6, (1, 8))
    with torch.no_grad():                                         # the diagnostic helpers run under no_grad
        phi = m.prior_bank.encode(tok).phi
        g = m._gamma_coupling_terms(tok, phi)
        c, me, tot = float(g["coupling"]), float(g["meta_entropy"]), float(g["total"])
        # the loss term is the per-(B,H,N) MEAN of the same reduced free energy; total is its SUM
        loss_term = float(m._gamma_coupling_term(tok, phi))
    assert abs((c + me) - tot) < 1e-4                             # envelope: coupling + meta == -tau log Z
    H = len(m.group.irrep_dims)
    assert abs(loss_term - tot / (1 * H * 8)) < 1e-5


# ---- (3) reduction-consistent total assembly (obs 18497) ------------------------------------------

def test_diagnostics_total_is_consistent_weighted_sum():
    m = _active()
    cfg = m.cfg
    tok = torch.randint(0, 6, (2, 8))
    d = m.diagnostics(tok)
    lb = cfg.lambda_beta                                           # 1.0 on the pure path
    expected = (d["self_coupling"] + lb * d["belief_coupling"] + lb * d["attention_entropy"]
                + cfg.lambda_h * d["hyper_prior"]
                + cfg.gamma_coupling * (d["gamma_coupling"] + d["gamma_meta_entropy"]))
    assert abs(d["total"] - expected) < 1e-3                       # every block folded at ONE (sum) scale


# ---- (4) hyper-prior vector / term consistency (byte-identical refactor) --------------------------

def test_hyper_prior_term_is_mean_of_kl_vector():
    m = _active()
    tok = torch.randint(0, 6, (2, 8))
    kl = m._hyper_prior_kl(tok)                                    # (B, N)
    assert kl.shape == (2, 8) and torch.isfinite(kl).all()
    assert torch.equal(m._hyper_prior_term(tok), kl.mean())       # exact: term IS the mean


# ---- (5) gamma_attention_maps shape + gating ------------------------------------------------------

def test_gamma_attention_maps_shape_and_gating():
    tok = torch.randint(0, 6, (2, 8))
    m = _active()
    g = m.gamma_attention_maps(tok)
    assert g is not None and g.shape == (len(m.group.irrep_dims), 8, 8)
    assert torch.isfinite(g).all()
    # rows are softmax over the causal key set -> sum to 1 within the active set
    assert torch.allclose(g.sum(dim=-1), torch.ones_like(g.sum(dim=-1)), atol=1e-4)
    assert _model().gamma_attention_maps(tok) is None             # pure path: no s tables


# ---- (6) extractor gating (None when the table is absent) -----------------------------------------

def test_model_channel_extractors_gate_on_their_tables():
    tok = torch.randint(0, 6, (2, 8))
    pure = _model()
    assert extract.model_channel_belief(pure, tok) is None
    assert extract.hyper_prior_centroid(pure, tok) is None
    assert extract.hyper_prior_coupling(pure, tok) is None
    assert extract.gamma_attention(pure, tok) is None
    # model_channel prior only (no r): s + gamma present, r-based extractors None
    mc = _model(prior_source="model_channel")
    assert extract.model_channel_belief(mc, tok) is not None
    assert extract.gamma_attention(mc, tok) is not None
    assert extract.hyper_prior_centroid(mc, tok) is None          # r created only on lambda_h>0 / s_e_step
    assert extract.hyper_prior_coupling(mc, tok) is None
    # full channel: all four present and well-shaped
    m = _active()
    sb = extract.model_channel_belief(m, tok)
    assert set(sb) == {"mu_mean", "mu_std", "sigma_mean", "spectrum", "eff_rank"}
    assert sb["spectrum"].shape == (8, 4) and sb["eff_rank"].shape == (8,)
    rc = extract.hyper_prior_centroid(m, tok)
    assert set(rc) == {"r_mu", "r_sigma", "s_mu_mean", "s_mu_std", "s_sigma_mean"}
    assert rc["r_mu"].shape == (4,)
    hc = extract.hyper_prior_coupling(m, tok)
    assert hc["kl_s_r"].shape == (8,) and torch.isfinite(hc["kl_s_r"]).all()


# ---- (7) every new figure renders ----------------------------------------------------------------

def test_model_channel_figures_render(tmp_path):
    figs.set_publication_style()
    tok = torch.randint(0, 6, (2, 8))
    m = _active()
    cases = [
        ("s.png",     figs.plot_model_channel_belief,  extract.model_channel_belief(m, tok)),
        ("r.png",     figs.plot_hyper_prior_centroid,  extract.hyper_prior_centroid(m, tok)),
        ("h.png",     figs.plot_hyper_prior_coupling,  extract.hyper_prior_coupling(m, tok)),
        ("gamma.png", figs.plot_gamma_attention,       extract.gamma_attention(m, tok)),
    ]
    for name, fn, data in cases:
        p = tmp_path / name
        fig = fn(data, path=str(p))
        figs.plt.close(fig)
        assert p.exists() and p.stat().st_size > 0, name


def test_model_channel_terms_figure_renders(tmp_path):
    figs.set_publication_style()
    n = 30
    hist = {"step": list(range(1, n + 1)),
            "hyper_prior":        [0.5 + 0.01 * i for i in range(n)],
            "gamma_coupling":     [0.3 + 0.005 * i for i in range(n)],
            "gamma_meta_entropy": [max(0.0, 0.2 - 0.003 * i) for i in range(n)]}
    p = tmp_path / "mct.png"
    figs.plt.close(figs.plot_model_channel_terms(hist, path=str(p)))
    assert p.exists() and p.stat().st_size > 0
    # partial keys (only gamma logged) must also render
    p2 = tmp_path / "mct2.png"
    figs.plt.close(figs.plot_model_channel_terms(
        {"step": list(range(1, n + 1)), "gamma_coupling": [0.3] * n}, path=str(p2)))
    assert p2.exists() and p2.stat().st_size > 0


# ---- (8) metrics.csv columns under an active channel, NONE on the pure path -----------------------

def test_metrics_csv_has_model_channel_columns(tmp_path):
    cfg = _cfg(s_e_step=True, prior_source="model_channel", lambda_h=0.25, gamma_coupling=0.75)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "run", cfg, model, dataset="synthetic-period3")
    train(model, _loader(), cfg, n_steps=4, log_interval=2, eval_interval=2,
          val_loader=_loader(seed=1), artifacts=art)
    rows = list(csv.DictReader(open(tmp_path / "run" / "metrics.csv")))
    for col in ("hyper_prior", "gamma_coupling", "gamma_meta_entropy"):
        assert col in rows[0], f"{col} column missing"
        assert all(math.isfinite(float(r[col])) for r in rows)


def test_metrics_csv_pure_path_has_no_model_channel_columns(tmp_path):
    cfg = _cfg()
    torch.manual_seed(0)
    model = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "run", cfg, model, dataset="synthetic-period3")
    train(model, _loader(), cfg, n_steps=4, log_interval=2, eval_interval=2,
          val_loader=_loader(seed=1), artifacts=art)
    rows = list(csv.DictReader(open(tmp_path / "run" / "metrics.csv")))
    assert not ({"hyper_prior", "gamma_coupling", "gamma_meta_entropy"} & set(rows[0]))


# ---- (9) generate_figures emits the publication figures under an active channel -------------------

def test_generate_figures_emits_model_channel_figures(tmp_path):
    on = _active()
    written = {p.name for p in generate_figures(tmp_path / "on", model=on, loader=_loader(), max_sequences=16)}
    for name in ("model_channel_belief.png", "hyper_prior_centroid.png",
                 "hyper_prior_coupling.png", "gamma_attention.png"):
        assert name in written, name
    off = _model()
    written_off = {p.name for p in generate_figures(tmp_path / "off", model=off, loader=_loader(), max_sequences=16)}
    assert not ({"model_channel_belief.png", "hyper_prior_centroid.png",
                 "hyper_prior_coupling.png", "gamma_attention.png"} & written_off)


# ---- (10) finalize_run's root figures emit model_channel_terms.png iff active ---------------------

def test_finalize_emits_model_channel_terms_iff_active(tmp_path):
    cfg = _cfg(s_e_step=True, prior_source="model_channel", lambda_h=0.25, gamma_coupling=0.75)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "run", cfg, model, dataset="synthetic-period3")
    losses = train(model, _loader(), cfg, n_steps=4, log_interval=2, eval_interval=2,
                   val_loader=_loader(seed=1), artifacts=art)
    finalize_run(model, art, cfg, test_loader=_loader(seed=2), losses=losses)
    assert (tmp_path / "run" / "model_channel_terms.png").exists()

    cfg0 = _cfg()
    torch.manual_seed(0)
    model0 = VFEModel(cfg0)
    art0 = RunArtifacts(tmp_path / "run0", cfg0, model0, dataset="synthetic-period3")
    losses0 = train(model0, _loader(), cfg0, n_steps=4, log_interval=2, eval_interval=2,
                    val_loader=_loader(seed=1), artifacts=art0)
    finalize_run(model0, art0, cfg0, test_loader=_loader(seed=2), losses=losses0)
    assert not (tmp_path / "run0" / "model_channel_terms.png").exists()
