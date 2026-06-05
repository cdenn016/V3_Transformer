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
from vfe3.run_artifacts import RunArtifacts
from vfe3.viz.extract import converged_state
from vfe3.viz.report import generate_figures


def _loader(seed=0, n=600, seq_len=8, bs=8):
    g = torch.Generator().manual_seed(seed)
    base = torch.arange(3).repeat(n // 3 + 2)                  # period-3 stream over {0,1,2}
    ds = TokenWindows(base[:n].long(), seq_len)
    return DataLoader(ds, batch_size=bs, shuffle=False, drop_last=True, generator=g)


def _cfg(**kw):
    base = dict(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                n_e_steps=2, e_mu_lr=0.1, e_phi_lr=0.05)
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
              "spd_ellipses.png", "holonomy_curvature.png", "numerical_trust.png"}
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
