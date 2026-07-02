r"""Round-3 artifact-hygiene fixes (audit 2026-07-01 round-3).

Pins three raising-path behaviors: ``_atomic_replace`` deletes its orphaned same-directory tmp
when the publish ultimately fails (while the success path still consumes it), the ``_emit``
figure driver closes a pyplot figure a thunk registered before raising (tight_layout/savefig),
and the sigma-gate artifact writer disambiguates lossy checkpoint-id slugs with a stable hash
so two distinct ids never overwrite each other's PASS/FAIL record.
"""

import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from vfe3 import run_artifacts
from vfe3.config import VFE3Config
from vfe3.data.datasets import TokenWindows
from vfe3.inference.sigma_gate import write_sigma_gate_artifact
from vfe3.model.model import VFEModel
from vfe3.run_artifacts import _atomic_replace


# ---------------------------------------------------------------- _atomic_replace (punch item 9)

def test_atomic_replace_normal_path_consumes_tmp(tmp_path):
    final = tmp_path / "artifact.json"
    tmp   = tmp_path / "artifact.json.tmp"
    tmp.write_text('{"ok": true}')
    _atomic_replace(final, tmp)
    assert final.exists() and final.read_text() == '{"ok": true}'
    assert not tmp.exists()                                    # consumed by the rename, not orphaned


def test_atomic_replace_cleans_tmp_on_final_permission_error(tmp_path, monkeypatch):
    final = tmp_path / "artifact.json"
    tmp   = tmp_path / "artifact.json.tmp"
    tmp.write_text("{}")

    def _deny(src, dst):
        raise PermissionError("simulated Windows open-handle lock")

    monkeypatch.setattr(run_artifacts.os, "replace", _deny)
    with pytest.raises(PermissionError):
        _atomic_replace(final, tmp, delay=0.0, retries=3)
    assert not tmp.exists()                                    # orphan removed on the raising path
    assert not final.exists()


def test_atomic_replace_cleans_tmp_on_unexpected_error(tmp_path, monkeypatch):
    final = tmp_path / "artifact.json"
    tmp   = tmp_path / "artifact.json.tmp"
    tmp.write_text("{}")

    def _boom(src, dst):
        raise ValueError("simulated non-PermissionError failure")

    monkeypatch.setattr(run_artifacts.os, "replace", _boom)
    with pytest.raises(ValueError):                            # re-raised, never swallowed
        _atomic_replace(final, tmp, delay=0.0, retries=3)
    assert not tmp.exists()
    assert not final.exists()


# ---------------------------------------------------------------- report._emit figure leak (punch item 8)

def _loader(seed=0, n=600, seq_len=8, bs=8):
    g = torch.Generator().manual_seed(seed)
    base = torch.arange(3).repeat(n // 3 + 2)                  # period-3 stream over {0,1,2}
    ds = TokenWindows(base[:n].long(), seq_len)
    return DataLoader(ds, batch_size=bs, shuffle=False, drop_last=True, generator=g)


def _model(**kw):
    base = dict(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                n_e_steps=2, e_q_mu_lr=0.1, e_phi_lr=0.05)
    base.update(kw)
    torch.manual_seed(0)
    return VFEModel(VFE3Config(**base))


def test_emit_closes_figure_registered_by_raising_thunk(tmp_path, monkeypatch):
    # A thunk can create a pyplot-managed figure and then raise (tight_layout/savefig) before _emit
    # ever receives it; the except path must close everything the thunk registered.
    from vfe3.viz import figures as figs
    from vfe3.viz.report import generate_figures

    def _leaky_plot(*args, **kwargs):
        figs.plt.figure()                                      # registered but never returned
        raise RuntimeError("simulated savefig failure")

    monkeypatch.setattr(figs, "plot_estep_convergence", _leaky_plot)
    before = set(figs.plt.get_fignums())
    generate_figures(tmp_path / "run", model=_model(), loader=_loader(), max_sequences=16)
    assert set(figs.plt.get_fignums()) == before               # no leaked registry entries
    assert not (tmp_path / "run" / "figures" / "estep_convergence.png").exists()


# ---------------------------------------------------------------- sigma-gate slug collision (punch item 7)

def test_sigma_gate_colliding_checkpoint_ids_write_distinct_files(tmp_path):
    # The slug is lossy ("ckpt a" / "ckpt:a" both map to "ckpt_a"); the raw-id hash suffix keeps
    # the two records distinct instead of the second silently overwriting the first (C15 pattern).
    p1 = write_sigma_gate_artifact({"status": "PASS"}, checkpoint_id="ckpt a",
                                   spec_commit="x", seeds=(6,), out_dir=str(tmp_path))
    p2 = write_sigma_gate_artifact({"status": "FAIL"}, checkpoint_id="ckpt:a",
                                   spec_commit="x", seeds=(6,), out_dir=str(tmp_path))
    assert p1 != p2
    assert Path(p1).exists() and Path(p2).exists()
    r1 = json.loads(Path(p1).read_text(encoding="utf-8"))
    r2 = json.loads(Path(p2).read_text(encoding="utf-8"))
    assert r1["checkpoint_id"] == "ckpt a" and r1["status"] == "PASS"
    assert r2["checkpoint_id"] == "ckpt:a" and r2["status"] == "FAIL"
