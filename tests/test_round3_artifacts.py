r"""Round-3 artifact-hygiene fixes (audit 2026-07-01 round-3).

Pins two raising-path behaviors: ``_atomic_replace`` deletes its orphaned same-directory tmp
when the publish ultimately fails (while the success path still consumes it), the ``_emit``
figure driver closes a pyplot figure a thunk registered before raising (tight_layout/savefig).
"""

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from vfe3 import run_artifacts
from vfe3.config import VFE3Config
from vfe3.data.datasets import TokenWindows
from vfe3.model.model import VFEModel
from vfe3.run_artifacts import _atomic_replace
from vfe3.train import _loader_data_identity


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


@pytest.mark.parametrize("failure_mode", ["thunk", "save"])
def test_comparison_emit_closes_figures_after_thunk_or_save_failure(
    tmp_path,
    monkeypatch,
    failure_mode,
):
    from vfe3.viz import figures as figs
    from vfe3.viz import report

    model = _model()
    art = run_artifacts.RunArtifacts(
        tmp_path / "run",
        model.cfg,
        model,
        dataset="synthetic-period3",
    )
    selection_data_identity = _loader_data_identity(_loader(), model.cfg.vocab_size)
    art.bind_selection_data_identity(selection_data_identity)
    art.maybe_save_best(1, model, 1.0)
    art.save_json("provenance.json", {
        "code_identity_sha256":    art.code_identity_sha256,
        "selection_data_identity": selection_data_identity,
    })
    monkeypatch.setattr(report, "_build_loader", lambda *args, **kwargs: _loader())
    monkeypatch.setattr(report.extract, "vocab_prediction_stats", lambda *args, **kwargs: {})
    monkeypatch.setattr(report.extract, "decode_readout", lambda *args, **kwargs: None)

    def _failing_plot(*args, **kwargs):
        fig = figs.plt.figure()
        if failure_mode == "thunk":
            raise RuntimeError("simulated comparison thunk failure")

        def _fail_save(*save_args, **save_kwargs):
            Path(save_args[0]).write_bytes(b"PARTIAL")
            raise RuntimeError("simulated comparison save failure")

        fig.savefig = _fail_save
        return fig

    monkeypatch.setattr(figs, "plot_vocab_probability_heatmap", _failing_plot)
    before = set(figs.plt.get_fignums())
    comparison_dir = tmp_path / "comparison"
    comparison_dir.mkdir()
    target = comparison_dir / "vocab_probability_heatmap_compare.png"
    target.write_bytes(b"SENTINEL")

    report.vocab_comparison_figures([art.run_dir], comparison_dir)

    assert set(figs.plt.get_fignums()) == before
    assert target.read_bytes() == b"SENTINEL"
    assert not list(comparison_dir.glob(".vocab_probability_heatmap_compare.*.tmp.png"))
