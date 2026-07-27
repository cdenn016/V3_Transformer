r"""A failed periodic checkpoint write must not destroy the training run (2026-07-27).

Observed: a wikitext-103 K=210 run died at step 120000 of 180000 -- 10 hours of GPU work --
because ``torch.save`` inside ``RunArtifacts.save_checkpoint`` hit a transient stream failure
(``ios_base::badbit set: iostream stream error``, then the ``unexpected pos`` cascade raised by
``PyTorchStreamWriter.__exit__`` during unwind). The write is environmental and non-reproducible
(the same 3 GB payload rewrites fine into the same directory), so the defect that matters is
ours: ``train`` called ``save_checkpoint`` unguarded while every other periodic side effect in
that block -- validation diagnostics, attention-map replay, figures -- is deliberately
best-effort so that "a viz/replay fault never kills training".

A checkpoint is a convenience for resuming, not a correctness requirement of the step loop, so
losing one must cost one checkpoint, not the whole run.

The leak leg is the same incident's second half: the orphaned ``.step_120000.pt.<tag>.tmp``
survived at 2.48 GB because ``_unique_sibling_temp`` gets exactly one ``unlink`` attempt and
swallows its ``OSError``. Windows still held the failed writer's handle at unwind time (the file
was freely deletable minutes later), so the single attempt lost a race it would have won on a
retry. That is load-bearing now: once a failed save is non-fatal, an unretried cleanup leaks
multiple GB per checkpoint interval and eventually fills the disk the run is still writing to.
"""

import logging
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from vfe3.config import VFE3Config
from vfe3.data.datasets import TokenWindows
from vfe3.model.model import VFEModel
from vfe3.run_artifacts import RunArtifacts
from vfe3.train import build_optimizer, train


def _const_loader(seq_len: int = 8, bs: int = 4) -> DataLoader:
    base = torch.full((seq_len * 6,), 1, dtype=torch.long)
    return DataLoader(TokenWindows(base, seq_len), batch_size=bs, shuffle=False, drop_last=True)


def _cfg(**kw) -> VFE3Config:
    base = dict(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                n_e_steps=1, e_q_mu_lr=0.1, e_phi_lr=0.0, m_phi_lr=0.0,
                warmup_steps=1, max_steps=4)
    base.update(kw)
    return VFE3Config(**base)


def test_periodic_checkpoint_write_failure_does_not_kill_training(tmp_path, monkeypatch, caplog):
    r"""The whole point: a raising ``save_checkpoint`` costs the checkpoint, not the run."""
    cfg = _cfg(checkpoint_interval=2, max_steps=4)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    artifacts = RunArtifacts(tmp_path / "run", cfg, model)

    attempts = []

    def _exploding_save(self, step, *args, **kwargs):
        attempts.append(int(step))
        raise RuntimeError("ios_base::badbit set: iostream stream error")

    monkeypatch.setattr(RunArtifacts, "save_checkpoint", _exploding_save)

    with caplog.at_level(logging.WARNING):
        losses = train(model, _const_loader(), cfg, n_steps=4, artifacts=artifacts)

    assert len(losses) == 4                       # every step ran; the run was NOT aborted
    assert attempts == [2, 4]                     # and each interval genuinely attempted a save
    assert any("checkpoint" in record.message.lower() for record in caplog.records), \
        "a swallowed checkpoint failure must still be reported loudly"


def test_failed_checkpoint_save_does_not_leak_its_temp_file(tmp_path, monkeypatch):
    r"""A transiently-locked temp must still be cleaned up, not abandoned at multi-GB size."""
    cfg = _cfg()
    torch.manual_seed(0)
    model = VFEModel(cfg)
    optimizer = build_optimizer(model, cfg)
    artifacts = RunArtifacts(tmp_path / "run", cfg, model)

    real_unlink = Path.unlink
    calls = {"n": 0}

    def _transiently_locked_unlink(self, *args, **kwargs):
        # Mimic Windows holding the failed writer's handle for the first attempt only.
        if self.name.endswith(".tmp"):
            calls["n"] += 1
            if calls["n"] == 1:
                raise PermissionError(32, "The process cannot access the file")
        return real_unlink(self, *args, **kwargs)

    def _failing_torch_save(obj, path, *args, **kwargs):
        Path(path).write_bytes(b"\0" * 4096)      # partial write, as a real failure leaves behind
        raise RuntimeError("ios_base::badbit set: iostream stream error")

    monkeypatch.setattr(Path, "unlink", _transiently_locked_unlink)
    monkeypatch.setattr("vfe3.run_artifacts.torch.save", _failing_torch_save)

    with pytest.raises(RuntimeError):
        artifacts.save_checkpoint(4, model, optimizer, cfg)

    leaked = list((tmp_path / "run" / "checkpoints").glob("*.tmp"))
    assert leaked == [], f"failed save leaked its temp file: {[p.name for p in leaked]}"
