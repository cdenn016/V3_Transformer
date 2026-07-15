"""Corpus-pass coordinate and loss-boundary regressions."""

import csv

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel
from vfe3.run_artifacts import RunArtifacts
from vfe3.train import _training_cursor_fields, train
from vfe3.viz.figures import plot_trajectory


def test_training_cursor_fields_are_one_based_and_continuous() -> None:
    rows = [_training_cursor_fields(step, 3) for step in range(1, 6)]
    assert [(r["epoch"], r["batch_in_epoch"]) for r in rows] == [
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 1),
        (2, 2),
    ]
    assert all(r["steps_per_epoch"] == 3 for r in rows)
    assert rows[-1]["corpus_pass"] == pytest.approx(5.0 / 3.0)


def test_training_rows_persist_epoch_coordinates(tmp_path) -> None:
    cfg = VFE3Config(
        vocab_size=8,
        embed_dim=4,
        n_heads=1,
        max_seq_len=4,
        n_layers=1,
        n_e_steps=1,
        e_phi_lr=0.0,
        pos_phi="none",
        warmup_steps=1,
        max_steps=5,
    )
    tokens = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]])
    targets = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]])
    loader = DataLoader(TensorDataset(tokens, targets), batch_size=1, shuffle=False)
    model = VFEModel(cfg)
    artifacts = RunArtifacts(tmp_path / "run", cfg, model)

    train(model, loader, cfg, n_steps=5, log_interval=1, artifacts=artifacts)

    with open(tmp_path / "run" / "metrics.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [(int(r["epoch"]), int(r["batch_in_epoch"])) for r in rows] == [
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 1),
        (2, 2),
    ]
    assert [int(r["steps_per_epoch"]) for r in rows] == [3] * 5
    assert float(rows[-1]["corpus_pass"]) == pytest.approx(5.0 / 3.0)


def test_loss_trajectory_draws_complete_corpus_pass_boundaries() -> None:
    fig = plot_trajectory(
        [3.0, 2.0, 1.0, 0.5],
        steps=[1, 2, 3, 4],
        epoch_boundaries=[2, 4],
    )
    vertical = []
    for line in fig.axes[0].lines:
        xdata = list(line.get_xdata())
        if len(xdata) == 2 and xdata[0] == xdata[1]:
            vertical.append(float(xdata[0]))
    assert vertical == [2.0, 4.0]
