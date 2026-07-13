import json

import numpy as np
import pytest
import torch

import vfe3.viz.figures as figures
from vfe3.inference.sigma_gate import evaluate_sigma_gate, write_sigma_gate_artifact
from vfe3.geometry.retraction import retract_spd_diagonal, retract_spd_full
from vfe3.metrics import belief_spectrum


def test_belief_spectrum_surfaces_nonpositive_covariance() -> None:
    spectrum = belief_spectrum(torch.tensor([[4.0, 1.0, -0.25]]), diagonal=True)

    assert torch.equal(spectrum["eigenvalues"], torch.tensor([[4.0, 1.0, -0.25]]))
    assert torch.isinf(spectrum["condition"]).all()
    assert not bool(spectrum["is_positive_definite"].all())


def test_belief_spectrum_figure_omits_disabled_sigma_ceiling() -> None:
    figure = figures.plot_belief_spectrum(torch.ones(3, 2), sigma_max=None)
    try:
        labels = {line.get_label() for line in figure.axes[1].lines}
        assert r"$\sigma_{\max}$ ceiling" not in labels
    finally:
        figures.plt.close(figure)


def test_sigma_trust_region_has_one_l2_geometry_for_diagonal_and_full() -> None:
    sigma_diag = torch.ones(2, dtype=torch.float64)
    delta_diag = torch.tensor([3.0, 4.0], dtype=torch.float64)

    diagonal = retract_spd_diagonal(
        sigma_diag, delta_diag, trust_region=1.0, eps=1e-12, sigma_max=None,
    )
    full = retract_spd_full(
        torch.diag_embed(sigma_diag),
        torch.diag_embed(delta_diag),
        trust_region=1.0,
        eps=1e-12,
        sigma_max=None,
    )

    assert torch.allclose(diagonal, torch.diagonal(full), atol=1e-10, rtol=1e-10)
    assert torch.allclose(torch.log(diagonal), torch.tensor([0.6, 0.8], dtype=torch.float64),
                          atol=1e-10, rtol=1e-10)


def test_sigma_gate_fails_closed_with_too_few_tokens_and_stays_strict_json() -> None:
    sigma = torch.tensor([0.1, 0.2, 0.3])
    ce = torch.tensor([1.0, 1.1, 1.2])
    conf = torch.tensor([0.8, 0.7, 0.6])
    correct = torch.tensor([1.0, 1.0, 0.0])

    record = evaluate_sigma_gate(
        sigma,
        ce,
        conf,
        correct,
        n_strata=10,
        n_bins=10,
        n_boot=8,
        n_perm=8,
    )

    assert record["status"] == "FAIL"
    assert record["failure_reason"] == "insufficient_tokens"
    assert record["minimum_tokens"] == 10
    json.dumps(record, allow_nan=False)


def test_sigma_gate_artifact_rejects_nonfinite_json(tmp_path) -> None:
    with pytest.raises(ValueError, match="Out of range float values"):
        write_sigma_gate_artifact(
            {"status": "FAIL", "invalid_statistic": float("nan")},
            checkpoint_id="nonfinite",
            spec_commit="test",
            seeds=(0,),
            out_dir=str(tmp_path),
        )

    assert list(tmp_path.iterdir()) == []


def test_umap_embed_can_reuse_one_isolated_worker() -> None:
    class _Worker:
        def __init__(self) -> None:
            self.calls = 0

        def embed(self, features, **_kwargs):
            self.calls += 1
            return np.zeros((features.shape[0], 2), dtype=float)

    worker = _Worker()
    features = np.arange(12, dtype=float).reshape(4, 3)

    first = figures.umap_embed(features, worker=worker)
    second = figures.umap_embed(features + 1.0, worker=worker)

    assert worker.calls == 2
    assert first.shape == second.shape == (4, 2)


def test_umap_worker_reuses_one_process_for_two_embeddings() -> None:
    pytest.importorskip("umap")
    features = np.arange(120, dtype=float).reshape(40, 3)

    with figures.UMAPWorker() as worker:
        first = figures.umap_embed(features, worker=worker)
        pid = worker._proc.pid
        second = figures.umap_embed(features + 1.0, worker=worker)

        assert worker._counter == 2
        assert worker._proc.pid == pid
        assert first.shape == second.shape == (40, 2)
