import json
import os
import subprocess

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


def test_umap_worker_mocked_protocol_reuses_one_process(monkeypatch) -> None:
    processes = []

    class _Stdin:
        def __init__(self) -> None:
            self.requests = []
            self.inputs = []
            self.outputs = []
            self.pending_statuses = []
            self.flush_count = 0
            self.closed = False

        def write(self, line: str) -> int:
            request = json.loads(line)
            features = np.load(request["input"])
            n_components = int(request["n_components"])
            output = np.stack(
                [features[:, component % features.shape[1]]
                 + float(request["seed"]) + component * float(request["min_dist"])
                 for component in range(n_components)],
                axis=1,
            )
            np.save(request["output"], output)
            self.requests.append(request)
            self.inputs.append(features.copy())
            self.outputs.append(output.copy())
            self.pending_statuses.append(request["status"])
            return len(line)

        def flush(self) -> None:
            self.flush_count += 1

        def close(self) -> None:
            self.closed = True

    class _Process:
        def __init__(self, args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs
            self.stdin = _Stdin()
            self.wait_timeouts = []
            self.kill_count = 0
            self.poll_count = 0
            self.status_publications = 0

        def poll(self):
            self.poll_count += 1
            status = self.stdin.pending_statuses.pop(0)
            with open(status, "w", encoding="utf-8") as handle:
                json.dump({"ok": True}, handle)
            self.status_publications += 1
            return None

        def wait(self, timeout=None):
            self.wait_timeouts.append(timeout)
            return 0

        def kill(self) -> None:
            self.kill_count += 1

    def _popen(args, **kwargs):
        process = _Process(args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", _popen)
    first_features = np.arange(15, dtype=float).reshape(5, 3)
    second_features = first_features + 10.0

    with figures.UMAPWorker(timeout=0.25) as worker:
        assert worker._proc is None
        assert processes == []

        first = worker.embed(
            first_features,
            n_neighbors=3,
            min_dist=0.2,
            n_components=2,
            seed=7,
        )
        process = worker._proc
        workdir = worker._workdir
        second = worker.embed(
            second_features,
            n_neighbors=4,
            min_dist=0.4,
            n_components=2,
            seed=11,
        )

        assert len(processes) == 1
        assert worker._proc is process
        assert worker._counter == 2
        assert process.stdin.flush_count == 2
        assert process.poll_count == 2
        assert process.status_publications == 2
        assert process.stdin.pending_statuses == []
        assert process.stdin.inputs[0].tolist() == first_features.tolist()
        assert process.stdin.inputs[1].tolist() == second_features.tolist()
        assert [
            {key: request[key] for key in ("n_neighbors", "min_dist", "n_components", "seed")}
            for request in process.stdin.requests
        ] == [
            {"n_neighbors": 3, "min_dist": 0.2, "n_components": 2, "seed": 7},
            {"n_neighbors": 4, "min_dist": 0.4, "n_components": 2, "seed": 11},
        ]
        assert np.array_equal(first, process.stdin.outputs[0])
        assert np.array_equal(second, process.stdin.outputs[1])
        assert all(
            not any(os.path.exists(path) for path in (
                request["input"], request["output"], request["status"],
                f"{request['status']}.tmp",
            ))
            for request in process.stdin.requests
        )
        assert os.path.isdir(workdir)

    assert process.stdin.closed
    assert process.wait_timeouts == [5.0]
    assert process.kill_count == 0
    assert worker._proc is None
    assert worker._stderr_handle is None
    assert worker._workdir is None
    assert not os.path.exists(workdir)


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
