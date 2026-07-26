import json
import os
import subprocess

import numpy as np
import pytest
import torch

import vfe3.viz.figures as figures
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

    with figures.UMAPWorker(timeout=5.0) as worker:
        assert worker._procs == []
        assert processes == []

        first = worker.embed(
            first_features,
            n_neighbors=3,
            min_dist=0.2,
            n_components=2,
            seed=7,
        )
        process = worker._procs[0]
        workdir = worker._workdir
        second = worker.embed(
            second_features,
            n_neighbors=4,
            min_dist=0.4,
            n_components=2,
            seed=11,
        )

        # Single-embedding callers must still pay for exactly ONE interpreter: the pool added on
        # 2026-07-26 grows on demand, so a serial caller never starts a second process.
        assert len(processes) == 1
        assert worker._procs == [process]
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
    assert worker._procs == []
    assert worker._stderr_handles == []
    assert worker._workdir is None
    assert not os.path.exists(workdir)


# --- 2026-07-26: UMAP was ~95% of figure wall clock on one of 24 cores -------------------


def test_embed_many_fans_the_seeds_across_concurrent_interpreters(monkeypatch) -> None:
    r"""The controlled protocol's seeds must be SUBMITTED before any of them is collected.

    Each UMAP fit is pinned single-threaded by its ``random_state``, so the pool is the only
    parallelism available. A pool that submitted and awaited one request at a time would be the
    serial loop with extra machinery, and the wall clock would not move -- so the ordering, not
    merely the process count, is what this pins.
    """
    import subprocess

    from vfe3.viz import figures

    events = []

    class _Stdin:
        def __init__(self, owner) -> None:
            self.owner = owner
            self.pending = []
            self.closed = False

        def write(self, line: str) -> int:
            request = json.loads(line)
            events.append(("submit", self.owner, int(request["seed"])))
            features = np.load(request["input"])
            np.save(request["output"], features[:, :int(request["n_components"])]
                    + float(request["seed"]))
            self.pending.append((request["status"], int(request["seed"])))
            return len(line)

        def flush(self) -> None:
            pass

        def close(self) -> None:
            self.closed = True

    class _Process:
        def __init__(self, index) -> None:
            self.stdin = _Stdin(index)

        def poll(self):
            status, seed = self.stdin.pending.pop(0)
            events.append(("collect", self.stdin.owner, seed))
            with open(status, "w", encoding="utf-8") as handle:
                json.dump({"ok": True}, handle)
            return None

        def wait(self, timeout=None):
            return 0

        def kill(self) -> None:                      # pragma: no cover - not reached
            raise AssertionError("no worker should be killed")

    created = []

    def _popen(args, **kwargs):
        process = _Process(len(created))
        created.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", _popen)
    features = np.arange(24, dtype=float).reshape(8, 3)

    with figures.UMAPWorker(timeout=5.0, max_workers=4) as worker:
        coords = worker.embed_many(features, seeds=(0, 1, 2, 3),
                                   n_neighbors=3, min_dist=0.1, n_components=2)

    assert sorted(coords) == [0, 1, 2, 3]
    assert len(created) == 4                         # one interpreter per seed, none reused serially
    kinds = [kind for kind, _owner, _seed in events]
    # THE point of the pool: all four fits are in flight before the first result is taken. A serial
    # loop would interleave as submit/collect/submit/collect and this reads as four then four.
    assert kinds == ["submit"] * 4 + ["collect"] * 4
    assert [seed for kind, _owner, seed in events if kind == "submit"] == [0, 1, 2, 3]
    assert {owner for _kind, owner, _seed in events} == {0, 1, 2, 3}
    # Every fit sees the SAME feature bank, written once.
    for seed in (0, 1, 2, 3):
        assert np.allclose(coords[seed], features[:, :2] + seed)


def test_embed_many_never_exceeds_max_workers(monkeypatch) -> None:
    r"""More seeds than workers queues onto the existing pool rather than forking per seed."""
    import subprocess

    from vfe3.viz import figures

    class _Stdin:
        def __init__(self) -> None:
            self.pending = []
            self.closed = False

        def write(self, line: str) -> int:
            request = json.loads(line)
            features = np.load(request["input"])
            np.save(request["output"], features[:, :int(request["n_components"])])
            self.pending.append(request["status"])
            return len(line)

        def flush(self) -> None:
            pass

        def close(self) -> None:
            self.closed = True

    class _Process:
        def __init__(self) -> None:
            self.stdin = _Stdin()

        def poll(self):
            status = self.stdin.pending.pop(0)
            with open(status, "w", encoding="utf-8") as handle:
                json.dump({"ok": True}, handle)
            return None

        def wait(self, timeout=None):
            return 0

        def kill(self) -> None:                      # pragma: no cover - not reached
            raise AssertionError("no worker should be killed")

    created = []
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: created.append(_Process()) or created[-1])
    features = np.arange(24, dtype=float).reshape(8, 3)

    with figures.UMAPWorker(timeout=5.0, max_workers=2) as worker:
        coords = worker.embed_many(features, seeds=(0, 1, 2, 3, 4),
                                   n_neighbors=3, min_dist=0.1, n_components=2)

    assert sorted(coords) == [0, 1, 2, 3, 4]
    assert len(created) == 2


def test_embed_many_rejects_duplicate_seeds() -> None:
    from vfe3.viz import figures

    with figures.UMAPWorker(timeout=5.0) as worker:
        with pytest.raises(ValueError, match="distinct seeds"):
            worker.embed_many(np.zeros((4, 3)), seeds=(1, 1),
                              n_neighbors=3, min_dist=0.1, n_components=2)


def test_numba_cache_is_persistent_and_survives_close(monkeypatch, tmp_path) -> None:
    r"""The worker pointed NUMBA_CACHE_DIR at its own scratch tempdir, which close() removed.

    umap-learn's numba kernels were therefore JIT-compiled from scratch on every report and the
    cache never survived a single run. It must now live OUTSIDE the request scratch.
    """
    import subprocess

    from vfe3.viz import figures

    cache_root = tmp_path / "persistent-cache"
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(cache_root))
    cache_root.mkdir()
    envs = []

    class _Process:
        def __init__(self) -> None:
            self.stdin = type("S", (), {"close": lambda self: None})()

        def wait(self, timeout=None):
            return 0

        def kill(self) -> None:                      # pragma: no cover - not reached
            raise AssertionError("no worker should be killed")

    def _popen(args, **kwargs):
        envs.append(kwargs["env"])
        return _Process()

    monkeypatch.setattr(subprocess, "Popen", _popen)

    worker = figures.UMAPWorker(timeout=5.0)
    worker._ensure(2)
    workdir = worker._workdir
    worker.close()

    assert len(envs) == 2
    for env in envs:
        assert env["NUMBA_CACHE_DIR"] == str(cache_root)
        assert workdir not in env["NUMBA_CACHE_DIR"]
    assert cache_root.is_dir()                       # outlives the worker
    assert not os.path.exists(workdir)               # the request scratch does not
