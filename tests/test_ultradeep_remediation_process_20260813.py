"""Task 7: bounded UMAP recovery and periodic-generation isolation."""

import shutil
from threading import Event, Thread
import time

import numpy as np
import pytest
import torch

from vfe3 import train as train_module
from vfe3.viz import figures


def test_timed_out_umap_worker_is_replaced_and_next_request_succeeds(monkeypatch):
    """A request-specific hang may only retire its own worker and must be recoverable."""
    controlled_worker = (
        "import json, os, sys, time\n"
        "import numpy as np\n"
        "for line in sys.stdin:\n"
        "    request = json.loads(line)\n"
        "    if int(request['seed']) == 1:\n"
        "        while True: time.sleep(0.01)\n"
        "    np.save(request['output'], np.zeros((2, 2), dtype=np.float32))\n"
        "    tmp = request['status'] + '.tmp'\n"
        "    with open(tmp, 'w', encoding='utf-8') as h: json.dump({'ok': True}, h)\n"
        "    os.replace(tmp, request['status'])\n"
    )
    monkeypatch.setattr(figures, "_UMAP_WORKER_SRC", controlled_worker)
    worker = figures.UMAPWorker(timeout=0.10, cleanup_timeout=0.25, max_workers=1)
    features = np.zeros((2, 3), dtype=np.float32)
    try:
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="exceeded"):
            worker.embed(features, n_neighbors=2, min_dist=0.1, n_components=2, seed=1)
        assert time.monotonic() - started < 0.75
        assert worker._procs == []
        assert worker.cleanup_statuses[-1]["reason"] == "timeout"
        assert worker.cleanup_statuses[-1]["terminated"] is True

        recovered = worker.embed(features, n_neighbors=2, min_dist=0.1, n_components=2, seed=2)
        assert np.array_equal(recovered, np.zeros((2, 2), dtype=np.float32))
        assert len(worker._procs) == 1
    finally:
        worker.close()


def test_umap_close_is_bounded_and_retires_in_reverse_order(monkeypatch):
    """Finalization reports every owned worker without an unbounded wait."""
    controlled_worker = "import sys\nfor _line in sys.stdin:\n    pass\n"
    monkeypatch.setattr(figures, "_UMAP_WORKER_SRC", controlled_worker)
    worker = figures.UMAPWorker(timeout=0.10, cleanup_timeout=0.25, max_workers=2)
    worker._ensure(2)
    pids = [proc.pid for proc in worker._procs]

    started = time.monotonic()
    statuses = worker.close()

    assert time.monotonic() - started < 0.75
    assert [status["pid"] for status in statuses] == list(reversed(pids))
    assert all(status["reason"] == "close" for status in statuses)
    assert all(status["reaped"] is True for status in statuses)
    assert worker._procs == []
    assert worker._stderr_handles == []


class _ExplodingGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.5)
        self.branch = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.1))
        self.seen_training = None

    def generate(self, prompt, new_tokens, *, greedy):
        self.seen_training = [module.training for module in self.modules()]
        torch.rand(3)
        raise RuntimeError("controlled sample failure")


def test_periodic_generation_restores_mixed_modes_and_rng_when_it_raises(monkeypatch):
    """Sampling is evaluation-only and cannot perturb the next training step's state."""
    model = _ExplodingGenerator()
    model.train(True)
    model.branch.train(False)  # mixed submodule modes are intentional and must round-trip exactly.
    original_modes = [module.training for module in model.modules()]
    torch.manual_seed(1234)
    original_cpu = torch.get_rng_state().clone()
    fake_cuda = [torch.tensor([4, 2, 0], dtype=torch.uint8)]
    restored_cuda = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [state.clone() for state in fake_cuda])
    monkeypatch.setattr(torch.cuda, "set_rng_state_all",
                        lambda states: restored_cuda.append([state.clone() for state in states]))

    with pytest.raises(RuntimeError, match="controlled sample failure"):
        train_module._periodic_generation(
            model, torch.tensor([[1, 2]]), 3, lambda ids: str(list(ids)),
        )

    assert model.seen_training == [False] * len(original_modes)
    assert [module.training for module in model.modules()] == original_modes
    assert torch.equal(torch.get_rng_state(), original_cpu)
    assert len(restored_cuda) == 1
    assert [state.tolist() for state in restored_cuda[0]] == [state.tolist() for state in fake_cuda]


def test_periodic_generation_cpu_restore_failure_is_fail_closed(monkeypatch):
    model = _ExplodingGenerator()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch,
        "set_rng_state",
        lambda _state: (_ for _ in ()).throw(RuntimeError("cpu restore failed")),
    )

    with pytest.raises(train_module.PeriodicGenerationRestorationError, match="cpu restore failed"):
        train_module._periodic_generation(
            model, torch.tensor([[1, 2]]), 3, lambda ids: str(list(ids)),
        )


def test_periodic_generation_partial_cuda_restore_failure_is_fail_closed(monkeypatch):
    model = _ExplodingGenerator()
    fake_cuda = [torch.tensor([1], dtype=torch.uint8), torch.tensor([2], dtype=torch.uint8)]
    partial = []

    def _partial_then_raise(states):
        partial.append(states[0].clone())
        raise RuntimeError("cuda restore failed after device zero")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [state.clone() for state in fake_cuda])
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", _partial_then_raise)

    with pytest.raises(train_module.PeriodicGenerationRestorationError, match="cuda restore failed"):
        train_module._periodic_generation(
            model, torch.tensor([[1, 2]]), 3, lambda ids: str(list(ids)),
        )

    assert [state.tolist() for state in partial] == [[1]]


def test_multiworker_timeout_cleanup_is_removed_or_structured(monkeypatch):
    controlled_worker = (
        "import json, os, sys, time\n"
        "import numpy as np\n"
        "for line in sys.stdin:\n"
        "    request = json.loads(line)\n"
        "    if int(request['seed']) == 1:\n"
        "        while True: time.sleep(0.01)\n"
        "    time.sleep(0.20)\n"
        "    np.save(request['output'], np.zeros((2, 2), dtype=np.float32))\n"
        "    tmp = request['status'] + '.tmp'\n"
        "    with open(tmp, 'w', encoding='utf-8') as h: json.dump({'ok': True}, h)\n"
        "    os.replace(tmp, request['status'])\n"
    )
    monkeypatch.setattr(figures, "_UMAP_WORKER_SRC", controlled_worker)
    worker = figures.UMAPWorker(timeout=0.05, cleanup_timeout=0.25, max_workers=2)
    real_remove = figures.os.remove
    injected = {"done": False}

    def _sharing_violation_once(path):
        if not injected["done"] and str(path).endswith("_out.npy"):
            injected["done"] = True
            raise PermissionError("controlled sharing violation")
        return real_remove(path)

    monkeypatch.setattr(figures.os, "remove", _sharing_violation_once)
    try:
        with pytest.raises(TimeoutError, match="exceeded"):
            worker.embed_many(
                np.zeros((2, 3), dtype=np.float32), seeds=(1, 2),
                n_neighbors=2, min_dist=0.1, n_components=2,
            )
        workdir = worker._workdir
    finally:
        worker.close()

    assert injected["done"] is True
    assert any(
        status["reason"] == "request_cleanup" and "sharing violation" in status["cleanup_error"]
        for status in worker.cleanup_statuses
    )
    assert workdir is not None and not figures.os.path.exists(workdir)


def test_workdir_cleanup_failure_is_structured_and_not_raised(monkeypatch):
    controlled_worker = "import sys\nfor _line in sys.stdin:\n    pass\n"
    monkeypatch.setattr(figures, "_UMAP_WORKER_SRC", controlled_worker)
    worker = figures.UMAPWorker(timeout=0.10, cleanup_timeout=0.25, max_workers=1)
    worker._ensure(1)
    monkeypatch.setattr(
        shutil,
        "rmtree",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(PermissionError("workdir busy")),
    )

    statuses = worker.close()

    assert any(
        status["reason"] == "workdir_cleanup" and "workdir busy" in status["cleanup_error"]
        for status in statuses
    )


def test_concurrent_double_close_retires_each_worker_once(monkeypatch):
    controlled_worker = "import time\nwhile True: time.sleep(0.01)\n"
    monkeypatch.setattr(figures, "_UMAP_WORKER_SRC", controlled_worker)
    worker = figures.UMAPWorker(timeout=0.10, cleanup_timeout=0.25, max_workers=2)
    worker._ensure(2)
    pids = [proc.pid for proc in worker._procs]
    release = Event()
    errors = []

    def _close():
        release.wait()
        try:
            worker.close()
        except BaseException as exc:
            errors.append(exc)

    threads = [Thread(target=_close), Thread(target=_close)]
    for thread in threads:
        thread.start()
    release.set()
    for thread in threads:
        thread.join(timeout=1.0)

    assert not any(thread.is_alive() for thread in threads)
    assert errors == []
    retired_pids = [status["pid"] for status in worker.cleanup_statuses if status.get("pid")]
    assert sorted(retired_pids) == sorted(pids)
    assert len(retired_pids) == len(set(retired_pids))


def test_close_racing_timeout_is_once_only_and_bounded(monkeypatch):
    controlled_worker = (
        "import json, sys, time\n"
        "for line in sys.stdin:\n"
        "    json.loads(line)\n"
        "    while True: time.sleep(0.01)\n"
    )
    monkeypatch.setattr(figures, "_UMAP_WORKER_SRC", controlled_worker)
    worker = figures.UMAPWorker(timeout=0.08, cleanup_timeout=0.25, max_workers=1)
    started = Event()
    errors = []

    def _embed():
        started.set()
        try:
            worker.embed(
                np.zeros((2, 3), dtype=np.float32), n_neighbors=2,
                min_dist=0.1, n_components=2, seed=1,
            )
        except (TimeoutError, OSError):
            pass
        except BaseException as exc:
            errors.append(exc)

    embed_thread = Thread(target=_embed)
    embed_thread.start()
    started.wait(timeout=0.25)
    deadline = time.monotonic() + 0.25
    while not worker._procs and time.monotonic() < deadline:
        Event().wait(0.005)
    close_thread = Thread(target=worker.close)
    close_thread.start()
    embed_thread.join(timeout=1.0)
    close_thread.join(timeout=1.0)

    assert not embed_thread.is_alive() and not close_thread.is_alive()
    assert errors == []
    retired_pids = [status["pid"] for status in worker.cleanup_statuses if status.get("pid")]
    assert len(retired_pids) == len(set(retired_pids))


def test_close_racing_blocked_submit_flush_is_bounded_and_once_only(monkeypatch):
    flush_entered = Event()
    release_flush = Event()
    lifecycle_events = []

    class _BlockedStdin:
        def write(self, _payload):
            return None

        def flush(self):
            flush_entered.set()
            release_flush.wait(5.0)

        def close(self):
            lifecycle_events.append("stdin_close")
            release_flush.wait(5.0)

        def fileno(self):
            raise OSError("synthetic stream has no file descriptor")

    class _Process:
        pid = 43210

        def __init__(self):
            self.stdin = _BlockedStdin()
            self.returncode = None

        def poll(self):
            return self.returncode

    class _Tree:
        def __init__(self):
            self.process = _Process()

        def terminate(self, *, reason, timeout):
            lifecycle_events.append("tree_terminate")
            self.process.returncode = -9
            return {
                "pid": self.process.pid,
                "reason": reason,
                "terminated": True,
                "reaped": True,
                "root_reaped": True,
                "tree_termination_confirmed": True,
                "returncode": -9,
                "cleanup_error": None,
                "elapsed_s": min(timeout, 0.001),
            }

    monkeypatch.setattr(figures, "spawn_process_tree", lambda *_a, **_k: _Tree())
    worker = figures.UMAPWorker(timeout=0.05, cleanup_timeout=0.10, max_workers=1)
    errors = []

    def _embed():
        try:
            worker.embed(
                np.zeros((2, 3), dtype=np.float32), n_neighbors=2,
                min_dist=0.1, n_components=2, seed=1,
            )
        except (TimeoutError, OSError):
            pass
        except BaseException as exc:
            errors.append(exc)

    embed_thread = Thread(target=_embed)
    close_thread = Thread(target=worker.close)
    started = time.monotonic()
    try:
        embed_thread.start()
        assert flush_entered.wait(0.25)
        close_thread.start()
        embed_thread.join(timeout=0.60)
        close_thread.join(timeout=0.60)

        assert time.monotonic() - started < 0.75
        assert not embed_thread.is_alive() and not close_thread.is_alive()
        assert errors == []
        assert "stdin_close" in lifecycle_events
        assert lifecycle_events.index("tree_terminate") < lifecycle_events.index("stdin_close")
        statuses = [status for status in worker.cleanup_statuses if status.get("pid")]
        assert len(statuses) == 1
        assert statuses[0]["pid"] == 43210
    finally:
        release_flush.set()
        embed_thread.join(timeout=0.25)
        close_thread.join(timeout=0.25)
