"""Task 7: bounded UMAP recovery and periodic-generation isolation."""

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
