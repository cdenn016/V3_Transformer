import csv
import math
from pathlib import Path

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel
from vfe3.run_artifacts import RunArtifacts


class _ManualClock:
    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _tiny_training_cfg(**overrides: object) -> VFE3Config:
    values: dict[str, object] = {
        "vocab_size":        6,
        "embed_dim":         4,
        "n_heads":           2,
        "max_seq_len":       4,
        "n_layers":          1,
        "n_e_steps":         1,
        "e_phi_lr":          0.0,
        "m_phi_lr":          0.0,
        "warmup_steps":      1,
        "max_steps":         4,
        "eval_max_batches":  1,
        "checkpoint_interval": 0,
        "generate_figures":  False,
    }
    values.update(overrides)
    return VFE3Config(**values)


def _direct_batches() -> list[tuple[torch.Tensor, torch.Tensor]]:
    tokens = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    targets = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]], dtype=torch.long)
    return [(tokens, targets)]


def test_cpu_timer_aggregates_and_resets_windows() -> None:
    from vfe3.timing import TrainingTimer

    clock = _ManualClock()
    timer = TrainingTimer(torch.device("cpu"), clock=clock)
    timer.start_step()
    clock.advance(0.010)
    timer.finish_step(n_tokens=8)
    clock.advance(0.005)
    timer.start_step()
    clock.advance(0.030)
    timer.finish_step(n_tokens=24)

    clean = timer.sample_train_window()
    assert clean.train_step_ms_mean == pytest.approx(20.0)
    assert clean.train_steps_per_s == pytest.approx(50.0)
    assert clean.train_step_tokens_per_s == pytest.approx(800.0)

    clock.advance(0.010)
    pipeline = timer.sample_pipeline_window()
    assert pipeline.pipeline_tokens_per_s == pytest.approx(32 / 0.055)
    assert pipeline.wall_clock_s == pytest.approx(0.055)

    clock.advance(0.100)
    timer.reset_pipeline_window()
    timer.start_step()
    clock.advance(0.020)
    timer.finish_step(n_tokens=16)
    assert timer.sample_train_window().train_step_tokens_per_s == pytest.approx(800.0)
    assert timer.sample_pipeline_window().pipeline_tokens_per_s == pytest.approx(800.0)


def test_cuda_timer_uses_only_event_specific_reporting_syncs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vfe3.timing import TrainingTimer

    trace: list[tuple] = []
    durations = {("s1", "e1"): 4.0, ("s2", "e2"): 6.0}
    stream = object()

    class _FakeEvent:
        def __init__(self, name: str) -> None:
            self.name = name

        def record(self, selected_stream: object) -> None:
            trace.append(("record", self.name, selected_stream))

        def synchronize(self) -> None:
            trace.append(("sync", self.name))

        def elapsed_time(self, other: "_FakeEvent") -> float:
            trace.append(("elapsed", self.name, other.name))
            return durations[(self.name, other.name)]

    events = iter(_FakeEvent(name) for name in ("s1", "e1", "s2", "e2", "pipeline"))

    def _event_factory(*, enable_timing: bool = False) -> _FakeEvent:
        trace.append(("factory", enable_timing))
        return next(events)

    def _stream_factory(device: torch.device) -> object:
        trace.append(("stream", device))
        return stream

    def _global_sync_forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("global CUDA synchronization is forbidden")

    monkeypatch.setattr(torch.cuda, "synchronize", _global_sync_forbidden)
    timer = TrainingTimer(
        torch.device("cuda:3"),
        clock=_ManualClock(),
        cuda_event_factory=_event_factory,
        cuda_stream_factory=_stream_factory,
    )
    timer.start_step()
    timer.finish_step(n_tokens=8)
    timer.start_step()
    timer.finish_step(n_tokens=24)
    assert not [item for item in trace if item[0] == "sync"]

    clean = timer.sample_train_window()
    assert [item for item in trace if item[0] == "sync"] == [("sync", "e2")]
    assert clean.train_step_ms_mean == pytest.approx(5.0)
    assert clean.train_steps_per_s == pytest.approx(200.0)
    assert clean.train_step_tokens_per_s == pytest.approx(3200.0)

    timer.sample_pipeline_window()
    assert [item for item in trace if item[0] == "sync"] == [
        ("sync", "e2"),
        ("sync", "pipeline"),
    ]


def test_train_timing_attributes_callbacks_to_triggering_pipeline_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vfe3.train as train_module
    from vfe3.timing import TrainingTimer

    torch.manual_seed(0)
    clock = _ManualClock()
    cfg = _tiny_training_cfg(max_steps=2, checkpoint_interval=1)
    model = VFEModel(cfg)
    artifacts = RunArtifacts(tmp_path / "run", cfg, model, device="cpu")

    real_train_step = train_module.train_step
    real_maybe_save_best = artifacts.maybe_save_best
    real_save_checkpoint = artifacts.save_checkpoint
    real_log_metrics = artifacts.log_metrics

    def _timed_train_step(*args: object, **kwargs: object) -> float:
        result = real_train_step(*args, **kwargs)
        clock.advance(2.0)
        return result

    def _timed_maybe_save_best(*args: object, **kwargs: object) -> bool:
        result = real_maybe_save_best(*args, **kwargs)
        clock.advance(10.0)
        return result

    def _timed_save_checkpoint(*args: object, **kwargs: object) -> Path:
        result = real_save_checkpoint(*args, **kwargs)
        clock.advance(20.0)
        return result

    def _timed_log_metrics(*args: object, **kwargs: object) -> None:
        real_log_metrics(*args, **kwargs)
        clock.advance(100.0)

    def _timer_factory(device: torch.device) -> TrainingTimer:
        return TrainingTimer(device, clock=clock)

    monkeypatch.setattr(train_module, "TrainingTimer", _timer_factory, raising=False)
    monkeypatch.setattr(train_module, "train_step", _timed_train_step)
    monkeypatch.setattr(artifacts, "maybe_save_best", _timed_maybe_save_best)
    monkeypatch.setattr(artifacts, "save_checkpoint", _timed_save_checkpoint)
    monkeypatch.setattr(artifacts, "log_metrics", _timed_log_metrics)

    train_module.train(
        model,
        _direct_batches(),
        cfg,
        n_steps=2,
        log_interval=1,
        eval_interval=1,
        val_loader=_direct_batches(),
        device=torch.device("cpu"),
        artifacts=artifacts,
        generate_samples=False,
    )

    assert [row["train_step_ms_mean"] for row in artifacts.history] == pytest.approx(
        [2000.0, 2000.0]
    )
    assert [row["train_step_tokens_per_s"] for row in artifacts.history] == pytest.approx(
        [4.0, 4.0]
    )
    assert [row["pipeline_tokens_per_s"] for row in artifacts.history] == pytest.approx(
        [0.25, 0.25]
    )
    assert [row["wall_clock_s"] for row in artifacts.history] == pytest.approx([32.0, 164.0])
    assert all(
        row["tokens_per_s"] == row["pipeline_tokens_per_s"] for row in artifacts.history
    )


def test_mixed_log_eval_rows_keep_timing_schema_rectangular(
    tmp_path: Path,
) -> None:
    import vfe3.train as train_module

    torch.manual_seed(0)
    cfg = _tiny_training_cfg(max_steps=3)
    model = VFEModel(cfg)
    artifacts = RunArtifacts(tmp_path / "run", cfg, model, device="cpu")

    train_module.train(
        model,
        _direct_batches(),
        cfg,
        n_steps=3,
        log_interval=2,
        eval_interval=3,
        val_loader=_direct_batches(),
        device=torch.device("cpu"),
        artifacts=artifacts,
        generate_samples=False,
    )

    assert [row["step"] for row in artifacts.history] == [2, 3]
    expected_fields = set(artifacts.history[0])
    assert all(set(row) == expected_fields for row in artifacts.history)
    assert {
        "train_step_ms_mean",
        "train_step_tokens_per_s",
        "pipeline_tokens_per_s",
        "tokens_per_s",
        "wall_clock_s",
    } <= expected_fields
    assert all(
        row["tokens_per_s"] == row["pipeline_tokens_per_s"] for row in artifacts.history
    )
    with artifacts.csv_path.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 2
    assert all(set(row) == expected_fields for row in csv_rows)


def test_cpu_train_on_cuda_capable_host_never_calls_cuda_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vfe3.train as train_module

    torch.manual_seed(0)
    cfg = _tiny_training_cfg(max_steps=1)
    model = VFEModel(cfg)
    artifacts = RunArtifacts(tmp_path / "run", cfg, model, device="cpu")

    def _cuda_helper_forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("CPU training called a CUDA helper")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "Event", _cuda_helper_forbidden)
    monkeypatch.setattr(torch.cuda, "current_stream", _cuda_helper_forbidden)
    monkeypatch.setattr(torch.cuda, "synchronize", _cuda_helper_forbidden)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", _cuda_helper_forbidden)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", _cuda_helper_forbidden)

    train_module.train(
        model,
        _direct_batches(),
        cfg,
        n_steps=1,
        log_interval=1,
        device=torch.device("cpu"),
        artifacts=artifacts,
        generate_samples=False,
    )

    assert len(artifacts.history) == 1
    assert math.isnan(artifacts.history[0]["peak_mem_mb"])


def test_silent_train_route_does_not_construct_timer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vfe3.train as train_module

    torch.manual_seed(0)
    cfg = _tiny_training_cfg(max_steps=1)
    model = VFEModel(cfg)

    def _timer_construction_forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("silent training constructed a timer")

    monkeypatch.setattr(train_module, "TrainingTimer", _timer_construction_forbidden)
    losses = train_module.train(
        model,
        _direct_batches(),
        cfg,
        n_steps=1,
        log_interval=None,
        eval_interval=None,
        val_loader=None,
        device=torch.device("cpu"),
        artifacts=None,
        generate_samples=False,
    )

    assert len(losses) == 1
