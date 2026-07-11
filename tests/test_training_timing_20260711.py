import pytest
import torch


class _ManualClock:
    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


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
