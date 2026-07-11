# P6 Training and Pipeline Timing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the contaminated legacy throughput window with explicit clean `train_step` timing
and correctly aligned end-to-end pipeline timing on CPU and CUDA.

**Architecture:** Add one focused `TrainingTimer` that accumulates exact token counts, CPU call-wall
durations, and CUDA event pairs without synchronizing nonreporting steps. `train()` brackets only
`train_step`, samples the clean window before callbacks, samples the pipeline after callbacks,
checkpointing, and row construction, then resets after the CSV append.

**Tech Stack:** Python 3.10+, PyTorch CUDA events, `time.perf_counter`, pytest, JUnit XML, and the
existing `RunArtifacts` CSV layer.

## Global Constraints

- Preserve model outputs, optimizer semantics, scheduler cadence, RNG consumption, data order,
  configuration defaults, and all mathematically pure paths.
- Clean timing brackets only `train_step`; Metropolis, EMA, evaluation, generation, figures,
  best-state writes, checkpoints, and row construction belong only to pipeline timing.
- CUDA uses event-specific synchronization at the two reporting boundaries and never calls global
  `torch.cuda.synchronize()`; nonreporting steps do not synchronize.
- CPU execution must not resolve or call CUDA event, stream, synchronization, or memory helpers even
  when CUDA is available to the process.
- Persist `train_step_ms_mean`, `train_step_tokens_per_s`, and `pipeline_tokens_per_s` on every row.
  Preserve `tokens_per_s` as an exact alias of `pipeline_tokens_per_s`; this is schema compatibility,
  not historical semantic continuity.
- Exclude each `log_metrics` append from both adjacent pipeline-rate windows, while cumulative
  `wall_clock_s` includes prior appends.
- Leave the native tqdm progress display unchanged.
- Do not add another pytest quiet flag. Read all pass, skip, failure, and error counts from JUnit XML.
- Update only the existing `docs/2026-07-11-edits.md` post-edit record for this date.

## File Map

- Create `vfe3/timing.py`: backend-aware timing state and immutable timing samples.
- Create `tests/test_training_timing_20260711.py`: deterministic CPU, fake-CUDA, integration,
  silent-path, device-guard, schema, and real-CUDA regressions.
- Modify `vfe3/train.py`: bracket the clean step, align callbacks/checkpoint/row construction, emit
  metrics, and use the resolved-device memory guard.
- Modify `tests/test_run_diagnostics_2026_06_13.py`: extend the canonical metrics schema assertion.
- Modify `check_gpu_tests.py`: add the real-CUDA P6 smoke node to the existing T6 list.
- Modify `docs/superpowers/specs/2026-07-11-p6-training-timing-design.md`: retain the approved
  synchronization correction and final implementation status.
- Modify `docs/2026-07-11-edits.md`: record the implementation and machine-read verification.

### Task 1: Implement the backend-aware timing component test-first

**Files:**

- Create: `tests/test_training_timing_20260711.py`
- Create: `vfe3/timing.py`

**Interfaces:**

- Produces `TrainTimingSample(train_step_ms_mean, train_steps_per_s,
  train_step_tokens_per_s)`.
- Produces `PipelineTimingSample(pipeline_tokens_per_s, wall_clock_s)`.
- Produces `TrainingTimer.start_step()`, `finish_step(*, n_tokens: int)`,
  `sample_train_window()`, `sample_pipeline_window()`, and `reset_pipeline_window()`.

- [ ] **Step 1: Write the deterministic CPU and fake-CUDA tests**

Create the manual clock and tests below. Keep the `TrainingTimer` imports inside the tests for the
initial red run so a missing module is reported as the intended test failure rather than a collection
failure.

```python
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
```

- [ ] **Step 2: Run the unit tests and verify the intended red state**

Run:

```powershell
python -m pytest tests/test_training_timing_20260711.py --junitxml=C:\tmp\vfe3-p6-timer-red.xml
```

Expected: nonzero exit with `ModuleNotFoundError: No module named 'vfe3.timing'` recorded for the two
new tests.

- [ ] **Step 3: Add the minimal timing implementation**

Implement immutable samples and a `TrainingTimer` with a `1e-9` rate floor. Resolve event and stream
factories only when `device.type == "cuda"`. `sample_train_window()` synchronizes the last clean end
event and clears only clean-step state. `sample_pipeline_window()` records and synchronizes a
non-timing boundary event before reading the wall clock, but does not reset. `reset_pipeline_window()`
runs only after a successful CSV append.

```python
import time
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

import torch


_RATE_FLOOR_S = 1e-9


class _TimingEvent(Protocol):
    def record(self, stream: object) -> None: ...
    def synchronize(self) -> None: ...
    def elapsed_time(self, end_event: "_TimingEvent") -> float: ...


@dataclass(frozen=True)
class TrainTimingSample:
    train_step_ms_mean:      float
    train_steps_per_s:       float
    train_step_tokens_per_s: float


@dataclass(frozen=True)
class PipelineTimingSample:
    pipeline_tokens_per_s: float
    wall_clock_s:           float


class TrainingTimer:
    def __init__(
        self,
        device: torch.device,

        *,
        clock:               Optional[Callable[[], float]]              = None,
        cuda_event_factory:  Optional[Callable[..., object]]            = None,
        cuda_stream_factory: Optional[Callable[[torch.device], object]] = None,
    ) -> None:
        self._device = device
        self._clock = time.perf_counter if clock is None else clock
        self._is_cuda = device.type == "cuda"
        self._event_factory: Optional[Callable[..., _TimingEvent]] = None
        self._stream: Optional[object] = None
        if self._is_cuda:
            self._event_factory = torch.cuda.Event if cuda_event_factory is None else cuda_event_factory
            stream_factory = torch.cuda.current_stream if cuda_stream_factory is None else cuda_stream_factory
            self._stream = stream_factory(device)
        self._train_origin = self._clock()
        self._pipeline_origin = self._train_origin
        self._active_step: Optional[tuple[_TimingEvent, _TimingEvent]] = None
        self._event_pairs: list[tuple[_TimingEvent, _TimingEvent]] = []
        self._cpu_step_origin: Optional[float] = None
        self._cpu_train_seconds = 0.0
        self._train_steps = 0
        self._train_tokens = 0
        self._pipeline_tokens = 0

    def start_step(self) -> None:
        if self._is_cuda:
            assert self._event_factory is not None
            start = self._event_factory(enable_timing=True)
            end = self._event_factory(enable_timing=True)
            start.record(self._stream)
            self._active_step = (start, end)
        else:
            self._cpu_step_origin = self._clock()

    def finish_step(
        self,
        *,
        n_tokens: int,
    ) -> None:
        if self._is_cuda:
            assert self._active_step is not None
            start, end = self._active_step
            end.record(self._stream)
            self._event_pairs.append((start, end))
            self._active_step = None
        else:
            assert self._cpu_step_origin is not None
            self._cpu_train_seconds += self._clock() - self._cpu_step_origin
            self._cpu_step_origin = None
        self._train_steps += 1
        self._train_tokens += int(n_tokens)
        self._pipeline_tokens += int(n_tokens)

    def sample_train_window(self) -> TrainTimingSample:
        if self._is_cuda:
            self._event_pairs[-1][1].synchronize()
            total_seconds = sum(
                float(start.elapsed_time(end)) for start, end in self._event_pairs
            ) / 1000.0
        else:
            total_seconds = self._cpu_train_seconds
        denominator = max(total_seconds, _RATE_FLOOR_S)
        sample = TrainTimingSample(
            train_step_ms_mean=(total_seconds * 1000.0) / self._train_steps,
            train_steps_per_s=self._train_steps / denominator,
            train_step_tokens_per_s=self._train_tokens / denominator,
        )
        self._event_pairs.clear()
        self._cpu_train_seconds = 0.0
        self._train_steps = 0
        self._train_tokens = 0
        return sample

    def sample_pipeline_window(self) -> PipelineTimingSample:
        if self._is_cuda:
            assert self._event_factory is not None
            boundary = self._event_factory(enable_timing=False)
            boundary.record(self._stream)
            boundary.synchronize()
        now = self._clock()
        elapsed = now - self._pipeline_origin
        return PipelineTimingSample(
            pipeline_tokens_per_s=self._pipeline_tokens / max(elapsed, _RATE_FLOOR_S),
            wall_clock_s=now - self._train_origin,
        )

    def reset_pipeline_window(self) -> None:
        self._pipeline_origin = self._clock()
        self._pipeline_tokens = 0
```

Do not add model, optimizer, or artifact dependencies to this module.

- [ ] **Step 4: Run the timer tests and inspect machine-readable counts**

Run the same command with `C:\tmp\vfe3-p6-timer-green.xml`. Expected: 2 tests, 0 failures, 0
errors. Read the XML attributes before reporting the count.

- [ ] **Step 5: Commit the timing component**

```powershell
git add -- vfe3/timing.py tests/test_training_timing_20260711.py
git commit -m "feat(perf): add reporting-window timer"
```

### Task 2: Wire timing into the training loop with aligned callback windows

**Files:**

- Modify: `tests/test_training_timing_20260711.py`
- Modify: `vfe3/train.py:14-44, 1027-1321`

**Interfaces:**

- Consumes the five `TrainingTimer` operations from Task 1.
- Produces the three explicit timing columns plus the legacy alias and post-callback `wall_clock_s`.

- [ ] **Step 1: Add failing training-loop integration regressions**

Add tiny CPU configuration and direct-batch helpers copied locally from existing tests. Add four
regressions: callback/checkpoint/append attribution with a manual clock; mixed log/eval rectangular
CSV rows; CPU training on a mocked CUDA-capable host that forbids every CUDA helper; and a silent
route whose timer factory raises if constructed.

The attribution test must wrap real `train_step`, `maybe_save_best`, `save_checkpoint`, and
`log_metrics` so their virtual durations are 2, 10, 20, and 100 seconds. With 8 tokens per step it
must assert:

```python
assert [row["train_step_ms_mean"] for row in artifacts.history] == pytest.approx([2000.0, 2000.0])
assert [row["train_step_tokens_per_s"] for row in artifacts.history] == pytest.approx([4.0, 4.0])
assert [row["pipeline_tokens_per_s"] for row in artifacts.history] == pytest.approx([0.25, 0.25])
assert [row["wall_clock_s"] for row in artifacts.history] == pytest.approx([32.0, 164.0])
assert all(row["tokens_per_s"] == row["pipeline_tokens_per_s"] for row in artifacts.history)
```

The 100-second CSV append is absent from both 32-second pipeline windows but present in cumulative
wall time before the second row. The CPU-on-CUDA-host test must mock `torch.cuda.is_available()` to
return true and make `Event`, `current_stream`, `synchronize`, `max_memory_allocated`, and
`reset_peak_memory_stats` raise. The silent test must patch `vfe3.train.TrainingTimer` with an
exploding constructor and run with logging, evaluation, and artifacts disabled.

- [ ] **Step 2: Run the integration nodes and verify the red state**

Run:

```powershell
python -m pytest tests/test_training_timing_20260711.py -k "train or mixed or silent" --junitxml=C:\tmp\vfe3-p6-loop-red.xml
```

Expected: failures because `train.py` does not import or invoke `TrainingTimer`, does not emit the
new columns, samples callbacks in the wrong window, and uses global CUDA availability for memory.

- [ ] **Step 3: Make the surgical training-loop change**

Remove the now-unused `time` import and import `TrainingTimer`. After resume iterator replay, create
the timer only when logging can occur or an artifact-backed eval row can occur:

```python
timing_enabled = bool(log_interval) or (
    artifacts is not None and bool(eval_interval) and val_loader is not None
)
timer = TrainingTimer(device) if timing_enabled else None
```

Bracket only the existing call:

```python
if timer is not None:
    timer.start_step()
losses.append(train_step(model, optimizer, scheduler, tokens, targets,
                         grad_clip=grad_clip, grad_accum_steps=cfg.grad_accum_steps,
                         scaler=scaler, metrics_out=step_metrics, status_out=step_status))
if timer is not None:
    timer.finish_step(n_tokens=tokens.numel())
```

On `do_log or do_csv`, call `sample_train_window()`. Use its `train_steps_per_s` in the console and
rename the label to `train it/s`. Keep peak-memory sampling at this location, but guard on
`device.type == "cuda"` and pass `device` into both CUDA memory helpers.

Move the complete periodic checkpoint block to after evaluation/EMA restoration and before ordinary
row construction. Build all ordinary fields, then sample and append timing fields together:

```python
pipeline_timing = timer.sample_pipeline_window()
row["train_step_ms_mean"]      = train_timing.train_step_ms_mean
row["train_step_tokens_per_s"] = train_timing.train_step_tokens_per_s
row["pipeline_tokens_per_s"]   = pipeline_timing.pipeline_tokens_per_s
row["tokens_per_s"]            = pipeline_timing.pipeline_tokens_per_s
row["wall_clock_s"]            = pipeline_timing.wall_clock_s
artifacts.log_metrics(row)
timer.reset_pipeline_window()
```

Delete `win_t0`, `win_i0`, `train_t0`, `rate`, `toks_per_s`, and the old throughput/window reset.
Leave `_step_indices()` and tqdm untouched.

- [ ] **Step 4: Run the integration file and unchanged silent-equivalence test**

```powershell
python -m pytest tests/test_training_timing_20260711.py tests/test_train.py::test_silent_and_logging_paths_are_bitwise_identical --junitxml=C:\tmp\vfe3-p6-loop-green.xml
```

Expected: all CPU nodes pass; the real-CUDA node introduced in Task 3 is not present yet.

- [ ] **Step 5: Commit the training-loop integration**

```powershell
git add -- vfe3/train.py tests/test_training_timing_20260711.py
git commit -m "feat(perf): split train-step and pipeline rates"
```

### Task 3: Pin schema compatibility and real CUDA execution

**Files:**

- Modify: `tests/test_training_timing_20260711.py`
- Modify: `tests/test_run_diagnostics_2026_06_13.py:160-180`
- Modify: `check_gpu_tests.py`

**Interfaces:**

- Consumes the persisted timing fields from Task 2.
- Produces a CUDA smoke node reachable from the existing click-to-run GPU verifier.

- [ ] **Step 1: Extend the canonical schema regression**

Require `train_step_ms_mean`, `train_step_tokens_per_s`, and `pipeline_tokens_per_s` alongside the
legacy `tokens_per_s`, then assert every populated row satisfies exact alias equality and finite
explicit timing values.

- [ ] **Step 2: Add the real-CUDA smoke test and verifier route**

Add a `pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")` test that runs two
tiny CUDA training steps with `log_interval=1`, no evaluation, and real `RunArtifacts`. Assert both
rows contain finite nonnegative step milliseconds and wall time, positive clean and pipeline token
rates, and exact legacy alias equality. Add its node ID to `check_gpu_tests.py`'s T6 list:

```python
"tests/test_training_timing_20260711.py::test_real_cuda_training_timing_smoke",
```

- [ ] **Step 3: Run focused CPU compatibility verification**

```powershell
python -m pytest tests/test_training_timing_20260711.py tests/test_exp8_buildout.py tests/test_run_diagnostics_2026_06_13.py tests/test_checkpoint_resume.py tests/test_run_artifacts.py::test_train_with_artifacts_writes_files tests/test_report.py::test_metrics_csv_logs_at_log_cadence tests/test_train.py::test_silent_and_logging_paths_are_bitwise_identical --junitxml=C:\tmp\vfe3-p6-focused-cpu.xml
```

Expected: zero failures and errors; the CUDA smoke is skipped under a CPU-only interpreter. Read the
actual counts from XML.

- [ ] **Step 4: Run the dedicated RTX 5090 verification**

```powershell
$env:VFE3_TEST_DEVICE = "cuda"
& "C:\anaconda\python.exe" -m pytest tests/test_training_timing_20260711.py::test_real_cuda_training_timing_smoke tests/test_run_diagnostics_2026_06_13.py::test_metrics_csv_has_tier1_columns_and_is_rectangular tests/test_exp8_buildout.py::test_wall_clock_column_in_metrics --junitxml=C:\tmp\vfe3-p6-focused-cuda.xml
```

Expected: zero failures, errors, or skips. Record the interpreter, PyTorch, CUDA, and device identity
with the exact test counts; do not impose a throughput threshold.

- [ ] **Step 5: Commit the compatibility and CUDA coverage**

```powershell
git add -- tests/test_training_timing_20260711.py tests/test_run_diagnostics_2026_06_13.py check_gpu_tests.py
git commit -m "test(perf): pin P6 timing boundaries"
```

### Task 4: Verify, review, document, push, and merge

**Files:**

- Modify: `docs/2026-07-11-edits.md`
- Modify: `docs/superpowers/specs/2026-07-11-p6-training-timing-design.md`

- [ ] **Step 1: Run source checks and the complete default CPU suite**

```powershell
python -m compileall vfe3 tests/test_training_timing_20260711.py
git diff --check
python -m pytest --junitxml=C:\tmp\vfe3-p6-full-cpu.xml
```

Expected: compilation succeeds, the diff check is empty, and JUnit records zero failures/errors.

- [ ] **Step 2: Request independent code review**

Dispatch a reviewer against the approved spec and all branch commits. Resolve every Critical or
Important finding with a failing regression before changing production code, then rerun the focused
CPU and CUDA commands affected by the repair.

- [ ] **Step 3: Record exact implementation and verification evidence**

Update the existing P6 section in `docs/2026-07-11-edits.md` with the created and modified files,
the two reporting-only CUDA synchronization boundaries, the checkpoint/row ordering, the explicit
metric semantics, the CPU and CUDA environments, and the exact JUnit attributes read from each XML.
Change the design status to `implemented and verified` only after all required commands are green.

- [ ] **Step 4: Commit the final documentation and remove task-owned XML files**

```powershell
git add -- docs/2026-07-11-edits.md docs/superpowers/specs/2026-07-11-p6-training-timing-design.md
git commit -m "docs: record P6 timing verification"
```

Delete only `C:\tmp\vfe3-p6-*.xml` files created by this plan after their counts are recorded.

- [ ] **Step 5: Complete the mandatory Git lifecycle**

Fetch and inspect `origin/main`. Rebase the isolated task branch if the remote advanced, rerun the
affected verification, push `codex/p6-training-timing-20260711`, then fast-forward remote `main` from
the verified task HEAD. Fetch again and inspect `origin/main`. Do not fast-forward the user's live
checkout while `ablation.py`, `train_vfe3.py`, or any other user WIP is dirty. Remove the temporary
worktree and local task branch only after the remote branch and `main` are confirmed.

Final reporting must include the task commits, pushed branch, resulting `origin/main` SHA,
machine-read CPU/CUDA counts, worktree removal, and the actual live `git status --short` with the
remaining files identified as user-owned WIP.
