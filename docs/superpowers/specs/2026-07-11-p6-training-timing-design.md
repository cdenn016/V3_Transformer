# Design: P6 Training and Pipeline Throughput Instrumentation

Date: 2026-07-11
Status: implemented and verified
Branch: `codex/p6-training-timing-20260711`
Base: `origin/main` at `81dd6ae12dd9774c6fe348646bd668580eb38701`
Worktree: `C:\tmp\V3_Transformer_p6_training_timing_20260711`

## Purpose

This change addresses P6 in
`docs/audits/deep-audit-and-wikitext103-performance-investigation-2026-07-09.md`.
The current throughput window begins before the training loop, is sampled after the training step,
and is reset before validation, sampling, figure generation, best-state persistence, and periodic
checkpoint persistence. A callback therefore enters the following row's denominator instead of the
row that caused it. The persisted `tokens_per_s` value is consequently neither isolated training-step
throughput nor a correctly aligned end-to-end pipeline rate.

The implementation will report two measurements with explicit boundaries. Clean training-step
timing will cover only the existing `train_step(...)` call. Pipeline timing will cover the complete
reporting window, including batch acquisition, host-to-device copies, diagnostics, the training
step, post-step hooks, console reporting, validation, sampling, figures, best-state persistence, and
periodic resumable checkpoints. The change is instrumentation only: it will not alter model
mathematics, optimizer behavior, random-number consumption, data order, configuration defaults, or
the theoretically pure paths.

## Measurement contract

The clean boundary begins immediately before `train_step(...)`, after any pre-step diagnostic replay,
and ends immediately after `train_step(...)` returns, before the Metropolis reflection sweep and EMA
update. The boundary encloses the forward pass, loss, backward pass, gradient handling, optimizer
update, and scheduler work already owned by `train_step`, while excluding batch loading, device
transfer, diagnostic replay, EMA, validation, generation, visualization, and persistence callbacks.
CPU timing measures the wall duration of that entire call. CUDA timing measures device elapsed time
on the current training stream between the two boundary events; it is not a portable wall-clock
measurement of every host-only action inside the call.

On CUDA, each completed training step records one start event and one end event on the resolved
training device's current stream. Finishing a step does not synchronize it. At a log or metrics-row
boundary, the latest clean-step end event is synchronized, after which the elapsed milliseconds for
every pair in that reporting window are summed. This supplies the clean rate for the existing
pre-callback console line. After callbacks, checkpointing, and ordinary row construction, a second
event is recorded and synchronized on that same stream before pipeline wall time is sampled. The
second boundary prevents asynchronous callback work from spilling into the next pipeline window.
Both synchronizations are event-specific and occur only on reporting steps; the implementation will
not call global `torch.cuda.synchronize()`. An operator who selects a reporting cadence of one step
therefore requests both reporting boundaries after every step. On CPU, both boundaries use
`time.perf_counter()` without CUDA calls. The device decision is based on the resolved training
device's `device.type`, never on the process-wide result of `torch.cuda.is_available()`. The current
training path does not dispatch unjoined auxiliary-stream work. If that changes, the auxiliary
streams must join the timed stream before the boundary event or the metric must be relabeled; the P6
helper will not claim to time unordered work on another stream.

The pipeline window begins after resume-data replay and immediately before the first ordinary batch
acquisition. After a persisted row, the next window begins only after that row has been appended.
Its numerator is the exact sum of `tokens.numel()` for the completed training steps, rather than the
last batch shape multiplied by the step count. Its denominator ends after all callbacks and any
periodic checkpoint caused by the reporting step, but immediately before the metrics row containing
the measurement is appended. The append itself is measurement bookkeeping and is excluded from both
the row being written and the next window. A checkpoint that occurs between reporting steps remains
inside the later reporting window; a checkpoint that coincides with a reporting step is completed
before that row's pipeline sample.

All ordinary row construction, including device-to-host scalar extraction, will occur before the
pipeline sample and will therefore be charged to the current pipeline window. Only insertion of the
already computed timing fields and the `log_metrics` append occur after sampling. Resetting the
pipeline origin after the append excludes that append from both adjacent rate windows.

Both rate denominators retain the existing `1e-9`-second floor so a timer with zero representable
duration cannot produce an infinite value. Every reporting window contains at least one completed
training step under the current loop contract, so the helper does not need a synthetic zero-step
sample.

## Timing component and data flow

A focused `TrainingTimer` component will live in `vfe3/timing.py`. It will own the CPU timestamps or
CUDA event pairs, the clean-window step and token counts, the independent pipeline token count, the
pipeline window origin, and the cumulative training origin used by `wall_clock_s`. Its public
operations will express five transitions: start a training step, finish that step with its token
count, sample and clear the clean training-step window, sample the pipeline window, and reset the
pipeline window after a successful metrics append.
Clock and CUDA-event construction will have narrow test seams so deterministic unit tests do not
sleep or require a GPU.

The training loop will create the timer only when a console throughput report or an artifact metrics
row can be emitted. The documented silent route, with no log cadence and no artifact row cadence,
will construct no timer and invoke no timing operation. The helper will not inspect the model, move
tensors, draw random values, or mutate optimizer state.

At each step, the loop will retain its present batch, diagnostic, and `train_step` order. The timer's
clean start and finish operations will bracket only `train_step`. When `do_log` or `do_csv` is true,
the loop will take one clean-window snapshot before callbacks. The console line will use the
snapshot's clean steps-per-second value and label it `train it/s`, so it cannot be mistaken for
pipeline throughput. Evaluation and artifact callbacks then run in their existing order. Periodic
checkpoint persistence will move ahead of the final pipeline sample so its cost belongs to the row
that caused it. The ordinary metrics row will then be constructed, the pipeline sample will supply
its final timing fields, the row will be appended, and the pipeline window will reset.

The built-in `tqdm` display will remain unchanged. It is an interactive whole-loop progress estimate,
not a persisted performance measurement. The P6 contract applies to the explicit console field and
CSV columns.

## Persisted metrics and compatibility

Every artifact metrics row will add `train_step_ms_mean`, the arithmetic mean clean step duration in
milliseconds; `train_step_tokens_per_s`, the exact clean-window token count divided by summed clean
step time; and `pipeline_tokens_per_s`, the exact pipeline-window token count divided by pipeline
wall time. The existing `tokens_per_s` column will remain as a compatibility alias whose value is
identical to `pipeline_tokens_per_s`. Existing report readers therefore continue to find their
legacy field, while new analysis can select the measurement it actually needs. This is schema
compatibility, not semantic continuity: pre-P6 `tokens_per_s` values used the contaminated old
window, so direct historical comparisons across the instrumentation change are invalid unless the
analysis separates runs by code version and metric definition.

The cumulative `wall_clock_s` sample will move to the same post-callback, post-checkpoint boundary as
the pipeline sample. It will include all prior training and callback work and exclude only the
current metrics append, which cannot be represented inside the row being appended. All new fields
will be present from the first row for every run that writes metrics, preserving the rectangular CSV
schema enforced by `RunArtifacts.log_metrics` across mixed log and evaluation cadences.

`peak_mem_mb` is outside the P6 throughput redefinition. Its established sampling point and meaning
will remain unchanged. The branch explicitly accepts one companion correctness fix at that adjacent
seam: its device guard will use the resolved training device so a CPU run on a CUDA-capable host
neither queries nor resets unrelated CUDA memory statistics.

## Failure and edge behavior

Existing best-effort validation diagnostics, generation, and figure error handling will remain in
place. A fatal periodic-checkpoint error will still propagate. Because checkpoint duration belongs
to the current pipeline window, a checkpoint on a reporting step must finish before that row is
written; if it fails, that incomplete reporting row will not be appended. The reverse failure is
also possible after the reordering: a checkpoint can be durable while a later row-construction or
metrics-append failure leaves no corresponding CSV row. This accepted crash-consistency trade-off is
the consequence of assigning checkpoint time to the row that caused it; checkpoint recovery remains
authoritative over the observational metrics file.

CPU runs remain fully supported even when CUDA is installed. CUDA runs create events only for the
actual resolved CUDA device. The implementation will not synchronize when an individual step merely
finishes, will not add synchronization on nonreporting steps, will not call global device
synchronization, and will not impose a performance threshold that could make correctness tests
sensitive to machine load. Reporting steps use the clean-step and post-callback event boundaries
described above because omitting the latter would undercount asynchronous callback work.

## Test and verification design

Implementation will proceed test-first. A deterministic CPU unit test will drive a manual clock
through several unequal step intervals and assert the exact mean milliseconds, clean steps per
second, token rate, pipeline rate, and reset behavior. A fake CUDA-event test will assert event
recording order, millisecond conversion, exact aggregation, the two event-specific reporting
synchronizations, and no synchronization at individual step completion. A CPU-on-CUDA-host regression will
make CUDA helper calls raise and prove that a CPU timer and CPU training row never touch them.

A training-loop integration regression will assign deterministic virtual durations to a training
step and to callbacks. It will prove that callback time changes `pipeline_tokens_per_s` but does not
change `train_step_ms_mean` or `train_step_tokens_per_s`, and that the callback is charged to the row
that triggers it rather than the next row. A mixed log/evaluation cadence regression will assert
that every CSV row has the same field set, `tokens_per_s == pipeline_tokens_per_s`, and
`wall_clock_s` is sampled after the evaluation and persistence callbacks. A silent-path regression
will assert zero timing construction and calls when no timing consumer is active.

Existing diagnostics, artifact, and report regressions will be rerun to protect row cadence and
downstream compatibility. A small RTX 5090 smoke test will exercise the real CUDA-event route and
assert finite nonnegative timing values without enforcing a throughput target. Final pass counts
will be read from JUnit XML rather than inferred from console output.

## Scope exclusions and acceptance

This P6 branch will not implement asynchronous best-state writing, alter checkpoint contents, tune
CUDA kernels, change transport settings, redesign the progress bar, revise peak-memory semantics, or
run a long performance benchmark. Those are separate performance tasks.

The implementation is accepted when clean timing brackets only `train_step`, pipeline timing assigns
callbacks and checkpoints to the correct reporting window, CUDA synchronization is event-specific
and occurs only at the two explicit reporting boundaries, CPU execution makes no CUDA calls, all
metrics rows remain rectangular, the legacy field is preserved as the documented alias, and focused
CPU plus CUDA verification passes without a model-output or random-stream change.

Implementation does not begin until this written specification is committed, self-reviewed, and
accepted by the user. A detailed test-driven implementation plan follows that acceptance.
