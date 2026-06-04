# 2026-06-04 — Live per-step it/s progress bar in training

## What changed

`vfe3/train.py` now drives the M-step loop in `train()` through an optional `tqdm` progress
bar whose built-in rate readout reports live `it/s` on every step, matching the per-step
throughput display of the VFE_2.0 trainer. Because both click-to-run entry points
(`train_vfe3.py` and `ablation.py`) and the deprecated `run_training` helper all train through
this single `train()` chokepoint, the live readout appears everywhere training runs, with no
change to either entry-point file.

Previously the only throughput number was the windowed `it/s` printed inside the
formatted `Step .../... | Loss | CE | H(b) | it/s | PPL` line, which is emitted once every
`log_interval` steps (one hundred for `train_vfe3.py`, one thousand for `ablation.py`). That
windowed average is retained unchanged; the new bar supplies the live, per-step rate alongside
it.

## How it is wired

`tqdm` is imported at module scope behind a `try/except ImportError` so it remains an optional
dependency: when it is absent the loop falls back to a plain `range(n_steps)` and the periodic
formatted lines still print exactly as before. The bar is created only when `log_interval` is
truthy, which preserves the documented silent path verbatim — with `log_interval` falsy the
loop wraps `contextlib.nullcontext()`, draws no extra RNG, runs no extra forward, and prints
nothing, so `test_silent_and_logging_paths_are_bitwise_identical` continues to hold.

The loop is iterated through a small generator (`_step_indices`) that holds
`tqdm.contrib.logging.logging_redirect_tqdm` open for the whole loop by suspending at its
`yield` inside the `with` block. This routes the periodic `logger.info` lines (and the
validation block, and any logging done inside `RunArtifacts`) above the bar rather than letting
them interleave with it on `stderr`, and it closes the bar on normal completion or on an
exception propagating out of the loop body. The loop body itself is untouched, so the numeric
training path is unchanged.

The bar is constructed with `ascii=True`. The default `tqdm` fill glyph is the Unicode block
`U+2588`, which the cp1252 codec of a Windows console cannot encode and which would raise a
`UnicodeEncodeError` mid-run; the `" #"` ASCII rendering is safe on any console. V3's log lines
are already ASCII (`H(b)`, not `H(beta)`), so the bar glyph was the only remaining encoding
risk, and a single keyword argument removes it without the global `stdout`/`stderr`
reconfiguration the VFE_2.0 trainer used.

## Verification

`python -m py_compile vfe3/train.py` passes. A CPU smoke run on the synthetic period-3 stream
confirmed that with `log_interval=1` the bar shows live `it/s` and the formatted lines render
above it, and that with `log_interval=0` nothing is printed. `tests/test_train.py` passes in
full (fifteen tests, zero failures or errors, read from the JUnit XML; the lone `xpassed` is
the pre-existing non-strict cutover xfail and is unrelated to this change).
