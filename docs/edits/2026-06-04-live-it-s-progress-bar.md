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


# 2026-06-04 — `lambda_beta`: belief-coupling weight (VFE_2.0 `lambda_align` parity)

## What changed

A scalar weight `lambda_beta` on the entire belief-coupling block of the free energy was added,
both as a constant config field and as an optional learned scalar. It is the V3 name for
VFE_2.0's `lambda_align`:

```
F = Sum_i [ alpha_i D(q_i||p_i) (+R)
          + lambda_beta ( Sum_j beta_ij E_ij + tau Sum_j beta_ij log(beta_ij/pi_ij) )
          - ell_i ].
```

At `lambda_beta = 1.0` the path is byte-identical to the previous canonical/pure F; values away
from 1 reweight the attention/coupling contribution against the `alpha` self-term and the
likelihood. It scales the belief (q) channel only; the model-channel (`gamma`) block is
untouched. VFE_2.0's separate surrogate-only `lambda_soft`/`lambda_softmax` knob was deliberately
not ported. The full design and its correctness invariant are recorded in
`docs/superpowers/specs/2026-06-04-lambda-beta-coupling-weight-design.md`.

## Correctness invariant

`lambda_beta` scales the `coupling` and `entropy` terms by the same factor and leaves
`beta = softmax(-E/tau)` untouched (no lambda inside the softmax). Because scaling the whole
block does not move its argmin over the simplex, `beta*` stays stationary, so the envelope
identity still gives `d/dtheta[lambda_beta (coupling+entropy)] = lambda_beta Sum_j beta* dE/dtheta`.
The analytic kernel therefore scales only its `pair` term by `lambda_beta` while the autograd
oracle differentiates `lambda_beta F`, and the two remain in exact agreement.

## How it is wired

Constant path (rides `cfg` like `mass_phi`): the new field `lambda_beta: float = 1.0` (validated
`>= 0`) is threaded `model -> vfe_stack -> vfe_block -> e_step -> e_step_iteration` and consumed
in three places — `free_energy` (scales `coupling`+`entropy`, which covers the oracle gradient,
the F monitor, and diagnostics), `_diag_kl_filtering_kernel` via `belief_gradients` (scales the
`pair_mu`/`pair_sig` terms, leaving the self terms unscaled), and `phi_alignment_loss` (scales
the coupling block but not the `mass_phi` penalty, so the effective phi step is
`e_phi_lr * lambda_beta * grad`).

Learnable path (mirrors `log_alpha`): the new field `learnable_lambda_beta: bool = False`; when
True, `VFEModel.__init__` creates `log_lambda_beta = nn.Parameter(zeros())` (init 0 ->
`lambda_beta = 1.0`), the model threads a single effective `lambda_beta` (the constant
`cfg.lambda_beta`, or the live `exp(log_lambda_beta)` tensor) through the same chain, and the
M-step CE backpropagates to `log_lambda_beta` through the unrolled E-step (the mu/sigma path
only, since `grad_phi` is detached — exactly the signal path `log_alpha` already has).
`build_optimizer` adds `log_lambda_beta` to a group at `m_phi_lr`, and the same `detach_e_step`
footgun warning is emitted.

Learnable diagnostics: on a learnable run the periodic-eval block in `train.py` records
`lambda_beta = exp(log_lambda_beta)` in `metrics.csv` (the column appears only on learnable
runs, keeping the CSV rectangular), and `run_artifacts._save_figures` writes `lambda_beta.png`
from the logged history (the same conditional-trajectory pattern as `holonomy`). The figure is
produced inside `finalize_run`, which `train_vfe3.py` calls but `ablation.py` cells do not, so
(by user decision) a full `train_vfe3.py` learnable run yields both the CSV column and the PNG
while an ablation `learnable` cell yields the CSV column only. `metrics.free_energy_terms` also
gains a `lambda_beta` argument so the monitored `total`/`free_energy_total` is the scaled F
`self + lambda_beta (belief_coupling + entropy)` (the raw component columns are unchanged).

Sweeps: `ablation.py` gains a numeric `lambda_beta` sweep (`[0.25, 0.5, 1.0, 2.0, 4.0]`, added to
`SWEEP_ORDER`) and a multi-arm `learnable_lambda_beta` sweep (`constant` vs `learnable`); both
new fields are added to `BASELINE_CONFIG` and to the `train_vfe3.py` click-to-run config.

## Verification

The gate test `tests/test_lambda_beta.py` (19 tests) pins kernel == oracle at
`lambda_beta in {0.5, 2.0}` (single-block and per-head), the gradient being affine in
`lambda_beta` (catching a lambda-into-softmax leak), `lambda_beta = 1.0` byte-identical to the
default, the learnable parameter's gradient flow / freeze-under-detach / optimizer coverage, and
config rejection of a negative value. The full suite is green (525 tests, zero failures or errors,
the lone `xpassed` the pre-existing cutover xfail), and a CPU end-to-end learnable run confirmed
the `lambda_beta` column in `metrics.csv` (moving away from 1.0) and the `lambda_beta.png` figure.
