# Design spec: fp16 training GradScaler (close the silent-mistrain footgun)

**Date:** 2026-06-08
**Status:** approved design, pre-implementation
**Roadmap:** lifecycle-audit Tier-2 #5 (`docs/audits/audit-2026-06-07-lifecycle-multiagent.md`)
**Scope:** contained — `vfe3/train.py` only, plus tests.

## Motivation

`amp_dtype='fp16'` is accepted at config and wraps the E-step in `autocast(fp16)`
(`model.py:293`), but the training loop (`train.py:257-280`) does an UNSCALED `loss.backward()`
-> `optimizer.step()`. Through the deep unrolled E-step, fp16 gradients underflow to zero, so
fp16 silently mistrains. This wires a `torch.amp.GradScaler` so fp16 trains correctly. (bf16 and
fp32 need no scaler; on the user's 5090 **bf16 is the recommended mixed-precision path** — this fix
closes the fp16 footgun, it is not the default training path.)

## Hard constraint: the covariance stays fp32

`sigma` / the SPD linear algebra (Cholesky, `eigh`, log/exp maps, retraction) MUST run in fp32.
fp16's narrow range and ~3-digit mantissa destroy positive-definiteness and NaN the Cholesky/`eigh`
backward under near-degenerate eigenvalues (obs 16360). This is already enforced structurally: the
E-step's SPD / `matrix_exp` islands and the decode+CE run under `autocast(enabled=False)` fp32 guards
"regardless of `amp_dtype`", so they hold under fp16 exactly as under bf16. The GradScaler is
ORTHOGONAL: it scales the scalar loss to prevent fp16 gradient *underflow* on the fp16-computed parts
(mean / energy), and gradient-magnitude scaling does not change which ops ran in which precision.
`sigma` never enters fp16. The design VERIFIES this with a test (fp16 forward keeps logits/`sigma`
finite), it does not merely assume it.

## Design

All in `vfe3/train.py`.

- **Build one scaler in `train()`** (near the optimizer build, `:375`):
  `scaler = torch.amp.GradScaler(enabled=(cfg.amp_dtype == 'fp16'))` (device resolved from the run
  device; if the modern `torch.amp.GradScaler` device kwarg is needed, pass it). Thread `scaler` into
  every `train_step(...)` call (`:440`).
- **`train_step` gains a `scaler` parameter** and uses it:
  - single-step path: `scaler.scale(loss).backward()` (was `loss.backward()`),
  - grad-accum path: `scaler.scale(loss_mb / grad_accum_steps).backward()` per microbatch,
  - before `clip_grad_norm_`: `scaler.unscale_(optimizer)` (so the clip sees real-magnitude grads),
  - `scaler.step(optimizer)` (was `optimizer.step()`), then `scaler.update()`.
  - `scheduler.step()` is unchanged. (Note: `scaler.step` may skip the optimizer step on an
    inf/nan-grad iteration; `scheduler.step()` still advances — acceptable, matches standard AMP
    recipes; the skip is rare and self-correcting via `update()` lowering the scale.)
- **Default / bf16 / fp32 are byte-identical:** `GradScaler(enabled=False)` makes `scale` return the
  loss unchanged, `unscale_` a no-op, and `step` a plain `optimizer.step()`. So the `amp_dtype != 'fp16'`
  paths (including the default `None`) are bitwise-unchanged. No config change is needed (`fp16` is
  already an accepted `amp_dtype`).

## Option A: support fp16 + the geometric gauge M-step

`m_phi_natural_grad=True` uses the custom `GaugeNaturalGradAdamW` (steps `phi` from `p.grad` outside
plain AdamW). `GradScaler.unscale_`/`step` operate on the optimizer's `param_groups` and read standard
unscaled `.grad`, so the gauge optimizer should compose with the scaler. The design SUPPORTS the
combination and pins it with a test that, under `amp_dtype='fp16'` + `m_phi_natural_grad=True`, the
gauge-frame tables actually move after a scaled step (i.e. the scaler does not silently no-op or break
the gauge update). No `__post_init__` rejection is added.

## Testing strategy

- **No-op / byte-identity (CPU):** with `enabled=False` (any non-fp16 `amp_dtype`), `train_step`
  produces the same parameter update as the pre-change unscaled path — the default/bf16/fp32 training
  step is unchanged.
- **fp16 forward keeps `sigma` fp32 (CPU):** an `amp_dtype='fp16'` forward yields finite logits/loss
  (no Cholesky/`eigh` NaN), confirming the SPD islands hold under fp16.
- **fp16 scaler actually scales + steps:** under `amp_dtype='fp16'` with `enabled=True`, a `train_step`
  advances the scaler scale and updates parameters (the gradients are not underflowed to zero). If
  `torch.amp.GradScaler` is not functional on CPU for this check, gate this test behind
  `torch.cuda.is_available()` (skip on CPU) and note it; do NOT loosen it to a wiring-only spy if a
  real scaled step is testable.
- **fp16 + `m_phi_natural_grad` gauge step (Option A):** the gauge-frame tables move under a scaled
  step (CUDA-gated if needed, same rationale).

## Out of scope (noted follow-ups)

- **Scaler state in the checkpoint.** `save_checkpoint`/`load_checkpoint` do not round-trip
  `scaler.state_dict()`, so a resumed fp16 run restarts the scale factor (it re-warms within a few
  steps — a brief transient, not a gradient-correctness defect). A clean fp16 resume would add a
  `scaler_state` field; deferred.
- **bf16 path:** unchanged and recommended on the 5090 (no scaler needed).
