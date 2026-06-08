# fp16 training GradScaler — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire a `torch.amp.GradScaler` into the training loop so `amp_dtype='fp16'` trains correctly (today fp16 gradients underflow through the unrolled E-step and silently mistrain), while keeping the default/bf16/fp32 paths byte-identical and the covariance in fp32.

**Architecture:** Build one `GradScaler(enabled=(cfg.amp_dtype=='fp16'))` in `train()`, thread it into `train_step`, and replace the raw `loss.backward()`/`optimizer.step()` with `scaler.scale(loss).backward()` / `scaler.unscale_(optimizer)` (before grad-clip) / `scaler.step(optimizer)` / `scaler.update()`. `enabled=False` is a documented no-op so non-fp16 paths are bitwise-unchanged. The SPD/`sigma` islands already run fp32 under `autocast(enabled=False)` regardless of `amp_dtype`, so the scaler is orthogonal to the covariance.

**Tech Stack:** Python, PyTorch (`torch.amp.GradScaler`), pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-06-08-fp16-gradscaler-design.md`

---

## File structure
- `vfe3/train.py` — `train_step` gains an optional `scaler` param and uses it; `train()` builds the scaler (gated on `amp_dtype=='fp16'`) and threads it in.
- `tests/test_fp16_gradscaler.py` — new test module.

No other files. (No config change: `amp_dtype='fp16'` is already accepted.)

---

## Task 1: Wire the GradScaler into `train_step` + `train()` (no-op-preserving)

**Files:**
- Modify: `vfe3/train.py` (`train_step` `:225-280`; `train()` scaler build near `:375`/`:404` and the `train_step(...)` call at `:440`)
- Test: `tests/test_fp16_gradscaler.py`

- [ ] **Step 1: Write the failing tests** — create `tests/test_fp16_gradscaler.py`:

```python
r"""fp16 training GradScaler (close the silent-mistrain footgun). Spec:
docs/superpowers/specs/2026-06-08-fp16-gradscaler-design.md.

The default / bf16 / fp32 paths must be byte-identical (scaler enabled=False is a no-op);
sigma stays fp32 under fp16 (no Cholesky/eigh NaN).
"""

import copy

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel
from vfe3.train import build_optimizer, train_step


def _tiny_cfg(**overrides) -> VFE3Config:
    base = dict(vocab_size=8, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=1)
    base.update(overrides)
    return VFE3Config(**base)


def _sched(opt):
    return torch.optim.lr_scheduler.LambdaLR(opt, lambda _step: 1.0)


def _batch(cfg, b=2, n=4):
    tok = torch.randint(0, cfg.vocab_size, (b, n))
    tgt = torch.randint(0, cfg.vocab_size, (b, n))
    return tok, tgt


def test_gradscaler_disabled_is_byte_identical_to_unscaled_step():
    # amp_dtype=None -> scaler enabled=False -> the wired train_step must produce the SAME
    # parameter update as the original unscaled loss.backward()/optimizer.step() path.
    cfg = _tiny_cfg()  # amp_dtype defaults to None
    torch.manual_seed(0); mA = VFEModel(cfg)
    torch.manual_seed(0); mB = VFEModel(cfg)
    tok, tgt = _batch(cfg)

    # A: through the (newly wired) train_step with no scaler passed (default-disabled).
    optA = build_optimizer(mA, cfg); schA = _sched(optA)
    train_step(mA, optA, schA, tok, tgt, grad_clip=1.0)

    # B: the original hand-rolled unscaled step.
    optB = build_optimizer(mB, cfg)
    optB.zero_grad(set_to_none=True)
    _, lossB, _ = mB(tok, tgt)
    lossB.backward()
    torch.nn.utils.clip_grad_norm_(mB.parameters(), 1.0)
    optB.step()

    for (na, pa), (nb, pb) in zip(mA.named_parameters(), mB.named_parameters()):
        assert torch.equal(pa, pb), f"param {na} diverged from the unscaled reference"


def test_fp16_forward_keeps_logits_and_sigma_finite():
    # The SPD/sigma islands must stay fp32 under amp_dtype='fp16' (no Cholesky/eigh NaN).
    torch.manual_seed(0)
    m = VFEModel(_tiny_cfg(amp_dtype="fp16"))
    tok = torch.randint(0, m.cfg.vocab_size, (2, 4))
    lg = m(tok)
    assert torch.isfinite(lg).all()
```

- [ ] **Step 2: Run to verify the first test fails (signature not wired) / second passes**

Run: `pytest tests/test_fp16_gradscaler.py -v`
Expected: `test_gradscaler_disabled_is_byte_identical_to_unscaled_step` may already PASS if `train_step` happens to match (the wiring is what we are adding) — but more importantly run it to establish the baseline. `test_fp16_forward_keeps_logits_and_sigma_finite` should PASS already (it tests existing forward behavior; if it FAILS, the sigma-fp32 islands do NOT hold under fp16 — STOP and report, because that is a real pre-existing bug this fix must surface).

- [ ] **Step 3: Wire the scaler into `train_step`**

In `vfe3/train.py`, change the `train_step` signature to add a keyword-only `scaler` param (Optional, default None so existing direct callers stay byte-identical):

```python
def train_step(
    model:     VFEModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    tokens:    torch.Tensor,             # (B, N) input token ids
    targets:   torch.Tensor,             # (B, N) next-token ids (-100 = ignore)

    *,
    grad_clip:        float = 1.0,
    grad_accum_steps: int   = 1,
    scaler:           Optional['torch.amp.GradScaler'] = None,
) -> float:
```

Inside the body, resolve a disabled scaler when none is passed, scale every backward, unscale before the clip, and step via the scaler:

```python
    # A disabled scaler is a documented no-op (scale -> identity, unscale_ -> nothing,
    # step -> optimizer.step()), so scaler=None keeps this path byte-identical to the unscaled loop.
    _scaler = scaler if scaler is not None else torch.amp.GradScaler(
        device=tokens.device.type, enabled=False)

    optimizer.zero_grad(set_to_none=True)
    if grad_accum_steps == 1:
        _, loss, _ = model(tokens, targets)
        _scaler.scale(loss).backward()
        step_loss = float(loss.detach())
    else:
        if tokens.shape[0] % grad_accum_steps != 0:
            raise ValueError(
                f"grad_accum_steps={grad_accum_steps} must divide the batch size "
                f"{tokens.shape[0]} for equal microbatches; got remainder "
                f"{tokens.shape[0] % grad_accum_steps}."
            )
        tok_chunks = torch.chunk(tokens, grad_accum_steps, dim=0)
        tgt_chunks = torch.chunk(targets, grad_accum_steps, dim=0)
        step_loss = 0.0
        for tok_mb, tgt_mb in zip(tok_chunks, tgt_chunks):
            _, loss_mb, _ = model(tok_mb, tgt_mb)
            _scaler.scale(loss_mb / grad_accum_steps).backward()
            step_loss += float(loss_mb.detach()) / grad_accum_steps
    if grad_clip is not None and grad_clip > 0:
        _scaler.unscale_(optimizer)                # unscale BEFORE clipping so the threshold is real
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    _scaler.step(optimizer)
    _scaler.update()
    scheduler.step()
    return step_loss
```

IMPORTANT subtlety: `scaler.unscale_(optimizer)` may be called at most once per step and only when grads exist. When `grad_clip` is falsy/<=0 the code above skips `unscale_`; that is fine — `scaler.step` will unscale internally. Verify `Optional` is imported in `train.py` (it is used elsewhere; if not, add `from typing import Optional`). If `torch.amp.GradScaler(device=..., enabled=False)` raises on this torch version (older API), fall back to `torch.cuda.amp.GradScaler(enabled=False)` and note the adaptation; the enabled=False instance must be a true no-op regardless.

- [ ] **Step 4: Build + thread the scaler in `train()`**

In `vfe3/train.py` `train()`, after `device` is resolved (around `:404`, `device = model.prior_bank.mu_embed.device`) and before the step loop, build the scaler:

```python
    # fp16 training needs loss scaling (gradients underflow through the unrolled E-step); bf16/fp32
    # do not. enabled=False is a no-op, so non-fp16 amp_dtype keeps this loop byte-identical.
    scaler = torch.amp.GradScaler(device=device.type, enabled=(cfg.amp_dtype == "fp16"))
```

and pass it into the `train_step(...)` call (`:440`):

```python
            losses.append(train_step(model, optimizer, scheduler, tokens, targets,
                                      grad_clip=grad_clip, grad_accum_steps=cfg.grad_accum_steps,
                                      scaler=scaler))
```

- [ ] **Step 5: Run to verify pass**

Run: `pytest tests/test_fp16_gradscaler.py -v` (NO extra `-q`). Both tests PASS.

- [ ] **Step 6: Regression**

Run: `pytest tests/test_train.py tests/test_amp.py --junitxml=out_t1.xml`; read tests/failures/errors from `out_t1.xml`; confirm 0 failures/errors (the default-path train tests must still pass, proving the no-op wiring). Delete `out_t1.xml` (no temp files; do not commit it).

- [ ] **Step 7: Commit**

```bash
git add vfe3/train.py tests/test_fp16_gradscaler.py
git commit -m "feat(train): wire fp16 GradScaler (no-op for default/bf16/fp32; fp16 trains correctly)"
```

---

## Task 2: fp16 actually scales + the gauge-optimizer composes with the scaler (Option A)

**Files:**
- Test: `tests/test_fp16_gradscaler.py` (append)

- [ ] **Step 1: Write the tests** — append:

```python
def _run_one_fp16_step(cfg):
    torch.manual_seed(0)
    m = VFEModel(cfg)
    opt = build_optimizer(m, cfg); sch = _sched(opt)
    scaler = torch.amp.GradScaler(device="cpu", enabled=True)
    before = {n: p.detach().clone() for n, p in m.named_parameters() if p.requires_grad}
    tok, tgt = _batch(cfg)
    train_step(m, opt, sch, tok, tgt, grad_clip=1.0, scaler=scaler)
    moved = any(not torch.equal(before[n], p) for n, p in m.named_parameters()
                if p.requires_grad and n in before)
    return m, scaler, moved


@pytest.mark.skipif(  # CPU GradScaler(enabled=True) may be a non-functional stub on some builds
    not hasattr(torch.amp.GradScaler(device="cpu", enabled=True), "scale"),
    reason="GradScaler unavailable")
def test_fp16_scaler_scales_and_updates_params():
    # Under amp_dtype='fp16' with an ENABLED scaler, a train_step must move parameters
    # (i.e. fp16 grads are scaled, not underflowed to zero).
    cfg = _tiny_cfg(amp_dtype="fp16")
    m, scaler, moved = _run_one_fp16_step(cfg)
    assert torch.isfinite(scaler.get_scale() if scaler.is_enabled() else torch.tensor(1.0))
    assert moved, "no parameter moved under the enabled fp16 scaler (gradients underflowed?)"


def test_fp16_with_gauge_natural_grad_steps_the_frame(monkeypatch):
    # Option A: fp16 + m_phi_natural_grad (custom GaugeNaturalGradAdamW) must compose with the
    # scaler -- the gauge-frame table must move under a scaled step (not silently no-op).
    cfg = _tiny_cfg(amp_dtype="fp16", m_phi_natural_grad=True, pos_phi="learned")
    torch.manual_seed(0)
    m = VFEModel(cfg)
    opt = build_optimizer(m, cfg); sch = _sched(opt)
    scaler = torch.amp.GradScaler(device="cpu", enabled=True)
    phi0 = m.prior_bank.phi_embed.detach().clone()
    tok, tgt = _batch(cfg)
    train_step(m, opt, sch, tok, tgt, grad_clip=1.0, scaler=scaler)
    assert not torch.equal(phi0, m.prior_bank.phi_embed), "gauge frame did not move under fp16 scaler"
```

- [ ] **Step 2: Run to see behavior**

Run: `pytest tests/test_fp16_gradscaler.py -k "scales_and_updates or gauge_natural_grad" -v`
Expected: ideally PASS on CPU. If CPU `GradScaler(enabled=True)` is a non-functional stub (no real scaling) so a test fails for an infrastructure reason rather than a logic bug, CONVERT that test to CUDA-gated by adding `@pytest.mark.skipif(not torch.cuda.is_available(), reason="fp16 scaler needs CUDA")` and building the model/scaler on `cuda`; do NOT delete the assertion or weaken it to a wiring-only spy. Note in your report which path each test took (CPU-real vs CUDA-gated) and WHY.

- [ ] **Step 3: Make them pass** (adapt gating per Step 2). If the gauge test reveals a REAL incompatibility (the scaler's inf-check or `unscale_` breaks `GaugeNaturalGradAdamW`), STOP and report BLOCKED with the traceback — that flips the design to option B (reject fp16+gauge), which needs user sign-off.

- [ ] **Step 4: Full suite**

Run: `pytest --junitxml=out_full.xml` (NO extra `-q`). Read tests/failures/errors/skipped from `out_full.xml`; confirm 0 failures/0 errors. Record the numbers. Delete `out_full.xml`.

- [ ] **Step 5: Changelog + commit**

Append a brief `## fp16 training GradScaler` section to `docs/edits/2026-06-08-decode-bias.md`: what it does (scaler wired into `train_step`/`train`, `enabled=(amp_dtype=='fp16')`), default/bf16/fp32 byte-identical (no-op scaler), sigma stays fp32 (existing islands; verified), Option A gauge-optimizer composes (test), the out-of-scope scaler-state-on-resume follow-up, and the verified full-suite count. Then:

```bash
git add vfe3/train.py tests/test_fp16_gradscaler.py docs/edits/2026-06-08-decode-bias.md
git commit -m "test(train): fp16 scaler scales + composes with the gauge optimizer (Option A)"
```

---

## Self-Review

**Spec coverage:** scaler wiring (Task 1 Steps 3-4) ✓; default/bf16/fp32 byte-identity (Task 1 `test_gradscaler_disabled_is_byte_identical_to_unscaled_step`) ✓; sigma-fp32 under fp16 (Task 1 `test_fp16_forward_keeps_logits_and_sigma_finite`) ✓; fp16 actually scales (Task 2 `test_fp16_scaler_scales_and_updates_params`) ✓; Option A gauge-optimizer (Task 2 `test_fp16_with_gauge_natural_grad_steps_the_frame`) ✓; grad-accum scaled per-microbatch (Task 1 Step 3 code) ✓; out-of-scope scaler-state-on-resume (changelog note) ✓.

**Placeholder scan:** the CPU-vs-CUDA gating in Task 2 Step 2 is a real decision point handled with an explicit empirical rule (try CPU, gate to CUDA on infra failure, never weaken the assertion), not a vague TODO. No other placeholders.

**Type/consistency:** `train_step(..., scaler=...)` signature is consistent between Task 1 (definition + `train()` call) and Task 2 (test calls). `torch.amp.GradScaler(device=..., enabled=...)` used consistently; the older-API fallback is named.

---

## Executor must confirm against the live repo
1. `Optional` is imported in `train.py` (add `from typing import Optional` if missing).
2. The exact `torch.amp.GradScaler` constructor on the installed torch (device kwarg vs `torch.cuda.amp.GradScaler`); `enabled=False` must be a true no-op either way.
3. Whether CPU `GradScaler(enabled=True)` really scales (decides CPU-real vs CUDA-gated for the Task 2 tests).
