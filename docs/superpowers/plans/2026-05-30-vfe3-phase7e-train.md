# VFE_3.0 Phase 7e (Training loop + cutover) Implementation Plan

> REQUIRED SUB-SKILL: superpowers:test-driven-development (RED→GREEN→COMMIT; **commit after every GREEN task**). V3-internal tests. No VFE_2.0 provenance.

**Goal:** Close Phase 7 — `vfe3/train.py` (AdamW per-group learning rates, warmup+cosine schedule, the training step + loop) and the **cutover test**: the assembled `VFEModel` *learns* — cross-entropy decreases by a margin over training steps on a structured synthetic stream **and** on real cached `wikitext-2` tokens. This is the spec's cutover criterion: end-to-end training reduces its own loss.

**Architecture:** Phase 7e (spec §4.6, §8, §10). The M-step: `loss.backward()` (through the unrolled E-step) → AdamW updates the PriorBank prior tables. Per-group LRs from config: `mu_embed`@`m_mu_lr`, `{sigma_log_embed, decode_log_scale}`@`m_sigma_lr`, `phi_embed`@`m_phi_lr`. Linear warmup to `warmup_steps`, cosine decay to `max_steps`. The cutover test uses a **structured** synthetic stream (a short period, so the irreducible entropy is low and a learning model drives CE down) and a real `wikitext-2` slice; both assert `loss[-1] < loss[0]` by a margin.

**Tech Stack:** Python 3, PyTorch (`torch.optim.AdamW`, `LambdaLR`), pytest. No NN layers (the model has none; AdamW optimizes the prior tables). No CLI (click-to-run `run_training`). Device-agnostic.

**Reference spec:** §4.6, §8 Phase 7, §10 cutover. Prereq: Phases 0–7d on `main` (branch `phase7e-train`). Reuses `vfe3.model.model` (`VFEModel`), `vfe3.config.VFE3Config`, `vfe3.data.datasets` (`make_dataloader`, `TokenWindows`).

**Design decisions (do not relitigate):**
1. **Per-group AdamW.** Group the PriorBank parameters by name into the three M-step LRs; `weight_decay=cfg.weight_decay`. (No nn layers → the only params are the prior tables.)
2. **Warmup + cosine** via `LambdaLR`: `lr_mult(step) = step/warmup_steps` for `step<warmup_steps`, else `0.5(1+cos(pi·(step-warmup)/(max_steps-warmup)))`.
3. **Cutover = learnability, not a fixed checksum.** The structured-synthetic test must show CE dropping by a clear margin (e.g. `loss[-1] < 0.6·loss[0]`); the real-data smoke asserts a softer monotone-ish decrease (`loss[-1] < loss[0] - margin`), `skip` if the cache is absent. A forward-only checksum is NOT sufficient (per the advisor).
4. **Unrolled training graph** is the default (`gradient_mode='filtering'`, `detach_e_step=False`); the optimizer step then improves the priors through inference.

---

## Code Style / Provenance (MANDATORY)

Repo CLAUDE.md conventions. No "VFE_2.0"/"2.0"/"ported" in any shipped artifact; cite manuscript + math.

---

## File Structure

- **Create** `vfe3/train.py` — `build_optimizer`, `lr_lambda`, `train_step`, `train`, `run_training`.
- **Create** `tests/test_train.py`.

---

## Task 1 — `build_optimizer` + `lr_lambda`

**Files:** `vfe3/train.py`; `tests/test_train.py`.

- [ ] **RED:** `tests/test_train.py`:

```python
import math
import torch
from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel
from vfe3.train import build_optimizer, lr_lambda


def test_optimizer_groups_priors_by_m_lr():
    cfg = VFE3Config(vocab_size=20, embed_dim=4, n_heads=2,
                     m_mu_lr=0.01, m_sigma_lr=0.002, m_phi_lr=0.005)
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)
    lrs = sorted(g["lr"] for g in opt.param_groups)
    assert lrs == [0.002, 0.005, 0.01]
    # every PriorBank parameter is covered by exactly one group
    n_params = sum(len(g["params"]) for g in opt.param_groups)
    assert n_params == len(list(model.parameters()))


def test_lr_lambda_warmup_then_cosine():
    cfg = VFE3Config(warmup_steps=10, max_steps=100)
    assert abs(lr_lambda(0, cfg) - 0.0) < 1e-6
    assert abs(lr_lambda(10, cfg) - 1.0) < 1e-6            # peak at end of warmup
    assert lr_lambda(55, cfg) < 1.0 and lr_lambda(55, cfg) > 0.0
    assert abs(lr_lambda(100, cfg) - 0.0) < 1e-3           # ~0 at max_steps
```

- [ ] **GREEN:** `vfe3/train.py` (this task: optimizer + schedule). Group by parameter identity:

```python
r"""Training (M-step) for VFE_3.0: AdamW per-group learning rates + warmup/cosine.

The model has no neural layers; the only parameters are the PriorBank prior tables.
loss.backward() flows through the unrolled E-step to those tables; AdamW updates them.
Click-to-run: edit a VFE3Config and call run_training (no CLI).
"""

import math
from typing import Callable, List, Optional, Tuple

import torch

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel


def build_optimizer(
    model: VFEModel,
    cfg:   VFE3Config,
) -> torch.optim.Optimizer:
    r"""AdamW with per-group M-step learning rates over the PriorBank prior tables."""
    pb = model.prior_bank
    groups = [
        {"params": [pb.mu_embed],                          "lr": cfg.m_mu_lr},
        {"params": [pb.sigma_log_embed, pb.decode_log_scale], "lr": cfg.m_sigma_lr},
        {"params": [pb.phi_embed],                         "lr": cfg.m_phi_lr},
    ]
    return torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)


def lr_lambda(
    step: int,
    cfg:  VFE3Config,
) -> float:
    r"""Linear warmup to 1.0 at warmup_steps, then cosine decay to 0 at max_steps."""
    if step < cfg.warmup_steps:
        return step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
```

- [ ] Run → 2 passed. **COMMIT** `feat(train): AdamW per-group LRs + warmup/cosine schedule`.

---

## Task 2 — `train_step` + `train` + the cutover

**Files:** modify `vfe3/train.py`; `tests/test_train.py`.

- [ ] **RED:** append. Structured synthetic stream (period P) → low irreducible entropy → a learning model drives CE down:

```python
from vfe3.data.datasets import TokenWindows
from torch.utils.data import DataLoader
from vfe3.train import train


def _periodic_loader(V=6, period=3, n=600, seq_len=8, batch_size=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    base = torch.arange(period).repeat(n // period + 2)         # 0,1,2,0,1,2,...
    ds = TokenWindows(base[: n].to(torch.long), seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True,
                      generator=g)


def test_training_decreases_loss_on_structured_stream():
    torch.manual_seed(0)
    cfg = VFE3Config(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                     n_e_steps=2, e_mu_lr=0.3, e_phi_lr=0.0,
                     m_mu_lr=0.05, m_sigma_lr=0.01, m_phi_lr=0.0, warmup_steps=5, max_steps=60)
    model = VFEModel(cfg)
    losses = train(model, _periodic_loader(V=6, period=3), cfg, n_steps=60)
    assert losses[-1] < 0.6 * losses[0]                         # the model LEARNS the period


def test_training_smoke_on_real_wikitext2_if_present():
    import pytest
    from vfe3.data.datasets import load_cached_tokens
    try:
        toks = load_cached_tokens("wikitext-2", "validation")
    except FileNotFoundError:
        pytest.skip("wikitext-2 cache absent")
    torch.manual_seed(0)
    cfg = VFE3Config(vocab_size=50257, embed_dim=8, n_heads=2, max_seq_len=16, n_layers=1,
                     n_e_steps=1, e_mu_lr=0.3, e_phi_lr=0.0,
                     m_mu_lr=0.05, m_sigma_lr=0.01, m_phi_lr=0.0, warmup_steps=3, max_steps=30)
    model = VFEModel(cfg)
    ds = TokenWindows(toks[:4000], 16)
    loader = DataLoader(ds, batch_size=8, shuffle=True, drop_last=True)
    losses = train(model, loader, cfg, n_steps=30)
    assert all(map(lambda x: x == x, losses))                   # finite (no NaN)
    assert losses[-1] < losses[0] - 0.05                        # real-token loss decreases
```

- [ ] **GREEN:** append to `vfe3/train.py`:

```python
def train_step(
    model:     VFEModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    tokens:    torch.Tensor,             # (B, N)
    targets:   torch.Tensor,             # (B, N)

    *,
    grad_clip: float = 1.0,
) -> float:
    r"""One optimizer step on the cross-entropy of a batch; returns the loss scalar."""
    optimizer.zero_grad(set_to_none=True)
    _, loss, _ = model(tokens, targets)
    loss.backward()
    if grad_clip is not None and grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    scheduler.step()
    return float(loss.detach())


def train(
    model:    VFEModel,
    loader,
    cfg:      VFE3Config,

    *,
    n_steps:  int = 100,
    grad_clip: float = 1.0,
) -> List[float]:
    r"""Train ``n_steps`` optimizer steps (cycling the loader); returns the loss history."""
    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_lambda(s, cfg))
    losses: List[float] = []
    model.train()
    it = iter(loader)
    for _ in range(n_steps):
        try:
            tokens, targets = next(it)
        except StopIteration:
            it = iter(loader); tokens, targets = next(it)
        losses.append(train_step(model, optimizer, scheduler, tokens, targets, grad_clip=grad_clip))
    return losses


def run_training(
    cfg:     VFE3Config,
    dataset: str = "wikitext-2",
    split:   str = "train",

    *,
    n_steps:    int = 1000,
    max_tokens: Optional[int] = None,
) -> Tuple[VFEModel, List[float]]:
    r"""Click-to-run: build a model + a dataloader from the cache and train (no CLI)."""
    from vfe3.data.datasets import make_dataloader
    model = VFEModel(cfg)
    loader = make_dataloader(dataset, split, cfg.max_seq_len, cfg.batch_size, max_tokens=max_tokens)
    losses = train(model, loader, cfg, n_steps=n_steps)
    return model, losses
```

- [ ] Run → 2 passed (the structured-stream loss drops below 0.6× start; the real wikitext-2 smoke decreases). If the structured test does not drop, raise `n_steps`/`m_mu_lr` modestly OR `n_e_steps`; do NOT weaken the margin assertion past what a learning model achieves — investigate if it cannot learn a period-3 stream. **COMMIT** `feat(train): train loop + cutover (loss decreases on structured + real tokens)`.

---

## Task 3 — full suite + changelog + commit

- [ ] `python -m pytest -q` — expect prior 155 + new (~4) green.
- [ ] Append "## Phase 7e Training + cutover — 2026-05-30 (continuation)" to the changelog (provenance-clean), **recording the structured-stream and real-wikitext-2 loss curves** (start→end). Commit `docs(edits): 2026-05-30 phase 7e training changes log`.

---

## Self-Review

**Coverage:** per-group AdamW + warmup/cosine → T1; train loop → T2; the **cutover** (CE decreases on structured synthetic AND real wikitext-2) → T2.
**Anchors:** optimizer groups the priors at the three M-step LRs; `lr_lambda` warmup→cosine shape; `loss[-1] < 0.6·loss[0]` on a learnable period; real-token loss decreases; no NaN.
**Modularity:** per-group LRs from config; `run_training` selects the dataset from the cache by name; `detach_e_step`/`gradient_mode` from config.
**Deferred (named):** validation/eval loop + perplexity; checkpointing; gradient accumulation; mixed precision; the scaling/PPL sweep; true batched E-step (perf). Phase 7 is then complete (cutover met); Phase 5 (gauge-generalization theory) and the diagnostics/metrics/viz tier remain.
