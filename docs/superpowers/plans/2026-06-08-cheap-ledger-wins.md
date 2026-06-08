# Three cheap ledger wins (M6 b0/c0 sequence, T3 per-head ALiBi, T1 per-head kappa) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generalize three scalar config knobs into per-coordinate / per-head forms — `b0`/`c0` accept a length-`K` list (M6), ALiBi gets the Press per-head geometric slope (T3), and `kappa`/`kappa_gamma` accept a length-`n_heads` list (T1) — each opt-in with a default-byte-identical pure path.

**Architecture:** The kernels already accept the generalized types; the work is the config leg plus correct broadcasting. M6 converts a `list[float]` to a `(K,)` tensor at the consumption boundary; T3 returns a `(H,N,N)` per-head bias from the alibi priors; T1 reshapes a `(H,)` tau to `(H,1,1)` at every softmax site via one helper. Three independent tasks, default-byte-identity test as the guard for each.

**Tech Stack:** Python, PyTorch (float32), pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-06-08-cheap-ledger-wins-design.md`

---

## File structure
- `vfe3/config.py` — type widening + validation for `b0`/`c0` (M6), `alibi_slope` (T3), `kappa`/`kappa_gamma` (T1).
- `vfe3/attention_prior.py` — per-head Press slopes (T3).
- `vfe3/free_energy.py` — `_broadcast_tau` helper + reshape at tau sites (T1).
- `vfe3/gradients/kernels.py` — reshape tau at the kernel softmax (T1).
- `vfe3/model/model.py`, `vfe3/model/block.py` — thread the converted b0/c0 (M6) and per-head kappa/alibi (T1/T3).
- `tests/test_cheap_ledger_wins.py` — new test module (all three tasks add here).

---

## Task 1: M6 — `b0`/`c0` accept a `list[float]`

**Files:**
- Modify: `vfe3/config.py` (`b0`/`c0` type + `__post_init__` validation)
- Modify: `vfe3/model/model.py`, `vfe3/model/block.py` (convert `list -> (K,) tensor` at the b0/c0 consumption sites)
- Test: `tests/test_cheap_ledger_wins.py`

- [ ] **Step 1: Write the failing tests** — create `tests/test_cheap_ledger_wins.py`:

```python
r"""Cheap ledger wins: M6 (b0/c0 sequence), T3 (per-head ALiBi), T1 (per-head kappa).
Spec: docs/superpowers/specs/2026-06-08-cheap-ledger-wins-design.md. Each default byte-identical.
"""

import dataclasses
import json

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel


def _tiny_cfg(**overrides) -> VFE3Config:
    base = dict(vocab_size=8, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=1)
    base.update(overrides)
    return VFE3Config(**base)


def test_b0_c0_default_scalar():
    cfg = VFE3Config()
    assert cfg.b0 == 1.0 and cfg.c0 == 1.0


def test_b0_c0_list_length_must_match_embed_dim():
    with pytest.raises(ValueError, match="b0"):
        _tiny_cfg(b0=[1.0, 1.0, 1.0])           # embed_dim=4, list len 3 -> reject
    # correct length is accepted:
    cfg = _tiny_cfg(b0=[1.0, 2.0, 3.0, 4.0])
    assert list(cfg.b0) == [1.0, 2.0, 3.0, 4.0]


def test_b0_c0_list_entries_must_be_positive():
    with pytest.raises(ValueError, match="c0"):
        _tiny_cfg(c0=[1.0, 0.0, 1.0, 1.0])


def test_b0_list_config_is_json_serializable():
    cfg = _tiny_cfg(b0=[1.0, 2.0, 3.0, 4.0])
    json.dumps(dataclasses.asdict(cfg))         # must not raise (list -> json, no tensor)


def test_b0_list_threads_per_coord_alpha_into_the_model():
    # Under state_dependent_per_coord, a (K,) b0 must reach the model and produce a finite forward
    # whose logits differ from the scalar-b0 baseline (the per-coord constants actually bite).
    torch.manual_seed(0)
    base = VFEModel(_tiny_cfg(alpha_mode="state_dependent_per_coord", b0=1.0, n_e_steps=2))
    torch.manual_seed(0)
    perc = VFEModel(_tiny_cfg(alpha_mode="state_dependent_per_coord",
                              b0=[0.2, 0.5, 2.0, 5.0], n_e_steps=2))
    tok = torch.randint(0, base.cfg.vocab_size, (2, 4))
    lg_base, lg_perc = base(tok), perc(tok)
    assert torch.isfinite(lg_perc).all()
    assert not torch.allclose(lg_base, lg_perc)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_cheap_ledger_wins.py -k "b0 or c0" -v`
Expected: FAIL (`VFE3Config(b0=[...])` either accepts a list with no validation or the model can't consume it).

- [ ] **Step 3: Widen the type + validate in `vfe3/config.py`**

Change the `b0`/`c0` field annotations from `float` to `float | List[float]` (`List` is already imported; if not, add `from typing import List`). In `__post_init__`, add (place near the other numeric checks):

```python
        for _name in ("b0", "c0"):
            _v = getattr(self, _name)
            if isinstance(_v, (list, tuple)):
                if len(_v) != self.embed_dim:
                    raise ValueError(
                        f"{_name} list must have length embed_dim={self.embed_dim}, got {len(_v)}")
                if any(x <= 0.0 for x in _v):
                    raise ValueError(f"{_name} entries must be > 0, got {_v}")
```

(Do NOT add a new scalar `>0` check unless one already exists for b0/c0 — keep the scalar path's behavior unchanged. Only the list case is newly validated.)

- [ ] **Step 4: Convert `list -> (K,) tensor` at the consumption sites**

`b0`/`c0` are read from `cfg` and passed to the alpha kernel. Grep for the read sites: `grep -n "cfg.b0\|cfg.c0\|b0=cfg.b0\|self.cfg.b0" vfe3/model/model.py vfe3/model/block.py`. The known sites are the E-step call (`block.py:52`, `b0=cfg.b0, c0=cfg.c0`) and the M-step self-coupling in `model.py` (`b0=cfg.b0, c0=cfg.c0`). At EACH site, convert a list to a device-correct `(K,)` tensor with this local helper — add it once near the top of `vfe3/model/block.py` and import/reuse it (or define a tiny module-level helper in each file):

```python
def _as_coeff(v, device):
    r"""Pass a scalar b0/c0 through unchanged; turn a list into a (K,) float32 tensor on device."""
    return torch.as_tensor(v, dtype=torch.float32, device=device) if isinstance(v, (list, tuple)) else v
```

In `block.py`, change `b0=cfg.b0, c0=cfg.c0` (in the `e_step(...)` call) to
`b0=_as_coeff(cfg.b0, belief.mu.device), c0=_as_coeff(cfg.c0, belief.mu.device)`.
In `model.py`'s M-step self-coupling, change `b0=cfg.b0, c0=cfg.c0` to
`b0=_as_coeff(cfg.b0, out.mu.device), c0=_as_coeff(cfg.c0, out.mu.device)` (use whatever belief tensor
is in scope there for the device). A scalar float passes through unchanged, so the default path is
byte-identical; a list becomes a `(K,)` tensor the per-coord alpha (`alpha_i.py:121`, already
`(K,)`-capable) consumes.

- [ ] **Step 5: Run to verify pass**

Run: `pytest tests/test_cheap_ledger_wins.py -k "b0 or c0" -v` — expect 5 passed.

- [ ] **Step 6: Regression**

Run: `pytest tests/test_config.py tests/test_alpha_i.py tests/test_model.py --junitxml=out_m6.xml` (use the alpha test file's real name if different; grep `tests/` for `alpha`); read tests/failures/errors; confirm 0 failures/errors; delete `out_m6.xml`.

- [ ] **Step 7: Commit**

```bash
git add vfe3/config.py vfe3/model/model.py vfe3/model/block.py tests/test_cheap_ledger_wins.py
git commit -m "feat(config): b0/c0 accept a length-K list (per-coordinate alpha constants; M6)"
```

---

## Task 2: T3 — per-head ALiBi (Press geometric slope)

**Files:**
- Modify: `vfe3/attention_prior.py` (`prior_alibi`, `prior_causal_alibi`: per-head `(H,N,N)`)
- Modify: `vfe3/config.py` (`alibi_slope: float = 1.0`)
- Modify: `vfe3/model/model.py` (`_attention_log_prior` passes `n_heads`/`alibi_slope`)
- Test: `tests/test_cheap_ledger_wins.py`

- [ ] **Step 1: Write the failing tests** — append:

```python
def test_prior_alibi_per_head_press_slopes():
    from vfe3.attention_prior import get_prior
    H, N = 4, 5
    B = get_prior("alibi")(N, N, n_heads=H, alibi_slope=1.0)
    assert B.shape == (H, N, N)
    # Press geometric: slope_h = 2^(-8(h+1)/H); the per-head bias magnitude at |i-j|=1 is the slope.
    # Head 0 has the LARGEST slope (steepest), decaying with h.
    s0 = -B[0, 0, 1].item()         # = slope_0 * 1
    s_last = -B[H - 1, 0, 1].item()
    assert s0 > s_last > 0
    assert B[0, 2, 2].item() == 0.0   # zero on the diagonal (|i-j|=0)
    assert torch.allclose(B[1], B[1].transpose(-1, -2))   # symmetric in (i,j) per head


def test_prior_causal_alibi_per_head_keeps_mask():
    from vfe3.attention_prior import get_prior
    H, N = 2, 4
    B = get_prior("causal_alibi")(N, N, n_heads=H, alibi_slope=1.0)
    assert B.shape == (H, N, N)
    assert torch.isinf(B[0, 0, 1]) and B[0, 0, 1] < 0     # j>i masked to -inf per head
    assert B[0, 1, 0].item() != float("-inf")             # j<=i allowed


def test_alibi_slope_config_field_default():
    assert VFE3Config().alibi_slope == 1.0


def test_default_causal_forward_byte_identical_to_pre_change():
    # The default attention_prior is 'causal', so T3 must not touch the default forward.
    torch.manual_seed(0); m = VFEModel(_tiny_cfg())   # attention_prior default = causal
    tok = torch.randint(0, m.cfg.vocab_size, (2, 4))
    assert torch.isfinite(m(tok)).all()
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_cheap_ledger_wins.py -k "alibi" -v`
Expected: FAIL (the priors return `(N,N)` and ignore `n_heads`; no `alibi_slope` config).

- [ ] **Step 3: Per-head slopes in `vfe3/attention_prior.py`**

Replace `prior_alibi` and `prior_causal_alibi` bodies so a per-head slope vector is built when `n_heads` is given. Add a small module-level helper and use it in both:

```python
def _press_slopes(n_heads: int, base: float, device, dtype) -> torch.Tensor:
    r"""Press et al. geometric per-head ALiBi slopes: slope_h = base * 2^(-8(h+1)/n_heads)."""
    h = torch.arange(1, n_heads + 1, device=device, dtype=dtype)
    return base * torch.pow(2.0, -8.0 * h / n_heads)            # (H,)
```

`prior_alibi` — add `n_heads: int = 1` and `alibi_slope: float = 1.0` keyword params, and:
```python
    i = torch.arange(n_query, device=device).unsqueeze(-1)
    j = torch.arange(n_key, device=device).unsqueeze(0)
    dist = (i - j).abs().to(dtype)                              # (N, N)
    slopes = _press_slopes(n_heads, alibi_slope, device, dtype)  # (H,)
    return (-slopes.view(n_heads, 1, 1) * dist).to(dtype)        # (H, N, N)
```

`prior_causal_alibi` — same per-head bias, then apply the causal `-inf` mask per head:
```python
    i = torch.arange(n_query, device=device).unsqueeze(-1)
    j = torch.arange(n_key, device=device).unsqueeze(0)
    dist = (i - j).abs().to(dtype)
    slopes = _press_slopes(n_heads, alibi_slope, device, dtype)
    B = (-slopes.view(n_heads, 1, 1) * dist).to(dtype)          # (H, N, N)
    allowed = (j <= i)                                         # (N, N)
    return B.masked_fill(~allowed.unsqueeze(0), float("-inf"))  # (H, N, N)
```

(Both still accept `**kwargs` so unrelated callers passing extra keys do not break. With the default
`n_heads=1` they return `(1, N, N)` — a single Press slope; a caller wanting the legacy `(N,N)` can
squeeze, but the model call site below passes the real `n_heads`.)

- [ ] **Step 4: Add `alibi_slope` config + pass `n_heads` from the call site**

In `vfe3/config.py`, add `alibi_slope: float = 1.0` (near the attention-prior fields; validate `> 0` if the file validates such floats, else leave unvalidated to match neighbors).

In `vfe3/model/model.py`, find `_attention_log_prior` (grep `def _attention_log_prior`). It calls the registered prior with `n_query`/`n_key`/`device`. Add `n_heads=self.cfg.n_heads, alibi_slope=self.cfg.alibi_slope` to that call (the non-alibi priors ignore the extra kwargs via `**kwargs`). Confirm the returned `(H,N,N)` (for alibi) broadcasts against the `(B,H,N,N)` energy the attention consumes — the causal/uniform priors still return `(N,N)` and broadcast as before, so the default path is unchanged.

- [ ] **Step 5: Run to verify pass**

Run: `pytest tests/test_cheap_ledger_wins.py -k "alibi or causal_forward" -v` — expect 4 passed.

- [ ] **Step 6: Regression**

Run: `pytest tests/test_attention_prior.py tests/test_model.py tests/test_free_energy.py --junitxml=out_t3.xml` (use the real attention-prior test filename; grep if unsure); read tests/failures/errors; confirm 0; delete `out_t3.xml`.

- [ ] **Step 7: Commit**

```bash
git add vfe3/attention_prior.py vfe3/config.py vfe3/model/model.py tests/test_cheap_ledger_wins.py
git commit -m "feat(attention): per-head Press ALiBi slopes + config-reachable alibi_slope (T3)"
```

---

## Task 3: T1 — per-head `kappa` / `kappa_gamma`

**Files:**
- Modify: `vfe3/config.py` (`kappa`/`kappa_gamma` type + validation)
- Modify: `vfe3/free_energy.py` (`_broadcast_tau` helper + reshape at tau sites)
- Modify: `vfe3/gradients/kernels.py` (reshape tau at the kernel softmax)
- Modify: `vfe3/model/model.py` / `vfe3/model/block.py` (build the `(H,)` kappa tensor for `attention_tau`)
- Test: `tests/test_cheap_ledger_wins.py`

- [ ] **Step 1: Write the failing tests** — append:

```python
def test_attention_tau_returns_per_head_vector():
    from vfe3.free_energy import attention_tau
    tau = attention_tau(torch.tensor([1.0, 2.0]), irrep_dims=[3, 3])
    assert tau.shape == (2,)
    assert torch.allclose(tau, torch.tensor([1.0, 2.0]) * (3 ** 0.5))


def test_kappa_default_scalar_byte_identical():
    torch.manual_seed(0); a = VFEModel(_tiny_cfg(kappa=1.0))
    torch.manual_seed(0); b = VFEModel(_tiny_cfg(kappa=1.0))
    tok = torch.randint(0, a.cfg.vocab_size, (2, 4))
    assert torch.equal(a(tok), b(tok))


def test_kappa_equal_list_equals_scalar():
    # A per-head list with all-equal entries must match the scalar path bitwise.
    torch.manual_seed(0); sca = VFEModel(_tiny_cfg(kappa=1.5, n_e_steps=2))
    torch.manual_seed(0); lst = VFEModel(_tiny_cfg(kappa=[1.5, 1.5], n_e_steps=2))
    tok = torch.randint(0, sca.cfg.vocab_size, (2, 4))
    assert torch.allclose(sca(tok), lst(tok), atol=1e-6, rtol=1e-5)


def test_kappa_per_head_changes_logits():
    torch.manual_seed(0); sca = VFEModel(_tiny_cfg(kappa=1.5, n_e_steps=2))
    torch.manual_seed(0); per = VFEModel(_tiny_cfg(kappa=[0.5, 4.0], n_e_steps=2))
    tok = torch.randint(0, sca.cfg.vocab_size, (2, 4))
    assert not torch.allclose(sca(tok), per(tok))


def test_single_block_group_rejects_list_kappa():
    # Per-head kappa needs equal irrep blocks; a single-block group must reject a list.
    with pytest.raises(ValueError, match="kappa"):
        _tiny_cfg(gauge_group="glk", kappa=[1.0, 2.0])   # use the codebase's single-block group name
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_cheap_ledger_wins.py -k "kappa or attention_tau" -v`
Expected: FAIL (config rejects list kappa or the `(H,)` tau is not broadcast and shapes mismatch).

- [ ] **Step 3: Config — widen `kappa`/`kappa_gamma` + validate**

In `vfe3/config.py`, change `kappa`/`kappa_gamma` from `float` to `float | List[float]`. In `__post_init__`, add a check that a list kappa is only valid for an equal-block group and matches `n_heads`. Determine the single-block group name(s) by reading the gauge-group registry; a robust form:

```python
        for _name in ("kappa", "kappa_gamma"):
            _v = getattr(self, _name)
            if isinstance(_v, (list, tuple)):
                if self.gauge_group not in ("block_glk", "tied_block_glk"):
                    raise ValueError(
                        f"{_name} list (per-head) requires an equal-block group "
                        f"(block_glk/tied_block_glk); got gauge_group={self.gauge_group!r}")
                if len(_v) != self.n_heads:
                    raise ValueError(
                        f"{_name} list must have length n_heads={self.n_heads}, got {len(_v)}")
```
(Verify the equal-block group names against the gauge-group registry in `config.py`/`groups.py`; adapt the tuple if the names differ. The single-block test in Step 1 uses one such non-equal-block name — align it with a real registered single-block group.)

- [ ] **Step 4: `_broadcast_tau` helper + reshape at the tau sites in `vfe3/free_energy.py`**

Add near `attention_tau`:
```python
def _broadcast_tau(tau, energy_ndim: int):
    r"""A scalar/0-d tau passes through; a per-head (H,) tau is reshaped to (H,) + (1,)*(ndim-3)
    so it aligns with the head axis of an (..., H, N, N) energy (block_glk). For a (B,N,N) /
    (N,N) energy with a scalar tau this is a no-op."""
    import torch as _t
    if isinstance(tau, _t.Tensor) and tau.dim() == 1:
        return tau.view(tau.shape[0], *([1] * (energy_ndim - 3)))
    return tau
```
At each `-energy / tau` and `tau * (...)` site (`attention_weights:211`, `log_partition:233`,
`reduced_free_energy:252` via its `log_partition` call, and the entropy term `free_energy:314`),
replace `tau` with `_broadcast_tau(tau, energy.dim())` (use the local energy/`beta` tensor's `.dim()`).
Grep `vfe3/free_energy.py` for every `tau` arithmetic use and cover them all — the
`test_kappa_default_scalar_byte_identical` test is the guard that the scalar path is unchanged.

- [ ] **Step 5: Reshape tau in the belief-gradient kernel**

Grep `vfe3/gradients/kernels.py` for `tau` (the closed-form kernel forms `beta = softmax(-E/tau)` and the gradient). Import or re-define the same `_broadcast_tau` (import it: `from vfe3.free_energy import _broadcast_tau`) and wrap every `tau` arithmetic use with `_broadcast_tau(tau, <energy>.dim())`. A scalar tau is unchanged.

- [ ] **Step 6: Build the `(H,)` kappa tensor for `attention_tau`**

`attention_tau(cfg.kappa, irrep_dims)` already returns `(H,)` if `kappa` is a `(H,)` tensor. At the call sites that pass `cfg.kappa`/`cfg.kappa_gamma` (grep `attention_tau(` across `vfe3/`: `block.py:50`, `model.py` gamma block / `_refine_s`), convert a list to a tensor:
`attention_tau(_as_coeff(cfg.kappa, <device>), group.irrep_dims)` — reuse the `_as_coeff` helper from Task 1 (move it to a shared spot, e.g. top of `block.py`, and import where needed, OR re-define the one-liner). A scalar passes through unchanged.

- [ ] **Step 7: Run to verify pass**

Run: `pytest tests/test_cheap_ledger_wins.py -k "kappa or attention_tau" -v` — expect 5 passed.

- [ ] **Step 8: Full suite**

Run: `pytest --junitxml=out_full.xml` (NO extra `-q`). Read tests/failures/errors/skipped from `out_full.xml`; confirm 0 failures and 0 errors. Record the number. Delete `out_full.xml`. If anything fails, STOP and report BLOCKED with the failing names (a non-byte-identical default-config test = a missed tau site).

- [ ] **Step 9: Changelog + commit**

Append a brief `## Cheap ledger wins (M6 b0/c0 list, T3 per-head ALiBi, T1 per-head kappa)` section to `docs/edits/2026-06-08-decode-bias.md` (what each does, default byte-identical, files, verified suite count). Then:
```bash
git add vfe3/config.py vfe3/free_energy.py vfe3/gradients/kernels.py vfe3/model/model.py vfe3/model/block.py tests/test_cheap_ledger_wins.py docs/edits/2026-06-08-decode-bias.md
git commit -m "feat(free_energy): per-head kappa via (H,1,1) tau broadcast (T1)"
```

---

## Self-Review

**Spec coverage:** M6 type+validation+conversion (Task 1) ✓; M6 json-serializable (Task 1 test) ✓; T3 per-head Press slopes + `alibi_slope` config + call-site `n_heads` (Task 2) ✓; T3 default-causal byte-identical (Task 2 test) ✓; T1 type+validation, `_broadcast_tau`, free_energy + kernel sites, kappa tensor build (Task 3) ✓; T1 default byte-identical + equal-list==scalar (Task 3 tests) ✓.

**Placeholder scan:** the "grep for the consumption sites / tau uses and cover them all" instructions are real discovery steps (the exact line set must be confirmed against the live tree), each backed by a byte-identity guard test — not vague TODOs. The single-block group name in the T1 reject test and the equal-block group tuple in validation are flagged to confirm against the registry.

**Type consistency:** `_as_coeff(v, device)` (Task 1) is reused in Task 3 Step 6; `_broadcast_tau(tau, energy_ndim)` is defined in Task 3 Step 4 and imported into the kernel in Step 5 — consistent names. `alibi_slope`, `n_heads` kwargs match between the prior (Task 2 Step 3) and the call site (Step 4).

## Executor must confirm against the live repo
1. The single-block gauge-group name(s) for the T1 reject test + the equal-block tuple in validation (read the gauge-group registry).
2. The exact test filenames for the regression runs (`test_alpha_i.py`, `test_attention_prior.py` — grep `tests/`).
3. Every `tau` arithmetic site in `free_energy.py` and `gradients/kernels.py` (the byte-identity test guards misses).
4. Whether `cfg.b0`/`cfg.c0` already have a scalar `>0` validation (don't duplicate / don't change scalar behavior).
