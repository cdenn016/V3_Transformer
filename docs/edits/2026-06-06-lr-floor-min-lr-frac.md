# 2026-06-06 — Fractional LR floor (`min_lr_frac`) + zero-base-LR ZeroDivisionError fix

Follow-on to the earlier `min_lr` LR-floor work. The shared absolute `min_lr` floors every
M-step group at the SAME value regardless of base, breaking the deliberately-tuned
m_mu:m_sigma:m_phi ratios at the cosine tail (the floor is ~4% of `m_sigma_lr`'s base but
only ~1.5% of `m_mu_lr`'s). Added a fractional floor that scales with each group's own base.

## Pre-existing bug found and fixed

The absolute-floor lambda `max(cfg.min_lr / base, cosine)` divides by `base`. Any M-step
group with base LR 0 — the common `m_phi_lr=0.0` frozen-gauge config, plus the
`pos_phi='learned'` group also at `m_phi_lr` — gives `base=0`, so `min_lr / 0` raised
`ZeroDivisionError` at scheduler construction inside `train()`. Confirmed empirically
(`base_lrs: [0.025, 0.0025, 0.0, 0.0]` → `CRASH: ZeroDivisionError`), and the
`tests/test_train.py` suite was already RED on it (6 failures, all train()-with-`m_phi_lr=0`
paths). Introduced by today's earlier `min_lr` commit `ad0e811`.

## Changes

- **`config.py`** — added `min_lr_frac: float = 0.0` (validated `>= 0`, and `min_lr` added
  to the same non-negativity guard). Default 0.0 keeps current behavior; `0.0` with
  `min_lr=0` is the pure half-cosine-to-zero.

- **`train.py`** — extracted the per-group floor into a module-level helper
  `_floor_lr_lambdas(base_lrs, cfg)`. Each group's multiplier floor is
  `max(min_lr/base, min_lr_frac)`, so the absolute LR floors at
  `max(min_lr, min_lr_frac * base)` — `min_lr` shared across groups, `min_lr_frac`
  proportional to each base (ratios preserved into the tail). A `base == 0` group DROPS the
  `min_lr/base` term (no division) and stays frozen: an absolute `min_lr` does not resurrect
  a channel the user chose to freeze. `train()` now calls the helper instead of the inline
  lambda (stale comment replaced).

- **`tests/test_train.py`** — the two existing scheduler tests now call `_floor_lr_lambdas`
  (they previously duplicated the production lambda inline, so they tested a divergent copy
  and could never catch the base=0 crash). Added three tests:
  `test_fractional_floor_scales_each_group_to_min_lr_frac_times_base` (each group floors at
  `min_lr_frac*base`, ratios preserved), `test_floor_is_max_of_absolute_min_lr_and_fractional`
  (combined `max(min_lr, frac*base)`), and `test_floor_lambdas_handle_zero_base_lr_without_dividing`
  (no ZeroDivisionError; frozen group stays at 0).

## Verification

- TDD: the new tests were confirmed RED first (`ImportError: cannot import name
  '_floor_lr_lambdas'`), then the helper/config/wiring were implemented. The user declined
  the GREEN test run, so the GREEN pass is NOT machine-confirmed. Re-run
  `pytest tests/test_train.py tests/test_config.py` to confirm the new tests pass and the 6
  prior ZeroDivisionError failures are cleared.
