# Learnable hyper-prior centroid `r` (opt-in, default frozen)

Date: 2026-06-13
Status: approved (brainstorm), pending implementation

## Problem

The hyper-prior channel adds `lambda_h * mean_i KL(s_i || r)` to the loss, pulling the
model-channel beliefs `s_i` toward a global centroid `r = (r_mu, r_sigma_log)`. Today `r`
is a hardcoded **frozen** parameter (`requires_grad=False`, `prior_bank.py:186-187`): a
fixed centroid the `s_i` are regularized toward. Two `TODO(B)` markers
(`prior_bank.py:188`, `model.py:639`) flag the deferred un-freezing of `r`.

The manuscript-pure un-freezing is a token-dependent, top-down hyper-prior
`r_i = Omega_tilde[s^{(s+1)}]` driven by a scale-(s+1) meta-agent that does not exist in
this codebase — **out of scope**. This spec implements the realistic in-scope increment:
an **opt-in toggle** that makes the global `r` a trainable (empirical-Bayes) centroid,
keeping the frozen behavior as the default.

## Why frozen is the safe default (the collapse)

Freely training `r` alongside `s` against *only* `lambda_h * KL(s||r)` is degenerate: the
joint optimum is `s_i = r = const` for all `i`, where `KL(s||r) -> 0` and the regularizer
vanishes carrying no learning signal. The freeze prevents this (documented at
`train.py:61`, `prior_bank.py:184-185`). A learnable `r` is only meaningful when `s` carries
an independent, data-anchored force — i.e. when `prior_source='model_channel'` routes the
cross-entropy gradient into the `s` tables (and, under `s_e_step`, the refined `s` feeds the
prediction). Then `r` learns the population centroid of the data-anchored `s` — standard
empirical Bayes.

## Design

A single new toggle, default-off, mirroring the existing `s`-table treatment.

### `vfe3/config.py`
- New field `learnable_r: bool = False` (default = current frozen behavior).
- `__post_init__` guard warning: fire `warnings.warn(...)` when **all** hold —
  `learnable_r=True`, `r` is consumed by the forward hyper-prior term
  (`lambda_h > 0 and not s_e_step`), and `s` is not data-anchored
  (`prior_source != 'model_channel'`). In that regime the only force on `s`/`r` is
  `KL(s||r)`, so a free `r` collapses it. Note: `s_e_step` already requires
  `model_channel` (anchored, no warning); `gamma`-only without `model_channel` still
  collapses (warned).

### `vfe3/model/prior_bank.py`
- `PriorBank.__init__` gains `learnable_r: bool = False`.
- `r_mu`/`r_sigma_log` created with `requires_grad=learnable_r` (was hardcoded `False`).
  The frozen default is byte-identical to today.
- The `TODO(B)` comment is updated: the top-down `r_i = Omega_tilde[...]` form stays
  deferred (needs the meta-agent); `learnable_r` is the empirical-Bayes stand-in.

### `vfe3/model/model.py`
- Thread `learnable_r=cfg.learnable_r` into the `PriorBank(...)` construction.
- No change to `_hyper_prior_term` / `_refine_s`: they read `r` identically; gradient now
  flows to `r` when it is a trainable leaf — through the forward term when `lambda_h>0`,
  and through the unrolled `_refine_s` self-coupling target under `s_e_step`.

### `vfe3/train.py` `build_optimizer`
- When `pb.r_mu` exists **and** `pb.r_mu.requires_grad` is True, append groups
  `r_mu` @ `m_mu_lr` and `r_sigma_log` @ `m_sigma_lr` (mirroring the `s` tables; shared
  `weight_decay`). The exact-coverage guard then passes — a trainable `r` *must* be grouped
  or it would silently never update. Frozen `r` stays exempt (guard skips non-trainable
  params). Docstring note updated.

## Testing (`tests/test_hyperprior.py`)
- Regression pin: `learnable_r=False` (default) -> `r` frozen, ungrouped, `grad is None`.
- `learnable_r=True` + `prior_source='model_channel'`: `r.requires_grad` True;
  `build_optimizer` groups both `r` tables; after `backward()` `r_mu.grad` is finite & nonzero.
- Forward loss at init is identical between frozen and learnable `r` (only `requires_grad`
  differs).
- Degenerate regime (`learnable_r=True`, `lambda_h>0`, `prior_source='token'`, no gamma, no
  `s_e_step`) emits the collapse warning; `model_channel` anchor emits none.

## Out of scope
- Token-dependent top-down meta-agent `r_i` (no scale-(s+1) agent exists).
- Any anti-collapse mechanism beyond the warning (user-accepted "you own the foot-gun").

## Verification
- Full `pytest` over the hyper-prior / live-s / config / train suites green.
- Adversarial-review workflow over the diff (optimizer coverage, collapse-guard predicate,
  gauge/gradient-flow implications, test coverage) before completion.
