# 2026-06-08 edits

## decode_bias — learned per-vocab log-unigram bias on the linear decode (VFE_2.0 parity)

**Why.** WikiText-103 PPL gap V3 (~154) vs VFE_2.0 (~140), both on the matched linear-decode
path (`use_prior_bank=False`). V2's winning run set `decode_bias=true` — a zero-init learned
per-vocab bias `b` (a log-unigram prior) added to `logits = mu_q @ W^T + b`, in a weight-decay-free
optimizer group. V3 had no bias at all and no `decode_bias` config field. At `embed_dim=20`,
`vocab=50257` a per-vocab bias is the cheapest large PPL lever (it captures token base rates the
rank-20 means otherwise have to encode), so this is the leading suspect for the gap.

**Change (opt-in, default OFF; pure path unchanged).**
- `vfe3/config.py`: new field `decode_bias: bool = False`; `__post_init__` warns it is inert under
  `use_prior_bank=True` (the KL decode's per-vocab priors already play the unigram role).
- `vfe3/model/prior_bank.py`: on the `use_prior_bank=False` path, create `output_proj_bias` (V,)
  zero-init when `decode_bias`, else `None`; drawn after `output_proj_weight` so the weight RNG is
  unchanged. `_decode_linear` adds the bias when present (`logits = mu_q @ W^T + b`).
- `vfe3/model/model.py`: thread `decode_bias=cfg.decode_bias` into the PriorBank construction.
- `vfe3/train.py`: `build_optimizer` puts `output_proj_bias` in its own `weight_decay=0.0` group
  at `m_mu_lr` (decaying a unigram prior toward zero biases it to flat).
- `tests/test_decode_bias.py` (new, 6 tests): param created only on linear path when enabled;
  zero-init bit-identical to no-bias; exact per-vocab additive shift; gradient descends toward
  log-unigram; WD-free optimizer group; inert-under-prior-bank warning.

**Verification.** Full suite `695 passed, 1 xpassed` (was 690; +6 new, +the bias tests; no
regressions). Effect on the ~15 PPL gap is to be measured by a within-V3 ablation
(`decode_bias` on vs off, otherwise identical config).

## phi_weight_decay — separate AdamW decay for the gauge-frame tables

**Why.** PPL-gap finding #2 (`docs/comparisons/2026-06-08-ppl-gap-aspects.md`): on the default
(`m_phi_natural_grad=False`) path V3 weight-decays `phi_embed`/`pos_phi_free` at the generic
`weight_decay` (the wd=0 protection was gated on natural-grad only), while VFE_2.0 always pins its
gauge frames at wd=0. Decoupled AdamW decay sets an LR-invariant frame-norm ceiling
(`|phi*| ~ E[normalized-grad]/wd`) that pulls the transport `exp(phi.G)` toward identity — not
recoverable by the M-step LRs. Adversarially confirmed (→ medium).

**Change (default-preserving for the active run).**
- `vfe3/config.py`: new field `phi_weight_decay: float = 0.065` (added to the `>= 0`/NaN
  validation loop). Default 0.065 = the active run's `weight_decay`, so that run is unchanged; set
  it to 0 for VFE_2.0's gauge-frame protection, or sweep it.
- `vfe3/train.py build_optimizer`: the `phi_embed` group and the learned `pos_phi_free` group take
  `weight_decay=cfg.phi_weight_decay`; the `m_phi_natural_grad=True` path still forces 0 on the
  gauge groups (natural-grad steps phi outside AdamW).
- `train_vfe3.py:175`: `phi_weight_decay = 0.065` (set by user; visible sweep knob).
- `tests/test_phi_weight_decay.py` (new, 5 tests): default 0.065; phi group uses it and is distinct
  from the generic `weight_decay`; override protects phi only (mu untouched); `pos_phi_free` covered;
  natural-grad forces 0 regardless.

**Verification.** Full suite `700 passed, 1 xpassed` (+5 new; no regressions; the default-config
phi-decay change 0.05→0.065 broke nothing). NOTE: a default-constructed `VFE3Config` now decays phi
at 0.065 vs the generic `weight_decay` default 0.05 — intentional per the separate-knob design.

## E-step mu trust-region (default OFF) — PPL-gap finding #3

**Why.** PPL-gap finding #3: V2 box-clamps every E-step mean update (winning run `e_mu_q_trust=5.0`,
`mu_trust_mode='box'`, `_numerics.apply_mu_trust_region`); V3's update was unbounded. At T=1 the
coupling pull is the whole mean step and the linear decode reads only `mu_final`, so a present-vs-
absent clamp changes the logits. Whether it *binds* in normal training is the open question — the
ablation will tell; this adds the knob, default-off so current behavior is bit-identical.

**Change (opt-in, default OFF = current unbounded update).**
- `vfe3/numerics.py`: `apply_mu_trust_region(delta_mu, sigma_q, *, trust, mode, is_diagonal, eps)` —
  whitened per-coordinate `box` clamp (V2's mode) or direction-preserving Mahalanobis `ball`. Exact
  VFE_2.0 `_numerics.apply_mu_trust_region` parity.
- `vfe3/config.py`: `e_mu_q_trust: Optional[float] = None` (None = off) + `mu_trust_mode='box'`,
  with validation (`>0 or None`; mode in box/ball).
- `vfe3/inference/e_step.py`: the mean update clamps `delta_mu` when `e_mu_q_trust` is set
  (`is_diagonal` mirrors the SPD-retraction rank rule); `None` reproduces `mu = mu - e_mu_lr*nat_mu`
  bit-for-bit. Threaded through `e_step`'s knob bag (accept-and-ignore in `free_energy_value`).
- `vfe3/model/block.py`, `vfe3/viz/extract.py`: pass `e_mu_q_trust`/`mu_trust_mode` from cfg.
- `tests/test_mu_trust_region.py` (new, 8 tests): box/ball/full-cov/no-op helper math; config
  default off; e_step None == current behavior; a tight box binds and changes mu.

**Verification.** New tests 8/8 pass; mu-trust default-off path is bit-identical. Full suite
`707 passed, 1 failed` where the 1 failure (`test_config_checkpoint_interval_default_and_validated`,
asserts default 0) is from an UNRELATED working-tree change to `checkpoint_interval` (0→25000), not
this work.

## To sweep finding #2/#3 (V2 parity)
`phi_weight_decay = 0.0` and/or `e_mu_q_trust = 5.0, mu_trust_mode = "box"` in `train_vfe3.py`.

## Banner corpus-coverage + epoch reporting

**Why.** When launching training/ablation the user wants the init banner to state what
fraction of the wiki* corpus a run covers and how many epochs that is, derived from steps,
batch size, seq_len, and stride.

**Change.**
- `vfe3/train.py`: new shared helper `coverage_lines(loader, n_steps, dataset, *, full_corpus_tokens=None)`.
  One optimizer step consumes exactly one batch (`grad_accum_steps` only `torch.chunk`-subdivides
  that batch; it draws no extra batches — verified at `train.py:432-437`), so
  `epochs = n_steps / len(loader)` (`len(loader)` is `drop_last`-aware). Default `stride == seq_len`
  tiles the corpus once per epoch, so unique `corpus=%` saturates at one epoch (`min(1, epochs)`) and
  the `Nx passes` multiple carries the rest. Emits `data:` (tokens/windows/seq_len/stride) and
  `coverage:` (epochs/corpus%/steps-per-epoch/tokens_seen) lines; a `stream:` line reports the loaded
  stream as a % of the full uncapped corpus only when `MAX_TOKENS` caps it. Synthetic anchor omits the
  `% of wiki` line.
- `vfe3/train._banner` and `train_vfe3._banner`: take `train_loader` (and `full_corpus_tokens`) and
  splice the coverage lines into the banner; `train_vfe3.main` computes the uncapped count via
  `load_cached_tokens` only when `MAX_TOKENS` is active.
- `ablation.py`: per-cell print now also emits `coverage_lines(...)`.

## Config-wired belief-table init scales — mu_init_std / sigma_init / phi_scale

**Why.** `PriorBank.__init__` already accepted `mu_init_std`/`sigma_init`/`phi_scale`, but
`VFEModel` constructed the bank without passing them, so they were pinned at the PriorBank
defaults and unreachable from config. The user wants the init variance of `mu`, `Sigma`, and
`phi` adjustable (and sweepable).

**Change.**
- `vfe3/config.py`: three new fields `mu_init_std=0.02`, `sigma_init=1.0`, `phi_scale=0.01`
  (model-structure group). Validation: `sigma_init>0` (log is taken); `mu_init_std`/`phi_scale`
  `>= 0` (0 = deterministic zero table).
- `vfe3/model/model.py`: thread the three into the `PriorBank(...)` construction.
- `ablation.py`: the three added to `BASELINE_CONFIG` and as single-field sweeps (`mu_init_std`,
  `sigma_init`, `phi_scale`).
- `tests/test_belief_init_scales.py` (new, 5 tests): defaults match; scales thread through to
  `mu_embed`/`phi_embed` std and the constant `sigma_log_embed`; zero scales give zero tables;
  non-positive `sigma_init` and negative `mu_init_std` rejected.

**Note (decode init).** On the `use_prior_bank=False` linear-decode path the output weight is
`xavier_uniform_` (fan-based, no std knob) — left unchanged; encode still uses the prior-bank
tables above on both paths.

**Verification.** New 5/5 pass; `test_model`/`test_prior_bank`/`test_train` 50 passed +1 xpassed,
no regressions; ablation sweep field-validation OK.

## Audit-doc status sync — `docs/audits/audit-2026-06-07-lifecycle-multiagent.md`

Doc-only (no code change). Marked the verified findings closed since the audit: **V2** (`close_basis`
wired, `model.py:62-64` + `geometry/closure.py`), **V3**/**V4** (freeze warning now covers the
`unroll`+oracle route and `pos_phi='learned'`, `config.py:736-766` + `model.py:204-214`), **V6**
(`free_energy_terms` threads `alpha_reg`/`include_attention_entropy`, `metrics.py:104-150`); and
Tier-2 #4 **checkpoint resume** (`run_artifacts.py:145,174` + `train(resume_from=...)`). Added a
Status column + status block; annotated resolved (not deleted) to keep the forensic record. V5 (low,
doc-only) and V1 (refuted) left open/n/a.
