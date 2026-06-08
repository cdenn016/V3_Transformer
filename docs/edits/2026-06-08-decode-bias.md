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

## s_e_step + e_s_*_lr fields (live model channel, default off)

Config-only. Added three fields to `vfe3/config.py` (E-step group): `s_e_step: bool = False`,
`e_s_mu_lr: float = 0.1`, `e_s_sigma_lr: float = 0.1`. Validation: negative `e_s_*_lr` raises;
`s_e_step=True` with `prior_source != 'model_channel'` raises; `s_e_step=True` with both
`lambda_h=0` and `gamma_coupling=0` warns (inert path). New test file
`tests/test_live_s_model_channel.py` (4 tests). Full suite: 717 passed, 1 xpassed.

## PriorBank s-tables + frozen r under s_e_step (Task 2 of live model channel)

**Why.** `s_e_step=True` enables the live model-channel E-step that refines the per-token
model beliefs `s_i`. The `PriorBank` tables (`s_mu_embed`, `s_sigma_log_embed`) and the frozen
hyper-prior centroid (`r_mu`, `r_sigma_log`) must exist whenever `s_e_step` is on, independent of
`lambda_h` and `gamma_coupling`.

**Change.**
- `vfe3/model/prior_bank.py`: added `s_e_step: bool = False` keyword param to `PriorBank.__init__`
  (placed after `prior_source`, vertical-alignment convention matched); stored as `self.s_e_step`.
  Extended the s-table gate to `lambda_h > 0.0 or gamma_coupling > 0.0 or prior_source == "model_channel" or s_e_step`.
  Extended the `r` gate to `lambda_h > 0.0 or s_e_step`. Both blocks remain the last parameters
  drawn in `__init__` so the belief tables keep their RNG draw and stay byte-identical.
- `vfe3/model/model.py`: added `s_e_step=cfg.s_e_step` alongside `prior_source=cfg.prior_source`
  in the `PriorBank(...)` construction.
- `tests/test_live_s_model_channel.py`: two new tests (`test_s_tables_and_frozen_r_created_under_s_e_step`,
  `test_belief_tables_byte_identical_with_or_without_s_e_step`).

**Verification.** 36 passed (full targeted regression: `test_live_s_model_channel.py`,
`test_prior_bank.py`, `test_model.py`); no regressions.

## VFEModel._refine_s — model-channel E-step (Task 3 of live model channel)

**Why.** With the s-tables and frozen r in place (Task 2), the model needs a method that runs
the s-channel E-step: refine s toward r plus the gamma model-consensus with the gauge frame
held fixed. This is the precursor to wiring the live channel into `forward` (Task 4).

**Change (vfe3/model/model.py only; method not yet called by forward).**
- New private method `VFEModel._refine_s(token_ids, phi0, *, e_step_gradient)` inserted between
  `_apply_pos_phi` and `forward`. It calls the existing channel-agnostic `e_step` with
  `BeliefState(mu=s_mu, sigma=s_sigma, phi=phi0)`, `r_mu`/`r_sigma` as the prior, `e_phi_lr=0.0`
  (frame held fixed), `transport_mode="flat"` (tied flat cocycle), `value=cfg.lambda_h` (s->r
  self-coupling), and `lambda_beta=cfg.gamma_coupling` (s->s consensus weight).
- No kwarg or config-field adaptations were needed: all names matched exactly.

**Tests (tests/test_live_s_model_channel.py, 2 new tests).**
- `test_refine_s_preserves_shape_and_zero_lr_is_static`: shape (B,N,K) correct; `e_s_lr=0` is
  a no-op (s1 == s0).
- `test_refine_s_moves_s_with_nonzero_lr`: `e_s_lr=0.5` moves s away from its initial value.

**Verification.** `9 passed` (full `test_live_s_model_channel.py`); 2 new tests pass.

## VFEModel.forward wires live s channel (Task 4 of live model channel)

**What.** Under `s_e_step=True`, `forward` now: (1) calls `_refine_s(token_ids, beliefs.phi,
e_step_gradient=...)` inside the `with run, amp:` block immediately before `vfe_stack`, then
replaces the belief's `mu`/`sigma` with the refined `s1` — so both the initial belief state and
the prior passed to `vfe_stack` are the live s; (2) suppresses the loss-level `lambda_h` and
`gamma_coupling` blocks (guarded `and not self.cfg.s_e_step`) because those forces now live
inside `_refine_s` and double-counting them would be wrong. The pure `s_e_step=False` path is
unchanged.

**Tests (tests/test_live_s_model_channel.py, 4 new tests).**
- `test_default_off_forward_is_unchanged_by_the_new_code`: `s_e_step=False` gives finite logits.
- `test_s_e_step_changes_logits_at_n_e_steps_1`: same seed + bit-identical belief tables; the
  live channel alone changes the logits.
- `test_e_s_lr_zero_reduces_to_static_model_channel`: `e_s_lr=0` gives logits matching the
  static `model_channel` path (`atol=1e-6, rtol=1e-5`; NOT bit-exact due to SPD retraction
  exp/log roundtrip, but well within tolerance).
- `test_s_e_step_gradient_reaches_s_tables_at_t1`: `loss.backward()` produces non-zero grad on
  `s_mu_embed` via the unrolled `_refine_s`.

**Verification.** `13 passed` (full `test_live_s_model_channel.py`); regression `test_model.py` +
`test_train.py`: `44 passed, 1 xpassed`, 0 failures, 0 errors.

## VFEModel.diagnostics wires live s channel (Task 5 of live model channel)

**What.** Under `s_e_step=True`, `diagnostics` now refines s and anchors the single-sequence
belief to it before `vfe_stack`, giving train/inference parity for a model trained with the live
channel. The insertion point is after `belief = BeliefState(...)` (pos-phi already applied) and
before `vfe_stack`. Because `_refine_s` is batched (`(B, N)`) and `diagnostics` builds an
unbatched belief (`(N, K)`), the call uses `token_ids[:1]` and `belief.phi.unsqueeze(0)`, then
indexes `[0]` off each returned tensor. `generate` required no edit: it delegates to `forward`,
which already has the s_e_step anchor from Task 4. The pure `s_e_step=False` path is unchanged.

**Tests (tests/test_live_s_model_channel.py, 2 new tests).**
- `test_generate_runs_under_s_e_step`: `generate(prompt, max_new_tokens=2)` runs without raising
  and returns shape `(1, 5)` under `s_e_step=True`.
- `test_diagnostics_runs_under_s_e_step`: `diagnostics(tok)` runs without raising and returns a
  non-None dict under `s_e_step=True`.

**Verification.** `15 passed` (full `test_live_s_model_channel.py`); regression `test_model.py`:
`24 passed, 0 failures, 0 errors` (confirmed via junitxml).

**Note.** `attention_maps` (model.py line ~821) builds the same single-sequence belief pattern
and currently has the same no-live-s gap; it is out of scope for this task.

## Live model channel s (s_e_step) — dynamic prior tie, default OFF

Tasks 1–6 implement the live model-channel E-step across config, `PriorBank`, `VFEModel`, and tests. Per-task notes are recorded in sections above; this section gives the consolidated picture.

**What it does.** Under `s_e_step=True` (requires `prior_source='model_channel'`), `forward` calls `VFEModel._refine_s(token_ids, phi0)` immediately before `vfe_stack`. That method runs the channel-agnostic `e_step` on the `(s_mu, s_sigma)` tables with the frozen global centroid `r` as the self-target (`lambda_h` coupling), `gamma_coupling` as the model-consensus weight, and `e_phi_lr=0` so the gauge frame is held fixed. The refined `s1_mu`, `s1_sigma` then replace the belief's `mu`/`sigma` and also serve as the prior passed to `vfe_stack`, anchoring every E-step iteration to the live s. Because the self-coupling force is present inside `_refine_s`, s reaches the vicinity of `mu_final` even at the operative `n_e_steps=1`. The `r` centroid is frozen (`requires_grad=False`); see `TODO(B)` below.

**Default-OFF invariant.** `s_e_step=False` is byte-identical to the pre-feature code: s-tables are drawn last (belief tables unchanged), the `_refine_s` branch is never entered, and the `lambda_h`/`gamma` supersede guards (`and not self.cfg.s_e_step`) reduce to the originals.

**Supersede logic.** Under `s_e_step` the loss-level `lambda_h` and `gamma_coupling` blocks are skipped (those forces now live inside `_refine_s`; double-counting would be incorrect). The frozen global `r` is the manuscript-consistent stand-in for the top-down meta-agent; `TODO(B)` in `prior_bank.py` and `model.py` marks the deferred upgrade to a token-dependent, per-token hyper-prior once the scale-(s+1) meta-agent exists.

**Limiting case.** Setting `e_s_mu_lr=e_s_sigma_lr=0` makes `_refine_s` a no-op (s1 == s0), recovering the static `prior_source='model_channel'` tie. This is the manuscript's slow-channel limit (`e_s_lr -> 0`) and is verified by `test_e_s_lr_zero_reduces_to_static_model_channel` (atol=1e-6).

**Parity.** `generate` inherits the live-s behaviour via `forward` with no additional edits. `diagnostics` anchors explicitly: it calls `_refine_s(token_ids[:1], phi.unsqueeze(0))` and indexes `[0]` off the returned tensors to match its unbatched `(N, K)` belief. KNOWN FOLLOW-UP: `attention_maps` builds an un-anchored single-sequence belief and does NOT yet use the live s under `s_e_step` (visualisation-only; tracked as a follow-up).

**Files.** `vfe3/config.py` (three new fields: `s_e_step`, `e_s_mu_lr`, `e_s_sigma_lr`; validation); `vfe3/model/prior_bank.py` (s-tables gate, r gate, `encode_s`, `TODO(B)` comment); `vfe3/model/model.py` (`_refine_s`, forward anchor + supersede guards, diagnostics anchor, `TODO(B)` comment); `tests/test_live_s_model_channel.py` (17 tests). Spec: `docs/superpowers/specs/2026-06-08-live-s-model-channel-design.md`; plan: `docs/superpowers/plans/2026-06-08-live-s-model-channel.md`.

**To ablate (user adds to ablation.py).** A sweep entry of the form

```python
    "s_e_step": {
        "param": "s_e_step", "values": [False, True],
        "fixed": {"prior_source": "model_channel", "lambda_h": 1.0, "gamma_coupling": 1.0},
    },
```

plus companion sweeps over `e_s_mu_lr` and `e_s_sigma_lr`, following the schema of existing entries.

**Verification.** Full suite: `tests=731, failures=0, errors=0, skipped=0` (731 passed, 1 xpassed).

## Audit-doc status sync — `docs/audits/audit-2026-06-07-lifecycle-multiagent.md`

Doc-only (no code change). Marked the verified findings closed since the audit: **V2** (`close_basis`
wired, `model.py:62-64` + `geometry/closure.py`), **V3**/**V4** (freeze warning now covers the
`unroll`+oracle route and `pos_phi='learned'`, `config.py:736-766` + `model.py:204-214`), **V6**
(`free_energy_terms` threads `alpha_reg`/`include_attention_entropy`, `metrics.py:104-150`); and
Tier-2 #4 **checkpoint resume** (`run_artifacts.py:145,174` + `train(resume_from=...)`). Added a
Status column + status block; annotated resolved (not deleted) to keep the forensic record. V5 (low,
doc-only) and V1 (refuted) left open/n/a.
