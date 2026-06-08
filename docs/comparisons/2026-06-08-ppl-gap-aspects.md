# VFE_2.0 → V3: structural aspects behind the ~15 PPL gap (WikiText-103)

Date: 2026-06-08. Method: 8-subsystem fan-out reading BOTH repos under the *matched winning-run
config* (V2 `system_info.json` vs V3 `train_vfe3.py`), each finding pipelined into an adversarial
verifier that attacked impact and structurality. Matched config: linear decode
(`use_prior_bank=False`), K=20, L=1, heads=2, T=1, diagonal, flat transport, `decode_n_components=1`,
`gauge_fixed_priors=false`, head mixer on, KL divergence, seed 6.

## Bottom line

The gap (V3 ~154 vs V2 ~140 test PPL) is a **real model-quality gap, not a measurement artifact**:
the data-eval verifier loaded the actual `wikitext-103_test_tiktoken_tokens.pt` cache and confirmed
both repos score PPL on the **byte-identical** token stream, non-overlapping 128-windows, token-CE,
and `exp(min(ce,20))` formula (one off-by-one window out of ~2190 — negligible). So closing it means
adopting V2 structure, and the candidates reduce to **three**, two prominent suspects are **refuted**.

## Incorporate these (confirmed structural movers, matched-config-active, LR-independent)

1. **`decode_bias` — DONE (2026-06-08).** Learned per-vocab log-unigram bias on the linear decode
   (`logits = mu_q @ W^T + b`), zero-init, weight-decay-free. V2 had it (`model.py:102-106`,
   winning run `decode_bias=true`); V3 had no bias and no field. Highest-confidence lever at K=20,
   V=50257. Implemented; measurement held per user.

2. **Gauge-frame weight-decay protection.** V2 routes `phi_embed`/`pos_phi` to a dedicated `m_phi`
   group pinned at `weight_decay=0` (`trainer.py:305-306,377,386`). V3 only sets `weight_decay=0`
   on the phi/pos groups **when `m_phi_natural_grad=True`** (`train.py:74-77`); at the matched config
   (`m_phi_natural_grad=False`) the phi group inherits `cfg.weight_decay=0.065`, so **V3 decays its
   gauge frames toward identity transport** (verified empirically: `phi_embed wd=0.065`). Decoupled
   AdamW decay is gradient-independent, so the equilibrium `|phi*| ~ E[normalized-grad]/wd` is set by
   `wd` **alone** — NOT absorbable by the M-step LRs the user sweeps. Verdict: confirmed → medium.
   Mechanism is load-bearing: it suppresses the very gauge mechanism the model is built on.
   *Fix:* in `vfe3/train.py build_optimizer`, set `weight_decay=0.0` on the phi group (and the
   pos_phi_free group) **unconditionally**, not only under `m_phi_natural_grad`. One-line change;
   default-safe (phi should never be L2-pulled toward identity).

3. **E-step mu trust-region.** V2 passes every mean update through `apply_mu_trust_region(delta_mu,
   sigma, trust=5.0, mode='box')` — a per-coordinate ±5σ (whitened) box (`e_step.py:1445-1454`,
   `_numerics.py:145-166`); winning run `e_mu_q_trust=5.0`. V3's update is unbounded:
   `mu = belief.mu - e_mu_lr*nat_mu` with only an SPD trust on **sigma** (`inference/e_step.py:373`).
   At T=1 the coupling pull is the whole mu update and the linear decode reads only `mu_final`, so a
   present-vs-absent box changes the converged mean (hence logits). Verdict: confirmed → medium, but
   **impact depends on whether the box binds in normal training** — the open empirical question.
   *Fix:* add `e_mu_q_trust: Optional[float] = 5.0` + `mu_trust_mode='box'` to `VFE3Config`; in
   `e_step.py:373` clamp `delta_mu` to `±e_mu_q_trust * sqrt(belief.sigma)` before the retraction.

## Do NOT chase (refuted / inert at the matched config)

- **Measurement confound — NONE.** Identical token stream, windowing, CE, PPL formula (verified on
  the real cache). The gap is genuine.
- **RiemannianAdamW / Killing gauge optimizer — INERT.** V2's `killing_inv` is `(1/K)·I` (conformal
  scalar, zero off-diagonal); AdamW's per-coordinate normalization is invariant to any diagonal
  positive rescaling, so the Killing precond is a no-op under Adam (`trainer.py:445-462` vs
  `train.py` AdamW). Consequence: `m_phi_natural_grad`/`phi_precond_mode=pullback_per_block` only
  bite when `m_phi_natural_grad=True`; switching the precond alone on the default path changes
  nothing. (The non-conformal pullback metric *would* reshape the step — but only under natural-grad.)
- **Init scale (decode W ~20× smaller Xavier vs Kaiming; mu_embed ~20× larger 0.02 vs 0.001).**
  Real code difference, but the two factors cancel in the bilinear logit and Adam normalizes
  magnitude — both verifiers downgraded to none/low.
- **`normalize_ce_by_dim` (V2 true / V3 absent).** Training-side `1/√K` loss scale, captured BEFORE
  the scale for logging, so it does not change reported PPL; otherwise equivalent to an LR change.
- **Generators / flat transport / head-mixer placement / decode temperature / mixture / weight
  tying / final-norm.** Byte-identical or gated off at the matched config.

## Recommended next step

When the GPU is free, a within-V3 ablation over (a) `decode_bias`, (b) phi `wd=0`, (c) mu
trust-region, otherwise identical config, isolates each contribution. Expectation: `decode_bias`
dominant; phi-wd a clean LR-independent gain worth taking regardless; mu-trust the wildcard
(measure whether the box binds). Items (a) and (b) are safe to adopt now; (c) warrants the ablation
before committing a default.
