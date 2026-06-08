# Design spec: a live model channel `s` that feeds the belief as its prior (dynamic fiber tie)

**Date:** 2026-06-08
**Status:** approved design, pre-implementation
**Roadmap:** lifecycle-audit Tier-1 #1 (`docs/audits/audit-2026-06-07-lifecycle-multiagent.md`)
**Scope:** option **A** only (live `s` that feeds `q`); the meta-agent / shadow-prior (option **B**)
is explicitly out of scope and is marked with `TODO(B)` pointers.

## Motivation

The manuscript (`Participatory_it_from_bit.tex`) runs four statistical fields in two parallel stacks:
beliefs `q` and priors `p` on the **state fiber**, and generative models `s` and hyper-priors `r` on
the **model fiber**. Beliefs are the fast, per-occurrence representation a token settles into in
context; the model `s_i` is the token's slow, standing disposition — its theory of how it relates to
its neighbors, independent of the current sentence. The belief channel couples neighbors through
`beta`-attention over states; the model channel couples them through `gamma`-attention over
dispositions.

In the present codebase the model channel exists only as scaffolding (committed 2026-06-02). The
`s`-tables, the `lambda_h * KL(s||r)` hyper-prior term, the `gamma_coupling` model-coupling block, and
the `prior_source='model_channel'` static tie are all present, but `s` is **predictively inert**: the
`gamma`-block transport is built from `out.phi.detach()` (`model.py:541`) and computed at the loss
level outside the E-step, so `s` trains its own tables and never influences the belief the decode
reads. This spec makes `s` a **live field with its own E-step** whose refined value becomes the
belief's prior.

**Honest framing.** The manuscript freezes the slow subsystem `(s, r)` in its own language-model
experiments (`gamma_ij = 0`, line 636) and places the per-token *linguistic* meaning of `s`/`r` out of
scope (line 1632). The math is fully specified; the interpretation is not. This build is therefore a
deliberate, documented design choice — making the model channel live and observing what a per-token
disposition learns to be — not a transcription of a manuscript result. It is an opt-in toggle; the
pure path keeps the manuscript's frozen-slow-channel behavior.

## Coupling mechanism (and why it had to change)

The first draft coupled `s` to `q` through the **shared gauge frame** (the `gamma`-block added to the
`phi` E-step objective). That mechanism is **inert at the operative config `n_e_steps=1`**: in
`e_step_iteration` the `q`-update runs first (using `Omega` from the start-of-iteration `phi`) and the
`phi`-step runs last, so a frame bent by `s` lands only in `out.phi`, which no further iteration
consumes; decode reads only `mu_final`/`sigma_final` (`model.py:414-420`, decode is `phi`-free,
`prior_bank.py:237-284`). At `n_e_steps=1` the shared-frame route therefore never reaches the loss.

This spec instead uses the **dynamic fiber tie** (manuscript line 1399, "the model-channel terms do
not contribute unless the belief and model fibers are explicitly tied"): the refined `s_i` becomes the
belief's prior `p_i`. Because the belief E-step self-couples to its prior on **every** iteration —
including the single iteration at `n_e_steps=1` — `s` reaches `mu_final` and the cross-entropy directly.
This is the live, per-context generalization of the existing static `prior_source='model_channel'` tie
(which uses the raw `s`-table as the prior); here the prior is the `s`-table **refined** by a model-
channel E-step before it is used.

## Invariant: the default is completely off (byte-identical)

This is a hard requirement, not merely a default value. With the master toggle `s_e_step=False`
(the default):

- No model-channel belief state is constructed or refined; the belief E-step takes the exact prior it
  takes today (the token table, or the `s`-table under the pre-existing `prior_source='model_channel'`).
- No new parameter is created on the default path, so no RNG is consumed and every existing table is
  byte-unchanged (the `s`-tables, when created, are drawn **last**, after every belief table).
- The existing loss-level `lambda_h` / `gamma_coupling` terms (`model.py:484-559`) behave exactly as
  before (gated on their own weights, independent of `s_e_step`).

A golden byte-identity test pins this: a fixed seed and config with `s_e_step=False` must produce
logits and loss identical (bitwise, fp32) to the pre-change build. The new `e_s_*_lr` fields are read
only when `s_e_step=True`, so their defaults cannot perturb the pure path.

## Architecture: refine `s`, then feed it in as the belief's prior

When `s_e_step` is on, the block runs two E-steps in sequence, sharing the encoded gauge frame `phi0`:

```
1. encode:    q0, phi0 <- prior table lookup;  s0 <- encode_s(token_ids)          # (B,N,K)
2. s-refine:  s1 <- e_step(s0, self=r, transport=Omega_tilde(phi0),               # NEW model-channel
                           self_weight=lambda_h, couple=gamma_coupling,           #   E-step; phi0 FIXED
                           lr=(e_s_mu_lr, e_s_sigma_lr), n_iter=n_e_steps)         #   updates (mu_s,sigma_s)
3. belief:    p <- s1;  q0 <- s1;                                                  # the fiber tie
              q1 <- e_step(q0, self=p=s1, transport=Omega(phi0), ... )            # UNCHANGED machinery
4. decode:    logits/CE from mu_final=q1.mu, sigma_final=q1.sigma                  # refs = s-table (V,K)
```

The decisive reuse: step 2 is the belief E-step machinery (`e_step` / `e_step_iteration` ->
`belief_gradients` -> `natural_gradient` -> retraction) applied to the model channel — `belief_gradients`
is already channel-agnostic. The model channel supplies `r` as the self-target (`alpha_mode='constant'`,
`value=lambda_h` — not the belief's state-dependent alpha), `Omega_tilde(phi0)` as transport,
`gamma_coupling` as the coupling weight, and `kappa_gamma`/`gamma_attention_prior` for the softmax. The
`s`-tables are diagonal `(V, K)`, so the `s`-refine uses `DiagonalGaussian` regardless of `cfg.family`
(matching the existing `gamma`-block). Step 2 holds `phi0` **fixed** (it updates only `mu_s, sigma_s`),
so the belief E-step in step 3 is structurally identical to today — only its prior input changes.

At `n_e_steps=1` this is live: `q0 = p = s1`, so the belief's self-coupling `KL(q0||p)` is zero at init
and `mu_final = s1 - e_mu_lr * (coupling gradient)` — i.e. `mu_final` is the refined disposition shifted
by neighbor coupling. The cross-entropy depends on `mu_final`, which depends on `s1`, which depends on
the `s`-tables; gradient flows back to the `s`-tables through the unrolled belief E-step. The channel
is not dead at T=1.

## Training and the frozen `r`

Under live `s`, the model channel earns its keep by improving next-token cross-entropy: the `s`-tables
train through `s-refine -> s1 -> belief prior -> belief E-step -> decode -> CE`. The existing
**loss-level `lambda_h` and `gamma_coupling` regularizer terms are superseded when `s_e_step` is on**
— they were the inert stand-ins for the not-yet-built E-step, and the `lambda_h * KL(s||r)` /
`gamma`-consensus forces now live inside the `s`-refine instead. When `s_e_step` is on, `model.py` skips
those two loss blocks; the converged `s1` remains available for diagnostics.

`r` stays **frozen** (`requires_grad=False`), consistent with the manuscript's treatment of the
top-scale hyper-prior as a fixed boundary condition (line 554) and with the existing collapse-guard
note (`prior_bank.py:180`: freely training a single global `r` alongside `s` drives `KL(s||r) -> 0`).
A genuinely token-dependent, trainable `r` requires a meta-agent to transport it down
(`r_i = Omega_tilde[s_I^{(s+1)}]`); that is option **B**, out of scope here.

## Components touched

- **`vfe3/config.py`** — new fields and validation (see Config surface). The `s_e_step` toggle plus a
  warning when it is on with both `lambda_h == 0` and `gamma_coupling == 0` (the `s`-refine would have no
  force, `s1 == s0`, and the channel reduces to the static `prior_source='model_channel'` tie).
- **`vfe3/model/prior_bank.py`** — extend the table-creation gate so the `s`-tables are created when
  `s_e_step` is on (today: `lambda_h>0 or gamma_coupling>0 or prior_source=='model_channel'`); `r` is
  created (frozen `(K,)`) whenever `s_e_step` is on, so the `s`-refine always has a well-defined
  self-target and `lambda_h=0` simply zeroes its pull. The `s`-tables are created **last** (after every
  belief table, as today), so the belief tables are byte-identical with or without `s_e_step`. `encode_s`
  is reused unchanged to seed `s0`.
- **`vfe3/inference/e_step.py`** — no structural change. The `s`-refine is a call into the existing
  `e_step` with model-channel arguments (`r` as the prior, constant alpha, `gamma`/`kappa_gamma`/
  `gamma_attention_prior`, `e_s_*_lr`). If a separate model-channel helper is cleaner than overloading
  `e_step`'s keyword surface, add a thin `model_refine` wrapper that forwards to `e_step_iteration`;
  either way the gradient kernel and retraction are reused, not reimplemented.
- **`vfe3/model/block.py`** — when `s_e_step` is on: look up `s0` via `encode_s`, run the `s`-refine to
  `s1`, and pass `s1` as the belief E-step's `(mu_p, sigma_p)` **and** initial belief `q0` (overriding
  the prior-table lookup). Off: unchanged.
- **`vfe3/model/model.py`** — thread `s_e_step` and the new lrs into the block call; under `s_e_step`,
  skip the superseded loss-level `lambda_h`/`gamma` blocks; place the `TODO(B)` pointer at the
  frozen-`r` definition and the prior-handoff / shadow-prior site.
- **`vfe3/viz/extract.py`** — thread the new knobs for trajectory-extraction parity (accept-and-ignore
  where the diagnostic value function does not consume them).

Decode references (`pi_v`, the `(V, K)` `s`-table under `prior_source='model_channel'`) are unchanged:
`s1` is a per-position belief `(B, N, K)`, not a per-vocab table, so it feeds the belief prior, not the
decode reference set. No change to families, retractions, transport regimes, the decode kernels, or the
M-step.

## Config surface (minimal)

- `s_e_step: bool = False` — master toggle; gates the `s`-refine and the prior override. The pure path.
  **Requires `prior_source='model_channel'`** (config validation): the `s`-tables then serve as the
  model's vocab table for both encode and decode, so the belief is anchored to `s` *and* decoded
  against `s` (one coherent table); `s_e_step` adds the live per-context refinement on top of that
  static tie. With this, `e_s_lr=0` collapses exactly to the existing static `model_channel` path.
- `e_s_mu_lr: float` and `e_s_sigma_lr: float` — the model-channel refine learning rates. Small values
  realize the manuscript's slow channel (the limit `e_s_lr -> 0` gives `s1 = s0`, i.e. the static
  `prior_source='model_channel'` tie); defaults mirror the belief E-step rates (proposed `0.1`, to be
  tuned down in sweeps). Inert when `s_e_step=False`. Validated `>= 0`.
- Reuses existing fields: `lambda_h` (the `s->r` self-coupling weight), `gamma_coupling` (the `s->s`
  coupling weight), `kappa_gamma` (its softmax temperature), `gamma_attention_prior` (its attention
  prior). The `s`-refine reuses `n_e_steps` for its iteration count.

## Testing strategy

- **Default-off golden (the invariant):** fixed seed, `s_e_step=False`; logits and loss bitwise-equal
  to the pre-change build; no new parameter created (parameter count and RNG draw unchanged).
- **`e_s_lr = 0` reduction:** `s_e_step=True` with `e_s_mu_lr = e_s_sigma_lr = 0` gives `s1 = s0`, so
  the forward is bitwise-equal to a static `prior_source='model_channel'` run at the same seed — the
  *only* new behavior is the refinement, and it vanishes when the refine lr is zero.
- **Positive coupling test (liveness at T=1):** `s_e_step=True`, `n_e_steps=1`, nonzero `e_s_*_lr`, and
  `gamma_coupling>0` (or `lambda_h>0`) **changes `mu_final`/logits** versus the static-`s` run — the
  refinement reaches the decode in a single iteration.
- **Gradient-to-`s`-tables test:** under `s_e_step` at `n_e_steps=1`, the `s`-tables receive a nonzero
  gradient from the CE loss (guards against the dead-channel failure the shared-frame route had).
- **Property tests:** model-channel self-divergence `D(s||r) >= 0` and `= 0` at `s = r`; the `s`-refine
  is a descent step on the model-channel free energy (finite-difference check against the
  autograd-of-`F` oracle, as the belief channel is checked).
- **Gauge equivariance:** verify the `s`-refine-then-belief-E-step forward is equivariant under the
  group action (the `s`-refine and belief step share `phi0` and the same transport machinery; confirm
  the fiber tie does not break it — written as a check, not an assumed inheritance).
- **Supersede check:** with `s_e_step=True`, the loss-level `lambda_h`/`gamma` terms are not added (no
  double-count); with `s_e_step=False` they are added exactly as before.

## Documented departure (timescale)

Refining `s` per-forward departs from the manuscript convention of holding the slow channel fixed
*during* inference (the slow timescale lives in the across-training / meta-agent dynamics). This is
recorded as an opt-in departure recovered to faithful behavior by a small `e_s_*_lr` — the limit
`e_s_lr -> 0` is the frozen-`s` (static-table) tie, and `s_e_step=False` is the manuscript behavior
exactly.

## Out of scope (option B and beyond)

- **Meta-agent / scale-(s+1) hierarchy** and a genuinely token-dependent, top-down-derived
  `r_i = Omega_tilde[s_I^{(s+1)}]`. Marked with `TODO(B)` at the frozen-`r` definition
  (`prior_bank.py:185`) and the prior-handoff site, pointing to a future spec.
- **Training the global `r`** (collapse hazard without a meta-agent).
- **The shared-frame coupling** (`gamma`-block in the `phi` E-step, manuscript line 1420). It is inert
  at `n_e_steps=1`; if pursued later it activates at `n_e_steps>=2` and is additive to the fiber tie.
- **The canonical observation-likelihood term** `-E_q[log p(o|k)]`, itself gated on the shadow prior
  (`docs/2026-06-07-observation-likelihood-term-brainstorm.md`).
- **A distinct model-fiber representation for `Omega_tilde`.** Reuses the existing single representation
  (as the current `gamma`-block does); a separate model-fiber irrep is a later seam.

## Open decisions to confirm

1. Toggle name `s_e_step` (versus `live_s` / `couple_s_to_q`).
2. Default `e_s_mu_lr`/`e_s_sigma_lr` values (proposed `0.1`, mirroring the belief rates; inert at the
   default-off path regardless).
3. Whether the `s`-refine reuses `n_e_steps` for its iteration count (proposed) or gets its own
   `n_s_steps`.
