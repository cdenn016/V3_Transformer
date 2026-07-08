# Design: omega_direct Phase 3 — gamma / s-channel frame-fidelity

Date: 2026-07-08
Status: design, autonomous continuation (unblocks the user's active training run)
Branch: `feat/omega-direct-phase3`
Predecessors: Phase 1 (`2026-07-07-…`), Phase 2 (`2026-07-08-…-phase2-…`), both shipped to main.

## Motivation

The user's training run hit the Phase-1/2 config gate: `gauge_parameterization="omega_direct"` rejects
`lambda_gamma>0` / `s_e_step` / `gamma_as_beta_prior`. The gate exists because the gamma / model-coupling (s)
channel — the model belief `s_i = N(s_mu, s_sigma)` coupled by `gamma_ij KL(s_i || Ω_ij s_j)` — builds its
transport from the `exp(phi)` cocycle, not the stored frame `U`, so under omega_direct it would silently
transport the s-channel by the wrong frame. This phase gives the s-channel frame-fidelity and removes the gate.

## The core design

Under omega_direct there is exactly **one** per-token frame `U_i` (`belief.omega`, sourced from
`prior_bank.omega_embed` via the canonical, compact-storage-aware `_omega_lookup`, `prior_bank.py:346-371`).
The s-channel is defined to **reuse the belief frame** — `encode_s` returns only `(s_mu, s_sigma)`, there is no
`s_omega_embed` — so there is no new frame or storage; Phase 3 threads an **existing** tensor.

The decisive fact: **the downstream transport funnel is already omega-aware.** `_gamma_energy`
(`model.py:1194-1242`) already carries an `omega` param and gates `gp = cfg.gauge_parameterization if omega is
not None else "phi"`, then calls `build_belief_transport(..., gauge_parameterization=gp, omega=omega)`;
`build_belief_transport` dispatches `omega_direct` first (`e_step.py:178-187`, so `transport_mode="flat"` is
ignored on that branch); and the `e_step` funnel threads `gauge_parameterization` + `belief.omega` end-to-end.
The **only** defect is that the s-channel call sites drop the frame before reaching this machinery. So the work
is threading the frame to those sites (no new transport code):

1. **Forward gamma loss** (`lambda_gamma>0`): `_gamma_coupling_term` (`model.py:1244`→`_gamma_energy` at 1258)
   passes no omega. Add an `omega` kwarg; at the call site (`model.py:1102`) pass `belief.omega.detach()`
   (guarded), matching the existing `belief.phi.detach()` — this loss is deliberately frame-inert (its gradient
   reaches only the s tables through `encode_s`, not the frame), so **detach for parity**.
2. **Diagnostic gamma split**: `_gamma_coupling_terms` (`model.py:1267`→1290) + caller (`model.py:1758`, pass
   `out.omega`). Add the `omega` kwarg + forward.
3. **`_fold_gamma_prior`** (`gamma_as_beta_prior`, the forward-**value**-critical fold): `_fold_gamma_prior`
   (`model.py:1548`→`_gamma_energy` at 1576, under `no_grad`) + all callers — forward hot path (`model.py:816`,
   `beliefs.omega`) and diagnostic replays (`model.py:1328, 1655, 1934, 2047`, guarded `belief.omega`). This
   fold returns `log_prior` into `vfe_stack` (`model.py:821`), shaping the belief E-step softmax → `mu_final`
   → logits → CE, so a wrong frame here corrupts the **prediction**, not just a side loss. Under `no_grad`,
   attach-vs-detach is moot — pass `beliefs.omega` directly.
4. **`_refine_s`** (`s_e_step`, the s E-step — the disjunct the user trips): `_refine_s` (`model.py:594-674`)
   builds `BeliefState(mu=s_mu, sigma=s_sigma, phi=phi0)` with `omega=None` and calls `e_step` without
   `gauge_parameterization`, so it silently transports s by `exp(phi0)`. Fix (no caller edits): derive
   `omega_s = pb._omega_lookup(token_ids) if cfg.gauge_parameterization=="omega_direct" else None`, set
   `omega=omega_s` in the BeliefState, and add `gauge_parameterization=cfg.gauge_parameterization` to the
   `e_step` call. Keep `transport_mode="flat"` (ignored under the omega dispatch). **`omega_s` stays ATTACHED**
   — on the phi path `_refine_s` backprops into `phi_embed` through `exp(phi0)`, so for omega_direct to be a
   faithful re-chart of the *same* model, the s-refine must train `omega_embed` symmetrically; do NOT detach.
5. **viz/extract.py** three `gamma_as_beta_prior` folds (`extract.py:54, 211, 283`): sites 211/283 have
   `beliefs.omega` in scope — forward it. Site 54's `BeliefState` (`extract.py:46`) is built **without** omega
   (unlike its model.py siblings) — populate it (`omega=enc.omega[0] if enc.omega is not None else None`),
   then forward. (The `_refine_s` calls at extract.py:48/206/277 need nothing — item 4 re-derives internally.)
6. **Remove the config gate** (`config.py:941-955`, comment + raise) — **last**, after 1-5, so no combination is
   accepted before its transport path is frame-faithful. All three toggles un-gate together (each maps to one
   fixed path; `gamma_as_beta_prior` already requires `lambda_gamma>0`). The other omega_direct guards
   (`transport_mode='flat'`, `e_phi_lr=0`, eligible group, reflection) stay and pass for the target config.

## One gate STAYS (out of scope)

The runtime `NotImplementedError` at `e_step.py:330-333` (filtered/frozen-keys branch of `free_energy_value`,
where `_transport_qk` hand-builds `Ω` from phi and cannot read `belief.omega`) is a **belief-channel
diagnostic** (`keys=belief`), not on any s-channel route. Keep it and its test
(`test_omega_direct.py:136-151`). Giving the filtered diagnostic frame-fidelity is a separate optional item.

## Byte-identity (the two must-not-change paths)

(i) `gauge_parameterization="phi"`: every new `omega` arg is `None`, every `gp` falls to `"phi"`, so all calls
reproduce today's. (ii) The shipped omega_direct **belief** path with the gamma channel OFF (`lambda_gamma=0`,
`s_e_step=False`, `gamma_as_beta_prior=False`): none of `_gamma_coupling_term`/`_gamma_coupling_terms`/
`_fold_gamma_prior`/`_refine_s` is entered. Guard every forward with `... if X.omega is not None else None`
(template `model.py:1344`). **Footgun**: `build_transport_from_element(None, group)` raises `AttributeError`
(`omega.double()`), so frame-present ⇔ `gp=="omega_direct"` must stay coupled — never `omega=None` with
`gp="omega_direct"`.

## Gradient parity (the load-bearing correctness point)

- `_refine_s`: `omega_s` **attached** (trains `omega_embed`, mirrors `phi_embed`). The `e_step` funnel already
  keeps `omega` attached across iterations + the truncation boundary (`e_step.py:920-925, 937`). Under
  `e_step_gradient="detach"` the whole forward E-step runs `no_grad` (`model.py:760-761`) and `_refine_s`
  inherits it — symmetric on both charts; do not hard-freeze beyond the estimator.
- Forward gamma loss (`model.py:1102`): **detached** (`belief.omega.detach()`), frame-inert, parity with
  `phi.detach()`.

## Scope / tests

**This phase:** the 6 items above + tests + the pre-existing `gauge_group` ablation-arm bug fix. **Deferred
(unchanged):** the STE learnable det-sign (needs the user's two-tier decision), tower compact storage, the
`so_n` ρ(O(N)) seed, the filtered-diagnostic frame-fidelity, and `e_phi_lr>0`.

Tests: invert `test_omega_direct_rejects_active_gamma_channel` (all three now construct); keep
`test_free_energy_value_filtered_keys_rejects_omega_direct`; add (a) a differential "s-channel uses U not
exp(phi)" test per subsystem (hold phi fixed, vary U between I and U2, assert the gamma energy / refined s
differ), (b) a finite-forward at the user's target (`omega_direct, lambda_gamma=0.75, s_e_step=True,
gamma_as_beta_prior=True, prior_source="model_channel", family="gaussian_diagonal"`), and (c) gauge invariance
with gamma ON (co-transform s/r tables + `mu_embed`/`omega_embed` by `U→gU`, decode invariant to fp64).
Ablation: drop the `lambda_gamma=0/s_e_step=False` overrides from the omega cells so they inherit the baseline;
flip the assertions. Pre-existing `gauge_group` bug: `tied_block_glk` (`ablation.py:387`) add
`phi_precond_mode="killing"`; `so3_spin2x4` (`ablation.py:391-392`) add `n_heads=4`.

## Risks

The forward hot path is **not** already frame-faithful in the default case: `shared_omega` carries the frame
only when `share_refine_s_transport` (default OFF), so on the default path `_refine_s` builds from phi with
`gp="phi"`, `omega=None` — item 4 fixes it regardless of that toggle. The diagnostic/viz replays must reproduce
the same frame the forward uses (item 5's `extract.py:46` populate), or figures diverge from the forward. All
new gradient is either attached-by-design (`_refine_s`), detached-by-design (gamma loss), or under `no_grad`
(`_fold_gamma_prior`).
