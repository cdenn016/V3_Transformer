# Learnable reflection on the phi path — `R·exp(φ)` (design)

**Date:** 2026-07-08
**Branch:** feat/phi-reflection (off main 8f9717c)
**Status:** design approved (conversational); implementation pending
**Sibling:** `docs/superpowers/specs/2026-07-08-omega-direct-metropolis-detsign-design.md` (the `omega_direct` learnable det-sign). This design gives the SAME learnable discrete orientation to the default `phi` parameterization, cheaply, by prepending a reflection to `exp(φ)`.

## 1. Idea

The `phi` parameterization stores each token's frame as `gᵢ = exp(φᵢ)`, which always lands in the identity component `GL⁺(K)` because `det exp(φ) = e^{tr φ} > 0`. Prepending a per-token discrete reflection `Rᵢ` (with `det Rᵢ = −1`) gives

$$ g_i = R_i\,\exp(\phi_i), \qquad \det g_i = \det(R_i)\,e^{\operatorname{tr}\phi_i} = -\,e^{\operatorname{tr}\phi_i} < 0 \text{ when } R_i \text{ is a reflection}, $$

so the frame now reaches **both** orientation components while `exp(φ)` keeps carrying the continuous part. `Rᵢ` is a single per-token bit, learned by the SAME ΔF-gated Metropolis move as the `omega_direct` det-sign. This is the coset decomposition `GL(K) = {I, R}·GL⁺(K)`.

**Transport stays flat.** `Ωᵢⱼ = gᵢgⱼ⁻¹ = Rᵢ exp(φᵢ)exp(−φⱼ)Rⱼ` (using `R⁻¹ = R`); the cocycle `Ωᵢⱼ Ωⱼₖ Ωₖᵢ = I` holds for any `g`, so holonomy still vanishes.

**Reach vs `omega_direct`.** `R·exp(φ)` reaches `R·image(exp)`, not the full `GL(K)` — it misses the non-exp interior of `GL⁺(K)` (matrices with odd-multiplicity negative real eigenvalues), which `omega_direct`'s stored `U` reaches. Marginal for near-identity frames; it is the honest theoretical difference. In exchange it is far lighter: `n_gen` params + **1 sign bit** per token, reusing the mature φ machinery (BCH, Killing, positional-φ).

**Efficacy caveat (same as `omega_direct`).** Under a diagonal covariance family `R=diag(−1,1,…)` leaves the (squared) diagonal congruence exactly invariant, so the reflection bites only through the mean (sign of component 0), not the covariance. Real covariance information only with a full/off-diagonal family or a low temperature. Warned at construction (mirrors `omega_direct`).

## 2. Scope (full) and deferrals

**Full channel coverage** (user-selected). The reflection folds into the belief transport (`build_belief_transport`, phi branch) — covering the belief E-step and the decode — AND is threaded through the gamma / model-coupling (s) channel (the forward gamma loss `_gamma_coupling_term`, the `gamma_as_beta_prior` fold `_fold_gamma_prior`, and the `s_e_step` E-step `_refine_s`), exactly as Phase 3 threaded `omega` for `omega_direct`. So `phi_reflection` is usable together with an active gamma channel; there is NO gamma gate. Learnable via `phi_reflection="metropolis"`; default `"off"` is byte-identical.

The mechanism is uniform: everywhere the phi-path transport is built from `belief.phi`, the per-position `belief.reflection` is passed alongside so the §3 fold applies. Because the fold lives in `build_belief_transport`, each channel's ΔF and forward value automatically see the reflected transport.

**Deferred:** RoPE / regime_ii interactions (target `transport_mode="flat"`, `pos_rotation="none"`; reflection + RoPE composition is a later concern), and the STE variant (`# TODO(STE)`).

## 3. The reflection fold (group-agnostic)

`R = reflection_element(K) = diag(−1,1,…,1)` at the K level for every eligible group (for `block_glk` this is block 0's `diag(−1,1,…)`; for `so_k` it is the `O(K)\SO(K)` reflection). In the `FactoredTransport` (`vfe3/geometry/transport.py:43`, factors `exp_phi = exp(φᵢ)`, `exp_neg_phi = exp(−φⱼ)`):

- `exp_phi[i] ← Rᵢ @ exp_phi[i]` = negate **row 0** of `exp_phi[i]` iff `sᵢ = −1`.
- `exp_neg_phi[j] ← exp_neg_phi[j] @ Rⱼ` = negate **column 0** of `exp_neg_phi[j]` iff `sⱼ = −1`.

Then `Ω = exp_phi @ exp_neg_phi = Rᵢ exp(φᵢ)exp(−φⱼ)Rⱼ`, and every downstream factored/dense contraction (mean transport `Ωμ`, covariance sandwich `ΩΣΩᵀ`) is correct with no further change. The fold is applied inside `build_belief_transport`'s phi branch when a per-position sign vector is present.

## 4. Data flow

- **`prior_bank.reflection_sign`** — a registered buffer of shape `(V,)`, values in `{+1,−1}` (NOT an `nn.Parameter`; it is discrete state updated by the Metropolis move, not gradient), created only when `gauge_parameterization=="phi"` and `phi_reflection!="off"` (idiom of the gated `omega_embed`). Default all `+1` (identity, `det>0`). `"init_seed"` seeds every other token to `−1`.
- **`BeliefState.reflection`** — a new trailing `Optional[torch.Tensor] = None` field (per-position `(…,N)` sign, `+1`/`−1`), populated at `encode` from `reflection_sign[token_ids]` when the buffer exists; `None` on the pure path. Mirrors the `omega` field addition.
- **`build_belief_transport(phi, group, *, reflection=None, …)`** — new keyword; when `reflection` is not None and `gauge_parameterization=="phi"`, apply the §3 fold to the built factors. `None` (default) is byte-identical.
- ALL phi-path transport call sites pass `reflection=belief.reflection` (guarded `... if belief.reflection is not None else None`) alongside `phi`: the belief E-step, the decode transport, AND the gamma/s-channel — `_gamma_coupling_term`/`_gamma_energy` (forward gamma loss + diagnostics), `_fold_gamma_prior` (the `gamma_as_beta_prior` fold), and `_refine_s` (the `s_e_step` E-step). This mirrors the Phase 3 `omega` threading map one-for-one (same call sites), with the same detach/no_grad/attached gradient discipline per channel (gamma loss detaches the sign is moot — the sign is a non-differentiable buffer, so no gradient flows through `reflection` on any path; it is updated only by the Metropolis move).

## 5. The Metropolis move (shared with `omega_direct`)

Reuse the sweep/accept/seed structure of `metropolis_omega_step`. The move dispatches on the active reflection mode:

- **`omega_direct` + `omega_reflection="metropolis"`** (existing): flip `omega_embed[i]`, ΔF via the omega transport.
- **`phi` + `phi_reflection="metropolis"`** (new): flip `reflection_sign[i]` (`s → −s`), ΔF via the phi transport with the flipped sign (a fixed-belief block move at the converged beliefs, exactly as before), accept `min(1,e^{−ΔF/T})` from the seeded generator, flip the buffer in place on accept.

Factor the common sweep (unique-token loop, carried `F_cur`, seeded accept) so both modes share it; only the per-token *flip* and the *trial-frame construction* differ. The train-seam (`_maybe_metropolis_omega`) fires whenever EITHER reflection mode is `"metropolis"`; `omega_metropolis_temperature` / `omega_metropolis_every` govern both (rename-free reuse).

## 6. Config surface

- `phi_reflection: "off" | "init_seed" | "metropolis"` (mirrors `omega_reflection`; default `"off"`). Validation, inside a `gauge_parameterization=="phi"` guard:
  - requires `gauge_group in _REFLECT_OK = ("glk","block_glk","so_k")` (reject `sp`/`sp_n` vacuous `det≡+1`; `so_n`/`tied_block_glk` deferred reflection seed — same reasons and messages as `omega_reflection`).
  - (no gamma gate — the reflection is threaded through the gamma/s-channel, §2/§4.)
  - a `UserWarning` for a diagonal covariance family (§1 efficacy caveat).
  - reject `phi_reflection="ste"` (`NotImplementedError`, `# TODO(STE)`), mirroring `omega_reflection`.
- The two `omega_metropolis_*` knobs are reused (they name the move, not the storage). Default `"off"` ⇒ no buffer, no move, no RNG, `state_dict` byte-identical.

## 7. Testing (TDD, CPU-bound, K<6)

1. `phi_reflection` config: constructs for `glk`/`block_glk`/`so_k` (including with an active gamma channel); rejects `sp`/`sp_n`/`so_n`/`tied`; rejects `"ste"`; requires the phi path.
7. Gamma/s-channel frame-fidelity: with `phi_reflection` on and a non-`+1` sign, the gamma-coupling energy and the `_refine_s` refined `s` differ from the all-`+1` frame (the reflection is USED in each channel), mirroring the Phase 3 `omega` frame-fidelity tests; and a gauge-invariance check with the reflection co-transformed stays invariant.
2. Reflection fold correctness: for a fixed φ and a `−1` sign on token i, the built `Ωᵢⱼ` equals `R @ exp(φᵢ)exp(−φⱼ) @ R` computed independently (fp5), and `det Ωᵢⱼ < 0` for `sᵢ≠sⱼ`.
3. Frame reaches `det<0`: with `phi_reflection="init_seed"`, `det(gᵢ) < 0` for seeded tokens and the decode differs from the all-`+1` frame (the reflection is USED).
4. Metropolis on the sign: exact-ΔF anchor (masked-flip ΔF == independent `reflection_sign`-flip recompute); off-is-noop (no buffer, no RNG, `state_dict` identical); downhill accepted flips the sign; uphill gated; seeded-reproducible.
5. Byte-identity: `phi_reflection="off"` leaves the phi path and `state_dict` unchanged; a full `train()` under `"off"` is unaffected.
6. End-to-end: a tiny `train()` with `phi_reflection="metropolis"`, `gaussian_full` (so the covariance channel bites), runs finite and flips signs across steps.

## 8. Implementation inventory

1. `vfe3/config.py` — `phi_reflection` field + validation (§6).
2. `vfe3/belief.py` — the trailing `reflection` field on `BeliefState`.
3. `vfe3/model/prior_bank.py` — the gated `reflection_sign` buffer + `encode` populating `belief.reflection`.
4. `vfe3/geometry/transport.py` / `vfe3/inference/e_step.py` — `reflection` kwarg on `build_belief_transport` (phi branch) applying the §3 fold; ALL phi-path transport call sites (belief E-step, decode) pass `belief.reflection`.
5. `vfe3/model/model.py` — thread `belief.reflection` through the gamma/s-channel call sites (the Phase 3 `omega` map: `_gamma_coupling_term`/`_gamma_energy`, `_fold_gamma_prior`, `_refine_s` + their callers/diagnostics); generalize the Metropolis move to the phi-reflection mode (shared sweep, mode-specific flip + trial frame); the `# TODO(STE)` marker.
6. `vfe3/train.py` — the seam already fires on `"metropolis"`; extend its gate to include `phi_reflection`.
7. `tests/test_phi_reflection.py` — §7.

No change to the pure `phi_reflection="off"` path or the shipped `omega_reflection` modes.
