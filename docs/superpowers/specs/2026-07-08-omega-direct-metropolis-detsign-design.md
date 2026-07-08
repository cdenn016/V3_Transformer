# omega_direct learnable det-sign — ΔF-gated Metropolis flip (design)

**Date:** 2026-07-08
**Branch:** feat/omega-metropolis-detsign (off main)
**Status:** design approved; implementation pending
**Predecessor:** `docs/superpowers/specs/2026-07-07-omega-direct-gauge-parameterization-design.md` §3.4 (the discrete det-sign component) and §4 (config surface). This design implements the deferred "option 3" (learnable det-sign) as the **ΔF-gated Metropolis flip**. The straight-through-gradient (STE) variant remains deferred behind a `# TODO(STE)` marker.

## 1. Background and scope

For `GL(K)` and `O(K)`, the group has two disconnected sheets separated by `det U = 0`. As shown in the predecessor spec §3.4, this is a **free-energy barrier**, not merely topology: as `det U -> 0` the congruence `Sigma -> U Sigma U^T` collapses a covariance direction and every `Sigma^{-1}` / `log det Sigma` term diverges (clamped to `kl_max`, a flat plateau with vanishing gradient). Gradient-based VFE descent therefore cannot cross sheets; a token stays in whichever det-sign it was initialized in. The frame decomposes as `U = R . U^0`, with `U^0` the continuous part (already handled by the Riemannian retraction, shipped in omega_direct Phases 1-3) and `R` a discrete reflection representative (`det R = -1`).

Today `omega_reflection` offers `"off"` (pure `det > 0` field) and `"init_seed"` (per-token `det < 0` fixed at initialization). This design adds `"metropolis"`: the det-sign becomes **learnable** by a ΔF-gated Metropolis move, so a token can migrate between sheets during training when doing so lowers the free energy.

**Principle.** The total free energy `F` is the single shared objective. Gradient descent already minimizes `F` over the continuous frame coordinates (`GL^+` via the retraction); the Metropolis move minimizes the *same* `F` over the discrete det-sign (`pi_0(G) = Z/2`). This is a coordinate-descent split: continuous coordinates by gradient, the discrete det-sign by accept/reject. The two never interfere — the Lie-exp retraction stays in the identity component (never crosses `det = 0`), so the det-sign changes **only** through this move.

**Out of scope (deferred, marked in code):** the straight-through-gradient (STE) variant of the learnable det-sign. `omega_reflection = "ste"` raises `NotImplementedError`; a `# TODO(STE)` comment at the move site records the alternative.

## 2. The move

Run per training step (subject to a cadence knob), **after** the continuous gradient / M-step, under `torch.no_grad()`. It is a **single-site sequential** Metropolis sweep over the unique token ids present in the current batch:

1. Run a no_grad `forward_beliefs` on the batch to obtain the current converged beliefs `q` (carrying `belief.omega` = the per-position frames looked up from `omega_embed[token_ids]`) and the prior `(mu_p, sigma_p)`. Evaluate `F_cur = free_energy_value(q, mu_p, sigma_p, ...)` once.
2. For each unique token id `i` in the batch, **in sequence**:
   - Propose the flip `U_i -> R . U_i`, where `R = reflection_element(...)` (`det R = -1`; `R` is an involution, so the proposal is symmetric and the Metropolis-Hastings ratio reduces to the plain Metropolis acceptance — no proposal-density correction).
   - Form the trial belief `q'` by applying `R` to `belief.omega` at every position where `token_ids == i` (a masked left-multiply), leaving all other positions untouched. Evaluate `F_trial = free_energy_value(q', ...)`. `Delta_F = F_trial - F_cur`.
   - Accept with probability `min(1, exp(-Delta_F / T))`, drawn from a seeded generator. On accept: mutate the source table `omega_embed[i] <- R . omega_embed[i]` in place (respecting the compact/full storage layout, see §3), set `F_cur <- F_trial`, and carry the flipped belief forward (so the next token's `Delta_F` is measured against the post-accept state — a correct MCMC chain). On reject: leave `omega_embed[i]` and `F_cur` unchanged.

**Exactness.** `free_energy_value` evaluates `F` exactly at the given beliefs; the move holds the beliefs `q` fixed and flips only the frame, so `Delta_F` is the exact change in the joint `F(q, U)` under the proposed block move (a Metropolis-within-Gibbs step on the joint objective). Beliefs are NOT re-converged per proposal — that would be the marginal `Delta_F` and is not what this move targets. `F_cur` is computed once per step and carried across the sweep, so the cost is one no_grad forward plus one `free_energy_value` trial per unique batch token per step (the "exact, `O(unique_batch)` re-evals" option selected during design).

**Temperature.** `T = omega_metropolis_temperature` (default `1.0`), fixed (no annealing schedule — deferred as YAGNI). `F` is in nats; `T = 1` treats `F` directly as the Metropolis energy.

## 3. Config surface

- `_VALID_OMEGA_REFLECTION = ("off", "init_seed", "metropolis")`. (`"ste"` is NOT added here — it raises via an explicit guard, see §5.)
- `omega_reflection: "metropolis"` — new mode.
- `omega_metropolis_temperature: float = 1.0` — the `T` in the acceptance rule. Must be `> 0` (validated).
- `omega_metropolis_every: int = 1` — cadence in optimizer steps (`1` = every step; `>1` runs the sweep every Nth step to amortize the exact re-eval). Must be `>= 1`.
- **Per-group validation** (reuse the existing `init_seed` cross-check gate): `"metropolis"` requires `gauge_parameterization == "omega_direct"` and is accepted for **exactly the same `_REFLECT_OK = ("glk", "block_glk", "so_k")` set that gates `init_seed`** — the groups where `reflection_element` yields a valid `det < 0` element (`glk`/`block_glk` reach full `det < 0`; `so_k` reaches `O(K) \ SO(K)`). Because `metropolis` reuses the same `reflection_element` machinery, its eligibility is identical to `init_seed`'s by construction. It is **rejected**, fail-loud, for every other group, for two distinct reasons the error message should distinguish: `sp` / `sp_n` are connected with `det ≡ +1`, so the flip is **vacuous**; `so_n` / `tied_block_glk` need a group-specific reflection seed (`rho(O(N))` image / tied replicated element) that is **deferred infrastructure**, exactly as `init_seed` defers them today. This is a deliberate scope refinement from the design sketch (which mentioned `so_n`): implementing the `so_n` `rho(O(N))` reflection is out of scope for this feature and would be a separate extension shared with `init_seed`.
- Default `omega_reflection = "off"` is unchanged: the move is never entered, no new parameter or RNG draw occurs, and the pure `det > 0` path plus the whole `state_dict` stay **byte-identical**.

## 4. Where it lives

- A model method `metropolis_omega_step(token_ids, *, generator) -> dict` on `VFEModel` (returns a small stats dict — proposed/accepted counts, mean `Delta_F` — for logging). It is a **no-op** (early return) unless `cfg.omega_reflection == "metropolis"`. It owns the no_grad forward, the `free_energy_value` re-eval, the masked-flip trial-belief construction, the seeded acceptance, and the in-place `omega_embed` mutation. It reuses the existing `reflection_element` helper and the same block-layout logic `prior_bank`'s `init_seed` path uses, so the flip is correct for full `(V,K,K)` and compact `(V,H,d,d)` / `(V,d,d)` storage.
- **Single training-loop seam:** `vfe3/train.py::train_step`, immediately after `optimizer.step()`, a guarded + cadence-checked call: `if cfg.omega_reflection == "metropolis" and step % cfg.omega_metropolis_every == 0: model.metropolis_omega_step(token_ids, generator=...)`. Both `train_vfe3.py` and `ablation.py` route through `vfe3/train.py`, so this is the only integration point. The stats dict is threaded into the existing per-step logging where convenient (optional, non-load-bearing).

## 5. STE deferral marker

`omega_reflection = "ste"` is rejected at config with a clear `NotImplementedError`/`ValueError` ("`omega_reflection='ste'` (straight-through det-sign) is not implemented; use `'metropolis'` for the learnable det-sign or `'init_seed'` for a fixed one"). At the move site in the model, a comment records the alternative:

```
# TODO(STE): straight-through-gradient variant of the learnable det-sign -- propose per-token
# sign flips accepted through a straight-through estimator (biased but differentiable) instead
# of this DeltaF-gated Metropolis accept/reject. See GL(K)_attention.tex eq:ok_transport.
```

## 6. Determinism

The acceptance draws use a `torch.Generator` seeded deterministically from `cfg.seed` (constructed once, owned by the model or threaded from the train loop), so the accept/reject sequence is reproducible on CPU for a fixed seed and data order — consistent with the project's `_seed_everything` contract. The `free_energy_value` re-eval inherits the same GPU-kernel nondeterminism caveat as the rest of the forward (documented, not addressed here).

## 7. Pure-path preservation

The default (`omega_reflection = "off"`) and the shipped `"init_seed"` path are untouched: no new parameter is created, the move method returns immediately, no RNG is drawn, and the `state_dict` is byte-identical. The theoretically-pure `det > 0` field remains the default. `"metropolis"` is a documented opt-in.

## 8. Testing (TDD, CPU-bound, K < 6)

1. **off is byte-identical:** with `omega_reflection="off"`, `metropolis_omega_step` returns without touching `omega_embed`, drawing RNG, or changing `state_dict`.
2. **downhill flip always accepted:** construct a tiny model where flipping a token's det-sign lowers `F` (e.g. seed the token into the wrong sheet); one sweep flips it; `det(omega_embed[i])` changes sign and `F` decreases.
3. **uphill flip gated:** when the flip raises `F`, acceptance matches `draw < exp(-Delta_F/T)` for a known seeded `draw` (assert both an accepted and a rejected case by choosing `T`/seed).
4. **exact ΔF:** the `Delta_F` the move computes equals a full independent `free_energy_value` recompute with `omega_embed[i]` actually flipped (pins that the masked trial-belief flip matches the source-table flip).
5. **seeded reproducibility:** two runs with the same seed and inputs produce the identical accept/reject sequence and identical resulting `omega_embed`.
6. **det-sign actually flips on accept:** `sign(det(omega_embed[i]))` toggles `+ <-> -` exactly on accepted proposals, for `glk` and (compact) `block_glk` / `so_k`.
7. **per-group gating:** `omega_reflection="metropolis"` raises at config for `sp` / `sp_n` (vacuous) and for `so_n` / `tied_block_glk` (deferred reflection seed); constructs for `glk` / `block_glk` / `so_k` (the `_REFLECT_OK` set); `"ste"` raises `NotImplementedError`.
8. **train seam no-op when off / cadence honored:** `train_step` calls the move only under `"metropolis"` and only every `omega_metropolis_every` steps.

## 9. Implementation inventory

1. `vfe3/config.py`: add `"metropolis"` to `_VALID_OMEGA_REFLECTION`; add `omega_metropolis_temperature` / `omega_metropolis_every` fields with range validation; extend the per-group `omega_reflection` cross-check to accept metropolis for the eligible groups and reject `sp`/`sp_n` (and `"ste"` with a NotImplementedError).
2. `vfe3/model/model.py`: add `metropolis_omega_step(token_ids, *, generator)` (no-op unless the mode is on); the `# TODO(STE)` marker; reuse `reflection_element` and the block-layout flip.
3. `vfe3/train.py`: the single guarded + cadence-checked call after `optimizer.step()` in `train_step`; construct/thread the seeded generator.
4. `tests/test_omega_direct.py` (or a focused new test module): the eight tests in §8.

No change to the pure path, the belief E-step, the continuous retraction, or any shipped `omega_reflection` mode.
