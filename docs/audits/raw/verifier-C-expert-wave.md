# Verifier C — expert-wave medium/low findings + negative-result re-derivation

Returned 2026-07-27 ~10:12 CDT. Independent `general-purpose` verifier. Temp probes cleaned up;
no repository file modified.

## Verdict table

| # | Source | Finding (short) | Verdict | Reachable? | Evidence |
|---|---|---|---|---|---|
| NA-4 | numerical-analyst | Graph-connected trust-region norms -> NaN double backward, 4 retractions | **CONFIRMED** | unreachable; **and no double-backward consumer exists anywhere** | `lie_ops.py:711-712,728-729`; `retraction.py:519-520,632-633` |
| NA-5 | numerical-analyst | Laplace KL hardcoded fp32, negative divergence | **CONFIRMED** | unreachable (live `gaussian_diagonal`) | `laplace.py:237-243` |
| NA-6 | numerical-analyst | `bohning_emission_terms` has no fp32 island under live AMP | **CONFIRMED** | unreachable (`emission_mode='off'`) | `emission.py:110-140`; `model.py:1126` in `with run, amp:` at `:1046` |
| NA-7 | numerical-analyst | `condition_number`/`floor_eigenvalues` downcast fp64->fp32 | **CONFIRMED** | **unreachable — NO production caller at all** | `numerics.py:299,259` vs idiom `:230` |
| NA-8 | numerical-analyst | `safe_spd_inverse` runs `pinv` over the whole batch | **CONFIRMED** (2 sub-claims weakened) | unreachable — no production caller | `numerics.py:248-249` |
| NA-9 | numerical-analyst | Bohning curvature cancellation + "unreachable eps floor" | **PARTIAL — cancellation real; eps-floor sub-claim REFUTED** | unreachable | `emission.py:82-84` |
| V-2 | variational | Bohning anchored pre-stack, fused against each layer's `mu_p` | **CONFIRMED** | doubly unreachable: `emission_mode='off'` AND `n_layers=1` | `stack.py:142` vs `:151`; `kernels.py:701` |
| V-3 | variational | Metropolis dF adds summed belief F to mean model F | **CONFIRMED** | unreachable (`phi_reflection`/`omega_reflection` = "off") | `model.py:1308`, `:2107-2108`; `free_energy.py:531-537` |
| V-4 | variational | two-hop gated `> 0.0` vs `!= 0.0` on 4 descent paths | **CONFIRMED** | unreachable (`config.py:2900-2901` rejects <0) | `e_step.py:665` |
| GT-2 | gauge-theorist | `per_head_gauge_invariants` publishes an SVD ratio as gauge invariant | **CONFIRMED** | **REACHABLE AND LIVE** | `metrics.py:954-955`, consumed `model.py:2996,3022,3052` |
| GT-3 | gauge-theorist | Transport dispatch discards `transport_mode` when `right_phi` present | **CONFIRMED** | unreachable (live `flat` — nothing to discard) | `e_step.py:131-144` before `:145 get_transport`; `:373-381` |
| TML-3 | transformer-ml | `collect_beta_channel_decomposition` ablates gamma too | **CONFIRMED** | **REACHABLE AND LIVE** | `run_artifacts.py:3583-3595`; `model.py:2377,2383` |
| TML-4 | transformer-ml | `attn_entropy_min`/`collapsed_heads` structurally constant | **CONFIRMED** | **REACHABLE AND LIVE** | `model.py:3145-3148` |
| TML-5 | transformer-ml | `pos_phi='frozen'` leaves H-1 heads position-blind | **CONFIRMED** | unreachable (live `pos_phi="learned"`) | `positional_phi.py:98-99`; `generators.py:106-113` |
| TML-6 | transformer-ml | `viz/extract.py` recomputes beta at base tau | **CONFIRMED** | unreachable (`query_adaptive_tau=False`) | `extract.py:992-995,1097-1100` |
| TML-7 | transformer-ml | Replays omit the emission term | **CONFIRMED** | unreachable (`emission_mode='off'`) | `model.py:2740-2751,3290-3304`; `extract.py:526-534` |
| GEO-1 | geometer | SPD bounds applied to raw dispersion slot, family-blind | **CONFIRMED** | unreachable (live Gaussian: dispersion IS covariance) | `e_step.py:1147-1150`; `retraction.py:461-465`; `laplace.py:65,89-102` |
| GEO-2 | geometer | `estep_residuals` hardwires AIRM | **CONFIRMED** (magnitude smaller) | unreachable (live `spd_affine`+diagonal => exact) | `metrics.py:1469`; `extract.py:920-924` |
| GEO-3 | geometer | `killing`/`killing_per_block` are scalars | **CONFIRMED** | unreachable (live `pullback_per_block`, `e_phi_lr=0`) | `phi_preconditioner.py:809-821,940-953,980-1012` |
| GEO-4 | geometer | Prior handoff uses a Euclidean lerp | **CONFIRMED** (self-declared "not an error") | inert at live `n_layers=1` | `stack.py:152`; `extract.py:912` |
| INFO-3 | info-geometer | Gaussian effective-rank floor breaks scale invariance | **CONFIRMED** (exact table reproduced) | **REACHABLE AND LIVE** | `gaussian.py:130-133`; `metrics.py:37,502-523` |
| INFO-4 | info-geometer | Generic Renyi-from-`A` has no natural-parameter gate | **CONFIRMED** (dormancy confirmed) | unreachable — **all 5** families supply `renyi_closed_form` | `base.py:568-571`; `gaussian.py:652-659` |
| INFO-5 | info-geometer | `renyi_order` discarded by 3 of 4 functionals | **CONFIRMED** | unreachable at live `renyi`; live under any other functional | `base.py:615-687`; `config.py:2956-3032` |
| INFO-6 | info-geometer | `alpha>1` warning states the Gaussian condition for all families | **CONFIRMED** (fires verbatim for Laplace) | unreachable (live `renyi_order=1.0`) | `base.py:30-38`, dispatched `:594-596` |
| IE-3 | impl-engineer | Under `mm_exact` every E-step lr/precond/trust knob unread; detector silent | **CONFIRMED** | **REACHABLE AND LIVE — "the most consequential item on my list"** | `e_step.py:1001-1058` vs `:1097,1121,1125,1130,1136,1148-1149`; `config.py:2957` |
| IE-4 | impl-engineer | `exp_fp64_*` never reach the non-flat vertex exp | **CONFIRMED** | unreachable (live `flat`, which DOES forward them) | `transport.py:903-913,1075,1088` vs flat `:808-812`; `e_step.py:938-955` |
| IE-5 | impl-engineer | Purity ledger omits emission and the norm seam | **CONFIRMED** (reproduced `on_pure_path=True`) | structurally live; live run already False for another reason | `run_artifacts.py:4116-4134,4152-4211` (0 hits) |
| IE-6 | impl-engineer | `across_layer_belief_trace` replays at init temperature | **CONFIRMED** | unreachable (`learnable_kappa_beta=False`) | `extract.py:902-908` vs `:531,668,1058` |
| IE-7 | impl-engineer | `randomize_e_steps` double-draws for the model channel | **CONFIRMED** | unreachable (`randomize_e_steps=False`) | `model.py:870,926-929`; `e_step.py:1433-1436` |

## Refutations / weakenings the verifier made

- **NA-9 eps-floor sub-claim REFUTED.** For a genuinely constant column the verifier measured
  **exactly `1.000000e-12`** at every constant tested (`c in {0.1, 0.7, 1.0, 3.0, 10.0}`) — the clamp
  DOES bind. The investigator's `~5e-3` residual estimate assumes naive sequential summation;
  torch's blocked `.sum()` is far more accurate. The cancellation half is real
  (`8.9e-08 -> 1.5e-05 -> 2.7e-04` as column mean goes `0 -> 0.2 -> 1.0` at V=50257).
- **NA-7 / NA-8 have NO production caller.** `condition_number` is reached only from a
  `@register_monitor` whose `run_monitors` has no caller; `metrics.belief_spectrum` computes its own
  condition inline at `metrics.py:562-572`. Severity -> dead-utility.
- **NA-8 sub-claims inaccurate:** `safe_cholesky` at `:207-209` ALSO evaluates its retry on the whole
  batch (safe only because `cholesky_ex` never raises), and the "never raises" docstring attaches to
  `cholesky_ex`, not to `safe_spd_inverse`.
- **NA-4:** a global search for `create_graph=True` finds it only at `gradients/oracle.py:230-232`
  and `kernels.py:389,420` — **none takes a second derivative through a retraction**. Latent hazard
  on public primitives only.
- **GEO-2 magnitude smaller than claimed:** worst `|AIRM - LE|/LE` over 200 draws was
  **19.9-23.1%** across tangent scales, versus the investigator's 33.3%. Both controls check out
  exactly.
- **IE-4 nuance:** the docstring at `e_step.py:351-354` describes the builder behavior as
  deliberate; the `_omega_builder` omission is the sharper half of the finding.
- **GEO-1 correction:** the family hooks have more consumers than the finding listed
  (`metrics.py:488,567,570,648`, `model.py:88-92`, `numerics.py:165`, `extract.py:1322`,
  `figures.py:3028,3031,3086`) — but none is the retraction, which is the load-bearing part.

## NEGATIVE-RESULT RE-DERIVATION — ten items, ALL TEN UPHOLD

No claimed-clean item turned out defective.

1. **`retract_spd_full` is the exact affine-invariant exp map.** Independently built
   `S^{1/2} expm(S^{-1/2} X S^{-1/2}) S^{1/2}` at K=4 float64: relative residual **2.192e-13**
   (investigator: 3.78e-16; different draw/conditioning, both unambiguously round-off). **Upholds.**
2. **Congruence equivariance on the pure path.** With `sigma_max=None`:
   `||R(gSg^T, gXg^T) - gR(S,X)g^T||/||.|| = **2.685e-15**`. With `sigma_max=10.0` it deviates by
   **0.322** (investigator: 0.756), confirming the break comes only from the documented spectral cap.
   **Upholds.**
3. **Diagonal `spd_affine` == `log_euclidean` bit-for-bit:** measured **0.0**. **Upholds.**
4. **`spd_geodesic_distance` IS the AIRM distance:** `0.113910` vs independently computed
   `||Sigma^{-1/2} X Sigma^{-1/2}||_F = 0.113910`. **Upholds.**
5. **`alpha -> 1` recovers `KL(q||p)`, NOT the reverse.** At `alpha=1.0`:
   `|renyi(q,p) - KL(q||p)| = **0.000e+00**`, `|renyi(q,p) - KL(p||q)| = **9.427e-01**` (the two KLs
   are 3.2182 and 2.2755). From below, `alpha=1-1e-6` gives residual `4.79e-06`. Direction
   unambiguously forward. **Upholds.**
6. **Beta IS a genuine stationary point of the implemented F.** Verifier rebuilt the row Lagrangian
   and substituted `beta*_j = pi_j exp(-E_j/tau)/Z`: residuals **`[0.0, -2.22e-16, 0.0]`**,
   `sum beta* - 1 = 0` symbolically, and the envelope identity `block(beta*) + lam*tau*log Z`
   simplifies to **exactly 0**. With the entropy removed the derivative is `E_j*lam + nu` for every
   `j` — b-independent, hence no interior solution. Every element **upholds**.
7. **Entropy-suppressed surrogate is fenced off from the envelope kernel.** `uses_kernel_route`
   (`kernels.py:304`) includes `and include_attention_entropy`, so `False` cannot reach the
   closed-form kernel; `free_energy.py:448` differentiates `(beta * energy).sum()` with beta live.
   **Upholds.**
8. **`sp(2m,R)` generators satisfy `JA + A^T J = 0`.** At K = 2, 4, 6:
   `||JA + A^T J||_inf = **0.000e+00 exactly**` for every generator; generator count and rank both
   `m(2m+1)` (3, 10, 21); bracket-closure residual **0.000e+00**. CLEANER than the investigator's
   claimed 6.7e-16. **Upholds.**
9. **`log_prior` is added BEFORE the softmax.** `free_energy.py:329-332`: `logits = -energy/tau`;
   `logits = logits + log_prior`; `return torch.softmax(logits, dim=-1)`. **Upholds.**
10. **Max attention mass strictly above the diagonal is exactly 0.0.** `causal`, `causal_noself`,
    `causal_alibi_noself` each give **exactly `0.000000e+00`** above the diagonal with row sums
    `1.000000`; the `uniform` control gives `2.501900e-01`, proving the probe is sensitive. No
    future-mass leak. **Upholds.**
