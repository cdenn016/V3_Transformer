# Design: omega_direct Phase 2 — other groups + compact block storage

Date: 2026-07-08
Status: design, autonomous continuation of the Phase-1 build (shipped to main at 7c82583)
Branch: `feat/omega-direct-phase2`
Predecessor: `docs/superpowers/specs/2026-07-07-omega-direct-gauge-parameterization-design.md`

## One-line summary

Extend `gauge_parameterization="omega_direct"` from `glk`/`block_glk` to the remaining structure groups
(`tied_block_glk`, `sp`, `sp_n`, `so_k`, `so_n`), add an exact transpose inverse for the compact (orthogonal)
groups, add fail-closed per-group reflection-seed cross-checks, and add opt-in compact block-diagonal storage
that removes the Phase-1 `(V,K,K)` memory waste on `block_glk`. The pure `phi`/exp default and the shipped
`omega_direct` `glk`/`block_glk` path both stay byte-identical. The learnable det-sign (STE) is designed here
but deferred pending user confirmation.

## 1. Why the groups differ (the load-bearing distinction)

`omega_direct` gives the frame two capabilities `phi` lacks:

1. The **non-exp interior**: `phi` re-forms `U = exp(phi)` every forward, so it is confined to `image(exp)`
   forever; `omega_direct` stores `U` and steps it by `U ← U·exp(step)` (`lie_ops.py` `retract_omega`),
   accumulating *products* of exponentials that generate the whole connected identity component. This is a
   real gain exactly when `exp` is NOT surjective onto the identity component — true for the non-compact
   groups (`glk`, `block_glk`, `tied_block_glk` over `GL⁺`; `sp`, `sp_n` over `Sp(2m,R)`, whose exp is
   famously non-surjective) and FALSE for `so_k`/`so_n` (`exp: so(N)→SO(N)` is surjective, compact
   connected). This capability is exercised THIS phase by the ordinary M-step optimizer — no STE.
2. The **det<0 disconnected-component reach**: `det exp = e^{tr} > 0` always, so only `omega_direct` can be
   *seeded* into another component. But the retraction is det-sign-preserving and no gradient crosses the
   `det=0` free-energy barrier, so without the deferred learnable mechanism this reach is FROZEN at init.

Sorting the five groups:

- `sp`, `sp_n`: **REAL** value this phase — the non-exp interior of `Sp(2m,R)`, live via the optimizer. No
  det<0 component at all (`det ≡ +1`, `Sp` connected), so reflections are vacuous.
- `tied_block_glk`: **REAL** — the non-exp interior of the shared `GL(d)` block, live; small engineering
  delta over the shipped `block_glk`.
- `so_k`, `so_n`: **MARGINAL** this phase — no non-exp interior (SO is exp-surjective, so `phi` already
  covers `SO` fully and *without* drift), and their only distinctive reach (det<0 into `O(K)` / `ρ(O(N))`)
  is frozen until the STE. Their live deliverables are the exact-transpose inverse (a correctness/speed win),
  orthogonality-drift control, a cleaner gauge-invariance test, and STE-ready scaffolding. Included for
  completeness and honestly labeled, NOT sold as a live det<0 feature.

## 2. Per-group extension

For every group the frame is a per-token element `U_i` in the structure group, transport is the same cocycle
`Ω_ij = U_i U_j⁻¹`, and the optimizer's natural-gradient projection onto the group's own generator span
(`gauge_optim.py` omega branch) is already group-agnostic — it carries any group onto its own manifold with
no change. The differences are the inverse, the reflection, and (for SO) drift.

- **Inverse.** `skew_symmetric=True` groups (`so_k`, `so_n`) have `U⁻¹ = Uᵀ` exactly and for free —
  `build_transport_from_element` must branch on `group.skew_symmetric` and use the transpose instead of the
  fp64 `inv`. This is exact (`exp_neg_phi = exp_phiᵀ` makes `Ω` exactly orthogonal and the congruence an
  exact isometry) and faster (no fp64 LU, no autocast island). The `skew_symmetric=False` groups
  (`glk`, `block_glk`, `tied_block_glk`, `sp`, `sp_n`) keep the exact fp64 `inv` — so the shipped
  `glk`/`block_glk` path is byte-identical.
- **Reflection seed** (`omega_reflection="init_seed"`), fail-closed per group:
  - `so_k`: accept — the shipped `reflection_element(K) = diag(-1,1,…)` is a valid `O(K)` element with
    `det=-1`, and `exp(so(K))` cannot leave `SO(K)`, so init-seeding is the only route to the `O(K)\SO(K)`
    sheet. The one new group where the shipped seed is group-correct as-is.
  - `sp`, `sp_n`: **reject** — no det<0 component; `diag(-1,1,…)` is not even a symplectic element.
  - `so_n`: **reject this phase** — the reachable det<0 elements are `ρ(O(N)\SO(N))`, a proper irrep-image
    subset (and det-sign is irrep-dependent); the ambient `diag(-1,…)` is generically not in the structure
    group and would void the divergence-invariance guarantee. The `ρ`-image seed is deferred.
  - `tied_block_glk`: **reject this phase** — the ambient seed puts `-1` in head 0 only, breaking the tie
    (unequal blocks). A correct tied seed replicates one det<0 `GL(d)` block into every head; deferred with
    the STE.
- **Retraction / drift.** `retract_omega` keeps `U` on the identity component of each group. For the
  orthogonal groups (`so_k`, `so_n`) fp32 accumulation of `exp(skew)` products walks `U` off `O(K)` over
  many M-steps (the phi path avoids this by re-forming `exp(phi)` fresh). Add periodic re-orthogonalization
  (a polar/QR projection of the `omega_embed` rows, every `omega_reorth_every` M-steps, default a modest
  cadence) so `Uᵀ` stays the exact inverse and the congruence stays an isometry. Non-compact groups need no
  such control (the F-barrier plus fp64 island is the same footprint as the shipped `glk`).

## 3. Compact block storage (opt-in)

Phase 1 stores the frame as a full `(V,K,K)` table even for `block_glk`, wasting `~H×` (off-blocks frozen at
zero). Compact storage, behind a new default-OFF flag `omega_compact_storage: bool = False`:

- **Which groups.** Untied equal-block `block_glk` → `(V,H,d,d)`. Tied `tied_block_glk` → `(V,d,d)`. Both
  match `phi_embed = (V, n_gen)` param count exactly (`V·H·d² = V·n_gen`; tied `V·d² = V·n_gen`). Single-block
  groups (`glk`, `so_k`, `sp`, `irrep_dims=[K]`) have nothing to compact — keep `(V,K,K)`. The irrep towers
  (`so_n`, `sp_n`) keep `(V,K,K)` this phase (the faithful compact object is the defining-rep element run
  through the irrep maps — an element-vs-coordinate tension deferred).
- **Assembly at encode.** Encode looks up the compact `(B,N,H,d,d)` and assembles the transport-ready
  block-diagonal `(B,N,K,K)` `belief.omega` via the existing differentiable scatter
  (`lie_ops.py` `_from_equal_diag_blocks`); off-blocks stay exactly zero, and autograd routes the dense-Ω
  gradient back to the compact table with no manual adjoint. For `tied_block_glk` the single `g` broadcasts
  into all `H` blocks; the broadcast adjoint sums the `H` per-block gradients onto `g` — exactly the tied
  update. **The belief still carries the assembled `(B,N,K,K)`**, so the entire downstream transport stack
  is untouched; only the `PriorBank` table+encode and the optimizer change.
- **Optimizer.** With the compact table, `p.grad` is already `(V,H,d,d)`. Reshape to `(A·H,d,d)`, run the
  existing `extract_phi`/`retract_omega` against a single-block `gl(d)` basis, reshape back — `H` small
  `d×d` solves/exps instead of one `K×K`. `PriorBank.__init__` must retain `irrep_dims` (currently dropped)
  so encode knows `(H,d)`.
- **State-dict safety.** Compaction changes the table shape, which would break a Phase-1 `omega_direct`
  checkpoint. The opt-in default-OFF flag keeps the shipped `(V,K,K)` path byte-identical.

## 4. STE / learnable det-sign — DESIGN ONLY, deferred pending user confirmation

The manuscript names a straight-through estimator (`GL(K)_attention.tex:1157`), but the more principled
realization for this no-NN codebase is a **ΔF-gated flip**: propose `U_v → R U_v` on a batch-active vocab
row, evaluate `ΔF` via the already-omega_direct-aware `free_energy_value(…, keys=None)`, accept iff `ΔF<0`
(greedy) or with `min(1, exp(-ΔF/T))` (Metropolis). It is discrete coordinate descent (ICM) on `π₀(G)^V`,
reads the exact free energy, adds **no new parameter, no backprop, no fabricated surrogate gradient**, and
composes with the existing `no_grad` `R`-multiply seed machinery. It runs in a new M-step-adjacent `no_grad`
pass after `optimizer.step()`, never in the E-step or autograd (no freeze footgun). Cost budget: `O(N)`
per proposed flip via an incremental row+column ΔF (naive full rescore is `O(N³)`/sweep), `O(N²)`/sweep,
amortized by a cadence. Per-group: no-op for `sp`/`sp_n`; `so_*` reach only `ρ(O(N))`.

**Two-tier open question for the user (build nothing until answered):**
1. *Empirical gate*: does a per-token det<0 reflection improve free energy / metrics at all? The `init_seed`
   ablation this phase ships (fixed hand-placed reflection on `glk`/`block_glk`/`so_k`) is exactly the test —
   run it before committing to any learnable mechanism.
2. *Only if positive, the mechanism*: the manuscript's straight-through estimator (new zero-init logit,
   backprop through a fabricated gradient, manuscript fidelity) vs the ΔF-gated flip (no new parameter, VFE
   charter). Recommendation: the ΔF-gate, keeping `"ste"` as a reserved `omega_reflection` value for
   manuscript fidelity.

## 5. Scope

**This phase (6 tasks, all default-OFF-safe):** skew transpose inverse; config gate widening + fail-closed
per-group reflection cross-check; full-table extension for `sp`/`sp_n`/`tied_block_glk`; full-table extension
for `so_k`/`so_n` with drift control; opt-in compact storage for `block_glk`/`tied_block_glk`; ablation
fan-out. See the implementation plan.

**Deferred:** the STE / ΔF learnable det-sign (pending §4 confirmation); tower compaction for `so_n`/`sp_n`;
the `ρ(O(N))`-image seed for `so_n`; tied reflection seed for `tied_block_glk`; and (still, from Phase 1) the
gamma/s-channel omega-fidelity, `e_phi_lr>0` E-step frame refinement, and `viz/extract` diagnostics.

## 6. Risks

- **SP conditioning**: `sp`/`sp_n` are non-compact and pay the fp64 `inv`, inheriting exactly `glk`'s
  (already-accepted) conditioning footprint. `so_k`/`so_n` are orthogonal (`cond=1`), so the inverse is fine
  and the transpose is safe.
- **Orthogonality drift** (the one genuinely new hazard): fp32 `exp(skew)` products walk `U` off `O(K)`;
  mitigated by periodic re-orthogonalization (§2).
- **Compact-storage correctness**: the scatter must be differentiable and reproduce the full element with
  off-blocks held at zero (`allclose(assemble(compact), full)`); the per-block retraction must equal the
  full-table step restricted to blocks (natural-gradient-step-equivalence test); the tied broadcast adjoint
  must *sum* the `H` per-block gradients onto `g`.
- **Phase-1 gates preserved**: `transport_mode='flat'`, `e_phi_lr==0`, and the gamma/s-channel rejection stay
  for every new group (the s-channel still transports by the phi cocycle). The new per-group reflection
  cross-check runs after the static-tuple validation and inside the `omega_direct` whitelist. State whether a
  cross-coupled `block_glk` (collapses to `[K]`) is omega_direct-eligible (it behaves glk-like: single-block
  dense, det<0 reachable, no compaction).
