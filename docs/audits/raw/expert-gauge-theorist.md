# Expert — audit-gauge-theorist (equivariance, holonomy, cocycle, irreps)

Returned 2026-07-27 ~08:55 CDT. Verbatim findings; NOT yet verified.
Bands: **no critical**, 1 high, 1 medium, 1 low.

## Verified clean under this lens (executed or read in full)

The congruence sandwich `Omega Sigma Omega^T` in every dense/factored/compact/direct-link route
(transport.py:2188-2395, 2481-2559, 2584-2635, 2668-2808 — index patterns check out as
`Omega^h Sigma^{(h,g)} Omega^{g T}`); the `Omega_ij = U_i U_j^{-1}` cocycle construction including
the exact right positional factor and the RoPE `right` fold `V_i = U_i R_i^T`; the exact-congruence
pullback `Omega_ij^{-1} = Omega_ji` via pair transpose and its `ell_i - ell_j` determinant;
`sp(2m,R)` generators (`JA + A^T J = 0` to 0.0, full rank `m(2m+1)`, bracket residual 6.7e-16 at
K=2,4,6); `so_n`/`sp_n` irrep towers (exact skewness 2.2e-16 for `so`, correctly declared non-skew
for `sp`, build-time bracket-homomorphism assert); CG intertwiner equivariance and the isotypic
mixer commutant; per-irrep-block energy/`tau`/`beta` decomposition under `block_glk`'s untied gauge;
`holonomy_wilson_sampled` (`Re Tr(H)` — correctly invariant) versus the honestly-labeled
frame-dependent `holonomy_deviation`.

---

### 1. E-step feeds the chart natural gradient into a group-product retraction without the `Psi_L(ad_phi)` trivialization the M-step applies
**Location:** vfe3/inference/e_step.py:1195-1205 (with vfe3/geometry/phi_preconditioner.py:1215-1219)
**Severity:** high
**Evidence:** `_precond_pullback_per_block` returns only the chart-metric solve
```python
G_metric = pullback_metric_per_block(phi, generators, irrep_dims, ...)
sol = torch.linalg.solve(G_metric + eps * eye, grad_phi.double().unsqueeze(-1))
return sol.squeeze(-1).to(orig_dtype)          # v_phi = G(phi)^-1 grad, a CHART tangent
```
and the E-step hands that straight to a retraction that composes it as a right group factor
(`retract_phi -> _retract_core -> compose_bch(phi, update)` = `log(exp(phi) exp(update))`,
lie_ops.py:713-726):
```python
grad_phi = precondition_phi_gradient(grad_phi, belief.phi, group.generators,
                                     mode=phi_precond_mode, irrep_dims=group.irrep_dims)
phi = retract_phi(belief.phi, -grad_phi, group, step_size=e_phi_lr, mode=phi_retract_mode, ...)
```
The strict M-step sibling does the conversion explicitly — phi_preconditioner.py:556,581 computes
`v_phi = cholesky_solve(...)` then `xi = einsum("...ab,...b->...a", psi_left, v_phi)`, and
gauge_optim.py:263,280-287 composes `-lr * direction.xi` through `compose_bch`. Both conventions
exist in the tree; only the M-step is consistent. Executed at `block_glk` K=4 (`gl(2)^2`), 200
draws, `||phi|| ~ 2.2` (retraction cap 5.0), float64:
```
median relative direction mismatch ||Psi_L^-1 v - v||/||v|| = 0.313
max mismatch = 3.654
ASCENT steps (<grad, realized_step> <= 0): 1/200   most negative = -884.9
BCH(phi, xi) chart move vs intended v : rel err 4.758e-03   <-- M-step convention
BCH(phi, v ) chart move vs intended v : rel err 7.797e-01   <-- E-step convention
```
Since `log(exp(X) exp(tY)) = X + t*dexp^{-1}_X(Y) + O(t^2)` with
`dexp^{-1}_X = ad_X/(1-e^{-ad_X}) = Psi_L(ad_X)^{-1}` (Nakahara 2003 section 5.6; Hall, *Lie
Groups*, Thm 5.4), BCH-composing `v_phi` realizes a chart step `Psi_L(ad_phi)^{-1} v_phi`, and
`Psi_L^{-1} G^{-1}` is not symmetric, so descent is not guaranteed away from the identity. The
pairing `phi_precond_mode='pullback_per_block'` + `phi_retract_mode='bch'` is exactly what the live
`train_vfe3.py` sets (inert only because `e_phi_lr = 0.00`).
**Fix:** Have the pullback preconditioners return `xi = Psi_L(ad_phi) @ (G(phi)^{-1} grad)`
(reusing `_adaptive_phi_differentials`) whenever the retraction mode is `'bch'`, or restrict the
pullback modes to `phi_retract_mode='euclidean'` at config validation.

### 2. `per_head_gauge_invariants` returns a singular-value ratio as a gauge invariant, and the run report consumes it as one
**Location:** vfe3/metrics.py:935-960 (consumed at vfe3/model/model.py:2996, 3022, 3052)
**Severity:** medium
**Evidence:** the function named `per_head_gauge_invariants` returns
```python
s = torch.linalg.svdvals(blk)
anisos.append(s[..., 0] / s[..., -1].clamp(min=eps))
...
return {"logdet": ..., "anisotropy": torch.stack(anisos, dim=-1)}
```
and model.py:3052 publishes `d["gauge_head_aniso_mean"] =
float(_ghi["anisotropy"].float().mean())`. Singular values are invariant only under ORTHOGONAL
conjugation; the sibling `group_gauge_invariant` (metrics.py:924-931) already replaced this exact
quantity for `sp` with the eigenvalue-modulus squeeze for that reason, and states the action is
GL(K) conjugation `g exp(phi) g^{-1}`. Executed at `block_glk` K=4, `g = exp(sum_a c_a G_a)` drawn
from the group's own generators, float64:
```
--- conjugation g U g^-1
  anisotropy base  [1.6659, 1.6851, 1.6481, 2.3367, 1.8652, 1.6285]
  anisotropy moved [1.5275, 2.6471, 3.4546, 4.1151, 1.4351, 3.9815]
  max |d logdet|    = 3.33e-16
  max rel |d aniso| = 1.4449
eig-modulus squeeze base/conj: 0.6980112265576064 0.6980112265576063
--- left action g U (the action metrics.gauge_equivariance_residual applies to the compact vertex)
  max |d logdet|    = 1.0547
  max rel |d aniso| = 2.0006
```
The `logdet` entry is invariant to 3.3e-16 under conjugation; the `anisotropy` entry moves by 144%,
and 200% under the left vertex action. Under either reading of the gauge action the reported
"per-head gauge invariant" is frame-dependent (Nakahara 2003 section 10.5: only conjugacy-class
functions of the holonomy/frame — trace, determinant, eigenvalue spectrum — are gauge invariant).
**Fix:** Rename the `anisotropy` entry to a frame-dependent conditioning probe (it duplicates
`vertex_cond`), or replace it with the conjugation-invariant eigenvalue-modulus squeeze
`max_k log|lambda_k| - min_k log|lambda_k|` already used by `group_gauge_invariant` for `sp`.

### 3. Transport dispatch silently discards the selected connection regime whenever a right positional factor is present
**Location:** vfe3/inference/e_step.py:131-144 and vfe3/inference/e_step.py:373-381
**Severity:** low
**Evidence:** both dispatchers branch on `right_phi` **before** consulting the registry, so
`transport_mode` is never read:
```python
    if right_phi is not None:
        built = build_factored_transport(
            phi, group, gauge_mode=gauge_mode, ..., right_phi=right_phi,
        )
        ...
        return built.to_dense_omega() if materialize else built
    build = get_transport(transport_mode)
```
```python
    elif right_phi is not None or _can_fuse_flat(transport_mode, group):
        built = build_factored_transport(phi, group, ..., right_phi=right_phi, ...)
```
`build_factored_transport` returns a container with `same_frame_flat_cocycle=True`
(transport.py:2061-2063, 2095-2096), so with `transport_mode='regime_ii'`/`'regime_ii_covariant'`/
`'regime_ii_link*'` and a non-`None` `right_phi` the learned `connection_W`/`connection_M`/
`connection_L` never enter the operator, holonomy is identically identity, and the connection
parameter receives no gradient — while the run record still names the non-flat mode. The only thing
preventing this is a check three modules away (config.py:967-977, which requires
`transport_mode='flat'` for `pos_phi_compose='group_product'`); `build_belief_transport` and
`_transport` are public entry points that both `phi_alignment_loss` and `model._gamma_energy` call
directly with `right_phi`. Same silent-flat-fallback failure mode the codebase already fenced with
an explicit raise in families/frame_gaussian.py:109-134 and families/exact_congruence.py:204-210.
**Fix:** Raise in `_transport`/`build_belief_transport` when `right_phi is not None` and
`transport_mode != 'flat'`, rather than relying on the distant config gate.
