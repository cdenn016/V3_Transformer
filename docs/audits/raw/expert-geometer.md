# Expert — audit-geometer (SPD manifold, exp/log, transport, geodesics)

Returned 2026-07-27 ~09:22 CDT (retry after the 09:00 rate-limit kill). Verbatim; NOT yet verified.
Bands: **no critical, no high**, 2 medium, 2 low. All checks executed at K=4, float64.

## Clean negatives (extensive — this lens came back largely clean, with executed proofs)

- `retract_spd_full` IS the exact affine-invariant exponential map:
  `||R(S,X) - S^{1/2} expm(S^{-1/2} X S^{-1/2}) S^{1/2}|| / ||.|| = 3.78e-16`; log round-trip
  `log_S(exp_S(X)) = X` to `2.40e-13`.
- The retraction is exactly congruence-equivariant on the pure path (`sigma_max=None`):
  `R(gSg^T, gXg^T) = g R(S,X) g^T` to `4.11e-15`. It deviates (`0.756`) only once the documented
  `sigma_max` spectral cap is active — which `run_artifacts.py:4104` already records as
  `spd_retraction_exact = sigma_max is None`.
- `retract_logeuclidean_full` equals `expm(logm(S) + Dlog_S[X])` to `9.18e-16`; `_frechet_log_spd`
  matches central finite differences of `logm` to `3.19e-08`.
- Diagonal and full arms agree exactly on diagonal input (`1.11e-16`); `log_euclidean` equals
  `spd_affine` on commuting diagonal covariances to `0.0` — the equivalence `config.py:2238` asserts.
- `spd_geodesic_distance` is exactly the AIRM distance and congruence-invariant
  (`0.09028975110856227` vs `0.09028975110859051`), equal to `||Sigma^{-1/2} X Sigma^{-1/2}||_F`.
- `natural_gradient` returns exactly `2 Sigma G Sigma` (`1.42e-14`), the correct raised-index
  gradient for `g(dS1,dS2) = (1/2) tr(S^{-1} dS1 S^{-1} dS2)`; the diagonal arm `2 sigma^2 grad` and
  the Laplace arm `b^2 grad` are likewise the correct Fisher inverses. `retract_spd_full`'s trust
  region bounds `||Sigma^{-1/2} X Sigma^{-1/2}||_F`, which is congruence-invariant, so the step
  bound is itself affine-invariant.
- `pullback_metric` implements the correct left-trivialized differential
  `D exp_phi(H) = Psi(ad_phi)(H) exp(phi)` with `Psi(z) = (e^z-1)/z` (Hall, *Lie Groups* Thm 5.4);
  `ad` index placement, series coefficients `1/(k+1)!`, and column extraction all check out.
- `so_n`/`sp_n` irrep towers are genuine Lie-algebra homomorphism images: image bracket residual
  `0.00e+00`-`7.77e-16`; `so` images exactly skew (`2.22e-16`), `sp` non-skew as declared.
- The RoPE right-fold `V_i = U_i R_i^T`, its compact per-block slicing, and the left-insertion
  `R_i Omega_ij R_j^T` einsums are index-correct; `fold_rope_into_frame` yields an ordinary
  coboundary.
- `exact_congruence.py`'s pullback identity
  `2E = sum s_tilde/s_j + sum (mu_j - a)^2/s_j - K + sum log s_j - sum log s_i + 2 log|det Omega|`
  is algebraically correct, as are the fast-route `Omega_ij^{-1} = Omega_ji` index transpose and
  the `ell_i - ell_j` determinant.
- `gauge_invariant_edge_features`' three scalars are genuinely invariant under the common `g_i`
  pushforward, so Regime-II-covariant's `Omega_ij -> g_i Omega_ij g_j^{-1}` claim holds;
  `holonomy_wilson_sampled`'s `Re Tr(H)/K` is the conjugation-invariant observable while
  `holonomy_deviation`'s Frobenius form is correctly flagged frame-dependent.

---

### 1. The SPD retraction's `[eps, sigma_max]` bounds are applied to the raw dispersion slot, so the same config value means a different physical covariance per family
**Location:** vfe3/inference/e_step.py:1147-1150; bound applied at vfe3/geometry/retraction.py:461-466; family conversion that exists but is never consulted at vfe3/families/laplace.py:97-107
**Severity:** medium
**Evidence:** The E-step passes the config scalars straight through with no family key:
```python
sigma = get_retraction(spd_retract_mode)(
    belief.sigma, -e_q_sigma_lr * nat_sigma, belief.mu.dim(),
    trust_region=e_sigma_q_trust, eps=eps, sigma_max=sigma_max,
)
```
and `retract_spd_diagonal` clamps that slot directly
(`sigma_new.clamp(min=lower_bound, max=upper_bound)`). For `family='laplace_diagonal'` the slot is
the Laplace scale `b`, not a variance — the family itself declares
`dispersion_is_covariance = False` and supplies `covariance_diagonal(b) = 2 b^2`,
`covariance_floor(eps) = 2 eps^2`. Executed (defaults `eps=1e-6`, `sigma_max=10.0`):
```
retraction output on the belief-sigma slot: [[9.999999046325684, 1.0000001111620804e-06, 1.0]]
Gaussian:  dispersion in [1e-06, 10] -> variance in [1e-06, 10]
Laplace :  dispersion in [1e-06, 10] -> variance in [2e-12, 200]
declared covariance_floor(eps): gaussian=1e-06  laplace=2e-12
```
So `sigma_max=10.0` is a 20x looser covariance ceiling and `eps=1e-6` a 5e5x tighter covariance
floor under Laplace than under Gaussian, from the identical config. `covariance_floor` /
`covariance_diagonal` are consumed only by metrics.py:488,499 and viz/figures.py:3028 — never by the
retraction. The same blindness gives the shared `e_sigma_q_trust=5.0` a Fisher-norm meaning of
`5/sqrt(2)` for the Gaussian (`g = ds^2/(2s^2)`) and `5` for the Laplace (`g = db^2/b^2`). The
exponential map itself is correct for both: `b exp(v/b)` with `v = b^2 grad` is the exact geodesic
of `g = db^2/b^2`.
**Fix:** Route the retraction's floor/ceiling (and the trust radius) through the family's
`covariance_floor` / dispersion-to-covariance hook so `cfg.sigma_max` denotes one physical
covariance bound across families.

### 2. `estep_residuals` and the layer-depth diagnostic hardwire the affine-invariant distance while the retraction geometry is config-selectable
**Location:** vfe3/metrics.py:1469; vfe3/viz/extract.py:920-924
**Severity:** medium
**Evidence:** `r_sigma = spd_geodesic_distance(sigma_traj[:-1], sigma_traj[1:], diagonal=diagonal,
eps=eps)` reads no `spd_retract_mode`, yet `spd_geodesic_distance` advertises itself
(metrics.py:777-779) as "the metric the SPD retraction itself uses, so belief-trajectory /
E-step-residual lengths are measured in the geometry the inference actually moves in."
`log_euclidean` is a registered, validated `spd_retract_mode` (config.py:2228) and is a *distinct*
geometry under `family='gaussian_full'` (config.py:2233-2243 says so explicitly).
`run_artifacts.py:4105-4114` does branch on the mode; the residual metric does not. Executed, K=4,
200 draws, `retract_logeuclidean_full` step, comparing the reported AIRM length against the true
log-Euclidean chart length `||logm(S2) - logm(S)||_F`:
```
log_euclidean retraction, 200 draws K=4: worst |AIRM - LE| / LE = 33.3%
```
Single-draw detail: `AIRM-reported r_sigma=0.584844   true LE chart length=0.570890   rel
discrepancy=2.444%`. On `spd_affine` the reported value is exact
(`0.585036 == ||S^-1/2 X S^-1/2||_F`), and on any diagonal family the two retractions coincide
bit-for-bit (measured `0.0`), so the mismatch is confined to `gaussian_full` + `log_euclidean`.
**Fix:** Key the residual/interpolation metric on `spd_retract_mode` (log-Euclidean chart length for
`log_euclidean`, AIRM otherwise), or drop the "the metric the retraction uses" claim.

### 3. The Killing preconditioner is exactly a scalar on every shipped orthonormal-basis group, so `killing` and `killing_per_block` are learning-rate reparameterizations of `none`
**Location:** vfe3/geometry/phi_preconditioner.py:809-821 (`killing_metric`), :940-953 (`_precond_killing`), :980-1012 (per-block)
**Severity:** low
**Evidence:** `killing_metric` returns `2*K*gram - 2*outer(traces, traces)`. For a
Frobenius-orthonormal generator basis (`groups.py` declares `frobenius_gram_uniform=1.0` for `glk`
and `block_glk`, `2.0` for `so_k`) `gram` is a multiple of `I` and `outer(traces,traces)` is rank one
supported on the `K` diagonal generators with `||v||^2 = K`, so the spectrum is
`{2K (multiplicity n_gen-1), 0}` — and `center_reg` defaults to exactly `2*K`, lifting the one null
direction to the same value. Executed:
```
glk              n_gen= 16 K=4  killing eigs(min,max)=1.776e-15,8  Minv==c*I ? maxdev=3.910e-17  c=0.125  1/(2K)=0.125
block_glk        n_gen=  8 K=4  killing eigs(min,max)=1.776e-15,8  Minv==c*I ? maxdev=3.036e-17  c=0.125  1/(2K)=0.125
                 per_block: Minv==c*I ? maxdev=0.000e+00  c=0.25  1/(2d)=0.25   ratio killing/per_block = 0.5
so_k             n_gen=  6 K=4  killing eigs(min,max)=16,16  Minv==c*I ? maxdev=0.000e+00  c=0.0625
                 precond('killing') / grad: min=0.125 max=0.125     (identical on every component)
```
`killing` and `killing_per_block` differ only by the constant factor `n_heads` (`K/d`), because the
full variant applies gl(K)'s `2K` trace form to the proper subalgebra `gl(d)^{+H}` whose own Killing
form carries `2d` (do Carmo 1992 section 2 / Lee 2012 Ch. 20: the Killing form of a direct sum is
the direct sum of the summands' forms, not the restriction of an ambient one). Only
`tied_block_glk` gets a genuinely non-scalar metric (`maxdev=3.125e-02`); the live config uses
`pullback_per_block`, which IS a real position-dependent metric.
**Fix:** Either document these two registry entries as constant rescales on orthonormal bases (and
drop the "natural gradient" framing there), or build the per-block form from each summand's own
Killing form so the two are not redundant.

### 4. The inter-layer prior handoff interpolates the SPD covariance by a Euclidean lerp while the rest of the pipeline uses geodesic and natural-parameter blends
**Location:** vfe3/model/stack.py:152 (and the diagnostic replica at vfe3/viz/extract.py:912)
**Severity:** low
**Evidence:** `sigma_p = (1.0 - rho_s) * sigma_p + rho_s * belief.sigma`, with
`prior_handoff_sigma = 0.1` in the live `train_vfe3.py:379`. Three different interpolations of the
same SPD object coexist: this arithmetic lerp; the affine-invariant geodesic
`Sigma^{1/2} exp(Sigma^{-1/2} X Sigma^{-1/2}) Sigma^{1/2}` in retraction.py:529 (verified exact,
residual `3.78e-16`); and the natural-parameter (precision) blend
`lam_new = (1-eta) lam_old + eta / sigma_star` at e_step.py:1053-1056. The lerp IS SPD-preserving
and congruence-equivariant (`g((1-t)A + tB)g^T = (1-t)gAg^T + t gBg^T`), so it is not an error — but
by AM-GM in the Loewner order it sits strictly above the geodesic mean
(`det((1-t)A+tB) >= det(A)^{1-t} det(B)^t`, Bhatia 2007 section 4.1), so the handed-off prior
covariance is systematically inflated relative to the AIRM interpolation the belief itself travels
along, which directly softens the `KL(q_i||p_i)` self-coupling.
**Fix:** Offer the AIRM geodesic (or the precision blend already used by `mm_exact`) as the
covariance handoff and make the interpolation an explicit registry choice rather than a hardcoded
lerp.
