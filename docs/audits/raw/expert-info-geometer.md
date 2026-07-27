# Expert — audit-info-geometer (KL, f-divergence, Fisher, alpha conventions)

Returned 2026-07-27 ~09:36 CDT (retry after the 09:00 rate-limit kill). Verbatim; NOT yet verified.
Bands: **no critical**, 2 high, 2 medium, 2 low.

## Clean negatives (the core divergence layer verified correct)

No critical findings. The Gaussian (diagonal and full), exact-congruence, frame-intrinsic, and
Laplace closed forms are each correct against an independent derivation and a float64 reference.
Verified: `D(p||p) = 0` exactly; randomized non-negativity over 400 draws x 4 functionals x 3
families x 4 orders (no violations); the `alpha -> 1` KL limit **and its direction** (`KL(q||p)`,
not the reverse); the generic Bregman/A-path against the closed form; per-coordinate sums against
summed forms; and the exact-congruence pullback identity against a dense full-covariance KL (max
error 1.8e-14 dense route, 1.6e-14 on both fast routes). Argument order
`D(q_i || Omega_ij q_j)` and `KL(s_i || r)` is correct at every call site.

---

### 1. KL clamp zeroes the restoring self-coupling gradient, and no unclamped path exists
**Location:** vfe3/config.py:918
**Severity:** high
**Evidence:** `safe_kl_clamp` (vfe3/families/base.py:26-27) is applied unconditionally to every
divergence, and the validator forbids the only value that would disable it:
```python
if not (math.isfinite(self.kl_max) and self.kl_max > 0.0):
    raise ValueError(f"kl_max must be finite and positive, got {self.kl_max}")
```
The kernel and the oracle both gate the self-term on the upper bound (kernels.py:157,160 and
:669,672), so the clamp is not a numeric guard — it is the objective. Executed, `N=3, K=4`, flat
cocycle:
```
raw self KL(q||p) per token: [288.0, 338.0, 242.0]
kl_max=     100:  ||grad_mu||=1.424392e+00     grad_mu row0 = [0.0, 0.0, 0.0, 0.0]
kl_max=   1e+09:  ||grad_mu||=4.178620e+01     grad_mu row0 = [12.0, 12.0, 12.0, 12.0]
oracle kl_max=100: ||grad_mu||=1.424392e+00      (kernel and oracle agree)
```
The covariance term alone saturates it at the dataclass default: a belief sitting exactly on its
prior mean with `sigma` at the `eps=1e-6` floor gives
```
K=  4: self-KL = 25.63   grad_sigma[kl_max=100] = -4.999995e+05   unclamped = -4.999995e+05
K= 20: self-KL = 128.16  grad_sigma[kl_max=100] =  0.000000e+00   unclamped = -4.999995e+05
```
so the prior anchor switches off exactly where the belief is most pathologically overconfident. A
function constant on an open set is no longer a divergence and induces no metric there (Amari 2016
section 1.3; Cover & Thomas 2006 section 2.3).
**Fix:** Admit `kl_max = float('inf')` in the validator (the decode boundary already passes `inf`
through `safe_kl_clamp` successfully) so the unclamped functional is reachable as the pure path.

> ORCHESTRATOR NOTE: this is a **PURE-PATH EXISTENCE** claim, which is the one category the user
> explicitly cares about ("there should ALWAYS exist a theoretically pure path under appropriate
> toggles"). Distinct from a default-toggle complaint, which is out of scope. ESCALATE to the
> challenge tier — the skeptic should test whether `inf` genuinely fails validation and whether the
> unclamped path is truly unreachable by any other route.

### 2. `route_grow_k` inherits the K-independent `kl_max` its sibling route explicitly corrects
**Location:** scaling.py:499
**Severity:** high
**Evidence:** `BASELINE["kl_max"] = 8 * config["embed_dim"]` is fixed once at scaling.py:479 for
`embed_dim=20`, i.e. 160. Route A overrides only the width:
```python
return [{"label": f"K{k}", "route": "grow_K", "scale_knob": "embed_dim",
         "overrides": {"embed_dim": k, "n_heads": n_heads, "gauge_group": "block_glk"}}
        for k in embed_dims]
```
Its three siblings all set the ceiling per cell — `route_grow_k_mup` at :529 (`"kl_max": 8 * k`),
`route_grow_k_fixed_block` at :560, `route_vary_block_fixed_k` at :580 — and `route_grow_k_mup`'s
own docstring (:518-520) names the defect ("the baseline freezes kl_max at 8*train_K (a width
confound...)"). The active grid runs both on the same width list:
`ROUTES["grow_K"] = route_grow_k([20, 40, 60, 80, 100, 120], ...)` at :701. So route A's divergence
ceiling falls from 8.0 to 1.33 nats/coordinate across the very axis it measures, while route A'
holds it at 8.0 — the two width exponents are not measured under the same functional.
**Fix:** Add `"kl_max": 8 * k` to `route_grow_k`'s overrides, matching the other three builders.

### 3. Gaussian effective-rank floor destroys the participation ratio's scale invariance
**Location:** vfe3/families/gaussian.py:130
**Severity:** medium
**Evidence:** `(sum lam)^2 / sum lam^2` is homogeneous of degree 0, so it is exactly `K` for any
isotropic spectrum at any scale. `metrics.effective_rank` floors the denominator (metrics.py:37:
`s2 = (lam ** 2).sum(dim=-1).clamp(min=eps)`) with the family-supplied value, and the Gaussian
families return `eps` itself — a covariance^1 quantity floored against a covariance^2 denominator:
```python
@classmethod
def effective_rank_floor(cls, eps: float) -> float:
    return eps
```
The live diagnostic passes `eps=cfg.eps` (model.py:3107-3112). Executed at `K=20`, flat spectrum
(true value 20 at every scale):
```
     sigma    gaussian_diagonal   laplace_diagonal
     1e+00            20.000000          20.000000
     1e-03            19.999998          20.000000
     1e-04             4.000000          20.000000
     1e-05             0.040000          20.000000
     1e-06             0.000400          20.000000
```
`laplace_diagonal` is correct because it opts into `effective_rank_rescale = True`
(laplace.py:66, consumed at metrics.py:512-519); the Gaussian families leave it `False`. At the live
`K=210` the floor binds below `sigma ~ 7e-5`, well inside the admissible `[eps, sigma_max]` band, and
`eff_rank_p5/median/p95` then report a maximally *flat* covariance as a sub-rank-1 collapse — the
opposite of what the metric exists to detect.
**Fix:** Set `effective_rank_rescale = True` on the Gaussian families (or return
`covariance_floor(eps)**2` from `effective_rank_floor`) so the ratio stays scale-invariant.

### 4. Generic Renyi-from-`A` path has no natural-parameter-domain gate
**Location:** vfe3/families/base.py:568
**Severity:** medium
**Evidence:** `D_alpha = [A(theta_blend) - alpha A(theta_q) - (1-alpha) A(theta_p)]/(alpha-1)` is
defined only when `theta_blend = alpha*theta_q + (1-alpha)*theta_p` lies in the natural parameter
space `Theta` (Amari & Nagaoka 2000 section 2.3; Nielsen 2020). The generic path evaluates it
unconditionally:
```python
blend = tuple(alpha * a + (1.0 - alpha) * b for a, b in zip(tq, tp))
div = (cls.log_partition_at(blend)
       - alpha * cls.log_partition_at(tq)
       - (1.0 - alpha) * cls.log_partition_at(tp)) / (alpha - 1.0)
```
and `FullGaussian.log_partition_at` (gaussian.py:528) hides the out-of-domain condition behind
`safe_cholesky(neg2t2, rounds=5)`, whose escalating ridge factors a mildly indefinite `-2*theta_2`
as PD and reports `ok=True`. Executed (`K=4`, `alpha=1.5`):
```
min eig covariance blend      : -0.09767583068835056
min eig natural blend (-2*t2) : -0.0011989817851934832
log_partition_at(blend)       : 5.428487655492118        <- finite, not NaN
closed-form  D_1.5(q||p)      : 1000000000.0             (kl_max: correctly undefined)
generic A-path D_1.5(q||p)    : 1.7648168111730191       <- silently wrong finite value
```
The closed form was explicitly hardened against exactly this (gaussian.py:652-659, the
`eigvalsh(...)[..., 0] > 0` gate); the generic path — the documented add-a-family extension seam —
was not. Every currently registered family supplies `renyi_closed_form`, so this is dormant.
**Fix:** Gate `_renyi_from_log_partition` on the blend's membership in `Theta` (mask to NaN when
`log_partition_at`'s factorization fails without a ridge), mirroring the closed form's spectrum test.

### 5. `renyi_order` is silently discarded by three of the four registered divergence functionals
**Location:** vfe3/families/base.py:622
**Severity:** low
**Evidence:** `squared_hellinger`, `bhattacharyya`, and `jeffreys` absorb the order in `**kwargs`
and hardcode `0.5`/`0.5`/`1.0`. Executed on a diagonal Gaussian pair:
```
  renyi              D at alpha=[0.25,0.5,1.0,2.0]: ['0.791759','1.545878','3.007889','6.113068']
  squared_hellinger  D at alpha=[0.25,0.5,1.0,2.0]: ['0.538346','0.538346','0.538346','0.538346']
  bhattacharyya      D at alpha=[0.25,0.5,1.0,2.0]: ['0.772939','0.772939','0.772939','0.772939']
  jeffreys           D at alpha=[0.25,0.5,1.0,2.0]: ['6.277686','6.277686','6.277686','6.277686']
```
The config's inert-field reporter, which exists precisely to catch this (config.py:2956-3031, e.g.
`_inert.append("b0/c0 (only read by a state_dependent lambda_alpha_mode)")`), has no `renyi_order`
entry — so a sweep over `renyi_order` with a non-alpha functional produces byte-identical arms under
distinct run labels with no warning. The only record is a comment at config.py:1049.
**Fix:** Append `renyi_order` to the `_inert` list when `divergence_family` has no registered order
parameter.

### 6. `alpha > 1` warning states the Gaussian blend condition for every family
**Location:** vfe3/families/base.py:32
**Severity:** low
**Evidence:** `renyi()` (base.py:595-596) dispatches the warning generically on
`type(q).__name__`, but the text is Gaussian-specific:
```python
f"the blend (1-alpha)*Sigma_q + alpha*Sigma_t may be non-positive-definite "
f"(diagonal clamps; full may fail Cholesky and return NaN)."
```
Executed, this fires verbatim with `family='DiagonalLaplace'`, whose actual divergence condition is
`c_q + c_p = alpha/b_q + (1-alpha)/b_p > 0` on a two-region exponential integral (laplace.py:252,
258, 315) — no covariance blend and no Cholesky involved.
**Fix:** Move the warning text behind a family classmethod, or state the condition family-neutrally.
