# Expert — audit-numerical-analyst (conditioning, finite precision, SPD)

Returned 2026-07-27 ~08:45 CDT. Verbatim findings; NOT yet verified.
All probes run under `C:/anaconda/python.exe` (torch 2.10.0.dev+cu128, CUDA available).
No repository files modified; probes ran from the scratchpad.

Bands: **no critical**, 3 high, 4 medium, 2 low. (Investigator's own ratings — see orchestrator
note on #1, which the orchestrator considers critical-class.)

---

### 1. `mm_exact_update`'s autocast island silently drops the entire emission factor
**Location:** vfe3/gradients/kernels.py:573-594
**Severity:** high (investigator) — **ORCHESTRATOR: CRITICAL-CLASS, see note**
**Evidence:** The fp32 re-entry forwards 15 of the 17 keyword arguments; `emission_weight` and
`emission` are absent, so the recursive call takes the defaults `emission_weight=0.0,
emission=None` and the guarded block at `:692` never runs:
```python
    if torch.is_autocast_enabled(mu.device.type):
        with torch.autocast(device_type=mu.device.type, enabled=False):
            return mm_exact_update(
                mu.float(), sigma.float(), mu_p.float(), sigma_p.float(),
                ...
                irrep_dims=irrep_dims,
                log_prior=(log_prior.float() if log_prior is not None else None),
            )
```
Not a corner case: `config.py:1648-1656` *requires* `e_step_update='mm_exact'` whenever
`emission_mode != 'off'`, and the live `train_vfe3.py` sets `amp_dtype='bf16'`. Executed
reproduction (K=4, N=6, two 2x2 blocks, identity transport):
```
||mu*(emission on, fp32) - mu*(emission off)||         = 5.468173980712891
||mu*(emission on, bf16 autocast) - mu*(emission off)|| = 0.0
emission term SURVIVES autocast? False
```
Config validation warns only that `emission_weight == 0.0` is inert
(`config.py:1664-1669`); nothing detects this.
**Fix:** Forward `emission_weight=emission_weight` and
`emission=(emission[0].float(), emission[1].float()) if emission is not None else None` in the
autocast recursion.

> ORCHESTRATOR NOTE — HIGHEST PRIORITY OF THE AUDIT SO FAR. This makes the categorical emission
> factor (shipped yesterday, commit 2b7a96d) a **silent no-op under the user's live AMP config**.
> The user's pre-registered next experiment is exactly the emission arms at K=20. If this holds,
> those runs would measure nothing and would be misread as "emission doesn't help." The two
> independent gates compose: emission REQUIRES `mm_exact`, and `mm_exact` under autocast drops
> emission. Escalate to the challenge tier first; notify the user before they launch.
> NOTE the interaction with the code-reviewer's emission finding (`emission.py:110-140`
> detachment) and this agent's own #6 (`emission.py` has no fp32 island) — three separate
> findings converge on the same new module. Treat as a cluster.

### 2. The diagonal-congruence fp64 escalation tests the sign, not the accuracy, so a 100%-wrong variance passes
**Location:** vfe3/geometry/transport.py:1428-1431 (guard), used at :2712-2714 and :2617-2619
**Severity:** high
**Evidence:** The escalation criterion is a single sign test on the whole reduction:
```python
    with torch.no_grad():
        worst = float(out.amin())
    if worst >= 0.0:
        return out
```
Cancellation in the mixed-sign factored quadratic form loses accuracy long before it loses sign.
Measured against an fp64 dense `sum_l Omega_kl^2 sigma_l` reference, per-block `||phi_h||_F` fixed
exactly, `_factored_diagonal_covariance` (d=6, H=2, N=8):
```
 r      cond(U)     escalated?   max rel err
 5.0   5.495e+02       False     5.1505e-04
 7.0   6.212e+03       False     5.1461e-03
 8.0   2.064e+04       False     3.4019e-01
 9.0   7.294e+03       False     2.3364e-01
10.0   2.794e+04       False     1.2665e+00
```
and the compact sibling `_compact_factored_diagonal_covariance`:
```
 8.0   6.439e+03       False     3.9834e-02
 8.0   1.912e+03       False     6.2647e-02
10.0   5.371e+04        True     2.3734e-04   <- escalation works when it fires
```
At `r=10, cond 2.8e4` the transported variance is off by 127% with every entry strictly positive.
`TRANSPORT_CLAMP_MAX_NORM = 20.0` (transport.py:1393) is the norm the codebase itself treats as the
operating bound, so this band is inside the admitted regime, and the corrupted `sigma_t` feeds the
attention energy grid directly.
**Fix:** Key the escalation on a cheap accuracy proxy as well as the sign — escalate when
`out.amin() < 0` **or** `max_h cond(exp_blocks_h)^2 * finfo(work).eps` exceeds a tolerance (a
Higham 2002 section 3.1 backward-error bound for the quadratic form), not on `amin() >= 0` alone.

### 3. fp32 cancellation in `diag_kl_unclamped` flips the self-coupling saturation mask off for ~50% of near-converged tokens
**Location:** vfe3/families/gaussian.py:47-53, consumed at vfe3/gradients/kernels.py:156-160
**Severity:** high
**Evidence:** `KL = 0.5*(trace + mahal - K + logdet)` sums three O(K) quantities that cancel to
O(K r^2); in fp32 the absolute rounding floor is ~1e-6 for K=30, sigma~4. The gradient kernel then
gates the *entire* alpha self-term on the sign of that value:
```python
        raw_self  = _raw_diag_kl(mu_q, sigma_q, mu_p, sigma_p, eps=eps)         # (N,)
        self_mask = ((raw_self > 0.0) & (raw_self < kl_max)).to(mu_q.dtype).unsqueeze(-1)
```
Executed (4096 tokens, K=30, sigma in [2,5], relative belief-prior separation `r`):
```
   sep r  true KL (med)  frac mask==0  min fp32 raw
   1e-03     1.1827e-05        0.0000    4.5598e-06
   3e-04     1.0696e-06        0.1375   -1.7285e-06
   1e-04     1.1871e-07        0.4656   -2.2352e-06
   1e-05     1.1988e-09        0.5129   -2.7418e-06
```
`forward_beliefs` anchors `q0 == p`, so the E-step operates precisely in this band. The repo
already contains the fix for the *pair* grid — `pairwise_stats.py:96-110` performs the identical
reduction in float64 — but the self term never routes through it.
**Fix:** Perform the three coordinate reductions in `diag_kl_unclamped` /
`diag_kl_unclamped_per_coord` in float64 and cast back, matching `diagonal_kl_pair_stats`.

### 4. Graph-connected trust-region norms make the zero-tangent double backward NaN in four retractions
**Location:** vfe3/geometry/lie_ops.py:711-712 and :728-729; vfe3/geometry/retraction.py:519-520 and :632-633
**Severity:** medium
**Evidence:** (The diagonal siblings at `retraction.py:458/706` are the known C-09 and are not
re-reported.) The norms are built inside the autograd graph:
```python
    update = step_size * delta_phi
    if trust_region is not None and trust_region > 0:
        u_norm = update.norm(dim=-1, keepdim=True)
        update = update * (trust_region / (u_norm + eps)).clamp(max=1.0)
```
`||.||` is not twice differentiable at the origin. Executed:
```
--- phi retraction (_retract_core, gl(2) basis) ---
  phi=0, delta=0 :  grad(delta) finite? True  grad(delta)= [1.0, 1.0, 1.0, 1.0]
  double backward: NON-FINITE [nan, nan, nan, nan]
retract_spd_full (R-norm, retraction.py:519): 1st backward finite=True, double backward -> NaN
retract_logeuclidean_full (t_norm, retraction.py:632): 1st backward finite=True, double backward -> NaN
```
The codebase already fixed exactly this pattern twice for exactly this reason —
`_omega_retract_cayley` (lie_ops.py:889) and `stable_matrix_exp_pair` (transport.py:1515) both wrap
the norm+scale in `torch.no_grad()` — but these four sites were not updated.
**Fix:** Compute the norm and clamp factor under `torch.no_grad()` at all four sites, as the two
already-fixed siblings do (the clamp is a safeguard, not part of the modeled operator, so the
detached scale is byte-identical wherever it is inactive).

### 5. The Laplace KL branch is hardcoded to float32 and returns a negative divergence near convergence
**Location:** vfe3/families/laplace.py:237-243
**Severity:** medium
**Evidence:** Inputs unconditionally downcast, no float64 island — unlike the `alpha != 1` branch
immediately below (:244-248, which does `.double()`):
```python
        mu_q = self.mu.float()
        b_q = self.sigma.float().clamp(min=eps)
        mu_p = other.mu.float()
        b_p = other.sigma.float().clamp(min=eps)
        s = (mu_q - mu_p).abs()
        if abs(alpha - 1.0) < 1e-6:
            return torch.log(b_p) - torch.log(b_q) + s / b_p + (b_q / b_p) * torch.exp(-s / b_q) - 1.0
```
Executed (K=30, b~[2,5]; the "fp64" column is identical because the method downcasts regardless of
input dtype):
```
       r   Laplace value    (summed over K)
   1e-02     1.6204e-03
   1e-03     1.6332e-05
   1e-04    -1.7881e-07     <- negative divergence
   1e-05    -2.9802e-07     <- negative divergence
```
`safe_kl_clamp` hides the value at 0 but the sign is consumed raw by
`pair_mask = ((energy > 0.0) & ...)` (kernels.py:477), so the pair is dropped from the coupling
derivative.
**Fix:** Evaluate the `alpha == 1` branch in a float64 island and cast back, as `alpha != 1` does.

### 6. `bohning_emission_terms` has no fp32 island and runs its softmax in bf16 under the live AMP context
**Location:** vfe3/emission.py:110-140, invoked at vfe3/model/model.py:1126 inside `with run, amp:` (model.py:1046)
**Severity:** medium
**Evidence:** Every autocast-eligible op in the three passes
(`expansion @ tile.transpose(-1,-2)`) runs at the autocast dtype; there is no
`torch.amp.autocast(..., enabled=False)` anywhere in the module, unlike `belief_gradients`
(kernels.py:405), `mm_exact_update` (kernels.py:573), `stable_matrix_exp_pair` (transport.py:1597),
`_factored_diagonal_covariance` (transport.py:2700) and the dense congruence (transport.py:2359,
whose comment records the identical fix for a measured 4.63e-03 bf16 error). Executed (V=20000,
K=16, CUDA):
```
emission g under bf16 autocast: ||g_bf16 - g_fp32||/||g_fp32|| = 0.006030585616827011
```
Separately verified **CORRECT**: the streaming running-max/renormalize algebra is exactly
equivalent to a one-shot logsumexp (`max abs diff 3.46e-06, max rel 5.50e-05` vs
`W[ids] - softmax(mu@W.T+b) @ W` at V=3000), and it stays finite at `max|logit| = 3443.8`. The
`exp_sum.clamp(min=eps)` at :131 is provably inert (the running-max term contributes exactly 1).
**Fix:** Wrap the three-pass body in `with torch.amp.autocast(mu_p.device.type, enabled=False):`
and `.float()` the inputs, matching the sibling kernels.

### 7. `condition_number` and `floor_eigenvalues` downcast float64 inputs to float32 before `eigh`
**Location:** vfe3/numerics.py:299 and :259
**Severity:** medium
**Evidence:**
```python
    evals = torch.linalg.eigvalsh(_symmetrize(matrix.float()))
    lam_min = evals[..., 0]
```
```python
    M = _symmetrize(matrix.float())
    evals, evecs = torch.linalg.eigh(M)
```
`.float()` is a hard downcast, not a promotion, contradicting the "float64 stays float64" policy
implemented in `safe_spd_inverse` (:230), `retract_spd_full` (:499) and `natural_gradient` (:741).
A float32 `eigvalsh` cannot resolve `lambda_min` below ~`eps*lambda_max`, so an SPD matrix past
cond ~1e8 is reported non-PD. Executed on exactly-SPD float64 inputs (K=6, geometric spectrum):
```
   true cond       reported
     1.0e+04     9.99638e+03
     1.0e+06     1.00413e+06
     1.0e+08     8.83694e+07
     1.0e+10             inf   <- reported as "not positive definite"
```
`floor_eigenvalues(fp64, floor=1e-12)` on `diag(1, 1e-9, 1e-9, 1e-9)` returns min eigenvalue
`9.99999972e-10` — the fp32 round trip perturbs a value the floor was never meant to touch.
**Fix:** Use `matrix.double() if matrix.dtype == torch.float64 else matrix.float()` in both, the
promotion idiom already used at numerics.py:230.

### 8. `safe_spd_inverse` runs `pinv` over the whole batch and can raise despite the "never raises" contract
**Location:** vfe3/numerics.py:248-249
**Severity:** low
**Evidence:**
```python
    if not bool(ok.all()):                                   # pinv ONLY the still-failed elements
        out = torch.where(ok.unsqueeze(-1).unsqueeze(-1), out, torch.linalg.pinv(M))
```
The comment says "pinv ONLY the still-failed elements", but `torch.linalg.pinv(M)` is evaluated on
the entire batch before `torch.where` selects — every well-conditioned sibling pays a full SVD, and
an SVD non-convergence in *any* batch element raises `LinAlgError`, defeating the escalating-ladder
design whose premise (`cholesky_ex`, :234/:242) is that one bad element cannot abort the call. The
sibling `safe_cholesky` (:207-209) correctly restricts its retry to the failed mask.
**Fix:** Index the failed elements (`M[~ok]`) into `pinv` and scatter back.

### 9. `bohning_curvature_diagonal` uses the cancellation-prone `E[x^2]-E[x]^2` form behind an eps floor unreachable in float32
**Location:** vfe3/emission.py:82-84
**Severity:** low
**Evidence:**
```python
    sum_sq = (weight * weight).sum(dim=0)                          # (K,) sum_v W_vk^2
    sq_sum = weight.sum(dim=0) ** 2                                # (K,) (sum_v W_vk)^2
    return (0.5 * (sum_sq - sq_sum / vocab_size)).clamp(min=eps)   # (K,) SPD guard for the fusion
```
This is `V/2` times the one-pass variance identity (Higham 2002 section 1.9 flags exactly this
form). Executed at the live `V=50257, K=210`, W columns std 0.02, increasing mean `m`, vs a float64
reference:
```
  col mean m     fp64 d_k     fp32 d_k    rel err
       0.000  9.88365e+00  9.88365e+00  2.379e-07
       0.200  9.88354e+00  9.88354e+00  3.903e-05
       1.000  9.85218e+00  9.85156e+00  8.689e-04
```
The `clamp(min=1e-12)` is also unreachable: for a genuinely constant column (`d_k = 0` exactly) the
fp32 residual is ~`V*c^2*1e-7 ~ 5e-3`, five orders above the floor, so the advertised "SPD guard
for the fusion" never binds and the returned curvature is a rounding artifact rather than zero.
**Fix:** Use the two-pass centered form `0.5 * ((W - W.mean(0)) ** 2).sum(0)` — algebraically
identical, cancellation-free, non-negative by construction.
