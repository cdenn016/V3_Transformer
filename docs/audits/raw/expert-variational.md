# Expert — audit-variational (free energy / ELBO / EM separation)

Returned 2026-07-27 ~08:52 CDT. Verbatim findings; NOT yet verified.
Bands: **no critical**, 1 high, 2 medium, 1 low.

## Clean negatives worth recording (checked, no defect found)

- **Softmax stationarity of the implemented F.** SymPy on the exact expression `free_energy`
  builds (`lam*(sum_j b_j E_j + tau*sum(b_j log b_j - b_j log pi_j))` under `sum b = 1`) returns
  `residual vs pi_j exp(-E_j/tau)/Z : 0` and `block(beta*) + lam*tau*log Z = 0`. So
  `attention_weights` (free_energy.py:329-332), `log_partition` (:351-357) and
  `reduced_free_energy` (:382) are mutually coherent and beta IS a genuine stationary point, not a
  delta. With the entropy removed the row Lagrangian derivative is `E_j*lam + nu` for every `j` —
  no interior solution, confirming the manuscript's claim.
- **Entropy-suppressed surrogate is fenced off from the envelope kernel.** `uses_kernel_route`
  requires `include_attention_entropy` (kernels.py:304), so `include_attention_entropy=False`
  routes to the autograd oracle, which differentiates `(beta * energy).sum()` with `beta` live
  (free_energy.py:434, 448) and therefore carries the `-tau^-1 Cov_beta(E, grad E)` term exactly.
  `phi_alignment_loss` mirrors this (e_step.py:805-806).
- **Alpha envelope.** SymPy: `d/dD[alpha*(D)*D + R(alpha*(D))] - alpha*(D) = 0` for
  `alpha* = c0/(b0+D)`, `R = b0*alpha - c0*log alpha`, so `alpha_gradient_coefficient`
  (alpha_i.py:153) is the exact coefficient of the F self-block, not an approximation.
- **E-step descent.** Executed `n_iter=8`, `K=4, N=5`: gradient route `0.004815 -> 0.002882`,
  strictly monotone; `mm_exact` `0.004815 -> 0.002474`, converged by iteration 3 (the single
  non-monotone flag is at the 1e-9 fp32 tolerance, values identical to 6 dp).
- **No label leakage into inference.** `targets` first appears at `model.py:1567`, after
  `forward_beliefs` returns (:1543); the emission reads `token_ids` (x_t) only (model.py:2404,
  emission.py:140).
- **Emission sigma arm is self-consistent.** `sigma_star = (a + pair_mass)/P` with `P` including
  `emission_weight*d` is the exact stationary point of the `E_q`-averaged Bohning surrogate (whose
  sigma dependence is `0.5*w*sum_k d_k sigma_k`); residual `1.8e-07`.

---

### 1. The E-step's emission block is absent from `free_energy_value`, so every logged/scored F measures a different functional than the belief descends
**Location:** vfe3/inference/e_step.py:471
**Severity:** high
**Evidence:** The evaluator declares the data term as inert:
```python
471	    emission_weight:           float = 0.0,            # accepted-and-ignored iteration-only knob (emission block)
472	    emission:                  Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # accepted-and-ignored iteration-only runtime object
```
Nothing downstream of line 472 reads either name, while `e_step_iteration` forwards both into
`mm_exact_update` (e_step.py:1029), which fuses them into `prec`/`numerator`
(gradients/kernels.py:700-701). Executed on a tiny model (`K=4, V=16, N=4, n_layers=1,
emission_mode='separate', emission_weight=1.0, e_step_update='mm_exact'`, CUDA interpreter):
```
F(no emission arg)   = 0.00637892447412014
F(emission supplied) = 0.00637892447412014   <- identical: the block is dropped
||mu_on - mu_off||   = 0.43443334102630615
reported F:  start 0.006379   emission-off step 0.002218   emission-on step 0.615477
true descended objective F + emission:  start 4.825152  on 3.802504  off 4.837981
```
One E-step with the emission on lowers the objective it minimizes (4.825 -> 3.803) while the
reported F *rises* by 96x (0.0064 -> 0.6155). The same blind spot reaches the Metropolis acceptance
test (model/model.py:1281, which never passes `emission`) and the viz replay, whose `_iter_kwargs`
carries `emission_weight` from `e_step_shared_kwargs` but never the `(d, g)` tuple
(viz/extract.py:313-337), so `emission is not None` is False and the replayed E-step silently drops
the data term production keeps. This is the D-08 class exactly: `estep_f_drop`,
`estep_f_nondecreasing_frac` (train.py:1048-1054) and `estep_final_f_per_token`
(run_artifacts.py:2916) all become measurements of a functional nothing optimizes.
Canon: Bishop 2006 section 10.1 — the reported bound must be the bound the coordinate updates
minimize.
**Fix:** Give `free_energy_value` a live `emission`/`emission_weight` branch adding
`w*(0.5(mu-z0)^T diag(d)(mu-z0) - (mu-z0)^T g + 0.5 tr(diag(d) Sigma))` and thread the `(d, g)`
tuple through `_f_diag`, the Metropolis scorer, and `_fe_kwargs`/`_iter_kwargs`.

> ORCHESTRATOR NOTE: this is a direct consequence of the accept-and-ignore parameter added
> yesterday (2b7a96d) to resolve a 47-test TypeError cascade. Declaring the parameter fixed the
> crash; leaving it inert left the evaluator measuring a different functional. Part of the
> EMISSION CLUSTER — escalate.

### 2. The Bohning majorizer is anchored at the pre-stack mean but fused against each layer's own prior, so at `n_layers > 1` the fusion minimizes a quadratic centered on the wrong point
**Location:** vfe3/gradients/kernels.py:701
**Severity:** medium
**Evidence:** `(d, g)` is built once, from the belief entering the stack:
```python
model.py:1126	            emission = self._emission_terms(token_ids, beliefs.mu)
```
and handed unchanged to every block (stack.py:142), while the block's prior mean advances each
layer (stack.py:151: `mu_p = (1.0 - rho) * mu_p + rho * belief.mu`, `prior_handoff_rho` default
`1.0`). The fusion then re-centers the quadratic on that moving `mu_p`:
```python
701	        numerator = numerator + emission_weight * (emission_prec * mu_p + emission_pull)
```
`g = W^T(e_x - softmax(W z0))` is evaluated at `z0`, so the implemented pull is `D(z - mu_p) - g`
where the majorizer of `-log softmax(Wz)_x` gives `D(z - z0) - g`; they differ by `D(mu_p - z0)`.
Executed gradient-residual check of the beta-frozen surrogate anchored at `z0`, evaluated at the
`(mu*, sigma*)` `mm_exact_update` returns (`N=3, K=4`, max pair energy 0.72 << kl_max):
```
layer prior == Bohning expansion point (n_layers=1):  |dF/dmu|=1.199e-07 |dF/dsigma|=1.837e-07
layer prior != expansion point (n_layers>1, rho=1):   |dF/dmu|=5.541e-01 |dF/dsigma|=1.837e-07
```
Exact when the anchor matches, off by O(1) when it does not — the fused point is no longer the
minimizer of any majorizer of the emission NLL. (The sigma arm is unaffected: `(a + pair_mass)/P`
is the exact stationary point of the `E_q`-averaged surrogate, anchor-independent.)
**Fix:** Store the expansion point `z0` alongside `(d, g)` and use `emission_prec * z0 +
emission_pull` in the numerator, or rebuild `(d, g)` per layer from that layer's `mu_p`.

### 3. The Metropolis Delta-F adds a batch-summed belief free energy to a batch-mean model free energy, suppressing the h/s channel by 1/(B*N)
**Location:** vfe3/model/model.py:1308
**Severity:** medium
**Evidence:** `free_energy_value` reduces with `.sum()` over every batch and query axis
(free_energy.py:440, :448, :469), while `_model_channel_free_energy` reduces its three s-channel
rows with `model_reduction="mean"` (model.py:2107-2108 -> free_energy.py:585-587 ->
`_reduce_row(..., "mean")`). The scorer adds them without rescaling:
```python
1308	                total = total + self._model_channel_free_energy(
1309	                    context.token_ids, belief, s_belief=s_belief)
```
Executed, tiny model `B=4, N=5, K=4, lambda_h=0.25, lambda_gamma=0.75`:
```
B*N = 20
free_energy_value (belief channel, SUM over B and N) = 0.028852282091975212
_model_channel_free_energy (s channel, MEAN over B,N) = 0.0009673853637650609
the same s rows reduced by SUM would be              = 0.019347707275301218
```
and on unit rows, `q self_coupling (sum) = 30.0` vs `s hyper_prior (mean) = 1.0` at `B*N = 30`. The
docstring at model.py:1259 claims "the Metropolis DeltaF is the exact change in the joint F under
the block move"; the assembled scalar is instead `F_belief + (1/BN)*F_model`, so a frame flip that
improves the gauge-transported model consensus is weighted 1/(B*N) against its belief-channel cost.
Reachable under `phi_reflection='metropolis'` or `omega_reflection='metropolis'` with either
model-channel weight positive.
**Fix:** Call `_model_channel_free_energy` with `model_reduction="sum"` from the Metropolis scorer
(keeping `"mean"` for the training loss, which is commensurate with the per-token `ce`).

### 4. `free_energy_value` gates the two-hop block on `> 0.0` while every other consumer gates on `!= 0.0`
**Location:** vfe3/inference/e_step.py:665
**Severity:** low
**Evidence:** The reported-F path `665	    if lambda_twohop > 0.0:` against the descended paths —
free_energy.py:471 `if lambda_twohop != 0.0:`, e_step.py:807 (`phi_alignment_loss`)
`if lambda_twohop != 0.0:`, kernels.py:170 `if lambda_twohop != 0.0:`. A negative `lambda_twohop`
is descended by the kernel, the oracle and the phi sub-step but omitted from the logged and
Metropolis-scored F. `config.py:2900` rejects `lambda_twohop < 0`, so no shipped config reaches it;
live only for direct callers (viz extractors, tests).
**Fix:** Change `> 0.0` to `!= 0.0` so the authoritative evaluator matches the four descent paths.
