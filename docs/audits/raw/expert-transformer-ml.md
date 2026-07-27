# Expert — audit-transformer-ml (attention kernel, masking, positional)

Returned 2026-07-27 ~09:00 CDT. Verbatim findings; NOT yet verified.
Bands: **no critical**, 2 high, 3 medium, 2 low.

> HARNESS NOTE: this agent's run carried "safety classifier was unavailable" — its output warrants
> the same source-level verification as every other finding, no more and no less. Its finding #2
> INDEPENDENTLY REPRODUCES the numerical-analyst's #1 (autocast drops emission) with a different
> probe, which is meaningful corroboration.

## Verified clean under this lens (executed on a tiny model K=4, n_heads=2, N=4)

**No critical findings** — no future-mass leak, no post-softmax masking, no wrong softmax axis, no
head/block-axis mixup. `log_prior` is added to the logits *before* `torch.softmax(..., dim=-1)`
(free_energy.py:329-332); beta rows sum to 1.0000; max mass strictly above the diagonal is exactly
`0.0`; `tau = 1.4142135623730951 == kappa*sqrt(d_head)` with the scaling inside the exponent. The
per-irrep-block beta collapses correctly to a single headless `(N,N)` beta at `len(irrep_dims)==1`
(free_energy.py:204-205). The RoPE `right` insertion yields the relative rotation
`U_i R(theta_j-theta_i) U_j^{-1}` (transport.py:367-401). T5 bucketing matches the HuggingFace
reference including the `bidirectional` half-range split and the `max_exact` boundary
(attention_prior.py:287-307).

---

### 1. Attention log-prior is floored at -27.6 nats by the gamma fold, truncating ALiBi's long-range decay
**Location:** vfe3/model/model.py:2396-2399
**Severity:** high
**Evidence:** The fold builds the effective belief-channel prior in probability space then takes a
**clamped** log:
```python
pi  = pi / pi.sum(dim=-1, keepdim=True).clamp(min=log_eps)    # normalize on active support
out = torch.log(pi.clamp(min=log_eps))                        # (B, [H,] N, N)
```
with `log_eps: float = 1e-12` and no caller overriding it (`_effective_beta_log_prior`,
model.py:2449-2454). This is exactly the failure mode `free_energy.py:456-460` already removed on
the F side ("the old `torch.log(softmax(...).clamp(min=log_eps))` floored a finite deep-tail entry
at ~-27.6 nats"); the fold reintroduces it upstream, in the prior itself. Executed at the live
config's shapes (`N=128`, `n_heads=7`, `alibi_slope=1.0`,
`beta/gamma_attention_prior='causal_alibi_noself'`, `gamma_prior_weight=0.5`), with `gamma` taken as
the *best case* (identical positional shape, so the distortion is purely the floor):
```
entries on the causal support: 56903
entries whose log-prior is FLOORED (>1e-6 nats off): 2954
max nats of distortion introduced by the 1e-12 floor: 36.301727294921875
  head 0: floored=  253/8129  max_nats=5.378
  head 4: floored= 2701/8129  max_nats=36.302   (Press slope 0.5)
```
The Press-slope-0.5 head's ALiBi bias reaches `-0.5*127 = -63.5`; the floor replaces it with
`-27.6`, so 33% of that head's causal entries carry a flat prior instead of a graded one. Against an
energy term spanning `kl_max/tau = 8*210/sqrt(30) ~ 306` nats, a 36-nat prior shift is decisive.
`gamma_as_beta_prior=True`, `lambda_gamma=0.75`, `gamma_prior_weight=0.5` are all set in the live
`train_vfe3.py`. Contrast Press et al. 2022 section 3, where the bias is linear in distance with no
floor.
**Fix:** Compose the mixture in log space (`torch.logaddexp(log(1-w)+log_softmax(log_prior),
log(w)+log(gamma))`) or drop the `clamp` and rely on the `-inf`/support masking already present.

### 2. `mm_exact_update`'s autocast re-entry silently drops the emission block
**Location:** vfe3/gradients/kernels.py:573-594
**Severity:** high
**Evidence:** *(INDEPENDENT CONFIRMATION of the numerical-analyst's finding #1, different probe.)*
The fp32 island recurses with an explicit argument list omitting both `emission` and
`emission_weight`, so the nested call falls back to `emission=None, emission_weight=0.0` and the
Bohning block at lines 692-701 never runs. The sibling `belief_gradients` island (lines 405-433)
forwards every argument, so the asymmetry is local to this seam. Executed (`N=3, K=4,
irrep_dims=[2,2]`, causal `log_prior`):
```
no-amp: ||mu*(emission on) - mu*(emission off)|| = 0.47322821617126465
amp   : ||mu*(emission on) - mu*(emission off)|| = 0.0
amp on == amp off exactly? True
```
Both trigger conditions are already set in the live `train_vfe3.py` (`amp_dtype='bf16'` line 392,
`e_step_update="mm_exact"` line 408); only `emission_mode` defaulting to `'off'` keeps it inert
today.
**Fix:** Forward `emission=emission` and `emission_weight=emission_weight` (tuple cast to float) in
the autocast recursion.

### 3. `collect_beta_channel_decomposition` ablates the model-channel gamma along with beta
**Location:** vfe3/run_artifacts.py:3583-3595
**Severity:** medium
**Evidence:** The patch list includes the definition module itself:
```python
    original = free_energy_module.attention_weights
    binders = [module for module in (free_energy_module, kernels_module, e_step_module)
               if getattr(module, "attention_weights", None) is original]
```
while `model.py:2377` resolves its binding at call time inside the gamma fold
(`from vfe3.free_energy import attention_weights`, then line 2383
`gamma = attention_weights(e_s, tau=gamma_tau, log_prior=gamma_log_prior)`), so the gamma softmax is
intercepted too. Executed on a tiny model with `gamma_as_beta_prior=True, lambda_gamma=1.0,
gamma_prior_weight=0.5`:
```
binders patched: ['vfe3.free_energy', 'vfe3.gradients.kernels', 'vfe3.inference.e_step']
total attention_weights calls intercepted: 2
of which reached through the GAMMA model-channel path: 1
```
Under the `no_energy` arm the gamma posterior collapses to `softmax(gamma_log_prior)`, and under
`no_prior` gamma's own positional bias is flattened, so `delta_ce.content_channel` /
`delta_ce.positional_channel` mix the belief and model channels. The live config has
`gamma_as_beta_prior=True` and `lambda_gamma=0.75`.
**Fix:** Route the interception through a flag the belief callers set (or patch only
`kernels_module`/`e_step_module` and give `_fold_gamma_prior` a module-level binding).

### 4. `attn_entropy_min` and `attn_entropy_collapsed_heads` are structurally constant under every causal prior
**Location:** vfe3/model/model.py:3145-3148
**Severity:** medium
**Evidence:**
```python
        ent_rows = metrics.attention_entropy_rows(beta)             # (N,) single head or (H, N) multi-head
        head_min = ent_rows.min(dim=-1).values if ent_rows.dim() >= 2 else ent_rows.min().reshape(1)
        d["attn_entropy_min"]             = float(head_min.min())
        d["attn_entropy_collapsed_heads"] = float((head_min < _LOG2).float().sum())
```
Query row 0 has exactly one allowed key under `causal`/`causal_noself`/`causal_alibi*`/
`causal_windowed`, so `beta[.., 0, :]` is one-hot and `H_0 = 0` for every head, layer, seed and
parameter value. The minimum over query rows can therefore never exceed the eps artifact. Executed
(`K=4, n_heads=2, N=4`, two seeds):
```
causal                 seed=0  attn_entropy_min=8.28931e-11  collapsed_heads=2.0  attn_entropy=0.7945
causal                 seed=7  attn_entropy_min=8.28931e-11  collapsed_heads=2.0
causal_noself          seed=0  attn_entropy_min=8.28931e-11  collapsed_heads=2.0
causal_alibi_noself    seed=0  attn_entropy_min=8.28931e-11  collapsed_heads=2.0
uniform                seed=0  attn_entropy_min=1.38629      collapsed_heads=0.0
```
Both are logged to metrics.csv (train.py:921-922, 1028) and plotted (viz/figures.py:1639-1640) as a
head-collapse signal. Same tautology family as the already-reported `estep_grad_norm_sigma=0.0`,
different metric.
**Fix:** Restrict the min/collapse reduction to query rows whose causal active set exceeds one key
(e.g. `i >= 1`), or normalize each row's entropy by `log(active_set_size_i)`.

### 5. `pos_phi='frozen'` injects position into a single generator axis, leaving H-1 heads position-blind
**Location:** vfe3/model/positional_phi.py:97-100
**Severity:** medium
**Evidence:**
```python
    r"""Parameter-free Lie-algebra ALiBi: pos_phi_i = (i * scale) on one generator axis."""
    coords = torch.zeros(n, n_gen, device=device, dtype=dtype)
    coords[:, frozen_axis] = torch.arange(n, device=device, dtype=dtype) * scale
```
with `frozen_axis` fixed at 0 (no config knob threads it; `_apply_pos_phi`/`_pos_phi_right` at
model.py:731-753 never pass it). Under `block_glk` the generators partition per head
(generators.py:106-113, `G[gen_offset + idx, start + i, start + j] = 1.0` with
`gen_offset = h * d_head**2`), so generator 0 is head 0's `E_00`, entirely inside head 0's diagonal
block, and every BCH bracket with it stays in that block. Executed (`K=4, n_heads=2, n_gen=8`,
`compose_mode='bch'`, order 4):
```
gen 0 nonzero at [[0, 0]]   gen 4 nonzero at [[2, 2]]
per-coordinate |delta| after frozen pos_phi:
tensor([1.5009e+00, 4.8019e-02, 1.9552e-01, 1.3209e-03, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00])
```
Head 1's four coordinates are exactly unchanged. Because the pair energy is computed per irrep
block (free_energy.py:204-235), heads 1..H-1 receive zero positional content; at the live shape
(`n_heads=7`, `embed_dim=210`) the positional signal reaches one coordinate out of 210. Does not
match the per-head positional coverage ALiBi provides (Press et al. 2022 section 3 gives every head
its own slope) nor gauge-RoPE's, which rotates pairs inside *every* block (geometry/rope.py:81-90).
**Fix:** Spread the frozen coordinates across one generator per irrep block (e.g. the
block-diagonal `E_00` of each head, scaled by a per-head slope) instead of a single global axis.

### 6. `viz/extract.py` recomputes beta at the base temperature, ignoring `query_adaptive_tau`
**Location:** vfe3/viz/extract.py:992-995 and 1097-1100
**Severity:** low
**Evidence:** Both seams call
```python
    beta = attention_weights(
        energy,
        tau=attention_tau(model.effective_kappa_beta(out.mu.device), model.group.irrep_dims),
        log_prior=log_prior)
```
i.e. the base `attention_tau` only, while every sibling replay routes through `_beta_tau` —
model.py:2809-2812 (`diagnostics`), 2568-2572 (`_attention_map_for_belief`), 3261 and 3304
(`attention_maps`) — which applies `query_adaptive_tau(sigma, tau, irrep_dims, c=cfg.query_tau_c)`
(model.py:2468-2472). `converged_state`'s docstring claims it "Mirrors `VFEModel.diagnostics`
EXACTLY (same active config)". Under `query_adaptive_tau=True` the beta these two produce is a
different softmax from the one the forward ran, and it feeds `nan_beta`, the causal panel
(`metrics.causal_sanity`), and the attention-structure figures.
**Fix:** Replace both with `model._beta_tau(out.sigma, out.mu, attention_tau(...))`.

### 7. Diagnostic and figure replays omit the emission term the forward passes
**Location:** vfe3/model/model.py:2740-2751, 3290-3304; vfe3/viz/extract.py:526-534
**Severity:** low
**Evidence:** `forward_beliefs` builds and threads the Bohning pair (model.py:1126-1130):
```python
            emission = self._emission_terms(token_ids, beliefs.mu)
            out = vfe_stack(
                beliefs, beliefs.mu, beliefs.sigma, self.group, self.cfg,
                emission=emission,
```
The three replay seams construct their `vfe_stack` / `vfe_block` calls with no `emission=` argument
at all, so under `emission_mode != 'off'` with `emission_weight != 0` their E-step descends a
different objective than the trained forward. Default `emission_mode='off'` keeps it inert today.
**Fix:** Pass `emission=self._emission_terms(token_ids, belief.mu)` (and
`model._emission_terms(...)` in `extract.py`) in each replay.
