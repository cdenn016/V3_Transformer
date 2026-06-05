# 2026-06-05 — M-step self-coupling now carries the E-step alpha_i (alpha_hat unified)

## What prompted this

A question about Algorithm 1's M-step loss term `\hat\alpha sum_i D_KL(q_i*||p_i)`
(`GL(K)_attention.tex:2111`): is `alpha_hat` the same coefficient as the E-step
self-coupling `alpha_i` (`GL(K)_attention.tex:2069`), in particular the
`state_dependent_per_coord` per-coordinate `alpha_i^(k)* = c0^(k)/(b0^(k)+D^(k))`?

It was not. Three distinct objects shared the "alpha" name at one call site
(`model.py`): `cfg.alpha_div` (the Rényi functional parameter), `cfg.alpha_mode`
(the E-step self-coupling form), and `cfg.mstep_self_coupling_weight` (`alpha_hat`,
a flat scalar). The old M-step term computed the bare divergence and scaled it by
the flat `alpha_hat`:

```python
sc = self_divergence_for_alpha(..., alpha_mode=cfg.alpha_mode).mean()
loss = loss + cfg.mstep_self_coupling_weight * sc
```

`self_divergence_for_alpha` returns `D`, not `alpha·D`. So even under
`alpha_mode='state_dependent_per_coord'` (which both entry points set —
`train_vfe3.py:108`, `ablation.py:126`), passing `alpha_mode` there only changed the
SHAPE of `D` (per-coord `(B,N,K)` vs summed `(B,N)`, hence the `.mean()`
normalization), and applied NO per-coordinate `alpha^(k)` weighting. The E-step used
per-coord `alpha_i^(k)` while the M-step used a flat scalar — a live inconsistency
once `mstep_self_coupling_weight > 0` (dormant only because the term is default-off).

## The fix (theory)

Under the single-free-energy variational-EM reading (one `F`, E-step over beliefs,
M-step over parameters), the M-step self-term IS the `F` self-coupling term at the
converged belief, so it must carry the same `alpha_i`. By the alpha-envelope —
`alpha* = c0/(b0+D)` is the stationary point of `alpha*D + R(alpha)`, so
`d/dalpha[alpha*D + R] = 0` there — the M-step gradient w.r.t. the priors is
`alpha_i*.detach() · dD/dtheta` (and `R` drops). This detach is EXACT for all three
closed-form forms (`constant`, `state_dependent`, `state_dependent_per_coord`); it
holds independent of E-step convergence, so `n_e_steps=1` does not weaken it.
`mstep_self_coupling_weight` (= `alpha_hat`) is demoted to an overall scale.

Supporting consistency argument: `alpha_phi` (= `mass_phi`) is already shared across
the E-step (`2106`) and M-step (`2111`) in both manuscript and code; the
self-coupling alpha should be shared the same way.

## The change (code)

`vfe3/model/model.py` (M-step self-coupling block, ~`model.py:421`): compute `D` via
`self_divergence_for_alpha`, weight it by `self_coupling_alpha(D, mode=cfg.alpha_mode,
value=cfg.alpha, b0=cfg.b0, c0=cfg.c0, log_alpha=...)`'s coefficient (detached),
sum over the coordinate axis when `alpha_is_per_coord(cfg.alpha_mode)`, then mean
over `(B,N)`. This mirrors the diagnostics reduction exactly
(`model.py` diagnostics `self_div`/`alpha` → `metrics.free_energy_terms:124`,
`metrics.self_coupling_profile:661`).

- For learnable alpha (NN-exception), the M-step alpha is detached; `log_alpha` still
  trains through the E-step path as before (documented in the code comment).
- `vfe3/config.py:140`: comment updated — `alpha_hat` is now the overall scale on
  `sum_i alpha_i D(q_i*||p_i)`, with `alpha_i` the E-step form.

## Backward compatibility

At the default `alpha_mode='constant'`, `alpha=1.0`: `alpha_i ≡ 1`, so the term is
byte-identical to the previous mean-`D` form. The four existing tests in
`tests/test_mstep_self_coupling.py` (which use the constant default) pass unchanged.
The semantics of the knob change only under the state-dependent forms.

## Tests

Added `tests/test_mstep_self_coupling.py::test_per_coord_alpha_weighting` plus oracle
`_converged_self_coupling_per_coord` (recomputes `alpha^(k)* = c0/(b0+D^(k))` from
the closed form, independent of `self_coupling_alpha`): under
`state_dependent_per_coord`, `loss == ce + w * mean_i sum_k alpha_i^(k)* D_i^(k)`.
Confirmed failing against the old flat-scalar code, passing after the fix.

Full suite: 589 passed, 1 xpassed, 0 failures, 0 errors (junitxml-read).
