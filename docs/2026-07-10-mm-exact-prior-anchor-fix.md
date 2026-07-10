# mm_exact drops the prior anchor at initialization — diagnosis and fix spec (2026-07-10)

Status: **FIXED** on branch `fix/mm-exact-prior-anchor-20260710` (2026-07-10, Cowork session).
The sections below are the original diagnosis; the "Fix applied" section at the end records
what was changed and the test evidence. Line numbers are pre-fix.

## Summary

`e_step_update='mm_exact'` produces degraded performance vs `'gradient'`, and preliminary
sweeps found the best mm_exact results at `lambda_beta = 0`. Both observations trace to one
defect: `mm_exact_update` reuses the gradient kernel's self-term saturation mask
`m_i = 1[0 < D(q_i||p_i) < kl_max]` as the prior PRECISION WEIGHT in the closed-form fusion,
and the model always enters the E-step with the belief anchored exactly to the prior
(`q0 == p`), where `D == 0.0` exactly and the mask zeroes the prior term for every token.

## Mechanism

1. `model.py:841` (`forward_beliefs`) calls `vfe_stack(beliefs, beliefs.mu, beliefs.sigma, ...)`
   — the initial belief and the prior are the same tensors, so at layer 1 iteration 1,
   `raw_self = KL(q||p) == 0.0` bit-exactly (with `s_e_step=True` both are the refined s1).
2. `kernels.py:465-471` (`mm_exact_update`) builds `a = self_mask * coef` with
   `self_mask = (raw_self > 0.0) & (raw_self < kl_max)`. At `raw_self == 0` the lower gate
   makes `a = 0`, so the fusion

       mu*_i = ( a_i mu_p/sp + sum_j w_ij mu_t/st ) / ( a_i/sp + sum_j w_ij/st )

   returns the pure attention consensus with zero prior anchoring. Under
   `beta_attention_prior="causal_alibi_noself"` the self edge is masked, so token i's own
   content is entirely absent from `mu*_i`. With `n_e_steps=1` and `mm_damping=1.0` that one
   pathological iteration is the whole E-step: token identity is exactly erased before decode.
3. In the GRADIENT kernel (`kernels.py:132-137`) the same mask is correct and must NOT be
   changed: it mirrors the zero derivative of `safe_kl_clamp` at the boundary, and the true
   self-gradient at q==p is zero anyway. The boundary case is measure-zero for gradients but
   measure-one for the mm fusion, because the initialization sits exactly on it.
4. The correct MM coefficient at D=0 is the alpha envelope value `alpha* = c0/(b0+0)`
   (= 1 under the current config), not 0 (`alpha_i.py:87-102`).
5. Secondary consequence of `a = 0`: `mu*` is built solely from DETACHED transported keys
   (filtering split, `kernels.py:451`) weighted by live beta, so the value path from the mean
   tables to the loss is severed for the whole E-step output; the tables learn only through
   the attention weights. Restoring `a > 0` also restores the live `mu_p -> mu* -> decode` path.

## Why the lambda_beta = 0 sweep result is an artifact

At `lambda_beta = 0` AND `a = 0`, the fused precision is 0 and the m12 degenerate guard
(`kernels.py:489-494`) keeps the belief at q0 — an identity E-step that decodes the refined
prior with full token identity intact. The sweep therefore compared "identity E-step" against
"E-step that replaces token identity with self-excluded neighbor consensus". It was not
evidence that the coupling term hurts.

## Verified evidence (fp64 repro, N=6, K=4, flat transport, state_dependent alpha)

- q0 == p exactly: `self_mask = 0` for every token; `max|mu* - pure_consensus| = 0.0`;
  `max|mu* - mu_p| = 1.35`.
- q0 = p + 1e-4 noise: `self_mask = 1`; `mu*` differs from pure consensus by 0.65 — the update
  is discontinuous exactly at the state the model always initializes in.
- lambda_beta = 0, q0 == p: `max|mu* - q0| = 0.0` (identity E-step via degenerate guard).
- Gradient route from the same point (`e_q_mu_lr = 0.9`): mean per-coordinate retention of the
  original mean ~0.16 (nonzero and live); mm_exact retention exactly 0.
- With the fix below: update is continuous (perturbation test agrees to 1e-7) and lands on the
  anchored fusion, `max|mu* - mu_p| = 0.70` (~halfway between prior and consensus at
  sigma_p ~ sigma_t, sum_j beta = 1) — more conservative than the lr=0.9 gradient step.

## The fix (one line, two occurrences, in mm_exact_update ONLY)

In `vfe3/gradients/kernels.py`, inside `mm_exact_update` (lines ~465-470), gate only the UPPER
saturation in both alpha branches:

```python
# per-position branch (~line 467):
self_mask = (raw_self < kl_max).to(mu.dtype).unsqueeze(-1)
# per-coordinate branch (~line 470):
self_mask = (raw_self < kl_max).to(mu.dtype)
```

Do NOT touch `_diag_kl_filtering_kernel` (the gradient kernel's mask stays
`(raw_self > 0.0) & (raw_self < kl_max)` — it is exactly oracle-matching there). Rationale for
keeping only the upper gate in the fusion: the upper clamp genuinely flattens the objective in
a neighborhood (weight 0 is the correct majorizer coefficient there); the lower boundary is
only reached at D == 0, where the correct weight is `alpha* = c0/b0`, and a slightly negative
fp `raw_self` gives `alpha* ~ c0/(b0+D) ~ 1`, which is also fine.

## Behavior changes to expect after the fix

- q0 == p, lambda_beta > 0: `mu*` becomes the anchored fusion instead of pure consensus.
- q0 == p, lambda_beta = 0: `a = alpha* > 0`, so the degenerate guard no longer fires;
  `mu* = mu_p`, `sigma* = sigma_p` — identical VALUE to before at initialization (q0 == p),
  but now reached through a live graph in `mu_p`.
- lambda_beta = 0 is value-identical everywhere, on and off initialization: off init the old
  mask was already 1 (D > 0), so the update snapped to the prior before and after the fix; at
  exact init the old degenerate guard returned q0 == p, the same numbers the anchored path now
  produces. (An earlier draft of this note claimed a q != p behavior change here; that was
  wrong.) The only VALUE change from the fix is at D == 0 with lambda_beta > 0: unanchored
  consensus -> anchored fusion, plus the restored live mu_p gradient path.

## Testing (CPU, K < 6 per repo policy)

1. Check existing mm_exact tests for goldens pinned at q0 == p — they would have pinned the
   buggy consensus values and will need re-pinning.
2. Add pinned tests (suggested, K=4, N=6 or smaller):
   - anchor-at-init: q0 == p, lambda_beta = 1 -> `mu*` depends on `mu_p` (not equal to the
     beta-weighted transported-neighbor average) and `a` used is `c0/(b0+0)`.
   - continuity: `mu*(q0 = p)` vs `mu*(q0 = p + 1e-7 noise)` agree to ~1e-6.
   - upper gate preserved: a row whose raw self-KL exceeds kl_max still gets `a = 0`.
   - grad-route regression: `_diag_kl_filtering_kernel` output at q0 == p unchanged
     (byte-identical) — confirms the gradient kernel was not touched.
3. Run the full suite with a machine-readable pass count (no extra `-q`; or
   `--junitxml=out.xml`), per the tooling discipline in CLAUDE.md.

## After the fix: re-run the ablation

Re-run the mm_exact x lambda_beta sweep. Prediction: lambda_beta > 0 stops being dominated by
lambda_beta = 0, and mm_exact becomes competitive with the gradient route (it is now the exact
anchored coordinate minimizer). `mm_damping ~ 0.5` is a cheap robustness check alongside.

## Footnotes

- Latent, not currently operative (config uses `e_step_gradient='unroll'`):
  `mm_damping=1.0` + `straight_through` fully severs the belief chain in the mm branch
  (`e_step.py:691-712` — `mu = mu_star.detach()` with no (1-eta) term), unlike the gradient
  path's straight-through which keeps the additive identity Jacobian. Worth a config-time
  warning eventually; out of scope for this fix.
- Separate finding from the same session (also unfixed): `z_loss_weight=True` passes config
  validation as bool and runs the z-loss at coefficient 1.0 (~10^4 x the standard 1e-4).
  Optional hardening: reject `isinstance(z_loss_weight, bool)` in `__post_init__` and warn
  above ~0.01. Use a float (e.g. 1e-4) when sweeping z-loss.

## Fix applied (2026-07-10)

Branch `fix/mm-exact-prior-anchor-20260710` (off origin/main 2f82f7f). `vfe3/gradients/kernels.py`:
13 insertions, 3 deletions, all inside `mm_exact_update` — the two `self_mask` lines now gate
only the upper saturation (`raw_self < kl_max`), with a rationale comment block above the
branch and a docstring correction. The gradient kernel (`_diag_kl_filtering_kernel`) is
untouched, and `tests/test_mm_exact_prior_anchor.py::test_gradient_kernel_keeps_two_sided_mask`
pins that asymmetry by source inspection.

New tests (`tests/test_mm_exact_prior_anchor.py`, K=3, N=5, CPU-fast): prior anchored at exact
init (pinned against the hand-computed a=1 fusion, and pinned NOT equal to the unanchored
consensus), continuity at init (fp64, 1e-7 perturbation), upper gate still zeroes a saturated
self term, lambda_beta=0 returns the prior on and off init, gradient-kernel mask guard, and
gradient-kernel zero self-grad at saturation.

Test evidence (junitxml-read, per the tooling discipline): targeted subset
(test_mm_exact_prior_anchor + test_tier12_estep + test_gradients_kernels + test_belief_cache)
tests=47 failures=0 errors=0; post-restore confirm (test_mm_exact_prior_anchor +
test_tier12_estep) tests=25 failures=0 errors=0. Full suite tests=1759 failures=9 errors=0
skipped=12; the 9 failures (test_deterministic, test_fullcov_alpha_roadmap_2026_06_13,
test_phase0_forward_beliefs x3, test_run_artifacts, test_run_naming x3) were re-run with the
ORIGINAL origin/main kernels.py in place and fail identically — they are pre-existing in the
worktree's unrelated uncommitted WIP (model.py, prior_bank.py, run_artifacts.py et al., the
2026-07-09 audit follow-on), not caused by this fix. The pre-existing mm goldens in
test_tier12_estep (stationarity, monotone descent, saturated-row stays-put) all still pass —
the saturated-row test's setup (prior == belief, identity transport) now returns
mu* = mu_p = mu via the anchored path, the same values the degenerate guard produced.
