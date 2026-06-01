# Per-coordinate self-coupling alpha^(k) — design

Date: 2026-06-01
Status: approved, pre-implementation
Branch: vfe3-artifacts-priorbank-2026-05-31 (committed directly, per user choice)

## Motivation

The self-coupling registry in `vfe3/alpha_i.py` advertises three forms: `constant`,
`state_dependent` (one alpha per token from the summed self-divergence), and
`state_dependent_per_coord` (a separate alpha per coordinate). The third is a stub. It
is registered and selectable, but the pipeline feeds it the summed per-position
self-divergence `D(q_i||p_i)` of shape `(..., N)`, so it broadcasts to a single alpha per
token and reproduces `state_dependent` exactly. The form emits a `RuntimeWarning` saying
so on every call, and that warning is what surfaced from `train_vfe3.py` (which selects
this mode) at `vfe3/alpha_i.py:129` and `vfe3/model/model.py:245`.

This design realises the per-coordinate path so the mode delivers what it advertises: a
self-term `sum_k alpha^(k) D^(k)` with `alpha^(k)* = c0/(b0 + D^(k))`, each coordinate's
coupling responding to that coordinate's own divergence from its prior. The single-alpha
forms remain the default pure path, unchanged; per-coordinate alpha is a generalised,
per-coordinate-reweighted self-coupling layered behind an opt-in mode.

## What already supports per-coordinate shape

Three load-bearing facts make this mostly a wiring change rather than new mathematics.

First, `free_energy` (`vfe3/free_energy.py:163-193`) already documents `self_div` and
`alpha` as `(..., N)` or `(..., N, K)` and forms `self_term = alpha * self_div + alpha_reg`
then sums over every trailing axis. Fed per-coordinate tensors it computes
`sum_k alpha^(k) D^(k)` with no change.

Second, `free_energy_terms` (`vfe3/metrics.py:116`) likewise reduces `(alpha * self_div).sum()`
over all axes, so the diagnostics decomposition is per-coordinate-correct once fed
per-coordinate inputs.

Third, the analytic kernel `_diag_kl_filtering_kernel` (`vfe3/gradients/kernels.py:60-102`)
already types `alpha_coef` as `(N,1) or (N,K)` and multiplies it into the self-term, so a
per-coordinate coefficient broadcasts coordinate-wise without a signature change.

The autograd oracle needs no structural change either. `belief_gradients_autograd`
(`vfe3/gradients/oracle.py:68-77`) feeds one `sd` into both `self_coupling_alpha` and
`free_energy`; if `sd` is `(N,K)` then `alpha`, `reg`, and the assembled F are all
per-coordinate and autograd differentiates `sum_k alpha^(k) D^(k)` correctly.

## Mathematical scope and correctness

The per-coordinate self-divergence is the diagonal Gaussian Renyi/KL with the coordinate
sum left undone:

    D^(k)(q_i||p_i) = 1/2 ( s_k/t_k + (mu_p^k - mu_q^k)^2/t_k - 1 + log(t_k/s_k) )    (KL, alpha_div=1)

and the Renyi-blend analog for alpha_div != 1. The `-K` of the summed form becomes `-1`
per coordinate, and `sum_k D^(k)` recovers the pre-clamp summed divergence.

This decomposition exists only for the diagonal family. Full-covariance KL couples
coordinates through the trace term `tr(Sigma_t^{-1} Sigma_q)` and the log-determinants,
which do not split into a coordinate sum, so there is no per-coordinate divergence to feed
a per-coordinate alpha. It also presumes the Renyi functional (the only registered one).
Both restrictions are enforced, at config construction and inside the per-coordinate
divergence, by raising rather than silently summing.

The envelope cancellation that pins the analytic kernel to the autograd oracle holds
coordinate-wise. At the stationary `alpha^(k)* = c0/(b0 + D^(k))`, with the regulariser
`R(alpha^(k)) = b0 alpha^(k) - c0 log alpha^(k)` present in F, the derivative of
`alpha^(k)*(D^(k)) D^(k) + R(alpha^(k)*(D^(k)))` with respect to the belief is
`alpha^(k)* dD^(k)/d(belief)`; the product-rule corrections cancel independently for each
k because each `D^(k)`, `alpha^(k)`, `R^(k)` depends on coordinate k alone. So the kernel
coefficient remains `alpha^(k)*` with no correction, exactly as in the per-position case.

One genuinely new correctness detail: the kernel's self-term saturation mask. The summed
path zeros the self-term where the clamped divergence saturates, `m_i = 1[0 < D_i < kl_max]`,
because the oracle differentiates through `safe_kl_clamp` whose gradient is zero outside
`(0, kl_max)`. The per-coordinate path applies `safe_kl_clamp` per coordinate, so the mask
must be per-coordinate `m_i^(k) = 1[0 < D_i^(k) < kl_max]` to stay exactly equal to the
oracle. This mask is invisible in the unsaturated regime (it is identically 1), so it is
the one piece of new logic that demands a test in the saturated regime.

## Modularity: alpha_mode declares its divergence-reduction need

Routing must not hardcode `if alpha_mode == "state_dependent_per_coord"` at the consumer
sites, so that future alpha forms slot in by registration alone. Each registered alpha form
declares whether it consumes a per-coordinate (unsummed) divergence:

- `register_alpha(name, *, per_coord=False)` stores the flag alongside the callable.
- `alpha_is_per_coord(mode) -> bool` queries it.
- `state_dependent_per_coord` registers with `per_coord=True`; the others default `False`.

A single routing function, `self_divergence_for_alpha(..., *, alpha_mode, ...)`, returns the
per-coordinate divergence `(..., N, K)` when the selected form declares `per_coord=True`,
and the summed divergence `(..., N)` otherwise. Every consumer that turns `D` into alpha
calls this one router. A future per-coordinate alpha variant is added by registering it with
`per_coord=True`; no consumer is edited.

## Components and changes

`vfe3/alpha_i.py`
  Extend `register_alpha` with the `per_coord` keyword and a parallel flag store; add
  `alpha_is_per_coord`. Mark `state_dependent_per_coord` with `per_coord=True` and delete
  its degradation `warnings.warn` (no longer accurate once it receives per-coordinate D).

`vfe3/divergence.py`
  Add `gaussian_diagonal_renyi_per_coord(mu_q, sigma_q, mu_t, sigma_t, *, alpha, kl_max, eps)
  -> (..., K)`: the diagonal Renyi/KL terms without the coordinate sum, with `safe_kl_clamp`
  applied per coordinate. Standalone; the existing summed kernel is left untouched so its
  clamp-the-sum semantics and golden tests are preserved.

`vfe3/free_energy.py`
  Add `self_divergence_per_coord(...) -> (..., N, K)` dispatching on `family` and
  `divergence_family`, raising for anything but `gaussian_diagonal` + `renyi`. Add
  `self_divergence_for_alpha(..., *, alpha_mode, ...)` the router.

`vfe3/gradients/oracle.py`
  Replace `sd = self_divergence(...)` with `sd = self_divergence_for_alpha(..., alpha_mode=alpha_mode, ...)`.

`vfe3/gradients/kernels.py`
  Compute `sd` via the router; keep the legacy `(N,1)` broadcast for per-position forms but
  pass the `(N,K)` coefficient through unchanged for per-coordinate forms (drop the
  unconditional `.unsqueeze(-1)`). In `_diag_kl_filtering_kernel`, select a per-coordinate
  saturation mask when `alpha_coef.shape[-1] > 1`, via a new `_raw_diag_kl_per_coord`. Note
  in the docstring that the shape-driven branch is correct at K=1 (the two coincide).

`vfe3/inference/e_step.py`
  `free_energy_value` computes `sd` via the router so the per-coordinate self-term enters F.

`vfe3/model/model.py`
  `diagnostics` computes `self_div` via the router so the reported alpha and free-energy
  terms match the actual forward and no warning fires.

`vfe3/config.py`
  In `__post_init__`, after the existing `alpha_mode` validation, reject
  `alpha_is_per_coord(alpha_mode)` together with a non-diagonal family, matching the existing
  `tied_block_glk` / `killing_per_block` cross-validation pattern.

## Decisions made explicit

Per-coordinate `kl_max` clamps each coordinate at `kl_max`, so a token's total
self-divergence can reach `K * kl_max` — a different regularisation scale than the
summed-then-clamped path. This is the natural per-coordinate analog and is the chosen
behaviour; it is tuned through `kl_max`. Stated here because it would otherwise read as a
bug to a future reader.

The mode is diagonal-family and Renyi-functional only, by mathematical necessity, enforced
at config construction and inside the per-coordinate divergence.

The default pure path is unchanged. `constant` and `state_dependent` keep the single-alpha
canonical self-coupling; per-coordinate alpha is an opt-in generalisation.

## Tests (test-driven)

1. Saturated-coordinate kernel-equals-oracle: a belief where at least one coordinate's
   `D^(k)` exceeds `kl_max` while others do not; assert the analytic kernel gradient equals
   the filtering autograd oracle. This is the only test that exercises the new per-coordinate
   mask.
2. Per-coordinate alpha is `(N,K)` and varies across k: guards against silent
   re-degradation now that the warning is removed.
3. Config rejects `state_dependent_per_coord` with `gaussian_full`.
4. `self_divergence_per_coord` raises on full covariance and on a non-Renyi functional.
5. Regression: `constant` and `state_dependent` paths unchanged (existing golden and
   gradient tests continue to pass).
