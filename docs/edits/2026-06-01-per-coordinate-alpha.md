# 2026-06-01 — Per-coordinate self-coupling alpha^(k)

Branch: vfe3-artifacts-priorbank-2026-05-31 (committed directly).
Design spec: docs/superpowers/specs/2026-06-01-per-coordinate-alpha-design.md.

## Motivation

Investigating warnings from `train_vfe3.py` surfaced that `alpha_mode='state_dependent_per_coord'`
was a stub: the registry advertised a per-coordinate self-coupling but the pipeline fed it the
summed per-position self-divergence, so it broadcast to one alpha per token (identical to
`state_dependent`) and emitted a `RuntimeWarning` on every call (`vfe3/alpha_i.py:129`,
`vfe3/model/model.py:245`). This change realises the per-coordinate path so the mode delivers a
self-term `sum_k alpha^(k) D^(k)` with `alpha^(k)* = c0/(b0 + D^(k))`. The Triton
`cuobjdump.exe` / `nvdisasm.exe` warnings in the same console output are unrelated environment
notices (missing CUDA-toolkit disassembly binaries on PATH) and were not code-addressed.

## What changed

`vfe3/alpha_i.py`
  `register_alpha(name, *, per_coord=False)` now stores a per-form flag; `alpha_is_per_coord(mode)`
  queries it; `state_dependent_per_coord` is registered with `per_coord=True`. The degradation
  `warnings.warn` (and the now-unused `import warnings`) were removed — the form receives the
  per-coordinate divergence and no longer degrades.

`vfe3/divergence.py`
  New `gaussian_diagonal_renyi_per_coord(...) -> (..., K)`: the diagonal Renyi/KL terms left
  unsummed, each clamped independently by `safe_kl_clamp`. The summed kernel is untouched (its
  clamp-the-sum semantics and golden tests are preserved).

`vfe3/free_energy.py`
  New `self_divergence_per_coord(...)` (dispatches on family and functional; raises for anything
  but diagonal + renyi) and `self_divergence_for_alpha(..., *, alpha_mode, ...)` — the single
  routing seam that returns `(..., N, K)` per-coordinate when the selected alpha form declares
  `per_coord=True`, else the summed `(..., N)`.

`vfe3/gradients/oracle.py`, `vfe3/gradients/kernels.py`, `vfe3/inference/e_step.py`,
`vfe3/model/model.py`
  All four alpha consumers now obtain the self-divergence through `self_divergence_for_alpha`.
  In the analytic kernel the unconditional `.unsqueeze(-1)` on the coefficient is gated to the
  per-position case (per-coordinate `sd` is already `(N,K)`), and `_diag_kl_filtering_kernel`
  selects a per-coordinate saturation mask (new `_raw_diag_kl_per_coord`) when `alpha_coef` is
  `(N,K)`, so a saturated coordinate is gated without killing its unsaturated neighbours.

`vfe3/config.py`
  `__post_init__` rejects a per-coordinate alpha form together with a non-diagonal family (the
  per-coordinate divergence does not exist for full covariance), matching the existing
  `tied_block_glk` / `killing_per_block` cross-validation pattern.

## Modularity

No consumer hardcodes the mode name. Each alpha form declares its divergence-reduction need at
registration (`per_coord=`), and `self_divergence_for_alpha` reads that declaration. A future
per-coordinate alpha variant slots in by registering with `per_coord=True`; no call site is edited.
The default pure path (`constant`, `state_dependent`) is unchanged.

## Tests

Six new tests (test-driven, all watched RED then GREEN):
  - `test_alpha_is_per_coord_declares_reduction_need`, `test_register_alpha_per_coord_flag_is_modular`
    (registry declaration, test_alpha_i.py)
  - `test_self_divergence_for_alpha_routes_by_declared_reduction`,
    `test_self_divergence_per_coord_requires_diagonal_renyi` (router + guards, test_free_energy.py)
  - `test_per_coord_alpha_requires_diagonal_family` (config guard, test_config.py)
  - `test_per_coord_alpha_saturation_mask_is_per_coordinate` (the critical gate, test_gradients_kernels.py):
    a mixed-saturation belief (coordinate 0 saturated, coordinate 1 not) pins the analytic kernel
    equal to the filtering autograd oracle and proves coordinate 1's restoring force survives — the
    only regime in which the new per-coordinate mask is observable.

Full suite after the change: `tests=254 failures=0 errors=0` (read from junitxml; 248 baseline + 6
new). End-to-end smoke (forward + diagnostics under `state_dependent_per_coord`, `RuntimeWarning`
promoted to error) passes, confirming the degradation warning is gone and all four consumers route
correctly.

## Mathematical verification

- Per-coordinate decomposition: the diagonal Gaussian KL/Renyi sums over coordinates, so the
  unsummed per-coordinate term is well-defined and `sum_k D^(k)` recovers the pre-clamp summed
  divergence (the `-K` of the summed form becomes `-1` per coordinate). Verified by
  `test_self_divergence_for_alpha_routes_by_declared_reduction` (`per.sum(-1) == summed`).
- Full covariance: KL couples coordinates through `tr(Sigma_t^{-1} Sigma_q)` and the
  log-determinants and does not decompose; the per-coordinate path raises rather than summing
  the wrong thing. Verified by guard test and config cross-validation.
- Envelope cancellation holds coordinate-wise: at `alpha^(k)* = c0/(b0 + D^(k))` with `R^(k)`
  present in F, `d/d(belief)[alpha^(k)* D^(k) + R^(k)] = alpha^(k)* dD^(k)/d(belief)` independently
  per k, because each `D^(k)`, `alpha^(k)`, `R^(k)` depends on coordinate k alone under the
  diagonal family. The analytic kernel therefore matches the autograd oracle with no product-rule
  correction. Rigorously verified by `test_per_coord_alpha_saturation_mask_is_per_coordinate`
  (kernel == oracle for both a saturated and an unsaturated coordinate).
- Per-coordinate clamp: each coordinate's `D^(k)` is clamped at `kl_max`, so a token's total can
  reach `K * kl_max` — a deliberate per-coordinate regularisation scale (design decision), not a
  bug.
