# C9 Implementation Report

## Scope

Implemented overflow-safe Regime-II soft caps, precision routing for large exact skew exponentials, and a log-space Laplace Renyi route for convergent orders greater than one. No defaults, configuration values, or pure exact routes changed.

## TDD evidence

Eight focused tests were added before production edits. The RED run recorded six failures and two passing controls. All three overflow-scale Regime-II cap routes collapsed their off-diagonal operators to identity, the large skew exponential differed from the float64 reference, the large-separation Laplace term returned infinity, and a divergent blended coordinate left a NaN gradient. The small-skew float32 identity control and moderate numerical-quadrature oracle already passed.

After the production patch, the same eight nodes passed with zero failures and zero errors.

## Implementation

`_soft_cap_frobenius` evaluates the exact smooth-cap norm algebra in a differentiable float64 island before casting the capped matrix back to its public dtype. The bilinear, covariant, and shared direct-link Regime-II builders now use this helper.

`stable_matrix_exp_pair` retains its dimension rule but additionally enters float64 for a skew-symmetric matrix whose exponentiated norm reaches the existing norm threshold. Small skew matrices remain on the original float32 path exactly.

For Laplace Renyi order greater than one, the convergent integral is split into its three positive spatial intervals and combined with `torch.logaddexp`. The middle divided difference uses a stable log-sinhc representation. Non-positive tail blends are replaced with safe constants before branch algebra and then mapped to the existing NaN-to-`kl_max` policy.

## Verification

- RED nodes: 8 tests, 6 failures, 0 errors, 2 passing controls.
- GREEN nodes: 8 tests, 0 failures, 0 errors.
- Curated geometry, Regime-II bilinear/covariant/direct-link, transport, tier-12 transport, and Laplace suites: 153 collected, 152 passed, 1 skipped, 0 failures, 0 errors.
- The single skip is the existing CUDA-only Laplace parity test on the CPU test run.

## Self-review

The Laplace expression was re-derived interval by interval, including its normalization and convergence condition. The cap helper preserves the original smooth map and gradient path while preventing float32 pre-square overflow. The skew precision condition is restricted to skew-symmetric inputs, so non-skew dimension-mode behavior is unchanged. `git diff --check` reported no whitespace errors. No remaining C9 concern was identified.
