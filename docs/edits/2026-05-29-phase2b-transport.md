# Phase 2b Transport — 2026-05-29

## Files created

- `vfe3/geometry/transport.py` — new module; three public functions
- `tests/golden/test_transport_golden.py` — 6 golden + structural tests

## Files modified

- `tests/golden/conftest.py` — added `vfe2_transport` session fixture

## Changes

### `vfe3/geometry/transport.py`

Three functions ported from VFE_2.0 and adapted to the GaugeGroup API:

**`stable_matrix_exp_pair(matrix, *, max_norm, dim_threshold, skew_symmetric, only_forward)`**
Frobenius-norm clamp + float64 upcast before `torch.linalg.matrix_exp`. Keyword-only
stability knobs. Golden-equal to `transformer.core.gauge_utils.stable_matrix_exp_pair`.

**`compute_transport_operators(phi, group, *, gauge_mode)`**
phi/exp flat transport: `Omega_ij = exp(phi_i . G) exp(-phi_j . G)` in GL+(K).
Accepts a `GaugeGroup` (carries generators + `skew_symmetric` flag) rather than raw
generator tensor + a separate skew flag, matching the VFE_3.0 group-registry design.
`gauge_mode='trivial'` returns identity transport. Golden-equal to
`transformer.core.transport_ops.compute_transport_operators` (flat path,
`enforce_orthogonal=False`).

**`compute_transport_operators_direct(omega, *, gauge_mode, eps)`**
Direct-Omega flat transport: `Omega_ij = Omega_i @ Omega_j^{-1}` for general GL(K).
LU solve primary path; ridge-then-pinv fallbacks for near-singular inputs.
`gauge_mode='trivial'` returns identity. Golden-equal to
`transformer.core.transport_ops.compute_transport_operators_direct` (flat path).

### `tests/golden/conftest.py`

Added `vfe2_transport` fixture (session-scoped) that imports
`transformer.core.{gauge_utils, transport_ops}` from the sibling VFE_2.0 checkout,
skipping if unavailable.

## Test results

```
52 passed in 0.07s
```

All 6 new golden/structural tests pass; no regressions in the existing 46 tests.

## Commit

`7de4a8b feat(geometry): stable_matrix_exp_pair, golden-equal to 2.0`
(contains all three tasks — transport.py written in one pass with all three functions)

---

## Phase 2b Tasks 4-6 — 2026-05-29 (continuation)

### Task 4: belief action — `transport_mean` / `transport_covariance`

**`transport_mean(omega, mu)`** — einsum `bijkl,bjl->bijk`: gauge action
`mu_t[i,j] = Omega_ij @ mu_j`. Returns (B, N, N, K).

**`transport_covariance(omega, sigma, *, diagonal_out)`** — sandwich product
`Sigma_t[i,j] = Omega_ij Sigma_j Omega_ij^T`. Full input (B,N,K,K) → full
(B,N,N,K,K) via `bijkl,bjlm,bijnm->bijkn`. Diagonal input (B,N,K) → diagonal
approximation (B,N,N,K) via `bijkl,bijkl,bjl->bijk`, matching VFE_2.0 attention.py:270.
`diagonal_out=None` auto-detects by ndim.

Tests added in `tests/test_transport.py`:
- `test_transport_mean_identity_at_phi_zero` — phi=0 gives identity
- `test_transport_covariance_full_is_spd` — SPD preserved under sandwich
- `test_transport_covariance_diag_matches_full_diagonal` — diag approx = diagonal of full
- `test_transport_covariance_diag_matches_vfe2_formula` — matches explicit einsum reference

### Task 5: `omega_to_block_exp_pairs`

Slices a block-diagonal Omega (B,N,K,K) into per-block (block, block_inv) pairs, one
per entry in `irrep_dims`. Per-block inverse via `torch.linalg.solve`; ridge then pinv
fallbacks for near-singular blocks. Ported from VFE_2.0 `transport_ops.py:554-602`.

Golden test `test_omega_to_block_exp_pairs_matches_vfe2` added to
`tests/golden/test_transport_golden.py` — matches VFE_2.0 output at atol=1e-4.

### Task 6: property tests (equivariance + det<0 representability)

Two property tests added in `tests/test_transport.py`:

**`test_transported_kl_is_gauge_consistent`** — verifies KL gauge-equivariance: applying
a common SO(K) rotation h to both transported beliefs leaves KL(q || Omega @ k)
unchanged (at atol=1e-3). Passed without tolerance adjustment, confirming the
transport/divergence pipeline is consistent.

**`test_direct_omega_represents_reflection`** — verifies the direct-Omega path can
represent det < 0 elements: a reflection diag([-1,1,1,1]) is preserved through
`compute_transport_operators_direct`, and `det(omega_i[0,0]) < 0` holds.

### Test results (Tasks 4-6)

```
59 passed in 0.08s
```

All 59 tests pass (53 pre-existing + 6 new in test_transport.py + 1 new golden test).

### Commits

- `110fe06 feat(geometry): belief action transport_mean / transport_covariance (full + diag)`
- `eed0723 feat(geometry): omega_to_block_exp_pairs block slicing, golden-equal to 2.0`

Task 6 property tests were included in the `110fe06` commit (test file written in one
pass); no separate commit needed.
