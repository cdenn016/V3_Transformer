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
