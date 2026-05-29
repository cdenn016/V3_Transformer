# Phase 2b Transport — 2026-05-29

## Files created

- `vfe3/geometry/transport.py` — new module; seven public functions
  (`stable_matrix_exp_pair`, `compute_transport_operators`,
  `compute_transport_operators_direct`, `transport_mean`,
  `transport_covariance`, `omega_to_block_exp_pairs`)
- `tests/golden/test_transport_golden.py` — 7 golden + structural tests
- `tests/test_transport.py` — belief-action unit + equivariance/det<0 property tests

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

---

## Phase 2c Manifold Retractions + Fisher Preconditioner — 2026-05-29 (continuation)

### Files created

- `vfe3/geometry/retraction.py` — three public functions:
  `retract_spd_diagonal`, `retract_spd_full`, `natural_gradient`
- `tests/golden/test_retraction_golden.py` — 4 golden equivalence tests vs VFE_2.0
- `tests/test_retraction.py` — 4 property/formula tests

### Files modified

- `tests/golden/conftest.py` — added `vfe2_retract` session fixture importing
  `transformer.core.{vfe_utils, vfe_gradients}` from the sibling VFE_2.0 checkout

### Changes

**`retract_spd_diagonal(sigma_diag, delta_sigma, *, step_size, trust_region, eps, sigma_max)`**
Diagonal SPD retraction `sigma_new = sigma * exp(step_size * clamp(delta/sigma, ±trust_region))`.
Positivity by construction; clamped to `[eps, sigma_max]`. Ported from VFE_2.0
`vfe_utils.retract_spd_diagonal_torch` (line 727). Golden-equal at atol=1e-5.

**`retract_spd_full(sigma, delta_sigma, *, step_size, trust_region, eps, sigma_max)`**
Full SPD retraction via the affine-invariant exponential map:
`Sigma_new = S^{1/2} exp(S^{-1/2} (tau dS) S^{-1/2}) S^{1/2}`.
Frobenius trust region on the whitened tangent; eigenvalue clamp `[eps, sigma_max^2]`
on output. Uses `torch.linalg.eigh`; VFE_2.0's gap-regularized `_safe_eigh` custom
backward is deferred to a hardening pass (forward values match on well-conditioned
inputs). Ported from VFE_2.0 `vfe_utils.retract_spd_torch` (line 635).
Golden-equal at atol=1e-3 (eigh sign/order invariance under V diag V^T).

**`natural_gradient(grad_mu, grad_sigma, sigma_q, *, eps)`**
Fisher preconditioner converting Euclidean gradients to natural gradients:
`nat_mu = Sigma @ grad_mu`, `nat_sigma = 2 Sigma @ grad_sigma @ Sigma` (diagonal: element-wise
`2 sigma^2 grad_sigma`). Diagonal vs. full path auto-detected by `sigma_q.dim() == grad_mu.dim()`.
Ported from VFE_2.0 `vfe_gradients.compute_natural_gradient_gpu` (line 1938).
Golden-equal at atol=1e-5 (diagonal) / 1e-4 (full).

### Test results

```
67 passed in 0.09s
```

All 8 new tests pass (4 golden + 4 property); no regressions in the prior 59 tests.

### Commits

- `ee3c1f5 feat(geometry): diagonal SPD retraction, golden-equal to 2.0`
  (contains all three retraction functions — written in a single pass)
- `60dab84 test(geometry): SPD-preservation + Fisher-formula properties`

---

## Phase 2d φ Lie-Algebra Retraction — 2026-05-29 (continuation)

Built the gauge-frame (φ) Lie-algebra retraction subsystem. Self-contained, V3-internal
tests only (analytic known-value + property); correctness pinned by hand-derived anchors,
not by re-running the implementation's own formula.

### Files created

- `vfe3/geometry/lie_ops.py` — coordinate↔matrix maps (`embed_phi`, `extract_phi`,
  `gram_pinv`), Lie bracket (`lie_bracket_matrix`, `lie_bracket_coords`), a
  `{euclidean, bch}` composition registry (`register_compose`/`get_compose`/`compose_phi`),
  the GL(K)/SO(N) retractions (`retract_glk`, `retract_son`, shared `_retract_core`), and
  determinant control (`project_phi_to_slk`, `clamp_phi_trace`).
- `tests/test_lie_ops.py` — embed/extract round-trip + overcomplete projection, so(3)
  structure constants, BCH (commuting-exact + residual-rate + degree-5 coord pin).
- `tests/test_phi_retraction.py` — retract_glk/son (trust region, max-norm ceiling,
  det>0, SO(N) orthogonality), the unclamped euclidean-update pin, det control, dispatcher.

### Files modified

- `vfe3/geometry/retraction.py` — added the group-aware `retract_phi(phi, delta_phi, group, *, …)`
  dispatcher (imports `GaugeGroup` + the `lie_ops` retraction/det-control names + `math`).

### Changes

**Composition is a registry seam.** Default `euclidean` (the manuscript working update
`φ⁺ = φ − η ∂F/∂φ`; 𝔤 is a vector space — `GL(K)_supplementary.tex` §Gauge Frame
Gradients). `bch` is the higher-order chart correction; the exact group retraction
`U←U·exp(−ηΔ)` is named as the `omega_direct` transport partner (not built).

**BCH in matrix space, extracted once.** `compose_bch` embeds φ₁,φ₂, accumulates the
symmetric Dynkin series (to order 4 / degree 5) with matrix commutators, then extracts
coordinates a single time — exact up to truncation for a closed subalgebra.

**Coordinate extraction is overcomplete-safe.** `extract_phi` solves the Frobenius Gram
system via pseudo-inverse; `gram_pinv` computes the Gram + pinv **in float64** then casts
back (the overcomplete sl(K) basis from `generate_glk(include_identity=False)` has a true
1-D nullspace whose eigenvalue is ~−8e-8 in float32, larger than `rcond` — a float32 pinv
would inject noise into the trace direction). Per the spec's "float32 storage; float64
internal where conditioning demands it." For a complete orthonormal basis Gram = c·I and
the upcast is a no-op.

**Determinant control.** `project_phi_to_slk` removes the per-block trace component
(`V_h[a] = tr(G_a|block h)`, `φ − Σ_h (φ·V_h/‖V_h‖²)V_h`) so `det(Ω_h)=1`; `clamp_phi_trace`
soft-bounds `|tr|≤T`. The dispatcher applies det control only on the GL(K) path; the
SO(N) path skips it (det = 1 automatic for skew generators).

### Analytic anchors (independent of the implementation)

- so(3) structure constants `[G₀,G₁]=−G₂, [G₀,G₂]=+G₁, [G₁,G₂]=−G₀`.
- BCH residual log-log slope = order+2; measured **3.001 / 4.005 / 5.001 / 5.999** for
  orders 1–4 → all Dynkin coefficients verified, none transposed. A direct coordinate pin
  against a matrix-log reference additionally catches a zeroed/sign-flipped degree-5 block.
- `embed∘extract∘embed = embed` for the overcomplete sl(K) set.
- `det(exp)>0` for GL⁺(K); SO(N) orthogonality + det +1; per-block unit det after sl(K)
  projection; `|tr|≤T` after the soft clamp.
- The unclamped euclidean update equals `φ + step·δ` (pins the update formula directly; the
  det>0 / orthogonality tests are properties of `exp(embed(·))`, true for any output).

### Adversarial review

A 4-expert panel (gauge-theory, numerics, implementation-wiring, code-quality) found no
logic bugs; it surfaced two test-quality gaps, both fixed by strengthening tests (no
analytic assertion weakened): the `max_norm` clamp lacked discriminating coverage (the
trust-region pre-clamp capped the step before the ceiling, so the one-sided `≤5.0`
assertion could not distinguish GL's 5.0 from SO's π — split into a trust-region test and
a ceiling-binding equality test), and the "BCH catches a wrong coefficient" claim was
overstated (a 3% error in the degree-5 coefficient is masked by the O(ε⁶) truncation —
claim corrected and a direct coordinate-pin test added).

### Test results

```
68 passed
```

14 new tests (6 `test_lie_ops.py` + 8 `test_phi_retraction.py`); no regressions in the
54 pre-existing tests.

### Commits

- `527bd18 feat(geometry): lie_ops embed/extract/bracket (V3-internal analytic tests)`
- `dc89283 feat(geometry): phi composition registry (euclidean + BCH), residual-rate pinned`
- `368ee14 feat(geometry): GL(K)/SO(N) phi retraction (group-membership + det>0 pinned)`
- `e8a5012 feat(geometry): phi determinant control (sl(K) projection + trace clamp)`
- `610a72d feat(geometry): retract_phi dispatcher (group-aware GL(K)/SO(N) + det control)`
- `8f2a149 test(geometry): make max_norm clamp and BCH degree-5 coeffs genuinely tested`
- `4b4c6ab test(geometry): pin the unclamped euclidean phi-update directly`

Merged to `main` (fast-forward) and pushed to `origin/main`.
