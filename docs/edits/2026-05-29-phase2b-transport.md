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

## Phase 2e φ-Gradient Preconditioner — 2026-05-29 (continuation)

Built the gauge-frame (φ) gradient preconditioner — the φ analog of the (μ,σ) Fisher
natural gradient. It conditions a Euclidean φ-gradient before the Phase 2d retraction
(the E-step is `grad → precondition → retract`). Self-contained, V3-internal tests only
(analytic known-value + property + finite-difference); correctness pinned by hand-derived
anchors and an independent finite-difference oracle, not by re-running the implementation's
own formula.

### Files created

- `vfe3/geometry/phi_preconditioner.py` — a `{none, clip, killing, killing_per_block,
  pullback}` preconditioning registry (`register_precond`/`get_precond`/
  `precondition_phi_gradient`), the Cartan-involution Killing metric (`killing_metric`,
  `build_killing_preconditioner`), the block-diagonal variant (`_generator_block_index`,
  `build_killing_preconditioner_per_block`), and the position-dependent pullback metric
  (`_structure_constants`, `pullback_metric`). Pure: operates on a generator TENSOR
  (+ optional `irrep_dims`), never a `GaugeGroup`.
- `tests/test_phi_preconditioner.py` — none/clip behavior, the gl(2) Killing literal +
  so(3) positivity tell, center-reg exact-inverse-on-sl(K), per-block block-diagonal
  structure, pullback φ=0 == Frobenius Gram + finite-difference-of-exp match + K-guard.

### Files modified

- (none — additive only.)

### Changes

**Preconditioning is a registry seam, default `none`.** The canonical update applies no
metric correction: the exponential map exp: 𝔤 → G provides natural coordinates and the
gradient lives in the Lie algebra 𝔤, a vector space (`GL(K)_supplementary.tex`, §Gauge
Frame Preconditioning). `none` is the identity; `clip` is the practical robustness baseline
`grad·min(1, c/‖grad‖)`; `killing`/`killing_per_block`/`pullback` are the principled
toggles. Coordinates in, coordinates out `(…,n_gen)` — same units as `retract_phi`'s
`delta_phi`. Each registered rule takes `**kwargs` so the dispatcher forwards one uniform
argument set; a rule reads only the knobs it needs.

**`killing_metric` is the Cartan-involution form, not the bare Killing form.** `g̃_ab =
2K·tr(Gₐᵀ G_b) − 2·tr(Gₐ)·tr(G_b)`, i.e. `−B(θX,Y)` with the Cartan involution θ(X) = −Xᵀ;
positive-definite on sl(K). `gram` is the FROBENIUS inner product `tr(Gₐᵀ G_b) = Σ_ij
Gₐ[i,j]G_b[i,j]`. The bare Killing form `B(X,Y) = 2K·tr(XY) − 2tr(X)tr(Y)` of gl(K) is
indefinite (negative-definite on skew directions) and is deliberately NOT used; the gl(2)
literal anchor below discriminates the two.

**Center-regularization regularizes the numerical nullspace, not a hardcoded direction.**
`build_killing_preconditioner` eigendecomposes g̃ (`eigh`, **float64**, symmetrized first),
lifts eigenvalues with `|λ| < tol` to `center_reg` (default 2K), then inverts via
`V·diag(1/λ)·Vᵀ`. Non-null eigenvalues are untouched, so the inverse is EXACT on sl(K)
(`g̃ @ (Minv @ v) = v` for v ⊥ the center) — a ridge `center_reg·I` would perturb every
direction and fail this. so(K), already PD with no near-null eigenvalue, acquires no
regularization. Storage dtype is restored on return (float32 in/out, float64 internal where
conditioning demands it).

**Per-block Killing is a direct-sum metric.** `build_killing_preconditioner_per_block`
groups generators by irrep block (`_generator_block_index` asserts single-block support),
builds the local-dimension `d_h` Cartan metric on each group, and assembles a
block-diagonal inverse in generator-index order — no cross-block coupling. A single global
block (`irrep_dims == [K]`, e.g. cross-coupled bases) reduces to the global
`build_killing_preconditioner`.

**Pullback is the position-dependent natural gradient, finite-difference-pinned and
K-guarded.** `pullback_metric` computes `G_ab(φ) = ⟨d exp_φ(Tₐ), d exp_φ(T_b)⟩_F` with
`d exp_φ(T) = Ψ(ad_φ)(T)·exp(φ)`, `Ψ(z) = (eᶻ−1)/z = Σ_{k≥0} zᵏ/(k+1)!`, where ad_φ acts on
coordinates via the structure constants `(ad_φ)_{cb} = Σ_a φ^a f[a,b,c]` and `f[a,b,c]` are
the coordinates of `[Gₐ,G_b]` (reusing the bracket + Gram-pseudo-inverse extraction). The
Ψ-series is accumulated in float64 and summed **adaptively**: terms accrue until the new
term's max |entry| drops below `series_tol` (default 1e-12), capped at `series_order`
(default 40). The truncation error of Ψ(ad_φ) grows with ‖φ‖ (the ad_φ eigenvalues scale
with ‖φ‖), so a fixed low order is inaccurate in exactly the non-compact large-norm regime
the pullback metric exists for; the adaptive cutoff holds it to the FD-of-exp oracle across
`retract_glk`'s full `max_norm = 5` range (e.g. ‖φ‖ = 2 → ~4e-8, ‖φ‖ = 5 → ~3e-6). The
`(k+1)!` coefficient is a **float** divisor — an int divisor overflows tensor division past
order ~20. `d exp_φ(eₐ)` in coords is column a of `Ψ(ad_φ)`, embedded and right-multiplied
by `exp(φ)`. The structure-constants tensor is O(n_gen²·K²), so K > `max_k` (default 12)
raises before allocation. The `pullback` rule solves `(G(φ) + εI)·nat = grad` rather than
forming an explicit inverse.

**Device-agnostic throughout (RTX 5090 / CUDA).** Every internally-created tensor is built
on the input's device: `pullback_metric`'s identity (`torch.eye(…, device=ad.device)`) and
`build_killing_preconditioner_per_block`'s block-assembly buffer and block-index buffer
(`torch.zeros`/`torch.full(…, device=generators.device)`). Without the explicit `device=`,
those tensors default to CPU and crash on cross-device matmul/index-assign when φ and the
generators live on CUDA — the multi-block `killing_per_block` and `pullback` modes were the
affected runtime paths. The Killing single-block path was already device-safe
(`.double()`/`full_like`/`.to(orig_dtype)` all inherit the input device).

### Analytic anchors (independent of the implementation)

- gl(2) elementary basis (E00,E01,E10,E11), K=2: `killing_metric =
  [[2,0,0,−2],[0,4,0,0],[0,0,4,0],[−2,0,0,2]]`, eigenvalues {0,4,4,4}, the null direction
  being the identity/center (E00+E11). The bare Killing form would give a different,
  indefinite matrix — this literal is the discriminator.
- so(3) (skew, tr=0): g̃ = 2K·gram, gram = 2·I (‖L_ij‖_F² = 2), K=3 → g̃ = 12·I,
  positive-definite. The bare Killing form is negative-definite on skew — the sign is the
  tell.
- Center-reg exact inverse on sl(K): with the gl(2) null eigenvalue lifted to 4, the
  trace-free directions satisfy `g̃ @ (Minv @ v) = v` exactly; the full regularized metric
  has spectrum {4,4,4,4} (PD). A ridge would not preserve the sl(K) eigenvalues.
- Per-block: the (8,8) inverse for gl(2)⊕gl(2) is block-diagonal in 4+4 (zero cross-block
  coupling), and each diagonal block equals the single-head gl(2) Killing inverse.
- Pullback @ φ=0 equals the Frobenius Gram (Ψ(0)=I, exp(0)=I); pullback(φ) matches the
  central finite-difference of `matrix_exp` (`Jₐ = ∂_ε exp(embed(φ±ε eₐ))`, `G_FD[a,b] =
  tr(JₐᵀJ_b)`) to 1e-4 on so(3) at φ=(0.4,−0.3,0.5) — the independent oracle validating the
  Ψ-series and operator ordering; symmetric PD.
- Pullback at the SHIPPED DEFAULT knobs (no explicit `series_order`) on a non-compact
  symmetric gl(2) φ at ‖φ‖=2 matches the FD-of-exp oracle to 1e-4 — pins the default in the
  large-norm regime, not just the small-angle compact corner. A second default-knob check
  drives ‖φ‖=5 and asserts the metric is finite (the float-coefficient series does not
  overflow at the order-40 cap).

### Adversarial review

A 4-expert panel (gauge-theory, runtime-wiring, code-quality returned; the numerics lens
failed to emit structured output, but its substantive findings were caught redundantly by
the other lenses). 6 of 7 actionable findings confirmed and fixed in `07f4674`: the fixed
order-6 Ψ-series was inaccurate in the large-‖φ‖ regime the pullback metric exists for (now
adaptive), a latent `OverflowError` from an int factorial divisor (now float), and two CUDA
device-mismatch bugs (`torch.eye`/`torch.zeros`/`torch.full` without `device=`) on the
`pullback` and multi-block `killing_per_block` paths. These device fixes are verified by
code reading and the full CPU suite; they were NOT exercised on a GPU (no CUDA in the build
environment) and remain to be confirmed on the RTX 5090.

One finding was rejected as a code change but is a correct theory observation, recorded in
the `killing_metric` module docstring: the Cartan-involution metric uses the Frobenius form
`tr(G_aᵀ G_b)`, which is Ad-invariant only under the compact subgroup (`tr((gXg⁻¹)ᵀ gYg⁻¹)
= tr(XᵀY)` iff `gᵀg = I`). So the Killing-preconditioned natural gradient is gauge-
equivariant under SO(N) but NOT under general GL(K) in the non-compact (symmetric)
directions — a left-/Ad(K)-invariant metric, not bi-invariant. The `pullback` metric is the
position-dependent alternative for the non-compact regime. No shipped artifact claims full
GL(K) gauge-equivariance for the Killing modes.

### Test results

```
82 passed
```

14 new tests in `tests/test_phi_preconditioner.py` (3 none/clip + 4 Killing + 2 per-block +
5 pullback — φ=0 Gram, FD-of-exp on so(3), FD-of-exp at the default order on a non-compact
gl(2) φ, the order-40 finite/no-overflow check, and the K-guard); no regressions in the 68
pre-existing tests.

### Commits

- `aa53e0d feat(geometry): phi preconditioner registry (none default + clip)`
- `dd8a6ee feat(geometry): Killing (Cartan-involution) preconditioner, nullspace-regularized`
- `4bb23d5 feat(geometry): per-block Killing preconditioner (block-diagonal natural gradient)`
- `3884e60 feat(geometry): pullback natural-gradient preconditioner (FD-of-exp pinned)`
- `d8b1475 docs(edits): 2026-05-29 phase 2e phi-preconditioner changes log`
- `07f4674 fix(geometry): device-agnostic phi preconditioner + adaptive pullback Psi-series`
- (this entry) docstring caveat: Killing metric is Ad(K)-invariant, not bi-invariant on GL(K).
