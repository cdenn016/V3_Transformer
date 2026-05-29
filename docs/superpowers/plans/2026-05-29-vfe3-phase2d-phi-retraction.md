# VFE_3.0 Phase 2d (φ Lie-Algebra Retraction) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:test-driven-development (RED→GREEN→COMMIT per step). Steps use checkbox (`- [ ]`) syntax. Tests are **V3-internal** (analytic known-value + property); there is **no** golden comparison against VFE_2.0 — 2.0 is a math *guide* only, never imported or compared.

**Goal:** Build the φ (gauge-frame) Lie-algebra retraction subsystem — `vfe3/geometry/lie_ops.py` (coordinate↔matrix maps, Lie bracket, a **composition registry** `{euclidean, bch}`, the GL(K)/SO(N) retractions, and determinant control) plus a `retract_phi(group, …)` dispatcher in `vfe3/geometry/retraction.py`. Correctness is pinned by hand-derived analytic values and properties (structure constants, BCH residual rate, group membership, det control), not by re-running the implementation's own formula.

**Architecture:** Fourth geometry sub-phase. The transport layer (Phase 2b) builds `Ω_ij = exp(φ_i)exp(−φ_j)` from a frame `φ`; this phase supplies the *update* that moves `φ` along a tangent step while keeping the resulting group element on `GL⁺(K)` / `SO(N)`. `lie_ops.py` is **pure** (operates on coordinate tensors + a generator tensor; it does NOT import `GaugeGroup`). The group-aware `retract_phi` dispatcher lives in `retraction.py` (which already imports nothing group-specific; it will now import `GaugeGroup` + `lie_ops`). The φ-gradient **preconditioner** (Killing/Cartan metric, pullback) is **Phase 2e**, named below as an extension point — it is split out because the Killing metric is singular on the trace direction and needs its own center-regularization treatment and test surface.

**Tech Stack:** Python 3, PyTorch (float32 storage; float64 internal where conditioning demands it), pytest. No NN. No CLI.

**Reference spec:** `docs/superpowers/specs/2026-05-29-vfe3-clean-room-design.md` (§4.2, the retraction seam). Prereq: Phases 2a/2b/2c on branch `phase2d-phi-retraction`.

**Manuscript theory (authority):**
- φ ∈ 𝔤 (Lie algebra, a vector space); group element `U_i = exp(φ_i) ∈ GL⁺(K)` since `det(exp φ)=exp(tr φ)>0` (`GL(K)_attention.tex` ll. 457, 564, 619; `GL(K)_supplementary.tex` ll. 459–476, 561).
- **Canonical** update is the exact group retraction `U_i ← U_i·exp(−η_φ ∇̃_{φ_i}F)` (`Participatory_it_from_bit.tex` ll. 2707–2717); its **chart-coordinate** form is the BCH step `φ⁺ = φ − ηΔ + ½[φ, ηΔ] + O(η²)`; the **working/default** implementation is the plain Lie-algebra step `φ⁺ = φ − η_φ ∂F/∂φ` (`GL(K)_supplementary.tex` ll. 550–557 — "no metric correction is applied … the gradient naturally lives in 𝔤").
- Regime I flat cocycle `Ω_ij = exp(φ_i)exp(−φ_j)`, vanishing holonomy (default); Regime II edge-relaxed `Ω_ij = exp(φ_i)exp(δ_ij·G)exp(−φ_j)` deferred (`GL(K)_attention.tex` ll. 640–666).
- GL(K) gauge-invariance of KL: `D_KL(Ω_*P‖Ω_*Q)=D_KL(P‖Q)` (`GL(K)_attention.tex` ll. 520–556) — the property the retraction must preserve.

**VFE_2.0 guide (math reference only; do NOT import or golden-test):**
- `math_utils/generators/lie_ops.py`: `retract_glK_torch` (503–546), `retract_soN_torch` (197–235), `glK_compose_bch_torch` (431–500), `soN_compose_bch_torch` (129–194), brackets (58–85, 359–388).
- `transformer/core/vfe_utils.py::_retract_phi` (802–926) dispatcher; `_apply_det_control` (928–998).
- Constants: GL(K) `trust_region=0.1, max_norm=5.0`; SO(N) `trust_region=0.3, max_norm=π`; `bch_order=4` (degree-5, six Dynkin terms, O(ε⁶) error); det-control `eps=1e-12`.

**Design decisions settled before this plan (do not relitigate in code):**
1. **Composition is a registry seam**, not hardcoded. `_COMPOSE = {"euclidean", "bch"}`; default `"euclidean"` (manuscript working path + 2.0's "BCH bracket can explode for GL(K)" stability note). `"bch"` is the higher-order chart correction toggle. The *exact* group retraction `U←U·exp(−ηΔ)` is the natural partner of the existing `omega_direct` transport parameterization and is named as a Phase-2e/transport extension point, not built here.
2. **BCH is computed in matrix space, extracted once** — embed φ₁,φ₂ to matrices, accumulate the Dynkin series with matrix commutators, `extract` the result a single time. For a closed subalgebra all nested brackets stay in the span, so this is exact up to the truncation order and avoids repeated coordinate projections (a clean-room simplification over 2.0's per-bracket extraction).
3. **Generators are the acting K×K representation** (from `groups.py`). Brackets are computed directly in that representation — no separate "N×N gauge generator" rep as in 2.0.
4. **`matrix_exp` has one home**: tests and any group-element construction route through `transport.stable_matrix_exp_pair` (Frobenius clamp + float64 upcast already there).

---

## Code Style (MANDATORY — repo CLAUDE.md)

Argument order: tensors first; then `float|Tensor`; then undefined floats/ints/bools; then defined floats, defined ints, defined bools (defined `str` placed with the defined scalars); then `Optional`; then `**kwargs`. Vertical alignment of names / type annotations / `=` / trailing `#` comments; blank lines separate type groups; shape comments at critical points; type hints on every signature; docstrings carry the LaTeX/math form. Variable names match paper notation (`phi`, `delta_phi`, `generators`, `irrep_dims`).

## Provenance (MANDATORY — convention as of commit `114839c`)

V3 is a self-contained implementation, **not a port**. **No shipped artifact — module/function docstring, inline comment, test name, or test comment — may mention "VFE_2.0", "2.0", "ported", or any 2.0 file/line.** Cite only the **manuscript** (`GL(K)_attention.tex`, `GL(K)_supplementary.tex`, `Participatory_it_from_bit.tex`) and the **math itself**. The "VFE_2.0 guide" references in *this plan's prose* are internal derivation notes for the author only; they must not survive into code. (This plan's code snippets already follow the rule — keep it that way.)

---

## File Structure

- **Create** `vfe3/geometry/lie_ops.py` — `embed_phi`, `extract_phi`, `gram_pinv`, `lie_bracket_matrix`, `lie_bracket_coords`, `register_compose`/`get_compose`, `compose_euclidean`, `compose_bch`, `retract_glk`, `retract_son`, `project_phi_to_slk`, `clamp_phi_trace`.
- **Modify** `vfe3/geometry/retraction.py` — add the group-aware `retract_phi` dispatcher.
- **Create** `tests/test_lie_ops.py` — embed/extract round-trip + projection, bracket structure constants, BCH (commuting-exact + residual-rate).
- **Create** `tests/test_phi_retraction.py` — retract_glk/son (trust region, max norm, group membership, det>0), det control, dispatcher.

---

## Task 1 — `lie_ops.py`: embed / extract / bracket

**Files:** Create `vfe3/geometry/lie_ops.py`; Test `tests/test_lie_ops.py`.

- [ ] **Step 1 (RED): failing tests** — create `tests/test_lie_ops.py`:

```python
import math

import torch

from vfe3.geometry.generators import generate_glk, generate_son
from vfe3.geometry.lie_ops import (
    embed_phi,
    extract_phi,
    lie_bracket_coords,
)


def test_embed_extract_roundtrip_independent_basis():
    # gl(2) elementary basis is orthonormal under Frobenius -> extract(embed(c)) == c.
    G = generate_glk(2)                                   # (4, 2, 2)
    c = torch.randn(3, 4)
    out = extract_phi(embed_phi(c, G), G)
    assert torch.allclose(out, c, atol=1e-6)


def test_embed_extract_projection_overcomplete():
    # sl(K) spanning set (include_identity=False) is OVERCOMPLETE (rank K^2-1):
    # extract(embed(c)) need NOT equal c, but embed o extract o embed == embed.
    G = generate_glk(3, include_identity=False)           # (<=9, 3, 3), rank 8
    c = torch.randn(2, G.shape[0])
    M = embed_phi(c, G)
    M2 = embed_phi(extract_phi(M, G), G)
    assert torch.allclose(M2, M, atol=1e-5)


def test_bracket_so3_structure_constants():
    # generate_son(3) basis: G0=E01-E10, G1=E02-E20, G2=E12-E21.
    # Hand-derived: [G0,G1]=-G2, [G0,G2]=+G1, [G1,G2]=-G0.
    G = generate_son(3)                                   # (3, 3, 3)
    e = torch.eye(3)
    c01 = lie_bracket_coords(e[0], e[1], G)
    c02 = lie_bracket_coords(e[0], e[2], G)
    c12 = lie_bracket_coords(e[1], e[2], G)
    assert torch.allclose(c01, torch.tensor([0.0, 0.0, -1.0]), atol=1e-6)
    assert torch.allclose(c02, torch.tensor([0.0, 1.0,  0.0]), atol=1e-6)
    assert torch.allclose(c12, torch.tensor([-1.0, 0.0, 0.0]), atol=1e-6)
```

- [ ] **Step 2:** Run `python -m pytest tests/test_lie_ops.py -q` — expect FAIL (ImportError).

- [ ] **Step 3 (GREEN): implement** — create `vfe3/geometry/lie_ops.py` with the header + these functions:

```python
r"""Lie-algebra retraction primitives for VFE_3.0 gauge frames (Gaussian-specialized).

The gauge frame phi lives in a Lie algebra g (a vector space) as coordinates in a
generator basis {G_a}: the algebra element is embed(phi) = sum_a phi^a G_a. The
group element U = exp(embed(phi)) lies in GL+(K) (det>0) or SO(N). This module
supplies: coordinate<->matrix maps, the Lie bracket, a composition registry
(euclidean step or BCH chart correction), the GL(K)/SO(N) retractions, and
determinant control. Pure: operates on a generator TENSOR, not a GaugeGroup.
"""

from typing import Callable, Dict, List, Optional

import torch


def embed_phi(
    phi:        torch.Tensor,             # (..., n_gen) Lie-algebra coordinates
    generators: torch.Tensor,             # (n_gen, K, K) basis
) -> torch.Tensor:                        # (..., K, K) matrix sum_a phi^a G_a
    r"""Coordinates -> algebra element: embed(phi) = sum_a phi^a G_a."""
    return torch.einsum("...a,aij->...ij", phi, generators)


def gram_pinv(
    generators: torch.Tensor,             # (n_gen, K, K) basis

    *,
    rcond:      float = 1e-10,
) -> torch.Tensor:                        # (n_gen, n_gen) pseudo-inverse of Gram
    r"""Pseudo-inverse of the Frobenius Gram matrix Gram_ab = <G_a, G_b>_F.

    pinv (not inv) so overcomplete / rank-deficient spanning sets (e.g. the
    sl(K) set from generate_glk(include_identity=False)) are handled.
    """
    gram = torch.einsum("aij,bij->ab", generators, generators)
    return torch.linalg.pinv(gram, rcond=rcond)


def extract_phi(
    matrix:     torch.Tensor,             # (..., K, K) element of span{G_a}
    generators: torch.Tensor,             # (n_gen, K, K) basis

    *,
    gram_pinv_: Optional[torch.Tensor] = None,   # cached gram_pinv(generators)
) -> torch.Tensor:                        # (..., n_gen) min-norm coordinates
    r"""Algebra element -> coordinates by least squares against the Gram matrix.

    Solves Gram c = g with g_b = <G_b, matrix>_F, c = Gram^+ g (min-norm solution
    when the basis is overcomplete). For an orthonormal basis Gram = I and
    c_a = <G_a, matrix>_F.
    """
    gp = gram_pinv(generators) if gram_pinv_ is None else gram_pinv_
    g = torch.einsum("aij,...ij->...a", generators, matrix)
    return torch.einsum("...a,ab->...b", g, gp)


def lie_bracket_matrix(
    A: torch.Tensor,                      # (..., K, K)
    B: torch.Tensor,                      # (..., K, K)
) -> torch.Tensor:                        # (..., K, K) [A,B] = AB - BA
    r"""Matrix commutator [A, B] = AB - BA (sign convention AB - BA)."""
    return A @ B - B @ A


def lie_bracket_coords(
    phi1:       torch.Tensor,             # (..., n_gen)
    phi2:       torch.Tensor,             # (..., n_gen)
    generators: torch.Tensor,             # (n_gen, K, K)

    *,
    gram_pinv_: Optional[torch.Tensor] = None,
) -> torch.Tensor:                        # (..., n_gen) coords of [embed phi1, embed phi2]
    r"""Bracket in coordinates: extract([embed(phi1), embed(phi2)])."""
    A = embed_phi(phi1, generators)
    B = embed_phi(phi2, generators)
    return extract_phi(lie_bracket_matrix(A, B), generators, gram_pinv_=gram_pinv_)
```

- [ ] **Step 4:** Run `python -m pytest tests/test_lie_ops.py -q` — expect 3 passed.

- [ ] **Step 5 (COMMIT):**
```
git add vfe3/geometry/lie_ops.py tests/test_lie_ops.py
git commit -m "feat(geometry): lie_ops embed/extract/bracket (V3-internal analytic tests)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2 — composition registry: `euclidean` + `bch`

**Files:** Modify `vfe3/geometry/lie_ops.py`; Test `tests/test_lie_ops.py`.

- [ ] **Step 1 (RED): failing tests** — append to `tests/test_lie_ops.py`:

```python
from vfe3.geometry.lie_ops import compose_phi, get_compose


def test_compose_euclidean_is_sum():
    G = generate_glk(2)
    a, b = torch.randn(4), torch.randn(4)
    assert torch.allclose(compose_phi(a, b, G, mode="euclidean"), a + b, atol=1e-6)


def test_bch_commuting_is_exact():
    # Two diagonal gl(3) elements commute -> BCH == euclidean sum exactly.
    G = generate_glk(3)
    a = torch.zeros(9); a[0] = 0.7          # E00 direction
    b = torch.zeros(9); b[8] = -0.4         # E22 direction
    z = compose_phi(a, b, G, mode="bch", order=4)
    assert torch.allclose(z, a + b, atol=1e-6)


def _bch_residual(order: int, eps: float) -> float:
    # || exp(embed(bch(eps X, eps Y))) - exp(embed(eps X)) exp(embed(eps Y)) ||_F.
    # float64 so the slope is not floored by float32 round-off.
    torch.manual_seed(0)
    G = generate_son(3).double()
    X = eps * torch.tensor([0.9, -0.3, 0.5], dtype=torch.float64)
    Y = eps * torch.tensor([-0.2, 0.7, 0.4], dtype=torch.float64)
    z = compose_phi(X, Y, G, mode="bch", order=order)
    lhs = torch.linalg.matrix_exp(embed_phi(z, G))
    rhs = torch.linalg.matrix_exp(embed_phi(X, G)) @ torch.linalg.matrix_exp(embed_phi(Y, G))
    return float(torch.linalg.norm(lhs - rhs))


def test_bch_residual_rate_order_matches_slope():
    # Truncation error of order-k BCH is O(eps^(k+2)); the log-log slope of the
    # residual vs eps must be ~ k+2. This pins the series structure + low-order
    # coefficients (a missing/gross/structural term degrades the slope); it does
    # NOT catch a small error in the highest INCLUDED coefficient — that is pinned
    # exactly by test_bch_order4_coords_match_matrix_log (direct coord check).
    for order, expected in [(1, 3.0), (2, 4.0), (4, 6.0)]:
        eps = [0.2, 0.1, 0.05]
        r = [_bch_residual(order, e) for e in eps]
        slope = (math.log(r[0]) - math.log(r[-1])) / (math.log(eps[0]) - math.log(eps[-1]))
        assert abs(slope - expected) < 0.8, f"order={order}: slope={slope:.2f} != {expected}"
```

- [ ] **Step 2:** Run — expect FAIL (ImportError).

- [ ] **Step 3 (GREEN): implement** — append to `vfe3/geometry/lie_ops.py`. The composition seam is a registry. BCH is the symmetric Dynkin series for `log(exp A exp B)` truncated at the degree the `order` selects (order k ⇒ all terms up to degree k+1 ⇒ error O(ε^{k+2})):

```python
_COMPOSE: Dict[str, Callable[..., torch.Tensor]] = {}


def register_compose(name: str) -> Callable:
    """Decorator registering a composition rule phi1,phi2 -> composed coords."""
    def _wrap(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        _COMPOSE[name] = fn
        return fn
    return _wrap


def get_compose(name: str) -> Callable[..., torch.Tensor]:
    """Return the registered composition rule (KeyError if absent)."""
    if name not in _COMPOSE:
        raise KeyError(f"no composition rule {name!r}; available: {sorted(_COMPOSE)}")
    return _COMPOSE[name]


@register_compose("euclidean")
def compose_euclidean(
    phi1:       torch.Tensor,             # (..., n_gen)
    phi2:       torch.Tensor,             # (..., n_gen)
    generators: torch.Tensor,             # (n_gen, K, K) (unused; kept for a uniform seam)

    *,
    order:      int = 0,
    gram_pinv_: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Plain Lie-algebra step phi1 + phi2 (exact iff [phi1, phi2] = 0).

    The manuscript working/default update: g is a vector space, so the tangent
    step is the sum of coordinates (GL(K)_supplementary.tex ll. 550-557).
    """
    return phi1 + phi2


@register_compose("bch")
def compose_bch(
    phi1:       torch.Tensor,             # (..., n_gen)
    phi2:       torch.Tensor,             # (..., n_gen)
    generators: torch.Tensor,             # (n_gen, K, K)

    *,
    order:      int = 4,
    gram_pinv_: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""BCH chart correction: coords of log(exp(embed phi1) exp(embed phi2)).

    Symmetric Dynkin series (matrix space, extracted once). Terms by `order`:
      order>=1: + 1/2 [X,Y]
      order>=2: + 1/12 ([X,[X,Y]] - [Y,[X,Y]])
      order>=3: - 1/24 [Y,[X,[X,Y]]]
      order>=4: - 1/720 ([Y,[Y,[Y,[Y,X]]]] + [X,[X,[X,[X,Y]]]])
                + 1/360 ([X,[Y,[Y,[Y,X]]]] + [Y,[X,[X,[X,Y]]]])
                + 1/120 ([Y,[X,[Y,[X,Y]]]] + [X,[Y,[X,[Y,X]]]])
    Truncation error is O(||X||^{order+2} + ||Y||^{order+2}).
    """
    X = embed_phi(phi1, generators)
    Z = embed_phi(phi1, generators) + embed_phi(phi2, generators)
    Y = embed_phi(phi2, generators)
    br = lie_bracket_matrix
    if order >= 1:
        XY = br(X, Y)
        Z = Z + 0.5 * XY
    if order >= 2:
        Z = Z + (1.0 / 12.0) * (br(X, XY) - br(Y, XY))
    if order >= 3:
        Z = Z - (1.0 / 24.0) * br(Y, br(X, XY))
    if order >= 4:
        YX  = br(Y, X)
        YYX = br(Y, YX); YYYX = br(Y, YYX)
        XXY = br(X, XY); XXXY = br(X, XXY)
        Z = Z - (1.0 / 720.0) * (br(Y, YYYX) + br(X, XXXY))
        Z = Z + (1.0 / 360.0) * (br(X, YYYX) + br(Y, XXXY))
        Z = Z + (1.0 / 120.0) * (br(Y, br(X, br(Y, XY))) + br(X, br(Y, br(X, YX))))
    return extract_phi(Z, generators, gram_pinv_=gram_pinv_)


def compose_phi(
    phi1:       torch.Tensor,             # (..., n_gen)
    phi2:       torch.Tensor,             # (..., n_gen)
    generators: torch.Tensor,             # (n_gen, K, K)

    *,
    order:      int = 4,
    mode:       str = "euclidean",
    gram_pinv_: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Dispatch to the registered composition rule `mode`."""
    return get_compose(mode)(phi1, phi2, generators, order=order, gram_pinv_=gram_pinv_)
```

> **Implementation note:** verify the degree-5 block against BOTH the residual-rate test (catches a structural/missing term: slope drops below 6) AND the direct coordinate check `test_bch_order4_coords_match_matrix_log` (catches a zeroed/sign-flipped coefficient against the matrix-log reference). The slope alone cannot catch a small error in the leading degree-5 coefficient — it is masked by the O(ε⁶) truncation. The `1/120` pair is the symmetric-form term `[Y,[X,[Y,[X,Y]]]] + [X,[Y,[X,[Y,X]]]]`. The tests are the arbiter — do not "fix" a test to match the code.

- [ ] **Step 4:** Run — expect all `test_lie_ops.py` passing (6 tests).

- [ ] **Step 5 (COMMIT):**
```
git add vfe3/geometry/lie_ops.py tests/test_lie_ops.py
git commit -m "feat(geometry): phi composition registry (euclidean + BCH), residual-rate pinned

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3 — `retract_glk` / `retract_son`

**Files:** Modify `vfe3/geometry/lie_ops.py`; Test `tests/test_phi_retraction.py`.

- [ ] **Step 1 (RED): failing tests** — create `tests/test_phi_retraction.py`:

```python
import math

import torch

from vfe3.geometry.generators import generate_glk, generate_son
from vfe3.geometry.lie_ops import embed_phi, retract_glk, retract_son


def test_retract_glk_trust_region_and_max_norm():
    G = generate_glk(3)                                   # (9,3,3)
    phi = 0.1 * torch.randn(5, 9)
    delta = 50.0 * torch.randn(5, 9)                      # huge -> both clamps active
    out = retract_glk(phi, delta, G, step_size=1.0, trust_region=0.1, max_norm=5.0)
    assert (out.norm(dim=-1) <= 5.0 + 1e-5).all()


def test_retract_glk_keeps_det_positive():
    # det(exp(embed phi)) = exp(tr) > 0 always: the GL+(K) identity-component property.
    G = generate_glk(3)
    phi = 0.3 * torch.randn(8, 9)
    delta = torch.randn(8, 9)
    out = retract_glk(phi, delta, G)
    dets = torch.linalg.det(torch.linalg.matrix_exp(embed_phi(out, G)))
    assert (dets > 0).all()


def test_retract_son_stays_orthogonal():
    # SO(N): embed(phi) is skew -> exp is orthogonal with det +1 (group membership).
    G = generate_son(4)                                   # (6,4,4)
    phi = 0.2 * torch.randn(7, 6)
    delta = torch.randn(7, 6)
    out = retract_son(phi, delta, G, max_norm=math.pi)
    A = embed_phi(out, G)
    assert torch.allclose(A, -A.transpose(-1, -2), atol=1e-5)          # skew
    R = torch.linalg.matrix_exp(A)
    eye = torch.eye(4).expand_as(R)
    assert torch.allclose(R @ R.transpose(-1, -2), eye, atol=1e-4)     # orthogonal
    assert torch.allclose(torch.linalg.det(R), torch.ones(7), atol=1e-4)
```

- [ ] **Step 2:** Run `python -m pytest tests/test_phi_retraction.py -q` — expect FAIL.

- [ ] **Step 3 (GREEN): implement** — append to `vfe3/geometry/lie_ops.py`:

```python
def _retract_core(
    phi:          torch.Tensor,           # (..., n_gen) current frame
    delta_phi:    torch.Tensor,           # (..., n_gen) tangent step direction

    *,
    step_size:    float = 1.0,
    trust_region: float = 0.1,
    max_norm:     float = 5.0,
    eps:          float = 1e-6,
    order:        int   = 4,
    mode:         str   = "euclidean",
    generators:   Optional[torch.Tensor]  = None,
    gram_pinv_:   Optional[torch.Tensor]  = None,
) -> torch.Tensor:
    r"""Shared retraction: scale -> trust-region clamp -> compose -> max-norm clamp.

      update   = clamp_||.|| ( step_size * delta_phi , trust_region )
      phi_new  = compose(phi, update; mode, order)
      phi_new <- clamp_||.|| ( phi_new , max_norm )
    Trust region and max norm are applied to the coordinate-vector norm.
    """
    update = step_size * delta_phi
    if trust_region is not None and trust_region > 0:
        u_norm = update.norm(dim=-1, keepdim=True)
        update = update * (trust_region / (u_norm + eps)).clamp(max=1.0)
    phi_new = compose_phi(phi, update, generators, order=order, mode=mode, gram_pinv_=gram_pinv_)
    if max_norm is not None and max_norm > 0:
        n_norm = phi_new.norm(dim=-1, keepdim=True)
        phi_new = torch.where(n_norm > max_norm, phi_new * (max_norm / (n_norm + eps)), phi_new)
    return phi_new


def retract_glk(
    phi:          torch.Tensor,           # (..., n_gen) current GL(K) frame
    delta_phi:    torch.Tensor,           # (..., n_gen) tangent step

    generators:   torch.Tensor,           # (n_gen, K, K)

    *,
    step_size:    float = 1.0,
    trust_region: float = 0.1,            # tighter than SO(N): GL(K) is non-compact
    max_norm:     float = 5.0,            # bounds singular values to ~[e^-5, e^5]
    eps:          float = 1e-6,
    order:        int   = 4,
    mode:         str   = "euclidean",
    gram_pinv_:   Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""GL(K) retraction (no det control here; the dispatcher applies it)."""
    return _retract_core(
        phi, delta_phi, step_size=step_size, trust_region=trust_region,
        max_norm=max_norm, eps=eps, order=order, mode=mode,
        generators=generators, gram_pinv_=gram_pinv_,
    )


def retract_son(
    phi:          torch.Tensor,           # (..., n_gen) current SO(N) frame
    delta_phi:    torch.Tensor,           # (..., n_gen) tangent step

    generators:   torch.Tensor,           # (n_gen, K, K) skew so(N) basis

    *,
    step_size:    float = 1.0,
    trust_region: float = 0.3,            # compact group
    max_norm:     float = math.pi,        # bounds principal angles
    eps:          float = 1e-6,
    order:        int   = 4,
    mode:         str   = "euclidean",
    gram_pinv_:   Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""SO(N) retraction. det(exp) = 1 automatic (skew generators)."""
    return _retract_core(
        phi, delta_phi, step_size=step_size, trust_region=trust_region,
        max_norm=max_norm, eps=eps, order=order, mode=mode,
        generators=generators, gram_pinv_=gram_pinv_,
    )
```

> Add `import math` to the module header. (`generators` is a required positional placed after the two tensor args; it is itself a tensor so it belongs in the leading tensor group — keep `phi, delta_phi, generators` contiguous with a blank line before `*` per style.)

- [ ] **Step 4:** Run — expect 3 passed.

- [ ] **Step 5 (COMMIT):**
```
git add vfe3/geometry/lie_ops.py tests/test_phi_retraction.py
git commit -m "feat(geometry): GL(K)/SO(N) phi retraction (group-membership + det>0 pinned)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4 — determinant control: `project_phi_to_slk` + `clamp_phi_trace`

**Files:** Modify `vfe3/geometry/lie_ops.py`; Test `tests/test_phi_retraction.py`.

- [ ] **Step 1 (RED): failing tests** — append to `tests/test_phi_retraction.py`:

```python
from vfe3.geometry.generators import generate_glk_multihead
from vfe3.geometry.lie_ops import clamp_phi_trace, project_phi_to_slk


def _block_traces(phi, G, irrep_dims):
    A = embed_phi(phi, G)
    outs, start = [], 0
    for d in irrep_dims:
        end = start + d
        outs.append(A[..., start:end, start:end].diagonal(dim1=-2, dim2=-1).sum(-1))
        start = end
    return torch.stack(outs, dim=-1)                      # (..., n_blocks)


def test_project_slk_zeros_block_trace_and_unit_det():
    G = generate_glk_multihead(6, 2)                      # 2 blocks of gl(3)
    irrep = [3, 3]
    phi = 0.5 * torch.randn(5, G.shape[0])
    out = project_phi_to_slk(phi, G, irrep)
    assert torch.allclose(_block_traces(out, G, irrep), torch.zeros(5, 2), atol=1e-5)
    # det of each block's group element == 1
    A = embed_phi(out, G)
    for s, d in [(0, 3), (3, 3)]:
        blk = A[..., s:s + d, s:s + d]
        det = torch.linalg.det(torch.linalg.matrix_exp(blk))
        assert torch.allclose(det, torch.ones(5), atol=1e-4)


def test_clamp_phi_trace_bounds_block_trace():
    G = generate_glk_multihead(6, 2)
    irrep = [3, 3]
    phi = 2.0 * torch.randn(5, G.shape[0])                # large traces
    T = 0.5
    out = clamp_phi_trace(phi, G, irrep, trace_max=T)
    assert (_block_traces(out, G, irrep).abs() <= T + 1e-4).all()
```

- [ ] **Step 2:** Run — expect FAIL.

- [ ] **Step 3 (GREEN): implement** — append to `vfe3/geometry/lie_ops.py`. `V[h,a] = tr(G_a |_{block h})`; the trace component of `phi` along block `h` is `s_h = phi · V_h`:

```python
def _block_trace_vectors(
    generators: torch.Tensor,             # (n_gen, K, K)
    irrep_dims: List[int],                # block sizes; sum == K

    *,
    eps:        float = 1e-12,
) -> torch.Tensor:                        # (n_blocks, n_gen) V[h,a] = tr(G_a|block h)
    r"""Per-block trace functionals V[h,a] = tr(G_a restricted to block h)."""
    rows, start = [], 0
    for d in irrep_dims:
        end = start + d
        rows.append(generators[:, start:end, start:end].diagonal(dim1=-2, dim2=-1).sum(-1))
        start = end
    return torch.stack(rows, dim=0)                       # (n_blocks, n_gen)


def project_phi_to_slk(
    phi:        torch.Tensor,             # (..., n_gen)
    generators: torch.Tensor,             # (n_gen, K, K)
    irrep_dims: List[int],                # block sizes; sum == K

    *,
    eps:        float = 1e-12,
) -> torch.Tensor:                        # (..., n_gen) per-block trace-free
    r"""Hard projection to sl(K) per block: remove the trace component so

        det(Omega_h) = exp(tr(embed(phi)|block h)) = 1.
    phi <- phi - sum_h (phi . V_h / ||V_h||^2) V_h.
    """
    V = _block_trace_vectors(generators, irrep_dims)      # (H, n_gen)
    v_norm_sq = (V * V).sum(-1).clamp(min=eps)            # (H,)
    s = phi @ V.transpose(-1, -2)                         # (..., H)
    coeffs = s / v_norm_sq                                # (..., H)
    return phi - torch.einsum("...h,hg->...g", coeffs, V)


def clamp_phi_trace(
    phi:        torch.Tensor,             # (..., n_gen)
    generators: torch.Tensor,             # (n_gen, K, K)
    irrep_dims: List[int],                # block sizes; sum == K

    *,
    trace_max:  float = 5.0,              # soft cap T on |tr(embed(phi)|block h)|
    eps:        float = 1e-12,
) -> torch.Tensor:                        # (..., n_gen) with |s_h| <= T
    r"""Soft per-block trace clamp: rescale only the trace component so |s_h| <= T,

    bounding log|det(Omega_h)|. Off-trace (sl(K)) directions are untouched.
    """
    V = _block_trace_vectors(generators, irrep_dims)      # (H, n_gen)
    v_norm_sq = (V * V).sum(-1).clamp(min=eps)            # (H,)
    s = phi @ V.transpose(-1, -2)                         # (..., H)
    s_clamped = s.clamp(min=-trace_max, max=trace_max)
    delta = (s_clamped - s) / v_norm_sq                   # (..., H)
    return phi + torch.einsum("...h,hg->...g", delta, V)
```

- [ ] **Step 4:** Run — expect 2 passed.

- [ ] **Step 5 (COMMIT):**
```
git add vfe3/geometry/lie_ops.py tests/test_phi_retraction.py
git commit -m "feat(geometry): phi determinant control (sl(K) projection + trace clamp)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5 — `retract_phi` dispatcher in `retraction.py`

**Files:** Modify `vfe3/geometry/retraction.py`; Test `tests/test_phi_retraction.py`.

- [ ] **Step 1 (RED): failing tests** — append to `tests/test_phi_retraction.py`:

```python
from vfe3.geometry.groups import get_group
from vfe3.geometry.retraction import retract_phi


def test_retract_phi_glk_with_slk_projection():
    grp = get_group("block_glk")(6, 2)                    # block GL(3)^2, irrep [3,3]
    phi = 0.5 * torch.randn(4, grp.generators.shape[0])
    delta = torch.randn_like(phi)
    out = retract_phi(phi, delta, grp, project_slk=True)
    assert torch.allclose(_block_traces(out, grp.generators, grp.irrep_dims),
                          torch.zeros(4, 2), atol=1e-4)


def test_retract_phi_son_path_orthogonal_no_det_control():
    grp = get_group("so_k")(4)
    phi = 0.2 * torch.randn(4, grp.generators.shape[0])
    delta = torch.randn_like(phi)
    out = retract_phi(phi, delta, grp)                    # skew -> SO path, det control ignored
    R = torch.linalg.matrix_exp(embed_phi(out, grp.generators))
    eye = torch.eye(4).expand_as(R)
    assert torch.allclose(R @ R.transpose(-1, -2), eye, atol=1e-4)


def test_retract_phi_defaults_pick_group_constants():
    # GL(K) default max_norm=5.0; a huge delta saturates to that, not pi.
    grp = get_group("glk")(3)
    phi = torch.zeros(2, 9)
    delta = 1e3 * torch.ones(2, 9)
    out = retract_phi(phi, delta, grp)
    assert (out.norm(dim=-1) <= 5.0 + 1e-4).all()
```

- [ ] **Step 2:** Run — expect FAIL (ImportError: retract_phi).

- [ ] **Step 3 (GREEN): implement** — append to `vfe3/geometry/retraction.py` (add imports `from vfe3.geometry.groups import GaugeGroup` and the needed `lie_ops` names + `Optional`):

```python
def retract_phi(
    phi:          torch.Tensor,           # (..., n_gen) current gauge frame
    delta_phi:    torch.Tensor,           # (..., n_gen) tangent step (e.g. -grad_phi)
    group:        GaugeGroup,             # supplies generators, skew flag, irrep_dims

    *,
    step_size:    float = 1.0,
    eps:          float = 1e-6,
    order:        int   = 4,
    project_slk:  bool  = False,
    mode:         str   = "euclidean",

    trust_region: Optional[float] = None, # None -> group default (GL:0.1, SO:0.3)
    max_norm:     Optional[float] = None, # None -> group default (GL:5.0, SO:pi)
    trace_clamp:  Optional[float] = None, # soft per-block |tr| cap (GL only)
) -> torch.Tensor:
    r"""Group-aware phi retraction dispatcher (Gaussian-specialized).

    Skew group (SO(N)) -> retract_son, det control is a no-op (det exp = 1).
    Non-skew (GL(K))   -> retract_glk, then optional det control:
      project_slk=True  hard-projects each block to sl(K) (det Omega_h = 1);
      else trace_clamp soft-bounds |tr| per block. Defaults for trust_region /
      max_norm are taken from the group's compactness when not given.
    """
    G = group.generators
    if trust_region is None:
        trust_region = 0.3 if group.skew_symmetric else 0.1
    if max_norm is None:
        max_norm = math.pi if group.skew_symmetric else 5.0

    if group.skew_symmetric:
        return retract_son(
            phi, delta_phi, G, step_size=step_size, trust_region=trust_region,
            max_norm=max_norm, eps=eps, order=order, mode=mode,
        )

    phi_new = retract_glk(
        phi, delta_phi, G, step_size=step_size, trust_region=trust_region,
        max_norm=max_norm, eps=eps, order=order, mode=mode,
    )
    if project_slk:
        phi_new = project_phi_to_slk(phi_new, G, group.irrep_dims)
    elif trace_clamp is not None:
        phi_new = clamp_phi_trace(phi_new, G, group.irrep_dims, trace_max=trace_clamp)
    return phi_new
```

> `retraction.py` currently has no `import math`; add it. Import `retract_glk, retract_son, project_phi_to_slk, clamp_phi_trace` from `vfe3.geometry.lie_ops`.

- [ ] **Step 4:** Run — expect 3 passed.

- [ ] **Step 5 (COMMIT):**
```
git add vfe3/geometry/retraction.py tests/test_phi_retraction.py
git commit -m "feat(geometry): retract_phi dispatcher (group-aware GL(K)/SO(N) + det control)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6 — full suite + final commit

- [ ] **Step 1:** `python -m pytest -q` — expect ALL prior + new tests pass (no regressions in divergence / gauge_groups / transport / retraction / config).
- [ ] **Step 2:** If anything fails, fix the implementation (never weaken an analytic assertion to pass).
- [ ] **Step 3 (COMMIT, if Task-by-task commits left anything uncommitted):**
```
git add -A
git commit -m "test(geometry): phase 2d phi-retraction suite green

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage (Phase 2d slice of §4.2 "the phi Lie-algebra retraction"):**
- φ↔matrix maps + robust (overcomplete-safe) coordinate extraction → Task 1.
- Composition seam `{euclidean, bch}` (registry; default euclidean; BCH residual-rate pinned) → Task 2.
- GL(K)/SO(N) retraction (trust region, max norm, group membership, det>0) → Task 3.
- Determinant control (sl(K) projection + soft trace clamp) → Task 4.
- Group-aware dispatcher wiring to `GaugeGroup` → Task 5.

**Hand-derived analytic anchors (independent of the implementation):**
- so(3) structure constants `[G0,G1]=-G2, [G0,G2]=+G1, [G1,G2]=-G0`.
- BCH log-log residual slope = `order+2` (pins series structure + low-order coefficients; catches missing/gross/structural terms). Exact high-order coefficients pinned by a direct coordinate check against the matrix-log reference (`test_bch_order4_coords_match_matrix_log`).
- `embed∘extract∘embed = embed` (overcomplete-safe; no hidden independence assumption).
- det(exp)>0 for GL(K); orthogonal + det +1 for SO(N); per-block det=1 after sl(K) projection; |tr|≤T after clamp.

**Deferred (named extension points, NOT built here):**
- **Phase 2e — φ-gradient preconditioner**: `precondition_phi_gradient` registry (`clip` default, `killing`/`killing_per_block` with the **center-regularized** Cartan metric `g̃=2K·gram−2·tr⊗tr` — singular on the trace direction, so `center_reg·P_center` before inversion; `pullback` natural gradient registered as a stub, 2.0-flagged OOM-infeasible for K≥15). Analytic anchor: the gl(2) Killing matrix `g̃=[[2,0,0,-2],[0,4,0,0],[0,0,4,0],[-2,0,0,2]]` (nullspace = I direction).
- **Exact group retraction** `U←U·exp(−ηΔ)`: the `omega_direct` transport partner (no single φ); a transport/2e concern.
- **Regime II** edge-relaxed cocycle `δ_ij`; **RoPE on μ**; **VFEHeadMixer**; the gap-regularized `_safe_eigh` backward.

**Placeholder scan:** none — every function is implemented and tested in-phase; `pullback` is explicitly Phase 2e.

**Type/name consistency:** `embed_phi(phi, generators)`, `extract_phi(matrix, generators, *, gram_pinv_)`, `lie_bracket_coords(phi1, phi2, generators, *, gram_pinv_)`, `compose_phi(phi1, phi2, generators, *, order, mode, gram_pinv_)`, `retract_glk/son(phi, delta_phi, generators, *, …)`, `project_phi_to_slk(phi, generators, irrep_dims, *, eps)`, `clamp_phi_trace(phi, generators, irrep_dims, *, trace_max, eps)`, `retract_phi(phi, delta_phi, group, *, …)` — consistent across tasks and tests.
