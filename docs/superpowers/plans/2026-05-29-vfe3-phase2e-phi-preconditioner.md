# VFE_3.0 Phase 2e (φ-Gradient Preconditioner) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:test-driven-development (RED→GREEN→COMMIT per step). Tests are **V3-internal** (analytic known-value + property + finite-difference). V3 is self-contained; see the Provenance rule below.

**Goal:** Build `vfe3/geometry/phi_preconditioner.py` — the φ-gradient preconditioner seam that conditions a Euclidean φ-gradient before the retraction. A config-selected registry of metrics: `none` (default), `clip`, `killing`, `killing_per_block`, and the position-dependent `pullback` natural gradient. Coordinate-space in/out, so it composes directly with Phase 2d's `retract_phi` (the E-step is `grad → precondition → retract`).

**Architecture:** Fifth geometry sub-phase, the φ analog of Phase 2c's `(μ,σ)` Fisher `natural_gradient`. `phi_preconditioner.py` is **pure** (operates on a `grad_phi`/`phi` coordinate tensor + a generator tensor + optional `irrep_dims`; it does NOT import `GaugeGroup`). The metric layer is a registry so a variant is added by writing-and-registering, never by editing call sites. The **canonical default is `none`** (the manuscript applies no metric correction — the gradient lives in the Lie algebra 𝔤, a vector space); `clip`/`killing`/`pullback` are toggles preserving a theoretically pure (position-dependent natural-gradient) path.

**Tech Stack:** Python 3, PyTorch (float32 storage; float64 internal where conditioning demands it — SPD eigendecomposition forced float64), pytest. No NN. No CLI.

**Reference spec:** `docs/superpowers/specs/2026-05-29-vfe3-clean-room-design.md` (§4.2 — "the Fisher / natural-gradient preconditioner"). Prereq: Phases 2a–2d on `main` (this branch: `phase2e-phi-preconditioner`). Reuses Phase 2d `lie_ops`: `embed_phi`, `extract_phi`, `gram_pinv`, `lie_bracket_coords`, and `transport.stable_matrix_exp_pair`.

**Manuscript theory (authority):**
- The canonical φ update applies **no metric correction**: "the exponential map exp: 𝔤 → G provides natural coordinates … the gradient naturally lives in 𝔤" (`GL(K)_supplementary.tex` §Gauge Frame Preconditioning). The preconditioners below are the manuscript's four named *options* (norm-clip baseline; Cartan-decomposition damping; Killing/Cartan-involution metric; pullback natural gradient).
- **Killing / Cartan-involution metric:** `g̃_ab = 2K·tr(Tₐᵀ T_b) − 2·tr(Tₐ)·tr(T_b)`, positive-definite on sl(K) (`GL(K)_supplementary.tex`, eq. killing_metric). It is `−B(θX,Y)` with the Cartan involution θ(X) = −Xᵀ; the **bare** Killing form `B(X,Y)=2K·tr(XY)−2tr(X)tr(Y)` of gl(K) is **indefinite** and is deliberately NOT used.
- **Pullback natural gradient (position-dependent):** `G_ab(φ) = ⟨d exp_φ(Tₐ), d exp_φ(T_b)⟩_F` where `d exp_φ(T) = Ψ(ad_φ)(T)·exp(φ)`, `Ψ(z) = (eᶻ−1)/z = Σ_{k≥0} zᵏ/(k+1)!`, `ad_φ(T_b)=[φ,T_b]`. On symmetric directions `exp(φ)exp(φ)ᵀ` grows to compensate the exponential amplification of `d exp`.
- Gauge-equivariance of the φ-flow under `U_i ↦ U_i g` (`Participatory_it_from_bit.tex` ll. 2718–2722) — the property the preconditioned update preserves.

**Design decisions settled before this plan (do not relitigate in code):**
1. **`gram` is the Frobenius inner product** `gram_ab = tr(Tₐᵀ T_b) = Σ_ij G_a[i,j]G_b[i,j]`. The Killing metric is the **Cartan-involution form** `2K·gram − 2·tr⊗tr`. The bare Killing `tr(Tₐ T_b)` is indefinite — using it is a bug. The literal gl(2) anchor below discriminates the two.
2. **Center-regularization regularizes the numerical nullspace**, not a hardcoded direction: eigendecompose `g̃` (eigh, float64), lift eigenvalues with `|λ| < tol` to `center_reg` (default `2K`), then invert. This (a) makes the inverse **exact on sl(K)** (non-null eigenvalues untouched — a ridge `center_reg·I` would NOT), and (b) leaves so(K) (already PD, no near-null eigenvalue) **unregularized**. The center direction need not be identified explicitly.
3. **Default mode is `none`** (manuscript: no correction). `clip` is the practical robustness baseline; `killing`/`killing_per_block`/`pullback` are principled toggles.
4. **Pullback is included, FD-pinned and K-guarded.** The finite-difference of `exp` (`Jₐ = ∂_ε exp(embed(φ+ε eₐ))`, `G_FD[a,b]=tr(JₐᵀJ_b)`) is the **arbiter**; hardcode nothing about operator ordering — let the FD check validate the Ψ-series. Guard `K > max_k` (default 12; the structure-constants tensor is O(n_gen²·K²), OOM for K≥15).
5. **Coords-in/coords-out seam:** `precondition_phi_gradient(grad_phi, phi, generators, *, …) -> (...,n_gen)`, same units as `retract_phi`'s `delta_phi`.

---

## Code Style (MANDATORY — repo CLAUDE.md)

Argument order: tensors first; then `float|Tensor`; then undefined floats/ints/bools; then defined floats, defined ints, defined bools (defined `str` with the defined scalars); then `Optional`; then `**kwargs`. Vertical alignment of names / type annotations / `=` / trailing `#` comments; blank lines separate type groups; shape comments at critical points; type hints everywhere; docstrings carry the LaTeX/math form. Names match paper notation (`grad_phi`, `phi`, `generators`, `irrep_dims`, `center_reg`).

## Provenance (MANDATORY — convention as of commit `114839c`)

V3 is a self-contained implementation, **not a port**. **No shipped artifact — docstring, comment, test name, or test comment — may mention "VFE_2.0", "2.0", "ported", or any 2.0 file/line.** Cite only the manuscript (`GL(K)_attention.tex`, `GL(K)_supplementary.tex`, `Participatory_it_from_bit.tex`) and the math.

---

## File Structure

- **Create** `vfe3/geometry/phi_preconditioner.py` — `killing_metric`, `build_killing_preconditioner`, `build_killing_preconditioner_per_block`, `pullback_metric`, `register_precond`/`get_precond`, the `none`/`clip`/`killing`/`killing_per_block`/`pullback` rules, and the `precondition_phi_gradient` dispatcher.
- **Create** `tests/test_phi_preconditioner.py` — clip/none, the gl(2) Killing literal + so(K) PD tell, center-reg exact-inverse-on-sl(K), per-block block-diagonal structure, pullback φ=0==Gram + FD-of-exp match + K-guard.

---

## Task 1 — registry + `none` + `clip` + dispatcher

**Files:** Create `vfe3/geometry/phi_preconditioner.py`; Test `tests/test_phi_preconditioner.py`.

- [ ] **Step 1 (RED):** create `tests/test_phi_preconditioner.py`:

```python
import math

import torch

from vfe3.geometry.generators import generate_glk, generate_glk_multihead, generate_son
from vfe3.geometry.phi_preconditioner import precondition_phi_gradient


def test_none_is_identity():
    G = generate_glk(3)
    grad = torch.randn(4, 9)
    out = precondition_phi_gradient(grad, torch.zeros(4, 9), G, mode="none")
    assert torch.allclose(out, grad, atol=1e-7)


def test_clip_scales_large_gradient_to_c():
    G = generate_glk(3)
    grad = 100.0 * torch.ones(2, 9)                       # norm >> c
    out = precondition_phi_gradient(grad, torch.zeros(2, 9), G, mode="clip", clip_c=10.0)
    assert torch.allclose(out.norm(dim=-1), torch.full((2,), 10.0), atol=1e-3)


def test_clip_leaves_small_gradient_unchanged():
    G = generate_glk(3)
    grad = 0.01 * torch.ones(2, 9)                        # norm << c
    out = precondition_phi_gradient(grad, torch.zeros(2, 9), G, mode="clip", clip_c=10.0)
    assert torch.allclose(out, grad, atol=1e-7)
```

- [ ] **Step 2:** Run `python -m pytest tests/test_phi_preconditioner.py -q` — expect FAIL (ImportError).

- [ ] **Step 3 (GREEN):** create `vfe3/geometry/phi_preconditioner.py`:

```python
r"""Gauge-frame (phi) gradient preconditioner for VFE_3.0 (Gaussian-specialized).

Conditions a Euclidean gradient grad_phi (coordinates in a generator basis) before
the Lie-algebra retraction. A config-selected registry of metrics:
  none              identity (the canonical update: no metric correction; the
                    gradient lives in the Lie algebra g, a vector space).
  clip              norm-clip baseline grad * min(1, c / ||grad||).
  killing           Cartan-involution metric g~ = 2K*gram - 2*tr(x)tr(.), center-
                    regularized then inverted (natural gradient grad @ g~^{-1}).
  killing_per_block block-diagonal Killing metric (per irrep block).
  pullback          position-dependent natural gradient via the differential of
                    the exponential map: G_ab(phi) = <d exp_phi(T_a), d exp_phi(T_b)>_F.
Coordinates in, coordinates out (..., n_gen) -- same units as retract_phi's
delta_phi, so the E-step is grad -> precondition -> retract. Pure: takes a
generator TENSOR, not a GaugeGroup.
"""

from typing import Callable, Dict, List, Optional

import torch

from vfe3.geometry.lie_ops import embed_phi, extract_phi, gram_pinv

_PRECOND: Dict[str, Callable[..., torch.Tensor]] = {}


def register_precond(name: str) -> Callable:
    """Decorator registering a preconditioning rule grad_phi -> preconditioned grad."""
    def _wrap(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        _PRECOND[name] = fn
        return fn
    return _wrap


def get_precond(name: str) -> Callable[..., torch.Tensor]:
    """Return the registered preconditioning rule (KeyError if absent)."""
    if name not in _PRECOND:
        raise KeyError(f"no preconditioner {name!r}; available: {sorted(_PRECOND)}")
    return _PRECOND[name]


@register_precond("none")
def _precond_none(
    grad_phi:   torch.Tensor,             # (..., n_gen)
    phi:        torch.Tensor,             # (..., n_gen) (unused)
    generators: torch.Tensor,             # (n_gen, K, K) (unused)

    **kwargs,
) -> torch.Tensor:
    r"""Identity: the canonical no-correction update (gradient lives in g)."""
    return grad_phi


@register_precond("clip")
def _precond_clip(
    grad_phi:   torch.Tensor,             # (..., n_gen)
    phi:        torch.Tensor,             # (..., n_gen) (unused)
    generators: torch.Tensor,             # (n_gen, K, K) (unused)

    *,
    clip_c:     float = 10.0,
    eps:        float = 1e-6,
    **kwargs,
) -> torch.Tensor:
    r"""Norm-clip baseline grad * min(1, clip_c / ||grad||)."""
    norm = grad_phi.norm(dim=-1, keepdim=True)
    return torch.where(norm > clip_c, grad_phi * (clip_c / (norm + eps)), grad_phi)


def precondition_phi_gradient(
    grad_phi:     torch.Tensor,           # (..., n_gen) Euclidean grad wrt phi coords
    phi:          torch.Tensor,           # (..., n_gen) current frame (used by pullback)
    generators:   torch.Tensor,           # (n_gen, K, K)

    *,
    clip_c:       float = 10.0,
    series_order: int   = 6,
    mode:         str   = "none",

    center_reg:   Optional[float]        = None,   # None -> 2*K
    irrep_dims:   Optional[List[int]]    = None,   # required for killing_per_block
    inv_metric:   Optional[torch.Tensor] = None,   # cached Killing inverse (n_gen, n_gen)
) -> torch.Tensor:                        # (..., n_gen) preconditioned gradient
    r"""Dispatch to the registered preconditioning rule `mode` (default 'none')."""
    return get_precond(mode)(
        grad_phi, phi, generators,
        clip_c=clip_c, series_order=series_order, center_reg=center_reg,
        irrep_dims=irrep_dims, inv_metric=inv_metric,
    )
```

> Each registered rule accepts `**kwargs` so the dispatcher can forward a uniform argument set; a rule reads only the knobs it needs.

- [ ] **Step 4:** Run — expect 3 passed.
- [ ] **Step 5 (COMMIT):**
```
git add vfe3/geometry/phi_preconditioner.py tests/test_phi_preconditioner.py
git commit -m "feat(geometry): phi preconditioner registry (none default + clip)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2 — Killing (Cartan-involution) metric + center-regularized inverse

**Files:** Modify `vfe3/geometry/phi_preconditioner.py`; Test `tests/test_phi_preconditioner.py`.

- [ ] **Step 1 (RED):** append tests. The gl(2) literal is the discriminating anchor (bare Killing fails it); the so(3) value is the positivity tell:

```python
from vfe3.geometry.phi_preconditioner import build_killing_preconditioner, killing_metric


def test_killing_metric_gl2_exact():
    # Cartan-involution form 2K*gram - 2*tr(x)tr(.) on the gl(2) elementary basis
    # (E00,E01,E10,E11), K=2. Bare Killing tr(T_a T_b) would give a DIFFERENT,
    # indefinite matrix; this literal discriminates them.
    G = generate_glk(2)
    M = killing_metric(G)
    expected = torch.tensor([[2., 0., 0., -2.],
                             [0., 4., 0.,  0.],
                             [0., 0., 4.,  0.],
                             [-2., 0., 0., 2.]])
    assert torch.allclose(M, expected, atol=1e-5)
    evals = torch.linalg.eigvalsh(M)
    assert torch.allclose(evals.sort().values, torch.tensor([0., 4., 4., 4.]), atol=1e-4)


def test_killing_metric_so3_is_positive_definite():
    # so(3): skew generators have tr=0 so g~ = 2K*gram. gram = 2*I (||L_ij||_F^2=2),
    # K=3 -> g~ = 12*I, POSITIVE-definite. (Bare Killing is negative-definite on skew.)
    G = generate_son(3)
    M = killing_metric(G)
    assert torch.allclose(M, 12.0 * torch.eye(3), atol=1e-4)
    assert (torch.linalg.eigvalsh(M) > 0).all()


def test_killing_preconditioner_exact_inverse_on_slk():
    # Center-reg lifts ONLY the numerical nullspace (the center/identity direction);
    # on sl(K) the regularized inverse is the TRUE Killing inverse: g~ @ (Minv @ v) = v
    # for v perpendicular to the center. A ridge center_reg*I would fail this.
    G = generate_glk(2)
    M = killing_metric(G)
    Minv = build_killing_preconditioner(G, center_reg=4.0)
    v = torch.tensor([0., 1.3, -0.7, 0.])                 # in sl(2): E01,E10 dirs (trace-free)
    assert torch.allclose(M @ (Minv @ v), v, atol=1e-5)
    # full regularized metric is PD (gl(2): eigenvalues {0,4,4,4} -> {4,4,4,4})
    reg = torch.linalg.inv(Minv)
    assert (torch.linalg.eigvalsh(reg) > 1e-6).all()


def test_killing_mode_applies_inverse_metric():
    G = generate_glk(2)
    grad = torch.tensor([[0., 2.0, -1.0, 0.]])
    Minv = build_killing_preconditioner(G, center_reg=4.0)
    out = precondition_phi_gradient(grad, torch.zeros(1, 4), G, mode="killing", center_reg=4.0)
    assert torch.allclose(out, grad @ Minv, atol=1e-6)
```

- [ ] **Step 2:** Run — expect FAIL (ImportError).

- [ ] **Step 3 (GREEN):** append to `phi_preconditioner.py`:

```python
def killing_metric(
    generators: torch.Tensor,             # (n_gen, K, K) basis
) -> torch.Tensor:                        # (n_gen, n_gen) Cartan-involution metric
    r"""Cartan-involution Killing metric g~_ab = 2K*tr(G_a^T G_b) - 2 tr(G_a) tr(G_b).

    Equals -B(theta X, Y) with theta(X) = -X^T; positive-definite on sl(K). The
    bare Killing form B(X,Y) = 2K*tr(XY) - 2 tr(X)tr(Y) is indefinite and is NOT
    used. ``gram`` is the FROBENIUS inner product tr(G_a^T G_b).
    """
    K = generators.shape[-1]
    gram   = torch.einsum("aij,bij->ab", generators, generators)      # tr(G_a^T G_b)
    traces = generators.diagonal(dim1=-2, dim2=-1).sum(-1)            # (n_gen,)
    return 2.0 * K * gram - 2.0 * torch.outer(traces, traces)


def build_killing_preconditioner(
    generators: torch.Tensor,             # (n_gen, K, K) basis

    *,
    center_reg: Optional[float] = None,   # None -> 2*K; lifts the numerical nullspace
    tol:        float           = 1e-6,
) -> torch.Tensor:                        # (n_gen, n_gen) regularized inverse metric
    r"""Inverse Killing metric, regularizing only the numerical nullspace.

    eigh(g~) -> (lambda, V); eigenvalues with |lambda| < tol (the center/identity
    direction) are lifted to ``center_reg`` before inversion. Non-null eigenvalues
    are untouched, so the inverse is EXACT on sl(K) (a ridge center_reg*I is not).
    so(K) (already positive-definite) acquires no regularization. eigh in float64.
    """
    K = generators.shape[-1]
    reg = float(2 * K) if center_reg is None else float(center_reg)
    orig_dtype = generators.dtype
    M = killing_metric(generators).double()
    M = 0.5 * (M + M.transpose(-1, -2))
    evals, evecs = torch.linalg.eigh(M)
    evals = torch.where(evals.abs() < tol, torch.full_like(evals, reg), evals)
    inv = (evecs * (1.0 / evals).unsqueeze(-2)) @ evecs.transpose(-1, -2)
    return inv.to(orig_dtype)


@register_precond("killing")
def _precond_killing(
    grad_phi:   torch.Tensor,             # (..., n_gen)
    phi:        torch.Tensor,             # (..., n_gen) (unused)
    generators: torch.Tensor,             # (n_gen, K, K)

    *,
    center_reg: Optional[float]        = None,
    inv_metric: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    r"""Natural gradient under the (center-regularized) Killing metric: grad @ g~^{-1}."""
    Minv = build_killing_preconditioner(generators, center_reg=center_reg) if inv_metric is None else inv_metric
    return torch.einsum("...a,ab->...b", grad_phi, Minv)
```

- [ ] **Step 4:** Run — expect 4 passed.
- [ ] **Step 5 (COMMIT):**
```
git add vfe3/geometry/phi_preconditioner.py tests/test_phi_preconditioner.py
git commit -m "feat(geometry): Killing (Cartan-involution) preconditioner, nullspace-regularized

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3 — per-block Killing preconditioner

**Files:** Modify `vfe3/geometry/phi_preconditioner.py`; Test `tests/test_phi_preconditioner.py`.

- [ ] **Step 1 (RED):** append:

```python
from vfe3.geometry.phi_preconditioner import build_killing_preconditioner_per_block


def test_per_block_is_block_diagonal_and_matches_local_killing():
    # block_glk = gl(2) (+) gl(2): generators grouped by head (4 per head). The
    # per-block metric couples only same-block generators and uses the local
    # block dimension d_h=2, so the (8,8) inverse is block-diagonal in 4+4.
    G = generate_glk_multihead(4, 2)                      # 8 generators, irrep [2,2]
    irrep = [2, 2]
    Minv = build_killing_preconditioner_per_block(G, irrep, center_reg=4.0)
    # off-diagonal cross-block coupling is zero
    assert torch.allclose(Minv[:4, 4:], torch.zeros(4, 4), atol=1e-6)
    assert torch.allclose(Minv[4:, :4], torch.zeros(4, 4), atol=1e-6)
    # each diagonal block equals the single-head gl(2) Killing inverse
    head0 = generate_glk(2)
    Minv_head = build_killing_preconditioner(head0, center_reg=4.0)
    assert torch.allclose(Minv[:4, :4], Minv_head, atol=1e-5)


def test_killing_per_block_mode():
    G = generate_glk_multihead(4, 2)
    irrep = [2, 2]
    grad = torch.randn(3, 8)
    Minv = build_killing_preconditioner_per_block(G, irrep, center_reg=4.0)
    out = precondition_phi_gradient(grad, torch.zeros(3, 8), G, mode="killing_per_block",
                                    irrep_dims=irrep, center_reg=4.0)
    assert torch.allclose(out, grad @ Minv, atol=1e-6)
```

- [ ] **Step 2:** Run — expect FAIL.

- [ ] **Step 3 (GREEN):** append. Each generator belongs to exactly one irrep block (its support lies in that block's K-rows/cols); group generators by block, build the local d_h-Killing on each group, and assemble a block-diagonal inverse in generator-index order. Assert each generator is single-block (cross-coupled bases — `irrep_dims=[K]` — fall back to the single global block, i.e. `build_killing_preconditioner`):

```python
def _generator_block_index(
    generators: torch.Tensor,             # (n_gen, K, K)
    irrep_dims: List[int],                # block sizes; sum == K

    *,
    tol:        float = 1e-9,
) -> torch.Tensor:                        # (n_gen,) block id per generator
    r"""Block membership of each generator (asserts single-block support)."""
    bounds, start = [], 0
    for d in irrep_dims:
        bounds.append((start, start + d)); start += d
    n_gen = generators.shape[0]
    block_of = torch.full((n_gen,), -1, dtype=torch.long)
    for a in range(n_gen):
        mass = []
        for (s, e) in bounds:
            mass.append(float(generators[a, s:e, s:e].abs().sum()))
        total = float(generators[a].abs().sum())
        hits = [h for h, m in enumerate(mass) if m > tol]
        if len(hits) != 1 or abs(sum(mass) - total) > tol:
            raise ValueError(f"generator {a} is not supported in a single irrep block")
        block_of[a] = hits[0]
    return block_of


def build_killing_preconditioner_per_block(
    generators: torch.Tensor,             # (n_gen, K, K)
    irrep_dims: List[int],                # block sizes; sum == K

    *,
    center_reg: Optional[float] = None,
    tol:        float           = 1e-6,
) -> torch.Tensor:                        # (n_gen, n_gen) block-diagonal inverse metric
    r"""Block-diagonal Killing inverse: per-block local-dimension Cartan metric.

    Single global block (irrep_dims == [K], e.g. cross-coupled bases) reduces to
    build_killing_preconditioner. Otherwise each generator's own block supplies
    the local Killing metric (block dimension d_h), with no cross-block coupling.
    """
    if len(irrep_dims) == 1:
        return build_killing_preconditioner(generators, center_reg=center_reg, tol=tol)
    block_of = _generator_block_index(generators, irrep_dims)
    n_gen = generators.shape[0]
    Minv = torch.zeros(n_gen, n_gen, dtype=generators.dtype)
    start = 0
    for h, d in enumerate(irrep_dims):
        idx = (block_of == h).nonzero(as_tuple=True)[0]
        sub = generators[idx][:, start:start + d, start:start + d].contiguous()   # local d_h rep
        sub_inv = build_killing_preconditioner(sub, center_reg=center_reg, tol=tol)
        Minv[idx.unsqueeze(-1), idx.unsqueeze(0)] = sub_inv
        start += d
    return Minv


@register_precond("killing_per_block")
def _precond_killing_per_block(
    grad_phi:   torch.Tensor,             # (..., n_gen)
    phi:        torch.Tensor,             # (..., n_gen) (unused)
    generators: torch.Tensor,             # (n_gen, K, K)

    *,
    center_reg: Optional[float]     = None,
    irrep_dims: Optional[List[int]] = None,
    inv_metric: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    r"""Natural gradient under the per-block Killing metric."""
    if inv_metric is None:
        if irrep_dims is None:
            raise ValueError("killing_per_block requires irrep_dims")
        inv_metric = build_killing_preconditioner_per_block(generators, irrep_dims, center_reg=center_reg)
    return torch.einsum("...a,ab->...b", grad_phi, inv_metric)
```

- [ ] **Step 4:** Run — expect 2 passed.
- [ ] **Step 5 (COMMIT):**
```
git add vfe3/geometry/phi_preconditioner.py tests/test_phi_preconditioner.py
git commit -m "feat(geometry): per-block Killing preconditioner (block-diagonal natural gradient)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4 — pullback natural gradient (position-dependent), FD-pinned

**Files:** Modify `vfe3/geometry/phi_preconditioner.py`; Test `tests/test_phi_preconditioner.py`.

- [ ] **Step 1 (RED):** append. The **finite-difference of exp is the arbiter** — it validates the entire Ψ-series and operator ordering independently:

```python
from vfe3.geometry.lie_ops import embed_phi
from vfe3.geometry.phi_preconditioner import pullback_metric


def _fd_dexp_metric(phi_vec, G, eps=1e-4):
    # Independent oracle: G_ab = <d exp_phi(e_a), d exp_phi(e_b)>_F via central
    # finite differences of matrix_exp. d exp_phi(e_a) = d/dt exp(embed(phi + t e_a)).
    n = G.shape[0]
    J = []
    for a in range(n):
        ea = torch.zeros(n, dtype=torch.float64); ea[a] = 1.0
        plus  = torch.linalg.matrix_exp(embed_phi((phi_vec + eps * ea), G))
        minus = torch.linalg.matrix_exp(embed_phi((phi_vec - eps * ea), G))
        J.append((plus - minus) / (2 * eps))
    J = torch.stack(J, 0)                                 # (n, K, K)
    return torch.einsum("aij,bij->ab", J, J)              # tr(J_a^T J_b)


def test_pullback_at_zero_is_frobenius_gram():
    # Psi(0) = I, exp(0) = I -> G_ab = tr(G_a^T G_b) = Frobenius Gram.
    G = generate_glk(2)
    Gmetric = pullback_metric(torch.zeros(4), G)
    gram = torch.einsum("aij,bij->ab", G, G)
    assert torch.allclose(Gmetric, gram, atol=1e-5)


def test_pullback_matches_finite_difference_of_exp():
    # The genuine independent check: closed Psi-series vs FD-of-exp. Validates the
    # operator ordering and every series coefficient.
    G = generate_son(3).double()                          # K=3, compact -> well-behaved
    phi = torch.tensor([0.4, -0.3, 0.5], dtype=torch.float64)
    Gclosed = pullback_metric(phi, G, series_order=10)
    Gfd = _fd_dexp_metric(phi, G)
    assert torch.allclose(Gclosed, Gfd, atol=1e-4)
    # symmetric PD
    assert torch.allclose(Gclosed, Gclosed.transpose(-1, -2), atol=1e-6)
    assert (torch.linalg.eigvalsh(Gclosed) > 0).all()


def test_pullback_k_guard():
    import pytest
    G = generate_glk(13)                                  # K=13 > max_k
    with pytest.raises((ValueError, RuntimeError)):
        pullback_metric(torch.zeros(169), G, max_k=12)
```

- [ ] **Step 2:** Run — expect FAIL.

- [ ] **Step 3 (GREEN):** append. Structure constants `f[a,b,c]` (coords of `[G_a,G_b]`) reuse the bracket+extract machinery; `ad_phi` is the adjoint matrix; `Ψ(ad_phi)` is the truncated Taylor series; `d exp_phi(e_a)` in coords is column `a` of `Ψ(ad_phi)`, embedded and right-multiplied by `exp(phi)`. **Let the FD test arbitrate the ordering — do not assume it from memory.**

```python
def _structure_constants(
    generators: torch.Tensor,             # (n_gen, K, K)

    *,
    gram_pinv_: Optional[torch.Tensor] = None,
) -> torch.Tensor:                        # (n_gen, n_gen, n_gen) f[a,b,c]: [G_a,G_b]=sum_c f G_c
    r"""Structure constants f[a,b,c] = coords_c([G_a, G_b]) in the generator basis."""
    G = generators
    brak = torch.einsum("aij,bjk->abik", G, G) - torch.einsum("bij,ajk->abik", G, G)   # [G_a,G_b]
    gp = gram_pinv(G) if gram_pinv_ is None else gram_pinv_
    coords = torch.einsum("cij,abij->abc", G, brak)       # <G_c, [G_a,G_b]>
    return torch.einsum("abc,cd->abd", coords, gp)


def pullback_metric(
    phi:          torch.Tensor,           # (..., n_gen) frame coordinates
    generators:   torch.Tensor,           # (n_gen, K, K)

    *,
    series_order: int = 6,
    max_k:        int = 12,
) -> torch.Tensor:                        # (..., n_gen, n_gen) position-dependent metric
    r"""Pullback natural-gradient metric G_ab(phi) = <d exp_phi(T_a), d exp_phi(T_b)>_F.

    d exp_phi(T) = Psi(ad_phi)(T) exp(phi), Psi(z) = (e^z - 1)/z = sum_k z^k/(k+1)!.
    ad_phi acts on coordinates: (ad_phi)_{cb} = sum_a phi^a f[a,b,c]. The structure-
    constants tensor is O(n_gen^2 K^2); guarded for K > max_k (infeasible for large K).
    The finite-difference of exp is the correctness arbiter for this kernel.
    """
    K = generators.shape[-1]
    if K > max_k:
        raise ValueError(f"pullback_metric: K={K} exceeds max_k={max_k} (structure-constants OOM)")
    n_gen = generators.shape[0]
    orig_dtype = phi.dtype
    G = generators.double()
    phi = phi.double()

    f = _structure_constants(G)                            # (n_gen,n_gen,n_gen) f[a,b,c]
    ad = torch.einsum("...a,abc->...cb", phi, f)           # (...,n_gen,n_gen) (ad_phi)_{cb}

    eye = torch.eye(n_gen, dtype=ad.dtype).expand_as(ad)
    psi = eye.clone()
    ad_pow = eye.clone()
    for k in range(1, series_order):
        ad_pow = torch.einsum("...ij,...jk->...ik", ad_pow, ad)
        psi = psi + ad_pow / math.factorial(k + 1)

    # d exp_phi(e_a) coords = psi @ e_a = column a of psi -> embed -> times exp(phi)
    W = torch.einsum("...ca,cij->...aij", psi, G)          # (...,n_gen,K,K) Psi(ad_phi)(T_a)
    exp_phi = torch.linalg.matrix_exp(torch.einsum("...a,aij->...ij", phi, G))
    dexp = torch.einsum("...aij,...jk->...aik", W, exp_phi)
    metric = torch.einsum("...aij,...bij->...ab", dexp, dexp)
    return metric.to(orig_dtype)


@register_precond("pullback")
def _precond_pullback(
    grad_phi:     torch.Tensor,           # (..., n_gen)
    phi:          torch.Tensor,           # (..., n_gen)
    generators:   torch.Tensor,           # (n_gen, K, K)

    *,
    series_order: int   = 6,
    eps:          float = 1e-6,
    **kwargs,
) -> torch.Tensor:
    r"""Position-dependent natural gradient: solve G(phi) nat = grad_phi."""
    G_metric = pullback_metric(phi, generators, series_order=series_order)
    eye = torch.eye(G_metric.shape[-1], dtype=G_metric.dtype, device=G_metric.device)
    sol = torch.linalg.solve(G_metric + eps * eye, grad_phi.unsqueeze(-1))
    return sol.squeeze(-1)
```

- [ ] **Step 4:** Run — expect 4 passed. (If `test_pullback_matches_finite_difference_of_exp` fails, the Ψ-series/ordering is wrong — re-derive against `d exp_φ(T)=Ψ(ad_φ)(T)exp(φ)`; the FD is right, do not touch the test.)
- [ ] **Step 5 (COMMIT):**
```
git add vfe3/geometry/phi_preconditioner.py tests/test_phi_preconditioner.py
git commit -m "feat(geometry): pullback natural-gradient preconditioner (FD-of-exp pinned)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5 — full suite + final verification

- [ ] **Step 1:** `python -m pytest -q` from the repo root — expect all prior (68) + new (~13) green, no regressions.
- [ ] **Step 2:** If anything fails, fix the implementation (never weaken an analytic/FD assertion).
- [ ] **Step 3 (COMMIT, if needed):**
```
git add -A
git commit -m "test(geometry): phase 2e phi-preconditioner suite green

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage (Phase 2e slice of §4.2 "Fisher / natural-gradient preconditioner", φ side):**
- Registry seam `none`/`clip`/`killing`/`killing_per_block`/`pullback` → Tasks 1–4.
- Cartan-involution Killing metric, center-regularized inverse → Task 2.
- Per-block (direct-sum) variant → Task 3.
- Position-dependent natural gradient (pullback) → Task 4.

**Hand-derived analytic anchors (independent of the implementation):**
- gl(2) Killing literal `[[2,0,0,-2],[0,4,0,0],[0,0,4,0],[-2,0,0,2]]` (discriminates Cartan-involution vs bare Killing), eigenvalues `{0,4,4,4}`.
- so(3) Killing `= 12·I`, positive-definite (the bare-Killing-is-negative-on-skew sign tell).
- Exact Killing inverse on sl(K) after nullspace regularization (ridge would fail).
- Per-block block-diagonal structure; each block = local-dimension Killing.
- Pullback `@φ=0 == Frobenius Gram`; pullback `== FD-of-exp` (independent oracle); symmetric PD.

**Deferred (named extension points, NOT built here):**
- Cartan-decomposition damping (manuscript Option 2: `M = I − (1−λ_sym)P_sym`, λ_sym≈0.1).
- Structure-constants caching / large-K pullback (currently K-guarded).
- The group-aware overload (`precondition_phi_gradient(group, …)`) and E-step wiring (Phase 6).
- The exact group retraction `U←U·exp(−ηΔ)` (Phase 2d's named `omega_direct` partner).

**Placeholder scan:** none — every mode is implemented and tested in-phase.

**Type/name consistency:** `precondition_phi_gradient(grad_phi, phi, generators, *, clip_c, series_order, mode, center_reg, irrep_dims, inv_metric)`, `killing_metric(generators)`, `build_killing_preconditioner(generators, *, center_reg, tol)`, `build_killing_preconditioner_per_block(generators, irrep_dims, *, center_reg, tol)`, `pullback_metric(phi, generators, *, series_order, max_k)` — consistent across tasks and tests. Coords-in/coords-out `(...,n_gen)` matches `retract_phi`'s `delta_phi` (E-step: grad → precondition → retract).
