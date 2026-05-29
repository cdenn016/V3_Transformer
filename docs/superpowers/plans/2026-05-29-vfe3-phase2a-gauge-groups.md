# VFE_3.0 Phase 2a (Gauge Groups + Generators) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `vfe3/geometry/` group layer — generator construction (`GL(K)`, block-diagonal `GL(K)=⊕GL(d_h)`, `SO(K)`, cross-head coupling), Lie-bracket closure, and a config-selectable `GaugeGroup` registry — proven numerically equal to VFE_2.0's `math_utils/generators/` by golden tests.

**Architecture:** First sub-phase of the geometry layer. Establishes the *structure-group* axis (one of the two modular axes in spec §4.2). Transport (Regime I), the belief action, retractions (Regime II), and RoPE/head-mixer follow in Phase 2b/2c/2d. The group is the dispatch object transport consumes; cross-coupling and Lie closure extend the block-diagonal default into larger subalgebras of `gl(K)`.

**Tech Stack:** Python 3, PyTorch (generators built in float64 then cast to float32, matching 2.0), pytest. No neural-network components. No CLI parsing.

**Scope note:** Self-contained, testable. Builds on the merged Phase 0/1 trunk (`main`). Phase 2b (transport) is in `2026-05-29-vfe3-phase2b-transport.md` and will be revised to consume `GaugeGroup` before it is executed.

**Reference spec:** `docs/superpowers/specs/2026-05-29-vfe3-clean-room-design.md` (§4.2 geometry).

**2.0 references being matched** (in `C:\Users\chris and christine\Desktop\VFE_2.0\math_utils\generators\`):
- `builders.py::generate_glK_generators` (515) — full gl(K) basis `E_ij`.
- `builders.py::generate_glK_multihead_generators` (563) — block-diagonal `⊕gl(d_head)`.
- `builders.py::generate_glK_cross_head_generators` (608) — diagonal blocks + off-diagonal coupling blocks.
- `builders.py::generate_soN_generators` (424) — skew `L_ij = E_ij − E_ji`.
- `builders.py::_dedup_cross_couplings` (36) — dedup directed coupling pairs.
- `closure.py::close_under_brackets` (199) — SVD Lie-bracket closure; inputs preserved verbatim as the first `n_gen` rows.

The golden tests import 2.0 via the existing harness: the `vfe2_kl` fixture already inserts the VFE_2.0 root on `sys.path`, so `math_utils.generators.builders` / `.closure` import directly.

---

## Code Style (MANDATORY)

Follow the repo `CLAUDE.md` convention: tensors first, then `'float | torch.Tensor'`, then undefined floats/ints/bools, then defined floats/ints/bools, then `Optional`, then `**kwargs`; names, type annotations, `=` signs, and trailing `#` comments vertically aligned; blank lines between type groups; tensor shape comments at critical points; keyword-only (`*`) scalar knobs. Builders here mirror 2.0's keyword-only `*` style.

---

## File Structure

- Create: `vfe3/geometry/__init__.py` — empty package marker.
- Create: `vfe3/geometry/generators.py` — `generate_glk`, `generate_glk_multihead`, `generate_glk_cross_head`, `generate_son`, `_dedup_cross_couplings`.
- Create: `vfe3/geometry/closure.py` — `close_under_brackets` (faithful port).
- Create: `vfe3/geometry/groups.py` — `GaugeGroup` dataclass + registry (`register_group`/`get_group`) with `glk`, `block_glk`, `so_k` builders and cross-coupling support; admissibility metadata.
- Create: `tests/test_gauge_groups.py` — unit + property tests (block-diagonal structure, skew property, cross-coupling shapes, registry, admissibility/equivariance).
- Create: `tests/golden/test_generators_golden.py` — golden vs 2.0 `math_utils.generators`.

Helper for the golden conftest: a `vfe2_gen` fixture parallel to `vfe2_kl`. Add it to `tests/golden/conftest.py`.

---

## Task 1: golden fixture for the 2.0 generators module

**Files:**
- Modify: `tests/golden/conftest.py`

- [ ] **Step 1: Add the fixture**

Append to `tests/golden/conftest.py`:

```python
@pytest.fixture(scope="session")
def vfe2_gen():
    """Return the 2.0 generator builders + closure, or skip if unavailable."""
    root = _vfe2_root()
    if root is None:
        pytest.skip("VFE_2.0 checkout not found (set VFE2_ROOT)")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from math_utils.generators import builders, closure
    except ImportError as exc:
        pytest.skip(f"could not import VFE_2.0 generators: {exc}")
    return {"builders": builders, "closure": closure}
```

- [ ] **Step 2: Commit**

```
git add tests/golden/conftest.py
git commit -m "test(golden): vfe2_gen fixture for 2.0 generator builders/closure

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: elementary generators (`generators.py`)

**Files:**
- Create: `vfe3/geometry/__init__.py` (empty)
- Create: `vfe3/geometry/generators.py`
- Test: `tests/golden/test_generators_golden.py`

- [ ] **Step 1: Create the package marker**

Create `vfe3/geometry/__init__.py` (empty file).

- [ ] **Step 2: Write the failing golden test**

Create `tests/golden/test_generators_golden.py`:

```python
import pytest
import torch


def test_glk_matches_vfe2(vfe2_gen):
    from vfe3.geometry.generators import generate_glk
    ref = vfe2_gen["builders"].generate_glK_generators(5)
    got = generate_glk(5)
    assert torch.equal(got, ref)


def test_glk_sl_matches_vfe2(vfe2_gen):
    from vfe3.geometry.generators import generate_glk
    ref = vfe2_gen["builders"].generate_glK_generators(4, include_identity=False)
    got = generate_glk(4, include_identity=False)
    assert torch.allclose(got, ref, atol=1e-6)


def test_glk_multihead_matches_vfe2(vfe2_gen):
    from vfe3.geometry.generators import generate_glk_multihead
    ref = vfe2_gen["builders"].generate_glK_multihead_generators(6, 3)
    got = generate_glk_multihead(6, 3)
    assert torch.equal(got, ref)


def test_glk_cross_head_matches_vfe2(vfe2_gen):
    from vfe3.geometry.generators import generate_glk_cross_head
    pairs = [(0, 1), (1, 2)]
    ref = vfe2_gen["builders"].generate_glK_cross_head_generators(6, 3, pairs)
    got = generate_glk_cross_head(6, 3, pairs)
    assert torch.equal(got, ref)


def test_son_matches_vfe2(vfe2_gen):
    from vfe3.geometry.generators import generate_son
    ref = vfe2_gen["builders"].generate_soN_generators(5)
    got = generate_son(5)
    assert torch.equal(got, ref)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/golden/test_generators_golden.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vfe3.geometry.generators'`.

- [ ] **Step 4: Write minimal implementation**

Create `vfe3/geometry/generators.py`:

```python
r"""Lie-algebra generator construction for VFE_3.0 gauge groups.

Ported from VFE_2.0 math_utils/generators/builders.py. Generators are built in
float64 (exact integer entries) then cast to the requested dtype, matching 2.0.
Conventions:
  gl(K)            : full K^2 basis E_ij (1 at (i,j)), row-major.
  block GL(d_head) : per-head gl(d_head) embedded in the head's diagonal block.
  cross-head       : diagonal blocks + off-diagonal E_ij blocks per coupling.
  so(N)            : skew L_ij = E_ij - E_ji for i < j.
"""

import math
from typing import List, Tuple

import torch


def _dedup_cross_couplings(
    pairs: List[Tuple[int, int]],
) -> Tuple[List[Tuple[int, int]], int]:
    r"""Drop exact duplicate directed pairs, preserving first-seen order.

    Directed: (a, b) and (b, a) are distinct. Returns (deduped, n_removed).
    """
    seen:    set = set()
    out:     List[Tuple[int, int]] = []
    removed: int = 0
    for a, b in pairs:
        key = (int(a), int(b))
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        out.append(key)
    return out, removed


def generate_glk(
    K:                int,

    *,
    include_identity: bool                            = True,
    device:           'torch.device | str | None'     = None,
    dtype:            torch.dtype                      = torch.float32,
) -> torch.Tensor:
    r"""gl(K) generators (full K^2 basis E_ij), or sl(K) if include_identity=False.

    Returns (K^2, K, K), or (K^2 - 1, K, K) for sl(K).
    """
    if K < 1:
        raise ValueError(f"K must be >= 1 for GL(K), got K={K}")

    n_generators = K * K
    G = torch.zeros(n_generators, K, K, dtype=torch.float64)

    idx = 0
    for i in range(K):
        for j in range(K):
            G[idx, i, j] = 1.0
            idx += 1

    if not include_identity:
        I_K       = torch.eye(K, dtype=torch.float64)
        trace_dir = I_K / math.sqrt(K)
        projected = []
        for g in range(n_generators):
            overlap = torch.sum(G[g] * trace_dir)
            G_proj  = G[g] - overlap * trace_dir
            if torch.linalg.norm(G_proj) > 1e-8:
                projected.append(G_proj)
        G = torch.stack(projected, dim=0)

    return G.to(dtype).to(device)


def generate_glk_multihead(
    K:                int,
    n_heads:          int,

    *,
    device:           'torch.device | str | None'     = None,
    dtype:            torch.dtype                      = torch.float32,
) -> torch.Tensor:
    r"""Block-diagonal gl(d_head) generators: GL(d_head)^H subset of GL(K).

    d_head = K // n_heads. Returns (n_heads * d_head^2, K, K).
    """
    if K % n_heads != 0:
        raise ValueError(f"K={K} must be divisible by n_heads={n_heads}")

    d_head         = K // n_heads
    n_gen_per_head = d_head * d_head
    n_generators   = n_heads * n_gen_per_head

    G = torch.zeros(n_generators, K, K, dtype=torch.float64)
    for h in range(n_heads):
        start      = h * d_head
        gen_offset = h * n_gen_per_head
        idx        = 0
        for i in range(d_head):
            for j in range(d_head):
                G[gen_offset + idx, start + i, start + j] = 1.0
                idx += 1

    return G.to(dtype).to(device)


def generate_glk_cross_head(
    K:                int,
    n_heads:          int,
    cross_couplings:  List[Tuple[int, int]],

    *,
    device:           'torch.device | str | None'     = None,
    dtype:            torch.dtype                      = torch.float32,
) -> torch.Tensor:
    r"""Block-diagonal gl(d_head) plus off-diagonal coupling blocks.

    For each directed pair (a, b), d_head^2 elementary matrices map head a's
    rows into head b's columns. Returns
    (n_heads * d_head^2 + len(dedup(cross)) * d_head^2, K, K).
    """
    if K % n_heads != 0:
        raise ValueError(f"K={K} not divisible by n_heads={n_heads}")

    cross_couplings, _ = _dedup_cross_couplings(list(cross_couplings))

    d_head      = K // n_heads
    n_gen_diag  = n_heads * d_head * d_head
    n_gen_cross = len(cross_couplings) * d_head * d_head
    n_gen_total = n_gen_diag + n_gen_cross

    G = torch.zeros(n_gen_total, K, K, dtype=torch.float64)

    for h in range(n_heads):
        start      = h * d_head
        gen_offset = h * d_head * d_head
        idx        = 0
        for i in range(d_head):
            for j in range(d_head):
                G[gen_offset + idx, start + i, start + j] = 1.0
                idx += 1

    for pair_idx, (a, b) in enumerate(cross_couplings):
        if a == b:
            raise ValueError(f"Self-coupling ({a},{a}) not allowed")
        if not (0 <= a < n_heads and 0 <= b < n_heads):
            raise ValueError(f"Head indices ({a},{b}) out of range [0, {n_heads})")
        a_start    = a * d_head
        b_start    = b * d_head
        gen_offset = n_gen_diag + pair_idx * d_head * d_head
        idx        = 0
        for i in range(d_head):
            for j in range(d_head):
                G[gen_offset + idx, a_start + i, b_start + j] = 1.0
                idx += 1

    return G.to(dtype).to(device)


def generate_son(
    N:                int,

    *,
    device:           'torch.device | str | None'     = None,
    dtype:            torch.dtype                      = torch.float32,
) -> torch.Tensor:
    r"""so(N) generators L_ij = E_ij - E_ji for i < j. Returns (N(N-1)/2, N, N)."""
    if N < 2:
        raise ValueError(f"N must be >= 2 for SO(N), got N={N}")

    n_generators = N * (N - 1) // 2
    G = torch.zeros(n_generators, N, N, dtype=torch.float64)
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            G[idx, i, j] = 1.0
            G[idx, j, i] = -1.0
            idx += 1

    return G.to(dtype).to(device)
```

Note: `generate_son` omits 2.0's optional `validate=` self-check (a pure assertion path, not part of the returned value). The golden test pins the returned tensor to 2.0's `generate_soN_generators(N)` (whose default `validate=True` does not change the output).

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/golden/test_generators_golden.py -v`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```
git add vfe3/geometry/__init__.py vfe3/geometry/generators.py tests/golden/test_generators_golden.py
git commit -m "feat(geometry): gl(K)/block/cross-head/so(N) generators, golden-equal to 2.0

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Lie-bracket closure (`closure.py`)

**Files:**
- Create: `vfe3/geometry/closure.py`
- Test: `tests/golden/test_generators_golden.py`

This is a faithful port of `close_under_brackets` (VFE_2.0 `math_utils/generators/closure.py:199`). Port the function verbatim, preserving every invariant: float64 internal arithmetic cast back to input dtype; the returned basis contains the input generators VERBATIM as its first `n_gen` rows; appended directions are the unit-norm left-singular vectors of the projected-out commutators, sign-pinned and degeneracy-ordered via the helper `_sign_pin_and_order` (port `_sign_pin_and_order` and `_lexsort_rows` too — closure.py:53 and :108); `max_dim` default `K*K`; same `tol`/`degenerate_tol` defaults; return `(closed_generators, info_dict)` with the same `info` keys (`n_iters`, `n_added`, `final_dim`, `initial_dim`, `converged`, `hit_max_dim`). Acceptance is golden equality to 2.0, not re-derivation — copy the three functions and adapt only imports.

- [ ] **Step 1: Write the failing golden test**

Append to `tests/golden/test_generators_golden.py`:

```python
def test_closure_of_two_cross_blocks_matches_vfe2(vfe2_gen):
    from vfe3.geometry.generators import generate_glk_cross_head
    from vfe3.geometry.closure import close_under_brackets
    # A single directed cross-coupling is NOT Lie-closed; closing it pulls in
    # the reverse block + extra diagonal directions.
    gens = generate_glk_cross_head(4, 2, [(0, 1)])
    ref_closed, ref_info = vfe2_gen["closure"].close_under_brackets(gens)
    got_closed, got_info = close_under_brackets(gens)
    assert got_closed.shape == ref_closed.shape
    assert torch.allclose(got_closed, ref_closed, atol=1e-6)
    assert got_info["final_dim"] == ref_info["final_dim"]
    assert got_info["converged"] == ref_info["converged"]
    # Inputs preserved verbatim as the first n_gen rows.
    assert torch.allclose(got_closed[: gens.shape[0]], gens, atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/golden/test_generators_golden.py::test_closure_of_two_cross_blocks_matches_vfe2 -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vfe3.geometry.closure'`.

- [ ] **Step 3: Write implementation (faithful port)**

Create `vfe3/geometry/closure.py` by copying `close_under_brackets`, `_sign_pin_and_order`, and `_lexsort_rows` verbatim from `C:\Users\chris and christine\Desktop\VFE_2.0\math_utils\generators\closure.py` (lines 199-end, 53-107, 108-123 respectively), adjusting only the module docstring and imports (`import torch`, `from typing import Dict, Optional, Tuple`). Do not change the math. Read the source file directly to copy it exactly.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/golden/test_generators_golden.py::test_closure_of_two_cross_blocks_matches_vfe2 -v`
Expected: 1 passed. If the closed bases differ only by column sign/order, the `_sign_pin_and_order` port is incomplete — fix it to match 2.0 exactly (do not loosen the assertion).

- [ ] **Step 5: Commit**

```
git add vfe3/geometry/closure.py tests/golden/test_generators_golden.py
git commit -m "feat(geometry): Lie-bracket closure (close_under_brackets), golden-equal to 2.0

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: `GaugeGroup` registry (`groups.py`)

**Files:**
- Create: `vfe3/geometry/groups.py`
- Test: `tests/test_gauge_groups.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gauge_groups.py`:

```python
import pytest
import torch

from vfe3.geometry.groups import GaugeGroup, get_group, register_group


def test_glk_group_full_basis():
    grp = get_group("glk")(K=5)
    assert grp.generators.shape == (25, 5, 5)
    assert grp.irrep_dims == [5]
    assert grp.skew_symmetric is False


def test_block_glk_group_is_block_diagonal():
    grp = get_group("block_glk")(K=6, n_heads=3)
    assert grp.irrep_dims == [2, 2, 2]
    # Every generator is supported only within one diagonal 2x2 block.
    d = 2
    for g in grp.generators:
        for h in range(3):
            block = g[h * d:(h + 1) * d, h * d:(h + 1) * d]
            outside = g.clone()
            outside[h * d:(h + 1) * d, h * d:(h + 1) * d] = 0.0
        # off-block entries must all be zero for block-diagonal generators
        mask = torch.ones(6, 6)
        for h in range(3):
            mask[h * d:(h + 1) * d, h * d:(h + 1) * d] = 0.0
        assert torch.count_nonzero(g * mask) == 0


def test_so_k_group_is_skew():
    grp = get_group("so_k")(K=4)
    assert grp.skew_symmetric is True
    assert torch.allclose(
        grp.generators + grp.generators.transpose(-1, -2),
        torch.zeros_like(grp.generators),
        atol=1e-6,
    )


def test_block_glk_with_cross_coupling_grows_basis():
    base = get_group("block_glk")(K=6, n_heads=3)
    coupled = get_group("block_glk")(K=6, n_heads=3, cross_couplings=[(0, 1)])
    # Cross coupling adds d_head^2 = 4 off-block generators (before closure).
    assert coupled.generators.shape[0] == base.generators.shape[0] + 4


def test_unknown_group_raises():
    with pytest.raises(KeyError):
        get_group("not_a_group")


def test_gaussian_admissibility_is_declared():
    grp = get_group("glk")(K=4)
    assert grp.invariant_for("gaussian") is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_gauge_groups.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vfe3.geometry.groups'`.

- [ ] **Step 3: Write minimal implementation**

Create `vfe3/geometry/groups.py`:

```python
r"""Gauge-group registry for VFE_3.0 (structure-group axis of geometry).

A GaugeGroup bundles the Lie-algebra generators with the metadata transport
needs (block/irrep structure, skew flag) and declares the families whose
divergence is invariant under its representation (admissibility). Groups are
config-selected by name so variants swap without editing call sites.

Admissibility: a (family, group) pair is valid iff the family's divergence is
invariant under common pushforward by the group's representation,
D(rho(g) q || rho(g) p) = D(q || p). For the Gaussian family with the GL(K)
congruence action (mu -> g mu, Sigma -> g Sigma g^T) this holds for every
g in G <= GL(K), so every group here is admissible for "gaussian".
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch

from vfe3.geometry.closure import close_under_brackets
from vfe3.geometry.generators import (
    generate_glk,
    generate_glk_cross_head,
    generate_glk_multihead,
    generate_son,
)


@dataclass
class GaugeGroup:
    """A structure group plus the metadata the transport layer consumes."""

    name:            str
    generators:      torch.Tensor          # (n_gen, K, K) Lie-algebra basis
    irrep_dims:      List[int]             # block sizes; sum == K
    skew_symmetric:  bool                  # exp(-M) = exp(M)^T fast path
    invariant_families: Tuple[str, ...] = ("gaussian",)

    def invariant_for(self, family: str) -> bool:
        """Whether the divergence of ``family`` is invariant under this group."""
        return family in self.invariant_families


_GROUPS: Dict[str, Callable[..., GaugeGroup]] = {}


def register_group(name: str) -> Callable:
    """Decorator registering a GaugeGroup builder under ``name``."""
    def _wrap(fn: Callable[..., GaugeGroup]) -> Callable[..., GaugeGroup]:
        _GROUPS[name] = fn
        return fn
    return _wrap


def get_group(name: str) -> Callable[..., GaugeGroup]:
    """Return the registered GaugeGroup builder for ``name`` (KeyError if absent)."""
    if name not in _GROUPS:
        raise KeyError(
            f"no gauge group registered under {name!r}; available: {sorted(_GROUPS)}"
        )
    return _GROUPS[name]


@register_group("glk")
def _build_glk(
    K:       int,

    *,
    dtype:   torch.dtype = torch.float32,
) -> GaugeGroup:
    """Full GL(K): single block, full gl(K) generators."""
    G = generate_glk(K, dtype=dtype)
    return GaugeGroup(name="glk", generators=G, irrep_dims=[K], skew_symmetric=False)


@register_group("block_glk")
def _build_block_glk(
    K:               int,
    n_heads:         int,

    *,
    cross_couplings: Optional[List[Tuple[int, int]]] = None,
    close_basis:     bool                            = False,
    dtype:           torch.dtype                     = torch.float32,
) -> GaugeGroup:
    """Block-diagonal GL(K) = GL(d_head)^n_heads, optional cross-head coupling.

    With ``cross_couplings`` the basis includes off-block generators; with
    ``close_basis=True`` it is closed under the Lie bracket into a subalgebra
    of gl(K) (so the exponentiated group is well-defined).
    """
    d_head = K // n_heads
    if cross_couplings:
        G = generate_glk_cross_head(K, n_heads, cross_couplings, dtype=dtype)
        if close_basis:
            G, _ = close_under_brackets(G)
    else:
        G = generate_glk_multihead(K, n_heads, dtype=dtype)
    return GaugeGroup(
        name="block_glk",
        generators=G,
        irrep_dims=[d_head] * n_heads,
        skew_symmetric=False,
    )


@register_group("so_k")
def _build_so_k(
    K:       int,

    *,
    dtype:   torch.dtype = torch.float32,
) -> GaugeGroup:
    """SO(K): skew-symmetric so(K) generators (single block)."""
    G = generate_son(K, dtype=dtype)
    return GaugeGroup(name="so_k", generators=G, irrep_dims=[K], skew_symmetric=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_gauge_groups.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```
git add vfe3/geometry/groups.py tests/test_gauge_groups.py
git commit -m "feat(geometry): GaugeGroup registry (glk/block_glk/so_k) + admissibility

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: gauge-admissibility property test (divergence invariance under the group)

**Files:**
- Test: `tests/test_gauge_groups.py`

The admissibility flag must reflect reality: applying any group element to both KL arguments leaves the Gaussian KL unchanged. This pins the `invariant_for("gaussian")` declaration to an actual numerical invariance for each registered group.

- [ ] **Step 1: Write the property test**

Append to `tests/test_gauge_groups.py`:

```python
@pytest.mark.parametrize("spec", [
    ("glk",       {"K": 4}),
    ("block_glk", {"K": 6, "n_heads": 3}),
    ("so_k",      {"K": 4}),
])
def test_full_kl_invariant_under_group_pushforward(spec):
    # For a random group element g = exp(sum_a c_a G_a), the Gaussian KL is
    # invariant under common pushforward mu->g mu, Sigma->g Sigma g^T.
    from vfe3.divergence import kl
    name, kwargs = spec
    grp = get_group(name)(**kwargs)
    K = sum(grp.irrep_dims)
    gen = torch.Generator().manual_seed(0)
    coeff = 0.2 * torch.randn(grp.generators.shape[0], generator=gen)
    M = torch.einsum("a,aij->ij", coeff, grp.generators)
    g = torch.linalg.matrix_exp(M)                              # (K, K) in G

    mu_q = torch.randn(5, K, generator=gen)
    mu_p = torch.randn(5, K, generator=gen)
    Aq = torch.randn(5, K, K, generator=gen)
    Ap = torch.randn(5, K, K, generator=gen)
    S_q = Aq @ Aq.transpose(-1, -2) + torch.eye(K)
    S_p = Ap @ Ap.transpose(-1, -2) + torch.eye(K)

    base = kl(mu_q, S_q, mu_p, S_p, family="gaussian_full")
    mu_q2 = torch.einsum("kl,nl->nk", g, mu_q)
    mu_p2 = torch.einsum("kl,nl->nk", g, mu_p)
    S_q2 = g @ S_q @ g.transpose(-1, -2)
    S_p2 = g @ S_p @ g.transpose(-1, -2)
    moved = kl(mu_q2, S_q2, mu_p2, S_p2, family="gaussian_full")

    assert grp.invariant_for("gaussian")
    assert torch.allclose(base, moved, atol=1e-3, rtol=1e-3)
```

- [ ] **Step 2: Run the property test**

Run: `python -m pytest tests/test_gauge_groups.py::test_full_kl_invariant_under_group_pushforward -v`
Expected: 3 passed.

- [ ] **Step 3: Run the full suite**

Run: `python -m pytest -q`
Expected: all pass (Phase 0/1 + Phase 2a).

- [ ] **Step 4: Commit**

```
git add tests/test_gauge_groups.py
git commit -m "test(geometry): Gaussian KL invariance under group pushforward (admissibility)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage (Phase 2a slice of §4.2):**
- Gauge-group registry (GL(K), block-diagonal ⊕GL(d_h), SO(K)) → Tasks 2, 4.
- Cross-coupling generators + Lie closure (super-block subalgebra) → Tasks 2, 3, 4.
- Admissibility (divergence invariant under common pushforward) → Tasks 4, 5.
- Golden equivalence vs `math_utils/generators/` → Tasks 2, 3.

**Deferred (named, to later geometry sub-phases):**
- Transport operator `Omega = exp(phi_i)exp(-phi_j)`, block-diagonal exp, belief action → Phase 2b (consumes `GaugeGroup`).
- Regime II edge-relaxed cocycle, SPD/φ retractions, Fisher preconditioner → Phase 2c.
- RoPE on μ, `VFEHeadMixer` → Phase 2d.
- `super_block_dims`/`super_block_head_groups` reordering caches → built in Phase 2b when transport needs the block layout.
- SO(3)/tesseral and sym2-traceless irreps, multi-irrep SO(N) → added to the registry when a model needs them.

**Placeholder scan:** none. Task 3 is a faithful verbatim port (acceptance = golden equality), with the exact source file:line to copy from — not a placeholder.

**Type/name consistency:** `generate_glk`/`generate_glk_multihead`/`generate_glk_cross_head`/`generate_son`/`_dedup_cross_couplings` (generators.py); `close_under_brackets` (closure.py); `GaugeGroup`/`register_group`/`get_group` with builders `glk`/`block_glk`/`so_k` (groups.py) are used consistently across tasks and tests. `irrep_dims` sums to K in every group.
