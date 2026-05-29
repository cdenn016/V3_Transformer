# VFE_3.0 Phase 2b (Transport / Regime I belief action) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `vfe3/geometry/transport.py` — the Regime I gauge transport operator driven by a `GaugeGroup`, in BOTH parameterizations (`phi`/exp → GL⁺(K), and direct-Ω → general GL(K)), the belief action (μ→Ωμ, Σ→ΩΣΩᵀ), and the block-slice helper — proven numerically equal to VFE_2.0, with a gauge-equivariance property test.

**Architecture:** Second geometry sub-phase. Consumes the `GaugeGroup` from Phase 2a. Three orthogonal axes per spec §4.2: structure group (2a), connection regime (Regime I here; II later), and **parameterization** — `phi` exp form `Ω_ij = exp(φ_i)exp(−φ_j) ∈ GL⁺(K)` (det>0, invertible by construction), and `omega_direct` form `Ω_ij = Ω_i Ω_j⁻¹` reaching general GL(K) (det can be <0). SPD/φ retractions, Regime II, RoPE/head-mixer, and the block-diagonal exp *optimization* are Phase 2c/2d.

**Tech Stack:** Python 3, PyTorch (float32; float64 inside matrix_exp for K≥20), pytest. No NN. No CLI.

**Reference spec:** `docs/superpowers/specs/2026-05-29-vfe3-clean-room-design.md` (§4.2). Prereq: Phase 2a (`vfe3/geometry/{groups,generators,closure}.py`).

**2.0 references being matched** (`C:\Users\chris and christine\Desktop\VFE_2.0`):
- `gauge_utils.py::stable_matrix_exp_pair` (53–131).
- `transport_ops.py::compute_transport_operators` (285–433), flat + trivial.
- `transport_ops.py::compute_transport_operators_direct` (440–551), flat + trivial.
- `transport_ops.py::omega_to_block_exp_pairs` (554–602).
- diagonal sandwich approx `attention.py:270`: `Σ_t[i,j,k] = Σ_l Ω[i,j,k,l]² σ[j,l]`.

---

## Code Style (MANDATORY)

Repo `CLAUDE.md`: tensors first, type-grouped with blank lines, names/types/`=`/comments vertically aligned, keyword-only `*` scalar knobs.

---

## File Structure

- Create: `vfe3/geometry/transport.py` — `stable_matrix_exp_pair`, `compute_transport_operators` (phi/exp, group-driven), `compute_transport_operators_direct` (omega_direct), `transport_mean`, `transport_covariance`, `omega_to_block_exp_pairs`.
- Modify: `tests/golden/conftest.py` (add `vfe2_transport` fixture).
- Create: `tests/golden/test_transport_golden.py` — golden vs 2.0.
- Create: `tests/test_transport.py` — belief-action unit + equivariance + det<0 representability.

---

## Task 1: `stable_matrix_exp_pair` + golden fixture

**Files:** Create `vfe3/geometry/transport.py`; Modify `tests/golden/conftest.py`; Test `tests/golden/test_transport_golden.py`.

- [ ] **Step 1: Add the golden fixture** — append to `tests/golden/conftest.py`:

```python
@pytest.fixture(scope="session")
def vfe2_transport():
    """Return the 2.0 transport + gauge modules, or skip if unavailable."""
    root = _vfe2_root()
    if root is None:
        pytest.skip("VFE_2.0 checkout not found (set VFE2_ROOT)")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from transformer.core import gauge_utils, transport_ops
    except ImportError as exc:
        pytest.skip(f"could not import VFE_2.0 transport: {exc}")
    return {"gauge_utils": gauge_utils, "transport_ops": transport_ops}
```

- [ ] **Step 2: Failing golden test** — create `tests/golden/test_transport_golden.py`:

```python
import pytest
import torch


def test_stable_matrix_exp_pair_matches_vfe2(vfe2_transport):
    from vfe3.geometry.transport import stable_matrix_exp_pair
    g = torch.Generator(device="cpu").manual_seed(0)
    M = torch.randn(2, 3, 4, 4, generator=g)
    ref_pos, ref_neg = vfe2_transport["gauge_utils"].stable_matrix_exp_pair(
        M, skew_symmetric=False
    )
    pos, neg = stable_matrix_exp_pair(M, skew_symmetric=False)
    assert torch.allclose(pos, ref_pos, atol=1e-5, rtol=1e-5)
    assert torch.allclose(neg, ref_neg, atol=1e-5, rtol=1e-5)
```

- [ ] **Step 3: Run — expect FAIL** (`ModuleNotFoundError: vfe3.geometry.transport`).
Run: `python -m pytest tests/golden/test_transport_golden.py -v`

- [ ] **Step 4: Create `vfe3/geometry/transport.py`:**

```python
r"""Gauge transport for VFE_3.0 (Regime I, Gaussian / location-scale specific).

Two parameterizations of the flat (Regime I) transport:
  phi (exp):    Omega_ij = exp(phi_i . G) exp(-phi_j . G) in GL+(K) (det>0).
  omega_direct: Omega_ij = Omega_i Omega_j^{-1} for general GL(K) (det may be <0).
Belief action: mu -> Omega @ mu, Sigma -> Omega @ Sigma @ Omega^T (sandwich;
diagonal approximation for speed). Regime II, retractions, RoPE are later phases.
"""

from typing import Dict, List, Optional, Tuple

import torch

from vfe3.geometry.groups import GaugeGroup

TransportDict = Dict[str, torch.Tensor]


def stable_matrix_exp_pair(
    matrix:         torch.Tensor,             # (..., d, d) Lie-algebra matrices

    *,
    max_norm:       float = 15.0,
    dim_threshold:  int   = 20,
    skew_symmetric: bool  = False,
    only_forward:   bool  = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    r"""exp(M) and optionally exp(-M) with Frobenius-norm clamp + float64 upcast.

    Ported from VFE_2.0 stable_matrix_exp_pair (gauge_utils.py:53-131).
    """
    mat_norm = matrix.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    scale = (max_norm / mat_norm).clamp(max=1.0)
    matrix = matrix * scale

    d = matrix.shape[-1]
    orig_dtype = matrix.dtype
    with torch.amp.autocast('cuda', enabled=False):
        if d >= dim_threshold:
            matrix_up = matrix.double().contiguous()
        else:
            matrix_up = matrix.float().contiguous()
        exp_pos = torch.linalg.matrix_exp(matrix_up).to(orig_dtype)
        if only_forward:
            exp_neg = None
        elif skew_symmetric:
            exp_neg = exp_pos.transpose(-1, -2)
        else:
            exp_neg = torch.linalg.matrix_exp(-matrix_up).to(orig_dtype)
    return exp_pos, exp_neg
```

- [ ] **Step 5: Run — expect 1 passed.**

- [ ] **Step 6: Commit**

```
git add vfe3/geometry/transport.py tests/golden/conftest.py tests/golden/test_transport_golden.py
git commit -m "feat(geometry): stable_matrix_exp_pair, golden-equal to 2.0

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `compute_transport_operators` (phi/exp, group-driven, Regime I + trivial)

**Files:** Modify `vfe3/geometry/transport.py`; Test `tests/golden/test_transport_golden.py`.

- [ ] **Step 1: Failing golden test** — append:

```python
def test_transport_operators_learned_matches_vfe2(vfe2_transport):
    from vfe3.geometry.transport import compute_transport_operators
    from vfe3.geometry.groups import get_group
    grp = get_group("so_k")(K=4)
    g = torch.Generator(device="cpu").manual_seed(2)
    phi = 0.3 * torch.randn(2, 3, grp.generators.shape[0], generator=g)
    ref = vfe2_transport["transport_ops"].compute_transport_operators(
        phi, grp.generators, enforce_orthogonal=False, gauge_mode="learned"
    )
    got = compute_transport_operators(phi, grp, gauge_mode="learned")
    assert torch.allclose(got["exp_phi"],     ref["exp_phi"],     atol=1e-5, rtol=1e-5)
    assert torch.allclose(got["exp_neg_phi"], ref["exp_neg_phi"], atol=1e-5, rtol=1e-5)
    assert torch.allclose(got["Omega"],       ref["Omega"],       atol=1e-5, rtol=1e-5)


def test_transport_operators_glk_matches_vfe2(vfe2_transport):
    from vfe3.geometry.transport import compute_transport_operators
    from vfe3.geometry.groups import get_group
    grp = get_group("block_glk")(K=6, n_heads=3)
    g = torch.Generator(device="cpu").manual_seed(5)
    phi = 0.2 * torch.randn(2, 3, grp.generators.shape[0], generator=g)
    ref = vfe2_transport["transport_ops"].compute_transport_operators(
        phi, grp.generators, enforce_orthogonal=False, gauge_mode="learned"
    )
    got = compute_transport_operators(phi, grp, gauge_mode="learned")
    assert torch.allclose(got["Omega"], ref["Omega"], atol=1e-5, rtol=1e-5)


def test_transport_operators_trivial_is_identity():
    from vfe3.geometry.transport import compute_transport_operators
    from vfe3.geometry.groups import get_group
    grp = get_group("so_k")(K=4)
    phi = torch.zeros(2, 3, grp.generators.shape[0])
    out = compute_transport_operators(phi, grp, gauge_mode="trivial")
    assert torch.allclose(out["Omega"], torch.eye(4).expand(2, 3, 3, 4, 4), atol=1e-6)
```

- [ ] **Step 2: Run — expect FAIL** (`ImportError`).

- [ ] **Step 3: Implement** — append:

```python
def compute_transport_operators(
    phi:        torch.Tensor,             # (B, N, n_gen) gauge frames
    group:      GaugeGroup,               # supplies generators, skew flag, irrep_dims

    *,
    gauge_mode: str = "learned",          # 'learned' (Regime I flat) or 'trivial'
) -> TransportDict:
    r"""phi/exp transport Omega_ij = exp(phi_i) @ exp(-phi_j) in GL+(K).

    Ported from VFE_2.0 compute_transport_operators (transport_ops.py:285-433),
    flat path. 'trivial' returns Omega = I. Returns 'exp_phi' (B,N,K,K),
    'exp_neg_phi' (B,N,K,K), 'Omega' (B,N,N,K,K).
    """
    B, N, _ = phi.shape
    generators = group.generators
    K = generators.shape[-1]
    dtype = phi.dtype
    device = phi.device

    if gauge_mode == "trivial":
        eye_K = torch.eye(K, device=device, dtype=dtype)
        return {
            "exp_phi":     eye_K.expand(B, N, K, K).contiguous(),
            "exp_neg_phi": eye_K.expand(B, N, K, K).contiguous(),
            "Omega":       eye_K.expand(B, N, N, K, K).contiguous(),
        }
    if gauge_mode != "learned":
        raise ValueError(f"gauge_mode must be 'learned' or 'trivial', got {gauge_mode!r}")

    phi_matrix = torch.einsum("bna,aij->bnij", phi, generators)
    exp_phi, exp_neg_phi = stable_matrix_exp_pair(
        phi_matrix, skew_symmetric=group.skew_symmetric
    )
    omega = torch.einsum("bikl,bjlm->bijkm", exp_phi, exp_neg_phi)
    return {"exp_phi": exp_phi, "exp_neg_phi": exp_neg_phi, "Omega": omega}
```

- [ ] **Step 4: Run — expect 3 passed.**

- [ ] **Step 5: Commit**

```
git add vfe3/geometry/transport.py tests/golden/test_transport_golden.py
git commit -m "feat(geometry): GaugeGroup-driven phi/exp transport (GL+(K), Regime I), golden vs 2.0

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: direct-Ω transport `compute_transport_operators_direct` (general GL(K))

**Files:** Modify `vfe3/geometry/transport.py`; Test `tests/golden/test_transport_golden.py`.

- [ ] **Step 1: Failing golden test** — append:

```python
def test_transport_operators_direct_matches_vfe2(vfe2_transport):
    from vfe3.geometry.transport import compute_transport_operators_direct
    g = torch.Generator(device="cpu").manual_seed(11)
    # Per-token Omega_i near identity (well-conditioned -> exact solve path).
    omega = torch.eye(4) + 0.2 * torch.randn(2, 3, 4, 4, generator=g)
    ref = vfe2_transport["transport_ops"].compute_transport_operators_direct(
        omega, gauge_mode="learned"
    )
    got = compute_transport_operators_direct(omega, gauge_mode="learned")
    assert torch.allclose(got["omega_j_inv"], ref["omega_j_inv"], atol=1e-4, rtol=1e-4)
    assert torch.allclose(got["Omega"],       ref["Omega"],       atol=1e-4, rtol=1e-4)


def test_transport_operators_direct_trivial_is_identity():
    from vfe3.geometry.transport import compute_transport_operators_direct
    omega = torch.eye(4) + 0.1 * torch.randn(2, 3, 4, 4)
    out = compute_transport_operators_direct(omega, gauge_mode="trivial")
    assert torch.allclose(out["Omega"], torch.eye(4).expand(2, 3, 3, 4, 4), atol=1e-6)
```

- [ ] **Step 2: Run — expect FAIL** (`ImportError`).

- [ ] **Step 3: Implement** — append:

```python
def compute_transport_operators_direct(
    omega:      torch.Tensor,             # (B, N, K, K) per-token group elements Omega_i

    *,
    gauge_mode: str   = "learned",        # 'learned' (flat cocycle) or 'trivial'
    eps:        float = 1e-6,
) -> TransportDict:
    r"""Direct-Omega transport Omega_ij = Omega_i @ Omega_j^{-1} (general GL(K)).

    Ported from VFE_2.0 compute_transport_operators_direct (transport_ops.py:440),
    flat path. Reaches all of GL(K) (det may be < 0; needs an external det
    penalty to stay invertible). Inverse via LU solve (exact cocycle), with a
    ridge then pinv fallback for near-singular Omega. 'trivial' returns Omega=I.
    Returns 'omega_i' (B,N,K,K), 'omega_j_inv' (B,N,K,K), 'Omega' (B,N,N,K,K).
    """
    B, N, K, _ = omega.shape
    dtype = omega.dtype
    device = omega.device

    if gauge_mode == "trivial":
        eye_K = torch.eye(K, device=device, dtype=dtype)
        return {
            "omega_i":     eye_K.expand(B, N, K, K).contiguous(),
            "omega_j_inv": eye_K.expand(B, N, K, K).contiguous(),
            "Omega":       eye_K.expand(B, N, N, K, K).contiguous(),
        }
    if gauge_mode != "learned":
        raise ValueError(f"gauge_mode must be 'learned' or 'trivial', got {gauge_mode!r}")

    eye_K = torch.eye(K, device=device, dtype=dtype)
    try:
        omega_j_inv = torch.linalg.solve(omega, eye_K.expand_as(omega))
    except (torch.linalg.LinAlgError, RuntimeError):
        try:
            omega_j_inv = torch.linalg.solve(omega + eps * eye_K, eye_K.expand_as(omega))
        except (torch.linalg.LinAlgError, RuntimeError):
            omega_j_inv = torch.linalg.pinv(omega)

    omega_ij = torch.einsum("bikl,bjlm->bijkm", omega, omega_j_inv)
    return {"omega_i": omega, "omega_j_inv": omega_j_inv, "Omega": omega_ij}
```

Note: V3 drops 2.0's `numerical_monitor` telemetry in the fallback branches (observability only; values identical). Non-flat (`connection_delta`) is Regime II → Phase 2c.

- [ ] **Step 4: Run — expect 2 passed.**

- [ ] **Step 5: Commit**

```
git add vfe3/geometry/transport.py tests/golden/test_transport_golden.py
git commit -m "feat(geometry): direct-Omega transport (general GL(K)), golden-equal to 2.0

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: belief action `transport_mean` / `transport_covariance`

**Files:** Modify `vfe3/geometry/transport.py`; Test `tests/test_transport.py`.

- [ ] **Step 1: Failing tests** — create `tests/test_transport.py`:

```python
import pytest
import torch

from vfe3.geometry.groups import get_group
from vfe3.geometry.transport import (
    compute_transport_operators,
    transport_covariance,
    transport_mean,
)


def _omega(seed, K=4):
    grp = get_group("so_k")(K=K)
    g = torch.Generator().manual_seed(seed)
    phi = 0.3 * torch.randn(2, 3, grp.generators.shape[0], generator=g)
    return compute_transport_operators(phi, grp, gauge_mode="learned")["Omega"], g


def test_transport_mean_identity_at_phi_zero():
    grp = get_group("so_k")(K=4)
    phi = torch.zeros(2, 3, grp.generators.shape[0])
    omega = compute_transport_operators(phi, grp, gauge_mode="learned")["Omega"]
    g = torch.Generator().manual_seed(0)
    mu = torch.randn(2, 3, 4, generator=g)
    mu_t = transport_mean(omega, mu)
    assert torch.allclose(mu_t, mu.unsqueeze(1).expand(2, 3, 3, 4), atol=1e-5)


def test_transport_covariance_full_is_spd():
    omega, g = _omega(1)
    A = torch.randn(2, 3, 4, 4, generator=g)
    sigma = A @ A.transpose(-1, -2) + torch.eye(4)
    sigma_t = transport_covariance(omega, sigma)
    assert torch.allclose(sigma_t, sigma_t.transpose(-1, -2), atol=1e-4)
    assert (torch.linalg.eigvalsh(sigma_t) > 0).all()


def test_transport_covariance_diag_matches_full_diagonal():
    omega, g = _omega(2)
    sigma_diag = torch.rand(2, 3, 4, generator=g) + 0.1
    full = transport_covariance(omega, torch.diag_embed(sigma_diag))
    approx = transport_covariance(omega, sigma_diag)
    assert torch.allclose(approx, torch.diagonal(full, dim1=-2, dim2=-1), atol=1e-5)


def test_transport_covariance_diag_matches_vfe2_formula():
    omega, g = _omega(3)
    sigma_diag = torch.rand(2, 3, 4, generator=g) + 0.1
    approx = transport_covariance(omega, sigma_diag)
    ref = torch.einsum("bijkl,bijkl,bjl->bijk", omega, omega, sigma_diag)
    assert torch.allclose(approx, ref, atol=1e-6)
```

- [ ] **Step 2: Run — expect FAIL** (`ImportError`).

- [ ] **Step 3: Implement** — append:

```python
def transport_mean(
    omega: torch.Tensor,             # (B, N, N, K, K) pairwise transport
    mu:    torch.Tensor,             # (B, N, K) source (key, index j) means
) -> torch.Tensor:
    r"""Gauge action on means: mu_t[i,j] = Omega_ij @ mu_j. Returns (B, N, N, K)."""
    return torch.einsum("bijkl,bjl->bijk", omega, mu)


def transport_covariance(
    omega: torch.Tensor,             # (B, N, N, K, K) pairwise transport
    sigma: torch.Tensor,             # (B, N, K) diagonal OR (B, N, K, K) full

    *,
    diagonal_out: Optional[bool] = None,
) -> torch.Tensor:
    r"""Sandwich action Sigma_t[i,j] = Omega_ij Sigma_j Omega_ij^T.

    Full input (B,N,K,K) -> full (B,N,N,K,K). Diagonal input (B,N,K) -> the
    diagonal approximation (B,N,N,K), Sigma_t[i,j,k] = sum_l Omega_ijkl^2
    sigma_jl (matches 2.0 attention.py:270).
    """
    is_diag = sigma.dim() == omega.dim() - 2 if diagonal_out is None else diagonal_out
    if is_diag:
        return torch.einsum("bijkl,bijkl,bjl->bijk", omega, omega, sigma)
    return torch.einsum("bijkl,bjlm,bijnm->bijkn", omega, sigma, omega)
```

- [ ] **Step 4: Run — expect 4 passed.**

- [ ] **Step 5: Commit**

```
git add vfe3/geometry/transport.py tests/test_transport.py
git commit -m "feat(geometry): belief action transport_mean / transport_covariance (full + diag)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: `omega_to_block_exp_pairs` (block slicing by irrep_dims)

**Files:** Modify `vfe3/geometry/transport.py`; Test `tests/golden/test_transport_golden.py`.

- [ ] **Step 1: Failing golden test** — append:

```python
def test_omega_to_block_exp_pairs_matches_vfe2(vfe2_transport):
    from vfe3.geometry.transport import (
        compute_transport_operators,
        omega_to_block_exp_pairs,
    )
    from vfe3.geometry.groups import get_group
    grp = get_group("block_glk")(K=6, n_heads=3)
    g = torch.Generator(device="cpu").manual_seed(7)
    phi = 0.2 * torch.randn(2, 3, grp.generators.shape[0], generator=g)
    exp_phi = compute_transport_operators(phi, grp, gauge_mode="learned")["exp_phi"]
    ref = vfe2_transport["transport_ops"].omega_to_block_exp_pairs(exp_phi, grp.irrep_dims)
    got = omega_to_block_exp_pairs(exp_phi, grp.irrep_dims)
    assert len(got) == len(ref)
    for (gb, gbi), (rb, rbi) in zip(got, ref):
        assert torch.allclose(gb, rb, atol=1e-5)
        assert torch.allclose(gbi, rbi, atol=1e-4)
```

- [ ] **Step 2: Run — expect FAIL** (`ImportError`).

- [ ] **Step 3: Implement** — append:

```python
def omega_to_block_exp_pairs(
    omega:      torch.Tensor,        # (B, N, K, K) per-token group elements
    irrep_dims: List[int],           # block sizes; sum == K

    *,
    eps:        float = 1e-6,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    r"""Slice a block-diagonal Omega into per-block (block, block_inv) pairs.

    Ported from VFE_2.0 omega_to_block_exp_pairs (transport_ops.py:554-602).
    Per-block inverse via solve, with ridge then pinv fallback. Returns a list
    aligned with irrep_dims, each a pair of (B, N, d, d) tensors.
    """
    id_sum = sum(irrep_dims)
    K = omega.shape[-1]
    if id_sum != K:
        raise ValueError(f"omega_to_block_exp_pairs: sum(irrep_dims)={id_sum} != K={K}")

    results: List[Tuple[torch.Tensor, torch.Tensor]] = []
    start = 0
    for d in irrep_dims:
        end = start + d
        omega_blk = omega[:, :, start:end, start:end].contiguous()
        eye_d = torch.eye(d, device=omega_blk.device, dtype=omega_blk.dtype)
        try:
            omega_blk_inv = torch.linalg.solve(omega_blk, eye_d.expand_as(omega_blk))
        except (torch.linalg.LinAlgError, RuntimeError):
            try:
                omega_blk_inv = torch.linalg.solve(
                    omega_blk + eps * eye_d, eye_d.expand_as(omega_blk)
                )
            except (torch.linalg.LinAlgError, RuntimeError):
                omega_blk_inv = torch.linalg.pinv(omega_blk)
        results.append((omega_blk, omega_blk_inv))
        start = end
    return results
```

- [ ] **Step 4: Run — expect 1 passed.**

- [ ] **Step 5: Commit**

```
git add vfe3/geometry/transport.py tests/golden/test_transport_golden.py
git commit -m "feat(geometry): omega_to_block_exp_pairs block slicing, golden-equal to 2.0

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: equivariance + GL(K) det<0 representability property tests

**Files:** Test `tests/test_transport.py`.

- [ ] **Step 1: Property tests** — append to `tests/test_transport.py`:

```python
def test_transported_kl_is_gauge_consistent():
    # KL between query belief and transported key belief is unchanged when both
    # are further pushed by a common group element (KL invariance under common
    # pushforward).
    from vfe3.divergence import kl
    grp = get_group("so_k")(K=4)
    g = torch.Generator().manual_seed(9)
    phi = 0.3 * torch.randn(2, 3, grp.generators.shape[0], generator=g)
    omega = compute_transport_operators(phi, grp, gauge_mode="learned")["Omega"]

    mu_q = torch.randn(2, 3, 4, generator=g)
    mu_k = torch.randn(2, 3, 4, generator=g)
    Aq = torch.randn(2, 3, 4, 4, generator=g)
    Ak = torch.randn(2, 3, 4, 4, generator=g)
    S_q = Aq @ Aq.transpose(-1, -2) + torch.eye(4)
    S_k = Ak @ Ak.transpose(-1, -2) + torch.eye(4)

    mu_kt = transport_mean(omega, mu_k)
    S_kt = transport_covariance(omega, S_k)
    mu_qb = mu_q.unsqueeze(2).expand(2, 3, 3, 4)
    S_qb = S_q.unsqueeze(2).expand(2, 3, 3, 4, 4)
    base = kl(mu_qb, S_qb, mu_kt, S_kt, family="gaussian_full")

    coeff = 0.25 * torch.randn(grp.generators.shape[0], generator=g)
    h = torch.linalg.matrix_exp(torch.einsum("a,aij->ij", coeff, grp.generators))
    mu_qb2 = torch.einsum("kl,bijl->bijk", h, mu_qb)
    mu_kt2 = torch.einsum("kl,bijl->bijk", h, mu_kt)
    S_qb2 = torch.einsum("kl,bijlm,nm->bijkn", h, S_qb, h)
    S_kt2 = torch.einsum("kl,bijlm,nm->bijkn", h, S_kt, h)
    moved = kl(mu_qb2, S_qb2, mu_kt2, S_kt2, family="gaussian_full")
    assert torch.allclose(base, moved, atol=1e-3, rtol=1e-3)


def test_direct_omega_represents_reflection():
    # The phi/exp path lives in GL+(K) (det>0). The direct path can represent a
    # reflection (det<0), which exp(phi) never can.
    from vfe3.geometry.transport import compute_transport_operators_direct
    refl = torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0]))   # det = -1
    omega = refl.expand(1, 2, 4, 4).contiguous()
    out = compute_transport_operators_direct(omega, gauge_mode="learned")
    # Omega_i has negative determinant -> outside GL+(K).
    assert torch.det(out["omega_i"][0, 0]) < 0
```

- [ ] **Step 2: Run — expect 2 passed.**

- [ ] **Step 3: Full suite** `python -m pytest -q` — expect all pass.

- [ ] **Step 4: Commit**

```
git add tests/test_transport.py
git commit -m "test(geometry): gauge consistency + direct-Omega det<0 representability

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage (Phase 2b slice of §4.2):**
- Regime I transport, phi/exp parameterization (GL⁺(K)), group-driven → Tasks 1–2.
- Direct-Ω parameterization (general GL(K), det<0) → Task 3 (golden vs 2.0).
- Belief action μ→Ωμ, Σ→ΩΣΩᵀ (full + diagonal approx) → Task 4.
- Block slicing by `irrep_dims` → Task 5.
- Gauge equivariance + GL(K) det<0 representability → Task 6.

**Deferred (named):**
- Regime II edge-relaxed cocycle (phi and direct forms both take `connection_delta`) → Phase 2c.
- SPD/φ retractions, Fisher preconditioner → Phase 2c.
- RoPE on μ, `VFEHeadMixer` → Phase 2d.
- Block-diagonal exp OPTIMIZATION + contiguous super-block decomposition for cross-coupled groups (head reordering) → Phase 2c.
- Newton-Schulz SO(K) re-orthogonalization, 'constant' gauge, omega det-penalty regularizer (a training-loss term, not transport) → later.

**Placeholder scan:** none.

**Type/name consistency:** `stable_matrix_exp_pair`, `compute_transport_operators(phi, group, *, gauge_mode)`, `compute_transport_operators_direct(omega, *, gauge_mode, eps)`, `transport_mean`, `transport_covariance`, `omega_to_block_exp_pairs(omega, irrep_dims)` consistent across tasks/tests; consumes `GaugeGroup.generators`/`.skew_symmetric`/`.irrep_dims`.
