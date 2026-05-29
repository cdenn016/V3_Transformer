# VFE_3.0 Phase 2a (Transport / Gauge) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `vfe3/geometry/transport.py` — the gauge transport layer (`Ω_ij = exp(φ_i)·exp(−φ_j)` and the sandwich action μ→Ωμ, Σ→ΩΣΩᵀ) — proven numerically equal to VFE_2.0's transport operators by golden tests, with gauge-equivariance property tests.

**Architecture:** Second layer of the bottom-up rebuild, sitting above `divergence.py`. This is the Gaussian-/location-scale-specific `geometry/` layer the spec walls off from the family-agnostic divergence layer. Phase 2a covers the flat learned gauge and the trivial gauge plus the belief action; non-flat connection transport, Newton-Schulz SO(K) re-orthogonalization, the 'constant' per-head gauge, and the omega-direct parameterization are deferred to Phase 2b/2c. SPD/φ retractions and RoPE are separate sub-plans (Phase 2b, 2c).

**Tech Stack:** Python 3, PyTorch (float32; float64 upcast inside matrix_exp for K≥20; CUDA-capable, CPU for tests), pytest. No neural-network components. No CLI parsing.

**Scope note:** One of several Phase 2 sub-plans. Self-contained and testable. Builds on the merged Phase 0/1 trunk (`main`).

**Reference spec:** `docs/superpowers/specs/2026-05-29-vfe3-clean-room-design.md` (§4.2 geometry).

**2.0 references being matched** (in `C:\Users\chris and christine\Desktop\VFE_2.0`):
- `transformer/core/gauge_utils.py::stable_matrix_exp_pair` (lines 53–131) — norm-clamped, float64-upcast exp(M)/exp(−M) pair.
- `transformer/core/transport_ops.py::compute_transport_operators` (lines 285–433) — flat path `Ω_ij = exp(φ_i)exp(−φ_j)`, trivial mode `Ω=I`.
- The diagonal sandwich approximation used in `transformer/core/attention.py:270`: `Σ_t[i,j,k] = Σ_l Ω[i,j,k,l]² σ[j,l]` (the diagonal of `ΩΣΩᵀ`).

---

## Code Style (MANDATORY)

Follow the repo `CLAUDE.md` signature convention: all `torch.Tensor` args first, then `'float | torch.Tensor'`, then undefined floats/ints/bools, then defined floats/ints/bools, then `Optional`, then `**kwargs`; names, type annotations, `=` signs, and trailing `#` comments vertically aligned; blank lines between type groups; tensor shape comments at critical points; keyword-only (`*`) scalar knobs.

---

## File Structure

- Create: `vfe3/geometry/__init__.py` — empty package marker.
- Create: `vfe3/geometry/transport.py` — `stable_matrix_exp_pair`, `compute_transport_operators` (flat learned + trivial), `transport_mean`, `transport_covariance` (full + diagonal approx).
- Create: `tests/test_transport.py` — unit + property tests (φ=0 ⇒ Ω=I; transported full cov is SPD; diagonal approx equals the diagonal of the full sandwich; equivariance round-trip).
- Create: `tests/golden/test_transport_golden.py` — golden equivalence vs 2.0 `stable_matrix_exp_pair` and `compute_transport_operators`.

Generators are passed in as a tensor argument (the generator-construction module is a separate later concern); tests supply small skew-symmetric (SO) and GL basis generators directly, identical to what the 2.0 reference receives.

---

## Task 1: geometry package + `stable_matrix_exp_pair`

**Files:**
- Create: `vfe3/geometry/__init__.py` (empty)
- Create: `vfe3/geometry/transport.py`
- Test: `tests/golden/test_transport_golden.py`

- [ ] **Step 1: Create the package marker**

Create `vfe3/geometry/__init__.py` (empty file).

- [ ] **Step 2: Write the failing golden test**

Create `tests/golden/test_transport_golden.py`:

```python
import pytest
import torch


def _vfe2_gauge():
    # The golden conftest puts VFE_2.0 on sys.path via the vfe2_kl fixture;
    # import the gauge module the same way.
    from transformer.core import gauge_utils
    return gauge_utils


def test_stable_matrix_exp_pair_matches_vfe2(vfe2_kl, device):
    from vfe3.geometry.transport import stable_matrix_exp_pair
    g = torch.Generator(device="cpu").manual_seed(0)
    M = torch.randn(2, 3, 4, 4, generator=g).to(device)  # (B, N, K, K)
    ref_pos, ref_neg = _vfe2_gauge().stable_matrix_exp_pair(M, skew_symmetric=False)
    pos, neg = stable_matrix_exp_pair(M, skew_symmetric=False)
    assert torch.allclose(pos, ref_pos, atol=1e-5, rtol=1e-5)
    assert torch.allclose(neg, ref_neg, atol=1e-5, rtol=1e-5)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/golden/test_transport_golden.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vfe3.geometry.transport'`.

- [ ] **Step 4: Write minimal implementation**

Create `vfe3/geometry/transport.py`:

```python
r"""Gauge transport for VFE_3.0 (Gaussian / location-scale specific).

Transport operator (flat, learned gauge):
    Omega_ij = exp(phi_i . G) @ exp(-phi_j . G),
covering the identity component GL+(K). The belief action is
    mu  -> Omega @ mu,
    Sigma -> Omega @ Sigma @ Omega^T   (the sandwich product).

This layer is deliberately Gaussian/location-scale specialized and kept
separate from the family-agnostic divergence seam.
"""

from typing import Dict, Optional, Tuple

import torch


def stable_matrix_exp_pair(
    matrix:         torch.Tensor,             # (..., d, d) Lie-algebra matrices

    *,
    max_norm:       float = 15.0,
    dim_threshold:  int   = 20,
    skew_symmetric: bool  = False,
    only_forward:   bool  = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    r"""Compute exp(M) and optionally exp(-M) with norm clamping + float64 upcast.

    Ported from VFE_2.0 ``stable_matrix_exp_pair`` (gauge_utils.py:53-131).
    Clamps the Frobenius norm to ``max_norm`` (gradient flows through the
    scale so phi still gets shrink signal), upcasts to float64 when
    ``d >= dim_threshold``, and uses ``exp(-M) = exp(M)^T`` for skew-symmetric
    M. For ``skew_symmetric=True`` exp(-M) is the transpose; for
    ``only_forward=True`` exp(-M) is None.
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

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/golden/test_transport_golden.py -v`
Expected: 1 passed.

- [ ] **Step 6: Commit**

```
git add vfe3/geometry/__init__.py vfe3/geometry/transport.py tests/golden/test_transport_golden.py
git commit -m "feat(geometry): stable_matrix_exp_pair, golden-equal to 2.0

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `compute_transport_operators` (flat learned + trivial)

**Files:**
- Modify: `vfe3/geometry/transport.py`
- Test: `tests/golden/test_transport_golden.py`

- [ ] **Step 1: Write the failing golden test**

Append to `tests/golden/test_transport_golden.py`:

```python
def _skew_generators(K, seed):
    # n_gen = K(K-1)/2 skew-symmetric basis matrices (so(K)).
    g = torch.Generator(device="cpu").manual_seed(seed)
    gens = []
    for i in range(K):
        for j in range(i + 1, K):
            A = torch.zeros(K, K)
            A[i, j] = 1.0
            A[j, i] = -1.0
            gens.append(A)
    return torch.stack(gens, dim=0)  # (n_gen, K, K)


def test_transport_operators_learned_matches_vfe2(vfe2_kl, device):
    from vfe3.geometry.transport import compute_transport_operators
    from transformer.core.transport_ops import (
        compute_transport_operators as ref_ct,
    )
    K = 4
    gens = _skew_generators(K, seed=1).to(device)          # (n_gen, K, K)
    g = torch.Generator(device="cpu").manual_seed(2)
    phi = (0.3 * torch.randn(2, 3, gens.shape[0], generator=g)).to(device)  # (B, N, n_gen)

    ref = ref_ct(phi, gens, enforce_orthogonal=False, gauge_mode="learned")
    got = compute_transport_operators(phi, gens, gauge_mode="learned")

    assert torch.allclose(got["exp_phi"],     ref["exp_phi"],     atol=1e-5, rtol=1e-5)
    assert torch.allclose(got["exp_neg_phi"], ref["exp_neg_phi"], atol=1e-5, rtol=1e-5)
    assert torch.allclose(got["Omega"],       ref["Omega"],       atol=1e-5, rtol=1e-5)


def test_transport_operators_trivial_is_identity(device):
    from vfe3.geometry.transport import compute_transport_operators
    K = 4
    gens = _skew_generators(K, seed=1).to(device)
    phi = torch.zeros(2, 3, gens.shape[0], device=device)
    out = compute_transport_operators(phi, gens, gauge_mode="trivial")
    eye = torch.eye(K, device=device)
    assert torch.allclose(out["Omega"], eye.expand(2, 3, 3, K, K), atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/golden/test_transport_golden.py -k transport_operators -v`
Expected: FAIL with `ImportError: cannot import name 'compute_transport_operators'`.

- [ ] **Step 3: Write minimal implementation**

Append to `vfe3/geometry/transport.py`:

```python
TransportDict = Dict[str, torch.Tensor]


def compute_transport_operators(
    phi:        torch.Tensor,             # (B, N, n_gen) gauge frames
    generators: torch.Tensor,            # (n_gen, K, K) Lie-algebra generators

    *,
    gauge_mode:          str  = "learned",   # 'learned' or 'trivial'
    skew_symmetric:      bool = False,        # exp(-M)=exp(M)^T when generators skew
) -> TransportDict:
    r"""Flat transport operators Omega_ij = exp(phi_i) @ exp(-phi_j).

    Ported from VFE_2.0 ``compute_transport_operators`` (transport_ops.py:285-433),
    flat path only. 'trivial' returns Omega = I (standard attention / gauge
    fixing). Non-flat connection, Newton-Schulz SO(K) re-orthogonalization, the
    'constant' per-head gauge, and omega-direct are deferred to Phase 2b/2c.

    Returns a dict with 'exp_phi' (B,N,K,K), 'exp_neg_phi' (B,N,K,K), and
    'Omega' (B,N,N,K,K).
    """
    B, N, _ = phi.shape
    K = generators.shape[1]
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

    # phi . G : combine gauge frames with generators -> (B, N, K, K)
    phi_matrix = torch.einsum("bna,aij->bnij", phi, generators)
    exp_phi, exp_neg_phi = stable_matrix_exp_pair(
        phi_matrix, skew_symmetric=skew_symmetric
    )
    # Omega_ij = exp(phi_i) @ exp(-phi_j) -> (B, N, N, K, K)
    omega = torch.einsum("bikl,bjlm->bijkm", exp_phi, exp_neg_phi)
    return {"exp_phi": exp_phi, "exp_neg_phi": exp_neg_phi, "Omega": omega}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/golden/test_transport_golden.py -k transport_operators -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```
git add vfe3/geometry/transport.py tests/golden/test_transport_golden.py
git commit -m "feat(geometry): flat transport operators Omega=exp(phi_i)exp(-phi_j), golden-equal to 2.0

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: belief action — `transport_mean` and `transport_covariance`

**Files:**
- Modify: `vfe3/geometry/transport.py`
- Test: `tests/test_transport.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_transport.py`:

```python
import pytest
import torch

from vfe3.geometry.transport import (
    compute_transport_operators,
    transport_covariance,
    transport_mean,
)


def _skew_generators(K):
    gens = []
    for i in range(K):
        for j in range(i + 1, K):
            A = torch.zeros(K, K)
            A[i, j] = 1.0
            A[j, i] = -1.0
            gens.append(A)
    return torch.stack(gens, dim=0)


def test_transport_mean_identity_at_phi_zero():
    K = 4
    gens = _skew_generators(K)
    phi = torch.zeros(2, 3, gens.shape[0])
    omega = compute_transport_operators(phi, gens, gauge_mode="learned")["Omega"]
    g = torch.Generator().manual_seed(0)
    mu = torch.randn(2, 3, K, generator=g)
    mu_t = transport_mean(omega, mu)               # (B, N, N, K)
    # At phi=0, Omega=I, so transported mean at (i, j) equals mu_j.
    assert torch.allclose(mu_t, mu.unsqueeze(1).expand(2, 3, 3, K), atol=1e-5)


def test_transport_covariance_full_is_spd():
    K = 4
    gens = _skew_generators(K)
    g = torch.Generator().manual_seed(1)
    phi = 0.3 * torch.randn(2, 3, gens.shape[0], generator=g)
    omega = compute_transport_operators(phi, gens, gauge_mode="learned")["Omega"]
    A = torch.randn(2, 3, K, K, generator=g)
    sigma = A @ A.transpose(-1, -2) + torch.eye(K)         # (B, N, K, K) SPD
    sigma_t = transport_covariance(omega, sigma)           # (B, N, N, K, K)
    # Symmetric and positive-definite for every (i, j).
    assert torch.allclose(sigma_t, sigma_t.transpose(-1, -2), atol=1e-4)
    eig = torch.linalg.eigvalsh(sigma_t)
    assert (eig > 0).all()


def test_transport_covariance_diag_matches_full_diagonal():
    K = 4
    gens = _skew_generators(K)
    g = torch.Generator().manual_seed(2)
    phi = 0.3 * torch.randn(2, 3, gens.shape[0], generator=g)
    omega = compute_transport_operators(phi, gens, gauge_mode="learned")["Omega"]
    sigma_diag = torch.rand(2, 3, K, generator=g) + 0.1    # (B, N, K)
    # Diagonal approx == diagonal of the full sandwich of diag(sigma).
    full = transport_covariance(omega, torch.diag_embed(sigma_diag))
    approx = transport_covariance(omega, sigma_diag)       # (B, N, N, K)
    assert torch.allclose(approx, torch.diagonal(full, dim1=-2, dim2=-1), atol=1e-5)


def test_transport_covariance_diag_matches_vfe2_formula():
    K = 4
    gens = _skew_generators(K)
    g = torch.Generator().manual_seed(3)
    phi = 0.3 * torch.randn(2, 3, gens.shape[0], generator=g)
    omega = compute_transport_operators(phi, gens, gauge_mode="learned")["Omega"]
    sigma_diag = torch.rand(2, 3, K, generator=g) + 0.1
    approx = transport_covariance(omega, sigma_diag)
    ref = torch.einsum("bijkl,bijkl,bjl->bijk", omega, omega, sigma_diag)
    assert torch.allclose(approx, ref, atol=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_transport.py -v`
Expected: FAIL with `ImportError: cannot import name 'transport_covariance'`.

- [ ] **Step 3: Write minimal implementation**

Append to `vfe3/geometry/transport.py`:

```python
def transport_mean(
    omega: torch.Tensor,             # (B, N, N, K, K) pairwise transport
    mu:    torch.Tensor,             # (B, N, K) source means (key side, index j)
) -> torch.Tensor:
    r"""Apply the gauge action to means: mu_t[i,j] = Omega_ij @ mu_j.

    Returns (B, N, N, K): the key mean mu_j transported into position i.
    """
    return torch.einsum("bijkl,bjl->bijk", omega, mu)


def transport_covariance(
    omega: torch.Tensor,             # (B, N, N, K, K) pairwise transport
    sigma: torch.Tensor,             # (B, N, K) diagonal OR (B, N, K, K) full

    *,
    diagonal_out: Optional[bool] = None,   # default: infer from sigma.ndim
) -> torch.Tensor:
    r"""Apply the sandwich product to covariances: Sigma_t[i,j] = Omega_ij Sigma_j Omega_ij^T.

    Full input ``(B, N, K, K)`` returns the full sandwich ``(B, N, N, K, K)``.
    Diagonal input ``(B, N, K)`` returns the diagonal approximation
    ``(B, N, N, K)`` equal to ``diag(Omega diag(sigma) Omega^T)``, i.e.
    ``Sigma_t[i,j,k] = sum_l Omega_ijkl^2 sigma_jl`` (matches VFE_2.0
    attention.py:270). The full off-diagonal sandwich is not materialized in
    the diagonal path (the documented diagonal approximation for speed).
    """
    is_diag = sigma.dim() == omega.dim() - 2 if diagonal_out is None else diagonal_out
    if is_diag:
        # diag(Omega diag(sigma) Omega^T)_k = sum_l Omega_kl^2 sigma_l
        return torch.einsum("bijkl,bijkl,bjl->bijk", omega, omega, sigma)
    # full sandwich Omega_ij @ Sigma_j @ Omega_ij^T
    return torch.einsum("bijkl,bjlm,bijnm->bijkn", omega, sigma, omega)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_transport.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```
git add vfe3/geometry/transport.py tests/test_transport.py
git commit -m "feat(geometry): gauge action transport_mean / transport_covariance (full + diag)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: gauge-equivariance property test

**Files:**
- Test: `tests/test_transport.py`

This is the spec's required geometry property: transport then divergence is gauge-consistent. Concretely, KL between two beliefs is unchanged when both are transported by the SAME Ω (a left GL action acting on both q and p by the same group element preserves the Gaussian KL).

- [ ] **Step 1: Write the property test**

Append to `tests/test_transport.py`:

```python
def test_full_kl_is_gauge_invariant_under_common_omega():
    # KL(N(mu_q, S_q) || N(mu_p, S_p)) == KL after both are mapped by the same
    # invertible Omega: mu -> Omega mu, S -> Omega S Omega^T. This is the gauge
    # invariance the transport layer must preserve (full covariance).
    from vfe3.divergence import kl
    g = torch.Generator().manual_seed(7)
    K = 4
    mu_q = torch.randn(5, K, generator=g)
    mu_p = torch.randn(5, K, generator=g)
    Aq = torch.randn(5, K, K, generator=g)
    Ap = torch.randn(5, K, K, generator=g)
    S_q = Aq @ Aq.transpose(-1, -2) + torch.eye(K)
    S_p = Ap @ Ap.transpose(-1, -2) + torch.eye(K)

    base = kl(mu_q, S_q, mu_p, S_p, family="gaussian_full")

    # A single shared invertible Omega per row (well-conditioned).
    W = torch.randn(5, K, K, generator=g)
    omega = W @ W.transpose(-1, -2) + 2.0 * torch.eye(K)   # SPD -> invertible
    mu_q2 = torch.einsum("nkl,nl->nk", omega, mu_q)
    mu_p2 = torch.einsum("nkl,nl->nk", omega, mu_p)
    S_q2 = omega @ S_q @ omega.transpose(-1, -2)
    S_p2 = omega @ S_p @ omega.transpose(-1, -2)

    moved = kl(mu_q2, S_q2, mu_p2, S_p2, family="gaussian_full")
    assert torch.allclose(base, moved, atol=1e-3, rtol=1e-3)
```

- [ ] **Step 2: Run the property test**

Run: `python -m pytest tests/test_transport.py::test_full_kl_is_gauge_invariant_under_common_omega -v`
Expected: 1 passed.

- [ ] **Step 3: Run the full suite**

Run: `python -m pytest -q`
Expected: all pass (Phase 0/1 + Phase 2a; golden tests pass with `..\VFE_2.0` present).

- [ ] **Step 4: Commit**

```
git add tests/test_transport.py
git commit -m "test(geometry): gauge invariance of full KL under a common Omega

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage (Phase 2a slice):**
- §4.2 transport `Omega_ij = exp(phi_i) exp(-phi_j)` → Tasks 1–2 (golden vs 2.0).
- §4.2 sandwich action μ→Ωμ, Σ→ΩΣΩᵀ → Task 3 (`transport_mean`, `transport_covariance`; full + documented diagonal approximation).
- §7 gauge-equivariance property test → Task 4.
- Modularity / clean separation: `geometry/` holds only transport (no divergence, no family logic); divergence stays in `divergence.py`.

**Deferred (named so they are not forgotten), to Phase 2b/2c:**
- Non-flat connection transport `Omega_ij = exp(phi_i) exp(alpha delta_ij) exp(-phi_j)` and `cocycle_relaxation`.
- Newton-Schulz SO(K) re-orthogonalization (`enforce_orthogonal`).
- 'constant' per-head gauge and the omega-direct parameterization.
- SPD retraction (`retract_spd_torch`, `retract_spd_diagonal_torch`), φ Lie-algebra retraction (`_retract_phi`), Fisher/natural-gradient preconditioner — Phase 2b.
- RoPE on μ and `_apply_rope_to_covariance` — Phase 2c.

**Placeholder scan:** none — every step has runnable code or commands.

**Type/name consistency:** `stable_matrix_exp_pair`, `compute_transport_operators` (returns dict with `exp_phi`/`exp_neg_phi`/`Omega`), `transport_mean`, `transport_covariance` are used consistently across tasks and tests. Generators are passed as a tensor argument throughout; tests build skew-symmetric so(K) generators identically for V3 and the 2.0 reference.
