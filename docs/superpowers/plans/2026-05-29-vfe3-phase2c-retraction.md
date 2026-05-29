# VFE_3.0 Phase 2c (Manifold Retractions + Fisher Preconditioner) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build `vfe3/geometry/retraction.py` — the SPD-manifold retractions (diagonal + full, affine-invariant exp map) and the Fisher / natural-gradient preconditioner — golden-equal to VFE_2.0, with SPD-preservation property tests.

**Architecture:** Third geometry sub-phase. These are the belief-manifold *update* operations the E-step (Phase 6) needs: the SPD retraction keeps Σ on the SPD manifold under a tangent step; the Fisher preconditioner converts Euclidean (μ,σ) gradients to natural gradients (`nat_μ = Σ∇μ`, `nat_σ = 2Σ∇σΣ`). Self-contained (no `math_utils` dependency). The φ Lie-algebra retraction (which needs the `math_utils` BCH/det-control Lie-retraction subsystem), Regime II, and the block-diagonal-exp performance optimization are the *next* phase.

**Tech Stack:** Python 3, PyTorch (float32; eigendecomposition forced float32), pytest. No NN. No CLI.

**Reference spec:** `docs/superpowers/specs/2026-05-29-vfe3-clean-room-design.md` (§4.2). Prereq: Phases 2a/2b on branch.

**2.0 references** (`C:\Users\chris and christine\Desktop\VFE_2.0`):
- `vfe_utils.py::retract_spd_diagonal_torch` (727–782) — `σ_new = σ·exp(τ·clamp(δσ/σ))`, clamp `[eps, sigma_max]`.
- `vfe_utils.py::retract_spd_torch` (635–724) — affine-invariant SPD exp map.
- `vfe_gradients.py::compute_natural_gradient_gpu` (≈1936–1994) — Fisher preconditioner.

---

## Code Style (MANDATORY)

Repo `CLAUDE.md`: tensors first, type-grouped with blank lines, names/types/`=`/comments aligned, keyword-only `*` knobs.

---

## File Structure

- Create: `vfe3/geometry/retraction.py` — `retract_spd_diagonal`, `retract_spd_full`, `natural_gradient`.
- Modify: `tests/golden/conftest.py` — add `vfe2_retract` fixture (vfe_utils + vfe_gradients).
- Create: `tests/golden/test_retraction_golden.py` — golden vs 2.0.
- Create: `tests/test_retraction.py` — SPD-preservation + Fisher-formula property tests.

---

## Task 1: golden fixture + `retract_spd_diagonal`

**Files:** Create `vfe3/geometry/retraction.py`; Modify `tests/golden/conftest.py`; Test `tests/golden/test_retraction_golden.py`.

- [ ] **Step 1: Add fixture** — append to `tests/golden/conftest.py`:

```python
@pytest.fixture(scope="session")
def vfe2_retract():
    """Return the 2.0 vfe_utils + vfe_gradients modules, or skip if unavailable."""
    root = _vfe2_root()
    if root is None:
        pytest.skip("VFE_2.0 checkout not found (set VFE2_ROOT)")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from transformer.core import vfe_gradients, vfe_utils
    except ImportError as exc:
        pytest.skip(f"could not import VFE_2.0 retraction modules: {exc}")
    return {"vfe_utils": vfe_utils, "vfe_gradients": vfe_gradients}
```

- [ ] **Step 2: Failing golden test** — create `tests/golden/test_retraction_golden.py`:

```python
import pytest
import torch


def test_retract_spd_diagonal_matches_vfe2(vfe2_retract):
    from vfe3.geometry.retraction import retract_spd_diagonal
    g = torch.Generator(device="cpu").manual_seed(0)
    sigma = torch.rand(2, 3, 5, generator=g) + 0.1
    delta = 0.5 * torch.randn(2, 3, 5, generator=g)
    ref = vfe2_retract["vfe_utils"].retract_spd_diagonal_torch(
        sigma, delta, step_size=1.0, trust_region=5.0, eps=1e-6, sigma_max=5.0
    )
    got = retract_spd_diagonal(sigma, delta)
    assert torch.allclose(got, ref, atol=1e-5, rtol=1e-5)
```

- [ ] **Step 3: Run — expect FAIL** (`ModuleNotFoundError`).
Run: `python -m pytest tests/golden/test_retraction_golden.py -v`

- [ ] **Step 4: Create `vfe3/geometry/retraction.py`:**

```python
r"""SPD-manifold retractions + Fisher natural-gradient preconditioner (VFE_3.0).

The SPD retraction keeps Sigma on the SPD manifold under a tangent update; the
Fisher preconditioner converts Euclidean (mu, sigma) gradients to natural
gradients. Ported from VFE_2.0 vfe_utils.py / vfe_gradients.py. The phi
Lie-algebra retraction is a separate phase.
"""

from typing import Tuple

import torch


def retract_spd_diagonal(
    sigma_diag:   torch.Tensor,             # (..., K) diagonal variances
    delta_sigma:  torch.Tensor,             # (..., K) diagonal tangent

    *,
    step_size:    float = 1.0,
    trust_region: float = 5.0,
    eps:          float = 1e-6,
    sigma_max:    float = 5.0,
) -> torch.Tensor:
    r"""Diagonal SPD retraction sigma_new = sigma * exp(tau * clamp(dsigma/sigma)).

    Positivity by construction (exp > 0); clamped to [eps, sigma_max]. Ported
    from VFE_2.0 retract_spd_diagonal_torch (vfe_utils.py:727).
    """
    orig_dtype = sigma_diag.dtype
    with torch.amp.autocast('cuda', enabled=False):
        sigma_safe = sigma_diag.float().clamp(min=eps)
        delta_sigma = delta_sigma.float()
        whitened = delta_sigma / sigma_safe
        if trust_region is not None and trust_region > 0:
            whitened = whitened.clamp(-trust_region, trust_region)
        exp_arg = (step_size * whitened).clamp(-50.0, 50.0)
        sigma_new = sigma_safe * torch.exp(exp_arg)
    return sigma_new.clamp(min=eps, max=sigma_max).to(orig_dtype)
```

- [ ] **Step 5: Run — expect 1 passed.**

- [ ] **Step 6: Commit**

```
git add vfe3/geometry/retraction.py tests/golden/conftest.py tests/golden/test_retraction_golden.py
git commit -m "feat(geometry): diagonal SPD retraction, golden-equal to 2.0

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `retract_spd_full` (affine-invariant exp map)

**Files:** Modify `vfe3/geometry/retraction.py`; Test `tests/golden/test_retraction_golden.py`.

- [ ] **Step 1: Failing golden test** — append:

```python
def test_retract_spd_full_matches_vfe2(vfe2_retract):
    from vfe3.geometry.retraction import retract_spd_full
    g = torch.Generator(device="cpu").manual_seed(1)
    A = torch.randn(2, 3, 4, 4, generator=g)
    sigma = A @ A.transpose(-1, -2) + torch.eye(4)        # SPD, well-conditioned
    D = torch.randn(2, 3, 4, 4, generator=g)
    delta = 0.3 * (D + D.transpose(-1, -2))               # symmetric tangent
    ref = vfe2_retract["vfe_utils"].retract_spd_torch(
        sigma, delta, step_size=1.0, trust_region=2.0, eps=1e-6, sigma_max=5.0
    )
    got = retract_spd_full(sigma, delta)
    assert torch.allclose(got, ref, atol=1e-3, rtol=1e-3)
```

- [ ] **Step 2: Run — expect FAIL** (`ImportError`).

- [ ] **Step 3: Implement** — append to `vfe3/geometry/retraction.py`:

```python
def retract_spd_full(
    sigma:        torch.Tensor,             # (..., K, K) SPD covariances
    delta_sigma:  torch.Tensor,             # (..., K, K) symmetric tangent

    *,
    step_size:    float = 1.0,
    trust_region: float = 2.0,
    eps:          float = 1e-6,
    sigma_max:    float = 5.0,
) -> torch.Tensor:
    r"""Full SPD retraction via the affine-invariant exponential map.

        Sigma_new = S^{1/2} exp(S^{-1/2} (tau dSigma) S^{-1/2}) S^{1/2},
    with a Frobenius trust region on the whitened tangent and an eigenvalue
    floor/ceiling [eps, sigma_max^2]. Ported from VFE_2.0 retract_spd_torch
    (vfe_utils.py:635). Uses torch.linalg.eigh; 2.0's gap-regularized custom
    backward (_safe_eigh) is a gradient-stability feature deferred to a later
    hardening pass (forward values match on well-conditioned inputs).
    """
    orig_shape = sigma.shape
    orig_dtype = sigma.dtype
    if sigma.dim() == 4:
        B, N, K, _ = sigma.shape
        sigma = sigma.reshape(B * N, K, K)
        delta_sigma = delta_sigma.reshape(B * N, K, K)

    with torch.amp.autocast('cuda', enabled=False):
        sigma = sigma.float()
        delta_sigma = delta_sigma.float()
        sigma = 0.5 * (sigma + sigma.transpose(-1, -2))
        delta_sigma = 0.5 * (delta_sigma + delta_sigma.transpose(-1, -2))

        eigenvalues, eigenvectors = torch.linalg.eigh(sigma)
        eigenvalues = eigenvalues.clamp(min=eps)
        sqrt_eig     = torch.sqrt(eigenvalues)
        inv_sqrt_eig = 1.0 / sqrt_eig
        sigma_sqrt     = eigenvectors * sqrt_eig.unsqueeze(-2)     @ eigenvectors.transpose(-1, -2)
        sigma_inv_sqrt = eigenvectors * inv_sqrt_eig.unsqueeze(-2) @ eigenvectors.transpose(-1, -2)

        R = sigma_inv_sqrt @ (step_size * delta_sigma) @ sigma_inv_sqrt
        R = 0.5 * (R + R.transpose(-1, -2))
        if trust_region is not None and trust_region > 0:
            R_norm = torch.linalg.norm(R, ord='fro', dim=(-2, -1), keepdim=True)
            R = R * torch.clamp(trust_region / (R_norm + eps), max=1.0)

        R_eval, R_evec = torch.linalg.eigh(R)
        R_eval = R_eval.clamp(-50.0, 50.0)
        exp_R = R_evec * torch.exp(R_eval).unsqueeze(-2) @ R_evec.transpose(-1, -2)

        sigma_new = sigma_sqrt @ exp_R @ sigma_sqrt
        sigma_new = 0.5 * (sigma_new + sigma_new.transpose(-1, -2))

        eig_new, vec_new = torch.linalg.eigh(sigma_new)
        eig_new = eig_new.clamp(min=eps, max=sigma_max * sigma_max)
        sigma_new = vec_new * eig_new.unsqueeze(-2) @ vec_new.transpose(-1, -2)

    sigma_new = sigma_new.to(orig_dtype)
    if len(orig_shape) == 4:
        sigma_new = sigma_new.reshape(orig_shape)
    return sigma_new
```

- [ ] **Step 4: Run — expect 1 passed.** (atol 1e-3 because eigh sign/order differs from 2.0's `_safe_eigh`, but the reconstructed `V diag V^T` is invariant.)

- [ ] **Step 5: Commit**

```
git add vfe3/geometry/retraction.py tests/golden/test_retraction_golden.py
git commit -m "feat(geometry): full SPD retraction (affine-invariant exp map), golden vs 2.0

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `natural_gradient` (Fisher preconditioner)

**Files:** Modify `vfe3/geometry/retraction.py`; Test `tests/golden/test_retraction_golden.py`.

- [ ] **Step 1: Failing golden test** — append:

```python
def test_natural_gradient_diag_matches_vfe2(vfe2_retract):
    from vfe3.geometry.retraction import natural_gradient
    g = torch.Generator(device="cpu").manual_seed(2)
    sigma = torch.rand(2, 3, 5, generator=g) + 0.1
    gmu = torch.randn(2, 3, 5, generator=g)
    gsig = torch.randn(2, 3, 5, generator=g)
    rmu, rsig = vfe2_retract["vfe_gradients"].compute_natural_gradient_gpu(gmu, gsig, sigma)
    nmu, nsig = natural_gradient(gmu, gsig, sigma)
    assert torch.allclose(nmu, rmu, atol=1e-5, rtol=1e-5)
    assert torch.allclose(nsig, rsig, atol=1e-5, rtol=1e-5)


def test_natural_gradient_full_matches_vfe2(vfe2_retract):
    from vfe3.geometry.retraction import natural_gradient
    g = torch.Generator(device="cpu").manual_seed(3)
    A = torch.randn(2, 3, 4, 4, generator=g)
    sigma = A @ A.transpose(-1, -2) + torch.eye(4)
    gmu = torch.randn(2, 3, 4, generator=g)
    Gs = torch.randn(2, 3, 4, 4, generator=g)
    gsig = 0.5 * (Gs + Gs.transpose(-1, -2))
    rmu, rsig = vfe2_retract["vfe_gradients"].compute_natural_gradient_gpu(gmu, gsig, sigma)
    nmu, nsig = natural_gradient(gmu, gsig, sigma)
    assert torch.allclose(nmu, rmu, atol=1e-4, rtol=1e-4)
    assert torch.allclose(nsig, rsig, atol=1e-4, rtol=1e-4)
```

- [ ] **Step 2: Run — expect FAIL** (`ImportError`).

- [ ] **Step 3: Implement** — append to `vfe3/geometry/retraction.py`:

```python
def natural_gradient(
    grad_mu:    torch.Tensor,             # (..., K) Euclidean grad wrt mu
    grad_sigma: torch.Tensor,             # (..., K) or (..., K, K) Euclidean grad wrt sigma
    sigma_q:    torch.Tensor,             # (..., K) diagonal OR (..., K, K) full covariance

    *,
    eps:        float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Fisher preconditioner: Euclidean -> natural gradient for a Gaussian.

        nat_mu    = Sigma grad_mu
        nat_sigma = 2 Sigma grad_sigma Sigma   (diagonal: 2 sigma^2 grad_sigma)
    The Fisher metric on Sigma is g(dS1,dS2) = (1/2) tr(S^-1 dS1 S^-1 dS2), so
    g^{kk} = 2 sigma_k^2 in the diagonal case. Ported from VFE_2.0
    compute_natural_gradient_gpu (vfe_gradients.py:1936).
    """
    is_diagonal = sigma_q.dim() == grad_mu.dim()
    orig_dtype = sigma_q.dtype
    with torch.amp.autocast('cuda', enabled=False):
        sigma_q = sigma_q.float()
        grad_mu = grad_mu.float()
        grad_sigma = grad_sigma.float()
        if is_diagonal:
            sigma_safe = sigma_q.clamp(min=eps)
            nat_grad_mu    = sigma_safe * grad_mu
            nat_grad_sigma = 2.0 * sigma_safe * sigma_safe * grad_sigma
        else:
            nat_grad_mu    = torch.einsum('...ij,...j->...i', sigma_q, grad_mu)
            nat_grad_sigma = 2.0 * torch.einsum('...ij,...jk,...kl->...il', sigma_q, grad_sigma, sigma_q)
            nat_grad_sigma = 0.5 * (nat_grad_sigma + nat_grad_sigma.transpose(-1, -2))
    return nat_grad_mu.to(orig_dtype), nat_grad_sigma.to(orig_dtype)
```

- [ ] **Step 4: Run — expect 2 passed.**

- [ ] **Step 5: Commit**

```
git add vfe3/geometry/retraction.py tests/golden/test_retraction_golden.py
git commit -m "feat(geometry): Fisher natural-gradient preconditioner, golden-equal to 2.0

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: SPD-preservation + Fisher-formula property tests

**Files:** Test `tests/test_retraction.py`.

- [ ] **Step 1: Property tests** — create `tests/test_retraction.py`:

```python
import pytest
import torch

from vfe3.geometry.retraction import (
    natural_gradient,
    retract_spd_diagonal,
    retract_spd_full,
)


def test_diagonal_retraction_positive_and_bounded():
    g = torch.Generator().manual_seed(0)
    sigma = torch.rand(4, 6, generator=g) + 0.1
    delta = 5.0 * torch.randn(4, 6, generator=g)          # large step -> trust region
    out = retract_spd_diagonal(sigma, delta, sigma_max=5.0)
    assert (out >= 1e-6).all()
    assert (out <= 5.0 + 1e-6).all()


def test_full_retraction_stays_spd():
    g = torch.Generator().manual_seed(1)
    A = torch.randn(3, 4, 4, generator=g)
    sigma = A @ A.transpose(-1, -2) + torch.eye(4)
    D = torch.randn(3, 4, 4, generator=g)
    delta = 0.5 * (D + D.transpose(-1, -2))
    out = retract_spd_full(sigma, delta)
    assert torch.allclose(out, out.transpose(-1, -2), atol=1e-4)
    assert (torch.linalg.eigvalsh(out) > 0).all()


def test_full_retraction_identity_tangent_is_identity():
    g = torch.Generator().manual_seed(2)
    A = torch.randn(3, 4, 4, generator=g)
    sigma = A @ A.transpose(-1, -2) + torch.eye(4)
    zero = torch.zeros(3, 4, 4)
    out = retract_spd_full(sigma, zero)
    # exp(0) = I -> Sigma_new = Sigma (up to the eigen floor/ceiling, here inactive).
    assert torch.allclose(out, sigma, atol=1e-3)


def test_natural_gradient_diagonal_formula():
    g = torch.Generator().manual_seed(3)
    sigma = torch.rand(4, 5, generator=g) + 0.1
    gmu = torch.randn(4, 5, generator=g)
    gsig = torch.randn(4, 5, generator=g)
    nmu, nsig = natural_gradient(gmu, gsig, sigma)
    assert torch.allclose(nmu, sigma * gmu, atol=1e-6)
    assert torch.allclose(nsig, 2.0 * sigma * sigma * gsig, atol=1e-6)
```

- [ ] **Step 2: Run** `python -m pytest tests/test_retraction.py -v` — expect 4 passed.

- [ ] **Step 3: Full suite** `python -m pytest -q` — expect all pass.

- [ ] **Step 4: Commit**

```
git add tests/test_retraction.py
git commit -m "test(geometry): SPD-preservation + Fisher-formula properties

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage (Phase 2c slice of §4.2):**
- SPD exponential-map retraction (diagonal + full) → Tasks 1–2 (golden vs 2.0).
- Fisher / natural-gradient preconditioner → Task 3 (golden vs 2.0).
- SPD preservation + Fisher formula → Task 4.

**Deferred (named, to the next geometry sub-phase):**
- φ Lie-algebra retraction (`_retract_phi` + `retract_glK_torch`/`retract_soN_torch` + det control from `math_utils/generators/`) — its own focused phase (BCH expansion + det control subsystem).
- Regime II edge-relaxed cocycle (`connection_delta`) in transport.
- Block-diagonal-exp performance optimization + cross-coupled super-block decomposition.
- RoPE on μ, `VFEHeadMixer`.
- The gap-regularized `_safe_eigh` custom backward (gradient stability) for the full retraction.

**Placeholder scan:** none.

**Type/name consistency:** `retract_spd_diagonal(sigma_diag, delta_sigma, *, step_size, trust_region, eps, sigma_max)`, `retract_spd_full(sigma, delta_sigma, *, ...)`, `natural_gradient(grad_mu, grad_sigma, sigma_q, *, eps)` consistent across tasks and tests.
