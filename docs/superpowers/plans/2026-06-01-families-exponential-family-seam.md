# families/ ExponentialFamily seam + parameter-object divergence signature — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the four-tensor `(mu_q, sigma_q, mu_t, sigma_t)` divergence signature with a family-typed parameter object, housed behind a new `vfe3/families/` exponential-family layer, so a non-`(mean, covariance)` family can pass the divergence interface — with zero behavior change to the live Gaussian path.

**Architecture:** A `BeliefParams` ABC (parameter container + family math) with `DiagonalGaussian` / `FullGaussian` subclasses carrying the existing closed forms verbatim, plus a generic Renyi/KL-from-`A(theta)` for families that define only the log-partition. Divergence functions take and return parameter objects. The gauge/transport layer and the hand gradient kernel stay tensor-based and Gaussian; parameter objects are constructed only at the divergence call boundary. Built additively (Phase 1), routed through the existing tensor API to prove byte-identity (Phase 2), then the signature is flipped and consumers converted (Phase 3).

**Tech Stack:** Python 3.14, PyTorch (float32, CPU/CUDA), pytest. Spec: `docs/superpowers/specs/2026-06-01-families-exponential-family-seam-design.md`.

**Verification discipline (CLAUDE.md):** run pytest with NO extra `-q` (pyproject sets `-q`; a second makes `-qq` and hides the count). Read pass counts from `--junitxml` or the `N passed` line, never from memory. The full suite is 259 tests before this plan; it must stay green throughout with Gaussian numerics byte-identical.

---

## File Structure

- Create `vfe3/families/__init__.py` — package marker.
- Create `vfe3/families/base.py` — `BeliefParams` ABC; family registry (`register_family`, `get_family`, `family_cov_kind`, `divergence_families`); functional registry (`register_functional`, `get_functional`); `renyi`/`kl` functionals with closed-form-vs-generic dispatch; the generic `_renyi_from_log_partition`; `safe_kl_clamp` + `_warn_alpha_gt_one` + `_logdet_chol` (moved from `divergence.py`).
- Create `vfe3/families/gaussian.py` — `DiagonalGaussian`, `FullGaussian` (closed forms ported verbatim; natural/log_partition_at/entropy/expected_statistic; block/broadcast_over_keys); register both.
- Modify `vfe3/divergence.py` — Phase 2: delegate to families (keep tensor API). Phase 3: re-export `renyi`/`kl` (now param-typed), `safe_kl_clamp`, `family_cov_kind`, `divergence_families`, `register_functional`/`get_functional`; remove the moved kernel bodies.
- Modify `vfe3/free_energy.py`, `vfe3/inference/e_step.py`, `vfe3/gradients/oracle.py`, `vfe3/model/prior_bank.py`, `vfe3/model/model.py` — Phase 3: build/pass parameter objects at the divergence boundary.
- Tests: `tests/test_families.py` (new), plus edits to `tests/test_divergence.py`, `tests/test_free_energy.py` for the signature flip.
- Untouched: `vfe3/gradients/kernels.py` (hand kernel stays tensor-based), `vfe3/geometry/*` (transport stays Gaussian/tensor-based), `vfe3/config.py` (the `family_cov_kind`-based validation already added stays correct once `family_cov_kind` is re-exported from `divergence.py`).

---

## Phase 1 — the families/ package (additive, parallel, fully tested)

### Task 1: families package + `BeliefParams` ABC + family registry

**Files:**
- Create: `vfe3/families/__init__.py`
- Create: `vfe3/families/base.py`
- Test: `tests/test_families.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_families.py
import math

import pytest
import torch


def test_family_registry_register_get_and_cov_kind():
    from vfe3.families.base import (
        BeliefParams, register_family, get_family, family_cov_kind, divergence_families,
    )

    class _ToyParams(BeliefParams):
        cov_kind = "diagonal"
        def __init__(self, x): self.x = x
        def coordinate_dim(self): return self.x.shape[-1]
        def block(self, start, end): return _ToyParams(self.x[..., start:end])
        def broadcast_over_keys(self): return _ToyParams(self.x.unsqueeze(-2))
        def natural(self): return (self.x,)
        @classmethod
        def log_partition_at(cls, theta): return theta[0].sum(dim=-1)
        def entropy(self): return self.x.sum(dim=-1) * 0.0

    register_family("toy_reg_test")(_ToyParams)
    try:
        assert get_family("toy_reg_test") is _ToyParams
        assert family_cov_kind("toy_reg_test") == "diagonal"
        assert "toy_reg_test" in divergence_families()
    finally:
        from vfe3.families.base import _FAMILIES
        _FAMILIES.pop("toy_reg_test", None)


def test_family_cov_kind_unregistered_raises():
    from vfe3.families.base import family_cov_kind
    with pytest.raises(KeyError):
        family_cov_kind("no_such_family")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_families.py -p no:cacheprovider`
Expected: FAIL (ModuleNotFoundError: no `vfe3.families`).

- [ ] **Step 3: Write minimal implementation**

```python
# vfe3/families/__init__.py
```
(empty file)

```python
# vfe3/families/base.py
r"""The exponential-family parameter layer for VFE_3.0.

A ``BeliefParams`` is a batched parameter container plus the family's math
(natural<->moment, log-partition A(theta), entropy, divergences). The divergence
functional ``renyi`` (KL = alpha 1) dispatches on the parameter object: a family with a
``renyi_closed_form`` method uses it (the pinned Gaussian moment forms); a family that
defines only ``log_partition_at`` (and ``natural``/``expected_statistic``) gets the
generic Bregman/Renyi-from-A divergence for free. This is the seam a new exponential
family slots in behind -- by writing-and-registering a subclass, never editing call sites.
"""

import warnings
from abc import ABC, abstractmethod
from typing import Callable, ClassVar, Dict, Tuple, Type

import torch


def safe_kl_clamp(
    kl:     torch.Tensor,

    *,
    kl_max: float = 100.0,
) -> torch.Tensor:
    r"""Clamp to [0, kl_max]; map NaN/+inf -> kl_max, -inf -> 0."""
    kl = kl.clamp(min=0.0, max=kl_max)
    return kl.nan_to_num(nan=kl_max, posinf=kl_max, neginf=0.0)


def _warn_alpha_gt_one(alpha: float, family: str) -> None:
    r"""Warn that alpha > 1 leaves the convex regime of the Renyi blend."""
    warnings.warn(
        f"renyi: alpha={alpha} > 1 (family={family!r}) leaves the convex regime; "
        f"the blend (1-alpha)*Sigma_q + alpha*Sigma_t may be non-positive-definite "
        f"(diagonal clamps; full may fail Cholesky and return NaN).",
        RuntimeWarning,
        stacklevel=3,
    )


def _logdet_chol(L: torch.Tensor) -> torch.Tensor:
    r"""log|Sigma| for SPD Sigma = L Lᵀ from its Cholesky factor L."""
    return 2.0 * torch.log(
        torch.diagonal(L, dim1=-2, dim2=-1).clamp(min=1e-12)
    ).sum(dim=-1)


class BeliefParams(ABC):
    r"""Batched parameters of an exponential family, with the family's behavior.

    Concrete subclasses hold the family's tensors (with arbitrary leading batch dims and a
    trailing coordinate structure) and implement the interface below. ``cov_kind`` is the
    single source of truth for the covariance structure (replacing name sniffing).
    """

    cov_kind: ClassVar[str]

    @abstractmethod
    def coordinate_dim(self) -> int:
        r"""K, the number of belief coordinates."""

    @abstractmethod
    def block(self, start: int, end: int) -> "BeliefParams":
        r"""The parameters restricted to coordinate block [start, end) (per-irrep slice)."""

    @abstractmethod
    def broadcast_over_keys(self) -> "BeliefParams":
        r"""Insert a singleton key axis so a query (..., N, K) broadcasts against keys
        (..., N, N, K) in the pairwise energy."""

    @abstractmethod
    def natural(self) -> Tuple[torch.Tensor, ...]:
        r"""Natural parameters theta from these (moment) parameters."""

    @classmethod
    @abstractmethod
    def log_partition_at(cls, theta: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        r"""Log-partition A(theta) at arbitrary natural coordinates theta."""

    @abstractmethod
    def entropy(self) -> torch.Tensor:
        r"""Differential entropy H of this distribution."""


_FAMILIES: Dict[str, Type[BeliefParams]] = {}


def register_family(name: str) -> Callable:
    r"""Register a ``BeliefParams`` subclass under ``name`` (the config ``family`` value)."""
    def _wrap(cls: Type[BeliefParams]) -> Type[BeliefParams]:
        _FAMILIES[name] = cls
        return cls
    return _wrap


def get_family(name: str) -> Type[BeliefParams]:
    r"""The registered ``BeliefParams`` subclass for ``name`` (KeyError if absent)."""
    if name not in _FAMILIES:
        raise KeyError(f"no family registered under {name!r}; available: {sorted(_FAMILIES)}")
    return _FAMILIES[name]


def family_cov_kind(name: str) -> str:
    r"""Covariance structure ("diagonal" | "full") of family ``name``, from its subclass."""
    return get_family(name).cov_kind


def divergence_families() -> Tuple[str, ...]:
    r"""Registered family names (the valid ``family`` config values)."""
    return tuple(sorted(_FAMILIES))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_families.py -p no:cacheprovider`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add vfe3/families/__init__.py vfe3/families/base.py tests/test_families.py
git commit -m "feat(families): BeliefParams ABC + family registry (Phase 1)"
```

---

### Task 2: functional registry + `renyi`/`kl` + generic Renyi/KL-from-A

**Files:**
- Modify: `vfe3/families/base.py`
- Test: `tests/test_families.py`

- [ ] **Step 1: Write the failing test** (toy univariate-exponential family exercises the GENERIC A-path: p(x;lambda)=lambda exp(-lambda x), natural eta=-lambda, A(eta)=-log(-eta), E[T]=1/lambda; KL(l1||l2)=log(l1/l2)+l2/l1-1; Renyi closed form 1/(a-1)[-log(a l1+(1-a) l2)+a log l1+(1-a) log l2])

```python
# tests/test_families.py  (append)
class _ExpFamily(BeliefParams):
    """Univariate exponential, parameter lam>0. Defines ONLY natural/log_partition/
    expected_statistic -- no moment closed form, so it drives the generic A-path."""
    cov_kind = "diagonal"
    def __init__(self, lam): self.lam = lam
    def coordinate_dim(self): return 1
    def block(self, start, end): return _ExpFamily(self.lam)
    def broadcast_over_keys(self): return _ExpFamily(self.lam.unsqueeze(-1))
    def natural(self): return (-self.lam,)
    @classmethod
    def log_partition_at(cls, theta): return -torch.log(-theta[0])
    def expected_statistic(self): return (1.0 / self.lam,)
    def entropy(self): return 1.0 - torch.log(self.lam)


def test_generic_kl_from_A_matches_exponential_closed_form():
    from vfe3.families.base import kl
    from tests.test_families import _ExpFamily  # defined above in this module
    l1 = torch.tensor([2.0, 0.5, 1.0])
    l2 = torch.tensor([1.0, 1.5, 1.0])
    got = kl(_ExpFamily(l1), _ExpFamily(l2))
    want = torch.log(l1 / l2) + l2 / l1 - 1.0
    assert torch.allclose(got, want, atol=1e-5), (got, want)


def test_generic_renyi_from_A_matches_exponential_closed_form():
    from vfe3.families.base import renyi
    l1 = torch.tensor([2.0, 0.5])
    l2 = torch.tensor([1.0, 1.5])
    for a in (0.3, 0.7):
        got = renyi(_ExpFamily(l1), _ExpFamily(l2), alpha=a)
        want = (-torch.log(a * l1 + (1.0 - a) * l2) + a * torch.log(l1)
                + (1.0 - a) * torch.log(l2)) / (a - 1.0)
        assert torch.allclose(got, want, atol=1e-5), (a, got, want)
```

Note: `_ExpFamily` is defined at module level in `tests/test_families.py` (move the class above the test functions; the `from tests.test_families import _ExpFamily` line is illustrative — reference the module-level class directly).

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_families.py -k generic -p no:cacheprovider`
Expected: FAIL (ImportError: cannot import name `kl`/`renyi`).

- [ ] **Step 3: Write minimal implementation** (append to `vfe3/families/base.py`)

```python
_FUNCTIONALS: Dict[str, Callable] = {}


def register_functional(name: str) -> Callable:
    r"""Register a divergence functional (renyi, ...) under ``name`` (the ``divergence_family``)."""
    def _wrap(fn: Callable) -> Callable:
        _FUNCTIONALS[name] = fn
        return fn
    return _wrap


def get_functional(name: str) -> Callable:
    if name not in _FUNCTIONALS:
        raise KeyError(f"no functional registered under {name!r}; available: {sorted(_FUNCTIONALS)}")
    return _FUNCTIONALS[name]


def _renyi_from_log_partition(
    q:       BeliefParams,
    p:       BeliefParams,

    *,
    alpha:   float,
    kl_max:  float,
    eps:     float,
) -> torch.Tensor:
    r"""Generic Renyi/KL from the log-partition (for families with no closed form).

    alpha != 1:  R = 1/(alpha-1) [ A(alpha*tq + (1-alpha)*tp) - alpha*A(tq) - (1-alpha)*A(tp) ].
    alpha == 1:  KL = A(tp) - A(tq) - <gradA(tq), tp - tq>, gradA(tq) = E_q[T] (expected_statistic).
    """
    cls = type(q)
    tq = q.natural()
    tp = p.natural()
    if abs(alpha - 1.0) < 1e-6:
        grad = q.expected_statistic()                       # E_q[T] = gradA(theta_q)
        bregman = cls.log_partition_at(tp) - cls.log_partition_at(tq)
        for g, a, b in zip(grad, tq, tp):
            bregman = bregman - (g * (b - a)).sum(dim=-1) if g.dim() else bregman - g * (b - a)
        div = bregman
    else:
        blend = tuple(alpha * a + (1.0 - alpha) * b for a, b in zip(tq, tp))
        div = (cls.log_partition_at(blend)
               - alpha * cls.log_partition_at(tq)
               - (1.0 - alpha) * cls.log_partition_at(tp)) / (alpha - 1.0)
    return safe_kl_clamp(div, kl_max=kl_max)


def renyi(
    q:       BeliefParams,
    p:       BeliefParams,

    *,
    alpha:   float = 1.0,
    kl_max:  float = 100.0,
    eps:     float = 1e-6,
) -> torch.Tensor:
    r"""Renyi alpha-divergence D_alpha(q || p) between two parameter objects (KL at alpha=1).

    Uses ``q.renyi_closed_form`` when the family provides one (the pinned Gaussian moment
    form); otherwise the generic Bregman/Renyi-from-A path.
    """
    if alpha <= 0.0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    if alpha > 1.0:
        _warn_alpha_gt_one(alpha, type(q).__name__)
    closed = getattr(q, "renyi_closed_form", None)
    if closed is not None:
        return closed(p, alpha=alpha, kl_max=kl_max, eps=eps)
    return _renyi_from_log_partition(q, p, alpha=alpha, kl_max=kl_max, eps=eps)


def kl(
    q:       BeliefParams,
    p:       BeliefParams,

    *,
    kl_max:  float = 100.0,
    eps:     float = 1e-6,
) -> torch.Tensor:
    r"""KL(q || p) = Renyi at alpha = 1."""
    return renyi(q, p, alpha=1.0, kl_max=kl_max, eps=eps)


register_functional("renyi")(renyi)
```

Note on the `expected_statistic` zip loop: the natural components of `_ExpFamily` are shape `(...,)` scalars per element (one parameter), so `g.dim()` may be 0 only for true scalars; for tensor parameters use the `.sum(dim=-1)` branch. The expression handles per-coordinate `(..., K)` naturals (Gaussian) by summing the contracted statistic over the coordinate axis. Keep the simple form: `bregman = bregman - sum((g * (b - a)).sum(dim=-1) for g, a, b in zip(grad, tq, tp))` and require each natural component to carry the coordinate axis as its last dim. Replace the loop in the code above with:

```python
        inner = sum(((g * (b - a)).sum(dim=-1) for g, a, b in zip(grad, tq, tp)))
        div = cls.log_partition_at(tp) - cls.log_partition_at(tq) - inner
```

For `_ExpFamily`, make `natural`/`expected_statistic` return shape `(..., 1)` (add a trailing dim) so `.sum(dim=-1)` is well-defined; adjust the toy class fields accordingly (`self.lam` shape `(...,)`, `natural` returns `(-self.lam.unsqueeze(-1)? )` — keep `lam` already carrying a coordinate axis of size 1). Verify the toy test passes with this convention before moving on.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_families.py -k generic -p no:cacheprovider`
Expected: PASS (2 passed). If shapes mismatch, fix the toy class's trailing-dim convention until green.

- [ ] **Step 5: Commit**

```bash
git add vfe3/families/base.py tests/test_families.py
git commit -m "feat(families): renyi/kl functionals + generic Renyi/KL-from-A (Phase 1)"
```

---

### Task 3: `DiagonalGaussian` (closed forms ported verbatim + natural/A + block/broadcast)

**Files:**
- Create: `vfe3/families/gaussian.py`
- Test: `tests/test_families.py`

Reference: port the diagonal closed form from `vfe3/divergence.py:96-134` (`_gaussian_diagonal_renyi`) and the per-coord form from `vfe3/divergence.py:201-242` (`gaussian_diagonal_renyi_per_coord`) verbatim into methods.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_families.py  (append)
def test_diagonal_gaussian_closed_form_matches_legacy_divergence():
    from vfe3.divergence import renyi as legacy_renyi          # still the tensor API at this point
    from vfe3.families.gaussian import DiagonalGaussian
    from vfe3.families.base import renyi as fam_renyi
    torch.manual_seed(3)
    mu_q, mu_p = torch.randn(5, 4), torch.randn(5, 4)
    s_q, s_p = torch.rand(5, 4) + 0.5, torch.rand(5, 4) + 0.5
    for a in (0.5, 1.0):
        want = legacy_renyi(mu_q, s_q, mu_p, s_p, alpha=a, family="gaussian_diagonal")
        got = fam_renyi(DiagonalGaussian(mu_q, s_q), DiagonalGaussian(mu_p, s_p), alpha=a)
        assert torch.allclose(got, want, atol=1e-6), (a, (got - want).abs().max())


def test_diagonal_gaussian_generic_from_A_equals_closed_form():
    from vfe3.families.gaussian import DiagonalGaussian
    from vfe3.families.base import _renyi_from_log_partition
    torch.manual_seed(4)
    mu_q, mu_p = torch.randn(6, 3), torch.randn(6, 3)
    s_q, s_p = torch.rand(6, 3) + 0.5, torch.rand(6, 3) + 0.5
    q, p = DiagonalGaussian(mu_q, s_q), DiagonalGaussian(mu_p, s_p)
    for a in (0.5, 1.0):
        closed = q.renyi_closed_form(p, alpha=a, kl_max=float("inf"), eps=1e-6)
        generic = _renyi_from_log_partition(q, p, alpha=a, kl_max=float("inf"), eps=1e-6)
        assert torch.allclose(closed, generic, atol=1e-4), (a, (closed - generic).abs().max())


def test_diagonal_block_and_broadcast():
    from vfe3.families.gaussian import DiagonalGaussian
    mu, s = torch.randn(2, 6), torch.rand(2, 6) + 0.5
    q = DiagonalGaussian(mu, s)
    qb = q.block(2, 4)
    assert torch.equal(qb.mu, mu[..., 2:4]) and torch.equal(qb.sigma, s[..., 2:4])
    qk = q.broadcast_over_keys()
    assert qk.mu.shape == (2, 1, 6) and qk.sigma.shape == (2, 1, 6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_families.py -k diagonal -p no:cacheprovider`
Expected: FAIL (ModuleNotFoundError: `vfe3.families.gaussian`).

- [ ] **Step 3: Write minimal implementation**

```python
# vfe3/families/gaussian.py
r"""Gaussian exponential families (diagonal and full covariance) for VFE_3.0.

The closed-form Renyi/KL kernels are ported verbatim from the legacy ``divergence.py``
moment forms, so the live numerics are byte-identical; natural-parameter and log-partition
maps are added so the generic Bregman/Renyi-from-A path can be pinned against them.
"""

from typing import Tuple

import torch

from vfe3.families.base import (
    BeliefParams, register_family, safe_kl_clamp, _logdet_chol,
)


@register_family("gaussian_diagonal")
class DiagonalGaussian(BeliefParams):
    r"""Diagonal Gaussian: mu (..., K), sigma (..., K) variances.

    Natural theta = (mu/sigma, -1/(2 sigma)); A(theta) = sum_k [ -t1^2/(4 t2) - 1/2 log(-2 t2) ];
    E[T] = (mu, mu^2 + sigma).
    """

    cov_kind = "diagonal"

    def __init__(self, mu: torch.Tensor, sigma: torch.Tensor) -> None:
        self.mu = mu
        self.sigma = sigma

    def coordinate_dim(self) -> int:
        return self.mu.shape[-1]

    def block(self, start: int, end: int) -> "DiagonalGaussian":
        return DiagonalGaussian(self.mu[..., start:end], self.sigma[..., start:end])

    def broadcast_over_keys(self) -> "DiagonalGaussian":
        return DiagonalGaussian(self.mu.unsqueeze(-2), self.sigma.unsqueeze(-2))

    def natural(self) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.sigma.clamp(min=1e-12)
        return (self.mu / s, -1.0 / (2.0 * s))

    @classmethod
    def log_partition_at(cls, theta: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        t1, t2 = theta
        return (-(t1 ** 2) / (4.0 * t2) - 0.5 * torch.log(-2.0 * t2)).sum(dim=-1)

    def expected_statistic(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return (self.mu, self.mu ** 2 + self.sigma)

    def entropy(self) -> torch.Tensor:
        import math
        return 0.5 * (torch.log(self.sigma.clamp(min=1e-12)) + math.log(2.0 * math.pi * math.e)).sum(dim=-1)

    def renyi_closed_form(
        self, other: "DiagonalGaussian", *, alpha: float = 1.0,
        kl_max: float = 100.0, eps: float = 1e-6,
    ) -> torch.Tensor:
        # ported verbatim from divergence.py _gaussian_diagonal_renyi (mu_q=self, mu_t=other)
        K = self.mu.shape[-1]
        mu_q = self.mu.float()
        sigma_q = self.sigma.float().clamp(min=eps)
        mu_t = other.mu.float()
        sigma_t = other.sigma.float().clamp(min=eps)
        if abs(alpha - 1.0) < 1e-6:
            trace_term = (sigma_q / sigma_t).sum(dim=-1)
            delta = mu_t - mu_q
            mahal_term = ((delta ** 2) / sigma_t).sum(dim=-1)
            logdet_term = (torch.log(sigma_t) - torch.log(sigma_q)).sum(dim=-1)
            div = 0.5 * (trace_term + mahal_term - K + logdet_term)
        else:
            sigma_blend = ((1.0 - alpha) * sigma_q + alpha * sigma_t).clamp(min=eps)
            delta = mu_t - mu_q
            mahal_term = (alpha * (delta ** 2) / sigma_blend).sum(dim=-1)
            logdet_per_dim = ((1.0 - alpha) * torch.log(sigma_q)
                              + alpha * torch.log(sigma_t) - torch.log(sigma_blend))
            logdet_term = logdet_per_dim.sum(dim=-1) / (alpha - 1.0)
            div = 0.5 * (mahal_term + logdet_term)
        return safe_kl_clamp(div, kl_max=kl_max)

    def renyi_per_coord(
        self, other: "DiagonalGaussian", *, alpha: float = 1.0,
        kl_max: float = 100.0, eps: float = 1e-6,
    ) -> torch.Tensor:
        # ported verbatim from divergence.py gaussian_diagonal_renyi_per_coord
        mu_q = self.mu.float()
        sigma_q = self.sigma.float().clamp(min=eps)
        mu_t = other.mu.float()
        sigma_t = other.sigma.float().clamp(min=eps)
        delta = mu_t - mu_q
        if abs(alpha - 1.0) < 1e-6:
            per_coord = 0.5 * (sigma_q / sigma_t + (delta ** 2) / sigma_t - 1.0
                               + torch.log(sigma_t) - torch.log(sigma_q))
        else:
            sigma_blend = ((1.0 - alpha) * sigma_q + alpha * sigma_t).clamp(min=eps)
            mahal = alpha * (delta ** 2) / sigma_blend
            logdet = ((1.0 - alpha) * torch.log(sigma_q) + alpha * torch.log(sigma_t)
                      - torch.log(sigma_blend)) / (alpha - 1.0)
            per_coord = 0.5 * (mahal + logdet)
        return safe_kl_clamp(per_coord, kl_max=kl_max)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_families.py -k diagonal -p no:cacheprovider`
Expected: PASS (3 passed). The generic-equals-closed test confirms the natural/A maps.

- [ ] **Step 5: Commit**

```bash
git add vfe3/families/gaussian.py tests/test_families.py
git commit -m "feat(families): DiagonalGaussian closed forms + natural/A maps (Phase 1)"
```

---

### Task 4: `FullGaussian` (Cholesky closed form ported + marginal block + natural/A)

**Files:**
- Modify: `vfe3/families/gaussian.py`
- Test: `tests/test_families.py`

Reference: port the full closed form from `vfe3/divergence.py:137-198` (`_gaussian_full_renyi`) verbatim.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_families.py  (append)
def test_full_gaussian_closed_form_matches_legacy_and_block():
    from vfe3.divergence import renyi as legacy_renyi
    from vfe3.families.gaussian import FullGaussian
    from vfe3.families.base import renyi as fam_renyi
    torch.manual_seed(5)
    N, K = 4, 3
    mu_q, mu_p = torch.randn(N, K), torch.randn(N, K)
    Aq = torch.randn(N, K, K); s_q = Aq @ Aq.transpose(-1, -2) + K * torch.eye(K)
    Ap = torch.randn(N, K, K); s_p = Ap @ Ap.transpose(-1, -2) + K * torch.eye(K)
    for a in (0.5, 1.0):
        want = legacy_renyi(mu_q, s_q, mu_p, s_p, alpha=a, family="gaussian_full")
        got = fam_renyi(FullGaussian(mu_q, s_q), FullGaussian(mu_p, s_p), alpha=a)
        assert torch.allclose(got, want, atol=1e-4), (a, (got - want).abs().max())
    qb = FullGaussian(mu_q, s_q).block(1, 3)
    assert torch.equal(qb.mu, mu_q[..., 1:3]) and torch.equal(qb.sigma, s_q[..., 1:3, 1:3])


def test_full_gaussian_per_coord_raises():
    import pytest
    from vfe3.families.gaussian import FullGaussian
    q = FullGaussian(torch.zeros(2, 2), torch.eye(2).expand(2, 2, 2))
    with pytest.raises((AttributeError, NotImplementedError)):
        q.renyi_per_coord(q, alpha=1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_families.py -k full_gaussian -p no:cacheprovider`
Expected: FAIL (ImportError: cannot import `FullGaussian`).

- [ ] **Step 3: Write minimal implementation** (append to `vfe3/families/gaussian.py`)

```python
@register_family("gaussian_full")
class FullGaussian(BeliefParams):
    r"""Full-covariance Gaussian: mu (..., K), sigma (..., K, K) SPD covariance.

    Natural theta = (Sigma^{-1} mu, -1/2 Sigma^{-1}); A(theta) = -1/4 t1^T t2^{-1} t1 - 1/2 log|-2 t2|.
    """

    cov_kind = "full"

    def __init__(self, mu: torch.Tensor, sigma: torch.Tensor) -> None:
        self.mu = mu
        self.sigma = sigma

    def coordinate_dim(self) -> int:
        return self.mu.shape[-1]

    def block(self, start: int, end: int) -> "FullGaussian":
        return FullGaussian(self.mu[..., start:end], self.sigma[..., start:end, start:end])

    def broadcast_over_keys(self) -> "FullGaussian":
        return FullGaussian(self.mu.unsqueeze(-2), self.sigma.unsqueeze(-3))

    def natural(self) -> Tuple[torch.Tensor, torch.Tensor]:
        eye = torch.eye(self.mu.shape[-1], device=self.mu.device, dtype=self.mu.dtype)
        prec = torch.linalg.solve(self.sigma + 1e-6 * eye, eye.expand_as(self.sigma))
        t1 = (prec @ self.mu.unsqueeze(-1)).squeeze(-1)
        return (t1, -0.5 * prec)

    @classmethod
    def log_partition_at(cls, theta: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        t1, t2 = theta
        neg2t2 = -2.0 * t2
        L = torch.linalg.cholesky(neg2t2)
        inv_neg2t2 = torch.cholesky_inverse(L)
        quad = (t1.unsqueeze(-2) @ inv_neg2t2 @ t1.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        return 0.5 * quad - 0.5 * _logdet_chol(L)

    def expected_statistic(self) -> Tuple[torch.Tensor, torch.Tensor]:
        outer = self.mu.unsqueeze(-1) * self.mu.unsqueeze(-2)
        return (self.mu, self.sigma + outer)

    def entropy(self) -> torch.Tensor:
        import math
        K = self.mu.shape[-1]
        L = torch.linalg.cholesky(self.sigma)
        return 0.5 * _logdet_chol(L) + 0.5 * K * math.log(2.0 * math.pi * math.e)

    def renyi_closed_form(
        self, other: "FullGaussian", *, alpha: float = 1.0,
        kl_max: float = 100.0, eps: float = 1e-6,
    ) -> torch.Tensor:
        # ported verbatim from divergence.py _gaussian_full_renyi (mu_q=self, mu_t=other)
        K = self.mu.shape[-1]
        device = self.mu.device
        mu_q = self.mu.float(); sigma_q = self.sigma.float()
        mu_t = other.mu.float(); sigma_t = other.sigma.float()
        eye = torch.eye(K, device=device, dtype=torch.float32)
        sigma_q_reg = sigma_q + eps * eye
        sigma_t_reg = sigma_t + eps * eye
        if abs(alpha - 1.0) < 1e-6:
            L_p = torch.linalg.cholesky(sigma_t_reg)
            Y = torch.linalg.solve_triangular(L_p, sigma_q_reg, upper=False)
            Z = torch.linalg.solve_triangular(L_p.transpose(-1, -2), Y, upper=True)
            trace_term = torch.diagonal(Z, dim1=-2, dim2=-1).sum(dim=-1)
            delta_mu = mu_t - mu_q
            v = torch.linalg.solve_triangular(L_p, delta_mu.unsqueeze(-1), upper=False).squeeze(-1)
            mahal_term = (v ** 2).sum(dim=-1)
            logdet_p = _logdet_chol(L_p)
            logdet_q = _logdet_chol(torch.linalg.cholesky(sigma_q_reg))
            div = 0.5 * (trace_term + mahal_term - K + logdet_p - logdet_q)
        else:
            sigma_blend = (1.0 - alpha) * sigma_q_reg + alpha * sigma_t_reg
            sigma_blend = 0.5 * (sigma_blend + sigma_blend.transpose(-1, -2))
            L_blend = torch.linalg.cholesky(sigma_blend)
            delta_mu = mu_t - mu_q
            v = torch.linalg.solve_triangular(L_blend, delta_mu.unsqueeze(-1), upper=False).squeeze(-1)
            mahal_term = alpha * (v ** 2).sum(dim=-1)
            logdet_q = _logdet_chol(torch.linalg.cholesky(sigma_q_reg))
            logdet_t = _logdet_chol(torch.linalg.cholesky(sigma_t_reg))
            logdet_blend = _logdet_chol(L_blend)
            logdet_term = ((1.0 - alpha) * logdet_q + alpha * logdet_t - logdet_blend) / (alpha - 1.0)
            div = 0.5 * (mahal_term + logdet_term)
        return safe_kl_clamp(div, kl_max=kl_max)
```

(FullGaussian intentionally does NOT define `renyi_per_coord`, so the per-coord guard raises `AttributeError` for full covariance — matching the legacy guard.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_families.py -k full_gaussian -p no:cacheprovider`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add vfe3/families/gaussian.py tests/test_families.py
git commit -m "feat(families): FullGaussian closed form + natural/A maps (Phase 1)"
```

---

## Phase 2 — route the legacy divergence.py through families (byte-identity gate)

### Task 5: delegate `divergence.py` to families; full-suite equivalence gate

**Files:**
- Modify: `vfe3/divergence.py`
- Modify: `vfe3/free_energy.py` (import site only)
- Test: full suite

**Why:** prove the families closed forms produce byte-identical results through the EXISTING tensor callers before flipping any signature. After this task the live path computes through `families`, but the public tensor API is unchanged, so the 259-test suite is the equivalence gate.

- [ ] **Step 1: Write the failing test** (a guard that the families path is the one in use)

```python
# tests/test_divergence.py  (append)
def test_divergence_delegates_to_families():
    """renyi(...) must route through the families layer (DiagonalGaussian closed form)."""
    import torch
    from vfe3.divergence import renyi
    from vfe3.families.gaussian import DiagonalGaussian
    from vfe3.families.base import renyi as fam_renyi
    mu_q, mu_p = torch.randn(3, 2), torch.randn(3, 2)
    s_q, s_p = torch.rand(3, 2) + 0.5, torch.rand(3, 2) + 0.5
    got = renyi(mu_q, s_q, mu_p, s_p, alpha=0.5, family="gaussian_diagonal")
    want = fam_renyi(DiagonalGaussian(mu_q, s_q), DiagonalGaussian(mu_p, s_p), alpha=0.5)
    assert torch.allclose(got, want, atol=0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_divergence.py -k delegates -p no:cacheprovider`
Expected: FAIL (the legacy renyi computes via its own inline kernel; `atol=0.0` may differ by float reassociation, or the import differs). If it already matches bit-for-bit, change the assertion to identity via patching; otherwise proceed to Step 3 which makes delegation literal.

- [ ] **Step 3: Rewrite `vfe3/divergence.py` to delegate.** Replace the registry + `_gaussian_*_renyi` kernels + `gaussian_diagonal_renyi_per_coord` + `renyi`/`kl` bodies with delegation to families, keeping the tensor signature. Final `divergence.py`:

```python
r"""The divergence seam for VFE_3.0 (tensor-API facade over the families layer).

Renyi alpha-divergence is the primitive; KL is its alpha = 1 special case. The closed
forms and the exponential-family abstraction live in ``vfe3.families``; this module keeps
the historical tensor-tuple entry points (and re-exports the families registry helpers) so
existing callers are unaffected during the parameter-object migration.
"""

from typing import Tuple

import torch

from vfe3.families.base import (
    safe_kl_clamp, family_cov_kind, divergence_families,
    register_functional, get_functional,
    renyi as _renyi_params, kl as _kl_params,
)
from vfe3.families.base import _warn_alpha_gt_one  # noqa: F401  (kept for back-compat imports)
from vfe3.families import gaussian as _gaussian   # noqa: F401  (registers the Gaussian families)
from vfe3.families.base import get_family


def renyi(mu_q, sigma_q, mu_t, sigma_t, *, alpha=1.0, kl_max=100.0, eps=1e-6,
          family="gaussian_diagonal"):
    cls = get_family(family)
    return _renyi_params(cls(mu_q, sigma_q), cls(mu_t, sigma_t), alpha=alpha, kl_max=kl_max, eps=eps)


def kl(mu_q, sigma_q, mu_t, sigma_t, *, kl_max=100.0, eps=1e-6, family="gaussian_diagonal"):
    return renyi(mu_q, sigma_q, mu_t, sigma_t, alpha=1.0, kl_max=kl_max, eps=eps, family=family)


def gaussian_diagonal_renyi_per_coord(mu_q, sigma_q, mu_t, sigma_t, *, alpha=1.0,
                                      kl_max=100.0, eps=1e-6):
    from vfe3.families.gaussian import DiagonalGaussian
    return DiagonalGaussian(mu_q, sigma_q).renyi_per_coord(
        DiagonalGaussian(mu_t, sigma_t), alpha=alpha, kl_max=kl_max, eps=eps)
```

(Remove the old `_DIVERGENCES`/`_COV_KIND`/`register_divergence`/`get_divergence` registry and the inline kernels — they now live in families. Update `tests/test_divergence.py::test_registry_register_and_get` and `test_registry_unknown_raises` to use `register_family`/`get_family` from `vfe3.families.base`, or delete them if fully superseded by `tests/test_families.py`.) In `vfe3/free_energy.py`, change the import `from vfe3.divergence import family_cov_kind, gaussian_diagonal_renyi_per_coord, get_functional` to import `family_cov_kind` and `get_functional` from `vfe3.divergence` (unchanged) — both are still exported.

- [ ] **Step 4: Run the FULL suite**

Run: `python -m pytest --junitxml=$env:TEMP\vfe3.xml -p no:cacheprovider` (PowerShell) — read `tests=/failures=/errors=` from the XML.
Expected: `failures=0 errors=0`, total = 259 + the new families tests. Any numeric drift here is a real regression — fix before commit (the closed forms are byte-for-byte the same code, so drift means a wiring error).

- [ ] **Step 5: Commit**

```bash
git add vfe3/divergence.py vfe3/free_energy.py tests/test_divergence.py
git commit -m "refactor(divergence): delegate tensor API to the families layer (Phase 2)"
```

---

## Phase 3 — flip to the parameter-object signature + convert consumers

Each task converts one consumer to build parameter objects at the divergence boundary and keeps the full suite green. The order is leaf-to-root so each step is independently testable.

### Task 6: `free_energy.py` energy/self-divergence take parameter objects

**Files:**
- Modify: `vfe3/free_energy.py`
- Test: `tests/test_free_energy.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_free_energy.py  (append)
def test_pairwise_energy_accepts_belief_params():
    import torch
    from vfe3.families.gaussian import DiagonalGaussian
    from vfe3.free_energy import pairwise_energy
    torch.manual_seed(8)
    N, K = 3, 4
    q = DiagonalGaussian(torch.randn(N, K), torch.rand(N, K) + 0.5)
    key = DiagonalGaussian(torch.randn(N, N, K), torch.rand(N, N, K) + 0.5)
    E = pairwise_energy(q, key, alpha=1.0)
    assert E.shape == (N, N)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_free_energy.py -k accepts_belief_params -p no:cacheprovider`
Expected: FAIL (pairwise_energy still expects tensors / wrong arg count).

- [ ] **Step 3: Convert `pairwise_energy`, `self_divergence`, `self_divergence_per_coord`, `self_divergence_for_alpha`** to take `BeliefParams`. New `pairwise_energy`:

```python
def pairwise_energy(q, key, *, alpha=1.0, kl_max=100.0, eps=1e-6,
                    divergence_family="renyi", irrep_dims=None):
    r"""Per-pair belief-coupling energy E_ij = D(q_i || Omega_ij q_j) via the families seam.
    q: BeliefParams (..., N, K); key: transported-key BeliefParams (..., N, N, K)."""
    functional = get_functional(divergence_family)
    q_b = q.broadcast_over_keys()                       # (..., N, 1, K)
    if irrep_dims is None or len(irrep_dims) == 1:
        return functional(q_b, key, alpha=alpha, kl_max=kl_max, eps=eps)
    energies, start = [], 0
    for d in irrep_dims:
        end = start + d
        energies.append(functional(q_b.block(start, end), key.block(start, end),
                                    alpha=alpha, kl_max=kl_max, eps=eps))
        start = end
    return torch.stack(energies, dim=-3)               # (..., H, N, N)
```

`self_divergence(q, p, *, ...)` -> `get_functional(divergence_family)(q, p, alpha=..., ...)`.
`self_divergence_per_coord(q, p, *, ...)` -> dispatch via `family_cov_kind`: require `q.cov_kind == "diagonal"` (else raise the existing ValueError), then `q.renyi_per_coord(p, alpha=..., ...)`; keep the `divergence_family == "renyi"` guard.
`self_divergence_for_alpha(q, p, *, alpha_mode, ...)` -> `self_divergence_per_coord` when `alpha_is_per_coord(alpha_mode)` else `self_divergence`.

Remove the now-unused `family`/`get_functional`-on-tensors plumbing and the `is_diagonal = family_cov_kind(family) == "diagonal"` line (the params object now owns its layout via `broadcast_over_keys`/`block`).

- [ ] **Step 4: Run the FULL suite**

Run: `python -m pytest --junitxml=$env:TEMP\vfe3.xml -p no:cacheprovider` — read counts from the XML.
Expected: many `test_free_energy.py` callers fail until updated. Update the existing `test_free_energy.py` energy/self-divergence tests to pass `DiagonalGaussian`/`FullGaussian` objects (the inputs they already build as tensors get wrapped). Re-run until `failures=0`.

- [ ] **Step 5: Commit**

```bash
git add vfe3/free_energy.py tests/test_free_energy.py
git commit -m "refactor(free_energy): energy/self-divergence take BeliefParams (Phase 3)"
```

---

### Task 7: `inference/e_step.py` builds parameter objects

**Files:**
- Modify: `vfe3/inference/e_step.py`
- Test: full suite (`tests/test_e_step.py`, `tests/test_free_energy.py`)

- [ ] **Step 1:** Identify every `pairwise_energy(...)` / `self_divergence*(...)` call in `e_step.py` (the energy and self-coupling sites, including `_transport_qk` mixed-frame and the filtered objective). At each, wrap the moment tensors via `get_family(cfg.family)(mu, sigma)` (query) and `get_family(cfg.family)(mu_t, sigma_t)` (transported key). Add `from vfe3.families.base import get_family` (or thread the family class once at the top of the E-step).

- [ ] **Step 2: Run the FULL suite**

Run: `python -m pytest --junitxml=$env:TEMP\vfe3.xml -p no:cacheprovider`
Expected: `tests/test_e_step.py` failures until the calls are wrapped; then `failures=0`. The E-step numerics are byte-identical (same closed forms).

- [ ] **Step 3: Commit**

```bash
git add vfe3/inference/e_step.py
git commit -m "refactor(e_step): build BeliefParams at the divergence boundary (Phase 3)"
```

---

### Task 8: `gradients/oracle.py` builds parameter objects

**Files:**
- Modify: `vfe3/gradients/oracle.py`
- Test: `tests/test_gradients_oracle.py`, `tests/test_gradients_kernels.py`

- [ ] **Step 1:** In `free_energy_value`/the oracle energy assembly, wrap the query and transported-key moments into `get_family(family)(...)` before the `pairwise_energy`/`self_divergence_for_alpha` calls. The autograd path is unchanged (the params hold the same grad-connected tensors). `gradients/kernels.py` is NOT modified (the hand kernel stays tensor-based; its `family == "gaussian_diagonal"` availability guard stays).

- [ ] **Step 2: Run the FULL suite** (the finite-difference oracle checks are the gate)

Run: `python -m pytest tests/test_gradients_oracle.py tests/test_gradients_kernels.py -p no:cacheprovider`
Expected: PASS (gradients byte-identical; the autograd graph is unchanged). Then run the full suite.

- [ ] **Step 3: Commit**

```bash
git add vfe3/gradients/oracle.py
git commit -m "refactor(oracle): build BeliefParams at the energy boundary (Phase 3)"
```

---

### Task 9: `model/prior_bank.py` decode + `model/model.py` diagnostics build parameter objects

**Files:**
- Modify: `vfe3/model/prior_bank.py`, `vfe3/model/model.py`
- Test: `tests/test_prior_bank.py`, `tests/test_model.py`, `tests/test_use_prior_bank.py`

- [ ] **Step 1:** In `prior_bank.reference_decode` (and any KL call in the decode path that uses `divergence.kl`), build parameter objects for the query and the per-vocab prior and call `vfe3.families.base.kl`. The fused `_decode_diagonal` analytic matmul is unchanged (it does not call `kl`). In `model.diagnostics`, wrap the moments before `pairwise_energy`/`self_divergence_for_alpha`.

- [ ] **Step 2: Run the FULL suite**

Run: `python -m pytest --junitxml=$env:TEMP\vfe3.xml -p no:cacheprovider`
Expected: `failures=0` (decode atol-1e-3 golden preserved).

- [ ] **Step 3: Commit**

```bash
git add vfe3/model/prior_bank.py vfe3/model/model.py
git commit -m "refactor(model): decode/diagnostics build BeliefParams (Phase 3)"
```

---

### Task 10: flip the public `divergence.py` API to parameter objects; remove the tensor shim

**Files:**
- Modify: `vfe3/divergence.py`
- Test: full suite

- [ ] **Step 1:** Now that every in-repo caller passes parameter objects, make `divergence.py` re-export the param-typed functionals directly:

```python
from vfe3.families.base import (
    renyi, kl, safe_kl_clamp, family_cov_kind, divergence_families,
    register_functional, get_functional, register_family, get_family,
)
from vfe3.families import gaussian as _gaussian  # noqa: F401  (registers Gaussian families)
```

Delete the tensor-tuple `renyi`/`kl`/`gaussian_diagonal_renyi_per_coord` wrappers. Grep the repo for any remaining tensor-style call (`renyi(mu`, `kl(mu`, `gaussian_diagonal_renyi_per_coord(`) and convert; there should be none outside tests already updated.

- [ ] **Step 2: Run the FULL suite**

Run: `python -m pytest --junitxml=$env:TEMP\vfe3.xml -p no:cacheprovider`
Expected: `failures=0 errors=0`. Read the total from the XML.

- [ ] **Step 3: Commit**

```bash
git add vfe3/divergence.py
git commit -m "refactor(divergence): public renyi/kl take BeliefParams; drop tensor shim (Phase 3)"
```

---

## Final verification

- [ ] Run the full suite once more via `--junitxml`; record `tests=/failures=/errors=` (expect `failures=0 errors=0`, total = 259 + the families tests).
- [ ] Grep that no production module still imports a removed symbol (`_DIVERGENCES`, `register_divergence`, the tensor `renyi(mu...)` form).
- [ ] Confirm `config.py` validation still works: `python -c "from vfe3.config import VFE3Config; VFE3Config(); VFE3Config(family='gaussian_full', diagonal_covariance=False)"`.
- [ ] Append a post-edit section to `docs/edits/2026-06-01-per-coordinate-alpha.md` describing the families/ seam + M2 (files, the byte-identity gate, the toy-family validation, test count).
- [ ] (Optional, ultracode) sympy check: symbolically confirm the diagonal-Gaussian generic Renyi-from-A equals the moment closed form for K=1, alpha symbolic, as derivation insurance beyond the numeric pin.

## Self-review notes (author)

- Spec coverage: base.py interface (Task 1-2), Gaussian families with closed forms + natural/A (Task 3-4), generic Renyi/KL-from-A (Task 2), toy-family validation (Task 2), consumer conversion list (Task 6-9), public signature flip (Task 10), byte-identity gate (Task 5 + every Phase-3 full-suite run). All spec sections map to a task.
- The `expected_statistic` Bregman-KL convention is the one ambiguity; Task 2 Step 3 pins the trailing-coordinate-axis convention and the toy test gates it before any Gaussian work depends on it.
- Out-of-scope items (mixtures, categorical likelihood, Phase-5 gauge, BeliefState redesign, register_transport, alpha>1 hardening) are not touched by any task.
