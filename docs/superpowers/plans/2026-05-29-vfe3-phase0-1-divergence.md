# VFE_3.0 Phase 0 + Phase 1 (Repo Scaffold + Divergence Seam) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the V3_Transformer repo and build `divergence.py` — the single config-selected divergence seam (Rényi primitive, KL as α=1) — proven numerically equal to VFE_2.0's Gaussian KL/Rényi kernels by golden tests.

**Architecture:** Bottom-up clean-room rebuild. This plan delivers the lowest layer: a modular divergence module backed by a registry (`gaussian_diagonal`, `gaussian_full`), with KL recovered as the α=1 case of Rényi. Every kernel is pinned to a 2.0 reference by a golden equivalence test that imports a sibling VFE_2.0 checkout. No gauge, no transport, no gradients yet — those are later phases.

**Tech Stack:** Python 3, PyTorch (float32; CUDA on the user's RTX5090, CPU-runnable for tests), pytest. No CLI arg parsing (project policy: click-to-run entry points only). No neural-network components.

**Scope note:** This is the first of several per-phase plans (the full spec spans Phases 0–8). It is self-contained and testable on its own. Phase 2 (geometry), Phase 3 (free_energy + alpha_i), Phase 4 (gradients) get their own plans.

**Reference spec:** `docs/superpowers/specs/2026-05-29-vfe3-clean-room-design.md`

**2.0 reference kernels being matched** (in `C:\Users\chris and christine\Desktop\VFE_2.0`):
- `transformer/core/kl_computation.py::_kl_kernel_diagonal` (lines 365–459) — diagonal KL/Rényi.
- `transformer/core/kl_computation.py::_kl_kernel_dense` (lines 122–362) — full-cov KL/Rényi.
- `transformer/core/kl_computation.py::safe_kl_clamp` (lines 70–115) — clamp to `[0, kl_max]`, NaN/+inf→kl_max.

---

## File Structure

This plan creates these files in `C:\Users\chris and christine\Desktop\V3_Transformer`:

- Create: `pyproject.toml` — package metadata + pytest config.
- Create: `.gitignore` — Python ignores.
- Create: `README.md` — one-paragraph repo description.
- Create: `vfe3/__init__.py` — package marker.
- Create: `vfe3/config.py` — `VFE3Config` dataclass (only the fields this phase needs: `eps`, `kl_max`, `divergence_family`, `alpha_div`).
- Create: `vfe3/divergence.py` — the divergence seam: `safe_kl_clamp`, registry (`register_divergence`/`get_divergence`), `_gaussian_diagonal_renyi`, `_gaussian_full_renyi`, public `renyi`/`kl`.
- Create: `tests/__init__.py` — empty.
- Create: `tests/conftest.py` — device fixture.
- Create: `tests/golden/__init__.py` — empty.
- Create: `tests/golden/conftest.py` — locate + import the sibling VFE_2.0 checkout (`VFE2_ROOT` env var, default `../VFE_2.0`); skip golden tests if unavailable.
- Create: `tests/golden/test_divergence_golden.py` — golden equivalence vs 2.0 kernels.
- Create: `tests/test_divergence.py` — unit + property tests (registry, KL=Rényi at α=1, KL(p‖p)=0, non-negativity).
- Create: `tests/test_config.py` — config instantiation/validation.

---

## Code Style (MANDATORY for every function in this build)

Every function written in this build must follow the project's signature convention so the codebase stays uniformly tidy and readable. This applies to the code blocks below and to any function a subagent writes.

Argument ordering (top to bottom): all `torch.Tensor` first, then `'float | torch.Tensor'`, then undefined floats (no default), then undefined ints, then undefined bools, then defined floats (with default), then defined ints, then defined bools, then `Optional[...]`, then `**kwargs` last.

Vertical alignment: parameter names, type annotations, `=` signs, and trailing `#` comments are each aligned to a common column across the whole signature. Blank lines separate the type groups. Tensor shape comments at critical points.

Reference example (the canonical style — match it):

```python
def _compute_rope_full_gauge_gradient_per_head(
    mu_h:                   torch.Tensor,            # (B, N, d_h) per-head means
    sigma_h:                torch.Tensor,            # (B, N, d_h) diagonal or (B, N, d_h, d_h) full covariance
    phi:                    torch.Tensor,            # (B, N, n_gen) gauge frames

    alpha:                  'float | torch.Tensor',
    kappa:                  'float | torch.Tensor',

    lambda_belief:          float,
    eps:                    float,

    d_h:                    int,

    enforce_orthogonal:     bool                   = False,

    cached_block_exp_pairs: Optional[list]         = None,
    mask:                   Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
```

Other conventions: type hints on every signature; docstrings with the LaTeX/math form for any non-trivial formula; variable names match paper notation (`mu_q`, `sigma_q`, `alpha`, `kappa`); keyword-only (`*`) for the non-tensor scalar knobs that the registry passes by name.

---

## Task 0.1: Initialize the V3_Transformer repo and scaffolding

**Files:**
- Create: `.gitignore`, `pyproject.toml`, `README.md`, `CLAUDE.md`, `vfe3/__init__.py`, `tests/__init__.py`, `tests/golden/__init__.py`

- [ ] **Step 1: Initialize git and wire the remote**

Run (PowerShell, from the repo root `C:\Users\chris and christine\Desktop\V3_Transformer`):

```powershell
git init
git branch -M main
git remote add origin https://github.com/cdenn016/V3_Transformer.git
```

Expected: `Initialized empty Git repository ...`; `git remote -v` shows `origin  https://github.com/cdenn016/V3_Transformer.git (fetch/push)`.

- [ ] **Step 2: Write `.gitignore`**

Create `.gitignore`:

```gitignore
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.ipynb_checkpoints/
.venv/
build/
dist/
*.tmp.*
```

- [ ] **Step 3: Write `pyproject.toml`**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "vfe3"
version = "0.0.1"
description = "Gauge-theoretic VFE transformer, clean-room rebuild (V3)"
requires-python = ">=3.10"
dependencies = ["torch"]

[tool.setuptools.packages.find]
include = ["vfe3*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

- [ ] **Step 4: Write `README.md`**

Create `README.md`:

```markdown
# V3_Transformer (VFE_3.0)

Clean-room rebuild of the gauge-theoretic variational free energy transformer.
No neural networks: all capacity comes from iterative VFE minimization over
Gaussian belief tuples `(mu, Sigma, phi)`. Built bottom-up with every layer
numerically pinned to VFE_2.0 by golden tests. See
`docs/superpowers/specs/2026-05-29-vfe3-clean-room-design.md`.
```

- [ ] **Step 5: Create package and test markers**

Create `vfe3/__init__.py` (empty file).
Create `tests/__init__.py` (empty file).
Create `tests/golden/__init__.py` (empty file).

- [ ] **Step 5b: Write `CLAUDE.md` (carry the code-style convention into the repo)**

Create `CLAUDE.md`:

```markdown
# V3_Transformer (VFE_3.0)

Clean-room rebuild of the gauge-theoretic VFE transformer. No neural networks:
all capacity comes from iterative VFE minimization over Gaussian belief tuples
`(mu, Sigma, phi)`. Built bottom-up, every layer numerically pinned to VFE_2.0
by golden tests. See `docs/superpowers/specs/2026-05-29-vfe3-clean-room-design.md`.

## Hard constraints
- NO neural networks (no nn.Linear, no MLP, no activations).
- NO CLI arg parsing; entry points are click-to-run (edit config dicts, then run).
- float32 throughout; CUDA where applicable (user has an RTX 5090).
- High modularity: a config-selected registry behind every seam (divergence,
  alpha_i, family, transport/gauge, retraction, decode). Add a variant by
  writing-and-registering it, never by editing call sites.
- Always preserve a theoretically pure path under appropriate toggles.

## Function signature convention (MANDATORY)
Argument order: all torch.Tensor first, then 'float | torch.Tensor', then
undefined floats, undefined ints, undefined bools, then defined floats,
defined ints, defined bools, then Optional, then **kwargs last.

Vertical alignment: names, type annotations, `=` signs, and trailing `#`
comments are each aligned to a common column. Blank lines separate type
groups. Tensor shape comments at critical points. Type hints on every
signature. Docstrings carry the LaTeX/math form for non-trivial formulas.
Variable names match paper notation (mu_q, sigma_q, alpha, kappa).

Example:

    def kernel(
        mu_q:    torch.Tensor,             # (..., K) query means
        sigma_q: torch.Tensor,             # (..., K) query variances

        *,
        alpha:   float = 1.0,
        kl_max:  float = 100.0,
        eps:     float = 1e-6,
    ) -> torch.Tensor:

## Testing
Golden equivalence vs a pinned VFE_2.0 checkout for every ported kernel;
finite-difference gradient checks against the autograd-of-F oracle (later
phases); property tests (non-negativity, self-divergence zero, gauge
equivariance). Tests are device-agnostic (default CPU; set
VFE3_TEST_DEVICE=cuda for the GPU).
```

- [ ] **Step 6: Commit the scaffold (spec + plan + skeleton) to main**

```powershell
git add .gitignore pyproject.toml README.md CLAUDE.md vfe3/__init__.py tests/__init__.py tests/golden/__init__.py docs/
git commit -m @'
chore: scaffold V3_Transformer clean-room repo

Initialize package, pytest config, gitignore, and carry the clean-room
design spec + Phase 0/1 plan into the new repo.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
'@
```

Expected: a commit on `main` listing the scaffold files plus `docs/superpowers/specs/...` and `docs/superpowers/plans/...`.

- [ ] **Step 7: Create the working branch**

```powershell
git checkout -b phase01-divergence
```

Expected: `Switched to a new branch 'phase01-divergence'`.

---

## Task 0.2: Config dataclass

**Files:**
- Create: `vfe3/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_config.py`:

```python
from vfe3.config import VFE3Config


def test_config_defaults():
    cfg = VFE3Config()
    assert cfg.eps == 1e-6
    assert cfg.kl_max == 100.0
    assert cfg.divergence_family == "gaussian_diagonal"
    assert cfg.alpha_div == 1.0


def test_config_rejects_unknown_family():
    import pytest
    with pytest.raises(ValueError):
        VFE3Config(divergence_family="not_a_family")


def test_config_rejects_nonpositive_alpha():
    import pytest
    with pytest.raises(ValueError):
        VFE3Config(alpha_div=0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vfe3.config'`.

- [ ] **Step 3: Write minimal implementation**

Create `vfe3/config.py`:

```python
"""Configuration for VFE_3.0. Single dataclass, single validation block.

No CLI parsing (project policy: click-to-run). Edit fields directly.
"""

from dataclasses import dataclass

_VALID_DIVERGENCE_FAMILIES = ("gaussian_diagonal", "gaussian_full")


@dataclass
class VFE3Config:
    """Phase 0/1 configuration surface (divergence layer only).

    Attributes:
        eps:               Regularization floor for variances / covariances.
        kl_max:            Upper clamp on divergence values.
        divergence_family: Registry key selecting the divergence kernel.
        alpha_div:         Renyi order; 1.0 recovers standard KL.
    """

    eps:               float = 1e-6
    kl_max:            float = 100.0
    divergence_family: str   = "gaussian_diagonal"
    alpha_div:         float = 1.0

    def __post_init__(self) -> None:
        if self.divergence_family not in _VALID_DIVERGENCE_FAMILIES:
            raise ValueError(
                f"divergence_family must be one of {_VALID_DIVERGENCE_FAMILIES}, "
                f"got {self.divergence_family!r}"
            )
        if self.alpha_div <= 0.0:
            raise ValueError(f"alpha_div must be positive, got {self.alpha_div}")
        if self.eps <= 0.0:
            raise ValueError(f"eps must be positive, got {self.eps}")
        if self.kl_max <= 0.0:
            raise ValueError(f"kl_max must be positive, got {self.kl_max}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```powershell
git add vfe3/config.py tests/test_config.py
git commit -m @'
feat(config): VFE3Config dataclass for the divergence layer

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
'@
```

---

## Task 0.3: Golden harness — import the sibling VFE_2.0 checkout

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/golden/conftest.py`
- Test (smoke): `tests/golden/test_divergence_golden.py` (first test only)

- [ ] **Step 1: Write the device fixture**

Create `tests/conftest.py`:

```python
import pytest
import torch


@pytest.fixture
def device():
    # Tests are device-agnostic; default CPU for portability.
    # Set VFE3_TEST_DEVICE=cuda to run on the GPU.
    import os
    name = os.environ.get("VFE3_TEST_DEVICE", "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA requested but not available")
    return torch.device(name)
```

- [ ] **Step 2: Write the golden conftest (locate + import 2.0)**

Create `tests/golden/conftest.py`:

```python
"""Make a sibling VFE_2.0 checkout importable for golden equivalence tests.

Resolution order for the 2.0 root:
  1. env var VFE2_ROOT
  2. sibling directory ../VFE_2.0 relative to this repo root

If the 2.0 checkout cannot be found or imported, golden tests are skipped
(not failed) so the suite still runs in environments without 2.0 present.
"""

import os
import sys
from pathlib import Path

import pytest


def _vfe2_root() -> Path | None:
    env = os.environ.get("VFE2_ROOT")
    if env:
        p = Path(env)
        return p if p.exists() else None
    # repo root = parents[2] of this file (tests/golden/conftest.py)
    sibling = Path(__file__).resolve().parents[2].parent / "VFE_2.0"
    return sibling if sibling.exists() else None


@pytest.fixture(scope="session")
def vfe2_kl():
    """Return the 2.0 kl_computation module, or skip if unavailable."""
    root = _vfe2_root()
    if root is None:
        pytest.skip("VFE_2.0 checkout not found (set VFE2_ROOT)")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from transformer.core import kl_computation
    except Exception as exc:  # import error -> skip, don't fail
        pytest.skip(f"could not import VFE_2.0 kl_computation: {exc}")
    return kl_computation
```

- [ ] **Step 3: Write the smoke golden test**

Create `tests/golden/test_divergence_golden.py`:

```python
import torch


def test_can_import_vfe2_kernels(vfe2_kl):
    # Smoke test: the 2.0 reference kernels are importable.
    assert hasattr(vfe2_kl, "_kl_kernel_diagonal")
    assert hasattr(vfe2_kl, "_kl_kernel_dense")
    assert hasattr(vfe2_kl, "safe_kl_clamp")
```

- [ ] **Step 4: Run the smoke test**

Run: `pytest tests/golden/test_divergence_golden.py -v`
Expected: 1 passed (or 1 skipped if `VFE_2.0` is absent — acceptable, but on this machine it is present at `..\VFE_2.0`, so expect PASS).

- [ ] **Step 5: Commit**

```powershell
git add tests/conftest.py tests/golden/conftest.py tests/golden/test_divergence_golden.py
git commit -m @'
test(golden): harness to import sibling VFE_2.0 reference kernels

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
'@
```

---

## Task 1.1: `safe_kl_clamp` in the divergence module

**Files:**
- Create: `vfe3/divergence.py`
- Test: `tests/test_divergence.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_divergence.py`:

```python
import torch

from vfe3.divergence import safe_kl_clamp


def test_safe_kl_clamp_bounds_and_nan():
    x = torch.tensor([-1.0, 0.5, 1e9, float("nan"), float("inf"), float("-inf")])
    out = safe_kl_clamp(x, kl_max=100.0)
    assert torch.equal(
        out, torch.tensor([0.0, 0.5, 100.0, 100.0, 100.0, 0.0])
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_divergence.py::test_safe_kl_clamp_bounds_and_nan -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vfe3.divergence'`.

- [ ] **Step 3: Write minimal implementation**

Create `vfe3/divergence.py`:

```python
r"""The divergence seam for VFE_3.0.

Renyi alpha-divergence is the primitive; KL is its alpha = 1 special case.
Every caller imports the divergence it needs from this single module.
Concrete kernels are registered by family name so variants swap by config.

Diagonal Gaussian KL:
    KL(q || p) = 1/2 ( sum_k s_k/t_k + sum_k (mu_t^k - mu_q^k)^2/t_k
                       - K + sum_k log(t_k/s_k) )
Diagonal Gaussian Renyi (blend sigma_b = (1-a) s + a t):
    D_a(q || p) = 1/2 [ a sum_k (mu_t-mu_q)^2/sigma_b
                        + 1/(a-1) sum_k ((1-a) log s + a log t - log sigma_b) ]
"""

from typing import Callable, Dict

import torch


def safe_kl_clamp(
    kl:     torch.Tensor,

    kl_max: float = 100.0,
) -> torch.Tensor:
    r"""Clamp to [0, kl_max]; map NaN/+inf -> kl_max, -inf -> 0.

    Matches VFE_2.0 ``safe_kl_clamp`` default (non-propagating) policy:
    degenerate pairs become repulsive (kl_max) so a downstream softmax
    ignores them rather than attending to them.
    """
    kl = kl.clamp(min=0.0, max=kl_max)
    return kl.nan_to_num(nan=kl_max, posinf=kl_max, neginf=0.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_divergence.py::test_safe_kl_clamp_bounds_and_nan -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```powershell
git add vfe3/divergence.py tests/test_divergence.py
git commit -m @'
feat(divergence): safe_kl_clamp matching 2.0 non-propagating policy

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
'@
```

---

## Task 1.2: Divergence registry seam

**Files:**
- Modify: `vfe3/divergence.py`
- Test: `tests/test_divergence.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_divergence.py`:

```python
def test_registry_register_and_get():
    from vfe3.divergence import register_divergence, get_divergence

    @register_divergence("dummy_family")
    def _dummy(mu_q, sigma_q, mu_t, sigma_t, *, alpha, kl_max, eps):
        return mu_q.sum(dim=-1) * 0.0

    fn = get_divergence("dummy_family")
    assert fn is _dummy


def test_registry_unknown_raises():
    import pytest
    from vfe3.divergence import get_divergence
    with pytest.raises(KeyError):
        get_divergence("no_such_family")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_divergence.py::test_registry_register_and_get -v`
Expected: FAIL with `ImportError: cannot import name 'register_divergence'`.

- [ ] **Step 3: Write minimal implementation**

Append to `vfe3/divergence.py` (after `safe_kl_clamp`):

```python
# ---------------------------------------------------------------------------
# Registry: family name -> divergence callable. Variants swap by config.
# Signature: fn(mu_q, sigma_q, mu_t, sigma_t, *, alpha, kl_max, eps) -> Tensor
# ---------------------------------------------------------------------------
_DIVERGENCES: Dict[str, Callable] = {}


def register_divergence(name: str) -> Callable:
    """Decorator registering a divergence kernel under ``name``."""
    def _wrap(fn: Callable) -> Callable:
        _DIVERGENCES[name] = fn
        return fn
    return _wrap


def get_divergence(name: str) -> Callable:
    """Return the registered divergence kernel for ``name`` (KeyError if absent)."""
    if name not in _DIVERGENCES:
        raise KeyError(
            f"no divergence registered under {name!r}; "
            f"available: {sorted(_DIVERGENCES)}"
        )
    return _DIVERGENCES[name]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_divergence.py -k registry -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```powershell
git add vfe3/divergence.py tests/test_divergence.py
git commit -m @'
feat(divergence): config-selectable divergence registry seam

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
'@
```

---

## Task 1.3: Diagonal Gaussian kernel — KL branch (α=1) + golden test

**Files:**
- Modify: `vfe3/divergence.py`
- Test: `tests/golden/test_divergence_golden.py`

- [ ] **Step 1: Write the failing golden test**

Append to `tests/golden/test_divergence_golden.py`:

```python
def _rand_diag(B, N, K, device, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    mu_q = torch.randn(B, N, K, generator=g).to(device)
    mu_t = torch.randn(B, N, K, generator=g).to(device)
    # variances in [0.1, 1.1], well-conditioned
    sigma_q = (torch.rand(B, N, K, generator=g) + 0.1).to(device)
    sigma_t = (torch.rand(B, N, K, generator=g) + 0.1).to(device)
    return mu_q, sigma_q, mu_t, sigma_t


def test_diagonal_kl_matches_vfe2(vfe2_kl, device):
    from vfe3.divergence import get_divergence
    mu_q, sigma_q, mu_t, sigma_t = _rand_diag(2, 4, 5, device, seed=0)
    ref = vfe2_kl._kl_kernel_diagonal(
        mu_q, sigma_q, mu_t, sigma_t, kl_max=100.0, eps=1e-6, alpha_div=1.0
    )
    got = get_divergence("gaussian_diagonal")(
        mu_q, sigma_q, mu_t, sigma_t, alpha=1.0, kl_max=100.0, eps=1e-6
    )
    assert torch.allclose(got, ref, atol=1e-5, rtol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/golden/test_divergence_golden.py::test_diagonal_kl_matches_vfe2 -v`
Expected: FAIL with `KeyError: "no divergence registered under 'gaussian_diagonal'..."`.

- [ ] **Step 3: Write minimal implementation**

Append to `vfe3/divergence.py`:

```python
@register_divergence("gaussian_diagonal")
def _gaussian_diagonal_renyi(
    mu_q:    torch.Tensor,             # (..., K) query means
    sigma_q: torch.Tensor,             # (..., K) query diagonal variances
    mu_t:    torch.Tensor,             # (..., K) transported key means
    sigma_t: torch.Tensor,             # (..., K) transported key diagonal variances

    *,
    alpha:   float = 1.0,
    kl_max:  float = 100.0,
    eps:     float = 1e-6,
) -> torch.Tensor:
    r"""Diagonal Gaussian Renyi divergence; KL at ``alpha == 1``.

    Ported from VFE_2.0 ``_kl_kernel_diagonal`` (kl_computation.py:419-459).
    """
    K = mu_q.shape[-1]
    mu_q = mu_q.float()
    sigma_q = sigma_q.float().clamp(min=eps)
    mu_t = mu_t.float()
    sigma_t = sigma_t.float().clamp(min=eps)

    if abs(alpha - 1.0) < 1e-6:
        trace_term  = (sigma_q / sigma_t).sum(dim=-1)
        delta       = mu_t - mu_q
        mahal_term  = ((delta ** 2) / sigma_t).sum(dim=-1)
        logdet_term = (torch.log(sigma_t) - torch.log(sigma_q)).sum(dim=-1)
        div = 0.5 * (trace_term + mahal_term - K + logdet_term)
    else:
        sigma_blend = ((1.0 - alpha) * sigma_q + alpha * sigma_t).clamp(min=eps)
        delta       = mu_t - mu_q
        mahal_term  = (alpha * (delta ** 2) / sigma_blend).sum(dim=-1)
        logdet_per_dim = (
            (1.0 - alpha) * torch.log(sigma_q)
            + alpha * torch.log(sigma_t)
            - torch.log(sigma_blend)
        )
        logdet_term = logdet_per_dim.sum(dim=-1) / (alpha - 1.0)
        div = 0.5 * (mahal_term + logdet_term)

    return safe_kl_clamp(div, kl_max)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/golden/test_divergence_golden.py::test_diagonal_kl_matches_vfe2 -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```powershell
git add vfe3/divergence.py tests/golden/test_divergence_golden.py
git commit -m @'
feat(divergence): diagonal Gaussian KL kernel, golden-equal to 2.0

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
'@
```

---

## Task 1.4: Diagonal Gaussian kernel — Rényi branch (α≠1) golden test

**Files:**
- Test: `tests/golden/test_divergence_golden.py`

(The implementation from Task 1.3 already covers α≠1; this task pins it.)

- [ ] **Step 1: Write the failing test**

Append to `tests/golden/test_divergence_golden.py`:

```python
import pytest


@pytest.mark.parametrize("alpha", [0.5, 0.9, 1.5, 2.0])
def test_diagonal_renyi_matches_vfe2(vfe2_kl, device, alpha):
    from vfe3.divergence import get_divergence
    mu_q, sigma_q, mu_t, sigma_t = _rand_diag(2, 4, 5, device, seed=1)
    ref = vfe2_kl._kl_kernel_diagonal(
        mu_q, sigma_q, mu_t, sigma_t, kl_max=100.0, eps=1e-6, alpha_div=alpha
    )
    got = get_divergence("gaussian_diagonal")(
        mu_q, sigma_q, mu_t, sigma_t, alpha=alpha, kl_max=100.0, eps=1e-6
    )
    assert torch.allclose(got, ref, atol=1e-5, rtol=1e-5)
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/golden/test_divergence_golden.py::test_diagonal_renyi_matches_vfe2 -v`
Expected: 4 passed.

(If any α>1 case mismatches due to the blend clamp, confirm both sides use `eps=1e-6`; the clamp is identical in both implementations.)

- [ ] **Step 3: Commit**

```powershell
git add tests/golden/test_divergence_golden.py
git commit -m @'
test(golden): pin diagonal Renyi (alpha != 1) to 2.0 across alpha grid

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
'@
```

---

## Task 1.5: Full-covariance Gaussian kernel — KL branch (α=1) + golden test

**Files:**
- Modify: `vfe3/divergence.py`
- Test: `tests/golden/test_divergence_golden.py`

- [ ] **Step 1: Write the failing golden test**

Append to `tests/golden/test_divergence_golden.py`:

```python
def _rand_full(B, N, K, device, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    mu_q = torch.randn(B, N, K, generator=g).to(device)
    mu_t = torch.randn(B, N, K, generator=g).to(device)
    Aq = torch.randn(B, N, K, K, generator=g)
    At = torch.randn(B, N, K, K, generator=g)
    eye = torch.eye(K)
    # SPD, well-conditioned: A A^T + I
    sigma_q = (Aq @ Aq.transpose(-1, -2) + eye).to(device)
    sigma_t = (At @ At.transpose(-1, -2) + eye).to(device)
    return mu_q, sigma_q, mu_t, sigma_t


def test_full_kl_matches_vfe2(vfe2_kl, device):
    from vfe3.divergence import get_divergence
    mu_q, sigma_q, mu_t, sigma_t = _rand_full(2, 3, 4, device, seed=2)
    ref = vfe2_kl._kl_kernel_dense(
        mu_q, sigma_q, mu_t, sigma_t, kl_max=100.0, eps=1e-6, alpha_div=1.0
    )
    got = get_divergence("gaussian_full")(
        mu_q, sigma_q, mu_t, sigma_t, alpha=1.0, kl_max=100.0, eps=1e-6
    )
    assert torch.allclose(got, ref, atol=1e-4, rtol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/golden/test_divergence_golden.py::test_full_kl_matches_vfe2 -v`
Expected: FAIL with `KeyError: "no divergence registered under 'gaussian_full'..."`.

- [ ] **Step 3: Write minimal implementation**

Append to `vfe3/divergence.py`:

```python
@register_divergence("gaussian_full")
def _gaussian_full_renyi(
    mu_q:    torch.Tensor,             # (..., K) query means
    sigma_q: torch.Tensor,             # (..., K, K) query covariances
    mu_t:    torch.Tensor,             # (..., K) transported key means
    sigma_t: torch.Tensor,             # (..., K, K) transported key covariances

    *,
    alpha:   float = 1.0,
    kl_max:  float = 100.0,
    eps:     float = 1e-6,
) -> torch.Tensor:
    r"""Full-covariance Gaussian Renyi divergence; KL at ``alpha == 1``.

    Ported from VFE_2.0 ``_kl_kernel_dense`` (kl_computation.py:270-330),
    with the default ``eps * I`` regularization. The 5-round Cholesky
    escalation and NaN-pair masking are 2.0 robustness features not needed
    for well-conditioned inputs; they are deferred to a later hardening task.
    """
    K = mu_q.shape[-1]
    device = mu_q.device
    mu_q = mu_q.float()
    sigma_q = sigma_q.float()
    mu_t = mu_t.float()
    sigma_t = sigma_t.float()

    eye = torch.eye(K, device=device, dtype=torch.float32)
    sigma_q_reg = sigma_q + eps * eye
    sigma_t_reg = sigma_t + eps * eye

    def _logdet_chol(L: torch.Tensor) -> torch.Tensor:
        return 2.0 * torch.log(
            torch.diagonal(L, dim1=-2, dim2=-1).clamp(min=1e-12)
        ).sum(dim=-1)

    if abs(alpha - 1.0) < 1e-6:
        L_p = torch.linalg.cholesky(sigma_t_reg)
        Y = torch.linalg.solve_triangular(L_p, sigma_q_reg, upper=False)
        Z = torch.linalg.solve_triangular(L_p.transpose(-1, -2), Y, upper=True)
        trace_term = torch.diagonal(Z, dim1=-2, dim2=-1).sum(dim=-1)
        delta_mu = mu_t - mu_q
        v = torch.linalg.solve_triangular(
            L_p, delta_mu.unsqueeze(-1), upper=False
        ).squeeze(-1)
        mahal_term = (v ** 2).sum(dim=-1)
        logdet_p = _logdet_chol(L_p)
        logdet_q = _logdet_chol(torch.linalg.cholesky(sigma_q_reg))
        div = 0.5 * (trace_term + mahal_term - K + logdet_p - logdet_q)
    else:
        sigma_blend = (1.0 - alpha) * sigma_q_reg + alpha * sigma_t_reg
        sigma_blend = 0.5 * (sigma_blend + sigma_blend.transpose(-1, -2))
        L_blend = torch.linalg.cholesky(sigma_blend)
        delta_mu = mu_t - mu_q
        v = torch.linalg.solve_triangular(
            L_blend, delta_mu.unsqueeze(-1), upper=False
        ).squeeze(-1)
        mahal_term = alpha * (v ** 2).sum(dim=-1)
        logdet_q = _logdet_chol(torch.linalg.cholesky(sigma_q_reg))
        logdet_t = _logdet_chol(torch.linalg.cholesky(sigma_t_reg))
        logdet_blend = _logdet_chol(L_blend)
        logdet_term = (
            (1.0 - alpha) * logdet_q + alpha * logdet_t - logdet_blend
        ) / (alpha - 1.0)
        div = 0.5 * (mahal_term + logdet_term)

    return safe_kl_clamp(div, kl_max)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/golden/test_divergence_golden.py::test_full_kl_matches_vfe2 -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```powershell
git add vfe3/divergence.py tests/golden/test_divergence_golden.py
git commit -m @'
feat(divergence): full-cov Gaussian KL kernel, golden-equal to 2.0

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
'@
```

---

## Task 1.6: Full-covariance Gaussian kernel — Rényi branch (α≠1) golden test

**Files:**
- Test: `tests/golden/test_divergence_golden.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/golden/test_divergence_golden.py`:

```python
@pytest.mark.parametrize("alpha", [0.5, 0.9, 1.5])
def test_full_renyi_matches_vfe2(vfe2_kl, device, alpha):
    from vfe3.divergence import get_divergence
    mu_q, sigma_q, mu_t, sigma_t = _rand_full(2, 3, 4, device, seed=3)
    ref = vfe2_kl._kl_kernel_dense(
        mu_q, sigma_q, mu_t, sigma_t, kl_max=100.0, eps=1e-6, alpha_div=alpha
    )
    got = get_divergence("gaussian_full")(
        mu_q, sigma_q, mu_t, sigma_t, alpha=alpha, kl_max=100.0, eps=1e-6
    )
    assert torch.allclose(got, ref, atol=1e-4, rtol=1e-4)
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/golden/test_divergence_golden.py::test_full_renyi_matches_vfe2 -v`
Expected: 3 passed.

- [ ] **Step 3: Commit**

```powershell
git add tests/golden/test_divergence_golden.py
git commit -m @'
test(golden): pin full-cov Renyi (alpha != 1) to 2.0 across alpha grid

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
'@
```

---

## Task 1.7: Public `renyi` / `kl` dispatch API

**Files:**
- Modify: `vfe3/divergence.py`
- Test: `tests/test_divergence.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_divergence.py`:

```python
def test_kl_equals_renyi_at_alpha_one():
    from vfe3.divergence import kl, renyi
    g = torch.Generator().manual_seed(7)
    mu_q = torch.randn(3, 5, generator=g)
    mu_t = torch.randn(3, 5, generator=g)
    sigma_q = torch.rand(3, 5, generator=g) + 0.1
    sigma_t = torch.rand(3, 5, generator=g) + 0.1
    a = kl(mu_q, sigma_q, mu_t, sigma_t, family="gaussian_diagonal")
    b = renyi(mu_q, sigma_q, mu_t, sigma_t, alpha=1.0, family="gaussian_diagonal")
    assert torch.allclose(a, b)


def test_renyi_dispatches_on_family():
    from vfe3.divergence import renyi
    g = torch.Generator().manual_seed(8)
    mu = torch.randn(2, 4, generator=g)
    A = torch.randn(2, 4, 4, generator=g)
    sigma_full = A @ A.transpose(-1, -2) + torch.eye(4)
    out = renyi(mu, sigma_full, mu, sigma_full, alpha=1.0, family="gaussian_full")
    assert out.shape == (2,)
    assert torch.allclose(out, torch.zeros(2), atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_divergence.py -k "renyi or kl_equals" -v`
Expected: FAIL with `ImportError: cannot import name 'renyi'`.

- [ ] **Step 3: Write minimal implementation**

Append to `vfe3/divergence.py`:

```python
def renyi(
    mu_q:    torch.Tensor,
    sigma_q: torch.Tensor,
    mu_t:    torch.Tensor,
    sigma_t: torch.Tensor,

    *,
    alpha:   float = 1.0,
    kl_max:  float = 100.0,
    eps:     float = 1e-6,
    family:  str   = "gaussian_diagonal",
) -> torch.Tensor:
    """Renyi alpha-divergence D_alpha(q || p) for the selected family."""
    return get_divergence(family)(
        mu_q, sigma_q, mu_t, sigma_t, alpha=alpha, kl_max=kl_max, eps=eps
    )


def kl(
    mu_q:    torch.Tensor,
    sigma_q: torch.Tensor,
    mu_t:    torch.Tensor,
    sigma_t: torch.Tensor,

    *,
    kl_max:  float = 100.0,
    eps:     float = 1e-6,
    family:  str   = "gaussian_diagonal",
) -> torch.Tensor:
    """KL(q || p) = Renyi at alpha = 1."""
    return renyi(
        mu_q, sigma_q, mu_t, sigma_t,
        alpha=1.0, kl_max=kl_max, eps=eps, family=family,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_divergence.py -k "renyi or kl_equals" -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```powershell
git add vfe3/divergence.py tests/test_divergence.py
git commit -m @'
feat(divergence): public renyi/kl dispatch API (KL = Renyi at alpha=1)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
'@
```

---

## Task 1.8: Divergence property tests

**Files:**
- Test: `tests/test_divergence.py`

- [ ] **Step 1: Write the failing/passing property tests**

Append to `tests/test_divergence.py`:

```python
import pytest


@pytest.mark.parametrize("family", ["gaussian_diagonal", "gaussian_full"])
def test_self_divergence_is_zero(family):
    from vfe3.divergence import kl
    g = torch.Generator().manual_seed(11)
    mu = torch.randn(4, 6, generator=g)
    if family == "gaussian_diagonal":
        sigma = torch.rand(4, 6, generator=g) + 0.1
    else:
        A = torch.randn(4, 6, 6, generator=g)
        sigma = A @ A.transpose(-1, -2) + torch.eye(6)
    out = kl(mu, sigma, mu, sigma, family=family)
    assert torch.allclose(out, torch.zeros(4), atol=1e-4)


@pytest.mark.parametrize("family", ["gaussian_diagonal", "gaussian_full"])
def test_divergence_nonnegative(family):
    from vfe3.divergence import kl
    g = torch.Generator().manual_seed(12)
    mu_q = torch.randn(8, 6, generator=g)
    mu_t = torch.randn(8, 6, generator=g)
    if family == "gaussian_diagonal":
        sigma_q = torch.rand(8, 6, generator=g) + 0.1
        sigma_t = torch.rand(8, 6, generator=g) + 0.1
    else:
        Aq = torch.randn(8, 6, 6, generator=g)
        At = torch.randn(8, 6, 6, generator=g)
        sigma_q = Aq @ Aq.transpose(-1, -2) + torch.eye(6)
        sigma_t = At @ At.transpose(-1, -2) + torch.eye(6)
    out = kl(mu_q, sigma_q, mu_t, sigma_t, family=family)
    assert (out >= 0.0).all()
```

- [ ] **Step 2: Run the property tests**

Run: `pytest tests/test_divergence.py -k "self_divergence or nonnegative" -v`
Expected: 4 passed.

- [ ] **Step 3: Run the full suite**

Run: `pytest -v`
Expected: all pass (golden tests pass with `..\VFE_2.0` present, else skipped).

- [ ] **Step 4: Commit**

```powershell
git add tests/test_divergence.py
git commit -m @'
test(divergence): self-divergence zero and non-negativity properties

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
'@
```

---

## Self-Review

**Spec coverage (this plan covers the Phase 0 + Phase 1 slice of the spec):**
- Spec §4.8 golden harness → Task 0.3 (`tests/golden/conftest.py` importing a pinned 2.0 checkout).
- Spec §4.1 `divergence.py` single seam, Rényi primitive / KL as α=1 → Tasks 1.1–1.7.
- Spec §4 modularity (registry behind every seam) → Task 1.2 registry; families selected by name in Tasks 1.3/1.5/1.7.
- Spec §4.7 clean config dataclass → Task 0.2 (divergence-layer fields only; later phases extend it).
- Spec §7 golden + property tests → Tasks 1.3–1.8.
- Spec Phase 1 acceptance (golden vs `kl_computation`) → Tasks 1.3–1.6. (The `gauge_utils` fused kernels and `prior_bank` decode KLs named in the spec's Phase 1 are matched in a later task once the block-diagonal layout exists; this plan pins the two base kernels `_kl_kernel_diagonal`/`_kl_kernel_dense` that those reduce to.)

**Deferred to later plans (named here so they are not forgotten):** the `families/` `ExponentialFamily` interface (`base.py`, `gaussian.py` parameter representation, `log_partition`, `entropy`) is introduced in the Phase 3/5 exp-family work; this plan keeps `divergence.py` self-contained with the two Gaussian kernels registered directly, exactly as the spec says for the Gaussian-only stage. The full-cov 5-round Cholesky escalation and NaN-pair masking from 2.0 are deferred to a robustness-hardening task.

**Placeholder scan:** none — every step has runnable code or commands.

**Type/name consistency:** `register_divergence`/`get_divergence`, kernel signature `(mu_q, sigma_q, mu_t, sigma_t, *, alpha, kl_max, eps)`, family keys `"gaussian_diagonal"`/`"gaussian_full"`, and public `renyi`/`kl` are used consistently across Tasks 1.2–1.8 and the golden tests.
