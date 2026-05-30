# VFE_3.0 Phase 4 (Gradient Oracle + Belief Kernels: filtering / smoothing) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:test-driven-development (RED→GREEN→COMMIT per step). Tests are **V3-internal** (autograd oracle + finite-difference + analytic). V3 is self-contained; see the Provenance rule.

**Goal:** Build the belief-gradient layer: the **autograd-of-F oracle** (the correctness source of truth) and the optimized **hand-derived diagonal-KL belief kernel**, behind a family-keyed registry with **oracle fallback**, and behind a `gradient_mode` seam (`filtering` = query-side mean-field default; `smoothing` = full gradient, the theoretically pure path under a toggle). The hand kernel is pinned to match the **filtering** oracle (NOT the full oracle) and finite-difference. The φ-gradient stays autograd.

**Architecture:** Phase 4 of the spec (§4.4). The belief-coupling free energy `Σ_ij β_ij KL(q_i || Ω_ij q_j)` makes each token's belief appear in two roles — **query** (row i, first KL argument) and **key** (column i, second argument, transported by Ω). The true `∇_{q_i}F` is therefore the **full** gradient with a query-side (row) sum and a key-side (column) sum, the key-side being the second-argument KL gradient pulled back through `Ωᵀ` (manuscript `eq:envelope_gradient_belief`, `eq:keyside_partial`). The **default** belief update in both the manuscript and the reference is **query-side only** ("local mean-field coordinate-ascent, holding other beliefs fixed" = *filtering*); the **full** gradient (adding the key-side = *smoothing*) is an explicit opt-in. This phase ships:
- `gradients/oracle.py` — autograd of canonical F, supporting both modes via a **query-leaf / key-detached split** (filtering) vs a **single shared leaf** (smoothing). This is the reference for every divergence/family/mode.
- `gradients/kernels.py` — the hand-derived **query-side diagonal-KL** (filtering) kernel, registered by family, with a `belief_gradients(...)` dispatch that uses the kernel only for `filtering + gaussian_diagonal + KL` and **falls back to the oracle** otherwise.
- `alpha_i.py` gains `alpha_gradient_coefficient` (the effective coefficient `a_i` multiplying `∂D`; `= α*` at the state-dependent stationary point, by the Phase-3 α-envelope).

The key-side (smoothing) **kernel**, full-covariance, fused-attention, and RoPE-gauge kernels are **deferred** (smoothing uses the oracle now). Preconditioning (Phase 2c `natural_gradient`) and the retraction (Phase 2d) stay downstream in the E-step (Phase 6) — kernels and oracle compare **raw** Euclidean `∂F`.

**Tech Stack:** Python 3, PyTorch (float32; float64 only where conditioning demands), pytest. No NN. No CLI. Device-agnostic.

**Reference spec:** §4.4 + §7. Prereq: Phases 0–3 on `main` (branch `phase4-gradients`). Reuses `divergence`, `free_energy` (`pairwise_energy`, `self_divergence`, `attention_weights`, `free_energy`), `alpha_i`, `geometry/transport` (`transport_mean`, `transport_covariance`).

**Manuscript theory (authority):**
- Full belief gradient (`Participatory_it_from_bit.tex` `eq:envelope_gradient_belief`):
  `∇_{x_i}F_red = α ∂₁D(q_i‖p_i) + Σ_j β*_ij ∂₁E_ij + Σ_ℓ β*_ℓi ∂₂E_ℓi − ∂ℓ_i`, where the key-side `∂₂E_ℓi` is pulled back through `Ωᵀ`.
- Query-side partial (`GL(K)_attention.tex` `eq:queryside_partial`): `∂E_ij/∂μ_i = (Ω_ij Σ_j Ω_ijᵀ)⁻¹(μ_i − Ω_ij μ_j)` → diagonal: `(μ_i − Ω_ij μ_j)/σ_t,ij`.
- Key-side partial (`eq:keyside_partial`): `∂E_ℓi/∂μ_i = −Ω_ℓiᵀ (Ω_ℓi Σ_i Ω_ℓiᵀ)⁻¹(μ_ℓ − Ω_ℓi μ_i)`; transport relation `∂E_ij/∂μ_j = −Ω_ijᵀ ∂E_ij/∂μ_i`.
- Filtering vs smoothing: the default reference applies only the query-side (filtering) row sum; the opt-in path descends the full gradient (smoothing). The α-envelope (Phase 3): at α*, `∇_q F = α* ∂_q D` (the α′·D and R′ paths cancel — *requires R present in F*).

Diagonal-KL query-side (filtering) kernel for token i:
```
grad_mu_i    = a_i * (mu_i - mu_p_i)/sigma_p_i  +  Sum_j beta_ij * (mu_i - mu_t_ij)/sigma_t_ij
grad_sigma_i = a_i * 0.5*(1/sigma_p_i - 1/sigma_q_i)  +  Sum_j beta_ij * 0.5*(1/sigma_t_ij - 1/sigma_q_i)
```
with `mu_t_ij = Ω_ij mu_j`, `sigma_t_ij = diag(Ω_ij Σ_j Ω_ijᵀ)`, `a_i` the α-coefficient from `alpha_i`.

**Design decisions settled before this plan (do not relitigate):**
1. **The filtering oracle is a query-leaf / key-detached SPLIT, not a global detach.** `mu_q`/`sigma_q` are the leaves (query + self role); `mu_k = mu_q.detach()`, `sigma_k = sigma_q.detach()` (key role). F's first KL argument + self-coupling use `mu_q,sigma_q`; every transported second argument uses `mu_k,sigma_k`. `filtering = autograd.grad(F, [mu_q,sigma_q])` (column-i contributions don't flow → query-side exactly; β being live in `mu_q` cancels by the envelope, canonical only). `smoothing` = the SAME F with a single shared leaf (`mu_k = mu_q`, no detach → query+key, the Ωᵀ pullback flows automatically through `transport_mean`). A naive single global `detach()` is WRONG (kills the query role); one shared leaf gives the FULL gradient, not filtering. If anyone "simplifies" the split to one leaf, the `kernel == filtering-oracle` test silently compares against the full gradient and the validation goes hollow.
2. **Family-keyed kernel registry + oracle fallback.** `belief_gradients(...)` uses the hand kernel only for `gradient_mode='filtering'` AND `family='gaussian_diagonal'` AND `alpha_div==1` (KL) AND canonical; **every other case falls back to `belief_gradients_autograd`** (smoothing, non-KL family, Rényi α≠1, surrogate). A new divergence then works immediately and correctly via the oracle, and can be accelerated later by registering a kernel.
3. **The kernel matches the FILTERING oracle, never the full oracle.** They differ by exactly the key-side term. Validate with FD (`filtering-oracle == FD(F_filt, keys frozen)`, `full-oracle == FD(full F)`); the analytic key-side (`eq:keyside_partial`) is deferred with the key-side kernel.
4. **α-coefficient `a_i` from `alpha_i.alpha_gradient_coefficient`** = `α*` for state-dependent (envelope; needs R in F), `value` for constant. No product-rule correction at α*.
5. `gradient_mode` is **config-selectable** (`smoothing` = pure path under toggle). φ-gradient stays autograd. Modes named `filtering`/`smoothing`; "query-side"/"full" are the docstring gloss.

---

## Code Style (MANDATORY — repo CLAUDE.md)

Tensors first; then `float|Tensor`; undefined; defined scalars; `Optional`; `**kwargs`. Vertical alignment of names/types/`=`/trailing-`#`; type hints; docstrings carry the LaTeX/math; shape comments. Names match notation (`mu_q`, `sigma_q`, `mu_t`, `sigma_t`, `beta`, `a_i`/`alpha_coef`, `gradient_mode`).

## Provenance (MANDATORY — convention as of commit `114839c`)

No shipped artifact (docstring/comment/test name/test comment) may contain "VFE_2.0", "2.0", or "ported". Cite the manuscript + math only.

---

## File Structure

- **Create** `vfe3/gradients/__init__.py` (empty package marker).
- **Create** `vfe3/gradients/oracle.py` — `belief_gradients_autograd`.
- **Create** `vfe3/gradients/kernels.py` — `register_kernel`/`get_kernel`, `_diag_kl_filtering_kernel`, `belief_gradients`.
- **Modify** `vfe3/alpha_i.py` — add `alpha_gradient_coefficient`.
- **Create** `tests/test_gradients_oracle.py`, `tests/test_gradients_kernels.py`, and extend `tests/test_alpha_i.py`.

---

## Task 1 — `alpha_i.py`: `alpha_gradient_coefficient`

**Files:** Modify `vfe3/alpha_i.py`; Test `tests/test_alpha_i.py`.

- [ ] **Step 1 (RED):** append to `tests/test_alpha_i.py`:

```python
from vfe3.alpha_i import alpha_gradient_coefficient


def test_alpha_grad_coefficient_constant_is_value():
    kl = torch.rand(3, 5)
    assert torch.allclose(alpha_gradient_coefficient(kl, mode="constant", value=2.0),
                          torch.full((3, 5), 2.0))


def test_alpha_grad_coefficient_state_dependent_is_alpha_star():
    # By the alpha-envelope, the effective coefficient is alpha* itself (the
    # alpha'*D and R' paths cancel at the stationary alpha* = c0/(b0+KL)).
    kl = torch.tensor([0.0, 1.0, 4.0])
    b0, c0 = 0.5, 2.0
    coef = alpha_gradient_coefficient(kl, mode="state_dependent", b0=b0, c0=c0)
    assert torch.allclose(coef, c0 / (b0 + kl), atol=1e-6)
```

- [ ] **Step 2:** Run — expect FAIL (ImportError).

- [ ] **Step 3 (GREEN):** append to `vfe3/alpha_i.py`:

```python
def alpha_gradient_coefficient(
    kl:    torch.Tensor,             # (..., N) or (..., N, K) self-divergence

    *,
    value: float = 1.0,
    b0:    'float | torch.Tensor' = 1.0,
    c0:    'float | torch.Tensor' = 1.0,
    mode:  str = "constant",
) -> torch.Tensor:
    r"""Effective coefficient a_i multiplying d D(q_i||p_i) in the belief gradient.

    By the alpha-envelope, at the state-dependent stationary point alpha* the
    coefficient is alpha* itself: d/dx[alpha*(D)*D + R(alpha*(D))] = alpha* dD/dx,
    because alpha + alpha'(D + b0 - c0/alpha) and the bracket vanishes at alpha*.
    So no product-rule correction is needed (R must be present in F). Constant
    mode returns ``value``.
    """
    if mode == "constant":
        return torch.full_like(kl, value)
    if mode in ("state_dependent", "state_dependent_per_coord"):
        return c0 / (b0 + kl).clamp(min=1e-12)
    raise ValueError(f"unknown alpha mode {mode!r}")
```

- [ ] **Step 4:** Run — expect 2 passed.
- [ ] **Step 5 (COMMIT):**
```
git add vfe3/alpha_i.py tests/test_alpha_i.py
git commit -m "feat(gradients): alpha_gradient_coefficient (envelope alpha*, no product-rule correction)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2 — `gradients/oracle.py`: autograd reference (filtering / smoothing)

**Files:** Create `vfe3/gradients/__init__.py`, `vfe3/gradients/oracle.py`; Test `tests/test_gradients_oracle.py`.

- [ ] **Step 1 (RED):** create `tests/test_gradients_oracle.py`. Build a real `omega` from a small φ via Phase 2b transport:

```python
import torch

from vfe3.geometry.groups import get_group
from vfe3.geometry.transport import compute_transport_operators
from vfe3.gradients.oracle import belief_gradients_autograd


def _setup(N=3, K=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    grp = get_group("glk")(K)
    phi = 0.15 * torch.randn(1, N, grp.generators.shape[0], generator=g)
    omega = compute_transport_operators(phi, grp)["Omega"][0]          # (N, N, K, K)
    mu = torch.randn(N, K, generator=g)
    sigma = torch.rand(N, K, generator=g) + 0.5
    mu_p = torch.randn(N, K, generator=g)
    sigma_p = torch.rand(N, K, generator=g) + 0.5
    return mu, sigma, mu_p, sigma_p, omega


def _F_filtering(mu_q, sigma_q, mu_p, sigma_p, omega, tau):
    # F as a function of the QUERY leaf only: keys are the DETACHED copy.
    from vfe3.free_energy import free_energy, pairwise_energy, self_divergence
    from vfe3.geometry.transport import transport_covariance, transport_mean
    mu_k, sigma_k = mu_q.detach(), sigma_q.detach()
    mu_t = transport_mean(omega.unsqueeze(0), mu_k.unsqueeze(0))[0]
    sigma_t = transport_covariance(omega.unsqueeze(0), sigma_k.unsqueeze(0))[0]
    sd = self_divergence(mu_q, sigma_q, mu_p, sigma_p)
    energy = pairwise_energy(mu_q, sigma_q, mu_t, sigma_t)
    alpha = torch.ones(mu_q.shape[0])
    return free_energy(sd, energy, alpha, tau=tau, include_attention_entropy=True)


def test_filtering_oracle_matches_finite_difference_of_F_filt():
    mu, sigma, mu_p, sigma_p, omega = _setup()
    tau = 1.5
    gmu, gsig = belief_gradients_autograd(mu, sigma, mu_p, sigma_p, omega,
                                          tau=tau, gradient_mode="filtering")
    eps = 5e-3
    gmu_fd = torch.zeros_like(mu)
    for a in range(mu.shape[0]):
        for b in range(mu.shape[1]):
            d = torch.zeros_like(mu); d[a, b] = eps
            fp = _F_filtering(mu + d, sigma, mu_p, sigma_p, omega, tau)
            fm = _F_filtering(mu - d, sigma, mu_p, sigma_p, omega, tau)
            gmu_fd[a, b] = (fp - fm) / (2 * eps)
    assert torch.allclose(gmu, gmu_fd, atol=1e-3, rtol=1e-3)


def test_smoothing_differs_from_filtering_by_keyside():
    mu, sigma, mu_p, sigma_p, omega = _setup()
    gf_mu, _ = belief_gradients_autograd(mu, sigma, mu_p, sigma_p, omega,
                                         tau=1.5, gradient_mode="filtering")
    gs_mu, _ = belief_gradients_autograd(mu, sigma, mu_p, sigma_p, omega,
                                         tau=1.5, gradient_mode="smoothing")
    # the key-side (column) term is non-zero -> the two modes differ
    assert not torch.allclose(gf_mu, gs_mu, atol=1e-4)
```

- [ ] **Step 2:** Run — expect FAIL.

- [ ] **Step 3 (GREEN):** create `vfe3/gradients/__init__.py` (empty) and `vfe3/gradients/oracle.py`:

```python
r"""Autograd belief-gradient oracle for VFE_3.0 (the correctness source of truth).

The reduced free energy F_red is differentiated w.r.t. the Gaussian belief
(mu, sigma) by torch.autograd. Two modes for the belief-coupling term, in which a
token appears both as the query (first KL argument) and the transported key
(second argument):
  filtering  query-side only: keys are a DETACHED copy of the belief, so only the
             first-argument (row) gradient flows -- the mean-field coordinate-ascent
             default (holding other beliefs fixed).
  smoothing  full gradient: keys share the belief leaf, so the second-argument
             (column) gradient flows back through the transport (Omega^T pullback)
             -- the theoretically pure d F_red.
Reference for every family / divergence / mode; the hand kernels are pinned to the
FILTERING oracle.
"""

from typing import Optional, Tuple

import torch

from vfe3.alpha_i import self_coupling_alpha
from vfe3.free_energy import free_energy, pairwise_energy, self_divergence
from vfe3.geometry.transport import transport_covariance, transport_mean


def belief_gradients_autograd(
    mu:           torch.Tensor,           # (N, K) belief means (the variable)
    sigma:        torch.Tensor,           # (N, K) belief variances
    mu_p:         torch.Tensor,           # (N, K) prior means
    sigma_p:      torch.Tensor,           # (N, K) prior variances
    omega:        torch.Tensor,           # (N, N, K, K) transport operators Omega_ij

    *,
    tau:          float = 1.0,
    alpha_div:    float = 1.0,
    kl_max:       float = 100.0,
    eps:          float = 1e-6,
    b0:           float = 1.0,
    c0:           float = 1.0,

    include_attention_entropy: bool = True,
    gradient_mode:             str  = "filtering",
    family:                    str  = "gaussian_diagonal",
    alpha_mode:                str  = "constant",

    log_prior:                 Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:   # (grad_mu, grad_sigma), each (N, K)
    r"""Autograd of canonical F_red w.r.t. (mu, sigma). See module docstring for modes."""
    mu_q = mu.detach().clone().requires_grad_(True)
    sigma_q = sigma.detach().clone().requires_grad_(True)

    if gradient_mode == "filtering":
        mu_k, sigma_k = mu_q.detach(), sigma_q.detach()       # key role frozen
    elif gradient_mode == "smoothing":
        mu_k, sigma_k = mu_q, sigma_q                          # shared leaf -> full grad
    else:
        raise ValueError(f"gradient_mode must be 'filtering' or 'smoothing', got {gradient_mode!r}")

    mu_t = transport_mean(omega.unsqueeze(0), mu_k.unsqueeze(0))[0]            # (N, N, K)
    sigma_t = transport_covariance(omega.unsqueeze(0), sigma_k.unsqueeze(0))[0]

    sd = self_divergence(mu_q, sigma_q, mu_p, sigma_p, alpha=alpha_div, kl_max=kl_max, eps=eps, family=family)
    alpha, reg = self_coupling_alpha(sd, mode=alpha_mode, b0=b0, c0=c0)
    energy = pairwise_energy(mu_q, sigma_q, mu_t, sigma_t, alpha=alpha_div, kl_max=kl_max, eps=eps, family=family)
    F = free_energy(
        sd, energy, alpha, tau=tau, include_attention_entropy=include_attention_entropy,
        log_prior=log_prior, alpha_reg=(reg if alpha_mode != "constant" else None),
    )
    grad_mu, grad_sigma = torch.autograd.grad(F, [mu_q, sigma_q])
    return grad_mu.detach(), grad_sigma.detach()
```

- [ ] **Step 4:** Run — expect 2 passed.
- [ ] **Step 5 (COMMIT):**
```
git add vfe3/gradients/__init__.py vfe3/gradients/oracle.py tests/test_gradients_oracle.py
git commit -m "feat(gradients): autograd belief-gradient oracle (filtering / smoothing split)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3 — `gradients/kernels.py`: diagonal-KL filtering kernel + registry/fallback

**Files:** Create `vfe3/gradients/kernels.py`; Test `tests/test_gradients_kernels.py`.

- [ ] **Step 1 (RED):** create `tests/test_gradients_kernels.py` (reuse `_setup` from the oracle test):

```python
import torch

from vfe3.geometry.groups import get_group
from vfe3.geometry.transport import compute_transport_operators
from vfe3.gradients.kernels import belief_gradients
from vfe3.gradients.oracle import belief_gradients_autograd


def _setup(N=3, K=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    grp = get_group("glk")(K)
    phi = 0.15 * torch.randn(1, N, grp.generators.shape[0], generator=g)
    omega = compute_transport_operators(phi, grp)["Omega"][0]
    mu = torch.randn(N, K, generator=g); sigma = torch.rand(N, K, generator=g) + 0.5
    mu_p = torch.randn(N, K, generator=g); sigma_p = torch.rand(N, K, generator=g) + 0.5
    return mu, sigma, mu_p, sigma_p, omega


def test_kernel_matches_filtering_oracle_constant_alpha():
    args = _setup()
    km, ks = belief_gradients(*args, tau=1.5, gradient_mode="filtering")
    om, os_ = belief_gradients_autograd(*args, tau=1.5, gradient_mode="filtering")
    assert torch.allclose(km, om, atol=1e-5)
    assert torch.allclose(ks, os_, atol=1e-5)


def test_kernel_differs_from_smoothing_oracle():
    args = _setup()
    km, _ = belief_gradients(*args, tau=1.5, gradient_mode="filtering")
    sm, _ = belief_gradients_autograd(*args, tau=1.5, gradient_mode="smoothing")
    assert not torch.allclose(km, sm, atol=1e-4)              # key-side term is real


def test_dispatch_falls_back_to_oracle():
    args = _setup()
    # smoothing -> oracle
    a = belief_gradients(*args, tau=1.5, gradient_mode="smoothing")
    b = belief_gradients_autograd(*args, tau=1.5, gradient_mode="smoothing")
    assert torch.allclose(a[0], b[0], atol=1e-6) and torch.allclose(a[1], b[1], atol=1e-6)
    # non-KL (Renyi alpha_div != 1) -> oracle
    c = belief_gradients(*args, tau=1.5, gradient_mode="filtering", alpha_div=0.5)
    d = belief_gradients_autograd(*args, tau=1.5, gradient_mode="filtering", alpha_div=0.5)
    assert torch.allclose(c[0], d[0], atol=1e-5) and torch.allclose(c[1], d[1], atol=1e-5)
```

- [ ] **Step 2:** Run — expect FAIL.

- [ ] **Step 3 (GREEN):** create `vfe3/gradients/kernels.py`:

```python
r"""Optimized hand-derived belief-gradient kernels for VFE_3.0, with oracle fallback.

A family-keyed registry of analytic (grad_mu, grad_sigma) kernels for the QUERY-SIDE
(filtering) gradient. belief_gradients() uses the registered kernel only for the
filtering + gaussian_diagonal + KL (alpha_div=1) + canonical case; every other case
(smoothing, non-KL family, Renyi alpha != 1, surrogate) FALLS BACK to the autograd
oracle -- so a new divergence works immediately and correctly, accelerated later by
registering a kernel. Kernels return RAW Euclidean dF (no preconditioning/retraction).
"""

from typing import Callable, Dict, Optional, Tuple

import torch

from vfe3.alpha_i import alpha_gradient_coefficient
from vfe3.free_energy import attention_weights, pairwise_energy, self_divergence
from vfe3.geometry.transport import transport_covariance, transport_mean
from vfe3.gradients.oracle import belief_gradients_autograd

_KERNELS: Dict[str, Callable] = {}


def register_kernel(name: str) -> Callable:
    """Decorator registering a query-side belief-gradient kernel under family ``name``."""
    def _wrap(fn: Callable) -> Callable:
        _KERNELS[name] = fn
        return fn
    return _wrap


def has_kernel(name: str) -> bool:
    """Whether a hand kernel is registered for family ``name``."""
    return name in _KERNELS


@register_kernel("gaussian_diagonal")
def _diag_kl_filtering_kernel(
    mu_q:       torch.Tensor,             # (N, K)
    sigma_q:    torch.Tensor,             # (N, K)
    mu_p:       torch.Tensor,             # (N, K)
    sigma_p:    torch.Tensor,             # (N, K)
    mu_t:       torch.Tensor,             # (N, N, K) transported key means
    sigma_t:    torch.Tensor,             # (N, N, K) transported key variances
    beta:       torch.Tensor,             # (N, N) attention weights
    alpha_coef: torch.Tensor,             # (N, 1) or (N, K) self-coupling coefficient

    *,
    eps:        float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Diagonal-KL query-side (filtering) gradient.

      grad_mu_i    = a_i (mu_i - mu_p_i)/sigma_p_i + Sum_j beta_ij (mu_i - mu_t_ij)/sigma_t_ij
      grad_sigma_i = a_i 0.5(1/sigma_p_i - 1/sigma_q_i)
                     + Sum_j beta_ij 0.5(1/sigma_t_ij - 1/sigma_q_i)
    """
    sp = sigma_p.clamp(min=eps); sq = sigma_q.clamp(min=eps); st = sigma_t.clamp(min=eps)

    self_mu  = alpha_coef * (mu_q - mu_p) / sp
    pair_mu  = torch.einsum("ij,ijk->ik", beta, (mu_q.unsqueeze(-2) - mu_t) / st)
    grad_mu  = self_mu + pair_mu

    self_sig = alpha_coef * 0.5 * (1.0 / sp - 1.0 / sq)
    pair_sig = torch.einsum("ij,ijk->ik", beta, 0.5 * (1.0 / st - 1.0 / sq.unsqueeze(-2)))
    grad_sigma = self_sig + pair_sig
    return grad_mu, grad_sigma


def belief_gradients(
    mu:           torch.Tensor,           # (N, K)
    sigma:        torch.Tensor,           # (N, K)
    mu_p:         torch.Tensor,           # (N, K)
    sigma_p:      torch.Tensor,           # (N, K)
    omega:        torch.Tensor,           # (N, N, K, K)

    *,
    tau:          float = 1.0,
    alpha_div:    float = 1.0,
    kl_max:       float = 100.0,
    eps:          float = 1e-6,
    b0:           float = 1.0,
    c0:           float = 1.0,

    include_attention_entropy: bool = True,
    gradient_mode:             str  = "filtering",
    family:                    str  = "gaussian_diagonal",
    alpha_mode:                str  = "constant",
    value:                     float = 1.0,

    log_prior:                 Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Belief gradient: hand kernel for filtering+gaussian_diagonal+KL+canonical, else oracle."""
    use_kernel = (
        gradient_mode == "filtering"
        and family == "gaussian_diagonal"
        and abs(alpha_div - 1.0) < 1e-9
        and include_attention_entropy
        and has_kernel(family)
    )
    if not use_kernel:
        return belief_gradients_autograd(
            mu, sigma, mu_p, sigma_p, omega, tau=tau, alpha_div=alpha_div,
            kl_max=kl_max, eps=eps, b0=b0, c0=c0,
            include_attention_entropy=include_attention_entropy,
            gradient_mode=gradient_mode, family=family, alpha_mode=alpha_mode,
            log_prior=log_prior,
        )

    mu_k, sigma_k = mu.detach(), sigma.detach()
    mu_t = transport_mean(omega.unsqueeze(0), mu_k.unsqueeze(0))[0]
    sigma_t = transport_covariance(omega.unsqueeze(0), sigma_k.unsqueeze(0))[0]
    sd = self_divergence(mu, sigma, mu_p, sigma_p, alpha=1.0, kl_max=kl_max, eps=eps, family=family)
    energy = pairwise_energy(mu, sigma, mu_t, sigma_t, alpha=1.0, kl_max=kl_max, eps=eps, family=family)
    beta = attention_weights(energy, tau=tau, log_prior=log_prior)
    coef = alpha_gradient_coefficient(sd, value=value, b0=b0, c0=c0, mode=alpha_mode).unsqueeze(-1)
    return get_kernel(family)(mu, sigma, mu_p, sigma_p, mu_t, sigma_t, beta, coef, eps=eps)


def get_kernel(name: str) -> Callable:
    """Return the registered kernel for family ``name`` (KeyError if absent)."""
    if name not in _KERNELS:
        raise KeyError(f"no belief-gradient kernel for family {name!r}; available: {sorted(_KERNELS)}")
    return _KERNELS[name]
```

- [ ] **Step 4:** Run — expect 3 passed.
- [ ] **Step 5 (COMMIT):**
```
git add vfe3/gradients/kernels.py tests/test_gradients_kernels.py
git commit -m "feat(gradients): diagonal-KL filtering kernel + family registry with oracle fallback

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4 — state-dependent α (with R) + known-value

**Files:** Test `tests/test_gradients_kernels.py`.

- [ ] **Step 1 (RED):** append. The α* cancellation test must put **R on both sides** (the oracle includes `alpha_reg`; the kernel uses the `alpha*` coefficient):

```python
def test_kernel_matches_filtering_oracle_state_dependent_alpha_with_R():
    args = _setup()
    km, ks = belief_gradients(*args, tau=1.5, gradient_mode="filtering",
                              alpha_mode="state_dependent", b0=0.5, c0=2.0)
    om, os_ = belief_gradients_autograd(*args, tau=1.5, gradient_mode="filtering",
                                        alpha_mode="state_dependent", b0=0.5, c0=2.0)
    assert torch.allclose(km, om, atol=1e-5)                 # alpha* cancellation (R on both sides)
    assert torch.allclose(ks, os_, atol=1e-5)


def test_self_gradient_vanishes_when_q_equals_p_and_identity_transport():
    K = 2
    grp = get_group("glk")(K)
    N = 3
    omega = torch.eye(K).expand(N, N, K, K).contiguous()      # identity transport
    mu = torch.randn(N, K); sigma = torch.rand(N, K) + 0.5
    # q == p: self term zero; identity transport + q==key -> belief-coupling mu term zero too
    gmu, _ = belief_gradients(mu, sigma, mu.clone(), sigma.clone(), omega,
                              tau=1.5, gradient_mode="filtering")
    assert torch.allclose(gmu, torch.zeros(N, K), atol=1e-5)
```

- [ ] **Step 2:** Run — expect 2 passed (no new implementation; exercises Task 1–3 code). If the state-dependent test fails, either `alpha_reg` is missing from the oracle's F (it must be present) or the coefficient is not `alpha*` — fix the code, not the test.

- [ ] **Step 3 (COMMIT):**
```
git add tests/test_gradients_kernels.py
git commit -m "test(gradients): alpha* cancellation (R on both sides) + q==p self-gradient zero

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5 — full suite + changelog + final

- [ ] **Step 1:** `python -m pytest -q` — expect all prior (101) + new (~9) green, no regressions.
- [ ] **Step 2:** Append "## Phase 4 Gradient Oracle + Belief Kernels — 2026-05-29 (continuation)" to `docs/edits/2026-05-29-phase2b-transport.md` (match format; provenance-clean). Commit `docs(edits): 2026-05-29 phase 4 gradients changes log`.
- [ ] **Step 3:** Final `git add -A` of phase files only (never the user's CLAUDE.md / untracked dirs) if anything is uncommitted.

---

## Self-Review

**Spec coverage (Phase 4, §4.4 + §7):**
- Autograd-of-F oracle (correctness source of truth), filtering + smoothing → Task 2.
- Hand-derived diagonal-KL query-side kernel, finite-difference + oracle pinned → Tasks 2–3.
- Family-keyed registry + oracle fallback (divergence modularity at the gradient layer) → Task 3.
- α-gradient supplied by `alpha_i` (envelope α*, no product-rule correction) → Task 1.
- The query/key/full distinction (filtering vs smoothing) as a config toggle → Tasks 2–4.

**Hand-derived anchors (independent of the implementation):**
- `filtering-oracle == FD(F_filt, keys frozen)`; `smoothing ≠ filtering` (key-side is real).
- `kernel == filtering-oracle` (constant α AND state-dependent α with R) — the α* cancellation.
- `kernel ≠ smoothing-oracle` (the key-side term is the gap).
- dispatch fallback: smoothing / non-KL / Rényi α≠1 → oracle (matches it).
- `q==p` + identity transport → self-gradient zero.

**Deferred (named extension points):** the key-side (smoothing) **kernel** (`eq:keyside_partial`, the Ωᵀ pullback + diagonal-sandwich `∂σ_t/∂σ`); full-covariance kernel; fused-attention kernel; RoPE-gauge kernel; the φ-gradient hand kernel (stays autograd). Preconditioning + retraction stay in the E-step (Phase 6).

**Type/name consistency:** `alpha_gradient_coefficient(kl, *, value, b0, c0, mode)`, `belief_gradients_autograd(mu, sigma, mu_p, sigma_p, omega, *, …, gradient_mode, family, alpha_mode, log_prior)`, `belief_gradients(...)` (same signature + `value`), `_diag_kl_filtering_kernel(mu_q, sigma_q, mu_p, sigma_p, mu_t, sigma_t, beta, alpha_coef, *, eps)`. Modes `filtering`/`smoothing`; kernel matches the filtering oracle; smoothing falls back to the oracle.
