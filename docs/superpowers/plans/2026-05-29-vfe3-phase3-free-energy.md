# VFE_3.0 Phase 3 (Single Scalar Free Energy F + α-coupling + attention prior) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:test-driven-development (RED→GREEN→COMMIT per step). Tests are **V3-internal** (analytic known-value + property + finite-difference + autograd). V3 is self-contained; see the Provenance rule.

**Goal:** Build the single authoritative scalar free energy `F = Σ_i F_i` and the two seams it pulls from: the self-coupling coefficient (`alpha_i.py`) and the attention prior (`attention_prior.py`). `free_energy.py` is the one place F is materialized; it is **divergence-agnostic** (consumes whatever the `divergence` registry produces) so a future divergence slots in by registration + config, never by editing F. Canonical (with attention entropy) vs surrogate is a single toggle. Correctness is pinned by the envelope identity, β-stationarity, the canonical−surrogate gradient gap, and autograd-vs-finite-difference — not by re-running F's own formula.

**Architecture:** Phase 3 of the spec. Sits above the divergence layer (Phase 1) and geometry (Phase 2); consumed by the E-step (Phase 6) and decode (Phase 7). Three seams, each a config-selected registry so variants swap without editing call sites:
- **divergence** (`divergence.py`, already built): `free_energy` never calls a concrete kernel — it assembles the scalar from per-pair energies `E_ij` and self-divergences `D(q_i‖p_i)` that come from `divergence.renyi`/`kl` with a config-selected `family`/`alpha`. A new divergence (different α, full-cov, or a future non-KL divergence) registers under a new family name with the same signature and is selected by config; `free_energy` and the β-layer are untouched. The β/envelope/gradient-gap properties treat `E_ij` as an opaque per-pair energy, so they hold for any divergence.
- **self-coupling** (`alpha_i.py`, new): the α_i forms (constant, state-dependent, per-coordinate) + the precision regularizer R(α).
- **attention prior** (`attention_prior.py`, new): the π_ij forms (uniform, causal, ALiBi, …) as a log-prior bias.

`free_energy.py` is **pure** (operates on belief/energy tensors + the seam outputs; no model object). The observation likelihood `ℓ_i = E_q[log p(o_i|k_i)]` is an **optional passed-in term** (default 0): the Gaussian-template observation model is the Phase 7 decode, and the two key properties do not involve ℓ, so F is fully defined and testable now. The hyper-prior `λ_h·KL(s‖h)` and model-coupling `γ·KL(s_i‖Ω s_j)` terms are **named extension points** (absent from the default path, never half-wired).

**Tech Stack:** Python 3, PyTorch (float32 storage; float64 internal only where conditioning demands it), pytest. No NN. No CLI. Device-agnostic (tests default CPU; `VFE3_TEST_DEVICE=cuda` for GPU — never hardcode `.cpu()`/`.cuda()` or a device-less `torch.eye`/`zeros`/`full`; build internal tensors on the input's device).

**Reference spec:** `docs/superpowers/specs/2026-05-29-vfe3-clean-room-design.md` (§4.3 free_energy + alpha_i; §7 properties). Prereq: Phases 0–2 on `main` (this branch: `phase3-free-energy`). Reuses `divergence.renyi`/`kl`.

**Manuscript theory (authority — `Participatory_it_from_bit.tex` `eq:free_energy_functional_final`, `eq:beta_optimal`; `GL(K)_attention.tex` `eq:autograd_envelope_gap`):**

Default per-position free energy (λ_h = 0, γ = 0):
```
F_i = α_i · D(q_i ‖ p_i)                           # self-coupling
    + Σ_j β_ij · E_ij                              # belief coupling, E_ij = D(q_i ‖ Ω_ij q_j)
    + τ Σ_j β_ij · log(β_ij / π_ij)                # attention entropy (canonical only)
    − ℓ_i                                          # observation likelihood (optional; default 0)
```
with `β*_ij = π_ij exp(−E_ij/τ) / Σ_k π_ik exp(−E_ik/τ)` (softmax over keys), `τ = κ√K` (κ a learnable scalar; the standard transformer is κ=1), `π_ij` the attention prior (uniform 1/N default). `D` is the divergence from the seam (KL = Renyi at α=1).

Three load-bearing identities (all verified by hand — the implementation must reproduce them):
- **Envelope:** substituting β* into the canonical β-block gives the reduced free energy: `Σ_j β*_ij E_ij + τ Σ_j β*_ij log(β*_ij/π_ij) = −τ log Z_i`, where `Z_i = Σ_j π_ij exp(−E_ij/τ)`. (Because `log(β*_j/π_j) = −E_j/τ − log Z`.)
- **Stationarity:** at β*, `E_ij + τ log(β*_ij/π_ij)` is **constant across j** (= −τ log Z_i); β* is the stationary point of the row-Lagrangian *only when the entropy term is present* (the surrogate's argmin is a delta, not a softmax).
- **Gradient gap (sign pinned):** `∇_x⟨E⟩_{β*} − ∇_x F_red = −τ⁻¹ Cov_{β*}(E, ∂_x E)`, i.e. `autograd(surrogate β-block) − autograd(canonical β-block) = −τ⁻¹ Cov_{β*}(E, ∂_x E)`, with `Cov_{β*}(A,B) = Σ_j β*_j A_j B_j − (Σ_j β*_j A_j)(Σ_j β*_j B_j)`. Follows from `∂_x β*_j = −τ⁻¹ β*_j (∂_x E_j − ⟨∂_x E⟩_{β*})`.

Self-coupling α (manuscript `eq:state_dependent_alpha`): default constant α=1; state-dependent `α*_i = c₀/(b₀ + D(q_i‖p_i))` minimizes `α D + R(α)` with `R(α) = b₀α − c₀ log α`; the per-coordinate form `α^(k)* = c₀^(k)/(b₀^(k) + D^(k))` is the manuscript's implemented choice. **α-envelope:** at α*, `∂F/∂α = 0`, so `∇_q F = α*·∂_q D` (the explicit α-path vanishes) — the structural twin of the β-envelope; pinning it now de-risks Phase 4's hand-derived α product-rule correction.

**Design decisions settled before this plan (do not relitigate in code):**
1. **F is divergence-agnostic.** `free_energy` consumes precomputed `self_div` (D(q_i‖p_i)) and `energy` (E_ij) tensors; it never calls a concrete divergence. The thin helpers `self_divergence`/`pairwise_energy` route through `divergence.renyi(..., family=…, alpha=…)`, so swapping the divergence is a `family`/`alpha` change. (A future non-KL divergence registers under a new family name; nothing else changes.)
2. **β is passed explicitly** to `free_energy` (so β-stationarity is testable), with `attention_weights(...)` computing β* = softmax(B − E/τ). The gradient-gap test builds β* **live** as a function of x — never detached — else the test is vacuous.
3. **Attention prior is a registry seam** producing a **log-prior bias** `B_ij` (additive in logits): `uniform`→0, `causal`→0/−∞, `alibi`→−m|i−j|. The normalized prior used in the entropy term is `π = softmax_j(B)`. Property tests use a **non-uniform** prior (uniform π cancels in β*, entropy, and Z simultaneously and would hide a π-wiring or `τ log N` bug).
4. **Likelihood optional (default 0); λ_h, γ named extension points.**
5. **Coords/units:** `free_energy` returns a scalar; `attention_weights` is `(…,N,N)`-shaped. Everything device-agnostic, float32 in/out.

---

## Code Style (MANDATORY — repo CLAUDE.md)

Tensors first; then `float|Tensor`; undefined floats/ints/bools; defined floats/ints/bools (defined `str` with them); `Optional`; `**kwargs`. Vertical alignment of names/types/`=`/trailing-`#`; type hints everywhere; docstrings carry the LaTeX/math form; shape comments. Names match notation (`mu_q`, `sigma_q`, `beta`, `tau`, `kappa`, `alpha`, `pi`/`log_prior`). A theoretically pure path must exist under toggles; opt-in extras documented.

## Provenance (MANDATORY — convention as of commit `114839c`)

V3 is self-contained, **not a port**. No shipped artifact (docstring/comment/test name/test comment) may contain "VFE_2.0", "2.0", or "ported". Cite the manuscript and the math only.

---

## File Structure

- **Create** `vfe3/alpha_i.py` — `register_alpha`/`get_alpha`, `alpha_constant`, `alpha_state_dependent`, `alpha_state_dependent_per_coord`, `alpha_regularizer`, `self_coupling_alpha` dispatcher.
- **Create** `vfe3/attention_prior.py` — `register_prior`/`get_prior`, `prior_uniform`, `prior_causal`, `prior_alibi`, `attention_log_prior` dispatcher (returns log-prior bias `B_ij`).
- **Create** `vfe3/free_energy.py` — `effective_temperature`, `pairwise_energy`, `self_divergence`, `attention_weights`, `log_partition`, `reduced_free_energy`, `free_energy`.
- **Create** `tests/test_alpha_i.py`, `tests/test_attention_prior.py`, `tests/test_free_energy.py`.

---

## Task 1 — `alpha_i.py`: self-coupling registry

**Files:** Create `vfe3/alpha_i.py`; Test `tests/test_alpha_i.py`.

- [ ] **Step 1 (RED):** create `tests/test_alpha_i.py`:

```python
import torch

from vfe3.alpha_i import alpha_regularizer, self_coupling_alpha


def test_constant_alpha_is_value_zero_reg():
    kl = torch.rand(3, 5)
    a, r = self_coupling_alpha(kl, mode="constant", value=1.0)
    assert torch.allclose(a, torch.ones(3, 5))
    assert torch.allclose(r, torch.zeros(3, 5))


def test_state_dependent_alpha_formula_and_minimizes_objective():
    # alpha* = c0/(b0 + KL) is the stationary point of  alpha*KL + b0*alpha - c0*log(alpha).
    kl = torch.tensor([0.0, 1.0, 4.0])
    b0, c0 = 0.5, 2.0
    a, r = self_coupling_alpha(kl, mode="state_dependent", b0=b0, c0=c0)
    assert torch.allclose(a, c0 / (b0 + kl), atol=1e-6)
    # d/d alpha [alpha*KL + b0*alpha - c0*log alpha] = KL + b0 - c0/alpha == 0 at alpha*
    grad = kl + b0 - c0 / a
    assert torch.allclose(grad, torch.zeros_like(grad), atol=1e-5)


def test_per_coord_alpha_uses_per_dimension_kl():
    kl = torch.rand(2, 4, 3) + 0.1                       # (..., N, K) per-coordinate KL
    b0 = torch.full((3,), 0.5)
    c0 = torch.full((3,), 2.0)
    a, r = self_coupling_alpha(kl, mode="state_dependent_per_coord", b0=b0, c0=c0)
    assert torch.allclose(a, c0 / (b0 + kl), atol=1e-6)
```

- [ ] **Step 2:** Run — expect FAIL (ImportError).

- [ ] **Step 3 (GREEN):** create `vfe3/alpha_i.py`:

```python
r"""Self-coupling coefficient alpha_i for VFE_3.0 (the weight on D(q_i || p_i)).

A config-selected registry of forms:
  constant                  alpha = value (default 1.0); no regularizer.
  state_dependent           alpha*_i = c0 / (b0 + D(q_i||p_i)), the stationary
                            point of alpha*D + R(alpha), R(alpha)=b0*alpha - c0*log alpha.
  state_dependent_per_coord per-coordinate alpha^(k)* = c0^(k)/(b0^(k) + D^(k)).
Pure: a function of the (per-position or per-coordinate) self-divergence D.
"""

from typing import Callable, Dict, Tuple

import torch

_ALPHAS: Dict[str, Callable] = {}


def register_alpha(name: str) -> Callable:
    """Decorator registering an alpha form D -> (alpha, regularizer)."""
    def _wrap(fn: Callable) -> Callable:
        _ALPHAS[name] = fn
        return fn
    return _wrap


def get_alpha(name: str) -> Callable:
    """Return the registered alpha form (KeyError if absent)."""
    if name not in _ALPHAS:
        raise KeyError(f"no alpha form {name!r}; available: {sorted(_ALPHAS)}")
    return _ALPHAS[name]


def alpha_regularizer(
    alpha: torch.Tensor,             # (...) coupling coefficient

    *,
    b0:    'float | torch.Tensor' = 1.0,
    c0:    'float | torch.Tensor' = 1.0,
) -> torch.Tensor:
    r"""Precision regularizer R(alpha) = b0*alpha - c0*log(alpha)."""
    return b0 * alpha - c0 * torch.log(alpha.clamp(min=1e-12))


@register_alpha("constant")
def alpha_constant(
    kl:    torch.Tensor,             # (..., N) or (..., N, K) self-divergence (unused)

    *,
    value: float = 1.0,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Constant alpha = value, zero regularizer."""
    return torch.full_like(kl, value), torch.zeros_like(kl)


@register_alpha("state_dependent")
def alpha_state_dependent(
    kl:    torch.Tensor,             # (..., N) per-position self-divergence

    *,
    b0:    'float | torch.Tensor' = 1.0,
    c0:    'float | torch.Tensor' = 1.0,
    eps:   float = 1e-12,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""State-dependent alpha*_i = c0 / (b0 + D(q_i||p_i)); R(alpha*)."""
    alpha = c0 / (b0 + kl).clamp(min=eps)
    return alpha, alpha_regularizer(alpha, b0=b0, c0=c0)


@register_alpha("state_dependent_per_coord")
def alpha_state_dependent_per_coord(
    kl:    torch.Tensor,             # (..., N, K) per-coordinate self-divergence

    *,
    b0:    'float | torch.Tensor' = 1.0,   # scalar or (K,)
    c0:    'float | torch.Tensor' = 1.0,   # scalar or (K,)
    eps:   float = 1e-12,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Per-coordinate alpha^(k)* = c0^(k)/(b0^(k) + D^(k)); R summed by caller."""
    alpha = c0 / (b0 + kl).clamp(min=eps)
    return alpha, alpha_regularizer(alpha, b0=b0, c0=c0)


def self_coupling_alpha(
    kl:    torch.Tensor,             # (..., N) or (..., N, K) self-divergence

    *,
    value: float = 1.0,
    b0:    'float | torch.Tensor' = 1.0,
    c0:    'float | torch.Tensor' = 1.0,
    mode:  str = "constant",
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Dispatch to the registered alpha form `mode`; returns (alpha, regularizer)."""
    return get_alpha(mode)(kl, value=value, b0=b0, c0=c0)
```

- [ ] **Step 4:** Run — expect 3 passed.
- [ ] **Step 5 (COMMIT):**
```
git add vfe3/alpha_i.py tests/test_alpha_i.py
git commit -m "feat(free-energy): self-coupling alpha registry (constant / state-dependent / per-coord)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2 — `attention_prior.py`: attention-prior registry

**Files:** Create `vfe3/attention_prior.py`; Test `tests/test_attention_prior.py`.

- [ ] **Step 1 (RED):** create `tests/test_attention_prior.py`:

```python
import torch

from vfe3.attention_prior import attention_log_prior


def test_uniform_is_zero_bias():
    B = attention_log_prior("uniform", 4, 4)
    assert torch.allclose(B, torch.zeros(4, 4))


def test_causal_masks_future_keys():
    B = attention_log_prior("causal", 3, 3)
    # j > i masked (-inf), j <= i allowed (0)
    assert torch.isneginf(B[0, 1]) and torch.isneginf(B[0, 2]) and torch.isneginf(B[1, 2])
    assert B[2, 0] == 0.0 and B[1, 1] == 0.0 and B[2, 2] == 0.0


def test_alibi_is_linear_in_distance():
    B = attention_log_prior("alibi", 4, 4, slope=0.5)
    # B_ij = -slope * |i - j|
    for i in range(4):
        for j in range(4):
            assert torch.isclose(B[i, j], torch.tensor(-0.5 * abs(i - j)), atol=1e-6)
```

- [ ] **Step 2:** Run — expect FAIL.

- [ ] **Step 3 (GREEN):** create `vfe3/attention_prior.py`. Each prior returns a **log-prior bias** `B_ij` added to the attention logits (`β* = softmax_j(B_ij − E_ij/τ)`); the normalized prior used in the entropy term is `π = softmax_j(B)`.

```python
r"""Attention-prior registry for VFE_3.0 (the prior pi_ij over keys).

Each prior returns a LOG-PRIOR BIAS B_ij added to the attention logits:
    beta*_ij = softmax_j(B_ij - E_ij / tau),
and the normalized prior used in the attention-entropy term is pi = softmax_j(B).
  uniform  B = 0            -> pi_ij = 1/N (manuscript default).
  causal   B = 0 (j<=i), -inf (j>i)   -> uniform over the causal active set.
  alibi    B_ij = -slope*|i-j|        -> linear distance bias (Press et al.).
Config-selected so a new prior (learned bias, windowed, ...) slots in by
register_prior without editing the free-energy call site.
"""

from typing import Callable, Dict, Optional

import torch

_PRIORS: Dict[str, Callable] = {}


def register_prior(name: str) -> Callable:
    """Decorator registering an attention-prior builder -> log-prior bias (Nq, Nk)."""
    def _wrap(fn: Callable) -> Callable:
        _PRIORS[name] = fn
        return fn
    return _wrap


def get_prior(name: str) -> Callable:
    """Return the registered attention-prior builder (KeyError if absent)."""
    if name not in _PRIORS:
        raise KeyError(f"no attention prior {name!r}; available: {sorted(_PRIORS)}")
    return _PRIORS[name]


@register_prior("uniform")
def prior_uniform(
    n_query: int,
    n_key:   int,

    *,
    device:  'torch.device | str | None' = None,
    dtype:   torch.dtype                  = torch.float32,
    **kwargs,
) -> torch.Tensor:
    r"""Uniform prior: zero log-bias (pi_ij = 1/N after softmax)."""
    return torch.zeros(n_query, n_key, device=device, dtype=dtype)


@register_prior("causal")
def prior_causal(
    n_query: int,
    n_key:   int,

    *,
    device:  'torch.device | str | None' = None,
    dtype:   torch.dtype                  = torch.float32,
    **kwargs,
) -> torch.Tensor:
    r"""Causal prior: 0 for key j <= query i, -inf for j > i."""
    i = torch.arange(n_query, device=device).unsqueeze(-1)
    j = torch.arange(n_key, device=device).unsqueeze(0)
    allowed = j <= i
    B = torch.zeros(n_query, n_key, device=device, dtype=dtype)
    return B.masked_fill(~allowed, float("-inf"))


@register_prior("alibi")
def prior_alibi(
    n_query: int,
    n_key:   int,

    *,
    slope:   float                        = 1.0,
    device:  'torch.device | str | None'  = None,
    dtype:   torch.dtype                   = torch.float32,
    **kwargs,
) -> torch.Tensor:
    r"""ALiBi prior: B_ij = -slope * |i - j| (linear distance bias)."""
    i = torch.arange(n_query, device=device).unsqueeze(-1)
    j = torch.arange(n_key, device=device).unsqueeze(0)
    return (-slope * (i - j).abs()).to(dtype)


def attention_log_prior(
    name:    str,
    n_query: int,
    n_key:   int,

    *,
    slope:   float                        = 1.0,
    device:  'torch.device | str | None'  = None,
    dtype:   torch.dtype                   = torch.float32,
) -> torch.Tensor:
    r"""Dispatch to the registered attention-prior `name`; returns log-prior bias (Nq, Nk)."""
    return get_prior(name)(n_query, n_key, slope=slope, device=device, dtype=dtype)
```

- [ ] **Step 4:** Run — expect 3 passed.
- [ ] **Step 5 (COMMIT):**
```
git add vfe3/attention_prior.py tests/test_attention_prior.py
git commit -m "feat(free-energy): attention-prior registry (uniform / causal / alibi log-bias)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3 — `free_energy.py`: β*, log-partition, envelope

**Files:** Create `vfe3/free_energy.py`; Test `tests/test_free_energy.py`.

- [ ] **Step 1 (RED):** create `tests/test_free_energy.py`. **Use a non-uniform prior** throughout (uniform π cancels everywhere and hides π-wiring / `τ log N` bugs):

```python
import math

import torch

from vfe3.free_energy import (
    attention_weights,
    effective_temperature,
    log_partition,
    reduced_free_energy,
)

# A concrete non-uniform setup reused across tests.
_E   = torch.tensor([1.0, 2.0, 0.5])               # distinct per-key energies
_PI  = torch.tensor([0.5, 0.3, 0.2])               # normalized non-uniform prior
_B   = torch.log(_PI)                              # log-prior bias
_TAU = 2.0


def test_temperature_is_kappa_sqrt_k():
    assert math.isclose(effective_temperature(1.5, 16), 1.5 * 4.0, rel_tol=1e-6)


def test_beta_is_softmax_logprior_minus_energy_over_tau():
    beta = attention_weights(_E, log_prior=_B, tau=_TAU)
    logits = _B - _E / _TAU
    expect = torch.softmax(logits, dim=-1)
    assert torch.allclose(beta, expect, atol=1e-6)
    assert torch.allclose(beta.sum(-1), torch.tensor(1.0), atol=1e-6)


def test_envelope_identity_canonical_block_equals_neg_tau_logZ():
    # Sum_j beta* E + tau Sum_j beta* log(beta*/pi) == -tau log Z, with non-uniform pi.
    beta = attention_weights(_E, log_prior=_B, tau=_TAU)
    pi = torch.softmax(_B, dim=-1)
    canon_block = (beta * _E).sum(-1) + _TAU * (beta * (torch.log(beta) - torch.log(pi))).sum(-1)
    fred = reduced_free_energy(_E, log_prior=_B, tau=_TAU)        # -tau log Z
    assert torch.allclose(canon_block, fred, atol=1e-5)
    # hand-computed literal backstop (catches a tau*log N offset):
    assert torch.allclose(fred, torch.tensor(1.1264), atol=1e-3)


def test_stationarity_residual_constant_across_keys():
    # At beta*, E_j + tau log(beta*_j/pi_j) is the SAME for every key j (= -tau log Z).
    beta = attention_weights(_E, log_prior=_B, tau=_TAU)
    pi = torch.softmax(_B, dim=-1)
    residual = _E + _TAU * (torch.log(beta) - torch.log(pi))
    assert (residual.max() - residual.min()).abs() < 1e-5
    assert torch.allclose(residual.mean(), reduced_free_energy(_E, log_prior=_B, tau=_TAU), atol=1e-5)
```

- [ ] **Step 2:** Run — expect FAIL.

- [ ] **Step 3 (GREEN):** create `vfe3/free_energy.py` (this task: the β/partition layer; the scalar F is Task 4):

```python
r"""The single authoritative scalar free energy F = sum_i F_i for VFE_3.0.

F is divergence-agnostic: it assembles the scalar from per-pair energies E_ij and
self-divergences D(q_i||p_i) supplied by the `divergence` registry, so a new
divergence slots in by registration + config, never by editing F. Canonical (with
the attention-entropy term) vs surrogate is a single toggle. The attention prior
is a log-bias B_ij from the `attention_prior` seam; beta* = softmax_j(B - E/tau).
"""

from typing import Optional

import torch

from vfe3.divergence import renyi


def effective_temperature(
    kappa: 'float | torch.Tensor',        # learnable sharpness scalar
    K:     int,                            # belief dimension
) -> 'float | torch.Tensor':
    r"""Softmax temperature tau = kappa * sqrt(K) (standard transformer is kappa=1)."""
    return kappa * (K ** 0.5)


def pairwise_energy(
    mu_q:    torch.Tensor,                 # (..., N, K) query means
    sigma_q: torch.Tensor,                 # (..., N, K[,K]) query (co)variances
    mu_t:    torch.Tensor,                 # (..., N, N, K) transported key means Omega_ij mu_j
    sigma_t: torch.Tensor,                 # (..., N, N, K[,K]) transported key (co)variances

    *,
    alpha:   float = 1.0,
    kl_max:  float = 100.0,
    eps:     float = 1e-6,
    family:  str   = "gaussian_diagonal",
) -> torch.Tensor:                         # (..., N, N) E_ij = D(q_i || Omega_ij q_j)
    r"""Per-pair belief-coupling energy via the divergence seam (KL = Renyi at alpha=1).

    Divergence-agnostic: swapping `family`/`alpha` (or registering a new kernel)
    changes the energy without touching the free-energy assembly.
    """
    mu_q_b = mu_q.unsqueeze(-2)            # (..., N, 1, K) broadcast query over keys
    sigma_q_b = sigma_q.unsqueeze(-2) if sigma_q.dim() == mu_q.dim() else sigma_q.unsqueeze(-3)
    return renyi(mu_q_b, sigma_q_b, mu_t, sigma_t, alpha=alpha, kl_max=kl_max, eps=eps, family=family)


def self_divergence(
    mu_q:    torch.Tensor,                 # (..., N, K)
    sigma_q: torch.Tensor,                 # (..., N, K[,K])
    mu_p:    torch.Tensor,                 # (..., N, K)
    sigma_p: torch.Tensor,                 # (..., N, K[,K])

    *,
    alpha:   float = 1.0,
    kl_max:  float = 100.0,
    eps:     float = 1e-6,
    family:  str   = "gaussian_diagonal",
) -> torch.Tensor:                         # (..., N) D(q_i || p_i)
    r"""Self-coupling divergence via the seam."""
    return renyi(mu_q, sigma_q, mu_p, sigma_p, alpha=alpha, kl_max=kl_max, eps=eps, family=family)


def attention_weights(
    energy:    torch.Tensor,               # (..., N) or (..., N, N) per-key energies E_ij

    *,
    tau:       float = 1.0,
    log_prior: Optional[torch.Tensor] = None,   # (..., N/NxN) bias B_ij; None -> 0
) -> torch.Tensor:                         # (...) softmax_j(B - E/tau)
    r"""Attention weights beta*_ij = softmax_j(B_ij - E_ij / tau)."""
    logits = -energy / tau
    if log_prior is not None:
        logits = logits + log_prior
    return torch.softmax(logits, dim=-1)


def log_partition(
    energy:    torch.Tensor,               # (..., N) or (..., N, N)

    *,
    tau:       float = 1.0,
    log_prior: Optional[torch.Tensor] = None,
) -> torch.Tensor:                         # (...) log Z_i = logsumexp_j(B - E/tau)
    r"""Log-partition log Z_i = logsumexp_j(B_ij - E_ij / tau)."""
    logits = -energy / tau
    if log_prior is not None:
        logits = logits + log_prior
    return torch.logsumexp(logits, dim=-1)


def reduced_free_energy(
    energy:    torch.Tensor,               # (..., N) or (..., N, N)

    *,
    tau:       float = 1.0,
    log_prior: Optional[torch.Tensor] = None,
) -> torch.Tensor:                         # (...) F_red,i = -tau log Z_i
    r"""Reduced (envelope) free energy F_red,i = -tau log Z_i; equals the canonical
    beta-block evaluated at beta*."""
    return -tau * log_partition(energy, tau=tau, log_prior=log_prior)
```

- [ ] **Step 4:** Run — expect 4 passed (the `1.1264` literal confirms the non-uniform Z, not a `τ log N`-shifted one).
- [ ] **Step 5 (COMMIT):**
```
git add vfe3/free_energy.py tests/test_free_energy.py
git commit -m "feat(free-energy): attention weights, log-partition, envelope (non-uniform prior pinned)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4 — `free_energy.py`: the scalar F (canonical / surrogate)

**Files:** Modify `vfe3/free_energy.py`; Test `tests/test_free_energy.py`.

- [ ] **Step 1 (RED):** append tests:

```python
from vfe3.free_energy import free_energy


def test_canonical_minus_surrogate_is_tau_times_entropy():
    # Canonical F - surrogate F = tau * Sum_i Sum_j beta* log(beta*/pi)  (the entropy block).
    N = 3
    self_div = torch.zeros(N)                            # alpha term zero (isolate beta block)
    energy = torch.tensor([[1.0, 2.0, 0.5],
                           [0.7, 0.3, 1.1],
                           [1.2, 0.9, 0.4]])
    B = torch.log(torch.tensor([0.5, 0.3, 0.2]))
    log_prior = B.expand(N, N)
    alpha = torch.zeros(N)
    fe_canon = free_energy(self_div, energy, alpha, log_prior=log_prior, tau=2.0,
                           include_attention_entropy=True)
    fe_surr  = free_energy(self_div, energy, alpha, log_prior=log_prior, tau=2.0,
                           include_attention_entropy=False)
    beta = attention_weights(energy, log_prior=log_prior, tau=2.0)
    pi = torch.softmax(log_prior, dim=-1)
    entropy_block = 2.0 * (beta * (torch.log(beta) - torch.log(pi))).sum()
    assert torch.allclose(fe_canon - fe_surr, entropy_block, atol=1e-5)


def test_known_value_F_self_coupling_only():
    # q == p -> self_div == 0; energy all-equal + uniform prior -> beta uniform.
    # With alpha=2, self_div=[0.5,1.0], no entropy (surrogate), energy uniform=c:
    # F = sum_i alpha_i*self_div_i + sum_ij beta_ij*c. beta uniform=1/N so sum_j beta*c=c.
    self_div = torch.tensor([0.5, 1.0])
    energy = torch.full((2, 2), 0.3)
    alpha = torch.full((2,), 2.0)
    fe = free_energy(self_div, energy, alpha, log_prior=None, tau=1.0,
                     include_attention_entropy=False)
    expect = (2.0 * 0.5 + 2.0 * 1.0) + (0.3 + 0.3)
    assert torch.allclose(fe, torch.tensor(expect), atol=1e-5)


def test_autograd_F_matches_finite_difference():
    torch.manual_seed(0)
    N, K = 3, 4
    mu_q = torch.randn(N, K, requires_grad=True)
    base = {"sigma_q": torch.rand(N, K) + 0.5, "mu_p": torch.randn(N, K),
            "sigma_p": torch.rand(N, K) + 0.5}
    from vfe3.free_energy import self_divergence

    def scalar(mu):
        sd = self_divergence(mu, base["sigma_q"], base["mu_p"], base["sigma_p"])
        energy = torch.cdist(mu, mu) ** 2 + 0.1           # a smooth differentiable (N,N) energy
        alpha = torch.ones(N)
        return free_energy(sd, energy, alpha, log_prior=None, tau=1.5,
                           include_attention_entropy=True)

    F = scalar(mu_q); F.backward()
    g_auto = mu_q.grad.clone()
    eps = 1e-3
    g_fd = torch.zeros_like(mu_q)
    with torch.no_grad():
        for a in range(N):
            for b in range(K):
                d = torch.zeros(N, K); d[a, b] = eps
                g_fd[a, b] = (scalar(mu_q + d) - scalar(mu_q - d)) / (2 * eps)
    assert torch.allclose(g_auto, g_fd, atol=1e-3, rtol=1e-3)
```

- [ ] **Step 2:** Run — expect FAIL (ImportError: free_energy).

- [ ] **Step 3 (GREEN):** append to `vfe3/free_energy.py`:

```python
def free_energy(
    self_div:                  torch.Tensor,        # (..., N) or (..., N, K) D(q_i||p_i)
    energy:                    torch.Tensor,        # (..., N, N) E_ij belief-coupling energies
    alpha:                     torch.Tensor,        # (..., N) or (..., N, K) self-coupling

    *,
    tau:                       float = 1.0,
    include_attention_entropy: bool  = True,

    log_prior:                 Optional[torch.Tensor] = None,   # (..., N, N) attention log-prior
    alpha_reg:                 Optional[torch.Tensor] = None,   # (..., N[,K]) R(alpha) if state-dep
    log_likelihood:            Optional[torch.Tensor] = None,   # (..., N) E_q[log p(o|k)]
) -> torch.Tensor:                                  # scalar F = sum_i F_i
    r"""Single authoritative scalar free energy (default path; lambda_h=0, gamma=0).

        F = sum_i [ alpha_i . D(q_i||p_i)            (+ R(alpha_i) if state-dependent)
                  + sum_j beta_ij E_ij
                  + tau sum_j beta_ij log(beta_ij/pi_ij)     (canonical only)
                  - ell_i ]
    beta_ij = softmax_j(log_prior - E/tau); pi = softmax_j(log_prior). The hyper-prior
    lambda_h KL(s||h) and model-coupling gamma KL(s_i||Omega s_j) are extension points,
    absent from this default path. Divergence-agnostic: `self_div`/`energy` come from the
    divergence seam, so a new divergence requires no change here.
    """
    beta = attention_weights(energy, tau=tau, log_prior=log_prior)        # (..., N, N)

    # self-coupling (sum over coordinate axis too when alpha/self_div are per-coord)
    self_term = alpha * self_div
    if alpha_reg is not None:
        self_term = self_term + alpha_reg
    self_total = self_term.sum()

    coupling = (beta * energy).sum()

    F = self_total + coupling
    if include_attention_entropy:
        pi = torch.softmax(log_prior, dim=-1) if log_prior is not None \
            else torch.full_like(beta, 1.0 / beta.shape[-1])
        entropy = tau * (beta * (torch.log(beta.clamp(min=1e-12)) - torch.log(pi.clamp(min=1e-12)))).sum()
        F = F + entropy
    if log_likelihood is not None:
        F = F - log_likelihood.sum()
    return F
```

- [ ] **Step 4:** Run — expect 3 passed.
- [ ] **Step 5 (COMMIT):**
```
git add vfe3/free_energy.py tests/test_free_energy.py
git commit -m "feat(free-energy): scalar F = sum_i F_i, canonical/surrogate toggle, likelihood optional

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5 — the gradient gap + α-envelope (crown jewels)

**Files:** Test `tests/test_free_energy.py`.

- [ ] **Step 1 (RED):** append. **β\* must be a LIVE function of x** — never detached — or the test is vacuous:

```python
def test_gradient_gap_canonical_minus_surrogate_is_neg_cov_over_tau():
    # The envelope theorem: with beta* a LIVE function of x, autograd of the canonical
    # beta-block collapses to the envelope Sum_j beta* dE; the surrogate keeps the
    # dbeta term, and (surrogate - canonical) gradients = -tau^{-1} Cov_beta*(E, dE).
    torch.manual_seed(1)
    tau = 1.5
    x = torch.randn(4, requires_grad=True)               # a differentiable parameter
    A = torch.randn(3, 4)
    log_prior = torch.log(torch.tensor([0.5, 0.3, 0.2]))  # non-uniform

    def energy_of(x_):                                   # (3,) energies, differentiable in x
        return (A @ x_) ** 2 + 0.2

    def canonical_block(x_):
        E = energy_of(x_)
        beta = torch.softmax(log_prior - E / tau, dim=-1)
        pi = torch.softmax(log_prior, dim=-1)
        return (beta * E).sum() + tau * (beta * (torch.log(beta) - torch.log(pi))).sum()

    def surrogate_block(x_):
        E = energy_of(x_)
        beta = torch.softmax(log_prior - E / tau, dim=-1)
        return (beta * E).sum()

    gc = torch.autograd.grad(canonical_block(x), x)[0]
    gs = torch.autograd.grad(surrogate_block(x), x)[0]

    # envelope: canonical grad == Sum_j beta* dE_j  (beta* detached here on purpose)
    E = energy_of(x)
    beta = torch.softmax(log_prior - E / tau, dim=-1).detach()
    JE = torch.autograd.functional.jacobian(energy_of, x)     # (3,4) dE_j/dx
    env = (beta.unsqueeze(-1) * JE).sum(0)
    assert torch.allclose(gc, env, atol=1e-4)

    # gap == -tau^{-1} Cov_beta*(E, dE)
    Edet = E.detach()
    mean_E  = (beta * Edet).sum()
    mean_J  = (beta.unsqueeze(-1) * JE).sum(0)                 # (4,)
    cross   = (beta.unsqueeze(-1) * Edet.unsqueeze(-1) * JE).sum(0)   # (4,)
    cov = cross - mean_E * mean_J
    assert torch.allclose(gs - gc, -cov / tau, atol=1e-4)


def test_alpha_envelope_grad_q_F_equals_alpha_star_times_grad_q_D():
    # State-dependent alpha*: at alpha*, dF/dalpha = 0, so d/dq [alpha*(D)*D + R(alpha*(D))]
    # == alpha* * dD/dq (the explicit alpha-path vanishes). De-risks Phase 4.
    from vfe3.alpha_i import self_coupling_alpha
    from vfe3.free_energy import self_divergence

    b0, c0 = 0.5, 2.0
    mu_q = torch.randn(2, 3, requires_grad=True)
    sigma_q = torch.rand(2, 3) + 0.5
    mu_p = torch.randn(2, 3); sigma_p = torch.rand(2, 3) + 0.5

    def adaptive_self(mu):
        D = self_divergence(mu, sigma_q, mu_p, sigma_p)       # (2,)
        a, r = self_coupling_alpha(D, mode="state_dependent", b0=b0, c0=c0)
        return (a * D + r).sum()

    g_full = torch.autograd.grad(adaptive_self(mu_q), mu_q)[0]
    # envelope RHS: alpha*(D) detached, times dD/dq
    D = self_divergence(mu_q, sigma_q, mu_p, sigma_p)
    a_star = (c0 / (b0 + D)).detach()
    g_env = torch.autograd.grad((a_star * D).sum(), mu_q)[0]
    assert torch.allclose(g_full, g_env, atol=1e-5)
```

- [ ] **Step 2:** Run — expect FAIL then implement nothing new (these exercise existing code); they should pass once written correctly. If `test_gradient_gap...` fails, the sign or the live-β wiring is wrong — the covariance identity is authoritative.

- [ ] **Step 3:** Run — expect 2 passed.
- [ ] **Step 4 (COMMIT):**
```
git add tests/test_free_energy.py
git commit -m "test(free-energy): envelope gradient gap (-cov/tau) + alpha-envelope, live beta

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6 — full suite + changelog + final

- [ ] **Step 1:** `python -m pytest -q` — expect all prior (82) + new (~16) green, no regressions.
- [ ] **Step 2:** Append a "## Phase 3 Free Energy + alpha + attention prior — 2026-05-29 (continuation)" section to `docs/edits/2026-05-29-phase2b-transport.md` (match the existing format; provenance-clean). Commit as `docs(edits): 2026-05-29 phase 3 free energy changes log`.
- [ ] **Step 3:** If Task-by-task commits left nothing else, done; otherwise `git add -A` your phase files (never the user's CLAUDE.md / untracked dirs) and commit.

---

## Self-Review

**Spec coverage (Phase 3, §4.3 + §7):**
- Single scalar F = Σ_i F_i, canonical/surrogate by one toggle → Task 4.
- α_i self-coupling forms + regularizer → Task 1.
- Attention prior seam (uniform/causal/alibi) → Task 2.
- β*, log-partition, reduced/envelope free energy → Task 3.
- Envelope identity, β-stationarity, gradient gap, α-envelope, autograd-vs-FD → Tasks 3–5.

**Modularity (registry behind every seam):** divergence (existing, family/alpha config-selected; F divergence-agnostic), self-coupling (`alpha_i`), attention prior (`attention_prior`). A new divergence / α form / attention prior slots in by `@register_*` + config; `free_energy` is never edited.

**Hand-derived anchors (independent of the implementation):**
- β* = softmax(B − E/τ); envelope `canonical β-block = −τ log Z` (+ literal `1.1264` for E=[1,2,0.5], π=[.5,.3,.2], τ=2).
- Stationarity: `E_j + τ log(β*_j/π_j)` constant across j.
- Gradient gap: `∇(surrogate) − ∇(canonical) = −τ⁻¹ Cov_β*(E, ∂E)` (sign pinned; β* live).
- α* = c₀/(b₀+D) minimizes αD+R; α-envelope `∇_q F = α*·∂_q D`.
- All property tests use a **non-uniform** prior.

**Deferred (named extension points):** hyper-prior `λ_h KL(s‖h)`; model-coupling `γ KL(s_i‖Ω s_j)`; learned/windowed attention priors; the Gaussian-template observation likelihood (Phase 7 decode); learnable κ as a trained parameter (Phase 6/7).

**Type/name consistency:** `self_coupling_alpha(kl, *, value, b0, c0, mode)`, `attention_log_prior(name, n_query, n_key, *, slope, device, dtype)`, `attention_weights(energy, *, tau, log_prior)`, `reduced_free_energy(energy, *, tau, log_prior)`, `free_energy(self_div, energy, alpha, *, tau, include_attention_entropy, log_prior, alpha_reg, log_likelihood)` — consistent across tasks and tests. F is coords/divergence-agnostic; β passed explicitly; β* live in the gap test.
