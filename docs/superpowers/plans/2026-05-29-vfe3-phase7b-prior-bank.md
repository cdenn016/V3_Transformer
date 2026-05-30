# VFE_3.0 Phase 7b (PriorBank encode/decode + MahalanobisNorm) Implementation Plan

> REQUIRED SUB-SKILL: superpowers:test-driven-development (RED→GREEN→COMMIT per task, **commit after every GREEN task** so a crash leaves a recoverable partial). V3-internal tests. No VFE_2.0 provenance.

**Goal:** Build the encode/decode boundary — `vfe3/model/prior_bank.py` (learnable Gaussian vocab priors; `encode` tokens→belief; `decode` belief→logits as `−KL(q‖π_v)/τ`) and `vfe3/geometry/norms.py` (gauge-equivariant MahalanobisNorm). The decode is the load-bearing kernel: it is **divergence-agnostic** (scores via the `divergence` seam) and pinned both exactly (`logit_v == −kl_seam(q,π_v)/τ`) and shift-invariantly (`log_softmax`).

**Architecture:** Phase 7b (spec §4.6). The PriorBank holds the learnable priors `π_v = N(μ_v, Σ_v)` (+ gauge frame `φ_v`) as **parameter tables** (`nn.Parameter` — these are *priors*, not neural maps; the V3 no-NN rule bans `nn.Linear`/MLP/activations, not learnable parameters). `encode(token_ids)` looks them up into a `BeliefState` (the initial belief `q=p`). `decode(μ_q, σ_q)` scores the posterior against every vocab prior → logits. The decode **replaces** `output_proj=nn.Linear`. MahalanobisNorm is pure math (no params).

**Decode kernel (THE critical piece):** for posterior `q=N(μ_q,σ_q)` and vocab prior `π_v=N(μ_v,σ_v)` (diagonal),
```
logit_{i,v} = -KL(q_i || pi_v) / tau_eff
KL(q||pi_v) = 0.5 [ sum_k( sigma_q_k/sigma_v_k + (mu_q_k - mu_v_k)^2/sigma_v_k ) - K + sum_k log(sigma_v_k/sigma_q_k) ]
tau_eff = decode_tau * exp(-clamp(decode_log_scale, -3, 3))   # learnable scale
```
The `−K` and `−Σ_k log σ_q,k` terms are v-independent (per-position constant) and drop under softmax/CE; the EXACT default keeps them (one cheap per-position sum) so `logit_v == −KL/τ` holds exactly. The v-dependent part uses a single fused matmul: `combined = [σ_q+μ_q⊙μ_q, −2μ_q] @ [1/σ_v, μ_v/σ_v]^T`, plus the v-only bias `Σ_k(μ_v²/σ_v + log σ_v)`.

**Tech Stack:** Python 3, PyTorch (float32; nn.Parameter), pytest. No NN layers (no Linear/MLP/activations). No CLI. Device-agnostic.

**Reference spec:** §4.6. Prereq: Phases 0–7a on `main` (branch `phase7-model`). Reuses `vfe3.divergence` (`kl`), `vfe3.belief.BeliefState`, `vfe3.config.VFE3Config`.

**Design decisions (do not relitigate):**
1. **Decode is divergence-agnostic + double-pinned.** A reference decode broadcasts `divergence.kl` over V (general, slow, O(B·N·V·K)); the fused diagonal decode is pinned to it **exactly** (`logit == −kl/τ`, the per-position term kept) AND under `log_softmax` (shift-invariant — catches a dropped-term or `τ`-scale bug). A `decode_mode` registry: `diagonal` (fused, default) — `full` (exact Cholesky) is a named stub.
2. **No-NN:** prior tables are `nn.Parameter`; PriorBank is an `nn.Module` *parameter container* with a pure-VFE forward; zero `nn.Linear`/MLP/activations. Document each table as a prior, not a neural map.
3. **encode_mode registry:** `per_token` (lookup `μ_embed`,`σ_log_embed`,`φ_embed`) default; `gauge_fixed` (gauge orbit from a shared base) a named stub.
4. Batched `(B, N)` tokens → `(B, N, K)` beliefs / `(B, N, V)` logits.

---

## Code Style / Provenance (MANDATORY)

Repo CLAUDE.md conventions (arg ordering, vertical alignment, type hints, math docstrings, shape comments). No shipped artifact may contain "VFE_2.0", "2.0", or "ported"; cite the manuscript + math.

---

## File Structure

- **Create** `vfe3/model/__init__.py` (empty), `vfe3/model/prior_bank.py` — `PriorBank` (encode, decode, reference_decode), `register_decode`/`register_encode` registries.
- **Create** `vfe3/geometry/norms.py` — `MahalanobisNorm`, `register_norm`/`get_norm`.
- **Create** `tests/test_prior_bank.py`, `tests/test_norms.py`.

---

## Task 1 — `MahalanobisNorm`

**Files:** `vfe3/geometry/norms.py`; `tests/test_norms.py`.

- [ ] **RED:** `tests/test_norms.py`:

```python
import torch
from vfe3.geometry.norms import MahalanobisNorm


def test_mahalanobis_formula_diagonal():
    K = 4
    norm = MahalanobisNorm(K)
    mu = torch.randn(3, K); sigma = torch.rand(3, K) + 0.5
    out = norm(mu, sigma)
    s2 = (mu ** 2 / sigma).sum(-1, keepdim=True)
    assert torch.allclose(out, mu * torch.sqrt(K / s2), atol=1e-5)


def test_mahalanobis_is_gauge_invariant_scale():
    # The Mahalanobis scalar mu^T Sigma^-1 mu is invariant under mu->g mu, Sigma->g Sigma g^T,
    # so the norm SCALE sqrt(K/s2) is gauge-invariant; out transforms as a vector (out -> g out).
    K = 3
    norm = MahalanobisNorm(K)
    g = torch.randn(K, K); g = g + 2 * torch.eye(K)              # invertible
    mu = torch.randn(2, K); sigma_full = torch.eye(K).expand(2, K, K).contiguous()
    out = norm(mu, sigma_full)
    mu_g = mu @ g.T
    sig_g = g @ sigma_full @ g.T
    out_g = norm(mu_g, sig_g)
    assert torch.allclose(out_g, out @ g.T, atol=1e-4)
```

- [ ] **GREEN:** `vfe3/geometry/norms.py`:

```python
r"""Gauge-equivariant normalization for VFE_3.0 belief means.

MahalanobisNorm rescales mu by the gauge-invariant Mahalanobis length:
    mu_norm = mu * sqrt(K / (mu^T Sigma^-1 mu + eps)).
Since mu^T Sigma^-1 mu is invariant under mu->g mu, Sigma->g Sigma g^T, the scale is
gauge-invariant and mu_norm transforms as a vector. Pure math, no parameters.
"""

from typing import Callable, Dict

import torch

_NORMS: Dict[str, Callable] = {}


def register_norm(name: str) -> Callable:
    """Decorator registering a norm builder under ``name``."""
    def _wrap(fn: Callable) -> Callable:
        _NORMS[name] = fn
        return fn
    return _wrap


def get_norm(name: str) -> Callable:
    """Return the registered norm builder (KeyError if absent)."""
    if name not in _NORMS:
        raise KeyError(f"no norm registered under {name!r}; available: {sorted(_NORMS)}")
    return _NORMS[name]


class MahalanobisNorm:
    """mu_norm = mu * sqrt(K / (mu^T Sigma^-1 mu + eps))."""

    def __init__(self, K: int, *, eps: float = 1e-6) -> None:
        self.K = K
        self.eps = eps

    def __call__(
        self,
        mu:    torch.Tensor,             # (..., K) means
        sigma: torch.Tensor,             # (..., K) diagonal OR (..., K, K) full
    ) -> torch.Tensor:                   # (..., K) rescaled means
        if sigma.dim() == mu.dim():
            s2 = (mu ** 2 / sigma.clamp(min=self.eps)).sum(dim=-1, keepdim=True)
        else:
            sig_inv_mu = torch.linalg.solve(sigma, mu.unsqueeze(-1)).squeeze(-1)
            s2 = (mu * sig_inv_mu).sum(dim=-1, keepdim=True)
        return mu * torch.sqrt(self.K / s2.clamp(min=self.eps))


@register_norm("none")
def _norm_none(K: int, **kwargs):
    """Identity norm."""
    return lambda mu, sigma: mu


@register_norm("mahalanobis")
def _norm_mahalanobis(K: int, *, eps: float = 1e-6, **kwargs):
    """MahalanobisNorm builder."""
    return MahalanobisNorm(K, eps=eps)
```

- [ ] Run → 2 passed. **COMMIT** `feat(model): gauge-equivariant MahalanobisNorm + norm registry`.

---

## Task 2 — `PriorBank.encode` + the prior tables

**Files:** `vfe3/model/__init__.py`, `vfe3/model/prior_bank.py`; `tests/test_prior_bank.py`.

- [ ] **RED:** `tests/test_prior_bank.py`:

```python
import torch
from vfe3.belief import BeliefState
from vfe3.model.prior_bank import PriorBank


def test_encode_shapes_and_positive_sigma():
    V, K, n_gen = 20, 4, 16
    pb = PriorBank(V, K, n_gen)
    tokens = torch.randint(0, V, (2, 5))
    b = pb.encode(tokens)
    assert isinstance(b, BeliefState)
    assert b.mu.shape == (2, 5, K) and b.sigma.shape == (2, 5, K) and b.phi.shape == (2, 5, n_gen)
    assert (b.sigma > 0).all()


def test_encode_is_a_lookup():
    V, K, n_gen = 6, 3, 9
    pb = PriorBank(V, K, n_gen)
    b = pb.encode(torch.tensor([[0, 0]]))
    assert torch.allclose(b.mu[0, 0], b.mu[0, 1])             # same token -> same prior
```

- [ ] **GREEN:** `vfe3/model/prior_bank.py` (encode part). Prior tables as `nn.Parameter`; `sigma = exp(sigma_log)` for positivity:

```python
r"""PriorBank for VFE_3.0: learnable Gaussian vocab priors + the KL decode boundary.

Holds the per-vocabulary prior pi_v = N(mu_v, Sigma_v) with gauge frame phi_v as
PARAMETER TABLES (nn.Parameter -- priors, not neural maps; the no-NN rule bans
nn.Linear/MLP/activations, not learnable parameters). encode(token_ids) looks them
up into the initial belief (q = p); decode(mu_q, sigma_q) scores the posterior
against every prior as logits = -KL(q || pi_v)/tau (the divergence seam), replacing
a linear output projection.
"""

from typing import Callable, Dict

import torch
from torch import nn

from vfe3.belief import BeliefState
from vfe3.divergence import kl


class PriorBank(nn.Module):
    """Learnable Gaussian vocab priors; encode (lookup) and decode (-KL/tau)."""

    def __init__(
        self,
        vocab_size:  int,
        K:           int,
        n_gen:       int,

        *,
        mu_init_std:    float = 0.02,
        sigma_init:     float = 1.0,
        phi_scale:      float = 0.01,
        decode_tau:     float = 1.0,
        eps:            float = 1e-6,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.K = K
        self.decode_tau = decode_tau
        self.eps = eps

        self.mu_embed        = nn.Parameter(mu_init_std * torch.randn(vocab_size, K))
        self.sigma_log_embed = nn.Parameter(torch.full((vocab_size, K), float(torch.log(torch.tensor(sigma_init)))))
        self.phi_embed       = nn.Parameter(phi_scale * torch.randn(vocab_size, n_gen))
        self.decode_log_scale = nn.Parameter(torch.zeros(1))

    def encode(
        self,
        token_ids: torch.Tensor,         # (B, N) integer token ids
    ) -> BeliefState:
        r"""Look up the per-token Gaussian prior as the initial belief (q = p)."""
        mu = self.mu_embed[token_ids]                        # (B, N, K)
        sigma = torch.exp(self.sigma_log_embed[token_ids]).clamp(min=self.eps)
        phi = self.phi_embed[token_ids]                      # (B, N, n_gen)
        return BeliefState(mu=mu, sigma=sigma, phi=phi)
```

- [ ] Run → 2 passed. **COMMIT** `feat(model): PriorBank learnable vocab priors + encode lookup`.

---

## Task 3 — `PriorBank.decode` (−KL/τ) + the double pin

**Files:** modify `vfe3/model/prior_bank.py`; `tests/test_prior_bank.py`.

- [ ] **RED:** append. The reference decode broadcasts the divergence seam over V; the fused decode is pinned to it exactly AND under log-softmax:

```python
import torch.nn.functional as F
from vfe3.divergence import kl as _kl


def _reference_decode(pb, mu_q, sigma_q, tau):
    # General reference: -KL(q_i || pi_v)/tau by broadcasting the divergence seam over V.
    V = pb.vocab_size
    mu_v = pb.mu_embed; sigma_v = torch.exp(pb.sigma_log_embed)
    mu_q_b = mu_q.unsqueeze(-2)                               # (B,N,1,K)
    sigma_q_b = sigma_q.unsqueeze(-2)
    klv = _kl(mu_q_b, sigma_q_b, mu_v, sigma_v)               # (B,N,V) via broadcast
    return -klv / tau


def test_decode_matches_divergence_seam_exactly():
    V, K, n_gen = 12, 4, 16
    pb = PriorBank(V, K, n_gen)
    mu_q = torch.randn(2, 3, K); sigma_q = torch.rand(2, 3, K) + 0.5
    logits = pb.decode(mu_q, sigma_q)
    ref = _reference_decode(pb, mu_q, sigma_q, pb.decode_tau)  # decode_log_scale=0 -> tau_eff=decode_tau
    assert torch.allclose(logits, ref, atol=1e-3)             # EXACT -KL/tau (per-position term kept)
    # shift-invariant pin (robust to a dropped-constant variant):
    assert torch.allclose(F.log_softmax(logits, dim=-1), F.log_softmax(ref, dim=-1), atol=1e-4)


def test_decode_tau_scaling():
    V, K = 10, 3
    pb = PriorBank(V, K, 9)
    mu_q = torch.randn(1, 2, K); sigma_q = torch.rand(1, 2, K) + 0.5
    l1 = pb.decode(mu_q, sigma_q, tau=1.0)
    l2 = pb.decode(mu_q, sigma_q, tau=2.0)
    assert torch.allclose(l1, 2.0 * l2, atol=1e-3)            # logits ~ 1/tau
```

- [ ] **GREEN:** append `decode` to `PriorBank`. Exact diagonal `−KL/τ` (the per-position `−K − Σ log σ_q` kept), v-dependent part via fused matmul:

```python
    def decode(
        self,
        mu_q:    torch.Tensor,           # (B, N, K) posterior means
        sigma_q: torch.Tensor,           # (B, N, K) posterior variances

        *,
        tau:     float = None,           # override decode_tau; None -> self.decode_tau
    ) -> torch.Tensor:                   # (B, N, V) logits = -KL(q || pi_v)/tau_eff
        r"""Decode logits_{i,v} = -KL(q_i || pi_v)/tau_eff, exact diagonal closed form.

            KL = 0.5[ sum_k(sigma_q/sigma_v + (mu_q-mu_v)^2/sigma_v) - K + sum_k log(sigma_v/sigma_q) ]
        v-dependent part A_v via a single fused matmul; the per-position (-K - sum log sigma_q)
        is kept so logits == -KL/tau_eff exactly (it is constant in v and drops under softmax).
        """
        base_tau = self.decode_tau if tau is None else tau
        tau_eff = base_tau * torch.exp(-self.decode_log_scale.clamp(-3.0, 3.0))

        sigma_v = torch.exp(self.sigma_log_embed).clamp(min=self.eps)     # (V, K)
        mu_v = self.mu_embed                                             # (V, K)
        inv_v = 1.0 / sigma_v                                            # (V, K)

        lhs = torch.cat([sigma_q + mu_q ** 2, -2.0 * mu_q], dim=-1)       # (B, N, 2K)
        rhs = torch.cat([inv_v, mu_v * inv_v], dim=-1)                    # (V, 2K)
        a_v = lhs @ rhs.transpose(-1, -2)                                 # (B, N, V): sum_k[(sigma_q+mu_q^2-2 mu_q mu_v)/sigma_v]
        a_v = a_v + (mu_v ** 2 * inv_v).sum(-1) + torch.log(sigma_v).sum(-1)   # + sum_k(mu_v^2/sigma_v + log sigma_v)
        # a_v now == sum_k(sigma_q/sigma_v + (mu_q-mu_v)^2/sigma_v) + sum_k log sigma_v = 2 KL + K + sum_k log sigma_q
        per_pos = self.K + torch.log(sigma_q.clamp(min=self.eps)).sum(-1, keepdim=True)   # (B,N,1)
        kl_v = 0.5 * (a_v - per_pos)
        return -kl_v / tau_eff
```

- [ ] Run → 2 passed. **COMMIT** `feat(model): PriorBank KL decode (-KL/tau), seam-exact + log-softmax pinned`.

---

## Task 4 — full suite + changelog + commit

- [ ] `python -m pytest -q` — expect prior 130 + new (~6) green, no regressions.
- [ ] Append "## Phase 7b PriorBank + MahalanobisNorm — 2026-05-29 (continuation)" to `docs/edits/2026-05-29-phase2b-transport.md` (provenance-clean). Commit `docs(edits): 2026-05-29 phase 7b prior-bank changes log`.

---

## Self-Review

**Coverage:** MahalanobisNorm (gauge-invariant scale) → T1; PriorBank encode (lookup, q=p) → T2; decode (−KL/τ exact + log-softmax pinned, divergence-agnostic) → T3.
**Anchors:** Mahalanobis scale gauge-invariant; decode `== −kl_seam/τ` exactly + `log_softmax` equal; `logits ∝ 1/τ`.
**Modularity:** decode/encode/norm registries; decode scores via the `divergence` seam (a new divergence needs no decode edit).
**No-NN:** prior tables are `nn.Parameter` (priors); zero `nn.Linear`/MLP/activations; decode replaces `output_proj`.
**Deferred (named):** `full` (Cholesky) decode; `gauge_fixed` encode; positional φ (BCH); the block/stack/model assembly (Phase 7c).
