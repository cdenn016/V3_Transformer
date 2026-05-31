r"""Diagnostic metrics for VFE_3.0 runs (publication-oriented, registry-backed).

A registry of named metrics over beliefs / transport / attention. Each metric reads
what it needs from a keyword context and returns a plain float (or a small dict), so
new probes slot in by ``register_metric`` without editing call sites; ``compute_metrics``
emits a CSV/JSON-friendly record. The metrics are pure measurements (no gradients, no
side effects) suitable for logging a training run or a figure.

Provided: effective_rank (spectral participation ratio), attention_entropy (row entropy
of beta), holonomy_deviation (cocycle/curvature departure of the transport from flat),
gauge_trace_spread (spread of log|det Omega| = tr embed(phi)), and free_energy_terms
(the per-term F decomposition).
"""

import math
from typing import Callable, Dict, List, Optional, Tuple

import torch


def effective_rank(
    spectrum: torch.Tensor,              # (..., K) non-negative spectrum (diagonal variances or eigenvalues)

    *,
    eps:      float = 1e-12,
) -> torch.Tensor:                       # (...) participation-ratio effective rank
    r"""Spectral effective rank (sum lam)^2 / sum lam^2 over the last axis.

    Equals K when the spectrum is flat (all equal) and -> 1 when one mode dominates. For a
    full covariance, pass its eigenvalues (``torch.linalg.eigvalsh(Sigma)``); for a diagonal
    belief, the variances ARE the spectrum.
    """
    lam = spectrum.float().clamp(min=0.0)
    s1 = lam.sum(dim=-1)
    s2 = (lam ** 2).sum(dim=-1).clamp(min=eps)
    return (s1 * s1 / s2).to(spectrum.dtype)


def attention_entropy(
    beta: torch.Tensor,                  # (..., N, N) row-stochastic attention weights

    *,
    eps:  float = 1e-12,
) -> torch.Tensor:                       # () mean row entropy
    r"""Mean Shannon entropy -Σ_j β_ij log β_ij over query rows (log N for uniform β)."""
    h = -(beta.clamp(min=eps) * torch.log(beta.clamp(min=eps))).sum(dim=-1)
    return h.mean()


def holonomy_deviation(
    omega: torch.Tensor,                 # (N, N, K, K) pairwise transport Omega_ij

    *,
    max_triangles: int = 512,
) -> torch.Tensor:                       # () mean ||Omega_ij Omega_jk Omega_ki - I||_F
    r"""Curvature proxy: mean Frobenius departure of the triangle holonomy from identity.

    For a flat (Regime I) cocycle Omega_ij = exp(phi_i)exp(-phi_j) every triangle closes
    (H_ijk = I) so the deviation is ~0; a non-flat / non-cocycle transport gives > 0.
    """
    N, K = omega.shape[0], omega.shape[-1]
    eye = torch.eye(K, device=omega.device, dtype=omega.dtype)

    # Enumerate the first ``max_triangles`` distinct (i, j, k) triples in the same
    # row-major order the nested-loop form used, then evaluate them as ONE batched
    # (T, K, K) matmul rather than T Python-dispatched (K, K) matmuls -- same triangles,
    # same value, one kernel launch instead of T.
    triples: List[Tuple[int, int, int]] = []
    for i in range(N):
        for j in range(N):
            if j == i:
                continue
            for k in range(N):
                if k == i or k == j:
                    continue
                triples.append((i, j, k))
                if len(triples) >= max_triangles:
                    break
            if len(triples) >= max_triangles:
                break
        if len(triples) >= max_triangles:
            break
    if not triples:
        return torch.tensor(0.0, device=omega.device, dtype=omega.dtype)

    idx = torch.tensor(triples, device=omega.device)                      # (T, 3)
    o_ij = omega[idx[:, 0], idx[:, 1]]                                     # (T, K, K)
    o_jk = omega[idx[:, 1], idx[:, 2]]                                     # (T, K, K)
    o_ki = omega[idx[:, 2], idx[:, 0]]                                     # (T, K, K)
    H = o_ij @ o_jk @ o_ki                                                # (T, K, K) holonomy
    return torch.linalg.norm(H - eye, dim=(-2, -1)).mean()


def gauge_trace_spread(
    phi:        torch.Tensor,            # (..., n_gen) gauge-frame coordinates
    generators: torch.Tensor,           # (n_gen, K, K)
) -> torch.Tensor:                       # () std of tr(embed(phi)) = std of log|det exp(embed phi)|
    r"""Spread (std) of log|det Omega| across tokens; 0 at phi = 0 (Omega = I)."""
    traces = generators.diagonal(dim1=-2, dim2=-1).sum(-1)        # (n_gen,) tr(G_a)
    logdet = torch.einsum("...a,a->...", phi, traces)            # tr(embed(phi)) = sum_a phi^a tr(G_a)
    return logdet.flatten().std(unbiased=False)


def free_energy_terms(
    self_div: torch.Tensor,              # (..., N) D(q_i||p_i)
    energy:   torch.Tensor,              # (..., N, N) E_ij
    beta:     torch.Tensor,              # (..., N, N) attention weights
    alpha:    torch.Tensor,              # (..., N) self-coupling

    *,
    tau:       float = 1.0,
    log_prior: Optional[torch.Tensor] = None,
    eps:       float = 1e-12,
) -> Dict[str, float]:
    r"""Per-term free-energy decomposition: self-coupling, belief-coupling, attention entropy."""
    self_coupling = float((alpha * self_div).sum())
    belief_coupling = float((beta * energy).sum())
    pi = torch.softmax(log_prior, dim=-1) if log_prior is not None else torch.full_like(beta, 1.0 / beta.shape[-1])
    entropy = float(tau * (beta * (torch.log(beta.clamp(min=eps)) - torch.log(pi.clamp(min=eps)))).sum())
    return {
        "self_coupling":   self_coupling,
        "belief_coupling": belief_coupling,
        "attention_entropy": entropy,
        "total":           self_coupling + belief_coupling + entropy,
    }


# ---------------------------------------------------------------------------
# Registry: name -> metric(**context). New probes slot in by name.
# ---------------------------------------------------------------------------
_METRICS: Dict[str, Callable] = {}


def register_metric(name: str) -> Callable:
    """Decorator registering a metric that reads its inputs from the context kwargs."""
    def _wrap(fn: Callable) -> Callable:
        _METRICS[name] = fn
        return fn
    return _wrap


def get_metric(name: str) -> Callable:
    """Return the registered metric (KeyError if absent)."""
    if name not in _METRICS:
        raise KeyError(f"no metric {name!r}; available: {sorted(_METRICS)}")
    return _METRICS[name]


# Each metric's OWN context key is REQUIRED (no None default): a missing or mis-keyed
# context now raises TypeError at the call instead of an AttributeError deep inside the
# kernel (effective_rank(None) etc.). The trailing **kw stays only to absorb SIBLING
# metrics' context keys, since ``compute_metrics`` floods the full context to every metric.
@register_metric("effective_rank")
def _m_eff_rank(*, sigma: torch.Tensor, **kw) -> float:
    """Mean spectral effective rank of the belief covariances."""
    return float(effective_rank(sigma).mean())


@register_metric("attention_entropy")
def _m_attn_entropy(*, beta: torch.Tensor, **kw) -> float:
    """Mean attention row entropy."""
    return float(attention_entropy(beta))


@register_metric("holonomy_deviation")
def _m_holonomy(*, omega: torch.Tensor, **kw) -> float:
    """Mean triangle-holonomy departure from identity (curvature proxy)."""
    return float(holonomy_deviation(omega))


@register_metric("gauge_trace_spread")
def _m_gauge_spread(*, phi: torch.Tensor, generators: torch.Tensor, **kw) -> float:
    """Spread of log|det Omega| across tokens."""
    return float(gauge_trace_spread(phi, generators))


@register_metric("free_energy_terms")
def _m_free_energy_terms(*, self_div=None, energy=None, beta=None, alpha=None,
                         tau=1.0, log_prior=None, **kw) -> Dict[str, float]:
    """Per-term free-energy decomposition (self-coupling, belief-coupling, attention entropy)."""
    return free_energy_terms(self_div, energy, beta, alpha, tau=tau, log_prior=log_prior)


def compute_metrics(
    names: List[str],

    **context,
) -> Dict[str, float]:
    r"""Run the named metrics against the keyword ``context`` (sigma=, beta=, omega=, ...)."""
    return {n: get_metric(n)(**context) for n in names}
