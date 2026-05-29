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
