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
    K:     int,                            # dimension the recovery is derived over (per-head d_k)
) -> 'float | torch.Tensor':
    r"""Softmax temperature tau = kappa * sqrt(K) (standard transformer is kappa=1).

    Generic primitive: pass the dimension over which standard scaled dot-product
    attention is recovered. The model passes the PER-HEAD dimension d_k = d_head
    (see VFE3Config.tau and audit finding 6c), so kappa=1 reproduces the Vaswani
    sqrt(d_k) temperature per head, not sqrt(K) over the full belief.
    """
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

    The key axis is inserted from the `family`'s covariance structure, not from a
    `sigma_q.dim() == mu_q.dim()` guess: a diagonal sigma_q is (..., N, K) and gets
    the key axis at -2 -> (..., N, 1, K); a full sigma_q is (..., N, K, K) and gets
    it at -3 -> (..., N, 1, K, K). This stays correct when sigma_q carries a leading
    batch dim that mu_q does not (the dim-equality heuristic misclassified that case).
    """
    is_diagonal = "diagonal" in family
    mu_q_b = mu_q.unsqueeze(-2)            # (..., N, 1, K) broadcast query over keys
    sigma_q_b = sigma_q.unsqueeze(-2) if is_diagonal else sigma_q.unsqueeze(-3)
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
) -> torch.Tensor:                         # (...) log Z_i = logsumexp_j(log pi - E/tau)
    r"""Log-partition log Z_i = logsumexp_j(log pi_ij - E_ij / tau), pi = softmax_j(B).

    The partition Z_i = Sum_j pi_ij exp(-E_ij/tau) is built from the NORMALIZED
    prior pi (not the raw log-bias B), so the envelope identity
    Sum_j beta*_ij E_ij + tau Sum_j beta*_ij log(beta*_ij/pi_ij) = -tau log Z_i
    holds for ANY prior the seam emits. Equivalently log Z = logsumexp(B - E/tau)
    - logsumexp(B); using log_softmax(B) subtracts that per-row normalizer in one
    step. With a None prior pi is uniform 1/N, so the bias is -log(N).
    """
    logits = -energy / tau
    if log_prior is not None:
        logits = logits + torch.log_softmax(log_prior, dim=-1)
    else:
        logits = logits - torch.log(torch.tensor(float(energy.shape[-1]),
                                                  device=energy.device, dtype=energy.dtype))
    return torch.logsumexp(logits, dim=-1)


def reduced_free_energy(
    energy:    torch.Tensor,               # (..., N) or (..., N, N)

    *,
    tau:       float = 1.0,
    log_prior: Optional[torch.Tensor] = None,
) -> torch.Tensor:                         # (...) F_red,i = -tau log Z_i
    r"""Reduced (envelope) free energy F_red,i = -tau log Z_i; equals the canonical
    beta-block evaluated at beta* for ANY prior (log_partition normalizes the
    prior internally, so the +tau logsumexp(B) per-row offset cannot leak in)."""
    return -tau * log_partition(energy, tau=tau, log_prior=log_prior)


def free_energy(
    self_div:                  torch.Tensor,        # (..., N) or (..., N, K) D(q_i||p_i)
    energy:                    torch.Tensor,        # (..., N, N) E_ij belief-coupling energies
    alpha:                     torch.Tensor,        # (..., N) or (..., N, K) self-coupling

    *,
    tau:                       float = 1.0,
    log_eps:                   float = 1e-12,                   # floor for log(beta)/log(pi) in the entropy term
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
        entropy = tau * (beta * (torch.log(beta.clamp(min=log_eps)) - torch.log(pi.clamp(min=log_eps)))).sum()
        F = F + entropy
    if log_likelihood is not None:
        F = F - log_likelihood.sum()
    return F
