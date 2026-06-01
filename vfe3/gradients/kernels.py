r"""Optimized hand-derived belief-gradient kernels for VFE_3.0, with oracle fallback.

A family-keyed registry of analytic (grad_mu, grad_sigma) kernels for the QUERY-SIDE
(filtering) gradient. belief_gradients() uses the registered kernel only for the
filtering + gaussian_diagonal + KL (alpha_div=1) + canonical case; every other case
(smoothing, non-KL family, Renyi alpha != 1, surrogate) FALLS BACK to the autograd
oracle -- so a new divergence works immediately and correctly, accelerated later by
registering a kernel. Kernels return RAW Euclidean dF (no preconditioning/retraction).
"""

from typing import Callable, Dict, List, Optional, Tuple

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


def _raw_diag_kl(
    mu_q:    torch.Tensor,             # (N, K) query means
    sigma_q: torch.Tensor,             # (N, K) query variances
    mu_p:    torch.Tensor,             # (N, K) prior means
    sigma_p: torch.Tensor,             # (N, K) prior variances

    *,
    eps:     float = 1e-6,
) -> torch.Tensor:                     # (N,) UNCLAMPED KL(q_i || p_i)
    r"""Unclamped diagonal KL(q||p) = 0.5 Sum_k (s_k/t_k + (mu_p-mu_q)^2/t_k - 1 + log(t_k/s_k)).

    The divergence seam returns the clamped value safe_kl_clamp(D, [0, kl_max]);
    this returns the raw D so the kernel can reproduce the oracle's saturation
    mask (the oracle differentiates THROUGH the clamp, whose gradient is 0 once
    D leaves (0, kl_max)).
    """
    sq = sigma_q.clamp(min=eps); sp = sigma_p.clamp(min=eps)
    trace  = (sq / sp).sum(dim=-1)
    mahal  = (((mu_p - mu_q) ** 2) / sp).sum(dim=-1)
    logdet = (torch.log(sp) - torch.log(sq)).sum(dim=-1)
    return 0.5 * (trace + mahal - mu_q.shape[-1] + logdet)


@register_kernel("gaussian_diagonal")
def _diag_kl_filtering_kernel(
    mu_q:       torch.Tensor,             # (N, K)
    sigma_q:    torch.Tensor,             # (N, K)
    mu_p:       torch.Tensor,             # (N, K)
    sigma_p:    torch.Tensor,             # (N, K)
    mu_t:       torch.Tensor,             # (N, N, K) transported key means
    sigma_t:    torch.Tensor,             # (N, N, K) transported key variances
    beta_coord: torch.Tensor,             # (N, N, K) PER-COORDINATE attention weights
    alpha_coef: torch.Tensor,             # (N, 1) or (N, K) self-coupling coefficient

    *,
    kl_max:     float = 100.0,
    eps:        float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Diagonal-KL query-side (filtering) gradient (per-head aware).

      grad_mu_i    = m_i a_i (mu_i - mu_p_i)/sigma_p_i + Sum_j beta_ij^(h(k)) (mu_i - mu_t_ij)/sigma_t_ij
      grad_sigma_i = m_i a_i 0.5(1/sigma_p_i - 1/sigma_q_i)
                     + Sum_j beta_ij^(h(k)) 0.5(1/sigma_t_ij - 1/sigma_q_i)

    ``beta_coord`` is the attention weight broadcast to coordinate k via k's irrep block h(k)
    (the per-head weight beta_ij^(h)); for a single block it is the one beta_ij repeated across
    every coordinate, so the per-coordinate einsum ``ijk,ijk->ik`` reduces bit-identically to the
    legacy ``ij,ijk->ik``. The caller (belief_gradients) builds beta_coord from the per-head beta.

    Self-term saturation mask m_i = 1[0 < D(q_i||p_i) < kl_max]: the oracle differentiates through
    safe_kl_clamp(D, [0, kl_max]), whose gradient is 0 once the raw self-divergence saturates the
    clamp, so the hand kernel zeros its self-term there to stay EXACTLY equal to the filtering
    oracle. The pairwise term needs no mask: a saturated E_ij drives beta_ij -> 0 on both sides.
    """
    sp = sigma_p.clamp(min=eps); sq = sigma_q.clamp(min=eps); st = sigma_t.clamp(min=eps)

    raw_self = _raw_diag_kl(mu_q, sigma_q, mu_p, sigma_p, eps=eps)              # (N,)
    self_mask = ((raw_self > 0.0) & (raw_self < kl_max)).to(mu_q.dtype).unsqueeze(-1)

    self_mu  = self_mask * alpha_coef * (mu_q - mu_p) / sp
    pair_mu  = torch.einsum("...ijk,...ijk->...ik", beta_coord, (mu_q.unsqueeze(-2) - mu_t) / st)
    grad_mu  = self_mu + pair_mu

    self_sig = self_mask * alpha_coef * 0.5 * (1.0 / sp - 1.0 / sq)
    pair_sig = torch.einsum("...ijk,...ijk->...ik", beta_coord, 0.5 * (1.0 / st - 1.0 / sq.unsqueeze(-2)))
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
    divergence_family:         str  = "renyi",
    alpha_mode:                str  = "constant",
    value:                     float = 1.0,

    irrep_dims:                Optional[List[int]]    = None,
    log_prior:                 Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Belief gradient: hand kernel for filtering+gaussian_diagonal+KL+canonical, else oracle.

    ``irrep_dims`` (when more than one block) makes attention PER HEAD: the energy/beta carry a
    head axis and the per-coordinate beta the kernel consumes is head h's weight on coordinate k.
    """
    use_kernel = (
        gradient_mode == "filtering"
        and family == "gaussian_diagonal"
        and divergence_family == "renyi"
        and abs(alpha_div - 1.0) < 1e-9
        and include_attention_entropy
        and has_kernel(family)
    )
    if not use_kernel:
        return belief_gradients_autograd(
            mu, sigma, mu_p, sigma_p, omega, tau=tau, alpha_div=alpha_div,
            kl_max=kl_max, eps=eps, b0=b0, c0=c0, value=value,
            include_attention_entropy=include_attention_entropy,
            gradient_mode=gradient_mode, family=family, divergence_family=divergence_family,
            alpha_mode=alpha_mode, irrep_dims=irrep_dims, log_prior=log_prior,
        )

    mu_k, sigma_k = mu.detach(), sigma.detach()
    mu_t = transport_mean(omega, mu_k)                 # rank-agnostic: (N,N,K) or (B,N,N,K)
    sigma_t = transport_covariance(omega, sigma_k)
    sd = self_divergence(mu, sigma, mu_p, sigma_p, alpha=1.0, kl_max=kl_max, eps=eps,
                         family=family, divergence_family=divergence_family)
    energy = pairwise_energy(mu, sigma, mu_t, sigma_t, alpha=1.0, kl_max=kl_max, eps=eps,
                             family=family, divergence_family=divergence_family, irrep_dims=irrep_dims)
    beta = attention_weights(energy, tau=tau, log_prior=log_prior)   # (N,N) or (H,N,N)
    beta_coord = _beta_to_coordinate(beta, irrep_dims, mu.shape[-1])  # (N,N,K) per-coordinate
    coef = alpha_gradient_coefficient(sd, value=value, b0=b0, c0=c0, mode=alpha_mode).unsqueeze(-1)
    return get_kernel(family)(mu, sigma, mu_p, sigma_p, mu_t, sigma_t, beta_coord, coef, kl_max=kl_max, eps=eps)


def _beta_to_coordinate(
    beta:       torch.Tensor,             # (N, N) single-block OR (H, N, N) per-head
    irrep_dims: Optional[List[int]],      # block sizes; None/[K] -> single block
    K:          int,                      # total belief dimension
) -> torch.Tensor:                        # (N, N, K) per-coordinate attention weight
    r"""Broadcast attention weights to coordinate k via k's irrep block h(k).

    Single block (irrep_dims None or length 1): the one beta_ij repeated across every
    coordinate. Per-head ((H,N,N) with H>1): head h's weight repeated across its d_head
    coordinates, so coordinate k carries beta_ij^(h(k)).
    """
    if irrep_dims is None or len(irrep_dims) == 1:
        return beta.unsqueeze(-1).expand(*beta.shape, K)
    reps = torch.tensor(irrep_dims, device=beta.device)
    return torch.repeat_interleave(beta.movedim(-3, -1), reps, dim=-1)   # (N,N,H)->(N,N,K)


def get_kernel(name: str) -> Callable:
    """Return the registered kernel for family ``name`` (KeyError if absent)."""
    if name not in _KERNELS:
        raise KeyError(f"no belief-gradient kernel for family {name!r}; available: {sorted(_KERNELS)}")
    return _KERNELS[name]
