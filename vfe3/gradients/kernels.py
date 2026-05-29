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
