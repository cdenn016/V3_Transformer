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
    value:        float = 1.0,

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
    alpha, reg = self_coupling_alpha(sd, mode=alpha_mode, value=value, b0=b0, c0=c0)
    energy = pairwise_energy(mu_q, sigma_q, mu_t, sigma_t, alpha=alpha_div, kl_max=kl_max, eps=eps, family=family)
    F = free_energy(
        sd, energy, alpha, tau=tau, include_attention_entropy=include_attention_entropy,
        log_prior=log_prior, alpha_reg=(reg if alpha_mode != "constant" else None),
    )
    grad_mu, grad_sigma = torch.autograd.grad(F, [mu_q, sigma_q])
    return grad_mu.detach(), grad_sigma.detach()
