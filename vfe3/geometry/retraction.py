r"""SPD-manifold retractions + Fisher natural-gradient preconditioner (VFE_3.0).

The SPD retraction keeps Sigma on the SPD manifold under a tangent update; the
Fisher preconditioner converts Euclidean (mu, sigma) gradients to natural
gradients. The phi Lie-algebra retraction is a separate phase.
"""

import math
from typing import Optional, Tuple

import torch

from vfe3.geometry.groups import GaugeGroup
from vfe3.geometry.lie_ops import (
    clamp_phi_trace,
    project_phi_to_slk,
    retract_glk,
    retract_son,
)


def retract_spd_diagonal(
    sigma_diag:   torch.Tensor,             # (..., K) diagonal variances
    delta_sigma:  torch.Tensor,             # (..., K) diagonal tangent

    *,
    step_size:    float = 1.0,
    trust_region: float = 5.0,
    eps:          float = 1e-6,
    sigma_max:    float = 5.0,
) -> torch.Tensor:
    r"""Diagonal SPD retraction sigma_new = sigma * exp(tau * clamp(dsigma/sigma)).

    Positivity by construction (exp > 0); clamped to [eps, sigma_max].
    """
    orig_dtype = sigma_diag.dtype
    with torch.amp.autocast('cuda', enabled=False):
        sigma_safe = sigma_diag.float().clamp(min=eps)
        delta_sigma = delta_sigma.float()
        whitened = delta_sigma / sigma_safe
        if trust_region is not None and trust_region > 0:
            whitened = whitened.clamp(-trust_region, trust_region)
        exp_arg = (step_size * whitened).clamp(-50.0, 50.0)
        sigma_new = sigma_safe * torch.exp(exp_arg)
    return sigma_new.clamp(min=eps, max=sigma_max).to(orig_dtype)


def retract_spd_full(
    sigma:        torch.Tensor,             # (..., K, K) SPD covariances
    delta_sigma:  torch.Tensor,             # (..., K, K) symmetric tangent

    *,
    step_size:    float = 1.0,
    trust_region: float = 2.0,
    eps:          float = 1e-6,
    sigma_max:    float = 5.0,
) -> torch.Tensor:
    r"""Full SPD retraction via the affine-invariant exponential map.

        Sigma_new = S^{1/2} exp(S^{-1/2} (tau dSigma) S^{-1/2}) S^{1/2},
    with a Frobenius trust region on the whitened tangent and an eigenvalue
    floor/ceiling [eps, sigma_max^2]. Uses torch.linalg.eigh; a gap-regularized
    eigh backward for gradient stability on near-degenerate spectra is deferred
    to a later hardening pass.
    """
    orig_shape = sigma.shape
    orig_dtype = sigma.dtype
    if sigma.dim() == 4:
        B, N, K, _ = sigma.shape
        sigma = sigma.reshape(B * N, K, K)
        delta_sigma = delta_sigma.reshape(B * N, K, K)

    with torch.amp.autocast('cuda', enabled=False):
        sigma = sigma.float()
        delta_sigma = delta_sigma.float()
        sigma = 0.5 * (sigma + sigma.transpose(-1, -2))
        delta_sigma = 0.5 * (delta_sigma + delta_sigma.transpose(-1, -2))

        eigenvalues, eigenvectors = torch.linalg.eigh(sigma)
        eigenvalues = eigenvalues.clamp(min=eps)
        sqrt_eig     = torch.sqrt(eigenvalues)
        inv_sqrt_eig = 1.0 / sqrt_eig
        sigma_sqrt     = eigenvectors * sqrt_eig.unsqueeze(-2)     @ eigenvectors.transpose(-1, -2)
        sigma_inv_sqrt = eigenvectors * inv_sqrt_eig.unsqueeze(-2) @ eigenvectors.transpose(-1, -2)

        R = sigma_inv_sqrt @ (step_size * delta_sigma) @ sigma_inv_sqrt
        R = 0.5 * (R + R.transpose(-1, -2))
        if trust_region is not None and trust_region > 0:
            R_norm = torch.linalg.norm(R, ord='fro', dim=(-2, -1), keepdim=True)
            R = R * torch.clamp(trust_region / (R_norm + eps), max=1.0)

        R_eval, R_evec = torch.linalg.eigh(R)
        R_eval = R_eval.clamp(-50.0, 50.0)
        exp_R = R_evec * torch.exp(R_eval).unsqueeze(-2) @ R_evec.transpose(-1, -2)

        sigma_new = sigma_sqrt @ exp_R @ sigma_sqrt
        sigma_new = 0.5 * (sigma_new + sigma_new.transpose(-1, -2))

        eig_new, vec_new = torch.linalg.eigh(sigma_new)
        eig_new = eig_new.clamp(min=eps, max=sigma_max * sigma_max)
        sigma_new = vec_new * eig_new.unsqueeze(-2) @ vec_new.transpose(-1, -2)

    sigma_new = sigma_new.to(orig_dtype)
    if len(orig_shape) == 4:
        sigma_new = sigma_new.reshape(orig_shape)
    return sigma_new


def natural_gradient(
    grad_mu:    torch.Tensor,             # (..., K) Euclidean grad wrt mu
    grad_sigma: torch.Tensor,             # (..., K) or (..., K, K) Euclidean grad wrt sigma
    sigma_q:    torch.Tensor,             # (..., K) diagonal OR (..., K, K) full covariance

    *,
    eps:        float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Fisher preconditioner: Euclidean -> natural gradient for a Gaussian.

        nat_mu    = Sigma grad_mu
        nat_sigma = 2 Sigma grad_sigma Sigma   (diagonal: 2 sigma^2 grad_sigma)
    The Fisher metric on Sigma is g(dS1,dS2) = (1/2) tr(S^-1 dS1 S^-1 dS2), so
    g^{kk} = 2 sigma_k^2 in the diagonal case.

    Diagonal vs full is detected by ``sigma_q.dim() == grad_mu.dim()`` (diagonal
    sigma matches the mean rank; full sigma has one extra trailing dim). This is
    correct for the belief shapes used here ((B,N,K) diagonal, (B,N,K,K) full);
    a caller passing other ranks must pass matching mean/sigma ranks.
    """
    is_diagonal = sigma_q.dim() == grad_mu.dim()
    orig_dtype = sigma_q.dtype
    with torch.amp.autocast('cuda', enabled=False):
        sigma_q    = sigma_q.float()
        grad_mu    = grad_mu.float()
        grad_sigma = grad_sigma.float()
        if is_diagonal:
            sigma_safe     = sigma_q.clamp(min=eps)
            nat_grad_mu    = sigma_safe * grad_mu
            nat_grad_sigma = 2.0 * sigma_safe * sigma_safe * grad_sigma
        else:
            nat_grad_mu    = torch.einsum('...ij,...j->...i', sigma_q, grad_mu)
            nat_grad_sigma = 2.0 * torch.einsum('...ij,...jk,...kl->...il', sigma_q, grad_sigma, sigma_q)
            nat_grad_sigma = 0.5 * (nat_grad_sigma + nat_grad_sigma.transpose(-1, -2))
    return nat_grad_mu.to(orig_dtype), nat_grad_sigma.to(orig_dtype)


def retract_phi(
    phi:          torch.Tensor,           # (..., n_gen) current gauge frame
    delta_phi:    torch.Tensor,           # (..., n_gen) tangent step (e.g. -grad_phi)
    group:        GaugeGroup,             # supplies generators, skew flag, irrep_dims

    *,
    step_size:    float = 1.0,
    eps:          float = 1e-6,
    order:        int   = 4,
    project_slk:  bool  = False,
    mode:         str   = "euclidean",

    trust_region: Optional[float] = None, # None -> group default (GL:0.1, SO:0.3)
    max_norm:     Optional[float] = None, # None -> group default (GL:5.0, SO:pi)
    trace_clamp:  Optional[float] = None, # soft per-block |tr| cap (GL only)
) -> torch.Tensor:
    r"""Group-aware phi retraction dispatcher (Gaussian-specialized).

    Skew group (SO(N)) -> retract_son, det control is a no-op (det exp = 1).
    Non-skew (GL(K))   -> retract_glk, then optional det control:
      project_slk=True  hard-projects each block to sl(K) (det Omega_h = 1);
      else trace_clamp soft-bounds |tr| per block. Defaults for trust_region /
      max_norm are taken from the group's compactness when not given.
    """
    G = group.generators
    if trust_region is None:
        trust_region = 0.3 if group.skew_symmetric else 0.1
    if max_norm is None:
        max_norm = math.pi if group.skew_symmetric else 5.0

    if group.skew_symmetric:
        return retract_son(
            phi, delta_phi, G, step_size=step_size, trust_region=trust_region,
            max_norm=max_norm, eps=eps, order=order, mode=mode,
        )

    phi_new = retract_glk(
        phi, delta_phi, G, step_size=step_size, trust_region=trust_region,
        max_norm=max_norm, eps=eps, order=order, mode=mode,
    )
    if project_slk:
        phi_new = project_phi_to_slk(phi_new, G, group.irrep_dims)
    elif trace_clamp is not None:
        phi_new = clamp_phi_trace(phi_new, G, group.irrep_dims, trace_max=trace_clamp)
    return phi_new
