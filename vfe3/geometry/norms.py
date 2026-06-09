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
    r"""mu_norm = mu * sqrt(K / (mu^T Sigma^-1 mu + eps)).

    The Mahalanobis scalar ``s2 = mu^T Sigma^-1 mu`` is gauge-invariant: under
    mu->g mu, Sigma->g Sigma g^T it maps to ``mu^T g^T (g Sigma g^T)^-1 g mu = mu^T Sigma^-1 mu``.
    The scale ``sqrt(K/s2)`` is therefore invariant and ``mu_norm`` transforms as a
    vector. Pure math, no parameters.
    """

    def __init__(
        self,
        K:   int,

        *,
        eps: float = 1e-6,
    ) -> None:
        self.K = K
        self.eps = eps

    def __call__(
        self,
        mu:    torch.Tensor,             # (..., K) means
        sigma: torch.Tensor,             # (..., K) diagonal OR (..., K, K) full covariances
    ) -> torch.Tensor:                   # (..., K) rescaled means
        r"""Rescale ``mu`` by the gauge-invariant Mahalanobis length."""
        if sigma.dim() == mu.dim():
            s2 = (mu ** 2 / sigma.clamp(min=self.eps)).sum(dim=-1, keepdim=True)
        else:
            # eps * I regularization (matching divergence._gaussian_full_renyi) so a
            # singular / near-singular Sigma does not raise torch._C._LinAlgError and
            # crash the forward pass; bounds the conditioning the solve sees.
            eye = torch.eye(self.K, device=sigma.device, dtype=sigma.dtype)        # (K, K)
            sigma_reg = sigma + self.eps * eye                                     # (..., K, K)
            sig_inv_mu = torch.linalg.solve(sigma_reg, mu.unsqueeze(-1)).squeeze(-1)   # Sigma^-1 mu
            s2 = (mu * sig_inv_mu).sum(dim=-1, keepdim=True)                       # mu^T Sigma^-1 mu
        return mu * torch.sqrt(self.K / s2.clamp(min=self.eps))


@register_norm("none")
def _norm_none(K: int, **kwargs) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Identity norm (no rescaling)."""
    def _identity(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return mu
    return _identity


@register_norm("mahalanobis")
def _norm_mahalanobis(K: int, *, eps: float = 1e-6, **kwargs) -> MahalanobisNorm:
    """MahalanobisNorm builder."""
    return MahalanobisNorm(K, eps=eps)
