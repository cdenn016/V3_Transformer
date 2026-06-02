r"""Gaussian exponential families (diagonal and full covariance) for VFE_3.0.

The closed-form Renyi/KL kernels are ported verbatim from the legacy ``divergence.py``
moment forms, so the live numerics are byte-identical; natural-parameter and log-partition
maps are added so the generic Bregman/Renyi-from-A path can be pinned against them.
"""

import math
from typing import Tuple

import torch

from vfe3.families.base import (
    BeliefParams, register_family, safe_kl_clamp, _logdet_chol,
)


@register_family("gaussian_diagonal")
class DiagonalGaussian(BeliefParams):
    r"""Diagonal Gaussian: mu (..., K), sigma (..., K) variances.

    Natural theta = (mu/sigma, -1/(2 sigma)); A(theta) = sum_k [ -t1^2/(4 t2) - 1/2 log(-2 t2) ];
    E[T] = (mu, mu^2 + sigma).
    """

    cov_kind = "diagonal"

    def __init__(self, mu: torch.Tensor, sigma: torch.Tensor) -> None:
        self.mu = mu
        self.sigma = sigma

    def coordinate_dim(self) -> int:
        return self.mu.shape[-1]

    def block(self, start: int, end: int) -> "DiagonalGaussian":
        return DiagonalGaussian(self.mu[..., start:end], self.sigma[..., start:end])

    def broadcast_over_keys(self) -> "DiagonalGaussian":
        return DiagonalGaussian(self.mu.unsqueeze(-2), self.sigma.unsqueeze(-2))

    def natural(self) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.sigma.clamp(min=1e-12)
        return (self.mu / s, -1.0 / (2.0 * s))

    @classmethod
    def log_partition_at(cls, theta: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        t1, t2 = theta
        return (-(t1 ** 2) / (4.0 * t2) - 0.5 * torch.log(-2.0 * t2)).sum(dim=-1)

    def expected_statistic(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return (self.mu, self.mu ** 2 + self.sigma)

    def entropy(self) -> torch.Tensor:
        return 0.5 * (
            torch.log(self.sigma.clamp(min=1e-12)) + math.log(2.0 * math.pi * math.e)
        ).sum(dim=-1)

    def renyi_closed_form(
        self,
        other:   "DiagonalGaussian",

        *,
        alpha:   float = 1.0,
        kl_max:  float = 100.0,
        eps:     float = 1e-6,
    ) -> torch.Tensor:
        r"""Closed-form diagonal Gaussian Renyi/KL (ported verbatim from
        ``divergence._gaussian_diagonal_renyi``; mu_q=self, mu_t=other)."""
        K = self.mu.shape[-1]
        mu_q = self.mu.float()
        sigma_q = self.sigma.float().clamp(min=eps)
        mu_t = other.mu.float()
        sigma_t = other.sigma.float().clamp(min=eps)
        if abs(alpha - 1.0) < 1e-6:
            trace_term  = (sigma_q / sigma_t).sum(dim=-1)
            delta       = mu_t - mu_q
            mahal_term  = ((delta ** 2) / sigma_t).sum(dim=-1)
            logdet_term = (torch.log(sigma_t) - torch.log(sigma_q)).sum(dim=-1)
            div = 0.5 * (trace_term + mahal_term - K + logdet_term)
        else:
            sigma_blend = ((1.0 - alpha) * sigma_q + alpha * sigma_t).clamp(min=eps)
            delta       = mu_t - mu_q
            mahal_term  = (alpha * (delta ** 2) / sigma_blend).sum(dim=-1)
            logdet_per_dim = (
                (1.0 - alpha) * torch.log(sigma_q)
                + alpha * torch.log(sigma_t)
                - torch.log(sigma_blend)
            )
            logdet_term = logdet_per_dim.sum(dim=-1) / (alpha - 1.0)
            div = 0.5 * (mahal_term + logdet_term)
        return safe_kl_clamp(div, kl_max=kl_max)

    def renyi_per_coord(
        self,
        other:   "DiagonalGaussian",

        *,
        alpha:   float = 1.0,
        kl_max:  float = 100.0,
        eps:     float = 1e-6,
    ) -> torch.Tensor:
        r"""Per-coordinate diagonal Gaussian Renyi/KL (ported verbatim from
        ``divergence.gaussian_diagonal_renyi_per_coord``)."""
        mu_q = self.mu.float()
        sigma_q = self.sigma.float().clamp(min=eps)
        mu_t = other.mu.float()
        sigma_t = other.sigma.float().clamp(min=eps)
        delta = mu_t - mu_q
        if abs(alpha - 1.0) < 1e-6:
            per_coord = 0.5 * (
                sigma_q / sigma_t + (delta ** 2) / sigma_t - 1.0
                + torch.log(sigma_t) - torch.log(sigma_q)
            )
        else:
            sigma_blend = ((1.0 - alpha) * sigma_q + alpha * sigma_t).clamp(min=eps)
            mahal       = alpha * (delta ** 2) / sigma_blend
            logdet      = (
                (1.0 - alpha) * torch.log(sigma_q)
                + alpha * torch.log(sigma_t)
                - torch.log(sigma_blend)
            ) / (alpha - 1.0)
            per_coord = 0.5 * (mahal + logdet)
        return safe_kl_clamp(per_coord, kl_max=kl_max)
