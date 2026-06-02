r"""Gaussian exponential families (diagonal and full covariance) for VFE_3.0.

The closed-form Renyi/KL kernels are ported verbatim from the legacy ``divergence.py``
moment forms, so the live numerics are byte-identical; natural-parameter and log-partition
maps are added so the generic Bregman/Renyi-from-A path can be pinned against them.
"""

import math
from typing import List, Tuple

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

    @classmethod
    def stack(cls, parts: List["DiagonalGaussian"], *, dim: int = 0) -> "DiagonalGaussian":
        return DiagonalGaussian(
            torch.stack([p.mu for p in parts], dim=dim),
            torch.stack([p.sigma for p in parts], dim=dim),
        )

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
        r"""Per-coordinate diagonal Gaussian Renyi/KL: the diagonal Renyi/KL coordinate terms
        left UNSUMMED, each clamped independently (sum over k recovers ``renyi_closed_form``)."""
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


@register_family("gaussian_full")
class FullGaussian(BeliefParams):
    r"""Full-covariance Gaussian: mu (..., K), sigma (..., K, K) SPD covariance.

    Natural theta = (Sigma^{-1} mu, -1/2 Sigma^{-1}); A(theta) = -1/4 t1^T t2^{-1} t1 - 1/2 log|-2 t2|.
    """

    cov_kind = "full"

    def __init__(self, mu: torch.Tensor, sigma: torch.Tensor) -> None:
        self.mu = mu
        self.sigma = sigma

    def coordinate_dim(self) -> int:
        return self.mu.shape[-1]

    def block(self, start: int, end: int) -> "FullGaussian":
        return FullGaussian(self.mu[..., start:end], self.sigma[..., start:end, start:end])

    def broadcast_over_keys(self) -> "FullGaussian":
        return FullGaussian(self.mu.unsqueeze(-2), self.sigma.unsqueeze(-3))

    @classmethod
    def stack(cls, parts: List["FullGaussian"], *, dim: int = 0) -> "FullGaussian":
        return FullGaussian(
            torch.stack([p.mu for p in parts], dim=dim),
            torch.stack([p.sigma for p in parts], dim=dim),
        )

    def natural(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Ridge the covariance before inverting to the precision (the natural parameter); the
        # 1e-6 jitter matches the eps the closed-form full kernel uses for SPD safety.
        eye = torch.eye(self.mu.shape[-1], device=self.mu.device, dtype=self.mu.dtype)
        prec = torch.linalg.solve(self.sigma + 1e-6 * eye, eye.expand_as(self.sigma))
        t1 = (prec @ self.mu.unsqueeze(-1)).squeeze(-1)
        return (t1, -0.5 * prec)

    @classmethod
    def log_partition_at(cls, theta: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        t1, t2 = theta
        neg2t2 = -2.0 * t2
        L = torch.linalg.cholesky(neg2t2)
        inv_neg2t2 = torch.cholesky_inverse(L)
        quad = (t1.unsqueeze(-2) @ inv_neg2t2 @ t1.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        return 0.5 * quad - 0.5 * _logdet_chol(L)

    def expected_statistic(self) -> Tuple[torch.Tensor, torch.Tensor]:
        outer = self.mu.unsqueeze(-1) * self.mu.unsqueeze(-2)
        return (self.mu, self.sigma + outer)

    def entropy(self) -> torch.Tensor:
        K = self.mu.shape[-1]
        L = torch.linalg.cholesky(self.sigma)
        return 0.5 * _logdet_chol(L) + 0.5 * K * math.log(2.0 * math.pi * math.e)

    def renyi_closed_form(
        self,
        other:   "FullGaussian",

        *,
        alpha:   float = 1.0,
        kl_max:  float = 100.0,
        eps:     float = 1e-6,
    ) -> torch.Tensor:
        r"""Closed-form full-covariance Gaussian Renyi/KL (ported verbatim from
        ``divergence._gaussian_full_renyi``; mu_q=self, mu_t=other)."""
        K = self.mu.shape[-1]
        device = self.mu.device
        mu_q = self.mu.float()
        sigma_q = self.sigma.float()
        mu_t = other.mu.float()
        sigma_t = other.sigma.float()
        eye = torch.eye(K, device=device, dtype=torch.float32)
        sigma_q_reg = sigma_q + eps * eye
        sigma_t_reg = sigma_t + eps * eye
        if abs(alpha - 1.0) < 1e-6:
            L_p = torch.linalg.cholesky(sigma_t_reg)
            Y = torch.linalg.solve_triangular(L_p, sigma_q_reg, upper=False)
            Z = torch.linalg.solve_triangular(L_p.transpose(-1, -2), Y, upper=True)
            trace_term = torch.diagonal(Z, dim1=-2, dim2=-1).sum(dim=-1)
            delta_mu = mu_t - mu_q
            v = torch.linalg.solve_triangular(
                L_p, delta_mu.unsqueeze(-1), upper=False
            ).squeeze(-1)
            mahal_term = (v ** 2).sum(dim=-1)
            logdet_p = _logdet_chol(L_p)
            logdet_q = _logdet_chol(torch.linalg.cholesky(sigma_q_reg))
            div = 0.5 * (trace_term + mahal_term - K + logdet_p - logdet_q)
        else:
            sigma_blend = (1.0 - alpha) * sigma_q_reg + alpha * sigma_t_reg
            sigma_blend = 0.5 * (sigma_blend + sigma_blend.transpose(-1, -2))
            L_blend = torch.linalg.cholesky(sigma_blend)
            delta_mu = mu_t - mu_q
            v = torch.linalg.solve_triangular(
                L_blend, delta_mu.unsqueeze(-1), upper=False
            ).squeeze(-1)
            mahal_term = alpha * (v ** 2).sum(dim=-1)
            logdet_q = _logdet_chol(torch.linalg.cholesky(sigma_q_reg))
            logdet_t = _logdet_chol(torch.linalg.cholesky(sigma_t_reg))
            logdet_blend = _logdet_chol(L_blend)
            logdet_term = (
                (1.0 - alpha) * logdet_q + alpha * logdet_t - logdet_blend
            ) / (alpha - 1.0)
            div = 0.5 * (mahal_term + logdet_term)
        return safe_kl_clamp(div, kl_max=kl_max)
