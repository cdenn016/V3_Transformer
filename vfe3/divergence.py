r"""The divergence seam for VFE_3.0.

Renyi alpha-divergence is the primitive; KL is its alpha = 1 special case.
Every caller imports the divergence it needs from this single module.
Concrete kernels are registered by family name so variants swap by config.

Diagonal Gaussian KL:
    KL(q || p) = 1/2 ( sum_k s_k/t_k + sum_k (mu_t^k - mu_q^k)^2/t_k
                       - K + sum_k log(t_k/s_k) )
Diagonal Gaussian Renyi (blend sigma_b = (1-a) s + a t):
    D_a(q || p) = 1/2 [ a sum_k (mu_t-mu_q)^2/sigma_b
                        + 1/(a-1) sum_k ((1-a) log s + a log t - log sigma_b) ]
"""

import warnings
from typing import Callable, Dict

import torch


def _warn_alpha_gt_one(alpha: float, where: str) -> None:
    """Warn that alpha > 1 leaves the convex regime of the Renyi blend.

    For alpha > 1 the blend ``(1 - alpha) Sigma_q + alpha Sigma_t`` is not a
    convex combination and may be non-positive-definite: the diagonal kernel
    clamps it (returning a saturated value), and the full kernel can fail the
    Cholesky (where VFE_2.0 returns NaN). Robust alpha > 1 handling is a
    deferred hardening task. Python's default warning filter shows this once
    per call-site, so it does not spam an inner loop.
    """
    warnings.warn(
        f"{where}: alpha={alpha} > 1 leaves the convex regime; the Renyi blend "
        f"(1-alpha)*Sigma_q + alpha*Sigma_t may be non-positive-definite "
        f"(diagonal clamps; full may fail Cholesky, where 2.0 returns NaN).",
        RuntimeWarning,
        stacklevel=3,
    )


def safe_kl_clamp(
    kl:     torch.Tensor,

    kl_max: float = 100.0,
) -> torch.Tensor:
    r"""Clamp to [0, kl_max]; map NaN/+inf -> kl_max, -inf -> 0.

    Matches VFE_2.0 ``safe_kl_clamp`` default (non-propagating) policy:
    degenerate pairs become repulsive (kl_max) so a downstream softmax
    ignores them rather than attending to them.
    """
    kl = kl.clamp(min=0.0, max=kl_max)
    return kl.nan_to_num(nan=kl_max, posinf=kl_max, neginf=0.0)


# ---------------------------------------------------------------------------
# Registry: family name -> divergence callable. Variants swap by config.
# Signature: fn(mu_q, sigma_q, mu_t, sigma_t, *, alpha, kl_max, eps) -> Tensor
# ---------------------------------------------------------------------------
_DIVERGENCES: Dict[str, Callable] = {}


def register_divergence(name: str) -> Callable:
    """Decorator registering a divergence kernel under ``name``."""
    def _wrap(fn: Callable) -> Callable:
        _DIVERGENCES[name] = fn
        return fn
    return _wrap


def get_divergence(name: str) -> Callable:
    """Return the registered divergence kernel for ``name`` (KeyError if absent)."""
    if name not in _DIVERGENCES:
        raise KeyError(
            f"no divergence registered under {name!r}; "
            f"available: {sorted(_DIVERGENCES)}"
        )
    return _DIVERGENCES[name]


@register_divergence("gaussian_diagonal")
def _gaussian_diagonal_renyi(
    mu_q:    torch.Tensor,             # (..., K) query means
    sigma_q: torch.Tensor,             # (..., K) query diagonal variances
    mu_t:    torch.Tensor,             # (..., K) transported key means
    sigma_t: torch.Tensor,             # (..., K) transported key diagonal variances

    *,
    alpha:   float = 1.0,
    kl_max:  float = 100.0,
    eps:     float = 1e-6,
) -> torch.Tensor:
    r"""Diagonal Gaussian Renyi divergence; KL at ``alpha == 1``.

    Ported from VFE_2.0 ``_kl_kernel_diagonal`` (kl_computation.py:419-459).
    """
    K = mu_q.shape[-1]
    mu_q = mu_q.float()
    sigma_q = sigma_q.float().clamp(min=eps)
    mu_t = mu_t.float()
    sigma_t = sigma_t.float().clamp(min=eps)

    if alpha > 1.0:
        _warn_alpha_gt_one(alpha, "_gaussian_diagonal_renyi")

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

    return safe_kl_clamp(div, kl_max)


@register_divergence("gaussian_full")
def _gaussian_full_renyi(
    mu_q:    torch.Tensor,             # (..., K) query means
    sigma_q: torch.Tensor,             # (..., K, K) query covariances
    mu_t:    torch.Tensor,             # (..., K) transported key means
    sigma_t: torch.Tensor,             # (..., K, K) transported key covariances

    *,
    alpha:   float = 1.0,
    kl_max:  float = 100.0,
    eps:     float = 1e-6,
) -> torch.Tensor:
    r"""Full-covariance Gaussian Renyi divergence; KL at ``alpha == 1``.

    Ported from VFE_2.0 ``_kl_kernel_dense`` (kl_computation.py:270-330),
    with the default ``eps * I`` regularization. The 5-round Cholesky
    escalation and NaN-pair masking are 2.0 robustness features not needed
    for well-conditioned inputs; they are deferred to a later hardening task.
    For ``alpha > 1`` the blend can be indefinite and the Cholesky may fail
    (2.0 returns NaN there); a RuntimeWarning is emitted in that regime.
    """
    K = mu_q.shape[-1]
    device = mu_q.device
    mu_q = mu_q.float()
    sigma_q = sigma_q.float()
    mu_t = mu_t.float()
    sigma_t = sigma_t.float()

    if alpha > 1.0:
        _warn_alpha_gt_one(alpha, "_gaussian_full_renyi")

    eye = torch.eye(K, device=device, dtype=torch.float32)
    sigma_q_reg = sigma_q + eps * eye
    sigma_t_reg = sigma_t + eps * eye

    def _logdet_chol(L: torch.Tensor) -> torch.Tensor:
        return 2.0 * torch.log(
            torch.diagonal(L, dim1=-2, dim2=-1).clamp(min=1e-12)
        ).sum(dim=-1)

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

    return safe_kl_clamp(div, kl_max)


def renyi(
    mu_q:    torch.Tensor,
    sigma_q: torch.Tensor,
    mu_t:    torch.Tensor,
    sigma_t: torch.Tensor,

    *,
    alpha:   float = 1.0,
    kl_max:  float = 100.0,
    eps:     float = 1e-6,
    family:  str   = "gaussian_diagonal",
) -> torch.Tensor:
    """Renyi alpha-divergence D_alpha(q || p) for the selected family."""
    return get_divergence(family)(
        mu_q, sigma_q, mu_t, sigma_t, alpha=alpha, kl_max=kl_max, eps=eps
    )


def kl(
    mu_q:    torch.Tensor,
    sigma_q: torch.Tensor,
    mu_t:    torch.Tensor,
    sigma_t: torch.Tensor,

    *,
    kl_max:  float = 100.0,
    eps:     float = 1e-6,
    family:  str   = "gaussian_diagonal",
) -> torch.Tensor:
    """KL(q || p) = Renyi at alpha = 1."""
    return renyi(
        mu_q, sigma_q, mu_t, sigma_t,
        alpha=1.0, kl_max=kl_max, eps=eps, family=family,
    )
