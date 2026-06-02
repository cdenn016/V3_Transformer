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
from typing import Callable, Dict, Tuple

import torch


def _warn_alpha_gt_one(alpha: float, family: str) -> None:
    """Warn that alpha > 1 leaves the convex regime of the Renyi blend.

    For alpha > 1 the blend ``(1 - alpha) Sigma_q + alpha Sigma_t`` is not a
    convex combination and may be non-positive-definite: the diagonal kernel
    clamps it (returning a saturated value), and the full kernel can fail the
    Cholesky (which then returns NaN). Robust alpha > 1 handling is a
    deferred hardening task. Python's default warning filter shows this once
    per unique message + call-site, so it does not spam an inner loop.

    Called from the public ``renyi`` boundary so ``stacklevel=3`` points the
    warning at the user's ``renyi(...)`` call rather than library internals.
    """
    warnings.warn(
        f"renyi: alpha={alpha} > 1 (family={family!r}) leaves the convex regime; "
        f"the blend (1-alpha)*Sigma_q + alpha*Sigma_t may be non-positive-definite "
        f"(diagonal clamps; full may fail Cholesky and return NaN).",
        RuntimeWarning,
        stacklevel=3,
    )


def _logdet_chol(L: torch.Tensor) -> torch.Tensor:
    r"""log|Sigma| for an SPD Sigma = L Lᵀ given its Cholesky factor L.

    Uses ``log|Sigma| = 2 sum_k log L_kk`` with a floor on the diagonal for
    numerical safety.
    """
    return 2.0 * torch.log(
        torch.diagonal(L, dim1=-2, dim2=-1).clamp(min=1e-12)
    ).sum(dim=-1)


def safe_kl_clamp(
    kl:     torch.Tensor,

    *,
    kl_max: float = 100.0,
) -> torch.Tensor:
    r"""Clamp to [0, kl_max]; map NaN/+inf -> kl_max, -inf -> 0.

    Non-propagating clamp policy: degenerate pairs become repulsive (kl_max)
    so a downstream softmax ignores them rather than attending to them.
    """
    kl = kl.clamp(min=0.0, max=kl_max)
    # clamp(min=0) already maps -inf -> 0; neginf=0.0 is kept for explicitness
    # (kept explicit for clarity).
    return kl.nan_to_num(nan=kl_max, posinf=kl_max, neginf=0.0)


# ---------------------------------------------------------------------------
# Registry: family name -> divergence callable. Variants swap by config.
# Signature: fn(mu_q, sigma_q, mu_t, sigma_t, *, alpha, kl_max, eps) -> Tensor
# ---------------------------------------------------------------------------
_DIVERGENCES: Dict[str, Callable] = {}
_COV_KIND:    Dict[str, str]      = {}     # family name -> "diagonal" | "full" covariance structure


def register_divergence(name: str, *, cov_kind: str) -> Callable:
    """Decorator registering a divergence kernel under ``name`` with its covariance
    structure ``cov_kind`` ("diagonal" | "full").

    Consumers dispatch on the declared ``cov_kind`` (via ``family_cov_kind``), never by
    sniffing the family name, so a new covariance family slots in by declaring its
    structure at registration -- no call site is edited.
    """
    def _wrap(fn: Callable) -> Callable:
        _DIVERGENCES[name] = fn
        _COV_KIND[name] = cov_kind
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


def family_cov_kind(name: str) -> str:
    """Covariance structure ("diagonal" | "full") declared for family ``name`` at
    registration (KeyError if absent).

    The single source of truth for whether a family carries diagonal variances or a full
    covariance, replacing name-substring sniffing (``"diagonal" in family``), which would
    silently misclassify a family whose name lacks the substring.
    """
    if name not in _COV_KIND:
        raise KeyError(
            f"no divergence family registered under {name!r}; "
            f"available: {sorted(_COV_KIND)}"
        )
    return _COV_KIND[name]


def divergence_families() -> Tuple[str, ...]:
    """Registered covariance-kernel family names (the valid ``family`` config values),
    derived from the registry so a newly registered family is a valid config family
    without editing config."""
    return tuple(sorted(_DIVERGENCES))


@register_divergence("gaussian_diagonal", cov_kind="diagonal")
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
    r"""Closed-form diagonal Gaussian Renyi/KL."""
    K = mu_q.shape[-1]
    mu_q = mu_q.float()
    sigma_q = sigma_q.float().clamp(min=eps)
    mu_t = mu_t.float()
    sigma_t = sigma_t.float().clamp(min=eps)

    # alpha is validated (positive) and alpha>1 warned at the renyi() boundary.
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


@register_divergence("gaussian_full", cov_kind="full")
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
    r"""Full-covariance Gaussian Renyi/KL via Cholesky, with the default ``eps * I`` regularization.

    A 5-round Cholesky escalation and NaN-pair masking for ill-conditioned inputs
    are deferred robustness features; for alpha > 1 the blend can be indefinite
    and the Cholesky may fail (returning NaN), and the public ``renyi`` emits a
    RuntimeWarning in that regime. alpha is validated (positive) at the ``renyi``
    boundary.
    """
    K = mu_q.shape[-1]
    device = mu_q.device
    mu_q = mu_q.float()
    sigma_q = sigma_q.float()
    mu_t = mu_t.float()
    sigma_t = sigma_t.float()

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


def gaussian_diagonal_renyi_per_coord(
    mu_q:    torch.Tensor,             # (..., K) query means
    sigma_q: torch.Tensor,             # (..., K) query diagonal variances
    mu_t:    torch.Tensor,             # (..., K) transported key means
    sigma_t: torch.Tensor,             # (..., K) transported key diagonal variances

    *,
    alpha:   float = 1.0,
    kl_max:  float = 100.0,
    eps:     float = 1e-6,
) -> torch.Tensor:                     # (..., K) per-coordinate diagonal Renyi/KL D^(k)
    r"""Per-coordinate diagonal Gaussian Renyi/KL: the coordinate terms of
    ``_gaussian_diagonal_renyi`` left UNSUMMED, each clamped independently by safe_kl_clamp.

    The ``-K`` of the summed form becomes ``-1`` per coordinate, so ``sum_k`` of this recovers
    the pre-clamp summed divergence. The clamp is PER COORDINATE (each D^(k) capped at kl_max),
    so a token's total can reach K*kl_max -- the per-coordinate regularisation scale (design
    decision, see the spec). Diagonal family only: full-covariance KL couples coordinates
    through the trace and log-determinant and does not decompose into a coordinate sum.
    """
    mu_q = mu_q.float()
    sigma_q = sigma_q.float().clamp(min=eps)
    mu_t = mu_t.float()
    sigma_t = sigma_t.float().clamp(min=eps)
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


def renyi(
    mu_q:    torch.Tensor,             # (..., K) query means
    sigma_q: torch.Tensor,             # (..., K) or (..., K, K) query (co)variances
    mu_t:    torch.Tensor,             # (..., K) transported key means
    sigma_t: torch.Tensor,             # (..., K) or (..., K, K) transported (co)variances

    *,
    alpha:   float = 1.0,
    kl_max:  float = 100.0,
    eps:     float = 1e-6,
    family:  str   = "gaussian_diagonal",
) -> torch.Tensor:
    """Renyi alpha-divergence D_alpha(q || p) for the selected family.

    Validates alpha (must be positive) and warns when alpha > 1 (the blend
    leaves the convex regime); see ``_warn_alpha_gt_one``.
    """
    if alpha <= 0.0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    if alpha > 1.0:
        _warn_alpha_gt_one(alpha, family)
    return get_divergence(family)(
        mu_q, sigma_q, mu_t, sigma_t, alpha=alpha, kl_max=kl_max, eps=eps
    )


def kl(
    mu_q:    torch.Tensor,             # (..., K) query means
    sigma_q: torch.Tensor,             # (..., K) or (..., K, K) query (co)variances
    mu_t:    torch.Tensor,             # (..., K) transported key means
    sigma_t: torch.Tensor,             # (..., K) or (..., K, K) transported (co)variances

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


# ---------------------------------------------------------------------------
# Functional registry: divergence_family name -> divergence FUNCTIONAL. Distinct from the
# per-family covariance kernels above (gaussian_diagonal/gaussian_full): the functional is
# the f-divergence form (Renyi today, parameterized by alpha; KL is alpha=1), and dispatches
# the covariance kernel via its own `family` argument. A new functional (a different
# f-divergence) slots in by register_functional, never by editing the energy call sites.
# Signature: fn(mu_q, sigma_q, mu_t, sigma_t, *, alpha, kl_max, eps, family) -> Tensor
# ---------------------------------------------------------------------------
_FUNCTIONALS: Dict[str, Callable] = {}


def register_functional(name: str) -> Callable:
    """Decorator registering a divergence functional under ``name``."""
    def _wrap(fn: Callable) -> Callable:
        _FUNCTIONALS[name] = fn
        return fn
    return _wrap


def get_functional(name: str) -> Callable:
    """Return the registered divergence functional for ``name`` (KeyError if absent)."""
    if name not in _FUNCTIONALS:
        raise KeyError(
            f"no divergence functional registered under {name!r}; "
            f"available: {sorted(_FUNCTIONALS)}"
        )
    return _FUNCTIONALS[name]


register_functional("renyi")(renyi)
