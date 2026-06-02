r"""The divergence seam for VFE_3.0 (tensor-API facade over the families layer).

Renyi alpha-divergence is the primitive; KL is its alpha = 1 special case. The closed
forms and the exponential-family abstraction now live in ``vfe3.families``; this module
keeps the historical tensor-tuple entry points (and re-exports the families registry
helpers) so existing callers are unaffected during the parameter-object migration.

Diagonal Gaussian KL:
    KL(q || p) = 1/2 ( sum_k s_k/t_k + sum_k (mu_t^k - mu_q^k)^2/t_k
                       - K + sum_k log(t_k/s_k) )
Diagonal Gaussian Renyi (blend sigma_b = (1-a) s + a t):
    D_a(q || p) = 1/2 [ a sum_k (mu_t-mu_q)^2/sigma_b
                        + 1/(a-1) sum_k ((1-a) log s + a log t - log sigma_b) ]
"""

import torch

from vfe3.families.base import (
    safe_kl_clamp,
    family_cov_kind,
    divergence_families,
    register_functional,
    get_functional,
    get_family,
)
from vfe3.families.base import _warn_alpha_gt_one  # noqa: F401  (kept for back-compat imports)
from vfe3.families import gaussian as _gaussian     # noqa: F401  (registers the Gaussian families)
from vfe3.families.base import renyi as _renyi_params


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
    r"""Renyi alpha-divergence D_alpha(q || p) for the selected family.

    Tensor-API facade: wraps the moment tensors in the registered ``BeliefParams`` subclass
    for ``family`` and delegates to ``vfe3.families.base.renyi`` (which validates alpha,
    warns for alpha > 1, and dispatches the family's closed form).
    """
    cls = get_family(family)
    return _renyi_params(
        cls(mu_q, sigma_q), cls(mu_t, sigma_t),
        alpha=alpha, kl_max=kl_max, eps=eps,
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
    r"""KL(q || p) = Renyi at alpha = 1."""
    return renyi(
        mu_q, sigma_q, mu_t, sigma_t,
        alpha=1.0, kl_max=kl_max, eps=eps, family=family,
    )


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
    r"""Per-coordinate diagonal Gaussian Renyi/KL (the coordinate terms left UNSUMMED, each
    clamped independently). Tensor-API facade over ``DiagonalGaussian.renyi_per_coord``."""
    from vfe3.families.gaussian import DiagonalGaussian
    return DiagonalGaussian(mu_q, sigma_q).renyi_per_coord(
        DiagonalGaussian(mu_t, sigma_t), alpha=alpha, kl_max=kl_max, eps=eps,
    )


# The functional registry (``divergence_family`` name -> functional) lives in
# ``vfe3.families.base``; ``base`` registers the PARAM-typed ``renyi`` there at import. The
# energy call sites (``free_energy.pairwise_energy``/``self_divergence``) still invoke the
# functional with the historical TENSOR signature, so re-register the tensor ``renyi`` under
# "renyi" here (this mutates the shared ``base._FUNCTIONALS``) to keep those callers working
# during Phase 2; Phase 3 flips the call sites to parameter objects.
register_functional("renyi")(renyi)
