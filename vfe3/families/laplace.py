r"""Diagonal (factorized) Laplace belief family for VFE_3.0 -- the first non-Gaussian family.

Per coordinate ``p(x_k) = (1 / 2 b_k) exp(-|x_k - mu_k| / b_k)`` (location ``mu``, scale ``b``).
The belief tuple ``(mu, sigma)`` is reused verbatim: ``mu`` is the location and ``sigma`` is the
per-coordinate scale ``b`` (>0, floored by the SPD retraction exactly as a variance is). Genuinely
non-Gaussian: exponential (L1) tails, excess kurtosis 3, a non-differentiable cusp at the mode; the
fixed-location sufficient statistic is ``|x - mu|`` (not the Gaussian ``(x, x^2)``), so no
reparameterization maps a Laplace marginal to a Gaussian one. It is the canonical heavy-tailed /
sparse (LASSO) prior counterpart to the Gaussian.

NOT a natural exponential family when the location varies: ``|x - mu|`` depends on the parameter
``mu``, so the log-density is not affine in a fixed sufficient statistic and there is NO joint
log-partition ``A(theta)`` over ``(mu, b)``. The generic Bregman/Renyi-from-``A`` path
(``families/base.py``) is therefore unavailable here; ``natural`` / ``log_partition_at`` raise, and
the family ships a mandatory closed-form ``renyi_closed_form`` (the divergence dispatch in
``base.renyi`` prefers it whenever present). The Laplace IS an EF only in the degenerate scale-only
slice (``eta = -1/b``, ``T = |x - mu|`` with ``mu`` known), which the live varying-location belief is
not, so the override is required rather than optional.

DIVERGENCES (per coordinate; summed over k by ``renyi_closed_form``). With ``s = |mu_q - mu_p|``:

  KL (alpha = 1):  D = log(b_p / b_q) + s / b_p + (b_q / b_p) exp(-s / b_q) - 1   (>= 0, 0 iff q == p).

  Renyi (alpha in (0,1]):  D_alpha = 1/(alpha - 1) * log( int q^alpha p^{1-alpha} dx ). With
  c_q = alpha / b_q, c_p = (1-alpha) / b_p (both > 0 on (0,1)), the two-region exponential integral is

      int = (1 / 2) [ (e^{-c_p s} + e^{-c_q s}) / (c_q + c_p)
                      + (e^{-c_q s} - e^{-c_p s}) / (c_p - c_q) ],

  with a REMOVABLE singularity at ``c_p == c_q`` i.e. ``alpha* = b_q / (b_p + b_q)`` (the second term
  -> ``s exp(-c s)`` in the limit). The singularity is per-coordinate and data-dependent (each k has
  its own ``alpha*_k``), unlike the Gaussian's fixed band around ``alpha = 1``; the alpha != 1 branch
  is evaluated in float64 (catastrophic cancellation otherwise loses ~all fp32 digits within ~1e-6 of
  ``alpha*``), with a ``where`` limit branch at ``|c_p - c_q| < 1e-12``. The float64 island also
  covers the outer ``log(int) / (alpha - 1)`` cancellation as ``alpha -> 1``. For ``alpha > 1`` the
  blend rate can go non-positive (``c_q + c_p <= 0``), the integral diverges, and that coordinate maps
  to NaN -> ``kl_max`` -- mirroring the Gaussian non-convex-regime policy.

  Validated to ~1e-10 against deterministic float64 trapezoidal integration (KL, Renyi on (0,1),
  the singularity limit, and self-divergence) before this module was written.

TRANSPORT (honest scope). Gauge transport acts purely on the raw ``mu`` / ``sigma`` tensors before
the family is constructed (``transport_mean`` / ``transport_covariance``), so a location-scale family
rides it with no code change. It is distributionally EXACT only under a permutation/sign (1x1-block)
gauge element. Under any non-permutation rotation -- including a compact SO(d) within-block rotation
of ``block_glk`` (``d_head > 1``) and the non-orthogonal ``glk`` action -- a rotated diagonal Laplace
is not a diagonal Laplace, so transport reinterprets ``diag(Omega diag(b) Omega^T)`` as the new
marginal scale: a marginal-scale projection, not exact equivariance. This is the SAME diagonal
projection the diagonal Gaussian already incurs in the live pipeline (only the FULL-covariance
Gaussian KL is congruence-invariant under the live groups), PLUS a shape-level break (Laplace is not
closed under rotation at all). The architecture already tolerates non-invariant transport scoring
(the diagonal Gaussian runs as a projection), so this is documented, not a defect -- but it is not
"exact at identity, drifts" the way the head-mixer is; it is a standing projection under any
non-permutation gauge. Use under permutation/sign gauge for exactness.
"""

import math
from typing import List, NoReturn, Tuple

import torch

from vfe3.families.base import BeliefParams, register_family, safe_kl_clamp


@register_family("laplace_diagonal")
class DiagonalLaplace(BeliefParams):
    r"""Factorized Laplace: ``mu`` (..., K) location, ``sigma`` (..., K) per-coordinate scale ``b``."""

    cov_kind = "diagonal"

    def __init__(self, mu: torch.Tensor, sigma: torch.Tensor) -> None:
        self.mu = mu                                   # (..., K) location
        self.sigma = sigma                             # (..., K) scale b (the belief sigma slot)

    def coordinate_dim(self) -> int:
        return self.mu.shape[-1]

    def block(self, start: int, end: int) -> "DiagonalLaplace":
        return DiagonalLaplace(self.mu[..., start:end], self.sigma[..., start:end])

    def broadcast_over_keys(self) -> "DiagonalLaplace":
        return DiagonalLaplace(self.mu.unsqueeze(-2), self.sigma.unsqueeze(-2))

    @classmethod
    def stack(cls, parts: List["DiagonalLaplace"], *, dim: int = 0) -> "DiagonalLaplace":
        return DiagonalLaplace(
            torch.stack([p.mu for p in parts], dim=dim),
            torch.stack([p.sigma for p in parts], dim=dim),
        )

    def natural(self) -> NoReturn:
        raise NotImplementedError(
            "DiagonalLaplace is not a natural exponential family when the location varies "
            "(the sufficient statistic |x - mu| depends on the parameter mu), so it has no "
            "natural parameterization theta. It carries a closed-form renyi_closed_form instead; "
            "the generic Bregman/Renyi-from-A divergence path is not available for this family."
        )

    @classmethod
    def log_partition_at(cls, theta: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError(
            "DiagonalLaplace has no joint log-partition A(theta) over (mu, b): it is a natural "
            "exponential family only in the degenerate fixed-location, scale-only slice. Use its "
            "closed-form renyi_closed_form (the divergence dispatch prefers it); the generic "
            "Renyi-from-A path cannot serve this family."
        )

    def entropy(self) -> torch.Tensor:
        # H = sum_k log(2 b_k e) = sum_k [ log(2 b_k) + 1 ].
        return (torch.log(2.0 * self.sigma.clamp(min=1e-12)) + 1.0).sum(dim=-1)

    def natural_gradient(
        self,
        grad_mu:    torch.Tensor,             # (..., K) Euclidean grad wrt mu (location)
        grad_sigma: torch.Tensor,             # (..., K) Euclidean grad wrt the scale b

        *,
        eps:        float = 1e-6,
    ) -> Tuple[torch.Tensor, torch.Tensor]:   # (nat_mu, nat_b) = Fisher^{-1} grad
        r"""Diagonal-Laplace Fisher preconditioner. The Laplace(mu, b) Fisher information is
        DIAGONAL and EQUAL on both coordinates, ``I_mu = I_b = 1 / b^2`` (a location-scale family;
        verified symbolically), so the natural gradient is ``b^2 * grad`` on BOTH the location and
        the scale. This differs from the Gaussian preconditioner ``(sigma*grad_mu, 2 sigma^2
        grad_sigma)``: routing the Laplace belief through the Gaussian Fisher mis-scales the mean by
        a state-dependent ``1/b`` (a wrong direction on the product manifold, not a rescaled LR).
        ``b^2 > 0`` strictly, so the step is sign-preserving and ``grad=0 -> step=0`` (every
        stationary point of F preserved). ``eps`` floors ``b`` (the belief sigma floor, ``cfg.eps``)
        exactly as the divergence does."""
        orig_dtype = self.sigma.dtype
        with torch.amp.autocast('cuda', enabled=False):
            b2        = self.sigma.float().clamp(min=eps).square()      # (..., K) b^2 = I^{-1}
            nat_mu    = b2 * grad_mu.float()
            nat_sigma = b2 * grad_sigma.float()
        return nat_mu.to(orig_dtype), nat_sigma.to(orig_dtype)

    def _renyi_terms(
        self,
        other:  "DiagonalLaplace",

        *,
        alpha:  float,
        eps:    float,
    ) -> torch.Tensor:                                 # (..., K) UNSUMMED per-coordinate divergence
        r"""Per-coordinate Renyi/KL D^(k)(self || other), unclamped. KL (|alpha-1|<1e-6) uses the
        exact discrete form in float32; alpha != 1 uses the float64 integral form (covers the
        per-coordinate removable singularity at c_p==c_q and the alpha->1 outer cancellation)."""
        mu_q = self.mu.float()
        b_q = self.sigma.float().clamp(min=eps)
        mu_p = other.mu.float()
        b_p = other.sigma.float().clamp(min=eps)
        s = (mu_q - mu_p).abs()
        if abs(alpha - 1.0) < 1e-6:
            return torch.log(b_p) - torch.log(b_q) + s / b_p + (b_q / b_p) * torch.exp(-s / b_q) - 1.0
        # alpha != 1: float64 island (no byte-identity constraint for a new family).
        mu_q64 = mu_q.double()
        b_q64  = b_q.double()
        mu_p64 = mu_p.double()
        b_p64  = b_p.double()
        s64    = (mu_q64 - mu_p64).abs()
        c_q  = alpha / b_q64
        c_p  = (1.0 - alpha) / b_p64
        csum = c_q + c_p
        d    = c_p - c_q
        e_q  = torch.exp(-c_q * s64)
        e_p  = torch.exp(-c_p * s64)
        term1 = (e_p + e_q) / csum
        # term2 = (e_q - e_p) / (c_p - c_q). Near the removable singularity c_p == c_q this is a 0/0
        # whose VALUE float64 survives but whose GRADIENT it does not -- the quotient rule amplifies
        # the cancellation by a further 1/d, so a coordinate at b_q ~ b_p (every self-pair, where
        # omega_ii ~ I gives d ~ 1e-7 fp noise, AND any pair when renyi_order ~ 0.5) gets a spurious
        # belief gradient. Rewrite it EXACTLY as term2 = exp(-c_m s) * s * sinhc(d s / 2), where
        # c_m = (c_p + c_q)/2 and sinhc(x) = sinh(x)/x, and use the Taylor sinhc (1 + x^2/6 + x^4/120)
        # for small |x| -- stable in BOTH value and autograd through the singularity, with the direct
        # ratio kept for larger |x| (where sinh would overflow). x is zeroed off-branch so the dead
        # Taylor branch never evaluates a large argument. Pinned by the FD-of-F oracle gradient test.
        c_m   = 0.5 * (c_p + c_q)
        x     = 0.5 * d * s64                                       # d s / 2
        small = x.abs() < 1e-3
        x_s   = torch.where(small, x, torch.zeros_like(x))          # Taylor sees only small x
        sinhc = 1.0 + x_s * x_s / 6.0 + x_s ** 4 / 120.0           # sinh(x)/x near 0
        safe_d = torch.where(small, torch.ones_like(d), d)
        term2 = torch.where(small,
                            torch.exp(-c_m * s64) * s64 * sinhc,
                            (e_q - e_p) / safe_d)
        integral = term1 + term2
        log_int = (torch.log(integral) - math.log(2.0)
                   - alpha * torch.log(b_q64) - (1.0 - alpha) * torch.log(b_p64))
        div = log_int / (alpha - 1.0)
        # alpha > 1 can drive the blend rate non-positive (integral diverges) -> NaN -> kl_max.
        div = torch.where(csum > 0.0, div, div.new_tensor(float("nan")))
        return div.to(mu_q.dtype)

    def renyi_closed_form(
        self,
        other:   "DiagonalLaplace",

        *,
        alpha:   float = 1.0,
        kl_max:  float = 100.0,
        eps:     float = 1e-6,
    ) -> torch.Tensor:
        r"""Closed-form factorized-Laplace Renyi/KL D_alpha(self || other), summed over coordinates.

        ``eps`` floors the scale ``b`` (the belief sigma floor, ``cfg.eps``). The summed scalar is
        clamped once (any NaN coordinate from a divergent alpha>1 blend -> kl_max for the whole pair,
        mirroring the Gaussian non-PD policy)."""
        terms = self._renyi_terms(other, alpha=alpha, eps=eps)        # (..., K)
        return safe_kl_clamp(terms.sum(dim=-1), kl_max=kl_max)

    def renyi_per_coord(
        self,
        other:   "DiagonalLaplace",

        *,
        alpha:   float = 1.0,
        kl_max:  float = 100.0,
        eps:     float = 1e-6,
    ) -> torch.Tensor:
        r"""Per-coordinate factorized-Laplace Renyi/KL: the unsummed (..., K) terms, each clamped
        independently. The factorized divergence decomposes coordinate-wise, so summing over k
        recovers ``renyi_closed_form`` when no coordinate saturates ``kl_max`` (the independent
        per-coordinate clamp is intended; only the recovery identity is conditional, as for the
        diagonal Gaussian)."""
        terms = self._renyi_terms(other, alpha=alpha, eps=eps)        # (..., K)
        return safe_kl_clamp(terms, kl_max=kl_max)
