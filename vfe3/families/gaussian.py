r"""Gaussian exponential families (diagonal and full covariance) for VFE_3.0.

The closed-form Renyi/KL kernels are ported verbatim from the legacy ``divergence.py``
moment forms, so the live numerics are byte-identical; natural-parameter and log-partition
maps are added so the generic Bregman/Renyi-from-A path can be pinned against them.
"""

import math
from typing import List, Optional, Tuple

import torch

from vfe3.families.base import (
    BeliefParams, register_family, safe_kl_clamp, _logdet_chol,
)
from vfe3.numerics import safe_cholesky


# fp32 catastrophic-cancellation band around the alpha->1 (KL) limit of the Renyi logdet term.
# The |alpha-1| < 1e-6 KL switch in each kernel is exact in float64, but in float32 the three
# nearly-equal logs in the logdet term cancel and are divided by a tiny (alpha-1), losing ~1%
# accuracy out to roughly |alpha-1| ~ 1e-3. Inside this band the closed forms evaluate the logdet
# term in float64 (the same dtype-guarded headroom the geometry Gram pseudo-inverse already uses)
# and cast back; outside it the float32 quotient is accurate and used verbatim (byte-identical).
_RENYI_KL_BAND: float = 1e-2


def diag_kl_unclamped(
    mu_q:    torch.Tensor,             # (..., N, K) query means
    sigma_q: torch.Tensor,             # (..., N, K) query variances
    mu_p:    torch.Tensor,             # (..., N, K) prior means
    sigma_p: torch.Tensor,             # (..., N, K) prior variances

    *,
    eps:     float = 1e-6,
) -> torch.Tensor:                     # (..., N) UNCLAMPED KL(q||p)
    r"""Unclamped diagonal KL(q||p) = 0.5 Sum_k (s_k/t_k + (mu_p - mu_q)^2/t_k - 1 + log(t_k/s_k)).

    Returns the raw divergence scalar (summed over the last coordinate axis) WITHOUT clamping,
    so callers that need the pre-clamp value for saturation masking (e.g. ``_diag_kl_filtering_kernel``)
    can inspect it before applying ``safe_kl_clamp``.
    """
    sq     = sigma_q.clamp(min=eps)
    sp     = sigma_p.clamp(min=eps)
    K      = mu_q.shape[-1]
    trace  = (sq / sp).sum(dim=-1)
    mahal  = (((mu_p - mu_q) ** 2) / sp).sum(dim=-1)
    logdet = (torch.log(sp) - torch.log(sq)).sum(dim=-1)
    return 0.5 * (trace + mahal - K + logdet)


def diag_kl_unclamped_per_coord(
    mu_q:    torch.Tensor,             # (..., N, K) query means
    sigma_q: torch.Tensor,             # (..., N, K) query variances
    mu_p:    torch.Tensor,             # (..., N, K) prior means
    sigma_p: torch.Tensor,             # (..., N, K) prior variances

    *,
    eps:     float = 1e-6,
) -> torch.Tensor:                     # (..., N, K) UNCLAMPED per-coordinate KL D^(k)(q||p)
    r"""Unclamped per-coordinate diagonal KL D^(k) = 0.5 (s_k/t_k + (mu_p - mu_q)^2/t_k - 1 + log(t_k/s_k)).

    The per-coordinate analog of ``diag_kl_unclamped``: the -K of the summed form becomes -1 per
    coordinate.  Summing over k recovers ``diag_kl_unclamped`` (no coordinate saturates the clamp).
    """
    sq = sigma_q.clamp(min=eps)
    sp = sigma_p.clamp(min=eps)
    return 0.5 * (sq / sp + ((mu_p - mu_q) ** 2) / sp - 1.0 + torch.log(sp) - torch.log(sq))


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
        ``divergence._gaussian_diagonal_renyi``; mu_q=self, mu_t=other).

        ``eps`` is the variance floor for the log/division terms and MUST equal the upstream belief
        sigma floor (``cfg.eps``, default 1e-6; the SPD retraction clamps every sigma to
        [eps, sigma_max], so on the live pipeline every input sigma is already >= cfg.eps and this
        clamp is inert defense-in-depth, never the binding floor). This is a DIFFERENT, looser floor
        than the hardcoded 1e-12 in ``natural()``/``entropy()`` (the generic A-path): the closed form
        tracks the pipeline floor, while the natural-parameter maps only need positivity for
        log(-2 t2)."""
        K = self.mu.shape[-1]
        compute_dtype = (torch.float64
                         if torch.float64 in (self.mu.dtype, self.sigma.dtype,
                                              other.mu.dtype, other.sigma.dtype)
                         else torch.float32)
        mu_q = self.mu.to(compute_dtype)
        sigma_q = self.sigma.to(compute_dtype).clamp(min=eps)
        mu_t = other.mu.to(compute_dtype)
        sigma_t = other.sigma.to(compute_dtype).clamp(min=eps)
        if abs(alpha - 1.0) < 1e-6:
            trace_term  = (sigma_q / sigma_t).sum(dim=-1)
            delta       = mu_t - mu_q
            mahal_term  = ((delta ** 2) / sigma_t).sum(dim=-1)
            logdet_term = (torch.log(sigma_t) - torch.log(sigma_q)).sum(dim=-1)
            div = 0.5 * (trace_term + mahal_term - K + logdet_term)
        else:
            # alpha in (0,1): the blend is a convex combination of positive variances, so it is
            # always > 0 and the mask below is inert (byte-identical to the old clamp-only path).
            # alpha > 1 leaves the convex regime: a coordinate's blend can go non-positive, which
            # makes the divergence undefined. clamp(min=eps) here only guards log/division on the
            # GOOD coordinates; a non-PD coordinate maps the whole pair to NaN -> kl_max below,
            # mirroring the full-cov safe_cholesky mask (instead of emitting a wrong finite value).
            raw_blend   = (1.0 - alpha) * sigma_q + alpha * sigma_t
            sigma_blend = raw_blend.clamp(min=eps)
            delta       = mu_t - mu_q
            mahal_term  = (alpha * (delta ** 2) / sigma_blend).sum(dim=-1)
            if abs(alpha - 1.0) < _RENYI_KL_BAND:
                # fp32 cancellation band: evaluate the logdet term in float64, then cast back.
                sq64 = sigma_q.double()
                st64 = sigma_t.double()
                sb64 = ((1.0 - alpha) * sq64 + alpha * st64).clamp(min=eps)
                logdet_term = (
                    ((1.0 - alpha) * torch.log(sq64) + alpha * torch.log(st64) - torch.log(sb64)).sum(dim=-1)
                    / (alpha - 1.0)
                ).to(sigma_q.dtype)
            else:
                logdet_per_dim = (
                    (1.0 - alpha) * torch.log(sigma_q)
                    + alpha * torch.log(sigma_t)
                    - torch.log(sigma_blend)
                )
                logdet_term = logdet_per_dim.sum(dim=-1) / (alpha - 1.0)
            div = 0.5 * (mahal_term + logdet_term)
            ok  = (raw_blend > 0.0).all(dim=-1)            # any non-PD coordinate -> NaN -> kl_max
            div = torch.where(ok, div, div.new_tensor(float("nan")))
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
        left UNSUMMED, each clamped independently. Summing over k recovers ``renyi_closed_form``
        ONLY when no per-coordinate term saturates ``kl_max``: this routine clamps each coordinate
        to kl_max independently, whereas the closed form clamps the single summed scalar once, so a
        coordinate that individually exceeds kl_max diverges between the two. The independent
        per-coordinate clamp is the intended behavior here; only the recovery identity is conditional."""
        compute_dtype = (torch.float64
                         if torch.float64 in (self.mu.dtype, self.sigma.dtype,
                                              other.mu.dtype, other.sigma.dtype)
                         else torch.float32)
        mu_q = self.mu.to(compute_dtype)
        sigma_q = self.sigma.to(compute_dtype).clamp(min=eps)
        mu_t = other.mu.to(compute_dtype)
        sigma_t = other.sigma.to(compute_dtype).clamp(min=eps)
        delta = mu_t - mu_q
        if abs(alpha - 1.0) < 1e-6:
            per_coord = 0.5 * (
                sigma_q / sigma_t + (delta ** 2) / sigma_t - 1.0
                + torch.log(sigma_t) - torch.log(sigma_q)
            )
        else:
            # See renyi_closed_form: alpha in (0,1) blend is always > 0 (mask inert, byte-identical);
            # for alpha > 1 a non-positive-blend coordinate is masked to NaN -> kl_max PER COORDINATE
            # (the per-coord twin of the summed mask), so a bad coordinate is gated without killing
            # its in-bounds neighbours.
            raw_blend   = (1.0 - alpha) * sigma_q + alpha * sigma_t
            sigma_blend = raw_blend.clamp(min=eps)
            mahal       = alpha * (delta ** 2) / sigma_blend
            if abs(alpha - 1.0) < _RENYI_KL_BAND:
                # fp32 cancellation band: evaluate the per-coord logdet term in float64 (see
                # renyi_closed_form / _RENYI_KL_BAND), then cast back.
                sq64   = sigma_q.double()
                st64   = sigma_t.double()
                sb64   = ((1.0 - alpha) * sq64 + alpha * st64).clamp(min=eps)
                logdet = (
                    ((1.0 - alpha) * torch.log(sq64) + alpha * torch.log(st64) - torch.log(sb64))
                    / (alpha - 1.0)
                ).to(sigma_q.dtype)
            else:
                logdet  = (
                    (1.0 - alpha) * torch.log(sigma_q)
                    + alpha * torch.log(sigma_t)
                    - torch.log(sigma_blend)
                ) / (alpha - 1.0)
            per_coord = 0.5 * (mahal + logdet)
            per_coord = torch.where(raw_blend > 0.0, per_coord, per_coord.new_tensor(float("nan")))
        return safe_kl_clamp(per_coord, kl_max=kl_max)

    def natural_gradient(
        self,
        grad_mu:    torch.Tensor,                    # (..., K) Euclidean grad wrt mu
        grad_sigma: Optional[torch.Tensor],          # None freezes sigma; else grad wrt variance

        *,
        eps:        float = 1e-6,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""Diagonal-Gaussian Fisher preconditioner ``(sigma*grad_mu, 2 sigma^2 grad_sigma)``.
        Delegates to the pinned geometry kernel so the live numerics stay byte-identical to the
        golden-tested ``retraction.natural_gradient`` (local import avoids a families<-geometry
        import edge)."""
        from vfe3.geometry.retraction import natural_gradient as _gaussian_natural_gradient
        return _gaussian_natural_gradient(grad_mu, grad_sigma, self.sigma, eps=eps)


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
        # 1e-6 jitter matches the eps the closed-form full kernel uses for SPD safety. NOT routed
        # through ``safe_spd_inverse``: that helper's Cholesky-inverse diverges from this LU solve by
        # ~1.5e-4 once cond(Sigma) >= 1e3 (reachable here, eps=1e-6 floor / sigma_max=5 cap give
        # cond up to ~5e6), so centralizing would CHANGE the live result -- pinned by
        # tests/test_fix_numerics_audit.py::test_natural_inverse_not_routed_solve_vs_safe_spd_diverges.
        eye = torch.eye(self.mu.shape[-1], device=self.mu.device, dtype=self.mu.dtype)
        prec = torch.linalg.solve(self.sigma + 1e-6 * eye, eye.expand_as(self.sigma))
        t1 = (prec @ self.mu.unsqueeze(-1)).squeeze(-1)
        return (t1, -0.5 * prec)

    @classmethod
    def log_partition_at(cls, theta: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        t1, t2 = theta
        neg2t2 = -2.0 * t2
        L, ok = safe_cholesky(neg2t2, rounds=5)        # never raises; ok=False on a non-PD -2*t2
        inv_neg2t2 = torch.cholesky_inverse(L)
        quad = (t1.unsqueeze(-2) @ inv_neg2t2 @ t1.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        out = 0.5 * quad - 0.5 * _logdet_chol(L)
        return torch.where(ok, out, out.new_tensor(float("nan")))

    def expected_statistic(self) -> Tuple[torch.Tensor, torch.Tensor]:
        outer = self.mu.unsqueeze(-1) * self.mu.unsqueeze(-2)
        return (self.mu, self.sigma + outer)

    def entropy(self) -> torch.Tensor:
        K = self.mu.shape[-1]
        L, ok = safe_cholesky(self.sigma, rounds=5)
        entropy = 0.5 * _logdet_chol(L) + 0.5 * K * math.log(2.0 * math.pi * math.e)
        return torch.where(ok, entropy, entropy.new_tensor(float("nan")))

    def natural_gradient(
        self,
        grad_mu:    torch.Tensor,                    # (..., K) Euclidean grad wrt mu
        grad_sigma: Optional[torch.Tensor],          # None freezes sigma; else grad wrt covariance

        *,
        eps:        float = 1e-6,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""Full-Gaussian Fisher preconditioner ``(Sigma grad_mu, 2 Sigma grad_sigma Sigma)``
        (symmetrized). Delegates to the pinned geometry kernel, which selects the full-covariance
        branch by rank (``sigma.dim() == grad_mu.dim() + 1``), so the numerics stay byte-identical
        to the golden-tested ``retraction.natural_gradient``."""
        from vfe3.geometry.retraction import natural_gradient as _gaussian_natural_gradient
        return _gaussian_natural_gradient(grad_mu, grad_sigma, self.sigma, eps=eps)

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
        # float64 computes in float64 (audit 2026-07-12 N11, the F12 dtype policy: an fp64 island
        # is preserved end to end instead of silently collapsing to fp32 precision -- measured ~4%
        # relative error at cond(Sigma)~1e6); half/fp32 keep the existing fp32 compute.
        compute_dtype = (torch.float64
                         if torch.float64 in (self.mu.dtype, self.sigma.dtype,
                                              other.mu.dtype, other.sigma.dtype)
                         else torch.float32)
        mu_q = self.mu.to(compute_dtype)
        sigma_q = self.sigma.to(compute_dtype)
        mu_t = other.mu.to(compute_dtype)
        sigma_t = other.sigma.to(compute_dtype)
        # NO unconditional eps ridge (audit 2026-07-05 m1): the old ``sigma + eps*eye`` scored
        # N(mu, Sigma + eps I) ALWAYS -- a standing 1e-6 bias in values and gradients on the pure
        # full-covariance path, unlike the diagonal path's clamp (inert whenever the SPD retraction
        # keeps sigma >= eps). Robustness against a numerically non-PD covariance is owned by
        # safe_cholesky below: round 0 adds ZERO jitter (valid-SPD inputs are byte-identical to
        # torch.linalg.cholesky), failures get an escalating eps ridge, and a pair failing every
        # round -> NaN -> safe_kl_clamp -> kl_max.
        if abs(alpha - 1.0) < 1e-6:
            # safe_cholesky (jittered cholesky_ex, never raises) hardens the KL against a
            # numerically non-PD prior/posterior covariance -- reachable once training shifts the
            # prior, and routed here for any full-cov / non-kernel config via the E-step oracle.
            # This matches the robustness the alpha != 1 branch already has. Round 0 adds zero
            # jitter, so valid-SPD inputs stay byte-identical to torch.linalg.cholesky; an element
            # that fails every round -> NaN -> safe_kl_clamp -> kl_max (mirroring that branch).
            L_p, ok_p = safe_cholesky(sigma_t, eps=eps, rounds=5)
            L_q, ok_q = safe_cholesky(sigma_q, eps=eps, rounds=5)
            Y = torch.linalg.solve_triangular(L_p, sigma_q, upper=False)
            Z = torch.linalg.solve_triangular(L_p.transpose(-1, -2), Y, upper=True)
            trace_term = torch.diagonal(Z, dim1=-2, dim2=-1).sum(dim=-1)
            delta_mu = mu_t - mu_q
            v = torch.linalg.solve_triangular(
                L_p, delta_mu.unsqueeze(-1), upper=False
            ).squeeze(-1)
            mahal_term = (v ** 2).sum(dim=-1)
            logdet_p = _logdet_chol(L_p)
            logdet_q = _logdet_chol(L_q)
            div = 0.5 * (trace_term + mahal_term - K + logdet_p - logdet_q)
            div = torch.where(ok_p & ok_q, div, div.new_tensor(float("nan")))
        else:
            # alpha > 1 leaves the convex regime: the blend can be indefinite for some
            # (i,j) pairs. safe_cholesky factors per element (cholesky_ex, never raises),
            # tries an escalating eps ridge on failures, and returns an `ok` mask; a pair
            # that fails ALL rounds is set to NaN so safe_kl_clamp maps it to kl_max while
            # good pairs in the same batch keep their finite divergence. Round 0 adds zero
            # extra jitter, so valid-SPD inputs stay byte-identical to torch.linalg.cholesky.
            sigma_blend = (1.0 - alpha) * sigma_q + alpha * sigma_t
            sigma_blend = 0.5 * (sigma_blend + sigma_blend.transpose(-1, -2))
            L_blend, _ = safe_cholesky(sigma_blend, eps=eps, rounds=5)  # factor for mahal_term only
            L_q, ok_q = safe_cholesky(sigma_q, eps=eps, rounds=5)
            L_t, ok_t = safe_cholesky(sigma_t, eps=eps, rounds=5)
            delta_mu = mu_t - mu_q
            v = torch.linalg.solve_triangular(
                L_blend, delta_mu.unsqueeze(-1), upper=False
            ).squeeze(-1)
            mahal_term = alpha * (v ** 2).sum(dim=-1)
            if abs(alpha - 1.0) < _RENYI_KL_BAND:
                # fp32 cancellation band: the three logdets nearly cancel before the /(alpha-1).
                # Recompute them in float64 via slogdet on the f64 regularized covariances; the
                # fp32 cholesky factors above still drive mahal_term and the ok mask.
                sq64 = sigma_q.double()
                st64 = sigma_t.double()
                sb64 = (1.0 - alpha) * sq64 + alpha * st64
                sb64 = 0.5 * (sb64 + sb64.transpose(-1, -2))
                logdet_term = (
                    ((1.0 - alpha) * torch.linalg.slogdet(sq64).logabsdet
                     + alpha * torch.linalg.slogdet(st64).logabsdet
                     - torch.linalg.slogdet(sb64).logabsdet)
                    / (alpha - 1.0)
                ).to(sigma_q.dtype)
            else:
                logdet_q = _logdet_chol(L_q)
                logdet_t = _logdet_chol(L_t)
                logdet_blend = _logdet_chol(L_blend)
                logdet_term = (
                    (1.0 - alpha) * logdet_q + alpha * logdet_t - logdet_blend
                ) / (alpha - 1.0)
            div = 0.5 * (mahal_term + logdet_term)
            # safe_cholesky's escalating ridge can factor an INDEFINITE blend as PD (its ok mask
            # True) even though alpha>1 left the convex regime and the Renyi divergence is undefined
            # there -- the fp64 slogdet then drops the sign and the value collapses to ~0 instead of
            # kl_max. Gate on the SIGN of the (symmetrized) blend spectrum so a non-PD blend -> NaN
            # -> kl_max regardless of the ridge (audit 2026-06-17). sigma_q/sigma_t are SPD on the
            # live pipeline (SPD retraction); safe_cholesky owns the off-pipeline non-PD handling.
            blend_pd = torch.linalg.eigvalsh(sigma_blend)[..., 0] > 0   # smallest eigenvalue > 0
            ok = blend_pd & ok_q & ok_t            # non-PD blend or failed factor -> NaN -> kl_max
            div = torch.where(ok, div, div.new_tensor(float("nan")))
        return safe_kl_clamp(div, kl_max=kl_max)
