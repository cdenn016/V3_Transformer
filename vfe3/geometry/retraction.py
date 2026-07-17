r"""SPD-manifold retractions + Fisher natural-gradient preconditioner (VFE_3.0).

The SPD retraction keeps Sigma on the SPD manifold under a tangent update; the
Fisher preconditioner converts Euclidean (mu, sigma) gradients to natural
gradients. The phi Lie-algebra retraction (retract_phi) lives here too, dispatching to the
GL(K)/SO(N) retractions in lie_ops.py.
"""

import math
from typing import Callable, Dict, Optional, Tuple

import torch

from vfe3.geometry.groups import GaugeGroup
from vfe3.geometry.lie_ops import (
    clamp_phi_trace,
    project_phi_to_slk,
    retract_glk,
    retract_son,
)


def _check_sigma_max(sigma_max: Optional[float], eps: float) -> None:
    r"""Reject an eigenvalue ceiling that would violate the SPD/eps invariant."""
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"eps must be finite and positive; got {eps!r}")
    if sigma_max is None:
        return
    if not math.isfinite(sigma_max) or sigma_max <= eps:
        raise ValueError(
            f"sigma_max must be None or finite and > eps ({eps}); got {sigma_max!r}"
        )


def _public_spd_bounds(
    dtype:     torch.dtype,
    eps:       float,
    sigma_max: Optional[float],
) -> Tuple[float, Optional[float]]:
    """Return strict representable public-dtype bounds or reject an empty interval."""
    _check_sigma_max(sigma_max, eps)
    lower_tensor = torch.tensor(float(eps), dtype=dtype)
    lower_tensor = torch.nextafter(lower_tensor, torch.full_like(lower_tensor, float("inf")))
    lower = float(lower_tensor)
    if not math.isfinite(lower) or lower <= float(eps):
        raise ValueError(f"eps={eps} has no finite representable strict upper neighbor in {dtype}")
    if sigma_max is None:
        return lower, None
    upper_tensor = torch.tensor(float(sigma_max), dtype=dtype)
    upper_tensor = torch.nextafter(upper_tensor, torch.full_like(upper_tensor, -float("inf")))
    upper = float(upper_tensor)
    if not math.isfinite(upper) or upper <= lower:
        raise ValueError(
            f"[{eps}, {sigma_max}] has no representable strict interior in {dtype}"
        )
    return lower, upper


_RETRACTIONS: Dict[str, Callable[..., torch.Tensor]] = {}


def register_retraction(name: str, *, override: bool = False) -> Callable:
    """Decorator registering an SPD covariance retraction sigma -> sigma_new.

    Duplicate keys fail closed (audit 2026-07-01 round-3): a second registration under an
    existing name silently shadowed the first. Pass ``override=True`` to replace deliberately.
    """
    def _wrap(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        if name in _RETRACTIONS and not override:
            raise KeyError(f"retraction {name!r} already registered; pass override=True to replace")
        _RETRACTIONS[name] = fn
        return fn
    return _wrap


def get_retraction(name: str) -> Callable[..., torch.Tensor]:
    """Return the registered SPD retraction (KeyError-with-available-list if absent)."""
    if name not in _RETRACTIONS:
        raise KeyError(f"no retraction {name!r}; available: {sorted(_RETRACTIONS)}")
    return _RETRACTIONS[name]


class _EighDamped(torch.autograd.Function):
    r"""``torch.linalg.eigh`` with a gap-regularized (Lorentzian-damped) backward.

    The symmetric-eigendecomposition adjoint carries ``1/(lambda_i - lambda_j)`` gap terms that
    diverge on repeated eigenvalues (Higham, *Functions of Matrices* 2008, Sec 3.2). At a degenerate
    spectrum -- e.g. the isotropic ``Sigma = I`` that is the default ``gaussian_full`` prior init --
    the eigenvectors are arbitrary but the downstream function ``V f(lambda) V^T`` (matrix
    sqrt / inv-sqrt / log / exp here) is smooth, so the *true* gradient is finite; only the
    eigendecomposition's intermediate adjoint blows up, poisoning the whole backward with NaN.

    Replacing ``1/Delta`` by the Lorentzian ``Delta / (Delta^2 + gap_eps)`` leaves a well-separated
    spectrum unchanged (relative error ~ ``gap_eps / Delta^2`` for gaps ``Delta``) while damping the
    degenerate gap term to 0 instead of +/-inf -- it kills the (physically immaterial) gradient
    component within a degenerate eigenspace rather than emitting NaN. Forward is bit-identical to
    ``torch.linalg.eigh`` (only the backward is regularized), so every forward-value contract on the
    retractions is preserved. Validated against the stock ``eigh`` backward on well-separated spectra
    (tests/test_retraction.py).
    """

    @staticmethod
    def forward(ctx, A: torch.Tensor, gap_eps):             # gap_eps: float or 0-d tensor  # noqa: D401
        w, V = torch.linalg.eigh(A)
        ctx.save_for_backward(w, V)
        ctx.gap_eps = gap_eps
        return w, V

    @staticmethod
    def backward(ctx, gw: Optional[torch.Tensor], gV: Optional[torch.Tensor]):
        w, V = ctx.saved_tensors
        Vt = V.transpose(-1, -2)
        # delta_ij = w_j - w_i (the symmetric-eigh adjoint's F_ij = 1/(w_j - w_i); the j index is the
        # eigenvector being perturbed). The diagonal (w_i - w_i = 0) gives F_ii = 0 for gap_eps > 0.
        delta = w.unsqueeze(-2) - w.unsqueeze(-1)            # (..., n, n), delta_ij = w_j - w_i
        F = delta / (delta * delta + ctx.gap_eps)           # Lorentzian-damped 1/(w_j - w_i)
        inner = F * (Vt @ gV) if gV is not None else torch.zeros_like(V)
        if gw is not None:
            inner = inner + torch.diag_embed(gw)
        gA = V @ inner @ Vt
        gA = 0.5 * (gA + gA.transpose(-1, -2))              # A is symmetric -> symmetric cotangent
        return gA, None


def _eigh_damped(A: torch.Tensor, gap_eps) -> Tuple[torch.Tensor, torch.Tensor]:  # gap_eps: float or 0-d tensor
    r"""``(eigenvalues, eigenvectors)`` of symmetric ``A`` with a gap-regularized backward (see
    :class:`_EighDamped`). Drop-in for ``torch.linalg.eigh`` on the full-cov SPD retraction paths so a
    degenerate spectrum (the ``Sigma = I`` default init) yields finite -- not NaN -- gradients on the
    unrolled E-step. ``gap_eps`` bounds the worst-case gap factor at ``1/(2 sqrt(gap_eps))``; the
    retraction call sites pass a spectrum-relative value via :func:`_rel_gap_eps`."""
    return _EighDamped.apply(A, gap_eps)


def _rel_gap_eps(
    A:     torch.Tensor,

    *,
    rel:    Optional[float] = None,
    floor:  Optional[float] = None,
) -> torch.Tensor:                         # (..., 1, 1) on-device tensor; no host-sync
    r"""Spectrum-relative ``gap_eps`` for :func:`_eigh_damped` on the SPD retraction paths (audit
    2026-06-13 L11). The fixed ``gap_eps=1e-8`` over-damps the eigh adjoint ``F_ij = 1/(w_j - w_i)``
    for MEANINGFUL gaps near the variance floor -- a resolvable gap of 1e-4 is biased ~50%. Scaling
    to ``(rel * ||A||_max)^2`` -- the squared fp32 noise floor of the spectrum -- damps only gaps
    below fp32 resolution (true degeneracy stays finite, ``F = 0``) and leaves resolvable gaps
    accurate. When omitted, ``rel`` is derived from the active dtype; ``finfo.tiny`` keeps a
    near-zero spectrum finite without imposing a coordinate-scale floor. Passing an explicit
    nonnegative ``floor`` preserves the diagnostic/test API; ``floor=0.0`` exposes the un-clamped
    relative value. Each matrix receives its own scale so unrelated batch elements cannot change
    its eigendecomposition adjoint. Returns ``(..., 1, 1)`` on A's device/dtype to avoid a CUDA
    host-sync."""
    dtype_info = torch.finfo(A.dtype)
    relative = 8.0 * dtype_info.eps if rel is None else rel
    if not math.isfinite(relative) or relative <= 0.0:
        raise ValueError(f"rel must be None or finite and positive, got {rel!r}")
    minimum = dtype_info.tiny if floor is None else floor
    if not math.isfinite(minimum) or minimum < 0.0:
        raise ValueError(f"floor must be None or finite and nonnegative, got {floor!r}")
    scale = A.detach().abs().amax(dim=(-2, -1), keepdim=True)
    return (relative * scale).pow(2).clamp(min=minimum)


def _spectral_values_and_derivatives(
    eigenvalues: torch.Tensor,
    kind:        str,
    lower:       float,
    upper:       Optional[float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Return ``f(lambda)`` and ``f'(lambda)`` for one bounded scalar matrix function."""
    if kind == "sqrt_floor":
        bounded = eigenvalues.clamp(min=lower)
        values = torch.sqrt(bounded)
        derivatives = torch.where(
            eigenvalues >= lower,
            0.5 / values,
            torch.zeros_like(values),
        )
    elif kind == "inv_sqrt_floor":
        bounded = eigenvalues.clamp(min=lower)
        values = torch.rsqrt(bounded)
        derivatives = torch.where(
            eigenvalues >= lower,
            -0.5 * values / bounded,
            torch.zeros_like(values),
        )
    elif kind == "exp_bounded":
        if upper is None:
            raise ValueError("exp_bounded requires an upper bound")
        bounded = eigenvalues.clamp(min=lower, max=upper)
        values = torch.exp(bounded)
        active = (eigenvalues >= lower) & (eigenvalues <= upper)
        derivatives = torch.where(active, values, torch.zeros_like(values))
    elif kind == "project":
        values = eigenvalues.clamp(min=lower) if upper is None else eigenvalues.clamp(
            min=lower,
            max=upper,
        )
        active = eigenvalues >= lower
        if upper is not None:
            active = active & (eigenvalues <= upper)
        derivatives = active.to(eigenvalues.dtype)
    else:
        raise ValueError(f"unknown symmetric spectral function {kind!r}")
    return values, derivatives


class _SymmetricSpectralMap(torch.autograd.Function):
    r"""Symmetric matrix function with Loewner divided differences in the adjoint."""

    @staticmethod
    def forward(
        ctx,
        matrix: torch.Tensor,
        kind:   str,
        lower:  float,
        upper:  Optional[float],
    ) -> torch.Tensor:
        symmetric = 0.5 * (matrix + matrix.transpose(-1, -2))
        eigenvalues, eigenvectors = _eigh_damped(symmetric, _rel_gap_eps(symmetric))
        values, _ = _spectral_values_and_derivatives(eigenvalues, kind, lower, upper)
        ctx.save_for_backward(eigenvalues, eigenvectors)
        ctx.kind = kind
        ctx.lower = lower
        ctx.upper = upper
        return eigenvectors * values.unsqueeze(-2) @ eigenvectors.transpose(-1, -2)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        eigenvalues, eigenvectors = ctx.saved_tensors
        values, _ = _spectral_values_and_derivatives(
            eigenvalues,
            ctx.kind,
            ctx.lower,
            ctx.upper,
        )
        lambda_i = eigenvalues.unsqueeze(-1)
        lambda_j = eigenvalues.unsqueeze(-2)
        gap = lambda_i - lambda_j
        value_gap = values.unsqueeze(-1) - values.unsqueeze(-2)
        scale = torch.maximum(lambda_i.abs(), lambda_j.abs())
        resolution = (
            8.0 * torch.finfo(eigenvalues.dtype).eps * scale
        ).clamp(min=torch.finfo(eigenvalues.dtype).tiny)
        repeated = gap.abs() <= resolution
        safe_gap = torch.where(repeated, torch.ones_like(gap), gap)
        divided_difference = value_gap / safe_gap
        midpoint = 0.5 * (lambda_i + lambda_j)
        _, repeated_limit = _spectral_values_and_derivatives(
            midpoint,
            ctx.kind,
            ctx.lower,
            ctx.upper,
        )
        divided_difference = torch.where(repeated, repeated_limit, divided_difference)

        eigenvectors_t = eigenvectors.transpose(-1, -2)
        symmetric_grad = 0.5 * (grad_output + grad_output.transpose(-1, -2))
        inner = eigenvectors_t @ symmetric_grad @ eigenvectors
        grad_matrix = eigenvectors @ (divided_difference * inner) @ eigenvectors_t
        grad_matrix = 0.5 * (grad_matrix + grad_matrix.transpose(-1, -2))
        return grad_matrix, None, None, None


def _symmetric_spectral_map(
    matrix: torch.Tensor,
    kind:   str,

    *,
    lower:  float,
    upper:  Optional[float] = None,
) -> torch.Tensor:
    return _SymmetricSpectralMap.apply(matrix, kind, lower, upper)


def _frechet_log_spd(
    sigma:   torch.Tensor,                  # (..., K, K) SPD base point
    tangent: torch.Tensor,                  # (..., K, K) symmetric ambient tangent

    *,
    eps:     float = 1e-6,
    eig:     Optional[Tuple[torch.Tensor, torch.Tensor]] = None,   # precomputed _eigh_damped of the
                                            # SYMMETRIZED sigma, PRE-clamp (audit 2026-07-12 N9:
                                            # lets retract_logeuclidean_full reuse its own eigh
                                            # instead of decomposing the identical matrix twice)
) -> torch.Tensor:
    r"""Fréchet derivative of the matrix logarithm at ``sigma`` applied to ``tangent``.

    In the eigenbasis ``sigma = V diag(lambda) V^T``,

        D log_sigma[H] = V (L odot (V^T H V)) V^T,

    where ``L_ii = 1 / lambda_i`` and
    ``L_ij = (log(lambda_i) - log(lambda_j)) / (lambda_i - lambda_j)``.
    The repeated-eigenvalue branch uses the continuous reciprocal-mean limit.
    """
    tangent = 0.5 * (tangent + tangent.transpose(-1, -2))

    if eig is None:
        sigma = 0.5 * (sigma + sigma.transpose(-1, -2))
        eigenvalues, eigenvectors = _eigh_damped(sigma, _rel_gap_eps(sigma))
    else:
        eigenvalues, eigenvectors = eig
    eigenvalues = eigenvalues.clamp(min=eps)
    log_eigenvalues = torch.log(eigenvalues)

    lambda_i = eigenvalues.unsqueeze(-1)
    lambda_j = eigenvalues.unsqueeze(-2)
    gap = lambda_i - lambda_j
    gap_scale = torch.maximum(lambda_i, lambda_j)
    resolution = 8.0 * torch.finfo(eigenvalues.dtype).eps * gap_scale
    near_repeated = gap.abs() <= resolution
    safe_gap = torch.where(near_repeated, torch.ones_like(gap), gap)
    divided_difference = (
        log_eigenvalues.unsqueeze(-1) - log_eigenvalues.unsqueeze(-2)
    ) / safe_gap
    repeated_limit = 2.0 / (lambda_i + lambda_j)
    divided_difference = torch.where(near_repeated, repeated_limit, divided_difference)

    eigenvectors_t = eigenvectors.transpose(-1, -2)
    tangent_eigenbasis = eigenvectors_t @ tangent @ eigenvectors
    chart_tangent = eigenvectors @ (divided_difference * tangent_eigenbasis) @ eigenvectors_t
    return 0.5 * (chart_tangent + chart_tangent.transpose(-1, -2))


def _certify_public_spd(
    matrix: torch.Tensor,                  # (..., K, K) projected covariance in its public dtype

    *,
    eps:       float,
    sigma_max: Optional[float] = None,
) -> torch.Tensor:
    r"""Certify the represented spectrum and repair only rows outside the public interval.

    Cholesky tests of ``Sigma - eps I`` and, when bounded, ``sigma_max I - Sigma`` certify strict
    represented-value headroom at both ends. The ordinary interior path performs no extra
    eigendecomposition and is byte-identical. A failing row alone is spectrally rebuilt with
    public-dtype roundoff clearance; a second certificate has an interior isotropic fallback.
    """
    fallback_value, _public_upper = _public_spd_bounds(matrix.dtype, eps, sigma_max)
    symmetric = 0.5 * (matrix + matrix.transpose(-1, -2))
    check_dtype = torch.float64 if symmetric.dtype == torch.float64 else torch.float32
    dimension = symmetric.shape[-1]
    eye_check = torch.eye(dimension, device=symmetric.device, dtype=check_dtype)
    flat = symmetric.reshape(-1, dimension, dimension)
    certificate = flat.to(check_dtype)
    _, lower_info = torch.linalg.cholesky_ex(
        certificate - float(eps) * eye_check,
        check_errors=False,
    )
    failed = lower_info.ne(0)
    if sigma_max is not None:
        _, upper_info = torch.linalg.cholesky_ex(
            float(sigma_max) * eye_check - certificate,
            check_errors=False,
        )
        failed = failed | upper_info.ne(0)
    if not bool(failed.any()):
        return symmetric

    repair_input = certificate[failed]
    eigenvalues, eigenvectors = _eigh_damped(repair_input, _rel_gap_eps(repair_input))
    scale = repair_input.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1.0)
    margin = scale * (8.0 * dimension * torch.finfo(symmetric.dtype).eps)
    margin_scalar = margin.squeeze(-1).squeeze(-1).unsqueeze(-1)
    if sigma_max is None:
        lower = torch.full_like(eigenvalues, float(eps)) + margin_scalar
        repaired_eigenvalues = torch.maximum(eigenvalues, lower)
        fallback_scale = torch.full_like(margin, fallback_value)
    else:
        interval = float(sigma_max) - float(eps)
        clearance = torch.minimum(
            margin_scalar,
            torch.full_like(margin_scalar, interval / 4.0),
        )
        lower = torch.full_like(eigenvalues, float(eps)) + clearance
        upper = torch.full_like(eigenvalues, float(sigma_max)) - clearance
        repaired_eigenvalues = torch.minimum(torch.maximum(eigenvalues, lower), upper)
        fallback_scale = torch.full_like(margin, fallback_value)
    repaired_check = (
        eigenvectors * repaired_eigenvalues.unsqueeze(-2)
        @ eigenvectors.transpose(-1, -2)
    )
    repaired = repaired_check.to(symmetric.dtype)
    repaired = 0.5 * (repaired + repaired.transpose(-1, -2))
    represented = repaired.to(check_dtype)
    _, repair_lower_info = torch.linalg.cholesky_ex(
        represented - float(eps) * eye_check,
        check_errors=False,
    )
    repair_failed = repair_lower_info.ne(0)
    if sigma_max is not None:
        _, repair_upper_info = torch.linalg.cholesky_ex(
            float(sigma_max) * eye_check - represented,
            check_errors=False,
        )
        repair_failed = repair_failed | repair_upper_info.ne(0)
    fallback = fallback_scale.to(symmetric.dtype) * torch.eye(
        dimension,
        device=symmetric.device,
        dtype=symmetric.dtype,
    )
    repaired = torch.where(
        repair_failed.unsqueeze(-1).unsqueeze(-1),
        fallback,
        repaired,
    )
    final_represented = repaired.to(check_dtype)
    _, final_lower_info = torch.linalg.cholesky_ex(
        final_represented - float(eps) * eye_check,
        check_errors=False,
    )
    final_failed = final_lower_info.ne(0)
    if sigma_max is not None:
        _, final_upper_info = torch.linalg.cholesky_ex(
            float(sigma_max) * eye_check - final_represented,
            check_errors=False,
        )
        final_failed = final_failed | final_upper_info.ne(0)
    if bool(final_failed.any()):
        raise ValueError(
            "the requested covariance interval has no certifiable interior fallback "
            f"in public dtype {symmetric.dtype}"
        )
    output = flat.clone()
    output[failed] = repaired
    return output.reshape_as(symmetric)


def retract_spd_diagonal(
    sigma_diag:   torch.Tensor,             # (..., K) diagonal variances
    delta_sigma:  torch.Tensor,             # (..., K) diagonal tangent

    *,
    step_size:    float          = 1.0,
    trust_region: float          = 5.0,
    eps:          float          = 1e-6,
    sigma_max:    Optional[float] = 10.0,   # matches VFE3Config.sigma_max
) -> torch.Tensor:
    r"""Diagonal SPD retraction sigma_new = sigma * exp(R), R = tau dsigma/sigma.

    ``trust_region`` projects the step-scaled whitened diagonal vector into one L2 ball. Its norm is
    exactly the Frobenius norm of the corresponding diagonal matrix, so the shared trust value has
    the same geometric meaning as the full-SPD arm. Positivity is by construction (exp > 0);
    clamped to [eps, sigma_max].
    When sigma_max is None the eigenvalue ceiling is skipped (pure-path: eps floor only).
    """
    orig_dtype = sigma_diag.dtype
    lower_bound, upper_bound = _public_spd_bounds(orig_dtype, eps, sigma_max)
    with torch.amp.autocast(sigma_diag.device.type, enabled=False):     # tensor-keyed (audit 2026-07-05 m10)
        # float64 stays float64 (audit 2026-07-12 N12, the retract_spd_full F12 policy); half promotes to fp32.
        compute_dtype = torch.float64 if orig_dtype == torch.float64 else torch.float32
        sigma_safe = sigma_diag.to(compute_dtype).clamp(min=eps)
        delta_sigma = delta_sigma.to(compute_dtype)
        whitened = delta_sigma / sigma_safe
        exp_arg = step_size * whitened
        if trust_region is not None and trust_region > 0:
            tangent_norm = torch.linalg.vector_norm(exp_arg, dim=-1, keepdim=True)
            exp_arg = exp_arg * torch.clamp(trust_region / (tangent_norm + eps), max=1.0)
        exp_arg = exp_arg.clamp(-50.0, 50.0)
        sigma_new = sigma_safe * torch.exp(exp_arg)
    sigma_new = (
        sigma_new.clamp(min=lower_bound)
        if upper_bound is None
        else sigma_new.clamp(min=lower_bound, max=upper_bound)
    )
    return sigma_new.to(orig_dtype)


def retract_spd_full(
    sigma:        torch.Tensor,             # (..., K, K) SPD covariances
    delta_sigma:  torch.Tensor,             # (..., K, K) symmetric tangent

    *,
    step_size:    float          = 1.0,
    trust_region: float          = 2.0,
    eps:          float          = 1e-6,
    sigma_max:    Optional[float] = 10.0,   # matches VFE3Config.sigma_max
) -> torch.Tensor:
    r"""Full SPD retraction via the affine-invariant exponential map.

        Sigma_new = S^{1/2} exp(S^{-1/2} (tau dSigma) S^{-1/2}) S^{1/2},
    with one Frobenius-ball trust region on the whole whitened tangent matrix and an eigenvalue
    floor/ceiling [eps, sigma_max] (eigenvalues are variances: the SAME physical ceiling the
    diagonal arm applies to sigma). Uses the gap-regularized ``_eigh_damped`` eigendecomposition so
    the unrolled backward stays finite on a degenerate spectrum -- the isotropic ``Sigma = I`` default
    gaussian_full init makes the stock eigh backward 100% NaN; forward values are unchanged.
    When sigma_max is None the eigenvalue ceiling is skipped (pure-path: eps floor only).
    """
    _check_sigma_max(sigma_max, eps)
    orig_shape = sigma.shape
    orig_dtype = sigma.dtype
    _public_spd_bounds(orig_dtype, eps, sigma_max)
    if sigma.dim() == 4:
        B, N, K, _ = sigma.shape
        sigma = sigma.reshape(B * N, K, K)
        delta_sigma = delta_sigma.reshape(B * N, K, K)

    with torch.amp.autocast(sigma.device.type, enabled=False):          # tensor-keyed (audit 2026-07-05 m10)
        compute_dtype = torch.float64 if orig_dtype == torch.float64 else torch.float32
        sigma = sigma.to(dtype=compute_dtype)
        delta_sigma = delta_sigma.to(dtype=compute_dtype)
        sigma = 0.5 * (sigma + sigma.transpose(-1, -2))
        delta_sigma = 0.5 * (delta_sigma + delta_sigma.transpose(-1, -2))

        sigma_sqrt = _symmetric_spectral_map(
            sigma,
            "sqrt_floor",
            lower=eps,
        )
        sigma_inv_sqrt = _symmetric_spectral_map(
            sigma,
            "inv_sqrt_floor",
            lower=eps,
        )

        R = sigma_inv_sqrt @ (step_size * delta_sigma) @ sigma_inv_sqrt
        R = 0.5 * (R + R.transpose(-1, -2))
        if trust_region is not None and trust_region > 0:
            R_norm = torch.linalg.norm(R, ord='fro', dim=(-2, -1), keepdim=True)
            R = R * torch.clamp(trust_region / (R_norm + eps), max=1.0)

        exp_R = _symmetric_spectral_map(
            R,
            "exp_bounded",
            lower=-50.0,
            upper=50.0,
        )

        sigma_new = sigma_sqrt @ exp_R @ sigma_sqrt
        sigma_new = 0.5 * (sigma_new + sigma_new.transpose(-1, -2))

        sigma_new = _symmetric_spectral_map(
            sigma_new,
            "project",
            lower=eps,
            upper=sigma_max,
        )

    sigma_new = _certify_public_spd(
        sigma_new.to(orig_dtype),
        eps=eps,
        sigma_max=sigma_max,
    )
    if len(orig_shape) == 4:
        sigma_new = sigma_new.reshape(orig_shape)
    return sigma_new


@register_retraction("spd_affine")
def retract_spd_affine(
    sigma:        torch.Tensor,             # (..., K) diagonal OR (..., K, K) full covariance
    delta_sigma:  torch.Tensor,             # matching rank: the tangent step (e.g. -e_q_sigma_lr * nat_sigma)

    mean_ndim:    int,                      # ndim of the belief mean; full cov iff sigma.dim() == mean_ndim + 1

    *,
    step_size:    float          = 1.0,
    trust_region: float          = 5.0,
    eps:          float          = 1e-6,
    sigma_max:    Optional[float] = 10.0,   # matches VFE3Config.sigma_max
) -> torch.Tensor:                          # (...) same rank as sigma
    r"""Affine-invariant SPD retraction (the manuscript-canonical default, GL(K)_supplementary.tex:640-645).

    The single registered home for the rank dispatch the E-step used to perform inline: a full
    covariance (sigma.dim() == mean_ndim + 1) steps along the affine-invariant geodesic via
    ``retract_spd_full``; a diagonal sigma (matching the mean rank) uses ``retract_spd_diagonal``.
    Both are the same affine-invariant exponential map,
        Sigma_new = Sigma^{1/2} exp(Sigma^{-1/2} (step_size dSigma) Sigma^{-1/2}) Sigma^{1/2},
    reduced to sigma_new = sigma exp(step_size dsigma/sigma) on the diagonal cone. The dispatcher
    forwards to the bare functions; the Fisher metric conversion stays in the E-step
    (``natural_gradient``), so the tangent ``delta_sigma`` arrives already preconditioned. The shared
    ``trust_region`` bounds the L2 norm of the step-scaled diagonal tangent and the Frobenius norm of
    the step-scaled full tangent, which are equal when the full tangent is diagonal.
    """
    if sigma.dim() == mean_ndim + 1:                     # full covariance (..., K, K)
        return retract_spd_full(
            sigma, delta_sigma, step_size=step_size, trust_region=trust_region, eps=eps, sigma_max=sigma_max,
        )
    return retract_spd_diagonal(                          # diagonal variances (..., K)
        sigma, delta_sigma, step_size=step_size, trust_region=trust_region, eps=eps, sigma_max=sigma_max,
    )


def retract_logeuclidean_full(
    sigma:        torch.Tensor,             # (..., K, K) SPD covariances
    delta_sigma:  torch.Tensor,             # (..., K, K) symmetric ambient tangent

    *,
    step_size:    float          = 1.0,
    trust_region: float          = 5.0,
    eps:          float          = 1e-6,
    sigma_max:    Optional[float] = 10.0,   # matches VFE3Config.sigma_max
) -> torch.Tensor:
    r"""Full log-Euclidean SPD retraction (Arsigny-Fillard-Pennec-Ayache 2006/2007).

        Sigma_new = expm( logm(Sigma) + step_size * Dlog_Sigma[delta_sigma] ),
    where logm/expm are the matrix log/exp in the eigenbasis (logm(Sigma) =
    V diag(log lambda_j) V^T; expm(M) = U diag(exp mu_j) U^T). Because logm(Sigma)
    is symmetric and expm of a symmetric matrix is SPD, this is SPD-preserving for
    ANY magnitude of the tangent (no trust region needed for positivity; the trust
    region is a Frobenius stability knob only). ``Dlog_Sigma`` converts the ambient
    covariance tangent into the Log-Euclidean chart, making the retraction's first
    derivative the identity. Input eigenvalues are floored at eps before log and the
    output spectrum is projected to [eps, sigma_max].
    """
    _check_sigma_max(sigma_max, eps)
    orig_shape = sigma.shape
    orig_dtype = sigma.dtype
    _public_spd_bounds(orig_dtype, eps, sigma_max)
    if sigma.dim() == 4:
        B, N, K, _ = sigma.shape
        sigma = sigma.reshape(B * N, K, K)
        delta_sigma = delta_sigma.reshape(B * N, K, K)

    with torch.amp.autocast(sigma.device.type, enabled=False):          # tensor-keyed (audit 2026-07-05 m10)
        compute_dtype = torch.float64 if orig_dtype == torch.float64 else torch.float32
        sigma = sigma.to(compute_dtype)
        delta_sigma = delta_sigma.to(compute_dtype)
        sigma = 0.5 * (sigma + sigma.transpose(-1, -2))
        delta_sigma = 0.5 * (delta_sigma + delta_sigma.transpose(-1, -2))

        eig_raw, eigenvectors = _eigh_damped(sigma, _rel_gap_eps(sigma))
        eigenvalues = eig_raw.clamp(min=eps)
        log_eig = torch.log(eigenvalues)
        log_sigma = eigenvectors * log_eig.unsqueeze(-2) @ eigenvectors.transpose(-1, -2)

        # Reuse this eigendecomposition for the Fréchet chart map (audit 2026-07-12 N9): sigma is
        # already symmetrized above and _frechet_log_spd applies the SAME eps clamp, so passing the
        # pre-clamp pair is byte-identical to its own eigh of the identical matrix.
        tangent = step_size * _frechet_log_spd(sigma, delta_sigma, eps=eps, eig=(eig_raw, eigenvectors))
        if trust_region is not None and trust_region > 0:                # clamp the TANGENT, not the
            t_norm  = torch.linalg.norm(tangent, ord='fro', dim=(-2, -1), keepdim=True)   # base point,
            tangent = tangent * torch.clamp(trust_region / (t_norm + eps), max=1.0)       # so R(S,0)=S.
        M = log_sigma + tangent
        M = 0.5 * (M + M.transpose(-1, -2))

        M_eval, M_evec = _eigh_damped(M, _rel_gap_eps(M))
        M_eval = M_eval.clamp(-50.0, 50.0)
        sigma_new = M_evec * torch.exp(M_eval).unsqueeze(-2) @ M_evec.transpose(-1, -2)
        sigma_new = 0.5 * (sigma_new + sigma_new.transpose(-1, -2))

        eig_new, vec_new = _eigh_damped(sigma_new, _rel_gap_eps(sigma_new))
        eig_new = eig_new.clamp(min=eps) if sigma_max is None else eig_new.clamp(min=eps, max=sigma_max)
        sigma_new = vec_new * eig_new.unsqueeze(-2) @ vec_new.transpose(-1, -2)

    sigma_new = _certify_public_spd(
        sigma_new.to(orig_dtype),
        eps=eps,
        sigma_max=sigma_max,
    )
    if len(orig_shape) == 4:
        sigma_new = sigma_new.reshape(orig_shape)
    return sigma_new


@register_retraction("log_euclidean")
def retract_log_euclidean(
    sigma:        torch.Tensor,             # (..., K) diagonal OR (..., K, K) full covariance
    delta_sigma:  torch.Tensor,             # matching-rank ambient covariance tangent

    mean_ndim:    int,                      # ndim of the belief mean; full cov iff sigma.dim() == mean_ndim + 1

    *,
    step_size:    float          = 1.0,
    trust_region: float          = 5.0,
    eps:          float          = 1e-6,
    sigma_max:    Optional[float] = 10.0,   # matches VFE3Config.sigma_max
) -> torch.Tensor:                          # (...) same rank as sigma
    r"""Log-Euclidean SPD retraction (spec reading 2a; Arsigny et al. 2006/2007).

    The ambient tangent ``delta_sigma`` is mapped into the matrix-log chart before the
    update closes in the SPD vector-space structure,
        Sigma_new = expm( logm(Sigma) + step_size * Dlog_Sigma[delta_sigma] ).
    SPD-preserving for ANY step magnitude (expm of a symmetric matrix is SPD), unlike a
    naive Euclidean step; the trust region is a stability knob, not a positivity guard.

    Full covariance (sigma.dim() == mean_ndim + 1) uses the two-eigh logm/expm in
    ``retract_logeuclidean_full``. The diagonal reduction uses
        Dlog_sigma[delta_sigma] = delta_sigma / sigma,
    so ``sigma_new = sigma * exp(step_size * delta_sigma / sigma)`` and the first
    derivative with respect to the ambient tangent is the identity.
    """
    _check_sigma_max(sigma_max, eps)
    if sigma.dim() == mean_ndim + 1:                     # full covariance (..., K, K)
        return retract_logeuclidean_full(
            sigma, delta_sigma, step_size=step_size, trust_region=trust_region, eps=eps, sigma_max=sigma_max,
        )
    # diagonal: Dlog_sigma[delta_sigma] = delta_sigma / sigma
    orig_dtype = sigma.dtype
    lower_bound, upper_bound = _public_spd_bounds(orig_dtype, eps, sigma_max)
    with torch.amp.autocast(sigma.device.type, enabled=False):          # tensor-keyed (audit 2026-07-05 m10)
        # float64 stays float64 (audit 2026-07-12 N12, matching the full arm above); half promotes to fp32.
        compute_dtype = torch.float64 if orig_dtype == torch.float64 else torch.float32
        sigma_safe  = sigma.to(compute_dtype).clamp(min=eps)
        delta_sigma = delta_sigma.to(compute_dtype) / sigma_safe
        if trust_region is not None and trust_region > 0:
            delta_sigma = delta_sigma.clamp(-trust_region, trust_region)
        exp_arg   = (step_size * delta_sigma).clamp(-50.0, 50.0)
        sigma_new = sigma_safe * torch.exp(exp_arg)
    sigma_new = (
        sigma_new.clamp(min=lower_bound)
        if upper_bound is None
        else sigma_new.clamp(min=lower_bound, max=upper_bound)
    )
    return sigma_new.to(orig_dtype)


def natural_gradient(
    grad_mu:    torch.Tensor,                    # (..., K) Euclidean grad wrt mu
    grad_sigma: Optional[torch.Tensor],          # None freezes sigma; else Euclidean grad wrt sigma
    sigma_q:    torch.Tensor,                    # (..., K) diagonal OR (..., K, K) full covariance

    *,
    eps:        float = 1e-6,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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
    with torch.amp.autocast(sigma_q.device.type, enabled=False):        # tensor-keyed (audit 2026-07-05 m10)
        # float64 stays float64 (audit 2026-07-12 N12; keyed on sigma_q, the metric); half promotes to fp32.
        compute_dtype = torch.float64 if orig_dtype == torch.float64 else torch.float32
        sigma_q    = sigma_q.to(compute_dtype)
        grad_mu    = grad_mu.to(compute_dtype)
        grad_sigma = None if grad_sigma is None else grad_sigma.to(compute_dtype)
        if is_diagonal:
            sigma_safe     = sigma_q.clamp(min=eps)
            nat_grad_mu    = sigma_safe * grad_mu
            nat_grad_sigma = (None if grad_sigma is None
                              else 2.0 * sigma_safe * sigma_safe * grad_sigma)
        else:
            nat_grad_mu    = torch.einsum('...ij,...j->...i', sigma_q, grad_mu)
            if grad_sigma is None:
                nat_grad_sigma = None
            else:
                nat_grad_sigma = 2.0 * torch.einsum('...ij,...jk,...kl->...il', sigma_q, grad_sigma, sigma_q)
                nat_grad_sigma = 0.5 * (nat_grad_sigma + nat_grad_sigma.transpose(-1, -2))
    return nat_grad_mu.to(orig_dtype), (None if nat_grad_sigma is None
                                        else nat_grad_sigma.to(orig_dtype))


def retract_phi(
    phi:          torch.Tensor,           # (..., n_gen) current gauge frame
    delta_phi:    torch.Tensor,           # (..., n_gen) tangent step (e.g. -grad_phi)
    group:        GaugeGroup,             # supplies generators, skew flag, irrep_dims

    *,
    step_size:      float           = 1.0,
    eps:            float           = 1e-6,
    order:          int             = 4,
    project_slk:    bool            = False,
    compact_blocks: bool            = False,
    mode:           str             = "euclidean",

    trust_region:   Optional[float] = None, # None -> group default (GL:0.1, SO:0.3)
    max_norm:       Optional[float] = None, # None -> group default (GL:5.0, SO:pi)
    trace_clamp:    Optional[float] = None, # soft per-block |tr| cap (GL only)
) -> torch.Tensor:
    r"""Group-aware phi retraction dispatcher (Gaussian-specialized).

    Skew group (SO(N)) -> retract_son, det control is a no-op (det exp = 1).
    Non-skew (GL(K))   -> retract_glk, then optional det control:
      project_slk=True  hard-projects each block to sl(K) (det Omega_h = 1);
      else trace_clamp soft-bounds |tr| per block. Defaults for trust_region /
      max_norm are taken from the group's compactness when not given.
    """
    G = group.generators
    compact_blocks = (
        compact_blocks
        and mode == "bch"
        and group.phi_coordinate_layout == "block_head_row_major"
    )
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
        compact_blocks=compact_blocks,
        block_dims=(group.irrep_dims if compact_blocks else None),
    )
    if project_slk:
        phi_new = project_phi_to_slk(phi_new, G, group.irrep_dims)
    elif trace_clamp is not None:
        phi_new = clamp_phi_trace(phi_new, G, group.irrep_dims, trace_max=trace_clamp)
    return phi_new
