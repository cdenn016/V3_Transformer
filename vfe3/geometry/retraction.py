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
    if sigma_max is None:
        return
    if not math.isfinite(sigma_max) or sigma_max < eps:
        raise ValueError(
            f"sigma_max must be None or finite and >= eps ({eps}); got {sigma_max!r}"
        )


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
    rel:   float = 1e-6,
    floor: float = 1e-12,
) -> torch.Tensor:                         # 0-d on-device tensor; no host-sync
    r"""Spectrum-relative ``gap_eps`` for :func:`_eigh_damped` on the SPD retraction paths (audit
    2026-06-13 L11). The fixed ``gap_eps=1e-8`` over-damps the eigh adjoint ``F_ij = 1/(w_j - w_i)``
    for MEANINGFUL gaps near the variance floor -- a resolvable gap of 1e-4 is biased ~50%. Scaling
    to ``(rel * ||A||_max)^2`` -- the squared fp32 noise floor of the spectrum -- damps only gaps
    below fp32 resolution (true degeneracy stays finite, ``F = 0``) and leaves resolvable gaps
    accurate. ``rel`` is a few machine epsilons; ``floor`` keeps a tiny near-zero spectrum finite.
    Returns a 0-d tensor on A's device/dtype to avoid a CUDA host-sync (no ``.item()``/``float()``)."""
    scale = A.detach().abs().amax()        # 0-d tensor, stays on device
    return (rel * scale).pow(2).clamp(min=floor)


def retract_spd_diagonal(
    sigma_diag:   torch.Tensor,             # (..., K) diagonal variances
    delta_sigma:  torch.Tensor,             # (..., K) diagonal tangent

    *,
    step_size:    float          = 1.0,
    trust_region: float          = 5.0,
    eps:          float          = 1e-6,
    sigma_max:    Optional[float] = 10.0,   # matches VFE3Config.sigma_max
) -> torch.Tensor:
    r"""Diagonal SPD retraction sigma_new = sigma * exp(tau * clamp(dsigma/sigma)).

    Positivity by construction (exp > 0); clamped to [eps, sigma_max].
    When sigma_max is None the eigenvalue ceiling is skipped (pure-path: eps floor only).
    """
    _check_sigma_max(sigma_max, eps)
    orig_dtype = sigma_diag.dtype
    with torch.amp.autocast(sigma_diag.device.type, enabled=False):     # tensor-keyed (audit 2026-07-05 m10)
        sigma_safe = sigma_diag.float().clamp(min=eps)
        delta_sigma = delta_sigma.float()
        whitened = delta_sigma / sigma_safe
        if trust_region is not None and trust_region > 0:
            whitened = whitened.clamp(-trust_region, trust_region)
        exp_arg = (step_size * whitened).clamp(-50.0, 50.0)
        sigma_new = sigma_safe * torch.exp(exp_arg)
    sigma_new = sigma_new.clamp(min=eps) if sigma_max is None else sigma_new.clamp(min=eps, max=sigma_max)
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
    with a Frobenius trust region on the whitened tangent and an eigenvalue
    floor/ceiling [eps, sigma_max] (eigenvalues are variances: the SAME physical ceiling the
    diagonal arm applies to sigma). Uses the gap-regularized ``_eigh_damped`` eigendecomposition so
    the unrolled backward stays finite on a degenerate spectrum -- the isotropic ``Sigma = I`` default
    gaussian_full init makes the stock eigh backward 100% NaN; forward values are unchanged.
    When sigma_max is None the eigenvalue ceiling is skipped (pure-path: eps floor only).
    """
    _check_sigma_max(sigma_max, eps)
    orig_shape = sigma.shape
    orig_dtype = sigma.dtype
    if sigma.dim() == 4:
        B, N, K, _ = sigma.shape
        sigma = sigma.reshape(B * N, K, K)
        delta_sigma = delta_sigma.reshape(B * N, K, K)

    with torch.amp.autocast(sigma.device.type, enabled=False):          # tensor-keyed (audit 2026-07-05 m10)
        sigma = sigma.float()
        delta_sigma = delta_sigma.float()
        sigma = 0.5 * (sigma + sigma.transpose(-1, -2))
        delta_sigma = 0.5 * (delta_sigma + delta_sigma.transpose(-1, -2))

        eigenvalues, eigenvectors = _eigh_damped(sigma, _rel_gap_eps(sigma))
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

        R_eval, R_evec = _eigh_damped(R, _rel_gap_eps(R))
        R_eval = R_eval.clamp(-50.0, 50.0)
        exp_R = R_evec * torch.exp(R_eval).unsqueeze(-2) @ R_evec.transpose(-1, -2)

        sigma_new = sigma_sqrt @ exp_R @ sigma_sqrt
        sigma_new = 0.5 * (sigma_new + sigma_new.transpose(-1, -2))

        eig_new, vec_new = _eigh_damped(sigma_new, _rel_gap_eps(sigma_new))
        eig_new = eig_new.clamp(min=eps) if sigma_max is None else eig_new.clamp(min=eps, max=sigma_max)
        sigma_new = vec_new * eig_new.unsqueeze(-2) @ vec_new.transpose(-1, -2)

    sigma_new = sigma_new.to(orig_dtype)
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
    reduced to sigma_new = sigma exp(step_size dsigma/sigma) on the diagonal cone. Behavior-preserving:
    a thin dispatcher that forwards verbatim to the bare functions; the Fisher metric conversion stays
    in the E-step (``natural_gradient``), so the tangent ``delta_sigma`` arrives already preconditioned.
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
    delta_log:    torch.Tensor,             # (..., K, K) symmetric tangent (taken in the log chart)

    *,
    step_size:    float          = 1.0,
    trust_region: float          = 5.0,
    eps:          float          = 1e-6,
    sigma_max:    Optional[float] = 10.0,   # matches VFE3Config.sigma_max
) -> torch.Tensor:
    r"""Full log-Euclidean SPD retraction (Arsigny-Fillard-Pennec-Ayache 2006/2007).

        Sigma_new = expm( logm(Sigma) + step_size * sym(delta_log) ),
    where logm/expm are the matrix log/exp in the eigenbasis (logm(Sigma) =
    V diag(log lambda_j) V^T; expm(M) = U diag(exp mu_j) U^T). Because logm(Sigma)
    is symmetric and expm of a symmetric matrix is SPD, this is SPD-preserving for
    ANY magnitude of the tangent (no trust region needed for positivity; the trust
    region is a Frobenius stability knob only). The tangent is taken directly in the
    matrix-log chart (spec reading 2a, the pure retraction). Uses torch.linalg.eigh
    twice (one for logm(Sigma), one for the expm of the log-sum), the same two-eigh
    structure as retract_spd_full; floors the input eigenvalues at eps before log and
    projects the output spectrum to [eps, sigma_max] (eigenvalues ARE variances: the same
    physical ceiling the diagonal arm and retract_spd_full apply, matching the code at ~264).
    """
    _check_sigma_max(sigma_max, eps)
    orig_shape = sigma.shape
    orig_dtype = sigma.dtype
    if sigma.dim() == 4:
        B, N, K, _ = sigma.shape
        sigma = sigma.reshape(B * N, K, K)
        delta_log = delta_log.reshape(B * N, K, K)

    with torch.amp.autocast(sigma.device.type, enabled=False):          # tensor-keyed (audit 2026-07-05 m10)
        sigma     = sigma.float()
        delta_log = delta_log.float()
        sigma     = 0.5 * (sigma + sigma.transpose(-1, -2))
        delta_log = 0.5 * (delta_log + delta_log.transpose(-1, -2))

        eigenvalues, eigenvectors = _eigh_damped(sigma, _rel_gap_eps(sigma))
        eigenvalues = eigenvalues.clamp(min=eps)
        log_eig = torch.log(eigenvalues)
        log_sigma = eigenvectors * log_eig.unsqueeze(-2) @ eigenvectors.transpose(-1, -2)

        tangent = step_size * delta_log
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

    sigma_new = sigma_new.to(orig_dtype)
    if len(orig_shape) == 4:
        sigma_new = sigma_new.reshape(orig_shape)
    return sigma_new


@register_retraction("log_euclidean")
def retract_log_euclidean(
    sigma:        torch.Tensor,             # (..., K) diagonal OR (..., K, K) full covariance
    delta_sigma:  torch.Tensor,             # matching rank: the tangent step, taken in the log chart

    mean_ndim:    int,                      # ndim of the belief mean; full cov iff sigma.dim() == mean_ndim + 1

    *,
    step_size:    float          = 1.0,
    trust_region: float          = 5.0,
    eps:          float          = 1e-6,
    sigma_max:    Optional[float] = 10.0,   # matches VFE3Config.sigma_max
) -> torch.Tensor:                          # (...) same rank as sigma
    r"""Log-Euclidean SPD retraction (spec reading 2a; Arsigny et al. 2006/2007).

    The pure log-Euclidean retraction: the tangent ``delta_sigma`` is interpreted directly
    in the matrix-log chart and the update closes in the SPD vector-space structure,
        Sigma_new = expm( logm(Sigma) + step_size * sym(delta_sigma) ).
    SPD-preserving for ANY step magnitude (expm of a symmetric matrix is SPD), unlike a
    naive Euclidean step; the trust region is a stability knob, not a positivity guard.

    Full covariance (sigma.dim() == mean_ndim + 1) uses the two-eigh logm/expm in
    ``retract_logeuclidean_full``. The diagonal case is the elementwise reduction
        sigma_new = sigma * exp(step_size * delta_sigma),
    applying the tangent in the log chart WITHOUT the affine 1/sigma whitening. NOTE: under
    this seam's pre-whitened tangent convention (the E-step hands the retraction the affine
    Fisher natural gradient 2 Sigma G Sigma, retraction.py natural_gradient), the diagonal LE
    step does NOT coincide with ``spd_affine`` -- affine whitens by 1/sigma (sigma_new =
    sigma exp(step delta/sigma)), LE does not -- so on a diagonal family LE is a non-canonical
    log-chart step, not the manuscript-pinned geometry. LE is a genuinely new variant only for
    the full-covariance family, where logm != elementwise log. 2b (the Daleckii-Krein Frechet
    natural gradient) is a deferred sub-flag per the spec, not built here.
    """
    _check_sigma_max(sigma_max, eps)
    if sigma.dim() == mean_ndim + 1:                     # full covariance (..., K, K)
        return retract_logeuclidean_full(
            sigma, delta_sigma, step_size=step_size, trust_region=trust_region, eps=eps, sigma_max=sigma_max,
        )
    # diagonal: logm/expm are elementwise; sigma_new = sigma * exp(step_size * delta_sigma)
    orig_dtype = sigma.dtype
    with torch.amp.autocast(sigma.device.type, enabled=False):          # tensor-keyed (audit 2026-07-05 m10)
        sigma_safe  = sigma.float().clamp(min=eps)
        delta_sigma = delta_sigma.float()
        if trust_region is not None and trust_region > 0:
            delta_sigma = delta_sigma.clamp(-trust_region, trust_region)
        exp_arg   = (step_size * delta_sigma).clamp(-50.0, 50.0)
        sigma_new = sigma_safe * torch.exp(exp_arg)
    sigma_new = sigma_new.clamp(min=eps) if sigma_max is None else sigma_new.clamp(min=eps, max=sigma_max)
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
        sigma_q    = sigma_q.float()
        grad_mu    = grad_mu.float()
        grad_sigma = None if grad_sigma is None else grad_sigma.float()
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
