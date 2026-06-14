r"""Numerical conditioning fallbacks + runtime monitors for VFE_3.0.

Two concerns, both modular (registry-backed):
  CONDITIONING FALLBACKS keep the SPD-manifold math finite under ill-conditioning:
    safe_spd_inverse (escalating-jitter Cholesky -> pinv), floor_eigenvalues,
    condition_number.
  RUNTIME MONITORS report numerical health during a run (nan/inf fraction, condition
    number, ...) as plain scalars, via a register_monitor registry so a new probe slots
    in without editing call sites. ``run_monitors`` emits a CSV/JSON-friendly record.

A theoretically pure path is always available (the unregularized op); the fallbacks are
guards that activate only when the pure path fails, and they are documented as such.
"""

from typing import Callable, Dict, List, Optional, Tuple

import torch


def _symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    """Average a matrix with its transpose (kills asymmetric round-off)."""
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def apply_mu_trust_region(
    delta_mu: torch.Tensor,              # (..., K) proposed mean step (e_q_mu_lr * nat_grad_mu)
    sigma_q:  torch.Tensor,              # (..., K) diagonal variances OR (..., K, K) covariance

    *,
    trust:       float = 5.0,
    mode:        str   = "box",
    is_diagonal: bool  = True,
    eps:         float = 1e-8,
) -> torch.Tensor:                       # (..., K) clamped step, same shape/dtype as delta_mu
    r"""Whitened E-step mean trust region (VFE_2.0 ``apply_mu_trust_region`` parity).

    Bounds the per-iteration mean update in :math:`\sigma`-whitened (Mahalanobis) units so a
    large VFE mean gradient cannot overshoot the belief by more than ``trust`` standard deviations:

        scale    = sqrt(diag(sigma_q)),  whitened = delta_mu / scale
        box      : clamp(whitened, -trust, +trust) * scale          (per-coordinate)
        ball     : delta_mu * min(trust / ||whitened||_2, 1)        (direction-preserving)

    ``box`` is V2's winning-run mode. This is a step-size guard, OFF by default at the call site
    (``e_mu_q_trust=None``); when ``trust`` does not bind it is the identity.
    """
    sigma_diag = sigma_q if is_diagonal else sigma_q.diagonal(dim1=-2, dim2=-1)
    scale = sigma_diag.clamp(min=eps).sqrt()
    whitened = delta_mu / scale
    if mode == "ball":
        norm2 = whitened.norm(dim=-1, keepdim=True)
        return delta_mu * (trust / norm2.clamp(min=eps)).clamp(max=1.0)
    if mode != "box":
        raise ValueError(f"apply_mu_trust_region mode={mode!r}; expected 'box' or 'ball'.")
    return whitened.clamp(-trust, trust) * scale


def safe_cholesky(
    matrix: torch.Tensor,                # (..., K, K) symmetric ~PD (per-element factored)

    *,
    eps:    float = 1e-6,
    rounds: int   = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:  # (factor (..., K, K), ok mask (...))
    r"""Per-element Cholesky that never raises, with optional per-element jitter escalation.

    Uses ``torch.linalg.cholesky_ex`` (returns a per-batch-element ``info``, does NOT raise)
    so that a single non-PD element cannot kill the whole batched call. Round 0 adds ZERO
    extra jitter, so on already-SPD inputs the returned factor is byte-identical to
    ``torch.linalg.cholesky`` (the pure path). Elements that fail (``info != 0``) are retried
    with an escalating ridge ``eps * 10^t`` for t = 0..rounds-1, applied ONLY to the failed
    elements so good elements keep their round-0 factor unperturbed.

    Returns the factor ``L`` together with a boolean ``ok`` mask (True where a PD factor was
    obtained). Callers MUST drive masking off ``ok`` (not finiteness): on failure ``cholesky_ex``
    returns a finite *partial* factor, not NaN, so a downstream ``logdet`` would otherwise be a
    finite-but-wrong value rather than NaN. The mask lets the caller inject NaN for failed
    elements so a ``safe_kl_clamp`` maps them to ``kl_max``.
    """
    M = _symmetrize(matrix)
    L, info = torch.linalg.cholesky_ex(M)
    ok = info == 0
    if rounds > 0 and not bool(ok.all()):
        K = M.shape[-1]
        eye = torch.eye(K, device=M.device, dtype=M.dtype)
        for t in range(rounds):
            if bool(ok.all()):
                break
            L_t, info_t = torch.linalg.cholesky_ex(M + (eps * (10.0 ** t)) * eye)
            newly = (~ok) & (info_t == 0)
            L = torch.where(newly.unsqueeze(-1).unsqueeze(-1), L_t, L)
            ok = ok | (info_t == 0)
    return L, ok


def safe_spd_inverse(
    matrix:    torch.Tensor,             # (..., K, K) symmetric ~PD

    *,
    eps:       float = 1e-6,
    max_tries: int   = 5,
) -> torch.Tensor:                       # (..., K, K) inverse
    r"""SPD inverse via Cholesky with escalating jitter, falling back to the pseudo-inverse.

    Per element (via ``cholesky_ex``, which never raises): tries ``cholesky_inverse`` on
    ``M + (eps * 10^t) I`` for t = 0..max_tries-1, escalating the ridge ONLY on the elements that
    still fail; an element where every jitter level fails falls back to ``pinv``. The per-element
    retry mirrors ``safe_cholesky`` so one non-PD batch element cannot poison the exact inverse of
    its well-conditioned siblings. The pure path is ``t=0`` with the documented default ridge.
    """
    M = _symmetrize(matrix.float())
    K = M.shape[-1]
    eye = torch.eye(K, device=M.device, dtype=M.dtype)
    L, info = torch.linalg.cholesky_ex(M + eps * eye)        # round 0: documented eps ridge
    ok = info == 0
    if bool(ok.all()):
        return torch.cholesky_inverse(L).to(matrix.dtype)
    out = torch.cholesky_inverse(L)                          # ok elements keep their good inverse
    for t in range(1, max_tries):                            # retry ONLY the still-failed elements
        if bool(ok.all()):
            break
        L_t, info_t = torch.linalg.cholesky_ex(M + (eps * (10.0 ** t)) * eye)
        newly = (~ok) & (info_t == 0)
        if bool(newly.any()):
            inv_t = torch.cholesky_inverse(L_t)
            out = torch.where(newly.unsqueeze(-1).unsqueeze(-1), inv_t, out)
            ok = ok | (info_t == 0)
    if not bool(ok.all()):                                   # pinv ONLY the still-failed elements
        out = torch.where(ok.unsqueeze(-1).unsqueeze(-1), out, torch.linalg.pinv(M))
    return out.to(matrix.dtype)


def floor_eigenvalues(
    matrix: torch.Tensor,                # (..., K, K) symmetric
    *,
    floor:  float = 1e-6,
) -> torch.Tensor:                       # (..., K, K) SPD with eigenvalues >= floor
    r"""Project a symmetric matrix to SPD by clamping its eigenvalues up to ``floor``."""
    M = _symmetrize(matrix.float())
    evals, evecs = torch.linalg.eigh(M)
    evals = evals.clamp(min=floor)
    out = (evecs * evals.unsqueeze(-2)) @ evecs.transpose(-1, -2)
    return _symmetrize(out).to(matrix.dtype)


def condition_number(
    matrix: torch.Tensor,                # (..., K, K) symmetric PD OR (..., K) diagonal variances
    *,
    eps:    float = 1e-12,
) -> torch.Tensor:                       # (...) lambda_max / lambda_min
    r"""Spectral condition number lambda_max / lambda_min (clamped at ``eps``).

    Accepts a full covariance (square trailing dims (..., K, K), via ``eigvalsh``) OR a diagonal
    variance spectrum (non-square trailing dims (..., K), max/min over the last axis) so the monitor
    works on the default ``gaussian_diagonal`` family instead of raising on the rank mismatch (audit
    2026-06-13 L13). The square-trailing test resolves the two for any (..., N, K) with N != K."""
    if matrix.dim() < 2 or matrix.shape[-1] != matrix.shape[-2]:
        spec = matrix.float().clamp(min=0.0)
        return (spec.max(dim=-1).values / spec.min(dim=-1).values.clamp(min=eps)).to(matrix.dtype)
    evals = torch.linalg.eigvalsh(_symmetrize(matrix.float()))
    return (evals[..., -1] / evals[..., 0].clamp(min=eps)).to(matrix.dtype)


def nan_inf_fraction(
    tensor: torch.Tensor,
) -> float:                              # fraction of non-finite entries in [0, 1]
    r"""Fraction of NaN/Inf entries (0.0 = all finite)."""
    if tensor.numel() == 0:
        return 0.0
    return float((~torch.isfinite(tensor)).float().mean())


def check_finite(
    tensor: torch.Tensor,
    name:   str = "tensor",

    *,
    raise_on_nonfinite: bool = False,
) -> bool:                               # True if all-finite
    r"""Report (and optionally raise on) non-finite entries; returns finiteness."""
    frac = nan_inf_fraction(tensor)
    if frac > 0.0:
        msg = f"{name}: {frac:.3%} non-finite entries"
        if raise_on_nonfinite:
            raise FloatingPointError(msg)
        import warnings
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        return False
    return True


# ---------------------------------------------------------------------------
# Monitor registry: name -> (tensor -> scalar). New probes slot in by name.
# ---------------------------------------------------------------------------
_MONITORS: Dict[str, Callable[[torch.Tensor], float]] = {}


def register_monitor(name: str) -> Callable:
    """Decorator registering a scalar numerical monitor under ``name``."""
    def _wrap(fn: Callable[[torch.Tensor], float]) -> Callable[[torch.Tensor], float]:
        _MONITORS[name] = fn
        return fn
    return _wrap


def get_monitor(name: str) -> Callable[[torch.Tensor], float]:
    """Return the registered monitor (KeyError if absent)."""
    if name not in _MONITORS:
        raise KeyError(f"no monitor {name!r}; available: {sorted(_MONITORS)}")
    return _MONITORS[name]


@register_monitor("nan_fraction")
def _mon_nan(tensor: torch.Tensor) -> float:
    """Fraction of non-finite entries."""
    return nan_inf_fraction(tensor)


@register_monitor("abs_max")
def _mon_absmax(tensor: torch.Tensor) -> float:
    """Largest absolute (finite) entry magnitude."""
    finite = tensor[torch.isfinite(tensor)]
    return float(finite.abs().max()) if finite.numel() else float("nan")


@register_monitor("condition_number")
def _mon_cond(matrix: torch.Tensor) -> float:
    """Spectral condition number (max over any leading batch)."""
    return float(condition_number(matrix).max())


def run_monitors(
    tensor:   torch.Tensor,
    monitors: Optional[List[str]] = None,
) -> Dict[str, float]:
    r"""Apply the named monitors to ``tensor``; returns a CSV/JSON-friendly record.

    ``monitors=None`` runs the family-agnostic probes (nan_fraction, abs_max); pass an
    explicit list to include matrix probes (e.g. condition_number) on SPD inputs.
    """
    names = ["nan_fraction", "abs_max"] if monitors is None else monitors
    return {n: get_monitor(n)(tensor) for n in names}
