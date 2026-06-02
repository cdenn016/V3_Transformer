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

    Tries ``cholesky_inverse`` on ``M + (eps * 10^t) I`` for t = 0..max_tries-1; if every
    jitter level fails (or the input is too ill-conditioned), returns ``pinv``. The pure
    path is ``t=0`` with the documented default ridge; larger jitter is the fallback.
    """
    M = _symmetrize(matrix.float())
    K = M.shape[-1]
    eye = torch.eye(K, device=M.device, dtype=M.dtype)
    for t in range(max_tries):
        ridge = eps * (10.0 ** t)
        try:
            L = torch.linalg.cholesky(M + ridge * eye)
            return torch.cholesky_inverse(L).to(matrix.dtype)
        except (torch.linalg.LinAlgError, RuntimeError):
            continue
    return torch.linalg.pinv(M).to(matrix.dtype)


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
    matrix: torch.Tensor,                # (..., K, K) symmetric PD
    *,
    eps:    float = 1e-12,
) -> torch.Tensor:                       # (...) lambda_max / lambda_min
    r"""Spectral condition number lambda_max / lambda_min (clamped at ``eps``)."""
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
