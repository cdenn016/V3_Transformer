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

import weakref
from collections import OrderedDict
from typing import Callable, Dict, List, Literal, Optional, Tuple

import torch


def _symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    """Average a matrix with its transpose (kills asymmetric round-off)."""
    return 0.5 * (matrix + matrix.transpose(-1, -2))


# Weak, mutation-aware cache for the bounded_variance_from_log overflow check (audit 2026-07-12
# N13; the lie_ops/killing-cache identity+version pattern). The hot-path callers (prior_bank's
# decode/encode table reads) re-read the SAME parameter tables several times per forward, and the
# check's bool((...).any()) is a device->host sync per call -- key the RESULT on
# (id, _version, ...) with a weakref liveness guard so an unchanged table syncs once, an in-place
# optimizer step (version bump) triggers exactly one recheck, and a dead/recycled id can never
# serve a stale verdict.
_MAX_LOG_CHECK_CACHE: 'OrderedDict[tuple, tuple]' = OrderedDict()
_MAX_LOG_CHECK_CACHE_MAXSIZE = 32


def _max_log_exceeded(
    log_sigma: torch.Tensor,
    max_log:   float,
) -> bool:
    """The one host-syncing overflow check (the cached slow path)."""
    return bool((log_sigma.detach() > max_log).any())


def _cached_max_log_exceeded(
    log_sigma: torch.Tensor,
    max_log:   float,
) -> bool:
    """Resolve the overflow check through the weak, mutation-aware identity/version cache."""
    if log_sigma.is_inference():
        # Inference tensors track NO _version counter (reading it raises), so there is no
        # mutation signal to key on -- fall back to the direct uncached check (the pre-cache
        # behavior; one sync per call, exactly as before the N13 cache).
        return _max_log_exceeded(log_sigma, max_log)
    key = (id(log_sigma), log_sigma._version, tuple(log_sigma.shape), log_sigma.dtype,
           log_sigma.device, max_log)
    cached = _MAX_LOG_CHECK_CACHE.get(key)
    if cached is not None:
        tensor_ref, exceeded = cached
        if tensor_ref() is log_sigma:
            _MAX_LOG_CHECK_CACHE.move_to_end(key)
            return exceeded
        del _MAX_LOG_CHECK_CACHE[key]

    exceeded = _max_log_exceeded(log_sigma, max_log)

    def _drop_dead_entry(tensor_ref: weakref.ReferenceType) -> None:
        current = _MAX_LOG_CHECK_CACHE.get(key)
        if current is not None and current[0] is tensor_ref:
            del _MAX_LOG_CHECK_CACHE[key]

    _MAX_LOG_CHECK_CACHE[key] = (weakref.ref(log_sigma, _drop_dead_entry), exceeded)
    _MAX_LOG_CHECK_CACHE.move_to_end(key)
    while len(_MAX_LOG_CHECK_CACHE) > _MAX_LOG_CHECK_CACHE_MAXSIZE:
        _MAX_LOG_CHECK_CACHE.popitem(last=False)
    return exceeded


def bounded_variance_from_log(
    log_sigma: torch.Tensor,

    *,
    eps:     float = 1e-6,
    max_log: float = 80.0,
) -> torch.Tensor:
    r"""Exponentiate a trainable log-variance without overflowing float32.

    Values in the normal ``[log(eps), max_log]`` range retain the ordinary ``exp`` map. Larger
    detached parameter values emit the numerical warning and are capped only for exponentiation;
    ``sigma_max`` is a separate belief-state retraction policy and is deliberately not used here.
    The overflow check is identity/version-cached (audit 2026-07-12 N13): one device->host sync
    per table mutation instead of per call; the warning still fires on every call while the table
    stays above ``max_log`` (from the cached host bool).
    """
    if _cached_max_log_exceeded(log_sigma, max_log):
        import warnings
        warnings.warn(
            f"trainable log-variance exceeds max_log={max_log:g}; clamping before exponentiation",
            RuntimeWarning,
            stacklevel=2,
        )
    return torch.exp(log_sigma.clamp(max=max_log)).clamp(min=eps)


def apply_mu_trust_region(
    delta_mu: torch.Tensor,              # (..., K) proposed mean step (e_q_mu_lr * nat_grad_mu)
    sigma_q:  torch.Tensor,              # (..., K) diagonal variances OR (..., K, K) covariance

    *,
    trust:       float = 5.0,
    mode:        str   = "box",
    is_diagonal: bool  = True,
    eps:         float = 1e-8,
) -> torch.Tensor:                       # (..., K) clamped step, same shape/dtype as delta_mu
    r"""Whitened E-step mean trust region.

    Bounds the per-iteration mean update in covariance-whitened (Mahalanobis) units so a large
    VFE mean gradient cannot overshoot the belief by more than ``trust`` standard deviations.
    Let ``L`` be ``diag(sqrt(sigma_q))`` for diagonal covariance or the round-zero Cholesky factor
    of a full covariance. Then:

        whitened = solve(L, delta_mu)
        box      : L @ clamp(whitened, -trust, +trust)
        ball     : L @ (whitened * min(trust / ||whitened||_2, 1))

    ``box`` is the recommended mode. This is a step-size guard, OFF by default at the call site
    (``e_mu_q_trust=None``). A failed full-covariance Cholesky uses the prior marginal-variance
    path for that batch element only.
    """
    if mode not in ("box", "ball"):
        raise ValueError(f"apply_mu_trust_region mode={mode!r}; expected 'box' or 'ball'.")

    if is_diagonal:
        scale = sigma_q.clamp(min=eps).sqrt()
        whitened = delta_mu / scale
        if mode == "ball":
            norm2 = whitened.norm(dim=-1, keepdim=True)
            return delta_mu * (trust / norm2.clamp(min=eps)).clamp(max=1.0)
        return whitened.clamp(-trust, trust) * scale

    factor, ok = safe_cholesky(sigma_q, eps=eps, rounds=0)
    eye = torch.eye(sigma_q.shape[-1], device=sigma_q.device, dtype=sigma_q.dtype)
    safe_factor = torch.where(ok.unsqueeze(-1).unsqueeze(-1), factor, eye.expand_as(factor))
    whitened = torch.linalg.solve_triangular(
        safe_factor,
        delta_mu.unsqueeze(-1),
        upper=False,
    ).squeeze(-1)
    if mode == "ball":
        norm2 = whitened.norm(dim=-1, keepdim=True)
        bounded = whitened * (trust / norm2.clamp(min=eps)).clamp(max=1.0)
    else:
        bounded = whitened.clamp(-trust, trust)
    full_out = (safe_factor @ bounded.unsqueeze(-1)).squeeze(-1)
    if bool(ok.all()):
        return full_out

    sigma_diag = sigma_q.diagonal(dim1=-2, dim2=-1)
    scale = sigma_diag.clamp(min=eps).sqrt()
    fallback_white = delta_mu / scale
    if mode == "ball":
        fallback_norm = fallback_white.norm(dim=-1, keepdim=True)
        fallback = delta_mu * (trust / fallback_norm.clamp(min=eps)).clamp(max=1.0)
    else:
        fallback = fallback_white.clamp(-trust, trust) * scale
    return torch.where(ok.unsqueeze(-1), full_out, fallback)


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
    # float64 stays float64 (audit 2026-07-12 N4/N12 dtype policy); half promotes to fp32.
    compute_dtype = torch.float64 if matrix.dtype == torch.float64 else torch.float32
    M = _symmetrize(matrix.to(compute_dtype))
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
    eps:    float                               = 1e-12,
    kind:   Literal["auto", "full", "diagonal"] = "auto",
) -> torch.Tensor:                       # (...) lambda_max / lambda_min
    r"""Spectral condition number lambda_max / lambda_min (clamped at ``eps``).

    ``kind='diagonal'`` always treats the last axis as a variance spectrum, including a square
    ``(N, K)`` table with ``N == K``. ``kind='full'`` requires square trailing dimensions and uses
    ``eigvalsh``. ``kind='auto'`` preserves the legacy shape inference: square trailing dimensions
    select the full-matrix path and every other non-scalar shape selects the diagonal path.
    """
    if kind not in ("auto", "full", "diagonal"):
        raise ValueError(
            f"condition_number kind must be 'auto', 'full', or 'diagonal', got {kind!r}")

    square = matrix.dim() >= 2 and matrix.shape[-1] == matrix.shape[-2]
    if kind == "full" and not square:
        raise ValueError(
            "condition_number kind='full' requires square trailing dimensions (..., K, K), "
            f"got shape {tuple(matrix.shape)}")
    full = kind == "full" or (kind == "auto" and square)
    if not full:
        if matrix.dim() == 0:
            raise ValueError("condition_number diagonal input must have at least one dimension")
        spec = matrix.float()
        lam_min = spec.min(dim=-1).values
        cond = spec.max(dim=-1).values / lam_min.clamp(min=eps)
        # non-positive variance -> no condition number; surface +inf, mirroring the full-matrix branch
        # (audit 2026-06-17 round 2 id1), not a large positive value from clamping a zero/negative up to eps.
        return torch.where(lam_min > 0, cond, cond.new_tensor(float("inf"))).to(matrix.dtype)
    evals = torch.linalg.eigvalsh(_symmetrize(matrix.float()))
    lam_min = evals[..., 0]
    cond = evals[..., -1] / lam_min.clamp(min=eps)
    # A non-PD matrix (lambda_min <= 0) has no condition number; surface +inf rather than the large
    # positive value clamping a negative lambda_min up to eps would give (which reads as a merely
    # ill-conditioned SPD matrix and hides the loss of positive-definiteness). (audit 2026-06-17)
    cond = torch.where(lam_min > 0, cond, cond.new_tensor(float("inf")))
    return cond.to(matrix.dtype)


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


def register_monitor(name: str, *, override: bool = False) -> Callable:
    """Decorator registering a scalar numerical monitor under ``name``.

    Duplicate keys fail closed (audit 2026-07-01 round-3): a second registration under an
    existing name silently shadowed the first. Pass ``override=True`` to replace deliberately.
    """
    def _wrap(fn: Callable[[torch.Tensor], float]) -> Callable[[torch.Tensor], float]:
        if name in _MONITORS and not override:
            raise KeyError(f"monitor {name!r} already registered; pass override=True to replace")
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
