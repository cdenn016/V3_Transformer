r"""Self-coupling coefficient alpha_i for VFE_3.0 (the weight on D(q_i || p_i)).

A config-selected registry of forms:
  constant                  alpha = value (default 1.0); no regularizer.
  state_dependent           alpha*_i = c0 / (b0 + D(q_i||p_i)), the stationary
                            point of alpha*D + R(alpha), R(alpha)=b0*alpha - c0*log alpha.
  state_dependent_per_coord per-coordinate alpha^(k)* = c0^(k)/(b0^(k) + D^(k)).
Pure: a function of the (per-position or per-coordinate) self-divergence D.
"""

import warnings
from typing import Callable, Dict, Tuple

import torch

_ALPHAS: Dict[str, Callable] = {}


def register_alpha(name: str) -> Callable:
    """Decorator registering an alpha form D -> (alpha, regularizer)."""
    def _wrap(fn: Callable) -> Callable:
        _ALPHAS[name] = fn
        return fn
    return _wrap


def get_alpha(name: str) -> Callable:
    """Return the registered alpha form (KeyError if absent)."""
    if name not in _ALPHAS:
        raise KeyError(f"no alpha form {name!r}; available: {sorted(_ALPHAS)}")
    return _ALPHAS[name]


def alpha_regularizer(
    alpha: torch.Tensor,             # (...) coupling coefficient

    *,
    b0:    'float | torch.Tensor' = 1.0,
    c0:    'float | torch.Tensor' = 1.0,
) -> torch.Tensor:
    r"""Precision regularizer R(alpha) = b0*alpha - c0*log(alpha)."""
    return b0 * alpha - c0 * torch.log(alpha.clamp(min=1e-12))


@register_alpha("constant")
def alpha_constant(
    kl:    torch.Tensor,             # (..., N) or (..., N, K) self-divergence (unused)

    *,
    value: float = 1.0,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Constant alpha = value, zero regularizer."""
    return torch.full_like(kl, value), torch.zeros_like(kl)


@register_alpha("state_dependent")
def alpha_state_dependent(
    kl:    torch.Tensor,             # (..., N) per-position self-divergence

    *,
    b0:    'float | torch.Tensor' = 1.0,
    c0:    'float | torch.Tensor' = 1.0,
    eps:   float = 1e-12,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""State-dependent alpha*_i = c0 / (b0 + D(q_i||p_i)); R(alpha*)."""
    alpha = c0 / (b0 + kl).clamp(min=eps)
    return alpha, alpha_regularizer(alpha, b0=b0, c0=c0)


@register_alpha("state_dependent_per_coord")
def alpha_state_dependent_per_coord(
    kl:    torch.Tensor,             # (..., N, K) per-coordinate self-divergence
                                     # (NOT (..., N): see the deferred-path note)

    *,
    b0:    'float | torch.Tensor' = 1.0,   # scalar or (K,)
    c0:    'float | torch.Tensor' = 1.0,   # scalar or (K,)
    eps:   float = 1e-12,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Per-coordinate alpha^(k)* = c0^(k)/(b0^(k) + D^(k)); R summed by caller.

    The formula is exact for an UNSUMMED per-coordinate self-divergence of shape
    (..., N, K). The shipped self_divergence (free_energy.self_divergence) sums
    over the coordinate axis and returns per-position (..., N); fed that, this
    form silently broadcasts to per-position alpha (one alpha per token), NOT the
    per-coordinate alpha^(k) advertised. A per-coordinate (unsummed) divergence
    variant that realizes this mode is a DEFERRED extension point; until it is
    registered and routed here, prefer `state_dependent` (per-position) for the
    summed D the current pipeline supplies.
    """
    warnings.warn(
        "alpha_mode='state_dependent_per_coord' currently receives the summed per-position "
        "self-divergence and silently degrades to per-position alpha (identical to "
        "'state_dependent'); the per-coordinate (unsummed) divergence is a deferred path. "
        "Use 'state_dependent' for per-position alpha.",
        RuntimeWarning,
        stacklevel=3,
    )
    alpha = c0 / (b0 + kl).clamp(min=eps)
    return alpha, alpha_regularizer(alpha, b0=b0, c0=c0)


def alpha_gradient_coefficient(
    kl:    torch.Tensor,             # (..., N) or (..., N, K) self-divergence

    *,
    value: float = 1.0,
    b0:    'float | torch.Tensor' = 1.0,
    c0:    'float | torch.Tensor' = 1.0,
    mode:  str = "constant",
) -> torch.Tensor:
    r"""Effective coefficient a_i multiplying d D(q_i||p_i) in the belief gradient.

    By the alpha-envelope, at the state-dependent stationary point alpha* the
    coefficient is alpha* itself: d/dx[alpha*(D)*D + R(alpha*(D))] = alpha* dD/dx,
    because alpha + alpha'(D + b0 - c0/alpha) and the bracket vanishes at alpha*.
    So no product-rule correction is needed (R must be present in F). Constant
    mode returns ``value``.

    The coefficient is the alpha leg of the SAME registered form used by the
    oracle (``self_coupling_alpha``), not a re-derived copy: constant -> value,
    state-dependent -> alpha* = c0/(b0 + D). Sharing the one formula makes the
    envelope-cancellation (kernel coefficient == oracle alpha) a structural
    identity rather than a maintained coincidence.
    """
    return self_coupling_alpha(kl, mode=mode, value=value, b0=b0, c0=c0)[0]


def self_coupling_alpha(
    kl:       torch.Tensor,          # (..., N) or (..., N, K) self-divergence

    *,
    mode:     str = "constant",
    **kwargs,                        # forwarded verbatim to the form (value / b0 / c0 / ...)
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Dispatch to the registered alpha form `mode`; returns (alpha, regularizer).

    Variant params are forwarded via **kwargs (each form declares its own:
    `constant` takes `value`; the state-dependent forms take `b0`/`c0`). A new
    form with a novel param slots in by `register_alpha` + config -- the call site
    is never edited (matching the divergence seam, which forwards only what the
    selected leaf declares).
    """
    return get_alpha(mode)(kl, **kwargs)
