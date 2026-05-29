r"""Self-coupling coefficient alpha_i for VFE_3.0 (the weight on D(q_i || p_i)).

A config-selected registry of forms:
  constant                  alpha = value (default 1.0); no regularizer.
  state_dependent           alpha*_i = c0 / (b0 + D(q_i||p_i)), the stationary
                            point of alpha*D + R(alpha), R(alpha)=b0*alpha - c0*log alpha.
  state_dependent_per_coord per-coordinate alpha^(k)* = c0^(k)/(b0^(k) + D^(k)).
Pure: a function of the (per-position or per-coordinate) self-divergence D.
"""

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

    *,
    b0:    'float | torch.Tensor' = 1.0,   # scalar or (K,)
    c0:    'float | torch.Tensor' = 1.0,   # scalar or (K,)
    eps:   float = 1e-12,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Per-coordinate alpha^(k)* = c0^(k)/(b0^(k) + D^(k)); R summed by caller."""
    alpha = c0 / (b0 + kl).clamp(min=eps)
    return alpha, alpha_regularizer(alpha, b0=b0, c0=c0)


def self_coupling_alpha(
    kl:    torch.Tensor,             # (..., N) or (..., N, K) self-divergence

    *,
    value: float = 1.0,
    b0:    'float | torch.Tensor' = 1.0,
    c0:    'float | torch.Tensor' = 1.0,
    mode:  str = "constant",
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Dispatch to the registered alpha form `mode`; returns (alpha, regularizer)."""
    return get_alpha(mode)(kl, value=value, b0=b0, c0=c0)
