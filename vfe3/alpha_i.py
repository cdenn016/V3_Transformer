r"""Self-coupling coefficient alpha_i for VFE_3.0 (the weight on D(q_i || p_i)).

A config-selected registry of forms:
  constant                  alpha = value (default 1.0); no regularizer.
  state_dependent           alpha*_i = c0 / (b0 + D(q_i||p_i)), the stationary
                            point of alpha*D + R(alpha), R(alpha)=b0*alpha - c0*log alpha.
  state_dependent_per_coord per-coordinate alpha^(k)* = c0^(k)/(b0^(k) + D^(k)).
Pure: a function of the (per-position or per-coordinate) self-divergence D.
"""

from typing import Callable, Dict, Optional, Tuple

import torch

_ALPHAS:          Dict[str, Callable] = {}
_ALPHA_PER_COORD: Dict[str, bool]     = {}


def register_alpha(name: str, *, per_coord: bool = False) -> Callable:
    """Decorator registering an alpha form D -> (alpha, regularizer).

    ``per_coord`` declares whether the form consumes a per-COORDINATE (unsummed) self-
    divergence of shape (..., N, K) rather than the per-position summed (..., N). The
    routing seam ``free_energy.self_divergence_for_alpha`` reads this flag to supply the
    correctly-shaped divergence, so a per-coordinate form slots in by registration alone --
    no consumer call site is edited.
    """
    def _wrap(fn: Callable) -> Callable:
        _ALPHAS[name] = fn
        _ALPHA_PER_COORD[name] = per_coord
        return fn
    return _wrap


def get_alpha(name: str) -> Callable:
    """Return the registered alpha form (KeyError if absent)."""
    if name not in _ALPHAS:
        raise KeyError(f"no alpha form {name!r}; available: {sorted(_ALPHAS)}")
    return _ALPHAS[name]


def alpha_is_per_coord(mode: str) -> bool:
    """Whether alpha form ``mode`` consumes a per-coordinate (unsummed) self-divergence."""
    if mode not in _ALPHA_PER_COORD:
        raise KeyError(f"no alpha form {mode!r}; available: {sorted(_ALPHAS)}")
    return _ALPHA_PER_COORD[mode]


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


@register_alpha("learnable")
def alpha_learnable(
    kl:        torch.Tensor,             # (..., N) or (..., N, K) self-divergence (unused: alpha is free)

    *,
    log_alpha: Optional[torch.Tensor] = None,   # scalar nn.Parameter; REQUIRED at call time (Optional
                                                # only for the registry's uniform kwargs bag -- every
                                                # alpha form shares one signature; None raises below)
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""NEURAL-NETWORK EXCEPTION (sanctioned, default-off): a LEARNED scalar self-coupling
    alpha = exp(log_alpha), where ``log_alpha`` is a model-owned ``nn.Parameter`` trained by
    backprop (cf. ``use_head_mixer`` / ``use_prior_bank``). The pure no-NN default path is the
    ``constant`` / ``state_dependent`` / ``state_dependent_per_coord`` forms above, unchanged.

    alpha = exp(log_alpha) is a FREE parameter, NOT a precision posterior summary, so the
    alpha-envelope cancellation that ``state_dependent`` relies on (an explicit R(alpha) in F
    whose product-rule path cancels at the stationary alpha*) does NOT apply here: there is no
    Gamma prior and no regularizer. The returned regularizer is zero, so F carries the plain
    self-term alpha*D and the belief gradient is the plain alpha*dD (matching the ``constant``
    form's contract). The scalar exp(log_alpha) is broadcast to ``kl``'s shape so the gradient
    flows back to ``log_alpha`` through both the E-step belief updates and the F/loss.

    log_alpha = 0 -> alpha = exp(0) = 1.0, exactly reproducing ``constant`` alpha=1.0 at init;
    exp keeps alpha strictly positive for any real log_alpha.
    """
    if log_alpha is None:
        raise ValueError("alpha_mode='learnable' requires log_alpha (the model's nn.Parameter)")
    return torch.exp(log_alpha) * torch.ones_like(kl), torch.zeros_like(kl)


@register_alpha("state_dependent_per_coord", per_coord=True)
def alpha_state_dependent_per_coord(
    kl:    torch.Tensor,             # (..., N, K) per-coordinate self-divergence D^(k)

    *,
    b0:    'float | torch.Tensor' = 1.0,   # scalar or (K,)
    c0:    'float | torch.Tensor' = 1.0,   # scalar or (K,)
    eps:   float = 1e-12,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Per-coordinate alpha^(k)* = c0^(k)/(b0^(k) + D^(k)); R summed by the caller's F.

    A pure function of whatever divergence it is given: the per-coordinate self-term
    sum_k alpha^(k) D^(k) results when fed the UNSUMMED per-coordinate divergence of shape
    (..., N, K) that ``free_energy.self_divergence_for_alpha`` routes here for this form
    (declared ``per_coord=True``). The per-coordinate divergence exists only for the
    diagonal family + Renyi functional, enforced by the router and at config construction.
    """
    alpha = c0 / (b0 + kl).clamp(min=eps)
    return alpha, alpha_regularizer(alpha, b0=b0, c0=c0)


def alpha_gradient_coefficient(
    kl:        torch.Tensor,             # (..., N) or (..., N, K) self-divergence

    *,
    value:     float = 1.0,
    b0:        'float | torch.Tensor' = 1.0,
    c0:        'float | torch.Tensor' = 1.0,
    mode:      str = "constant",
    log_alpha: Optional[torch.Tensor] = None,   # learned scalar (alpha=exp(log_alpha)); only 'learnable' reads it
) -> torch.Tensor:
    r"""Effective coefficient a_i multiplying d D(q_i||p_i) in the belief gradient.

    By the alpha-envelope, at the state-dependent stationary point alpha* the
    coefficient is alpha* itself: d/dx[alpha*(D)*D + R(alpha*(D))] = alpha* dD/dx,
    because alpha + alpha'(D + b0 - c0/alpha) and the bracket vanishes at alpha*.
    So no product-rule correction is needed (R must be present in F). Constant
    mode returns ``value``.

    The coefficient is the alpha leg of the SAME registered form used by the
    oracle (``self_coupling_alpha``), not a re-derived copy: constant -> value,
    state-dependent -> alpha* = c0/(b0 + D), learnable -> exp(log_alpha) (the
    learnable form has zero R, so its coefficient is just alpha). Sharing the one
    formula makes the envelope-cancellation (kernel coefficient == oracle alpha) a
    structural identity rather than a maintained coincidence.
    """
    return self_coupling_alpha(kl, mode=mode, value=value, b0=b0, c0=c0, log_alpha=log_alpha)[0]


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
