r"""Gauge-frame (phi) gradient preconditioner for VFE_3.0 (Gaussian-specialized).

Conditions a Euclidean gradient grad_phi (coordinates in a generator basis) before
the Lie-algebra retraction. A config-selected registry of metrics:
  none              identity (the canonical update: no metric correction; the
                    gradient lives in the Lie algebra g, a vector space).
  clip              norm-clip baseline grad * min(1, c / ||grad||).
  killing           Cartan-involution metric g~ = 2K*gram - 2*tr(x)tr(.), center-
                    regularized then inverted (natural gradient grad @ g~^{-1}).
  killing_per_block block-diagonal Killing metric (per irrep block).
  pullback          position-dependent natural gradient via the differential of
                    the exponential map: G_ab(phi) = <d exp_phi(T_a), d exp_phi(T_b)>_F.
Coordinates in, coordinates out (..., n_gen) -- same units as retract_phi's
delta_phi, so the E-step is grad -> precondition -> retract. Pure: takes a
generator TENSOR, not a GaugeGroup.
"""

import math
from typing import Callable, Dict, List, Optional

import torch

from vfe3.geometry.lie_ops import gram_pinv

_PRECOND: Dict[str, Callable[..., torch.Tensor]] = {}


def register_precond(name: str) -> Callable:
    """Decorator registering a preconditioning rule grad_phi -> preconditioned grad."""
    def _wrap(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        _PRECOND[name] = fn
        return fn
    return _wrap


def get_precond(name: str) -> Callable[..., torch.Tensor]:
    """Return the registered preconditioning rule (KeyError if absent)."""
    if name not in _PRECOND:
        raise KeyError(f"no preconditioner {name!r}; available: {sorted(_PRECOND)}")
    return _PRECOND[name]


@register_precond("none")
def _precond_none(
    grad_phi:   torch.Tensor,             # (..., n_gen)
    phi:        torch.Tensor,             # (..., n_gen) (unused)
    generators: torch.Tensor,             # (n_gen, K, K) (unused)

    **kwargs,
) -> torch.Tensor:
    r"""Identity: the canonical no-correction update (gradient lives in g)."""
    return grad_phi


@register_precond("clip")
def _precond_clip(
    grad_phi:   torch.Tensor,             # (..., n_gen)
    phi:        torch.Tensor,             # (..., n_gen) (unused)
    generators: torch.Tensor,             # (n_gen, K, K) (unused)

    *,
    clip_c:     float = 10.0,
    eps:        float = 1e-6,
    **kwargs,
) -> torch.Tensor:
    r"""Norm-clip baseline grad * min(1, clip_c / ||grad||)."""
    norm = grad_phi.norm(dim=-1, keepdim=True)
    return torch.where(norm > clip_c, grad_phi * (clip_c / (norm + eps)), grad_phi)


def killing_metric(
    generators: torch.Tensor,             # (n_gen, K, K) basis
) -> torch.Tensor:                        # (n_gen, n_gen) Cartan-involution metric
    r"""Cartan-involution Killing metric g~_ab = 2K*tr(G_a^T G_b) - 2 tr(G_a) tr(G_b).

    Equals -B(theta X, Y) with theta(X) = -X^T; positive-definite on sl(K). The
    bare Killing form B(X,Y) = 2K*tr(XY) - 2 tr(X)tr(Y) is indefinite and is NOT
    used. ``gram`` is the FROBENIUS inner product tr(G_a^T G_b).
    """
    K = generators.shape[-1]
    gram   = torch.einsum("aij,bij->ab", generators, generators)      # tr(G_a^T G_b)
    traces = generators.diagonal(dim1=-2, dim2=-1).sum(-1)            # (n_gen,)
    return 2.0 * K * gram - 2.0 * torch.outer(traces, traces)


def build_killing_preconditioner(
    generators: torch.Tensor,             # (n_gen, K, K) basis

    *,
    center_reg: Optional[float] = None,   # None -> 2*K; lifts the numerical nullspace
    tol:        float           = 1e-6,
) -> torch.Tensor:                        # (n_gen, n_gen) regularized inverse metric
    r"""Inverse Killing metric, regularizing only the numerical nullspace.

    eigh(g~) -> (lambda, V); eigenvalues with |lambda| < tol (the center/identity
    direction) are lifted to ``center_reg`` before inversion. Non-null eigenvalues
    are untouched, so the inverse is EXACT on sl(K) (a ridge center_reg*I is not).
    so(K) (already positive-definite) acquires no regularization. eigh in float64.
    """
    K = generators.shape[-1]
    reg = float(2 * K) if center_reg is None else float(center_reg)
    orig_dtype = generators.dtype
    M = killing_metric(generators).double()
    M = 0.5 * (M + M.transpose(-1, -2))
    evals, evecs = torch.linalg.eigh(M)
    evals = torch.where(evals.abs() < tol, torch.full_like(evals, reg), evals)
    inv = (evecs * (1.0 / evals).unsqueeze(-2)) @ evecs.transpose(-1, -2)
    return inv.to(orig_dtype)


@register_precond("killing")
def _precond_killing(
    grad_phi:   torch.Tensor,             # (..., n_gen)
    phi:        torch.Tensor,             # (..., n_gen) (unused)
    generators: torch.Tensor,             # (n_gen, K, K)

    *,
    center_reg: Optional[float]        = None,
    inv_metric: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    r"""Natural gradient under the (center-regularized) Killing metric: grad @ g~^{-1}."""
    Minv = build_killing_preconditioner(generators, center_reg=center_reg) if inv_metric is None else inv_metric
    return torch.einsum("...a,ab->...b", grad_phi, Minv)


def precondition_phi_gradient(
    grad_phi:     torch.Tensor,           # (..., n_gen) Euclidean grad wrt phi coords
    phi:          torch.Tensor,           # (..., n_gen) current frame (used by pullback)
    generators:   torch.Tensor,           # (n_gen, K, K)

    *,
    clip_c:       float = 10.0,
    series_order: int   = 6,
    mode:         str   = "none",

    center_reg:   Optional[float]        = None,   # None -> 2*K
    irrep_dims:   Optional[List[int]]    = None,   # required for killing_per_block
    inv_metric:   Optional[torch.Tensor] = None,   # cached Killing inverse (n_gen, n_gen)
) -> torch.Tensor:                        # (..., n_gen) preconditioned gradient
    r"""Dispatch to the registered preconditioning rule `mode` (default 'none')."""
    return get_precond(mode)(
        grad_phi, phi, generators,
        clip_c=clip_c, series_order=series_order, center_reg=center_reg,
        irrep_dims=irrep_dims, inv_metric=inv_metric,
    )
