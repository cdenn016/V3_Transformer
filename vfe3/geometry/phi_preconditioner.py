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


def _generator_block_index(
    generators: torch.Tensor,             # (n_gen, K, K)
    irrep_dims: List[int],                # block sizes; sum == K

    *,
    tol:        float = 1e-9,
) -> torch.Tensor:                        # (n_gen,) block id per generator
    r"""Block membership of each generator (asserts single-block support)."""
    bounds, start = [], 0
    for d in irrep_dims:
        bounds.append((start, start + d))
        start += d
    n_gen = generators.shape[0]
    block_of = torch.full((n_gen,), -1, dtype=torch.long, device=generators.device)
    for a in range(n_gen):
        mass  = [float(generators[a, s:e, s:e].abs().sum()) for (s, e) in bounds]
        total = float(generators[a].abs().sum())
        hits  = [h for h, m in enumerate(mass) if m > tol]
        if len(hits) != 1 or abs(sum(mass) - total) > tol:
            raise ValueError(f"generator {a} is not supported in a single irrep block")
        block_of[a] = hits[0]
    return block_of


def build_killing_preconditioner_per_block(
    generators: torch.Tensor,             # (n_gen, K, K)
    irrep_dims: List[int],                # block sizes; sum == K

    *,
    center_reg: Optional[float] = None,
    tol:        float           = 1e-6,
) -> torch.Tensor:                        # (n_gen, n_gen) block-diagonal inverse metric
    r"""Block-diagonal Killing inverse: per-block local-dimension Cartan metric.

    Single global block (irrep_dims == [K], e.g. cross-coupled bases) reduces to
    build_killing_preconditioner. Otherwise each generator's own block supplies
    the local Killing metric (block dimension d_h), with no cross-block coupling.
    """
    if len(irrep_dims) == 1:
        return build_killing_preconditioner(generators, center_reg=center_reg, tol=tol)
    block_of = _generator_block_index(generators, irrep_dims)
    n_gen = generators.shape[0]
    Minv  = torch.zeros(n_gen, n_gen, dtype=generators.dtype, device=generators.device)
    start = 0
    for h, d in enumerate(irrep_dims):
        idx     = (block_of == h).nonzero(as_tuple=True)[0]
        sub     = generators[idx][:, start:start + d, start:start + d].contiguous()   # local d_h rep
        sub_inv = build_killing_preconditioner(sub, center_reg=center_reg, tol=tol)
        Minv[idx.unsqueeze(-1), idx.unsqueeze(0)] = sub_inv
        start += d
    return Minv


@register_precond("killing_per_block")
def _precond_killing_per_block(
    grad_phi:   torch.Tensor,             # (..., n_gen)
    phi:        torch.Tensor,             # (..., n_gen) (unused)
    generators: torch.Tensor,             # (n_gen, K, K)

    *,
    center_reg: Optional[float]        = None,
    irrep_dims: Optional[List[int]]    = None,
    inv_metric: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    r"""Natural gradient under the per-block Killing metric."""
    if inv_metric is None:
        if irrep_dims is None:
            raise ValueError("killing_per_block requires irrep_dims")
        inv_metric = build_killing_preconditioner_per_block(generators, irrep_dims, center_reg=center_reg)
    return torch.einsum("...a,ab->...b", grad_phi, inv_metric)


def _structure_constants(
    generators: torch.Tensor,             # (n_gen, K, K)

    *,
    gram_pinv_: Optional[torch.Tensor] = None,
) -> torch.Tensor:                        # (n_gen, n_gen, n_gen) f[a,b,c]: [G_a,G_b]=sum_c f G_c
    r"""Structure constants f[a,b,c] = coords_c([G_a, G_b]) in the generator basis."""
    G = generators
    brak   = torch.einsum("aij,bjk->abik", G, G) - torch.einsum("bij,ajk->abik", G, G)   # [G_a,G_b]
    gp     = gram_pinv(G) if gram_pinv_ is None else gram_pinv_
    coords = torch.einsum("cij,abij->abc", G, brak)       # <G_c, [G_a,G_b]>
    return torch.einsum("abc,cd->abd", coords, gp)


def pullback_metric(
    phi:          torch.Tensor,           # (..., n_gen) frame coordinates
    generators:   torch.Tensor,           # (n_gen, K, K)

    *,
    series_tol:   float = 1e-12,
    series_order: int   = 40,
    max_k:        int   = 12,
) -> torch.Tensor:                        # (..., n_gen, n_gen) position-dependent metric
    r"""Pullback natural-gradient metric G_ab(phi) = <d exp_phi(T_a), d exp_phi(T_b)>_F.

    d exp_phi(T) = Psi(ad_phi)(T) exp(phi), Psi(z) = (e^z - 1)/z = sum_k z^k/(k+1)!.
    ad_phi acts on coordinates: (ad_phi)_{cb} = sum_a phi^a f[a,b,c]. The Psi series is
    summed adaptively: terms accumulate until the new term's max |entry| < series_tol,
    capped at series_order. Truncation error of Psi(ad_phi) grows with ||phi||, so a
    fixed low order is inaccurate in the non-compact (large-norm) regime the pullback
    metric exists for; the adaptive cutoff keeps it accurate up to retract_phi's max_norm.
    The 1/(k+1)! coefficient is a float (an int divisor overflows past order ~20). The
    structure-constants tensor is O(n_gen^2 K^2); guarded for K > max_k (infeasible for
    large K). The finite-difference of exp is the correctness arbiter for this kernel.
    """
    K = generators.shape[-1]
    if K > max_k:
        raise ValueError(f"pullback_metric: K={K} exceeds max_k={max_k} (structure-constants OOM)")
    n_gen      = generators.shape[0]
    orig_dtype = phi.dtype
    G          = generators.double()
    phi        = phi.double()

    f  = _structure_constants(G)                           # (n_gen,n_gen,n_gen) f[a,b,c]
    ad = torch.einsum("...a,abc->...cb", phi, f)           # (...,n_gen,n_gen) (ad_phi)_{cb}

    eye    = torch.eye(n_gen, dtype=ad.dtype, device=ad.device).expand_as(ad).clone()
    psi    = eye.clone()
    ad_pow = eye.clone()
    for k in range(1, series_order):
        ad_pow = torch.einsum("...ij,...jk->...ik", ad_pow, ad)
        term   = ad_pow * (1.0 / float(math.factorial(k + 1)))   # float coeff: int overflows >~20
        psi    = psi + term
        if float(term.abs().max()) < series_tol:           # converged: higher terms negligible
            break

    # d exp_phi(e_a) coords = psi @ e_a = column a of psi -> embed -> times exp(phi)
    W       = torch.einsum("...ca,cij->...aij", psi, G)    # (...,n_gen,K,K) Psi(ad_phi)(T_a)
    exp_phi = torch.linalg.matrix_exp(torch.einsum("...a,aij->...ij", phi, G))
    dexp    = torch.einsum("...aij,...jk->...aik", W, exp_phi)
    metric  = torch.einsum("...aij,...bij->...ab", dexp, dexp)
    return metric.to(orig_dtype)


@register_precond("pullback")
def _precond_pullback(
    grad_phi:     torch.Tensor,           # (..., n_gen)
    phi:          torch.Tensor,           # (..., n_gen)
    generators:   torch.Tensor,           # (n_gen, K, K)

    *,
    series_tol:   float = 1e-12,
    eps:          float = 1e-6,
    series_order: int   = 40,
    **kwargs,
) -> torch.Tensor:
    r"""Position-dependent natural gradient: solve G(phi) nat = grad_phi."""
    G_metric = pullback_metric(phi, generators, series_tol=series_tol, series_order=series_order)
    eye = torch.eye(G_metric.shape[-1], dtype=G_metric.dtype, device=G_metric.device)
    sol = torch.linalg.solve(G_metric + eps * eye, grad_phi.unsqueeze(-1))
    return sol.squeeze(-1)


def precondition_phi_gradient(
    grad_phi:     torch.Tensor,           # (..., n_gen) Euclidean grad wrt phi coords
    phi:          torch.Tensor,           # (..., n_gen) current frame (used by pullback)
    generators:   torch.Tensor,           # (n_gen, K, K)

    *,
    clip_c:       float = 10.0,
    series_tol:   float = 1e-12,
    series_order: int   = 40,
    mode:         str   = "none",

    center_reg:   Optional[float]        = None,   # None -> 2*K
    irrep_dims:   Optional[List[int]]    = None,   # required for killing_per_block
    inv_metric:   Optional[torch.Tensor] = None,   # cached Killing inverse (n_gen, n_gen)
) -> torch.Tensor:                        # (..., n_gen) preconditioned gradient
    r"""Dispatch to the registered preconditioning rule `mode` (default 'none')."""
    return get_precond(mode)(
        grad_phi, phi, generators,
        clip_c=clip_c, series_tol=series_tol, series_order=series_order,
        center_reg=center_reg, irrep_dims=irrep_dims, inv_metric=inv_metric,
    )
