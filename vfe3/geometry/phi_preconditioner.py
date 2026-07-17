r"""Gauge-frame (phi) gradient preconditioner for VFE_3.0 (Gaussian-specialized).

Conditions a Euclidean gradient grad_phi (coordinates in a generator basis) before
the Lie-algebra retraction. A config-selected registry of metrics:
  none               identity (the canonical update: no metric correction; the
                     gradient lives in the Lie algebra g, a vector space).
  clip               norm-clip baseline grad * min(1, c / ||grad||).
  killing            Cartan-involution metric g~ = 2K*gram - 2*tr(x)tr(.), center-
                     regularized then inverted (natural gradient grad @ g~^{-1}).
  killing_per_block  block-diagonal Killing metric (per irrep block).
  pullback           position-dependent natural gradient via the differential of
                     the exponential map: G_ab(phi) = <d exp_phi(T_a), d exp_phi(T_b)>_F.
  pullback_per_block per-irrep-block pullback metric (each block's local d_h-dim rep;
                     feasible at K > max_k where the full pullback raises).
Coordinates in, coordinates out (..., n_gen) -- same units as retract_phi's
delta_phi, so the E-step is grad -> precondition -> retract. Pure: takes a
generator TENSOR, not a GaugeGroup.

Gauge-equivariance caveat (killing / killing_per_block): the Cartan-involution
metric uses the Frobenius form tr(G_a^T G_b), which is Ad-invariant only under
the COMPACT subgroup (tr((gXg^-1)^T gYg^-1) = tr(X^T Y) iff g^T g = I). So the
Killing-preconditioned natural gradient is gauge-equivariant under SO(N) but NOT
under general GL(K) in the non-compact (symmetric) directions; it is a left-/
Ad(K)-invariant metric, not a bi-invariant one. The pullback metric is the
position-dependent alternative for the non-compact regime.
"""

from collections import OrderedDict
from dataclasses import dataclass
import math
import warnings
import weakref
from typing import Callable, Dict, List, Optional

import torch

from vfe3.geometry.lie_ops import gram_pinv, warn_if_basis_not_closed

_PRECOND: Dict[str, Callable[..., torch.Tensor]] = {}


@dataclass(frozen=True)
class PullbackGroupDirectionResult:
    v_phi:                               torch.Tensor
    xi:                                  torch.Tensor
    min_undamped_generalized_eigenvalue: torch.Tensor
    undamped_generalized_condition:      torch.Tensor
    damped_generalized_condition:        torch.Tensor
    scaled_solve_residual:               torch.Tensor
    series_order:                        int


_PHI_GROUP_DIRECTIONS: Dict[str, Callable[..., PullbackGroupDirectionResult]] = {}

_PHI_GROUP_MIN_SERIES_ORDER: int   = 40
_PHI_GROUP_MAX_SERIES_ORDER: int   = 128
_PHI_GROUP_SERIES_ORDER_STEP: int  = 8
_PHI_GROUP_TAIL_TOL: float         = 1e-12
_PHI_GROUP_GRAM_RIDGE: float       = 1e-6
_PHI_GROUP_MIN_UNDAMPED_EIG: float = 1e-8
_PHI_GROUP_MAX_DAMPED_COND: float  = 1e6
_PHI_GROUP_SOLVE_RESIDUAL: float   = 1e-10


def register_precond(name: str, *, override: bool = False) -> Callable:
    """Decorator registering a preconditioning rule grad_phi -> preconditioned grad.

    Duplicate keys fail closed (audit 2026-07-01 round-3): a second registration under an
    existing name silently shadowed the first. Pass ``override=True`` to replace deliberately.
    """
    def _wrap(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        if name in _PRECOND and not override:
            raise KeyError(f"phi preconditioner {name!r} already registered; pass override=True to replace")
        _PRECOND[name] = fn
        return fn
    return _wrap


def get_precond(name: str) -> Callable[..., torch.Tensor]:
    """Return the registered preconditioning rule (KeyError if absent)."""
    if name not in _PRECOND:
        raise KeyError(f"no preconditioner {name!r}; available: {sorted(_PRECOND)}")
    return _PRECOND[name]


def register_phi_group_direction(
    name: str,

    *,
    override: bool = False,
) -> Callable:
    """Register one strict M-step pullback-group direction rule."""
    def _wrap(
        fn: Callable[..., PullbackGroupDirectionResult],
    ) -> Callable[..., PullbackGroupDirectionResult]:
        if name in _PHI_GROUP_DIRECTIONS and not override:
            raise KeyError(
                f"phi group direction {name!r} already registered; pass override=True to replace"
            )
        _PHI_GROUP_DIRECTIONS[name] = fn
        return fn
    return _wrap


def get_phi_group_direction(
    name: str,
) -> Callable[..., PullbackGroupDirectionResult]:
    """Return a registered strict M-step pullback-group direction rule."""
    if name not in _PHI_GROUP_DIRECTIONS:
        raise KeyError(
            f"no phi group direction {name!r}; available: {sorted(_PHI_GROUP_DIRECTIONS)}"
        )
    return _PHI_GROUP_DIRECTIONS[name]


def pullback_group_direction(
    grad_phi:   torch.Tensor,             # (..., n_gen) processed outer-objective covector
    phi:        torch.Tensor,             # (..., n_gen) current chart coordinates
    generators: torch.Tensor,             # (n_gen, K, K) registered basis

    *,
    mode:       str,
    irrep_dims: Optional[List[int]] = None,
) -> PullbackGroupDirectionResult:
    r"""Return the strict M-step direction in an autocast-disabled float64 island.

    The chart covector is solved against the Gram-relative pullback system
    ``(J(phi)^T J(phi) + 1e-6 B) v_phi = grad_phi`` and then converted to the
    left-trivialized group velocity ``xi = Psi_L(ad_phi) v_phi``.
    """
    with torch.autocast(device_type=grad_phi.device.type, enabled=False):
        return get_phi_group_direction(mode)(
            grad_phi,
            phi,
            generators,
            irrep_dims=irrep_dims,
        )


def _strict_structure_constants(
    generators: torch.Tensor,             # (n_gen, K, K)

    *,
    closure_tol: float = 1e-4,
    max_k:       int   = 12,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Return certified bracket coordinates and the generator Gram in float64.

    For ``B_cd = <G_c, G_d>_F`` and ``[G_a, G_b] = f_ab^c G_c``, the coordinates
    solve ``B_cd f_ab^d = <G_c, [G_a, G_b]>_F`` through the Cholesky factor of
    ``B``. Reconstruction of every bracket certifies closure of the span.
    """
    K = generators.shape[-1]
    if K > max_k:
        raise ValueError(f"strict phi group direction: K={K} exceeds max_k={max_k}")
    basis = generators.to(dtype=torch.float64)
    gram = torch.einsum("aij,bij->ab", basis, basis)
    gram_factor, info = torch.linalg.cholesky_ex(gram)
    if bool((info != 0).any()):
        raise ValueError("strict phi group direction requires an independent generator basis")
    bracket = (
        torch.einsum("aij,bjk->abik", basis, basis)
        - torch.einsum("bij,ajk->abik", basis, basis)
    )
    pairings = torch.einsum("cij,abij->abc", basis, bracket)
    n_gen = basis.shape[0]
    coordinates = torch.cholesky_solve(
        pairings.reshape(-1, n_gen).transpose(0, 1),
        gram_factor,
    ).transpose(0, 1).reshape(n_gen, n_gen, n_gen)
    reconstructed = torch.einsum("abc,cij->abij", coordinates, basis)
    residual = torch.linalg.matrix_norm(reconstructed - bracket, dim=(-2, -1))
    bracket_norm = torch.linalg.matrix_norm(bracket, dim=(-2, -1))
    relative = residual / (bracket_norm + torch.finfo(torch.float64).eps)
    if not bool(torch.isfinite(relative).all()) or float(relative.max()) > closure_tol:
        raise ValueError(
            "strict phi group direction requires a bracket-closed generator span; "
            f"relative reconstruction residual={float(relative.max()):.3e}"
        )
    return coordinates, gram


def _adaptive_phi_differentials(
    ad: torch.Tensor,                     # (..., n_gen, n_gen)
) -> tuple[torch.Tensor, torch.Tensor, int]:
    r"""Evaluate certified right and left trivialized exponential differentials.

    The right series is ``Psi_R(z) = (exp(z) - 1) / z = sum z^k/(k+1)!`` and
    the left series is ``Psi_L(z) = (1 - exp(-z)) / z = sum (-z)^k/(k+1)!``.
    A geometric majorant for the first omitted term certifies both series at
    candidate orders from 40 through 128.
    """
    n_gen = ad.shape[-1]
    eye = torch.eye(n_gen, dtype=ad.dtype, device=ad.device).expand_as(ad).clone()
    psi_right = eye.clone()
    psi_left = eye.clone()
    term_right = eye.clone()
    term_left = eye.clone()
    norm_one = torch.linalg.matrix_norm(ad, ord=1, dim=(-2, -1))
    norm_inf = torch.linalg.matrix_norm(ad, ord=float("inf"), dim=(-2, -1))
    use_one = norm_one <= norm_inf
    alpha = torch.where(use_one, norm_one, norm_inf)
    bound_k = torch.ones_like(alpha)
    for k in range(1, _PHI_GROUP_MAX_SERIES_ORDER):
        term_right = torch.matmul(term_right, ad) / float(k + 1)
        term_left = -torch.matmul(term_left, ad) / float(k + 1)
        psi_right = psi_right + term_right
        psi_left = psi_left + term_left
        bound_k = bound_k * alpha / float(k + 1)
        order = k + 1
        if order < _PHI_GROUP_MIN_SERIES_ORDER or order % _PHI_GROUP_SERIES_ORDER_STEP:
            continue
        first_omitted = bound_k * alpha / float(order + 1)
        ratio = alpha / float(order + 2)
        tail = torch.where(
            ratio < 1.0,
            first_omitted / (1.0 - ratio),
            torch.full_like(ratio, torch.inf),
        )
        right_one = torch.linalg.matrix_norm(psi_right, ord=1, dim=(-2, -1))
        right_inf = torch.linalg.matrix_norm(psi_right, ord=float("inf"), dim=(-2, -1))
        left_one = torch.linalg.matrix_norm(psi_left, ord=1, dim=(-2, -1))
        left_inf = torch.linalg.matrix_norm(psi_left, ord=float("inf"), dim=(-2, -1))
        right_norm = torch.where(use_one, right_one, right_inf)
        left_norm = torch.where(use_one, left_one, left_inf)
        right_ok = tail <= _PHI_GROUP_TAIL_TOL * torch.maximum(torch.ones_like(tail), right_norm)
        left_ok = tail <= _PHI_GROUP_TAIL_TOL * torch.maximum(torch.ones_like(tail), left_norm)
        if bool((right_ok & left_ok & torch.isfinite(tail)).all()):
            return psi_right, psi_left, order
    raise ValueError(
        f"strict phi group differential series certificate failed by order "
        f"{_PHI_GROUP_MAX_SERIES_ORDER}"
    )


def _full_pullback_group_direction(
    grad_phi:   torch.Tensor,             # (..., n_gen)
    phi:        torch.Tensor,             # (..., n_gen)
    generators: torch.Tensor,             # (n_gen, K, K)
) -> PullbackGroupDirectionResult:
    r"""Compute one certified full-block pullback direction.

    With ``J_a = D exp_X[G_a]`` and ``M_ab = <J_a, J_b>_F``, this solves
    ``A v_phi = grad_phi`` for ``A = sym(M) + 1e-6 B`` by Cholesky. The returned
    group velocity is ``xi = Psi_L(ad_phi) v_phi``. Generalized eigenvalues are
    those of ``B^{-1/2} sym(M) B^{-1/2}`` and are certified before damping.
    """
    phi_64 = phi.to(dtype=torch.float64)
    grad_64 = grad_phi.to(dtype=torch.float64)
    basis_64 = generators.to(dtype=torch.float64)
    structure, gram = _strict_structure_constants(basis_64)
    ad = torch.einsum("...a,abc->...cb", phi_64, structure)
    psi_right, psi_left, series_order = _adaptive_phi_differentials(ad)
    pushed = torch.einsum("...ca,cij->...aij", psi_right, basis_64)
    exp_phi = torch.linalg.matrix_exp(torch.einsum("...a,aij->...ij", phi_64, basis_64))
    differential = torch.einsum("...aij,...jk->...aik", pushed, exp_phi)
    metric = torch.einsum("...aij,...bij->...ab", differential, differential)
    metric = 0.5 * (metric + metric.transpose(-1, -2))
    gram_factor, gram_info = torch.linalg.cholesky_ex(gram)
    if bool((gram_info != 0).any()):
        raise ValueError("strict phi group direction generator Gram factorization failed")
    identity = torch.eye(gram.shape[-1], dtype=gram.dtype, device=gram.device)
    gram_inverse_half = torch.linalg.solve_triangular(gram_factor, identity, upper=False)
    whitened = torch.matmul(torch.matmul(gram_inverse_half, metric), gram_inverse_half.transpose(-1, -2))
    undamped_eigenvalues = torch.linalg.eigvalsh(0.5 * (whitened + whitened.transpose(-1, -2)))
    min_undamped = undamped_eigenvalues[..., 0]
    max_undamped = undamped_eigenvalues[..., -1]
    undamped_condition = max_undamped / min_undamped
    undamped_finite = bool(torch.isfinite(undamped_eigenvalues).all())
    if not undamped_finite or bool((min_undamped < _PHI_GROUP_MIN_UNDAMPED_EIG).any()):
        minimum = float(min_undamped.min())
        raise ValueError(
            "strict phi group direction undamped generalized eigenvalue "
            f"{minimum:.3e} is below {_PHI_GROUP_MIN_UNDAMPED_EIG:.1e}"
        )
    damped = metric + _PHI_GROUP_GRAM_RIDGE * gram
    damped_eigenvalues = undamped_eigenvalues + _PHI_GROUP_GRAM_RIDGE
    damped_condition = damped_eigenvalues[..., -1] / damped_eigenvalues[..., 0]
    if (
        not bool(torch.isfinite(damped_condition).all())
        or bool((damped_condition > _PHI_GROUP_MAX_DAMPED_COND).any())
    ):
        maximum = float(damped_condition.max())
        raise ValueError(
            "strict phi group direction damped generalized condition "
            f"{maximum:.3e} exceeds {_PHI_GROUP_MAX_DAMPED_COND:.1e}"
        )
    factor, info = torch.linalg.cholesky_ex(damped)
    if bool((info != 0).any()):
        raise ValueError("strict phi group direction damped Cholesky factorization failed")
    v_phi = torch.cholesky_solve(grad_64.unsqueeze(-1), factor).squeeze(-1)
    residual = torch.linalg.vector_norm(
        torch.einsum("...ab,...b->...a", damped, v_phi) - grad_64,
        dim=-1,
    )
    scale = (
        torch.linalg.matrix_norm(damped, ord=2, dim=(-2, -1))
        * torch.linalg.vector_norm(v_phi, dim=-1)
        + torch.linalg.vector_norm(grad_64, dim=-1)
    )
    scaled_residual = residual / scale.clamp_min(torch.finfo(torch.float64).tiny)
    if (
        not bool(torch.isfinite(scaled_residual).all())
        or bool((scaled_residual > _PHI_GROUP_SOLVE_RESIDUAL).any())
    ):
        maximum = float(scaled_residual.max())
        raise ValueError(
            "strict phi group direction scaled solve residual "
            f"{maximum:.3e} exceeds {_PHI_GROUP_SOLVE_RESIDUAL:.1e}"
        )
    xi = torch.einsum("...ab,...b->...a", psi_left, v_phi)
    finite = all(
        bool(torch.isfinite(value).all())
        for value in (v_phi, xi, undamped_condition)
    )
    if not finite:
        raise ValueError("strict phi group direction produced a nonfinite certificate")
    return PullbackGroupDirectionResult(
        v_phi=v_phi,
        xi=xi,
        min_undamped_generalized_eigenvalue=min_undamped,
        undamped_generalized_condition=undamped_condition,
        damped_generalized_condition=damped_condition,
        scaled_solve_residual=scaled_residual,
        series_order=series_order,
    )


@register_phi_group_direction("pullback")
def _pullback_group_direction(
    grad_phi:   torch.Tensor,             # (..., n_gen)
    phi:        torch.Tensor,             # (..., n_gen)
    generators: torch.Tensor,             # (n_gen, K, K)

    *,
    irrep_dims: Optional[List[int]] = None,
) -> PullbackGroupDirectionResult:
    del irrep_dims
    return _full_pullback_group_direction(grad_phi, phi, generators)


@register_phi_group_direction("pullback_per_block")
def _pullback_group_direction_per_block(
    grad_phi:   torch.Tensor,             # (..., n_gen)
    phi:        torch.Tensor,             # (..., n_gen)
    generators: torch.Tensor,             # (n_gen, K, K)

    *,
    irrep_dims: Optional[List[int]] = None,
) -> PullbackGroupDirectionResult:
    r"""Solve each direct-sum block independently and aggregate global extrema.

    For ``g = direct_sum_h g_h``, each local system is
    ``A_h v_h = grad_h``. The global generalized condition is
    ``max_h(lambda_max,h) / min_h(lambda_min,h)`` rather than the largest local
    condition, while the solve certificate is the maximum local residual.
    """
    if irrep_dims is None:
        raise ValueError("pullback_per_block requires irrep_dims")
    if len(irrep_dims) == 1:
        return _full_pullback_group_direction(grad_phi, phi, generators)
    if any(d <= 0 for d in irrep_dims) or sum(irrep_dims) != generators.shape[-1]:
        raise ValueError("pullback_per_block requires positive irrep_dims summing to K")

    block_of = _generator_block_index(generators, irrep_dims)
    grad_64 = grad_phi.to(dtype=torch.float64)
    phi_64 = phi.to(dtype=torch.float64)
    v_phi = torch.zeros_like(grad_64)
    xi = torch.zeros_like(grad_64)
    local_results = []
    start = 0
    for block, dimension in enumerate(irrep_dims):
        index = (block_of == block).nonzero(as_tuple=True)[0]
        if index.numel() == 0:
            raise ValueError(f"pullback_per_block block {block} has no generators")
        local_basis = generators[index][
            :,
            start:start + dimension,
            start:start + dimension,
        ].contiguous()
        local = _full_pullback_group_direction(
            grad_64[..., index],
            phi_64[..., index],
            local_basis,
        )
        v_phi = v_phi.index_copy(-1, index, local.v_phi)
        xi = xi.index_copy(-1, index, local.xi)
        local_results.append(local)
        start += dimension

    local_min_undamped = torch.stack(
        [result.min_undamped_generalized_eigenvalue for result in local_results],
        dim=0,
    )
    local_max_undamped = torch.stack(
        [
            result.min_undamped_generalized_eigenvalue
            * result.undamped_generalized_condition
            for result in local_results
        ],
        dim=0,
    )
    min_undamped = local_min_undamped.amin(dim=0)
    max_undamped = local_max_undamped.amax(dim=0)
    undamped_condition = max_undamped / min_undamped
    if (
        not bool(torch.isfinite(min_undamped).all())
        or bool((min_undamped < _PHI_GROUP_MIN_UNDAMPED_EIG).any())
    ):
        minimum = float(min_undamped.min())
        raise ValueError(
            "strict phi group direction undamped generalized eigenvalue "
            f"{minimum:.3e} is below {_PHI_GROUP_MIN_UNDAMPED_EIG:.1e}"
        )

    local_min_damped = local_min_undamped + _PHI_GROUP_GRAM_RIDGE
    local_max_damped = local_max_undamped + _PHI_GROUP_GRAM_RIDGE
    damped_condition = local_max_damped.amax(dim=0) / local_min_damped.amin(dim=0)
    if (
        not bool(torch.isfinite(damped_condition).all())
        or bool((damped_condition > _PHI_GROUP_MAX_DAMPED_COND).any())
    ):
        maximum = float(damped_condition.max())
        raise ValueError(
            "strict phi group direction damped generalized condition "
            f"{maximum:.3e} exceeds {_PHI_GROUP_MAX_DAMPED_COND:.1e}"
        )

    scaled_residual = torch.stack(
        [result.scaled_solve_residual for result in local_results],
        dim=0,
    ).amax(dim=0)
    if (
        not bool(torch.isfinite(scaled_residual).all())
        or bool((scaled_residual > _PHI_GROUP_SOLVE_RESIDUAL).any())
    ):
        maximum = float(scaled_residual.max())
        raise ValueError(
            "strict phi group direction scaled solve residual "
            f"{maximum:.3e} exceeds {_PHI_GROUP_SOLVE_RESIDUAL:.1e}"
        )

    if not all(bool(torch.isfinite(value).all()) for value in (v_phi, xi, undamped_condition)):
        raise ValueError("strict phi group direction produced a nonfinite certificate")
    return PullbackGroupDirectionResult(
        v_phi=v_phi,
        xi=xi,
        min_undamped_generalized_eigenvalue=min_undamped,
        undamped_generalized_condition=undamped_condition,
        damped_generalized_condition=damped_condition,
        scaled_solve_residual=scaled_residual,
        series_order=max(result.series_order for result in local_results),
    )


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


# Memoized Killing inverses, weakly keyed on generator identity and mutation version. The inverse
# depends only on the fixed generator basis, so it is loop-invariant across every E-step iteration;
# caching avoids rebuilding an O(n_gen^3) float64 eigh when a Killing preconditioner is active.
_KILLING_INV_CACHE_MAXSIZE: int = 32
_KILLING_INV_CACHE: OrderedDict[
    tuple, tuple[weakref.ReferenceType[torch.Tensor], torch.Tensor]
] = OrderedDict()
_PULLBACK_SERIES_WARNED: bool = False           # warn-once guard for Psi-series non-convergence (m19)


def _killing_cache_key(
    generators: torch.Tensor,             # (n_gen, K, K) basis
    variant:    tuple,                    # ("full",) or ("per_block", irrep_dims)

    *,
    tol:        float           = 1e-6,
    center_reg: Optional[float] = None,
) -> tuple:
    """Return the identity/version cache key for one Killing-inverse variant."""
    return (
        id(generators), int(generators._version), tuple(generators.shape),
        generators.dtype, generators.device, variant, center_reg, tol,
    )


def _get_cached_killing_inverse(
    generators: torch.Tensor,             # (n_gen, K, K) basis
    key:        tuple,
) -> Optional[torch.Tensor]:
    """Return an identity-valid cache hit and promote it to most recently used."""
    cached = _KILLING_INV_CACHE.get(key)
    if cached is None:
        return None
    generators_ref, inverse = cached
    if generators_ref() is not generators:
        _KILLING_INV_CACHE.pop(key, None)
        return None
    _KILLING_INV_CACHE.move_to_end(key)
    return inverse


def _cache_killing_inverse(
    generators: torch.Tensor,             # (n_gen, K, K) basis
    inverse:    torch.Tensor,             # (n_gen, n_gen) inverse metric
    key:        tuple,
) -> torch.Tensor:
    """Store one weak cache entry and evict the least-recently-used overflow."""
    def _remove_dead_entry(
        generators_ref: weakref.ReferenceType[torch.Tensor],
    ) -> None:
        cached = _KILLING_INV_CACHE.get(key)
        if cached is not None and cached[0] is generators_ref:
            _KILLING_INV_CACHE.pop(key, None)

    generators_ref = weakref.ref(generators, _remove_dead_entry)
    _KILLING_INV_CACHE[key] = (generators_ref, inverse)
    _KILLING_INV_CACHE.move_to_end(key)
    while len(_KILLING_INV_CACHE) > _KILLING_INV_CACHE_MAXSIZE:
        _KILLING_INV_CACHE.popitem(last=False)
    return inverse


def _build_killing_preconditioner_uncached(
    generators: torch.Tensor,             # (n_gen, K, K) basis

    *,
    center_reg: Optional[float] = None,   # None -> 2*K; lifts the numerical nullspace
    tol:        float           = 1e-6,
) -> torch.Tensor:                        # (n_gen, n_gen) regularized inverse metric
    """Build the inverse Killing metric without reading or writing the shared cache."""
    with torch.no_grad():
        K = generators.shape[-1]
        reg = float(2 * K) if center_reg is None else float(center_reg)
        orig_dtype = generators.dtype
        M = killing_metric(generators).double()
        M = 0.5 * (M + M.transpose(-1, -2))
        evals, evecs = torch.linalg.eigh(M)
        evals = torch.where(evals.abs() < tol, torch.full_like(evals, reg), evals)
        inv = (evecs * (1.0 / evals).unsqueeze(-2)) @ evecs.transpose(-1, -2)
        return inv.to(orig_dtype)


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
    ASSUMPTION (audit 2026-06-13 L12): the magnitude cut ``|lambda| < tol`` isolates ONLY the
    intended center because the Killing/trace-form spectrum of the shipped groups is 0 (the center)
    or O(K) (the semisimple part) -- a clean gap, no genuine eigenvalue near ``tol``. A custom basis
    with a true small-but-nonzero Killing eigenvalue would have it wrongly lifted; on this inactive
    opt-in path (mode='none' is the default pure path) that case does not arise for shipped groups.
    Memoized weakly on the generator basis identity and mutation version (see
    ``_KILLING_INV_CACHE``): loop-invariant, so it is built once per
    (basis, version, center_reg, tol), not per E-step iteration.
    """
    key = _killing_cache_key(
        generators, ("full",), center_reg=center_reg, tol=tol,
    )
    cached = _get_cached_killing_inverse(generators, key)
    if cached is not None:
        return cached
    inverse = _build_killing_preconditioner_uncached(
        generators, center_reg=center_reg, tol=tol,
    )
    return _cache_killing_inverse(generators, inverse, key)


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
    key = _killing_cache_key(
        generators, ("per_block", tuple(irrep_dims)), center_reg=center_reg, tol=tol,
    )
    cached = _get_cached_killing_inverse(generators, key)
    if cached is not None:
        return cached
    block_of = _generator_block_index(generators, irrep_dims)
    n_gen = generators.shape[0]
    Minv  = torch.zeros(n_gen, n_gen, dtype=generators.dtype, device=generators.device)
    start = 0
    for h, d in enumerate(irrep_dims):
        idx     = (block_of == h).nonzero(as_tuple=True)[0]
        sub     = generators[idx][:, start:start + d, start:start + d].contiguous()   # local d_h rep
        sub_inv = _build_killing_preconditioner_uncached(sub, center_reg=center_reg, tol=tol)
        Minv[idx.unsqueeze(-1), idx.unsqueeze(0)] = sub_inv
        start += d
    return _cache_killing_inverse(generators, Minv, key)


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
    closure_tol: float                 = 1e-4,
    eps:         float                 = 1e-12,
    gram_pinv_:  Optional[torch.Tensor] = None,
) -> torch.Tensor:                        # (n_gen, n_gen, n_gen) f[a,b,c]: [G_a,G_b]=sum_c f G_c
    r"""Structure constants f[a,b,c] = coords_c([G_a, G_b]) in the generator basis.

    On a basis NOT closed under the Lie bracket (e.g. a 3+-head ``cross_couplings``
    chain built with ``close_basis=False``) the bracket :math:`[G_a, G_b]` carries an
    out-of-span component that the span projection
    :math:`f[a,b,c] = \langle G_c, [G_a,G_b]\rangle\,(\mathrm{Gram}^+)_{cd}` silently
    truncates, so the structure constants (and any pullback metric built on them) drop
    those terms. A diagnostic guard measures the max relative out-of-span residual
    :math:`\max_{a,b} \lVert [G_a,G_b] - \mathrm{embed}(f_{ab\cdot})\rVert_F /
    (\lVert [G_a,G_b]\rVert_F + \epsilon)` and warns once if it exceeds ``closure_tol``;
    on a closed (default direct-sum) basis the residual is ~0 so the guard is silent and
    the returned tensor is unchanged.
    """
    G = generators
    brak   = torch.einsum("aij,bjk->abik", G, G) - torch.einsum("bij,ajk->abik", G, G)   # [G_a,G_b]
    gp     = gram_pinv(G) if gram_pinv_ is None else gram_pinv_
    coords = torch.einsum("cij,abij->abc", G, brak)       # <G_c, [G_a,G_b]>
    f      = torch.einsum("abc,cd->abd", coords, gp)      # (n_gen, n_gen, n_gen)

    # Diagnostic (cached, one-time): warn if the basis is not bracket-closed, in which case the
    # span projection above truncates the out-of-span part of [G_a,G_b]. Depends only on the fixed
    # generators, so it runs once per basis (the shared lie_ops cache), off the hot path. Size-gated
    # (max_elements): large direct-sum bases are skipped as closed-by-construction (avoids the O(K^4)
    # OOM); small non-closed cross_couplings chains are scanned exactly.
    warn_if_basis_not_closed(G, where="_structure_constants (pullback metric)",
                             closure_tol=closure_tol, eps=eps, gram_pinv_=gp)
    return f


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
    summed through ``series_order`` and its final term is checked against ``series_tol``
    once after the loop. Truncation error of Psi(ad_phi) grows with ||phi||, so a fixed
    low order is inaccurate in the non-compact (large-norm) regime the pullback metric
    exists for; the default order keeps it accurate up to retract_phi's max_norm.
    The 1/(k+1)! coefficient is a float (an int divisor overflows past order ~20). The
    structure-constants tensor is O(n_gen^2 K^2); guarded for K > max_k (infeasible for
    large K). The finite-difference of exp is the correctness arbiter for this kernel.
    """
    K = generators.shape[-1]
    if K > max_k:
        raise ValueError(f"pullback_metric: K={K} exceeds max_k={max_k} (structure-constants OOM)")
    n_gen      = generators.shape[0]
    G          = generators.double()
    phi        = phi.double()

    f  = _structure_constants(G)                           # (n_gen,n_gen,n_gen) f[a,b,c]
    ad = torch.einsum("...a,abc->...cb", phi, f)           # (...,n_gen,n_gen) (ad_phi)_{cb}

    eye    = torch.eye(n_gen, dtype=ad.dtype, device=ad.device).expand_as(ad).clone()
    psi    = eye.clone()
    ad_pow = eye.clone()
    last_term_tensor = torch.zeros((), dtype=ad.dtype, device=ad.device)
    for k in range(1, series_order):
        ad_pow = torch.einsum("...ij,...jk->...ik", ad_pow, ad)
        term   = ad_pow * (1.0 / float(math.factorial(k + 1)))   # float coeff: int overflows >~20
        psi    = psi + term
        last_term_tensor = term.detach().abs().max()
    last_term = float(last_term_tensor)
    converged = last_term < series_tol
    global _PULLBACK_SERIES_WARNED
    if not converged and series_order > 1 and not _PULLBACK_SERIES_WARNED:
        _PULLBACK_SERIES_WARNED = True
        warnings.warn(                                     # m19: exhausting series_order silently returned a truncated metric
            f"pullback_metric: Psi(ad_phi) series did not converge in series_order={series_order} "
            f"(last term {last_term:.2e} >= series_tol={series_tol:.1e}); the pullback preconditioner "
            f"metric may be inaccurate. Raise series_order or reduce ||phi||. Warned once per process.",
            RuntimeWarning, stacklevel=2,
        )

    # d exp_phi(e_a) coords = psi @ e_a = column a of psi -> embed -> times exp(phi)
    W       = torch.einsum("...ca,cij->...aij", psi, G)    # (...,n_gen,K,K) Psi(ad_phi)(T_a)
    exp_phi = torch.linalg.matrix_exp(torch.einsum("...a,aij->...ij", phi, G))
    dexp    = torch.einsum("...aij,...jk->...aik", W, exp_phi)
    metric  = torch.einsum("...aij,...bij->...ab", dexp, dexp)
    return metric


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
    orig_dtype = grad_phi.dtype
    G_metric = pullback_metric(phi, generators, series_tol=series_tol, series_order=series_order)
    eye = torch.eye(G_metric.shape[-1], dtype=G_metric.dtype, device=G_metric.device)
    sol = torch.linalg.solve(G_metric + eps * eye, grad_phi.double().unsqueeze(-1))
    return sol.squeeze(-1).to(orig_dtype)


def pullback_metric_per_block(
    phi:          torch.Tensor,           # (..., n_gen) frame coordinates
    generators:   torch.Tensor,           # (n_gen, K, K)
    irrep_dims:   List[int],              # block sizes; sum == K

    *,
    series_tol:   float = 1e-12,
    series_order: int   = 40,
    max_k:        int   = 12,
) -> torch.Tensor:                        # (..., n_gen, n_gen) block-diagonal pullback metric
    r"""Block-diagonal pullback metric: per-irrep-block exp-map natural-gradient metric.

    For a block-diagonal algebra g = (+)_h gl(d_h) (block_glk: irrep_dims = [d_h]*H),
    phi is block-diagonal, so d exp_phi stays inside each block and the pullback metric
    G_ab(phi) = <d exp_phi(T_a), d exp_phi(T_b)>_F is itself block-diagonal: generators of
    distinct blocks have disjoint support, so their cross terms vanish. Each diagonal block
    is built by ``pullback_metric`` on that block's LOCAL d_h-dimensional representation
    (the d_h x d_h corner). This is the key feasibility win over the full ``pullback_metric``:
    the structure-constants tensor is O(n_gen_block^2 d_h^2) per block instead of
    O(n_gen^2 K^2), so a K = 20 block_glk (d_h = 10 <= max_k) is buildable where the full
    pullback (K = 20 > max_k) raises. Single global block (irrep_dims == [K]) reduces to
    ``pullback_metric``. Pure: takes a generator TENSOR, not a GaugeGroup.
    """
    if len(irrep_dims) == 1:
        return pullback_metric(phi, generators, series_tol=series_tol,
                               series_order=series_order, max_k=max_k)
    block_of = _generator_block_index(generators, irrep_dims)
    n_gen = generators.shape[0]
    batch = phi.shape[:-1]
    G_metric = torch.zeros(*batch, n_gen, n_gen, dtype=torch.float64, device=phi.device)
    start = 0
    for h, d in enumerate(irrep_dims):
        idx     = (block_of == h).nonzero(as_tuple=True)[0]
        sub     = generators[idx][:, start:start + d, start:start + d].contiguous()   # (n_h, d, d) local rep
        phi_blk = phi[..., idx]                                                        # (..., n_h)
        Gb      = pullback_metric(phi_blk, sub, series_tol=series_tol,
                                  series_order=series_order, max_k=max_k)              # (..., n_h, n_h)
        G_metric[..., idx.unsqueeze(-1), idx.unsqueeze(0)] = Gb
        start += d
    return G_metric


@register_precond("pullback_per_block")
def _precond_pullback_per_block(
    grad_phi:     torch.Tensor,           # (..., n_gen)
    phi:          torch.Tensor,           # (..., n_gen)
    generators:   torch.Tensor,           # (n_gen, K, K)

    *,
    series_tol:   float = 1e-12,
    eps:          float = 1e-6,
    series_order: int   = 40,
    irrep_dims:   Optional[List[int]] = None,
    **kwargs,
) -> torch.Tensor:
    r"""Per-block position-dependent natural gradient: solve G_block(phi) nat = grad_phi.

    The block-diagonal counterpart of ``pullback`` (the exact natural gradient for a
    block-diagonal gauge group like block_glk), feasible at K > max_k because each block's
    metric is built on its local d_h <= max_k representation."""
    if irrep_dims is None:
        raise ValueError("pullback_per_block requires irrep_dims")
    orig_dtype = grad_phi.dtype
    G_metric = pullback_metric_per_block(phi, generators, irrep_dims,
                                         series_tol=series_tol, series_order=series_order)
    eye = torch.eye(G_metric.shape[-1], dtype=G_metric.dtype, device=G_metric.device)
    sol = torch.linalg.solve(G_metric + eps * eye, grad_phi.double().unsqueeze(-1))
    return sol.squeeze(-1).to(orig_dtype)


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
