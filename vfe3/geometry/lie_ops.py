r"""Lie-algebra retraction primitives for VFE_3.0 gauge frames (Gaussian-specialized).

The gauge frame phi lives in a Lie algebra g (a vector space) as coordinates in a
generator basis {G_a}: the algebra element is embed(phi) = sum_a phi^a G_a. The
group element U = exp(embed(phi)) lies in GL+(K) (det>0) or SO(N). This module
supplies: coordinate<->matrix maps, the Lie bracket, a composition registry
(euclidean step or BCH chart correction), the GL(K)/SO(N) retractions, and
determinant control. Pure: operates on a generator TENSOR, not a GaugeGroup.
"""

import math
from typing import Callable, Dict, List, Optional

import torch


def embed_phi(
    phi:        torch.Tensor,             # (..., n_gen) Lie-algebra coordinates
    generators: torch.Tensor,             # (n_gen, K, K) basis
) -> torch.Tensor:                        # (..., K, K) matrix sum_a phi^a G_a
    r"""Coordinates -> algebra element: embed(phi) = sum_a phi^a G_a."""
    return torch.einsum("...a,aij->...ij", phi, generators)


def gram_pinv(
    generators: torch.Tensor,             # (n_gen, K, K) basis

    *,
    rcond:      float = 1e-10,
) -> torch.Tensor:                        # (n_gen, n_gen) pseudo-inverse of Gram
    r"""Pseudo-inverse of the Frobenius Gram matrix Gram_ab = <G_a, G_b>_F.

    pinv (not inv) so overcomplete / rank-deficient spanning sets (e.g. the
    sl(K) set from generate_glk(include_identity=False)) are handled. The Gram
    and its pinv are formed in float64 because the overcomplete spanning set has
    a true nullspace whose float32 eigenvalue is O(1e-7) -- larger than rcond, so
    a float32 pinv would treat the null direction as invertible and inject noise.
    """
    gram = torch.einsum("aij,bij->ab", generators.double(), generators.double())
    return torch.linalg.pinv(gram, rcond=rcond).to(generators.dtype)


def extract_phi(
    matrix:     torch.Tensor,             # (..., K, K) element of span{G_a}
    generators: torch.Tensor,             # (n_gen, K, K) basis

    *,
    gram_pinv_: Optional[torch.Tensor] = None,   # cached gram_pinv(generators)
) -> torch.Tensor:                        # (..., n_gen) min-norm coordinates
    r"""Algebra element -> coordinates by least squares against the Gram matrix.

    Solves Gram c = g with g_b = <G_b, matrix>_F, c = Gram^+ g (min-norm solution
    when the basis is overcomplete). For an orthonormal basis Gram = I and
    c_a = <G_a, matrix>_F.
    """
    gp = gram_pinv(generators) if gram_pinv_ is None else gram_pinv_
    g = torch.einsum("aij,...ij->...a", generators, matrix)
    return torch.einsum("...a,ab->...b", g, gp)


def lie_bracket_matrix(
    A: torch.Tensor,                      # (..., K, K)
    B: torch.Tensor,                      # (..., K, K)
) -> torch.Tensor:                        # (..., K, K) [A,B] = AB - BA
    r"""Matrix commutator [A, B] = AB - BA (sign convention AB - BA)."""
    return A @ B - B @ A


def lie_bracket_coords(
    phi1:       torch.Tensor,             # (..., n_gen)
    phi2:       torch.Tensor,             # (..., n_gen)
    generators: torch.Tensor,             # (n_gen, K, K)

    *,
    gram_pinv_: Optional[torch.Tensor] = None,
) -> torch.Tensor:                        # (..., n_gen) coords of [embed phi1, embed phi2]
    r"""Bracket in coordinates: extract([embed(phi1), embed(phi2)])."""
    A = embed_phi(phi1, generators)
    B = embed_phi(phi2, generators)
    return extract_phi(lie_bracket_matrix(A, B), generators, gram_pinv_=gram_pinv_)


_COMPOSE: Dict[str, Callable[..., torch.Tensor]] = {}


def register_compose(name: str) -> Callable:
    """Decorator registering a composition rule phi1,phi2 -> composed coords."""
    def _wrap(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        _COMPOSE[name] = fn
        return fn
    return _wrap


def get_compose(name: str) -> Callable[..., torch.Tensor]:
    """Return the registered composition rule (KeyError if absent)."""
    if name not in _COMPOSE:
        raise KeyError(f"no composition rule {name!r}; available: {sorted(_COMPOSE)}")
    return _COMPOSE[name]


@register_compose("euclidean")
def compose_euclidean(
    phi1:       torch.Tensor,             # (..., n_gen)
    phi2:       torch.Tensor,             # (..., n_gen)
    generators: torch.Tensor,             # (n_gen, K, K) (unused; kept for a uniform seam)

    *,
    order:      int = 0,
    gram_pinv_: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Plain Lie-algebra step phi1 + phi2 (exact iff [phi1, phi2] = 0).

    The manuscript working/default update: g is a vector space, so the tangent
    step is the sum of coordinates (GL(K)_supplementary.tex ll. 550-557).
    """
    return phi1 + phi2


@register_compose("bch")
def compose_bch(
    phi1:       torch.Tensor,             # (..., n_gen)
    phi2:       torch.Tensor,             # (..., n_gen)
    generators: torch.Tensor,             # (n_gen, K, K)

    *,
    order:      int = 4,
    gram_pinv_: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""BCH chart correction: coords of log(exp(embed phi1) exp(embed phi2)).

    Symmetric Dynkin series (matrix space, extracted once). Terms by `order`:
      order>=1: + 1/2 [X,Y]
      order>=2: + 1/12 ([X,[X,Y]] - [Y,[X,Y]])
      order>=3: - 1/24 [Y,[X,[X,Y]]]
      order>=4: - 1/720 ([Y,[Y,[Y,[Y,X]]]] + [X,[X,[X,[X,Y]]]])
                + 1/360 ([X,[Y,[Y,[Y,X]]]] + [Y,[X,[X,[X,Y]]]])
                + 1/120 ([Y,[X,[Y,[X,Y]]]] + [X,[Y,[X,[Y,X]]]])
    Truncation error is O(||X||^{order+2} + ||Y||^{order+2}).
    """
    X = embed_phi(phi1, generators)
    Y = embed_phi(phi2, generators)
    Z = X + Y
    br = lie_bracket_matrix
    if order >= 1:
        XY = br(X, Y)
        Z = Z + 0.5 * XY
    if order >= 2:
        Z = Z + (1.0 / 12.0) * (br(X, XY) - br(Y, XY))
    if order >= 3:
        Z = Z - (1.0 / 24.0) * br(Y, br(X, XY))
    if order >= 4:
        YX  = br(Y, X)
        YYX = br(Y, YX); YYYX = br(Y, YYX)
        XXY = br(X, XY); XXXY = br(X, XXY)
        Z = Z - (1.0 / 720.0) * (br(Y, YYYX) + br(X, XXXY))
        Z = Z + (1.0 / 360.0) * (br(X, YYYX) + br(Y, XXXY))
        Z = Z + (1.0 / 120.0) * (br(Y, br(X, br(Y, XY))) + br(X, br(Y, br(X, YX))))
    return extract_phi(Z, generators, gram_pinv_=gram_pinv_)


def compose_phi(
    phi1:       torch.Tensor,             # (..., n_gen)
    phi2:       torch.Tensor,             # (..., n_gen)
    generators: torch.Tensor,             # (n_gen, K, K)

    *,
    order:      int = 4,
    mode:       str = "euclidean",
    gram_pinv_: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Dispatch to the registered composition rule `mode`."""
    return get_compose(mode)(phi1, phi2, generators, order=order, gram_pinv_=gram_pinv_)


def _retract_core(
    phi:          torch.Tensor,           # (..., n_gen) current frame
    delta_phi:    torch.Tensor,           # (..., n_gen) tangent step direction

    *,
    step_size:    float = 1.0,
    trust_region: float = 0.1,
    max_norm:     float = 5.0,
    eps:          float = 1e-6,
    order:        int   = 4,
    mode:         str   = "euclidean",
    generators:   Optional[torch.Tensor]  = None,
    gram_pinv_:   Optional[torch.Tensor]  = None,
) -> torch.Tensor:
    r"""Shared retraction: scale -> trust-region clamp -> compose -> max-norm clamp.

      update   = clamp_||.|| ( step_size * delta_phi , trust_region )
      phi_new  = compose(phi, update; mode, order)
      phi_new <- clamp_||.|| ( phi_new , max_norm )
    Trust region and max norm are applied to the coordinate-vector norm.
    """
    update = step_size * delta_phi
    if trust_region is not None and trust_region > 0:
        u_norm = update.norm(dim=-1, keepdim=True)
        update = update * (trust_region / (u_norm + eps)).clamp(max=1.0)
    phi_new = compose_phi(phi, update, generators, order=order, mode=mode, gram_pinv_=gram_pinv_)
    if max_norm is not None and max_norm > 0:
        n_norm = phi_new.norm(dim=-1, keepdim=True)
        phi_new = torch.where(n_norm > max_norm, phi_new * (max_norm / (n_norm + eps)), phi_new)
    return phi_new


def retract_glk(
    phi:          torch.Tensor,           # (..., n_gen) current GL(K) frame
    delta_phi:    torch.Tensor,           # (..., n_gen) tangent step

    generators:   torch.Tensor,           # (n_gen, K, K)

    *,
    step_size:    float = 1.0,
    trust_region: float = 0.1,            # tighter than SO(N): GL(K) is non-compact
    max_norm:     float = 5.0,            # bounds singular values to ~[e^-5, e^5]
    eps:          float = 1e-6,
    order:        int   = 4,
    mode:         str   = "euclidean",
    gram_pinv_:   Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""GL(K) retraction (no det control here; the dispatcher applies it)."""
    return _retract_core(
        phi, delta_phi, step_size=step_size, trust_region=trust_region,
        max_norm=max_norm, eps=eps, order=order, mode=mode,
        generators=generators, gram_pinv_=gram_pinv_,
    )


def retract_son(
    phi:          torch.Tensor,           # (..., n_gen) current SO(N) frame
    delta_phi:    torch.Tensor,           # (..., n_gen) tangent step

    generators:   torch.Tensor,           # (n_gen, K, K) skew so(N) basis

    *,
    step_size:    float = 1.0,
    trust_region: float = 0.3,            # compact group
    max_norm:     float = math.pi,        # bounds principal angles
    eps:          float = 1e-6,
    order:        int   = 4,
    mode:         str   = "euclidean",
    gram_pinv_:   Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""SO(N) retraction. det(exp) = 1 automatic (skew generators)."""
    return _retract_core(
        phi, delta_phi, step_size=step_size, trust_region=trust_region,
        max_norm=max_norm, eps=eps, order=order, mode=mode,
        generators=generators, gram_pinv_=gram_pinv_,
    )
