r"""Lie-algebra retraction primitives for VFE_3.0 gauge frames (Gaussian-specialized).

The gauge frame phi lives in a Lie algebra g (a vector space) as coordinates in a
generator basis {G_a}: the algebra element is embed(phi) = sum_a phi^a G_a. The
group element U = exp(embed(phi)) lies in GL+(K) (det>0) or SO(N). This module
supplies: coordinate<->matrix maps, the Lie bracket, a composition registry
(euclidean step or BCH chart correction), the GL(K)/SO(N) retractions, and
determinant control. Pure: operates on a generator TENSOR, not a GaugeGroup.
"""

import math
import warnings
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


# Bracket-closure of a generator basis is a property of the FIXED generators, not of any
# belief phi, so it is measured ONCE per basis and cached -- never per compose/precondition
# call. Keyed by object id; the basis tensor is retained in the cache so its id cannot be
# reused (a stale-id false positive). The residual is computed under no_grad off the autograd
# graph; the old per-call float(Z) on a grad-carrying Z both host-synced the hot E-step and
# raised the "requires_grad to scalar" warning.
_BRACKET_CLOSURE_RES:    Dict[int, list] = {}      # id(generators) -> [generators_ref, max_rel_residual]
_BRACKET_CLOSURE_WARNED: set             = set()    # (id(generators), where) already-warned call sites


def warn_if_basis_not_closed(
    generators:  torch.Tensor,            # (n_gen, K, K) fixed gauge generator basis

    *,
    where:       str,                     # call-site label (also the per-site warn-once key)
    closure_tol: float                  = 1e-4,
    eps:         float                  = 1e-12,
    gram_pinv_:  Optional[torch.Tensor] = None,
) -> None:
    r"""Warn once (per call site) if ``generators`` is not closed under the Lie bracket.

    Measures the max relative out-of-span residual
    :math:`\max_{a,b} \lVert [G_a,G_b] - \mathrm{embed}(\mathrm{extract}([G_a,G_b]))\rVert_F /
    (\lVert [G_a,G_b]\rVert_F + \epsilon)`. On a closed (default direct-sum) basis this is ~0 and
    nothing is warned; on a non-closed 3+-head ``cross_couplings`` chain it is O(1) and the span
    projection in BCH composition / structure-constants silently truncates the out-of-span part.
    The result depends only on the generators, so it is cached and the hot path pays a dict lookup.
    """
    key = id(generators)
    entry = _BRACKET_CLOSURE_RES.get(key)
    if entry is None:
        with torch.no_grad():
            G    = generators                                                   # (n_gen, K, K)
            brak = torch.einsum("aij,bjk->abik", G, G) - torch.einsum("bij,ajk->abik", G, G)  # [G_a,G_b]
            proj = embed_phi(extract_phi(brak, G, gram_pinv_=gram_pinv_), G)     # span projection of each bracket
            res  = (brak - proj).norm(dim=(-2, -1))                             # (n_gen, n_gen)
            den  = brak.norm(dim=(-2, -1)) + eps                                # (n_gen, n_gen)
            max_res = float((res / den).max())
        entry = _BRACKET_CLOSURE_RES[key] = [generators, max_res]               # retain the basis -> stable id
    max_res = entry[1]
    wkey = (key, where)
    if max_res > closure_tol and wkey not in _BRACKET_CLOSURE_WARNED:
        _BRACKET_CLOSURE_WARNED.add(wkey)
        warnings.warn(
            f"{where}: gauge generator basis is not closed under the Lie bracket (max relative "
            f"out-of-span residual {max_res:.3e} > {closure_tol:.1e}); BCH composition / structure "
            f"constants truncate the out-of-span [G_a,G_b] terms. Build the group with "
            f"close_basis=True for cross-coupled (3+-head chain) bases.",
            UserWarning,
            stacklevel=3,
        )


@register_compose("bch")
def compose_bch(
    phi1:        torch.Tensor,            # (..., n_gen)
    phi2:        torch.Tensor,            # (..., n_gen)
    generators:  torch.Tensor,            # (n_gen, K, K)

    *,
    closure_tol: float                  = 1e-4,
    eps:         float                  = 1e-12,
    order:       int                    = 4,
    gram_pinv_:  Optional[torch.Tensor] = None,
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

    On a basis NOT closed under the Lie bracket the Dynkin commutator terms push
    :math:`Z` out of :math:`\mathrm{span}\{G_a\}`, and the final
    :math:`\mathrm{extract\_phi}(Z)` least-squares projection silently discards that
    out-of-span component. A diagnostic guard measures the max relative residual
    :math:`\max \lVert Z - \mathrm{embed}(\mathrm{extract}(Z))\rVert_F /
    (\lVert Z\rVert_F + \epsilon)` and warns once if it exceeds ``closure_tol``; on a
    closed (default direct-sum) basis the residual is ~0 so the guard is silent and the
    returned phi is unchanged.
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
    phi = extract_phi(Z, generators, gram_pinv_=gram_pinv_)

    # Diagnostic (cached, phi-independent): warn once if the generator basis is not bracket-closed,
    # in which case the Dynkin commutators leave span{G_a} and the extract_phi projection above drops
    # that part. Closure depends only on the fixed generators, so this never host-syncs the hot E-step
    # nor touches Z's autograd graph (the old per-call float(Z) did both -> the requires_grad warning).
    warn_if_basis_not_closed(generators, where="compose_bch",
                             closure_tol=closure_tol, eps=eps, gram_pinv_=gram_pinv_)
    return phi


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


def _block_trace_vectors(
    generators: torch.Tensor,             # (n_gen, K, K)
    irrep_dims: List[int],                # block sizes; sum == K

    *,
    eps:        float = 1e-12,
) -> torch.Tensor:                        # (n_blocks, n_gen) V[h,a] = tr(G_a|block h)
    r"""Per-block trace functionals V[h,a] = tr(G_a restricted to block h)."""
    rows, start = [], 0
    for d in irrep_dims:
        end = start + d
        rows.append(generators[:, start:end, start:end].diagonal(dim1=-2, dim2=-1).sum(-1))
        start = end
    return torch.stack(rows, dim=0)                       # (n_blocks, n_gen)


def project_phi_to_slk(
    phi:        torch.Tensor,             # (..., n_gen)
    generators: torch.Tensor,             # (n_gen, K, K)
    irrep_dims: List[int],                # block sizes; sum == K
) -> torch.Tensor:                        # (..., n_gen) per-block trace-free
    r"""Hard projection to sl(K) per block: remove the trace component so

        det(Omega_h) = exp(tr(embed(phi)|block h)) = 1.

    Orthogonal projection of phi off the span of the per-block trace functionals V_h:
    ``phi <- phi - (phi V^T) (V V^T)^+ V``. The JOINT Gram solve ``(V V^T)^+`` is required:
    for the tied gauge (``tied_block_glk``) the V_h coincide (generators kron(I_n, E_ij)), so the
    per-block-independent form ``s / ||V_h||^2`` would over-subtract by a factor of n_heads. For an
    untied gauge the V_h have disjoint support, ``V V^T`` is diagonal, and the pseudo-inverse reduces
    to ``1/||V_h||^2`` (so the untied result is unchanged); a fully traceless basis (so_k) gives
    ``V = 0`` and the projection is a no-op.
    """
    V = _block_trace_vectors(generators, irrep_dims)      # (H, n_gen)
    gram_pinv = torch.linalg.pinv(V @ V.transpose(-1, -2))   # (H, H); diag -> 1/||V_h||^2
    s = phi @ V.transpose(-1, -2)                         # (..., H)
    coeffs = s @ gram_pinv                                # (..., H) joint solve, not per-block
    return phi - torch.einsum("...h,hg->...g", coeffs, V)


def clamp_phi_trace(
    phi:        torch.Tensor,             # (..., n_gen)
    generators: torch.Tensor,             # (n_gen, K, K)
    irrep_dims: List[int],                # block sizes; sum == K

    *,
    trace_max:  float = 5.0,              # soft cap T on |tr(embed(phi)|block h)|
) -> torch.Tensor:                        # (..., n_gen) with |s_h| <= T
    r"""Soft per-block trace clamp: rescale only the trace component so |s_h| <= T,

    bounding log|det(Omega_h)|. Off-trace (sl(K)) directions are untouched. Realizes the clamped
    trace via the JOINT Gram solve ``delta = (s_clamped - s) (V V^T)^+`` so the tied gauge (coinciding
    V_h) is corrected once, not n_heads times; reduces to ``(s_clamped - s)/||V_h||^2`` for an
    orthogonal (untied) basis.
    """
    V = _block_trace_vectors(generators, irrep_dims)      # (H, n_gen)
    gram_pinv = torch.linalg.pinv(V @ V.transpose(-1, -2))   # (H, H); diag -> 1/||V_h||^2
    s = phi @ V.transpose(-1, -2)                         # (..., H)
    s_clamped = s.clamp(min=-trace_max, max=trace_max)
    delta = (s_clamped - s) @ gram_pinv                   # (..., H) joint solve, not per-block
    return phi + torch.einsum("...h,hg->...g", delta, V)
