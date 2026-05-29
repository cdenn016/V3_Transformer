r"""Lie-algebra retraction primitives for VFE_3.0 gauge frames (Gaussian-specialized).

The gauge frame phi lives in a Lie algebra g (a vector space) as coordinates in a
generator basis {G_a}: the algebra element is embed(phi) = sum_a phi^a G_a. The
group element U = exp(embed(phi)) lies in GL+(K) (det>0) or SO(N). This module
supplies: coordinate<->matrix maps, the Lie bracket, a composition registry
(euclidean step or BCH chart correction), the GL(K)/SO(N) retractions, and
determinant control. Pure: operates on a generator TENSOR, not a GaugeGroup.
"""

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
