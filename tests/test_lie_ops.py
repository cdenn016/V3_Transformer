import math

import torch

from vfe3.geometry.generators import generate_glk, generate_son
from vfe3.geometry.lie_ops import (
    embed_phi,
    extract_phi,
    lie_bracket_coords,
)


def test_embed_extract_roundtrip_independent_basis():
    # gl(2) elementary basis is orthonormal under Frobenius -> extract(embed(c)) == c.
    G = generate_glk(2)                                   # (4, 2, 2)
    c = torch.randn(3, 4)
    out = extract_phi(embed_phi(c, G), G)
    assert torch.allclose(out, c, atol=1e-6)


def test_embed_extract_projection_overcomplete():
    # sl(K) spanning set (include_identity=False) is OVERCOMPLETE (rank K^2-1):
    # extract(embed(c)) need NOT equal c, but embed o extract o embed == embed.
    G = generate_glk(3, include_identity=False)           # (<=9, 3, 3), rank 8
    c = torch.randn(2, G.shape[0])
    M = embed_phi(c, G)
    M2 = embed_phi(extract_phi(M, G), G)
    assert torch.allclose(M2, M, atol=1e-5)


def test_bracket_so3_structure_constants():
    # generate_son(3) basis: G0=E01-E10, G1=E02-E20, G2=E12-E21.
    # Hand-derived: [G0,G1]=-G2, [G0,G2]=+G1, [G1,G2]=-G0.
    G = generate_son(3)                                   # (3, 3, 3)
    e = torch.eye(3)
    c01 = lie_bracket_coords(e[0], e[1], G)
    c02 = lie_bracket_coords(e[0], e[2], G)
    c12 = lie_bracket_coords(e[1], e[2], G)
    assert torch.allclose(c01, torch.tensor([0.0, 0.0, -1.0]), atol=1e-6)
    assert torch.allclose(c02, torch.tensor([0.0, 1.0,  0.0]), atol=1e-6)
    assert torch.allclose(c12, torch.tensor([-1.0, 0.0, 0.0]), atol=1e-6)


from vfe3.geometry.lie_ops import compose_phi, get_compose


def test_compose_euclidean_is_sum():
    G = generate_glk(2)
    a, b = torch.randn(4), torch.randn(4)
    assert torch.allclose(compose_phi(a, b, G, mode="euclidean"), a + b, atol=1e-6)


def test_bch_commuting_is_exact():
    # Two diagonal gl(3) elements commute -> BCH == euclidean sum exactly.
    G = generate_glk(3)
    a = torch.zeros(9); a[0] = 0.7          # E00 direction
    b = torch.zeros(9); b[8] = -0.4         # E22 direction
    z = compose_phi(a, b, G, mode="bch", order=4)
    assert torch.allclose(z, a + b, atol=1e-6)


def _bch_residual(order: int, eps: float) -> float:
    # || exp(embed(bch(eps X, eps Y))) - exp(embed(eps X)) exp(embed(eps Y)) ||_F.
    # float64 so the slope is not floored by float32 round-off.
    torch.manual_seed(0)
    G = generate_son(3).double()
    X = eps * torch.tensor([0.9, -0.3, 0.5], dtype=torch.float64)
    Y = eps * torch.tensor([-0.2, 0.7, 0.4], dtype=torch.float64)
    z = compose_phi(X, Y, G, mode="bch", order=order)
    lhs = torch.linalg.matrix_exp(embed_phi(z, G))
    rhs = torch.linalg.matrix_exp(embed_phi(X, G)) @ torch.linalg.matrix_exp(embed_phi(Y, G))
    return float(torch.linalg.norm(lhs - rhs))


def test_bch_residual_rate_order_matches_slope():
    # Truncation error of order-k BCH is O(eps^(k+2)); the log-log slope of the
    # residual vs eps must be ~ k+2 (this is the genuine, independent check that
    # catches a wrong Dynkin coefficient -- a wrong term degrades the slope).
    for order, expected in [(1, 3.0), (2, 4.0), (4, 6.0)]:
        eps = [0.2, 0.1, 0.05]
        r = [_bch_residual(order, e) for e in eps]
        slope = (math.log(r[0]) - math.log(r[-1])) / (math.log(eps[0]) - math.log(eps[-1]))
        assert abs(slope - expected) < 0.8, f"order={order}: slope={slope:.2f} != {expected}"
