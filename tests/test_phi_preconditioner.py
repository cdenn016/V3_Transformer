import gc
import math
import weakref

import torch

import vfe3.geometry.phi_preconditioner as phi_preconditioner
from vfe3.geometry.generators import generate_glk, generate_glk_multihead, generate_son
from vfe3.geometry.groups import get_group
from vfe3.geometry.phi_preconditioner import precondition_phi_gradient


def test_none_is_identity():
    G = generate_glk(3)
    grad = torch.randn(4, 9)
    out = precondition_phi_gradient(grad, torch.zeros(4, 9), G, mode="none")
    assert torch.allclose(out, grad, atol=1e-7)


def test_clip_scales_large_gradient_to_c():
    G = generate_glk(3)
    grad = 100.0 * torch.ones(2, 9)                       # norm >> c
    out = precondition_phi_gradient(grad, torch.zeros(2, 9), G, mode="clip", clip_c=10.0)
    assert torch.allclose(out.norm(dim=-1), torch.full((2,), 10.0), atol=1e-3)


def test_killing_per_block_caches_parent_without_strong_retention():
    cache = phi_preconditioner._KILLING_INV_CACHE
    cache.clear()
    group = get_group("block_glk")(4, 2)
    generators = group.generators
    first = phi_preconditioner.build_killing_preconditioner_per_block(
        generators, group.irrep_dims,
    )
    second = phi_preconditioner.build_killing_preconditioner_per_block(
        generators, group.irrep_dims,
    )
    assert second is first
    ref = weakref.ref(generators)
    del group, generators, first, second
    gc.collect()
    assert ref() is None
    assert not cache


def test_killing_cache_does_not_retain_requires_grad_basis():
    cache = phi_preconditioner._KILLING_INV_CACHE
    cache.clear()
    generators = generate_son(3).clone().requires_grad_()
    inverse = phi_preconditioner.build_killing_preconditioner(generators)
    assert inverse.requires_grad is False
    ref = weakref.ref(generators)
    del generators, inverse
    gc.collect()
    assert ref() is None
    assert not cache


def test_killing_cache_recomputes_after_in_place_basis_mutation():
    cache = phi_preconditioner._KILLING_INV_CACHE
    cache.clear()
    generators = generate_son(3).clone()
    first = phi_preconditioner.build_killing_preconditioner(generators)
    generators.mul_(2.0)
    second = phi_preconditioner.build_killing_preconditioner(generators)
    assert second is not first
    assert not torch.allclose(second, first)


def test_killing_cache_is_lru_bounded():
    cache = phi_preconditioner._KILLING_INV_CACHE
    cache.clear()
    generators = [generate_son(3).clone() for _ in range(33)]
    inverses = [
        phi_preconditioner.build_killing_preconditioner(basis)
        for basis in generators[:32]
    ]
    assert phi_preconditioner.build_killing_preconditioner(generators[0]) is inverses[0]
    phi_preconditioner.build_killing_preconditioner(generators[32])
    assert len(cache) == 32
    assert phi_preconditioner.build_killing_preconditioner(generators[0]) is inverses[0]
    assert phi_preconditioner.build_killing_preconditioner(generators[1]) is not inverses[1]


def test_pullback_series_warns_on_non_convergence():
    # m19: pullback_metric's Psi series must WARN (not silently) when it exhausts series_order without
    # meeting the tolerance.
    import warnings as _w
    import vfe3.geometry.phi_preconditioner as pp
    G = generate_son(3)
    phi = torch.full((G.shape[0],), 3.0)            # large ||phi||: the order-2 term stays above tol
    pp._PULLBACK_SERIES_WARNED = False
    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter("always")
        pp.pullback_metric(phi, G, series_tol=1e-12, series_order=3)
    assert any("did not converge" in str(x.message) for x in rec)
    pp._PULLBACK_SERIES_WARNED = False              # converged case (zero phi) must NOT warn
    with _w.catch_warnings(record=True) as rec2:
        _w.simplefilter("always")
        pp.pullback_metric(torch.zeros(G.shape[0]), G, series_tol=1e-12, series_order=40)
    assert not any("did not converge" in str(x.message) for x in rec2)


def test_clip_leaves_small_gradient_unchanged():
    G = generate_glk(3)
    grad = 0.01 * torch.ones(2, 9)                        # norm << c
    out = precondition_phi_gradient(grad, torch.zeros(2, 9), G, mode="clip", clip_c=10.0)
    assert torch.allclose(out, grad, atol=1e-7)


from vfe3.geometry.phi_preconditioner import build_killing_preconditioner, killing_metric


def test_killing_metric_gl2_exact():
    # Cartan-involution form 2K*gram - 2*tr(x)tr(.) on the gl(2) elementary basis
    # (E00,E01,E10,E11), K=2. Bare Killing tr(T_a T_b) would give a DIFFERENT,
    # indefinite matrix; this literal discriminates them.
    G = generate_glk(2)
    M = killing_metric(G)
    expected = torch.tensor([[2., 0., 0., -2.],
                             [0., 4., 0.,  0.],
                             [0., 0., 4.,  0.],
                             [-2., 0., 0., 2.]])
    assert torch.allclose(M, expected, atol=1e-5)
    evals = torch.linalg.eigvalsh(M)
    assert torch.allclose(evals.sort().values, torch.tensor([0., 4., 4., 4.]), atol=1e-4)


def test_killing_metric_so3_is_positive_definite():
    # so(3): skew generators have tr=0 so g~ = 2K*gram. gram = 2*I (||L_ij||_F^2=2),
    # K=3 -> g~ = 12*I, POSITIVE-definite. (Bare Killing is negative-definite on skew.)
    G = generate_son(3)
    M = killing_metric(G)
    assert torch.allclose(M, 12.0 * torch.eye(3), atol=1e-4)
    assert (torch.linalg.eigvalsh(M) > 0).all()


def test_killing_preconditioner_exact_inverse_on_slk():
    # Center-reg lifts ONLY the numerical nullspace (the center/identity direction);
    # on sl(K) the regularized inverse is the TRUE Killing inverse: g~ @ (Minv @ v) = v
    # for v perpendicular to the center. A ridge center_reg*I would fail this.
    G = generate_glk(2)
    M = killing_metric(G)
    Minv = build_killing_preconditioner(G, center_reg=4.0)
    v = torch.tensor([0., 1.3, -0.7, 0.])                 # in sl(2): E01,E10 dirs (trace-free)
    assert torch.allclose(M @ (Minv @ v), v, atol=1e-5)
    # full regularized metric is PD (gl(2): eigenvalues {0,4,4,4} -> {4,4,4,4})
    reg = torch.linalg.inv(Minv)
    assert (torch.linalg.eigvalsh(reg) > 1e-6).all()


def test_killing_mode_applies_inverse_metric():
    G = generate_glk(2)
    grad = torch.tensor([[0., 2.0, -1.0, 0.]])
    Minv = build_killing_preconditioner(G, center_reg=4.0)
    out = precondition_phi_gradient(grad, torch.zeros(1, 4), G, mode="killing", center_reg=4.0)
    assert torch.allclose(out, grad @ Minv, atol=1e-6)


from vfe3.geometry.phi_preconditioner import build_killing_preconditioner_per_block


def test_per_block_is_block_diagonal_and_matches_local_killing():
    # block_glk = gl(2) (+) gl(2): generators grouped by head (4 per head). The
    # per-block metric couples only same-block generators and uses the local
    # block dimension d_h=2, so the (8,8) inverse is block-diagonal in 4+4.
    G = generate_glk_multihead(4, 2)                      # 8 generators, irrep [2,2]
    irrep = [2, 2]
    Minv = build_killing_preconditioner_per_block(G, irrep, center_reg=4.0)
    # off-diagonal cross-block coupling is zero
    assert torch.allclose(Minv[:4, 4:], torch.zeros(4, 4), atol=1e-6)
    assert torch.allclose(Minv[4:, :4], torch.zeros(4, 4), atol=1e-6)
    # each diagonal block equals the single-head gl(2) Killing inverse
    head0 = generate_glk(2)
    Minv_head = build_killing_preconditioner(head0, center_reg=4.0)
    assert torch.allclose(Minv[:4, :4], Minv_head, atol=1e-5)


def test_killing_per_block_mode():
    G = generate_glk_multihead(4, 2)
    irrep = [2, 2]
    grad = torch.randn(3, 8)
    Minv = build_killing_preconditioner_per_block(G, irrep, center_reg=4.0)
    out = precondition_phi_gradient(grad, torch.zeros(3, 8), G, mode="killing_per_block",
                                    irrep_dims=irrep, center_reg=4.0)
    assert torch.allclose(out, grad @ Minv, atol=1e-6)


from vfe3.geometry.lie_ops import embed_phi
from vfe3.geometry.phi_preconditioner import pullback_metric


def _fd_dexp_metric(phi_vec, G, eps=1e-4):
    # Independent oracle: G_ab = <d exp_phi(e_a), d exp_phi(e_b)>_F via central
    # finite differences of matrix_exp. d exp_phi(e_a) = d/dt exp(embed(phi + t e_a)).
    n = G.shape[0]
    J = []
    for a in range(n):
        ea = torch.zeros(n, dtype=torch.float64); ea[a] = 1.0
        plus  = torch.linalg.matrix_exp(embed_phi((phi_vec + eps * ea), G))
        minus = torch.linalg.matrix_exp(embed_phi((phi_vec - eps * ea), G))
        J.append((plus - minus) / (2 * eps))
    J = torch.stack(J, 0)                                 # (n, K, K)
    return torch.einsum("aij,bij->ab", J, J)              # tr(J_a^T J_b)


def test_pullback_at_zero_is_frobenius_gram():
    # Psi(0) = I, exp(0) = I -> G_ab = tr(G_a^T G_b) = Frobenius Gram.
    G = generate_glk(2)
    Gmetric = pullback_metric(torch.zeros(4), G)
    gram = torch.einsum("aij,bij->ab", G.double(), G.double())
    assert Gmetric.dtype == torch.float64
    assert torch.allclose(Gmetric, gram, atol=1e-5)


def test_pullback_matches_finite_difference_of_exp():
    # The genuine independent check: closed Psi-series vs FD-of-exp. Validates the
    # operator ordering and every series coefficient.
    G = generate_son(3).double()                          # K=3, compact -> well-behaved
    phi = torch.tensor([0.4, -0.3, 0.5], dtype=torch.float64)
    Gclosed = pullback_metric(phi, G, series_order=10)
    Gfd = _fd_dexp_metric(phi, G)
    assert torch.allclose(Gclosed, Gfd, atol=1e-4)
    # symmetric PD
    assert torch.allclose(Gclosed, Gclosed.transpose(-1, -2), atol=1e-6)
    assert (torch.linalg.eigvalsh(Gclosed) > 0).all()


def test_pullback_matches_fd_noncompact_at_default_order():
    # Pins the SHIPPED DEFAULT (no explicit series_order): a non-compact symmetric
    # gl(2) phi at ||phi||~2 -- well inside retract_glk's max_norm=5. ad_phi eigenvalues
    # scale with ||phi||, so a fixed low-order Psi truncation is inaccurate here; the
    # adaptive default must still match the FD-of-exp oracle in the regime the pullback
    # metric exists for. Closed-form (default knobs) must agree with FD to 1e-4.
    G = generate_glk(2).double()
    base = torch.tensor([0.6, 0.5, 0.5, -0.6], dtype=torch.float64)   # E00,E01,E10,E11
    phi = base / base.norm() * 2.0                        # ||phi|| = 2.0 (non-compact)
    Gclosed = pullback_metric(phi, G)                     # SHIPPED DEFAULT series knobs
    Gfd = _fd_dexp_metric(phi, G)
    assert torch.allclose(Gclosed, Gfd, atol=1e-4)
    assert torch.allclose(Gclosed, Gclosed.transpose(-1, -2), atol=1e-6)
    assert (torch.linalg.eigvalsh(Gclosed) > 0).all()


def test_pullback_default_order_no_factorial_overflow():
    # The Psi-series coefficient is a float: an int 1/(k+1)! divisor overflows tensor
    # division past order ~20. The shipped default cap (40) must run to completion
    # without raising on a non-compact phi that drives the series deep.
    G = generate_glk(2).double()
    base = torch.tensor([0.6, 0.5, 0.5, -0.6], dtype=torch.float64)
    phi = base / base.norm() * 5.0                        # ||phi|| = 5 (drives series deep)
    Gclosed = pullback_metric(phi, G)                     # default series_order=40
    assert torch.isfinite(Gclosed).all()


def test_pullback_k_guard():
    import pytest
    G = generate_glk(13)                                  # K=13 > max_k
    with pytest.raises((ValueError, RuntimeError)):
        pullback_metric(torch.zeros(169), G, max_k=12)


from vfe3.geometry.phi_preconditioner import pullback_metric_per_block


def test_pullback_per_block_single_block_matches_full():
    # irrep_dims == [K]: the per-block metric reduces to the full pullback exactly.
    G = generate_glk(3).double()
    phi = torch.randn(9, dtype=torch.float64) * 0.3
    a = pullback_metric_per_block(phi, G, [3])
    b = pullback_metric(phi, G)
    assert torch.allclose(a, b, atol=1e-8)


def test_pullback_per_block_is_block_diagonal_and_fd_per_block():
    # block_glk = gl(2) (+) gl(2): the pullback metric must be block-diagonal (distinct
    # blocks have disjoint support, so cross d-exp terms vanish), and EACH diagonal block
    # must match the FD-of-exp oracle on that block's LOCAL gl(2) representation.
    G = generate_glk_multihead(4, 2).double()             # K=4, d_h=2, 8 gens, irrep [2,2]
    irrep = [2, 2]
    torch.manual_seed(0)
    phi = torch.randn(8, dtype=torch.float64) * 0.5
    Gblk = pullback_metric_per_block(phi, G, irrep)        # (8, 8)
    assert torch.allclose(Gblk[:4, 4:], torch.zeros(4, 4, dtype=torch.float64), atol=1e-6)
    assert torch.allclose(Gblk[4:, :4], torch.zeros(4, 4, dtype=torch.float64), atol=1e-6)
    local = generate_glk(2).double()                      # the d_h=2 local rep both blocks use
    assert torch.allclose(Gblk[:4, :4], _fd_dexp_metric(phi[:4], local), atol=1e-4)
    assert torch.allclose(Gblk[4:, 4:], _fd_dexp_metric(phi[4:], local), atol=1e-4)


def test_pullback_per_block_feasible_at_k20_where_full_pullback_dies():
    # The feasibility win: full pullback raises at K=20 (>max_k), but per-block builds on
    # the d_h=10 (<=max_k) local rep, so the shipped block_glk K=20 is buildable.
    import pytest
    G20 = generate_glk_multihead(20, 2).double()          # K=20, d_h=10, 200 gens
    with pytest.raises((ValueError, RuntimeError)):
        pullback_metric(torch.zeros(400, dtype=torch.float64), G20)   # full: K=20 > max_k
    phi = torch.randn(200, dtype=torch.float64) * 0.1
    Gblk = pullback_metric_per_block(phi, G20, [10, 10])   # per-block: must NOT raise
    assert Gblk.shape == (200, 200)
    assert torch.isfinite(Gblk).all()
    assert torch.allclose(Gblk[:100, 100:], torch.zeros(100, 100, dtype=torch.float64), atol=1e-6)
    assert (torch.linalg.eigvalsh(Gblk[:100, :100]) > 0).all()


def test_pullback_per_block_mode_solves_metric_and_reduces_to_identity_at_zero():
    G = generate_glk_multihead(4, 2).double()
    irrep = [2, 2]
    torch.manual_seed(1)
    phi = torch.randn(3, 8, dtype=torch.float64) * 0.3
    grad = torch.randn(3, 8, dtype=torch.float64)
    out = precondition_phi_gradient(grad, phi, G, mode="pullback_per_block", irrep_dims=irrep)
    Gm = pullback_metric_per_block(phi, G, irrep)
    eye = torch.eye(8, dtype=torch.float64)
    expect = torch.linalg.solve(Gm + 1e-6 * eye, grad.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(out, expect, atol=1e-6)
    # at phi=0 the metric is the Frobenius gram = I (orthonormal E_ij), so nat-grad == grad.
    out0 = precondition_phi_gradient(grad, torch.zeros(3, 8, dtype=torch.float64), G,
                                     mode="pullback_per_block", irrep_dims=irrep)
    assert torch.allclose(out0, grad, atol=1e-4)
