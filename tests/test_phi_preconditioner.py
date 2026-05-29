import math

import torch

from vfe3.geometry.generators import generate_glk, generate_glk_multihead, generate_son
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
    gram = torch.einsum("aij,bij->ab", G, G)
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
