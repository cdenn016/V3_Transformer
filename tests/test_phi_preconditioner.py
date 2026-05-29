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
