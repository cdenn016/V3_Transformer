import math

import torch

from vfe3.geometry.generators import generate_glk, generate_son
from vfe3.geometry.lie_ops import embed_phi, retract_glk, retract_son


def test_retract_glk_trust_region_and_max_norm():
    G = generate_glk(3)                                   # (9,3,3)
    phi = 0.1 * torch.randn(5, 9)
    delta = 50.0 * torch.randn(5, 9)                      # huge -> both clamps active
    out = retract_glk(phi, delta, G, step_size=1.0, trust_region=0.1, max_norm=5.0)
    assert (out.norm(dim=-1) <= 5.0 + 1e-5).all()


def test_retract_glk_keeps_det_positive():
    # det(exp(embed phi)) = exp(tr) > 0 always: the GL+(K) identity-component property.
    G = generate_glk(3)
    phi = 0.3 * torch.randn(8, 9)
    delta = torch.randn(8, 9)
    out = retract_glk(phi, delta, G)
    dets = torch.linalg.det(torch.linalg.matrix_exp(embed_phi(out, G)))
    assert (dets > 0).all()


def test_retract_son_stays_orthogonal():
    # SO(N): embed(phi) is skew -> exp is orthogonal with det +1 (group membership).
    G = generate_son(4)                                   # (6,4,4)
    phi = 0.2 * torch.randn(7, 6)
    delta = torch.randn(7, 6)
    out = retract_son(phi, delta, G, max_norm=math.pi)
    A = embed_phi(out, G)
    assert torch.allclose(A, -A.transpose(-1, -2), atol=1e-5)          # skew
    R = torch.linalg.matrix_exp(A)
    eye = torch.eye(4).expand_as(R)
    assert torch.allclose(R @ R.transpose(-1, -2), eye, atol=1e-4)     # orthogonal
    assert torch.allclose(torch.linalg.det(R), torch.ones(7), atol=1e-4)
