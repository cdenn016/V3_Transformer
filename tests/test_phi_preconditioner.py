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
