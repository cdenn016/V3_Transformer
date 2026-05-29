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


from vfe3.geometry.generators import generate_glk_multihead
from vfe3.geometry.lie_ops import clamp_phi_trace, project_phi_to_slk


def _block_traces(phi, G, irrep_dims):
    A = embed_phi(phi, G)
    outs, start = [], 0
    for d in irrep_dims:
        end = start + d
        outs.append(A[..., start:end, start:end].diagonal(dim1=-2, dim2=-1).sum(-1))
        start = end
    return torch.stack(outs, dim=-1)                      # (..., n_blocks)


def test_project_slk_zeros_block_trace_and_unit_det():
    G = generate_glk_multihead(6, 2)                      # 2 blocks of gl(3)
    irrep = [3, 3]
    phi = 0.5 * torch.randn(5, G.shape[0])
    out = project_phi_to_slk(phi, G, irrep)
    assert torch.allclose(_block_traces(out, G, irrep), torch.zeros(5, 2), atol=1e-5)
    # det of each block's group element == 1
    A = embed_phi(out, G)
    for s, d in [(0, 3), (3, 3)]:
        blk = A[..., s:s + d, s:s + d]
        det = torch.linalg.det(torch.linalg.matrix_exp(blk))
        assert torch.allclose(det, torch.ones(5), atol=1e-4)


def test_clamp_phi_trace_bounds_block_trace():
    G = generate_glk_multihead(6, 2)
    irrep = [3, 3]
    phi = 2.0 * torch.randn(5, G.shape[0])                # large traces
    T = 0.5
    out = clamp_phi_trace(phi, G, irrep, trace_max=T)
    assert (_block_traces(out, G, irrep).abs() <= T + 1e-4).all()


from vfe3.geometry.groups import get_group
from vfe3.geometry.retraction import retract_phi


def test_retract_phi_glk_with_slk_projection():
    grp = get_group("block_glk")(6, 2)                    # block GL(3)^2, irrep [3,3]
    phi = 0.5 * torch.randn(4, grp.generators.shape[0])
    delta = torch.randn_like(phi)
    out = retract_phi(phi, delta, grp, project_slk=True)
    assert torch.allclose(_block_traces(out, grp.generators, grp.irrep_dims),
                          torch.zeros(4, 2), atol=1e-4)


def test_retract_phi_son_path_orthogonal_no_det_control():
    grp = get_group("so_k")(4)
    phi = 0.2 * torch.randn(4, grp.generators.shape[0])
    delta = torch.randn_like(phi)
    out = retract_phi(phi, delta, grp)                    # skew -> SO path, det control ignored
    R = torch.linalg.matrix_exp(embed_phi(out, grp.generators))
    eye = torch.eye(4).expand_as(R)
    assert torch.allclose(R @ R.transpose(-1, -2), eye, atol=1e-4)


def test_retract_phi_defaults_pick_group_constants():
    # GL(K) default max_norm=5.0; a huge delta saturates to that, not pi.
    grp = get_group("glk")(3)
    phi = torch.zeros(2, 9)
    delta = 1e3 * torch.ones(2, 9)
    out = retract_phi(phi, delta, grp)
    assert (out.norm(dim=-1) <= 5.0 + 1e-4).all()
