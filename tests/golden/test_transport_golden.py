import pytest
import torch


def test_stable_matrix_exp_pair_matches_vfe2(vfe2_transport):
    from vfe3.geometry.transport import stable_matrix_exp_pair
    g = torch.Generator(device="cpu").manual_seed(0)
    M = torch.randn(2, 3, 4, 4, generator=g)
    ref_pos, ref_neg = vfe2_transport["gauge_utils"].stable_matrix_exp_pair(
        M, skew_symmetric=False
    )
    pos, neg = stable_matrix_exp_pair(M, skew_symmetric=False)
    assert torch.allclose(pos, ref_pos, atol=1e-5, rtol=1e-5)
    assert torch.allclose(neg, ref_neg, atol=1e-5, rtol=1e-5)


def test_transport_operators_learned_matches_vfe2(vfe2_transport):
    from vfe3.geometry.transport import compute_transport_operators
    from vfe3.geometry.groups import get_group
    grp = get_group("so_k")(K=4)
    g = torch.Generator(device="cpu").manual_seed(2)
    phi = 0.3 * torch.randn(2, 3, grp.generators.shape[0], generator=g)
    ref = vfe2_transport["transport_ops"].compute_transport_operators(
        phi, grp.generators, enforce_orthogonal=False, gauge_mode="learned"
    )
    got = compute_transport_operators(phi, grp, gauge_mode="learned")
    assert torch.allclose(got["exp_phi"],     ref["exp_phi"],     atol=1e-5, rtol=1e-5)
    assert torch.allclose(got["exp_neg_phi"], ref["exp_neg_phi"], atol=1e-5, rtol=1e-5)
    assert torch.allclose(got["Omega"],       ref["Omega"],       atol=1e-5, rtol=1e-5)


def test_transport_operators_glk_matches_vfe2(vfe2_transport):
    from vfe3.geometry.transport import compute_transport_operators
    from vfe3.geometry.groups import get_group
    grp = get_group("block_glk")(K=6, n_heads=3)
    g = torch.Generator(device="cpu").manual_seed(5)
    phi = 0.2 * torch.randn(2, 3, grp.generators.shape[0], generator=g)
    ref = vfe2_transport["transport_ops"].compute_transport_operators(
        phi, grp.generators, enforce_orthogonal=False, gauge_mode="learned"
    )
    got = compute_transport_operators(phi, grp, gauge_mode="learned")
    assert torch.allclose(got["Omega"], ref["Omega"], atol=1e-5, rtol=1e-5)


def test_transport_operators_trivial_is_identity():
    from vfe3.geometry.transport import compute_transport_operators
    from vfe3.geometry.groups import get_group
    grp = get_group("so_k")(K=4)
    phi = torch.zeros(2, 3, grp.generators.shape[0])
    out = compute_transport_operators(phi, grp, gauge_mode="trivial")
    assert torch.allclose(out["Omega"], torch.eye(4).expand(2, 3, 3, 4, 4), atol=1e-6)


def test_transport_operators_direct_matches_vfe2(vfe2_transport):
    from vfe3.geometry.transport import compute_transport_operators_direct
    g = torch.Generator(device="cpu").manual_seed(11)
    omega = torch.eye(4) + 0.2 * torch.randn(2, 3, 4, 4, generator=g)
    ref = vfe2_transport["transport_ops"].compute_transport_operators_direct(
        omega, gauge_mode="learned"
    )
    got = compute_transport_operators_direct(omega, gauge_mode="learned")
    assert torch.allclose(got["omega_j_inv"], ref["omega_j_inv"], atol=1e-4, rtol=1e-4)
    assert torch.allclose(got["Omega"],       ref["Omega"],       atol=1e-4, rtol=1e-4)


def test_transport_operators_direct_trivial_is_identity():
    from vfe3.geometry.transport import compute_transport_operators_direct
    omega = torch.eye(4) + 0.1 * torch.randn(2, 3, 4, 4)
    out = compute_transport_operators_direct(omega, gauge_mode="trivial")
    assert torch.allclose(out["Omega"], torch.eye(4).expand(2, 3, 3, 4, 4), atol=1e-6)


def test_omega_to_block_exp_pairs_matches_vfe2(vfe2_transport):
    from vfe3.geometry.transport import (
        compute_transport_operators,
        omega_to_block_exp_pairs,
    )
    from vfe3.geometry.groups import get_group
    grp = get_group("block_glk")(K=6, n_heads=3)
    g = torch.Generator(device="cpu").manual_seed(7)
    phi = 0.2 * torch.randn(2, 3, grp.generators.shape[0], generator=g)
    exp_phi = compute_transport_operators(phi, grp, gauge_mode="learned")["exp_phi"]
    ref = vfe2_transport["transport_ops"].omega_to_block_exp_pairs(exp_phi, grp.irrep_dims)
    got = omega_to_block_exp_pairs(exp_phi, grp.irrep_dims)
    assert len(got) == len(ref)
    for (gb, gbi), (rb, rbi) in zip(got, ref):
        assert torch.allclose(gb, rb, atol=1e-5)
        assert torch.allclose(gbi, rbi, atol=1e-4)
