import pytest
import torch


def test_glk_matches_vfe2(vfe2_gen):
    from vfe3.geometry.generators import generate_glk
    ref = vfe2_gen["builders"].generate_glK_generators(5)
    got = generate_glk(5)
    assert torch.equal(got, ref)


def test_glk_sl_matches_vfe2(vfe2_gen):
    from vfe3.geometry.generators import generate_glk
    ref = vfe2_gen["builders"].generate_glK_generators(4, include_identity=False)
    got = generate_glk(4, include_identity=False)
    assert torch.allclose(got, ref, atol=1e-6)


def test_glk_multihead_matches_vfe2(vfe2_gen):
    from vfe3.geometry.generators import generate_glk_multihead
    ref = vfe2_gen["builders"].generate_glK_multihead_generators(6, 3)
    got = generate_glk_multihead(6, 3)
    assert torch.equal(got, ref)


def test_glk_cross_head_matches_vfe2(vfe2_gen):
    from vfe3.geometry.generators import generate_glk_cross_head
    pairs = [(0, 1), (1, 2)]
    ref = vfe2_gen["builders"].generate_glK_cross_head_generators(6, 3, pairs)
    got = generate_glk_cross_head(6, 3, pairs)
    assert torch.equal(got, ref)


def test_son_matches_vfe2(vfe2_gen):
    from vfe3.geometry.generators import generate_son
    ref = vfe2_gen["builders"].generate_soN_generators(5)
    got = generate_son(5)
    assert torch.equal(got, ref)


def test_closure_adds_bracket_direction_matches_vfe2(vfe2_gen):
    from vfe3.geometry.closure import close_under_brackets
    # {E_01, E_10} is NOT Lie-closed: [E_01, E_10] = E_00 - E_11 is a new
    # direction outside their span, so closure must ADD it (exercises the
    # additive SVD path, not just input preservation).
    E01 = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    E10 = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    gens = torch.stack([E01, E10], dim=0)                    # (2, 2, 2)
    ref_closed, ref_info = vfe2_gen["closure"].close_under_brackets(gens)
    got_closed, got_info = close_under_brackets(gens)
    # Closure genuinely grew the basis (gl(2) is 4-dim; sl(2) here -> 3).
    assert got_info["n_added"] >= 1
    assert got_info["final_dim"] == ref_info["final_dim"]
    assert got_info["converged"] == ref_info["converged"]
    assert got_closed.shape == ref_closed.shape
    assert torch.allclose(got_closed, ref_closed, atol=1e-6)
    # Inputs preserved verbatim as the first n_gen rows.
    assert torch.allclose(got_closed[: gens.shape[0]], gens, atol=1e-6)
