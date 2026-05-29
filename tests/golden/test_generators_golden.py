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
