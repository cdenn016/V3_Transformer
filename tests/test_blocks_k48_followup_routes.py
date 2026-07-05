r"""Routes for the blocks_K48 follow-up battery that upgrades the red/blue REMAND (surviving sub-claim
S1). Arm 1 = matched 491.52M-token blocks_K48 sweep over the S1 window GL(3)..GL(24); Arm 3 = the same
window under the TIED gauge (n_gen = d_head^2 instead of b*K), which asks whether per-block UNTIED
richness drives the S1 curve or the tied variant matches it at far fewer parameters. See
docs/2026-07-05-blocks-k48-followup-experiment-spec.md and
docs/debates/2026-07-05-blocks-k48-gauge-block-scaling-axis/.
"""
import pytest

from vfe3.geometry.groups import get_group
import scaling

S1_BLOCKS = [3, 6, 8, 12, 24]
S1_HEADS  = [16, 8, 6, 4, 2]          # n_heads = K / b at K = 48
K = 48


def test_arm1_matched_budget_route_shape():
    cells = scaling.ROUTES["blocks_K48_2x"]
    assert [c["label"]                for c in cells] == [f"K48_GL{b}" for b in S1_BLOCKS]
    assert [c["overrides"]["n_heads"] for c in cells] == S1_HEADS
    assert all(c["overrides"]["gauge_group"] == "block_glk" for c in cells)
    # distinct route tag so scaling_analysis does not conflate the 491.52M points with the 245.76M run
    assert all(c["route"] == "blocks_K48_2x" for c in cells)


def test_arm3_tied_route_shape():
    cells = scaling.ROUTES["blocks_K48_tied_2x"]
    assert [c["overrides"]["n_heads"] for c in cells] == S1_HEADS
    assert all(c["overrides"]["gauge_group"] == "tied_block_glk" for c in cells)
    assert all(c["route"] == "blocks_K48_tied_2x" for c in cells)


def test_base_blocks_k48_route_unchanged():
    # the gauge_group/tag parameters are default-preserving: the original route is untouched
    cells = scaling.ROUTES["blocks_K48"]
    assert all(c["route"] == "blocks_K48"          for c in cells)
    assert all(c["overrides"]["gauge_group"] == "block_glk" for c in cells)


@pytest.mark.parametrize("b,h", list(zip(S1_BLOCKS, S1_HEADS)))
def test_ngen_block_vs_tied_builds(b, h):
    # geometry actually builds (not just the route dicts) and the n_gen counts are the arm's lever
    block = get_group("block_glk")(K, h)
    tied  = get_group("tied_block_glk")(K, h)
    assert block.generators.shape == (b * K, K, K)   # untied n_gen = K^2 / h = b * K
    assert tied.generators.shape  == (b * b, K, K)   # tied   n_gen = d_head^2 = b^2
    assert tied.generators.shape[0] < block.generators.shape[0]   # Arm 3: strictly fewer params
    assert block.irrep_dims == [b] * h and tied.irrep_dims == [b] * h
