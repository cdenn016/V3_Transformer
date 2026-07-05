r"""Arm 2a -- the non-gauge V x m table control for the blocks_K48 REMAND upgrade.

The SAME learned (V, n_gen) phi table is used NON-structurally: a frozen random readout R maps each
token's n_gen-dim code to an additive K-dim mean shift, and the returned phi is ZERO so the transport
Omega = exp(phi.G) = I (no gl(g) congruence). Matched learned params (V*n_gen, as the gauge cell) with
the gl(g) structure removed -- the capacity-vs-structure control. Deliberately NOT gauge equivariant.
See docs/2026-07-05-blocks-k48-followup-experiment-spec.md (Arm 2a).
"""
import torch

from vfe3.model.prior_bank import PriorBank, get_encode
import scaling


def _bank(encode_mode, V=12, K=4, n_gen=6):
    torch.manual_seed(0)
    return PriorBank(V, K, n_gen, phi_scale=0.1, encode_mode=encode_mode, use_prior_bank=False)


def test_additive_encode_is_registered():
    get_encode("per_token_additive")   # KeyError if absent


def test_additive_R_is_a_frozen_buffer_matched_to_ngen():
    pb = _bank("per_token_additive")
    assert hasattr(pb, "additive_R")
    assert pb.additive_R.shape == (pb.K, pb.n_gen)
    # R is a buffer, NOT a learned parameter -> learned-param count stays V*n_gen (matched to the gauge cell)
    assert id(pb.additive_R) not in {id(p) for p in pb.parameters()}
    assert pb.additive_R in set(pb.buffers())


def test_additive_encode_zeros_phi_and_shifts_mu():
    pb = _bank("per_token_additive")
    tok = torch.tensor([[0, 1, 2]])
    b = pb.encode(tok)
    # phi returned as ZERO -> Omega = exp(0)exp(0) = I (no gl(g) transport)
    assert torch.allclose(b.phi, torch.zeros_like(b.phi))
    # mu = prior_mu + phi_table @ R^T  (structure-free additive shift)
    prior = pb._prior_mu_table()[tok]
    shift = pb.phi_embed[tok] @ pb.additive_R.t()
    assert torch.allclose(b.mu, prior + shift, atol=1e-6)
    assert not torch.allclose(b.mu, prior)          # the shift is actually applied


def test_additive_phi_embed_receives_gradient():
    # the (V, n_gen) table is LEARNED and active via the additive path (not inert)
    pb = _bank("per_token_additive")
    b = pb.encode(torch.tensor([[0, 1, 2]]))
    b.mu.sum().backward()
    assert pb.phi_embed.grad is not None and pb.phi_embed.grad.abs().sum() > 0


def test_per_token_default_path_unchanged():
    pb = _bank("per_token")
    tok = torch.tensor([[0, 1, 2]])
    b = pb.encode(tok)
    assert torch.allclose(b.phi, pb.phi_embed[tok])   # default: phi IS the table (it transports)
    assert not hasattr(pb, "additive_R")              # no buffer on the pure path


def test_arm2_control_route_shape():
    cells = scaling.ROUTES["blocks_K48_ctrl_2x"]
    assert [c["overrides"]["n_heads"] for c in cells] == [16, 8, 6, 4, 2]
    # keep block_glk so n_gen (= 48*b) is matched to the gauge cell, but encode non-structurally
    assert all(c["overrides"]["gauge_group"] == "block_glk"       for c in cells)
    assert all(c["overrides"]["encode_mode"] == "per_token_additive" for c in cells)
    assert all(c["overrides"]["pos_phi"] == "none"                for c in cells)   # no positional transport either
    assert all(c["route"] == "blocks_K48_ctrl_2x"                 for c in cells)
