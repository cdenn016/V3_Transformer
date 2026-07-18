r"""Tier-1 transport production behavior and numerical parity oracles (2026-07-05).

Production requests per-head transport-mean contraction. The retained low-level false-valued
path is a dense-versus-factored parity oracle. The norm-keyed float64 island of
stable_matrix_exp_pair remains controlled by cfg.exp_fp64_mode / cfg.exp_fp64_norm_threshold.

Pins: (a) the per-head factored mean equals the dense-K mean to fp32 reassociation (allclose
atol 1e-6), including the RoPE-wrapped route; (b) 'norm' mode with an unreachable threshold is
the fp32 path (allclose 1e-5 against the dim-rule fp64 result on small-norm phi; byte-equal to a
forced-fp32 dim reference), while a reachable threshold re-enters the fp64 island byte-identically
to the dim rule; (c) every toggle at its default is byte-identical to the pre-toggle build.
"""

import torch

from vfe3.geometry.groups import get_group
from vfe3.geometry.transport import (
    FactoredTransport,
    stable_matrix_exp_pair,
    transport_mean,
)
from vfe3.inference.e_step import build_belief_transport

B, N = 2, 6


def _phi(group, seed, scale=0.3, b=B, n=N):
    g = torch.Generator().manual_seed(seed)
    return scale * torch.randn(b, n, group.generators.shape[0], generator=g)


def _mu(K, seed, b=B, n=N):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, n, K, generator=g)


def _block_diag_matrix(block_dims, block_norm, seed, b=B, n=N):
    r"""(b, n, K, K) block-diagonal Lie-algebra matrix with EXACT per-block Frobenius norm."""
    g = torch.Generator().manual_seed(seed)
    K = sum(block_dims)
    m = torch.zeros(b, n, K, K)
    start = 0
    for d in block_dims:
        end = start + d
        blk = torch.randn(b, n, d, d, generator=g)
        blk = blk / blk.norm(dim=(-2, -1), keepdim=True) * block_norm
        m[..., start:end, start:end] = blk
        start = end
    return m


# -- (a) per-head transport_mean vs the dense-K contraction --------------------------------------


def test_per_head_transport_mean_matches_dense(device):
    grp = get_group("block_glk")(8, 2, device=device)
    phi = _phi(grp, 0).to(device)
    mu = _mu(8, 1).to(device)

    off = build_belief_transport(phi, grp, transport_mode="flat")
    on = build_belief_transport(phi, grp, transport_mode="flat", transport_mean_per_head=True)
    assert isinstance(off, FactoredTransport) and not off.mean_per_head
    assert isinstance(on, FactoredTransport) and on.mean_per_head

    got_off = transport_mean(off, mu)                          # dense-K factored contraction
    got_on = transport_mean(on, mu)                            # per-head contraction
    assert got_on.shape == (B, N, N, 8)
    assert torch.allclose(got_on, got_off, atol=1e-6)
    # Independent reference: the fully dense (B, N, N, K, K) Omega mean.
    dense = transport_mean(off.to_dense_omega(), mu)
    assert torch.allclose(got_on, dense, atol=1e-6)


def test_per_head_transport_mean_rope_wrapped_matches_dense(device):
    grp = get_group("block_glk")(8, 2, device=device)
    phi = _phi(grp, 2).to(device)
    mu = _mu(8, 3).to(device)
    g = torch.Generator().manual_seed(4)
    rope, _ = torch.linalg.qr(torch.randn(N, 8, 8, generator=g))   # (N, K, K) orthogonal rotations
    rope = rope.to(device)

    off = build_belief_transport(phi, grp, transport_mode="flat", rope=rope)
    on = build_belief_transport(phi, grp, transport_mode="flat", rope=rope,
                                transport_mean_per_head=True)
    assert isinstance(on.base, FactoredTransport) and on.base.mean_per_head
    assert torch.allclose(transport_mean(on, mu), transport_mean(off, mu), atol=1e-6)


# -- (b) norm-keyed float64 island ----------------------------------------------------------------


def test_norm_mode_unreachable_threshold_is_fp32_and_close_to_dim(device):
    # d_head = 20 fires the dim rule (fp64); 'norm' with an unreachable threshold stays fp32.
    grp = get_group("block_glk")(40, 2, device=device)
    phi = _phi(grp, 5, scale=0.1).to(device)
    mu = _mu(40, 6).to(device)

    dim_t = build_belief_transport(phi, grp, transport_mode="flat")
    norm_t = build_belief_transport(phi, grp, transport_mode="flat",
                                    exp_fp64_mode="norm", exp_fp64_norm_threshold=1e9)
    assert torch.allclose(norm_t.exp_phi, dim_t.exp_phi, atol=1e-5)
    assert torch.allclose(norm_t.exp_neg_phi, dim_t.exp_neg_phi, atol=1e-5)
    assert torch.allclose(transport_mean(norm_t, mu), transport_mean(dim_t, mu), atol=1e-4)


def test_norm_mode_zero_threshold_equals_dim_fp64_exactly(device):
    # Threshold 0.0 is always reached (norms >= 0), so 'norm' takes the SAME fp64 island as the
    # dim rule at d_head = 20 -> byte-identical factors.
    grp = get_group("block_glk")(40, 2, device=device)
    phi = _phi(grp, 7, scale=0.1).to(device)

    dim_t = build_belief_transport(phi, grp, transport_mode="flat")
    norm0_t = build_belief_transport(phi, grp, transport_mode="flat",
                                     exp_fp64_mode="norm", exp_fp64_norm_threshold=0.0)
    assert torch.equal(norm0_t.exp_phi, dim_t.exp_phi)
    assert torch.equal(norm0_t.exp_neg_phi, dim_t.exp_neg_phi)


def test_stable_exp_norm_mode_small_norm_takes_fp32_path_exactly(device):
    # Small clamped block norms (0.5 << 5.0) keep 'norm' mode on the fp32 path: byte-identical to
    # the dim rule with the threshold pushed out of reach (the forced-fp32 reference), at d=25
    # blocks where the default dim rule would upcast.
    bd = [25, 25]
    m = _block_diag_matrix(bd, 0.5, seed=8).to(device)
    ref_pos, ref_neg = stable_matrix_exp_pair(m, block_dims=bd, exp_dim=25, dim_threshold=1000)
    got_pos, got_neg = stable_matrix_exp_pair(m, block_dims=bd, exp_dim=25,
                                              exp_fp64_mode="norm", exp_fp64_norm_threshold=5.0)
    assert torch.equal(got_pos, ref_pos)
    assert torch.equal(got_neg, ref_neg)


def test_stable_exp_norm_mode_large_norm_reenters_fp64_island(device):
    # Genuinely large block norms (8.0 >= 5.0, below the max_norm=15 clamp) re-enter the fp64
    # island: byte-identical to the dim rule's fp64 result at d=25 blocks.
    bd = [25, 25]
    m = _block_diag_matrix(bd, 8.0, seed=9).to(device)
    ref_pos, ref_neg = stable_matrix_exp_pair(m, block_dims=bd, exp_dim=25)   # dim rule -> fp64
    got_pos, got_neg = stable_matrix_exp_pair(m, block_dims=bd, exp_dim=25,
                                              exp_fp64_mode="norm", exp_fp64_norm_threshold=5.0)
    assert torch.equal(got_pos, ref_pos)
    assert torch.equal(got_neg, ref_neg)


# -- (c) defaults byte-identical -------------------------------------------------------------------


def test_defaults_byte_identical(device):
    grp = get_group("block_glk")(8, 2, device=device)
    phi = _phi(grp, 10).to(device)
    mu = _mu(8, 11).to(device)

    bare = build_belief_transport(phi, grp, transport_mode="flat")
    kw = build_belief_transport(phi, grp, transport_mode="flat",
                                transport_mean_per_head=False,
                                exp_fp64_mode="dim", exp_fp64_norm_threshold=5.0)
    assert torch.equal(bare.exp_phi, kw.exp_phi)
    assert torch.equal(bare.exp_neg_phi, kw.exp_neg_phi)
    assert torch.equal(transport_mean(bare, mu), transport_mean(kw, mu))


def test_stable_exp_defaults_byte_identical(device):
    bd = [4, 4]
    m = _block_diag_matrix(bd, 0.5, seed=12).to(device)
    bare_pos, bare_neg = stable_matrix_exp_pair(m, block_dims=bd, exp_dim=4)
    kw_pos, kw_neg = stable_matrix_exp_pair(m, block_dims=bd, exp_dim=4,
                                            exp_fp64_mode="dim", exp_fp64_norm_threshold=5.0)
    assert torch.equal(bare_pos, kw_pos)
    assert torch.equal(bare_neg, kw_neg)
