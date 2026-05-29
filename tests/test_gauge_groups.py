import pytest
import torch

from vfe3.geometry.groups import GaugeGroup, get_group, register_group


def test_glk_group_full_basis():
    grp = get_group("glk")(K=5)
    assert grp.generators.shape == (25, 5, 5)
    assert grp.irrep_dims == [5]
    assert grp.skew_symmetric is False


def test_block_glk_group_is_block_diagonal():
    grp = get_group("block_glk")(K=6, n_heads=3)
    assert grp.irrep_dims == [2, 2, 2]
    d = 2
    mask = torch.ones(6, 6)
    for h in range(3):
        mask[h * d:(h + 1) * d, h * d:(h + 1) * d] = 0.0
    for g in grp.generators:
        assert torch.count_nonzero(g * mask) == 0


def test_so_k_group_is_skew():
    grp = get_group("so_k")(K=4)
    assert grp.skew_symmetric is True
    assert torch.allclose(
        grp.generators + grp.generators.transpose(-1, -2),
        torch.zeros_like(grp.generators),
        atol=1e-6,
    )


def test_block_glk_with_cross_coupling_grows_basis():
    base = get_group("block_glk")(K=6, n_heads=3)
    coupled = get_group("block_glk")(K=6, n_heads=3, cross_couplings=[(0, 1)])
    assert coupled.generators.shape[0] == base.generators.shape[0] + 4
    # A cross-coupled group is not block-diagonal with d_head blocks; its
    # irrep structure is reported as the single block [K] (super-block
    # decomposition is deferred to Phase 2b transport).
    assert base.irrep_dims == [2, 2, 2]
    assert coupled.irrep_dims == [6]


def test_unknown_group_raises():
    with pytest.raises(KeyError):
        get_group("not_a_group")


def test_gaussian_admissibility_is_declared():
    grp = get_group("glk")(K=4)
    assert grp.invariant_for("gaussian") is True


@pytest.mark.parametrize("spec", [
    ("glk",       {"K": 4}),
    ("block_glk", {"K": 6, "n_heads": 3}),
    ("so_k",      {"K": 4}),
])
def test_full_kl_invariant_under_group_pushforward(spec):
    # For a random group element g = exp(sum_a c_a G_a), the Gaussian KL is
    # invariant under common pushforward mu->g mu, Sigma->g Sigma g^T.
    from vfe3.divergence import kl
    name, kwargs = spec
    grp = get_group(name)(**kwargs)
    K = sum(grp.irrep_dims)
    gen = torch.Generator().manual_seed(0)
    coeff = 0.2 * torch.randn(grp.generators.shape[0], generator=gen)
    M = torch.einsum("a,aij->ij", coeff, grp.generators)
    g = torch.linalg.matrix_exp(M)                              # (K, K) in G

    mu_q = torch.randn(5, K, generator=gen)
    mu_p = torch.randn(5, K, generator=gen)
    Aq = torch.randn(5, K, K, generator=gen)
    Ap = torch.randn(5, K, K, generator=gen)
    S_q = Aq @ Aq.transpose(-1, -2) + torch.eye(K)
    S_p = Ap @ Ap.transpose(-1, -2) + torch.eye(K)

    base = kl(mu_q, S_q, mu_p, S_p, family="gaussian_full")
    mu_q2 = torch.einsum("kl,nl->nk", g, mu_q)
    mu_p2 = torch.einsum("kl,nl->nk", g, mu_p)
    S_q2 = g @ S_q @ g.transpose(-1, -2)
    S_p2 = g @ S_p @ g.transpose(-1, -2)
    moved = kl(mu_q2, S_q2, mu_p2, S_p2, family="gaussian_full")

    assert grp.invariant_for("gaussian")
    assert torch.allclose(base, moved, atol=1e-3, rtol=1e-3)


def test_glk_generators_exact_entries():
    grp = get_group("glk")(K=2)
    expected = torch.tensor([
        [[1., 0.], [0., 0.]],   # E_00
        [[0., 1.], [0., 0.]],   # E_01
        [[0., 0.], [1., 0.]],   # E_10
        [[0., 0.], [0., 1.]],   # E_11
    ])
    assert torch.equal(grp.generators, expected)


def test_son_generators_exact_entries():
    grp = get_group("so_k")(K=2)
    expected = torch.tensor([[[0., 1.], [-1., 0.]]])   # L_01 = E_01 - E_10
    assert torch.equal(grp.generators, expected)


def test_closure_of_e01_e10_is_sl2():
    # {E01, E10} closes to a 3-dim algebra (adds E00 - E11 direction).
    from vfe3.geometry.closure import close_under_brackets
    E01 = torch.tensor([[0., 1.], [0., 0.]])
    E10 = torch.tensor([[0., 0.], [1., 0.]])
    gens = torch.stack([E01, E10], dim=0)
    closed, info = close_under_brackets(gens)
    assert info["final_dim"] == 3 and info["converged"]
    # Re-closing an already-closed basis adds nothing.
    closed2, info2 = close_under_brackets(closed)
    assert info2["n_added"] == 0
