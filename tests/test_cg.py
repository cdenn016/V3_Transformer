r"""Numerical Clebsch-Gordan intertwiners over the irrep registry."""

import pytest
import torch

from vfe3.geometry.generators import generate_son
from vfe3.geometry.irreps import irrep_generators


def test_irrep_generators_public_builder():
    G_def = generate_son(3, dtype=torch.float64)
    rho = irrep_generators(G_def, algebra="so", label="l2")
    assert rho.shape == (3, 5, 5)
    assert (rho + rho.transpose(-1, -2)).abs().max() < 1e-12


def test_so3_selection_rules():
    from vfe3.geometry.cg import cg_intertwiners
    # l1 (x) l1 = l0 (+) l1 (+) l2 : each target multiplicity 1
    for c, n in (("l0", 1), ("l1", 1), ("l2", 1), ("l3", 0)):
        C = cg_intertwiners(3, algebra="so", label_a="l1", label_b="l1", label_c=c)
        assert C.shape[0] == n, (c, C.shape)
    # l1 (x) l2 = l1 (+) l2 (+) l3 : no l0
    assert cg_intertwiners(3, algebra="so", label_a="l1", label_b="l2", label_c="l0").shape[0] == 0
    assert cg_intertwiners(3, algebra="so", label_a="l1", label_b="l2", label_c="l3").shape[0] == 1


def test_cg_intertwiner_is_equivariant():
    from vfe3.geometry.cg import cg_intertwiners
    from vfe3.geometry.irreps import irrep_generators
    G_def = generate_son(3, dtype=torch.float64)
    ra = irrep_generators(G_def, algebra="so", label="l1")
    rb = irrep_generators(G_def, algebra="so", label="l2")
    rc = irrep_generators(G_def, algebra="so", label="l2")
    C = cg_intertwiners(3, algebra="so", label_a="l1", label_b="l2", label_c="l2")[0]  # (5, 15)
    gen = torch.Generator().manual_seed(0)
    coeff = 0.4 * torch.randn(3, generator=gen, dtype=torch.float64)
    ga = torch.linalg.matrix_exp(torch.einsum("a,aij->ij", coeff, ra))
    gb = torch.linalg.matrix_exp(torch.einsum("a,aij->ij", coeff, rb))
    gc = torch.linalg.matrix_exp(torch.einsum("a,aij->ij", coeff, rc))
    x = torch.randn(3, generator=gen, dtype=torch.float64)
    y = torch.randn(5, generator=gen, dtype=torch.float64)
    lhs = C @ torch.kron(ga @ x, gb @ y)                 # C(g x (x) g y)
    rhs = gc @ (C @ torch.kron(x, y))                    # g C(x (x) y)
    assert (lhs - rhs).abs().max() < 1e-10


def test_cg_selection_enumerates_admissible_triples():
    from vfe3.geometry.cg import cg_selection
    sel = {(a, b, c) for a, b, c, _ in cg_selection(3, algebra="so",
                                                    labels=["l0", "l1", "l2"])}
    assert ("l1", "l1", "l2") in sel
    assert ("l1", "l2", "l1") in sel
    assert ("l0", "l1", "l1") in sel                     # l0 source acts as a learned gate
    assert ("l1", "l1", "l3") not in sel                 # target not in the spec
    # unordered source pairs: (l2, l1) never appears (canonical order a <= b)
    assert all(a <= b for a, b, _c in sel)


def test_cg_cost_guard():
    from vfe3.geometry.cg import cg_intertwiners
    with pytest.raises(ValueError, match="construction size"):
        cg_intertwiners(8, algebra="so", label_a="l3", label_b="l3", label_c="l3")


def test_cg_cache_immune_to_caller_mutation():
    from vfe3.geometry.cg import cg_intertwiners
    C1 = cg_intertwiners(3, algebra="so", label_a="l1", label_b="l1", label_c="l2")
    C1.mul_(0.0)                                         # caller misbehaves in place
    C2 = cg_intertwiners(3, algebra="so", label_a="l1", label_b="l1", label_c="l2")
    assert C2.abs().max() > 0                            # cache unharmed
    assert C1.data_ptr() != C2.data_ptr()                # no aliasing


def _tower_group():
    from vfe3.geometry.groups import get_group
    return get_group("so_n")(9, group_n=3,
                             irrep_spec=[("l0", 1), ("l1", 1), ("l2", 1)],
                             dtype=torch.float64)


def test_cg_coupling_zero_init_is_exact_passthrough():
    from vfe3.model.cg_coupling import CGCoupling
    grp = _tower_group()
    cpl = CGCoupling(3, "so", grp.irrep_dims, grp.irrep_labels).double()
    assert cpl.path_weights.shape[0] > 0
    mu = torch.randn(2, 4, 9, dtype=torch.float64)
    sig = torch.rand(2, 4, 9, dtype=torch.float64)
    mu2, sig2 = cpl(mu, sig)
    assert torch.equal(mu2, mu) and torch.equal(sig2, sig)


def test_cg_coupling_means_update_is_exactly_equivariant():
    from vfe3.model.cg_coupling import CGCoupling
    grp = _tower_group()
    cpl = CGCoupling(3, "so", grp.irrep_dims, grp.irrep_labels).double()
    with torch.no_grad():
        cpl.path_weights.copy_(0.3 * torch.randn(cpl.path_weights.shape[0],
                                                 dtype=torch.float64))
    g = torch.linalg.matrix_exp(
        torch.einsum("a,aij->ij", 0.4 * torch.randn(3, dtype=torch.float64), grp.generators))
    mu = torch.randn(5, 9, dtype=torch.float64)
    sig = torch.rand(5, 9, dtype=torch.float64)
    out_then_g = torch.einsum("kl,nl->nk", g, cpl(mu, sig)[0])
    g_then_out = cpl(torch.einsum("kl,nl->nk", g, mu), sig)[0]
    assert (out_then_g - g_then_out).abs().max() < 1e-12


def test_cg_coupling_self_product_reaches_other_types():
    # zero everything except one l1 (x) l1 -> l2 path: the l2 head must move, others must not.
    from vfe3.model.cg_coupling import CGCoupling
    grp = _tower_group()
    cpl = CGCoupling(3, "so", grp.irrep_dims, grp.irrep_labels).double()
    idx = next(p for p, (a, b, c) in enumerate(cpl.path_types)
               if (a, b, c) == ("l1", "l1", "l2"))
    with torch.no_grad():
        cpl.path_weights[idx] = 1.0
    mu = torch.randn(3, 9, dtype=torch.float64)
    mu2, _ = cpl(mu, torch.ones(3, 9, dtype=torch.float64))
    assert not torch.allclose(mu2[:, 4:9], mu[:, 4:9])   # l2 head updated
    assert torch.equal(mu2[:, 0:4], mu[:, 0:4])          # l0 and l1 heads untouched
