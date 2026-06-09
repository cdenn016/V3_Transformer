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
