r"""Executable group/family admissibility verifier (PL17).

`GaugeGroup.invariant_for(family)` is only a string-membership test against `invariant_families`.
`check_admissible(group, family)` turns that declaration into a verified invariant: it draws random
group elements g = exp(sum_a c_a G_a), pushes a random Gaussian belief PAIR forward by the family's
representation (the GL(K) congruence mu -> g mu, Sigma -> g Sigma g^T), and asserts the registered
divergence is invariant, D(rho(g) q || rho(g) p) = D(q || p), to tolerance.

The check is non-vacuous: for the FULL Gaussian it is invariant under every g in GL(K) (so every
registered group passes, confirming the 'gaussian' declaration), but for the DIAGONAL Gaussian the
congruence breaks the diagonal structure, so the verifier correctly returns False under a
non-diagonal group -- catching exactly the kind of wrongly-declared admissibility the roadmap warns
about.
"""

import pytest
import torch

from vfe3.geometry.groups import check_admissible, get_group

GROUPS = [
    ("glk",            {"K": 4}),
    ("block_glk",      {"K": 6, "n_heads": 3}),
    ("tied_block_glk", {"K": 6, "n_heads": 3}),
    ("so_k",           {"K": 4}),
    ("sp",             {"K": 4}),
    ("so_n",           {"K": 4, "group_n": 3, "irrep_spec": [("l0", 1), ("l1", 1)]}),
    ("sp_n",           {"K": 5, "group_n": 4, "irrep_spec": [("sym0", 1), ("sym1", 1)]}),
]


@pytest.mark.parametrize("name,kwargs", GROUPS)
def test_full_gaussian_admissible_for_every_registered_group(name, kwargs):
    grp = get_group(name)(**kwargs)
    # the executable verifier confirms the divergence is invariant under the group's congruence
    assert check_admissible(grp, "gaussian", n_samples=6) is True
    # and the boolean declaration matches the verified truth
    assert grp.invariant_for("gaussian") == check_admissible(grp, "gaussian", n_samples=6)


def test_squared_hellinger_also_invariant_for_full_gaussian():
    grp = get_group("glk")(K=4)
    assert check_admissible(grp, "gaussian", functional="squared_hellinger", n_samples=6) is True


def test_renyi_alpha_not_one_invariant_for_full_gaussian():
    grp = get_group("sp")(K=4)
    assert check_admissible(grp, "gaussian", alpha=0.5, n_samples=6) is True


def test_diagonal_gaussian_not_invariant_under_general_glk_congruence():
    # NEGATIVE CONTROL: the diagonal family's divergence is NOT invariant under a general GL(K)
    # congruence (g Sigma g^T is not diagonal), so the verifier must return False -- proving it is
    # not a vacuous always-True check.
    grp = get_group("glk")(K=4)
    assert check_admissible(grp, "gaussian_diagonal", n_samples=6) is False


def test_unknown_family_representation_raises():
    grp = get_group("glk")(K=4)
    with pytest.raises(NotImplementedError):
        check_admissible(grp, "categorical")
