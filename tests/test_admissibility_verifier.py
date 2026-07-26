r"""Executable group/family admissibility verifier (PL17).

`GaugeGroup.invariant_for(family)` is only a string-membership test against `invariant_families`.
`check_admissible(group, family)` turns that declaration into a verified invariant: it draws random
group elements g = exp(sum_a c_a G_a), pushes a random Gaussian belief PAIR forward by the family's
representation (the GL(K) congruence mu -> g mu, Sigma -> g Sigma g^T), and asserts the registered
divergence is invariant, D(rho(g) q || rho(g) p) = D(q || p), to tolerance.

The check is non-vacuous: for the FULL Gaussian it is invariant under every g in GL(K) (so every
registered group passes, confirming the 'gaussian_full' declaration), but for a DIAGONAL family the
congruence breaks the diagonal structure, so the verifier correctly returns False under a
non-diagonal group -- catching exactly the kind of wrongly-declared admissibility the roadmap warns
about.

Audit 2026-07-26 E-05: the family names here were 'gaussian', which is NOT a registered family, and
the verifier branched on three such literals. It now dispatches through the registry's
``from_covariance`` hook, so every registered family is covered -- including the two added in
2026-07, neither of which any invariance test could reach before.
"""

import pytest
import torch

from vfe3.families.base import divergence_families
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
    assert check_admissible(grp, "gaussian_full", n_samples=6) is True
    # and the boolean declaration matches the verified truth
    assert grp.invariant_for("gaussian_full") == check_admissible(grp, "gaussian_full", n_samples=6)


@pytest.mark.parametrize("name,kwargs", GROUPS)
def test_declaration_matches_the_verifier_for_every_registered_family(name, kwargs):
    # E-05: the declaration is now pinned against the executable certificate across the WHOLE family
    # registry, not just the one name the old literals covered, so `invariant_families` cannot drift
    # from measured behavior and a newly registered family is checked the moment it is registered.
    grp = get_group(name)(**kwargs)
    for family in divergence_families():
        try:
            verified = check_admissible(grp, family, n_samples=4)
        except NotImplementedError:
            # No full-covariance readout: the documented extension point. Such a family must not be
            # DECLARED invariant, since nothing can verify the declaration.
            assert not grp.invariant_for(family), family
            continue
        assert grp.invariant_for(family) == verified, family


def test_squared_hellinger_also_invariant_for_full_gaussian():
    grp = get_group("glk")(K=4)
    assert check_admissible(grp, "gaussian_full", functional="squared_hellinger", n_samples=6) is True


def test_renyi_alpha_not_one_invariant_for_full_gaussian():
    grp = get_group("sp")(K=4)
    assert check_admissible(grp, "gaussian_full", alpha=0.5, n_samples=6) is True


@pytest.mark.parametrize(
    "family", ["gaussian_diagonal", "gaussian_diagonal_exact", "gaussian_frame_diagonal"])
def test_diagonal_families_not_invariant_under_general_glk_congruence(family):
    # NEGATIVE CONTROL: a diagonal family's stored-parameter divergence is NOT invariant under a
    # general GL(K) congruence (g Sigma g^T is not diagonal), so the verifier must return False --
    # proving it is not a vacuous always-True check. The two 2026-07 families are included because
    # before E-05 the verifier could not reach them at all.
    grp = get_group("glk")(K=4)
    assert check_admissible(grp, family, n_samples=6) is False


def test_unregistered_family_raises_the_registry_error():
    grp = get_group("glk")(K=4)
    with pytest.raises(KeyError, match="no family registered"):
        check_admissible(grp, "categorical")


def test_registered_family_without_a_covariance_readout_raises_not_implemented():
    # The documented extension point: laplace_diagonal IS registered, but its scale parameter is not
    # a second moment, so it has no from_covariance and the verifier reports the gap rather than
    # silently pushing a Gaussian representation through a non-Gaussian family.
    grp = get_group("glk")(K=4)
    with pytest.raises(NotImplementedError, match="from_covariance"):
        check_admissible(grp, "laplace_diagonal")
