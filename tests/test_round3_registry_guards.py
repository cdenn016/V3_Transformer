r"""Round-3 registry-guard tests (audit 2026-07-01 round-3): every register_* decorator in the
round-3 batch fails closed on a duplicate key (KeyError) and replaces only under override=True,
completing the F12 fail-closed contract across the config-selected seams. Also pins that
``override`` is KEYWORD-ONLY on every guarded register_* (punch item 6: register_policy had it
positional).

Each check saves the original entry first and restores it via override=True in a ``finally``,
so global registry state is unchanged for the rest of the suite.
"""
import inspect

import pytest

import vfe3.families.gaussian  # noqa: F401  -- registers the gaussian families at import
from vfe3 import numerics as numerics_mod
from vfe3.families import base as families_base
from vfe3.geometry import groups as groups_mod
from vfe3.geometry import irreps as irreps_mod
from vfe3.geometry import norms as norms_mod
from vfe3.geometry import phi_preconditioner as precond_mod
from vfe3.geometry import retraction as retraction_mod
from vfe3.geometry import rope as rope_mod
from vfe3.gradients import kernels as kernels_mod
from vfe3.inference import policy as policy_mod
from vfe3.model import positional_phi as pos_phi_mod
from vfe3.model import prior_bank as prior_bank_mod


# (register_fn, backing registry dict, a key known to be registered at import time)
_DECORATOR_REGISTRIES = [
    (families_base.register_family,               families_base._FAMILIES,              "gaussian_diagonal"),
    (families_base.register_functional,           families_base._FUNCTIONALS,           "renyi"),
    (families_base.register_functional_per_coord, families_base._FUNCTIONALS_PER_COORD, "renyi"),
    (kernels_mod.register_kernel,                 kernels_mod._KERNELS,                 "gaussian_diagonal"),
    (retraction_mod.register_retraction,          retraction_mod._RETRACTIONS,          "spd_affine"),
    (norms_mod.register_norm,                     norms_mod._NORMS,                     "mahalanobis"),
    (groups_mod.register_group,                   groups_mod._GROUPS,                   "glk"),
    (precond_mod.register_precond,                precond_mod._PRECOND,                 "none"),
    (rope_mod.register_pos_rotation,              rope_mod._POS_ROTATIONS,              "rope"),
    (pos_phi_mod.register_pos_phi,                pos_phi_mod._POS_PHI,                 "none"),
    (prior_bank_mod.register_encode,              prior_bank_mod._ENCODERS,             "per_token"),
    (prior_bank_mod.register_decode,              prior_bank_mod._DECODERS,             "diagonal"),
    (numerics_mod.register_monitor,               numerics_mod._MONITORS,               "nan_fraction"),
    (policy_mod.register_preference,              policy_mod._PREFERENCES,              "flat"),
    (policy_mod.register_ambiguity,               policy_mod._AMBIGUITIES,              "likelihood_entropy"),
]

_IDS = [reg.__name__ for reg, _, _ in _DECORATOR_REGISTRIES]


@pytest.mark.parametrize("reg, registry, name", _DECORATOR_REGISTRIES, ids=_IDS)
def test_duplicate_key_fails_closed_and_override_replaces(reg, registry, name):
    assert name in registry, f"expected {name!r} to be pre-registered"
    orig = registry[name]
    try:
        def _dup(*args, **kwargs):
            pass
        with pytest.raises(KeyError):
            reg(name)(_dup)
        assert registry[name] is orig               # the first registration survived

        def _replacement(*args, **kwargs):
            pass
        assert reg(name, override=True)(_replacement) is _replacement
        assert registry[name] is _replacement       # explicit override replaces it
    finally:
        reg(name, override=True)(orig)              # restore global registry state
    assert registry[name] is orig


def test_register_irrep_duplicate_key_fails_closed_and_override_replaces():
    # register_irrep is a direct-call registration (not a decorator): same fail-closed contract.
    key = "so:l"
    assert key in irreps_mod._IRREPS
    orig_dim_fn, orig_build_fn = irreps_mod._IRREPS[key]

    def _dup_dim(n, p):
        return 0

    def _dup_build(g, p):
        return g

    try:
        with pytest.raises(KeyError):
            irreps_mod.register_irrep(key, _dup_dim, _dup_build)
        assert irreps_mod._IRREPS[key] == (orig_dim_fn, orig_build_fn)

        irreps_mod.register_irrep(key, _dup_dim, _dup_build, override=True)
        assert irreps_mod._IRREPS[key] == (_dup_dim, _dup_build)
    finally:
        irreps_mod.register_irrep(key, orig_dim_fn, orig_build_fn, override=True)
    assert irreps_mod._IRREPS[key] == (orig_dim_fn, orig_build_fn)


@pytest.mark.parametrize(
    "reg",
    [reg for reg, _, _ in _DECORATOR_REGISTRIES]
    + [policy_mod.register_policy, irreps_mod.register_irrep],
    ids=_IDS + ["register_policy", "register_irrep"],
)
def test_override_parameter_is_keyword_only(reg):
    # punch item 6: register_policy's override was positional; the convention (register_alpha,
    # register_prior, register_compose) is keyword-only, pinned here for the whole batch.
    param = inspect.signature(reg).parameters["override"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is False
