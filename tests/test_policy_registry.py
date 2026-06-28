r"""Phase 0 smoke test for the active-inference EFE policy seam: the three registries
(vfe3/inference/policy.py) and the VFE3Config policy fields + validation, all default-off
(spec docs/superpowers/specs/2026-06-28-active-inference-efe-policy-scorer-spec.md, Sections 3.2-3.3).
"""
import pytest

from vfe3.config import VFE3Config
from vfe3.inference.policy import (
    _AMBIGUITIES,
    _POLICIES,
    _PREFERENCES,
    get_ambiguity,
    get_policy,
    get_preference,
    register_ambiguity,
    register_policy,
    register_preference,
)


def test_default_config_constructs_with_policy_off():
    cfg = VFE3Config()
    assert cfg.policy_mode == "none"
    assert cfg.policy_horizon == 1 and cfg.policy_top_k == 8
    assert cfg.policy_precision == 1.0 and cfg.policy_preference == "task"
    assert cfg.policy_score_terms == ("risk", "ambiguity")
    assert cfg.policy_sigma_ambiguity_validated is False
    assert cfg.policy_sigma_gate_artifact is None


def test_none_policy_registered_but_never_dispatched():
    assert "none" in _POLICIES
    fn = get_policy("none")
    with pytest.raises(RuntimeError):           # exists only for config validation; never called
        fn()


def test_get_policy_unknown_key_raises_with_available():
    with pytest.raises(KeyError) as e:
        get_policy("not_a_real_policy")
    assert "none" in str(e.value)               # the error lists the available keys


def test_register_get_roundtrip_does_not_leak():
    for reg, get, store in ((register_policy, get_policy, _POLICIES),
                            (register_preference, get_preference, _PREFERENCES),
                            (register_ambiguity, get_ambiguity, _AMBIGUITIES)):
        name = "__phase0_smoke__"
        try:
            @reg(name)
            def _f():
                return 42
            assert get(name)() == 42
        finally:
            store.pop(name, None)
        assert name not in store


def test_phase1_registrations_present():
    # Phase 1 fills the registries; unknown keys still raise.
    assert {"none", "efe_one_step", "logprob_control", "efe_rollout"} <= set(_POLICIES)
    assert {"task", "held_out_predictive", "flat"} <= set(_PREFERENCES)
    assert {"likelihood_entropy", "sigma_mc"} <= set(_AMBIGUITIES)
    with pytest.raises(KeyError):
        get_preference("not_a_preference")
    with pytest.raises(KeyError):
        get_ambiguity("not_an_ambiguity")


def test_config_accepts_registered_and_rejects_unknown_policy_mode():
    for mode in ("none", "efe_one_step", "logprob_control", "efe_rollout"):
        VFE3Config(policy_mode=mode)            # all registered keys validate (no dispatch at construct)
    with pytest.raises(ValueError):
        VFE3Config(policy_mode="not_a_real_policy")


def test_config_rejects_bad_policy_numerics():
    with pytest.raises(ValueError):
        VFE3Config(policy_top_k=0)
    with pytest.raises(ValueError):
        VFE3Config(policy_horizon=0)
    with pytest.raises(ValueError):
        VFE3Config(policy_precision=0.0)


def test_sigma_gate_flag_requires_artifact():
    with pytest.raises(ValueError):
        VFE3Config(policy_sigma_ambiguity_validated=True)
    # with an artifact reference the structural guard is satisfied (content check is the gate step's job)
    cfg = VFE3Config(policy_sigma_ambiguity_validated=True,
                     policy_sigma_gate_artifact="vfe3_policy_results/sigma_gate/ckpt.json")
    assert cfg.policy_sigma_ambiguity_validated is True
