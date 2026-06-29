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
    # non-'none' modes need a context-free preference in the generic config (audit F4 guard); 'flat' is it
    for mode in ("none", "efe_one_step", "logprob_control"):
        VFE3Config(policy_mode=mode, policy_preference="flat")   # H=1 registered keys validate
    VFE3Config(policy_mode="efe_rollout", policy_preference="flat", policy_horizon=2)  # H>1 scorer needs horizon>1
    with pytest.raises(ValueError):
        VFE3Config(policy_mode="not_a_real_policy")


def test_generic_policy_rejects_context_requiring_preference():
    # audit F4 (2026-06-28): generate() cannot feed a per-episode goal / p_data, so 'task' and
    # 'held_out_predictive' are invalid with policy_mode != 'none'. The DEFAULT preference is 'task',
    # so simply turning on a scorer must raise at construction, not fail mid-generate with a missing-goal
    # TypeError. (The default config, policy_mode='none', is unaffected -- the guard only fires once a
    # scorer is enabled.)
    for pref in ("task", "held_out_predictive"):
        with pytest.raises(ValueError):
            VFE3Config(policy_mode="efe_one_step", policy_preference=pref)
    # fail-closed: even an unrecognized preference is rejected once a scorer is on (the old deny-list
    # form would have let it through to a mid-generate failure).
    with pytest.raises(ValueError):
        VFE3Config(policy_mode="efe_one_step", policy_preference="some_future_preference")
    VFE3Config(policy_mode="efe_one_step", policy_preference="flat")   # the context-free preference is OK
    VFE3Config(policy_preference="task")                               # 'none' default: guard does not fire


def test_config_rejects_bad_policy_numerics():
    with pytest.raises(ValueError):
        VFE3Config(policy_top_k=0)
    with pytest.raises(ValueError):
        VFE3Config(policy_horizon=0)
    with pytest.raises(ValueError):
        VFE3Config(policy_precision=0.0)


def test_config_rejects_invalid_policy_score_terms():
    # audit F5 (2026-06-28): a typo'd or empty score-term set must fail at construction, not as a cryptic
    # KeyError deep inside _policy_efe_one_step where G(pi) is summed over the terms.
    with pytest.raises(ValueError):
        VFE3Config(policy_score_terms=("nonsense",))
    with pytest.raises(ValueError):
        VFE3Config(policy_score_terms=())                         # empty: G has nothing to sum
    VFE3Config(policy_score_terms=("risk",))                      # any subset of the EFE terms is valid
    VFE3Config(policy_score_terms=("risk", "ambiguity", "epistemic"))


def test_config_rejects_efe_one_step_with_horizon_gt_1():
    # audit F5: efe_one_step is the H=1 scorer; horizon!=1 raised a ValueError mid-generate before, now
    # it is rejected at construction.
    with pytest.raises(ValueError):
        VFE3Config(policy_mode="efe_one_step", policy_preference="flat", policy_horizon=2)
    VFE3Config(policy_mode="efe_one_step", policy_preference="flat", policy_horizon=1)   # the H=1 pairing


def test_config_rejects_policy_top_k_over_vocab():
    # audit F5: the candidate menu cannot be wider than the vocabulary (base_logits.topk(Kp) would raise
    # a cryptic torch index error). Enforced only once a scorer is on; inert on the default 'none' path.
    with pytest.raises(ValueError):
        VFE3Config(policy_mode="efe_one_step", policy_preference="flat", vocab_size=32, policy_top_k=64)
    VFE3Config(policy_mode="efe_one_step", policy_preference="flat", vocab_size=64, policy_top_k=8)
    VFE3Config(vocab_size=32, policy_top_k=64)                    # 'none' path: top_k>vocab is inert, allowed


def test_sigma_gate_flag_requires_passing_artifact(tmp_path):
    import json
    # no artifact reference -> structural failure
    with pytest.raises(ValueError):
        VFE3Config(policy_sigma_ambiguity_validated=True)
    # named but missing file -> content failure (cannot flip the flag without the record)
    with pytest.raises(ValueError):
        VFE3Config(policy_sigma_ambiguity_validated=True,
                   policy_sigma_gate_artifact=str(tmp_path / "absent.json"))
    # a FAIL record cannot silently validate the flag (Guard 4 content check)
    fail = tmp_path / "fail.json"
    fail.write_text(json.dumps({"status": "FAIL", "spec_commit": "x"}), encoding="utf-8")
    with pytest.raises(ValueError):
        VFE3Config(policy_sigma_ambiguity_validated=True, policy_sigma_gate_artifact=str(fail))
    # a PASS record is required for the flag to be set
    ok = tmp_path / "pass.json"
    ok.write_text(json.dumps({"status": "PASS", "spec_commit": "x"}), encoding="utf-8")
    cfg = VFE3Config(policy_sigma_ambiguity_validated=True, policy_sigma_gate_artifact=str(ok))
    assert cfg.policy_sigma_ambiguity_validated is True
