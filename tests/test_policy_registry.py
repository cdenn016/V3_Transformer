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


def test_register_policy_duplicate_key_fails_closed():
    # audit F12 (2026-07-01): a second @register_policy under an existing name must fail closed
    # (KeyError) rather than silently shadowing the first; override=True is the explicit escape.
    name = "__f12_dup_smoke__"
    try:
        @register_policy(name)
        def _first():
            return 1
        with pytest.raises(KeyError):
            @register_policy(name)
            def _second():
                return 2
        assert get_policy(name)() == 1                      # the first registration survived

        @register_policy(name, override=True)
        def _third():
            return 3
        assert get_policy(name)() == 3                      # explicit override replaces it
    finally:
        _POLICIES.pop(name, None)
    assert name not in _POLICIES


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


def test_logprob_control_rejects_ignored_config_fields():
    VFE3Config(policy_mode="logprob_control", policy_preference="flat")
    with pytest.raises(ValueError, match="logprob_control.*policy_horizon"):
        VFE3Config(policy_mode="logprob_control", policy_preference="flat", policy_horizon=2)
    for score_terms in (("risk",), ("ambiguity", "risk")):
        with pytest.raises(ValueError, match="logprob_control.*policy_score_terms"):
            VFE3Config(policy_mode="logprob_control", policy_preference="flat",
                       policy_score_terms=score_terms)


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


@pytest.mark.parametrize("bad_precision", [float("nan"), float("inf"), float("-inf")])
def test_config_rejects_nonfinite_policy_precision(bad_precision):
    with pytest.raises(ValueError, match="policy_precision"):
        VFE3Config(policy_precision=bad_precision)


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


def test_config_defaults_ambiguity_mode_and_samples():
    cfg = VFE3Config()
    assert cfg.policy_ambiguity_mode == "likelihood_entropy"
    assert cfg.policy_sigma_mc_samples == 16


def test_config_rejects_unknown_ambiguity_mode():
    with pytest.raises(ValueError, match="policy_ambiguity_mode"):
        VFE3Config(policy_ambiguity_mode="not_a_registered_ambiguity")


def _pass_artifact(tmp_path):
    import json
    from vfe3.inference.sigma_gate import sigma_gate_spec_identity
    ok = tmp_path / "pass.json"
    ok.write_text(json.dumps({"status": "PASS", "spec_commit": sigma_gate_spec_identity()}),
                  encoding="utf-8")
    return str(ok)


def test_config_sigma_mc_requires_all_preconditions(tmp_path):
    art = _pass_artifact(tmp_path)
    # sigma_mc needs an EFE scorer, gaussian family, the validated flag, an artifact, and 16 samples.
    with pytest.raises(ValueError):                               # policy_mode not an EFE scorer
        VFE3Config(policy_ambiguity_mode="sigma_mc", policy_mode="logprob_control",
                   policy_preference="flat")
    with pytest.raises(ValueError):                               # missing the validated flag
        VFE3Config(policy_ambiguity_mode="sigma_mc", policy_mode="efe_one_step",
                   policy_preference="flat", family="gaussian_diagonal")
    with pytest.raises(ValueError):                               # validated flag but no artifact
        VFE3Config(policy_ambiguity_mode="sigma_mc", policy_mode="efe_one_step",
                   policy_preference="flat", family="gaussian_diagonal",
                   policy_sigma_ambiguity_validated=True)


def test_config_sigma_mc_rejects_non_16_sample_count(tmp_path):
    art = _pass_artifact(tmp_path)
    for bad in (1, 8, 15, 17, 32):
        with pytest.raises(ValueError):
            VFE3Config(policy_ambiguity_mode="sigma_mc", policy_mode="efe_one_step",
                       policy_preference="flat", family="gaussian_diagonal",
                       policy_sigma_ambiguity_validated=True, policy_sigma_gate_artifact=art,
                       policy_sigma_mc_samples=bad)


def test_config_sigma_mc_rejected_under_production_fail_manifest(tmp_path):
    # All structural preconditions satisfied, but the production spec identity is registered FAIL, so a
    # sigma_mc config cannot be constructed against the shipped preregistry (fail-closed).
    art = _pass_artifact(tmp_path)
    with pytest.raises(ValueError):
        VFE3Config(policy_ambiguity_mode="sigma_mc", policy_mode="efe_one_step",
                   policy_preference="flat", family="gaussian_diagonal",
                   policy_sigma_ambiguity_validated=True, policy_sigma_gate_artifact=art,
                   policy_sigma_mc_samples=16)


def test_config_validated_flag_does_not_turn_on_sigma_mc(tmp_path):
    # policy_sigma_ambiguity_validated=True under likelihood_entropy is still allowed and inert; the
    # flag must never enable sigma_mc by itself.
    art = _pass_artifact(tmp_path)
    cfg = VFE3Config(policy_sigma_ambiguity_validated=True, policy_sigma_gate_artifact=art)
    assert cfg.policy_ambiguity_mode == "likelihood_entropy"


def test_sigma_gate_flag_has_no_executable_consumer(tmp_path):
    # audit F5 (2026-07-01), updated for PB-06: policy_sigma_ambiguity_validated=True (with a PASS
    # artifact) is a PRECONDITION RECORD ONLY -- routing to 'sigma_mc' exists now but ONLY via
    # policy_ambiguity_mode='sigma_mc' (validated + consumer-gated), which stays at its default
    # 'likelihood_entropy' here, so the scorer runs fine (no sigma_mc RuntimeError). This pins that
    # the validated flag does NOT turn the gated estimator on by itself.
    import json
    import torch
    from vfe3.model.model import VFEModel
    ok = tmp_path / "pass.json"
    ok.write_text(json.dumps({"status": "PASS", "spec_commit": "x"}), encoding="utf-8")
    torch.manual_seed(0)
    m = VFEModel(VFE3Config(
        vocab_size=16, embed_dim=8, n_heads=2, max_seq_len=16,
        policy_mode="efe_one_step", policy_preference="flat",
        policy_sigma_ambiguity_validated=True, policy_sigma_gate_artifact=str(ok)))
    V = m.cfg.vocab_size
    ctx = torch.randint(0, V, (1, 5))
    cand = torch.randint(0, V, (1, 4, 1))
    pref = get_preference("flat")(m.prior_bank)
    with torch.no_grad():
        out = get_policy("efe_one_step")(ctx, cand, pref, m)   # default ambiguity_mode
    assert torch.isfinite(out.score).all()                     # scored, no sigma_mc dispatch
    # the ambiguity IS the likelihood_entropy value (== predictive entropy at v1), not a sigma term
    from vfe3.inference.policy import _rollout_predictive
    with torch.no_grad():
        q_log, _ = _rollout_predictive(ctx, cand, m)
    est = get_ambiguity("likelihood_entropy")(q_log)           # AmbiguityEstimate (PB-06)
    assert torch.allclose(out.ambiguity, est.expected_conditional_entropy, atol=1e-6)
    # and the generic generate() path also completes under the validated flag
    with torch.no_grad():
        seq = m.generate(ctx, 2, greedy=True)
    assert seq.shape == (1, 7)


# ======================================================================================
# Task 5 (PB-05/PB-06): the click-to-run driver exposes only the validated policy fields.
# ======================================================================================

def test_generate_efe_exposes_the_four_sigma_fields_and_no_identity_overrides():
    # audit PB-05/06: _POLICY_FIELDS is the exact set of policy knobs generate_efe.py's CONFIG can
    # override. The four sigma-gate fields must be present; the four consumer-derived identities
    # (model_behavior_sha256, spec_identity, code_identity_sha256, measurement_context_sha256) must
    # NOT be, since VFEModel.generate derives them itself from the live model/source/corpus.
    import generate_efe
    assert set(generate_efe._POLICY_FIELDS) == {
        "policy_mode", "policy_preference", "policy_score_terms", "policy_top_k",
        "policy_precision", "policy_horizon", "policy_ambiguity_mode", "policy_sigma_mc_samples",
        "policy_sigma_ambiguity_validated", "policy_sigma_gate_artifact",
    }
    identity_fields = {"checkpoint", "model_behavior_sha256", "spec_identity",
                       "code_identity_sha256", "measurement_context_sha256"}
    assert not identity_fields & set(generate_efe._POLICY_FIELDS)


def test_generate_efe_driver_rejects_sigma_mc_override_before_generation(tmp_path, monkeypatch):
    # Task 5: a user who points the newly exposed sigma fields at a structurally valid PASS artifact
    # must still be rejected BEFORE any model is built or weights are loaded, because the shipped
    # preregistry resolves the live spec identity to FAIL (mirrors
    # test_config_sigma_mc_rejected_under_production_fail_manifest, but through the actual driver entry
    # point generate_efe._build_model rather than VFE3Config directly).
    import generate_efe

    art = _pass_artifact(tmp_path)
    config_dict = dict(family="gaussian_diagonal")               # simulates the checkpoint's embedded config
    overrides = {key: value for key, value in dict(
        policy_mode="efe_one_step", policy_preference="flat",
        policy_score_terms=("risk", "ambiguity"), policy_top_k=8, policy_precision=1.0,
        policy_horizon=1, policy_ambiguity_mode="sigma_mc", policy_sigma_mc_samples=16,
        policy_sigma_ambiguity_validated=True, policy_sigma_gate_artifact=art,
    ).items() if key in generate_efe._POLICY_FIELDS}
    assert overrides["policy_ambiguity_mode"] == "sigma_mc"       # the extended field actually flows through

    def _boom(*args, **kwargs):
        raise AssertionError("VFEModel must not be constructed when the sigma_mc gate rejects the config")
    monkeypatch.setattr(generate_efe, "VFEModel", _boom)

    with pytest.raises(ValueError, match="not registered as PASS"):
        generate_efe._build_model(config_dict, {}, policy_overrides=overrides, device="cpu")
