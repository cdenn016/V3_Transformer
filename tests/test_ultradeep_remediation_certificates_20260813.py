from __future__ import annotations

import types

import pytest

import ablation
from vfe3.geometry import norms as norms_module
from vfe3.geometry import transport as transport_module

from vfe3.contracts import ExecutableBuildMetadata
from vfe3.model.block_mlp import block_mlp_build_metadata
from vfe3.run_artifacts import _pure_path_report


def _cfg(**overrides: object) -> types.SimpleNamespace:
    values: dict[str, object] = {
        "include_attention_entropy": True,
        "transport_mode": "flat",
        "lambda_alpha_mode": "constant",
        "lambda_beta": 1.0,
        "use_prior_bank": True,
        "use_head_mixer": False,
        "head_mixer_compatibility": "disabled",
        "head_mixer_gauge_compatible": True,
        "use_priorbank_head_evidence_mixer": False,
        "encode_mode": "per_token",
        "use_block_mlp": False,
        "block_mlp_mode": "coordinate",
        "block_mlp_covariance": "passthrough",
        "block_mlp_expansion": 4,
        "block_mlp_covariance_floor": 1e-4,
        "block_mlp_activation": "gelu",
        "block_mlp_dropout": 0.0,
        "m_block_mlp_lr": None,
        "m_p_mu_lr": 1e-3,
        "precision_weighted_attention": False,
        "gauge_transport": "on",
        "pos_rotation": "none",
        "rope_full_gauge": False,
        "rope_on_value": True,
        "lambda_gamma": 0.0,
        "s_e_step": False,
        "skip_belief_sigma_update": False,
        "lambda_twohop": 0.0,
        "gauge_parameterization": "phi",
        "omega_reflection": "off",
        "phi_reflection": "off",
        "gauge_group": "glk",
        "family": "gaussian_full",
        "e_step_update": "gradient",
        "mm_damping": 1.0,
        "query_adaptive_tau": False,
        "query_tau_c": 1.0,
        "spd_retract_mode": "spd_affine",
        "sigma_max": None,
        "norm_type_block": "none",
        "norm_type_final": "none",
        "layernorm_affine": False,
        "m_phi_update_mode": "adamw",
        "m_phi_group_trust_radius": 0.1,
        "phi_mstep_max_matrix_norm": None,
        "transport_chart_max_norm": 12.0,
        "e_phi_lr": 0.0,
        "emission_mode": "off",
        "emission_weight": 0.0,
        "beta_attention_prior": "causal",
        "gamma_attention_prior": "causal",
        "t5_bidirectional": False,
        "n_heads": 1,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


def _block_build(*, enabled: bool, mode: str) -> ExecutableBuildMetadata:
    return ExecutableBuildMetadata(
        block_mlp=block_mlp_build_metadata(
            enabled=enabled,
            mode=mode,
            covariance="passthrough",
            expansion=4,
            activation="gelu",
            dropout=0.0,
            covariance_floor=1e-4,
            embed_dim=4,
            irrep_dims=(2, 2),
            n_layers=1,
        )
    )


@pytest.mark.parametrize(
    ("overrides", "history", "expected_gauge", "expected_exactness", "expected_theory"),
    [
        ({}, [], True, "not_applicable", True),
        ({"transport_mode": "regime_ii"}, [], False, "approximate", False),
        ({"transport_mode": "regime_ii_link"}, [], False, "approximate", False),
        ({"transport_mode": "regime_ii_link_charted"}, [], True, "unknown", False),
        ({"transport_mode": "regime_ii_covariant"}, [], True, "unknown", False),
        (
            {"transport_mode": "regime_ii_covariant"},
            [{"regime_ii_covariant_feature_exact": 1.0}],
            True,
            "exact",
            False,
        ),
        (
            {"transport_mode": "regime_ii_covariant"},
            [{"regime_ii_covariant_feature_exact": 0.0}],
            True,
            "approximate",
            False,
        ),
        (
            {"transport_mode": "regime_ii_covariant"},
            [{"regime_ii_covariant_feature_exact": 1.0}, {}],
            True,
            "unknown",
            False,
        ),
        (
            {"transport_mode": "regime_ii_covariant"},
            [{"regime_ii_covariant_feature_exact": 0.0}, {}],
            True,
            "approximate",
            False,
        ),
        ({"norm_type_block": "layernorm"}, [], False, "not_applicable", False),
        ({"norm_type_final": "layernorm"}, [], False, "not_applicable", False),
        ({"sigma_max": 10.0}, [], False, "not_applicable", False),
        ({"phi_mstep_max_matrix_norm": 5.0}, [], False, "not_applicable", False),
        ({"m_phi_update_mode": "pullback_group"}, [], False, "not_applicable", False),
        (
            {"query_adaptive_tau": True, "query_tau_c": 1.0},
            [],
            False,
            "not_applicable",
            False,
        ),
        (
            {"query_adaptive_tau": True, "query_tau_c": 0.0},
            [],
            True,
            "not_applicable",
            True,
        ),
    ],
)
def test_certificate_truth_table(
    overrides: dict[str, object],
    history: list[dict[str, object]],
    expected_gauge: bool,
    expected_exactness: str,
    expected_theory: bool,
) -> None:
    report = _pure_path_report(_cfg(**overrides), history)

    assert report["on_gauge_pure_path"] is expected_gauge
    assert report["on_causal_lm_path"] is True
    assert report["transport_exactness_status"] == expected_exactness
    assert report["on_theory_pure_path"] is expected_theory


def test_block_mlp_gauge_certificate_uses_immutable_executable_metadata() -> None:
    cfg = _cfg(use_block_mlp=False, block_mlp_mode="gauge_gate")
    report = _pure_path_report(
        cfg,
        [],
        executable_build=_block_build(enabled=True, mode="coordinate"),
    )

    assert report["gauge_flags"]["block_mlp_intertwiner_compatible"] is False
    assert report["on_gauge_pure_path"] is False


@pytest.mark.parametrize(
    ("beta_prior", "gamma_prior", "t5_bidirectional", "expected"),
    [
        ("causal", "causal_alibi_noself", False, True),
        ("uniform", "causal", False, False),
        ("causal", "uniform", False, False),
        ("windowed", "causal", False, False),
        ("t5_relative_bias", "causal", False, True),
        ("t5_relative_bias", "causal", True, False),
    ],
)
def test_causality_comes_from_both_active_prior_registrations(
    beta_prior: str,
    gamma_prior: str,
    t5_bidirectional: bool,
    expected: bool,
) -> None:
    report = _pure_path_report(
        _cfg(
            beta_attention_prior=beta_prior,
            gamma_attention_prior=gamma_prior,
            t5_bidirectional=t5_bidirectional,
        ),
        [],
    )

    assert report["on_causal_lm_path"] is expected
    assert report["on_gauge_pure_path"] is True
    assert report["on_theory_pure_path"] is expected


def test_block_gl_reflection_reports_only_block_zero_as_accessible() -> None:
    report = _pure_path_report(
        _cfg(
            gauge_group="block_glk",
            n_heads=3,
            phi_reflection="init_seed",
        ),
        [],
        reflection_scope="block_0_probe",
        reflection_group_component_count=3,
    )

    reflection = report["reflection"]
    assert reflection["effective_scope"] == "block_zero_only"
    assert reflection["effective_subgroup"] == "GL(d_0) x product_{h>0} GL+(d_h)"
    assert reflection["accessible_blocks"] == [0]
    assert reflection["accessible_component_count"] == 2
    assert reflection["total_component_count"] == 8
    assert report["on_gauge_pure_path"] is False


def test_certificate_schema_is_additive() -> None:
    report = _pure_path_report(_cfg(), [])
    preexisting = {
        "on_pure_path",
        "pure_flags",
        "gauge_flags",
        "on_gauge_pure_path",
        "config_toggles",
        "converged_stress",
    }
    additions = {
        "on_causal_lm_path",
        "transport_exactness_status",
        "on_theory_pure_path",
        "causal_flags",
        "theory_flags",
        "reflection",
    }

    assert preexisting <= report.keys()
    assert additions <= report.keys()

_GAUGE_KEYS = {
    "learned_gauge_transport",
    "transport_gauge_equivariant",
    "no_positional_rotation",
    "no_model_channel_coupling",
    "phi_parameterization",
    "no_reflection_sampling",
    "family_group_invariant",
    "head_mixer_intertwiner_compatible",
    "block_mlp_intertwiner_compatible",
    "block_norm_gauge_equivariant",
    "final_norm_gauge_equivariant",
    "no_fixed_coordinate_spectral_cap",
    "no_phi_mstep_chart_cap",
    "no_pullback_trust_region",
    "no_transport_exponential_clipping",
    "no_e_step_phi_retraction_clipping",
    "no_query_adaptive_trace_temperature",
    "no_fixed_basis_emission",
    "no_additive_encoder_control",
}


def test_gauge_truth_table_passing_fixture_covers_every_enumerated_key() -> None:
    report = _pure_path_report(_cfg(), [])

    assert set(report["gauge_flags"]) == _GAUGE_KEYS
    assert all(report["gauge_flags"].values())
    assert report["on_gauge_pure_path"] is True


@pytest.mark.parametrize(
    ("failed_key", "overrides", "block_mode"),
    [
        ("learned_gauge_transport", {"gauge_transport": "off"}, None),
        ("transport_gauge_equivariant", {"transport_mode": "regime_ii"}, None),
        ("no_positional_rotation", {"pos_rotation": "rope"}, None),
        ("no_model_channel_coupling", {"lambda_gamma": 0.1}, None),
        ("phi_parameterization", {"gauge_parameterization": "omega_direct"}, None),
        ("no_reflection_sampling", {"phi_reflection": "init_seed"}, None),
        ("family_group_invariant", {"family": "gaussian_diagonal"}, None),
        (
            "head_mixer_intertwiner_compatible",
            {
                "use_head_mixer": True,
                "gauge_group": "block_glk",
                "head_mixer_compatibility": "independent_head_nonintertwiner",
                "head_mixer_gauge_compatible": False,
            },
            None,
        ),
        ("block_mlp_intertwiner_compatible", {}, "coordinate"),
        ("block_norm_gauge_equivariant", {"norm_type_block": "layernorm"}, None),
        ("final_norm_gauge_equivariant", {"norm_type_final": "layernorm"}, None),
        ("no_fixed_coordinate_spectral_cap", {"sigma_max": 10.0}, None),
        ("no_phi_mstep_chart_cap", {"phi_mstep_max_matrix_norm": 5.0}, None),
        ("no_pullback_trust_region", {"m_phi_update_mode": "pullback_group"}, None),
        ("no_transport_exponential_clipping", {"transport_chart_max_norm": None}, None),
        ("no_transport_exponential_clipping", {"transport_chart_max_norm": 21.0}, None),
        ("no_e_step_phi_retraction_clipping", {"e_phi_lr": 0.1}, None),
        (
            "no_query_adaptive_trace_temperature",
            {"query_adaptive_tau": True, "query_tau_c": 1.0},
            None,
        ),
        (
            "no_fixed_basis_emission",
            {"emission_mode": "separate", "emission_weight": 0.5},
            None,
        ),
        ("no_additive_encoder_control", {"encode_mode": "per_token_additive"}, None),
    ],
)
def test_each_active_gauge_breaker_fails_exactly_its_registered_key(
    failed_key: str,
    overrides: dict[str, object],
    block_mode: str | None,
) -> None:
    executable_build = (
        _block_build(enabled=True, mode=block_mode) if block_mode is not None else None
    )
    report = _pure_path_report(
        _cfg(**overrides),
        [],
        executable_build=executable_build,
    )

    failed = {name for name, enabled in report["gauge_flags"].items() if not enabled}
    assert failed == {failed_key}
    assert report["on_gauge_pure_path"] is False


_THEORY_REQUIREMENT_CASES = [
    ("canonical_attention_entropy", {"include_attention_entropy": False}, [], None),
    (
        "flat_transport",
        {"transport_mode": "regime_ii_covariant"},
        [{"regime_ii_covariant_feature_exact": 1.0}],
        None,
    ),
    ("constant_lambda_alpha", {"lambda_alpha_mode": "state_dependent"}, [], None),
    ("prior_bank_decode", {"use_prior_bank": False}, [], None),
    ("no_head_mixer", {"use_head_mixer": True}, [], None),
    (
        "no_priorbank_head_evidence_mixer",
        {"use_priorbank_head_evidence_mixer": True},
        [],
        None,
    ),
    (
        "unweighted_attention+no_fixed_prior_surrogate",
        {"precision_weighted_attention": True},
        [],
        None,
    ),
    ("full_sigma_update", {"skip_belief_sigma_update": True}, [], None),
    ("no_twohop_coupling", {"lambda_twohop": 0.1}, [], None),
    ("no_block_mlp", {"use_block_mlp": True, "block_mlp_mode": "gauge_gate"}, [], None),
    ("gradient_e_step_update", {"e_step_update": "mm_exact"}, [], None),
]


@pytest.mark.parametrize(
    ("case", "overrides", "history", "expected_false"),
    [
        (
            case,
            overrides,
            history,
            {"unweighted_attention", "no_fixed_prior_surrogate"}
            if case == "unweighted_attention+no_fixed_prior_surrogate"
            else {case},
        )
        for case, overrides, history, _unused in _THEORY_REQUIREMENT_CASES
    ],
)
def test_theory_purity_transparently_fails_each_legacy_requirement(
    case: str,
    overrides: dict[str, object],
    history: list[dict[str, object]],
    expected_false: set[str],
) -> None:
    del case
    report = _pure_path_report(_cfg(**overrides), history)

    assert {name for name, value in report["pure_flags"].items() if not value} == expected_false
    assert {name for name, value in report["theory_flags"].items() if not value} == expected_false
    assert report["on_gauge_pure_path"] is True
    assert report["on_causal_lm_path"] is True
    assert report["transport_exactness_status"] in {"exact", "not_applicable"}
    assert report["on_theory_pure_path"] is False


def test_theory_flags_enumerate_all_legacy_requirements_and_new_facets() -> None:
    report = _pure_path_report(_cfg(), [])

    assert report["pure_flags"].keys() <= report["theory_flags"].keys()
    assert {
        name: report["theory_flags"][name]
        for name in report["pure_flags"]
    } == report["pure_flags"]
    assert set(report["theory_flags"]) == {
        *report["pure_flags"],
        "gauge_pure_path",
        "causal_lm_path",
        "transport_exact_when_applicable",
    }
    assert report["on_theory_pure_path"] is True


@pytest.mark.parametrize(
    ("history", "expected"),
    [
        ([{"regime_ii_covariant_feature_exact": 1.0}], "exact"),
        ([{"regime_ii_covariant_feature_exact": 1.0}, {}], "unknown"),
        ([{"regime_ii_covariant_feature_exact": 0.0}, {}], "approximate"),
    ],
)
def test_successful_ablation_reporting_uses_completed_runtime_history(
    history: list[dict[str, object]],
    expected: str,
) -> None:
    fields = ablation._gauge_reporting_fields(
        _cfg(transport_mode="regime_ii_covariant"),
        history=history,
    )

    assert fields["transport_exactness_status"] == expected


def _install_transport_registration(
    name: str,
    registration: transport_module.TransportRegistration,
    *,
    gauge_equivariant: bool,
) -> None:
    transport_module.register_transport(
        name,
        covariance_class=registration.covariance_class,
        needs_mu=registration.needs_mu,
        needs_sigma=registration.needs_sigma,
        batch_independent=registration.batch_independent,
        pair_transport_kind=registration.pair_transport_kind,
        rope_right_foldable=registration.rope_right_foldable,
        state_builder=registration.state_builder,
        serialization_keys=registration.serialization_keys,
        offdiag_serialization_keys=registration.offdiag_serialization_keys,
        gauge_equivariant=gauge_equivariant,
        runtime_exactness_key=registration.runtime_exactness_key,
        override=True,
    )(registration.callable)


def test_gauge_certificate_tracks_transport_registration_override_and_restore() -> None:
    original = transport_module.get_transport_registration("flat")
    try:
        _install_transport_registration("flat", original, gauge_equivariant=False)
        report = _pure_path_report(_cfg(), [])
        failed = {name for name, value in report["gauge_flags"].items() if not value}
        assert failed == {"transport_gauge_equivariant"}
        assert report["on_gauge_pure_path"] is False
    finally:
        _install_transport_registration(
            "flat", original, gauge_equivariant=original.gauge_equivariant
        )
    assert transport_module.get_transport_registration("flat") == original


def test_gauge_certificate_tracks_active_norm_callable_and_restore() -> None:
    original = norms_module._NORMS["none"]

    def replacement(*args, **kwargs):
        del args, kwargs
        raise AssertionError("reporting must not construct the norm")

    norms_module._NORMS["none"] = replacement
    try:
        report = _pure_path_report(_cfg(norm_type_final="mahalanobis"), [])
        failed = {name for name, value in report["gauge_flags"].items() if not value}
        assert failed == {"block_norm_gauge_equivariant"}
        assert report["on_gauge_pure_path"] is False
    finally:
        norms_module._NORMS["none"] = original

    assert norms_module.get_norm_registration("none").gauge_equivariant is True

@pytest.mark.parametrize(
    ("overrides", "history", "expected_failed_theory_keys"),
    [
        ({"norm_type_block": "layernorm"}, [], {"gauge_pure_path"}),
        ({"beta_attention_prior": "uniform"}, [], {"causal_lm_path"}),
        (
            {"transport_mode": "regime_ii_covariant"},
            [{"regime_ii_covariant_feature_exact": 1.0}, {}],
            {"flat_transport", "transport_exact_when_applicable"},
        ),
    ],
)
def test_theory_facets_fail_transparently_and_independently(
    overrides: dict[str, object],
    history: list[dict[str, object]],
    expected_failed_theory_keys: set[str],
) -> None:
    report = _pure_path_report(_cfg(**overrides), history)

    failed = {name for name, value in report["theory_flags"].items() if not value}
    assert failed == expected_failed_theory_keys
    assert report["on_theory_pure_path"] is False