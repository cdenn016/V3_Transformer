from __future__ import annotations

import types

import pytest

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
            True,
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
