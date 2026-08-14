"""Task 6 reporting/accounting regressions (audit 2026-08-13)."""
from __future__ import annotations

import math

import pytest
import torch


def test_perplexity_is_exact_and_overflow_is_infinite() -> None:
    from vfe3.train import _perplexity_from_ce

    assert _perplexity_from_ce(25.0) == math.exp(25.0)
    assert _perplexity_from_ce(1_000.0) == float("inf")


def test_target_accounting_requires_exact_partition() -> None:
    from vfe3.train import _target_accounting

    assert _target_accounting(9, 6, 3) == {
        "expected_targets": 9,
        "scored_targets": 6,
        "excluded_targets": 3,
    }
    with pytest.raises(RuntimeError, match=r"expected_targets == scored_targets \+ excluded_targets"):
        _target_accounting(9, 6, 2)


def test_train_step_device_counts_are_propagated() -> None:
    """Task 5 decoder counts must survive Task 6's aggregation seam."""
    from vfe3.train import _target_accounting_from_device

    counts = _target_accounting_from_device(
        torch.tensor(7, dtype=torch.int64),
        torch.tensor(5, dtype=torch.int64),
        torch.tensor(2, dtype=torch.int64),
    )
    assert counts == {"expected_targets": 7, "scored_targets": 5, "excluded_targets": 2}


def test_final_iterate_has_no_unsupported_convergence_claim() -> None:
    from vfe3.run_artifacts import _estep_iterate_evidence

    record = _estep_iterate_evidence(
        free_energy=[3.0],
        halted=False,
        fixed_point_residual=None,
        fixed_point_tolerance=None,
    )
    assert record == {
        "iterate_label": "final_iterate",
        "descent_evidence": False,
        "convergence_evidence": False,
        "fixed_point_evidence": False,
    }


def test_target_blindness_has_no_correlation_sign_expectation() -> None:
    from vfe3.run_artifacts import _target_blind_objective_interpretation

    text = _target_blind_objective_interpretation()
    assert "separate" in text.lower()
    assert "positive" not in text.lower()
    assert "negative" not in text.lower()
    assert "correlation" not in text.lower()


def test_registered_divergence_label_is_not_hardcoded_kl(monkeypatch) -> None:
    from vfe3.viz import figures

    class Functional:
        diagnostic_label = "registered alpha objective"
        diagnostic_units = "registered-units"

    monkeypatch.setattr(figures, "get_functional", lambda _name: Functional())
    assert figures._registered_divergence_axis("non_kl") == (
        "registered alpha objective",
        "registered-units",
    )


def test_data_seed_none_is_nonshared_unspecified() -> None:
    import multiseed_analysis as multiseed

    status = multiseed._data_order_status([None, None])
    assert status["shared"] is False
    assert status["status"] == "nonshared_unspecified"


def test_scheduler_metadata_matches_one_percent_floor() -> None:
    from vfe3.run_artifacts import _scheduler_metadata

    assert _scheduler_metadata(min_lr=0.0, min_lr_frac=0.01) == {
        "kind": "warmup_half_cosine_with_floor",
        "absolute_floor": 0.0,
        "fractional_floor": 0.01,
        "floor_description": "max(absolute_floor, fractional_floor * group_base_lr)",
    }


def test_diagnostic_policy_fields_are_complete() -> None:
    from vfe3.model.model import _numerical_policy_diagnostics

    assert _numerical_policy_diagnostics(
        m_phi_group_trust_radius=0.2,
        phi_mstep_max_matrix_norm=0.3,
        transport_chart_max_norm=8.0,
        exp_fp64_norm_threshold=4.0,
    ) == {
        "m_phi_group_trust_radius": 0.2,
        "phi_mstep_max_matrix_norm": 0.3,
        "transport_chart_max_norm": 8.0,
        "exp_fp64_norm_threshold": 4.0,
    }


def test_state_specific_free_energy_totals_do_not_mix_components() -> None:
    from vfe3.model.model import _state_specific_free_energy_diagnostics

    pre = {"self_coupling": 1.0, "belief_coupling": 2.0, "attention_entropy": 3.0}
    post = {"self_coupling": 10.0, "belief_coupling": 20.0, "attention_entropy": 30.0}
    out = _state_specific_free_energy_diagnostics(pre, post)
    assert out["pre_block_mlp_total"] == 6.0
    assert out["post_block_mlp_total"] == 60.0
    assert out["total"] == out["post_block_mlp_total"]
    with pytest.raises(RuntimeError, match="separate component evaluations"):
        _state_specific_free_energy_diagnostics(pre, pre)
