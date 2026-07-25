"""Runnable E-step depth and phi-control ablation registry regressions."""

from ablation import BASELINE_CONFIG, SWEEPS, make_run_overrides, validate_sweeps
from vfe3.config import VFE3Config


EXPECTED_LABELS = {
    "estep_depth_damping": [
        "fixed_T1_eta1.00",
        "fixed_T3_eta1.00",
        "fixed_T5_eta1.00",
        "fixed_T5_eta0.75",
        "random_T1-5_evalT5_eta1.00",
        "random_T1-5_evalT5_eta0.75",
    ],
    "phi_chart_control": [
        "adamw_unbounded",
        "adamw_mass0.01",
        "adamw_lr0.003",
        "pullback_group_lr0.0015",
        "adamw_projected_norm5",
    ],
    "pos_phi_composition": ["bch", "group_product", "none"],
}


def test_recommended_estep_and_phi_sweeps_are_registered_and_runnable() -> None:
    names = list(EXPECTED_LABELS)
    validate_sweeps(names)

    for name, expected in EXPECTED_LABELS.items():
        runs = make_run_overrides(name)
        assert [label for label, _ in runs] == expected
        assert SWEEPS[name].get("collect_diagnostics") is True
        for _, overrides in runs:
            VFE3Config(**{**BASELINE_CONFIG, **overrides})


def test_recommended_sweep_activation_matches_click_run_order() -> None:
    from ablation import SWEEPS, SWEEP_ORDER

    # Assert the INVARIANT, not the literal selection (audit 2026-07-25 F19). SWEEP_ORDER is the
    # owner's click-to-run choice of which sweeps to activate and changes constantly, so pinning its
    # contents made this test report a config edit as a regression. What must hold is that every
    # activated name resolves to a registered sweep and that the order names no duplicates.
    assert SWEEP_ORDER, "no sweeps activated"
    unknown = [name for name in SWEEP_ORDER if name not in SWEEPS]
    assert not unknown, f"SWEEP_ORDER names unregistered sweeps: {unknown}"
    assert len(set(SWEEP_ORDER)) == len(SWEEP_ORDER), "SWEEP_ORDER repeats a sweep"
