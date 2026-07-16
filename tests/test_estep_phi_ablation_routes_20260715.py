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
        "pullback_natgrad_lr0.0015",
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


def test_recommended_sweeps_remain_opt_in() -> None:
    from ablation import SWEEP_ORDER

    assert set(EXPECTED_LABELS).isdisjoint(SWEEP_ORDER)
