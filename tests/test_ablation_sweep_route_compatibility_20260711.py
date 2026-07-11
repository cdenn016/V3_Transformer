import pytest

import ablation
import train_vfe3
from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel


def test_active_update_rule_config_values_are_preserved() -> None:
    assert ablation.BASELINE_CONFIG["e_step_update"] == "mm_exact"
    assert ablation.BASELINE_CONFIG["mm_damping"] == 0.75
    assert train_vfe3.config["e_step_update"] == "mm_exact"
    assert train_vfe3.config["mm_damping"] == 0.75


@pytest.mark.parametrize(
    "sweep_name",
    ["attention_entropy", "gauge_equivariance", "pos_extrapolation", "regime_ii"],
)
def test_route_constrained_sweep_arms_use_local_gradient_contract(sweep_name: str) -> None:
    runs = ablation.make_run_overrides(sweep_name)
    assert runs
    for label, overrides in runs:
        assert overrides.get("e_step_update") == "gradient", (
            f"{sweep_name}/{label} must select the controlled gradient update locally"
        )
        cfg_dict = ablation._cell_cfg_dict(
            {**overrides, "vocab_size": 64, "max_seq_len": 16},
            seed=0,
            max_steps=1,
        )
        assert VFEModel(VFE3Config(**cfg_dict)) is not None
