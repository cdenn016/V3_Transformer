"""Regression coverage for the B5 finite objective/update-control contract."""

import pytest

from vfe3.config import VFE3Config


NONNEGATIVE_CONTROLS = (
    "mass_phi",
    "mstep_self_coupling_weight",
    "lambda_beta",
    "lambda_h",
    "lambda_gamma",
    "e_q_mu_lr",
    "e_q_sigma_lr",
    "e_phi_lr",
    "e_s_mu_lr",
    "e_s_sigma_lr",
    "lambda_twohop",
)

NONFINITE_VALUES = (
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="positive-infinity"),
    pytest.param(float("-inf"), id="negative-infinity"),
)


@pytest.mark.parametrize("name", NONNEGATIVE_CONTROLS)
@pytest.mark.parametrize("value", NONFINITE_VALUES)
def test_b5_nonnegative_controls_reject_every_nonfinite_value(
    name:  str,
    value: float,
) -> None:
    with pytest.raises(ValueError, match=name):
        VFE3Config(**{name: value})


@pytest.mark.parametrize("name", NONNEGATIVE_CONTROLS)
def test_b5_nonnegative_controls_continue_to_accept_zero(name: str) -> None:
    cfg = VFE3Config(**{name: 0.0})

    assert getattr(cfg, name) == 0.0


@pytest.mark.parametrize("gamma_as_beta_prior", [False, True])
@pytest.mark.parametrize("value", NONFINITE_VALUES)
def test_gamma_prior_weight_rejects_nonfinite_values_on_both_routes(
    value:               float,
    gamma_as_beta_prior: bool,
) -> None:
    with pytest.raises(ValueError, match="gamma_prior_weight"):
        VFE3Config(
            lambda_gamma=1.0 if gamma_as_beta_prior else 0.0,
            gamma_as_beta_prior=gamma_as_beta_prior,
            gamma_prior_weight=value,
        )


@pytest.mark.parametrize("gamma_as_beta_prior", [False, True])
@pytest.mark.parametrize("value", [0.0, 1.0])
def test_gamma_prior_weight_keeps_closed_interval_boundaries_valid(
    value:               float,
    gamma_as_beta_prior: bool,
) -> None:
    cfg = VFE3Config(
        lambda_gamma=1.0 if gamma_as_beta_prior else 0.0,
        gamma_as_beta_prior=gamma_as_beta_prior,
        gamma_prior_weight=value,
    )

    assert cfg.gamma_prior_weight == value
