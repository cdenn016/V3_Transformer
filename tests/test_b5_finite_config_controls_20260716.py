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
    "mu_init_std",
    "phi_scale",
    "m_p_mu_lr",
    "m_p_sigma_lr",
    "m_phi_lr",
    "m_s_phi_lr",
    "weight_decay",
    "phi_weight_decay",
    "min_lr",
    "min_lr_frac",
    "connection_weight_decay",
    "sigma_weight_decay",
)

NONFINITE_VALUES = (
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="positive-infinity"),
    pytest.param(float("-inf"), id="negative-infinity"),
)

POSITIVE_FINITE_CONTROLS = (
    "kappa_beta",
    "kappa_gamma",
    "sigma_init",
    "e_mu_q_trust",
    "decode_tau",
    "exp_fp64_norm_threshold",
    "e_step_halt_tol",
)

NONNEGATIVE_FINITE_CONTROLS = (
    "unigram_kappa",
    "z_loss_weight",
)


@pytest.mark.parametrize("value", NONFINITE_VALUES)
def test_eps_rejects_nonfinite_values_on_uncapped_airm_path(value: float) -> None:
    with pytest.raises(ValueError, match="eps"):
        VFE3Config(eps=value, sigma_max=None)


@pytest.mark.parametrize("name", POSITIVE_FINITE_CONTROLS)
@pytest.mark.parametrize("value", NONFINITE_VALUES)
def test_active_positive_controls_reject_every_nonfinite_value(
    name:  str,
    value: float,
) -> None:
    with pytest.raises(ValueError, match=name):
        VFE3Config(**{name: value})


@pytest.mark.parametrize("name", ("kappa_beta", "kappa_gamma"))
@pytest.mark.parametrize("value", NONFINITE_VALUES)
def test_per_head_kappa_rejects_every_nonfinite_entry(name: str, value: float) -> None:
    with pytest.raises(ValueError, match=name):
        VFE3Config(
            embed_dim=4,
            n_heads=2,
            gauge_group="block_glk",
            **{name: [1.0, value]},
        )


@pytest.mark.parametrize("value", NONFINITE_VALUES)
def test_active_precision_attention_offset_rejects_every_nonfinite_value(value: float) -> None:
    with pytest.raises(ValueError, match="precision_attention_b0"):
        VFE3Config(precision_weighted_attention=True, precision_attention_b0=value)


@pytest.mark.parametrize("name", NONNEGATIVE_FINITE_CONTROLS)
@pytest.mark.parametrize("value", NONFINITE_VALUES)
def test_adjacent_nonnegative_controls_reject_every_nonfinite_value(
    name:  str,
    value: float,
) -> None:
    with pytest.raises(ValueError, match=name):
        VFE3Config(**{name: value})


@pytest.mark.parametrize("name", ("b0", "c0", "b0_h", "c0_h"))
@pytest.mark.parametrize("value", NONFINITE_VALUES)
def test_state_dependent_envelope_scalars_reject_every_nonfinite_value(
    name:  str,
    value: float,
) -> None:
    with pytest.raises(ValueError, match=name):
        VFE3Config(**{name: value})


@pytest.mark.parametrize("name", ("b0", "c0", "b0_h", "c0_h"))
@pytest.mark.parametrize("value", NONFINITE_VALUES)
def test_state_dependent_envelope_lists_reject_every_nonfinite_entry(
    name:  str,
    value: float,
) -> None:
    kwargs = {
        "embed_dim": 4,
        "n_heads": 1,
        name: [1.0, value, 1.0, 1.0],
    }
    if name in {"b0", "c0"}:
        kwargs["lambda_alpha_mode"] = "state_dependent_per_coord"
    else:
        kwargs["lambda_h_mode"] = "state_dependent_per_coord"

    with pytest.raises(ValueError, match=name):
        VFE3Config(**kwargs)


@pytest.mark.parametrize("name", ("pos_phi_scale", "m_phi_group_trust_radius"))
@pytest.mark.parametrize("value", NONFINITE_VALUES)
def test_active_frame_controls_reject_every_nonfinite_value(
    name:  str,
    value: float,
) -> None:
    with pytest.raises(ValueError, match=name):
        VFE3Config(**{name: value})


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
