"""Focused tests for the 2026-06-07 config-audit fixes (vfe3/config.py).

Covers the config-only audit findings: the config-level freeze-warning (learnable params
under straight_through/detach; pos_phi='learned' is the DEFAULT and is warned at the MODEL
level instead), the new unroll+oracle detached-tangent freeze-warning, that amp_dtype='fp16'
is accepted for forward (fp16 TRAINING / GradScaler is a documented buildout), the
m_phi_natural_grad footgun warning, the new close_basis field, and that the DEFAULT config
constructs silently.

These tests touch ONLY the config dataclass (no model build), so they are device-agnostic
and do not depend on modules under concurrent edit. The construction-time validator does a
local ``from vfe3.model.prior_bank import _DECODERS, _ENCODERS`` (edit 5), so an import error
there would surface here; that is the only cross-module coupling.
"""

import warnings

import pytest

from vfe3.config import VFE3Config


def test_amp_dtype_fp16_accepted_for_forward() -> None:
    """amp_dtype='fp16' is accepted (forward/inference path). fp16 TRAINING needs a GradScaler
    in the M-step -- a documented buildout -- so it is NOT enforced-rejected at config time
    (rejecting would also block the legitimate fp16 inference path tests/test_amp.py pins)."""
    cfg = VFE3Config(amp_dtype="fp16")
    assert cfg.amp_dtype == "fp16"


def test_m_phi_natural_grad_without_pullback_warns() -> None:
    """m_phi_natural_grad=True + non-pullback phi_precond_mode warns (no geometric metric)."""
    with pytest.warns(UserWarning, match="phi_precond_mode"):
        VFE3Config(m_phi_natural_grad=True, phi_precond_mode="none")


def test_close_basis_field_exists_and_defaults_none() -> None:
    """The close_basis gauge-seam field exists and AUTO-defaults to None."""
    cfg = VFE3Config()
    assert cfg.close_basis is None


def test_unroll_oracle_route_warns_for_active_param() -> None:
    """unroll + oracle-routing family (gaussian_full) + an E-step-only learnable param warns.

    pos_phi='learned' (the default) is the active E-step-only param; switching the family
    off the closed-form kernel (gaussian_full) routes the belief gradient to the detached
    oracle, so the unroll signal is truncated -> the second freeze-warning must fire.
    """
    with pytest.warns(UserWarning, match="oracle"):
        VFE3Config(
            family="gaussian_full",
            decode_mode="full",
        )


def test_default_config_constructs_silently() -> None:
    """The DEFAULT config constructs with no error and emits NO warning.

    This is the load-bearing negative test: the default (filtering / gaussian_diagonal /
    renyi / renyi_order=1.0 / entropy=True) is exactly the closed-form kernel path, so neither
    freeze-warning may fire even though pos_phi='learned' is on by default. Promoting every
    warning to an error makes any spurious warning a test failure.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cfg = VFE3Config()
    assert cfg.pos_phi == "learned"
    assert cfg.e_step_gradient == "unroll"
