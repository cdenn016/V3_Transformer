import pytest
import torch
from vfe3.config import VFE3Config


def _omega_cfg(**over):
    base = dict(gauge_parameterization="omega_direct", transport_mode="flat", e_phi_lr=0.0,
                embed_dim=4, n_heads=1, use_head_mixer=False, lambda_gamma=0.0, s_e_step=False)
    base.update(over)
    return VFE3Config(**base)


def test_metropolis_constructs_for_reflect_ok_groups():
    for grp, over in (("glk", {}), ("block_glk", {"n_heads": 2}), ("so_k", {})):
        cfg = _omega_cfg(omega_reflection="metropolis", gauge_group=grp, **over)
        assert cfg.omega_reflection == "metropolis"
        assert cfg.omega_metropolis_temperature == 1.0     # default
        assert cfg.omega_metropolis_every == 1             # default


def test_metropolis_rejected_vacuous_and_deferred_groups():
    for grp, over in (("sp", {}), ("sp_n", {"embed_dim": 5, "group_n": 4,
                                            "irrep_spec": [("sym0", 1), ("sym1", 1)]}),
                      ("so_n", {"group_n": 3, "irrep_spec": [("l0", 1), ("l1", 1)]}),
                      ("tied_block_glk", {"n_heads": 2})):
        with pytest.raises(ValueError):
            _omega_cfg(omega_reflection="metropolis", gauge_group=grp, **over)


def test_ste_not_implemented():
    with pytest.raises((NotImplementedError, ValueError), match="ste"):
        _omega_cfg(omega_reflection="ste", gauge_group="glk")


def test_metropolis_temperature_and_cadence_validated():
    with pytest.raises(ValueError):
        _omega_cfg(omega_reflection="metropolis", gauge_group="glk", omega_metropolis_temperature=0.0)
    with pytest.raises(ValueError):
        _omega_cfg(omega_reflection="metropolis", gauge_group="glk", omega_metropolis_every=0)


def test_off_and_init_seed_unchanged():
    assert _omega_cfg(gauge_group="glk").omega_reflection == "off"
    assert _omega_cfg(omega_reflection="init_seed", gauge_group="glk").omega_reflection == "init_seed"
