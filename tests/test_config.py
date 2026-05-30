import pytest

from vfe3.config import VFE3Config


def test_config_defaults():
    cfg = VFE3Config()
    assert cfg.eps == 1e-6
    assert cfg.kl_max == 100.0
    assert cfg.divergence_family == "gaussian_diagonal"
    assert cfg.alpha_div == 1.0


def test_config_rejects_unknown_family():
    with pytest.raises(ValueError):
        VFE3Config(divergence_family="not_a_family")


def test_config_rejects_nonpositive_alpha():
    with pytest.raises(ValueError):
        VFE3Config(alpha_div=0.0)


def test_config_rejects_nonpositive_eps():
    with pytest.raises(ValueError):
        VFE3Config(eps=0.0)


def test_config_rejects_nonpositive_kl_max():
    with pytest.raises(ValueError):
        VFE3Config(kl_max=0.0)


# --- Phase 7 full-config fields --------------------------------------------
def test_config_model_defaults():
    cfg = VFE3Config()
    assert cfg.embed_dim == 64 and cfg.n_heads == 8 and cfg.n_layers == 1
    assert cfg.gauge_group == "block_glk" and cfg.decode_mode == "diagonal"
    assert cfg.use_prior_bank is True


def test_tau_is_kappa_sqrt_k_and_d_head():
    cfg = VFE3Config(embed_dim=16, n_heads=4, kappa=1.5)
    assert abs(cfg.tau - 1.5 * 4.0) < 1e-9
    assert cfg.d_head == 4


def test_config_rejects_embed_dim_not_divisible_by_heads():
    with pytest.raises(ValueError):
        VFE3Config(embed_dim=10, n_heads=3)


def test_config_rejects_unknown_gauge_group_and_decode_mode():
    with pytest.raises(ValueError):
        VFE3Config(gauge_group="not_a_group")
    with pytest.raises(ValueError):
        VFE3Config(decode_mode="not_a_mode")


def test_config_rejects_negative_learning_rate_and_bad_rho():
    with pytest.raises(ValueError):
        VFE3Config(e_mu_lr=-0.1)
    with pytest.raises(ValueError):
        VFE3Config(prior_handoff_rho=1.5)
