import pytest

from vfe3.config import VFE3Config


def test_config_defaults():
    cfg = VFE3Config()
    assert cfg.eps == 1e-6
    assert cfg.kl_max == 100.0
    assert cfg.divergence_family == "renyi"
    assert cfg.alpha_div == 1.0


def test_config_rejects_unknown_family():
    with pytest.raises(ValueError):
        VFE3Config(divergence_family="not_a_family")


def test_divergence_family_is_functional_seam_distinct_from_family():
    """divergence_family selects the divergence FUNCTIONAL (renyi, ...); family selects the
    covariance structure. They are distinct seams: a covariance-family value is not a valid
    functional, and the default functional is 'renyi'."""
    cfg = VFE3Config()
    assert cfg.divergence_family == "renyi" and cfg.family == "gaussian_diagonal"
    with pytest.raises(ValueError):
        VFE3Config(divergence_family="gaussian_diagonal")   # covariance family != functional
    with pytest.raises(ValueError):
        VFE3Config(divergence_family="not_a_functional")


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


# --- Audit 2026-05-31: dead / trapping toggles are live + rejected, not silent ----
def test_config_rejects_omega_direct_gauge_parameterization():
    """omega_direct needs a per-token GL(K) matrix the no-NN belief (phi only) cannot supply,
    so it is rejected at construction rather than silently aliased to the 'phi' path."""
    with pytest.raises(NotImplementedError):
        VFE3Config(gauge_parameterization="omega_direct")
    assert VFE3Config(gauge_parameterization="phi").gauge_parameterization == "phi"


def test_config_rejects_use_prior_bank_false():
    """use_prior_bank=False has no alternative encode/decode path; rejected, not a silent no-op."""
    with pytest.raises(NotImplementedError):
        VFE3Config(use_prior_bank=False)


def test_config_rejects_gauge_fixed_encode():
    """encode_mode='gauge_fixed' is an unimplemented stub; rejected at construction, not at forward."""
    with pytest.raises(NotImplementedError):
        VFE3Config(encode_mode="gauge_fixed")


def test_config_has_state_dependent_alpha_shape_params():
    """b0/c0 (state-dependent-alpha shape parameters) are configurable and validated positive."""
    cfg = VFE3Config()
    assert cfg.b0 == 1.0 and cfg.c0 == 1.0
    with pytest.raises(ValueError):
        VFE3Config(b0=0.0)
    with pytest.raises(ValueError):
        VFE3Config(c0=-1.0)


def test_config_phi_retract_mode_validated():
    """phi_retract_mode selects the Lie-algebra composition chart (euclidean | bch)."""
    assert VFE3Config().phi_retract_mode == "euclidean"
    assert VFE3Config(phi_retract_mode="bch").phi_retract_mode == "bch"
    with pytest.raises(ValueError):
        VFE3Config(phi_retract_mode="not_a_mode")
