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


def test_tau_is_kappa_sqrt_d_head():
    # Audit finding 6c: tau = kappa * sqrt(d_head) (per-head, Vaswani sqrt(d_k)),
    # NOT sqrt(embed_dim). embed_dim=16, n_heads=4 -> d_head=4 -> sqrt(d_head)=2.
    cfg = VFE3Config(embed_dim=16, n_heads=4, kappa=1.5)
    assert abs(cfg.tau - 1.5 * 2.0) < 1e-9
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


# --- audit Group 2: wired/enforced config seams (were dead knobs) ---

def test_use_prior_bank_false_rejected():
    """use_prior_bank=False has no alternative path: the knob is live and rejects it."""
    with pytest.raises(NotImplementedError):
        VFE3Config(use_prior_bank=False)


def test_divergence_family_must_equal_family():
    """divergence_family is enforced consistent with the single source of truth `family`."""
    with pytest.raises(ValueError):
        VFE3Config(divergence_family="gaussian_full")  # family defaults to gaussian_diagonal


def test_diagonal_covariance_must_agree_with_family():
    """diagonal_covariance is enforced consistent with family."""
    with pytest.raises(ValueError):
        VFE3Config(diagonal_covariance=False)          # family defaults to gaussian_diagonal
    # the consistent full-covariance triple is accepted
    VFE3Config(family="gaussian_full", divergence_family="gaussian_full", diagonal_covariance=False)


def test_phi_retract_mode_validated():
    with pytest.raises(ValueError):
        VFE3Config(phi_retract_mode="not_a_mode")
    assert VFE3Config(phi_retract_mode="bch").phi_retract_mode == "bch"


def test_seed_field_present():
    assert VFE3Config(seed=7).seed == 7


def test_gauge_parameterization_omega_direct_rejected():
    """omega_direct has no per-token GL(K) source in the no-NN belief: live knob, rejected."""
    with pytest.raises(NotImplementedError):
        VFE3Config(gauge_parameterization="omega_direct")
    assert VFE3Config(gauge_parameterization="phi").gauge_parameterization == "phi"
