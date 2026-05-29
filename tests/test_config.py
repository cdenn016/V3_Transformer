from vfe3.config import VFE3Config


def test_config_defaults():
    cfg = VFE3Config()
    assert cfg.eps == 1e-6
    assert cfg.kl_max == 100.0
    assert cfg.divergence_family == "gaussian_diagonal"
    assert cfg.alpha_div == 1.0


def test_config_rejects_unknown_family():
    import pytest
    with pytest.raises(ValueError):
        VFE3Config(divergence_family="not_a_family")


def test_config_rejects_nonpositive_alpha():
    import pytest
    with pytest.raises(ValueError):
        VFE3Config(alpha_div=0.0)
