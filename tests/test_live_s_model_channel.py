r"""Live model channel s (dynamic prior tie), default-off. Spec:
docs/superpowers/specs/2026-06-08-live-s-model-channel-design.md.
"""

import warnings

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel


def _tiny_cfg(**overrides) -> VFE3Config:
    r"""Minimal model config (embed_dim=4, n_heads=2, vocab=8, seq=4, 1 layer)."""
    base = dict(vocab_size=8, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=1)
    base.update(overrides)
    return VFE3Config(**base)


def test_s_e_step_defaults_off():
    cfg = VFE3Config()
    assert cfg.s_e_step is False
    assert cfg.e_s_mu_lr == 0.1
    assert cfg.e_s_sigma_lr == 0.1


def test_s_e_step_lr_validation_rejects_negative():
    with pytest.raises(ValueError, match="e_s_mu_lr"):
        _tiny_cfg(s_e_step=True, prior_source="model_channel", e_s_mu_lr=-1.0)
    with pytest.raises(ValueError, match="e_s_sigma_lr"):
        _tiny_cfg(s_e_step=True, prior_source="model_channel", e_s_sigma_lr=-0.5)


def test_s_e_step_requires_model_channel_prior_source():
    # s_e_step anchors the belief to s AND must decode against s -> require model_channel.
    with pytest.raises(ValueError, match="prior_source"):
        _tiny_cfg(s_e_step=True, prior_source="token")


def test_s_e_step_inert_misconfig_warns():
    with pytest.warns(UserWarning, match="s_e_step"):
        _tiny_cfg(s_e_step=True, prior_source="model_channel",
                  lambda_h=0.0, gamma_coupling=0.0)


def test_s_tables_and_frozen_r_created_under_s_e_step():
    m = VFEModel(_tiny_cfg(s_e_step=True, prior_source="model_channel",
                           lambda_h=1.0, gamma_coupling=1.0))
    pb = m.prior_bank
    assert getattr(pb, "s_mu_embed", None) is not None
    assert getattr(pb, "r_mu", None) is not None
    assert pb.r_mu.requires_grad is False
    assert pb.r_sigma_log.requires_grad is False


def test_belief_tables_byte_identical_with_or_without_s_e_step():
    # s-tables are drawn LAST, so the belief tables (drawn first) are bit-identical.
    torch.manual_seed(0); off = VFEModel(_tiny_cfg(s_e_step=False))
    torch.manual_seed(0); on = VFEModel(_tiny_cfg(s_e_step=True, prior_source="model_channel",
                                                  lambda_h=1.0, gamma_coupling=1.0))
    assert torch.equal(off.prior_bank.mu_embed, on.prior_bank.mu_embed)
    assert torch.equal(off.prior_bank.phi_embed, on.prior_bank.phi_embed)
    assert torch.equal(off.prior_bank.sigma_log_embed, on.prior_bank.sigma_log_embed)
