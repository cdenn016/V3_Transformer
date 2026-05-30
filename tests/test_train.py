import math

import torch

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel
from vfe3.train import build_optimizer, lr_lambda


def test_optimizer_groups_priors_by_m_lr():
    cfg = VFE3Config(vocab_size=20, embed_dim=4, n_heads=2,
                     m_mu_lr=0.01, m_sigma_lr=0.002, m_phi_lr=0.005)
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)
    lrs = sorted(g["lr"] for g in opt.param_groups)
    assert lrs == [0.002, 0.005, 0.01]
    # every PriorBank parameter is covered by exactly one group
    n_params = sum(len(g["params"]) for g in opt.param_groups)
    assert n_params == len(list(model.parameters()))


def test_lr_lambda_warmup_then_cosine():
    cfg = VFE3Config(warmup_steps=10, max_steps=100)
    assert abs(lr_lambda(0, cfg) - 0.0) < 1e-6
    assert abs(lr_lambda(10, cfg) - 1.0) < 1e-6            # peak at end of warmup
    assert lr_lambda(55, cfg) < 1.0 and lr_lambda(55, cfg) > 0.0
    assert abs(lr_lambda(100, cfg) - 0.0) < 1e-3           # ~0 at max_steps
