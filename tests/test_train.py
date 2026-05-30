import math

import torch
from torch.utils.data import DataLoader

from vfe3.config import VFE3Config
from vfe3.data.datasets import TokenWindows
from vfe3.model.model import VFEModel
from vfe3.train import build_optimizer, lr_lambda, train


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


def _periodic_loader(V=6, period=3, n=600, seq_len=8, batch_size=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    base = torch.arange(period).repeat(n // period + 2)         # 0,1,2,0,1,2,...
    ds = TokenWindows(base[: n].to(torch.long), seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True,
                      generator=g)


def test_training_decreases_loss_on_structured_stream():
    # The period-3 shift (0->1->2->0) is a DIRECTED map: predicting the next token from
    # the current one. Causal attention only AVERAGES past beliefs, so with the gauge
    # frame frozen (e_phi_lr=m_phi_lr=0) the priors collapse to a symmetric "predict the
    # marginal over the active tokens" optimum and CE pins at ln(3) ~ 1.099. The gauge
    # transport Omega_ij(phi) is the one degree of freedom that applies a DIRECTED (non-
    # averaging) rotation to coupled beliefs; turning it on (e_phi_lr, m_phi_lr > 0)
    # breaks the symmetry and drives CE below ln(3). See the changelog for the curve.
    torch.manual_seed(0)
    cfg = VFE3Config(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                     n_e_steps=3, e_mu_lr=0.3, e_phi_lr=0.3,
                     m_mu_lr=0.05, m_sigma_lr=0.01, m_phi_lr=0.05, warmup_steps=5, max_steps=200)
    model = VFEModel(cfg)
    losses = train(model, _periodic_loader(V=6, period=3), cfg, n_steps=200)
    assert losses[-1] < 0.6 * losses[0]                         # the model LEARNS the period


def test_training_smoke_on_real_wikitext2_if_present():
    import pytest
    from vfe3.data.datasets import load_cached_tokens
    try:
        toks = load_cached_tokens("wikitext-2", "validation")
    except FileNotFoundError:
        pytest.skip("wikitext-2 cache absent")
    torch.manual_seed(0)
    cfg = VFE3Config(vocab_size=50257, embed_dim=8, n_heads=2, max_seq_len=16, n_layers=1,
                     n_e_steps=1, e_mu_lr=0.3, e_phi_lr=0.0,
                     m_mu_lr=0.05, m_sigma_lr=0.01, m_phi_lr=0.0, warmup_steps=3, max_steps=30)
    model = VFEModel(cfg)
    ds = TokenWindows(toks[:4000], 16)
    loader = DataLoader(ds, batch_size=8, shuffle=True, drop_last=True)
    losses = train(model, loader, cfg, n_steps=30)
    assert all(map(lambda x: x == x, losses))                   # finite (no NaN)
    assert losses[-1] < losses[0] - 0.05                        # real-token loss decreases
