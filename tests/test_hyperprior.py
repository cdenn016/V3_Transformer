r"""Hyper-prior channel, first increment: lambda_h * mean_i KL(s_i || r).

Manuscript Participatory_it_from_bit.tex eq:pointwise_free_energy (lines 1241-1249):
the canonical two-tier free energy carries a hyper-prior term lambda_h sum_i KL(s_i||r_i)
regularizing the model-channel beliefs s_i toward the hyper-prior centroid r. This first
increment wires the SECOND (model) belief channel s_i + global hyper-prior r end-to-end at
the smallest scope: new learned PriorBank tables (s_mu_embed/s_sigma_log_embed, r_mu/r_sigma_log)
created ONLY when lambda_h>0, encoded per token as a diagonal Gaussian s_i, and added to the
training loss as lambda_h * mean_i KL(s_i||r). s_i does NOT yet couple into the belief q / the
prediction path (the h->s->p->q coupling, the s-channel E-step update, and the gamma
model-coupling block are all DEFERRED to increment 2). Default-off (lambda_h=0): no s/r tables,
loss byte-identical to the term-absent path.

These tests pin the contracts: (1) default-off has no s/r tables and loss == ce (the pure path);
(2) the term is EXACTLY lambda_h * mean KL(s||r) (the linear-in-lambda_h oracle), with hp
recomputed independently from the model's s/r tables; (3) the channel trains (finite grads on
the s/r params); (4) s == r => term 0 (self-zero sanity).
"""

import torch

from vfe3.config import VFE3Config
from vfe3.families.gaussian import DiagonalGaussian
from vfe3.free_energy import self_divergence
from vfe3.model.model import VFEModel


def _hyperprior_term(model: VFEModel, tokens: torch.Tensor) -> torch.Tensor:
    r"""Independent oracle: recompute mean_i KL(s_i || r) from the model's s/r tables by the
    SAME recipe forward uses (encode s per token, broadcast r, self_divergence, .mean())."""
    cfg = model.cfg
    pb = model.prior_bank
    s_mu, s_sigma = pb.encode_s(tokens)                       # (B, N, K)
    r_mu = pb.r_mu                                            # (K,)
    r_sigma = torch.exp(pb.r_sigma_log).clamp(min=cfg.eps)    # (K,)
    return self_divergence(
        DiagonalGaussian(s_mu, s_sigma), DiagonalGaussian(r_mu, r_sigma),
        alpha=cfg.alpha_div, kl_max=cfg.kl_max, eps=cfg.eps,
        divergence_family=cfg.divergence_family,
    ).mean()


def _make_model(lambda_h: float, *, seed: int = 0) -> VFEModel:
    cfg = VFE3Config(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=1,
                     n_e_steps=1, e_mu_lr=0.5, e_phi_lr=0.0, mass_phi=0.0,
                     mstep_self_coupling_weight=0.0, lambda_h=lambda_h, seed=seed)
    torch.manual_seed(seed)          # the model does NOT self-seed; pin RNG before construction
    return VFEModel(cfg)


def test_default_off_no_tables_and_loss_is_ce():
    # Default-off (lambda_h=0): no s/r tables exist, and loss == ce (mass_phi=0 too) -- the new
    # code is fully inert on the pure path.
    model = _make_model(0.0)
    assert not hasattr(model.prior_bank, "s_mu_embed")
    assert not hasattr(model.prior_bank, "r_mu")
    tokens = torch.randint(0, 20, (3, 5))
    targets = torch.randint(0, 20, (3, 5))
    _, loss, ce = model(tokens, targets)             # ce returned is ce.detach()
    assert torch.allclose(loss, ce)


def test_linear_in_lambda_h():
    # The oracle: loss_w - loss_0 == w * mean KL(s||r), with hp recomputed independently from the
    # lambda_h>0 model's s/r tables. Both models share the seed, and the s/r draws come LAST in
    # PriorBank.__init__, so the belief tables are byte-identical between the two models.
    w = 0.5
    model_0 = _make_model(0.0)
    model_w = _make_model(w)
    assert torch.equal(model_0.prior_bank.mu_embed, model_w.prior_bank.mu_embed)   # belief tables identical
    assert torch.equal(model_0.prior_bank.phi_embed, model_w.prior_bank.phi_embed)
    tokens = torch.randint(0, 20, (3, 5))
    targets = torch.randint(0, 20, (3, 5))
    _, loss_0, _ = model_0(tokens, targets)
    _, loss_w, _ = model_w(tokens, targets)
    hp = _hyperprior_term(model_w, tokens)
    assert hp > 1e-6                                  # non-vacuous: s != r at init
    assert torch.allclose(loss_w - loss_0, w * hp, atol=1e-6)


def test_grad_flows_to_s_and_r_tables():
    # The channel trains: after forward+backward with lambda_h>0, the s/r table params have
    # finite, nonzero grad.
    model = _make_model(0.5)
    tokens = torch.randint(0, 20, (3, 5))
    targets = torch.randint(0, 20, (3, 5))
    _, loss, _ = model(tokens, targets)
    loss.backward()
    for name in ("s_mu_embed", "s_sigma_log_embed", "r_mu", "r_sigma_log"):
        grad = getattr(model.prior_bank, name).grad
        assert grad is not None, f"{name} received no grad"
        assert torch.isfinite(grad).all(), f"{name} grad not finite"
    assert model.prior_bank.s_mu_embed.grad.abs().sum() > 0
    assert model.prior_bank.r_mu.grad.abs().sum() > 0


def test_self_zero_when_s_equals_r():
    # Self-divergence sanity: if s and r are set equal, the term is 0.
    model = _make_model(0.5)
    pb = model.prior_bank
    with torch.no_grad():
        # Force every token's s onto a single (K,) vector equal to r, and matching variances.
        pb.s_mu_embed.copy_(pb.r_mu.unsqueeze(0).expand_as(pb.s_mu_embed))
        pb.s_sigma_log_embed.copy_(pb.r_sigma_log.unsqueeze(0).expand_as(pb.s_sigma_log_embed))
    tokens = torch.randint(0, 20, (3, 5))
    hp = _hyperprior_term(model, tokens)
    assert torch.allclose(hp, torch.zeros_like(hp), atol=1e-6)
