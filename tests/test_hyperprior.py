r"""Hyper-prior channel, first increment: lambda_h * mean_i KL(s_i || r).

Manuscript Participatory_it_from_bit.tex eq:pointwise_free_energy (lines 1241-1249):
the canonical two-tier free energy carries a hyper-prior term lambda_h sum_i KL(s_i||r_i)
regularizing the model-channel beliefs s_i toward the hyper-prior centroid r. This first
increment wires the SECOND (model) belief channel s_i + global hyper-prior r end-to-end at
the smallest scope: new learned PriorBank tables (s_mu_embed/s_sigma_log_embed, r_mu/r_sigma_log)
created when lambda_h>0, encoded per token as a diagonal Gaussian s_i, and added to the training
loss as lambda_h * mean_i KL(s_i||r). s_i does NOT couple into the belief q / the prediction path
(the h->s->p->q coupling and the s-channel E-step update remain DEFERRED; the gamma model-coupling
block, which shares these s tables, is built in test_gamma_coupling.py and is likewise predictively
inert). Default-off (lambda_h=0): no s/r tables, loss byte-identical to the term-absent path.

These tests pin the contracts: (1) default-off has no s/r tables and loss == ce (the pure path);
(2) the term is EXACTLY lambda_h * mean KL(s||r) (the linear-in-lambda_h oracle), with hp
recomputed independently from the model's s/r tables; (3) the model beliefs s train while the
hyper-prior centroid r is FROZEN (requires_grad=False), and build_optimizer (which exempts frozen
params) groups s but not r; (4) s == r => term 0 (self-zero sanity).
"""

import warnings

import pytest
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
        alpha=cfg.renyi_order, kl_max=cfg.kl_max, eps=cfg.eps,
        divergence_family=cfg.divergence_family,
    ).mean()


def _make_model(lambda_h: float, *, seed: int = 0) -> VFEModel:
    cfg = VFE3Config(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=1,
                     n_e_steps=1, e_q_mu_lr=0.5, e_phi_lr=0.0, mass_phi=0.0,
                     mstep_self_coupling_weight=0.0, lambda_h=lambda_h, seed=seed)
    # pos_phi default "learned" is fine: pos_phi_free is seeded from a dedicated cfg.seed generator
    # (model.py), so it is byte-identical between the lambda_h=0 and lambda_h>0 models and
    # loss_w - loss_0 == w * mean KL(s||r) holds exactly (no RNG divergence in the CE part)
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


def test_grad_flows_to_s_tables_r_is_frozen():
    # The model beliefs s_i train; the hyper-prior centroid r is a FROZEN fixed centroid
    # (requires_grad=False), so it never receives a gradient. Free-training r alongside s would
    # trivially collapse KL(s||r)->0; the manuscript sets r "from a higher, slower meta-level"
    # (GL(K)_supplementary.tex:1081), so a fixed r is the manuscript-consistent stand-in.
    model = _make_model(0.5)
    tokens = torch.randint(0, 20, (3, 5))
    targets = torch.randint(0, 20, (3, 5))
    _, loss, _ = model(tokens, targets)
    loss.backward()
    for name in ("s_mu_embed", "s_sigma_log_embed"):
        grad = getattr(model.prior_bank, name).grad
        assert grad is not None, f"{name} received no grad"
        assert torch.isfinite(grad).all(), f"{name} grad not finite"
    assert model.prior_bank.s_mu_embed.grad.abs().sum() > 0
    # r is frozen: not trainable, receives no gradient
    assert model.prior_bank.r_mu.requires_grad is False
    assert model.prior_bank.r_sigma_log.requires_grad is False
    assert model.prior_bank.r_mu.grad is None


def test_build_optimizer_with_lambda_h_excludes_frozen_r():
    # With lambda_h>0 the s tables (trainable) plus the FROZEN r tables exist. build_optimizer must
    # succeed (the exact-coverage guard exempts non-trainable params) and group s but NOT r.
    from vfe3.train import build_optimizer
    model = _make_model(0.5)
    opt = build_optimizer(model, model.cfg)               # must NOT raise
    opt_params = {id(p) for g in opt.param_groups for p in g["params"]}
    assert id(model.prior_bank.s_mu_embed) in opt_params       # s trains
    assert id(model.prior_bank.r_mu) not in opt_params         # frozen r is not optimized


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


# ---------------------------------------------------------------------------
# learnable_r: opt-in trainable hyper-prior centroid (default frozen).
# Spec: docs/superpowers/specs/2026-06-13-learnable-hyper-prior-r-design.md.
# The TODO(B) un-freezing: a single learnable_r toggle makes r a trainable
# empirical-Bayes centroid; the default stays frozen (byte-identical current
# behavior). A config guard warns when r is un-frozen while s is unanchored
# (only KL(s||r) binds them -> KL(s||r)->0 collapse).
# ---------------------------------------------------------------------------
def _lr_model(
    *,
    learnable_r: bool,
    seed:        int  = 0,
) -> VFEModel:
    r"""Tiny pure-path model (use_prior_bank=True) with the hyper-prior channel live (lambda_h>0) and s
    data-anchored (prior_source='model_channel'), i.e. the non-collapse regime for a learnable r."""
    cfg = VFE3Config(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=1,
                     n_e_steps=1, e_q_mu_lr=0.5, e_phi_lr=0.0, mass_phi=0.0,
                     mstep_self_coupling_weight=0.0, use_prior_bank=True,
                     lambda_h=0.5, prior_source="model_channel",
                     learnable_r=learnable_r, seed=seed)
    torch.manual_seed(seed)          # the model does NOT self-seed; pin RNG before construction
    return VFEModel(cfg)


def test_learnable_r_defaults_false():
    # The default is frozen r (current behavior): the toggle must default off.
    assert VFE3Config().learnable_r is False


def test_default_path_r_still_frozen_and_ungrouped():
    # Regression pin: with learnable_r unset, r is frozen and build_optimizer does NOT group it
    # (the coverage guard exempts non-trainable params).
    from vfe3.train import build_optimizer
    m = _lr_model(learnable_r=False)
    assert m.prior_bank.r_mu.requires_grad is False
    assert m.prior_bank.r_sigma_log.requires_grad is False
    opt = build_optimizer(m, m.cfg)
    opt_params = {id(p) for g in opt.param_groups for p in g["params"]}
    assert id(m.prior_bank.r_mu) not in opt_params
    assert id(m.prior_bank.r_sigma_log) not in opt_params


def test_learnable_r_makes_r_trainable():
    # The opt-in toggle un-freezes r: both centroid tables become trainable leaves.
    m = _lr_model(learnable_r=True)
    assert m.prior_bank.r_mu.requires_grad is True
    assert m.prior_bank.r_sigma_log.requires_grad is True


def test_learnable_r_grad_reaches_r():
    # With r un-frozen, the lambda_h*KL(s||r) term backprops a finite, nonzero gradient into r
    # (s != r at init, so the centroid actually moves).
    m = _lr_model(learnable_r=True)
    tokens = torch.randint(0, 20, (3, 5))
    targets = torch.randint(0, 20, (3, 5))
    _, loss, _ = m(tokens, targets)
    loss.backward()
    g_mu = m.prior_bank.r_mu.grad
    g_sig = m.prior_bank.r_sigma_log.grad
    assert g_mu is not None and torch.isfinite(g_mu).all() and g_mu.abs().sum() > 0
    assert g_sig is not None and torch.isfinite(g_sig).all() and g_sig.abs().sum() > 0


def test_build_optimizer_groups_learnable_r():
    # A trainable r MUST be grouped or the exact-coverage guard fails (it would silently never
    # train). Both centroid tables are grouped (mean@m_p_mu_lr, log-scale@m_p_sigma_lr like s).
    from vfe3.train import build_optimizer
    m = _lr_model(learnable_r=True)
    opt = build_optimizer(m, m.cfg)                       # must NOT raise the coverage guard
    opt_params = {id(p) for g in opt.param_groups for p in g["params"]}
    assert id(m.prior_bank.r_mu) in opt_params
    assert id(m.prior_bank.r_sigma_log) in opt_params


def test_forward_loss_identical_frozen_vs_learnable_at_init():
    # Un-freezing only flips requires_grad: the r tables (drawn last) are byte-identical and the
    # forward loss at init is unchanged. learnable_r changes training dynamics, not the forward value.
    m0 = _lr_model(learnable_r=False, seed=0)
    m1 = _lr_model(learnable_r=True, seed=0)
    assert torch.equal(m0.prior_bank.r_mu, m1.prior_bank.r_mu)
    assert torch.equal(m0.prior_bank.s_mu_embed, m1.prior_bank.s_mu_embed)
    tokens = torch.randint(0, 20, (3, 5))
    targets = torch.randint(0, 20, (3, 5))
    _, l0, _ = m0(tokens, targets)
    _, l1, _ = m1(tokens, targets)
    assert torch.equal(l0, l1)       # byte-identical: un-freezing only flips requires_grad


def test_learnable_r_collapse_warning_when_s_unanchored():
    # Un-freezing r while s is NOT data-anchored (prior_source='token', no gamma, no s_e_step)
    # leaves KL(s||r) the only force on s/r -> the collapse the freeze guards against. Warn.
    with pytest.warns(UserWarning, match="learnable_r"):
        VFE3Config(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5,
                   lambda_h=0.5, learnable_r=True)        # prior_source='token', lambda_gamma=0


def test_learnable_r_no_collapse_warning_when_model_channel_anchors():
    # prior_source='model_channel' routes the CE gradient into s, anchoring it -> r learning the
    # population centroid is empirical Bayes, not a collapse. No collapse warning.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        VFE3Config(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5,
                   lambda_h=0.5, learnable_r=True, prior_source="model_channel")
    assert not any("learnable_r" in str(wi.message) for wi in caught)


def test_learnable_r_grad_reaches_r_under_s_e_step():
    # Second gradient route: under s_e_step the forward hyper-prior term is OFF and r is instead the
    # self-coupling target of the s E-step (_refine_s). With r un-frozen, grad still reaches it through
    # the unrolled refine (s is anchored by CE under the required model_channel, so this is non-collapse).
    cfg = VFE3Config(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=1,
                     n_e_steps=1, e_q_mu_lr=0.5, e_phi_lr=0.0, mass_phi=0.0,
                     mstep_self_coupling_weight=0.0, use_prior_bank=True,
                     lambda_h=1.0, lambda_gamma=1.0, prior_source="model_channel",
                     s_e_step=True, e_s_mu_lr=0.5, learnable_r=True, seed=0)
    torch.manual_seed(0)
    m = VFEModel(cfg)
    tokens = torch.randint(0, 20, (2, 5))
    targets = torch.randint(0, 20, (2, 5))
    _, loss, _ = m(tokens, targets)
    loss.backward()
    g = m.prior_bank.r_mu.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


def test_learnable_r_inert_warning_when_no_r_channel():
    # learnable_r=True but lambda_h=0 and not s_e_step: r is never created (lambda_gamma>0 builds only
    # the s tables), so the toggle is a silent no-op -> warn the user it has no effect.
    with pytest.warns(UserWarning, match="learnable_r=True has no effect"):
        VFE3Config(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5,
                   lambda_h=0.0, lambda_gamma=0.5, learnable_r=True)


def test_learnable_r_centroid_has_no_weight_decay():
    # A hyper-prior centroid is a prior, not capacity: its optimizer groups must NOT carry weight decay
    # (decaying it biases r toward the degenerate (0,1) fixed point), like the unigram-bias/gauge groups.
    from vfe3.train import build_optimizer
    m = _lr_model(learnable_r=True)
    opt = build_optimizer(m, m.cfg)
    rids = {id(m.prior_bank.r_mu), id(m.prior_bank.r_sigma_log)}
    r_groups = [g for g in opt.param_groups if any(id(p) in rids for p in g["params"])]
    assert len(r_groups) == 2
    assert all(g["weight_decay"] == 0.0 for g in r_groups)


def test_learnable_r_grouped_under_natural_grad_optimizer():
    # The other authorized optimizer path: m_phi_natural_grad=True returns GaugeManifoldAdamW; a
    # trainable r must still be grouped (in a plain non-gauge group, mean@m_p_mu_lr / log-scale@m_p_sigma_lr).
    from vfe3.train import build_optimizer
    from vfe3.gauge_optim import GaugeManifoldAdamW
    cfg = VFE3Config(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=1,
                     n_e_steps=1, use_prior_bank=True, lambda_h=0.5, prior_source="model_channel",
                     learnable_r=True, m_phi_natural_grad=True,
                     phi_precond_mode="pullback_per_block", seed=0)
    torch.manual_seed(0)
    m = VFEModel(cfg)
    opt = build_optimizer(m, cfg)
    assert isinstance(opt, GaugeManifoldAdamW)
    opt_params = {id(p) for g in opt.param_groups for p in g["params"]}
    assert id(m.prior_bank.r_mu) in opt_params
    assert id(m.prior_bank.r_sigma_log) in opt_params


# ---------------------------------------------------------------------------
# lambda_h_mode: the model-fiber analogue of lambda_alpha_mode (constant / state_dependent
# envelope), and r_update_mode='barycenter': the closed-form forward-KL
# barycenter M-step for r. Spec: docs/.../2026-06-13-lambda-h-mode-and-r-update-mechanism.md.
# lambda_h weights KL(s_i||r) the way lambda_alpha weights KL(q_i||p_i); the manuscript names the
# state-dependent lambda_h "a parallel extension not developed here" (Participatory ~3766).
# ---------------------------------------------------------------------------
from vfe3.lambda_h_i import hyper_prior_lambda_h, _LAMBDA_H_MODES


def _scored_model(lambda_h: float, *, lambda_h_mode: str = "constant",
                  b0_h: float = 1.0, c0_h: float = 1.0, seed: int = 0) -> VFEModel:
    r"""Scored-regime model (s_e_step=False): the hyper-prior term is added directly to the loss as
    _hyper_prior_term, so the lambda_h_mode weighting/regularizer is exercised at the loss level."""
    cfg = VFE3Config(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=1,
                     n_e_steps=1, e_q_mu_lr=0.5, e_phi_lr=0.0, mass_phi=0.0,
                     mstep_self_coupling_weight=0.0, lambda_h=lambda_h, lambda_h_mode=lambda_h_mode,
                     b0_h=b0_h, c0_h=c0_h, seed=seed)
    torch.manual_seed(seed)
    return VFEModel(cfg)


def test_lambda_h_mode_and_r_update_mode_defaults():
    # Defaults preserve current behavior exactly: constant lambda_h, gradient r.
    c = VFE3Config()
    assert c.lambda_h_mode == "constant"
    assert c.r_update_mode == "gradient"
    assert c.b0_h == 1.0 and c.c0_h == 1.0


def test_constant_lambda_h_mode_is_byte_identical():
    # constant lambda_h_mode reproduces the pre-registry weighting: loss_w - loss_0 == w * mean KL.
    w = 0.5
    m0 = _scored_model(0.0); mw = _scored_model(w, lambda_h_mode="constant")
    tokens = torch.randint(0, 20, (3, 5)); targets = torch.randint(0, 20, (3, 5))
    _, l0, _ = m0(tokens, targets)
    _, lw, _ = mw(tokens, targets)
    hp = _hyperprior_term(mw, tokens)            # mean KL oracle
    assert hp > 1e-6
    assert torch.allclose(lw - l0, w * hp, atol=1e-6)


def test_state_dependent_lambda_h_scored_term_matches_envelope():
    # The scored term equals mean_i[ c0_h/(b0_h+KL(s_i||r)) * KL + R_h ], the exact lambda_h envelope.
    m = _scored_model(0.5, lambda_h_mode="state_dependent", b0_h=0.3, c0_h=0.7)
    tokens = torch.randint(0, 20, (3, 5))
    kl = m._hyper_prior_kl(tokens)
    lam, reg = hyper_prior_lambda_h(kl, mode="state_dependent", value=0.5, b0_h=0.3, c0_h=0.7)
    oracle = (lam * kl + reg).mean()
    assert torch.allclose(m._hyper_prior_term(tokens), oracle, atol=1e-7)
    # lambda_h*_i = c0_h/(b0_h+KL): bounded above by c0_h/b0_h, strictly positive, KL-decreasing.
    assert (lam <= 0.7 / 0.3 + 1e-6).all() and (lam > 0).all()


def test_state_dependent_lambda_h_grad_flows_to_s():
    # The state-dependent scored term backprops a finite, nonzero gradient into the s tables.
    m = _scored_model(0.5, lambda_h_mode="state_dependent")
    tokens = torch.randint(0, 20, (3, 5)); targets = torch.randint(0, 20, (3, 5))
    _, loss, _ = m(tokens, targets)
    loss.backward()
    g = m.prior_bank.s_mu_embed.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


def test_state_dependent_lambda_h_in_s_e_step_trains_s():
    # The s E-step routing (model.py _refine_s): under s_e_step the lambda_h_mode is threaded into the
    # e_step self-coupling (with R_h via alpha_reg), so a state_dependent lambda_h trains s through the
    # unrolled refine. (model_channel anchors s, so this is the non-collapse regime.)
    cfg = VFE3Config(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=1,
                     n_e_steps=1, e_q_mu_lr=0.5, e_phi_lr=0.0, mass_phi=0.0,
                     mstep_self_coupling_weight=0.0, use_prior_bank=True,
                     lambda_h=1.0, lambda_h_mode="state_dependent", prior_source="model_channel",
                     s_e_step=True, e_s_mu_lr=0.5, seed=0)
    torch.manual_seed(0)
    m = VFEModel(cfg)
    tokens = torch.randint(0, 20, (2, 5)); targets = torch.randint(0, 20, (2, 5))
    _, loss, _ = m(tokens, targets)
    loss.backward()
    g = m.prior_bank.s_mu_embed.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


def test_lambda_h_mode_inert_warning_when_lambda_h_zero():
    # A non-constant lambda_h_mode with lambda_h=0 has no effect (no channel) -> warn.
    with pytest.warns(UserWarning, match="lambda_h_mode"):
        VFE3Config(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5,
                   lambda_h=0.0, lambda_h_mode="state_dependent")


def test_barycenter_r_sets_closed_form_centroid():
    # PriorBank.barycenter_r_ sets r to the moment-matched barycenter of the s tables:
    # r_mu = mean_v s_mu_v, r_sigma = mean_v[s_sigma_v + (s_mu_v - r_mu)^2].
    m = _lr_model(learnable_r=True)
    pb = m.prior_bank
    with torch.no_grad():
        pb.s_mu_embed.copy_(torch.randn_like(pb.s_mu_embed))
        pb.s_sigma_log_embed.copy_(0.3 * torch.randn_like(pb.s_sigma_log_embed))
    pb.barycenter_r_()
    s_mu = pb.s_mu_embed
    s_sig = torch.exp(pb.s_sigma_log_embed).clamp(min=m.cfg.eps)
    r_mu_cf = s_mu.mean(0)
    r_var_cf = (s_sig + (s_mu - r_mu_cf) ** 2).mean(0)
    assert torch.allclose(pb.r_mu, r_mu_cf, atol=1e-6)
    assert torch.allclose(torch.exp(pb.r_sigma_log), r_var_cf, atol=1e-5)


def test_r_update_mode_barycenter_keeps_r_out_of_optimizer():
    # r_update_mode='barycenter' un-grads r (it is set by the closed-form M-step, not AdamW): r is
    # requires_grad=False and NOT grouped, while the s tables still train.
    from vfe3.train import build_optimizer
    cfg = VFE3Config(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=1,
                     n_e_steps=1, use_prior_bank=True, lambda_h=0.5, prior_source="model_channel",
                     learnable_r=True, r_update_mode="barycenter", seed=0)
    torch.manual_seed(0)
    m = VFEModel(cfg)
    assert m.prior_bank.r_mu.requires_grad is False
    assert m.prior_bank.r_sigma_log.requires_grad is False
    opt = build_optimizer(m, cfg)                # frozen r is exempt from the coverage guard
    opt_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    assert id(m.prior_bank.r_mu) not in opt_ids
    assert id(m.prior_bank.s_mu_embed) in opt_ids


def test_r_update_mode_barycenter_inert_warning_without_learnable_r():
    # barycenter with learnable_r=False is a no-op (frozen r never updates) -> warn.
    with pytest.warns(UserWarning, match="r_update_mode='barycenter' has no effect"):
        VFE3Config(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5,
                   lambda_h=0.5, prior_source="model_channel",
                   learnable_r=False, r_update_mode="barycenter")


def test_barycenter_warns_under_non_canonical_divergence():
    # barycenter_r_ is the alpha=1 forward-KL m-projection (reads no cfg); the scored gradient path
    # descends D_alpha at cfg.renyi_order/divergence_family with the lambda_h_mode envelope. Under a
    # non-canonical setting the 'barycenter' and 'gradient' r-updates do not share a fixed point -> warn.
    with pytest.warns(UserWarning, match="exact M-step only for renyi_order=1.0"):
        VFE3Config(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5,
                   lambda_h=0.5, prior_source="model_channel",
                   learnable_r=True, r_update_mode="barycenter", renyi_order=2.0)


def test_barycenter_no_divergence_warning_at_canonical_kl():
    # At the canonical KL objective (renyi_order=1, divergence_family='renyi', lambda_h_mode='constant')
    # the closed-form barycenter IS the exact M-step, so the divergence-mismatch warning must NOT fire.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        VFE3Config(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5,
                   lambda_h=0.5, prior_source="model_channel",
                   learnable_r=True, r_update_mode="barycenter")
    assert not any("exact M-step only for renyi_order" in str(wi.message) for wi in caught)
