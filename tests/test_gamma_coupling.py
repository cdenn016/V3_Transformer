r"""Model-coupling channel, second increment: lambda_gamma * mean_i F_red^s_i.

Manuscript Participatory_it_from_bit.tex eq:pointwise_free_energy (lines 1241-1249): the
canonical two-tier free energy carries a model-coupling block
    sum_ij [ gamma_ij KL(s_i || Omega_tilde_ij s_j) + tau_g gamma_ij log(gamma_ij/pi^s_ij) ]
with gamma_ij = softmax_j(log pi^s - E^s/tau_g) the optimal model-channel attention and, at the
optimum, the block equals the reduced (envelope) form -tau_g log Z^s_i. This increment wires it
end-to-end at the smallest scope, mirroring the hyper-prior increment (test_hyperprior.py): the
gamma block is added to the TRAINING LOSS (not the E-step F), reusing pairwise_energy +
reduced_free_energy on the model-channel s tables, with TIED transport Omega_tilde = Omega built
from the CONVERGED belief gauge frame out.phi.

DETACH CONTRACT (the predictive-inertness guarantee). Omega is built from out.phi.DETACHED, so the
gamma gradient flows ONLY to the s tables: the forward (logits/ce) is byte-identical to the
gamma=0 path and the model channel is predictively INERT this increment (s does not feed q). This
deliberately SEVERS the phi<-gamma coupling that full tied transport would carry in the canonical
E-step F; restoring it (or keeping it severed) is part of the deferred s->q design, NOT this term.

These tests pin the contracts:
  (1) default-off (gamma=0, lambda_h=0): no s/r tables, loss == ce (the pure path);
  (2) gamma>0 alone creates the s tables but NOT the r tables (r is hyper-prior-only);
  (3) linearity/isolation oracle: loss_w - loss_0 == w * (recomputed gamma term), the term
      recomputed from the model's s tables + Omega(encode().phi) by the same recipe forward uses.
      This shares the transport/energy PRIMITIVES with the impl, so it pins WIRING (right s tables,
      right phi), ISOLATION (additive, ce-independent), REDUCTION (mean), per-head handling, and
      LINEARITY -- it does NOT independently re-derive E_s == D(s_i||Omega s_j) from first
      principles; test (9) does that;
  (4) predictive inertness: mutating the s tables leaves logits/ce byte-identical;
  (5) detach contract on the REAL forward: gamma adds nothing to phi_embed.grad (equal across
      gamma=0/gamma=w), while the s tables receive grad only at gamma>0;
  (6) envelope identity for the gamma channel: reduced block == explicit softmax assembly;
  (7) self-zero: identity transport (phi=0) + equal s tables => E_s=0 => term 0;
  (8) config validation + the tau_gamma property;
  (9) formula-independent energy check at NONZERO phi (Omega != I): the per-pair gamma energy
      equals an analytic diagonal-Gaussian KL(s_i || Omega_ij s_j) computed by hand, so E_s is the
      RIGHT divergence of the RIGHT (transported) tensors -- the one check the shared-primitive
      oracle (3) cannot make, and the one that exercises real transport content.
"""

import math

import torch

from vfe3.attention_prior import attention_log_prior
from vfe3.config import VFE3Config
from vfe3.families.gaussian import DiagonalGaussian
from vfe3.free_energy import attention_tau, attention_weights, pairwise_energy, reduced_free_energy
from vfe3.geometry.transport import (
    compute_transport_operators,
    transport_covariance,
    transport_mean,
)
from vfe3.model.model import VFEModel


def _make_model(
    lambda_gamma:   float = 0.0,
    lambda_h:       float = 0.0,
    *,
    seed:                  int   = 0,
    kappa_gamma:           float = 1.0,
    gamma_attention_prior: str   = "causal",
) -> VFEModel:
    cfg = VFE3Config(
        vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=1,
        n_e_steps=1, e_q_mu_lr=0.5, e_phi_lr=0.0, mass_phi=0.0,
        mstep_self_coupling_weight=0.0,
        lambda_h=lambda_h, lambda_gamma=lambda_gamma,
        kappa_gamma=kappa_gamma, gamma_attention_prior=gamma_attention_prior, seed=seed,
        pos_phi="none",   # the _gamma_term oracle assumes out.phi == encode().phi (e_phi_lr=0), which
                          # pos_phi COMPOSITION breaks (pos_phi folds into the frame before transport),
                          # independent of pos_phi_free's RNG -- so this stays on the pure no-PE path
    )
    torch.manual_seed(seed)              # the model does NOT self-seed; pin RNG before construction
    return VFEModel(cfg)


def _gamma_term(model: VFEModel, tokens: torch.Tensor) -> torch.Tensor:
    r"""Independent oracle: recompute mean_i F_red^s_i from the model's s tables + Omega(encode().phi),
    by the SAME recipe forward uses. Relies on e_phi_lr==0 so out.phi == encode().phi (the E-step
    leaves the gauge frame untouched, e_step.py:324), letting the oracle skip re-running vfe_stack."""
    cfg = model.cfg
    assert cfg.e_phi_lr == 0.0, "oracle assumes e_phi_lr==0 so out.phi == encode().phi"
    pb = model.prior_bank
    s_mu, s_sigma = pb.encode_s(tokens)                                   # (B, N, K)
    phi = pb.encode(tokens).phi                                          # (B, N, n_gen) == out.phi at e_phi_lr=0
    omega = compute_transport_operators(phi.detach(), model.group)["Omega"]   # (B, N, N, K, K) tied + detached
    s_mu_t = transport_mean(omega, s_mu)                                 # (B, N, N, K)
    s_sigma_t = transport_covariance(omega, s_sigma)                     # (B, N, N, K) diagonal sandwich
    e_s = pairwise_energy(
        DiagonalGaussian(s_mu, s_sigma), DiagonalGaussian(s_mu_t, s_sigma_t),
        alpha=cfg.renyi_order, kl_max=cfg.kl_max, eps=cfg.eps,
        divergence_family=cfg.divergence_family, irrep_dims=model.group.irrep_dims,
    )                                                                    # (B, H, N, N)
    n = tokens.shape[1]
    log_prior = attention_log_prior(
        cfg.gamma_attention_prior, n, n, device=tokens.device, dtype=s_mu.dtype,
    )                                                                    # (N, N)
    gamma_tau = attention_tau(cfg.kappa_gamma, model.group.irrep_dims)   # mirrors the impl (group-aware)
    return reduced_free_energy(e_s, tau=gamma_tau, log_prior=log_prior).mean()


# ---- (1) pure path ---------------------------------------------------------------------------

def test_default_off_no_tables_and_loss_is_ce():
    model = _make_model(0.0, 0.0)
    assert not hasattr(model.prior_bank, "s_mu_embed")
    assert not hasattr(model.prior_bank, "r_mu")
    tokens = torch.randint(0, 20, (3, 5))
    targets = torch.randint(0, 20, (3, 5))
    _, loss, ce = model(tokens, targets)
    assert torch.allclose(loss, ce)


# ---- (2) the s-table gate splits from the r-table (hyper-prior) gate -------------------------

def test_gamma_creates_s_tables_but_not_r_tables():
    model = _make_model(lambda_gamma=0.5, lambda_h=0.0)
    assert hasattr(model.prior_bank, "s_mu_embed")          # model channel s tables exist for gamma
    assert hasattr(model.prior_bank, "s_sigma_log_embed")
    assert not hasattr(model.prior_bank, "r_mu")            # r is hyper-prior-only (lambda_h>0)


# ---- (3) GOLD oracle: linear in lambda_gamma, against an independent recomputation ----------

def test_gamma_linear_against_independent_recomputation():
    w = 0.5
    model_0 = _make_model(0.0)
    model_w = _make_model(w)
    # belief tables byte-identical (s tables are drawn LAST in PriorBank.__init__, same seed)
    assert torch.equal(model_0.prior_bank.mu_embed, model_w.prior_bank.mu_embed)
    assert torch.equal(model_0.prior_bank.phi_embed, model_w.prior_bank.phi_embed)
    # make the s tables clearly distinct so the gamma term is robustly non-vacuous
    torch.manual_seed(123)
    with torch.no_grad():
        model_w.prior_bank.s_mu_embed.normal_(0.0, 0.5)
    tokens = torch.randint(0, 20, (3, 5))
    targets = torch.randint(0, 20, (3, 5))
    _, loss_0, _ = model_0(tokens, targets)
    _, loss_w, _ = model_w(tokens, targets)
    g = _gamma_term(model_w, tokens)
    assert g > 1e-6                                          # non-vacuous
    assert torch.allclose(loss_w - loss_0, w * g, atol=1e-6)


# ---- (4) predictive inertness: s tables never reach the prediction path ----------------------

def test_gamma_predictively_inert_logits_and_ce_independent_of_s():
    model = _make_model(0.5)
    tokens = torch.randint(0, 20, (3, 5))
    targets = torch.randint(0, 20, (3, 5))
    logits1 = model(tokens)
    _, _, ce1 = model(tokens, targets)
    with torch.no_grad():
        model.prior_bank.s_mu_embed.normal_(0.0, 1.0)       # perturb the model channel
        model.prior_bank.s_sigma_log_embed.normal_(0.0, 1.0)
    logits2 = model(tokens)
    _, _, ce2 = model(tokens, targets)
    assert torch.equal(logits1, logits2)                    # prediction unchanged
    assert torch.equal(ce1, ce2)                            # pure CE unchanged


# ---- (5) detach contract on the REAL forward: gamma touches s, not phi ------------------------

def test_gamma_grad_flows_to_s_not_phi():
    w = 0.5
    model_0 = _make_model(0.0)
    model_w = _make_model(w)
    tokens = torch.randint(0, 20, (3, 5))
    targets = torch.randint(0, 20, (3, 5))
    _, loss_0, _ = model_0(tokens, targets)
    loss_0.backward()
    _, loss_w, _ = model_w(tokens, targets)
    loss_w.backward()
    # gamma added NOTHING to the gauge-frame gradient (detached Omega) nor the belief-mean gradient
    assert torch.equal(model_0.prior_bank.phi_embed.grad, model_w.prior_bank.phi_embed.grad)
    assert torch.equal(model_0.prior_bank.mu_embed.grad, model_w.prior_bank.mu_embed.grad)
    # the s tables train only at gamma>0
    assert not hasattr(model_0.prior_bank, "s_mu_embed")
    g_s = model_w.prior_bank.s_mu_embed.grad
    assert g_s is not None and torch.isfinite(g_s).all() and g_s.abs().sum() > 0


# ---- (6) envelope identity for the gamma channel ---------------------------------------------

def test_gamma_envelope_identity():
    model = _make_model(0.5)
    tokens = torch.randint(0, 20, (2, 5))
    cfg = model.cfg
    pb = model.prior_bank
    s_mu, s_sigma = pb.encode_s(tokens)
    phi = pb.encode(tokens).phi
    omega = compute_transport_operators(phi.detach(), model.group)["Omega"]
    e_s = pairwise_energy(
        DiagonalGaussian(s_mu, s_sigma),
        DiagonalGaussian(transport_mean(omega, s_mu), transport_covariance(omega, s_sigma)),
        alpha=cfg.renyi_order, kl_max=cfg.kl_max, eps=cfg.eps,
        divergence_family=cfg.divergence_family, irrep_dims=model.group.irrep_dims,
    )                                                                    # (B, H, N, N)
    n = tokens.shape[1]
    log_prior = attention_log_prior(cfg.gamma_attention_prior, n, n, dtype=s_mu.dtype)
    gamma_tau = attention_tau(cfg.kappa_gamma, model.group.irrep_dims)   # mirrors the impl (group-aware)
    reduced = reduced_free_energy(e_s, tau=gamma_tau, log_prior=log_prior)                 # (B, H, N)
    gamma = attention_weights(e_s, tau=gamma_tau, log_prior=log_prior)                     # (B, H, N, N)
    pi = torch.softmax(log_prior, dim=-1)
    explicit = (gamma * e_s).sum(-1) + gamma_tau * (
        gamma * (torch.log(gamma.clamp(min=1e-12)) - torch.log(pi.clamp(min=1e-12)))
    ).sum(-1)
    assert torch.allclose(reduced, explicit, atol=1e-5)


# ---- (7) self-zero: identity transport + equal s tables => term 0 ----------------------------

def test_gamma_self_zero_under_identity_transport():
    model = _make_model(0.5)
    pb = model.prior_bank
    with torch.no_grad():
        pb.phi_embed.zero_()                                # phi=0 => Omega_ij = I
        pb.s_mu_embed.copy_(pb.s_mu_embed[0:1].expand_as(pb.s_mu_embed))        # every token's s equal
        pb.s_sigma_log_embed.copy_(pb.s_sigma_log_embed[0:1].expand_as(pb.s_sigma_log_embed))
    tokens = torch.randint(0, 20, (2, 5))
    g = _gamma_term(model, tokens)
    assert torch.allclose(g, torch.zeros_like(g), atol=1e-6)


# ---- (8) config validation + the tau_gamma property ------------------------------------------

def test_tau_gamma_property():
    cfg = VFE3Config(vocab_size=20, embed_dim=8, n_heads=2, kappa_gamma=2.0)
    assert math.isclose(cfg.tau_gamma, 2.0 * (cfg.d_head ** 0.5))

def test_config_negative_gamma_coupling_raises():
    try:
        VFE3Config(vocab_size=20, embed_dim=4, n_heads=2, lambda_gamma=-0.1)
    except ValueError:
        return
    raise AssertionError("expected ValueError for lambda_gamma < 0")

def test_config_nonpositive_kappa_gamma_raises():
    try:
        VFE3Config(vocab_size=20, embed_dim=4, n_heads=2, kappa_gamma=0.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for kappa_gamma <= 0")

def test_config_invalid_gamma_attention_prior_raises():
    try:
        VFE3Config(vocab_size=20, embed_dim=4, n_heads=2, gamma_attention_prior="bogus")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown gamma_attention_prior")


# ---- (9) formula-independent energy check at NONZERO phi (Omega != I) -------------------------

def test_gamma_energy_equals_analytic_kl_at_nonzero_phi():
    # The one check the shared-primitive oracle (3) cannot make: the per-pair gamma energy E_s[i,j]
    # equals an analytic diagonal-Gaussian KL(s_i || Omega_ij s_j) computed by hand, at NONZERO phi
    # (so Omega_01 != I -- this also exercises real transport content, unlike the phi~0 fixtures).
    # Single-block glk so Omega is one full KxK operator (no per-head split). The analytic side
    # builds the transported key via a PLAIN matmul on the Omega_01 slice and writes out the diagonal
    # KL formula explicitly -- independent of transport_mean/transport_covariance and of the renyi
    # kernel -- so a wrong orientation (Omega^T or Omega_ji) or a wrong divergence direction fails.
    cfg = VFE3Config(vocab_size=20, embed_dim=2, n_heads=1, gauge_group="glk", max_seq_len=4,
                     n_layers=1, n_e_steps=1, e_q_mu_lr=0.5, e_phi_lr=0.0, lambda_gamma=0.5, seed=0)
    assert cfg.renyi_order == 1.0                            # the analytic is the KL (alpha=1) limit
    torch.manual_seed(0)
    model = VFEModel(cfg)
    pb = model.prior_bank
    with torch.no_grad():                                   # distinct s + nonzero phi => Omega_01 != I
        pb.s_mu_embed.normal_(0.0, 0.7)
        pb.s_sigma_log_embed.normal_(0.0, 0.3)
        pb.phi_embed.normal_(0.0, 0.2)
    tokens = torch.tensor([[3, 7]])
    s_mu, s_sigma = pb.encode_s(tokens)                      # (1, 2, K)
    phi = pb.encode(tokens).phi                             # (1, 2, n_gen)
    omega = compute_transport_operators(phi.detach(), model.group)["Omega"]   # (1, 2, 2, K, K)
    e_s = pairwise_energy(
        DiagonalGaussian(s_mu, s_sigma),
        DiagonalGaussian(transport_mean(omega, s_mu), transport_covariance(omega, s_sigma)),
        alpha=cfg.renyi_order, kl_max=cfg.kl_max, eps=cfg.eps,
        divergence_family=cfg.divergence_family, irrep_dims=model.group.irrep_dims,
    )                                                        # (1, N, N) single-block
    o01 = omega[0, 0, 1]                                     # (K, K) the (i=0, j=1) operator
    mu0, sig0 = s_mu[0, 0], s_sigma[0, 0]                    # query s_0
    mu1, sig1 = s_mu[0, 1], s_sigma[0, 1]                    # key   s_1
    mu1_eff = o01 @ mu1                                      # Omega_01 @ mu_1 (plain matmul)
    sig1_eff = torch.diagonal(o01 @ torch.diag(sig1) @ o01.T)   # diag(Omega Sigma Omega^T)
    analytic = 0.5 * (sig0 / sig1_eff + (mu1_eff - mu0) ** 2 / sig1_eff - 1.0
                      + torch.log(sig1_eff / sig0)).sum()   # KL(N(mu0,sig0) || N(mu1_eff,sig1_eff))
    assert torch.allclose(e_s[0, 0, 1], analytic, atol=1e-5)


# ---- (10) gamma temperature is group-aware: sqrt(K) on a SINGLE-BLOCK group, not sqrt(d_head) ----

def test_gamma_tau_is_group_aware_for_single_block_group():
    r"""The gamma model-coupling softmax temperature must span the SAME dimension its energy
    accumulates over (the rule encoded in free_energy.attention_tau). For a SINGLE-BLOCK group
    (glk reports irrep_dims=[K]) the per-pair energy E^s_ij = D(s_i||Omega_ij s_j) accumulates over
    the FULL K, so tau = kappa_gamma*sqrt(K) -- NOT kappa_gamma*sqrt(d_head) = kappa_gamma*sqrt(K/n_heads).
    The belief beta channel already uses attention_tau everywhere; only the gamma channel used the
    scalar cfg.tau_gamma (sqrt(d_head)), which understates the temperature by sqrt(n_heads) on a
    single-block group. This pins the gamma term against an INDEPENDENT oracle whose tau is the literal
    kappa_gamma*sqrt(K) (NOT a call to attention_tau), on glk with n_heads=2 (where sqrt(d_head) != sqrt(K)),
    so it fails against the sqrt(d_head) bug and passes only for the group-aware temperature."""
    w = 0.5
    K = 4                                                    # embed_dim; glk single block of size K
    base = dict(
        vocab_size=20, embed_dim=K, n_heads=2, gauge_group="glk", max_seq_len=5, n_layers=1,
        n_e_steps=1, e_q_mu_lr=0.5, e_phi_lr=0.0, mass_phi=0.0, mstep_self_coupling_weight=0.0,
        lambda_h=0.0, kappa_gamma=1.0, gamma_attention_prior="causal", pos_phi="none", seed=0,
    )
    torch.manual_seed(0); model_0 = VFEModel(VFE3Config(lambda_gamma=0.0, **base))
    torch.manual_seed(0); model_w = VFEModel(VFE3Config(lambda_gamma=w,   **base))
    assert model_w.group.irrep_dims == [K]                  # single block: d_energy = K, d_head = K // 2
    # belief tables byte-identical (s tables drawn LAST), then make s clearly distinct -> non-vacuous term
    assert torch.equal(model_0.prior_bank.mu_embed, model_w.prior_bank.mu_embed)
    torch.manual_seed(123)
    with torch.no_grad():
        model_w.prior_bank.s_mu_embed.normal_(0.0, 0.5)
    tokens  = torch.randint(0, 20, (3, 5))
    targets = torch.randint(0, 20, (3, 5))
    _, loss_0, _ = model_0(tokens, targets)
    _, loss_w, _ = model_w(tokens, targets)

    # independent oracle: same recipe as the forward, but tau written as the literal sqrt(K) single-block value
    cfg = model_w.cfg
    pb = model_w.prior_bank
    s_mu, s_sigma = pb.encode_s(tokens)
    phi = pb.encode(tokens).phi
    omega = compute_transport_operators(phi.detach(), model_w.group)["Omega"]
    e_s = pairwise_energy(
        DiagonalGaussian(s_mu, s_sigma),
        DiagonalGaussian(transport_mean(omega, s_mu), transport_covariance(omega, s_sigma)),
        alpha=cfg.renyi_order, kl_max=cfg.kl_max, eps=cfg.eps,
        divergence_family=cfg.divergence_family, irrep_dims=model_w.group.irrep_dims,
    )
    n = tokens.shape[1]
    log_prior = attention_log_prior(cfg.gamma_attention_prior, n, n, dtype=s_mu.dtype)
    tau_correct = cfg.kappa_gamma * (K ** 0.5)              # literal: single-block energy spans K
    g = reduced_free_energy(e_s, tau=tau_correct, log_prior=log_prior).mean()
    assert g > 1e-6                                          # non-vacuous
    assert torch.allclose(loss_w - loss_0, w * g, atol=1e-6)

    # the buggy sqrt(d_head) temperature genuinely differs here (guards against a coincident fixture)
    tau_bug = cfg.kappa_gamma * (cfg.d_head ** 0.5)         # = sqrt(2) != sqrt(4)
    g_bug = reduced_free_energy(e_s, tau=tau_bug, log_prior=log_prior).mean()
    assert not torch.allclose(g, g_bug, atol=1e-6)
    # the group-aware temperature equals the literal (sanity: attention_tau over [K] is sqrt(K))
    assert math.isclose(attention_tau(cfg.kappa_gamma, model_w.group.irrep_dims), tau_correct)
