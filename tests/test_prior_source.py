r"""s->q coupling (Realization A): replace the belief prior with the model channel via prior_source.

The hyper-prior/gamma increments wired a model channel s_i that stays predictively INERT. This
increment makes the model channel actually DRIVE predictions via the SAME-SCALE hierarchical-Bayes
prior of GL(K)_supplementary.tex:1083-1085 (p_i(k_i) = integral p_i(k_i|m_i) s_i(m_i) dm_i). The
user's design choice: REPLACE -- p_i = s_i (the identity conditional, K_model=K) -- realized as a
default-off config toggle `prior_source`. THEORETICAL TENSION (disclosed): the main
Participatory_it_from_bit.tex:1440 instead makes p_i a CROSS-SCALE shadow (the meta-agent's belief
q^(s+1) transported down) and states "s_i does not act through p_i at the same scale"; this toggle
takes the supplementary's same-scale reading, NOT the main manuscript's cross-scale one.

  - prior_source="token" (default): the belief prior is the token table mu_embed/sigma_log_embed,
    EXACTLY as before (byte-identical pure path).
  - prior_source="model_channel": the belief prior is the model-channel s tables. The reroute is
    CONSISTENT across all three places the prior is consumed -- encode (q_i(0) = p_i = s_i), the
    E-step self-coupling target alpha*KL(q_i||p_i), AND the decode per-vocab readout -KL(q||p_v) --
    so the belief lives near s AND is read out against s. Because the s tables are also coupled by
    the gamma model-coupling and lambda_h hyper-prior, the model channel's structure now flows into
    the prior and hence predictions: s -> p -> q. phi (the gauge frame) stays the belief table
    (tied); the s tables carry only the diagonal (mu, sigma).

Oracles (Realization A has two byte-identity floors, so this is NOT the oracle-free trap):
  (1) prior_source="token" is byte-identical to the pre-toggle path (full suite stays green);
  (2) COPY-EQUIVALENCE: prior_source="model_channel" with the s tables COPIED from the belief prior
      tables is byte-identical to prior_source="token" (the reroute reads the same numbers);
  (3) directional: under model_channel the s tables are LIVE (perturbing s changes predictions) and
      mu_embed is DEAD (perturbing it does nothing) -- the swap is exact;
  (4) grad: under model_channel the prior gradient trains s, not mu_embed;
  (5) config validation.
"""

import torch

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel


def _make(prior_source: str = "token", *, lambda_gamma: float = 0.0, lambda_h: float = 0.0,
          seed: int = 0) -> VFEModel:
    cfg = VFE3Config(
        vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=1,
        n_e_steps=1, e_q_mu_lr=0.5, e_phi_lr=0.0, mass_phi=0.0, mstep_self_coupling_weight=0.0,
        prior_source=prior_source, lambda_gamma=lambda_gamma, lambda_h=lambda_h, seed=seed,
    )   # pos_phi="learned" (default) is fine: pos_phi_free is seeded from a dedicated cfg.seed
        # generator (model.py), so it is byte-identical across token vs model_channel models
    torch.manual_seed(seed)              # the model does NOT self-seed; pin RNG before construction
    return VFEModel(cfg)


# ---- (1) pure path ---------------------------------------------------------------------------

def test_default_token_source_has_no_s_tables_and_loss_is_ce():
    m = _make("token")
    assert not hasattr(m.prior_bank, "s_mu_embed")
    tok = torch.randint(0, 20, (3, 5))
    tgt = torch.randint(0, 20, (3, 5))
    _, loss, ce = m(tok, tgt)
    assert torch.allclose(loss, ce)


# ---- (gate) model_channel forces the s tables even without gamma/lambda_h --------------------

def test_model_channel_forces_s_tables():
    m = _make("model_channel")           # gamma=lambda_h=0, but prior_source needs the s tables
    assert hasattr(m.prior_bank, "s_mu_embed")
    assert hasattr(m.prior_bank, "s_sigma_log_embed")


# ---- (2) COPY-EQUIVALENCE oracle: model_channel with s := belief prior == token source --------

def test_copy_equivalence_model_channel_equals_token():
    m_tok = _make("token")               # no s tables
    m_mc = _make("model_channel")        # s tables drawn LAST -> belief tables byte-identical
    assert torch.equal(m_tok.prior_bank.mu_embed, m_mc.prior_bank.mu_embed)
    assert torch.equal(m_tok.prior_bank.sigma_log_embed, m_mc.prior_bank.sigma_log_embed)
    assert torch.equal(m_tok.prior_bank.phi_embed, m_mc.prior_bank.phi_embed)
    with torch.no_grad():                # set the model-channel prior EQUAL to the belief prior
        m_mc.prior_bank.s_mu_embed.copy_(m_mc.prior_bank.mu_embed)
        m_mc.prior_bank.s_sigma_log_embed.copy_(m_mc.prior_bank.sigma_log_embed)
    tok = torch.randint(0, 20, (3, 5))
    tgt = torch.randint(0, 20, (3, 5))
    log_t, loss_t, ce_t = m_tok(tok, tgt)
    log_m, loss_m, ce_m = m_mc(tok, tgt)
    assert torch.equal(log_t, log_m)     # encode + self-coupling + decode all read the same prior
    assert torch.equal(ce_t, ce_m)
    assert torch.equal(loss_t, loss_m)


def test_copy_equivalence_holds_through_mstep_and_multilayer():
    # Completeness (closes the n_e_steps=1/scw=0 gap in the test above): the copy-equivalence floor
    # must also hold THROUGH the M-step self-coupling rebuild (model.py) and the multi-layer
    # prior_handoff fold -- both re-read the (rerouted) encode prior. Build with
    # mstep_self_coupling_weight>0, n_e_steps=2, n_layers=2 and assert byte-identity end-to-end.
    def build(src: str) -> VFEModel:
        cfg = VFE3Config(
            vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=6, n_layers=2,
            n_e_steps=2, e_q_mu_lr=0.5, e_phi_lr=0.0, mass_phi=0.0,
            mstep_self_coupling_weight=0.5, prior_source=src, seed=0,
        )   # pos_phi default "learned" is fine: seed-dedicated pos_phi_free is model-invariant
        torch.manual_seed(0)
        return VFEModel(cfg)
    m_tok = build("token")
    m_mc = build("model_channel")
    with torch.no_grad():
        m_mc.prior_bank.s_mu_embed.copy_(m_mc.prior_bank.mu_embed)
        m_mc.prior_bank.s_sigma_log_embed.copy_(m_mc.prior_bank.sigma_log_embed)
    tok = torch.randint(0, 20, (3, 6))
    tgt = torch.randint(0, 20, (3, 6))
    log_t, loss_t, _ = m_tok(tok, tgt)       # loss_t includes the M-step self-coupling term
    log_m, loss_m, _ = m_mc(tok, tgt)
    assert torch.equal(log_t, log_m)
    assert torch.equal(loss_t, loss_m)       # the rerouted self-coupling rebuild stays byte-identical


# ---- (3) directional: s is the LIVE prior, mu_embed is DEAD, under model_channel --------------

def test_model_channel_s_is_live_mu_embed_is_dead():
    m = _make("model_channel")
    tok = torch.randint(0, 20, (3, 5))
    tgt = torch.randint(0, 20, (3, 5))
    log0 = m(tok, tgt)[0]
    torch.manual_seed(1)
    with torch.no_grad():                # perturb the LIVE prior -> predictions move
        m.prior_bank.s_mu_embed.add_(torch.randn_like(m.prior_bank.s_mu_embed))
    log1 = m(tok, tgt)[0]
    assert not torch.equal(log0, log1)
    # mu_embed is bypassed under model_channel: perturbing it does nothing
    log_a = m(tok, tgt)[0]
    torch.manual_seed(2)
    with torch.no_grad():
        m.prior_bank.mu_embed.add_(torch.randn_like(m.prior_bank.mu_embed))
    log_b = m(tok, tgt)[0]
    assert torch.equal(log_a, log_b)


# ---- (4) grad: the prior gradient trains s, not mu_embed, under model_channel -----------------

def test_model_channel_grad_trains_s_not_mu_embed():
    m = _make("model_channel")
    tok = torch.randint(0, 20, (3, 5))
    tgt = torch.randint(0, 20, (3, 5))
    _, loss, _ = m(tok, tgt)
    loss.backward()
    g_s = m.prior_bank.s_mu_embed.grad
    assert g_s is not None and torch.isfinite(g_s).all() and g_s.abs().sum() > 0
    assert m.prior_bank.mu_embed.grad is None        # mu_embed dead under model_channel -> no grad


# ---- (5) config validation -------------------------------------------------------------------

def test_config_invalid_prior_source_raises():
    try:
        VFE3Config(vocab_size=20, embed_dim=4, n_heads=2, prior_source="bogus")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown prior_source")


# ---- (6) end-to-end: the s prior is in the optimizer and actually trains under model_channel ---

def test_model_channel_optimizer_steps_s_prior():
    # The forward/backward tests only prove s gets a GRADIENT; training also needs the optimizer to
    # STEP it. build_optimizer's exact-coverage guard raises on any ungrouped parameter, so the s
    # tables (the live prior under model_channel) MUST be in a param group, else the model cannot
    # train its prior. This is the "it actually drives predictions" oracle.
    from vfe3.train import build_optimizer
    m = _make("model_channel")
    opt = build_optimizer(m, m.cfg)                          # must NOT raise (s tables covered)
    opt_params = {id(p) for g in opt.param_groups for p in g["params"]}
    assert id(m.prior_bank.s_mu_embed) in opt_params
    assert id(m.prior_bank.s_sigma_log_embed) in opt_params
    tok = torch.randint(0, 20, (3, 5))
    tgt = torch.randint(0, 20, (3, 5))
    before_mu = m.prior_bank.s_mu_embed.detach().clone()
    before_sig = m.prior_bank.s_sigma_log_embed.detach().clone()
    _, loss, _ = m(tok, tgt)
    loss.backward()
    opt.step()
    assert not torch.equal(before_mu, m.prior_bank.s_mu_embed)          # the live prior MEAN moved
    assert not torch.equal(before_sig, m.prior_bank.s_sigma_log_embed)  # ...and its (log) variance
