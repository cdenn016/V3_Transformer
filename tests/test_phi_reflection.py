import pytest, torch
from vfe3.config import VFE3Config


def _cfg(**over):
    base = dict(gauge_parameterization="phi", gauge_group="glk", embed_dim=4, n_heads=1,
                transport_mode="flat", use_head_mixer=False)
    base.update(over); return VFE3Config(**base)


def test_phi_reflection_field_and_gating():
    assert _cfg().phi_reflection == "off"                                  # default
    for grp, over in (("glk", {}), ("block_glk", {"n_heads": 2}), ("so_k", {})):
        assert _cfg(phi_reflection="metropolis", gauge_group=grp, **over).phi_reflection == "metropolis"
    for grp, over in (("sp", {}), ("so_n", {"group_n": 3, "irrep_spec": [("l0", 1), ("l1", 1)]}),
                      ("tied_block_glk", {"n_heads": 2})):
        with pytest.raises(ValueError):
            _cfg(phi_reflection="metropolis", gauge_group=grp, **over)


def test_phi_reflection_requires_phi_path():
    with pytest.raises(ValueError):        # metropolis reflection is a phi-path feature
        VFE3Config(gauge_parameterization="omega_direct", gauge_group="glk", embed_dim=4, n_heads=1,
                   transport_mode="flat", e_phi_lr=0.0, use_head_mixer=False, phi_reflection="metropolis")


def test_phi_reflection_ste_not_implemented():
    with pytest.raises((NotImplementedError, ValueError), match="ste"):
        _cfg(phi_reflection="ste")


def test_belief_carries_reflection_field():
    from vfe3.belief import BeliefState
    b = BeliefState(mu=torch.zeros(1, 3, 4), sigma=torch.ones(1, 3, 4), phi=torch.zeros(1, 3, 6))
    assert b.reflection is None                                            # default
    b2 = b._replace(reflection=torch.ones(1, 3))
    assert b2.reflection is not None


def test_prior_bank_reflection_sign_gated_and_encode_populates():
    from vfe3.model.model import VFEModel
    m = VFEModel(_cfg(phi_reflection="init_seed", vocab_size=6, max_seq_len=4, n_layers=1))
    assert hasattr(m.prior_bank, "reflection_sign")
    assert m.prior_bank.reflection_sign.shape == (6,)
    assert set(m.prior_bank.reflection_sign.tolist()) <= {1.0, -1.0}
    enc = m.prior_bank.encode(torch.tensor([[0, 1, 2, 3]]))
    assert enc.reflection is not None and enc.reflection.shape == (1, 4)
    # off path: no buffer, no belief field
    m_off = VFEModel(_cfg(vocab_size=6, max_seq_len=4, n_layers=1))
    assert not hasattr(m_off.prior_bank, "reflection_sign")
    assert m_off.prior_bank.encode(torch.tensor([[0, 1]])).reflection is None


def _dense_omega(built):
    """Dense (B, N, N, K, K) Omega from either representation the belief transport returns."""
    from vfe3.geometry.transport import FactoredTransport
    return built.to_dense_omega() if isinstance(built, FactoredTransport) else built


@pytest.mark.parametrize("group_name, group_kw", [("glk", {}), ("block_glk", {"n_heads": 2})])
def test_reflection_fold_matches_R_Omega_R_and_flips_det(group_name, group_kw):
    # The fold-correctness anchor (spec sec 3): the built Omega with a per-token reflection sign
    # must equal an INDEPENDENT R_i @ Omega_base @ R_j recompute to fp5, on BOTH the dense (glk)
    # and factored (block_glk) forward-transport representations.
    from vfe3.inference.e_step import build_belief_transport
    from vfe3.geometry.groups import get_group
    from vfe3.geometry.generators import reflection_element

    K, N = 4, 3
    grp = get_group(group_name)(K=K, **group_kw)
    torch.manual_seed(0)
    phi = 0.2 * torch.randn(1, N, grp.generators.shape[0])
    sign = torch.tensor([[1.0, -1.0, 1.0]])                                  # token 1 reflected

    base = build_belief_transport(phi, grp, transport_mode="flat", gauge_parameterization="phi")
    refl = build_belief_transport(phi, grp, transport_mode="flat", gauge_parameterization="phi",
                                  reflection=sign)
    Om_base = _dense_omega(base)                                             # (1, N, N, K, K)
    Om_refl = _dense_omega(refl)

    # Independent recompute R_i Omega_ij R_j (R = diag(-1,1,...,1) is symmetric, so the same
    # per-token matrices index the query slot i and the key slot j).
    R    = reflection_element(K)
    eye  = torch.eye(K)
    Rmat = torch.where(sign[..., None, None] < 0, R, eye)                    # (1, N, K, K)
    exp  = torch.einsum("bikl,bijlm,bjmn->bijkn", Rmat, Om_base, Rmat)       # (1, N, N, K, K)
    assert torch.allclose(Om_refl, exp, atol=1e-5)

    # det flips for a pair with s_i != s_j (query i=0 -> +1, key j=1 -> -1 -> one R factor).
    assert torch.det(Om_refl[0, 0, 1]) * torch.det(Om_base[0, 0, 1]) < 0

    # reflection=None (default) is byte-identical to the no-reflection call.
    none = build_belief_transport(phi, grp, transport_mode="flat", gauge_parameterization="phi",
                                  reflection=None)
    assert torch.equal(_dense_omega(none), Om_base)


# --- Task 3: reflection reaches the gamma / s-channel and the F-eval --------------------------

def _model(**over):
    """A tiny phi-path model with the model (s) channel present (prior_source='model_channel')."""
    from vfe3.model.model import VFEModel
    base = dict(prior_source="model_channel", family="gaussian_diagonal", decode_mode="diagonal",
                vocab_size=6, max_seq_len=4, n_layers=1)
    base.update(over)
    return VFEModel(_cfg(**base))


def test_gamma_coupling_term_uses_reflection():
    # 3A frame-fidelity (mirrors test_gamma_coupling_term_uses_stored_frame_not_phi_cocycle): with
    # two different reflection signs the forward gamma-coupling energy must DIFFER -- the reflection
    # R_i is folded into the s-channel transport Omega_ij -> R_i Omega_ij R_j. phi is held at zero
    # (exp(phi)=I: the flat cocycle is frame-blind), so ONLY the reflection can move the energy.
    K, N = 4, 3
    m = _model()
    n_gen = m.group.generators.shape[0]
    tok = torch.randint(0, 6, (1, N), generator=torch.Generator().manual_seed(1))
    phi = torch.zeros(1, N, n_gen)
    sign_a = torch.tensor([[1.0,  1.0, 1.0]])                    # all +1 == identity reflection
    sign_b = torch.tensor([[1.0, -1.0, 1.0]])                    # token 1 reflected
    e_a = m._gamma_coupling_term(tok, phi, reflection=sign_a)
    e_b = m._gamma_coupling_term(tok, phi, reflection=sign_b)
    assert not torch.allclose(e_a, e_b, atol=1e-6), \
        "gamma coupling term is blind to the reflection sign (fold not threaded)"
    # all-+1 reflection is byte-identical to reflection=None (pure fold).
    e_none = m._gamma_coupling_term(tok, phi, reflection=None)
    assert torch.allclose(e_a, e_none, atol=1e-7)


def test_refine_s_uses_reflection():
    # 3A frame-fidelity (mirrors test_refine_s_uses_stored_frame_not_phi_cocycle): the s_e_step
    # E-step (_refine_s) must transport the gamma model-coupling by the reflected frame. phi0 is held
    # at zero, lambda_gamma is bumped post-construction so the coupling term has nonzero weight, and
    # the reflection_sign buffer is flipped between two calls -> the refined s must DIFFER.
    K, N, V = 4, 3, 6
    m = _model(lambda_h=1.0, lambda_h_mode="constant", phi_reflection="init_seed", vocab_size=V)
    m.cfg.lambda_gamma = 0.75                                    # nonzero s-coupling weight (post-ctor bump)
    n_gen = m.group.generators.shape[0]
    tok = torch.randint(0, V, (1, N), generator=torch.Generator().manual_seed(1))
    phi0 = torch.zeros(1, N, n_gen)                              # exp(phi0)=I: the flat cocycle is frame-blind

    with torch.no_grad():
        m.prior_bank.reflection_sign.fill_(1.0)                 # all +1
    s_mu1, _ = m._refine_s(tok, phi0)
    with torch.no_grad():
        m.prior_bank.reflection_sign.fill_(1.0)
        m.prior_bank.reflection_sign[tok[0, 1].item()] = -1.0   # flip one token
    s_mu2, _ = m._refine_s(tok, phi0)
    assert not torch.allclose(s_mu1, s_mu2, atol=1e-6), \
        "_refine_s is blind to the reflection sign (not populated / not threaded through e_step)"


def test_free_energy_value_reflects():
    # 3B LOAD-BEARING (Task 4's Metropolis computes DeltaF = F(flipped reflection) - F(reflection),
    # so free_energy_value MUST be reflection-dependent or the whole learnable move is inert). Two
    # beliefs identical but for belief.reflection must give DIFFERENT global F. phi is nonzero so the
    # transport Omega_ij != I and the reflected mean bites.
    from vfe3.geometry.groups import get_group
    from vfe3.inference.e_step import free_energy_value
    from vfe3.belief import BeliefState
    K, N = 4, 3
    grp = get_group("glk")(K=K)
    n_gen = grp.generators.shape[0]
    g = torch.Generator().manual_seed(0)
    mu      = torch.randn(N, K, generator=g)
    sigma   = torch.rand(N, K, generator=g) + 0.5
    phi     = 0.3 * torch.randn(N, n_gen, generator=g)          # NONZERO -> Omega_ij != I for i != j
    mu_p    = torch.randn(N, K, generator=g)
    sigma_p = torch.rand(N, K, generator=g) + 0.5
    sign_a = torch.tensor([1.0,  1.0, 1.0])
    sign_b = torch.tensor([1.0, -1.0, 1.0])
    b_a = BeliefState(mu=mu, sigma=sigma, phi=phi, reflection=sign_a)
    b_b = BeliefState(mu=mu, sigma=sigma, phi=phi, reflection=sign_b)
    F_a = free_energy_value(b_a, mu_p, sigma_p, grp, tau=1.5)
    F_b = free_energy_value(b_b, mu_p, sigma_p, grp, tau=1.5)
    assert torch.isfinite(F_a) and torch.isfinite(F_b)
    assert not torch.allclose(F_a, F_b, atol=1e-6), \
        "free_energy_value ignores belief.reflection -- Task 4 Metropolis DeltaF would be inert"
    # all-+1 reflection is byte-identical to reflection=None (pure fold).
    b_none = BeliefState(mu=mu, sigma=sigma, phi=phi, reflection=None)
    F_none = free_energy_value(b_none, mu_p, sigma_p, grp, tau=1.5)
    assert torch.allclose(F_a, F_none, atol=1e-7)


def test_block_preserves_reflection_end_to_end():
    # 3C: block.py's post-E-step transforms (block_norm / head_mixer / cg_coupling) must PRESERVE
    # belief.reflection. With block_norm active + n_layers>1 + phi_reflection on, the reflection must
    # survive to the returned belief -- the bare BeliefState(mu,sigma,phi) reconstruction dropped it,
    # so layer 2 lost the frame (and omega, for omega_direct). The forward must also run finite.
    m = _model(phi_reflection="init_seed", norm_type_block="mahalanobis", n_layers=2)
    tok = torch.tensor([[0, 1, 2, 3]])
    belief, _ = m.forward_beliefs(tok)
    assert belief.reflection is not None, "block.py dropped belief.reflection past layer 1"
    assert belief.reflection.shape == (1, 4)
    assert torch.isfinite(belief.mu).all()


# --------------------------------------------------------------------------------------------------
# Task 4: LEARNABLE phi reflection -- the shared DeltaF-gated Metropolis sweep flips reflection_sign
# under gauge_parameterization='phi' + phi_reflection='metropolis' (mirrors the omega_direct move in
# tests/test_omega_metropolis.py; the sweep/accept/seed structure is shared, only the per-token flip
# and the trial-frame construction differ).
# --------------------------------------------------------------------------------------------------
import math
from vfe3.model.model import VFEModel


def _metro_model(**over):
    # tiny phi-path model with phi_reflection='metropolis' the move can act on; K<6, single-digit dims.
    # phi is frozen (e_phi_lr=0 -> exp(phi)=I: the flat cocycle is frame-blind) so ONLY the per-token
    # reflection can move the belief-coupling energy -- isolating the sign the move learns.
    base = dict(gauge_parameterization="phi", gauge_group="glk", embed_dim=4, n_heads=1,
                vocab_size=6, max_seq_len=4, n_layers=1, n_e_steps=2, transport_mode="flat",
                e_phi_lr=0.0, use_head_mixer=False, family="gaussian_diagonal", decode_mode="diagonal",
                lambda_gamma=0.0, s_e_step=False, phi_reflection="metropolis")
    base.update(over)
    return VFEModel(VFE3Config(**base))


def test_phi_off_is_noop_no_rng_no_mutation():
    # phi_reflection in {'off','init_seed'} -> metropolis_omega_step is a no-op {}: no buffer mutation,
    # no RNG draw (the generator state is byte-identical after the call).
    for mode in ("off", "init_seed"):
        m = VFEModel(_cfg(phi_reflection=mode, vocab_size=6, max_seq_len=4, n_layers=1))
        had_buf = hasattr(m.prior_bank, "reflection_sign")
        before  = m.prior_bank.reflection_sign.detach().clone() if had_buf else None
        g       = torch.Generator().manual_seed(0)
        g_state = g.get_state().clone()
        stats   = m.metropolis_omega_step(torch.tensor([[0, 1, 2]]), generator=g)
        assert stats == {}                                              # no-op dispatch
        if had_buf:                                                     # init_seed created the buffer
            assert torch.equal(m.prior_bank.reflection_sign, before)    # untouched
        assert torch.equal(g.get_state(), g_state)                      # generator UNUSED (no draw)


def test_phi_exact_delta_f_matches_independent_recompute():
    # LOAD-BEARING exact-DeltaF anchor: the per-token DeltaF the move computes (masked flip of
    # belief.reflection) MUST equal an INDEPENDENT recompute that flips the SOURCE buffer
    # reflection_sign[token] and re-looks-up the per-position sign -- pinning masked-flip == source-flip.
    # Distinct tokens so the fold R_i Omega_ij R_j is nontrivial and DeltaF genuinely nonzero.
    torch.manual_seed(0)
    m = _metro_model(vocab_size=4)
    tok = torch.tensor([[0, 1, 2, 3]])
    tid = 1
    belief, mu_p, sigma_p = m._metropolis_prepare(tok)
    assert belief.reflection is not None                               # reflection actually enters F
    dF_move = m._metropolis_delta_f(belief, mu_p, sigma_p, tok, tid)
    assert dF_move == dF_move                                          # finite (not NaN)
    assert abs(dF_move) > 0.0                                          # genuinely nonzero (distinct tokens)
    # Independent oracle: flip the source buffer row, re-look-up the (fixed-belief) per-position sign,
    # recompute F.
    F_cur = m._metropolis_free_energy(belief, mu_p, sigma_p)
    m._flip_reflection_sign_row(tid)
    relooked = m.prior_bank.reflection_sign[tok]                       # per-position signs from flipped buffer
    F_trial = m._metropolis_free_energy(belief._replace(reflection=relooked), mu_p, sigma_p)
    m._flip_reflection_sign_row(tid)                                   # restore (sign flip is involutory)
    dF_indep = F_trial - F_cur
    assert abs(dF_move - dF_indep) < 1e-5                              # exact-DeltaF anchor (fp5)


def test_phi_downhill_flip_accepted_and_toggles_sign():
    # Seed one token into the reflected sheet, then a sweep is free to flip it: a repeated-token batch
    # gives Omega_ij == R R == I (frame-agnostic) so DeltaF==0 -> accepted, and its sign toggles.
    m = _metro_model()
    with torch.no_grad():
        m.prior_bank.reflection_sign[1] = -1.0                        # seed token 1 into the reflected sheet
    tok = torch.tensor([[1, 1, 1]])                                    # token 1 everywhere
    sign_before = m.prior_bank.reflection_sign[1].item()
    m.cfg.omega_metropolis_temperature = 1e-3                          # low T: downhill move (near-)det. accepted
    stats = m.metropolis_omega_step(tok, generator=torch.Generator().manual_seed(0))
    sign_after = m.prior_bank.reflection_sign[1].item()
    assert stats["proposed"] >= 1
    if stats["accepted"] >= 1:                                         # accepted -> sign toggles
        assert sign_before * sign_after < 0.0


def test_phi_uphill_flip_gated_by_metropolis_acceptance():
    # Pin the STOCHASTIC accept branch (dF>0, gated by u < exp(-dF/T)). Distinct tokens make the
    # fold Omega_ij -> R_i Omega_ij R_j nontrivial so flipping a token's sign genuinely changes F; at
    # this model seed flipping token 0 STRICTLY INCREASES F -- a real uphill proposal.
    torch.manual_seed(1)
    m = _metro_model(vocab_size=4)
    tok = torch.tensor([[0, 1, 2, 3]])
    belief, mu_p, sigma_p = m._metropolis_prepare(tok)
    dF0 = m._metropolis_delta_f(belief, mu_p, sigma_p, tok, 0)
    assert dF0 > 0.0                                                   # genuinely uphill at this seed

    # (1) Tiny temperature: dF0/T so negative that exp(-dF0/T) underflows to 0.0 -> token 0 is
    # deterministically rejected and its source sign is unmutated.
    m.cfg.omega_metropolis_temperature = 1e-6
    sign0_before = m.prior_bank.reflection_sign[0].item()
    m.metropolis_omega_step(tok, generator=torch.Generator().manual_seed(0))
    assert m.prior_bank.reflection_sign[0].item() == sign0_before     # uphill token 0 rejected at tiny T

    # (2) Exact-rule pin: torch.unique sorts ascending, so token 0 is the FIRST proposal, scored
    # against the untouched _metropolis_prepare belief (the SAME dF0). Reproduce the sweep's first
    # torch.rand draw independently and assert the move's own accept/reject == dF<=0 or u<exp(-dF/T).
    m.cfg.omega_metropolis_temperature = 1e-3                          # moderate T: a genuine 0<p<1 draw
    T = m.cfg.omega_metropolis_temperature
    seed = 3
    u0 = torch.rand((), generator=torch.Generator().manual_seed(seed)).item()
    expect_accept0 = (dF0 <= 0.0) or (u0 < math.exp(-dF0 / T))
    assert expect_accept0                                             # this seed's draw clears the threshold
    sign_before = m.prior_bank.reflection_sign[0].item()
    m.metropolis_omega_step(tok, generator=torch.Generator().manual_seed(seed))
    sign_after = m.prior_bank.reflection_sign[0].item()
    accepted0 = (sign_before * sign_after < 0.0)                      # sign toggles iff token 0 was flipped
    assert accepted0 == expect_accept0


def test_phi_metropolis_step_stats_finite():
    torch.manual_seed(0)
    m = _metro_model(vocab_size=4)
    tok = torch.tensor([[0, 1, 2, 3]])
    stats = m.metropolis_omega_step(tok, generator=torch.Generator().manual_seed(0))
    assert stats["proposed"] == 4                                     # one proposal per unique token
    assert 0 <= stats["accepted"] <= stats["proposed"]
    assert "mean_delta_f" in stats and stats["mean_delta_f"] == stats["mean_delta_f"]  # finite


def test_phi_seeded_reproducible():
    tok = torch.tensor([[0, 1, 2, 3, 0, 1]])
    m1 = _metro_model(); m2 = _metro_model()
    with torch.no_grad():                                             # identical F landscape (all tables)
        m2.prior_bank.load_state_dict(m1.prior_bank.state_dict())
    s1 = m1.metropolis_omega_step(tok, generator=torch.Generator().manual_seed(7))
    s2 = m2.metropolis_omega_step(tok, generator=torch.Generator().manual_seed(7))
    assert s1 == s2
    assert torch.equal(m1.prior_bank.reflection_sign, m2.prior_bank.reflection_sign)


def test_phi_train_seam_gated_and_cadence(monkeypatch):
    # The train seam _maybe_metropolis_omega fires when phi_reflection=='metropolis' too (not only the
    # omega mode), honoring the shared omega_metropolis_every cadence.
    m = _metro_model()
    calls = {"n": 0}
    def _spy(token_ids, *, generator):
        calls["n"] += 1; return {}
    monkeypatch.setattr(m, "metropolis_omega_step", _spy)
    from vfe3.train import _maybe_metropolis_omega
    gen = torch.Generator().manual_seed(0)
    tok = torch.tensor([[0, 1, 2]])
    m.cfg.omega_metropolis_every = 2                                  # fires on steps 0 and 2, not 1
    _maybe_metropolis_omega(m, tok, step=0, generator=gen); assert calls["n"] == 1
    _maybe_metropolis_omega(m, tok, step=1, generator=gen); assert calls["n"] == 1
    _maybe_metropolis_omega(m, tok, step=2, generator=gen); assert calls["n"] == 2
    m.cfg.phi_reflection = "off"                                      # off -> never
    _maybe_metropolis_omega(m, tok, step=0, generator=gen); assert calls["n"] == 2
