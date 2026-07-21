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


def test_reflection_certificate_preserves_only_implicit_same_table() -> None:
    from vfe3.geometry.groups import get_group
    from vfe3.geometry.transport import FactoredTransport, transport_mean
    from vfe3.inference.e_step import _apply_reflection, build_belief_transport

    group = get_group("block_glk")(4, 2)
    phi = torch.zeros(1, 3, group.generators.shape[0])
    base = build_belief_transport(phi, group, transport_mode="flat")
    assert isinstance(base, FactoredTransport)
    query_sign = torch.tensor([[1.0, -1.0, 1.0]])
    key_sign = torch.tensor([[-1.0, -1.0, 1.0]])

    implicit = _apply_reflection(base, query_sign)
    explicit_equal = _apply_reflection(
        base, query_sign, key_reflection=query_sign.clone(),
    )
    mismatched = _apply_reflection(base, query_sign, key_reflection=key_sign)

    assert implicit.same_frame_flat_cocycle
    assert not explicit_equal.same_frame_flat_cocycle
    assert not mismatched.same_frame_flat_cocycle

    mu = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4) + 1.0
    self_links = torch.arange(mu.shape[-2])
    assert torch.equal(transport_mean(implicit, mu)[:, self_links, self_links], mu)
    raw_key = torch.einsum("...jlp,...jp->...jl", mismatched.exp_neg_phi, mu)
    expected = torch.einsum("...ikl,...jl->...ijk", mismatched.exp_phi, raw_key)
    assert torch.equal(transport_mean(mismatched, mu), expected)
    assert not torch.equal(expected[:, self_links, self_links], mu)


def test_nonflat_oracle_transport_threads_reflection() -> None:
    from vfe3.belief import BeliefState
    from vfe3.geometry.groups import get_group
    from vfe3.inference.e_step import build_belief_transport, e_step_iteration

    torch.manual_seed(23)
    n, k = 3, 4
    group = get_group("glk")(k)
    n_gen = group.generators.shape[0]
    mu = 0.5 * torch.randn(n, k, dtype=torch.float32)
    sigma = torch.rand(n, k, dtype=torch.float32) + 0.6
    phi = 0.1 * torch.randn(n, n_gen, dtype=torch.float32)
    mu_p = torch.randn(n, k, dtype=torch.float32)
    sigma_p = torch.rand(n, k, dtype=torch.float32) + 0.6
    connection_W = 0.2 * torch.randn(n_gen, k, k, dtype=torch.float32)
    positive = torch.ones(n, dtype=torch.float32)
    mixed = torch.tensor([1.0, -1.0, 1.0], dtype=torch.float32)

    positive_transport = build_belief_transport(
        phi, group, transport_mode="regime_ii", mu=mu,
        connection_W=connection_W, reflection=positive,
    )
    mixed_transport = build_belief_transport(
        phi, group, transport_mode="regime_ii", mu=mu,
        connection_W=connection_W, reflection=mixed,
    )
    assert not torch.allclose(
        _dense_omega(positive_transport), _dense_omega(mixed_transport), atol=1e-6,
    )

    positive_belief = BeliefState(mu=mu, sigma=sigma, phi=phi, reflection=positive)
    mixed_belief = positive_belief._replace(reflection=mixed)
    kwargs = dict(
        e_q_mu_lr=0.05,
        e_q_sigma_lr=0.0,
        e_phi_lr=0.0,
        skip_belief_sigma_update=True,
        gradient_mode="filtering",
        transport_mode="regime_ii",
        connection_W=connection_W,
    )
    positive_step = e_step_iteration(positive_belief, mu_p, sigma_p, group, **kwargs)
    mixed_step = e_step_iteration(mixed_belief, mu_p, sigma_p, group, **kwargs)

    assert not torch.allclose(positive_step.mu, mixed_step.mu, atol=1e-6)


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


def test_free_energy_value_reflects_filtered_keys():
    # Coverage for the FILTERED (keys != None) transport fold in free_energy_value: that branch builds
    # omega via _transport_qk(query.phi, keys.phi) then applies the KEY-ASYMMETRIC reflection
    # (_apply_reflection with an independent key_reflection). The global-path test above never drives
    # this branch (keys defaults to the query belief), so pin it here: with the query sign held at +1
    # (query fold is a no-op), flipping a KEY-slot sign must change the frozen-keys F, and an all-+1
    # keys reflection must equal keys.reflection=None (pure fold). Diagnostic-only path; bites through
    # the transported key mean's component-0 sign (diagonal sigma leaves the covariance invariant).
    from vfe3.geometry.groups import get_group
    from vfe3.inference.e_step import free_energy_value
    from vfe3.belief import BeliefState
    K, N = 4, 3
    grp = get_group("glk")(K=K)
    n_gen = grp.generators.shape[0]
    g = torch.Generator().manual_seed(1)
    mu      = torch.randn(N, K, generator=g)
    sigma   = torch.rand(N, K, generator=g) + 0.5
    phi     = 0.3 * torch.randn(N, n_gen, generator=g)          # NONZERO -> Omega_ij != I for i != j
    mu_p    = torch.randn(N, K, generator=g)
    sigma_p = torch.rand(N, K, generator=g) + 0.5
    q       = BeliefState(mu=mu, sigma=sigma, phi=phi, reflection=torch.ones(N))   # query fixed at +1
    keys_a  = BeliefState(mu=mu, sigma=sigma, phi=phi, reflection=torch.tensor([1.0,  1.0, 1.0]))
    keys_b  = BeliefState(mu=mu, sigma=sigma, phi=phi, reflection=torch.tensor([1.0, -1.0, 1.0]))
    F_a = free_energy_value(q, mu_p, sigma_p, grp, tau=1.5, keys=keys_a)
    F_b = free_energy_value(q, mu_p, sigma_p, grp, tau=1.5, keys=keys_b)
    assert torch.isfinite(F_a) and torch.isfinite(F_b)
    assert not torch.allclose(F_a, F_b, atol=1e-6), \
        "free_energy_value's filtered (keys!=None) branch ignores the key reflection"
    keys_none = BeliefState(mu=mu, sigma=sigma, phi=phi, reflection=None)
    F_none = free_energy_value(q, mu_p, sigma_p, grp, tau=1.5, keys=keys_none)
    assert torch.allclose(F_a, F_none, atol=1e-7)               # all-+1 keys == keys reflection=None


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
    context = m._metropolis_prepare(tok)
    belief = context.belief
    assert belief.reflection is not None                               # reflection actually enters F
    dF_move = m._metropolis_delta_f(context, tid)
    assert dF_move == dF_move                                          # finite (not NaN)
    assert abs(dF_move) > 0.0                                          # genuinely nonzero (distinct tokens)
    # Independent oracle: flip the source buffer row, re-look-up the (fixed-belief) per-position sign,
    # recompute F.
    F_cur = m._metropolis_free_energy(belief, context)
    m._flip_reflection_sign_row(tid)
    relooked = m.prior_bank.reflection_sign[tok]                       # per-position signs from flipped buffer
    F_trial = m._metropolis_free_energy(belief._replace(reflection=relooked), context)
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
    context = m._metropolis_prepare(tok)
    dF0 = m._metropolis_delta_f(context, 0)
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


# --------------------------------------------------------------------------------------------------
# Task 5: end-to-end capstone -- full-scope gamma-on finite forward, det<0 reachability + use, and
# gauge invariance with phi_reflection configured. Mirrors the omega_direct Phase-3 capstone in
# tests/test_omega_direct.py: test_omega_direct_user_target_config_finite_forward and
# test_omega_direct_full_model_gauge_invariance_gamma_on. TEST-ONLY: no source changes.
# --------------------------------------------------------------------------------------------------

def test_phi_reflection_gamma_on_full_scope_finite_forward():
    """CAPSTONE: gauge_parameterization='phi' + phi_reflection='metropolis' with the FULL gamma /
    model-coupling (s) channel ON (lambda_gamma>0, s_e_step=True, gamma_as_beta_prior=True,
    prior_source='model_channel') constructs and runs to a FINITE forward -- proving the reflection is
    threaded through EVERY channel (belief E-step, decode, gamma coupling, s E-step) with NO gate, the
    phi-path analogue of omega_direct's test_omega_direct_user_target_config_finite_forward.

    Deviation from the literal brief: the brief specified family='gaussian_full'/decode_mode='full',
    but s_e_step=True unconditionally requires a DIAGONAL family regardless of gauge_parameterization
    or phi_reflection -- config.py raises "s_e_step=True refines the model channel as a DIAGONAL
    Gaussian (the s/r tables are diagonal by construction), incompatible with family='gaussian_full'"
    at construction (verified empirically: it raises before phi_reflection is ever exercised). This is
    the SAME pre-existing, orthogonal constraint that forces omega_direct's own capstone onto
    gaussian_diagonal; using it here too isolates the phi_reflection integration question this test is
    actually meant to answer. use_prior_bank=True routes decode through the KL-to-prior kernel (the
    pure path this feature targets -- the raw dataclass default is False, the linear-decode ablation,
    under which decode_mode is silently ignored and the converged covariance never reaches the logits)."""
    torch.manual_seed(0)
    m = VFEModel(_cfg(phi_reflection="metropolis", lambda_gamma=0.75, s_e_step=True,
                      gamma_as_beta_prior=True, prior_source="model_channel",
                      family="gaussian_diagonal", decode_mode="diagonal", use_prior_bank=True,
                      vocab_size=6, max_seq_len=4, n_layers=1))
    m.eval()
    tok = torch.randint(0, 6, (1, 4), generator=torch.Generator().manual_seed(2))
    assert torch.isfinite(m(tok)[0]).all()


def test_phi_reflection_det_negative_reachable_and_used_end_to_end():
    """det<0 reachable + USED end-to-end. family='gaussian_full' (+ use_prior_bank=True,
    decode_mode='full', the KL-to-prior decode) so the reflection bites the COVARIANCE, not just the
    mean -- under gaussian_diagonal the fold leaves the (squared) diagonal congruence exactly
    invariant (spec sec 1 efficacy caveat: "the reflection bites only through the mean, sign of
    component 0, not the covariance"), which would make this test near-vacuous.

    Flipping reflection_sign for a subset of tokens changes the decode logits (the reflection is used
    through the FULL forward: encode -> belief E-step -> decode), and the effective per-token frame
    R_i . exp(phi_i) -- built independently from the model's own phi table and group generators, NOT
    via build_belief_transport -- has det<0 for a reflected token."""
    from vfe3.geometry.generators import reflection_element
    K, V, N = 4, 6, 4
    m = VFEModel(_cfg(phi_reflection="metropolis", family="gaussian_full", decode_mode="full",
                      use_prior_bank=True, vocab_size=V, max_seq_len=N, n_layers=1))
    m.eval()
    tok = torch.randint(0, V, (1, N), generator=torch.Generator().manual_seed(2))

    with torch.no_grad():
        m.prior_bank.reflection_sign.fill_(1.0)                         # identity sheet everywhere
        logits_a = m(tok)[0].clone()
        m.prior_bank.reflection_sign[tok[0, 1].item()] = -1.0           # reflect a subset of tokens
        m.prior_bank.reflection_sign[tok[0, 2].item()] = -1.0
        logits_b = m(tok)[0].clone()
    assert not torch.allclose(logits_a, logits_b, atol=1e-6), \
        "reflection sign flip left the decode logits unchanged -- reflection not used end-to-end"

    # The effective frame R_i . exp(phi_i) has det<0 for the reflected token at position 1, built
    # directly (independent of build_belief_transport) from the model's own phi table and generators.
    enc     = m.prior_bank.encode(tok)                                  # raw (unreflected) phi lookup
    R       = reflection_element(K)
    phi_i   = enc.phi[0, 1]
    exp_phi = torch.linalg.matrix_exp(torch.einsum("a,aij->ij", phi_i, m.group.generators))
    assert torch.det(R @ exp_phi) < 0.0


def test_phi_reflection_full_model_gauge_invariance_gamma_on():
    """Gauge-COVARIANCE regression guard, the phi-path analogue of omega_direct's
    test_omega_direct_full_model_gauge_invariance_gamma_on: with the gamma / model-coupling (s)
    channel ON (lambda_gamma>0, s_e_step=True, gamma_as_beta_prior=True,
    prior_source='model_channel') and phi_reflection='metropolis' configured, a global ORTHOGONAL
    gauge transform g of the live model-channel tables (s_mu_embed, r_mu) leaves the decode logits
    invariant to fp64.

    HONEST FRAMING (mandatory): a gauge-INVARIANCE test certifies COVARIANCE -- that no
    gauge-breaking term was introduced by threading phi_reflection through the gamma/s channel -- it
    does NOT by itself certify the reflection is USED. Frame-use is pinned separately by the Task 2/3/4
    tests: test_gamma_coupling_term_uses_reflection, test_refine_s_uses_reflection,
    test_free_energy_value_reflects, and the Metropolis exact-DeltaF anchor
    test_phi_exact_delta_f_matches_independent_recompute.

    The reflection co-transform is EMPIRICALLY INERT here, exactly as it was for omega_direct's
    diagonal-family gamma-on sibling: phi_embed is zeroed (Omega_ij = I for every pair, frame-blind),
    and reflection_sign is left at its construction-time default (all +1 -- 'metropolis' seeds no
    tokens, unlike 'init_seed'), so R_i = I for every token and the fold R_i Omega_ij R_j collapses to
    the identity transport regardless of g. Unlike omega_direct's omega_embed (a genuine per-token
    GL(K) frame table that CAN be co-transformed, U -> gU), reflection_sign is a DISCRETE +/-1 buffer:
    there is no continuous action of g on it to co-transform in the first place, so this test does not
    (and structurally cannot, for a generic continuous g) exercise a nontrivial reflection pattern -- it
    is left at the default identity sheet and simply not touched. This test therefore certifies only
    that phi_reflection's presence in the config does not break the gamma+s pipeline's gauge
    covariance; it says nothing about whether the reflection bites."""
    from vfe3.geometry.generators import generate_son
    torch.manual_seed(0)
    m = VFEModel(_cfg(phi_reflection="metropolis", lambda_gamma=0.75, s_e_step=True,
                      gamma_as_beta_prior=True, prior_source="model_channel",
                      family="gaussian_diagonal", decode_mode="diagonal", use_prior_bank=True,
                      vocab_size=6, max_seq_len=4, n_layers=1))
    with torch.no_grad():
        m.prior_bank.phi_embed.zero_()                                   # frames -> identity (Omega = I)
        m.prior_bank.s_sigma_log_embed.zero_()                           # model-channel Sigma = I
        m.prior_bank.r_sigma_log.zero_()                                 # hyper-prior Sigma = I
        m.prior_bank.r_mu.copy_(torch.tensor([0.1, -0.2, 0.15, -0.05]))  # nonzero r_mu, co-transform not vacuous
        if hasattr(m, "pos_phi_free"):
            m.pos_phi_free.zero_()
    m = m.double()
    m.eval()
    gen_so = generate_son(4).to(torch.float64)                          # skew -> matrix_exp is orthogonal
    c = 0.3 * torch.randn(gen_so.shape[0], generator=torch.Generator().manual_seed(1)).to(gen_so.dtype)
    g = torch.linalg.matrix_exp(torch.einsum("a,aij->ij", c, gen_so))   # g in O(4): g g^T = I
    eye = torch.eye(4, dtype=g.dtype)
    assert torch.allclose(g @ g.transpose(-1, -2), eye, atol=1e-6)      # so(4) => g orthogonal
    tok = torch.randint(0, 6, (1, 4), generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        l0 = m(tok)[0].clone()
        m.prior_bank.s_mu_embed.copy_(torch.einsum("kl,vl->vk", g, m.prior_bank.s_mu_embed))
        m.prior_bank.r_mu.copy_(g @ m.prior_bank.r_mu)
        l1 = m(tok)[0].clone()
    assert float((l0 - l1).abs().max()) < 1e-5
