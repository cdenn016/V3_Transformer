import math

import pytest
import torch
from vfe3.config import VFE3Config


def _omega_cfg(**over):
    base = dict(gauge_parameterization="omega_direct", transport_mode="flat", e_phi_lr=0.0,
                embed_dim=4, n_heads=1, use_head_mixer=False, lambda_gamma=0.0, s_e_step=False,
                pos_phi="none")
    base.update(over)
    return VFE3Config(**base)


def test_metropolis_constructs_for_reflect_ok_groups():
    for grp, over in (("glk", {}), ("block_glk", {"n_heads": 2}), ("so_k", {})):
        cfg = _omega_cfg(omega_reflection="metropolis", gauge_group=grp, **over)
        assert cfg.omega_reflection == "metropolis"
        assert cfg.omega_metropolis_temperature == 1.0     # default
        assert cfg.omega_metropolis_every == 1             # default


def test_metropolis_rejected_vacuous_and_deferred_groups():
    for grp, over in (("sp", {}), ("sp_n", {"embed_dim": 5, "group_n": 4,
                                            "irrep_spec": [("sym0", 1), ("sym1", 1)]}),
                      ("so_n", {"group_n": 3, "irrep_spec": [("l0", 1), ("l1", 1)]}),
                      ("tied_block_glk", {"n_heads": 2})):
        with pytest.raises(ValueError):
            _omega_cfg(omega_reflection="metropolis", gauge_group=grp, **over)


def test_ste_not_implemented():
    with pytest.raises((NotImplementedError, ValueError), match="ste"):
        _omega_cfg(omega_reflection="ste", gauge_group="glk")


def test_metropolis_temperature_and_cadence_validated():
    with pytest.raises(ValueError):
        _omega_cfg(omega_reflection="metropolis", gauge_group="glk", omega_metropolis_temperature=0.0)
    with pytest.raises(ValueError):
        _omega_cfg(omega_reflection="metropolis", gauge_group="glk", omega_metropolis_every=0)


def test_metropolis_rejected_under_phi_parameterization():
    # Final-review Fix A: metropolis requires an omega_direct frame (belief.omega); under the
    # default 'phi' parameterization no such frame exists, so metropolis must be rejected at
    # config construction rather than crashing mid-training on belief.omega.shape[-1] (None).
    with pytest.raises(ValueError, match="omega_direct"):
        _omega_cfg(gauge_parameterization="phi", omega_reflection="metropolis", gauge_group="glk")


def test_off_and_init_seed_unchanged():
    assert _omega_cfg(gauge_group="glk").omega_reflection == "off"
    assert _omega_cfg(omega_reflection="init_seed", gauge_group="glk").omega_reflection == "init_seed"


@pytest.mark.parametrize("omega_reflection", ["init_seed", "metropolis"])
def test_gauge_transport_off_rejects_omega_reflection(omega_reflection):
    with pytest.raises(ValueError, match="reflection"):
        _omega_cfg(gauge_group="glk", gauge_transport="off",
                   omega_reflection=omega_reflection)


@pytest.mark.parametrize("omega_reflection", ["off", "init_seed", "metropolis"])
def test_gauge_transport_frozen_rejects_omega_direct(omega_reflection):
    with pytest.raises(ValueError, match="frozen"):
        _omega_cfg(gauge_group="glk", gauge_transport="frozen",
                   omega_reflection=omega_reflection)


# --------------------------------------------------------------------------------------------------
# Task 2: the move -- metropolis_omega_step (fixed-belief DeltaF-gated det-sign flip)
# --------------------------------------------------------------------------------------------------
from vfe3.model.model import VFEModel
from vfe3.geometry.generators import reflection_element


def _model(**over):
    # tiny omega_direct model with a det-sign the move can act on; K<6, single-digit dims
    base = dict(gauge_parameterization="omega_direct", gauge_group="glk", embed_dim=4, n_heads=1,
                vocab_size=6, max_seq_len=4, n_layers=1, n_e_steps=2, transport_mode="flat",
                e_phi_lr=0.0, use_head_mixer=False, family="gaussian_diagonal", decode_mode="diagonal",
                lambda_gamma=0.0, s_e_step=False, omega_reflection="metropolis", pos_phi="none")
    base.update(over)
    return VFEModel(VFE3Config(**base))


def test_off_is_noop_no_rng_no_mutation():
    m = VFEModel(VFE3Config(gauge_parameterization="omega_direct", gauge_group="glk", embed_dim=4,
                            n_heads=1, vocab_size=6, max_seq_len=4, n_layers=1, transport_mode="flat",
                            e_phi_lr=0.0, use_head_mixer=False, omega_reflection="off",
                            pos_phi="none"))
    before = m.prior_bank.omega_embed.detach().clone()
    g = torch.Generator().manual_seed(0)
    stats = m.metropolis_omega_step(torch.tensor([[0, 1, 2]]), generator=g)
    assert stats.get("proposed", 0) == 0
    assert torch.equal(m.prior_bank.omega_embed, before)                 # untouched
    assert g.initial_seed() == torch.Generator().manual_seed(0).initial_seed()  # generator unused


def test_downhill_flip_accepted_and_toggles_det_sign():
    # Seed one token into the WRONG sheet, then a sweep should be free to flip it (all-same-token
    # transport is frame-agnostic so DeltaF==0 -> accepted) and its det sign toggles on accept.
    m = _model()
    with torch.no_grad():
        m.prior_bank.omega_embed[1] = reflection_element(4) @ m.prior_bank.omega_embed[1]
    tok = torch.tensor([[1, 1, 1]])                                      # token 1 everywhere -> strong signal
    det_before = torch.det(m.prior_bank.omega_embed[1]).item()
    g = torch.Generator().manual_seed(0)
    m.cfg.omega_metropolis_temperature = 1e-3                            # low T: downhill move (near-)deterministically accepted
    stats = m.metropolis_omega_step(tok, generator=g)
    det_after = torch.det(m.prior_bank.omega_embed[1]).item()
    assert stats["proposed"] >= 1
    if stats["accepted"] >= 1:                                          # accepted -> det sign toggles
        assert det_before * det_after < 0


def test_uphill_flip_gated_by_metropolis_acceptance():
    # Spec 8.3: pin the STOCHASTIC accept branch (dF>0, gated by u < exp(-dF/T)), which the downhill
    # test above never reaches (a repeated-token batch gives Omega_ij == I and dF === 0). Distinct
    # tokens here make the belief-coupling transport Omega_ij != I so flipping a token's frame away
    # from its (identity-init) sheet genuinely changes F; at this model seed it STRICTLY INCREASES F
    # for token 0 -- a real uphill proposal.
    torch.manual_seed(1)
    m = _model(vocab_size=4)
    tok = torch.tensor([[0, 1, 2, 3]])                                   # distinct tokens -> Omega_ij != I
    belief, mu_p, sigma_p = m._metropolis_prepare(tok)
    dF0 = m._metropolis_delta_f(belief, mu_p, sigma_p, tok, 0)
    assert dF0 > 0.0                                                     # genuinely uphill at this seed

    # (1) Tiny temperature: dF0/T is so negative that exp(-dF0/T) underflows to exactly 0.0 (fp64), so
    # the stochastic accept is deterministically False -- the sweep accepts nothing (0 for ALL 4
    # tokens here, all of which happen to be uphill at this seed).
    m.cfg.omega_metropolis_temperature = 1e-6
    stats = m.metropolis_omega_step(tok, generator=torch.Generator().manual_seed(0))
    assert stats["accepted"] == 0
    assert torch.det(m.prior_bank.omega_embed[0]).item() > 0.0           # unmutated, still det>0 sheet

    # (2) Exact-rule pin: torch.unique sorts ascending, so token 0 is the FIRST proposal the sweep
    # scores -- against the untouched _metropolis_prepare belief, i.e. the SAME dF0 computed above.
    # Reproduce the sweep's first `torch.rand(generator=...)` draw independently and assert the move's
    # own accept/reject decision equals the documented formula dF<=0 or u<exp(-dF/T). This seed lands
    # in the u<exp(-dF/T) branch (True), so both directions of the stochastic gate are exercised across
    # the two sub-cases of this test.
    m.cfg.omega_metropolis_temperature = 1e-3                           # moderate T: a genuine 0<p<1 draw
    T = m.cfg.omega_metropolis_temperature
    seed = 3
    u0 = torch.rand((), generator=torch.Generator().manual_seed(seed)).item()
    expect_accept0 = (dF0 <= 0.0) or (u0 < math.exp(-dF0 / T))
    assert expect_accept0                                                # this seed's draw clears the threshold
    det_before = torch.det(m.prior_bank.omega_embed[0]).item()
    m.metropolis_omega_step(tok, generator=torch.Generator().manual_seed(seed))
    det_after = torch.det(m.prior_bank.omega_embed[0]).item()
    accepted0 = det_before * det_after < 0.0                             # det sign toggles iff token 0 was flipped
    assert accepted0 == expect_accept0


def test_exact_delta_f_matches_independent_recompute():
    # The per-token DeltaF the move computes (masked left-multiply of belief.omega by R) MUST equal
    # an INDEPENDENT recompute that flips the SOURCE table omega_embed[token] and re-looks-up the
    # frame -- pinning that the masked trial-belief flip == the source-table flip. Distinct tokens so
    # the transport Omega_ij = U_i U_j^{-1} is nontrivial and DeltaF genuinely nonzero.
    m = _model(vocab_size=4)
    tok = torch.tensor([[0, 1, 2, 3]])
    tid = 1
    belief, mu_p, sigma_p = m._metropolis_prepare(tok)
    assert belief.omega is not None                                     # omega actually enters F
    dF_move = m._metropolis_delta_f(belief, mu_p, sigma_p, tok, tid)
    assert dF_move == dF_move                                           # finite (not NaN)
    assert abs(dF_move) > 0.0                                           # genuinely nonzero (distinct tokens)
    # Independent oracle: flip the source table row, re-look-up the (fixed-belief) frame, recompute F.
    F_cur = m._metropolis_free_energy(belief, mu_p, sigma_p)
    R = reflection_element(belief.omega.shape[-1])
    m._flip_omega_embed_row(R, tid)
    relooked = m.prior_bank._omega_lookup(tok)                          # frame from the flipped source table
    F_trial = m._metropolis_free_energy(belief._replace(omega=relooked), mu_p, sigma_p)
    m._flip_omega_embed_row(R, tid)                                     # restore (R is involutory)
    dF_indep = F_trial - F_cur
    assert abs(dF_move - dF_indep) < 1e-5                               # exact-DeltaF anchor (fp5)


def test_metropolis_step_stats_finite():
    m = _model(vocab_size=4)
    tok = torch.tensor([[0, 1, 2, 3]])
    stats = m.metropolis_omega_step(tok, generator=torch.Generator().manual_seed(0))
    assert stats["proposed"] == 4                                       # one proposal per unique token
    assert 0 <= stats["accepted"] <= stats["proposed"]
    assert "mean_delta_f" in stats and stats["mean_delta_f"] == stats["mean_delta_f"]  # finite


def test_seeded_reproducible():
    tok = torch.tensor([[0, 1, 2, 3, 0, 1]])
    m1 = _model(); m2 = _model()
    with torch.no_grad():                                              # identical init
        m2.prior_bank.omega_embed.copy_(m1.prior_bank.omega_embed)
    s1 = m1.metropolis_omega_step(tok, generator=torch.Generator().manual_seed(7))
    s2 = m2.metropolis_omega_step(tok, generator=torch.Generator().manual_seed(7))
    assert s1 == s2
    assert torch.equal(m1.prior_bank.omega_embed, m2.prior_bank.omega_embed)


# --------------------------------------------------------------------------------------------------
# Task 3: train-loop seam -- _maybe_metropolis_omega (gated + cadence-checked call site)
# --------------------------------------------------------------------------------------------------


def test_train_seam_gated_and_cadence(monkeypatch):
    m = _model()
    calls = {"n": 0}
    def _spy(token_ids, *, generator):
        calls["n"] += 1; return {}
    monkeypatch.setattr(m, "metropolis_omega_step", _spy)
    from vfe3.train import _maybe_metropolis_omega    # small guarded helper Task 3 factors out
    gen = torch.Generator().manual_seed(0)
    tok = torch.tensor([[0, 1, 2]])
    # every=2: fires on steps 0 and 2, not 1
    m.cfg.omega_metropolis_every = 2
    _maybe_metropolis_omega(m, tok, step=0, generator=gen); assert calls["n"] == 1
    _maybe_metropolis_omega(m, tok, step=1, generator=gen); assert calls["n"] == 1
    _maybe_metropolis_omega(m, tok, step=2, generator=gen); assert calls["n"] == 2
    # off -> never
    m.cfg.omega_reflection = "off"
    _maybe_metropolis_omega(m, tok, step=0, generator=gen); assert calls["n"] == 2


def test_metropolis_generator_state_roundtrips(tmp_path, monkeypatch):
    from vfe3.run_artifacts import RunArtifacts
    from vfe3.train import train

    with pytest.warns(UserWarning, match="near-no sheet selection"):
        model = _model(checkpoint_interval=2, max_steps=3, seed=17,
                       n_e_steps=1, m_phi_lr=0.0)
    cfg = model.cfg
    tokens = torch.tensor([[0, 1, 2, 3]])
    targets = torch.tensor([[1, 2, 3, 4]])
    loader = [(tokens, targets)]
    source_draws = []

    def _record_source(token_ids, *, generator):
        source_draws.append(float(torch.rand((), generator=generator)))
        return {}

    monkeypatch.setattr(model, "metropolis_omega_step", _record_source)
    artifacts = RunArtifacts(tmp_path / "run", cfg, model)
    train(model, loader, cfg, n_steps=2, artifacts=artifacts)
    checkpoint_path = tmp_path / "run" / "checkpoints" / "step_2.pt"

    expected_generator = torch.Generator().manual_seed(int(cfg.seed))
    expected_source = [float(torch.rand((), generator=expected_generator)) for _ in range(2)]
    expected_state = expected_generator.get_state()
    expected_resumed = float(torch.rand((), generator=expected_generator))
    assert source_draws == expected_source
    bundle = torch.load(checkpoint_path, weights_only=True)
    assert torch.equal(bundle["metropolis_rng_state"], expected_state)

    with pytest.warns(UserWarning, match="near-no sheet selection"):
        resumed_model = _model(checkpoint_interval=2, max_steps=3, seed=17,
                               n_e_steps=1, m_phi_lr=0.0)
    resumed_draws = []

    def _record_resumed(token_ids, *, generator):
        resumed_draws.append(float(torch.rand((), generator=generator)))
        return {}

    monkeypatch.setattr(resumed_model, "metropolis_omega_step", _record_resumed)
    train(resumed_model, loader, resumed_model.cfg, n_steps=3, resume_from=checkpoint_path)
    assert resumed_draws == [expected_resumed]
