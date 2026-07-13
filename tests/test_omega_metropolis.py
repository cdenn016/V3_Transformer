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
    context = m._metropolis_prepare(tok)
    dF0 = m._metropolis_delta_f(context, 0)
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
    context = m._metropolis_prepare(tok)
    belief = context.belief
    assert belief.omega is not None                                     # omega actually enters F
    dF_move = m._metropolis_delta_f(context, tid)
    assert dF_move == dF_move                                           # finite (not NaN)
    assert abs(dF_move) > 0.0                                           # genuinely nonzero (distinct tokens)
    # Independent oracle: flip the source table row, re-look-up the (fixed-belief) frame, recompute F.
    F_cur = m._metropolis_free_energy(belief, context)
    R = reflection_element(belief.omega.shape[-1])
    m._flip_omega_embed_row(R, tid)
    relooked = m.prior_bank._omega_lookup(tok)                          # frame from the flipped source table
    F_trial = m._metropolis_free_energy(belief._replace(omega=relooked), context)
    m._flip_omega_embed_row(R, tid)                                     # restore (R is involutory)
    dF_indep = F_trial - F_cur
    assert abs(dF_move - dF_indep) < 1e-5                               # exact-DeltaF anchor (fp5)


def test_compact_metropolis_uses_block_reflection_and_matches_source_flip(monkeypatch):
    import vfe3.geometry.generators as generators_module
    from vfe3.geometry.lie_ops import CompactBlockElement
    from vfe3.geometry.transport import CompactFactoredTransport

    m = _model(
        vocab_size=4, gauge_group="block_glk", n_heads=2,
        omega_compact_storage=True,
    )
    tok = torch.tensor([[0, 1, 2, 3]])
    tid = 1
    context = m._metropolis_prepare(tok)
    belief = context.belief
    assert isinstance(belief.omega, CompactBlockElement)
    d = belief.omega.block_dim
    R_d = reflection_element(d)

    def _forbid_dense(*args, **kwargs):
        raise AssertionError("compact Metropolis materialized a dense K x K element")

    monkeypatch.setattr(CompactBlockElement, "to_dense", _forbid_dense)
    monkeypatch.setattr(CompactFactoredTransport, "to_dense_omega", _forbid_dense)
    original_reflection = generators_module.reflection_element
    reflection_sizes = []

    def _tracked_reflection(size, **kwargs):
        reflection_sizes.append(size)
        if size == m.cfg.embed_dim:
            raise AssertionError("compact Metropolis allocated reflection_element(K)")
        return original_reflection(size, **kwargs)

    monkeypatch.setattr(generators_module, "reflection_element", _tracked_reflection)

    dF_move = m._metropolis_delta_f(context, tid)
    F_cur = m._metropolis_free_energy(belief, context)
    m._flip_omega_embed_row(R_d, tid)
    relooked = m.prior_bank._omega_lookup(tok)
    F_trial = m._metropolis_free_energy(belief._replace(omega=relooked), context)
    m._flip_omega_embed_row(R_d, tid)

    assert abs(dF_move - (F_trial - F_cur)) < 1e-5
    stats = m.metropolis_omega_step(tok, generator=torch.Generator().manual_seed(0))
    assert reflection_sizes and all(size == d for size in reflection_sizes)
    assert "reflection_scope" not in stats
    assert m.prior_bank.reflection_scope == "block_0_probe"


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


# --------------------------------------------------------------------------------------------------
# Audit 2026-07-12 N7: _metropolis_free_energy evaluates the batch in ONE free_energy_value call
# --------------------------------------------------------------------------------------------------
from vfe3.belief import BeliefState


def _reference_per_sequence_sum(model, context, *, mode):
    """The pre-N7 per-sequence reference: one single-sequence free_energy_value per batch row,
    summed as Python floats. Pins the batched implementation to the same total. The N7 configs run
    with the folds OFF and n_layers=1, so the effective prior is the RAW attention prior and the
    final-block prior is the encode prior -- this raw per-sequence reference matches the folded scorer
    there exactly."""
    from vfe3.free_energy import attention_tau
    from vfe3.inference.e_step import free_energy_value
    from vfe3.model.block import _as_coeff
    cfg, grp = model.cfg, model.group
    gp = "omega_direct" if mode == "omega" else "phi"
    belief, mu_p, sigma_p = context.belief, context.mu_p, context.sigma_p
    dev = belief.mu.device
    tau = attention_tau(model.effective_kappa_beta(dev), grp.irrep_dims)
    log_prior = model._attention_log_prior(belief.mu.shape[-2], dev)
    total = 0.0
    with torch.no_grad():
        for b in range(belief.mu.shape[0]):
            bel = BeliefState(
                mu=belief.mu[b], sigma=belief.sigma[b],
                phi=(belief.phi[b] if belief.phi is not None else None),
                omega=(belief.omega[b] if belief.omega is not None else None),
                reflection=(belief.reflection[b] if belief.reflection is not None else None))
            total += free_energy_value(
                bel, mu_p[b], sigma_p[b], grp,
                tau=tau, renyi_order=cfg.renyi_order, value=cfg.lambda_alpha,
                b0=_as_coeff(cfg.b0, dev), c0=_as_coeff(cfg.c0, dev),
                lambda_beta=cfg.lambda_beta, kl_max=cfg.kl_max, eps=cfg.eps,
                include_attention_entropy=cfg.include_attention_entropy,
                family=cfg.family, divergence_family=cfg.divergence_family,
                lambda_alpha_mode=cfg.lambda_alpha_mode,
                gauge_parameterization=gp, log_prior=log_prior,
            ).item()
    return total


@pytest.mark.parametrize("mode", ["omega", "phi"])
def test_metropolis_free_energy_single_batched_eval_matches_per_sequence_sum(mode, monkeypatch):
    """One batched free_energy_value call (one host sync) per fixed-belief F -- not one per
    sequence -- with the total pinned to the per-sequence sum. free_energy_value reduces by
    sum() over every leading axis and sequences are independent, so the batched scalar IS the
    per-sequence sum up to float summation order."""
    import vfe3.inference.e_step as e_step_module

    if mode == "omega":
        m = _model(n_e_steps=1)
    else:
        m = VFEModel(VFE3Config(
            gauge_parameterization="phi", gauge_group="glk", embed_dim=4, n_heads=1,
            vocab_size=6, max_seq_len=4, n_layers=1, n_e_steps=1, transport_mode="flat",
            e_phi_lr=0.0, use_head_mixer=False, family="gaussian_diagonal",
            decode_mode="diagonal", lambda_gamma=0.0, s_e_step=False,
            phi_reflection="metropolis", pos_phi="none"))
    tokens = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0], [4, 4, 5, 1]])       # B=3
    context = m._metropolis_prepare(tokens, mode=mode)

    ref = _reference_per_sequence_sum(m, context, mode=mode)

    real_fe = e_step_module.free_energy_value
    calls = {"n": 0}

    def _counting_fe(*args, **kwargs):
        calls["n"] += 1
        return real_fe(*args, **kwargs)

    monkeypatch.setattr(e_step_module, "free_energy_value", _counting_fe)
    total = m._metropolis_free_energy(context.belief, context, mode=mode)
    assert calls["n"] == 1, f"expected ONE batched F eval, got {calls['n']} (per-sequence loop)"
    assert total == pytest.approx(ref, rel=1e-5, abs=1e-5)
