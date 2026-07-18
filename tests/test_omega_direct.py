import warnings

import pytest
import torch

from vfe3.belief import BeliefState
from vfe3.config import VFE3Config
from vfe3.geometry.generators import generate_glk, generate_son, reflection_element
from vfe3.geometry.groups import get_group
from vfe3.geometry.lie_ops import CompactBlockElement, retract_omega
from vfe3.geometry.transport import (RopeTransport, build_transport_from_element, compute_transport_operators,
                                      group_element_inverse, transport_covariance, transport_mean,
                                      CompactFactoredTransport, FactoredTransport)
from vfe3.inference.e_step import (
    _transport,
    build_belief_transport,
    e_step,
    e_step_iteration,
    free_energy_value,
)
from vfe3.model.model import VFEModel
from vfe3.model.prior_bank import PriorBank


def _cfg(**over):
    base = dict(vocab_size=6, embed_dim=4, n_heads=1, max_seq_len=4, n_layers=1, n_e_steps=2,
                gauge_group="glk", family="gaussian_full", transport_mode="flat",
                pos_rotation="none", use_head_mixer=False, use_prior_bank=True, decode_mode="full",
                pos_phi="none", e_phi_lr=0.0)
    base.update(over)
    return VFE3Config(**base)


def test_beliefstate_omega_field_optional_and_addressable():
    mu = torch.zeros(1, 3, 4); sigma = torch.ones(1, 3, 4); phi = torch.zeros(1, 3, 5)
    b = BeliefState(mu=mu, sigma=sigma, phi=phi)
    assert b.omega is None                                   # default: phi path untouched
    U = torch.eye(4).expand(1, 3, 4, 4)
    b2 = b._replace(omega=U)
    assert torch.equal(b2.omega, U)
    assert b2.mu is mu and b2.phi is phi                     # other fields preserved


def test_reflection_element_is_det_negative_orthogonal():
    R = reflection_element(4)
    assert R.shape == (4, 4)
    assert torch.det(R) < 0                                  # reaches the other GL component
    assert torch.allclose(R @ R.transpose(-1, -2), torch.eye(4), atol=1e-7)   # reflection: R R^T = I
    assert torch.allclose(R @ R, torch.eye(4), atol=1e-7)   # involutory


def test_element_transport_cocycle_and_identity():
    grp = get_group("glk")(K=4)
    g = torch.Generator().manual_seed(7)
    # Random near-identity invertible frames per token.
    U = torch.eye(4) + 0.1 * torch.randn(1, 3, 4, 4, generator=g)
    built = build_transport_from_element(U, grp)
    omega = built["Omega"]                                    # glk -> dict path
    # cocycle: Omega_ij Omega_jk = Omega_ik  (U_i U_j^{-1} U_j U_k^{-1} = U_i U_k^{-1})
    lhs = omega[0, 0, 1] @ omega[0, 1, 2]
    assert torch.allclose(lhs, omega[0, 0, 2], atol=1e-4)
    # identity frames -> Omega = I
    I = torch.eye(4).expand(1, 3, 4, 4)
    omega_I = build_transport_from_element(I, grp)["Omega"]
    assert torch.allclose(omega_I, torch.eye(4).expand(1, 3, 3, 4, 4), atol=1e-6)


def test_element_transport_matches_phi_path_when_U_equals_exp_phi():
    grp = get_group("glk")(K=3)
    g = torch.Generator().manual_seed(3)
    phi = 0.2 * torch.randn(1, 3, grp.generators.shape[0], generator=g)
    ref = compute_transport_operators(phi, grp, gauge_mode="learned")
    U = ref["exp_phi"]                                        # U_i := exp(phi_i)
    got = build_transport_from_element(U, grp)["Omega"]
    assert torch.allclose(got, ref["Omega"], atol=1e-5)      # same cocycle, exp-free assembly


def test_element_transport_reaches_det_negative():
    grp = get_group("glk")(K=4)
    R = reflection_element(4)
    U = torch.stack([torch.eye(4), R], dim=0).unsqueeze(0)   # (1, 2, 4, 4): token0 det>0, token1 det<0
    omega = build_transport_from_element(U, grp)["Omega"]
    assert torch.det(omega[0, 0, 1]) < 0                     # I @ R^{-1} has det < 0


def test_element_transport_block_glk_is_factored():
    grp = get_group("block_glk")(K=4, n_heads=2)             # irrep_dims [2,2]
    U = torch.eye(4).expand(1, 3, 4, 4).contiguous()
    built = build_transport_from_element(U, grp)
    assert isinstance(built, FactoredTransport)
    assert built.same_frame_flat_cocycle
    mu = torch.randn(1, 3, 4)
    mt = transport_mean(built, mu)                           # (1,3,3,4) via the factored fast path
    assert mt.shape == (1, 3, 3, 4)


def test_element_transport_omega_direct_wires_mean_per_head():
    K, H, d, N = 6, 3, 2, 4
    grp = get_group("block_glk")(K=K, n_heads=H)
    gen = torch.Generator().manual_seed(31)
    blocks = torch.eye(d).expand(1, N, H, d, d).clone()
    blocks = blocks + 0.06 * torch.randn(blocks.shape, generator=gen)
    compact = CompactBlockElement(blocks, K)
    element = compact.to_dense()
    phi = torch.zeros(1, N, grp.generators.shape[0])          # ignored on the omega-direct path
    mu = torch.randn(1, N, K, generator=gen)

    default = build_transport_from_element(element, grp)
    direct = build_transport_from_element(element, grp, mean_per_head=True)
    forward = build_belief_transport(
        phi, grp, gauge_parameterization="omega_direct", omega=element,
        transport_mean_per_head=True,
    )

    assert isinstance(default, FactoredTransport) and not default.mean_per_head
    assert isinstance(direct, FactoredTransport) and direct.mean_per_head
    assert isinstance(forward, FactoredTransport) and forward.mean_per_head
    dense = direct.to_dense_omega()
    expected = transport_mean(dense, mu)
    assert torch.allclose(transport_mean(direct, mu), expected, atol=2e-6, rtol=1e-5)
    assert torch.allclose(transport_mean(forward, mu), expected, atol=2e-6, rtol=1e-5)


def test_compact_element_transport_omega_direct_wires_mean_per_head():
    K, H, d, N = 6, 3, 2, 4
    grp = get_group("block_glk")(K=K, n_heads=H)
    gen = torch.Generator().manual_seed(32)
    blocks = torch.eye(d).expand(1, N, H, d, d).clone()
    blocks = blocks + 0.06 * torch.randn(blocks.shape, generator=gen)
    element = CompactBlockElement(blocks, K)
    phi = torch.zeros(1, N, grp.generators.shape[0])          # ignored on the omega-direct path
    mu = torch.randn(1, N, K, generator=gen)

    direct = build_transport_from_element(element, grp, mean_per_head=True)
    forward = build_belief_transport(
        phi, grp, gauge_parameterization="omega_direct", omega=element,
        transport_mean_per_head=True,
    )
    replay = _transport(
        phi, grp, gauge_parameterization="omega_direct", omega=element,
        transport_mean_per_head=True,
    )

    for built in (direct, forward, replay):
        assert isinstance(built, CompactFactoredTransport) and built.mean_per_head
        assert built.same_frame_flat_cocycle
    dense = direct.to_dense_omega()
    expected = transport_mean(dense, mu)
    for built in (direct, forward, replay):
        assert torch.allclose(transport_mean(built, mu), expected, atol=2e-6, rtol=1e-5)


def test_build_belief_transport_omega_direct_branch():
    grp = get_group("glk")(K=3)
    phi = torch.zeros(1, 3, grp.generators.shape[0])          # ignored on the omega path
    U = torch.eye(3) + 0.1 * torch.randn(1, 3, 3, 3, generator=torch.Generator().manual_seed(1))
    built = build_belief_transport(phi, grp, gauge_parameterization="omega_direct", omega=U)
    ref = build_transport_from_element(U, grp)["Omega"]
    # glk single-block: build_belief_transport normalizes the builder dict to a dense (B,N,N,K,K)
    # Omega, matching the phi path (so the forward consumers get a tensor, never a raw dict).
    assert torch.allclose(built, ref, atol=1e-6)
    # default axis unchanged: phi path returns its usual object
    phi_out = build_belief_transport(phi, grp)                # default 'phi' path, phi=0 -> Omega = I
    eye = torch.eye(3).expand(1, 3, 3, 3, 3)
    assert torch.allclose(phi_out, eye, atol=1e-6)            # glk single-block returns a dense Omega tensor


def test_lower_level_omega_direct_rejects_phi_estep_update():
    # Config rejects this unsupported cross-chart update, but direct E-step callers bypass config.
    # Fail at the lower API boundary before the phi update can silently optimize an inert chart.
    K, N = 4, 3
    grp = get_group("glk")(K=K)
    n_gen = grp.generators.shape[0]
    g = torch.Generator().manual_seed(2)
    mu = 0.1 * torch.randn(1, N, K, generator=g)
    sigma = torch.ones(1, N, K)
    phi = torch.zeros(1, N, n_gen)
    U = torch.eye(K) + 0.05 * torch.randn(1, N, K, K, generator=g)   # invertible near-identity frames
    belief = BeliefState(mu=mu, sigma=sigma, phi=phi, omega=U)
    mu_p = torch.zeros(1, N, K)
    sigma_p = torch.ones(1, N, K)

    with pytest.raises(ValueError, match="e_phi_lr"):
        e_step_iteration(belief, mu_p, sigma_p, grp, e_phi_lr=0.1,
                         gauge_parameterization="omega_direct")
    with pytest.raises(ValueError, match="e_phi_lr"):
        e_step(belief, mu_p, sigma_p, grp, n_iter=2, e_phi_lr=0.1,
               gauge_parameterization="omega_direct")


def test_multilayer_post_estep_transforms_preserve_omega_direct_frame():
    # block.py applies block_norm after each E-step. Its BeliefState rebuild must retain the stored
    # omega frame so layer 2 consumes the same U_i that layer 1 did, rather than falling back to phi.
    model = VFEModel(_cfg(gauge_parameterization="omega_direct", n_layers=2,
                          norm_type_block="mahalanobis"))
    token_ids = torch.tensor([[0, 1, 2, 3]])
    expected = model.prior_bank._omega_lookup(token_ids)

    belief, _ = model.forward_beliefs(token_ids)

    assert belief.omega is not None
    assert torch.equal(belief.omega, expected)
    assert torch.isfinite(belief.mu).all()


def test_free_energy_value_filtered_keys_rejects_omega_direct():
    # FIX 3: the filtered (frozen-keys) free energy hand-builds Omega from phi frames, so it would
    # silently use the WRONG frame under omega_direct. It must reject rather than degrade.
    K, N = 4, 3
    grp = get_group("glk")(K=K)
    n_gen = grp.generators.shape[0]
    mu = 0.1 * torch.randn(1, N, K)
    sigma = torch.ones(1, N, K)
    phi = torch.zeros(1, N, n_gen)
    U = torch.eye(K).expand(1, N, K, K).contiguous()
    belief = BeliefState(mu=mu, sigma=sigma, phi=phi, omega=U)
    mu_p = torch.zeros(1, N, K)
    sigma_p = torch.ones(1, N, K)
    with pytest.raises(NotImplementedError):
        free_energy_value(belief, mu_p, sigma_p, grp,
                          gauge_parameterization="omega_direct", keys=belief)


def test_prior_bank_omega_table_gated_and_encodes_identity():
    # default (phi) path: no omega table, no omega on the belief
    pb_phi = PriorBank(vocab_size=6, K=4, n_gen=16)
    assert not hasattr(pb_phi, "omega_embed")
    assert pb_phi.encode(torch.zeros(1, 3, dtype=torch.long)).omega is None
    # omega_direct path: table exists, identity init, belief carries (B,N,K,K)
    pb = PriorBank(vocab_size=6, K=4, n_gen=16, gauge_parameterization="omega_direct", irrep_dims=[4])
    assert pb.omega_embed.shape == (6, 4, 4)
    assert torch.allclose(pb.omega_embed, torch.eye(4).expand(6, 4, 4), atol=1e-7)
    b = pb.encode(torch.zeros(1, 3, dtype=torch.long))
    assert b.omega.shape == (1, 3, 4, 4)
    assert torch.allclose(b.omega, torch.eye(4).expand(1, 3, 4, 4), atol=1e-7)


def test_prior_bank_rejects_omega_direct_additive_encoder():
    with pytest.raises(ValueError, match="per_token_additive"):
        PriorBank(vocab_size=6, K=4, n_gen=16, gauge_parameterization="omega_direct",
                  irrep_dims=[4], encode_mode="per_token_additive")


@pytest.mark.parametrize("value", [0, 1, "False", None])
def test_prior_bank_omega_compact_storage_requires_strict_bool(value):
    with pytest.raises(ValueError, match="omega_compact_storage"):
        PriorBank(vocab_size=6, K=4, n_gen=16, omega_compact_storage=value)


def test_prior_bank_omega_reflection_seeds_det_negative():
    pb = PriorBank(vocab_size=6, K=4, n_gen=16, gauge_parameterization="omega_direct",
                   irrep_dims=[4], omega_reflection="init_seed")
    dets = torch.det(pb.omega_embed)
    assert (dets < 0).any()                                  # some tokens seeded into det<0


def test_full_model_forward_omega_direct_finite_and_matches_identity_gauge():
    tok = torch.randint(0, 6, (1, 4), generator=torch.Generator().manual_seed(2))
    torch.manual_seed(0); m_od = VFEModel(_cfg(gauge_parameterization="omega_direct"))
    with torch.no_grad():
        logits_od = m_od(tok)[0]
    assert torch.isfinite(logits_od).all()
    # identity-init omega_direct == phi path with frames zeroed (both give Omega = I)
    torch.manual_seed(0); m_phi = VFEModel(_cfg(gauge_parameterization="phi"))
    with torch.no_grad():
        m_phi.prior_bank.phi_embed.zero_()
        if hasattr(m_phi, "pos_phi_free"):
            m_phi.pos_phi_free.zero_()
        logits_phi = m_phi(tok)[0]
    assert torch.allclose(logits_od, logits_phi, atol=1e-5)


def test_omega_direct_reflection_changes_logits():
    tok = torch.randint(0, 6, (1, 4), generator=torch.Generator().manual_seed(2))
    torch.manual_seed(0); m0 = VFEModel(_cfg(gauge_parameterization="omega_direct"))
    torch.manual_seed(0); m1 = VFEModel(_cfg(gauge_parameterization="omega_direct",
                                             omega_reflection="init_seed"))
    with torch.no_grad():
        d = (m0(tok)[0] - m1(tok)[0]).abs().max()
    assert d > 1e-4                                          # the stored det<0 frame actually feeds the forward


def test_retract_omega_stays_in_component_and_group():
    G = generate_glk(3)                                       # (9,3,3)
    U = torch.eye(3).expand(4, 3, 3).contiguous()
    xi = 0.05 * torch.randn(4, 9, generator=torch.Generator().manual_seed(0))
    for mode in ("lie_exp", "cayley"):
        Un = retract_omega(U, xi, G, mode=mode)
        assert Un.shape == (4, 3, 3)
        assert (torch.det(Un) > 0).all()                     # retraction preserves the det>0 component
        # a det<0 base stays det<0 (component preserved)
        Rneg = U.clone(); Rneg[:, 0, 0] = -1.0
        assert (torch.det(retract_omega(Rneg, xi, G, mode=mode)) < 0).all()


def test_retract_omega_cayley_large_step_stays_in_component():
    # Regression: without the ||A||_F<2 trust-region clamp, cayley flips the component for a large
    # step -- A=diag(3,0,0) gives det(cayley(A)) = 2.5/-0.5 = -5 < 0. The clamp keeps it in-component.
    G = generate_glk(3)                                       # (9,3,3); generator 0 = E_00 = diag(1,0,0)
    U = torch.eye(3).expand(4, 3, 3).contiguous()
    xi = torch.zeros(4, 9); xi[:, 0] = 3.0                    # A = diag(3,0,0): the previously-flipping case
    assert (torch.det(retract_omega(U, xi, G, mode="cayley")) > 0).all()   # det>0 base stays det>0
    Rneg = U.clone(); Rneg[:, 0, 0] = -1.0
    assert (torch.det(retract_omega(Rneg, xi, G, mode="cayley")) < 0).all()  # det<0 base stays det<0


def test_gauge_optim_omega_step_moves_active_rows_only():
    from vfe3.gauge_optim import GaugeManifoldAdamW
    group = get_group("glk")(K=3)
    U = torch.nn.Parameter(torch.eye(3).expand(5, 3, 3).contiguous())
    opt = GaugeManifoldAdamW([{"params": [U], "lr": 0.1, "omega": True, "weight_decay": 0.0}],
                                group, phi_group_trust_radius=0.1,
                                phi_chart_max_norm=5.0, phi_bch_residual_max=1e-6,
                                phi_precond_mode="pullback", weight_decay=0.0)
    U.grad = torch.zeros_like(U)
    U.grad[2] = torch.randn(3, 3, generator=torch.Generator().manual_seed(1))   # only row 2 active
    before = U.data.clone()
    opt.step()
    assert torch.allclose(U.data[0], before[0])              # inactive rows untouched
    assert not torch.allclose(U.data[2], before[2])          # active row moved
    assert torch.det(U.data[2]) > 0                           # still in GL+(3)


def test_omega_direct_full_model_gauge_invariance():
    """A global gauge transform of the tied prior tables leaves omega_direct decode logits invariant
    (fp64), and the linear-decode arm has bite (fp32) -- the same t8 contract as the phi path."""
    # The model's gauge group is glk (omega_direct is glk-scoped), but the END-TO-END decode
    # invariance needs an ORTHOGONAL g: the decode/self-coupling prior covariance is the diagonal
    # (V,K) sigma_log_embed table, so only g with g Sigma g^T representable in it are invariant.
    # We therefore draw g from the SKEW so(4) generators (matrix_exp of skew is orthogonal) --
    # exactly t8's reason for so_k (test_gauge_groups.py:188). The full GL(K) covariance law is
    # pinned separately by the transport-level test below (no decode/sigma limitation).
    def delta(dbl, **over):
        torch.manual_seed(0); m = VFEModel(_cfg(gauge_parameterization="omega_direct", **over))
        with torch.no_grad():
            m.prior_bank.omega_embed.copy_(torch.eye(4).expand(6, 4, 4))       # frames -> identity
            m.prior_bank.sigma_log_embed.zero_()                              # Sigma = I
            if hasattr(m, "pos_phi_free"):
                m.pos_phi_free.zero_()
        if dbl: m = m.double()
        m.eval()
        # orthogonal g so the diagonal Sigma=I readout stays representable (g I g^T = I); the
        # full-GL(K) covariance is pinned separately by the transport-level test below.
        gen_so = generate_son(4).to(torch.float64 if dbl else torch.float32)   # skew -> matrix_exp is orthogonal
        c = 0.3 * torch.randn(gen_so.shape[0], generator=torch.Generator().manual_seed(1)).to(gen_so.dtype)
        g = torch.linalg.matrix_exp(torch.einsum("a,aij->ij", c, gen_so))      # g in O(4): g g^T = I
        eye = torch.eye(4, dtype=g.dtype)
        assert torch.allclose(g @ g.transpose(-1, -2), eye, atol=1e-6)         # so(4) => g orthogonal
        tok = torch.randint(0, 6, (1, 4), generator=torch.Generator().manual_seed(2))
        with torch.no_grad():
            l0 = m(tok)[0].clone()
            m.prior_bank.mu_embed.copy_(torch.einsum("kl,vl->vk", g, m.prior_bank.mu_embed))
            # co-transform the stored frame: U -> g U (the cocycle U_i U_j^{-1} is g-invariant)
            m.prior_bank.omega_embed.copy_(torch.einsum("kl,vlm->vkm", g, m.prior_bank.omega_embed))
            l1 = m(tok)[0].clone()
        return float((l0 - l1).abs().max())
    assert delta(dbl=True) < 1e-5
    assert delta(dbl=False, use_prior_bank=False) > 1e-4       # linear decode does not co-transform -> bite


def test_omega_direct_user_target_config_finite_forward():
    """CAPSTONE: the user's training config -- omega_direct with the FULL gamma / model-coupling (s)
    channel ON (lambda_gamma>0, s_e_step, gamma_as_beta_prior, prior_source='model_channel') --
    constructs and runs to a FINITE forward. Phase-3 removed the config gate that used to reject this
    combination; if this raises or returns non-finite, the s-channel frame threading (Tasks 1-4) left
    an integration gap. Tiny dims (K=4, N=4, vocab=6); gaussian_diagonal because the live s E-step
    refines the model channel as a diagonal Gaussian."""
    torch.manual_seed(0)
    m = VFEModel(_cfg(gauge_parameterization="omega_direct", gauge_group="glk",
                      lambda_gamma=0.75, s_e_step=True, gamma_as_beta_prior=True,
                      prior_source="model_channel", family="gaussian_diagonal",
                      decode_mode="diagonal"))
    m.eval()
    tok = torch.randint(0, 6, (1, 4), generator=torch.Generator().manual_seed(2))
    assert torch.isfinite(m(tok)[0]).all()


def test_omega_direct_full_model_gauge_invariance_gamma_on():
    """Gauge-COVARIANCE regression guard, sibling of test_omega_direct_full_model_gauge_invariance
    with the gamma / model-coupling (s) channel ON: a global ORTHOGONAL gauge transform g of ALL tied
    prior tables -- belief means mu, model-channel means s_mu, hyper-prior r_mu, and the stored frames
    omega (U -> gU) -- leaves the omega_direct decode logits invariant to fp64 with the full gamma / s
    wiring active. What this certifies is narrow but real: the Phase-3 gamma+s pipeline introduces NO
    gauge-BREAKING term -- a future edit that added one (a term not equivariant under a global g) would
    make this test fail.

    What it does NOT certify: it does NOT distinguish the U-transport from an exp(phi) transport. Both
    are gauge-COVARIANT, so both pass a gauge-INVARIANCE test identically. Concretely, with every base
    frame set to identity a global g maps every token frame to the same g, so the relative s-channel
    cocycle Omega_ij = U_i U_j^{-1} = g g^{-1} = I for all pairs -- exactly what a frame-blind exp(phi)=I
    s-channel builds -- and under gamma_as_beta_prior the gamma channel reaches the decode only through
    gauge-invariant scalar attention weights while under prior_source='model_channel' the s_mu rotation
    self-cancels between prior and belief. Reverting Task 3 would NOT change this test's result.

    The s-channel frame-USE (that it transports by the stored U, not exp(phi)) is certified elsewhere,
    by the per-task frame-fidelity unit tests that directly VARY the frame at zero phi and assert the
    output changes: test_refine_s_uses_stored_frame_not_phi_cocycle (Task 3, the s E-step) and the
    Task 1/2 gamma-coupling tests test_gamma_coupling_term_uses_stored_frame_not_phi_cocycle,
    test_gamma_coupling_terms_split_uses_stored_frame_not_phi_cocycle, and
    test_fold_gamma_prior_uses_stored_frame_not_phi_cocycle.

    Setup notes: an orthogonal g keeps the diagonal Sigma=I readout representable (g I g^T = I), exactly
    as the gamma-off sibling; every sigma table is zeroed so Sigma=I wherever the co-transform lands.
    s_e_step forces the diagonal family, so this uses gaussian_diagonal (the sibling uses _cfg's full).
    r_mu is set nonzero and co-transformed for a complete gauge; it is consumed only under lambda_h>0
    (inert here, harmless)."""
    torch.manual_seed(0)
    m = VFEModel(_cfg(gauge_parameterization="omega_direct", gauge_group="glk",
                      lambda_gamma=0.75, s_e_step=True, gamma_as_beta_prior=True,
                      prior_source="model_channel", family="gaussian_diagonal",
                      decode_mode="diagonal"))
    with torch.no_grad():
        m.prior_bank.omega_embed.copy_(torch.eye(4).expand(6, 4, 4))       # frames -> identity
        m.prior_bank.sigma_log_embed.zero_()                              # belief Sigma = I
        m.prior_bank.s_sigma_log_embed.zero_()                            # model-channel Sigma = I
        m.prior_bank.r_sigma_log.zero_()                                  # hyper-prior Sigma = I
        m.prior_bank.r_mu.copy_(torch.tensor([0.1, -0.2, 0.15, -0.05]))   # nonzero r_mu so its co-transform is not vacuous
        if hasattr(m, "pos_phi_free"):
            m.pos_phi_free.zero_()
    m = m.double()
    m.eval()
    # orthogonal g so the diagonal Sigma=I readout stays representable (g I g^T = I).
    gen_so = generate_son(4).to(torch.float64)                            # skew -> matrix_exp is orthogonal
    c = 0.3 * torch.randn(gen_so.shape[0], generator=torch.Generator().manual_seed(1)).to(gen_so.dtype)
    g = torch.linalg.matrix_exp(torch.einsum("a,aij->ij", c, gen_so))     # g in O(4): g g^T = I
    eye = torch.eye(4, dtype=g.dtype)
    assert torch.allclose(g @ g.transpose(-1, -2), eye, atol=1e-6)        # so(4) => g orthogonal
    tok = torch.randint(0, 6, (1, 4), generator=torch.Generator().manual_seed(2))
    # Co-transform EVERY table by g so the gauge transform is complete (gauge-covariance completeness).
    # Note these s/frame co-transforms are empirically INERT for the decode under this config, not
    # load-bearing: with identity base frames the frame co-transform U -> gU is a no-op on the relative
    # cocycle (g g^{-1} = I), the s_mu rotation self-cancels between prior and belief under
    # prior_source='model_channel', and the gamma channel reaches the decode only through
    # gauge-invariant scalar attention weights under gamma_as_beta_prior. They are applied for a
    # consistent global gauge; the frame-USE they might seem to exercise is pinned by the per-task
    # frame-fidelity tests named in the docstring, NOT here.
    with torch.no_grad():
        l0 = m(tok)[0].clone()
        m.prior_bank.mu_embed.copy_(torch.einsum("kl,vl->vk", g, m.prior_bank.mu_embed))
        m.prior_bank.s_mu_embed.copy_(torch.einsum("kl,vl->vk", g, m.prior_bank.s_mu_embed))   # s means -> g s (inert here)
        m.prior_bank.r_mu.copy_(g @ m.prior_bank.r_mu)                                          # hyper-prior mean -> g r (inert; lambda_h=0)
        # co-transform the stored frame: U -> g U (cocycle U_i U_j^{-1} g-invariant; a no-op on the decode here)
        m.prior_bank.omega_embed.copy_(torch.einsum("kl,vlm->vkm", g, m.prior_bank.omega_embed))
        l1 = m(tok)[0].clone()
    assert float((l0 - l1).abs().max()) < 1e-5


def test_omega_direct_transport_covariance_law():
    """The omega_direct cocycle transports covariantly under a PER-TOKEN general GL(K) gauge:
    Omega_ij -> g_i Omega_ij g_j^{-1}. Pins the full GL(K) property directly at the transport
    level (fp64), with no decode/diagonal-sigma limitation -- the general-g claim the end-to-end
    decode test cannot make (its diagonal Sigma readout forces g orthogonal)."""
    from vfe3.geometry.groups import get_group
    from vfe3.geometry.transport import build_transport_from_element
    grp = get_group("glk")(K=3)
    gen = torch.Generator().manual_seed(5)
    N = 4
    U = (torch.eye(3) + 0.15 * torch.randn(1, N, 3, 3, generator=gen)).double()      # invertible frames
    g = (torch.eye(3) + 0.2 * torch.randn(N, 3, 3, generator=gen)).double()          # per-token general GL(3)
    ginv = torch.linalg.inv(g)
    om  = build_transport_from_element(U, grp)["Omega"]                               # (1,N,N,3,3)
    Ug  = torch.einsum("nkl,bnlm->bnkm", g, U)                                        # U'_i = g_i U_i
    omg = build_transport_from_element(Ug, grp)["Omega"]
    # expected: Omega'_ij = g_i Omega_ij g_j^{-1}
    exp = torch.einsum("ikl,bijlm,jmn->bijkn", g, om, ginv)
    assert torch.allclose(omg, exp, atol=1e-9)


def test_group_element_inverse_skew_rows_match_true_inverse_and_gradient():
    group = get_group("so_k")(K=3)
    clean_0 = torch.eye(3)
    clean_1 = torch.tensor([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0],
    ])
    drifted_0 = torch.tensor([
        [1.10, 0.20, 0.00],
        [0.00, 0.90, 0.10],
        [0.00, 0.00, 1.05],
    ])
    drifted_1 = torch.tensor([
        [0.95, 0.00, 0.15],
        [0.05, 1.20, 0.00],
        [0.00, 0.10, 0.85],
    ])
    omega = torch.stack([clean_0, drifted_0, drifted_1, clean_1]).reshape(2, 2, 3, 3)
    omega = omega.requires_grad_(True)
    got = group_element_inverse(omega, group, residual_tol=1e-4)
    true_inverse = torch.linalg.inv(omega.double()).to(omega.dtype)

    assert got.shape == omega.shape
    assert got.dtype == omega.dtype
    assert got.device == omega.device
    assert torch.allclose(got, true_inverse, atol=1e-7, rtol=1e-6)

    weights = torch.arange(got.numel(), device=got.device, dtype=got.dtype).reshape_as(got) / got.numel()
    grad = torch.autograd.grad((got * weights).sum(), omega)[0]

    omega_ref = omega.detach().clone().requires_grad_(True)
    reference = torch.linalg.inv(omega_ref.double()).to(omega_ref.dtype)
    grad_reference = torch.autograd.grad((reference * weights).sum(), omega_ref)[0]

    assert torch.isfinite(grad).all()
    assert torch.isfinite(grad_reference).all()
    assert torch.allclose(grad, grad_reference, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("group_name", ["so_k", "glk"])
@pytest.mark.parametrize("residual_tol", [-1.0, float("nan"), float("inf")])
def test_group_element_inverse_rejects_invalid_residual_tol(group_name, residual_tol):
    group = get_group(group_name)(K=3)
    omega = torch.eye(3).reshape(1, 1, 3, 3)
    with pytest.raises(ValueError, match="residual_tol must be finite and nonnegative"):
        group_element_inverse(omega, group, residual_tol=residual_tol)


def test_element_transport_skew_group_uses_true_inverse_for_rounded_frame():
    grp = get_group("so_k")(K=4)                              # skew_symmetric=True -> U in O(4)
    g = torch.Generator().manual_seed(11)
    xi = 0.2 * torch.randn(1, 3, grp.generators.shape[0], generator=g)
    U = retract_omega(torch.eye(4).expand(1, 3, 4, 4).contiguous(), xi, grp.generators)  # in SO(4)
    built = build_transport_from_element(U, grp)              # single block -> dict
    true_inverse = torch.linalg.inv(U.double()).to(U.dtype)
    torch.testing.assert_close(built["exp_neg_phi"], true_inverse, atol=0.0, rtol=0.0)
    assert not torch.equal(built["exp_neg_phi"], U.transpose(-1, -2))
    # cocycle telescopes
    om = built["Omega"]
    assert torch.allclose(om[0, 0, 1] @ om[0, 1, 2], om[0, 0, 2], atol=1e-5)
    # Omega is exactly orthogonal (isometry)
    eye = torch.eye(4).expand(1, 3, 3, 4, 4)
    assert torch.allclose(torch.einsum("...kl,...ml->...km", om, om), eye, atol=1e-4)


def test_skew_element_transport_uses_true_inverse_after_orthogonality_drift(device):
    grp = get_group("so_k")(K=3, device=device)
    U = torch.eye(3, device=device).expand(1, 2, 3, 3).clone()
    U[0, 1] = torch.tensor([
        [1.00, 0.30, 0.00],
        [0.05, 1.20, 0.00],
        [0.00, 0.10, 0.85],
    ], device=device)

    built = build_transport_from_element(U, grp)
    true_inverse = torch.linalg.inv(U.double()).to(U.dtype)

    assert built["exp_neg_phi"].device == U.device
    assert torch.allclose(built["exp_neg_phi"][0, 1], true_inverse[0, 1], atol=1e-7, rtol=1e-6)
    assert not torch.allclose(
        built["exp_neg_phi"][0, 1], U[0, 1].transpose(-1, -2), atol=1e-5, rtol=1e-5,
    )


def test_element_transport_nonskew_still_uses_inv_byte_identical():
    grp = get_group("glk")(K=3)                               # skew_symmetric=False -> unchanged inv path
    U = (torch.eye(3) + 0.1 * torch.randn(1, 2, 3, 3, generator=torch.Generator().manual_seed(1)))
    built = build_transport_from_element(U, grp)
    ref_inv = torch.linalg.inv(U.double()).to(U.dtype)
    assert torch.allclose(built["exp_neg_phi"], ref_inv, atol=0)   # exact same fp64-inv path as shipped


def test_omega_direct_sp_symplectic_membership_and_cocycle():
    # Phase 2 Task 3: sp is skew_symmetric=False (real value of the non-exp-interior omega_direct
    # reach), reusing the existing (V,K,K) table + fp64-inv transport + group-agnostic optimizer --
    # no source change. This pins the group-membership invariant: a retracted frame stays IN Sp(4,R)
    # (preserves the symplectic form J), and the assembled cocycle still telescopes.
    grp = get_group("sp")(K=4)                                 # Sp(4,R); n_gen = m(2m+1) = 2*5 = 10
    assert grp.generators.shape[0] == 10
    g = torch.Generator().manual_seed(3)
    xi = 0.15 * torch.randn(1, 3, grp.generators.shape[0], generator=g)
    U = retract_omega(torch.eye(4).expand(1, 3, 4, 4).contiguous(), xi, grp.generators)  # in Sp(4,R)
    # symplectic form J = [[0,I],[-I,0]] preserved: U^T J U = J
    m = 2; J = torch.zeros(4, 4); J[:m, m:] = torch.eye(m); J[m:, :m] = -torch.eye(m)
    UtJU = U.transpose(-1, -2) @ J @ U
    assert torch.allclose(UtJU, J.expand_as(UtJU), atol=1e-4)
    om = build_transport_from_element(U, grp)["Omega"]         # single block (irrep_dims=[4]) -> dict
    assert torch.allclose(om[0, 0, 1] @ om[0, 1, 2], om[0, 0, 2], atol=1e-4)   # cocycle


def test_sp_omega_membership_diagnostic_detects_drift():
    from vfe3.gauge_optim import GaugeManifoldAdamW

    grp = get_group("sp")(K=4)
    U = torch.nn.Parameter(torch.eye(4).expand(3, 4, 4).clone())
    with torch.no_grad():
        U[1, 0, 0] = 1.25
    opt = GaugeManifoldAdamW(
        [{"params": [U], "lr": 0.0, "omega": True, "weight_decay": 0.0}],
        grp, phi_group_trust_radius=0.1, phi_chart_max_norm=5.0,
        phi_bch_residual_max=1e-6, phi_precond_mode="pullback", weight_decay=0.0,
    )
    opt._collect_gauge_diag = True
    U.grad = torch.zeros_like(U)
    U.grad[1, 0, 0] = 1.0

    opt.step()

    assert opt._gauge_diag["omega_symplectic_residual_max"] > 0.1


def test_omega_direct_full_model_forward_sp_spn_tied():
    # Phase 2 Task 3: full-model forward smoke for the three skew_symmetric=False groups whose
    # omega_direct reach is REAL and live via the ordinary M-step optimizer (Task 2 opened the
    # config gate; this proves each group actually builds and runs end-to-end under omega_direct).
    tok = torch.randint(0, 6, (1, 4), generator=torch.Generator().manual_seed(2))
    for over in (dict(gauge_group="sp", embed_dim=4, n_heads=1),
                 dict(gauge_group="sp_n", embed_dim=5, n_heads=1, group_n=4,
                      irrep_spec=[("sym0", 1), ("sym1", 1)]),          # dims [1,4] sum to embed_dim=5
                 dict(gauge_group="tied_block_glk", embed_dim=4, n_heads=2)):
        torch.manual_seed(0)
        m = VFEModel(_cfg(gauge_parameterization="omega_direct", use_head_mixer=False, **over))
        with torch.no_grad():
            logits = m(tok)[0]
        assert torch.isfinite(logits).all()


def test_omega_direct_so_k_orthogonal_and_reflection_reach():
    grp = get_group("so_k")(K=4)
    U = retract_omega(torch.eye(4).expand(1, 3, 4, 4).contiguous(),
                      0.2 * torch.randn(1, 3, grp.generators.shape[0], generator=torch.Generator().manual_seed(4)),
                      grp.generators)
    assert torch.allclose(U @ U.transpose(-1, -2), torch.eye(4).expand_as(U @ U.transpose(-1, -2)), atol=1e-4)
    assert (torch.det(U) > 0).all()                            # retraction stays in SO(4)
    # init_seed reaches det<0 (O(4)\SO(4)) via reflection_element
    R = reflection_element(4)
    assert torch.det(R @ U[0, 0]) < 0


def test_omega_reorth_projects_drifted_element_back_to_O_K():
    # a slightly non-orthogonal element (fp32-drift analog) is re-orthogonalized by the helper/optimizer path
    U = torch.eye(4) + 0.05 * torch.randn(4, 4, generator=torch.Generator().manual_seed(5))
    # polar factor: U = Q P, Q orthogonal
    from vfe3.gauge_optim import _polar_orthogonalize
    Q = _polar_orthogonalize(U.unsqueeze(0))[0]
    assert torch.allclose(Q @ Q.transpose(-1, -2), torch.eye(4), atol=1e-5)


def test_gauge_optim_omega_reorth_fires_on_cadence_for_single_block_skew(monkeypatch):
    """Integration test: STEPS the optimizer (not just the _polar_orthogonalize helper) for a
    single-block skew group (so_k, irrep_dims=[4]) with omega_reorth_every=2, and confirms polar
    reorth fires exactly on the cadence step, not before -- the ONLY case where reorth is correct
    (rho(SO(K)) = SO(K), so O(K) equals the structure group up to the reflection component)."""
    import vfe3.gauge_optim as gauge_optim_mod
    from vfe3.gauge_optim import GaugeManifoldAdamW
    grp = get_group("so_k")(K=4)
    assert grp.irrep_dims == [4]                              # single block
    calls = []
    orig_polar = gauge_optim_mod._polar_orthogonalize
    def _spy(U):
        calls.append(1)
        return orig_polar(U)
    monkeypatch.setattr(gauge_optim_mod, "_polar_orthogonalize", _spy)

    U = torch.nn.Parameter(torch.eye(4).expand(3, 4, 4).contiguous())
    opt = GaugeManifoldAdamW([{"params": [U], "lr": 0.05, "omega": True, "weight_decay": 0.0}],
                                grp, phi_group_trust_radius=0.1,
                                phi_chart_max_norm=5.0, phi_bch_residual_max=1e-6,
                                phi_precond_mode="pullback", omega_reorth_every=2,
                                weight_decay=0.0)
    gen = torch.Generator().manual_seed(3)
    U.grad = torch.zeros_like(U)
    U.grad[0] = 0.3 * torch.randn(4, 4, generator=gen)        # drifting grad, step 1
    opt.step()                                                 # M-step 1: cadence not hit (1 % 2 != 0)
    assert len(calls) == 0
    U.grad = torch.zeros_like(U)
    U.grad[1] = 0.3 * torch.randn(4, 4, generator=gen)        # drifting grad, step 2
    opt.step()                                                 # M-step 2: cadence hit (2 % 2 == 0)
    assert len(calls) == 1
    eye = torch.eye(4).expand(3, 4, 4)
    assert torch.allclose(U.data @ U.data.transpose(-1, -2), eye, atol=1e-5)   # snapped back onto O(4)


def test_gauge_optim_omega_reorth_is_noop_for_irrep_tower():
    """FIX (whole-branch review): for an irrep TOWER (so_n, len(irrep_dims) > 1) the stored
    omega_embed is a faithful rho(SO(N)) image -- a proper submanifold of O(K) -- so
    _polar_orthogonalize's nearest-O(K) projection is NOT guaranteed to stay in that image. The
    reorth block must gate off (len(irrep_dims) == 1) and be a no-op here, even though
    skew_symmetric=True (a skew-only gate would have fired and silently relaxed the structure
    group)."""
    from vfe3.gauge_optim import GaugeManifoldAdamW
    grp = get_group("so_n")(K=6, group_n=3, irrep_spec=[("l1", 2)])   # SO(3) l1 x2 -> dims [3,3]
    assert grp.irrep_dims == [3, 3]                            # multi-block tower
    assert grp.skew_symmetric is True

    U = torch.nn.Parameter(torch.eye(6).expand(2, 6, 6).contiguous())
    opt = GaugeManifoldAdamW([{"params": [U], "lr": 0.05, "omega": True, "weight_decay": 0.0}],
                                grp, phi_group_trust_radius=0.1,
                                phi_chart_max_norm=5.0, phi_bch_residual_max=1e-6,
                                phi_precond_mode="pullback", omega_reorth_every=1,
                                weight_decay=0.0)
    with torch.no_grad():
        U.data[0, :3, :3] = 1.2 * torch.eye(3)                # deliberately non-orthogonal, inactive row
    non_orth_before = U.data[0].clone()
    U.grad = torch.zeros_like(U)
    U.grad[1] = 0.1 * torch.randn(6, 6, generator=torch.Generator().manual_seed(2))  # only row 1 active
    opt.step()                                                 # reorth_every=1: cadence hit every step
    # row 0 (inactive, deliberately off O(6)) must be left EXACTLY as found -- NOT snapped to the
    # nearest O(6) matrix -- because len(irrep_dims) > 1 gates the reorth off entirely.
    assert torch.equal(U.data[0], non_orth_before)
    assert not torch.allclose(U.data[0] @ U.data[0].transpose(-1, -2), torch.eye(6), atol=1e-3)


def test_omega_compact_storage_param_parity_and_assembly():
    pb_full = PriorBank(vocab_size=6, K=4, n_gen=8, gauge_parameterization="omega_direct", irrep_dims=[2, 2])
    pb_cmp  = PriorBank(vocab_size=6, K=4, n_gen=8, gauge_parameterization="omega_direct", irrep_dims=[2, 2],
                        omega_compact_storage=True, gauge_group_name="block_glk")
    assert pb_full.omega_embed.shape == (6, 4, 4)             # full (V,K,K)
    assert pb_cmp.omega_embed.shape == (6, 2, 2, 2)           # compact (V,H,d,d)
    assert pb_cmp.omega_embed.numel() == 6 * 8                # == V * n_gen (matches phi_embed)
    # identity init assembles to the block-diagonal identity element
    tok = torch.zeros(1, 3, dtype=torch.long)
    om = pb_cmp.encode(tok).omega
    assert om.shape == (1, 3, 4, 4)
    assert isinstance(om, CompactBlockElement)
    assert torch.allclose(om.to_dense(), torch.eye(4).expand(1, 3, 4, 4), atol=1e-7)
    # off-blocks are exactly zero for a non-identity compact frame
    with torch.no_grad():
        pb_cmp.omega_embed[0, 0] = torch.tensor([[1.2, 0.3], [0.0, 0.9]])
    om2 = pb_cmp.encode(torch.zeros(1, 1, dtype=torch.long)).omega[0, 0].to_dense()
    assert torch.allclose(om2[:2, 2:], torch.zeros(2, 2)) and torch.allclose(om2[2:, :2], torch.zeros(2, 2))
    # block 0 carries the edited compact block; block 1 stays identity (independent heads)
    assert torch.allclose(om2[:2, :2], torch.tensor([[1.2, 0.3], [0.0, 0.9]]), atol=1e-7)
    assert torch.allclose(om2[2:, 2:], torch.eye(2), atol=1e-7)


def test_tied_compact_element_rejects_partial_logical_matrix_slice():
    blocks = torch.eye(2).expand(2, 3, 2, 2).clone()
    element = CompactBlockElement(blocks, K=4, tied=True)

    leading = element[:, :1]
    explicit_full_matrix = element[..., :, :]
    assert leading.shape == (2, 1, 4, 4)
    assert torch.equal(leading.blocks, blocks[:, :1])
    assert torch.equal(explicit_full_matrix.blocks, blocks)
    with pytest.raises(ValueError, match="matrix axes intact.*to_dense"):
        element[..., :1, :1]


def test_tied_compact_element_rejects_multiaxis_boolean_mask():
    blocks = torch.eye(2).expand(2, 3, 2, 2).clone()
    element = CompactBlockElement(blocks, K=4, tied=True)
    mask_BN = torch.tensor([[True, False, True], [False, True, False]])

    with pytest.raises(ValueError, match="boolean advanced indexing.*blocks.*to_dense"):
        element[mask_BN, :2]


@pytest.mark.parametrize("owner", [None, "so_n", "sp_n", "unknown"])
def test_omega_compact_storage_requires_block_glk_owner(owner):
    with pytest.raises(ValueError, match="explicit gauge_group_name"):
        PriorBank(
            vocab_size=6, K=4, n_gen=8, gauge_parameterization="omega_direct",
            irrep_dims=[2, 2], omega_compact_storage=True, gauge_group_name=owner,
        )


@pytest.mark.parametrize(
    "owner,tied",
    [("block_glk", True), ("tied_block_glk", False)],
)
def test_omega_compact_storage_rejects_inconsistent_tie_metadata(owner, tied):
    with pytest.raises(ValueError, match="inconsistent"):
        PriorBank(
            vocab_size=6, K=4, n_gen=4, gauge_parameterization="omega_direct",
            irrep_dims=[2, 2], omega_compact_storage=True,
            gauge_group_name=owner, gauge_group_is_tied=tied,
        )


def test_compact_transport_inverts_blocks_without_dense_K_matrix(monkeypatch):
    import vfe3.geometry.lie_ops as lie_ops_module

    K, H, d = 6, 3, 2
    grp = get_group("block_glk")(K=K, n_heads=H)
    pb = PriorBank(
        vocab_size=6, K=K, n_gen=H * d * d, gauge_parameterization="omega_direct",
        irrep_dims=[d] * H, omega_compact_storage=True, gauge_group_name="block_glk",
    )
    with torch.no_grad():
        pb.omega_embed.add_(
            0.05 * torch.randn(pb.omega_embed.shape, generator=torch.Generator().manual_seed(17)))

    def _forbid_dense_reconstruction(*args, **kwargs):
        raise AssertionError("compact transport reconstructed a dense K x K element")

    monkeypatch.setattr(lie_ops_module, "_from_equal_diag_blocks", _forbid_dense_reconstruction)
    element = pb._omega_lookup(torch.tensor([[0, 1, 2]]))
    assert isinstance(element, CompactBlockElement)

    seen_inverse_shapes = []
    original_inverse = torch.linalg.inv

    def _inverse_spy(matrix):
        seen_inverse_shapes.append(tuple(matrix.shape[-2:]))
        return original_inverse(matrix)

    monkeypatch.setattr(torch.linalg, "inv", _inverse_spy)
    built = build_transport_from_element(element, grp)

    assert isinstance(built, CompactFactoredTransport)
    assert seen_inverse_shapes == [(d, d)]
    assert built.exp_blocks.shape == (1, 3, H, d, d)
    assert built.inv_blocks.shape == (1, 3, H, d, d)


def test_full_and_compact_omega_transport_match():
    K, H, d, N = 6, 3, 2, 4
    grp = get_group("block_glk")(K=K, n_heads=H)
    gen = torch.Generator().manual_seed(23)
    blocks = torch.eye(d).expand(1, N, H, d, d).clone()
    blocks = blocks + 0.08 * torch.randn(blocks.shape, generator=gen)
    compact = CompactBlockElement(blocks, K)
    compact_transport = build_transport_from_element(compact, grp)
    dense_omega = compact_transport.to_dense_omega()                  # canonical represented operator
    mu = torch.randn(1, N, K, generator=gen)
    sigma_diag = torch.rand(1, N, K, generator=gen) + 0.5
    A = torch.randn(1, N, K, K, generator=gen)
    sigma_full = A @ A.transpose(-1, -2) + 0.5 * torch.eye(K)

    assert torch.allclose(
        transport_mean(compact_transport, mu), transport_mean(dense_omega, mu),
        atol=2e-6, rtol=1e-5,
    )
    assert torch.allclose(
        transport_covariance(compact_transport, sigma_diag),
        transport_covariance(dense_omega, sigma_diag), atol=2e-6, rtol=1e-5,
    )
    assert torch.allclose(
        transport_covariance(compact_transport, sigma_full),
        transport_covariance(dense_omega, sigma_full), atol=2e-5, rtol=1e-5,
    )


def test_compact_factored_unsqueeze_negative_alias_and_matrix_rejection():
    H, d, K, L, M = 2, 2, 4, 2, 3
    exp_blocks = torch.eye(d).expand(L, H, d, d).clone()
    inv_blocks = torch.eye(d).expand(M, H, d, d).clone()
    factored = CompactFactoredTransport(exp_blocks, inv_blocks, K)

    positive = factored.unsqueeze(0)
    negative = factored.unsqueeze(-5)                         # logical (L,M,K,K) rank -> leading alias
    assert torch.equal(positive.exp_blocks, negative.exp_blocks)
    assert torch.equal(positive.inv_blocks, negative.inv_blocks)
    for dim in (1, 2, 3, 4, -1, -2, -3, -4):
        with pytest.raises(ValueError, match="leading axis"):
            factored.unsqueeze(dim)
    assert torch.equal(
        positive.unsqueeze(0).exp_blocks, positive.unsqueeze(-6).exp_blocks)
    assert torch.equal(
        positive.unsqueeze(1).exp_blocks, positive.unsqueeze(-5).exp_blocks)
    with pytest.raises(ValueError, match="leading axis"):
        positive.unsqueeze(-4)


@pytest.mark.parametrize(
    "irrep_dims,match",
    [
        (None, "explicit irrep_dims"),
        ([4], "more than one"),
        ([3, 1], "equal irrep dimensions"),
        ([2, 0], "positive int"),
        ([2, True], "positive int"),
        ([2, 2], r"sum\(irrep_dims\)==K"),
    ],
)
def test_compact_prior_bank_rejects_invalid_requested_layout(irrep_dims, match):
    with pytest.raises(ValueError, match=match):
        PriorBank(
            vocab_size=4, K=5, n_gen=8, gauge_parameterization="omega_direct",
            irrep_dims=irrep_dims, omega_compact_storage=True,
            gauge_group_name="block_glk",
        )


def test_reflection_scope_is_owner_driven_and_public():
    with pytest.warns(UserWarning, match="block-0 probe"):
        block = PriorBank(
            vocab_size=4, K=4, n_gen=8, gauge_parameterization="omega_direct",
            irrep_dims=[2, 2], omega_compact_storage=True,
            gauge_group_name="block_glk", omega_reflection="init_seed",
        )
    towers = [
        PriorBank(
            vocab_size=4, K=4, n_gen=8, gauge_parameterization="omega_direct",
            irrep_dims=[2, 2], gauge_group_name=owner, omega_reflection="init_seed",
        )
        for owner in ("so_n", "sp_n")
    ]

    assert block.reflection_scope == "block_0_probe"
    assert all(tower.reflection_scope == "full_element" for tower in towers)


@pytest.mark.parametrize(
    "builder_kwargs",
    [
        {"n_heads": 1},
        {"n_heads": 2, "cross_couplings": [(0, 1)]},
    ],
    ids=["one_head", "cross_coupled"],
)
def test_single_block_block_glk_reflection_scope_is_full_element(builder_kwargs):
    group = get_group("block_glk")(K=4, **builder_kwargs)
    assert group.irrep_dims == [4]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bank = PriorBank(
            vocab_size=4, K=4, n_gen=group.generators.shape[0],
            gauge_parameterization="omega_direct", irrep_dims=list(group.irrep_dims),
            gauge_group_name=group.name, omega_reflection="init_seed",
        )

    assert bank.reflection_scope == "full_element"
    assert not caught


@pytest.mark.parametrize("L,M,d", [(2, 5, 2), (3, 2, 4)])
def test_compact_mixed_query_key_transport_matches_dense_oracle(L, M, d, monkeypatch):
    import vfe3.geometry.transport as transport_module

    H, K = 2, 2 * d
    gen = torch.Generator().manual_seed(101 + d)
    exp_blocks = torch.eye(d).expand(1, L, H, d, d).clone()
    inv_blocks = torch.eye(d).expand(1, M, H, d, d).clone()
    exp_blocks = exp_blocks + 0.04 * torch.randn(exp_blocks.shape, generator=gen)
    inv_blocks = inv_blocks + 0.04 * torch.randn(inv_blocks.shape, generator=gen)
    compact = CompactFactoredTransport(exp_blocks, inv_blocks, K)
    dense = compact.to_dense_omega()
    mu = torch.randn(1, M, K, generator=gen)
    sigma_diag = torch.rand(1, M, K, generator=gen) + 0.5
    A = torch.randn(1, M, K, K, generator=gen)
    sigma_full = A @ A.transpose(-1, -2) + 0.5 * torch.eye(K)

    assert torch.allclose(transport_mean(compact, mu), transport_mean(dense, mu), atol=2e-6, rtol=1e-5)
    original_pairs = transport_module._compact_pair_blocks

    def _forbid_pairs(*args, **kwargs):
        raise AssertionError("diagonal covariance allocated pair blocks")

    monkeypatch.setattr(transport_module, "_compact_pair_blocks", _forbid_pairs)
    assert torch.allclose(
        transport_covariance(compact, sigma_diag),
        transport_covariance(dense, sigma_diag), atol=2e-6, rtol=1e-5)
    monkeypatch.setattr(transport_module, "_compact_pair_blocks", original_pairs)
    assert torch.allclose(
        transport_covariance(compact, sigma_full),
        transport_covariance(dense, sigma_full), atol=2e-5, rtol=1e-5)


def test_compact_diagonal_covariance_d_greater_than_keys_never_builds_pairs(monkeypatch):
    import vfe3.geometry.transport as transport_module

    L, M, H, d, K = 3, 2, 2, 4, 8
    gen = torch.Generator().manual_seed(107)
    exp_blocks = torch.eye(d).expand(1, L, H, d, d).clone()
    inv_blocks = torch.eye(d).expand(1, M, H, d, d).clone()
    exp_blocks += 0.03 * torch.randn(exp_blocks.shape, generator=gen)
    inv_blocks += 0.03 * torch.randn(inv_blocks.shape, generator=gen)
    compact = CompactFactoredTransport(exp_blocks, inv_blocks, K)
    dense = compact.to_dense_omega()
    sigma = torch.rand(1, M, K, generator=gen) + 0.5

    def _forbid_pairs(*args, **kwargs):
        raise AssertionError("diagonal covariance allocated pair blocks")

    monkeypatch.setattr(transport_module, "_compact_pair_blocks", _forbid_pairs)
    got = transport_covariance(compact, sigma)
    expected = transport_covariance(dense, sigma)
    assert torch.allclose(got, expected, atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize("representation", ["dense_vertex", "compact"])
def test_uncertified_arbitrary_factored_factors_do_not_overwrite_self_links(
    representation: str,
) -> None:
    N, H, d, K = 3, 2, 2, 4
    if representation == "compact":
        exp_blocks = 2.0 * torch.eye(d).expand(N, H, d, d).clone()
        inv_blocks = 3.0 * torch.eye(d).expand(N, H, d, d).clone()
        factored = CompactFactoredTransport(exp_blocks, inv_blocks, K)
    else:
        exp_phi = 2.0 * torch.eye(K).expand(N, K, K).clone()
        exp_neg_phi = 3.0 * torch.eye(K).expand(N, K, K).clone()
        factored = FactoredTransport(exp_phi, exp_neg_phi, irrep_dims=[d] * H)
    mu = torch.arange(N * K, dtype=torch.float32).reshape(N, K) + 1.0
    sigma = torch.arange(N * K, dtype=torch.float32).reshape(N, K) + 0.5
    self_links = torch.arange(N)

    mean = transport_mean(factored, mu)
    covariance = transport_covariance(factored, sigma, diagonal_out=True)

    assert not factored.same_frame_flat_cocycle
    assert torch.equal(mean[self_links, self_links], 6.0 * mu)
    assert torch.equal(covariance[self_links, self_links], 36.0 * sigma)


def test_compact_full_gauge_rope_covariance_matches_dense_oracle():
    N, H, d, K = 3, 2, 2, 4
    gen = torch.Generator().manual_seed(109)
    blocks = torch.eye(d).expand(N, H, d, d).clone()
    blocks += 0.04 * torch.randn(blocks.shape, generator=gen)
    compact = CompactFactoredTransport(blocks, torch.linalg.inv(blocks.double()).to(blocks.dtype), K)
    dense = compact.to_dense_omega()
    angles = torch.tensor([0.1, 0.3, 0.6])
    rotations = torch.stack([
        torch.tensor([[torch.cos(a), -torch.sin(a)], [torch.sin(a), torch.cos(a)]])
        for a in angles
    ])
    rope_blocks = rotations.unsqueeze(1).expand(N, H, d, d)
    rope = CompactBlockElement(rope_blocks, K).to_dense()
    A = torch.randn(N, K, K, generator=gen)
    sigma = A @ A.transpose(-1, -2) + 0.5 * torch.eye(K)

    got = transport_covariance(RopeTransport(compact, rope, on_cov=True), sigma)
    expected = transport_covariance(RopeTransport(dense, rope, on_cov=True), sigma)
    assert torch.allclose(got, expected, atol=2e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "representation,L,M",
    [("compact", 2, 3), ("dense", 3, 2)],
)
def test_rope_transport_rejects_nonsquare_query_key_base(representation, L, M):
    H, d, K = 2, 2, 4
    exp_blocks = torch.eye(d).expand(L, H, d, d).clone()
    inv_blocks = torch.eye(d).expand(M, H, d, d).clone()
    compact = CompactFactoredTransport(exp_blocks, inv_blocks, K)
    base = compact if representation == "compact" else compact.to_dense_omega()
    rope = torch.eye(K).expand(L, K, K).clone()

    with pytest.raises(ValueError, match="square token transport"):
        RopeTransport(base, rope)


@pytest.mark.parametrize("representation", ["compact", "dense"])
def test_rope_transport_rejects_mismatched_rotation_length(representation):
    N, H, d, K = 3, 2, 2, 4
    blocks = torch.eye(d).expand(N, H, d, d).clone()
    compact = CompactFactoredTransport(blocks, blocks.clone(), K)
    base = compact if representation == "compact" else compact.to_dense_omega()
    rope = torch.eye(K).expand(N - 1, K, K).clone()

    with pytest.raises(ValueError, match="rope token length"):
        RopeTransport(base, rope)


def test_rope_transport_rejects_dense_singleton_matrix_axis():
    N, K = 3, 4
    base = torch.zeros(N, N, 1, K)
    rope = torch.eye(K).expand(N, K, K).clone()

    with pytest.raises(ValueError, match="dense base.*square K x K"):
        RopeTransport(base, rope)


@pytest.mark.parametrize(
    "exp_shape,inv_shape,match",
    [
        ((3, 1, 4), (3, 4, 4), "each end in square K x K"),
        ((3, 4, 4), (3, 5, 5), "same K"),
        ((1, 3, 4, 4), (2, 3, 4, 4), "matching leading batch shapes"),
    ],
    ids=["singleton_matrix_axis", "different_K", "different_batch_shape"],
)
def test_rope_transport_rejects_malformed_factored_base(exp_shape, inv_shape, match):
    base = FactoredTransport(
        torch.zeros(exp_shape), torch.zeros(inv_shape), irrep_dims=[2, 2])
    rope = torch.eye(4).expand(3, 4, 4).clone()

    with pytest.raises(ValueError, match=match):
        RopeTransport(base, rope)


def test_compact_sampled_metrics_match_canonical_dense_operator():
    import vfe3.metrics as metrics_module

    N, H, d, K = 5, 2, 2, 4
    gen = torch.Generator().manual_seed(113)
    exp_blocks = torch.eye(d).expand(N, H, d, d).clone()
    inv_blocks = torch.eye(d).expand(N, H, d, d).clone()
    exp_blocks += 0.08 * torch.randn(exp_blocks.shape, generator=gen)
    inv_blocks += 0.07 * torch.randn(inv_blocks.shape, generator=gen)
    compact = CompactFactoredTransport(exp_blocks, inv_blocks, K)
    dense = compact.to_dense_omega()

    assert torch.allclose(
        metrics_module.transport_asymmetry(compact),
        metrics_module.transport_asymmetry(dense), atol=2e-6, rtol=1e-5)
    compact_h = metrics_module.holonomy_deviation_sampled(
        compact, n_triples=24, n_boot=16, seed=7)
    dense_h = metrics_module.holonomy_deviation_sampled(
        dense, n_triples=24, n_boot=16, seed=7)
    assert compact_h.keys() == dense_h.keys()
    for key in compact_h:
        assert torch.allclose(compact_h[key], dense_h[key], atol=2e-5, rtol=1e-5), key
    compact_w = metrics_module.holonomy_wilson_sampled(
        compact, n_heads=H, irrep_dims=[d] * H, n_triples=24, n_boot=16, seed=7)
    dense_w = metrics_module.holonomy_wilson_sampled(
        dense, n_heads=H, irrep_dims=[d] * H, n_triples=24, n_boot=16, seed=7)
    assert compact_w.keys() == dense_w.keys()
    for key in compact_w:
        assert torch.allclose(compact_w[key], dense_w[key], atol=2e-5, rtol=1e-5), key
    assert torch.allclose(
        metrics_module.cocycle_residual_sampled(compact, n_triples=24, seed=7),
        metrics_module.cocycle_residual_sampled(dense, n_triples=24, seed=7),
        atol=2e-5, rtol=1e-5)


@pytest.mark.parametrize("L,M", [(2, 3), (3, 2)])
@pytest.mark.parametrize("representation", ["compact", "dense"])
@pytest.mark.parametrize(
    "metric_name",
    [
        "transport_asymmetry",
        "holonomy_deviation_sampled",
        "holonomy_wilson_sampled",
        "cocycle_residual_sampled",
    ],
)
def test_square_graph_metrics_reject_rectangular_transport(
    metric_name, representation, L, M,
):
    import vfe3.metrics as metrics_module

    H, d, K = 2, 2, 4
    exp_blocks = torch.eye(d).expand(L, H, d, d).clone()
    inv_blocks = torch.eye(d).expand(M, H, d, d).clone()
    compact = CompactFactoredTransport(exp_blocks, inv_blocks, K)
    omega = compact if representation == "compact" else compact.to_dense_omega()
    metric = getattr(metrics_module, metric_name)

    with pytest.raises(ValueError, match=r"square token transport \(Nq == Nk\)"):
        metric(omega)


@pytest.mark.parametrize("representation", ["compact", "dense"])
@pytest.mark.parametrize(
    "metric_name",
    [
        "holonomy_deviation_sampled",
        "holonomy_wilson_sampled",
        "cocycle_residual_sampled",
    ],
)
def test_sampled_graph_metrics_reject_batched_square_transport_before_empty_return(
    metric_name, representation,
):
    import vfe3.metrics as metrics_module

    B, N, H, d, K = 1, 2, 2, 2, 4
    blocks = torch.eye(d).expand(B, N, H, d, d).clone()
    compact = CompactFactoredTransport(blocks, blocks.clone(), K)
    omega = compact if representation == "compact" else compact.to_dense_omega()
    metric = getattr(metrics_module, metric_name)

    with pytest.raises(ValueError, match=r"requires (?:an )?unbatched"):
        metric(omega)


@pytest.mark.parametrize(
    "metric_name",
    [
        "transport_asymmetry",
        "holonomy_deviation_sampled",
        "holonomy_wilson_sampled",
        "cocycle_residual_sampled",
    ],
)
def test_graph_metrics_reject_dense_nonsquare_matrix_axes(metric_name):
    import vfe3.metrics as metrics_module

    omega = torch.zeros(3, 3, 1, 4)
    metric = getattr(metrics_module, metric_name)

    with pytest.raises(ValueError, match="trailing square K x K"):
        metric(omega)


def test_compact_free_energy_and_diagnostics_never_dense_materialize(monkeypatch):
    import vfe3.geometry.transport as transport_module

    m = VFEModel(_cfg(
        gauge_parameterization="omega_direct", gauge_group="block_glk", n_heads=2,
        omega_compact_storage=True, family="gaussian_diagonal", decode_mode="diagonal",
        n_e_steps=1,
    ))
    token_ids = torch.tensor([[0, 1, 2, 3]])
    active_blocks = torch.tensor([
        [[[2.0, 0.0], [0.0, 1.0]], [[0.5, 0.0], [0.0, 1.0]]],
        [[[4.0, 0.0], [0.0, 1.0]], [[0.25, 0.0], [0.0, 2.0]]],
        [[[1.0, 0.0], [0.0, 3.0]], [[2.0, 0.0], [0.0, 0.5]]],
        [[[0.5, 0.0], [0.0, 2.0]], [[1.0, 0.0], [0.0, 4.0]]],
    ])
    with torch.no_grad():
        m.prior_bank.omega_embed[:4].copy_(active_blocks)
        m.prior_bank.phi_embed.fill_(7.0)                     # deliberately unrelated inactive chart
    enc = m.prior_bank.encode(token_ids)
    belief = BeliefState(
        mu=enc.mu[0], sigma=enc.sigma[0], phi=enc.phi[0], omega=enc.omega[0])

    def _forbid_dense(*args, **kwargs):
        raise AssertionError("compact omega crossed an implicit dense compatibility boundary")

    monkeypatch.setattr(CompactBlockElement, "to_dense", _forbid_dense)
    monkeypatch.setattr(CompactFactoredTransport, "to_dense_omega", _forbid_dense)
    monkeypatch.setattr(transport_module, "compute_transport_operators", _forbid_dense)

    value = free_energy_value(
        belief, belief.mu, belief.sigma, m.group,
        family="gaussian_diagonal", gauge_parameterization="omega_direct")
    diagnostics = m.diagnostics(token_ids)

    assert torch.isfinite(value)
    assert all(torch.isfinite(torch.tensor(v)) for v in diagnostics.values())
    block_logdet = torch.linalg.slogdet(active_blocks).logabsdet
    block_svd = torch.linalg.svdvals(active_blocks)
    full_logdet = block_logdet.sum(dim=-1)
    represented_cond = (
        block_svd[..., 0].amax(dim=-1) / block_svd[..., -1].amin(dim=-1))
    anisotropy = block_svd[..., 0] / block_svd[..., -1]
    assert diagnostics["gauge_trace_spread"] == pytest.approx(
        float(full_logdet.std(unbiased=False)), rel=1e-6)
    assert diagnostics["gauge_invariant_mean"] == pytest.approx(float(full_logdet.mean()), rel=1e-6)
    assert diagnostics["gauge_invariant_spread"] == pytest.approx(
        float(full_logdet.std(unbiased=False)), rel=1e-6)
    assert diagnostics["vertex_cond_max"] == pytest.approx(float(represented_cond.max()), rel=1e-6)
    assert diagnostics["gauge_head_aniso_mean"] == pytest.approx(float(anisotropy.mean()), rel=1e-6)
    assert diagnostics["gauge_head_logdet_spread"] == pytest.approx(
        float(block_logdet.std(unbiased=False)), rel=1e-6)


def test_tied_compact_forward_backward_with_gamma_and_rope():
    m = VFEModel(_cfg(
        gauge_parameterization="omega_direct", gauge_group="tied_block_glk", n_heads=2,
        omega_compact_storage=True, family="gaussian_full", decode_mode="full",
        n_e_steps=1, lambda_gamma=0.5, s_e_step=False, pos_rotation="rope",
        rope_full_gauge=True, oracle_unroll_grad=True,
    ))
    token_ids = torch.tensor([[0, 1, 2, 3]])

    logits = m(token_ids)[0]
    logits.square().mean().backward()

    assert torch.isfinite(logits).all()
    assert m.prior_bank.omega_embed.grad is not None
    assert torch.isfinite(m.prior_bank.omega_embed.grad).all()


def test_omega_compact_tied_shares_one_block():
    pb = PriorBank(vocab_size=6, K=4, n_gen=4, gauge_parameterization="omega_direct", irrep_dims=[2, 2],
                   omega_compact_storage=True, gauge_group_is_tied=True,
                   gauge_group_name="tied_block_glk")   # tied flag threaded from model.py
    assert pb.omega_embed.shape == (6, 2, 2)                  # (V,d,d) one shared block
    with torch.no_grad():                                     # a non-identity shared block
        pb.omega_embed[0] = torch.tensor([[1.3, 0.2], [0.0, 0.8]])
    om = pb.encode(torch.zeros(1, 1, dtype=torch.long)).omega[0, 0].to_dense()
    assert torch.allclose(om[:2, :2], om[2:, 2:], atol=1e-7)  # same block in both heads
    assert torch.allclose(om[:2, 2:], torch.zeros(2, 2)) and torch.allclose(om[2:, :2], torch.zeros(2, 2))


def test_omega_compact_optimizer_step_equals_full_step_on_blocks():
    """One optimizer step on the compact (V,H,d,d) table equals the full (V,K,K) step restricted
    to the blocks (fp): the per-block gl(d) retraction is the block-diagonal gl(K) retraction."""
    from vfe3.gauge_optim import GaugeManifoldAdamW
    from vfe3.geometry.lie_ops import _from_equal_diag_blocks
    V, H, d, K = 6, 2, 2, 4
    group = get_group("block_glk")(K=K, n_heads=H)
    G_full = group.generators                                  # (8, 4, 4) block_glk basis
    g = torch.Generator().manual_seed(11)
    blocks = torch.eye(d).expand(V, H, d, d).clone() + 0.1 * torch.randn(V, H, d, d, generator=g)  # (V,H,d,d)
    Eb = torch.randn(V, H, d, d, generator=g)                 # per-block gradient
    Eb[3] = 0.0                                               # one inactive row (exercises the active mask)

    pb_cmp  = PriorBank(V, K, H * d * d, gauge_parameterization="omega_direct", irrep_dims=[d] * H,
                        omega_compact_storage=True, gauge_group_name="block_glk")
    pb_full = PriorBank(V, K, H * d * d, gauge_parameterization="omega_direct", irrep_dims=[d] * H)
    with torch.no_grad():
        pb_cmp.omega_embed.copy_(blocks)
        pb_full.omega_embed.copy_(_from_equal_diag_blocks(blocks, K))     # same element, dense
    pb_cmp.omega_embed.grad  = Eb.clone()
    pb_full.omega_embed.grad = _from_equal_diag_blocks(Eb, K)             # same grad, off-blocks zero

    def _opt(pb):
        return GaugeManifoldAdamW([{"params": [pb.omega_embed], "lr": 0.1, "omega": True,
                                       "weight_decay": 0.0}], group,
                                      phi_group_trust_radius=0.1, phi_chart_max_norm=5.0,
                                      phi_bch_residual_max=1e-6,
                                      phi_precond_mode="pullback_per_block", weight_decay=0.0)
    _opt(pb_cmp).step()
    _opt(pb_full).step()

    assembled = _from_equal_diag_blocks(pb_cmp.omega_embed.data, K)       # (V,K,K)
    assert torch.allclose(assembled, pb_full.omega_embed.data, atol=1e-5)
    assert torch.allclose(pb_cmp.omega_embed.data[3], blocks[3], atol=1e-6)   # inactive row untouched


def test_omega_compact_tied_optimizer_step_moves_shared_block():
    """A tied (V,d,d) compact table (dim 3) must NOT fall through to the full (V,K,K) path -- it
    steps directly on the shared gl(d) block. Regression for the einsum size-mismatch crash the
    dim-only detection caused (RuntimeError: einsum ... size 2 ... does not broadcast ... size 4)."""
    from vfe3.gauge_optim import GaugeManifoldAdamW
    V, H, d, K = 6, 2, 2, 4
    group = get_group("tied_block_glk")(K=K, n_heads=H)
    G_full = group.generators                                # (4, 4, 4) tied basis; last dim == K
    pb = PriorBank(V, K, d * d, gauge_parameterization="omega_direct", irrep_dims=[d] * H,
                   omega_compact_storage=True, gauge_group_is_tied=True,
                   gauge_group_name="tied_block_glk")
    assert pb.omega_embed.shape == (V, d, d)                 # (V,d,d), dim 3
    g = torch.Generator().manual_seed(7)
    grad = torch.randn(V, d, d, generator=g)
    grad[3] = 0.0                                            # one inactive row
    pb.omega_embed.grad = grad.clone()
    before = pb.omega_embed.data.clone()
    opt = GaugeManifoldAdamW([{"params": [pb.omega_embed], "lr": 0.1, "omega": True,
                                  "weight_decay": 0.0}], group,
                                 phi_group_trust_radius=0.1, phi_chart_max_norm=5.0,
                                 phi_bch_residual_max=1e-6, phi_precond_mode="pullback",
                                 weight_decay=0.0)
    opt.step()                                               # MUST NOT crash
    U = pb.omega_embed.data
    assert not torch.allclose(U[0], before[0])               # active shared block moved
    assert torch.allclose(U[3], before[3], atol=1e-6)        # inactive row untouched
    assert torch.det(U[0]) > 0                               # stays in GL+(d)
    # the assembled frame still shares the ONE stepped block across both heads, off-blocks zero
    om = pb.encode(torch.zeros(1, 1, dtype=torch.long)).omega[0, 0].to_dense()
    assert torch.allclose(om[:2, :2], om[2:, 2:], atol=1e-7)
    assert torch.allclose(om[:2, 2:], torch.zeros(2, 2)) and torch.allclose(om[2:, :2], torch.zeros(2, 2))


def test_omega_compact_tied_step_magnitude_equals_full_tied_step():
    """The compact-tied step must MATCH the full-tied step, not just move. The full tied generators
    kron(I_H, E_ij) have Frobenius Gram = H*I, so the full-tied natural gradient carries a 1/H the
    intrinsic gl(d) basis (Gram = I) omits; without the 1/H rescale the compact-tied step is H x too
    large. Asserts assemble(compact_stepped) == full_stepped, tight tol (the H x bug would fail this)."""
    from vfe3.gauge_optim import GaugeManifoldAdamW
    from vfe3.geometry.lie_ops import _from_equal_diag_blocks
    V, H, d, K = 6, 2, 2, 4
    group = get_group("tied_block_glk")(K=K, n_heads=H)
    G_tied = group.generators                                # (4,4,4) tied basis; Frobenius Gram = H*I
    # sanity: the tied Gram really is H*I (the load-bearing fact behind the 1/H)
    assert torch.allclose(torch.einsum("aij,bij->ab", G_tied, G_tied),
                          float(H) * torch.eye(d * d), atol=1e-6)
    gen = torch.Generator().manual_seed(13)
    per_slot = torch.randn(V, H, d, d, generator=gen)        # H distinct per-slot block grads
    E_full = _from_equal_diag_blocks(per_slot, K)            # (V,K,K) block-diagonal dense grad
    E_cmp  = per_slot.sum(dim=1)                             # (V,d,d) broadcast adjoint = sum over slots

    pb_full = PriorBank(V, K, d * d, gauge_parameterization="omega_direct", irrep_dims=[d] * H)  # (V,K,K) I init
    pb_cmp  = PriorBank(V, K, d * d, gauge_parameterization="omega_direct", irrep_dims=[d] * H,
                        omega_compact_storage=True, gauge_group_is_tied=True,
                        gauge_group_name="tied_block_glk")                     # (V,d,d) I init
    pb_full.omega_embed.grad = E_full.clone()
    pb_cmp.omega_embed.grad  = E_cmp.clone()

    def _opt(pb):
        return GaugeManifoldAdamW([{"params": [pb.omega_embed], "lr": 0.1, "omega": True,
                                       "weight_decay": 0.0}], group,
                                      phi_group_trust_radius=0.1, phi_chart_max_norm=5.0,
                                      phi_bch_residual_max=1e-6, phi_precond_mode="pullback",
                                      weight_decay=0.0)
    _opt(pb_full).step()
    _opt(pb_cmp).step()

    assembled = pb_cmp._omega_lookup(torch.arange(V).unsqueeze(0))[0].to_dense()
    assert torch.allclose(assembled, pb_full.omega_embed.data, atol=1e-5)


def test_omega_compact_flag_is_noop_for_equal_dim_towers():
    """so_n/sp_n equal-dim irrep towers (e.g. irrep_dims=[3,3]) match the block STRUCTURE but are
    irrep IMAGES of one element, not independent blocks -- omega_compact_storage must be a NO-OP for
    them (full (V,K,K) table), decided by the group gate in model.py. A compacted (V,H,d,d) tower
    would break the tower gauge and void param parity (V*H*d^2 != V*n_gen)."""
    def _tower_cfg(**over):
        base = dict(vocab_size=6, n_heads=1, max_seq_len=4, n_layers=1, n_e_steps=2,
                    gauge_parameterization="omega_direct", family="gaussian_full", transport_mode="flat",
                    pos_rotation="none", use_head_mixer=False, use_prior_bank=True, decode_mode="full",
                    pos_phi="none", e_phi_lr=0.0)
        base.update(over)
        return VFE3Config(**base)
    over = dict(gauge_group="so_n", embed_dim=6, group_n=3, irrep_spec=[("l1", 2)])   # SO(3) l1 x2 -> [3,3]
    m_off = VFEModel(_tower_cfg(omega_compact_storage=False, **over))
    m_on  = VFEModel(_tower_cfg(omega_compact_storage=True,  **over))
    assert list(m_on.group.irrep_dims) == [3, 3]             # equal-dim tower: matches the block structure
    assert m_on.prior_bank._omega_compact is False           # flag suppressed for the tower
    assert m_on.prior_bank.omega_embed.shape == (6, 6, 6)    # full (V,K,K), NOT compacted to (6,2,3,3)
    # param count unaffected by the flag (both full (V,K,K))
    assert m_on.prior_bank.omega_embed.numel() == m_off.prior_bank.omega_embed.numel()


def test_omega_compact_tied_backward_sums_head_slot_gradients():
    """End-to-end autograd through the explicit compatibility conversion: every other compact-storage
    test sets p.grad manually; none does a real loss.backward() through dense compatibility output.
    Encodes a token batch, explicitly converts belief.omega to (B,N,K,K), and checks
    omega_embed.grad has the right shape AND -- for the TIED shared (V,d,d) block -- equals the SUM
    over the H head-slots of the assembled element's own gradient (the tied broadcast adjoint).
    Weights the loss with random per-entry coefficients (not a
    plain .sum()) so a bug that averaged instead of summed over H, or dropped a head slot, would not
    be masked by an all-ones gradient."""
    V, H, d, K = 6, 2, 2, 4
    pb = PriorBank(V, K, d * d, gauge_parameterization="omega_direct", irrep_dims=[d] * H,
                   omega_compact_storage=True, gauge_group_is_tied=True,
                   gauge_group_name="tied_block_glk")
    assert pb.omega_embed.shape == (V, d, d)                  # (V,d,d): one shared block

    tok = torch.tensor([[0, 1, 2]])                            # (1,3), distinct tokens -> clean per-token check
    belief = pb.encode(tok)
    assert belief.omega.shape == (1, 3, K, K)

    w = torch.randn(1, 3, K, K, generator=torch.Generator().manual_seed(0))  # random per-entry weights
    loss = (belief.omega.to_dense() * w).sum()
    loss.backward()

    assert pb.omega_embed.grad is not None
    assert pb.omega_embed.grad.shape == (V, d, d)             # correct shape: broadcast adjoint onto the compact table
    for n, v in enumerate(tok[0].tolist()):
        # broadcast adjoint: dL/d(shared block) = SUM over the H diagonal head-slots of w's own block
        expected = sum(w[0, n, h * d:(h + 1) * d, h * d:(h + 1) * d] for h in range(H))
        assert torch.allclose(pb.omega_embed.grad[v], expected, atol=1e-6)
    untouched = [v for v in range(V) if v not in tok[0].tolist()][0]
    assert torch.allclose(pb.omega_embed.grad[untouched], torch.zeros(d, d))  # no gradient for unseen tokens


def test_ablation_omega_direct_arm_builds():
    # Build every cell of the "gauge_parameterization" sweep the way the runner does -- baseline
    # (BASELINE_CONFIG, which sets lambda_gamma=0.75 and s_e_step=True) merged with the arm
    # overrides -- so this exercises the real BASELINE_CONFIG interaction. The arm fans out one
    # phi baseline plus one omega_direct cell per gauge_group in vfe3.config's omega-eligible set
    # (glk, block_glk, tied_block_glk, so_k, sp, so_n, sp_n). Phase 3 gave omega_direct s-channel
    # frame-fidelity, so every cell now INHERITS BASELINE_CONFIG's gamma-on settings (no per-cell
    # lambda_gamma/s_e_step override) -- an apples-to-apples gamma-on comparison across gauge charts.
    import ablation

    runs = ablation.make_run_overrides("gauge_parameterization")
    labels = [label for label, _ in runs]
    assert len(labels) == len(set(labels))                     # no duplicate cell labels
    assert "phi" in labels
    omega_labels = [label for label in labels if label != "phi"]
    assert len(omega_labels) == 7                              # one per _OMEGA_GROUPS entry
    assert all(label.startswith("omega_direct_") for label in omega_labels)

    built = {}
    for label, overrides in runs:
        cfg_dict = ablation._cell_cfg_dict(overrides, seed=0, max_steps=1)
        # Keep all semantic baseline/sweep interactions while making the construction regression
        # small enough for the unit suite (omega_direct otherwise allocates V x K x K at GPT-2 V).
        cfg_dict.update(vocab_size=8, max_seq_len=4, batch_size=1)
        cfg = VFE3Config(**cfg_dict)
        model = VFEModel(cfg)                                  # must not raise for ANY cell
        built[label] = model.cfg

    assert built["phi"].gauge_parameterization == "phi"
    seen_groups = set()
    for label in omega_labels:
        cfg = built[label]
        assert cfg.gauge_parameterization == "omega_direct"
        assert cfg.lambda_gamma == built["phi"].lambda_gamma > 0.0   # inherits BASELINE gamma-on (no per-cell override), apples-to-apples with phi
        assert cfg.s_e_step == built["phi"].s_e_step                 # ditto for the s-channel E-step
        seen_groups.add(cfg.gauge_group)
    assert seen_groups == {"glk", "block_glk", "tied_block_glk", "so_k", "sp", "so_n", "sp_n"}


def test_gauge_parameterization_sweep_disables_positional_phi_for_all_cells():
    import ablation

    runs = ablation.make_run_overrides("gauge_parameterization")

    assert runs
    assert all(overrides["pos_phi"] == "none" for _, overrides in runs)


def test_omega_direct_optimizer_warns_that_phi_update_policy_does_not_apply():
    from vfe3.train import build_optimizer

    model = VFEModel(_cfg(gauge_parameterization="omega_direct"))
    with pytest.warns(UserWarning, match="retraction SGD"):
        build_optimizer(model, model.cfg)


def test_gamma_coupling_term_uses_stored_frame_not_phi_cocycle():
    # Phase 3 Task 1: _gamma_coupling_term (the forward gamma loss body, via _gamma_energy) must
    # transport the s-channel by the STORED belief frame U (belief.omega) under omega_direct, not
    # the flat phi-cocycle exp(phi_i)exp(-phi_j). The config gate (config.py:946) still blocks
    # omega_direct + active gamma, so the model is built with the gamma channel OFF
    # (lambda_gamma=0, s_e_step=False, gamma_as_beta_prior=False -- _cfg's defaults) but with the s
    # tables present via prior_source='model_channel' (which does not trip the gate), then
    # _gamma_coupling_term is called DIRECTLY -- it does not re-check cfg. phi is held at zero
    # (exp(phi)=I identically), so a phi-based rebuild is blind to omega and returns the IDENTICAL
    # energy for any U (pre-fix bug); post-fix, two different frames U1=I, U2 must give DIFFERENT
    # energies.
    K, N = 4, 3
    torch.manual_seed(0)
    m = VFEModel(_cfg(gauge_parameterization="omega_direct", prior_source="model_channel",
                      family="gaussian_diagonal", decode_mode="diagonal"))
    grp = m.group
    n_gen = grp.generators.shape[0]
    tok = torch.randint(0, 6, (1, N), generator=torch.Generator().manual_seed(1))
    phi = torch.zeros(1, N, n_gen)                               # exp(phi)=I: phi-cocycle is frame-blind
    U1 = torch.eye(K).expand(1, N, K, K).contiguous()            # identity frame
    xi = 0.3 * torch.randn(1, N, n_gen, generator=torch.Generator().manual_seed(2))
    U2 = retract_omega(U1, xi, grp.generators)                   # a DIFFERENT, non-identity frame
    assert not torch.allclose(U2, U1, atol=1e-4)

    e1 = m._gamma_coupling_term(tok, phi, omega=U1)
    e2 = m._gamma_coupling_term(tok, phi, omega=U2)
    assert not torch.allclose(e1, e2, atol=1e-6), \
        "gamma coupling term is blind to the stored omega frame (still using the phi cocycle)"


def test_gamma_coupling_terms_split_uses_stored_frame_not_phi_cocycle():
    # Same defect, the diagnostic split sibling _gamma_coupling_terms (feeds model.py:1758's
    # d["gamma_coupling"]/d["gamma_meta_entropy"]). Mirrors the test above.
    K, N = 4, 3
    torch.manual_seed(0)
    m = VFEModel(_cfg(gauge_parameterization="omega_direct", prior_source="model_channel",
                      family="gaussian_diagonal", decode_mode="diagonal"))
    grp = m.group
    n_gen = grp.generators.shape[0]
    tok = torch.randint(0, 6, (1, N), generator=torch.Generator().manual_seed(1))
    phi = torch.zeros(1, N, n_gen)
    U1 = torch.eye(K).expand(1, N, K, K).contiguous()
    xi = 0.3 * torch.randn(1, N, n_gen, generator=torch.Generator().manual_seed(2))
    U2 = retract_omega(U1, xi, grp.generators)

    g1 = m._gamma_coupling_terms(tok, phi, omega=U1)
    g2 = m._gamma_coupling_terms(tok, phi, omega=U2)
    assert not torch.allclose(g1["total"], g2["total"], atol=1e-6), \
        "gamma coupling split is blind to the stored omega frame (still using the phi cocycle)"


def test_fold_gamma_prior_uses_stored_frame_not_phi_cocycle():
    # Phase 3 Task 2: _fold_gamma_prior (the gamma_as_beta_prior forward-VALUE fold, via
    # _gamma_energy) must transport the s-channel by the STORED belief frame U (belief.omega)
    # under omega_direct, not the flat phi-cocycle exp(phi_i)exp(-phi_j). The config gate
    # (config.py:946) still blocks omega_direct + active gamma, so the model is built with the
    # gamma channel OFF (lambda_gamma=0, s_e_step=False, gamma_as_beta_prior=False -- _cfg's
    # defaults) but with the s tables present via prior_source='model_channel' (which does not
    # trip the gate), then _fold_gamma_prior is called DIRECTLY -- it does not re-check cfg. phi
    # is held at zero (exp(phi)=I identically), so a phi-based rebuild is blind to omega and
    # returns the IDENTICAL log_prior for any U (pre-fix bug); post-fix, two different frames
    # U1=I, U2 must give DIFFERENT log_prior. _fold_gamma_prior already runs under torch.no_grad,
    # so there is no gradient concern here.
    K, N = 4, 3
    torch.manual_seed(0)
    m = VFEModel(_cfg(gauge_parameterization="omega_direct", prior_source="model_channel",
                      family="gaussian_diagonal", decode_mode="diagonal"))
    grp = m.group
    n_gen = grp.generators.shape[0]
    tok = torch.randint(0, 6, (1, N), generator=torch.Generator().manual_seed(1))
    phi = torch.zeros(1, N, n_gen)                               # exp(phi)=I: phi-cocycle is frame-blind
    U1 = torch.eye(K).expand(1, N, K, K).contiguous()            # identity frame
    xi = 0.3 * torch.randn(1, N, n_gen, generator=torch.Generator().manual_seed(2))
    U2 = retract_omega(U1, xi, grp.generators)                   # a DIFFERENT, non-identity frame
    assert not torch.allclose(U2, U1, atol=1e-4)

    lp1 = m._fold_gamma_prior(None, tok, phi, omega=U1)
    lp2 = m._fold_gamma_prior(None, tok, phi, omega=U2)
    assert not torch.allclose(lp1, lp2, atol=1e-6), \
        "fold_gamma_prior is blind to the stored omega frame (still using the phi cocycle)"


def test_refine_s_uses_stored_frame_not_phi_cocycle():
    # Phase 3 Task 3: _refine_s (the s_e_step E-step) must transport the model-coupling (gamma)
    # term by the STORED belief frame U (pb.omega_embed) under omega_direct, not the flat
    # phi-cocycle exp(phi_i)exp(-phi_j). The config gate (config.py:946) still blocks omega_direct +
    # active gamma at CONSTRUCTION time, so the model is built with the gamma channel OFF
    # (lambda_gamma=0, s_e_step=False, gamma_as_beta_prior=False -- _cfg's defaults) plus
    # lambda_h=1.0 (lambda_h is NOT in the gate's reject list, and lambda_h>0 alone -- independent of
    # lambda_gamma/s_e_step -- is what builds the r table _refine_s reads) and
    # prior_source='model_channel' so the s tables exist. cfg.lambda_gamma is then bumped to a
    # nonzero value AFTER construction (the same "method reads cfg live at call time" maneuver the
    # brief uses for the gauge_parameterization gate) so the omega-transported coupling term has
    # nonzero weight: at lambda_gamma=0 the pair/coupling contribution is multiplied by exactly zero
    # (kernels.py: grad_mu = self_mu + lambda_beta*pair_mu, lambda_beta=cfg.lambda_gamma), so the
    # refined s would be independent of omega regardless of the fix -- a vacuous test. phi is held at
    # zero (exp(phi)=I identically), so a phi-based rebuild is blind to omega and returns IDENTICAL
    # refined s for any U (pre-fix bug); post-fix, two different frames U1=I, U2 must give DIFFERENT
    # refined s.
    K, N, V = 4, 3, 6
    torch.manual_seed(0)
    m = VFEModel(_cfg(gauge_parameterization="omega_direct", prior_source="model_channel",
                      family="gaussian_diagonal", decode_mode="diagonal",
                      lambda_h=1.0, lambda_h_mode="constant"))
    m.cfg.lambda_gamma = 0.75                        # bumped post-construction, bypassing the
                                                      # construction-time omega_direct gate
    grp = m.group
    n_gen = grp.generators.shape[0]
    tok = torch.randint(0, V, (1, N), generator=torch.Generator().manual_seed(1))
    phi0 = torch.zeros(1, N, n_gen)                  # exp(phi0)=I: phi-cocycle is frame-blind

    U1 = torch.eye(K).expand(V, K, K).contiguous()   # identity frame for the whole vocab table
    xi = 0.3 * torch.randn(V, n_gen, generator=torch.Generator().manual_seed(2))
    U2 = retract_omega(U1, xi, grp.generators)       # a DIFFERENT, non-identity frame
    assert not torch.allclose(U2, U1, atol=1e-4)

    with torch.no_grad():
        m.prior_bank.omega_embed.copy_(U1)
    s_mu1, s_sigma1 = m._refine_s(tok, phi0)
    with torch.no_grad():
        m.prior_bank.omega_embed.copy_(U2)
    s_mu2, s_sigma2 = m._refine_s(tok, phi0)
    assert not torch.allclose(s_mu1, s_mu2, atol=1e-6), \
        "_refine_s is blind to the stored omega frame (still using the phi cocycle)"

    # Load-bearing gradient-flow pin: omega_s must stay ATTACHED (no .detach()) inside _refine_s so
    # the s E-step trains omega_embed through the unrolled trajectory, exactly as it trains
    # phi_embed. A detached lookup would silently freeze the frame's gradient.
    grad, = torch.autograd.grad(s_mu2.sum(), m.prior_bank.omega_embed)
    assert grad is not None and torch.count_nonzero(grad) > 0


def test_refine_s_phi_path_unaffected():
    # Byte-identity guard (Phase 3 Task 3): under gauge_parameterization='phi', _refine_s must be
    # untouched by the omega threading -- omega_s resolves to None and gauge_parameterization='phi'
    # is passed through, so e_step falls to its pre-existing phi branch. Same construction/mutation
    # recipe as the omega_direct test above (lambda_h=1.0 builds the r table; lambda_gamma bumped
    # post-construction so the coupling term is non-vacuous), but gauge_parameterization='phi' this
    # time -- refined s must be IDENTICAL regardless of what is stashed in omega_embed (indeed no
    # omega_embed table is even created off the omega_direct path).
    K, N, V = 4, 3, 6
    torch.manual_seed(0)
    m = VFEModel(_cfg(gauge_parameterization="phi", prior_source="model_channel",
                      family="gaussian_diagonal", decode_mode="diagonal",
                      lambda_h=1.0, lambda_h_mode="constant"))
    assert not hasattr(m.prior_bank, "omega_embed")   # phi path: no omega table at all
    m.cfg.lambda_gamma = 0.75
    grp = m.group
    n_gen = grp.generators.shape[0]
    tok = torch.randint(0, V, (1, N), generator=torch.Generator().manual_seed(1))
    phi0 = 0.2 * torch.randn(1, N, n_gen, generator=torch.Generator().manual_seed(3))

    s_mu1, s_sigma1 = m._refine_s(tok, phi0)
    s_mu2, s_sigma2 = m._refine_s(tok, phi0)
    assert torch.equal(s_mu1, s_mu2) and torch.equal(s_sigma1, s_sigma2)   # byte-identical, deterministic
