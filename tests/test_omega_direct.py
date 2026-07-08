import pytest
import torch

from vfe3.belief import BeliefState
from vfe3.config import VFE3Config
from vfe3.geometry.generators import generate_glk, generate_son, reflection_element
from vfe3.geometry.groups import get_group
from vfe3.geometry.lie_ops import retract_omega
from vfe3.geometry.transport import (build_transport_from_element, compute_transport_operators,
                                      transport_mean, FactoredTransport)
from vfe3.inference.belief_cache import cache_supported, rollout_predictive_cached
from vfe3.inference.e_step import build_belief_transport, e_step, e_step_iteration, free_energy_value
from vfe3.model.model import VFEModel
from vfe3.model.prior_bank import PriorBank


def _cfg(**over):
    base = dict(vocab_size=6, embed_dim=4, n_heads=1, max_seq_len=4, n_layers=1, n_e_steps=2,
                gauge_group="glk", family="gaussian_full", transport_mode="flat",
                pos_rotation="none", use_head_mixer=False, use_prior_bank=True, decode_mode="full",
                e_phi_lr=0.0)
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
    mu = torch.randn(1, 3, 4)
    mt = transport_mean(built, mu)                           # (1,3,3,4) via the factored fast path
    assert mt.shape == (1, 3, 3, 4)


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


def test_e_step_preserves_omega_across_belief_rebuilds():
    # Regression for the belief-reconstruction omega drop: e_step_iteration returns a rebuilt
    # BeliefState, and if it drops the constant omega frame then the NEXT iteration's transport
    # build reads belief.omega == None and build_transport_from_element(None, ...) crashes. Chaining
    # two iterations at e_phi_lr>0 (so the per-iteration rebuild path fires, not the e_phi_lr==0
    # hoist) exercises exactly that: iter 2 reads iter 1's returned omega.
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

    # (1) two chained iterations (iter 2 reads iter 1's rebuilt belief.omega): no crash + omega kept.
    b1 = e_step_iteration(belief, mu_p, sigma_p, grp, e_phi_lr=0.1,
                          gauge_parameterization="omega_direct")
    assert b1.omega is not None                               # FIX 1: rebuild must carry omega through
    b2 = e_step_iteration(b1, mu_p, sigma_p, grp, e_phi_lr=0.1,
                          gauge_parameterization="omega_direct")
    assert b2.omega is not None
    assert torch.equal(b2.omega, U)                           # constant frame, unchanged by the E-step

    # (2) end-to-end e_step (n_iter=2) also returns a belief that still carries omega.
    out = e_step(belief, mu_p, sigma_p, grp, n_iter=2, e_phi_lr=0.1,
                 gauge_parameterization="omega_direct")
    assert out.omega is not None


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
    from vfe3.gauge_optim import GaugeNaturalGradAdamW
    G = generate_glk(3)
    U = torch.nn.Parameter(torch.eye(3).expand(5, 3, 3).contiguous())
    opt = GaugeNaturalGradAdamW([{"params": [U], "lr": 0.1, "omega": True, "weight_decay": 0.0}],
                                G, [3], gauge_momentum=0.0)
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


def test_cache_supported_admits_omega_direct_flat():
    # cache_supported (belief_cache.py:56-87) keys only on the closed-form-filtering / flat-transport
    # / single-block regime -- it does not branch on gauge_parameterization at all, so a config that
    # is otherwise cache-eligible is admitted under omega_direct exactly as under phi. _cfg's base
    # (family='gaussian_full', n_e_steps=2) is outside the cache-eligible regime for reasons unrelated
    # to gauge_parameterization, so both branches override back to the cache-eligible values here.
    assert cache_supported(_cfg(gauge_parameterization="omega_direct", family="gaussian_diagonal",
                                decode_mode="diagonal", n_e_steps=1)) is True
    assert cache_supported(_cfg(gauge_parameterization="phi", family="gaussian_diagonal",
                                decode_mode="diagonal", n_e_steps=1)) is True


def test_omega_direct_cache_rebuild_matches_full_rollout():
    # Task 10: the KV-cache rebuild branch (belief_cache.py::_appended_belief_step) must, under
    # omega_direct, transport with the STORED element (U_q, inv(U_k)) rather than exp(phi_q)/
    # exp(-phi_k) -- phi_embed is still a small nonzero random table under omega_direct (unused by
    # the live forward/e_step path but NOT automatically zero), so the pre-fix phi-based rebuild is
    # silently wrong once the stored omega frame is perturbed away from identity.
    # policy._rollout_predictive short-circuits to rollout_predictive_cached once cache_supported is
    # True, so it cannot serve as an independent "full recompute" oracle here; instead this builds the
    # full recompute directly via model.rollout_beliefs, exactly as _rollout_predictive's own
    # non-cached branch does.
    torch.manual_seed(0)
    m = VFEModel(_cfg(gauge_parameterization="omega_direct", family="gaussian_diagonal",
                      decode_mode="diagonal", n_e_steps=1))
    assert cache_supported(m.cfg)
    with torch.no_grad():
        g = torch.Generator().manual_seed(3)
        m.prior_bank.omega_embed.copy_(
            torch.eye(4).expand(6, 4, 4) + 0.2 * torch.randn(6, 4, 4, generator=g))

    B, N, Kp, L, V = 2, 2, 2, 1, m.cfg.vocab_size
    torch.manual_seed(1)
    context    = torch.randint(0, V, (B, N))
    candidates = torch.randint(0, V, (B, Kp, L))

    with torch.no_grad():
        base_logits = m.forward(context)[:, -1, :]

        ctx_exp = context.unsqueeze(1).expand(B, Kp, N)
        ext = torch.cat([ctx_exp, candidates], dim=2).reshape(B * Kp, N + L)
        _belief, logits = m.rollout_beliefs(ext, return_logits=True)
        q_full = torch.log_softmax(logits[:, -1, :], dim=-1).reshape(B, Kp, -1)

        q_cache, lp_cache = rollout_predictive_cached(context, candidates, m, base_logits=base_logits)

    base_logp = torch.log_softmax(base_logits, dim=-1)
    lp_full = torch.gather(base_logp, 1, candidates[:, :, 0])
    assert torch.allclose(lp_cache, lp_full, atol=1e-6)
    # Tight tolerance (no rtol slack): a phi-based rebuild that ignores the perturbed omega frame
    # measurably diverges here (~7e-5 max abs diff observed pre-fix), which the loose 1e-4 rtol used
    # by test_belief_cache.py's phi-path golden test would mask against these O(1) log-prob magnitudes.
    assert torch.allclose(q_cache, q_full, atol=2e-5), \
        f"max |dq|={float((q_cache - q_full).abs().max()):.2e}"


def test_appended_belief_step_omega_direct_uses_stored_frame():
    # Direct unit-level regression for the rebuild branch itself (Step 3): with the SAME (small,
    # near-zero) phi field but a DIFFERENT stored omega frame, the omega_direct rebuild must produce
    # a DIFFERENT appended belief. A phi-based rebuild is blind to omega and would return the exact
    # SAME output regardless of U -- a crisp, deterministic pre-fix/post-fix signal that sidesteps the
    # near-cancellation numerical noise a realistic end-to-end rollout can hide behind (see the tight-
    # tolerance rationale above).
    from types import SimpleNamespace

    from vfe3.inference.belief_cache import _appended_belief_step

    K, N, L = 4, 2, 1
    M = N + L
    grp = get_group("glk")(K=K)
    n_gen = grp.generators.shape[0]
    model_ns = SimpleNamespace(
        cfg=_cfg(gauge_parameterization="omega_direct", family="gaussian_diagonal", decode_mode="diagonal"),
        group=grp,
    )
    g     = torch.Generator().manual_seed(9)
    mu    = torch.randn(1, M, K, generator=g)
    sigma = torch.ones(1, M, K)
    phi   = 0.01 * torch.randn(1, M, n_gen, generator=g)          # small, mimics phi_embed's default scale
    U1 = torch.eye(K).expand(1, M, K, K).contiguous()             # identity frames
    U2 = torch.eye(K) + 0.4 * torch.randn(1, M, K, K, generator=g)  # substantially different frames
    log_prior_app = torch.zeros(1, L, M)                          # uniform prior over all M keys

    b1 = BeliefState(mu=mu, sigma=sigma, phi=phi, omega=U1)
    b2 = BeliefState(mu=mu, sigma=sigma, phi=phi, omega=U2)
    out1 = _appended_belief_step(b1, log_prior_app, model_ns, N, 1.0)
    out2 = _appended_belief_step(b2, log_prior_app, model_ns, N, 1.0)
    assert not torch.allclose(out1.mu, out2.mu, atol=1e-6)         # the stored frame must feed the rebuild


def test_ablation_omega_direct_arm_builds():
    cfg = _cfg(gauge_parameterization="omega_direct", gauge_group="glk", use_head_mixer=False)
    assert cfg.gauge_parameterization == "omega_direct"
