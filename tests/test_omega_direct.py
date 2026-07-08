import pytest
import torch

from vfe3.belief import BeliefState
from vfe3.config import VFE3Config
from vfe3.geometry.generators import generate_glk, reflection_element
from vfe3.geometry.groups import get_group
from vfe3.geometry.lie_ops import retract_omega
from vfe3.geometry.transport import (build_transport_from_element, compute_transport_operators,
                                      transport_mean, FactoredTransport)
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
