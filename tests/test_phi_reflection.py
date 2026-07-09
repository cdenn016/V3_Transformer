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
