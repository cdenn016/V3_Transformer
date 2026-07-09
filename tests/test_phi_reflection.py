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
