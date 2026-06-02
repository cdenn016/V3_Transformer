import pytest
import torch

from vfe3.geometry.groups import get_group
from vfe3.model.positional_phi import (
    get_pos_phi, positional_phi_coords, apply_positional_phi,
)


def _glk_group(k=4):
    return get_group("glk")(k)


def test_none_returns_none_coords():
    coords = positional_phi_coords("none", 5, 3, device=torch.device("cpu"), dtype=torch.float32)
    assert coords is None


def test_frozen_coords_are_position_times_scale_on_one_axis():
    coords = positional_phi_coords("frozen", 4, 3, scale=0.1, frozen_axis=0,
                                   device=torch.device("cpu"), dtype=torch.float32)
    assert coords.shape == (4, 3)
    assert torch.allclose(coords[:, 0], torch.tensor([0.0, 0.1, 0.2, 0.3]))
    assert torch.allclose(coords[:, 1:], torch.zeros(4, 2))


def test_learned_coords_slice_the_table():
    table = torch.randn(8, 3)
    coords = positional_phi_coords("learned", 4, 3, pos_phi_free=table,
                                   device=torch.device("cpu"), dtype=torch.float32)
    assert torch.equal(coords, table[:4])


def test_apply_none_is_identity():
    g = _glk_group()
    phi = torch.randn(2, 5, g.generators.shape[0])
    out = apply_positional_phi(phi, g, mode="none")
    assert torch.equal(out, phi)


def test_get_pos_phi_unknown_raises_keyerror():
    with pytest.raises(KeyError):
        get_pos_phi("not_a_mode")


from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel


def _cfg(**kw):
    base = dict(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                n_e_steps=1, e_mu_lr=0.1, e_phi_lr=0.0, m_phi_lr=0.0,
                warmup_steps=1, max_steps=4)
    base.update(kw)
    return VFE3Config(**base)


def test_pos_phi_none_logits_byte_identical_to_no_field():
    torch.manual_seed(0)
    x = torch.randint(0, 6, (2, 8))
    m = VFEModel(_cfg(pos_phi="none"))
    logits_a = m(x)
    logits_b = m(x)
    assert torch.equal(logits_a, logits_b)              # determinism guard
    assert not hasattr(m, "pos_phi_free")               # no parameter created on the pure path


def test_pos_phi_learned_creates_parameter_and_changes_logits():
    torch.manual_seed(0)
    x = torch.randint(0, 6, (2, 8))
    base = VFEModel(_cfg(pos_phi="none"))
    learned = VFEModel(_cfg(pos_phi="learned", pos_phi_scale=0.3))
    learned.load_state_dict(base.state_dict(), strict=False)   # share priors; pos_phi_free is extra
    assert hasattr(learned, "pos_phi_free")
    assert learned.pos_phi_free.shape == (8, base.group.generators.shape[0])
    with torch.no_grad():
        learned.pos_phi_free.add_(0.2)
    assert not torch.allclose(base(x), learned(x), atol=1e-5)


def test_pos_phi_learned_receives_gradient():
    torch.manual_seed(0)
    x = torch.randint(0, 6, (2, 8))
    y = torch.randint(0, 6, (2, 8))
    m = VFEModel(_cfg(pos_phi="learned", pos_phi_scale=0.3))
    with torch.no_grad():
        m.pos_phi_free.add_(0.1)
    _, loss, _ = m(x, y)
    loss.backward()
    assert m.pos_phi_free.grad is not None
    assert m.pos_phi_free.grad.abs().sum() > 0
