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
