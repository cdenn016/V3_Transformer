import pytest
import torch

from vfe3.belief import BeliefState
from vfe3.geometry.generators import reflection_element


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
