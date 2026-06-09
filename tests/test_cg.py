r"""Numerical Clebsch-Gordan intertwiners over the irrep registry."""

import pytest
import torch

from vfe3.geometry.generators import generate_son
from vfe3.geometry.irreps import irrep_generators


def test_irrep_generators_public_builder():
    G_def = generate_son(3, dtype=torch.float64)
    rho = irrep_generators(G_def, algebra="so", label="l2")
    assert rho.shape == (3, 5, 5)
    assert (rho + rho.transpose(-1, -2)).abs().max() < 1e-12
