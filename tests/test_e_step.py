import torch

from vfe3.belief import BeliefState
from vfe3.geometry.groups import get_group
from vfe3.inference.e_step import free_energy_value


def _belief(N=3, K=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    grp = get_group("glk")(K)
    n_gen = grp.generators.shape[0]
    b = BeliefState(
        mu=torch.randn(N, K, generator=g),
        sigma=torch.rand(N, K, generator=g) + 0.5,
        phi=0.1 * torch.randn(N, n_gen, generator=g),
    )
    mu_p = torch.randn(N, K, generator=g)
    sigma_p = torch.rand(N, K, generator=g) + 0.5
    return b, mu_p, sigma_p, grp


def test_belief_state_fields():
    b, *_ = _belief()
    assert b.mu.shape == (3, 2) and b.sigma.shape == (3, 2)


def test_free_energy_value_is_finite_scalar():
    b, mu_p, sigma_p, grp = _belief()
    F = free_energy_value(b, mu_p, sigma_p, grp, tau=1.5)
    assert F.shape == () and torch.isfinite(F)


def test_free_energy_filtering_equals_global_at_a_point():
    # F_filt and global F are the SAME NUMBER at a fixed belief (detach changes
    # gradients, not the value); they differ only as functions under a step.
    b, mu_p, sigma_p, grp = _belief()
    Fg = free_energy_value(b, mu_p, sigma_p, grp, tau=1.5, keys=None)
    Ff = free_energy_value(b, mu_p, sigma_p, grp, tau=1.5, keys=b)   # keys frozen at b
    assert torch.allclose(Fg, Ff, atol=1e-6)
