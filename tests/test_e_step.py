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


# --- Task 2: one inner iteration -------------------------------------------
from vfe3.inference.e_step import e_step_iteration


def test_iteration_keeps_sigma_positive_and_shapes():
    b, mu_p, sigma_p, grp = _belief()
    out = e_step_iteration(b, mu_p, sigma_p, grp, tau=1.5,
                           e_mu_lr=0.05, e_sigma_lr=0.05, e_phi_lr=0.05)
    assert (out.sigma > 0).all()
    assert out.mu.shape == b.mu.shape and out.phi.shape == b.phi.shape


def test_decoupled_learning_rates_freeze_components():
    b, mu_p, sigma_p, grp = _belief()
    o1 = e_step_iteration(b, mu_p, sigma_p, grp, tau=1.5, e_mu_lr=0.05, e_sigma_lr=0.05, e_phi_lr=0.0)
    assert torch.allclose(o1.phi, b.phi, atol=1e-7)
    o2 = e_step_iteration(b, mu_p, sigma_p, grp, tau=1.5, e_mu_lr=0.0, e_sigma_lr=0.05, e_phi_lr=0.0)
    assert torch.allclose(o2.mu, b.mu, atol=1e-7)


# --- Task 3: descent directions (the right objective per mode) -------------
def test_filtering_step_descends_F_filt():
    # filtering (query-side) gradient descends F with KEYS FROZEN at the pre-step belief.
    b, mu_p, sigma_p, grp = _belief()
    F_before = free_energy_value(b, mu_p, sigma_p, grp, tau=1.5, keys=b)
    out = e_step_iteration(b, mu_p, sigma_p, grp, tau=1.5, e_mu_lr=1e-3, e_sigma_lr=1e-3,
                           e_phi_lr=0.0, gradient_mode="filtering", e_sigma_q_trust=0.0)
    F_after = free_energy_value(out, mu_p, sigma_p, grp, tau=1.5, keys=b)   # SAME frozen keys b
    assert F_after < F_before


def test_smoothing_step_descends_global_F():
    b, mu_p, sigma_p, grp = _belief()
    F_before = free_energy_value(b, mu_p, sigma_p, grp, tau=1.5)            # global (keys=belief)
    out = e_step_iteration(b, mu_p, sigma_p, grp, tau=1.5, e_mu_lr=1e-3, e_sigma_lr=1e-3,
                           e_phi_lr=0.0, gradient_mode="smoothing", e_sigma_q_trust=0.0)
    F_after = free_energy_value(out, mu_p, sigma_p, grp, tau=1.5)
    assert F_after < F_before


def test_phi_step_descends_global_F_with_beliefs_frozen():
    b, mu_p, sigma_p, grp = _belief()
    F_before = free_energy_value(b, mu_p, sigma_p, grp, tau=1.5)
    out = e_step_iteration(b, mu_p, sigma_p, grp, tau=1.5, e_mu_lr=0.0, e_sigma_lr=0.0, e_phi_lr=1e-3)
    F_after = free_energy_value(out, mu_p, sigma_p, grp, tau=1.5)
    assert F_after < F_before


# --- Task 4: e_step loop + trajectory + fixed-seed regression --------------
from vfe3.inference.e_step import e_step

EXPECTED_CHECKSUM = 6.6499   # frozen from the first trusted green run (seed=7, n_iter=3)


def test_e_step_runs_n_iter_and_returns_trajectory():
    b, mu_p, sigma_p, grp = _belief()
    out, traj = e_step(b, mu_p, sigma_p, grp, tau=1.5, n_iter=5,
                       e_mu_lr=1e-2, e_sigma_lr=1e-2, e_phi_lr=1e-2, return_trajectory=True)
    assert len(traj) == 6
    assert (out.sigma > 0).all()


def test_smoothing_loop_decreases_F_overall():
    b, mu_p, sigma_p, grp = _belief()
    out, traj = e_step(b, mu_p, sigma_p, grp, tau=1.5, n_iter=10,
                       e_mu_lr=2e-3, e_sigma_lr=2e-3, e_phi_lr=2e-3,
                       gradient_mode="smoothing", e_sigma_q_trust=0.0, return_trajectory=True)
    assert traj[-1] < traj[0]


def test_fixed_seed_regression():
    b, mu_p, sigma_p, grp = _belief(seed=7)
    out = e_step(b, mu_p, sigma_p, grp, tau=1.5, n_iter=3,
                 e_mu_lr=1e-2, e_sigma_lr=1e-2, e_phi_lr=1e-2)
    assert torch.isfinite(out.mu).all() and torch.isfinite(out.sigma).all() and torch.isfinite(out.phi).all()
    checksum = float(out.mu.sum() + out.sigma.sum() + out.phi.sum())
    assert abs(checksum - EXPECTED_CHECKSUM) < 1e-3
