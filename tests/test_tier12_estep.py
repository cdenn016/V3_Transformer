r"""Tier-1/Tier-2 E-step toggle tests (2026-07-05 improvement toggles).

Covers: e_step_update='mm_exact' (stationarity at frozen beta + monotone filtered-F
descent), lambda_twohop (kernel-side gradient term, byte-identity at 0, hop-weight
sanity), the randomized/truncated/halting E-step loop control, skip_belief_sigma_update
(kernel skip + pass-through sigma), and compile_pair_kernel (eager fallback fidelity).

Device-agnostic: default CPU; set VFE3_TEST_DEVICE=cuda for the GPU (conftest fixture).
"""

import dataclasses
import warnings

import pytest
import torch

from vfe3.belief import BeliefState
from vfe3.families.base import get_family
from vfe3.free_energy import attention_weights, pairwise_energy
from vfe3.geometry.groups import get_group
from vfe3.geometry.transport import (
    compute_transport_operators,
    transport_covariance,
    transport_mean,
)
from vfe3.gradients.kernels import belief_gradients, get_kernel, mm_exact_update
from vfe3.inference.e_step import e_step, e_step_iteration, free_energy_value


def _setup(N=4, K=2, seed=0, device=torch.device("cpu")):
    g = torch.Generator().manual_seed(seed)
    grp = get_group("glk")(K)
    grp = dataclasses.replace(grp, generators=grp.generators.to(device))
    phi = (0.15 * torch.randn(1, N, grp.generators.shape[0], generator=g)).to(device)
    omega = compute_transport_operators(phi, grp)["Omega"][0]
    mu = torch.randn(N, K, generator=g).to(device)
    sigma = (torch.rand(N, K, generator=g) + 0.5).to(device)
    mu_p = torch.randn(N, K, generator=g).to(device)
    sigma_p = (torch.rand(N, K, generator=g) + 0.5).to(device)
    return mu, sigma, mu_p, sigma_p, omega, grp, phi


def _frozen_intermediates(mu, sigma, omega, tau=1.5, log_prior=None):
    """The SAME frozen (start-point) intermediates the kernel/mm build: transported key
    moments, masked beta, constant-alpha coefficient."""
    fam = get_family("gaussian_diagonal")
    mu_t = transport_mean(omega, mu)
    sigma_t = transport_covariance(omega, sigma, diagonal_out=True)
    energy = pairwise_energy(fam(mu, sigma), fam(mu_t, sigma_t), alpha=1.0)
    beta = attention_weights(energy, tau=tau, log_prior=log_prior)
    beta_m = beta * ((energy > 0.0) & (energy < 100.0)).to(beta.dtype)
    coef = torch.ones(mu.shape[0], 1, device=mu.device)
    return mu_t, sigma_t, beta_m, coef


# --- (a) mm_exact stationarity ----------------------------------------------

def test_mm_exact_stationarity_frozen_beta(device):
    mu, sigma, mu_p, sigma_p, omega, grp, _ = _setup(device=device)
    mu_star, sigma_star = mm_exact_update(mu, sigma, mu_p, sigma_p, omega, tau=1.5)
    mu_t, sigma_t, beta_m, coef = _frozen_intermediates(mu, sigma, omega, tau=1.5)
    # the analytic kernel gradients AT (mu*, sigma*) with FROZEN beta / transported moments
    g_mu, g_sigma = get_kernel("gaussian_diagonal")(
        mu_star, sigma_star, mu_p, sigma_p, mu_t, sigma_t, beta_m, coef)
    assert torch.allclose(g_mu, torch.zeros_like(g_mu), atol=1e-4)
    assert torch.allclose(g_sigma, torch.zeros_like(g_sigma), atol=1e-4)


def test_mm_exact_stationarity_folds_twohop(device):
    # total pair weight on (i,k) = lambda_beta*beta_ik + lambda_twohop*W2_ik in the fusion sums,
    # so the kernel gradient WITH the same lambda_twohop must vanish at the fused point too.
    mu, sigma, mu_p, sigma_p, omega, grp, _ = _setup(seed=1, device=device)
    lt = 0.7
    mu_star, sigma_star = mm_exact_update(mu, sigma, mu_p, sigma_p, omega, tau=1.5,
                                          lambda_twohop=lt)
    mu_t, sigma_t, beta_m, coef = _frozen_intermediates(mu, sigma, omega, tau=1.5)
    g_mu, g_sigma = get_kernel("gaussian_diagonal")(
        mu_star, sigma_star, mu_p, sigma_p, mu_t, sigma_t, beta_m, coef, lambda_twohop=lt)
    assert torch.allclose(g_mu, torch.zeros_like(g_mu), atol=1e-4)
    assert torch.allclose(g_sigma, torch.zeros_like(g_sigma), atol=1e-4)


# --- (b) mm_exact monotonicity ----------------------------------------------

def test_mm_exact_monotone_filtered_f_descent(device):
    # F_filt(q1) <= F_hat(q1, beta0) <= F_hat(q0, beta0) = F_filt(q0): the fusion is the exact
    # minimizer of the beta-frozen majorizer, so one mm step cannot increase the frozen-keys F.
    # The untransported self-edge (E_ii = 0 at the start point) is clamp-masked OUT of the
    # majorizer, so exclude it from beta via the attention prior for a clean bound.
    mu, sigma, mu_p, sigma_p, omega, grp, phi = _setup(N=5, seed=2, device=device)
    N = mu.shape[0]
    log_prior = torch.zeros(N, N, device=device).fill_diagonal_(float("-inf"))
    b0 = BeliefState(mu=mu, sigma=sigma, phi=phi[0])
    b1 = e_step_iteration(b0, mu_p, sigma_p, grp, tau=1.5, e_phi_lr=0.0,
                          e_step_update="mm_exact", mm_damping=1.0, log_prior=log_prior)
    F0 = free_energy_value(b0, mu_p, sigma_p, grp, tau=1.5, keys=b0, log_prior=log_prior)
    F1 = free_energy_value(b1, mu_p, sigma_p, grp, tau=1.5, keys=b0, log_prior=log_prior)
    assert torch.isfinite(F1)
    assert F1.item() < F0.item()


def test_mm_exact_rejects_oracle_route(device):
    mu, sigma, mu_p, sigma_p, omega, grp, phi = _setup(device=device)
    b0 = BeliefState(mu=mu, sigma=sigma, phi=phi[0])
    with pytest.raises(ValueError, match="mm_exact"):
        e_step_iteration(b0, mu_p, sigma_p, grp, tau=1.5, e_phi_lr=0.0,
                         e_step_update="mm_exact", gradient_mode="smoothing")


# --- (c) two-hop kernel term -------------------------------------------------

def test_twohop_zero_is_byte_identical(device):
    mu, sigma, mu_p, sigma_p, omega, grp, _ = _setup(device=device)
    base = belief_gradients(mu, sigma, mu_p, sigma_p, omega, tau=1.5)
    with_kw = belief_gradients(mu, sigma, mu_p, sigma_p, omega, tau=1.5, lambda_twohop=0.0)
    assert torch.equal(base[0], with_kw[0])
    assert torch.equal(base[1], with_kw[1])


def test_twohop_changes_gradients(device):
    mu, sigma, mu_p, sigma_p, omega, grp, _ = _setup(device=device)
    base = belief_gradients(mu, sigma, mu_p, sigma_p, omega, tau=1.5)
    hop = belief_gradients(mu, sigma, mu_p, sigma_p, omega, tau=1.5, lambda_twohop=0.5)
    assert not torch.allclose(hop[0], base[0], atol=1e-6)
    assert not torch.allclose(hop[1], base[1], atol=1e-6)


def test_twohop_weights_row_stochastic_and_causal(device):
    # W2 = beta @ beta of a row-stochastic causal beta is row-stochastic and causal.
    mu, sigma, mu_p, sigma_p, omega, grp, _ = _setup(N=5, seed=3, device=device)
    N = mu.shape[0]
    lp = torch.zeros(N, N, device=device)
    lp = lp.masked_fill(torch.triu(torch.ones(N, N, dtype=torch.bool, device=device),
                                   diagonal=1), float("-inf"))
    fam = get_family("gaussian_diagonal")
    mu_t = transport_mean(omega, mu)
    sigma_t = transport_covariance(omega, sigma, diagonal_out=True)
    energy = pairwise_energy(fam(mu, sigma), fam(mu_t, sigma_t), alpha=1.0)
    beta = attention_weights(energy, tau=1.5, log_prior=lp)
    w2 = torch.matmul(beta, beta)
    assert torch.allclose(w2.sum(dim=-1), torch.ones(N, device=device), atol=1e-5)
    assert torch.equal(w2.triu(diagonal=1), torch.zeros_like(w2.triu(diagonal=1)))


# --- (d) loop control ---------------------------------------------------------

def _belief_and_prior(device, N=4, K=2, seed=4):
    mu, sigma, mu_p, sigma_p, omega, grp, phi = _setup(N=N, K=K, seed=seed, device=device)
    return BeliefState(mu=mu, sigma=sigma, phi=phi[0]), mu_p, sigma_p, grp


def test_randomize_off_loop_count_unchanged(device):
    b0, mu_p, sigma_p, grp = _belief_and_prior(device)
    _, traj = e_step(b0, mu_p, sigma_p, grp, n_iter=3, tau=1.5, e_phi_lr=0.0,
                     return_trajectory=True)
    assert len(traj) == 4                              # initial F + one per iteration


def test_randomized_depth_samples_t_when_grad_enabled(device):
    b0, mu_p, sigma_p, grp = _belief_and_prior(device)
    assert torch.is_grad_enabled()                     # the training-forward discriminator
    _, traj = e_step(b0, mu_p, sigma_p, grp, n_iter=5, tau=1.5, e_phi_lr=0.0,
                     randomize_e_steps=True, e_steps_min=2, e_steps_max=2,
                     return_trajectory=True)
    assert len(traj) == 3                              # T == 2, not n_iter == 5


def test_randomized_depth_inert_at_eval(device):
    b0, mu_p, sigma_p, grp = _belief_and_prior(device)
    with torch.no_grad():
        _, traj = e_step(b0, mu_p, sigma_p, grp, n_iter=3, tau=1.5, e_phi_lr=0.0,
                         randomize_e_steps=True, e_steps_min=1, e_steps_max=1,
                         return_trajectory=True)
    assert len(traj) == 4                              # eval keeps the deterministic n_iter


def test_halt_tol_exits_after_one_iteration_at_eval(device):
    b0, mu_p, sigma_p, grp = _belief_and_prior(device)
    with torch.no_grad():
        _, traj = e_step(b0, mu_p, sigma_p, grp, n_iter=5, tau=1.5, e_phi_lr=0.0,
                         e_step_halt_tol=1e6, return_trajectory=True)
    assert len(traj) == 2                              # huge tol -> halt after iteration 1


def test_backprop_last_truncates_but_signal_flows(device):
    b0, mu_p, sigma_p, grp = _belief_and_prior(device)
    mu_p = mu_p.clone().requires_grad_(True)
    out = e_step(b0, mu_p, sigma_p, grp, n_iter=3, tau=1.5, e_phi_lr=0.0,
                 e_steps_backprop_last=1)
    out.mu.sum().backward()
    assert mu_p.grad is not None and torch.isfinite(mu_p.grad).all()
    assert mu_p.grad.abs().sum() > 0                   # the last-k window still reaches the prior


# --- (e) skip_belief_sigma_update ---------------------------------------------

def test_skip_sigma_kernel_returns_none_and_mu_unchanged(device):
    mu, sigma, mu_p, sigma_p, omega, grp, _ = _setup(device=device)
    base = belief_gradients(mu, sigma, mu_p, sigma_p, omega, tau=1.5)
    g_mu, g_sigma = belief_gradients(mu, sigma, mu_p, sigma_p, omega, tau=1.5,
                                     need_sigma_grad=False)
    assert g_sigma is None
    assert torch.equal(g_mu, base[0])                  # mu gradient untouched by the skip


def test_skip_sigma_passes_through_gradient_path(device):
    b0, mu_p, sigma_p, grp = _belief_and_prior(device, seed=5)
    out = e_step(b0, mu_p, sigma_p, grp, n_iter=2, tau=1.5, e_phi_lr=0.0,
                 skip_belief_sigma_update=True)
    assert torch.equal(out.sigma, b0.sigma)            # sigma passes through UNCHANGED
    assert not torch.allclose(out.mu, b0.mu, atol=1e-6)


def test_skip_sigma_passes_through_mm_path(device):
    b0, mu_p, sigma_p, grp = _belief_and_prior(device, seed=6)
    out = e_step_iteration(b0, mu_p, sigma_p, grp, tau=1.5, e_phi_lr=0.0,
                           e_step_update="mm_exact", mm_damping=1.0,
                           skip_belief_sigma_update=True)
    assert torch.equal(out.sigma, b0.sigma)
    mu_star, _ = mm_exact_update(b0.mu, b0.sigma, mu_p, sigma_p,
                                 compute_transport_operators(b0.phi.unsqueeze(0), grp)["Omega"][0],
                                 tau=1.5)
    assert torch.allclose(out.mu, mu_star, atol=1e-6)  # eta=1 lands exactly on mu*


# --- compile_pair_kernel --------------------------------------------------------

def test_compile_pair_kernel_matches_eager_or_falls_back(device):
    mu, sigma, mu_p, sigma_p, omega, grp, _ = _setup(device=device)
    base = belief_gradients(mu, sigma, mu_p, sigma_p, omega, tau=1.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                # eager-fallback warning is expected on Windows
        comp = belief_gradients(mu, sigma, mu_p, sigma_p, omega, tau=1.5,
                                compile_pair_kernel=True)
    assert torch.allclose(comp[0], base[0], atol=1e-5)
    assert torch.allclose(comp[1], base[1], atol=1e-5)
