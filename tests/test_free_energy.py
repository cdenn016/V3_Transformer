import math

import torch

from vfe3.free_energy import (
    attention_weights,
    effective_temperature,
    log_partition,
    reduced_free_energy,
)

# A concrete non-uniform setup reused across tests.
_E   = torch.tensor([1.0, 2.0, 0.5])               # distinct per-key energies
_PI  = torch.tensor([0.5, 0.3, 0.2])               # normalized non-uniform prior
_B   = torch.log(_PI)                              # log-prior bias
_TAU = 2.0


def test_temperature_is_kappa_sqrt_k():
    assert math.isclose(effective_temperature(1.5, 16), 1.5 * 4.0, rel_tol=1e-6)


def test_beta_is_softmax_logprior_minus_energy_over_tau():
    beta = attention_weights(_E, log_prior=_B, tau=_TAU)
    logits = _B - _E / _TAU
    expect = torch.softmax(logits, dim=-1)
    assert torch.allclose(beta, expect, atol=1e-6)
    assert torch.allclose(beta.sum(-1), torch.tensor(1.0), atol=1e-6)


def test_envelope_identity_canonical_block_equals_neg_tau_logZ():
    # Sum_j beta* E + tau Sum_j beta* log(beta*/pi) == -tau log Z, with non-uniform pi.
    beta = attention_weights(_E, log_prior=_B, tau=_TAU)
    pi = torch.softmax(_B, dim=-1)
    canon_block = (beta * _E).sum(-1) + _TAU * (beta * (torch.log(beta) - torch.log(pi))).sum(-1)
    fred = reduced_free_energy(_E, log_prior=_B, tau=_TAU)        # -tau log Z
    assert torch.allclose(canon_block, fred, atol=1e-5)
    # hand-computed literal backstop (catches a tau*log N offset):
    assert torch.allclose(fred, torch.tensor(1.1264), atol=1e-3)


def test_stationarity_residual_constant_across_keys():
    # At beta*, E_j + tau log(beta*_j/pi_j) is the SAME for every key j (= -tau log Z).
    beta = attention_weights(_E, log_prior=_B, tau=_TAU)
    pi = torch.softmax(_B, dim=-1)
    residual = _E + _TAU * (torch.log(beta) - torch.log(pi))
    assert (residual.max() - residual.min()).abs() < 1e-5
    assert torch.allclose(residual.mean(), reduced_free_energy(_E, log_prior=_B, tau=_TAU), atol=1e-5)


from vfe3.free_energy import free_energy


def test_canonical_minus_surrogate_is_tau_times_entropy():
    # Canonical F - surrogate F = tau * Sum_i Sum_j beta* log(beta*/pi)  (the entropy block).
    N = 3
    self_div = torch.zeros(N)                            # alpha term zero (isolate beta block)
    energy = torch.tensor([[1.0, 2.0, 0.5],
                           [0.7, 0.3, 1.1],
                           [1.2, 0.9, 0.4]])
    B = torch.log(torch.tensor([0.5, 0.3, 0.2]))
    log_prior = B.expand(N, N)
    alpha = torch.zeros(N)
    fe_canon = free_energy(self_div, energy, alpha, log_prior=log_prior, tau=2.0,
                           include_attention_entropy=True)
    fe_surr  = free_energy(self_div, energy, alpha, log_prior=log_prior, tau=2.0,
                           include_attention_entropy=False)
    beta = attention_weights(energy, log_prior=log_prior, tau=2.0)
    pi = torch.softmax(log_prior, dim=-1)
    entropy_block = 2.0 * (beta * (torch.log(beta) - torch.log(pi))).sum()
    assert torch.allclose(fe_canon - fe_surr, entropy_block, atol=1e-5)


def test_known_value_F_self_coupling_only():
    # q == p -> self_div == 0; energy all-equal + uniform prior -> beta uniform.
    # With alpha=2, self_div=[0.5,1.0], no entropy (surrogate), energy uniform=c:
    # F = sum_i alpha_i*self_div_i + sum_ij beta_ij*c. beta uniform=1/N so sum_j beta*c=c.
    self_div = torch.tensor([0.5, 1.0])
    energy = torch.full((2, 2), 0.3)
    alpha = torch.full((2,), 2.0)
    fe = free_energy(self_div, energy, alpha, log_prior=None, tau=1.0,
                     include_attention_entropy=False)
    expect = (2.0 * 0.5 + 2.0 * 1.0) + (0.3 + 0.3)
    assert torch.allclose(fe, torch.tensor(expect), atol=1e-5)


def test_autograd_F_matches_finite_difference():
    torch.manual_seed(0)
    N, K = 3, 4
    mu_q = torch.randn(N, K, requires_grad=True)
    base = {"sigma_q": torch.rand(N, K) + 0.5, "mu_p": torch.randn(N, K),
            "sigma_p": torch.rand(N, K) + 0.5}
    from vfe3.free_energy import self_divergence

    def scalar(mu):
        sd = self_divergence(mu, base["sigma_q"], base["mu_p"], base["sigma_p"])
        energy = torch.cdist(mu, mu) ** 2 + 0.1           # a smooth differentiable (N,N) energy
        alpha = torch.ones(N)
        return free_energy(sd, energy, alpha, log_prior=None, tau=1.5,
                           include_attention_entropy=True)

    F = scalar(mu_q); F.backward()
    g_auto = mu_q.grad.clone()
    eps = 1e-3
    g_fd = torch.zeros_like(mu_q)
    with torch.no_grad():
        for a in range(N):
            for b in range(K):
                d = torch.zeros(N, K); d[a, b] = eps
                g_fd[a, b] = (scalar(mu_q + d) - scalar(mu_q - d)) / (2 * eps)
    assert torch.allclose(g_auto, g_fd, atol=1e-3, rtol=1e-3)
