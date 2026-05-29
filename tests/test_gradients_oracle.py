import torch

from vfe3.geometry.groups import get_group
from vfe3.geometry.transport import compute_transport_operators
from vfe3.gradients.oracle import belief_gradients_autograd


def _setup(N=3, K=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    grp = get_group("glk")(K)
    phi = 0.15 * torch.randn(1, N, grp.generators.shape[0], generator=g)
    omega = compute_transport_operators(phi, grp)["Omega"][0]          # (N, N, K, K)
    mu = torch.randn(N, K, generator=g)
    sigma = torch.rand(N, K, generator=g) + 0.5
    mu_p = torch.randn(N, K, generator=g)
    sigma_p = torch.rand(N, K, generator=g) + 0.5
    return mu, sigma, mu_p, sigma_p, omega


def _F_filtering(mu_q, sigma_q, mu_p, sigma_p, mu_t, sigma_t, tau):
    # F as a function of the QUERY role only. The transported keys (mu_t, sigma_t)
    # are FROZEN: built once from the unperturbed belief and passed in, so a
    # finite difference of this F holds the key role fixed (the filtering F).
    # (Re-deriving mu_t = mu_q.detach() inside would NOT freeze the keys under FD,
    # since .detach() blocks autograd, not numeric perturbation -- that would
    # measure the full/smoothing gradient instead.)
    from vfe3.free_energy import free_energy, pairwise_energy, self_divergence
    sd = self_divergence(mu_q, sigma_q, mu_p, sigma_p)
    energy = pairwise_energy(mu_q, sigma_q, mu_t, sigma_t)
    alpha = torch.ones(mu_q.shape[0])
    return free_energy(sd, energy, alpha, tau=tau, include_attention_entropy=True)


def test_filtering_oracle_matches_finite_difference_of_F_filt():
    from vfe3.geometry.transport import transport_covariance, transport_mean
    mu, sigma, mu_p, sigma_p, omega = _setup()
    tau = 1.5
    gmu, gsig = belief_gradients_autograd(mu, sigma, mu_p, sigma_p, omega,
                                          tau=tau, gradient_mode="filtering")
    # frozen keys from the unperturbed belief (the filtering split)
    mu_t = transport_mean(omega.unsqueeze(0), mu.unsqueeze(0))[0]
    sigma_t = transport_covariance(omega.unsqueeze(0), sigma.unsqueeze(0))[0]
    eps = 5e-3
    gmu_fd = torch.zeros_like(mu)
    for a in range(mu.shape[0]):
        for b in range(mu.shape[1]):
            d = torch.zeros_like(mu); d[a, b] = eps
            fp = _F_filtering(mu + d, sigma, mu_p, sigma_p, mu_t, sigma_t, tau)
            fm = _F_filtering(mu - d, sigma, mu_p, sigma_p, mu_t, sigma_t, tau)
            gmu_fd[a, b] = (fp - fm) / (2 * eps)
    assert torch.allclose(gmu, gmu_fd, atol=1e-3, rtol=1e-3)


def test_smoothing_differs_from_filtering_by_keyside():
    mu, sigma, mu_p, sigma_p, omega = _setup()
    gf_mu, _ = belief_gradients_autograd(mu, sigma, mu_p, sigma_p, omega,
                                         tau=1.5, gradient_mode="filtering")
    gs_mu, _ = belief_gradients_autograd(mu, sigma, mu_p, sigma_p, omega,
                                         tau=1.5, gradient_mode="smoothing")
    # the key-side (column) term is non-zero -> the two modes differ
    assert not torch.allclose(gf_mu, gs_mu, atol=1e-4)
