import torch

from vfe3.geometry.groups import get_group
from vfe3.geometry.transport import compute_transport_operators
from vfe3.gradients.kernels import belief_gradients
from vfe3.gradients.oracle import belief_gradients_autograd


def _setup(N=3, K=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    grp = get_group("glk")(K)
    phi = 0.15 * torch.randn(1, N, grp.generators.shape[0], generator=g)
    omega = compute_transport_operators(phi, grp)["Omega"][0]
    mu = torch.randn(N, K, generator=g); sigma = torch.rand(N, K, generator=g) + 0.5
    mu_p = torch.randn(N, K, generator=g); sigma_p = torch.rand(N, K, generator=g) + 0.5
    return mu, sigma, mu_p, sigma_p, omega


def test_kernel_matches_filtering_oracle_constant_alpha():
    args = _setup()
    km, ks = belief_gradients(*args, tau=1.5, gradient_mode="filtering")
    om, os_ = belief_gradients_autograd(*args, tau=1.5, gradient_mode="filtering")
    assert torch.allclose(km, om, atol=1e-5)
    assert torch.allclose(ks, os_, atol=1e-5)


def test_kernel_differs_from_smoothing_oracle():
    args = _setup()
    km, _ = belief_gradients(*args, tau=1.5, gradient_mode="filtering")
    sm, _ = belief_gradients_autograd(*args, tau=1.5, gradient_mode="smoothing")
    assert not torch.allclose(km, sm, atol=1e-4)              # key-side term is real


def test_dispatch_falls_back_to_oracle():
    args = _setup()
    # smoothing -> oracle
    a = belief_gradients(*args, tau=1.5, gradient_mode="smoothing")
    b = belief_gradients_autograd(*args, tau=1.5, gradient_mode="smoothing")
    assert torch.allclose(a[0], b[0], atol=1e-6) and torch.allclose(a[1], b[1], atol=1e-6)
    # non-KL (Renyi alpha_div != 1) -> oracle
    c = belief_gradients(*args, tau=1.5, gradient_mode="filtering", alpha_div=0.5)
    d = belief_gradients_autograd(*args, tau=1.5, gradient_mode="filtering", alpha_div=0.5)
    assert torch.allclose(c[0], d[0], atol=1e-5) and torch.allclose(c[1], d[1], atol=1e-5)


def test_kernel_matches_filtering_oracle_state_dependent_alpha_with_R():
    args = _setup()
    km, ks = belief_gradients(*args, tau=1.5, gradient_mode="filtering",
                              alpha_mode="state_dependent", b0=0.5, c0=2.0)
    om, os_ = belief_gradients_autograd(*args, tau=1.5, gradient_mode="filtering",
                                        alpha_mode="state_dependent", b0=0.5, c0=2.0)
    assert torch.allclose(km, om, atol=1e-5)                 # alpha* cancellation (R on both sides)
    assert torch.allclose(ks, os_, atol=1e-5)


def test_constant_value_honored_on_kernel_and_oracle_fallback():
    # The constant-alpha weight `value` scales the self-coupling term. It must be
    # honored identically on the kernel path (filtering+KL) and on every oracle
    # fallback (smoothing, non-KL) -- two callers with the same (mu,...,value) must
    # not get different self-coupling gradients depending only on dispatch branch.
    args = _setup()
    # kernel path: value flows
    km3, _ = belief_gradients(*args, tau=1.5, gradient_mode="filtering", value=3.0)
    km1, _ = belief_gradients(*args, tau=1.5, gradient_mode="filtering", value=1.0)
    assert not torch.allclose(km3, km1, atol=1e-4)
    # kernel path pinned to the oracle AT value != 1
    om3, _ = belief_gradients_autograd(*args, tau=1.5, gradient_mode="filtering", value=3.0)
    assert torch.allclose(km3, om3, atol=1e-5)
    # smoothing fallback: value must flow too (not silently reset to 1.0)
    sm3 = belief_gradients(*args, tau=1.5, gradient_mode="smoothing", value=3.0)
    sm1 = belief_gradients(*args, tau=1.5, gradient_mode="smoothing", value=1.0)
    assert not torch.allclose(sm3[0], sm1[0], atol=1e-4)
    so3 = belief_gradients_autograd(*args, tau=1.5, gradient_mode="smoothing", value=3.0)
    assert torch.allclose(sm3[0], so3[0], atol=1e-6) and torch.allclose(sm3[1], so3[1], atol=1e-6)


def test_kernel_honors_clamp_saturation_self_term():
    # Once the raw self-divergence D(q||p) exceeds kl_max the oracle differentiates
    # through safe_kl_clamp (clamp(max=kl_max)), whose gradient is 0; the hand kernel
    # must zero its self-term there to stay EXACTLY equal to the filtering oracle.
    N, K = 1, 2
    omega = torch.eye(K).expand(N, N, K, K).contiguous()
    # mean-driven saturation: D = 0.5 * (2 * 20^2) = 400 > kl_max=100
    mu = torch.zeros(N, K); sigma = torch.ones(N, K)
    mu_p = torch.full((N, K), 20.0); sigma_p = torch.ones(N, K)
    km, ks = belief_gradients(mu, sigma, mu_p, sigma_p, omega, gradient_mode="filtering")
    om, os_ = belief_gradients_autograd(mu, sigma, mu_p, sigma_p, omega, gradient_mode="filtering")
    assert torch.allclose(km, om, atol=1e-5) and torch.allclose(ks, os_, atol=1e-5)
    assert torch.allclose(km, torch.zeros_like(km), atol=1e-5)   # restoring force gated off (== oracle)
    # variance-driven saturation: D dominated by 1/sigma_p
    mu2 = torch.zeros(N, K); sigma2 = torch.ones(N, K)
    mu_p2 = torch.zeros(N, K); sigma_p2 = torch.full((N, K), 1e-8)
    km2, ks2 = belief_gradients(mu2, sigma2, mu_p2, sigma_p2, omega, gradient_mode="filtering")
    om2, os2 = belief_gradients_autograd(mu2, sigma2, mu_p2, sigma_p2, omega, gradient_mode="filtering")
    assert torch.allclose(km2, om2, atol=1e-5) and torch.allclose(ks2, os2, atol=1e-5)


def test_self_gradient_vanishes_when_q_equals_p_and_identity_transport():
    K = 2
    N = 3
    omega = torch.eye(K).expand(N, N, K, K).contiguous()      # identity transport
    # Equal means across tokens: with q == p the self term is zero, AND with
    # identity transport mu_t_ij = mu_j = mu_i, so each coupling-mean residual
    # (mu_i - mu_t_ij) vanishes -- nothing to disagree about => zero gradient.
    # (Distinct means would leave the belief-coupling row sum
    #  Sum_j beta_ij (mu_i - mu_j)/sigma_t_ij non-zero even at q == p.)
    mu = torch.randn(1, K).expand(N, K).contiguous(); sigma = torch.rand(N, K) + 0.5
    gmu, _ = belief_gradients(mu, sigma, mu.clone(), sigma.clone(), omega,
                              tau=1.5, gradient_mode="filtering")
    assert torch.allclose(gmu, torch.zeros(N, K), atol=1e-5)
