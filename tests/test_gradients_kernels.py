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
