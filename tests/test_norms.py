import torch
from vfe3.geometry.norms import MahalanobisNorm


def test_mahalanobis_formula_diagonal():
    K = 4
    norm = MahalanobisNorm(K)
    mu = torch.randn(3, K); sigma = torch.rand(3, K) + 0.5
    out = norm(mu, sigma)
    s2 = (mu ** 2 / sigma).sum(-1, keepdim=True)
    assert torch.allclose(out, mu * torch.sqrt(K / s2), atol=1e-5)


def test_mahalanobis_is_gauge_invariant_scale():
    # The Mahalanobis scalar mu^T Sigma^-1 mu is invariant under mu->g mu, Sigma->g Sigma g^T,
    # so the norm SCALE sqrt(K/s2) is gauge-invariant; out transforms as a vector (out -> g out).
    K = 3
    norm = MahalanobisNorm(K)
    g = torch.randn(K, K); g = g + 2 * torch.eye(K)              # invertible
    mu = torch.randn(2, K); sigma_full = torch.eye(K).expand(2, K, K).contiguous()
    out = norm(mu, sigma_full)
    mu_g = mu @ g.T
    sig_g = g @ sigma_full @ g.T
    out_g = norm(mu_g, sig_g)
    assert torch.allclose(out_g, out @ g.T, atol=1e-4)
