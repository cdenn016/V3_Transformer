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
    # Seeded (suite convention) and g conditioned to kappa < 10: Sigma_g = g g^T has
    # kappa(Sigma_g) = kappa(g)^2, and the full-cov solve amplifies fp32 roundoff by that
    # factor, so an unconditioned g=randn+2I gives an O(1) residual ~1.5% of the time --
    # not a math failure but fp32 conditioning. Restricting to well-conditioned g keeps
    # the gauge-invariance claim a clean atol-1e-4 check rather than a flaky one.
    K = 3
    rng = torch.Generator().manual_seed(0)
    norm = MahalanobisNorm(K)
    while True:
        g = torch.randn(K, K, generator=rng) + 2 * torch.eye(K)  # invertible
        if torch.linalg.cond(g).item() < 10.0:                   # well-conditioned draw
            break
    mu = torch.randn(2, K, generator=rng)
    sigma_full = torch.eye(K).expand(2, K, K).contiguous()
    out = norm(mu, sigma_full)
    mu_g = mu @ g.T
    sig_g = g @ sigma_full @ g.T
    out_g = norm(mu_g, sig_g)
    assert torch.allclose(out_g, out @ g.T, atol=1e-4)
