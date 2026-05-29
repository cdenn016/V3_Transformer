import pytest
import torch

from vfe3.divergence import safe_kl_clamp


def test_safe_kl_clamp_bounds_and_nan():
    x = torch.tensor([-1.0, 0.5, 1e9, float("nan"), float("inf"), float("-inf")])
    out = safe_kl_clamp(x, kl_max=100.0)
    assert torch.equal(
        out, torch.tensor([0.0, 0.5, 100.0, 100.0, 100.0, 0.0])
    )


def test_registry_register_and_get():
    from vfe3.divergence import register_divergence, get_divergence

    @register_divergence("dummy_family")
    def _dummy(mu_q, sigma_q, mu_t, sigma_t, *, alpha, kl_max, eps):
        return mu_q.sum(dim=-1) * 0.0

    fn = get_divergence("dummy_family")
    assert fn is _dummy


def test_registry_unknown_raises():
    from vfe3.divergence import get_divergence
    with pytest.raises(KeyError):
        get_divergence("no_such_family")


def test_kl_equals_renyi_at_alpha_one():
    from vfe3.divergence import kl, renyi
    g = torch.Generator().manual_seed(7)
    mu_q = torch.randn(3, 5, generator=g)
    mu_t = torch.randn(3, 5, generator=g)
    sigma_q = torch.rand(3, 5, generator=g) + 0.1
    sigma_t = torch.rand(3, 5, generator=g) + 0.1
    a = kl(mu_q, sigma_q, mu_t, sigma_t, family="gaussian_diagonal")
    b = renyi(mu_q, sigma_q, mu_t, sigma_t, alpha=1.0, family="gaussian_diagonal")
    assert torch.allclose(a, b)


def test_renyi_dispatches_on_family():
    from vfe3.divergence import renyi
    g = torch.Generator().manual_seed(8)
    mu = torch.randn(2, 4, generator=g)
    A = torch.randn(2, 4, 4, generator=g)
    sigma_full = A @ A.transpose(-1, -2) + torch.eye(4)
    out = renyi(mu, sigma_full, mu, sigma_full, alpha=1.0, family="gaussian_full")
    assert out.shape == (2,)
    assert torch.allclose(out, torch.zeros(2), atol=1e-4)


@pytest.mark.parametrize("family", ["gaussian_diagonal", "gaussian_full"])
def test_self_divergence_is_zero(family):
    from vfe3.divergence import kl
    g = torch.Generator().manual_seed(11)
    mu = torch.randn(4, 6, generator=g)
    if family == "gaussian_diagonal":
        sigma = torch.rand(4, 6, generator=g) + 0.1
    else:
        A = torch.randn(4, 6, 6, generator=g)
        sigma = A @ A.transpose(-1, -2) + torch.eye(6)
    out = kl(mu, sigma, mu, sigma, family=family)
    assert torch.allclose(out, torch.zeros(4), atol=1e-4)


@pytest.mark.parametrize("family", ["gaussian_diagonal", "gaussian_full"])
def test_divergence_nonnegative(family):
    from vfe3.divergence import kl
    g = torch.Generator().manual_seed(12)
    mu_q = torch.randn(8, 6, generator=g)
    mu_t = torch.randn(8, 6, generator=g)
    if family == "gaussian_diagonal":
        sigma_q = torch.rand(8, 6, generator=g) + 0.1
        sigma_t = torch.rand(8, 6, generator=g) + 0.1
    else:
        Aq = torch.randn(8, 6, 6, generator=g)
        At = torch.randn(8, 6, 6, generator=g)
        sigma_q = Aq @ Aq.transpose(-1, -2) + torch.eye(6)
        sigma_t = At @ At.transpose(-1, -2) + torch.eye(6)
    out = kl(mu_q, sigma_q, mu_t, sigma_t, family=family)
    assert (out >= 0.0).all()


@pytest.mark.parametrize("family", ["gaussian_diagonal", "gaussian_full"])
def test_alpha_gt_one_warns(family):
    # Equal q/t covariance makes the blend (1-a)S + aS == S, guaranteed SPD
    # for any alpha, so the warning fires without a Cholesky failure.
    from vfe3.divergence import renyi
    g = torch.Generator().manual_seed(13)
    mu_q = torch.randn(3, 4, generator=g)
    mu_t = torch.randn(3, 4, generator=g)
    if family == "gaussian_diagonal":
        sigma = torch.rand(3, 4, generator=g) + 0.5
    else:
        A = torch.randn(3, 4, 4, generator=g)
        sigma = A @ A.transpose(-1, -2) + torch.eye(4)
    with pytest.warns(RuntimeWarning, match=r"alpha=1.5 > 1"):
        renyi(mu_q, sigma, mu_t, sigma, alpha=1.5, family=family)


@pytest.mark.parametrize("alpha", [0.5, 1.0])
def test_alpha_le_one_does_not_warn(alpha):
    import warnings

    from vfe3.divergence import renyi
    g = torch.Generator().manual_seed(14)
    mu_q = torch.randn(3, 5, generator=g)
    mu_t = torch.randn(3, 5, generator=g)
    sigma_q = torch.rand(3, 5, generator=g) + 0.1
    sigma_t = torch.rand(3, 5, generator=g) + 0.1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        renyi(mu_q, sigma_q, mu_t, sigma_t, alpha=alpha, family="gaussian_diagonal")
    assert not any(issubclass(w.category, RuntimeWarning) for w in caught)
