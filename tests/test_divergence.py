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
    from vfe3.divergence import register_divergence, get_divergence, _DIVERGENCES

    @register_divergence("dummy_family", cov_kind="diagonal")
    def _dummy(mu_q, sigma_q, mu_t, sigma_t, *, alpha, kl_max, eps):
        return mu_q.sum(dim=-1) * 0.0

    try:
        fn = get_divergence("dummy_family")
        assert fn is _dummy
    finally:
        # Do not leak the test kernel into the global registry.
        from vfe3.divergence import _COV_KIND
        _DIVERGENCES.pop("dummy_family", None)
        _COV_KIND.pop("dummy_family", None)


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


@pytest.mark.parametrize("family", ["gaussian_diagonal", "gaussian_full"])
@pytest.mark.parametrize("alpha", [0.5, 1.0])
def test_alpha_le_one_does_not_warn(alpha, family):
    import warnings

    from vfe3.divergence import renyi
    g = torch.Generator().manual_seed(14)
    mu_q = torch.randn(3, 4, generator=g)
    mu_t = torch.randn(3, 4, generator=g)
    if family == "gaussian_diagonal":
        sigma_q = torch.rand(3, 4, generator=g) + 0.1
        sigma_t = torch.rand(3, 4, generator=g) + 0.1
    else:
        Aq = torch.randn(3, 4, 4, generator=g)
        At = torch.randn(3, 4, 4, generator=g)
        sigma_q = Aq @ Aq.transpose(-1, -2) + torch.eye(4)
        sigma_t = At @ At.transpose(-1, -2) + torch.eye(4)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        renyi(mu_q, sigma_q, mu_t, sigma_t, alpha=alpha, family=family)
    assert not any(issubclass(w.category, RuntimeWarning) for w in caught)


@pytest.mark.parametrize("bad_alpha", [0.0, -1.0])
def test_alpha_nonpositive_raises(bad_alpha):
    from vfe3.divergence import renyi
    g = torch.Generator().manual_seed(15)
    mu = torch.randn(2, 4, generator=g)
    sigma = torch.rand(2, 4, generator=g) + 0.1
    with pytest.raises(ValueError):
        renyi(mu, sigma, mu, sigma, alpha=bad_alpha, family="gaussian_diagonal")


def test_diagonal_kl_matches_torch_distributions():
    # Independent reference: PyTorch's own Normal KL (summed over dims).
    from torch.distributions import Normal, kl_divergence
    from vfe3.divergence import kl
    g = torch.Generator().manual_seed(20)
    mu_q = torch.randn(4, 5, generator=g)
    mu_t = torch.randn(4, 5, generator=g)
    var_q = torch.rand(4, 5, generator=g) + 0.2
    var_t = torch.rand(4, 5, generator=g) + 0.2
    ref = kl_divergence(Normal(mu_q, var_q.sqrt()), Normal(mu_t, var_t.sqrt())).sum(-1)
    got = kl(mu_q, var_q, mu_t, var_t, family="gaussian_diagonal")
    assert torch.allclose(got, ref, atol=1e-5, rtol=1e-5)


def test_full_kl_matches_torch_distributions():
    # Independent reference: PyTorch's own MultivariateNormal KL.
    from torch.distributions import MultivariateNormal, kl_divergence
    from vfe3.divergence import kl
    g = torch.Generator().manual_seed(21)
    K = 4
    mu_q = torch.randn(3, K, generator=g)
    mu_t = torch.randn(3, K, generator=g)
    Aq = torch.randn(3, K, K, generator=g)
    At = torch.randn(3, K, K, generator=g)
    S_q = Aq @ Aq.transpose(-1, -2) + torch.eye(K)
    S_t = At @ At.transpose(-1, -2) + torch.eye(K)
    ref = kl_divergence(MultivariateNormal(mu_q, S_q), MultivariateNormal(mu_t, S_t))
    got = kl(mu_q, S_q, mu_t, S_t, family="gaussian_full", eps=0.0)
    assert torch.allclose(got, ref, atol=1e-3, rtol=1e-3)


def test_diagonal_kl_closed_form_1d():
    # 1-D Gaussian KL: 0.5 (v1/v2 + (m2-m1)^2/v2 - 1 + ln(v2/v1)).
    from vfe3.divergence import kl
    mu_q = torch.tensor([[0.5]])
    mu_t = torch.tensor([[-1.0]])
    v_q = torch.tensor([[2.0]])
    v_t = torch.tensor([[0.5]])
    expected = 0.5 * (2.0 / 0.5 + (1.5 ** 2) / 0.5 - 1.0 + torch.log(torch.tensor(0.5 / 2.0)))
    got = kl(mu_q, v_q, mu_t, v_t, family="gaussian_diagonal")
    assert torch.allclose(got, expected.reshape(1), atol=1e-5)


def test_register_divergence_records_cov_kind():
    """A family declares its covariance structure at registration; consumers read the declared
    kind via family_cov_kind rather than sniffing the family name."""
    from vfe3.divergence import family_cov_kind

    assert family_cov_kind("gaussian_diagonal") == "diagonal"
    assert family_cov_kind("gaussian_full") == "full"


def test_family_cov_kind_unregistered_raises():
    from vfe3.divergence import family_cov_kind

    with pytest.raises(KeyError):
        family_cov_kind("no_such_family")
