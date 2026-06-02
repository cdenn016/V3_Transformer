import math

import pytest
import torch

from vfe3.families.base import BeliefParams


class _ExpFamily(BeliefParams):
    r"""Univariate exponential family, parameter lam > 0.

    p(x; lam) = lam exp(-lam x), natural eta = -lam, A(eta) = -log(-eta),
    E[T] = 1/lam. Defines ONLY natural/log_partition_at/expected_statistic (no
    moment closed form), so it drives the generic Bregman/Renyi-from-A path.
    Per the families convention, ``natural``/``expected_statistic`` carry a
    trailing coordinate axis of size 1 (so the inner-product .sum(dim=-1) is
    well-defined), while ``log_partition_at`` sums that coordinate axis.
    """

    cov_kind = "diagonal"

    def __init__(self, lam): self.lam = lam                          # (...,)
    def coordinate_dim(self): return 1
    def block(self, start, end): return _ExpFamily(self.lam)
    def broadcast_over_keys(self): return _ExpFamily(self.lam.unsqueeze(-1))
    def natural(self): return (-self.lam.unsqueeze(-1),)             # (..., 1)
    @classmethod
    def log_partition_at(cls, theta): return (-torch.log(-theta[0])).sum(dim=-1)
    def expected_statistic(self): return (1.0 / self.lam.unsqueeze(-1),)  # (..., 1)
    def entropy(self): return 1.0 - torch.log(self.lam)


def test_family_registry_register_get_and_cov_kind():
    from vfe3.families.base import (
        BeliefParams, register_family, get_family, family_cov_kind, divergence_families,
    )

    class _ToyParams(BeliefParams):
        cov_kind = "diagonal"
        def __init__(self, x): self.x = x
        def coordinate_dim(self): return self.x.shape[-1]
        def block(self, start, end): return _ToyParams(self.x[..., start:end])
        def broadcast_over_keys(self): return _ToyParams(self.x.unsqueeze(-2))
        def natural(self): return (self.x,)
        @classmethod
        def log_partition_at(cls, theta): return theta[0].sum(dim=-1)
        def entropy(self): return self.x.sum(dim=-1) * 0.0

    register_family("toy_reg_test")(_ToyParams)
    try:
        assert get_family("toy_reg_test") is _ToyParams
        assert family_cov_kind("toy_reg_test") == "diagonal"
        assert "toy_reg_test" in divergence_families()
    finally:
        from vfe3.families.base import _FAMILIES
        _FAMILIES.pop("toy_reg_test", None)


def test_family_cov_kind_unregistered_raises():
    from vfe3.families.base import family_cov_kind
    with pytest.raises(KeyError):
        family_cov_kind("no_such_family")


def test_generic_kl_from_A_matches_exponential_closed_form():
    from vfe3.families.base import kl
    l1 = torch.tensor([2.0, 0.5, 1.0])
    l2 = torch.tensor([1.0, 1.5, 1.0])
    got = kl(_ExpFamily(l1), _ExpFamily(l2))
    want = torch.log(l1 / l2) + l2 / l1 - 1.0
    assert torch.allclose(got, want, atol=1e-5), (got, want)


def test_generic_renyi_from_A_matches_exponential_closed_form():
    from vfe3.families.base import renyi
    l1 = torch.tensor([2.0, 0.5])
    l2 = torch.tensor([1.0, 1.5])
    for a in (0.3, 0.7):
        got = renyi(_ExpFamily(l1), _ExpFamily(l2), alpha=a)
        want = (-torch.log(a * l1 + (1.0 - a) * l2) + a * torch.log(l1)
                + (1.0 - a) * torch.log(l2)) / (a - 1.0)
        assert torch.allclose(got, want, atol=1e-5), (a, got, want)


def test_diagonal_gaussian_closed_form_matches_legacy_divergence():
    from vfe3.divergence import renyi as legacy_renyi          # still the tensor API at this point
    from vfe3.families.gaussian import DiagonalGaussian
    from vfe3.families.base import renyi as fam_renyi
    torch.manual_seed(3)
    mu_q, mu_p = torch.randn(5, 4), torch.randn(5, 4)
    s_q, s_p = torch.rand(5, 4) + 0.5, torch.rand(5, 4) + 0.5
    for a in (0.5, 1.0):
        want = legacy_renyi(mu_q, s_q, mu_p, s_p, alpha=a, family="gaussian_diagonal")
        got = fam_renyi(DiagonalGaussian(mu_q, s_q), DiagonalGaussian(mu_p, s_p), alpha=a)
        assert torch.allclose(got, want, atol=1e-6), (a, (got - want).abs().max())


def test_diagonal_gaussian_generic_from_A_equals_closed_form():
    from vfe3.families.gaussian import DiagonalGaussian
    from vfe3.families.base import _renyi_from_log_partition
    torch.manual_seed(4)
    mu_q, mu_p = torch.randn(6, 3), torch.randn(6, 3)
    s_q, s_p = torch.rand(6, 3) + 0.5, torch.rand(6, 3) + 0.5
    q, p = DiagonalGaussian(mu_q, s_q), DiagonalGaussian(mu_p, s_p)
    for a in (0.5, 1.0):
        closed = q.renyi_closed_form(p, alpha=a, kl_max=float("inf"), eps=1e-6)
        generic = _renyi_from_log_partition(q, p, alpha=a, kl_max=float("inf"), eps=1e-6)
        assert torch.allclose(closed, generic, atol=1e-4), (a, (closed - generic).abs().max())


def test_diagonal_block_and_broadcast():
    from vfe3.families.gaussian import DiagonalGaussian
    mu, s = torch.randn(2, 6), torch.rand(2, 6) + 0.5
    q = DiagonalGaussian(mu, s)
    qb = q.block(2, 4)
    assert torch.equal(qb.mu, mu[..., 2:4]) and torch.equal(qb.sigma, s[..., 2:4])
    qk = q.broadcast_over_keys()
    assert qk.mu.shape == (2, 1, 6) and qk.sigma.shape == (2, 1, 6)
