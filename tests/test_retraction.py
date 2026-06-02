import pytest
import torch

from vfe3.geometry.retraction import (
    get_retraction,
    natural_gradient,
    register_retraction,
    retract_spd_diagonal,
    retract_spd_full,
)


def test_diagonal_retraction_positive_and_bounded():
    g = torch.Generator().manual_seed(0)
    sigma = torch.rand(4, 6, generator=g) + 0.1
    delta = 5.0 * torch.randn(4, 6, generator=g)
    out = retract_spd_diagonal(sigma, delta, sigma_max=5.0)
    assert (out >= 1e-6).all()
    assert (out <= 5.0 + 1e-6).all()


def test_full_retraction_stays_spd():
    g = torch.Generator().manual_seed(1)
    A = torch.randn(3, 4, 4, generator=g)
    sigma = A @ A.transpose(-1, -2) + torch.eye(4)
    D = torch.randn(3, 4, 4, generator=g)
    delta = 0.5 * (D + D.transpose(-1, -2))
    out = retract_spd_full(sigma, delta)
    assert torch.allclose(out, out.transpose(-1, -2), atol=1e-4)
    assert (torch.linalg.eigvalsh(out) > 0).all()


def test_full_retraction_identity_tangent_is_identity():
    g = torch.Generator().manual_seed(2)
    A = torch.randn(3, 4, 4, generator=g)
    sigma = A @ A.transpose(-1, -2) + torch.eye(4)
    zero = torch.zeros(3, 4, 4)
    out = retract_spd_full(sigma, zero)
    assert torch.allclose(out, sigma, atol=1e-3)


def test_natural_gradient_diagonal_formula():
    g = torch.Generator().manual_seed(3)
    sigma = torch.rand(4, 5, generator=g) + 0.1
    gmu = torch.randn(4, 5, generator=g)
    gsig = torch.randn(4, 5, generator=g)
    nmu, nsig = natural_gradient(gmu, gsig, sigma)
    assert torch.allclose(nmu, sigma * gmu, atol=1e-6)
    assert torch.allclose(nsig, 2.0 * sigma * sigma * gsig, atol=1e-6)


def test_full_retraction_K1_matches_diagonal_formula():
    # For K=1 the affine-invariant SPD exp map reduces to the diagonal rule
    # sigma_new = sigma * exp(tau * delta/sigma).
    sigma = torch.tensor([[[2.0]]])          # (1,1,1) as (B,K,K) with K=1
    delta = torch.tensor([[[0.6]]])
    out = retract_spd_full(sigma, delta, trust_region=0.0)   # disable TR for exact check
    expected = 2.0 * torch.exp(torch.tensor(0.6 / 2.0))
    assert torch.allclose(out.reshape(()), expected, atol=1e-4)


# --- register_retraction / get_retraction seam (roadmap item 4) ------------
def test_retraction_registry_round_trip():
    """register_retraction/get_retraction round-trip and the unknown-name KeyError."""
    sentinel = object()

    @register_retraction("_test_dummy_retraction")
    def _dummy(*args, **kwargs):
        return sentinel

    try:
        assert get_retraction("_test_dummy_retraction")() is sentinel
        with pytest.raises(KeyError):
            get_retraction("nope")
    finally:
        from vfe3.geometry.retraction import _RETRACTIONS
        _RETRACTIONS.pop("_test_dummy_retraction", None)


def test_spd_affine_is_registered():
    """The default affine-invariant SPD retraction is registered under 'spd_affine'."""
    assert callable(get_retraction("spd_affine"))


def test_spd_affine_bit_identical_diagonal():
    """spd_affine reproduces the bare retract_spd_diagonal call bit-for-bit (atol=0)."""
    g = torch.Generator().manual_seed(11)
    sigma = torch.rand(3, 4, 5, generator=g) + 0.1      # (B, N, K) diagonal
    nat_sigma = torch.randn(3, 4, 5, generator=g)
    step = 0.015
    legacy = retract_spd_diagonal(
        sigma, -step * nat_sigma, trust_region=5.0, eps=1e-6, sigma_max=5.0,
    )
    seam = get_retraction("spd_affine")(
        sigma, -step * nat_sigma, mean_ndim=3, trust_region=5.0, eps=1e-6, sigma_max=5.0,
    )
    assert torch.equal(seam, legacy)


def test_spd_affine_bit_identical_full():
    """spd_affine reproduces the bare retract_spd_full call bit-for-bit (atol=0)."""
    g = torch.Generator().manual_seed(12)
    A = torch.randn(3, 4, 5, 5, generator=g)
    sigma = A @ A.transpose(-1, -2) + torch.eye(5)       # (B, N, K, K) full SPD
    D = torch.randn(3, 4, 5, 5, generator=g)
    nat_sigma = 0.5 * (D + D.transpose(-1, -2))
    step = 0.015
    legacy = retract_spd_full(
        sigma, -step * nat_sigma, trust_region=5.0, eps=1e-6, sigma_max=5.0,
    )
    seam = get_retraction("spd_affine")(
        sigma, -step * nat_sigma, mean_ndim=3, trust_region=5.0, eps=1e-6, sigma_max=5.0,
    )
    assert torch.equal(seam, legacy)
