r"""Regression pins for the behavioral fixes from the 2026-06-14 overnight deep audit.

Covers the three NON-doc changes (the doc-only fixes L1/L2/L6/L7 and the prior_bank M1 docstring
are not string-pinned, per repo convention):
  M1 -- config warns when use_prior_bank=True decodes at fixed alpha=1 KL under a non-KL/non-alpha=1
        E-step seam (and is silent on the pure renyi/renyi_order=1 path).
  L5 -- the generic Renyi-from-A path is float64-guarded near alpha=1 (was ~1e-2 fp32 rel err).
  L8 -- safe_spd_inverse retries per element, so one non-PD batch element no longer poisons the
        exact Cholesky inverse of its well-conditioned siblings.
"""

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.families.base import _RENYI_KL_BAND, _renyi_from_log_partition
from vfe3.families.gaussian import DiagonalGaussian
from vfe3.numerics import safe_spd_inverse


# --- M1 -> PB-14 (2026-07-12): the decode/E-step divergence mismatch is now a HARD capability
# error, not a warning. A noncanonical divergence under use_prior_bank=True must select a
# family-consistent decode kernel ('family'/'family_chunked'); the fast gaussian kernels are
# rejected (they would read the belief out under the wrong geometry). --------------------------
_PB14_SUBSTR = "family-consistent decode_mode"


def test_m1_noncanonical_divergence_requires_family_consistent_decode():
    # renyi_order != 1 under the fast diagonal kernel: rejected.
    with pytest.raises(ValueError, match=_PB14_SUBSTR):
        VFE3Config(use_prior_bank=True, renyi_order=0.5)
    # a non-renyi functional under the fast diagonal kernel: rejected.
    with pytest.raises(ValueError, match=_PB14_SUBSTR):
        VFE3Config(use_prior_bank=True, divergence_family="squared_hellinger")
    # the family-consistent decode reads the configured divergence out and is accepted.
    VFE3Config(use_prior_bank=True, renyi_order=0.5, decode_mode="family")
    VFE3Config(use_prior_bank=True, divergence_family="squared_hellinger", decode_mode="family_chunked")


def test_m1_pure_kl_seam_and_linear_decode_construct():
    # the pure path (renyi / renyi_order=1) needs no family-consistent decode.
    VFE3Config(use_prior_bank=True)
    # use_prior_bank=False decodes linearly -- the decode/E-step mismatch does not apply.
    VFE3Config(use_prior_bank=False, renyi_order=0.5)


# --- L5: generic Renyi-from-A float64 cancellation guard near alpha=1 --------------------------
def _diag_pair(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    mu_q = torch.randn(6, generator=g)
    s_q = torch.rand(6, generator=g) + 0.5
    mu_p = torch.randn(6, generator=g)
    s_p = torch.rand(6, generator=g) + 0.5
    q = DiagonalGaussian(mu_q, s_q)
    p = DiagonalGaussian(mu_p, s_p)
    q64 = DiagonalGaussian(mu_q.double(), s_q.double())
    p64 = DiagonalGaussian(mu_p.double(), s_p.double())
    return q, p, q64, p64


@pytest.mark.parametrize("alpha", [1.0 + 5e-6, 1.0 + 1e-4, 1.0 - 1e-4])
def test_l5_generic_renyi_band_matches_float64(alpha):
    # inside the band but outside the <1e-6 KL switch: fp32 generic path must track the fp64 ref.
    assert 1e-6 <= abs(alpha - 1.0) < _RENYI_KL_BAND
    q, p, q64, p64 = _diag_pair()
    g = _renyi_from_log_partition(q, p, alpha=alpha, kl_max=float("inf"), eps=1e-6)
    r = _renyi_from_log_partition(q64, p64, alpha=alpha, kl_max=float("inf"), eps=1e-6)
    rel = (g - r).abs() / r.abs()
    assert rel < 1e-4, f"in-band generic Renyi fp32 rel err {rel.item():.2e} at alpha={alpha}"


@pytest.mark.parametrize("alpha", [0.5, 1.5])
def test_l5_generic_renyi_out_of_band_accurate(alpha):
    # outside the band the unchanged fp32 quotient is already accurate.
    q, p, q64, p64 = _diag_pair()
    g = _renyi_from_log_partition(q, p, alpha=alpha, kl_max=float("inf"), eps=1e-6)
    r = _renyi_from_log_partition(q64, p64, alpha=alpha, kl_max=float("inf"), eps=1e-6)
    assert ((g - r).abs() / r.abs()) < 1e-5


# --- L8: per-element safe_spd_inverse (no whole-batch poisoning) -------------------------------
def test_l8_safe_spd_inverse_per_element_no_poison():
    good = torch.tensor([[2.0, 0.5], [0.5, 1.0]])      # well-conditioned SPD
    bad = torch.tensor([[-1.0, 0.0], [0.0, -1.0]])     # neg-def: fails cholesky at every ridge
    inv = safe_spd_inverse(torch.stack([good, bad]))
    # the good element keeps the exact Cholesky inverse (identical to the single-matrix result,
    # i.e. it is NOT dragged onto the pinv path by its non-PD sibling).
    single = safe_spd_inverse(good)
    assert torch.allclose(inv[0], single, atol=0.0)
    assert torch.allclose(inv[0], torch.linalg.inv(good), atol=1e-4)
    assert torch.isfinite(inv[1]).all()                # non-PD element falls back to a finite pinv
