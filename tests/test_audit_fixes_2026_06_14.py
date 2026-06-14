r"""Regression pins for the behavioral fixes from the 2026-06-14 overnight deep audit.

Covers the three NON-doc changes (the doc-only fixes L1/L2/L6/L7 and the prior_bank M1 docstring
are not string-pinned, per repo convention):
  M1 -- config warns when use_prior_bank=True decodes at fixed alpha=1 KL under a non-KL/non-alpha=1
        E-step seam (and is silent on the pure renyi/renyi_order=1 path).
  L5 -- the generic Renyi-from-A path is float64-guarded near alpha=1 (was ~1e-2 fp32 rel err).
  L8 -- safe_spd_inverse retries per element, so one non-PD batch element no longer poisons the
        exact Cholesky inverse of its well-conditioned siblings.
"""

import warnings

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.families.base import _RENYI_KL_BAND, _renyi_from_log_partition
from vfe3.families.gaussian import DiagonalGaussian
from vfe3.numerics import safe_spd_inverse


def _warns_matching(substr: str, **cfg_kw) -> bool:
    """True iff building VFE3Config(**cfg_kw) emits a warning whose message contains substr."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        VFE3Config(**cfg_kw)
    return any(substr in str(w.message) for w in rec)


# --- M1: decode-vs-E-step divergence-mismatch warning -----------------------------------------
_M1_SUBSTR = "decodes at a FIXED alpha=1 KL"


def test_m1_use_prior_bank_nonkl_seam_warns():
    # renyi_order != 1 under the prior-bank (KL) decode: warn.
    assert _warns_matching(_M1_SUBSTR, use_prior_bank=True, renyi_order=0.5)
    # a non-renyi functional under the prior-bank decode: warn.
    assert _warns_matching(_M1_SUBSTR, use_prior_bank=True, divergence_family="squared_hellinger")


def test_m1_pure_kl_seam_does_not_warn():
    # the pure path (renyi / renyi_order=1) is silent.
    assert not _warns_matching(_M1_SUBSTR, use_prior_bank=True)
    # use_prior_bank=False decodes linearly -- the decode/E-step mismatch does not apply.
    assert not _warns_matching(_M1_SUBSTR, use_prior_bank=False, renyi_order=0.5)


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


def test_l8_safe_spd_inverse_all_good_unchanged():
    g = torch.Generator().manual_seed(1)
    a = torch.randn(3, 4, 4, generator=g)
    spd = a @ a.transpose(-1, -2) + 0.1 * torch.eye(4)
    inv = safe_spd_inverse(spd)
    assert torch.allclose(inv, torch.linalg.inv(spd), atol=1e-3)
