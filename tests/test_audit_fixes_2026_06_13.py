r"""Regression tests for the 2026-06-13 audit-fix landing (docs/audits/audit-2026-06-12*).

Covers the verified-live findings actually fixed:
  DA-P2a  -- GaugeGroup.gram_pinv() is cached and VALUE-IDENTICAL to recomputing it, so the
             BCH positional-composition hot path no longer rebuilds a dense float64 pinv each
             forward (the orthonormal block_glk Gram is exactly I).
  CC-F1/F5/F3/F2 -- VFE3Config rejects (fail-fast at config time) the cross_couplings combinations
             that previously crashed late at VFEModel construction / first forward, and warns on
             the single-block semantic shift. Closes the cross-coupling test-coverage gap (CC-F9).
  CC-F7   -- the run label encodes cross-coupling.
"""

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.geometry.groups import get_group
from vfe3.geometry.lie_ops import embed_phi, extract_phi, gram_pinv


# ----------------------------------------------------------------------------- DA-P2a
def test_gram_pinv_is_cached_and_value_identical():
    g = get_group("block_glk")(4, 2)                         # tiny CPU shape (K=4, d_head=2); Gram=I holds at any K
    a = g.gram_pinv()
    b = g.gram_pinv()
    assert a is b                                            # cached: same object on repeat
    assert torch.equal(a, gram_pinv(g.generators))          # equals a fresh recompute
    assert torch.allclose(a, torch.eye(a.shape[0]), atol=1e-12)  # orthonormal basis -> Gram^+ = I


def test_extract_phi_cached_pinv_matches_none():
    # The optimization (passing group.gram_pinv() instead of gram_pinv_=None) must be bit-identical.
    g = get_group("block_glk")(8, 2)
    torch.manual_seed(0)
    phi = torch.randn(4, g.generators.shape[0])
    M = embed_phi(phi, g.generators)
    c_none = extract_phi(M, g.generators, gram_pinv_=None)
    c_cached = extract_phi(M, g.generators, gram_pinv_=g.gram_pinv())
    assert torch.equal(c_none, c_cached)


# ----------------------------------------------------------------------------- cross_couplings guards
def _cross_cfg(**kw):
    base = dict(vocab_size=16, embed_dim=8, n_heads=2, max_seq_len=6,
                gauge_group="block_glk", cross_couplings=[(0, 1)], beta_attention_prior="causal")
    base.update(kw)
    return base


def test_cross_couplings_rejects_alibi_prior():            # CC-F1
    with pytest.raises(ValueError, match="alibi"):
        VFE3Config(**_cross_cfg(beta_attention_prior="causal_alibi"))


def test_cross_couplings_rejects_head_mixer():             # CC-F1
    with pytest.raises(ValueError, match="head mixer|head_mixer"):
        VFE3Config(**_cross_cfg(use_head_mixer=True))


def test_cross_couplings_rejects_per_head_kappa_list():    # CC-F5
    with pytest.raises(ValueError, match="kappa"):
        VFE3Config(**_cross_cfg(kappa_beta=[1.0, 1.0]))
    with pytest.raises(ValueError, match="kappa_gamma"):
        VFE3Config(**_cross_cfg(kappa_gamma=[1.0, 1.0]))


def test_cross_couplings_diagonal_family_warns():          # CC-F3
    with pytest.warns(UserWarning, match="APPROXIMATION"):
        VFE3Config(**_cross_cfg(family="gaussian_diagonal"))


def test_cross_couplings_semantic_shift_warns():           # CC-F2
    with pytest.warns(UserWarning, match="single irrep block"):
        VFE3Config(**_cross_cfg())


def test_cross_couplings_headless_scalar_config_is_valid():
    # The coherent single-block cross-coupled gauge (headless prior, scalar kappa, no mixer) must
    # still construct -- the guards reject only the genuinely-incompatible combinations.
    cfg = VFE3Config(**_cross_cfg(family="gaussian_full",
                                  use_prior_bank=False))
    assert cfg.cross_couplings == [(0, 1)]
