r"""Regression pins for the behavioral fixes from the 2026-06-14 ultra-deep audit punch list.

Covers the non-doc changes (the doc-only Fix 6 _refine_s RoPE note and the perf-only Fix 7
oh.double() hoist are not string-pinned -- Fix 7's value-equality is held by
test_fullcov_alpha_roadmap_2026_06_13.test_factored_full_cov_sandwich_equals_dense):
  Fix 2 -- the Fisher natural-gradient preconditioner is FAMILY-KEYED: DiagonalLaplace uses its own
           Fisher (I_mu=I_b=1/b^2 -> b^2*grad on both coords), the Gaussian families delegate
           byte-identically to the pinned geometry kernel, and the base contract raises.
  Fix 1 -- config warns when use_prior_bank=True decodes a non-Gaussian belief through the hardcoded
           Gaussian KL readout (silent on a Gaussian family / use_prior_bank=False).
  Fix 3 -- config rejects t5_max_distance <= t5_num_buckets//2 (the log-bucketing denominator).
  Fix 4 -- the t5_bias freeze warning fires under BOTH severing E-step estimators (detach AND
           straight_through), and t5_learnable_bias with no t5 channel warns it is inert.
  Fix 5 -- config warns when s_e_step refines a non-Gaussian belief's model channel as Gaussian.
"""

import warnings

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.families.base import BeliefParams, get_family
from vfe3.geometry.retraction import natural_gradient as _geom_natural_gradient


def _warns_matching(substr: str, **cfg_kw) -> bool:
    """True iff building VFE3Config(**cfg_kw) emits a warning whose message contains substr."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        VFE3Config(**cfg_kw)
    return any(substr in str(w.message) for w in rec)


def _model_warns_matching(substr: str, **cfg_kw) -> bool:
    """True iff constructing a VFEModel from this config emits a warning containing substr."""
    from vfe3.model.model import VFEModel
    base = dict(vocab_size=12, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                n_e_steps=1, e_phi_lr=0.0, m_phi_lr=0.0)
    base.update(cfg_kw)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        VFEModel(VFE3Config(**base))
    return any(substr in str(w.message) for w in rec)


# --- Fix 2: family-keyed Fisher natural gradient ----------------------------------------------
def _grads(seed: int = 0, K: int = 8):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(4, K, generator=g), torch.randn(4, K, generator=g),
            torch.randn(4, K, generator=g), torch.rand(4, K, generator=g) + 0.3)


def test_laplace_natural_gradient_is_b_squared_grad_on_both_coords():
    # Laplace(mu,b) Fisher I_mu=I_b=1/b^2 (verified symbolically) -> nat-grad = b^2 * grad on BOTH.
    grad_mu, grad_sigma, mu, b = _grads()
    nat_mu, nat_sigma = get_family("laplace_diagonal")(mu, b).natural_gradient(
        grad_mu, grad_sigma, eps=1e-6)
    b2 = b.clamp(min=1e-6) ** 2
    assert torch.allclose(nat_mu, b2 * grad_mu, atol=1e-6)
    assert torch.allclose(nat_sigma, b2 * grad_sigma, atol=1e-6)


def test_gaussian_natural_gradient_delegates_byte_identically():
    # The Gaussian families must reproduce the pinned geometry kernel EXACTLY (byte-identical),
    # so the golden Gaussian E-step trajectory is unchanged by the family-keying.
    grad_mu, grad_sigma, mu, sigma = _grads()
    gnm, gns = get_family("gaussian_diagonal")(mu, sigma).natural_gradient(
        grad_mu, grad_sigma, eps=1e-6)
    fnm, fns = _geom_natural_gradient(grad_mu, grad_sigma, sigma, eps=1e-6)
    assert torch.equal(gnm, fnm) and torch.equal(gns, fns)
    # full covariance: the free fn selects the full branch by rank; delegation must match it.
    g = torch.Generator().manual_seed(3)
    A = torch.randn(3, 5, 5, generator=g)
    cov = A @ A.transpose(-1, -2) + 0.1 * torch.eye(5)
    gmu_f = torch.randn(3, 5, generator=g)
    gsig_f = torch.randn(3, 5, 5, generator=g)
    fnm2, fns2 = get_family("gaussian_full")(gmu_f, cov).natural_gradient(gmu_f, gsig_f, eps=1e-6)
    rnm2, rns2 = _geom_natural_gradient(gmu_f, gsig_f, cov, eps=1e-6)
    assert torch.equal(fnm2, rnm2) and torch.equal(fns2, rns2)


def test_laplace_mean_step_differs_from_gaussian_fisher():
    # The bug: the Gaussian Fisher mis-scales the Laplace mean by a state-dependent 1/b (it uses
    # sigma*grad = b*grad, not b^2*grad). The two preconditioners must genuinely differ on the mean.
    grad_mu, grad_sigma, mu, b = _grads()
    lap_mu, _ = get_family("laplace_diagonal")(mu, b).natural_gradient(grad_mu, grad_sigma, eps=1e-6)
    gauss_mu, _ = _geom_natural_gradient(grad_mu, grad_sigma, b, eps=1e-6)   # b mistaken as variance
    assert not torch.allclose(lap_mu, gauss_mu)
    # specifically the Laplace mean is the Gaussian mean times b (b^2*grad vs b*grad).
    assert torch.allclose(lap_mu, b.clamp(min=1e-6) * gauss_mu, atol=1e-6)


def test_laplace_natural_gradient_sign_preserving_and_zero_at_zero():
    # b^2 > 0 strictly: the preconditioner is sign-preserving and grad=0 -> step=0 (so every
    # stationary point of F is preserved -- the challenge-tier property).
    _, _, mu, b = _grads()
    grad_mu = torch.randn_like(mu)
    nat_mu, nat_sigma = get_family("laplace_diagonal")(mu, b).natural_gradient(
        grad_mu, torch.zeros_like(b), eps=1e-6)
    assert torch.all(torch.sign(nat_mu) == torch.sign(grad_mu))
    assert torch.allclose(nat_sigma, torch.zeros_like(nat_sigma))


def test_base_natural_gradient_raises_for_undeclared_family():
    # A family that does not declare its Fisher must raise (no silent Gaussian default) so a new
    # family cannot ride the wrong metric undetected.
    class _Bare(BeliefParams):
        cov_kind = "diagonal"
        def coordinate_dim(self): return 1
        def block(self, s, e): return self
        def broadcast_over_keys(self): return self
        def natural(self): raise NotImplementedError
        @classmethod
        def log_partition_at(cls, theta): raise NotImplementedError
        def entropy(self): return torch.zeros(1)
    with pytest.raises(NotImplementedError, match="natural_gradient"):
        _Bare().natural_gradient(torch.zeros(2), torch.zeros(2))


# --- Fix 1 -> PB-14 (2026-07-12): a non-Gaussian belief under use_prior_bank=True is now a HARD
# capability error unless the decode is family-consistent ('family'/'family_chunked'), which reads
# the belief out under its own geometry. The fast gaussian kernels are rejected. ---------------
_F1 = "family-consistent decode_mode"


def test_laplace_use_prior_bank_requires_family_consistent_decode():
    with pytest.raises(ValueError, match=_F1):
        VFE3Config(family="laplace_diagonal", use_prior_bank=True)          # fast gaussian kernel rejected
    # the family-consistent decode reads the Laplace belief out under the Laplace divergence.
    VFE3Config(family="laplace_diagonal", use_prior_bank=True, decode_mode="family")


def test_gaussian_or_linear_decode_construct():
    # canonical gaussian + KL keeps its fast kernel.
    VFE3Config(family="gaussian_diagonal", use_prior_bank=True)
    # use_prior_bank=False is the family-agnostic linear decode -> no mismatch.
    VFE3Config(family="laplace_diagonal", use_prior_bank=False)


# --- Fix 3: T5 bucket denominator guard --------------------------------------------------------
def test_t5_max_distance_guard_rejects_degenerate_combo():
    # md == nb//2 -> log(1)=0 -> division by zero; md < nb//2 -> negative bucket index.
    with pytest.raises(ValueError, match="t5_max_distance"):
        VFE3Config(t5_num_buckets=64, t5_max_distance=32)          # == nb//2
    with pytest.raises(ValueError, match="t5_max_distance"):
        VFE3Config(t5_num_buckets=64, t5_max_distance=16)          # < nb//2


def test_t5_max_distance_guard_accepts_valid_combos():
    VFE3Config(t5_num_buckets=64, t5_max_distance=33)              # just above nb//2
    VFE3Config()                                                   # defaults 32/128 are safe


# --- Fix 4: t5_bias freeze warning under BOTH severing estimators + inert toggle ---------------
_F4 = "freezes t5_bias"


def test_t5_bias_freeze_warns_under_straight_through_and_detach():
    # straight_through severs the detached belief tangent (the bug: previously unwarned).
    assert _model_warns_matching(_F4, beta_attention_prior="t5_relative_bias",
                                 t5_learnable_bias=True, e_step_gradient="straight_through")
    # detach (the originally-warned case) still warns.
    assert _model_warns_matching(_F4, beta_attention_prior="t5_relative_bias",
                                 t5_learnable_bias=True, detach_e_step=True)


def test_t5_bias_no_freeze_warning_under_unroll():
    assert not _model_warns_matching(_F4, beta_attention_prior="t5_relative_bias",
                                     t5_learnable_bias=True)        # default unroll trains it


def test_t5_learnable_bias_inert_without_channel_warns():
    # t5_learnable_bias=True but no t5_relative_bias channel: silently inert -> warn (CR-1).
    assert _model_warns_matching("the toggle is inert", beta_attention_prior="uniform",
                                 t5_learnable_bias=True)


# --- Fix 5 -> PB-11 (2026-07-12): _refine_s now refines the model channel in cfg.family, so the old
# "runs Gaussian while the belief is <family>" mixed-family warning is obsolete and removed. -----
_F5 = "mixed-family"


def test_s_e_step_non_gaussian_family_no_longer_warns_mixed_family():
    # _refine_s dispatches through get_family(cfg.family): a Laplace s channel IS refined as Laplace,
    # so there is no Gaussian/Laplace mismatch to warn about anymore.
    assert not _warns_matching(_F5, family="laplace_diagonal", s_e_step=True,
                               prior_source="model_channel", lambda_h=1.0)


def test_s_e_step_gaussian_family_silent():
    assert not _warns_matching(_F5, family="gaussian_diagonal", s_e_step=True,
                               prior_source="model_channel", lambda_h=1.0)
