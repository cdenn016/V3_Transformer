r"""lambda_h_mode='state_dependent_per_coord': the per-coordinate hyper-prior coupling.

The model-fiber analogue of lambda_alpha_mode='state_dependent_per_coord' (alpha_i.py). Each
model-channel coordinate k gets its own envelope weight lambda_h^(k)* = c0_h^(k)/(b0_h^(k)+KL_k)
on the per-coordinate hyper-prior divergence KL_k(s_i||r), with the regularizer R_h^(k) summed over
coordinates. Unlike the per-token state_dependent form (one weight on the SUMMED KL), this feeds the
UNSUMMED per-coordinate KL(s_i||r) (shape (...,N,K)) into the same envelope, so coordinates far from
the prior are shrunk differently from coordinates near it. Measured heterogeneity of the trained
per-coordinate KL(s||r) (2026-06-16) is large (within-token CV ~0.7 for frequent tokens), so the
per-coordinate weight genuinely differentiates -- the motivation for this knob.

These tests pin: (1) the registry accepts the mode and declares it per-coord; (2) the envelope is
the per-coordinate c0_h/(b0_h+KL_k) with the alpha per-coord form; (3) config validation mirrors
alpha's per-coord guards (diagonal family, decomposable divergence, (K,) b0_h/c0_h list); (4) the
model's weighted hyper-prior term equals the per-coordinate envelope oracle summed over K, differs
from the per-token form, and backprops into the s tables (both the scored and s_e_step routes).
"""

import warnings

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.families.gaussian import DiagonalGaussian
from vfe3.free_energy import self_divergence_per_coord
from vfe3.lambda_h_i import hyper_prior_lambda_h, _LAMBDA_H_MODES
from vfe3.model.model import VFEModel

PC = "state_dependent_per_coord"


def _pc_model(lambda_h: float = 0.5, *, lambda_h_mode: str = PC,
              b0_h=1.0, c0_h=1.0, seed: int = 0, **over) -> VFEModel:
    r"""Tiny scored-regime model (s_e_step=False) with a per-coord hyper-prior channel; the term is
    added to the loss as _hyper_prior_term so the weighting/regularizer is exercised at loss level."""
    cfg = VFE3Config(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=1,
                     n_e_steps=1, e_q_mu_lr=0.5, e_phi_lr=0.0, mass_phi=0.0,
                     mstep_self_coupling_weight=0.0, lambda_h=lambda_h,
                     lambda_h_mode=lambda_h_mode, b0_h=b0_h, c0_h=c0_h, seed=seed, **over)
    torch.manual_seed(seed)
    return VFEModel(cfg)


# --------------------------------------------------------------------------- registry

def test_per_coord_in_lambda_h_modes():
    # The new mode must be a valid lambda_h_mode (so _require accepts it at config construction).
    assert PC in _LAMBDA_H_MODES


def test_lambda_h_is_per_coord_flag():
    # lambda_h declares the per-coord reduction need (like alpha_is_per_coord), validating membership.
    from vfe3.lambda_h_i import lambda_h_is_per_coord
    assert lambda_h_is_per_coord(PC) is True
    assert lambda_h_is_per_coord("state_dependent") is False
    assert lambda_h_is_per_coord("constant") is False
    with pytest.raises(KeyError):
        lambda_h_is_per_coord("not_a_mode")


def test_per_coord_lambda_h_uses_per_dimension_kl():
    # The envelope is the per-coordinate c0_h/(b0_h+KL_k) on an unsummed (..,N,K) divergence, with
    # per-coordinate (K,) b0_h/c0_h, delegating to alpha's verified per-coord form.
    kl = torch.rand(2, 5, 4) + 0.1                       # (..., N, K) per-coordinate KL
    b0 = torch.tensor([0.3, 0.5, 0.7, 1.1])
    c0 = torch.tensor([0.7, 1.0, 1.3, 2.0])
    lam, reg = hyper_prior_lambda_h(kl, mode=PC, b0_h=b0, c0_h=c0)
    assert lam.shape == kl.shape                         # per-coordinate, NOT reduced
    assert torch.allclose(lam, c0 / (b0 + kl), atol=1e-6)
    # R_h^(k) = b0_k*lam_k - c0_k*log(lam_k), the envelope regularizer per coordinate
    assert torch.allclose(reg, b0 * lam - c0 * torch.log(lam.clamp(min=1e-12)), atol=1e-6)


# --------------------------------------------------------------------------- config validation

def test_config_accepts_per_coord_lambda_h_on_diagonal_family():
    c = VFE3Config(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5,
                   lambda_h=0.5, lambda_h_mode=PC)        # gaussian_diagonal + renyi defaults
    assert c.lambda_h_mode == PC


def test_per_coord_lambda_h_rejects_full_covariance_family():
    # The per-coordinate hyper-prior divergence needs a coordinate-decomposable family.
    with pytest.raises(ValueError, match="per-coordinate"):
        VFE3Config(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5,
                   lambda_h=0.5, lambda_h_mode=PC, family="gaussian_full")


def test_per_coord_lambda_h_rejects_nondecomposable_divergence():
    # squared_hellinger does not decompose coordinate-wise -> reject at construction.
    with pytest.raises(ValueError, match="decompose"):
        VFE3Config(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5,
                   lambda_h=0.5, lambda_h_mode=PC, divergence_family="squared_hellinger")


def test_b0_h_list_requires_per_coord_mode():
    # A (K,) b0_h/c0_h list shapes the per-coordinate envelope; it requires the per-coord mode.
    with pytest.raises(ValueError, match="per-coordinate"):
        VFE3Config(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5,
                   lambda_h=0.5, lambda_h_mode="state_dependent", b0_h=[0.3, 0.5, 0.7, 1.1])


def test_b0_h_list_length_must_equal_embed_dim():
    with pytest.raises(ValueError, match="length"):
        VFE3Config(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5,
                   lambda_h=0.5, lambda_h_mode=PC, b0_h=[0.3, 0.5, 0.7])     # len 3 != embed_dim 4


def test_b0_h_list_accepted_with_per_coord_mode():
    c = VFE3Config(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5,
                   lambda_h=0.5, lambda_h_mode=PC, b0_h=[0.3, 0.5, 0.7, 1.1], c0_h=[0.7, 1.0, 1.3, 2.0])
    assert c.b0_h == [0.3, 0.5, 0.7, 1.1]


# --------------------------------------------------------------------------- model integration

def test_per_coord_lambda_h_term_matches_per_coordinate_envelope():
    # The model's weighted hyper-prior term equals mean_i sum_k [ c0_h/(b0_h+KL_k)*KL_k + R_h^(k) ],
    # i.e. the per-coordinate envelope oracle summed over the K coordinate axis.
    m = _pc_model(0.5, b0_h=0.3, c0_h=0.7)
    tokens = torch.randint(0, 20, (3, 5))
    kl_pc = m._hyper_prior_kl(tokens, per_coord=True)            # (B, N, K) per-coordinate KL
    assert kl_pc.shape == (3, 5, 4)
    lam, reg = hyper_prior_lambda_h(kl_pc, mode=PC, value=0.5, b0_h=0.3, c0_h=0.7)
    oracle = (lam * kl_pc + reg).sum(dim=-1).mean()             # sum over K, mean over tokens
    assert torch.allclose(m._hyper_prior_term(tokens), oracle, atol=1e-7)


def test_per_coord_lambda_h_differs_from_per_token():
    # Per-coord feeds the UNSUMMED KL into the envelope then sums; per-token feeds the SUMMED KL.
    # They are genuinely different functionals (the whole motivation), so the terms must differ.
    m = _pc_model(0.5, b0_h=1.0, c0_h=1.0)
    tokens = torch.randint(0, 20, (3, 5))
    kl_pc = m._hyper_prior_kl(tokens, per_coord=True)           # (B, N, K)
    kl_tok = kl_pc.sum(dim=-1)                                  # (B, N) summed
    lam_pc, reg_pc = hyper_prior_lambda_h(kl_pc, mode=PC, b0_h=1.0, c0_h=1.0)
    lam_tok, reg_tok = hyper_prior_lambda_h(kl_tok, mode="state_dependent", b0_h=1.0, c0_h=1.0)
    per_coord_term = (lam_pc * kl_pc + reg_pc).sum(dim=-1).mean()
    per_token_term = (lam_tok * kl_tok + reg_tok).mean()
    assert not torch.allclose(per_coord_term, per_token_term, atol=1e-3)


def test_per_coord_lambda_h_grad_flows_to_s():
    # The per-coordinate scored term backprops a finite, nonzero gradient into the s tables.
    m = _pc_model(0.5)
    tokens = torch.randint(0, 20, (3, 5)); targets = torch.randint(0, 20, (3, 5))
    _, loss, _ = m(tokens, targets)
    loss.backward()
    g = m.prior_bank.s_mu_embed.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


def test_per_coord_lambda_h_in_s_e_step_trains_s():
    # The s E-step inherits per-coord routing through the shared lambda_alpha_mode seam (zero e_step
    # edits): under s_e_step the per-coord lambda_h trains s through the unrolled refine.
    cfg = VFE3Config(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=1,
                     n_e_steps=1, e_q_mu_lr=0.5, e_phi_lr=0.0, mass_phi=0.0,
                     mstep_self_coupling_weight=0.0, use_prior_bank=True,
                     lambda_h=1.0, lambda_h_mode=PC, prior_source="model_channel",
                     s_e_step=True, e_s_mu_lr=0.5, seed=0)
    torch.manual_seed(0)
    m = VFEModel(cfg)
    tokens = torch.randint(0, 20, (2, 5)); targets = torch.randint(0, 20, (2, 5))
    _, loss, _ = m(tokens, targets)
    loss.backward()
    g = m.prior_bank.s_mu_embed.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


def test_per_coord_lambda_h_per_coordinate_b0_list_in_model():
    # A (K,) b0_h/c0_h list reaches the model envelope (via _as_coeff): the term matches the oracle
    # computed with the per-coordinate (K,) priors.
    b0 = [0.3, 0.5, 0.7, 1.1]; c0 = [0.7, 1.0, 1.3, 2.0]
    m = _pc_model(0.5, b0_h=b0, c0_h=c0)
    tokens = torch.randint(0, 20, (3, 5))
    kl_pc = m._hyper_prior_kl(tokens, per_coord=True)
    lam, reg = hyper_prior_lambda_h(kl_pc, mode=PC, value=0.5,
                                    b0_h=torch.tensor(b0), c0_h=torch.tensor(c0))
    oracle = (lam * kl_pc + reg).sum(dim=-1).mean()
    assert torch.allclose(m._hyper_prior_term(tokens), oracle, atol=1e-6)
