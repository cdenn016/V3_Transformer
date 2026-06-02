r"""Straight-through E-step gradient mode (manuscript Algorithm 1, GL(K)_attention.tex:2050).

The estimator: each inner E-step update computes its tangent (delta) DETACHED, but rebuilds
the belief grad-connected to the PREVIOUS belief (mu_next = mu_prev + delta.detach(),
sigma_next = retract(sigma_prev, delta.detach())), so d belief_next/d belief_prev = I flows
WITHOUT the second-order d delta/d belief_prev term that full unroll keeps. The phi step is
already straight-through; this brings the mu/sigma updates to match.

Oracle: straight_through changes only the BACKWARD, never the forward, so the converged
belief / logits / loss VALUE is byte-identical to unroll (torch.equal). The gradient flows,
but DIFFERS from unroll (the dropped second-order term).
"""

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel


def _model(e_step_gradient="unroll", detach_e_step=False, **over):
    cfg = dict(
        vocab_size=15, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=1,
        n_e_steps=2, e_mu_lr=0.1, e_sigma_lr=0.02, e_phi_lr=0.0,
        gradient_mode="filtering", e_step_gradient=e_step_gradient,
        detach_e_step=detach_e_step,
    )
    cfg.update(over)
    return VFEModel(VFE3Config(**cfg))


def _data(seed=0, V=15, B=2, N=4):
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, V, (B, N), generator=g)
    targets = torch.randint(0, V, (B, N), generator=g)
    return tokens, targets


# --- config validation -----------------------------------------------------
def test_config_accepts_the_three_modes():
    for mode in ("unroll", "straight_through", "detach"):
        cfg = VFE3Config(e_step_gradient=mode)
        assert cfg.e_step_gradient == mode


def test_config_rejects_unknown_mode():
    with pytest.raises(ValueError):
        VFE3Config(e_step_gradient="not_a_mode")


def test_config_default_is_unroll():
    assert VFE3Config().e_step_gradient == "unroll"


def test_config_detach_e_step_true_implies_detach_effective():
    # Back-compat: detach_e_step=True (with the default e_step_gradient='unroll') is accepted
    # and the effective mode is 'detach'.
    cfg = VFE3Config(detach_e_step=True)
    assert cfg.effective_e_step_gradient == "detach"


def test_config_contradictory_detach_plus_nonunroll_raises():
    # detach_e_step=True with a non-unroll e_step_gradient is contradictory -> ValueError.
    with pytest.raises(ValueError):
        VFE3Config(detach_e_step=True, e_step_gradient="straight_through")
    with pytest.raises(ValueError):
        VFE3Config(detach_e_step=True, e_step_gradient="detach")


def test_config_effective_mode_passthrough():
    assert VFE3Config(e_step_gradient="unroll").effective_e_step_gradient == "unroll"
    assert VFE3Config(e_step_gradient="straight_through").effective_e_step_gradient == "straight_through"
    assert VFE3Config(e_step_gradient="detach").effective_e_step_gradient == "detach"


# --- forward identity (the strong oracle) -----------------------------------
def test_straight_through_forward_byte_identical_to_unroll():
    # straight_through only changes the BACKWARD: mu_prev + delta is the same NUMBER whether
    # or not delta's graph is tracked, so converged logits/loss are byte-identical (torch.equal).
    tokens, targets = _data()
    m_un = _model("unroll")
    m_st = _model("straight_through")
    # same parameters (same seed in __init__ -> PriorBank init is seeded by run_training, not
    # here, so copy state to force identical weights).
    m_st.load_state_dict(m_un.state_dict())

    logits_un, loss_un, ce_un = m_un(tokens, targets)
    logits_st, loss_st, ce_st = m_st(tokens, targets)
    assert torch.equal(logits_un, logits_st)
    assert torch.equal(loss_un, loss_st)
    assert torch.equal(ce_un, ce_st)


def test_detach_forward_byte_identical_to_detach_e_step_true():
    # e_step_gradient='detach' reproduces detach_e_step=True: forward identical.
    tokens, targets = _data()
    m_new = _model("detach")
    m_old = _model("unroll", detach_e_step=True)
    m_old.load_state_dict(m_new.state_dict())
    logits_new, loss_new, _ = m_new(tokens, targets)
    logits_old, loss_old, _ = m_old(tokens, targets)
    assert torch.equal(logits_new, logits_old)
    assert torch.equal(loss_new, loss_old)


# --- gradients flow under straight_through ----------------------------------
def test_straight_through_grad_flows_to_encode_tables():
    tokens, targets = _data()
    model = _model("straight_through")
    _, loss, _ = model(tokens, targets)
    loss.backward()
    assert model.prior_bank.mu_embed.grad is not None
    assert torch.isfinite(model.prior_bank.mu_embed.grad).all()
    assert model.prior_bank.mu_embed.grad.abs().sum() > 0
    assert model.prior_bank.sigma_log_embed.grad is not None
    assert torch.isfinite(model.prior_bank.sigma_log_embed.grad).all()


# --- gradients DIFFER from unroll (distinct estimators) ---------------------
def test_straight_through_grad_differs_from_unroll():
    # unroll keeps the second-order d delta/d belief term; straight_through drops it. With a
    # config where the delta depends on the belief (e_mu_lr>0, e_sigma_lr>0, n_e_steps>=2 to
    # compound the second-order term), the mu_embed.grad must DIFFER between the two estimators.
    tokens, targets = _data()
    m_un = _model("unroll")
    m_st = _model("straight_through")
    m_st.load_state_dict(m_un.state_dict())

    _, loss_un, _ = m_un(tokens, targets)
    loss_un.backward()
    g_un = m_un.prior_bank.mu_embed.grad.clone()

    _, loss_st, _ = m_st(tokens, targets)
    loss_st.backward()
    g_st = m_st.prior_bank.mu_embed.grad.clone()

    assert not torch.allclose(g_st, g_un)


def test_detach_grad_matches_detach_e_step_true():
    # e_step_gradient='detach' reproduces detach_e_step=True backward: the E-step gradient is
    # absent on both, so the encode mu prior grad (reached only via decode) is identical.
    tokens, targets = _data()
    m_new = _model("detach")
    m_old = _model("unroll", detach_e_step=True)
    m_old.load_state_dict(m_new.state_dict())

    _, loss_new, _ = m_new(tokens, targets)
    loss_new.backward()
    _, loss_old, _ = m_old(tokens, targets)
    loss_old.backward()

    assert torch.allclose(
        m_new.prior_bank.mu_embed.grad, m_old.prior_bank.mu_embed.grad, atol=0.0, rtol=0.0
    )
    # phi prior is frozen under both detach paths (E-step severed)
    assert m_new.prior_bank.phi_embed.grad is None
    assert m_old.prior_bank.phi_embed.grad is None
