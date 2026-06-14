r"""Tests for the LEARNABLE self-coupling alpha (a sanctioned nn.Parameter NN exception).

lambda_alpha_mode="learnable" creates a model-owned scalar log_alpha (alpha = exp(log_alpha));
log_alpha=0 -> alpha=1.0, so learnable-at-init must reproduce the constant alpha=1.0 pure
path exactly. Default-off: no log_alpha attribute is created under the (default) constant form.
"""

import torch

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel


def _cfg(**over):
    base = dict(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=2,
                n_e_steps=2, e_q_mu_lr=0.05, e_phi_lr=0.02)
    base.update(over)
    return VFE3Config(**base)


def test_config_accepts_learnable_alpha_mode():
    # "learnable" is a registered alpha form, so the registry-validated set accepts it.
    cfg = _cfg(lambda_alpha_mode="learnable")
    assert cfg.lambda_alpha_mode == "learnable"


def test_default_off_no_log_alpha_attribute():
    # Default (constant) model creates NO log_alpha parameter -- the pure no-NN path.
    model = VFEModel(_cfg())                       # lambda_alpha_mode defaults to "constant"
    assert not hasattr(model, "log_alpha")
    # state_dependent (also pure) likewise creates no log_alpha.
    model_sd = VFEModel(_cfg(lambda_alpha_mode="state_dependent"))
    assert not hasattr(model_sd, "log_alpha")


def test_learnable_creates_scalar_log_alpha_param():
    import torch.nn as nn
    model = VFEModel(_cfg(lambda_alpha_mode="learnable"))
    assert isinstance(model.log_alpha, nn.Parameter)
    assert model.log_alpha.shape == ()
    assert float(model.log_alpha.detach()) == 0.0   # init 0 -> alpha = exp(0) = 1.0


def test_learnable_init_equals_constant_one():
    # The independent oracle: learnable-at-init (log_alpha=0 -> alpha=1.0) must produce the
    # SAME forward as the constant alpha=1.0 pure path on the same seed/config.
    tok = torch.randint(0, 20, (3, 5))
    tgt = torch.randint(0, 20, (3, 5))

    torch.manual_seed(0)
    m_const = VFEModel(_cfg(lambda_alpha_mode="constant", lambda_alpha=1.0))
    torch.manual_seed(0)
    m_learn = VFEModel(_cfg(lambda_alpha_mode="learnable"))

    logits_c, loss_c, _ = m_const(tok, tgt)
    logits_l, loss_l, _ = m_learn(tok, tgt)
    assert torch.equal(logits_c, logits_l)        # byte-identical at init
    assert torch.equal(loss_c, loss_l)


def test_learnable_log_alpha_grad_populated():
    # The learned alpha actually trains: log_alpha.grad is finite, non-None, and nonzero after
    # backward (the self-coupling reaches the loss through the unrolled E-step).
    model = VFEModel(_cfg(lambda_alpha_mode="learnable", n_layers=1, n_e_steps=4,
                          e_q_mu_lr=0.3, e_q_sigma_lr=0.1, e_phi_lr=0.0))
    tok = torch.randint(0, 20, (2, 5))
    tgt = torch.randint(0, 20, (2, 5))
    _, loss, _ = model(tok, tgt)
    loss.backward()
    assert model.log_alpha.grad is not None
    assert torch.isfinite(model.log_alpha.grad)
    assert model.log_alpha.grad.abs() > 0          # genuinely in the loss graph, not a dead param


def test_learnable_alpha_changes_forward_when_log_alpha_moves():
    # Moving log_alpha away from 0 must change the converged belief / loss (the learned alpha
    # is genuinely consumed in F, not a dead parameter). A strong-enough E-step (more iterations,
    # larger learning rates) makes the small self-coupling signal observable above 1e-6.
    tok = torch.randint(0, 20, (2, 5))
    tgt = torch.randint(0, 20, (2, 5))
    cfg_kw = dict(lambda_alpha_mode="learnable", n_layers=1, n_e_steps=4,
                  e_q_mu_lr=0.3, e_q_sigma_lr=0.1, e_phi_lr=0.0,
                  use_prior_bank=True,   # observe alpha through the KL-to-prior decode (it reads
                  #                        sigma, where the self-coupling signal is strongest; the
                  #                        linear-decode default discards sigma)
                  pos_phi="none")   # isolate the alpha sensitivity: the positional gauge composition
                                    # damps the self-coupling signal below the 1e-6 detection floor
    torch.manual_seed(0)
    model = VFEModel(_cfg(**cfg_kw))
    _, loss0, _ = model(tok, tgt)
    with torch.no_grad():
        model.log_alpha.copy_(torch.log(torch.tensor(5.0)))     # alpha = 5.0
    _, loss1, _ = model(tok, tgt)
    assert not torch.allclose(loss0, loss1, atol=1e-6)


def test_learnable_with_detach_e_step_warns_and_freezes_log_alpha():
    # Footgun: the detached (no_grad) E-step severs log_alpha from the loss (alpha enters F only
    # through the E-step), so it stays frozen. __init__ must warn, and log_alpha.grad must be None.
    import pytest
    with pytest.warns(UserWarning, match="freezes log_alpha"):
        model = VFEModel(_cfg(lambda_alpha_mode="learnable", detach_e_step=True))
    tok = torch.randint(0, 20, (2, 5))
    tgt = torch.randint(0, 20, (2, 5))
    _, loss, _ = model(tok, tgt)
    loss.backward()
    assert model.log_alpha.grad is None              # frozen under detach


def test_learnable_diagnostics_runs():
    # diagnostics() consumes the learned alpha too (no-grad), and must run with finite outputs.
    import math
    model = VFEModel(_cfg(lambda_alpha_mode="learnable", n_layers=1))
    tok = torch.randint(0, 20, (2, 5))
    d = model.diagnostics(tok)
    assert math.isfinite(d["self_coupling"])
