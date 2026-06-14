r"""use_prior_bank=False: the linear-projection decode boundary (VFE_2.0 parity).

The theoretically pure default (use_prior_bank=True) decodes via the KL-to-prior readout
(logits = -KL(q_i || pi_v)/tau_eff). VFE_2.0 also exposes a use_prior_bank=False ablation
whose decode is a plain linear output projection mu -> logits (sigma discarded); the user
gets better results there and wants the with/without comparison in V3 too. The encode and
the free-energy self-coupling stay on the PriorBank either way -- the toggle controls only
the decode side. V3 realizes the projection as a raw (V, K) nn.Parameter matmul (no
nn.Linear module), the single learned linear readout the user authorized as an exception.
"""

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel
from vfe3.model.prior_bank import PriorBank
from vfe3.train import build_optimizer


def test_linear_decode_is_plain_projection_of_mu():
    # use_prior_bank=False decode is exactly logits = mu_q @ W^T, W = output_proj_weight (V, K).
    V, K, n_gen = 12, 4, 16
    pb = PriorBank(V, K, n_gen, use_prior_bank=False)
    assert pb.output_proj_weight is not None
    assert pb.output_proj_weight.shape == (V, K)
    mu_q = torch.randn(2, 3, K)
    sigma_q = torch.rand(2, 3, K) + 0.5
    logits = pb.decode(mu_q, sigma_q)
    expected = mu_q @ pb.output_proj_weight.t()
    assert logits.shape == (2, 3, V)
    assert torch.allclose(logits, expected, atol=1e-5)


def test_linear_decode_discards_sigma():
    # The linear projection reads only mu: changing sigma must not move the logits.
    V, K, n_gen = 8, 4, 16
    pb = PriorBank(V, K, n_gen, use_prior_bank=False)
    mu_q = torch.randn(1, 2, K)
    l1 = pb.decode(mu_q, torch.rand(1, 2, K) + 0.5)
    l2 = pb.decode(mu_q, torch.rand(1, 2, K) + 5.0)
    assert torch.allclose(l1, l2)


def test_prior_bank_true_has_no_output_proj():
    # The pure path carries no extra projection weight.
    pb = PriorBank(6, 3, 9, use_prior_bank=True)
    assert pb.output_proj_weight is None


def test_encode_is_unchanged_by_use_prior_bank_toggle():
    # The toggle controls only decode; encode stays a per-token prior-bank lookup.
    V, K, n_gen = 6, 3, 9
    torch.manual_seed(0); pb_t = PriorBank(V, K, n_gen, use_prior_bank=True)
    torch.manual_seed(0); pb_f = PriorBank(V, K, n_gen, use_prior_bank=False)
    tok = torch.tensor([[0, 1, 2]])
    bt, bf = pb_t.encode(tok), pb_f.encode(tok)
    assert torch.allclose(bt.mu, bf.mu)
    assert torch.allclose(bt.sigma, bf.sigma)
    assert torch.allclose(bt.phi, bf.phi)


def test_no_nn_linear_module_even_under_use_prior_bank_false():
    # The exception is a single learned weight matrix (nn.Parameter), NOT an nn.Linear/MLP.
    import torch.nn as nn
    cfg = VFE3Config(vocab_size=10, embed_dim=4, n_heads=2, max_seq_len=3, use_prior_bank=False)
    model = VFEModel(cfg)
    for m in model.modules():
        assert not isinstance(m, (nn.Linear, nn.MultiheadAttention, nn.RNNBase, nn.Conv1d))


def test_model_forward_and_backward_under_use_prior_bank_false():
    cfg = VFE3Config(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=1,
                     n_e_steps=1, e_q_mu_lr=0.05, e_phi_lr=0.0, use_prior_bank=False)
    model = VFEModel(cfg)
    tokens = torch.randint(0, 20, (3, 5))
    targets = torch.randint(0, 20, (3, 5))
    logits, loss, _ = model(tokens, targets)
    assert logits.shape == (3, 5, 20)
    assert torch.isfinite(loss)
    loss.backward()
    g = model.prior_bank.output_proj_weight.grad
    assert g is not None and g.abs().sum() > 0          # the projection actually trains
    # encode-side mean prior still reached (encode stays on the prior bank)
    assert model.prior_bank.mu_embed.grad is not None
    assert model.prior_bank.mu_embed.grad.abs().sum() > 0


def test_optimizer_covers_all_params_under_use_prior_bank_false():
    # build_optimizer must group the output projection too: every model parameter is covered
    # by exactly one group (a missing group would silently freeze that weight).
    cfg = VFE3Config(vocab_size=20, embed_dim=4, n_heads=2, use_prior_bank=False)
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)
    grouped = {p for grp in opt.param_groups for p in grp["params"]}
    assert grouped == set(model.parameters())
    assert model.prior_bank.output_proj_weight in grouped


def test_optimizer_still_exactly_covers_params_under_use_prior_bank_true():
    cfg = VFE3Config(vocab_size=20, embed_dim=4, n_heads=2, use_prior_bank=True)
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)
    grouped = {p for grp in opt.param_groups for p in grp["params"]}
    assert grouped == set(model.parameters())


def test_use_prior_bank_false_with_detach_e_step_warns_encode_tables_frozen():
    # Audit guardrail: under use_prior_bank=False AND detach_e_step=True the detached E-step
    # severs the encode tables and the linear decode reads only mu_final, so only
    # output_proj_weight trains. The model must warn at construction (a silent freeze otherwise).
    cfg = VFE3Config(vocab_size=10, embed_dim=4, n_heads=2, max_seq_len=3,
                     use_prior_bank=False, detach_e_step=True)
    with pytest.warns(UserWarning, match="freezes the encode prior tables"):
        VFEModel(cfg)
