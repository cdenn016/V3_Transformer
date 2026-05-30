import torch
from vfe3.belief import BeliefState
from vfe3.config import VFE3Config
from vfe3.geometry.groups import get_group
from vfe3.model.block import vfe_block
from vfe3.model.stack import vfe_stack


def _belief(N=4, K=4, n_gen=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return BeliefState(
        mu=torch.randn(N, K, generator=g),
        sigma=torch.rand(N, K, generator=g) + 0.5,
        phi=0.1 * torch.randn(N, n_gen, generator=g),
    )


def test_block_runs_e_step_and_preserves_shapes():
    cfg = VFE3Config(embed_dim=4, n_heads=2, n_e_steps=2, e_mu_lr=0.05, e_phi_lr=0.0)
    grp = get_group("block_glk")(4, 2)
    b = _belief(K=4, n_gen=grp.generators.shape[0])
    out = vfe_block(b, b.mu, b.sigma, grp, cfg)
    assert out.mu.shape == b.mu.shape and (out.sigma > 0).all()


def test_stack_handoff_updates_prior_across_blocks():
    cfg = VFE3Config(embed_dim=4, n_heads=2, n_layers=3, n_e_steps=1,
                     e_mu_lr=0.05, e_phi_lr=0.0, prior_handoff_rho=1.0)
    grp = get_group("block_glk")(4, 2)
    b = _belief(K=4, n_gen=grp.generators.shape[0])
    out = vfe_stack(b, b.mu, b.sigma, grp, cfg)
    assert out.mu.shape == b.mu.shape and (out.sigma > 0).all()
    # with rho=1 and a nonzero E-step, the stack moves the belief off the input
    assert not torch.allclose(out.mu, b.mu, atol=1e-4)


from vfe3.model.model import VFEModel


def test_model_forward_shapes_and_loss():
    cfg = VFE3Config(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=2,
                     n_e_steps=1, e_mu_lr=0.05, e_phi_lr=0.0)
    model = VFEModel(cfg)
    tokens = torch.randint(0, 20, (3, 5))
    logits = model(tokens)
    assert logits.shape == (3, 5, 20)
    targets = torch.randint(0, 20, (3, 5))
    _, loss, _ = model(tokens, targets)
    assert loss.shape == () and torch.isfinite(loss)


def test_loss_backward_reaches_prior_tables():
    # THE crown jewel: the unrolled E-step keeps the training graph connected, so the
    # M-step gradient reaches the encode/phi prior parameters (not just decode).
    cfg = VFE3Config(vocab_size=15, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=2,
                     n_e_steps=2, e_mu_lr=0.05, e_phi_lr=0.02, gradient_mode="filtering")
    model = VFEModel(cfg)
    tokens = torch.randint(0, 15, (2, 4)); targets = torch.randint(0, 15, (2, 4))
    _, loss, _ = model(tokens, targets)
    loss.backward()
    assert model.prior_bank.mu_embed.grad is not None
    assert model.prior_bank.phi_embed.grad is not None
    assert model.prior_bank.mu_embed.grad.abs().sum() > 0          # gradient actually flows
    assert model.prior_bank.phi_embed.grad.abs().sum() > 0


def test_model_has_no_nn_layers():
    import torch.nn as nn
    cfg = VFE3Config(vocab_size=10, embed_dim=4, n_heads=2, max_seq_len=3)
    model = VFEModel(cfg)
    for m in model.modules():
        assert not isinstance(m, (nn.Linear, nn.MultiheadAttention, nn.RNNBase, nn.Conv1d))
