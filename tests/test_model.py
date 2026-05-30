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
