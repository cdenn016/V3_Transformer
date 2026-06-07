import torch
from torch import nn

from vfe3.geometry.generators import generate_glk_multihead
from vfe3.geometry.phi_preconditioner import precondition_phi_gradient
from vfe3.gauge_optim import GaugeNaturalGradAdamW


def test_gauge_step_is_natural_gradient_on_active_rows_only():
    # The gauge group is stepped by phi <- phi - lr * (momentum*buf + natgrad), NOT by AdamW.
    # With momentum 0 the step is exactly -lr * natural_gradient; rows with no gradient (tokens
    # absent from the "batch") are untouched; the non-gauge param still gets a standard AdamW move.
    G = generate_glk_multihead(4, 2).float()                  # K=4, d_h=2, n_gen=8, irrep [2,2]
    irrep = [2, 2]
    torch.manual_seed(0)
    phi = nn.Parameter(torch.randn(6, 8) * 0.5)               # gauge-frame table (V=6, n_gen=8)
    w   = nn.Parameter(torch.randn(6, 8))                     # a non-gauge parameter
    opt = GaugeNaturalGradAdamW(
        [{"params": [phi], "lr": 0.1, "gauge": True, "weight_decay": 0.0},
         {"params": [w],   "lr": 0.1}],
        G, irrep, precond_mode="pullback_per_block", gauge_momentum=0.0,
        weight_decay=0.0,
    )
    phi0, w0 = phi.detach().clone(), w.detach().clone()
    g = torch.zeros(6, 8)
    g[1] = torch.randn(8)
    g[4] = torch.randn(8)                                     # only rows 1 and 4 are "active"
    phi.grad = g.clone()
    w.grad = torch.randn(6, 8)
    opt.step()

    for r in (0, 2, 3, 5):                                    # inactive rows untouched
        assert torch.allclose(phi.detach()[r], phi0[r], atol=1e-7)
    nat = precondition_phi_gradient(g[[1, 4]], phi0[[1, 4]], G,
                                    mode="pullback_per_block", irrep_dims=irrep)
    assert torch.allclose(phi.detach()[1], phi0[1] - 0.1 * nat[0], atol=1e-5)
    assert torch.allclose(phi.detach()[4], phi0[4] - 0.1 * nat[1], atol=1e-5)
    assert not torch.allclose(w.detach(), w0)                 # non-gauge param moved (AdamW)


def test_pullback_carries_geometry_killing_does_not():
    # The crux of "geometric correctness": at nonzero phi the pullback natural-gradient direction
    # DEPARTS from the raw gradient (the exp-map metric is non-conformal), whereas the per-block
    # Killing metric is conformal (a scalar * I), so its natural gradient stays PARALLEL to the raw
    # gradient -- a no-op direction. Only pullback changes the trajectory.
    G = generate_glk_multihead(4, 2).double()
    irrep = [2, 2]
    torch.manual_seed(2)
    phi  = torch.randn(5, 8, dtype=torch.float64) * 0.6
    grad = torch.randn(5, 8, dtype=torch.float64)
    nat_pb  = precondition_phi_gradient(grad, phi, G, mode="pullback_per_block", irrep_dims=irrep)
    nat_kil = precondition_phi_gradient(grad, phi, G, mode="killing_per_block", irrep_dims=irrep)
    cos_pb  = torch.cosine_similarity(nat_pb,  grad, dim=-1)
    cos_kil = torch.cosine_similarity(nat_kil, grad, dim=-1)
    assert (cos_kil > 1 - 1e-9).all()                        # Killing per-block: parallel to grad
    assert (cos_pb.abs() < 0.999).any()                      # pullback: genuinely rotates the step


def test_build_optimizer_selects_natural_grad_and_trains_phi_embed():
    # End-to-end wiring: m_phi_natural_grad=True -> build_optimizer returns GaugeNaturalGradAdamW,
    # and one train step runs and moves the gauge-frame table (geometric path is live, not dead).
    from vfe3.config import VFE3Config
    from vfe3.model.model import VFEModel
    from vfe3.train import build_optimizer, train_step

    cfg = VFE3Config(
        vocab_size=12, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
        gauge_group="block_glk", pos_phi="none",
        m_phi_natural_grad=True, phi_precond_mode="pullback_per_block",
        m_phi_lr=0.1, m_gauge_momentum=0.9,
    )
    torch.manual_seed(0)
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)
    assert isinstance(opt, GaugeNaturalGradAdamW)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    phi0 = model.prior_bank.phi_embed.detach().clone()
    tokens  = torch.randint(0, 12, (4, 8))
    targets = torch.randint(0, 12, (4, 8))
    loss = train_step(model, opt, sched, tokens, targets, grad_clip=1.0)
    assert torch.isfinite(torch.tensor(float(loss)))
    assert not torch.allclose(model.prior_bank.phi_embed.detach(), phi0)
