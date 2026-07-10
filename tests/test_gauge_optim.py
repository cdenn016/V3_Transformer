import pytest
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


def test_adam_rule_with_identity_metric_matches_plain_adamw():
    # gauge_update_rule='adam' runs Adam's m/v/bias-correction ON the natural gradient. With
    # precond_mode='none' the metric is the identity (nat == grad), so the gauge step must reproduce
    # plain AdamW on phi (weight_decay 0, same betas/eps) -- this pins the new rule to the optimizer
    # it is meant to be, and is exactly why the conformal-killing arm collapses to AdamW.
    G = generate_glk_multihead(4, 2).double()
    irrep = [2, 2]
    torch.manual_seed(0)
    init  = torch.randn(6, 8, dtype=torch.float64) * 0.5
    phi_g = nn.Parameter(init.clone())                       # stepped by the gauge 'adam' rule
    phi_r = nn.Parameter(init.clone())                       # stepped by reference AdamW
    opt_g = GaugeNaturalGradAdamW(
        [{"params": [phi_g], "lr": 0.05, "gauge": True, "weight_decay": 0.0}],
        G, irrep, precond_mode="none", gauge_update_rule="adam", weight_decay=0.0,
    )
    opt_r = torch.optim.AdamW([phi_r], lr=0.05, weight_decay=0.0)
    torch.manual_seed(1)
    for _ in range(20):
        g = torch.randn(6, 8, dtype=torch.float64)           # dense -> every row active (nat == grad)
        phi_g.grad = g.clone()
        phi_r.grad = g.clone()
        opt_g.step()
        opt_r.step()
        assert torch.allclose(phi_g.detach(), phi_r.detach(), atol=1e-10, rtol=0)


def test_adam_rule_normalizes_tiny_gradient_unlike_heavy_ball():
    # The empirical failure mode the toggle fixes: a tiny/badly-scaled phi gradient barely moves the
    # frame under heavy-ball (no magnitude normalization), but the 'adam' rule rescales each
    # coordinate by 1/sqrt(v) to an ~lr-sized step, so phi actually trains. Same gradient, same lr.
    G = generate_glk_multihead(4, 2).double()
    irrep = [2, 2]
    torch.manual_seed(0)
    init = torch.randn(6, 8, dtype=torch.float64) * 0.5
    tiny = torch.randn(6, 8, dtype=torch.float64) * 1e-3     # small, like the real phi gradient (~0.06)

    def run(rule):
        phi = nn.Parameter(init.clone())
        opt = GaugeNaturalGradAdamW(
            [{"params": [phi], "lr": 0.05, "gauge": True, "weight_decay": 0.0}],
            G, irrep, precond_mode="none", gauge_update_rule=rule,
            gauge_momentum=0.9, weight_decay=0.0,
        )
        for _ in range(10):
            phi.grad = tiny.clone()
            opt.step()
        return (phi.detach() - init).norm().item()

    moved_hb   = run("heavy_ball")
    moved_adam = run("adam")
    assert moved_adam > 20 * moved_hb                         # adam normalizes; heavy-ball crawls


def test_state_dict_roundtrips_omega_reorth_cadence(monkeypatch):
    import vfe3.gauge_optim as gauge_optim_mod
    from vfe3.geometry.groups import get_group

    group = get_group("so_k")(K=4)

    def _optimizer(U):
        return GaugeNaturalGradAdamW(
            [{"params": [U], "lr": 0.05, "omega": True, "weight_decay": 0.0}],
            group.generators, group.irrep_dims, gauge_momentum=0.0,
            skew_symmetric=True, omega_reorth_every=3,
        )

    source_U = nn.Parameter(torch.eye(4).expand(2, 4, 4).contiguous())
    source_opt = _optimizer(source_U)
    source_opt.step()
    source_opt.step()
    state = source_opt.state_dict()
    assert state["optimizer_extra"]["omega_step"] == 2

    calls = []
    original_polar = gauge_optim_mod._polar_orthogonalize

    def _spy(U):
        calls.append(1)
        return original_polar(U)

    monkeypatch.setattr(gauge_optim_mod, "_polar_orthogonalize", _spy)
    resumed_U = nn.Parameter(torch.eye(4).expand(2, 4, 4).contiguous())
    resumed_opt = _optimizer(resumed_U)
    resumed_opt.load_state_dict(state)
    resumed_opt.step()
    assert resumed_opt._omega_step == 3
    assert calls == [1]

    legacy = {k: v for k, v in state.items() if k != "optimizer_extra"}
    legacy_U = nn.Parameter(torch.eye(4).expand(2, 4, 4).contiguous())
    legacy_opt = _optimizer(legacy_U)
    with pytest.warns(UserWarning, match="non-exact resume"):
        legacy_opt.load_state_dict(legacy)
    assert legacy_opt._omega_step == 0
