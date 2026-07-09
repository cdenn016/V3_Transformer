r"""Curated regression tests for truncated E-step oracle backpropagation."""

import torch

from vfe3.belief import BeliefState
from vfe3.geometry.groups import GaugeGroup, get_group
from vfe3.inference.e_step import build_belief_transport, e_step


def _truncation_case(
    seed: int = 0,
) -> tuple[BeliefState, torch.Tensor, torch.Tensor, GaugeGroup]:
    generator = torch.Generator().manual_seed(seed)
    n_tokens, width = 3, 2
    group = get_group("glk")(width)
    n_gen = group.generators.shape[0]
    belief = BeliefState(
        mu=torch.randn(n_tokens, width, generator=generator),
        sigma=torch.rand(n_tokens, width, generator=generator) + 0.75,
        phi=0.12 * torch.randn(n_tokens, n_gen, generator=generator),
    )
    mu_p = torch.randn(n_tokens, width, generator=generator)
    sigma_p = torch.rand(n_tokens, width, generator=generator) + 0.75
    return belief, mu_p, sigma_p, group


def test_shared_prebuilt_transport_respects_truncation_boundary() -> None:
    belief, mu_p, sigma_p, group = _truncation_case(seed=31)
    source_phi = belief.phi.detach().clone().requires_grad_(True)
    belief = belief._replace(phi=source_phi)
    prebuilt_transport = build_belief_transport(source_phi, group)

    out = e_step(
        belief,
        mu_p,
        sigma_p,
        group,
        n_iter=3,
        e_phi_lr=0.0,
        e_steps_backprop_last=1,
        prebuilt_transport=prebuilt_transport,
    )
    loss = out.mu.square().sum()
    source_grad, = torch.autograd.grad(loss, source_phi, allow_unused=True, retain_graph=True)

    assert source_grad is None or torch.count_nonzero(source_grad) == 0
    assert out.phi.is_leaf and out.phi.requires_grad
    boundary_grad, = torch.autograd.grad(loss, out.phi)
    assert torch.isfinite(boundary_grad).all()
    assert torch.count_nonzero(boundary_grad) > 0


def test_oracle_last_k_restores_prior_gradient() -> None:
    belief, mu_p, sigma_p, group = _truncation_case(seed=37)
    mu_p = mu_p.requires_grad_(True)

    out = e_step(
        belief,
        mu_p,
        sigma_p,
        group,
        n_iter=3,
        e_phi_lr=0.0,
        e_steps_backprop_last=1,
        renyi_order=0.5,
        oracle_unroll_grad=True,
    )

    assert out.mu.requires_grad and out.mu.grad_fn is not None
    prior_grad, = torch.autograd.grad(out.mu.square().sum(), mu_p)
    assert torch.isfinite(prior_grad).all()
    assert torch.count_nonzero(prior_grad) > 0


def test_backprop_last_equal_total_matches_full_unroll() -> None:
    base, mu_p_value, sigma_p, group = _truncation_case(seed=41)

    def run(backprop_last: int) -> tuple[BeliefState, torch.Tensor]:
        belief = base._replace(
            mu=base.mu.detach().clone().requires_grad_(True),
            sigma=base.sigma.detach().clone().requires_grad_(True),
            phi=base.phi.detach().clone().requires_grad_(True),
        )
        mu_p = mu_p_value.detach().clone().requires_grad_(True)
        out = e_step(
            belief,
            mu_p,
            sigma_p,
            group,
            n_iter=3,
            e_phi_lr=0.0,
            e_steps_backprop_last=backprop_last,
            renyi_order=0.5,
            oracle_unroll_grad=True,
        )
        prior_grad, = torch.autograd.grad(out.mu.square().sum(), mu_p)
        return out, prior_grad

    full, full_prior_grad = run(0)
    equal_total, equal_total_prior_grad = run(3)

    assert torch.equal(equal_total.mu, full.mu)
    assert torch.equal(equal_total.sigma, full.sigma)
    assert torch.equal(equal_total.phi, full.phi)
    assert torch.equal(equal_total_prior_grad, full_prior_grad)
