r"""A single VFE block for VFE_3.0: E-step belief inference + optional norm.

Parameter-free: all learnable capacity is the PriorBank's prior tables; the block
runs the iterative E-step (Phase 6) and an optional gauge-equivariant norm on the
mean. The belief handoff across blocks lives in stack.py.
"""

from typing import Any, Optional

import torch

from vfe3.belief import BeliefState
from vfe3.config import VFE3Config
from vfe3.geometry.groups import GaugeGroup
from vfe3.inference.e_step import e_step


def vfe_block(
    belief:     BeliefState,
    mu_p:       torch.Tensor,             # (N, K) prior means
    sigma_p:    torch.Tensor,             # (N, K) prior variances
    group:      GaugeGroup,
    cfg:        VFE3Config,

    *,
    log_prior:  Optional[torch.Tensor] = None,
    block_norm: Optional[Any]          = None,   # cached norm instance (None -> no block norm)
) -> BeliefState:
    r"""Run n_e_steps of the E-step from ``belief`` toward the prior, then optional norm."""
    out = e_step(
        belief, mu_p, sigma_p, group,
        n_iter=cfg.n_e_steps, tau=cfg.tau,
        e_mu_lr=cfg.e_mu_lr, e_sigma_lr=cfg.e_sigma_lr, e_phi_lr=cfg.e_phi_lr,
        alpha_div=cfg.alpha_div, value=cfg.alpha, b0=cfg.b0, c0=cfg.c0,
        kl_max=cfg.kl_max, eps=cfg.eps,
        sigma_max=cfg.sigma_max, e_sigma_q_trust=cfg.e_sigma_q_trust, mass_phi=cfg.mass_phi,
        include_attention_entropy=cfg.include_attention_entropy,
        gradient_mode=cfg.gradient_mode, family=cfg.family, divergence_family=cfg.divergence_family,
        alpha_mode=cfg.alpha_mode,
        phi_precond_mode=cfg.phi_precond_mode, phi_retract_mode=cfg.phi_retract_mode,
        spd_retract_mode=cfg.spd_retract_mode, transport_mode=cfg.transport_mode,
        log_prior=log_prior,
    )
    if block_norm is not None:               # cached parameter-free norm (audit 2d/4f)
        out = BeliefState(mu=block_norm(out.mu, out.sigma), sigma=out.sigma, phi=out.phi)
    return out
