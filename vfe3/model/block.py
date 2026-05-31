r"""A single VFE block for VFE_3.0: E-step belief inference + optional norm.

Parameter-free: all learnable capacity is the PriorBank's prior tables; the block
runs the iterative E-step (Phase 6) and an optional gauge-equivariant norm on the
mean. The belief handoff across blocks lives in stack.py.
"""

from typing import Optional

import torch

from vfe3.belief import BeliefState
from vfe3.config import VFE3Config
from vfe3.geometry.groups import GaugeGroup
from vfe3.geometry.norms import get_norm
from vfe3.inference.e_step import e_step


def vfe_block(
    belief:    BeliefState,
    mu_p:      torch.Tensor,             # (N, K) prior means
    sigma_p:   torch.Tensor,             # (N, K) prior variances
    group:     GaugeGroup,
    cfg:       VFE3Config,

    *,
    log_prior: Optional[torch.Tensor] = None,
) -> BeliefState:
    r"""Run n_e_steps of the E-step from ``belief`` toward the prior, then optional norm."""
    out = e_step(
        belief, mu_p, sigma_p, group,
        n_iter=cfg.n_e_steps, tau=cfg.tau,
        e_mu_lr=cfg.e_mu_lr, e_sigma_lr=cfg.e_sigma_lr, e_phi_lr=cfg.e_phi_lr,
        alpha_div=cfg.alpha_div, value=cfg.alpha, b0=cfg.b0, c0=cfg.c0,
        kl_max=cfg.kl_max, eps=cfg.eps,
        sigma_max=cfg.sigma_max, e_sigma_q_trust=cfg.e_sigma_q_trust,
        include_attention_entropy=cfg.include_attention_entropy,
        gradient_mode=cfg.gradient_mode, family=cfg.family, alpha_mode=cfg.alpha_mode,
        phi_precond_mode=cfg.phi_precond_mode, phi_retract_mode=cfg.phi_retract_mode,
        log_prior=log_prior,
    )
    if cfg.norm_type_block != "none":
        norm = get_norm(cfg.norm_type_block)(cfg.embed_dim, eps=cfg.eps)
        out = BeliefState(mu=norm(out.mu, out.sigma), sigma=out.sigma, phi=out.phi)
    return out
