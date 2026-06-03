r"""A single VFE block for VFE_3.0: E-step belief inference + optional norm.

Parameter-free: all learnable capacity is the PriorBank's prior tables; the block
runs the iterative E-step (Phase 6) and an optional gauge-equivariant norm on the
mean. The belief handoff across blocks lives in stack.py.
"""

from typing import Callable, Optional

import torch

from vfe3.belief import BeliefState
from vfe3.config import VFE3Config
from vfe3.geometry.groups import GaugeGroup
from vfe3.free_energy import attention_tau
from vfe3.inference.e_step import e_step


def vfe_block(
    belief:     BeliefState,
    mu_p:       torch.Tensor,             # (N, K) prior means
    sigma_p:    torch.Tensor,             # (N, K) prior variances
    group:      GaugeGroup,
    cfg:        VFE3Config,

    *,
    log_prior:       Optional[torch.Tensor]    = None,
    block_norm:      Optional[Callable[..., torch.Tensor]] = None,   # cached norm instance (None -> off)
    log_alpha:       Optional[torch.Tensor]    = None,   # learned scalar self-coupling (None -> pure path)
    connection_W:    Optional[torch.Tensor]    = None,   # learned bilinear connection for regime_ii (NN exception; None -> pure path)
    e_step_gradient: str                       = "unroll",  # E-step backward estimator (unroll | straight_through | detach)
    rope:            Optional[torch.Tensor]    = None,   # (N, K, K) gauge-RoPE rotation (None -> off)
    rope_on_cov:     bool                      = False,  # full-gauge: rotate covariance too
) -> BeliefState:
    r"""Run n_e_steps of the E-step from ``belief`` toward the prior, then optional norm.

    ``log_alpha`` is the model's learned self-coupling nn.Parameter (alpha = exp(log_alpha))
    under alpha_mode='learnable', forwarded to the E-step; None on the pure path. ``connection_W``
    is the model's learned bilinear Regime-II connection (a sanctioned NN exception) forwarded under
    transport_mode='regime_ii'; None on the pure (flat) path. ``e_step_gradient`` is the E-step
    backward estimator forwarded to the E-step (unroll | straight_through | detach). ``rope`` is
    the precomputed block-diagonal positional rotation R(theta) (None = off, the pure path);
    ``rope_on_cov`` enables the full-gauge covariance sandwich rotation."""
    out = e_step(
        belief, mu_p, sigma_p, group,
        n_iter=cfg.n_e_steps, tau=attention_tau(cfg.kappa, group.irrep_dims),
        e_mu_lr=cfg.e_mu_lr, e_sigma_lr=cfg.e_sigma_lr, e_phi_lr=cfg.e_phi_lr,
        alpha_div=cfg.alpha_div, value=cfg.alpha, b0=cfg.b0, c0=cfg.c0, log_alpha=log_alpha,
        kl_max=cfg.kl_max, eps=cfg.eps,
        sigma_max=cfg.sigma_max, e_sigma_q_trust=cfg.e_sigma_q_trust, mass_phi=cfg.mass_phi,
        include_attention_entropy=cfg.include_attention_entropy,
        gradient_mode=cfg.gradient_mode, family=cfg.family, divergence_family=cfg.divergence_family,
        alpha_mode=cfg.alpha_mode,
        phi_precond_mode=cfg.phi_precond_mode, phi_retract_mode=cfg.phi_retract_mode,
        spd_retract_mode=cfg.spd_retract_mode, transport_mode=cfg.transport_mode,
        cocycle_relaxation=cfg.cocycle_relaxation, connection_W=connection_W,
        e_step_gradient=e_step_gradient, oracle_unroll_grad=cfg.oracle_unroll_grad,
        log_prior=log_prior,
        rope=rope, rope_on_cov=rope_on_cov,
    )
    if block_norm is not None:               # cached parameter-free norm (audit 2d/4f)
        out = BeliefState(mu=block_norm(out.mu, out.sigma), sigma=out.sigma, phi=out.phi)
    return out
