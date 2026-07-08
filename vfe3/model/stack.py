r"""The VFE block stack for VFE_3.0: L blocks with the belief handoff mu_q -> mu_p.

After each block the updated belief becomes (a blend toward) the next block's prior:
mu_p_next = (1 - rho) mu_p + rho mu_q (rho = prior_handoff_rho); sigma_p frozen at the
embedding by default; phi flows through the belief, not the prior.

Placement note (audit 2026-06-09 overnight F23): the opt-in head_mixer / cg_coupling
transforms run INSIDE each block (after its E-step, before its norm), so the belief
handed off above is the POST-transform belief — at n_layers > 1 the transforms recurse
into every subsequent block's prior. The manuscript places the mixer in the single
W_O-readout slot (Manuscripts-Theory/GL(K)_attention.tex) and concedes genuine
cross-head capacity at depth > 1, but does not state this per-block prior-handoff
recursion; the pre-transform converged belief stays available via the ``capture``
out-param (the M-step self-coupling reads it from there).
"""

from typing import Callable, Optional

import torch

from vfe3.belief import BeliefState
from vfe3.config import VFE3Config
from vfe3.geometry.groups import GaugeGroup
from vfe3.free_energy import attention_tau, query_adaptive_tau
from vfe3.model.block import _as_coeff, vfe_block


def vfe_stack(
    belief:     BeliefState,
    mu_p:       torch.Tensor,             # (N, K) initial prior means
    sigma_p:    torch.Tensor,             # (N, K) initial prior variances
    group:      GaugeGroup,
    cfg:        VFE3Config,

    *,
    log_prior:       Optional[torch.Tensor]    = None,
    block_norm:      Optional[Callable[..., torch.Tensor]] = None,   # cached norm instance (None -> off)
    head_mixer:      Optional[Callable[..., 'tuple']]      = None,   # opt-in Schur head mixer (None -> off)
    cg_coupling:     Optional[Callable[..., 'tuple']]      = None,   # opt-in CG cross-type coupling (None -> off)
    lambda_beta:     'float | torch.Tensor'    = 1.0,    # belief-coupling weight (cfg.lambda_beta)
    connection_W:    Optional[torch.Tensor]    = None,   # learned bilinear connection for regime_ii (NN exception; None -> pure path)
    connection_M:    Optional[torch.Tensor]    = None,   # learned covariant connection for regime_ii_covariant (Route B; None -> pure path)
    connection_L:    Optional[torch.Tensor]    = None,   # learned direct link for regime_ii_link* (NN exception; None -> pure path)
    e_step_gradient: str                       = "unroll",  # E-step backward estimator (unroll | straight_through | detach)
    rope:            Optional[torch.Tensor]    = None,   # (N, K, K) gauge-RoPE rotation (None -> off)
    rope_on_cov:     bool                      = False,  # full-gauge: rotate covariance too
    rope_on_value:   bool                      = True,   # False -> value aggregation uses the un-rotated base
    capture:         Optional[dict]            = None,   # out-param: LAST block's converged (pre-transform) belief under 'converged'
    grad_record:     Optional[dict]            = None,   # diag out-param: LAST block's E-step belief-grad norms (None -> no capture)
    prebuilt_transport: Optional[object]       = None,   # share_refine_s_transport: one flat transport shared across blocks (valid: e_phi_lr==0 + flat, phi loop-invariant)
    gauge_parameterization: str                = "phi",  # 'phi' (exp(phi.G) path) | 'omega_direct' (stored GL(K) element, read from belief.omega)
    kappa_beta_override: 'Optional[float | torch.Tensor]' = None,   # learnable_kappa_beta: live exp(log_kappa_beta) (t5-exception family); None -> cfg.kappa_beta
) -> BeliefState:
    r"""Run L = cfg.n_layers blocks, handing the belief mean off to the next prior.

    ``connection_W`` is the model's learned bilinear Regime-II connection (a sanctioned NN exception) forwarded under
    transport_mode='regime_ii'; None on the pure (flat) path. ``e_step_gradient`` is the E-step
    backward estimator forwarded to the E-step ('unroll' default keeps the second-order trajectory
    gradient, 'straight_through' detaches the per-iteration tangent; both share the forward value).
    'detach' is handled by the caller's no_grad wrapper, so here it behaves like 'unroll'.
    ``rope`` is the precomputed block-diagonal positional rotation R(theta) (None = off, the pure
    path); ``rope_on_cov`` enables the full-gauge covariance sandwich rotation."""
    rho = cfg.prior_handoff_rho
    rho_s = cfg.prior_handoff_sigma
    # Hoist the loop-invariant temperature computation out of the per-layer vfe_block call.
    # attention_tau depends only on the kappa, group.irrep_dims, and device -- all constant across
    # layers -- so computing it once here and passing it as tau avoids L redundant calls.
    # kappa_beta_override carries the model's LEARNED exp(log_kappa_beta) under
    # learnable_kappa_beta (un-detached, so the loss backpropagates to it); None is the pure
    # config path, byte-identical to before.
    tau = attention_tau(
        kappa_beta_override if kappa_beta_override is not None
        else _as_coeff(cfg.kappa_beta, belief.mu.device),
        group.irrep_dims)
    for _ in range(cfg.n_layers):
        # Per-query adaptive temperature (cfg.query_adaptive_tau, default OFF): rescale the hoisted
        # per-head tau by the DETACHED uncertainty trace of the belief ENTERING this block,
        # tau_i,h = tau_h (1 + c tr_h(Sigma_i)/d_h) -- the query-side dual of the detached precision
        # key bias, and like that bias it is held fixed across the block's inner E-step iterations
        # (recomputed per BLOCK from the current belief, not per inner iteration). OFF path: tau_b
        # IS the hoisted scalar/(H,) tau, byte-identical.
        if cfg.query_adaptive_tau:
            sig   = belief.sigma if belief.sigma.dim() == belief.mu.dim() \
                else belief.sigma.diagonal(dim1=-2, dim2=-1)     # full cov -> per-coordinate variances
            tau_b = query_adaptive_tau(sig, tau, group.irrep_dims, c=cfg.query_tau_c)
        else:
            tau_b = tau
        belief = vfe_block(belief, mu_p, sigma_p, group, cfg, log_prior=log_prior,
                           block_norm=block_norm, head_mixer=head_mixer, cg_coupling=cg_coupling,
                           lambda_beta=lambda_beta,
                           connection_W=connection_W, connection_M=connection_M,
                           connection_L=connection_L,
                           e_step_gradient=e_step_gradient, rope=rope, rope_on_cov=rope_on_cov,
                           rope_on_value=rope_on_value, tau=tau_b,
                           capture=capture, grad_record=grad_record,   # each block overwrites; last wins
                           prebuilt_transport=prebuilt_transport,      # phi is loop-invariant when e_phi_lr==0
                           gauge_parameterization=gauge_parameterization)
        mu_p = (1.0 - rho) * mu_p + rho * belief.mu
        sigma_p = (1.0 - rho_s) * sigma_p + rho_s * belief.sigma
    return belief
