r"""A single VFE block for VFE_3.0: E-step belief inference + optional norm.

Parameter-free (the block owns no nn.Parameters): on the pure default path the learnable capacity
is the PriorBank's prior tables, but the block also receives the opt-in learned exceptions
(head mixer, CG coupling, connection_W) as arguments and applies them; the block
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


def _as_coeff(v: 'float | list', device: torch.device) -> 'float | torch.Tensor':
    r"""Pass a scalar b0/c0 through unchanged; turn a list into a (K,) float32 tensor on device."""
    return torch.as_tensor(v, dtype=torch.float32, device=device) if isinstance(v, (list, tuple)) else v


def vfe_block(
    belief:     BeliefState,
    mu_p:       torch.Tensor,             # (N, K) prior means
    sigma_p:    torch.Tensor,             # (N, K) prior variances
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
    tau:             'Optional[float | torch.Tensor]' = None,  # softmax temperature (precomputed by vfe_stack; None -> compute here)
    capture:         Optional[dict]            = None,   # out-param: stashes the CONVERGED (pre-transform) belief under 'converged'
    grad_record:     Optional[dict]            = None,   # diag out-param: E-step belief-grad norms (None -> no capture)
    state_record:    Optional[dict]            = None,   # diag out-param: E-step belief/F trace (None -> no capture)
    prebuilt_transport: Optional[object]       = None,   # share_refine_s_transport: caller-built flat transport (None -> e_step builds its own)
    gauge_parameterization: str                = "phi",  # 'phi' (exp(phi.G) path) | 'omega_direct' (stored GL(K) element, read from belief.omega)
) -> BeliefState:
    r"""Run n_e_steps of the E-step from ``belief`` toward the prior, then optional norm.

    ``lambda_beta`` is the belief-coupling weight (the constant cfg.lambda_beta); 1.0 is the pure F.
    ``connection_W`` is the model's learned
    bilinear Regime-II connection (a sanctioned NN exception) forwarded under
    transport_mode='regime_ii'; None on the pure (flat) path. ``e_step_gradient`` is the E-step
    backward estimator forwarded to the E-step (unroll | straight_through | detach). ``rope`` is
    the precomputed block-diagonal positional rotation R(theta) (None = off, the pure path);
    ``rope_on_cov`` enables the full-gauge covariance sandwich rotation. ``tau`` is the softmax
    temperature (kappa * sqrt(dim_h)); when None it is computed here from cfg.kappa_beta and
    group.irrep_dims. Pass a precomputed value from vfe_stack to avoid recomputing each layer."""
    if tau is None:
        tau = attention_tau(_as_coeff(cfg.kappa_beta, belief.mu.device), group.irrep_dims)
    out = e_step(
        belief, mu_p, sigma_p, group,
        n_iter=cfg.n_e_steps, tau=tau,
        e_q_mu_lr=cfg.e_q_mu_lr, e_q_sigma_lr=cfg.e_q_sigma_lr, e_phi_lr=cfg.e_phi_lr,
        renyi_order=cfg.renyi_order, value=cfg.lambda_alpha, b0=_as_coeff(cfg.b0, belief.mu.device), c0=_as_coeff(cfg.c0, belief.mu.device),
        lambda_beta=lambda_beta,
        kl_max=cfg.kl_max, eps=cfg.eps,
        sigma_max=cfg.sigma_max, e_sigma_q_trust=cfg.e_sigma_q_trust, mass_phi=cfg.mass_phi,
        e_mu_q_trust=cfg.e_mu_q_trust, mu_trust_mode=cfg.mu_trust_mode,
        e_step_mu_precond=cfg.e_step_mu_precond,
        include_attention_entropy=cfg.include_attention_entropy,
        gradient_mode=cfg.gradient_mode, family=cfg.family, divergence_family=cfg.divergence_family,
        lambda_alpha_mode=cfg.lambda_alpha_mode,
        phi_precond_mode=cfg.phi_precond_mode, phi_retract_mode=cfg.phi_retract_mode,
        spd_retract_mode=cfg.spd_retract_mode, transport_mode=cfg.transport_mode,
        cocycle_relaxation=cfg.cocycle_relaxation, connection_W=connection_W, connection_M=connection_M,
        connection_L=connection_L, link_alpha=cfg.link_alpha, link_soft_cap=cfg.link_soft_cap,
        clamp_monitor=cfg.transport_clamp_monitor,
        e_step_gradient=e_step_gradient, oracle_unroll_grad=cfg.oracle_unroll_grad,
        grad_record=grad_record, state_record=state_record,
        log_prior=log_prior,
        rope=rope, rope_on_cov=rope_on_cov, rope_on_value=rope_on_value,
        # Tier-1/Tier-2 toggles (2026-07-05; all default OFF/byte-identical). The iteration-only
        # knobs (e_step_update/mm_damping/lambda_twohop/skip_belief_sigma_update/
        # compile_pair_kernel) ride e_step's **kwargs into e_step_iteration AND the diagnostic
        # free_energy_value (which accepts-and-ignores the iteration-only ones, and HONORS
        # lambda_twohop in the logged F); the loop-control knobs bind explicitly on e_step.
        # skip_belief_sigma_update is a BELIEF-channel toggle: _refine_s never passes it, so the
        # s-channel sigma update is untouched.
        e_step_update=cfg.e_step_update, mm_damping=cfg.mm_damping,
        lambda_twohop=cfg.lambda_twohop,
        skip_belief_sigma_update=cfg.skip_belief_sigma_update,
        compile_pair_kernel=cfg.compile_pair_kernel,
        transport_mean_per_head=cfg.transport_mean_per_head,
        exp_fp64_mode=cfg.exp_fp64_mode,
        exp_fp64_norm_threshold=cfg.exp_fp64_norm_threshold,
        randomize_e_steps=cfg.randomize_e_steps,
        e_steps_min=cfg.e_steps_min, e_steps_max=cfg.e_steps_max,
        e_steps_backprop_last=cfg.e_steps_backprop_last,
        e_step_halt_tol=cfg.e_step_halt_tol,
        prebuilt_transport=prebuilt_transport,
        gauge_parameterization=gauge_parameterization,
    )
    if capture is not None:
        # The CONVERGED variational belief q* -- what the E-step's F was minimized over,
        # BEFORE the post-inference transforms below. The manuscript's self-coupling
        # alpha*KL(q||p) is pinned to THIS belief (audit 2026-06-09 overnight F19); the
        # M-step regularizer and the diagnostics F self-term read it from here. With the
        # transforms off (the pure path) it is the SAME object the block returns.
        capture["converged"] = out
    if head_mixer is not None:               # opt-in head mixing: after the E-step, BEFORE the norm
        mu_mixed, sigma_mixed = head_mixer(out.mu, out.sigma)   # so the mixed belief feeds norm + handoff
        out = out._replace(mu=mu_mixed, sigma=sigma_mixed)   # (per-block apply order); _replace preserves phi/omega/reflection
    if cg_coupling is not None:              # opt-in CG cross-type coupling: after mixing, before norm
        mu_cg, sigma_cg = cg_coupling(out.mu, out.sigma)
        out = out._replace(mu=mu_cg, sigma=sigma_cg)   # _replace preserves phi/omega/reflection
    if block_norm is not None:               # cached parameter-free norm (audit 2d/4f)
        out = out._replace(mu=block_norm(out.mu, out.sigma))   # _replace preserves phi/omega/reflection
    return out
