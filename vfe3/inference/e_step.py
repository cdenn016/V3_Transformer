r"""The E-step for VFE_3.0: an iterative natural-gradient descent on F over the
Gaussian belief (mu, sigma, phi).

One inner iteration (all positions in parallel, updates sequential):
  transport Omega(phi) -> belief_gradients (envelope kernel / oracle) -> Fisher
  preconditioner -> retract mu (Euclidean) + sigma (SPD) -> phi (autograd of the
  canonical belief-coupling block -> precondition -> Lie retraction).
Decoupled learning rates and trust regions. Parallel mean-field updates are not
guaranteed monotone per iteration; F-descent holds as a DIRECTION property
(filtering descends F_filt; smoothing and the phi step descend global F).
"""

from typing import List, Optional, Tuple

import torch

from vfe3.alpha_i import self_coupling_alpha
from vfe3.belief import BeliefState
from vfe3.families.base import get_family
from vfe3.free_energy import attention_weights, free_energy, pairwise_energy, reduced_free_energy, self_divergence_for_alpha
from vfe3.geometry.groups import GaugeGroup
from vfe3.geometry.phi_preconditioner import precondition_phi_gradient
from vfe3.geometry.retraction import get_retraction, retract_phi
from vfe3.numerics import apply_mu_trust_region
from vfe3.geometry.transport import (
    _TRANSPORT_BATCH_INDEPENDENT,
    _TRANSPORT_NEEDS_MU,
    _TRANSPORT_NEEDS_SIGMA,
    FactoredTransport,
    RopeTransport,
    build_factored_transport,
    compute_transport_operators,
    get_transport,
    transport_covariance,
    transport_mean,
)
from vfe3.gradients.kernels import belief_gradients, uses_kernel_route


def _transport(
    phi:                torch.Tensor,             # (N, n_gen) or (B, N, n_gen)
    group:              GaugeGroup,

    *,
    transport_mode:     str                    = "flat",   # connection-regime registry key (default = flat)
    gauge_mode:         str                    = "learned", # 'learned' | 'trivial' (forwarded to the builder)
    cocycle_relaxation: float                  = 1.0,       # regime_ii homotopy alpha; 0 -> flat
    mu:                 Optional[torch.Tensor] = None,      # (N, K) or (B, N, K) means; regime_ii edge connection reads these
    sigma:              Optional[torch.Tensor] = None,      # variances; regime_ii_covariant features read these
    link_alpha:         float                  = 1.0,       # direct-link scale (regime_ii_link / _charted)
    link_soft_cap:      float                  = 6.0,       # direct-link embedded-Frobenius soft cap
    clamp_monitor:      bool                   = False,     # opt-in: warn when the exp Frobenius clamp fires (host sync)
    connection_W:       Optional[torch.Tensor] = None,      # (n_gen, K, K) learned bilinear connection (regime_ii, NN exception)
    connection_M:       Optional[torch.Tensor] = None,      # (n_gen, 3) learned covariant connection (regime_ii_covariant, NN exception)
    connection_L:       Optional[torch.Tensor] = None,      # (max_seq, max_seq, n_gen) learned direct link (regime_ii_link*, NN exception)
    mu_key:             Optional[torch.Tensor] = None,      # regime_ii KEY-slot means (None -> mu; oracle detach split)
    sigma_key:          Optional[torch.Tensor] = None,      # regime_ii_covariant KEY-slot variances (None -> sigma)
) -> torch.Tensor:                            # (N, N, K, K) or (B, N, N, K, K) Omega_ij
    r"""Build the pairwise transport Omega_ij via the connection-regime registry.

    The build is config-selected through ``get_transport(transport_mode)``; the default 'flat' is
    the Regime-I phi-cocycle Omega_ij = exp(phi_i) exp(-phi_j) (byte-identical to a direct
    ``compute_transport_operators`` call, mu/connection_W ignored). 'regime_ii' is the NON-FLAT
    edge-relaxed cocycle (a sanctioned learned-connection NN exception): it reads the CURRENT belief
    means ``mu`` and the learned ``connection_W`` to insert the edge factor exp(delta_ij . G), so it
    must be REBUILT as mu updates each E-step iteration (flat is mu-independent). ``gauge_mode`` is
    forwarded to the builder (audit 2026-06-10 F6: previously dropped, so the builders always saw
    the 'learned' default).

    Rank-aware: a 2-D (N, n_gen) frame (the unbatched diagnostics / trajectory path) is transported
    as a batch of one and stripped back to (N, N, K, K); a 3-D (B, N, n_gen) frame (the batched
    forward) flows straight through. ``mu``/``mu_key`` are unsqueezed to match so the builder always
    sees batched (B, N, K) means."""
    build = get_transport(transport_mode)
    batch_independent = transport_mode in _TRANSPORT_BATCH_INDEPENDENT
    if phi.dim() == 2:
        mu_b = mu.unsqueeze(0) if mu is not None else None
        mu_kb = mu_key.unsqueeze(0) if mu_key is not None else None
        sig_b = sigma.unsqueeze(0) if sigma is not None else None
        sig_kb = sigma_key.unsqueeze(0) if sigma_key is not None else None
        omega = build(phi.unsqueeze(0), group, gauge_mode=gauge_mode, mu=mu_b, mu_key=mu_kb,
                      sigma=sig_b, sigma_key=sig_kb,
                      connection_W=connection_W, connection_M=connection_M, connection_L=connection_L,
                      link_alpha=link_alpha, link_soft_cap=link_soft_cap, clamp_monitor=clamp_monitor,
                      cocycle_relaxation=cocycle_relaxation)["Omega"]
        # A batch-independent builder (regime_ii_link) already returns (N,N,K,K); ordinary builders
        # return (1,N,N,K,K) on the unbatched diagnostics path -> strip the batch-of-one.
        return omega if batch_independent else omega[0]
    return build(phi, group, gauge_mode=gauge_mode, mu=mu, mu_key=mu_key,
                 sigma=sigma, sigma_key=sigma_key,
                 connection_W=connection_W, connection_M=connection_M, connection_L=connection_L,
                 link_alpha=link_alpha, link_soft_cap=link_soft_cap, clamp_monitor=clamp_monitor,
                 cocycle_relaxation=cocycle_relaxation)["Omega"]


def _can_fuse_flat(transport_mode: str, group: GaugeGroup) -> bool:
    r"""Whether the forward belief-transport may skip the dense (B,N,N,K,K) Omega (P0 #2).

    The fused factored route is valid ONLY when the connection is flat (``transport_mode='flat'``;
    regime_ii's Omega is mu-dependent and carries the edge delta factor, so it must stay dense) AND
    the group is genuinely block-diagonal with EQUAL blocks (``len(irrep_dims) > 1`` and all blocks
    the same size -- block_glk / tied_block_glk). Single-block (glk, so_k, cross-coupled block_glk
    all report ``irrep_dims=[K]``) keeps the dense path.
    """
    return (
        transport_mode == "flat"
        and len(group.irrep_dims) > 1
        and len(set(group.irrep_dims)) == 1
    )


def build_belief_transport(
    phi:                torch.Tensor,             # (B, N, n_gen) batched gauge frames
    group:              GaugeGroup,

    *,
    transport_mode:     str                    = "flat",   # connection-regime registry key
    gauge_mode:         str                    = "learned", # 'learned' | 'trivial' (forwarded to the builder)
    cocycle_relaxation: float                  = 1.0,       # regime_ii homotopy alpha; 0 -> flat
    link_alpha:         float                  = 1.0,       # direct-link scale (regime_ii_link / _charted)
    link_soft_cap:      float                  = 6.0,       # direct-link embedded-Frobenius soft cap
    clamp_monitor:      bool                   = False,     # opt-in: warn when the exp Frobenius clamp fires (host sync)
    rope_on_cov:        bool                   = False,     # rotate the covariance too (full-gauge)
    rope_on_value:      bool                   = True,      # False -> value aggregation uses the un-rotated base
    mu:                 Optional[torch.Tensor] = None,      # regime_ii edge connection reads these
    sigma:              Optional[torch.Tensor] = None,      # regime_ii_covariant features read these
    connection_W:       Optional[torch.Tensor] = None,      # regime_ii learned bilinear connection
    connection_M:       Optional[torch.Tensor] = None,      # regime_ii_covariant learned covariant connection (Route B)
    connection_L:       Optional[torch.Tensor] = None,      # regime_ii_link* learned direct link
    mu_key:             Optional[torch.Tensor] = None,      # regime_ii KEY-slot means (None -> mu)
    sigma_key:          Optional[torch.Tensor] = None,      # regime_ii_covariant KEY-slot variances (None -> sigma)
    rope:               Optional[torch.Tensor] = None,      # (N, K, K) gauge-RoPE rotation (None -> off)
) -> 'torch.Tensor | FactoredTransport | RopeTransport':
    r"""Build the FORWARD belief-transport for one E-step iteration (the hot path, P0 #2).

    On the flat + block-diagonal-with-equal-blocks path (``_can_fuse_flat``) returns a
    :class:`FactoredTransport` -- the per-token exps only, NEVER the dense (B,N,N,K,K) Omega --
    which ``transport_mean`` / ``transport_covariance`` consume on a fused fast path (the mean is an
    exact reassociation; the diagonal covariance factors per head). Every other case
    (regime_ii, single-block / cross-coupled groups, trivial gauge) falls back to ``_transport``'s
    DENSE Omega exactly as before, so those numerics are unchanged; that fallback is REGISTRY-driven
    (belief tensors gated on ``_TRANSPORT_NEEDS_MU`` / ``_TRANSPORT_NEEDS_SIGMA`` membership, never
    on literal mode names). The factored container flows
    OPAQUELY through ``belief_gradients`` (kernel + oracle), which only ever forward it to
    ``transport_mean`` / ``transport_covariance``; the oracle's full-cov / smoothing routes rebuild
    the dense Omega from the factors on demand (byte-identical).

    When ``rope`` is given the built transport is wrapped in a :class:`RopeTransport` before being
    returned; ``transport_mean`` / ``transport_covariance`` consume it opaquely, so no downstream
    changes are needed.
    """
    if _can_fuse_flat(transport_mode, group):
        built = build_factored_transport(phi, group, gauge_mode=gauge_mode, clamp_monitor=clamp_monitor)
    else:
        # Registry-driven routing (audit 2026-07-01 round-3, punch 12b): forward ALL state kwargs
        # once, gating the belief tensors on the registry's needs-sets (the add-by-registering
        # contract) instead of matching literal mode names; builders tolerate unused kwargs.
        built = _transport(phi, group, transport_mode=transport_mode, gauge_mode=gauge_mode,
                           mu=(mu if transport_mode in _TRANSPORT_NEEDS_MU else None),
                           mu_key=(mu_key if transport_mode in _TRANSPORT_NEEDS_MU else None),
                           sigma=(sigma if transport_mode in _TRANSPORT_NEEDS_SIGMA else None),
                           sigma_key=(sigma_key if transport_mode in _TRANSPORT_NEEDS_SIGMA else None),
                           connection_W=connection_W, connection_M=connection_M, connection_L=connection_L,
                           link_alpha=link_alpha, link_soft_cap=link_soft_cap, clamp_monitor=clamp_monitor,
                           cocycle_relaxation=cocycle_relaxation)
    if rope is None:
        return built
    return RopeTransport(base=built, rope=rope, on_cov=rope_on_cov, on_value=rope_on_value)


def _transport_qk(
    query_phi: torch.Tensor,         # (N, n_gen) current query frames phi_i
    key_phi:   torch.Tensor,         # (N, n_gen) frozen key frames phi_j
    group:     GaugeGroup,
) -> torch.Tensor:                   # (N, N, K, K) Omega_ij = exp(phi_i^q) exp(-phi_j^k)
    r"""Mixed-frame transport for the FILTERED objective: the query frame phi_i is current
    (belief) and the key frame phi_j is frozen (keys). Reduces to ``_transport`` exactly when
    query_phi == key_phi (the global / keys-None case)."""
    exp_q     = compute_transport_operators(query_phi.unsqueeze(0), group)["exp_phi"][0]      # exp(phi_i^q)
    exp_neg_k = compute_transport_operators(key_phi.unsqueeze(0), group)["exp_neg_phi"][0]    # exp(-phi_j^k)
    return torch.einsum("ikl,jlm->ijkm", exp_q, exp_neg_k)


def free_energy_value(
    belief:                    BeliefState,
    mu_p:                      torch.Tensor,        # (N, K) prior means
    sigma_p:                   torch.Tensor,        # (N, K) prior variances
    group:                     GaugeGroup,

    *,
    tau:                       'float | torch.Tensor' = 1.0,
    renyi_order:               float = 1.0,
    value:                     float = 1.0,
    b0:                        'float | torch.Tensor' = 1.0,   # scalar, or (K,) per-coord for state_dependent_per_coord
    c0:                        'float | torch.Tensor' = 1.0,   # scalar, or (K,) per-coord
    lambda_beta:               'float | torch.Tensor' = 1.0,   # weight on the belief-coupling block (1.0 = pure)
    kl_max:                    float = 100.0,
    eps:                       float = 1e-6,
    sigma_max:                 float = 10.0,           # matches VFE3Config.sigma_max; accepted-and-ignored iteration-only knob
    e_sigma_q_trust:           float = 5.0,            # accepted-and-ignored iteration-only knob
    e_mu_q_trust:              Optional[float] = None, # accepted-and-ignored iteration-only knob
    mu_trust_mode:             str  = "box",           # accepted-and-ignored iteration-only knob
    e_step_mu_precond:         str  = "fisher",        # accepted-and-ignored iteration-only knob
    mass_phi:                  float = 0.0,            # accepted-and-ignored iteration-only knob (phi penalty)

    include_attention_entropy: bool = True,
    rope_on_cov:               bool = False,           # full-gauge: rotate the covariance sandwich too
    rope_on_value:             bool = True,            # False -> value aggregation uses the un-rotated base
    family:                    str  = "gaussian_diagonal",
    divergence_family:         str  = "renyi",
    lambda_alpha_mode:         str  = "constant",
    gradient_mode:             str  = "filtering",     # accepted-and-ignored iteration-only knob
    phi_precond_mode:          str  = "none",          # accepted-and-ignored iteration-only knob
    phi_retract_mode:          str  = "euclidean",     # accepted-and-ignored iteration-only knob
    spd_retract_mode:          str  = "spd_affine",    # accepted-and-ignored iteration-only knob
    transport_mode:            str  = "flat",          # HONORED for global F; raises for frozen keys
    cocycle_relaxation:        float = 1.0,            # HONORED for the global-F transport build (regime_ii)
    link_alpha:                float = 1.0,            # HONORED for the global-F direct-link build (regime_ii_link*)
    link_soft_cap:             float = 6.0,            # HONORED for the global-F direct-link build (regime_ii_link*)
    clamp_monitor:             bool = False,           # HONORED for the global-F transport build (exp clamp diagnostic)

    rope:                      Optional[torch.Tensor] = None,   # (N, K, K) gauge-RoPE rotation (None -> off)
    log_prior:                 Optional[torch.Tensor] = None,
    log_alpha:                 Optional[torch.Tensor] = None,   # learned scalar self-coupling (None -> pure path)
    connection_W:              Optional[torch.Tensor] = None,   # HONORED for the global-F transport build (regime_ii NN exception)
    connection_M:              Optional[torch.Tensor] = None,   # HONORED for the global-F transport build (regime_ii_covariant, Route B)
    connection_L:              Optional[torch.Tensor] = None,   # HONORED for the global-F transport build (regime_ii_link*)
    keys:                      Optional[BeliefState]  = None,   # None -> global F; else keys frozen at `keys`
) -> torch.Tensor:                   # scalar F
    r"""Scalar free energy of a belief. ``keys=None`` -> global F (keys = the belief);
    ``keys`` given -> F with the transported keys frozen at ``keys`` (the F_filt objective).

        F = Sum_i [ alpha_i D(q_i||p_i) (+ R(alpha_i))
                  + Sum_j beta_ij E_ij + tau Sum_j beta_ij log(beta_ij/pi_ij) ],
        E_ij = D(q_i || Omega_ij q_j),  beta = softmax_j(log_prior - E/tau).

    The step-size / E-step-only knobs (gradient_mode, phi_precond_mode, phi_retract_mode,
    spd_retract_mode, sigma_max, e_sigma_q_trust) are declared here as EXPLICIT accept-and-ignore
    parameters (not a blanket ``**kwargs`` sink) so the common ``e_step`` call site may forward one knob
    bag to both this and ``e_step_iteration`` while a MISSPELLED real parameter still raises
    ``TypeError`` here instead of being silently swallowed. ``transport_mode`` / ``connection_W`` /
    ``cocycle_relaxation`` ARE honored for the global (``keys=None``) F so the logged trajectory matches
    the objective the beliefs descend under regime_ii; the filtered (frozen-keys) F has no non-flat
    transport form and raises under a non-flat ``transport_mode``. ``log_alpha`` is also honored (it
    flows into ``self_coupling_alpha`` below). ``rope``/``rope_on_cov`` are honored too (audit
    2026-06-09 PP6): the built transport is wrapped in :class:`RopeTransport` exactly as on the
    model path, so under gauge-RoPE the logged F is the F being descended, not the un-rotated one.
    """
    # keys=None -> global F (query = key = belief). keys given -> filtered F: the transport
    # Omega_ij uses the CURRENT query frame phi_i (belief) and the FROZEN key frame phi_j (keys),
    # and the transported key beliefs come from `keys`; only the key side is frozen.
    key_belief = belief if keys is None else keys
    if keys is None:
        # Build Omega under the ACTIVE connection regime so the logged global F matches the objective
        # the beliefs actually descend (regime_ii reads the means + learned connection_W; flat ignores
        # both). Previously this always used flat transport, so under regime_ii the trajectory was a
        # flat-transport diagnostic, not the regime_ii objective.
        omega = _transport(
            belief.phi, group, transport_mode=transport_mode,
            mu=(belief.mu if transport_mode in _TRANSPORT_NEEDS_MU else None),
            sigma=(belief.sigma if transport_mode in _TRANSPORT_NEEDS_SIGMA else None),
            connection_W=connection_W, connection_M=connection_M, connection_L=connection_L,
            link_alpha=link_alpha, link_soft_cap=link_soft_cap, clamp_monitor=clamp_monitor,
            cocycle_relaxation=cocycle_relaxation,
        )
    else:
        # The filtered (mixed current-query / frozen-key frame) transport has no regime_ii form;
        # reject a non-flat mode rather than silently logging a flat-transport filtered F.
        if transport_mode != "flat":
            raise NotImplementedError(
                f"free_energy_value with frozen keys is flat-only; got transport_mode="
                f"{transport_mode!r} (the filtered diagnostic has no non-flat transport form)"
            )
        omega = _transport_qk(belief.phi, keys.phi, group)
    mu_tv = sigma_tv = None
    if rope is not None:
        # Mirror the model path: R_i Omega_ij R_j^T on the means (and the covariance sandwich
        # under the full gauge). The Rope einsums are rank-agnostic, so no unsqueeze dance.
        wrapped = RopeTransport(base=omega, rope=rope, on_cov=rope_on_cov, on_value=rope_on_value)
        mu_t = transport_mean(wrapped, key_belief.mu)
        sigma_t = transport_covariance(wrapped, key_belief.sigma)
        if not rope_on_value:
            # Diagnostic fidelity: log the DECOUPLED F the beliefs actually descend -- beta from the
            # rotated SCORE energy (mu_t/sigma_t), coupling sum from the UN-rotated base VALUE energy.
            mu_tv = transport_mean(wrapped.base, key_belief.mu)
            sigma_tv = transport_covariance(wrapped.base, key_belief.sigma)
    else:
        mu_t = transport_mean(omega.unsqueeze(0), key_belief.mu.unsqueeze(0))[0]
        sigma_t = transport_covariance(omega.unsqueeze(0), key_belief.sigma.unsqueeze(0))[0]

    fam = get_family(family)
    sd = self_divergence_for_alpha(fam(belief.mu, belief.sigma), fam(mu_p, sigma_p), alpha=renyi_order, kl_max=kl_max,
                                   eps=eps, divergence_family=divergence_family, lambda_alpha_mode=lambda_alpha_mode)
    alpha, reg = self_coupling_alpha(sd, value=value, mode=lambda_alpha_mode, b0=b0, c0=c0, log_alpha=log_alpha)
    energy = pairwise_energy(fam(belief.mu, belief.sigma), fam(mu_t, sigma_t), alpha=renyi_order, kl_max=kl_max, eps=eps,
                             divergence_family=divergence_family, irrep_dims=group.irrep_dims)
    coupling_energy = None
    if mu_tv is not None:
        coupling_energy = pairwise_energy(fam(belief.mu, belief.sigma), fam(mu_tv, sigma_tv), alpha=renyi_order,
                                          kl_max=kl_max, eps=eps, divergence_family=divergence_family,
                                          irrep_dims=group.irrep_dims)
    return free_energy(
        sd, energy, alpha, tau=tau, lambda_beta=lambda_beta,
        include_attention_entropy=include_attention_entropy,
        log_prior=log_prior, alpha_reg=(reg if lambda_alpha_mode != "constant" else None),
        coupling_energy=coupling_energy,
    )


def phi_alignment_loss(
    mu:        torch.Tensor,             # (N, K)
    sigma:     torch.Tensor,             # (N, K)
    phi:       torch.Tensor,             # (N, n_gen) -- the differentiated variable
    group:     GaugeGroup,

    *,
    tau:       'float | torch.Tensor' = 1.0,
    renyi_order: float = 1.0,
    kl_max:    float = 100.0,
    eps:       float = 1e-6,
    mass_phi:  float = 0.0,
    lambda_beta: 'float | torch.Tensor' = 1.0,        # weight on the belief-coupling block (1.0 = pure)
    family:    str   = "gaussian_diagonal",
    divergence_family: str = "renyi",

    include_attention_entropy: bool  = True,
    transport_mode:            str   = "flat",        # connection-regime registry key (default flat)
    cocycle_relaxation:        float = 1.0,           # regime_ii homotopy alpha; 0 -> flat
    link_alpha:                float = 1.0,           # direct-link scale (regime_ii_link / _charted)
    link_soft_cap:             float = 6.0,           # direct-link embedded-Frobenius soft cap
    clamp_monitor:             bool  = False,         # opt-in: warn when the exp Frobenius clamp fires
    rope_on_cov:               bool  = False,         # gauge-RoPE: rotate the covariance sandwich too
    rope_on_value:             bool  = True,          # False -> value aggregation uses the un-rotated base

    rope:         Optional[torch.Tensor] = None,      # (N,K,K) gauge-RoPE rotation (None -> off)
    log_prior:    Optional[torch.Tensor] = None,
    connection_W: Optional[torch.Tensor] = None,      # learned regime_ii connection (held fixed here)
    connection_M: Optional[torch.Tensor] = None,      # learned regime_ii_covariant connection (Route B; held fixed here)
    connection_L: Optional[torch.Tensor] = None,      # learned regime_ii_link* direct link (held fixed here)
) -> torch.Tensor:
    r"""Canonical belief-coupling block as a function of phi (mu, sigma fixed), plus the
    gauge-frame penalty (manuscript Algorithm 1, line for nabla_phi F):

        L(phi) = lambda_beta Sum_ij [ beta_ij E_ij + tau beta_ij log(beta_ij/pi_ij) ] + (mass_phi/2) ||phi||^2,
        E_ij = D(q_i || Omega_ij(phi) q_j),  beta = softmax_j(log_prior - E/tau).
    Both roles of phi flow (Omega_ij depends on phi_i and phi_j); autograd gives the envelope
    phi-gradient. ``lambda_beta`` (1.0 = pure) scales the coupling block but NOT the ``mass_phi``
    penalty, so the effective phi step is e_phi_lr * lambda_beta * nabla (lambda_beta and e_phi_lr
    interact). The ``mass_phi`` term makes the phi E-step descend the PENALIZED
    objective during inference (distinct from the outer M-step ||phi||^2 on the learned prior
    table). The canonical (entropy) branch reuses ``reduced_free_energy``, the -tau log Z envelope
    form.
    """
    # Build Omega under the ACTIVE connection regime so the phi step descends the SAME objective as
    # the mu/sigma step. regime_ii reads the (fixed) belief means mu and the learned connection_W;
    # mu and connection_W are held constant w.r.t. the phi gradient (only phi varies). The
    # factored-when-fusable dispatch (audit 2026-06-09 PE3) builds the per-token exps FROM THE
    # LIVE phi leaf -- the factored container is differentiable in phi, so the envelope
    # phi-gradient is preserved while the per-iteration dense (N,N,K,K) Omega disappears on the
    # flat equal-blocks path.
    omega = build_belief_transport(phi, group, transport_mode=transport_mode, mu=mu, sigma=sigma,
                                   connection_W=connection_W, connection_M=connection_M,
                                   connection_L=connection_L, link_alpha=link_alpha,
                                   link_soft_cap=link_soft_cap, clamp_monitor=clamp_monitor,
                                   cocycle_relaxation=cocycle_relaxation,
                                   rope=rope, rope_on_cov=rope_on_cov, rope_on_value=rope_on_value)
    mu_t = transport_mean(omega, mu)              # rank-agnostic: (N,N,K) or (B,N,N,K)
    sigma_t = transport_covariance(omega, sigma)
    fam = get_family(family)
    energy = pairwise_energy(fam(mu, sigma), fam(mu_t, sigma_t), alpha=renyi_order, kl_max=kl_max, eps=eps,
                             divergence_family=divergence_family, irrep_dims=group.irrep_dims)
    mass = 0.5 * mass_phi * (phi ** 2).sum() if mass_phi > 0.0 else 0.0
    if include_attention_entropy:
        return lambda_beta * reduced_free_energy(energy, tau=tau, log_prior=log_prior).sum() + mass
    beta = attention_weights(energy, tau=tau, log_prior=log_prior)
    return lambda_beta * (beta * energy).sum() + mass


def e_step_iteration(
    belief:                    BeliefState,
    mu_p:                      torch.Tensor,        # (N, K)
    sigma_p:                   torch.Tensor,        # (N, K)
    group:                     GaugeGroup,

    *,
    tau:                       'float | torch.Tensor' = 1.0,
    e_q_mu_lr:                 float = 0.1,
    e_q_sigma_lr:              float = 0.1,
    e_phi_lr:                  float = 0.1,
    renyi_order:               float = 1.0,
    value:                     float = 1.0,
    b0:                        'float | torch.Tensor' = 1.0,   # scalar, or (K,) per-coord for state_dependent_per_coord
    c0:                        'float | torch.Tensor' = 1.0,   # scalar, or (K,) per-coord
    lambda_beta:               'float | torch.Tensor' = 1.0,   # weight on the belief-coupling block (1.0 = pure)
    kl_max:                    float = 100.0,
    eps:                       float = 1e-6,
    sigma_max:                 float = 10.0,           # matches VFE3Config.sigma_max
    e_sigma_q_trust:           float = 5.0,
    e_mu_q_trust:              Optional[float] = None,   # mean trust radius (sigma units); None = unbounded
    mu_trust_mode:             str  = "box",             # "box" | "ball" (only when e_mu_q_trust is not None)
    e_step_mu_precond:         str  = "fisher",          # "fisher" (nat-grad mean) | "raw" (B3/EXP-14 mu-arm)
    mass_phi:                  float = 0.0,

    include_attention_entropy: bool = True,
    gradient_mode:             str  = "filtering",
    family:                    str  = "gaussian_diagonal",
    divergence_family:         str  = "renyi",
    lambda_alpha_mode:         str  = "constant",
    phi_precond_mode:          str  = "none",
    phi_retract_mode:          str  = "euclidean",
    spd_retract_mode:          str  = "spd_affine",
    transport_mode:            str  = "flat",
    e_step_gradient:           str  = "unroll",               # backward estimator: unroll | straight_through | detach
    oracle_unroll_grad:        bool = False,                  # opt-in: oracle returns a differentiable grad (unroll)
    cocycle_relaxation:        float = 1.0,                    # regime_ii homotopy alpha; 0 -> flat (ignored by flat)
    link_alpha:                float = 1.0,                    # direct-link scale (regime_ii_link / _charted)
    link_soft_cap:             float = 6.0,                    # direct-link embedded-Frobenius soft cap
    clamp_monitor:             bool = False,                   # opt-in: warn when the exp Frobenius clamp fires (host sync)

    log_prior:                 Optional[torch.Tensor] = None,
    log_alpha:                 Optional[torch.Tensor] = None,   # learned scalar self-coupling (None -> pure path)
    connection_W:              Optional[torch.Tensor] = None,   # learned bilinear connection for regime_ii (NN exception; None -> pure path)
    connection_M:              Optional[torch.Tensor] = None,   # learned covariant connection for regime_ii_covariant (Route B; None -> pure path)
    connection_L:              Optional[torch.Tensor] = None,   # learned direct link for regime_ii_link* (NN exception; None -> pure path)
    rope:                      Optional[torch.Tensor] = None,   # (N, K, K) gauge-RoPE rotation
    rope_on_cov:               bool                   = False,  # full-gauge: rotate covariance too
    rope_on_value:             bool                   = True,   # False -> value aggregation uses the un-rotated base
    grad_record:               Optional[dict]         = None,   # diag out-param: stashes ||grad_mu/sigma/phi|| (None -> no capture)
    _prebuilt_omega:           'torch.Tensor | FactoredTransport | RopeTransport | None' = None,   # PRIVATE: flat-path cache from e_step (phi-invariant when e_phi_lr==0)
) -> BeliefState:
    r"""One inner E-step iteration: mu, sigma (Fisher natgrad + SPD retraction) then phi
    (autograd of the alignment block + preconditioner + Lie retraction).

    The belief-transport build is config-selected through the connection-regime registry
    (``transport_mode``); the default 'flat' is the Regime-I phi-cocycle (byte-identical).
    'regime_ii' is the non-flat edge-relaxed cocycle and consumes the CURRENT belief means and the
    learned ``connection_W`` (a sanctioned NN exception); because that Omega depends on mu it is
    rebuilt from ``belief.mu`` every iteration here (flat is mu-independent). ``log_alpha`` is the
    learned self-coupling nn.Parameter (alpha = exp(log_alpha)) under lambda_alpha_mode='learnable' (None on
    the pure path); it flows into the belief gradient so the loss backpropagates to it through the
    unrolled E-step. ``connection_W`` likewise flows in only through these belief updates, so the
    loss backpropagates to it (and a detached E-step would freeze it, mirroring log_alpha)."""
    # Build the forward belief-transport (P0 #2): on the flat + block-diagonal-with-equal-blocks
    # path this is a FactoredTransport (the per-token exps only, NO dense (B,N,N,K,K) Omega), which
    # the belief-gradient kernel / oracle consume opaquely through transport_mean / covariance;
    # single-block / cross-coupled groups keep the dense Omega exactly as before.
    #
    # regime_ii (audit 2026-06-10 F1/F2): the mu-DEPENDENT Omega is NOT pre-built here. The hand
    # kernel is the flat-transport gradient (it drops d Omega/d mu), so regime_ii always routes to
    # the autograd oracle, and the oracle must rebuild the transport from its own differentiation
    # leaves for the gradient VALUE to carry d Omega/d mu in every regime (train, eval, detached
    # E-step). The builder closure below binds phi/W/rope and receives (mu_query, sigma_query,
    # mu_key, sigma_key) from the oracle -- the key slots are the oracle's detached keys under
    # filtering (query-side coordinate ascent) and the shared live leaves under smoothing (full
    # gradient). Pre-building here as well would only double the most expensive build in the
    # codebase.
    if transport_mode in _TRANSPORT_NEEDS_MU:                       # mu-dependent Omega -> autograd oracle
        omega = None

        def _omega_builder(mu_q: torch.Tensor, sigma_q: torch.Tensor,
                           mu_k: torch.Tensor, sigma_k: torch.Tensor):
            return build_belief_transport(
                belief.phi, group, transport_mode=transport_mode,
                mu=mu_q, mu_key=mu_k, connection_W=connection_W,
                # regime_ii_covariant: the gauge-invariant features also read the belief variances;
                # thread the ORACLE's sigma leaves (query slot live, key slot detached under
                # filtering -- the same coordinate-ascent split as mu/mu_key) so autograd sees
                # d Omega/d sigma on every path, not only the live unroll (audit 2026-07-01 C4).
                sigma=sigma_q, sigma_key=sigma_k, connection_M=connection_M,
                cocycle_relaxation=cocycle_relaxation, clamp_monitor=clamp_monitor,
                rope=rope, rope_on_cov=rope_on_cov,
                rope_on_value=rope_on_value,
            )
        omega_builder = _omega_builder
    else:
        # Use the hoisted transport when the caller pre-built it (flat path, e_phi_lr==0 so
        # belief.phi is iteration-invariant). Fall back to building here for direct calls or
        # when phi is live (e_phi_lr > 0 means the previous iteration updated phi).
        if _prebuilt_omega is not None:
            omega = _prebuilt_omega
        else:
            omega = build_belief_transport(
                belief.phi, group, transport_mode=transport_mode,
                mu=belief.mu, connection_W=connection_W, connection_L=connection_L,
                link_alpha=link_alpha, link_soft_cap=link_soft_cap, clamp_monitor=clamp_monitor,
                cocycle_relaxation=cocycle_relaxation,
                rope=rope, rope_on_cov=rope_on_cov, rope_on_value=rope_on_value,
            )
        omega_builder = None
    # Runtime truncation warning (audit 2026-06-09 P8): under the 'unroll' estimator with a LIVE
    # belief, a non-kernel route is served by the autograd oracle, which returns a DETACHED
    # tangent unless oracle_unroll_grad=True -- silently severing the unrolled-through-inference
    # signal to the prior tables. The config-time warning covers only the learnable-parameter
    # cases; this fires when the truncation actually happens (default warnings filter: once).
    if (e_step_gradient == "unroll" and not oracle_unroll_grad and belief.mu.requires_grad
            and not uses_kernel_route(
                renyi_order=renyi_order, gradient_mode=gradient_mode, family=family,
                divergence_family=divergence_family,
                include_attention_entropy=include_attention_entropy,
                transport_mode=transport_mode,
                decoupled_value_gauge=(rope is not None and not rope_on_value))):
        import warnings
        warnings.warn(
            "e_step_gradient='unroll' is being served by the autograd ORACLE with "
            "oracle_unroll_grad=False: the belief gradient is a detached tangent, so the "
            "unrolled-through-inference signal to the prior tables is truncated. Set "
            "oracle_unroll_grad=True for the differentiable oracle (or use the kernel route).",
            UserWarning,
            stacklevel=2,
        )
    grad_mu, grad_sigma = belief_gradients(
        belief.mu, belief.sigma, mu_p, sigma_p, omega,
        tau=tau, renyi_order=renyi_order, value=value, b0=b0, c0=c0, lambda_beta=lambda_beta,
        kl_max=kl_max, eps=eps,
        include_attention_entropy=include_attention_entropy, gradient_mode=gradient_mode,
        family=family, divergence_family=divergence_family, lambda_alpha_mode=lambda_alpha_mode,
        transport_mode=transport_mode, omega_builder=omega_builder,
        irrep_dims=group.irrep_dims, log_prior=log_prior, log_alpha=log_alpha,
        # Opt-in unrolled-oracle: make the autograd oracle (non-kernel families) return a
        # differentiable belief gradient so the through-inference signal reaches the prior, matching
        # the hand kernel. Gated on the explicit oracle_unroll_grad toggle (default OFF preserves the
        # detached oracle); only meaningful under the 'unroll' estimator (straight_through detaches
        # downstream, detach runs under no_grad).
        create_graph=(oracle_unroll_grad and e_step_gradient == "unroll"),
    )
    if grad_record is not None:
        # Diagnostic: the RAW belief-gradient L2 norms ||grad_mu||, ||grad_sigma|| (pre-natural-grade,
        # the E-step analogue of the M-step's raw per-group grad_norm). Detached 0-dim tensors (no host
        # sync here, no graph retained); phi defaults to 0 and is overwritten below iff phi steps.
        grad_record["mu"]    = grad_mu.detach().pow(2).sum().sqrt()
        grad_record["sigma"] = grad_sigma.detach().pow(2).sum().sqrt()
        grad_record["phi"]   = grad_mu.new_zeros(())
    # Fisher preconditioner is FAMILY-KEYED (add-by-registering): each BeliefParams owns its Fisher
    # metric, so a non-Gaussian family (e.g. laplace_diagonal, I_mu=I_b=1/b^2) is descended in its
    # own geometry instead of the hardcoded Gaussian Fisher. The Gaussian families delegate to the
    # pinned geometry kernel (byte-identical); 'family' is the same key passed to belief_gradients.
    nat_mu, nat_sigma = get_family(family)(belief.mu, belief.sigma).natural_gradient(
        grad_mu, grad_sigma, eps=eps)

    # STRAIGHT-THROUGH (manuscript Algorithm 1, GL(K)_attention.tex:2050): detach the per-iteration
    # tangent so only the ADDITIVE chain stays live -- the belief is rebuilt as mu_prev + delta and
    # retract(sigma_prev, delta) below, giving d belief_next/d belief_prev = I WITHOUT the
    # second-order d delta/d belief_prev term the unrolled path keeps. The tangent is detached AFTER
    # natural_gradient (which reintroduces a live belief.sigma dependence), not at grad_mu/grad_sigma,
    # so no second-order term leaks through the Fisher metric. 'unroll' (and the no_grad-wrapped
    # 'detach') leave nat_mu/nat_sigma untouched, so those lines are byte-identical to before and the
    # forward VALUE is unchanged (detach never alters a number). This mirrors the phi step, which is
    # already straight-through (fresh detached leaf, create_graph=False).
    if e_step_gradient == "straight_through":
        nat_mu, nat_sigma = nat_mu.detach(), nat_sigma.detach()

    # B3/EXP-14 mean-arm ablation: descend the Fisher natural gradient nat_mu (= Sigma*grad_mu for a
    # diagonal Gaussian) by default, or the raw Euclidean grad_mu under e_step_mu_precond='raw'. The
    # SPD sigma retraction below is unchanged either way, isolating the MEAN preconditioner; grad_mu is
    # detached under straight_through to match nat_mu's per-iteration tangent severance.
    if e_step_mu_precond == "raw":
        mu_grad = grad_mu.detach() if e_step_gradient == "straight_through" else grad_mu
    else:
        mu_grad = nat_mu
    delta_mu = e_q_mu_lr * mu_grad
    # E-step MEAN trust region (default OFF). When e_mu_q_trust is set, bound the
    # per-iteration mean step in sigma-whitened units before the additive update; None reproduces the
    # bare mu = belief.mu - e_q_mu_lr*nat_mu bit-for-bit. is_diagonal mirrors the SPD-retraction rank
    # rule below (full cov iff sigma.dim() == mu.dim() + 1).
    if e_mu_q_trust is not None:
        delta_mu = apply_mu_trust_region(
            delta_mu, belief.sigma, trust=e_mu_q_trust, mode=mu_trust_mode,
            is_diagonal=(belief.sigma.dim() == belief.mu.dim()), eps=eps,
        )
    mu = belief.mu - delta_mu
    # The registered SPD retraction owns the diagonal-vs-full rank decision internally (full cov iff
    # sigma.dim() == mu.dim() + 1); the E-step no longer branches on rank to select the retraction.
    sigma = get_retraction(spd_retract_mode)(
        belief.sigma, -e_q_sigma_lr * nat_sigma, belief.mu.dim(),
        trust_region=e_sigma_q_trust, eps=eps, sigma_max=sigma_max,
    )

    phi = belief.phi
    if e_phi_lr > 0.0:
        # Sequential (Gauss-Seidel) coordinate descent, by design (audit 2026-06-10 F15): the phi
        # substep is evaluated at the UPDATED (mu, sigma) with the pre-update phi -- each substep
        # descends F at the current point of the sweep, so under regime_ii the edge factor inside
        # phi_alignment_loss reads the post-mu-step means. This mixed state is the standard
        # coordinate-descent sweep, not a frozen joint state.
        # The phi natural gradient fundamentally requires autograd (autograd.grad on a
        # fresh requires_grad leaf), so it must run under an enable_grad island even when
        # the caller wraps the stack in no_grad (the detach_e_step / fixed-point regime).
        # create_graph defaults to False, so grad_phi is detached from the outer graph and
        # acts as a constant tangent there; on the default unrolled path enable_grad is a
        # no-op and the phi-graph connection still flows belief.phi -> retract_phi -> omega.
        with torch.enable_grad():
            phi_g = belief.phi.detach().clone().requires_grad_(True)
            L = phi_alignment_loss(
                mu, sigma, phi_g, group, tau=tau, renyi_order=renyi_order, kl_max=kl_max, eps=eps,
                mass_phi=mass_phi, lambda_beta=lambda_beta, family=family, divergence_family=divergence_family,
                include_attention_entropy=include_attention_entropy, log_prior=log_prior,
                transport_mode=transport_mode, cocycle_relaxation=cocycle_relaxation,
                # gauge-RoPE: the phi step must descend the SAME rotated belief-coupling block as the
                # mu/sigma step, else under pos_rotation='rope' + e_phi_lr>0 phi optimizes a different
                # free energy than mu/sigma (audit 2026-06-17 round 2 id15). None/off -> byte-identical.
                rope=rope, rope_on_cov=rope_on_cov, rope_on_value=rope_on_value,
                # INTENTIONAL asymmetry (audit 2026-06-09 D3): connection_W is detached here, so
                # the learned Regime-II connection trains ONLY through the mu/sigma belief path,
                # never through the phi-step autograd island (whose grad is a constant tangent to
                # the outer graph anyway when create_graph=False). Removing the detach would leak
                # second-order phi-step terms into connection_W's gradient.
                connection_W=(connection_W.detach() if connection_W is not None else None),
                connection_M=(connection_M.detach() if connection_M is not None else None),
                connection_L=(connection_L.detach() if connection_L is not None else None),
                link_alpha=link_alpha, link_soft_cap=link_soft_cap, clamp_monitor=clamp_monitor,
            )
            grad_phi = torch.autograd.grad(L, phi_g)[0]
        if grad_record is not None:                      # RAW phi-gradient norm (pre-precondition)
            grad_record["phi"] = grad_phi.detach().pow(2).sum().sqrt()
        grad_phi = precondition_phi_gradient(
            grad_phi, belief.phi, group.generators, mode=phi_precond_mode, irrep_dims=group.irrep_dims,
        )
        phi = retract_phi(belief.phi, -grad_phi, group, step_size=e_phi_lr, mode=phi_retract_mode)

    return BeliefState(mu=mu, sigma=sigma, phi=phi)


def e_step(
    belief:            BeliefState,
    mu_p:              torch.Tensor,        # (N, K)
    sigma_p:           torch.Tensor,        # (N, K)
    group:             GaugeGroup,

    *,
    n_iter:            int   = 1,
    tau:               'float | torch.Tensor' = 1.0,
    e_q_mu_lr:         float = 0.1,
    e_q_sigma_lr:      float = 0.1,
    e_phi_lr:          float = 0.1,
    return_trajectory: bool  = False,
    e_step_gradient:   str   = "unroll",
    oracle_unroll_grad: bool = False,            # explicit (not in kwargs): keep it off the F_diag bag
    grad_record:       Optional[dict]         = None,   # diag out-param (explicit): LAST iteration's belief-grad norms
    rope:              Optional[torch.Tensor] = None,
    rope_on_cov:       bool                   = False,
    rope_on_value:     bool                   = True,

    log_prior:         Optional[torch.Tensor] = None,
    **kwargs,
) -> 'BeliefState | Tuple[BeliefState, List[float]]':
    r"""Iterate ``e_step_iteration`` ``n_iter`` times (parallel mean-field). Optionally
    returns the global-F trajectory (a DIAGNOSTIC; parallel updates are not guaranteed
    monotone per iteration).

    ``e_step_gradient`` is the backward estimator forwarded to each iteration: 'unroll' (default,
    full second-order trajectory gradient) or 'straight_through' (per-iteration tangent detached,
    additive chain live). It is an EXPLICIT keyword (not in ``**kwargs``) so it binds here and does
    NOT ride the forwarded knob bag into the diagnostic ``free_energy_value`` (which rejects unknown
    kwargs). 'detach' is realized by the caller wrapping this in no_grad, so it is treated as
    'unroll' here (no_grad already severs every gradient)."""
    traj: List[float] = []

    # Hoist the flat transport when phi is frozen across iterations (e_phi_lr==0).  On the flat
    # path the transport depends only on belief.phi, which the phi sub-step never updates when
    # e_phi_lr==0, so it is iteration-invariant.  Build ONCE here and pass it into every
    # e_step_iteration call via _prebuilt_omega, eliminating the redundant per-iteration rebuild.
    # When e_phi_lr > 0 phi changes each iteration so we leave _prebuilt_omega=None and let
    # e_step_iteration rebuild as before.  regime_ii is excluded because its Omega is mu-dependent
    # (mu changes every iteration regardless of e_phi_lr).
    transport_mode_kw: str = kwargs.get("transport_mode", "flat")
    _hoisted_omega: 'torch.Tensor | FactoredTransport | RopeTransport | None' = None
    if e_phi_lr == 0.0 and transport_mode_kw == "flat":
        _hoisted_omega = build_belief_transport(
            belief.phi, group,
            transport_mode="flat",
            gauge_mode=kwargs.get("gauge_mode", "learned"),
            clamp_monitor=kwargs.get("clamp_monitor", False),
            rope=rope, rope_on_cov=rope_on_cov, rope_on_value=rope_on_value,
        )

    def _f_diag(b: BeliefState) -> float:
        # Diagnostic scalar: under no_grad so the logged trajectory never enters the
        # training graph, and .item() instead of float(tensor) makes the host sync explicit.
        # rope/rope_on_cov are forwarded explicitly (audit PP6): the logged F carries the same
        # RoPE-wrapped transport the iterations descend.
        # NOTE (audit 2026-06-13): under a non-'constant' lambda_alpha_mode/lambda_h_mode the forwarded kwargs
        # carry the regularizer into free_energy_value (alpha_reg=R), so the logged F is the AUGMENTED
        # objective F+R the iterations actually descend (the envelope-correct value), not the bare
        # divergence-weighted F. At the constant default alpha_reg=None, so the trajectory is the pure F.
        with torch.no_grad():
            return free_energy_value(b, mu_p, sigma_p, group, tau=tau, log_prior=log_prior,
                                     rope=rope, rope_on_cov=rope_on_cov, rope_on_value=rope_on_value,
                                     **kwargs).item()

    if return_trajectory:
        traj.append(_f_diag(belief))
    for _ in range(n_iter):
        belief = e_step_iteration(
            belief, mu_p, sigma_p, group, tau=tau,
            e_q_mu_lr=e_q_mu_lr, e_q_sigma_lr=e_q_sigma_lr, e_phi_lr=e_phi_lr,
            e_step_gradient=e_step_gradient, oracle_unroll_grad=oracle_unroll_grad,
            grad_record=grad_record,                       # last iteration overwrites -> converged-ish grad
            log_prior=log_prior, rope=rope, rope_on_cov=rope_on_cov, rope_on_value=rope_on_value,
            _prebuilt_omega=_hoisted_omega,
            **kwargs,
        )
        if return_trajectory:
            traj.append(_f_diag(belief))
    return (belief, traj) if return_trajectory else belief
