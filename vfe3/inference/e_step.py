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
from vfe3.geometry.retraction import get_retraction, natural_gradient, retract_phi
from vfe3.geometry.transport import compute_transport_operators, get_transport, transport_covariance, transport_mean
from vfe3.gradients.kernels import belief_gradients


def _transport(
    phi:                torch.Tensor,             # (N, n_gen) or (B, N, n_gen)
    group:              GaugeGroup,

    *,
    transport_mode:     str                    = "flat",   # connection-regime registry key (default = flat)
    mu:                 Optional[torch.Tensor] = None,      # (N, K) or (B, N, K) means; regime_ii edge connection reads these
    connection_W:       Optional[torch.Tensor] = None,      # (n_gen, K, K) learned bilinear connection (regime_ii, NN exception)
    cocycle_relaxation: float                  = 1.0,       # regime_ii homotopy alpha; 0 -> flat
) -> torch.Tensor:                            # (N, N, K, K) or (B, N, N, K, K) Omega_ij
    r"""Build the pairwise transport Omega_ij via the connection-regime registry.

    The build is config-selected through ``get_transport(transport_mode)``; the default 'flat' is
    the Regime-I phi-cocycle Omega_ij = exp(phi_i) exp(-phi_j) (byte-identical to a direct
    ``compute_transport_operators`` call, mu/connection_W ignored). 'regime_ii' is the NON-FLAT
    edge-relaxed cocycle (a sanctioned learned-connection NN exception): it reads the CURRENT belief
    means ``mu`` and the learned ``connection_W`` to insert the edge factor exp(delta_ij . G), so it
    must be REBUILT as mu updates each E-step iteration (flat is mu-independent).

    Rank-aware: a 2-D (N, n_gen) frame (the unbatched diagnostics / trajectory path) is transported
    as a batch of one and stripped back to (N, N, K, K); a 3-D (B, N, n_gen) frame (the batched
    forward) flows straight through. ``mu`` is unsqueezed to match so the builder always sees a
    batched (B, N, K) mean."""
    build = get_transport(transport_mode)
    if phi.dim() == 2:
        mu_b = mu.unsqueeze(0) if mu is not None else None
        return build(phi.unsqueeze(0), group, mu=mu_b, connection_W=connection_W,
                     cocycle_relaxation=cocycle_relaxation)["Omega"][0]
    return build(phi, group, mu=mu, connection_W=connection_W,
                 cocycle_relaxation=cocycle_relaxation)["Omega"]


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
    tau:                       float = 1.0,
    alpha_div:                 float = 1.0,
    value:                     float = 1.0,
    b0:                        float = 1.0,
    c0:                        float = 1.0,
    kl_max:                    float = 100.0,
    eps:                       float = 1e-6,
    sigma_max:                 float = 5.0,            # accepted-and-ignored iteration-only knob
    e_sigma_q_trust:           float = 5.0,            # accepted-and-ignored iteration-only knob

    include_attention_entropy: bool = True,
    family:                    str  = "gaussian_diagonal",
    divergence_family:         str  = "renyi",
    alpha_mode:                str  = "constant",
    gradient_mode:             str  = "filtering",     # accepted-and-ignored iteration-only knob
    phi_precond_mode:          str  = "none",          # accepted-and-ignored iteration-only knob
    phi_retract_mode:          str  = "euclidean",     # accepted-and-ignored iteration-only knob
    spd_retract_mode:          str  = "spd_affine",    # accepted-and-ignored iteration-only knob
    transport_mode:            str  = "flat",          # accepted-and-ignored iteration-only knob
    cocycle_relaxation:        float = 1.0,            # accepted-and-ignored iteration-only knob (regime_ii)

    log_prior:                 Optional[torch.Tensor] = None,
    log_alpha:                 Optional[torch.Tensor] = None,   # learned scalar self-coupling (None -> pure path)
    connection_W:              Optional[torch.Tensor] = None,   # accepted-and-ignored iteration-only knob (regime_ii NN exception)
    keys:                      Optional[BeliefState]  = None,   # None -> global F; else keys frozen at `keys`
) -> torch.Tensor:                   # scalar F
    r"""Scalar free energy of a belief. ``keys=None`` -> global F (keys = the belief);
    ``keys`` given -> F with the transported keys frozen at ``keys`` (the F_filt objective).

        F = Sum_i [ alpha_i D(q_i||p_i) (+ R(alpha_i))
                  + Sum_j beta_ij E_ij + tau Sum_j beta_ij log(beta_ij/pi_ij) ],
        E_ij = D(q_i || Omega_ij q_j),  beta = softmax_j(log_prior - E/tau).

    The iteration-only knobs (gradient_mode, phi_precond_mode, phi_retract_mode,
    spd_retract_mode, transport_mode, cocycle_relaxation, connection_W, sigma_max, e_sigma_q_trust)
    are declared here as EXPLICIT accept-and-ignore parameters (not a blanket ``**kwargs`` sink) so the
    common ``e_step`` call site may forward one knob bag to both this and ``e_step_iteration`` while a
    MISSPELLED real parameter still raises ``TypeError`` here instead of being silently swallowed. NOTE:
    the trajectory-diagnostic F here always uses the FLAT transport (its internal ``_transport`` omits
    transport_mode), so under regime_ii the logged F-trajectory is a flat-transport diagnostic, not the
    regime_ii objective -- wiring it is out of scope (matches log_alpha, also not threaded here).
    """
    # keys=None -> global F (query = key = belief). keys given -> filtered F: the transport
    # Omega_ij uses the CURRENT query frame phi_i (belief) and the FROZEN key frame phi_j (keys),
    # and the transported key beliefs come from `keys`; only the key side is frozen.
    key_belief = belief if keys is None else keys
    omega = _transport(belief.phi, group) if keys is None else _transport_qk(belief.phi, keys.phi, group)
    mu_t = transport_mean(omega.unsqueeze(0), key_belief.mu.unsqueeze(0))[0]
    sigma_t = transport_covariance(omega.unsqueeze(0), key_belief.sigma.unsqueeze(0))[0]

    fam = get_family(family)
    sd = self_divergence_for_alpha(fam(belief.mu, belief.sigma), fam(mu_p, sigma_p), alpha=alpha_div, kl_max=kl_max,
                                   eps=eps, divergence_family=divergence_family, alpha_mode=alpha_mode)
    alpha, reg = self_coupling_alpha(sd, value=value, mode=alpha_mode, b0=b0, c0=c0, log_alpha=log_alpha)
    energy = pairwise_energy(fam(belief.mu, belief.sigma), fam(mu_t, sigma_t), alpha=alpha_div, kl_max=kl_max, eps=eps,
                             divergence_family=divergence_family, irrep_dims=group.irrep_dims)
    return free_energy(
        sd, energy, alpha, tau=tau, include_attention_entropy=include_attention_entropy,
        log_prior=log_prior, alpha_reg=(reg if alpha_mode != "constant" else None),
    )


def phi_alignment_loss(
    mu:        torch.Tensor,             # (N, K)
    sigma:     torch.Tensor,             # (N, K)
    phi:       torch.Tensor,             # (N, n_gen) -- the differentiated variable
    group:     GaugeGroup,

    *,
    tau:       float = 1.0,
    alpha_div: float = 1.0,
    kl_max:    float = 100.0,
    eps:       float = 1e-6,
    mass_phi:  float = 0.0,
    family:    str   = "gaussian_diagonal",
    divergence_family: str = "renyi",

    include_attention_entropy: bool = True,
    log_prior: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Canonical belief-coupling block as a function of phi (mu, sigma fixed), plus the
    gauge-frame penalty (manuscript Algorithm 1, line for nabla_phi F):

        L(phi) = Sum_ij [ beta_ij E_ij + tau beta_ij log(beta_ij/pi_ij) ] + (mass_phi/2) ||phi||^2,
        E_ij = D(q_i || Omega_ij(phi) q_j),  beta = softmax_j(log_prior - E/tau).
    Both roles of phi flow (Omega_ij depends on phi_i and phi_j); autograd gives the envelope
    phi-gradient. The ``mass_phi`` term makes the phi E-step descend the PENALIZED objective
    during inference (distinct from the outer M-step ||phi||^2 on the learned prior table). The
    canonical (entropy) branch reuses ``reduced_free_energy``, the -tau log Z envelope form.
    """
    omega = _transport(phi, group)
    mu_t = transport_mean(omega, mu)              # rank-agnostic: (N,N,K) or (B,N,N,K)
    sigma_t = transport_covariance(omega, sigma)
    fam = get_family(family)
    energy = pairwise_energy(fam(mu, sigma), fam(mu_t, sigma_t), alpha=alpha_div, kl_max=kl_max, eps=eps,
                             divergence_family=divergence_family, irrep_dims=group.irrep_dims)
    mass = 0.5 * mass_phi * (phi ** 2).sum() if mass_phi > 0.0 else 0.0
    if include_attention_entropy:
        return reduced_free_energy(energy, tau=tau, log_prior=log_prior).sum() + mass
    beta = attention_weights(energy, tau=tau, log_prior=log_prior)
    return (beta * energy).sum() + mass


def e_step_iteration(
    belief:                    BeliefState,
    mu_p:                      torch.Tensor,        # (N, K)
    sigma_p:                   torch.Tensor,        # (N, K)
    group:                     GaugeGroup,

    *,
    tau:                       float = 1.0,
    e_mu_lr:                   float = 0.1,
    e_sigma_lr:                float = 0.1,
    e_phi_lr:                  float = 0.1,
    alpha_div:                 float = 1.0,
    value:                     float = 1.0,
    b0:                        float = 1.0,
    c0:                        float = 1.0,
    kl_max:                    float = 100.0,
    eps:                       float = 1e-6,
    sigma_max:                 float = 5.0,
    e_sigma_q_trust:           float = 5.0,
    mass_phi:                  float = 0.0,

    include_attention_entropy: bool = True,
    gradient_mode:             str  = "filtering",
    family:                    str  = "gaussian_diagonal",
    divergence_family:         str  = "renyi",
    alpha_mode:                str  = "constant",
    phi_precond_mode:          str  = "none",
    phi_retract_mode:          str  = "euclidean",
    spd_retract_mode:          str  = "spd_affine",
    transport_mode:            str  = "flat",
    e_step_gradient:           str  = "unroll",               # backward estimator: unroll | straight_through | detach
    cocycle_relaxation:        float = 1.0,                    # regime_ii homotopy alpha; 0 -> flat (ignored by flat)

    log_prior:                 Optional[torch.Tensor] = None,
    log_alpha:                 Optional[torch.Tensor] = None,   # learned scalar self-coupling (None -> pure path)
    connection_W:              Optional[torch.Tensor] = None,   # learned bilinear connection for regime_ii (NN exception; None -> pure path)
) -> BeliefState:
    r"""One inner E-step iteration: mu, sigma (Fisher natgrad + SPD retraction) then phi
    (autograd of the alignment block + preconditioner + Lie retraction).

    The belief-transport build is config-selected through the connection-regime registry
    (``transport_mode``); the default 'flat' is the Regime-I phi-cocycle (byte-identical).
    'regime_ii' is the non-flat edge-relaxed cocycle and consumes the CURRENT belief means and the
    learned ``connection_W`` (a sanctioned NN exception); because that Omega depends on mu it is
    rebuilt from ``belief.mu`` every iteration here (flat is mu-independent). ``log_alpha`` is the
    learned self-coupling nn.Parameter (alpha = exp(log_alpha)) under alpha_mode='learnable' (None on
    the pure path); it flows into the belief gradient so the loss backpropagates to it through the
    unrolled E-step. ``connection_W`` likewise flows in only through these belief updates, so the
    loss backpropagates to it (and a detached E-step would freeze it, mirroring log_alpha)."""
    # Only regime_ii needs the belief means + learned connection; flat ignores them, so the default
    # call passes the un-threaded defaults (mu=None) and is byte-identical to the pre-regime_ii path.
    transport_kw = (
        dict(mu=belief.mu, connection_W=connection_W, cocycle_relaxation=cocycle_relaxation)
        if transport_mode == "regime_ii" else {}
    )
    omega = _transport(belief.phi, group, transport_mode=transport_mode, **transport_kw)
    grad_mu, grad_sigma = belief_gradients(
        belief.mu, belief.sigma, mu_p, sigma_p, omega,
        tau=tau, alpha_div=alpha_div, value=value, b0=b0, c0=c0, kl_max=kl_max, eps=eps,
        include_attention_entropy=include_attention_entropy, gradient_mode=gradient_mode,
        family=family, divergence_family=divergence_family, alpha_mode=alpha_mode,
        irrep_dims=group.irrep_dims, log_prior=log_prior, log_alpha=log_alpha,
    )
    nat_mu, nat_sigma = natural_gradient(grad_mu, grad_sigma, belief.sigma, eps=eps)

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

    mu = belief.mu - e_mu_lr * nat_mu
    # The registered SPD retraction owns the diagonal-vs-full rank decision internally (full cov iff
    # sigma.dim() == mu.dim() + 1); the E-step no longer branches on rank to select the retraction.
    sigma = get_retraction(spd_retract_mode)(
        belief.sigma, -e_sigma_lr * nat_sigma, belief.mu.dim(),
        trust_region=e_sigma_q_trust, eps=eps, sigma_max=sigma_max,
    )

    phi = belief.phi
    if e_phi_lr > 0.0:
        # The phi natural gradient fundamentally requires autograd (autograd.grad on a
        # fresh requires_grad leaf), so it must run under an enable_grad island even when
        # the caller wraps the stack in no_grad (the detach_e_step / fixed-point regime).
        # create_graph defaults to False, so grad_phi is detached from the outer graph and
        # acts as a constant tangent there; on the default unrolled path enable_grad is a
        # no-op and the phi-graph connection still flows belief.phi -> retract_phi -> omega.
        with torch.enable_grad():
            phi_g = belief.phi.detach().clone().requires_grad_(True)
            L = phi_alignment_loss(
                mu, sigma, phi_g, group, tau=tau, alpha_div=alpha_div, kl_max=kl_max, eps=eps,
                mass_phi=mass_phi, family=family, divergence_family=divergence_family,
                include_attention_entropy=include_attention_entropy, log_prior=log_prior,
            )
            grad_phi = torch.autograd.grad(L, phi_g)[0]
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
    tau:               float = 1.0,
    e_mu_lr:           float = 0.1,
    e_sigma_lr:        float = 0.1,
    e_phi_lr:          float = 0.1,
    return_trajectory: bool  = False,
    e_step_gradient:   str   = "unroll",

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

    def _f_diag(b: BeliefState) -> float:
        # Diagnostic scalar: under no_grad so the logged trajectory never enters the
        # training graph, and .item() instead of float(tensor) makes the host sync explicit.
        with torch.no_grad():
            return free_energy_value(b, mu_p, sigma_p, group, tau=tau, log_prior=log_prior, **kwargs).item()

    if return_trajectory:
        traj.append(_f_diag(belief))
    for _ in range(n_iter):
        belief = e_step_iteration(
            belief, mu_p, sigma_p, group, tau=tau,
            e_mu_lr=e_mu_lr, e_sigma_lr=e_sigma_lr, e_phi_lr=e_phi_lr,
            e_step_gradient=e_step_gradient, log_prior=log_prior, **kwargs,
        )
        if return_trajectory:
            traj.append(_f_diag(belief))
    return (belief, traj) if return_trajectory else belief
