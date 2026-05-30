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
from vfe3.free_energy import attention_weights, free_energy, pairwise_energy, reduced_free_energy, self_divergence
from vfe3.geometry.groups import GaugeGroup
from vfe3.geometry.phi_preconditioner import precondition_phi_gradient
from vfe3.geometry.retraction import natural_gradient, retract_phi, retract_spd_diagonal, retract_spd_full
from vfe3.geometry.transport import compute_transport_operators, transport_covariance, transport_mean
from vfe3.gradients.kernels import belief_gradients


def _transport(
    phi:   torch.Tensor,             # (N, n_gen)
    group: GaugeGroup,
) -> torch.Tensor:                   # (N, N, K, K) Omega_ij
    r"""Build the pairwise transport Omega_ij = exp(phi_i) exp(-phi_j)."""
    return compute_transport_operators(phi.unsqueeze(0), group)["Omega"][0]


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
    alpha_mode:                str  = "constant",
    gradient_mode:             str  = "filtering",     # accepted-and-ignored iteration-only knob
    phi_precond_mode:          str  = "none",          # accepted-and-ignored iteration-only knob
    phi_retract_mode:          str  = "euclidean",     # accepted-and-ignored iteration-only knob

    log_prior:                 Optional[torch.Tensor] = None,
    keys:                      Optional[BeliefState]  = None,   # None -> global F; else keys frozen at `keys`
) -> torch.Tensor:                   # scalar F
    r"""Scalar free energy of a belief. ``keys=None`` -> global F (keys = the belief);
    ``keys`` given -> F with the transported keys frozen at ``keys`` (the F_filt objective).

        F = Sum_i [ alpha_i D(q_i||p_i) (+ R(alpha_i))
                  + Sum_j beta_ij E_ij + tau Sum_j beta_ij log(beta_ij/pi_ij) ],
        E_ij = D(q_i || Omega_ij q_j),  beta = softmax_j(log_prior - E/tau).

    The iteration-only knobs (gradient_mode, phi_precond_mode, phi_retract_mode,
    sigma_max, e_sigma_q_trust) are declared here as EXPLICIT accept-and-ignore
    parameters (not a blanket ``**kwargs`` sink) so the common ``e_step`` call site may
    forward one knob bag to both this and ``e_step_iteration`` while a MISSPELLED real
    parameter still raises ``TypeError`` here instead of being silently swallowed.
    """
    key_belief = belief if keys is None else keys
    omega = _transport(key_belief.phi, group)
    mu_t = transport_mean(omega.unsqueeze(0), key_belief.mu.unsqueeze(0))[0]
    sigma_t = transport_covariance(omega.unsqueeze(0), key_belief.sigma.unsqueeze(0))[0]

    sd = self_divergence(belief.mu, belief.sigma, mu_p, sigma_p, alpha=alpha_div, kl_max=kl_max, eps=eps, family=family)
    alpha, reg = self_coupling_alpha(sd, value=value, mode=alpha_mode, b0=b0, c0=c0)
    energy = pairwise_energy(belief.mu, belief.sigma, mu_t, sigma_t, alpha=alpha_div, kl_max=kl_max, eps=eps, family=family)
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
    family:    str   = "gaussian_diagonal",

    include_attention_entropy: bool = True,
    log_prior: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Canonical belief-coupling block as a function of phi (mu, sigma fixed):

        L(phi) = Sum_ij [ beta_ij E_ij + tau beta_ij log(beta_ij/pi_ij) ],
        E_ij = D(q_i || Omega_ij(phi) q_j),  beta = softmax_j(log_prior - E/tau).
    Both roles of phi flow (Omega_ij depends on phi_i and phi_j); autograd gives the
    envelope phi-gradient. The canonical (entropy) branch reuses ``reduced_free_energy``,
    the -tau log Z envelope form of that block (the pi-fallback + clamp live there, once).
    """
    omega = _transport(phi, group)
    mu_t = transport_mean(omega.unsqueeze(0), mu.unsqueeze(0))[0]
    sigma_t = transport_covariance(omega.unsqueeze(0), sigma.unsqueeze(0))[0]
    energy = pairwise_energy(mu, sigma, mu_t, sigma_t, alpha=alpha_div, kl_max=kl_max, eps=eps, family=family)
    if include_attention_entropy:
        return reduced_free_energy(energy, tau=tau, log_prior=log_prior).sum()
    beta = attention_weights(energy, tau=tau, log_prior=log_prior)
    return (beta * energy).sum()


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

    include_attention_entropy: bool = True,
    gradient_mode:             str  = "filtering",
    family:                    str  = "gaussian_diagonal",
    alpha_mode:                str  = "constant",
    phi_precond_mode:          str  = "none",
    phi_retract_mode:          str  = "euclidean",

    log_prior:                 Optional[torch.Tensor] = None,
) -> BeliefState:
    r"""One inner E-step iteration: mu, sigma (Fisher natgrad + SPD retraction) then phi
    (autograd of the alignment block + preconditioner + Lie retraction)."""
    omega = _transport(belief.phi, group)
    grad_mu, grad_sigma = belief_gradients(
        belief.mu, belief.sigma, mu_p, sigma_p, omega,
        tau=tau, alpha_div=alpha_div, value=value, b0=b0, c0=c0, kl_max=kl_max, eps=eps,
        include_attention_entropy=include_attention_entropy, gradient_mode=gradient_mode,
        family=family, alpha_mode=alpha_mode, log_prior=log_prior,
    )
    nat_mu, nat_sigma = natural_gradient(grad_mu, grad_sigma, belief.sigma, eps=eps)

    mu = belief.mu - e_mu_lr * nat_mu
    if belief.sigma.dim() == belief.mu.dim() + 1:        # full covariance (..., K, K)
        sigma = retract_spd_full(
            belief.sigma, -e_sigma_lr * nat_sigma, trust_region=e_sigma_q_trust, eps=eps, sigma_max=sigma_max,
        )
    else:                                                # diagonal variances (..., K)
        sigma = retract_spd_diagonal(
            belief.sigma, -e_sigma_lr * nat_sigma, trust_region=e_sigma_q_trust, eps=eps, sigma_max=sigma_max,
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
                family=family, include_attention_entropy=include_attention_entropy, log_prior=log_prior,
            )
            grad_phi = torch.autograd.grad(L, phi_g)[0]
        grad_phi = precondition_phi_gradient(grad_phi, belief.phi, group.generators, mode=phi_precond_mode)
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

    log_prior:         Optional[torch.Tensor] = None,
    **kwargs,
) -> 'BeliefState | Tuple[BeliefState, List[float]]':
    r"""Iterate ``e_step_iteration`` ``n_iter`` times (parallel mean-field). Optionally
    returns the global-F trajectory (a DIAGNOSTIC; parallel updates are not guaranteed
    monotone per iteration)."""
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
            e_mu_lr=e_mu_lr, e_sigma_lr=e_sigma_lr, e_phi_lr=e_phi_lr, log_prior=log_prior, **kwargs,
        )
        if return_trajectory:
            traj.append(_f_diag(belief))
    return (belief, traj) if return_trajectory else belief
