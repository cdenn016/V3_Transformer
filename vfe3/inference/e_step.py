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
from vfe3.free_energy import attention_weights, free_energy, pairwise_energy, self_divergence
from vfe3.geometry.groups import GaugeGroup
from vfe3.geometry.phi_preconditioner import precondition_phi_gradient
from vfe3.geometry.retraction import natural_gradient, retract_phi, retract_spd_diagonal
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

    include_attention_entropy: bool = True,
    family:                    str  = "gaussian_diagonal",
    alpha_mode:                str  = "constant",

    log_prior:                 Optional[torch.Tensor] = None,
    keys:                      Optional[BeliefState]  = None,   # None -> global F; else keys frozen at `keys`
    **kwargs,                                                   # accept-and-ignore iteration-only knobs
) -> torch.Tensor:                   # scalar F
    r"""Scalar free energy of a belief. ``keys=None`` -> global F (keys = the belief);
    ``keys`` given -> F with the transported keys frozen at ``keys`` (the F_filt objective).

        F = Sum_i [ alpha_i D(q_i||p_i) (+ R(alpha_i))
                  + Sum_j beta_ij E_ij + tau Sum_j beta_ij log(beta_ij/pi_ij) ],
        E_ij = D(q_i || Omega_ij q_j),  beta = softmax_j(log_prior - E/tau).

    The shared ``**kwargs`` sink swallows the iteration-only knobs (gradient_mode,
    e_mu_lr/e_sigma_lr/e_phi_lr, sigma_max, e_sigma_q_trust, phi_precond_mode) so the
    common call site in ``e_step`` may forward one knob bag to both this and
    ``e_step_iteration`` without error; the shared knobs bind to real parameters here.
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
