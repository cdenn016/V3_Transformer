r"""The VFE block stack for VFE_3.0: L blocks with the belief handoff mu_q -> mu_p.

After each block the updated belief becomes (a blend toward) the next block's prior:
mu_p_next = (1 - rho) mu_p + rho mu_q (rho = prior_handoff_rho); sigma_p frozen at the
embedding by default; phi flows through the belief, not the prior.
"""

from typing import Any, Optional

import torch

from vfe3.belief import BeliefState
from vfe3.config import VFE3Config
from vfe3.geometry.groups import GaugeGroup
from vfe3.model.block import vfe_block


def vfe_stack(
    belief:     BeliefState,
    mu_p:       torch.Tensor,             # (N, K) initial prior means
    sigma_p:    torch.Tensor,             # (N, K) initial prior variances
    group:      GaugeGroup,
    cfg:        VFE3Config,

    *,
    log_prior:    Optional[torch.Tensor]    = None,
    block_norm:   Optional[Any]             = None,   # cached norm instance (None -> no block norm)
    log_alpha:    Optional[torch.Tensor]    = None,   # learned scalar self-coupling (None -> pure path)
    connection_W: Optional[torch.Tensor]    = None,   # learned bilinear connection for regime_ii (NN exception; None -> pure path)
) -> BeliefState:
    r"""Run L = cfg.n_layers blocks, handing the belief mean off to the next prior.

    ``log_alpha`` is the model's learned scalar self-coupling parameter under
    alpha_mode='learnable' (a sanctioned nn.Parameter NN exception; alpha = exp(log_alpha)),
    forwarded to the E-step; None on every pure no-NN alpha_mode (the default path). ``connection_W``
    is the model's learned bilinear Regime-II connection (a sanctioned NN exception) forwarded under
    transport_mode='regime_ii'; None on the pure (flat) path."""
    rho = cfg.prior_handoff_rho
    rho_s = cfg.prior_handoff_sigma
    for _ in range(cfg.n_layers):
        belief = vfe_block(belief, mu_p, sigma_p, group, cfg, log_prior=log_prior,
                           block_norm=block_norm, log_alpha=log_alpha, connection_W=connection_W)
        mu_p = (1.0 - rho) * mu_p + rho * belief.mu
        sigma_p = (1.0 - rho_s) * sigma_p + rho_s * belief.sigma
    return belief
