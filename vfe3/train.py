r"""Training (M-step) for VFE_3.0: AdamW per-group learning rates + warmup/cosine.

The model has no neural layers; the only parameters are the PriorBank prior tables.
``loss.backward()`` flows through the unrolled E-step to those tables; AdamW updates
them. The M-step minimizes the cross-entropy of the decode boundary over the prior
tables, with the E-step (the differentiable filtering kernel) unrolled into the graph,
so a gradient step on the priors improves inference end to end. Click-to-run: edit a
``VFE3Config`` and call ``run_training`` (no CLI).
"""

import math
from typing import List, Optional, Tuple

import torch

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel


def build_optimizer(
    model: VFEModel,
    cfg:   VFE3Config,
) -> torch.optim.Optimizer:
    r"""AdamW with per-group M-step learning rates over the PriorBank prior tables.

    The three prior tables carry distinct natural scales, so each is given its own
    M-step learning rate: the mean table ``mu_embed`` at ``m_mu_lr``; the (log) scale
    tables ``sigma_log_embed`` and the decode temperature ``decode_log_scale`` together
    at ``m_sigma_lr``; the gauge-frame coordinates ``phi_embed`` at ``m_phi_lr``. The
    weight decay ``cfg.weight_decay`` is shared. These are the only parameters in the
    model (no neural layers), so the grouping covers ``model.parameters()`` exactly.
    """
    pb = model.prior_bank
    groups = [
        {"params": [pb.mu_embed],                              "lr": cfg.m_mu_lr},
        {"params": [pb.sigma_log_embed, pb.decode_log_scale],  "lr": cfg.m_sigma_lr},
        {"params": [pb.phi_embed],                             "lr": cfg.m_phi_lr},
    ]
    return torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)


def lr_lambda(
    step: int,
    cfg:  VFE3Config,
) -> float:
    r"""Learning-rate multiplier: linear warmup then cosine decay.

    Linear warmup to 1.0 over ``warmup_steps``, then a half-cosine to 0.0 at
    ``max_steps``::

        lr_mult(t) = t / warmup_steps                                 t <  warmup_steps
                   = 0.5 (1 + cos(pi * (t - warmup) / (max - warmup))) t >= warmup_steps

    The cosine argument is clamped to [0, pi] so steps beyond ``max_steps`` stay at 0.
    """
    if step < cfg.warmup_steps:
        return step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
