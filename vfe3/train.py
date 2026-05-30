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


def train_step(
    model:     VFEModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    tokens:    torch.Tensor,             # (B, N) input token ids
    targets:   torch.Tensor,             # (B, N) next-token ids (-100 = ignore)

    *,
    grad_clip: float = 1.0,
) -> float:
    r"""One M-step on the cross-entropy of a batch; returns the loss scalar.

    Zeroes the prior-table gradients, runs the forward (encode -> unrolled E-step ->
    decode -> CE), backpropagates the loss through inference to the prior tables, clips
    the global gradient norm to ``grad_clip``, then takes one AdamW + scheduler step.
    """
    optimizer.zero_grad(set_to_none=True)
    _, loss, _ = model(tokens, targets)
    loss.backward()
    if grad_clip is not None and grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    scheduler.step()
    return float(loss.detach())


def train(
    model:  VFEModel,
    loader,
    cfg:    VFE3Config,

    *,
    n_steps:   int   = 100,
    grad_clip: float = 1.0,
) -> List[float]:
    r"""Train ``n_steps`` M-step iterations (cycling the loader); return the loss history.

    Builds the per-group AdamW optimizer and the warmup/cosine ``LambdaLR``, then takes
    ``n_steps`` gradient steps, re-iterating the loader when it is exhausted. The loss
    history is the per-step cross-entropy; the cutover criterion is that it decreases.
    """
    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_lambda(s, cfg))
    losses: List[float] = []
    model.train()
    it = iter(loader)
    for _ in range(n_steps):
        try:
            tokens, targets = next(it)
        except StopIteration:
            it = iter(loader)
            tokens, targets = next(it)
        tokens = tokens.to(model.prior_bank.mu_embed.device)
        targets = targets.to(model.prior_bank.mu_embed.device)
        losses.append(train_step(model, optimizer, scheduler, tokens, targets, grad_clip=grad_clip))
    return losses


def run_training(
    cfg:     VFE3Config,
    dataset: str = "wikitext-2",
    split:   str = "train",

    *,
    n_steps:    int           = 1000,
    max_tokens: Optional[int] = None,
) -> Tuple[VFEModel, List[float]]:
    r"""Click-to-run entry: build a model + a cached dataloader by name and train (no CLI).

    Constructs a ``VFEModel`` from ``cfg``, a causal-LM dataloader from the tokenized
    cache for ``dataset``/``split`` (capped at ``max_tokens`` for fast runs), and trains
    for ``n_steps`` M-steps. Returns the trained model and its loss history.
    """
    from vfe3.data.datasets import make_dataloader

    model = VFEModel(cfg)
    loader = make_dataloader(dataset, split, cfg.max_seq_len, cfg.batch_size, max_tokens=max_tokens)
    losses = train(model, loader, cfg, n_steps=n_steps)
    return model, losses
