r"""Training (M-step) for VFE_3.0: AdamW per-group learning rates + warmup/cosine.

The model has no neural layers; the only parameters are the PriorBank prior tables.
``loss.backward()`` flows through the unrolled E-step to those tables; AdamW updates
them. The M-step minimizes the cross-entropy of the decode boundary over the prior
tables, with the E-step (the differentiable filtering kernel) unrolled into the graph,
so a gradient step on the priors improves inference end to end. Click-to-run: edit a
``VFE3Config`` and call ``run_training`` (no CLI).
"""

import logging
import math
import time
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from vfe3.config import VFE3Config
from vfe3.data.datasets import make_dataloader
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
    # fused AdamW (one CUDA kernel for the whole M-step) when the priors live on CUDA; it is
    # CUDA-only, so on a CPU box this is the standard AdamW. Per-group LRs are honored either way.
    use_fused = pb.mu_embed.is_cuda
    return torch.optim.AdamW(groups, weight_decay=cfg.weight_decay, fused=use_fused)


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


@torch.no_grad()
def evaluate(
    model:  VFEModel,
    loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],   # yields (tokens, targets) batches

    *,
    max_batches: Optional[int]          = None,
    device:      Optional[torch.device] = None,
) -> Dict[str, float]:
    r"""Token-weighted corpus evaluation. Returns ``{ce, ppl, bpc}`` (CE in nats).

    .. math::
        \mathrm{CE} = \frac{\sum_b n_b\, \mathrm{ce}_b}{\sum_b n_b},\quad
        \mathrm{PPL} = e^{\min(\mathrm{CE},\,20)},\quad
        \mathrm{BPC} = \mathrm{CE} / \ln 2,

    with ``n_b`` the number of non-ignored (``!= -100``) target tokens in batch ``b``.
    Aggregating by token count (not per-batch mean) reproduces one cross-entropy over
    the concatenated corpus, including a partial last batch.
    """
    if device is None:
        device = model.prior_bank.mu_embed.device
    was_training = model.training
    model.eval()
    try:
        total_nats = 0.0
        total_tok = 0
        for i, (tokens, targets) in enumerate(loader):
            tokens = tokens.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            _, _, ce = model(tokens, targets)
            n_b = int((targets != -100).sum())
            total_nats += float(ce) * n_b
            total_tok += n_b
            if max_batches is not None and i + 1 >= max_batches:
                break               # draw exactly max_batches (process-then-break; no extra pull)
        ce = total_nats / max(total_tok, 1)
    finally:
        if was_training:
            model.train()
    return {
        "ce":  ce,
        "ppl": math.exp(min(ce, 20.0)),
        "bpc": ce / math.log(2.0),
    }


def train(
    model:  VFEModel,
    loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],   # yields (tokens, targets) batches
    cfg:    VFE3Config,

    *,
    n_steps:   int   = 100,
    grad_clip: float = 1.0,

    log_interval:  Optional[int]            = None,
    eval_interval: Optional[int]            = None,
    val_loader:    Optional[Iterable]       = None,
    device:        Optional[torch.device]   = None,
    logger:        Optional[logging.Logger] = None,
) -> List[float]:
    r"""Train ``n_steps`` M-step iterations (cycling the loader); return the loss history.

    Builds the per-group AdamW optimizer and the warmup/cosine ``LambdaLR``, then takes
    ``n_steps`` gradient steps, re-iterating the loader when it is exhausted. The loss
    history is the per-step cross-entropy; the cutover criterion is that it decreases.

    With ``log_interval`` falsy (``None`` or ``0``) and ``eval_interval`` falsy the loop
    is bitwise-identical to the silent path: the two truthiness-guarded blocks
    short-circuit, drawing no RNG, running no extra forward, and printing nothing. When
    ``log_interval`` is positive a VFE_2.0-style per-step line is emitted every
    ``log_interval`` steps (CE and diagnostics recomputed under ``no_grad`` only at those
    steps, off the training graph); when ``eval_interval`` is positive and ``val_loader``
    is given a validation block is emitted every ``eval_interval`` steps.
    """
    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_lambda(s, cfg))
    losses: List[float] = []
    model.train()
    logger = logger or logging.getLogger(__name__)
    if device is None:
        device = model.prior_bank.mu_embed.device
    it = iter(loader)
    win_t0 = time.perf_counter()
    win_i0 = 0
    for step in range(n_steps):
        try:
            tokens, targets = next(it)
        except StopIteration:
            it = iter(loader)
            tokens, targets = next(it)
        tokens = tokens.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        losses.append(train_step(model, optimizer, scheduler, tokens, targets, grad_clip=grad_clip))

        if log_interval and (step + 1) % log_interval == 0:
            with torch.no_grad():
                _, _, ce_t = model(tokens, targets)         # true CE (nats), off graph
            ce = float(ce_t)
            d = model.diagnostics(tokens)
            rate = (step + 1 - win_i0) / max(time.perf_counter() - win_t0, 1e-9)
            logger.info(
                "Step %d/%d | Loss: %.4f | CE: %.4f | H(b): %.3f | it/s: %.2f | PPL: %.1f",
                step + 1, n_steps, losses[-1], ce, d["attn_entropy"], rate, math.exp(min(ce, 20.0)),
            )
            logger.info(
                "    F: self %.4f | belief %.4f | entropy %.4f | total %.4f | eff_rank %.2f | BPC %.4f",
                d["self_coupling"], d["belief_coupling"], d["attention_entropy"],
                d["total"], d["effective_rank"], ce / math.log(2.0),
            )
            win_t0 = time.perf_counter()
            win_i0 = step + 1

        if eval_interval and val_loader is not None and (step + 1) % eval_interval == 0:
            m = evaluate(model, val_loader, max_batches=cfg.eval_max_batches, device=device)
            logger.info("  Validation @ step %d:", step + 1)
            logger.info(
                "    Loss: %.4f | CE: %.4f | PPL: %.1f | BPC: %.4f",
                m["ce"], m["ce"], m["ppl"], m["bpc"],
            )
    return losses


def _banner(model: VFEModel, cfg: VFE3Config, dataset: str, device: torch.device, n_steps: int) -> str:
    r"""Compact VFE_2.0-style init banner (no FLOPs counter, no lambda_h: V3 has neither)."""
    n_params = sum(p.numel() for p in model.parameters())
    bar = "=" * 64
    return "\n".join([
        bar,
        f" Gauge VFE Transformer | {n_params} params | {device}",
        bar,
        f" K={cfg.embed_dim}  N={cfg.max_seq_len}  L={cfg.n_layers}  heads={cfg.n_heads}  "
        f"group={cfg.gauge_group}  family={cfg.family}",
        f" steps={n_steps}  batch={cfg.batch_size}  dataset={dataset}",
        f" M-LRs: mu={cfg.m_mu_lr}  sigma={cfg.m_sigma_lr}  phi={cfg.m_phi_lr}",
        f" VFE: alpha={cfg.alpha}  kappa={cfg.kappa}  tau={cfg.tau:.4f}  mass_phi={cfg.mass_phi}",
        f" seed={cfg.seed}",
        bar,
    ])


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
    cache for ``dataset``/``split`` (capped at ``max_tokens`` for fast runs), prints the
    init banner, and trains for ``n_steps`` M-steps with the config-selected console
    logging (``cfg.log_interval``, ``cfg.eval_interval``). Returns the trained model and
    its loss history.
    """
    torch.manual_seed(cfg.seed)              # reproducible prior-table init + data order
    model = VFEModel(cfg)
    device = model.prior_bank.mu_embed.device
    loader = make_dataloader(dataset, split, cfg.max_seq_len, cfg.batch_size, max_tokens=max_tokens)
    logger = logging.getLogger(__name__)
    logger.info(_banner(model, cfg, dataset, device, n_steps))
    losses = train(
        model, loader, cfg,
        n_steps=n_steps,
        log_interval=cfg.log_interval,
        eval_interval=cfg.eval_interval,
        val_loader=loader,
        device=device,
    )
    return model, losses
