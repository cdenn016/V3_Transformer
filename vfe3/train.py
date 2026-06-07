r"""Training (M-step) for VFE_3.0: AdamW per-group learning rates + warmup/cosine.

The model has no neural layers; the only parameters are the PriorBank prior tables.
``loss.backward()`` flows through the unrolled E-step to those tables; AdamW updates
them. The M-step minimizes the cross-entropy of the decode boundary over the prior
tables, with the E-step (the differentiable filtering kernel) unrolled into the graph,
so a gradient step on the priors improves inference end to end. Click-to-run: edit a
``VFE3Config`` and call ``run_training`` (no CLI).
"""

import contextlib
import logging
import math
import time
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

try:                                                # live per-step it/s via a tqdm progress bar
    from tqdm import tqdm as _tqdm                  # (plain tqdm, not tqdm.auto: the notebook
    from tqdm.contrib.logging import (              # widget is swallowed by some Run-button
        logging_redirect_tqdm as _redirect_logging,  # consumers / non-TTY stdout)
    )
except ImportError:                                 # tqdm optional: absent -> no bar, the periodic
    _tqdm = None                                    # log lines still emit at log_interval as before
    _redirect_logging = contextlib.nullcontext

from vfe3.config import VFE3Config
from vfe3.data.datasets import make_dataloader
from vfe3.free_energy import attention_tau
from vfe3.model.model import VFEModel

if TYPE_CHECKING:                                    # avoid an import cycle (run_artifacts imports evaluate)
    from vfe3.run_artifacts import RunArtifacts


def build_optimizer(
    model: VFEModel,
    cfg:   VFE3Config,
) -> torch.optim.Optimizer:
    r"""AdamW with per-group M-step learning rates over the PriorBank prior tables.

    The three prior tables carry distinct natural scales, so each is given its own
    M-step learning rate: the mean table ``mu_embed`` at ``m_mu_lr``; the (log) scale
    tables ``sigma_log_embed`` and the decode temperature ``decode_log_scale`` together
    at ``m_sigma_lr``; the gauge-frame coordinates ``phi_embed`` at ``m_phi_lr``. The
    weight decay ``cfg.weight_decay`` is shared.

    Optional parameters are grouped only when their toggle is on: the linear decode weight
    ``output_proj_weight`` (use_prior_bank=False) at ``m_mu_lr`` (a mean-readout scale); the
    head-mixer ``mixer_delta`` (use_head_mixer=True) at ``m_mu_lr``; the model-channel tables
    ``s_mu_embed``/``s_sigma_log_embed`` (lambda_h>0, gamma_coupling>0, or
    prior_source='model_channel') and the hyper-prior centroid ``r_mu``/``r_sigma_log``
    (lambda_h>0), each split mean@``m_mu_lr`` / log-scale@``m_sigma_lr`` like the belief tables.
    A final assertion pins that the groups cover ``model.parameters()`` EXACTLY -- a new
    parameter that is forgotten here would otherwise silently never receive a gradient.
    The hyper-prior centroid ``r_mu``/``r_sigma_log`` (lambda_h>0) is FROZEN (requires_grad=False, set in
    prior_bank.py) -- a fixed centroid per the manuscript's "higher, slower meta-level"
    (GL(K)_supplementary.tex:1081); the coverage guard exempts it, so it needs no group and is never
    updated (freely training r alongside s would collapse KL(s||r)->0).
    The learned MODEL-level parameters are grouped likewise when their toggle is on: the Regime-II
    edge connection ``connection_W`` (transport_mode='regime_ii') at ``m_phi_lr`` (a gauge-connection
    scale) and the learnable self-coupling ``log_alpha`` (alpha_mode='learnable') at ``m_mu_lr`` -- so
    those sanctioned-NN-exception toggles train rather than tripping the coverage guard.
    """
    pb = model.prior_bank
    groups = [
        {"params": [pb.mu_embed],                              "lr": cfg.m_mu_lr},
        {"params": [pb.sigma_log_embed, pb.decode_log_scale],  "lr": cfg.m_sigma_lr},
        {"params": [pb.phi_embed],                             "lr": cfg.m_phi_lr},
    ]
    if pb.output_proj_weight is not None:                       # use_prior_bank=False linear decode
        groups.append({"params": [pb.output_proj_weight], "lr": cfg.m_mu_lr})
    if getattr(model, "head_mixer", None) is not None:          # use_head_mixer=True Schur mixer
        groups.append({"params": list(model.head_mixer.parameters()), "lr": cfg.m_mu_lr})
    if getattr(model, "pos_phi_free", None) is not None:        # pos_phi='learned' positional table
        groups.append({"params": [model.pos_phi_free], "lr": cfg.m_phi_lr})  # a gauge-frame scale
    if getattr(pb, "s_mu_embed", None) is not None:             # model-channel s tables (gamma_coupling>0 or
        groups.append({"params": [pb.s_mu_embed],        "lr": cfg.m_mu_lr})    # prior_source=model_channel):
        groups.append({"params": [pb.s_sigma_log_embed], "lr": cfg.m_sigma_lr})  # mean@m_mu_lr, log-scale@
        # m_sigma_lr, mirroring the belief tables. s is the model channel / (under model_channel) the
        # live belief prior, so it must train. The hyper-prior CENTROID r (lambda_h>0) is NOT grouped
        # because it is FROZEN (requires_grad=False, prior_bank.py) -- a fixed centroid per the
        # manuscript's "higher, slower meta-level"; the coverage guard exempts non-trainable params.
    if getattr(model, "connection_W", None) is not None:        # transport_mode='regime_ii' learned
        groups.append({"params": [model.connection_W], "lr": cfg.m_phi_lr})  # connection -> gauge LR
    if getattr(model, "log_alpha", None) is not None:           # alpha_mode='learnable' scalar coupling
        groups.append({"params": [model.log_alpha], "lr": cfg.m_mu_lr})
    if getattr(model, "log_lambda_beta", None) is not None:     # learnable_lambda_beta scalar belief-coupling weight
        groups.append({"params": [model.log_lambda_beta], "lr": cfg.m_phi_lr})  # a coupling/gauge-scale LR

    # Exact-coverage guard: every TRAINABLE model parameter (requires_grad=True) must land in exactly
    # one group. A missing group would leave that weight frozen (no AdamW update) with no error -- the
    # bug class the optimizer is most prone to as new learnable seams (output_proj, head mixer, ...) are
    # added. Non-trainable params (requires_grad=False, e.g. the FROZEN hyper-prior centroid r) are
    # intentionally exempt: they are fixed by design and need no optimizer group.
    # NOTE: this guards GROUPING/coverage, not gradient FLOW. A grouped parameter can still receive
    # a null gradient under specific opt-in toggles, by design: phi_embed under detach_e_step=True
    # (the E-step is detached; test-pinned in test_model.py), decode_log_scale under
    # use_prior_bank=False (the linear decode discards tau_eff), ALL encode tables under
    # use_prior_bank=False AND detach_e_step=True (only output_proj_weight reaches the loss; the
    # model emits a warning for that combination), and mu_embed/sigma_log_embed under
    # prior_source='model_channel' (the prior reroutes to the s tables, so the belief tables are dead
    # but stay grouped -- AdamW skips a None-grad param, though shared weight_decay still decays the
    # dead table harmlessly since it is never read). These are intentional, not coverage bugs.
    grouped = {p for g in groups for p in g["params"]}
    missing = {p for p in model.parameters() if p.requires_grad} - grouped   # frozen params are exempt
    if missing:
        raise AssertionError(
            f"build_optimizer left {len(missing)} model parameter(s) ungrouped; they would never "
            f"train. Add them to a param group."
        )

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


def _default_sample_decoder(
    cfg: VFE3Config,
) -> 'Optional[Callable[[Sequence[int]], str]]':
    r"""A best-guess tiktoken ``decode(ids) -> str`` from ``cfg.vocab_size``, or None.

    Activates ONLY for a recognized real-corpus tokenizer vocab -- gpt2 (~50257) or cl100k
    (~100277) -- so a click-to-run on wikitext-*/wiki-* prints sample text with no wiring, while
    a tiny synthetic/test vocab (e.g. 6) gets no decoder and stays silent (the pure path is
    preserved without an extra toggle). The ranges tolerate vocab padding. Lazy-imports tiktoken;
    returns None if it is absent. An explicit ``sample_decode`` argument always takes precedence."""
    try:
        import tiktoken
    except ImportError:
        return None
    if 40_000 <= cfg.vocab_size <= 60_000:
        enc = tiktoken.get_encoding("gpt2")
    elif 90_000 <= cfg.vocab_size <= 110_000:
        enc = tiktoken.get_encoding("cl100k_base")
    else:
        return None
    return lambda ids: enc.decode([int(t) for t in ids])


def train_step(
    model:     VFEModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    tokens:    torch.Tensor,             # (B, N) input token ids
    targets:   torch.Tensor,             # (B, N) next-token ids (-100 = ignore)

    *,
    grad_clip:        float = 1.0,
    grad_accum_steps: int   = 1,
) -> float:
    r"""One M-step (one optimizer step) on the cross-entropy of a batch; returns the loss.

    Zeroes the prior-table gradients, runs the forward (encode -> unrolled E-step ->
    decode -> CE), backpropagates the loss through inference to the prior tables, clips
    the global gradient norm to ``grad_clip``, then takes one AdamW + scheduler step.

    With ``grad_accum_steps == K > 1`` the batch is split into ``K`` equal chunks along
    the batch axis; each chunk's loss is divided by ``K`` and ``backward()``-ed,
    ACCUMULATING into ``.grad``, and the single clip + ``optimizer.step()`` +
    ``scheduler.step()`` fires once after all ``K`` microbatches. Because the model's CE
    and the extra F terms are MEANS over the batch axis and there is no cross-sequence
    dependency, the accumulated ``.grad`` equals (to round-off) the gradient of one
    backward on the full batch when the microbatches carry EQUAL counted-token counts
    (i.e. ``B % K == 0`` and no per-position ``ignore_index`` re-weighting); this gives a
    larger EFFECTIVE batch without the memory of one big forward. A "step" stays an
    OPTIMIZER step (the scheduler/warmup/max_steps accounting is unchanged). The grad-clip
    is applied ONCE to the accumulated (already mean-normalized) gradient at the boundary,
    so the threshold is NOT rescaled by ``K``. The returned loss is the mean over the
    ``K`` microbatches (the accumulation-boundary loss). ``K == 1`` is byte-identical to
    the single-backward path (no chunking, no divide). Requires ``B % K == 0``.
    """
    optimizer.zero_grad(set_to_none=True)
    if grad_accum_steps == 1:                                   # default path: byte-identical to the single-step loop
        _, loss, _ = model(tokens, targets)
        loss.backward()
        step_loss = float(loss.detach())
    else:
        if tokens.shape[0] % grad_accum_steps != 0:            # equal-token microbatches require an even split
            raise ValueError(
                f"grad_accum_steps={grad_accum_steps} must divide the batch size "
                f"{tokens.shape[0]} for equal microbatches; got remainder "
                f"{tokens.shape[0] % grad_accum_steps}."
            )
        tok_chunks = torch.chunk(tokens, grad_accum_steps, dim=0)
        tgt_chunks = torch.chunk(targets, grad_accum_steps, dim=0)
        step_loss = 0.0
        for tok_mb, tgt_mb in zip(tok_chunks, tgt_chunks):
            _, loss_mb, _ = model(tok_mb, tgt_mb)
            (loss_mb / grad_accum_steps).backward()            # accumulate the mean-normalized microbatch grad
            step_loss += float(loss_mb.detach()) / grad_accum_steps
    if grad_clip is not None and grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    scheduler.step()
    return step_loss


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
    artifacts:     Optional["RunArtifacts"] = None,

    generate_samples:  bool                                     = True,   # False -> pure silent path (no sample text)
    sample_decode:     Optional[Callable[[Sequence[int]], str]] = None,   # token-ids -> text; None -> auto by vocab
    sample_new_tokens: int                                      = 40,     # greedy continuation length
    sample_prompt_len: int                                      = 6,     # seq-0 prompt length to continue
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
    steps, off the training graph), AND -- when ``tqdm`` is installed -- the step loop runs
    under a ``tqdm`` progress bar whose built-in rate readout shows live ``it/s`` every step
    (the formatted lines render above it via ``logging_redirect_tqdm``); when ``eval_interval``
    is positive and ``val_loader`` is given a validation block is emitted every
    ``eval_interval`` steps.
    """
    optimizer = build_optimizer(model, cfg)
    # Warmup/cosine multiplier, floored so each group's ABSOLUTE LR never decays below cfg.min_lr.
    # LambdaLR scales each group's base LR by its lambda; per-group lambdas (one per base LR) floor
    # the product: base * max(min_lr/base, cosine) = max(min_lr, base * cosine). With cfg.min_lr=0.0
    # this is exactly the pure half-cosine-to-zero (the theoretically pure path). The (lambda base: ...)
    # closure captures each base LR by value, dodging late-binding over the loop variable.
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    sched_lambdas = [
        (lambda base: lambda s: max(cfg.min_lr / base, lr_lambda(s, cfg)))(b)
        for b in base_lrs
    ]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, sched_lambdas)
    losses: List[float] = []
    model.train()
    logger = logger or logging.getLogger(__name__)
    if device is None:
        device = model.prior_bank.mu_embed.device
    # Live per-step it/s: iterate the step loop through a tqdm bar whose built-in rate readout
    # refreshes every step. Gated on log_interval so the documented silent path (log_interval
    # falsy) stays bitwise-identical -- no bar, no redirect, nothing printed. The generator holds
    # logging_redirect_tqdm open across the whole loop (it suspends at `yield` INSIDE the `with`),
    # so the periodic logger.info lines below render above the bar instead of interleaving with it
    # on stderr; it closes the bar on normal exit or exception.
    show_bar = bool(log_interval) and _tqdm is not None

    def _step_indices() -> Iterable[int]:
        if not show_bar:
            yield from range(n_steps)
            return
        bar = _tqdm(range(n_steps), desc="Training", total=n_steps, ascii=True)  # ascii=True: the
        #                          default block glyph U+2588 is not cp1252-encodable on a Windows
        #                          console (raises UnicodeEncodeError mid-run); " #" renders anywhere
        with _redirect_logging():
            try:
                yield from bar
            finally:
                bar.close()

    it = iter(loader)
    win_t0 = time.perf_counter()
    win_i0 = 0
    last_val: Dict[str, float] = {}                  # most recent validation, carried into each CSV row
    for step in _step_indices():
        try:
            tokens, targets = next(it)
        except StopIteration:
            it = iter(loader)
            tokens, targets = next(it)
        tokens = tokens.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        losses.append(train_step(model, optimizer, scheduler, tokens, targets,
                                  grad_clip=grad_clip, grad_accum_steps=cfg.grad_accum_steps))

        do_log  = bool(log_interval) and (step + 1) % log_interval == 0
        do_eval = bool(eval_interval) and val_loader is not None and (step + 1) % eval_interval == 0
        do_csv  = artifacts is not None and (do_log or do_eval)

        if do_log or do_csv:                                 # converged CE + diagnostics (off graph), ONCE
            with torch.no_grad():
                _, _, ce_t = model(tokens, targets)          # true CE (nats)
            ce = float(ce_t)
            d = model.diagnostics(tokens)

        if do_log:
            rate = (step + 1 - win_i0) / max(time.perf_counter() - win_t0, 1e-9)
            logger.info(
                "Step %d/%d | Loss: %.4f | CE: %.4f | H(b): %.3f | it/s: %.2f | \n\n         Train PPL: %.1f \n",
                step + 1, n_steps, losses[-1], ce, d["attn_entropy"], rate, math.exp(min(ce, 20.0)),
            )
            logger.info(
                "    F: self %.4f | belief %.4f | entropy %.4f | total %.4f | eff_rank %.2f | BPC %.4f",
                d["self_coupling"], d["belief_coupling"], d["attention_entropy"],
                d["total"], d["effective_rank"], ce / math.log(2.0),
            )
            win_t0 = time.perf_counter()
            win_i0 = step + 1

        if do_eval:
            m = evaluate(model, val_loader, max_batches=cfg.eval_max_batches, device=device)
            logger.info(" \n Validation @ step %d:", step + 1)
            logger.info(                                         # val has no separate loss; CE is the loss
                "\n       CE: %.4f \n      Val PPL: %.1f \n       BPC: %.4f \n\n",
                m["ce"], m["ppl"], m["bpc"],
            )
            # Sample text directly below the BPC value. ``generate_samples=False`` forces the pure
            # silent path (no generation, no Sample line). Otherwise the decoder is an explicit
            # ``sample_decode`` if given, else an AUTO-DEFAULT picked from cfg.vocab_size (gpt2 /
            # cl100k) -- so a real click-to-run prints samples with no wiring, while tiny
            # synthetic/test vocabs get no decoder. When a decoder exists, greedily continue seq 0 of
            # the live batch by sample_new_tokens and decode prompt + continuation. Best-effort: a
            # generation/decode error is logged, never fatal (model.generate is @torch.no_grad).
            decode = None if not generate_samples else (
                sample_decode if sample_decode is not None else _default_sample_decoder(cfg))
            if decode is not None:
                try:
                    prompt = tokens[:1, :sample_prompt_len]                       # (1, P) seq-0 prompt
                    gen = model.generate(prompt, sample_new_tokens, greedy=True)[0]
                    p_txt = decode(prompt[0].tolist())
                    c_txt = decode(gen[prompt.shape[1]:].tolist())
                    logger.info("       Sample: %r  ->  %r\n", p_txt, c_txt)
                except Exception as exc:                                          # never let sampling kill training
                    logger.warning("       (sample generation failed: %s)", exc)
            last_val = {"ce": m["ce"], "ppl": m["ppl"], "bpc": m["bpc"]}
            if artifacts is not None:
                artifacts.maybe_save_best(step + 1, model, m["ppl"])
                # Per-layer/per-head attention heatmap grid for this eval (off the graph, seq 0 of
                # the live batch). Best-effort inside save_attention_maps: a viz error is logged,
                # never fatal to the run. Kept at EVAL cadence (one grid per eval, not per log).
                artifacts.save_attention_maps(step + 1, model.attention_maps(tokens), logger=logger)

        # Persistence is opt-in: with no artifacts object do_csv is False, so the silent/in-memory
        # path is unchanged. A metrics.csv row is written every LOG_INTERVAL (and every eval) -- the
        # dense per-step diagnostics off the graph -- with the most recent validation carried forward
        # (NaN until the first eval; fresh on a step where the eval above just ran). The four F-stack
        # diagnostics are per-sequence SUMS over seq 0, normalized to PER TOKEN so they are
        # commensurate with val_ce, a token-weighted mean (nats/token; see audit-2026-06-05 Finding 2).
        if do_csv:
            n_tok = max(int(tokens.shape[1]), 1)
            lrs = scheduler.get_last_lr()                     # per-group current LR (groups 0,1,2 = mu,sigma,phi)
            row = {
                "step":              step + 1,
                "train_loss":        losses[-1],
                "train_ce":          ce,                      # true CE (nats), off the graph
                "train_ppl":         math.exp(min(ce, 20.0)),  # train perplexity = exp(CE), mirrors the console line
                "lr_mu":             float(lrs[0]),           # group 0 = mu_embed          (m_mu_lr)
                "lr_sigma":          float(lrs[1]),           # group 1 = sigma_log+decode  (m_sigma_lr)
                "lr_phi":            float(lrs[2]),           # group 2 = phi_embed         (m_phi_lr)
                "val_ce":            last_val.get("ce",  float("nan")),
                "val_ppl":           last_val.get("ppl", float("nan")),
                "val_bpc":           last_val.get("bpc", float("nan")),
                "attn_entropy":       d["attn_entropy"],
                "self_coupling":      d["self_coupling"]     / n_tok,
                "belief_coupling":    d["belief_coupling"]   / n_tok,
                "attention_entropy":  d["attention_entropy"] / n_tok,
                "free_energy_total":  d["total"]             / n_tok,
                "effective_rank":     d["effective_rank"],
                "holonomy_deviation": d["holonomy_deviation"],
                "gauge_trace_spread": d["gauge_trace_spread"],
            }
            # Learnable belief-coupling weight: record lambda_beta = exp(log_lambda_beta) so its
            # trajectory lands in metrics.csv (and the figure). The column exists only on a learnable
            # run (config is fixed per run, so the CSV stays rectangular).
            _llb = getattr(model, "log_lambda_beta", None)
            if _llb is not None:
                row["lambda_beta"] = float(_llb.detach().exp())
            artifacts.log_metrics(row)

        # Periodic resumable checkpoint (opt-in; needs the artifacts dir and the optimizer state).
        if (artifacts is not None and cfg.checkpoint_interval
                and (step + 1) % cfg.checkpoint_interval == 0):
            artifacts.save_checkpoint(step + 1, model, optimizer, cfg)
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
        f" VFE: alpha={cfg.alpha}  kappa={cfg.kappa}  "
        f"tau={attention_tau(cfg.kappa, model.group.irrep_dims):.4f}  mass_phi={cfg.mass_phi}",
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

    DEPRECATED / minimal: superseded by ``train_vfe3.main()``, which is the canonical entry
    point. This helper passes no ``artifacts`` (so ``checkpoint_interval``/best-model/CSV are
    never written), reuses ``loader`` as the validation loader (train == val), and never runs
    the end-of-run test eval. Kept only for the lightweight in-process smoke use it already had.
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
