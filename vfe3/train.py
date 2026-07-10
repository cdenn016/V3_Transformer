r"""Training (M-step) for VFE_3.0: AdamW per-group learning rates + warmup/cosine.

The model has no neural layers (no nn.Linear/MLP/activation). The trainable parameters are the
PriorBank prior tables plus the model-owned tables their toggles create -- the default
``pos_phi='learned'`` positional table, and the default-OFF exceptions (head mixer, regime_ii
connection, learnable T5 bias, linear decode).
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
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

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
from vfe3.ema import EMA
from vfe3.free_energy import attention_tau
from vfe3.model.block import _as_coeff
from vfe3.model.model import VFEModel
from vfe3.run_artifacts import RunArtifacts          # top-level safe: run_artifacts imports evaluate
#                                                      lazily (function-local), so there is no cycle
from vfe3.runtime import seed_everything
from vfe3.geometry.transport import TRANSPORT_CLAMP_MAX_NORM   # single source for the phi-clamp threshold (M2)


_PHI_CLAMP_WARNED: bool = False


def _warn_phi_transport_clamp(
    model:    VFEModel,

    max_norm: float = TRANSPORT_CLAMP_MAX_NORM,   # single source: stable_matrix_exp_pair's Frobenius clamp
) -> None:
    r"""Warn ONCE when a gauge-frame table's embedded Frobenius norm exceeds the transport clamp.

    ``stable_matrix_exp_pair`` rescales any ``||M||_F > max_norm`` (default 15) and returns the
    surrogate ``exp(max_norm * M/||M||_F)``, NOT ``exp(M)``; its per-call monitor is opt-in because
    it costs a host sync on the E-step hot path. The M-step, however, steps ``phi_embed`` /
    ``pos_phi_free`` with NO trust region (``GaugeNaturalGradAdamW`` / plain AdamW), so a drifting
    row silently enters the surrogate regime (audit 2026-07-05 m8). This check runs only on
    log/eval-cadence steps (off the hot path) and uses the Gram form
    ``||sum_a phi^a G_a||_F^2 = phi^T Gram phi``, ``Gram_ab = tr(G_a^T G_b)`` -- one (rows, n_gen)
    matmul, never materializing the (rows, K, K) embedding.
    """
    global _PHI_CLAMP_WARNED
    if _PHI_CLAMP_WARNED:
        return
    gen = model.group.generators                       # (n_gen, K, K)
    tables = [("phi_embed", getattr(model.prior_bank, "phi_embed", None)),
              ("pos_phi_free", getattr(model, "pos_phi_free", None))]
    with torch.no_grad():
        gram = torch.einsum("aij,bij->ab", gen, gen)   # (n_gen, n_gen) = tr(G_a^T G_b)
        for name, tab in tables:
            if tab is None:
                continue
            phi = tab.reshape(-1, tab.shape[-1]).to(gram.dtype)
            frob2_max = ((phi @ gram) * phi).sum(-1).max()
            if bool(frob2_max > max_norm ** 2):        # host sync: log-cadence only
                import warnings
                warnings.warn(
                    f"{name}: embedded gauge-frame Frobenius norm "
                    f"{float(frob2_max.clamp(min=0.0).sqrt()):.2f} exceeds the transport clamp "
                    f"max_norm={max_norm}; stable_matrix_exp_pair now returns the clamped surrogate "
                    f"exp(max_norm*M/||M||), not exp(M). Bound the frame (mass_phi, lower m_phi_lr) "
                    f"or accept the surrogate transport. Warned once; further drift is not re-reported.",
                    RuntimeWarning, stacklevel=2,
                )
                _PHI_CLAMP_WARNED = True
                return


def build_optimizer(
    model: VFEModel,
    cfg:   VFE3Config,
) -> torch.optim.Optimizer:
    r"""AdamW with per-group M-step learning rates over the PriorBank prior tables.

    The three prior tables carry distinct natural scales, so each is given its own
    M-step learning rate: the mean table ``mu_embed`` at ``m_p_mu_lr``; the (log) scale
    tables ``sigma_log_embed`` and the decode temperature ``decode_log_scale`` together
    at ``m_p_sigma_lr``; the gauge-frame coordinates ``phi_embed`` at ``m_phi_lr``. The
    weight decay ``cfg.weight_decay`` is shared.

    Optional parameters are grouped only when their toggle is on: the linear decode weight
    ``output_proj_weight`` (use_prior_bank=False) at ``m_p_mu_lr`` (a mean-readout scale); the
    head-mixer ``mixer_delta`` (use_head_mixer=True) at ``m_p_mu_lr``; the model-channel tables
    ``s_mu_embed``/``s_sigma_log_embed`` (lambda_h>0, lambda_gamma>0, or
    prior_source='model_channel') and the hyper-prior centroid ``r_mu``/``r_sigma_log``
    (lambda_h>0), each split mean@``m_p_mu_lr`` / log-scale@``m_p_sigma_lr`` like the belief tables.
    A final assertion pins that the groups cover ``model.parameters()`` EXACTLY -- a new
    parameter that is forgotten here would otherwise silently never receive a gradient.
    The hyper-prior centroid ``r_mu``/``r_sigma_log`` (lambda_h>0) is FROZEN by default
    (requires_grad=False, set in prior_bank.py) -- a fixed centroid per the manuscript's "higher, slower
    meta-level" (GL(K)_supplementary.tex:1081); the coverage guard exempts it, so it needs no group and is
    never updated (freely training an unanchored r alongside s would collapse KL(s||r)->0). Under
    ``cfg.learnable_r=True`` it is un-frozen and grouped here (mean@``m_p_mu_lr``, log-scale@``m_p_sigma_lr``,
    like the s tables) so it trains as an empirical-Bayes centroid.
    The learned MODEL-level parameters are grouped likewise when their toggle is on: the Regime-II
    edge connection ``connection_W`` (transport_mode='regime_ii') at ``m_phi_lr`` (a gauge-connection
    scale) -- so those sanctioned-NN-exception toggles train rather than tripping the coverage guard.
    """
    pb = model.prior_bank
    # Geometric gauge M-step (opt-in, cfg.m_phi_natural_grad): the gauge-frame coordinate groups
    # (phi_embed, and the full-width pos_phi_free) are flagged gauge=True so GaugeNaturalGradAdamW
    # steps them by natural gradient under cfg.phi_precond_mode instead of AdamW; weight_decay=0 on
    # those groups (Euclidean L2 on phi is non-geometric -- mass_phi shrinks the frame in the loss).
    # Default OFF: the flag is absent and every group is plain AdamW, byte-identical to before.
    nat = cfg.m_phi_natural_grad
    # omega_direct (cfg.gauge_parameterization='omega_direct'): omega_embed holds GL(K) group elements
    # U directly (not phi coordinates), so it is grouped {"omega": True} and stepped by the group-
    # manifold retraction in GaugeNaturalGradAdamW. Default ('phi') leaves this False and the branch dead.
    omega_direct = cfg.gauge_parameterization == "omega_direct"
    n_gen = model.group.generators.shape[0]
    # Each group carries an explicit "role" in {mu, sigma, phi} -- the belief-component family it
    # steps (mean-LR / scale-LR / gauge-LR). The grad-norm decomposition (train_step) aggregates the
    # pre-clip grad by role, so the figure attributes the signal correctly REGARDLESS of group order
    # or which tables are live (e.g. under prior_source='model_channel' the dead mu_embed contributes
    # 0 while the live s_mu_embed carries the mean signal -- both are role='mu'). Role is used in
    # preference to the group INDEX (the old 0/1/2 assumption broke whenever a config rerouted the
    # active mean/scale capacity off mu_embed/sigma_log_embed) and to the LR VALUE (m_p_mu_lr and
    # m_phi_lr may coincide). Extra dict keys ride alongside "gauge"/"weight_decay" and are ignored
    # by AdamW / GaugeNaturalGradAdamW.
    phi_group = {"params": [pb.phi_embed], "lr": cfg.m_phi_lr, "weight_decay": cfg.phi_weight_decay,
                 "role": "phi"}
    if nat:
        phi_group["gauge"] = True
        phi_group["weight_decay"] = 0.0
    # sigma_weight_decay (default None = inherit the global weight_decay, the long-standing
    # behavior): a dedicated AdamW decay for the log-variance tables. The global decay pulls
    # log sigma toward 0 (sigma toward 1) -- an unintended lognormal prior fighting the configured
    # sigma_init on the KL-decode path; sigma_weight_decay=0.0 exempts the sigma sector. Applied to
    # every sigma-role CAPACITY table (belief, s-channel, untied decode); the centroid r_sigma_log
    # keeps its existing hard 0.0 exemption.
    sigma_wd = {} if cfg.sigma_weight_decay is None else {"weight_decay": cfg.sigma_weight_decay}
    groups = [
        {"params": [pb.mu_embed],                              "lr": cfg.m_p_mu_lr,    "role": "mu"},
        {"params": [pb.sigma_log_embed, pb.decode_log_scale],  "lr": cfg.m_p_sigma_lr, "role": "sigma", **sigma_wd},
        phi_group,
    ]
    if omega_direct:                                           # omega_embed holds GL(K) elements U directly
        # Stepped by the group-manifold retraction (weight_decay=0: Euclidean L2 on a group element is
        # non-geometric, the same exemption the gauge frame carries). role='phi' -> gauge-LR + phi grad-norm.
        groups.append({"params": [pb.omega_embed], "lr": cfg.m_phi_lr,
                       "weight_decay": 0.0, "role": "phi", "omega": True})
    if getattr(pb, "decode_mu_embed", None) is not None:        # untie_decode_bank=True decode tables
        # Cloned from the encode tables at init (step-0 byte-identical decode), trained separately so
        # the decode direction can decouple from the E-step prior/self-coupling target; grouped like
        # the tables they were cloned from (mean@m_p_mu_lr, log-scale@m_p_sigma_lr).
        groups.append({"params": [pb.decode_mu_embed],        "lr": cfg.m_p_mu_lr,    "role": "mu"})
        groups.append({"params": [pb.decode_sigma_log_embed], "lr": cfg.m_p_sigma_lr, "role": "sigma", **sigma_wd})
    if pb.output_proj_weight is not None:                       # use_prior_bank=False linear decode
        groups.append({"params": [pb.output_proj_weight], "lr": cfg.m_p_mu_lr, "role": "mu"})
    if pb.output_proj_bias is not None:                         # decode_bias: learned log-unigram prior
        # weight_decay=0 -- decaying a unigram prior toward zero biases it to a flat distribution
        # (the same protection phi/Omega carry).
        groups.append({"params": [pb.output_proj_bias], "lr": cfg.m_p_mu_lr, "weight_decay": 0.0, "role": "mu"})
    if getattr(model, "head_mixer", None) is not None:          # use_head_mixer=True Schur mixer
        groups.append({"params": list(model.head_mixer.parameters()), "lr": cfg.m_p_mu_lr, "role": "mu"})
    if getattr(model, "cg_coupling", None) is not None:         # use_cg_coupling=True CG path weights
        groups.append({"params": [model.cg_coupling.path_weights], "lr": cfg.m_p_mu_lr, "role": "mu"})
    if getattr(model, "pos_phi_free", None) is not None:        # pos_phi='learned' positional table
        pos_group = {"params": [model.pos_phi_free], "lr": cfg.m_phi_lr,      # a gauge-frame scale
                     "weight_decay": cfg.phi_weight_decay, "role": "phi"}     # decayed like phi_embed
        # Natural-grad the positional frame too. pos_phi_free is created at FULL coordinate width
        # (n_gen) and project_phi_to_slk preserves that width, so this width guard is currently
        # ALWAYS satisfied (audit 2026-06-13 L2: there is no reduced-width chart today). It is kept
        # defensively: a future reduced-width pos_phi would be shape-incompatible with the
        # full-generator pullback metric and must fall back to AdamW here.
        if nat and model.pos_phi_free.shape[-1] == n_gen:
            pos_group["gauge"] = True
            pos_group["weight_decay"] = 0.0
        groups.append(pos_group)
    if getattr(pb, "s_mu_embed", None) is not None:             # model-channel s tables (lambda_gamma>0 or
        groups.append({"params": [pb.s_mu_embed],        "lr": cfg.m_p_mu_lr,    "role": "mu"})    # prior_source=model_channel):
        groups.append({"params": [pb.s_sigma_log_embed], "lr": cfg.m_p_sigma_lr, "role": "sigma", **sigma_wd})  # mean@m_p_mu_lr, log-scale@
        # m_p_sigma_lr, mirroring the belief tables. s is the model channel / (under model_channel) the
        # live belief prior, so it must train. The hyper-prior CENTROID r is grouped only when
        # learnable_r un-freezes it (next block); FROZEN-by-default r (requires_grad=False, prior_bank.py)
        # is exempt from the coverage guard -- a fixed centroid per the manuscript's "higher, slower
        # meta-level".
    if getattr(pb, "r_mu", None) is not None and pb.r_mu.requires_grad:  # learnable_r=True: un-frozen r
        # weight_decay=0: r is a hyper-prior CENTROID, not capacity. L2-decaying it pulls the learned
        # centroid toward the degenerate (r_mu=0, r_sigma=1) fixed point, fighting the empirical-Bayes
        # population-centroid objective (and corrupting the KL(s||r) m-projection at sigma_init != 1) --
        # the same exemption the learned unigram-bias prior (output_proj_bias) and the gauge frame carry.
        groups.append({"params": [pb.r_mu],        "lr": cfg.m_p_mu_lr,    "weight_decay": 0.0, "role": "mu"})     # centroid mean
        groups.append({"params": [pb.r_sigma_log], "lr": cfg.m_p_sigma_lr, "weight_decay": 0.0, "role": "sigma"})  # centroid log-scale
    if getattr(model, "connection_W", None) is not None:        # transport_mode='regime_ii' learned
        w_group = {"params": [model.connection_W], "lr": cfg.m_phi_lr, "role": "phi"}   # connection -> gauge LR
        if cfg.connection_weight_decay is not None:             # dedicated connection-norm ceiling
            w_group["weight_decay"] = cfg.connection_weight_decay   # (audit 2026-06-10 F9); None ->
        groups.append(w_group)                                  # inherit the global weight_decay
    if getattr(model, "connection_M", None) is not None:        # transport_mode='regime_ii_covariant' (Route B)
        m_group = {"params": [model.connection_M], "lr": cfg.m_phi_lr, "role": "phi"}   # connection -> gauge LR
        if cfg.connection_weight_decay is not None:             # shares the connection-norm ceiling
            m_group["weight_decay"] = cfg.connection_weight_decay
        groups.append(m_group)
    if getattr(model, "connection_L", None) is not None:        # transport_mode='regime_ii_link' / '_charted'
        l_group = {"params": [model.connection_L], "lr": cfg.m_phi_lr, "role": "phi"}   # direct link -> gauge LR
        if cfg.connection_weight_decay is not None:             # shares the connection-norm ceiling
            l_group["weight_decay"] = cfg.connection_weight_decay
        groups.append(l_group)
    if getattr(model, "t5_bias", None) is not None:             # t5_learnable_bias=True relative-position bias
        # weight_decay=0: the per-bucket T5 bias b_{i-j} is a relative-position PRIOR shaping the
        # attention pi, not capacity; L2-decaying it toward zero biases the prior toward a flat/uniform
        # relative-position distribution (the same exemption output_proj_bias / r / the gauge frame
        # carry). role='mu' is the catch-all for learned non-variance/non-gauge tables (head_mixer,
        # ...); the bias is not a gauge frame, so it steps under the mean LR, not m_phi_lr.
        groups.append({"params": [model.t5_bias], "lr": cfg.m_p_mu_lr, "weight_decay": 0.0, "role": "mu"})
    if getattr(model, "log_kappa_beta", None) is not None:      # learnable_kappa_beta=True per-block temperature
        # weight_decay=0: decaying log_kappa toward 0 pulls tau back to the fixed Vaswani
        # calibration (a prior, not capacity) -- the same exemption t5_bias/output_proj_bias/r
        # carry. role='mu' is the catch-all for learned non-variance/non-gauge tables; NO gauge
        # flag (the temperature touches no gauge transport), so the group rides as plain AdamW
        # even under GaugeNaturalGradAdamW.
        groups.append({"params": [model.log_kappa_beta],  "lr": cfg.m_p_mu_lr, "weight_decay": 0.0, "role": "mu"})
    if getattr(model, "log_kappa_gamma", None) is not None:     # learnable_kappa_gamma=True (model channel)
        groups.append({"params": [model.log_kappa_gamma], "lr": cfg.m_p_mu_lr, "weight_decay": 0.0, "role": "mu"})

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
    # but stay grouped -- AdamW skips a None-grad param ENTIRELY, so neither an update NOR weight
    # decay fires on the dead table; audit 2026-06-13 L3). These are intentional, not coverage bugs.
    grouped = {p for g in groups for p in g["params"]}
    missing = {p for p in model.parameters() if p.requires_grad} - grouped   # frozen params are exempt
    if missing:
        raise AssertionError(
            f"build_optimizer left {len(missing)} model parameter(s) ungrouped; they would never "
            f"train. Add them to a param group."
        )

    if nat or omega_direct:
        # Geometrically-correct gauge frame: natural-gradient + momentum on the gauge groups under
        # cfg.phi_precond_mode (set it to 'pullback_per_block' for the exact exp-map metric -- killing
        # is conformal, so under this manual natural-grad step (which AdamW never normalizes) it is a
        # direction-preserving effective-LR rescale by the conformal factor, NOT a no-op; only the
        # non-conformal pullback metric reshapes the step direction), AdamW on every other group.
        # Under omega_direct the {"omega": True} group is stepped by the group-manifold retraction
        # (cfg.omega_retract_mode) instead; both custom steps live in GaugeNaturalGradAdamW.
        # fused is off: the custom gauge step bypasses the fused kernel, and the non-gauge groups are few.
        from vfe3.gauge_optim import GaugeNaturalGradAdamW
        return GaugeNaturalGradAdamW(
            groups, model.group.generators, list(model.group.irrep_dims),
            precond_mode=cfg.phi_precond_mode, gauge_momentum=cfg.m_gauge_momentum,
            gauge_update_rule=cfg.m_gauge_update_rule,
            omega_retract_mode=cfg.omega_retract_mode,
            skew_symmetric=model.group.skew_symmetric,
            omega_reorth_every=cfg.omega_reorth_every,
            weight_decay=cfg.weight_decay,
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


def _floor_lr_lambdas(
    base_lrs: Sequence[float],
    cfg:      VFE3Config,
) -> List[Callable[[int], float]]:
    r"""Per-group ``LambdaLR`` multipliers that floor each group's ABSOLUTE LR at
    ``max(cfg.min_lr, cfg.min_lr_frac * base)``.

    A ``LambdaLR`` scales each group's base LR by its multiplier, so flooring the
    *multiplier* at ``max(min_lr/base, min_lr_frac)`` floors the *product*::

        base * max(min_lr/base, min_lr_frac, cosine) = max(min_lr, min_lr_frac*base, base*cosine).

    The absolute floor ``min_lr`` is shared across groups; the fractional floor
    ``min_lr_frac`` scales with each group's own base, preserving the m_mu:m_sigma:m_phi
    ratios into the cosine tail. A group whose base LR is 0 (a deliberately frozen
    channel, e.g. ``m_phi_lr=0``) drops the ``min_lr/base`` term (no division by zero)
    and stays frozen: ``min_lr`` does not resurrect it. With ``min_lr=min_lr_frac=0``
    this is the pure half-cosine-to-zero.
    """
    def make(base: float) -> Callable[[int], float]:
        abs_mult = cfg.min_lr / base if base > 0.0 else 0.0     # shared absolute floor (skip if frozen)
        floor    = max(abs_mult, cfg.min_lr_frac)               # combined per-group multiplier floor
        return lambda s: max(floor, lr_lambda(s, cfg))
    return [make(b) for b in base_lrs]


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
    grad_clip:        float                              = 1.0,
    grad_accum_steps: int                                = 1,
    scaler:           Optional['torch.amp.GradScaler']  = None,
    metrics_out:      Optional[dict]                     = None,
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

    ``scaler`` is an optional :class:`torch.amp.GradScaler` for fp16 training (prevents
    gradient underflow through the unrolled E-step). A disabled scaler (``enabled=False``)
    is a documented no-op: ``scale`` is identity, ``unscale_`` is a no-op, and ``step``
    calls ``optimizer.step()`` directly — so passing ``scaler=None`` (or an
    ``enabled=False`` instance) keeps this function byte-identical to the unscaled path.
    """
    # A disabled scaler is a documented no-op (scale -> identity, unscale_ -> nothing,
    # step -> optimizer.step()), so scaler=None keeps this path byte-identical to the unscaled loop.
    _scaler = scaler if scaler is not None else torch.amp.GradScaler(
        device=tokens.device.type, enabled=False)

    optimizer.zero_grad(set_to_none=True)
    _mb_tok: List[int] = []                                     # per-microbatch counted-token spread (accum only)
    # E-step belief-gradient capture: a dict the forward fills with the raw ||grad_mu/sigma/phi|| of F
    # (the inference analogue of the M-step per-role grad norms). Created ONLY when metrics are being
    # logged this step; None -> the forward skips the capture entirely (zero overhead, byte-identical).
    _egrad = {} if metrics_out is not None else None
    if grad_accum_steps == 1:                                   # default path: byte-identical to the single-step loop
        _, loss, ce = model(tokens, targets, estep_grad_out=_egrad)
        _scaler.scale(loss).backward()
        _loss_det = loss.detach()                               # host read DEFERRED: fused with the grad-finite
        step_loss = None                                        # flag at the gate below (audit 2026-07-01 round-3)
        # CE is synced to a Python float only when a metrics dict is being filled this step (it feeds
        # metrics_out['train_ce'] and nothing else); on a silent step the extra D2H copy is skipped.
        step_ce = (float(ce.detach()) if (ce is not None and metrics_out is not None) else float("nan"))
    else:
        if tokens.shape[0] % grad_accum_steps != 0:            # equal-token microbatches require an even split
            raise ValueError(
                f"grad_accum_steps={grad_accum_steps} must divide the batch size "
                f"{tokens.shape[0]} for equal microbatches; got remainder "
                f"{tokens.shape[0] % grad_accum_steps}."
            )
        tok_chunks = torch.chunk(tokens, grad_accum_steps, dim=0)
        tgt_chunks = torch.chunk(targets, grad_accum_steps, dim=0)
        # Token-weighted accumulation: weight each microbatch's mean loss by its valid-token fraction
        # n_mb/n_tot so the accumulated gradient equals the full-batch token-mean even under uneven
        # ignore-padding across the batch axis. Uniform 1/grad_accum_steps is exact only when the
        # microbatches carry EQUAL counted-token counts (e.g. the default unpadded loader), where
        # n_mb/n_tot == 1/grad_accum_steps and this is byte-identical to the prior weighting.
        _mb_tok[:] = [int((tc != -100).sum()) for tc in tgt_chunks]   # counted tokens per microbatch (spread = bias)
        n_tot = max(sum(_mb_tok), 1)
        # Uneven counted-token microbatches (audit 2026-07-01 C8): the n_mb/n_tot weight below is
        # exact for the token-mean CE but only APPROXIMATE for the non-CE regularizers (mass_phi,
        # mstep_self_coupling, lambda_h, gamma), which are means over (B, N)/state and do not scale
        # with target tokens. The default unpadded loader has equal counts (w == 1/K exactly), so
        # this warning fires only in the regime where the weighting actually diverges.
        if grad_accum_steps > 1 and _mb_tok and (max(_mb_tok) != min(_mb_tok)):
            import warnings
            warnings.warn(
                "grad_accum_steps>1 with uneven counted-token microbatches: non-CE regularizers "
                "(mass_phi, mstep_self_coupling, lambda_h, gamma) are token-weighted by n_mb/n_tot "
                "rather than by their own reduction, so their accumulated gradient is an "
                "approximation. Use an unpadded/equal-token loader or grad_accum_steps=1 for the "
                "exact objective.",
                RuntimeWarning, stacklevel=2,
            )
        step_loss = 0.0
        step_ce = 0.0
        for tok_mb, tgt_mb, n_mb in zip(tok_chunks, tgt_chunks, _mb_tok):
            _, loss_mb, ce_mb = model(tok_mb, tgt_mb, estep_grad_out=_egrad)   # last microbatch wins the E-step grads
            w = n_mb / n_tot                                          # token-mean weight (valid-token fraction)
            _scaler.scale(loss_mb * w).backward()                     # accumulate the token-weighted microbatch grad
            step_loss += float(loss_mb.detach()) * w
            if metrics_out is not None:                               # CE synced only on a logged step (PERF)
                step_ce += (float(ce_mb.detach()) if ce_mb is not None else float("nan")) * w
    # Unscale once when EITHER clipping or metrics capture needs true-unit gradients; GradScaler
    # tracks that unscale_ already ran so the later _scaler.step() does not re-unscale. With a
    # disabled scaler (the default) unscale_ is a no-op, and with metrics_out=None this is
    # byte-identical to the prior clip-only path.
    need_unscale = (grad_clip is not None and grad_clip > 0) or (metrics_out is not None)
    if need_unscale:
        _scaler.unscale_(optimizer)
    # Finite-GRADIENT gate (audit 2026-07-01 F1): a FINITE scalar loss can still carry a NaN/Inf
    # parameter gradient through the unrolled E-step on a degenerate batch; stepping AdamW on it
    # would permanently poison the exp_avg/exp_avg_sq moment buffers. Checked on EVERY step on the
    # disabled-scaler default path; the deferred step-loss value and the grad-finite flag ride ONE
    # fused D2H transfer, so the default path keeps exactly one unconditional sync per step
    # (audit 2026-07-01 round-3). The ENABLED fp16 scaler path already skips internally via
    # found_inf, so it stays byte-identical and is not re-checked here.
    _scaler_enabled = scaler is not None and scaler.is_enabled()
    grad_finite = True
    if not _scaler_enabled:
        _flags = [torch.isfinite(p.grad).all()
                  for g in optimizer.param_groups
                  for p in g["params"] if p.grad is not None]
        if _flags and step_loss is None:                        # fuse the deferred loss read with the flag
            _pair = torch.stack((_loss_det.float(), torch.stack(_flags).all().float())).tolist()
            step_loss   = _pair[0]
            grad_finite = bool(_pair[1])
        elif _flags:                                            # accum path: step_loss already a Python float
            grad_finite = bool(torch.stack(_flags).all())
    if step_loss is None:                                       # fp16 / no-grads fallback: one plain loss sync
        step_loss = float(_loss_det)
    if metrics_out is not None:
        # Pre-clip gradient health -- the global L2 norm clip_grad_norm_ RETURNS-and-discards, plus
        # per-ROLE norms (mu/sigma/phi from each group's "role" tag in build_optimizer, aggregated in
        # quadrature across ALL groups carrying that role). Role -- not the old groups[0/1/2] index --
        # so the LIVE tables are attributed correctly under any config (e.g. under
        # prior_source='model_channel' the dead mu_embed adds 0 while the live s_mu_embed carries the
        # role='mu' signal; previously mu/sigma logged a flat 0). Captured AFTER unscale_ but BEFORE
        # clip so the value is the true pre-clip gradient magnitude. The three roles partition every
        # group, so role grads sum in quadrature to grad_norm.
        total_sq: float = 0.0
        role_g_sq = {"mu": 0.0, "sigma": 0.0, "phi": 0.0}
        role_w_sq = {"mu": 0.0, "sigma": 0.0, "phi": 0.0}
        for g in optimizer.param_groups:
            gsq = sum(float(p.grad.detach().pow(2).sum())
                      for p in g["params"] if p.grad is not None)
            total_sq += gsq
            _role = g.get("role")
            if _role in role_g_sq:
                role_g_sq[_role] += gsq
                role_w_sq[_role] += sum(float(p.detach().pow(2).sum()) for p in g["params"])
        metrics_out["grad_norm"] = total_sq ** 0.5
        for _name in ("mu", "sigma", "phi"):
            metrics_out[f"grad_norm_{_name}"] = role_g_sq[_name] ** 0.5
            # weight norm of the same role: with the logged grad_norm + per-group LR, the
            # update-to-weight ratio (the ~1e-3 LR-scale sanity check) is derivable offline.
            metrics_out[f"weight_norm_{_name}"] = role_w_sq[_name] ** 0.5
        # E-step belief-gradient norms: the raw ||grad_mu/sigma/phi|| of F captured inside the last
        # E-step iteration (model.forward estep_grad_out) -- the INFERENCE analogue of the M-step
        # per-role grads above (parameter learning). 0.0 for a component whose substep is off (e.g.
        # phi when e_phi_lr=0); the figure masks non-positive points.
        if _egrad is not None:
            for _name in ("mu", "sigma", "phi"):
                metrics_out[f"estep_grad_norm_{_name}"] = float(_egrad.get(_name, 0.0))
        metrics_out["loss_finite"] = float(math.isfinite(step_loss))
        metrics_out["train_ce"] = step_ce            # pre-step CE (matches step_loss; not a post-update re-forward)
        if _mb_tok:                                             # grad_accum_steps>1: token-spread bias check
            metrics_out["grad_accum_tok_spread"] = float(max(_mb_tok) - min(_mb_tok))
        if scaler is not None and scaler.is_enabled():          # fp16: surface the loss-scale (Tier-2 health)
            metrics_out["grad_scale"] = float(scaler.get_scale())
    # NaN/Inf skip on the default (disabled-scaler) path: a non-finite loss OR a non-finite
    # parameter gradient (grad_finite above -- a finite loss does NOT imply finite grads) would
    # PERMANENTLY poison AdamW's moment buffers in a single step with no recovery. The fp16
    # GradScaler path already skips internally via found_inf; mirror that here by dropping the
    # grads and not stepping when the scaler is disabled (audits 2026-06-17 r2, 2026-07-01 F1).
    skip_step = (not _scaler_enabled) and ((not math.isfinite(step_loss)) or (not grad_finite))
    if metrics_out is not None:                              # expose the gate for tests/CSV
        metrics_out["step_skipped"] = float(skip_step)
        metrics_out["grad_finite"]  = float(grad_finite)
    if grad_clip is not None and grad_clip > 0 and not skip_step:
        if getattr(model.cfg, "grad_clip_per_role", False):
            # Per-role clipping (default OFF): the global L2 norm below is dominated by phi_embed
            # (V x n_gen, the bulk of all parameters), so when it binds it silently rescales every
            # OTHER role's effective LR by a phi-noise-coupled factor -- the kl_max silent-bind
            # pattern. Clip each role's parameter set to grad_clip separately instead (the roles
            # partition the optimizer groups; see build_optimizer).
            role_params: dict = {}
            for _g in optimizer.param_groups:
                role_params.setdefault(_g.get("role", "other"), []).extend(_g["params"])
            for _ps in role_params.values():
                torch.nn.utils.clip_grad_norm_(_ps, grad_clip)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    if skip_step:
        optimizer.zero_grad(set_to_none=True)                # drop poisoned grads; do NOT step AdamW
    else:
        _scaler.step(optimizer)
    _scaler.update()
    # Closed-form hyper-prior M-step (r_update_mode='barycenter'): after AdamW updates the s tables,
    # set the centroid r to their forward-KL barycenter (the closed-form variational M-step, in place
    # of an AdamW step on r -- r is ungrouped/frozen-from-the-optimizer under this mode). No-op for the
    # default gradient r (trained inside optimizer.step above) and for frozen r (learnable_r=False).
    _cfg = model.cfg
    if (not skip_step) and _cfg.learnable_r and _cfg.r_update_mode == "barycenter":
        model.prior_bank.barycenter_r_()   # gated with the optimizer step: never M-step on poisoned grads
    scheduler.step()                       # UNCONDITIONAL: resume rebuilds LambdaLR at last_epoch=start_step-1
    #                                        assuming exactly one scheduler.step per loop iteration
    return step_loss


def _maybe_metropolis_omega(
    model:     VFEModel,
    token_ids: torch.Tensor,             # (B, N) input token ids (the SAME batch fed to train_step)

    *,
    step:      int,
    generator: torch.Generator,          # persistent seeded RNG, threaded across steps
) -> None:
    r"""Gated + cadence-checked call to the learnable-reflection Metropolis det-sign sweep.

    No-op unless a learnable-reflection mode is active -- ``cfg.omega_reflection == 'metropolis'``
    (omega_direct frame) OR ``cfg.phi_reflection == 'metropolis'`` (phi reflection_sign) -- and even
    then fires only every ``cfg.omega_metropolis_every`` optimizer steps (the two ``omega_metropolis_*``
    knobs name the shared move, not the storage, so they govern both modes). Factored out of the
    training loop so the seam is a single guarded line there (see design spec Sec.4);
    ``model.metropolis_omega_step`` is itself a no-op under any other mode, so this gate is a fast-path
    short-circuit, not the sole safety net.
    """
    cfg = model.cfg
    if ((cfg.omega_reflection == "metropolis" or cfg.phi_reflection == "metropolis")
            and (step % cfg.omega_metropolis_every == 0)):
        model.metropolis_omega_step(token_ids, generator=generator)


@torch.no_grad()
def evaluate(
    model:  VFEModel,
    loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],   # yields (tokens, targets) batches

    *,
    max_batches:     Optional[int]          = None,
    tokens_per_char: float                  = 1.0,
    device:          Optional[torch.device] = None,
) -> Dict[str, float]:
    r"""Token-weighted corpus evaluation. Returns ``{ce, ppl, bpc}`` (CE in nats).

    .. math::
        \mathrm{CE} = \frac{\sum_b n_b\, \mathrm{ce}_b}{\sum_b n_b},\quad
        \mathrm{PPL} = e^{\min(\mathrm{CE},\,20)},\quad
        \mathrm{BPC} = \frac{\mathrm{CE}}{\ln 2}\,\cdot\,\mathrm{tokens\_per\_char},

    with ``n_b`` the number of non-ignored (``!= -100``) target tokens in batch ``b``.
    Aggregating by token count (not per-batch mean) reproduces one cross-entropy over
    the concatenated corpus, including a partial last batch. ``tokens_per_char`` is the
    bits-per-CHARACTER correction (``n_tokens / n_codepoints`` from
    :func:`vfe3.data.datasets.tokens_per_char`); the default 1.0 leaves the honest
    bits-per-TOKEN value (correct for a character-level / synthetic stream, and the value
    to pass when the char count is unavailable).
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
        "bpc": (ce / math.log(2.0)) * tokens_per_char,
    }


# Fixed column set for the held-out per-eval probes (initialized to NaN before training and carried
# forward like last_val, so every metrics.csv row carries the SAME columns -- rectangular -- whether
# or not an eval has run yet). Config-conditional keys (val_head_redundancy_js needs >= 2 heads;
# pos_loss_* needs targets and seq_len >= 4) stay NaN on runs where they do not apply.
_VAL_DIAG_KEYS = (
    "val_self_coupling", "val_self_divergence", "val_belief_coupling", "val_attention_entropy",
    "val_free_energy_total",
    "val_attn_entropy", "val_effective_rank", "val_belief_cond_median", "val_attn_entropy_min",
    "val_attn_entropy_min_all", "val_attn_collapsed_heads", "val_future_leakage", "val_row_sum_error",
    "val_pos_content_r2", "val_prev_token_mass", "val_period_match_mass", "val_head_redundancy_js",
    "estep_f_drop", "estep_f_nondecreasing_frac", "estep_r_mu_last", "estep_r_sigma_last",
    "estep_r_phi_last", "pos_loss_first_q", "pos_loss_last_q", "pos_loss_ratio",
    "val_builder_resid",
    # held-out gauge / SPD / Fisher geometry (surfaced from the already-computed val diagnostics dict)
    "val_holonomy_wilson", "val_cocycle_residual", "val_gauge_invariant_spread",
    "val_fisher_trace_mean", "val_belief_cond_p95", "val_phi_norm_mean", "val_phi_norm_std",
    "val_guard_sigma_floor_frac", "val_guard_sigma_ceil_frac", "val_guard_energy_klmax_frac",
    "val_nonfinite_frac",
)


@torch.no_grad()
def _val_diagnostics(
    model:      VFEModel,
    val_loader: Iterable,
    device:     torch.device,
) -> Dict[str, float]:
    r"""Held-out per-eval probes (the train-loop ``diagnostics`` runs on the live TRAIN batch).

    Computes the validation-side F decomposition, attention-map structure across ALL layers/heads
    (entropy collapse, causal-mask sanity, positional-vs-content, induction/copy, head redundancy),
    the E-step F-descent + belief-residual convergence certificate, and the per-position
    within-sequence loss. Off the graph (no_grad). Best-effort: the CALLER wraps this so a replay
    error simply leaves the previous values carried forward; the returned subset is ``.update``-d
    into a NaN-initialized dict, so the CSV stays rectangular.
    """
    import torch.nn.functional as F
    from vfe3 import metrics as M
    from vfe3.viz import extract as ex

    out: Dict[str, float] = {}
    batch = next(iter(val_loader))
    val_tok, val_tgt = (batch if isinstance(batch, (tuple, list)) else (batch, None))
    val_tok = val_tok.to(device)
    val_tgt = val_tgt.to(device) if val_tgt is not None else None
    vn = max(int(val_tok.shape[1]), 1)

    vd = model.diagnostics(val_tok)                              # held-out F decomposition (per token)
    out["val_self_coupling"]      = vd["self_coupling"]     / vn
    out["val_self_divergence"]    = vd["self_divergence"]   / vn
    out["val_belief_coupling"]    = vd["belief_coupling"]   / vn
    out["val_attention_entropy"]  = vd["attention_entropy"] / vn
    out["val_free_energy_total"]  = vd["total"]             / vn
    out["val_attn_entropy"]       = vd["attn_entropy"]
    out["val_effective_rank"]     = vd["effective_rank"]
    out["val_belief_cond_median"] = vd["belief_cond_median"]
    out["val_attn_entropy_min"]   = vd["attn_entropy_min"]
    # Held-out gauge / SPD / Fisher geometry: vd = model.diagnostics(val_tok) ALREADY computed the full
    # geometry dict above, so surfacing these is near-free -- the held-out counterpart to the train-batch
    # geometry-health columns, the more credible evidence for LEARNED geometry vs a train-batch artifact.
    out["val_holonomy_wilson"]         = vd["holonomy_wilson"]
    out["val_cocycle_residual"]        = vd["cocycle_residual"]
    out["val_gauge_invariant_spread"]  = vd["gauge_invariant_spread"]
    out["val_fisher_trace_mean"]       = vd["fisher_trace_mean"]
    out["val_belief_cond_p95"]         = vd["belief_cond_p95"]
    out["val_phi_norm_mean"]           = vd["phi_norm_mean"]
    out["val_phi_norm_std"]            = vd["phi_norm_std"]
    out["val_guard_sigma_floor_frac"]  = vd["guard_sigma_floor_frac"]
    out["val_guard_sigma_ceil_frac"]   = vd["guard_sigma_ceil_frac"]
    out["val_guard_energy_klmax_frac"] = vd["guard_energy_klmax_frac"]
    out["val_nonfinite_frac"]          = vd["nonfinite_frac"]

    amaps = model.attention_maps(val_tok)                       # (L, H, N, N) all layers/heads
    hmin = M.attention_entropy_rows(amaps).min(dim=-1).values   # (L, H) per-head min row entropy
    out["val_attn_entropy_min_all"] = float(hmin.min())
    out["val_attn_collapsed_heads"] = float((hmin < 0.6931471805599453).float().sum())
    cs = M.causal_sanity(amaps)
    out["val_future_leakage"] = float(cs["future_leakage"].max())   # soft causal prior can leak silently
    out["val_row_sum_error"]  = float(cs["row_sum_error"].max())
    out["val_pos_content_r2"] = float(M.positional_content_score(amaps).mean())
    sh = M.structured_head_scores(amaps)
    out["val_prev_token_mass"]   = float(sh["prev_token"].mean())
    out["val_period_match_mass"] = float(sh["period_match"].mean())
    if amaps.shape[1] > 1:                                       # head redundancy needs >= 2 heads
        h = amaps.shape[1]
        off = ~torch.eye(h, dtype=torch.bool, device=amaps.device)
        out["val_head_redundancy_js"] = float(torch.stack(
            [M.head_redundancy_js(amaps[li])[off].mean() for li in range(amaps.shape[0])]).mean())

    tr = ex.e_step_belief_trace(model, val_tok)                 # E-step inner-loop trajectory
    f = tr["free_energy"] / vn                                  # PER-TOKEN (free_energy_value is a per-seq SUM)
    out["estep_f_drop"] = float(f[-1] - f[0])                   # < 0 = F descended over the inner loop
    # fraction of inner iterations that did NOT decrease F. EXPECTED to be nonzero for parallel
    # (Jacobi) mean-field with a finite step (e_step.py: 'not guaranteed monotone per iteration') --
    # a descent-quality readout, not a convergence-FAILURE flag.
    out["estep_f_nondecreasing_frac"] = (
        float((f[1:] > f[:-1] + 1e-9).float().mean()) if f.numel() > 1 else 0.0)
    res = M.estep_residuals(tr["mu"], tr["sigma"], tr["phi"])   # last-iter belief change (SPD metric for sigma)
    for _nm, _key in (("r_mu", "estep_r_mu_last"), ("r_sigma", "estep_r_sigma_last"),
                      ("r_phi", "estep_r_phi_last")):
        out[_key] = float(res[_nm][-1].mean()) if res[_nm].numel() else 0.0

    if val_tgt is not None:                                     # per-position within-sequence loss
        vlog = model(val_tok)                                   # (B, N, V) inference path
        b, n = val_tok.shape
        per = F.cross_entropy(vlog.reshape(-1, vlog.shape[-1]).float(), val_tgt.reshape(-1),
                              ignore_index=-100, reduction="none").reshape(b, n)
        valid = (val_tgt != -100).float()
        pos_ce = (per * valid).sum(0) / valid.sum(0).clamp(min=1.0)   # (N,) mean CE at each position
        q = n // 4
        if q > 0:
            out["pos_loss_first_q"] = float(pos_ce[:q].mean())
            out["pos_loss_last_q"]  = float(pos_ce[-q:].mean())
            out["pos_loss_ratio"]   = float(pos_ce[-q:].mean() / pos_ce[:q].mean().clamp(min=1e-9))

    # Builder-break gauge-equivariance residual per eval (A2/EXP-9): the head-mixer congruence defect
    # at the converged belief (~eps under the tied gauge; climbs as the untied block_glk mixer drifts
    # from identity -> the residual-drift-vs-step series). Only with a head mixer; isolated so a replay
    # fault drops just this scalar (NaN-rectangular), not the whole val-diag row.
    if model.head_mixer is not None:
        try:
            cst = ex.converged_state(model, val_tok)
            br = M.head_mixer_gauge_residual(cst["mu"], cst["sigma"], model.head_mixer, model.group,
                                             diagonal=model.cfg.diagonal_covariance)
            out["val_builder_resid"] = float(torch.cat([br["mu_residual"], br["sigma_residual"]]).median())
        except Exception:                                          # leave NaN; never fail the eval
            pass
    return out


def train(
    model:  VFEModel,
    loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],   # yields (tokens, targets) batches
    cfg:    VFE3Config,

    *,
    n_steps:   int   = 100,
    grad_clip: float = 1.0,

    log_interval:    Optional[int]            = None,
    eval_interval:   Optional[int]            = None,
    val_loader:      Optional[Iterable]       = None,
    tokens_per_char: float                    = 1.0,    # val BPC char-correction (1.0 = bits/token)
    device:          Optional[torch.device]   = None,
    logger:          Optional[logging.Logger] = None,
    artifacts:       Optional["RunArtifacts"] = None,
    resume_from:     'Optional[str | Path]'   = None,   # checkpoints/step_<N>.pt to resume from (None -> cfg.resume_from -> from scratch)

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
    ``log_interval`` is positive a per-step line is emitted every
    ``log_interval`` steps (CE and diagnostics recomputed under ``no_grad`` only at those
    steps, off the training graph), AND -- when ``tqdm`` is installed -- the step loop runs
    under a ``tqdm`` progress bar whose built-in rate readout shows live ``it/s`` every step
    (the formatted lines render above it via ``logging_redirect_tqdm``); when ``eval_interval``
    is positive and ``val_loader`` is given a validation block is emitted every
    ``eval_interval`` steps.
    """
    optimizer = build_optimizer(model, cfg)
    # Warmup/cosine multiplier, floored per group so each group's ABSOLUTE LR never decays below
    # max(cfg.min_lr, cfg.min_lr_frac * base) -- see _floor_lr_lambdas. With cfg.min_lr=cfg.min_lr_frac=0
    # this is exactly the pure half-cosine-to-zero (the theoretically pure path). base_lrs are the
    # CONFIGURED per-group LRs, captured before any scheduler multiplier or resume-load mutates group['lr'].
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    # Opt-in RESUME (PL8): an explicit resume_from arg, else cfg.resume_from, else from scratch. When set,
    # restore model weights + AdamW momentum + RNG from the checkpoint and rebuild the cosine LambdaLR at
    # the saved step so the continuation is equivalent to an uninterrupted run. start_step is the number of
    # completed M-steps; the loop runs range(start_step, n_steps). resume_path None leaves the from-scratch
    # path byte-identical (scheduler built with last_epoch=-1, loop from 0).
    resume_path = resume_from if resume_from is not None else cfg.resume_from
    start_step = 0
    if device is None:
        device = model.prior_bank.mu_embed.device
    loader_sampler = getattr(loader, "sampler", None)
    shuffled_loader = isinstance(loader_sampler, torch.utils.data.RandomSampler)
    loader_generator = getattr(loader, "generator", None)
    resume_data_state: Dict[str, object] = {}
    if (resume_path is not None and shuffled_loader
            and not isinstance(loader_generator, torch.Generator)):
        raise RuntimeError(
            "exact shuffled resume requires loader.generator to expose a torch.Generator")
    # Metropolis det-sign sweep (opt-in, default OFF): a single persistent CPU generator, seeded
    # once from cfg.seed, threaded across every step so the accept/reject sequence is reproducible
    # (design spec Sec.6). It is constructed before resume so load_checkpoint can restore its private
    # state. Constructing/seeding a LOCAL torch.Generator never touches the global RNG stream, so this
    # stays inert when neither learnable-reflection mode is active.
    metro_gen = torch.Generator().manual_seed(int(cfg.seed))
    # fp16 training needs loss scaling (gradients underflow through the unrolled E-step); bf16/fp32
    # do not. enabled=False is a no-op, so non-fp16 amp_dtype keeps this loop byte-identical.
    # Created BEFORE the resume block so load_checkpoint can restore its scale/growth state
    # (audit 2026-06-09 IE3).
    scaler = torch.amp.GradScaler(device=device.type, enabled=(cfg.amp_dtype == "fp16"))
    # Opt-in EMA / Polyak averaging (default OFF -> ema is None and every ema-guarded block below is a
    # no-op, leaving the loop byte-identical to the pure path). Built BEFORE the resume load so a
    # resumed run restores its shadow from the bundle rather than re-seeding it from the resumed iterate.
    ema = EMA(model, decay=cfg.ema_decay) if cfg.use_ema else None
    if resume_path is not None:
        from vfe3.run_artifacts import load_checkpoint           # local import avoids any import cycle
        start_step = load_checkpoint(resume_path, model, optimizer, map_location=device,
                                     scaler=scaler, cfg=cfg, ema=ema, artifacts=artifacts,
                                     metropolis_generator=metro_gen,
                                     data_state=resume_data_state)
        if shuffled_loader and not resume_data_state:
            raise RuntimeError(
                "exact shuffled resume requires checkpoint data_state; this checkpoint predates "
                "shuffled iterator persistence")
        # LambdaLR with last_epoch != -1 requires 'initial_lr' on every group; set it from the configured
        # base (not the post-load group['lr'], which the restored optimizer state overwrote with base*cos).
        for group, base in zip(optimizer.param_groups, base_lrs):
            group["initial_lr"] = base
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, _floor_lr_lambdas(base_lrs, cfg), last_epoch=start_step - 1)
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _floor_lr_lambdas(base_lrs, cfg))
    # Unigram log-prior decode (cfg.decode_unigram_prior, default OFF): fill the PriorBank's
    # unigram table from the TRAINING stream once, before the loop -- a fixed data statistic
    # (add-one-smoothed log frequencies), not a learned parameter. The counts come from the
    # loader's TokenWindows dataset (its flat token stream); a loader without one warns and
    # leaves the table unset (the decode then warns once and is a value no-op).
    if cfg.decode_unigram_prior:
        _tok = getattr(getattr(loader, "dataset", None), "tokens", None)
        if _tok is not None:
            _counts = torch.bincount(_tok.reshape(-1), minlength=cfg.vocab_size).to(torch.float32)
            model.prior_bank.set_unigram_log_prior(_counts.to(device))
        else:
            import warnings
            warnings.warn(
                "decode_unigram_prior=True but the training loader exposes no flat token stream "
                "(loader.dataset.tokens); the unigram table stays unset (decode warns, value "
                "no-op). Call model.prior_bank.set_unigram_log_prior(counts) manually.",
                UserWarning, stacklevel=2)
    losses: List[float] = []
    model.train()
    logger = logger or logging.getLogger(__name__)
    # Live per-step it/s: iterate the step loop through a tqdm bar whose built-in rate readout
    # refreshes every step. Gated on log_interval so the documented silent path (log_interval
    # falsy) stays bitwise-identical -- no bar, no redirect, nothing printed. The generator holds
    # logging_redirect_tqdm open across the whole loop (it suspends at `yield` INSIDE the `with`),
    # so the periodic logger.info lines below render above the bar instead of interleaving with it
    # on stderr; it closes the bar on normal exit or exception.
    show_bar = bool(log_interval) and _tqdm is not None

    def _step_indices() -> Iterable[int]:
        if not show_bar:
            yield from range(start_step, n_steps)               # range start_step..n_steps (== 0..n_steps from scratch)
            return
        bar = _tqdm(range(start_step, n_steps), desc="Training", total=n_steps,
                    initial=start_step, ascii=True)             # ascii=True: the default block glyph
        #                          U+2588 is not cp1252-encodable on a Windows console (raises
        #                          UnicodeEncodeError mid-run); " #" renders anywhere. initial=start_step
        #                          keeps the bar's absolute step readout correct on a resumed run.
        with _redirect_logging():
            try:
                yield from bar
            finally:
                bar.close()

    epoch = 0
    batches_consumed = 0
    if resume_data_state:
        required_data_state = {"epoch_start_generator_state", "batches_consumed", "epoch"}
        missing_data_state = required_data_state - resume_data_state.keys()
        if missing_data_state:
            raise RuntimeError(
                f"checkpoint data_state is missing required field(s) {sorted(missing_data_state)}")
        if not isinstance(loader_generator, torch.Generator):
            raise RuntimeError(
                "exact data resume requires loader.generator to expose a torch.Generator")
        saved_generator_state = resume_data_state["epoch_start_generator_state"]
        if not isinstance(saved_generator_state, torch.Tensor):
            raise RuntimeError("checkpoint data_state epoch_start_generator_state must be a tensor")
        epoch = int(resume_data_state["epoch"])
        saved_batches_consumed = int(resume_data_state["batches_consumed"])
        if epoch < 0 or saved_batches_consumed < 0:
            raise RuntimeError("checkpoint data_state epoch and batches_consumed must be non-negative")
        epoch_start_generator_state = saved_generator_state.cpu().clone()
        loader_generator.set_state(epoch_start_generator_state)
    else:
        saved_batches_consumed = 0
        epoch_start_generator_state = (loader_generator.get_state().clone()
                                       if isinstance(loader_generator, torch.Generator) else None)
    it = iter(loader)
    for _ in range(saved_batches_consumed):
        try:
            next(it)
        except StopIteration as exc:
            raise RuntimeError(
                "checkpoint data_state cannot be replayed by the current loader: "
                "batches_consumed exceeds the saved epoch") from exc
        batches_consumed += 1
    win_t0 = time.perf_counter()
    train_t0 = win_t0                                 # cumulative wall-clock origin (D1/EXP-8; resets on resume)
    win_i0 = start_step
    last_val: Dict[str, float] = {}                  # most recent validation, carried into each CSV row
    last_val_diag: Dict[str, float] = {k: float("nan") for k in _VAL_DIAG_KEYS}   # held-out probes, carried forward
    for step in _step_indices():
        try:
            tokens, targets = next(it)
        except StopIteration:
            epoch += 1
            batches_consumed = 0
            epoch_start_generator_state = (loader_generator.get_state().clone()
                                           if isinstance(loader_generator, torch.Generator) else None)
            it = iter(loader)
            tokens, targets = next(it)
        batches_consumed += 1
        tokens = tokens.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        do_log  = bool(log_interval) and (step + 1) % log_interval == 0
        do_eval = bool(eval_interval) and val_loader is not None and (step + 1) % eval_interval == 0
        do_csv  = artifacts is not None and (do_log or do_eval)
        # Capture pre-clip gradient health only on a step that will log or persist, so the silent
        # hot path stays byte-identical (metrics_out=None -> no extra unscale_, no grad-norm pass).
        step_metrics: Optional[Dict[str, float]] = {} if (do_log or do_csv) else None
        if hasattr(optimizer, "_collect_gauge_diag"):        # D1/EXP-8: gate the gauge M-step diagnostics
            optimizer._collect_gauge_diag = bool(do_log or do_csv)   # to log/eval steps (silent path unchanged)
        if do_log or do_csv:                                 # off the hot path (audit 2026-07-05 m8)
            _warn_phi_transport_clamp(model)
            # m7: diagnostics on the PRE-step weights so the logged F-decomposition shares the pre-step
            # provenance of train_loss/train_ce (train_ce is captured pre-step inside train_step). Under
            # no_grad, so no weight change and no global-RNG draw (randomize_e_steps is gated on grad_on),
            # leaving the train_step RNG stream and weights byte-identical.
            d = model.diagnostics(tokens)
        losses.append(train_step(model, optimizer, scheduler, tokens, targets,
                                  grad_clip=grad_clip, grad_accum_steps=cfg.grad_accum_steps,
                                  scaler=scaler, metrics_out=step_metrics))
        # Metropolis det-sign sweep (opt-in, default OFF): runs on the POST-optimizer-step model,
        # gated + cadence-checked by the helper; inert (no call, no generator draw) unless
        # cfg.omega_reflection == 'metropolis'. tokens is the SAME input batch just fed to train_step.
        _maybe_metropolis_omega(model, tokens, step=step, generator=metro_gen)
        if ema is not None:
            ema.update(model)                            # blend the post-step weights into the shadow

        if do_log or do_csv:                                 # diagnostics (off graph), ONCE
            # train_ce is the PRE-step CE captured inside train_step (matches train_loss); the old
            # post-step re-forward made train_ce one optimizer step ahead of train_loss (audit r2 id5).
            ce = step_metrics.get("train_ce", float("nan")) if step_metrics is not None else float("nan")
            # d (the F-decomposition) is computed PRE-step above so it shares train_ce's provenance (m7).
            # Throughput + peak memory over the window since the last log/eval, then reset the window.
            # Computed here (not inside do_log) so a CSV row on an eval-only step still carries the rate.
            now = time.perf_counter()
            rate = (step + 1 - win_i0) / max(now - win_t0, 1e-9)             # optimizer steps/s
            toks_per_s = rate * int(tokens.shape[0]) * int(tokens.shape[1])  # tokens/s
            if torch.cuda.is_available():
                peak_mem_mb = torch.cuda.max_memory_allocated() / 1e6
                torch.cuda.reset_peak_memory_stats()
            else:
                peak_mem_mb = float("nan")
            win_t0, win_i0 = now, step + 1

        if do_log:
            logger.info(
                "Step %d/%d | Loss: %.4f | CE: %.4f | H(b): %.3f | it/s: %.2f | \n\n         Train PPL: %.1f \n",
                step + 1, n_steps, losses[-1], ce, d["attn_entropy"], rate, math.exp(min(ce, 20.0)),
            )
            logger.info(
                "    F: self %.4f | belief %.4f | entropy %.4f | total %.4f | eff_rank %.2f | BPC %.4f",
                d["self_coupling"], d["belief_coupling"], d["attention_entropy"],
                d["total"], d["effective_rank"], (ce / math.log(2.0)) * tokens_per_char,
            )

        if do_eval:
            if ema is not None:                          # eval / best-save / samples on the averaged weights
                ema.store(model)
                ema.copy_to(model)
            m = evaluate(model, val_loader, max_batches=cfg.eval_max_batches,
                         tokens_per_char=tokens_per_char, device=device)
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
                # Held-out per-eval probes (validation F decomposition, attention-map structure,
                # E-step convergence certificate, per-position loss). Best-effort: a replay error
                # RESETS the probes to NaN (blank CSV cells) so a previous eval's values are never
                # carried forward as if fresh (audit 2026-07-01 F11), and a viz/replay fault never
                # kills training.
                try:
                    last_val_diag.update(_val_diagnostics(model, val_loader, device))
                except Exception as exc:
                    logger.warning("       (validation diagnostics failed: %s); continuing", exc)
                    last_val_diag.update({k: float("nan") for k in _VAL_DIAG_KEYS})
                artifacts.maybe_save_best(step + 1, model, m["ppl"])
                # Per-layer/per-head attention heatmap grid for this eval (off the graph, seq 0 of
                # the live batch), plus the model-coupling (gamma) heatmaps in a distinct color
                # (viridis vs magma; gamma_attention_maps returns None when the model channel is
                # off -> no-op). The model REPLAYS (attention_maps / gamma_attention_maps) are
                # argument expressions evaluated HERE in the caller, OUTSIDE the save helpers'
                # internal try/except, so guard them too -- a replay error must never kill training
                # (audit 2026-07-01 F11). Kept at EVAL cadence (one grid per eval, not per log).
                try:
                    artifacts.save_attention_maps(step + 1, model.attention_maps(tokens), logger=logger)
                    artifacts.save_gamma_attention_maps(step + 1, model.gamma_attention_maps(tokens), logger=logger)
                except Exception as exc:
                    logger.warning("       (attention-map replay failed: %s); continuing", exc)
            if ema is not None:
                ema.restore(model)                       # live SGD weights back before the next train_step

        # Persistence is opt-in: with no artifacts object do_csv is False, so the silent/in-memory
        # path is unchanged. A metrics.csv row is written every LOG_INTERVAL (and every eval) -- the
        # dense per-step diagnostics off the graph. The EVAL-CADENCE columns (val_ce/ppl/bpc,
        # generalization_gap, and the held-out val_*/estep_*/pos_loss_* probes) carry a value ONLY on a
        # step where the eval above just ran (do_eval); on the denser log-interval rows in between they
        # are NaN, which log_metrics renders as a BLANK cell -- so each validation appears exactly once
        # per eval_interval, not carried forward to every log line. The
        # in-memory history keeps the NaN (figures already drop non-finite rows). The four F-stack
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
                "lr_mu":             float(lrs[0]),           # group 0 = mu_embed          (m_p_mu_lr)
                "lr_sigma":          float(lrs[1]),           # group 1 = sigma_log+decode  (m_p_sigma_lr)
                "lr_phi":            float(lrs[2]),           # group 2 = phi_embed         (m_phi_lr)
                "val_ce":            last_val["ce"]  if do_eval else float("nan"),  # eval-cadence: fresh on
                "val_ppl":           last_val["ppl"] if do_eval else float("nan"),  # an eval step (last_val just
                "val_bpc":           last_val["bpc"] if do_eval else float("nan"),  # refreshed), blank otherwise
                "attn_entropy":       d["attn_entropy"],
                "self_coupling":      d["self_coupling"]     / n_tok,   # alpha-regularized F self-term sum_i[alpha_i D + R(alpha_i)]
                "self_divergence":    d["self_divergence"]   / n_tok,   # raw sum_i D(q_i||p_i) drift; == self_coupling only at lambda_alpha_mode='constant'
                "belief_coupling":    d["belief_coupling"]   / n_tok,
                "attention_entropy":  d["attention_entropy"] / n_tok,
                "free_energy_total":  d["total"]             / n_tok,
                "effective_rank":     d["effective_rank"],
                "holonomy_deviation": d["holonomy_deviation"],
                "gauge_trace_spread": d["gauge_trace_spread"],
            }
            # Throughput + peak memory (Tier-1): tokens/s over the window and CUDA peak MB.
            row["tokens_per_s"] = toks_per_s
            row["peak_mem_mb"]  = peak_mem_mb
            # Learnable softmax temperatures (default-off): log the live kappa values once per
            # metrics row so finalize_run can plot their training trajectory. The variance is the
            # population variance across irrep blocks/heads at that step (0 for a single block).
            for _kp, _param in (("kappa_beta", getattr(model, "log_kappa_beta", None)),
                                ("kappa_gamma", getattr(model, "log_kappa_gamma", None))):
                if _param is not None:
                    _kv = torch.exp(_param.detach()).float().reshape(-1)
                    row[f"{_kp}_mean"] = float(_kv.mean())
                    row[f"{_kp}_var"]  = float(_kv.var(unbiased=False))
                    # Per-block companion to the aggregate mean/var: one column per irrep block for
                    # kappa_b and the effective softmax temperature tau_b = kappa_b * sqrt(d_b) (d_b
                    # the gauge-irrep block size), so finalize_run can draw a line per block in the
                    # kappa/tau panels. n_blocks is static per run -> the CSV stays rectangular.
                    _ch   = _kp.split("_", 1)[1]                   # "beta" | "gamma"
                    _dims = model.group.irrep_dims
                    for _bi in range(_kv.numel()):
                        _kb = float(_kv[_bi])
                        row[f"kappa_{_ch}_b{_bi}"] = _kb
                        row[f"tau_{_ch}_b{_bi}"]   = _kb * float(_dims[_bi]) ** 0.5
            # Cumulative wall time since training start (D1/EXP-8): the x-axis of the per-wall-clock
            # convergence curve (val_ppl vs wall_clock_s). 'now' is the window timestamp captured above.
            row["wall_clock_s"] = now - train_t0
            # Generalization gap (Tier-1): val-set CE minus the per-step train CE (positive = overfit,
            # the standard convention). The train side is seq-0 (diagnostics runs on seq 0) while val is
            # the token-weighted val-set mean, so read it as a TREND, not an absolute. Eval-cadence like
            # val_ce above: written only on an eval row, blank (NaN) on the log-interval rows in between.
            # (The complexity-vs-fit comparison is free_energy_total vs
            # val_ce, both already columns; a separate "elbo_ce_gap" would subtract a CE from a
            # complexity-only F -- d["total"] carries no -E_q[log p] data term -- so it is NOT emitted.)
            row["generalization_gap"] = (last_val["ce"] - ce) if do_eval else float("nan")
            # Extended per-eval diagnostics (Tier-1/2): already-reduced gauge / geometry / numerical-
            # health scalars from diagnostics() -- NOT per-token sums, so logged RAW (no /n_tok).
            # Conditional keys (connection_w_norm, head_mixer_drift) appear only with their toggle; the
            # config is fixed per run so the CSV stays rectangular.
            for _dk in ("holonomy_ci_lo", "holonomy_ci_hi", "holonomy_wilson",
                        "gauge_invariant_mean", "gauge_invariant_spread",
                        "phi_norm_mean", "phi_norm_std",
                        "belief_cond_median", "belief_cond_p95", "belief_cond_max", "belief_pd_margin",
                        "eff_rank_p5", "eff_rank_median", "eff_rank_p95",
                        "fisher_trace_mean", "fisher_trace_median",
                        "guard_sigma_floor_frac", "guard_sigma_ceil_frac",
                        "guard_energy_klmax_frac", "guard_selfdiv_klmax_frac",
                        "nonfinite_frac", "renyi_band_frac",
                        "attn_entropy_min", "attn_entropy_collapsed_heads",
                        "cocycle_residual", "vertex_cond_max", "sandwich_absmax", "transport_asymmetry",
                        "energy_abs_asymmetry", "energy_rel_asymmetry",
                        "gauge_head_aniso_mean", "gauge_head_logdet_spread",
                        "connection_w_norm", "connection_m_norm",
                        "connection_l_norm", "connection_l_offdiag_norm", "head_mixer_drift"):
                if _dk in d:
                    row[_dk] = d[_dk]
            # Gradient health (Tier-1/2): global + per-group (mu/sigma/phi) grad AND weight norms (so
            # the update-to-weight ratio is derivable with the logged LR), loss finiteness, the fp16
            # loss-scale, and the grad-accum token spread -- captured by train_step into step_metrics.
            if step_metrics:
                for _gk in ("grad_norm", "grad_norm_mu", "grad_norm_sigma", "grad_norm_phi",
                            "weight_norm_mu", "weight_norm_sigma", "weight_norm_phi",
                            "estep_grad_norm_mu", "estep_grad_norm_sigma", "estep_grad_norm_phi",
                            "loss_finite", "grad_finite", "step_skipped",
                            "grad_scale", "grad_accum_tok_spread"):
                    if _gk in step_metrics:
                        row[_gk] = step_metrics[_gk]
            # Gauge M-step geometry diagnostics (D1/EXP-8): cos(nat,grad) and the pullback metric
            # condition number, stashed by GaugeNaturalGradAdamW on this (log/eval) step. Written with a
            # FIXED key set per run (NaN default, like the _VAL_DIAG_KEYS block below) so the columns are
            # defined from the FIRST logged row regardless of active-rows timing -- log_metrics locks
            # fieldnames on row 0, so a key first appearing later would break the CSV. cos_nat_phi for any
            # natural-grad gauge run; pullback_cond_* only on the pullback modes (config-fixed per run);
            # plain AdamW has no _gauge_diag attr, so its columns are absent and that CSV stays rectangular.
            if hasattr(optimizer, "_gauge_diag"):
                _gd = optimizer._gauge_diag or {}
                row["cos_nat_phi"] = _gd.get("cos_nat_phi", float("nan"))
                if getattr(optimizer, "_precond_mode", "") in ("pullback", "pullback_per_block"):
                    row["pullback_cond_median"] = _gd.get("pullback_cond_median", float("nan"))
                    row["pullback_cond_max"]    = _gd.get("pullback_cond_max", float("nan"))
            # Held-out per-eval probes (Tier-2): the full fixed val-diag column set. Eval-cadence like
            # val_ce above -- the fresh probe values on an eval row, NaN (rendered blank) on the
            # log-interval rows in between, NOT carried forward. The key set is identical in both
            # branches (_VAL_DIAG_KEYS), so the CSV stays rectangular.
            row.update(last_val_diag if do_eval else {k: float("nan") for k in _VAL_DIAG_KEYS})
            # Model-channel F blocks (per-token, like the belief blocks above): present iff the
            # s-channel tables exist (diagnostics gates them on STATIC config -- lambda_h / gamma /
            # prior_source / s_e_step), so the column set is fixed per run and the CSV stays
            # rectangular. free_energy_total above already carries their WEIGHTED contribution
            # (diagnostics folds it into d["total"] at the same per-sequence-sum scale, so the
            # uniform /n_tok normalizes every block consistently -- audit obs 18497). The raw blocks
            # are stored for the model_channel_terms figure; hyper_prior_weighted is the EXACT weighted
            # contribution folded into total (state_dependent lambda_h != cfg.lambda_h*raw,
            # so the F-decomposition figure reads this directly), while the gamma block is scaled by
            # cfg.lambda_gamma in that figure, exactly as the belief block is scaled by lambda_beta.
            for _mck in ("hyper_prior", "hyper_prior_weighted", "gamma_coupling", "gamma_meta_entropy"):
                if _mck in d:
                    row[_mck] = d[_mck] / n_tok
            artifacts.log_metrics(row)

        # Periodic resumable checkpoint (opt-in; needs the artifacts dir and the optimizer state).
        if (artifacts is not None and cfg.checkpoint_interval
                and (step + 1) % cfg.checkpoint_interval == 0):
            checkpoint_data_state = ({
                "epoch_start_generator_state": epoch_start_generator_state,
                "batches_consumed":            batches_consumed,
                "epoch":                       epoch,
            } if epoch_start_generator_state is not None else None)
            artifacts.save_checkpoint(step + 1, model, optimizer, cfg, scaler=scaler, ema=ema,
                                      metropolis_generator=metro_gen,
                                      data_state=checkpoint_data_state)
    if ema is not None:
        ema.copy_to(model)                               # the trained model IS the averaged weights
    return losses


def coverage_lines(
    loader:     object,                  # DataLoader over a TokenWindows stream
    n_steps:    int,                     # optimizer steps == batches consumed (grad_accum subdivides one batch)
    dataset:    str,

    *,
    full_corpus_tokens: Optional[int] = None,   # uncapped corpus size; emits a "stream is X% of full" line when it exceeds the loaded stream
) -> List[str]:
    r"""Banner lines reporting corpus coverage and epoch count for a training run.

    One optimizer step pulls exactly one ``batch_size`` window-batch from ``loader``
    (``grad_accum_steps`` only ``torch.chunk``-subdivides that single batch -- it draws no extra
    batches), so ``n_steps`` is the number of batches consumed. With ``steps_per_epoch = len(loader)``
    (already ``drop_last``-aware), ``epochs = n_steps / steps_per_epoch``. The default
    ``stride == seq_len`` tiles the corpus exactly once per epoch, so unique corpus coverage saturates
    at one epoch (``min(1, epochs)``); past that the corpus is re-seen ``epochs`` times.
    ``tokens_seen = n_steps * batch_size * seq_len`` is the raw token throughput (counts re-seen tokens).
    """
    ds = loader.dataset
    T = int(ds.tokens.numel())                       # tokens in the (possibly capped) stream
    seq_len = int(ds.seq_len)
    stride = int(ds.stride)
    batch = int(loader.batch_size)
    windows = len(ds)
    spe = len(loader)                                # batches per epoch (honours drop_last)
    epochs = (n_steps / spe) if spe > 0 else float("nan")
    tokens_seen = n_steps * batch * seq_len

    lines = [
        f" data: {dataset}  tokens={T:,}  windows={windows:,}  seq_len={seq_len} stride={stride}",
    ]
    if dataset == "synthetic-period3":               # "% of wiki" is meaningless on the synthetic anchor
        lines.append(f" coverage: epochs={epochs:.2f}  steps/epoch={spe:,}  tokens_seen={tokens_seen:,}")
    else:
        cov_pct = min(1.0, epochs) * 100.0           # unique fraction of corpus, saturating at one epoch
        passes = f"  ({epochs:.2f}x passes)" if epochs >= 1.0 else ""
        lines.append(
            f" coverage: epochs={epochs:.2f}  corpus={cov_pct:.1f}%{passes}  "
            f"steps/epoch={spe:,}  tokens_seen={tokens_seen:,}"
        )
        if full_corpus_tokens is not None and full_corpus_tokens > T:
            frac = T / full_corpus_tokens * 100.0
            lines.append(f" stream: {T:,}/{full_corpus_tokens:,} tokens = {frac:.1f}% of full {dataset} (capped)")
    return lines


def _fmt_tau(cfg: VFE3Config, model: VFEModel) -> str:
    r"""Banner format for the softmax temperature: a scalar kappa -> '1.5000', a per-head
    (list) kappa -> '[t0, t1, ...]' (T1). Converts a list kappa to a tensor first so attention_tau
    does not choke on a raw list."""
    dev = model.prior_bank.mu_embed.device
    tau = attention_tau(_as_coeff(cfg.kappa_beta, dev), model.group.irrep_dims)
    if isinstance(tau, torch.Tensor) and tau.dim() >= 1:
        return "[" + ", ".join(f"{x:.4f}" for x in tau.tolist()) + "]"
    return f"{float(tau):.4f}"


def parameter_report(
    model:  VFEModel,

    *,
    device: Optional[torch.device] = None,
    batch:  int = 2,
    seqlen: int = 16,
) -> Dict[str, object]:
    r"""Total params plus the ACTUAL-trained-under-this-config count.

    ``total`` is ``sum(p.numel())``; ``trainable`` filters ``requires_grad`` (catches the frozen
    hyper-prior centroid r under ``learnable_r=False``). ``live`` is MEASURED, not inferred from
    toggles: one synthetic forward+backward runs and a parameter counts as live only if it receives
    a non-None gradient -- so config-dead tables that stay allocated and grouped
    (``mu_embed``/``sigma_log_embed`` under ``prior_source='model_channel'``, ``decode_log_scale``
    under ``use_prior_bank=False``, ``phi_embed`` under ``detach_e_step=True``, ...) are counted as
    dead REGARDLESS of which toggle silenced them, so the report never drifts from the config as the
    toggles change. The synthetic ids come from a LOCAL generator (the global RNG is untouched) and
    ``.grad`` is cleared afterward, so the probe is side-effect-free. Best-effort: any failure (e.g. a
    config whose loss does not require grad) falls back to ``live = trainable`` with ``probed=False``.
    """
    named     = list(model.named_parameters())
    total     = sum(p.numel() for _, p in named)
    trainable = sum(p.numel() for _, p in named if p.requires_grad)
    rep: Dict[str, object] = {"total": total, "trainable": trainable, "live": trainable,
                              "dead": 0, "dead_names": [], "probed": False}
    try:
        device = device or next(model.parameters()).device
        n = max(2, min(int(seqlen), int(model.cfg.max_seq_len)))
        g = torch.Generator().manual_seed(0)                       # local: global RNG untouched
        ids = torch.randint(0, int(model.cfg.vocab_size), (int(batch), n), generator=g).to(device)
        tgt = torch.randint(0, int(model.cfg.vocab_size), (int(batch), n), generator=g).to(device)
        model.zero_grad(set_to_none=True)
        rng_state = torch.get_rng_state()                          # m31: the probe forward may draw the
        try:                                                       # global RNG (randomize_e_steps); restore it
            out  = model(ids, tgt)                                 # (logits|None, loss, ce) with targets
            loss = out[1] if isinstance(out, tuple) else out
            loss.backward()
        finally:
            torch.set_rng_state(rng_state)                         # global stream untouched, per the docstring
        dead_names = [name for name, p in named if p.requires_grad and p.grad is None]
        live = sum(p.numel() for _, p in named if p.requires_grad and p.grad is not None)
        model.zero_grad(set_to_none=True)                          # leave no grads for training
        rep.update(live=live, dead=trainable - live, dead_names=dead_names, probed=True)
    except Exception as exc:
        rep["probe_error"] = repr(exc)                             # surfaced, not swallowed; live/dead stay unknown
    return rep


def _banner(
    model:       VFEModel,
    cfg:         VFE3Config,
    dataset:     str,
    device:      torch.device,
    n_steps:     int,
    train_loader: object = None,
) -> str:
    r"""Compact init banner (no FLOPs counter; lambda_h is omitted from the printed lines)."""
    rep = parameter_report(model, device=device)
    bar = "=" * 64
    cov = coverage_lines(train_loader, n_steps, dataset) if train_loader is not None else []
    live_note = (f" ({rep['live']:,} live, {rep['dead']:,} dead)"
                 if rep["probed"] and rep["dead"] else "")
    probe_note = "" if rep["probed"] else "  [live/dead probe failed]"   # distinguish failure from all-live
    dead_line = ([" dead under config (no grad): "
                  + ", ".join(n.replace("prior_bank.", "") for n in rep["dead_names"])]
                 if rep["probed"] and rep["dead_names"] else [])
    return "\n".join([
        bar,
        f" Gauge VFE Transformer | {rep['total']:,} params{live_note}{probe_note} | {device}",
        bar,
        f" K={cfg.embed_dim}  N={cfg.max_seq_len}  L={cfg.n_layers}  heads={len(model.group.irrep_dims)}  "
        f"group={cfg.gauge_group}  family={cfg.family}",
        f" steps={n_steps}  batch={cfg.batch_size}  dataset={dataset}",
        *cov,
        *dead_line,
        f" M-LRs: mu={cfg.m_p_mu_lr}  sigma={cfg.m_p_sigma_lr}  phi={cfg.m_phi_lr}",
        f" VFE: lambda_alpha={cfg.lambda_alpha}  kappa_beta={cfg.kappa_beta}  "
        f"tau={_fmt_tau(cfg, model)}  mass_phi={cfg.mass_phi}",
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
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VFEModel(cfg).to(device)         # move to CUDA where available (mirrors train_vfe3.main)
    loader = make_dataloader(dataset, split, cfg.max_seq_len, cfg.batch_size, max_tokens=max_tokens)
    logger = logging.getLogger(__name__)
    logger.info(_banner(model, cfg, dataset, device, n_steps, train_loader=loader))
    losses = train(
        model, loader, cfg,
        n_steps=n_steps,
        log_interval=cfg.log_interval,
        eval_interval=cfg.eval_interval,
        val_loader=loader,
        device=device,
    )
    return model, losses
