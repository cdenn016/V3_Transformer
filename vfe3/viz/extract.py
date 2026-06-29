r"""Belief / evaluation EXTRACTORS for VFE_3.0 publication figures.

These RUN the model -- reload-and-recompute converged beliefs, loop the E-step capturing the
belief trajectory, replay the block stack per layer, score a loader per unit, or tally numerical
health. They have side effects / drive the model, so they live here rather than in
``vfe3.metrics`` (whose contract is pure, side-effect-free measurement). Each returns plain
tensors / dicts that the pure metrics and the figure functions consume. Everything runs under
``torch.no_grad`` and OFF the training hot path.
"""

from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

from vfe3 import metrics
from vfe3.alpha_i import self_coupling_alpha
from vfe3.belief import BeliefState
from vfe3.families.base import get_family
from vfe3.model.block import _as_coeff       # list b0/c0 -> (K,) tensor (M6), mirroring vfe_block
from vfe3.free_energy import (
    attention_tau,
    attention_weights,
    pairwise_energy,
    self_divergence_for_alpha,
)
from vfe3.inference.e_step import _transport, e_step_iteration, free_energy_value
from vfe3.geometry.transport import transport_covariance, transport_mean
from vfe3.model.block import vfe_block


def _model_device(model) -> torch.device:
    return model.prior_bank.mu_embed.device


def _lambda_beta(model) -> 'float | torch.Tensor':
    r"""The live belief-coupling weight (constant cfg.lambda_beta, or exp(log_lambda_beta))."""
    llb = getattr(model, "log_lambda_beta", None)
    return model.cfg.lambda_beta if llb is None else llb.exp()


def _encode_one(model, token_ids: torch.Tensor) -> Tuple[BeliefState, torch.Tensor, Optional[torch.Tensor]]:
    r"""Encode sequence 0 to the initial belief (pos_phi applied), with its log-prior and RoPE.

    Under ``s_e_step`` the live model channel is replayed too (audit 2026-06-09 IE2): s is refined
    with the frozen gauge frame and the belief is anchored to it, exactly as ``forward`` /
    ``diagnostics`` do -- the callers' ``mu_p = belief.mu`` handoff then anchors to the refined s,
    so every extracted trajectory/figure describes the model that actually trained."""
    enc = model.prior_bank.encode(token_ids[:1])                  # (1, N, ...)
    belief = BeliefState(mu=enc.mu[0], sigma=enc.sigma[0], phi=model._apply_pos_phi(enc.phi[0]))
    if model.cfg.s_e_step:
        s_mu1, s_sigma1 = model._refine_s(token_ids[:1], belief.phi.unsqueeze(0))
        belief = belief._replace(mu=s_mu1[0], sigma=s_sigma1[0])
    n = belief.mu.shape[0]
    log_prior = model._attention_log_prior(n, token_ids.device)
    rope = model._rope_rotation(n, token_ids.device)
    return belief, log_prior, rope


def _iter_kwargs(model, log_prior: torch.Tensor, rope: Optional[torch.Tensor]) -> dict:
    r"""The full ``e_step_iteration`` knob bag assembled from the model config (mirrors vfe_block)."""
    cfg = model.cfg
    return dict(
        tau=attention_tau(_as_coeff(cfg.kappa_beta, model.prior_bank.mu_embed.device), model.group.irrep_dims),
        e_q_mu_lr=cfg.e_q_mu_lr, e_q_sigma_lr=cfg.e_q_sigma_lr, e_phi_lr=cfg.e_phi_lr,
        renyi_order=cfg.renyi_order, value=cfg.lambda_alpha,
        b0=_as_coeff(cfg.b0, model.prior_bank.mu_embed.device),
        c0=_as_coeff(cfg.c0, model.prior_bank.mu_embed.device),
        lambda_beta=_lambda_beta(model), kl_max=cfg.kl_max, eps=cfg.eps,
        sigma_max=cfg.sigma_max, e_sigma_q_trust=cfg.e_sigma_q_trust, mass_phi=cfg.mass_phi,
        e_mu_q_trust=cfg.e_mu_q_trust, mu_trust_mode=cfg.mu_trust_mode,
        e_step_mu_precond=cfg.e_step_mu_precond,
        include_attention_entropy=cfg.include_attention_entropy, gradient_mode=cfg.gradient_mode,
        family=cfg.family, divergence_family=cfg.divergence_family, lambda_alpha_mode=cfg.lambda_alpha_mode,
        phi_precond_mode=cfg.phi_precond_mode, phi_retract_mode=cfg.phi_retract_mode,
        spd_retract_mode=cfg.spd_retract_mode, transport_mode=cfg.transport_mode,
        cocycle_relaxation=cfg.cocycle_relaxation, connection_W=getattr(model, "connection_W", None),
        connection_M=getattr(model, "connection_M", None),
        log_prior=log_prior, log_alpha=getattr(model, "log_alpha", None),
        rope=rope, rope_on_cov=cfg.rope_full_gauge, rope_on_value=cfg.rope_on_value,
    )


def _fe_kwargs(model, log_prior: torch.Tensor, rope: Optional[torch.Tensor] = None) -> dict:
    r"""The ``free_energy_value`` knob subset (it rejects the iteration-only step-size knobs).
    ``rope`` is honored (audit PP6): the logged F carries the RoPE-wrapped transport."""
    cfg = model.cfg
    return dict(
        tau=attention_tau(_as_coeff(cfg.kappa_beta, model.prior_bank.mu_embed.device), model.group.irrep_dims),
        renyi_order=cfg.renyi_order, value=cfg.lambda_alpha,
        b0=_as_coeff(cfg.b0, model.prior_bank.mu_embed.device),
        c0=_as_coeff(cfg.c0, model.prior_bank.mu_embed.device),
        lambda_beta=_lambda_beta(model), kl_max=cfg.kl_max, eps=cfg.eps,
        include_attention_entropy=cfg.include_attention_entropy, family=cfg.family,
        divergence_family=cfg.divergence_family, lambda_alpha_mode=cfg.lambda_alpha_mode,
        transport_mode=cfg.transport_mode, cocycle_relaxation=cfg.cocycle_relaxation,
        log_prior=log_prior, log_alpha=getattr(model, "log_alpha", None),
        connection_W=getattr(model, "connection_W", None),
        connection_M=getattr(model, "connection_M", None),
        rope=rope, rope_on_cov=cfg.rope_full_gauge, rope_on_value=cfg.rope_on_value,
    )


@torch.no_grad()
def per_unit_eval_nats(
    model,
    loader:      Iterable[Tuple[torch.Tensor, torch.Tensor]],   # yields (tokens, targets) batches

    *,
    device:      Optional[torch.device] = None,
    max_batches: Optional[int]          = None,
) -> Dict[str, torch.Tensor]:
    r"""Per-SEQUENCE and per-TOKEN cross-entropy (nats) over a loader.

    The training ``evaluate`` retains only the aggregate token-weighted {ce, ppl, bpc}; the
    single-seed bootstrap bands (``metrics.bootstrap_ce_band`` over sequences,
    ``metrics.bootstrap_token_ce_band`` over tokens) need the per-unit nats this produces. Runs
    ``model(tokens)`` for full logits (the inference path; the fused chunked decode is bypassed by
    passing no targets) and reduces the cross-entropy with ``reduction='none'``.

    Returns ``per_seq_nats`` (S,) summed nats per sequence, ``per_seq_tokens`` (S,) the
    non-ignored token count per sequence, and ``per_token_nats`` (M,) the valid per-position nats.
    """
    device = device or _model_device(model)
    was_training = model.training
    model.eval()
    seq_nats: List[torch.Tensor] = []
    seq_tok: List[torch.Tensor] = []
    tok_nats: List[torch.Tensor] = []
    try:
        for i, (tokens, targets) in enumerate(loader):
            tokens = tokens.to(device)
            targets = targets.to(device)
            b, n = tokens.shape
            logits = model(tokens)                                # (B, N, V) inference path
            v = logits.shape[-1]
            per = F.cross_entropy(logits.reshape(-1, v).float(), targets.reshape(-1),
                                  ignore_index=-100, reduction="none").reshape(b, n)
            valid = (targets != -100)
            seq_nats.append((per * valid).sum(dim=1))
            seq_tok.append(valid.sum(dim=1).to(per.dtype))
            tok_nats.append(per[valid])
            if max_batches is not None and i + 1 >= max_batches:
                break
    finally:
        if was_training:
            model.train()
    return {
        "per_seq_nats":   torch.cat(seq_nats),
        "per_seq_tokens": torch.cat(seq_tok),
        "per_token_nats": torch.cat(tok_nats),
    }


@torch.no_grad()
def belief_ce_bank(
    model,
    loader:      Iterable[Tuple[torch.Tensor, torch.Tensor]],   # yields (tokens, targets) batches

    *,
    device:      Optional[torch.device] = None,
    max_batches: Optional[int]          = None,
) -> Dict[str, torch.Tensor]:
    r"""The Sigma_q <-> CE JOIN for the calibration probe (B1/EXP-3): per-token belief-covariance
    trace tr(Sigma_q) aligned with the decode cross-entropy that same belief produced.

    For each ``(tokens, targets)`` batch this REPLAYS model.forward's belief path EXACTLY
    (``prior_bank.encode`` -> pos_phi -> the s-refine anchor under ``s_e_step`` -> the precision-bias
    fold -> ``vfe_stack`` with the trained head_mixer / cg_coupling / block_norm) for the per-token
    covariance ``out.sigma``, AND the inference forward ``model(tokens)`` for the decode logits, then
    reduces the cross-entropy with ``reduction='none'``. The s-refine + precision fold are the two
    steps belief_bank omits; they are reinstated here (no-ops on the pure default path) so the traced
    Sigma_q IS the belief whose mean produced the logits -- otherwise tr(Sigma_q) and CE come from
    different beliefs under the live ``s_e_step=True`` / ``precision_weighted_attention=True`` config
    and the calibration headline reads off the wrong covariance. The belief at position n is the one
    whose mean produced the logits predicting target n, so tr(Sigma_q)[n] and CE[n] are aligned
    position-by-position; only valid targets (``targets != -100``) are kept. Returns the flattened
    per-token ``tr_sigma`` (M,), ``ce`` (M,) nats, the predicted-token ``token_ids`` (M,), and the
    per-token ``conf`` (M,) max-softmax probability and ``correct`` (M,) argmax-equals-gold indicator --
    the aligned pairs the Spearman rho / CV gate (``vfe3.metrics.spearman_rho`` / ``cv``), the
    sigma-validation gate (``vfe3.inference.sigma_gate``), and the Sigma-stratified-error /
    Sigma-CE-scatter figures consume. ``max_batches`` caps the join.
    """
    from vfe3.model.stack import vfe_stack
    device = device or _model_device(model)
    cfg = model.cfg
    was_training = model.training
    model.eval()
    tr_sig, ces, tids, confs, corrects = [], [], [], [], []
    try:
        for i, (tokens, targets) in enumerate(loader):
            tokens = tokens.to(device)
            targets = targets.to(device)
            n = tokens.shape[1]
            logits = model(tokens)                                # (B, N, V) inference path
            flat_logits = logits.reshape(-1, logits.shape[-1]).float()           # (B*N, V)
            per = F.cross_entropy(flat_logits, targets.reshape(-1),
                                  ignore_index=-100, reduction="none")            # (B*N,)
            conf_flat, pred_flat = flat_logits.softmax(dim=-1).max(dim=-1)        # (B*N,) confidence, argmax
            beliefs = model.prior_bank.encode(tokens)             # mirror forward so the traced sigma
            beliefs = beliefs._replace(phi=model._apply_pos_phi(beliefs.phi))   # IS the decode's belief
            if cfg.s_e_step:                                       # anchor belief to the refined model channel
                s_mu1, s_sigma1 = model._refine_s(tokens, beliefs.phi)
                beliefs = beliefs._replace(mu=s_mu1, sigma=s_sigma1)
            log_prior = model._attention_log_prior(n, device)
            log_prior = model._fold_precision_bias(log_prior, beliefs.sigma)   # no-op unless precision_weighted_attention
            rope = model._rope_rotation(n, device)
            out = vfe_stack(
                beliefs, beliefs.mu, beliefs.sigma, model.group, cfg,
                log_prior=log_prior, block_norm=model.block_norm,
                head_mixer=model.head_mixer, cg_coupling=model.cg_coupling,
                log_alpha=getattr(model, "log_alpha", None), lambda_beta=_lambda_beta(model),
                connection_W=getattr(model, "connection_W", None),
                connection_M=getattr(model, "connection_M", None),
                rope=rope, rope_on_cov=cfg.rope_full_gauge, rope_on_value=cfg.rope_on_value,
            )
            trs = metrics.sigma_trace(out.sigma, diagonal=cfg.diagonal_covariance).reshape(-1)   # (B*N,)
            tgt = targets.reshape(-1)
            valid = tgt != -100
            tr_sig.append(trs[valid])
            ces.append(per[valid])
            tids.append(tgt[valid])
            confs.append(conf_flat[valid])
            corrects.append((pred_flat[valid] == tgt[valid]).float())
            if max_batches is not None and i + 1 >= max_batches:
                break
    finally:
        if was_training:
            model.train()
    return {
        "tr_sigma":  torch.cat(tr_sig),
        "ce":        torch.cat(ces),
        "token_ids": torch.cat(tids),
        "conf":      torch.cat(confs),               # per-token max softmax prob (sigma-gate ECE)
        "correct":   torch.cat(corrects),            # per-token 1.0 if argmax == gold (sigma-gate ECE)
    }


@torch.no_grad()
def belief_bank(
    model,
    token_batches:  Iterable[torch.Tensor],   # iterable of (B, N) token-id batches

    *,
    device:         Optional[torch.device] = None,
    max_sequences:  Optional[int]          = None,
) -> Dict[str, torch.Tensor]:
    r"""Collect converged beliefs (mu, Sigma, phi) over many sequences into one bank.

    For each batch runs the model's belief pipeline (prior_bank.encode -> pos_phi -> vfe_stack
    with the SAME per-block head_mixer / cg_coupling / block_norm the training forward applies,
    mirroring forward up to the stack's handoff belief; only final-norm decode prep is omitted) and
    stacks the per-token converged ``mu`` (M, K), ``sigma`` (M, K) or (M, K, K), ``phi``
    (M, n_gen), with ``token_ids`` (M,) and ``seq_idx`` (M,). Feeds the mu / Sigma / phi UMAP
    triptych and the at-scale clustering scores.
    """
    from vfe3.model.stack import vfe_stack
    device = device or _model_device(model)
    cfg = model.cfg
    was_training = model.training
    model.eval()
    mus, sigmas, phis, tids, sidx = [], [], [], [], []
    seq_counter = 0
    try:
        for tokens in token_batches:
            tokens = tokens.to(device)
            beliefs = model.prior_bank.encode(tokens)
            beliefs = beliefs._replace(phi=model._apply_pos_phi(beliefs.phi))
            n = tokens.shape[1]
            log_prior = model._attention_log_prior(n, device)
            rope = model._rope_rotation(n, device)
            out = vfe_stack(
                beliefs, beliefs.mu, beliefs.sigma, model.group, cfg,
                log_prior=log_prior, block_norm=model.block_norm,
                head_mixer=model.head_mixer, cg_coupling=model.cg_coupling,   # replay the trained
                log_alpha=getattr(model, "log_alpha", None), lambda_beta=_lambda_beta(model),  # model
                connection_W=getattr(model, "connection_W", None),
                connection_M=getattr(model, "connection_M", None),
                rope=rope, rope_on_cov=cfg.rope_full_gauge, rope_on_value=cfg.rope_on_value,
            )
            b = tokens.shape[0]
            mus.append(out.mu.reshape(b * n, -1))
            sigmas.append(out.sigma.reshape(b * n, *out.sigma.shape[2:]))
            phis.append(out.phi.reshape(b * n, -1))
            tids.append(tokens.reshape(b * n))
            sidx.append(torch.arange(seq_counter, seq_counter + b, device=device).repeat_interleave(n))
            seq_counter += b
            if max_sequences is not None and seq_counter >= max_sequences:
                break
    finally:
        if was_training:
            model.train()
    return {
        "mu":        torch.cat(mus),
        "sigma":     torch.cat(sigmas),
        "phi":       torch.cat(phis),
        "token_ids": torch.cat(tids),
        "seq_idx":   torch.cat(sidx),
    }


@torch.no_grad()
def e_step_belief_trace(
    model,
    token_ids: torch.Tensor,           # (B, N) token ids; only sequence 0 is used

    *,
    n_iter:    Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    r"""Loop the inner E-step capturing the FULL belief (and F) at every iteration for one sequence.

    The E-step ``return_trajectory`` yields the free-energy floats only; this captures the belief
    tuple (mu, sigma, phi) per inner iteration so the belief PATH and the SPD-metric residuals can
    be drawn. ``n_iter`` defaults to the trained ``cfg.n_e_steps`` (crank it up to show convergence
    past the trained budget). Returns ``mu`` (T+1, N, K), ``sigma`` (T+1, N, [K]), ``phi``
    (T+1, N, n_gen), and ``free_energy`` (T+1,) the global F at each iterate.
    """
    device = token_ids.device
    n_iter = int(n_iter if n_iter is not None else model.cfg.n_e_steps)
    belief, log_prior, rope = _encode_one(model, token_ids)
    mu_p, sigma_p = belief.mu, belief.sigma
    ikw = _iter_kwargs(model, log_prior, rope)
    fkw = _fe_kwargs(model, log_prior, rope)

    mus = [belief.mu]
    sigmas = [belief.sigma]
    phis = [belief.phi]
    fs = [free_energy_value(belief, mu_p, sigma_p, model.group, **fkw)]
    for _ in range(n_iter):
        belief = e_step_iteration(belief, mu_p, sigma_p, model.group, **ikw)
        mus.append(belief.mu)
        sigmas.append(belief.sigma)
        phis.append(belief.phi)
        fs.append(free_energy_value(belief, mu_p, sigma_p, model.group, **fkw))
    return {
        "mu":           torch.stack(mus),
        "sigma":        torch.stack(sigmas),
        "phi":          torch.stack(phis),
        "free_energy":  torch.stack([f.reshape(()) for f in fs]),
    }


@torch.no_grad()
def across_layer_belief_trace(
    model,
    token_ids: torch.Tensor,           # (B, N) token ids; only sequence 0 is used
) -> Dict[str, torch.Tensor]:
    r"""Per-LAYER converged beliefs by replaying the block stack (mirrors vfe_stack's handoff).

    Returns ``mu`` (L, N, K), ``sigma`` (L, N, [K]), the cumulative affine-invariant SPD geodesic
    distance ``d_ai`` (L,) of each layer's covariance from layer 0, and per-layer mean effective
    rank ``effective_rank`` (L,). Shows how the belief geometry transforms with inference depth.
    """
    cfg = model.cfg
    belief, log_prior, rope = _encode_one(model, token_ids)
    mu_p, sigma_p = belief.mu, belief.sigma
    rho, rho_s = cfg.prior_handoff_rho, cfg.prior_handoff_sigma
    mus, sigmas = [], []
    for _ in range(cfg.n_layers):
        belief = vfe_block(
            belief, mu_p, sigma_p, model.group, cfg, log_prior=log_prior,
            block_norm=model.block_norm, head_mixer=model.head_mixer,
            cg_coupling=model.cg_coupling,                       # replay the trained model
            log_alpha=getattr(model, "log_alpha", None),
            lambda_beta=_lambda_beta(model), connection_W=getattr(model, "connection_W", None),
            connection_M=getattr(model, "connection_M", None),
            rope=rope, rope_on_cov=cfg.rope_full_gauge, rope_on_value=cfg.rope_on_value,
        )
        mus.append(belief.mu)
        sigmas.append(belief.sigma)
        mu_p = (1.0 - rho) * mu_p + rho * belief.mu
        sigma_p = (1.0 - rho_s) * sigma_p + rho_s * belief.sigma
    mu_stack = torch.stack(mus)                                   # (L, N, K)
    sig_stack = torch.stack(sigmas)                              # (L, N, [K])
    base = sig_stack[0].unsqueeze(0).expand_as(sig_stack)
    d_ai = metrics.spd_geodesic_distance(base, sig_stack).mean(dim=-1)   # (L,) mean over tokens
    eff = metrics.effective_rank_per_token(sig_stack).mean(dim=-1)       # (L,)
    r_one = metrics.rank_one_residual(mu_stack)                          # (L,) Dong r(X) per layer (F2/EXP-7)
    return {"mu": mu_stack, "sigma": sig_stack, "d_ai": d_ai,
            "effective_rank": eff, "rank_one_residual": r_one}


@torch.no_grad()
def numerical_health(
    model,
    token_ids: torch.Tensor,           # (B, N) token ids; only sequence 0 is used
) -> Dict[str, float]:
    r"""Non-finite fractions of the converged intermediates + the worst covariance conditioning.

    Replays the converged belief and its energy / attention (as model.diagnostics does) and
    reports ``nan_inf_fraction`` for mu, sigma, phi, energy, and beta, plus the maximum spectral
    condition number of the belief covariances. A near-zero finiteness map certifies stability is
    genuine convergence, not masked blow-ups. (Numerical-FALLBACK activation counters --
    safe_cholesky jitter rounds, pinv fallbacks -- would require instrumenting numerics and are
    left to a future pass.)
    """
    from vfe3.numerics import condition_number, nan_inf_fraction
    cfg = model.cfg
    belief, log_prior, rope = _encode_one(model, token_ids)
    ikw = _iter_kwargs(model, log_prior, rope)
    out = belief
    for _ in range(cfg.n_e_steps):
        out = e_step_iteration(out, belief.mu, belief.sigma, model.group, **ikw)
    # Build Omega under the ACTIVE connection regime (audit 2026-06-10 F8e): this previously
    # defaulted to flat transport, so under regime_ii the reported nan/beta/energy fractions
    # described a flat-transport belief, not the model that trained. Mirrors converged_state.
    omega = _transport(
        out.phi, model.group, transport_mode=cfg.transport_mode,
        mu=(out.mu if cfg.transport_mode in ("regime_ii", "regime_ii_covariant") else None),
        sigma=(out.sigma if cfg.transport_mode == "regime_ii_covariant" else None),
        connection_W=getattr(model, "connection_W", None),
        connection_M=getattr(model, "connection_M", None),
        cocycle_relaxation=cfg.cocycle_relaxation,
    )
    # Wrap in RopeTransport under pos_rotation='rope' so the reported nan/energy/beta fractions
    # describe the RoPE-rotated belief the model runs, mirroring converged_state/diagnostics (r2 id11).
    if rope is not None:
        rope_omega = RopeTransport(base=omega, rope=rope, on_cov=cfg.rope_full_gauge,
                                   on_value=cfg.rope_on_value)
        mu_t    = transport_mean(rope_omega, out.mu)
        sigma_t = transport_covariance(rope_omega, out.sigma)
    else:
        mu_t = transport_mean(omega.unsqueeze(0), out.mu.unsqueeze(0))[0]
        sigma_t = transport_covariance(omega.unsqueeze(0), out.sigma.unsqueeze(0))[0]
    fam = get_family(cfg.family)
    energy = pairwise_energy(fam(out.mu, out.sigma), fam(mu_t, sigma_t), alpha=cfg.renyi_order,
                             kl_max=cfg.kl_max, eps=cfg.eps, divergence_family=cfg.divergence_family,
                             irrep_dims=model.group.irrep_dims)
    beta = attention_weights(energy, tau=attention_tau(_as_coeff(cfg.kappa_beta, out.mu.device), model.group.irrep_dims), log_prior=log_prior)
    spec = out.sigma if out.sigma.dim() == out.mu.dim() else condition_number(out.sigma)
    cond = float(condition_number(out.sigma).max()) if out.sigma.dim() > out.mu.dim() else \
        float((spec.amax(dim=-1) / spec.amin(dim=-1).clamp(min=1e-12)).max())
    health = {
        "nan_mu":     nan_inf_fraction(out.mu),
        "nan_sigma":  nan_inf_fraction(out.sigma),
        "nan_phi":    nan_inf_fraction(out.phi),
        "nan_energy": nan_inf_fraction(energy),
        "nan_beta":   nan_inf_fraction(beta),
        "max_condition": cond,
    }
    # regime_ii edge-factor saturation readout (audit 2026-06-10 F3 monitoring): the PRE-cap
    # per-edge ||delta_ij||_2 against the builder's smooth cap (12). A max far above the cap
    # means the trained connection lives in the saturated regime (the operator is norm-limited);
    # off the hot path, eval-interval cost only.
    if cfg.transport_mode == "regime_ii" and getattr(model, "connection_W", None) is not None:
        delta = cfg.cocycle_relaxation * torch.einsum(
            "ik,akl,jl->ija", out.mu, model.connection_W, out.mu)
        health["regime_ii_delta_max_norm"] = float(delta.norm(dim=-1).max())
    return health


@torch.no_grad()
def converged_state(
    model,
    token_ids: torch.Tensor,           # (B, N) token ids; only sequence 0 is used
) -> Dict[str, torch.Tensor]:
    r"""The converged-belief diagnostic state of one sequence, as tensors for the figures.

    Mirrors :meth:`VFEModel.diagnostics` EXACTLY (same active config: transport mode,
    connection_W, rope, family, divergence, alpha) but returns the underlying tensors the scalar
    diagnostics discard. The gauge-equivariance certificate, per-head gauge invariants, belief
    spectrum / SPD ellipses, and the guard-saturation / causal panels all read from these. Returns
    the converged ``mu`` (N, K), ``sigma`` (N, K) or (N, K, K), ``phi`` (N, n_gen), the per-token
    vertex factor ``exp_phi`` (N, K, K) = exp(embed(phi_i)), the pre-rope pairwise transport
    ``omega`` (N, N, K, K) (the phi-cocycle the equivariance/holonomy metrics co-transform),
    the pairwise ``energy`` and attention ``beta`` ((N, N) or (H, N, N)), and the per-token
    self-divergence ``self_div`` (N,) or (N, K).
    """
    from vfe3.model.stack import vfe_stack
    from vfe3.geometry.transport import RopeTransport, compute_transport_operators

    cfg = model.cfg
    was_training = model.training
    model.eval()
    try:
        belief, log_prior, rope = _encode_one(model, token_ids)
        cap: dict = {}                                       # q* capture (F self-term, as diagnostics)
        out = vfe_stack(
            belief, belief.mu, belief.sigma, model.group, cfg,
            log_prior=log_prior, block_norm=model.block_norm,
            head_mixer=model.head_mixer, cg_coupling=model.cg_coupling,   # replay the trained model
            log_alpha=getattr(model, "log_alpha", None), lambda_beta=_lambda_beta(model),
            connection_W=getattr(model, "connection_W", None),
            connection_M=getattr(model, "connection_M", None),
            rope=rope, rope_on_cov=cfg.rope_full_gauge, rope_on_value=cfg.rope_on_value,
            capture=cap,
        )
        rho, rho_s = cfg.prior_handoff_rho, cfg.prior_handoff_sigma     # rebuild last-block prior
        mu_p, sigma_p = belief.mu, belief.sigma                         # exact iff L==1 or rho==0
        for _ in range(cfg.n_layers - 1):
            mu_p = (1.0 - rho) * mu_p + rho * out.mu
            sigma_p = (1.0 - rho_s) * sigma_p + rho_s * out.sigma
        omega = _transport(                                            # (N, N, K, K) phi-cocycle (pre-rope)
            out.phi, model.group, transport_mode=cfg.transport_mode,
            mu=(out.mu if cfg.transport_mode in ("regime_ii", "regime_ii_covariant") else None),
            sigma=(out.sigma if cfg.transport_mode == "regime_ii_covariant" else None),
            connection_W=getattr(model, "connection_W", None),
            connection_M=getattr(model, "connection_M", None),
            cocycle_relaxation=cfg.cocycle_relaxation,
        )
        if rope is not None:
            rope_omega = RopeTransport(base=omega, rope=rope, on_cov=cfg.rope_full_gauge,
                                       on_value=cfg.rope_on_value)
            mu_t    = transport_mean(rope_omega, out.mu)
            sigma_t = transport_covariance(rope_omega, out.sigma)
        else:
            mu_t    = transport_mean(omega.unsqueeze(0), out.mu.unsqueeze(0))[0]
            sigma_t = transport_covariance(omega.unsqueeze(0), out.sigma.unsqueeze(0))[0]
        fam = get_family(cfg.family)
        energy = pairwise_energy(
            fam(out.mu, out.sigma), fam(mu_t, sigma_t), alpha=cfg.renyi_order,
            kl_max=cfg.kl_max, eps=cfg.eps, divergence_family=cfg.divergence_family,
            irrep_dims=model.group.irrep_dims,
        )
        beta = attention_weights(energy, tau=attention_tau(_as_coeff(cfg.kappa_beta, out.mu.device), model.group.irrep_dims), log_prior=log_prior)
        _q_conv = cap["converged"]                           # q*: the F self-term reads the pre-
        self_div = self_divergence_for_alpha(                # transform converged belief (F19,
            fam(_q_conv.mu, _q_conv.sigma), fam(mu_p, sigma_p), alpha=cfg.renyi_order,   # as diagnostics)
            kl_max=cfg.kl_max, eps=cfg.eps, divergence_family=cfg.divergence_family,
            lambda_alpha_mode=cfg.lambda_alpha_mode,
        )
        exp_phi = compute_transport_operators(out.phi.unsqueeze(0), model.group)["exp_phi"][0]
    finally:
        if was_training:
            model.train()
    return {
        "mu":       out.mu,
        "sigma":    out.sigma,
        "phi":      out.phi,
        "exp_phi":  exp_phi,
        "omega":    omega,
        "energy":   energy,
        "beta":     beta,
        "self_div": self_div,
    }


def attention_entropy_cov_gap(
    model,
    token_ids: torch.Tensor,           # (B, N) token ids; only sequence 0 is used
) -> Dict[str, torch.Tensor]:
    r"""C1/EXP-4: the attention-entropy gradient gap -tau^{-1} Cov_beta(E, dE) on the converged belief.

    The production closed-form kernel CANNOT compute the attention-entropy term, so it descends the
    SURROGATE belief gradient; only the autograd ORACLE with ``include_attention_entropy=True``
    realizes the canonical gradient. Differencing the oracle gradient with the entropy term ON vs OFF
    on the SAME converged-belief snapshot isolates exactly the entropy term's contribution, which by
    the envelope identity equals ``lambda_beta * (-tau^{-1} Cov_beta(E, dE))`` (= the bare identity on
    the pure ``lambda_beta=1`` path; the scalar identity is proven in
    ``tests/test_free_energy.py::test_gradient_gap_canonical_minus_surrogate_is_neg_cov_over_tau``).
    Returns the per-token gap magnitude ``cov_gap_per_token`` (N,) = ||g_canon - g_surrogate|| over
    the (mu, sigma) blocks, and the scalar mean ``cov_gap``. The attention prior is folded with the
    precision bias (``_fold_precision_bias``) exactly as forward does, so beta is the one the trained
    model uses (load-bearing under ``precision_weighted_attention=True``, the baseline). Built for the
    single-block (L=1) operating point and the default flat-transport path (regime_ii's mu-dependent
    Omega would need an omega_builder; neither is the EXP-4 operating point).
    """
    from vfe3.gradients.oracle import belief_gradients_autograd
    from vfe3.geometry.transport import RopeTransport
    cfg = model.cfg
    dev = _model_device(model)
    with torch.no_grad():                                          # snapshot build is grad-free
        belief, log_prior, rope = _encode_one(model, token_ids)
        log_prior = model._fold_precision_bias(log_prior, belief.sigma)   # match forward's beta prior
        ikw = _iter_kwargs(model, log_prior, rope)
        mu_p, sigma_p = belief.mu, belief.sigma
        out = belief
        for _ in range(cfg.n_e_steps):                            # converge to q* (the operating point)
            out = e_step_iteration(out, mu_p, sigma_p, model.group, **ikw)
        omega = _transport(
            out.phi, model.group, transport_mode=cfg.transport_mode,
            mu=(out.mu if cfg.transport_mode in ("regime_ii", "regime_ii_covariant") else None),
            sigma=(out.sigma if cfg.transport_mode == "regime_ii_covariant" else None),
            connection_W=getattr(model, "connection_W", None),
            connection_M=getattr(model, "connection_M", None),
            cocycle_relaxation=cfg.cocycle_relaxation,
        )
        if rope is not None:
            omega = RopeTransport(base=omega, rope=rope, on_cov=cfg.rope_full_gauge,
                                  on_value=cfg.rope_on_value)
    kw = dict(                                                    # the oracle's free-energy knob bag
        tau=attention_tau(_as_coeff(cfg.kappa_beta, dev), model.group.irrep_dims),
        renyi_order=cfg.renyi_order, kl_max=cfg.kl_max, eps=cfg.eps,
        b0=_as_coeff(cfg.b0, dev), c0=_as_coeff(cfg.c0, dev), value=cfg.lambda_alpha,
        lambda_beta=_lambda_beta(model), gradient_mode=cfg.gradient_mode, family=cfg.family,
        divergence_family=cfg.divergence_family, lambda_alpha_mode=cfg.lambda_alpha_mode,
        irrep_dims=model.group.irrep_dims, log_prior=log_prior,
        log_alpha=getattr(model, "log_alpha", None),
    )                                                             # grad ON here (outside the no_grad block)
    gc_mu, gc_sig = belief_gradients_autograd(out.mu, out.sigma, mu_p, sigma_p, omega,
                                              include_attention_entropy=True, **kw)
    gs_mu, gs_sig = belief_gradients_autograd(out.mu, out.sigma, mu_p, sigma_p, omega,
                                              include_attention_entropy=False, **kw)
    n = out.mu.shape[0]
    per_tok = torch.sqrt((gc_mu - gs_mu).pow(2).reshape(n, -1).sum(dim=-1)
                         + (gc_sig - gs_sig).pow(2).reshape(n, -1).sum(dim=-1))   # (N,)
    return {"cov_gap_per_token": per_tok, "cov_gap": per_tok.mean()}


@torch.no_grad()
def s_channel_refinement(model, token_ids: torch.Tensor) -> Optional[dict]:
    r"""Model-channel (s) refinement diagnostics for the ``s_e_step=True`` path (sequence 0).

    Returns ``None`` when ``cfg.s_e_step`` is False (the s-channel does not run, so the figure is
    skipped). Otherwise replays the SAME two steps ``forward``/``diagnostics`` take -- the static
    encode ``s0 = encode_s(tokens)`` and the refined ``s1 = _refine_s(tokens, pos_phi(phi))`` under
    the frozen gauge frame -- and measures, per token position, how far the model channel moved and
    how it tracks the frozen hyper-prior centroid ``r = (r_mu, exp(r_sigma_log))``:

      mu_delta[i]       = ||s1_mu[i]   - s0_mu[i]||_2                         (mean refinement)
      logsigma_delta[i] = ||log s1_sigma[i] - log s0_sigma[i]||_2            (variance refinement)
      kl_s0_r[i], kl_s1_r[i] = KL(s . || r) before / after refinement        (consensus toward r)

    The s-channel descends ``lambda_h * KL(s||r) + lambda_gamma * model-consensus``, so a healthy
    refinement pulls ``KL(s1||r) < KL(s0||r)``; the deltas show where on the sequence it acts.
    """
    if not model.cfg.s_e_step:
        return None
    from vfe3.families.gaussian import DiagonalGaussian
    from vfe3.divergence import get_functional
    cfg, pb = model.cfg, model.prior_bank
    enc  = pb.encode(token_ids[:1])                                # (1, N, ...)
    phi0 = model._apply_pos_phi(enc.phi[0]).unsqueeze(0)           # (1, N, n_gen) frozen gauge frame
    s0_mu, s0_sigma = (t[0] for t in pb.encode_s(token_ids[:1]))   # (N, K) static model channel
    s1_mu, s1_sigma = (t[0] for t in model._refine_s(token_ids[:1], phi0))  # (N, K) refined
    r_mu    = pb.r_mu.expand_as(s1_mu)                             # (N, K) frozen hyper-prior centroid
    r_sigma = torch.exp(pb.r_sigma_log).clamp(min=cfg.eps).expand_as(s1_sigma)
    kl = get_functional("renyi")                                  # KL = renyi at alpha=1
    r  = DiagonalGaussian(r_mu, r_sigma)
    kl_s0_r = kl(DiagonalGaussian(s0_mu, s0_sigma), r, alpha=1.0, kl_max=cfg.kl_max, eps=cfg.eps)  # (N,)
    kl_s1_r = kl(DiagonalGaussian(s1_mu, s1_sigma), r, alpha=1.0, kl_max=cfg.kl_max, eps=cfg.eps)  # (N,)
    return {
        "mu_delta":       (s1_mu - s0_mu).norm(dim=-1).cpu(),
        "logsigma_delta": (s1_sigma.clamp(min=cfg.eps).log()
                           - s0_sigma.clamp(min=cfg.eps).log()).norm(dim=-1).cpu(),
        "kl_s0_r":        kl_s0_r.cpu(),
        "kl_s1_r":        kl_s1_r.cpu(),
    }


@torch.no_grad()
def model_channel_belief(model, token_ids: torch.Tensor) -> Optional[dict]:
    r"""Model-channel beliefs s_i = N(s_mu, s_sigma) for sequence 0 (the ``s`` figure).

    Returns ``None`` when the model channel is inactive (no s tables). Otherwise looks up the static
    s tables and returns the per-coordinate population statistics over tokens (``mu_mean``, ``mu_std``,
    ``sigma_mean``), the per-token variance spectrum sorted descending (``spectrum`` (N, K) -- the s
    tables are diagonal so the variances ARE the spectrum), and the per-token effective rank
    (``eff_rank`` (N,)). Available on the broader model-channel-active path (lambda_h>0 OR
    lambda_gamma>0 OR prior_source=='model_channel' OR s_e_step), so it covers prior_source=='model_channel'
    where the s_e_step refinement figure does not run."""
    if not model._model_channel_active:
        return None
    cfg, pb = model.cfg, model.prior_bank
    s_mu, s_sigma = (t[0] for t in pb.encode_s(token_ids[:1]))     # (N, K)
    lam = torch.sort(s_sigma.clamp(min=cfg.eps), dim=-1, descending=True).values   # (N, K)
    return {
        "mu_mean":    s_mu.mean(dim=0).cpu(),                      # (K,)
        "mu_std":     s_mu.std(dim=0).cpu(),                       # (K,)
        "sigma_mean": s_sigma.mean(dim=0).cpu(),                   # (K,)
        "spectrum":   lam.cpu(),                                   # (N, K)
        "eff_rank":   metrics.effective_rank(s_sigma).cpu(),       # (N,)
    }


@torch.no_grad()
def hyper_prior_centroid(model, token_ids: torch.Tensor) -> Optional[dict]:
    r"""The hyper-prior centroid r and how the model channel s clusters around it (the ``r`` figure).

    Returns ``None`` when r does not exist (it is created only when lambda_h>0 OR s_e_step). Otherwise
    returns the centroid per coordinate (``r_mu`` (K,), ``r_sigma`` (K,) = exp(r_sigma_log)) and the
    model-channel population per coordinate over sequence 0 (``s_mu_mean``, ``s_mu_std``, ``s_sigma_mean``),
    so the figure can show the consensus r against the s distribution it anchors."""
    cfg, pb = model.cfg, model.prior_bank
    if getattr(pb, "r_mu", None) is None:
        return None
    s_mu, s_sigma = (t[0] for t in pb.encode_s(token_ids[:1]))     # (N, K)
    return {
        "r_mu":         pb.r_mu.detach().cpu(),                                      # (K,)
        "r_sigma":      torch.exp(pb.r_sigma_log).clamp(min=cfg.eps).detach().cpu(), # (K,)
        "s_mu_mean":    s_mu.mean(dim=0).cpu(),                                      # (K,)
        "s_mu_std":     s_mu.std(dim=0).cpu(),                                       # (K,)
        "s_sigma_mean": s_sigma.mean(dim=0).cpu(),                                   # (K,)
    }


@torch.no_grad()
def hyper_prior_coupling(model, token_ids: torch.Tensor) -> Optional[dict]:
    r"""Per-token hyper-prior divergence KL(s_i||r) for sequence 0 (the ``h`` figure: the lambda_h block).

    Returns ``None`` when r does not exist. Uses ``model._hyper_prior_kl`` so the plotted per-position
    KL(s_i||r) is the SAME quantity the diagnostics decomposition and the forward loss carry; ``lambda_h``
    is returned for the title (the weight the block enters F with)."""
    pb = model.prior_bank
    if getattr(pb, "r_mu", None) is None:
        return None
    s_belief = model._refined_s_belief(token_ids)                  # s1 under s_e_step (M2), else None (raw s tables)
    kl = model._hyper_prior_kl(token_ids[:1], s_belief=s_belief)[0]   # (N,)
    return {"kl_s_r": kl.cpu(), "lambda_h": float(model.cfg.lambda_h)}


@torch.no_grad()
def gamma_attention(model, token_ids: torch.Tensor) -> Optional[dict]:
    r"""Model-coupling attention gamma_ij for sequence 0 (the gamma figure), via
    :meth:`VFEModel.gamma_attention_maps`.

    Returns ``None`` when the model channel is inactive. Otherwise returns the per-head gamma weights
    ``gamma`` (H, N, N) = softmax_j(log pi^s - E^s/tau_g) on the model-channel beliefs s under the tied
    flat transport from the converged belief gauge frame -- the s-channel analogue of the belief beta maps."""
    g = model.gamma_attention_maps(token_ids[:1])
    if g is None:
        return None
    return {"gamma": g.cpu()}


@torch.no_grad()
def model_channel_bank(
    model,
    token_batches:  Iterable[torch.Tensor],   # iterable of (B, N) token-id batches

    *,
    device:         Optional[torch.device] = None,
    max_sequences:  Optional[int]          = None,
) -> Optional[Dict[str, torch.Tensor]]:
    r"""Collect the model-channel beliefs s_i over many sequences -- the bank for the model-channel UMAP.

    ``None`` when the model channel is inactive (no s tables). The s-channel sibling of :func:`belief_bank`:
    for each batch it looks up the static model-channel belief ``s_i = N(s_mu, s_sigma)`` via ``encode_s``,
    and -- when ``s_e_step`` -- refines it through the s E-step (so s is position-dependent, exactly as the
    belief bank's q is), then stacks the per-token ``mu = s_mu`` (M, K) and ``sigma = s_sigma`` (M, K) with
    ``token_ids`` (M,) and ``seq_idx`` (M,). There is NO ``phi`` channel: the model channel shares the belief
    gauge frame, so an s-channel phi UMAP would duplicate the belief one. Feeds ``plot_belief_umap`` directly
    (channels mu / sigma), so the redesigned cluster-and-distinctive-token view applies to the slow channel.
    NB i and j are token POSITIONS, not separate channels: this single bank embeds every s_i (all positions);
    the i<->j pairing lives only in the gamma_ij model-coupling attention (see :func:`gamma_attention`).
    """
    if not model._model_channel_active:
        return None
    device = device or _model_device(model)
    was_training = model.training
    model.eval()
    mus, sigmas, tids, sidx = [], [], [], []
    seq_counter = 0
    try:
        for tokens in token_batches:
            tokens = tokens.to(device)
            s_mu, s_sigma = model.prior_bank.encode_s(tokens)         # (B, N, K) static model channel
            if model.cfg.s_e_step:                                    # refine -> position-dependent, like q
                phi0 = model._apply_pos_phi(model.prior_bank.encode(tokens).phi)
                s_mu, s_sigma = model._refine_s(tokens, phi0)
            b, n = tokens.shape
            mus.append(s_mu.reshape(b * n, -1))
            sigmas.append(s_sigma.reshape(b * n, -1))
            tids.append(tokens.reshape(b * n))
            sidx.append(torch.arange(seq_counter, seq_counter + b, device=device).repeat_interleave(n))
            seq_counter += b
            if max_sequences is not None and seq_counter >= max_sequences:
                break
    finally:
        if was_training:
            model.train()
    return {
        "mu":        torch.cat(mus),
        "sigma":     torch.cat(sigmas),
        "token_ids": torch.cat(tids),
        "seq_idx":   torch.cat(sidx),
    }


def _vocab_display_panel(
    probs:   torch.Tensor,               # (P0, V) per-position softmax for ONE sequence
    tokens:  torch.Tensor,               # (N,)    that sequence's input token ids

    *,
    max_rows:      int = 40,
    per_pos_k:     int = 3,
    max_positions: int = 64,
) -> Dict[str, object]:
    r"""Select a compact ``(R, P)`` probability sub-grid for the Seq x top-k heatmap.

    The full ``p(o_{n+1} | o_{<=n})`` over ``V = 50257`` tokens is illegible, so the y-axis is the
    union of each shown position's top-``per_pos_k`` predicted token ids together with the true
    next-token id, ranked by summed probability over the shown positions and capped at
    ``max_rows``; the x-axis is the first ``max_positions`` positions. ``disp_truth_row`` gives,
    per position, the row index of the true next token (-1 when it falls outside the shown rows),
    for the ground-truth overlay.
    """
    P = min(int(probs.shape[0]), max_positions)
    p = probs[:P]                                                # (P, V)
    context_ids = tokens[:P]                                     # (P,) token AT each shown position
    target_ids  = tokens[1:P + 1]                               # (P,) the true next token
    topk = p.topk(min(per_pos_k, p.shape[-1]), dim=-1).indices.reshape(-1)   # (P*k,)
    cand = torch.cat([topk, target_ids]).unique()               # (C,) candidate row ids
    mass = p[:, cand].sum(dim=0)                                # (C,) total mass per candidate
    order = mass.argsort(descending=True)[:max_rows]
    row_ids = cand[order]                                        # (R,)
    disp_probs = p[:, row_ids].transpose(0, 1).contiguous()     # (R, P)
    pos_in = {int(t): i for i, t in enumerate(row_ids.tolist())}
    truth_row = torch.tensor([pos_in.get(int(t), -1) for t in target_ids.tolist()], dtype=torch.long)
    return {
        "disp_context_ids": context_ids.cpu(),
        "disp_target_ids":  target_ids.cpu(),
        "row_ids":          row_ids.cpu(),
        "disp_probs":       disp_probs.cpu(),
        "disp_truth_row":   truth_row,
    }


@torch.no_grad()
def vocab_prediction_stats(
    model,
    token_batches:  Iterable[torch.Tensor],   # iterable of (B, N) token-id batches

    *,
    device:         Optional[torch.device] = None,
    max_rows:       int = 40,
    per_pos_k:      int = 3,
    max_positions:  int = 64,
    max_pairs:      int = 200_000,
) -> Dict[str, object]:
    r"""Next-token predictive distributions over the vocabulary, for the vocab-probability figures.

    Runs the inference forward (``model(tokens)`` -> ``(B, N, V)`` logits) and softmaxes to
    ``p(o_{n+1} | o_{<=n})``. Logit row ``n`` predicts token ``n+1``, so the aggregate uses
    positions ``0..N-2`` with targets ``tokens[:, 1:]``; the final position's target lies outside
    the window and is dropped.

    Returns one dict feeding three figures:
      display (first sequence) -- ``disp_context_ids`` (P,), ``disp_target_ids`` (P,),
        ``row_ids`` (R,), ``disp_probs`` (R, P), ``disp_truth_row`` (P,);
      calibration (all positions) -- ``mean_pred_prob`` (V,), ``unigram`` (V,),
        ``mean_pred_entropy`` (), ``unigram_entropy`` (), ``n_positions`` ();
      confusion (capped sample of <= ``max_pairs``) -- ``true_ids`` (M,), ``pred_ids`` (M,) argmax.
    Entropies are in nats: ``mean_pred_entropy`` is the mean conditional ``H(p(.|context))`` and
    ``unigram_entropy`` is ``H`` of the empirical marginal; their gap is the context information
    the model uses, which collapses toward zero when predictions degenerate to the marginal.
    """
    device = device or _model_device(model)
    was_training = model.training
    model.eval()
    V = int(model.cfg.vocab_size)
    prob_sum = torch.zeros(V, dtype=torch.float64, device=device)
    tgt_count = torch.zeros(V, dtype=torch.float64, device=device)
    ent_sum = torch.zeros((), dtype=torch.float64, device=device)
    n_pos = 0
    true_ids: List[torch.Tensor] = []
    pred_ids: List[torch.Tensor] = []
    n_pairs = 0
    disp: Optional[Dict[str, object]] = None
    try:
        for batch in token_batches:
            tokens = batch[0] if isinstance(batch, (tuple, list)) else batch
            tokens = tokens.to(device)
            if tokens.dim() == 1:
                tokens = tokens.unsqueeze(0)
            b, n = tokens.shape
            if n < 2:
                continue
            logits = model(tokens)[:, :-1].float()                  # (B, N-1, V) drop last (no in-window target)
            targets = tokens[:, 1:]                                 # (B, N-1) true next token
            probs = torch.softmax(logits, dim=-1)                   # (B, N-1, V)
            flatp = probs.reshape(-1, V)                            # (B*(N-1), V)
            prob_sum += flatp.sum(dim=0).double()
            ent_sum += -(flatp * flatp.clamp_min(1e-12).log()).sum(dim=-1).sum().double()
            tgt_flat = targets.reshape(-1)
            tgt_count += torch.bincount(tgt_flat, minlength=V).double()
            n_pos += int(tgt_flat.numel())
            if n_pairs < max_pairs:
                take = min(max_pairs - n_pairs, int(tgt_flat.numel()))
                true_ids.append(tgt_flat[:take].cpu())
                pred_ids.append(flatp[:take].argmax(dim=-1).cpu())
                n_pairs += take
            if disp is None:
                disp = _vocab_display_panel(probs[0], tokens[0], max_rows=max_rows,
                                            per_pos_k=per_pos_k, max_positions=max_positions)
    finally:
        if was_training:
            model.train()
    if disp is None:
        raise ValueError("vocab_prediction_stats: no usable (N>=2) batch in token_batches")
    denom = max(n_pos, 1)
    mean_pred_prob = prob_sum / denom
    unigram = tgt_count / denom
    uni_ent = float(-(unigram * unigram.clamp_min(1e-12).log()).sum())
    return {
        **disp,
        "mean_pred_prob":    mean_pred_prob.float().cpu(),
        "unigram":           unigram.float().cpu(),
        "mean_pred_entropy": float(ent_sum / denom),
        "unigram_entropy":   uni_ent,
        "n_positions":       int(n_pos),
        "true_ids":          torch.cat(true_ids) if true_ids else torch.empty(0, dtype=torch.long),
        "pred_ids":          torch.cat(pred_ids) if pred_ids else torch.empty(0, dtype=torch.long),
    }


@torch.no_grad()
def decode_readout(
    model,

    *,
    max_rows: int = 96,
) -> Optional[Dict[str, object]]:
    r"""The linear-decode readout matrix ``W`` (``logits = mu_q @ W^T``) for the Decode-W heatmap.

    Returns ``None`` on the ``use_prior_bank=True`` KL-to-prior decode (no ``W`` exists there).
    Otherwise selects the ``max_rows`` highest-L2-norm vocabulary rows -- the output directions the
    readout most strongly distinguishes -- and returns ``weight`` (R, K), their ``row_ids`` (R,),
    the per-row ``row_norm`` (R,), and the optional log-unigram ``bias`` (R,).
    """
    pb = model.prior_bank
    W = getattr(pb, "output_proj_weight", None)
    if W is None:
        return None
    W = W.detach().float()
    norms = W.norm(dim=1)                                        # (V,)
    row_ids = norms.argsort(descending=True)[:max_rows]
    bias = getattr(pb, "output_proj_bias", None)
    return {
        "weight":   W[row_ids].cpu(),
        "row_ids":  row_ids.cpu(),
        "row_norm": norms[row_ids].cpu(),
        "bias":     (bias.detach().float()[row_ids].cpu() if bias is not None else None),
    }
