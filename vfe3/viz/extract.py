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
        include_attention_entropy=cfg.include_attention_entropy, gradient_mode=cfg.gradient_mode,
        family=cfg.family, divergence_family=cfg.divergence_family, lambda_alpha_mode=cfg.lambda_alpha_mode,
        phi_precond_mode=cfg.phi_precond_mode, phi_retract_mode=cfg.phi_retract_mode,
        spd_retract_mode=cfg.spd_retract_mode, transport_mode=cfg.transport_mode,
        cocycle_relaxation=cfg.cocycle_relaxation, connection_W=getattr(model, "connection_W", None),
        log_prior=log_prior, log_alpha=getattr(model, "log_alpha", None),
        rope=rope, rope_on_cov=cfg.rope_full_gauge,
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
        rope=rope, rope_on_cov=cfg.rope_full_gauge,
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
                rope=rope, rope_on_cov=cfg.rope_full_gauge,
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
            rope=rope, rope_on_cov=cfg.rope_full_gauge,
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
    return {"mu": mu_stack, "sigma": sig_stack, "d_ai": d_ai, "effective_rank": eff}


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
        mu=(out.mu if cfg.transport_mode == "regime_ii" else None),
        connection_W=getattr(model, "connection_W", None),
        cocycle_relaxation=cfg.cocycle_relaxation,
    )
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
            rope=rope, rope_on_cov=cfg.rope_full_gauge,
            capture=cap,
        )
        rho, rho_s = cfg.prior_handoff_rho, cfg.prior_handoff_sigma     # rebuild last-block prior
        mu_p, sigma_p = belief.mu, belief.sigma                         # exact iff L==1 or rho==0
        for _ in range(cfg.n_layers - 1):
            mu_p = (1.0 - rho) * mu_p + rho * out.mu
            sigma_p = (1.0 - rho_s) * sigma_p + rho_s * out.sigma
        omega = _transport(                                            # (N, N, K, K) phi-cocycle (pre-rope)
            out.phi, model.group, transport_mode=cfg.transport_mode,
            mu=(out.mu if cfg.transport_mode == "regime_ii" else None),
            connection_W=getattr(model, "connection_W", None),
            cocycle_relaxation=cfg.cocycle_relaxation,
        )
        if rope is not None:
            rope_omega = RopeTransport(base=omega, rope=rope, on_cov=cfg.rope_full_gauge)
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
    kl = model._hyper_prior_kl(token_ids[:1])[0]                   # (N,)
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
