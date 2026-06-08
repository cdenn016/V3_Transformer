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
    r"""Encode sequence 0 to the initial belief (pos_phi applied), with its log-prior and RoPE."""
    enc = model.prior_bank.encode(token_ids[:1])                  # (1, N, ...)
    belief = BeliefState(mu=enc.mu[0], sigma=enc.sigma[0], phi=model._apply_pos_phi(enc.phi[0]))
    n = belief.mu.shape[0]
    log_prior = model._attention_log_prior(n, token_ids.device)
    rope = model._rope_rotation(n, token_ids.device)
    return belief, log_prior, rope


def _iter_kwargs(model, log_prior: torch.Tensor, rope: Optional[torch.Tensor]) -> dict:
    r"""The full ``e_step_iteration`` knob bag assembled from the model config (mirrors vfe_block)."""
    cfg = model.cfg
    return dict(
        tau=attention_tau(cfg.kappa, model.group.irrep_dims),
        e_mu_lr=cfg.e_mu_lr, e_sigma_lr=cfg.e_sigma_lr, e_phi_lr=cfg.e_phi_lr,
        alpha_div=cfg.alpha_div, value=cfg.alpha,
        b0=_as_coeff(cfg.b0, model.prior_bank.mu_embed.device),
        c0=_as_coeff(cfg.c0, model.prior_bank.mu_embed.device),
        lambda_beta=_lambda_beta(model), kl_max=cfg.kl_max, eps=cfg.eps,
        sigma_max=cfg.sigma_max, e_sigma_q_trust=cfg.e_sigma_q_trust, mass_phi=cfg.mass_phi,
        e_mu_q_trust=cfg.e_mu_q_trust, mu_trust_mode=cfg.mu_trust_mode,
        include_attention_entropy=cfg.include_attention_entropy, gradient_mode=cfg.gradient_mode,
        family=cfg.family, divergence_family=cfg.divergence_family, alpha_mode=cfg.alpha_mode,
        phi_precond_mode=cfg.phi_precond_mode, phi_retract_mode=cfg.phi_retract_mode,
        spd_retract_mode=cfg.spd_retract_mode, transport_mode=cfg.transport_mode,
        cocycle_relaxation=cfg.cocycle_relaxation, connection_W=getattr(model, "connection_W", None),
        log_prior=log_prior, log_alpha=getattr(model, "log_alpha", None),
        rope=rope, rope_on_cov=cfg.rope_full_gauge,
    )


def _fe_kwargs(model, log_prior: torch.Tensor) -> dict:
    r"""The ``free_energy_value`` knob subset (it rejects the iteration-only step-size knobs)."""
    cfg = model.cfg
    return dict(
        tau=attention_tau(cfg.kappa, model.group.irrep_dims),
        alpha_div=cfg.alpha_div, value=cfg.alpha,
        b0=_as_coeff(cfg.b0, model.prior_bank.mu_embed.device),
        c0=_as_coeff(cfg.c0, model.prior_bank.mu_embed.device),
        lambda_beta=_lambda_beta(model), kl_max=cfg.kl_max, eps=cfg.eps,
        include_attention_entropy=cfg.include_attention_entropy, family=cfg.family,
        divergence_family=cfg.divergence_family, alpha_mode=cfg.alpha_mode,
        transport_mode=cfg.transport_mode, cocycle_relaxation=cfg.cocycle_relaxation,
        log_prior=log_prior, log_alpha=getattr(model, "log_alpha", None),
        connection_W=getattr(model, "connection_W", None),
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

    For each batch runs the model's belief pipeline (prior_bank.encode -> pos_phi -> vfe_stack,
    mirroring forward up to the converged belief, BEFORE head-mixer / final-norm decode prep) and
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
                log_alpha=getattr(model, "log_alpha", None), lambda_beta=_lambda_beta(model),
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
    fkw = _fe_kwargs(model, log_prior)

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
            block_norm=model.block_norm, log_alpha=getattr(model, "log_alpha", None),
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
    omega = _transport(out.phi, model.group)
    mu_t = transport_mean(omega.unsqueeze(0), out.mu.unsqueeze(0))[0]
    sigma_t = transport_covariance(omega.unsqueeze(0), out.sigma.unsqueeze(0))[0]
    fam = get_family(cfg.family)
    energy = pairwise_energy(fam(out.mu, out.sigma), fam(mu_t, sigma_t), alpha=cfg.alpha_div,
                             kl_max=cfg.kl_max, eps=cfg.eps, divergence_family=cfg.divergence_family,
                             irrep_dims=model.group.irrep_dims)
    beta = attention_weights(energy, tau=attention_tau(cfg.kappa, model.group.irrep_dims), log_prior=log_prior)
    spec = out.sigma if out.sigma.dim() == out.mu.dim() else condition_number(out.sigma)
    cond = float(condition_number(out.sigma).max()) if out.sigma.dim() > out.mu.dim() else \
        float((spec.amax(dim=-1) / spec.amin(dim=-1).clamp(min=1e-12)).max())
    return {
        "nan_mu":     nan_inf_fraction(out.mu),
        "nan_sigma":  nan_inf_fraction(out.sigma),
        "nan_phi":    nan_inf_fraction(out.phi),
        "nan_energy": nan_inf_fraction(energy),
        "nan_beta":   nan_inf_fraction(beta),
        "max_condition": cond,
    }


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
        out = vfe_stack(
            belief, belief.mu, belief.sigma, model.group, cfg,
            log_prior=log_prior, block_norm=model.block_norm,
            log_alpha=getattr(model, "log_alpha", None), lambda_beta=_lambda_beta(model),
            connection_W=getattr(model, "connection_W", None),
            rope=rope, rope_on_cov=cfg.rope_full_gauge,
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
            fam(out.mu, out.sigma), fam(mu_t, sigma_t), alpha=cfg.alpha_div,
            kl_max=cfg.kl_max, eps=cfg.eps, divergence_family=cfg.divergence_family,
            irrep_dims=model.group.irrep_dims,
        )
        beta = attention_weights(energy, tau=attention_tau(cfg.kappa, model.group.irrep_dims), log_prior=log_prior)
        self_div = self_divergence_for_alpha(
            fam(out.mu, out.sigma), fam(mu_p, sigma_p), alpha=cfg.alpha_div,
            kl_max=cfg.kl_max, eps=cfg.eps, divergence_family=cfg.divergence_family,
            alpha_mode=cfg.alpha_mode,
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
