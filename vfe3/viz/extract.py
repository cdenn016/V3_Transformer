r"""Belief / evaluation EXTRACTORS for VFE_3.0 publication figures.

These RUN the model -- reload-and-recompute converged beliefs, loop the E-step capturing the
belief trajectory, replay the block stack per layer, score a loader per unit, or tally numerical
health. They have side effects / drive the model, so they live here rather than in
``vfe3.metrics`` (whose contract is pure, side-effect-free measurement). Each returns plain
tensors / dicts that the pure metrics and the figure functions consume. Everything runs under
``torch.no_grad`` and OFF the training hot path.
"""

from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

from vfe3 import metrics
from vfe3.alpha_i import self_coupling_alpha
from vfe3.belief import BeliefState
from vfe3.families.base import get_family, kl
from vfe3.model.block import _as_coeff, e_step_shared_kwargs   # shared cfg->kwargs bag (audit 2026-07-12 N5)
from vfe3.free_energy import (
    attention_tau,
    attention_weights,
    pairwise_energy,
    self_divergence_for_alpha,
)
from vfe3.inference.e_step import (
    _transport,
    canonical_e_step_update,
    e_step_iteration,
    free_energy_value,
)
# Transport-mode state-routing sets (registry metadata, as model.py): the extractors below feed
# mu/sigma to _transport by membership here, never by matching literal mode names.
from vfe3.geometry.lie_ops import CompactBlockElement
from vfe3.geometry.transport import (CompactFactoredTransport, _TRANSPORT_NEEDS_MU,
                                     _TRANSPORT_NEEDS_SIGMA, transport_mean)
from vfe3.model.block import vfe_block
from vfe3.numerics import bounded_variance_from_log

if TYPE_CHECKING:
    from vfe3.model.model import DiagnosticSnapshot


def _model_device(model) -> torch.device:
    return model.prior_bank.mu_embed.device


def _validate_bank_caps(
    *,
    max_tokens:    Optional[int],
    max_sequences: Optional[int],
) -> None:
    """Reject ambiguous or empty belief-bank population requests."""
    if max_tokens is not None and max_sequences is not None:
        raise ValueError("max_tokens and max_sequences are mutually exclusive")
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if max_sequences is not None and max_sequences <= 0:
        raise ValueError("max_sequences must be positive")


def _slice_bank_to_cap(
    bank: Dict[str, torch.Tensor],

    *,
    max_tokens:    Optional[int],
    max_sequences: Optional[int],
) -> Dict[str, torch.Tensor]:
    """Slice every token-aligned bank field to the requested exact population."""
    if max_tokens is not None:
        limit = min(max_tokens, bank["token_ids"].shape[0])
    elif max_sequences is not None:
        limit = int((bank["seq_idx"] < max_sequences).sum().item())
    else:
        return bank
    return {key: value[:limit] for key, value in bank.items()}


def _snapshot_sequence(belief: BeliefState, index: int = 0) -> BeliefState:
    r"""Select one batch row while preserving every optional belief/frame channel."""
    return belief._replace(
        mu=belief.mu[index], sigma=belief.sigma[index], phi=belief.phi[index],
        s=belief.s[index] if belief.s is not None else None,
        r=belief.r[index] if belief.r is not None else None,
        omega=belief.omega[index] if belief.omega is not None else None,
        reflection=belief.reflection[index] if belief.reflection is not None else None,
        right_phi=(belief.right_phi[index]
                   if belief.right_phi is not None and belief.right_phi.dim() == belief.phi.dim()
                   else belief.right_phi),
    )


def _encode_one(model, token_ids: torch.Tensor) -> Tuple[BeliefState, torch.Tensor, Optional[torch.Tensor]]:
    r"""Encode sequence 0 to the initial belief (pos_phi applied), with its log-prior and RoPE.

    Under ``s_e_step`` the live model channel is replayed too (audit 2026-06-09 IE2): s is refined
    with the frozen gauge frame and the belief is anchored to it, exactly as ``forward`` /
    ``diagnostics`` do -- the callers' ``mu_p = belief.mu`` handoff then anchors to the refined s,
    so every extracted trajectory/figure describes the model that actually trained."""
    enc = model.prior_bank.encode(token_ids[:1])                  # (1, N, ...)
    belief = BeliefState(
        mu=enc.mu[0], sigma=enc.sigma[0], phi=model._apply_pos_phi(enc.phi[0]),
        omega=enc.omega[0] if enc.omega is not None else None,       # omega-direct GL(K) frame
        reflection=enc.reflection[0] if enc.reflection is not None else None,   # phi-path sign
        right_phi=model._pos_phi_right(enc.phi[0]),
    )
    model_phi = model._resolve_model_frame(token_ids[:1], belief.phi.unsqueeze(0))
    n = belief.mu.shape[0]
    rope = model._rope_rotation(n, token_ids.device)
    s_belief = None
    if model.cfg.s_e_step:
        s_mu1, s_sigma1 = model._refine_s(token_ids[:1], model_phi, rope=rope)
        s_belief = (s_mu1, s_sigma1)
        belief = belief._replace(mu=s_mu1[0], sigma=s_sigma1[0])
    log_prior = model._attention_log_prior(n, token_ids.device)
    log_prior = model._fold_precision_bias(log_prior, belief.sigma)   # no-op unless precision_weighted_attention;
    if model.cfg.gamma_as_beta_prior:                                # m4: match forward's hierarchical gamma prior fold
        tied_model_frame = model.cfg.s_frame_mode == "tied"
        log_prior = model._fold_gamma_prior(log_prior, token_ids[:1], model_phi,
                                            omega=(belief.omega.unsqueeze(0)
                                                   if tied_model_frame and belief.omega is not None else None),
                                            reflection=(belief.reflection.unsqueeze(0)
                                                        if tied_model_frame and belief.reflection is not None else None),
                                            s_belief=s_belief)[0]
    return belief, log_prior, rope                                    # forward's beliefs.sigma (model.py:762)


def _iter_kwargs(model, log_prior: torch.Tensor, rope: Optional[torch.Tensor]) -> dict:
    r"""The full ``e_step_iteration`` knob bag: the production shared cfg bag
    (``e_step_shared_kwargs``, audit 2026-07-12 N5 -- previously a hand-rolled copy that silently
    dropped ``e_step_update``/``mm_damping``/``lambda_twohop``/``skip_belief_sigma_update``) plus
    the runtime extras ``vfe_block``/``e_step`` bind per call (tau, step sizes, connections,
    transport toggles, log_prior, rope)."""
    cfg = model.cfg
    kw = e_step_shared_kwargs(cfg, _model_device(model))
    kw.update(
        tau=attention_tau(model.effective_kappa_beta(_model_device(model)), model.group.irrep_dims),
        e_q_mu_lr=cfg.e_q_mu_lr, e_q_sigma_lr=cfg.e_q_sigma_lr, e_phi_lr=cfg.e_phi_lr,
        lambda_beta=cfg.lambda_beta,
        gauge_parameterization=cfg.gauge_parameterization,
        connection_W=getattr(model, "connection_W", None),
        connection_M=getattr(model, "connection_M", None),
        connection_L=getattr(model, "connection_L", None),
        compact_phi_block_transport=model._compact_phi_blocks_enabled(),
        transport_mean_per_head=cfg.transport_mean_per_head,
        exp_fp64_mode=cfg.exp_fp64_mode,
        exp_fp64_norm_threshold=cfg.exp_fp64_norm_threshold,
        log_prior=log_prior,
        rope=rope, rope_on_cov=cfg.rope_full_gauge, rope_on_value=cfg.rope_on_value,
    )
    return kw


def _fe_kwargs(model, log_prior: torch.Tensor, rope: Optional[torch.Tensor] = None) -> dict:
    r"""The ``free_energy_value`` knob bag: the SAME shared cfg bag production's ``e_step``
    forwards to its diagnostic F (``free_energy_value`` declares the iteration-only knobs as
    explicit accept-and-ignore parameters and HONORS ``lambda_twohop``, audit 2026-07-12 N5), plus
    the runtime extras. ``rope`` is honored (audit PP6): the logged F carries the RoPE-wrapped
    transport. The step-size knobs (e_q_mu_lr/e_q_sigma_lr/e_phi_lr) stay off this bag --
    ``free_energy_value`` rejects them, exactly as production binds them on ``e_step`` only."""
    cfg = model.cfg
    kw = e_step_shared_kwargs(cfg, _model_device(model))
    kw.update(
        tau=attention_tau(model.effective_kappa_beta(_model_device(model)), model.group.irrep_dims),
        lambda_beta=cfg.lambda_beta,
        gauge_parameterization=cfg.gauge_parameterization,
        connection_W=getattr(model, "connection_W", None),
        connection_M=getattr(model, "connection_M", None),
        connection_L=getattr(model, "connection_L", None),
        log_prior=log_prior,
        rope=rope, rope_on_cov=cfg.rope_full_gauge, rope_on_value=cfg.rope_on_value,
    )
    return kw


@torch.no_grad()
def per_unit_eval_nats(
    model,
    loader:      Iterable[Tuple[torch.Tensor, torch.Tensor]],   # yields (tokens, targets) batches

    *,
    device:      Optional[torch.device] = None,
    max_batches: Optional[int]          = None,
) -> Dict[str, torch.Tensor]:
    r"""Per-SEQUENCE and per-TOKEN cross-entropy (nats) over a loader.

    The training ``evaluate`` retains only the aggregate token-weighted
    {ce, ppl, bits_per_token, bpc}; the
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
    reduces the cross-entropy with ``reduction='none'``. The s-refine + precision fold are applied
    here (and, since the C9 fix, in ``belief_bank`` too; no-ops on the pure default path) so the traced
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
            beliefs = beliefs._replace(
                phi=model._apply_pos_phi(beliefs.phi),
                right_phi=model._pos_phi_right(beliefs.phi),
            )                                                       # IS the decode's belief
            model_phi = model._resolve_model_frame(tokens, beliefs.phi)
            rope = model._rope_rotation(n, device)
            s_belief = None
            if cfg.s_e_step:                                       # anchor belief to the refined model channel
                s_mu1, s_sigma1 = model._refine_s(tokens, model_phi, rope=rope)
                s_belief = (s_mu1, s_sigma1)
                beliefs = beliefs._replace(mu=s_mu1, sigma=s_sigma1)
            log_prior = model._attention_log_prior(n, device)
            log_prior = model._fold_precision_bias(log_prior, beliefs.sigma)   # no-op unless precision_weighted_attention
            if model.cfg.gamma_as_beta_prior:                                # m4: match forward's hierarchical gamma prior fold
                tied_model_frame = model.cfg.s_frame_mode == "tied"
                log_prior = model._fold_gamma_prior(log_prior, tokens, model_phi,
                                                    omega=(beliefs.omega if tied_model_frame else None),
                                                    reflection=(beliefs.reflection if tied_model_frame else None),
                                                    s_belief=s_belief)
            out = vfe_stack(
                beliefs, beliefs.mu, beliefs.sigma, model.group, cfg,
                log_prior=log_prior, block_norm=model.block_norm,
                head_mixer=model.head_mixer, cg_coupling=model.cg_coupling,
                lambda_beta=cfg.lambda_beta,
                kappa_beta_override=model.effective_kappa_beta(device),   # learned tau, not init (audit M1)
                connection_W=getattr(model, "connection_W", None),
                connection_M=getattr(model, "connection_M", None),
                connection_L=getattr(model, "connection_L", None),
                rope=rope, rope_on_cov=cfg.rope_full_gauge, rope_on_value=cfg.rope_on_value,
                gauge_parameterization=cfg.gauge_parameterization,
            )
            trs = metrics.sigma_trace(
                out.sigma,
                diagonal=cfg.diagonal_covariance,
                family=cfg.family,
            ).reshape(-1)                                                    # (B*N,)
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
    max_tokens:     Optional[int]          = None,
    max_sequences:  Optional[int]          = None,
) -> Dict[str, torch.Tensor]:
    r"""Collect converged beliefs (mu, Sigma, phi) over many sequences into one bank.

    For each batch runs the model's belief pipeline (prior_bank.encode -> pos_phi -> vfe_stack
    with the SAME per-block head_mixer / cg_coupling / block_norm the training forward applies,
    mirroring forward up to the stack's handoff belief; only final-norm decode prep is omitted) and
    stacks the per-token converged ``mu`` (M, K), ``sigma`` (M, K) or (M, K, K), ``phi``
    (M, n_gen), with ``token_ids`` (M,), ``seq_idx`` (M,), and within-sequence ``pos_idx``
    (M,). ``max_tokens`` and ``max_sequences`` are mutually exclusive; either cap is applied
    exactly to every aligned field. Feeds the mu / Sigma / phi UMAP triptych and the at-scale
    clustering scores.
    """
    from vfe3.model.stack import vfe_stack
    _validate_bank_caps(max_tokens=max_tokens, max_sequences=max_sequences)
    device = device or _model_device(model)
    cfg = model.cfg
    was_training = model.training
    model.eval()
    mus, sigmas, phis, tids, sidx, pidx = [], [], [], [], [], []
    seq_counter = 0
    try:
        for tokens in token_batches:
            tokens = tokens.to(device)
            beliefs = model.prior_bank.encode(tokens)
            beliefs = beliefs._replace(
                phi=model._apply_pos_phi(beliefs.phi),
                right_phi=model._pos_phi_right(beliefs.phi),
            )
            model_phi = model._resolve_model_frame(tokens, beliefs.phi)
            n = tokens.shape[1]
            rope = model._rope_rotation(n, device)
            s_belief = None
            if cfg.s_e_step:                                       # anchor belief to the refined model channel
                s_mu1, s_sigma1 = model._refine_s(tokens, model_phi, rope=rope)
                s_belief = (s_mu1, s_sigma1)
                beliefs = beliefs._replace(mu=s_mu1, sigma=s_sigma1)
            log_prior = model._attention_log_prior(n, device)
            log_prior = model._fold_precision_bias(log_prior, beliefs.sigma)   # no-op unless precision_weighted_attention
            if model.cfg.gamma_as_beta_prior:                                # m4: match forward's hierarchical gamma prior fold
                tied_model_frame = model.cfg.s_frame_mode == "tied"
                log_prior = model._fold_gamma_prior(log_prior, tokens, model_phi,
                                                    omega=(beliefs.omega if tied_model_frame else None),
                                                    reflection=(beliefs.reflection if tied_model_frame else None),
                                                    s_belief=s_belief)
            out = vfe_stack(
                beliefs, beliefs.mu, beliefs.sigma, model.group, cfg,
                log_prior=log_prior, block_norm=model.block_norm,
                head_mixer=model.head_mixer, cg_coupling=model.cg_coupling,   # replay the trained
                lambda_beta=cfg.lambda_beta,  # model
                kappa_beta_override=model.effective_kappa_beta(device),   # learned tau, not init (audit M1)
                connection_W=getattr(model, "connection_W", None),
                connection_M=getattr(model, "connection_M", None),
                connection_L=getattr(model, "connection_L", None),
                rope=rope, rope_on_cov=cfg.rope_full_gauge, rope_on_value=cfg.rope_on_value,
                gauge_parameterization=cfg.gauge_parameterization,
            )
            b = tokens.shape[0]
            mus.append(out.mu.reshape(b * n, -1))
            sigmas.append(out.sigma.reshape(b * n, *out.sigma.shape[2:]))
            phis.append(out.phi.reshape(b * n, -1))
            tids.append(tokens.reshape(b * n))
            sidx.append(torch.arange(seq_counter, seq_counter + b, device=device).repeat_interleave(n))
            pidx.append(torch.arange(n, device=device).repeat(b))
            seq_counter += b
            token_count = sum(batch.shape[0] for batch in tids)
            if ((max_tokens is not None and token_count >= max_tokens)
                    or (max_sequences is not None and seq_counter >= max_sequences)):
                break
    finally:
        if was_training:
            model.train()
    bank = {
        "mu":        torch.cat(mus),
        "sigma":     torch.cat(sigmas),
        "phi":       torch.cat(phis),
        "token_ids": torch.cat(tids),
        "seq_idx":   torch.cat(sidx),
        "pos_idx":   torch.cat(pidx),
    }
    return _slice_bank_to_cap(
        bank,
        max_tokens=max_tokens,
        max_sequences=max_sequences,
    )


@torch.no_grad()
def e_step_belief_trace(
    model,
    token_ids: torch.Tensor,           # (B, N) token ids; only sequence 0 is used

    *,
    n_iter:    Optional[int] = None,
    snapshot: 'Optional[DiagnosticSnapshot]' = None,
) -> Dict[str, torch.Tensor]:
    r"""Loop the inner E-step capturing the FULL belief (and F) at every iteration for one sequence.

    The E-step ``return_trajectory`` yields the free-energy floats only; this captures the belief
    tuple (mu, sigma, phi) per inner iteration so the belief PATH and the SPD-metric residuals can
    be drawn. ``n_iter`` defaults to the trained ``cfg.n_e_steps`` (crank it up to show convergence
    past the trained budget). Returns ``mu`` (T+1, N, K), ``sigma`` (T+1, N, [K]), ``phi``
    (T+1, N, n_gen), and ``free_energy`` (T+1,) the global F at each iterate.
    """
    if snapshot is not None:
        snapshot = model._validate_diagnostic_snapshot(token_ids, snapshot)
        actual_n_iter = len(snapshot.trace_states) - 1
        if n_iter is not None and int(n_iter) != actual_n_iter:
            raise ValueError(
                f"snapshot contains {actual_n_iter} E-step iterations, requested n_iter={n_iter}")
        return {
            "mu": torch.stack([belief.mu[0] for belief in snapshot.trace_states]),
            "sigma": torch.stack([belief.sigma[0] for belief in snapshot.trace_states]),
            "phi": torch.stack([belief.phi[0] for belief in snapshot.trace_states]),
            "free_energy": snapshot.trace_free_energy,
        }
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
def e_step_fixed_point_diagnostics(
    model,
    token_ids: torch.Tensor,           # (B, N) token ids; only sequence 0 is used
) -> Dict[str, float]:
    r"""Measure configured movement and the distinct one-step-ahead residual.

    The executable map is replayed for the configured ``T`` iterations and once more. The
    ``estep_r_*_last`` fields retain their historical step-length definitions on
    ``q_{T-1} -> q_T``. The ``estep_fp_*`` fields measure ``q_T -> q_{T+1}``, which is the
    actual residual of the live state-dependent update map. For the frozen-surrogate updater,
    ``estep_target_gap`` measures the distance from the damped next iterate to the corresponding
    undamped frozen-surrogate minimizer.
    """
    cfg = model.cfg
    belief0, log_prior, rope = _encode_one(model, token_ids)
    mu_p, sigma_p = belief0.mu, belief0.sigma
    ikw = _iter_kwargs(model, log_prior, rope)
    states = [belief0]
    belief = belief0
    for _ in range(int(cfg.n_e_steps) + 1):
        belief = e_step_iteration(belief, mu_p, sigma_p, model.group, **ikw)
        states.append(belief)

    q_t = states[int(cfg.n_e_steps)]
    q_next = states[int(cfg.n_e_steps) + 1]
    out: Dict[str, float] = {}
    if cfg.n_e_steps > 0:
        configured = metrics.estep_residuals(
            torch.stack([state.mu for state in states[: cfg.n_e_steps + 1]]),
            torch.stack([state.sigma for state in states[: cfg.n_e_steps + 1]]),
            torch.stack([state.phi for state in states[: cfg.n_e_steps + 1]]),
            diagonal=cfg.diagonal_covariance,
        )
        for name, key in (
            ("r_mu", "estep_r_mu_last"),
            ("r_sigma", "estep_r_sigma_last"),
            ("r_phi", "estep_r_phi_last"),
        ):
            out[key] = float(configured[name][-1].mean())
    else:
        out.update({
            "estep_r_mu_last":    0.0,
            "estep_r_sigma_last": 0.0,
            "estep_r_phi_last":   0.0,
        })

    out["estep_fp_mu_rms"] = float((q_next.mu - q_t.mu).square().mean().sqrt())
    out["estep_fp_sigma_rms"] = float((q_next.sigma - q_t.sigma).square().mean().sqrt())
    out["estep_fp_phi_rms"] = float((q_next.phi - q_t.phi).square().mean().sqrt())

    family = get_family(cfg.family)
    out["estep_fp_kl"] = float(kl(
        family(q_next.mu, q_next.sigma),
        family(q_t.mu, q_t.sigma),
        kl_max=cfg.kl_max,
        eps=cfg.eps,
    ).mean())

    beta_t = model._attention_map_for_belief(q_t, log_prior, rope).float().clamp(min=cfg.eps)
    beta_next = model._attention_map_for_belief(q_next, log_prior, rope).float().clamp(min=cfg.eps)
    beta_mid = 0.5 * (beta_t + beta_next)
    beta_js = 0.5 * (
        (beta_t * (beta_t.log() - beta_mid.log())).sum(dim=-1)
        + (beta_next * (beta_next.log() - beta_mid.log())).sum(dim=-1)
    )
    out["estep_beta_js"] = float(beta_js.mean())

    prior = family(mu_p, sigma_p)

    def _alpha(state: BeliefState) -> torch.Tensor:
        divergence = self_divergence_for_alpha(
            family(state.mu, state.sigma),
            prior,
            alpha=cfg.renyi_order,
            kl_max=cfg.kl_max,
            eps=cfg.eps,
            divergence_family=cfg.divergence_family,
            lambda_alpha_mode=cfg.lambda_alpha_mode,
        )
        return self_coupling_alpha(
            divergence,
            mode=cfg.lambda_alpha_mode,
            value=cfg.lambda_alpha,
            b0=_as_coeff(cfg.b0, state.mu.device),
            c0=_as_coeff(cfg.c0, state.mu.device),
        )[0]

    out["estep_alpha_rms_delta"] = float((_alpha(q_next) - _alpha(q_t)).square().mean().sqrt())

    out["estep_target_gap"] = float("nan")
    if canonical_e_step_update(cfg.e_step_update) == "mm_exact":
        target_kwargs = dict(ikw)
        target_kwargs["mm_damping"] = 1.0
        q_target = e_step_iteration(q_t, mu_p, sigma_p, model.group, **target_kwargs)
        target_mse = torch.stack([
            (q_next.mu - q_target.mu).square().mean(),
            (q_next.sigma - q_target.sigma).square().mean(),
            (q_next.phi - q_target.phi).square().mean(),
        ]).mean()
        out["estep_target_gap"] = float(target_mse.sqrt())
    return out


@torch.no_grad()
def across_layer_belief_trace(
    model,
    token_ids: torch.Tensor,           # (B, N) token ids; only sequence 0 is used

    *,
    snapshot: 'Optional[DiagnosticSnapshot]' = None,
) -> Dict[str, torch.Tensor]:
    r"""Per-LAYER converged beliefs by replaying the block stack (mirrors vfe_stack's handoff).

    Returns ``mu`` (L, N, K), ``sigma`` (L, N, [K]), the cumulative affine-invariant SPD geodesic
    distance ``d_ai`` (L,) of each layer's covariance from layer 0, and per-layer mean effective
    rank ``effective_rank`` (L,). Shows how the belief geometry transforms with inference depth.
    """
    cfg = model.cfg
    if snapshot is None:
        belief, log_prior, rope = _encode_one(model, token_ids)
        mu_p, sigma_p = belief.mu, belief.sigma
        rho, rho_s = cfg.prior_handoff_rho, cfg.prior_handoff_sigma
        mus, sigmas = [], []
        for _ in range(cfg.n_layers):
            belief = vfe_block(
                belief, mu_p, sigma_p, model.group, cfg, log_prior=log_prior,
                block_norm=model.block_norm, head_mixer=model.head_mixer,
                cg_coupling=model.cg_coupling,                       # replay the trained model
                lambda_beta=cfg.lambda_beta, connection_W=getattr(model, "connection_W", None),
                connection_M=getattr(model, "connection_M", None),
                connection_L=getattr(model, "connection_L", None),
                rope=rope, rope_on_cov=cfg.rope_full_gauge, rope_on_value=cfg.rope_on_value,
                gauge_parameterization=cfg.gauge_parameterization,
            )
            mus.append(belief.mu)
            sigmas.append(belief.sigma)
            mu_p = (1.0 - rho) * mu_p + rho * belief.mu
            sigma_p = (1.0 - rho_s) * sigma_p + rho_s * belief.sigma
    else:
        snapshot = model._validate_diagnostic_snapshot(token_ids, snapshot)
        mus = [belief.mu[0] for belief in snapshot.layer_outputs]
        sigmas = [belief.sigma[0] for belief in snapshot.layer_outputs]
    mu_stack = torch.stack(mus)                                   # (L, N, K)
    sig_stack = torch.stack(sigmas)                              # (L, N, [K])
    base = sig_stack[0].unsqueeze(0).expand_as(sig_stack)
    d_ai = metrics.spd_geodesic_distance(base, sig_stack).mean(dim=-1)   # (L,) mean over tokens
    eff = metrics.effective_rank_per_token(
        sig_stack,
        diagonal=cfg.diagonal_covariance,
        eps=cfg.eps,
        family=cfg.family,
    ).mean(dim=-1)                                                        # (L,)
    r_one = metrics.rank_one_residual(mu_stack)                          # (L,) Dong r(X) per layer (F2/EXP-7)
    return {"mu": mu_stack, "sigma": sig_stack, "d_ai": d_ai,
            "effective_rank": eff, "rank_one_residual": r_one}


@torch.no_grad()
def numerical_health(
    model,
    token_ids: torch.Tensor,           # (B, N) token ids; only sequence 0 is used

    *,
    snapshot: 'Optional[DiagnosticSnapshot]' = None,
) -> Dict[str, float]:
    r"""Non-finite fractions of the converged intermediates + the worst covariance conditioning.

    Replays the converged belief and its energy / attention (as model.diagnostics does) and
    reports ``nan_inf_fraction`` for mu, sigma, phi, energy, and beta, plus the maximum spectral
    condition number of the belief covariances. A near-zero finiteness map certifies stability is
    genuine convergence, not masked blow-ups. (Numerical-FALLBACK activation counters --
    safe_cholesky jitter rounds, pinv fallbacks -- would require instrumenting numerics and are
    left to a future pass.)
    """
    from vfe3.numerics import nan_inf_fraction
    from vfe3.geometry.transport import RopeTransport   # m5: used below under pos_rotation='rope' (was NameError)
    cfg = model.cfg
    if snapshot is None:
        belief, log_prior, rope = _encode_one(model, token_ids)
        ikw = _iter_kwargs(model, log_prior, rope)
        out = belief
        for _ in range(cfg.n_e_steps):
            out = e_step_iteration(out, belief.mu, belief.sigma, model.group, **ikw)
    else:
        snapshot = model._validate_diagnostic_snapshot(token_ids, snapshot)
        out = _snapshot_sequence(snapshot.layer_converged[0])
        log_prior = model._first_sequence_log_prior(snapshot.log_prior, token_ids.shape[0])
        rope = snapshot.rope
    # Build Omega under the ACTIVE connection regime (audit 2026-06-10 F8e): this previously
    # defaulted to flat transport, so under regime_ii the reported nan/beta/energy fractions
    # described a flat-transport belief, not the model that trained. Mirrors converged_state.
    omega = _transport(
        out.phi, model.group, transport_mode=cfg.transport_mode,
        gauge_parameterization=cfg.gauge_parameterization,
        omega=out.omega,
        reflection=out.reflection,
        right_phi=out.right_phi,
        mu=(out.mu if cfg.transport_mode in _TRANSPORT_NEEDS_MU else None),
        sigma=(out.sigma if cfg.transport_mode in _TRANSPORT_NEEDS_SIGMA else None),
        connection_W=getattr(model, "connection_W", None),
        connection_M=getattr(model, "connection_M", None),
        connection_L=getattr(model, "connection_L", None),
        link_alpha=cfg.link_alpha, link_soft_cap=cfg.link_soft_cap,
        cocycle_relaxation=cfg.cocycle_relaxation,
    )
    # Wrap in RopeTransport under pos_rotation='rope' so the reported nan/energy/beta fractions
    # describe the RoPE-rotated belief the model runs, mirroring converged_state/diagnostics (r2 id11).
    fam = get_family(cfg.family)
    if rope is not None:
        rope_omega = RopeTransport(base=omega, rope=rope, on_cov=cfg.rope_full_gauge,
                                   on_value=cfg.rope_on_value)
        mu_t    = transport_mean(rope_omega, out.mu)
        sigma_t = fam.transport_dispersion(out.sigma, rope_omega)
    else:
        mu_t = transport_mean(omega.unsqueeze(0), out.mu.unsqueeze(0))[0]
        sigma_t = fam.transport_dispersion(out.sigma.unsqueeze(0), omega.unsqueeze(0))[0]
    energy = pairwise_energy(fam(out.mu, out.sigma), fam(mu_t, sigma_t), alpha=cfg.renyi_order,
                             kl_max=cfg.kl_max, eps=cfg.eps, divergence_family=cfg.divergence_family,
                             irrep_dims=model.group.irrep_dims)
    beta = attention_weights(
        energy,
        tau=attention_tau(model.effective_kappa_beta(out.mu.device), model.group.irrep_dims),
        log_prior=log_prior)
    condition = metrics.belief_spectrum(
        out.sigma,
        diagonal=(out.sigma.dim() == out.mu.dim()),
        eps=cfg.eps,
        family=cfg.family,
    )["condition"]
    cond = float(condition.max())
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

    *,
    snapshot: 'Optional[DiagnosticSnapshot]' = None,
) -> Dict[str, torch.Tensor]:
    r"""The converged-belief diagnostic state of one sequence, as tensors for the figures.

    Mirrors :meth:`VFEModel.diagnostics` EXACTLY (same active config: transport mode,
    connection_W, rope, family, divergence, alpha) but returns the underlying tensors the scalar
    diagnostics discard. The gauge-equivariance certificate, per-head gauge invariants, belief
    spectrum / SPD ellipses, and the guard-saturation / causal panels all read from these. Returns
    the converged ``mu`` (N, K), ``sigma`` (N, K) or (N, K, K), ``phi`` (N, n_gen), the active
    per-token vertex factor under the compatibility key ``exp_phi`` (N, K, K), and the dense pre-rope
    pairwise transport ``omega`` (N, N, K, K) that the equivariance/holonomy metrics consume,
    the pairwise ``energy`` and attention ``beta`` ((N, N) or (H, N, N)), and the per-token
    self-divergence ``self_div`` (N,) or (N, K).
    """
    from vfe3.model.stack import vfe_stack
    from vfe3.geometry.transport import RopeTransport, build_factored_transport, compute_transport_operators

    cfg = model.cfg
    was_training = model.training
    model.eval()
    try:
        if snapshot is None:
            belief, log_prior, rope = _encode_one(model, token_ids)
            cap: dict = {}                                       # q* capture (F self-term, as diagnostics)
            out = vfe_stack(
                belief, belief.mu, belief.sigma, model.group, cfg,
                log_prior=log_prior, block_norm=model.block_norm,
                head_mixer=model.head_mixer, cg_coupling=model.cg_coupling,
                lambda_beta=cfg.lambda_beta,
                kappa_beta_override=model.effective_kappa_beta(belief.mu.device),
                connection_W=getattr(model, "connection_W", None),
                connection_M=getattr(model, "connection_M", None),
                connection_L=getattr(model, "connection_L", None),
                rope=rope, rope_on_cov=cfg.rope_full_gauge, rope_on_value=cfg.rope_on_value,
                capture=cap,
                gauge_parameterization=cfg.gauge_parameterization,
            )
        else:
            snapshot = model._validate_diagnostic_snapshot(token_ids, snapshot)
            belief = _snapshot_sequence(snapshot.initial_belief)
            out = _snapshot_sequence(snapshot.stack_output)
            log_prior = model._first_sequence_log_prior(snapshot.log_prior, token_ids.shape[0])
            rope = snapshot.rope
            cap = {"converged": _snapshot_sequence(snapshot.layer_converged[-1])}
        rho, rho_s = cfg.prior_handoff_rho, cfg.prior_handoff_sigma     # rebuild last-block prior
        mu_p, sigma_p = belief.mu, belief.sigma                         # exact iff L==1 or rho==0
        for _ in range(cfg.n_layers - 1):
            mu_p = (1.0 - rho) * mu_p + rho * out.mu
            sigma_p = (1.0 - rho_s) * sigma_p + rho_s * out.sigma
        omega = _transport(                                            # active pairwise transport (pre-rope)
            out.phi, model.group, transport_mode=cfg.transport_mode,
            gauge_parameterization=cfg.gauge_parameterization,
            omega=out.omega,
            reflection=out.reflection,
            right_phi=out.right_phi,
            mu=(out.mu if cfg.transport_mode in _TRANSPORT_NEEDS_MU else None),
            sigma=(out.sigma if cfg.transport_mode in _TRANSPORT_NEEDS_SIGMA else None),
            connection_W=getattr(model, "connection_W", None),
            connection_M=getattr(model, "connection_M", None),
            connection_L=getattr(model, "connection_L", None),
            link_alpha=cfg.link_alpha, link_soft_cap=cfg.link_soft_cap,
            cocycle_relaxation=cfg.cocycle_relaxation,
        )
        fam = get_family(cfg.family)
        if rope is not None:
            rope_omega = RopeTransport(base=omega, rope=rope, on_cov=cfg.rope_full_gauge,
                                       on_value=cfg.rope_on_value)
            mu_t    = transport_mean(rope_omega, out.mu)
            sigma_t = fam.transport_dispersion(out.sigma, rope_omega)
        else:
            mu_t    = transport_mean(omega.unsqueeze(0), out.mu.unsqueeze(0))[0]
            sigma_t = fam.transport_dispersion(out.sigma.unsqueeze(0), omega.unsqueeze(0))[0]
        energy = pairwise_energy(
            fam(out.mu, out.sigma), fam(mu_t, sigma_t), alpha=cfg.renyi_order,
            kl_max=cfg.kl_max, eps=cfg.eps, divergence_family=cfg.divergence_family,
            irrep_dims=model.group.irrep_dims,
        )
        if snapshot is None:
            beta = attention_weights(
                energy,
                tau=attention_tau(model.effective_kappa_beta(out.mu.device), model.group.irrep_dims),
                log_prior=log_prior)
        else:
            beta = snapshot.beta_maps[-1]
            if energy.dim() == 2:
                beta = beta[0]
        _q_conv = cap["converged"]                           # q*: the F self-term reads the pre-
        self_div = self_divergence_for_alpha(                # transform converged belief (F19,
            fam(_q_conv.mu, _q_conv.sigma), fam(mu_p, sigma_p), alpha=cfg.renyi_order,   # as diagnostics)
            kl_max=cfg.kl_max, eps=cfg.eps, divergence_family=cfg.divergence_family,
            lambda_alpha_mode=cfg.lambda_alpha_mode,
        )
        if out.omega is not None:
            # Report compatibility: retain the historical ``exp_phi`` key while exposing the ACTIVE
            # omega-direct vertex frame. Compact storage stays compact through inference and is made
            # dense only at this explicit, off-hot-path figure boundary.
            exp_phi = out.omega.to_dense() if isinstance(out.omega, CompactBlockElement) else out.omega
        else:
            if out.right_phi is not None:
                exp_phi = build_factored_transport(
                    out.phi.unsqueeze(0), model.group, right_phi=out.right_phi,
                ).exp_phi[0]
            else:
                exp_phi = compute_transport_operators(out.phi.unsqueeze(0), model.group)["exp_phi"][0]
            if out.reflection is not None:
                # Active phi-path vertex factor g_i = R_i exp(phi_i); scaling row zero applies the
                # left reflection R_i = diag(sign_i, 1, ...), exactly matching _transport's fold.
                exp_phi = exp_phi.clone()
                exp_phi[..., 0, :] *= out.reflection[..., None]
        # ``report.py`` has legacy dense consumers (gauge-equivariance and curvature-field panels).
        # Preserve their public tensor contract without forcing live inference to densify compact U.
        omega_dense = omega.to_dense_omega() if isinstance(omega, CompactFactoredTransport) else omega
    finally:
        if was_training:
            model.train()
    return {
        "mu":       out.mu,
        "sigma":    out.sigma,
        "phi":      out.phi,
        "exp_phi":  exp_phi,
        "omega":    omega_dense,
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
        belief, log_prior, rope = _encode_one(model, token_ids)   # _encode_one folds the precision bias
        ikw = _iter_kwargs(model, log_prior, rope)
        mu_p, sigma_p = belief.mu, belief.sigma
        out = belief
        for _ in range(cfg.n_e_steps):                            # converge to q* (the operating point)
            out = e_step_iteration(out, mu_p, sigma_p, model.group, **ikw)
        omega = _transport(
            out.phi, model.group, transport_mode=cfg.transport_mode,
            gauge_parameterization=cfg.gauge_parameterization,
            omega=out.omega,
            reflection=out.reflection,
            right_phi=out.right_phi,
            mu=(out.mu if cfg.transport_mode in _TRANSPORT_NEEDS_MU else None),
            sigma=(out.sigma if cfg.transport_mode in _TRANSPORT_NEEDS_SIGMA else None),
            connection_W=getattr(model, "connection_W", None),
            connection_M=getattr(model, "connection_M", None),
            connection_L=getattr(model, "connection_L", None),
            link_alpha=cfg.link_alpha, link_soft_cap=cfg.link_soft_cap,
            cocycle_relaxation=cfg.cocycle_relaxation,
        )
        if rope is not None:
            omega = RopeTransport(base=omega, rope=rope, on_cov=cfg.rope_full_gauge,
                                  on_value=cfg.rope_on_value)
    kw = dict(                                                    # the oracle's free-energy knob bag
        tau=attention_tau(model.effective_kappa_beta(dev), model.group.irrep_dims),
        renyi_order=cfg.renyi_order, kl_max=cfg.kl_max, eps=cfg.eps,
        b0=_as_coeff(cfg.b0, dev), c0=_as_coeff(cfg.c0, dev), value=cfg.lambda_alpha,
        lambda_beta=cfg.lambda_beta, gradient_mode=cfg.gradient_mode, family=cfg.family,
        divergence_family=cfg.divergence_family, lambda_alpha_mode=cfg.lambda_alpha_mode,
        irrep_dims=model.group.irrep_dims, log_prior=log_prior,
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
def s_channel_refinement(
    model,
    token_ids: torch.Tensor,

    *,
    snapshot: 'Optional[DiagnosticSnapshot]' = None,
) -> Optional[dict]:
    r"""Model-channel (s) refinement diagnostics for the ``s_e_step=True`` path (sequence 0).

    Returns ``None`` when ``cfg.s_e_step`` is False (the s-channel does not run, so the figure is
    skipped). Otherwise replays the SAME two steps ``forward``/``diagnostics`` take -- the static
    encode ``s0 = encode_s(tokens)`` and the refined ``s1 = _refine_s(tokens, model_phi)`` under
    the selected fixed model frame -- and measures, per token position, how far the model channel moved and
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
    if snapshot is None:
        enc  = pb.encode(token_ids[:1])                                # (1, N, ...)
        belief_phi = model._apply_pos_phi(enc.phi[0]).unsqueeze(0)     # (1, N, n_gen) belief frame
        phi0 = model._resolve_model_frame(token_ids[:1], belief_phi)  # (1, N, n_gen) model frame
        rope = model._rope_rotation(token_ids.shape[1], token_ids.device)
        s0 = pb.encode_s(token_ids[:1])                               # static model channel
        s1 = model._refine_s(token_ids[:1], phi0, rope=rope)          # refined model channel
    else:
        snapshot = model._validate_diagnostic_snapshot(token_ids, snapshot)
        s0 = snapshot.s_encoded_belief
        s1 = snapshot.s_belief
        if s0 is None or s1 is None:
            raise RuntimeError("diagnostic snapshot is missing the active model-channel beliefs")
    s0_mu, s0_sigma = (tensor[0] for tensor in s0)                    # (N, K) static
    s1_mu, s1_sigma = (tensor[0] for tensor in s1)                    # (N, K) refined
    r_mu    = pb.r_mu.expand_as(s1_mu)                             # (N, K) frozen hyper-prior centroid
    r_sigma = bounded_variance_from_log(pb.r_sigma_log, eps=cfg.eps).expand_as(s1_sigma)
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
def model_channel_belief(
    model,
    token_ids: torch.Tensor,

    *,
    snapshot: 'Optional[DiagnosticSnapshot]' = None,
) -> Optional[dict]:
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
    if snapshot is None:
        s_encoded = pb.encode_s(token_ids[:1])
    else:
        snapshot = model._validate_diagnostic_snapshot(token_ids, snapshot)
        s_encoded = snapshot.s_encoded_belief
        if s_encoded is None:
            raise RuntimeError("diagnostic snapshot is missing the active model-channel belief")
    s_mu, s_sigma = (tensor[0] for tensor in s_encoded)            # (N, K)
    family = get_family(cfg.family)
    statistics = family.diagnostic_statistics(s_sigma, eps=cfg.eps)
    lam = torch.sort(
        statistics["covariance_spectrum"],
        dim=-1,
        descending=True,
    ).values
    result = {
        "mu_mean":    s_mu.mean(dim=0).cpu(),                      # (K,)
        "mu_std":     s_mu.std(dim=0).cpu(),                       # (K,)
        "sigma_mean": s_sigma.mean(dim=0).cpu(),                   # (K,)
        "spectrum":   lam.cpu(),                                   # (N, K)
        "eff_rank":   metrics.effective_rank_per_token(
            s_sigma,
            diagonal=cfg.diagonal_covariance,
            eps=cfg.eps,
            family=cfg.family,
        ).cpu(),                                                     # (N,)
    }
    if not family.dispersion_is_covariance:
        labels = family.diagnostic_labels()
        result["dispersion_label"] = labels["dispersion"]
        result["spectrum_label"] = labels["covariance_spectrum"]
    return result


@torch.no_grad()
def hyper_prior_centroid(
    model,
    token_ids: torch.Tensor,

    *,
    snapshot: 'Optional[DiagnosticSnapshot]' = None,
) -> Optional[dict]:
    r"""The hyper-prior centroid r and how the model channel s clusters around it (the ``r`` figure).

    Returns ``None`` when r does not exist (it is created only when lambda_h>0 OR s_e_step). Otherwise
    returns the centroid per coordinate (``r_mu`` (K,), ``r_sigma`` (K,) = exp(r_sigma_log)) and the
    model-channel population per coordinate over sequence 0 (``s_mu_mean``, ``s_mu_std``, ``s_sigma_mean``),
    so the figure can show the consensus r against the s distribution it anchors."""
    cfg, pb = model.cfg, model.prior_bank
    if getattr(pb, "r_mu", None) is None:
        return None
    if snapshot is None:
        s_encoded = pb.encode_s(token_ids[:1])
    else:
        snapshot = model._validate_diagnostic_snapshot(token_ids, snapshot)
        s_encoded = snapshot.s_encoded_belief
        if s_encoded is None:
            raise RuntimeError("diagnostic snapshot is missing the active model-channel belief")
    s_mu, s_sigma = (tensor[0] for tensor in s_encoded)            # (N, K)
    r_sigma = bounded_variance_from_log(pb.r_sigma_log, eps=cfg.eps).detach().cpu()  # (K,)
    return {
        "r_mu":         pb.r_mu.detach().cpu(),                                      # (K,)
        "r_sigma":      r_sigma,                                                      # (K,)
        "s_mu_mean":    s_mu.mean(dim=0).cpu(),                                      # (K,)
        "s_mu_std":     s_mu.std(dim=0).cpu(),                                       # (K,)
        "s_sigma_mean": s_sigma.mean(dim=0).cpu(),                                   # (K,)
    }


@torch.no_grad()
def hyper_prior_coupling(
    model,
    token_ids: torch.Tensor,

    *,
    snapshot: 'Optional[DiagnosticSnapshot]' = None,
) -> Optional[dict]:
    r"""Per-token hyper-prior divergence KL(s_i||r) for sequence 0 (the ``h`` figure: the lambda_h block).

    Returns ``None`` when r does not exist. Uses ``model._hyper_prior_kl`` so the plotted per-position
    KL(s_i||r) is the SAME quantity the diagnostics decomposition and the forward loss carry; ``lambda_h``
    is returned for the title (the weight the block enters F with)."""
    pb = model.prior_bank
    if getattr(pb, "r_mu", None) is None:
        return None
    if snapshot is None:
        s_belief = model._refined_s_belief(token_ids)              # s1 under s_e_step; None -> raw
    else:
        snapshot = model._validate_diagnostic_snapshot(token_ids, snapshot)
        s_belief = snapshot.s_belief
        if s_belief is None:
            raise RuntimeError("diagnostic snapshot is missing the active model-channel belief")
    kl = model._hyper_prior_kl(token_ids[:1], s_belief=s_belief)[0]   # (N,)
    return {"kl_s_r": kl.cpu(), "lambda_h": float(model.cfg.lambda_h)}


@torch.no_grad()
def gamma_attention(
    model,
    token_ids: torch.Tensor,

    *,
    snapshot: 'Optional[DiagnosticSnapshot]' = None,
) -> Optional[dict]:
    r"""Model-coupling attention gamma_ij for sequence 0 (the gamma figure), via
    :meth:`VFEModel.gamma_attention_maps`.

    Returns ``None`` when the model channel is inactive. Otherwise returns the per-head gamma weights
    ``gamma`` (H, N, N) = softmax_j(log pi^s - E^s/tau_g) on the model-channel beliefs s under the
    selected effective model frame -- the s-channel analogue of the belief beta maps."""
    ids = token_ids if snapshot is not None else token_ids[:1]
    g = model.gamma_attention_maps(ids, snapshot=snapshot)
    if g is None:
        return None
    return {"gamma": g.cpu()}


@torch.no_grad()
def model_channel_bank(
    model,
    token_batches:  Iterable[torch.Tensor],   # iterable of (B, N) token-id batches

    *,
    device:         Optional[torch.device] = None,
    max_tokens:     Optional[int]          = None,
    max_sequences:  Optional[int]          = None,
) -> Optional[Dict[str, torch.Tensor]]:
    r"""Collect the model-channel beliefs s_i over many sequences -- the bank for the model-channel UMAP.

    ``None`` when the model channel is inactive (no s tables). The s-channel sibling of :func:`belief_bank`:
    for each batch it looks up the static model-channel belief ``s_i = N(s_mu, s_sigma)`` via ``encode_s``,
    and -- when ``s_e_step`` -- refines it through the s E-step (so s is position-dependent, exactly as the
    belief bank's q is), then stacks the per-token ``mu = s_mu`` (M, K) and ``sigma = s_sigma`` (M, K) with
    ``token_ids`` (M,), ``seq_idx`` (M,), and within-sequence ``pos_idx`` (M,). Under
    ``s_frame_mode='phi_tilde'`` the bank also carries the independently resolved model frame
    ``phi`` (M, n_gen), enabling a model-frame UMAP. Tied mode omits that duplicate channel. Feeds
    ``plot_belief_umap`` directly, so the redesigned cluster-and-distinctive-token view applies to the
    slow channel.
    NB i and j are token POSITIONS, not separate channels: this single bank embeds every s_i (all positions);
    the i<->j pairing lives only in the gamma_ij model-coupling attention (see :func:`gamma_attention`).
    """
    _validate_bank_caps(max_tokens=max_tokens, max_sequences=max_sequences)
    if not model._model_channel_active:
        return None
    device = device or _model_device(model)
    was_training = model.training
    model.eval()
    mus, sigmas, phis, tids, sidx, pidx = [], [], [], [], [], []
    seq_counter = 0
    try:
        for tokens in token_batches:
            tokens = tokens.to(device)
            s_mu, s_sigma = model.prior_bank.encode_s(tokens)         # (B, N, K) static model channel
            b, n = tokens.shape
            model_phi = None
            if model.cfg.s_e_step:                                    # refine -> position-dependent, like q
                belief_phi = model._apply_pos_phi(model.prior_bank.encode(tokens).phi)
                model_phi = model._resolve_model_frame(tokens, belief_phi)
                rope = model._rope_rotation(n, device)
                s_mu, s_sigma = model._refine_s(tokens, model_phi, rope=rope)
            mus.append(s_mu.reshape(b * n, -1))
            sigmas.append(s_sigma.reshape(b * n, -1))
            if model.cfg.s_frame_mode == "phi_tilde":
                if model_phi is None:
                    belief_phi = model._apply_pos_phi(model.prior_bank.encode(tokens).phi)
                    model_phi = model._resolve_model_frame(tokens, belief_phi)
                phis.append(model_phi.reshape(b * n, -1))
            tids.append(tokens.reshape(b * n))
            sidx.append(torch.arange(seq_counter, seq_counter + b, device=device).repeat_interleave(n))
            pidx.append(torch.arange(n, device=device).repeat(b))
            seq_counter += b
            token_count = sum(batch.shape[0] for batch in tids)
            if ((max_tokens is not None and token_count >= max_tokens)
                    or (max_sequences is not None and seq_counter >= max_sequences)):
                break
    finally:
        if was_training:
            model.train()
    bank = {
        "mu":        torch.cat(mus),
        "sigma":     torch.cat(sigmas),
        "token_ids": torch.cat(tids),
        "seq_idx":   torch.cat(sidx),
        "pos_idx":   torch.cat(pidx),
    }
    if phis:
        bank["phi"] = torch.cat(phis)
    return _slice_bank_to_cap(
        bank,
        max_tokens=max_tokens,
        max_sequences=max_sequences,
    )


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
