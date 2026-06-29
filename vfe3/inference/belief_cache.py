r"""Belief-prefix cache for the no-grad EFE rollout (Phase 3a).

The EFE scorer rolls a shared context belief forward over Kp candidate continuations (and, for
``efe_rollout``, over an H-step horizon). The naive path
(``vfe3/inference/policy.py::_rollout_predictive``) re-runs the FULL belief E-step over
(context + candidate) for every candidate, so the context-context attention block is recomputed Kp
times and -- because the scorer reads only the LAST position's decode -- thrown away. This module
computes that block ONCE: under a causal attention prior the converged beliefs at context positions
0..N-1 are invariant to any appended token (the future is masked to exactly zero weight), so only the
appended positions' E-step rows need recomputing, attending causally to the (shared) context.

Scope (:func:`cache_supported`): the fast path is verified exact ONLY under the config it is golden-
tested against (``tests/test_belief_cache.py``) -- a single block (``n_layers=1``), a single E-step
iteration (``n_e_steps=1``), the closed-form filtering kernel (``gaussian_diagonal`` + KL +
``include_attention_entropy``), flat transport, a causal belief prior, a frozen gauge frame
(``e_phi_lr=0``), no gauge-RoPE (``pos_rotation='none'``), no model channel (``s_e_step=False``), no
precision-bias fold, no learned head mixer / CG coupling, and the Fisher mean preconditioner with no
mean trust region. ANY other config makes :func:`cache_supported` return ``False`` and the caller
falls back to the full recompute in ``policy.py`` (correct, just slower). The fast path reuses the
SAME primitives the live E-step uses (``transport_mean`` / ``transport_covariance``,
``pairwise_energy``, ``attention_weights``, the registered belief kernel, the family Fisher natural
gradient, the SPD retraction), so it reproduces the full recompute to within float tolerance.

Correctness rests on the causal-invariance linchpin verified by the ``wf_a12bc02f-988`` cacheability
audit: with ``beta_attention_prior='causal'`` (j>i -> -inf -> exp = 0 exactly),
``gradient_mode='filtering'`` (keys detached, no key back-reaction), and ``e_phi_lr=0`` (no
cross-position phi autograd), a context position's converged belief does not depend on any later
token. Equality is to FLOAT TOLERANCE, not byte-identity: masked-future terms are exact zeros, but a
partial recompute reduces over a different key-axis length and float addition is non-associative (GPU
reduction order in particular).
"""

from typing import Optional, Tuple

import torch

from vfe3.alpha_i import alpha_gradient_coefficient, alpha_is_per_coord
from vfe3.belief import BeliefState
from vfe3.families.base import get_family
from vfe3.free_energy import attention_tau, attention_weights, pairwise_energy, self_divergence_for_alpha
from vfe3.geometry.retraction import get_retraction
from vfe3.geometry.transport import compute_transport_operators, transport_covariance, transport_mean
from vfe3.gradients.kernels import get_kernel

# Belief priors that mask the future to exactly zero weight (j>i -> -inf -> exp = 0), the condition
# under which appending a token leaves every context position's converged belief unchanged.
_CAUSAL_PRIORS: frozenset = frozenset({"causal", "causal_alibi", "causal_windowed"})


def _as_coeff(v: 'float | list | tuple', device: torch.device) -> 'float | torch.Tensor':
    r"""Pass a scalar b0/c0/kappa through unchanged; turn a list into a (K,) float32 tensor on device
    (inlined from ``vfe3.model.block._as_coeff`` to avoid an inference -> model import cycle)."""
    return torch.as_tensor(v, dtype=torch.float32, device=device) if isinstance(v, (list, tuple)) else v


def cache_supported(cfg: 'object') -> bool:
    r"""Whether the prefix-cache fast path is verified exact for ``cfg`` (else the caller falls back to
    the full recompute). The conjunction of the golden-tested validity conditions (module docstring):
    one block, one E-step iteration, the closed-form filtering kernel, flat transport, a causal belief
    prior, a frozen gauge frame, and none of the cross-position / non-kernel toggles active."""
    return (
        cfg.n_layers == 1
        and cfg.n_e_steps == 1
        and cfg.gradient_mode == "filtering"
        and cfg.family == "gaussian_diagonal"
        and cfg.divergence_family == "renyi"
        and abs(cfg.renyi_order - 1.0) < 1e-9
        and cfg.include_attention_entropy
        and cfg.transport_mode == "flat"
        and cfg.e_phi_lr == 0.0
        and cfg.beta_attention_prior in _CAUSAL_PRIORS
        and not cfg.s_e_step
        and not cfg.precision_weighted_attention
        and cfg.pos_rotation == "none"
        and not cfg.use_head_mixer
        and not cfg.use_cg_coupling
        and cfg.e_step_mu_precond == "fisher"
        and cfg.e_mu_q_trust is None
    )


def _appended_belief_step(
    beliefs:       BeliefState,             # (B', M) iteration-0 (encode + pos_phi) FULL field
    log_prior_app: torch.Tensor,            # (L, M) or (H, L, M) causal prior, appended query rows

    model:         'object',                 # VFEModel: group / cfg / learned-scalar handles
    n_context:     int,                      # N: appended positions are [N:]
    tau:           'float | torch.Tensor',   # softmax temperature kappa*sqrt(dim_h)

) -> BeliefState:
    r"""One filtering-kernel E-step iteration for the APPENDED query rows against the full (causal)
    key field, mirroring ``belief_gradients`` (kernel branch) + ``e_step_iteration`` (Fisher natural
    gradient + SPD retraction) restricted to query positions ``[N:]``. With ``n_e_steps=1`` the keys
    are the iteration-0 (encode) field and the layer-0 self-coupling prior is the encode belief
    (q0 == p0), exactly as ``forward_beliefs`` passes ``mu_p = beliefs.mu`` into the first block."""
    cfg, group = model.cfg, model.group
    fam = get_family(cfg.family)
    eps, kl_max = cfg.eps, cfg.kl_max
    N = n_context

    mu_q,  sig_q,  phi_q = beliefs.mu[:, N:], beliefs.sigma[:, N:], beliefs.phi[:, N:]   # (B', L, .)
    mu_k,  sig_k         = beliefs.mu.detach(), beliefs.sigma.detach()                   # (B', M, K) frozen keys
    phi_k                = beliefs.phi                                                   # (B', M, n_gen)
    mu_p,  sig_p         = mu_q, sig_q                                                   # layer-0 prior = encode (q0==p0)

    # Transported keys for the appended query rows: flat Omega_ij = exp(phi_i^q) exp(-phi_j^k).
    exp_q     = compute_transport_operators(phi_q, group)["exp_phi"]                     # (B', L, K, K)
    exp_neg_k = compute_transport_operators(phi_k, group)["exp_neg_phi"]                 # (B', M, K, K)
    omega     = torch.einsum("bikl,bjlm->bijkm", exp_q, exp_neg_k)                       # (B', L, M, K, K)
    mu_t      = transport_mean(omega, mu_k)                                              # (B', L, M, K)
    sig_t     = transport_covariance(omega, sig_k)                                       # (B', L, M, K)

    sd = self_divergence_for_alpha(
        fam(mu_q, sig_q), fam(mu_p, sig_p), alpha=1.0, kl_max=kl_max, eps=eps,
        divergence_family=cfg.divergence_family, lambda_alpha_mode=cfg.lambda_alpha_mode,
    )
    energy = pairwise_energy(
        fam(mu_q, sig_q), fam(mu_t, sig_t), alpha=1.0, kl_max=kl_max, eps=eps,
        divergence_family=cfg.divergence_family, irrep_dims=group.irrep_dims,
    )
    beta      = attention_weights(energy, tau=tau, log_prior=log_prior_app)              # (B', L, M)
    pair_mask = ((energy > 0.0) & (energy < kl_max)).to(beta.dtype)

    log_alpha = getattr(model, "log_alpha", None)
    coef = alpha_gradient_coefficient(
        sd, value=cfg.lambda_alpha, b0=_as_coeff(cfg.b0, mu_q.device), c0=_as_coeff(cfg.c0, mu_q.device),
        mode=cfg.lambda_alpha_mode, log_alpha=log_alpha,
    )
    if not alpha_is_per_coord(cfg.lambda_alpha_mode):
        coef = coef.unsqueeze(-1)
    _llb        = getattr(model, "log_lambda_beta", None)
    lambda_beta = cfg.lambda_beta if _llb is None else _llb.exp()

    grad_mu, grad_sigma = get_kernel(cfg.family)(
        mu_q, sig_q, mu_p, sig_p, mu_t, sig_t, beta * pair_mask, coef,
        kl_max=kl_max, eps=eps, lambda_beta=lambda_beta, irrep_dims=group.irrep_dims,
    )
    nat_mu, nat_sigma = fam(mu_q, sig_q).natural_gradient(grad_mu, grad_sigma, eps=eps)
    mu_new  = mu_q - cfg.e_q_mu_lr * nat_mu
    sig_new = get_retraction(cfg.spd_retract_mode)(
        sig_q, -cfg.e_q_sigma_lr * nat_sigma, mu_q.dim(),
        trust_region=cfg.e_sigma_q_trust, eps=eps, sigma_max=cfg.sigma_max,
    )
    return BeliefState(mu=mu_new, sigma=sig_new, phi=phi_q)


@torch.no_grad()
def rollout_predictive_cached(
    context:     torch.Tensor,             # (B, N) context ids
    candidates:  torch.Tensor,             # (B, Kp, L) candidate continuation ids

    model:       'object',                  # VFEModel: prior_bank / group / cfg + belief helpers

    *,
    base_logits: Optional[torch.Tensor] = None,   # (B, V) reused base last-position logits
) -> 'Tuple[torch.Tensor, torch.Tensor]':
    r"""Cache-accelerated drop-in for ``policy._rollout_predictive`` on the :func:`cache_supported`
    path. Returns the predicted outcome distribution q(o|pi) = p(o | q*_pi) as log-probs (B, Kp, V)
    read from the LAST appended position, and the raw continuation log-prob (B, Kp) of the first
    action token under the BASE predictive -- identical semantics to ``_rollout_predictive``, but the
    Kp candidates share one context pass: only the appended positions' E-step rows are recomputed,
    attending causally to the shared context. The caller guarantees N + L <= ``cfg.max_seq_len`` (no
    sliding-window eviction, which would invalidate the prefix)."""
    B, N = context.shape
    Kp, L = candidates.shape[1], candidates.shape[2]
    M = N + L
    device = context.device

    ext = torch.cat([context.unsqueeze(1).expand(B, Kp, N), candidates], dim=2).reshape(B * Kp, M)

    # Iteration-0 field: encode + positional phi. With n_e_steps=1 these are exactly the keys the
    # appended rows attend to (the field the first E-step iteration reads).
    beliefs = model.prior_bank.encode(ext)                                  # (B*Kp, M) belief
    beliefs = beliefs._replace(phi=model._apply_pos_phi(beliefs.phi))

    log_prior = model._attention_log_prior(M, device)                       # (M, M) or (H, M, M) causal
    log_prior_app = log_prior[..., N:, :]                                   # appended query rows -> (..., L, M)
    tau = attention_tau(_as_coeff(model.cfg.kappa_beta, device), model.group.irrep_dims)

    app = _appended_belief_step(beliefs, log_prior_app, model, N, tau)

    # Post-E-step per-position transforms then decode, matching forward_beliefs at n_layers=1:
    # block_norm (if any) -> (no inter-block handoff) -> final_norm (if any) -> decode.
    mu_out = app.mu
    if model.block_norm is not None:
        mu_out = model.block_norm(mu_out, app.sigma)
    if model.final_norm is not None:
        mu_out = model.final_norm(mu_out, app.sigma)
    logits = model.prior_bank.decode(mu_out.float(), app.sigma.float())     # (B*Kp, L, V)
    q_log = torch.log_softmax(logits[:, -1, :], dim=-1).reshape(B, Kp, -1)  # (B, Kp, V) = log q(o|pi)

    if base_logits is None:
        base_logits = model.forward(context)[:, -1, :]                      # (B, V) base last-position logits
    base_logp = torch.log_softmax(base_logits, dim=-1)
    log_prob = torch.gather(base_logp, 1, candidates[:, :, 0])              # (B, Kp) logprob of the first action
    return q_log, log_prob
