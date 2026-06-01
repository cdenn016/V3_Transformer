r"""The full VFE_3.0 model: encode -> E-step inference -> decode -> cross-entropy.

No neural layers: the only parameters are the PriorBank's prior tables. The E-step
is unrolled into the training graph (the differentiable filtering kernel), so the CE
loss backpropagates through inference to the encode/phi priors. Batching loops over
the batch around the (unbatched) E-step; decode and CE are batched.
"""

import inspect
from contextlib import nullcontext
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from vfe3.attention_prior import attention_log_prior
from vfe3.belief import BeliefState
from vfe3.config import VFE3Config
from vfe3.geometry.groups import GaugeGroup, get_group
from vfe3.geometry.norms import get_norm
from vfe3.model.head_mixer import HeadMixer
from vfe3.model.prior_bank import PriorBank
from vfe3.model.stack import vfe_stack


def _positional_arity(builder: Callable) -> int:
    r"""Count the builder's required positional parameters (the K, n_heads, ... axes)."""
    n = 0
    for p in inspect.signature(builder).parameters.values():
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty:
            n += 1
    return n


def build_group(cfg: VFE3Config) -> GaugeGroup:
    r"""Construct the gauge group from config, dispatching on the builder's positional
    arity so a newly registered group slots in by ``register_group`` alone (no call-site
    edit). Arity 1 -> ``builder(K)`` (glk, so_k); arity 2 -> ``builder(K, n_heads)``
    (block_glk). Higher arities are an unsupported registration error."""
    builder = get_group(cfg.gauge_group)
    arity = _positional_arity(builder)
    if arity == 1:
        return builder(cfg.embed_dim)
    if arity == 2:
        return builder(cfg.embed_dim, cfg.n_heads)
    raise ValueError(
        f"gauge group {cfg.gauge_group!r} builder has unsupported positional arity {arity}; "
        f"build_group dispatches K (arity 1) or (K, n_heads) (arity 2)"
    )


class VFEModel(nn.Module):
    """encode -> E-step stack -> decode -> CE. Parameters live only in the PriorBank."""

    def __init__(self, cfg: VFE3Config) -> None:
        super().__init__()
        # Reproducibility is pinned at the entry point run_training (torch.manual_seed(cfg.seed)
        # before model + loader are built), NOT here: seeding inside __init__ would clobber a
        # caller-set RNG state (e.g. a test that seeds then constructs several models).
        self.cfg = cfg
        self.group = build_group(cfg)
        n_gen = self.group.generators.shape[0]
        self.prior_bank = PriorBank(
            cfg.vocab_size, cfg.embed_dim, n_gen,
            decode_tau=cfg.decode_tau, eps=cfg.eps,
            diagonal_covariance=cfg.diagonal_covariance,
            use_prior_bank=cfg.use_prior_bank,
            encode_mode=cfg.encode_mode, decode_mode=cfg.decode_mode,
        )
        # Stateless norm instances built ONCE (audit 2d/4f): they are parameter-free pure
        # maps (K, eps), so re-instantiating them per block/forward only churned objects.
        self.block_norm = get_norm(cfg.norm_type_block)(cfg.embed_dim, eps=cfg.eps) \
            if cfg.norm_type_block != "none" else None
        self.final_norm = get_norm(cfg.norm_type_final)(cfg.embed_dim, eps=cfg.eps) \
            if cfg.norm_type_final != "none" else None
        # Opt-in Schur-commutant head mixer (default off). Built ONCE from the gauge group's
        # irrep blocks; HeadMixer rejects a single-block group at construction (glk / so_k have
        # nothing to mix), so a bad gauge_group + use_head_mixer pair fails here, not at forward.
        self.head_mixer = HeadMixer(self.group.irrep_dims) if cfg.use_head_mixer else None
        if (not cfg.use_prior_bank) and cfg.detach_e_step:
            # Joint-toggle footgun (audit 2026-05-31): the detached E-step severs the encode prior
            # tables (mu/sigma/phi_embed) from the loss, and the linear decode reads only mu_final,
            # so ONLY output_proj_weight would train -- the prior bank is effectively frozen.
            import warnings
            warnings.warn(
                "use_prior_bank=False with detach_e_step=True freezes the encode prior tables "
                "(mu_embed/sigma_log_embed/phi_embed): the detached E-step cuts them off and the "
                "linear decode reads only mu_final, so only output_proj_weight trains. Set "
                "detach_e_step=False to learn the prior tables.",
                stacklevel=2,
            )
        # Causal/attention log-prior is loop-invariant for fixed (N, device, dtype); cache it
        # (audit 4e) keyed on those so it is built once, not every forward. Not an nn.buffer
        # because it depends on the runtime N (sequence length), which varies across calls.
        self._log_prior_cache: dict = {}

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True) -> "VFEModel":
        r"""Carry the gauge group's generators through ``.to(...)`` / ``.cuda()`` etc.

        ``self.group`` is a plain ``GaugeGroup`` dataclass, not an ``nn.Module``, so its
        ``generators`` tensor is outside the parameter/buffer system and would NOT follow a
        dtype/device move -- leaving the E-step transport (belief.phi, which DOES move)
        matmul'd against stale-device/dtype generators. Re-map them here so the module's
        device/dtype contract holds (CLAUDE.md: device-agnostic, float32-with-CUDA)."""
        super()._apply(fn, recurse)
        self.group.generators = fn(self.group.generators)
        self._log_prior_cache.clear()        # device/dtype moved: cached masks are now stale
        return self

    def _attention_log_prior(
        self,
        n:      int,                          # sequence length N (varies across calls)
        device: torch.device,
    ) -> torch.Tensor:
        r"""Loop-invariant attention log-prior, cached on (N, device, dtype) (audit 4e).

        The dtype is taken from the prior-bank mean table so the mask matches the belief
        dtype after a ``.to(torch.float64)`` move (audit 2f: the old call omitted dtype)."""
        dtype = self.prior_bank.mu_embed.dtype
        key = (n, device, dtype)
        cached = self._log_prior_cache.get(key)
        if cached is None:
            cached = attention_log_prior(self.cfg.attention_prior, n, n, device=device, dtype=dtype)
            self._log_prior_cache[key] = cached
        return cached

    def forward(
        self,
        token_ids: torch.Tensor,         # (B, N) integer token ids
        targets:   Optional[torch.Tensor] = None,   # (B, N) next-token ids (-100 = ignore)
    ) -> 'torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]':
        r"""Forward pass; returns logits, or (logits, loss, ce) when targets are given."""
        B, N = token_ids.shape
        beliefs = self.prior_bank.encode(token_ids)              # (B, N, K) ...
        log_prior = self._attention_log_prior(N, token_ids.device)

        # The E-step stack is vectorized over the batch (audit 4c): the belief tuple carries a
        # leading B axis through transport / gradients / retraction in one set of kernels, instead
        # of a serial per-sequence Python loop. Sequences are independent (each reads only its own
        # belief and the shared, sequence-independent log_prior), so the batched result equals the
        # per-sample result (pinned by tests/test_perf_equivalence.py::test_batched_forward_equals_per_sample).
        run = torch.no_grad() if self.cfg.detach_e_step else nullcontext()
        with run:
            out = vfe_stack(beliefs, beliefs.mu, beliefs.sigma, self.group, self.cfg,
                            log_prior=log_prior, block_norm=self.block_norm)
        mu_final = out.mu                                        # (B, N, K)
        sigma_final = out.sigma

        if self.head_mixer is not None:                          # opt-in head mixing, after E-step / before norm
            mu_final, sigma_final = self.head_mixer(mu_final, sigma_final)

        if self.final_norm is not None:                          # config-selected final norm (cached)
            mu_final = self.final_norm(mu_final, sigma_final)

        logits = self.prior_bank.decode(mu_final, sigma_final)   # (B, N, V)
        if targets is None:
            return logits

        flat_logits = logits.reshape(-1, self.cfg.vocab_size)
        flat_targets = targets.reshape(-1)
        if (flat_targets != -100).any():
            ce = F.cross_entropy(flat_logits, flat_targets, ignore_index=-100)
        else:
            # All-ignore microbatch: F.cross_entropy returns 0/0 = NaN (mean over zero
            # counted tokens), which poisons logging / NaN-guards / grad-accum means. Emit
            # a finite, grad-connected zero instead (a dead-but-clean step).
            ce = flat_logits.sum() * 0.0
        loss = ce
        if self.cfg.mass_phi > 0.0:
            # M-step gauge-frame penalty (manuscript Algorithm 1 M-step loss): regularizes the
            # CONVERGED output phi -> backprops to the learned prior table phi_embed. This is the
            # outer-loss role; mass_phi ALSO enters the inner phi E-step objective (e_step:
            # phi_alignment_loss), shaping the inference trajectory. Both roles are in the
            # manuscript algorithm (E-step phi gradient and M-step loss both carry alpha_phi/2||phi||^2).
            loss = loss + 0.5 * self.cfg.mass_phi * (out.phi ** 2).mean()
        return logits, loss, ce.detach()

    @torch.no_grad()
    def diagnostics(
        self,
        token_ids: torch.Tensor,           # (B, N) token ids; only sequence 0 is used
    ) -> dict:
        r"""Faithful per-step VFE diagnostics at the converged belief (no_grad).

        Recomputes the SAME quantities the E-step uses (transport, pairwise energy
        E_ij, attention weights beta_ij, self-divergence D(q_i||p_i), self-coupling
        alpha_i) at the converged belief returned by :func:`vfe_stack`, then feeds
        them to :mod:`vfe3.metrics`. Every knob matches what the forward pass passed
        to the E-step (``cfg.tau``, ``cfg.family``, ``cfg.divergence_family``,
        ``cfg.alpha_div``, ``cfg.kl_max``, ``cfg.eps``,
        ``cfg.alpha_mode``/``value``/``b0``/``c0``, ``group.irrep_dims``, the cached
        attention log-prior), so the diagnostic beta is the attention pattern at the
        fixed point, not a re-derivation.

        This is OFF the training hot path: it is never called on a train step, adds no
        argument or branch to :meth:`forward`, and runs under ``torch.no_grad``. The
        last-block prior is reconstructed by mirroring :func:`vfe_stack`'s
        ``prior_handoff`` fold; this is EXACT when ``n_layers == 1`` (default) or
        ``prior_handoff_rho == 0``, and an approximation otherwise (the single
        converged belief stands in for the per-block intermediates).

        Returns ``{attn_entropy, self_coupling, belief_coupling, attention_entropy,
        total, effective_rank}`` (nats; ``effective_rank`` is the per-token
        belief-variance spectrum effective rank, not an attention rank).
        """
        from vfe3.inference.e_step import _transport
        from vfe3.geometry.transport import transport_mean, transport_covariance
        from vfe3.free_energy import pairwise_energy, self_divergence_for_alpha, attention_weights
        from vfe3.alpha_i import self_coupling_alpha
        from vfe3 import metrics

        cfg = self.cfg
        enc = self.prior_bank.encode(token_ids[:1])                    # (1, N, ...)
        belief = BeliefState(mu=enc.mu[0], sigma=enc.sigma[0], phi=enc.phi[0])
        n = belief.mu.shape[0]
        log_prior = self._attention_log_prior(n, token_ids.device)    # (N, N)
        out = vfe_stack(                                              # converged belief
            belief, belief.mu, belief.sigma, self.group, cfg,
            log_prior=log_prior, block_norm=self.block_norm,
        )

        rho = cfg.prior_handoff_rho                                  # rebuild last-block prior
        rho_s = cfg.prior_handoff_sigma
        mu_p, sigma_p = belief.mu, belief.sigma
        for _ in range(cfg.n_layers - 1):                           # exact iff L==1 or rho==0
            mu_p = (1.0 - rho) * mu_p + rho * out.mu
            sigma_p = (1.0 - rho_s) * sigma_p + rho_s * out.sigma

        omega = _transport(out.phi, self.group)                     # (N, N, K, K)
        mu_t = transport_mean(omega.unsqueeze(0), out.mu.unsqueeze(0))[0]
        sigma_t = transport_covariance(omega.unsqueeze(0), out.sigma.unsqueeze(0))[0]
        energy = pairwise_energy(                                    # (N, N) or (H, N, N)
            out.mu, out.sigma, mu_t, sigma_t,
            alpha=cfg.alpha_div, kl_max=cfg.kl_max, eps=cfg.eps,
            family=cfg.family, divergence_family=cfg.divergence_family,
            irrep_dims=self.group.irrep_dims,
        )
        beta = attention_weights(energy, tau=cfg.tau, log_prior=log_prior)
        self_div = self_divergence_for_alpha(                        # (N,) or (N, K) per-coord
            out.mu, out.sigma, mu_p, sigma_p,
            alpha=cfg.alpha_div, kl_max=cfg.kl_max, eps=cfg.eps,
            family=cfg.family, divergence_family=cfg.divergence_family, alpha_mode=cfg.alpha_mode,
        )
        alpha, _ = self_coupling_alpha(
            self_div, mode=cfg.alpha_mode, value=cfg.alpha, b0=cfg.b0, c0=cfg.c0,
        )

        d = {"attn_entropy": float(metrics.attention_entropy(beta))}
        terms = metrics.free_energy_terms(self_div, energy, beta, alpha, tau=cfg.tau, log_prior=log_prior)
        d.update({k: float(v) for k, v in terms.items()})
        spec = out.sigma if out.sigma.dim() == out.mu.dim() else torch.linalg.eigvalsh(out.sigma)
        d["effective_rank"] = float(metrics.effective_rank(spec).mean())
        # Gauge-geometry probes (diagnostics tier): the curvature proxy -- mean Frobenius departure
        # of the triangle holonomy Omega_ij Omega_jk Omega_ki from I (0 for the flat phi-cocycle) --
        # and the spread of log|det Omega| = tr(embed(phi)) across tokens (0 at phi=0). Pure
        # measurements at the converged transport; off the training graph (no_grad).
        d["holonomy_deviation"] = float(metrics.holonomy_deviation(omega))
        d["gauge_trace_spread"] = float(metrics.gauge_trace_spread(out.phi, self.group.generators))
        return d
