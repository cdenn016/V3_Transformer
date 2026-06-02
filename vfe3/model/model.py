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
    (block_glk). Higher arities are an unsupported registration error.

    ``cfg.cross_couplings`` (off-block GL(K) head coupling) is forwarded as a keyword only
    when set AND the selected builder accepts it (block_glk); otherwise the call is the bare
    positional dispatch, so the default (``cross_couplings=None``) path produces the SAME group
    object as before (byte-identical). Config validation already rejects cross_couplings against
    a builder that does not accept the kwarg, so this is the forwarding seam, not a second guard."""
    builder = get_group(cfg.gauge_group)
    arity = _positional_arity(builder)
    kwargs: dict = {}
    if cfg.cross_couplings is not None and "cross_couplings" in inspect.signature(builder).parameters:
        kwargs["cross_couplings"] = cfg.cross_couplings
    if arity == 1:
        return builder(cfg.embed_dim, **kwargs)
    if arity == 2:
        return builder(cfg.embed_dim, cfg.n_heads, **kwargs)
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
            lambda_h=cfg.lambda_h,
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
        # NEURAL-NETWORK EXCEPTION (sanctioned, default-off): a LEARNED scalar self-coupling alpha.
        # When alpha_mode='learnable', create log_alpha as a trainable nn.Parameter; the consumed
        # coupling is alpha = exp(log_alpha) (always positive). Init 0 -> alpha = exp(0) = 1.0, so a
        # learnable model is byte-identical to the constant alpha=1.0 pure path at step 0. For every
        # other (pure no-NN) alpha_mode the parameter is NOT created at all (no log_alpha attribute),
        # so the default path is param-free.
        # NEURAL-NETWORK EXCEPTION (sanctioned, default-off): the LEARNED bilinear edge connection
        # W for Regime-II (non-flat) transport. When transport_mode='regime_ii', create connection_W
        # as a trainable nn.Parameter of shape (n_gen, K, K); the edge connection is
        # delta_ij^a = cocycle_relaxation * (mu_i^T W^a mu_j) (transport._build_regime_ii). Init ZERO
        # -> delta = 0 -> exp(0) = I -> Omega = exp(phi_i)exp(-phi_j) (the flat cocycle), so a
        # regime_ii model is byte-flat at init. For the default flat (pure no-NN) regime the parameter
        # is NOT created (no connection_W attribute), so the default path is param-free here.
        if cfg.transport_mode == "regime_ii":
            self.connection_W = nn.Parameter(torch.zeros(n_gen, cfg.embed_dim, cfg.embed_dim))
            if cfg.detach_e_step:
                # Footgun (mirrors log_alpha / use_prior_bank): connection_W enters the loss ONLY
                # through the E-step belief updates, but detach_e_step wraps the E-step in no_grad,
                # so connection_W receives NO gradient and stays frozen at its zero init (the flat
                # cocycle). Set detach_e_step=False to train the learned connection.
                import warnings
                warnings.warn(
                    "transport_mode='regime_ii' with detach_e_step=True freezes connection_W: the "
                    "learned edge connection enters the loss only through the E-step, which the "
                    "detached (no_grad) E-step severs, so connection_W.grad is None and the transport "
                    "stays flat. Set detach_e_step=False to train the Regime-II connection.",
                    stacklevel=2,
                )
        if cfg.alpha_mode == "learnable":
            self.log_alpha = nn.Parameter(torch.zeros(()))
            if cfg.detach_e_step:
                # Footgun (mirrors the use_prior_bank+detach warning below): log_alpha enters the
                # loss ONLY through the E-step belief updates, but detach_e_step wraps the whole
                # E-step in no_grad, so log_alpha receives NO gradient and stays frozen at its init
                # (alpha = 1.0). Set detach_e_step=False to train the learned alpha.
                import warnings
                warnings.warn(
                    "alpha_mode='learnable' with detach_e_step=True freezes log_alpha: the learned "
                    "self-coupling alpha enters the loss only through the E-step, which the detached "
                    "(no_grad) E-step severs, so log_alpha.grad is None and alpha stays at its init "
                    "1.0. Set detach_e_step=False to train the learnable alpha.",
                    stacklevel=2,
                )
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

    def _amp_context(
        self,
        device: torch.device,            # resolves autocast device_type ('cuda' | 'cpu')
    ):
        r"""Opt-in autocast context for the E-step (cfg.amp_dtype), else a nullcontext.

        amp_dtype=None (default) returns ``nullcontext()`` so the default path NEVER instantiates
        a ``torch.autocast`` object -- the forward stays byte-identical to the no-AMP build. 'bf16'
        / 'fp16' return ``torch.autocast(device_type=device.type, dtype=...)``; device_type is taken
        from the runtime tensors so a CPU box still exercises the path (a hardcoded 'cuda' autocast
        is inert on CPU tensors)."""
        if self.cfg.amp_dtype is None:
            return nullcontext()
        dtype = torch.bfloat16 if self.cfg.amp_dtype == "bf16" else torch.float16
        return torch.autocast(device_type=device.type, dtype=dtype)

    def _amp_off_context(
        self,
        device: torch.device,            # resolves autocast device_type for the disable wrapper
    ):
        r"""fp32 island for the decode + cross-entropy, else a nullcontext.

        amp_dtype=None (default) returns ``nullcontext()`` -- no autocast object is entered on the
        default path (byte-identity). When AMP is on, returns ``torch.autocast(..., enabled=False)``
        so no FURTHER downcasting happens inside the decode/CE; the actual fp32 guarantee comes from
        ``.float()``-ing the decode/CE inputs at the call site (autocast-disable alone cannot upcast
        a tensor that already arrived bf16)."""
        if self.cfg.amp_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=device.type, enabled=False)

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
        # log_alpha: the learned scalar self-coupling parameter (alpha = exp(log_alpha)) when
        # alpha_mode='learnable', else None (the param-free pure path). Threaded through the
        # E-step so the loss backpropagates to log_alpha. getattr keeps the default path's call
        # identical: None forwards a defaulted-None keyword that every alpha form ignores.
        log_alpha = getattr(self, "log_alpha", None)
        # connection_W: the learned bilinear Regime-II edge connection (a sanctioned NN exception)
        # when transport_mode='regime_ii', else None (the flat pure path). Threaded through the
        # E-step like log_alpha so the loss backpropagates to it; getattr keeps the flat path's call
        # identical (None forwards a defaulted kwarg the flat builder ignores).
        connection_W = getattr(self, "connection_W", None)
        # E-step backward estimator. The EFFECTIVE mode reconciles the legacy detach_e_step bool
        # with e_step_gradient (cfg.effective_e_step_gradient): 'detach' wraps the whole E-step in
        # no_grad (the legacy detach_e_step=True path); 'unroll' (default) and 'straight_through'
        # both run grad-enabled and thread the mode down to e_step_iteration, which only changes the
        # mu/sigma backward (straight_through detaches the per-iteration tangent; unroll keeps the
        # second-order term). Forward VALUE is identical across unroll/straight_through.
        e_step_gradient = self.cfg.effective_e_step_gradient
        run = torch.no_grad() if e_step_gradient == "detach" else nullcontext()
        # Opt-in mixed precision (cfg.amp_dtype): wrap the E-step / belief pipeline in autocast for
        # CUDA throughput. amp_dtype=None (default) -> nullcontext -> NO autocast object is ever
        # instantiated on the default path, so logits/loss are byte-identical to the no-AMP build.
        # device_type is resolved from the runtime tensors (not hardcoded 'cuda') so the path is
        # exercised on whatever device the tokens live on; the matrix_exp / SPD islands inside
        # vfe_stack keep their own autocast(enabled=False) fp32 guards regardless. The decode + CE
        # below are protected separately (their inputs are .float()-ed; see _amp_off_context).
        amp = self._amp_context(token_ids.device)
        with run, amp:
            out = vfe_stack(beliefs, beliefs.mu, beliefs.sigma, self.group, self.cfg,
                            log_prior=log_prior, block_norm=self.block_norm, log_alpha=log_alpha,
                            connection_W=connection_W, e_step_gradient=e_step_gradient)
        mu_final = out.mu                                        # (B, N, K)
        sigma_final = out.sigma

        if self.head_mixer is not None:                          # opt-in head mixing, after E-step / before norm
            mu_final, sigma_final = self.head_mixer(mu_final, sigma_final)

        if self.final_norm is not None:                          # config-selected final norm (cached)
            mu_final = self.final_norm(mu_final, sigma_final)

        # Decode + cross-entropy fp32 island. The decode matmul (_decode_diagonal) reconstructs the
        # Mahalanobis term via a catastrophically-cancelling subtraction pinned at atol-1e-3, and CE
        # is a log-sum-exp over V=50257; both MUST stay fp32 even when amp_dtype is on. The
        # load-bearing guard is the explicit .float() on the inputs (autocast(enabled=False) only
        # blocks FURTHER downcasting -- it does NOT upcast a tensor that already arrived bf16 from
        # the autocast E-step), mirroring retraction.py's in-island sigma.float(). On the default
        # fp32 path .float() is a value-identical no-op AND the island is a nullcontext (see
        # _amp_off_context), so this block is byte-identical to the no-AMP build.
        with self._amp_off_context(token_ids.device):
            logits = self.prior_bank.decode(mu_final.float(), sigma_final.float())   # (B, N, V) fp32
        if targets is None:
            return logits

        with self._amp_off_context(token_ids.device):
            flat_logits = logits.reshape(-1, self.cfg.vocab_size).float()
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
        if self.cfg.mstep_self_coupling_weight > 0.0:
            # M-step self-coupling regularizer (manuscript Algorithm 1, GL(K)_attention.tex:2083):
            # L += alpha_hat * sum_i KL(q_i*||p_i), the mean self-divergence of the CONVERGED belief
            # (out.mu/out.sigma, BEFORE head_mixer/norm) vs the per-block prior. Opt-in, default-off
            # (weight 0 -> byte-identical to the pure path). Grad-connected (no detach), so it
            # backprops to the learned prior tables, like mass_phi. The last-block prior is rebuilt
            # by mirroring vfe_stack's prior_handoff fold; EXACT at n_layers=1 (loop empty -> p =
            # encode belief), an approximation otherwise (one converged belief stands in for the
            # per-block intermediates), matching diagnostics().
            from vfe3.families import get_family
            from vfe3.free_energy import self_divergence_for_alpha
            cfg = self.cfg
            rho, rho_s = cfg.prior_handoff_rho, cfg.prior_handoff_sigma
            mu_p, sigma_p = beliefs.mu, beliefs.sigma
            for _ in range(cfg.n_layers - 1):
                mu_p = (1.0 - rho) * mu_p + rho * out.mu
                sigma_p = (1.0 - rho_s) * sigma_p + rho_s * out.sigma
            fam = get_family(cfg.family)
            sc = self_divergence_for_alpha(
                fam(out.mu, out.sigma), fam(mu_p, sigma_p),
                alpha=cfg.alpha_div, kl_max=cfg.kl_max, eps=cfg.eps,
                divergence_family=cfg.divergence_family, alpha_mode=cfg.alpha_mode,
            ).mean()
            loss = loss + cfg.mstep_self_coupling_weight * sc
        if self.cfg.lambda_h > 0.0:
            # HYPER-PRIOR CHANNEL (manuscript Participatory_it_from_bit.tex eq:pointwise_free_energy,
            # lines 1241-1249): L += lambda_h * mean_i KL(s_i||r), the model-channel beliefs s_i
            # regularized toward the global hyper-prior centroid r. Opt-in, default-off
            # (lambda_h=0 -> byte-identical to the term-absent path). Grad-connected (no detach), so
            # it backprops to the learned s/r tables (the channel trains), and computed from the
            # converged s/r tables OUTSIDE the E-step (s_i does not couple into q this increment).
            # FIRST INCREMENT scope: s_i is encoded fresh here and consumed ONLY by this term; the
            # h->s->p->q coupling, the gamma model-coupling block, and the s-channel E-step update
            # are DEFERRED to increment 2. The covariance kernel is DiagonalGaussian regardless of
            # cfg.family (the s/r tables are always diagonal (V,K)/(K,)); divergence_family is the
            # orthogonal functional seam. r (K,) broadcasts over the (B, N) token axis.
            from vfe3.families.gaussian import DiagonalGaussian
            from vfe3.free_energy import self_divergence
            cfg = self.cfg
            pb = self.prior_bank
            s_mu, s_sigma = pb.encode_s(token_ids)                       # (B, N, K)
            r_mu = pb.r_mu                                               # (K,)
            r_sigma = torch.exp(pb.r_sigma_log).clamp(min=cfg.eps)       # (K,)
            hp = self_divergence(
                DiagonalGaussian(s_mu, s_sigma), DiagonalGaussian(r_mu, r_sigma),
                alpha=cfg.alpha_div, kl_max=cfg.kl_max, eps=cfg.eps,
                divergence_family=cfg.divergence_family,
            ).mean()
            loss = loss + cfg.lambda_h * hp
        return logits, loss, ce.detach()

    @torch.no_grad()
    def generate(
        self,
        token_ids:      torch.Tensor,        # (B, N0) prompt token ids

        max_new_tokens: int,

        *,
        temperature:    float          = 1.0,    # >0; applied to logits before sampling; ignored if greedy
        top_k:          Optional[int]   = None,  # keep the k largest-logit tokens, -inf the rest
        top_p:          Optional[float] = None,  # nucleus: smallest set with softmax cumsum >= p
        greedy:         bool           = False,  # True -> argmax; ignores temperature/top_k/top_p
    ) -> torch.Tensor:                       # (B, N0 + max_new_tokens) prompt followed by generated ids
        r"""Autoregressively extend each prompt by ``max_new_tokens`` tokens.

        Reuses :meth:`forward` (``targets=None`` -> logits ``(B, N, V)``): each step
        feeds the running sequence -- TRUNCATED to the last ``cfg.max_seq_len`` tokens,
        since the model and its attention prior are built for ``N <= max_seq_len`` --
        through ``forward``, reads the last-position logits ``logits[:, -1, :]``, turns
        them into a next token, and appends it. The returned sequence keeps the FULL
        prompt (including any portion beyond ``max_seq_len``) followed by the generated
        ids. Because it only calls ``forward`` and never the training/loss branch, it
        cannot corrupt training (runs under ``torch.no_grad``).

        Greedy (``greedy=True``) takes the argmax and ignores ``temperature``/``top_k``/
        ``top_p``. Otherwise the logits are divided by ``temperature``, then ``top_k``
        (keep the k largest, ``-inf`` the rest), then ``top_p`` (nucleus: smallest set
        whose softmax cumsum reaches ``p``, ``-inf`` the rest, always keeping the top
        token), then softmaxed and sampled with :func:`torch.multinomial`.

        This is the correct-but-slow first version: it re-runs the FULL forward (encode
        -> E-step -> decode) for every generated token. Incremental belief reuse across
        steps is a future optimization.
        """
        seq = token_ids
        for _ in range(max_new_tokens):
            context = seq[:, -self.cfg.max_seq_len:]                 # (B, <=max_seq_len)
            logits = self.forward(context)                          # (B, n, V)
            logits = logits[:, -1, :]                               # (B, V) last position
            if greedy:
                next_token = logits.argmax(dim=-1, keepdim=True)    # (B, 1)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = logits.topk(top_k, dim=-1).values[:, -1:]  # (B, 1) k-th largest
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                if top_p is not None:
                    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
                    cumprobs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                    # Keep the smallest nucleus whose cumprob reaches top_p; the strict
                    # shift always keeps the top token (its cumprob>=p never removes it).
                    remove = cumprobs - sorted_logits.softmax(dim=-1) >= top_p
                    remove_unsorted = remove.scatter(-1, sorted_idx, remove)
                    logits = logits.masked_fill(remove_unsorted, float("-inf"))
                probs = logits.softmax(dim=-1)                      # (B, V)
                next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            seq = torch.cat([seq, next_token], dim=-1)
        return seq

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
        from vfe3.families.base import get_family
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
            log_alpha=getattr(self, "log_alpha", None),               # learned scalar (None on the pure path)
            connection_W=getattr(self, "connection_W", None),         # learned Regime-II connection (None on the flat pure path)
        )

        rho = cfg.prior_handoff_rho                                  # rebuild last-block prior
        rho_s = cfg.prior_handoff_sigma
        mu_p, sigma_p = belief.mu, belief.sigma
        for _ in range(cfg.n_layers - 1):                           # exact iff L==1 or rho==0
            mu_p = (1.0 - rho) * mu_p + rho * out.mu
            sigma_p = (1.0 - rho_s) * sigma_p + rho_s * out.sigma

        # Match the forward's transport regime so holonomy_deviation reads the ACTUAL connection
        # (flat -> ~0; regime_ii with a trained connection_W -> the non-trivial holonomy). regime_ii
        # reads the converged means out.mu and the learned connection_W; flat ignores both.
        omega = _transport(                                          # (N, N, K, K)
            out.phi, self.group, transport_mode=cfg.transport_mode,
            mu=(out.mu if cfg.transport_mode == "regime_ii" else None),
            connection_W=getattr(self, "connection_W", None),
            cocycle_relaxation=cfg.cocycle_relaxation,
        )
        mu_t = transport_mean(omega.unsqueeze(0), out.mu.unsqueeze(0))[0]
        sigma_t = transport_covariance(omega.unsqueeze(0), out.sigma.unsqueeze(0))[0]
        fam = get_family(cfg.family)
        energy = pairwise_energy(                                    # (N, N) or (H, N, N)
            fam(out.mu, out.sigma), fam(mu_t, sigma_t),
            alpha=cfg.alpha_div, kl_max=cfg.kl_max, eps=cfg.eps,
            divergence_family=cfg.divergence_family,
            irrep_dims=self.group.irrep_dims,
        )
        beta = attention_weights(energy, tau=cfg.tau, log_prior=log_prior)
        self_div = self_divergence_for_alpha(                        # (N,) or (N, K) per-coord
            fam(out.mu, out.sigma), fam(mu_p, sigma_p),
            alpha=cfg.alpha_div, kl_max=cfg.kl_max, eps=cfg.eps,
            divergence_family=cfg.divergence_family, alpha_mode=cfg.alpha_mode,
        )
        alpha, _ = self_coupling_alpha(
            self_div, mode=cfg.alpha_mode, value=cfg.alpha, b0=cfg.b0, c0=cfg.c0,
            log_alpha=getattr(self, "log_alpha", None),     # learned scalar (None on the pure path)
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
