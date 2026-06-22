r"""The full VFE_3.0 model: encode -> E-step inference -> decode -> cross-entropy.

No neural layers (no nn.Linear/MLP/activation): on the pure default path the parameters are the
PriorBank's prior tables, plus the model-owned learned tables their toggles create -- the default
pos_phi='learned' positional table, and the default-OFF head mixer, CG coupling, regime_ii
connection, and learnable alpha/lambda/T5-bias scalars. The E-step
is unrolled into the training graph (the differentiable filtering kernel), so the CE
loss backpropagates through inference to the encode/phi priors. Batching loops over
the batch around the (unbatched) E-step; decode and CE are batched.
"""

import inspect
from contextlib import nullcontext
from typing import Callable, Optional, Sequence, Tuple, Dict

import torch
import torch.nn.functional as F
from torch import nn

from vfe3.attention_prior import attention_log_prior
from vfe3.belief import BeliefState
from vfe3.config import VFE3Config
from vfe3.geometry.groups import GaugeGroup, get_group
from vfe3.geometry.norms import get_norm
from vfe3.geometry.rope import get_pos_rotation
from vfe3.geometry.transport import RopeTransport
from vfe3.model.head_mixer import HeadMixer
from vfe3.model.block import _as_coeff, vfe_block
from vfe3.model.positional_phi import apply_positional_phi
from vfe3.model.prior_bank import PriorBank
from vfe3.model.stack import vfe_stack


# Transport-mode membership sets (Fix B): centralize the string literals so a new regime
# registers here and all comparisons update automatically.
_REGIME_NEEDS_MU    = frozenset({"regime_ii", "regime_ii_covariant"})   # regimes that pass mu to _transport
_REGIME_NEEDS_SIGMA = frozenset({"regime_ii_covariant"})                # regimes that pass sigma to _transport


def _precision_key_bias(
    sigma:      torch.Tensor,        # (B, N, K) per-key belief variances

    *,
    b0:         float = 1.0,
    irrep_dims: 'Optional[Sequence[int]]' = None,   # per-head block sizes (sum == K); None -> global trace
) -> torch.Tensor:                   # (B, N) global, or (B, N, H) per-head, log-reliability
    r"""Per-key reliability bias for precision-weighted attention: ``-log(b0 + tr Sigma_j)``.

    A more-uncertain key (larger variance ``tr Sigma_j``) gets a MORE NEGATIVE additive bias, so it
    is down-weighted in the attention softmax. Folded into the attention ``log_prior`` by the caller
    (detached there, so the closed-form belief kernel treats it as a fixed prior). A uniform-over-keys
    ``Sigma`` gives a constant-in-key bias that the softmax absorbs (no effect); only key-to-key
    variance in ``Sigma`` changes attention.

    ``irrep_dims=None`` (default): the GLOBAL trace over all K coordinates -> ``(B, N)``. When the
    per-head gauge-block sizes are given, the trace is taken PER BLOCK (head) -> ``(B, N, H)``, so
    each head down-weights keys by its OWN block uncertainty.
    """
    if irrep_dims is None:
        return -torch.log(b0 + sigma.sum(dim=-1))                          # (B, N) global trace
    tr = torch.stack([blk.sum(dim=-1)                                      # (B, N, H) per-block traces
                      for blk in sigma.split(list(irrep_dims), dim=-1)], dim=-1)
    return -torch.log(b0 + tr)


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
    a builder that does not accept the kwarg, so this is the forwarding seam, not a second guard.
    ``cfg.group_n`` / ``cfg.irrep_spec`` are forwarded the same way (the so_n/sp_n irrep-tower
    builders; config validation rejects them for any other group)."""
    builder = get_group(cfg.gauge_group)
    arity = _positional_arity(builder)
    params = inspect.signature(builder).parameters
    kwargs: dict = {}
    if cfg.cross_couplings is not None and "cross_couplings" in params:
        kwargs["cross_couplings"] = cfg.cross_couplings
    if cfg.group_n is not None and "group_n" in params:
        kwargs["group_n"] = cfg.group_n
    if cfg.irrep_spec is not None and "irrep_spec" in params:
        kwargs["irrep_spec"] = cfg.irrep_spec
    # close_basis: AUTO (None) defaults to closing the basis under the Lie bracket exactly when a
    # cross_couplings chain is present (so the exponentiated off-block group is a well-defined
    # subalgebra of gl(K)); explicit True/False overrides. Forwarded only when the builder accepts
    # it. On the DEFAULT path (cross_couplings=None) close resolves to False and the builder's own
    # default is False, so the group object is byte-identical to before.
    close = cfg.close_basis if cfg.close_basis is not None else (cfg.cross_couplings is not None)
    if "close_basis" in params:
        kwargs["close_basis"] = close
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
        # ALiBi-family priors carry a per-head (n_heads, N, N) axis, while the energy's head axis
        # is len(irrep_dims); a mismatch right-aligns the prior's head axis against the BATCH axis
        # of a single-block (B, N, N) energy -- silent corruption at B=1, RuntimeError otherwise
        # (audit 2026-06-09 P1). Reject at construction. Single-block groups may run alibi with
        # n_heads=1 (the (1, N, N) prior is squeezed to the (N, N) convention in
        # _attention_log_prior below).
        for _pname in ("beta_attention_prior", "gamma_attention_prior"):
            if (getattr(cfg, _pname) in ("alibi", "causal_alibi")
                    and cfg.n_heads != len(self.group.irrep_dims)):
                raise ValueError(
                    f"{_pname}={getattr(cfg, _pname)!r} builds an (n_heads, N, N) prior but the "
                    f"energy head axis is {len(self.group.irrep_dims)} "
                    f"(irrep_dims={self.group.irrep_dims}); set "
                    f"n_heads={len(self.group.irrep_dims)} or use a headless prior "
                    f"(uniform/causal/windowed/...)."
                )
        n_gen = self.group.generators.shape[0]
        self.prior_bank = PriorBank(
            cfg.vocab_size, cfg.embed_dim, n_gen,
            mu_init_std=cfg.mu_init_std, sigma_init=cfg.sigma_init, phi_scale=cfg.phi_scale,
            decode_tau=cfg.decode_tau, eps=cfg.eps,
            diagonal_covariance=cfg.diagonal_covariance,
            use_prior_bank=cfg.use_prior_bank, decode_bias=cfg.decode_bias,
            decode_precision_scaled=cfg.decode_precision_scaled,
            encode_mode=cfg.encode_mode, decode_mode=cfg.decode_mode,
            decode_chunk_size=cfg.decode_chunk_size,
            lambda_h=cfg.lambda_h, lambda_gamma=cfg.lambda_gamma,
            prior_source=cfg.prior_source, s_e_step=cfg.s_e_step,
            # r is a GRADIENT leaf only under r_update_mode='gradient'; under 'barycenter' it is
            # set in-place each M-step by the closed-form barycenter (PriorBank.barycenter_r_,
            # driven from train_step) and so must stay ungrouped/requires_grad=False.
            learnable_r=cfg.learnable_r and cfg.r_update_mode == "gradient",
        )
        # Stateless norm instances built ONCE (audit 2d/4f): they are parameter-free pure
        # maps (K, eps), so re-instantiating them per block/forward only churned objects.
        self.block_norm = get_norm(cfg.norm_type_block)(cfg.embed_dim, eps=cfg.eps) \
            if cfg.norm_type_block != "none" else None
        self.final_norm = get_norm(cfg.norm_type_final)(cfg.embed_dim, eps=cfg.eps) \
            if cfg.norm_type_final != "none" else None
        # Opt-in Schur-commutant head mixer (default off). Built ONCE from the gauge group's
        # irrep blocks. Label-less groups need >= 2 EQUAL blocks (block_glk/tied_block_glk);
        # labeled irrep towers (so_n/sp_n) mix per isotypic component (mults-one towers get
        # per-head scalar gains -- the entire linear commutant there). Bad pairings fail here,
        # not at forward.
        self.head_mixer = HeadMixer(self.group.irrep_dims,
                                    irrep_labels=self.group.irrep_labels) \
            if cfg.use_head_mixer else None
        # Opt-in CG cross-type coupling (default off; so_n/sp_n only). Built ONCE from the
        # group's labels; CGCoupling raises at construction when no admissible paths exist.
        # The algebra key comes from the GROUP OBJECT (set by the so_n/sp_n builders), not a
        # re-derivation from the config string, so a newly registered labeled-tower group
        # cannot mis-dispatch here (audit 2026-06-09 overnight RF1).
        if cfg.use_cg_coupling:
            from vfe3.model.cg_coupling import CGCoupling
            self.cg_coupling = CGCoupling(
                cfg.group_n, self.group.algebra,
                self.group.irrep_dims, self.group.irrep_labels)
        else:
            self.cg_coupling = None
        if (cfg.use_head_mixer or cfg.use_cg_coupling) \
                and cfg.effective_e_step_gradient == "detach":
            # Footgun (mirrors connection_W / log_alpha / pos_phi_free above and below): the
            # mixer and the CG coupling are applied INSIDE the vfe_stack call, which the
            # 'detach' estimator wraps wholesale in no_grad (block.py:73-78 under
            # model.forward's `run`), so mixer_deltas / path_weights build no graph, receive
            # no gradient, and silently stay frozen at their identity/zero init -- the model
            # trains its other parameters and LOOKS healthy while these two opt-in components
            # never adapt (audit 2026-06-09 overnight F31, challenge-upheld). Gate on the
            # EFFECTIVE estimator so both the detach_e_step bool and the
            # e_step_gradient='detach' string route warn; 'unroll' and 'straight_through'
            # run the stack grad-enabled and train them.
            import warnings
            warnings.warn(
                "use_head_mixer/use_cg_coupling with the effective E-step estimator 'detach' "
                "freezes mixer_deltas/path_weights: both modules are applied inside the "
                "no_grad-wrapped vfe_stack, so they receive NO gradient and stay at their "
                "identity/zero init. Use an 'unroll' E-step (detach_e_step=False, "
                "e_step_gradient='unroll') or 'straight_through' to train them.",
                stacklevel=2,
            )
        # NEURAL-NETWORK EXCEPTION (sanctioned, default-off): a LEARNED scalar self-coupling alpha.
        # When lambda_alpha_mode='learnable', create log_alpha as a trainable nn.Parameter; the consumed
        # coupling is alpha = exp(log_alpha) (always positive). Init 0 -> alpha = exp(0) = 1.0, so a
        # learnable model is byte-identical to the constant alpha=1.0 pure path at step 0. For every
        # other (pure no-NN) lambda_alpha_mode the parameter is NOT created at all (no log_alpha attribute),
        # so the default path is param-free.
        # NEURAL-NETWORK EXCEPTION (sanctioned, default-off): the LEARNED bilinear edge connection
        # W for Regime-II (non-flat) transport. When transport_mode='regime_ii', create connection_W
        # as a trainable nn.Parameter of shape (n_gen, K, K); the edge connection is
        # delta_ij^a = cocycle_relaxation * (mu_i^T W^a mu_j) (transport._build_regime_ii). Init ZERO
        # -> delta = 0 -> exp(0) = I -> Omega = exp(phi_i)exp(-phi_j) (the flat cocycle), so a
        # regime_ii model is flat at init to fp32 tolerance (atol 1e-6 pinned; NOT bit-exact -- the
        # zero-tensor W takes the generic einsum path to keep d Omega/d W alive; audit 2026-06-10
        # F11). For the default flat (pure no-NN) regime the parameter is NOT created (no
        # connection_W attribute), so the default path is param-free here.
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
        # NEURAL-NETWORK EXCEPTION (sanctioned, default-off): the LEARNED gauge-COVARIANT (Route B)
        # Regime-II connection M. delta_ij^a = cocycle_relaxation * sum_f M^a_f I^f_ij, with I^f the
        # GAUGE-INVARIANT (Mahalanobis, trace, log-det) features of the (query, transported-key)
        # belief pair (transport._build_regime_ii_covariant). Unlike connection_W (gauge-invariant
        # ONLY at W=0), the transport stays gauge-covariant (Omega_ij -> g_i Omega_ij g_j^{-1}) for
        # ANY M. Shape (n_gen, 3); init ZERO -> delta=0 -> flat cocycle at init (fp32; the generic
        # path keeps d Omega/d M alive). NOT created on the flat / regime_ii paths (param-free).
        if cfg.transport_mode == "regime_ii_covariant":
            self.connection_M = nn.Parameter(torch.zeros(n_gen, 3))
            if cfg.detach_e_step:
                import warnings
                warnings.warn(
                    "transport_mode='regime_ii_covariant' with detach_e_step=True freezes connection_M: "
                    "the learned edge connection enters the loss only through the E-step, which the "
                    "detached (no_grad) E-step severs, so connection_M.grad is None and the transport "
                    "stays flat. Set detach_e_step=False to train the Route-B connection.",
                    stacklevel=2,
                )
        if cfg.lambda_alpha_mode == "learnable":
            self.log_alpha = nn.Parameter(torch.zeros(()))
            if cfg.detach_e_step:
                # Footgun (mirrors the use_prior_bank+detach warning below): log_alpha enters the
                # loss ONLY through the E-step belief updates, but detach_e_step wraps the whole
                # E-step in no_grad, so log_alpha receives NO gradient and stays frozen at its init
                # (alpha = 1.0). Set detach_e_step=False to train the learned alpha.
                import warnings
                warnings.warn(
                    "lambda_alpha_mode='learnable' with detach_e_step=True freezes log_alpha: the learned "
                    "self-coupling alpha enters the loss only through the E-step, which the detached "
                    "(no_grad) E-step severs, so log_alpha.grad is None and alpha stays at its init "
                    "1.0. Set detach_e_step=False to train the learnable alpha.",
                    stacklevel=2,
                )
        # NEURAL-NETWORK EXCEPTION (sanctioned, default-off): a LEARNED belief-coupling weight
        # lambda_beta. When learnable_lambda_beta=True, create
        # log_lambda_beta as a scalar nn.Parameter; the consumed weight is lambda_beta =
        # exp(log_lambda_beta) (always positive). Init 0 -> lambda_beta = exp(0) = 1.0, byte-identical
        # to the constant-1.0 pure path at step 0. For learnable_lambda_beta=False the parameter is
        # NOT created (no log_lambda_beta attribute), so the default path is param-free.
        if cfg.learnable_lambda_beta:
            self.log_lambda_beta = nn.Parameter(torch.zeros(()))
            if cfg.detach_e_step:
                # Footgun (mirrors log_alpha): log_lambda_beta enters the loss ONLY through the E-step
                # belief updates, but detach_e_step wraps the whole E-step in no_grad, so it receives
                # NO gradient and stays frozen at its init (lambda_beta = 1.0).
                import warnings
                warnings.warn(
                    "learnable_lambda_beta=True with detach_e_step=True freezes log_lambda_beta: the "
                    "learned belief-coupling weight enters the loss only through the E-step, which the "
                    "detached (no_grad) E-step severs, so log_lambda_beta.grad is None and lambda_beta "
                    "stays at its init 1.0. Set detach_e_step=False to train the learnable lambda_beta.",
                    stacklevel=2,
                )
        # NEURAL-NETWORK EXCEPTION (sanctioned, default-off): a LEARNED hyper-prior weight lambda_h
        # (the model-fiber analogue of lambda_alpha_mode='learnable'). When lambda_h_mode='learnable', create
        # log_lambda_h as a scalar nn.Parameter; the consumed weight is lambda_h = exp(log_lambda_h)
        # (always positive). Init log(cfg.lambda_h) -> lambda_h = cfg.lambda_h, byte-identical to the
        # constant lambda_h at step 0 (unlike log_alpha/log_lambda_beta, whose constant default is 1.0
        # so they init at 0). For every other lambda_h_mode the parameter is NOT created, so those paths
        # are param-free. Trains through the scored forward term (s_e_step=False) or the unrolled s
        # E-step (_refine_s, s_e_step=True) -- the latter is the E-step-tangent route the detach/oracle
        # config warnings cover.
        # Create log_lambda_h only on the active-channel path (lambda_h>0 or s_e_step), matching the
        # r-table gate (prior_bank.py): with lambda_h=0 and not s_e_step the hyper-prior channel is
        # inert (no r table, scored term gated off) and config warns, so creating the parameter would
        # only orphan it. getattr(...,"log_lambda_h",None) at every consumer keeps this None-safe.
        if cfg.lambda_h_mode == "learnable" and (cfg.lambda_h > 0.0 or cfg.s_e_step):
            self.log_lambda_h = nn.Parameter(torch.tensor(max(float(cfg.lambda_h), cfg.eps)).log())
            if cfg.detach_e_step and cfg.s_e_step:
                # Footgun (mirrors log_alpha / log_lambda_beta): under s_e_step the learned lambda_h
                # enters the loss ONLY through the s E-step (_refine_s), which detach_e_step wraps in
                # no_grad, so log_lambda_h receives NO gradient and stays frozen at its init. This guard
                # deliberately keys on the LEGACY detach_e_step bool; the e_step_gradient='detach' /
                # 'straight_through' route (with detach_e_step=False) is the complementary CONFIG-level
                # warning's job (config.py, keyed on the e_step_gradient literal), so the two cover both
                # freeze routes with no double-warn -- do NOT broaden this to effective_e_step_gradient.
                import warnings
                warnings.warn(
                    "lambda_h_mode='learnable' with detach_e_step=True and s_e_step=True freezes "
                    "log_lambda_h: under s_e_step the learned hyper-prior weight enters the loss only "
                    "through the s E-step, which the detached (no_grad) E-step severs, so "
                    "log_lambda_h.grad is None and lambda_h stays at its init. Set detach_e_step=False "
                    "to train it.",
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
        # Gauge-RoPE rotation R(theta) is similarly loop-invariant for fixed (N, device, dtype);
        # cache keyed on (N, device, dtype) so it is built once. None on the 'none' path.
        self._rope_cache: dict = {}
        # BCH positional encoding (default-off): a learned per-position Lie-algebra element table
        # composed into the gauge frame before transport. Created ONLY for pos_phi='learned' (a raw
        # nn.Parameter like log_alpha/connection_W, not a network); the "none"/"frozen" paths add no
        # parameter, so the pure path stays param-free. Init scaled by pos_phi_scale.
        if cfg.pos_phi == "learned":
            # Seed pos_phi_free from a DEDICATED generator (cfg.seed), independent of the global RNG
            # stream position: its init no longer depends on whether the conditional model-channel
            # s/r prior tables were drawn, so pos_phi_free is byte-identical across token vs
            # model_channel models (the s-channel byte-identity oracles then hold under the learned
            # default). A separate generator does NOT clobber the caller's global RNG, so the
            # reproducibility seam stays at run_training; mu/sigma/phi inits are unchanged.
            _g = torch.Generator().manual_seed(int(cfg.seed))
            self.pos_phi_free = nn.Parameter(
                torch.randn(cfg.max_seq_len, n_gen, generator=_g) * cfg.pos_phi_scale)
            if cfg.effective_e_step_gradient in ("detach", "straight_through"):
                # Footgun (mirrors log_alpha / connection_W): pos_phi_free enters the loss ONLY
                # through the E-step belief transport. The 'detach' and 'straight_through' estimators
                # both sever that path (no_grad / a detached tangent), so the positional table
                # receives no gradient and stays frozen at init. Gate on the EFFECTIVE estimator
                # (cfg.effective_e_step_gradient reconciles the legacy detach_e_step bool with the
                # string e_step_gradient route) so the string-estimator paths warn too. Set the
                # effective E-step estimator to 'unroll' (detach_e_step=False) to learn it.
                import warnings
                warnings.warn(
                    "pos_phi='learned' with the effective E-step estimator "
                    f"{cfg.effective_e_step_gradient!r} freezes pos_phi_free: the positional gauge "
                    "element enters the loss only through the E-step transport, which the "
                    "detached / straight-through E-step severs. Use an 'unroll' E-step "
                    "(detach_e_step=False, e_step_gradient='unroll') to train it.",
                    stacklevel=2,
                )
        # NEURAL-NETWORK EXCEPTION (sanctioned, default-off): a LEARNED T5 relative-position
        # attention-bias table. When t5_learnable_bias=True, create t5_bias as a trainable
        # nn.Parameter of shape (t5_num_buckets,); the attention log-prior reads it as the per-bucket
        # bias b_{i-j} (manuscript GL(K)_attention.tex:826-838: pi_j ∝ exp(b_{i-j}), beta_ij =
        # softmax_j(-E_ij/tau + b_{i-j}), the first-principles T5 derivation). Init to the fixed-table
        # default -log1p(bucket) so a learnable model is byte-identical to the fixed t5_relative_bias
        # prior at step 0, then trains. Unlike the gauge/value exceptions this bias is a scalar
        # function of position OFFSET only and touches NO gauge transport, so it does NOT break gauge
        # equivariance. Created ONLY when t5_relative_bias is an active channel (mirrors the log_lambda_h
        # active-channel gate so the parameter is never orphaned); else no t5_bias attribute and the
        # pure path stays param-free (the fixed-table default still runs).
        if cfg.t5_learnable_bias and "t5_relative_bias" in (cfg.beta_attention_prior, cfg.gamma_attention_prior):
            self.t5_bias = nn.Parameter(-torch.log1p(torch.arange(cfg.t5_num_buckets, dtype=torch.float32)))
            if cfg.effective_e_step_gradient in ("detach", "straight_through"):
                # Footgun (mirrors log_alpha / connection_W / pos_phi_free, warned at config.py:1267):
                # the attention log-prior is consumed INSIDE the E-step, and BOTH severing estimators
                # cut t5_bias's only gradient path -- 'detach' wraps the whole E-step in no_grad, and
                # 'straight_through' detaches the per-iteration belief tangent (e_step.py), the sole
                # carrier of the attention-prior signal into the belief. Either way t5_bias.grad is
                # None and the bias stays frozen at its fixed-table init; only the 'unroll' (default)
                # E-step trains it.
                import warnings
                warnings.warn(
                    f"t5_learnable_bias=True with the effective E-step estimator "
                    f"{cfg.effective_e_step_gradient!r} freezes t5_bias: the T5 relative-position bias "
                    f"enters the loss only through the attention log-prior consumed inside the E-step, "
                    f"whose gradient both 'detach' (no_grad) and 'straight_through' (detached tangent) "
                    f"sever, so t5_bias.grad is None and the bias stays at its fixed-table init. Use an "
                    f"'unroll' E-step (detach_e_step=False, e_step_gradient='unroll') to train it.",
                    stacklevel=2,
                )
        elif cfg.t5_learnable_bias:
            # t5_learnable_bias=True but no 't5_relative_bias' channel is active (neither beta nor
            # gamma attention prior is 't5_relative_bias'): no t5_bias parameter is created and the
            # toggle is silently inert. Warn so the dead toggle is not mistaken for a trained bias.
            import warnings
            warnings.warn(
                "t5_learnable_bias=True but no 't5_relative_bias' attention channel is active "
                "(beta_attention_prior / gamma_attention_prior): no learnable t5_bias is created and "
                "the toggle is inert. Set beta_attention_prior or gamma_attention_prior to "
                "'t5_relative_bias' to use the learnable bias.",
                UserWarning, stacklevel=2,
            )

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
        self._rope_cache.clear()             # device/dtype moved: cached rotations are now stale
        return self

    def _attention_log_prior(
        self,
        n:      int,                          # sequence length N (varies across calls)
        device: torch.device,

        *,
        prior:  Optional[str] = None,         # prior-registry name; None -> cfg.beta_attention_prior (belief block)
    ) -> torch.Tensor:
        r"""Loop-invariant attention log-prior, cached on (name, N, device, dtype) (audit 4e).

        The dtype is taken from the prior-bank mean table so the mask matches the belief
        dtype after a ``.to(torch.float64)`` move (audit 2f: the old call omitted dtype). ``prior``
        lets the gamma model-coupling block reuse the same cache under its own attention prior."""
        name = prior if prior is not None else self.cfg.beta_attention_prior
        dtype = self.prior_bank.mu_embed.dtype
        # Learnable T5 bias: the per-bucket table is a live nn.Parameter that changes every step, so
        # the (name, N, ...) cache MUST be bypassed -- a cached tensor would serve a stale table and
        # sever the gradient. Build fresh each call, passing the parameter as bias_values so the loss
        # backpropagates to t5_bias (through the E-step). getattr keeps this None-safe when the param
        # was not created (t5_learnable_bias off, or no active t5 channel); .to preserves the graph.
        if name == "t5_relative_bias" and getattr(self, "t5_bias", None) is not None:
            out = attention_log_prior(
                name, n, n,
                device=device, dtype=dtype,
                n_heads=self.cfg.n_heads, alibi_slope=self.cfg.alibi_slope,
                window=self.cfg.attention_window,
                num_buckets=self.cfg.t5_num_buckets, max_distance=self.cfg.t5_max_distance,
                bias_values=self.t5_bias.to(device=device, dtype=dtype))
            if out.dim() == 3 and len(self.group.irrep_dims) == 1:
                out = out.squeeze(0)
            return out
        # The key carries every cfg field a builder consumes (n_heads, alibi_slope, window, T5
        # bucketing) so a post-construction cfg mutation cannot serve a stale prior (audit PP4/P9).
        key = (name, n, device, dtype, self.cfg.n_heads, self.cfg.alibi_slope,
               self.cfg.attention_window, self.cfg.t5_num_buckets, self.cfg.t5_max_distance)
        cached = self._log_prior_cache.get(key)
        if cached is None:
            cached = attention_log_prior(
                name, n, n,
                device=device, dtype=dtype,
                n_heads=self.cfg.n_heads, alibi_slope=self.cfg.alibi_slope,
                window=self.cfg.attention_window,
                num_buckets=self.cfg.t5_num_buckets, max_distance=self.cfg.t5_max_distance,
            )
            # A (1, N, N) per-head prior on a single-block group collapses to the (N, N)
            # single-block convention (the energy carries no head axis there); the construction
            # guard above pins n_heads == 1 for that case, so shape[0] == 1 here.
            if cached.dim() == 3 and len(self.group.irrep_dims) == 1:
                cached = cached.squeeze(0)
            self._log_prior_cache[key] = cached
        return cached

    def _rope_rotation(
        self,
        n:      int,                          # sequence length N (varies across calls)
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        r"""Cached gauge-RoPE rotation R(theta) for length n (None when pos_rotation='none')."""
        if self.cfg.pos_rotation == "none":
            return None
        dtype = self.prior_bank.mu_embed.dtype
        key = (n, device, dtype)
        cached = self._rope_cache.get(key)
        if cached is None:
            cached = get_pos_rotation(self.cfg.pos_rotation)(
                torch.arange(n, device=device), self.group.irrep_dims,
                base=self.cfg.rope_base, device=device, dtype=dtype)
            self._rope_cache[key] = cached
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
        # Explicit, defensive mapping (no bare-else silent-fp16 fallthrough). config.py
        # (_require) accepts amp_dtype in (None, 'bf16', 'fp16'), so both 'bf16' and 'fp16' are
        # reachable non-None values; fp16 training is loss-scaled by the GradScaler in train.py
        # (enabled when amp_dtype=='fp16'). Map both and raise on anything else.
        if self.cfg.amp_dtype == "bf16":
            dtype = torch.bfloat16
        elif self.cfg.amp_dtype == "fp16":
            dtype = torch.float16
        else:
            raise ValueError(f"unsupported amp_dtype {self.cfg.amp_dtype!r}")
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

    def _apply_pos_phi(self, phi: torch.Tensor) -> torch.Tensor:
        r"""Compose the configured BCH positional element into the gauge frame (no-op for 'none')."""
        if self.cfg.pos_phi == "none":
            return phi
        return apply_positional_phi(
            phi, self.group,
            mode=self.cfg.pos_phi, compose_mode=self.cfg.pos_phi_compose,
            order=self.cfg.bch_pe_order, scale=self.cfg.pos_phi_scale,
            project_slk=self.cfg.pos_phi_project_slk,
            pos_phi_free=getattr(self, "pos_phi_free", None),
        )

    def _refine_s(
        self,
        token_ids:       torch.Tensor,   # (B, N) integer token ids
        phi0:            torch.Tensor,   # (B, N, n_gen) encoded gauge frame (shared, held FIXED)

        *,
        e_step_gradient: str = "unroll",
    ) -> 'tuple[torch.Tensor, torch.Tensor]':
        r"""Refine the model channel s by its own E-step toward the frozen hyper-prior r plus the
        gamma model-consensus, with the shared gauge frame phi0 held fixed (e_phi_lr=0). Returns the
        refined (mu_s, sigma_s); the s-tables train through the unrolled trajectory."""
        from vfe3.belief import BeliefState
        from vfe3.inference.e_step import e_step
        from vfe3.free_energy import attention_tau

        cfg, pb, grp = self.cfg, self.prior_bank, self.group
        s_mu, s_sigma = pb.encode_s(token_ids)                         # (B, N, K)
        r_mu    = pb.r_mu.expand_as(s_mu)                              # (B, N, K) frozen r broadcast
        r_sigma = torch.exp(pb.r_sigma_log).clamp(min=cfg.eps).expand_as(s_sigma)
        gamma_tau       = attention_tau(_as_coeff(cfg.kappa_gamma, s_mu.device), grp.irrep_dims)
        gamma_log_prior = self._attention_log_prior(
            token_ids.shape[1], token_ids.device, prior=cfg.gamma_attention_prior,
        )
        out = e_step(
            BeliefState(mu=s_mu, sigma=s_sigma, phi=phi0), r_mu, r_sigma, grp,
            n_iter=cfg.n_e_steps,         tau=gamma_tau,
            e_q_mu_lr=cfg.e_s_mu_lr,      e_q_sigma_lr=cfg.e_s_sigma_lr, e_phi_lr=0.0,
            # The s-channel self-coupling weight IS lambda_h (the hyper-prior precision): route it
            # through the lambda_h_mode registry, not a hardcoded constant. e_step's self_coupling_alpha
            # consumes (value, lambda_alpha_mode, b0, c0, log_alpha) exactly as lambda_h_i.hyper_prior_lambda_h
            # does. ENVELOPE CANCELLATION (audit 2026-06-13): under state_dependent the s E-step gets the
            # correct lam*(KL)*dKL gradient by the envelope theorem -- on the LIVE kernel route the
            # belief-gradient kernel multiplies dKL by the envelope COEFFICIENT alpha*=c0_h/(b0_h+KL) and
            # never literally adds R_h (R_h's d/dbelief is 0, so omitting it is exact); only the autograd
            # ORACLE route's free_energy_value materializes alpha_reg=R_h. Either way the descent
            # direction is correct. learnable feeds log_lambda_h. b0_h/c0_h are the hyper-prior's own
            # precision shape (NOT alpha's b0/c0). NOTE: under state_dependent, value=cfg.lambda_h is
            # IGNORED (alpha_state_dependent reads only b0_h/c0_h); the coupling magnitude is c0_h/(b0_h+KL),
            # and cfg.lambda_h then acts ONLY as the channel-on gate -- it does not scale the s coupling.
            renyi_order=cfg.renyi_order,   value=cfg.lambda_h,          lambda_alpha_mode=cfg.lambda_h_mode,
            b0=_as_coeff(cfg.b0_h, s_mu.device), c0=_as_coeff(cfg.c0_h, s_mu.device),
            log_alpha=getattr(self, "log_lambda_h", None),
            lambda_beta=cfg.lambda_gamma,
            kl_max=cfg.kl_max,             eps=cfg.eps,
            sigma_max=cfg.sigma_max,       e_sigma_q_trust=cfg.e_sigma_q_trust,
            e_mu_q_trust=cfg.e_mu_q_trust, mu_trust_mode=cfg.mu_trust_mode,
            include_attention_entropy=cfg.include_attention_entropy,
            gradient_mode=cfg.gradient_mode,
            family="gaussian_diagonal",
            divergence_family=cfg.divergence_family,
            phi_precond_mode=cfg.phi_precond_mode,
            phi_retract_mode=cfg.phi_retract_mode,
            spd_retract_mode=cfg.spd_retract_mode,
            # TIED FLAT transport for the s-channel, INTENTIONALLY ignoring cfg.transport_mode
            # (audit 2026-06-10 F7, mirroring _gamma_coupling_term's documented choice): the model
            # channel refines under the flat phi0 cocycle even when the belief channel runs
            # regime_ii -- the learned edge connection is a belief-channel object (delta reads the
            # belief means, not s), and the s E-step runs with phi held fixed. Thread
            # cfg.transport_mode + connection_W here if the s-channel is ever meant to share the
            # learned connection. RoPE is LIKEWISE not forwarded (no rope/rope_on_cov/rope_on_value):
            # the s-channel E-step is position-coupled only through gamma attention, never
            # RoPE-transported, even when the belief channel runs pos_rotation='rope' (the belief
            # E-step gets rope at model.forward; both s_e_step and rope default OFF, so this is an
            # inconsistency only under the double opt-in). Thread the rope args here if the model
            # channel is ever meant to be RoPE-transported too.
            transport_mode="flat",
            e_step_gradient=e_step_gradient,
            oracle_unroll_grad=cfg.oracle_unroll_grad,
            log_prior=gamma_log_prior,
        )
        return out.mu, out.sigma

    @torch.no_grad()
    def _refined_s_belief(
        self,
        token_ids: torch.Tensor,             # (B, N) integer token ids
    ) -> 'Optional[tuple[torch.Tensor, torch.Tensor]]':   # refined (mu_s, sigma_s) for seq 0, or None
        r"""The refined model-channel belief s1 = (mu_s, sigma_s) the forward uses under ``s_e_step``
        (sequence 0), or ``None`` when ``s_e_step`` is off so callers fall back to the raw s tables.

        M2: lets the model-channel diagnostics and figures (hyper-prior KL, gamma energy / attention)
        read the SAME refined s the forward computes (model.py forward s_e_step branch), instead of
        re-encoding the un-refined tables. Mirrors the forward refine (``_refine_s`` with the
        encoded+positional gauge frame held fixed); no_grad, off the hot path.
        """
        if not self.cfg.s_e_step:
            return None
        enc = self.prior_bank.encode(token_ids[:1])
        phi0 = self._apply_pos_phi(enc.phi[0]).unsqueeze(0)          # (1, N, n_gen) encoded frame, fixed
        return self._refine_s(token_ids[:1], phi0)                   # (1, N, K) x2

    def forward(
        self,
        token_ids: torch.Tensor,         # (B, N) integer token ids
        targets:   Optional[torch.Tensor] = None,   # (B, N) next-token ids (-100 = ignore)

        *,
        estep_grad_out: Optional[dict] = None,   # diag out-param: filled with the E-step belief-grad norms
    ) -> 'torch.Tensor | Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]':
        r"""Forward pass; returns logits, or (logits, loss, ce) when targets are given.

        On the fused-chunked training path logits is None (callers discard it there), hence the
        Optional first element of the training tuple. When ``estep_grad_out`` (a dict) is passed, it
        is filled with the LAST-block / LAST-iteration raw E-step belief-gradient norms
        ``{'mu','sigma','phi'}`` (||grad_mu/sigma/phi|| of F over the belief tuple) -- the E-step
        analogue of the M-step per-group grad norms; default None is zero-overhead and byte-identical."""
        B, N = token_ids.shape
        beliefs = self.prior_bank.encode(token_ids)              # (B, N, K) ...
        beliefs = beliefs._replace(phi=self._apply_pos_phi(beliefs.phi))
        log_prior = self._attention_log_prior(N, token_ids.device)
        rope = self._rope_rotation(N, token_ids.device)

        # The E-step stack is vectorized over the batch (audit 4c): the belief tuple carries a
        # leading B axis through transport / gradients / retraction in one set of kernels, instead
        # of a serial per-sequence Python loop. Sequences are independent (each reads only its own
        # belief and the shared, sequence-independent log_prior), so the batched result equals the
        # per-sample result (pinned by tests/test_perf_equivalence.py::test_batched_forward_equals_per_sample).
        # log_alpha: the learned scalar self-coupling parameter (alpha = exp(log_alpha)) when
        # lambda_alpha_mode='learnable', else None (the param-free pure path). Threaded through the
        # E-step so the loss backpropagates to log_alpha. getattr keeps the default path's call
        # identical: None forwards a defaulted-None keyword that every alpha form ignores.
        log_alpha = getattr(self, "log_alpha", None)
        # lambda_beta: the belief-coupling weight. The learned exp(log_lambda_beta) when
        # learnable_lambda_beta=True (a live tensor, so the loss backpropagates to log_lambda_beta
        # through the unrolled E-step, exactly like log_alpha), else the constant cfg.lambda_beta.
        # A single value is threaded (not the raw param), so consumers just multiply by it.
        _llb = getattr(self, "log_lambda_beta", None)
        lambda_beta = self.cfg.lambda_beta if _llb is None else _llb.exp()
        # connection_W: the learned bilinear Regime-II edge connection (a sanctioned NN exception)
        # when transport_mode='regime_ii', else None (the flat pure path). Threaded through the
        # E-step like log_alpha so the loss backpropagates to it; getattr keeps the flat path's call
        # identical (None forwards a defaulted kwarg the flat builder ignores).
        connection_W = getattr(self, "connection_W", None)
        # connection_M: the learned gauge-COVARIANT (Route B) Regime-II connection when
        # transport_mode='regime_ii_covariant', else None (flat / regime_ii pure paths). Threaded
        # through the E-step like connection_W so the loss backpropagates to it.
        connection_M = getattr(self, "connection_M", None)
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
            if self.cfg.s_e_step:
                # Live model channel: refine s (phi0 fixed), then anchor the belief to it -- q0 and
                # the belief prior (mu_p, sigma_p) both become the refined s1. The belief E-step
                # self-couples to its prior every iteration, so s reaches mu_final even at n_e_steps=1.
                s_mu1, s_sigma1 = self._refine_s(token_ids, beliefs.phi, e_step_gradient=e_step_gradient)
                beliefs = beliefs._replace(mu=s_mu1, sigma=s_sigma1)
            # Precision-weighted attention (default OFF): fold a DETACHED per-key reliability bias
            # -log(b0 + tr Sigma_j) into log_prior so attention down-weights high-variance keys before
            # the softmax. Detached -> the closed-form belief kernel treats it as a fixed prior (exact).
            # Uses the belief sigma ENTERING the block (post s-refine): an intentional fixed encode-time
            # reliability prior held across the E-step, NOT a per-iteration one (r2 id21). The shared
            # helper folds the SAME prior in diagnostics()/attention_maps() (r2 id22).
            log_prior = self._fold_precision_bias(log_prior, beliefs.sigma)
            # capture: the last block's CONVERGED (pre-transform) belief q*, consumed by the
            # M-step self-coupling term below (manuscript: the self-term reads q*, not the
            # transformed handoff; audit 2026-06-09 overnight F19). None when the term is off.
            cap = {} if self.cfg.mstep_self_coupling_weight > 0.0 else None
            grad_rec = {} if estep_grad_out is not None else None   # E-step belief-grad capture (gated, off by default)
            out = vfe_stack(beliefs, beliefs.mu, beliefs.sigma, self.group, self.cfg,
                            log_prior=log_prior, block_norm=self.block_norm,
                            head_mixer=self.head_mixer, cg_coupling=self.cg_coupling,
                            log_alpha=log_alpha,
                            lambda_beta=lambda_beta,
                            connection_W=connection_W, connection_M=connection_M,
                            e_step_gradient=e_step_gradient,
                            rope=rope, rope_on_cov=self.cfg.rope_full_gauge,
                            rope_on_value=self.cfg.rope_on_value,
                            capture=cap, grad_record=grad_rec)
        if estep_grad_out is not None:                           # one host sync, only when requested
            for _gk in ("mu", "sigma", "phi"):
                _gv = grad_rec.get(_gk) if grad_rec is not None else None
                estep_grad_out[_gk] = float(_gv) if _gv is not None else 0.0
        mu_final = out.mu                                        # (B, N, K); head mixer (if any) applied PER BLOCK
        sigma_final = out.sigma                                  # inside vfe_stack now, not post-stack

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
        # Fused chunked-vocab decode+CE (decode_mode='diagonal_chunked', training path only): when
        # targets are given, compute the cross-entropy by iterating V in chunks and accumulating a
        # streaming logsumexp + a target-logit gather, so the (B, N, V) logit tensor is NEVER
        # materialized (the memory win). KL-readout (use_prior_bank=True) routes to the diagonal
        # kernel's fused CE (equal to 'diagonal' decode -> F.cross_entropy to atol-1e-3,
        # tests/test_chunked_decode.py); the linear ablation (use_prior_bank=False) routes to its
        # own fused CE over logits = mu @ W^T (+ b) (vram audit 2026-06-10: the dense linear path
        # retained logits + cross_entropy's log-softmax copy, ~2 x B*N*V fp32, the single largest
        # decode cost at large B). logits is None on this branch by design -- forming them would
        # defeat the purpose; the training/eval callers (train.py) discard the returned logits.
        # Inference (targets=None) still routes through decode() below for full logits.
        fused_chunked = (
            targets is not None
            and self.cfg.decode_mode in ("diagonal_chunked", "full_chunked")
        )
        if fused_chunked:
            with self._amp_off_context(token_ids.device):
                if self.cfg.use_prior_bank and self.cfg.decode_mode == "full_chunked":
                    # full-covariance KL CE via the diagonal-prior closed form: no (B,N,V) logits
                    # AND no (B,N,V,K,K) per-pair Cholesky workspace (decode_ce_full_chunked).
                    ce = self.prior_bank.decode_ce_full_chunked(
                        mu_final.float(), sigma_final.float(), targets,
                    )
                elif self.cfg.use_prior_bank:
                    ce = self.prior_bank.decode_ce_diagonal_chunked(
                        mu_final.float(), sigma_final.float(), targets,
                    )
                else:
                    # linear decode; the chunked-CE path is rank-agnostic, so a '*_chunked'
                    # decode_mode (diagonal_chunked or full_chunked) both route here. sigma_q is
                    # forwarded only when decode_precision_scaled (the head reads mu/(sigma+eps)),
                    # matching the dense _decode_linear path; None keeps the bare-mean path identical.
                    ce = self.prior_bank.decode_ce_linear_chunked(
                        mu_final.float(), targets,
                        sigma_q=(sigma_final.float() if self.cfg.decode_precision_scaled else None),
                    )
            logits = None                                        # no (B, N, V) tensor on the fused path
        else:
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
            # M-step self-coupling regularizer (manuscript Algorithm 1, GL(K)_attention.tex:2111):
            # L += alpha_hat * sum_i alpha_i D(q_i*||p_i), the alpha-weighted self-coupling of the
            # CONVERGED variational belief q* (captured by vfe_block BEFORE head_mixer /
            # cg_coupling / block_norm -- the belief the E-step's F was actually minimized over,
            # which the manuscript pins the self-term to) against the per-block prior. The prior
            # fold below mirrors vfe_stack's handoff with the TRANSFORMED outputs (out.mu), since
            # that is what the real stack hands the next block. With the three transform toggles
            # at their defaults q* IS the returned `out` (same object), so the pure path is
            # unchanged; under the toggles the term now reads q* rather than the transformed
            # handoff T(q*) it accidentally read before (audit 2026-06-09 overnight F19,
            # challenge-upheld; restores the documented intent and E-step/M-step consistency).
            # alpha_i is the SAME registered self-coupling form as the E-step / diagnostics
            # (self_coupling_alpha keyed off cfg.lambda_alpha_mode), so under state_dependent_per_coord the
            # term carries the per-token, per-coordinate alpha_i^(k)* = c0/(b0+D^(k)) rather than a
            # flat scalar; cfg.mstep_self_coupling_weight (= alpha_hat) is the overall scale. alpha_i
            # is DETACHED: by the alpha-envelope (alpha* is the stationary point of alpha*D + R(alpha),
            # so d/dalpha[alpha*D + R] = 0 there), the M-step gradient of the F self-term w.r.t. the
            # priors is alpha_i* dD/dtheta with alpha_i* held fixed -- detaching it (and dropping R)
            # is exact for the closed-form forms (constant/state_dependent/state_dependent_per_coord);
            # for the learnable NN-exception alpha, log_alpha still trains through the E-step path.
            # At constant alpha=1.0 (the default) alpha_i==1, byte-identical to the prior mean-D form.
            # Grad-connected through D (no detach on D), so it backprops to the learned prior tables,
            # like mass_phi. The last-block prior is rebuilt by mirroring vfe_stack's prior_handoff
            # fold; EXACT at n_layers=1 (loop empty -> p = encode belief), an approximation otherwise
            # (one converged belief stands in for the per-block intermediates), matching diagnostics().
            from vfe3.families import get_family
            from vfe3.free_energy import self_divergence_for_alpha
            from vfe3.alpha_i import self_coupling_alpha, alpha_is_per_coord
            cfg = self.cfg
            rho, rho_s = cfg.prior_handoff_rho, cfg.prior_handoff_sigma
            mu_p, sigma_p = beliefs.mu, beliefs.sigma
            for _ in range(cfg.n_layers - 1):
                mu_p = (1.0 - rho) * mu_p + rho * out.mu
                sigma_p = (1.0 - rho_s) * sigma_p + rho_s * out.sigma
            fam = get_family(cfg.family)
            q_conv = cap["converged"]                           # q*: pre-transform converged belief
            self_div = self_divergence_for_alpha(               # (B, N) summed, or (B, N, K) per-coord
                fam(q_conv.mu, q_conv.sigma), fam(mu_p, sigma_p),
                alpha=cfg.renyi_order, kl_max=cfg.kl_max, eps=cfg.eps,
                divergence_family=cfg.divergence_family, lambda_alpha_mode=cfg.lambda_alpha_mode,
            )
            alpha_sc, _ = self_coupling_alpha(                  # SAME form as the E-step / diagnostics
                self_div, mode=cfg.lambda_alpha_mode, value=cfg.lambda_alpha, b0=_as_coeff(cfg.b0, out.mu.device), c0=_as_coeff(cfg.c0, out.mu.device),
                log_alpha=getattr(self, "log_alpha", None),
            )
            coupling = alpha_sc.detach() * self_div            # alpha_i^(k)* D^(k) (envelope: alpha* fixed)
            if alpha_is_per_coord(cfg.lambda_alpha_mode):
                coupling = coupling.sum(dim=-1)                # sum_k alpha^(k) D^(k) -> per-token
            sc = coupling.mean()                               # mean over batch and tokens (B, N)
            loss = loss + cfg.mstep_self_coupling_weight * sc
        # DESIGN NOTE (cross-scale boundary, audit 2026-06-15): r is frozen by default (learnable_r=False).
        # learnable_r=True + r_update_mode='gradient' trains it as an empirical-Bayes centroid; on the scored
        # s_e_step=False path grad flows through THIS KL(s||r) term, but under s_e_step=True that term is
        # gated off (see the TRANSPARENCY note just below) and r instead trains through the unrolled _refine_s
        # E-step, where it inherits the straight_through/detach/oracle-truncation freeze footguns (config
        # __post_init__ warns). The empirical-Bayes reading is non-degenerate only when s is data-anchored
        # (prior_source='model_channel' or s_e_step); else KL(s||r) collapses s->r. The global r is a
        # stand-in along TWO axes: frozen-vs-learned (this toggle) AND token-UNIFORM (one (K,) tensor
        # broadcast over all tokens) vs a token-dependent r_i. NEITHER axis is an unfilled gap. The
        # manuscript's true token-dependent hyper-prior is the CROSS-SCALE shadow r_i=Omega_tilde[s^(s+1)]
        # (Participatory_it_from_bit.tex eq:cross_scale_shadow / eq:topdown_priors, line 2300): the
        # model-fiber transport of a GENUINELY EMERGED scale-(s+1) meta-agent (licensed by the
        # free-energy-improvement test, PIFB line 2164). No such meta-agent exists in this single-scale
        # transformer; the manuscript treats single-scale p_i,r_i as PRIMITIVE boundary conditions (PIFB
        # lines 554, 636) and assigns the full Omega_{i,I} transport + Ouroboros tower to a SEPARATE codebase
        # (MAgent_Model/gauge_agent/, PIFB line 2334, which disclaims the transformer's cross-layer handoff
        # as "not the implementation of the present subsection"). So the frozen global r IS the sanctioned
        # s_max boundary -- the NAMED special case of the self-referential closure
        # r_i^(top)=sum_j w_j Omega_tilde_ij[s_j] "held at its initial value rather than recomputed from the
        # active hierarchy" (PIFB line 2332) -- NOT a placeholder awaiting a missing feature. A within-vfe3
        # token-dependent r_i is buildable ONLY as that self-referential-closure special case (an
        # interpretive single-scale stand-in, NOT eq:cross_scale_shadow); even then its gauge payoff is
        # latent (no independent model-fiber frame phi_tilde: Omega_tilde reads the belief frame out.phi, so
        # transporting r against it is a rho_model != rho_state category error) and transport does not cure
        # the s->r collapse (orthogonal; only data-anchoring does). Out of scope by design, not deferred.
        # TRANSPARENCY (audit 2026-06-13 L17/L18): both s-channel blocks below are gated on
        # `not s_e_step`. Under s_e_step=True the SAME hyper-prior/gamma objective is realized as the
        # E-step descent direction that refines s (in _refine_s), so scoring it here too would
        # double-count -- the assembled scalar loss is then deliberately NOT literally
        # F + lambda_h KL(s||r) + gamma-block (consistent EM, not an omission). When scored
        # (s_e_step=False) the s-channel blocks reduce with mean() (per-position) while the belief
        # channel sums (free_energy.py); lambda_h / lambda_gamma are calibrated against that
        # per-position scale, a fixed 1/(B*N) relative to the sum-reduced belief block.
        if self.cfg.lambda_h > 0.0 and not self.cfg.s_e_step:
            # HYPER-PRIOR CHANNEL (manuscript Participatory_it_from_bit.tex eq:pointwise_free_energy,
            # lines 1241-1249): L += lambda_h * mean_i KL(s_i||r), the model-channel beliefs s_i
            # regularized toward the global hyper-prior centroid r. Opt-in, default-off
            # (lambda_h=0 -> byte-identical to the term-absent path). Grad-connected (no detach), so
            # it backprops to the learned s/r tables (the channel trains), and computed from the
            # converged s/r tables OUTSIDE the E-step (s_i does not couple into q this increment).
            # The h->s->p->q coupling and the s-channel E-step update remain DEFERRED. The weight is
            # now applied INSIDE _hyper_prior_term via the lambda_h_mode registry (constant: cfg.lambda_h;
            # state_dependent: the envelope lambda_h*_i=c0_h/(b0_h+KL) + R_h; learnable: exp(log_lambda_h)),
            # so the term is added directly with NO external lambda_h factor (byte-identical for constant).
            loss = loss + self._hyper_prior_term(token_ids)
        if self.cfg.lambda_gamma > 0.0 and not self.cfg.s_e_step:
            # MODEL-COUPLING CHANNEL (manuscript Participatory_it_from_bit.tex eq:pointwise_free_energy,
            # lines 1241-1249): L += lambda_gamma * mean_i F_red^s_i, the reduced (envelope) form of
            # the model-coupling block sum_ij [ gamma_ij KL(s_i||Omega_tilde_ij s_j) + tau_g gamma_ij
            # log(gamma_ij/pi^s_ij) ], with optimal gamma_ij = softmax_j(log pi^s - E^s/tau_g) and, at
            # the optimum, the block = -tau_g log Z^s_i. The s-channel is the SAME softmax-over-KL
            # object as the belief beta block, so it REUSES pairwise_energy + reduced_free_energy with
            # (q,p,beta,pi,tau) -> (s,Omega s,gamma,pi^s,tau_g). The s tables are always diagonal (V,K),
            # so the kernel is DiagonalGaussian regardless of cfg.family; divergence_family is the
            # orthogonal functional seam. TIED transport: Omega_tilde is the flat phi-cocycle
            # exp(phi_i)exp(-phi_j) from the CONVERGED belief frame out.phi (exact tie for the default
            # flat regime; a documented simplification under regime_ii), DETACHED -- so the gamma
            # gradient flows ONLY to the s tables and the forward (logits/ce above) is byte-identical
            # to the gamma=0 path (the model channel is predictively INERT: s does NOT feed q). The
            # detach deliberately severs the phi<-gamma coupling that full tied transport would carry in
            # the canonical E-step F; restoring it (or keeping it severed) is part of the deferred s->q
            # design, NOT this term. Computed once per forward at the loss level (like diagnostics()).
            # The body lives in _gamma_coupling_term so diagnostics logs the SAME term (audit V2).
            loss = loss + self.cfg.lambda_gamma * self._gamma_coupling_term(
                token_ids, out.phi.detach())
        return logits, loss, ce.detach()

    @property
    def _model_channel_active(self) -> bool:
        r"""Whether the model channel (the s tables) exists: any of ``lambda_h>0``,
        ``lambda_gamma>0``, ``prior_source=='model_channel'``, or ``s_e_step``. Matches
        :class:`PriorBank`'s s-table creation gate, so the s/r/h/gamma diagnostics and figures
        gate on the SAME condition the tables are built under."""
        cfg = self.cfg
        return (cfg.lambda_h > 0.0 or cfg.lambda_gamma > 0.0
                or cfg.prior_source == "model_channel" or cfg.s_e_step)

    def _hyper_prior_kl(
        self,
        token_ids: torch.Tensor,             # (B, N) integer token ids

        *,
        s_belief:  'Optional[tuple[torch.Tensor, torch.Tensor]]' = None,  # refined (mu_s, sigma_s); None -> raw s tables
        per_coord: bool = False,             # True -> unsummed per-coordinate KL_k (..,N,K) for the per-coord lambda_h form
    ) -> torch.Tensor:                       # (B, N) KL(s_i || r); (B, N, K) when per_coord
        r"""Per-token hyper-prior divergence KL(s_i||r), unreduced (the lambda_h block integrand).

        s_i is the refined model belief when ``s_belief`` is supplied (the forward's s1 under
        ``s_e_step``, so the diagnostic reads the SAME s the model uses), else encoded fresh from the
        s tables; measured against the global centroid r, grad-connected (no detach). The covariance
        kernel is DiagonalGaussian regardless of cfg.family (the s/r tables are always diagonal
        (V,K)/(K,)); r (K,) broadcasts over the (B, N) token axis. :meth:`_hyper_prior_term` reduces
        this to its mean (the forward-loss scale); :meth:`diagnostics` and the s/r/h figures consume
        the per-token vector / its sum.
        """
        from vfe3.families.gaussian import DiagonalGaussian
        from vfe3.free_energy import self_divergence, self_divergence_per_coord
        cfg = self.cfg
        pb = self.prior_bank
        s_mu, s_sigma = pb.encode_s(token_ids) if s_belief is None else s_belief   # (B, N, K)
        r_mu = pb.r_mu                                               # (K,)
        r_sigma = torch.exp(pb.r_sigma_log).clamp(min=cfg.eps)       # (K,)
        div = self_divergence_per_coord if per_coord else self_divergence
        return div(
            DiagonalGaussian(s_mu, s_sigma), DiagonalGaussian(r_mu, r_sigma),
            alpha=cfg.renyi_order, kl_max=cfg.kl_max, eps=cfg.eps,
            divergence_family=cfg.divergence_family,
        )                                                            # (B, N) summed, or (B, N, K) per-coord

    def _hyper_prior_weighted(
        self,
        token_ids: torch.Tensor,             # (B, N) integer token ids

        *,
        s_belief:  'Optional[tuple[torch.Tensor, torch.Tensor]]' = None,  # refined (mu_s, sigma_s); None -> raw s tables
    ) -> torch.Tensor:                       # (B, N) lambda_h_i KL(s_i||r) + R_h(lambda_h_i)
        r"""Per-token WEIGHTED+regularized hyper-prior integrand lambda_h_i*KL(s_i||r) + R_h(lambda_h_i).

        Applies the lambda_h_mode registry (vfe3/lambda_h_i.py) to the raw per-token KL: ``constant``
        -> cfg.lambda_h*KL with R_h=0 (byte-identical to the pre-registry cfg.lambda_h weighting);
        ``state_dependent`` -> the envelope lambda_h*_i = c0_h/(b0_h+KL) PLUS R_h, left UNDETACHED so
        autograd's product rule cancels to lambda_h*_i dKL by the envelope theorem (R_h must be in F
        for that cancellation); ``learnable`` -> exp(log_lambda_h)*KL (grad flows to log_lambda_h).
        """
        from vfe3.lambda_h_i import hyper_prior_lambda_h, lambda_h_is_per_coord
        cfg = self.cfg
        per_coord = lambda_h_is_per_coord(cfg.lambda_h_mode)
        kl_s = self._hyper_prior_kl(token_ids, s_belief=s_belief, per_coord=per_coord)  # (B,N) or (B,N,K)
        lam, reg = hyper_prior_lambda_h(
            kl_s, mode=cfg.lambda_h_mode, value=cfg.lambda_h,
            b0_h=_as_coeff(cfg.b0_h, kl_s.device), c0_h=_as_coeff(cfg.c0_h, kl_s.device),
            log_lambda_h=getattr(self, "log_lambda_h", None),
        )
        term = lam * kl_s
        if cfg.lambda_h_mode in ("state_dependent", "state_dependent_per_coord"):
            # Only the state-dependent envelopes carry a nonzero R_h (constant/learnable return a zero
            # regularizer); add it UNDETACHED so autograd's product rule cancels to lam*dKL by the
            # envelope theorem. Gating on the state-dependent forms (not '!= constant') skips the
            # zero-add on the learnable path. The per-coord form carries R_h^(k) per coordinate.
            term = term + reg                                         # R_h in F -> envelope cancellation
        if per_coord:
            term = term.sum(dim=-1)                                  # (B,N,K) -> (B,N): sum the per-coordinate envelope
        return term                                                  # (B, N)

    def _hyper_prior_term(
        self,
        token_ids: torch.Tensor,             # (B, N) integer token ids
    ) -> torch.Tensor:                       # () mean_i [ lambda_h_i KL(s_i||r) + R_h ]
        r"""Forward-loss reduction (mean over tokens) of :meth:`_hyper_prior_weighted` -- the FULLY
        WEIGHTED lambda_h block (the caller adds it to the loss directly, with NO external lambda_h
        factor). At lambda_h_mode='constant' this is cfg.lambda_h * mean_i KL(s_i||r), byte-identical
        to the prior ``cfg.lambda_h * _hyper_prior_term`` form. The raw per-token KL stays available as
        :meth:`_hyper_prior_kl` for diagnostics / the s/r/h figures.
        """
        return self._hyper_prior_weighted(token_ids).mean()

    def _gamma_energy(
        self,
        token_ids: torch.Tensor,             # (B, N) integer token ids
        phi:       torch.Tensor,             # (B, N, n_gen) gauge frame (TIED flat transport)

        *,
        s_belief:  'Optional[tuple[torch.Tensor, torch.Tensor]]' = None,  # refined (mu_s, sigma_s); None -> raw s tables
    ) -> 'tuple[torch.Tensor, float | torch.Tensor, Optional[torch.Tensor]]':
        r"""Shared model-coupling setup: the s-channel pairwise energy E^s_ij, the gamma softmax
        temperature tau_g, and the gamma attention log-prior.

        The s-channel mirror of the belief beta channel under TIED FLAT transport from ``phi``
        (Omega_tilde_ij = exp(phi_i) exp(-phi_j)): E^s_ij = D(s_i || Omega_tilde_ij s_j) on the
        diagonal s tables, per irrep block (head). Transport is factored-when-fusable (audit P4),
        exactly the E-step dispatch. Consumed by :meth:`_gamma_coupling_term` (the forward loss),
        :meth:`_gamma_coupling_terms` (the split diagnostic), and :meth:`gamma_attention_maps`
        (the gamma_ij figure), so all three read the SAME energy/temperature/prior.
        """
        from vfe3.families.gaussian import DiagonalGaussian
        from vfe3.free_energy import attention_tau, pairwise_energy
        from vfe3.geometry.transport import transport_covariance, transport_mean
        from vfe3.inference.e_step import build_belief_transport
        cfg = self.cfg
        pb = self.prior_bank
        s_mu, s_sigma = pb.encode_s(token_ids) if s_belief is None else s_belief   # (B, N, K)
        n_pos = token_ids.shape[1]
        omega = build_belief_transport(phi, self.group, transport_mode="flat")
        s_mu_t = transport_mean(omega, s_mu)                         # (B, N, N, K)
        s_sigma_t = transport_covariance(omega, s_sigma)            # (B, N, N, K) diagonal sandwich
        e_s = pairwise_energy(
            DiagonalGaussian(s_mu, s_sigma), DiagonalGaussian(s_mu_t, s_sigma_t),
            alpha=cfg.renyi_order, kl_max=cfg.kl_max, eps=cfg.eps,
            divergence_family=cfg.divergence_family, irrep_dims=self.group.irrep_dims,
        )                                                            # (B,H,N,N) block_glk; (B,N,N) single-block
        gamma_log_prior = self._attention_log_prior(
            n_pos, token_ids.device, prior=cfg.gamma_attention_prior,
        )                                                            # (N, N), cached buffer
        # Group-aware temperature: tau spans the dimension the energy accumulates over (the
        # gauge-irrep block size), exactly as the belief beta channel does. kappa_gamma is
        # gamma's own sharpness handle (not cfg.kappa_beta).
        gamma_tau = attention_tau(_as_coeff(cfg.kappa_gamma, e_s.device), self.group.irrep_dims)
        return e_s, gamma_tau, gamma_log_prior

    def _gamma_coupling_term(
        self,
        token_ids: torch.Tensor,             # (B, N) integer token ids
        phi:       torch.Tensor,             # (B, N, n_gen) converged gauge frame (detached by caller)
    ) -> torch.Tensor:                       # () model-coupling block (UNWEIGHTED)
        r"""The gamma model-coupling block at the given gauge frame (UNWEIGHTED).

        gamma = softmax_j(log pi^s - E^s/tau_g) over the :meth:`_gamma_energy` energy, reduced to
        either the canonical envelope -tau_g log Z^s (include_attention_entropy=True) or the
        surrogate sum_j gamma_ij E^s_ij (False; audit P6 -- one toggle, both channels). Shared by
        ``forward`` (grad to the s tables; caller detaches phi) and :meth:`diagnostics` (audit V2).
        :meth:`_gamma_coupling_terms` is the SPLIT (coupling vs meta-entropy) diagnostic sibling.
        """
        from vfe3.free_energy import attention_weights, reduced_free_energy
        e_s, gamma_tau, gamma_log_prior = self._gamma_energy(token_ids, phi)
        if self.cfg.include_attention_entropy:
            # canonical: the envelope -tau_g log Z^s equals coupling + entropy at gamma*
            return reduced_free_energy(e_s, tau=gamma_tau, log_prior=gamma_log_prior).mean()
        # Surrogate parity with the belief channel (audit 2026-06-09 P6): with the
        # attention-entropy term suppressed the block is sum_j gamma_ij E^s_ij at gamma*.
        gamma_w = attention_weights(e_s, tau=gamma_tau, log_prior=gamma_log_prior)
        return (gamma_w * e_s).sum(dim=-1).mean()

    def _gamma_coupling_terms(
        self,
        token_ids: torch.Tensor,             # (B, N) integer token ids
        phi:       torch.Tensor,             # (B, N, n_gen) converged gauge frame

        *,
        eps:       float = 1e-12,
        s_belief:  'Optional[tuple[torch.Tensor, torch.Tensor]]' = None,  # refined (mu_s, sigma_s); None -> raw s tables
    ) -> 'Dict[str, torch.Tensor]':          # SUM-scale {coupling, meta_entropy, total}
        r"""Split the gamma model-coupling block into its coupling and meta-entropy parts (UNWEIGHTED,
        SUM over heads and token pairs -- the scale the belief blocks report in :meth:`diagnostics`).

            coupling     = sum_{h,i,j} gamma_ij^(h) E^s_ij^(h)
            meta_entropy = sum_{h,i,j} tau_g gamma_ij^(h) log( gamma_ij^(h) / pi^s_ij )
            total        = sum_{h,i} ( -tau_g log Z^s_i )           (= coupling + meta_entropy at gamma*)

        The s-channel mirror of :func:`vfe3.metrics.free_energy_terms` (belief_coupling /
        attention_entropy): the diagnostic the figure pipeline needs to show gamma_ij KL(s_i||Omega s_j)
        and its meta-entropy SEPARATELY, which the fused envelope :meth:`_gamma_coupling_term` returns
        does not expose. pi^s is softmax(gamma_log_prior) (uniform 1/N when no prior); tau_g broadcasts
        per head exactly as the belief entropy term does.
        """
        from vfe3.free_energy import _broadcast_tau, attention_weights, reduced_free_energy
        e_s, gamma_tau, gamma_log_prior = self._gamma_energy(token_ids, phi, s_belief=s_belief)
        gamma_w = attention_weights(e_s, tau=gamma_tau, log_prior=gamma_log_prior)
        pi_s = (torch.softmax(gamma_log_prior, dim=-1) if gamma_log_prior is not None
                else torch.full_like(gamma_w, 1.0 / gamma_w.shape[-1]))
        tau_e = _broadcast_tau(gamma_tau, e_s)                       # (H,1,1) per-head, else scalar
        coupling = (gamma_w * e_s).sum()
        meta = (tau_e * gamma_w
                * (torch.log(gamma_w.clamp(min=eps)) - torch.log(pi_s.clamp(min=eps)))).sum()
        total = reduced_free_energy(e_s, tau=gamma_tau, log_prior=gamma_log_prior).sum()
        return {"coupling": coupling, "meta_entropy": meta, "total": total}

    @torch.no_grad()
    def gamma_attention_maps(
        self,
        token_ids: torch.Tensor,             # (B, N) token ids; only sequence 0 is used
    ) -> Optional[torch.Tensor]:             # (H, N, N) gamma_ij, or None when the s channel is off
        r"""Per-head model-coupling attention gamma_ij for sequence 0 (no_grad), the s-channel mirror
        of :meth:`attention_maps`.

        gamma_ij = softmax_j( log pi^s_ij - E^s_ij / tau_g ) on the model-channel beliefs s under the
        TIED FLAT transport from the CONVERGED belief gauge frame (the frame :meth:`diagnostics` and
        the gamma loss evaluate the block at). Returns ``(H, N, N)`` (rows = query i, cols = key j;
        H = len(group.irrep_dims), 1 for a single-block group) or ``None`` when the model channel is
        inactive (no s tables). OFF the training hot path (no_grad); for periodic figure generation.
        """
        if not self._model_channel_active:
            return None
        from vfe3.free_energy import attention_weights
        enc = self.prior_bank.encode(token_ids[:1])
        belief = BeliefState(mu=enc.mu[0], sigma=enc.sigma[0], phi=self._apply_pos_phi(enc.phi[0]))
        s_belief = self._refined_s_belief(token_ids)                  # s1 under s_e_step (M2), else None (raw s tables)
        if s_belief is not None:
            belief = belief._replace(mu=s_belief[0][0], sigma=s_belief[1][0])
        n = belief.mu.shape[0]
        log_prior = self._attention_log_prior(n, token_ids.device)
        log_prior = self._fold_precision_bias(log_prior, belief.sigma)  # match forward/diagnostics/attention_maps (r2 id22)
        rope = self._rope_rotation(n, token_ids.device)
        _llb = getattr(self, "log_lambda_beta", None)
        out = vfe_stack(                                             # converged belief gauge frame
            belief, belief.mu, belief.sigma, self.group, self.cfg,
            log_prior=log_prior, block_norm=self.block_norm,
            head_mixer=self.head_mixer, cg_coupling=self.cg_coupling,
            log_alpha=getattr(self, "log_alpha", None),
            lambda_beta=(self.cfg.lambda_beta if _llb is None else _llb.exp()),
            connection_W=getattr(self, "connection_W", None),
            connection_M=getattr(self, "connection_M", None),
            rope=rope, rope_on_cov=self.cfg.rope_full_gauge, rope_on_value=self.cfg.rope_on_value,
        )
        e_s, gamma_tau, gamma_log_prior = self._gamma_energy(token_ids[:1], out.phi.unsqueeze(0))
        gamma = attention_weights(e_s, tau=gamma_tau, log_prior=gamma_log_prior)[0]   # drop batch
        if gamma.dim() == 2:                                        # single-block group -> add an H=1 axis
            gamma = gamma.unsqueeze(0)
        return gamma                                                # (H, N, N)

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
                    sorted_probs = sorted_logits.softmax(dim=-1)       # compute the softmax once
                    cumprobs = sorted_probs.cumsum(dim=-1)
                    # Keep the smallest nucleus whose cumprob reaches top_p; the strict
                    # shift always keeps the top token (its cumprob>=p never removes it).
                    remove = cumprobs - sorted_probs >= top_p
                    remove_unsorted = remove.scatter(-1, sorted_idx, remove)
                    logits = logits.masked_fill(remove_unsorted, float("-inf"))
                probs = logits.softmax(dim=-1)                      # (B, V)
                next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            seq = torch.cat([seq, next_token], dim=-1)
        return seq

    @torch.no_grad()
    def _fold_precision_bias(
        self,
        log_prior: Optional[torch.Tensor],   # (N,N)/(H,N,N) position prior (batched or not), or None
        sigma:     torch.Tensor,             # (..., N, K) key belief variances (batched or unbatched)
    ) -> Optional[torch.Tensor]:
        r"""Fold the detached precision-weighted-attention reliability bias ``-log(b0 + tr Sigma_j)``
        into ``log_prior``, broadcasting over query (and head). Shared by ``forward`` and the
        diagnostic replays (``diagnostics``/``attention_maps``) so every belief-channel consumer scores
        the SAME attention prior the forward E-step descends (audit 2026-06-17 r2 id22). No-op (returns
        ``log_prior`` unchanged) when ``precision_weighted_attention`` is off. Rank-robust: ``sigma``
        may be ``(B, N, K)`` (forward) or ``(N, K)`` (diagnostics)."""
        if not self.cfg.precision_weighted_attention:
            return log_prior
        b0 = self.cfg.precision_attention_b0
        if len(self.group.irrep_dims) == 1:                # headless (.., N, N) energy: NO head axis
            kb = _precision_key_bias(sigma, b0=b0).detach()                       # (.., N)
            kb = kb.unsqueeze(-2)                                                 # (.., 1, N)
        elif self.cfg.precision_attention_per_head:        # per-head (.., H, N, N) energy
            kb = _precision_key_bias(sigma, b0=b0, irrep_dims=self.group.irrep_dims).detach()  # (.., N, H)
            kb = kb.transpose(-1, -2).unsqueeze(-2)                               # (.., H, 1, N)
        else:                                              # global bias, multi-block: head-broadcast
            kb = _precision_key_bias(sigma, b0=b0).detach()                       # (.., N)
            kb = kb.unsqueeze(-2).unsqueeze(-2)                                   # (.., 1, 1, N)
        return kb if log_prior is None else log_prior + kb

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
        ``cfg.renyi_order``, ``cfg.kl_max``, ``cfg.eps``,
        ``cfg.lambda_alpha_mode``/``value``/``b0``/``c0``, ``group.irrep_dims``, the cached
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
        from vfe3.geometry.transport import transport_mean, transport_covariance, compute_transport_operators
        from vfe3.families.base import get_family
        from vfe3.free_energy import pairwise_energy, self_divergence_for_alpha, attention_weights, attention_tau
        from vfe3.alpha_i import self_coupling_alpha
        from vfe3 import metrics
        from vfe3 import numerics

        cfg = self.cfg
        enc = self.prior_bank.encode(token_ids[:1])                    # (1, N, ...)
        belief = BeliefState(mu=enc.mu[0], sigma=enc.sigma[0], phi=self._apply_pos_phi(enc.phi[0]))
        s_belief = self._refined_s_belief(token_ids)                  # s1 under s_e_step (M2), else None (raw s tables)
        if s_belief is not None:
            belief = belief._replace(mu=s_belief[0][0], sigma=s_belief[1][0])
        n = belief.mu.shape[0]
        log_prior = self._attention_log_prior(n, token_ids.device)    # (N, N)
        log_prior = self._fold_precision_bias(log_prior, belief.sigma)  # match forward's prior (r2 id22)
        _llb = getattr(self, "log_lambda_beta", None)
        rope = self._rope_rotation(n, token_ids.device)               # rope shapes the converged belief (as forward)
        cap: dict = {}                                                # q* capture (F self-term reads it, as forward)
        out = vfe_stack(                                              # converged belief
            belief, belief.mu, belief.sigma, self.group, cfg,
            log_prior=log_prior, block_norm=self.block_norm,
            head_mixer=self.head_mixer,                               # per-block mixing -> diagnostics' final belief matches forward
            cg_coupling=self.cg_coupling,
            log_alpha=getattr(self, "log_alpha", None),               # learned scalar (None on the pure path)
            lambda_beta=(cfg.lambda_beta if _llb is None else _llb.exp()),   # learned/constant coupling weight
            connection_W=getattr(self, "connection_W", None),         # learned Regime-II connection (None on the flat pure path)
            connection_M=getattr(self, "connection_M", None),         # learned covariant (Route B) connection (None unless regime_ii_covariant)
            rope=rope, rope_on_cov=cfg.rope_full_gauge,               # match forward: converge WITH rope, not post-hoc
            rope_on_value=cfg.rope_on_value,
            capture=cap,
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
        # (rope was computed above and now also shaped the converged belief.)
        omega = _transport(                                          # (N, N, K, K)
            out.phi, self.group, transport_mode=cfg.transport_mode,
            mu=(out.mu if cfg.transport_mode in _REGIME_NEEDS_MU else None),
            sigma=(out.sigma if cfg.transport_mode in _REGIME_NEEDS_SIGMA else None),
            connection_W=getattr(self, "connection_W", None),
            connection_M=getattr(self, "connection_M", None),
            cocycle_relaxation=cfg.cocycle_relaxation,
        )
        if rope is not None:
            rope_omega = RopeTransport(base=omega, rope=rope, on_cov=cfg.rope_full_gauge,
                                       on_value=cfg.rope_on_value)
            mu_t    = transport_mean(rope_omega, out.mu)             # (N, N, K)
            sigma_t = transport_covariance(rope_omega, out.sigma)    # (N, N, K)
        else:
            mu_t    = transport_mean(omega.unsqueeze(0), out.mu.unsqueeze(0))[0]
            sigma_t = transport_covariance(omega.unsqueeze(0), out.sigma.unsqueeze(0))[0]
        fam = get_family(cfg.family)
        energy = pairwise_energy(                                    # (N, N) or (H, N, N)
            fam(out.mu, out.sigma), fam(mu_t, sigma_t),
            alpha=cfg.renyi_order, kl_max=cfg.kl_max, eps=cfg.eps,
            divergence_family=cfg.divergence_family,
            irrep_dims=self.group.irrep_dims,
        )
        beta = attention_weights(energy, tau=attention_tau(_as_coeff(cfg.kappa_beta, out.mu.device), self.group.irrep_dims), log_prior=log_prior)
        _q_conv = cap["converged"]                                   # q*: the F self-term reads the
        self_div = self_divergence_for_alpha(                        # pre-transform converged belief
            fam(_q_conv.mu, _q_conv.sigma), fam(mu_p, sigma_p),      # (matches the M-step term; F19)
            alpha=cfg.renyi_order, kl_max=cfg.kl_max, eps=cfg.eps,
            divergence_family=cfg.divergence_family, lambda_alpha_mode=cfg.lambda_alpha_mode,
        )
        alpha, alpha_reg = self_coupling_alpha(
            self_div, mode=cfg.lambda_alpha_mode, value=cfg.lambda_alpha, b0=_as_coeff(cfg.b0, out.mu.device), c0=_as_coeff(cfg.c0, out.mu.device),
            log_alpha=getattr(self, "log_alpha", None),     # learned scalar (None on the pure path)
        )

        d = {"attn_entropy": float(metrics.attention_entropy(beta))}
        _lb = cfg.lambda_beta if _llb is None else float(_llb.detach().exp())   # scaled-F total reflects lambda_beta
        terms = metrics.free_energy_terms(self_div, energy, beta, alpha,
                                          tau=attention_tau(_as_coeff(cfg.kappa_beta, out.mu.device), self.group.irrep_dims),
                                          lambda_beta=_lb, log_prior=log_prior,
                                          include_attention_entropy=cfg.include_attention_entropy,
                                          alpha_reg=(alpha_reg if cfg.lambda_alpha_mode != "constant" else None))
        d.update({k: float(v) for k, v in terms.items()})
        # Raw (un-regularized) belief->prior drift sum_i D(q_i||p_i): the divergence WITHOUT the
        # alpha_i coefficient OR the R(alpha_i) regularizer that free_energy_terms folds into
        # self_coupling. Under lambda_alpha_mode='constant' (alpha=1, R=0) the two coincide; under the
        # state-dependent envelope self_coupling is pinned near the K*c0 regularizer floor (alpha_i*
        # D + R = c0[1 + log((b0+D)/c0)] per coord), so this raw term is the informative drift signal.
        d["self_divergence"] = float(self_div.sum())            # sum over tokens (and coords if per-coord)
        # Model-channel F blocks (audit obs 18497 + the s/r/h/gamma figures): surface the hyper-prior
        # and the gamma model-coupling block whenever their tables exist, INDEPENDENT of the loss
        # gating. The loss folds these into the s-refinement under s_e_step to avoid a double GRADIENT,
        # but the free-energy VALUE still carries them, so the diagnostic decomposition reports them in
        # EVERY model-channel regime (including s_e_step=True, where they were previously invisible).
        # Reported RAW and SUM-scale over seq 0 (like the belief blocks self/belief/attention above),
        # gamma SPLIT into its coupling and meta-entropy parts; the WEIGHTED contributions are folded
        # into ``total`` at the SAME sum scale, so train.py's uniform per-token /n_tok yields a
        # commensurate decomposition (the prior fold added per-token MEANS into the per-sequence SUM
        # total -- a 1/N under-weight of the model channel).
        if cfg.lambda_h > 0.0 or cfg.s_e_step:                       # r table exists on this path
            d["hyper_prior"] = float(self._hyper_prior_kl(token_ids[:1], s_belief=s_belief).sum())   # sum_i KL(s_i||r) (refined s1 under s_e_step)
            _hp_weighted = float(self._hyper_prior_weighted(token_ids[:1], s_belief=s_belief).sum())  # WEIGHTED (lambda_h_mode); == cfg.lambda_h*hyper_prior for 'constant'
            d["hyper_prior_weighted"] = _hp_weighted                                      # EXACT contribution folded into total (state_dependent/learnable != cfg.lambda_h*raw); the F-decomposition figure reads this
            d["total"] += _hp_weighted
        if cfg.lambda_gamma > 0.0 or cfg.s_e_step:                  # gamma block evaluated at out.phi
            g = self._gamma_coupling_terms(token_ids[:1], out.phi.unsqueeze(0), s_belief=s_belief)
            d["gamma_coupling"]     = float(g["coupling"])           # raw sum_{h,i,j} gamma E^s
            d["gamma_meta_entropy"] = float(g["meta_entropy"])       # raw sum_{h,i,j} tau_g gamma log(gamma/pi^s)
            d["total"] += cfg.lambda_gamma * (d["gamma_coupling"] + d["gamma_meta_entropy"])
        spec = out.sigma if out.sigma.dim() == out.mu.dim() else torch.linalg.eigvalsh(out.sigma)
        d["effective_rank"] = float(metrics.effective_rank(spec).mean())
        # Gauge-geometry probes (diagnostics tier): the curvature proxy -- mean Frobenius departure
        # of the triangle holonomy Omega_ij Omega_jk Omega_ki from I (0 for the flat phi-cocycle) --
        # and the spread of log|det Omega| = tr(embed(phi)) across tokens (0 at phi=0). Pure
        # measurements at the converged transport; off the training graph (no_grad).
        # Curvature proxy from the SAMPLED estimator (seeded random distinct triples) rather than the
        # deterministic row-major prefix, which at N=128/max_triangles=512 covers only anchor i=0's
        # local neighborhood -- a systematically biased sample. The sampled mean is representative and
        # still ~0 on the flat phi-cocycle (flatness certificate); the dict key is unchanged.
        # ---- extended per-eval observability (2026-06-13 run-diagnostics rollout) ----
        # Every reduction below reads tensors already materialized above (out.mu/sigma/phi, exp_phi,
        # omega, energy, beta, self_div); no extra forward, no_grad. NEW keys only -- d["total"] and
        # the existing block values are untouched (test_model_channel_diagnostics pins total's closure;
        # test_regime_ii pins d["holonomy_deviation"], whose semantics is preserved as the mean below).
        _LOG2 = 0.6931471805599453                                   # row-entropy floor for a 2-way split
        _diag = out.sigma.dim() == out.mu.dim()                      # diagonal (N,K) vs full (N,K,K);
        #   passed explicitly to the spectrum/Fisher/guard metrics below because shape-squareness
        #   auto-inference mis-reads a diagonal (N, K) table as a full covariance when N == K
        #   (e.g. max_seq_len == embed_dim) -- the same dim-based test the effective_rank line uses.

        hol = metrics.holonomy_deviation_sampled(omega)
        d["holonomy_deviation"] = float(hol["mean"])                 # unchanged key/semantics
        d["holonomy_ci_lo"]     = float(hol["ci_lo"])                # bootstrap band: real curvature vs jitter
        d["holonomy_ci_hi"]     = float(hol["ci_hi"])
        # Manuscript-canonical gauge invariant: the Wilson-action density 1 - Re Tr(H)/K (PIFB:862-869),
        # the trace complement of the Frobenius certificate above; ~0 on the flat cocycle, > 0 under regime_ii.
        d["holonomy_wilson"]    = float(metrics.holonomy_wilson_sampled(omega)["deviation_mean"])
        d["gauge_trace_spread"] = float(metrics.gauge_trace_spread(out.phi, self.group.generators))

        # Group-correct gauge invariant: gauge_trace_spread is identically 0 on SO(N)/Sp(2m) (traceless
        # generators), so dispatch the right invariant of exp(phi) and report its spread over tokens.
        exp_phi = compute_transport_operators(out.phi.unsqueeze(0), self.group)["exp_phi"][0]  # (N, K, K)
        ginv = metrics.group_gauge_invariant(exp_phi, self.group).float()
        d["gauge_invariant_mean"]   = float(ginv.mean())
        d["gauge_invariant_spread"] = float(ginv.std(unbiased=False))

        # Transport DIRECTEDNESS + conditioning + sandwich overflow. NOTE: directedness
        # (transport_asymmetry, energy_*_asymmetry) is intrinsic to ANY nonzero gauge -- it is nonzero
        # on the FLAT cocycle (Omega_ji = Omega_ij^{-1} != Omega_ij), so it is NOT a curvature /
        # non-flatness signal; cocycle_residual and holonomy_deviation are the flatness diagnostics.
        d["cocycle_residual"] = float(metrics.cocycle_residual_sampled(omega))   # composition-law flatness
        _svd_v = torch.linalg.svdvals(exp_phi)                      # (N, K) vertex-factor singular values
        d["vertex_cond_max"]  = float((_svd_v[..., 0] / _svd_v[..., -1].clamp(min=cfg.eps)).max())
        #   FLAT path: pairwise cond(Omega_ij) = cond(exp_phi_i exp(-phi_j)) <= vertex_cond_max^2. Under
        #   regime_ii the edge factor exp(delta_ij) adds conditioning NOT captured here -- sandwich_absmax
        #   below is the direct (Omega Sigma Omega^T) overflow signal that DOES see it.
        d["sandwich_absmax"]  = float(sigma_t.abs().max())          # |Omega Sigma Omega^T| overflow vs fp32 ~1e7
        d["transport_asymmetry"]  = float(metrics.transport_asymmetry(omega).mean())
        _ed = metrics.energy_directedness(energy)
        d["energy_abs_asymmetry"] = float(_ed["abs_asymmetry"])
        d["energy_rel_asymmetry"] = float(_ed["rel_asymmetry"])
        # Per-head gauge specialization: do the per-head GL(d_head) frames specialize, or collapse to a
        # shared frame? Mean block anisotropy + spread of log|det| across heads (single block -> 0
        # spread). Informative for block_glk / tied_block_glk (independent per-block GL frames); for the
        # orthogonal/symplectic tied towers (so_k/so_n/sp_n) it is structurally vacuous (det=1, unit
        # singular values, one shared group element), so read it only on the GL-block groups.
        _ghi = metrics.per_head_gauge_invariants(exp_phi, self.group.irrep_dims)
        d["gauge_head_aniso_mean"]    = float(_ghi["anisotropy"].float().mean())
        d["gauge_head_logdet_spread"] = float(_ghi["logdet"].float().std(unbiased=False))

        # phi frame magnitude: a collapse to phi=0 silently degenerates to an UNGAUGED transformer
        # (trivially equivariant, so no equivariance metric flags it).
        phi_norm = torch.linalg.norm(out.phi, dim=-1)               # (N,)
        d["phi_norm_mean"] = float(phi_norm.mean())
        d["phi_norm_std"]  = float(phi_norm.std(unbiased=False))

        # Belief covariance conditioning + PD margin (effective_rank is blind to one collapsing mode).
        bs = metrics.belief_spectrum(out.sigma, diagonal=_diag, eps=cfg.eps)
        cond = bs["condition"].float()
        d["belief_cond_median"] = float(cond.median())
        d["belief_cond_p95"]    = float(torch.quantile(cond, 0.95))
        d["belief_cond_max"]    = float(cond.max())
        # clamp lam_min at eps before dividing -- matches the floor belief_spectrum's condition number
        # uses, so a floored / sub-floor belief reads ~1.0 (not 0.0) consistently across the reductions.
        d["belief_pd_margin"]   = float((bs["eigenvalues"][..., -1].clamp(min=cfg.eps).float() / cfg.eps).min())

        # Per-token effective-rank distribution (the logged mean hides a bimodal rank-1/rank-K collapse).
        er = metrics.effective_rank_per_token(out.sigma, diagonal=_diag, eps=cfg.eps).float()
        d["eff_rank_p5"]     = float(torch.quantile(er, 0.05))
        d["eff_rank_median"] = float(er.median())
        d["eff_rank_p95"]    = float(torch.quantile(er, 0.95))

        # Belief Fisher-information trace (tr Sigma^-1 / 2 = total belief precision/confidence).
        fish = metrics.fisher_trace(out.sigma, diagonal=_diag, eps=cfg.eps).float()
        d["fisher_trace_mean"]   = float(fish.mean())
        d["fisher_trace_median"] = float(fish.median())

        # Numerical safety rails inert (pure path) vs load-bearing (fixed point is a clamp artifact)?
        gs = metrics.guard_saturation(out.sigma, energy, self_div, diagonal=_diag,
                                      eps=cfg.eps, sigma_max=cfg.sigma_max, kl_max=cfg.kl_max)
        for _k, _v in gs.items():
            d[f"guard_{_k}"] = float(_v)
        # Renyi cancellation-band proximity: fraction of energies in [0.9, 1.0)*kl_max where the fp32
        # Renyi closed form catastrophically cancels (a softer signal than guard's exact-pin saturation).
        d["renyi_band_frac"] = float(((energy > 0.9 * cfg.kl_max) & (energy < cfg.kl_max)).float().mean())

        # Non-finite fraction over the converged operator tensors (one NaN silently poisons AdamW).
        d["nonfinite_frac"] = float(max(
            numerics.nan_inf_fraction(out.mu),  numerics.nan_inf_fraction(out.sigma),
            numerics.nan_inf_fraction(out.phi), numerics.nan_inf_fraction(energy),
            numerics.nan_inf_fraction(beta),
        ))

        # Attention-entropy COLLAPSE: per-head min row entropy + count of near-deterministic heads at
        # the converged (last-block) belief; the single logged attn_entropy averages collapse away.
        ent_rows = metrics.attention_entropy_rows(beta)             # (N,) single head or (H, N) multi-head
        head_min = ent_rows.min(dim=-1).values if ent_rows.dim() >= 2 else ent_rows.min().reshape(1)
        d["attn_entropy_min"]             = float(head_min.min())
        d["attn_entropy_collapsed_heads"] = float((head_min < _LOG2).float().sum())

        # Equivariance-break order parameters (CONDITIONAL columns, mirroring lambda_beta): present
        # only when the breaking toggle is on, so the per-run CSV stays rectangular.
        _cW = getattr(self, "connection_W", None)
        if _cW is not None:                                          # transport_mode='regime_ii'
            d["connection_w_norm"] = float(torch.linalg.norm(_cW.detach()))
        _cM = getattr(self, "connection_M", None)
        if _cM is not None:                                          # transport_mode='regime_ii_covariant'
            d["connection_m_norm"] = float(torch.linalg.norm(_cM.detach()))
        _hm = getattr(self, "head_mixer", None)
        if _hm is not None and hasattr(_hm, "mixer_deltas"):        # use_head_mixer=True
            d["head_mixer_drift"] = max(
                (float(torch.linalg.norm(p.detach())) for p in _hm.mixer_deltas), default=0.0)
        return d

    @torch.no_grad()
    def attention_maps(
        self,
        token_ids: torch.Tensor,           # (B, N) token ids; only sequence 0 is used
    ) -> torch.Tensor:                     # (L, H, N, N) per-layer, per-head attention beta_ij
        r"""Per-layer, per-head attention weights ``beta_ij`` for sequence 0 (no_grad).

        Replays the :func:`vfe_stack` block loop one block at a time -- mirroring its
        ``mu_p``/``sigma_p`` handoff (``prior_handoff_rho``/``prior_handoff_sigma``) line for
        line -- and, at the CONVERGED output belief of each block, recomputes the attention
        pattern the SAME way :meth:`diagnostics` does at the final belief: transport
        Omega_ij(phi) -> pairwise energy E_ij = D(q_i || Omega_ij q_j) -> beta = softmax_j
        (log_prior - E/tau). The per-irrep-block (per-head) energy gives a leading head axis
        ``H = len(group.irrep_dims)`` (1 for a single-block group: glk / so_k), so the result is
        ``(L, H, N, N)`` (rows = query i, cols = key j).

        This is OFF the training hot path (no_grad, no graph) and is intended for periodic
        figure generation, not every step. By construction the LAST layer's map equals the
        attention :meth:`diagnostics` reads (byte-identical at ``n_layers == 1``, where the
        stack is a single block and the handoff loop is empty; an approximation otherwise, since
        diagnostics folds the FINAL belief into the handoff while this replay uses each block's
        own output -- the EXACT trajectory the model ran).
        """
        from vfe3.inference.e_step import _transport
        from vfe3.geometry.transport import transport_mean, transport_covariance
        from vfe3.families.base import get_family
        from vfe3.free_energy import pairwise_energy, attention_weights, attention_tau

        cfg = self.cfg
        enc = self.prior_bank.encode(token_ids[:1])                   # (1, N, ...)
        belief = BeliefState(mu=enc.mu[0], sigma=enc.sigma[0], phi=self._apply_pos_phi(enc.phi[0]))
        if cfg.s_e_step:
            # Live model channel (audit 2026-06-09 IE1): refine s and anchor the replayed belief
            # (q0 AND the handoff prior below) to it, exactly as forward/diagnostics do, so the
            # figure attention replays the model that actually trained.
            s_mu1, s_sigma1 = self._refine_s(token_ids[:1], belief.phi.unsqueeze(0))
            belief = belief._replace(mu=s_mu1[0], sigma=s_sigma1[0])
        n = belief.mu.shape[0]
        log_prior = self._attention_log_prior(n, token_ids.device)   # (N, N)
        log_prior = self._fold_precision_bias(log_prior, belief.sigma)  # match forward's prior (r2 id22)
        fam = get_family(cfg.family)
        rho, rho_s = cfg.prior_handoff_rho, cfg.prior_handoff_sigma
        mu_p, sigma_p = belief.mu, belief.sigma

        rope = self._rope_rotation(n, token_ids.device)
        maps = []
        for _ in range(cfg.n_layers):
            _llb = getattr(self, "log_lambda_beta", None)
            belief = vfe_block(                                       # converged belief at this block
                belief, mu_p, sigma_p, self.group, cfg, log_prior=log_prior,
                block_norm=self.block_norm,
                head_mixer=self.head_mixer,                            # replay the mixer too (audit
                log_alpha=getattr(self, "log_alpha", None),            # 2026-06-09 overnight F32)
                lambda_beta=(cfg.lambda_beta if _llb is None else _llb.exp()),
                connection_W=getattr(self, "connection_W", None),
                connection_M=getattr(self, "connection_M", None),     # learned covariant (Route B) connection
                cg_coupling=self.cg_coupling,
                rope=rope, rope_on_cov=cfg.rope_full_gauge,            # match forward: converge WITH rope
                rope_on_value=cfg.rope_on_value,
            )
            # Attention at the converged belief, recomputed exactly as diagnostics does: the
            # transport regime is matched so regime_ii reads the means + learned connection_W
            # (flat ignores both), and the energy is per-irrep-block (per-head).
            omega = _transport(
                belief.phi, self.group, transport_mode=cfg.transport_mode,
                mu=(belief.mu if cfg.transport_mode in _REGIME_NEEDS_MU else None),
                sigma=(belief.sigma if cfg.transport_mode in _REGIME_NEEDS_SIGMA else None),
                connection_W=getattr(self, "connection_W", None),
                connection_M=getattr(self, "connection_M", None),
                cocycle_relaxation=cfg.cocycle_relaxation,
            )                                                        # (N, N, K, K)
            if rope is not None:
                rope_omega = RopeTransport(base=omega, rope=rope, on_cov=cfg.rope_full_gauge,
                                           on_value=cfg.rope_on_value)
                mu_t    = transport_mean(rope_omega, belief.mu)          # (N, N, K)
                sigma_t = transport_covariance(rope_omega, belief.sigma) # (N, N, K)
            else:
                mu_t    = transport_mean(omega.unsqueeze(0), belief.mu.unsqueeze(0))[0]
                sigma_t = transport_covariance(omega.unsqueeze(0), belief.sigma.unsqueeze(0))[0]
            energy = pairwise_energy(                                 # (N, N) or (H, N, N)
                fam(belief.mu, belief.sigma), fam(mu_t, sigma_t),
                alpha=cfg.renyi_order, kl_max=cfg.kl_max, eps=cfg.eps,
                divergence_family=cfg.divergence_family, irrep_dims=self.group.irrep_dims,
            )
            beta = attention_weights(energy, tau=attention_tau(_as_coeff(cfg.kappa_beta, belief.mu.device), self.group.irrep_dims), log_prior=log_prior)
            if beta.dim() == 2:                                      # single-block group -> add an H=1 axis
                beta = beta.unsqueeze(0)
            maps.append(beta)                                        # (H, N, N)

            mu_p = (1.0 - rho) * mu_p + rho * belief.mu              # handoff (mirrors vfe_stack)
            sigma_p = (1.0 - rho_s) * sigma_p + rho_s * belief.sigma
        return torch.stack(maps, dim=0)                              # (L, H, N, N)

    @torch.no_grad()
    def diagnostics_per_layer(
        self,
        token_ids: torch.Tensor,           # (B, N) token ids; only sequence 0 is used
    ) -> dict:                             # each value a list of length L = cfg.n_layers
        r"""Per-LAYER (inference-depth) belief-channel diagnostics for sequence 0 (no_grad).

        :meth:`diagnostics` collapses the block stack to the FINAL belief and reports one scalar per
        metric, so the metrics.csv and the converged-state figures never expose the depth axis. This
        replays the :func:`vfe_stack` block loop one block at a time -- mirroring its
        ``mu_p``/``sigma_p`` handoff EXACTLY as :meth:`attention_maps` does -- and at each block's
        CONVERGED belief recomputes the SAME belief-channel quantities :meth:`diagnostics` uses
        (transport Omega_ij(phi), pairwise energy E_ij, attention beta_ij, self-divergence
        D(q_i||p_i) against THAT block's prior, self-coupling alpha_i), then the same
        :mod:`vfe3.metrics` reductions. Unlike diagnostics' last-block prior reconstruction, the
        self-term here reads each block's OWN handoff prior, so the per-layer self-coupling is exact.

        The model-channel blocks (hyper-prior, gamma) are a single hierarchical coupling evaluated
        once at the converged frame, NOT iterated per block, so they are deliberately absent: the
        per-layer ``total`` is the BELIEF-channel free energy at that depth. OFF the training hot path
        (no_grad, no graph); intended for periodic figure / per-layer-CSV generation, not every step.

        Returns a dict of L-length lists: ``self_coupling``, ``belief_coupling``,
        ``attention_entropy``, ``total`` (belief-channel F), ``self_divergence``,
        ``holonomy_deviation``, ``holonomy_wilson``, ``gauge_trace_spread``,
        ``gauge_invariant_spread``, ``effective_rank``, ``attn_entropy``, ``belief_cond_median``,
        ``phi_norm_mean``.
        """
        from vfe3.inference.e_step import _transport
        from vfe3.geometry.transport import (transport_mean, transport_covariance,
                                             compute_transport_operators)
        from vfe3.families.base import get_family
        from vfe3.free_energy import (pairwise_energy, self_divergence_for_alpha,
                                      attention_weights, attention_tau)
        from vfe3.alpha_i import self_coupling_alpha
        from vfe3 import metrics

        cfg = self.cfg
        enc = self.prior_bank.encode(token_ids[:1])                   # (1, N, ...)
        belief = BeliefState(mu=enc.mu[0], sigma=enc.sigma[0], phi=self._apply_pos_phi(enc.phi[0]))
        if cfg.s_e_step:                                              # anchor q0 + handoff to refined s (as forward)
            s_mu1, s_sigma1 = self._refine_s(token_ids[:1], belief.phi.unsqueeze(0))
            belief = belief._replace(mu=s_mu1[0], sigma=s_sigma1[0])
        n = belief.mu.shape[0]
        log_prior = self._attention_log_prior(n, token_ids.device)   # (N, N)
        log_prior = self._fold_precision_bias(log_prior, belief.sigma)  # match forward's prior
        fam = get_family(cfg.family)
        _llb = getattr(self, "log_lambda_beta", None)
        _lb = cfg.lambda_beta if _llb is None else float(_llb.detach().exp())
        _tau = attention_tau(_as_coeff(cfg.kappa_beta, belief.mu.device), self.group.irrep_dims)
        rho, rho_s = cfg.prior_handoff_rho, cfg.prior_handoff_sigma
        mu_p, sigma_p = belief.mu, belief.sigma
        rope = self._rope_rotation(n, token_ids.device)

        keys = ("self_coupling", "belief_coupling", "attention_entropy", "total", "self_divergence",
                "holonomy_deviation", "holonomy_wilson", "gauge_trace_spread", "gauge_invariant_spread",
                "effective_rank", "attn_entropy", "belief_cond_median", "phi_norm_mean")
        rec: dict = {k: [] for k in keys}
        for _ in range(cfg.n_layers):
            cap: dict = {}                                            # pre-transform converged belief (F self-term)
            belief = vfe_block(                                       # converged belief at this block
                belief, mu_p, sigma_p, self.group, cfg, log_prior=log_prior,
                block_norm=self.block_norm, head_mixer=self.head_mixer, cg_coupling=self.cg_coupling,
                log_alpha=getattr(self, "log_alpha", None),
                lambda_beta=(cfg.lambda_beta if _llb is None else _llb.exp()),
                connection_W=getattr(self, "connection_W", None),
                connection_M=getattr(self, "connection_M", None),
                rope=rope, rope_on_cov=cfg.rope_full_gauge, rope_on_value=cfg.rope_on_value,
                capture=cap,
            )
            omega = _transport(                                       # (N, N, K, K) under the ACTIVE regime
                belief.phi, self.group, transport_mode=cfg.transport_mode,
                mu=(belief.mu if cfg.transport_mode in _REGIME_NEEDS_MU else None),
                sigma=(belief.sigma if cfg.transport_mode in _REGIME_NEEDS_SIGMA else None),
                connection_W=getattr(self, "connection_W", None),
                connection_M=getattr(self, "connection_M", None),
                cocycle_relaxation=cfg.cocycle_relaxation,
            )
            if rope is not None:
                rope_omega = RopeTransport(base=omega, rope=rope, on_cov=cfg.rope_full_gauge,
                                           on_value=cfg.rope_on_value)
                mu_t    = transport_mean(rope_omega, belief.mu)
                sigma_t = transport_covariance(rope_omega, belief.sigma)
            else:
                mu_t    = transport_mean(omega.unsqueeze(0), belief.mu.unsqueeze(0))[0]
                sigma_t = transport_covariance(omega.unsqueeze(0), belief.sigma.unsqueeze(0))[0]
            energy = pairwise_energy(                                 # (N, N) or (H, N, N)
                fam(belief.mu, belief.sigma), fam(mu_t, sigma_t),
                alpha=cfg.renyi_order, kl_max=cfg.kl_max, eps=cfg.eps,
                divergence_family=cfg.divergence_family, irrep_dims=self.group.irrep_dims,
            )
            beta = attention_weights(energy, tau=_tau, log_prior=log_prior)
            _q = cap["converged"]                                    # self-term reads THIS block's prior (per-layer exact)
            self_div = self_divergence_for_alpha(
                fam(_q.mu, _q.sigma), fam(mu_p, sigma_p),
                alpha=cfg.renyi_order, kl_max=cfg.kl_max, eps=cfg.eps,
                divergence_family=cfg.divergence_family, lambda_alpha_mode=cfg.lambda_alpha_mode,
            )
            alpha, alpha_reg = self_coupling_alpha(
                self_div, mode=cfg.lambda_alpha_mode, value=cfg.lambda_alpha,
                b0=_as_coeff(cfg.b0, belief.mu.device), c0=_as_coeff(cfg.c0, belief.mu.device),
                log_alpha=getattr(self, "log_alpha", None),
            )
            terms = metrics.free_energy_terms(
                self_div, energy, beta, alpha, tau=_tau, lambda_beta=_lb, log_prior=log_prior,
                include_attention_entropy=cfg.include_attention_entropy,
                alpha_reg=(alpha_reg if cfg.lambda_alpha_mode != "constant" else None),
            )
            rec["self_coupling"].append(float(terms["self_coupling"]))
            rec["belief_coupling"].append(float(terms["belief_coupling"]))
            rec["attention_entropy"].append(float(terms["attention_entropy"]))
            rec["total"].append(float(terms["total"]))
            rec["self_divergence"].append(float(self_div.sum()))
            rec["holonomy_deviation"].append(float(metrics.holonomy_deviation_sampled(omega)["mean"]))
            rec["holonomy_wilson"].append(float(metrics.holonomy_wilson_sampled(omega)["deviation_mean"]))
            rec["gauge_trace_spread"].append(float(metrics.gauge_trace_spread(belief.phi, self.group.generators)))
            exp_phi = compute_transport_operators(belief.phi.unsqueeze(0), self.group)["exp_phi"][0]
            rec["gauge_invariant_spread"].append(
                float(metrics.group_gauge_invariant(exp_phi, self.group).float().std(unbiased=False)))
            _diag = belief.sigma.dim() == belief.mu.dim()
            spec = belief.sigma if _diag else torch.linalg.eigvalsh(belief.sigma)
            rec["effective_rank"].append(float(metrics.effective_rank(spec).mean()))
            rec["attn_entropy"].append(float(metrics.attention_entropy(beta)))
            bs = metrics.belief_spectrum(belief.sigma, diagonal=_diag, eps=cfg.eps)
            rec["belief_cond_median"].append(float(bs["condition"].float().median()))
            rec["phi_norm_mean"].append(float(torch.linalg.norm(belief.phi, dim=-1).mean()))

            mu_p = (1.0 - rho) * mu_p + rho * belief.mu              # handoff (mirrors vfe_stack)
            sigma_p = (1.0 - rho_s) * sigma_p + rho_s * belief.sigma
        return rec
