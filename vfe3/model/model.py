r"""The full VFE_3.0 model: encode -> E-step inference -> decode -> cross-entropy.

No neural layers (no nn.Linear/MLP/activation): on the pure default path the parameters are the
PriorBank's prior tables, plus the model-owned learned tables their toggles create -- the default
pos_phi='learned' positional table, and the default-OFF head mixer, CG coupling, regime_ii
connection, and learnable T5-bias scalar. The E-step
is unrolled into the training graph (the differentiable filtering kernel), so the CE
loss backpropagates through inference to the encode/phi priors. Batching loops over
the batch around the (unbatched) E-step; decode and CE are batched.
"""

import inspect
import math
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Tuple, Dict

import torch
import torch.nn.functional as F
from torch import nn

from vfe3.attention_prior import attention_log_prior
from vfe3.belief import BeliefState
from vfe3.config import VFE3Config
from vfe3.contracts import (
    EffectiveBetaPriorContext,
    EStepGradientOutput,
    EStepGradientRecord,
    MetropolisObjectiveContext,
    MStepCapture,
)
from vfe3.geometry.groups import GaugeGroup, get_group
from vfe3.geometry.lie_ops import CompactBlockElement, project_phi_to_slk
from vfe3.geometry.norms import get_norm
from vfe3.geometry.rope import get_pos_rotation
from vfe3.geometry.transport import (CompactFactoredTransport, DirectLinkTransport, FactoredTransport, RopeTransport,
                                     _TRANSPORT_NEEDS_MU, _TRANSPORT_NEEDS_SIGMA)
from vfe3.model.head_mixer import HeadMixer
from vfe3.model.block import _as_coeff, vfe_block
from vfe3.model.model_frame import resolve_model_frame
from vfe3.model.positional_phi import apply_positional_phi, positional_phi_coords
from vfe3.model.prior_bank import PriorBank, get_decode_registration
from vfe3.model.stack import vfe_stack
from vfe3.families.base import get_family


# Transport-mode state-routing sets: which regimes' Omega builders read mu/sigma. Sourced from the
# transport registry metadata (register_transport(needs_mu=/needs_sigma=)) so a new stateful regime
# advertises its requirements AT REGISTRATION; the callers below feed mu/sigma by membership here,
# never by matching literal mode names (the add-by-registering contract).
_REGIME_NEEDS_MU    = _TRANSPORT_NEEDS_MU
_REGIME_NEEDS_SIGMA = _TRANSPORT_NEEDS_SIGMA


def _precision_key_bias(
    sigma:      torch.Tensor,        # (B, N, K)/(B, N, K, K) family dispersion

    *,
    b0:         float = 1.0,
    family:     str   = "gaussian_diagonal",
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
    covariance_diagonal = get_family(family).covariance_diagonal(sigma)
    if irrep_dims is None:
        return -torch.log(b0 + covariance_diagonal.sum(dim=-1))             # (B, N) global trace
    tr = torch.stack([blk.sum(dim=-1)                                       # (B, N, H) per-block traces
                      for blk in covariance_diagonal.split(list(irrep_dims), dim=-1)], dim=-1)
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


@dataclass(frozen=True)
class DiagnosticSnapshot:
    r"""Frozen tensors captured from one no-grad belief forward for evaluation consumers."""

    owner:             object = field(repr=False, compare=False)
    token_ids:         torch.Tensor
    encoded_belief:    BeliefState
    initial_belief:    BeliefState
    layer_priors:      'Tuple[Tuple[torch.Tensor, torch.Tensor], ...]'
    layer_converged:   'Tuple[BeliefState, ...]'
    layer_outputs:     'Tuple[BeliefState, ...]'
    stack_output:      BeliefState
    final_belief:      BeliefState
    logits:            torch.Tensor
    beta_maps:         torch.Tensor
    gamma_maps:        Optional[torch.Tensor]
    model_phi:         Optional[torch.Tensor]
    trace_states:      'Tuple[BeliefState, ...]'
    trace_free_energy: torch.Tensor
    s_encoded_belief:  'Optional[Tuple[torch.Tensor, torch.Tensor]]'
    s_belief:          'Optional[Tuple[torch.Tensor, torch.Tensor]]'
    rope:              Optional[torch.Tensor]
    log_prior:         Optional[torch.Tensor]


def _freeze_tensor(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return None if value is None else value.detach().clone()


def _freeze_frame(
    value: 'torch.Tensor | CompactBlockElement | None',
) -> 'torch.Tensor | CompactBlockElement | None':
    if value is None:
        return None
    return value.detach().clone()


def _freeze_belief(belief: BeliefState) -> BeliefState:
    return belief._replace(
        mu=_freeze_tensor(belief.mu),
        sigma=_freeze_tensor(belief.sigma),
        phi=_freeze_tensor(belief.phi),
        s=_freeze_tensor(belief.s),
        r=_freeze_tensor(belief.r),
        omega=_freeze_frame(belief.omega),
        reflection=_freeze_tensor(belief.reflection),
        right_phi=_freeze_tensor(belief.right_phi),
    )


def _sequence_belief(belief: BeliefState, index: int = 0) -> BeliefState:
    return belief._replace(
        mu=belief.mu[index],
        sigma=belief.sigma[index],
        phi=belief.phi[index],
        s=belief.s[index] if belief.s is not None else None,
        r=belief.r[index] if belief.r is not None else None,
        omega=belief.omega[index] if belief.omega is not None else None,
        reflection=belief.reflection[index] if belief.reflection is not None else None,
        right_phi=(belief.right_phi[index]
                   if belief.right_phi is not None and belief.right_phi.dim() == belief.phi.dim()
                   else belief.right_phi),
    )


class VFEModel(nn.Module):
    """encode -> E-step stack -> decode -> CE. Parameters live only in the PriorBank."""

    def __init__(self, cfg: VFE3Config) -> None:
        super().__init__()
        # Reproducibility is pinned at the entry point run_training (torch.manual_seed(cfg.seed)
        # before model + loader are built), NOT here: seeding inside __init__ would clobber a
        # caller-set RNG state (e.g. a test that seeds then constructs several models).
        self.cfg = cfg
        self._transport_status = {"regime_ii_covariant_feature_exact": True}
        self.group = build_group(cfg)
        # ALiBi-family priors carry a per-head (n_heads, N, N) axis, while the energy's head axis
        # is len(irrep_dims); a mismatch right-aligns the prior's head axis against the BATCH axis
        # of a single-block (B, N, N) energy -- silent corruption at B=1, RuntimeError otherwise
        # (audit 2026-06-09 P1). Reject at construction. Single-block groups may run alibi with
        # n_heads=1 (the (1, N, N) prior is squeezed to the (N, N) convention in
        # _attention_log_prior below).
        for _pname in ("beta_attention_prior", "gamma_attention_prior"):
            if (getattr(cfg, _pname) in ("alibi", "causal_alibi", "causal_alibi_noself")
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
            family=cfg.family,
            # divergence_family/renyi_order feed the family-consistent decode kernels
            # (decode_mode='family'/'family_chunked'), so the readout scores the SAME divergence the
            # E-step minimized (PB-14). The fast gaussian kernels ignore them.
            divergence_family=cfg.divergence_family, renyi_order=cfg.renyi_order,
            use_prior_bank=cfg.use_prior_bank, decode_bias=cfg.decode_bias,
            encode_mode=cfg.encode_mode, decode_mode=cfg.decode_mode,
            decode_chunk_size=cfg.decode_chunk_size,
            lambda_h=cfg.lambda_h, lambda_gamma=cfg.lambda_gamma,
            prior_source=cfg.prior_source, s_frame_mode=cfg.s_frame_mode,
            s_e_step=cfg.s_e_step,
            # r is a GRADIENT leaf only under r_update_mode='gradient'; under 'barycenter' it is
            # set in-place each M-step by the closed-form barycenter (PriorBank.barycenter_r_,
            # driven from train_step) and so must stay ungrouped/requires_grad=False.
            learnable_r=cfg.learnable_r and cfg.r_update_mode == "gradient",
            # Tier-1/Tier-2 decode toggles (2026-07-05; all default OFF, byte-identical):
            unigram_kappa=cfg.unigram_kappa,
            decode_unigram_prior=cfg.decode_unigram_prior,
            untie_decode_bank=cfg.untie_decode_bank and cfg.use_prior_bank,
            gauge_parameterization=cfg.gauge_parameterization,
            irrep_dims=list(self.group.irrep_dims),
            omega_reflection=cfg.omega_reflection,
            phi_reflection=cfg.phi_reflection,
            # Compact block storage is opt-in (default OFF), and eligibility is a GROUP property decided
            # HERE where the group is known (the bank does not hold the GaugeGroup): only the equal-block
            # GL groups have independent per-head blocks -- untied block_glk stores H blocks (V,H,d,d),
            # tied tied_block_glk shares one block (V,d,d). The irrep towers so_n/sp_n can ALSO have
            # equal irrep_dims (e.g. [3,3]) but are irrep IMAGES of one element, NOT independent blocks;
            # compacting them would break the tower gauge, void param parity, and (so_n) void the
            # transpose inverse. So the flag is passed through ONLY for block_glk/tied_block_glk; every
            # other group keeps the full (V,K,K) table (the flag is a no-op for them).
            omega_compact_storage=(cfg.omega_compact_storage
                                   and cfg.gauge_group in ("block_glk", "tied_block_glk")),
            gauge_group_is_tied=(cfg.gauge_group == "tied_block_glk"),
            gauge_group_name=self.group.name,
        )
        # Norm instances built ONCE (audit 2d/4f). The gauge-pure norms (none/mahalanobis) and the
        # parameter-free layernorm are stateless pure maps (K, eps); layernorm_affine=True makes a
        # "layernorm" seam an AffineLayerNorm nn.Module whose gamma/beta REGISTER as submodule params
        # (grouped by build_optimizer). Each seam gets its own affine pair; the block seam's affine
        # is shared across the L blocks (the block norm is a single shared instance). affine=... is
        # ignored by the none/mahalanobis builders (they accept **kwargs).
        self.block_norm = get_norm(cfg.norm_type_block)(cfg.embed_dim, eps=cfg.eps,
                                                         family=cfg.family,
                                                         affine=cfg.layernorm_affine) \
            if cfg.norm_type_block != "none" else None
        self.final_norm = get_norm(cfg.norm_type_final)(cfg.embed_dim, eps=cfg.eps,
                                                         family=cfg.family,
                                                         affine=cfg.layernorm_affine) \
            if cfg.norm_type_final != "none" else None
        # layernorm_affine footgun / inert warnings (mirror learnable_kappa). The affine gamma/beta
        # exist only on a "layernorm" seam; on the BLOCK seam they are applied inside the E-step, so
        # a severing estimator freezes them.
        if cfg.layernorm_affine:
            _ln_block = cfg.norm_type_block == "layernorm"
            _ln_final = cfg.norm_type_final == "layernorm"
            if not (_ln_block or _ln_final):
                import warnings
                warnings.warn(
                    "layernorm_affine=True but neither norm_type_block nor norm_type_final is "
                    "'layernorm': no affine gamma/beta are created and the toggle is inert. Set a "
                    "norm seam to 'layernorm' to use the learned affine.",
                    UserWarning, stacklevel=2,
                )
            elif _ln_block and cfg.effective_e_step_gradient == "detach":
                # Unlike learnable_kappa (which enters only the E-step TANGENT and is also frozen by
                # 'straight_through'), the block affine is applied to the belief VALUE, so it trains
                # under 'unroll' AND 'straight_through' (both keep the value differentiable) and is
                # frozen ONLY by the fully-detached E-step, which no_grads the whole stack.
                import warnings
                warnings.warn(
                    "layernorm_affine=True with norm_type_block='layernorm' and the effective "
                    "E-step estimator 'detach' freezes the BLOCK norm's gamma/beta: 'detach' runs "
                    "the whole belief stack (including the block norm) under no_grad, so the affine "
                    "receives no gradient. Use 'unroll' or 'straight_through' (both keep the belief "
                    "value differentiable), or place the affine layernorm on norm_type_final "
                    "(post-stack, trains under any estimator).",
                    UserWarning, stacklevel=2,
                )
        # Opt-in Schur-commutant head mixer (default off). Built ONCE from the gauge group's
        # irrep blocks. Label-less groups need >= 2 EQUAL blocks (block_glk/tied_block_glk);
        # labeled irrep towers (so_n/sp_n) mix per isotypic component (mults-one towers get
        # per-head scalar gains -- the entire linear commutant there). Bad pairings fail here,
        # not at forward.
        self.head_mixer = HeadMixer(self.group.irrep_dims, family=cfg.family,
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
                self.group.irrep_dims, self.group.irrep_labels,
                cg_covariance_mode=cfg.cg_covariance_mode)
        else:
            self.cg_coupling = None
        if (cfg.use_head_mixer
                or (cfg.use_cg_coupling and cfg.cg_energy_weight == 0.0)) \
                and cfg.effective_e_step_gradient == "detach":
            # Footgun (mirrors connection_W / pos_phi_free above and below): the
            # mixer and the CG coupling are applied INSIDE the vfe_stack call, which the
            # 'detach' estimator wraps wholesale in no_grad (block.py:73-78 under
            # model.forward's `run`), so mixer_deltas / path_weights build no graph, receive
            # no gradient, and silently stay frozen at their identity/zero init -- the model
            # trains its other parameters and LOOKS healthy while these two opt-in components
            # never adapt (audit 2026-06-09 overnight F31, challenge-upheld). Gate on the
            # EFFECTIVE estimator so both the detach_e_step bool and the
            # e_step_gradient='detach' string route warn; 'unroll' and 'straight_through'
            # run the stack grad-enabled and train them. EXCEPTION (PB-13): with
            # cg_energy_weight>0 the post-stack torch.enable_grad re-evaluation trains
            # path_weights through the CG moment energy even under detach (the mean->CE path
            # stays detached), so the 'frozen at zero init' claim is false there and the CG
            # clause is gated off; the head mixer has no such side channel and always warns.
            import warnings
            warnings.warn(
                "use_head_mixer/use_cg_coupling with the effective E-step estimator 'detach' "
                "freezes mixer_deltas/path_weights: both modules are applied inside the "
                "no_grad-wrapped vfe_stack, so they receive NO gradient and stay at their "
                "identity/zero init. Use an 'unroll' E-step (detach_e_step=False, "
                "e_step_gradient='unroll') or 'straight_through' to train them (or, for the "
                "CG coupling alone, cg_energy_weight>0, whose detached-mode re-evaluation "
                "trains path_weights).",
                stacklevel=2,
            )
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
                # Footgun (mirrors use_prior_bank): connection_W enters the loss ONLY
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
        # NEURAL-NETWORK EXCEPTION (sanctioned, default-off): the LEARNED DIRECT LINK connection_L.
        # Both direct-link modes -- 'regime_ii_link' (bare: Omega_ij = exp(link_alpha A_ij . G)) and
        # 'regime_ii_link_charted' (charted: exp(phi_i) exp(A) exp(-phi_j)) -- read the SAME learned
        # table A = connection_L of shape (max_seq_len, max_seq_len, n_gen). Init ZERO -> the bare link
        # is identity links and the charted link is the flat cocycle, so a link model is flat-equivalent
        # at init to fp32 tolerance (the zero table takes the generic exp path to keep
        # d Omega/d connection_L alive). NOT created on any other transport_mode (the path is param-free).
        if cfg.transport_mode in ("regime_ii_link", "regime_ii_link_charted"):
            self.connection_L = nn.Parameter(torch.zeros(cfg.max_seq_len, cfg.max_seq_len, n_gen))
            if cfg.detach_e_step:
                # Footgun (mirrors connection_W / connection_M): connection_L enters the loss ONLY
                # through the E-step belief updates, so the detached (no_grad) E-step freezes it at its
                # zero init (the flat/identity transport). Set detach_e_step=False to train the link.
                import warnings
                warnings.warn(
                    f"transport_mode={cfg.transport_mode!r} with detach_e_step=True freezes "
                    "connection_L: the learned direct link enters the loss only through the E-step, "
                    "which the detached (no_grad) E-step severs, so connection_L.grad is None and the "
                    "transport stays flat. Set detach_e_step=False to train the direct link.",
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
        # nn.Parameter like connection_W, not a network); the "none"/"frozen" paths add no
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
            if cfg.s_frame_mode == "phi_tilde":
                self.s_pos_phi_free = nn.Parameter(self.pos_phi_free.detach().clone())
            if cfg.effective_e_step_gradient in ("detach", "straight_through"):
                # Footgun (mirrors connection_W): pos_phi_free enters the loss ONLY
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
        # equivariance. Created ONLY when t5_relative_bias is an active channel (so the parameter is
        # never orphaned); else no t5_bias attribute and the
        # pure path stays param-free (the fixed-table default still runs).
        if cfg.t5_learnable_bias and "t5_relative_bias" in (cfg.beta_attention_prior, cfg.gamma_attention_prior):
            self.t5_bias = nn.Parameter(-torch.log1p(torch.arange(cfg.t5_num_buckets, dtype=torch.float32)))
            if cfg.effective_e_step_gradient in ("detach", "straight_through"):
                # Footgun (mirrors connection_W / pos_phi_free, warned at config.py:1267):
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
        # SANCTIONED LEARNED-SCALAR EXCEPTION (t5-exception family, default OFF): learnable
        # per-irrep-block softmax temperatures, kappa = exp(log_kappa). Log-space keeps tau
        # strictly positive for ANY parameter value (tau divides the softmax logits), and
        # log(1.0) = 0 / exp(0) = 1.0 are exact, so a learnable model is byte-identical to the
        # config-scalar path at construction. Shape (len(irrep_dims),) -- exactly the length
        # attention_tau validates (n_heads for block_glk/tied_block_glk, the tower block count
        # for so_n/sp_n, 1 under cross_couplings). Init reads cfg.kappa_* (a scalar broadcasts;
        # a per-head list, already length-validated by __post_init__, seeds elementwise) and
        # draws zero RNG. A per-block scalar temperature multiplies the already-gauge-invariant
        # per-block energy inside the softmax and touches NO gauge transport, so it does NOT
        # break gauge equivariance (the cleanest exception class, like t5_bias).
        if cfg.learnable_kappa_beta:
            k0 = cfg.kappa_beta
            k0_vec = (torch.tensor(list(k0), dtype=torch.float32) if isinstance(k0, (list, tuple))
                      else torch.full((len(self.group.irrep_dims),), float(k0)))
            self.log_kappa_beta = nn.Parameter(torch.log(k0_vec))
            if cfg.effective_e_step_gradient in ("detach", "straight_through"):
                # Footgun (mirrors t5_bias above): kappa_beta enters the loss ONLY through the
                # E-step softmax temperature tau = kappa * sqrt(d_block); both severing estimators
                # cut that path ('detach' wraps the E-step in no_grad, 'straight_through' detaches
                # the per-iteration belief tangent), so log_kappa_beta.grad is None and the
                # temperature stays at its config init.
                import warnings
                warnings.warn(
                    f"learnable_kappa_beta=True with the effective E-step estimator "
                    f"{cfg.effective_e_step_gradient!r} freezes log_kappa_beta: the belief-channel "
                    f"temperature enters the loss only through the E-step softmax, which this "
                    f"estimator severs. Use an 'unroll' E-step (detach_e_step=False, "
                    f"e_step_gradient='unroll') to train it.",
                    stacklevel=2,
                )
        if cfg.learnable_kappa_gamma:
            k0 = cfg.kappa_gamma
            k0_vec = (torch.tensor(list(k0), dtype=torch.float32) if isinstance(k0, (list, tuple))
                      else torch.full((len(self.group.irrep_dims),), float(k0)))
            self.log_kappa_gamma = nn.Parameter(torch.log(k0_vec))
            if (not cfg.s_e_step) and cfg.lambda_gamma == 0.0:
                # No gamma loss path at all (the scored gamma block needs lambda_gamma > 0; the
                # s-refine E-step needs s_e_step=True): the toggle is inert. Warn (mirrors the
                # t5 inert warning above) so a dead toggle is not mistaken for a trained
                # temperature.
                import warnings
                warnings.warn(
                    "learnable_kappa_gamma=True but no gamma loss path is active (lambda_gamma == 0 "
                    "and s_e_step=False): log_kappa_gamma receives no gradient and the toggle is "
                    "inert. Set lambda_gamma > 0 (scored gamma block) or s_e_step=True to use it.",
                    UserWarning, stacklevel=2,
                )
            elif cfg.s_e_step and cfg.effective_e_step_gradient in ("detach", "straight_through"):
                # Under s_e_step the scored gamma block is skipped and kappa_gamma is consumed
                # only inside _refine_s's E-step, which these estimators sever (the same footgun
                # as log_kappa_beta). NB under s_e_step=False + lambda_gamma > 0 the gamma block
                # is assembled at the LOSS level (outside the E-step wrapper), so log_kappa_gamma
                # trains under ANY estimator and no warning fires.
                import warnings
                warnings.warn(
                    f"learnable_kappa_gamma=True with s_e_step=True and the effective E-step "
                    f"estimator {cfg.effective_e_step_gradient!r} freezes log_kappa_gamma: under "
                    f"s_e_step the model-channel temperature enters the loss only through "
                    f"_refine_s's E-step, which this estimator severs. Use an 'unroll' E-step "
                    f"(detach_e_step=False, e_step_gradient='unroll') to train it.",
                    stacklevel=2,
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
        key = (n, device, dtype, self.cfg.pos_rotation, self.cfg.rope_base)
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

    def _compact_phi_blocks_enabled(self) -> bool:
        r"""Whether this model is on the canonical route supported by the packed phi fast path."""
        cfg = self.cfg
        return (
            cfg.compact_phi_block_transport
            and cfg.gauge_parameterization == "phi"
            and cfg.transport_mode == "flat"
            and cfg.phi_reflection == "off"
            and self.group.phi_coordinate_layout == "block_head_row_major"
        )

    def _apply_pos_phi(self, phi: torch.Tensor) -> torch.Tensor:
        r"""Return the stored left coordinates, composing in-chart modes before transport."""
        if self.cfg.pos_phi == "none":
            return phi
        if self.cfg.pos_phi_compose == "group_product":
            return phi
        return apply_positional_phi(
            phi, self.group,
            mode=self.cfg.pos_phi, compose_mode=self.cfg.pos_phi_compose,
            order=self.cfg.bch_pe_order, scale=self.cfg.pos_phi_scale,
            project_slk=self.cfg.pos_phi_project_slk,
            compact_blocks=self._compact_phi_blocks_enabled(),
            pos_phi_free=getattr(self, "pos_phi_free", None),
            bch_residual_max=self.cfg.bch_residual_max,
        )

    def _pos_phi_right(self, phi: torch.Tensor) -> Optional[torch.Tensor]:
        r"""Return positional coordinates Y for the exact frame exp(X) exp(Y), else ``None``."""
        if self.cfg.pos_phi_compose != "group_product" or self.cfg.pos_phi == "none":
            return None
        coords = positional_phi_coords(
            self.cfg.pos_phi, phi.shape[-2], phi.shape[-1],
            scale=self.cfg.pos_phi_scale,
            pos_phi_free=getattr(self, "pos_phi_free", None),
            device=phi.device, dtype=phi.dtype,
        )
        if coords is not None and self.cfg.pos_phi_project_slk:
            coords = project_phi_to_slk(coords, self.group.generators, self.group.irrep_dims)
        return coords

    def _resolve_model_frame(
        self,
        token_ids:  torch.Tensor,        # (B, N) integer token ids
        belief_phi: torch.Tensor,        # (B, N, n_gen) already composed belief frame
    ) -> torch.Tensor:                   # (B, N, n_gen) effective model-channel frame
        r"""Resolve the tied or independently stored model-channel frame once."""
        independent = self.cfg.s_frame_mode == "phi_tilde"
        return resolve_model_frame(
            belief_phi,
            mode=self.cfg.s_frame_mode,
            model_phi=self.prior_bank.s_phi(token_ids) if independent else None,
            group=self.group,
            pos_phi_free=getattr(self, "s_pos_phi_free", None) if independent else None,
            pos_phi=self.cfg.pos_phi,
            compose_mode=self.cfg.pos_phi_compose,
            bch_order=self.cfg.bch_pe_order,
            bch_residual_max=self.cfg.bch_residual_max,
            pos_phi_scale=self.cfg.pos_phi_scale,
            project_slk=self.cfg.pos_phi_project_slk,
            compact_blocks=self._compact_phi_blocks_enabled(),
        )

    def effective_kappa_beta(self, device: torch.device) -> 'float | torch.Tensor':
        r"""The belief-channel kappa actually in force: exp(log_kappa_beta) (live, differentiable)
        under learnable_kappa_beta, else the config constant via _as_coeff (a scalar float, or an
        (H,) tensor for a per-head list) -- value-identical to the learnable init."""
        p = getattr(self, "log_kappa_beta", None)
        return torch.exp(p).to(device) if p is not None else _as_coeff(self.cfg.kappa_beta, device)

    def effective_kappa_gamma(self, device: torch.device) -> 'float | torch.Tensor':
        r"""The model-channel kappa actually in force (the kappa_gamma analogue of
        :meth:`effective_kappa_beta`)."""
        p = getattr(self, "log_kappa_gamma", None)
        return torch.exp(p).to(device) if p is not None else _as_coeff(self.cfg.kappa_gamma, device)

    def _model_channel_connection_kwargs(self) -> dict:
        r"""The connection-law knob bag the model channel SHARES with the belief channel (PB-11):
        the active learned connections (``connection_W``/``M``/``L``, all None on the flat pure path)
        plus the link/cocycle/clamp controls. Threaded into BOTH :meth:`_refine_s`'s E-step and
        :meth:`_gamma_energy`'s transport build so the s-fiber transports under the SAME connection
        the belief E-step uses, instead of an isolated flat cocycle. On the flat pure path every
        connection is None and the controls are their inert defaults, so the call is byte-identical.
        The per-mode decision of WHICH belief tensors (s_mu/s_sigma) feed the builder is made by the
        caller from the transport-registration metadata, not here."""
        cfg = self.cfg
        return dict(
            connection_W=getattr(self, "connection_W", None),
            connection_M=getattr(self, "connection_M", None),
            connection_L=getattr(self, "connection_L", None),
            link_alpha=cfg.link_alpha, link_soft_cap=cfg.link_soft_cap,
            cocycle_relaxation=cfg.cocycle_relaxation,
            clamp_monitor=cfg.transport_clamp_monitor,
        )

    def _refine_s(
        self,
        token_ids:          torch.Tensor,   # (B, N) integer token ids
        phi0:               torch.Tensor,   # (B, N, n_gen) effective model frame (held FIXED)

        *,
        e_step_gradient:    str                    = "unroll",
        rope:               Optional[torch.Tensor] = None,      # (N, K, K) gauge-RoPE rotation (None -> off)
        prebuilt_transport: 'torch.Tensor | CompactFactoredTransport | DirectLinkTransport | FactoredTransport | RopeTransport | None' = None,
    ) -> 'tuple[torch.Tensor, torch.Tensor]':
        r"""Refine the model channel s by its own E-step toward the frozen hyper-prior r plus the
        gamma model-consensus, with the selected model frame phi0 held fixed (e_phi_lr=0). Returns the
        refined (mu_s, sigma_s); the s-tables train through the unrolled trajectory."""
        from vfe3.belief import BeliefState
        from vfe3.inference.e_step import e_step
        from vfe3.free_energy import attention_tau

        cfg, pb, grp = self.cfg, self.prior_bank, self.group
        s_mu, s_sigma = pb.encode_s(token_ids)                         # (B, N, K)
        # omega_direct frame fidelity (Phase 3 Task 3): re-derive the stored belief frame U_i so the
        # s-channel E-step transports the gamma coupling by U_i U_j^{-1} (via e_step's internal
        # build_belief_transport), not the flat exp(phi0) cocycle phi0 is held at. ATTACHED (no
        # detach): the s-refine trains omega_embed through the unrolled trajectory exactly as it
        # trains phi_embed.
        omega_s = pb._omega_lookup(token_ids) if cfg.gauge_parameterization == "omega_direct" else None
        # phi-reflection frame fidelity: re-derive the per-token sign R_i so the s-channel E-step
        # transports the gamma coupling by R_i exp(phi0_i) exp(-phi0_j) R_j (via e_step's internal
        # build_belief_transport), matching the belief channel. A BUFFER lookup (non-differentiable),
        # so -- unlike omega_s -- there is NO detach/attach concern; None on the pure path.
        reflection_s = pb.reflection_sign[token_ids] if cfg.phi_reflection != "off" else None
        # Family-rank frozen centroid r (PB-11): r_parameters() returns the (K,) diagonal / (K,K) full
        # covariance matching cfg.family, so the full-covariance s E-step scores against a full r
        # (the prior fix that unblocks s_e_step + gaussian_full; the diagonal path is byte-identical
        # since r_parameters uses the SAME bounded_variance_from_log(pb.r_sigma_log, eps=pb.eps)).
        r_mu_t, r_sigma_t = pb.r_parameters()                         # (K,) / (K,) or (K,K)
        r_mu    = r_mu_t.expand_as(s_mu)                              # (B, N, K) frozen r broadcast
        r_sigma = r_sigma_t.expand_as(s_sigma)                        # (B,N,K) diag or (B,N,K,K) full
        gamma_tau       = attention_tau(self.effective_kappa_gamma(s_mu.device), grp.irrep_dims)
        gamma_log_prior = self._attention_log_prior(
            token_ids.shape[1], token_ids.device, prior=cfg.gamma_attention_prior,
        )
        # cfg.lambda_h is the absolute channel gate. State-dependent coefficient forms ignore their
        # scalar value, so dispatch the zero gate through the constant-zero form before refinement.
        lambda_h_mode = "constant" if cfg.lambda_h == 0.0 else cfg.lambda_h_mode
        out = e_step(
            BeliefState(mu=s_mu, sigma=s_sigma, phi=phi0, omega=omega_s,
                        reflection=reflection_s, right_phi=self._pos_phi_right(phi0)),
            r_mu, r_sigma, grp,
            n_iter=cfg.n_e_steps,         tau=gamma_tau,
            e_q_mu_lr=cfg.e_s_mu_lr,      e_q_sigma_lr=cfg.e_s_sigma_lr, e_phi_lr=0.0,
            # The s-channel self-coupling weight IS lambda_h (the hyper-prior precision): route it
            # through the lambda_h_mode registry, not a hardcoded constant. e_step's self_coupling_alpha
            # consumes (value, lambda_alpha_mode, b0, c0) exactly as lambda_h_i.hyper_prior_lambda_h
            # does. ENVELOPE CANCELLATION (audit 2026-06-13): under state_dependent the s E-step gets the
            # correct lam*(KL)*dKL gradient by the envelope theorem -- on the LIVE kernel route the
            # belief-gradient kernel multiplies dKL by the envelope COEFFICIENT alpha*=c0_h/(b0_h+KL) and
            # never literally adds R_h (R_h's d/dbelief is 0, so omitting it is exact); only the autograd
            # ORACLE route's free_energy_value materializes alpha_reg=R_h. Either way the descent
            # direction is correct. b0_h/c0_h are the hyper-prior's own
            # precision shape (NOT alpha's b0/c0). NOTE: under state_dependent, value=cfg.lambda_h is
            # IGNORED (alpha_state_dependent reads only b0_h/c0_h); the coupling magnitude is c0_h/(b0_h+KL),
            # and cfg.lambda_h then acts ONLY as the channel-on gate -- it does not scale the s coupling.
            renyi_order=cfg.renyi_order,   value=cfg.lambda_h,          lambda_alpha_mode=lambda_h_mode,
            b0=_as_coeff(cfg.b0_h, s_mu.device), c0=_as_coeff(cfg.c0_h, s_mu.device),
            lambda_beta=cfg.lambda_gamma,
            kl_max=cfg.kl_max,             eps=cfg.eps,
            sigma_max=cfg.sigma_max,       e_sigma_q_trust=cfg.e_sigma_q_trust,
            e_mu_q_trust=cfg.e_mu_q_trust, mu_trust_mode=cfg.mu_trust_mode,
            include_attention_entropy=cfg.include_attention_entropy,
            gradient_mode=cfg.gradient_mode,
            # Thread the mean-arm preconditioner selection (audit 2026-07-05 m11): without this the
            # s-refine silently ran the default 'fisher' even under e_step_mu_precond='raw',
            # contaminating the B3/EXP-14 mean-arm ablation (raw belief channel, Fisher s channel).
            e_step_mu_precond=cfg.e_step_mu_precond,
            family=cfg.family,
            divergence_family=cfg.divergence_family,
            phi_precond_mode=cfg.phi_precond_mode,
            phi_retract_mode=cfg.phi_retract_mode,
            spd_retract_mode=cfg.spd_retract_mode,
            # SHARED connection regime for the s-channel (PB-11): the model channel refines under
            # cfg.transport_mode with the SAME learned connection the belief E-step uses, threaded via
            # _model_channel_connection_kwargs. The stateful regime_ii/covariant Omega is rebuilt each
            # iteration from the CHANNEL-LOCAL s means/covariances (e_step reads belief.mu/sigma, which
            # here are s_mu/s_sigma), not the belief means; the link modes read only connection_L. phi0
            # is still held fixed (e_phi_lr=0). Flat -> every connection None == the byte-identical pure
            # path. share_refine_s_transport (the flat prebuilt hoist) is gated to flat at its caller,
            # so a non-flat mode never receives a flat prebuilt transport.
            transport_mode=cfg.transport_mode,
            gauge_parameterization=cfg.gauge_parameterization,
            **self._model_channel_connection_kwargs(),
            e_step_gradient=e_step_gradient,
            oracle_unroll_grad=cfg.oracle_unroll_grad,
            # Tier-1 transport perf toggles: the s-channel E-step shares the flat transport
            # numerics with the belief channel (all default OFF/byte-identical).
            transport_mean_per_head=cfg.transport_mean_per_head,
            compact_phi_block_transport=self._compact_phi_blocks_enabled(),
            reuse_pairwise_kl_stats=cfg.reuse_pairwise_kl_stats,
            exp_fp64_mode=cfg.exp_fp64_mode,
            exp_fp64_norm_threshold=cfg.exp_fp64_norm_threshold,
            transport_chart_max_norm=cfg.transport_chart_max_norm,
            transport_status=self._transport_status,
            e_step_update=cfg.e_step_update,
            mm_damping=cfg.mm_damping,
            randomize_e_steps=cfg.randomize_e_steps,
            e_steps_min=cfg.e_steps_min,
            e_steps_max=cfg.e_steps_max,
            e_steps_backprop_last=cfg.e_steps_backprop_last,
            e_step_halt_tol=cfg.e_step_halt_tol,
            log_prior=gamma_log_prior,
            rope=rope,
            rope_on_cov=cfg.rope_full_gauge,
            rope_on_value=cfg.rope_on_value,
            prebuilt_transport=prebuilt_transport,
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
        belief_phi = self._apply_pos_phi(enc.phi[0]).unsqueeze(0)    # (1, N, n_gen) belief frame
        phi0 = self._resolve_model_frame(token_ids[:1], belief_phi) # (1, N, n_gen) model frame
        rope = self._rope_rotation(token_ids.shape[1], token_ids.device)
        return self._refine_s(token_ids[:1], phi0, rope=rope)        # (1, N, K) x2

    def forward_beliefs(
        self,
        token_ids:      torch.Tensor,                    # (B, N) integer token ids

        *,
        return_logits:  bool           = False,          # also decode logits; else logits is None
        decode_last:    bool           = False,          # decode only the last position as (B, 1, V)
        capture:        Optional[MStepCapture]       = None,   # out-param: M-step intermediates (q*, final-block prior, raw out)
        estep_grad_out: Optional[EStepGradientOutput] = None,   # diag out-param: E-step belief-grad norms (forwarded)
    ) -> 'Tuple[BeliefState, Optional[torch.Tensor]]':
        r"""Run the belief pipeline and return the converged belief q* (post final_norm), optionally
        with the decoded logits. This is the single belief-production seam shared by ``forward``,
        ``generate`` (via the policy layer) and the EFE scorer.

        Factors the (previously inline) sequence q_i(0) = p_i = encode(token) -> phi <- pos_phi ->
        (optional s-refine) -> precision-bias fold -> vfe_stack (L blocks of E-step belief descent) ->
        final_norm, i.e. the map from token ids to the converged Gaussian tuple q* = (mu*, Sigma*, phi*).
        The returned ``BeliefState`` carries mu = final_norm(mu_final, sigma_final), sigma = sigma_final,
        and phi = out.phi UNCHANGED (final_norm transforms only the mean), so a caller reads q*.phi for
        the M-step gauge penalty exactly as forward does. ``return_logits`` decodes p(o | q*_i) via
        ``prior_bank.decode`` inside the SAME fp32 island forward's inference branch uses. By default
        the decode covers every position and remains byte-identical to the pre-refactor
        ``forward(targets=None)`` return; ``decode_last=True`` decodes only the final belief position
        and returns logits shaped ``(B, 1, V)``.

        ``capture`` (an out-param dict, non-None only when ``mstep_self_coupling_weight>0``) is filled
        with the pre-transform intermediates the M-step self-coupling term reads and cannot recover
        from the post-final_norm belief: ``capture['converged']`` (the last block's converged
        pre-transform q*) and ``capture['final_block_prior']`` (the live ``(mu_p, sigma_p)`` entering
        that block), both written by ``vfe_stack``. ``capture['prior']`` retains the encode-time prior
        (post s-refine), and ``capture['out']`` retains the raw stack output (pre final_norm).

        Grad-transparent: it carries the SAME internal ``run = no_grad() if e_step_gradient=='detach'
        else nullcontext()`` and ``amp`` contexts forward establishes, so a grad-enabled training caller
        and a no-grad inference caller both get the identical forward value. The no-grad property used by
        the policy layer comes from the caller's ``@torch.no_grad`` scope (``generate``,
        ``rollout_beliefs``), not from this method.
        """
        B, N = token_ids.shape
        if decode_last and N <= 0:
            raise ValueError("decode_last=True requires a nonempty token context")
        beliefs = self.prior_bank.encode(token_ids)              # (B, N, K) ...
        beliefs = beliefs._replace(
            phi=self._apply_pos_phi(beliefs.phi),
            right_phi=self._pos_phi_right(beliefs.phi),
        )
        model_phi = self._resolve_model_frame(token_ids, beliefs.phi)
        diagnostic_capture = capture.get("diagnostic") if capture is not None else None
        if diagnostic_capture is not None:
            diagnostic_capture["encoded_belief"] = beliefs
            diagnostic_capture["model_phi"] = model_phi
        log_prior = self._attention_log_prior(N, token_ids.device)
        rope = self._rope_rotation(N, token_ids.device)

        # The E-step stack is vectorized over the batch (audit 4c): the belief tuple carries a
        # leading B axis through transport / gradients / retraction in one set of kernels, instead
        # of a serial per-sequence Python loop. Sequences are independent (each reads only its own
        # belief and the shared, sequence-independent log_prior), so the batched result equals the
        # per-sample result (pinned by tests/test_perf_equivalence.py::test_batched_forward_equals_per_sample).
        # lambda_beta: the belief-coupling weight (the constant cfg.lambda_beta).
        lambda_beta = self.cfg.lambda_beta
        # connection_W: the learned bilinear Regime-II edge connection (a sanctioned NN exception)
        # when transport_mode='regime_ii', else None (the flat pure path). Threaded through the
        # E-step so the loss backpropagates to it; getattr keeps the flat path's call
        # identical (None forwards a defaulted kwarg the flat builder ignores).
        connection_W = getattr(self, "connection_W", None)
        # connection_M: the learned gauge-COVARIANT (Route B) Regime-II connection when
        # transport_mode='regime_ii_covariant', else None (flat / regime_ii pure paths). Threaded
        # through the E-step like connection_W so the loss backpropagates to it.
        connection_M = getattr(self, "connection_M", None)
        # connection_L: the learned DIRECT LINK for regime_ii_link / regime_ii_link_charted, else None.
        # Threaded through the E-step like connection_W (link_alpha/link_soft_cap come from cfg in the block).
        connection_L = getattr(self, "connection_L", None)
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
        # exercised on whatever device the tokens live on. The outer autocast remains active through
        # transport construction, model-channel refinement, and vfe_stack; only the existing
        # matrix-exp/SPD kernels and the autograd oracle's inner objective/derivative builder enter
        # their own narrow fp32 islands. Decode + CE are protected separately below (their inputs are
        # .float()-ed; see _amp_off_context).
        amp = self._amp_context(token_ids.device)
        with run, amp:
            shared_omega = None
            if (self.cfg.s_frame_mode == "tied"
                    and self.cfg.share_refine_s_transport
                    and self.cfg.transport_mode == "flat"
                    and self.cfg.e_phi_lr == 0.0
                    and rope is None):
                # share_refine_s_transport (default OFF): the flat transport built inside _refine_s's
                # E-step and the one built inside the belief E-step below consume the IDENTICAL phi
                # (both hold it fixed at e_phi_lr=0), so ONE build serves both -- skipping a redundant
                # matrix-exp pair (+ its backward) per forward, and per LAYER at n_layers > 1 (phi is
                # loop-invariant when e_phi_lr==0). The rope gate matters: the belief channel folds
                # gauge-RoPE into both channels, so the unwrapped shared transport cannot serve either
                # rotated hoist. Outside the guard each e_step keeps its own authoritative build.
                from vfe3.inference.e_step import build_belief_transport
                shared_omega = build_belief_transport(
                    beliefs.phi, self.group,
                    transport_mode="flat",
                    gauge_parameterization=self.cfg.gauge_parameterization, omega=beliefs.omega,
                    right_phi=beliefs.right_phi,
                    reflection=beliefs.reflection if beliefs.reflection is not None else None,   # phi-path reflection fold (None -> byte-identical)
                    clamp_monitor=self.cfg.transport_clamp_monitor,
                    # Tier-1 transport perf toggles: the shared build must carry the same island
                    # keying / per-head mean flag the per-e_step hoists would have used.
                    transport_mean_per_head=self.cfg.transport_mean_per_head,
                    compact_phi_block_transport=self._compact_phi_blocks_enabled(),
                    exp_fp64_mode=self.cfg.exp_fp64_mode,
                    exp_fp64_norm_threshold=self.cfg.exp_fp64_norm_threshold,
                    validity_max_norm=self.cfg.transport_chart_max_norm,
                    exactness_out=self._transport_status,
                )
            s_belief = None
            if self.cfg.s_e_step:
                # Live model channel: refine s (phi0 fixed), then anchor the belief to it -- q0 and
                # the belief prior (mu_p, sigma_p) both become the refined s1. The belief E-step
                # self-couples to its prior every iteration, so s reaches mu_final even at n_e_steps=1.
                s_mu1, s_sigma1 = self._refine_s(
                    token_ids, model_phi, e_step_gradient=e_step_gradient,
                    rope=rope, prebuilt_transport=shared_omega)
                s_belief = (s_mu1, s_sigma1)
                beliefs = beliefs._replace(mu=s_mu1, sigma=s_sigma1)
            # Effective belief-channel attention prior: fold the DETACHED precision-weighted reliability
            # bias -log(b0 + tr Sigma_j) (cfg.precision_weighted_attention) and, under
            # cfg.gamma_as_beta_prior, the DETACHED hierarchical gamma prior onto the RAW
            # _attention_log_prior. Both folds are default-OFF and detached -> the closed-form belief
            # kernel treats the result as a fixed prior (exact). Captured ONCE at this fixed pre-stack
            # seam (precision_sigma is the belief sigma ENTERING the block, post s-refine: an intentional
            # fixed encode-time reliability prior held across the E-step, NOT a per-iteration one --
            # r2 id21) so the reflection/two-hop scorers reuse the SAME builder and score the SAME
            # objective (audit PB-12). _effective_beta_log_prior is the single authoritative constructor.
            beta_prior_context = EffectiveBetaPriorContext(
                token_ids=token_ids,
                base_log_prior=log_prior,
                precision_sigma=beliefs.sigma,
                model_phi=model_phi,
                s_mu=(s_belief[0] if s_belief is not None else None),
                s_sigma=(s_belief[1] if s_belief is not None else None),
            )
            log_prior = self._effective_beta_log_prior(beliefs, beta_prior_context)
            if capture is not None:
                capture["beta_prior_context"] = beta_prior_context
            if diagnostic_capture is not None:
                diagnostic_capture["initial_belief"] = beliefs
                diagnostic_capture["s_belief"] = (
                    self.prior_bank.encode_s(token_ids)
                    if s_belief is None and self._model_channel_active else s_belief)
                diagnostic_capture["rope"] = rope
                diagnostic_capture["log_prior"] = log_prior
            # capture: the last block's CONVERGED (pre-transform) belief q*, consumed by the
            # M-step self-coupling term in forward (manuscript: the self-term reads q*, not the
            # transformed handoff; audit 2026-06-09 overnight F19). None when the term is off.
            grad_rec: Optional[EStepGradientRecord] = (
                {} if estep_grad_out is not None else None
            )                                                       # E-step belief-grad capture (gated, off by default)
            out = vfe_stack(
                beliefs, beliefs.mu, beliefs.sigma, self.group, self.cfg,
                log_prior=log_prior, block_norm=self.block_norm,
                head_mixer=self.head_mixer, cg_coupling=self.cg_coupling,
                lambda_beta=lambda_beta,
                connection_W=connection_W, connection_M=connection_M,
                connection_L=connection_L,
                e_step_gradient=e_step_gradient,
                rope=rope, rope_on_cov=self.cfg.rope_full_gauge,
                rope_on_value=self.cfg.rope_on_value,
                capture=capture, grad_record=grad_rec,
                transport_status=self._transport_status,
                prebuilt_transport=shared_omega,
                gauge_parameterization=self.cfg.gauge_parameterization,
                kappa_beta_override=self.effective_kappa_beta(token_ids.device))
        if estep_grad_out is not None:                           # one host sync, only when requested
            for _gk in ("mu", "sigma", "phi"):
                _gv = grad_rec.get(_gk) if grad_rec is not None else None
                estep_grad_out[_gk] = float(_gv) if _gv is not None else 0.0
        mu_final = out.mu                                        # (B, N, K); head mixer (if any) applied PER BLOCK
        sigma_final = out.sigma                                  # inside vfe_stack now, not post-stack

        if self.final_norm is not None:                          # config-selected final norm (cached)
            mu_final = self.final_norm(mu_final, sigma_final)

        belief = BeliefState(mu=mu_final, sigma=sigma_final, phi=out.phi, omega=out.omega,   # carry the GL(K) frame under omega_direct (None on the phi path)
                             reflection=out.reflection, right_phi=out.right_phi)              # carry exact positional right factor / reflection
        if capture is not None:
            # M-step out-param enrichment: vfe_stack already wrote capture['converged'] (q*); add the
            # encode-time prior p (post s-refine) and the raw pre-final_norm stack output for callers
            # that need the complete pre-normalization pipeline state.
            capture["prior"] = beliefs
            capture["out"]   = out
        logits = None
        if return_logits:
            mu_decode = mu_final[:, -1:] if decode_last else mu_final
            sigma_decode = sigma_final[:, -1:] if decode_last else sigma_final
            with self._amp_off_context(token_ids.device):
                logits = self.prior_bank.decode(mu_decode.float(), sigma_decode.float())
        return belief, logits

    # ----------------------------------------------------------------------------------------------
    # Learnable discrete reflection: DeltaF-gated Metropolis flip (fixed-belief block move). The SAME
    # sweep learns the det-sign under BOTH parameterizations -- it flips omega_embed under
    # gauge_parameterization='omega_direct' + omega_reflection='metropolis', or the per-token
    # reflection_sign buffer under gauge_parameterization='phi' + phi_reflection='metropolis'. Only the
    # per-token flip and the trial-frame construction differ; the sweep/accept/seed structure is shared.
    # See docs/superpowers/specs/2026-07-08-omega-direct-metropolis-detsign-design.md (omega) and
    # docs/superpowers/specs/2026-07-08-phi-reflection-design.md Sec.5 (phi).
    # ----------------------------------------------------------------------------------------------
    def _reflection_metropolis_mode(self) -> str:
        r"""Resolve the active learnable-reflection Metropolis mode: ``'omega'`` iff
        (``gauge_parameterization=='omega_direct'`` and ``omega_reflection=='metropolis'``), ``'phi'``
        iff (``gauge_parameterization=='phi'`` and ``phi_reflection=='metropolis'``), else ``''`` (no-op).

        The two are mutually exclusive by construction: omega_direct has no ``reflection_sign`` buffer,
        phi has no ``omega_embed`` frame, and config validation rejects ``omega_reflection=='metropolis'``
        under phi (so the omega branch here fires exactly when the pre-generalization guard
        ``omega_reflection=='metropolis'`` did -- byte-identical dispatch on the omega path)."""
        cfg = self.cfg
        if cfg.gauge_parameterization == "omega_direct" and cfg.omega_reflection == "metropolis":
            return "omega"
        if cfg.gauge_parameterization == "phi" and cfg.phi_reflection == "metropolis":
            return "phi"
        return ""

    def _metropolis_prepare(
        self,
        token_ids: torch.Tensor,             # (B, N) integer token ids

        *,
        mode:      Optional[str] = None,     # 'omega' | 'phi'; None -> resolve from cfg
    ) -> MetropolisObjectiveContext:
        r"""Fixed q/p state for the (fixed-belief) Metropolis det-sign F-eval, as ONE
        :class:`MetropolisObjectiveContext` (audit PB-12).

        Runs the belief pipeline once under no_grad and captures everything the scorer needs to
        evaluate the EXACT active fixed-belief objective, held FIXED across the sweep so only the frame
        is flipped (``belief.omega`` under 'omega', ``belief.reflection`` under 'phi'):

          - ``belief`` -- the final block's converged q* carrying the frame the E-step minimized
            (``capture['converged']``, pre-final_norm; falls back to the returned post-norm belief only
            if that lacks the mode's frame field). The sweep initializes ``f_cur`` and the sequential
            current state from THIS exact object before constructing any trial.
          - ``mu_p``/``sigma_p`` -- the HANDOFF-ADJUSTED prior entering the FINAL block
            (``capture['final_block_prior']``), NOT the encode prior (they coincide only at n_layers=1),
            so the self-coupling and coupling terms score against the same p the final block descended.
          - ``tau`` -- the final block's ENTRY-derived query-adaptive temperature
            (``capture['final_block_tau']``), the tau that PRODUCED q*, not a tau recomputed from the
            converged sigma.
          - ``rope`` -- the positional RoPE rotation for this token length (None when off).
          - ``prior`` -- the fixed pre-stack :class:`EffectiveBetaPriorContext`; the scorer rebuilds the
            candidate-dependent effective prior per proposal from it and the trial frame.
        """
        mode = mode or self._reflection_metropolis_mode()
        with torch.no_grad():
            cap: Dict = {}
            belief, _ = self.forward_beliefs(token_ids, capture=cap)
            conv      = cap.get("converged")
            if mode == "omega":
                belief_f = conv if (conv is not None and conv.omega is not None) else belief
            else:                                     # phi: prefer the converged frame carrying the sign
                belief_f = conv if (conv is not None and conv.reflection is not None) else belief
            mu_p, sigma_p = cap["final_block_prior"]  # handoff-adjusted prior ENTERING the final block
            rope = self._rope_rotation(token_ids.shape[-1], token_ids.device)
        return MetropolisObjectiveContext(
            token_ids=token_ids, mu_p=mu_p, sigma_p=sigma_p, belief=belief_f,
            tau=cap["final_block_tau"], rope=rope, prior=cap["beta_prior_context"])

    def _metropolis_free_energy(
        self,
        belief:  BeliefState,                        # candidate belief carrying .omega (B,N,K,K) or .reflection (B,N)
        context: MetropolisObjectiveContext,         # fixed q/p state (prior, tau, prior moments, rope)

        *,
        mode:    Optional[str] = None,               # 'omega' | 'phi'; None -> resolve from cfg
    ) -> float:
        r"""Scalar free energy of a FIXED belief under the EXACT active objective, summed over the batch
        (sequences are independent). Audit PB-12: scores the SAME F the E-step descended.

        Rebuilds the candidate-dependent effective attention prior via
        ``_effective_beta_log_prior(belief, context.prior)`` -- the precision fold reads the FIXED
        pre-stack ``context.prior.precision_sigma`` (frame-blind; identical for current and trial), and
        only the tied-gamma fold varies with the proposed frame -- then evaluates ``free_energy_value``
        with the FIXED captured tau (``context.tau``, the final block's entry-derived query-adaptive
        temperature), the FIXED handoff-adjusted prior moments (``context.mu_p``/``context.sigma_p``),
        the honored ``lambda_twohop``, and the ACTIVE transport/RoPE/numerics controls
        (``transport_mode``, ``connection_W``/``M``/``L``, cocycle/link/clamp, ``transport_mean_per_head``,
        ``context.rope`` + ``rope_on_cov``/``rope_on_value``, ``exp_fp64_mode``/``exp_fp64_norm_threshold``).
        Current and trial thus differ ONLY in the proposed frame/reflection, so the Metropolis DeltaF is
        the exact change in the joint F under the block move.

        ONE batched ``free_energy_value`` call (audit 2026-07-12 N7): the softmax/prior/energy ops are
        batch-broadcasting and the final reduction sums over every leading axis, so the batched scalar
        IS the per-sequence sum (sequences are independent; attention rows never mix batch elements) --
        one host sync per F. The absolute F need not equal the training loss, but the DeltaF is exact.

        Caveat (optimizer-moment staleness, see :meth:`metropolis_omega_step`): the accepted flip is
        written to the source table AFTER ``optimizer.step()``, so AdamW's moment buffers for a flipped
        row stay stale for one step -- unchanged by this objective-parity work."""
        from vfe3.inference.e_step import free_energy_value
        cfg, grp = self.cfg, self.group
        mode      = mode or self._reflection_metropolis_mode()
        gp        = "omega_direct" if mode == "omega" else "phi"
        dev       = belief.mu.device
        b0        = _as_coeff(cfg.b0, dev)
        c0        = _as_coeff(cfg.c0, dev)
        with torch.no_grad():
            # Per-candidate effective prior: the SAME authoritative builder the forward E-step used, so
            # the folded precision/tied-gamma prior the belief converged under is reproduced exactly.
            log_prior = self._effective_beta_log_prior(belief, context.prior)
            total = free_energy_value(
                belief, context.mu_p, context.sigma_p, grp,
                tau=context.tau, renyi_order=cfg.renyi_order, value=cfg.lambda_alpha, b0=b0, c0=c0,
                lambda_beta=cfg.lambda_beta, kl_max=cfg.kl_max, eps=cfg.eps,
                lambda_twohop=cfg.lambda_twohop,
                include_attention_entropy=cfg.include_attention_entropy,
                family=cfg.family, divergence_family=cfg.divergence_family,
                lambda_alpha_mode=cfg.lambda_alpha_mode,
                gauge_parameterization=gp, log_prior=log_prior,
                transport_mode=cfg.transport_mode,
                connection_W=getattr(self, "connection_W", None),
                connection_M=getattr(self, "connection_M", None),
                connection_L=getattr(self, "connection_L", None),
                cocycle_relaxation=cfg.cocycle_relaxation,
                link_alpha=cfg.link_alpha, link_soft_cap=cfg.link_soft_cap,
                clamp_monitor=cfg.transport_clamp_monitor,
                transport_mean_per_head=cfg.transport_mean_per_head,
                rope=context.rope, rope_on_cov=cfg.rope_full_gauge, rope_on_value=cfg.rope_on_value,
                exp_fp64_mode=cfg.exp_fp64_mode, exp_fp64_norm_threshold=cfg.exp_fp64_norm_threshold,
            )
            if cfg.lambda_h > 0.0 or cfg.lambda_gamma > 0.0:
                s_belief = (
                    (context.prior.s_mu, context.prior.s_sigma)
                    if context.prior.s_mu is not None and context.prior.s_sigma is not None
                    else None
                )
                total = total + self._model_channel_free_energy(
                    context.token_ids, belief, s_belief=s_belief)
        return total.item()

    def _metropolis_trial_belief(
        self,
        belief:    BeliefState,              # fixed belief carrying .omega (B, N, K, K) or .reflection (B, N)
        token_ids: torch.Tensor,             # (B, N) integer token ids

        token_id:  int,                      # the token whose det-sign is flipped
        *,
        mode:      Optional[str] = None,     # 'omega' | 'phi'; None -> resolve from cfg
    ) -> BeliefState:
        r"""Trial belief with the frame at every ``token_ids == token_id`` position reflected, all other
        positions and the beliefs (mu, sigma) held FIXED.

        Under ``'omega'`` mode the frame is left-multiplied by the canonical reflection
        R = reflection_element(K) (det R = -1) on dense storage. Compact block-diagonal storage applies
        the equivalent operator without allocating K x K: reflection_element(d) multiplies block 0 and
        every other block remains unchanged, matching the source-table flip in
        :meth:`_flip_omega_embed_row`. Under ``'phi'`` mode the per-token reflection sign is negated
        (s -> -s) at the masked positions; the §3 fold in ``build_belief_transport`` then applies
        R_i Omega_ij R_j at F-eval, matching the source-buffer flip in :meth:`_flip_reflection_sign_row`."""
        mode = mode or self._reflection_metropolis_mode()
        mask = (token_ids == token_id)                                                       # (B, N)
        if mode == "omega":
            from vfe3.geometry.generators import reflection_element
            if isinstance(belief.omega, CompactBlockElement):
                if belief.omega.tied:
                    raise ValueError("compact Metropolis reflection requires untied block storage")
                r = reflection_element(
                    belief.omega.block_dim, dtype=belief.omega.dtype, device=belief.omega.device)
                trial_omega = belief.omega.clone()
                selected = trial_omega.blocks[mask].clone()                                  # (#,H,d,d)
                selected[:, 0] = r @ selected[:, 0]                                           # block-0 probe
                trial_omega.blocks[mask] = selected
            else:
                k = belief.omega.shape[-1]
                r = reflection_element(k, dtype=belief.omega.dtype, device=belief.omega.device)
                trial_omega = belief.omega.clone()
                trial_omega[mask] = torch.einsum(
                    "kl,...lm->...km", r, trial_omega[mask])                                 # R @ U
            return belief._replace(omega=trial_omega)
        trial_reflection = belief.reflection.clone()                                         # phi: s -> -s at masked
        trial_reflection[mask] *= -1.0
        return belief._replace(reflection=trial_reflection)

    def _metropolis_delta_f(
        self,
        context:   MetropolisObjectiveContext,   # fixed q/p state + current belief (context.belief)

        token_id:  int,                          # token whose det-sign flip is scored
        *,
        mode:      Optional[str] = None,         # 'omega' | 'phi'; None -> resolve from cfg
    ) -> float:
        r"""Exact fixed-belief DeltaF = F(trial) - F(current) for flipping ``token_id``'s det-sign,
        scored against the EXACT active objective (audit PB-12).

        The sweep in :meth:`metropolis_omega_step` carries F_cur forward for efficiency; this helper
        recomputes both terms from ``context.belief`` so the exact-DeltaF regression test can compare it
        against an independent source-flip (pinning the masked trial-belief flip == the source
        ``omega_embed`` / ``reflection_sign`` flip). Current and trial reuse the SAME fixed
        precision/tau/prior moments; only the proposed frame/reflection differs."""
        mode  = mode or self._reflection_metropolis_mode()
        belief = context.belief
        trial = self._metropolis_trial_belief(belief, context.token_ids, token_id, mode=mode)
        return (self._metropolis_free_energy(trial, context, mode=mode)
                - self._metropolis_free_energy(belief, context, mode=mode))

    def _flip_omega_embed_row(
        self,
        R:        torch.Tensor,              # full reflection_element(K), or compact reflection_element(d)

        token_id: int,                       # source-table row (token id) to flip
    ) -> None:
        r"""Left-multiply the stored frame of ``token_id`` by the reflection R IN PLACE, toggling its
        det-sign. Respects the storage layout (mirrors the init_seed seeding in prior_bank): full
        (V, K, K) -> R_K @ row; compact (V, H, d, d) -> R_d applied to block 0 only
        (blocks 1..H-1 are unchanged under the represented full reflection).

        This in-place write happens after optimizer.step() and does not touch AdamW's exp_avg/
        exp_avg_sq buffers for this row -- the moment staleness caveat, see :meth:`metropolis_omega_step`."""
        pb = self.prior_bank
        with torch.no_grad():
            if getattr(pb, "_omega_compact", False):
                # [token_id, 0] indexes block 0 of the untied (V, H, d, d) table -- correct only for
                # that layout. A TIED (V, d, d) table has no block axis, so [token_id, 0] would index
                # row 0 of the shared block and silently corrupt the frame instead of flipping it.
                # Currently unreachable: tied groups (tied_block_glk) are rejected at config for
                # omega_reflection='metropolis' (same _REFLECT_OK gate as init_seed). Assert loud
                # rather than let a future widening of that gate corrupt silently.
                assert pb.omega_embed.dim() == 4, (
                    "compact det-sign flip assumes untied (V,H,d,d); tied (V,d,d) is gated out at config")
                d = pb.omega_embed.shape[-1]                            # compact block size
                if R.shape != (d, d):
                    raise ValueError(
                        f"compact reflection must have shape ({d}, {d}), got {tuple(R.shape)}")
                pb.omega_embed[token_id, 0] = R @ pb.omega_embed[token_id, 0]
            else:
                pb.omega_embed[token_id] = R @ pb.omega_embed[token_id]

    def _flip_reflection_sign_row(
        self,
        token_id: int,                       # source-buffer row (token id) whose reflection sign is flipped
    ) -> None:
        r"""Negate the stored per-token reflection sign of ``token_id`` IN PLACE (s -> -s), toggling its
        det-sign on the phi path. The phi-mode parallel of :meth:`_flip_omega_embed_row`: the sign lives
        in the (V,) ``reflection_sign`` buffer (block 0's diag(-1,1,...) at the K level, per the §3
        reflection fold). It is a non-differentiable discrete buffer flipped ONLY by the Metropolis
        move, so the optimizer-moment-staleness caveat of :meth:`_flip_omega_embed_row` does NOT apply
        (a buffer carries no AdamW exp_avg/exp_avg_sq)."""
        pb = self.prior_bank
        with torch.no_grad():
            pb.reflection_sign[token_id] *= -1.0

    def metropolis_omega_step(
        self,
        token_ids: torch.Tensor,             # (B, N) integer token ids

        *,
        generator: torch.Generator,          # seeded RNG for the accept draws (reproducibility)
    ) -> dict:
        r"""One DeltaF-gated Metropolis sweep over the discrete det-sign of the stored frames of the
        unique tokens in ``token_ids``. Dispatches on :meth:`_reflection_metropolis_mode`: no-op
        (returns ``{}``) unless a learnable-reflection mode is active -- ``'omega'`` flips the stored
        GL(K) frame ``omega_embed`` (gauge_parameterization='omega_direct' + omega_reflection=
        'metropolis'), ``'phi'`` flips the per-token ``reflection_sign`` buffer (gauge_parameterization=
        'phi' + phi_reflection='metropolis'). Both share this sweep; only the per-token flip and the
        trial-frame construction differ. The beliefs are held FIXED (a Metropolis-within-Gibbs block
        move on the joint F): each proposed sign flip is an orthogonal involution (dense
        ``reflection_element(K)`` or its compact block-0 ``reflection_element(d)`` representation),
        both with determinant -1. The proposal is symmetric, so the Hastings ratio reduces to the
        plain Metropolis acceptance probability min(1, exp(-DeltaF / T)). On acceptance the source table
        (``omega_embed`` / ``reflection_sign``) is mutated in place and the flipped belief is carried
        forward, so the next token's DeltaF is measured against the post-accept state (a correct MCMC
        chain). Everything runs under no_grad. Returns a small stats dict (proposed/accepted counts,
        mean DeltaF) for logging. See docs/superpowers/specs/2026-07-08-omega-direct-metropolis-detsign-
        design.md (omega) and docs/superpowers/specs/2026-07-08-phi-reflection-design.md Sec.5 (phi).

        Objective parity (audit PB-12): DeltaF is scored against the EXACT active fixed-belief objective
        -- the folded precision/tied-gamma attention prior, the two-hop coupling block, the query-adaptive
        (final-block entry-derived) tau, the handoff-adjusted final-block prior, and the active
        transport/RoPE numerics all match the F the E-step descended (see :meth:`_metropolis_free_energy`
        and :meth:`_metropolis_prepare`).

        Caveat (optimizer-moment staleness, final-review Fix C): ``omega_embed`` is mutated in place
        AFTER ``optimizer.step()`` for the iteration, so AdamW's ``exp_avg``/``exp_avg_sq`` moment
        buffers for a flipped row are not reflected and stay stale for one step (bounded, self-
        correcting in the near-inert flip regime; see spec Sec.7).

        # TODO(STE): straight-through-gradient variant of the learnable det-sign -- propose per-token
        # sign flips accepted through a straight-through estimator (biased but differentiable) instead
        # of this DeltaF-gated Metropolis accept/reject. See GL(K)_attention.tex eq:ok_transport.
        """
        cfg = self.cfg
        mode = self._reflection_metropolis_mode()
        if not mode:
            return {}
        temp = float(cfg.omega_metropolis_temperature)
        with torch.no_grad():
            context = self._metropolis_prepare(token_ids, mode=mode)
            belief  = context.belief                                   # current state: init from the exact prepared object
            R = None
            if mode == "omega":                                         # phi flips a scalar sign, no R needed
                from vfe3.geometry.generators import reflection_element
                reflection_dim = (
                    belief.omega.block_dim
                    if isinstance(belief.omega, CompactBlockElement)
                    else belief.omega.shape[-1]
                )
                R = reflection_element(
                    reflection_dim, dtype=belief.omega.dtype, device=belief.omega.device)
            f_cur = self._metropolis_free_energy(belief, context, mode=mode)
            proposed = accepted = 0
            dfs: list = []
            for tid in torch.unique(token_ids).tolist():
                trial   = self._metropolis_trial_belief(belief, context.token_ids, tid, mode=mode)
                f_trial = self._metropolis_free_energy(trial, context, mode=mode)
                df      = f_trial - f_cur
                dfs.append(df)
                proposed += 1
                u = torch.rand((), generator=generator).item()          # one draw per proposal (deterministic RNG use)
                if df <= 0.0 or u < math.exp(-df / temp):               # min(1, exp(-df/T)) accept
                    accepted += 1
                    f_cur  = f_trial
                    belief = trial                                      # carry the flipped belief forward
                    if mode == "omega":
                        self._flip_omega_embed_row(R, int(tid))         # mutate the source table in place
                    else:
                        self._flip_reflection_sign_row(int(tid))        # mutate the source buffer in place
            return {"proposed": proposed, "accepted": accepted,
                    "mean_delta_f": (sum(dfs) / len(dfs)) if dfs else 0.0}

    @torch.no_grad()
    def rollout_beliefs(
        self,
        token_ids:     torch.Tensor,                     # (B, N) context ids (the action prefix)  -> D

        *,
        return_logits: bool          = True,             # continuation scoring needs the decode    -> A
        decode_last:   bool          = False,            # decode only terminal logits as (B, 1, V)
    ) -> 'Tuple[BeliefState, Optional[torch.Tensor]]':
        r"""Public no-grad belief rollout: the active-inference contract's D (initial belief from the
        current context) and the one-step B (transition rule) building block. A single forward of
        ``token_ids`` through the shared belief seam under no_grad, returning (q*, logits). Appending a
        candidate ACTION token to ``token_ids`` and re-calling realizes one transition q*_t -> q*_{t+1};
        iterating it H times is the fixed-horizon rollout. The environment's response to a committed
        action is appended by the generation loop AFTER selection, never inside the scored rollout.
        By default it returns the SAME tensors ``forward`` would, so it adds no new numerical path;
        ``decode_last=True`` preserves the complete belief and decodes only terminal ``(B, 1, V)`` logits.
        """
        return self.forward_beliefs(
            token_ids, return_logits=return_logits, decode_last=decode_last)

    def forward(
        self,
        token_ids: torch.Tensor,         # (B, N) integer token ids
        targets:   Optional[torch.Tensor] = None,   # (B, N) next-token ids (-100 = ignore)

        *,
        estep_grad_out: Optional[EStepGradientOutput] = None,   # diag out-param: filled with the E-step belief-grad norms
    ) -> 'torch.Tensor | Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]':
        r"""Forward pass; returns logits, or (logits, loss, ce) when targets are given.

        On the fused-chunked training path logits is None (callers discard it there), hence the
        Optional first element of the training tuple. When ``estep_grad_out`` (a dict) is passed, it
        is filled with the LAST-block / LAST-iteration raw E-step belief-gradient norms
        ``{'mu','sigma','phi'}`` (||grad_mu/sigma/phi|| of F over the belief tuple) -- the E-step
        analogue of the M-step per-group grad norms; default None is zero-overhead and byte-identical.

        Belief production is factored into :meth:`forward_beliefs` (the shared seam); this method is the
        decode + cross-entropy + M-step assembly on top of it."""
        if targets is None:
            # Inference: logits via the shared belief seam. estep_grad_out is forwarded so the
            # diagnostic still fills on this path (byte-identical to the pre-refactor return).
            return self.forward_beliefs(token_ids, return_logits=True, estep_grad_out=estep_grad_out)[1]
        # Training path: produce the converged belief q* (no (B,N,V) logits) via the shared seam, then
        # run the existing decode + cross-entropy + M-step assembly reading belief.mu / sigma / phi.
        # cap (non-None only when the M-step self-coupling term is on) is filled by forward_beliefs
        # with the converged q*, the live final-block prior, the encode-time prior, and the raw
        # pre-final_norm stack output.
        cap: Optional[MStepCapture] = (
            {} if (self.cfg.mstep_self_coupling_weight > 0.0
                   or self.cfg.cg_energy_weight > 0.0) else None
        )
        if cap is not None and self.cfg.cg_energy_weight > 0.0:
            # Initialize the CG moment-energy lists ONLY when the regularizer is on. A capture
            # allocated solely for M-step self-coupling leaves these keys absent, so vfe_block never
            # appends to (or stacks) a CG list (PB-13).
            cap["cg_moment_energy_rows"] = []
            cap["cg_pre_moments"] = []
        belief, _ = self.forward_beliefs(token_ids, return_logits=False,
                                         capture=cap, estep_grad_out=estep_grad_out)
        mu_final, sigma_final = belief.mu, belief.sigma          # (B, N, K) post final_norm; sigma = out.sigma

        # Decode + cross-entropy fp32 island. The decode matmul (_decode_diagonal) reconstructs the
        # Mahalanobis term via a catastrophically-cancelling subtraction pinned at atol-1e-3, and CE
        # is a log-sum-exp over V=50257; both MUST stay fp32 even when amp_dtype is on. The
        # load-bearing guard is the explicit .float() on the inputs (autocast(enabled=False) only
        # blocks FURTHER downcasting -- it does NOT upcast a tensor that already arrived bf16 from
        # the autocast E-step), mirroring retraction.py's in-island sigma.float(). On the default
        # fp32 path .float() is a value-identical no-op AND the island is a nullcontext (see
        # _amp_off_context), so this block is byte-identical to the no-AMP build.
        # Select the same active decoder record PriorBank.decode uses: the configured prior-bank
        # mode on the geometric path, or linear on the no-prior ablation. Capability routing and
        # fused dispatch then come from that one record, so an unrelated configured mode cannot
        # control the active linear path. logits is None on the fused branch by design.
        active_decode_mode = self.cfg.decode_mode if self.cfg.use_prior_bank else "linear"
        decode_registration = get_decode_registration(active_decode_mode)
        fused_chunked = (
            targets is not None
            and decode_registration.supports_chunked
        )
        if fused_chunked:
            with self._amp_off_context(token_ids.device):
                if self.cfg.use_prior_bank:
                    ce = decode_registration.fused_ce(
                        self.prior_bank, mu_final.float(), sigma_final.float(), targets,
                        z_loss_weight=self.cfg.z_loss_weight,
                    )
                else:
                    ce = decode_registration.fused_ce(
                        self.prior_bank, mu_final.float(), targets,
                        z_loss_weight=self.cfg.z_loss_weight,
                    )
            logits = None                                        # no (B, N, V) tensor on the fused path
        else:
            with self._amp_off_context(token_ids.device):
                logits = self.prior_bank.decode(mu_final.float(), sigma_final.float())   # (B, N, V) fp32
            # targets is guaranteed not None here (the inference path returned via forward_beliefs above).
            with self._amp_off_context(token_ids.device):
                flat_logits = logits.reshape(-1, self.cfg.vocab_size).float()
                flat_targets = targets.reshape(-1)
                # Branchless masked mean (no host sync to test .any()): sum-reduced CE over the
                # non-ignored tokens divided by a device-side clamped count. An all-ignore microbatch
                # gives 0/1 = a finite grad-connected 0; F.cross_entropy's default mean would be
                # 0/0 = NaN there, poisoning logging / NaN-guards / grad-accum means.
                n_valid = (flat_targets != -100).sum().clamp_min(1)
                ce = F.cross_entropy(flat_logits, flat_targets, ignore_index=-100,
                                     reduction="sum") / n_valid
                # z-loss (m20): the four fused chunked kernels add w * mean(logsumexp^2); the dense
                # branch dropped it, so z_loss_weight>0 was silently inert on the default diagonal/full
                # decode. Mirror the chunked formula with the same clamped n_valid; the >0 guard keeps
                # the default (0.0) byte-identical.
                if self.cfg.z_loss_weight > 0.0:
                    valid = (flat_targets != -100).to(flat_logits.dtype)
                    lse = torch.logsumexp(flat_logits, dim=-1)
                    ce = ce + self.cfg.z_loss_weight * (lse ** 2 * valid).sum() / n_valid
        loss = ce
        if self.cfg.mass_phi > 0.0:
            # M-step gauge-frame penalty (manuscript Algorithm 1 M-step loss): regularizes the
            # CONVERGED output phi -> backprops to the learned prior table phi_embed. This is the
            # outer-loss role; mass_phi ALSO enters the inner phi E-step objective (e_step:
            # phi_alignment_loss), shaping the inference trajectory. Both roles are in the
            # manuscript algorithm (E-step phi gradient and M-step loss both carry alpha_phi/2||phi||^2).
            loss = loss + 0.5 * self.cfg.mass_phi * (belief.phi ** 2).mean()
        if self.cfg.mstep_self_coupling_weight > 0.0:
            # M-step self-coupling regularizer (manuscript Algorithm 1, GL(K)_attention.tex:2111):
            # L += alpha_hat * sum_i alpha_i D(q_i*||p_i), the alpha-weighted self-coupling of the
            # CONVERGED variational belief q* (captured by vfe_block BEFORE head_mixer /
            # cg_coupling / block_norm -- the belief the E-step's F was actually minimized over,
            # which the manuscript pins the self-term to) against the live prior captured immediately
            # before the final block. With the three transform toggles
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
            # prior is alpha_i* dD/dtheta with alpha_i* held fixed -- detaching it (and dropping R)
            # is exact for the closed-form forms (constant/state_dependent/state_dependent_per_coord).
            # At constant alpha=1.0 (the default) alpha_i==1, byte-identical to the prior mean-D form.
            # Grad-connected through D (no detach on D), so it backprops to the learned prior tables,
            # like mass_phi. Capturing the final-block input preserves the exact stack recurrence at
            # every depth; at n_layers=1 it is the encode-time prior, preserving the existing path.
            from vfe3.families import get_family
            from vfe3.free_energy import self_divergence_for_alpha
            from vfe3.alpha_i import self_coupling_alpha, alpha_is_per_coord
            cfg = self.cfg
            mu_p, sigma_p = cap["final_block_prior"]            # live prior entering the final block
            fam = get_family(cfg.family)
            q_conv = cap["converged"]                           # q*: pre-transform converged belief
            self_div = self_divergence_for_alpha(               # (B, N) summed, or (B, N, K) per-coord
                fam(q_conv.mu, q_conv.sigma), fam(mu_p, sigma_p),
                alpha=cfg.renyi_order, kl_max=cfg.kl_max, eps=cfg.eps,
                divergence_family=cfg.divergence_family, lambda_alpha_mode=cfg.lambda_alpha_mode,
            )
            alpha_sc, _ = self_coupling_alpha(                  # SAME form as the E-step / diagnostics
                self_div, mode=cfg.lambda_alpha_mode, value=cfg.lambda_alpha, b0=_as_coeff(cfg.b0, cap["out"].mu.device), c0=_as_coeff(cfg.c0, cap["out"].mu.device),
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
        # HYPER-PRIOR + MODEL-COUPLING CHANNELS via the SINGLE typed hierarchical evaluator (PB-10):
        # the two scored s-channel blocks (both gated on `not s_e_step`) are assembled as zero q rows
        # plus live h/s rows through hierarchical_free_energy_terms with model_reduction='mean', and
        # added ONCE to the unchanged legacy q loss instead of two independent += increments. The
        # canonical gamma branch now carries the mathematically equal coupling-plus-entropy
        # decomposition (via _gamma_coupling_rows) rather than the fused reduced envelope, so its fp32
        # association changes (no bitwise claim on the active model channel); the all-off pure path
        # (lambda_h == lambda_gamma == 0) skips the block entirely and stays byte-identical.
        if (self.cfg.lambda_h > 0.0 or self.cfg.lambda_gamma > 0.0) and not self.cfg.s_e_step:
            loss = loss + self._model_channel_free_energy(token_ids, belief)
        if self.cfg.cg_energy_weight > 0.0:
            # CG MOMENT-ENERGY REGULARIZER (PB-13): a q-only term added to the outer objective ONCE,
            # NEVER routed through hierarchical_free_energy_terms and never reweighting the canonical
            # q/p/s/h total. Under an attached E-step estimator each block already appended its
            # grad-connected D(q_post||q_pre) rows; under the 'detach' estimator the blocks stashed the
            # detached pre-CG moments, and the shared CGCoupling.forward_moments is re-evaluated here
            # under torch.enable_grad from each fixed pair -- detached from belief inference but
            # attached to path_weights. The token-then-layer mean (each row.mean() over tokens, then
            # mean over layers) is weighted and added; the per-layer values and the assembled
            # objective_total_with_cg are reported on the diagnostic side channel only.
            from vfe3.model.cg_coupling import cg_moment_energy_rows
            cfg = self.cfg
            if cfg.effective_e_step_gradient == "detach":
                per_layer_rows = []
                with torch.enable_grad():
                    for pre_mu, pre_sigma in cap["cg_pre_moments"]:
                        res = self.cg_coupling.forward_moments(pre_mu, pre_sigma)
                        per_layer_rows.append(cg_moment_energy_rows(
                            pre_mu, pre_sigma, res.mu, res.sigma,
                            renyi_order=cfg.renyi_order, kl_max=cfg.kl_max, eps=cfg.eps,
                            family=cfg.family, divergence_family=cfg.divergence_family))
            else:
                per_layer_rows = cap["cg_moment_energy_rows"]
            if (len(per_layer_rows) != cfg.n_layers
                    or any(r.numel() == 0 for r in per_layer_rows)):
                raise RuntimeError(
                    f"cg_energy_weight>0 requires exactly n_layers={cfg.n_layers} nonempty CG "
                    f"moment-energy row tensors; got {len(per_layer_rows)} "
                    f"({[tuple(r.shape) for r in per_layer_rows]}). An empty capture must never "
                    f"reach torch.stack."
                )
            layer_means = torch.stack([rows.mean() for rows in per_layer_rows])   # (n_layers,)
            cg_moment_energy = layer_means.mean()
            loss = loss + cfg.cg_energy_weight * cg_moment_energy
            self._cg_energy_diagnostics = {
                "cg_moment_energy":        cg_moment_energy.detach(),
                "cg_moment_energy_layers": layer_means.detach(),
                "objective_total_with_cg": loss.detach(),
            }
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
        kernel is ``get_family(cfg.family)`` (PB-11): a diagonal/Laplace family scores diagonal
        (V,K)/(K,) s/r, while ``gaussian_full`` scores the full SPD s/r covariances that
        ``encode_s``/``r_parameters`` now assemble from the packed Cholesky tables; r broadcasts over
        the (B, N) token axis. :meth:`_hyper_prior_term` reduces this to its mean (the forward-loss
        scale); :meth:`diagnostics` and the s/r/h figures consume the per-token vector / its sum.
        """
        from vfe3.divergence import get_family
        from vfe3.free_energy import self_divergence, self_divergence_per_coord
        cfg = self.cfg
        pb = self.prior_bank
        family = get_family(cfg.family)
        s_mu, s_sigma = pb.encode_s(token_ids) if s_belief is None else s_belief   # (B,N,K) or (B,N,K,K)
        r_mu, r_sigma = pb.r_parameters()                           # (K,) / (K,K)
        div = self_divergence_per_coord if per_coord else self_divergence
        return div(
            family(s_mu, s_sigma), family(r_mu, r_sigma),
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
        for that cancellation). ``cfg.lambda_h == 0.0`` is an absolute gate evaluated before mode
        dispatch, including for state-dependent forms whose registry implementation ignores ``value``.
        """
        cfg = self.cfg
        if cfg.lambda_h == 0.0:
            kl_s = self._hyper_prior_kl(token_ids, s_belief=s_belief)
            return torch.zeros_like(kl_s)
        from vfe3.lambda_h_i import hyper_prior_lambda_h, lambda_h_is_per_coord
        per_coord = lambda_h_is_per_coord(cfg.lambda_h_mode)
        kl_s = self._hyper_prior_kl(token_ids, s_belief=s_belief, per_coord=per_coord)  # (B,N) or (B,N,K)
        lam, reg = hyper_prior_lambda_h(
            kl_s, mode=cfg.lambda_h_mode, value=cfg.lambda_h,
            b0_h=_as_coeff(cfg.b0_h, kl_s.device), c0_h=_as_coeff(cfg.c0_h, kl_s.device),
        )
        term = lam * kl_s
        if cfg.lambda_h_mode in ("state_dependent", "state_dependent_per_coord"):
            # Only the state-dependent envelopes carry a nonzero R_h (constant returns a zero
            # regularizer); add it UNDETACHED so autograd's product rule cancels to lam*dKL by the
            # envelope theorem. Gating on the state-dependent forms (not '!= constant') skips the
            # zero-add on the constant path. The per-coord form carries R_h^(k) per coordinate.
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
        phi:       torch.Tensor,             # (B, N, n_gen) explicit effective model frame

        *,
        omega:     'torch.Tensor | CompactBlockElement | None' = None,  # stored belief frame; None -> phi
        reflection: Optional[torch.Tensor] = None,  # (B, N) per-token sign s_i; phi-path R_i Omega_ij R_j fold; None -> off
        s_belief:  'Optional[tuple[torch.Tensor, torch.Tensor]]' = None,  # refined (mu_s, sigma_s); None -> raw s tables
    ) -> 'tuple[torch.Tensor, float | torch.Tensor, Optional[torch.Tensor]]':
        r"""Shared model-coupling setup: the s-channel pairwise energy E^s_ij, the gamma softmax
        temperature tau_g, and the gamma attention log-prior.

        The s-channel mirror of the belief beta channel (PB-11): the s tables are scored in the
        CONFIGURED family ``get_family(cfg.family)`` and transported through the SAME connection
        regime the belief E-step uses, E^s_ij = D(s_i || Omega_ij s_j) per irrep block (head), with
        Omega_ij built by ``cfg.transport_mode`` (flat -> exp(phi_i) exp(-phi_j); regime_ii/covariant
        -> the learned edge factor read from the CHANNEL-LOCAL s means/covariances; the link modes ->
        the shared connection_L). Transport is factored-when-fusable on the flat path (audit P4),
        exactly the E-step dispatch. Consumed by :meth:`_gamma_coupling_term` (the forward loss),
        :meth:`_gamma_coupling_terms` (the split diagnostic), and :meth:`gamma_attention_maps`
        (the gamma_ij figure), so all three read the SAME energy/temperature/prior.
        """
        from vfe3.families.base import get_family
        from vfe3.free_energy import attention_tau, pairwise_energy
        from vfe3.geometry.transport import (
            _TRANSPORT_NEEDS_MU, _TRANSPORT_NEEDS_SIGMA, transport_mean,
        )
        from vfe3.inference.e_step import build_belief_transport
        cfg = self.cfg
        pb = self.prior_bank
        fam = get_family(cfg.family)
        s_mu, s_sigma = pb.encode_s(token_ids) if s_belief is None else s_belief   # (B,N,K) or (B,N,K,K)
        n_pos = token_ids.shape[1]
        # omega_direct only when the caller actually supplies the belief frame (omega is not None);
        # callers with no frame in scope pass omega=None and the s-channel uses the phi cocycle. Under
        # the default 'phi' parameterization omega is always None, so this is byte-identical.
        gp = cfg.gauge_parameterization if omega is not None else "phi"
        # Share the belief channel's connection regime (PB-11): build the s-fiber transport through the
        # SAME registry, gating the belief tensors it consumes on the transport-registration metadata
        # (needs_mu/needs_sigma) rather than a mode-name conditional -- the stateful regime_ii/covariant
        # transport reads the CHANNEL-LOCAL s means/covariances, the belief-independent link modes read
        # only connection_L. Flat -> mu/sigma gated to None + every connection None == the byte-identical
        # pure path.
        tm = cfg.transport_mode
        omega = build_belief_transport(phi, self.group, transport_mode=tm,
                                       gauge_parameterization=gp, omega=omega,
                                       reflection=reflection,
                                       right_phi=self._pos_phi_right(phi),
                                       mu=(s_mu if tm in _TRANSPORT_NEEDS_MU else None),
                                       sigma=(s_sigma if tm in _TRANSPORT_NEEDS_SIGMA else None),
                                       transport_mean_per_head=cfg.transport_mean_per_head,
                                       compact_phi_block_transport=self._compact_phi_blocks_enabled(),
                                       rope=self._rope_rotation(n_pos, token_ids.device),
                                       rope_on_cov=cfg.rope_full_gauge,
                                       rope_on_value=cfg.rope_on_value,
                                       validity_max_norm=cfg.transport_chart_max_norm,
                                       exactness_out=self._transport_status,
                                       **self._model_channel_connection_kwargs())
        s_mu_t = transport_mean(omega, s_mu)                         # (B, N, N, K)
        # diagonal_out resolves the diagonal (B,N,K) vs full (B,N,K,K) sandwich EXACTLY as the belief
        # channel does (gradients/kernels.py, oracle.py: diagonal_out=(sigma.dim()==mu.dim())). This is
        # load-bearing for the batch-independent bare link, whose batch-collapsed operator makes the
        # rank-gap heuristic mis-read a batched diagonal sigma as full; harmless (same branch) elsewhere.
        s_sigma_t = fam.transport_dispersion(                        # (B,N,N,K) diag or (B,N,N,K,K) full
            s_sigma, omega, diagonal_out=(s_sigma.dim() == s_mu.dim()))
        e_s = pairwise_energy(
            fam(s_mu, s_sigma), fam(s_mu_t, s_sigma_t),
            alpha=cfg.renyi_order, kl_max=cfg.kl_max, eps=cfg.eps,
            divergence_family=cfg.divergence_family, irrep_dims=self.group.irrep_dims,
        )                                                            # (B,H,N,N) block_glk; (B,N,N) single-block
        gamma_log_prior = self._attention_log_prior(
            n_pos, token_ids.device, prior=cfg.gamma_attention_prior,
        )                                                            # (N, N), cached buffer
        # Group-aware temperature: tau spans the dimension the energy accumulates over (the
        # gauge-irrep block size), exactly as the belief beta channel does. kappa_gamma is
        # gamma's own sharpness handle (not cfg.kappa_beta).
        gamma_tau = attention_tau(self.effective_kappa_gamma(e_s.device), self.group.irrep_dims)
        return e_s, gamma_tau, gamma_log_prior

    def _gamma_coupling_term(
        self,
        token_ids: torch.Tensor,             # (B, N) integer token ids
        phi:       torch.Tensor,             # (B, N, n_gen) converged gauge frame (detached by caller)

        *,
        omega:     'torch.Tensor | CompactBlockElement | None' = None,  # stored belief frame; None -> phi
        reflection: Optional[torch.Tensor] = None,  # (B, N) per-token sign s_i; phi-path R_i Omega_ij R_j fold; None -> off
    ) -> torch.Tensor:                       # () model-coupling block (UNWEIGHTED)
        r"""The gamma model-coupling block at the given gauge frame (UNWEIGHTED).

        gamma = softmax_j(log pi^s - E^s/tau_g) over the :meth:`_gamma_energy` energy, reduced to
        either the canonical envelope -tau_g log Z^s (include_attention_entropy=True) or the
        surrogate sum_j gamma_ij E^s_ij (False; audit P6 -- one toggle, both channels). Shared by
        ``forward`` (grad to the s tables; caller detaches phi/omega) and :meth:`diagnostics`
        (audit V2). :meth:`_gamma_coupling_terms` is the SPLIT (coupling vs meta-entropy)
        diagnostic sibling.
        """
        from vfe3.free_energy import attention_weights, reduced_free_energy
        e_s, gamma_tau, gamma_log_prior = self._gamma_energy(token_ids, phi, omega=omega, reflection=reflection)
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
        omega:     'torch.Tensor | CompactBlockElement | None' = None,  # stored belief frame; None -> phi
        reflection: Optional[torch.Tensor] = None,  # (B, N) per-token sign s_i; phi-path R_i Omega_ij R_j fold; None -> off
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
        e_s, gamma_tau, gamma_log_prior = self._gamma_energy(token_ids, phi, omega=omega, reflection=reflection, s_belief=s_belief)
        gamma_w = attention_weights(e_s, tau=gamma_tau, log_prior=gamma_log_prior)
        log_pi_s = (torch.log_softmax(gamma_log_prior, dim=-1)
                    if gamma_log_prior is not None
                    else torch.full_like(gamma_w, -math.log(gamma_w.shape[-1])))
        log_pi_s = torch.where(torch.isfinite(log_pi_s), log_pi_s, torch.zeros_like(log_pi_s))
        tau_e = _broadcast_tau(gamma_tau, e_s)                       # (H,1,1) per-head, else scalar
        coupling = (gamma_w * e_s).sum()
        gamma_for_log = torch.where(gamma_w > 0.0, gamma_w, torch.ones_like(gamma_w))
        meta = (tau_e * (torch.special.xlogy(gamma_w, gamma_for_log)
                         - gamma_w * log_pi_s)).sum()
        total = reduced_free_energy(e_s, tau=gamma_tau, log_prior=gamma_log_prior).sum()
        return {"coupling": coupling, "meta_entropy": meta, "total": total}

    def _gamma_coupling_rows(
        self,
        token_ids:      torch.Tensor,       # (B, N) integer token ids
        phi:            torch.Tensor,       # (B, N, n_gen) explicit model frame (caller detaches)

        *,
        head_reduction: str,                # 'mean' (training) | 'sum' (diagnostics) over heads
        eps:            float = 1e-12,
        omega:          'torch.Tensor | CompactBlockElement | None' = None,  # stored belief frame; None -> phi
        reflection:     Optional[torch.Tensor] = None,  # (B, N) per-token sign s_i; None -> off
        s_belief:       'Optional[tuple[torch.Tensor, torch.Tensor]]' = None,  # refined (mu_s, sigma_s)
    ) -> 'tuple[torch.Tensor, torch.Tensor]':  # (B, N) coupling rows, (B, N) meta-entropy rows
        r"""Per-query (B, N) split of the gamma model-coupling block: the coupling row
        sum_j gamma_ij E^s_ij and the meta-entropy row tau_g sum_j gamma_ij log(gamma_ij/pi^s_ij),
        each UNWEIGHTED (the caller applies lambda_gamma). After summing the key axis the head axis is
        reduced by ``head_reduction``: ``"mean"`` for the forward loss -- turning (B,H,N) into (B,N)
        so that ``model_reduction="mean"`` reproduces the live :meth:`_gamma_coupling_term` average
        over B/H/N -- and ``"sum"`` for diagnostics, preserving its existing sum-over-heads scale. The
        single-block (single-head) path inserts and then removes a singleton head axis so both
        reductions run through the same code. gamma is the exact softmax of E^s (undetached), so the
        s tables carry gradient; the row split is the mathematically equal coupling-plus-entropy form
        of the fused envelope (their fp32 association differs -- no bitwise claim)."""
        from vfe3.free_energy import _broadcast_tau, attention_weights
        e_s, gamma_tau, gamma_log_prior = self._gamma_energy(
            token_ids, phi, omega=omega, reflection=reflection, s_belief=s_belief)
        gamma_w = attention_weights(e_s, tau=gamma_tau, log_prior=gamma_log_prior)
        log_pi_s = (torch.log_softmax(gamma_log_prior, dim=-1)
                    if gamma_log_prior is not None
                    else torch.full_like(gamma_w, -math.log(gamma_w.shape[-1])))
        log_pi_s = torch.where(torch.isfinite(log_pi_s), log_pi_s, torch.zeros_like(log_pi_s))
        tau_e = _broadcast_tau(gamma_tau, e_s)                        # (H,1,1) per-head, else scalar
        coupling_key = (gamma_w * e_s).sum(dim=-1)                    # (B, [H,] N)
        gamma_for_log = torch.where(gamma_w > 0.0, gamma_w, torch.ones_like(gamma_w))
        meta_key = (tau_e * (torch.special.xlogy(gamma_w, gamma_for_log)
                             - gamma_w * log_pi_s)).sum(dim=-1)
        if coupling_key.dim() == 2:                                   # single-block: insert singleton head
            coupling_key = coupling_key.unsqueeze(1)                  # (B, 1, N)
            meta_key = meta_key.unsqueeze(1)
        if head_reduction == "mean":
            return coupling_key.mean(dim=1), meta_key.mean(dim=1)     # (B, N)
        if head_reduction == "sum":
            return coupling_key.sum(dim=1), meta_key.sum(dim=1)       # (B, N)
        raise ValueError(f"head_reduction must be 'mean' or 'sum', got {head_reduction!r}")

    def _model_channel_free_energy(
        self,
        token_ids: torch.Tensor,             # (B, N) integer token ids
        belief:   BeliefState,               # fixed q carrying the candidate model-frame state

        *,
        s_belief: 'Optional[tuple[torch.Tensor, torch.Tensor]]' = None,
    ) -> torch.Tensor:                       # () mean-reduced active h/s/gamma objective
        r"""Authoritative active model-channel objective used by training and Metropolis scoring.

        The hyper-prior rows and the split gamma coupling/meta-entropy rows are assembled through
        :func:`hierarchical_free_energy_terms` with the production mean reduction. The tied frame is
        deliberately detached before gamma evaluation: gamma is value-sensitive to a candidate frame
        (so a Metropolis block move can score it) while the direct gradient route remains passive.
        ``s_belief`` supplies the fixed refined model belief under ``s_e_step``; ``None`` reads the raw
        model tables, matching the scored production route.
        """
        from vfe3.free_energy import hierarchical_free_energy_terms

        cfg = self.cfg
        if cfg.lambda_h == 0.0 and cfg.lambda_gamma == 0.0:
            return belief.mu.new_zeros(())

        hyper_prior_rows = None
        model_coupling_rows = None
        meta_entropy_rows = None
        if cfg.lambda_h > 0.0:
            hyper_prior_rows = self._hyper_prior_weighted(
                token_ids, s_belief=s_belief)
        if cfg.lambda_gamma > 0.0:
            tied_model_frame = cfg.s_frame_mode == "tied"
            model_phi = self._resolve_model_frame(token_ids, belief.phi).detach()
            c_rows, me_rows = self._gamma_coupling_rows(
                token_ids,
                model_phi,
                head_reduction="mean",
                omega=(belief.omega.detach()
                       if tied_model_frame and belief.omega is not None else None),
                reflection=(belief.reflection.detach()
                            if tied_model_frame and belief.reflection is not None else None),
                s_belief=s_belief,
            )
            model_coupling_rows = cfg.lambda_gamma * c_rows
            meta_entropy_rows = (
                cfg.lambda_gamma * me_rows
                if cfg.include_attention_entropy else torch.zeros_like(c_rows)
            )

        ref = hyper_prior_rows if hyper_prior_rows is not None else model_coupling_rows
        zeros = torch.zeros_like(ref)
        if hyper_prior_rows is None:
            hyper_prior_rows = zeros
        if model_coupling_rows is None:
            model_coupling_rows = zeros
            meta_entropy_rows = zeros
        return hierarchical_free_energy_terms(
            zeros,
            zeros,
            zeros,
            zeros,
            hyper_prior_rows,
            model_coupling_rows,
            meta_entropy_rows,
            zeros,
            q_reduction="sum",
            model_reduction="mean",
        ).total

    @torch.no_grad()
    def gamma_attention_maps(
        self,
        token_ids: torch.Tensor,             # (B, N) token ids; only sequence 0 is used

        *,
        snapshot:  Optional[DiagnosticSnapshot] = None,
    ) -> Optional[torch.Tensor]:             # (H, N, N) gamma_ij, or None when the s channel is off
        r"""Per-head model-coupling attention gamma_ij for sequence 0 (no_grad), the s-channel mirror
        of :meth:`attention_maps`.

        gamma_ij = softmax_j( log pi^s_ij - E^s_ij / tau_g ) on the model-channel beliefs s under the
        selected FLAT model transport (the tied converged belief frame or the independently stored
        phi_tilde frame). Returns ``(H, N, N)`` (rows = query i, cols = key j;
        H = len(group.irrep_dims), 1 for a single-block group) or ``None`` when the model channel is
        inactive (no s tables). OFF the training hot path (no_grad); for periodic figure generation.
        """
        if snapshot is not None:
            snapshot = self._validate_diagnostic_snapshot(token_ids, snapshot)
        if not self._model_channel_active:
            return None
        if snapshot is not None:
            return snapshot.gamma_maps
        from vfe3.free_energy import attention_weights
        enc = self.prior_bank.encode(token_ids[:1])
        belief = BeliefState(mu=enc.mu[0], sigma=enc.sigma[0], phi=self._apply_pos_phi(enc.phi[0]),
                             omega=enc.omega[0] if enc.omega is not None else None,   # carry the GL(K) frame under omega_direct
                             reflection=enc.reflection[0] if enc.reflection is not None else None,
                             right_phi=self._pos_phi_right(enc.phi[0]))
        model_phi = self._resolve_model_frame(token_ids[:1], belief.phi.unsqueeze(0))
        s_belief = self._refined_s_belief(token_ids)                  # s1 under s_e_step (M2), else None (raw s tables)
        if s_belief is not None:
            belief = belief._replace(mu=s_belief[0][0], sigma=s_belief[1][0])
        n = belief.mu.shape[0]
        log_prior = self._attention_log_prior(n, token_ids.device)
        log_prior = self._fold_precision_bias(log_prior, belief.sigma)  # match forward/diagnostics/attention_maps (r2 id22)
        if self.cfg.gamma_as_beta_prior:                             # m4: match forward's hierarchical gamma prior fold
            tied_model_frame = self.cfg.s_frame_mode == "tied"
            log_prior = self._fold_gamma_prior(log_prior, token_ids[:1], model_phi,
                                               omega=(belief.omega.unsqueeze(0)
                                                      if tied_model_frame and belief.omega is not None else None),
                                               reflection=(belief.reflection.unsqueeze(0)
                                                           if tied_model_frame and belief.reflection is not None else None),
                                               s_belief=s_belief)[0]
        rope = self._rope_rotation(n, token_ids.device)
        out = vfe_stack(                                             # converged belief gauge frame
            belief, belief.mu, belief.sigma, self.group, self.cfg,
            log_prior=log_prior, block_norm=self.block_norm,
            head_mixer=self.head_mixer, cg_coupling=self.cg_coupling,
            lambda_beta=self.cfg.lambda_beta,
            connection_W=getattr(self, "connection_W", None),
            connection_M=getattr(self, "connection_M", None),
            connection_L=getattr(self, "connection_L", None),
            rope=rope, rope_on_cov=self.cfg.rope_full_gauge, rope_on_value=self.cfg.rope_on_value,
            transport_status=self._transport_status,
            gauge_parameterization=self.cfg.gauge_parameterization,
            kappa_beta_override=self.effective_kappa_beta(belief.mu.device),
        )
        tied_model_frame = self.cfg.s_frame_mode == "tied"
        model_phi = self._resolve_model_frame(token_ids[:1], out.phi.unsqueeze(0))
        e_s, gamma_tau, gamma_log_prior = self._gamma_energy(
            token_ids[:1], model_phi,
            omega=(out.omega.unsqueeze(0) if tied_model_frame and out.omega is not None else None),
            reflection=(out.reflection.unsqueeze(0)
                        if tied_model_frame and out.reflection is not None else None),
            s_belief=s_belief)
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
        temperature:    float           = 1.0,   # >0; applied to logits before sampling; ignored if greedy
        greedy:         bool            = False, # True -> argmax; ignores temperature/top_k/top_p
        top_k:          Optional[int]   = None,  # keep the k largest-logit tokens, -inf the rest
        top_p:          Optional[float] = None,  # nucleus: smallest set with softmax cumsum >= p
    ) -> torch.Tensor:                       # (B, N0 + max_new_tokens) prompt followed by generated ids
        r"""Autoregressively extend each prompt by ``max_new_tokens`` tokens.

        Each step feeds the running sequence through :meth:`forward_beliefs` and decodes only the
        last position as logits shaped ``(B, 1, V)``. The ordinary sampler truncates its context to
        the last ``cfg.max_seq_len`` tokens, while a policy sampler preserves its complete context
        and rejects a context-plus-candidate that exceeds the model bound. The returned sequence
        keeps the FULL prompt followed by the generated ids. Because generation never enters the
        training/loss branch, it cannot corrupt training (runs under ``torch.no_grad``).

        Greedy (``greedy=True``) takes the argmax and ignores ``temperature``/``top_k``/
        ``top_p``. Otherwise the logits are divided by ``temperature``, then ``top_k``
        (keep the k largest, ``-inf`` the rest), then ``top_p`` (nucleus: smallest set
        whose softmax cumsum reaches ``p``, ``-inf`` the rest, always keeping the top
        token), then softmaxed and sampled with :func:`torch.multinomial`.

        This is the correct-but-slow first version: it re-runs the FULL forward (encode
        -> E-step -> decode) for every generated token. Incremental belief reuse across
        steps is a future optimization.
        """
        # audit F9 (2026-06-28): under a policy scorer the next token comes from _policy_select, which
        # uses policy_top_k / policy_precision from config and does NOT consume the call-time sampler
        # knobs. Reject them rather than silently ignoring them; 'greedy' IS honored (argmax vs sample
        # of the policy posterior).
        if self.cfg.policy_mode != "none" and (temperature != 1.0 or top_k is not None or top_p is not None):
            raise ValueError(
                "temperature/top_k/top_p are ignored when policy_mode != 'none' (the EFE policy posterior "
                "uses policy_top_k and policy_precision); drop them or set policy_mode='none'. 'greedy' is "
                "honored (argmax vs sample of the policy posterior).")
        # audit C13 (2026-07-01): validate the sampler arguments up front. A negative max_new_tokens
        # would silently no-op (empty loop, prompt returned unchanged); temperature<=0, out-of-range
        # top_k, and top_p outside (0, 1] fail late or produce invalid probabilities. Greedy ignores
        # temperature/top_k/top_p (and the policy path rejects non-defaults above), so those three
        # are checked only on the sampled policy_mode='none' path.
        if max_new_tokens < 0:
            raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")
        if token_ids.shape[1] <= 0:
            raise ValueError("generate requires a nonempty token context")
        if not greedy and self.cfg.policy_mode == "none":
            if not (temperature > 0.0):
                raise ValueError(f"temperature must be > 0, got {temperature}")
            if top_k is not None and not (1 <= top_k <= self.cfg.vocab_size):
                raise ValueError(f"top_k must be in [1, vocab_size={self.cfg.vocab_size}], got {top_k}")
            if top_p is not None and not (0.0 < top_p <= 1.0):
                raise ValueError(f"top_p must be in (0, 1], got {top_p}")
        # audit F10 (2026-07-01), warn-only (mirrors the D3 link-mode memory estimator, f3387b9):
        # generate() has NO incremental belief/KV cache -- every generated token re-runs the FULL
        # forward (encode -> E-step -> decode) over the whole <=max_seq_len window. Estimate the
        # dominant fp32 per-forward transients at the max_seq_len bound -- the (B, H, N, N)
        # attention/KL maps plus the (B, N, V) logits and (B, N, K) beliefs -- and warn ONCE past
        # the documented 2 GiB budget; never raise (the pure path stays runnable). Incremental
        # belief reuse across steps is the deferred optimization (see the docstring above).
        _B, _N, _K = token_ids.shape[0], self.cfg.max_seq_len, self.cfg.embed_dim
        _est_bytes = 4 * _B * (self.cfg.n_heads * _N * _N + _N * self.cfg.vocab_size + _N * _K)
        if _est_bytes > 2 * 1024 ** 3:                           # documented budget: 2 GiB per forward
            import warnings
            warnings.warn(
                f"generate(): estimated per-forward peak ~{_est_bytes / 2 ** 30:.1f} GiB (B={_B}, "
                f"N={_N}, heads={self.cfg.n_heads}, V={self.cfg.vocab_size}) exceeds the 2 GiB "
                "budget, and generate() re-runs the FULL forward (encode -> E-step -> decode) for "
                "EVERY generated token -- there is no incremental belief/KV cache yet (the "
                "incremental-cache optimization is deferred; see the generate docstring). Expect "
                "O(max_new_tokens * forward(max_seq_len)) time and this peak per step; reduce the "
                "batch/context or generate fewer tokens.",
                UserWarning,
                stacklevel=2,
            )
        # PB-06 sigma-consumer gate: WHEN AND ONLY WHEN the ambiguity arm is the gated 'sigma_mc', derive
        # the four live identities ONCE and verify the pre-registered consumer gate before any rollout,
        # then thread them through _policy_select. policy_ambiguity_mode='likelihood_entropy' (and
        # policy_mode='none') never hashes the model/code/corpus, inspects the specification, or reads the
        # artifact -- the pure path stays untouched. Providers are resolved through vfe3.inference.sigma_gate
        # at call time (no unpatchable aliases).
        sigma_gate_ids: Dict[str, Optional[str]] = {}
        if self.cfg.policy_ambiguity_mode == "sigma_mc":
            from vfe3.inference import sigma_gate
            from vfe3.run_artifacts import (model_behavior_fingerprint, semantic_config_fingerprint,
                                            sigma_behavior_config)
            behavior = model_behavior_fingerprint(sigma_behavior_config(self.cfg), self.state_dict())
            spec = sigma_gate.sigma_gate_spec_identity()
            if spec == "unknown":
                raise ValueError(
                    "policy_ambiguity_mode='sigma_mc' requires a resolvable governing specification "
                    "identity; sigma_gate_spec_identity() returned 'unknown'.")
            code = sigma_gate.sigma_consumer_code_identity()
            meas_context = sigma_gate.sigma_measurement_context(self.cfg)
            context_fp = semantic_config_fingerprint(meas_context)
            sigma_gate.verify_sigma_consumer_gate(
                self.cfg.policy_sigma_gate_artifact,
                actual_model_behavior_sha256=behavior,
                actual_spec_identity=spec,
                actual_code_identity_sha256=code,
                actual_measurement_context_sha256=context_fp,
            )
            sigma_gate_ids = dict(
                model_behavior_sha256=behavior,
                sigma_spec_identity=spec,
                sigma_code_identity_sha256=code,
                sigma_measurement_context_sha256=context_fp,
            )
        seq = token_ids
        for _ in range(max_new_tokens):
            if self.cfg.policy_mode == "none":
                context = seq[:, -self.cfg.max_seq_len:]                 # (B, <=max_seq_len)
                _belief, decoded = self.forward_beliefs(
                    context, return_logits=True, decode_last=True)
                logits = decoded[:, 0, :]                                # (B, V) last position
                invalid_row = torch.isnan(logits).any(dim=-1) | torch.isposinf(logits).any(dim=-1)
                if bool(invalid_row.any()):
                    rows = invalid_row.nonzero(as_tuple=False).flatten().tolist()
                    raise ValueError(
                        f"generation logits contain NaN or +inf values in rows {rows}")
                finite_row = torch.isfinite(logits).any(dim=-1)
                if not bool(finite_row.all()):
                    rows = (~finite_row).nonzero(as_tuple=False).flatten().tolist()
                    raise ValueError(f"generation logits have no finite value in rows {rows}")
                if greedy:
                    next_token = logits.argmax(dim=-1, keepdim=True)    # (B, 1)
                    retained_logits = torch.gather(logits, 1, next_token)
                    retained_finite = torch.isfinite(retained_logits).all(dim=-1)
                    if not bool(retained_finite.all()):
                        rows = (~retained_finite).nonzero(as_tuple=False).flatten().tolist()
                        raise ValueError(
                            f"generation retained logits contain non-finite values in rows {rows}")
                else:
                    logits = logits / temperature
                    retained = torch.ones_like(logits, dtype=torch.bool)
                    if top_k is not None:
                        kth = logits.topk(top_k, dim=-1).values[:, -1:]  # (B, 1) k-th largest
                        remove_topk = logits < kth
                        retained = ~remove_topk
                        logits = logits.masked_fill(remove_topk, float("-inf"))
                    retained_finite = torch.where(
                        retained, torch.isfinite(logits), torch.ones_like(retained)
                    ).all(dim=-1)
                    if not bool(retained_finite.all()):
                        rows = (~retained_finite).nonzero(as_tuple=False).flatten().tolist()
                        raise ValueError(
                            f"generation retained logits contain non-finite values in rows {rows}")
                    if top_p is not None:
                        sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
                        sorted_probs = sorted_logits.softmax(dim=-1)       # compute the softmax once
                        cumprobs = sorted_probs.cumsum(dim=-1)
                        # Keep the smallest nucleus whose cumprob reaches top_p; the strict
                        # shift always keeps the top token (its cumprob>=p never removes it).
                        remove = cumprobs - sorted_probs >= top_p
                        remove_unsorted = remove.scatter(-1, sorted_idx, remove)
                        retained = retained & ~remove_unsorted
                        logits = logits.masked_fill(remove_unsorted, float("-inf"))
                    retained_finite = torch.where(
                        retained, torch.isfinite(logits), torch.ones_like(retained)
                    ).all(dim=-1)
                    if not bool(retained_finite.all()):
                        rows = (~retained_finite).nonzero(as_tuple=False).flatten().tolist()
                        raise ValueError(
                            f"generation retained logits contain non-finite values in rows {rows}")
                    probs = logits.softmax(dim=-1)                      # (B, V)
                    next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            else:
                # EFE policy rerank (no_grad). Reached only under a non-default policy_mode toggle, so
                # default generation (policy_mode='none') is byte-identical (spec Section 3.4).
                context = seq                                             # policy paths never truncate context
                next_token = self._policy_select(context, greedy=greedy, **sigma_gate_ids)   # (B, 1)
            seq = torch.cat([seq, next_token], dim=-1)
        return seq

    @torch.no_grad()
    def _policy_select(
        self,
        context: torch.Tensor,           # (B, N) current-context ids

        *,
        greedy:  bool = True,            # True -> argmax the policy posterior; else sample it
        model_behavior_sha256:      Optional[str] = None,   # PB-06 consumer-gate identities (sigma_mc only)
        sigma_spec_identity:        Optional[str] = None,
        sigma_code_identity_sha256: Optional[str] = None,
        sigma_measurement_context_sha256: Optional[str] = None,
    ) -> torch.Tensor:                   # (B, 1) selected next-token id
        r"""EFE policy selection over a top-``policy_top_k`` candidate menu (spec Section 3.4).

        Decodes the base last-position logits once (the pre-registered candidate generator E), then
        constructs the candidate menu: a one-step ``(B, Kp, 1)`` top-k menu for every mode except
        ``policy_mode='efe_rollout'``, which builds the bounded H-step beam menu ``(B, Kp, H)`` through
        :func:`vfe3.inference.candidate_menu.build_topk_policy_menu` (audit PB-05). The menu is scored
        through the configured ``policy_mode`` scorer with the candidate prior E (the base softmax over
        the one-step menu, or the beam log-softmax for efe_rollout), and the argmax (or a sample) of the
        policy posterior selects a policy; the committed token is that policy's FIRST action (identical
        to the selected token in the one-step case). The environment response to a committed action is
        appended by the closed-loop driver, never here (the scored rollout appends the action only; spec
        Section 2.2).

        Note: ``policy_preference='task'`` / ``'held_out_predictive'`` need per-episode / per-corpus
        context (the goal, or p_data) that ``generate`` does not supply, so those preferences are
        driven through the closed-loop experiment harness, which calls the scorer directly; under
        ``generate`` the meaningful preference is the global ``'flat'``.
        """
        from vfe3.inference.candidate_menu import build_topk_policy_menu
        from vfe3.inference.policy import _validate_policy_context, get_policy, get_preference
        # audit PB-05 (2026-07-12): efe_rollout (horizon>1) is now reachable through generate() via the
        # bounded H-step beam menu (vfe3/inference/candidate_menu.py); only candidate CONSTRUCTION
        # branches on policy_mode -- the one-step block below is unchanged. The scorer's cache gate is
        # NOT relaxed: efe_rollout still fails closed on a cache-unsupported config inside
        # get_policy('efe_rollout') (spec Section 3.5), not here.
        horizon = self.cfg.policy_horizon if self.cfg.policy_mode == "efe_rollout" else 1
        _validate_policy_context(context, horizon, self.cfg.max_seq_len)
        _belief, decoded = self.forward_beliefs(context, return_logits=True, decode_last=True)
        base_logits = decoded[:, 0, :]                               # (B, V) base last-position logits
        invalid_row = torch.isnan(base_logits).any(dim=-1) | torch.isposinf(base_logits).any(dim=-1)
        if bool(invalid_row.any()):
            rows = invalid_row.nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(
                f"policy base logits contain NaN or +inf values in rows {rows}")
        finite_row = torch.isfinite(base_logits).any(dim=-1)
        if not bool(finite_row.all()):
            rows = (~finite_row).nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(f"policy base logits have no finite value in rows {rows}")
        if self.cfg.policy_mode == "efe_rollout":
            candidates, log_prior = build_topk_policy_menu(        # (B, Kp, H) H-action beams, (B, Kp) log E
                context, base_logits, self,
                horizon=horizon, width=self.cfg.policy_top_k,
            )
        else:
            Kp = self.cfg.policy_top_k
            topk = base_logits.topk(Kp, dim=-1).indices            # (B, Kp) candidate token ids (generator E)
            candidates = topk.unsqueeze(-1)                        # (B, Kp, 1) one-step action tokens
            menu_logits = torch.gather(base_logits, 1, topk)      # (B, Kp) base logits over the menu
            retained_finite = torch.isfinite(menu_logits).all(dim=-1)
            if not bool(retained_finite.all()):
                rows = (~retained_finite).nonzero(as_tuple=False).flatten().tolist()
                raise ValueError(f"policy menu logits contain non-finite retained values in rows {rows}")
            log_prior = torch.log_softmax(menu_logits, dim=-1)    # (B, Kp) log E(pi): base softmax over menu
        preference = get_preference(self.cfg.policy_preference)(
            self.prior_bank, device=base_logits.device)            # (V,)/(B,V) log p(o|C), on the model device (audit F5)
        out = get_policy(self.cfg.policy_mode)(
            context, candidates, preference, self,
            gamma=self.cfg.policy_precision, horizon=self.cfg.policy_horizon,
            score_terms=self.cfg.policy_score_terms, log_prior=log_prior, base_logits=base_logits,
            ambiguity_mode=self.cfg.policy_ambiguity_mode,
            model_behavior_sha256=model_behavior_sha256,
            sigma_spec_identity=sigma_spec_identity,
            sigma_code_identity_sha256=sigma_code_identity_sha256,
            sigma_measurement_context_sha256=sigma_measurement_context_sha256,
        )
        if greedy:
            idx = out.policy_posterior.argmax(dim=-1, keepdim=True)         # (B, 1) menu index
        else:
            idx = torch.multinomial(out.policy_posterior, num_samples=1)    # (B, 1)
        selected = torch.gather(                                   # (B, 1, H) the selected policy sequence
            candidates, 1, idx.unsqueeze(-1).expand(-1, -1, candidates.shape[-1]),
        )
        return selected[:, 0, :1]                                  # (B, 1) FIRST action of the selected policy

    def _fold_precision_bias(
        self,
        log_prior: Optional[torch.Tensor],   # (N,N)/(H,N,N) position prior (batched or not), or None
        sigma:     torch.Tensor,             # (..., N, K) diag, or (..., N, K, K) full, key belief cov
    ) -> Optional[torch.Tensor]:
        r"""Fold the detached precision-weighted-attention reliability bias ``-log(b0 + tr Sigma_j)``
        into ``log_prior``, broadcasting over query (and head). Shared by ``forward`` and the
        diagnostic replays (``diagnostics``/``attention_maps``) so every belief-channel consumer scores
        the SAME attention prior the forward E-step descends (audit 2026-06-17 r2 id22). No-op (returns
        ``log_prior`` unchanged) when ``precision_weighted_attention`` is off. Rank-robust: ``sigma``
        may be ``(B, N, K)`` (forward) or ``(N, K)`` (diagnostics), and under ``family='gaussian_full'``
        the full covariance ``(.., N, K, K)`` -- reduced to its per-coordinate variances (the diagonal)
        so ``tr Sigma_j`` is the matrix trace, not a sum over a covariance row.

        NOT ``@torch.no_grad()`` (audit 2026-07-05 M1): only the reliability bias ``kb`` is meant to
        be detached (each branch calls ``.detach()`` explicitly below); a ``no_grad`` wrapper would
        additionally sever the graph of the ``log_prior`` it is added to -- the ONLY gradient path of
        the learnable T5 bias -- silently freezing ``t5_bias`` under
        ``precision_weighted_attention=True`` + ``t5_learnable_bias=True``. Values are identical
        either way; only the autograd graph of the returned sum differs."""
        if not self.cfg.precision_weighted_attention:
            return log_prior
        b0 = self.cfg.precision_attention_b0
        if len(self.group.irrep_dims) == 1:                # headless (.., N, N) energy: NO head axis
            kb = _precision_key_bias(sigma, b0=b0, family=self.cfg.family).detach()   # (.., N)
            kb = kb.unsqueeze(-2)                                                 # (.., 1, N)
        elif self.cfg.precision_attention_per_head:        # per-head (.., H, N, N) energy
            kb = _precision_key_bias(
                sigma,
                b0=b0,
                family=self.cfg.family,
                irrep_dims=self.group.irrep_dims,
            ).detach()                                                           # (.., N, H)
            kb = kb.transpose(-1, -2).unsqueeze(-2)                               # (.., H, 1, N)
        else:                                              # global bias, multi-block: head-broadcast
            kb = _precision_key_bias(sigma, b0=b0, family=self.cfg.family).detach()   # (.., N)
            kb = kb.unsqueeze(-2).unsqueeze(-2)                                   # (.., 1, 1, N)
        return kb if log_prior is None else log_prior + kb

    def _fold_gamma_prior(
        self,
        log_prior: Optional[torch.Tensor],   # (N,N)/(H,N,N) belief log-prior (precision bias already folded), or None
        token_ids: torch.Tensor,             # (B, N) integer token ids
        phi:       torch.Tensor,             # (B, N, n_gen) explicit effective model frame

        *,
        log_eps:    float                                              = 1e-12,  # floor for log(pi) on allowed support

        omega:      'torch.Tensor | CompactBlockElement | None'        = None,   # stored GL(K) frame
        reflection: Optional[torch.Tensor]                             = None,   # (B, N) phi-path reflection sign
        s_belief:   'Optional[tuple[torch.Tensor, torch.Tensor]]'       = None,   # refined (mu_s, sigma_s); raw tables if None
    ) -> torch.Tensor:                       # (B, [H,] N, N) mixed log-prior
        r"""Hierarchical attention prior (cfg.gamma_as_beta_prior): fold the model channel's DETACHED
        posterior gamma into the belief channel's attention prior in PROBABILITY space,

            pi_ij = (1 - w) * softmax_j(B_ij) + w * gamma_ij,      w = cfg.gamma_prior_weight,
            gamma_ij = softmax_j(B^s_ij - E^s_ij / tau_gamma)      (the _gamma_energy machinery),

        and return log(pi). Rows renormalize by construction (a convex mixture of two row-normalized
        distributions; the explicit renormalization below is an fp32 guard). Both channels share the
        causal support (config validation pins lambda_gamma > 0 so the s tables exist), so pi is
        EXACTLY 0 where the belief prior forbids; those entries are re-pinned to -inf rather than the
        log_eps floor. gamma is computed under ``torch.no_grad`` (the detached-fixed-prior footprint
        of ``_fold_precision_bias``): no gradient reaches the s tables through the belief prior, and
        the closed-form belief kernel treats the fold as a fixed prior (exact). The mixture itself is
        composed OUTSIDE the no_grad so ``log_prior``'s own graph (the learnable T5 bias) stays live.
        An UNDETACHED variant -- training s through the belief attention -- is deliberately deferred.
        """
        from vfe3.free_energy import attention_weights
        w = self.cfg.gamma_prior_weight
        with torch.no_grad():
            e_s, gamma_tau, gamma_log_prior = self._gamma_energy(
                token_ids, phi, omega=omega, reflection=reflection, s_belief=s_belief,
            )
            gamma = attention_weights(e_s, tau=gamma_tau, log_prior=gamma_log_prior)  # (B, [H,] N, N)
        if log_prior is None:
            pi_b    = torch.full_like(gamma, 1.0 / gamma.shape[-1])   # uniform prior over keys
            support = None
        else:
            pi_b    = torch.softmax(log_prior, dim=-1)                # rows normalized on the causal support
            support = torch.isfinite(log_prior)                       # shared causal mask (both channels)
        pi  = (1.0 - w) * pi_b + w * gamma                            # probability-space mixture
        if support is not None:
            # The beta support is authoritative. Gamma may have been normalized on a wider support;
            # remove that forbidden mass BEFORE normalizing the mixture so every retained row sums
            # to one on the keys the belief channel can actually attend to.
            pi = pi.masked_fill(~support, 0.0)
        pi  = pi / pi.sum(dim=-1, keepdim=True).clamp(min=log_eps)    # normalize on active support
        out = torch.log(pi.clamp(min=log_eps))                        # (B, [H,] N, N)
        if support is not None:
            out = out.masked_fill(~support, float("-inf"))            # keep the EXACT -inf causal structure
        return out

    def _effective_beta_log_prior(
        self,
        belief:  BeliefState,                      # candidate belief (supplies the tied-gamma frame)
        context: EffectiveBetaPriorContext,        # fixed pre-stack capture (raw prior, precision sigma, model frame, refined s)
    ) -> Optional[torch.Tensor]:
        r"""The single authoritative belief-channel attention log-prior the E-step descends.

        Folds the DETACHED precision-weighted reliability bias ``-log(b0 + tr Sigma_j)`` and, under
        ``cfg.gamma_as_beta_prior``, the DETACHED hierarchical gamma prior onto ``context.base_log_prior``
        (the RAW ``_attention_log_prior``), reproducing the pre-refactor inline ``forward_beliefs`` fold
        sequence exactly. Shared by the forward and the reflection/two-hop scorers (audit PB-12) so every
        belief-channel consumer scores the SAME objective.

        The precision fold ALWAYS reads ``context.precision_sigma`` -- the FIXED pre-stack belief
        covariance -- never ``belief.sigma``, so a candidate belief with a different covariance leaves the
        precision-only prior EXACTLY unchanged. Under ``gamma_as_beta_prior`` the CANDIDATE ``belief``
        supplies the tied-gamma frame (its ``omega``/``reflection``) ONLY when ``s_frame_mode=='tied'``;
        the independent ``phi_tilde`` model frame consumes neither, so a belief-frame reflection leaves the
        gamma fold unchanged. NOT ``@torch.no_grad()``: each fold detaches its own contribution
        internally, so ``context.base_log_prior``'s graph (the learnable T5 relative-position bias) stays
        live while the precision/gamma contributions stay detached. The helper allocates no persistent
        state and never mutates or caches a candidate-dependent tensor."""
        log_prior = self._fold_precision_bias(context.base_log_prior, context.precision_sigma)
        if self.cfg.gamma_as_beta_prior:
            tied_model_frame = self.cfg.s_frame_mode == "tied"
            s_belief = None if context.s_mu is None else (context.s_mu, context.s_sigma)
            log_prior = self._fold_gamma_prior(
                log_prior, context.token_ids, context.model_phi,
                omega=(belief.omega if tied_model_frame else None),
                reflection=(belief.reflection if tied_model_frame else None),
                s_belief=s_belief,
            )
        return log_prior

    def _beta_tau(
        self,
        sigma: torch.Tensor,                 # (..., N, K) diag or (..., N, K, K) full belief covariance
        mu:    torch.Tensor,                 # (..., N, K) belief means (rank reference: full iff sigma rank = mu rank + 1)
        tau:   'float | torch.Tensor',       # base attention_tau (scalar or (H,))
    ) -> 'float | torch.Tensor':
        r"""The belief channel's effective softmax temperature for the diagnostic replays: the base
        ``tau`` unchanged (cfg.query_adaptive_tau off -- byte-identical), or the per-query adaptive
        tau_{i,h} = tau_h (1 + c tr_h(Sigma_i)/d_h) (``query_adaptive_tau``; DETACHED, from the
        CURRENT belief sigma), matching what vfe_stack passes the forward E-step. The gamma model
        channel keeps its scalar tau_gamma."""
        if not self.cfg.query_adaptive_tau:
            return tau
        from vfe3.free_energy import query_adaptive_tau
        sig = sigma if sigma.dim() == mu.dim() else sigma.diagonal(dim1=-2, dim2=-1)
        return query_adaptive_tau(sig, tau, self.group.irrep_dims, c=self.cfg.query_tau_c)

    def _first_sequence_log_prior(
        self,
        log_prior: Optional[torch.Tensor],
        batch_size: int,
    ) -> Optional[torch.Tensor]:
        r"""Drop a captured batch axis while preserving the optional attention-head axis."""
        if log_prior is None:
            return None
        unbatched_rank = 2 if len(self.group.irrep_dims) == 1 else 3
        if log_prior.dim() == unbatched_rank + 1 and log_prior.shape[0] == batch_size:
            return log_prior[0]
        return log_prior

    def _validate_diagnostic_snapshot(
        self,
        token_ids: torch.Tensor,
        snapshot: DiagnosticSnapshot,
    ) -> DiagnosticSnapshot:
        if snapshot.owner is not self:
            raise ValueError("diagnostic snapshot belongs to a different model instance")
        if not torch.equal(snapshot.token_ids, token_ids):
            raise ValueError("diagnostic snapshot token_ids do not match this request")
        return snapshot

    @torch.no_grad()
    def _attention_map_for_belief(
        self,
        belief:   BeliefState,
        log_prior: Optional[torch.Tensor],
        rope:      Optional[torch.Tensor],
    ) -> torch.Tensor:                         # (H, N, N)
        r"""Compute beta from an already captured block output, without replaying inference."""
        from vfe3.inference.e_step import _transport
        from vfe3.geometry.transport import transport_mean
        from vfe3.families.base import get_family
        from vfe3.free_energy import pairwise_energy, attention_weights, attention_tau

        cfg = self.cfg
        fam = get_family(cfg.family)
        omega = _transport(
            belief.phi, self.group, transport_mode=cfg.transport_mode,
            mu=(belief.mu if cfg.transport_mode in _REGIME_NEEDS_MU else None),
            sigma=(belief.sigma if cfg.transport_mode in _REGIME_NEEDS_SIGMA else None),
            connection_W=getattr(self, "connection_W", None),
            connection_M=getattr(self, "connection_M", None),
            connection_L=getattr(self, "connection_L", None),
            link_alpha=cfg.link_alpha, link_soft_cap=cfg.link_soft_cap,
            clamp_monitor=cfg.transport_clamp_monitor,
            validity_max_norm=cfg.transport_chart_max_norm,
            exactness_out=self._transport_status,
            cocycle_relaxation=cfg.cocycle_relaxation,
            gauge_parameterization=cfg.gauge_parameterization,
            omega=belief.omega,
            reflection=belief.reflection,
            right_phi=belief.right_phi,
        )
        if rope is not None:
            rope_omega = RopeTransport(
                base=omega, rope=rope, on_cov=cfg.rope_full_gauge,
                on_value=cfg.rope_on_value)
            mu_t = transport_mean(rope_omega, belief.mu)
            sigma_t = fam.transport_dispersion(belief.sigma, rope_omega)
        else:
            mu_t = transport_mean(omega.unsqueeze(0), belief.mu.unsqueeze(0))[0]
            sigma_t = fam.transport_dispersion(
                belief.sigma.unsqueeze(0), omega.unsqueeze(0))[0]
        energy = pairwise_energy(
            fam(belief.mu, belief.sigma), fam(mu_t, sigma_t),
            alpha=cfg.renyi_order, kl_max=cfg.kl_max, eps=cfg.eps,
            divergence_family=cfg.divergence_family, irrep_dims=self.group.irrep_dims,
        )
        base_tau = attention_tau(
            self.effective_kappa_beta(belief.mu.device), self.group.irrep_dims)
        beta = attention_weights(
            energy, tau=self._beta_tau(belief.sigma, belief.mu, base_tau),
            log_prior=log_prior)
        return beta.unsqueeze(0) if beta.dim() == 2 else beta

    @torch.no_grad()
    def _gamma_map_for_belief(
        self,
        token_ids:   torch.Tensor,
        belief:      BeliefState,
        s_belief:    'Optional[Tuple[torch.Tensor, torch.Tensor]]',
    ) -> Optional[torch.Tensor]:                # (H, N, N), or None
        if not self._model_channel_active:
            return None
        from vfe3.free_energy import attention_weights

        state = _sequence_belief(belief)
        s0 = None if s_belief is None else (s_belief[0][:1], s_belief[1][:1])
        tied_model_frame = self.cfg.s_frame_mode == "tied"
        model_phi = self._resolve_model_frame(token_ids[:1], state.phi.unsqueeze(0))
        e_s, gamma_tau, gamma_log_prior = self._gamma_energy(
            token_ids[:1], model_phi,
            omega=(state.omega.unsqueeze(0)
                   if tied_model_frame and state.omega is not None else None),
            reflection=(state.reflection.unsqueeze(0)
                        if tied_model_frame and state.reflection is not None else None),
            s_belief=s0,
        )
        gamma = attention_weights(e_s, tau=gamma_tau, log_prior=gamma_log_prior)[0]
        return gamma.unsqueeze(0) if gamma.dim() == 2 else gamma

    @torch.no_grad()
    def build_diagnostic_snapshot(
        self,
        token_ids: torch.Tensor,                 # (B, N) integer token ids
    ) -> DiagnosticSnapshot:
        r"""Capture one immutable evaluation snapshot from exactly one belief forward.

        The capture path records the encoded and stack-entry beliefs, every block's live prior,
        pre-transform converged belief and post-transform output, plus the first block's actual
        E-step states and free-energy values. Beta/gamma maps are then reduced from those captured
        tensors; no encode, E-step, block, or stack replay is performed.
        """
        diagnostic: dict = {}
        capture = {"diagnostic": diagnostic}
        final_belief, logits = self.forward_beliefs(
            token_ids, return_logits=True, capture=capture)
        if logits is None:
            raise RuntimeError("diagnostic snapshot requested logits but the decoder returned None")

        log_prior = diagnostic["log_prior"]
        log_prior0 = self._first_sequence_log_prior(log_prior, token_ids.shape[0])
        layer_outputs = tuple(diagnostic["layer_outputs"])
        beta_maps = torch.stack([
            self._attention_map_for_belief(_sequence_belief(layer), log_prior0, diagnostic["rope"])
            for layer in layer_outputs
        ])
        gamma_maps = self._gamma_map_for_belief(
            token_ids, capture["out"], diagnostic["s_belief"])
        trace = diagnostic["e_step_trace"]
        trace_free_energy = torch.stack([
            value.reshape(()) for value in trace["free_energy"]])
        s_belief = diagnostic["s_belief"]
        s_encoded_belief = None
        if self._model_channel_active:
            # The forward capture owns refined s1 under s_e_step. Preserve one static s0 lookup too
            # so report consumers can compare s0/s1 without replaying the model-channel E-step.
            s_encoded_belief = (self.prior_bank.encode_s(token_ids)
                                if self.cfg.s_e_step else s_belief)

        return DiagnosticSnapshot(
            owner=self,
            token_ids=_freeze_tensor(token_ids),
            encoded_belief=_freeze_belief(diagnostic["encoded_belief"]),
            initial_belief=_freeze_belief(diagnostic["initial_belief"]),
            layer_priors=tuple(
                (_freeze_tensor(mu_p), _freeze_tensor(sigma_p))
                for mu_p, sigma_p in diagnostic["layer_priors"]),
            layer_converged=tuple(
                _freeze_belief(belief) for belief in diagnostic["layer_converged"]),
            layer_outputs=tuple(_freeze_belief(belief) for belief in layer_outputs),
            stack_output=_freeze_belief(capture["out"]),
            final_belief=_freeze_belief(final_belief),
            logits=_freeze_tensor(logits),
            beta_maps=_freeze_tensor(beta_maps),
            gamma_maps=_freeze_tensor(gamma_maps),
            model_phi=_freeze_tensor(
                self._resolve_model_frame(token_ids, capture["out"].phi)),
            trace_states=tuple(_freeze_belief(belief) for belief in trace["beliefs"]),
            trace_free_energy=_freeze_tensor(trace_free_energy),
            s_encoded_belief=(None if s_encoded_belief is None else (
                _freeze_tensor(s_encoded_belief[0]), _freeze_tensor(s_encoded_belief[1]))),
            s_belief=(None if s_belief is None else (
                _freeze_tensor(s_belief[0]), _freeze_tensor(s_belief[1]))),
            rope=_freeze_tensor(diagnostic["rope"]),
            log_prior=_freeze_tensor(log_prior),
        )

    @torch.no_grad()
    def diagnostics(
        self,
        token_ids: torch.Tensor,           # (B, N) token ids; only sequence 0 is used

        *,
        log_likelihood: Optional[torch.Tensor] = None,  # (N,) optional E_q[log p(o|k)] diagnostic seam
        snapshot:       Optional[DiagnosticSnapshot] = None,  # one-forward captured evaluation state
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
        total, effective_rank}`` plus conditional ``twohop_coupling`` and
        ``observation_likelihood`` fields (nats; ``effective_rank`` is the per-token
        belief-variance spectrum effective rank, not an attention rank).
        """
        from vfe3.inference.e_step import _transport
        from vfe3.geometry.transport import transport_mean, compute_transport_operators
        from vfe3.families.base import get_family
        from vfe3.free_energy import pairwise_energy, self_divergence_for_alpha, attention_weights, attention_tau
        from vfe3.alpha_i import self_coupling_alpha
        from vfe3 import metrics
        from vfe3 import numerics

        cfg = self.cfg
        fam = get_family(cfg.family)
        if snapshot is None:
            enc = self.prior_bank.encode(token_ids[:1])                    # (1, N, ...)
            belief = BeliefState(mu=enc.mu[0], sigma=enc.sigma[0], phi=self._apply_pos_phi(enc.phi[0]),
                                 omega=enc.omega[0] if enc.omega is not None else None,
                                 reflection=enc.reflection[0] if enc.reflection is not None else None,
                                 right_phi=self._pos_phi_right(enc.phi[0]))
            initial_model_phi = self._resolve_model_frame(token_ids[:1], belief.phi.unsqueeze(0))
            s_belief = self._refined_s_belief(token_ids)
            if s_belief is not None:
                belief = belief._replace(mu=s_belief[0][0], sigma=s_belief[1][0])
            n = belief.mu.shape[0]
            log_prior = self._attention_log_prior(n, token_ids.device)
            log_prior = self._fold_precision_bias(log_prior, belief.sigma)
            if self.cfg.gamma_as_beta_prior:
                tied_model_frame = cfg.s_frame_mode == "tied"
                log_prior = self._fold_gamma_prior(
                    log_prior, token_ids[:1], initial_model_phi,
                    omega=(belief.omega.unsqueeze(0)
                           if tied_model_frame and belief.omega is not None else None),
                    reflection=(belief.reflection.unsqueeze(0)
                                if tied_model_frame and belief.reflection is not None else None),
                    s_belief=s_belief)[0]
            rope = self._rope_rotation(n, token_ids.device)
            cap: dict = {}
            out = vfe_stack(
                belief, belief.mu, belief.sigma, self.group, cfg,
                log_prior=log_prior, block_norm=self.block_norm,
                head_mixer=self.head_mixer, cg_coupling=self.cg_coupling,
                lambda_beta=cfg.lambda_beta,
                connection_W=getattr(self, "connection_W", None),
                connection_M=getattr(self, "connection_M", None),
                connection_L=getattr(self, "connection_L", None),
                rope=rope, rope_on_cov=cfg.rope_full_gauge,
                rope_on_value=cfg.rope_on_value,
                capture=cap,
                transport_status=self._transport_status,
                gauge_parameterization=cfg.gauge_parameterization,
                kappa_beta_override=self.effective_kappa_beta(belief.mu.device),
            )
        else:
            snapshot = self._validate_diagnostic_snapshot(token_ids, snapshot)
            belief = _sequence_belief(snapshot.initial_belief)
            s_belief = None if snapshot.s_belief is None else (
                snapshot.s_belief[0][:1],
                snapshot.s_belief[1][:1],
            )
            n = belief.mu.shape[0]
            log_prior = self._first_sequence_log_prior(snapshot.log_prior, token_ids.shape[0])
            rope = snapshot.rope
            out = _sequence_belief(snapshot.stack_output)
            cap = {"converged": _sequence_belief(snapshot.layer_converged[-1])}

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
            connection_L=getattr(self, "connection_L", None),
            link_alpha=cfg.link_alpha, link_soft_cap=cfg.link_soft_cap,
            clamp_monitor=cfg.transport_clamp_monitor,
            validity_max_norm=cfg.transport_chart_max_norm,
            exactness_out=self._transport_status,
            cocycle_relaxation=cfg.cocycle_relaxation,
            gauge_parameterization=cfg.gauge_parameterization,
            omega=out.omega,                                          # omega_direct: Omega_ij = U_i U_j^{-1} (det<0 visible)
            reflection=out.reflection,                                # phi-path R_i Omega_ij R_j fold (None -> unchanged)
            right_phi=out.right_phi,
        )
        mu_tv = sigma_tv = None
        if rope is not None:
            rope_omega = RopeTransport(base=omega, rope=rope, on_cov=cfg.rope_full_gauge,
                                       on_value=cfg.rope_on_value)
            mu_t    = transport_mean(rope_omega, out.mu)             # (N, N, K)
            sigma_t = fam.transport_dispersion(out.sigma, rope_omega) # (N, N, K)
            if not cfg.rope_on_value:
                mu_tv    = transport_mean(omega, out.mu)
                sigma_tv = fam.transport_dispersion(out.sigma, omega)
        else:
            mu_t    = transport_mean(omega.unsqueeze(0), out.mu.unsqueeze(0))[0]
            sigma_t = fam.transport_dispersion(out.sigma.unsqueeze(0), omega.unsqueeze(0))[0]
        energy = pairwise_energy(                                    # (N, N) or (H, N, N)
            fam(out.mu, out.sigma), fam(mu_t, sigma_t),
            alpha=cfg.renyi_order, kl_max=cfg.kl_max, eps=cfg.eps,
            divergence_family=cfg.divergence_family,
            irrep_dims=self.group.irrep_dims,
        )
        coupling_energy = None
        if mu_tv is not None:
            coupling_energy = pairwise_energy(
                fam(out.mu, out.sigma), fam(mu_tv, sigma_tv),
                alpha=cfg.renyi_order, kl_max=cfg.kl_max, eps=cfg.eps,
                divergence_family=cfg.divergence_family,
                irrep_dims=self.group.irrep_dims,
            )
        # query_adaptive_tau (default OFF): _beta_tau returns the base tau unchanged, else the
        # per-query tau from the CONVERGED belief sigma (the state this diagnostic scores).
        _tau_b = self._beta_tau(out.sigma, out.mu,
                                attention_tau(self.effective_kappa_beta(out.mu.device), self.group.irrep_dims))
        if snapshot is None:
            beta = attention_weights(energy, tau=_tau_b, log_prior=log_prior)
        else:
            beta = snapshot.beta_maps[-1]
            if energy.dim() == 2:
                beta = beta[0]
        _q_conv = cap["converged"]                                   # q*: the F self-term reads the
        self_div = self_divergence_for_alpha(                        # pre-transform converged belief
            fam(_q_conv.mu, _q_conv.sigma), fam(mu_p, sigma_p),      # (matches the M-step term; F19)
            alpha=cfg.renyi_order, kl_max=cfg.kl_max, eps=cfg.eps,
            divergence_family=cfg.divergence_family, lambda_alpha_mode=cfg.lambda_alpha_mode,
        )
        alpha, alpha_reg = self_coupling_alpha(
            self_div, mode=cfg.lambda_alpha_mode, value=cfg.lambda_alpha, b0=_as_coeff(cfg.b0, out.mu.device), c0=_as_coeff(cfg.c0, out.mu.device),
        )

        # PB-07: route the migrated numeric diagnostics (row entropy, free-energy decomposition,
        # belief effective rank) through the metric registry instead of three direct calls, so a
        # config-selected override dispatches here too. The context carries exactly the inputs the
        # three wrappers read. ``_diag`` is the dimension-based diagonal(N,K)-vs-full(N,K,K) flag
        # passed explicitly to effective_rank because shape-squareness auto-inference mis-reads a
        # diagonal (N, K) table as a full covariance when N == K (e.g. max_seq_len == embed_dim).
        # ``attention_entropy`` (row entropy) is aliased to ``attn_entropy`` so it does not overwrite
        # the free-energy component of the same name that ``free_energy_terms`` flattens in; the
        # flatten carries an overwrite-collision guard. Holonomy/gauge metrics keep their bespoke
        # paths below (they consume sibling confidence bounds / active-frame branches).
        _lb = cfg.lambda_beta   # scaled-F total reflects lambda_beta
        _diag = out.sigma.dim() == out.mu.dim()                     # diagonal (N,K) vs full (N,K,K)
        metric_context = {
            "sigma":                     out.sigma,
            "diagonal":                  _diag,
            "family":                    cfg.family,
            "self_div":                  self_div,
            "energy":                    energy,
            "beta":                      beta,
            "alpha":                     alpha,
            "tau":                       _tau_b,
            "lambda_beta":               _lb,
            "lambda_twohop":             cfg.lambda_twohop,
            "include_attention_entropy": cfg.include_attention_entropy,
            "log_prior":                 log_prior,
            "alpha_reg":                 (alpha_reg if cfg.lambda_alpha_mode != "constant" else None),
            "coupling_energy":           coupling_energy,
            "log_likelihood":            log_likelihood,
        }
        registered = metrics.compute_metrics(list(metrics.DIAGNOSTIC_METRIC_NAMES), **metric_context)
        d: Dict[str, float] = {}
        for metric_name, output_name, flatten in metrics.DIAGNOSTIC_METRIC_OUTPUTS:
            value = registered[metric_name]
            if flatten:
                overlap = set(value) & set(d)
                if overlap:
                    raise KeyError(f"diagnostic metric {metric_name!r} would overwrite {sorted(overlap)}")
                d.update({key: float(item) for key, item in value.items()})
            else:
                d[output_name] = float(value)
        d["regime_ii_covariant_feature_exact"] = float(bool(
            self._transport_status["regime_ii_covariant_feature_exact"]))
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
        # The whole q/p/s/h decomposition is reassembled ONCE through the single typed evaluator
        # (PB-10): the live hyper-prior/gamma rows join the per-query belief rows (built from the
        # captured beta) with q_reduction='sum', model_reduction='sum' -- the persisted per-sequence
        # SUM scale -- instead of incrementing d["total"] independently per model block.
        hyper_active = cfg.lambda_h > 0.0 or cfg.s_e_step            # r table exists on this path
        gamma_active = cfg.lambda_gamma > 0.0 or cfg.s_e_step        # gamma block evaluated at out.phi
        if hyper_active or gamma_active:
            from vfe3.alpha_i import alpha_is_per_coord
            from vfe3.free_energy import _belief_free_energy_rows, hierarchical_free_energy_terms
            belief_rows = _belief_free_energy_rows(                  # (N,) q rows for seq 0 (reuse beta)
                self_div, energy, alpha, tau=_tau_b, lambda_beta=_lb,
                lambda_twohop=cfg.lambda_twohop,
                include_attention_entropy=cfg.include_attention_entropy,
                log_prior=log_prior,
                alpha_reg=(alpha_reg if cfg.lambda_alpha_mode != "constant" else None),
                coupling_energy=coupling_energy, log_likelihood=log_likelihood,
                beta_override=beta,
                per_coord=alpha_is_per_coord(cfg.lambda_alpha_mode))  # exact axis truth (no inference)
            zeros = torch.zeros_like(belief_rows.self_coupling)      # (N,)
            hyper_prior_rows = zeros
            model_coupling_rows = zeros
            meta_entropy_rows = zeros
            if hyper_active:
                d["hyper_prior"] = float(self._hyper_prior_kl(token_ids[:1], s_belief=s_belief).sum())   # sum_i KL(s_i||r) (refined s1 under s_e_step)
                hyper_prior_rows = self._hyper_prior_weighted(token_ids[:1], s_belief=s_belief)[0]        # (N,) WEIGHTED (lambda_h_mode); == cfg.lambda_h*KL for 'constant'
                d["hyper_prior_weighted"] = float(hyper_prior_rows.sum())                                 # EXACT contribution folded into total; the F-decomposition figure reads this
            if gamma_active:
                # Executable gradient-scope metadata: gamma is evaluated at the live frame value,
                # but the production objective detaches that frame before this block. Direct
                # gamma-to-frame optimization is therefore intentionally passive on every route.
                d["gamma_direct_frame_grad_active"] = 0.0
                tied_model_frame = cfg.s_frame_mode == "tied"
                gamma_model_phi = (self._resolve_model_frame(token_ids[:1], out.phi.unsqueeze(0))
                                   if snapshot is None else snapshot.model_phi[:1])
                c_rows, me_rows = self._gamma_coupling_rows(         # (1, N) rows: sum over heads
                    token_ids[:1], gamma_model_phi, head_reduction="sum",
                    omega=(out.omega.unsqueeze(0)
                           if tied_model_frame and out.omega is not None else None),
                    reflection=(out.reflection.unsqueeze(0)
                                if tied_model_frame and out.reflection is not None else None),
                    s_belief=s_belief)
                d["gamma_coupling"]     = float(c_rows[0].sum())     # raw sum_{h,i,j} gamma E^s
                d["gamma_meta_entropy"] = float(me_rows[0].sum())    # raw sum_{h,i,j} tau_g gamma log(gamma/pi^s)
                model_coupling_rows = cfg.lambda_gamma * c_rows[0]   # (N,)
                meta_entropy_rows = cfg.lambda_gamma * me_rows[0]
            d["total"] = float(hierarchical_free_energy_terms(
                belief_rows.self_coupling, belief_rows.belief_coupling,
                belief_rows.attention_entropy, belief_rows.twohop_coupling,
                hyper_prior_rows, model_coupling_rows, meta_entropy_rows,
                belief_rows.observation_nll,
                q_reduction="sum", model_reduction="sum").total)
        # Gauge-geometry probes (diagnostics tier): the curvature proxy -- mean Frobenius departure
        # of the triangle holonomy Omega_ij Omega_jk Omega_ki from I (0 for the flat phi-cocycle) --
        # and the spread of log|det Omega| = tr(embed(phi)) across tokens (0 at phi=0). Pure
        # measurements at the converged transport; off the training graph (no_grad).
        # Curvature proxy from the SAMPLED estimator (seeded random distinct triples) rather than the
        # deterministic row-major prefix, which at N=128/max_triangles=512 covers only anchor i=0's
        # local neighborhood -- a systematically biased sample. The sampled mean is representative and
        # still ~0 on the flat phi-cocycle (flatness certificate); the dict key is unchanged.
        # ---- extended per-eval observability (2026-06-13 run-diagnostics rollout) ----
        # Every reduction below reads tensors already materialized above (out.mu/sigma/active frame,
        # omega, energy, beta, self_div); no extra forward, no_grad. NEW keys only -- d["total"] and
        # the existing block values are untouched (test_model_channel_diagnostics pins total's closure;
        # test_regime_ii pins d["holonomy_deviation"], whose semantics is preserved as the mean below).
        _LOG2 = 0.6931471805599453                                   # row-entropy floor for a 2-way split
        # ``_diag`` (diagonal (N,K) vs full (N,K,K)) is computed once in the registry-dispatch block
        # above and reused by the spectrum/Fisher/guard metrics below; it is passed explicitly because
        # shape-squareness auto-inference mis-reads a diagonal (N, K) table as full when N == K.

        hol = metrics.holonomy_deviation_sampled(omega)
        d["holonomy_deviation"] = float(hol["mean"])                 # unchanged key/semantics
        d["holonomy_ci_lo"]     = float(hol["ci_lo"])                # bootstrap band: real curvature vs jitter
        d["holonomy_ci_hi"]     = float(hol["ci_hi"])
        # Manuscript-canonical gauge invariant: the Wilson-action density 1 - Re Tr(H)/K (PIFB:862-869),
        # the trace complement of the Frobenius certificate above; ~0 on the flat cocycle, > 0 under regime_ii.
        d["holonomy_wilson"]    = float(metrics.holonomy_wilson_sampled(omega)["deviation_mean"])
        # Active-frame health. Under omega_direct, ``phi`` is an inactive table and can disagree
        # arbitrarily with the stored element; derive every frame invariant from ``out.omega`` and do
        # not exponentiate inactive coordinates. Compact blocks are reduced directly as the represented
        # block-diagonal element. The phi path uses the active reflected vertex when present.
        if out.omega is not None:
            if isinstance(out.omega, CompactBlockElement):
                active_blocks = out.omega.expanded_blocks()                    # (N,H,d,d)
                block_logdet = torch.linalg.slogdet(active_blocks).logabsdet    # (N,H)
                block_svd = torch.linalg.svdvals(active_blocks)                 # (N,H,d)
                full_logdet = block_logdet.sum(dim=-1)                          # (N,)
                d["gauge_trace_spread"] = float(full_logdet.std(unbiased=False))
                ginv = full_logdet.float()                                      # compact owners are GL blocks
                vertex_cond = (
                    block_svd[..., 0].amax(dim=-1)
                    / block_svd[..., -1].amin(dim=-1).clamp(min=cfg.eps))
                _ghi = {
                    "logdet": block_logdet,
                    "anisotropy": block_svd[..., 0] / block_svd[..., -1].clamp(min=cfg.eps),
                }
            else:
                active_vertex = out.omega                                       # (N,K,K) stored element
                active_logdet = torch.linalg.slogdet(active_vertex).logabsdet
                d["gauge_trace_spread"] = float(active_logdet.std(unbiased=False))
                ginv = metrics.group_gauge_invariant(active_vertex, self.group).float()
                active_svd = torch.linalg.svdvals(active_vertex)
                vertex_cond = active_svd[..., 0] / active_svd[..., -1].clamp(min=cfg.eps)
                _ghi = metrics.per_head_gauge_invariants(active_vertex, self.group.irrep_dims)
        else:
            if out.right_phi is not None:
                from vfe3.geometry.transport import build_factored_transport
                active_vertex = build_factored_transport(
                    out.phi, self.group,
                    exp_fp64_mode=cfg.exp_fp64_mode,
                    exp_fp64_norm_threshold=cfg.exp_fp64_norm_threshold,
                    clamp_monitor=cfg.transport_clamp_monitor,
                    validity_max_norm=cfg.transport_chart_max_norm,
                    right_phi=out.right_phi,
                ).exp_phi
            else:
                active_vertex = compute_transport_operators(
                    out.phi.unsqueeze(0), self.group,
                    validity_max_norm=cfg.transport_chart_max_norm)["exp_phi"][0]  # (N,K,K)
            if out.reflection is not None:
                # Active disconnected-component frame g_i = R_i exp(phi_i). Scaling row zero
                # applies the left factor R_i = diag(sign_i, 1, ...) used by the transport fold.
                active_vertex = active_vertex.clone()
                active_vertex[..., 0, :] *= out.reflection[..., None]
            active_logdet = torch.linalg.slogdet(active_vertex).logabsdet
            d["gauge_trace_spread"] = float(active_logdet.std(unbiased=False))
            ginv = metrics.group_gauge_invariant(active_vertex, self.group).float()
            active_svd = torch.linalg.svdvals(active_vertex)
            vertex_cond = active_svd[..., 0] / active_svd[..., -1].clamp(min=cfg.eps)
            _ghi = metrics.per_head_gauge_invariants(active_vertex, self.group.irrep_dims)
        d["gauge_invariant_mean"]   = float(ginv.mean())
        d["gauge_invariant_spread"] = float(ginv.std(unbiased=False))

        # Transport DIRECTEDNESS + conditioning + sandwich overflow. NOTE: directedness
        # (transport_asymmetry, energy_*_asymmetry) is intrinsic to ANY nonzero gauge -- it is nonzero
        # on the FLAT cocycle (Omega_ji = Omega_ij^{-1} != Omega_ij), so it is NOT a curvature /
        # non-flatness signal; cocycle_residual and holonomy_deviation are the flatness diagnostics.
        d["cocycle_residual"] = float(metrics.cocycle_residual_sampled(omega))   # composition-law flatness
        d["vertex_cond_max"]  = float(vertex_cond.max())
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
        d["gauge_head_aniso_mean"]    = float(_ghi["anisotropy"].float().mean())
        d["gauge_head_logdet_spread"] = float(_ghi["logdet"].float().std(unbiased=False))

        # phi frame magnitude: a collapse to phi=0 silently degenerates to an UNGAUGED transformer
        # (trivially equivariant, so no equivariance metric flags it).
        phi_norm = torch.linalg.norm(out.phi, dim=-1)               # (N,)
        d["phi_norm_mean"] = float(phi_norm.mean())
        d["phi_norm_std"]  = float(phi_norm.std(unbiased=False))
        if cfg.gauge_parameterization == "phi":
            from vfe3.geometry.transport import TRANSPORT_CLAMP_MAX_NORM
            phi_matrix = torch.einsum(
                "na,aij->nij", out.phi.float(), self.group.generators.float())
            matrix_norm = torch.linalg.matrix_norm(phi_matrix, ord="fro", dim=(-2, -1))
            exp_scale = (
                TRANSPORT_CLAMP_MAX_NORM / matrix_norm.clamp(min=cfg.eps)
            ).clamp(max=1.0)
            d["phi_matrix_norm_median"] = float(matrix_norm.median())
            d["phi_matrix_norm_p95"] = float(torch.quantile(matrix_norm, 0.95))
            d["phi_matrix_norm_p99"] = float(torch.quantile(matrix_norm, 0.99))
            d["phi_matrix_norm_max"] = float(matrix_norm.max())
            d["phi_exp_clamp_frac"] = float((exp_scale < 1.0).float().mean())
            d["phi_exp_scale_min"] = float(exp_scale.min())
            d["vertex_cond_median"] = float(vertex_cond.float().median())
            d["vertex_cond_p95"] = float(torch.quantile(vertex_cond.float(), 0.95))
            d["vertex_cond_p99"] = float(torch.quantile(vertex_cond.float(), 0.99))
            if out.right_phi is not None:
                right_matrix = torch.einsum(
                    "na,aij->nij", out.right_phi.float(), self.group.generators.float(),
                )
                right_norm = torch.linalg.matrix_norm(right_matrix, ord="fro", dim=(-2, -1))
                right_scale = (
                    TRANSPORT_CLAMP_MAX_NORM / right_norm.clamp(min=cfg.eps)
                ).clamp(max=1.0)
                d["pos_phi_matrix_norm_p95"] = float(torch.quantile(right_norm, 0.95))
                d["pos_phi_matrix_norm_p99"] = float(torch.quantile(right_norm, 0.99))
                d["pos_phi_matrix_norm_max"] = float(right_norm.max())
                d["pos_phi_exp_clamp_frac"] = float((right_scale < 1.0).float().mean())
                d["pos_phi_exp_scale_min"] = float(right_scale.min())

        # Belief covariance conditioning + PD margin (effective_rank is blind to one collapsing mode).
        bs = metrics.belief_spectrum(
            out.sigma,
            diagonal=_diag,
            eps=cfg.eps,
            family=cfg.family,
        )
        cond = bs["condition"].float()
        d["belief_cond_median"] = float(cond.median())
        d["belief_cond_p95"]    = float(torch.quantile(cond, 0.95))
        d["belief_cond_max"]    = float(cond.max())
        # clamp lam_min at eps before dividing -- matches the floor belief_spectrum's condition number
        # uses, so a floored / sub-floor belief reads ~1.0 (not 0.0) consistently across the reductions.
        d["belief_pd_margin"]   = float((bs["eigenvalues"][..., -1].clamp(min=cfg.eps).float() / cfg.eps).min())

        # Per-token effective-rank distribution (the logged mean hides a bimodal rank-1/rank-K collapse).
        er = metrics.effective_rank_per_token(
            out.sigma,
            diagonal=_diag,
            eps=cfg.eps,
            family=cfg.family,
        ).float()
        d["eff_rank_p5"]     = float(torch.quantile(er, 0.05))
        d["eff_rank_median"] = float(er.median())
        d["eff_rank_p95"]    = float(torch.quantile(er, 0.95))

        # One-half mean-block Fisher trace (the KL quadratic coefficient and belief precision).
        fish = metrics.half_fisher_trace(
            out.sigma,
            diagonal=_diag,
            eps=cfg.eps,
            family=cfg.family,
        ).float()
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
        _cL = getattr(self, "connection_L", None)
        if _cL is not None:                                          # transport_mode='regime_ii_link' / '_charted'
            d["connection_l_norm"]         = float(torch.linalg.norm(_cL.detach()))
            d["connection_l_offdiag_norm"] = float(torch.linalg.norm(
                _cL.detach()[~torch.eye(_cL.shape[0], dtype=torch.bool, device=_cL.device)]))
        _hm = getattr(self, "head_mixer", None)
        if _hm is not None and hasattr(_hm, "mixer_deltas"):        # use_head_mixer=True
            d["head_mixer_drift"] = max(
                (float(torch.linalg.norm(p.detach())) for p in _hm.mixer_deltas), default=0.0)
        return d

    @torch.no_grad()
    def attention_maps(
        self,
        token_ids: torch.Tensor,           # (B, N) token ids; only sequence 0 is used

        *,
        snapshot:  Optional[DiagnosticSnapshot] = None,
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
        if snapshot is not None:
            return self._validate_diagnostic_snapshot(token_ids, snapshot).beta_maps
        from vfe3.inference.e_step import _transport
        from vfe3.geometry.transport import transport_mean
        from vfe3.families.base import get_family
        from vfe3.free_energy import pairwise_energy, attention_weights, attention_tau

        cfg = self.cfg
        enc = self.prior_bank.encode(token_ids[:1])                   # (1, N, ...)
        belief = BeliefState(mu=enc.mu[0], sigma=enc.sigma[0], phi=self._apply_pos_phi(enc.phi[0]),
                             omega=enc.omega[0] if enc.omega is not None else None,   # carry the GL(K) frame under omega_direct
                             reflection=enc.reflection[0] if enc.reflection is not None else None,
                             right_phi=self._pos_phi_right(enc.phi[0]))
        model_phi = self._resolve_model_frame(token_ids[:1], belief.phi.unsqueeze(0))
        n = belief.mu.shape[0]
        rope = self._rope_rotation(n, token_ids.device)
        s_belief = None
        if cfg.s_e_step:
            # Live model channel (audit 2026-06-09 IE1): refine s and anchor the replayed belief
            # (q0 AND the handoff prior below) to it, exactly as forward/diagnostics do, so the
            # figure attention replays the model that actually trained.
            s_mu1, s_sigma1 = self._refine_s(token_ids[:1], model_phi, rope=rope)
            s_belief = (s_mu1, s_sigma1)
            belief = belief._replace(mu=s_mu1[0], sigma=s_sigma1[0])
        log_prior = self._attention_log_prior(n, token_ids.device)   # (N, N)
        log_prior = self._fold_precision_bias(log_prior, belief.sigma)  # match forward's prior (r2 id22)
        if self.cfg.gamma_as_beta_prior:                             # m4: match forward's hierarchical gamma prior fold
            tied_model_frame = cfg.s_frame_mode == "tied"
            log_prior = self._fold_gamma_prior(log_prior, token_ids[:1], model_phi,
                                               omega=(belief.omega.unsqueeze(0)
                                                      if tied_model_frame and belief.omega is not None else None),
                                               reflection=(belief.reflection.unsqueeze(0)
                                                           if tied_model_frame and belief.reflection is not None else None),
                                               s_belief=s_belief)[0]
        fam = get_family(cfg.family)
        rho, rho_s = cfg.prior_handoff_rho, cfg.prior_handoff_sigma
        mu_p, sigma_p = belief.mu, belief.sigma

        _base_tau = attention_tau(self.effective_kappa_beta(belief.mu.device), self.group.irrep_dims)
        maps = []
        for _ in range(cfg.n_layers):
            belief = vfe_block(                                       # converged belief at this block
                belief, mu_p, sigma_p, self.group, cfg, log_prior=log_prior,
                block_norm=self.block_norm,
                head_mixer=self.head_mixer,                            # replay the mixer too (audit 2026-06-09 overnight F32)
                lambda_beta=cfg.lambda_beta,
                connection_W=getattr(self, "connection_W", None),
                connection_M=getattr(self, "connection_M", None),     # learned covariant (Route B) connection
                connection_L=getattr(self, "connection_L", None),     # learned direct link
                cg_coupling=self.cg_coupling,
                rope=rope, rope_on_cov=cfg.rope_full_gauge,            # match forward: converge WITH rope
                rope_on_value=cfg.rope_on_value,
                transport_status=self._transport_status,
                gauge_parameterization=cfg.gauge_parameterization,
                # query_adaptive_tau replay fidelity: the ENTERING belief's per-query tau, exactly as
                # vfe_stack passes the forward E-step; OFF path returns _base_tau (value-identical to
                # the tau vfe_block would compute itself).
                tau=self._beta_tau(belief.sigma, belief.mu, _base_tau),
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
                connection_L=getattr(self, "connection_L", None),
                link_alpha=cfg.link_alpha, link_soft_cap=cfg.link_soft_cap,
                clamp_monitor=cfg.transport_clamp_monitor,
                validity_max_norm=cfg.transport_chart_max_norm,
                exactness_out=self._transport_status,
                cocycle_relaxation=cfg.cocycle_relaxation,
                gauge_parameterization=cfg.gauge_parameterization,
                omega=belief.omega,                                  # omega_direct: Omega_ij = U_i U_j^{-1} (det<0 visible)
                reflection=belief.reflection,                        # phi-path R_i Omega_ij R_j fold (None -> unchanged)
                right_phi=belief.right_phi,
            )                                                        # (N, N, K, K)
            if rope is not None:
                rope_omega = RopeTransport(base=omega, rope=rope, on_cov=cfg.rope_full_gauge,
                                           on_value=cfg.rope_on_value)
                mu_t    = transport_mean(rope_omega, belief.mu)          # (N, N, K)
                sigma_t = fam.transport_dispersion(belief.sigma, rope_omega) # (N, N, K)
            else:
                mu_t    = transport_mean(omega.unsqueeze(0), belief.mu.unsqueeze(0))[0]
                sigma_t = fam.transport_dispersion(
                    belief.sigma.unsqueeze(0), omega.unsqueeze(0))[0]
            energy = pairwise_energy(                                 # (N, N) or (H, N, N)
                fam(belief.mu, belief.sigma), fam(mu_t, sigma_t),
                alpha=cfg.renyi_order, kl_max=cfg.kl_max, eps=cfg.eps,
                divergence_family=cfg.divergence_family, irrep_dims=self.group.irrep_dims,
            )
            beta = attention_weights(energy, tau=self._beta_tau(belief.sigma, belief.mu, _base_tau),
                                     log_prior=log_prior)            # converged-belief tau (as diagnostics)
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

        *,
        log_likelihood: Optional[torch.Tensor] = None,  # (N,) optional E_q[log p(o|k)] diagnostic seam
        snapshot:       Optional[DiagnosticSnapshot] = None,
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
        from vfe3.geometry.transport import transport_mean, compute_transport_operators
        from vfe3.families.base import get_family
        from vfe3.free_energy import (pairwise_energy, self_divergence_for_alpha,
                                      attention_weights, attention_tau)
        from vfe3.alpha_i import self_coupling_alpha
        from vfe3 import metrics

        cfg = self.cfg
        if snapshot is None:
            enc = self.prior_bank.encode(token_ids[:1])                   # (1, N, ...)
            belief = BeliefState(mu=enc.mu[0], sigma=enc.sigma[0], phi=self._apply_pos_phi(enc.phi[0]),
                                 omega=enc.omega[0] if enc.omega is not None else None,
                                 reflection=enc.reflection[0] if enc.reflection is not None else None,
                                 right_phi=self._pos_phi_right(enc.phi[0]))
            model_phi = self._resolve_model_frame(token_ids[:1], belief.phi.unsqueeze(0))
            n = belief.mu.shape[0]
            rope = self._rope_rotation(n, token_ids.device)
            s_belief = None
            if cfg.s_e_step:
                s_mu1, s_sigma1 = self._refine_s(token_ids[:1], model_phi, rope=rope)
                s_belief = (s_mu1, s_sigma1)
                belief = belief._replace(mu=s_mu1[0], sigma=s_sigma1[0])
            log_prior = self._attention_log_prior(n, token_ids.device)
            log_prior = self._fold_precision_bias(log_prior, belief.sigma)
            if self.cfg.gamma_as_beta_prior:
                tied_model_frame = cfg.s_frame_mode == "tied"
                log_prior = self._fold_gamma_prior(
                    log_prior, token_ids[:1], model_phi,
                    omega=(belief.omega.unsqueeze(0)
                           if tied_model_frame and belief.omega is not None else None),
                    reflection=(belief.reflection.unsqueeze(0)
                                if tied_model_frame and belief.reflection is not None else None),
                    s_belief=s_belief)[0]
        else:
            snapshot = self._validate_diagnostic_snapshot(token_ids, snapshot)
            belief = _sequence_belief(snapshot.initial_belief)
            n = belief.mu.shape[0]
            rope = snapshot.rope
            log_prior = self._first_sequence_log_prior(snapshot.log_prior, token_ids.shape[0])
        fam = get_family(cfg.family)
        _lb = cfg.lambda_beta
        _tau = attention_tau(self.effective_kappa_beta(belief.mu.device), self.group.irrep_dims)
        rho, rho_s = cfg.prior_handoff_rho, cfg.prior_handoff_sigma
        mu_p, sigma_p = belief.mu, belief.sigma
        keys = ["self_coupling", "belief_coupling", "attention_entropy", "total", "self_divergence",
                "holonomy_deviation", "holonomy_wilson", "gauge_trace_spread", "gauge_invariant_spread",
                "effective_rank", "attn_entropy", "belief_cond_median", "phi_norm_mean"]
        if cfg.lambda_twohop != 0.0:
            keys.append("twohop_coupling")
        if log_likelihood is not None:
            keys.append("observation_likelihood")
        rec: dict = {k: [] for k in keys}
        for layer_index in range(cfg.n_layers):
            if snapshot is None:
                cap: dict = {}
                belief = vfe_block(
                    belief, mu_p, sigma_p, self.group, cfg, log_prior=log_prior,
                    block_norm=self.block_norm, head_mixer=self.head_mixer, cg_coupling=self.cg_coupling,
                    lambda_beta=cfg.lambda_beta,
                    connection_W=getattr(self, "connection_W", None),
                    connection_M=getattr(self, "connection_M", None),
                    connection_L=getattr(self, "connection_L", None),
                    rope=rope, rope_on_cov=cfg.rope_full_gauge, rope_on_value=cfg.rope_on_value,
                    transport_status=self._transport_status,
                    gauge_parameterization=cfg.gauge_parameterization,
                    tau=self._beta_tau(belief.sigma, belief.mu, _tau),
                    capture=cap,
                )
            else:
                prior = snapshot.layer_priors[layer_index]
                mu_p, sigma_p = prior[0][0], prior[1][0]
                belief = _sequence_belief(snapshot.layer_outputs[layer_index])
                cap = {"converged": _sequence_belief(snapshot.layer_converged[layer_index])}
            _tau_c = self._beta_tau(belief.sigma, belief.mu, _tau)   # converged-belief tau (as diagnostics)
            omega = _transport(                                       # (N, N, K, K) under the ACTIVE regime
                belief.phi, self.group, transport_mode=cfg.transport_mode,
                mu=(belief.mu if cfg.transport_mode in _REGIME_NEEDS_MU else None),
                sigma=(belief.sigma if cfg.transport_mode in _REGIME_NEEDS_SIGMA else None),
                connection_W=getattr(self, "connection_W", None),
                connection_M=getattr(self, "connection_M", None),
                connection_L=getattr(self, "connection_L", None),
                link_alpha=cfg.link_alpha, link_soft_cap=cfg.link_soft_cap,
                clamp_monitor=cfg.transport_clamp_monitor,
                validity_max_norm=cfg.transport_chart_max_norm,
                exactness_out=self._transport_status,
                cocycle_relaxation=cfg.cocycle_relaxation,
                gauge_parameterization=cfg.gauge_parameterization,
                omega=belief.omega,                                  # omega_direct: Omega_ij = U_i U_j^{-1} (det<0 visible)
                reflection=belief.reflection,                        # phi-path R_i Omega_ij R_j fold (None -> unchanged)
                right_phi=belief.right_phi,
            )
            mu_tv = sigma_tv = None
            if rope is not None:
                rope_omega = RopeTransport(base=omega, rope=rope, on_cov=cfg.rope_full_gauge,
                                           on_value=cfg.rope_on_value)
                mu_t    = transport_mean(rope_omega, belief.mu)
                sigma_t = fam.transport_dispersion(belief.sigma, rope_omega)
                if not cfg.rope_on_value:
                    mu_tv    = transport_mean(omega, belief.mu)
                    sigma_tv = fam.transport_dispersion(belief.sigma, omega)
            else:
                mu_t    = transport_mean(omega.unsqueeze(0), belief.mu.unsqueeze(0))[0]
                sigma_t = fam.transport_dispersion(
                    belief.sigma.unsqueeze(0), omega.unsqueeze(0))[0]
            energy = pairwise_energy(                                 # (N, N) or (H, N, N)
                fam(belief.mu, belief.sigma), fam(mu_t, sigma_t),
                alpha=cfg.renyi_order, kl_max=cfg.kl_max, eps=cfg.eps,
                divergence_family=cfg.divergence_family, irrep_dims=self.group.irrep_dims,
            )
            coupling_energy = None
            if mu_tv is not None:
                coupling_energy = pairwise_energy(
                    fam(belief.mu, belief.sigma), fam(mu_tv, sigma_tv),
                    alpha=cfg.renyi_order, kl_max=cfg.kl_max, eps=cfg.eps,
                    divergence_family=cfg.divergence_family,
                    irrep_dims=self.group.irrep_dims,
                )
            if snapshot is None:
                beta = attention_weights(energy, tau=_tau_c, log_prior=log_prior)
            else:
                beta = snapshot.beta_maps[layer_index]
                if energy.dim() == 2:
                    beta = beta[0]
            _q = cap["converged"]                                    # self-term reads THIS block's prior (per-layer exact)
            self_div = self_divergence_for_alpha(
                fam(_q.mu, _q.sigma), fam(mu_p, sigma_p),
                alpha=cfg.renyi_order, kl_max=cfg.kl_max, eps=cfg.eps,
                divergence_family=cfg.divergence_family, lambda_alpha_mode=cfg.lambda_alpha_mode,
            )
            alpha, alpha_reg = self_coupling_alpha(
                self_div, mode=cfg.lambda_alpha_mode, value=cfg.lambda_alpha,
                b0=_as_coeff(cfg.b0, belief.mu.device), c0=_as_coeff(cfg.c0, belief.mu.device),
            )
            terms = metrics.free_energy_terms(
                self_div, energy, beta, alpha, tau=_tau_c, lambda_beta=_lb, log_prior=log_prior,
                lambda_twohop=cfg.lambda_twohop,
                include_attention_entropy=cfg.include_attention_entropy,
                alpha_reg=(alpha_reg if cfg.lambda_alpha_mode != "constant" else None),
                coupling_energy=coupling_energy,
                log_likelihood=log_likelihood,
            )
            rec["self_coupling"].append(float(terms["self_coupling"]))
            rec["belief_coupling"].append(float(terms["belief_coupling"]))
            rec["attention_entropy"].append(float(terms["attention_entropy"]))
            rec["total"].append(float(terms["total"]))
            if "twohop_coupling" in terms:
                rec["twohop_coupling"].append(float(terms["twohop_coupling"]))
            if "observation_likelihood" in terms:
                rec["observation_likelihood"].append(float(terms["observation_likelihood"]))
            rec["self_divergence"].append(float(self_div.sum()))
            rec["holonomy_deviation"].append(float(metrics.holonomy_deviation_sampled(omega)["mean"]))
            rec["holonomy_wilson"].append(float(metrics.holonomy_wilson_sampled(omega)["deviation_mean"]))
            if belief.omega is not None:
                if isinstance(belief.omega, CompactBlockElement):
                    block_logdet = torch.linalg.slogdet(
                        belief.omega.expanded_blocks()).logabsdet
                    active_invariant = block_logdet.sum(dim=-1).float()
                else:
                    active_logdet = torch.linalg.slogdet(belief.omega).logabsdet
                    active_invariant = metrics.group_gauge_invariant(
                        belief.omega, self.group).float()
                rec["gauge_trace_spread"].append(
                    float(active_logdet.std(unbiased=False))
                    if not isinstance(belief.omega, CompactBlockElement)
                    else float(active_invariant.std(unbiased=False)))
            else:
                rec["gauge_trace_spread"].append(float(
                    metrics.gauge_trace_spread(belief.phi, self.group.generators)))
                active_vertex = compute_transport_operators(
                    belief.phi.unsqueeze(0), self.group,
                    validity_max_norm=cfg.transport_chart_max_norm)["exp_phi"][0]
                if belief.reflection is not None:
                    active_vertex = active_vertex.clone()
                    active_vertex[..., 0, :] *= belief.reflection[..., None]
                active_invariant = metrics.group_gauge_invariant(
                    active_vertex, self.group).float()
            rec["gauge_invariant_spread"].append(
                float(active_invariant.std(unbiased=False)))
            _diag = belief.sigma.dim() == belief.mu.dim()
            rec["attn_entropy"].append(float(metrics.attention_entropy(beta)))
            bs = metrics.belief_spectrum(
                belief.sigma,
                diagonal=_diag,
                eps=cfg.eps,
                family=cfg.family,
            )
            rec["effective_rank"].append(float(bs["effective_rank"].mean()))
            rec["belief_cond_median"].append(float(bs["condition"].float().median()))
            # Coordinate-chart diagnostic only: under omega_direct phi is intentionally inactive.
            rec["phi_norm_mean"].append(float(torch.linalg.norm(belief.phi, dim=-1).mean()))

            if snapshot is None:
                mu_p = (1.0 - rho) * mu_p + rho * belief.mu          # handoff (mirrors vfe_stack)
                sigma_p = (1.0 - rho_s) * sigma_p + rho_s * belief.sigma
        return rec
