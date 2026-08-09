r"""PriorBank for VFE_3.0: learnable Gaussian vocab priors + the KL decode boundary.

Holds the routed per-vocabulary prior pi_v = N(mu_v, Sigma_v) with gauge frame phi_v as
PARAMETER TABLES (nn.Parameter -- priors, not neural maps; the no-NN rule bans
nn.Linear/MLP/activations, not learnable parameters). The built-in model-channel route
uses the s tables directly and omits redundant base mean/variance tables. encode(token_ids)
looks the active tables up into the initial belief (q = p); decode(mu_q, sigma_q) scores the
posterior against every active prior as logits = -KL(q || pi_v)/tau_eff (the divergence seam),
replacing a linear output projection.

Modularity:
    encode_mode registry -- ``per_token`` (table lookup, default); ``gauge_fixed`` a
        named stub (gauge orbit from a shared base belief).
    decode_mode registry -- ``diagonal`` (fused closed form, default); ``diagonal_chunked``
        (fused decode+CE, inference delegates to ``diagonal``); ``full`` (exact full-covariance
        Cholesky decode); ``full_chunked`` (full-cov KL via the diagonal-prior closed form);
        ``family`` / ``family_chunked`` (family/divergence-consistent decode: logits =
        -D_configured(q || pi_v)/tau_eff through the CONFIGURED family and divergence functional,
        both covariance ranks); ``expected_likelihood_chunked`` (log N(mu_q; mu_v, Sigma_q + Sigma_v)
        Gaussian-convolution scoring, diagonal only); plus the registered-but-config-excluded
        ``linear`` ablation kernel (reached via use_prior_bank=False).

Decode seam (PB-14): the family-consistent ``family``/``family_chunked`` kernels AND the
authoritative ``reference_decode`` score logits = -D_configured(q || pi_v)/tau_eff through the
CONFIGURED family (``self.family``) and divergence functional (``self.divergence_family`` at
``self.renyi_order``), so the readout matches the E-step geometry. The fast ``diagonal``/``full``
kernels remain the OPTIMIZED gaussian_* + renyi(alpha=1) implementations (they hardcode gaussian
alpha=1 KL and ignore divergence_family/renyi_order); config pairs those single-rank kernels only
with a canonical gaussian/renyi/alpha=1 seam, and REQUIRES a ``family_consistent`` decoder for any
non-Gaussian family or noncanonical divergence under ``use_prior_bank=True``. The registry seam is
honored at the COVARIANCE-STRUCTURE granularity (``DecodeRegistration.covariance_kinds``): a new
covariance structure or a new family-consistent readout is added by writing-and-registering a decode
kernel, never by editing a call site. The full kernels score a full q against the intentionally
DIAGONAL vocabulary-prior table (promoted with diag_embed only when the family is full).
``reference_decode`` is the slow per-V seam-call cross-check the fused canonical kernels are pinned
to EXACTLY (and under ``log_softmax``) on the canonical path.
"""

import warnings
from collections import OrderedDict
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Dict, FrozenSet, List, Mapping, Optional, Protocol, Tuple

import torch
import torch.utils.checkpoint as _checkpoint
from torch import nn

from vfe3.model.head_evidence import normalized_head_evidence_weights
from vfe3.contracts import CanonicalFrameContext
from vfe3.model.canonical_content import pullback_diagonal_query

from vfe3.belief import BeliefState
from vfe3.divergence import family_cov_kind, get_family, get_functional, kl, renyi
from vfe3.families.base import _logdet_chol
from vfe3.families.gaussian import FullGaussian, full_cov_kl_precision
from vfe3.families.covariance_tables import (
    covariance_from_packed,
    packed_from_covariance,
    packed_strict_lower_size,
)
from vfe3.geometry.lie_ops import CompactBlockElement
from vfe3.numerics import (
    _count_decode_logdet_fallback,
    bounded_variance_from_log,
    safe_cholesky,
)


# --- decode a_v working precision (audit 2026-08-06 F32) -------------------------------------
# Every decode kernel needs
#     a_v = sum_k [ (sq_k + (mc_q_k - mc_v_k)^2) / sigma_v_k ] + sum_k log sigma_v_k,
# evaluated as mc_q^2 - 2 mc_q mc_v + mc_v^2 so the v-dependent part is a single (2K)-inner-product
# GEMM over the vocabulary -- the reason the chunked kernels are cheap. F32 filed this as a
# CANCELLATION defect in that expanded square and proposed differencing before squaring.
#
# MEASURED, and the attribution is wrong (K=20, V=64, uniform sigma_v, vs a float64 reference):
#
#   sigma_v   expanded fp32   exact fp32   expanded fp64   exact fp64
#   4e+00        5.20e-06      4.49e-06        1.42e-14      0.00e+00
#   1e-02        2.50e-03      1.52e-03        5.46e-12      0.00e+00
#   1e-04        2.22e-01      1.26e-01        4.66e-10      0.00e+00
#
# Differencing before squaring buys a factor of ~1.8. Doing the SAME expanded algebra in float64
# buys nine decades. So the defect is not the factorization, it is accumulating terms of size
# 1/sigma_v (2e5 at sigma_v=1e-4, against fp32's ~7 significant digits) and then dividing the
# result by 2*tau = 0.016. The exact form was implemented, measured, and dropped: a 1.8x accuracy
# gain does not pay for giving up the GEMM and materializing a (B, N, Vc, K) transient per chunk.
#
# "fp64" therefore keeps the expanded algebra and raises the working precision, and the island must
# extend THROUGH the logit -- a_v itself is large, so rounding it back to float32 before subtracting
# per_pos and dividing by tau would throw the accuracy straight back away. Callers cast the finished
# O(1) logit, not a_v. The severity is latent either way: sigma_log_embed trains freely against only
# the eps=1e-6 floor, so this is harmless at init (2.4e-4 induced logit error) and about one order
# of magnitude of shrinkage from material. Default "fp32" is bit-identical to every run on disk.
_DECODE_AV_PRECISIONS = ("fp32", "fp64")
_DECODE_AV_PRECISION: str = "fp32"


def set_decode_av_precision(precision: str) -> str:
    r"""Set the process-wide decode ``a_v`` working precision; returns the previous value."""
    global _DECODE_AV_PRECISION
    if precision not in _DECODE_AV_PRECISIONS:
        raise ValueError(
            f"decode_av_precision must be one of {_DECODE_AV_PRECISIONS}, got {precision!r}")
    previous = _DECODE_AV_PRECISION
    _DECODE_AV_PRECISION = precision
    return previous


def decode_av_precision() -> str:
    r"""Return the active decode ``a_v`` working precision."""
    return _DECODE_AV_PRECISION


DECODE_CE_CHECKPOINT_AUTO_BYTES: int = 2 * 1024 ** 3   # 2 GiB; see decode_ce_checkpoint in config.py

# Transient per-chunk workspace ceiling for a FULL family's functional route (audit 2026-08-07).
# ``decode_ce_family_chunked`` promotes the diagonal vocabulary prior with ``diag_embed`` and hands
# the pair to the registered functional, whose workspace is (B, N, Vc, K, K) -- K^2 times the
# (B, N, Vc) a diagonal family pays. ``decode_chunk_size`` is ONE knob shared by every decode kernel
# and is sized for the inner=1 kernels, so the raw value reaches this route unscaled: at the live
# B=64, N=128, K=20, chunk=8192 that is a measured 100.00 GiB allocation per chunk and an outright
# ``CUDA out of memory``, against 6.9 GiB peak for the whole ``full_chunked`` decode at the identical
# shape. This ceiling bounds the TRANSIENT peak by narrowing the slice; it does not change the
# reduction, which is value-equal across widths (pinned by
# tests/test_family_chunked_decode_20260807.py::test_decode_chunk_size_bounds_the_functional_workspace).
DECODE_CE_FAMILY_WORKSPACE_BYTES: int = 1024 ** 3      # 1 GiB transient ceiling per vocab slice

# ``_full_gaussian_kl_terms`` materializes TWO (B, N, Vc, K, K) tensors, not one: the forward
# substitution ``Y = solve_triangular(L_p, sigma_q)`` and the back substitution
# ``Z = solve_triangular(L_p^T, Y)`` (families/gaussian.py:117-118), both live simultaneously and
# both retained into backward. Counting one was a 2x undercount of the only quantity the checkpoint
# gate and the workspace ceiling are allowed to reason about.
DECODE_CE_FAMILY_WORKSETS: int = 2


def _uses_canonical_full_family_decode(
    pb:       'PriorBank',
    mu_q:     torch.Tensor,
    sigma_q:  torch.Tensor,
) -> bool:
    r"""Whether a family-consistent full decode is exactly the analytic KL decoder.

    This deliberately resolves the live registries and compares identities rather than names:
    users may replace the ``gaussian_full`` family or ``renyi`` functional under their original
    public names, in which case the generic family route remains the authoritative behavior.
    """
    if get_family(pb.family) is not FullGaussian or get_functional(pb.divergence_family) is not renyi:
        return False
    if type(pb.renyi_order) not in (int, float) or pb.renyi_order != 1.0:
        return False
    if full_cov_kl_precision() != "fp32_escalate" or decode_av_precision() != "fp32":
        return False

    # Do not call _decode_sigma_log_table() here.  For the model-channel full path it materializes
    # marginal decode variances from the packed table, and the analytic delegate must materialize
    # that vocabulary table only once in its actual scoring body.
    if pb.untie_decode_bank:
        raw_decode_tables = (
            getattr(pb, "decode_mu_embed", None),
            getattr(pb, "decode_sigma_log_embed", None),
        )
    elif pb.prior_source == "model_channel":
        raw_decode_tables = (
            getattr(pb, "s_mu_embed", None),
            getattr(pb, "s_sigma_log_embed", None),
            getattr(pb, "s_sigma_lower_embed", None),
        )
    else:
        raw_decode_tables = (
            getattr(pb, "mu_embed", None),
            getattr(pb, "sigma_log_embed", None),
        )
    if any(table is None for table in raw_decode_tables):
        return False
    dtypes = (mu_q.dtype, sigma_q.dtype, *(table.dtype for table in raw_decode_tables),
              pb.decode_log_scale.dtype)
    return dtypes[0] in (torch.float32, torch.float64) and all(dtype == dtypes[0] for dtype in dtypes)


def _decode_ce_workspace_scalar_bytes(
    ref: torch.Tensor,
    workspace_bytes_per_scalar: Optional[int],
) -> int:
    r"""Resolve the optional scalar-byte override while keeping byte arithmetic integral."""
    scalar_bytes = ref.element_size() if workspace_bytes_per_scalar is None else workspace_bytes_per_scalar
    if type(scalar_bytes) is not int or scalar_bytes <= 0:
        raise ValueError("workspace_bytes_per_scalar must be a positive integer")
    return scalar_bytes


def _decode_ce_chunk_activation_bytes(
    ref:         torch.Tensor,            # (B, N, ...) tensor sharing the closure's (B, N) and dtype
    chunk_width: int,                     # Vc for this iteration
    inner:       int                = 1,   # extra trailing-dim multiplier (e.g. K or K*K) the
                                            # closure's largest workspace carries beyond (B, N, Vc)
    *,
    workspace_bytes_per_scalar: Optional[int] = None,
) -> int:
    r"""Bytes of the largest (B, N, chunk_width[, inner]) tensor a decode_ce_*_chunked kernel's
    ``_chunk_summaries`` closure actually materializes for THIS chunk, at ``ref``'s dtype -- the
    exact quantity ``decode_ce_checkpoint='auto'`` (config.py) gates the per-chunk checkpoint on.
    Computed from the real shapes/dtype in play (batch, positions, chunk_width, dtype.itemsize),
    not a guessed formula; ``inner`` lets callers whose chunk workspace carries an extra (K) or
    (K, K) axis (``decode_ce_expected_likelihood_chunked``, ``decode_ce_family_chunked``) report
    that workspace's true size rather than under-counting it as (B, N, Vc).
    """
    scalar_bytes = _decode_ce_workspace_scalar_bytes(ref, workspace_bytes_per_scalar)
    batch, positions = ref.shape[0], ref.shape[1]
    return batch * positions * chunk_width * inner * scalar_bytes


def _decode_ce_should_checkpoint(
    checkpoint_mode:  str,                # pb.decode_ce_checkpoint: "always" | "never" | "auto"
    grad_active:      bool,               # torch.is_grad_enabled() and the checkpointed input requires_grad
    activation_bytes: int,                # _decode_ce_chunk_activation_bytes(...) for this chunk
) -> bool:
    r"""Central dispatch for every ``decode_ce_*_chunked`` kernel's per-chunk gradient checkpoint.

    Grad-inactive calls (eval / ``no_grad``, or a checkpointed input that itself carries no grad)
    never checkpoint in any mode -- there is nothing for a checkpoint to save memory against, and
    checkpointing it anyway would only pay a pointless recompute. Under grad: "always" reproduces
    the pre-2026-08-07 unconditional behavior (checkpoint every chunk); "never" always keeps the
    chunk workspace live for backward; "auto" checkpoints only when ``activation_bytes`` -- the size
    of the tensor the checkpoint would actually be saving -- exceeds ``DECODE_CE_CHECKPOINT_AUTO_BYTES``.
    """
    if not grad_active:
        return False
    if checkpoint_mode == "always":
        return True
    if checkpoint_mode == "never":
        return False
    return activation_bytes > DECODE_CE_CHECKPOINT_AUTO_BYTES               # "auto"


def _decode_ce_family_effective_chunk(
    ref:       torch.Tensor,              # (B, N, ...) tensor sharing the closure's (B, N) and dtype
    requested: int,                       # pb.decode_chunk_size (or an explicit chunk_size override)
    inner:     int,                       # trailing-dim multiplier the functional workspace carries
    worksets:  int = DECODE_CE_FAMILY_WORKSETS,
    *,
    workspace_bytes_per_scalar: Optional[int] = None,
) -> int:                                 # vocab slice width that keeps the transient under the cap
    r"""Narrow a decode vocab slice so a FULL family's functional workspace stays bounded.

    ``decode_chunk_size`` is a single knob read by five decode kernels (prior_bank.py:1185, 1382,
    1496, 1580, 1673) and is sized for the ones whose per-chunk workspace is ``(B, N, Vc)``. A full
    family's is ``(B, N, Vc, K, K)`` -- ``K*K`` times larger, times ``worksets`` simultaneous copies
    -- so the same integer means two very different amounts of memory depending on the family, and
    the config validator has no family-aware rule for it (config.py:2515 checks only ``>= 1``).

    Returns ``requested`` unchanged when ``inner <= 1`` (every diagonal kernel: byte-identical, this
    helper is inert for them) and otherwise the largest width whose transient fits
    ``DECODE_CE_FAMILY_WORKSPACE_BYTES``, floored at 1 so the loop always makes progress. Narrowing
    a slice is a TILING change, not a numerical one: the vocabulary reduction is a logsumexp over
    independent slices, so value and gradient are identical at any width.
    """
    scalar_bytes = _decode_ce_workspace_scalar_bytes(ref, workspace_bytes_per_scalar)
    if inner <= 1:
        return requested
    per_entry = ref.shape[0] * ref.shape[1] * inner * worksets * scalar_bytes
    if per_entry <= 0:
        return requested
    return max(1, min(requested, DECODE_CE_FAMILY_WORKSPACE_BYTES // per_entry))


def _full_family_workspace_bytes_per_scalar(
    family_cls: type,
    ref: torch.Tensor,
    *public_operands: torch.Tensor,
) -> int:
    r"""Return a conservative call-time workspace scalar size for the built-in full Gaussian.

    The full Gaussian may compute its pair grid in fp64 under every supported full-covariance
    precision policy (unconditionally for ``fp64`` and on escalation for the fp32 policies), and
    any public fp64 operand selects that public dtype directly.  This is intentionally identity-
    scoped to the resolved built-in :class:`FullGaussian`: custom family/functionals expose no
    workspace contract, so their existing reference-dtype sizing remains untouched.
    """
    if family_cls is FullGaussian and (
            any(operand.dtype == torch.float64 for operand in public_operands)
            or full_cov_kl_precision() in ("fp64", "fp32_escalate", "fp32_escalate_cond")):
        return 8
    return ref.element_size()


def _decode_av_lhs(
    sq:          torch.Tensor,                   # (..., N, K) query variances
    mc_q:        torch.Tensor,                   # (..., N, K) centered query means
    coord_delta: Optional[torch.Tensor] = None,  # (K,) optional head-evidence coefficients w_h - 1
) -> torch.Tensor:                        # (..., N, 2K)
    r"""The v-independent left factor, hoisted out of the chunk loop."""
    quadratic = sq + mc_q ** 2
    linear = -2.0 * mc_q
    if coord_delta is None:
        return torch.cat([quadratic, linear], dim=-1)
    return torch.cat([
        quadratic * coord_delta,
        linear * coord_delta,
    ], dim=-1)


def _decode_av(
    sq:    torch.Tensor,                  # (..., N, K) query variances (diag(Sigma_q) for the full family)
    mc_q:  torch.Tensor,                  # (..., N, K) centered query means
    mc_v:  torch.Tensor,                  # (Vc, K) centered prior means
    inv_v: torch.Tensor,                  # (Vc, K) 1/sigma_v
    lsum:  torch.Tensor,                  # (Vc,) sum_k log sigma_v

    *,
    lhs:         Optional[torch.Tensor] = None,  # (..., N, 2K) precomputed _decode_av_lhs
    coord_delta: Optional[torch.Tensor] = None,  # (K,) optional head-evidence coefficients w_h - 1
) -> torch.Tensor:                        # (..., N, Vc), float64 under the fp64 policy
    r"""``a_v`` under the active working precision -- the ONE place this algebra is written.

    Shared by all four decode kernels (diagonal/full x chunked-CE/logits) so the policy cannot apply
    to some and not others, which would let the fused CE and the logits it is pinned against
    disagree. Returns float64 under ``"fp64"`` ON PURPOSE: the caller must keep the island open
    through ``-0.5 * (a_v - per_pos) / tau_eff`` and cast the finished logit, because a_v is the
    large quantity and rounding it here would discard the entire gain. See the module note.
    """
    if _DECODE_AV_PRECISION == "fp64" and sq.dtype is not torch.float64:
        return _decode_av(
            sq.double(), mc_q.double(), mc_v.double(), inv_v.double(), lsum.double(),
            lhs=None if lhs is None else lhs.double(),
            coord_delta=None if coord_delta is None else coord_delta.double())
    if lhs is None or lhs.dtype is not sq.dtype:
        lhs = _decode_av_lhs(sq, mc_q, coord_delta)
    rhs = torch.cat([inv_v, mc_v * inv_v], dim=-1)             # (Vc, 2K)
    if coord_delta is None:
        return lhs @ rhs.transpose(-1, -2) + (mc_v ** 2 * inv_v).sum(-1) + lsum
    return (
        lhs @ rhs.transpose(-1, -2)
        + (mc_v ** 2 * inv_v * coord_delta).sum(-1)
        + lsum
    )


def _decode_head_evidence_kl_delta(
    sq:             torch.Tensor,
    mc_q:           torch.Tensor,
    mc_v:           torch.Tensor,
    inv_v:          torch.Tensor,
    log_sigma_v:    torch.Tensor,
    coord_delta:    torch.Tensor,
    delta_per_pos:  torch.Tensor,
    evidence_lhs:   Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Canonical baseline-plus-delta head-evidence KL contribution for one vocab slice."""
    delta_a_v = _decode_av(
        sq,
        mc_q,
        mc_v,
        inv_v,
        (log_sigma_v * coord_delta).sum(-1),
        lhs=evidence_lhs,
        coord_delta=coord_delta,
    )
    return 0.5 * (delta_a_v - delta_per_pos)


def _decode_analytic_kl_logits(
    a_v:          torch.Tensor,
    per_pos:      torch.Tensor,
    tau_eff:      torch.Tensor,
    output_dtype: torch.dtype,
    *,
    evidence_delta: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Apply the shared canonical KL floor, optional block delta, and decode temperature."""
    kl_v = (0.5 * (a_v - per_pos)).clamp(min=0.0)
    if evidence_delta is not None:
        kl_v = kl_v + evidence_delta
    return (-kl_v / tau_eff).to(output_dtype)


# ---------------------------------------------------------------------------
# Registries: mode name -> callable. Variants swap by config; add a variant by
# writing-and-registering it, never by editing call sites.
#   encode: fn(pb, token_ids) -> BeliefState
#   decode: fn(pb, mu_q, sigma_q, tau_eff) -> logits (B, N, V)
# ---------------------------------------------------------------------------
_ENCODERS: 'Dict[str, EncodeCallable]' = {}


@dataclass(frozen=True)
class EncodeRegistration:
    """An encode callable and registration-owned prior-table routing capabilities."""

    callable:               'EncodeCallable'
    can_omit_base_mean:     bool = False
    can_omit_base_variance: bool = False


_ENCODER_REGISTRATIONS: Dict[str, EncodeRegistration] = {}


@dataclass(frozen=True)
class DecodeRegistration:
    """A decode callable and all routing capabilities attached to that callable.

    ``covariance_kinds`` is the resolved set of family covariance structures the decoder scores
    ("diagonal" and/or "full"); config validates ``family_cov_kind(cfg.family) in covariance_kinds``
    rather than treating ``supports_full`` as an exclusive rank bit, so a dual-rank decoder (e.g.
    ``family``) accepts BOTH a diagonal and a full family. ``supports_full`` is retained (public,
    read by legacy callers) and stays coherent with the set: it is ``"full" in covariance_kinds``.
    ``family_consistent`` flags a decoder that scores logits = -D_configured(q||p_v)/tau_eff through
    the CONFIGURED family AND divergence functional (as opposed to the fast kernels' hardcoded
    gaussian alpha=1 KL); config requires a family_consistent decoder for any non-Gaussian family or
    noncanonical divergence under ``use_prior_bank=True``.

    Direct construction ``DecodeRegistration(callable, supports_full, supports_chunked, fused_ce)``
    stays source-compatible: the two new fields default, and ``__post_init__`` derives the legacy
    singleton ``covariance_kinds`` from ``supports_full`` when it is not supplied.
    """

    callable:          'DecodeCallable'
    supports_full:     bool
    supports_chunked:  bool
    fused_ce:          'Optional[FusedCECallable]'
    family_consistent: bool                        = False
    covariance_kinds:  'Optional[FrozenSet[str]]'  = None
    can_omit_base_mean: bool                       = False
    can_omit_base_variance: bool                   = False

    def __post_init__(self) -> None:
        # Resolve the covariance-kind set. Omitted -> the legacy singleton derived from
        # supports_full (a frozen dataclass, so the resolved value is written via object.__setattr__).
        if self.covariance_kinds is None:
            object.__setattr__(
                self, "covariance_kinds",
                frozenset({"full"} if self.supports_full else {"diagonal"}),
            )
        else:
            object.__setattr__(self, "covariance_kinds", frozenset(self.covariance_kinds))


_DECODERS: Dict[str, DecodeRegistration] = {}

# Once-per-process guard for the decode_unigram_prior=True-with-unset-table warning
# (the decode then degenerates to the current uniform-prior behavior).
_WARNED_UNIGRAM_UNSET: bool = False


def register_encode(
    name: str,

    *,
    override:               bool           = False,
    can_omit_base_mean:     bool = False,
    can_omit_base_variance: bool = False,
) -> 'Callable[[EncodeCallable], EncodeCallable]':
    """Decorator registering an encode kernel under ``name``.

    Duplicate keys fail closed (audit 2026-07-01 round-3): a second registration under an
    existing name silently shadowed the first. Pass ``override=True`` to replace deliberately.
    """
    def _wrap(fn: 'EncodeCallable') -> 'EncodeCallable':
        if name in _ENCODERS and not override:
            raise KeyError(f"encode mode {name!r} already registered; pass override=True to replace")
        # Capabilities belong to this NAME registration, never to the callable object. An alias or
        # functools.wraps wrapper is therefore conservative unless this invocation declares otherwise.
        if type(can_omit_base_mean) is not bool or type(can_omit_base_variance) is not bool:
            raise TypeError("encoder base-table omission declarations must be bools")
        _ENCODERS[name] = fn
        _ENCODER_REGISTRATIONS[name] = EncodeRegistration(
            callable=fn,
            can_omit_base_mean=can_omit_base_mean,
            can_omit_base_variance=can_omit_base_variance,
        )
        return fn
    return _wrap


def get_encode(name: str) -> 'EncodeCallable':
    """Return the registered encode kernel for ``name`` (KeyError if absent)."""
    if name not in _ENCODERS:
        raise KeyError(
            f"no encode mode registered under {name!r}; available: {sorted(_ENCODERS)}"
        )
    return _ENCODERS[name]


def get_encode_registration(name: str) -> EncodeRegistration:
    """Return the registration-owned encode capabilities for ``name`` (KeyError if absent)."""
    if name not in _ENCODERS or name not in _ENCODER_REGISTRATIONS:
        raise KeyError(
            f"no encode mode registered under {name!r}; available: {sorted(_ENCODERS)}"
        )
    registration = _ENCODER_REGISTRATIONS[name]
    if registration.callable is not _ENCODERS[name]:
        raise RuntimeError(f"encode mode {name!r} registry record is inconsistent")
    return registration


def register_decode(
    name: str,

    *,
    supports_full:     Optional[bool]              = None,
    supports_chunked:  bool                        = False,
    override:          bool                        = False,
    family_consistent: bool                        = False,
    fused_ce:          'Optional[FusedCECallable]'      = None,
    covariance_kinds:  'Optional[FrozenSet[str]]'       = None,
    can_omit_base_mean:     bool                        = False,
    can_omit_base_variance: bool                        = False,
) -> 'Callable[[DecodeCallable], DecodeCallable]':
    """Decorator registering a decode kernel under ``name``.

    ``covariance_kinds`` is the resolved set of family covariance structures the decoder scores.
    OMITTED -> derive the legacy singleton from ``supports_full`` (``{"full"}`` when True, else
    ``{"diagonal"}``); SUPPLIED -> derive ``supports_full`` from membership (``"full" in kinds``) and
    reject an explicitly contradictory legacy ``supports_full``. Every existing
    ``register_decode(..., supports_full=True|False)`` call therefore keeps its old behavior.
    ``family_consistent`` marks a decoder that reads logits out through the CONFIGURED family and
    divergence functional. ``supports_chunked`` advertises a fused chunked-CE training path, whose
    callable is ``fused_ce``. The callable and all capabilities are replaced atomically, so an
    override cannot retain stale routing metadata from the prior registration.

    Duplicate keys fail closed (audit 2026-07-01 round-3): a second registration under an
    existing name silently shadowed the first. Pass ``override=True`` to replace deliberately.
    """
    if covariance_kinds is None:
        resolved_full  = bool(supports_full) if supports_full is not None else False
        resolved_kinds = frozenset({"full"} if resolved_full else {"diagonal"})
    else:
        resolved_kinds = frozenset(covariance_kinds)
        if not resolved_kinds or not resolved_kinds <= {"diagonal", "full"}:
            raise ValueError(
                f"decode mode {name!r} covariance_kinds must be a nonempty subset of "
                f"{{'diagonal', 'full'}}, got {sorted(resolved_kinds)}"
            )
        resolved_full = "full" in resolved_kinds
        if supports_full is not None and bool(supports_full) != resolved_full:
            raise ValueError(
                f"decode mode {name!r} has contradictory metadata: supports_full={supports_full} "
                f"but covariance_kinds={sorted(resolved_kinds)} implies supports_full={resolved_full}"
            )

    def _wrap(fn: 'DecodeCallable') -> 'DecodeCallable':
        if name in _DECODERS and not override:
            raise KeyError(f"decode mode {name!r} already registered; pass override=True to replace")
        if supports_chunked != (fused_ce is not None):
            raise ValueError(
                f"decode mode {name!r} must declare supports_chunked=True exactly when fused_ce "
                f"is provided"
            )
        # Capabilities belong only to this registration record. Callable identity, aliases, and
        # functools.wraps attributes cannot confer omission rights on a new registered name.
        if type(can_omit_base_mean) is not bool or type(can_omit_base_variance) is not bool:
            raise TypeError("decoder base-table omission declarations must be bools")
        _DECODERS[name] = DecodeRegistration(
            callable=fn,
            supports_full=resolved_full,
            supports_chunked=supports_chunked,
            fused_ce=fused_ce,
            family_consistent=family_consistent,
            covariance_kinds=resolved_kinds,
            can_omit_base_mean=can_omit_base_mean,
            can_omit_base_variance=can_omit_base_variance,
        )
        return fn
    return _wrap


def get_decode_registration(name: str) -> DecodeRegistration:
    """Return the complete registration record for ``name`` (KeyError if absent)."""
    if name not in _DECODERS:
        raise KeyError(
            f"no decode mode registered under {name!r}; available: {sorted(_DECODERS)}"
        )
    return _DECODERS[name]


def get_decode(name: str) -> 'DecodeCallable':
    """Return the registered decode kernel for ``name`` (KeyError if absent)."""
    return get_decode_registration(name).callable


_LEGACY_DORMANT_PRIOR_TABLES = {
    "prior_bank.mu_embed":        "prior_bank.s_mu_embed",
    "prior_bank.sigma_log_embed": "prior_bank.s_sigma_log_embed",
}


def normalize_legacy_model_state(
    saved_state:          object,
    expected_model_state: Mapping[str, torch.Tensor],

    *,
    context: str = "model state",
) -> object:
    r"""Return a nonmutating exact migration of legacy dormant token-prior tables.

    A current model-channel route registers both base tables as ``None`` and therefore omits them
    from ``state_dict``. Legacy builds serialized those two inert tensors. Only those exact keys may
    be discarded, and only when the live state exposes the corresponding routed s table. Shape,
    dtype, layout, and finiteness are validated against that live table before removal. Token-prior
    models retain the base keys in their expected state and therefore remain fully strict.
    """
    if not isinstance(saved_state, Mapping):
        return saved_state
    removable = [
        key for key in _LEGACY_DORMANT_PRIOR_TABLES
        if key in saved_state and key not in expected_model_state
    ]
    if not removable:
        return saved_state
    for key in removable:
        routed_key = _LEGACY_DORMANT_PRIOR_TABLES[key]
        expected = expected_model_state.get(routed_key)
        actual = saved_state[key]
        if expected is None:
            raise RuntimeError(
                f"{context} legacy dormant prior table {key!r} has no live routed-table contract"
            )
        if (not isinstance(actual, torch.Tensor)
                or actual.shape != expected.shape
                or actual.dtype != expected.dtype
                or actual.layout != expected.layout):
            raise RuntimeError(
                f"{context} legacy dormant prior table {key!r} has an incompatible "
                "shape/dtype/layout"
            )
        if ((actual.is_floating_point() or actual.is_complex())
                and not bool(torch.isfinite(actual).all().item())):
            raise RuntimeError(
                f"{context} legacy dormant prior table {key!r} contains nonfinite values"
            )
    normalized = OrderedDict(saved_state)
    metadata = getattr(saved_state, "_metadata", None)
    if metadata is not None:
        normalized._metadata = metadata  # type: ignore[attr-defined]
    for key in removable:
        del normalized[key]
    return normalized


class PriorBank(nn.Module):
    r"""Learnable Gaussian vocab priors; encode (lookup) and decode (-KL/tau_eff).

    The active routed mean/variance tables and ``phi_embed`` (V, n_gen) parameterize
    pi_v = N(mu_v, exp(sigma_log_v)) with gauge frame phi_v. Token-prior routes allocate
    ``mu_embed`` and ``sigma_log_embed``; built-in model-channel routes consume their s-table
    counterparts and register the redundant base names as ``None``. They are PRIORS
    (nn.Parameter), not a neural map: there is no nn.Linear/MLP/activation anywhere in this
    module. The learnable scalar ``decode_log_scale`` tunes the decode temperature.
    """

    output_proj_weight: Optional[nn.Parameter]   # (V, K) linear-decode weight; None unless use_prior_bank=False
    emission_proj_weight: Optional[nn.Parameter] # (V, K) emission readout; None unless emission_mode='separate'
    output_proj_bias:   Optional[nn.Parameter]   # (V,) linear-decode log-unigram bias; None unless use_prior_bank=False and decode_bias
    mu_embed:           Optional[nn.Parameter]   # (V, K) token-prior mean; None when no executable route consumes it
    sigma_log_embed:    Optional[nn.Parameter]   # (V, K) token-prior log variance; same routing contract

    def __init__(
        self,
        vocab_size:   int,
        K:            int,
        n_gen:        int,

        *,
        mu_init_std:         float = 0.02,
        sigma_init:          float = 1.0,
        phi_scale:           float = 0.01,
        decode_tau:          float = 1.0,
        eps:                 float = 1e-6,
        diagonal_covariance: bool  = True,
        family:              str   = "gaussian_diagonal",
        divergence_family:   str   = "renyi",
        renyi_order:         float = 1.0,
        use_prior_bank:      bool  = True,
        decode_bias:         bool  = False,
        encode_mode:         str   = "per_token",
        decode_mode:         str   = "diagonal",
        decode_chunk_size:   int   = 8192,
        decode_ce_checkpoint: str  = "auto",
        lambda_h:            float = 0.0,
        lambda_gamma:        float = 0.0,
        prior_source:        str   = "token",
        s_frame_mode:        str   = "tied",
        s_e_step:            bool  = False,
        learnable_r:         bool  = False,

        unigram_kappa:        float = 1.0,
        decode_unigram_prior: bool  = False,
        emission_mode:        str   = "off",     # "off" | "shared" | "separate"; 'separate' allocates its own (V,K) readout
        untie_decode_bank:    bool  = False,

        gauge_parameterization: str                 = "phi",
        irrep_dims:             Optional[List[int]] = None,
        use_priorbank_head_evidence_mixer: bool     = False,
        omega_reflection:       str                 = "off",
        phi_reflection:         str                 = "off",
        omega_compact_storage:  bool                = False,
        gauge_group_is_tied:    bool                = False,
        gauge_group_name:       Optional[str]       = None,
    ) -> None:
        super().__init__()
        if encode_mode == "canonical_content_gauge":
            incompatible = []
            if family != "gaussian_frame_diagonal":
                incompatible.append(f"family={family!r}")
            if gauge_parameterization != "phi":
                incompatible.append(f"gauge_parameterization={gauge_parameterization!r}")
            if prior_source != "token":
                incompatible.append(f"prior_source={prior_source!r}")
            if s_e_step:
                incompatible.append("s_e_step=True")
            if not use_prior_bank:
                incompatible.append("use_prior_bank=False")
            if decode_mode not in ("diagonal", "diagonal_chunked"):
                incompatible.append(f"decode_mode={decode_mode!r}")
            if omega_reflection != "off":
                incompatible.append(f"omega_reflection={omega_reflection!r}")
            if phi_reflection != "off":
                incompatible.append(f"phi_reflection={phi_reflection!r}")
            if incompatible:
                raise ValueError(
                    "encode_mode='canonical_content_gauge' requires "
                    "family='gaussian_frame_diagonal', gauge_parameterization='phi', "
                    "prior_source='token', s_e_step=False, use_prior_bank=True, "
                    "decode_mode='diagonal'/'diagonal_chunked', and both reflection modes off; "
                    "got " + ", ".join(incompatible)
                )
        if encode_mode == "canonical_content_projected":
            incompatible = []
            if family != "gaussian_diagonal":
                incompatible.append(f"family={family!r}")
            if gauge_parameterization != "phi":
                incompatible.append(f"gauge_parameterization={gauge_parameterization!r}")
            if prior_source != "token":
                incompatible.append(f"prior_source={prior_source!r}")
            if s_e_step:
                incompatible.append("s_e_step=True")
            if not use_prior_bank:
                incompatible.append("use_prior_bank=False")
            if decode_mode not in ("full", "full_chunked"):
                incompatible.append(f"decode_mode={decode_mode!r}")
            if omega_reflection != "off":
                incompatible.append(f"omega_reflection={omega_reflection!r}")
            if phi_reflection != "off":
                incompatible.append(f"phi_reflection={phi_reflection!r}")
            if incompatible:
                raise ValueError(
                    "encode_mode='canonical_content_projected' requires "
                    "family='gaussian_diagonal', gauge_parameterization='phi', "
                    "prior_source='token', s_e_step=False, use_prior_bank=True, "
                    "decode_mode='full'/'full_chunked', and both reflection modes off; "
                    "got " + ", ".join(incompatible)
                )
            if not has_builtin_projected_full_decoder(decode_mode):
                raise ValueError(
                    "encode_mode='canonical_content_projected' requires the built-in analytic "
                    "'full'/'full_chunked' decode registration and callable identities; "
                    "registry overrides are not eligible."
                )
        if gauge_parameterization == "omega_direct" and encode_mode == "per_token_additive":
            raise ValueError(
                "gauge_parameterization='omega_direct' is incompatible with "
                "encode_mode='per_token_additive': the additive encoder returns no stored omega "
                "frame. Use encode_mode='per_token', or gauge_parameterization='phi' for the "
                "additive control."
            )
        if type(omega_compact_storage) is not bool:
            raise ValueError(
                "omega_compact_storage must be a bool, got "
                f"{type(omega_compact_storage).__name__}: {omega_compact_storage!r}"
            )
        if omega_compact_storage:
            compact_groups = {"block_glk", "tied_block_glk"}
            if gauge_group_name not in compact_groups:
                raise ValueError(
                    "omega_compact_storage requires explicit gauge_group_name='block_glk' or "
                    f"'tied_block_glk'; got {gauge_group_name!r}")
            expected_tied = gauge_group_name == "tied_block_glk"
            if gauge_group_is_tied != expected_tied:
                raise ValueError(
                    "gauge_group_is_tied is inconsistent with gauge_group_name: "
                    f"group={gauge_group_name!r}, tied={gauge_group_is_tied!r}")
            if irrep_dims is None:
                raise ValueError("omega_compact_storage requires explicit irrep_dims")
            if len(irrep_dims) <= 1:
                raise ValueError(
                    "omega_compact_storage requires more than one irrep block; "
                    f"got irrep_dims={irrep_dims!r}")
            if any(type(d) is not int or d <= 0 for d in irrep_dims):
                raise ValueError(
                    "omega_compact_storage requires every irrep dimension to be a positive int; "
                    f"got irrep_dims={irrep_dims!r}")
            if len(set(irrep_dims)) != 1:
                raise ValueError(
                    "omega_compact_storage requires equal irrep dimensions; "
                    f"got irrep_dims={irrep_dims!r}")
            if sum(irrep_dims) != K:
                raise ValueError(
                    f"omega_compact_storage requires sum(irrep_dims)==K; "
                    f"got sum={sum(irrep_dims)}, K={K}")
        self.vocab_size = vocab_size
        self.K = K
        self.n_gen = n_gen
        self.decode_tau = decode_tau
        self.eps = eps
        self.diagonal_covariance = diagonal_covariance
        # family drives the model-channel (s/r) covariance rank: 'full' -> packed strict-lower
        # Cholesky tables (SPD covariance), else the diagonal log-variance tables. The vocabulary
        # prior and decode variance tables stay diagonal in EVERY family (PB-11).
        self.family = family
        # divergence_family / renyi_order drive the family-consistent decode kernels
        # (decode_mode='family'/'family_chunked'): logits = -D_configured(q||p_v)/tau_eff scored
        # through get_functional(divergence_family) at alpha=renyi_order. The fast gaussian kernels
        # (diagonal/full) ignore them (they hardcode gaussian alpha=1 KL); config only pairs those
        # with a canonical gaussian/renyi/alpha=1 seam. Defaults reproduce the old fixed-KL readout.
        self.divergence_family = divergence_family
        self.renyi_order = renyi_order
        self._s_cov_kind = family_cov_kind(family)
        self.use_prior_bank = use_prior_bank
        self.encode_mode = encode_mode
        self.decode_mode = decode_mode
        self.decode_chunk_size = decode_chunk_size
        self.decode_ce_checkpoint = decode_ce_checkpoint
        self.prior_source = prior_source
        self.s_frame_mode = s_frame_mode
        self.s_e_step = s_e_step
        self.unigram_kappa = unigram_kappa
        self.decode_unigram_prior = decode_unigram_prior
        self.gauge_parameterization = gauge_parameterization
        self.gauge_group_name = gauge_group_name
        self.irrep_dims = irrep_dims
        if use_priorbank_head_evidence_mixer:
            if irrep_dims is None:
                raise ValueError("head evidence requires explicit gauge-block dimensions")
            self._head_evidence_irrep_dims = tuple(int(dim) for dim in irrep_dims)
            if (len(self._head_evidence_irrep_dims) < 2
                    or any(dim <= 0 for dim in self._head_evidence_irrep_dims)):
                raise ValueError("head evidence requires at least two positive gauge blocks")
            self.head_evidence_logits = nn.Parameter(torch.zeros(len(self._head_evidence_irrep_dims)))
        # untie applies to the KL-to-bank decode only (the linear ablation is already untied by
        # construction), so the flag is resolved against use_prior_bank once, here.
        self.untie_decode_bank = untie_decode_bank and use_prior_bank

        sigma_log_init = float(torch.log(torch.tensor(sigma_init)))
        encoder = get_encode_registration(encode_mode)
        decoder = get_decode_registration(decode_mode if use_prior_bank else "linear")
        model_channel_route = prior_source == "model_channel"
        self.base_mean_consumed = not (
            model_channel_route
            and encoder.can_omit_base_mean
            and decoder.can_omit_base_mean
        )
        self.base_variance_consumed = not (
            model_channel_route
            and encoder.can_omit_base_variance
            and decoder.can_omit_base_variance
        )
        if self.base_mean_consumed:
            self.mu_embed = nn.Parameter(mu_init_std * torch.randn(vocab_size, K))
        else:
            # Preserve the established downstream phi/s/output initialization under the same seed.
            # The historical dormant Parameter consumed this one random draw; advancing the stream
            # without retaining the tensor removes capacity while leaving every live table unchanged.
            _ = torch.randn(vocab_size, K)
            self.register_parameter("mu_embed", None)
        if self.base_variance_consumed:
            self.sigma_log_embed = nn.Parameter(torch.full((vocab_size, K), sigma_log_init))
        else:
            self.register_parameter("sigma_log_embed", None)
        self.phi_embed        = nn.Parameter(phi_scale * torch.randn(vocab_size, n_gen))
        if s_frame_mode == "phi_tilde":
            self.s_phi_embed = nn.Parameter(self.phi_embed.detach().clone())
        self.decode_log_scale = nn.Parameter(torch.zeros(1))

        # Arm-2 control (encode_mode='per_token_additive'): a NON-structural use of the SAME learned
        # (V, n_gen) phi table. A FROZEN random readout R (K, n_gen) maps each token's n_gen-dim code
        # to an additive K-dim mean shift, and encode returns phi=0 so Omega = exp(phi.G) = I (no gl(g)
        # transport). Isolates raw phi-table CAPACITY (V*n_gen learned params, matched to the gauge
        # cell) from the gl(g) generator STRUCTURE. R is a buffer (not a Parameter, so learned-param
        # count is unchanged), seeded for reproducibility, scaled 1/sqrt(n_gen) so the per-dim shift
        # std matches phi_scale at init. Deliberately breaks gauge equivariance -- that IS the control.
        if encode_mode == "per_token_additive":
            _r_gen = torch.Generator().manual_seed(0)
            self.register_buffer(
                "additive_R",
                torch.randn(K, n_gen, generator=_r_gen) / (float(n_gen) ** 0.5),
            )

        # use_prior_bank=False (linear-decode ablation): decode is a plain linear projection
        # logits = mu_q @ W^T through a learned (V, K) weight, the single authorized neural
        # exception (a lone linear output readout; see CLAUDE.md). Realized as a raw nn.Parameter
        # matmul -- NOT an nn.Linear/MLP -- so no neural-layer class enters the module. Created
        # only on the ablation path so the pure path (use_prior_bank=True) carries no extra weight.
        # Xavier-uniform init (PyTorch's nn.Linear default), no bias (a constant shift in
        # V that softmax/cross-entropy absorbs). Encode stays the prior-bank lookup either way.
        if use_prior_bank:
            self.output_proj_weight = None
            self.output_proj_bias   = None
        else:
            self.output_proj_weight = nn.Parameter(torch.empty(vocab_size, K))
            nn.init.xavier_uniform_(self.output_proj_weight)
            # Optional per-vocab bias (decode_bias): a *per-class* bias is NOT a softmax-invariant
            # constant shift -- it is a learned log-unigram prior, and a 50k Zipfian vocab is the
            # opposite of balanced, so the bias-free map can represent token base rates only by
            # spending rank-K mean capacity. Zero-init -> logits bit-identical to decode_bias=False
            # at construction (drawn AFTER the weight, so the weight's RNG is unchanged); the CE
            # gradient drives it toward log p(token). Routed to a weight-decay-free optimizer group
            # in build_optimizer (decaying a unigram prior toward zero biases it to flat).
            self.output_proj_bias = (
                nn.Parameter(torch.zeros(vocab_size)) if decode_bias else None
            )

        # Categorical emission readout (emission_mode='separate'): the emission factor in the
        # belief's Markov blanket gets its OWN (V, K) table instead of reusing the decode table.
        # V3 decodes position t against x_{t+1} while the emission pulls toward x_t, so under
        # 'shared' one linear map carries both roles; 'separate' removes that competition at the
        # cost of decoupling the factor from the decoder that scores the prediction. Same
        # Xavier-uniform init as the decode weight, drawn LAST so neither existing table's RNG
        # stream moves (an emission_mode='off' build is byte-identical to before). None on the
        # pure path and under 'shared', which reads output_proj_weight.
        self.emission_proj_weight = (
            nn.Parameter(torch.empty(vocab_size, K)) if emission_mode == "separate" else None
        )
        if self.emission_proj_weight is not None:
            nn.init.xavier_uniform_(self.emission_proj_weight)

        # MODEL CHANNEL (manuscript eq:pointwise_free_energy), default-OFF. The model-channel belief
        # tables s_mu_embed/s_sigma_log_embed (V, K) -- a per-token DIAGONAL Gaussian s_i looked up
        # like the belief tables -- back BOTH the hyper-prior term lambda_h*KL(s||r) and the gamma
        # model-coupling block lambda_gamma*F_red^s, so they are created whenever EITHER channel is
        # active (lambda_h>0 OR lambda_gamma>0). The global hyper-prior r_mu/r_sigma_log (K,) -- a
        # single diagonal Gaussian the s_i are regularized toward (the manuscript centroid) -- is
        # consumed ONLY by the hyper-prior term, so it stays gated on lambda_h>0. These are PRIORS
        # (nn.Parameter), not a neural map. They are created LAST and only on the active-channel path:
        # the default (both 0) path draws zero new RNG, so the belief tables above are byte-unchanged
        # and the pure path is param-free. s drawn BEFORE r preserves the existing lambda_h>0 RNG order
        # (byte-identical to the hyper-prior-only build). s init mirrors the belief tables (small mu,
        # sigma matching sigma_init); r init: mu=0, sigma matching sigma_init -- so s != r at init
        # (KL(s||r) > 0, the channel has a gradient). When prior_source='model_channel', these same
        # tables supply the belief prior p, including their packed full covariance; s_e_step may
        # additionally refine that prior before the belief-channel E-step.
        if lambda_h > 0.0 or lambda_gamma > 0.0 or prior_source == "model_channel" or s_e_step:
            self.s_mu_embed        = nn.Parameter(mu_init_std * torch.randn(vocab_size, K))
            self.s_sigma_log_embed = nn.Parameter(torch.full((vocab_size, K), sigma_log_init))
            if self._s_cov_kind == "full":
                # gaussian_full model channel (PB-11): the packed strict-lower Cholesky (V, K*(K-1)//2)
                # completing s_sigma_log_embed's diagonal into a full SPD covariance L L^T. ZERO-init
                # (torch.zeros, no RNG) so the initial model-channel covariances are exactly diagonal
                # AND the RNG order of every subsequent table is byte-unchanged from the pre-PB-11 build.
                # Diagonal/Laplace channels create no packed key -> pure diagonal state_dict is identical.
                self.s_sigma_lower_embed = nn.Parameter(
                    torch.zeros(vocab_size, packed_strict_lower_size(K)))
        if lambda_h > 0.0 or s_e_step:
            # Hyper-prior centroid r (r_mu, r_sigma_log): the centroid the model beliefs s_i are
            # regularized toward via lambda_h*KL(s_i||r). DEFAULT FROZEN (learnable_r=False,
            # requires_grad=False): the fixed centroid the manuscript determines "from a higher, slower
            # meta-level" (GL(K)_supplementary.tex:1081); with no meta-level built, a FIXED r is the
            # manuscript-consistent stand-in, and freezing prevents the KL(s||r)->0 collapse that freely
            # training r alongside an unanchored s would cause. learnable_r=True un-freezes r as an
            # empirical-Bayes population centroid (grouped in build_optimizer like the s tables);
            # meaningful only when s carries an independent data force (prior_source='model_channel'),
            # which VFE3Config.__post_init__ warns about.
            self.r_mu              = nn.Parameter(torch.zeros(K), requires_grad=learnable_r)
            self.r_sigma_log       = nn.Parameter(torch.full((K,), sigma_log_init), requires_grad=learnable_r)
            if self._s_cov_kind == "full":
                # The packed strict-lower Cholesky of the centroid r (gaussian_full, PB-11): zero-init
                # (r starts diagonal), grouped/frozen exactly like r_sigma_log via learnable_r.
                self.r_sigma_lower = nn.Parameter(
                    torch.zeros(packed_strict_lower_size(K)), requires_grad=learnable_r)
            # DESIGN NOTE (audit 2026-06-15): the token-dependent top-down hyper-prior
            # r_i = Omega_tilde[s_I^{(s+1)}] (PIFB eq:cross_scale_shadow / eq:topdown_priors) is the
            # model-fiber transport of a GENUINELY EMERGED scale-(s+1) meta-agent, and is OUT OF SCOPE for
            # this single-scale transformer -- NOT a deferred gap. The manuscript treats single-scale r_i as
            # a PRIMITIVE boundary condition (PIFB lines 554, 636) and assigns the full transport + Ouroboros
            # tower to MAgent_Model/gauge_agent/ (PIFB line 2334). The frozen global r above IS the sanctioned
            # s_max boundary -- the named "held at its initial value rather than recomputed" special case of
            # the self-referential closure (PIFB line 2332). learnable_r is the same-scale empirical-Bayes
            # stand-in (a different axis: frozen-vs-learned, still token-uniform).

        # Unigram log-prior decode table (decode_unigram_prior=True): a non-trainable (V,) buffer
        # log pi_v holding the smoothed corpus unigram log-frequencies, added to EVERY decode
        # path's logits as kappa * log pi_v (the Bayes class prior; a DATA statistic set by
        # set_unigram_log_prior, not a learned parameter). Created only on the toggled path
        # (matching additive_R / the s tables) so the default state_dict is byte-identical.
        # Init zeros = the current implicit uniform prior; decode warns once per process while
        # the table is still unset.
        if decode_unigram_prior:
            self.register_buffer("unigram_log_prior", torch.zeros(vocab_size))
            self._unigram_set = False                                   # flips on set_unigram_log_prior
        # Untied decode bank (untie_decode_bank=True, use_prior_bank=True only): decode reads its
        # OWN (V, K) tables decode_mu_embed / decode_sigma_log_embed, cloned from the tables decode
        # would otherwise read (_prior_mu_table -- the s tables under prior_source='model_channel',
        # else the encode tables) so step 0 is byte-identical, then trained separately. Encode and
        # the alpha-KL self-coupling target keep the original tables. Cloning draws no RNG, so the
        # default path's table init is byte-unchanged.
        if self.untie_decode_bank:
            self.decode_mu_embed        = nn.Parameter(self._prior_mu_table().clone().detach())
            self.decode_sigma_log_embed = nn.Parameter(self._prior_sigma_log_table().clone().detach())

        # omega_direct: a per-token GL(K) group element table (identity init -> step-0 == trivial gauge).
        # Created ONLY on the omega_direct path so the default state_dict is byte-identical. Block-
        # diagonal by construction for block_glk (identity is diagonal; the group retraction keeps it so).
        #
        # omega_compact_storage (opt-in, default OFF): for an EQUAL-block group (untied block_glk /
        # tied tied_block_glk; irrep_dims = [d]*H, H>1) the full (V,K,K) table wastes ~H x (off-blocks
        # frozen zero). Store the H distinct blocks (V,H,d,d) untied, or the ONE shared block (V,d,d)
        # tied -- both matching phi_embed's V*n_gen param count exactly (V*H*d^2 / V*d^2 = V*n_gen).
        # encode carries these blocks in CompactBlockElement; inverse and transport stay blockwise,
        # while explicit compatibility callers may request a dense element. Compaction changes the
        # table SHAPE (would break a Phase-1 (V,K,K) checkpoint), so
        # the opt-in flag is the state-dict safety: default OFF keeps the shipped (V,K,K) path
        # byte-identical. Single-block groups (glk/so_k/sp) and the irrep towers (so_n/sp_n) keep
        # (V,K,K) this phase (nothing to compact / element-vs-coordinate tension deferred).
        self._omega_compact = False
        self._omega_tied    = bool(gauge_group_is_tied)
        self.reflection_scope = "full_element"
        if (gauge_group_name == "block_glk"
                and irrep_dims is not None
                and len(irrep_dims) > 1
                and (omega_reflection != "off" or phi_reflection != "off")):
            # Multi-block block_glk is a product GL(d)^H with 2^H orientation sectors. The existing
            # reflection proposal is diag(-1,1,...) at K scale, so it probes block 0 only; keep the
            # proposal unchanged and label its intentionally limited scope instead of implying all
            # sectors. One-head and cross-coupled block_glk report irrep_dims=[K] and therefore use
            # the complete represented element rather than this product-group label.
            self.reflection_scope = "block_0_probe"
            warnings.warn(
                "block-GL reflection is a block-0 probe: the existing proposal flips only the first "
                "GL(d) block and does not explore all 2^H product-group orientation sectors.",
                UserWarning,
                stacklevel=2,
            )
        if gauge_parameterization == "omega_direct":
            dims = irrep_dims
            compact = omega_compact_storage
            self._omega_compact = compact
            if compact:
                H, d = len(dims), dims[0]
                eye_d = torch.eye(d)
                if gauge_group_is_tied:                       # (V,d,d): one block shared across H heads
                    self.omega_embed = nn.Parameter(eye_d.expand(vocab_size, d, d).clone())
                else:                                         # (V,H,d,d): H independent blocks
                    self.omega_embed = nn.Parameter(eye_d.expand(vocab_size, H, d, d).clone())
                    if omega_reflection == "init_seed":
                        # reflection_element(K) = diag(-1,1,...,1) is block 0 = reflection_element(d),
                        # blocks 1..H-1 = I_d, so seeding block 0 of every OTHER token assembles to the
                        # identical det<0 element the full (V,K,K) path seeds. (tied rejects init_seed at
                        # config, so the (V,d,d) branch needs no seed.)
                        from vfe3.geometry.generators import reflection_element
                        Rd = reflection_element(d)
                        with torch.no_grad():
                            self.omega_embed[1::2, 0] = Rd
            else:
                eye_K = torch.eye(K)
                self.omega_embed = nn.Parameter(eye_K.expand(vocab_size, K, K).clone())
                if omega_reflection == "init_seed":
                    from vfe3.geometry.generators import reflection_element
                    R = reflection_element(K)
                    with torch.no_grad():                    # seed every OTHER token into the det<0 sheet
                        self.omega_embed[1::2] = R

        # phi_reflection: a per-token discrete reflection sign R_i (det<0 iff sign==-1), prepended to
        # exp(phi_i) as g_i = R_i exp(phi_i) (see docs/superpowers/specs/2026-07-08-phi-reflection-
        # design.md). A register_buffer, NOT nn.Parameter: discrete state flipped by the Metropolis
        # move, not gradient. Created ONLY on the phi path when phi_reflection != 'off' so the default
        # state_dict is byte-identical. Default all +1 (identity, det>0); 'init_seed' seeds every OTHER
        # token to -1, mirroring omega_embed's [1::2] init_seed above.
        if gauge_parameterization == "phi" and phi_reflection != "off":
            self.register_buffer("reflection_sign", torch.ones(vocab_size))
            if phi_reflection == "init_seed":
                with torch.no_grad():
                    self.reflection_sign[1::2] = -1.0

    def head_evidence_weights(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return normalized gauge-block and coordinate evidence weights when enabled."""
        if not hasattr(self, "head_evidence_logits"):
            raise RuntimeError("head evidence weights require use_priorbank_head_evidence_mixer=True")
        weights = normalized_head_evidence_weights(
            self.head_evidence_logits,
            self._head_evidence_irrep_dims,
            dtype=dtype,
            device=device,
        )
        return weights.head, weights.coordinate

    def _head_evidence_deltas(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return exact ``w_h - 1`` coefficients for baseline-plus-delta scoring."""
        weights = normalized_head_evidence_weights(
            self.head_evidence_logits,
            self._head_evidence_irrep_dims,
            dtype=dtype,
            device=device,
        ).minus_identity()
        return weights.head, weights.coordinate

    def encode(
        self,
        token_ids: torch.Tensor,         # (B, N) integer token ids
    ) -> BeliefState:
        r"""Look up the per-token Gaussian prior as the initial belief (q = p)."""
        belief = get_encode(self.encode_mode)(self, token_ids)
        if hasattr(self, "reflection_sign"):
            belief = belief._replace(reflection=self.reflection_sign[token_ids])
        return belief

    def _omega_lookup(
        self,
        token_ids: torch.Tensor,         # (B, N) integer token ids
    ) -> 'torch.Tensor | CompactBlockElement':
        r"""Look up the per-token gauge frame U_i without changing its storage representation.

        Non-compact (default): a plain (V, K, K) table lookup. Compact
        (``_omega_compact``): return ``CompactBlockElement`` around the live looked-up
        (B, N, H, d, d) / tied (B, N, d, d) blocks. Inverse and transport contractions consume
        those blocks directly. Dense K x K reconstruction is available only through the container's
        explicit compatibility method ``to_dense()``.
        """
        g = self.omega_embed[token_ids]                                      # (B,N,K,K) or (B,N,H,d,d)/(B,N,d,d)
        if not self._omega_compact:
            return g
        return CompactBlockElement(g, self.K, tied=self._omega_tied)

    def encode_s(
        self,
        token_ids: torch.Tensor,         # (B, N) integer token ids
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Look up the per-token model-channel belief s_i = N(s_mu, s_sigma).

        Returns (s_mu, s_sigma) with s_mu (B, N, K); the covariance rank FOLLOWS the family
        (``family_cov_kind``): a diagonal/Laplace family yields the positive variances
        exp(s_sigma_log).clamp(min=eps) as (B, N, K), while ``gaussian_full`` assembles the packed
        strict-lower Cholesky into the full SPD covariance L L^T as (B, N, K, K). Available on the
        active-model-channel path (lambda_h>0, lambda_gamma>0, prior_source='model_channel', or
        s_e_step, where the s tables are created); consumed as ``get_family(cfg.family)(s_mu,
        s_sigma)`` by the hyper-prior term lambda_h*KL(s_i||r). The s->q coupling is a separate
        path: ``prior_source='model_channel'`` routes the belief prior to these same s tables,
        including the packed full covariance when the configured family is ``gaussian_full``.
        """
        s_mu = self.s_mu_embed[token_ids]                                       # (B, N, K)
        if self._s_cov_kind == "full":
            s_sigma = covariance_from_packed(
                self.s_sigma_log_embed[token_ids], self.s_sigma_lower_embed[token_ids], eps=self.eps,
            )                                                                     # (B, N, K, K)
        else:
            s_sigma = bounded_variance_from_log(
                self.s_sigma_log_embed[token_ids], eps=self.eps,
            )                                                                     # (B, N, K)
        return s_mu, s_sigma

    def r_parameters(self) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""The global hyper-prior centroid r = N(r_mu, r_sigma) with covariance rank FOLLOWING the
        family (``family_cov_kind``).

        Returns (r_mu, r_sigma) with r_mu (K,); a diagonal/Laplace family yields the positive
        variances exp(r_sigma_log).clamp(min=eps) as (K,), while ``gaussian_full`` assembles the
        packed strict-lower Cholesky into the full SPD covariance L L^T as (K, K). Available on the
        centroid path (lambda_h>0 or s_e_step, where the r tables are created); consumed as
        ``get_family(cfg.family)(r_mu, r_sigma)`` by the hyper-prior term (replacing the direct
        log-variance reads so a full family carries its off-diagonal centroid covariance).
        """
        r_mu = self.r_mu                                                        # (K,)
        if self._s_cov_kind == "full":
            r_sigma = covariance_from_packed(
                self.r_sigma_log, self.r_sigma_lower, eps=self.eps,
            )                                                                     # (K, K)
        else:
            r_sigma = bounded_variance_from_log(self.r_sigma_log, eps=self.eps)  # (K,)
        return r_mu, r_sigma

    def s_phi(
        self,
        token_ids: torch.Tensor,         # (B, N) integer token ids
    ) -> torch.Tensor:                   # (B, N, n_gen) model-channel frame coordinates
        r"""Look up the independently stored model-channel frame coordinates."""
        return self.s_phi_embed[token_ids]

    @torch.no_grad()
    def barycenter_r_(self) -> None:
        r"""Closed-form forward-KL barycenter M-step for the hyper-prior centroid r (IN PLACE).

        Sets r to the moment-matched centroid (m-projection) of the model-channel s tables:
        ``r_mu = mean_v s_mu_v`` and ``r_sigma = mean_v[s_sigma_v + (s_mu_v - r_mu)^2]`` (within-table
        variance plus the spread of the means) -- the unique minimizer of ``sum_v KL(s_v || r)`` for
        diagonal Gaussians (Amari-Nagaoka m-projection = moment matching; the diagonal unit-weight
        specialization of the manuscript meta-agent barycenter). Computed over the FULL vocab s tables
        (the population centroid, batch-independent), under no_grad: in r_update_mode='barycenter' r is
        NOT an optimizer leaf (requires_grad=False), so it carries no gradient and is set here once per
        M-step (driven from train_step).

        POPULATION (audit 2026-06-13): this is the exact argmin of the UNIFORM-over-vocab objective
        ``sum_v KL(s_v||r)`` -- one equal-weight row per vocab type. The scored hyper-prior term
        (``_hyper_prior_term``) reduces with mean() over (B,N) token OCCURRENCES, i.e. the
        frequency-weighted ``sum_v f_v KL(s_v||r)``; the uniform centroid equals that argmin only for a
        uniform token distribution, so for a Zipfian vocab the two differ. Treat this as the
        empirical-Bayes prior-over-TYPES centroid, NOT the argmin of the frequency-weighted scored loss.
        It is also the UNCLAMPED moment-match, whereas the scored KL runs through kl_max (so the two
        targets diverge for far-drifted rows). Under s_e_step=True r additionally couples to the CE
        through _refine_s, so it is only a consistent population target there -- prefer
        r_update_mode='gradient' for the scored s_e_step=False exactness and the s_e_step coupled regime.

        DIVERGENCE (audit 2026-06-14): this closed form is the ALPHA=1 (KL) m-projection and reads NO
        cfg, so it is the exact M-step only for the canonical KL objective (renyi_order=1,
        divergence_family='renyi', lambda_h_mode='constant'). The scored gradient path descends
        D_alpha(s||r) at cfg.renyi_order / cfg.divergence_family with the lambda_h_mode envelope, so
        under any non-canonical setting the 'barycenter' and 'gradient' r-updates do NOT share a fixed
        point (VFE3Config.__post_init__ warns). It also drops the model-fiber transport Omega_tilde and
        the per-type weights of the manuscript meta-agent barycenter, so it is a same-scale,
        UNTRANSPORTED, uniform-weight centroid -- not the cross-scale shadow r_i=Omega_tilde[s^(s+1)].

        FAMILY (PB-11): for ``gaussian_full`` the moment match runs over FULL covariances --
        ``r_Sigma = mean_v[Sigma_s_v + (s_mu_v - r_mu)(s_mu_v - r_mu)^T]`` (within-covariance plus
        the outer product of the mean spread), the full-Gaussian m-projection -- and is written back
        through the packed Cholesky (r_sigma_log + r_sigma_lower). The diagonal branch is unchanged.
        """
        if self._s_cov_kind == "full":
            s_mu = self.s_mu_embed                                               # (V, K)
            s_sigma = covariance_from_packed(
                self.s_sigma_log_embed, self.s_sigma_lower_embed, eps=self.eps,
            )                                                                     # (V, K, K)
            r_mu = s_mu.mean(dim=0)                                              # (K,)
            centered = s_mu - r_mu                                               # (V, K)
            outer = centered.unsqueeze(-1) * centered.unsqueeze(-2)              # (V, K, K)
            r_sigma = (s_sigma + outer).mean(dim=0)                             # (K, K) within + between
            r_log_diag, r_packed = packed_from_covariance(r_sigma, eps=self.eps)
            self.r_mu.copy_(r_mu)
            self.r_sigma_log.copy_(r_log_diag)
            self.r_sigma_lower.copy_(r_packed)
            return
        s_mu = self.s_mu_embed                                                   # (V, K)
        s_sigma = bounded_variance_from_log(self.s_sigma_log_embed, eps=self.eps)  # (V, K)
        r_mu = s_mu.mean(dim=0)                                                  # (K,)
        r_var = (s_sigma + (s_mu - r_mu) ** 2).mean(dim=0)                       # (K,) within + between
        self.r_mu.copy_(r_mu)
        self.r_sigma_log.copy_(torch.log(r_var.clamp(min=self.eps)))

    def _prior_mu_table(self) -> torch.Tensor:
        r"""The (V, K) mean prior table feeding p_i: the model-channel s tables when
        prior_source=='model_channel' (s->q REPLACE: p_i = s_i), else the belief table mu_embed
        (default). Routed through ONE accessor so encode (q_i(0)=p_i), the E-step self-coupling
        target alpha*KL(q_i||p_i), and the decode per-vocab readout -KL(q||p_v) all consume the SAME
        prior, keeping p_i = s_i consistent. On the default 'token' path this returns self.mu_embed
        (the identical tensor), so the pre-toggle path is byte-identical.
        """
        return self.s_mu_embed if self.prior_source == "model_channel" else self.mu_embed

    def _prior_sigma_log_table(self) -> torch.Tensor:
        r"""The (V, K) log-variance prior table feeding p_i; the model-channel sibling of
        _prior_mu_table (see there). 'token' -> self.sigma_log_embed (byte-identical)."""
        return self.s_sigma_log_embed if self.prior_source == "model_channel" else self.sigma_log_embed

    def _decode_mu_table(self) -> torch.Tensor:
        r"""The (V, K) mean table the DECODE boundary scores against: the untied decode table
        decode_mu_embed when untie_decode_bank created it, else the shared prior table
        (_prior_mu_table). Encode and the E-step self-coupling target always read the prior
        table, so the untie toggle splits ONLY the decode readout. On the default (tied) path
        this returns the identical tensor _prior_mu_table does -- byte-identical.
        """
        return self.decode_mu_embed if self.untie_decode_bank else self._prior_mu_table()

    @torch.no_grad()
    def decode_sigma_v_min(self) -> float:
        r"""The smallest decode prior variance in the table -- the margin behind ``a_v``'s accuracy.

        ``a_v`` accumulates terms of size ``1/sigma_v`` and is then divided by ``2*tau``, so the
        working precision has to carry roughly ``log10(1/(sigma_v * tau))`` digits before the logits
        start moving (audit 2026-08-06 F32). This is the one number that says how much headroom is
        left: at the ``sigma_init``-scale table the induced logit error is ~2e-4, at 1e-3 it is ~1.5
        logits, and at 1e-4 ~12. ``sigma_log_embed`` trains freely against only the ``eps`` floor, so
        the margin is not structurally bounded -- it has to be watched. Switch
        ``decode_av_precision="fp64"`` if this falls far enough to matter.
        """
        return float(bounded_variance_from_log(
            self._decode_sigma_log_table(), eps=self.eps).min())

    def _decode_sigma_log_table(self) -> torch.Tensor:
        r"""The (V, K) log-variance decode table; the sigma sibling of _decode_mu_table (see there).

        When the shared prior table is a PACKED Cholesky (``gaussian_full`` with
        ``prior_source='model_channel'``), its stored values are log squared Cholesky pivots, not log
        marginal variances. The decode needs a log-variance, so it must read
        ``log(diag(L L^T))``; reading the raw table understated every marginal variance by the
        row's off-diagonal energy, so the encode prior (``L L^T``, via
        :func:`_encode_prior_sigma`) and the decode prior (``diag(pivot^2)``) diverged as soon as the
        packed table trained away from its zero init (audit 2026-07-25 F8). Untied decode tables are
        genuine log-variances and are returned unchanged.
        """
        if self.untie_decode_bank:
            return self.decode_sigma_log_embed
        packed = getattr(self, "s_sigma_lower_embed", None)
        if packed is not None and self.prior_source == "model_channel":
            from vfe3.families.covariance_tables import marginal_log_variance_from_packed
            return marginal_log_variance_from_packed(
                self._prior_sigma_log_table(), packed, eps=self.eps)
        return self._prior_sigma_log_table()

    @torch.no_grad()
    def set_unigram_log_prior(
        self,
        counts: torch.Tensor,            # (V,) corpus unigram COUNTS (integer or float, >= 0)
    ) -> None:
        r"""Fill the unigram decode table with add-one-smoothed log-frequencies (IN PLACE).

            log pi_v = log((counts_v + 1) / (sum_v counts_v + V)),
        the Laplace (add-one) smoothed unigram log-prior: every token gets one pseudo-count, so
        zero-count tokens carry a finite log pi_v = -log(total + V) instead of -inf, and
        sum_v pi_v = 1 exactly. Requires construction with decode_unigram_prior=True (the buffer
        exists only on the toggled path).
        """
        if not self.decode_unigram_prior:
            raise RuntimeError(
                "set_unigram_log_prior requires decode_unigram_prior=True at construction "
                "(the unigram_log_prior buffer exists only on the toggled path)."
            )
        if counts.shape != (self.vocab_size,):
            raise ValueError(
                f"counts must have shape ({self.vocab_size},), got {tuple(counts.shape)}"
            )
        counts_f = counts.to(dtype=self.unigram_log_prior.dtype,
                             device=self.unigram_log_prior.device)          # (V,)
        self.unigram_log_prior.copy_(
            torch.log((counts_f + 1.0) / (counts_f.sum() + float(self.vocab_size)))
        )
        self._unigram_set = True

    def _unigram_bias(self) -> torch.Tensor:
        r"""The (V,) additive decode bias kappa * log pi_v (decode_unigram_prior=True only).

        Warns ONCE PER PROCESS while the table is still all-zero (never set): the decode then
        degenerates to the pre-toggle uniform prior (kappa * 0 = 0, a value no-op). A table
        restored nonzero through load_state_dict counts as set.
        """
        global _WARNED_UNIGRAM_UNSET
        if not self._unigram_set:
            if bool((self.unigram_log_prior != 0.0).any()):
                self._unigram_set = True                                 # restored via state_dict
            elif not _WARNED_UNIGRAM_UNSET:
                _WARNED_UNIGRAM_UNSET = True
                warnings.warn(
                    "decode_unigram_prior=True but the unigram_log_prior table is unset "
                    "(all-zero): the decode degenerates to the uniform prior. Call "
                    "PriorBank.set_unigram_log_prior(counts) with the (V,) corpus counts.",
                    UserWarning, stacklevel=3,
                )
        return self.unigram_kappa * self.unigram_log_prior

    def _tau_eff(
        self,
        tau: Optional[float] = None,     # override decode_tau; None -> self.decode_tau
    ) -> torch.Tensor:
        r"""Effective decode temperature tau_eff = tau * exp(-clamp(decode_log_scale, -3, 3))."""
        base_tau = self.decode_tau if tau is None else tau
        return base_tau * torch.exp(-self.decode_log_scale.clamp(-3.0, 3.0))

    def _query_in_decode_frame(
        self,
        mu_q:           torch.Tensor,
        sigma_q:        torch.Tensor,
        canonical_frame: Optional[CanonicalFrameContext],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Return the query in the coordinates owned by the active decode bank.

        Ordinary encoders already produce queries in their decode coordinates and reject a frame
        context so a stale or accidentally threaded frame cannot change their established values.
        ``canonical_content_projected`` instead materializes a diagonal query in the realized frame;
        its canonical diagonal vocabulary bank can only score that query after the exact inverse
        vertex factor from the SAME forward reconstructs a full canonical covariance.
        """
        projected = self.encode_mode == "canonical_content_projected"
        if not projected:
            if canonical_frame is not None:
                raise ValueError(
                    "canonical_frame is only valid for "
                    "encode_mode='canonical_content_projected'"
                )
            return mu_q, sigma_q
        if canonical_frame is None:
            raise ValueError(
                "encode_mode='canonical_content_projected' requires canonical_frame from the "
                "same forward query at every decode boundary"
            )
        if not isinstance(canonical_frame, CanonicalFrameContext):
            raise ValueError(
                "canonical_frame must be a CanonicalFrameContext captured from the same forward "
                f"query, got {type(canonical_frame).__name__}"
            )
        if mu_q.shape != sigma_q.shape:
            raise ValueError(
                "projected decode requires diagonal query mean/variance with identical shapes, got "
                f"{tuple(mu_q.shape)} and {tuple(sigma_q.shape)}"
            )
        if mu_q.dim() < 1:
            raise ValueError("projected decode query must have a trailing coordinate axis")
        expected_frame_shape = (*mu_q.shape[:-1], mu_q.shape[-1], mu_q.shape[-1])
        if (canonical_frame.forward.shape != expected_frame_shape
                or canonical_frame.inverse.shape != expected_frame_shape):
            raise ValueError(
                "canonical_frame shape must match the projected query exactly, expected "
                f"{expected_frame_shape}, got forward={tuple(canonical_frame.forward.shape)} and "
                f"inverse={tuple(canonical_frame.inverse.shape)}"
            )
        if (canonical_frame.forward.dtype != mu_q.dtype
                or canonical_frame.inverse.dtype != mu_q.dtype
                or sigma_q.dtype != mu_q.dtype):
            raise ValueError(
                "canonical_frame dtype must match the projected query exactly, got "
                f"query={mu_q.dtype}, variance={sigma_q.dtype}, "
                f"forward={canonical_frame.forward.dtype}, inverse={canonical_frame.inverse.dtype}"
            )
        if (canonical_frame.forward.device != mu_q.device
                or canonical_frame.inverse.device != mu_q.device
                or sigma_q.device != mu_q.device):
            raise ValueError(
                "canonical_frame device must match the projected query exactly, got "
                f"query={mu_q.device}, variance={sigma_q.device}, "
                f"forward={canonical_frame.forward.device}, inverse={canonical_frame.inverse.device}"
            )
        return pullback_diagonal_query(mu_q, sigma_q, canonical_frame.inverse)

    def decode(
        self,
        mu_q:    torch.Tensor,           # (B, N, K) posterior means
        sigma_q: torch.Tensor,           # (B, N, K) posterior variances

        *,
        tau:             Optional[float] = None,  # override decode_tau; None -> self.decode_tau
        canonical_frame: Optional[CanonicalFrameContext] = None,
    ) -> torch.Tensor:                   # (B, N, V) logits
        r"""Decode logits via the selected kernel; ``use_prior_bank`` is the single gate.

        True (the opt-in pure path): the KL-to-prior readout -KL(q_i || pi_v)/tau_eff with the
        covariance structure given by ``decode_mode`` (diagonal | full). False (ablation): the
        ``linear`` kernel logits = mu_q @ W^T (sigma_q and tau_eff ignored). Routing here -- not
        through a second config value -- keeps ``decode_mode`` and ``use_prior_bank`` from ever
        silently disagreeing (the linear path simply does not consult ``decode_mode``).

        Under ``decode_unigram_prior=True`` the unigram log-prior bias kappa * log pi_v is added
        HERE, after the registered kernel, so every decode mode (linear included) gets it from
        one seam; toggle off adds nothing (byte-identical)."""
        mu_q, sigma_q = self._query_in_decode_frame(mu_q, sigma_q, canonical_frame)
        mode = self.decode_mode if self.use_prior_bank else "linear"
        logits = get_decode(mode)(self, mu_q, sigma_q, self._tau_eff(tau))
        if self.decode_unigram_prior:
            logits = logits + self._unigram_bias()                       # (B, N, V) + (V,)
        return logits

    def reference_decode(
        self,
        mu_q:    torch.Tensor,           # (B, N, K) posterior means
        sigma_q: torch.Tensor,           # (B, N, K) posterior variances

        *,
        tau:     Optional[float] = None,  # override decode_tau; None -> self.decode_tau
    ) -> torch.Tensor:                   # (B, N, V) logits = -D_configured(q || pi_v)/tau_eff
        r"""Authoritative reference decode: -D_configured(q_i || pi_v)/tau_eff via the seam.

        Dispatches through the CONFIGURED family (``self.family``) and divergence functional
        (``self.divergence_family`` at ``self.renyi_order``), broadcasting the seam over the
        vocabulary V in one shot (general but slow, O(B*N*V*K)). This is the same computation the
        registered ``family`` kernel performs, so it stays the oracle for the fast canonical kernels:
        for a canonical gaussian + renyi + alpha=1 config it equals the fused ``diagonal``/``full``
        kernels exactly (and under log-softmax); for a non-Gaussian family or a noncanonical
        divergence it reads the belief out under the SAME geometry the E-step minimized.

        The seam is invoked with ``kl_max=inf``: a DECODE must preserve the full divergence ranking
        over the vocabulary, so the saturation policy (default ``kl_max=100``, which flattens every
        distant prior to a single -100 logit and destroys the argmax) is disabled here. The full q is
        scored against the intentionally DIAGONAL vocabulary-prior table (promoted with diag_embed
        only for a full family). (``nan_to_num`` inside ``safe_kl_clamp`` still maps NaN/+inf from
        degenerate pairs to +inf -> -inf logits.)
        """
        tau_eff = self._tau_eff(tau)
        logits = _decode_family(self, mu_q, sigma_q, tau_eff)           # configured family/divergence
        if self.decode_unigram_prior:
            logits = logits + self._unigram_bias()                       # same seam as decode()
        return logits

    def _validate_fused_ce_targets(
        self,
        targets: torch.Tensor,           # (B, N) next-token ids

        *,
        ignore_index: int = -100,
    ) -> None:
        """Reject nonignored targets outside the vocabulary before fused CE reduction."""
        counted = targets != ignore_index
        invalid = counted & ((targets < 0) | (targets >= self.vocab_size))
        if bool(invalid.any()):
            invalid_target = int(targets[invalid][0].item())
            raise IndexError(f"Target {invalid_target} is out of bounds.")

    def decode_ce_diagonal_chunked(
        self,
        mu_q:    torch.Tensor,           # (B, N, K) posterior means
        sigma_q: torch.Tensor,           # (B, N, K) posterior variances
        targets: torch.Tensor,           # (B, N) next-token ids (-100 = ignore)

        *,
        z_loss_weight: float           = 0.0,   # z-loss coefficient on mean(logsumexp^2); 0.0 = OFF
        tau:           Optional[float] = None,   # override decode_tau; None -> self.decode_tau
        chunk_size:    Optional[int]   = None,   # vocab-chunk width; None -> self.decode_chunk_size
        ignore_index:  int             = -100,
    ) -> torch.Tensor:                   # () scalar mean cross-entropy
        r"""Fused chunked-vocab cross-entropy: the ``diagonal`` decode CE WITHOUT a (B, N, V) tensor.

        Iterates the vocabulary in chunks ``[v0, v1)``, computing each chunk's logits with the SAME
        closed form (and the SAME global centering offset ``c = mean_v(mu_v)``) as ``_decode_diagonal``,
        reducing each chunk to its per-position ``logsumexp`` and gathering the target-token logit, so
        the full ``(B, N, V)`` logit tensor is never materialized. Per position the cross-entropy is
        ``logsumexp_v(logit_v) - logit_target`` (= -log-softmax at the target); the loss is the mean
        over non-ignored positions, exactly matching ``F.cross_entropy(decode(...), targets, ignore_index)``.

        The offset ``c`` is a per-coordinate ``(1, K)`` mean over ALL V, computed in one ``O(V*K)``
        pass with no big tensor, so it is IDENTICAL to the full path (the closed form is
        offset-invariant: ``(mu_q - c) - (mu_v - c) == mu_q - mu_v``). The V-axis reduction (the
        chunk ``logsumexp`` and the target gather) happens INSIDE a gradient-checkpointed function
        that returns only the two ``(B, N)`` per-chunk summaries, so the ``(B, N, Vc)`` chunk logit
        is born and dies inside the checkpoint -- it is recomputed in backward and never crosses the
        boundary (without this the downstream ``logsumexp``/``exp``/``gather`` would save it and the
        peak would stay ``(B, N, V)``). Recompute is deterministic (no RNG here), so value and
        gradient match the full path exactly.

        ``decode_unigram_prior=True`` adds the chunk slice of kappa * log pi_v to each chunk's
        logits BEFORE its logsumexp/gather, so the streamed CE equals the dense CE over the
        shifted logits. ``z_loss_weight > 0`` adds z_loss_weight * mean_i(logsumexp_v logit)^2
        (the streamed total logsumexp, already computed for the CE) -- the guard keeps 0.0
        byte-identical to the pre-kwarg path.
        """
        self._validate_fused_ce_targets(targets, ignore_index=ignore_index)
        tau_eff = self._tau_eff(tau)
        chunk = self.decode_chunk_size if chunk_size is None else chunk_size
        V = self.vocab_size

        sigma_v_all = bounded_variance_from_log(
            self._decode_sigma_log_table(), eps=self.eps,
        )                                                                             # (V, K)
        mu_v_all = self._decode_mu_table()                                  # (V, K) decode table (untied if set)
        c = mu_v_all.mean(dim=0, keepdim=True)                              # (1, K) global v-independent shift
        u_all = self._unigram_bias() if self.decode_unigram_prior else None  # (V,) kappa*log pi_v or None

        mc_q = mu_q - c                                                     # (B, N, K) centered query means
        lhs = _decode_av_lhs(sigma_q, mc_q)                                 # (B, N, 2K) expanded-form left factor
        # Per-position, v-INDEPENDENT term of -KL/tau_eff: it cancels in the CE difference
        # (logsumexp - target_logit) but is carried so each chunk's logits equal _decode_diagonal's.
        per_pos = self.K + torch.log(sigma_q.clamp(min=self.eps)).sum(-1, keepdim=True)  # (B, N, 1)
        coord_delta = None
        delta_per_pos = None
        evidence_lhs = None
        if hasattr(self, "head_evidence_logits"):
            _, coord_delta = self._head_evidence_deltas(dtype=mu_q.dtype, device=mu_q.device)
            delta_per_pos = (
                coord_delta * (1.0 + torch.log(sigma_q.clamp(min=self.eps)))
            ).sum(-1, keepdim=True)
            evidence_lhs = _decode_av_lhs(sigma_q, mc_q, coord_delta)
            if _DECODE_AV_PRECISION == "fp64" and evidence_lhs.dtype is not torch.float64:
                evidence_lhs = evidence_lhs.double()

        def _chunk_summaries(lhs_:    torch.Tensor, per_pos_:        torch.Tensor,
                             mu_v_c:  torch.Tensor, inv_v_c:         torch.Tensor,
                             log_v_c: torch.Tensor, lsum_c:          torch.Tensor,
                             in_chunk_f: torch.Tensor,
                             local_idx: torch.Tensor,
                             u_c:     Optional[torch.Tensor],
                             sq_:     torch.Tensor,
                             mc_q_:   torch.Tensor,
                             coord_delta_: Optional[torch.Tensor],
                             delta_per_pos_: Optional[torch.Tensor],
                             evidence_lhs_: Optional[torch.Tensor]) -> 'tuple[torch.Tensor, torch.Tensor]':
            r"""Reduce one vocab chunk to (lse_chunk, target_contrib), both (B, N), on the inside.

            logit_{i,v} = -0.5(a_v - per_pos)/tau_eff over the chunk (see _decode_diagonal). The
            full (B, N, Vc) chunk logit lives only here so checkpointing frees it after forward.
            ``in_chunk_f`` is a 0/1 (B, N) mask selecting positions whose target falls in this chunk.

            ``sq_``/``mc_q_`` carry the UNEXPANDED query pieces alongside ``lhs_`` so the "exact"
            a_v form can difference before squaring (audit 2026-08-06 F32); the default expanded
            form uses ``lhs_`` and ignores them.
            """
            a_v = _decode_av(sq_, mc_q_, mu_v_c, inv_v_c, lsum_c, lhs=lhs_)   # (B, N, Vc)
            evidence_delta = None
            if coord_delta_ is not None and delta_per_pos_ is not None:
                evidence_delta = _decode_head_evidence_kl_delta(
                    sq_, mc_q_, mu_v_c, inv_v_c, log_v_c,
                    coord_delta_, delta_per_pos_, evidence_lhs_,
                )
            logit_chunk = _decode_analytic_kl_logits(
                a_v, per_pos_, tau_eff, lhs_.dtype, evidence_delta=evidence_delta)
            if u_c is not None:
                logit_chunk = logit_chunk + u_c                            # unigram log-prior chunk slice
            lse_chunk = torch.logsumexp(logit_chunk, dim=-1)               # (B, N)
            gathered = logit_chunk.gather(-1, local_idx.unsqueeze(-1)).squeeze(-1)  # (B, N)
            # SELECT rather than multiply (audit 2026-08-06 F31): `gathered * in_chunk_f` is
            # `-inf * 0.0` = NaN in every chunk that does NOT contain the target, so a degenerate
            # position poisoned target_logit through chunks it has no business contributing to.
            # Byte-identical for finite logits (x*1.0 == x, x*0.0 == 0.0).
            return lse_chunk, torch.where(in_chunk_f > 0, gathered, torch.zeros_like(gathered))

        valid = targets != ignore_index                                    # (B, N) bool
        lse_chunks = []
        target_logit = torch.zeros(mu_q.shape[:-1], device=mu_q.device, dtype=mu_q.dtype)  # (B, N)

        for v0 in range(0, V, chunk):
            v1 = min(v0 + chunk, V)
            mc_v_c = (mu_v_all[v0:v1] - c)                                  # (Vc, K) centered prior means
            inv_v_c = 1.0 / sigma_v_all[v0:v1]                             # (Vc, K)
            log_v_c = torch.log(sigma_v_all[v0:v1])                        # (Vc, K)
            lsum_c = log_v_c.sum(-1)                                      # (Vc,)
            u_c = u_all[v0:v1] if u_all is not None else None              # (Vc,) or None
            # Target gather indices: positions whose target lands in [v0, v1). Ignored positions have
            # target < 0 < v0, so they never match -> target_logit stays 0 for them and `valid` excludes
            # them from the mean. local_idx is clamped to a safe range for the out-of-window rows.
            in_chunk = (targets >= v0) & (targets < v1)                    # (B, N) bool
            in_chunk_f = in_chunk.to(mu_q.dtype)                           # (B, N) 0/1, carried into the checkpoint
            local_idx = (targets - v0).clamp(min=0, max=v1 - v0 - 1)       # (B, N) safe gather index
            grad_active = torch.is_grad_enabled() and lhs.requires_grad
            activation_bytes = _decode_ce_chunk_activation_bytes(lhs, v1 - v0)
            if _decode_ce_should_checkpoint(self.decode_ce_checkpoint, grad_active, activation_bytes):
                lse_chunk, contrib = _checkpoint.checkpoint(
                    _chunk_summaries, lhs, per_pos, mc_v_c, inv_v_c, log_v_c, lsum_c,
                    in_chunk_f, local_idx,
                    u_c, sigma_q, mc_q, coord_delta, delta_per_pos, evidence_lhs,
                    use_reentrant=False,
                )
            else:
                lse_chunk, contrib = _chunk_summaries(
                    lhs, per_pos, mc_v_c, inv_v_c, log_v_c, lsum_c, in_chunk_f, local_idx, u_c,
                    sigma_q, mc_q, coord_delta, delta_per_pos, evidence_lhs,
                )
            lse_chunks.append(lse_chunk)
            target_logit = target_logit + contrib                          # exactly one chunk contributes per valid pos

        # Combine the per-chunk logsumexps into the full-V logsumexp. The stacked summaries are
        # (n_chunks, B, N) = B*N*ceil(V/chunk), negligible vs (B, N, V).
        logsumexp_v = torch.logsumexp(torch.stack(lse_chunks, dim=0), dim=0)  # (B, N)
        ce_per_pos = logsumexp_v - target_logit                           # (B, N) = -log-softmax at target
        # Device-side masked mean: clamp the denominator so an all-ignore microbatch yields a finite
        # grad-connected 0 (the numerator is then 0) without a host sync to branch on valid.sum() == 0.
        # (Matches the full path, whose F.cross_entropy mean over zero counted tokens would be NaN.)
        ce = (ce_per_pos * valid).sum() / valid.sum().clamp_min(1)
        if z_loss_weight > 0.0:
            # z-loss: z_loss_weight * mean_i (log Z_i)^2 over the counted positions, log Z_i the
            # streamed full-V logsumexp above -- calibrates log Z ~ 0 so the decode approximates a
            # normalized observation model. The 0.0 guard keeps the default path byte-identical.
            ce = ce + z_loss_weight * (logsumexp_v ** 2 * valid).sum() / valid.sum().clamp_min(1)
        return ce

    def decode_degenerate_positions(
        self,
        sigma_q: torch.Tensor,           # (B, N, K) diagonal or (B, N, K, K) full posterior dispersion

        *,
        canonical_frame: Optional[CanonicalFrameContext] = None,
    ) -> Optional[torch.Tensor]:         # (B, N) True where the decode cannot score, else None
        r"""Positions the full-covariance decode cannot score, for a CE consumer holding its own targets.

        The fused chunked kernels own their ignore mask and fold this in directly
        (``decode_ce_full_chunked``). The DENSE branch instead hands ``(B, N, V)`` logits to
        ``F.cross_entropy`` in the model, which has no way to learn that a position is degenerate --
        so it asks here and marks those positions ``ignore_index``, keeping the two paths in parity
        on the exclusion contract (audit 2026-08-06 F31).

        ``None`` means "nothing to exclude": a diagonal dispersion is scored without a Cholesky and
        has no failure mode to report. Does NOT touch the fallback counter -- the decode kernel this
        accompanies has already counted the same event.
        """
        if self.encode_mode == "canonical_content_projected" or canonical_frame is not None:
            if sigma_q.dim() < 1:
                raise ValueError("decode dispersion must have a trailing coordinate axis")
            mu_placeholder = torch.zeros_like(sigma_q)
            _, sigma_q = self._query_in_decode_frame(
                mu_placeholder, sigma_q, canonical_frame)
        if sigma_q.dim() < 4 or sigma_q.shape[-1] != sigma_q.shape[-2]:
            return None                                       # diagonal dispersion: no factorization
        _, ok = safe_cholesky(sigma_q, eps=self.eps, rounds=5)
        return ~ok

    def _full_cov_query_invariants(
        self,
        sigma_q: torch.Tensor,           # (B, N, K, K) posterior covariances
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:   # diag_sq, logdet_q, SPD-ok mask
        r"""Per-position, v-INDEPENDENT pieces of the full-cov KL against a DIAGONAL prior.

        Scoring a full q = N(mu_q, Sigma_q) against a diagonal prior pi_v = N(mu_v, diag(sigma_v))
        is the gaussian_full KL with a DIAGONAL second covariance, which collapses every per-pair
        (K, K) Cholesky into matmuls over V PLUS one per-position log|Sigma_q|. This returns the two
        pieces that depend on q only (not on the vocabulary):
            diag_sq = diag(Sigma_q)                    (B, N, K)  -- the raw query variances
            logdet_q = log|Sigma_q|                    (B, N)
        with the same round-zero-first factorization as the gaussian_full closed form
        (families/gaussian.py renyi_closed_form), so the diagonal-prior closed form is value-equal
        to ``_decode_full`` (the per-pair Cholesky seam) without ever forming a (B, N, V, K, K)
        workspace. ``safe_cholesky`` (jittered, never raises) yields a finite log-det where its
        ``ok`` mask is True; a position where every jitter round fails (non-PD Sigma_q) gets
        logdet_q = -inf, so per_pos = K + logdet_q drives every vocab logit to -inf there --
        matching the dense ``_decode_full`` path and PINNED as a cross-path parity contract by
        test_fullcov_alpha_roadmap_2026_06_13::test_full_cov_chunked_matches_dense_on_non_pd
        (audit 2026-07-01 F6). The SPD retraction keeps Sigma_q PD in training, so ok is all-True
        and the -inf branch never engages on the pure path.

        RESOLVED (audit 2026-08-06 F31): a degenerate position is EXCLUDED from the cross-entropy,
        not scored. It used to take ``logdet_q = -inf``, pinned as a cross-path parity contract by
        test_fullcov_alpha_roadmap_2026_06_13::test_full_cov_chunked_matches_dense_on_non_pd (audit
        2026-07-01 F6) -- but an all--inf logit row is not merely an -inf score, it is NaN
        downstream: ``log_softmax`` of it is NaN (on the DENSE path too), and in the chunked
        reduction ``logsumexp_v - target_logit`` is ``-inf - (-inf)``. So the pinned contract was
        itself the NaN generator, and one degenerate position NaN'd the scalar CE for the whole
        batch on BOTH paths.

        A position whose ``Sigma_q`` is not positive definite has no valid likelihood, so scoring it
        is a fiction under any sentinel; the honest reading is that it contributes nothing and its
        absence is visible in the token count. ``ok`` is therefore returned for the CE seams to fold
        into their ignore mask, and ``logdet_q`` gets a FINITE placeholder so nothing NaNs before the
        exclusion applies (the value is free: ``per_pos`` is v-INDEPENDENT and cancels exactly in
        every logit difference, so it cannot bias a surviving position). The event stays counted, so
        a run can tell how many tokens the denominator lost.
        """
        diag_sq = torch.diagonal(sigma_q, dim1=-2, dim2=-1)                # (B, N, K) = diag(Sigma_q)
        L, ok = safe_cholesky(sigma_q, eps=self.eps, rounds=5)
        _count_decode_logdet_fallback(ok)
        # Mask the FACTOR, not the log-det. A failed cholesky_ex returns a finite PARTIAL factor
        # whose diagonal can be zero or negative, so ``log(diag L)`` is -inf/NaN and masking
        # afterwards would leave a ``0 * inf`` NaN in the gradient to L even though the value came
        # out clean. Substituting the identity makes the value (logdet = 0) and the gradient exactly
        # zero there. Byte-identical wherever ``ok``, which on the pure path is everywhere.
        eye = torch.eye(L.shape[-1], device=L.device, dtype=L.dtype)
        logdet_q = _logdet_chol(torch.where(ok[..., None, None], L, eye))   # (B, N)
        return diag_sq, logdet_q, ok

    def _head_evidence_full_marginal_invariants(
        self,
        sigma_q: torch.Tensor,           # (B, N, K, K) posterior covariances
        head_delta: torch.Tensor,        # (H,) exact head-evidence coefficients w_h - 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # weighted (d_h + logdet marginal), all-block-ok
        """Compute the query-only marginal entropy delta under the safe-Cholesky policy."""
        per_pos = torch.zeros(sigma_q.shape[:-2], dtype=sigma_q.dtype, device=sigma_q.device)
        all_ok = torch.ones(sigma_q.shape[:-2], dtype=torch.bool, device=sigma_q.device)
        start = 0
        for delta_h, dim in zip(head_delta, self._head_evidence_irrep_dims):
            stop = start + dim
            marginal = sigma_q[..., start:stop, start:stop]
            L_h, ok_h = safe_cholesky(marginal, eps=self.eps, rounds=5)
            eye_h = torch.eye(dim, dtype=L_h.dtype, device=L_h.device)
            logdet_h = _logdet_chol(torch.where(ok_h[..., None, None], L_h, eye_h))
            per_pos = per_pos + delta_h * (dim + logdet_h)
            all_ok = all_ok & ok_h
            start = stop
        return per_pos.unsqueeze(-1), all_ok

    def decode_ce_full_chunked(
        self,
        mu_q:    torch.Tensor,           # (B, N, K) posterior means
        sigma_q: torch.Tensor,           # (B, N, K, K) posterior covariances
        targets: torch.Tensor,           # (B, N) next-token ids (-100 = ignore)

        *,
        z_loss_weight: float           = 0.0,   # z-loss coefficient on mean(logsumexp^2); 0.0 = OFF
        tau:           Optional[float] = None,   # override decode_tau; None -> self.decode_tau
        chunk_size:    Optional[int]   = None,   # vocab-chunk width; None -> self.decode_chunk_size
        ignore_index:  int             = -100,
        canonical_frame: Optional[CanonicalFrameContext] = None,
    ) -> torch.Tensor:                   # () scalar mean cross-entropy
        r"""Fused chunked-vocab cross-entropy for the FULL-covariance KL decode WITHOUT the dense
        (B, N, V) logits OR the (B, N, V, K, K) per-pair Cholesky workspace ``_decode_full`` builds.

        The prior table is DIAGONAL (sigma_log_embed), so KL(q_full || pi_v_diag) needs no per-pair
        (K, K) Cholesky: the v-dependent trace and Mahalanobis terms are matmuls over V and the only
        (K, K) work is ONE log|Sigma_q| per position (``_full_cov_query_invariants``). This is the
        full-cov twin of ``decode_ce_diagonal_chunked`` -- same streaming logsumexp + target gather
        inside a gradient checkpoint, same global centering offset c = mean_v(mu_v) for fp32
        stability -- with the diagonal query variance replaced by diag(Sigma_q), the prior kept at
        its existing variance floor, and the per-position v-independent term K + sum_k log sigma_q
        replaced by K + log|Sigma_q|. Value-equal to F.cross_entropy(_decode_full(...)) to the
        decode's atol-1e-3 (tests/test_fullcov_alpha_roadmap_2026_06_13.py). The unigram-prior
        chunk-slice add and the z_loss_weight term follow ``decode_ce_diagonal_chunked`` exactly
        (see there); both default OFF / byte-identical.
        """
        mu_q, sigma_q = self._query_in_decode_frame(mu_q, sigma_q, canonical_frame)
        self._validate_fused_ce_targets(targets, ignore_index=ignore_index)
        tau_eff = self._tau_eff(tau)
        chunk = self.decode_chunk_size if chunk_size is None else chunk_size
        V = self.vocab_size

        sigma_v_all = bounded_variance_from_log(
            self._decode_sigma_log_table(), eps=self.eps,
        )                                                                             # (V, K)
        mu_v_all = self._decode_mu_table()                                  # (V, K) decode table (untied if set)
        c = mu_v_all.mean(dim=0, keepdim=True)                              # (1, K) global v-independent shift
        u_all = self._unigram_bias() if self.decode_unigram_prior else None  # (V,) kappa*log pi_v or None

        diag_sq, logdet_q, spd_ok = self._full_cov_query_invariants(sigma_q)  # (B,N,K), (B,N), (B,N)
        mc_q = mu_q - c                                                     # (B, N, K) centered query means
        lhs = _decode_av_lhs(diag_sq, mc_q)                                 # (B, N, 2K) expanded-form left factor
        # v-INDEPENDENT term of -KL/tau_eff (cancels in the CE difference, carried so each chunk's
        # logits equal _decode_full's): K + log|Sigma_q| (the full-cov analogue of K + sum_k log sigma_q).
        per_pos = self.K + logdet_q.unsqueeze(-1)                          # (B, N, 1)
        coord_delta = None
        delta_per_pos = None
        evidence_lhs = None
        block_ok = torch.ones_like(spd_ok)
        if hasattr(self, "head_evidence_logits"):
            head_delta, coord_delta = self._head_evidence_deltas(
                dtype=mu_q.dtype, device=mu_q.device)
            delta_per_pos, block_ok = self._head_evidence_full_marginal_invariants(
                sigma_q, head_delta)
            evidence_lhs = _decode_av_lhs(diag_sq, mc_q, coord_delta)
            if _DECODE_AV_PRECISION == "fp64" and evidence_lhs.dtype is not torch.float64:
                evidence_lhs = evidence_lhs.double()

        def _chunk_summaries(lhs_:    torch.Tensor, per_pos_:        torch.Tensor,
                             mu_v_c:  torch.Tensor, inv_v_c:         torch.Tensor,
                             log_v_c: torch.Tensor, lsum_c:          torch.Tensor,
                             in_chunk_f: torch.Tensor,
                             local_idx: torch.Tensor,
                             u_c:     Optional[torch.Tensor],
                             sq_:     torch.Tensor,
                             mc_q_:   torch.Tensor,
                             coord_delta_: Optional[torch.Tensor],
                             delta_per_pos_: Optional[torch.Tensor],
                             evidence_lhs_: Optional[torch.Tensor]) -> 'tuple[torch.Tensor, torch.Tensor]':
            r"""Reduce one vocab chunk to (lse_chunk, target_contrib), both (B, N), on the inside.

            a_v = sum_k[(diag(Sigma_q) + (mc_q-mc_v)^2)/sigma_v] + sum_k log sigma_v
                = trace_term + mahalanobis + log|diag(sigma_v)|, the gaussian_full KL with a
            diagonal prior; logit = -0.5(a_v - per_pos)/tau_eff. The (B, N, Vc) chunk logit lives
            only here so checkpointing frees it after forward.

            ``sq_``/``mc_q_`` carry the UNEXPANDED query pieces alongside ``lhs_`` so the "exact"
            a_v form can difference before squaring (audit 2026-08-06 F32).
            """
            a_v = _decode_av(sq_, mc_q_, mu_v_c, inv_v_c, lsum_c, lhs=lhs_)   # (B, N, Vc)
            evidence_delta = None
            if coord_delta_ is not None and delta_per_pos_ is not None:
                evidence_delta = _decode_head_evidence_kl_delta(
                    sq_, mc_q_, mu_v_c, inv_v_c, log_v_c,
                    coord_delta_, delta_per_pos_, evidence_lhs_,
                )
            logit_chunk = _decode_analytic_kl_logits(
                a_v, per_pos_, tau_eff, lhs_.dtype, evidence_delta=evidence_delta)
            if u_c is not None:
                logit_chunk = logit_chunk + u_c                            # unigram log-prior chunk slice
            lse_chunk = torch.logsumexp(logit_chunk, dim=-1)               # (B, N)
            gathered = logit_chunk.gather(-1, local_idx.unsqueeze(-1)).squeeze(-1)  # (B, N)
            # SELECT rather than multiply (audit 2026-08-06 F31): `gathered * in_chunk_f` is
            # `-inf * 0.0` = NaN in every chunk that does NOT contain the target, so a degenerate
            # position poisoned target_logit through chunks it has no business contributing to.
            # Byte-identical for finite logits (x*1.0 == x, x*0.0 == 0.0).
            return lse_chunk, torch.where(in_chunk_f > 0, gathered, torch.zeros_like(gathered))

        # A position whose Sigma_q is not PD is EXCLUDED, exactly like an ignore_index token, rather
        # than scored (audit 2026-08-06 F31); it leaves both the numerator and the denominator, and
        # _count_decode_logdet_fallback has already recorded it. The dense path drops the same
        # positions via PriorBank.decode_degenerate_positions, so the two stay in parity.
        valid = (targets != ignore_index) & spd_ok & block_ok              # (B, N) bool
        lse_chunks = []
        target_logit = torch.zeros(mu_q.shape[:-1], device=mu_q.device, dtype=mu_q.dtype)  # (B, N)

        for v0 in range(0, V, chunk):
            v1 = min(v0 + chunk, V)
            mc_v_c = (mu_v_all[v0:v1] - c)                                  # (Vc, K) centered prior means
            inv_v_c = 1.0 / sigma_v_all[v0:v1]                             # (Vc, K) = 1/sigma_v
            log_v_c = torch.log(sigma_v_all[v0:v1])                        # (Vc, K)
            lsum_c = log_v_c.sum(-1)                                      # (Vc,) = sum_k log sigma_v
            u_c = u_all[v0:v1] if u_all is not None else None              # (Vc,) or None
            in_chunk = (targets >= v0) & (targets < v1)                    # (B, N) bool
            in_chunk_f = in_chunk.to(mu_q.dtype)                           # (B, N) 0/1, carried into the checkpoint
            local_idx = (targets - v0).clamp(min=0, max=v1 - v0 - 1)       # (B, N) safe gather index
            grad_active = torch.is_grad_enabled() and lhs.requires_grad
            activation_bytes = _decode_ce_chunk_activation_bytes(lhs, v1 - v0)
            if _decode_ce_should_checkpoint(self.decode_ce_checkpoint, grad_active, activation_bytes):
                lse_chunk, contrib = _checkpoint.checkpoint(
                    _chunk_summaries, lhs, per_pos, mc_v_c, inv_v_c, log_v_c, lsum_c,
                    in_chunk_f, local_idx, u_c, diag_sq, mc_q, coord_delta, delta_per_pos,
                    evidence_lhs,
                    use_reentrant=False,
                )
            else:
                lse_chunk, contrib = _chunk_summaries(
                    lhs, per_pos, mc_v_c, inv_v_c, log_v_c, lsum_c, in_chunk_f, local_idx, u_c,
                    diag_sq, mc_q, coord_delta, delta_per_pos, evidence_lhs,
                )
            lse_chunks.append(lse_chunk)
            target_logit = target_logit + contrib                          # exactly one chunk contributes per valid pos

        logsumexp_v = torch.logsumexp(torch.stack(lse_chunks, dim=0), dim=0)  # (B, N)
        ce_per_pos = logsumexp_v - target_logit                            # (B, N) = -log-softmax at target
        # Device-side masked mean: clamp the denominator so an all-ignore microbatch yields a finite
        # grad-connected 0 (the numerator is then 0) without a host sync to branch on valid.sum() == 0.
        ce = (ce_per_pos * valid).sum() / valid.sum().clamp_min(1)
        if z_loss_weight > 0.0:
            # z-loss on the streamed log Z (see decode_ce_diagonal_chunked); 0.0 guard = byte-identical.
            ce = ce + z_loss_weight * (logsumexp_v ** 2 * valid).sum() / valid.sum().clamp_min(1)
        return ce

    def decode_ce_linear_chunked(
        self,
        mu_q:    torch.Tensor,           # (B, N, K) posterior means
        targets: torch.Tensor,           # (B, N) next-token ids (-100 = ignore)

        *,
        z_loss_weight: float                 = 0.0,   # z-loss coefficient on mean(logsumexp^2); 0.0 = OFF
        chunk_size:    Optional[int]         = None,  # vocab-chunk width; None -> self.decode_chunk_size
        ignore_index:  int                   = -100,
    ) -> torch.Tensor:                   # () scalar mean cross-entropy
        r"""Fused chunked-vocab cross-entropy for the LINEAR decode (``use_prior_bank=False``).

        The ``_decode_linear`` CE -- ``logits = x @ W^T (+ b)`` -> ``F.cross_entropy`` -- WITHOUT
        the (B, N, V) logit tensor (plus cross_entropy's same-size log-softmax copy, both retained
        for backward on the dense path; the dominant decode VRAM at large B, vram audit 2026-06-10).
        Same streaming contract as ``decode_ce_diagonal_chunked``: each vocab chunk's logits are
        born and die inside a gradient-checkpointed reduction that returns only the (B, N) chunk
        logsumexp and target-logit summaries; recompute is deterministic, so value and gradient (to
        mu_q, W, and b) match the dense path exactly. The unigram-prior chunk-slice add and the
        z_loss_weight term follow ``decode_ce_diagonal_chunked`` (see there); both default OFF.
        """
        self._validate_fused_ce_targets(targets, ignore_index=ignore_index)
        chunk = self.decode_chunk_size if chunk_size is None else chunk_size
        V = self.vocab_size
        W = self.output_proj_weight                                        # (V, K)
        bias = self.output_proj_bias                                       # (V,) or None
        u_all = self._unigram_bias() if self.decode_unigram_prior else None  # (V,) kappa*log pi_v or None

        def _chunk_summaries(mu_:     torch.Tensor, w_c:       torch.Tensor,
                             in_chunk_f: torch.Tensor, local_idx: torch.Tensor,
                             b_c:     Optional[torch.Tensor],
                             u_c:     Optional[torch.Tensor]) -> 'tuple[torch.Tensor, torch.Tensor]':
            r"""Reduce one vocab chunk to (lse_chunk, target_contrib), both (B, N), on the inside."""
            logit_chunk = mu_ @ w_c.transpose(-1, -2)                      # (B, N, Vc)
            if b_c is not None:
                logit_chunk = logit_chunk + b_c                            # learned log-unigram prior
            if u_c is not None:
                logit_chunk = logit_chunk + u_c                            # fixed unigram log-prior slice
            lse_chunk = torch.logsumexp(logit_chunk, dim=-1)               # (B, N)
            gathered = logit_chunk.gather(-1, local_idx.unsqueeze(-1)).squeeze(-1)  # (B, N)
            # SELECT rather than multiply (audit 2026-08-06 F31): `gathered * in_chunk_f` is
            # `-inf * 0.0` = NaN in every chunk that does NOT contain the target, so a degenerate
            # position poisoned target_logit through chunks it has no business contributing to.
            # Byte-identical for finite logits (x*1.0 == x, x*0.0 == 0.0).
            return lse_chunk, torch.where(in_chunk_f > 0, gathered, torch.zeros_like(gathered))

        valid = targets != ignore_index                                    # (B, N) bool
        lse_chunks = []
        target_logit = torch.zeros(mu_q.shape[:-1], device=mu_q.device, dtype=mu_q.dtype)  # (B, N)

        for v0 in range(0, V, chunk):
            v1 = min(v0 + chunk, V)
            w_c = W[v0:v1]                                                 # (Vc, K)
            b_c = bias[v0:v1] if bias is not None else None                # (Vc,) or None
            u_c = u_all[v0:v1] if u_all is not None else None              # (Vc,) or None
            in_chunk = (targets >= v0) & (targets < v1)                    # (B, N) bool
            in_chunk_f = in_chunk.to(mu_q.dtype)                           # (B, N) 0/1, carried into the checkpoint
            local_idx = (targets - v0).clamp(min=0, max=v1 - v0 - 1)       # (B, N) safe gather index
            grad_active = torch.is_grad_enabled() and (mu_q.requires_grad or W.requires_grad)
            activation_bytes = _decode_ce_chunk_activation_bytes(mu_q, v1 - v0)
            if _decode_ce_should_checkpoint(self.decode_ce_checkpoint, grad_active, activation_bytes):
                lse_chunk, contrib = _checkpoint.checkpoint(
                    _chunk_summaries, mu_q, w_c, in_chunk_f, local_idx, b_c, u_c,
                    use_reentrant=False,
                )
            else:
                lse_chunk, contrib = _chunk_summaries(mu_q, w_c, in_chunk_f, local_idx, b_c, u_c)
            lse_chunks.append(lse_chunk)
            target_logit = target_logit + contrib                          # exactly one chunk contributes per valid pos

        logsumexp_v = torch.logsumexp(torch.stack(lse_chunks, dim=0), dim=0)  # (B, N)
        ce_per_pos = logsumexp_v - target_logit                           # (B, N) = -log-softmax at target
        # Device-side masked mean: clamp the denominator so an all-ignore microbatch yields a finite
        # grad-connected 0 (the numerator is then 0) without a host sync to branch on valid.sum() == 0.
        ce = (ce_per_pos * valid).sum() / valid.sum().clamp_min(1)
        if z_loss_weight > 0.0:
            # z-loss on the streamed log Z (see decode_ce_diagonal_chunked); 0.0 guard = byte-identical.
            ce = ce + z_loss_weight * (logsumexp_v ** 2 * valid).sum() / valid.sum().clamp_min(1)
        return ce

    def decode_ce_expected_likelihood_chunked(
        self,
        mu_q:    torch.Tensor,           # (B, N, K) posterior means
        sigma_q: torch.Tensor,           # (B, N, K) posterior variances
        targets: torch.Tensor,           # (B, N) next-token ids (-100 = ignore)

        *,
        z_loss_weight: float           = 0.0,   # z-loss coefficient on mean(logsumexp^2); 0.0 = OFF
        tau:           Optional[float] = None,   # override decode_tau; None -> self.decode_tau
        chunk_size:    Optional[int]   = None,   # vocab-chunk width; None -> self.decode_chunk_size
        ignore_index:  int             = -100,
    ) -> torch.Tensor:                   # () scalar mean cross-entropy
        r"""Fused chunked-vocab cross-entropy for the EXPECTED-LIKELIHOOD decode (diagonal only).

        The fused-CE twin of ``decode_mode='expected_likelihood_chunked'`` (see
        ``_decode_expected_likelihood_chunked`` for the scoring math): the same streaming contract
        as ``decode_ce_diagonal_chunked`` -- each chunk's (B, N, Vc) logits are born and die inside
        a gradient-checkpointed reduction returning only the (B, N) chunk logsumexp and target
        summaries, so the (B, N, V) tensor is never materialized. The couplings sigma_q + sigma_v
        block the diagonal kernel's single-matmul trick, so each chunk broadcasts a (B, N, Vc, K)
        workspace instead (bounded by the chunk width; freed by the checkpoint). The unigram-prior
        chunk-slice add and the z_loss_weight term follow ``decode_ce_diagonal_chunked`` (see
        there); both default OFF.
        """
        self._validate_fused_ce_targets(targets, ignore_index=ignore_index)
        tau_eff = self._tau_eff(tau)
        chunk = self.decode_chunk_size if chunk_size is None else chunk_size
        V = self.vocab_size

        sigma_v_all = bounded_variance_from_log(
            self._decode_sigma_log_table(), eps=self.eps,
        )                                                                             # (V, K)
        mu_v_all = self._decode_mu_table()                                  # (V, K) decode table (untied if set)
        u_all = self._unigram_bias() if self.decode_unigram_prior else None  # (V,) kappa*log pi_v or None

        def _chunk_summaries(mu_q_:   torch.Tensor, sigma_q_:   torch.Tensor,
                             mu_v_c:  torch.Tensor, sigma_v_c:  torch.Tensor,
                             in_chunk_f: torch.Tensor, local_idx: torch.Tensor,
                             u_c:     Optional[torch.Tensor]) -> 'tuple[torch.Tensor, torch.Tensor]':
            r"""Reduce one vocab chunk to (lse_chunk, target_contrib), both (B, N), on the inside."""
            d = mu_q_.unsqueeze(-2) - mu_v_c                               # (B, N, Vc, K)
            s = sigma_q_.unsqueeze(-2) + sigma_v_c                         # (B, N, Vc, K) convolved variances
            logit_chunk = -0.5 * (d ** 2 / s + torch.log(s)).sum(-1) / tau_eff   # (B, N, Vc)
            if u_c is not None:
                logit_chunk = logit_chunk + u_c                            # unigram log-prior chunk slice
            lse_chunk = torch.logsumexp(logit_chunk, dim=-1)               # (B, N)
            gathered = logit_chunk.gather(-1, local_idx.unsqueeze(-1)).squeeze(-1)  # (B, N)
            # SELECT rather than multiply (audit 2026-08-06 F31): `gathered * in_chunk_f` is
            # `-inf * 0.0` = NaN in every chunk that does NOT contain the target, so a degenerate
            # position poisoned target_logit through chunks it has no business contributing to.
            # Byte-identical for finite logits (x*1.0 == x, x*0.0 == 0.0).
            return lse_chunk, torch.where(in_chunk_f > 0, gathered, torch.zeros_like(gathered))

        valid = targets != ignore_index                                    # (B, N) bool
        lse_chunks = []
        target_logit = torch.zeros(mu_q.shape[:-1], device=mu_q.device, dtype=mu_q.dtype)  # (B, N)

        for v0 in range(0, V, chunk):
            v1 = min(v0 + chunk, V)
            mu_v_c = mu_v_all[v0:v1]                                       # (Vc, K)
            sigma_v_c = sigma_v_all[v0:v1]                                 # (Vc, K)
            u_c = u_all[v0:v1] if u_all is not None else None              # (Vc,) or None
            in_chunk = (targets >= v0) & (targets < v1)                    # (B, N) bool
            in_chunk_f = in_chunk.to(mu_q.dtype)                           # (B, N) 0/1, carried into the checkpoint
            local_idx = (targets - v0).clamp(min=0, max=v1 - v0 - 1)       # (B, N) safe gather index
            grad_active = torch.is_grad_enabled() and (mu_q.requires_grad or mu_v_all.requires_grad)
            # This closure's largest workspace is (B, N, Vc, K) (the `d`/`s` broadcast the diagonal
            # kernel's single-matmul trick cannot use, per the docstring above), so the byte estimate
            # carries the extra K factor a bare (B, N, Vc) count would miss.
            activation_bytes = _decode_ce_chunk_activation_bytes(mu_q, v1 - v0, inner=self.K)
            if _decode_ce_should_checkpoint(self.decode_ce_checkpoint, grad_active, activation_bytes):
                lse_chunk, contrib = _checkpoint.checkpoint(
                    _chunk_summaries, mu_q, sigma_q, mu_v_c, sigma_v_c, in_chunk_f, local_idx,
                    u_c, use_reentrant=False,
                )
            else:
                lse_chunk, contrib = _chunk_summaries(
                    mu_q, sigma_q, mu_v_c, sigma_v_c, in_chunk_f, local_idx, u_c
                )
            lse_chunks.append(lse_chunk)
            target_logit = target_logit + contrib                          # exactly one chunk contributes per valid pos

        logsumexp_v = torch.logsumexp(torch.stack(lse_chunks, dim=0), dim=0)  # (B, N)
        ce_per_pos = logsumexp_v - target_logit                           # (B, N) = -log-softmax at target
        # Device-side masked mean: clamp the denominator so an all-ignore microbatch yields a finite
        # grad-connected 0 (the numerator is then 0) without a host sync to branch on valid.sum() == 0.
        ce = (ce_per_pos * valid).sum() / valid.sum().clamp_min(1)
        if z_loss_weight > 0.0:
            # z-loss on the streamed log Z (see decode_ce_diagonal_chunked); 0.0 guard = byte-identical.
            ce = ce + z_loss_weight * (logsumexp_v ** 2 * valid).sum() / valid.sum().clamp_min(1)
        return ce

    def decode_ce_family_chunked(
        self,
        mu_q:    torch.Tensor,           # (B, N, K) posterior means
        sigma_q: torch.Tensor,           # (B, N, K) or (B, N, K, K) posterior (co)variances
        targets: torch.Tensor,           # (B, N) next-token ids (-100 = ignore)

        *,
        z_loss_weight: float           = 0.0,   # z-loss coefficient on mean(logsumexp^2); 0.0 = OFF
        tau:           Optional[float] = None,   # override decode_tau; None -> self.decode_tau
        chunk_size:    Optional[int]   = None,   # vocab-chunk width; None -> self.decode_chunk_size
        ignore_index:  int             = -100,
    ) -> torch.Tensor:                   # () scalar mean cross-entropy
        r"""Fused chunked-vocab cross-entropy for the FAMILY-consistent decode (``decode_mode=
        'family_chunked'``) WITHOUT the dense (B, N, V) logits.

        The family-consistent twin of ``decode_ce_diagonal_chunked``: each vocab chunk streams
        through the SAME registered functional ``get_functional(self.divergence_family)`` at
        ``alpha=self.renyi_order`` (logits = -D_configured(q || pi_v)/tau_eff, ``kl_max=inf``) and the
        same fused log-sum-exp/gather reduction inside a gradient checkpoint, so the (B, N, V) tensor
        is never materialized. The vocabulary prior table is DIAGONAL; a FULL family promotes each
        chunk with ``diag_embed`` and materializes only a (B, N, Vc, K, K) functional workspace inside
        the checkpoint (never a full SPD vocabulary table). Value/gradient-equal to the dense
        ``family`` decode -> cross-entropy. The unigram-prior chunk-slice add and the z_loss_weight
        term follow ``decode_ce_diagonal_chunked`` (see there); both default OFF.
        """
        if _uses_canonical_full_family_decode(self, mu_q, sigma_q):
            return self.decode_ce_full_chunked(
                mu_q,
                sigma_q,
                targets,
                z_loss_weight=z_loss_weight,
                tau=tau,
                chunk_size=chunk_size,
                ignore_index=ignore_index,
            )

        self._validate_fused_ce_targets(targets, ignore_index=ignore_index)
        tau_eff = self._tau_eff(tau)
        chunk = self.decode_chunk_size if chunk_size is None else chunk_size
        V = self.vocab_size

        family_cls = get_family(self.family)
        is_full = family_cls.cov_kind == "full"
        functional = get_functional(self.divergence_family)

        # A FULL family's functional workspace is (B, N, Vc, K, K) x DECODE_CE_FAMILY_WORKSETS, so
        # the shared decode_chunk_size -- sized for the (B, N, Vc) kernels -- has to be re-read in
        # this route's own units (audit 2026-08-07). Inert for a diagonal family (inner == 1).
        inner = self.K * self.K if is_full else 1
        workspace_bytes_per_scalar = _full_family_workspace_bytes_per_scalar(
            family_cls, mu_q, mu_q, sigma_q,
        )
        chunk = _decode_ce_family_effective_chunk(
            mu_q, chunk, inner, workspace_bytes_per_scalar=workspace_bytes_per_scalar,
        )

        sigma_v_all = bounded_variance_from_log(
            self._decode_sigma_log_table(), eps=self.eps,
        )                                                                             # (V, K) diagonal prior
        mu_v_all = self._decode_mu_table()                                  # (V, K) decode table (untied if set)
        u_all = self._unigram_bias() if self.decode_unigram_prior else None  # (V,) kappa*log pi_v or None

        # Positions the decode cannot score (audit 2026-08-07; see the `valid` mask below). The
        # covariance is SANITIZED here rather than the energy being masked afterwards: with
        # kl_max=inf a non-PD Sigma_q yields NaN, `nan * 0.0` is `nan` at the masked mean, and
        # logsumexp/softmax backward turns the zero gradient into `0 * nan` as well -- so a masked
        # value would still poison both passes. Substituting a benign SPD covariance at positions
        # that are excluded anyway keeps the whole graph finite and cannot affect the loss, because
        # `valid` drops these from the numerator AND the denominator. None => diagonal dispersion,
        # no factorization, nothing to sanitize (byte-identical for the diagonal family).
        degenerate = self.decode_degenerate_positions(sigma_q)               # (B, N) bool, or None
        if degenerate is not None:
            eye = torch.eye(sigma_q.shape[-1], device=sigma_q.device, dtype=sigma_q.dtype)
            sigma_q = torch.where(degenerate[..., None, None], eye.expand_as(sigma_q), sigma_q)

        q_mu = mu_q.unsqueeze(-2)                                            # (B, N, 1, K)
        q_sigma = sigma_q.unsqueeze(-3 if is_full else -2)                   # (B, N, 1, K[, K])

        def _chunk_summaries(q_mu_:   torch.Tensor, q_sigma_:   torch.Tensor,
                             mu_v_c:  torch.Tensor, sigma_v_c:  torch.Tensor,
                             in_chunk_f: torch.Tensor, local_idx: torch.Tensor,
                             u_c:     Optional[torch.Tensor]) -> 'tuple[torch.Tensor, torch.Tensor]':
            r"""Reduce one vocab chunk to (lse_chunk, target_contrib), both (B, N), on the inside.

            The functional workspace ((B, N, Vc) diagonal / (B, N, Vc, K, K) full) is born and dies
            here so checkpointing frees it after forward; recompute is deterministic (the functional
            has no RNG), so value and gradient match the dense family decode exactly.
            """
            q = family_cls(q_mu_, q_sigma_)
            p = family_cls(mu_v_c, sigma_v_c)
            energy = functional(q, p, alpha=self.renyi_order,
                                kl_max=float("inf"), eps=self.eps)         # (B, N, Vc)
            logit_chunk = -energy / tau_eff                                # (B, N, Vc)
            if u_c is not None:
                logit_chunk = logit_chunk + u_c                            # unigram log-prior chunk slice
            lse_chunk = torch.logsumexp(logit_chunk, dim=-1)               # (B, N)
            gathered = logit_chunk.gather(-1, local_idx.unsqueeze(-1)).squeeze(-1)  # (B, N)
            # SELECT rather than multiply (audit 2026-08-06 F31): `gathered * in_chunk_f` is
            # `-inf * 0.0` = NaN in every chunk that does NOT contain the target, so a degenerate
            # position poisoned target_logit through chunks it has no business contributing to.
            # Byte-identical for finite logits (x*1.0 == x, x*0.0 == 0.0).
            return lse_chunk, torch.where(in_chunk_f > 0, gathered, torch.zeros_like(gathered))

        # A position whose Sigma_q is not PD is EXCLUDED, exactly like an ignore_index token, rather
        # than scored (audit 2026-08-06 F31). This route was left OUT of that fix (audit 2026-08-07):
        # the mask here was a bare ``targets != ignore_index``, so one degenerate position sent a
        # non-finite energy through logsumexp and NaN'd the scalar CE for the WHOLE batch -- measured
        # ``fused CE=nan`` here against ``3.0012598`` on the full_chunked twin at identical inputs.
        # The dense branch's guard (model.py) does not cover this route either, because
        # ``family_chunked`` registers supports_chunked=True and is served by the fused branch.
        # ``decode_degenerate_positions`` returns None for a diagonal dispersion (no factorization,
        # so no failure mode to report), which keeps the diagonal family byte-identical. The mask is
        # computed once above, before the covariance is sanitized.
        valid = targets != ignore_index                                    # (B, N) bool
        if degenerate is not None:
            valid = valid & ~degenerate
        lse_chunks = []
        target_logit = torch.zeros(mu_q.shape[:-1], device=mu_q.device, dtype=mu_q.dtype)  # (B, N)

        # Promote the whole diagonal prior table ONCE rather than per slice when it is small enough
        # to be worth it (audit 2026-08-07). The per-slice ``diag_embed`` writes the same (V, K, K)
        # bytes in total but pays one kernel launch per slice, and this route now runs many more,
        # narrower slices than the raw chunk implied; ``[v0:v1]`` of a promoted table is a view.
        #
        # GATED, because hoisting is not free at every shape. Under grad the per-slice tensors are
        # all retained anyway (``checkpoint`` saves its inputs), so the totals match and hoisting is
        # strictly fewer launches -- but under ``no_grad`` each slice would otherwise be freed as the
        # loop advances, and materializing the whole table would raise the peak from one slice to
        # V*K*K. At the live V=50257, K=20 that table is 80 MB; at K=210 it would be 8.9 GB. The
        # ceiling keeps the hoist to shapes where it cannot become the new memory problem.
        promoted_bytes = V * inner * sigma_v_all.element_size()
        hoist_prior = is_full and promoted_bytes <= DECODE_CE_FAMILY_WORKSPACE_BYTES
        sigma_v_full = torch.diag_embed(sigma_v_all) if hoist_prior else sigma_v_all   # (V, K[, K])

        # Both of these are loop-INVARIANT: the checkpoint decision is now keyed on the whole-vocab
        # retention (see below), and grad_active reads only the two leaves. Computing them per slice
        # re-derived the same answer once per iteration.
        grad_active = torch.is_grad_enabled() and (mu_q.requires_grad or mu_v_all.requires_grad)
        # What "not checkpointing" actually costs is the workspace of EVERY slice, held live into
        # backward at once -- not one slice's (audit 2026-08-07). That total is B*N*V*inner*itemsize,
        # INVARIANT under the slice width, so gating on the per-slice figure made the decision a
        # function of the very knob that cannot change it: narrowing the slice (which this route now
        # does above to bound the transient) would have walked the per-slice estimate under the
        # threshold and silently switched checkpointing OFF, retaining every slice instead of one.
        # The width is passed as V, and `worksets` counts the two simultaneous (B, N, Vc, K, K)
        # triangular-solve buffers the full route really allocates (families/gaussian.py:117-118),
        # which the one-tensor estimate halved.
        activation_bytes = _decode_ce_chunk_activation_bytes(
            mu_q, V, inner=inner * (DECODE_CE_FAMILY_WORKSETS if is_full else 1),
            workspace_bytes_per_scalar=workspace_bytes_per_scalar,
        )
        should_checkpoint = _decode_ce_should_checkpoint(
            self.decode_ce_checkpoint, grad_active, activation_bytes)

        for v0 in range(0, V, chunk):
            v1 = min(v0 + chunk, V)
            mu_v_c = mu_v_all[v0:v1]                                       # (Vc, K)
            sigma_v_c = sigma_v_full[v0:v1]                                # (Vc, K[, K])
            if is_full and not hoist_prior:
                sigma_v_c = torch.diag_embed(sigma_v_c)                    # table too large to hoist: promote per slice
            u_c = u_all[v0:v1] if u_all is not None else None              # (Vc,) or None
            in_chunk = (targets >= v0) & (targets < v1)                    # (B, N) bool
            in_chunk_f = in_chunk.to(mu_q.dtype)                           # (B, N) 0/1, carried into the checkpoint
            local_idx = (targets - v0).clamp(min=0, max=v1 - v0 - 1)       # (B, N) safe gather index
            if should_checkpoint:
                lse_chunk, contrib = _checkpoint.checkpoint(
                    _chunk_summaries, q_mu, q_sigma, mu_v_c, sigma_v_c, in_chunk_f, local_idx,
                    u_c, use_reentrant=False,
                )
            else:
                lse_chunk, contrib = _chunk_summaries(
                    q_mu, q_sigma, mu_v_c, sigma_v_c, in_chunk_f, local_idx, u_c
                )
            lse_chunks.append(lse_chunk)
            target_logit = target_logit + contrib                          # exactly one chunk contributes per valid pos

        logsumexp_v = torch.logsumexp(torch.stack(lse_chunks, dim=0), dim=0)  # (B, N)
        ce_per_pos = logsumexp_v - target_logit                            # (B, N) = -log-softmax at target
        # Device-side masked mean: clamp the denominator so an all-ignore microbatch yields a finite
        # grad-connected 0 (the numerator is then 0) without a host sync to branch on valid.sum() == 0.
        ce = (ce_per_pos * valid).sum() / valid.sum().clamp_min(1)
        if z_loss_weight > 0.0:
            # z-loss on the streamed log Z (see decode_ce_diagonal_chunked); 0.0 guard = byte-identical.
            ce = ce + z_loss_weight * (logsumexp_v ** 2 * valid).sum() / valid.sum().clamp_min(1)
        return ce


EncodeCallable = Callable[[PriorBank, torch.Tensor], BeliefState]
DecodeCallable = Callable[
    [PriorBank, torch.Tensor, torch.Tensor, torch.Tensor],
    torch.Tensor,
]


class GeometricFusedCECallable(Protocol):
    """Fused CE contract for covariance-aware geometric decoders."""

    def __call__(
        self,
        pb:            PriorBank,
        mu_q:          torch.Tensor,
        sigma_q:       torch.Tensor,
        targets:       torch.Tensor,

        *,
        z_loss_weight: float           = 0.0,
        tau:           Optional[float] = None,
        chunk_size:    Optional[int]   = None,
        ignore_index:  int             = -100,
    ) -> torch.Tensor:
        ...


class FrameAwareGeometricFusedCECallable(Protocol):
    """Geometric fused CE contract for the projected full-covariance decode boundary."""

    def __call__(
        self,
        pb:              PriorBank,
        mu_q:            torch.Tensor,
        sigma_q:         torch.Tensor,
        targets:         torch.Tensor,

        *,
        z_loss_weight:   float                           = 0.0,
        tau:             Optional[float]                 = None,
        chunk_size:      Optional[int]                   = None,
        ignore_index:    int                             = -100,
        canonical_frame: Optional[CanonicalFrameContext] = None,
    ) -> torch.Tensor:
        ...


class LinearFusedCECallable(Protocol):
    """Fused CE contract for the mean-only linear decoder."""

    def __call__(
        self,
        pb:            PriorBank,
        mu_q:          torch.Tensor,
        targets:       torch.Tensor,

        *,
        z_loss_weight: float         = 0.0,
        chunk_size:    Optional[int] = None,
        ignore_index:  int           = -100,
    ) -> torch.Tensor:
        ...


FusedCECallable = (
    GeometricFusedCECallable
    | FrameAwareGeometricFusedCECallable
    | LinearFusedCECallable
)


def _encode_prior_sigma(
    pb:        PriorBank,
    token_ids: torch.Tensor,             # (B, N) integer token ids
) -> torch.Tensor:
    """Look up the configured belief prior covariance without discarding model-channel rank."""
    log_diag = pb._prior_sigma_log_table()[token_ids]                    # (B, N, K)
    if pb.diagonal_covariance:
        return bounded_variance_from_log(log_diag, eps=pb.eps)
    if pb.prior_source == "model_channel":
        return covariance_from_packed(
            log_diag,
            pb.s_sigma_lower_embed[token_ids],
            eps=pb.eps,
        )                                                                # (B, N, K, K)
    return torch.diag_embed(bounded_variance_from_log(log_diag, eps=pb.eps))


@register_encode("per_token", can_omit_base_mean=True, can_omit_base_variance=True)
def _encode_per_token(
    pb:        PriorBank,
    token_ids: torch.Tensor,             # (B, N) integer token ids
) -> BeliefState:
    r"""Per-token table lookup: token_ids -> (mu_v, sigma_v, phi_v) as the belief q = p.

    Diagonal family: sigma is the (B, N, K) variance vector. A full token-table prior promotes its
    diagonal variances to (B, N, K, K). A full ``model_channel`` prior reconstructs the complete
    packed s covariance, so the s-to-p route preserves learned correlations. The mean and gauge
    tables are shared across families.
    """
    mu = pb._prior_mu_table()[token_ids]                                     # (B, N, K) prior (s if model_channel)
    sigma = _encode_prior_sigma(pb, token_ids)                               # (B,N,K) or (B,N,K,K)
    phi = pb.phi_embed[token_ids]                                            # (B, N, n_gen)
    omega = pb._omega_lookup(token_ids) if getattr(pb, "gauge_parameterization", "phi") == "omega_direct" else None
    return BeliefState(mu=mu, sigma=sigma, phi=phi, omega=omega)


@register_encode("canonical_content_gauge", can_omit_base_mean=True, can_omit_base_variance=True)
def _encode_canonical_content_gauge(
    pb:        PriorBank,
    token_ids: torch.Tensor,             # (B, N) integer token ids
) -> BeliefState:
    r"""Exact frame-intrinsic control: the existing tables are canonical ``(a_v, s_v, phi_v)``."""
    return _encode_per_token(pb, token_ids)


@register_encode("canonical_content_projected", can_omit_base_mean=True, can_omit_base_variance=True)
def _encode_canonical_content_projected(
    pb:        PriorBank,
    token_ids: torch.Tensor,             # (B, N) integer token ids
) -> BeliefState:
    r"""Return canonical table coordinates for model-owned realized-frame materialization."""
    return _encode_per_token(pb, token_ids)


@register_encode("per_token_additive", can_omit_base_mean=True, can_omit_base_variance=True)
def _encode_per_token_additive(
    pb:        PriorBank,
    token_ids: torch.Tensor,             # (B, N) integer token ids
) -> BeliefState:
    r"""Arm-2 control: the SAME learned (V, n_gen) phi table used NON-structurally.

    Each token's phi code is mapped by the FROZEN readout ``pb.additive_R`` (K, n_gen) to an additive
    mean shift ``mu += phi @ R^T``, and the returned phi is ZERO so the transport
    ``Omega = exp(phi.G) exp(-phi.G) = I`` (no gl(g) congruence). The learned parameter count is the
    gauge cell's (``V*n_gen`` in ``phi_embed``; ``R`` is a frozen buffer), so this isolates raw phi-table
    CAPACITY from the gl(g) generator STRUCTURE -- the capacity-vs-structure control for the blocks_K48
    REMAND (docs/2026-07-05-blocks-k48-followup-experiment-spec.md, Arm 2a). Deliberately NOT gauge
    equivariant; use with ``transport_mode='flat'`` and ``pos_phi='none'`` so no other channel transports.
    """
    mu = pb._prior_mu_table()[token_ids]                                     # (B, N, K) prior (s if model_channel)
    sigma = _encode_prior_sigma(pb, token_ids)                               # (B,N,K) or (B,N,K,K)
    phi_code = pb.phi_embed[token_ids]                                       # (B, N, n_gen) learned table
    mu = mu + phi_code @ pb.additive_R.t()                                   # (B, N, K) structure-free shift
    phi = torch.zeros_like(phi_code)                                         # Omega = I: no gl(g) transport
    return BeliefState(mu=mu, sigma=sigma, phi=phi)


@register_encode("gauge_fixed")
def _encode_gauge_fixed(
    pb:        PriorBank,
    token_ids: torch.Tensor,             # (B, N) integer token ids
) -> BeliefState:
    r"""NAMED STUB: gauge-fixed encode (gauge orbit from a shared base belief).

    Deferred: would realize every prior as a gauge transform of one shared base
    belief, so the vocabulary varies only along the gauge orbit. Not yet implemented.
    """
    raise NotImplementedError(
        "encode_mode='gauge_fixed' is a named stub (gauge orbit from a shared base); "
        "use 'per_token'."
    )


@register_decode("diagonal", can_omit_base_mean=True, can_omit_base_variance=True)
def _decode_diagonal(
    pb:      PriorBank,
    mu_q:    torch.Tensor,               # (B, N, K) posterior means
    sigma_q: torch.Tensor,               # (B, N, K) posterior variances
    tau_eff: torch.Tensor,               # () effective temperature
) -> torch.Tensor:                       # (B, N, V) logits = -KL(q || pi_v)/tau_eff
    r"""Exact diagonal -KL/tau_eff in closed form via a single fused matmul.

        KL = 0.5[ sum_k(sigma_q/sigma_v + (mu_q-mu_v)^2/sigma_v) - K + sum_k log(sigma_v/sigma_q) ]
    The v-dependent part A_v expands the Mahalanobis/trace terms into one matmul:
        lhs = [sigma_q + mc_q^2, -2 mc_q]            (B, N, 2K)
        rhs = [1/sigma_v,        mc_v/sigma_v]       (V, 2K)
        A_v = lhs @ rhs^T + sum_k(mc_v^2/sigma_v + log sigma_v)
            == sum_k(sigma_q/sigma_v + (mc_q-mc_v)^2/sigma_v) + sum_k log sigma_v
            == 2 KL + K + sum_k log sigma_q.
    The per-position (-K - sum_k log sigma_q) is v-INDEPENDENT (drops under softmax) but
    is KEPT so logits == -KL/tau_eff EXACTLY.

    NUMERICS: the Mahalanobis term ``(mu_q - mu_v)^2`` is reconstructed by the matmul as
    ``mc_q^2 - 2 mc_q mc_v + mc_v^2``, a subtraction of large near-equal quantities that
    catastrophically cancels in float32 once the means carry a large common offset (the
    error grows like eps * mu^2 / sigma_v and breaks the atol-1e-3 seam pin at modest
    |mu| / tight sigma_v). We remove the common offset BEFORE the matmul by subtracting
    the v-independent shift ``c = mean_v(mu_v)`` (per dim) from both means; since
    ``(mu_q - c) - (mu_v - c) == mu_q - mu_v`` the closed form is unchanged exactly while
    the canceled magnitude collapses to the residual spread of the means.
    """
    sigma_v = bounded_variance_from_log(pb._decode_sigma_log_table(), eps=pb.eps)  # (V, K)
    mu_v = pb._decode_mu_table()                                        # (V, K) decode table (untied if set)
    inv_v = 1.0 / sigma_v                                               # (V, K) = 1/sigma_v

    c = mu_v.mean(dim=0, keepdim=True)                                  # (1, K) v-independent shift
    mc_v = mu_v - c                                                     # (V, K) centered prior means
    mc_q = mu_q - c                                                     # (B, N, K) centered query means

    a_v = _decode_av(                                                    # (B, N, V)
        sigma_q, mc_q, mc_v, inv_v, torch.log(sigma_v).sum(-1))
    # == sum_k[(sigma_q + mc_q^2 - 2 mc_q mc_v)/sigma_v] + sum_k(mc_v^2/sigma_v + log sigma_v)
    # under the default expanded form; see _decode_av for the "exact" alternative (F32).
    # a_v == sum_k(sigma_q/sigma_v + (mc_q-mc_v)^2/sigma_v) + sum_k log sigma_v
    #     == sum_k(sigma_q/sigma_v + (mu_q-mu_v)^2/sigma_v) + sum_k log sigma_v = 2 KL + K + sum_k log sigma_q
    per_pos = pb.K + torch.log(sigma_q.clamp(min=pb.eps)).sum(-1, keepdim=True)   # (B, N, 1) = K + sum_k log sigma_q
    # .to() after the clamp keeps the fp64 a_v island open through the subtraction (F32); it is an
    # identity no-op under the default fp32 policy.
    evidence_delta = None
    if hasattr(pb, "head_evidence_logits"):
        _, coord_delta = pb._head_evidence_deltas(dtype=mu_q.dtype, device=mu_q.device)
        delta_per_pos = (
            coord_delta * (1.0 + torch.log(sigma_q.clamp(min=pb.eps)))
        ).sum(-1, keepdim=True)
        evidence_delta = _decode_head_evidence_kl_delta(
            sigma_q, mc_q, mc_v, inv_v, torch.log(sigma_v), coord_delta, delta_per_pos)
    return _decode_analytic_kl_logits(
        a_v, per_pos, tau_eff, mu_q.dtype, evidence_delta=evidence_delta)


@register_decode(
    "diagonal_chunked",
    supports_chunked=True,
    fused_ce=PriorBank.decode_ce_diagonal_chunked,
    can_omit_base_mean=True,
    can_omit_base_variance=True,
)
def _decode_diagonal_chunked(
    pb:      PriorBank,
    mu_q:    torch.Tensor,               # (B, N, K) posterior means
    sigma_q: torch.Tensor,               # (B, N, K) posterior variances
    tau_eff: torch.Tensor,               # () effective temperature
) -> torch.Tensor:                       # (B, N, V) logits = -KL(q || pi_v)/tau_eff
    r"""Inference (targets=None) decode for ``decode_mode='diagonal_chunked'``: full diagonal logits.

    The chunked mode's training memory win is the FUSED decode+CE in ``decode_ce_diagonal_chunked``
    (it never forms ``(B, N, V)``). When ``decode`` is called for logits (sampling / generation /
    inference), correctness is what matters, so this delegates to the exact ``diagonal`` kernel --
    the returned logits are byte-identical to ``decode_mode='diagonal'``.
    """
    return _decode_diagonal(pb, mu_q, sigma_q, tau_eff)


@register_decode(
    "full",
    supports_full=True,
    can_omit_base_mean=True,
    can_omit_base_variance=True,
)
def _decode_full(
    pb:      PriorBank,
    mu_q:    torch.Tensor,               # (B, N, K) posterior means
    sigma_q: torch.Tensor,               # (B, N, K, K) posterior covariances
    tau_eff: torch.Tensor,               # () effective temperature
) -> torch.Tensor:                       # (B, N, V) logits = -KL(q || pi_v)/tau_eff
    r"""Exact full-covariance decode logits_{i,v} = -KL(q_i || pi_v)/tau_eff via Cholesky.

    Scores the full-covariance posterior q_i = N(mu_q, Sigma_q) against every vocab prior
    pi_v through the ``gaussian_full`` divergence seam (Cholesky KL). The prior table is
    diagonal (sigma_log_embed), embedded as a diagonal full covariance diag(exp(sigma_log_v))
    so a full q is scored against it. As in ``reference_decode`` the seam is invoked with
    ``kl_max=inf`` so the full KL ranking over the vocabulary is preserved (decode must not
    saturate distant priors to a single logit). General but O(B*N*V*K^3) (per-pair Cholesky):
    the theoretically pure full-covariance path, not the fast diagonal kernel.
    """
    mu_v = pb._decode_mu_table()                                         # (V, K) decode table (untied if set)
    sigma_v = torch.diag_embed(
        bounded_variance_from_log(pb._decode_sigma_log_table(), eps=pb.eps)
    )                                                                                       # (V, K, K) diagonal-as-full
    mu_q_b = mu_q.unsqueeze(-2)                                          # (B, N, 1, K)
    sigma_q_b = sigma_q.unsqueeze(-3)                                    # (B, N, 1, K, K)
    full = get_family("gaussian_full")
    kl_v = kl(
        full(mu_q_b, sigma_q_b),
        full(mu_v, sigma_v),
        kl_max=float("inf"),
        eps=pb.eps,
    )                                                                       # (B, N, V)
    diag_sq, _, spd_ok = pb._full_cov_query_invariants(sigma_q)              # (B,N,K), _, (B,N)
    if hasattr(pb, "head_evidence_logits"):
        head_delta, coord_delta = pb._head_evidence_deltas(
            dtype=mu_q.dtype, device=mu_q.device)
        delta_per_pos, block_ok = pb._head_evidence_full_marginal_invariants(
            sigma_q, head_delta)
        spd_ok = spd_ok & block_ok
        c = mu_v.mean(dim=0, keepdim=True)
        mc_v = mu_v - c
        mc_q = mu_q - c
        inv_v = 1.0 / torch.diagonal(sigma_v, dim1=-2, dim2=-1)
        kl_v = kl_v + _decode_head_evidence_kl_delta(
            diag_sq, mc_q, mc_v, inv_v, torch.log(1.0 / inv_v), coord_delta,
            delta_per_pos)
    # Same exclusion contract as the chunked twin (audit 2026-08-06 F31). Here the -inf arrived
    # indirectly -- the family seam maps a failed Cholesky to NaN and ``kl_max=inf`` maps that to
    # inf -- so the row has to be neutralized after the fact. One extra (B, N) Cholesky against this
    # kernel's own O(B*N*V*K^3) per-pair factorization is not measurable.
    kl_v = torch.where(spd_ok.unsqueeze(-1), kl_v, torch.zeros_like(kl_v))
    return (-kl_v / tau_eff).to(mu_q.dtype)


@register_decode(
    "full_chunked",
    supports_full=True,
    supports_chunked=True,
    fused_ce=PriorBank.decode_ce_full_chunked,
    can_omit_base_mean=True,
    can_omit_base_variance=True,
)
def _decode_full_chunked(
    pb:      PriorBank,
    mu_q:    torch.Tensor,               # (B, N, K) posterior means
    sigma_q: torch.Tensor,               # (B, N, K, K) posterior covariances
    tau_eff: torch.Tensor,               # () effective temperature
) -> torch.Tensor:                       # (B, N, V) logits = -KL(q || pi_v)/tau_eff
    r"""Inference (targets=None) decode for ``decode_mode='full_chunked'``: full-cov KL logits via
    the DIAGONAL-prior closed form -- NO per-pair (K, K) Cholesky.

    The training memory win is the fused decode+CE in ``decode_ce_full_chunked`` (never forms
    (B, N, V)); for logits (sampling / generation) this materializes (B, N, V) -- inherent to
    producing every vocab logit -- but still avoids the (B, N, V, K, K) Cholesky/solve workspace
    that ``_decode_full`` builds, by exploiting the diagonal prior (see ``_full_cov_query_invariants``).
    Value-equal to ``decode_mode='full'`` to atol-1e-3 (tests/test_fullcov_alpha_roadmap_2026_06_13.py).
    """
    sigma_v = bounded_variance_from_log(pb._decode_sigma_log_table(), eps=pb.eps)  # (V, K) diagonal decode variances
    mu_v = pb._decode_mu_table()                                         # (V, K) decode table (untied if set)
    inv_v = 1.0 / sigma_v                                                # (V, K) = 1/sigma_v

    diag_sq, logdet_q, spd_ok = pb._full_cov_query_invariants(sigma_q)   # (B,N,K), (B,N), (B,N)
    c = mu_v.mean(dim=0, keepdim=True)                                   # (1, K) v-independent shift
    mc_v = mu_v - c                                                      # (V, K) centered prior means
    mc_q = mu_q - c                                                      # (B, N, K) centered query means

    a_v = _decode_av(                                                    # (B, N, V) trace + mahalanobis
        diag_sq, mc_q, mc_v, inv_v, torch.log(sigma_v).sum(-1))
    per_pos = pb.K + logdet_q.unsqueeze(-1)                              # (B, N, 1) = K + log|Sigma_q|
    # .to() after the clamp keeps the fp64 a_v island open through the subtraction (F32); it is an
    # identity no-op under the default fp32 policy.
    evidence_delta = None
    if hasattr(pb, "head_evidence_logits"):
        head_delta, coord_delta = pb._head_evidence_deltas(
            dtype=mu_q.dtype, device=mu_q.device)
        delta_per_pos, block_ok = pb._head_evidence_full_marginal_invariants(
            sigma_q, head_delta)
        spd_ok = spd_ok & block_ok
        evidence_delta = _decode_head_evidence_kl_delta(
            diag_sq, mc_q, mc_v, inv_v, torch.log(sigma_v), coord_delta, delta_per_pos)
    logits = _decode_analytic_kl_logits(
        a_v, per_pos, tau_eff, mu_q.dtype, evidence_delta=evidence_delta)
    # A non-PD Sigma_q has no valid likelihood: emit an INFORMATIONLESS uniform row rather than a
    # score built on a placeholder log-det (audit 2026-08-06 F31). log_softmax of a uniform row is
    # -log V, finite; the all--inf row this used to produce maps to NaN. The CE seams exclude the
    # position outright -- see decode_ce_full_chunked and decode_degenerate_positions.
    return torch.where(spd_ok.unsqueeze(-1), logits, torch.zeros_like(logits))


def _family_logits(
    pb:      PriorBank,
    mu_q:    torch.Tensor,               # (B, N, K) posterior means
    sigma_q: torch.Tensor,               # (B, N, K) or (B, N, K, K) posterior (co)variances
    tau_eff: torch.Tensor,               # () effective temperature

    *,
    chunk:   Optional[int],              # vocab-slice width; None -> one slice (the dense kernel)
) -> torch.Tensor:                       # (B, N, V) logits = -D_configured(q || pi_v)/tau_eff
    r"""Shared body of the ``family`` / ``family_chunked`` LOGITS kernels (audit 2026-08-07).

    ``chunk`` is the only difference between the two modes, so they share one implementation and
    cannot drift. Slicing the vocabulary bounds the live functional workspace to (B, N, chunk[, K, K])
    instead of (B, N, V[, K, K]); the concatenated result is value-identical because the functional
    is applied elementwise over the vocabulary axis (each pi_v is scored independently).

    Degenerate positions are handled exactly as ``_decode_full_chunked`` handles them (audit
    2026-08-06 F31): a non-PD Sigma_q has no valid likelihood, so it is scored on a substituted
    benign covariance and then overwritten with an INFORMATIONLESS uniform row, rather than emitting
    the NaN that ``kl_max=inf`` would otherwise produce and that would propagate through any
    downstream ``log_softmax`` or sampling step.
    """
    family_cls = get_family(pb.family)
    is_full = family_cls.cov_kind == "full"
    functional = get_functional(pb.divergence_family)
    V = pb.vocab_size

    degenerate = pb.decode_degenerate_positions(sigma_q)                    # (B, N) bool, or None
    if degenerate is not None:
        eye = torch.eye(sigma_q.shape[-1], device=sigma_q.device, dtype=sigma_q.dtype)
        sigma_q = torch.where(degenerate[..., None, None], eye.expand_as(sigma_q), sigma_q)

    q = family_cls(mu_q.unsqueeze(-2), sigma_q.unsqueeze(-3 if is_full else -2))
    p_sigma_all = bounded_variance_from_log(
        pb._decode_sigma_log_table(), eps=pb.eps
    )                                                                       # (V, K) diagonal prior variances
    mu_v_all = pb._decode_mu_table()                                        # (V, K) decode table (untied if set)

    if chunk is None:
        width = V
    else:
        requested = max(1, min(int(chunk), V))
        inner = pb.K * pb.K if is_full else 1
        workspace_bytes_per_scalar = _full_family_workspace_bytes_per_scalar(
            family_cls, mu_q, mu_q, sigma_q,
        )
        width = _decode_ce_family_effective_chunk(
            mu_q, requested, inner, workspace_bytes_per_scalar=workspace_bytes_per_scalar,
        )
    slices = []
    for v0 in range(0, V, width):
        v1 = min(v0 + width, V)
        p_sigma_c = p_sigma_all[v0:v1]                                      # (Vc, K)
        p = family_cls(mu_v_all[v0:v1],
                       torch.diag_embed(p_sigma_c) if is_full else p_sigma_c)
        energy = functional(q, p, alpha=pb.renyi_order,
                            kl_max=float("inf"), eps=pb.eps)                # (B, N, Vc)
        slices.append(-energy / tau_eff)
    logits = slices[0] if len(slices) == 1 else torch.cat(slices, dim=-1)   # (B, N, V)

    if degenerate is not None:
        logits = torch.where(degenerate.unsqueeze(-1), torch.zeros_like(logits), logits)
    return logits


@register_decode(
    "family",
    covariance_kinds=frozenset({"diagonal", "full"}),
    family_consistent=True,
    can_omit_base_mean=True,
    can_omit_base_variance=True,
)
def _decode_family(
    pb:      PriorBank,
    mu_q:    torch.Tensor,               # (B, N, K) posterior means
    sigma_q: torch.Tensor,               # (B, N, K) or (B, N, K, K) posterior (co)variances
    tau_eff: torch.Tensor,               # () effective temperature
) -> torch.Tensor:                       # (B, N, V) logits = -D_configured(q || pi_v)/tau_eff
    r"""Family/divergence-consistent decode (PB-14): logits = -D_configured(q_i || pi_v)/tau_eff.

    Scores the posterior q_i against every vocabulary prior pi_v through the CONFIGURED belief family
    ``pb.family`` and divergence functional ``pb.divergence_family`` at ``alpha=pb.renyi_order``, so
    the readout matches the E-step geometry rather than a hardcoded gaussian alpha=1 KL. As in the
    other decode kernels the seam is invoked with ``kl_max=inf`` (a DECODE must preserve the full
    divergence ranking over the vocabulary). The vocabulary prior table is intentionally DIAGONAL in
    every family (PB-11); a full family promotes it with ``diag_embed`` so a full q is scored against
    a diagonal-as-full prior. Broadcasting the functional over V materializes a (B, N, V) energy
    (a full family a (B, N, V, K, K) workspace): general but O(B*N*V*...); the training memory win is
    the fused CE twin ``decode_ce_family_chunked``. For a canonical gaussian + renyi + alpha=1 config
    this equals the fast ``diagonal``/``full`` kernels (and ``reference_decode`` is pinned to it)."""
    return _family_logits(pb, mu_q, sigma_q, tau_eff, chunk=None)


@register_decode(
    "family_chunked",
    covariance_kinds=frozenset({"diagonal", "full"}),
    family_consistent=True,
    supports_chunked=True,
    fused_ce=PriorBank.decode_ce_family_chunked,
    can_omit_base_mean=True,
    can_omit_base_variance=True,
)
def _decode_family_chunked(
    pb:      PriorBank,
    mu_q:    torch.Tensor,               # (B, N, K) posterior means
    sigma_q: torch.Tensor,               # (B, N, K) or (B, N, K, K) posterior (co)variances
    tau_eff: torch.Tensor,               # () effective temperature
) -> torch.Tensor:                       # (B, N, V) logits = -D_configured(q || pi_v)/tau_eff
    r"""Inference (targets=None) decode for ``decode_mode='family_chunked'``: full family logits.

    The chunked mode's training memory win is the FUSED decode+CE in ``decode_ce_family_chunked``
    (it never forms (B, N, V)). Producing every vocab logit inherently materializes (B, N, V), but
    the FUNCTIONAL WORKSPACE behind it does not have to be materialized all at once, and for a full
    family that workspace is the (B, N, V, K, K) tensor -- K^2 times the output.

    Until audit 2026-08-07 this delegated to the unchunked ``_decode_family``, so ``decode_chunk_size``
    was accepted and silently ignored on this path: measured identical 49.15 MB workspace at both
    chunk=512 and chunk=8192, and a single 321.6 MB (1, 2, 50257, 20, 20) allocation at the live
    vocab/embed -- about 41 GB at N=256, against 8 MB for ``full_chunked``. The knob is now honoured
    here, which is what makes the mode usable at realistic N. Value-identical to ``decode_mode=
    'family'``: both call ``_family_logits``, differing only in the slice width.
    """
    if _uses_canonical_full_family_decode(pb, mu_q, sigma_q):
        return _decode_full_chunked(pb, mu_q, sigma_q, tau_eff)
    return _family_logits(pb, mu_q, sigma_q, tau_eff, chunk=pb.decode_chunk_size)


@register_decode(
    "expected_likelihood_chunked",
    supports_chunked=True,
    fused_ce=PriorBank.decode_ce_expected_likelihood_chunked,
    can_omit_base_mean=True,
    can_omit_base_variance=True,
)
def _decode_expected_likelihood_chunked(
    pb:      PriorBank,
    mu_q:    torch.Tensor,               # (B, N, K) posterior means
    sigma_q: torch.Tensor,               # (B, N, K) posterior variances
    tau_eff: torch.Tensor,               # () effective temperature
) -> torch.Tensor:                       # (B, N, V) expected-likelihood logits
    r"""Expected-likelihood decode: logits from the exact Gaussian-convolution marginal.

        E_{x~q_i}[N(x; mu_v, Sigma_v)] = N(mu_q_i; mu_v, Sigma_q_i + Sigma_v)
    (the Gaussian convolution identity: the expectation of one Gaussian density under another
    integrates to a Gaussian in the mean difference with the SUMMED covariances). Taking the log,
    dropping the v-independent constant -(K/2) log(2 pi), and tempering by tau_eff (diagonal
    family):

        logit_{i,v} = -1/(2 tau_eff) * sum_k [ (mu_q - mu_v)^2 / (sigma_q + sigma_v)
                                               + log(sigma_q + sigma_v) ].

    Unlike the -KL readout this scores q as an OBSERVATION model marginal (Bayes-exact up to the
    dropped constant), so a diffuse prior pi_v is penalized through log(sigma_q + sigma_v) rather
    than rewarded through the KL's 1/sigma_v flattening. DIAGONAL family only (registered without
    is_full, so the config rank cross-check pairs it with diagonal families by construction).
    Chunked over the vocabulary: the couplings sigma_q + sigma_v block the diagonal kernel's
    single-matmul trick, so each chunk broadcasts a (B, N, Vc, K) workspace; the (B, N, V) output
    is inherent to producing every logit (the training memory win is the fused CE twin
    ``decode_ce_expected_likelihood_chunked``).
    """
    sigma_v_all = bounded_variance_from_log(
        pb._decode_sigma_log_table(), eps=pb.eps,
    )                                                                         # (V, K)
    mu_v_all = pb._decode_mu_table()                                     # (V, K) decode table (untied if set)
    chunk = pb.decode_chunk_size
    V = pb.vocab_size

    logit_chunks = []
    for v0 in range(0, V, chunk):
        v1 = min(v0 + chunk, V)
        d = mu_q.unsqueeze(-2) - mu_v_all[v0:v1]                         # (B, N, Vc, K)
        s = sigma_q.unsqueeze(-2) + sigma_v_all[v0:v1]                   # (B, N, Vc, K) convolved variances
        logit_chunks.append(-0.5 * (d ** 2 / s + torch.log(s)).sum(-1) / tau_eff)  # (B, N, Vc)
    return torch.cat(logit_chunks, dim=-1)                               # (B, N, V)


@register_decode(
    "linear",
    supports_chunked=True,
    fused_ce=PriorBank.decode_ce_linear_chunked,
    can_omit_base_mean=True,
    can_omit_base_variance=True,
)
def _decode_linear(
    pb:      PriorBank,
    mu_q:    torch.Tensor,               # (B, N, K) posterior means
    sigma_q: torch.Tensor,               # (B, N, K) posterior variances (DISCARDED)
    tau_eff: torch.Tensor,               # () effective temperature (DISCARDED)
) -> torch.Tensor:                       # (B, N, V) logits = mu_q @ W^T (+ b)
    r"""Linear-projection decode (use_prior_bank=False): logits = mu_q @ W^T (+ b).

    The one authorized neural exception: a single learned (V, K) output weight applied to the
    converged mean, with NO KL geometry at the decode boundary (the decode temperature is discarded;
    only encode + the E-step remain gauge-aware). Realized as a raw nn.Parameter matmul, not an
    nn.Linear module. With ``decode_bias`` a learned per-vocab log-unigram bias ``b`` (V,) is added
    (see __init__). The pure KL-readout path is always available under use_prior_bank=True; this is
    the opt-in ablation the user uses to compare with/without the prior-bank decode.
    """
    x = mu_q                                                           # bare converged mean
    logits = x @ pb.output_proj_weight.transpose(-1, -2)               # (B, N, V)
    if pb.output_proj_bias is not None:
        logits = logits + pb.output_proj_bias                           # learned log-unigram prior
    return logits


# The evidence mixer has a deliberately narrower contract than the general decode registry: its
# canonical-KL interpretation is valid only for these import-time registrations. Keeping both the
# registration records and their callables pins an override=True alias/replacement fail-closed.
_HEAD_EVIDENCE_CANONICAL_DECODERS: Mapping[str, DecodeRegistration] = MappingProxyType({
    "diagonal": _DECODERS["diagonal"],
    "diagonal_chunked": _DECODERS["diagonal_chunked"],
    "full": _DECODERS["full"],
    "full_chunked": _DECODERS["full_chunked"],
})
_HEAD_EVIDENCE_CANONICAL_DECODER_CALLABLES: Mapping[str, DecodeCallable] = MappingProxyType({
    name: registration.callable
    for name, registration in _HEAD_EVIDENCE_CANONICAL_DECODERS.items()
})


def has_builtin_head_evidence_decoder(name: str) -> bool:
    """Whether *name* still resolves to its import-time canonical KL decoder exactly."""
    registration = _DECODERS.get(name)
    return (
        registration is _HEAD_EVIDENCE_CANONICAL_DECODERS.get(name)
        and registration is not None
        and registration.callable is _HEAD_EVIDENCE_CANONICAL_DECODER_CALLABLES[name]
    )


# Projected canonical content has a stricter boundary than the general registry: its diagonal query
# is pulled back to full covariance before these exact analytic scorers, and only full_chunked's
# import-time fused callable accepts the same-forward canonical_frame keyword. Pin the complete
# registrations plus both callable identities so an override cannot inherit this private contract by
# copying metadata or reusing only one of the built-in callables.
_PROJECTED_FULL_DECODERS: Mapping[str, DecodeRegistration] = MappingProxyType({
    "full": _HEAD_EVIDENCE_CANONICAL_DECODERS["full"],
    "full_chunked": _HEAD_EVIDENCE_CANONICAL_DECODERS["full_chunked"],
})
_PROJECTED_FULL_DECODER_CALLABLES: Mapping[str, DecodeCallable] = MappingProxyType({
    name: registration.callable
    for name, registration in _PROJECTED_FULL_DECODERS.items()
})
_PROJECTED_FULL_FUSED_CALLABLES: Mapping[str, Optional[FusedCECallable]] = MappingProxyType({
    name: registration.fused_ce
    for name, registration in _PROJECTED_FULL_DECODERS.items()
})


def has_builtin_projected_full_decoder(name: str) -> bool:
    """Whether *name* still owns the import-time projected full decode contract exactly."""
    registration = _DECODERS.get(name)
    builtin = _PROJECTED_FULL_DECODERS.get(name)
    return (
        registration is builtin
        and registration is not None
        and registration.callable is _PROJECTED_FULL_DECODER_CALLABLES[name]
        and registration.fused_ce is _PROJECTED_FULL_FUSED_CALLABLES[name]
    )
