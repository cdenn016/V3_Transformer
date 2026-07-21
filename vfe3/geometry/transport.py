r"""Gauge transport for VFE_3.0 (Regime I, Gaussian / location-scale specific).

Two parameterizations of the flat (Regime I) transport:
  phi (exp):    Omega_ij = exp(phi_i . G) exp(-phi_j . G) in GL+(K) (det>0).
  omega_direct: Omega_ij = Omega_i Omega_j^{-1} for general GL(K) (det may be <0).
Belief action: mu -> Omega @ mu, Sigma -> Omega @ Sigma @ Omega^T (sandwich;
diagonal-covariance fast path plus an exact full-covariance congruence). Regime II and retractions are separate modules;
gauge-RoPE folds a positional rotation into the transport via :class:`RopeTransport`.
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List, Mapping, Optional, Tuple

import torch
from torch import nn

from vfe3.families.base import _logdet_chol
from vfe3.geometry.groups import GaugeGroup
from vfe3.geometry.lie_ops import CompactBlockElement, _equal_diag_blocks
from vfe3.numerics import safe_cholesky

if TYPE_CHECKING:
    from vfe3.config import VFE3Config


@dataclass
class FactoredTransport:
    r"""The flat phi-cocycle transport in FACTORED form: the per-token (B, N, K, K) vertex
    exponentials, NOT the dense (B, N, N, K, K) pairwise Omega.

    On the flat + block-diagonal-with-equal-blocks path the dense Omega_ij = exp(phi_i) exp(-phi_j)
    is never materialized; instead ``transport_mean`` / ``transport_covariance`` consume this
    container on a fast path that fuses the exps into the contraction (P0 #2, perf doc
    docs/perf/2026-05-31-speedup-opportunities.md): the mean is an EXACT reassociation, and the
    DIAGONAL sandwich factors per head by block-diagonality (a (d, d) operation per head, never
    the full K x K square). A FULL-covariance input rebuilds the dense Omega from the factors
    (byte-identical to ``compute_transport_operators``), so the unfused sandwich is unchanged.

    ``irrep_dims`` (equal blocks, length > 1) drives the per-head slicing of the diagonal cov.

    ``mean_per_head`` (Tier-1 perf toggle, default False = byte-identical): when set,
    ``transport_mean`` contracts each gauge block separately (``_factored_per_head_mean``, the
    mean twin of the per-head diagonal cov) instead of the dense full-K einsum -- the same sum
    with the exactly-zero off-block terms dropped, equal to fp32 reassociation.
    """

    exp_phi:                 torch.Tensor  # (..., N, K, K) exp(phi_i . G)
    exp_neg_phi:             torch.Tensor  # (..., N, K, K) exp(-phi_j . G)
    irrep_dims:              List[int]     # equal block sizes; sum == K, len > 1
    mean_per_head:           bool = False  # transport_mean contracts per gauge block
    same_frame_flat_cocycle: bool = False  # one vertex table supplies U_i and true inverse U_j^-1

    def __post_init__(self) -> None:
        if type(self.same_frame_flat_cocycle) is not bool:
            raise ValueError(
                "same_frame_flat_cocycle must be a bool, got "
                f"{type(self.same_frame_flat_cocycle).__name__}: "
                f"{self.same_frame_flat_cocycle!r}")

    def to_dense_omega(self) -> torch.Tensor:
        r"""Rebuild the dense Omega_ij = exp(phi_i) exp(-phi_j) (..., N, N, K, K).

        Byte-identical to ``compute_transport_operators``'s Omega einsum (same factors, same
        ``ikl,jlm->ijkm`` contraction); used to keep the FULL-covariance sandwich and any
        consumer that needs the explicit operator on the existing dense code path. Rank-agnostic
        via the leading ellipsis (an optional batch axis flows through; the unbatched call matches).
        """
        return torch.einsum("...ikl,...jlm->...ijkm", self.exp_phi, self.exp_neg_phi)


@dataclass
class DirectLinkTransport:
    r"""Direct edge transport with an optional live vertex chart, never a dense pairwise product.

    ``exp_link`` is the batch-independent edge factor ``L_ij`` with shape ``(N, N, K, K)``.
    The bare link stores no vertex factors. The charted link stores ``exp_phi_i`` and
    ``exp_neg_phi_j`` so consumers contract

        Omega_ij = exp_phi_i L_ij exp_neg_phi_j

    directly into means and covariances. :meth:`to_dense_omega` is the explicit compatibility
    boundary for diagnostics and legacy registry callers that genuinely require ``Omega``.
    """

    exp_link:     torch.Tensor
    exp_phi:      Optional[torch.Tensor] = None
    exp_neg_phi:  Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        if self.exp_link.dim() != 4 or self.exp_link.shape[-1] != self.exp_link.shape[-2]:
            raise ValueError(
                "direct-link edge factors must have (Nq, Nk, K, K) square-matrix layout, got "
                f"{tuple(self.exp_link.shape)}")
        if (self.exp_phi is None) != (self.exp_neg_phi is None):
            raise ValueError("charted direct-link transport requires both vertex factors or neither")
        if self.exp_phi is None:
            return
        compatible = (
            self.exp_phi.dim() >= 3
            and self.exp_neg_phi.dim() == self.exp_phi.dim()
            and self.exp_phi.shape[:-3] == self.exp_neg_phi.shape[:-3]
            and self.exp_phi.shape[-2] == self.exp_phi.shape[-1]
            and self.exp_neg_phi.shape[-2] == self.exp_neg_phi.shape[-1]
            and self.exp_phi.shape[-1] == self.exp_link.shape[-1]
            and self.exp_neg_phi.shape[-1] == self.exp_link.shape[-1]
            and self.exp_phi.shape[-3] == self.exp_link.shape[-4]
            and self.exp_neg_phi.shape[-3] == self.exp_link.shape[-3]
        )
        if not compatible:
            raise ValueError(
                "charted direct-link vertex factors must match the edge token and matrix axes, got "
                f"link={tuple(self.exp_link.shape)}, exp_phi={tuple(self.exp_phi.shape)}, "
                f"exp_neg_phi={tuple(self.exp_neg_phi.shape)}")

    def to_dense_omega(self) -> torch.Tensor:
        r"""Materialize ``Omega_ij`` only for an explicit compatibility consumer."""
        if self.exp_phi is None:
            return self.exp_link
        return torch.einsum(
            "...ikl,ijlm,...jmn->...ijkn", self.exp_phi, self.exp_link, self.exp_neg_phi)


TransportDict = Dict[str, 'torch.Tensor | DirectLinkTransport | None']


@dataclass
class CompactFactoredTransport:
    r"""Flat equal-block transport whose vertex factors stay ``(..., N, H, d, d)``.

    ``exp_blocks`` and ``inv_blocks`` are the stored vertex elements and their true per-block
    inverses. Mean, diagonal-covariance, and full-covariance contractions consume these factors
    without forming a vertex or pairwise ``K x K`` matrix. :meth:`to_dense_omega` is the explicit
    compatibility boundary for legacy consumers that require a dense pairwise operator.
    """

    exp_blocks:              torch.Tensor  # (..., N, H, d, d) stored U_i blocks
    inv_blocks:              torch.Tensor  # (..., N, H, d, d) true U_j^{-1} blocks
    K:                       int
    mean_per_head:           bool = False
    same_frame_flat_cocycle: bool = False

    def __post_init__(self) -> None:
        compatible = (
            self.exp_blocks.dim() == self.inv_blocks.dim()
            and self.exp_blocks.shape[:-4] == self.inv_blocks.shape[:-4]
            and self.exp_blocks.shape[-3:] == self.inv_blocks.shape[-3:]
        )
        if not compatible:
            raise ValueError(
                "compact transport factors must share leading batch and trailing block shapes "
                "(their query/key token counts may differ), got "
                f"{tuple(self.exp_blocks.shape)} and {tuple(self.inv_blocks.shape)}")
        if self.exp_blocks.dim() < 4 or self.exp_blocks.shape[-1] != self.exp_blocks.shape[-2]:
            raise ValueError(
                "compact transport factors must have (..., N, H, d, d) square-block layout, got "
                f"{tuple(self.exp_blocks.shape)}")
        H, d = self.exp_blocks.shape[-3], self.exp_blocks.shape[-1]
        if type(self.K) is not int or self.K <= 0 or H * d != self.K:
            raise ValueError(f"compact transport has H={H}, d={d}, but H*d != K={self.K!r}")
        if type(self.mean_per_head) is not bool:
            raise ValueError(
                "mean_per_head must be a bool, got "
                f"{type(self.mean_per_head).__name__}: {self.mean_per_head!r}")
        if type(self.same_frame_flat_cocycle) is not bool:
            raise ValueError(
                "same_frame_flat_cocycle must be a bool, got "
                f"{type(self.same_frame_flat_cocycle).__name__}: "
                f"{self.same_frame_flat_cocycle!r}")

    @property
    def n_blocks(self) -> int:
        return self.exp_blocks.shape[-3]

    @property
    def block_dim(self) -> int:
        return self.exp_blocks.shape[-1]

    @property
    def irrep_dims(self) -> List[int]:
        return [self.block_dim] * self.n_blocks

    @property
    def device(self) -> torch.device:
        return self.exp_blocks.device

    @property
    def dtype(self) -> torch.dtype:
        return self.exp_blocks.dtype

    def unsqueeze(self, dim: int) -> 'CompactFactoredTransport':
        r"""Insert a leading batch axis without materializing either pairwise or dense matrices."""
        n_leading = self.exp_blocks.dim() - 4
        logical_rank = n_leading + 4                    # conceptual (..., Nq, Nk, K, K) transport
        normalized = dim if dim >= 0 else dim + logical_rank + 1
        if normalized < 0 or normalized > n_leading:
            raise ValueError(
                "CompactFactoredTransport.unsqueeze may only add a leading axis before "
                "the conceptual (Nq, Nk, K, K) transport axes")
        return CompactFactoredTransport(
            self.exp_blocks.unsqueeze(normalized),
            self.inv_blocks.unsqueeze(normalized),
            self.K,
            mean_per_head=self.mean_per_head,
            same_frame_flat_cocycle=self.same_frame_flat_cocycle,
        )

    def to_dense_omega(self) -> torch.Tensor:
        r"""Explicit compatibility conversion to dense ``(..., N, N, K, K)`` transport."""
        exp_dense = CompactBlockElement(self.exp_blocks, self.K).to_dense()
        inv_dense = CompactBlockElement(self.inv_blocks, self.K).to_dense()
        return torch.einsum("...ikl,...jlm->...ijkm", exp_dense, inv_dense)


@dataclass
class RopeTransport:
    r"""A built transport wrapped with a gauge-RoPE positional rotation R(theta).

    ``base`` is the un-rotated transport (a dense (N,N,K,K) Omega OR a FactoredTransport). The
    effective operator is Omega^RoPE_ij = R(theta_i) Omega_ij R(theta_j)^T. ``transport_mean``
    always applies the rotation; ``transport_covariance`` applies it only when ``on_cov`` (the
    means+covariance "full-gauge" regime, which the config gates to full covariance). Means-only
    (``on_cov=False``) leaves the covariance sandwich on the un-rotated ``base`` -- numerically
    identical to no RoPE for the covariance tensor itself, so the diagonal-covariance path stays
    valid. NOTE: under means-only the mean transports under R_i Omega_ij R_j^T but the covariance
    under the bare Omega_ij, so the transported (mu_t, Sigma_t) is NOT a single coherent congruence
    image -- affine/Mahalanobis invariants (e.g. mu^T Sigma^{-1} mu, norms.MahalanobisNorm) are not
    preserved for that belief. The coherent pure path is ``on_cov=True`` (rope_full_gauge), where
    both transform under the same rotated operator.

    ``on_value`` (default True = the coherent single-gauge path) factors the transport into an
    ATTENTION gauge and a VALUE gauge (GL(K)_attention.tex:1909): the attention score
    D_KL(q_i || R_i Omega_ij R_j^T q_j) always carries the rotation, but with ``on_value=False`` the
    value aggregation mu_hat_i = sum_j beta_ij Omega_ij mu_j uses the UN-rotated base transport --
    exactly RoPE's "position-dependent attention, position-independent values" asymmetry. The flag is
    consumed at the GRADIENT layer: ``gradients/oracle.py`` builds the value-gauge coupling energy from
    ``base`` while beta comes from the rotated score energy, and ``gradients/kernels.py`` routes the
    decoupled case to the oracle (beta is no longer the coupling sum's stationary point, so the
    closed-form envelope kernel does not apply). ``transport_mean`` / ``transport_covariance`` always
    honor the rotation (the score path); the value path transports on ``base`` directly.
    """

    base:                    'torch.Tensor | DirectLinkTransport | FactoredTransport | CompactFactoredTransport'
    rope:                    torch.Tensor  # (N, K, K) block-diagonal orthogonal rotation
    on_cov:                  bool = False
    on_value:                bool = True   # False -> value aggregation uses the UN-rotated base (RoPE Q/K only)
    same_frame_flat_cocycle: bool = False  # trusted same-table RoPE around an already certified base

    def __post_init__(self) -> None:
        if type(self.same_frame_flat_cocycle) is not bool:
            raise ValueError(
                "same_frame_flat_cocycle must be a bool, got "
                f"{type(self.same_frame_flat_cocycle).__name__}: "
                f"{self.same_frame_flat_cocycle!r}")
        if isinstance(self.base, DirectLinkTransport):
            if self.base.exp_phi is None:
                n_query = self.base.exp_link.shape[-4]
                n_key = self.base.exp_link.shape[-3]
            else:
                n_query = self.base.exp_phi.shape[-3]
                n_key = self.base.exp_neg_phi.shape[-3]
            K = self.base.exp_link.shape[-1]
        elif isinstance(self.base, CompactFactoredTransport):
            n_query = self.base.exp_blocks.shape[-4]
            n_key = self.base.inv_blocks.shape[-4]
            K = self.base.K
        elif isinstance(self.base, FactoredTransport):
            if self.base.exp_phi.dim() < 3 or self.base.exp_neg_phi.dim() < 3:
                raise ValueError("RopeTransport factored base must have (..., N, K, K) factors")
            if (self.base.exp_phi.shape[-2] != self.base.exp_phi.shape[-1]
                    or self.base.exp_neg_phi.shape[-2] != self.base.exp_neg_phi.shape[-1]):
                raise ValueError(
                    "RopeTransport factored base factors must each end in square K x K matrix "
                    f"axes; got {tuple(self.base.exp_phi.shape)} and "
                    f"{tuple(self.base.exp_neg_phi.shape)}")
            if self.base.exp_phi.shape[-1] != self.base.exp_neg_phi.shape[-1]:
                raise ValueError(
                    "RopeTransport factored base factors must use the same K; got "
                    f"Kq={self.base.exp_phi.shape[-1]}, Kk={self.base.exp_neg_phi.shape[-1]}")
            if self.base.exp_phi.shape[:-3] != self.base.exp_neg_phi.shape[:-3]:
                raise ValueError(
                    "RopeTransport factored base factors must have matching leading batch shapes; "
                    f"got {tuple(self.base.exp_phi.shape[:-3])} and "
                    f"{tuple(self.base.exp_neg_phi.shape[:-3])}")
            n_query = self.base.exp_phi.shape[-3]
            n_key = self.base.exp_neg_phi.shape[-3]
            K = self.base.exp_phi.shape[-1]
        else:
            if self.base.dim() < 4:
                raise ValueError("RopeTransport dense base must have (..., Nq, Nk, K, K) layout")
            if self.base.shape[-2] != self.base.shape[-1]:
                raise ValueError(
                    "RopeTransport dense base must end in square K x K matrix axes; got "
                    f"shape {tuple(self.base.shape)}")
            n_query = self.base.shape[-4]
            n_key = self.base.shape[-3]
            K = self.base.shape[-1]
        if n_query != n_key:
            raise ValueError(
                "RopeTransport requires a square token transport because one rope tensor is "
                f"shared by query and key rotations; got Nq={n_query}, Nk={n_key}")
        if (self.rope.dim() < 3 or self.rope.shape[-1] != K
                or self.rope.shape[-2] != K):
            raise ValueError(
                f"RopeTransport rope must have (..., N, K, K) with K={K}, got "
                f"{tuple(self.rope.shape)}")
        if self.rope.shape[-3] != n_query:
            raise ValueError(
                "RopeTransport rope token length must match both transport token axes; "
                f"got rope N={self.rope.shape[-3]}, Nq=Nk={n_query}")


def _rope_dense_omega(
    base: 'torch.Tensor | DirectLinkTransport | FactoredTransport | CompactFactoredTransport',
    rope: torch.Tensor,
) -> torch.Tensor:
    r"""Effective dense Omega^RoPE_ij = R(theta_i) Omega_ij R(theta_j)^T (full-gauge / dense path)."""
    omega = (
        base.to_dense_omega()
        if isinstance(base, (DirectLinkTransport, FactoredTransport, CompactFactoredTransport)) else base
    )                                                                               # (...,N,N,K,K)
    # R_i Omega_ij R_j^T: contract R on the left of the i-axis output and the right (transposed) of j.
    rot = torch.einsum("...ikl,...ijlm,...jnm->...ijkn", rope, omega, rope)
    return rot


# -- connection-regime registry (orthogonal to the gauge_parameterization phi|omega_direct axis) --
# The CONNECTION REGIME (flat Regime I, the non-flat Regime II to come) is a registry-backed modular
# axis on equal footing with the structure group (clean-room spec sec 4.2): config-selected, added by
# writing-and-registering, never editing call sites. This is ORTHOGONAL to gauge_parameterization
# (phi|omega_direct), which chooses how a single flat transport is parameterized; the regime chooses
# whether the connection is flat at all. Both regimes are registered here: the flat phi-cocycle
# under 'flat' (:func:`_build_flat`, the no-NN pure default) and the non-flat edge-relaxed Regime II
# under 'regime_ii' (:func:`_build_regime_ii` below, the sanctioned default-OFF learned-connection
# exception; spec docs/superpowers/specs/2026-06-01-regime-ii-connection-design.md).
@dataclass(frozen=True)
class TransportRegistration:
    """A transport builder and every routing/reporting declaration attached to it."""

    callable:                   Callable[..., TransportDict]
    needs_mu:                   bool
    needs_sigma:                bool
    batch_independent:          bool
    covariance_class:           str
    state_builder:              'Optional[TransportStateBuilder]'
    serialization_keys:         Tuple[str, ...]
    offdiag_serialization_keys: Tuple[str, ...]


TransportState = Mapping[str, torch.Tensor]
TrainableTransportState = Dict[str, nn.Parameter]
TransportStateBuilder = Callable[['VFE3Config', GaugeGroup], TrainableTransportState]


_TRANSPORTS: Dict[str, TransportRegistration] = {}
_TRANSPORT_BUILDER_RESERVED_STATE_KEYS = frozenset({
    "phi",
    "group",
    "gauge_mode",
    "mu",
    "mu_key",
    "sigma",
    "sigma_key",
    "link_alpha",
    "link_soft_cap",
    "clamp_monitor",
    "exp_fp64_mode",
    "exp_fp64_norm_threshold",
    "cocycle_relaxation",
    "materialize",
    "validity_max_norm",
    "exactness_out",
})
_TRANSPORT_NEEDS_MU:    set = set()   # regimes whose Omega builder reads the belief means mu
_TRANSPORT_NEEDS_SIGMA: set = set()   # regimes whose Omega builder reads the belief covariance sigma
_TRANSPORT_BATCH_INDEPENDENT: set = set()   # regimes whose Omega is the SAME for every sequence in the
#                                             batch (depends only on a model parameter, not phi/mu/sigma),
#                                             so the builder returns a batch-collapsed (N,N,K,K) Omega that
#                                             broadcasts downstream instead of a dense (B,N,N,K,K).


def register_transport(
    name: str,

    *,
    covariance_class:           str,
    needs_mu:                   bool                              = False,
    needs_sigma:                bool                              = False,
    batch_independent:          bool                              = False,
    override:                   bool                              = False,
    state_builder:              'Optional[TransportStateBuilder]' = None,
    serialization_keys:         Tuple[str, ...]                   = (),
    offdiag_serialization_keys: Tuple[str, ...]                   = (),
) -> Callable:
    """Decorator registering a transport (connection-regime) builder under ``name``.

    ``covariance_class`` is the exact covariance/equivariance label emitted in run artifacts. It is
    mandatory so every config-selectable transport is reportable without a second literal dispatch
    table. ``needs_mu``/``needs_sigma`` are state-routing metadata: they declare which belief fields the
    regime's Omega builder consumes, so callers feed mu/sigma by querying the registry rather than
    matching literal mode names. Declaring them here keeps the add-by-registering contract -- a new
    stateful regime advertises its requirements at registration, not at every call site.

    ``state_builder`` optionally constructs the transport's complete trainable state from the active
    config and gauge group. ``serialization_keys`` declares the stable top-level model parameter names
    for that state. The model validates the builder result against this declaration and registers each
    parameter directly under its declared name, preserving external checkpoint keys without a container
    prefix. A stateful transport is therefore added entirely at registration; model and E-step routing
    consume one generic mapping and require no transport-specific branch. The optional
    ``offdiag_serialization_keys`` declaration identifies edge tables whose first two axes have a
    meaningful off-diagonal norm; shape alone cannot distinguish them from square coefficient tables.

    ``batch_independent`` declares that the builder's ``Omega`` does NOT depend on the batch (it is a
    function of a model parameter only -- the bare direct link ``regime_ii_link``), so the builder
    returns a batch-collapsed ``(N, N, K, K)`` Omega that ``transport_mean`` / ``transport_covariance``
    broadcast across the batch (the D3 memory collapse). ``_transport`` reads this flag to skip the
    per-sequence ``[0]`` strip it applies to ordinary ``(B, N, N, K, K)`` builders.

    Duplicate keys fail closed (audit 2026-07-01 round-3): a second registration under an existing
    name silently shadowed the first, so a config-selected seam could dispatch to an unintended
    implementation. Pass ``override=True`` to replace deliberately. The callable and all declarations
    are installed together as one frozen record; the legacy routing sets are then derived from that
    complete record, so stale membership cannot survive an override.
    """
    if not isinstance(covariance_class, str) or not covariance_class:
        raise ValueError("transport covariance_class must be a nonempty string")
    if state_builder is None and serialization_keys:
        raise ValueError("transport serialization_keys require a state_builder")
    if state_builder is not None and not serialization_keys:
        raise ValueError("a transport state_builder requires declared serialization_keys")
    if len(set(serialization_keys)) != len(serialization_keys):
        raise ValueError("transport serialization_keys must be unique")
    if any(not isinstance(key, str) or not key or "." in key for key in serialization_keys):
        raise ValueError("transport serialization_keys must be nonempty top-level parameter names")
    reserved_state_keys = set(serialization_keys) & _TRANSPORT_BUILDER_RESERVED_STATE_KEYS
    if reserved_state_keys:
        raise ValueError(
            "transport serialization_keys contain reserved transport-builder keyword(s): "
            f"{sorted(reserved_state_keys)}"
        )
    if not set(offdiag_serialization_keys).issubset(serialization_keys):
        raise ValueError("transport offdiag_serialization_keys must be declared serialization_keys")

    def _wrap(fn: Callable[..., TransportDict]) -> Callable[..., TransportDict]:
        if name in _TRANSPORTS and not override:
            raise KeyError(f"transport mode {name!r} already registered; pass override=True to replace")
        registration = TransportRegistration(
            callable=fn,
            needs_mu=needs_mu,
            needs_sigma=needs_sigma,
            batch_independent=batch_independent,
            covariance_class=covariance_class,
            state_builder=state_builder,
            serialization_keys=serialization_keys,
            offdiag_serialization_keys=offdiag_serialization_keys,
        )
        _TRANSPORTS[name] = registration
        # Compatibility views for existing hot-path membership checks. Their values are derived only
        # from the just-installed complete record; none is inherited from an overridden registration.
        _TRANSPORT_NEEDS_MU.discard(name)
        _TRANSPORT_NEEDS_SIGMA.discard(name)
        _TRANSPORT_BATCH_INDEPENDENT.discard(name)
        if registration.needs_mu:
            _TRANSPORT_NEEDS_MU.add(name)
        if registration.needs_sigma:
            _TRANSPORT_NEEDS_SIGMA.add(name)
        if registration.batch_independent:
            _TRANSPORT_BATCH_INDEPENDENT.add(name)
        return fn
    return _wrap


def get_transport_registration(name: str) -> TransportRegistration:
    """Return the complete registration record for ``name`` (KeyError if absent)."""
    if name not in _TRANSPORTS:
        raise KeyError(f"no transport {name!r}; available: {sorted(_TRANSPORTS)}")
    return _TRANSPORTS[name]


def get_transport(name: str) -> Callable[..., TransportDict]:
    """Return the registered transport builder (KeyError-with-available-list if absent)."""
    return get_transport_registration(name).callable


def merge_legacy_transport_state(
    transport_state: Optional[TransportState]  = None,

    *,
    connection_W:  Optional[torch.Tensor] = None,
    connection_M:  Optional[torch.Tensor] = None,
    connection_L:  Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    r"""Normalize the legacy direct-connection API into one generic state mapping.

    New model code passes only ``transport_state``. The named arguments remain a compatibility
    boundary for external callers written before registry-owned state; a conflicting duplicate is
    rejected instead of silently choosing one tensor.
    """
    merged = dict(transport_state or {})
    legacy = {
        "connection_W": connection_W,
        "connection_M": connection_M,
        "connection_L": connection_L,
    }
    for key, value in legacy.items():
        if value is None:
            continue
        if key in merged and merged[key] is not value:
            raise ValueError(f"transport state {key!r} was provided through both APIs")
        merged[key] = value
    return merged


def _build_regime_ii_state(
    cfg:   'VFE3Config',
    group: GaugeGroup,
) -> TrainableTransportState:
    r"""Zero-init bilinear connection ``W^a in R^(K x K)`` for curved Regime II."""
    n_gen = group.generators.shape[0]
    return {
        "connection_W": nn.Parameter(torch.zeros(n_gen, cfg.embed_dim, cfg.embed_dim)),
    }


def _build_regime_ii_covariant_state(
    cfg:   'VFE3Config',
    group: GaugeGroup,
) -> TrainableTransportState:
    r"""Zero-init invariant-feature coefficients ``M^a_f`` for covariant Regime II."""
    del cfg
    n_gen = group.generators.shape[0]
    return {
        "connection_M": nn.Parameter(torch.zeros(n_gen, 3)),
    }


def _build_regime_ii_link_state(
    cfg:   'VFE3Config',
    group: GaugeGroup,
) -> TrainableTransportState:
    r"""Zero-init direct-link coordinates ``A_ij^a`` shared by both link transports."""
    n_gen = group.generators.shape[0]
    return {
        "connection_L": nn.Parameter(torch.zeros(cfg.max_seq_len, cfg.max_seq_len, n_gen)),
    }


def gauge_invariant_edge_features(
    mu_q:   torch.Tensor,             # (..., K) query means
    cov_q:  torch.Tensor,             # (..., K, K) query covariances (SPD)
    mu_kt:  torch.Tensor,             # (..., K) transported key means
    cov_kt: torch.Tensor,             # (..., K, K) transported key covariances (SPD)

    *,
    eps:              float = 1e-6,
    return_exactness: bool  = False,
) -> 'torch.Tensor | Tuple[torch.Tensor, torch.Tensor]':
    r"""Gauge-invariant edge features for the covariant Regime-II (Route B) connection.

    The three components of $D_{KL}(\mathcal N(\mu_q, \Sigma_q) \| \mathcal N(\mu_{kt}, S))$
    with $S = \Sigma_{kt}$ the transported key covariance:

        Mahalanobis : $(\mu_q - \mu_{kt})^\top S^{-1} (\mu_q - \mu_{kt})$
        trace       : $\operatorname{tr}(S^{-1} \Sigma_q)$
        log-det     : $\log\det S - \log\det \Sigma_q$

    Each is invariant under a common $GL(K)$ push-forward ($\mu \mapsto g\mu$,
    $\Sigma \mapsto g\Sigma g^\top$) of BOTH beliefs: the trace and Mahalanobis terms cancel by
    congruence and the $(\det g)^2$ Jacobians cancel in the log-det ratio. An edge connection
    built from these features, $\delta_{ij}^a = \sum_f M^a_f I^f_{ij}$, therefore keeps
    $\exp(\delta_{ij}\cdot G)$ gauge-invariant and the Regime-II transport
    $\Omega_{ij} = \exp(\phi_i)\exp(\delta_{ij}\cdot G)\exp(-\phi_j)$ covariant
    ($\Omega_{ij} \mapsto g_i \Omega_{ij} g_j^\top{}^{-1}$) -- unlike the bilinear ``regime_ii``
    connection $\delta_{ij} = \mu_i^\top W \mu_j$, gauge-invariant only at $W=0$.
    """
    # FLOAT64 ISLAND + safe Cholesky. The transported-key congruence S = Omega^0 Sigma Omega^0^T
    # SQUARES cond(Omega^0) ~ exp(2||phi||) on the non-compact block_glk frame, so the Cholesky-solve
    # invariants are evaluated in float64 and cast back -- an fp32 Cholesky here loses the invariants
    # entirely (audit 2026-06-18: >100% rel error / non-PD at K=70), the same reason
    # transport_covariance upcasts its M4 sandwich. safe_cholesky degrades a non-PD S to NaN via its
    # ok mask (-> kl_max downstream) instead of raising a LinAlgError that aborts the whole forward.
    orig_dtype    = mu_q.dtype
    mu_q,  cov_q  = mu_q.double(),  cov_q.double()
    mu_kt, cov_kt = mu_kt.double(), cov_kt.double()
    def _factor_with_status(
        covariance: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        symmetric = 0.5 * (covariance + covariance.transpose(-1, -2))
        factor, info = torch.linalg.cholesky_ex(symmetric)
        exact = info == 0
        ok = exact
        if not bool(exact.all()):
            recovered_factor, recovered_ok = safe_cholesky(symmetric, eps=eps, rounds=5)
            factor = torch.where(
                ((~exact) & recovered_ok).unsqueeze(-1).unsqueeze(-1),
                recovered_factor,
                factor,
            )
            ok = recovered_ok
        return factor, ok, exact

    L_s, ok_s, exact_s = _factor_with_status(cov_kt)   # S = cov_kt = L_s L_s^T
    L_q, ok_q, exact_q = _factor_with_status(cov_q)

    delta_mu = (mu_q - mu_kt).unsqueeze(-1)                # (..., K, 1) prediction error
    sol      = torch.cholesky_solve(delta_mu, L_s)         # S^{-1} (mu_q - mu_kt)
    mahal    = (delta_mu * sol).sum(dim=(-2, -1))          # (...,)  Mahalanobis

    sinv_covq = torch.cholesky_solve(cov_q, L_s)           # S^{-1} Sigma_q  (..., K, K)
    trace     = torch.diagonal(sinv_covq, dim1=-2, dim2=-1).sum(dim=-1)   # (...,)  tr(S^{-1} Sigma_q)

    logdet_s = _logdet_chol(L_s)
    logdet_q = _logdet_chol(L_q)
    logdet   = logdet_s - logdet_q                         # (...,)  log det S - log det Sigma_q

    feats = torch.stack((mahal, trace, logdet), dim=-1)    # (..., 3)
    ok    = (ok_s & ok_q).unsqueeze(-1)                    # (..., 1) PD on both factors
    feats = torch.where(ok, feats, feats.new_tensor(float("nan")))   # non-PD edge -> NaN -> kl_max
    feats = feats.to(orig_dtype)                           # back to the caller's working dtype
    if return_exactness:
        return feats, exact_s & exact_q
    return feats


def _soft_cap_frobenius(
    matrix: torch.Tensor,             # (..., K, K) embedded Lie-algebra matrices

    *,
    max_norm: float,
) -> torch.Tensor:
    r"""Smoothly cap each matrix as $M / \sqrt{1 + \lVert M\rVert_F^2 / c^2}$.

    The norm algebra runs in float64 before squaring so a finite float32 ``M`` cannot overflow
    ``||M||_F^2`` and spuriously collapse the capped matrix to zero. The result returns to the
    caller's working dtype; the float64 island remains differentiable.
    """
    matrix64 = matrix.double()
    fro_sq64 = matrix64.square().sum(dim=(-2, -1), keepdim=True)
    scale64  = torch.rsqrt(1.0 + fro_sq64 / (max_norm * max_norm))
    return (matrix64 * scale64).to(matrix.dtype)


def _record_covariant_feature_exactness(
    exactness_out: Dict,
    exact:         'bool | torch.Tensor',

    *,
    device:        torch.device,
) -> None:
    r"""Merge one covariant-feature exactness result into a run-sticky status sink."""
    key = "regime_ii_covariant_feature_exact"
    current = torch.as_tensor(exactness_out.get(key, True), dtype=torch.bool, device=device)
    update = torch.as_tensor(exact, dtype=torch.bool, device=device)
    exactness_out[key] = current & update


@register_transport("flat", covariance_class="covariant (flat)")
def _build_flat(
    phi:        torch.Tensor,             # (B, N, n_gen) gauge frames
    group:      GaugeGroup,               # supplies generators, skew flag, irrep_dims

    *,
    gauge_mode:              str             = "learned",   # 'learned' (Regime I flat) or 'trivial'
    exp_fp64_mode:           str             = "dim",       # stable_matrix_exp_pair float64-island keying ('dim' | 'norm')
    exp_fp64_norm_threshold: float           = 5.0,         # 'norm' mode: max clamped block ||M||_F upcast threshold
    clamp_monitor:           bool            = False,       # opt-in: warn when the exp Frobenius clamp fires
    validity_max_norm:       Optional[float] = None,        # opt-in fail-closed pre-clamp chart bound
    **kwargs,                             # tolerated (a future non-flat builder shares this shape)
) -> TransportDict:
    r"""Flat (Regime I) phi-cocycle transport: the registered default.

    A thin adapter forwarding verbatim to :func:`compute_transport_operators`
    (Omega_ij = exp(phi_i) exp(-phi_j) in GL+(K)); bit-identical to calling it directly. Extra
    keyword args are tolerated and ignored so a future stateful non-flat (Regime II) builder can
    share this call shape without editing the registry call sites.
    """
    return compute_transport_operators(phi, group, gauge_mode=gauge_mode,
                                       exp_fp64_mode=exp_fp64_mode,
                                       exp_fp64_norm_threshold=exp_fp64_norm_threshold,
                                       clamp_monitor=clamp_monitor,
                                       validity_max_norm=validity_max_norm)


@register_transport(
    "regime_ii",
    covariance_class="gauge-fixed (non-covariant)",
    needs_mu=True,
    state_builder=_build_regime_ii_state,
    serialization_keys=("connection_W",),
)
def _build_regime_ii(
    phi:                torch.Tensor,             # (B, N, n_gen) gauge frames
    group:              GaugeGroup,               # supplies generators, skew flag, irrep_dims

    *,
    gauge_mode:         str                       = "learned",   # 'learned' (flat vertex factors) | 'trivial'
    cocycle_relaxation: float                     = 1.0,         # homotopy alpha in [0,1]; 0 -> flat
    delta_soft_cap:     float                     = 12.0,        # smooth bound on ||delta_ij . G||_F (< exp clamp max_norm=15)
    clamp_monitor:      bool                      = False,       # opt-in: warn when the exp Frobenius clamp fires
    validity_max_norm:  Optional[float]           = None,        # opt-in fail-closed pre-clamp chart bound
    mu:                 Optional[torch.Tensor]    = None,        # (B, N, K) QUERY-slot means; the bilinear delta reads these
    connection_W:       Optional[torch.Tensor]    = None,        # (n_gen, K, K) learned bilinear connection (NN exception)
    mu_key:             Optional[torch.Tensor]    = None,        # (B, N, K) KEY-slot means (None -> mu); the filtering
    #                                                              oracle passes a DETACHED key slot so d delta/d mu
    #                                                              flows query-side only (values are detach-invariant)
    **kwargs,                                                    # tolerated (shares the flat builder's call shape)
) -> TransportDict:
    r"""Regime-II edge-relaxed (NON-FLAT) transport (spec eq:edge_relaxed_omega).

    GAUGE-FIXED / NON-COVARIANT: delta_ij = mu_i^T W mu_j is gauge-invariant only at W=0; a
    trained nonzero W breaks gauge equivariance. For exact GL-covariant transport use
    transport_mode='regime_ii_covariant' (Route B).

    NEURAL-NETWORK EXCEPTION (sanctioned, default-OFF): this builder consumes the LEARNED
    bilinear connection ``connection_W`` (an nn.Parameter on the model, trained by backprop on
    CE). The no-NN flat builder (:func:`_build_flat`) is the default and the pure path; this is
    the non-flat regime selected only by ``transport_mode='regime_ii'``.

    The edge-relaxed cocycle inserts an edge-local connection between the vertex factors:

        Omega_ij = exp(phi_i . G) exp(delta_ij . G) exp(-phi_j . G),       i != j,
        Omega_ii = exp(phi_i . G) exp(-phi_i . G) = I (self-edge excluded: delta_ii := 0),
        delta_ij^a = cocycle_relaxation * (mu_i^T W^a mu_j),   a = 1..n_gen,
        delta_ij . G = sum_a delta_ij^a G_a   in g,
        delta_ij . G -> (delta_ij . G) / sqrt(1 + ||delta_ij . G||_F^2 / delta_soft_cap^2)   (cap),

    with ``{G_a} = group.generators``. Because delta is valued in the group's own generator
    coordinates, exp(delta_ij . G) lies in the group by construction (block structure / irrep_dims
    preserved). At ``connection_W=None`` or ``cocycle_relaxation=0`` the flat dict is returned
    byte-identically; an all-ZERO W tensor (the model init) takes the generic path and reduces to
    the flat cocycle to fp32 tolerance (atol 1e-6, pinned by tests/test_regime_ii.py -- NOT
    bit-exact: the extra exp(0)=I einsum reorders fp32 ops). A nonzero W gives the non-trivial
    triangle holonomy ``metrics.holonomy_deviation_sampled`` was built to read.

    NOT a symmetric cocycle: delta_ji = mu_j^T W^a mu_i != -delta_ij for a general learned W, so
    Omega_ji != Omega_ij^{-1} (reciprocity holds only for the flat cocycle). No current consumer
    assumes the inverse property -- the attention energies are directional and both directions are
    built independently -- but a future reverse-message path must NOT reuse the transpose/inverse
    shortcut here (audit 2026-06-10 F5).

    Design notes (audit 2026-06-10 F13): each coefficient delta_ij^a reads the FULL K-vector of
    both means through the unmasked (n_gen, K, K) ``W^a`` -- cross-head content coupling is
    intended (the connection is a model-level object, not a per-head projection). The bilinear is
    computed on RAW means: relative position enters the energy only through the outer gauge-RoPE
    rotation R_i Omega_ij R_j^T, never through delta itself (position-blind content interaction,
    by design).

    COST: unlike flat (mu-independent, O(N) vertex exponentials), Omega here depends on the CURRENT
    belief means mu and the edge factor is a PER-EDGE K x K matrix exponential -- O(N^2) matrix
    exponentials per build, and the build must be repeated as mu updates across E-step iterations
    (twice per iteration when the phi step runs; ``e_phi_lr=0`` halves the build count). The smooth
    ``delta_soft_cap`` is applied to the EMBEDDED matrix Frobenius norm ||delta . G||_F (audit
    2026-06-13 M3), so it keeps the edge factor below ``stable_matrix_exp_pair``'s hard Frobenius
    clamp (max_norm=15) for EVERY generator basis -- orthonormal (Gram=I: glk/block_glk, where it is
    value-equivalent to the old coordinate cap: analytically equal, ~5e-7 fp32 op-reorder) and the orthogonal-but-not-orthonormal towers
    (so_n/sp_n, where the old coordinate cap underbounded the operator and the exp silently fell back
    to the clamped surrogate). The exp is therefore always the EXACT operator, the cocycle_relaxation
    homotopy never saturates, and autograd never optimizes a clamped surrogate.

    Returns the SAME dict shape as the flat builder: 'exp_phi' (B,N,K,K), 'exp_neg_phi' (B,N,K,K),
    'Omega' (B,N,N,K,K).
    """
    # Flat fast path: no connection at all (None), or the homotopy collapses it (alpha=0). Trivial
    # vertex factors do NOT erase the edge: they leave Omega_ij = exp(delta_ij . G), matching the
    # charted direct-link contract. NOTE: we deliberately do NOT short-circuit on an all-ZERO
    # (but grad-requiring)
    # connection_W: at W=0 the edge factor exp(delta)=I numerically (so the W=0->flat oracle holds to
    # float tolerance), but d Omega / d W at W=0 is the generator structure (exp'(0)=I), NOT zero --
    # short-circuiting there would sever the autograd graph and freeze the parameter at init. The full
    # einsum path keeps W in the graph so the loss backpropagates to it.
    if connection_W is None or cocycle_relaxation == 0.0:
        return compute_transport_operators(
            phi, group, gauge_mode=gauge_mode, clamp_monitor=clamp_monitor,
            validity_max_norm=validity_max_norm)

    # Vertex factors exp(phi_i), exp(-phi_j) in FACTORED form (audit 2026-06-10 F8a): the same
    # stable exp machinery as the flat builder, WITHOUT materializing the dense (B, N, N, K, K)
    # flat Omega this path would immediately discard.
    fac = build_factored_transport(
        phi, group, gauge_mode=gauge_mode, clamp_monitor=clamp_monitor,
        validity_max_norm=validity_max_norm)
    exp_phi, exp_neg_phi = fac.exp_phi, fac.exp_neg_phi                         # (B, N, K, K)

    mu_k = mu_key if mu_key is not None else mu

    generators = group.generators                                              # (n_gen, K, K)
    block_dims = group.irrep_dims if len(group.irrep_dims) > 1 else None
    exp_dim    = max(block_dims) if block_dims is not None else None

    B, N, K = mu.shape[0], mu.shape[1], mu.shape[-1]
    cols  = torch.arange(N, device=mu.device)                                  # key indices (self-edge mask)
    chunk = _regime_ii_query_chunk(B, N, K)

    # Build Omega one QUERY-CHUNK at a time (audit 2026-07-01 F10, porting the covariant builder's
    # chunking): only (B, chunk, N, K, K) of each dense transient -- the edge Lie-algebra matrix,
    # its exponential, and the output Omega chunk -- is live at once. There is no cross-query
    # reduction, so each (i, j) operator is identical to a single-chunk build (chunk >= N collapses
    # to the original single-pass path bit-for-bit; small diagnostic builds stay one chunk).
    omega_chunks: List[torch.Tensor] = []
    for i0 in range(0, N, chunk):
        i1        = min(i0 + chunk, N)
        exp_phi_c = exp_phi[:, i0:i1]                                          # (B, C, K, K)
        # delta_ij^a = cocycle_relaxation * mu_i^T W^a mu_j -> (B, C, N, n_gen). ``mu`` fills the
        # QUERY (i) slot and ``mu_key`` the KEY (j) slot; the VALUES are identical for any detach
        # combination, but the filtering oracle passes a detached key slot so d delta / d mu flows
        # query-side only (mean-field coordinate ascent).
        delta_c   = cocycle_relaxation * torch.einsum(
            "bik,akl,bjl->bija", mu[:, i0:i1], connection_W, mu_k)             # (B, C, N, n_gen)
        # Self-edge exclusion (audit 2026-06-10 F4): the connection is an EDGE object; the
        # degenerate i==i "edge" transports along the constant path, so Omega_ii stays
        # exp(phi_i) exp(-phi_i) = I exactly as on the flat path (else delta_ii injects a spurious
        # self-energy into the unmasked softmax). The diagonal column for local row c is the
        # GLOBAL query index i0 + c (mirrors the covariant builder's masking).
        rows      = torch.arange(i0, i1, device=delta_c.device)                # (C,) global query indices
        self_edge = rows.unsqueeze(-1) == cols.unsqueeze(0)                    # (C, N) bool
        delta_c   = delta_c.masked_fill(self_edge.view(1, i1 - i0, N, 1), 0.0)
        # delta_ij . G = sum_a delta_ij^a G_a -> (B, C, N, K, K) Lie-algebra edge matrix
        delta_mat_c = torch.einsum("bija,akl->bijkl", delta_c, generators)
        # Smooth per-edge cap on the EMBEDDED MATRIX Frobenius norm (audit 2026-06-13 M3,
        # supersedes the 2026-06-10 F3 coordinate-norm cap): bounds ||delta . G||_F below
        # stable_matrix_exp_pair's hard clamp for ANY generator basis (the coordinate cap
        # underbounded the operator on so_n/sp_n towers). The squared norm (pow(2).sum, NO sqrt)
        # keeps the cap's gradient finite at delta_mat=0 (the W=0 oracle and d Omega/d W at W=0
        # are untouched -- the zero-norm NaN-grad trap) and the map is STRICTLY monotone in
        # cocycle_relaxation (the homotopy never saturates).
        delta_mat_c = _soft_cap_frobenius(delta_mat_c, max_norm=delta_soft_cap)
        # Per-edge group element exp(delta_ij . G); reuse the stable block-exp machinery
        # (only_forward: the edge factor enters Omega once, no exp(-delta) needed). exp_dim keys
        # the float64-island decision on the per-head block actually exponentiated (audit
        # 2026-06-10 F8c; the soft cap above keeps the blocks in the well-conditioned exp regime).
        exp_delta_c, _ = stable_matrix_exp_pair(
            delta_mat_c, skew_symmetric=group.skew_symmetric, only_forward=True,
            block_dims=block_dims, exp_dim=exp_dim, clamp_monitor=clamp_monitor,
            validity_max_norm=validity_max_norm,
        )                                                                      # (B, C, N, K, K)
        # Omega_ij = exp(phi_i) @ exp_delta_ij @ exp(-phi_j)
        omega_chunks.append(
            torch.einsum("bikl,bijlm,bjmn->bijkn", exp_phi_c, exp_delta_c, exp_neg_phi))

    omega = torch.cat(omega_chunks, dim=1) if len(omega_chunks) > 1 else omega_chunks[0]
    return {"exp_phi": exp_phi, "exp_neg_phi": exp_neg_phi, "Omega": omega}


# Query-chunk budget for the dense Regime-II covariant builder (OOM fix, 2026-06-18). The builder
# holds SEVERAL dense (B, chunk, N, K, K) transients AT ONCE -- the flat cocycle Omega^0, the
# transported-key covariance, the edge Lie-algebra matrix, its exponential, and the output Omega
# chunk -- so the peak working set is ~``_REGIME_II_LIVE_TRANSIENTS`` times ONE such tensor (the
# original budget modelled a single tensor and so underestimated peak by ~5x; audit 2026-06-18). At
# K=20 / 2-head / B=64 / N=128 one tensor is ~1.68 GB, so the full build OOMs on a 32 GB GPU while
# the flat K=80 run (factored, no dense Omega) fits. The chunk size bounds the SUM of the
# simultaneous transients under ``_REGIME_II_CHUNK_ELEMS`` fp32 elements; the build is otherwise
# unchanged (no cross-query reduction, so chunking is exactly value- and gradient-equivalent to one
# chunk). The short-lived float64 feature island doubles the bytes of its own share only.
_REGIME_II_CHUNK_ELEMS     = 64_000_000   # ~256 MB fp32 TOTAL peak working set across the transients
_REGIME_II_LIVE_TRANSIENTS = 5            # simultaneous dense (B, chunk, N, K, K) tensors held in the loop


def _regime_ii_query_chunk(
    b:  int,                          # batch size B
    n:  int,                          # sequence length N (query and key axes)
    k:  int,                          # belief dimension K

) -> int:                             # query-chunk size in [1, N]
    r"""Query-index chunk size bounding the dense working SET to ``_REGIME_II_CHUNK_ELEMS`` fp32
    elements. One query row holds ``_REGIME_II_LIVE_TRANSIENTS`` simultaneous (B, 1, N, K, K) tensors
    = ``_REGIME_II_LIVE_TRANSIENTS * B*N*K*K`` elements; the chunk is the largest count keeping that
    SUM under budget, clamped to [1, N]. A single-sequence diagnostic build (B=1) collapses to one
    chunk (no behavior change)."""
    per_row = max(1, _REGIME_II_LIVE_TRANSIENTS * b * n * k * k)
    return max(1, min(n, _REGIME_II_CHUNK_ELEMS // per_row))


@register_transport(
    "regime_ii_covariant",
    covariance_class="covariant",
    needs_mu=True,
    needs_sigma=True,
    state_builder=_build_regime_ii_covariant_state,
    serialization_keys=("connection_M",),
)
def _build_regime_ii_covariant(
    phi:                torch.Tensor,             # (B, N, n_gen) gauge frames
    group:              GaugeGroup,               # supplies generators, skew flag, irrep_dims

    *,
    gauge_mode:         str                       = "learned",   # 'learned' (flat vertex factors) | 'trivial'
    cocycle_relaxation: float                     = 1.0,         # homotopy alpha in [0,1]; 0 -> flat
    delta_soft_cap:     float                     = 12.0,        # smooth bound on ||delta_ij . G||_F
    clamp_monitor:      bool                      = False,       # opt-in: warn when the exp Frobenius clamp fires
    validity_max_norm:  Optional[float]           = None,        # opt-in fail-closed pre-clamp chart bound
    mu:                 Optional[torch.Tensor]    = None,        # (B, N, K) QUERY means
    sigma:              Optional[torch.Tensor]    = None,        # (B, N, K) diag OR (B, N, K, K) full QUERY covariance
    connection_M:       Optional[torch.Tensor]    = None,        # (n_gen, 3) learned invariant-feature map (NN exception)
    mu_key:             Optional[torch.Tensor]    = None,        # (B, N, K) KEY means (None -> mu; oracle detaches)
    sigma_key:          Optional[torch.Tensor]    = None,        # (B, N, ...) KEY covariance (None -> sigma)
    exactness_out:       Optional[Dict]            = None,        # opt-in numerical-status sink
    **kwargs,                                                    # tolerated (shares the flat builder's call shape)
) -> TransportDict:
    r"""Regime-II COVARIANT (Route B): a gauge-COVARIANT non-flat edge-relaxed transport.

    NEURAL-NETWORK EXCEPTION (sanctioned, default-OFF): consumes the LEARNED ``connection_M`` (an
    nn.Parameter trained by backprop). Unlike the bilinear ``regime_ii`` connection
    delta_ij = mu_i^T W mu_j (gauge-invariant ONLY at W=0), the edge coefficients here are built
    from GAUGE-INVARIANT scalar features of the (query, transported-key) belief pair, so the edge
    factor exp(delta_ij . G) is gauge-invariant and the transport stays COVARIANT
    (Omega_ij -> g_i Omega_ij g_j^{-1}) under GL(K) frame changes:

        delta_ij^a = cocycle_relaxation * sum_f M^a_f I^f_ij,                a = 1..n_gen,
        I^f_ij = (Mahalanobis, trace, log-det) of D_KL(q_i || Omega^0_ij q_j)  [flat transport],
        Omega_ij = exp(phi_i . G) exp(delta_ij . G) exp(-phi_j . G),   i != j (delta_ii := 0).

    The invariants I^f are evaluated under the FLAT vertex cocycle Omega^0 = exp(phi_i)exp(-phi_j)
    (each a gauge-invariant scalar per edge; see :func:`gauge_invariant_edge_features`); the curved
    Omega is then assembled with the same vertex factors. At ``connection_M=None`` or
    ``cocycle_relaxation=0`` the flat dict is returned byte-identically; an all-ZERO M reduces to
    the flat cocycle to fp32 tolerance (the generic path is kept so autograd to M survives at M=0,
    exp'(0)=I). A nonzero M gives non-trivial triangle holonomy (curvature > 0).

    TODO(Route A): the principled-on-a-COMPACT-subgroup variant. Restrict the connection to the
    commutant (intertwiners) of the gauge group on O(K)/U(K), where g^T W^a g = W^a holds by
    construction (e.g. W proportional to I gives delta_ij = mu_i . mu_j), giving an EXACTLY
    equivariant non-flat connection AND a bounded Wilson / Yang-Mills action. Route B (here) keeps
    the full GL(K) group at the cost of seeing the beliefs only through invariant scalar features.

    COST: like ``regime_ii``, O(N^2) per-edge matrix exponentials; additionally materializes the
    dense flat Omega^0 and does an O(N^2) per-edge K x K covariance congruence + Cholesky solve for
    the invariants. Opt-in / diagnostic; the default flat transport never reaches this builder.

    Returns the SAME dict shape as the flat builder: 'exp_phi' (B,N,K,K), 'exp_neg_phi' (B,N,K,K),
    'Omega' (B,N,N,K,K). When ``exactness_out`` is supplied, the builder records whether every
    edge feature used exact Cholesky factors rather than jitter recovery without changing that
    pinned output shape.
    """
    # Flat fast path (mirrors regime_ii): no connection or alpha=0. Trivial vertex factors retain
    # the invariant edge factor exp(delta_ij . G), matching the charted direct-link contract.
    # An all-zero (but grad-requiring) M is NOT short-circuited -- delta=0 reduces to flat to fp32,
    # but d Omega / d M at M=0 is the generator structure (exp'(0)=I), so the generic path keeps M
    # in the autograd graph (it would otherwise freeze at init).
    if connection_M is None or cocycle_relaxation == 0.0:
        if exactness_out is not None:
            _record_covariant_feature_exactness(exactness_out, True, device=phi.device)
        return compute_transport_operators(
            phi, group, gauge_mode=gauge_mode, clamp_monitor=clamp_monitor,
            validity_max_norm=validity_max_norm)

    # Contract guard (audit 2026-06-18): with a connection the edge features need the query belief
    # (mu, sigma); a missing one would otherwise raise an opaque AttributeError on `.dim()` below.
    if mu is None or sigma is None:
        raise ValueError(
            "regime_ii_covariant requires query means `mu` and covariance `sigma` when "
            f"`connection_M` is provided; got mu={'None' if mu is None else 'tensor'}, "
            f"sigma={'None' if sigma is None else 'tensor'}."
        )

    fac = build_factored_transport(
        phi, group, gauge_mode=gauge_mode, clamp_monitor=clamp_monitor,
        validity_max_norm=validity_max_norm)
    exp_phi, exp_neg_phi = fac.exp_phi, fac.exp_neg_phi                         # (B, N, K, K)

    mu_k     = mu_key   if mu_key   is not None else mu
    sigma_k  = sigma_key if sigma_key is not None else sigma
    diagonal = sigma.dim() == mu.dim()                                          # (B,N,K) vs (B,N,K,K)

    generators = group.generators                                              # (n_gen, K, K)
    block_dims = group.irrep_dims if len(group.irrep_dims) > 1 else None
    exp_dim    = max(block_dims) if block_dims is not None else None

    B, N, K = mu.shape[0], mu.shape[1], mu.shape[-1]
    cols  = torch.arange(N, device=mu.device)                                  # key indices (self-edge mask)
    chunk = _regime_ii_query_chunk(B, N, K)

    # Build Omega one QUERY-CHUNK at a time so only (B, chunk, N, K, K) of each dense transient is
    # live at once -- the dense (B, N, N, K, K) flat cocycle / transported-key covariance / Cholesky
    # factors / edge exp are NEVER materialized whole (the K=20 regime_ii_covariant OOM, 2026-06-18).
    # There is no cross-query reduction, so each (i, j) operator is identical to a single-chunk build
    # (chunk >= N collapses to the original code path, bit-for-bit).
    omega_chunks: List[torch.Tensor] = []
    feature_exact = torch.ones((), dtype=torch.bool, device=mu.device)
    for i0 in range(0, N, chunk):
        i1        = min(i0 + chunk, N)
        exp_phi_c = exp_phi[:, i0:i1]                                          # (B, C, K, K)
        # FLAT cocycle for this query chunk Omega^0_[i0:i1],j = exp(phi_i) exp(-phi_j) (B, C, N, K, K);
        # the gauge-invariant edge features are evaluated on it, the curved edge factor inserted after.
        # FLOAT64 ISLAND: the transported-key congruence Omega^0 Sigma Omega^0^T squares cond(Omega^0)
        # ~ exp(2||phi||), so an fp32 congruence destroys the gauge-invariant features on the
        # non-compact block_glk frame (audit 2026-06-18). The inputs (phi, mu, sigma) are well-
        # conditioned, so upcasting the congruence here is loss-free; the curved Omega assembly below
        # stays in the working dtype (it is not squared). feats are cast back before the delta contraction.
        ep_c64   = exp_phi_c.double()
        en_c64   = exp_neg_phi.double()
        omega0_c = torch.einsum("bikl,bjlm->bijkm", ep_c64, en_c64)           # (B, C, N, K, K) f64

        mu_q_c  = mu[:, i0:i1].unsqueeze(2).double()                          # (B, C, 1, K) query mean
        mu_kt_c = torch.einsum("bijkl,bjl->bijk", omega0_c, mu_k.double())     # (B, C, N, K) transported key mean
        if diagonal:                                                          # diagonal variances (B, N, K)
            cov_q_c  = torch.diag_embed(sigma[:, i0:i1].double()).unsqueeze(2) # (B, C, 1, K, K)
            cov_kt_c = torch.einsum("bijkl,bjl,bijml->bijkm", omega0_c, sigma_k.double(), omega0_c)
        else:                                                                 # full covariance (B, N, K, K)
            cov_q_c  = sigma[:, i0:i1].unsqueeze(2).double()                  # (B, C, 1, K, K)
            cov_kt_c = torch.einsum("bijkl,bjlm,bijnm->bijkn", omega0_c, sigma_k.double(), omega0_c)

        feats_c, exact_c = gauge_invariant_edge_features(
            mu_q_c,
            cov_q_c,
            mu_kt_c,
            cov_kt_c,
            return_exactness=True,
        )
        feature_exact = feature_exact & exact_c.all()
        feats_c = feats_c.to(exp_phi.dtype)                                   # back to the working dtype
        delta_c = cocycle_relaxation * torch.einsum("bijf,af->bija", feats_c, connection_M)   # (B, C, N, n_gen)

        # Self-edge exclusion (audit F4 parity): the connection is an EDGE object; Omega_ii stays the
        # flat identity exp(phi_i)exp(-phi_i) (delta_ii zeroed before the exp). The diagonal column for
        # local row c is the global query index i0 + c.
        rows      = torch.arange(i0, i1, device=delta_c.device)                # (C,) global query indices
        self_edge = rows.unsqueeze(-1) == cols.unsqueeze(0)                    # (C, N) bool
        delta_c   = delta_c.masked_fill(self_edge.view(1, i1 - i0, N, 1), 0.0)

        delta_mat_c = torch.einsum("bija,akl->bijkl", delta_c, generators)     # (B, C, N, K, K) Lie-algebra edge
        # Smooth per-edge Frobenius cap on the embedded operator (same safeguard / squared-norm-no-sqrt
        # finite-grad-at-zero trick as regime_ii); keeps stable_matrix_exp_pair on the EXACT operator.
        delta_mat_c = _soft_cap_frobenius(delta_mat_c, max_norm=delta_soft_cap)

        exp_delta_c, _ = stable_matrix_exp_pair(
            delta_mat_c, skew_symmetric=group.skew_symmetric, only_forward=True,
            block_dims=block_dims, exp_dim=exp_dim, clamp_monitor=clamp_monitor,
            validity_max_norm=validity_max_norm,
        )                                                                      # (B, C, N, K, K)
        # Omega_ij = exp(phi_i) @ exp_delta_ij @ exp(-phi_j)
        omega_chunks.append(
            torch.einsum("bikl,bijlm,bjmn->bijkn", exp_phi_c, exp_delta_c, exp_neg_phi))

    omega = torch.cat(omega_chunks, dim=1) if len(omega_chunks) > 1 else omega_chunks[0]
    if exactness_out is not None:
        _record_covariant_feature_exactness(
            exactness_out, feature_exact, device=feature_exact.device)
    return {
        "exp_phi": exp_phi,
        "exp_neg_phi": exp_neg_phi,
        "Omega": omega,
    }


def _direct_link_edge_exp(
    connection_L:  torch.Tensor,             # (>=N, >=N, n_gen) learned direct-link table A
    group:         GaugeGroup,               # supplies generators, skew flag, irrep_dims

    n_tok:         int,                       # active sequence length N

    *,
    link_alpha:        float                  = 1.0,
    link_soft_cap:     float                  = 6.0,
    clamp_monitor:     bool                   = False,  # opt-in: warn when the exp Frobenius clamp fires
    validity_max_norm: Optional[float]        = None,   # opt-in fail-closed pre-clamp chart bound
    device:            Optional[torch.device] = None,
    dtype:             Optional[torch.dtype]  = None,
) -> torch.Tensor:                            # (N, N, K, K) exp(link_alpha * A_ij . G)
    r"""The per-edge direct-link factor exp(link_alpha * A_ij . G), shared by the bare and charted
    direct-link builders.

    ``A = connection_L[:N, :N]`` sliced to the active length; the self-edge is masked to 0 (the link
    is an EDGE object, ``Omega_ii := I``), the EMBEDDED matrix ``A_ij . G = sum_a A_ij^a G_a`` is
    smooth-capped on its Frobenius norm (``||A . G||_F < link_soft_cap``, so ``stable_matrix_exp_pair``
    stays on the EXACT operator for every generator basis), and ``exp(.)`` runs in a float32 (or
    block-scale float64) island -- NEVER bf16/fp16. The squared-norm-no-sqrt cap keeps the gradient
    finite at ``A_ij . G = 0`` so the autograd ``d Omega / d connection_L`` at ``A=0`` (the generator
    structure ``exp'(0)=I``) survives, exactly as the ``regime_ii`` W=0 path."""
    n_gen = group.generators.shape[0]
    if connection_L.dim() != 3 or connection_L.shape[0] < n_tok or connection_L.shape[1] < n_tok:
        raise ValueError(
            "direct-link transport requires connection_L with shape (max_seq_len, max_seq_len, n_gen) "
            f"covering the active N={n_tok}; got {tuple(connection_L.shape)}."
        )
    if connection_L.shape[-1] != n_gen:
        raise ValueError(
            f"connection_L last dim must equal n_gen={n_gen}, got {connection_L.shape[-1]}."
        )
    device = device if device is not None else connection_L.device
    dtype = dtype if dtype is not None else connection_L.dtype
    # FLOAT32 ISLAND: the link exp (and its norm cap) never run in bf16/fp16, regardless of an outer
    # autocast (spec: no link exponential in low precision). stable_matrix_exp_pair adds its own
    # float64 island at the block scale when K >= dim_threshold.
    with torch.amp.autocast(connection_L.device.type, enabled=False):   # tensor-keyed (audit 2026-07-05 m10)
        link_coord = (link_alpha * connection_L[:n_tok, :n_tok, :]).to(device=device, dtype=torch.float32)
        eye_N = torch.eye(n_tok, dtype=torch.bool, device=device)
        link_coord = link_coord.masked_fill(eye_N.view(n_tok, n_tok, 1), 0.0)      # self-edge -> I
        generators = group.generators.to(device=device, dtype=torch.float32)
        link_mat = torch.einsum("ija,akl->ijkl", link_coord, generators)           # (N,N,K,K) Lie-algebra edge
        link_mat = _soft_cap_frobenius(link_mat, max_norm=link_soft_cap)
    block_dims = group.irrep_dims if len(group.irrep_dims) > 1 else None
    exp_link, _ = stable_matrix_exp_pair(
        link_mat, skew_symmetric=group.skew_symmetric, only_forward=True,
        block_dims=block_dims, exp_dim=(max(block_dims) if block_dims is not None else None),
        clamp_monitor=clamp_monitor,
        validity_max_norm=validity_max_norm,
    )                                                                              # (N, N, K, K)
    return exp_link.to(dtype)


@register_transport(
    "regime_ii_link",
    covariance_class="gauge-fixed",
    batch_independent=True,
    state_builder=_build_regime_ii_link_state,
    serialization_keys=("connection_L",),
    offdiag_serialization_keys=("connection_L",),
)
def _build_regime_ii_link(
    phi:                torch.Tensor,             # (B, N, n_gen) gauge frames (IGNORED: bare link reads only connection_L)
    group:              GaugeGroup,               # supplies generators, skew flag, irrep_dims

    *,
    gauge_mode:         str                    = "learned",
    link_alpha:         float                  = 1.0,
    link_soft_cap:      float                  = 6.0,
    clamp_monitor:      bool                   = False,  # opt-in: warn when the exp Frobenius clamp fires
    validity_max_norm:  Optional[float]        = None,   # opt-in fail-closed pre-clamp chart bound
    materialize:        bool                   = True,   # compatibility callers may request explicit Omega
    connection_L:       Optional[torch.Tensor] = None,   # (max_seq_len, max_seq_len, n_gen) learned direct link (NN exception)
    **kwargs,                                            # tolerated (shares the flat builder's call shape)
) -> TransportDict:
    r"""Bare direct-link (NON-FLAT) Regime-II transport (spec docs/research/2026-06-29-regime-ii-direct-link-spec.md).

    NEURAL-NETWORK EXCEPTION (sanctioned, default-OFF): consumes the LEARNED ``connection_L`` (an
    nn.Parameter trained by backprop). The direct group-valued link is the connection itself:

        Omega_ij = exp(link_alpha * A_ij . G),   A = connection_L,   i != j,   Omega_ii := I,

    reading ONLY ``connection_L`` -- no vertex frame ``phi``, no beliefs. Its flat limit (``A=0`` /
    ``link_alpha=0`` / ``connection_L=None``) is IDENTITY links ``Omega = I``, NOT the Regime-I vertex
    cocycle ``exp(phi_i)exp(-phi_j)``. Because the link discards the frames it is frame-INDEPENDENT and
    therefore does NOT satisfy the gauge-covariance law ``Omega_ij -> g_i Omega_ij g_j^{-1}`` -- a
    DOCUMENTED opt-in equivariance break in the ``connection_W`` / ``connection_M`` / ``regime_ii``
    family. Unlike ``connection_W`` (exact at ``W=0``, where ``regime_ii`` recovers the covariant flat
    cocycle), the bare link breaks for ALL ``connection_L``: even the ``A=0`` identity links satisfy
    ``I != g_i g_j^{-1}``. The EXACTLY covariant member is ``regime_ii_link_charted`` (the frame
    sandwich). A nonzero ``connection_L`` gives non-trivial triangle holonomy (curvature > 0).

    BATCH-INDEPENDENT: the forward builder returns :class:`DirectLinkTransport` with only the
    ``(N, N, K, K)`` edge factor. It never builds the unused vertex exponentials and broadcasts the
    edge across the batch during contraction. ``materialize=True`` is the explicit registry-level
    compatibility boundary; it returns the same logical ``(N, N, K, K)`` tensor without a batch copy.
    """
    N = phi.shape[1]
    K = group.generators.shape[-1]
    device, dtype = phi.device, phi.dtype
    # Flat (identity-link) fast path: no connection or link_alpha=0 -> Omega = I exactly. NOTE we do
    # NOT short-circuit on an all-ZERO (grad-requiring) connection_L: at A=0 exp(A)=I numerically, but
    # d Omega / d connection_L there is the generator structure (exp'(0)=I), so the generic path keeps
    # connection_L in the autograd graph (short-circuiting would freeze it at init).
    if connection_L is None or link_alpha == 0.0:
        exp_link = torch.eye(K, device=device, dtype=dtype).expand(N, N, K, K).contiguous()
    else:
        exp_link = _direct_link_edge_exp(
            connection_L, group, N,
            link_alpha=link_alpha,
            link_soft_cap=link_soft_cap,
            clamp_monitor=clamp_monitor,
            validity_max_norm=validity_max_norm,
            device=device,
            dtype=dtype,
        )
    direct = DirectLinkTransport(exp_link=exp_link)
    omega = direct.to_dense_omega() if materialize else direct
    return {"exp_phi": None, "exp_neg_phi": None, "Omega": omega}


@register_transport(
    "regime_ii_link_charted",
    covariance_class="covariant",
    state_builder=_build_regime_ii_link_state,
    serialization_keys=("connection_L",),
    offdiag_serialization_keys=("connection_L",),
)
def _build_regime_ii_link_charted(
    phi:                torch.Tensor,             # (B, N, n_gen) gauge frames
    group:              GaugeGroup,               # supplies generators, skew flag, irrep_dims

    *,
    gauge_mode:         str                    = "learned",
    link_alpha:         float                  = 1.0,
    link_soft_cap:      float                  = 6.0,
    clamp_monitor:      bool                   = False,  # opt-in: warn when the exp Frobenius clamp fires
    validity_max_norm:  Optional[float]        = None,   # opt-in fail-closed pre-clamp chart bound
    materialize:        bool                   = True,   # compatibility callers may request explicit Omega
    connection_L:       Optional[torch.Tensor] = None,   # (max_seq_len, max_seq_len, n_gen) learned direct link (NN exception)
    **kwargs,                                            # tolerated (shares the flat builder's call shape)
) -> TransportDict:
    r"""Charted direct-link (NON-FLAT) Regime-II transport: the gauge-EXACT direct-link member.

    NEURAL-NETWORK EXCEPTION (sanctioned, default-OFF): consumes the LEARNED ``connection_L``. The
    direct link is sandwiched between the co-transforming vertex frames:

        Omega_ij = exp(phi_i . G) exp(link_alpha * A_ij . G) exp(-phi_j . G),   A = connection_L,
        Omega_ii := I (self-edge masked).

    EXACTLY gauge-covariant for ANY constant ``A``: a frame change ``exp(phi_i) -> g_i exp(phi_i)``
    sends ``Omega_ij -> g_i Omega_ij g_j^{-1}`` (the co-transforming frames carry the entire
    conjugation and the constant middle factor reads nothing, so there is nothing to break). This is
    the OPPOSITE of ``regime_ii``, whose middle factor ``mu_i^T W mu_j`` reads the transforming beliefs
    through a non-invariant bilinear. Belief-INDEPENDENT (no needs_mu/needs_sigma -> kernel-eligible),
    but ``phi``-dependent, so it retains per-sequence vertex factors around the shared edge table.
    Its ``A=0`` limit is the Regime-I flat cocycle ``exp(phi_i)exp(-phi_j)`` (NOT the bare
    identity-link limit). Under ``gauge_mode='trivial'`` the vertex factors become identities but the
    direct edge factor remains ``exp(link_alpha * A_ij . G)``. A nonzero ``connection_L`` gives
    non-trivial triangle holonomy.

    The E-step requests :class:`DirectLinkTransport`, retaining the live vertex and edge factors
    without materializing ``(B,N,N,K,K)``. ``materialize=True`` remains the explicit compatibility
    boundary for direct registry callers and diagnostics.
    """
    fac = build_factored_transport(
        phi, group, gauge_mode=gauge_mode, clamp_monitor=clamp_monitor,
        validity_max_norm=validity_max_norm)
    exp_phi, exp_neg_phi = fac.exp_phi, fac.exp_neg_phi                         # (B, N, K, K)
    N = phi.shape[1]
    K = group.generators.shape[-1]
    # No connection / zero alpha is the exact flat cocycle. A grad-requiring zero connection still
    # takes the generic edge-exp route so d exp(0) / d connection_L remains live.
    if connection_L is None or link_alpha == 0.0:
        exp_link = torch.eye(K, device=phi.device, dtype=phi.dtype).expand(N, N, K, K).contiguous()
    else:
        exp_link = _direct_link_edge_exp(
            connection_L, group, N,
            link_alpha=link_alpha,
            link_soft_cap=link_soft_cap,
            clamp_monitor=clamp_monitor,
            validity_max_norm=validity_max_norm,
            device=phi.device,
            dtype=phi.dtype,
        )
    direct = DirectLinkTransport(
        exp_link=exp_link,
        exp_phi=exp_phi,
        exp_neg_phi=exp_neg_phi,
    )
    omega = direct.to_dense_omega() if materialize else direct
    return {"exp_phi": exp_phi, "exp_neg_phi": exp_neg_phi, "Omega": omega}


def _direct_link_dense_bytes(batch: int, n_tok: int, k: int, dtype: torch.dtype) -> int:
    r"""Bytes of a DENSE batched ``(B, N, N, K, K)`` link transport.

    This is the compatibility-only cost both direct-link modes avoid on the forward path. The bare
    link stores only the batch-independent ``(N, N, K, K)`` edge table. The charted link additionally
    stores two ``(B, N, K, K)`` vertex tables; its contractions never multiply them into a dense
    pairwise ``(B, N, N, K, K)`` operator."""
    bytes_per = torch.tensor([], dtype=dtype).element_size()
    return batch * n_tok * n_tok * k * k * bytes_per


# Frobenius-norm clamp for stable_matrix_exp_pair: above this the returned factor is the surrogate
# exp(max_norm*M/||M||_F), NOT exp(M). Exported so the M-step drift monitor
# (train._warn_phi_transport_clamp) trips at the SAME norm the clamp fires at -- the two cannot
# diverge (audit 2026-07-06 M2).
TRANSPORT_CLAMP_MAX_NORM: float = 20.0


def stable_matrix_exp_pair(
    matrix:                  torch.Tensor,       # (..., d, d) Lie-algebra matrices

    *,
    exp_fp64_mode:           str                 = "dim",    # float64-island keying: 'dim' (dimension rule) | 'norm'
    max_norm:                float               = TRANSPORT_CLAMP_MAX_NORM,
    exp_fp64_norm_threshold: float               = 5.0,      # 'norm' mode: upcast when max clamped block ||M||_F >= this
    dim_threshold:           int                 = 20,
    skew_symmetric:          bool                = False,
    only_forward:            bool                = False,
    clamp_monitor:           bool                = False,    # opt-in: warn when the Frobenius clamp fires (host sync)
    block_dims:              Optional[List[int]] = None,     # per-block sizes (sum==d) for a block-diagonal M
    exp_dim:                 Optional[int]       = None,     # dimension for the float64-island decision (None -> d)
    validity_max_norm:       Optional[float]     = None,     # opt-in fail-closed pre-clamp chart bound
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    r"""Represented exp(M) and optionally its float64-computed inverse.

    Frobenius-norm clamp + float64 upcast keep matrix_exp stable for large ||M||.
    The paired negative factor is the inverse of the represented forward factor, not a separately
    rounded evaluation of exp(-M), so the stored pair shares one finite-precision group element.

    SAFEGUARD, NOT THE EXACT OPERATOR: when ``||M||_F > max_norm`` the matrix is rescaled to
    ``max_norm``, so the returned factor is ``exp(max_norm * M/||M||_F)``, NOT ``exp(M)`` -- the
    singular values / determinant of the returned operator differ from the true exponential. This
    is a stability clamp on extreme inputs only; keep ||phi|| (and the regime_ii edge delta) below
    ``max_norm`` to stay exact. The per-call runtime monitor is OFF by default: detecting
    activation needs a tensor reduction (a host sync) on this hot path, which the perf budget
    avoids. Pass ``clamp_monitor=True`` (opt-in diagnostic) to accept that sync and emit a
    RuntimeWarning whenever the clamp fires -- the returned factor is then a surrogate, not exp(M).

    ``block_dims`` (audit 4b): when M is block-diagonal with these blocks (e.g. block_glk's
    GL(d_head)^H), exp(M) is exactly block-diagonal with the per-block exponentials, so each
    d_head x d_head block is exponentiated independently -- an O(H * d_head^3) cost instead of
    O(K^3) for the full K x K. The result is BIT-equivalent to the full exp (the global
    Frobenius clamp is applied to the WHOLE matrix first, and each block keeps the dtype the
    full-K path would pick, so neither the scale nor the precision changes). ``None`` (a single
    block, a cross-coupled basis, or a skew group) takes the full-matrix path unchanged.

    ``exp_dim`` (audit 2026-06-10 F8c, default None = unchanged): an explicit override of the
    dimension the float64-island decision keys on. By DEFAULT the per-block path keeps the
    full-K dtype so blocking never changes precision (the bit-equivalence pin above). A caller
    whose conditioning argument lives at the BLOCK scale -- the regime_ii per-edge factor, whose
    delta is norm-capped upstream -- may pass ``exp_dim=max(block_dims)`` to run small blocks in
    fp32 instead of upcasting the whole batch to float64 at K >= dim_threshold.

    ``exp_fp64_mode`` (Tier-1 toggle; default 'dim' = the long-standing dimension rule above,
    untouched): 'norm' keys the float64 island on the max CLAMPED block Frobenius norm instead of
    the dimension -- the conditioning argument for matrix_exp is a NORM argument (fp32 matrix_exp
    is ~1e-7 accurate at the small block norms the phi retraction guarantees, at any block dim),
    so small-norm blocks stay fp32 while the fp64 island stays REACHABLE for genuinely large
    norms (upcast when max ||M_block||_F >= ``exp_fp64_norm_threshold``). Costs one host sync (a
    scalar norm compare) per call; the clamp / monitor behavior above is identical in both modes.
    """
    # Global Frobenius clamp on the FULL matrix (one scale for all blocks) -- identical to the
    # un-blocked path, so block slicing below cannot change the operator. The norm/scale is kept
    # OUT of the autograd graph (no_grad): the clamp is a numerical SAFEGUARD, not part of the
    # modeled operator, and differentiating through the rescale (a) biases the gradient toward
    # the clamped surrogate where the clamp is active and (b) makes the norm's DOUBLE-backward
    # NaN on exactly-zero matrices (regime_ii's zeroed self-edges and the W=0 init, reached by
    # the unrolled oracle's create_graph path). Where the clamp is inactive (scale == 1.0 -- the
    # soft-capped regime_ii edge factor and any ||phi|| < max_norm) multiplying by the detached
    # constant 1.0 is byte-identical to the previous through-graph multiply.
    if validity_max_norm is not None and (
        not math.isfinite(validity_max_norm) or validity_max_norm <= 0.0
    ):
        raise ValueError(
            "validity_max_norm must be None or finite and positive, got "
            f"{validity_max_norm!r}"
        )

    with torch.no_grad():
        raw_mat_norm = matrix.norm(dim=(-2, -1), keepdim=True)
        if validity_max_norm is not None and bool((~torch.isfinite(raw_mat_norm)).any()):
            raise ValueError(
                "transport chart validity bound encountered a nonfinite matrix norm"
            )
        if validity_max_norm is not None and bool((raw_mat_norm > validity_max_norm).any()):
            observed = float(raw_mat_norm.max())
            raise ValueError(
                "transport chart validity bound exceeded before matrix-exponential clamp: "
                f"observed ||M||_F={observed:.6g} > {validity_max_norm:.6g}"
            )
        mat_norm = raw_mat_norm.clamp(min=1e-8)
        scale = (max_norm / mat_norm).clamp(max=1.0)
        if clamp_monitor:
            frac = (scale < 1.0).float().mean()
            if bool(frac > 0):
                import warnings
                warnings.warn(
                    f'stable_matrix_exp_pair: Frobenius clamp active on {float(frac):.1%} of matrices '
                    f'(max_norm={max_norm}); returned factor is a surrogate, not exp(M).',
                    RuntimeWarning, stacklevel=2,
                )
    matrix = matrix * scale

    d = matrix.shape[-1]
    orig_dtype = matrix.dtype
    if exp_fp64_mode == "norm":
        # Norm-keyed float64 island (Tier-1 toggle): upcast ONLY when the max CLAMPED block
        # Frobenius norm reaches the threshold. Computed from the already-clamped matrices (the
        # clamp above ran first), so the keying norm is the norm actually exponentiated; the
        # island stays reachable for genuinely large norms. One host sync (the bool compare),
        # opt-in by mode.
        with torch.no_grad():
            if block_dims is not None and len(block_dims) > 1:
                start = 0
                key_norm = matrix.new_zeros(())
                for blk in block_dims:
                    end = start + blk
                    key_norm = torch.maximum(
                        key_norm, matrix[..., start:end, start:end].norm(dim=(-2, -1)).max())
                    start = end
            else:
                key_norm = (mat_norm * scale).max()      # clamped full-matrix norm (single block)
        up_dtype = torch.float64 if bool(key_norm >= exp_fp64_norm_threshold) else torch.float32
    elif exp_fp64_mode == "dim":
        # 'dim': the long-standing dimension rule. The full-K path's dtype choice; the per-block
        # path forces the SAME dtype so a small block (d_head < dim_threshold) does not silently
        # drop to float32 and drift from the full exp. exp_dim (when given) overrides the keying
        # dimension -- see the docstring.
        d_eff = exp_dim if exp_dim is not None else d
        with torch.no_grad():
            large_skew = skew_symmetric and bool(
                (mat_norm * scale).max() >= exp_fp64_norm_threshold
            )
        up_dtype = torch.float64 if d_eff >= dim_threshold or large_skew else torch.float32
    else:
        raise ValueError(f"exp_fp64_mode must be 'dim' or 'norm', got {exp_fp64_mode!r}")

    # Keyed to the TENSOR's device (audit 2026-07-05 m10): the old 'cuda' literal left the island
    # open under a CPU autocast context (torch.amp.autocast('cpu', bf16)), which _amp_context
    # deliberately supports -- matrix_exp would then run in bf16 on CPU-AMP runs.
    with torch.amp.autocast(matrix.device.type, enabled=False):
        matrix_up = matrix.to(up_dtype).contiguous()

        if block_dims is not None and len(block_dims) > 1:
            exp_pos = _blockwise_matrix_exp(matrix_up, block_dims).to(orig_dtype)
            if only_forward:
                exp_neg = None
            else:
                exp_neg = _blockwise_group_inverse(exp_pos, block_dims)
            return exp_pos, exp_neg

        exp_pos = torch.linalg.matrix_exp(matrix_up).to(orig_dtype)
        if only_forward:
            exp_neg = None
        else:
            exp_neg = _checked_group_inverse(exp_pos)
    return exp_pos, exp_neg


def _blockwise_matrix_exp(
    matrix:     torch.Tensor,             # (..., d, d) block-diagonal Lie-algebra matrix
    block_dims: List[int],                # block sizes; sum == d
) -> torch.Tensor:                        # (..., d, d) block-diagonal exp
    r"""exp of a block-diagonal matrix = block-diagonal of the blocks' exps (audit 4b).

    Exact for a block-diagonal M (off-block entries are zero, so the blocks commute trivially
    and exp does not mix them; Higham, Functions of Matrices, Sec 10.3). Off-block entries of the
    output are left at zero -- matching the full exp, whose off-block entries are exactly zero for
    a block-diagonal input.

    When the blocks are EQUAL size (block_glk's GL(d_head)^H), the H diagonal blocks are stacked
    into one batched ``matrix_exp`` (a single call instead of H sequential ones -- the
    launch-bound pattern a GPU is starved by); ``matrix_exp`` evaluates each (d, d) block
    independently, so this is bit-identical to the per-block loop (pinned at 1e-12 by
    tests/test_perf_equivalence.py::test_per_block_exp_is_bit_equivalent_to_full_exp). Unequal
    block sizes (a general block-diagonal M) fall back to the per-block loop.
    """
    out = torch.zeros_like(matrix)
    if len(set(block_dims)) == 1 and len(block_dims) > 1:
        # Diagonal-block gather/scatter without the H-iteration Python loops (audit
        # 2026-06-09 overnight F3): viewing (..., H*d, H*d) as (..., H, d, H, d), the H
        # diagonal blocks are torch.diagonal over the two H axes -- one view each way --
        # so the read is a single copy and the write-back a single in-place copy_ into the
        # diagonal view of the zero output (same kernel-level semantics as the former
        # slice assignments, H+H fewer launches).
        H, d = len(block_dims), block_dims[0]
        batch = matrix.shape[:-2]
        m5 = matrix.reshape(*batch, H, d, H, d)
        blocks = torch.diagonal(m5, dim1=-4, dim2=-2).movedim(-1, 0).contiguous()  # (H, ..., d, d)
        exps = torch.linalg.matrix_exp(blocks)                  # one batched call
        out5 = out.reshape(*batch, H, d, H, d)
        torch.diagonal(out5, dim1=-4, dim2=-2).copy_(exps.movedim(0, -1))
        return out
    start = 0
    for dim in block_dims:
        end = start + dim
        blk = matrix[..., start:end, start:end].contiguous()
        out[..., start:end, start:end] = torch.linalg.matrix_exp(blk)
        start = end
    return out


def _blockwise_group_inverse(
    matrix:     torch.Tensor,             # (..., d, d) represented block-diagonal group element
    block_dims: List[int],                # block sizes; sum == d
) -> torch.Tensor:                        # (..., d, d) inverse with structural off-block zeros
    r"""Invert represented diagonal blocks in float64 without materializing a full dense solve."""
    out = torch.zeros_like(matrix)
    if len(set(block_dims)) == 1 and len(block_dims) > 1:
        H, d = len(block_dims), block_dims[0]
        batch = matrix.shape[:-2]
        matrix_view = matrix.reshape(*batch, H, d, H, d)
        blocks = torch.diagonal(
            matrix_view,
            dim1=-4,
            dim2=-2,
        ).movedim(-1, -3).contiguous()
        inverses = _checked_group_inverse(blocks)
        out_view = out.reshape(*batch, H, d, H, d)
        torch.diagonal(out_view, dim1=-4, dim2=-2).copy_(inverses.movedim(-3, -1))
        return out

    start = 0
    for dim in block_dims:
        end = start + dim
        block = matrix[..., start:end, start:end].contiguous()
        out[..., start:end, start:end] = _checked_group_inverse(block)
        start = end
    return out


def compute_transport_operators(
    phi:        torch.Tensor,             # (B, N, n_gen) gauge frames
    group:      GaugeGroup,               # supplies generators, skew flag, irrep_dims

    *,
    gauge_mode:              str             = "learned",   # 'learned' (Regime I flat) or 'trivial'
    exp_fp64_mode:           str             = "dim",       # stable_matrix_exp_pair float64-island keying ('dim' | 'norm')
    exp_fp64_norm_threshold: float           = 5.0,         # 'norm' mode: max clamped block ||M||_F upcast threshold
    clamp_monitor:           bool            = False,       # opt-in: warn when the exp Frobenius clamp fires
    validity_max_norm:       Optional[float] = None,        # opt-in fail-closed pre-clamp chart bound
) -> TransportDict:
    r"""phi/exp transport Omega_ij = exp(phi_i) @ exp(-phi_j) in GL+(K).

    Flat (Regime I) transport operator construction. 'trivial' returns Omega = I.
    Returns 'exp_phi' (B,N,K,K), 'exp_neg_phi' (B,N,K,K), 'Omega' (B,N,N,K,K).
    The 'constant' gauge mode is intentionally NOT supported (it would require a
    per-head learned Omega parameter, which this no-NN design does not have);
    'constant' raises ValueError.

    Conditioning note (audit 2026-06-09 overnight F26): for the NON-COMPACT groups
    (glk/block_glk/sp_n; skew_symmetric=False) Omega is not orthogonal and cond(Omega)
    grows like exp(2 ||phi_matrix||); at the phi retraction's default max_norm=5.0 a
    draw can reach cond ~1e7-1e10, and the full-covariance sandwich Omega Sigma Omega^T
    SQUARES it. ``transport_covariance`` evaluates that full-covariance sandwich in a
    float64 island (audit 2026-06-13 M4) so the squared conditioning no longer loses all
    fp32 digits; compact so towers give orthogonal Omega (cond = 1) and are unaffected.
    Omega itself is still built at the working dtype here, so for extreme draws prefer a
    compact group / diagonal family or bound phi via the retraction max_norm.
    """
    B, N, _ = phi.shape
    generators = group.generators
    K = generators.shape[-1]
    dtype = phi.dtype
    device = phi.device

    if gauge_mode == "trivial":
        eye_K = torch.eye(K, device=device, dtype=dtype)
        return {
            "exp_phi":     eye_K.expand(B, N, K, K).contiguous(),
            "exp_neg_phi": eye_K.expand(B, N, K, K).contiguous(),
            "Omega":       eye_K.expand(B, N, N, K, K).contiguous(),
        }
    if gauge_mode != "learned":
        raise ValueError(f"gauge_mode must be 'learned' or 'trivial', got {gauge_mode!r}")

    phi_matrix = torch.einsum("bna,aij->bnij", phi, generators)
    # Per-block exp when the group is genuinely block-diagonal (block_glk without cross-couplings
    # -> irrep_dims [d_head]*H); single-block ([K]: glk, so_k, cross-coupled) takes the full path.
    # exp_dim keys the float64-island decision on the dimension actually exponentiated -- the
    # per-head block -- mirroring the regime_ii edge exp (audit F8c) and the always-fp32
    # per-block exp at d_head < 20. The conditioning argument lives at the block scale: the
    # retraction bounds ||phi|| (coords) by max_norm=5.0, so each block's Frobenius norm is far
    # inside fp32 matrix_exp's exact regime; without the override every flat run at K >= 20 paid
    # a (B, N, K, K) float64 upcast (vram audit 2026-06-10: ~0.4 GB of f64 transients per build
    # plus fp64-throughput matrix_exp on a consumer GPU) for blocks that never needed it.
    block_dims = group.irrep_dims if len(group.irrep_dims) > 1 else None
    exp_phi, exp_neg_phi = stable_matrix_exp_pair(
        phi_matrix, skew_symmetric=group.skew_symmetric, block_dims=block_dims,
        exp_dim=(max(block_dims) if block_dims is not None else None),
        exp_fp64_mode=exp_fp64_mode, exp_fp64_norm_threshold=exp_fp64_norm_threshold,
        # m16: matrix_exp of a skew matrix is exactly orthogonal at ANY norm, so the Frobenius clamp
        # only gratuitously shortens the rotation on the pure so_n/so_k tower path (whose retraction
        # caps ||phi|| in COORDINATES, under-bounding the embedded norm). Disable it for skew; the
        # non-compact groups (glk/block_glk/sp_n) keep the clamp as a genuine exp-overflow safeguard.
        max_norm=(float("inf") if group.skew_symmetric else TRANSPORT_CLAMP_MAX_NORM),
        clamp_monitor=clamp_monitor,
        validity_max_norm=validity_max_norm,
    )
    omega = torch.einsum("bikl,bjlm->bijkm", exp_phi, exp_neg_phi)
    return {"exp_phi": exp_phi, "exp_neg_phi": exp_neg_phi, "Omega": omega}


def _checked_group_inverse(
    omega: torch.Tensor,                    # (..., d, d) stored full elements or compact blocks
) -> torch.Tensor:
    r"""True float64 inverse with immediate nonfinite/singular failure and dtype restoration."""
    with torch.no_grad():
        if not bool(torch.isfinite(omega).all()):
            raise FloatingPointError("omega group element contains nonfinite values before inversion")
    try:
        with torch.amp.autocast(omega.device.type, enabled=False):
            inverse64 = torch.linalg.inv(omega.double())
    except RuntimeError as exc:
        raise ValueError("omega group element is singular and cannot be inverted") from exc
    with torch.no_grad():
        if not bool(torch.isfinite(inverse64).all()):
            raise FloatingPointError(
                "omega group-element inverse is nonfinite; the element is singular or numerically "
                "unrepresentable")
    inverse = inverse64.to(omega.dtype)
    with torch.no_grad():
        if not bool(torch.isfinite(inverse).all()):
            raise FloatingPointError(
                f"omega group-element inverse is nonfinite after conversion to {omega.dtype}")
    return inverse


def group_element_inverse(
    omega:        'torch.Tensor | CompactBlockElement',  # (..., K, K) element or compact (...,H,d,d) blocks
    group:        GaugeGroup,               # supplies the skew-generator flag

    *,
    residual_tol: float = 1e-4,
) -> 'torch.Tensor | CompactBlockElement':
    r"""Return the represented element's true inverse, including rounded skew-group frames.

    A matrix exponential of a skew generator is orthogonal analytically, but its stored float32
    representation generally is not exactly orthogonal. Using a transpose for that representation
    breaks exact cocycle telescoping. ``residual_tol`` remains a validated compatibility argument;
    inverse selection no longer depends on a practically unreachable exact-equality fast path.
    """
    if not math.isfinite(residual_tol) or residual_tol < 0.0:
        raise ValueError(
            f"residual_tol must be finite and nonnegative, got {residual_tol!r}")

    if isinstance(omega, CompactBlockElement):
        inverse_blocks = _checked_group_inverse(omega.blocks)
        return CompactBlockElement(inverse_blocks, omega.K, tied=omega.tied)
    del group
    return _checked_group_inverse(omega)


def build_transport_from_element(
    omega:  'torch.Tensor | CompactBlockElement',  # dense (B,N,K,K) or compact block element
    group:  GaugeGroup,

    *,
    mean_per_head: bool = False,
) -> 'CompactFactoredTransport | FactoredTransport | TransportDict':
    r"""Exp-free flat cocycle from a stored group element: Omega_ij = U_i U_j^{-1}.

    The 'omega_direct' parameterization stores the frame as the element U_i itself rather than the
    Lie-algebra coordinate phi_i, so the transport is assembled WITHOUT any matrix exponential --
    only the inverse U_j^{-1}. Dense elements fill the FactoredTransport / builder-dict slots
    directly. CompactBlockElement inputs invert their d x d blocks and return
    CompactFactoredTransport, so every contraction remains compact.

    U_j^{-1} uses :func:`group_element_inverse`: every stored representation enters a bounded float64
    inverse island, including rounded skew-group frames. Public inverse factors return to the input dtype. For
    equal-block dense groups (block_glk) a FactoredTransport is returned so the per-head fast paths
    run; for a compact equal-block element, CompactFactoredTransport is returned; for a single block
    (glk), the dense {'exp_phi','exp_neg_phi','Omega'} dict is returned (matching
    compute_transport_operators' return shape). ``mean_per_head`` is stored on either factored
    container so downstream mean transport can honor the configured block contraction.
    """
    u_inv = group_element_inverse(omega, group)
    if isinstance(omega, CompactBlockElement):
        assert isinstance(u_inv, CompactBlockElement)
        return CompactFactoredTransport(
            omega.expanded_blocks(), u_inv.expanded_blocks(), omega.K,
            mean_per_head=mean_per_head,
            same_frame_flat_cocycle=True,
        )
    block_dims = group.irrep_dims
    if len(block_dims) > 1 and len(set(block_dims)) == 1:
        return FactoredTransport(
            exp_phi=omega, exp_neg_phi=u_inv, irrep_dims=list(block_dims),
            mean_per_head=mean_per_head,
            same_frame_flat_cocycle=True,
        )
    Omega = torch.einsum("...ikl,...jlm->...ijkm", omega, u_inv)   # (B, N, N, K, K)
    return {"exp_phi": omega, "exp_neg_phi": u_inv, "Omega": Omega}


def _stable_compact_glk_exp_pair(
    blocks:                  torch.Tensor,       # (..., H, d, d) block-diagonal gl(d)^H element

    *,
    exp_fp64_mode:           str             = "dim",     # float64-island keying: 'dim' | 'norm'
    exp_fp64_norm_threshold: float           = 5.0,
    clamp_monitor:           bool            = False,
    validity_max_norm:       Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Exponentiate packed blocks once and invert their represented values in float64."""
    if validity_max_norm is not None and (
        not math.isfinite(validity_max_norm) or validity_max_norm <= 0.0
    ):
        raise ValueError(
            "validity_max_norm must be None or finite and positive, got "
            f"{validity_max_norm!r}"
        )
    with torch.no_grad():
        raw_mat_norm = blocks.square().sum(dim=(-3, -2, -1), keepdim=True).sqrt()
        if validity_max_norm is not None and bool((~torch.isfinite(raw_mat_norm)).any()):
            raise ValueError(
                "transport chart validity bound encountered a nonfinite matrix norm"
            )
        if validity_max_norm is not None and bool((raw_mat_norm > validity_max_norm).any()):
            observed = float(raw_mat_norm.max())
            raise ValueError(
                "transport chart validity bound exceeded before matrix-exponential clamp: "
                f"observed ||M||_F={observed:.6g} > {validity_max_norm:.6g}"
            )
        mat_norm = raw_mat_norm.clamp(min=1e-8)
        scale = (TRANSPORT_CLAMP_MAX_NORM / mat_norm).clamp(max=1.0)
        if clamp_monitor:
            frac = (scale < 1.0).float().mean()
            if bool(frac > 0):
                import warnings
                warnings.warn(
                    f'stable_matrix_exp_pair: Frobenius clamp active on {float(frac):.1%} of matrices '
                    f'(max_norm={TRANSPORT_CLAMP_MAX_NORM}); returned factor is a surrogate, not exp(M).',
                    RuntimeWarning, stacklevel=2,
                )
    blocks = blocks * scale

    d = blocks.shape[-1]
    orig_dtype = blocks.dtype
    if exp_fp64_mode == "norm":
        with torch.no_grad():
            key_norm = blocks.norm(dim=(-2, -1)).max()
        up_dtype = torch.float64 if bool(key_norm >= exp_fp64_norm_threshold) else torch.float32
    elif exp_fp64_mode == "dim":
        up_dtype = torch.float64 if d >= 20 else torch.float32
    else:
        raise ValueError(f"exp_fp64_mode must be 'dim' or 'norm', got {exp_fp64_mode!r}")

    with torch.amp.autocast(blocks.device.type, enabled=False):
        blocks_up = blocks.to(up_dtype).contiguous()
        exp_pos = torch.linalg.matrix_exp(blocks_up).to(orig_dtype)
        exp_neg = _checked_group_inverse(exp_pos)
    return exp_pos, exp_neg


def build_factored_transport(
    phi:        torch.Tensor,             # (..., N, n_gen) gauge frames (optional leading batch axis)
    group:      GaugeGroup,               # block-diagonal with equal blocks (len(irrep_dims) > 1)

    *,
    gauge_mode:              str                    = "learned",   # 'learned' (Regime I flat) or 'trivial'
    exp_fp64_mode:           str                    = "dim",       # stable_matrix_exp_pair float64-island keying ('dim' | 'norm')
    exp_fp64_norm_threshold: float                  = 5.0,         # 'norm' mode: max clamped block ||M||_F upcast threshold
    clamp_monitor:           bool                   = False,       # opt-in: warn when the exp Frobenius clamp fires
    mean_per_head:           bool                   = False,       # container flag: transport_mean contracts per gauge block
    compact_blocks:          bool                   = False,       # canonical block_glk: retain (..., N, H, d, d) factors
    validity_max_norm:       Optional[float]        = None,        # opt-in fail-closed pre-clamp chart bound
    right_phi:               Optional[torch.Tensor] = None,        # (..., N, n_gen) exact right factor exp(Y)
) -> 'CompactFactoredTransport | FactoredTransport':
    r"""Flat phi-cocycle transport in FACTORED form, skipping the dense (..., N, N, K, K) Omega.

    Builds only the per-token vertex exponentials exp(phi_i), exp(-phi_j) (the same factors
    ``compute_transport_operators`` builds) and the ``ikl,jlm->ijkm`` Omega einsum is NEVER run.
    With ``compact_blocks=True`` on canonical uncoupled ``block_glk``, both the algebra embedding
    and the returned factors retain ``(..., N, H, d, d)`` storage; every other group and the
    default keep the legacy dense-vertex :class:`FactoredTransport` representation.
    The pairwise contraction is deferred into ``transport_mean`` / ``transport_covariance``'s fast
    path (P0 #2). Caller guards this to the flat + block-diagonal-with-equal-blocks path; here it
    only requires the exps, which the block-diagonal exp machinery already produces. Rank-agnostic
    via the leading ellipsis: a (B, N, n_gen) frame (batched forward) and a (N, n_gen) frame (the
    unbatched block / diagnostics path) both flow through.
    """
    block_dims = group.irrep_dims if len(group.irrep_dims) > 1 else None
    compact = (
        compact_blocks
        and group.phi_coordinate_layout == "block_head_row_major"
        and block_dims is not None
        and len(set(block_dims)) == 1
        and group.generators.shape[0] == len(block_dims) * block_dims[0] * block_dims[0]
    )
    if gauge_mode == "trivial":
        # Trivial gauge: exp = I. Build the same per-token factors the dense path would (the
        # caller's guard normally excludes trivial, but keep the container well-formed).
        K = group.generators.shape[-1]
        if compact:
            H, d = len(block_dims), block_dims[0]
            eye_d = torch.eye(d, device=phi.device, dtype=phi.dtype)
            eye = eye_d.expand(*phi.shape[:-1], H, d, d).contiguous()
            return CompactFactoredTransport(
                exp_blocks=eye, inv_blocks=eye, K=K, mean_per_head=mean_per_head,
                same_frame_flat_cocycle=True)
        eye_K = torch.eye(K, device=phi.device, dtype=phi.dtype)
        eye = eye_K.expand(*phi.shape[:-1], K, K).contiguous()
        return FactoredTransport(exp_phi=eye, exp_neg_phi=eye, irrep_dims=list(group.irrep_dims),
                                 mean_per_head=mean_per_head, same_frame_flat_cocycle=True)
    if gauge_mode != "learned":
        raise ValueError(f"gauge_mode must be 'learned' or 'trivial', got {gauge_mode!r}")

    if compact:
        H, d = len(block_dims), block_dims[0]
        # generate_glk_multihead orders the canonical uncoupled block_glk coordinates as
        # H consecutive row-major d x d matrices. The reshape is therefore the exact algebra
        # embedding with the structural zero off-block entries omitted.
        phi_blocks = phi.reshape(*phi.shape[:-1], H, d, d)
        if phi_blocks.dtype == torch.float32 and torch.is_autocast_enabled(phi.device.type):
            # The legacy einsum(phi, generators) is an autocast-eligible contraction. A reshape
            # performs no arithmetic and would otherwise leave packed factors in fp32, changing
            # both values and gradients under AMP. Mirror the contraction's public output dtype
            # without allocating the dense K x K embedding.
            phi_blocks = phi_blocks.to(torch.get_autocast_dtype(phi.device.type))
        exp_blocks, inv_blocks = _stable_compact_glk_exp_pair(
            phi_blocks,
            exp_fp64_mode=exp_fp64_mode,
            exp_fp64_norm_threshold=exp_fp64_norm_threshold,
            clamp_monitor=clamp_monitor,
            validity_max_norm=validity_max_norm,
        )
        if right_phi is not None:
            right_blocks = right_phi.reshape(*right_phi.shape[:-1], H, d, d)
            if right_blocks.dtype == torch.float32 and torch.is_autocast_enabled(phi.device.type):
                right_blocks = right_blocks.to(torch.get_autocast_dtype(phi.device.type))
            right_exp, right_inv = _stable_compact_glk_exp_pair(
                right_blocks,
                exp_fp64_mode=exp_fp64_mode,
                exp_fp64_norm_threshold=exp_fp64_norm_threshold,
                clamp_monitor=clamp_monitor,
                validity_max_norm=validity_max_norm,
            )
            exp_blocks = exp_blocks @ right_exp
            inv_blocks = right_inv @ inv_blocks
        return CompactFactoredTransport(
            exp_blocks=exp_blocks, inv_blocks=inv_blocks, K=group.generators.shape[-1],
            mean_per_head=mean_per_head, same_frame_flat_cocycle=True)

    phi_matrix = torch.einsum("...na,aij->...nij", phi, group.generators)
    # exp_dim: same block-scale float64-island keying as compute_transport_operators (see the
    # comment there) -- the factored hot path is exactly where the (B, N, K, K) f64 upcast hurt.
    exp_phi, exp_neg_phi = stable_matrix_exp_pair(
        phi_matrix, skew_symmetric=group.skew_symmetric, block_dims=block_dims,
        exp_dim=(max(block_dims) if block_dims is not None else None),
        exp_fp64_mode=exp_fp64_mode, exp_fp64_norm_threshold=exp_fp64_norm_threshold,
        # m16: matrix_exp of a skew matrix is exactly orthogonal at ANY norm, so the Frobenius clamp
        # only gratuitously shortens the rotation on the pure so_n/so_k tower path (whose retraction
        # caps ||phi|| in COORDINATES, under-bounding the embedded norm). Disable it for skew; the
        # non-compact groups (glk/block_glk/sp_n) keep the clamp as a genuine exp-overflow safeguard.
        max_norm=(float("inf") if group.skew_symmetric else TRANSPORT_CLAMP_MAX_NORM),
        clamp_monitor=clamp_monitor,
        validity_max_norm=validity_max_norm,
    )
    if right_phi is not None:
        right_matrix = torch.einsum("...na,aij->...nij", right_phi, group.generators)
        right_exp, right_inv = stable_matrix_exp_pair(
            right_matrix,
            skew_symmetric=group.skew_symmetric,
            block_dims=block_dims,
            exp_dim=(max(block_dims) if block_dims is not None else None),
            exp_fp64_mode=exp_fp64_mode,
            exp_fp64_norm_threshold=exp_fp64_norm_threshold,
            max_norm=(float("inf") if group.skew_symmetric else TRANSPORT_CLAMP_MAX_NORM),
            clamp_monitor=clamp_monitor,
            validity_max_norm=validity_max_norm,
        )
        exp_phi = exp_phi @ right_exp
        exp_neg_phi = right_inv @ exp_neg_phi
    return FactoredTransport(exp_phi=exp_phi, exp_neg_phi=exp_neg_phi, irrep_dims=list(group.irrep_dims),
                             mean_per_head=mean_per_head, same_frame_flat_cocycle=True)


def _certifies_same_frame_flat_cocycle(
    omega: 'torch.Tensor | CompactFactoredTransport | DirectLinkTransport | FactoredTransport | RopeTransport',
) -> bool:
    r"""Whether a transport certifies an analytically identity self map."""
    if isinstance(omega, (CompactFactoredTransport, FactoredTransport)):
        return omega.same_frame_flat_cocycle
    if isinstance(omega, RopeTransport):
        return (
            omega.same_frame_flat_cocycle
            and _certifies_same_frame_flat_cocycle(omega.base)
        )
    return False


def _restore_certified_self_links_(
    output: torch.Tensor,  # (..., N, N, *event) already allocated transported output
    source: torch.Tensor,  # (..., N, *event) source values
    omega:  'torch.Tensor | CompactFactoredTransport | DirectLinkTransport | FactoredTransport | RopeTransport',

    *,
    event_ndim: int,
) -> torch.Tensor:
    r"""Overwrite the diagonal view for a certified same-frame flat cocycle in place."""
    if not _certifies_same_frame_flat_cocycle(omega):
        return output
    query_axis = -(event_ndim + 2)
    key_axis = -(event_ndim + 1)
    source_token_axis = -(event_ndim + 1)
    if output.shape[query_axis] != output.shape[key_axis]:
        return output
    if source.shape[source_token_axis] != output.shape[key_axis]:
        return output
    diagonal = output.diagonal(offset=0, dim1=query_axis, dim2=key_axis)
    diagonal = diagonal.movedim(-1, source_token_axis)
    if diagonal.shape != source.shape:
        return output
    diagonal.copy_(source)
    return output


def transport_mean(
    omega: 'torch.Tensor | CompactFactoredTransport | DirectLinkTransport | FactoredTransport | RopeTransport',
    mu:    torch.Tensor,                                          # (..., N, K) source (key, index j) means
) -> torch.Tensor:
    r"""Gauge action on means: mu_t[i,j] = Omega_ij @ mu_j. Returns (..., N, N, K).

    Dense path (rank-agnostic via the leading ellipsis): an optional batch axis (B,N,N,K,K)+(B,N,K)
    flows through unchanged, and the unbatched (N,N,K,K)+(N,K) call is identical -- so the
    same primitive serves the batched forward and the unbatched diagnostics path.

    FACTORED path (``omega`` is a :class:`FactoredTransport`, the flat + block fast route): the exps
    are fused into the contraction -- compute m_j = exp(-phi_j) @ mu_j ONCE (B,N,K), then
    mu_t[i,j] = exp(phi_i) @ m_j -- an EXACT reassociation of the dense einsum (round-off level),
    never forming (B,N,N,K,K). Autograd-safe (differentiates through the live exps), so it survives
    the smoothing-mode oracle if a container ever reaches it. When the container carries
    ``mean_per_head=True`` (Tier-1 toggle, default False) the contraction runs per gauge block
    instead (:func:`_factored_per_head_mean`, fp32 reassociation only); a RoPE-wrapped factored
    base recurses through this same dispatch, so the toggle covers it too.

    ROPETRANSPORT path (``omega`` is a :class:`RopeTransport`): the gauge-RoPE rotation R(theta) is
    applied as R_i Omega_ij R_j^T mu_j -- pre-rotate the key mean by R_j^T, transport on the
    un-rotated base, post-rotate by R_i.
    """
    if isinstance(omega, RopeTransport):
        # mu_t[i,j] = R_i Omega_ij R_j^T mu_j: pre-rotate the key mean by R_j^T, transport on the
        # un-rotated base, post-rotate the result by R_i. R_j^T mu_j = sum_l R[j,l,k] mu[j,l].
        m = torch.einsum("...jlk,...jl->...jk", omega.rope, mu)        # (..., N, K)
        t = transport_mean(omega.base, m)                             # (..., N, N, K)
        out = torch.einsum("...ikl,...ijl->...ijk", omega.rope, t)    # post-rotate by R_i
        return _restore_certified_self_links_(out, mu, omega, event_ndim=1)
    if isinstance(omega, DirectLinkTransport):
        return _direct_link_mean(omega, mu)
    if isinstance(omega, CompactFactoredTransport):
        out = _compact_factored_mean(omega, mu)
        return _restore_certified_self_links_(out, mu, omega, event_ndim=1)
    if isinstance(omega, FactoredTransport):
        if omega.mean_per_head:
            out = _factored_per_head_mean(omega, mu)
        else:
            m = torch.einsum("...jlp,...jp->...jl", omega.exp_neg_phi, mu)  # (..., N, K): exp(-phi_j) @ mu_j
            out = torch.einsum("...ikl,...jl->...ijk", omega.exp_phi, m)    # (..., N, N, K): exp(phi_i) @ m_j
        return _restore_certified_self_links_(out, mu, omega, event_ndim=1)
    return torch.einsum("...ijkl,...jl->...ijk", omega, mu)


def transport_covariance(
    omega: 'torch.Tensor | CompactFactoredTransport | DirectLinkTransport | FactoredTransport | RopeTransport',
    sigma: torch.Tensor,                                          # (..., N, K) diagonal OR (..., N, K, K) full

    *,
    diagonal_out: Optional[bool] = None,
) -> torch.Tensor:
    r"""Sandwich action Sigma_t[i,j] = Omega_ij Sigma_j Omega_ij^T.

    Full input (...,N,K,K) -> full (...,N,N,K,K). Diagonal input (...,N,K) -> the
    diagonal approximation (...,N,N,K), Sigma_t[i,j,k] = sum_l Omega_ijkl^2
    sigma_jl (the diagonal of the full sandwich). Rank-agnostic via the leading
    ellipsis (optional batch axis); diagonal vs full is detected by the rank gap
    ``sigma.dim() == omega.dim() - 2``, which holds with or without the batch axis.

    FACTORED path (``omega`` is a :class:`FactoredTransport`): a DIAGONAL sigma is sandwiched
    per head -- by block-diagonality the (d, d) block Omega^(h) = exp(phi_i)^(h) exp(-phi_j)^(h)
    is the only nonzero part on head h's coordinates, so Sigma_t[i,j,k] = sum_l (Omega^(h)_ijkl)^2
    sigma_jl runs over head h only (the off-block Omega entries are exactly 0.0, so the full-l sum
    equals the in-block sum). This materializes H * d^2 per pair, never the full K^2 square -- the
    square sits inside the l-sum, so it does NOT factor by squaring the full exps; it factors by
    block-diagonality. A FULL sigma rebuilds the dense Omega (byte-identical) and runs the unchanged
    sandwich, so full covariance is never the round-off factoring.

    ROPETRANSPORT path (``omega`` is a :class:`RopeTransport`): means-only (``on_cov=False``) uses
    the un-rotated base covariance; full-gauge (``on_cov=True``) applies the RoPE congruence. A
    direct-link base remains factored through that congruence; other bases retain their established
    dense or compact dispatch.
    """
    if isinstance(omega, RopeTransport):
        if not omega.on_cov:
            return transport_covariance(omega.base, sigma, diagonal_out=diagonal_out)   # mu-only
        if isinstance(omega.base, DirectLinkTransport):
            is_diag = _direct_link_is_diagonal(omega.base, sigma, diagonal_out)
            if omega.base.exp_phi is None:
                exp_phi = omega.rope
                exp_neg_phi = omega.rope.transpose(-1, -2)
            else:
                exp_phi = torch.einsum(
                    "...ikl,...ilm->...ikm", omega.rope, omega.base.exp_phi)
                exp_neg_phi = torch.einsum(
                    "...ikl,...iml->...ikm", omega.base.exp_neg_phi, omega.rope)
            # Fold R_i and R_j^T into the optional vertices, preserving the direct-link diagonal
            # contraction instead of promoting diagonal sigma to a pairwise full covariance.
            rotated = DirectLinkTransport(
                exp_link=omega.base.exp_link,
                exp_phi=exp_phi,
                exp_neg_phi=exp_neg_phi,
            )
            return transport_covariance(rotated, sigma, diagonal_out=is_diag)
        if isinstance(omega.base, CompactFactoredTransport):
            blocks = _equal_diag_blocks(
                omega.rope, omega.base.n_blocks, omega.base.block_dim)
            exp_blocks = torch.einsum(
                "...nhkl,...nhlm->...nhkm", blocks, omega.base.exp_blocks)
            inv_blocks = torch.einsum(
                "...nhkl,...nhml->...nhkm", omega.base.inv_blocks, blocks)
            rotated = CompactFactoredTransport(
                exp_blocks, inv_blocks, omega.base.K,
                mean_per_head=omega.base.mean_per_head,
                same_frame_flat_cocycle=_certifies_same_frame_flat_cocycle(omega))
            return transport_covariance(rotated, sigma, diagonal_out=diagonal_out)
        # Other full-gauge bases use the established rotated dense operator.
        out = transport_covariance(_rope_dense_omega(omega.base, omega.rope), sigma,
                                   diagonal_out=diagonal_out)
        if isinstance(omega.base, FactoredTransport):
            is_diag = (
                sigma.dim() == omega.base.exp_phi.dim() - 1
                if diagonal_out is None else diagonal_out
            )
            return _restore_certified_self_links_(
                out, sigma, omega, event_ndim=(1 if is_diag else 2))
        return out
    if isinstance(omega, DirectLinkTransport):
        if _direct_link_is_diagonal(omega, sigma, diagonal_out):
            return _direct_link_diagonal_covariance(omega, sigma)
        return _direct_link_full_covariance(omega, sigma)
    if isinstance(omega, CompactFactoredTransport):
        is_diag = (
            sigma.dim() == omega.exp_blocks.dim() - 2
            if diagonal_out is None else diagonal_out
        )
        if is_diag:
            out = _compact_factored_diagonal_covariance(omega, sigma)
        else:
            out = _compact_factored_full_covariance(omega, sigma)
        return _restore_certified_self_links_(
            out, sigma, omega, event_ndim=(1 if is_diag else 2))
    if isinstance(omega, FactoredTransport):
        # Diagonal sigma is (..., N, K) -> same rank as exp_phi minus the trailing K axis; a full
        # sigma is (..., N, K, K) -> same rank as exp_phi (the dense-Omega rank-gap is +1 here
        # because the factored exps carry one fewer N axis than the dense (..., N, N, K, K)).
        is_diag = sigma.dim() == omega.exp_phi.dim() - 1 if diagonal_out is None else diagonal_out
        if is_diag:
            out = _factored_diagonal_covariance(omega, sigma)
        else:
            out = _factored_full_covariance(omega, sigma)
        return _restore_certified_self_links_(
            out, sigma, omega, event_ndim=(1 if is_diag else 2))
    is_diag = sigma.dim() == omega.dim() - 2 if diagonal_out is None else diagonal_out
    if is_diag:
        return torch.einsum("...ijkl,...ijkl,...jl->...ijk", omega, omega, sigma)
    # Rank-gap dispatch hardening (audit 2026-07-05 m7): a batch-INDEPENDENT (N, N, K, K) omega with
    # a BATCHED diagonal (B, N, K) sigma satisfies sigma.dim() == omega.dim() - 1 and previously fell
    # through to the full-covariance einsum -- a shape error at best, a silent mis-broadcast when
    # B == N. A genuine full sigma is (..., N, K, K): trailing SQUARE pair matching omega's K and
    # exactly one rank below omega. Validate before contracting; ambiguous callers must pass
    # ``diagonal_out`` explicitly (as the kernel/oracle call sites already do).
    if diagonal_out is None and (
        sigma.dim() != omega.dim() - 1
        or sigma.shape[-1] != sigma.shape[-2]
        or sigma.shape[-1] != omega.shape[-1]
    ):
        raise ValueError(
            f"transport_covariance: sigma shape {tuple(sigma.shape)} is neither a diagonal "
            f"(..., N, K) (rank omega.dim()-2) nor a full (..., N, K, K) (rank omega.dim()-1, "
            f"trailing square K={omega.shape[-1]}) match for omega shape {tuple(omega.shape)}. "
            f"Pass diagonal_out=True/False explicitly to disambiguate (e.g. a batched diagonal "
            f"sigma against a batch-independent omega)."
        )
    # Full-covariance congruence sandwich Omega Sigma Omega^T SQUARES cond(Omega) (audit 2026-06-13
    # M4). Evaluate the contraction in a float64 island (like the matrix-exp upcast) then cast back:
    # this CORRECTLY-ROUNDS the sandwich (removes the fp32 sum-over-l,m accumulation error), so the
    # fp32-stored result is the best fp32 representation of the true sandwich. NOTE this does not
    # rescue the EXTREME regime: for the non-compact groups (glk/block_glk/sp_n) cond(Omega) ~
    # exp(2||phi||) can reach ~1e6 at the retraction's default max_norm=5, and the squared sandwich
    # (~1e12) is then unrepresentable in fp32 STORAGE regardless of compute precision -- bound ||phi||
    # or use a compact group / diagonal family there. Reached only on the full-covariance path
    # (family='gaussian_full'); the diagonal default above and the compact (orthogonal Omega, cond=1)
    # groups are untouched, so the hot path is unchanged.
    out = torch.einsum("...ijkl,...jlm,...ijnm->...ijkn",
                       omega.double(), sigma.double(), omega.double())
    return out.to(sigma.dtype)


def transport_scale(
    scale: torch.Tensor,   # (..., N, K) independent location-scale-family marginal scales
    omega: 'torch.Tensor | CompactFactoredTransport | DirectLinkTransport | FactoredTransport | RopeTransport',

    *,
    diagonal_out: Optional[bool] = True,
) -> torch.Tensor:         # (..., N, N, K) variance-matching marginal scales
    r"""Degree-one transport for a factorized location-scale family's marginal scale.

    A signed diagonal/permutation operator maps independent scales exactly as
    ``b'_k = |Omega_kk| b_k``. For a general mixing operator the push-forward of independent
    Laplace coordinates is not factorized Laplace, so this seam returns the explicit
    variance-matching marginal projection

        b'_k = sqrt(sum_l Omega_kl^2 b_l^2).

    This projection is homogeneous of degree one and reduces to the exact ``|Omega| b`` law on
    the family-preserving subgroup. It is not a claim of distributional closure off that subgroup.
    The covariance transport owns every dense/factored/direct-link/RoPE dispatch, so applying it
    to the variance proxy ``b^2`` keeps those optimized paths shared without treating ``b`` itself
    as a degree-two variance.
    """
    variance_proxy = transport_covariance(
        omega,
        scale.square(),
        diagonal_out=diagonal_out,
    )
    variance_proxy = variance_proxy.clamp_min(0.0)
    positive = variance_proxy > 0.0
    safe_variance = torch.where(positive, variance_proxy, torch.ones_like(variance_proxy))
    return torch.where(positive, safe_variance.sqrt(), torch.zeros_like(safe_variance))


def _direct_link_mean(
    direct: DirectLinkTransport,
    mu:     torch.Tensor,               # (..., N, K) source means
) -> torch.Tensor:                      # (..., N, N, K) transported means
    r"""Contract direct edge and optional vertex factors without pairwise ``Omega``."""
    if direct.exp_phi is None:
        return torch.einsum("ijkl,...jl->...ijk", direct.exp_link, mu)
    key = torch.einsum("...jlp,...jp->...jl", direct.exp_neg_phi, mu)
    linked = torch.einsum("ijlm,...jm->...ijl", direct.exp_link, key)
    return torch.einsum("...ikl,...ijl->...ijk", direct.exp_phi, linked)


def _direct_link_is_diagonal(
    direct:       DirectLinkTransport,
    sigma:        torch.Tensor,
    diagonal_out: Optional[bool],
) -> bool:
    if diagonal_out is not None:
        return diagonal_out
    if direct.exp_phi is not None:
        return sigma.dim() == direct.exp_phi.dim() - 1
    # The bare link is batch-independent. As on the legacy dense route, a batched diagonal/full
    # ambiguity must be resolved by the caller's explicit ``diagonal_out`` flag.
    return sigma.dim() == direct.exp_link.dim() - 2


def _direct_link_key_covariance(
    direct: DirectLinkTransport,
    sigma:  torch.Tensor,               # (..., N, K) diagonal or (..., N, K, K) full

    *,
    diagonal: bool,
) -> torch.Tensor:                      # (..., N, K, K) key covariance after exp_neg_phi
    if direct.exp_neg_phi is None:
        return torch.diag_embed(sigma) if diagonal else sigma
    if diagonal:
        return torch.einsum(
            "...jkl,...jml,...jl->...jkm",
            direct.exp_neg_phi,
            direct.exp_neg_phi,
            sigma,
        )
    return torch.einsum(
        "...jkl,...jlm,...jnm->...jkn",
        direct.exp_neg_phi,
        sigma,
        direct.exp_neg_phi,
    )


def _direct_link_diagonal_covariance(
    direct: DirectLinkTransport,
    sigma:  torch.Tensor,               # (..., N, K) diagonal variances
) -> torch.Tensor:                      # (..., N, N, K) diagonal congruence
    r"""Diagonal congruence from live edge/vertex factors without pairwise ``Omega``."""
    if direct.exp_phi is None:
        return torch.einsum(
            "ijka,ijka,...ja->...ijk",
            direct.exp_link,
            direct.exp_link,
            sigma,
        )
    key_cov = _direct_link_key_covariance(direct, sigma, diagonal=True)
    # One output row at a time keeps the live pairwise object at (...,N,N,K), never
    # (...,N,N,K,K). Each row is (exp_phi_i[k,:] exp_link_ij), then r C_j r^T.
    parts: List[torch.Tensor] = []
    for k in range(direct.exp_link.shape[-1]):
        linked_row = torch.einsum(
            "...ia,ijab->...ijb", direct.exp_phi[..., k, :], direct.exp_link)
        parts.append(torch.einsum(
            "...ijb,...jbc,...ijc->...ij", linked_row, key_cov, linked_row))
    return torch.stack(parts, dim=-1)


def _direct_link_full_covariance(
    direct: DirectLinkTransport,
    sigma:  torch.Tensor,               # (..., N, K, K) full covariances
) -> torch.Tensor:                      # (..., N, N, K, K) full congruence output
    r"""Full direct-link congruence; the output is dense but no pairwise operator is built."""
    if direct.exp_neg_phi is None:
        key_cov = sigma.double()
    else:
        exp_neg_phi = direct.exp_neg_phi.double()
        key_cov = torch.einsum(
            "...jkl,...jlm,...jnm->...jkn",
            exp_neg_phi,
            sigma.double(),
            exp_neg_phi,
        )
    exp_link = direct.exp_link.double()
    edge_cov = torch.einsum(
        "ijka,...jab,ijnb->...ijkn",
        exp_link,
        key_cov,
        exp_link,
    )
    if direct.exp_phi is None:
        out = edge_cov
    else:
        exp_phi = direct.exp_phi.double()
        out = torch.einsum(
            "...ika,...ijab,...inb->...ijkn", exp_phi, edge_cov, exp_phi)
    return out.to(sigma.dtype)


def _compact_factored_mean(
    factored: CompactFactoredTransport,
    mu:       torch.Tensor,               # (..., N, K) source means
) -> torch.Tensor:                        # (..., N, N, K) transported means
    r"""Mean transport over ``H`` compact ``d x d`` blocks, never a dense ``K x K`` factor."""
    H, d = factored.n_blocks, factored.block_dim
    mu_blocks = mu.reshape(*mu.shape[:-1], H, d)                         # (..., N, H, d)
    key = torch.einsum(
        "...jhlp,...jhp->...jhl", factored.inv_blocks, mu_blocks)
    out = torch.einsum(
        "...ihkl,...jhl->...ijhk", factored.exp_blocks, key)
    return out.reshape(*out.shape[:-2], factored.K)


def _compact_pair_blocks(
    factored: CompactFactoredTransport,
) -> torch.Tensor:                        # (..., N, N, H, d, d) pairwise block operators
    r"""Build only pairwise ``d x d`` blocks ``U_i^(h) U_j^(-1,h)``."""
    return torch.einsum(
        "...ihkl,...jhlm->...ijhkm", factored.exp_blocks, factored.inv_blocks)


def _compact_factored_diagonal_covariance(
    factored: CompactFactoredTransport,
    sigma:    torch.Tensor,               # (..., N, K) diagonal variances
) -> torch.Tensor:                        # (..., N, N, K) diagonal sandwich
    r"""Diagonal congruence over compact blocks without a dense ``K x K`` operator.

    First form each key-side second moment
    ``C_j = U_j^-1 diag(sigma_j) U_j^-T`` and contract it with each query factor. This avoids the
    ``(..., N_q, N_k, H, d, d)`` pair-block allocation for every block and sequence shape.
    """
    H, d = factored.n_blocks, factored.block_dim
    sigma_blocks = sigma.reshape(*sigma.shape[:-1], H, d)                 # (..., N, H, d)
    key_second = torch.einsum(
        "...jhlp,...jhmp,...jhp->...jhlm",
        factored.inv_blocks, factored.inv_blocks, sigma_blocks)           # (...,Nk,H,d,d)
    out = torch.einsum(
        "...ihkl,...jhlm,...ihkm->...ijhk",
        factored.exp_blocks, key_second, factored.exp_blocks)
    return out.reshape(*out.shape[:-2], factored.K)


def _compact_factored_full_covariance(
    factored: CompactFactoredTransport,
    sigma:    torch.Tensor,               # (..., N, K, K) full SPD covariances
) -> torch.Tensor:                        # (..., N, N, K, K) full congruence output
    r"""Full congruence from compact blocks; the output is full but no dense operator is built."""
    H, d = factored.n_blocks, factored.block_dim
    sigma_blocks = sigma.reshape(*sigma.shape[:-3], sigma.shape[-3], H, d, H, d)
    omega_blocks = _compact_pair_blocks(factored).double()
    out = torch.einsum(
        "...ijhkl,...jhlgm,...ijgnm->...ijhkgn",
        omega_blocks, sigma_blocks.double(), omega_blocks)
    return out.reshape(*out.shape[:-4], factored.K, factored.K).to(sigma.dtype)


def _factored_per_head_mean(
    factored: FactoredTransport,
    mu:       torch.Tensor,               # (..., N, K) source (key, index j) means
) -> torch.Tensor:                        # (..., N, N, K) transported means
    r"""Per-head mean transport from the factored exps (the mean twin of
    ``_factored_diagonal_covariance``; active production transport numerics).

    For each head h on coordinates [start:end] the block Omega^(h)_ij = exp(phi_i)^(h) exp(-phi_j)^(h)
    is the only nonzero part of Omega on head h's rows (the off-block entries are exactly 0.0), so

        mu_t[i,j]^(h) = exp(phi_i)^(h) ( exp(-phi_j)^(h) mu_j^(h) ),

    and the full-K contraction equals the concatenation of the per-head (d, d) contractions -- the
    same sum with the exactly-zero off-block terms dropped, ~H x fewer FLOPs on the dominant pair
    GEMM. Equal to the dense-K einsum up to fp32 reassociation (pinned allclose atol 1e-6 by
    tests/test_tier12_transport.py). Rank-agnostic via the leading ellipsis.
    """
    parts: List[torch.Tensor] = []
    start = 0
    for d in factored.irrep_dims:
        end    = start + d
        ep     = factored.exp_phi[..., start:end, start:end]       # (..., N, d, d) exp(phi_i)^(h)
        en     = factored.exp_neg_phi[..., start:end, start:end]   # (..., N, d, d) exp(-phi_j)^(h)
        mu_blk = mu[..., start:end]                                # (..., N, d)
        m = torch.einsum("...jlp,...jp->...jl", en, mu_blk)        # (..., N, d): exp(-phi_j)^(h) mu_j^(h)
        parts.append(torch.einsum("...ikl,...jl->...ijk", ep, m))  # (..., N, N, d): exp(phi_i)^(h) m_j
        start = end
    return torch.cat(parts, dim=-1)                                # (..., N, N, K)


def _factored_diagonal_covariance(
    factored: FactoredTransport,
    sigma:    torch.Tensor,               # (..., N, K) diagonal variances
) -> torch.Tensor:                        # (..., N, N, K) diagonal sandwich
    r"""Per-head diagonal sandwich from the factored exps (P0 #2 covariance route).

    For each head h on coordinates [start:end] the block Omega^(h)_ij = exp(phi_i)^(h) exp(-phi_j)^(h)
    is a (d, d) operator and the diagonal sandwich is the quadratic form

        Sigma_t[i,j,k] = sum_l (Omega^(h)_ijkl)^2 sigma_jl
                       = sum_{m,n} ep_i[k,m] G_j[m,n] ep_i[k,n],   G_j = en_j diag(sigma_j) en_j^T,

    so the per-PAIR (..., N, N, d, d) block Omega never needs to exist (audit 2026-06-09 P3): the
    key-side second moment G_j is (..., N, d, d), the query-side outer product ep_i[k,m] ep_i[k,n]
    is (..., N, d, d, d), and the contraction lands directly on the (..., N, N, d) output -- peak
    memory N d^3 + N^2 d instead of N^2 d^2, at identical flop count. Exact w.r.t. the dense
    diagonal sandwich because the dense Omega is block-diagonal (off-block entries exactly 0.0);
    the regrouping is algebraically exact (rounding differs at fp32 epsilon, covered by the
    factored-vs-dense allclose pins). Rank-agnostic via the leading ellipsis.
    """
    parts: List[torch.Tensor] = []
    start = 0
    n_tokens = sigma.shape[-2]
    for d in factored.irrep_dims:
        end = start + d
        ep = factored.exp_phi[..., start:end, start:end]               # (..., N, d, d) exp(phi_i)^(h)
        en = factored.exp_neg_phi[..., start:end, start:end]           # (..., N, d, d) exp(-phi_j)^(h)
        sig_blk = sigma[..., start:end]                                # (..., N, d)
        if d <= n_tokens:
            g = torch.einsum("...jml,...jnl,...jl->...jmn", en, en, sig_blk)   # (..., N, d, d) G_j
            ep2 = ep.unsqueeze(-1) * ep.unsqueeze(-2)                  # (..., N, d, d, d) ep[k,m] ep[k,n]
            parts.append(torch.einsum("...ikmn,...jmn->...ijk", ep2, g))   # (..., N, N, d)
        else:
            # Tall-block regime d > N (audit 2026-06-09 overnight F4): the query-side outer
            # product N d^3 would EXCEED the per-pair dense block's N^2 d^2 there, so rebuild
            # the (exactly equivalent) per-pair block Omega and run the squared diagonal
            # sandwich; for d <= N (the usual case) the factored route above stays the
            # memory winner (N d^3 + N^2 d < N^2 d^2).
            omega_blk = torch.einsum("...ikm,...jml->...ijkl", ep, en)  # (..., N, N, d, d)
            parts.append(torch.einsum("...ijkl,...ijkl,...jl->...ijk",
                                      omega_blk, omega_blk, sig_blk))
        start = end
    return torch.cat(parts, dim=-1)                                    # (..., N, N, K)


def _factored_full_covariance(
    factored: FactoredTransport,
    sigma:    torch.Tensor,               # (..., N, K, K) full SPD covariances
) -> torch.Tensor:                        # (..., N, N, K, K) full congruence sandwich
    r"""Per-head full-covariance congruence sandwich Sigma_t[i,j] = Omega_ij Sigma_j Omega_ij^T from
    the factored exps, WITHOUT materializing the dense (..., N, N, K, K) Omega (the full-cov twin of
    ``_factored_diagonal_covariance``).

    For a block-diagonal gauge (block_glk: Omega = exp(phi_i) exp(-phi_j) is block-diagonal with
    EQUAL d x d blocks Omega^(h)_ij = exp(phi_i)^(h) exp(-phi_j)^(h)), the (h, h') output block is

        Sigma_t[i,j]^(h,h') = Omega^(h)_ij  Sigma_j^(h,h')  (Omega^(h')_ij)^T,

    because the off-block entries of Omega are exactly 0 (Higham, block-diagonal product), so only the
    per-head (d, d) blocks of Omega ever multiply. The OUTPUT is still the full (..., N, N, K, K)
    sandwich (a full Sigma's off-diagonal blocks survive, mapped by Omega^(h) on the left and
    Omega^(h') on the right -- exactly what the dense path computes). For equal block sizes, one
    batched float64 einsum carries both head axes and fuses the vertex factors, avoiding H^2 Python
    dispatches and the dense (..., N, N, K, K) Omega. Heterogeneous irrep dimensions retain the
    exact block-pair fallback because their rectangular blocks cannot share one head tensor.
    Value-equal to the dense ``transport_covariance(to_dense_omega(), sigma)``; pinned by
    tests/test_fullcov_alpha_roadmap_2026_06_13.py.
    """
    dims = factored.irrep_dims
    K = sum(dims)
    batch = sigma.shape[:-3]
    N = sigma.shape[-3]

    if len(set(dims)) == 1:
        H, d = len(dims), dims[0]
        exp_phi = _equal_diag_blocks(factored.exp_phi, H, d).double()       # (..., N, H, d, d)
        exp_neg = _equal_diag_blocks(factored.exp_neg_phi, H, d).double()  # (..., N, H, d, d)
        sigma_blocks = sigma.reshape(*batch, N, H, d, H, d).double()
        # One batched contraction over both explicit head axes. Algebraically this is
        # Omega_h Sigma_(h,g) Omega_g^T with Omega_h = exp_phi_h exp_neg_h, but fusing the
        # vertex factors avoids both the H^2 Python loop and H separate pair-operator builds.
        out = torch.einsum(
            "...ihax,...jhxb,...jhbgd,...igcy,...jgyd->...ijhagc",
            exp_phi,
            exp_neg,
            sigma_blocks,
            exp_phi,
            exp_neg,
        )
        return out.reshape(*batch, N, N, K, K).to(sigma.dtype)

    out = sigma.new_zeros(*batch, N, N, K, K)

    # Per-head pairwise operators Omega^(h)_ij = exp(phi_i)^(h) @ exp(-phi_j)^(h), each (..., N, N, d, d):
    # the per-head diagonal blocks of the dense Omega (its off-blocks are zero), never the dense K x K.
    blocks = []
    start = 0
    for d in dims:
        end = start + d
        ep = factored.exp_phi[..., start:end, start:end]               # (..., N, d, d) exp(phi_i)^(h)
        en = factored.exp_neg_phi[..., start:end, start:end]           # (..., N, d, d) exp(-phi_j)^(h)
        omega_h = torch.einsum("...ikl,...jlm->...ijkm", ep, en)       # (..., N, N, d, d)
        blocks.append((start, end, omega_h.double()))                  # float64 island (M4); cast ONCE per head
        start = end

    # Heterogeneous irrep dimensions cannot share one rectangular head tensor. Keep the exact
    # block-pair fallback: (h, h') output block = Omega^(h)_ij Sigma_j^(h,h') (Omega^(h')_ij)^T,
    # in a float64 island (the
    # congruence squares cond(Omega); audit M4) at the block scale, then cast back. The per-head
    # operators are pre-cast to float64 above (H casts) rather than re-cast inside this H x H loop
    # (which would be O(H^2) redundant casts of the same oh/oh2); only the distinct per-pair sigma
    # block is cast here.
    for s1, e1, oh in blocks:
        for s2, e2, oh2 in blocks:
            sig_blk = sigma[..., s1:e1, s2:e2]                          # (..., N, d1, d2) key-side block
            res = torch.einsum("...ijkl,...jlm,...ijnm->...ijkn",
                               oh, sig_blk.double(), oh2)
            out[..., s1:e1, s2:e2] = res.to(sigma.dtype)
    return out
