r"""Mixed AdamW and stateless local-group updates for stored gauge frames.

Ordinary parameter groups retain standard AdamW. A stored phi-factor group selected by the
``pullback_group`` policy is staged in float64 through the strict pullback direction, trust scaling,
order-four BCH, and exact right-product validation. Every phi candidate is validated before any phi
table is mutated; committed gradients are consumed so base AdamW cannot apply a second update.
Stored ``omega_direct`` elements retain their established group-retraction and cadence state.
"""

import math
import warnings
from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, List, Mapping, Optional

import torch

from vfe3.geometry.groups import GaugeGroup
from vfe3.geometry.phi_preconditioner import (
    PullbackGroupDirectionResult,
    pullback_group_direction,
)


_PHI_PROJECTION_TEMPORARY_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class PhiUpdatePolicy:
    optimizer_group_metadata: Mapping[str, object]
    requires_manifold_optimizer: bool
    requires_pullback_geometry: bool = False


_PHI_UPDATE_POLICIES: Dict[str, PhiUpdatePolicy] = {}


def register_phi_update_policy(
    name:   str,
    policy: PhiUpdatePolicy,

    *,
    override: bool = False,
) -> None:
    """Register immutable phi-update routing metadata under a duplicate-safe key."""
    if name in _PHI_UPDATE_POLICIES and not override:
        raise KeyError(
            f"phi update policy {name!r} already registered; pass override=True to replace"
        )
    _PHI_UPDATE_POLICIES[name] = PhiUpdatePolicy(
        optimizer_group_metadata=MappingProxyType(dict(policy.optimizer_group_metadata)),
        requires_manifold_optimizer=bool(policy.requires_manifold_optimizer),
        requires_pullback_geometry=bool(policy.requires_pullback_geometry),
    )


def get_phi_update_policy(name: str) -> PhiUpdatePolicy:
    """Return immutable routing metadata for the selected phi update."""
    if name not in _PHI_UPDATE_POLICIES:
        raise KeyError(
            f"no phi update policy {name!r}; available: {sorted(_PHI_UPDATE_POLICIES)}"
        )
    policy = _PHI_UPDATE_POLICIES[name]
    return PhiUpdatePolicy(
        optimizer_group_metadata=MappingProxyType(dict(policy.optimizer_group_metadata)),
        requires_manifold_optimizer=policy.requires_manifold_optimizer,
        requires_pullback_geometry=policy.requires_pullback_geometry,
    )


register_phi_update_policy(
    "adamw",
    PhiUpdatePolicy(
        optimizer_group_metadata={},
        requires_manifold_optimizer=False,
    ),
)
register_phi_update_policy(
    "pullback_group",
    PhiUpdatePolicy(
        optimizer_group_metadata={"pullback_group": True, "weight_decay": 0.0},
        requires_manifold_optimizer=True,
        requires_pullback_geometry=True,
    ),
)


def embedded_phi_frobenius_norm(
    phi:   torch.Tensor,                   # (..., n_gen) algebra coordinates
    group: GaugeGroup,

    *,
    warn_fallback: bool = True,
) -> torch.Tensor:                        # (...) ||sum_a phi_a G_a||_F
    r"""Exact embedded Frobenius norm through a certified diagonal Gram or dense fallback.

    If ``group`` certifies ``<G_a,G_b>_F = 0`` for ``a != b``, then
    ``||sum_a phi_a G_a||_F^2 = sum_a phi_a^2 ||G_a||_F^2``. Uncertified bases retain the exact
    dense embedding; they are correct but can be prohibitively expensive for a per-step global
    projection.
    """
    if phi.shape[-1] != group.generators.shape[0]:
        raise ValueError(
            "embedded phi norm requires full generator coordinates; got coordinate width "
            f"{phi.shape[-1]} and {group.generators.shape[0]} generators"
        )
    diagonal_fn = getattr(group, "gram_diagonal", None)
    diagonal = diagonal_fn() if diagonal_fn is not None else None
    if diagonal is not None:
        uniform = group.gram_diagonal_uniform()
        if uniform is not None:
            return torch.linalg.vector_norm(phi, dim=-1) * math.sqrt(uniform)
        weights = diagonal.to(device=phi.device, dtype=phi.dtype)
        return (phi.square() * weights).sum(dim=-1).clamp_min(0.0).sqrt()

    if warn_fallback and not getattr(group, "_phi_norm_fallback_warned", False):
        warnings.warn(
            f"gauge group {getattr(group, 'name', '<custom>')!r} has no diagonal "
            "generator-Gram certificate; "
            "phi matrix norms use the exact dense_fallback route, which can be expensive",
            RuntimeWarning,
            stacklevel=2,
        )
        group._phi_norm_fallback_warned = True
    basis = group.generators.to(device=phi.device, dtype=phi.dtype)
    embedded = torch.einsum("...a,aij->...ij", phi, basis)
    return torch.linalg.matrix_norm(embedded, ord="fro", dim=(-2, -1))


@dataclass(frozen=True)
class PullbackGroupCandidate:
    candidate_phi:           torch.Tensor
    trust_scale:             torch.Tensor
    backtracking_reductions: torch.Tensor
    candidate_chart_norm:    torch.Tensor
    group_product_residual:  torch.Tensor
    direction:               PullbackGroupDirectionResult


def _require_positive_finite(value: float, name: str) -> float:
    """Return a validated positive finite scalar."""
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return converted


def _pullback_group_product_residual(
    candidate_phi: torch.Tensor,             # (active, n_gen) BCH4 chart candidate
    phi:           torch.Tensor,             # (active, n_gen) current chart
    delta:         torch.Tensor,             # (active, n_gen) accepted right factor
    group:         GaugeGroup,
) -> torch.Tensor:                           # (active,) relative exact-product residual
    r"""Compare ``exp(candidate)`` with the exact right product ``exp(phi) exp(delta)``."""
    generators = group.generators.to(device=phi.device, dtype=torch.float64)
    compact_blocks = (
        group.name == "block_glk"
        and len(group.irrep_dims) > 1
        and len(set(group.irrep_dims)) == 1
        and generators.shape[0]
        == len(group.irrep_dims) * group.irrep_dims[0] * group.irrep_dims[0]
    )
    if compact_blocks:
        n_blocks = len(group.irrep_dims)
        block_dim = group.irrep_dims[0]

        def _blocks(coordinates: torch.Tensor) -> torch.Tensor:
            return coordinates.reshape(-1, n_blocks, block_dim, block_dim)

        candidate_element = torch.linalg.matrix_exp(_blocks(candidate_phi))
        current_element = torch.linalg.matrix_exp(_blocks(phi))
        right_element = torch.linalg.matrix_exp(_blocks(delta))
        reference = current_element @ right_element
        error_sq = (candidate_element - reference).square().sum(dim=(-3, -2, -1))
        reference_sq = reference.square().sum(dim=(-3, -2, -1))
        return error_sq.sqrt() / reference_sq.sqrt().clamp_min(torch.finfo(torch.float64).tiny)

    def _embedded(coordinates: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...a,aij->...ij", coordinates, generators)

    candidate_element = torch.linalg.matrix_exp(_embedded(candidate_phi))
    current_element = torch.linalg.matrix_exp(_embedded(phi))
    right_element = torch.linalg.matrix_exp(_embedded(delta))
    reference = current_element @ right_element
    error = torch.linalg.matrix_norm(candidate_element - reference, dim=(-2, -1))
    scale = torch.linalg.matrix_norm(reference, dim=(-2, -1))
    return error / scale.clamp_min(torch.finfo(torch.float64).tiny)


@torch.no_grad()
def stage_pullback_group_candidate(
    grad_phi: torch.Tensor,             # (active, n_gen) processed covector
    phi:      torch.Tensor,             # (active, n_gen) current chart
    group:    GaugeGroup,

    *,
    learning_rate:    float,
    trust_radius:     float,
    chart_max_norm:   float,
    bch_residual_max: float,
    phi_precond_mode: str,
    max_backtracks:   int = 10,
) -> PullbackGroupCandidate:
    r"""Stage one certified float64 right-group update without mutating inputs."""
    if not isinstance(group, GaugeGroup):
        raise TypeError(f"group must be a GaugeGroup, got {type(group).__name__}")
    if grad_phi.ndim != 2 or phi.ndim != 2 or grad_phi.shape != phi.shape:
        raise ValueError(
            "pullback group staging requires matching (active, n_gen) grad/phi tensors; "
            f"got {tuple(grad_phi.shape)} and {tuple(phi.shape)}"
        )
    if grad_phi.shape[-1] != group.generators.shape[0]:
        raise ValueError(
            "pullback group staging requires full generator coordinates; got width "
            f"{grad_phi.shape[-1]} and {group.generators.shape[0]} generators"
        )
    if grad_phi.device != phi.device:
        raise ValueError(
            f"pullback group grad/phi devices differ: {grad_phi.device} and {phi.device}"
        )
    learning_rate = float(learning_rate)
    if not math.isfinite(learning_rate) or learning_rate < 0.0:
        raise ValueError(
            f"learning_rate must be finite and nonnegative, got {learning_rate!r}"
        )
    trust_radius = _require_positive_finite(trust_radius, "trust_radius")
    chart_max_norm = _require_positive_finite(chart_max_norm, "chart_max_norm")
    bch_residual_max = _require_positive_finite(bch_residual_max, "bch_residual_max")
    if type(max_backtracks) is not int or max_backtracks < 0:
        raise ValueError(
            f"max_backtracks must be a nonnegative int, got {max_backtracks!r}"
        )
    if grad_phi.shape[0] == 0:
        raise ValueError("pullback group staging requires at least one active row")

    from vfe3.geometry.lie_ops import compose_bch

    with torch.amp.autocast(grad_phi.device.type, enabled=False):
        grad64 = grad_phi.to(dtype=torch.float64)
        phi64 = phi.to(dtype=torch.float64)
        generators64 = group.generators.to(device=phi.device, dtype=torch.float64)
        if not bool(torch.isfinite(grad64).all()) or not bool(torch.isfinite(phi64).all()):
            raise FloatingPointError("pullback group staging received a nonfinite grad or chart")
        direction = pullback_group_direction(
            grad64,
            phi64,
            generators64,
            mode=phi_precond_mode,
            irrep_dims=list(group.irrep_dims),
        )
        direction_tensors = (
            direction.v_phi,
            direction.xi,
            direction.min_undamped_generalized_eigenvalue,
            direction.undamped_generalized_condition,
            direction.damped_generalized_condition,
            direction.scaled_solve_residual,
        )
        if not all(bool(torch.isfinite(value).all()) for value in direction_tensors):
            raise FloatingPointError("pullback group direction returned a nonfinite certificate")

        right_factor = -learning_rate * direction.xi
        right_norm = embedded_phi_frobenius_norm(
            right_factor,
            group,
            warn_fallback=False,
        )
        tiny = torch.finfo(torch.float64).tiny
        trust_scale = (trust_radius / right_norm.clamp_min(tiny)).clamp(max=1.0)
        factor_scale = trust_scale.clone()
        reductions = torch.zeros(
            grad64.shape[0],
            dtype=torch.long,
            device=grad64.device,
        )
        residual_limit = min(1e-6, bch_residual_max)
        for attempt in range(max_backtracks + 1):
            delta = factor_scale.unsqueeze(-1) * right_factor
            candidate_phi = compose_bch(
                phi64,
                delta,
                generators64,
                order=4,
                block_dims=list(group.irrep_dims),
                compact_blocks=(group.name == "block_glk"),
            )
            candidate_chart_norm = embedded_phi_frobenius_norm(
                candidate_phi,
                group,
                warn_fallback=False,
            )
            if not bool(torch.isfinite(candidate_phi).all()) or not bool(
                torch.isfinite(candidate_chart_norm).all()
            ):
                raise FloatingPointError("pullback group staging produced a nonfinite candidate")
            if bool((candidate_chart_norm > chart_max_norm).any()):
                maximum = float(candidate_chart_norm.max())
                raise FloatingPointError(
                    "pullback group candidate chart norm "
                    f"{maximum:.6g} exceeds bound {chart_max_norm:.6g}"
                )
            group_product_residual = _pullback_group_product_residual(
                candidate_phi,
                phi64,
                delta,
                group,
            )
            if not bool(torch.isfinite(group_product_residual).all()):
                raise FloatingPointError(
                    "pullback group staging produced a nonfinite group-product residual"
                )
            failures = group_product_residual > residual_limit
            if not bool(failures.any()):
                return PullbackGroupCandidate(
                    candidate_phi=candidate_phi,
                    trust_scale=trust_scale,
                    backtracking_reductions=reductions,
                    candidate_chart_norm=candidate_chart_norm,
                    group_product_residual=group_product_residual,
                    direction=direction,
                )
            if attempt == max_backtracks:
                maximum = float(group_product_residual.max())
                raise FloatingPointError(
                    "pullback group BCH residual "
                    f"{maximum:.3e} exceeds {residual_limit:.1e} after "
                    f"{max_backtracks} reductions"
                )
            factor_scale = torch.where(failures, 0.5 * factor_scale, factor_scale)
            reductions = reductions + failures.to(dtype=torch.long)

    raise AssertionError("unreachable pullback group staging path")


def phi_projection_chunk_rows(
    coordinate_width: int,
    matrix_width:     int,
    element_size:     int,
    temporary_bytes:  int = _PHI_PROJECTION_TEMPORARY_BYTES,

    *,
    dense_fallback:   bool = False,
) -> int:
    """Rows whose norm and scale temporaries fit within the requested approximate byte budget."""
    if temporary_bytes < 1:
        raise ValueError(f"temporary_bytes must be positive, got {temporary_bytes}")
    working_elements = 2 * coordinate_width
    if dense_fallback:
        working_elements += matrix_width * matrix_width
    return max(1, temporary_bytes // max(working_elements * element_size, 1))


@torch.no_grad()
def project_phi_parameter_rows_(
    model:           torch.nn.Module,
    max_matrix_norm: float,

    *,
    temporary_bytes: int  = _PHI_PROJECTION_TEMPORARY_BYTES,
    collect_stats:   bool = True,
    chunk_rows:      Optional[int] = None,
) -> Dict[str, float]:
    r"""Project every trainable phi-table row to an embedded Frobenius-norm ball.

    Projection rescales algebra coordinates along their current ray and deliberately leaves
    optimizer moments unchanged. It covers belief token frames, independent model-channel frames,
    and both learned positional tables when present.
    """
    if not math.isfinite(max_matrix_norm) or max_matrix_norm <= 0.0:
        raise ValueError(
            f"max_matrix_norm must be finite and positive, got {max_matrix_norm}"
        )
    if chunk_rows is not None and (type(chunk_rows) is not int or chunk_rows < 1):
        raise ValueError(f"chunk_rows must be a positive int, got {chunk_rows!r}")
    if type(temporary_bytes) is not int or temporary_bytes < 1:
        raise ValueError(f"temporary_bytes must be a positive int, got {temporary_bytes!r}")
    tables = [
        getattr(model.prior_bank, "phi_embed", None),
        getattr(model.prior_bank, "s_phi_embed", None),
        getattr(model, "pos_phi_free", None),
        getattr(model, "s_pos_phi_free", None),
    ]
    unique = []
    seen = set()
    for table in tables:
        if table is not None and id(table) not in seen:
            unique.append(table)
            seen.add(id(table))

    group = model.group
    generators = group.generators
    total_rows = 0
    projected_rows = torch.zeros((), device=generators.device, dtype=torch.long)
    norm_max_before = torch.zeros((), device=generators.device, dtype=generators.dtype)
    minimum_scale = torch.ones((), device=generators.device, dtype=generators.dtype)
    for table in unique:
        rows = table.reshape(-1, table.shape[-1])
        if rows.shape[-1] != generators.shape[0]:
            raise ValueError(
                "phi chart projection requires full generator coordinates; got table width "
                f"{rows.shape[-1]} and {generators.shape[0]} generators"
            )
        total_rows += rows.shape[0]
        rows_per_chunk = chunk_rows or phi_projection_chunk_rows(
            rows.shape[-1],
            generators.shape[-1],
            rows.element_size(),
            temporary_bytes,
            dense_fallback=group.phi_norm_route() == "dense_fallback",
        )
        for start in range(0, rows.shape[0], rows_per_chunk):
            chunk = rows[start : start + rows_per_chunk]
            norm = embedded_phi_frobenius_norm(chunk, group)
            scale = (max_matrix_norm / norm.clamp(min=1e-12)).clamp(max=1.0)
            if collect_stats:
                projected_rows.add_((scale < 1.0).sum())
                norm_max_before.copy_(torch.maximum(norm_max_before, norm.max()))
                minimum_scale.copy_(torch.minimum(minimum_scale, scale.min()))
            chunk.mul_(scale.unsqueeze(-1))
    if not collect_stats:
        return {}
    projected_rows_value, norm_max_value, minimum_scale_value = torch.stack((
        projected_rows.to(dtype=generators.dtype),
        norm_max_before,
        minimum_scale,
    )).cpu().tolist()
    return {
        "phi_chart_projected_rows":     projected_rows_value,
        "phi_chart_total_rows":         float(total_rows),
        "phi_chart_projected_fraction": projected_rows_value / max(total_rows, 1),
        "phi_chart_preproject_max":     norm_max_value,
        "phi_chart_projection_scale_min": minimum_scale_value,
    }


def _polar_orthogonalize(
    U: torch.Tensor,                      # (..., K, K) possibly drifted-off-O(K) element
) -> torch.Tensor:                        # (..., K, K) nearest orthogonal matrix
    r"""Nearest orthogonal matrix to ``U`` via the polar decomposition :math:`Q = U (U^T U)^{-1/2}`.

    Uses the SVD polar factor :math:`Q = W V^T` (:math:`U = W S V^T`), the exact Frobenius-norm
    minimizer of :math:`\lVert U - Q \rVert_F` over :math:`O(K)`. Runs in a float64 island
    (autocast disabled) so the drift correction itself does not introduce fresh fp32 rounding;
    keeps a drifted skew-group frame exactly on :math:`O(K)` so ``U^T`` stays the exact inverse
    (the transpose fast path :func:`build_transport_from_element` relies on for skew groups).
    """
    with torch.amp.autocast(U.device.type, enabled=False):
        W, _, Vh = torch.linalg.svd(U.double(), full_matrices=False)
        return (W @ Vh).to(U.dtype)


def _omega_validation_failure(
    U: torch.Tensor,                      # (..., d, d) updated full elements or compact blocks
) -> torch.Tensor:                       # () bool, retained on device for aggregate validation
    r"""Native-dtype finite/nonsingular status without an eager host synchronization."""
    with torch.no_grad(), torch.amp.autocast(U.device.type, enabled=False):
        matrices = U.reshape(-1, U.shape[-1], U.shape[-1])
        _, _, info = torch.linalg.lu_factor_ex(matrices, check_errors=False)
        return torch.logical_or(~torch.isfinite(U).all(), (info != 0).any())


def _omega_determinant_failure(
    U: torch.Tensor,                      # (..., d, d) updated full elements or compact blocks
) -> torch.Tensor:                       # () bool, diagnostic-only float64 determinant status
    r"""Sparse determinant audit used only on an explicitly requested diagnostic step."""
    with torch.no_grad(), torch.amp.autocast(U.device.type, enabled=False):
        sign, logabsdet = torch.linalg.slogdet(
            U.reshape(-1, U.shape[-1], U.shape[-1]).double())
        return torch.logical_or((sign == 0).any(), ~torch.isfinite(logabsdet).all())


def _require_finite_nonsingular_omega(
    U: torch.Tensor,                      # (..., d, d) updated full elements or compact blocks
) -> None:
    r"""Fail closed with one host decision when a group element is nonfinite or singular."""
    if bool(_omega_validation_failure(U)):
        raise FloatingPointError("omega retraction produced a nonfinite or singular group element")


def _omega_condition_values(
    U: torch.Tensor,                      # (..., d, d) updated full elements or compact blocks
) -> torch.Tensor:                        # (...) spectral condition number per element/block
    r"""Float64 condition of each represented element for log-cadence diagnostics.

    An untied compact row has shape ``(A, H, d, d)`` and represents the block diagonal
    ``diag(U_1, ..., U_H)``. Its largest singular value is the largest singular value over every
    block and its smallest is the smallest over every block; taking ``cond`` block-by-block would
    miss cross-block scale separation. Dense and tied ``(A, d, d)`` rows keep their ordinary matrix
    condition.
    """
    with torch.no_grad(), torch.amp.autocast(U.device.type, enabled=False):
        if U.dim() == 4:                                      # (A,H,d,d) untied compact elements
            singular = torch.linalg.svdvals(U.double())       # (A,H,d), descending within each block
            return singular[..., 0].amax(dim=-1) / singular[..., -1].amin(dim=-1)
        return torch.linalg.cond(U.reshape(-1, U.shape[-1], U.shape[-1]).double())


def _symplectic_membership_residual(
    U: torch.Tensor,                      # (..., K, K), K even, defining Sp(K,R) representation
) -> torch.Tensor:                        # (...) relative ||U^T J U - J||_F
    r"""Relative defining-representation symplectic residual ``||U^T J U-J||_F/||J||_F``."""
    K = U.shape[-1]
    if K % 2 != 0:
        raise ValueError(f"symplectic omega diagnostic requires even K, got K={K}")
    with torch.no_grad(), torch.amp.autocast(U.device.type, enabled=False):
        m = K // 2
        J = torch.zeros(K, K, device=U.device, dtype=torch.float64)
        eye = torch.eye(m, device=U.device, dtype=torch.float64)
        J[:m, m:] = eye
        J[m:, :m] = -eye
        U64 = U.double()
        error = U64.transpose(-1, -2) @ J @ U64 - J
        return error.norm(dim=(-2, -1)) / J.norm()


class GaugeManifoldAdamW(torch.optim.AdamW):
    r"""AdamW for ordinary groups plus stateless phi and stored-element group retractions.

    A parameter group marked ``pullback_group=True`` is staged through
    :func:`stage_pullback_group_candidate` on active rows only. The update creates no optimizer
    state for the phi table. A group marked ``omega=True`` retains the established direct-element
    retraction and dirty-row cadence state.
    """

    # Signature-convention exception: torch.optim.Optimizer's contract REQUIRES params as the
    # first positional argument (super().__init__(params, ...)), so ``group`` follows it.
    def __init__(
        self,
        params,
        group: GaugeGroup,

        *,
        phi_group_trust_radius: float,
        phi_chart_max_norm:     float,
        phi_bch_residual_max:   float,
        phi_precond_mode:       str,
        omega_retract_mode:     str = "lie_exp",
        omega_reorth_every:     int = 0,
        **kwargs,
    ) -> None:
        if not isinstance(group, GaugeGroup):
            raise TypeError(f"group must be a GaugeGroup, got {type(group).__name__}")
        if type(omega_reorth_every) is not int or omega_reorth_every < 0:
            raise ValueError(
                "omega_reorth_every must be a nonnegative int, got "
                f"{type(omega_reorth_every).__name__}: {omega_reorth_every!r}"
            )
        phi_group_trust_radius = _require_positive_finite(
            phi_group_trust_radius,
            "phi_group_trust_radius",
        )
        phi_chart_max_norm = _require_positive_finite(
            phi_chart_max_norm,
            "phi_chart_max_norm",
        )
        phi_bch_residual_max = _require_positive_finite(
            phi_bch_residual_max,
            "phi_bch_residual_max",
        )
        super().__init__(params, **kwargs)
        self._group                  = group
        self._generators             = group.generators
        self._irrep_dims             = list(group.irrep_dims)
        self._precond_mode           = phi_precond_mode
        self._phi_group_trust_radius = phi_group_trust_radius
        self._phi_chart_max_norm     = phi_chart_max_norm
        self._phi_bch_residual_max   = phi_bch_residual_max
        # Group-manifold retraction for the omega_direct group (params flagged omega=True): 'lie_exp'
        # (matrix_exp; follows the one-parameter subgroup) or 'cayley' (exp-free (I-A/2)^{-1}(I+A/2)).
        self._omega_retract_mode = omega_retract_mode
        # Orthogonality-drift control for a SKEW (orthogonal, e.g. so_k/so_n) omega_direct group:
        # fp32 accumulation of exp(skew) retraction products walks the stored U off O(K) over many
        # M-steps, after which U^T stops being the exact inverse (the transpose fast path
        # build_transport_from_element relies on for skew groups). skew_symmetric mirrors
        # group.skew_symmetric; omega_reorth_every>0 turns on a
        # periodic polar re-orthogonalization every that many M-steps. Default 0 = off = byte-identical.
        self._skew_symmetric     = bool(group.skew_symmetric)
        self._omega_reorth_every = omega_reorth_every
        self._omega_step         = 0                     # M-step counter for the reorth cadence
        self._group_name         = group.name
        self._has_omega_group    = any(item.get("omega", False) for item in self.param_groups)
        # D1/EXP-8 training-time diagnostics, GATED: the caller (train.py) sets _collect_gauge_diag
        # True only on a log/eval step, so the silent hot path computes NOTHING extra. When set, step()
        # stashes cos(nat, grad) (1.0 for the conformal killing rescale; <1 when pullback reshapes the
        # direction) and -- on the pullback modes -- the per-token metric condition number into
        # _gauge_diag. Omega-direct groups additionally report active-row element condition, or the
        # defining Sp(K,R) membership residual for ``sp``. train.py reads the fixed keys into metrics.csv.
        self._collect_gauge_diag = False
        self._gauge_diag: dict   = {}

    def __setstate__(self, state) -> None:                             # type: ignore[override]
        r"""Restore generically, running Adam's step migration only where ``"step"`` exists.

        ``Optimizer.load_state_dict`` dispatches to ``__setstate__``; ``Adam.__setstate__`` assumes
        every non-empty per-parameter state carries ``"step"``. Omega state contains only its
        dirty-row mask, so inherited Adam migration remains unsafe. Restore through the base
        ``Optimizer`` and migrate only states that actually carry an Adam ``"step"`` slot.
        """
        torch.optim.Optimizer.__setstate__(self, state)
        for s in self.state.values():
            step = s.get("step")
            if step is not None and not torch.is_tensor(step):
                s["step"] = torch.tensor(float(step))

    def state_dict(self) -> Dict[str, object]:                         # type: ignore[override]
        r"""Serialize AdamW state plus the M-step count that drives omega reorthogonalization."""
        state = super().state_dict()
        state["optimizer_extra"] = {
            "omega_step":         int(self._omega_step),
            "omega_dirty_format": 1,
        }
        return state

    def load_state_dict(
        self,
        state_dict: Dict[str, object],
    ) -> None:                                                        # type: ignore[override]
        r"""Restore AdamW state, omega cadence, and versioned dirty-row masks.

        Pre-O5 checkpoints may carry ``omega_step`` but no dirty-mask format marker. If a live
        single-block skew cadence resumes from that format, every row is conservatively marked dirty:
        the missing mask cannot prove which rows accumulated drift before the checkpoint. Current
        checkpoints carry ``omega_dirty_format=1``; a missing mask in that format means clean.
        """
        core_state = dict(state_dict)
        extra = core_state.pop("optimizer_extra", None)
        super().load_state_dict(core_state)
        if isinstance(extra, dict) and "omega_step" in extra:
            self._omega_step = int(extra["omega_step"])
        else:
            import warnings
            self._omega_step = 0
            warnings.warn(
                "GaugeManifoldAdamW checkpoint has no optimizer_extra.omega_step; the omega "
                "reorthogonalization cadence restarts at zero (non-exact resume when "
                "omega_reorth_every > 0).",
                UserWarning,
                stacklevel=2,
            )

        current_dirty_format = isinstance(extra, dict) and extra.get("omega_dirty_format") == 1
        legacy_skew_cadence = (
            not current_dirty_format
            and self._has_omega_group
            and self._skew_symmetric
            and self._omega_reorth_every > 0
            and len(self._irrep_dims) == 1
        )
        for group in self.param_groups:
            if not group.get("omega", False):
                continue
            for p in group["params"]:
                dirty = self.state[p].get("omega_dirty")
                if dirty is None:
                    dirty = torch.full(
                        (p.shape[0],), legacy_skew_cadence,
                        dtype=torch.bool, device=p.device)
                else:
                    if dirty.shape != (p.shape[0],):
                        raise ValueError(
                            f"optimizer omega_dirty has shape {tuple(dirty.shape)}, expected "
                            f"({p.shape[0]},)")
                    dirty = dirty.to(device=p.device, dtype=torch.bool)
                    if legacy_skew_cadence:
                        dirty.fill_(True)
                self.state[p]["omega_dirty"] = dirty
        if legacy_skew_cadence:
            import warnings
            warnings.warn(
                "GaugeManifoldAdamW checkpoint predates optimizer_extra.omega_dirty_format; "
                "all omega rows were marked dirty so the next scheduled orthogonal projection "
                "cannot silently skip pre-checkpoint drift.",
                UserWarning,
                stacklevel=2,
            )

    def _compact_gld_basis(
        self,
        d:      int,                          # block dimension
        device: torch.device,
        dtype:  torch.dtype,
    ) -> 'tuple[torch.Tensor, torch.Tensor]':  # (d*d, d, d) gl(d) basis, (d*d, d*d) its gram_pinv
        r"""The reduced ``gl(d)`` generator basis + Gram pseudo-inverse for the per-block compact
        retraction, built and cached ONCE (per d, device, dtype).

        A compact ``omega_direct`` table is (V, H, d, d) / (V, d, d) blocks; each block steps on GL(d)
        under the full ``gl(d)`` elementary basis E_ij (``generate_glk(d)``, an orthonormal (d*d, d, d)
        set). This is the block-restriction of the full block-diagonal ``gl(K)`` basis, so the per-block
        step equals the full K x K step on the blocks. The Gram pseudo-inverse is cached alongside so
        ``extract_phi`` never recomputes it per step or per param (FIX 3).
        """
        cache = getattr(self, "_gld_cache", None)
        if cache is None:
            cache = self._gld_cache = {}
        key = (d, device, dtype)
        entry = cache.get(key)
        if entry is None:
            from vfe3.geometry.generators import generate_glk
            from vfe3.geometry.lie_ops import gram_pinv
            G = generate_glk(d, device=device, dtype=dtype)
            entry = cache[key] = (G, gram_pinv(G))
        return entry

    def _full_generator_basis(
        self,
        device: torch.device,
        dtype:  torch.dtype,
    ) -> 'tuple[torch.Tensor, torch.Tensor]':  # converted basis and its cached Gram pseudo-inverse
        r"""Cache the immutable full generator basis and factorization across optimizer steps."""
        source = self._generators
        cache = getattr(self, "_full_generator_cache", None)
        if cache is None:
            cache = self._full_generator_cache = {}
        key = (id(source), source._version, device, dtype)
        entry = cache.get(key)
        if entry is None:
            from vfe3.geometry.lie_ops import gram_pinv
            basis = source.to(device=device, dtype=dtype)
            entry = cache[key] = (basis, gram_pinv(basis))
        return entry

    @torch.no_grad()
    def step(self, closure=None):                                      # type: ignore[override]
        if closure is not None:
            # The manifold groups are stepped and their grads consumed (set to None) BELOW, before the
            # trailing super().step(). A closure that re-evaluates the loss would repopulate those grads
            # and let base AdamW step the frame a SECOND time. No caller passes a closure (GradScaler.step
            # and a bare optimizer.step() both call with closure=None), so reject it rather than risk the
            # double-step.
            raise NotImplementedError(
                "GaugeManifoldAdamW does not support closure-based steps: a manifold gradient is "
                "consumed before super().step(), so a closure that re-backpropagates would "
                "double-step the frame. Call step() with no closure."
            )
        collect = self._collect_gauge_diag                             # gated: True only on log/eval steps
        if collect:
            self._gauge_diag = {}                                      # never expose a prior attempted step
        cos_acc: List[float] = []
        cond_acc: List[torch.Tensor] = []
        omega_cond_acc: List[torch.Tensor] = []
        omega_sp_acc: List[torch.Tensor] = []

        # Stage every direct-frame candidate across every omega parameter group before mutating
        # any table, gradient, dirty mask, or cadence state. A late invalid candidate therefore
        # rejects the whole manifold step atomically instead of leaving earlier groups committed.
        omega_plans: Dict[int, list] = {}
        omega_validation_failures: List[torch.Tensor] = []
        K_full = self._generators.shape[-1]
        from vfe3.geometry.lie_ops import extract_phi, retract_omega
        for omega_group in self.param_groups:
            if not omega_group.get("omega", False):
                continue
            lr = omega_group["lr"]
            mode = getattr(self, "_omega_retract_mode", "lie_exp")
            pending_updates = []
            pending_zero_gradients = []
            for p in omega_group["params"]:
                if p.grad is None:
                    continue
                E = p.grad
                U = p.data
                untied_compact = U.dim() == 4
                tied_compact = U.dim() == 3 and U.shape[-1] < K_full
                act = E.reshape(E.shape[0], -1).abs().sum(dim=-1) > 0
                Ua, Ea = U[act], E[act]
                if Ua.shape[0] == 0:
                    pending_zero_gradients.append(p)
                    continue
                omega_validation_failures.append(_omega_validation_failure(Ua))
                if untied_compact or tied_compact:
                    d = Ua.shape[-1]
                    Gd, gp = self._compact_gld_basis(d, U.device, U.dtype)
                    Ua_r = Ua.reshape(-1, d, d)
                    Ea_r = Ea.reshape(-1, d, d)
                    xi = extract_phi(
                        torch.einsum("...lk,...lm->...km", Ua_r, Ea_r),
                        Gd,
                        gram_pinv_=gp,
                    )
                    if tied_compact:
                        xi = xi / (K_full // d)
                    Ur = retract_omega(Ua_r, -lr * xi, Gd, mode=mode).reshape(Ua.shape)
                else:
                    Gd, full_gp = self._full_generator_basis(U.device, U.dtype)
                    xi = extract_phi(
                        torch.einsum("...lk,...lm->...km", Ua, Ea),
                        Gd,
                        gram_pinv_=full_gp,
                    )
                    Ur = retract_omega(Ua, -lr * xi, Gd, mode=mode)
                omega_validation_failures.append(_omega_validation_failure(Ur))
                if collect:
                    omega_validation_failures.append(_omega_determinant_failure(Ur))
                pending_updates.append((p, U, act, Ur, untied_compact, tied_compact))
            omega_plans[id(omega_group)] = (pending_updates, pending_zero_gradients)

        if (omega_validation_failures
                and bool(torch.stack(omega_validation_failures).any())):
            raise FloatingPointError(
                "omega retraction produced a nonfinite or singular group element"
            )

        # Stage every stored phi-factor candidate before committing any manifold update. The processed
        # gradient is exactly the tensor present after GradScaler unscale and clipping at optimizer.step.
        # No pullback parameter touches ``self.state``; the route is stateless by construction.
        phi_plans: Dict[int, tuple] = {}
        for phi_group in self.param_groups:
            if not phi_group.get("pullback_group", False):
                continue
            pending_updates = []
            pending_zero_gradients = []
            for p in phi_group["params"]:
                if p.grad is None:
                    continue
                gradient = p.grad
                flat_gradient = gradient.reshape(-1, gradient.shape[-1])
                flat_phi = p.data.reshape(-1, p.data.shape[-1])
                active = flat_gradient.abs().sum(dim=-1) > 0
                if not bool(active.any()):
                    pending_zero_gradients.append(p)
                    continue
                candidate = stage_pullback_group_candidate(
                    flat_gradient[active],
                    flat_phi[active],
                    self._group,
                    learning_rate=phi_group["lr"],
                    trust_radius=self._phi_group_trust_radius,
                    chart_max_norm=self._phi_chart_max_norm,
                    bch_residual_max=self._phi_bch_residual_max,
                    phi_precond_mode=self._precond_mode,
                )
                if collect:
                    cosine = torch.nn.functional.cosine_similarity(
                        candidate.direction.v_phi,
                        flat_gradient[active].to(dtype=torch.float64),
                        dim=-1,
                    )
                    cos_acc.append(float(cosine.mean()))
                    cond_acc.append(
                        candidate.direction.damped_generalized_condition.reshape(-1)
                    )
                pending_updates.append((p, flat_phi, active, candidate))
            phi_plans[id(phi_group)] = (pending_updates, pending_zero_gradients)

        for group in self.param_groups:
            if group.get("pullback_group", False):
                pending_updates, pending_zero_gradients = phi_plans[id(group)]
                for p, flat_phi, active, candidate in pending_updates:
                    committed = candidate.candidate_phi.to(dtype=p.dtype)
                    flat_phi[active] = committed
                    p.grad = None
                for p in pending_zero_gradients:
                    p.grad = None
                continue
            if group.get("omega", False):
                # omega_direct group: the params ARE stored GL(K) group elements U (shape (V, K, K)),
                # not algebra coordinates. Step by a group-manifold retraction of the natural-gradient
                # tangent xi = Gram^{-1} proj_g(U^T E) (extract_phi computes exactly this): U <- U retr(-lr xi),
                # active (nonzero-grad) rows only, then consume the grad so base AdamW no-ops on it.
                # Compact storage (omega_compact_storage=True): the table is per-block stacks -- untied
                # (V, H, d, d) has dim 4, tied (V, d, d) has dim 3 with block dim d < K (a full table is
                # (V, K, K), dim 3 with last dim == K). Both step each block by the reduced gl(d) basis
                # (a K x K solve/exp per block collapses to small d x d ones), which equals the full
                # block-diagonal gl(K) step restricted to the blocks (the gl(K) generators have disjoint
                # block support). Detecting tied by dim ALONE mis-routes (V,d,d) to the full path and
                # crashes extract_phi with an einsum size mismatch -- hence the d < K test.
                pending_updates, pending_zero_gradients = omega_plans[id(group)]

                for p, U, act, Ur, untied_compact, tied_compact in pending_updates:
                    U[act] = Ur                                        # U <- U retr(-lr xi), after validation
                    state = self.state[p]
                    dirty = state.get("omega_dirty")
                    if dirty is None:
                        dirty = torch.zeros(U.shape[0], dtype=torch.bool, device=U.device)
                        state["omega_dirty"] = dirty
                    elif dirty.shape != (U.shape[0],):
                        raise ValueError(
                            f"optimizer omega_dirty has shape {tuple(dirty.shape)}, expected "
                            f"({U.shape[0]},)")
                    elif dirty.device != U.device or dirty.dtype != torch.bool:
                        dirty = dirty.to(device=U.device, dtype=torch.bool)
                        state["omega_dirty"] = dirty
                    dirty.logical_or_(act)
                    if collect:
                        updated = U[act]
                        if self._group_name == "sp" and not (untied_compact or tied_compact):
                            omega_sp_acc.append(_symplectic_membership_residual(updated).reshape(-1))
                        else:
                            omega_cond_acc.append(_omega_condition_values(updated).reshape(-1))
                    p.grad = None                                      # consumed: AdamW no-ops on it
                for p in pending_zero_gradients:
                    p.grad = None                                      # zero gradient is still consumed
                continue
        # Orthogonality-drift control (default OFF: omega_reorth_every=0 -> byte-identical).
        # Polar reorth guarantees O(K) membership, which equals the structure group ONLY for the
        # single-block defining rep (so_k: rho(SO(K)) = SO(K)). Irrep towers are excluded because
        # their faithful rho(SO(N)) image is a proper submanifold of O(K). The cadence is one clock
        # per optimizer step, after every omega group commits; when it fires, every eligible omega
        # group is projected on that same step, including dirty rows accumulated without new grads.
        if (self._has_omega_group and self._skew_symmetric and self._omega_reorth_every > 0
                and len(self._irrep_dims) == 1):
            self._omega_step += 1
            if self._omega_step % self._omega_reorth_every == 0:
                for omega_group in self.param_groups:
                    if not omega_group.get("omega", False):
                        continue
                    for p in omega_group["params"]:
                        dirty = self.state[p].get("omega_dirty")
                        if dirty is None:
                            continue
                        if dirty.shape != (p.shape[0],):
                            raise ValueError(
                                f"optimizer omega_dirty has shape {tuple(dirty.shape)}, expected "
                                f"({p.shape[0]},)")
                        if dirty.device != p.device or dirty.dtype != torch.bool:
                            dirty = dirty.to(device=p.device, dtype=torch.bool)
                            self.state[p]["omega_dirty"] = dirty
                        if not bool(dirty.any()):
                            continue
                        projected = _polar_orthogonalize(p.data[dirty])
                        _require_finite_nonsingular_omega(projected)
                        p.data[dirty] = projected
                        dirty.zero_()
        if collect:
            if cos_acc:
                self._gauge_diag["cos_nat_phi"] = sum(cos_acc) / len(cos_acc)
            if cond_acc:
                allc = torch.cat(cond_acc)
                self._gauge_diag["pullback_cond_median"] = float(allc.median())
                self._gauge_diag["pullback_cond_max"]    = float(allc.max())
            if omega_cond_acc:
                allc = torch.cat(omega_cond_acc)
                self._gauge_diag["omega_condition_median"] = float(allc.median())
                self._gauge_diag["omega_condition_max"]    = float(allc.max())
            if omega_sp_acc:
                allr = torch.cat(omega_sp_acc)
                self._gauge_diag["omega_symplectic_residual_median"] = float(allr.median())
                self._gauge_diag["omega_symplectic_residual_max"]    = float(allr.max())
        return super().step()                                          # closure already rejected above
