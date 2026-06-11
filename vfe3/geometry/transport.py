r"""Gauge transport for VFE_3.0 (Regime I, Gaussian / location-scale specific).

Two parameterizations of the flat (Regime I) transport:
  phi (exp):    Omega_ij = exp(phi_i . G) exp(-phi_j . G) in GL+(K) (det>0).
  omega_direct: Omega_ij = Omega_i Omega_j^{-1} for general GL(K) (det may be <0).
Belief action: mu -> Omega @ mu, Sigma -> Omega @ Sigma @ Omega^T (sandwich;
diagonal approximation for speed). Regime II and retractions are separate modules;
gauge-RoPE folds a positional rotation into the transport via :class:`RopeTransport`.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch

from vfe3.geometry.groups import GaugeGroup

TransportDict = Dict[str, torch.Tensor]


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
    """

    exp_phi:     torch.Tensor             # (..., N, K, K) exp(phi_i . G)
    exp_neg_phi: torch.Tensor             # (..., N, K, K) exp(-phi_j . G)
    irrep_dims:  List[int]                # equal block sizes; sum == K, len > 1

    def to_dense_omega(self) -> torch.Tensor:
        r"""Rebuild the dense Omega_ij = exp(phi_i) exp(-phi_j) (..., N, N, K, K).

        Byte-identical to ``compute_transport_operators``'s Omega einsum (same factors, same
        ``ikl,jlm->ijkm`` contraction); used to keep the FULL-covariance sandwich and any
        consumer that needs the explicit operator on the existing dense code path. Rank-agnostic
        via the leading ellipsis (an optional batch axis flows through; the unbatched call matches).
        """
        return torch.einsum("...ikl,...jlm->...ijkm", self.exp_phi, self.exp_neg_phi)


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
    """

    base:   'torch.Tensor | FactoredTransport'  # (N,N,K,K) dense OR factored transport
    rope:   torch.Tensor                        # (N, K, K) block-diagonal orthogonal rotation
    on_cov: bool = False


def _rope_dense_omega(base: 'torch.Tensor | FactoredTransport', rope: torch.Tensor) -> torch.Tensor:
    r"""Effective dense Omega^RoPE_ij = R(theta_i) Omega_ij R(theta_j)^T (full-gauge / dense path)."""
    omega = base.to_dense_omega() if isinstance(base, FactoredTransport) else base   # (...,N,N,K,K)
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
_TRANSPORTS: Dict[str, Callable[..., TransportDict]] = {}


def register_transport(name: str) -> Callable:
    """Decorator registering a transport (connection-regime) builder under ``name``."""
    def _wrap(fn: Callable[..., TransportDict]) -> Callable[..., TransportDict]:
        _TRANSPORTS[name] = fn
        return fn
    return _wrap


def get_transport(name: str) -> Callable[..., TransportDict]:
    """Return the registered transport builder (KeyError-with-available-list if absent)."""
    if name not in _TRANSPORTS:
        raise KeyError(f"no transport {name!r}; available: {sorted(_TRANSPORTS)}")
    return _TRANSPORTS[name]


@register_transport("flat")
def _build_flat(
    phi:        torch.Tensor,             # (B, N, n_gen) gauge frames
    group:      GaugeGroup,               # supplies generators, skew flag, irrep_dims

    *,
    gauge_mode: str = "learned",          # 'learned' (Regime I flat) or 'trivial'
    **kwargs,                             # tolerated (a future non-flat builder shares this shape)
) -> TransportDict:
    r"""Flat (Regime I) phi-cocycle transport: the registered default.

    A thin adapter forwarding verbatim to :func:`compute_transport_operators`
    (Omega_ij = exp(phi_i) exp(-phi_j) in GL+(K)); bit-identical to calling it directly. Extra
    keyword args are tolerated and ignored so a future stateful non-flat (Regime II) builder can
    share this call shape without editing the registry call sites.
    """
    return compute_transport_operators(phi, group, gauge_mode=gauge_mode)


@register_transport("regime_ii")
def _build_regime_ii(
    phi:                torch.Tensor,             # (B, N, n_gen) gauge frames
    group:              GaugeGroup,               # supplies generators, skew flag, irrep_dims

    *,
    gauge_mode:         str                       = "learned",   # 'learned' (flat vertex factors) | 'trivial'
    cocycle_relaxation: float                     = 1.0,         # homotopy alpha in [0,1]; 0 -> flat
    delta_soft_cap:     float                     = 12.0,        # smooth bound on ||delta_ij||_2 (< exp clamp max_norm=15)
    mu:                 Optional[torch.Tensor]    = None,        # (B, N, K) QUERY-slot means; the bilinear delta reads these
    connection_W:       Optional[torch.Tensor]    = None,        # (n_gen, K, K) learned bilinear connection (NN exception)
    mu_key:             Optional[torch.Tensor]    = None,        # (B, N, K) KEY-slot means (None -> mu); the filtering
    #                                                              oracle passes a DETACHED key slot so d delta/d mu
    #                                                              flows query-side only (values are detach-invariant)
    **kwargs,                                                    # tolerated (shares the flat builder's call shape)
) -> TransportDict:
    r"""Regime-II edge-relaxed (NON-FLAT) transport (spec eq:edge_relaxed_omega).

    NEURAL-NETWORK EXCEPTION (sanctioned, default-OFF): this builder consumes the LEARNED
    bilinear connection ``connection_W`` (an nn.Parameter on the model, trained by backprop on
    CE). The no-NN flat builder (:func:`_build_flat`) is the default and the pure path; this is
    the non-flat regime selected only by ``transport_mode='regime_ii'``.

    The edge-relaxed cocycle inserts an edge-local connection between the vertex factors:

        Omega_ij = exp(phi_i . G) exp(delta_ij . G) exp(-phi_j . G),       i != j,
        Omega_ii = exp(phi_i . G) exp(-phi_i . G) = I (self-edge excluded: delta_ii := 0),
        delta_ij^a = cocycle_relaxation * (mu_i^T W^a mu_j),   a = 1..n_gen,
        delta_ij -> delta_ij / sqrt(1 + ||delta_ij||^2 / delta_soft_cap^2)   (smooth norm cap),
        delta_ij . G = sum_a delta_ij^a G_a   in g,

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
    ``delta_soft_cap`` keeps ||delta . G||_F below ``stable_matrix_exp_pair``'s hard Frobenius
    clamp (max_norm=15) for the unit-Frobenius orthonormal generator bases the groups ship
    (||delta . G||_F = ||delta||_2 there), so the exp is always the EXACT operator, the
    cocycle_relaxation homotopy never saturates, and autograd never optimizes a clamped surrogate
    (audit 2026-06-10 F3); the hard clamp remains as a backstop for any non-orthonormal basis.

    Returns the SAME dict shape as the flat builder: 'exp_phi' (B,N,K,K), 'exp_neg_phi' (B,N,K,K),
    'Omega' (B,N,N,K,K).
    """
    # Flat fast path: no connection at all (None), the homotopy collapses it (alpha=0), or the
    # vertex factors are trivial -> delta plays no role -> Omega is exactly the flat cocycle. Skip the
    # O(N^2) edge exps. NOTE: we deliberately do NOT short-circuit on an all-ZERO (but grad-requiring)
    # connection_W: at W=0 the edge factor exp(delta)=I numerically (so the W=0->flat oracle holds to
    # float tolerance), but d Omega / d W at W=0 is the generator structure (exp'(0)=I), NOT zero --
    # short-circuiting there would sever the autograd graph and freeze the parameter at init. The full
    # einsum path keeps W in the graph so the loss backpropagates to it.
    if connection_W is None or cocycle_relaxation == 0.0 or gauge_mode == "trivial":
        return compute_transport_operators(phi, group, gauge_mode=gauge_mode)

    # Vertex factors exp(phi_i), exp(-phi_j) in FACTORED form (audit 2026-06-10 F8a): the same
    # stable exp machinery as the flat builder, WITHOUT materializing the dense (B, N, N, K, K)
    # flat Omega this path would immediately discard.
    fac = build_factored_transport(phi, group, gauge_mode=gauge_mode)
    exp_phi, exp_neg_phi = fac.exp_phi, fac.exp_neg_phi                         # (B, N, K, K)

    generators = group.generators                                              # (n_gen, K, K)
    # delta_ij^a = cocycle_relaxation * mu_i^T W^a mu_j  -> (B, N, N, n_gen). ``mu`` fills the
    # QUERY (i) slot and ``mu_key`` the KEY (j) slot; the VALUES are identical for any detach
    # combination, but the filtering oracle passes a detached key slot so d delta / d mu flows
    # query-side only (mean-field coordinate ascent).
    mu_k = mu_key if mu_key is not None else mu
    delta = cocycle_relaxation * torch.einsum("bik,akl,bjl->bija", mu, connection_W, mu_k)
    # Self-edge exclusion (audit 2026-06-10 F4): the connection is an EDGE object; the degenerate
    # i==i "edge" transports along the constant path, so Omega_ii stays exp(phi_i) exp(-phi_i) = I
    # exactly as on the flat path. Without this, delta_ii = mu_i^T W^a mu_i injects a spurious
    # nonzero self-energy E_ii into the (unmasked) attention softmax.
    n_tok = delta.shape[1]
    eye = torch.eye(n_tok, dtype=torch.bool, device=delta.device)
    delta = delta.masked_fill(eye.view(1, n_tok, n_tok, 1), 0.0)
    # Smooth per-edge norm cap (audit 2026-06-10 F3): delta is QUADRATIC in the unconstrained mean
    # scale (||delta|| ~ ||mu_i|| ||mu_j|| ||W||). delta -> delta * rsqrt(1 + ||delta||^2/cap^2)
    # bounds ||delta||_2 < delta_soft_cap, is the identity map to O(||delta||^2/cap^2) near zero
    # (the W=0 oracle and d Omega/d W at W=0 are untouched), and is STRICTLY monotone in
    # cocycle_relaxation everywhere -- the homotopy never saturates, unlike the hard matrix clamp
    # it pre-empts.
    sq = delta.pow(2).sum(dim=-1, keepdim=True)
    delta = delta * torch.rsqrt(1.0 + sq / (delta_soft_cap * delta_soft_cap))
    # delta_ij . G = sum_a delta_ij^a G_a  -> (B, N, N, K, K) Lie-algebra edge matrix
    delta_mat = torch.einsum("bija,akl->bijkl", delta, generators)
    # Per-edge group element exp(delta_ij . G); reuse the stable block-exp machinery
    # (only_forward: the edge factor enters Omega once, no exp(-delta) needed). exp_dim keys the
    # float64-island decision on the dimension actually exponentiated -- the per-head block --
    # so the O(N^2) edge exps of a block-diagonal group run at the block's own precision instead
    # of upcasting the whole (B, N, N, K, K) batch to float64 whenever K >= 20 (audit 2026-06-10
    # F8c; the soft cap above keeps the blocks in the well-conditioned exp regime).
    block_dims = group.irrep_dims if len(group.irrep_dims) > 1 else None
    exp_delta, _ = stable_matrix_exp_pair(
        delta_mat, skew_symmetric=group.skew_symmetric, only_forward=True, block_dims=block_dims,
        exp_dim=(max(block_dims) if block_dims is not None else None),
    )                                                                          # (B, N, N, K, K)

    # Omega_ij = exp(phi_i) @ exp_delta_ij @ exp(-phi_j)
    omega = torch.einsum("bikl,bijlm,bjmn->bijkn", exp_phi, exp_delta, exp_neg_phi)
    return {"exp_phi": exp_phi, "exp_neg_phi": exp_neg_phi, "Omega": omega}


def stable_matrix_exp_pair(
    matrix:         torch.Tensor,             # (..., d, d) Lie-algebra matrices

    *,
    max_norm:       float           = 15.0,
    dim_threshold:  int             = 20,
    skew_symmetric: bool            = False,
    only_forward:   bool            = False,
    block_dims:     Optional[List[int]] = None,   # per-block sizes (sum==d) for a block-diagonal M
    exp_dim:        Optional[int]       = None,   # dimension for the float64-island decision (None -> d)
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    r"""exp(M) and optionally exp(-M) with Frobenius-norm clamp + float64 upcast.

    Frobenius-norm clamp + float64 upcast keep matrix_exp stable for large ||M||.

    SAFEGUARD, NOT THE EXACT OPERATOR: when ``||M||_F > max_norm`` the matrix is rescaled to
    ``max_norm``, so the returned factor is ``exp(max_norm * M/||M||_F)``, NOT ``exp(M)`` -- the
    singular values / determinant of the returned operator differ from the true exponential. This
    is a stability clamp on extreme inputs only; keep ||phi|| (and the regime_ii edge delta) below
    ``max_norm`` to stay exact. A per-call runtime monitor is intentionally omitted: detecting
    activation needs a tensor reduction (a host sync) on this hot path, which the perf budget avoids.

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
    with torch.no_grad():
        mat_norm = matrix.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
        scale = (max_norm / mat_norm).clamp(max=1.0)
    matrix = matrix * scale

    d = matrix.shape[-1]
    orig_dtype = matrix.dtype
    # The full-K path's dtype choice; the per-block path forces the SAME dtype so a small block
    # (d_head < dim_threshold) does not silently drop to float32 and drift from the full exp.
    # exp_dim (when given) overrides the keying dimension -- see the docstring.
    d_eff = exp_dim if exp_dim is not None else d
    up_dtype = torch.float64 if d_eff >= dim_threshold else torch.float32

    with torch.amp.autocast('cuda', enabled=False):
        matrix_up = matrix.to(up_dtype).contiguous()

        if block_dims is not None and len(block_dims) > 1:
            exp_pos = _blockwise_matrix_exp(matrix_up, block_dims).to(orig_dtype)
            if only_forward:
                exp_neg = None
            elif skew_symmetric:
                exp_neg = exp_pos.transpose(-1, -2)
            else:
                exp_neg = _blockwise_matrix_exp(-matrix_up, block_dims).to(orig_dtype)
            return exp_pos, exp_neg

        exp_pos = torch.linalg.matrix_exp(matrix_up).to(orig_dtype)
        if only_forward:
            exp_neg = None
        elif skew_symmetric:
            exp_neg = exp_pos.transpose(-1, -2)
        else:
            exp_neg = torch.linalg.matrix_exp(-matrix_up).to(orig_dtype)
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


def compute_transport_operators(
    phi:        torch.Tensor,             # (B, N, n_gen) gauge frames
    group:      GaugeGroup,               # supplies generators, skew flag, irrep_dims

    *,
    gauge_mode: str = "learned",          # 'learned' (Regime I flat) or 'trivial'
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
    SQUARES it, so a gaussian_full run at fp32 can lose all significant digits there.
    Compact so towers give orthogonal Omega (cond = 1) and are unaffected. No guard is
    imposed here — bound phi via the retraction max_norm, or prefer a compact group /
    diagonal family when conditioning matters at fp32.
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
    block_dims = group.irrep_dims if len(group.irrep_dims) > 1 else None
    exp_phi, exp_neg_phi = stable_matrix_exp_pair(
        phi_matrix, skew_symmetric=group.skew_symmetric, block_dims=block_dims
    )
    omega = torch.einsum("bikl,bjlm->bijkm", exp_phi, exp_neg_phi)
    return {"exp_phi": exp_phi, "exp_neg_phi": exp_neg_phi, "Omega": omega}


def build_factored_transport(
    phi:        torch.Tensor,             # (..., N, n_gen) gauge frames (optional leading batch axis)
    group:      GaugeGroup,               # block-diagonal with equal blocks (len(irrep_dims) > 1)

    *,
    gauge_mode: str = "learned",          # 'learned' (Regime I flat) or 'trivial'
) -> FactoredTransport:
    r"""Flat phi-cocycle transport in FACTORED form, skipping the dense (..., N, N, K, K) Omega.

    Builds only the per-token vertex exponentials exp(phi_i), exp(-phi_j) (the same factors
    ``compute_transport_operators`` builds) and the ``ikl,jlm->ijkm`` Omega einsum is NEVER run.
    The pairwise contraction is deferred into ``transport_mean`` / ``transport_covariance``'s fast
    path (P0 #2). Caller guards this to the flat + block-diagonal-with-equal-blocks path; here it
    only requires the exps, which the block-diagonal exp machinery already produces. Rank-agnostic
    via the leading ellipsis: a (B, N, n_gen) frame (batched forward) and a (N, n_gen) frame (the
    unbatched block / diagnostics path) both flow through.
    """
    if gauge_mode == "trivial":
        # Trivial gauge: exp = I. Build the same per-token factors the dense path would (the
        # caller's guard normally excludes trivial, but keep the container well-formed).
        K = group.generators.shape[-1]
        eye_K = torch.eye(K, device=phi.device, dtype=phi.dtype)
        eye = eye_K.expand(*phi.shape[:-1], K, K).contiguous()
        return FactoredTransport(exp_phi=eye, exp_neg_phi=eye, irrep_dims=list(group.irrep_dims))
    if gauge_mode != "learned":
        raise ValueError(f"gauge_mode must be 'learned' or 'trivial', got {gauge_mode!r}")

    phi_matrix = torch.einsum("...na,aij->...nij", phi, group.generators)
    block_dims = group.irrep_dims if len(group.irrep_dims) > 1 else None
    exp_phi, exp_neg_phi = stable_matrix_exp_pair(
        phi_matrix, skew_symmetric=group.skew_symmetric, block_dims=block_dims
    )
    return FactoredTransport(exp_phi=exp_phi, exp_neg_phi=exp_neg_phi, irrep_dims=list(group.irrep_dims))


def transport_mean(
    omega: 'torch.Tensor | FactoredTransport | RopeTransport',   # (..., N, N, K, K) dense OR factored exps
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
    the smoothing-mode oracle if a container ever reaches it.

    ROPETRANSPORT path (``omega`` is a :class:`RopeTransport`): the gauge-RoPE rotation R(theta) is
    applied as R_i Omega_ij R_j^T mu_j -- pre-rotate the key mean by R_j^T, transport on the
    un-rotated base, post-rotate by R_i.
    """
    if isinstance(omega, RopeTransport):
        # mu_t[i,j] = R_i Omega_ij R_j^T mu_j: pre-rotate the key mean by R_j^T, transport on the
        # un-rotated base, post-rotate the result by R_i. R_j^T mu_j = sum_l R[j,l,k] mu[j,l].
        m = torch.einsum("...jlk,...jl->...jk", omega.rope, mu)        # (..., N, K)
        t = transport_mean(omega.base, m)                             # (..., N, N, K)
        return torch.einsum("...ikl,...ijl->...ijk", omega.rope, t)   # post-rotate by R_i
    if isinstance(omega, FactoredTransport):
        m = torch.einsum("...jlp,...jp->...jl", omega.exp_neg_phi, mu)  # (..., N, K): exp(-phi_j) @ mu_j
        return torch.einsum("...ikl,...jl->...ijk", omega.exp_phi, m)   # (..., N, N, K): exp(phi_i) @ m_j
    return torch.einsum("...ijkl,...jl->...ijk", omega, mu)


def transport_covariance(
    omega: 'torch.Tensor | FactoredTransport | RopeTransport',   # (..., N, N, K, K) dense OR factored exps
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
    the un-rotated base covariance; full-gauge (``on_cov=True``) sandwiches with the rotated dense
    operator.
    """
    if isinstance(omega, RopeTransport):
        if not omega.on_cov:
            return transport_covariance(omega.base, sigma, diagonal_out=diagonal_out)   # mu-only
        # full-gauge: sandwich with the rotated dense operator (requires full covariance).
        return transport_covariance(_rope_dense_omega(omega.base, omega.rope), sigma,
                                    diagonal_out=diagonal_out)
    if isinstance(omega, FactoredTransport):
        # Diagonal sigma is (..., N, K) -> same rank as exp_phi minus the trailing K axis; a full
        # sigma is (..., N, K, K) -> same rank as exp_phi (the dense-Omega rank-gap is +1 here
        # because the factored exps carry one fewer N axis than the dense (..., N, N, K, K)).
        is_diag = sigma.dim() == omega.exp_phi.dim() - 1 if diagonal_out is None else diagonal_out
        if not is_diag:
            return transport_covariance(omega.to_dense_omega(), sigma, diagonal_out=diagonal_out)
        return _factored_diagonal_covariance(omega, sigma)
    is_diag = sigma.dim() == omega.dim() - 2 if diagonal_out is None else diagonal_out
    if is_diag:
        return torch.einsum("...ijkl,...ijkl,...jl->...ijk", omega, omega, sigma)
    return torch.einsum("...ijkl,...jlm,...ijnm->...ijkn", omega, sigma, omega)


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
