r"""Gauge transport for VFE_3.0 (Regime I, Gaussian / location-scale specific).

Two parameterizations of the flat (Regime I) transport:
  phi (exp):    Omega_ij = exp(phi_i . G) exp(-phi_j . G) in GL+(K) (det>0).
  omega_direct: Omega_ij = Omega_i Omega_j^{-1} for general GL(K) (det may be <0).
Belief action: mu -> Omega @ mu, Sigma -> Omega @ Sigma @ Omega^T (sandwich;
diagonal approximation for speed). Regime II, retractions, RoPE are later phases.
"""

from typing import Callable, Dict, List, Optional, Tuple

import torch

from vfe3.geometry.groups import GaugeGroup

TransportDict = Dict[str, torch.Tensor]


# -- connection-regime registry (orthogonal to the gauge_parameterization phi|omega_direct axis) --
# The CONNECTION REGIME (flat Regime I, the non-flat Regime II to come) is a registry-backed modular
# axis on equal footing with the structure group (clean-room spec sec 4.2): config-selected, added by
# writing-and-registering, never editing call sites. This is ORTHOGONAL to gauge_parameterization
# (phi|omega_direct), which chooses how a single flat transport is parameterized; the regime chooses
# whether the connection is flat at all. Only the flat phi-cocycle is registered here; Regime II is
# design-spec'd (docs/superpowers/specs/2026-06-01-regime-ii-connection-design.md) and deferred.
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
    mu:                 Optional[torch.Tensor]    = None,        # (B, N, K) belief means; the bilinear delta reads these
    connection_W:       Optional[torch.Tensor]    = None,        # (n_gen, K, K) learned bilinear connection (NN exception)
    cocycle_relaxation: float                     = 1.0,         # homotopy alpha in [0,1]; 0 -> flat
    **kwargs,                                                    # tolerated (shares the flat builder's call shape)
) -> TransportDict:
    r"""Regime-II edge-relaxed (NON-FLAT) transport (spec eq:edge_relaxed_omega).

    NEURAL-NETWORK EXCEPTION (sanctioned, default-OFF): this builder consumes the LEARNED
    bilinear connection ``connection_W`` (an nn.Parameter on the model, trained by backprop on
    CE). The no-NN flat builder (:func:`_build_flat`) is the default and the pure path; this is
    the non-flat regime selected only by ``transport_mode='regime_ii'``.

    The edge-relaxed cocycle inserts an edge-local connection between the vertex factors:

        Omega_ij = exp(phi_i . G) exp(delta_ij . G) exp(-phi_j . G),
        delta_ij^a = cocycle_relaxation * (mu_i^T W^a mu_j),   a = 1..n_gen,
        delta_ij . G = sum_a delta_ij^a G_a   in g,

    with ``{G_a} = group.generators``. Because delta is valued in the group's own generator
    coordinates, exp(delta_ij . G) lies in the group by construction (block structure / irrep_dims
    preserved). At ``connection_W=None`` OR all-zero OR ``cocycle_relaxation=0`` the edge factor
    delta=0 -> exp(0)=I, so Omega = exp(phi_i) exp(-phi_j) is the flat (Regime-I) cocycle EXACTLY
    -- the byte-identity oracle (verified by tests/test_regime_ii.py). A nonzero W gives the
    non-trivial triangle holonomy ``metrics.holonomy_deviation`` was built to read.

    COST: unlike flat (mu-independent, O(N) vertex exponentials), Omega here depends on the CURRENT
    belief means mu and the edge factor is a PER-EDGE K x K matrix exponential -- O(N^2) matrix
    exponentials per build, and the build must be repeated as mu updates across E-step iterations.
    The Frobenius clamp in :func:`stable_matrix_exp_pair` (max_norm=15) also bounds ||delta . G||;
    safe at zero-init W, a note for a large trained connection.

    Returns the SAME dict shape as the flat builder: 'exp_phi' (B,N,K,K), 'exp_neg_phi' (B,N,K,K),
    'Omega' (B,N,N,K,K).
    """
    # Vertex factors exp(phi_i), exp(-phi_j): reuse the flat builder verbatim (the dominant cost is
    # the O(N^2) edge exps below, so this extra Omega matmul is cheap and we just discard its Omega).
    flat = compute_transport_operators(phi, group, gauge_mode=gauge_mode)
    exp_phi, exp_neg_phi = flat["exp_phi"], flat["exp_neg_phi"]                 # (B, N, K, K)

    # Flat fast path: no connection at all (None), the homotopy collapses it (alpha=0), or the
    # vertex factors are trivial -> delta plays no role -> Omega is exactly the flat cocycle. Skip the
    # O(N^2) edge exps. NOTE: we deliberately do NOT short-circuit on an all-ZERO (but grad-requiring)
    # connection_W: at W=0 the edge factor exp(delta)=I numerically (so the W=0->flat oracle holds to
    # float tolerance), but d Omega / d W at W=0 is the generator structure (exp'(0)=I), NOT zero --
    # short-circuiting there would sever the autograd graph and freeze the parameter at init. The full
    # einsum path keeps W in the graph so the loss backpropagates to it.
    if connection_W is None or cocycle_relaxation == 0.0 or gauge_mode == "trivial":
        return flat

    generators = group.generators                                              # (n_gen, K, K)
    # delta_ij^a = cocycle_relaxation * mu_i^T W^a mu_j  -> (B, N, N, n_gen)
    delta = cocycle_relaxation * torch.einsum("bik,akl,bjl->bija", mu, connection_W, mu)
    # delta_ij . G = sum_a delta_ij^a G_a  -> (B, N, N, K, K) Lie-algebra edge matrix
    delta_mat = torch.einsum("bija,akl->bijkl", delta, generators)
    # Per-edge group element exp(delta_ij . G); reuse the stable float64-island / block-exp machinery
    # (only_forward: the edge factor enters Omega once, no exp(-delta) needed).
    block_dims = group.irrep_dims if len(group.irrep_dims) > 1 else None
    exp_delta, _ = stable_matrix_exp_pair(
        delta_mat, skew_symmetric=group.skew_symmetric, only_forward=True, block_dims=block_dims,
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
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    r"""exp(M) and optionally exp(-M) with Frobenius-norm clamp + float64 upcast.

    Frobenius-norm clamp + float64 upcast keep matrix_exp stable for large ||M||.

    ``block_dims`` (audit 4b): when M is block-diagonal with these blocks (e.g. block_glk's
    GL(d_head)^H), exp(M) is exactly block-diagonal with the per-block exponentials, so each
    d_head x d_head block is exponentiated independently -- an O(H * d_head^3) cost instead of
    O(K^3) for the full K x K. The result is BIT-equivalent to the full exp (the global
    Frobenius clamp is applied to the WHOLE matrix first, and each block keeps the dtype the
    full-K path would pick, so neither the scale nor the precision changes). ``None`` (a single
    block, a cross-coupled basis, or a skew group) takes the full-matrix path unchanged.
    """
    # Global Frobenius clamp on the FULL matrix (one scale for all blocks) -- identical to the
    # un-blocked path, so block slicing below cannot change the operator.
    mat_norm = matrix.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    scale = (max_norm / mat_norm).clamp(max=1.0)
    matrix = matrix * scale

    d = matrix.shape[-1]
    orig_dtype = matrix.dtype
    # The full-K path's dtype choice; the per-block path forces the SAME dtype so a small block
    # (d_head < dim_threshold) does not silently drop to float32 and drift from the full exp.
    up_dtype = torch.float64 if d >= dim_threshold else torch.float32

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
        d = block_dims[0]
        blocks = torch.stack(
            [matrix[..., h * d:(h + 1) * d, h * d:(h + 1) * d] for h in range(len(block_dims))],
            dim=0,
        ).contiguous()                                          # (H, ..., d, d)
        exps = torch.linalg.matrix_exp(blocks)                  # one batched call
        for h in range(len(block_dims)):
            out[..., h * d:(h + 1) * d, h * d:(h + 1) * d] = exps[h]
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


def compute_transport_operators_direct(
    omega:      torch.Tensor,             # (B, N, K, K) per-token group elements Omega_i

    *,
    gauge_mode: str   = "learned",        # 'learned' (flat cocycle) or 'trivial'
    eps:        float = 1e-6,
) -> TransportDict:
    r"""Direct-Omega transport Omega_ij = Omega_i @ Omega_j^{-1} (general GL(K)).

    Flat (Regime I) direct-Omega transport. Reaches all of GL(K) (det may be < 0;
    needs an external det penalty to stay invertible). Inverse via LU solve (exact
    cocycle), with a ridge then pinv fallback for near-singular Omega. 'trivial'
    returns Omega=I. The 'constant' mode is intentionally unsupported (raises
    ValueError). The ridge ``eps`` is configurable (default 1e-6).
    Returns 'omega_i' (B,N,K,K), 'omega_j_inv' (B,N,K,K), 'Omega' (B,N,N,K,K).
    """
    B, N, K, _ = omega.shape
    dtype = omega.dtype
    device = omega.device

    if gauge_mode == "trivial":
        eye_K = torch.eye(K, device=device, dtype=dtype)
        return {
            "omega_i":     eye_K.expand(B, N, K, K).contiguous(),
            "omega_j_inv": eye_K.expand(B, N, K, K).contiguous(),
            "Omega":       eye_K.expand(B, N, N, K, K).contiguous(),
        }
    if gauge_mode != "learned":
        raise ValueError(f"gauge_mode must be 'learned' or 'trivial', got {gauge_mode!r}")

    eye_K = torch.eye(K, device=device, dtype=dtype)
    try:
        omega_j_inv = torch.linalg.solve(omega, eye_K.expand_as(omega))
    except (torch.linalg.LinAlgError, RuntimeError):
        try:
            omega_j_inv = torch.linalg.solve(omega + eps * eye_K, eye_K.expand_as(omega))
        except (torch.linalg.LinAlgError, RuntimeError):
            omega_j_inv = torch.linalg.pinv(omega)

    omega_ij = torch.einsum("bikl,bjlm->bijkm", omega, omega_j_inv)
    return {"omega_i": omega, "omega_j_inv": omega_j_inv, "Omega": omega_ij}


def transport_mean(
    omega: torch.Tensor,             # (..., N, N, K, K) pairwise transport
    mu:    torch.Tensor,             # (..., N, K) source (key, index j) means
) -> torch.Tensor:
    r"""Gauge action on means: mu_t[i,j] = Omega_ij @ mu_j. Returns (..., N, N, K).

    Rank-agnostic via the leading ellipsis: an optional batch axis (B,N,N,K,K)+(B,N,K)
    flows through unchanged, and the unbatched (N,N,K,K)+(N,K) call is identical -- so the
    same primitive serves the batched forward and the unbatched diagnostics path.
    """
    return torch.einsum("...ijkl,...jl->...ijk", omega, mu)


def transport_covariance(
    omega: torch.Tensor,             # (..., N, N, K, K) pairwise transport
    sigma: torch.Tensor,             # (..., N, K) diagonal OR (..., N, K, K) full

    *,
    diagonal_out: Optional[bool] = None,
) -> torch.Tensor:
    r"""Sandwich action Sigma_t[i,j] = Omega_ij Sigma_j Omega_ij^T.

    Full input (...,N,K,K) -> full (...,N,N,K,K). Diagonal input (...,N,K) -> the
    diagonal approximation (...,N,N,K), Sigma_t[i,j,k] = sum_l Omega_ijkl^2
    sigma_jl (the diagonal of the full sandwich). Rank-agnostic via the leading
    ellipsis (optional batch axis); diagonal vs full is detected by the rank gap
    ``sigma.dim() == omega.dim() - 2``, which holds with or without the batch axis.
    """
    is_diag = sigma.dim() == omega.dim() - 2 if diagonal_out is None else diagonal_out
    if is_diag:
        return torch.einsum("...ijkl,...ijkl,...jl->...ijk", omega, omega, sigma)
    return torch.einsum("...ijkl,...jlm,...ijnm->...ijkn", omega, sigma, omega)


def omega_to_block_exp_pairs(
    omega:      torch.Tensor,        # (B, N, K, K) per-token group elements
    irrep_dims: List[int],           # block sizes; sum == K

    *,
    eps:        float = 1e-6,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    r"""Slices a block-diagonal Omega into per-block (block, block_inv) pairs.

    Per-block inverse via solve, with ridge then pinv fallback. Returns a list
    aligned with irrep_dims, each a pair of (B, N, d, d) tensors.
    """
    id_sum = sum(irrep_dims)
    K = omega.shape[-1]
    if id_sum != K:
        raise ValueError(f"omega_to_block_exp_pairs: sum(irrep_dims)={id_sum} != K={K}")

    results: List[Tuple[torch.Tensor, torch.Tensor]] = []
    start = 0
    for d in irrep_dims:
        end = start + d
        omega_blk = omega[:, :, start:end, start:end].contiguous()
        eye_d = torch.eye(d, device=omega_blk.device, dtype=omega_blk.dtype)
        try:
            omega_blk_inv = torch.linalg.solve(omega_blk, eye_d.expand_as(omega_blk))
        except (torch.linalg.LinAlgError, RuntimeError):
            try:
                omega_blk_inv = torch.linalg.solve(
                    omega_blk + eps * eye_d, eye_d.expand_as(omega_blk)
                )
            except (torch.linalg.LinAlgError, RuntimeError):
                omega_blk_inv = torch.linalg.pinv(omega_blk)
        results.append((omega_blk, omega_blk_inv))
        start = end
    return results
