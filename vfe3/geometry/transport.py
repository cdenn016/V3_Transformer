r"""Gauge transport for VFE_3.0 (Regime I, Gaussian / location-scale specific).

Two parameterizations of the flat (Regime I) transport:
  phi (exp):    Omega_ij = exp(phi_i . G) exp(-phi_j . G) in GL+(K) (det>0).
  omega_direct: Omega_ij = Omega_i Omega_j^{-1} for general GL(K) (det may be <0).
Belief action: mu -> Omega @ mu, Sigma -> Omega @ Sigma @ Omega^T (sandwich;
diagonal-covariance fast path plus an exact full-covariance congruence). Regime II and retractions are separate modules;
gauge-RoPE folds a positional rotation into the transport via :class:`RopeTransport`.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch

from vfe3.geometry.groups import GaugeGroup
from vfe3.numerics import safe_cholesky

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

    ``mean_per_head`` (Tier-1 perf toggle, default False = byte-identical): when set,
    ``transport_mean`` contracts each gauge block separately (``_factored_per_head_mean``, the
    mean twin of the per-head diagonal cov) instead of the dense full-K einsum -- the same sum
    with the exactly-zero off-block terms dropped, equal to fp32 reassociation.
    """

    exp_phi:       torch.Tensor           # (..., N, K, K) exp(phi_i . G)
    exp_neg_phi:   torch.Tensor           # (..., N, K, K) exp(-phi_j . G)
    irrep_dims:    List[int]              # equal block sizes; sum == K, len > 1
    mean_per_head: bool = False           # transport_mean contracts per gauge block (fp32 reassociation only)

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

    ``on_value`` (default True = the coherent single-gauge path) factors the transport into an
    ATTENTION gauge and a VALUE gauge (GL(K)_attention.tex:1909): the attention score
    D_KL(q_i || R_i Omega_ij R_j^T q_j) always carries the rotation, but with ``on_value=False`` the
    value aggregation mu_hat_i = sum_j beta_ij Omega_ij mu_j uses the UN-rotated base transport --
    exactly RoPE's "position-dependent attention, position-independent values" asymmetry. The flag is
    consumed at the GRADIENT layer: ``gradients/oracle.py`` builds the value-gauge coupling energy from
    ``base`` while beta comes from the rotated score energy, and ``gradients/kernels.py`` routes the
    decoupled case to the oracle (beta is no longer the coupling sum's stationary point, so the
    closed-form envelope kernel does not apply). ``transport_mean`` / ``transport_covariance`` always
    honour the rotation (the score path); the value path transports on ``base`` directly.
    """

    base:     'torch.Tensor | FactoredTransport'  # (N,N,K,K) dense OR factored transport
    rope:     torch.Tensor                        # (N, K, K) block-diagonal orthogonal rotation
    on_cov:   bool = False
    on_value: bool = True                         # False -> value aggregation uses the UN-rotated base (RoPE Q/K only)


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
_TRANSPORT_NEEDS_MU:    set = set()   # regimes whose Omega builder reads the belief means mu
_TRANSPORT_NEEDS_SIGMA: set = set()   # regimes whose Omega builder reads the belief covariance sigma
_TRANSPORT_BATCH_INDEPENDENT: set = set()   # regimes whose Omega is the SAME for every sequence in the
#                                             batch (depends only on a model parameter, not phi/mu/sigma),
#                                             so the builder returns a batch-collapsed (N,N,K,K) Omega that
#                                             broadcasts downstream instead of a dense (B,N,N,K,K).


def register_transport(
    name: str, *, needs_mu: bool = False, needs_sigma: bool = False, batch_independent: bool = False,
    override: bool = False,
) -> Callable:
    """Decorator registering a transport (connection-regime) builder under ``name``.

    ``needs_mu``/``needs_sigma`` are state-routing metadata: they declare which belief fields the
    regime's Omega builder consumes, so callers feed mu/sigma by querying the registry rather than
    matching literal mode names. Declaring them here keeps the add-by-registering contract -- a new
    stateful regime advertises its requirements at registration, not at every call site.

    ``batch_independent`` declares that the builder's ``Omega`` does NOT depend on the batch (it is a
    function of a model parameter only -- the bare direct link ``regime_ii_link``), so the builder
    returns a batch-collapsed ``(N, N, K, K)`` Omega that ``transport_mean`` / ``transport_covariance``
    broadcast across the batch (the D3 memory collapse). ``_transport`` reads this flag to skip the
    per-sequence ``[0]`` strip it applies to ordinary ``(B, N, N, K, K)`` builders.

    Duplicate keys fail closed (audit 2026-07-01 round-3): a second registration under an existing
    name silently shadowed the first, so a config-selected seam could dispatch to an unintended
    implementation. Pass ``override=True`` to replace deliberately; the replacement DISCARDS the
    name from every metadata set first and re-adds per the new flags, so stale membership cannot
    survive an override.
    """
    def _wrap(fn: Callable[..., TransportDict]) -> Callable[..., TransportDict]:
        if name in _TRANSPORTS and not override:
            raise KeyError(f"transport mode {name!r} already registered; pass override=True to replace")
        _TRANSPORTS[name] = fn
        _TRANSPORT_NEEDS_MU.discard(name)             # override: drop stale metadata before re-adding
        _TRANSPORT_NEEDS_SIGMA.discard(name)
        _TRANSPORT_BATCH_INDEPENDENT.discard(name)
        if needs_mu:
            _TRANSPORT_NEEDS_MU.add(name)
        if needs_sigma:
            _TRANSPORT_NEEDS_SIGMA.add(name)
        if batch_independent:
            _TRANSPORT_BATCH_INDEPENDENT.add(name)
        return fn
    return _wrap


def get_transport(name: str) -> Callable[..., TransportDict]:
    """Return the registered transport builder (KeyError-with-available-list if absent)."""
    if name not in _TRANSPORTS:
        raise KeyError(f"no transport {name!r}; available: {sorted(_TRANSPORTS)}")
    return _TRANSPORTS[name]


def gauge_invariant_edge_features(
    mu_q:   torch.Tensor,             # (..., K) query means
    cov_q:  torch.Tensor,             # (..., K, K) query covariances (SPD)
    mu_kt:  torch.Tensor,             # (..., K) transported key means
    cov_kt: torch.Tensor,             # (..., K, K) transported key covariances (SPD)

    *,
    eps:    float = 1e-6,
) -> torch.Tensor:                    # (..., 3) [Mahalanobis, trace, log-det] edge features
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
    K             = mu_q.shape[-1]
    mu_q,  cov_q  = mu_q.double(),  cov_q.double()
    mu_kt, cov_kt = mu_kt.double(), cov_kt.double()
    eye = torch.eye(K, dtype=cov_kt.dtype, device=cov_kt.device)

    L_s, ok_s = safe_cholesky(cov_kt + eps * eye, rounds=5)   # S = cov_kt = L_s L_s^T
    L_q, ok_q = safe_cholesky(cov_q  + eps * eye, rounds=5)

    delta_mu = (mu_q - mu_kt).unsqueeze(-1)                # (..., K, 1) prediction error
    sol      = torch.cholesky_solve(delta_mu, L_s)         # S^{-1} (mu_q - mu_kt)
    mahal    = (delta_mu * sol).sum(dim=(-2, -1))          # (...,)  Mahalanobis

    sinv_covq = torch.cholesky_solve(cov_q, L_s)           # S^{-1} Sigma_q  (..., K, K)
    trace     = torch.diagonal(sinv_covq, dim1=-2, dim2=-1).sum(dim=-1)   # (...,)  tr(S^{-1} Sigma_q)

    logdet_s = 2.0 * torch.log(torch.diagonal(L_s, dim1=-2, dim2=-1)).sum(dim=-1)
    logdet_q = 2.0 * torch.log(torch.diagonal(L_q, dim1=-2, dim2=-1)).sum(dim=-1)
    logdet   = logdet_s - logdet_q                         # (...,)  log det S - log det Sigma_q

    feats = torch.stack((mahal, trace, logdet), dim=-1)    # (..., 3)
    ok    = (ok_s & ok_q).unsqueeze(-1)                    # (..., 1) PD on both factors
    feats = torch.where(ok, feats, feats.new_tensor(float("nan")))   # non-PD edge -> NaN -> kl_max
    return feats.to(orig_dtype)                            # back to the caller's working dtype


@register_transport("flat")
def _build_flat(
    phi:        torch.Tensor,             # (B, N, n_gen) gauge frames
    group:      GaugeGroup,               # supplies generators, skew flag, irrep_dims

    *,
    gauge_mode:              str   = "learned",   # 'learned' (Regime I flat) or 'trivial'
    exp_fp64_mode:           str   = "dim",       # stable_matrix_exp_pair float64-island keying ('dim' | 'norm')
    exp_fp64_norm_threshold: float = 5.0,         # 'norm' mode: max clamped block ||M||_F upcast threshold
    clamp_monitor:           bool  = False,       # opt-in: warn when the exp Frobenius clamp fires
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
                                       clamp_monitor=clamp_monitor)


@register_transport("regime_ii", needs_mu=True)
def _build_regime_ii(
    phi:                torch.Tensor,             # (B, N, n_gen) gauge frames
    group:              GaugeGroup,               # supplies generators, skew flag, irrep_dims

    *,
    gauge_mode:         str                       = "learned",   # 'learned' (flat vertex factors) | 'trivial'
    cocycle_relaxation: float                     = 1.0,         # homotopy alpha in [0,1]; 0 -> flat
    delta_soft_cap:     float                     = 12.0,        # smooth bound on ||delta_ij . G||_F (< exp clamp max_norm=15)
    clamp_monitor:      bool                      = False,       # opt-in: warn when the exp Frobenius clamp fires
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
    # Flat fast path: no connection at all (None), the homotopy collapses it (alpha=0), or the
    # vertex factors are trivial -> delta plays no role -> Omega is exactly the flat cocycle. Skip the
    # O(N^2) edge exps. NOTE: we deliberately do NOT short-circuit on an all-ZERO (but grad-requiring)
    # connection_W: at W=0 the edge factor exp(delta)=I numerically (so the W=0->flat oracle holds to
    # float tolerance), but d Omega / d W at W=0 is the generator structure (exp'(0)=I), NOT zero --
    # short-circuiting there would sever the autograd graph and freeze the parameter at init. The full
    # einsum path keeps W in the graph so the loss backpropagates to it.
    if connection_W is None or cocycle_relaxation == 0.0 or gauge_mode == "trivial":
        return compute_transport_operators(phi, group, gauge_mode=gauge_mode, clamp_monitor=clamp_monitor)

    # Vertex factors exp(phi_i), exp(-phi_j) in FACTORED form (audit 2026-06-10 F8a): the same
    # stable exp machinery as the flat builder, WITHOUT materializing the dense (B, N, N, K, K)
    # flat Omega this path would immediately discard.
    fac = build_factored_transport(phi, group, gauge_mode=gauge_mode, clamp_monitor=clamp_monitor)
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
        fro_sq      = delta_mat_c.pow(2).sum(dim=(-2, -1), keepdim=True)
        delta_mat_c = delta_mat_c * torch.rsqrt(1.0 + fro_sq / (delta_soft_cap * delta_soft_cap))
        # Per-edge group element exp(delta_ij . G); reuse the stable block-exp machinery
        # (only_forward: the edge factor enters Omega once, no exp(-delta) needed). exp_dim keys
        # the float64-island decision on the per-head block actually exponentiated (audit
        # 2026-06-10 F8c; the soft cap above keeps the blocks in the well-conditioned exp regime).
        exp_delta_c, _ = stable_matrix_exp_pair(
            delta_mat_c, skew_symmetric=group.skew_symmetric, only_forward=True,
            block_dims=block_dims, exp_dim=exp_dim, clamp_monitor=clamp_monitor,
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


@register_transport("regime_ii_covariant", needs_mu=True, needs_sigma=True)
def _build_regime_ii_covariant(
    phi:                torch.Tensor,             # (B, N, n_gen) gauge frames
    group:              GaugeGroup,               # supplies generators, skew flag, irrep_dims

    *,
    gauge_mode:         str                       = "learned",   # 'learned' (flat vertex factors) | 'trivial'
    cocycle_relaxation: float                     = 1.0,         # homotopy alpha in [0,1]; 0 -> flat
    delta_soft_cap:     float                     = 12.0,        # smooth bound on ||delta_ij . G||_F
    clamp_monitor:      bool                      = False,       # opt-in: warn when the exp Frobenius clamp fires
    mu:                 Optional[torch.Tensor]    = None,        # (B, N, K) QUERY means
    sigma:              Optional[torch.Tensor]    = None,        # (B, N, K) diag OR (B, N, K, K) full QUERY covariance
    connection_M:       Optional[torch.Tensor]    = None,        # (n_gen, 3) learned invariant-feature map (NN exception)
    mu_key:             Optional[torch.Tensor]    = None,        # (B, N, K) KEY means (None -> mu; oracle detaches)
    sigma_key:          Optional[torch.Tensor]    = None,        # (B, N, ...) KEY covariance (None -> sigma)
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
    'Omega' (B,N,N,K,K).
    """
    # Flat fast path (mirrors regime_ii): no connection, alpha=0, or trivial gauge -> flat cocycle.
    # An all-zero (but grad-requiring) M is NOT short-circuited -- delta=0 reduces to flat to fp32,
    # but d Omega / d M at M=0 is the generator structure (exp'(0)=I), so the generic path keeps M
    # in the autograd graph (it would otherwise freeze at init).
    if connection_M is None or cocycle_relaxation == 0.0 or gauge_mode == "trivial":
        return compute_transport_operators(phi, group, gauge_mode=gauge_mode, clamp_monitor=clamp_monitor)

    # Contract guard (audit 2026-06-18): with a connection the edge features need the query belief
    # (mu, sigma); a missing one would otherwise raise an opaque AttributeError on `.dim()` below.
    if mu is None or sigma is None:
        raise ValueError(
            "regime_ii_covariant requires query means `mu` and covariance `sigma` when "
            f"`connection_M` is provided; got mu={'None' if mu is None else 'tensor'}, "
            f"sigma={'None' if sigma is None else 'tensor'}."
        )

    fac = build_factored_transport(phi, group, gauge_mode=gauge_mode, clamp_monitor=clamp_monitor)
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

        feats_c = gauge_invariant_edge_features(mu_q_c, cov_q_c, mu_kt_c, cov_kt_c)        # (B, C, N, 3) f64
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
        fro_sq      = delta_mat_c.pow(2).sum(dim=(-2, -1), keepdim=True)
        delta_mat_c = delta_mat_c * torch.rsqrt(1.0 + fro_sq / (delta_soft_cap * delta_soft_cap))

        exp_delta_c, _ = stable_matrix_exp_pair(
            delta_mat_c, skew_symmetric=group.skew_symmetric, only_forward=True,
            block_dims=block_dims, exp_dim=exp_dim, clamp_monitor=clamp_monitor,
        )                                                                      # (B, C, N, K, K)
        # Omega_ij = exp(phi_i) @ exp_delta_ij @ exp(-phi_j)
        omega_chunks.append(
            torch.einsum("bikl,bijlm,bjmn->bijkn", exp_phi_c, exp_delta_c, exp_neg_phi))

    omega = torch.cat(omega_chunks, dim=1) if len(omega_chunks) > 1 else omega_chunks[0]
    return {"exp_phi": exp_phi, "exp_neg_phi": exp_neg_phi, "Omega": omega}


def _direct_link_edge_exp(
    connection_L:  torch.Tensor,             # (>=N, >=N, n_gen) learned direct-link table A
    group:         GaugeGroup,               # supplies generators, skew flag, irrep_dims

    n_tok:         int,                       # active sequence length N

    *,
    link_alpha:    float = 1.0,
    link_soft_cap: float = 6.0,
    clamp_monitor: bool  = False,             # opt-in: warn when the exp Frobenius clamp fires
    device:        Optional[torch.device] = None,
    dtype:         Optional[torch.dtype]  = None,
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
        fro_sq = link_mat.pow(2).sum(dim=(-2, -1), keepdim=True)
        link_mat = link_mat * torch.rsqrt(1.0 + fro_sq / (link_soft_cap * link_soft_cap))
    block_dims = group.irrep_dims if len(group.irrep_dims) > 1 else None
    exp_link, _ = stable_matrix_exp_pair(
        link_mat, skew_symmetric=group.skew_symmetric, only_forward=True,
        block_dims=block_dims, exp_dim=(max(block_dims) if block_dims is not None else None),
        clamp_monitor=clamp_monitor,
    )                                                                              # (N, N, K, K)
    return exp_link.to(dtype)


@register_transport("regime_ii_link", batch_independent=True)
def _build_regime_ii_link(
    phi:                torch.Tensor,             # (B, N, n_gen) gauge frames (IGNORED: bare link reads only connection_L)
    group:              GaugeGroup,               # supplies generators, skew flag, irrep_dims

    *,
    gauge_mode:         str                    = "learned",
    link_alpha:         float                  = 1.0,
    link_soft_cap:      float                  = 6.0,
    clamp_monitor:      bool                   = False,  # opt-in: warn when the exp Frobenius clamp fires
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

    BATCH-INDEPENDENT: ``Omega`` is a function of ``connection_L`` only, so the builder returns it at
    logical ``(N, N, K, K)`` (NO batch axis) and ``transport_mean`` / ``transport_covariance`` broadcast
    it across the batch -- the D3 memory collapse (a dense ``(B, N, N, K, K)`` would OOM at the stated
    operating point). ``_transport`` reads the ``batch_independent`` registry flag to skip its
    per-sequence ``[0]`` strip.

    Returns the SAME dict keys as the flat builder: 'exp_phi' (B,N,K,K), 'exp_neg_phi' (B,N,K,K) (both
    UNUSED for the bare link, kept for dict parity), 'Omega' (N,N,K,K).
    """
    fac = build_factored_transport(phi, group, gauge_mode=gauge_mode, clamp_monitor=clamp_monitor)
    exp_phi, exp_neg_phi = fac.exp_phi, fac.exp_neg_phi                         # (B, N, K, K), unused
    N = phi.shape[1]
    K = group.generators.shape[-1]
    device, dtype = phi.device, phi.dtype
    # Flat (identity-link) fast path: no connection or link_alpha=0 -> Omega = I exactly. NOTE we do
    # NOT short-circuit on an all-ZERO (grad-requiring) connection_L: at A=0 exp(A)=I numerically, but
    # d Omega / d connection_L there is the generator structure (exp'(0)=I), so the generic path keeps
    # connection_L in the autograd graph (short-circuiting would freeze it at init).
    if connection_L is None or link_alpha == 0.0:
        omega = torch.eye(K, device=device, dtype=dtype).expand(N, N, K, K).contiguous()
        return {"exp_phi": exp_phi, "exp_neg_phi": exp_neg_phi, "Omega": omega}
    omega = _direct_link_edge_exp(connection_L, group, N, link_alpha=link_alpha,
                                  link_soft_cap=link_soft_cap, clamp_monitor=clamp_monitor,
                                  device=device, dtype=dtype)                                # (N,N,K,K)
    return {"exp_phi": exp_phi, "exp_neg_phi": exp_neg_phi, "Omega": omega}


@register_transport("regime_ii_link_charted")
def _build_regime_ii_link_charted(
    phi:                torch.Tensor,             # (B, N, n_gen) gauge frames
    group:              GaugeGroup,               # supplies generators, skew flag, irrep_dims

    *,
    gauge_mode:         str                    = "learned",
    link_alpha:         float                  = 1.0,
    link_soft_cap:      float                  = 6.0,
    clamp_monitor:      bool                   = False,  # opt-in: warn when the exp Frobenius clamp fires
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
    but ``phi``-dependent (so it travels the dense, per-sequence path and is the more expensive of the
    two link modes). Its ``A=0`` limit is the Regime-I flat cocycle ``exp(phi_i)exp(-phi_j)`` (NOT the
    bare identity-link limit). A nonzero ``connection_L`` gives non-trivial triangle holonomy.

    Returns the flat builder's dict shape: 'exp_phi' (B,N,K,K), 'exp_neg_phi' (B,N,K,K), 'Omega'
    (B,N,N,K,K) (genuinely batched, because exp_phi is per-sequence).
    """
    # Flat (cocycle) fast path: no connection / link_alpha=0 / trivial gauge -> exp(phi_i)exp(-phi_j)
    # byte-identically. A zero (grad-requiring) connection_L is NOT short-circuited (autograd at A=0).
    if connection_L is None or link_alpha == 0.0 or gauge_mode == "trivial":
        return compute_transport_operators(phi, group, gauge_mode=gauge_mode, clamp_monitor=clamp_monitor)
    fac = build_factored_transport(phi, group, gauge_mode=gauge_mode, clamp_monitor=clamp_monitor)
    exp_phi, exp_neg_phi = fac.exp_phi, fac.exp_neg_phi                         # (B, N, K, K)
    N = phi.shape[1]
    exp_link = _direct_link_edge_exp(connection_L, group, N, link_alpha=link_alpha,
                                     link_soft_cap=link_soft_cap, clamp_monitor=clamp_monitor,
                                     device=phi.device, dtype=phi.dtype)                     # (N,N,K,K)
    # Omega_ij = exp(phi_i) @ exp(link A_ij) @ exp(-phi_j); the batch-independent exp_link broadcasts.
    omega = torch.einsum("bikl,ijlm,bjmn->bijkn", exp_phi, exp_link, exp_neg_phi)   # (B, N, N, K, K)
    return {"exp_phi": exp_phi, "exp_neg_phi": exp_neg_phi, "Omega": omega}


def _direct_link_dense_bytes(batch: int, n_tok: int, k: int, dtype: torch.dtype) -> int:
    r"""Bytes of a DENSE batched ``(B, N, N, K, K)`` link transport.

    This is the materialized cost the bare ``regime_ii_link`` AVOIDS: its Omega is batch-independent,
    so the builder returns it at logical ``(N, N, K, K)`` (a factor ``B`` smaller) and lets
    ``transport_mean`` / ``transport_covariance`` broadcast across the batch. The charted mode IS
    genuinely ``(B, N, N, K, K)`` (per-sequence frames), so this is its true footprint."""
    bytes_per = torch.tensor([], dtype=dtype).element_size()
    return batch * n_tok * n_tok * k * k * bytes_per


# Frobenius-norm clamp for stable_matrix_exp_pair: above this the returned factor is the surrogate
# exp(max_norm*M/||M||_F), NOT exp(M). Exported so the M-step drift monitor
# (train._warn_phi_transport_clamp) trips at the SAME norm the clamp fires at -- the two cannot
# diverge (audit 2026-07-06 M2).
TRANSPORT_CLAMP_MAX_NORM: float = 15.0


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
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    r"""exp(M) and optionally exp(-M) with Frobenius-norm clamp + float64 upcast.

    Frobenius-norm clamp + float64 upcast keep matrix_exp stable for large ||M||.

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
    with torch.no_grad():
        mat_norm = matrix.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
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
        up_dtype = torch.float64 if d_eff >= dim_threshold else torch.float32
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
    gauge_mode:              str   = "learned",   # 'learned' (Regime I flat) or 'trivial'
    exp_fp64_mode:           str   = "dim",       # stable_matrix_exp_pair float64-island keying ('dim' | 'norm')
    exp_fp64_norm_threshold: float = 5.0,         # 'norm' mode: max clamped block ||M||_F upcast threshold
    clamp_monitor:           bool  = False,       # opt-in: warn when the exp Frobenius clamp fires
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
    )
    omega = torch.einsum("bikl,bjlm->bijkm", exp_phi, exp_neg_phi)
    return {"exp_phi": exp_phi, "exp_neg_phi": exp_neg_phi, "Omega": omega}


def build_transport_from_element(
    omega:  torch.Tensor,             # (B, N, K, K) per-token GL(K) element U_i (block-diagonal for block_glk)
    group:  GaugeGroup,
) -> 'FactoredTransport | TransportDict':
    r"""Exp-free flat cocycle from a stored group element: Omega_ij = U_i U_j^{-1}.

    The 'omega_direct' parameterization stores the frame as the element U_i itself rather than the
    Lie-algebra coordinate phi_i, so the transport is assembled WITHOUT any matrix exponential --
    only the inverse U_j^{-1}. The FactoredTransport / builder-dict slots exp_phi / exp_neg_phi are
    filled with U_i and U_j^{-1} directly; every downstream consumer (transport_mean,
    transport_covariance, RoPE) reads only those two slots, so nothing else changes.

    U_j^{-1} is computed in a float64 island (the congruence Omega Sigma Omega^T squares cond(U); the
    inverse degrades as det U -> 0, which the free-energy barrier keeps a trained U away from). For
    the equal-block groups (block_glk) a FactoredTransport is returned so the per-head fast paths run;
    for a single block (glk) the dense {'exp_phi','exp_neg_phi','Omega'} dict is returned (matching
    compute_transport_operators' return shape).
    """
    if group.skew_symmetric:
        u_inv = omega.transpose(-1, -2)                           # U^{-1} = U^T exactly (orthogonal group), free
    else:
        with torch.amp.autocast(omega.device.type, enabled=False):   # fp64 island (non-compact inverse)
            u_inv = torch.linalg.inv(omega.double()).to(omega.dtype)  # (B, N, K, K)
    block_dims = group.irrep_dims
    if len(block_dims) > 1 and len(set(block_dims)) == 1:
        return FactoredTransport(exp_phi=omega, exp_neg_phi=u_inv, irrep_dims=list(block_dims))
    Omega = torch.einsum("...ikl,...jlm->...ijkm", omega, u_inv)   # (B, N, N, K, K)
    return {"exp_phi": omega, "exp_neg_phi": u_inv, "Omega": Omega}


def build_factored_transport(
    phi:        torch.Tensor,             # (..., N, n_gen) gauge frames (optional leading batch axis)
    group:      GaugeGroup,               # block-diagonal with equal blocks (len(irrep_dims) > 1)

    *,
    gauge_mode:              str   = "learned",   # 'learned' (Regime I flat) or 'trivial'
    exp_fp64_mode:           str   = "dim",       # stable_matrix_exp_pair float64-island keying ('dim' | 'norm')
    exp_fp64_norm_threshold: float = 5.0,         # 'norm' mode: max clamped block ||M||_F upcast threshold
    clamp_monitor:           bool  = False,       # opt-in: warn when the exp Frobenius clamp fires
    mean_per_head:           bool  = False,       # container flag: transport_mean contracts per gauge block
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
        return FactoredTransport(exp_phi=eye, exp_neg_phi=eye, irrep_dims=list(group.irrep_dims),
                                 mean_per_head=mean_per_head)
    if gauge_mode != "learned":
        raise ValueError(f"gauge_mode must be 'learned' or 'trivial', got {gauge_mode!r}")

    phi_matrix = torch.einsum("...na,aij->...nij", phi, group.generators)
    block_dims = group.irrep_dims if len(group.irrep_dims) > 1 else None
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
    )
    return FactoredTransport(exp_phi=exp_phi, exp_neg_phi=exp_neg_phi, irrep_dims=list(group.irrep_dims),
                             mean_per_head=mean_per_head)


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
        return torch.einsum("...ikl,...ijl->...ijk", omega.rope, t)   # post-rotate by R_i
    if isinstance(omega, FactoredTransport):
        if omega.mean_per_head:
            return _factored_per_head_mean(omega, mu)
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
            return _factored_full_covariance(omega, sigma)
        return _factored_diagonal_covariance(omega, sigma)
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


def _factored_per_head_mean(
    factored: FactoredTransport,
    mu:       torch.Tensor,               # (..., N, K) source (key, index j) means
) -> torch.Tensor:                        # (..., N, N, K) transported means
    r"""Per-head mean transport from the factored exps (the mean twin of
    ``_factored_diagonal_covariance``; cfg.transport_mean_per_head, Tier-1 perf toggle).

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
    Omega^(h') on the right -- exactly what the dense path computes); the win is that the dense
    (..., N, N, K, K) Omega is never built (only the H per-head (..., N, N, d, d) operators) and each
    block-pair sandwich runs in its own float64 island (M4) at the BLOCK scale, not the full K x K.
    Value-equal to the dense ``transport_covariance(to_dense_omega(), sigma)``; pinned by
    tests/test_fullcov_alpha_roadmap_2026_06_13.py.
    """
    dims = factored.irrep_dims
    K = sum(dims)
    batch = sigma.shape[:-3]
    N = sigma.shape[-3]
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

    # (h, h') output block = Omega^(h)_ij Sigma_j^(h,h') (Omega^(h')_ij)^T, in a float64 island (the
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
