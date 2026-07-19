r"""Optimized hand-derived belief-gradient kernels for VFE_3.0, with oracle fallback.

A family-keyed registry of analytic (grad_mu, grad_sigma) kernels for the QUERY-SIDE
(filtering) gradient. belief_gradients() uses the registered kernel only for the
filtering + gaussian_diagonal + KL (renyi_order=1) + canonical case; every other case
(smoothing, non-KL family, Renyi alpha != 1, surrogate) FALLS BACK to the autograd
oracle -- so a new divergence works immediately and correctly, accelerated later by
registering a kernel. Kernels return RAW Euclidean dF (no preconditioning/retraction).
"""

from typing import Callable, Dict, List, Optional, Tuple

import torch

from vfe3.alpha_i import alpha_gradient_coefficient, alpha_is_per_coord
from vfe3.families.base import get_family
from vfe3.families.gaussian import diag_kl_unclamped, diag_kl_unclamped_per_coord
from vfe3.free_energy import attention_weights, pairwise_energy, self_divergence_for_alpha
from vfe3.geometry.transport import (CompactFactoredTransport, DirectLinkTransport, FactoredTransport,
                                      RopeTransport, transport_covariance, transport_mean)
from vfe3.gradients.oracle import belief_gradients_autograd
from vfe3.gradients.pairwise_stats import diagonal_kl_pair_stats

_KERNELS: Dict[str, Callable] = {}
_COMPILED_KERNELS: Dict[str, Callable] = {}   # lazy torch.compile cache (compile_pair_kernel toggle)


def register_kernel(name: str, *, override: bool = False) -> Callable:
    """Decorator registering a query-side belief-gradient kernel under family ``name``.

    Duplicate keys fail closed (audit 2026-07-01 round-3): a second registration under an
    existing name silently shadowed the first. Pass ``override=True`` to replace deliberately.
    """
    def _wrap(fn: Callable) -> Callable:
        if name in _KERNELS and not override:
            raise KeyError(f"kernel {name!r} already registered; pass override=True to replace")
        _KERNELS[name] = fn
        _COMPILED_KERNELS.pop(name, None)
        return fn
    return _wrap


def has_kernel(name: str) -> bool:
    """Whether a hand kernel is registered for family ``name``."""
    return name in _KERNELS


def _raw_diag_kl(
    mu_q:    torch.Tensor,             # (N, K) query means
    sigma_q: torch.Tensor,             # (N, K) query variances
    mu_p:    torch.Tensor,             # (N, K) prior means
    sigma_p: torch.Tensor,             # (N, K) prior variances

    *,
    eps:     float = 1e-6,
) -> torch.Tensor:                     # (N,) UNCLAMPED KL(q_i || p_i)
    r"""Unclamped diagonal KL(q||p) = 0.5 Sum_k (s_k/t_k + (mu_p-mu_q)^2/t_k - 1 + log(t_k/s_k)).

    The divergence seam returns the clamped value safe_kl_clamp(D, [0, kl_max]);
    this returns the raw D so the kernel can reproduce the oracle's saturation
    mask (the oracle differentiates THROUGH the clamp, whose gradient is 0 once
    D leaves (0, kl_max)).
    """
    return diag_kl_unclamped(mu_q, sigma_q, mu_p, sigma_p, eps=eps)


def _raw_diag_kl_per_coord(
    mu_q:    torch.Tensor,             # (N, K) query means
    sigma_q: torch.Tensor,             # (N, K) query variances
    mu_p:    torch.Tensor,             # (N, K) prior means
    sigma_p: torch.Tensor,             # (N, K) prior variances

    *,
    eps:     float = 1e-6,
) -> torch.Tensor:                     # (N, K) UNCLAMPED per-coordinate KL D^(k)(q_i || p_i)
    r"""Unclamped per-coordinate diagonal KL D^(k) = 0.5 (s_k/t_k + (mu_p-mu_q)^2/t_k - 1 + log(t_k/s_k)).

    The per-coordinate analog of ``_raw_diag_kl`` (the -K of the summed form becomes -1 per
    coordinate). The per-coordinate self-term differentiates through a PER-COORDINATE
    safe_kl_clamp, so the kernel builds its saturation mask from this so a saturated
    coordinate is gated independently of the others -- matching the filtering oracle.
    """
    return diag_kl_unclamped_per_coord(mu_q, sigma_q, mu_p, sigma_p, eps=eps)


@register_kernel("gaussian_diagonal")
def _diag_kl_filtering_kernel(
    mu_q:       torch.Tensor,             # (N, K)
    sigma_q:    torch.Tensor,             # (N, K)
    mu_p:       torch.Tensor,             # (N, K)
    sigma_p:    torch.Tensor,             # (N, K)
    mu_t:       torch.Tensor,             # (N, N, K) transported key means
    sigma_t:    torch.Tensor,             # (N, N, K) transported key variances
    beta:       torch.Tensor,             # (N, N) or (H, N, N) raw attention weights
    alpha_coef: torch.Tensor,             # (N, 1) or (N, K) self-coupling coefficient

    *,
    kl_max:          float = 100.0,
    eps:             float = 1e-6,
    lambda_beta:     'float | torch.Tensor' = 1.0,   # weight on the belief-coupling (pair) term
    lambda_twohop:   float = 0.0,                # weight on the DETACHED two-hop (beta @ beta) pair term (0 = pure)
    need_sigma_grad: bool  = True,               # False -> skip the sigma pair contraction, return (grad_mu, None)
    irrep_dims:      Optional[List[int]]    = None, # block sizes; maps head h(k) onto coordinate k
    pair_mask:       Optional[torch.Tensor] = None, # destination-energy derivative mask; None means beta is pre-masked
    pair_inv_sigma_t: Optional[torch.Tensor] = None, # precomputed 1 / clamp(sigma_t); None keeps legacy arithmetic
    pair_delta_tq:    Optional[torch.Tensor] = None, # precomputed mu_t - mu_q; paired with pair_inv_sigma_t
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    r"""Diagonal-KL query-side (filtering) gradient (per-head aware).

      grad_mu_i    = m_i a_i (mu_i - mu_p_i)/sigma_p_i + lambda_beta Sum_j beta_ij^(h(k)) (mu_i - mu_t_ij)/sigma_t_ij
      grad_sigma_i = m_i a_i 0.5(1/sigma_p_i - 1/sigma_q_i)
                     + lambda_beta Sum_j beta_ij^(h(k)) 0.5(1/sigma_t_ij - 1/sigma_q_i)

    ``lambda_beta`` (1.0 = pure) scales ONLY the belief-coupling (pair) term, not the alpha
    self-term -- the analytic counterpart of scaling the post-softmax coupling+entropy block in
    ``free_energy`` (the envelope identity makes the pair term equal to d/dtheta of that block at
    beta*, so scaling both by the same factor keeps kernel == oracle).

    ``beta`` is the raw attention weight in its COMPACT per-head form. ``pair_mask`` gates only
    destination-energy derivatives: the one-hop derivative uses ``beta * pair_mask``, while the
    two-hop derivative uses ``(beta.detach() @ beta.detach()) * pair_mask``. Thus a zero or
    saturated intermediate edge may still carry hop mass, but a zero or saturated destination
    energy has the same zero derivative as the scalar clamp. For backward-compatible direct kernel
    calls, ``pair_mask=None`` treats ``beta`` as already masked. The coordinate-resolution weight
    beta_ij^(h(k)) is realized inside ``_pair_contract`` by
    contracting the head axis against a head-shaped VIEW of the pair difference, so the
    (..., N, N, K) ``beta_coord`` broadcast the old kernel consumed is never materialized
    (vram audit 2026-06-10: one full B x N^2 x K tensor saved for backward, for free).

    Self-term saturation mask m_i = 1[0 < D(q_i||p_i) < kl_max]: the oracle differentiates through
    safe_kl_clamp(D, [0, kl_max]), whose gradient is 0 once the raw self-divergence saturates the
    clamp, so the hand kernel zeros its self-term there to stay EXACTLY equal to the filtering
    oracle. The pairwise clamp mask is supplied by the caller from the clamped energy grid.

    The mask is shape-driven by ``alpha_coef``: a per-position coefficient (N,1) carries a summed
    self-divergence clamped as one scalar, so the mask is per-token (N,1); a per-coordinate
    coefficient (N,K) (the ``state_dependent_per_coord`` form) carries a coordinate-wise clamp, so
    the mask is per-coordinate (N,K) and a saturated coordinate is gated WITHOUT killing its
    unsaturated neighbours. The two coincide at K=1.
    """
    if (pair_inv_sigma_t is None) != (pair_delta_tq is None):
        raise ValueError(
            "pair_inv_sigma_t and pair_delta_tq must be provided together"
        )
    sp = sigma_p.clamp(min=eps); sq = sigma_q.clamp(min=eps)
    st = sigma_t.clamp(min=eps) if pair_inv_sigma_t is None else None

    if alpha_coef.shape[-1] == 1:                                               # per-position alpha
        raw_self  = _raw_diag_kl(mu_q, sigma_q, mu_p, sigma_p, eps=eps)         # (N,)
        self_mask = ((raw_self > 0.0) & (raw_self < kl_max)).to(mu_q.dtype).unsqueeze(-1)
    else:                                                                       # per-coordinate alpha
        raw_self  = _raw_diag_kl_per_coord(mu_q, sigma_q, mu_p, sigma_p, eps=eps)   # (N, K)
        self_mask = ((raw_self > 0.0) & (raw_self < kl_max)).to(mu_q.dtype)

    beta_pair = beta if pair_mask is None else beta * pair_mask

    # Two-hop hop weights W2_ik = Sum_j beta_ij beta_jk (per head), DETACHED -- the fixed
    # cross-workstream convention shared with the F-side term in free_energy: same pairwise KL
    # energy grid (mu_t/sigma_t already cover every (i, k) pair under the flat cocycle, where
    # Omega_ij Omega_jk = Omega_ik), no entropy term for the hop block. Compose the RAW attention
    # factors, then apply the clamp mask only to the destination energy derivative.
    w2 = None
    if lambda_twohop != 0.0:
        w2 = torch.matmul(beta.detach(), beta.detach())        # (..., N, N) or (..., H, N, N)
        if pair_mask is not None:
            w2 = w2 * pair_mask

    if pair_inv_sigma_t is None:
        diff_mu = (mu_q.unsqueeze(-2) - mu_t) / st             # (..., N, N, K)
    else:
        diff_mu = -pair_delta_tq * pair_inv_sigma_t             # (..., N, N, K)
    self_mu  = self_mask * alpha_coef * (mu_q - mu_p) / sp
    pair_mu  = _pair_contract(beta_pair, diff_mu, irrep_dims)
    grad_mu  = self_mu + lambda_beta * pair_mu
    if w2 is not None:
        grad_mu = grad_mu + lambda_twohop * _pair_contract(w2, diff_mu, irrep_dims)
    if not need_sigma_grad:
        # skip_belief_sigma_update: the caller freezes sigma, so the (..., N, N, K) sigma pair
        # contraction is dead compute -- skip it entirely rather than discarding its result.
        return grad_mu, None

    if pair_inv_sigma_t is None:
        diff_sig = 0.5 * (1.0 / st - 1.0 / sq.unsqueeze(-2))   # (..., N, N, K)
    else:
        diff_sig = 0.5 * (pair_inv_sigma_t - 1.0 / sq.unsqueeze(-2))
    self_sig = self_mask * alpha_coef * 0.5 * (1.0 / sp - 1.0 / sq)
    pair_sig = _pair_contract(beta_pair, diff_sig, irrep_dims)
    grad_sigma = self_sig + lambda_beta * pair_sig
    if w2 is not None:
        grad_sigma = grad_sigma + lambda_twohop * _pair_contract(w2, diff_sig, irrep_dims)
    return grad_mu, grad_sigma


def _pair_contract(
    beta:       torch.Tensor,             # (..., N, N) single-block OR (..., H, N, N) per-head
    diff:       torch.Tensor,             # (..., N, N, K) pair-difference tensor
    irrep_dims: Optional[List[int]],      # block sizes; None/[K] -> single block
) -> torch.Tensor:                        # (..., N, K) Sum_j beta_ij^(h(k)) diff_ijk
    r"""Pair-term contraction Sum_j beta_ij^(h(k)) diff_ijk WITHOUT the beta_coord broadcast.

    Single block: the legacy ``ij,ijk->ik`` einsum directly. Equal blocks: the per-head beta
    contracts against a (..., N, N, H, d) VIEW of ``diff`` (a reshape of the trailing K axis --
    no copy), the same per-(i, k) sums over j as the old coordinate-broadcast einsum, term for
    term. Unequal blocks (irrep towers): falls back to materializing beta_coord via
    ``_beta_to_coordinate`` (repeat_interleave has no view form).
    """
    if irrep_dims is None or len(irrep_dims) == 1:
        return torch.einsum("...ij,...ijk->...ik", beta, diff)
    if len(set(irrep_dims)) == 1:
        H, d = len(irrep_dims), irrep_dims[0]
        dv = diff.reshape(*diff.shape[:-1], H, d)                 # (..., N, N, H, d) view
        out = torch.einsum("...hij,...ijhd->...ihd", beta, dv)    # (..., N, H, d)
        return out.reshape(*out.shape[:-2], H * d)
    beta_coord = _beta_to_coordinate(beta, irrep_dims, diff.shape[-1])
    return torch.einsum("...ijk,...ijk->...ik", beta_coord, diff)


def _pair_mass(
    w:          torch.Tensor,             # (..., N, N) single-block OR (..., H, N, N) per-head weights
    irrep_dims: Optional[List[int]],      # block sizes; None/[K] -> single block

    K:          int,                      # total belief dimension
) -> torch.Tensor:                        # (..., N, K) Sum_j w_ij^(h(k))
    r"""Row mass Sum_j w_ij^(h(k)) broadcast to coordinate resolution (the mm fusion's pair mass).

    The coordinate map mirrors ``_beta_to_coordinate`` (head h's row sum repeated across its
    d_head coordinates) but sums over the key axis FIRST, so no (..., N, N, K) tensor is built.
    """
    row = w.sum(dim=-1)                                                  # (..., N) or (..., H, N)
    if irrep_dims is None or len(irrep_dims) == 1:
        return row.unsqueeze(-1).expand(*row.shape, K)
    x = row.movedim(-2, -1)                                              # (..., N, H)
    if len(set(irrep_dims)) == 1:                                        # equal blocks: view expand
        d = irrep_dims[0]
        return x.unsqueeze(-1).expand(*x.shape, d).reshape(*x.shape[:-1], x.shape[-1] * d)
    reps = torch.tensor(irrep_dims, device=w.device)                     # unequal blocks: gather
    return torch.repeat_interleave(x, reps, dim=-1)


def _get_compiled_kernel(name: str) -> Callable:
    r"""``torch.compile(dynamic=False)``-wrapped kernel for family ``name``, cached ONCE per
    process (lazy). A wrap-time failure falls back to the eager kernel with a single warning
    (call-time backend failures are handled at the ``belief_gradients`` call site, since
    torch.compile typically defers backend errors -- e.g. Windows without triton -- to the
    first invocation)."""
    cached = _COMPILED_KERNELS.get(name)
    if cached is not None:
        return cached
    fn = get_kernel(name)
    try:
        compiled = torch.compile(fn, dynamic=False)
    except Exception as exc:
        import warnings
        warnings.warn(
            f"torch.compile unavailable for the pair kernel ({exc!r}); using the eager kernel.",
            UserWarning, stacklevel=2,
        )
        compiled = fn
    _COMPILED_KERNELS[name] = compiled
    return compiled


def uses_kernel_route(
    *,
    renyi_order:               float,
    gradient_mode:             str,
    family:                    str,
    divergence_family:         str,
    include_attention_entropy: bool,
    transport_mode:            str  = "flat",
    decoupled_value_gauge:     bool = False,
) -> bool:
    r"""Whether ``belief_gradients`` serves the closed-form KERNEL (else the autograd oracle).

    The single source of truth for the kernel-coverage predicate, exposed so callers (the
    E-step's unroll-truncation warning, the config-time freeze warning) cannot drift from the
    dispatch below.

    ``transport_mode='regime_ii'`` excludes the kernel (audit 2026-06-10 F1): the hand kernel is
    the FLAT-transport gradient -- it treats the transported keys (Omega mu_j, Omega Sigma_j
    Omega^T) as constants in mu, but the regime_ii Omega depends on mu through
    delta_ij = mu_i^T W^a mu_j, so the kernel would silently descend a frozen-Omega objective.
    regime_ii routes to the autograd oracle, which rebuilds Omega from its differentiation
    leaves (``omega_builder``) and therefore carries the d Omega/d mu term.

    ``decoupled_value_gauge`` (RopeTransport.on_value=False) likewise excludes the kernel: with the
    attention gauge and value gauge factored apart (GL(K)_attention.tex:1909), beta is the softmax of
    the rotated SCORE energy but the coupling sum uses the un-rotated value energy, so beta is no
    longer that sum's stationary point and the closed-form envelope kernel (which assumes it is) does
    not apply. The oracle differentiates the decoupled F directly and carries the extra d beta/d mu
    term."""
    return (
        gradient_mode == "filtering"
        and family == "gaussian_diagonal"
        and divergence_family == "renyi"
        and abs(renyi_order - 1.0) < 1e-9
        and include_attention_entropy
        and transport_mode not in ("regime_ii", "regime_ii_covariant")
        and not decoupled_value_gauge
        and has_kernel(family)
    )


def _pairwise_stats_reuse_is_sound(
    omega: 'torch.Tensor | CompactFactoredTransport | DirectLinkTransport | FactoredTransport | RopeTransport',
) -> bool:
    r"""Whether the effective transport representation may reuse diagonal-KL pair statistics."""
    if isinstance(omega, RopeTransport):
        return _pairwise_stats_reuse_is_sound(omega.base)
    return not isinstance(omega, torch.Tensor)


def belief_gradients(
    mu:           torch.Tensor,           # (N, K)
    sigma:        torch.Tensor,           # (N, K)
    mu_p:         torch.Tensor,           # (N, K)
    sigma_p:      torch.Tensor,           # (N, K)
    omega:        'torch.Tensor | CompactFactoredTransport | DirectLinkTransport | FactoredTransport | RopeTransport | None',

    *,
    tau:          'float | torch.Tensor' = 1.0,
    renyi_order:  float = 1.0,
    kl_max:       float = 100.0,
    eps:          float = 1e-6,
    b0:           float = 1.0,
    c0:           float = 1.0,
    lambda_beta:  'float | torch.Tensor' = 1.0,   # weight on the belief-coupling block (1.0 = pure F)
    lambda_twohop: float = 0.0,                   # weight on the detached two-hop pair term

    include_attention_entropy: bool = True,
    create_graph:              bool = False,   # unroll: oracle returns a differentiable grad (to prior)
    need_sigma_grad:           bool = True,    # False -> kernel skips the sigma pair contraction (grad_sigma None)
    compile_pair_kernel:       bool = False,   # torch.compile the closed-form kernel (lazy cache; eager fallback)
    reuse_pairwise_kl_stats:   bool = False,   # reuse graph-live diagonal-KL pair statistics on the canonical route
    gradient_mode:             str  = "filtering",
    family:                    str  = "gaussian_diagonal",
    divergence_family:         str  = "renyi",
    lambda_alpha_mode:         str  = "constant",
    transport_mode:            str  = "flat",  # 'regime_ii' excludes the kernel (mu-dependent Omega)
    value:                     float = 1.0,

    irrep_dims:                Optional[List[int]]    = None,
    log_prior:                 Optional[torch.Tensor] = None,
    omega_builder:             Optional[Callable]     = None,   # (mu_q, sigma_q, mu_k, sigma_k) -> transport (regime_ii oracle rebuild)
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    r"""Belief gradient: hand kernel for filtering+gaussian_diagonal+KL+canonical+flat, else oracle.

    The closed-form kernel keeps the gradient live (analytic on the live belief), so the unrolled
    E-step signal reaches the prior; ``create_graph=True`` makes the oracle fallback do the same for
    the non-kernel families it serves (else its detached tangent would truncate that signal).

    ``irrep_dims`` (when more than one block) makes attention PER HEAD: the energy/beta carry a
    head axis and the per-coordinate beta the kernel consumes is head h's weight on coordinate k.

    ``transport_mode='regime_ii'`` always routes to the ORACLE (audit 2026-06-10 F1: the kernel is
    the flat-transport gradient and would drop d Omega/d mu); the caller supplies ``omega_builder``
    so the oracle rebuilds the mu-dependent transport from its differentiation leaves.
    """
    # Value-gauge decoupling (RopeTransport.on_value=False) breaks beta's stationarity for the
    # coupling sum, so the closed-form kernel does not apply -- route to the oracle (which builds the
    # value-gauge coupling energy). The omega object is the source of truth here.
    decoupled_value = isinstance(omega, RopeTransport) and not omega.on_value
    use_kernel = uses_kernel_route(
        renyi_order=renyi_order, gradient_mode=gradient_mode, family=family,
        divergence_family=divergence_family,
        include_attention_entropy=include_attention_entropy,
        transport_mode=transport_mode,
        decoupled_value_gauge=decoupled_value,
    )
    if not use_kernel:
        return belief_gradients_autograd(
            mu, sigma, mu_p, sigma_p, omega, tau=tau, renyi_order=renyi_order,
            kl_max=kl_max, eps=eps, b0=b0, c0=c0, value=value, lambda_beta=lambda_beta,
            lambda_twohop=lambda_twohop,
            include_attention_entropy=include_attention_entropy, create_graph=create_graph,
            need_sigma_grad=need_sigma_grad,
            gradient_mode=gradient_mode, family=family, divergence_family=divergence_family,
            lambda_alpha_mode=lambda_alpha_mode, irrep_dims=irrep_dims, log_prior=log_prior,
            omega_builder=omega_builder,
        )

    mu_k, sigma_k = mu.detach(), sigma.detach()
    mu_t = transport_mean(omega, mu_k)                 # rank-agnostic: (N,N,K) or (B,N,N,K)
    # diagonal_out from the BELIEF shape (diagonal iff sigma has the same rank as mu), not the omega
    # rank: the batch-collapsed (N,N,K,K) regime_ii_link omega would otherwise misinfer the full
    # sandwich against a batched diagonal sigma. Family-agnostic (covers every diagonal family); a
    # batched dense omega is unaffected (it would infer diagonal anyway). The kernel route is
    # gaussian_diagonal-only, so this is always True here, but the shape check stays robust.
    sigma_t = transport_covariance(omega, sigma_k, diagonal_out=(sigma_k.dim() == mu_k.dim()))
    fam = get_family(family)
    sd = self_divergence_for_alpha(fam(mu, sigma), fam(mu_p, sigma_p), alpha=1.0, kl_max=kl_max, eps=eps,
                                   divergence_family=divergence_family, lambda_alpha_mode=lambda_alpha_mode)
    pair_stats = None
    # A raw dense effective base carries no structural same-frame certificate, including through a
    # RoPE wrapper. On the single-block flat route, its numerically composed self links can make the
    # float64 statistics reduction strictly positive where the generic float32 energy and derivative
    # mask are exactly zero. Fail closed there; factored effective bases retain automatic reuse.
    if (reuse_pairwise_kl_stats
            and _pairwise_stats_reuse_is_sound(omega)
            and all(tensor.dtype == torch.float32 for tensor in (mu, sigma, mu_t, sigma_t))):
        pair_stats = diagonal_kl_pair_stats(
            mu,
            sigma,
            mu_t,
            sigma_t,
            kl_max=kl_max,
            eps=eps,
            irrep_dims=irrep_dims,
        )
    if pair_stats is None:
        energy = pairwise_energy(fam(mu, sigma), fam(mu_t, sigma_t), alpha=1.0, kl_max=kl_max, eps=eps,
                                 divergence_family=divergence_family, irrep_dims=irrep_dims)
    else:
        energy = pair_stats.energy
    beta = attention_weights(energy, tau=tau, log_prior=log_prior)   # (N,N) or (H,N,N)
    # Pair-term saturation mask (audit 2026-06-09 P7): the oracle differentiates
    # beta_ij * clamp(E_ij, [0, kl_max]), whose pair gradient VANISHES wherever the raw energy
    # saturates the clamp (the clamp emits the exact bounds, so the equality tests are robust).
    # Without the mask a fully saturated row softmaxes to uniform beta over constant energies and
    # the kernel's transported pair term deviates from autograd-of-F by orders of magnitude.
    # Mirrors the self-term mask inside the kernel body; beta itself (the weights) is unchanged.
    if pair_stats is None:
        pair_mask = ((energy > 0.0) & (energy < kl_max)).to(beta.dtype)
    else:
        pair_mask = pair_stats.pair_mask.to(beta.dtype)
    coef = alpha_gradient_coefficient(sd, value=value, b0=b0, c0=c0, mode=lambda_alpha_mode)
    if not alpha_is_per_coord(lambda_alpha_mode):
        coef = coef.unsqueeze(-1)                 # (N,) -> (N,1) per-position broadcast; per-coord sd is already (N,K)
    # The raw beta stays in its COMPACT per-head form; the kernel's _pair_contract realizes
    # beta_ij^(h(k)) against a head-shaped view of the pair difference, so the (..., N, N, K)
    # beta_coord broadcast is never materialized (vram audit 2026-06-10).
    kernel_args   = (mu, sigma, mu_p, sigma_p, mu_t, sigma_t, beta, coef)
    kernel_kwargs = dict(kl_max=kl_max, eps=eps, lambda_beta=lambda_beta,
                         lambda_twohop=lambda_twohop, need_sigma_grad=need_sigma_grad,
                         irrep_dims=irrep_dims, pair_mask=pair_mask)
    if pair_stats is not None:
        kernel_kwargs["pair_inv_sigma_t"] = pair_stats.inv_sigma_t
        kernel_kwargs["pair_delta_tq"] = pair_stats.delta_tq
    if compile_pair_kernel:
        try:
            return _get_compiled_kernel(family)(*kernel_args, **kernel_kwargs)
        except Exception as exc:
            # torch.compile defers backend errors (e.g. Windows without triton) to the first
            # invocation; fall back to the eager kernel PERMANENTLY (cache the eager fn) with a
            # single warning. The kernel is pure, so a rerun in eager is value-identical.
            import warnings
            warnings.warn(
                f"compiled pair kernel failed ({exc!r}); falling back to the eager kernel.",
                UserWarning, stacklevel=2,
            )
            _COMPILED_KERNELS[family] = get_kernel(family)
    return get_kernel(family)(*kernel_args, **kernel_kwargs)


def mm_exact_update(
    mu:           torch.Tensor,           # (N, K) or (B, N, K) belief means (query side, live)
    sigma:        torch.Tensor,           # (N, K) or (B, N, K) belief variances (live)
    mu_p:         torch.Tensor,           # (N, K) or (B, N, K) prior means
    sigma_p:      torch.Tensor,           # (N, K) or (B, N, K) prior variances
    omega:        'torch.Tensor | CompactFactoredTransport | DirectLinkTransport | FactoredTransport | RopeTransport',

    *,
    tau:          'float | torch.Tensor' = 1.0,
    b0:           'float | torch.Tensor' = 1.0,
    c0:           'float | torch.Tensor' = 1.0,
    lambda_beta:  'float | torch.Tensor' = 1.0,   # weight on the belief-coupling (pair) block

    kl_max:        float = 100.0,
    eps:           float = 1e-6,
    lambda_twohop: float = 0.0,                   # weight on the detached two-hop pair block
    value:         float = 1.0,

    lambda_alpha_mode: str = "constant",
    family:            str = "gaussian_diagonal",
    divergence_family: str = "renyi",

    need_sigma_update:       bool = True,       # False -> omit sigma fusion and return the input sigma exactly
    reuse_pairwise_kl_stats: bool = False,      # reuse graph-live diagonal-KL pair statistics on the canonical route

    irrep_dims:    Optional[List[int]]    = None,
    log_prior:     Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:          # (mu_star, sigma_star), each (..., N, K)
    r"""Closed-form coordinate minimizer of the beta-frozen, strict-pair-masked diagonal-KL
    surrogate (NOT a majorizer of the canonical frozen-attention objective -- the strict pair
    mask below excludes the structural E_ii = 0 self-pairs; see README "mm_exact").

    At frozen attention weights beta (and frozen transported key moments -- the filtering split),
    the per-coordinate stationary point of

        F_hat = Sum_i m_i a_i KL(q_i || p_i)
              + Sum_ij w_ij^(h(k)) KL(q_i || Omega_ij q_j),
        w_ij  = m_ij [lambda_beta beta_ij + lambda_twohop (beta beta)_ij],
        m_ij  = 1[0 < E_ij < kl_max],

    obtained by zeroing the hand kernel's grad_mu / grad_sigma expressions, is the precision fusion

        mu*_ik    = ( m_i a_i mu_p,ik / sigma_p,ik + Sum_j w_ij^(h(k)) mu_t,ijk / sigma_t,ijk ) / P_ik,
        sigma*_ik = ( m_i a_i + Sum_j w_ij^(h(k)) ) / P_ik,
        P_ik      =   m_i a_i / sigma_p,ik + Sum_j w_ij^(h(k)) / sigma_t,ijk,

    with m_i the self-term saturation mask (UPPER gate only, 1[D < kl_max]; 2026-07-10 fix --
    see the comment at the mask) and a_i the same alpha coefficient the kernel
    uses at the CURRENT point (state-dependent envelope frozen). mu* is sigma_q-independent (the
    kernel's grad_mu carries no sigma_q), so the pair is jointly exact in one evaluation. beta and
    the transported moments are the SAME intermediates ``belief_gradients`` builds. The one-hop
    derivative uses ``beta * pair_mask``; the two-hop derivative composes raw detached beta first,
    then applies ``pair_mask`` to the destination energy, matching the scalar functional. Thus the
    fusion is the exact minimizer of the objective the kernel's gradient descends.
    The returned sigma* carries only the ``eps`` floor; the caller applies its damping and the
    sigma_max cap. The graph stays LIVE through (mu*, sigma*) (analytic in mu/sigma via beta, sd),
    matching the kernel's unroll behavior; the caller detaches for straight_through.
    """
    mu_k, sigma_k = mu.detach(), sigma.detach()
    mu_t    = transport_mean(omega, mu_k)                        # (..., N, N, K) transported keys
    sigma_t = transport_covariance(omega, sigma_k, diagonal_out=(sigma_k.dim() == mu_k.dim()))
    fam = get_family(family)
    sd = self_divergence_for_alpha(fam(mu, sigma), fam(mu_p, sigma_p), alpha=1.0, kl_max=kl_max, eps=eps,
                                   divergence_family=divergence_family, lambda_alpha_mode=lambda_alpha_mode)
    pair_stats = None
    decoupled_value = isinstance(omega, RopeTransport) and not omega.on_value
    if (reuse_pairwise_kl_stats
            and _pairwise_stats_reuse_is_sound(omega)
            and family == "gaussian_diagonal"
            and divergence_family == "renyi"
            and not decoupled_value
            and all(tensor.dtype == torch.float32 for tensor in (mu, sigma, mu_t, sigma_t))):
        pair_stats = diagonal_kl_pair_stats(
            mu,
            sigma,
            mu_t,
            sigma_t,
            kl_max=kl_max,
            eps=eps,
            irrep_dims=irrep_dims,
        )
    if pair_stats is None:
        energy = pairwise_energy(fam(mu, sigma), fam(mu_t, sigma_t), alpha=1.0, kl_max=kl_max, eps=eps,
                                 divergence_family=divergence_family, irrep_dims=irrep_dims)
    else:
        energy = pair_stats.energy
    beta = attention_weights(energy, tau=tau, log_prior=log_prior)   # (..., N, N) or (..., H, N, N)
    if pair_stats is None:
        pair_mask = ((energy > 0.0) & (energy < kl_max)).to(beta.dtype)
    else:
        pair_mask = pair_stats.pair_mask.to(beta.dtype)
    coef = alpha_gradient_coefficient(sd, value=value, b0=b0, c0=c0, mode=lambda_alpha_mode)
    if not alpha_is_per_coord(lambda_alpha_mode):
        coef = coef.unsqueeze(-1)                                    # (N,) -> (N,1)

    # Self-term mask: UPPER gate ONLY (2026-07-10 fix). The gradient kernel keeps the lower
    # gate (raw_self > 0) because it mirrors the clamp's zero derivative and dD = 0 at q == p
    # anyway; here the mask is reused as the prior PRECISION WEIGHT in the fusion, and the
    # model enters the E-step with q0 == p EXACTLY (forward_beliefs anchors the belief to the
    # prior), so raw_self == 0 for every token: a lower gate zeroes the prior anchor and snaps
    # mu* to the self-excluded neighbor consensus on the first (often only) inner iteration.
    # The correct MM weight at D = 0 is the envelope alpha* = c0/(b0+0). The upper gate stays:
    # the kl_max clamp flattens the objective in a neighborhood there.
    # docs/2026-07-10-mm-exact-prior-anchor-fix.md
    if coef.shape[-1] == 1:                                          # per-position alpha
        raw_self  = _raw_diag_kl(mu, sigma, mu_p, sigma_p, eps=eps)
        self_mask = (raw_self < kl_max).to(mu.dtype).unsqueeze(-1)   # UPPER gate only (see comment above)
    else:                                                            # per-coordinate alpha
        raw_self  = _raw_diag_kl_per_coord(mu, sigma, mu_p, sigma_p, eps=eps)
        self_mask = (raw_self < kl_max).to(mu.dtype)                 # UPPER gate only (see comment above)
    a = self_mask * coef                                             # m_i a_i, (..., N, 1) or (..., N, K)

    w = lambda_beta * (beta * pair_mask)
    if lambda_twohop != 0.0:
        w2 = torch.matmul(beta.detach(), beta.detach())              # raw detached hop factors
        w = w + lambda_twohop * (w2 * pair_mask)                     # mask destination derivative only

    sp = sigma_p.clamp(min=eps)
    K = mu.shape[-1]
    prior_prec = a / sp                                              # (..., N, K) m a / sigma_p
    if pair_stats is None:
        st = sigma_t.clamp(min=eps)
        pair_prec = _pair_contract(w, 1.0 / st, irrep_dims)          # (..., N, K) Sum_j w / sigma_t
        pair_mean = _pair_contract(w, mu_t / st, irrep_dims)         # (..., N, K) Sum_j w mu_t / sigma_t
    else:
        pair_prec = _pair_contract(w, pair_stats.inv_sigma_t, irrep_dims)
        pair_mean = _pair_contract(w, mu_t * pair_stats.inv_sigma_t, irrep_dims)
    prec = prior_prec + pair_prec                                   # pre-clamp fused precision
    P = prec.clamp(min=eps)                                         # eps guards the all-saturated row
    mu_star = (a * mu_p / sp + pair_mean) / P
    # m12: on a fully saturated row (a==0 and all w==0) prec floors to 0; dividing the zero numerator by
    # eps snapped the belief to (0, eps), the OPPOSITE of the gradient route (which stays put). Keep the
    # live belief where prec floors, matching the gradient route and preserving graph liveness.
    degenerate = prec <= eps
    mu_star = torch.where(degenerate, mu, mu_star)
    if not need_sigma_update:
        return mu_star, sigma

    pair_mass  = _pair_mass(w, irrep_dims, K)                        # (..., N, K) Sum_j w
    sigma_star = ((a + pair_mass) / P).clamp(min=eps)
    sigma_star = torch.where(degenerate, sigma, sigma_star)
    return mu_star, sigma_star


def _beta_to_coordinate(
    beta:       torch.Tensor,             # (N, N) single-block OR (H, N, N) per-head
    irrep_dims: Optional[List[int]],      # block sizes; None/[K] -> single block
    K:          int,                      # total belief dimension
) -> torch.Tensor:                        # (N, N, K) per-coordinate attention weight
    r"""Broadcast attention weights to coordinate k via k's irrep block h(k).

    Single block (irrep_dims None or length 1): the one beta_ij repeated across every
    coordinate. Per-head ((H,N,N) with H>1): head h's weight repeated across its d_head
    coordinates, so coordinate k carries beta_ij^(h(k)).
    """
    if irrep_dims is None or len(irrep_dims) == 1:
        return beta.unsqueeze(-1).expand(*beta.shape, K)
    x = beta.movedim(-3, -1)                                             # (N, N, H)
    if len(set(irrep_dims)) == 1:                                        # equal blocks (block_glk):
        d = irrep_dims[0]                                                # expand/reshape, no gather --
        return x.unsqueeze(-1).expand(*x.shape, d).reshape(*x.shape[:-1], x.shape[-1] * d)  # bit-identical
    reps = torch.tensor(irrep_dims, device=beta.device)                 # unequal blocks: gather fallback
    return torch.repeat_interleave(x, reps, dim=-1)                      # (N,N,H)->(N,N,K)


def get_kernel(name: str) -> Callable:
    """Return the registered kernel for family ``name`` (KeyError if absent)."""
    if name not in _KERNELS:
        raise KeyError(f"no belief-gradient kernel for family {name!r}; available: {sorted(_KERNELS)}")
    return _KERNELS[name]
