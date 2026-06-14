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
from vfe3.free_energy import attention_weights, pairwise_energy, self_divergence_for_alpha
from vfe3.geometry.transport import transport_covariance, transport_mean
from vfe3.gradients.oracle import belief_gradients_autograd

_KERNELS: Dict[str, Callable] = {}


def register_kernel(name: str) -> Callable:
    """Decorator registering a query-side belief-gradient kernel under family ``name``."""
    def _wrap(fn: Callable) -> Callable:
        _KERNELS[name] = fn
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
    sq = sigma_q.clamp(min=eps); sp = sigma_p.clamp(min=eps)
    trace  = (sq / sp).sum(dim=-1)
    mahal  = (((mu_p - mu_q) ** 2) / sp).sum(dim=-1)
    logdet = (torch.log(sp) - torch.log(sq)).sum(dim=-1)
    return 0.5 * (trace + mahal - mu_q.shape[-1] + logdet)


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
    sq = sigma_q.clamp(min=eps); sp = sigma_p.clamp(min=eps)
    return 0.5 * (sq / sp + ((mu_p - mu_q) ** 2) / sp - 1.0 + torch.log(sp) - torch.log(sq))


@register_kernel("gaussian_diagonal")
def _diag_kl_filtering_kernel(
    mu_q:       torch.Tensor,             # (N, K)
    sigma_q:    torch.Tensor,             # (N, K)
    mu_p:       torch.Tensor,             # (N, K)
    sigma_p:    torch.Tensor,             # (N, K)
    mu_t:       torch.Tensor,             # (N, N, K) transported key means
    sigma_t:    torch.Tensor,             # (N, N, K) transported key variances
    beta:       torch.Tensor,             # (N, N) or (H, N, N) MASKED attention weights
    alpha_coef: torch.Tensor,             # (N, 1) or (N, K) self-coupling coefficient

    *,
    kl_max:      float = 100.0,
    eps:         float = 1e-6,
    lambda_beta: 'float | torch.Tensor' = 1.0,   # weight on the belief-coupling (pair) term
    irrep_dims:  Optional[List[int]] = None,     # block sizes; maps head h(k) onto coordinate k
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Diagonal-KL query-side (filtering) gradient (per-head aware).

      grad_mu_i    = m_i a_i (mu_i - mu_p_i)/sigma_p_i + lambda_beta Sum_j beta_ij^(h(k)) (mu_i - mu_t_ij)/sigma_t_ij
      grad_sigma_i = m_i a_i 0.5(1/sigma_p_i - 1/sigma_q_i)
                     + lambda_beta Sum_j beta_ij^(h(k)) 0.5(1/sigma_t_ij - 1/sigma_q_i)

    ``lambda_beta`` (1.0 = pure) scales ONLY the belief-coupling (pair) term, not the alpha
    self-term -- the analytic counterpart of scaling the post-softmax coupling+entropy block in
    ``free_energy`` (the envelope identity makes the pair term equal to d/dtheta of that block at
    beta*, so scaling both by the same factor keeps kernel == oracle).

    ``beta`` is the (already pair-masked) attention weight in its COMPACT per-head form; the
    coordinate-resolution weight beta_ij^(h(k)) is realized inside ``_pair_contract`` by
    contracting the head axis against a head-shaped VIEW of the pair difference, so the
    (..., N, N, K) ``beta_coord`` broadcast the old kernel consumed is never materialized
    (vram audit 2026-06-10: one full B x N^2 x K tensor saved for backward, for free).

    Self-term saturation mask m_i = 1[0 < D(q_i||p_i) < kl_max]: the oracle differentiates through
    safe_kl_clamp(D, [0, kl_max]), whose gradient is 0 once the raw self-divergence saturates the
    clamp, so the hand kernel zeros its self-term there to stay EXACTLY equal to the filtering
    oracle. The pairwise term needs no mask: a saturated E_ij drives beta_ij -> 0 on both sides.

    The mask is shape-driven by ``alpha_coef``: a per-position coefficient (N,1) carries a summed
    self-divergence clamped as one scalar, so the mask is per-token (N,1); a per-coordinate
    coefficient (N,K) (the ``state_dependent_per_coord`` form) carries a coordinate-wise clamp, so
    the mask is per-coordinate (N,K) and a saturated coordinate is gated WITHOUT killing its
    unsaturated neighbours. The two coincide at K=1.
    """
    sp = sigma_p.clamp(min=eps); sq = sigma_q.clamp(min=eps); st = sigma_t.clamp(min=eps)

    if alpha_coef.shape[-1] == 1:                                               # per-position alpha
        raw_self  = _raw_diag_kl(mu_q, sigma_q, mu_p, sigma_p, eps=eps)         # (N,)
        self_mask = ((raw_self > 0.0) & (raw_self < kl_max)).to(mu_q.dtype).unsqueeze(-1)
    else:                                                                       # per-coordinate alpha
        raw_self  = _raw_diag_kl_per_coord(mu_q, sigma_q, mu_p, sigma_p, eps=eps)   # (N, K)
        self_mask = ((raw_self > 0.0) & (raw_self < kl_max)).to(mu_q.dtype)

    self_mu  = self_mask * alpha_coef * (mu_q - mu_p) / sp
    pair_mu  = _pair_contract(beta, (mu_q.unsqueeze(-2) - mu_t) / st, irrep_dims)
    grad_mu  = self_mu + lambda_beta * pair_mu

    self_sig = self_mask * alpha_coef * 0.5 * (1.0 / sp - 1.0 / sq)
    pair_sig = _pair_contract(beta, 0.5 * (1.0 / st - 1.0 / sq.unsqueeze(-2)), irrep_dims)
    grad_sigma = self_sig + lambda_beta * pair_sig
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


def uses_kernel_route(
    *,
    renyi_order:               float,
    gradient_mode:             str,
    family:                    str,
    divergence_family:         str,
    include_attention_entropy: bool,
    transport_mode:            str  = "flat",
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
    leaves (``omega_builder``) and therefore carries the d Omega/d mu term."""
    return (
        gradient_mode == "filtering"
        and family == "gaussian_diagonal"
        and divergence_family == "renyi"
        and abs(renyi_order - 1.0) < 1e-9
        and include_attention_entropy
        and transport_mode != "regime_ii"
        and has_kernel(family)
    )


def belief_gradients(
    mu:           torch.Tensor,           # (N, K)
    sigma:        torch.Tensor,           # (N, K)
    mu_p:         torch.Tensor,           # (N, K)
    sigma_p:      torch.Tensor,           # (N, K)
    omega:        Optional[torch.Tensor], # (N, N, K, K); None ONLY with omega_builder (regime_ii)

    *,
    tau:          'float | torch.Tensor' = 1.0,
    renyi_order:  float = 1.0,
    kl_max:       float = 100.0,
    eps:          float = 1e-6,
    b0:           float = 1.0,
    c0:           float = 1.0,
    lambda_beta:  'float | torch.Tensor' = 1.0,   # weight on the belief-coupling block (1.0 = pure F)

    include_attention_entropy: bool = True,
    create_graph:              bool = False,   # unroll: oracle returns a differentiable grad (to prior)
    gradient_mode:             str  = "filtering",
    family:                    str  = "gaussian_diagonal",
    divergence_family:         str  = "renyi",
    lambda_alpha_mode:         str  = "constant",
    transport_mode:            str  = "flat",  # 'regime_ii' excludes the kernel (mu-dependent Omega)
    value:                     float = 1.0,

    irrep_dims:                Optional[List[int]]    = None,
    log_prior:                 Optional[torch.Tensor] = None,
    log_alpha:                 Optional[torch.Tensor] = None,   # learned scalar self-coupling (None -> pure path)
    omega_builder:             Optional[Callable]     = None,   # (mu_q, mu_k) -> transport (regime_ii oracle rebuild)
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    use_kernel = uses_kernel_route(
        renyi_order=renyi_order, gradient_mode=gradient_mode, family=family,
        divergence_family=divergence_family,
        include_attention_entropy=include_attention_entropy,
        transport_mode=transport_mode,
    )
    if not use_kernel:
        return belief_gradients_autograd(
            mu, sigma, mu_p, sigma_p, omega, tau=tau, renyi_order=renyi_order,
            kl_max=kl_max, eps=eps, b0=b0, c0=c0, value=value, lambda_beta=lambda_beta,
            include_attention_entropy=include_attention_entropy, create_graph=create_graph,
            gradient_mode=gradient_mode, family=family, divergence_family=divergence_family,
            lambda_alpha_mode=lambda_alpha_mode, irrep_dims=irrep_dims, log_prior=log_prior, log_alpha=log_alpha,
            omega_builder=omega_builder,
        )

    mu_k, sigma_k = mu.detach(), sigma.detach()
    mu_t = transport_mean(omega, mu_k)                 # rank-agnostic: (N,N,K) or (B,N,N,K)
    sigma_t = transport_covariance(omega, sigma_k)
    fam = get_family(family)
    sd = self_divergence_for_alpha(fam(mu, sigma), fam(mu_p, sigma_p), alpha=1.0, kl_max=kl_max, eps=eps,
                                   divergence_family=divergence_family, lambda_alpha_mode=lambda_alpha_mode)
    energy = pairwise_energy(fam(mu, sigma), fam(mu_t, sigma_t), alpha=1.0, kl_max=kl_max, eps=eps,
                             divergence_family=divergence_family, irrep_dims=irrep_dims)
    beta = attention_weights(energy, tau=tau, log_prior=log_prior)   # (N,N) or (H,N,N)
    # Pair-term saturation mask (audit 2026-06-09 P7): the oracle differentiates
    # beta_ij * clamp(E_ij, [0, kl_max]), whose pair gradient VANISHES wherever the raw energy
    # saturates the clamp (the clamp emits the exact bounds, so the equality tests are robust).
    # Without the mask a fully saturated row softmaxes to uniform beta over constant energies and
    # the kernel's transported pair term deviates from autograd-of-F by orders of magnitude.
    # Mirrors the self-term mask inside the kernel body; beta itself (the weights) is unchanged.
    pair_mask = ((energy > 0.0) & (energy < kl_max)).to(beta.dtype)
    coef = alpha_gradient_coefficient(sd, value=value, b0=b0, c0=c0, mode=lambda_alpha_mode, log_alpha=log_alpha)
    if not alpha_is_per_coord(lambda_alpha_mode):
        coef = coef.unsqueeze(-1)                 # (N,) -> (N,1) per-position broadcast; per-coord sd is already (N,K)
    # The masked beta stays in its COMPACT per-head form; the kernel's _pair_contract realizes
    # beta_ij^(h(k)) against a head-shaped view of the pair difference, so the (..., N, N, K)
    # beta_coord broadcast is never materialized (vram audit 2026-06-10).
    return get_kernel(family)(mu, sigma, mu_p, sigma_p, mu_t, sigma_t, beta * pair_mask, coef,
                              kl_max=kl_max, eps=eps, lambda_beta=lambda_beta, irrep_dims=irrep_dims)


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
