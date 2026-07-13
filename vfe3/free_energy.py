r"""The single authoritative scalar free energy F = sum_i F_i for VFE_3.0.

F is divergence-agnostic: it assembles the scalar from per-pair energies E_ij and
self-divergences D(q_i||p_i) supplied by the `divergence` registry, so a new
divergence slots in by registration + config, never by editing F. Canonical (with
the attention-entropy term) vs surrogate is a single toggle. The attention prior
is a log-bias B_ij from the `attention_prior` seam; beta* = softmax_j(B - E/tau).
"""

import math
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch

from vfe3.alpha_i import alpha_is_per_coord
from vfe3.divergence import (
    get_functional,
    get_functional_per_coord,
    has_per_coord_functional,
    divergence_functionals_per_coord,
)
from vfe3.families.base import BeliefParams


def _broadcast_tau(tau: 'float | torch.Tensor', energy: torch.Tensor) -> 'float | torch.Tensor':
    r"""Reshape a per-head (H,) tau so it broadcasts against an (..., H, N, N) energy.

    Scalar/0-d tau passes through unchanged (the single-block / scalar-kappa path).
    A 1-d (H,) tau is reshaped to (H, 1, 1): the head axis is always 3rd from the last
    in the energy tensor, so two trailing 1s align correctly for both the unbatched
    (H, N, N) and batched (B, H, N, N) layouts. The reshape also moves tau onto the
    energy's device (no-op when already there): attention_tau builds a CPU (H,) tau
    when a SCALAR kappa meets unequal irrep dims, and this is the one funnel every
    tau-consuming division passes through. A >= 2-d PER-QUERY tau (..., [H,] N, 1)
    (``query_adaptive_tau``) also passes through unchanged: its trailing singleton key
    axis already right-aligns against the (..., [H,] N, N) energy.
    """
    if isinstance(tau, torch.Tensor) and tau.dim() == 1:
        return tau.to(device=energy.device).reshape(tau.shape[0], 1, 1)
    return tau


def attention_tau(
    kappa:      'float | torch.Tensor',   # sharpness scalar or (H,) per-head (kappa=1 -> Vaswani recovery)
    irrep_dims: List[int],                # gauge-irrep block sizes; sum == K
) -> 'float | torch.Tensor':
    r"""Softmax temperature tau = kappa * sqrt(d_energy), where d_energy is the dimension the
    per-pair energy E_ij = D(q_i || Omega_ij q_j) accumulates over -- the gauge-irrep BLOCK size.

    Single-block groups (glk / so_k / sp report ``irrep_dims=[K]``) accumulate the divergence over
    the full K, so d_energy = K. Per-head multi-block groups (block_glk: ``irrep_dims=[d_head]*H``)
    accumulate per head, so d_energy = d_head. In both EQUAL-block cases d_energy =
    ``irrep_dims[0]`` and the return is the scalar kappa * sqrt(d) (byte-identical to before).
    UNEQUAL blocks (the so_n/sp_n irrep towers, e.g. the SO(3) spin dims [1, 3, 5, 7]) get a
    per-head (H,) tau with tau_h = kappa_h * sqrt(d_h), so every head's softmax runs at the
    Vaswani temperature of the dimension ITS energy accumulates over; a scalar kappa broadcasts
    across heads, a per-head (H,) kappa is elementwise. The (H,) tau broadcasts downstream via
    ``_broadcast_tau`` (which also handles device placement against the energy).
    """
    if isinstance(kappa, torch.Tensor) and kappa.dim() not in (0, 1):
        raise ValueError(
            f"kappa tensor must be 0-d (scalar) or 1-d (per-head); got {kappa.dim()}-d "
            f"shape {tuple(kappa.shape)}"
        )
    if (isinstance(kappa, torch.Tensor) and kappa.dim() == 1
            and kappa.shape[0] != len(irrep_dims)):
        raise ValueError(
            f"per-head kappa has {kappa.shape[0]} entries but the group has "
            f"{len(irrep_dims)} irrep blocks (irrep_dims={irrep_dims})"
        )
    if len(set(irrep_dims)) == 1:
        return kappa * (irrep_dims[0] ** 0.5)
    sqrt_d = _sqrt_dims(                                           # (H,) per-block sqrt(d_h), cached
        tuple(irrep_dims),
        kappa.device if isinstance(kappa, torch.Tensor) else torch.device("cpu"),
        kappa.dtype if isinstance(kappa, torch.Tensor) else torch.float32,
    )
    return kappa * sqrt_d


def query_adaptive_tau(
    sigma:      torch.Tensor,             # (..., N, K) DIAGONAL query-belief variances (detached here)
    tau:        'float | torch.Tensor',   # base temperature: scalar or per-head (H,) (attention_tau)
    irrep_dims: List[int],                # gauge-irrep block sizes; sum == K

    *,
    c:          float = 1.0,              # strength (cfg.query_tau_c); 0 -> the base tau on every row
) -> torch.Tensor:                        # (..., N, 1) single-block or (..., H, N, 1) per-head tau_i
    r"""Per-query adaptive softmax temperature (cfg.query_adaptive_tau):

        tau_{i,h} = tau_h * (1 + c * tr_h(Sigma_i) / d_h),

    tr_h the trace of query i's covariance over irrep block h (d_h = block size): an uncertain
    query (large tr Sigma) runs a HOTTER softmax and hedges over keys; a confident one commits.
    This is an opt-in non-compact-GL(K)-breaking baseline because
    ``tr(g Sigma g^T) != tr(Sigma)`` for a general non-orthogonal frame. It is invariant for
    orthogonal gauges and the ``c=0`` path is the gauge-pure base temperature.
    DETACHED state function of the CURRENT belief (``sigma`` is detached here), so no gradient
    flows into the belief through the temperature and the closed-form belief kernel -- which
    consumes tau only through beta = softmax_j(B - E/tau) -- stays exact. The returned per-row
    tau carries a trailing singleton key axis so it broadcasts against the (..., [H,] N, N)
    energy in ``attention_weights``/``log_partition``; ``reduced_free_energy`` squeezes that axis
    against the (..., [H,] N) log-partition. Monotone increasing in tr Sigma for c > 0; at c = 0
    it equals the base tau on every row (value-identical to the scalar path).
    """
    if not math.isfinite(c) or c < 0.0:
        raise ValueError(f"c must be finite and >= 0, got {c}")
    sig = sigma.detach()
    if len(irrep_dims) == 1:
        scale = 1.0 + c * sig.sum(dim=-1, keepdim=True) / float(irrep_dims[0])   # (..., N, 1)
        return tau * scale
    tr  = torch.stack([blk.sum(dim=-1)                                           # (..., H, N) per-block traces
                       for blk in sig.split(list(irrep_dims), dim=-1)], dim=-2)
    d_h = torch.tensor([float(d) for d in irrep_dims], device=sig.device, dtype=sig.dtype)
    scale = (1.0 + c * tr / d_h.view(-1, 1)).unsqueeze(-1)                       # (..., H, N, 1)
    if isinstance(tau, torch.Tensor) and tau.dim() == 1:                         # per-head (H,) base tau
        return tau.to(device=sig.device).view(-1, 1, 1) * scale
    return tau * scale


_SQRT_D_CACHE: Dict[Tuple[Tuple[int, ...], str, torch.dtype], torch.Tensor] = {}


def _sqrt_dims(dims: Tuple[int, ...], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    r"""Cached per-block sqrt(d_h) vector: attention_tau is called every vfe_block invocation,
    and the unequal-dims branch was allocating this loop-invariant tensor each time (audit
    2026-06-09 overnight F7)."""
    key = (dims, str(device), dtype)
    t = _SQRT_D_CACHE.get(key)
    if t is None:
        t = torch.tensor([float(d) for d in dims], device=device, dtype=dtype).sqrt()
        _SQRT_D_CACHE[key] = t
    return t


def _stackable_for_batching(
    q_b: BeliefParams,                     # broadcast query (..., N, 1, K)
    key: BeliefParams,                     # transported key  (..., N, N, K)
) -> bool:
    r"""Whether the equal-block batched path is bit-identical to the per-block loop.

    Stacking the H equal blocks along a NEW LEADING axis is the same arithmetic only when that
    axis does not perturb the mu/sigma broadcast. For a Gaussian family the canonical layout is
    sigma rank == mu rank (diagonal) or mu rank + 1 (full); the misclassified case (sigma carries
    a leading batch dim mu lacks) would make the stacked head axis right-align against sigma's
    first batch dim and broadcast spuriously. A family that does not expose mu/sigma tensors falls
    back to the loop. The guard is on q_b AND key, since pairwise_energy slices both.
    """
    expected_extra = {"diagonal": 0, "full": 1}.get(getattr(type(q_b), "cov_kind", None))
    if expected_extra is None:
        return False
    for params in (q_b, key):
        mu = getattr(params, "mu", None)
        sigma = getattr(params, "sigma", None)
        if mu is None or sigma is None:
            return False
        if sigma.dim() != mu.dim() + expected_extra:
            return False
    return True


def pairwise_energy(
    q:                 BeliefParams,        # (..., N, K) query belief
    key:               BeliefParams,        # (..., N, N, K) transported key belief Omega_ij q_j

    *,
    alpha:             float = 1.0,
    kl_max:            float = 100.0,
    eps:               float = 1e-6,
    divergence_family: str   = "renyi",

    irrep_dims:        Optional[List[int]] = None,
) -> torch.Tensor:                         # (..., N, N) or (..., H, N, N) E_ij = D(q_i || Omega_ij q_j)
    r"""Per-pair belief-coupling energy via the divergence seam (KL = Renyi at alpha=1).

    Divergence-agnostic on two orthogonal axes: ``divergence_family`` selects the FUNCTIONAL
    (renyi, ...) and the BeliefParams subclass of ``q``/``key`` the covariance kernel
    (gaussian_diagonal/gaussian_full), so a new f-divergence or covariance structure slots in
    by registration / constructing the right params, without editing here.

    PER-HEAD (GL(K) attention): when ``irrep_dims`` has more than one block the energy is
    computed PER IRREP BLOCK h -- E_ij^(h) = D(q_i^(h) || Omega_ij^(h) q_j^(h)) over block h's
    coordinates (the Gaussian MARGINAL d_head x d_head sub-block for the full family) -- and a
    leading head axis is returned: (..., H, N, N). With ``irrep_dims`` None or a single block
    the full-K energy (..., N, N) is returned, bit-identical to the legacy single-beta path.

    The query's key axis is inserted by ``broadcast_over_keys`` (the params own their covariance
    layout), staying correct when sigma_q carries a leading batch dim mu_q lacks.
    """
    functional = get_functional(divergence_family)
    q_b = q.broadcast_over_keys()          # (..., N, 1, K) broadcast query over keys

    if irrep_dims is None or len(irrep_dims) == 1:
        return functional(q_b, key, alpha=alpha, kl_max=kl_max, eps=eps)

    # EQUAL-size, more-than-one blocks (the default block_glk case): the H per-block divergences
    # are the SAME functional over H disjoint coordinate slices, so stack the H equal blocks along
    # a NEW LEADING axis and call the functional ONCE. The leading head axis is then moved to -3 to
    # match the loop's torch.stack(..., dim=-3) layout (..., H, N, N) EXACTLY. The stacked call is
    # bit-identical only when stacking does not perturb the mu/sigma broadcast: the misclassified
    # case where sigma carries a leading batch dim mu lacks would right-align the new head axis
    # against sigma's first batch dim (a spurious broadcast), so it is excluded by the rank guard
    # below and falls back to the per-block loop (which broadcasts correctly).
    H = len(irrep_dims)
    d = irrep_dims[0]
    if len(set(irrep_dims)) == 1 and _stackable_for_batching(q_b, key):
        q_parts   = [q_b.block(h * d, (h + 1) * d) for h in range(H)]      # H x (..., N, 1, d)
        key_parts = [key.block(h * d, (h + 1) * d) for h in range(H)]      # H x (..., N, N, d)
        q_stk   = type(q_b).stack(q_parts, dim=0)                          # (H, ..., N, 1, d)
        key_stk = type(key).stack(key_parts, dim=0)                        # (H, ..., N, N, d)
        e = functional(q_stk, key_stk, alpha=alpha, kl_max=kl_max, eps=eps)  # (H, ..., N, N)
        return torch.movedim(e, 0, -3)     # (..., H, N, N), matching the loop's stack(dim=-3)

    energies = []
    start = 0
    for db in irrep_dims:                  # one divergence per irrep block (head)
        end = start + db
        e_h = functional(
            q_b.block(start, end), key.block(start, end),
            alpha=alpha, kl_max=kl_max, eps=eps,
        )
        energies.append(e_h)
        start = end
    return torch.stack(energies, dim=-3)   # (..., H, N, N)


def self_divergence(
    q:                 BeliefParams,        # (..., N, K) belief
    p:                 BeliefParams,        # (..., N, K) prior

    *,
    alpha:             float = 1.0,
    kl_max:            float = 100.0,
    eps:               float = 1e-6,
    divergence_family: str   = "renyi",
) -> torch.Tensor:                         # (..., N) D(q_i || p_i)
    r"""Self-coupling divergence via the seam (full-K, not per-head: D(q_i||p_i) is the whole
    belief). ``divergence_family`` selects the functional; the params' subclass the kernel."""
    return get_functional(divergence_family)(
        q, p, alpha=alpha, kl_max=kl_max, eps=eps,
    )


def self_divergence_per_coord(
    q:                 BeliefParams,        # (..., N, K) belief
    p:                 BeliefParams,        # (..., N, K) prior

    *,
    alpha:             float = 1.0,
    kl_max:            float = 100.0,
    eps:               float = 1e-6,
    divergence_family: str   = "renyi",
) -> torch.Tensor:                         # (..., N, K) per-coordinate D^(k)(q_i||p_i)
    r"""Per-coordinate self-divergence D^(k)(q_i||p_i), unsummed over the coordinate axis.

    Defined only for a diagonal-covariance family (full-covariance KL couples coordinates
    through the trace and log-determinant and does not decompose) and for a divergence that
    decomposes coordinate-wise -- the per-coordinate functional registry: Renyi/KL, plus the
    two divergences AFFINE in the Renyi divergence (Bhattacharyya = 0.5 D_{1/2}, Jeffreys =
    KL + KL_rev). ``squared_hellinger`` is excluded (H^2 = 1 - exp(-D_{1/2}/2) is a nonlinear
    transform of the SUMMED divergence). Both are enforced by raising, so a non-decomposing
    functional cannot silently sum the wrong thing. Consumed by the ``state_dependent_per_coord``
    alpha form via ``self_divergence_for_alpha``.
    """
    if q.cov_kind != "diagonal":
        raise ValueError(
            f"self_divergence_per_coord needs a diagonal-covariance family (full-covariance KL "
            f"does not decompose coordinate-wise); got cov_kind={q.cov_kind!r}"
        )
    if not has_per_coord_functional(divergence_family):
        raise ValueError(
            f"self_divergence_per_coord has no per-coordinate form for "
            f"divergence_family={divergence_family!r}: only divergences that decompose "
            f"coordinate-wise on a diagonal Gaussian are registered "
            f"({divergence_functionals_per_coord()}). 'squared_hellinger' is excluded (H^2 = "
            f"1 - exp(-D_{{1/2}}/2) is a nonlinear transform of the summed divergence). Use a "
            f"decomposable divergence or a per-position lambda_alpha_mode."
        )
    return get_functional_per_coord(divergence_family)(q, p, alpha=alpha, kl_max=kl_max, eps=eps)


def self_divergence_for_alpha(
    q:                 BeliefParams,        # (..., N, K) belief
    p:                 BeliefParams,        # (..., N, K) prior

    *,
    alpha:             float = 1.0,
    kl_max:            float = 100.0,
    eps:               float = 1e-6,
    divergence_family: str   = "renyi",
    lambda_alpha_mode: str   = "constant",
) -> torch.Tensor:                         # (..., N) summed, or (..., N, K) per-coordinate
    r"""Self-divergence shaped for the selected alpha form: per-coordinate (..., N, K) when the
    form declares ``per_coord=True`` (``alpha_i.alpha_is_per_coord``), else the per-position
    summed (..., N). This is the single routing seam every alpha consumer (the autograd oracle,
    the analytic kernel, the e_step F value, model diagnostics) shares, so a new alpha form's
    divergence-reduction need is honoured by its registration alone, with no consumer edited.
    """
    if alpha_is_per_coord(lambda_alpha_mode):
        return self_divergence_per_coord(
            q, p, alpha=alpha, kl_max=kl_max, eps=eps,
            divergence_family=divergence_family,
        )
    return self_divergence(
        q, p, alpha=alpha, kl_max=kl_max, eps=eps,
        divergence_family=divergence_family,
    )


def attention_weights(
    energy:    torch.Tensor,               # (..., N) or (..., N, N) per-key energies E_ij

    *,
    tau:       'float | torch.Tensor' = 1.0,
    log_prior: Optional[torch.Tensor] = None,   # (..., N/NxN) bias B_ij; None -> 0
) -> torch.Tensor:                         # (...) softmax_j(B - E/tau)
    r"""Attention weights beta*_ij = softmax_j(B_ij - E_ij / tau)."""
    logits = -energy / _broadcast_tau(tau, energy)
    if log_prior is not None:
        logits = logits + log_prior
    return torch.softmax(logits, dim=-1)


def log_partition(
    energy:    torch.Tensor,               # (..., N) or (..., N, N)

    *,
    tau:       'float | torch.Tensor' = 1.0,
    log_prior: Optional[torch.Tensor] = None,
) -> torch.Tensor:                         # (...) log Z_i = logsumexp_j(log pi - E/tau)
    r"""Log-partition log Z_i = logsumexp_j(log pi_ij - E_ij / tau), pi = softmax_j(B).

    The partition Z_i = Sum_j pi_ij exp(-E_ij/tau) is built from the NORMALIZED
    prior pi (not the raw log-bias B), so the envelope identity
    Sum_j beta*_ij E_ij + tau Sum_j beta*_ij log(beta*_ij/pi_ij) = -tau log Z_i
    holds for ANY prior the seam emits. Equivalently log Z = logsumexp(B - E/tau)
    - logsumexp(B); using log_softmax(B) subtracts that per-row normalizer in one
    step. With a None prior pi is uniform 1/N, so the bias is -log(N).
    """
    logits = -energy / _broadcast_tau(tau, energy)
    if log_prior is not None:
        logits = logits + torch.log_softmax(log_prior, dim=-1)
    else:
        logits = logits - torch.log(torch.tensor(float(energy.shape[-1]),
                                                  device=energy.device, dtype=energy.dtype))
    return torch.logsumexp(logits, dim=-1)


def reduced_free_energy(
    energy:    torch.Tensor,               # (..., N) or (..., N, N)

    *,
    tau:       'float | torch.Tensor' = 1.0,
    log_prior: Optional[torch.Tensor] = None,
) -> torch.Tensor:                         # (...) F_red,i = -tau log Z_i
    r"""Reduced (envelope) free energy F_red,i = -tau log Z_i; equals the canonical
    beta-block evaluated at beta* for ANY prior (log_partition normalizes the
    prior internally, so the +tau logsumexp(B) per-row offset cannot leak in)."""
    lz = log_partition(energy, tau=tau, log_prior=log_prior)   # (..., N) or (..., H, N)
    # Per-head (H,) tau must broadcast against lz (..., H, N): reshape to (H, 1) so it aligns
    # with the head axis at -2 of lz. (H,1) right-aligns correctly for both (H,N) and (B,H,N).
    # Scalar tau passes through unchanged. The .to() mirrors _broadcast_tau's device hop (a
    # scalar-kappa x unequal-irrep-dims tau is born on CPU). A >= 2-d PER-QUERY tau
    # (..., [H,] N, 1) (query_adaptive_tau) drops its singleton key axis to align with lz's rows.
    if isinstance(tau, torch.Tensor) and tau.dim() == 1:
        _tau = tau.to(device=lz.device).reshape(tau.shape[0], 1)
    elif isinstance(tau, torch.Tensor) and tau.dim() >= 2:
        _tau = tau.squeeze(-1)
    else:
        _tau = tau
    return -_tau * lz


def free_energy(
    self_div:                  torch.Tensor,        # (..., N) or (..., N, K) D(q_i||p_i)
    energy:                    torch.Tensor,        # (..., N, N) E_ij belief-coupling energies
    alpha:                     torch.Tensor,        # (..., N) or (..., N, K) self-coupling

    *,
    tau:                       'float | torch.Tensor' = 1.0,
    lambda_beta:               'float | torch.Tensor' = 1.0,    # weight on the WHOLE belief-coupling block
    log_eps:                   float = 1e-12,                   # floor for log(beta)/log(pi) in the entropy term
    lambda_twohop:             float = 0.0,                     # weight on the two-hop coupling block (0 = pure F)
    include_attention_entropy: bool  = True,

    log_prior:                 Optional[torch.Tensor] = None,   # (..., N, N) attention log-prior
    alpha_reg:                 Optional[torch.Tensor] = None,   # (..., N[,K]) R(alpha) if state-dep
    coupling_energy:           Optional[torch.Tensor] = None,   # (..., N, N) VALUE-gauge energy for the coupling sum (None -> energy)
    log_likelihood:            Optional[torch.Tensor] = None,   # (..., N) E_q[log p(o|k)] observation term; GATED STUB (see docstring)
) -> torch.Tensor:                                  # scalar F = sum_i F_i
    r"""Single authoritative scalar free energy (default path; lambda_h=0, gamma=0).

        F = sum_i [ alpha_i . D(q_i||p_i)            (+ R(alpha_i) if state-dependent)
                  + lambda_beta . ( sum_j beta_ij E_ij
                                    + tau sum_j beta_ij log(beta_ij/pi_ij) )   (entropy: canonical only)
                  - ell_i ]
    beta_ij = softmax_j(log_prior - E/tau); pi = softmax_j(log_prior). ``lambda_beta`` (1.0 = the
    canonical/pure F) scales the COUPLING and ENTROPY together by the SAME factor and leaves beta
    untouched (no lambda inside the softmax), so beta = softmax(-E/tau) stays the stationary point of
    the scaled block and the envelope identity d/dtheta[lambda_beta (coupling+entropy)] =
    lambda_beta sum_j beta* dE/dtheta still holds -- keeping the analytic kernel (which scales its
    pair term by lambda_beta) in agreement with autograd of this F. ``lambda_twohop`` (0.0 = pure)
    adds the two-hop coupling block F_2 = lambda_2 sum_ik (beta beta)_ik E_ik with DETACHED hop
    weights and no entropy term (see the guarded block below). The hyper-prior lambda_h
    KL(s||h) and model-coupling gamma KL(s_i||Omega s_j) are extension points, absent from this
    default path.

    The observation/data term -E_q[log p(o|k)] enters via the optional ``log_likelihood`` arg
    (subtracted below) but is a GATED STUB: no production caller supplies it, and this is the
    ORACLE-path mirror only. The default belief descent runs through the analytic kernel
    (``gradients/kernels.py`` -> ``get_kernel``), which never calls this function, so a live
    observation pull must ALSO be injected into ``gradients/kernels.py`` and ``gradients/oracle.py``;
    wiring it here alone is inert. The term is non-vacuous only once the per-token prior is replaced
    by a top-down cross-scale shadow prior (PIFB:1233): in the current next-token model the input
    token already sets p_i (encode q=p) and the next token already drives the cross-entropy
    (``metrics.py`` books that CE as the data term), so an observation carries no information distinct
    from both. Do NOT "complete the canonical functional" by wiring ``log_likelihood`` here plus one
    caller and believing it live. See docs/2026-06-07-observation-likelihood-term-brainstorm.md.

    Divergence-agnostic: `self_div`/`energy` come from the divergence seam, so a new
    divergence requires no change here.
    """
    beta = attention_weights(energy, tau=tau, log_prior=log_prior)        # (..., N, N) from the SCORE energy

    # self-coupling (sum over coordinate axis too when alpha/self_div are per-coord)
    self_term = alpha * self_div
    if alpha_reg is not None:
        self_term = self_term + alpha_reg
    self_total = self_term.sum()

    # The coupling sum carries the belief-coupling energy beta_ij E_ij. ``coupling_energy`` (None ->
    # the score ``energy``, the coherent single-operator default) lets the VALUE gauge differ from the
    # ATTENTION gauge: beta is still softmax of the score energy, but the energy summed (the pull the
    # belief descends) is the value-gauge energy. With coupling_energy=energy this is byte-identical;
    # decoupled, beta is no longer the stationary point of the summed block, so the envelope theorem
    # fails and only the autograd oracle (not the closed-form kernel) computes the gradient correctly.
    coupling = (beta * (energy if coupling_energy is None else coupling_energy)).sum()

    F = self_total + lambda_beta * coupling
    if include_attention_entropy:
        # Uniform prior (log_prior=None): log pi = -log N is a SCALAR, so no (..., N, N) pi
        # tensor is materialized (audit 2026-06-09 overnight F8 / morning PE7). The max()
        # mirrors the tensor branch's clamp exactly (inert for any real N < 1/log_eps).
        if log_prior is not None:
            # Exact log-prior (m8): the old torch.log(softmax(...).clamp(min=log_eps)) floored a finite
            # deep-tail entry at ~-27.6 nats, so a strong finite prior made F and the autograd oracle
            # deviate from -tau logZ. log_softmax is exact; isfinite neutralizes hard-mask (-inf) entries
            # (beta==0 there, so 0*(.-log_pi) stays the correct 0 with no 0*(-inf)=NaN).
            log_pi = torch.log_softmax(log_prior, dim=-1)
            log_pi = torch.where(torch.isfinite(log_pi), log_pi, torch.zeros_like(log_pi))
        else:
            log_pi = math.log(max(1.0 / beta.shape[-1], log_eps))
        _tau_e = _broadcast_tau(tau, energy)          # (H,1,1) for per-head, scalar otherwise
        entropy = (_tau_e * (beta * (torch.log(beta.clamp(min=log_eps)) - log_pi))).sum()
        F = F + lambda_beta * entropy
    if lambda_twohop != 0.0:
        # Two-hop coupling block (cfg.lambda_twohop; 0.0 = OFF, pure canonical F):
        #     F_2 = lambda_2 * sum_{i,k} W2_ik E_ik,   W2 = beta beta (per head, over the key axis),
        # the beta-weighted two-step relaxation on the SAME pairwise energy grid (the flat cocycle
        # composes exactly, Omega_ij Omega_jk = Omega_ik, so the existing (i,k) transported energies
        # serve verbatim). W2 is DETACHED on both factors (the fixed hop-weight convention shared
        # with the analytic kernel's pair term) and carries NO entropy term (W2 is a derived weight,
        # not a variational row distribution with its own prior). The energy summed follows the
        # coupling term's value-gauge selection above (identical on the coherent default path).
        w2 = beta.detach() @ beta.detach()            # (..., [H,] N, N) hop weights W2_ik = sum_j b_ij b_jk
        F = F + lambda_twohop * (w2 * (energy if coupling_energy is None else coupling_energy)).sum()
    if log_likelihood is not None:                              # observation/data term -E_q[log p(o|k)] (gated stub; no live caller)
        F = F - log_likelihood.sum()
    return F


# ===========================================================================
# Typed hierarchical (q/p/s/h) decomposition (PB-10).
#
# ``free_energy`` above is the pure q-only training scalar and keeps its byte-for-byte one-shot
# ``.sum()`` reductions; it does NOT delegate to the rows below, because float32 one-shot reductions
# are not bitwise associative with the row-then-query reductions the evaluator performs. The rows and
# the evaluator are the SINGLE typed boundary the diagnostics and the opt-in scored h/s outer loss
# route their hierarchy assembly through, so the hyper-prior and gamma channels are no longer summed
# into the total by hand at each call site.
# ===========================================================================

class BeliefFreeEnergyRows(NamedTuple):
    r"""Per-query (..., N) belief-channel rows: already SIGNED and WEIGHTED contributions.

    ``self_coupling`` is alpha_i D(q_i||p_i) (+ R(alpha_i)) reduced over any coordinate axis;
    ``belief_coupling`` is lambda_beta sum_j beta_ij E_ij; ``attention_entropy`` is
    lambda_beta tau sum_j beta_ij log(beta_ij/pi_ij) (an exact zero row when the entropy is gated
    off); ``twohop_coupling`` is lambda_twohop sum_k (beta beta)_ik E_ik (an exact zero row when
    lambda_twohop == 0); ``observation_nll`` is -E_q[log p(o|x)] (an exact zero row when no
    observation term is supplied). Absent blocks are ``torch.zeros_like`` of the self row, never a
    hidden branch."""
    self_coupling:     torch.Tensor
    belief_coupling:   torch.Tensor
    attention_entropy: torch.Tensor
    twohop_coupling:   torch.Tensor
    observation_nll:   torch.Tensor


class HierarchicalFreeEnergyTerms(NamedTuple):
    r"""The eight reduced hierarchical components and their assembled ``total``.

    ``total`` is the field-order sum: self + belief coupling + attention entropy + two-hop +
    hyper-prior + model coupling + meta-entropy + observation NLL."""
    self_coupling:     torch.Tensor
    belief_coupling:   torch.Tensor
    attention_entropy: torch.Tensor
    twohop_coupling:   torch.Tensor
    hyper_prior:       torch.Tensor
    model_coupling:    torch.Tensor
    meta_entropy:      torch.Tensor
    observation_nll:   torch.Tensor
    total:             torch.Tensor


def _reduce_row(row: torch.Tensor, how: str) -> torch.Tensor:
    r"""Reduce a per-query row to a scalar by ``"sum"`` or ``"mean"`` over all batch/query entries."""
    if how == "sum":
        return row.sum()
    if how == "mean":
        return row.mean()
    raise ValueError(f"reduction must be 'sum' or 'mean', got {how!r}")


def hierarchical_free_energy_terms(
    self_coupling_rows:     torch.Tensor,  # (..., N), already alpha-weighted and regularized
    belief_coupling_rows:   torch.Tensor,  # (..., N), lambda_beta * sum_j beta_ij E_ij
    attention_entropy_rows: torch.Tensor,  # (..., N), lambda_beta * tau * sum_j beta log(beta/pi)
    twohop_coupling_rows:   torch.Tensor,  # (..., N), lambda_twohop * sum_k (beta beta)_ik E_ik
    hyper_prior_rows:       torch.Tensor,  # (..., N), lambda_h_i D(s_i||h) + R_h
    model_coupling_rows:    torch.Tensor,  # (..., N), lambda_gamma * sum_j gamma_ij E^s_ij
    meta_entropy_rows:      torch.Tensor,  # (..., N), lambda_gamma * tau_g * sum_j gamma log(gamma/pi_s)
    observation_nll_rows:   torch.Tensor,  # (..., N), -E_q[log p(o|x)]

    *,
    q_reduction:     str = "sum",
    model_reduction: str = "mean",
) -> HierarchicalFreeEnergyTerms:
    r"""The single authoritative scalar evaluator over eight already-signed, already-weighted rows.

    The first four rows and ``observation_nll_rows`` reduce with ``q_reduction``; the three s-channel
    rows (``hyper_prior``, ``model_coupling``, ``meta_entropy``) reduce with ``model_reduction``. Each
    reduction is ``"sum"`` or ``"mean"`` applied directly to every batch/query entry after the caller
    has already collapsed the structural (coordinate/key/head) axes. Absent blocks are passed as
    ``torch.zeros_like`` rows -- there is no hidden branch. The evaluator validates matching
    ``(..., N)`` shapes and devices, reduces each row exactly once, and assembles ``total`` in the
    fixed field order. It NEVER detaches or reweights an input (the only detach in this module lives
    inside :func:`_belief_free_energy_rows`, forming the fixed two-hop weights); whether the model
    frame was built passively or attached is the caller's choice, no longer hidden here."""
    rows = (self_coupling_rows, belief_coupling_rows, attention_entropy_rows,
            twohop_coupling_rows, hyper_prior_rows, model_coupling_rows,
            meta_entropy_rows, observation_nll_rows)
    for r in rows:
        if not isinstance(r, torch.Tensor):
            raise TypeError(f"every hierarchical row must be a torch.Tensor, got {type(r)!r}")
    ref = self_coupling_rows
    for r in rows:
        if r.shape != ref.shape:
            raise ValueError(
                f"hierarchical rows must share one (..., N) shape; got {tuple(r.shape)} vs "
                f"{tuple(ref.shape)}")
        if r.device != ref.device:
            raise ValueError(
                f"hierarchical rows must share one device; got {r.device} vs {ref.device}")

    self_coupling     = _reduce_row(self_coupling_rows,     q_reduction)
    belief_coupling   = _reduce_row(belief_coupling_rows,   q_reduction)
    attention_entropy = _reduce_row(attention_entropy_rows, q_reduction)
    twohop_coupling   = _reduce_row(twohop_coupling_rows,   q_reduction)
    hyper_prior       = _reduce_row(hyper_prior_rows,       model_reduction)
    model_coupling    = _reduce_row(model_coupling_rows,    model_reduction)
    meta_entropy      = _reduce_row(meta_entropy_rows,      model_reduction)
    observation_nll   = _reduce_row(observation_nll_rows,   q_reduction)

    total = (self_coupling + belief_coupling + attention_entropy + twohop_coupling
             + hyper_prior + model_coupling + meta_entropy + observation_nll)
    return HierarchicalFreeEnergyTerms(
        self_coupling, belief_coupling, attention_entropy, twohop_coupling,
        hyper_prior, model_coupling, meta_entropy, observation_nll, total,
    )


def _belief_free_energy_rows(
    self_div:                  torch.Tensor,        # (..., N) or (..., N, K) D(q_i||p_i)
    energy:                    torch.Tensor,        # (..., N, N) or (..., H, N, N) E_ij
    alpha:                     torch.Tensor,        # (..., N) or (..., N, K) self-coupling

    *,
    tau:                       'float | torch.Tensor' = 1.0,
    lambda_beta:               'float | torch.Tensor' = 1.0,
    log_eps:                   float = 1e-12,
    lambda_twohop:             float = 0.0,
    include_attention_entropy: bool  = True,

    log_prior:                 Optional[torch.Tensor] = None,   # (..., N, N) attention log-prior
    alpha_reg:                 Optional[torch.Tensor] = None,   # (..., N[,K]) R(alpha) if state-dep
    coupling_energy:           Optional[torch.Tensor] = None,   # (..., N, N) VALUE-gauge coupling energy
    log_likelihood:            Optional[torch.Tensor] = None,   # (..., N) E_q[log p(o|k)]
    beta_override:             Optional[torch.Tensor] = None,   # captured beta (avoids recompute)
    per_coord:                 Optional[bool] = None,           # self_div carries a coord axis; None -> infer
) -> BeliefFreeEnergyRows:
    r"""Per-query (..., N) decomposition of the belief channel that :func:`free_energy` sums to a
    scalar -- a NEW hierarchical/diagnostic path, NOT the scalar's reduction order.

    It collapses the coordinate, key, and optional head axes while preserving the batch/query axes.
    The five returned rows are already SIGNED and WEIGHTED (lambda_beta in both beta rows,
    lambda_twohop in the two-hop row, ``observation_nll = -log_likelihood`` when present); the
    attention-entropy row is an exact zero when the entropy is gated off, and the two-hop and
    observation rows are exact zeros when absent. ``beta_override`` supplies a captured beta so
    diagnostic weights are not recomputed. The ONLY detach here forms the fixed two-hop weights
    W2 = beta.detach() @ beta.detach()."""
    beta = (beta_override if beta_override is not None
            else attention_weights(energy, tau=tau, log_prior=log_prior))
    value_energy = energy if coupling_energy is None else coupling_energy

    # Axis bookkeeping. ``energy`` is (..., N, N) or (..., H, N, N) (the head axis, block_glk, is 3rd
    # from last); ``self_div`` is (..., N) or per-coordinate (..., N, K) (state_dependent_per_coord).
    # The rank gap g = energy.dim() - self_div.dim() then reads: g == 2 -> head, no coord; g == 0 ->
    # coord, no head; g == 1 is AMBIGUOUS between (no coord, no head) and (coord AND head), so it is
    # disambiguated by shape -- (no coord, no head) requires self_div.shape == energy.shape[:-1]
    # exactly. Callers that know the alpha form (diagnostics) pass ``per_coord`` explicitly, which
    # also covers the H == N == K collision the shape test cannot separate.
    gap = energy.dim() - self_div.dim()
    if per_coord is None:
        if gap == 0:
            per_coord = True
        elif gap == 2:
            per_coord = False
        elif gap == 1:
            per_coord = self_div.shape != energy.shape[:-1]
        else:
            raise ValueError(
                f"self_div rank {self_div.dim()} is inconsistent with energy rank {energy.dim()}")
    has_head = (gap + (1 if per_coord else 0)) >= 2

    def _collapse(x: torch.Tensor) -> torch.Tensor:                   # (..., [H,] N) -> (..., N)
        return x.sum(dim=-2) if has_head else x

    belief_row = lambda_beta * _collapse((beta * value_energy).sum(dim=-1))

    self_term = alpha * self_div
    if alpha_reg is not None:
        self_term = self_term + alpha_reg
    if per_coord:
        self_term = self_term.sum(dim=-1)                             # collapse the coordinate axis
    self_row = self_term
    if self_row.shape != belief_row.shape:
        raise ValueError(
            f"belief rows disagree on the (..., N) shape: self {tuple(self_row.shape)} vs "
            f"coupling {tuple(belief_row.shape)}; pass per_coord explicitly if the axis "
            f"inference misread the layout")

    if include_attention_entropy:
        if log_prior is not None:
            log_pi = torch.log_softmax(log_prior, dim=-1)
            log_pi = torch.where(torch.isfinite(log_pi), log_pi, torch.zeros_like(log_pi))
        else:
            log_pi = math.log(max(1.0 / beta.shape[-1], log_eps))
        _tau_e = _broadcast_tau(tau, energy)
        entropy_full = _tau_e * (beta * (torch.log(beta.clamp(min=log_eps)) - log_pi))
        entropy_row = lambda_beta * _collapse(entropy_full.sum(dim=-1))
    else:
        entropy_row = torch.zeros_like(self_row)

    if lambda_twohop != 0.0:
        w2 = beta.detach() @ beta.detach()                            # fixed hop weights (the ONLY detach)
        twohop_row = lambda_twohop * _collapse((w2 * value_energy).sum(dim=-1))
    else:
        twohop_row = torch.zeros_like(self_row)

    observation_row = (-log_likelihood if log_likelihood is not None
                       else torch.zeros_like(self_row))

    return BeliefFreeEnergyRows(self_row, belief_row, entropy_row, twohop_row, observation_row)
