r"""Gauge-group registry for VFE_3.0 (structure-group axis of geometry).

A GaugeGroup bundles the Lie-algebra generators with the metadata transport
needs (block/irrep structure, skew flag) and declares the families whose
divergence is invariant under its representation (admissibility). Groups are
config-selected by name so variants swap without editing call sites.

Admissibility: a (family, group) pair is valid iff the family's divergence is
invariant under common pushforward by the group's representation,
D(rho(g) q || rho(g) p) = D(q || p). For the Gaussian family with the GL(K)
congruence action (mu -> g mu, Sigma -> g Sigma g^T) this holds for every
g in G <= GL(K), so every group here is admissible for "gaussian".
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch

from vfe3.geometry.closure import close_under_brackets
from vfe3.geometry.generators import (
    generate_glk,
    generate_glk_cross_head,
    generate_glk_multihead,
    generate_glk_multihead_tied,
    generate_son,
    generate_sp,
)
from vfe3.geometry.irreps import direct_sum_generators


@dataclass
class GaugeGroup:
    """A structure group plus the metadata the transport layer consumes."""

    name:                 str
    generators:           torch.Tensor                       # (n_gen, K, K) Lie-algebra basis
    irrep_dims:           List[int]                          # block sizes; sum == K
    skew_symmetric:       bool                               # exp(-M) = exp(M)^T fast path
    omega_direct_capable: bool                   = False     # stored group-element frame U_i is implemented
    invariant_families:   Tuple[str, ...]        = ("gaussian", "gaussian_full")  # exactly invariant under the GL(K) congruence; the diagonal readout is NOT (see check_admissible)
    irrep_labels:         Optional[List[str]]    = None      # per-block label ('l1', 'sym2', ...); None = label-less
    algebra:              Optional[str]          = None      # irrep-registry algebra key ('so' | 'sp'); None = label-less

    # Cached pseudo-inverse of the generator Frobenius Gram (computed once, off __init__/eq/repr).
    _gram_pinv_cache:     Optional[torch.Tensor] = field(default=None, init=False, repr=False, compare=False)

    def gram_pinv(self) -> torch.Tensor:
        r"""Cached ``gram_pinv(generators)`` -- the Gram pseudo-inverse ``extract_phi`` projects with.

        ``extract_phi`` recovers algebra coordinates by solving ``Gram c = g`` with
        ``c = Gram^+ g``; ``Gram_ab = <G_a, G_b>_F`` depends ONLY on the fixed generator basis.
        The BCH positional-composition hot path (``compose_bch -> extract_phi``) recomputed a dense
        ``(n_gen, n_gen)`` float64 ``pinv`` every forward (for the orthonormal block_glk basis the
        Gram is exactly ``I``, so the recompute was pure waste). Caching is value-identical -- the
        returned tensor equals ``gram_pinv(self.generators)`` -- and recomputed only if the device
        changes (e.g. the group's generators were moved after the first call).
        """
        if (self._gram_pinv_cache is None
                or self._gram_pinv_cache.device != self.generators.device
                or self._gram_pinv_cache.dtype != self.generators.dtype):
            from vfe3.geometry.lie_ops import gram_pinv as _gram_pinv
            with torch.no_grad():
                self._gram_pinv_cache = _gram_pinv(self.generators)
        return self._gram_pinv_cache

    def __post_init__(self) -> None:
        K = self.generators.shape[-1]
        if sum(self.irrep_dims) != K:
            raise ValueError(
                f"sum(irrep_dims)={sum(self.irrep_dims)} must equal K={K}; "
                f"irrep_dims={self.irrep_dims}"
            )
        if self.irrep_labels is not None and len(self.irrep_labels) != len(self.irrep_dims):
            raise ValueError(
                f"irrep_labels has {len(self.irrep_labels)} entries but there are "
                f"{len(self.irrep_dims)} irrep blocks"
            )

    def invariant_for(self, family: str) -> bool:
        """Whether the divergence of ``family`` is invariant under this group."""
        return family in self.invariant_families


_GROUPS: Dict[str, Callable[..., GaugeGroup]] = {}


def register_group(
    name: str,

    *,
    override:             bool           = False,
    omega_direct_capable: Optional[bool] = None,
) -> Callable:
    """Decorator registering a GaugeGroup builder under ``name``.

    ``omega_direct_capable`` advertises whether the registered builder supports stored
    group-element frames. Duplicate keys fail closed (audit 2026-07-01 round-3): a second
    registration under an existing name silently shadowed the first. Pass ``override=True`` to
    replace deliberately.
    """
    def _wrap(fn: Callable[..., GaugeGroup]) -> Callable[..., GaugeGroup]:
        if name in _GROUPS and not override:
            raise KeyError(f"gauge group {name!r} already registered; pass override=True to replace")
        capability = (
            bool(getattr(fn, "omega_direct_capable", False))
            if omega_direct_capable is None
            else omega_direct_capable
        )
        setattr(fn, "omega_direct_capable", capability)
        _GROUPS[name] = fn
        return fn
    return _wrap


def get_group(name: str) -> Callable[..., GaugeGroup]:
    """Return the registered GaugeGroup builder for ``name`` (KeyError if absent)."""
    if name not in _GROUPS:
        raise KeyError(
            f"no gauge group registered under {name!r}; available: {sorted(_GROUPS)}"
        )
    return _GROUPS[name]


@register_group("glk", omega_direct_capable=True)
def _build_glk(
    K:       int,

    *,
    dtype:   torch.dtype                  = torch.float32,
    device:  'torch.device | str | None'  = None,
) -> GaugeGroup:
    """Full GL(K): single block, full gl(K) generators."""
    G = generate_glk(K, dtype=dtype, device=device)
    return GaugeGroup(
        name="glk",
        generators=G,
        irrep_dims=[K],
        skew_symmetric=False,
        omega_direct_capable=True,
    )


@register_group("block_glk", omega_direct_capable=True)
def _build_block_glk(
    K:               int,
    n_heads:         int,

    *,
    close_basis:     bool                            = False,
    dtype:           torch.dtype                     = torch.float32,
    device:          'torch.device | str | None'     = None,
    cross_couplings: Optional[List[Tuple[int, int]]] = None,
) -> GaugeGroup:
    """Block-diagonal GL(K) = GL(d_head)^n_heads, optional cross-head coupling.

    With ``cross_couplings`` the basis includes off-block generators; with
    ``close_basis=True`` it is closed under the Lie bracket into a subalgebra
    of gl(K) (so the exponentiated group is well-defined). A cross-coupled
    group is NOT block-diagonal with ``d_head`` blocks (its group elements have
    off-block entries), so ``irrep_dims`` is reported as the single block
    ``[K]``; the contiguous super-block decomposition (which needs head
    reordering) is a Phase 2b transport concern.
    """
    d_head = K // n_heads
    if cross_couplings:
        G = generate_glk_cross_head(K, n_heads, cross_couplings, dtype=dtype, device=device)
        if close_basis:
            G, _ = close_under_brackets(G)
        irrep_dims = [K]
    else:
        G = generate_glk_multihead(K, n_heads, dtype=dtype, device=device)
        irrep_dims = [d_head] * n_heads
    return GaugeGroup(
        name="block_glk",
        generators=G,
        irrep_dims=irrep_dims,
        skew_symmetric=False,
        omega_direct_capable=True,
    )


@register_group("tied_block_glk", omega_direct_capable=True)
def _build_tied_block_glk(
    K:               int,
    n_heads:         int,

    *,
    dtype:           torch.dtype                     = torch.float32,
    device:          'torch.device | str | None'     = None,
) -> GaugeGroup:
    r"""TIED block-diagonal GL(d_head): one shared GL(d_head) frame across all heads.

    Generators ``kron(I_{n_heads}, gl(d_head))`` (n_gen = d_head^2), so one per-token phi drives the
    SAME GL(d_head) element in every head -- a tied gauge. The group element stays K x K block-
    diagonal (``irrep_dims = [d_head] * n_heads``), so transport / per-head attention are unchanged;
    only the gauge is shared rather than per-head independent (``block_glk``). Under this tied gauge
    the Schur-commutant head mixer is exactly equivariant. NOTE: the per-block Killing preconditioner
    (``phi_precond_mode='killing_per_block'``) assumes generators that PARTITION per block (one gl
    per head); the tied generators each act on every block, so that mode does not apply here (config
    validation warns) -- use ``'none'``, ``'clip'``, or the ambient ``'killing'``.
    """
    d_head = K // n_heads
    G = generate_glk_multihead_tied(K, n_heads, dtype=dtype, device=device)
    return GaugeGroup(
        name="tied_block_glk",
        generators=G,
        irrep_dims=[d_head] * n_heads,
        skew_symmetric=False,
        omega_direct_capable=True,
    )


@register_group("so_k", omega_direct_capable=True)
def _build_so_k(
    K:       int,

    *,
    dtype:   torch.dtype                  = torch.float32,
    device:  'torch.device | str | None'  = None,
) -> GaugeGroup:
    """SO(K): skew-symmetric so(K) generators (single block)."""
    G = generate_son(K, dtype=dtype, device=device)
    return GaugeGroup(
        name="so_k",
        generators=G,
        irrep_dims=[K],
        skew_symmetric=True,
        omega_direct_capable=True,
    )


def check_admissible(
    group:      GaugeGroup,
    family:     str   = "gaussian",

    *,
    functional: str   = "renyi",
    alpha:      float = 1.0,
    n_samples:  int   = 8,
    batch:      int   = 5,
    scale:      float = 0.2,
    atol:       float = 1e-3,
    rtol:       float = 1e-3,
    seed:       int   = 0,
) -> bool:
    r"""Executably verify that ``functional`` is invariant under ``group``'s representation on ``family``.

    Draws ``n_samples`` random group elements ``g = exp(sum_a c_a G_a)`` (coefficients ~ ``scale`` *
    N(0,1)) and a random Gaussian belief PAIR, pushes the pair forward by the family's representation,
    and asserts ``D(rho(g) q || rho(g) p) == D(q || p)`` to tolerance, where ``D`` is the registered
    divergence ``functional`` (renyi/squared_hellinger). Returns ``True`` iff invariant for every
    sample, else ``False`` -- so it turns ``GaugeGroup.invariant_for``'s string declaration into a
    verified invariant and catches a wrongly-declared ``invariant_families``.

    The representation is family-specific. The FULL Gaussian uses the GL(K) congruence
    ``mu -> g mu, Sigma -> g Sigma g^T``, under which the divergence is invariant for every
    ``g in GL(K)``. The DIAGONAL Gaussian re-diagonalizes ``g Sigma g^T``, which is NOT invariant
    under a non-diagonal ``g`` (the verifier returns ``False``, correctly) -- the diagonal family is
    admissible only for the diagonal-scaling subgroup. A family with no implemented representation
    raises ``NotImplementedError`` (the extension point: expose its pushforward to widen this check).

    USAGE (audit 2026-06-13 L6): this is a VERIFICATION TOOL, exercised by the admissibility tests
    (test_admissibility_verifier.py), not a runtime guard wired into model/config construction --
    the per-build random-sample congruence draws would tax every model build for a contract that
    holds for all shipped (group, family) pairs. Run it in tests/CI when adding a group or declaring
    a new ``invariant_families`` entry; ``invariant_for`` is the cheap string-membership check used at
    runtime.
    """
    if family in ("gaussian", "gaussian_full"):
        diagonal_readout = False
    elif family == "gaussian_diagonal":
        diagonal_readout = True
    else:
        raise NotImplementedError(
            f"check_admissible implements only the Gaussian GL(K)-congruence representation; "
            f"family={family!r} needs its own pushforward map (expose it on the (family, group) "
            f"admissibility object to extend this check)."
        )
    from vfe3.families.base import get_functional                 # local import: avoid an import cycle
    from vfe3.families.gaussian import DiagonalGaussian, FullGaussian

    fn = get_functional(functional)
    G = group.generators
    K = sum(group.irrep_dims)
    dev, dt = G.device, G.dtype
    gen = torch.Generator(device=dev).manual_seed(seed)
    eye = torch.eye(K, device=dev, dtype=dt)

    def _div(mu_q, S_q, mu_p, S_p):
        if diagonal_readout:                                      # diagonal family reads only the variances
            q = DiagonalGaussian(mu_q, torch.diagonal(S_q, dim1=-2, dim2=-1))
            p = DiagonalGaussian(mu_p, torch.diagonal(S_p, dim1=-2, dim2=-1))
        else:
            q = FullGaussian(mu_q, S_q)
            p = FullGaussian(mu_p, S_p)
        return fn(q, p, alpha=alpha, kl_max=1e12)                 # kl_max high so no clamp masks invariance

    for _ in range(n_samples):
        coeff = scale * torch.randn(G.shape[0], generator=gen, device=dev, dtype=dt)
        g = torch.linalg.matrix_exp(torch.einsum("a,aij->ij", coeff, G))
        mu_q = torch.randn(batch, K, generator=gen, device=dev, dtype=dt)
        mu_p = torch.randn(batch, K, generator=gen, device=dev, dtype=dt)
        Aq = torch.randn(batch, K, K, generator=gen, device=dev, dtype=dt)
        Ap = torch.randn(batch, K, K, generator=gen, device=dev, dtype=dt)
        S_q = Aq @ Aq.transpose(-1, -2) + eye                    # random SPD
        S_p = Ap @ Ap.transpose(-1, -2) + eye
        base = _div(mu_q, S_q, mu_p, S_p)
        mu_q2 = torch.einsum("kl,nl->nk", g, mu_q)
        mu_p2 = torch.einsum("kl,nl->nk", g, mu_p)
        S_q2 = g @ S_q @ g.transpose(-1, -2)                     # congruence Sigma -> g Sigma g^T
        S_p2 = g @ S_p @ g.transpose(-1, -2)
        moved = _div(mu_q2, S_q2, mu_p2, S_p2)
        if not torch.allclose(base, moved, atol=atol, rtol=rtol):
            return False
    return True


@register_group("so_n", omega_direct_capable=True)
def _build_so_n(
    K:          int,

    *,
    group_n:    Optional[int]                   = None,
    irrep_spec: Optional[List[Tuple[str, int]]] = None,
    dtype:      torch.dtype                     = torch.float32,
    device:     'torch.device | str | None'     = None,
) -> GaugeGroup:
    r"""SO(N) acting on R^K through a DIRECT SUM of irreps (tied gauge; heads = irreps).

    The structure group is SO(N) with N = ``group_n`` DECOUPLED from K. ``irrep_spec`` lists
    (label, multiplicity) pairs; label 'l<p>' is the symmetric-traceless rank-p tensor irrep
    (for N = 3 the spin-p block, dim 2p + 1), laid out contiguously in spec order with
    sum(mult * dim) == K. One per-token phi coordinate vector (n_gen = N(N-1)/2) drives EVERY
    block through that block's irrep image -- the SAME group element in different irreps, a
    TIED gauge -- so the embedded generators are block-diagonal but do NOT partition per block
    (per-block phi preconditioners are undefined; config rejects them). All blocks are built
    in real orthonormal bases, so every embedded generator is exactly skew
    (skew_symmetric=True: exp(-M) = exp(M)^T transport fast path; det Omega = 1 structurally).
    'l0' (trivial) blocks carry ZERO generators: those coordinates are gauge-invariant by
    construction. Unequal block dims are supported end-to-end (per-block energy loop,
    per-head tau = kappa_h sqrt(d_h)); equal-dims specs (one label, mult > 1) additionally
    keep the fused factored-transport fast path and the Schur-commutant head mixer (kron(A, I_d)
    IS the commutant of mult copies of one irrep, and the tied gauge keeps it exactly
    equivariant).
    """
    if group_n is None or irrep_spec is None:
        raise ValueError(
            "gauge_group='so_n' requires group_n (the N of SO(N)) and irrep_spec "
            "([(label, mult), ...], labels 'l<p>'); set both in the config"
        )
    G_def = generate_son(group_n, dtype=torch.float64)
    G, dims = direct_sum_generators(G_def, algebra="so", irrep_spec=irrep_spec)
    if sum(dims) != K:
        raise ValueError(
            f"irrep_spec blocks {dims} sum to {sum(dims)} != K={K} (group_n={group_n})"
        )
    labels = [lab for lab, mult in irrep_spec for _ in range(int(mult))]
    return GaugeGroup(
        name="so_n",
        generators=G.to(dtype).to(device),
        irrep_dims=dims,
        skew_symmetric=True,
        omega_direct_capable=True,
        irrep_labels=labels,
        algebra="so",
    )


@register_group("sp_n", omega_direct_capable=True)
def _build_sp_n(
    K:          int,

    *,
    group_n:    Optional[int]                   = None,
    irrep_spec: Optional[List[Tuple[str, int]]] = None,
    dtype:      torch.dtype                     = torch.float32,
    device:     'torch.device | str | None'     = None,
) -> GaugeGroup:
    r"""Sp(2m,R) acting on R^K through a DIRECT SUM of Sym^p irreps (tied gauge).

    The structure group is Sp(group_n, R), group_n = 2m even, decoupled from K. Label
    'sym<p>' is Sym^p of the defining 2m-dim rep, dim C(2m+p-1, p) -- irreducible because
    contraction with the antisymmetric symplectic form annihilates symmetric tensors.
    'sym1' is the defining rep itself; 'sym0' is trivial (zero generators). One per-token
    phi (n_gen = m(2m+1)) drives every block (tied gauge; generators do not partition per
    block). sp images are NOT skew (skew_symmetric=False: transport pays both matrix
    exponentials, as for 'sp'/'glk'). Admissible for the Gaussian family via the GL(K)
    congruence, as every group here.
    """
    if group_n is None or irrep_spec is None:
        raise ValueError(
            "gauge_group='sp_n' requires group_n (= 2m, even) and irrep_spec "
            "([(label, mult), ...], labels 'sym<p>'); set both in the config"
        )
    G_def = generate_sp(group_n, dtype=torch.float64)
    G, dims = direct_sum_generators(G_def, algebra="sp", irrep_spec=irrep_spec)
    if sum(dims) != K:
        raise ValueError(
            f"irrep_spec blocks {dims} sum to {sum(dims)} != K={K} (group_n={group_n})"
        )
    labels = [lab for lab, mult in irrep_spec for _ in range(int(mult))]
    return GaugeGroup(
        name="sp_n",
        generators=G.to(dtype).to(device),
        irrep_dims=dims,
        skew_symmetric=False,
        omega_direct_capable=True,
        irrep_labels=labels,
        algebra="sp",
    )


@register_group("sp", omega_direct_capable=True)
def _build_sp(
    K:       int,

    *,
    dtype:   torch.dtype                  = torch.float32,
    device:  'torch.device | str | None'  = None,
) -> GaugeGroup:
    """Sp(2m,R): the real symplectic group (single block, NON-skew sp(2m,R) generators).

    K = 2m. sp(2m,R) = {A : J A + A^T J = 0} with J = [[0, I_m], [-I_m, 0]]; dim m(2m+1).
    The generators are not skew (skew_symmetric=False), so transport exponentiates them via
    the general matrix_exp path (as for glk). Admissible for the Gaussian family because the
    GL(K) congruence action makes the divergence invariant under any g in GL(K) <= Sp(2m,R).
    """
    G = generate_sp(K, dtype=dtype, device=device)
    return GaugeGroup(
        name="sp",
        generators=G,
        irrep_dims=[K],
        skew_symmetric=False,
        omega_direct_capable=True,
    )
