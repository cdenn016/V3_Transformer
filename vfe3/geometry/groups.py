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

from dataclasses import dataclass
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


@dataclass
class GaugeGroup:
    """A structure group plus the metadata the transport layer consumes."""

    name:               str
    generators:         torch.Tensor          # (n_gen, K, K) Lie-algebra basis
    irrep_dims:         List[int]             # block sizes; sum == K
    skew_symmetric:     bool                  # exp(-M) = exp(M)^T fast path
    invariant_families: Tuple[str, ...] = ("gaussian",)

    def __post_init__(self) -> None:
        K = self.generators.shape[-1]
        if sum(self.irrep_dims) != K:
            raise ValueError(
                f"sum(irrep_dims)={sum(self.irrep_dims)} must equal K={K}; "
                f"irrep_dims={self.irrep_dims}"
            )

    def invariant_for(self, family: str) -> bool:
        """Whether the divergence of ``family`` is invariant under this group."""
        return family in self.invariant_families


_GROUPS: Dict[str, Callable[..., GaugeGroup]] = {}


def register_group(name: str) -> Callable:
    """Decorator registering a GaugeGroup builder under ``name``."""
    def _wrap(fn: Callable[..., GaugeGroup]) -> Callable[..., GaugeGroup]:
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


@register_group("glk")
def _build_glk(
    K:       int,

    *,
    dtype:   torch.dtype                  = torch.float32,
    device:  'torch.device | str | None'  = None,
) -> GaugeGroup:
    """Full GL(K): single block, full gl(K) generators."""
    G = generate_glk(K, dtype=dtype, device=device)
    return GaugeGroup(name="glk", generators=G, irrep_dims=[K], skew_symmetric=False)


@register_group("block_glk")
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
    )


@register_group("tied_block_glk")
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
    )


@register_group("so_k")
def _build_so_k(
    K:       int,

    *,
    dtype:   torch.dtype                  = torch.float32,
    device:  'torch.device | str | None'  = None,
) -> GaugeGroup:
    """SO(K): skew-symmetric so(K) generators (single block)."""
    G = generate_son(K, dtype=dtype, device=device)
    return GaugeGroup(name="so_k", generators=G, irrep_dims=[K], skew_symmetric=True)


@register_group("sp")
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
    return GaugeGroup(name="sp", generators=G, irrep_dims=[K], skew_symmetric=False)
