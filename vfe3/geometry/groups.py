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
    generate_son,
)


@dataclass
class GaugeGroup:
    """A structure group plus the metadata the transport layer consumes."""

    name:               str
    generators:         torch.Tensor          # (n_gen, K, K) Lie-algebra basis
    irrep_dims:         List[int]             # block sizes; sum == K
    skew_symmetric:     bool                  # exp(-M) = exp(M)^T fast path
    invariant_families: Tuple[str, ...] = ("gaussian",)

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
    dtype:   torch.dtype = torch.float32,
) -> GaugeGroup:
    """Full GL(K): single block, full gl(K) generators."""
    G = generate_glk(K, dtype=dtype)
    return GaugeGroup(name="glk", generators=G, irrep_dims=[K], skew_symmetric=False)


@register_group("block_glk")
def _build_block_glk(
    K:               int,
    n_heads:         int,

    *,
    cross_couplings: Optional[List[Tuple[int, int]]] = None,
    close_basis:     bool                            = False,
    dtype:           torch.dtype                     = torch.float32,
) -> GaugeGroup:
    """Block-diagonal GL(K) = GL(d_head)^n_heads, optional cross-head coupling.

    With ``cross_couplings`` the basis includes off-block generators; with
    ``close_basis=True`` it is closed under the Lie bracket into a subalgebra
    of gl(K) (so the exponentiated group is well-defined).
    """
    d_head = K // n_heads
    if cross_couplings:
        G = generate_glk_cross_head(K, n_heads, cross_couplings, dtype=dtype)
        if close_basis:
            G, _ = close_under_brackets(G)
    else:
        G = generate_glk_multihead(K, n_heads, dtype=dtype)
    return GaugeGroup(
        name="block_glk",
        generators=G,
        irrep_dims=[d_head] * n_heads,
        skew_symmetric=False,
    )


@register_group("so_k")
def _build_so_k(
    K:       int,

    *,
    dtype:   torch.dtype = torch.float32,
) -> GaugeGroup:
    """SO(K): skew-symmetric so(K) generators (single block)."""
    G = generate_son(K, dtype=dtype)
    return GaugeGroup(name="so_k", generators=G, irrep_dims=[K], skew_symmetric=True)
