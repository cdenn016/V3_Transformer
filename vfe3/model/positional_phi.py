r"""Per-position gauge-frame positional encoding (BCH-PE) for VFE_3.0.

A registry of per-position Lie-algebra coordinate builders ``pos_phi_i in R^{n_gen}``
that are composed into the token gauge frame via :func:`vfe3.geometry.lie_ops.compose_phi`
BEFORE transport, so position enters through the gauge transport ``Omega_ij`` (the
self-transport ``Omega_ii = I`` is unaffected). Default-off: ``"none"`` returns no
coordinates and the frame is unchanged. ``"learned"`` slices a model-owned parameter
table; ``"frozen"`` is the parameter-free ``i * scale`` on one generator axis (a
Lie-algebra ALiBi).
"""

from typing import Callable, Dict, Optional

import torch

from vfe3.geometry.groups import GaugeGroup
from vfe3.geometry.lie_ops import compose_phi, project_phi_to_slk

_POS_PHI: Dict[str, Callable[..., Optional[torch.Tensor]]] = {}


def register_pos_phi(name: str) -> Callable:
    """Decorator registering a pos-phi coordinate builder -> (N, n_gen) coords or None."""
    def _wrap(fn: Callable[..., Optional[torch.Tensor]]) -> Callable[..., Optional[torch.Tensor]]:
        _POS_PHI[name] = fn
        return fn
    return _wrap


def get_pos_phi(name: str) -> Callable[..., Optional[torch.Tensor]]:
    """Return the registered pos-phi builder (KeyError-with-available-list if absent)."""
    if name not in _POS_PHI:
        raise KeyError(f"no pos_phi {name!r}; available: {sorted(_POS_PHI)}")
    return _POS_PHI[name]


@register_pos_phi("none")
def _pos_phi_none(
    n:     int,
    n_gen: int,

    *,
    device: torch.device,
    dtype:  torch.dtype = torch.float32,
    **kwargs,
) -> Optional[torch.Tensor]:
    r"""No positional element: returns None (the frame is left unchanged)."""
    return None


@register_pos_phi("learned")
def _pos_phi_learned(
    n:     int,
    n_gen: int,

    *,
    pos_phi_free: Optional[torch.Tensor] = None,   # (max_seq_len, n_gen) model-owned table
    device:       torch.device,
    dtype:        torch.dtype = torch.float32,
    **kwargs,
) -> torch.Tensor:
    r"""Learned absolute positional coords: the first ``n`` rows of the model's table."""
    if pos_phi_free is None:
        raise ValueError("pos_phi='learned' requires the model-owned pos_phi_free table")
    return pos_phi_free[:n]


@register_pos_phi("frozen")
def _pos_phi_frozen(
    n:     int,
    n_gen: int,

    *,
    scale:       float = 0.02,
    frozen_axis: int   = 0,
    device:      torch.device,
    dtype:       torch.dtype = torch.float32,
    **kwargs,
) -> torch.Tensor:
    r"""Parameter-free Lie-algebra ALiBi: pos_phi_i = (i * scale) on one generator axis."""
    coords = torch.zeros(n, n_gen, device=device, dtype=dtype)
    coords[:, frozen_axis] = torch.arange(n, device=device, dtype=dtype) * scale
    return coords


def positional_phi_coords(
    mode:  str,
    n:     int,
    n_gen: int,

    *,
    scale:        float = 0.02,
    frozen_axis:  int   = 0,
    pos_phi_free: Optional[torch.Tensor] = None,
    device:       torch.device,
    dtype:        torch.dtype = torch.float32,
    **kwargs,                              # variant params flow through (never edit this dispatcher)
) -> Optional[torch.Tensor]:
    r"""Dispatch to the registered pos-phi builder ``mode``; returns (N, n_gen) coords or None."""
    return get_pos_phi(mode)(
        n, n_gen, scale=scale, frozen_axis=frozen_axis,
        pos_phi_free=pos_phi_free, device=device, dtype=dtype, **kwargs,
    )


def apply_positional_phi(
    phi:   torch.Tensor,                  # (..., N, n_gen) token gauge frame
    group: GaugeGroup,

    *,
    mode:         str   = "none",
    compose_mode: str   = "bch",
    order:        int   = 4,
    scale:        float = 0.02,
    frozen_axis:  int   = 0,
    project_slk:  bool  = False,
    pos_phi_free: Optional[torch.Tensor] = None,
    **kwargs,                             # variant params flow through to the builder
) -> torch.Tensor:
    r"""Compose the per-position element into ``phi`` via ``compose_phi`` (BCH by default).

    ``"none"`` returns ``phi`` unchanged (byte-identical pure path). Otherwise the (N, n_gen)
    coords broadcast over any leading batch axis through ``compose_phi``. ``project_slk`` removes
    the per-block trace from the positional element so ``det(Omega_h) = 1`` is preserved.
    """
    n, n_gen = phi.shape[-2], phi.shape[-1]
    coords = positional_phi_coords(
        mode, n, n_gen, scale=scale, frozen_axis=frozen_axis,
        pos_phi_free=pos_phi_free, device=phi.device, dtype=phi.dtype, **kwargs,
    )
    if coords is None:
        return phi
    if project_slk:
        coords = project_phi_to_slk(coords, group.generators, group.irrep_dims)
    return compose_phi(phi, coords, group.generators, order=order, mode=compose_mode)
