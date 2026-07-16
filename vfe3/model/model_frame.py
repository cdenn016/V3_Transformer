r"""Registry-backed selection of the model-channel gauge frame."""

from typing import Callable, Dict, Optional, Tuple

import torch

from vfe3.geometry.groups import GaugeGroup
from vfe3.model.positional_phi import apply_positional_phi


ModelFrameBuilder = Callable[..., torch.Tensor]
_MODEL_FRAMES: Dict[str, ModelFrameBuilder] = {}


def register_model_frame(
    name: str,

    *,
    override: bool = False,
) -> Callable[[ModelFrameBuilder], ModelFrameBuilder]:
    r"""Register one model-frame builder; duplicate keys fail closed."""
    def _wrap(fn: ModelFrameBuilder) -> ModelFrameBuilder:
        if name in _MODEL_FRAMES and not override:
            raise KeyError(f"model frame {name!r} already registered; pass override=True to replace")
        _MODEL_FRAMES[name] = fn
        return fn
    return _wrap


def get_model_frame(name: str) -> ModelFrameBuilder:
    r"""Return a registered model-frame builder."""
    if name not in _MODEL_FRAMES:
        raise KeyError(f"no model frame {name!r}; available: {sorted(_MODEL_FRAMES)}")
    return _MODEL_FRAMES[name]


def model_frame_modes() -> Tuple[str, ...]:
    r"""Return the registered model-frame configuration keys."""
    return tuple(sorted(_MODEL_FRAMES))


@register_model_frame("tied")
def _tied_model_frame(
    belief_phi: torch.Tensor,

    **kwargs,
) -> torch.Tensor:
    r"""Reuse the already composed belief frame without allocation or transformation."""
    return belief_phi


@register_model_frame("phi_tilde")
def _independent_model_frame(
    belief_phi:       torch.Tensor,

    *,
    pos_phi_scale:   float                  = 0.02,
    bch_order:       int                    = 4,
    project_slk:     bool                   = False,
    compact_blocks:  bool                   = False,
    pos_phi:         str                    = "none",
    compose_mode:    str                    = "bch",
    model_phi:       Optional[torch.Tensor] = None,
    group:           Optional[GaugeGroup]   = None,
    pos_phi_free:    Optional[torch.Tensor] = None,
    bch_residual_max: Optional[float]        = None,
    **kwargs,
) -> torch.Tensor:
    r"""Compose the independently stored model token and positional frame coordinates."""
    del belief_phi
    if model_phi is None or group is None:
        raise ValueError("phi_tilde requires model_phi coordinates and a gauge group")
    return apply_positional_phi(
        model_phi,
        group,
        mode=pos_phi,
        compose_mode=compose_mode,
        order=bch_order,
        scale=pos_phi_scale,
        project_slk=project_slk,
        compact_blocks=compact_blocks,
        pos_phi_free=pos_phi_free,
        bch_residual_max=bch_residual_max,
    )


def resolve_model_frame(
    belief_phi:       torch.Tensor,

    *,
    mode:             str,

    pos_phi_scale:   float                  = 0.02,
    bch_order:       int                    = 4,
    project_slk:     bool                   = False,
    compact_blocks:  bool                   = False,
    pos_phi:         str                    = "none",
    compose_mode:    str                    = "bch",
    model_phi:       Optional[torch.Tensor] = None,
    group:           Optional[GaugeGroup]   = None,
    pos_phi_free:    Optional[torch.Tensor] = None,
    bch_residual_max: Optional[float]        = None,
) -> torch.Tensor:
    r"""Dispatch model-frame resolution without teaching consumers about frame storage."""
    frame_kwargs = {
        "model_phi": model_phi,
        "group": group,
        "pos_phi_free": pos_phi_free,
        "pos_phi": pos_phi,
        "compose_mode": compose_mode,
        "bch_order": bch_order,
        "pos_phi_scale": pos_phi_scale,
        "project_slk": project_slk,
    }
    if bch_residual_max is not None:
        frame_kwargs["bch_residual_max"] = bch_residual_max
    if compact_blocks:
        frame_kwargs["compact_blocks"] = True
    return get_model_frame(mode)(belief_phi, **frame_kwargs)
