r"""Bounded H-step candidate-policy menu (audit PB-05).

``policy_mode="efe_rollout"`` (``policy_horizon>1``) is config-valid but was unreachable through
:meth:`VFEModel.generate`: the generic policy path in ``vfe3/model/model.py`` builds only a one-step
``(B, Kp, 1)`` candidate menu, and no H-step candidate generator existed. This module supplies that
generator: :func:`build_topk_policy_menu` is a bounded BEAM search over action tokens, not the
Cartesian top-``width`` product (which would grow as ``width ** horizon``). At most ``width`` live
beams are retained at every depth. The first action reuses the caller's already-computed
``base_logits``; each later depth batches the current ``width`` prefixes through
``model.rollout_beliefs(return_logits=True, decode_last=True)`` once, adds the next-token
log-probability to each beam's accumulated score, and keeps the ``width`` best joint sequences. The
returned prior is ``log_softmax`` of the retained accumulated scores, i.e. the candidate-generator
distribution E the EFE scorer conditions on (spec
``docs/superpowers/specs/2026-06-28-active-inference-efe-policy-scorer-spec.md``); it is NOT a
distribution over the whole vocabulary's H-step continuations, only over the ``width`` retained
beams.
"""

from typing import TYPE_CHECKING, Tuple

import torch

from vfe3.inference.policy import _validate_policy_context

if TYPE_CHECKING:
    from vfe3.model.model import VFEModel


@torch.no_grad()
def build_topk_policy_menu(
    context:     torch.Tensor,             # (B, N) current policy context ids
    base_logits: torch.Tensor,             # (B, V) already-computed base last-position logits

    model:       "VFEModel",               # exposes rollout_beliefs and cfg.max_seq_len

    *,
    horizon: int,                          # H: number of action tokens per candidate policy
    width:   int,                          # beam width; live beam count at every depth <= width
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return candidates (B, width, horizon) and normalized log_prior (B, width)."""
    if context.dim() != 2:
        raise ValueError(f"context must have shape (B, N), got {tuple(context.shape)}")
    B, N = context.shape
    if base_logits.dim() != 2 or base_logits.shape[0] != B:
        raise ValueError(
            f"base_logits must have shape (B, V) with B={B}, got {tuple(base_logits.shape)}"
        )
    vocab_size = base_logits.shape[-1]
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if not 1 <= width <= vocab_size:
        raise ValueError(f"width must be in [1, {vocab_size}], got {width}")
    if not torch.isfinite(base_logits).all():
        raise ValueError("base_logits must be finite")
    _validate_policy_context(context, horizon, model.cfg.max_seq_len)

    first_logp = torch.log_softmax(base_logits, dim=-1)
    beam_score, first_token = first_logp.topk(width, dim=-1)
    candidates = first_token.unsqueeze(-1)

    for depth in range(1, horizon):
        beam_count = candidates.shape[1]
        ctx = context.unsqueeze(1).expand(B, beam_count, N).reshape(B * beam_count, N)
        actions = candidates.reshape(B * beam_count, depth)
        _, decoded = model.rollout_beliefs(
            torch.cat([ctx, actions], dim=-1),
            return_logits=True,
            decode_last=True,
        )
        next_logp = torch.log_softmax(decoded[:, 0, :], dim=-1)
        next_logp = next_logp.reshape(B, beam_count, vocab_size)
        if not torch.isfinite(next_logp).all():
            raise ValueError(f"rollout logits are non-finite at depth {depth + 1}")
        joint = beam_score.unsqueeze(-1) + next_logp
        beam_score, flat_index = joint.reshape(B, -1).topk(width, dim=-1)
        parent = torch.div(flat_index, vocab_size, rounding_mode="floor")
        token = flat_index.remainder(vocab_size)
        kept = torch.gather(
            candidates, 1, parent.unsqueeze(-1).expand(-1, -1, depth)
        )
        candidates = torch.cat([kept, token.unsqueeze(-1)], dim=-1)

    return candidates, torch.log_softmax(beam_score, dim=-1)
