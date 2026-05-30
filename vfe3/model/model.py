r"""The full VFE_3.0 model: encode -> E-step inference -> decode -> cross-entropy.

No neural layers: the only parameters are the PriorBank's prior tables. The E-step
is unrolled into the training graph (the differentiable filtering kernel), so the CE
loss backpropagates through inference to the encode/phi priors. Batching loops over
the batch around the (unbatched) E-step; decode and CE are batched.
"""

import inspect
from contextlib import nullcontext
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from vfe3.attention_prior import attention_log_prior
from vfe3.belief import BeliefState
from vfe3.config import VFE3Config
from vfe3.geometry.groups import GaugeGroup, get_group
from vfe3.geometry.norms import get_norm
from vfe3.model.prior_bank import PriorBank
from vfe3.model.stack import vfe_stack


def _positional_arity(builder: Callable) -> int:
    r"""Count the builder's required positional parameters (the K, n_heads, ... axes)."""
    n = 0
    for p in inspect.signature(builder).parameters.values():
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty:
            n += 1
    return n


def build_group(cfg: VFE3Config) -> GaugeGroup:
    r"""Construct the gauge group from config, dispatching on the builder's positional
    arity so a newly registered group slots in by ``register_group`` alone (no call-site
    edit). Arity 1 -> ``builder(K)`` (glk, so_k); arity 2 -> ``builder(K, n_heads)``
    (block_glk). Higher arities are an unsupported registration error."""
    builder = get_group(cfg.gauge_group)
    arity = _positional_arity(builder)
    if arity == 1:
        return builder(cfg.embed_dim)
    if arity == 2:
        return builder(cfg.embed_dim, cfg.n_heads)
    raise ValueError(
        f"gauge group {cfg.gauge_group!r} builder has unsupported positional arity {arity}; "
        f"build_group dispatches K (arity 1) or (K, n_heads) (arity 2)"
    )


class VFEModel(nn.Module):
    """encode -> E-step stack -> decode -> CE. Parameters live only in the PriorBank."""

    def __init__(self, cfg: VFE3Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.group = build_group(cfg)
        n_gen = self.group.generators.shape[0]
        self.prior_bank = PriorBank(
            cfg.vocab_size, cfg.embed_dim, n_gen,
            decode_tau=cfg.decode_tau, eps=cfg.eps,
            diagonal_covariance=cfg.diagonal_covariance,
            encode_mode=cfg.encode_mode, decode_mode=cfg.decode_mode,
        )

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True) -> "VFEModel":
        r"""Carry the gauge group's generators through ``.to(...)`` / ``.cuda()`` etc.

        ``self.group`` is a plain ``GaugeGroup`` dataclass, not an ``nn.Module``, so its
        ``generators`` tensor is outside the parameter/buffer system and would NOT follow a
        dtype/device move -- leaving the E-step transport (belief.phi, which DOES move)
        matmul'd against stale-device/dtype generators. Re-map them here so the module's
        device/dtype contract holds (CLAUDE.md: device-agnostic, float32-with-CUDA)."""
        super()._apply(fn, recurse)
        self.group.generators = fn(self.group.generators)
        return self

    def forward(
        self,
        token_ids: torch.Tensor,         # (B, N) integer token ids
        targets:   Optional[torch.Tensor] = None,   # (B, N) next-token ids (-100 = ignore)
    ) -> 'torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]':
        r"""Forward pass; returns logits, or (logits, loss, ce) when targets are given."""
        B, N = token_ids.shape
        beliefs = self.prior_bank.encode(token_ids)              # (B, N, K) ...
        log_prior = attention_log_prior(
            self.cfg.attention_prior, N, N, device=token_ids.device,
        )

        outs = []
        run = torch.no_grad() if self.cfg.detach_e_step else nullcontext()
        with run:
            for b in range(B):
                belief_b = BeliefState(mu=beliefs.mu[b], sigma=beliefs.sigma[b], phi=beliefs.phi[b])
                out_b = vfe_stack(belief_b, belief_b.mu, belief_b.sigma, self.group, self.cfg, log_prior=log_prior)
                outs.append(out_b)
        mu_final = torch.stack([o.mu for o in outs], dim=0)      # (B, N, K)
        sigma_final = torch.stack([o.sigma for o in outs], dim=0)

        if self.cfg.norm_type_final != "none":                   # config-selected final norm
            norm = get_norm(self.cfg.norm_type_final)(self.cfg.embed_dim, eps=self.cfg.eps)
            mu_final = norm(mu_final, sigma_final)

        logits = self.prior_bank.decode(mu_final, sigma_final)   # (B, N, V)
        if targets is None:
            return logits

        flat_logits = logits.reshape(-1, self.cfg.vocab_size)
        flat_targets = targets.reshape(-1)
        if (flat_targets != -100).any():
            ce = F.cross_entropy(flat_logits, flat_targets, ignore_index=-100)
        else:
            # All-ignore microbatch: F.cross_entropy returns 0/0 = NaN (mean over zero
            # counted tokens), which poisons logging / NaN-guards / grad-accum means. Emit
            # a finite, grad-connected zero instead (a dead-but-clean step).
            ce = flat_logits.sum() * 0.0
        loss = ce
        if self.cfg.mass_phi > 0.0:
            phi_all = torch.stack([o.phi for o in outs], dim=0)
            loss = loss + 0.5 * self.cfg.mass_phi * (phi_all ** 2).mean()
        return logits, loss, ce.detach()
