r"""The full VFE_3.0 model: encode -> E-step inference -> decode -> cross-entropy.

No neural layers: the only parameters are the PriorBank's prior tables. The E-step
is unrolled into the training graph (the differentiable filtering kernel), so the CE
loss backpropagates through inference to the encode/phi priors. Batching loops over
the batch around the (unbatched) E-step; decode and CE are batched.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from vfe3.attention_prior import attention_log_prior
from vfe3.belief import BeliefState
from vfe3.config import VFE3Config
from vfe3.geometry.groups import GaugeGroup, get_group
from vfe3.model.prior_bank import PriorBank
from vfe3.model.stack import vfe_stack


def build_group(cfg: VFE3Config) -> GaugeGroup:
    r"""Construct the gauge group from config (dispatch on the builder signature)."""
    builder = get_group(cfg.gauge_group)
    if cfg.gauge_group == "block_glk":
        return builder(cfg.embed_dim, cfg.n_heads)
    return builder(cfg.embed_dim)


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
            encode_mode=cfg.encode_mode, decode_mode=cfg.decode_mode,
        )

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
        run = torch.no_grad() if self.cfg.detach_e_step else _nullcontext()
        with run:
            for b in range(B):
                belief_b = BeliefState(mu=beliefs.mu[b], sigma=beliefs.sigma[b], phi=beliefs.phi[b])
                out_b = vfe_stack(belief_b, belief_b.mu, belief_b.sigma, self.group, self.cfg, log_prior=log_prior)
                outs.append(out_b)
        mu_final = torch.stack([o.mu for o in outs], dim=0)      # (B, N, K)
        sigma_final = torch.stack([o.sigma for o in outs], dim=0)

        logits = self.prior_bank.decode(mu_final, sigma_final)   # (B, N, V)
        if targets is None:
            return logits

        ce = F.cross_entropy(logits.reshape(-1, self.cfg.vocab_size), targets.reshape(-1), ignore_index=-100)
        loss = ce
        if self.cfg.mass_phi > 0.0:
            phi_all = torch.stack([o.phi for o in outs], dim=0)
            loss = loss + 0.5 * self.cfg.mass_phi * (phi_all ** 2).mean()
        return logits, loss, ce.detach()


class _nullcontext:
    def __enter__(self): return None
    def __exit__(self, *a): return False
