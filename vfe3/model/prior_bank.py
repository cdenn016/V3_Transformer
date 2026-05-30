r"""PriorBank for VFE_3.0: learnable Gaussian vocab priors + the KL decode boundary.

Holds the per-vocabulary prior pi_v = N(mu_v, Sigma_v) with gauge frame phi_v as
PARAMETER TABLES (nn.Parameter -- priors, not neural maps; the no-NN rule bans
nn.Linear/MLP/activations, not learnable parameters). encode(token_ids) looks them
up into the initial belief (q = p); decode(mu_q, sigma_q) scores the posterior
against every prior as logits = -KL(q || pi_v)/tau_eff (the divergence seam),
replacing a linear output projection.

Modularity:
    encode_mode registry -- ``per_token`` (table lookup, default); ``gauge_fixed`` a
        named stub (gauge orbit from a shared base belief).
    decode_mode registry -- ``diagonal`` (fused closed form, default); ``full`` a named
        stub (exact Cholesky for full covariances).
"""

from typing import Callable, Dict, Optional

import torch
from torch import nn

from vfe3.belief import BeliefState


# ---------------------------------------------------------------------------
# Registries: mode name -> callable. Variants swap by config; add a variant by
# writing-and-registering it, never by editing call sites.
#   encode: fn(pb, token_ids) -> BeliefState
#   decode: fn(pb, mu_q, sigma_q, tau_eff) -> logits (B, N, V)
# ---------------------------------------------------------------------------
_ENCODERS: Dict[str, Callable] = {}
_DECODERS: Dict[str, Callable] = {}


def register_encode(name: str) -> Callable:
    """Decorator registering an encode kernel under ``name``."""
    def _wrap(fn: Callable) -> Callable:
        _ENCODERS[name] = fn
        return fn
    return _wrap


def get_encode(name: str) -> Callable:
    """Return the registered encode kernel for ``name`` (KeyError if absent)."""
    if name not in _ENCODERS:
        raise KeyError(
            f"no encode mode registered under {name!r}; available: {sorted(_ENCODERS)}"
        )
    return _ENCODERS[name]


def register_decode(name: str) -> Callable:
    """Decorator registering a decode kernel under ``name``."""
    def _wrap(fn: Callable) -> Callable:
        _DECODERS[name] = fn
        return fn
    return _wrap


def get_decode(name: str) -> Callable:
    """Return the registered decode kernel for ``name`` (KeyError if absent)."""
    if name not in _DECODERS:
        raise KeyError(
            f"no decode mode registered under {name!r}; available: {sorted(_DECODERS)}"
        )
    return _DECODERS[name]


class PriorBank(nn.Module):
    r"""Learnable Gaussian vocab priors; encode (lookup) and decode (-KL/tau_eff).

    The tables ``mu_embed`` (V, K), ``sigma_log_embed`` (V, K), ``phi_embed`` (V, n_gen)
    parameterize the priors pi_v = N(mu_v, exp(sigma_log_v)) with gauge frame phi_v.
    They are PRIORS (nn.Parameter), not a neural map: there is no nn.Linear/MLP/activation
    anywhere in this module. The learnable scalar ``decode_log_scale`` tunes the decode
    temperature.
    """

    def __init__(
        self,
        vocab_size:   int,
        K:            int,
        n_gen:        int,

        *,
        mu_init_std:  float = 0.02,
        sigma_init:   float = 1.0,
        phi_scale:    float = 0.01,
        decode_tau:   float = 1.0,
        eps:          float = 1e-6,
        encode_mode:  str   = "per_token",
        decode_mode:  str   = "diagonal",
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.K = K
        self.n_gen = n_gen
        self.decode_tau = decode_tau
        self.eps = eps
        self.encode_mode = encode_mode
        self.decode_mode = decode_mode

        sigma_log_init = float(torch.log(torch.tensor(sigma_init)))
        self.mu_embed         = nn.Parameter(mu_init_std * torch.randn(vocab_size, K))
        self.sigma_log_embed  = nn.Parameter(torch.full((vocab_size, K), sigma_log_init))
        self.phi_embed        = nn.Parameter(phi_scale * torch.randn(vocab_size, n_gen))
        self.decode_log_scale = nn.Parameter(torch.zeros(1))

    def encode(
        self,
        token_ids: torch.Tensor,         # (B, N) integer token ids
    ) -> BeliefState:
        r"""Look up the per-token Gaussian prior as the initial belief (q = p)."""
        return get_encode(self.encode_mode)(self, token_ids)


@register_encode("per_token")
def _encode_per_token(
    pb:        PriorBank,
    token_ids: torch.Tensor,             # (B, N) integer token ids
) -> BeliefState:
    r"""Per-token table lookup: token_ids -> (mu_v, sigma_v, phi_v) as the belief q = p."""
    mu = pb.mu_embed[token_ids]                                          # (B, N, K)
    sigma = torch.exp(pb.sigma_log_embed[token_ids]).clamp(min=pb.eps)   # (B, N, K), sigma > 0
    phi = pb.phi_embed[token_ids]                                        # (B, N, n_gen)
    return BeliefState(mu=mu, sigma=sigma, phi=phi)


@register_encode("gauge_fixed")
def _encode_gauge_fixed(
    pb:        PriorBank,
    token_ids: torch.Tensor,             # (B, N) integer token ids
) -> BeliefState:
    r"""NAMED STUB: gauge-fixed encode (gauge orbit from a shared base belief).

    Deferred: would realize every prior as a gauge transform of one shared base
    belief, so the vocabulary varies only along the gauge orbit. Not yet implemented.
    """
    raise NotImplementedError(
        "encode_mode='gauge_fixed' is a named stub (gauge orbit from a shared base); "
        "use 'per_token'."
    )
