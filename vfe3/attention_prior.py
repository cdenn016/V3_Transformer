r"""Attention-prior registry for VFE_3.0 (the prior pi_ij over keys).

Each prior returns a LOG-PRIOR BIAS B_ij added to the attention logits:
    beta*_ij = softmax_j(B_ij - E_ij / tau),
and the normalized prior used in the attention-entropy term is pi = softmax_j(B).
  uniform  B = 0            -> pi_ij = 1/N (manuscript default).
  causal   B = 0 (j<=i), -inf (j>i)   -> uniform over the causal active set.
  alibi    B_ij = -slope*|i-j|        -> linear distance bias (Press et al.).
  causal_alibi  B_ij = -slope*(i-j) (j<=i), -inf (j>i) -> causal + ALiBi (Press et al.).
Config-selected so a new prior (learned bias, windowed, ...) slots in by
register_prior without editing the free-energy call site.
"""

from typing import Callable, Dict

import torch

_PRIORS: Dict[str, Callable] = {}


def register_prior(name: str) -> Callable:
    """Decorator registering an attention-prior builder -> log-prior bias (Nq, Nk)."""
    def _wrap(fn: Callable) -> Callable:
        _PRIORS[name] = fn
        return fn
    return _wrap


def get_prior(name: str) -> Callable:
    """Return the registered attention-prior builder (KeyError if absent)."""
    if name not in _PRIORS:
        raise KeyError(f"no attention prior {name!r}; available: {sorted(_PRIORS)}")
    return _PRIORS[name]


@register_prior("uniform")
def prior_uniform(
    n_query: int,
    n_key:   int,

    *,
    device:  'torch.device | str | None' = None,
    dtype:   torch.dtype                  = torch.float32,
    **kwargs,
) -> torch.Tensor:
    r"""Uniform prior: zero log-bias (pi_ij = 1/N after softmax)."""
    return torch.zeros(n_query, n_key, device=device, dtype=dtype)


@register_prior("causal")
def prior_causal(
    n_query: int,
    n_key:   int,

    *,
    device:  'torch.device | str | None' = None,
    dtype:   torch.dtype                  = torch.float32,
    **kwargs,
) -> torch.Tensor:
    r"""Causal prior: 0 for key j <= query i, -inf for j > i."""
    i = torch.arange(n_query, device=device).unsqueeze(-1)
    j = torch.arange(n_key, device=device).unsqueeze(0)
    allowed = j <= i
    B = torch.zeros(n_query, n_key, device=device, dtype=dtype)
    return B.masked_fill(~allowed, float("-inf"))


@register_prior("alibi")
def prior_alibi(
    n_query: int,
    n_key:   int,

    *,
    slope:   float                        = 1.0,
    device:  'torch.device | str | None'  = None,
    dtype:   torch.dtype                   = torch.float32,
    **kwargs,
) -> torch.Tensor:
    r"""ALiBi prior: B_ij = -slope * |i - j| (linear distance bias)."""
    i = torch.arange(n_query, device=device).unsqueeze(-1)
    j = torch.arange(n_key, device=device).unsqueeze(0)
    return (-slope * (i - j).abs()).to(dtype)


@register_prior("causal_alibi")
def prior_causal_alibi(
    n_query: int,
    n_key:   int,

    *,
    slope:   float                        = 1.0,
    device:  'torch.device | str | None'  = None,
    dtype:   torch.dtype                   = torch.float32,
    **kwargs,
) -> torch.Tensor:
    r"""Causal + ALiBi prior (Press et al. 2022 faithful, autoregressive form).

    The linear distance bias rides on top of the causal mask:
        B_ij = -slope * (i - j)   for j <= i   (allowed lower triangle),
        B_ij = -inf               for j >  i   (causal mask, identical to `causal`).
    On the causal triangle ``i - j >= 0``, so ``-slope*(i - j) == -slope*|i - j|``:
    the bias magnitude matches ALiBi's on the allowed keys, while ``-inf`` above the
    diagonal forbids attending to future keys. Distinct from the ``alibi`` prior,
    which is the bidirectional/symmetric ``-slope*|i - j|`` with no causal mask.
    """
    i = torch.arange(n_query, device=device).unsqueeze(-1)
    j = torch.arange(n_key, device=device).unsqueeze(0)
    allowed = j <= i
    B = (-slope * (i - j)).to(dtype)
    return B.masked_fill(~allowed, float("-inf"))


def attention_log_prior(
    name:    str,
    n_query: int,
    n_key:   int,

    *,
    device:  'torch.device | str | None'  = None,
    dtype:   torch.dtype                   = torch.float32,
    **kwargs,                              # variant params (e.g. alibi's `slope`)
) -> torch.Tensor:
    r"""Dispatch to the registered attention-prior `name`; returns log-prior bias (Nq, Nk).

    `device`/`dtype` are universal (every builder consumes them); variant-specific
    params (e.g. ALiBi's `slope`) flow through **kwargs, so a new prior with a novel
    param (windowed width, learned-bias handle, ...) slots in by `register_prior` +
    config without editing this dispatcher.
    """
    return get_prior(name)(n_query, n_key, device=device, dtype=dtype, **kwargs)
