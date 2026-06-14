r"""T5 relative-position bias and windowed (local-attention) attention priors (PL13-priors).

These extend the attention-prior registry beyond uniform/causal/alibi. Each is a pure builder
returning a log-prior bias (Nq, Nk); they are config-selectable through the live _PRIORS
registry (config.py validates beta_attention_prior / gamma_attention_prior against tuple(sorted(_PRIORS))),
so no config-validator edit is needed. Variant params (window, T5 bucketing) use defaults when
invoked through the model's call site, mirroring the existing alibi prior (whose slope is also
default-only from the model); tuning them from config is the separate call-site-threading item.
"""

import math

import torch

from vfe3.attention_prior import (
    _PRIORS,
    _t5_relative_position_bucket,
    attention_log_prior,
)
from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel


def test_new_priors_are_registered_and_config_selectable():
    for name in ("windowed", "causal_windowed", "t5_relative_bias"):
        assert name in _PRIORS
        # config validation accepts the new prior names against the live registry
        VFE3Config(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, beta_attention_prior=name)


def test_windowed_is_a_symmetric_band():
    B = attention_log_prior("windowed", 6, 6, window=2)
    inf = float("-inf")
    for i in range(6):
        for j in range(6):
            if abs(i - j) <= 2:
                assert B[i, j] == 0.0
            else:
                assert B[i, j] == inf


def test_causal_windowed_is_causal_and_banded():
    B = attention_log_prior("causal_windowed", 6, 6, window=2)
    inf = float("-inf")
    for i in range(6):
        for j in range(6):
            if 0 <= i - j <= 2:                     # past, within band
                assert B[i, j] == 0.0
            else:                                   # future OR too far back
                assert B[i, j] == inf


def test_t5_bucket_matches_reference_small_and_large():
    # Non-bidirectional (causal): future -> bucket 0; past distance d<max_exact is exact (bucket=d).
    rel = torch.tensor([[-3, -1, 0, 1, 5]])         # key - query
    buckets = _t5_relative_position_bucket(
        rel, bidirectional=False, num_buckets=32, max_distance=128)
    # rel>0 (future) clamps to past-distance 0 -> bucket 0; rel<=0 -> past distance |rel| (exact, <16)
    assert buckets.tolist() == [[3, 1, 0, 0, 0]]


def test_t5_bucket_is_monotone_nondecreasing_in_past_distance():
    rel = -torch.arange(0, 200).view(1, -1)         # past distances 0..199
    buckets = _t5_relative_position_bucket(
        rel, bidirectional=False, num_buckets=32, max_distance=128).flatten()
    diffs = buckets[1:] - buckets[:-1]
    assert torch.all(diffs >= 0)                    # never decreases as the key recedes into the past
    assert int(buckets.max()) == 31                 # clamped to num_buckets-1


def test_t5_relative_bias_is_causal_by_default():
    B = attention_log_prior("t5_relative_bias", 5, 5)
    inf = float("-inf")
    for i in range(5):
        for j in range(5):
            if j > i:
                assert B[i, j] == inf               # decoder form masks the future
            else:
                assert math.isfinite(B[i, j].item())


def test_t5_relative_bias_uses_supplied_learnable_table():
    # Passing a (num_buckets,) bias_values (a model's learnable handle) gathers per-bucket bias.
    table = torch.linspace(-1.0, -5.0, steps=8)
    B = attention_log_prior("t5_relative_bias", 4, 4, bias_values=table,
                            num_buckets=8, max_distance=16)
    # diagonal is relative distance 0 -> bucket 0 -> table[0]
    for i in range(4):
        assert torch.isclose(B[i, i], table[0])


def _model_forward_finite(prior_name: str):
    cfg = VFE3Config(vocab_size=12, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                     n_e_steps=1, e_phi_lr=0.0, m_phi_lr=0.0, beta_attention_prior=prior_name)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    tokens = torch.randint(0, 12, (2, 8))
    targets = torch.randint(0, 12, (2, 8))
    logits, _, ce = model(tokens, targets)
    assert torch.isfinite(logits).all() and torch.isfinite(ce)


def test_model_forward_under_windowed_priors():
    for name in ("causal_windowed", "windowed", "t5_relative_bias"):
        _model_forward_finite(name)
