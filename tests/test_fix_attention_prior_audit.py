r"""Audit fix: a Press-et-al.-faithful `causal_alibi` attention prior.

The `alibi` prior returns a SYMMETRIC bias B_ij = -slope*|i-j| with no causal
mask, so for an autoregressive LM it leaks to future keys (j > i). This pins the
new `causal_alibi` prior that adds the linear distance bias ON TOP OF the causal
mask, and guards that `causal`/`alibi`/`uniform` are unchanged.
"""

import torch

from vfe3.attention_prior import (
    _PRIORS,
    attention_log_prior,
    get_prior,
    prior_causal,
)


def test_causal_alibi_is_registered_and_retrievable():
    # (1) present in the registry and reachable via the getter / dispatcher.
    assert "causal_alibi" in _PRIORS
    assert get_prior("causal_alibi") is _PRIORS["causal_alibi"]
    B = attention_log_prior("causal_alibi", 4, 4, slope=0.5)
    assert B.shape == (4, 4)


def test_causal_alibi_masks_strictly_above_diagonal():
    # (2) -inf strictly above the diagonal (j > i); finite on and below it.
    n = 5
    B = attention_log_prior("causal_alibi", n, n, slope=0.5)
    for i in range(n):
        for j in range(n):
            if j > i:
                assert torch.isneginf(B[i, j]), (i, j, B[i, j])
            else:
                assert torch.isfinite(B[i, j]), (i, j, B[i, j])


def test_causal_alibi_equals_causal_mask_plus_alibi_on_triangle():
    # (3) on the causal triangle (j <= i) value == -slope*(i - j), i.e. the
    # causal mask plus the directional ALiBi linear bias.
    n = 6
    slope = 0.25
    B = attention_log_prior("causal_alibi", n, n, slope=slope)
    mask = prior_causal(n, n)  # 0 on j<=i, -inf on j>i
    for i in range(n):
        for j in range(n):
            if j <= i:
                assert mask[i, j] == 0.0  # masking construction reused verbatim
                expected = -slope * (i - j)
                assert torch.isclose(B[i, j], torch.tensor(expected), atol=1e-6)


def test_causal_alibi_is_not_symmetric():
    # (4) directional bias under the mask -> NOT symmetric, unlike `alibi`.
    n = 4
    B = attention_log_prior("causal_alibi", n, n, slope=0.5)
    assert not torch.equal(B, B.T)
    # Concretely: below-diagonal entry is finite while its transpose is -inf.
    assert torch.isfinite(B[2, 0]) and torch.isneginf(B[0, 2])


def test_default_slope_matches_alibi():
    # Default slope is consistent with `prior_alibi` (1.0): on the diagonal the
    # bias is 0, and one step below the diagonal it is -slope = -1.0.
    n = 3
    B = attention_log_prior("causal_alibi", n, n)
    assert B[1, 1] == 0.0
    assert torch.isclose(B[1, 0], torch.tensor(-1.0), atol=1e-6)


def test_existing_priors_unchanged_regression_guard():
    # (5) regression guard: recompute uniform / causal / alibi and pin behavior.
    assert torch.equal(attention_log_prior("uniform", 4, 4), torch.zeros(4, 4))

    Bc = attention_log_prior("causal", 3, 3)
    assert torch.isneginf(Bc[0, 1]) and torch.isneginf(Bc[0, 2]) and torch.isneginf(Bc[1, 2])
    assert Bc[2, 0] == 0.0 and Bc[1, 1] == 0.0 and Bc[2, 2] == 0.0

    Ba = attention_log_prior("alibi", 4, 4, slope=0.5)
    for i in range(4):
        for j in range(4):
            # alibi stays SYMMETRIC with no causal mask (the leak being documented).
            assert torch.isclose(Ba[i, j], torch.tensor(-0.5 * abs(i - j)), atol=1e-6)
    assert torch.equal(Ba, Ba.T)  # symmetric: distinguishes it from causal_alibi
