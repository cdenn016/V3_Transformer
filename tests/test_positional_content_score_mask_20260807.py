"""positional_content_score must measure attention, not the attention MASK.

WAVE5 finding. The score is an OLS fit of ``log(beta_ij)`` on the offset ``|i - j|`` over the
causal entries. It selected ``i >= j``, which includes pairs the prior FORBIDS: a masked pair has
logit -inf, so beta is exactly 0, and ``log(clamp(0, 1e-12)) = -27.63``. Those points sit at a
fixed offset and two orders of magnitude below any live weight, so they dominate the total sum of
squares and the "R^2 of attention vs distance" became "R^2 of the mask vs distance".

The decisive test is a PURE POSITIONAL prior: beta built with zero content energy, so the score is
at its attainable ceiling by construction and any large shortfall is the metric's own defect. That
ceiling is 0.9623 at N=64, not 1.0, because of a SECOND and separate defect -- the pooled OLS
ignores each row's own softmax normaliser. It is documented, measured and left unfixed in
``test_score_is_sequence_length_dependent_second_defect_documented`` below, since correcting it
would change what the metric means.
"""

import math

import pytest
import torch

from vfe3.metrics import positional_content_score


def _causal_offsets(n: int) -> torch.Tensor:
    ii = torch.arange(n).unsqueeze(-1)
    jj = torch.arange(n).unsqueeze(0)
    return (ii - jj).to(torch.float64)


def _pure_positional_beta(n: int, slope: float, *, noself: bool) -> torch.Tensor:
    """Row-normalised attention that is EXACTLY log-linear in the offset -> true R^2 == 1."""
    offs = _causal_offsets(n)
    logits = -slope * offs
    allowed = (offs > 0) if noself else (offs >= 0)
    logits = logits.masked_fill(~allowed, float("-inf"))
    beta = torch.softmax(logits, dim=-1)
    return beta.masked_fill(~allowed, 0.0)                       # exact zeros, as the prior emits


# ---------------------------------------------------------------------------------------------
# The defect itself
# ---------------------------------------------------------------------------------------------

def test_pure_positional_prior_with_masked_diagonal_scores_near_one():
    """'causal_alibi_noself' masks the diagonal. The score must ignore it, not fit it."""
    beta = _pure_positional_beta(64, slope=0.0625, noself=True).unsqueeze(0)   # (1, N, N)
    score = positional_content_score(beta)
    assert score.shape == (1,)
    # Pre-fix this read 1.26e-05. The ceiling is 0.9623 rather than 1.0 because of the SEPARATE
    # row-normaliser defect documented in test_score_is_sequence_length_dependent below.
    assert score.item() > 0.90, (
        f"pure positional prior scored {score.item():.6g}; the masked diagonal is being fitted"
    )


def test_masked_diagonal_does_not_change_the_score():
    """The ONLY difference between these two is whether the diagonal is admitted."""
    n, slope = 48, 0.05
    with_diag = _pure_positional_beta(n, slope, noself=False).unsqueeze(0)
    without = _pure_positional_beta(n, slope, noself=True).unsqueeze(0)
    a = positional_content_score(with_diag).item()
    b = positional_content_score(without).item()
    # THE core assertion: masking the diagonal must not move the score. Pre-fix, b collapsed to
    # ~1e-05 while a stayed high -- a 5-order-of-magnitude artefact of the mask alone.
    assert abs(a - b) < 0.05, f"masking the diagonal moved the score: {a:.6g} vs {b:.6g}"


def test_self_attention_weight_is_still_used_when_the_prior_admits_it():
    """The fix must exclude MASKED entries, not blanket-exclude the diagonal.

    A prior that admits self-attention carries a real d=0 weight; dropping it would discard data.
    Here the diagonal is deliberately off the positional trend, so a score that still reads ~1
    would prove the point was silently dropped.
    """
    n = 32
    beta = _pure_positional_beta(n, slope=0.05, noself=False)
    eye = torch.eye(n, dtype=torch.bool)
    beta = beta.masked_fill(eye, 0.0)
    beta = beta + torch.diag(torch.full((n,), 0.5, dtype=beta.dtype))   # large, off-trend, ADMITTED
    beta = beta / beta.sum(dim=-1, keepdim=True)
    score = positional_content_score(beta.unsqueeze(0)).item()
    assert score < 0.95, (
        f"score {score:.6g} is still ~1, so the admitted d=0 weight was dropped rather than fitted"
    )


def test_windowed_prior_masked_tail_is_excluded():
    """A windowed prior masks FAR-off-diagonal pairs -- the same defect at the other end."""
    n, window, slope = 64, 16, 0.0625
    offs = _causal_offsets(n)
    allowed = (offs > 0) & (offs <= window)
    logits = (-slope * offs).masked_fill(~allowed, float("-inf"))
    beta = torch.softmax(logits, dim=-1).masked_fill(~allowed, 0.0)
    # Rows 0 and 1 have <2 admitted keys; the intersected support keeps the fit well posed.
    score = positional_content_score(beta.unsqueeze(0)).item()
    assert score > 0.70, f"windowed pure-positional prior scored {score:.6g}"


# ---------------------------------------------------------------------------------------------
# Behaviour that must not regress
# ---------------------------------------------------------------------------------------------

def test_content_driven_attention_still_scores_low():
    """The metric must retain discriminative power: random content must NOT look positional."""
    torch.manual_seed(0)
    n = 64
    offs = _causal_offsets(n)
    allowed = offs > 0
    logits = torch.randn(2, n, n, dtype=torch.float64) * 3.0
    logits = logits.masked_fill(~allowed, float("-inf"))
    beta = torch.softmax(logits, dim=-1).masked_fill(~allowed, 0.0)
    scores = positional_content_score(beta)
    assert scores.shape == (2,)
    assert (scores < 0.5).all(), f"content-driven attention scored {scores.tolist()}"


def test_per_head_scores_are_independent():
    """Head 0 pure positional, head 1 pure content -> the two must separate."""
    torch.manual_seed(1)
    n = 64
    pos = _pure_positional_beta(n, slope=0.0625, noself=True)
    offs = _causal_offsets(n)
    allowed = offs > 0
    rand = torch.softmax(
        (torch.randn(n, n, dtype=torch.float64) * 3.0).masked_fill(~allowed, float("-inf")), dim=-1
    ).masked_fill(~allowed, 0.0)
    scores = positional_content_score(torch.stack([pos, rand], dim=0))
    assert scores[0].item() > 0.90, f"positional head scored {scores[0].item():.6g}"
    assert scores[1].item() < 0.5, f"content head scored {scores[1].item():.6g}"


def test_batched_leading_dims_preserved():
    beta = _pure_positional_beta(32, slope=0.05, noself=True)
    out = positional_content_score(beta.expand(3, 2, 32, 32))
    assert out.shape == (3, 2)
    assert torch.isfinite(out).all()


def test_degenerate_support_returns_nan_not_a_spurious_fit():
    """Fewer than 2 admitted pairs defines no slope; it must not report a confident number."""
    n = 4
    beta = torch.zeros(1, n, n, dtype=torch.float64)
    beta[0, 1, 0] = 1.0                                          # exactly one admitted pair
    out = positional_content_score(beta)
    assert out.shape == (1,)
    assert math.isnan(out.item()), f"expected NaN on a degenerate support, got {out.item()}"


def test_fp32_and_fp64_agree():
    b64 = _pure_positional_beta(64, slope=0.0625, noself=True).unsqueeze(0)
    a = positional_content_score(b64).item()
    b = positional_content_score(b64.to(torch.float32)).item()
    assert abs(a - b) < 1e-4, f"fp32 {b:.8g} vs fp64 {a:.8g}"


def test_score_is_sequence_length_dependent_second_defect_documented():
    """SEPARATE, UNFIXED defect -- recorded so it is not mistaken for the masking bug.

    The fit pools every row into one OLS, but each row carries its own softmax normaliser, so
    ``log beta_ij = -slope*(i-j) - logZ_i`` is not globally linear and R^2 < 1 even for a
    perfectly positional prior. The shortfall shrinks with N, so an IDENTICAL mechanism scores
    0.41 at N=8 and 0.996 at N=128 -- the score is not comparable across ``max_seq_len``, and
    ablation.py's 'positional_prior' sweep pins max_seq_len=128 against a baseline of 64.

    Removing the per-row normaliser recovers exactly 1.0, which localises the cause. Fixing it
    means a per-row (fixed-effects) intercept, which CHANGES WHAT THE METRIC MEANS -- deliberately
    left for the maintainer.
    """
    scores = {n: positional_content_score(
        _pure_positional_beta(n, slope=0.0625, noself=True).unsqueeze(0)).item()
        for n in (8, 16, 32, 64, 128)}
    assert scores[8] < 0.60 < scores[64], f"expected strong N-dependence, got {scores}"
    assert scores[128] > scores[64] > scores[32] > scores[16] > scores[8], scores

    # Localisation: kill the per-row normaliser spread and the score is exactly 1.
    for n in (8, 64):
        b = _pure_positional_beta(n, slope=0.0625, noself=True)
        b = b / b.clamp(min=1e-30).max(dim=-1, keepdim=True).values
        assert abs(positional_content_score(b.unsqueeze(0)).item() - 1.0) < 1e-9
