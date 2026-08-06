r"""Decode log-det fallback on a non-PD Sigma_q (audit 2026-08-06 C5/F31).

``_full_cov_query_invariants`` set ``logdet_q = -inf`` when every ``safe_cholesky`` jitter round
failed. That does not merely give an -inf logit at that position -- it NaNs the scalar CE for the
whole batch, by two independent routes:

  ``per_pos = K + logdet_q`` = -inf  ->  every vocab logit -inf  ->  ``logsumexp_v - target_logit``
  is ``-inf - (-inf)`` = NaN; and ``gathered * in_chunk_f`` is ``-inf * 0.0`` = NaN in every chunk
  that does not contain the target, so NaN enters ``target_logit`` first.

The value is irrelevant to the loss -- ``per_pos`` is v-INDEPENDENT and cancels exactly -- so a
finite fallback costs nothing and the event is counted instead.

Pins: (a) a hopelessly non-PD position yields a FINITE CE, where the old sentinel gave NaN;
(b) the event is counted; (c) the healthy path is untouched and counts zero; (d) the fallback value
genuinely does not move the loss, i.e. the cancellation is real.
"""

import pytest
import torch

from vfe3.families.base import _logdet_chol
from vfe3.numerics import (
    decode_logdet_fallback_elements,
    reset_decode_logdet_fallback_elements,
    safe_cholesky,
)

from tests.test_amp import _tiny_model


@pytest.fixture(autouse=True)
def _reset():
    reset_decode_logdet_fallback_elements()
    yield
    reset_decode_logdet_fallback_elements()


def _bank(**kw):
    model = _tiny_model(gauge_group="block_glk", n_heads=2, family="gaussian_full",
                        decode_mode="full_chunked", **kw)
    return model, model.prior_bank


def _inputs(model, non_pd_at=None):
    B, N, K = 2, 5, model.cfg.embed_dim
    g = torch.Generator().manual_seed(0)
    mu = torch.randn(B, N, K, generator=g)
    sigma = torch.eye(K).expand(B, N, K, K).clone()
    if non_pd_at is not None:
        sigma[non_pd_at] = -5.0 * torch.eye(K)      # fails every jitter round
    targets = torch.randint(0, 20, (B, N), generator=g)
    return mu, sigma, targets


def _old_inf_sentinel(self, sigma_q):
    r"""The pre-fix implementation, for the side-by-side comparison in (a)."""
    diag_sq = torch.diagonal(sigma_q, dim1=-2, dim2=-1)
    factor, ok = safe_cholesky(sigma_q, eps=self.eps, rounds=5)
    logdet = _logdet_chol(factor)
    return diag_sq, torch.where(ok, logdet, logdet.new_full((), float("-inf")))


# -- (a)/(b) the failure mode is gone and is visible ------------------------------------------


def test_non_pd_position_no_longer_nans_the_whole_batch(monkeypatch):
    import vfe3.model.prior_bank as prior_bank

    model, bank = _bank()
    mu, sigma, targets = _inputs(model, non_pd_at=(0, 2))

    ce = bank.decode_ce_full_chunked(mu, sigma, targets)
    assert torch.isfinite(ce), "a single non-PD position must not NaN the batch CE"
    assert decode_logdet_fallback_elements() == 1

    monkeypatch.setattr(
        prior_bank.PriorBank, "_full_cov_query_invariants", _old_inf_sentinel)
    assert torch.isnan(bank.decode_ce_full_chunked(mu, sigma, targets)), (
        "the -inf sentinel is supposed to be the failure this test pins; if it no longer NaNs, "
        "the surrounding code changed and this regression test needs rewriting")


# -- (c) the healthy path is untouched ---------------------------------------------------------


def test_healthy_path_counts_zero_and_stays_finite():
    model, bank = _bank()
    mu, sigma, targets = _inputs(model)
    ce = bank.decode_ce_full_chunked(mu, sigma, targets)
    assert torch.isfinite(ce)
    assert decode_logdet_fallback_elements() == 0


def test_model_forward_is_finite_and_counts_zero():
    model, _ = _bank()
    tokens = torch.randint(0, 20, (2, 5))
    loss = model(tokens, torch.randint(0, 20, (2, 5)))[1]
    assert torch.isfinite(loss)
    assert decode_logdet_fallback_elements() == 0


# -- (d) the fallback value really does cancel -------------------------------------------------


@pytest.mark.parametrize("fallback", [0.0, 12.5, -7.25])
def test_loss_is_independent_of_the_fallback_value(monkeypatch, fallback):
    r"""per_pos is v-independent, so ANY finite fallback gives the same CE. That is why zero is
    free -- and why encoding a 'penalty' in this magnitude would have been meaningless."""
    import vfe3.model.prior_bank as prior_bank

    def patched(self, sigma_q):
        diag_sq = torch.diagonal(sigma_q, dim1=-2, dim2=-1)
        factor, ok = safe_cholesky(sigma_q, eps=self.eps, rounds=5)
        logdet = _logdet_chol(factor)
        return diag_sq, torch.where(ok, logdet, torch.full_like(logdet, fallback))

    model, bank = _bank()
    mu, sigma, targets = _inputs(model, non_pd_at=(0, 2))
    baseline = bank.decode_ce_full_chunked(mu, sigma, targets)

    monkeypatch.setattr(prior_bank.PriorBank, "_full_cov_query_invariants", patched)
    assert torch.allclose(bank.decode_ce_full_chunked(mu, sigma, targets), baseline,
                          atol=1e-5, rtol=1e-5)
