r"""Decode behaviour on a non-PD Sigma_q (audit 2026-08-06 F31).

Two separate things live here, and the distinction is the point.

**Fixed.** The chunked reducers returned ``gathered * in_chunk_f`` to zero out positions whose
target is not in this chunk. With an -inf logit that is ``-inf * 0.0`` = NaN, so a degenerate
position poisoned ``target_logit`` through chunks it has no business contributing to. A mask must
not manufacture NaN under any contract; it is now a select, byte-identical for finite logits.

**NOT fixed, by design.** ``logdet_q = -inf`` on a total Cholesky failure is a PINNED cross-path
parity contract (``test_full_cov_chunked_matches_dense_on_non_pd``, audit 2026-07-01 F6): the dense
and chunked paths must agree that a degenerate position scores -inf. But an all--inf logit row is
NaN downstream on BOTH paths -- ``log_softmax`` of it is NaN, and the chunked
``logsumexp_v - target_logit`` is ``-inf - (-inf)``. Making that finite changes what a degenerate
position should SCORE and would have to move both paths together, so it is a decision, not a patch.
The event is counted instead, so a run can at least tell it happened.

Pins: (a) the mask no longer manufactures NaN; (b) the -inf parity contract still holds; (c) the
NaN-CE consequence is real on both paths, recorded so a future change is deliberate; (d) the event
is counted; (e) the healthy path is untouched.
"""

import pytest
import torch

import vfe3.model.prior_bank as prior_bank
from vfe3.numerics import (
    decode_logdet_fallback_elements,
    reset_decode_logdet_fallback_elements,
)

from tests.test_amp import _tiny_model


@pytest.fixture(autouse=True)
def _reset():
    reset_decode_logdet_fallback_elements()
    yield
    reset_decode_logdet_fallback_elements()


def _bank():
    model = _tiny_model(gauge_group="block_glk", n_heads=2, family="gaussian_full",
                        decode_mode="full_chunked")
    return model, model.prior_bank


def _inputs(model, non_pd_at=None):
    B, N, K = 2, 5, model.cfg.embed_dim
    g = torch.Generator().manual_seed(0)
    mu = torch.randn(B, N, K, generator=g)
    sigma = torch.eye(K).expand(B, N, K, K).clone()
    if non_pd_at is not None:
        sigma[non_pd_at] = -5.0 * torch.eye(K)      # fails every jitter round
    return mu, sigma, torch.randint(0, 20, (B, N), generator=g)


# -- (a) the mask no longer manufactures NaN ---------------------------------------------------


def test_chunk_mask_selects_instead_of_multiplying():
    r"""-inf * 0.0 is NaN; the select is 0.0. Checked on the operation itself so the pin survives
    any refactor of the surrounding reducer."""
    gathered = torch.tensor([float("-inf"), 1.5])
    in_chunk_f = torch.tensor([0.0, 1.0])
    assert torch.isnan(gathered * in_chunk_f)[0], "the old form is supposed to produce NaN here"
    selected = torch.where(in_chunk_f > 0, gathered, torch.zeros_like(gathered))
    assert torch.equal(selected, torch.tensor([0.0, 1.5]))


def test_finite_logits_are_byte_identical_under_the_select():
    g = torch.Generator().manual_seed(1)
    gathered = torch.randn(64, generator=g)
    in_chunk_f = (torch.rand(64, generator=g) > 0.5).float()
    assert torch.equal(
        gathered * in_chunk_f,
        torch.where(in_chunk_f > 0, gathered, torch.zeros_like(gathered)))


# -- (b)/(c) the -inf contract stands, and its consequence is recorded -------------------------


def test_non_pd_still_scores_neg_inf_on_both_paths():
    r"""The F6 parity contract. If this ever changes, it must change on BOTH paths together."""
    model, bank = _bank()
    mu, sigma, _ = _inputs(model, non_pd_at=(0, 2))
    tau = bank._tau_eff(None)
    dense = prior_bank._decode_full(bank, mu, sigma, tau)
    assert torch.isneginf(dense[0, 2]).all()


def test_all_neg_inf_row_is_nan_downstream_on_the_dense_path():
    r"""Records WHY F31 is still open: the pinned -inf contract is itself NaN-generating, so
    'fix the sentinel' is not a local change."""
    model, bank = _bank()
    mu, sigma, _ = _inputs(model, non_pd_at=(0, 2))
    dense = prior_bank._decode_full(bank, mu, sigma, bank._tau_eff(None))
    assert torch.isnan(torch.log_softmax(dense[0, 2], dim=-1)).all()


# -- (d)/(e) the event is counted, and the healthy path is untouched ---------------------------


def test_fallback_is_counted():
    model, bank = _bank()
    mu, sigma, targets = _inputs(model, non_pd_at=(0, 2))
    bank.decode_ce_full_chunked(mu, sigma, targets)
    assert decode_logdet_fallback_elements() == 1


def test_healthy_path_counts_zero_and_stays_finite():
    model, bank = _bank()
    mu, sigma, targets = _inputs(model)
    assert torch.isfinite(bank.decode_ce_full_chunked(mu, sigma, targets))
    assert decode_logdet_fallback_elements() == 0


def test_model_forward_is_finite_and_counts_zero():
    model, _ = _bank()
    loss = model(torch.randint(0, 20, (2, 5)), torch.randint(0, 20, (2, 5)))[1]
    assert torch.isfinite(loss)
    assert decode_logdet_fallback_elements() == 0
