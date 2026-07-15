"""Controlled belief-geometry comparison regressions (2026-07-14)."""

from types import SimpleNamespace

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel
from vfe3.viz import extract, report


def _tiny_model(*, model_channel: bool = False) -> VFEModel:
    overrides = {}
    if model_channel:
        overrides = {
            "s_e_step": True,
            "prior_source": "model_channel",
            "lambda_h": 0.25,
            "lambda_gamma": 0.75,
        }
    cfg = VFE3Config(
        vocab_size=20,
        embed_dim=4,
        n_heads=2,
        max_seq_len=4,
        n_layers=1,
        n_e_steps=1,
        e_q_mu_lr=0.1,
        e_phi_lr=0.0,
        **overrides,
    )
    torch.manual_seed(0)
    return VFEModel(cfg)


def _token_batches() -> list[torch.Tensor]:
    return [
        torch.arange(0, 8).reshape(2, 4),
        torch.arange(8, 16).reshape(2, 4),
    ]


def test_belief_bank_max_tokens_slices_every_aligned_field():
    bank = extract.belief_bank(_tiny_model(), _token_batches(), max_tokens=11)

    aligned = ("mu", "sigma", "phi", "token_ids", "seq_idx", "pos_idx")
    assert {bank[key].shape[0] for key in aligned} == {11}
    assert bank["token_ids"].tolist() == list(range(11))
    assert bank["seq_idx"].tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
    assert bank["pos_idx"].tolist() == [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2]


def test_belief_bank_max_sequences_is_exact_and_position_aligned():
    bank = extract.belief_bank(_tiny_model(), _token_batches(), max_sequences=3)

    assert bank["mu"].shape[0] == 12
    assert bank["seq_idx"].tolist() == [0] * 4 + [1] * 4 + [2] * 4
    assert bank["pos_idx"].tolist() == [0, 1, 2, 3] * 3


def test_model_channel_bank_max_tokens_slices_every_aligned_field():
    bank = extract.model_channel_bank(
        _tiny_model(model_channel=True),
        _token_batches(),
        max_tokens=11,
    )

    assert bank is not None
    aligned = ("mu", "sigma", "token_ids", "seq_idx", "pos_idx")
    assert {bank[key].shape[0] for key in aligned} == {11}
    assert bank["token_ids"].tolist() == list(range(11))
    assert bank["pos_idx"].tolist() == [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2]


@pytest.mark.parametrize("bank_name", ["belief_bank", "model_channel_bank"])
def test_banks_reject_ambiguous_population_caps(bank_name):
    model = _tiny_model(model_channel=bank_name == "model_channel_bank")
    bank_fn = getattr(extract, bank_name)

    with pytest.raises(ValueError, match="max_tokens.*max_sequences"):
        bank_fn(model, _token_batches(), max_tokens=8, max_sequences=2)


@pytest.mark.parametrize("cap_name", ["max_tokens", "max_sequences"])
@pytest.mark.parametrize("cap_value", [0, -1])
def test_belief_bank_rejects_nonpositive_population_caps(cap_name, cap_value):
    with pytest.raises(ValueError, match=cap_name):
        extract.belief_bank(_tiny_model(), _token_batches(), **{cap_name: cap_value})


@pytest.mark.parametrize(
    ("seq_len", "batch_size", "expected_batches"),
    [(128, 64, 2), (256, 32, 2), (512, 16, 2)],
)
def test_report_default_requests_same_controlled_token_population(
    seq_len,
    batch_size,
    expected_batches,
):
    cfg = SimpleNamespace(max_seq_len=seq_len, batch_size=batch_size)

    max_tokens, max_sequences, n_batches = report._resolve_bank_budget(
        cfg,
        max_tokens=None,
        max_sequences=None,
    )

    assert max_tokens == 16_384
    assert max_sequences is None
    assert n_batches == expected_batches
