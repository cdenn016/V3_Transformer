from __future__ import annotations

import pytest
import torch

import vfe3.geometry.transport as transport
import vfe3.model.prior_bank as prior_bank_module
from vfe3.config import VFE3Config
from vfe3.model.prior_bank import PriorBank


def _small_full_bank(*, promote: bool) -> PriorBank:
    bank = PriorBank(
        3,
        2,
        1,
        mu_init_std=0.0,
        family="gaussian_full",
        decode_mode="full_chunked",
        decode_ranking_fp64_escalation=promote,
    )
    with torch.no_grad():
        bank.mu_embed.copy_(torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]))
        bank.sigma_log_embed.zero_()
    return bank


def test_performance_safety_toggles_default_off_and_require_bool():
    cfg = VFE3Config()
    assert cfg.decode_ranking_fp64_escalation is False
    assert cfg.full_cov_congruence_certification is False
    with pytest.raises(ValueError, match="decode_ranking_fp64_escalation must be .*bool"):
        VFE3Config(decode_ranking_fp64_escalation=1)
    with pytest.raises(ValueError, match="full_cov_congruence_certification must be .*bool"):
        VFE3Config(full_cov_congruence_certification=1)


def test_chunked_decode_does_not_build_ranking_intervals_when_disabled(monkeypatch):
    bank = _small_full_bank(promote=False)

    def _unexpected(*args, **kwargs):
        raise AssertionError("disabled ranking escalation must stay on the streamed fp32 CE path")

    monkeypatch.setattr(prior_bank_module, "_chunk_ranking_summary", _unexpected)
    result = bank.decode_ce_full_chunked(
        torch.zeros(1, 1, 2),
        torch.eye(2).reshape(1, 1, 2, 2),
        torch.tensor([[0]]),
        chunk_size=2,
        return_stats=True,
    )
    assert result.ce.dtype is torch.float32


def test_chunked_decode_keeps_strict_ranking_escalation_as_opt_in(monkeypatch):
    bank = _small_full_bank(promote=True)
    calls = 0
    original = prior_bank_module._chunk_ranking_summary

    def _record(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(prior_bank_module, "_chunk_ranking_summary", _record)
    bank.decode_ce_full_chunked(
        torch.zeros(1, 1, 2),
        torch.eye(2).reshape(1, 1, 2, 2),
        torch.tensor([[0]]),
        chunk_size=2,
    )
    assert calls == 2


def test_full_congruence_skips_certificate_and_fp64_recompute_when_disabled(monkeypatch):
    previous = transport.set_full_cov_congruence_certification(False)
    try:
        def _unexpected(*args, **kwargs):
            raise AssertionError("disabled congruence certification must not factor the pair field")

        monkeypatch.setattr(transport, "validated_cholesky_solve", _unexpected)
        source = torch.eye(2, dtype=torch.float32)
        out = transport._certify_full_congruence(
            source,
            torch.float32,
            lambda dtype: source.to(dtype),
            source_dtype=torch.float32,
        )
        assert out is source
        assert out.dtype is torch.float32
    finally:
        transport.set_full_cov_congruence_certification(previous)


def test_full_congruence_certificate_remains_available_as_opt_in(monkeypatch):
    previous = transport.set_full_cov_congruence_certification(True)
    calls = 0
    original = transport.validated_cholesky_solve
    try:
        def _record(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(transport, "validated_cholesky_solve", _record)
        source = torch.eye(2, dtype=torch.float32)
        out = transport._certify_full_congruence(
            source,
            torch.float32,
            lambda dtype: source.to(dtype),
            source_dtype=torch.float32,
        )
        assert out.dtype is torch.float32
        assert calls == 1
    finally:
        transport.set_full_cov_congruence_certification(previous)
