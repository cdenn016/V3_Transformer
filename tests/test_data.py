import json

import numpy as np
import pytest
import torch

from vfe3.data.datasets import (
    TokenWindows,
    cache_path,
    default_cache_dir,
    get_tiktoken_decoder,
    load_cached_tokens,
)


def test_token_windows_shift_and_length():
    tokens = torch.arange(20)
    ds = TokenWindows(tokens, seq_len=5, stride=5)
    assert len(ds) == (20 - 5 - 1) // 5 + 1
    x, y = ds[0]
    assert x.shape == (5,) and y.shape == (5,)
    assert torch.equal(x, torch.arange(0, 5))
    assert torch.equal(y, torch.arange(1, 6))                 # target is input shifted by 1


def test_get_tiktoken_decoder_synthetic_is_none():
    # The synthetic anchor has no real tokenizer, so no decoder (the caller emits no sample text).
    assert get_tiktoken_decoder("synthetic-period3") is None


def test_get_tiktoken_decoder_roundtrips_when_tiktoken_present():
    # When tiktoken is installed, the gpt2 decoder round-trips a known id sequence to text;
    # skip cleanly on a box without tiktoken (the decoder is None there, never a crash).
    tiktoken = pytest.importorskip("tiktoken")
    dec = get_tiktoken_decoder("wikitext-103")
    assert dec is not None
    assert dec([15496, 995]) == "Hello world"                  # gpt2 ids for 'Hello world'


def test_token_windows_rejects_short_stream():
    with pytest.raises(ValueError):
        TokenWindows(torch.arange(3), seq_len=10)


def test_load_synthetic_pt_cache(tmp_path):
    # write a synthetic .pt cache under the loader's naming convention and load it
    p = cache_path("wikitext-103", "train", suffix="pt", cache_dir=tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    toks = torch.randint(0, 100, (500,), dtype=torch.int64)
    torch.save(toks, p)
    out = load_cached_tokens("wikitext-103", "train", cache_dir=tmp_path)
    assert out.dtype == torch.long and out.shape == (500,)
    assert torch.equal(out, toks)


def test_load_synthetic_bin_cache(tmp_path):
    # wiki-en-style int32 memmap + meta.json sidecar
    p = cache_path("wiki-en", "test", suffix="bin", cache_dir=tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    arr = np.random.randint(0, 100277, size=(300,), dtype=np.int32)
    arr.tofile(p)
    (p.parent / (p.name + ".meta.json")).write_text(json.dumps({"n_tokens": 300, "dtype": "int32"}))
    out = load_cached_tokens("wiki-en", "test", cache_dir=tmp_path)
    assert out.dtype == torch.long and out.shape == (300,)
    assert out.max() < 100277


def test_load_real_wikitext2_if_present():
    # uses the user's actual cache; skip cleanly if absent
    try:
        toks = load_cached_tokens("wikitext-2", "validation")
    except FileNotFoundError:
        pytest.skip("wikitext-2 validation cache not present")
    assert toks.dim() == 1 and toks.numel() > 1000 and toks.dtype == torch.long
    ds = TokenWindows(toks, seq_len=16)
    x, y = ds[0]
    assert x.shape == (16,) and y.shape == (16,)


def test_make_dataloader_eval_keeps_tail_and_is_sequential(monkeypatch):
    r"""Audit F1: an eval loader (shuffle=False, drop_last=False) must read the WHOLE split in a
    deterministic order, while the train loader (defaults shuffle=True, drop_last=True) drops the
    partial last batch. RED against the old make_dataloader, which had no drop_last param and
    hardcoded drop_last=True for every split."""
    from torch.utils.data import RandomSampler, SequentialSampler

    import vfe3.data.datasets as dsmod

    monkeypatch.setattr(dsmod, "load_cached_tokens", lambda *a, **k: torch.arange(30))
    seq_len, batch_size = 4, 3                       # 30 tokens -> 7 windows; 7 % 3 = 1 (partial tail)

    val = dsmod.make_dataloader("ds", "validation", seq_len, batch_size, shuffle=False, drop_last=False)
    n_full = len(val.dataset)
    assert n_full % batch_size != 0                  # precondition: there IS a partial tail
    assert val.drop_last is False
    assert isinstance(val.sampler, SequentialSampler)
    assert sum(b[0].shape[0] for b in val) == n_full                       # tail kept

    train = dsmod.make_dataloader("ds", "train", seq_len, batch_size)       # defaults: shuffle, drop_last
    assert train.drop_last is True
    assert isinstance(train.sampler, RandomSampler)
    assert sum(b[0].shape[0] for b in train) == n_full - (n_full % batch_size)   # tail dropped
