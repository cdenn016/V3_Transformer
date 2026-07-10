r"""Regression tests for the 2026-07-09 audit fixes in vfe3/data/datasets.py (cluster 'data').

Finding 42: load_cached_tokens gains ``limit`` (sliced BEFORE the int64 cast / memmap
materialization, cloned so full-corpus storage is released), make_dataloader threads
``max_tokens`` down as ``limit``, and cached_token_count reads the count from metadata
without materializing the stream.
Finding 44: cache_path validates dataset/split as safe single path components.
Finding 45: tokenizer_vocab_size records the per-tokenizer vocab bound and
validate_token_range (wired into make_dataloader via ``vocab_size``) rejects token ids
that would overrun the model's cfg.vocab_size-sized tables, naming the tokenizer.
"""
import json

import numpy as np
import pytest
import torch

import vfe3.data.datasets as dsmod
from vfe3.data.datasets import (
    cache_path,
    cached_token_count,
    load_cached_tokens,
    make_dataloader,
    tokenizer_vocab_size,
    validate_token_range,
)


# ---------------------------------------------------------------- Finding 42


def _write_pt_cache(tmp_path, dataset="wikitext-103", split="train", n=20):
    p = cache_path(dataset, split, suffix="pt", cache_dir=tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    toks = torch.arange(n, dtype=torch.int64)
    torch.save(toks, p)
    return toks


def _write_bin_cache(tmp_path, dataset="wiki-en", split="test", n=8):
    p = cache_path(dataset, split, suffix="bin", cache_dir=tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    arr = np.arange(n, dtype=np.int32)
    arr.tofile(p)
    (p.parent / (p.name + ".meta.json")).write_text(
        json.dumps({"n_tokens": n, "dtype": "int32"}))
    return arr


def test_load_pt_limit_slices_and_releases_full_storage(tmp_path):
    toks = _write_pt_cache(tmp_path, n=20)
    out = load_cached_tokens("wikitext-103", "train", cache_dir=tmp_path, limit=5)
    assert out.dtype == torch.long and out.shape == (5,)
    assert torch.equal(out, toks[:5])
    # the slice is cloned: its storage holds exactly 5 int64s, not the 20-token corpus
    assert out.untyped_storage().nbytes() == 5 * out.element_size()


def test_load_pt_limit_uses_mmap(tmp_path, monkeypatch):
    _write_pt_cache(tmp_path, n=20)
    real_load = torch.load
    seen = []

    def spy_load(*args, **kwargs):
        seen.append(kwargs.get("mmap"))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(dsmod.torch, "load", spy_load)
    out = load_cached_tokens("wikitext-103", "train", cache_dir=tmp_path, limit=5)
    assert out.shape == (5,)
    assert seen == [True]


def test_load_pt_limit_none_is_unchanged(tmp_path):
    toks = _write_pt_cache(tmp_path, n=20)
    out = load_cached_tokens("wikitext-103", "train", cache_dir=tmp_path)
    assert torch.equal(out, toks)


def test_load_bin_limit_slices_memmap(tmp_path):
    arr = _write_bin_cache(tmp_path, n=8)
    out = load_cached_tokens("wiki-en", "test", cache_dir=tmp_path, limit=4)
    assert out.dtype == torch.long and out.shape == (4,)
    assert torch.equal(out, torch.from_numpy(arr[:4]).to(torch.long))
    # a limit past the end clamps to the stream length
    out_all = load_cached_tokens("wiki-en", "test", cache_dir=tmp_path, limit=999)
    assert out_all.shape == (8,)


def test_cached_token_count_pt(tmp_path):
    _write_pt_cache(tmp_path, n=20)
    assert cached_token_count("wikitext-103", "train", cache_dir=tmp_path) == 20


def test_cached_token_count_bin_reads_metadata_only(tmp_path):
    # an EMPTY .bin file with an n_tokens=7 sidecar: the count must come from the
    # metadata alone (any materialization of the stream would see 0 tokens or fail)
    p = cache_path("wiki-en", "train", suffix="bin", cache_dir=tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch()
    (p.parent / (p.name + ".meta.json")).write_text(
        json.dumps({"n_tokens": 7, "dtype": "int32"}))
    assert cached_token_count("wiki-en", "train", cache_dir=tmp_path) == 7


def test_cached_token_count_missing_cache_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        cached_token_count("wikitext-103", "train", cache_dir=tmp_path)


def test_make_dataloader_threads_max_tokens_as_limit(monkeypatch):
    seen = {}

    def fake_load(dataset, split, *, cache_dir=None, limit=None):
        seen["limit"] = limit
        return torch.arange(30 if limit is None else limit, dtype=torch.long)

    monkeypatch.setattr(dsmod, "load_cached_tokens", fake_load)
    loader = make_dataloader("ds", "train", 4, 2, max_tokens=10)
    assert seen["limit"] == 10                       # cap reached the loader, not a post-hoc slice
    assert loader.dataset.tokens.numel() == 10


# ---------------------------------------------------------------- Finding 44


@pytest.mark.parametrize(
    "bad",
    ["", ".", "..", "a/b", "a\\b", "../evil", "C:evil", "bad name"],
)
def test_cache_path_rejects_unsafe_dataset(bad, tmp_path):
    with pytest.raises(ValueError):
        cache_path(bad, "train", cache_dir=tmp_path)


def test_cache_path_rejects_unsafe_split(tmp_path):
    with pytest.raises(ValueError):
        cache_path("wikitext-103", "../train", cache_dir=tmp_path)


def test_cache_path_accepts_known_names(tmp_path):
    p = cache_path("wikitext-103", "train", cache_dir=tmp_path)
    assert p.name == "wikitext-103_train_tiktoken_tokens.pt"
    assert p.parent == tmp_path


def test_load_cached_tokens_rejects_traversal_before_io(tmp_path):
    with pytest.raises(ValueError):
        load_cached_tokens("../evil", "train", cache_dir=tmp_path)


# ---------------------------------------------------------------- Finding 45


def test_tokenizer_vocab_size_per_dataset():
    assert tokenizer_vocab_size("wikitext-103") == 50257     # gpt2
    assert tokenizer_vocab_size("wiki-en") == 100277         # cl100k
    assert tokenizer_vocab_size("wiki-ja") == 100277


def test_validate_token_range_passes_in_range():
    validate_token_range(torch.tensor([0, 3, 7]), 8, dataset="wikitext-103")


def test_validate_token_range_rejects_overflow_naming_tokenizer():
    with pytest.raises(ValueError) as ei:
        validate_token_range(torch.tensor([0, 100200]), 50257, dataset="wiki-ja")
    msg = str(ei.value)
    assert "wiki-ja" in msg and "tiktoken_cl100k" in msg and "100277" in msg


def test_validate_token_range_rejects_negative_ids():
    with pytest.raises(ValueError):
        validate_token_range(torch.tensor([-1, 2]), 8, dataset="wikitext-103")


def test_make_dataloader_vocab_size_guard(monkeypatch):
    monkeypatch.setattr(dsmod, "load_cached_tokens",
                        lambda *a, **k: torch.arange(30, dtype=torch.long))   # ids 0..29
    with pytest.raises(ValueError):
        make_dataloader("ds", "train", 4, 2, vocab_size=8)
    ok = make_dataloader("ds", "train", 4, 2, vocab_size=64)                 # 29 < 64: fine
    assert len(ok.dataset) > 0
    default = make_dataloader("ds", "train", 4, 2)                           # no guard by default
    assert len(default.dataset) > 0
