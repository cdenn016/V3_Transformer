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
import ast
import gc
import json
from pathlib import Path

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


def _write_pt_cache(tmp_path, dataset="wikitext-103", split="train", n=20, dtype=torch.int64):
    p = cache_path(dataset, split, suffix="pt", cache_dir=tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    toks = torch.arange(n, dtype=dtype)
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


# ---------------------------------------------------------------- PB-08: mapped native-dtype storage


def test_uncapped_bin_stays_int32_and_capped_load_is_owned_long(tmp_path):
    _write_bin_cache(tmp_path, dataset="wiki-en", split="train", n=32)
    mapped = load_cached_tokens("wiki-en", "train", cache_dir=tmp_path)
    capped = load_cached_tokens("wiki-en", "train", cache_dir=tmp_path, limit=8)
    assert mapped.dtype == torch.int32
    assert mapped.numel() == 32
    assert capped.dtype == torch.long
    assert capped.tolist() == list(range(8))


def test_uncapped_pt_requests_mmap_and_preserves_native_dtype(tmp_path, monkeypatch):
    _write_pt_cache(tmp_path, dataset="wiki-en", split="train", n=32, dtype=torch.int32)
    original = torch.load
    mmap_values = []

    def tracked_load(*args, **kwargs):
        mmap_values.append(kwargs.get("mmap"))
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "load", tracked_load)
    mapped = load_cached_tokens("wiki-en", "train", cache_dir=tmp_path)
    assert mapped.dtype == torch.int32
    assert mmap_values == [True]


def test_uncapped_pt_int8_is_accepted_and_native(tmp_path):
    # a supported narrow integer dtype (int8) is mapped through uncapped without a widening copy,
    # and a capped load of the same cache still returns an owned int64 stream.
    _write_pt_cache(tmp_path, dataset="wikitext-103", split="train", n=16, dtype=torch.int8)
    mapped = load_cached_tokens("wikitext-103", "train", cache_dir=tmp_path)
    assert mapped.dtype == torch.int8
    assert mapped.tolist() == list(range(16))
    capped = load_cached_tokens("wikitext-103", "train", cache_dir=tmp_path, limit=4)
    assert capped.dtype == torch.long
    assert capped.tolist() == list(range(4))


@pytest.mark.parametrize("limit", [None, 4])
@pytest.mark.parametrize("dtype", [torch.bool, torch.float32])
def test_pt_unsupported_dtype_is_rejected_capped_and_uncapped(tmp_path, dtype, limit):
    # bool / floating .pt caches are rejected before any slice or cast, so a capped load can never
    # turn them into apparently-valid long ids (validation precedes slice/cast).
    p = cache_path("wikitext-103", "train", suffix="pt", cache_dir=tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    toks = torch.zeros(8, dtype=torch.bool) if dtype is torch.bool else torch.arange(8, dtype=dtype)
    torch.save(toks, p)
    with pytest.raises(TypeError):
        load_cached_tokens("wikitext-103", "train", cache_dir=tmp_path, limit=limit)


@pytest.mark.parametrize("limit", [None, 4])
def test_bin_unsupported_dtype_is_rejected(tmp_path, limit):
    # the NumPy dtype metadata is checked before np.memmap opens the corpus, so a float .bin cache
    # is rejected (capped or uncapped) without materializing the stream.
    p = cache_path("wiki-en", "test", suffix="bin", cache_dir=tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.arange(8, dtype=np.float32).tofile(p)
    (p.parent / (p.name + ".meta.json")).write_text(
        json.dumps({"n_tokens": 8, "dtype": "float32"}))
    with pytest.raises(TypeError):
        load_cached_tokens("wiki-en", "test", cache_dir=tmp_path, limit=limit)


def test_capped_int64_bin_is_owned_and_source_mutation_cannot_change_it(tmp_path):
    # a native int64 .bin cap clones the slice (a .to(long) no-op would keep the whole memmap): the
    # returned tensor holds only the cap's bytes and is immune to a later edit of the backing file.
    p = cache_path("wiki-en", "train", suffix="bin", cache_dir=tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.arange(32, dtype=np.int64).tofile(p)
    (p.parent / (p.name + ".meta.json")).write_text(
        json.dumps({"n_tokens": 32, "dtype": "int64"}))
    capped = load_cached_tokens("wiki-en", "train", cache_dir=tmp_path, limit=8)
    assert capped.dtype == torch.long
    assert capped.tolist() == list(range(8))
    # owned storage: exactly the cap, not the 32-token corpus
    assert capped.untyped_storage().nbytes() == 8 * capped.element_size()
    before = capped.clone()
    gc.collect()                                         # release the memmap handle (Windows) before rewrite
    np.arange(100, 132, dtype=np.int64).tofile(p)        # overwrite the corpus on disk
    assert torch.equal(capped, before)


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


def test_load_pt_limit_slices_before_int64_materialization(tmp_path, monkeypatch):
    pt = cache_path("wikitext-103", "train", suffix="pt", cache_dir=tmp_path)
    pt.parent.mkdir(parents=True, exist_ok=True)
    pt.touch()
    events = []

    class _LoadProbe:
        # a supported native dtype and a contiguous 1-D shape so the pre-slice validation passes;
        # the capped order is load -> validate (dtype/dim/contiguity) -> slice -> clone -> to(long).
        dtype = torch.int64

        def dim(self):
            return 1

        def is_contiguous(self):
            return True

        def __getitem__(self, key):
            assert key == slice(None, 5)
            events.append("slice")
            return self

        def clone(self):
            events.append("clone")
            return self

        def to(self, dtype):
            assert dtype is torch.long
            events.append("to-int64")
            return torch.arange(5, dtype=torch.long)

    seen = []

    def fake_load(*_args, **kwargs):
        seen.append(kwargs.get("mmap"))
        return _LoadProbe()

    monkeypatch.setattr(dsmod.torch, "load", fake_load)

    out = load_cached_tokens("wikitext-103", "train", cache_dir=tmp_path, limit=5)

    assert torch.equal(out, torch.arange(5, dtype=torch.long))
    assert seen == [True]
    assert events == ["slice", "clone", "to-int64"]


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


def test_load_bin_limit_slices_before_int64_materialization(tmp_path, monkeypatch):
    _write_bin_cache(tmp_path, n=8)
    events = []

    class _MemmapProbe:
        def __getitem__(self, key):
            assert key == slice(None, 4)
            events.append("slice")
            return self

    probe = _MemmapProbe()

    def fake_memmap(*_args, **_kwargs):
        events.append("memmap")
        return probe

    def fake_asarray(value):
        assert value is probe
        events.append("asarray")
        return np.arange(4, dtype=np.int32)

    monkeypatch.setattr(dsmod.np, "memmap", fake_memmap)
    monkeypatch.setattr(dsmod.np, "asarray", fake_asarray)

    out = load_cached_tokens("wiki-en", "test", cache_dir=tmp_path, limit=4)

    assert torch.equal(out, torch.arange(4, dtype=torch.long))
    assert events == ["memmap", "slice", "asarray"]


def test_cached_token_count_pt_uses_mmap(tmp_path, monkeypatch):
    _write_pt_cache(tmp_path, n=20)
    real_load = torch.load
    seen = []

    def spy_load(*args, **kwargs):
        seen.append(kwargs.get("mmap"))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(dsmod.torch, "load", spy_load)

    assert cached_token_count("wikitext-103", "train", cache_dir=tmp_path) == 20
    assert seen == [True]


def test_cached_token_count_bin_validates_metadata_without_mapping(tmp_path, monkeypatch):
    # Count retrieval validates exact bytes against metadata but never constructs a memmap.
    _write_bin_cache(tmp_path, dataset="wiki-en", split="train", n=7)
    monkeypatch.setattr(dsmod.np, "memmap",
                        lambda *a, **k: pytest.fail("token count must not map the corpus"))
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
    monkeypatch.setattr(dsmod, "cache_source_identity", lambda *a, **k: {
        "format": "pt", "tokenizer_tag": "tiktoken", "size_bytes": 30,
        "sha256": "0" * 64, "meta": None, "meta_sha256": None,
    })
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
    assert tokenizer_vocab_size("wiki-ar") == 100277


def test_validate_token_range_passes_in_range():
    validate_token_range(torch.tensor([0, 3, 7]), 8, dataset="wikitext-103")


def test_validate_token_range_rejects_overflow_naming_tokenizer():
    with pytest.raises(ValueError) as ei:
        validate_token_range(torch.tensor([0, 100200]), 50257, dataset="wiki-ja")
    msg = str(ei.value)
    assert "wiki-ja" in msg and "tiktoken_cl100k" in msg and "100277" in msg


def test_validate_token_range_rejects_negative_ids():
    with pytest.raises(ValueError) as ei:
        validate_token_range(torch.tensor([-1, 2]), 8, dataset="wikitext-103")
    msg = str(ei.value)
    assert "negative token id" in msg
    assert "repair or rebuild" in msg
    assert "set vocab_size" not in msg


@pytest.mark.parametrize(
    ("source_path", "callee", "expected_value_kind"),
    [
        ("train_vfe3.py",        "make_dataloader", "active-config"),
        ("sigma_gate_measure.py", "make_dataloader", "active-config"),
        ("vfe3/train.py",         "make_dataloader", "active-config"),
        ("vfe3/viz/report.py",    "make_dataloader", "active-config"),
        ("ablation.py",           "make_dataloader", "forwarded"),
        ("scaling.py",            "make_dataloader", "forwarded"),
        ("ablation.py",           "get_loader",      "active-config"),
        ("scaling.py",            "get_loader",      "active-config"),
    ],
)
def test_production_loader_calls_keep_vocab_size_guard(
    source_path,
    callee,
    expected_value_kind,
):
    tree = ast.parse(Path(source_path).read_text(encoding="utf-8"), filename=source_path)
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            isinstance(node.func, ast.Name) and node.func.id == callee
            or isinstance(node.func, ast.Attribute) and node.func.attr == callee
        )
    ]
    assert calls, (source_path, callee)
    for call in calls:
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        assert "vocab_size" in keywords, (source_path, callee)
        value = keywords["vocab_size"]
        if expected_value_kind == "forwarded":
            assert isinstance(value, ast.Name) and value.id == "vocab_size", source_path
        else:
            assert isinstance(value, ast.Attribute) and value.attr == "vocab_size", source_path


def test_make_dataloader_vocab_size_guard(monkeypatch):
    monkeypatch.setattr(dsmod, "load_cached_tokens",
                        lambda *a, **k: torch.arange(30, dtype=torch.long))   # ids 0..29
    monkeypatch.setattr(dsmod, "cache_source_identity", lambda *a, **k: {
        "format": "pt", "tokenizer_tag": "tiktoken", "size_bytes": 30,
        "sha256": "0" * 64, "meta": None, "meta_sha256": None,
    })
    with pytest.raises(ValueError):
        make_dataloader("ds", "train", 4, 2, vocab_size=8)
    ok = make_dataloader("ds", "train", 4, 2, vocab_size=64)                 # 29 < 64: fine
    assert len(ok.dataset) > 0
    default = make_dataloader("ds", "train", 4, 2)                           # no guard by default
    assert len(default.dataset) > 0
