r"""Tokenized-dataset loading for VFE_3.0 (reads a pre-tokenized cache; no tokenization).

The cache holds 1-D token-id streams under
``{cache_dir}/{dataset}_{split}_{tokenizer}_tokens.{pt|bin}``:
  .pt   a torch int64 tensor (torch.load), e.g. wikitext-103 (gpt2), wiki-ja (cl100k).
  .bin  a raw int32 memmap with an ``n_tokens`` sidecar ``.meta.json``, e.g. wiki-en.
``TokenWindows`` slices the stream into causal-LM (input, target) windows:
``input = tokens[i:i+L]``, ``target = tokens[i+1:i+L+1]``. No neural code, no CLI.
"""

import json
import os
import re
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

_CL100K_DATASETS = ("wiki-ja", "wiki-en")


def default_cache_dir() -> Path:
    """The shared tokenized cache: ``~/.cache/tokenized_cache``."""
    return Path.home() / ".cache" / "tokenized_cache"


def _tokenizer_tag(dataset: str) -> str:
    """Cache tokenizer tag: cl100k for wiki-ja/wiki-en, plain tiktoken (gpt2) otherwise."""
    return "tiktoken_cl100k" if dataset in _CL100K_DATASETS else "tiktoken"


_TOKENIZER_VOCAB_SIZE = {"tiktoken": 50257, "tiktoken_cl100k": 100277}


def tokenizer_vocab_size(dataset: str) -> int:
    """Return the vocabulary bound for the tokenizer used by ``dataset``'s cache."""
    return _TOKENIZER_VOCAB_SIZE[_tokenizer_tag(dataset)]


_SAFE_COMPONENT_RE = re.compile(r"[A-Za-z0-9._-]+")


def _validate_path_component(
    name: str,
    role: str,
) -> None:
    """Reject names that are not safe, non-special single path components."""
    if name in (".", "..") or _SAFE_COMPONENT_RE.fullmatch(name) is None:
        raise ValueError(
            f"{role} {name!r} is not a safe path component: use only letters, digits, "
            f"'.', '_', '-' (no path separators, drive prefixes, or special dot names)"
        )


def cache_path(
    dataset:    str,
    split:      str,

    *,
    suffix:     str            = "pt",
    cache_dir:  Optional[Path] = None,
) -> Path:
    """Build the cache file path for ``dataset``/``split`` with extension ``suffix``."""
    _validate_path_component(dataset, "dataset")
    _validate_path_component(split,   "split")
    root = default_cache_dir() if cache_dir is None else Path(cache_dir)
    return root / f"{dataset}_{split}_{_tokenizer_tag(dataset)}_tokens.{suffix}"


def load_cached_tokens(
    dataset:    str,
    split:      str            = "validation",

    *,
    cache_dir:  Optional[Path] = None,
    limit:      Optional[int]  = None,
) -> torch.Tensor:
    """Load the 1-D token-id stream for ``dataset``/``split`` as an int64 tensor.

    Tries the ``.pt`` (torch.load) cache, then the ``.bin`` int32 memmap (size from the
    ``.meta.json`` sidecar). ``limit`` is applied before int64 materialization. Raises
    FileNotFoundError if neither is present.
    """
    pt = cache_path(dataset, split, suffix="pt", cache_dir=cache_dir)
    if pt.exists():
        tokens = torch.load(pt, weights_only=True, mmap=(limit is not None)).reshape(-1)
        if limit is not None:
            tokens = tokens[:limit].clone()
        return tokens.to(torch.long)

    binp = cache_path(dataset, split, suffix="bin", cache_dir=cache_dir)
    if binp.exists():
        meta = json.loads(Path(str(binp) + ".meta.json").read_text())
        n = int(meta["n_tokens"])
        dtype = np.dtype(meta.get("dtype", "int32"))
        mm = np.memmap(binp, dtype=dtype, mode="r", shape=(n,))
        if limit is not None:
            mm = mm[:limit]
        return torch.from_numpy(np.asarray(mm)).to(torch.long)

    raise FileNotFoundError(
        f"no tokenized cache for {dataset!r}/{split!r}: tried {pt} and {binp}"
    )


def cached_token_count(
    dataset:    str,
    split:      str            = "validation",

    *,
    cache_dir:  Optional[Path] = None,
) -> int:
    """Return the cached token count without materializing the token stream."""
    pt = cache_path(dataset, split, suffix="pt", cache_dir=cache_dir)
    if pt.exists():
        return int(torch.load(pt, weights_only=True, mmap=True).numel())

    binp = cache_path(dataset, split, suffix="bin", cache_dir=cache_dir)
    if binp.exists():
        meta = json.loads(Path(str(binp) + ".meta.json").read_text())
        return int(meta["n_tokens"])

    raise FileNotFoundError(
        f"no tokenized cache for {dataset!r}/{split!r}: tried {pt} and {binp}"
    )


def validate_token_range(
    tokens:     torch.Tensor,        # (T,) 1-D token-id stream

    vocab_size: int,

    *,
    dataset:    str,
) -> None:
    """Raise ValueError unless every token id satisfies ``0 <= id < vocab_size``."""
    if tokens.numel() == 0:
        return
    lo, hi = int(tokens.min()), int(tokens.max())
    if lo < 0:
        raise ValueError(
            f"dataset {dataset!r} (tokenizer {_tokenizer_tag(dataset)!r}) contains negative token ids "
            f"down to {lo}; repair or rebuild the tokenized cache before loading it"
        )
    if hi >= vocab_size:
        required_vocab_size = max(hi + 1, tokenizer_vocab_size(dataset))
        raise ValueError(
            f"dataset {dataset!r} (tokenizer {_tokenizer_tag(dataset)!r}) has token ids in "
            f"[{lo}, {hi}] but vocab_size={vocab_size}: set vocab_size >= {required_vocab_size}"
        )


def get_tiktoken_decoder(
    dataset:   str,
) -> 'Optional[Callable[[Sequence[int]], str]]':
    """A ``decode(token_ids) -> str`` for ``dataset``'s tokenizer, or None if unavailable.

    Uses the SAME tokenizer the cache was built with (cl100k for wiki-ja/wiki-en, gpt2 otherwise,
    matching :func:`_tokenizer_tag`), so generated ids map back to text consistently. Lazy-imports
    tiktoken and returns None when tiktoken is absent or the dataset has no real tokenizer (the
    synthetic period-3 anchor), letting the caller treat None as "no sample text".
    """
    if dataset == "synthetic-period3":
        return None
    try:
        import tiktoken
    except ImportError:
        return None
    enc = tiktoken.get_encoding("cl100k_base" if dataset in _CL100K_DATASETS else "gpt2")
    return lambda ids: enc.decode([int(t) for t in ids])


_TOKENS_PER_CHAR_CACHE: dict = {}


def tokens_per_char(
    dataset:    str,
    split:      str,

    *,
    cache_dir:  Optional[Path] = None,
) -> Optional[float]:
    r"""Corpus constant ``n_tokens / n_unicode_codepoints`` for ``dataset``/``split``, or None.

    The bits-per-character correction factor: ``BPC = (CE / ln 2) * tokens_per_char`` turns the
    model's bits-per-TOKEN into true bits-per-CHARACTER, so PPL/BPC compare across tokenizers and
    languages (gpt2 vs cl100k; English vs Japanese/Arabic, where a token spans ~3 codepoints).
    Computed by decoding the split's cached token stream once with its OWN tokenizer (matching the
    cache via :func:`get_tiktoken_decoder`) and counting Unicode codepoints ``len(text)`` -- the
    character denominator ``n_chars = len(text)``. Returns None when the dataset has no
    real tokenizer (the synthetic anchor), tiktoken is absent, or the cache is missing; the caller
    then leaves ``tokens_per_char = 1.0`` (honest bits-per-token, labelled as such). Memoized per
    (dataset, split, cache_dir) -- intended for the SMALL val/test splits that are scored, a single
    decode pass over the held-out stream.
    """
    key = (dataset, split, str(cache_dir) if cache_dir is not None else None)
    if key in _TOKENS_PER_CHAR_CACHE:
        return _TOKENS_PER_CHAR_CACHE[key]
    decode = get_tiktoken_decoder(dataset)
    if decode is None:                                            # synthetic anchor / no tiktoken
        _TOKENS_PER_CHAR_CACHE[key] = None
        return None
    try:
        tokens = load_cached_tokens(dataset, split, cache_dir=cache_dir)
    except FileNotFoundError:
        _TOKENS_PER_CHAR_CACHE[key] = None
        return None
    n_tokens = int(tokens.numel())
    n_chars = len(decode(tokens.tolist()))                       # Unicode codepoints (lossless BPE round-trip)
    tpc = (n_tokens / n_chars) if n_chars > 0 else None
    _TOKENS_PER_CHAR_CACHE[key] = tpc
    return tpc


class TokenWindows(Dataset):
    """Causal-LM windows over a 1-D token stream: item ``i`` -> (input_L, target_L)."""

    def __init__(
        self,
        tokens:  torch.Tensor,           # (T,) 1-D token-id stream
        seq_len: int,

        *,
        stride:  Optional[int] = None,   # None -> dense (stride 1)
    ) -> None:
        if tokens.dim() != 1:
            raise ValueError(f"tokens must be 1-D, got shape {tuple(tokens.shape)}")
        self.tokens = tokens.to(torch.long)
        self.seq_len = seq_len
        self.stride = seq_len if stride is None else stride
        usable = self.tokens.numel() - seq_len - 1
        if usable < 0:
            raise ValueError(f"stream of {self.tokens.numel()} too short for seq_len={seq_len}")
        self.n = usable // self.stride + 1

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        end = start + self.seq_len
        return self.tokens[start:end], self.tokens[start + 1:end + 1]


def make_dataloader(
    dataset:     str,
    split:       str,
    seq_len:     int,
    batch_size:  int,

    *,
    stride:      Optional[int] = None,
    shuffle:     bool          = True,
    drop_last:   bool          = True,
    cache_dir:   Optional[Path] = None,
    max_tokens:  Optional[int] = None,   # cap the stream (fast smoke runs)
    vocab_size:  Optional[int] = None,   # check token ids fit cfg.vocab_size-sized tables
    generator:   Optional[torch.Generator] = None,   # fix the shuffle order independent of global RNG
) -> DataLoader:
    """Build a DataLoader of causal-LM windows from the cached ``dataset``/``split``.

    ``shuffle`` / ``drop_last`` default to the TRAIN regime (shuffle the stream, drop the partial
    last batch). Evaluation must pass ``shuffle=False, drop_last=False`` so validation/test are a
    stable corpus measurement that reads the WHOLE split (the token-weighted CE in
    ``train.evaluate`` is order-independent, but a dropped tail and a randomly-varying drawn subset
    are not -- see _select_loader). ``max_tokens`` is applied while loading; when supplied,
    ``vocab_size`` rejects cached ids that cannot index the model's vocabulary-sized tables."""
    tokens = load_cached_tokens(dataset, split, cache_dir=cache_dir, limit=max_tokens)
    if vocab_size is not None:
        validate_token_range(tokens, vocab_size, dataset=dataset)
    ds = TokenWindows(tokens, seq_len, stride=stride)
    # pin_memory only when a CUDA device exists (it would pin host pages uselessly on a CPU-only
    # box). With pinned host buffers the per-step .to(device, non_blocking=True) H2D copy in
    # train()/evaluate() can overlap compute; num_workers stays 0 (the dataset is an in-memory
    # tensor slice, so worker IPC would cost more than it saves).
    # generator=None (default) keeps the RandomSampler drawing each epoch permutation from the GLOBAL
    # RNG -- byte-identical to the historic behavior. Passing a seeded generator fixes the shuffle
    # order independent of the global RNG (used by the multi-seed variance floor so the data order is
    # shared across seeds while model-init RNG still varies; see train_vfe3.DATA_SEED / EXP-1).
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                      pin_memory=torch.cuda.is_available(), generator=generator)
