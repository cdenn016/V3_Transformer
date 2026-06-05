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


def cache_path(
    dataset:    str,
    split:      str,

    *,
    suffix:     str            = "pt",
    cache_dir:  Optional[Path] = None,
) -> Path:
    """Build the cache file path for ``dataset``/``split`` with extension ``suffix``."""
    root = default_cache_dir() if cache_dir is None else Path(cache_dir)
    return root / f"{dataset}_{split}_{_tokenizer_tag(dataset)}_tokens.{suffix}"


def load_cached_tokens(
    dataset:    str,
    split:      str            = "validation",

    *,
    cache_dir:  Optional[Path] = None,
) -> torch.Tensor:
    """Load the 1-D token-id stream for ``dataset``/``split`` as an int64 tensor.

    Tries the ``.pt`` (torch.load) cache, then the ``.bin`` int32 memmap (size from the
    ``.meta.json`` sidecar). Raises FileNotFoundError if neither is present.
    """
    pt = cache_path(dataset, split, suffix="pt", cache_dir=cache_dir)
    if pt.exists():
        tokens = torch.load(pt, weights_only=True)
        return tokens.to(torch.long).reshape(-1)

    binp = cache_path(dataset, split, suffix="bin", cache_dir=cache_dir)
    if binp.exists():
        meta = json.loads(Path(str(binp) + ".meta.json").read_text())
        n = int(meta["n_tokens"])
        dtype = np.dtype(meta.get("dtype", "int32"))
        mm = np.memmap(binp, dtype=dtype, mode="r", shape=(n,))
        return torch.from_numpy(np.asarray(mm)).to(torch.long)

    raise FileNotFoundError(
        f"no tokenized cache for {dataset!r}/{split!r}: tried {pt} and {binp}"
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
) -> DataLoader:
    """Build a DataLoader of causal-LM windows from the cached ``dataset``/``split``.

    ``shuffle`` / ``drop_last`` default to the TRAIN regime (shuffle the stream, drop the partial
    last batch). Evaluation must pass ``shuffle=False, drop_last=False`` so validation/test are a
    stable corpus measurement that reads the WHOLE split (the token-weighted CE in
    ``train.evaluate`` is order-independent, but a dropped tail and a randomly-varying drawn subset
    are not -- see _select_loader)."""
    tokens = load_cached_tokens(dataset, split, cache_dir=cache_dir)
    if max_tokens is not None:
        tokens = tokens[:max_tokens]
    ds = TokenWindows(tokens, seq_len, stride=stride)
    # pin_memory only when a CUDA device exists (it would pin host pages uselessly on a CPU-only
    # box). With pinned host buffers the per-step .to(device, non_blocking=True) H2D copy in
    # train()/evaluate() can overlap compute; num_workers stays 0 (the dataset is an in-memory
    # tensor slice, so worker IPC would cost more than it saves).
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                      pin_memory=torch.cuda.is_available())
