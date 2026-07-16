r"""Tokenized-dataset loading for VFE_3.0 (reads a pre-tokenized cache; no tokenization).

The cache holds 1-D token-id streams under
``{cache_dir}/{dataset}_{split}_{tokenizer}_tokens.{pt|bin}``:
  .pt   a torch int64 tensor (torch.load), e.g. wikitext-103 (gpt2), wiki-ja (cl100k).
  .bin  a raw int32 memmap with an ``n_tokens`` sidecar ``.meta.json``, e.g. wiki-en.
``TokenWindows`` slices the stream into causal-LM (input, target) windows:
``input = tokens[i:i+L]``, ``target = tokens[i+1:i+L+1]``. No neural code, no CLI.
"""

import codecs
import hashlib
import json
import re
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

_CL100K_DATASETS = ("wiki-ja", "wiki-en", "wiki-ar")

# Integer dtypes an on-disk token cache may hold. Uncapped loads keep any of these mapped in their
# native width (no corpus-sized int64 copy); a capped load owns an int64 clone. Used by the cache
# loader and TokenWindows to reject bool / floating / complex caches before they index a table.
SUPPORTED_TOKEN_DTYPES = frozenset(
    {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
)


def default_cache_dir() -> Path:
    """The shared tokenized cache: ``~/.cache/tokenized_cache``."""
    return Path.home() / ".cache" / "tokenized_cache"


def _tokenizer_tag(dataset: str) -> str:
    """Cache tokenizer tag: cl100k for multilingual wiki caches, GPT-2 otherwise."""
    return "tiktoken_cl100k" if dataset in _CL100K_DATASETS else "tiktoken"


_TOKENIZER_VOCAB_SIZE = {"tiktoken": 50257, "tiktoken_cl100k": 100277}
_TIKTOKEN_ENCODING_NAME = {"tiktoken": "gpt2", "tiktoken_cl100k": "cl100k_base"}


def tiktoken_encoding_name(dataset: str) -> str:
    """Return the tiktoken encoding used to build ``dataset``'s cache."""
    return _TIKTOKEN_ENCODING_NAME[_tokenizer_tag(dataset)]


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


def _sha256_file(path: Path) -> str:
    r"""Stream ``path`` in fixed 1 MiB blocks and return its SHA-256 hex digest.

    Reads the file in bounded blocks (never materializing the whole corpus in memory) so hashing a
    multi-hundred-megabyte tokenized cache stays within a fixed working set.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


_CACHE_SOURCE_IDENTITY_MEMO: dict = {}


def _binary_cache_metadata(
    path: Path,
) -> Tuple[Dict[str, object], np.dtype]:
    r"""Read and validate a binary-cache sidecar before any memory mapping.

    A binary token stream is valid only when the sidecar exists, ``n_tokens`` is an exact positive
    integer, the declared dtype is a supported integer token dtype, and the file contains exactly
    ``n_tokens * dtype.itemsize`` bytes. Both truncation and extension fail closed: neither is a
    legitimate alternate view of the declared logical corpus.
    """
    meta_path = Path(str(path) + ".meta.json")
    if not meta_path.is_file():
        raise FileNotFoundError(f"binary token cache metadata not found: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise ValueError(f"binary token cache metadata {meta_path} must be a JSON object")
    n_tokens = meta.get("n_tokens")
    if isinstance(n_tokens, bool) or not isinstance(n_tokens, int) or n_tokens <= 0:
        raise ValueError(
            f"binary token cache metadata {meta_path} n_tokens must be a positive integer")
    try:
        dtype = np.dtype(meta.get("dtype", "int32"))
    except TypeError as exc:
        raise TypeError(
            f"binary token cache metadata {meta_path} has invalid dtype {meta.get('dtype')!r}") from exc
    _require_supported_numpy_token_dtype(dtype, source=path)
    actual_bytes = int(path.stat().st_size)
    expected_bytes = int(n_tokens) * int(dtype.itemsize)
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"binary token cache {path} file bytes={actual_bytes}; expected exactly "
            f"{expected_bytes} from n_tokens={n_tokens} and dtype={dtype}")
    return dict(meta), dtype


def cache_source_identity(
    dataset:   str,
    split:     str = "validation",

    *,
    cache_dir: Optional[Path] = None,
) -> Dict[str, object]:
    r"""Return the tokenizer, format, byte size, and SHA-256 identity of one cache source.

    Resolves the concrete cache file for ``dataset``/``split`` -- the ``.pt`` tensor cache first,
    then the ``.bin`` memmap with its ``.meta.json`` sidecar -- and returns a mapping that binds
    reuse to the EXACT bytes on disk: the resolved cache ``format``, byte ``size_bytes``, the
    streamed ``sha256`` digest, the tokenizer tag, and (for ``.bin``) the parsed sidecar ``meta``
    plus its own ``meta_sha256`` digest, so a corpus edit that leaves the binary untouched but
    changes ``n_tokens`` is still caught. Raises ``FileNotFoundError`` when neither cache exists
    (the caller treats an unhashable source as "forbid reuse").

    Identities are memoized within one process by ``(resolved path, byte size, nanosecond mtime)``
    so a sweep hashes each unchanged corpus once; a file that changes size or mtime lands under a
    new key and is re-hashed, so the memo never serves a stale digest.
    """
    tokenizer_tag = _tokenizer_tag(dataset)
    pt = cache_path(dataset, split, suffix="pt", cache_dir=cache_dir)
    if pt.exists():
        resolved = pt.resolve()
        stat = resolved.stat()
        key = ("pt", str(resolved), stat.st_size, stat.st_mtime_ns,
               stat.st_ctime_ns, getattr(stat, "st_ino", None))
        cached = _CACHE_SOURCE_IDENTITY_MEMO.get(key)
        if cached is not None:
            return dict(cached)
        identity: Dict[str, object] = {
            "format":        "pt",
            "tokenizer_tag": tokenizer_tag,
            "size_bytes":    int(stat.st_size),
            "sha256":        _sha256_file(resolved),
            "meta":          None,
            "meta_sha256":   None,
        }
        _CACHE_SOURCE_IDENTITY_MEMO[key] = dict(identity)
        return identity

    binp = cache_path(dataset, split, suffix="bin", cache_dir=cache_dir)
    if binp.exists():
        resolved = binp.resolve()
        meta, _ = _binary_cache_metadata(resolved)
        stat = resolved.stat()
        meta_path = Path(str(binp) + ".meta.json")
        meta_stat = meta_path.stat()
        key = ("bin", str(resolved), stat.st_size, stat.st_mtime_ns,
               stat.st_ctime_ns, getattr(stat, "st_ino", None),
               meta_stat.st_size, meta_stat.st_mtime_ns, meta_stat.st_ctime_ns,
               getattr(meta_stat, "st_ino", None))
        cached = _CACHE_SOURCE_IDENTITY_MEMO.get(key)
        if cached is not None:
            return dict(cached)
        identity = {
            "format":        "bin",
            "tokenizer_tag": tokenizer_tag,
            "size_bytes":    int(stat.st_size),
            "sha256":        _sha256_file(resolved),
            "meta":          meta,
            "meta_sha256":   _sha256_file(meta_path),
        }
        _CACHE_SOURCE_IDENTITY_MEMO[key] = dict(identity)
        return identity

    raise FileNotFoundError(
        f"no tokenized cache for {dataset!r}/{split!r}: tried {pt} and {binp}"
    )


def _require_supported_token_dtype(
    tokens: torch.Tensor,
    source: Path,
) -> None:
    if tokens.dtype not in SUPPORTED_TOKEN_DTYPES:
        names = ", ".join(sorted(str(dtype) for dtype in SUPPORTED_TOKEN_DTYPES))
        raise TypeError(f"token cache {source} has dtype {tokens.dtype}; expected one of {names}")


def _require_supported_numpy_token_dtype(dtype: np.dtype, source: Path) -> None:
    try:
        torch_dtype = torch.from_numpy(np.empty((0,), dtype=dtype)).dtype
    except TypeError as exc:
        raise TypeError(f"token cache {source} has unsupported dtype {dtype}") from exc
    if torch_dtype not in SUPPORTED_TOKEN_DTYPES:
        names = ", ".join(sorted(str(item) for item in SUPPORTED_TOKEN_DTYPES))
        raise TypeError(f"token cache {source} has dtype {dtype}; expected one of {names}")


def load_cached_tokens(
    dataset:    str,
    split:      str            = "validation",

    *,
    cache_dir:  Optional[Path] = None,
    limit:      Optional[int]  = None,
) -> torch.Tensor:
    """Load the 1-D token-id stream for ``dataset``/``split``.

    Tries the ``.pt`` (torch.load) cache, then the ``.bin`` memmap (size and dtype from the
    ``.meta.json`` sidecar). An uncapped load keeps the corpus mapped in its NATIVE integer
    dtype (any of :data:`SUPPORTED_TOKEN_DTYPES`) -- no corpus-sized int64 copy is allocated;
    the caller (``TokenWindows``) converts to ``torch.long`` before indexing. A ``limit`` yields
    an OWNED ``torch.long`` clone of exactly the first ``limit`` tokens, sliced before any int64
    materialization. The dtype is validated before either branch can slice or cast, so bool /
    floating / complex caches are rejected rather than silently reinterpreted as token ids.
    Raises FileNotFoundError if neither cache is present.
    """
    pt = cache_path(dataset, split, suffix="pt", cache_dir=cache_dir)
    if pt.exists():
        tokens = torch.load(pt, weights_only=True, mmap=True)
        _require_supported_token_dtype(tokens, source=pt)
        if tokens.dim() != 1 or not tokens.is_contiguous():
            raise ValueError(f"token cache {pt} must be a contiguous 1-D tensor")
        if limit is not None:
            return tokens[:limit].clone().to(torch.long)
        return tokens

    binp = cache_path(dataset, split, suffix="bin", cache_dir=cache_dir)
    if binp.exists():
        meta, dtype = _binary_cache_metadata(binp)
        n = int(meta["n_tokens"])
        mm = np.memmap(binp, dtype=dtype, mode="r", shape=(n,))
        if limit is not None:
            capped = torch.from_numpy(np.asarray(mm[:limit]))
            return capped.clone() if capped.dtype == torch.long else capped.to(torch.long)
        tokens = torch.from_numpy(np.asarray(mm))
        _require_supported_token_dtype(tokens, source=binp)
        return tokens

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
        meta, _ = _binary_cache_metadata(binp)
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

    Uses the SAME tokenizer the cache was built with (cl100k for wiki-ja/wiki-en/wiki-ar, GPT-2 otherwise,
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
    enc = tiktoken.get_encoding(tiktoken_encoding_name(dataset))
    return lambda ids: enc.decode([int(t) for t in ids])


def get_tiktoken_byte_decoder(
    dataset: str,
) -> 'Optional[Callable[[Sequence[int]], bytes]]':
    r"""A bounded ``decode_bytes(token_ids)`` callable for the cache tokenizer, or ``None``.

    Byte decoding plus an incremental UTF-8 decoder lets :func:`tokens_per_char` preserve the exact
    whole-stream codepoint semantics even when one Unicode scalar spans tokens on opposite sides of
    a bounded token chunk.
    """
    if dataset == "synthetic-period3":
        return None
    try:
        import tiktoken
    except ImportError:
        return None
    enc = tiktoken.get_encoding(tiktoken_encoding_name(dataset))
    return lambda ids: enc.decode_bytes([int(token) for token in ids])


_TOKENS_PER_CHAR_CACHE: dict = {}


def tokens_per_char(
    dataset:    str,
    split:      str,

    *,
    cache_dir:   Optional[Path] = None,
    chunk_tokens: int          = 64 * 1024,
) -> Optional[float]:
    r"""Corpus constant ``n_tokens / n_unicode_codepoints`` for ``dataset``/``split``, or None.

    The bits-per-character correction factor: ``BPC = (CE / ln 2) * tokens_per_char`` turns the
    model's bits-per-TOKEN into true bits-per-CHARACTER, so PPL/BPC compare across tokenizers and
    languages (gpt2 vs cl100k; English vs Japanese/Arabic, where a token spans ~3 codepoints).
    Computed by decoding the split's cached token stream once with its OWN tokenizer and counting
    Unicode codepoints incrementally in bounded token chunks. The tiktoken path decodes bytes and
    carries UTF-8 state across chunk boundaries, so a codepoint split across two BPE tokens is counted
    exactly once. Returns None when the dataset has no real tokenizer (the synthetic anchor),
    tiktoken is absent, or the cache is missing. Memoization includes the exact cache
    source identity, so replacing bytes or metadata in place cannot serve stale normalization.
    """
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    decode_bytes = get_tiktoken_byte_decoder(dataset)
    decode_text = None if decode_bytes is not None else get_tiktoken_decoder(dataset)
    if decode_bytes is None and decode_text is None:              # synthetic anchor / no tiktoken
        return None
    try:
        source_identity = cache_source_identity(dataset, split, cache_dir=cache_dir)
        tokens = load_cached_tokens(dataset, split, cache_dir=cache_dir)
    except FileNotFoundError:
        return None
    key = (
        dataset,
        split,
        str(cache_dir) if cache_dir is not None else None,
        json.dumps(source_identity, sort_keys=True, separators=(",", ":")),
    )
    if key in _TOKENS_PER_CHAR_CACHE:
        return _TOKENS_PER_CHAR_CACHE[key]
    n_tokens = int(tokens.numel())
    n_chars = 0
    if decode_bytes is not None:
        utf8 = codecs.getincrementaldecoder("utf-8")(errors="replace")
        for start in range(0, n_tokens, chunk_tokens):
            token_chunk = tokens[start:start + chunk_tokens].tolist()
            n_chars += len(utf8.decode(decode_bytes(token_chunk), final=False))
        n_chars += len(utf8.decode(b"", final=True))
    else:
        assert decode_text is not None
        for start in range(0, n_tokens, chunk_tokens):
            token_chunk = tokens[start:start + chunk_tokens].tolist()
            n_chars += len(decode_text(token_chunk))
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
        stride:    Optional[int] = None,   # None -> non-overlapping seq_len stride
        pad_final: bool          = False,  # eval: pad the final partial window, target pad=-100
    ) -> None:
        if tokens.dim() != 1:
            raise ValueError(f"tokens must be 1-D, got shape {tuple(tokens.shape)}")
        if tokens.dtype not in SUPPORTED_TOKEN_DTYPES:
            raise ValueError(f"tokens must have integer dtype, got {tokens.dtype}")
        self.tokens = tokens
        self.seq_len = seq_len
        self.stride = seq_len if stride is None else stride
        self.pad_final = pad_final
        if self.seq_len <= 0 or self.stride <= 0:
            raise ValueError("seq_len and stride must be positive")
        if self.pad_final and self.stride != self.seq_len:
            raise ValueError("pad_final requires stride == seq_len for exactly-once transitions")
        if self.pad_final:
            transitions = self.tokens.numel() - 1
            if transitions <= 0:
                raise ValueError("a padded evaluation stream must contain at least two tokens")
            self.n = (transitions + self.stride - 1) // self.stride
            return
        usable = self.tokens.numel() - seq_len - 1
        if usable < 0:
            raise ValueError(f"stream of {self.tokens.numel()} too short for seq_len={seq_len}")
        self.n = usable // self.stride + 1

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx < 0:
            idx += self.n
        if idx < 0 or idx >= self.n:
            raise IndexError(idx)
        start = idx * self.stride
        end = start + self.seq_len
        window = self.tokens[start:end + 1].to(torch.long)
        if self.pad_final and window.numel() < self.seq_len + 1:
            n_real = max(int(window.numel()) - 1, 0)
            inputs = torch.zeros(self.seq_len, dtype=torch.long)
            targets = torch.full((self.seq_len,), -100, dtype=torch.long)
            if n_real:
                inputs[:n_real] = window[:-1]
                targets[:n_real] = window[1:]
            return inputs, targets
        return window[:-1], window[1:]


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
    source_identity = cache_source_identity(dataset, split, cache_dir=cache_dir)
    tokens = load_cached_tokens(dataset, split, cache_dir=cache_dir, limit=max_tokens)
    if vocab_size is not None:
        validate_token_range(tokens, vocab_size, dataset=dataset)
    ds = TokenWindows(
        tokens,
        seq_len,
        stride=stride,
        pad_final=(not shuffle and not drop_last),
    )
    ds.data_identity = {
        "schema_version":       1,
        "dataset":              dataset,
        "split":                split,
        "tokenizer_tag":        _tokenizer_tag(dataset),
        "tokenizer_encoding":   tiktoken_encoding_name(dataset),
        "tokenizer_vocab_size": tokenizer_vocab_size(dataset),
        "model_vocab_size":     (int(vocab_size) if vocab_size is not None else None),
        "max_tokens":           (int(max_tokens) if max_tokens is not None else None),
        "source":               source_identity,
    }
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
