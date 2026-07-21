"""Regression coverage for immutable binary-token cache source binding."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import vfe3.data.datasets as datasets


_DATASET = "wiki-en"


def _write_binary_cache(
    root:   Path,
    values: list[int],

    *,
    split:  str = "train",
) -> Path:
    payload = datasets.cache_path(_DATASET, split, suffix="bin", cache_dir=root)
    payload.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(values, dtype=np.int32).tofile(payload)
    Path(str(payload) + ".meta.json").write_text(
        json.dumps({"n_tokens": len(values), "dtype": "int32"}),
        encoding="utf-8",
    )
    return payload


def test_uncapped_binary_cache_is_owned_after_identity_binding(tmp_path: Path) -> None:
    payload = _write_binary_cache(tmp_path, list(range(8)))
    loader = datasets.make_dataloader(
        _DATASET,
        "train",
        3,
        1,
        shuffle=False,
        drop_last=False,
        cache_dir=tmp_path,
    )
    token_windows = loader.dataset
    recorded_identity = token_windows.data_identity["source"]
    original_input, original_target = token_windows[0]

    writable = np.memmap(payload, dtype=np.int32, mode="r+", shape=(8,))
    writable[:4] = np.asarray([70, 71, 72, 73], dtype=np.int32)
    writable.flush()
    del writable

    input_after, target_after = token_windows[0]

    assert token_windows.tokens.dtype == torch.int32
    assert torch.equal(input_after, original_input)
    assert torch.equal(target_after, original_target)
    assert token_windows.data_identity["source"] == recorded_identity


def test_symlinked_binary_cache_uses_one_resolved_sidecar_family(tmp_path: Path) -> None:
    target_payload = _write_binary_cache(tmp_path / "target", list(range(6)))
    link_payload = datasets.cache_path(_DATASET, "train", suffix="bin", cache_dir=tmp_path)
    link_payload.parent.mkdir(parents=True, exist_ok=True)
    try:
        link_payload.symlink_to(target_payload)
    except OSError as exc:
        pytest.skip(f"platform denies symlink creation: {exc}")
    Path(str(link_payload) + ".meta.json").write_text(
        json.dumps({"n_tokens": 12, "dtype": "int16"}),
        encoding="utf-8",
    )

    identity = datasets.cache_source_identity(_DATASET, "train", cache_dir=tmp_path)
    loaded = datasets.load_cached_tokens(_DATASET, "train", cache_dir=tmp_path)
    target_meta = Path(str(target_payload) + ".meta.json")

    assert identity["meta"] == {"n_tokens": 6, "dtype": "int32"}
    assert identity["meta_sha256"] == hashlib.sha256(target_meta.read_bytes()).hexdigest()
    assert loaded.dtype == torch.int32
    assert loaded.tolist() == list(range(6))
