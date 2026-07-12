# Out-of-Core Corpus Loading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep uncapped token corpora backed by mapped storage and convert only the active causal window or batch to `torch.long`.

**Architecture:** `load_cached_tokens()` preserves the cache's native integer dtype and mapped backing for uncapped loads. `TokenWindows` stores that read-only integer stream and performs one `L+1` conversion inside `__getitem__`, returning overlapping input/target views. Limited smoke-run loads retain their existing cloned int64 behavior.

**Tech Stack:** NumPy memmap, PyTorch mapped serialization, `Dataset`/`DataLoader`, pytest.

## Global Constraints

- Preserve token order, target shift, shuffle generator behavior, split boundaries, and vocabulary validation.
- Preserve the current limited-load contract: `max_tokens` yields an owned int64 clone.
- Uncapped `.bin` and `.pt` loads must not allocate a corpus-sized int64 copy.
- Convert to `torch.long` before any vocabulary-table index or model call.
- Keep tests data-only and small; the performance proof is storage behavior, not a production corpus fixture.
- Update `docs/2026-07-12-edits.md` as the single dated edit note.

---

### Task 1: Preserve mapped native-dtype storage on uncapped loads

**Files:**

- Modify: `vfe3/data/datasets.py:72-105`
- Modify: `tests/test_data.py`
- Modify: `tests/test_fixes_20260709_data.py`

**Interfaces:**

- Preserves: `load_cached_tokens(...) -> torch.Tensor`.
- Adds one module-level `SUPPORTED_TOKEN_DTYPES = frozenset({torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64})`, used by both the cache loader and `TokenWindows`. Uncapped tensors retain any dtype in that set; capped tensors remain owned `torch.int64`.

- [ ] **Step 1: Add failing `.bin` and `.pt` storage tests**

```python
def test_uncapped_bin_stays_int32_and_capped_load_is_owned_long(tmp_path):
    _write_bin_cache(tmp_path, dataset="wiki-en", split="train", n=32)
    mapped = load_cached_tokens("wiki-en", "train", cache_dir=tmp_path)
    capped = load_cached_tokens("wiki-en", "train", cache_dir=tmp_path, limit=8)
    assert mapped.dtype == torch.int32
    assert mapped.numel() == 32
    assert capped.dtype == torch.long
    assert capped.tolist() == list(range(8))
```

Add the `.pt` sibling with an executable load spy:

```python
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
```

Place both tests in `tests/test_fixes_20260709_data.py`, beside its existing cache writers. Extend that file's helper to `def _write_pt_cache(..., n=20, dtype=torch.int64)` and construct `torch.arange(n, dtype=dtype)`. `tests/test_data.py` has no private writer helpers; modify only its legacy uncapped dtype assertions there.

- [ ] **Step 2: Run the tests and observe full int64 materialization**

Run: `python -m pytest tests/test_data.py tests/test_fixes_20260709_data.py -k "uncapped or mapped or load_pt_limit or load_bin_limit" --junitxml=C:\tmp\vfe3-memmap-red.xml`

Expected: the `.bin` tensor is `torch.int64`, and the uncapped `.pt` call records `mmap=False`.

- [ ] **Step 3: Implement native-dtype uncapped loading**

For `.pt`:

```python
tokens = torch.load(pt, weights_only=True, mmap=True)
_require_supported_token_dtype(tokens, source=pt)
if tokens.dim() != 1 or not tokens.is_contiguous():
    raise ValueError(f"token cache {pt} must be a contiguous 1-D tensor")
if limit is not None:
    return tokens[:limit].clone().to(torch.long)
return tokens
```

For `.bin`:

```python
_require_supported_numpy_token_dtype(dtype, source=binp)
mm = np.memmap(binp, dtype=dtype, mode="r", shape=(n,))
if limit is not None:
    capped = torch.from_numpy(np.asarray(mm[:limit]))
    return capped.clone() if capped.dtype == torch.long else capped.to(torch.long)
tokens = torch.from_numpy(np.asarray(mm))
_require_supported_token_dtype(tokens, source=binp)
return tokens
```

Use this one validator before either branch can slice or cast:

```python
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
```

This ordering rejects bool, floating, complex, non-1-D/noncontiguous `.pt`, and unsupported unsigned native caches before a capped load could turn them into apparently valid long IDs or reshape-copy the corpus. Do not call `.to(torch.long)` on the uncapped tensor. Preserve the already pinned capped orders: `.pt` remains `load -> validate -> slice -> clone -> to(long)`, and `.bin` remains `memmap -> slice -> asarray -> from_numpy -> to(long)` for native non-int64 caches; the NumPy metadata check occurs before that event sequence and does not materialize the corpus. A native int64 `.bin` cap uses `clone()` after the same slice/asarray/from_numpy sequence because `.to(long)` would be a no-op retaining the full memmap storage. Do not rewrite `test_load_bin_limit_applied_before_materialization` to move slicing after materialization. Add one accepted int8 `.pt` fixture, rejected capped and uncapped bool/float fixtures, and a capped int64 `.bin` fixture whose storage bytes equal only the cap and whose source mutation cannot change the returned tensor. Update `tests/test_data.py::test_load_synthetic_bin_cache` to expect the uncapped native int32 tensor while retaining exact values; do not weaken capped-long assertions.

- [ ] **Step 4: Re-run the mapped-load tests**

Run: `python -m pytest tests/test_data.py tests/test_fixes_20260709_data.py -k "uncapped or mapped or load_pt_limit or load_bin_limit" --junitxml=C:\tmp\vfe3-memmap-load.xml`

Expected XML: zero failures and zero errors.

- [ ] **Step 5: Commit mapped loading**

```powershell
git add vfe3/data/datasets.py tests/test_data.py tests/test_fixes_20260709_data.py
git commit -m "perf: preserve mapped token storage"
```

### Task 2: Cast one active window instead of the corpus

**Files:**

- Modify: `vfe3/data/datasets.py:217-244`
- Modify: `tests/test_data.py`

**Interfaces:**

- Consumes: a one-dimensional tensor whose dtype is in the same `SUPPORTED_TOKEN_DTYPES` set enforced by the loader.
- Produces: `(input_ids, target_ids)` as `torch.long` tensors of shape `(seq_len,)`.

- [ ] **Step 1: Add native-int32 window equivalence tests**

```python
def test_token_windows_casts_only_the_active_l_plus_one_slice():
    tokens = torch.arange(20, dtype=torch.int32)
    ds = TokenWindows(tokens, seq_len=4, stride=4)
    x, y = ds[1]
    assert x.dtype == y.dtype == torch.long
    assert x.tolist() == [4, 5, 6, 7]
    assert y.tolist() == [5, 6, 7, 8]
    assert ds.tokens.dtype == torch.int32
```

Patch `torch.Tensor.to` only around one item lookup and record the receiver size:

```python
def test_token_windows_never_casts_the_backing_corpus(monkeypatch):
    tokens = torch.arange(100, dtype=torch.int32)
    ds = TokenWindows(tokens, seq_len=8, stride=8)
    original = torch.Tensor.to
    converted_numel = []

    def tracked_to(self, *args, **kwargs):
        converted_numel.append(self.numel())
        return original(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", tracked_to)
    x, y = ds[3]
    assert x.shape == y.shape == (8,)
    assert converted_numel == [9]
```

- [ ] **Step 2: Run the tests and confirm `TokenWindows.__init__` converts the full stream**

Run: `python -m pytest tests/test_data.py -k token_windows_casts --junitxml=C:\tmp\vfe3-window-cast-red.xml`

- [ ] **Step 3: Store native integers and cast one combined window**

```python
if tokens.dtype not in SUPPORTED_TOKEN_DTYPES:
    raise ValueError(f"tokens must have integer dtype, got {tokens.dtype}")
self.tokens = tokens

# __getitem__
window = self.tokens[start:end + 1].to(torch.long)
return window[:-1], window[1:]
```

One conversion preserves the input/target overlap and avoids allocating two independent windows.

- [ ] **Step 4: Run all dataset-window tests**

Run: `python -m pytest tests/test_data.py tests/test_data_generator.py --junitxml=C:\tmp\vfe3-window-cast.xml`

Expected XML: zero failures and zero errors.

- [ ] **Step 5: Commit window-local casting**

```powershell
git add vfe3/data/datasets.py tests/test_data.py
git commit -m "perf: cast token windows on demand"
```

### Task 3: Verify DataLoader, range validation, and device-transfer contracts

**Files:**

- Modify: `tests/test_data.py`
- Modify: `tests/test_train.py`

**Interfaces:**

- Preserves: `make_dataloader(...) -> DataLoader` batches of `torch.long` token IDs.
- Preserves: `validate_token_range()` before model indexing.

- [ ] **Step 1: Add end-to-end loader parity tests**

Create identical `.bin` and `.pt` caches. For `shuffle=False`, collect every batch from both and assert exact equality. For a seeded shuffled loader, assert the same order across two constructions. Test an out-of-range int32 token and require the existing validation error.

- [ ] **Step 2: Add a training-boundary dtype assertion**

Use a tiny `K=2`, vocabulary-eight model and one mapped int32 batch. Assert the loader yields long IDs and one `train_step` completes without an index-dtype conversion inside the model.

- [ ] **Step 3: Run the focused integration slice**

Run: `python -m pytest tests/test_data.py tests/test_fixes_20260709_data.py tests/test_train.py -k "mapped or memmap or token_window" --junitxml=C:\tmp\vfe3-memmap-integration.xml`

Expected XML: zero failures and zero errors.

- [ ] **Step 4: Commit integration tests**

```powershell
git add tests/test_data.py tests/test_train.py
git commit -m "test: pin out-of-core token batches"
```

### Task 4: Measure allocation behavior and document the contract

**Files:**

- Modify: `tests/test_data.py`
- Create: `tests/test_data_memmap_cuda.py`
- Modify: `README.md:1094-1110`
- Modify: `docs/2026-07-12-edits.md`

**Interfaces:**

- Produces: an explicit optional benchmark for mapped-host loading and H2D batch transfer.

- [ ] **Step 1: Add a storage-allocation regression**

Place this CPU regression in `tests/test_data.py`. Create a temporary int32 cache large enough to distinguish 4-byte from 8-byte storage without stressing CI. Assert the returned tensor's dtype and storage byte count are `4 * n_tokens`. Instrument `TokenWindows.__getitem__` and assert each requested item casts exactly `seq_len + 1` values and never the corpus. Separately assert the default-collated input and target batches are long tensors of shape `(batch_size, seq_len)`; document that collation owns `2 * batch_size * seq_len` long values after the per-item casts.

- [ ] **Step 2: Add an RTX 5090 smoke benchmark**

Create only this smoke in `tests/test_data_memmap_cuda.py` and mark it CUDA-only. Build one pinned loader batch, transfer it with `non_blocking=True`, synchronize, and assert exact token equality. Record timing for information only; do not set a brittle speed threshold.

- [ ] **Step 3: Document native mapped storage**

Update the cache documentation to state that uncapped streams retain native mapped integer storage and that `TokenWindows` returns long batches. Record that capped smoke loads remain owned long tensors.

- [ ] **Step 4: Run verification**

Run CPU: `python -m pytest tests/test_data.py tests/test_fixes_20260709_data.py tests/test_data_generator.py tests/test_train.py -k "data or token or memmap" --junitxml=C:\tmp\vfe3-memmap-focused.xml`

Run RTX 5090: `$env:VFE3_TEST_DEVICE='cuda'; python -m pytest tests/test_data_memmap_cuda.py --junitxml=C:\tmp\vfe3-memmap-cuda.xml`

Run full: `python -m pytest -x --junitxml=C:\tmp\vfe3-memmap-full.xml`

Expected for every XML: zero failures and zero errors; report tests and skips from XML.

- [ ] **Step 5: Commit benchmark and documentation**

```powershell
git add tests/test_data.py tests/test_data_memmap_cuda.py README.md docs/2026-07-12-edits.md
git commit -m "docs: record mapped corpus loading"
```

- [ ] **Step 6: Complete repository closeout.** Fetch and inspect `origin/main`, rebase or merge only inside the task worktree if the remote advanced, and rerun the affected verification. Push the task branch, fast-forward it into `main`, push `main`, and fetch again to verify the remote SHA. Fast-forward the user's live checkout only when its WIP cannot be altered; otherwise leave it untouched and report why. Remove the temporary worktree, delete the local task branch, remove task-owned XML/allocation artifacts, and show the final `git worktree list`, `git status --short` for the live checkout, task commit SHA, pushed branch, and resulting `origin/main` SHA.

## Self-review

The plan covers `.bin`, `.pt`, limited and uncapped loads, integer validation, causal-window parity, shuffle determinism, range guards, DataLoader output dtype, CPU storage behavior, and a real-CUDA transfer smoke. The model still receives long token IDs, and no cache or configuration semantics change.
