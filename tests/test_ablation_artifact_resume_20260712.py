r"""Artifact-resume integrity for the ablation runner (audit 2026-07-12 PB-01).

Default-on resume must not serve a stale cached cell after a source, tokenizer, corpus, or
semantic-config change. The reuse decision is now bound to a versioned ``cell_contract.json`` that
records the semantic-config fingerprint, dataset, data seed, token cap, tokenizer tag, per-split
corpus identity (byte size + streamed SHA-256), and the repository code identity. ``_cell_is_current``
returns true only for a successful cell whose persisted contract equals the freshly rebuilt one, and
fails closed on an absent, malformed, stale, or semantically incompatible artifact -- in particular a
legacy directory that carries only the old success marker and no contract.

These are pure-fixture tests: no model is built. Tiny temporary ``.pt``/``.bin`` caches stand in for
the corpus, ``_git_code_identity`` is monkeypatched to deterministic mappings, and the run-sweep
isolation tests fake ``run_single`` so a failing cell's per-cell error boundary is exercised without
training.
"""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import ablation
from vfe3.config import VFE3Config
from vfe3.data.datasets import cache_path, cache_source_identity

DATASET = "wikitext-103"                                      # tokenizer tag "tiktoken" -> *_tiktoken_tokens.*
FIXED_CODE_IDENTITY = {"git_sha": "a" * 40, "git_dirty": False, "git_dirty_fingerprint": None}
DIAG_FLAGS = {"collect_diagnostics": False, "collect_extrapolation": False}


def _write_pt_cache(cache_dir: Path, dataset: str, split: str, values) -> Path:
    r"""A tiny int64 ``.pt`` token cache for ``dataset``/``split`` under ``cache_dir``."""
    p = cache_path(dataset, split, suffix="pt", cache_dir=cache_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.tensor(list(values), dtype=torch.long), p)
    return p


def _write_bin_cache(cache_dir: Path, dataset: str, split: str, values) -> Path:
    r"""A tiny int32 ``.bin`` memmap cache plus its ``.meta.json`` sidecar."""
    p = cache_path(dataset, split, suffix="bin", cache_dir=cache_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.array(list(values), dtype=np.int32).tofile(p)
    Path(str(p) + ".meta.json").write_text(
        json.dumps({"n_tokens": len(list(values)), "dtype": "int32"}))
    return p


def _write_marker(run_dir: Path, **updates) -> None:
    r"""A successful cell headline marker (finite terminal PPL) with optional field overrides."""
    marker = {
        "label": "cell", "status": "success", "error_kind": None,
        "primary_val_ppl": 9.0, "final_val_ppl": 10.0, "seed": 6,
    }
    marker.update(updates)
    (run_dir / "ablation_result.json").write_text(json.dumps(marker), encoding="utf-8")


def _setup_cell(tmp_path: Path, monkeypatch, *, write_contract: bool = True):
    r"""A cell directory with tiny corpus caches, a success marker, and (optionally) its contract.

    Returns ``(run_dir, contract)`` where ``contract`` is the freshly built expected contract.
    """
    cache_dir = tmp_path / "cache"
    _write_pt_cache(cache_dir, DATASET, "train", range(10))
    _write_pt_cache(cache_dir, DATASET, "validation", range(5))
    monkeypatch.setattr(ablation, "_git_code_identity", lambda: dict(FIXED_CODE_IDENTITY))
    cfg = VFE3Config(**ablation._cell_cfg_dict({}, seed=6))
    contract = ablation._cell_contract(cfg, DATASET, DIAG_FLAGS, data_seed=3, cache_dir=cache_dir)
    run_dir = tmp_path / "cell"
    run_dir.mkdir()
    _write_marker(run_dir)
    if write_contract:
        (run_dir / "cell_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    return run_dir, contract


def _fake_source_ok(dataset, split, *, cache_dir=None):
    r"""A deterministic per-split source identity that needs no real corpus on disk."""
    return {"format": "pt", "tokenizer_tag": "tiktoken", "size_bytes": len(split),
            "sha256": "0" * 64 + split, "meta": None, "meta_sha256": None}


# =============================================================================
# _cell_is_current: fails closed unless a success marker AND a matching contract exist
# =============================================================================

def test_cell_reuse_requires_contract(tmp_path: Path, monkeypatch) -> None:
    # Legacy directory: old success marker, no cell_contract.json -> fail closed.
    legacy, contract = _setup_cell(tmp_path / "legacy", monkeypatch, write_contract=False)
    assert ablation._cell_is_current(legacy, contract) is False

    # Same directory once the matching contract is published -> reuse authorized.
    current, contract2 = _setup_cell(tmp_path / "current", monkeypatch, write_contract=True)
    assert ablation._cell_is_current(current, contract2) is True

    # A wrong schema version, a non-mapping, and an unparseable contract all fail closed.
    bad_schema = copy.deepcopy(contract2)
    bad_schema["schema_version"] = 999
    (current / "cell_contract.json").write_text(json.dumps(bad_schema), encoding="utf-8")
    assert ablation._cell_is_current(current, contract2) is False
    (current / "cell_contract.json").write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert ablation._cell_is_current(current, contract2) is False
    (current / "cell_contract.json").write_text("{ not json", encoding="utf-8")
    assert ablation._cell_is_current(current, contract2) is False

    # A published contract with a failed / non-finite / missing success marker fails closed.
    (current / "cell_contract.json").write_text(json.dumps(contract2), encoding="utf-8")
    _write_marker(current, status="failed", error_kind="train")
    assert ablation._cell_is_current(current, contract2) is False
    _write_marker(current, final_val_ppl=float("inf"))
    assert ablation._cell_is_current(current, contract2) is False
    (current / "ablation_result.json").unlink()
    assert ablation._cell_is_current(current, contract2) is False


def test_cell_reuse_rejects_code_identity_drift(tmp_path: Path, monkeypatch) -> None:
    run_dir, contract = _setup_cell(tmp_path, monkeypatch)
    assert ablation._cell_is_current(run_dir, contract) is True
    drifted = copy.deepcopy(contract)
    drifted["code_identity"]["git_sha"] = "b" * 40         # HEAD moved / tree changed
    assert ablation._cell_is_current(run_dir, drifted) is False


def test_cell_reuse_rejects_train_or_validation_source_drift(tmp_path: Path, monkeypatch) -> None:
    run_dir, contract = _setup_cell(tmp_path, monkeypatch)
    assert ablation._cell_is_current(run_dir, contract) is True

    train_drift = copy.deepcopy(contract)
    train_drift["train_source"]["sha256"] = "c" * 64
    assert ablation._cell_is_current(run_dir, train_drift) is False

    val_drift = copy.deepcopy(contract)
    val_drift["validation_source"]["sha256"] = "d" * 64
    assert ablation._cell_is_current(run_dir, val_drift) is False


def test_cell_reuse_rejects_semantic_config_drift(tmp_path: Path, monkeypatch) -> None:
    run_dir, contract = _setup_cell(tmp_path, monkeypatch)
    assert ablation._cell_is_current(run_dir, contract) is True
    drifted = copy.deepcopy(contract)
    drifted["semantic_config_fingerprint"] = "e" * 64      # baseline field edited off-label
    assert ablation._cell_is_current(run_dir, drifted) is False


def test_cell_reuse_rejects_max_tokens_or_dataset_drift(tmp_path: Path, monkeypatch) -> None:
    r"""Explicit per-axis regression protection: the loader seams (train token cap, session dataset)
    are contract fields, so a capped smoke cell can never be served for a full run and vice versa."""
    run_dir, contract = _setup_cell(tmp_path, monkeypatch)
    assert ablation._cell_is_current(run_dir, contract) is True
    capped = copy.deepcopy(contract)
    capped["max_tokens"] = 10_000
    assert ablation._cell_is_current(run_dir, capped) is False
    other_dataset = copy.deepcopy(contract)
    other_dataset["dataset"] = "wikitext-2"
    assert ablation._cell_is_current(run_dir, other_dataset) is False


def test_missing_requested_diagnostics_output_forbids_contract_publication(tmp_path: Path, monkeypatch) -> None:
    r"""A collect_diagnostics=True cell whose result carries NO diagnostic output is INCOMPLETE:
    _cell_diagnostics returns {} on wholesale converged_state failure (error_kind stays None and the
    terminal PPL stays finite), so without this gate the empty cell would publish a contract and be
    served as [CACHED] forever. It must instead be converted to a failed result with no contract, so
    the next run recomputes; a sibling cell WITH its requested output publishes and is reusable."""
    monkeypatch.setattr(ablation, "_git_code_identity", lambda: dict(FIXED_CODE_IDENTITY))
    monkeypatch.setattr(ablation, "cache_source_identity", _fake_source_ok)
    sweep_name = "contract_diag_output"
    monkeypatch.setitem(ablation.SWEEPS, sweep_name, {
        "description": "requested-diagnostics completeness gate",
        "collect_diagnostics": True,
        "collect_extrapolation": True,
    })
    monkeypatch.setattr(ablation, "make_run_overrides",
                        lambda _n: [("empty", {}), ("complete", {})])

    def fake_run_single(label, overrides, run_dir, **kwargs):
        result = {"label": label, "error_kind": None, "primary_val_ppl": 8.0,
                  "final_val_ppl": 9.0, "seed": 6}
        if label == "complete":
            result["attn_entropy"] = 1.0                    # requested diagnostics output present
            result["extrap_ce"] = []                        # requested extrapolation output present
        return result

    monkeypatch.setattr(ablation, "run_single", fake_run_single)
    monkeypatch.setattr(ablation, "_cleanup", lambda: None)
    ablation.run_sweep(sweep_name, tmp_path, dataset=DATASET, device=None, seed=6, resume=False)

    flags = {"collect_diagnostics": True, "collect_extrapolation": True}
    expected = ablation._expected_cell_contract_or_none({}, DATASET, flags, seed=6)
    assert expected is not None

    empty_dir = tmp_path / sweep_name / ablation._sanitize("empty")
    empty_marker = json.loads((empty_dir / "ablation_result.json").read_text(encoding="utf-8"))
    assert empty_marker["status"] == "failed"
    assert not (empty_dir / "cell_contract.json").exists()
    assert ablation._cell_is_current(empty_dir, expected) is False

    complete_dir = tmp_path / sweep_name / ablation._sanitize("complete")
    complete_marker = json.loads((complete_dir / "ablation_result.json").read_text(encoding="utf-8"))
    assert complete_marker["status"] == "success"
    assert (complete_dir / "cell_contract.json").exists()
    assert ablation._cell_is_current(complete_dir, expected) is True


# =============================================================================
# cache_source_identity: SHA-256 identity tracks the bytes on disk
# =============================================================================

def test_cache_source_identity_changes_when_cache_bytes_change(tmp_path: Path) -> None:
    _write_pt_cache(tmp_path, DATASET, "train", range(10))
    id1 = cache_source_identity(DATASET, "train", cache_dir=tmp_path)
    assert id1["format"] == "pt"
    assert isinstance(id1["sha256"], str) and id1["sha256"]
    _write_pt_cache(tmp_path, DATASET, "train", range(20))         # different bytes
    id2 = cache_source_identity(DATASET, "train", cache_dir=tmp_path)
    assert id2["sha256"] != id1["sha256"]

    # A .bin source hashes the binary and its sidecar separately; both digests are retained.
    _write_bin_cache(tmp_path, DATASET, "validation", range(8))
    b1 = cache_source_identity(DATASET, "validation", cache_dir=tmp_path)
    assert b1["format"] == "bin"
    assert b1["sha256"] and b1["meta_sha256"]
    _write_bin_cache(tmp_path, DATASET, "validation", range(9))    # binary + sidecar change
    b2 = cache_source_identity(DATASET, "validation", cache_dir=tmp_path)
    assert b2["sha256"] != b1["sha256"]
    assert b2["meta_sha256"] != b1["meta_sha256"]


# =============================================================================
# _expected_cell_contract_or_none: forbids reuse without aborting the sweep
# =============================================================================

def test_invalid_config_contract_forbids_reuse_without_aborting_sweep(tmp_path: Path, monkeypatch) -> None:
    # An unbuildable config yields no contract (reuse forbidden), and no exception escapes.
    assert ablation._expected_cell_contract_or_none(
        {"embed_dim": -4}, DATASET, DIAG_FLAGS, seed=6) is None

    # The isolated per-cell error boundary still holds: a rejected-config cell is recorded as a
    # failed row with no contract, and a following valid cell runs and publishes its contract.
    monkeypatch.setattr(ablation, "_git_code_identity", lambda: dict(FIXED_CODE_IDENTITY))
    monkeypatch.setattr(ablation, "cache_source_identity", _fake_source_ok)
    sweep_name = "contract_invalid"
    monkeypatch.setitem(ablation.SWEEPS, sweep_name, {"description": "invalid-config isolation"})
    monkeypatch.setattr(ablation, "make_run_overrides",
                        lambda _n: [("bad", {"embed_dim": -4}), ("good", {})])

    def fake_run_single(label, overrides, run_dir, **kwargs):
        if label == "bad":
            return {"label": label, "error_kind": "config", "error": "rejected",
                    "primary_val_ppl": float("inf"), "seed": 6}
        return {"label": label, "error_kind": None, "primary_val_ppl": 8.0,
                "final_val_ppl": 9.0, "seed": 6}

    monkeypatch.setattr(ablation, "run_single", fake_run_single)
    monkeypatch.setattr(ablation, "_cleanup", lambda: None)
    ablation.run_sweep(sweep_name, tmp_path, dataset=DATASET, device=None, seed=6, resume=False)

    bad_dir = tmp_path / sweep_name / ablation._sanitize("bad")
    good_dir = tmp_path / sweep_name / ablation._sanitize("good")
    bad_marker = json.loads((bad_dir / "ablation_result.json").read_text(encoding="utf-8"))
    good_marker = json.loads((good_dir / "ablation_result.json").read_text(encoding="utf-8"))
    assert bad_marker["status"] == "failed"
    assert not (bad_dir / "cell_contract.json").exists()
    assert good_marker["status"] == "success"
    assert (good_dir / "cell_contract.json").exists()


def test_missing_or_corrupt_source_contract_forbids_reuse_without_aborting_sweep(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ablation, "_git_code_identity", lambda: dict(FIXED_CODE_IDENTITY))

    # Source hashing failure yields no contract (reuse forbidden), and no exception escapes.
    def raise_source(dataset, split, *, cache_dir=None):
        raise FileNotFoundError("corpus vanished")

    monkeypatch.setattr(ablation, "cache_source_identity", raise_source)
    assert ablation._expected_cell_contract_or_none({}, DATASET, DIAG_FLAGS, seed=6) is None

    # Transient race: the source is missing while one cell's post-run contract is built (that
    # successful cell is converted to a failed result with no contract), then present for the next
    # cell (which publishes its contract). The sweep completes either way.
    state = {"ok": True}

    def stateful_source(dataset, split, *, cache_dir=None):
        if not state["ok"]:
            raise FileNotFoundError("corpus vanished mid-run")
        return _fake_source_ok(dataset, split, cache_dir=cache_dir)

    monkeypatch.setattr(ablation, "cache_source_identity", stateful_source)

    def fake_run_single(label, overrides, run_dir, **kwargs):
        state["ok"] = (label != "race")
        return {"label": label, "error_kind": None, "primary_val_ppl": 8.0,
                "final_val_ppl": 9.0, "seed": 6}

    sweep_name = "contract_source"
    monkeypatch.setitem(ablation.SWEEPS, sweep_name, {"description": "missing-source isolation"})
    monkeypatch.setattr(ablation, "make_run_overrides", lambda _n: [("race", {}), ("good", {})])
    monkeypatch.setattr(ablation, "run_single", fake_run_single)
    monkeypatch.setattr(ablation, "_cleanup", lambda: None)
    ablation.run_sweep(sweep_name, tmp_path, dataset=DATASET, device=None, seed=6, resume=False)

    race_dir = tmp_path / sweep_name / ablation._sanitize("race")
    good_dir = tmp_path / sweep_name / ablation._sanitize("good")
    race_marker = json.loads((race_dir / "ablation_result.json").read_text(encoding="utf-8"))
    good_marker = json.loads((good_dir / "ablation_result.json").read_text(encoding="utf-8"))
    assert race_marker["status"] == "failed"
    assert not (race_dir / "cell_contract.json").exists()
    assert good_marker["status"] == "success"
    assert (good_dir / "cell_contract.json").exists()
