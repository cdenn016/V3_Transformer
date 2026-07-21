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
import csv
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
DIAG_FLAGS = {
    "collect_diagnostics": False,
    "collect_extrapolation": False,
    "paired_token_bootstrap": False,
}


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
    checkpoint = run_dir / "checkpoints" / "terminal.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint.exists():
        checkpoint.write_bytes(b"owned terminal checkpoint")
    marker = {
        "label": "cell", "status": "success", "error_kind": None,
        "primary_val_ppl": 9.0, "final_val_ppl": 10.0, "seed": 6,
        "terminal_checkpoint": str(checkpoint),
        **DIAG_FLAGS,
    }
    identity = ablation._terminal_checkpoint_identity(run_dir, marker)
    assert identity is not None
    marker.update(identity)
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
    _write_marker(
        run_dir,
        cell_contract_fingerprint=ablation.semantic_config_fingerprint(contract),
    )
    if write_contract:
        (run_dir / "cell_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    return run_dir, contract


def _fake_source_ok(dataset, split, *, cache_dir=None):
    r"""A deterministic per-split source identity that needs no real corpus on disk."""
    return {"format": "pt", "tokenizer_tag": "tiktoken", "size_bytes": len(split),
            "sha256": "0" * 64 + split, "meta": None, "meta_sha256": None}


def _fake_loaded_sources():
    """The detached source contract a successful fake run_single must return to run_sweep."""
    return {
        split: _fake_source_ok(DATASET, split)
        for split in ("train", "validation")
    }


def _owned_terminal_checkpoint(run_dir: Path) -> str:
    """Create the production-required terminal checkpoint for one fake successful cell."""
    checkpoint = run_dir / "checkpoints" / "terminal.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"owned terminal checkpoint")
    return str(checkpoint)


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


def test_cell_reuse_rejects_new_contract_paired_with_old_success_marker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir, old_contract = _setup_cell(tmp_path, monkeypatch)
    new_contract = copy.deepcopy(old_contract)
    new_contract["semantic_config_fingerprint"] = "f" * 64
    (run_dir / "cell_contract.json").write_text(json.dumps(new_contract), encoding="utf-8")

    assert ablation._cell_is_current(run_dir, new_contract) is False


@pytest.mark.parametrize("field", sorted(DIAG_FLAGS))
def test_cell_reuse_rejects_missing_marker_diagnostic_flag(
    field: str,
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir, contract = _setup_cell(tmp_path, monkeypatch)
    marker_path = run_dir / "ablation_result.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    del marker[field]
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    assert ablation._cell_is_current(run_dir, contract) is False


@pytest.mark.parametrize(
    ("field", "raw"),
    (
        ("collect_diagnostics", "false"),
        ("collect_extrapolation", 0),
        ("paired_token_bootstrap", 1),
    ),
)
def test_cell_reuse_rejects_non_boolean_marker_diagnostic_flag(
    field: str,
    raw: object,
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir, contract = _setup_cell(tmp_path, monkeypatch)
    marker_path = run_dir / "ablation_result.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker[field] = raw
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    assert ablation._cell_is_current(run_dir, contract) is False


@pytest.mark.parametrize("field", sorted(DIAG_FLAGS))
def test_cell_reuse_rejects_marker_diagnostic_flag_disagreement(
    field: str,
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir, contract = _setup_cell(tmp_path, monkeypatch)
    marker_path = run_dir / "ablation_result.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker[field] = not DIAG_FLAGS[field]
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    assert ablation._cell_is_current(run_dir, contract) is False


@pytest.mark.parametrize("raw", ([], [1, 1], [True], [1.5], ["1"], [-1]))
def test_declared_sweep_seeds_require_nonempty_unique_exact_nonnegative_integers(raw) -> None:
    with pytest.raises(ValueError, match="seeds"):
        ablation._validated_sweep_seeds({"seeds": raw}, 6)


def test_recompute_invalidates_prior_success_marker_before_training(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sweep_name = "generation_invalidation"
    monkeypatch.setitem(ablation.SWEEPS, sweep_name, {"description": "generation guard"})
    monkeypatch.setattr(ablation, "make_run_overrides", lambda _name: [("cell", {})])
    contract = {"schema_version": ablation._CELL_CONTRACT_SCHEMA_VERSION, "generation": "new"}
    monkeypatch.setattr(
        ablation, "_expected_cell_contract_or_none", lambda *args, **kwargs: contract)
    monkeypatch.setattr(ablation, "_cell_is_current", lambda *args, **kwargs: False)

    run_dir = tmp_path / sweep_name / ablation._sanitize("cell")
    run_dir.mkdir(parents=True)
    _write_marker(
        run_dir,
        cell_contract_fingerprint=ablation.semantic_config_fingerprint(contract),
    )

    def interrupted_run(*args, **kwargs):
        del args, kwargs
        marker = json.loads((run_dir / "ablation_result.json").read_text(encoding="utf-8"))
        assert marker["status"] == "running"
        raise KeyboardInterrupt

    monkeypatch.setattr(ablation, "run_single", interrupted_run)
    with pytest.raises(KeyboardInterrupt):
        ablation.run_sweep(
            sweep_name,
            tmp_path,
            dataset=DATASET,
            device=None,
            seed=6,
            resume=True,
        )

    marker = json.loads((run_dir / "ablation_result.json").read_text(encoding="utf-8"))
    assert marker["status"] == "running"


def test_resumed_stale_cell_rejects_contract_drift_during_recompute(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sweep_name = "resumed_contract_drift"
    monkeypatch.setitem(ablation.SWEEPS, sweep_name, {"description": "contract drift guard"})
    monkeypatch.setattr(ablation, "make_run_overrides", lambda _name: [("cell", {})])
    before = {
        "schema_version": ablation._CELL_CONTRACT_SCHEMA_VERSION,
        "code_identity": {"git_sha": "a" * 40},
    }
    after = copy.deepcopy(before)
    after["code_identity"]["git_sha"] = "b" * 40
    builds = []

    def rebuild_contract(*args, **kwargs):
        del args, kwargs
        builds.append(None)
        return copy.deepcopy(before if len(builds) == 1 else after)

    monkeypatch.setattr(ablation, "_expected_cell_contract_or_none", rebuild_contract)
    monkeypatch.setattr(ablation, "_cell_is_current", lambda *args, **kwargs: False)
    monkeypatch.setattr(ablation, "_cleanup", lambda: None)

    def fake_run_single(label, overrides, run_dir, **kwargs):
        del label, overrides, kwargs
        return {
            "label": "cell",
            "error_kind": None,
            "primary_val_ppl": 8.0,
            "final_val_ppl": 9.0,
            "seed": 6,
            "terminal_checkpoint": _owned_terminal_checkpoint(run_dir),
            "_loaded_data_sources": _fake_loaded_sources(),
        }

    monkeypatch.setattr(ablation, "run_single", fake_run_single)

    run_dir = tmp_path / sweep_name / ablation._sanitize("cell")
    run_dir.mkdir(parents=True)
    _write_marker(run_dir, cell_contract_fingerprint="stale")
    stale_contract = {
        "schema_version": ablation._CELL_CONTRACT_SCHEMA_VERSION,
        "generation": "stale",
    }
    (run_dir / "cell_contract.json").write_text(json.dumps(stale_contract), encoding="utf-8")

    result = ablation.run_sweep(
        sweep_name,
        tmp_path,
        dataset=DATASET,
        device=None,
        seed=6,
        resume=True,
    )

    marker = json.loads((run_dir / "ablation_result.json").read_text(encoding="utf-8"))
    assert len(builds) == 2
    assert marker["status"] == "failed"
    assert marker["cell_contract_fingerprint"] is None
    assert not (run_dir / "cell_contract.json").exists()
    assert result == []


@pytest.mark.parametrize("resume", (False, True))
def test_recomputation_without_marker_rejects_contract_drift(
    tmp_path: Path,
    monkeypatch,
    resume: bool,
) -> None:
    sweep_name = f"fresh_contract_drift_{resume}"
    monkeypatch.setitem(ablation.SWEEPS, sweep_name, {"description": "contract drift guard"})
    monkeypatch.setattr(ablation, "make_run_overrides", lambda _name: [("cell", {})])
    before = {
        "schema_version": ablation._CELL_CONTRACT_SCHEMA_VERSION,
        "code_identity": {"git_sha": "a" * 40},
    }
    after = copy.deepcopy(before)
    after["code_identity"]["git_sha"] = "b" * 40
    builds = []

    def rebuild_contract(*args, **kwargs):
        del args, kwargs
        builds.append(None)
        return copy.deepcopy(before if len(builds) == 1 else after)

    monkeypatch.setattr(ablation, "_expected_cell_contract_or_none", rebuild_contract)
    monkeypatch.setattr(ablation, "_cleanup", lambda: None)

    def fake_run_single(label, overrides, run_dir, **kwargs):
        del label, overrides, kwargs
        return {
            "label": "cell",
            "error_kind": None,
            "primary_val_ppl": 8.0,
            "final_val_ppl": 9.0,
            "seed": 6,
            "terminal_checkpoint": _owned_terminal_checkpoint(run_dir),
            "_loaded_data_sources": _fake_loaded_sources(),
        }

    monkeypatch.setattr(ablation, "run_single", fake_run_single)

    result = ablation.run_sweep(
        sweep_name,
        tmp_path,
        dataset=DATASET,
        device=None,
        seed=6,
        resume=resume,
    )

    run_dir = tmp_path / sweep_name / ablation._sanitize("cell")
    marker = json.loads((run_dir / "ablation_result.json").read_text(encoding="utf-8"))
    assert len(builds) == 2
    assert marker["status"] == "failed"
    assert marker["cell_contract_fingerprint"] is None
    assert not (run_dir / "cell_contract.json").exists()
    assert result == []


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
        "extrapolation_lengths": [16, 32],
        "mandatory_extrapolation_lengths": [16, 32],
    })
    monkeypatch.setattr(ablation, "make_run_overrides",
                        lambda _n: [("empty", {}), ("complete", {})])

    def fake_run_single(label, overrides, run_dir, **kwargs):
        result = {"label": label, "error_kind": None, "primary_val_ppl": 8.0,
                  "final_val_ppl": 9.0, "seed": 6,
                  "terminal_checkpoint": _owned_terminal_checkpoint(run_dir),
                  "_loaded_data_sources": _fake_loaded_sources()}
        if label == "complete":
            result.update({                                 # every requested diagnostic is present
                "attn_entropy": 1.0,
                "energy_klmax_frac": 0.1,
                "gauge_resid_in": 1e-7,
                "gauge_resid_out": 1e-7,
                "omega_identity_dev": 0.2,
                "rank_resid": 0.8,
            })
            result["extrap_ce"] = [                         # requested extrapolation output present
                {"n": 16, "status": "success", "ce": 3.0, "ppl": 20.0,
                 "effective_batch_size": 4},
                {"n": 32, "status": "success", "ce": 3.1, "ppl": 22.2,
                 "effective_batch_size": 2},
            ]
        return result

    monkeypatch.setattr(ablation, "run_single", fake_run_single)
    monkeypatch.setattr(ablation, "_cleanup", lambda: None)
    ablation.run_sweep(
        sweep_name,
        tmp_path,
        dataset=DATASET,
        device=torch.device("cpu"),
        seed=6,
        resume=False,
    )

    # The contract's diagnostic_flags now also binds paired_token_bootstrap (PB-07); this sweep does
    # not request it, so the published contract records it false and the rebuilt expected must match.
    flags = {"collect_diagnostics": True, "collect_extrapolation": True,
             "paired_token_bootstrap": False}
    expected = ablation._expected_cell_contract_or_none(
        {}, DATASET, flags, seed=6,
        requested_extrapolation_lengths=(16, 32),
        mandatory_extrapolation_lengths=(16, 32),
    )
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
                "final_val_ppl": 9.0, "seed": 6,
                "terminal_checkpoint": _owned_terminal_checkpoint(run_dir),
                "_loaded_data_sources": _fake_loaded_sources()}

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

    # The shared pre-sweep filesystem snapshot is available. A run whose loaders cannot return the
    # identity of the data they actually loaded still fails closed, while the next run whose loaders
    # return the matching identity publishes normally. No per-cell filesystem re-hash is needed.
    monkeypatch.setattr(ablation, "cache_source_identity", _fake_source_ok)

    def fake_run_single(label, overrides, run_dir, **kwargs):
        return {"label": label, "error_kind": None, "primary_val_ppl": 8.0,
                "final_val_ppl": 9.0, "seed": 6,
                "terminal_checkpoint": _owned_terminal_checkpoint(run_dir),
                "_loaded_data_sources": (
                    None if label == "race" else _fake_loaded_sources())}

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


# =============================================================================
# Terminal artifact set for a default cell (PB-02): run_single finalizes + run_sweep gates
# =============================================================================
# A default cell (log/eval interval above max_steps, checkpoint_interval=0) used to finish with a
# headline number but write no metrics.csv / best_model.pt / resumable bundle. run_single now runs a
# validation-only finalizer as a train() terminal callback, and run_sweep publishes the reuse contract
# only after the returned terminal checkpoint exists. These build TINY real models (embed_dim=4,
# n_heads=1, two batches); loaders and code/corpus identity are monkeypatched so no real cache is needed.

import math

from torch.utils.data import DataLoader

from vfe3.data.datasets import TokenWindows
from vfe3.model.model import VFEModel
from vfe3.train import train


_TINY_BASELINE = dict(
    vocab_size=6, embed_dim=4, n_heads=1, max_seq_len=8, batch_size=4,
    max_steps=2, n_layers=1, n_e_steps=1, seed=6, warmup_steps=1,
    gauge_group="glk", use_head_mixer=False, generate_figures=False,
    e_q_mu_lr=0.5, e_q_sigma_lr=0.05, e_phi_lr=0.0,
    m_p_mu_lr=0.1, m_p_sigma_lr=0.05, m_phi_lr=0.0,
    kl_max=32,
)


def _tiny_train_loader(seed=7, n=64, seq_len=8, bs=4):
    g = torch.Generator().manual_seed(seed)
    base = torch.arange(3).repeat(n // 3 + 2)
    ds = TokenWindows(base[:n].long(), seq_len)
    return DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True, generator=g)


def _fake_get_loader(dataset, seq_len, batch_size, split, *, max_tokens=None, vocab_size=None):
    if split == "train":
        return _tiny_train_loader(seq_len=seq_len, bs=batch_size)
    base = torch.arange(3).repeat(48 // 3 + 2)
    ds = TokenWindows(base[:48].long(), seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)


def _patch_tiny_cell(monkeypatch):
    r"""Point run_single at tiny synthetic loaders + deterministic code/tokenizer identity."""
    monkeypatch.setattr(ablation, "BASELINE_CONFIG", dict(_TINY_BASELINE))
    monkeypatch.setattr(ablation, "get_loader", _fake_get_loader)
    monkeypatch.setattr(ablation, "tokens_per_char", lambda *a, **k: 1.0)
    monkeypatch.setattr(ablation, "_tokenizer_tag", lambda *a, **k: "tiktoken")
    monkeypatch.setattr("vfe3.run_artifacts._git_code_identity",
                        lambda *a, **k: dict(FIXED_CODE_IDENTITY))


def test_default_ablation_cell_writes_terminal_artifact_set(tmp_path, monkeypatch):
    _patch_tiny_cell(monkeypatch)
    torch.manual_seed(0)
    run_dir = tmp_path / "cell"
    result = ablation.run_single("cell", {}, run_dir, dataset=DATASET,
                                 device=torch.device("cpu"), seed=6, max_steps=2)

    # The finalizer returned the full merge headline (a default cell no longer relies on cadence).
    assert result["error_kind"] is None
    for key in ("primary_val_ppl", "final_val_ppl", "best_val_ppl", "best_step",
                "final_train_loss", "n_params", "terminal_checkpoint"):
        assert key in result

    # The advertised artifact set is on disk.
    for name in ("metrics.csv", "best_model.pt", "summary.json", "validation_results.json",
                 "provenance.json", "pure_path_report.json"):
        assert (run_dir / name).exists(), name
    ckpt_path = run_dir / "checkpoints" / "step_2.pt"
    assert ckpt_path.exists()
    assert result["terminal_checkpoint"] == str(ckpt_path)

    # The resumable bundle carries model / optimizer / scaler+EMA slots / RNG / data cursor / step /
    # best-selection metadata, and safe-loads under weights_only=True.
    bundle = torch.load(ckpt_path, weights_only=True)
    assert bundle["step"] == 2
    for slot in ("model_state", "optimizer_state", "rng_state", "scaler_state", "ema_state",
                 "data_state", "best_val_ppl", "best_step"):
        assert slot in bundle
    assert bundle["data_state"] is not None                       # shuffled train loader -> real cursor

    # summary.json is validation-only.
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["selection_split"] == "validation"
    for forbidden in ("test_ce", "test_ppl", "test_bpc"):
        assert forbidden not in summary

    # Resume exactly one additional optimizer step from the terminal checkpoint.
    torch.manual_seed(0)
    cfg_resume = VFE3Config(**ablation._cell_cfg_dict({}, seed=6, max_steps=3))
    fresh = VFEModel(cfg_resume)
    losses_resume = train(fresh, _tiny_train_loader(), cfg_resume, n_steps=3,
                          resume_from=str(ckpt_path), device=torch.device("cpu"))
    assert len(losses_resume) == 1                                 # start_step == 2 -> one more step


def test_run_single_finalizes_before_writing_success_contract(tmp_path, monkeypatch):
    r"""run_sweep publishes a reuse contract only after the returned terminal checkpoint exists: a
    success result whose terminal_checkpoint path is missing is gated to a failed cell (no contract)."""
    monkeypatch.setattr(ablation, "_git_code_identity", lambda: dict(FIXED_CODE_IDENTITY))
    monkeypatch.setattr(ablation, "cache_source_identity", _fake_source_ok)
    sweep_name = "terminal_gate"
    monkeypatch.setitem(ablation.SWEEPS, sweep_name, {"description": "terminal checkpoint gate"})
    monkeypatch.setattr(ablation, "make_run_overrides",
                        lambda _n: [("no_ckpt", {}), ("with_ckpt", {})])
    def fake_run_single(label, overrides, run_dir, **kwargs):
        run_dir.mkdir(parents=True, exist_ok=True)
        tc = run_dir / "checkpoints" / "step_2.pt"
        if label == "with_ckpt":
            tc.parent.mkdir(parents=True, exist_ok=True)
            tc.write_text("x", encoding="utf-8")
        return {"label": label, "error_kind": None, "primary_val_ppl": 8.0,
                "final_val_ppl": 9.0, "seed": 6, "terminal_checkpoint": str(tc),
                "_loaded_data_sources": _fake_loaded_sources()}

    monkeypatch.setattr(ablation, "run_single", fake_run_single)
    monkeypatch.setattr(ablation, "_cleanup", lambda: None)
    ablation.run_sweep(sweep_name, tmp_path, dataset=DATASET, device=None, seed=6, resume=False)

    no_ckpt_dir = tmp_path / sweep_name / ablation._sanitize("no_ckpt")
    with_ckpt_dir = tmp_path / sweep_name / ablation._sanitize("with_ckpt")
    no_marker = json.loads((no_ckpt_dir / "ablation_result.json").read_text(encoding="utf-8"))
    with_marker = json.loads((with_ckpt_dir / "ablation_result.json").read_text(encoding="utf-8"))
    assert no_marker["status"] == "failed"
    assert not (no_ckpt_dir / "cell_contract.json").exists()
    assert with_marker["status"] == "success"
    assert (with_ckpt_dir / "cell_contract.json").exists()


def test_run_single_terminal_merge_preserves_metadata_and_primary_val_ppl(tmp_path, monkeypatch):
    r"""An earlier periodic best below the final validation is the returned primary_val_ppl, while the
    label / seed / overrides / token cap / parameter count survive the finalizer merge."""
    _patch_tiny_cell(monkeypatch)
    # A single periodic eval sets a best of 2.0 (below the terminal 100.0); primary must equal that best.
    calls = {"n": 0}

    def fake_evaluate(model, loader, **kw):
        calls["n"] += 1
        ppl = 2.0 if calls["n"] == 1 else 100.0
        ce = math.log(ppl)
        return {"ce": ce, "ppl": ppl, "bits_per_token": ce / math.log(2.0), "bpc": None}

    monkeypatch.setattr("vfe3.train.evaluate", fake_evaluate)

    overrides = {"eval_interval": 2}                              # one periodic eval fires at step 2
    result = ablation.run_single("mycell", overrides, tmp_path / "cell", dataset=DATASET,
                                 device=torch.device("cpu"), seed=6, max_tokens=123, max_steps=2)

    assert result["primary_val_ppl"] == 2.0                       # min(periodic best 2.0, final 100.0)
    assert result["final_val_ppl"] == 100.0
    assert result["best_val_ppl"] == 2.0
    assert result["label"] == "mycell"                            # metadata preserved through the merge
    assert result["seed"] == 6
    assert result["overrides"] == {"eval_interval": 2}
    assert result["max_tokens"] == 123
    assert isinstance(result["n_params"], int) and result["n_params"] > 0
    assert result["error_kind"] is None
    assert "terminal_checkpoint" in result

    with (tmp_path / "cell" / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert set(rows[-1]) == set(rows[0])
    assert float(rows[-1]["val_ppl"]) == 100.0
    assert rows[-1]["train_loss"] == ""
    assert rows[-1]["attn_entropy"] == ""
