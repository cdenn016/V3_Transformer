r"""Tack-on accumulation for the ablation runner.

The "tack on a second value list to the first sweep's figure" behavior lives entirely in the
pure-data helpers (``_collect_sweep_results`` / ``_write_sweep_csv`` / ``_read_sweep_csv``), so it
is provable with zero training: write fake ``ablation_result.json`` markers and assert on the
accumulated frame. ``_plot_one_sweep`` is exercised only for "does not raise" (it is best-effort
and silently skips when matplotlib is unavailable), so the discriminating checks are data-level.
"""

import json
import math
from pathlib import Path

import pytest

import ablation


def _write_marker(sweep_dir: Path, label: str, ppl: float) -> None:
    r"""A minimal headline marker under the cell dir the runner would create for ``label``."""
    cell = sweep_dir / ablation._sanitize(label)
    cell.mkdir(parents=True, exist_ok=True)
    (cell / "ablation_result.json").write_text(
        json.dumps({
            "sweep": sweep_dir.name, "label": label, "error_kind": None,
            "status": "success", "primary_val_ppl": ppl, "final_val_ppl": ppl,
            "n_params": 1000, "seed": 6,
            "collect_diagnostics": False, "collect_extrapolation": False,
        }),
        encoding="utf-8",
    )


def test_collect_union_and_tack_on(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "kappa"
    sweep_dir.mkdir()

    # First sweep: kappa = 1, 2, 3, 4.
    for v in (1, 2, 3, 4):
        _write_marker(sweep_dir, f"kappa={v}", ppl=10.0 + v)
    union = ablation._collect_sweep_results(sweep_dir)
    assert len(union) == 4
    ablation._write_sweep_csv(sweep_dir, union)
    assert len(ablation._read_sweep_csv(sweep_dir)) == 4

    # Later: tack on kappa = 0.5, 2.2, 3.7 (a DIFFERENT value list).
    for v in (0.5, 2.2, 3.7):
        _write_marker(sweep_dir, f"kappa={v}", ppl=20.0 + v)
    union2 = ablation._collect_sweep_results(sweep_dir)
    assert len(union2) == 7                                  # old four + new three, not replaced
    ablation._write_sweep_csv(sweep_dir, union2)
    rows = ablation._read_sweep_csv(sweep_dir)
    assert len(rows) == 7

    # The merged figure's x-axis: union of both value lists, sorted by numeric value.
    xs = sorted(float(r["label"].split("=")[-1]) for r in rows)
    assert xs == [0.5, 1.0, 2.0, 2.2, 3.0, 3.7, 4.0]


def test_rerun_same_label_overwrites(tmp_path: Path) -> None:
    r"""Re-running the SAME label updates that one cell (no duplicate point)."""
    sweep_dir = tmp_path / "kappa"
    sweep_dir.mkdir()
    _write_marker(sweep_dir, "kappa=1", ppl=11.0)
    _write_marker(sweep_dir, "kappa=1", ppl=99.0)            # same dir -> overwrites the marker
    union = ablation._collect_sweep_results(sweep_dir)
    assert len(union) == 1
    assert union[0]["primary_val_ppl"] == 99.0


def test_int_float_spellings_stay_distinct(tmp_path: Path) -> None:
    r"""kappa=2 and kappa=2.0 sanitize to different dirs -> two points (documented caveat)."""
    sweep_dir = tmp_path / "kappa"
    sweep_dir.mkdir()
    _write_marker(sweep_dir, "kappa=2", ppl=12.0)
    _write_marker(sweep_dir, "kappa=2.0", ppl=13.0)
    assert len(ablation._collect_sweep_results(sweep_dir)) == 2


def test_unreadable_marker_is_skipped(tmp_path: Path) -> None:
    r"""A partial/corrupt marker is skipped, not fatal, so the rest of the union survives."""
    sweep_dir = tmp_path / "kappa"
    sweep_dir.mkdir()
    _write_marker(sweep_dir, "kappa=1", ppl=11.0)
    bad = sweep_dir / "kappa_2"
    bad.mkdir()
    (bad / "ablation_result.json").write_text("{not valid json", encoding="utf-8")
    union = ablation._collect_sweep_results(sweep_dir)
    assert len(union) == 1
    assert union[0]["label"] == "kappa=1"


def test_collect_sweep_results_rejects_malformed_failed_and_nonfinite_markers(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "kappa"
    sweep_dir.mkdir()
    _write_marker(sweep_dir, "valid", ppl=11.0)
    invalid = [
        [],
        {"label": "failed", "status": "failed", "error_kind": "train", "final_val_ppl": 11.0},
        {"label": "errored", "status": "success", "error_kind": "train", "final_val_ppl": 11.0},
        {"label": "infinite", "status": "success", "error_kind": None,
         "final_val_ppl": float("inf")},
        {"label": "missing", "status": "success", "error_kind": None},
    ]
    for i, marker in enumerate(invalid):
        cell = sweep_dir / f"invalid_{i}"
        cell.mkdir()
        (cell / "ablation_result.json").write_text(json.dumps(marker), encoding="utf-8")

    union = ablation._collect_sweep_results(sweep_dir)
    assert [marker["label"] for marker in union] == ["valid"]


def test_plot_one_sweep_does_not_raise(tmp_path: Path) -> None:
    r"""Best-effort plotting must never raise, with or without matplotlib installed."""
    sweep_dir = tmp_path / "kappa"
    sweep_dir.mkdir()
    for v in (1, 2, 3):
        _write_marker(sweep_dir, f"kappa={v}", ppl=10.0 + v)
    ablation._write_sweep_csv(sweep_dir, ablation._collect_sweep_results(sweep_dir))
    ablation._plot_one_sweep(sweep_dir, tmp_path / "figures")


def test_get_loader_threads_split_aware_shuffle_drop_last(monkeypatch) -> None:
    r"""ablation.get_loader must mirror train_vfe3._select_loader's F1 split-aware semantics:
    train requests shuffle=True/drop_last=True, validation/test request shuffle=False/drop_last=False,
    so the held-out metric reads the WHOLE split in a stable order (datasets.make_dataloader defaults
    to the TRAIN regime, so get_loader must pass the eval flags explicitly)."""
    captured: dict = {}

    def fake_make_dataloader(dataset, split, seq_len, batch_size, **kw):
        captured[split] = kw
        return object()                                      # a non-None sentinel get_loader caches

    monkeypatch.setattr(ablation, "make_dataloader", fake_make_dataloader)
    monkeypatch.setattr(ablation, "cache_source_identity", lambda dataset, split: {
        "format": "pt", "tokenizer_tag": "fixture", "size_bytes": len(split),
        "sha256": "0" * 64, "meta": None, "meta_sha256": None,
    })
    ablation._LOADER_CACHE.clear()
    ablation.get_loader("wikitext-103", 16, 4, "validation")
    ablation.get_loader("wikitext-103", 16, 4, "train", max_tokens=None)
    ablation._LOADER_CACHE.clear()
    assert captured["validation"].get("shuffle") is False
    assert captured["validation"].get("drop_last") is False
    assert captured["train"].get("shuffle") is True
    assert captured["train"].get("drop_last") is True


def test_run_sweep_markers_persist_requests_and_terminal_state(tmp_path: Path, monkeypatch) -> None:
    # Per-cell _cell_is_current staleness (dataset / max_tokens / diagnostic-flag / marker-validity)
    # is now bound to cell_contract.json and covered in tests/test_ablation_artifact_resume_20260712.py.
    # Stub the contract's code + corpus identity so this run-sweep marker test stays fast, deterministic,
    # and independent of any real tokenized cache on disk.
    monkeypatch.setattr(ablation, "_git_code_identity",
                        lambda: {"git_sha": "0" * 40, "git_dirty": False, "git_dirty_fingerprint": None})
    monkeypatch.setattr(ablation, "cache_source_identity",
                        lambda dataset, split, *, cache_dir=None: {
                            "format": "pt", "tokenizer_tag": "tiktoken", "size_bytes": len(split),
                            "sha256": "0" * 64, "meta": None, "meta_sha256": None})

    sweep_name = "marker_contract"
    monkeypatch.setitem(ablation.SWEEPS, sweep_name, {
        "description": "marker contract test",
        "collect_diagnostics": True,
        "collect_extrapolation": True,
    })
    monkeypatch.setattr(ablation, "make_run_overrides", lambda _name: [
        ("success", {}), ("failure", {}),
    ])

    def _fake_run_single(label, _overrides, _run_dir, **kwargs):
        assert kwargs["collect_diagnostics"] is True
        assert kwargs["collect_extrapolation"] is True
        if label == "success":
            return {
                "label": label,
                "error_kind": None,
                "primary_val_ppl": 8.0,
                "final_val_ppl": 9.0,
                "attn_entropy": 1.0,
                "extrap_ce": [],
                "_loaded_data_sources": {
                    split: {
                        "format": "pt", "tokenizer_tag": "tiktoken",
                        "size_bytes": len(split), "sha256": "0" * 64,
                        "meta": None, "meta_sha256": None,
                    }
                    for split in ("train", "validation")
                },
            }
        return {
            "label": label,
            "error_kind": "train",
            "error": "boom",
            "primary_val_ppl": float("inf"),
        }

    monkeypatch.setattr(ablation, "run_single", _fake_run_single)
    monkeypatch.setattr(ablation, "_cleanup", lambda: None)
    ablation.run_sweep(
        sweep_name, tmp_path, dataset="wikitext-103", device=None, seed=6, resume=False,
    )

    markers = {}
    for label in ("success", "failure"):
        path = tmp_path / sweep_name / ablation._sanitize(label) / "ablation_result.json"
        markers[label] = json.loads(path.read_text(encoding="utf-8"))
        assert markers[label]["collect_diagnostics"] is True
        assert markers[label]["collect_extrapolation"] is True
        assert "error_kind" in markers[label]

    assert markers["success"]["status"] == "success"
    assert markers["success"]["error_kind"] is None
    assert math.isfinite(markers["success"]["final_val_ppl"])
    assert markers["failure"]["status"] == "failed"
    assert markers["failure"]["error_kind"] == "train"
    assert "final_val_ppl" in markers["failure"]
    assert not math.isfinite(markers["failure"]["final_val_ppl"])

    # A reuse contract is published only for the successful cell (never for the failed one).
    success_dir = tmp_path / sweep_name / ablation._sanitize("success")
    failure_dir = tmp_path / sweep_name / ablation._sanitize("failure")
    assert (success_dir / "cell_contract.json").exists()
    assert not (failure_dir / "cell_contract.json").exists()


def test_expand_range_sign_mismatch_raises() -> None:
    r"""A sign-mismatched [start, stop, step] must raise, not silently expand to zero cells."""
    with pytest.raises(ValueError):
        ablation._expand_range([0, 5, -1])
    with pytest.raises(ValueError):
        ablation._expand_range([5, 0, 1])


def test_expand_range_valid_directions_unchanged() -> None:
    r"""Ascending, descending, and the degenerate single-point range all still expand."""
    assert ablation._expand_range([0, 4, 2]) == [0, 2, 4]
    assert ablation._expand_range([5, 0, -1]) == [5, 4, 3, 2, 1, 0]
    assert ablation._expand_range([2, 2, 1]) == [2]


def test_sanitize_distinct_labels_do_not_collide() -> None:
    r"""The char-replace map is lossy ('a=b', 'a b', 'a/b' all map to 'a_b'), so the appended
    raw-label hash must keep distinct labels in distinct run dirs, deterministically."""
    assert len({ablation._sanitize("a=b"), ablation._sanitize("a b"),
                ablation._sanitize("a/b")}) == 3
    assert ablation._sanitize("kappa=2") == ablation._sanitize("kappa=2")
