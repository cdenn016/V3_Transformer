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
            "primary_val_ppl": ppl, "n_params": 1000, "seed": 6,
        }),
        encoding="utf-8",
    )


def _write_resume_cell(
    run_dir: Path,
    marker: object,
    *,
    dataset: str = "wikitext-103",
    max_steps=None,
) -> None:
    from dataclasses import asdict
    from vfe3.config import VFE3Config

    run_dir.mkdir()
    cfg_dict = ablation._cell_cfg_dict({}, seed=6, max_steps=max_steps)
    saved = {
        "config": json.loads(json.dumps(asdict(VFE3Config(**cfg_dict)), default=str)),
        "dataset": dataset,
    }
    (run_dir / "config.json").write_text(json.dumps(saved), encoding="utf-8")
    (run_dir / "ablation_result.json").write_text(json.dumps(marker), encoding="utf-8")


def _successful_marker(**updates) -> dict:
    marker = {
        "label": "cell",
        "error_kind": None,
        "status": "success",
        "primary_val_ppl": 10.0,
        "final_val_ppl": 10.0,
        "seed": 6,
        "collect_diagnostics": False,
        "collect_extrapolation": False,
    }
    marker.update(updates)
    return marker


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
    ablation._LOADER_CACHE.clear()
    ablation.get_loader("wikitext-103", 16, 4, "validation")
    ablation.get_loader("wikitext-103", 16, 4, "train", max_tokens=None)
    ablation._LOADER_CACHE.clear()
    assert captured["validation"].get("shuffle") is False
    assert captured["validation"].get("drop_last") is False
    assert captured["train"].get("shuffle") is True
    assert captured["train"].get("drop_last") is True


def test_cell_is_current_false_on_dataset_change(tmp_path: Path) -> None:
    r"""Resume must not serve a cell trained on a DIFFERENT dataset as current. The cell's
    VFE3Config carries no dataset field (it is a session knob), so _cell_is_current must also
    compare the persisted top-level config.json 'dataset' against the current session dataset.
    The marker carries no max_tokens key (a pre-fix cell): a full-data resume (max_tokens=None)
    must still read it as current (missing key == None, the backward-compat path)."""
    run_dir = tmp_path / "cell"
    _write_resume_cell(run_dir, _successful_marker())

    assert ablation._cell_is_current(run_dir, {}, seed=6, dataset="wikitext-103") is True
    assert ablation._cell_is_current(run_dir, {}, seed=6, dataset="wikitext-2") is False


def test_cell_is_current_checks_max_tokens(tmp_path: Path) -> None:
    r"""max_tokens (the loader train-token cap) is not a VFE3Config field, so config.json alone
    cannot distinguish a capped smoke cell from a full run: _cell_is_current must also compare
    the max_tokens persisted in the ablation_result.json marker."""
    run_dir = tmp_path / "cell"
    ds = "wikitext-103"
    _write_resume_cell(run_dir, _successful_marker(max_tokens=1000), dataset=ds, max_steps=1)

    assert ablation._cell_is_current(run_dir, {}, seed=6, dataset=ds, max_steps=1,
                                     max_tokens=1000) is True
    assert ablation._cell_is_current(run_dir, {}, seed=6, dataset=ds, max_steps=1,
                                     max_tokens=None) is False


def test_cell_is_current_rejects_failed_or_incomplete_markers(tmp_path: Path) -> None:
    markers = [
        [],
        _successful_marker(status="failed", error_kind="train"),
        _successful_marker(error_kind="train"),
        _successful_marker(final_val_ppl=float("inf")),
        _successful_marker(final_val_ppl=float("nan")),
    ]
    missing_status = _successful_marker()
    missing_status.pop("status")
    markers.append(missing_status)
    missing_terminal = _successful_marker()
    missing_terminal.pop("final_val_ppl")
    markers.append(missing_terminal)

    for i, marker in enumerate(markers):
        run_dir = tmp_path / f"cell_{i}"
        _write_resume_cell(run_dir, marker)
        assert ablation._cell_is_current(
            run_dir, {}, seed=6, dataset="wikitext-103",
        ) is False


def test_cell_is_current_requires_requested_diagnostic_output(tmp_path: Path) -> None:
    flag_missing = tmp_path / "flag_missing"
    _write_resume_cell(flag_missing, _successful_marker())
    assert ablation._cell_is_current(
        flag_missing, {}, seed=6, dataset="wikitext-103", collect_diagnostics=True,
    ) is False

    output_missing = tmp_path / "output_missing"
    _write_resume_cell(output_missing, _successful_marker(collect_diagnostics=True))
    assert ablation._cell_is_current(
        output_missing, {}, seed=6, dataset="wikitext-103", collect_diagnostics=True,
    ) is False

    complete = tmp_path / "complete"
    _write_resume_cell(complete, _successful_marker(
        collect_diagnostics=True, attn_entropy=1.0,
    ))
    assert ablation._cell_is_current(
        complete, {}, seed=6, dataset="wikitext-103", collect_diagnostics=True,
    ) is True


def test_cell_is_current_requires_requested_extrapolation_output(tmp_path: Path) -> None:
    flag_missing = tmp_path / "flag_missing"
    _write_resume_cell(flag_missing, _successful_marker())
    assert ablation._cell_is_current(
        flag_missing, {}, seed=6, dataset="wikitext-103", collect_extrapolation=True,
    ) is False

    output_missing = tmp_path / "output_missing"
    _write_resume_cell(output_missing, _successful_marker(collect_extrapolation=True))
    assert ablation._cell_is_current(
        output_missing, {}, seed=6, dataset="wikitext-103", collect_extrapolation=True,
    ) is False

    complete = tmp_path / "complete"
    _write_resume_cell(complete, _successful_marker(
        collect_extrapolation=True, extrap_ce=[],
    ))
    assert ablation._cell_is_current(
        complete, {}, seed=6, dataset="wikitext-103", collect_extrapolation=True,
    ) is True


def test_run_sweep_markers_persist_requests_and_terminal_state(tmp_path: Path, monkeypatch) -> None:
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
