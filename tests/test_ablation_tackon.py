r"""Tack-on accumulation for the ablation runner.

The "tack on a second value list to the first sweep's figure" behavior lives entirely in the
pure-data helpers (``_collect_sweep_results`` / ``_write_sweep_csv`` / ``_read_sweep_csv``), so it
is provable with zero training: write fake ``ablation_result.json`` markers and assert on the
accumulated frame. ``_plot_one_sweep`` is exercised only for "does not raise" (it is best-effort
and silently skips when matplotlib is unavailable), so the discriminating checks are data-level.
"""

import json
from pathlib import Path

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
