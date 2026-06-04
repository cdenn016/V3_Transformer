# Ablation runner: contiguous train→analyze→plot, with tack-on accumulation

Date: 2026-06-04. Scope: `ablation.py` only (standalone; nothing else reads its `CONFIG`).

## Problem

`ablation.py` has a `CONFIG["mode"]` switch with values `train`/`analyze`/`plot`/`list`, so
inspecting a sweep means a second run with a different mode. Two further frictions block the
desired workflow:

1. The per-cell CSV writer rewrites `sweep_results.csv` from the *current in-memory value
   list only*. Re-running `kappa` with a new value list (e.g. `0.5, 2.2, 3.7` after
   `1, 2, 3, 4`) overwrites the earlier points instead of adding to them.
2. Plotting (`generate_plots`) regenerates *every* sweep's figure after *every* sweep, rather
   than the one that just finished.

## Goal

One contiguous run: train each sweep, then immediately write that sweep's CSV, print its
analysis table, and save its PPL figure, then move to the next sweep. After all sweeps, make
the cross-sweep comparison (sensitivity plot + best-per-sweep table). A later re-run with a
different value list "tacks on" its new cells to the existing figure.

## Decisions (confirmed with user)

- **Drop the mode trichotomy; keep list.** Replace `CONFIG["mode"]` with a boolean
  `CONFIG["list_only"]` (default `False`). `True` prints the sweep registry and exits; `False`
  runs the contiguous flow. Standalone `analyze`/`plot` modes are removed — a fully-cached
  re-run re-analyzes and re-plots for free (cached cells skip `run_single` entirely: no
  retrain, no loader build).
- **Tack-on = union of markers.** Accumulation is the union of every `ablation_result.json`
  marker under a sweep dir, keyed by cell directory (one per label).

## Design

### Accumulation

New `_collect_sweep_results(sweep_dir)` returns
`[json.load(m) for m in sorted(sweep_dir.glob("*/ablation_result.json"))]` (unreadable markers
skipped). Each label maps to its own subdirectory, so re-running the same label overwrites its
one marker while the others persist; the union is **additive and never subtracts**.

`run_sweep` writes the per-sweep CSV from this union (per executed cell, for liveness, plus a
final whole-frame write that also covers the all-cached case) and returns the union. The
"best" completion line is computed over the union.

### Contiguous flow (`main`)

```
if CONFIG["list_only"]: print registry; return
device / sweep_names / validate_sweeps  (unchanged)
for name in sweep_names:
    run_sweep(name, ...)            # trains + writes union CSV
    analyze_sweep(output_dir/name)  # this sweep's table (accumulated)
    _plot_one_sweep(output_dir/name, fig_dir)   # this sweep's PPL figure (tacked on)
_plot_sensitivity(output_dir, fig_dir)          # cross-sweep comparison, over ALL persisted sweeps
summarize_sweeps(output_dir)                     # best-per-sweep table, over ALL persisted sweeps
```

### Plot/analysis refactor (surgical)

`generate_plots` splits into `_plot_one_sweep(sweep_dir, fig_dir)` (one sweep's figure from its
accumulated CSV: numeric `param=value` → line, categorical arms → sorted bar) and
`_plot_sensitivity(output_dir, fig_dir)` (PPL-range bar per sweep, over every persisted sweep,
matching the original cross-sweep behavior). A `_plt_or_none()` helper imports matplotlib and
applies the publication style once, returning `None` on failure so plotting stays best-effort.
`analyze_all` splits into the existing per-sweep `analyze_sweep` plus `summarize_sweeps`
(best-per-sweep table). `generate_plots` and `analyze_all` are orphaned by these changes and
removed.

## Consequences (per the union decision)

- Additive, never subtracts: sweeping `kappa=1,2,3` after `1,2,3,4` keeps `kappa_4` on the
  figure. To remove a point, delete its cell directory.
- `kappa=2` and `kappa=2.0` sanitize to different dirs → two points at x=2.0. Don't mix
  int/float spellings of the same value.
- Config-rejected/crashed cells leave markers (no `config.json`); they persist in the CSV/table
  as `inf` and are filtered out of figures.

## Verification

Tack-on is pure data, testable with zero training (this box is CPU-only and defaults to
wikitext-103). `tests/test_ablation_tackon.py` writes fake markers and asserts:
`_collect_sweep_results` returns 4, then 7 after adding three; the union CSV has the matching
row count; the union x-values sort to `[0.5, 1, 2, 2.2, 3, 3.7, 4]`; re-running the same label
overwrites (count stays, value updates); `_plot_one_sweep` never raises (headless-safe).
