# 2026-06-04 - Deep audit report

Performed a code-focused deep audit on branch `codex-deep-audit-2026-06-04`. No implementation files were
changed. Added `docs/audits/audit-2026-06-04.md` documenting one confirmed medium configuration defect,
the green CPU test result under a workspace-local pytest base temp, and local verification limitations
for the default temp root, CPU-only Torch build, and optional UMAP/numba native trace.

Temporary audit outputs were cleaned up after verification: `.codex-audit-junit.xml` and
`.codex-pytest-tmp-20260604` were removed.

## Ablation: contiguous train/analyze/plot flow + tack-on accumulation

Reworked `ablation.py` so a run is one contiguous flow instead of a `mode` switch. The
`CONFIG["mode"]` key (`train`/`analyze`/`plot`/`list`) was replaced by a boolean
`CONFIG["list_only"]`: `True` prints the sweep registry and exits, `False` runs the flow. For
each sweep `main` now does `run_sweep` -> `analyze_sweep` (per-sweep table) -> `_plot_one_sweep`
(per-sweep PPL figure); after all sweeps it makes the cross-sweep comparison (`_plot_sensitivity`
+ `summarize_sweeps`). Design spec: `docs/superpowers/specs/2026-06-04-ablation-contiguous-flow-design.md`.

Tack-on: a new `_collect_sweep_results(sweep_dir)` returns the union of every
`*/ablation_result.json` marker, and `run_sweep` writes the per-sweep CSV (and returns) from that
union. A re-run with a different value list (e.g. `kappa=0.5,2.2,3.7` after `1,2,3,4`) now adds its
new cells to the existing figure rather than overwriting it; the union is additive and never
subtracts (to drop a point, delete its cell directory). The old `generate_plots` was split into
`_plt_or_none` / `_plot_one_sweep` / `_plot_sensitivity`, and `analyze_all` into the existing
`analyze_sweep` plus `summarize_sweeps`; `generate_plots` and `analyze_all` were removed.

Verification: `tests/test_ablation_tackon.py` (5 tests, all pass via `--junitxml`: union grows
4 -> 7 across two value lists, x-values sort to `[0.5,1,2,2.2,3,3.7,4]`, same-label re-run
overwrites, int/float spellings stay distinct, corrupt markers are skipped, `_plot_one_sweep`
never raises). An end-to-end synthetic smoke (two `main()` runs, `kappa=[1,2]` then `[2,3]`)
confirmed the contiguous flow produces 3 accumulated points (the overlapping `2.0` cached, not
duplicated) plus both figures; the throwaway smoke script and its temp output dir were removed.

## Audit Report Resave

Resaved `docs/audits/audit-2026-06-04.md` as `docs/audits/audit-new.md` at the user's request.

## Fix: per-coordinate alpha rejects non-Renyi functional at construction (prior audit M1)

Addressed the one confirmed finding of the prior `docs/audits/audit-2026-06-04.md` (M1). The config
already rejected `state_dependent_per_coord` alpha paired with a full-covariance family
(`vfe3/config.py`, the covariance guard), but it accepted the same per-coordinate alpha paired with a
non-Renyi `divergence_family` (e.g. `squared_hellinger`); the construction succeeded and only crashed at
the first forward, where `free_energy.self_divergence_per_coord` raises because the per-coordinate
self-divergence is registered only for the Renyi functional (KL = Renyi at alpha=1).

Added a construction guard directly after the covariance guard that mirrors the runtime raise:
`if alpha_is_per_coord(self.alpha_mode) and self.divergence_family != "renyi": raise ValueError(...)`.
It uses the `alpha_is_per_coord` registry helper (so any future per-coordinate alpha form is covered
without editing here) and the same `"renyi"` literal the runtime gate uses (so the construction guard
rejects exactly what the forward would, never over-rejecting). No valid path is lost: this doubly
opt-in combination already crashed at the first forward, so the guard only moves the failure earlier.

Verification (TDD, red then green): `tests/test_config.py::test_per_coord_alpha_requires_renyi_functional`
was confirmed FAILING against pre-fix code (`DID NOT RAISE ValueError`, since the config constructs the
non-Renyi per-coord pair on the default diagonal family) and PASSING after the guard; the existing
`test_per_coord_alpha_requires_diagonal_family` still passes (no over-rejection on the Renyi default).
The test uses the DEFAULT diagonal family on purpose so the covariance guard does not mask the functional
check. Full suite: `tests=569 failures=0 errors=0 skipped=0` (1 xpassed) read from the JUnit XML under a
workspace-local `--basetemp`; the temp dir and XML were removed after reading.

Replaced `docs/audits/audit-new.md` with the owner-filtered deep-audit report so the new filename reflects the latest disposition: neural output projection and learned/config-toggle paths are excluded from actionable status.
