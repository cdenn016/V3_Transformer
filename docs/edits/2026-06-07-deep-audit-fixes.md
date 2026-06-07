# 2026-06-07 — Overnight deep-audit fixes

Multi-agent deep audit (deep-audit skill via the Workflow tool, 30 finders → per-finding verifier →
adversarial challenge, all read-only) + the resulting fixes. Full audit report:
`docs/audits/audit-2026-06-07.md`. Verified-ledger entries: `docs/verified.md` (2026-06-07 section).

Tests: baseline `595 (2 failing)` → **`606 passed / 0 failed`** (junit). TDD throughout (every new
test confirmed RED first, then GREEN; no forward-value regression).

## Commit `f069a8a` — HIGH: gap-regularized eigh backward (full-cov SPD retraction)

- `vfe3/geometry/retraction.py`: added `_EighDamped` (custom autograd) + `_eigh_damped` wrapper — a
  drop-in for `torch.linalg.eigh` with a Lorentzian-damped gap `Delta/(Delta^2+gap_eps)` in the
  backward. Replaced all 6 eigh sites in `retract_spd_full` / `retract_logeuclidean_full`. Forward is
  bit-identical (damping is backward-only); fixes the 100%-NaN gradient at the isotropic `Sigma=I`
  default `gaussian_full` init on the default `unroll` estimator. Updated the stale "deferred" docstring.
- `tests/test_retraction.py`: +6 tests — agreement vs stock eigh backward (independent oracle), FD
  gradcheck, finite backward at `Sigma=I` (unit + both full retractions + end-to-end e_step).
- Review note: changes gradients on the opt-in `gaussian_full` family only; `gaussian_diagonal` untouched.

## Commit `ac19d2b` — safe batch

- `vfe3/config.py`: `min_lr_frac` default `0.01 → 0.0` (matches the 2026-06-06 edit-doc intent; greens
  the two scheduler tests; pure path is `min_lr=min_lr_frac=0`). NaN-reject in the min_lr/min_lr_frac
  validators (`v != v`). Coerce `cross_couplings` list pairs → tuples in `__post_init__` (fixes the
  cold-start `viz.report` JSON-round-trip crash).
- `ablation.py`: `get_loader` threads split-aware `shuffle`/`drop_last` (validation/test read the whole
  split, mirroring the train_vfe3 F1 fix); `_cell_is_current` now also compares the session `dataset`
  (resume no longer serves a wrong-dataset cell as current); removed a stray `''` SWEEPS literal.
- `vfe3/geometry/transport.py`: tightened the means-only `RopeTransport` docstring (mean/covariance
  operator mismatch → Mahalanobis invariant not preserved; coherent path is `rope_full_gauge`).
- `train_vfe3.py`: surfaced `min_lr_frac` + `amp_dtype` in the click-to-run dict; softened the
  "EVERY toggle" docstring claim.
- `tests/test_config.py`, `tests/test_ablation_tackon.py`: +5 regression tests.

## Housekeeping
- Deleted three atomic-write `*.tmp.*` scratch files from the repo root (tooling debris; the Edit tool
  leaves them — re-swept at session end).

## NOT fixed (deliberate — see the audit report for rationale)
- `amp_dtype='fp16'` GradScaler — documented deferred stub (config.py:288-291), left as-is.
- `straight_through` freezing learnable alpha/connection_W/lambda_beta — recommend a config guard;
  toggle-combination judgment call left for the author.
- Two dense-Omega hot spots (e_step phi-align, gamma forward) — perf-only, OFF by default; belongs to
  the dense-Omega speedup roadmap as a golden-equivalence refactor.
- `attention_tau` single temperature across heads — needs a focused manuscript check (likely by-design
  for the equal-block default); adjacent to the settled per-irrep-block-beta design.
- Pre-existing dead code / unused imports — left per CLAUDE.md ("mention, don't delete").
- ~30 LOW diagnostics/metrics findings — diagnostics-only; recommended as one follow-up pass.
