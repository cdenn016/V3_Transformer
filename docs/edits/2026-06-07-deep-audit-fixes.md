# 2026-06-07 — Overnight deep-audit fixes

Multi-agent deep audit (deep-audit skill via the Workflow tool, 30 finders → per-finding verifier →
adversarial challenge, all read-only) + the resulting fixes. Full audit report:
`docs/audits/audit-2026-06-07.md`. Verified-ledger entries: `docs/verified.md` (2026-06-07 section).

Tests: baseline `595 (2 failing)` → **`612 passed / 0 failed`** (junit). TDD throughout (every new
test confirmed RED first, then GREEN; no forward-value regression).

## Commit `63368fd` — CORRECTION: eigh-adjoint wrong sign (found by the audit's pass 2)

The `_EighDamped` backward in `f069a8a` shipped with a WRONG SIGN (`delta_ij = w_i - w_j` →
`F = +1/(w_i-w_j)`; the symmetric-eigh adjoint requires `1/(w_j-w_i)`). The forward / Sigma=I NaN-cure
was correct, but the backward returned plausible-but-wrong gradients on `gaussian_full` when
eigenvectors rotate. It shipped because the original agreement/FD tests used `(sqrtA*sqrtA).sum() =
tr(A) = sum(w)` — eigenvalue-only (`gV=0`), blind to the F-term sign.
- `retraction.py`: `delta = w_j - w_i`; corrected docstring error order (`gap_eps/Delta^2` relative).
- `test_retraction.py`: `_f_uses_eigvecs` rebuilt eigenvector-dependent (fixed asymmetric contraction);
  RED-on-wrong-sign confirmed, GREEN-on-fix. Verified vs stock eigh backward to 6.7e-14 + FD 1.3e-8.
- `config.py`: extended the straight_through frozen-param warning to its `detach` sibling.
- `test_config.py`: parametrized the warn test over all three triggers; cross_couplings asserts
  `== [(0,1)]` directly (requires real coercion).

## Commits `fed1c29`, `3b23764`
- `fed1c29`: non-breaking warning when `straight_through` would silently freeze a learnable param;
  verified.md symmetry-argument note for the eigh gradient at exact degeneracy.
- `3b23764`: model-level full-cov first-backward regression guard, verified by mutation to discriminate.

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


## Gauge geometric correctness + head-mixer-per-block + char-corrected BPC (branch vfe3-gauge-geometric-correctness-2026-06-07)

Verified: the Killing gauge metric is CONFORMAL in the E_ij basis (regularized inverse = exactly
(1/2K)*I; cosine(plain-AdamW step, Killing-whitened-AdamW step) = 1.0), so killing/killing_per_block
precond is a no-op and a port of VFE_2.0's RiemannianAdamW would do nothing. Only the position-dependent
PULLBACK metric carries geometry. (docs/verified.md.)

- **Gauge M-step (geometric).** New `pullback_per_block` precond (geometry/phi_preconditioner.py): exact
  per-irrep-block exp-map metric, feasible at K=20 where full `pullback` raises (K>12). New
  `GaugeNaturalGradAdamW` (gauge_optim.py): natural-gradient + momentum on phi_embed/pos_phi_free
  (active rows only, NO Adam normalization — which would re-flatten the metric), AdamW elsewhere. Opt-in
  cfg.m_phi_natural_grad (+ m_gauge_momentum); geometric path = m_phi_natural_grad=True AND
  phi_precond_mode="pullback_per_block".
- **Head mixer now PER-BLOCK** (model/{block,stack,model}.py): after E-step, before norm (V2 order);
  behavior-preserving at the shipped n_layers=1.
- **Char-corrected BPC** (data/datasets.py `tokens_per_char` + train.evaluate/train, run_artifacts,
  train_vfe3): BPC = (ce/ln2)*tokens_per_char (Unicode codepoints) so en/ja/ar compare; default 1.0 =
  bits/token. PPL/CE unchanged.
- **CE-by-dim: NOT added** (orthogonal to cross-dataset comparison — a function of K, not the dataset).
- Tests +12 (pullback_per_block, gauge_optim incl. pullback-rotates-vs-Killing-cannot, head-mixer-per-
  block, bpc). Full suite 624 passed, 0 failures.

## Run folders named by test PPL (separate from the audit)

`train_vfe3.py`: run folders now finalize as `vfe3_runs/<test_ppl:.2f>_<label>/` (e.g.
`154.29_wikitext-103_K20_block_glk_linear_mix`) — no timestamp — so runs sort by perplexity in the
file browser (automating a rename the user had been doing by hand). The folder is still created with a
`<timestamp>_<label>` name while training (the PPL is unknown until `finalize_run` scores the test
split); the entry point swaps the timestamp prefix for the test PPL at the end.

- Factored `_run_label(cfg, dataset)` (`<dataset>_K<embed_dim>_<group>[_linear][_mix]`) out of
  `_run_dir`, which now prefixes it with the timestamp.
- Added `_rename_run_by_ppl(run_dir, label, test_ppl, logger)`: renames to `<test_ppl:.2f>_<label>`;
  keeps the timestamped name when the PPL is missing/non-finite; on a name clash appends `_2`, `_3`,
  ...; a failed move (open handle / locked dir) is logged, never fatal (numbers are already on disk).
- `main()` now captures `finalize_run`'s result and calls the rename after figures are written.
- Placement: the entry point, not `finalize_run` — `ablation.py` does NOT call `finalize_run` (it
  copies the winning config into `train_vfe3.py` for the test eval), and `make_figures.py` finds runs
  by `config.json` + mtime, so neither depends on the timestamp being in the folder name. The ISO
  timestamp is still preserved inside each run's `config.json`.
- `tests/test_run_naming.py`: +7 tests (label/tags, timestamped in-progress dir, rename, collision
  suffix, non-finite/missing-PPL skip, absent-dir skip).

Also in this working tree: click-to-run config operating-point tuning (`kappa`, `mstep_self_coupling_weight`,
`min_lr`, `min_lr_frac`) — committed separately as it carries the run's config.

## Re-activation of the PPL rename hook (this commit)

The PR-#39 rename code (`_run_label` / `_rename_run_by_ppl`) landed on the branch, but the subsequent
merge of `main` (char-corrected BPC) into the branch resolved the `finalize_run(...)` call-site
conflict by taking main's side, dropping the `results = ...` capture and the `_rename_run_by_ppl(...)`
call. The functions were left defined but unreachable, so runs were NOT renamed by PPL. Restored the
two-line hook in `main()` (capture `results`, call `_rename_run_by_ppl` with `test_ppl`), and restored
the two post-edit doc sections above that the same merge dropped. No other code changed.

## UMAP figure-gen warnings (investigated, no code change)

End-of-run console warnings from `report.generate_figures` → `figures.umap_embed` are both benign.
`n_jobs value 1 overridden to 1 by setting random_state` is the reproducibility/parallelism tradeoff
(`umap_embed` passes `random_state=seed`; once per channel). `Spectral initialisation failed! The
eigenvector solver failed` is caught inside UMAP (`umap/spectral.py:547-554`) and falls back to random
init, so the embedding completes — all `belief_umap_*.png` were written. By loop order (mu→sigma→phi)
it was the phi (gauge) embedding; the trigger is a near-disconnected / tiny-eigengap kNN graph, which
is *consistent with* (unconfirmed — no wikitext cache on the CPU box to re-extract) low gauge-frame
diversity across tokens. Left as-is; the warnings are informative.
