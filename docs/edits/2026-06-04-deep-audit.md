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

## audit-new.md triage + fixes (26-finding 5-investigator audit)

Verified and triaged every actionable finding in `docs/audits/audit-new.md` with an adversarial
verify+triage workflow (six independent investigators + a synthesis pass, 7 agents) under the project
policies (audit policy: defaults need not be pure, only that a pure path EXISTS under a toggle;
leave-dead-code-mention-don't-delete; surgical changes; sanctioned NN exceptions). Findings were not
inherited from the audit's own verifier; each was re-checked against executable source. Three key
discriminators the audit's investigators missed:

- Config "allowlists block registered variants" is mostly a false positive. The static tuples equal
  their live registries for gauge_group and alpha_mode (a registry swap is a no-op today); the only
  static/registry gaps are DELIBERATELY GATED values -- decode's `linear` (reached solely via
  `use_prior_bank=False`; `output_proj_weight` is None under `use_prior_bank=True`, so admitting
  `decode_mode='linear'` would AttributeError) and encode's `gauge_fixed` stub (raises at config time).
  The curation is registry-minus-gated and intentional.
- Transport matrix-exp clamp (`max_norm=15`) / float64 upcast (`dim_threshold=20`) are HARDCODED
  defaults, not config toggles. The exact path is never tripped on the default (`n_heads=8`,
  `tied_block_glk` gives `||M||_F = sqrt(n_heads)*||phi|| = 14.14 < 15` at the `||phi||<=5` retraction
  ceiling), but IS reachable in principle on a valid non-default pure config (`tied_block_glk`,
  `n_heads>=10` -> `>=15`). float64 upcast is more precise (perf, not purity). Disposition: add an
  opt-in toggle, never flip the default (surfaced below, not implemented).
- Gamma temperature IS genuinely wrong for single-block groups (fixed below).

### Fixed (committed)

- `fix(gamma)`: the model-coupling block divided its per-pair energy by `cfg.tau_gamma =
  kappa_gamma*sqrt(d_head)`, but the energy accumulates over the gauge-irrep block size (the full K
  for single-block groups glk/so_k/sp, `irrep_dims=[K]`), whose correct temperature is
  `kappa_gamma*sqrt(K)`. Now uses `attention_tau(cfg.kappa_gamma, group.irrep_dims)` (the belief beta
  channel's rule). No-op on the default block_glk; corrects single-block + `gamma_coupling>0`, where
  under `prior_source='model_channel'` the gamma gradient shapes s (and thus q over training) -- a real
  training correction, not cosmetic. Red->green test on glk n_heads=2 with a literal-tau oracle.
- `fix(config)`: `cocycle_relaxation` had no validation; added a finite `[0,1]` guard (the bracketed
  form rejects NaN/inf) since it feeds the regime_ii connection directly.
- `fix(contracts)`: `oracle.py` `omega` annotation named unimported FactoredTransport/RopeTransport so
  `get_type_hints` raised NameError (TYPE_CHECKING does NOT fix runtime get_type_hints -- a real import
  is required; transport.py does not import oracle, so no cycle); chunked-forward return annotation
  widened to Optional first element; `compute_metrics` return widened (free_energy_terms returns a
  nested dict); top-p generation softmax computed once instead of twice (value-identical).
- `fix(diagnostics)`: RoPE diagnostics/attention-map replay now pass rope/rope_on_cov so the replayed
  belief matches the forward (was converged without rope, then rope applied post-hoc);
  `free_energy_value` global F now honors transport_mode/connection_W/cocycle_relaxation (was
  flat-only), filtered F raises under a non-flat mode, and the docstring's false log_alpha claim was
  corrected. Both are diagnostic-only; default flat + no-rope path is byte-identical.

### Surfaced for owner decision (NOT changed)

ADD-TOGGLE (policy: a pure path should exist under a toggle, and currently does not for the reachable
`tied_block_glk, n_heads>=10` case; the trigger is unverified in practice and the fix is the widest
blast radius -- hot-path signature + 3 callers + config -- so it is the owner's call):
- Transport matrix-exp Frobenius clamp has no exact/no-clamp toggle. Recommended: config fields
  `transport_exp_max_norm: Optional[float]=None` / `transport_exp_dim_threshold:int=20` threaded into
  `stable_matrix_exp_pair`, `None -> skip the rescale` (exact `exp(M)`), default 15.0 preserved. This
  also makes max_norm/dim_threshold config-selectable (answers the config-registry hardcoding angle).

SURFACE-ONLY (perf opts -- re-profile on the 5090, CPU rankings do not port; or documented deferrals;
or dead code which policy says mention-don't-delete):
- Regime-II builds+discards a dense flat Omega (`transport.py:164`); phi_alignment_loss uses dense
  `_transport` not the factored fast path (`e_step.py:248`); factored covariance per-head einsums not
  batched (`transport.py:467`); per-head energy slice+stack vs reshape (`free_energy.py:106`, and the
  reshape is diagonal-only -- full-covariance block() slices a sub-block, so not a clean drop-in).
- CUDA host-syncs: CE all-ignore branch (`model.py:403`), chunked-decode `n_valid==0`
  (`prior_bank.py:365`), per-step `float(loss)` (`train.py:199`), robust-Cholesky `bool(ok.all())`
  (default-off, `numerics.py:50`) -- all are NaN-guards / intrinsic to a documented return contract;
  removing them needs branchless rewrites + re-pinning, not surgical.
- Gamma channel uses flat transport regardless of `transport_mode` (`model.py:499`) -- documented,
  predictively-inert deferral (s-channel detached, does not feed logits); regime_ii gamma transport is
  deferred design, not a claimed feature.
- Legacy `run_training` lacks `.to(DEVICE)` (`train.py:475`) -- uncalled, deprecated helper.
- Gauge generators built in float64 then cast (construction-time only, more precise; non-issue).
- `cross_couplings` cannot round-trip through persisted JSON (`config.py:367`) -- LATENT: no code path
  reloads a config from JSON today, so per the simplicity policy this is deferred until a
  resume-from-checkpoint feature lands (one-line list->tuple normalize when it does).
- `train_vfe3.py` helper signatures miss type hints (`synthetic_period3_loader`, `_banner`) -- left
  untouched to avoid entangling with the owner's in-flight uncommitted edits to that file.

FALSE POSITIVE (audit mischaracterized; no action):
- "Config allowlists block registered variants" (the gated-value analysis above).
- "gauge_parameterization is a dead field" -- it is a deliberate fail-fast gate (`omega_direct` raises
  with a documented reason at config construction), working as designed.
- "invariant_families naming inconsistent" -- the literal mismatch (`('gaussian',)` vs
  `'gaussian_diagonal'`) is harmless: `invariant_for` is called only in tests, never on the runtime
  path, so the comparison never fires.

Verification: full CPU suite `tests=572 failures=0 errors=0 skipped=0` (1 xpassed) under a
workspace-local `--basetemp`, read from JUnit XML; temp dir + XML removed after reading. Five isolated
commits (per-coord-alpha guard, gamma temperature, contracts, cocycle guard, diagnostics); the owner's
uncommitted `ablation.py`/`train_vfe3.py` edits were kept out of every commit.
