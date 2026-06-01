# 2026-05-31 — Training artifacts, `use_prior_bank=False`, head mixer

Branch `vfe3-artifacts-priorbank-2026-05-31` (fresh from `origin/main` after the perf-speedup
merge, PR #6). Overnight feature work requested in one session: (1) wire up training
persistence/artifacts (checkpoints, best model, CSV metrics, plots, end-of-run **test** PPL);
(2) add a `use_prior_bank=False` linear-decode ablation mirroring VFE_2.0; (3) port VFE_2.0's
gauge-equivariant head mixer. The user authorized the two neural exceptions (linear decode;
head mixer that breaks strict equivariance under the default group) explicitly.

## Task 1a — `use_prior_bank=False` linear-projection decode (VFE_2.0 parity)

The pure default (`use_prior_bank=True`) decodes via the KL-to-prior readout
`logits = -KL(q_i || pi_v)/tau_eff`. VFE_2.0 also exposes a `use_prior_bank=False` ablation
(better results on the user's runs) whose decode is a plain linear projection of the converged
mean; the user wants the with/without comparison in V3. `use_prior_bank` is now the **single
decode gate** — encode and the free-energy self-coupling stay on the PriorBank; only decode
changes.

Changes:
- `vfe3/config.py`: removed the `NotImplementedError` that rejected `use_prior_bank=False`;
  documented it as the single decode gate (so `decode_mode` and `use_prior_bank` cannot silently
  disagree — the linear path simply never consults `decode_mode`). Default stays `True`.
- `vfe3/model/prior_bank.py`: `PriorBank.__init__` gains `use_prior_bank: bool = True`; creates
  `output_proj_weight` as a raw `(V, K)` `nn.Parameter` (Xavier-uniform, no bias) **only** on the
  ablation path (`None` otherwise). `decode()` routes to the new `linear` kernel when
  `use_prior_bank=False`. New `@register_decode("linear")` kernel returns `mu_q @ W^T` (sigma and
  tau_eff discarded). Realized as a Parameter matmul, **not** an `nn.Linear` module — no
  neural-layer class enters the module (so `test_model_has_no_nn_layers` holds even on the
  ablation path).
- `vfe3/model/model.py`: threads `use_prior_bank=cfg.use_prior_bank` into the PriorBank.
- `vfe3/train.py` `build_optimizer`: groups `output_proj_weight` (and, ahead of Task 1c, the head
  mixer) conditionally at `m_mu_lr`, and now **asserts the param groups cover
  `model.parameters()` exactly** — a forgotten group would silently freeze a weight.
- `CLAUDE.md`: documented both authorized neural exceptions (opt-in, default OFF, pure path
  always present).

Tests (TDD, RED→GREEN): new `tests/test_use_prior_bank.py` (8 tests: linear decode equals
`mu @ W^T`, discards sigma, no projection on the pure path, encode unchanged by the toggle, no
nn.Linear module on the ablation path, forward+backward trains `output_proj_weight`, optimizer
covers all params on both paths). Updated `tests/test_config.py::test_config_accepts_use_prior_bank_false`
(was `…rejects…`).

## Task 1c — Schur-commutant head mixer (`use_head_mixer`, opt-in)

Port of VFE_2.0's `VFEHeadMixer`. New `vfe3/model/head_mixer.py::HeadMixer`: one learned
`A = I + Delta in R^{n x n}` over the `n` equal-size gauge-irrep blocks (under `block_glk`: the
`n_heads` heads), embedded as `kron(A, I_d)` and applied to `mu` (`M mu`) and `Sigma`
(`M Sigma M^T`; diagonal closed form `sigma'[m] = sum_n A[m,n]^2 sigma[n]`, full-cov sandwich via
two einsums on the reshaped block axes). `Delta` stored as the delta-from-identity (zeros init)
so a mixer-on model is bitwise-identical to mixer-off at step 0.

Gating fact (settled by reading `generate_glk_multihead`, generators.py:106-114): V3's `block_glk`
gives each head its OWN independent `gl(d_head)` sub-algebra — an UNTIED per-block gauge — so
`kron(A, I_d)` does NOT commute with the per-head gauge action and the mixer breaks strict gauge
equivariance there (exact at init, deviates as `A` drifts). User-accepted as an opt-in exception;
the no-mixer default stays equivariant. The mixer requires >= 2 equal blocks, so a single-block
group (`glk` / `so_k`) + `use_head_mixer=True` raises at `VFEModel` construction.

Changes: `vfe3/config.py` `use_head_mixer: bool = False`; `vfe3/model/model.py` builds the mixer
from `group.irrep_dims` and applies it after `vfe_stack`, before the final norm; `build_optimizer`
groups its `mixer_delta` at `m_mu_lr` (already wired via the `getattr(model, "head_mixer", None)`
hook). Tests: new `tests/test_head_mixer.py` (9 tests: identity-init no-op, mean mixing, diagonal
`A^2` closed form, full-cov sandwich == explicit `kron`, equal-block validation, config toggle,
model no-op-at-init-then-trains, optimizer coverage, single-block rejection).

## Task 1b — training artifacts / persistence

The user's complaint: training ran end-to-end but saved nothing. New `vfe3/run_artifacts.py`:
`RunArtifacts` owns a run directory and the incremental writes; `finalize_run` does the end-of-run
test eval + summary + figures.

Per run (`vfe3_runs/<timestamp>_<dataset>_K<K>_<group>[_linear][_mix]/`): `config.json` (full
config + n_params/dataset/device/timestamp), `metrics.csv` (one row per periodic eval: step,
train_loss, lr, val_ce/ppl/bpc, and the converged diagnostics — attn entropy, the per-term
free-energy decomposition, effective rank), `checkpoints/step_<N>.pt` (resumable: model +
optimizer + config + step), `best_model.pt` (`state_dict` at the lowest val PPL), `test_results.json`
(held-out TEST-split eval on the **reloaded best-val checkpoint**), `summary.json`, and figures
(`loss_curve.png`, `val_ppl.png`, `free_energy_terms.png`).

Wiring: `vfe3/config.py` adds `checkpoint_interval: int = 0` (validated >= 0). `train()` gains an
opt-in `artifacts` parameter — with `artifacts=None` the loop is untouched (silent path
bitwise-identical, still pinned by `test_silent_and_logging_paths_are_bitwise_identical`); with an
artifacts object it logs a CSV row + saves best at each periodic eval and checkpoints at
`checkpoint_interval`. `train_vfe3.py` builds the run dir (`RUN_ROOT = "vfe3_runs"`, set to `None`
to disable), threads `artifacts` through `train`, and calls `finalize_run` on the dataset's `test`
split (the `wikitext-*` test caches exist; the synthetic anchor reuses its val loader). Figure
generation is best-effort (a plotting error is logged, never fatal). `.gitignore` ignores
`vfe3_runs/`.

Validation: TDD `tests/test_run_artifacts.py` (9 tests), AND a CPU smoke run of the **actual**
`train_vfe3.main()` entry point (tiny dims, synthetic stream) for BOTH the pure path and the
`use_prior_bank=False`+`use_head_mixer=True` ablation — both wrote config.json, metrics.csv,
best_model.pt, checkpoints (step_3, step_6), test_results.json, summary.json, and the figures to
disk, with the test eval correctly reloading the best-val checkpoint. The full 5090 wikitext-103
job is the user's to run; this validates the plumbing, not convergence.

## Suite

Full suite after Task 1: **240 tests, 0 failures, 0 errors** (239 passed + 1 non-strict xpass),
read from `--junitxml`. Branch fresh from `origin/main` (post perf-merge). No autonomous merge to
main (left for the user's review).

## Task 2 — deep audit + surgical fixes

Five-lens parallel audit (code-reviewer, debugger, refactoring-specialist, performance-engineer,
python-pro) + per-finding adversarial source verification, run as a workflow (read-only investigators
=> safe under the shared-tree edit-loss caveat). 33 agents, 28 findings, **28 CONFIRMED / 0 REFUTED**.
Report: `docs/audits/audit-2026-05-31-task1-buildout.md`. No hard-constraint violations (no-NN pure
path intact, sandwich transport correct, head-mixer algebra correct).

Triaged → fixed the genuine, surgical subset (mostly Task-1 code):
- `run_artifacts.py`: `torch.load(..., weights_only=True)` on the best-checkpoint reload (security,
  matches datasets.py); `finalize_run -> Dict[str, object]` (the dict mixes float/Optional/bool);
  `_save_free_energy_bar(figs: ModuleType)`; signature `=`-alignment fixed (MANDATORY convention).
- `prior_bank.py`: class-level `output_proj_weight: Optional[nn.Parameter]` annotation.
- `train.py`: validation log no longer prints val CE in both "Loss" and "CE" columns (now
  `CE | PPL | BPC`); `build_optimizer` comment clarifies the coverage assert guards GROUPING, not
  gradient FLOW, and enumerates the intentional null-gradient toggle cases; `run_training`
  docstring marked superseded by `train_vfe3.main()`.
- `train_vfe3.py`: same validation-log de-duplication on the final line.
- `model.py`: a `warnings.warn` guardrail when `use_prior_bank=False` AND `detach_e_step=True`
  (that combination freezes the encode prior tables — only `output_proj_weight` trains). New test
  `test_use_prior_bank_false_with_detach_e_step_warns_encode_tables_frozen`.

Deferred (pre-existing / intentional / explicitly-deferred), documented in the report:
- The dense `(B,N,N,K,K)` Ω at transport.py:152 saved by autograd each step, and the unexploited
  block-diagonal structure (the two "high" perf findings) — this IS the deferred perf P0 #2; it
  needs a golden-test-pinned perf pass on the 5090, not an audit-fix commit. Remains the top
  speedup-roadmap item.
- Per-forward GPU→CPU sync (`flat_targets.any()`), 3 forwards per log step, per-iter
  `torch.tensor(irrep_dims)`, matrix_exp scatter loop — pre-existing; fold into the GPU perf pass.
- `reference_decode`/`run_training` orphans, duplicate `_banner`, pre-existing type-precision
  (forward union, diagnostics `dict`, `block_norm: Optional[Any]`) — left per the surgical rule.

Full suite after audit fixes: **241 tests, 0 failures, 0 errors** (240 passed + 1 non-strict xpass).

## Task 3 Part 1 — diagnostics tier completion (gauge-geometry probes)

`vfe3/metrics.py` already provides `holonomy_deviation` (curvature proxy — mean Frobenius departure
of the triangle holonomy `Ω_ij Ω_jk Ω_ki` from `I`, ~0 for the flat φ-cocycle) and
`gauge_trace_spread` (std of `log|det Ω| = tr(embed(φ))` across tokens), but they were not surfaced.
Now `VFEModel.diagnostics()` computes both at the converged transport (`omega`, `out.phi`, and
`group.generators` were already in scope), the training CSV row carries `holonomy_deviation` +
`gauge_trace_spread` (added unconditionally to keep the CSV rectangular), and `finalize_run` writes
`holonomy.png` + `gauge_trace_spread.png` trajectories. Pure measurements, eval-cadence, no_grad —
no hot-path change. Tests: `test_diagnostics_includes_gauge_geometry_probes`,
`test_metrics_csv_includes_gauge_geometry_columns`, `test_finalize_writes_gauge_geometry_figure`.
Full suite: **244 tests, 0 failures, 0 errors** (243 passed + 1 non-strict xpass).

## Task 3 Part 2 — `tied_block_glk` group: an equivariant home for the head mixer

The 1c finding was that `block_glk`'s untied per-head gauge breaks the head mixer's equivariance.
Part 2 closes that gap with a theory-faithful structure-group variant. New
`generate_glk_multihead_tied(K, n_heads)` builds generators `kron(I_n, E_ij)` — the same `gl(d)`
basis replicated across all heads, so `n_gen = d²` and one φ drives `exp(sum_a phi_a kron(I,E_a)) =
kron(I, exp(M))`, i.e. the SAME `GL(d)` frame in every head (a TIED gauge). Registered as
`tied_block_glk` (irrep_dims `[d]*n_heads`, so transport / per-head attention are unchanged); added
to `_VALID_GAUGE_GROUPS`.

Viability gated by a throwaway probe before integration (per review): the group builds, the tied
generators give bit-identical head-blocks with zero off-block, forward+backward is finite, and the
tied+head_mixer combo trains. The probe also confirmed `killing_per_block` raises under the shared
generators (they do not partition per head) — so `config.__post_init__` now REJECTS
`tied_block_glk` + `killing_per_block` at construction (use `none`/`clip`/ambient `killing`).

Payoff, verified as a UNIT test on the mixer (not an end-to-end model claim): under a tied gauge
`Omega = kron(I_n, h)`, `M = kron(A, I_d)` commutes with `Omega`, so the FULL-COVARIANCE mixer is
EXACTLY gauge-equivariant — `mix(Omega mu, Omega Sigma Omega^T) == (Omega M mu, Omega M Sigma M^T
Omega^T)` (`test_head_mixer_equivariant_under_tied_gauge_full_cov`). CAVEAT (documented, not hidden):
the diagonal closed form is equivariant only under DIAGONAL gauges (the diagonal-of-sandwich
approximation V3 already uses), so it is deliberately not asserted under a general tied gauge. Tests
(5): tied-across-heads generator structure, model-runs-under-tied, config rejection, the full-cov
equivariance, plus the existing mixer suite. Full suite: **248 tests, 0 failures, 0 errors** (247
passed + 1 non-strict xpass).
