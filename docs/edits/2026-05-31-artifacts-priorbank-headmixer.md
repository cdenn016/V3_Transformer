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
