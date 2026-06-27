# Investigation of the Codex deep audit (audit-2026-06-27.md)

Date: 2026-06-27. Independent re-verification of all 20 Codex findings against the
**current** working tree (`feat/ablation-multiseed-a2-pmatch`, HEAD `2145b25`), not the
throwaway audit commit `ef1cee6` (which is not in this history, so every Codex line number
is stale and was re-located by content).

Method: a 20-agent fan-out (one skeptical investigator per finding) plus an adversarial
challenge pass on every finding upheld at medium+ severity, with reachability judged against
the live `train_vfe3.py` config. Three of the trickiest reachability claims (DBG-1 barycenter,
DBG-2 grad-accum, PY-1 closure) were additionally spot-checked by hand and agree with the fleet.

## Bottom line

Codex's grep located **real code** in essentially every case — but after accounting for
(a) reachability under the active config, (b) the project's documented intentional toggles, and
(c) actual PyTorch semantics, **none of the 20 findings survives at medium or high severity.**

- Confirmed critical: 0. Confirmed high: 0. Confirmed medium: 0.
- **False as stated:** DBG-1, REF-1 (2 findings).
- **Deliberate, tested stub (not a bug):** REF-5.
- **Severity none** (real code, zero harm): DBG-1, REF-1, REF-5, PERF-4, PY-1 (5).
- **Severity low** (real but dormant/trivial/cosmetic): the remaining 13.
- Codex's one "high" (CR-1) → low; its eight "medium" → low or none.

Codex over-weighted "the code literally does X" without asking "does X fire, and is X wrong."
The `CLAUDE.md` rules it kept tripping over: the pure path lives under a toggle (so a default
toggle being impure is not a defect), documented NN/numeric exceptions are intentional, and
"float32 throughout" means storage/working dtype, not intermediate accumulation.

## Results table

| ID | Codex | Verdict | Reachable on active run? | Final sev | Real defect to fix? |
|---|---|---|---|---|---|
| CR-1  | high | partial | opt-in (`resume_from`) | low | hardening only |
| CR-2  | medium | **real** | opt-in (full-cov + precision) | low | yes — trivial config guard |
| DBG-1 | medium | **false** | — | none | no |
| DBG-2 | medium | partial | opt-in (`grad_accum_steps>1`) | low | no (documented limitation) |
| DBG-3 | medium | partial | conditional | low | optional marker |
| DBG-4 | medium | partial | scaling.py only | low | optional hardening |
| REF-1 | low | **false** | never | none | no |
| REF-2 | medium | partial | never (latent) | low | yes — modularity contract |
| REF-3 | medium | partial | opt-in (regime_ii/rope + s-chan) | low | no (defensible design) |
| REF-4 | medium | **real** | never (latent) | low | yes — modularity contract |
| REF-5 | low | intentional | opt-in stub | none | no (tested fail-fast stub) |
| REF-6 | medium | partial | ablation.py only | low | yes — trivial consistency |
| PERF-1| medium | partial | yes | low | partial (CE sync only) |
| PERF-2| medium | partial | yes | low | yes — trivial branchless |
| PERF-3| low | **real** | non-default decode | low | optional |
| PERF-4| medium | partial | never (full-cov only) | none | no (documented fp64 island) |
| PY-1  | medium | partial | never (closure always None) | none | defensive only |
| PY-2  | medium | partial | inference only | low | optional guards |
| PY-3  | low | **real** | never (cosmetic) | low | typing nit |
| PY-4  | low | **real** | never (cosmetic) | low | typing nit |

## Corrections to Codex (where it was wrong, not just over-severe)

- **DBG-1 — FALSE.** The `barycenter_r_()` half cannot run on the active path: `r_update_mode`
  defaults to `"gradient"`, so the `== "barycenter"` guard is False; under gradient mode `r` is an
  optimizer leaf whose update is correctly skipped. `scheduler.step()` advancing on a skipped
  non-finite step is the standard PyTorch-AMP pattern and was *deliberately written to mirror* the
  enabled-GradScaler path (which also skips `optimizer.step()` via `found_inf` yet advances the
  scheduler). No desync. (Verified by hand: train.py:391-411, r_update_mode in train_vfe3.py:258.)
- **REF-1 — FALSE.** The premise "AdamW weight-decay drifts the config-dead tables" is wrong:
  `AdamW.step()` filters `if p.grad is None: continue` *before* decoupled weight decay, so a
  None-grad table is skipped entirely — no update **and** no decay. Under `prior_source="model_channel"`
  the belief tables get `.grad = None` (only `.dtype` reads touch them), confirmed dead by the
  live/dead probe. Grouping them is *required* by the exact-coverage guard. No harm exists.
- **PERF-2 — wrong function cited.** Codex pointed at `decode_ce_diagonal_chunked` (prior_bank.py:441),
  the `use_prior_bank=True` KL-decode twin, which is **OFF**. The active CE under `use_prior_bank=False`
  is `decode_ce_linear_chunked`. The Python-branch-on-CUDA-scalar pattern is real in both, but the
  "diagonal_chunked is hot" justification was attached to the inactive path, and the sync is per-step,
  not "on the rare all-ignore branch" as the title says.
- **PERF-4 — not "silent/undocumented."** The fp64 cast is an in-code-documented "float64 island
  (audit 2026-06-13 M4)" because the congruence sandwich squares cond(Omega); the result is cast back
  to fp32, so storage stays fp32. It only runs under `family="gaussian_full"` (off). Not a violation
  of "float32 throughout."
- **PY-1 — unreachable.** The double-step needs a non-`None` closure, but `GradScaler.step(optimizer)`
  (train.py:402) always passes `closure=None`. The path can never fire. (Verified by hand.)
- **CR-1 — high → low.** Opt-in (`resume_from` defaults None; a 15k-step run never even writes a
  resumable bundle since `checkpoint_interval=25000`). The file is a self-written artifact; every
  genuinely external load in the repo already uses `weights_only=True`. A recognized hardening gap on a
  single-user research box, not a practical RCE.

## Actionable punch list (genuine defects, all low/trivial)

These are the only items that are *real defects* (vs. dormant-but-correct or cosmetic) and align with
the project's own conventions. None affects the active default run's numbers.

1. **CR-2 — config should reject an unrunnable combo (trivial).** `decode_precision_scaled=True` with
   `family="gaussian_full"` divides `(B,N,K)` by `(B,N,K,K)` and crashes at the first forward. Add to
   `VFE3Config.__post_init__`: raise when `decode_precision_scaled and not use_prior_bank and not
   diagonal_covariance`, noting the precision-weighted linear decode's `eta = Sigma^-1 mu` is the
   diagonal form. Turns an opaque broadcast error into a clear rejection.

2. **REF-2 + REF-4 — registry metadata gaps (modularity contract).** `CLAUDE.md`: "add a variant by
   writing-and-registering, never by editing call sites." Transport state-routing
   (`_REGIME_NEEDS_MU/SIGMA` frozensets + `== "regime_ii"` gates in model.py/e_step.py) and decode
   rank/chunk routing (`("full","full_chunked")`, `("diagonal_chunked","full_chunked")` literal tuples
   in config.py/model.py) are hard-coded name lists, not registry metadata. No current config
   mis-routes (all registered modes are handled), but a newly-registered state-consuming transport or
   non-chunked decoder would validate yet mis-route. Move the capability flags into the registries.

3. **REF-6 — ablation `embed_dim` sweep leaves `kl_max` stale (trivial).** The sweep overrides only
   `embed_dim`; `_cell_cfg_dict()` does not recompute `kl_max = 8*K`, so the K=40/64 arms run at the
   baseline `kl_max=160` instead of 320/512, deviating from the 8*K convention used everywhere else
   (scaling.py, the `tied_block_glk_wide` arm). Currently slack (binding starts ~K>=120-200, above the
   sweep's max), so it does not confound *these* arms — but it would if the sweep were extended. Give
   each arm an explicit `kl_max: 8*K`, or recompute it when `embed_dim` is overridden and `kl_max` is not.

4. **PERF-2 — branchless chunked CE (trivial, micro-perf).** In the active `decode_ce_linear_chunked`
   (and its twins), replace `n_valid = valid.sum(); if n_valid == 0: ...` with
   `(ce_per_pos * valid).sum() / valid.sum().clamp_min(1)` to drop one per-step host sync while keeping
   the all-ignore grad-connected zero. Negligible payoff on a 5090 but a clean change.

### Optional / cosmetic (only if you want them)

- **DBG-4:** in scaling.py's cache-hit branch, reject a cached run whose `test_ce` is non-finite (a
  divergent-but-completed run); truncated runs already leave no `summary.json` and are re-run correctly.
- **DBG-3:** add a "(live/dead probe failed)" banner marker so a probe failure is distinguishable from a
  genuinely all-live config (today they render identically — ambiguity, not a false report).
- **PERF-1:** gate the redundant `step_ce` D2H copy on `metrics_out is not None` (the `step_loss` sync is
  required by the NaN-skip guard and is the return value — leave it).
- **PERF-3:** branchless dense CE (only reached on non-chunked decode modes).
- **PY-1:** add a `if closure is not None: raise NotImplementedError(...)` guard to
  `GaugeNaturalGradAdamW.step` to make the latent footgun explicit (path never fires today).
- **PY-2:** three one-line `assert`s on `temperature/top_k/top_p` in `generate()`'s non-greedy branch —
  but this tensions with the project's "no error handling for impossible scenarios" rule; inference-only.
- **PY-3 / PY-4:** typing nits — annotate `report.py`'s `model` as `Optional["VFEModel"]` under a
  `TYPE_CHECKING` guard; give `viz/figures.py` a `FigureFn = Callable[..., Figure]` alias and annotate
  `umap_embed`. No runtime effect; `viz/` is not imported on the training path.

### No action (false / intentional / dormant-and-correct)

DBG-1 (false), REF-1 (false), REF-5 (tested fail-fast stub), PERF-4 (documented fp64 island),
DBG-2 (documented, instrumented, inert by default), REF-3 (model-channel-flat is defensible — the
learned connection reads belief means, not s-means).
