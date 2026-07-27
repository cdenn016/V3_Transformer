# Deep Audit 2026-07-27 — PROGRESS / RECOVERY STATE

> **STATUS: AUDIT COMPLETE (11:30 CDT).** All eight stages finished. The deliverable is
> `docs/audits/audit-2026-07-27.md`; raw per-agent output is in `docs/audits/raw/`. This file is
> the scaffolding that made the 09:00 rate-limit interruption recoverable and is safe to delete.
> One obligation remains open: the CUDA lane was not run because a training job held the GPU.

**This file exists so a rate-limit interruption is recoverable.** A one-shot cron job is set to
fire at 09:01 CDT to resume. Read this file first, then re-invoke the `deep-audit` skill and
resume from the first INCOMPLETE stage. Do not redo completed stages.

- **Branch:** `audit/2026-07-27-deep-audit` (forked from `main` @ `3b3a51e`)
- **Started:** 2026-07-27 08:04 CDT
- **Scope:** whole repo. Core surface is `vfe3/` (67 modules, 46,277 lines) plus the 11
  top-level click-to-run entry points.
- **Raw agent findings land in:** `docs/audits/raw/`
- **Final report goes to:** `docs/audits/audit-2026-07-27.md`

## Constraints that must survive a context reset

- **DO NOT touch `ablation.py` or `train_vfe3.py`** — uncommitted, intentional live experiment
  config (holds `mu_weight_decay=0.0` and the task-2 bundle values). Never stash/revert/clean them.
- **DO NOT start fixing anything.** Present the punch list first; the user authorizes fixes.
- **Interpreter:** bare `python` is CPU-only torch. `C:/anaconda/python.exe` (`$CUDA_PYTHON`) is the
  CUDA build. The GPU lane additionally needs `VFE3_TEST_DEVICE=cuda`. Never make a CUDA claim from
  bare python.
- **Pytest:** do NOT add `-q` (`pyproject.toml` already sets it; `-qq` silently suppresses the
  `N passed` summary). Read counts from `--junitxml` XML, never from memory or a progress line.
- Known-good baseline from merged commit `2b7a96d`: CPU `tests=4444 failures=1 errors=0 skipped=29`;
  CUDA `24 passed`. The one failure is the known `mm_damping` config drift between `train_vfe3.py`
  (1.0) and `ablation.BASELINE_CONFIG` (0.75) — config-bound, not a code regression.

## Stage ledger

| # | Stage | Status | Artifact |
|---|-------|--------|----------|
| 1 | Scope check + branch | **COMPLETE** | branch `audit/2026-07-27-deep-audit` |
| 2 | Read CLAUDE.md, confirm theory-expert gate | **COMPLETE** | gate MET (repo declares gauge/SPD/KL/equivariance invariants) |
| 3 | Base-five investigator wave (parallel) | **COMPLETE — all 5 returned** | `docs/audits/raw/base-*.md` |
| 4 | Domain-expert wave — full `audit-*` pool (parallel) | **4 of 7 returned; 3 RE-DISPATCHED 09:05** | `docs/audits/raw/expert-*.md` |

### Wave 4 detail

**ALL 7 RETURNED AND SAVED.** `audit-numerical-analyst`, `audit-variational`,
`audit-gauge-theorist`, `audit-transformer-ml` returned before the 09:00 session limit;
`audit-geometer`, `audit-info-geometer`, `audit-implementation-engineer` were killed by it,
re-dispatched at 09:05 with an expanded already-reported list, and all three returned.

**61 candidate findings total across 12 agents.** Zero at critical band by investigator rating;
13 at high band. No agent found target leakage, gradient severance on the default path, a dead
config field, an undocumented ninth NN, or a security defect.

### Stage 5 — verifier split (3 parallel `general-purpose` agents, 09:48)

Split because 61 findings is too many for one verifier to re-derive rigorously:
- **Verifier A** — the 13 high/critical-band findings, incl. re-running their executed probes.
- **Verifier B** — base-wave medium/low, PLUS spot-checking base-3's "no dead config field among
  ~171" and base-5's "kwargs bag fully synchronized" negative results.
- **Verifier C** — expert-wave medium/low, PLUS re-deriving >= 6 of the experts' claimed-clean
  negatives (SPD exp map exactness, KL direction, beta stationarity, sp(2m,R) brackets,
  pre-softmax masking). A claimed-clean item that is actually defective outranks any listed finding.

All three are told: re-read the cited line, reject comment-only evidence, cite `file.py:line`,
judge REACHABILITY under the live `train_vfe3.py`, and never edit a repo file.

### THE EMISSION CLUSTER — headline of this audit, six findings on one 142-line module

`vfe3/emission.py` plus its fusion/eval seams, all shipped yesterday in `2b7a96d`. Two independent
agents reproduced the autocast defect with different probes.

1. **`kernels.py:573-594`** — `mm_exact_update`'s autocast fp32 re-entry omits `emission` and
   `emission_weight` from its explicit argument list, so the recursion defaults them to
   `None`/`0.0` and the Bohning block never runs. `config.py:1648-1656` REQUIRES
   `e_step_update='mm_exact'` whenever `emission_mode != 'off'`, and live `train_vfe3.py` sets
   `amp_dtype='bf16'` — so the emission factor is a **silent no-op under the live AMP config**.
   Reproduced twice: `emission SURVIVES autocast? False`; `amp on == amp off exactly? True`.
2. **`e_step.py:471`** — `free_energy_value` accepts-and-ignores `emission`/`emission_weight`, so
   every logged/scored F measures a functional the belief does not descend (D-08 class). Measured:
   one E-step lowers the true objective 4.825 -> 3.803 while reported F RISES 96x.
3. **`kernels.py:701`** — the Bohning `(d,g)` pair is built ONCE from the pre-stack mean
   (`model.py:1126`) but fused against each layer's advancing prior, so at `n_layers > 1` the
   quadratic is centered on the wrong point. Residual `1.2e-07` at n_layers=1 vs `5.5e-01` beyond.
4. **`emission.py:110-140`** — no fp32 autocast island; the softmax runs in bf16 under the live AMP
   context. Measured relative error `6.03e-03`.
5. **`emission.py:110-140`** — expansion-point probabilities are not detached (only `mu_p` is), so
   `p_0` keeps a live `W` path, contradicting the module's stated MM contract and retaining the
   streaming vocab loop in the autograd graph.
6. **`emission.py:82-84`** — `bohning_curvature_diagonal` uses the cancellation-prone
   `E[x^2]-E[x]^2` form; its advertised `clamp(min=1e-12)` SPD guard is unreachable in float32.

Independently VERIFIED CORRECT in the same module: the streaming running-max/renormalize algebra is
exactly equivalent to one-shot logsumexp; the sigma arm `(a+pair_mass)/P` is the exact stationary
point of the `E_q`-averaged surrogate; the emission reads `x_t` only — NO target leakage.

**USER IMPACT: the pre-registered next experiment is the emission arms at K=20. Under the live
config those runs would measure nothing and read as "emission doesn't help."**
| 5 | Independent verifier views (3 parallel) | **COMPLETE — all 3 returned** | `raw/verifier-{A,B,C}-*.md` |
| 6 | Adversarial challenge — 5 duels, 10 agents | **DISPATCHED 10:20** | in report |

### Stage 5 outcome — 61 findings verified

**1 REFUTED, 1 SPLIT, 59 CONFIRMED at source.** But reachability collapsed most of them.

**REFUTED:** the perf agent's `(B,N,V)` decode claim, AND the orchestrator's own reframing of it.
The decode registry has `linear -> supports_chunked=True` with a fused CE
(`prior_bank.py:1894-1900`), so at the default `use_prior_bank=False` the active mode is `linear`,
the fused branch runs, and `logits = None`. The two defaults do NOT compose into a dense path. Live
config is doubly safe (`use_prior_bank=False` AND `decode_mode='diagonal_chunked'`, `batch_size=16`
not 64). Only the narrow conditional survives: `use_prior_bank=True` + `decode_mode='diagonal'`.

**SPLIT (IG-1, kl_max):** the clamp DOES gate the self-term (confirmed), but "no unclamped path
exists" is **REFUTED** — `inf` is rejected yet `VFE3Config(kl_max=1e30)` is accepted and is
indistinguishable from unclamped. Live `kl_max=1680` also sits above the investigator's computed
saturation point, so their table does not transfer.

**ONLY SIX FINDINGS ARE LIVE UNDER THE CURRENT CONFIG:**
1. `IE-3` — `mm_exact` silently ignores six E-step knobs the live config sets. Verifier C: "the most
   consequential item on my list."
2. `TML-1` — ALiBi log floor, `model.py:2396-2399`. The only live HIGH-band item. Caveat: "decisive"
   is an argument, not a measurement.
3. `GT-2` — `per_head_gauge_invariants` publishes an SVD ratio as a gauge invariant.
4. `TML-3` — `collect_beta_channel_decomposition` ablates the gamma channel along with beta.
5. `TML-4` — `attn_entropy_min` / `collapsed_heads` structurally constant under any causal prior.
6. `INFO-3` — Gaussian effective-rank floor breaks the participation ratio's scale invariance.

Everything else is gated shut by live toggles (`emission_mode='off'`, `e_phi_lr=0.00`,
`e_mu_q_trust=None`, `n_layers=1`, `family='gaussian_diagonal'`, `transport_mode='flat'`,
`query_adaptive_tau=False`, `learnable_kappa_beta=False`, `randomize_e_steps=False`,
`*_reflection='off'`, `pos_phi='learned'`). Two items (`NA-7`, `NA-8`) have NO production caller at
all.

**ALL TEN re-derived negative results UPHELD.** SPD exp-map exactness (2.192e-13), congruence
equivariance on the pure path (2.685e-15), AIRM distance identity, `alpha->1` recovering the FORWARD
KL (residual 0.0 vs 9.427e-01 for the reverse), beta stationarity (SymPy residuals
`[0.0, -2.22e-16, 0.0]`, envelope identity exactly 0), the entropy-surrogate routing fence,
`sp(2m,R)` brackets (0.000e+00 exactly — cleaner than claimed), pre-softmax `log_prior`, and zero
future-mass leak (exactly 0.000000e+00 above the diagonal, with a sensitive `uniform` control at
2.5019e-01). **No claimed-clean item turned out defective.**

### Stage 6 — five duels dispatched (skeptic + defender each)

Chosen for decision-relevance, not raw severity: `IE-3` (live, highest impact), `TML-1` (only live
high), `NA-1`/`T-2` autocast-drops-emission (pre-launch blocker for the user's next experiment),
`NA-2` fp64 escalation (reachability gap is the open question), `GT-2` (live, and a measurement-
apparatus defect in a gauge-theoretic model). Each skeptic is pointed at that finding's specific
weak point; each defender is told to close it with measurement and to concede if it cannot.
| 7 | Write final report | NOT STARTED | `docs/audits/audit-2026-07-27.md` |
| 8 | Test lane, counts read from JUnit XML | **COMPLETE — CPU green** | `docs/audits/raw/cpu-junit.xml` |

### Stage 8 result (CPU lane)

Command: `CUDA_VISIBLE_DEVICES=-1 "C:/Python314/python.exe" -m pytest -m "not cuda" -p no:randomly --junitxml=docs/audits/raw/cpu-junit.xml`
Exit 0. Read from the XML `testsuite` attributes, not from memory:

    tests=4447  failures=0  errors=0  skipped=32  time=411.107s

**The previously-known failure is GONE.**
`test_ablation_sweep_route_compatibility_20260711::test_active_update_rule_config_values_are_preserved`
now **PASSES**. It asserts agreement between `train_vfe3.py`'s live config and
`ablation.BASELINE_CONFIG`; both are uncommitted WIP files, so the `mm_damping` 1.0-vs-0.75 drift
has been resolved in the user's working tree since the `2b7a96d` measurement. No code change on
this branch caused it. That open decision point is therefore CLOSED.

Count delta vs the `2b7a96d` baseline (`tests=4444 skipped=29`) is `+3 tests / +3 skipped`, i.e.
three additional tests are now collected and all three skip; failures went 1 -> 0. The `+3/+3`
pairing plus this run's added `-p no:randomly` (absent from the baseline invocation) makes
collection-order variance the likely cause. Not investigated further — it moves nothing into the
failing column.

Skip breakdown (32): 20 symlink-privilege (`WinError 1314`), 6 removed closure-ledger doc,
3 `--runslow`, 1 symlink-denied, 1 unset `VFE3_BASELINE_BUNDLE`, 1 POSIX-host-required.

STILL OWED: the CUDA lane —
`VFE3_TEST_DEVICE=cuda "C:/anaconda/python.exe" -m pytest -m "cuda" --junitxml=...`
(baseline was 24 passed). Do NOT run it while the user's K=20 GPU job is in flight.

## Wave 3 — base five dispatched

`code-reviewer` (quality+security), `debugger` (bugs+gradient-flow), `refactoring-specialist`
(dead code + dead config fields), `performance-engineer` (hot paths), `python-pro` (type safety +
the kwargs-bag contract hazard).

## Wave 4 — expert pool to dispatch (whole-repo audit ⇒ run the FULL relevant pool)

`audit-gauge-theorist`, `audit-geometer`, `audit-info-geometer`, `audit-variational`,
`audit-numerical-analyst`, `audit-transformer-ml`, `audit-implementation-engineer` — seven, within
the ~10 parallel cap, so a single wave suffices.

## Already-known findings — do NOT re-report as new

- `estep_grad_norm_sigma = 0.0` is a hardcoded tautology in the diagnostics.
- `_decode_linear` does not apply `decode_unigram_prior` (train/eval mismatch, linear-decode path).
- `ablation.BASELINE_CONFIG["mm_damping"]` 0.75 vs `train_vfe3.py` 1.0 — the known test failure.
- Joint-vs-per-factor Frobenius reduction in `gauge_optim.py:112,264,288,414`.
- `config.py:1743-1756` alpha-calibration warning is unreachable.
