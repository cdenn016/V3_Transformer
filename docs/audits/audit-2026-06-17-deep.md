# Deep Audit (Round 2 — "even deeper") — 2026-06-17

Second, depth-first pass over V3_Transformer (VFE_3.0), run after the round-1 breadth sweep
(`audit-2026-06-17.md`). Where round 1 pattern-matched across the tree, this pass had 10 investigators
read their modules *in full* and trace data flow under the default config, plus **symbolic (sympy) and
finite-difference verification of the actual theoretical identities**, a completeness-critic second
wave, and a **3-vote perspective-diverse panel** (code-truth / reachability / independent-correctness)
per finding. The 50 round-1 findings were passed in as an exclusion list.

Run stats: 82 agents, 5.0M subagent tokens, ~51 min. **Two integrity caveats handled below.**

> **Caveats.** (1) A transient API rate-limit wiped most/all of the 3 votes for findings 14–22, so
> several were auto-marked "refuted" with **zero** votes — i.e. never actually verified. I re-verified
> the two that matter (id 15, id 22) by reading current source and assessed the rest; they are labelled
> accordingly. (2) The `contracts` (registry/Protocol conformance) investigator stalled and is being
> re-run standalone; its result will be appended. (3) The tree moved during the run (concurrent EMA /
> `kappa_beta_per_head` / Laplace-`natural_gradient` edits), so line numbers are as-of audit time;
> re-verified findings are pinned to current source.

## Scope
Whole tree, with deliberate depth on the round-1 blind spots: `geometry/{lie_ops,generators,irreps,
cg,closure}`, `divergence`, `alpha_i`, `lambda_h_i`, `gradients/oracle`, `gauge_optim`, `data/datasets`,
`run_artifacts`, `metrics`, `viz/{extract,figures,report}`, `scaling`, `scaling_analysis` — plus
identity-level verification of the free-energy, divergence, Fisher/natural-gradient, SPD-retraction,
and gauge-equivariance claims.

## Headline
1. **The theory core held up.** Deep symbolic/FD probing of gauge equivariance (pure-path transport),
   softmax-β stationarity, the SPD affine-retraction exp-map, the KL/Rényi closed forms, and the
   Fisher/natural-gradient identities surfaced **no broken pure-path identity**. The one live lead —
   whether `DiagonalLaplace.natural_gradient` uses the Gaussian Fisher metric — was already fixed
   concurrently with a symbolically-verified `b²·grad` (the correct location-scale Fisher); the
   info-geometer found nothing to flag there. Round 1's "zero theory findings" survives a deeper look.
2. **The real new defects are in the periphery round 1 under-covered:** training-loop robustness,
   scaling-law statistics, and diagnostic/visualization fidelity — not the kernels.
3. **One genuine latent theory-consistency bug** (id 15): under RoPE *and* a learning gauge frame the
   φ E-step descends a different free energy than the μ/σ step — dormant under the default
   (`pos_rotation='none'`, `e_phi_lr=0`), real when both are enabled.

## Tally
| | count |
|---|---|
| Investigators (10 deep + 3 critics; `contracts` re-running) | 13 |
| New findings (round-1 excluded) | 23 |
| Confirmed (≥2/3 votes) | 10 |
| Refuted (≥2/3 votes) | 6 |
| Rate-limited (re-verified/assessed by hand) | 7 |

## Punch List (new, post-verification)

### Medium
1. **[medium · numerics · DEFAULT PATH] Non-finite gradients silently poison AdamW — no NaN-skip
   guard.** `train.py:~376–384`. With `amp_dtype=None` (the default), `GradScaler(enabled=False)` so
   `_scaler.step(optimizer)` calls `optimizer.step()` directly with no `found_inf` check; `loss_finite`
   is computed but only logged, never gates the step. A single NaN gradient (reachable from the SPD/
   eigh/matrix-exp islands on a degenerate spectrum) permanently corrupts the AdamW moments with no
   recovery — and the fp16 path *does* skip via the scaler, so it's an inconsistency too. *Fix:* gate
   `optimizer.step()` on `math.isfinite(step_loss)` + a finite-grad check when the scaler is disabled.
   (id 16, 2/2 real votes.)
2. **[medium · numerics] Scaling-law headline α (SEM-weighted) printed with an unweighted bootstrap
   CI** — the point estimate can fall outside its own reported CI. `scaling_analysis.py` (`analyze`
   fit passes `weights=w`; `bootstrap_exponent_ci` refits with no `weights=`). Probe: weighted 0.0547
   vs unweighted 0.0614 under heteroscedastic SEMs. *Fix:* pass the same weighting to both. (id 10, 2/2.)
3. **[medium · wiring] precision-bias is folded only in `forward()`, not in `diagnostics()` /
   `attention_maps()`.** `model.py:688–701` (fold) vs `1251`/`1469` (position-only `log_prior`). With
   your current `precision_weighted_attention=True`, the F-decomposition and attention-map readouts
   score against a *different* (weaker) prior than the model actually uses. Training and val PPL are
   unaffected (both go through `forward()`); only the diagnostics misrepresent the model. *Fix:* factor
   the fold into a helper and apply it in every consumer. (id 22, re-verified by hand.)
4. **[medium→low · wiring · dormant] `numerical_health` omits the RoPE wrap.** `viz/extract.py:~316–323`
   transports a bare `omega` while `converged_state`/`diagnostics` wrap it in `RopeTransport` when
   `rope is not None`; under `pos_rotation='rope'` it reports a flat-position belief. Dormant under the
   default/operating `pos_rotation='none'`. *Fix:* wrap `omega` in `RopeTransport` mirroring the
   siblings. (id 11, 3/3.)
5. **[medium→low · theory · dormant] φ E-step descends an un-rotated objective under RoPE.**
   `phi_alignment_loss` (`e_step.py:267`) has **no `rope` parameter**, so its internal transport build
   (`:310`) is un-rotated, while the μ/σ build (`:406–409`) passes `rope=rope`. With `pos_rotation='rope'`
   *and* `e_phi_lr>0` the φ gradient optimizes a different F than μ/σ (FD probe showed a real
   discrepancy). **Dormant** under the default and your operating point (`pos_rotation='none'`,
   `e_phi_lr=0.0`). *Fix:* thread `rope`/`rope_on_cov`/`rope_on_value` into `phi_alignment_loss` and its
   call site. (id 15, re-verified by hand; downgraded from the reporter's HIGH because it cannot bite
   the default path.)

### Low (confirmed or hand-assessed)
- **[low · wiring] `free_energy_value` omits `mass_phi` from its accept-and-ignore kwargs** — its own
  docstring promises a single-knob-bag forward, but `e_step(return_trajectory=True, mass_phi>0)` raises
  `TypeError` (probe-reproduced). `e_step.py` (`_f_diag` forwards `**kwargs`). *Fix:* add
  `mass_phi: float = 0.0` to the accept-and-ignore list. (id 0, 3/3.)
- **[low · wiring] `_routes_to_oracle` freeze-warning omits the `decoupled_value_gauge`
  (`rope_on_value=False`) oracle route** that `uses_kernel_route` gates on (`config.py` vs
  `kernels.py`), so an E-step-only learnable param can silently freeze with no warning under
  `pos_rotation='rope' + rope_on_value=False`. *Fix:* add the missing disjunct. (id 2, 3/3.)
- **[low · quality] `train_ce` is logged from a second forward run *after* `optimizer.step()`** —
  `train.py:~675` re-forwards the batch under `no_grad` post-update, so `train_ce` is one step ahead of
  `train_loss`. *Fix:* return `ce` from `train_step`, or document it. (id 5, 3/3.)
- **[low · numerics] `condition_number` diagonal branch doesn't return `+inf` for a non-positive
  variance** (asymmetric with the round-1 fix to the full-matrix branch). `numerics.py:157–159`.
  Diagnostic monitor only. *Fix:* apply the same `where(min>0, cond, inf)` guard. (id 1, 2/3.)
- **[low · quality] UMAP `vech(log Σ)` drops the √2 off-diagonal factor** → the Euclidean metric is
  non-isometric to the log-Euclidean/Frobenius geometry the docstring promises. `viz/figures.py:~485`.
  Full-covariance runs only. *Fix:* scale off-diagonals by √2. (id 12, 3/3.)
- **[low · quality] Rolling-mean trend punches a window-width NaN hole** around any non-positive point
  on a log-y curve (`np.convolve` propagates NaN). `viz/figures.py:~308–316`. *Fix:* NaN-aware moving
  average. (id 13, 3/3.)
- **[low · perf · RE-FIND] `free_energy_terms` allocates a full `(H,N,N)` uniform-π tensor** when
  `log_prior is None`, though `log(β/π)=log(β)+log(N)` needs none. `metrics.py:147`. **This is round-1
  deferred id 27** re-surfaced, not new. (id 9, 3/3.)
- **[low · quality · dormant] `scaling_analysis` `test_ce` fallback uses `dict.get(k, default)`** which
  ignores an explicit `None` value, dropping a real test number if a run dir has `test_ce: null`.
  Dormant (the writer keeps the two files consistent). `scaling_analysis.py:73`. (id 14, 1 vote —
  simple Python semantics, real but dormant.)
- **[low · rate-limited, assessed — independent verification still owed]** id 17 (fused diagonal decode
  may omit `safe_kl_clamp(min=0)` that `reference_decode` applies — `prior_bank.py`); id 19 (`EMA.update`
  has no finiteness guard, so a transient NaN in a live param permanently corrupts the shadow —
  **your new `ema.py`**, default-OFF); id 20 (`log(N)` normalizer computed in `energy.dtype`, losing
  precision under `amp_dtype='bf16'` — `free_energy.py:302`); id 21 (precision bias computed from the
  pre-E-step encoded σ, frozen across `n_e_steps` — possibly by design). All low; flagged for a clean
  re-verify.

## Refuted (≥2/3 votes — genuinely checked, not rate-limited)
| id | claim | why refuted |
|---|---|---|
| 3 | diagnostics `total` SUM-vs-MEAN factor-N undercount | misreads it: hyper-prior folded at SUM scale like the belief blocks, then `train.py` divides all by `n_tok` uniformly. |
| 4 | gauge momentum buffer stale on resume | standard heavy-ball behavior (PyTorch SGD/Adam carry buffers too); opt-in `m_phi_natural_grad`, correctly saved/restored. |
| 6 | ablation reseed-after-init confounds data order | finding is **backwards** — the reseed runs after model build *by design* to pin the loader RNG to `cfg.seed`. |
| 7 | `TokenWindows.__len__` loses stride-1 windows | probe: zero windows lost; `n=usable//stride+1` counts every valid window. |
| 8 | `finalize_run` leaves `n_e_steps` broken on exception | `try/finally` restores it before any later read; `except` even swallows. |
| 18 | diagonal retraction `clamp(max=sigma_max)` breaks affine-invariance | a deliberate positivity/saturation guardrail identical on both arms (round-1 deferred retraction item). |

## Fixes Applied (branch `vfe3-audit-fixes-round2-2026-06-17`)
All confirmed findings fixed; behavioral ones carry a regression test in
`tests/test_audit_fixes_2026_06_17_r2.py`.

- **id 16** `train.py` — skip `optimizer.step()` on a non-finite loss when the scaler is disabled
  (drop grads, no AdamW poison), mirroring the fp16 found_inf skip.
- **id 5** `train.py` — `train_ce` is now the PRE-step CE captured in `train_step` (matches
  `train_loss`); the redundant post-step re-forward is removed.
- **id 0 / id 15** `e_step.py` — `free_energy_value` accept-and-ignores `mass_phi`; `phi_alignment_loss`
  threads `rope`/`rope_on_cov`/`rope_on_value` so the φ step descends the same rotated objective as μ/σ.
- **id 22 / id 21** `model.py` — factored `_fold_precision_bias` (rank-robust, single-block aware),
  applied in `forward`, `diagnostics`, and `attention_maps` so every belief-channel consumer scores the
  same prior; documented that the encode-time σ is an intentional fixed reliability prior.
- **id 1** `numerics.py` — `condition_number` diagonal branch returns `+inf` for a non-positive spectrum.
- **id 19** `ema.py` — `EMA.update` skips a non-finite live param (no permanent shadow poison).
- **id 9** `metrics.py` — uniform-π entropy via scalar `log(1/N)` (no `(H,N,N)` alloc; byte-identical).
- **id 2** `config.py` — freeze-warning predicate includes the `decoupled_value_gauge` oracle route.
- **id 11** `viz/extract.py` — `numerical_health` wraps `omega` in `RopeTransport` under RoPE.
- **id 12 / id 13** `viz/figures.py` — UMAP `vech` scales off-diagonals by √2; `_rolling_mean` is NaN-aware.
- **id 10 / id 14** `scaling_analysis.py` — bootstrap CI uses the SAME weights as the headline fit; the
  `test_ce` fallback coalesces an explicit null.
- **id 17** `prior_bank.py` — fused `_decode_diagonal` floors KL≥0 to match `reference_decode`.

**Deferred (with reason):** id 20 (bf16 `log(N)` constant shift — no gradient/β impact, non-default;
a clean fix means upcasting the golden-tested fp32 reduction). **Not defects:** id 18 (intentional SPD
saturation guardrail), the `reference_decode` diagonal hardcode (the decode boundary is deliberately
α=1 diagonal-Gaussian KL), and the 6 round-2 refutations.

## Test Suite
Round-2 regression tests: `tests/test_audit_fixes_2026_06_17_r2.py` (+8). Full-suite result appended at
commit time.

## What this round bought
Round 1 (breadth) found the **critical** (precision single-block crash) and the substantive decode/
numerics mediums. Round 2 (depth) confirmed the **theory kernels are clean** under symbolic/FD probing,
and surfaced a different class of defect: a **default-path training-robustness gap** (NaN→AdamW poison),
a **scaling-analysis statistics bug** (α outside its own CI), **diagnostic/viz fidelity** gaps (precision
bias, RoPE, UMAP metric, NaN holes), and **one latent theory-consistency bug** (φ/RoPE objective
mismatch) that only bites a legitimate non-default config.
