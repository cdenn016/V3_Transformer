# Deep Audit — 2026-06-13 (theoretical purity + s_e_step / learnable_r buildout)

Multi-agent deep audit run via the `deep-audit` skill encoded as a deterministic Workflow
(run `wf_914582f1-c56`, 38 agents). Focus, per request: theoretical purity and the recent
`s_e_step` / `learnable_r` / `lambda_h_mode` / `r_update_mode` / `log_lambda_h` buildout (the
features reported to improve test ppl over their `=False` paths).

## Scope

The branch diff `main...HEAD` on `vfe3-fullcov-alpha-roadmap-2026-06-13` (62 files, +7702).
The audit centered on the surface merged **after** the 2026-06-13 ultradeep audit (commit
`1727ad3`) and therefore never previously reviewed:

- `vfe3/lambda_h_i.py` (new registry `hyper_prior_lambda_h`: constant / state_dependent / learnable)
- `vfe3/model/prior_bank.py` (`r_mu`/`r_sigma_log` with `requires_grad=learnable_r`; `barycenter_r_()`)
- `vfe3/model/model.py` (`_refine_s`, `_hyper_prior_kl`/`_hyper_prior_weighted`, `log_lambda_h`, gating)
- `vfe3/train.py` (`build_optimizer` log_lambda_h group; barycenter M-step after `optimizer.step()`)
- `vfe3/config.py` (`r_update_mode`/`lambda_h_mode`/`b0_h`/`c0_h` fields, guards, freeze warnings)
- `vfe3/inference/e_step.py`, `vfe3/gradients/kernels.py`, `vfe3/gradients/oracle.py` (R_h flow)

## Investigators Dispatched

- **Base five:** code-reviewer, debugger, refactoring-specialist, performance-engineer, python-pro.
- **Expert pool (full, theory gate met — whole-repo/whole-theory audit):** audit-variational (×2),
  audit-info-geometer, audit-gauge-theorist, audit-geometer, audit-numerical-analyst,
  audit-transformer-ml, audit-implementation-engineer.
- **Verifier:** one fresh `general-purpose` agent re-read the cited source for all 50 findings.
- **Challenge:** `audit-skeptic` vs `audit-defender` + `general-purpose` adjudicator on the 8
  highest-severity confirmed focal findings; 5 further duplicates of the same issue deferred.

## Headline verdict

**The buildout is theoretically sound and the cardinal-rule pure paths exist.** Every lens
independently confirmed, against executable source (not comments):

| Theory check | Status | Decisive source |
|---|---|---|
| Pure default path byte-identical to pre-buildout | **CONFIRMED** | implementation-engineer measured loss `3.6106016635894775` identical to a `main` checkout with pre-buildout knobs; no `r`/`s`/`log_lambda_h` params created at defaults. `model.py:121,694`, `prior_bank.py:193` |
| A pure path exists for each new toggle value | **CONFIRMED** | constant `lambda_h_mode` == pre-registry weighting; `gradient` `r_update_mode` == pre-toggle AdamW r; `s_e_step=False` == scored path |
| No-NN rule honored | **CONFIRMED** | `nn.Linear`/MLP/activation appear only in comments; `log_lambda_h` is the sanctioned scalar sibling of `log_alpha`; `r_mu`/`r_sigma_log` are prior tables |
| `barycenter_r_()` is the exact forward-KL m-projection (moment-matched diagonal centroid) | **CONFIRMED** | sympy + numeric argmin to ~1e-6–2e-8; law-of-total-variance `prior_bank.py:241-242`. (Caveat: for the *uniform-over-vocab* objective — see L-A.) |
| state_dependent `lambda_h` envelope `c0_h/(b0_h+KL)` + `R_h` cancellation | **CONFIRMED** | sympy `argmin_λ[λD+b0λ−c0logλ]=c0/(b0+D)`, `d/dD[envelope]=λ*`. R_h enters **both** the scored path (`model.py:783-784`) and the s E-step (oracle `e_step.py:249`/`oracle.py:129`; kernel route via the envelope coefficient `kernels.py:275`) |
| learnable `log_lambda_h` trains under the live config | **CONFIRMED** | nonzero grad measured (~−6.7e-4) under `s_e_step=True`+unroll through `_refine_s`; detach footgun warned `model.py:246`, `config.py:1207/1249` |
| Gauge equivariance preserved by `learnable_r`/`barycenter_r_` | **CONFIRMED** | `r` is frame-free `(K,)`, enters only inside `KL(s‖r)` (`model.py:757-760`); barycenter touches only the s tables |
| s E-step tied-flat transport is the correct/pure choice | **CONFIRMED** | `e_phi_lr=0` (phi fixed) + `connection_W` reads belief means not s ⇒ flat phi0-cocycle is the unique transport `model.py:455,483` |
| EM separation under `s_e_step` (no double-count, no label leakage) | **CONFIRMED** | scored hyper-prior/gamma blocks gated off under `s_e_step` (`model.py:694,706`); s trains through refined-s → CE |

**No critical or high finding survived.** The 50 raw findings collapse to **two distinct real
issues, both confined to the opt-in `r_update_mode='barycenter'` M-step**, plus a handful of
low-severity hygiene items. None removes a pure path; the live config (`r_update_mode='gradient'`)
is on the sound path.

## The two substantive findings (both opt-in, neither breaks a pure path)

### A. `r_update_mode='barycenter'` + `s_e_step=True` is a silent, unguarded inexact M-step

Found by 9 lenses (indices 0,2,10,15,24,32,34,36,37,41,43,46,48). **Resolved severity: MEDIUM.**

Under `s_e_step=True` the scored `lambda_h·KL(s‖r)` term is gated **off** (`model.py:694`,
`and not self.cfg.s_e_step`); `r` instead enters the cross-entropy only through the unrolled
`_refine_s` (`model.py:446-453` → `:547-548`, `prior_source='model_channel'`). The closed-form
`barycenter_r_()` snaps `r` each step (`train.py:351-352`, no `s_e_step` predicate) to the argmin
of the *isolated* `sum_v KL(s_v‖r)` block — a term that is **not in the scored loss** under this
regime. The design doc itself scopes the barycenter's "EXACT M-step" guarantee to `s_e_step=False`
and names AdamW-through-unroll as the correct tool for the coupled regime
(`docs/.../2026-06-13-lambda-h-mode-and-r-update-mechanism.md:104-108,185-186`). The only barycenter
guard (`config.py:939`) keys solely on `not learnable_r`; the combination constructs with **zero
warnings** — conspicuous because every sibling inert/inexact hyper-prior combination *is* warned
(`config.py:911-924,928-936,939-944`).

**Adversarial split (informative):** 4 duels UPHELD at medium (the medium camp: the barycenter
optimizes a term absent from the loss → "silent wrong numbers" class), 4 DOWNGRADED to low (the low
camp: `r` is `requires_grad=False`/unscored here, so this is a defensible block-coordinate anchor
and the `gradient` alternative is *also* not the joint argmin). The dispute is purely the
"wrong-M-step vs defensible-anchor" framing; **both sides agree it is reachable, silent, and
inexact relative to the documented guarantee, and that the pure path survives.** Resolved to
**medium** because the path's central selling point ("exact M-step") silently fails with no
diagnostic while all its siblings warn — and the user reports `s_e_step` as a live, ppl-improving
path. The fix is identical either way.

*Fix:* add a `__post_init__` warning when `r_update_mode=='barycenter' and s_e_step` directing the
user to `r_update_mode='gradient'` for the coupled regime.

*Live-config note:* the user's active config runs `r_update_mode='gradient'` — the **correct** tool
— so this is latent for them; it bites only on a future toggle flip.

### B. `barycenter_r_()` averages uniformly over the vocab table, not the frequency-weighted scored objective

Found by 11 lenses (indices 1,4,19,27,31,33,35,38,40,42,47); flagged `contradiction:true` (the
lenses disagree on whether it is a defect). **Resolved severity: LOW**, but theoretically the more
interesting item.

`barycenter_r_()` computes `r_mu = s_mu.mean(0)` over the full `(V,K)` table — uniform weight
`1/V` per vocab row (`prior_bank.py:241-242`). The scored term it stands in for reduces with
`.mean()` over `(B,N)` **token occurrences** (`model.py:797`) = frequency-weighted
`sum_v f_v KL(s_v‖r)`. sympy: the frequency-weighted argmin is `(Σ f_v μ_v)/(Σ f_v)`, which equals
the uniform mean only for a uniform token distribution. Numeric probe (Zipfian, finding #35):
objective `67.1` at the uniform barycenter vs `9.9` at the frequency barycenter. So the design doc's
"exact M-step optimum" claim (and the `prior_bank.py:236` docstring) is **exact for a different
objective than the one scored** — even in the `s_e_step=False` regime — unless tokens are uniform.

This is a genuine theory question about *which population the centroid targets*. The "no error"
camp (#19,#27,#40) is right that the code is the exact argmin of the uniform-vocab objective; the
"defect" camp is right that this is not the argmin of the frequency-weighted loss `r` appears in.
Not a pure-path break: `r_update_mode='gradient'` descends the correct frequency-weighted objective.

*Fix:* either frequency-weight `barycenter_r_()` by empirical token counts to match the scored
objective, or restate the docstring/spec/`config.py:218` claim as a *uniform-over-vocabulary
empirical-Bayes centroid* (an explicitly different objective), not "the exact M-step optimum."

## Low-severity confirmed items (hygiene; none affects a pure path)

| # | Item | Location | Fix |
|---|---|---|---|
| 3 | `log_lambda_h` freeze warning checks `cfg.detach_e_step`, not `cfg.effective_e_step_gradient`; the `e_step_gradient='detach'` (with `detach_e_step=False`) route freezes it silently (config-level warning `config.py:1203-1207` partially covers it) | `model.py:246` | check `effective_e_step_gradient=='detach'` |
| 39 | Under `r_update_mode='gradient'`+`learnable_r`+`s_e_step`, the oracle-detach (non-kernel) route can silently freeze `r_mu`/`r_sigma_log`/`log_lambda_h`; the freeze warning names `log_lambda_h` but not `r_mu`/`r_sigma_log` | `config.py:1241-1264` | add `r_mu`/`r_sigma_log` to the oracle-detach freeze warning |
| 45 | Comment claims the s E-step adds `R_h` via `alpha_reg`, but the live kernel uses the envelope coefficient and never adds `alpha_reg` (result still correct by the envelope theorem; the **comment lies** — CODE-FOCUS mandate) | `model.py:459` | reword: envelope coef makes the kernel grad equal the total derivative without literally adding `R_h` |
| 49 | Under `state_dependent` `lambda_h`, `cfg.lambda_h` (=0.25 in the live config) does **not** scale the coupling — only `c0_h/b0_h` do (by design, mirrors the alpha registry, but the live `0.25` misleads) | `model.py:463`, `alpha_i.py:73-84` | document that magnitude is set by `c0_h/b0_h` under state_dependent |
| 44 | barycenter (unclamped) vs scored `KL(s‖r)` (`kl_max`-clamped, `model.py:759`) optimize inconsistent objectives at large drift | `prior_bank.py:242` | document the unclamped target, or robustify consistently |
| 17 | `r_sigma = exp(r_sigma_log)` re-allocated every `_refine_s` call on the live `s_e_step` path | `model.py:447` | cache the `(K,)` exp / detach under barycenter mode |
| 18 | `encode_s` called twice per forward when `gamma_coupling>0` and `lambda_h>0` and not `s_e_step` | `model.py:754,820` | factor the `encode_s` call out, pass `(s_mu,s_sigma)` |
| 25 | `learnable` mode adds a zero-reg tensor (`alpha_learnable` returns zeros) via the `!= 'constant'` branch | `model.py:783` | gate the add on `== 'state_dependent'` |
| 26 | local `kl` shadows the `vfe3.divergence.kl` symbol name | `model.py:777` | rename to `kl_s`/`kl_val` |
| 12 | `log_lambda_h` param can be orphaned if a user suppresses the inert-channel warning (`lambda_h==0`+learnable+not s_e_step) | `model.py:244` | assert `lambda_h>0` when creating `log_lambda_h` |
| 14 | Under non-constant modes the logged F-trajectory includes `R_h` (correct descent object, but may surprise diagnostics readers) | `e_step.py:249` | document |
| 13 | barycenter variance formula correct; local `s_sigma` is variance not std (naming only) | `prior_bank.py:240` | optional rename to `s_var` |

## Verifier / challenge summary

- 50 findings: **49 CONFIRMED, 1 REFUTED, 0 inconclusive.**
- **REFUTED (#16):** "learnable `log_lambda_h` gradient dead under `s_e_step=False`+`detach_e_step=True`"
  — contradicted by its own evidence; the scored `_hyper_prior_term` (`model.py:705`) is outside the
  `no_grad` E-step wrapper, so `log_lambda_h` trains. Behavior correct.
- Many CONFIRMED findings are positive confirmations (#20–#23, #28–#30) that theory-critical paths
  are correct — useful negative results, not defects.
- Challenge tier: 8 duels on the focal guard-gap finding; verdict split 4 UPHELD-medium / 4
  DOWNGRADED-low (same underlying issue, resolved to medium above). 5 further duplicates deferred
  (indices 37,41,43,46,48 — all the same `barycenter+s_e_step` guard gap).

## Surviving Punch List (post-challenge, ranked)

1. **[medium]** Guard `r_update_mode='barycenter'` + `s_e_step=True` — `config.py:939` /
   `train.py:351-352`: add a `__post_init__` warning (closed-form barycenter is the exact M-step
   only when `s_e_step=False`; recommend `r_update_mode='gradient'` for the coupled regime).
2. **[low]** Reconcile the barycenter population — `prior_bank.py:241-242` / `config.py:218`:
   frequency-weight `barycenter_r_()` to match the scored objective, **or** restate the "exact
   M-step optimum" claim as a uniform-over-vocab empirical-Bayes centroid.
3. **[low]** Freeze-warning correctness/coverage — `model.py:246` (use `effective_e_step_gradient`);
   `config.py:1241-1264` (add `r_mu`/`r_sigma_log` to the oracle-detach freeze warning, #39).
4. **[low]** Fix the misleading comment at `model.py:459` (#45) and document the `state_dependent`
   magnitude knob `c0_h/b0_h` vs the inert `lambda_h` (#49).
5. **[low]** Hygiene/perf: cache `r_sigma` exp (#17), de-dup `encode_s` (#18), zero-reg short-circuit
   on learnable (#25), rename shadowed `kl` (#26), assert `lambda_h>0` for `log_lambda_h` (#12).

No fixes applied — punch list presented for authorization per the deep-audit workflow.

## Test Suite

- Command: `python -m pytest --junitxml=... -p no:cacheprovider` (full suite)
- Result: **963 passed, 0 failures, 0 errors, 0 skipped** (143.8s; read from junitxml `testsuite tests=963 failures=0 errors=0`).
- The clean suite includes the buildout's own pins (`test_hyperprior.py`, `test_ultradeep_fixes_2026_06_13.py`,
  `test_run_diagnostics_2026_06_13.py`) — the surviving punch-list items are guard/diagnostic gaps and
  overclaims, not behaviors any existing test asserts.
