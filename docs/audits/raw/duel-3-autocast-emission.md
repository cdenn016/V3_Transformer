# Duel 3 — `mm_exact_update` autocast re-entry drops the emission factor

## SKEPTIC (attack) — returned 10:36 CDT

**Verdict argued: DOWNGRADE critical -> HIGH.** Explicitly states it **confirms the mechanism
end-to-end and cannot break it**; attacks only the severity classification.

### Its measured reproduction (CUDA, K=4, V=16, N=4, `use_prior_bank=False`, `mm_exact`)
Instrumented every `mm_exact_update` entry as `(autocast_enabled, mu.dtype, emission_passed, weight)`:
```
amp_dtype=None    separate w=0.5: [(False,'float32',True,0.5)]     loss off=2.774196147918701 on=2.773625373840332
amp_dtype='bf16'  separate w=0.5: [(True,'float32',True,0.5),
                                   (False,'float32',False,0.0)]    loss off=on=2.774196147918701  identical=True
amp_dtype='bf16'  shared   w=0.5: same two frames                  loss off=on=2.774196147918701  identical=True
```
The island IS entered on the live device path (outer frame `autocast_enabled=True`); the nested
frame arrives with `emission_passed=False, weight=0.0`; the CE is bit-identical to
`emission_mode='off'`.

### Attacks that FAILED (all of them, reported by the skeptic)
- *Reachability / island not entered*: **failed** — outer frame shows `autocast_enabled=True`.
- *Compensation elsewhere*: **failed** — the only consumer of the tuple is `kernels.py:692`;
  `e_step.py:471` explicitly accepts-and-ignores `emission_weight` in the diagnostic F.
- *A test already covers it*: **failed** — `tests/test_emission_factor_20260726.py` contains **no
  `amp`/`autocast` reference at all**.
- *Validation warns on the combination*: **failed** — `config.py:1648-1672` fails closed on the
  route and on `emission_weight==0.0` inertness but says nothing about `amp_dtype`; the only
  amp-vs-config warning (`config.py:2428`) covers link transports.

### Attacks that LANDED (severity only)

1. **Classification precedent.** `docs/audit-results.md:61-63` rates A-01 —
   `gaussian_frame_diagonal`'s two transport seams discarding `omega` unconditionally — "critical
   (geometer) / high (three other lenses)". That defect sits in the **identical reachability class**
   (opt-in, default-off, one-line-edit selection) and its failure mode is **strictly worse**:
   measured `loc err=3.79, diag(var) err=31.5` — wrong, plausible, believable numbers. The emission
   drop yields an **exact no-op**: `mu*` remains the valid closed-form minimizer of the emission-off
   objective and the two arms agree to all 16 digits — the loudest signature short of a raised
   exception. Three of four lenses said high for the worse defect.

2. **Blast radius is nil today.** The string `emission` occurs **nowhere** in `train_vfe3.py` or
   `ablation.py`, including the ablation axis registry (`ablation.py:1061` has an `amp_dtype` axis
   but no emission axis). No existing result and no queued run is contaminated; the pure path is
   untouched.

3. **HALF the enable surface is already instrumented.** Measured under `emission_mode='separate'` +
   bf16, `parameter_report` (`train.py:2067`) returns
   `dead_names=['prior_bank.decode_log_scale','prior_bank.emission_proj_weight']` (dead 1 -> 65
   numel), which `train_vfe3.py:491-493` prints in the startup banner as
   **`dead under config (no grad): emission_proj_weight`**.

### What the skeptic CONCEDED (and why it argued downgrade, not drop)
> "I will not overstate it: under `emission_mode='shared'` — legal here because
> `train_vfe3.py:151` sets `use_prior_bank=False`, so the `config.py:1658` guard does not fire — I
> measured `dead_names` unchanged at `['prior_bank.decode_log_scale']`, because the shared table
> stays live off the decode. **That half is genuinely undetected**, which is why this is a downgrade
> and not a drop."

**PRACTICAL CONSEQUENCE FOR THE USER: running the `separate` arm will print a dead-parameter warning
in the startup banner. Running the `shared` arm will print NOTHING and silently measure a no-op.**

### Skeptic's decisive evidence
`docs/audit-results.md:61-63` (same-class, strictly worse defect rated high by three of four lenses)
against `kernels.py:573-594` + the measured `train_vfe3.py:491-493` banner detection under
`emission_mode='separate'`.

## DEFENDER — returned 10:47 CDT

**Verdict argued: UPHELD — pre-launch blocker (High on enable; not a live-run defect).**

### AST enumeration of both islands (repo rev 3b3a51e)
```
belief_gradients  (def 327): 22 kwonly params, island call at :407 forwards 22 -> MISSING: []
mm_exact_update   (def 509): 17 kwonly params, island call at :575 forwards 15 -> MISSING: ['emission_weight', 'emission']
```
`kernels.py:692` is the **sole** consumer of `emission` in the package.

### Measured on this machine (torch 2.10.0.dev20251210+cu128, CUDA available)
Kernel level, N=4, K=4, identity omega, synthetic Bohning `(d,g)`:

| device | context | `\|\|mu*(em on) - mu*(em off)\|\|` | `torch.equal` |
|---|---|---|---|
| cpu | no-amp | 2.558176 | False |
| cpu | bf16 autocast | **0.000000** | **True** |
| cuda | no-amp | 2.558176 | False |
| cuda | bf16 autocast | **0.000000** | **True** |

End-to-end tiny model (V=16, K=4, N=4, H=2, L=1, `emission_mode="separate"` held fixed in BOTH arms
so parameter allocation and RNG are identical; only `emission_weight` varies 0.0 -> 2.0):

| device | `amp_dtype` | `\|\|delta mu_final\|\|` | CE (w=0 -> w=2) |
|---|---|---|---|
| cpu | `None` | 0.968486 | 2.76476479 -> 2.71880364 |
| cpu | `'bf16'` | **0.000000** | 2.76476479 -> **2.76476479** |
| cuda | `None` | 1.246478 | 2.76750231 -> 2.75906205 |
| cuda | `'bf16'` | **0.000000** | 2.76750231 -> **2.76750231** |

### THE SHARPEST NEW EVIDENCE — the existing tests FAIL if AMP is switched on
Standalone replica of `test_live_emission_moves_the_belief` (`:137`) and
`test_the_readout_table_receives_an_e_step_gradient` (`:145`), same `BASE`, `amp_dtype` added:
```
[amp_dtype=None] moves_the_belief[  shared]: 2.76476479 -> 2.76373315   PASS
[amp_dtype=None] moves_the_belief[separate]: 2.76476479 -> 2.74875379   PASS
[amp_dtype=None] readout_table_gradient[separate]: grad=6.530e-02       PASS
[amp_dtype=bf16] moves_the_belief[  shared]: 2.76476479 -> 2.76476479   FAIL (no-op)
[amp_dtype=bf16] moves_the_belief[separate]: 2.76476479 -> 2.76476479   FAIL (no-op)
[amp_dtype=bf16] readout_table_gradient[separate]: grad=None            FAIL
```
**`grad=None` is the sharpest number.** In `separate` mode the E-step is the ONLY path to
`emission_proj_weight`, so under AMP autograd never reaches it — yet
`tests/test_emission_factor_20260726.py:162-169` pins that this table sits in an optimizer group.
**The run therefore also carries a permanently-frozen parameter that the optimizer steps over.**

### Test-coverage gap, enumerated
`emission` appears in exactly one test file; grep for `autocast|amp|bfloat16|float16` in it returns
**no matches**, and its `BASE` (`:19-20`) leaves `amp_dtype` at the `None` default. Conversely
`tests/test_amp.py` has 11 tests and none mentions `mm_exact` or `emission`.
`tests/test_audit_fixes_2026_07_26.py` contains both autocast tests (`:53,63,76,98,115`) AND
`mm_exact_update` tests (`:145,155`) — **but they are disjoint.** The `mm_exact` fp32 island has no
test of its own, with or without emission.

### It defeats an invariant the project already wrote in executable code
`config.py:1648-1651` raises on `e_step_update != "mm_exact"` with the stated rationale "Fail closed
rather than run silently inert... on the gradient/oracle route it would be accepted and dropped,
which is exactly the class of dead-config defect this project keeps finding after the fact."
**The AMP island reinstates precisely that failure through a door the validator does not watch.**
`config.py:1664-1672` even warns at `emission_weight == 0.0` that a byte-identical-to-`off` run is
unacceptable — and under bf16 the NONZERO-weight run is byte-identical to `off`.

### Falsification conditions, all probed
1. Autocast not live on the E-step device path — **falsified** (measured `True`, delta = 0.000000).
2. Another consumer compensates — **falsified** (`kernels.py:692` is the only read).
3. A test covers it — **falsified** (zero matches both directions).
4. `config.py` fails closed on the combination — **falsified** (`:1648-1672` checks only
   `e_step_update`, `use_prior_bank`, zero weight).
5. **The emission tuple would be dtype-invalid in the fp32 re-entry, so the fix is not a simple
   forward** — **falsified, and probed specifically as the honest way the finding could shrink.**
   Built at the `model.py:1126` seam INSIDE `torch.autocast('cpu', bfloat16)`, `_emission_terms`
   returns `d.dtype = torch.float32, g.dtype = torch.float32` (`emission.py:132` seeds `g` from
   `torch.zeros_like(mu_p)`, fp32 wins promotion). Passing the tuple verbatim works; a deliberately
   bf16 tuple also works at 1.08e-3 relative error. **The fix is a two-keyword forward, no dtype
   plumbing.**
6. The arms are intended to run at `amp_dtype=None` — **the one live route to a downgrade, and a
   config decision rather than a code fact.**

### Defender's explicit scoping
> "I am **not** claiming: that any shipped run is currently wrong; that the finding affects the
> gradient E-step route (`belief_gradients` forwards everything); or that anything is broken at
> `amp_dtype=None`." Also: "grep for `emission` in `ablation.py` returns nothing, so no emission arm
> is registered in the sweep yet either. The 'pre-registered next experiment' is context from the
> brief, not something I can verify in the repo."

## ADJUDICATION — **UPHELD at HIGH (downgraded from critical). Pre-launch blocker.**

**Both sides independently converged on HIGH**, from opposite directions — the skeptic tried four
attack lines, reported all four failed, and argued down from critical purely on classification
precedent (`docs/audit-results.md:61-63`, where a strictly worse same-class defect was rated high by
three of four lenses); the defender argued up from "off by default" on failure-mode-on-enable. No
compromise was needed; the cited source agrees with both.

Downgraded from the orchestrator's CRITICAL because it is **not a currently-active defect**:
`emission_mode` defaults to `'off'`, `train_vfe3.py` never sets it, and `emission` appears nowhere in
`ablation.py`. **No shipped result is contaminated.**

Held at HIGH rather than medium because the failure mode on enable is the worst kind: an exact,
silent no-op that produces a plausible null result, with the fail-closed config check *passing* and
thereby signaling the config is sound. Three independent facts seal it — zero AMP test coverage on
this seam, the existing pins provably FAIL when AMP is switched on, and `emission_proj_weight.grad`
is `None` so the `separate` arm also silently carries a frozen parameter.

**Operational split the skeptic surfaced and the defender did not contradict:** running `separate`
prints `dead under config (no grad): emission_proj_weight` in the startup banner
(`train_vfe3.py:491-493`); running `shared` prints **nothing**, because the shared table stays live
off the decode. So one arm self-reports and the other hands back a clean-looking null.

**Fix is confirmed trivial and dtype-safe:** forward the two keywords at `kernels.py:575`. The
defender probed the dtype objection specifically and it does not hold.
