# Verifier A — HIGH / CRITICAL-band findings

Returned 2026-07-27 ~10:05 CDT. Independent `general-purpose` verifier. Probes re-run under
`C:/anaconda/python.exe` (torch 2.10.0.dev+cu128, CUDA available). No repo file modified.

## Live config facts the verifier established (from the uncommitted `train_vfe3.py`)

`embed_dim=210`, `n_heads=7`, `max_seq_len=128`, `batch_size=16`, **`n_layers=1`**, `n_e_steps=1`,
`family="gaussian_diagonal"`, **`e_step_update="mm_exact"`** (:408), **`amp_dtype='bf16'`** (:392),
**`e_phi_lr=0.00`** (:326), **`e_mu_q_trust=None`** (:385), **`use_prior_bank=False`** (:151),
**`decode_mode='diagonal_chunked'`** (:154), `transport_mode="flat"`, `gauge_group="block_glk"`,
`gamma_as_beta_prior=True` (:431), `lambda_gamma=0.75`, `gamma_prior_weight=0.5` (:433),
`beta/gamma_attention_prior="causal_alibi_noself"`, `alibi_slope=1`, `kl_max=8*210=1680` (:473).
**`emission_mode` never appears in `train_vfe3.py`** -> dataclass default `"off"` (config.py:800).

## Verdict table

| # | Source | Finding (short) | Verdict | Reachable? | Evidence |
|---|---|---|---|---|---|
| NA-1 | numerical-analyst | `mm_exact_update` autocast island drops emission | **CONFIRMED** | only under `emission_mode!='off'` (live = off) | `kernels.py:573-594` vs `:692` |
| NA-2 | numerical-analyst | fp64 escalation tests sign, not accuracy | **CONFIRMED** | live route, conditional on \|\|phi\|\| | `transport.py:1428-1431`, used `:2619`, `:2714` |
| NA-3 | numerical-analyst | fp32 self-KL flips saturation mask | **CONFIRMED (mechanism) / UNREACHABLE** | no — gradient route only | `gaussian.py:47-53` -> `kernels.py:156-160`; gate `e_step.py:1001`/`1060` |
| V-1 | variational | `free_energy_value` drops the emission block | **CONFIRMED** | only under `emission_mode!='off'` | `e_step.py:471-472` |
| G-1 | gauge-theorist | E-step skips the `Psi_L(ad_phi)` trivialization | **CONFIRMED** | no — `e_phi_lr=0.00` | `e_step.py:1195-1205` vs `phi_preconditioner.py:581` |
| T-1 | transformer-ml | ALiBi log-prior floored at -27.6 nats | **CONFIRMED** | **YES — LIVE** | `model.py:2396-2399`, `log_eps=1e-12` at `:2355` |
| T-2 | transformer-ml | (dup) autocast drops emission | **CONFIRMED — agrees with NA-1** | as above | `kernels.py:573-594` |
| IG-1 | info-geometer | KL clamp is the objective; "no unclamped path" | **INCONCLUSIVE (SPLIT)** | clamp gating: yes; "no path": **REFUTED** | `config.py:918`, `base.py:26`, `kernels.py:157/179` |
| IG-2 | info-geometer | `route_grow_k` inherits K-independent `kl_max` | **CONFIRMED** | only when running `scaling.py` grow_K | `scaling.py:479`, `:499-501` vs `:529/560/595` |
| IE-1 | impl-engineer | `gaussian_frame_diagonal` + `e_phi_lr>0` crashes | **CONFIRMED** | no — live family/lr exclude it | `e_step.py:1192`; config grep = 0 |
| IE-2 | impl-engineer | (dup) Bohning anchored at layer-0 prior | **CONFIRMED — agrees with V-2** | needs emission ON **and** `n_layers>1` (live = 1) | `model.py:1126` -> `stack.py:142/151` -> `kernels.py:701` |
| CR-1 | code-reviewer | emission `p_0` never detached w.r.t. `W` | **CONFIRMED** | only under `emission_mode!='off'`, weight>0 | `emission.py:110`, `:138-139` vs docstring `:99-100` |
| DB-1 | debugger | full-cov trust region NaNs every coordinate | **CONFIRMED** | no — TRIPLE-gated | `numerics.py:151-161`; call site `e_step.py:1135` |
| PE-1 | perf-engineer | `(B,N,V)` logits "under the default `decode_mode`" | **REFUTED as titled** | no — default AND live both route to chunked | `model.py:1564`, `prior_bank.py:1896` |

---

## The one REFUTATION — and it corrects the orchestrator too

**PE-1.** The internal mechanics all verified (`model.py:1564`, `:1566-1569`, `:1585`;
`prior_bank.py:1666`, `:1671`; `diagonal` registers WITHOUT `supports_chunked` at `:1628` vs
`diagonal_chunked` WITH it at `:1675-1681`). **But the headline and the orchestrator's reframing are
both false.** The verifier queried the registry directly:
```
linear -> supports_chunked=True ; diagonal -> False ; diagonal_chunked -> True
```
`prior_bank.py:1894-1900` registers `"linear"` with
`supports_chunked=True, fused_ce=PriorBank.decode_ce_linear_chunked`. Since the config default is
`use_prior_bank=False` (config.py:498), `active_decode_mode` is `"linear"`, the fused branch at
`model.py:1570-1582` runs, and `logits = None` — **the two defaults do NOT compose into a dense
path at all.** The live config is doubly safe: `use_prior_bank=False` AND
`decode_mode='diagonal_chunked'`, with `batch_size=16` not the 64 used for the 1.65 GB arithmetic.
The only true statement is the conditional: `use_prior_bank=True` COMBINED WITH
`decode_mode='diagonal'` materializes `(B,N,V)` when a byte-identical chunked twin exists. Worth a
one-line config-guidance note, not a critical.

## The SPLIT verdict

**IG-1**, two separable claims. **(a) the clamp gates the self-term — CONFIRMED**: `base.py:26` is
`kl.clamp(min=0.0, max=kl_max)`; `kernels.py:157` then `:179` multiplies the whole self-gradient by
`self_mask`, and `kernels.py:669/673` folds it into the fusion weight `a`. **(b) "no unclamped path
exists" — REFUTED as an absolute.** `config.py:918` does reject `inf` (reproduced:
`ValueError: kl_max must be finite and positive, got inf`), but **`VFE3Config(kl_max=1e30)` is
ACCEPTED**, and `safe_kl_clamp(x, kl_max=1e30)` returns `[0.0, 0.5, 1e6, 1e30]` — indistinguishable
from unclamped over any representable divergence. A numerically-unclamped path IS reachable today.
Note also `clamp(min=0.0)` survives even at `inf`, so admitting `inf` would not by itself yield a
literally unclamped functional. Under the live config `kl_max = 1680` (8 nats/coord) and the
investigator's K=20 saturation table does not transfer: at the `eps=1e-6` sigma floor a K=210 belief
on its prior mean gives ~6.4 nats/coord, BELOW the live ceiling. **Verifier would drop this from
high.**

## Reachability downgrades (real bugs, dead under the live config)

- **NA-3** — reproduced closely (r=1e-4 -> 0.4609 masked, r=1e-5 -> 0.5039, min fp32 raw -2.8e-06),
  but `kernels.py:156-160` lives in `_diag_kl_filtering_kernel`, reached only via
  `belief_gradients`, whose only in-package call site is `e_step.py:1060` — the `else` branch of
  `if e_step_update == "mm_exact":`. Live is `mm_exact`, whose own self-mask at `kernels.py:669/672`
  is `(raw_self < kl_max)` — **upper gate only**, so a negative fp32 raw still passes.
  **Downgrade high -> medium.**
- **DB-1** — reproduced exactly (`box`/full -> `[nan,nan,nan]`; root cause confirmed independently:
  `solve_triangular(eye(3), [1,inf,2]) -> [1., inf, nan]`). But `apply_mu_trust_region` has exactly
  one call site, `e_step.py:1135`, inside the gradient branch `mm_exact` never enters, AND requires
  `family='gaussian_full'` (live: diagonal) AND `e_mu_q_trust is not None` (live: None). **Three
  independent gates, all closed. Medium at most.**
- **G-1** — confirmed at source, and the M-step sibling genuinely does convert
  (`phi_preconditioner.py:581` computes `xi = einsum("...ab,...b->...a", psi_left, v_phi)`).
  Verifier's probe: median relative chart-move error **6.51e-01 (E-step convention) vs 6.63e-04
  (M-step convention)**. It could **NOT** reproduce the claimed 1/200 ascent at `||phi||=2.2` —
  got **0/200**, min eig of `sym(Psi_L^{-1}G^{-1})` **+1.4e-02**. But the claim strengthens sharply
  higher up, still inside the retraction caps: `||phi||=3.5` -> **75/200** ascent draws (min eig
  -32.1); `4.5` -> **132/200**; `5.0` -> **154/200**. "Descent not guaranteed away from identity" is
  confirmed, just not at the quoted norm. Unreachable live (`e_step.py:1153` gates on
  `if e_phi_lr > 0.0:`, and live is 0.00).
- **IE-1** — reproduced with an identical frame stack. Verifier's severity note: this is a **loud
  crash**, not silent corruption; "high" is generous for a fail-loud validation gap.
- **IE-2/V-2** — reachability is tighter than EITHER investigator stated: needs
  `emission_mode != 'off'` **AND `n_layers > 1`**, and live is `n_layers = 1`.

## THE ONE LIVE HIGH-BAND FINDING

**T-1, the ALiBi log floor.** Verifier's reproduction is an **exact match** of the investigator's
numbers: 56,903 causal-support entries, **2,954 floored**, **max 36.302 nats**; head 4 (Press slope
0.5) 2,701/8,129 floored; head 0 253/8,129 at 5.378 nats; heads 1/2/3/5/6 untouched. All trigger
conditions are set in the live config. **Severity caveat the verifier adds:** the floored entries
carry prior mass below 1e-12, and the "36 nats against a 306-nat energy range is decisive" step is
an ARGUMENT, not a measurement — nobody measured the resulting change in `beta` or in CE. The
distortion is real and reachable; "decisive" is unproven.

## Duplicate-pair adjudication

- **NA-1 vs T-2** (autocast/emission): agree on mechanism. Reachability statements differ and
  **the transformer-ml agent is the accurate one** — `emission_mode` defaults `'off'` and
  `train_vfe3.py` never sets it, so this is inert TODAY. The numerical-analyst's "not a corner case"
  **overstates present reachability**. But `config.py:1648-1657` does force `mm_exact` whenever
  emission is enabled, so the two gates genuinely compose the moment the user flips `emission_mode`.
  **Orchestrator's CRITICAL-class escalation is justified as a PRE-LAUNCH BLOCKER, not as a
  currently-active defect.**
- **IE-2 vs V-2** (Bohning anchor): different instruments (gradient residual vs object identity),
  same conclusion, no contradiction. Verifier's own spy: layers 0/1/2 all `emission_id=2358149977152`
  and `mean|g|=0.32951226830482483` while `mean|mu_p|` drifts 0.01280 -> 0.16017 -> 0.25377.

## Other verifier-vs-investigator numeric differences (conclusions unchanged)

- **NA-2**: verifier got `r=10, cond(U)=1.8e4 -> max rel err 0.708` where the investigator reported
  1.27 (different random draw). Same conclusion: a 70%-wrong transported variance passes with every
  entry strictly positive. Verifier notes it did NOT measure what per-block `||phi||` a live run
  actually reaches, so this is reachable-in-principle only. Severity high defensible.
- **CR-1**: values match the frozen-`p_0` reference to 7.45e-08, but `d g/d W` differs by max abs
  **0.1639**, relative **7.1%** — not the investigator's 150%, which was at a different scale. The
  qualitative contract violation holds. Memory reproduced: **autograd streaming 681.9 MiB, `no_grad`
  streaming 161.2 MiB, autograd dense one-shot 425.6 MiB** -> the streaming loop peaks at **1.60x**
  the dense materialization it exists to avoid (investigator: 669.6/148.9/413.3, same conclusion).
