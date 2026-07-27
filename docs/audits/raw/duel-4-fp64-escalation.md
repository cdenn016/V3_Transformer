# Duel 4 — diagonal-congruence fp64 escalation tests sign, not accuracy

## SKEPTIC (attack) — returned 11:04 CDT

**Verdict argued: DOWNGRADE to LOW.** Closes the reachability gap the verifier flagged — using the
**actual trained checkpoint**, the strongest evidence produced anywhere in this audit.

### The load-bearing attack: the probe's `r` is a PER-BLOCK norm at d=6; every bound in the codebase is a JOINT norm over all H blocks
Both bounds are joint, in executable code:
- `gauge_optim.py:398` — `rows = table.reshape(-1, table.shape[-1])`; the projection ball is one row
  = the full `n_gen` coordinate vector = the full-K embedded matrix, **not a per-block ball.** Live
  bound `phi_mstep_max_matrix_norm = 12` (`train_vfe3.py:189`), gated live at `train.py:757-761`.
- `transport.py:1930` — `blocks.square().sum(dim=(-3,-2,-1))`, reduced over `(H,d,d)`. So
  `TRANSPORT_CLAMP_MAX_NORM = 20.0` is **also joint.** At the live `n_heads=7` that outer rail is
  **7.56 per block — below the r=8 onset of the investigator's own table.**

Since `||phi||_F^2 = sum_h ||phi_h||_F^2` and `transport.py:2037` reshapes coordinates into `H`
row-major `d x d` blocks in an orthonormal basis, per-block norm ~ `12/sqrt(7) = 4.54`.

### MEASURED ON THE TRAINED LIVE-SCALE CHECKPOINT
`vfe3_runs/20260726-225843_wikitext-103_K210_block_glk_linear_mix_s6/best_model.pt` (config matches
live: `embed_dim=210`, `n_heads=7`, `block_glk`, `transport_mode='flat'`,
`phi_mstep_max_matrix_norm=12`):
```
prior_bank.phi_embed (50257 rows x 7 blocks):
  full-K ||phi||_F      median 11.538  MAX 12.000   (projection binds exactly)
  per-block ||phi_h||_F median  4.266  MAX  5.698
  fraction of 351,799 blocks with ||phi_h||_F >= 8 : 0.000e+00   >= 10 : 0.000e+00
pos_phi_free (128 x 7):  per-block MAX 6.088; none >= 8

per-block cond(exp(phi_h)):  tokens median 13.3, p99 39.0, MAX 170.86
                             positions            MAX 250.01
```
against the probe's `cond(U) = 2.06e4` (r=8) and `2.79e4` (r=10). **Two orders of magnitude short.**
The conditioning axis is worse for the finding still, because cancellation tracks `cond`, not
Frobenius norm, and d=30 spreads the same budget over 900 entries instead of 36.

### Executed on the LIVE route
`model.py:715-723 _compact_phi_blocks_enabled()` is True under live `gauge_parameterization='phi'` /
`transport_mode='flat'` / `phi_reflection='off'` / `block_glk` layout, so **`:2619` is the live guard
and `:2714` is off-path.** Calling the real `_compact_factored_diagonal_covariance` on real trained
`phi` composed with the real `pos_phi_free` right factor, against an fp64 dense reference:
```
16 random real sequences, N=128, trained sigma:
  worst per-block cond(U)                 = 1065.4
  WORST fp32 relative error vs fp64 dense = 4.33e-03
  guard invocations 16, escalations 0, min(out) > 0 every time
Adversarial (the 64 worst-conditioned real token rows x the 64 worst real positions,
a batch no real draw produces): pair cond(Omega_ij) max 1.29e5,
  max rel err 3.29e-02, p99 2.36e-06, min(out) = +9.83e-05, escalations 0.
```
**At the live gauge the unescalated fp32 result is accurate to 4.3e-3 worst-case.** Even a
deliberately adversarial selection of real rows — driving pair `cond(Omega)` to 1.29e5, ABOVE the
probe's 2.8e4 — yields 3.3% at one tail entry with p99 at 2.4e-6, not 127%.

### What the skeptic conceded survives
> "The code fact is real — the guard is sign-only, and the fix is cheap. But reaching the failure
> band needs a *deliberately* built config with small `d_head` and a joint bound raised past
> `~10*sqrt(H)`, i.e. above both the live 12 and the 20 rail. `phi_exp_clamp_frac` is `0.0000`
> across every run in `vfe3_runs/`, and the finding's own compact-sibling row
> (`cond 5.37e4 -> escalated, 2.37e-4`) is the live function behaving correctly."

### Skeptic's decisive evidence
`gauge_optim.py:398` + `transport.py:1930` (both norms joint over H blocks), measured against the
trained checkpoint: max per-block `||phi_h||_F = 5.698` over all 351,799 trained blocks, max
per-block `cond(exp(phi_h)) = 170.86`, worst live fp32 congruence error `4.33e-03`.

## DEFENDER — returned 11:14 CDT

**Verdict argued: UPHELD as real and reachable — concede DOWNGRADE high -> MEDIUM.**

### IT CONTRADICTS THE SKEPTIC ON WHICH GUARD IS LIVE
Live `train_vfe3.py` sets `omega_compact_storage=False`, so `build_factored_transport` returns a
`FactoredTransport` (`transport.py:2095`), `transport_covariance` takes the
`isinstance(omega, FactoredTransport)` branch at `:2336` and calls `_factored_diagonal_covariance`
at `:2342` — **NOT** the compact sibling at `:2326`. So **`:2714` is the live guard; `:2619` is
dormant.** With `d = 210/7 = 30` and `n_tokens = 128`, `d <= n_tokens` at `:2706` holds and the
mixed-sign `_reduce` branch (`:2707-2712`) executes. Caller is `kernels.py:442`.

### RECORDED metrics.csv from the K=210 run (239 logged steps)
| metric | peak over run |
|---|---|
| `phi_matrix_norm_max` | **12.000** (pinned at the projection radius) |
| `phi_chart_preproject_max` | 13.15 (the M-step keeps pushing past the bound) |
| `phi_chart_projected_fraction` | 0.27 |
| `phi_exp_clamp_frac` / `phi_exp_scale_min` | 0.0 / 1.0 (**the 20.0 clamp never fires**) |
| `vertex_cond_median / p95 / p99 / max` | 289 / 1681 / 3051 / **9640.2** |
| `belief_cond_max` | **1837** |

Reconstructed from `best_model.pt`: per-block `||phi_h||_F` token median 4.27 / max 5.44;
positional median 4.38 / max 6.09. Per-block `cond(Omega_ij)` median **210**, p99 **3303**, max
**1.2e4** random, **7.6e4** on worst-conditioned real vocab rows. `model.py:3032-3034` states the
identity the finding needs: `cond(Omega_ij) <= vertex_cond_max^2`.

### MEASURED on the LIVE function with trained factors, sigma range set to RECORDED `belief_cond`
| belief_cond | guard fired? | `out.amin()` | median rel err | max rel err | frac > 1% |
|---|---|---|---|---|---|
| 36 | **False** | +1.00 | 1.2e-07 | 1.7e-02 | 0.0002% |
| 382 | **False** | +0.99 | 1.3e-07 | 1.6e-01 | 0.0017% |
| 674 | **False** | +0.95 | 1.3e-07 | 2.3e-01 | 0.0031% |
| **1837** (recorded max) | **False** | +0.69 | 1.3e-07 | **6.4e-01** | 0.0064% |

**The guard reads `amin` between +0.69 and +1.00 in every one of 12 trials — comfortably positive,
not marginal — while a transported variance is 64% wrong.** Reachability gap closed on trained
production weights.

**The guard's perversity is directly demonstrable:** on a wide-dynamic-range sigma it FIRED
(`amin = -0.5`) and the escalated answer was accurate to 8e-6; on the trained prior sigma it PASSED
a 9.2%-wrong entry (`amin = +1.04e-4`). It escalates on a case that was going to be fine and passes
the case that is wrong.

### THE DECISIVE SEVERITY NUMBER — the defender measured the blast radius rather than asserting it
Pushing the fp32-vs-fp64 `sigma_t` through `diagonal_kl_pair_stats` (`kl_max=100`, `eps=1e-6`,
`irrep_dims=[30]*7`) at the worst recorded conditioning: `E_ij` absolute error **median 0, p99
7.6e-06, max 0.064 nats**, with 0.0035% of pairs above 0.01 nats and **none above 0.1 nats**. The
docstring's own severity argument (`transport.py:1414-1416`) describes the **sign** failure, which
the guard DOES catch; the accuracy failure has no such amplification.

### SCALE TREND — the reason this matters for the user's roadmap
Same tail-selection procedure on the K=20 checkpoint (`148.25_no-head-mixer`, d=10, worst-conditioned
K=20 run at `vertex_cond_max=1216.6`): max rel err **9.9e-04**. K=210 (d=30): **6.4e-01**. **Nearly
three orders of magnitude worse over one decade of K**, on a path the user sweeps to K=300, with
`phi_mstep_max_matrix_norm` itself scaled with K (5 at K=20, 12 at K=210 — recorded).

### The defender's three concessions
1. **The `TRANSPORT_CLAMP_MAX_NORM = 20.0` framing is WRONG.** `phi_exp_clamp_frac = 0.0` and
   `phi_exp_scale_min = 1.0` in every recorded run — the 20.0 clamp has never fired. The live
   envelope is `phi_mstep_max_matrix_norm = 12` on the full-K norm, per-block 4.3-6.1. Citing 20.0
   inflates the admitted band ~4x per block. **(Agrees with the skeptic.)**
2. **The probe's `r` mapping is loose.** Conditioning at fixed `||phi_h||_F` depends strongly on
   `d`: `d=4,H=2` gives `cond=1.7e5` at `r=10`, while production `d=30,H=7` at `r=12` gives
   `cond=411`. The 127% data point was at `d=6`, not production `d=30`. The defensible mapping is
   via the recorded `vertex_cond`, not via `r`. **(Agrees with the skeptic.)**
3. **"100%-wrong"/"1.27" is NOT reached at the live config.** Production-calibrated max is 0.64;
   median stays at fp32 roundoff (1.3e-07).

Also disclosed: it used trained `prior_bank.s_sigma_log_embed` plus synthetic sigma calibrated to
recorded `belief_cond`, not the runtime E-step belief sigma. If the live belief sigma is more
adversarial than a log-uniform spectrum at the same condition number, its numbers understate.

## ADJUDICATION — **UPHELD, DOWNGRADED high -> MEDIUM**

**The two sides directly contradicted each other on which guard is live, and the orchestrator
settled it at source.** Verified: `train_vfe3.py:180` sets `omega_compact_storage = False`;
`transport.py:2320` dispatches `CompactFactoredTransport` to the compact path and
`isinstance(omega, FactoredTransport)` to `_factored_diagonal_covariance`; guard call sites are
`:2527`, `:2619`, `:2714`. **The defender is correct: `:2714` is live, `:2619` is the dormant
compact sibling.** The skeptic conflated `_compact_phi_blocks_enabled()` (phi block storage) with
`omega_compact_storage` (Omega storage) — different flags. Its central measurement therefore ran
against the **dormant** function, so its "worst live error 4.33e-03" and its DOWNGRADE-to-low rest
on the wrong code path.

That said, the skeptic's joint-vs-per-block norm argument is correct and the **defender
independently conceded it**, along with the `TRANSPORT_CLAMP_MAX_NORM` framing and the `r`-mapping
looseness. Both sides agree the original 127% figure is not reachable at the live config.

Severity resolves on the defender's downstream measurement, which is the number neither the
investigator nor the verifier produced: **max `|dE_ij| = 0.064` nats on 0.0035% of pairs, none above
0.1 nats.** That is not high. But it is not low either — the mechanism is confirmed on trained
production weights with the guard reading `amin = +0.69` while returning a 64%-wrong variance, and
the guard demonstrably escalates on safe cases while passing wrong ones.

**The scale trend is what earns the medium and belongs on the punch list:** 9.9e-04 at K=20 versus
6.4e-01 at K=210 — three orders over one decade of K — on a path the user actively sweeps toward
K=300. The proposed fix is nearly free because `vertex_cond` is already computed for exactly these
blocks at `model.py:3021`.
