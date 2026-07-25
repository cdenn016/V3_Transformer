# The gauge-frame norm bound: K=20 ablation decomposition, K=300 calibration, and the Stage 0 checkpoint measurements

Date: 2026-07-25
Author: agent-assisted investigation, all numbers machine-measured unless marked otherwise
Interpreter for every GPU/torch number below: `C:/anaconda/python.exe` (CUDA build, torch
2.10.0.dev20251210+cu128, RTX 5090)

## 1. What this document is

Two things prompted it. First, a reading of the two VFE 4.0 whitepapers
(`VFE_4.0/Manuscripts/vfe4_whitepaper/`, `VFE_4.0/Manuscripts/magent_elbo_whitepaper/`)
followed by an eight-agent sweep of the V3 codebase, which produced a ranked list of
performance hypotheses. Second, the observation that
`phi_mstep_max_matrix_norm=5` improves test PPL at K=20.

The agent sweep produced a number of confident claims. This document records what happened
when those claims were measured directly against the checkpoints. Three of the highest-ranked
ones did not survive. The document supersedes the Tier 1 section of
`docs/2026-07-25-vfe4-performance-hypotheses.md`, which was written before these measurements
existed and should be read only alongside this file.

Everything here is a checkpoint-only measurement. No model was trained, no application code was
modified, and no checkpoint on disk was written to.

## 2. The K=20 ablation grid, decomposed correctly

The four sibling runs in `vfe3_runs/ablations_single_seed/` are not four independent
single-knob arms. Reading each `config.json` against the baseline gives the actual diff:

| run directory | `pos_phi_compose` | `phi_mstep_max_matrix_norm` | test PPL |
|---|---|---|---|
| `138.53_wikitext-103_K20_block_glk_linear_mix_s6` | `bch` | `None` | 138.532 |
| `138.64_group-product` | `group_product` | `None` | 138.635 |
| `138.64_mstep-phi-norm=10` | `group_product` | `10` | 138.635 |
| `138.40_mstep-phi-norm=5` | `group_product` | `5` | **138.400** |

Both bound arms also switch the positional composition, so the run named for the bound is
really a two-knob change. Decomposing:

- `bch` to `group_product` alone: **+0.103 PPL, slightly worse.**
- adding `bound=10` on top: **exactly zero effect.** The two runs agree to twelve significant
  figures (138.635311323819) with identical phi statistics (median 3.2047, max 7.399). The
  `best_model.pt` md5s differ only because the serialized config differs. The reason is
  mechanical: the largest trained row norm at K=20 is 7.399, so a bound of 10 never binds.
- adding `bound=5` instead: **-0.235 PPL against the `group_product` arm, -0.132 against the
  original `bch` baseline.**

So the gain is attributable to the norm bound, not to the exact positional composition. This
matters for an earlier recommendation: switching `pos_phi_compose` to `group_product` on its own
is not supported by this evidence, and was slightly negative here. Its value appears to be that
it makes the bound meaningful rather than that exactness helps by itself.

**Significance caveat, stated plainly.** These are single-seed, 15k-step runs. The spread across
the whole grid is 0.235 PPL on a base of ~138.5, i.e. 0.17%. No seed-variance estimate exists for
this configuration, so a 0.13 PPL improvement is not established as larger than seed noise. The
mechanism below is a reason to take it seriously; it is not a substitute for a multi-seed repeat.

## 3. Why a bound helps: what it actually clips

For `block_glk` the generators are elementary matrices inside each block and are orthonormal
under the Frobenius inner product, so
`||sum_a phi_a G_a||_F == ||phi_v||_2` exactly. The M-step projection therefore bounds the
parameter-row 2-norm directly, and the row-norm distribution is the right object to reason about.

Measured over rows with `||phi_v|| > 1e-6` (dead rows excluded; see section 6):

| run | `n_gen` | init row norm | median | p99 | max | live rows | dead rows |
|---|---|---|---|---|---|---|---|
| K=20 baseline, no bound | 200 | 0.849 | 3.199 | — | 7.500 | — | — |
| K=20, bound 5 | 200 | 0.849 | 3.196 | — | **5.000** | — | — |
| K=20, bound 10 | 200 | 0.849 | 3.205 | — | 7.399 | — | — |
| K=300 baseline, no bound | 9000 | 5.692 | 10.310 | — | 38.344 | — | — |

The bound-5 run's max is exactly 5.000, confirming the projection is active and doing what it
claims. Its median is unchanged (3.196 versus 3.199), so the bound is a pure tail operation: it
touches the top of the distribution and leaves the bulk alone.

Fraction of live rows a given bound would clip:

| bound | K=20 (no-bound run) | K=300 (no-bound run) |
|---|---|---|
| 2 | 90.68% | 98.55% |
| **5** | **2.12%** | **92.87%** |
| 10 | 0.00% | 58.54% |
| 15 | 0.00% | 0.478% |
| 20 | 0.00% | 0.083% |
| 30 | 0.00% | 0.015% |
| 40 | 0.00% | 0.000% |

At K=20 a bound of 5 clips the top 2.1% of rows. At K=300 the same number would clip 92.9% of
them. **Porting the value 5 verbatim to K=300 would not reproduce the experiment; it would be a
different and far more violent intervention.**

## 4. Calibration to K=300

Four ways of asking "what is the K=300 equivalent of 5 at K=20":

| criterion | construction | K=300 value |
|---|---|---|
| (a) matched clip fraction | the quantile that 5 occupies in the K=20 distribution (97.879th) evaluated on the K=300 distribution | **12.93** |
| (b) matched ratio to init norm | `5 / 0.849 = 5.89x` init, applied to init 5.692 | 33.54 |
| (c) matched ratio to median trained norm | `5 / 3.199 = 1.563x` median, applied to median 10.310 | 16.12 |
| (d) per-block equivalence | bound scales with `sqrt(H)`; `5 * sqrt(10/2)` | 11.18 |

Three of the four cluster in **11 to 16**. Criterion (b) is the outlier at 33.5 because the
initialization norm scales as `phi_scale * sqrt(n_gen)` while the trained norms do not scale the
same way; it should be discounted.

The K=300 row-norm distribution is steep in exactly this region: 58.5% of rows exceed 10 but only
0.48% exceed 15, so the bulk sits between roughly 10 and 13 with a thin tail out to 38.3. The
choice inside the 11 to 16 window therefore changes the intervention substantially, and a value
below about 11 would start cutting into the bulk rather than the tail.

**Recommendation: `phi_mstep_max_matrix_norm = 13` as the primary arm** (matched clip fraction,
clips roughly 2% of rows exactly as the K=20 experiment did), with **16** as a gentler second arm.
Pair either with `pos_phi_compose='group_product'` to reproduce the K=20 condition. Do not use 5.

## 5. Stage 0: checkpoint-only measurements on the K=300 model

Baseline `data/55.41_wikitext-103_K300_block_glk_linear_mix_s6/best_model.pt`, scored on the full
wikitext-103 test split (137 batches of 16 x 128) with `vfe3.train.evaluate`.

**Reproduction note.** The model rebuilds under current HEAD at **498,747,258** parameters versus
the recorded 528,901,458, a difference of exactly 30,154,200, and `load_state_dict` reports all
keys matched after `normalize_legacy_model_state` drops `prior_bank.mu_embed` and
`prior_bank.sigma_log_embed`. This confirms byte-exactly that those two tables were inert during
the run. Scored under current HEAD the same weights give **CE 4.012327 / PPL 55.275**, against the
recorded 4.014838 / 55.414. The 0.0025-nat improvement comes from fixes landed after the run
(2026-07-20), so every comparison below uses 4.012327 as the baseline, not the recorded number.

### M1 attention temperature, `tau = kappa_beta * sqrt(30)`

| `kappa_beta` | tau | CE | PPL | delta CE |
|---|---|---|---|---|
| 2.00 | 10.954 | 4.014365 | 55.388 | +0.0020 |
| 1.50 | 8.216 | 4.013265 | 55.327 | +0.0009 |
| **1.00 (current)** | **5.477** | **4.012327** | **55.275** | **best** |
| 0.70 | 3.834 | 4.012718 | 55.297 | +0.0004 |
| 0.50 | 2.739 | 4.014827 | 55.414 | +0.0025 |
| 0.35 | 1.917 | 4.019809 | 55.690 | +0.0075 |
| 0.25 | 1.369 | 4.027949 | 56.146 | +0.0156 |
| 0.1826 | 1.000 | 4.038964 | 56.768 | +0.0266 |

The proposal was to move tau toward the ELBO-exact value of 1, on the argument that the content
energy is drowned by the positional prior. Measured, that direction is **monotonically worse**,
and the current setting is the minimum. The curve is also flat within roughly +/- 0.5 in
`kappa_beta` (under 0.003 nats), which argues against the "starved for content signal" reading.
Weights were trained at `kappa_beta=1.0`, so local optimality is partly circular and a retrained
arm could differ; but the advertised free win does not exist.

### M2 transport Frobenius clamp

| `TRANSPORT_CLAMP_MAX_NORM` | CE | PPL | delta CE |
|---|---|---|---|
| 10 | 4.599083 | 99.393 | +0.587 |
| **20 (current)** | **4.012327** | **55.275** | **best** |
| 40 | 4.204689 | 67.000 | +0.192 |
| 60 | 4.362218 | 78.431 | +0.350 |
| 1e6 | — | — | `FloatingPointError`: nonfinite omega before inversion |

The clamp had been identified as the run's largest defect, on the grounds that above the
threshold the code returns `exp(20 M / ||M||_F)` rather than `exp(M)`. That is true as a
statement about the operator, but the performance conclusion drawn from it was wrong. **The
trained weights depend on the clamp.** Widening it degrades monotonically and removing it
overflows to nonfinite. The surrogate operator is this model's operator.

This is the measurement that reconciles with the K=20 result. Relaxing the clamp after training
breaks a learned map; bounding `||phi||` during training so the clamp never has to fire is a
different and evidently better intervention. Sections 2 to 4 are the actionable form of this
finding; M2 rules out the post-hoc version.

Incidental: `max_norm=10` evaluated in 17 s versus 59 s at 20 and above. The difference is the
float32-to-float64 congruence re-computation guard at `vfe3/geometry/transport.py:2148` firing
when conditioning degrades, so that escalation costs roughly 3.5x on this workload. It is
triggered by ill-conditioning rather than being an independent switch.

### M3 `phi_embed` ablation

| variant | CE | PPL | delta CE |
|---|---|---|---|
| baseline | 4.012327 | 55.275 | — |
| `phi := 0` (identity gauge) | 5.928340 | 375.531 | +1.916 |
| `phi := single corpus-mean row` | 5.920477 | 372.589 | +1.908 |
| `phi := 4-bit per-row quantized` | 4.023128 | 55.876 | **+0.011** |
| `phi` restored | 4.012327 | 55.275 | 0.000 |

The table holds 452,313,000 parameters, 85.5% of the run's total, and the sweep had suggested it
might be decorative. It is not: zeroing it costs 320 PPL. Mean-collapse costs the same as zeroing,
which localizes the value in the **per-token variation** rather than the overall gauge level, as
expected since a globally constant frame is close to a no-op under equivariance.

But 4-bit per-row quantization costs only 0.6 PPL. The table tolerates an eightfold precision
reduction essentially for free, so it carries on the order of four useful bits per parameter
against thirty-two stored. The re-budgeting question is about precision and rank, not deletion.
The direct follow-up is a low-rank SVD truncation of `phi_embed`, which tests whether `n_gen`
can shrink; 4-bit robustness alone does not establish that.

The restored row reproduces the baseline exactly, so the in-place ablations were clean.

### M4 per-position cross-entropy

| context tokens available | CE |
|---|---|
| 0 | 5.599 |
| 1 | 4.982 |
| 2 to 3 | 4.713 |
| 4 to 7 | 4.417 |
| 8 to 15 | 4.151 |
| 16 to 31 | 4.033 |
| 32 to 63 | 3.959 |
| 64 to 127 | 3.929 |

Context is worth **1.68 nats** from position 0 to position 127. The prior reading, taken from the
logged `pos_loss_ratio = 0.9796`, was that the model barely uses context. That was an artifact of
comparing quartile means: the first quartile is dominated by positions 16 to 31, which are already
near-converged, so the summary compresses a large early gain into a small number. Computing the
same ratio directly from this curve gives **0.9282**, not 0.9796, and the discrepancy with the
logged metric could not be reconciled from the artifacts; the direct measurement is the one to
trust here.

The operational conclusion nevertheless survives, for a different reason. The model extracts 1.57
of the 1.68 nats within the first 32 tokens and then saturates: the marginal gain from the 32-to-63
band to the 64-to-127 band is **0.030 nats**. Extending the context window from 128 to 512 should
therefore be expected to buy very little, and remains a low priority.

### M5 low-rank SVD truncation of `phi_embed`

M3 showed the table tolerates 4-bit quantization for 0.6 PPL, which establishes redundancy in
*precision*. This asks the independent question of whether the 9000-dimensional rows occupy a
low-dimensional *subspace*. Exact truncated SVD via the Gram matrix, `phi_r = (phi V_r) V_r^T`
with `V_r` the top-r eigenvectors of `phi^T phi`.

Spectrum, as the rank needed to capture a share of Frobenius energy:

| energy captured | rank required (of 9000) |
|---|---|
| 50% | 2486 |
| 90% | 6760 |
| 95% | 7700 |
| 99% | 8681 |
| 99.9% | 8965 |

Leading singular values are 192.3, 153.9, 108.2, 99.8, 88.4, 85.6, 78.7, 76.2, decaying smoothly
with no knee. Scored:

| rank | CE | PPL | delta PPL |
|---|---|---|---|
| 16 | 5.667681 | 289.363 | +234.1 |
| 32 | 5.615133 | 274.550 | +219.3 |
| 64 | 5.443302 | 231.204 | +175.9 |
| 128 | 5.270362 | 194.486 | +139.2 |
| 256 | 5.117084 | 166.848 | +111.6 |
| 512 | 4.937342 | 139.399 | +84.1 |
| 1024 | 4.698955 | 109.832 | +54.6 |
| 2048 | 4.403184 | 81.711 | +26.4 |
| 4096 | 4.133061 | 62.369 | +7.09 |
| **9000 (full)** | **4.012327** | **55.275** | **—** |
| restored | 4.012327 | 55.275 | 0.000 |

**The table is rank-saturated.** It takes 2486 dimensions to reach even half the Frobenius
energy, and 8681 to reach 99%. Discarding half the spectrum (rank 4096, a 45% parameter saving)
still costs 7.1 PPL, and every smaller rank is catastrophic. There is no thin subspace to
exploit.

This settles the gauge re-budgeting question against every factorization proposal. A low-rank
`phi_embed = C B` would have to keep essentially full rank to preserve quality, at which point it
costs more parameters than the dense table, not fewer. It also argues against
`gauge_group='tied_block_glk'` (`n_gen = 900`) as a free win: that is a 10x dimensional cut on a
table which is already using its dimensions, so it should be treated as a substantial capacity
reduction to be paid for elsewhere, not as recovered slack.

Taken with M3, the characterization is precise: `phi_embed` is **highly redundant in precision
and saturated in rank**. Roughly four bits per parameter suffice, but all 9000 directions are
carrying signal. If the table is to be shrunk, quantization or a lower-precision storage dtype is
the supported route; dimensionality reduction is not.

## 6. Incidental findings worth keeping

**Dead vocabulary rows.** In the K=300 checkpoint a set of rows sits at `||phi_v|| = 0.000`
exactly. These are token types with zero training-corpus occurrences, driven to zero by decoupled
weight decay: `pb.phi_embed[token_ids]` produces a dense gradient, so AdamW decays all 50,257 rows
every step whether or not the token appeared. With `phi_weight_decay=0.03` and the run's schedule
the shrinkage factor is roughly `exp(-27)`. Rows for very low-count types sit below their random
initialization. `output_proj_weight` escapes this because the softmax gives every row a dense
gradient. Exempting the Zipfian encode tables from decay is free and independent of everything
above, though the achievable gain is bounded: the rare and mid strata together account for at most
about 8% of total test CE.

**Two tables were inert.** `prior_bank.mu_embed` and `prior_bank.sigma_log_embed`, 30,154,200
parameters, received no gradient during the run and remained at initialization
(`mu_embed.std() = 6.4976e-2` against `mu_init_std = 0.065`;
`sigma_log_embed.absmean() = 1.386295 = log 4` with std 3.6e-7). Commit `b362506` (2026-07-21)
already drops them; the run predates it.

**Conditioning stress is real.** The float32 congruence guard fires during ordinary evaluation
(`diagonal congruence lost nonnegativity, min=-329.324`, `vfe3/geometry/transport.py:2148`),
independently supporting the concern that the diagonal covariance family is poorly conditioned
under the untied block gauge.

## 7. What to run next

**Primary arm.** `phi_mstep_max_matrix_norm = 13`, `pos_phi_compose = 'group_product'`, at K=300,
everything else at the 55.41 configuration. This is the calibrated port of the K=20 result and is
the only proposal in this document with direct supporting evidence at another scale.

**Second arm.** The same with `phi_mstep_max_matrix_norm = 16`, to bracket the calibration window
given how steep the row-norm distribution is between 10 and 15.

**Free and independent.** `phi_weight_decay = 0.0`, `sigma_weight_decay = 0.0`, and a zero
weight-decay group for `s_mu_embed`, per section 6.

**Before spending a full run.** Repeat the K=20 grid over three or more seeds. The entire observed
spread is 0.235 PPL on a base of 138.5 and no variance estimate exists, so the effect being ported
to K=300 is currently unreplicated.

**Not worth pursuing.** Any factorization or dimensional reduction of `phi_embed`. M5 settles this:
the table is rank-saturated and there is no subspace to exploit. If its footprint must come down,
the supported route is reduced storage precision, since 4-bit costs only 0.6 PPL.

## 8. Claims that did not survive measurement

For the record, since these were ranked highly before being tested:

1. "The transport clamp is the largest defect, worth 1 to 4 PPL." Refuted by M2; the trained
   weights depend on the clamp and widening it is strictly harmful.
2. "Attention temperature is off by an order of magnitude; moving toward the ELBO-exact tau=1 is a
   free win." Refuted by M1; the current setting is the measured optimum.
3. "85% of parameters are decorative; the position curve is flat and the model barely uses
   context." Refuted by M3 and M4 respectively; the gauge table is load-bearing to the tune of 320
   PPL, and context is worth 1.68 nats.
4. "`pos_phi_compose='group_product'` is the right call on its own." Not supported; at K=20 it is
   slightly negative in isolation and only pays off combined with the norm bound.
5. "The gauge table can be factored or its `n_gen` shrunk to free capacity." Refuted by M5; the
   table is rank-saturated, needing 8681 of 9000 directions for 99% of its energy.

The BCH accuracy argument that motivated (4) is still arithmetically correct
(`bch_relative_error_median = 0.094`, max 0.580 in `phi_numerics.json`), which is a reminder that a
correct statement about an operator does not by itself predict the sign of a performance change.

## 9. Reproduction

Scripts live in the session scratchpad and are self-contained; each takes the run directory as a
constant at the top.

```
"C:/anaconda/python.exe" stage0_gate.py            # load + reproduce baseline, phi-norm histogram
"C:/anaconda/python.exe" stage0_measure.py         # M0, M1, M2
"C:/anaconda/python.exe" stage0_m3m4.py            # M3, M4
"C:/anaconda/python.exe" calibrate_phi_bound.py    # section 3 and 4 tables
```

`stage0_measure.py` aborts at `max_norm=1e6` by design, since that case raises
`FloatingPointError` from `_checked_group_inverse`; M3 and M4 were therefore run separately. All
mutations are in-memory on a GPU copy of the weights, with `phi_embed` restored from a pristine
clone and verified to reproduce the baseline exactly.
