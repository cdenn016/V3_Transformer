# V3 state of knowledge, 2026-07-25

Read this first. It consolidates one day's measurements into what is now established, what was
refuted, and what is open. Detail lives in four companion documents:

- `docs/2026-07-25-vfe4-performance-hypotheses.md` — the VFE 4.0 whitepaper hypothesis synthesis
- `docs/2026-07-25-phi-bound-calibration-and-stage0-report.md` — measurements M1-M6
- `docs/2026-07-25-shadow-prior-investigation.md` — six-expert panel and Stage A
- `docs/2026-07-25-edits.md` — the code changes made today

Baselines referenced throughout: `55.41_wikitext-103_K300_block_glk_linear_mix_s6` (K=300, 10 heads,
1 layer, 128 context, val PPL 54.18 / CE 4.0123) and
`vfe3_runs/ablations_single_seed/138.40_mstep-phi-norm=5` (K=20, 2 heads, val PPL 139.3).

## 1. Established

| Finding | Number | How measured |
|---|---|---|
| The gauge table is load-bearing | zeroing `phi_embed` costs ~1.9 nats (55.3 -> 375.5 PPL) | M3 |
| ...but precision-redundant | 4-bit per-row quantization costs 0.011 nats | M3 |
| ...and rank-saturated | rank 4096 of 9000 still costs 7.1 PPL; no thin subspace | M5 |
| The positional prior carries ~3x the content channel | flattening `log_prior` 0.612 nats vs ablating the coupling energy 0.210 | M6 |
| The gamma fold into beta is nearly free | 0.0029 nats (0.16 PPL) | M6 |
| The model barely uses long context | 32-63 -> 64-127 marginal gain 0.030 nats | M4 |
| `kappa_beta=1.0` is already optimal | 8-point sweep, minimum at 1.0 | M1 |
| The transport clamp is load-bearing | `max_norm=20` optimal; 10 -> PPL 99, 40 -> 67 | M2 |
| One belief E-step is worth ~1.4 nats | see Section 3 | depth decoupling |
| Further belief E-steps are worth ~0.012 nats | depths 2-8 | depth decoupling |
| The exact congruence energy is ~7x-426x the truncated one | median 7.2x, p99 426x, max 10912 nats vs `kl_max=160` | exact-congruence run |

The last row is the day's clearest structural result: the diagonal truncation in
`transport_covariance` is not an approximation error to be removed, it is the regularizer that makes
a non-compact GL(K) gauge usable. The exact congruence KL grows like `cond(Omega)^2` (observed
maximum tracks `cond_max^2` to within 10%), saturates the `kl_max` clamp, and the clamp's zero
derivative then removes the only force pulling `phi` back toward well-conditioned frames.

## 2. Refuted

Every item below was a leading hypothesis that cheap measurement killed. The base rate matters: of
the mechanism hypotheses ranked highest at the start of the day, essentially all were refuted.

- **Temperature tuning as a lever** (M1), **transport-clamp tuning** (M2), **low-rank factorization
  of `phi_embed`** (M5).
- **Exact-congruence covariance** as a performance play. Correct to 3e-13 against a dense reference,
  but 93-98% of pairwise energies saturate `kl_max` and the run diverges. Kept as a measurement
  instrument, default OFF.
- **Frame-intrinsic covariance** (`gaussian_frame_diagonal`). The Regime-I coboundary cancels, so
  `phi_embed` receives no gradient under any estimator; 139.3 -> 308.1 PPL. Kept as a gauge
  ablation, default OFF.
- **"Inference is anti-aligned with prediction."** The claim that motivated the shadow-prior
  investigation. Refuted in Section 3.
- **Consensus collapse / anchor decay / `kl_max` anchor dropout / rank collapse in the belief
  channel.** All four Stage A diagnostics are flat: body-frame dispersion moves 4.5% across eight
  iterations, `selfdiv_klmax_frac` is 0.000 at every depth, `alpha*` stays at 0.9998, effective rank
  is static, and re-running at `kl_max=1e6` reproduces every number byte for byte.
- **The cross-scale shadow prior** as a fix for any of the above. No gauge degeneracy exists for a
  likelihood to break (V3's two flat directions are parameterization redundancies invisible to every
  observable, already lifted by weight decay); the token cannot be both prior and observation; the
  shadow term is a Bethe/BP object rather than the mean-field ELBO term; and the architecture is
  Ladder VAE applied to text, which has no competitive record on any language benchmark.

## 3. The depth pathology was a shared config field

`cfg.n_e_steps` is read by the belief E-step (`model/block.py`) AND by the model-channel refinement
(`model/model.py::_refine_s`). Under `prior_source='model_channel'` the refined s IS the belief's
prior, so sweeping the one field moved the prior and the belief together. Pinning each loop in turn,
delta CE against each arm's own depth-1 baseline, four fixed test batches:

| depth | K=20 both | K=20 belief only | K=20 s only | K=300 both | K=300 belief only | K=300 s only |
|---|---|---|---|---|---|---|
| 2 | +0.327 | **+0.005** | +0.302 | +0.285 | **+0.004** | +0.236 |
| 3 | +1.147 | **+0.009** | +1.115 | +0.987 | **+0.009** | +0.902 |
| 5 | +2.826 | **+0.012** | +2.808 | +2.657 | **+0.012** | +2.605 |
| 8 | +3.479 | **+0.012** | +3.477 | +3.594 | **+0.013** | +3.584 |

The belief E-step supplies 0.3% of the effect. `r_mu` is a single `(K,)` vector broadcast to every
position (`model.py`, `r_mu_t.expand_as(s_mu)`), so iterating the model channel pulls every `s_i`
toward one global centroid and `p_i` stops depending on the token. That is the consensus channel
behaving as designed; the defect was a diagnostic that varied it under the belief loop's name, now
fixed (`s_e_step_n_iter`, two separate series, both depths recorded per point).

### The first step, decomposed

Turning both loops off and adding one step of each:

| configuration | K=20 CE | gain | K=300 CE | gain |
|---|---|---|---|---|
| belief 0, s 0 | 6.6064 | — | 6.1380 | — |
| belief 1, s 0 | 5.2637 | **1.343** | 4.7062 | **1.432** |
| belief 0, s 1 | 4.7525 | 1.854 | 4.2225 | 1.916 |
| belief 1, s 1 | 4.6592 | 1.947 | 4.1340 | 2.004 |

Two things follow. **One belief E-step is worth ~1.3-1.4 nats standalone** — the belief loop does
substantial work; it simply finishes in one step. And the two channels are heavily **redundant**:
individually they contribute 1.343 + 1.854 = 3.20 nats but jointly only 1.947. That redundancy
explains why `prior_source='token'` with `s_e_step=False` reaches comparable perplexity — the removed
channel was largely duplicating the belief loop, and a model trained without it covers the
difference.

Prediction for any `s_e_step=False` run: because `_refine_s` is gated off entirely, `n_e_steps`
drives the belief loop alone, so the depth-sensitivity curve should be nearly FLAT (~0.01-0.02 nats
out to depth 8) rather than showing the ~3.5-nat cliff. A curve that still explodes would mean the
decoupling above is incomplete.

## 4. Corrections issued today

Recorded so they are not re-derived from stale chat.

- "Free energy falls monotonically while cross-entropy rises" was wrong twice: depth 0 exists and is
  the worst CE in both runs (the first E-step lowers F *and* CE), and F is not monotone — it bottoms
  at depth 3/5 and rises after, so the largest CE damage happens where F is flat or rising.
- "The token enters as an initial condition, not a force" was wrong: `mm_exact` fuses
  `mu_star = (a*mu_p/sp + pair_mean)/P`, so the prior anchors every iteration.
- "The belief E-step is neutral" applied only to depths >= 2; the first step is worth ~1.4 nats.
- The pre-existing rope test failures are stale on `main`, not a working-tree artifact
  (`origin/main:vfe3/config.py` has `rope_on_value = False` while the test asserts True).
- Decode is ~4-6% of wall clock, not 98%; `decode_tau` is inert under `use_prior_bank=False`; the
  55-PPL run had hit its `min_lr_frac` floor rather than still improving.

## 5. Open, ranked

1. **The E-step finishes in one step.** One iteration is worth 1.4 nats, the next seven are worth
   0.012, and F moves 0.04 nats total. Whatever the iterative minimization contributes, it is a
   single learned transform rather than convergent inference. Given the architecture's premise —
   no neural networks, all capacity from iterative VFE minimization — this is the most important
   open question. Is the E-step doing inference at all, or one preconditioned step that training has
   shaped into a feature map?
2. **Attention is mostly positional.** The content channel is worth 0.210 nats against the
   positional prior's 0.612 (M6), and the model gains 0.030 nats from doubling context past 64
   tokens (M4). Whether the coupling energy can be made a stronger router is the second question,
   and it is bounded above by ~0.21 nats as currently constituted.
3. **Channel redundancy.** The belief and model channels overlap by ~1.25 nats of their combined
   3.20. What is each actually contributing that the other does not?
4. **Depth.** `n_layers=1` in both baselines. The layer stack's prior handoff is elementwise with no
   transport (`stack.py`), i.e. a degenerate one-token-per-agent shadow, and it has never been
   exercised at the trained config.

Not recommended: further work on the exact congruence, the frame-intrinsic family, or the shadow
prior, for the reasons in Section 2.

## 6. What the E-step actually is: one attention aggregation with a dominant residual

Open question 1 above, answered. Both framings in the question were wrong.

**Framing correction first.** `e_step_update='mm_exact'` computes the CLOSED-FORM stationary point of
the beta-frozen objective. Convergence in one step is therefore by construction, not evidence of a
degenerate E-step, and the earlier reading of "F flat after step 1" as suspicious was misplaced.

> **CORRECTED 2026-07-26.** The table below was produced by a probe that read the MODEL channel's
> fusion as the belief's and anchored on the token-uniform centroid `r` rather than the belief's own
> prior (audit finding B-01, fixed in `942f685`). The original numbers are struck through; the
> re-measurement under the corrected probe is in the right-hand columns. Raw record:
> `docs/2026-07-26-b01-remeasurement.json`.

Measured belief-loop only, with the model channel pinned at its trained depth via the new
`s_e_step_n_iter`:

| | K=20 (2026-07-25) | K=20 (re-trained) | K=300 (2026-07-25) | K=300 (re-measured) |
|---|---|---|---|---|
| `\|\|dmu\|\| / \|\|mu_p\|\|` after 1 step | ~~0.147~~ | **0.213** | ~~0.227~~ | **0.299** |
| ...after 8 steps (converged) | ~~0.200~~ | **0.222** | ~~0.323~~ | **0.319** |
| share of total displacement taken by step 1 | ~~73%~~ | **96%** | ~~70%~~ | **94%** |
| `cos(direction_8, direction_1)` | ~~+0.982~~ | **+0.984** | ~~+0.962~~ | **+0.965** |
| PAIR (attention) share of the fused precision | ~~0.190~~ | **0.153** | ~~0.298~~ | **0.196** |
| PRIOR (residual) share | ~~0.810~~ | **0.847** | ~~0.702~~ | **0.804** |

Both re-measured columns are at 64 sequences (see the sampling table below). The K=20 column is a
re-training, not a re-measurement of the original checkpoint, which no longer exists; the K=300
column is the original checkpoint re-measured.

**Attribution correction (2026-07-26, second pass).** An earlier version of this note identified the
K=20 checkpoint behind the published column as `307.49_wikitext-103_K20_block_glk_linear_mix_s6` and
concluded the share was undefined because that run uses `e_step_update='gradient'`. That was wrong.
The research-wiki source note records the actual checkpoint as
`vfe3_runs/ablations_single_seed/138.40_mstep-phi-norm=5` (K=20, 2 heads, validation PPL 139.3,
`mm_exact`). `307.49` is a different, gradient-route run; its numbers (displacement 0.501 -> 0.571,
`cos` +0.920) belong to that run alone and must not be read against the 2026-07-25 K=20 column. Those
two figures are additionally superseded by B-11 below: every gradient-route displacement measured
before that fix is a POST-`head_mixer` number and is not comparable to any `mm_exact` column.

**The published `0.190` is untraceable because its checkpoint is gone.** `ablations_single_seed/`
has since been cleared, and no run before 2026-07-26 persisted an `estep_character.json`. So `0.190`
cannot be re-measured or tied to an artifact: it is withdrawn, not corrected.

**But the replacement is a like-for-like re-training.** The 2026-07-26 run reaches validation PPL
**139.30** against the deleted checkpoint's **139.3** and carries the same
`phi_mstep_max_matrix_norm=5` that named that ablation cell, so it reproduces the original K=20
configuration rather than merely resembling it. Its `0.153` is therefore comparable to the published
`0.190` as a like-for-like pair, with the difference attributable to the B-01 probe fix rather than to
config drift — the one caveat being that the deleted cell's `config.json` cannot be diffed directly,
so equivalence rests on the PPL match and the shared `mstep-phi-norm` setting.

**Sampling sensitivity.** `138.40_wikitext-103_K20_block_glk_linear_mix_s6` persists its
`estep_character.json`, so unlike `0.190` this number has an artifact behind it. Both checkpoints
were then re-measured under one matched protocol at two sample sizes, because the first-batch draw
turns out to matter:

| probe sample | K=20 pair share | K=300 pair share | gap |
|---|---|---|---|
| 8 sequences | 0.1476 | 0.2130 | +0.065 |
| 64 sequences | 0.1533 | 0.1964 | +0.043 |

The K=20 value is stable across the two draws (0.148 / 0.153); the K=300 value is not (0.213 / 0.196),
so the single-batch share carries real sampling noise at the larger width and no one number should be
quoted without its sample size. Take the 64-sequence row as primary: it is the larger sample and it
matches the protocol the in-run probe uses.

At 64 sequences the two runs give displacement 0.213 -> 0.222 (K=20) against 0.299 -> 0.319 (K=300),
`cos(dir_8, dir_1)` +0.984 against +0.965, and a step-1 share of 96% against 94%.

The `KL(q* || p)` row is dropped: the corrected probe does not compute it, so no re-measured value
exists and the published one carries the same defect as the rest of the column.

**It is inference, in a precise and unglamorous sense.** The E-step computes the exact stationary
point of a well-defined objective, and the iteration is a well-conditioned, nearly straight-line
contraction: one step covers 94-96% of the total displacement on the two `mm_exact` runs and every
later step moves in essentially the same direction (`cos` +0.965 to +0.984 at depth 8). The belief
does move — 22% of its prior's norm at K=20, 32% at K=300 — so this is not a no-op.

The correction STRENGTHENS this reading rather than weakening it. Step 1 takes 94% of the total
displacement at K=300, not the 70% published, so the E-step is more nearly one-shot than the original
table suggested.

**But the fixed point is a convex blend, not a deep computation.** The `mm_exact` fusion
`mu* = (a mu_p/sp + Sum_j w_ij mu_t/st) / P` puts 80% of the fused precision on the PRIOR and 20% on
the gauge-transported neighbors at K=300, and 85%/15% at K=20. Functionally
`mu* ~ 0.8 mu_p + 0.2 (attention-weighted transported neighbors)` — an attention layer with a strong
residual path, computed as a VFE stationary point instead of a dot-product softmax. The premise that
capacity comes from ITERATIVE minimization is not what carries the model; one aggregation is.

### The route comparison, and the measurement defect that hid it (B-11, 2026-07-26)

A `e_step_update='gradient'` K=20 run finished against the `mm_exact` sibling above. Their
`config.json` files differ in exactly ONE field, `e_step_update`, so unlike the width pair this is a
genuinely controlled comparison — the only single-variable ablation in this section.

Its published displacement of 4.208 against `mm_exact`'s 0.213 was a measurement artifact. The probe
reads the `mm_exact` fusion window where that spy fires and the block window otherwise, and the block
window ended at `vfe_block`'s RETURN value — after `head_mixer`, `cg_coupling` and the block norm —
while the `mm_exact` window reads `mu_star` before them. On the `mm_exact` checkpoint the same forward
pass reads **0.2126 on one window and 4.5791 on the other**, with identical anchors, so the whole gap
is the post-E-step transform. With `use_head_mixer=True` and `norm_type_block='none'` that transform
is the head mixer, it moves the belief 4.43x its own norm, and 98.5% of the motion is a scalar gain of
5.33 (cosine 0.985) — magnitude carrying no inferential content. Both windows now end at
`capture['converged']`, and each point publishes `displacement_window`.

| | `mm_exact` | `gradient` (published) | `gradient` (corrected) |
|---|---|---|---|
| displacement after 1 step | 0.2126 | ~~4.2084~~ | **0.3297** |
| ...after 8 steps | 0.2220 | ~~4.2806~~ | **0.3915** |
| `cos(direction_8, direction_1)` | +0.9843 | ~~+0.9320~~ | **+0.5988** |
| share of displacement taken by step 1 | 96% | ~~98%~~ | **84%** |
| test PPL | 138.40 | — | 141.38 |
| converged E-step F/token | 32.327 | — | 31.853 |
| zero-E-step CE penalty (nats) | 1.806 | — | 1.125 |

The `mm_exact` column is untouched; its window was never the defective one.

The corrected reading inverts the apparent one. The gradient route displaces about 55% further than
`mm_exact` rather than twenty-fold, and its trajectory is genuinely curved — `cos` +0.599 against
+0.984 — so on that route depth changes DIRECTION and not merely magnitude. That is the one place in
this section where the E-step looks iterative in the interesting sense.

It does not pay. The gradient route reaches a LOWER converged free energy (31.85 against 32.33) with
more iterative-looking inference, and still loses three points of test perplexity (141.38 against
138.40) while extracting LESS from the E-step by the held-out depth-0 counterfactual (1.125 nats
against 1.806). Better minimization of F, more curvature in the belief trajectory, worse language
modeling. Both F/token figures are single-test-batch evaluations of the same canonical F at the
converged belief, and both PPL figures are full held-out splits, so the dissociation is not a
sampling artifact of the headline metric — though it is a single seed at one width and should not be
generalized past that.

**The width claim: SUPPORTED by two points, but not isolated to width.** The original wording, "the
attention share RISES with scale (0.190 -> 0.298 from K=20 to K=300)", had both endpoints wrong and
one of them unreproducible. The corrected pair is 0.153 -> 0.196 at 64 sequences (0.148 -> 0.213 at
8), so the DIRECTION the claim asserted survives re-measurement at both sample sizes, at roughly half
the published magnitude.

It is not, however, a controlled width experiment, and this is exactly the error the original claim
made. Besides `embed_dim`, the two runs differ in twelve config fields. `kl_max` (160 vs 2400) is
dismissed: `guard_energy_klmax_frac` is 0.0 in both, so the clamp never fires and the pair mask is
untouched by it. What remains uncontrolled is training length (15k vs 180k steps, 12x), head count
(2 vs 10, hence `d_head` 10 vs 30, which sets `tau = kappa*sqrt(d_head)` and therefore the softmax
sharpness the share depends on directly), batch size (64 vs 16), `pos_phi_compose`
(`group_product` vs `bch`), `phi_mstep_max_matrix_norm`, and the code revision. Any of these could
move the share on its own.

Head count is the awkward one: `n_heads` must divide `embed_dim`, so 10 heads is unavailable at
K=20 and `d_head` cannot be held fixed while width varies in this family. A width sweep at fixed
`d_head` (varying `n_heads` with K) or at fixed `n_heads` (varying `d_head`) would separate the two,
and until one is run the honest ceiling on this claim is "supported by two points, confounded with
depth-of-training and head geometry" rather than "the share rises with width".

What IS robust across both widths, and does not depend on the share question at all: the E-step is
~95% one-shot with `cos` above +0.96, and the residual dominates the fusion at both scales (0.85 at
K=20, 0.80 at K=300).

### Where the context actually enters

Randomizing the prefix while holding the final token fixed, relative L2 displacement of that
position's mean:

| stage | K=20 | K=300 |
|---|---|---|
| raw prior (`s=0`, belief `=0`) — the control | **0.00000** | **0.00000** |
| after model-channel refine (`s=1`, belief `=0`) | 0.521 | 0.812 |
| after the belief E-step (`s=1`, belief `=1`) | 0.584 | 0.872 |

The control is exactly zero, confirming the raw prior is a pure per-token lookup. The belief is
emphatically NOT context-blind. But by the time the belief E-step runs, its input already carries
most of the context-dependence: the MODEL CHANNEL injects it, taking the representation from 0.000
to 0.812 at K=300, and the belief E-step raises it to 0.872.

### The structural consequence

`_refine_s` is itself an attention aggregation — it runs an E-step with `lambda_gamma` coupling under
`gamma_attention_prior='causal_alibi_noself'` and its own temperature — and its output becomes the
belief's prior. So despite `n_layers=1`, the trained model is effectively

    token lookup -> s-channel attention -> belief attention (with a 70-81% residual) -> linear decode

a TWO-attention-layer network, not one. That reframes the channel redundancy of Section 3: the two
"channels" are two attention layers doing overlapping work, which is why removing one costs little
once the other is trained to compensate.

Scope. Everything above is measured at `prior_source='model_channel'` with `s_e_step=True`. Under
`prior_source='token'` with `s_e_step=False` the first attention layer is absent and the prior is a
pure lookup, so the belief E-step must carry all of the aggregation and its pair-precision share
should rise correspondingly. That is a free check on any such checkpoint.
