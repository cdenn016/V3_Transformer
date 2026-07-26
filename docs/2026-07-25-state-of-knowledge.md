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
