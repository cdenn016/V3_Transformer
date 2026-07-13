# ablation.py vs train_vfe3.py per-step timing investigation (2026-07-11)

## Finding

The reported "~10% slowdown of ablation.py vs train_vfe3.py under identical configurations" is
not an entry-point effect. Both scripts call the identical `vfe3.train.train()` loop, and an
adversarially verified audit of the loop, loaders, imports, and process state found no structural
difference costing measurable time (ablation's silent intervals do strictly less periphery work).
The gap decomposes into two config toggles that differed between the runs that were actually
timed, verified per-toggle from each run's on-disk `config.json` and `metrics.csv`:

| what ran (on-disk config.json)                    | gamma_as_beta_prior | reuse_pairwise_kl_stats | ms/step   |
|---------------------------------------------------|---------------------|-------------------------|-----------|
| train_vfe3 `20260711-170008`                      | False               | True                    | 74.1      |
| train_vfe3 `20260711-165636`                      | False               | False                   | 76.8      |
| ablation `lambda_h` cells (all six, 2026-07-11)   | True                | True                    | 78.4-79.0 |
| train_vfe3 `140.85_gamma-beta-prior` (pre-P3 era) | True                | False (field absent)    | 81.4      |

`gamma_as_beta_prior=True` costs +4.4 ms/step (~6%); `reuse_pairwise_kl_stats=False` costs
+2.6 ms/step (~3.5%); `lambda_h` 0 vs 0.25 is timing-negligible (~0.3 ms, bounded by the matched
`20260711-182831/182851` runs). The additive decomposition 74.1 + 2.6 + 0.3 + 4.4 = 81.4
reproduces the pre-P3 gbp=True run to 0.1 ms, and 81.4 vs 74.1 is +9.85% -- the observed "~10%"
appears when a gbp=True measurement from the pre-`reuse_pairwise_kl_stats` era is compared
against the current gbp=False + reuse=True train_vfe3 baseline. Today's like-for-like pairing
(ablation cell vs run `170008`) is +5-6%.

A premise correction from the same evidence: although `train_vfe3.py:383` carries
`gamma_as_beta_prior=True` in the current working tree, the `config.json` of all four train_vfe3
runs launched 2026-07-11 (16:56, 17:00, 18:28:31, 18:28:51) records `false`, while every ablation
cell records `true`. The timed comparison was never both-True; the source file was flipped after
those runs. Prediction: rerunning `train_vfe3.py` as it now stands (gbp=True, reuse=True) should
log ~7.85-7.9 s per 100 steps, matching the ablation cells with no gap.

## Mechanism of the gamma_as_beta_prior cost

With `s_e_step=True` the gamma energy is otherwise never computed per training step (the scored
`_gamma_coupling_term` is gated on `not s_e_step`, model.py:1556); gamma enters only through
`_refine_s`, which both configs pay identically. The fold at model.py:975-985 therefore adds one
full extra no-grad pass per forward -- `_fold_gamma_prior` -> `_gamma_energy` (model.py:2118-2136,
1689-1706): a third redundant flat-transport build (2 batched `matrix_exp` calls over 16,384
10x10 blocks at B=64, N=128, H=2, d_head=10), transported key means and covariances at
(B, N, N, K) ~ 84 MB each, the pairwise diagonal KL, the gamma softmax, and the probability-space
mixture pi = (1-w) softmax(B) + w gamma with renormalization, log, and -inf re-masking. This also
changes the math, not just the speed, which accounts for the earlier observation that the two
entry points produced different results under "identical" settings.

Two pieces of reclaimable compute sit inside that pass. First, `shared_omega` from
`share_refine_s_transport` (model.py:946) is not passed to the fold at model.py:982, so the fold
rebuilds the identical flat factored transport a third time per forward. Second, the fold's
internal `build_belief_transport` call (model.py:1689-1691) forwards none of the Tier-1 perf
kwargs, so `transport_mean_per_head` and `exp_fp64_mode` sit at their defaults regardless of the
config. Passing the prebuilt transport (and the Tier-1 kwargs) into `_fold_gamma_prior` is an
exactness-preserving optimization that would claw back a chunk of the 4.4 ms for configs that
keep gbp on permanently.

## What was ruled out

GPU contention (the four clean lambda_h cells ran strictly sequentially 17:07-18:26 with wall
times 1176.8/1180.2/1181.0/1181.3 s, spread 0.38%), thermal drift (the first cell was the
fastest), run-to-run variance (0.4-0.7%, an order of magnitude below the gap), killed-run bias
(run `170008`'s per-window times are flat over all 4900 steps), dataloader/import/logging
asymmetry between the entry points, and any interaction in which gbp=True disables P3 reuse (the
gate in kernels.py checks only family/divergence/fp32; `__post_init__` only validates).

No code was changed in this investigation.
