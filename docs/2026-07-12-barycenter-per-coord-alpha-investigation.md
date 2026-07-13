# Barycenter r M-step vs lambda_alpha_mode='state_dependent_per_coord' — investigation (2026-07-12)

Question investigated (two legs):

1. Is the claim that `r_update_mode='barycenter'` is the *exact* closed-form M-step for the
   hyper-prior centroid r in the scored `s_e_step=False` regime still correct when
   `lambda_alpha_mode='state_dependent_per_coord'` (with `prior_source='model_channel'`,
   `learnable_r=True`, `lambda_h_mode='constant'`)?
2. The user empirically observes *better* results with `r_update_mode='barycenter'` +
   `s_e_step=True` — the combination `config.py` warns is an INEXACT M-step. Why can that win?

Method: multi-agent workflow (theory derivation, exhaustive code-path trace, per-coord
envelope-kernel verification, numerical finite-difference experiment on the real model at tiny
dims, training-dynamics analysis), followed by two adversarial skeptics attacking the central
claims. Both skeptics returned UPHELD.

## Verdict, leg 1: the exactness claim is UNAFFECTED by per-coord alpha

Under `s_e_step=False`, r enters the training loss through exactly one term: the scored
hyper-prior block added at `model.py:1553` under the gate `lambda_h > 0.0 and not s_e_step`
(`model.py:1542`). Its entire compute chain — `_hyper_prior_kl` -> `_hyper_prior_weighted` ->
`_hyper_prior_term` (`model.py:1596-1676`) — reads only `renyi_order`, `divergence_family`,
`kl_max`, `lambda_h`, `lambda_h_mode`, `b0_h`, `c0_h`. `cfg.lambda_alpha_mode` appears nowhere in
it. The per-coord alpha lives on the belief self-coupling `D_k(q_i||p_i)`, and under
`prior_source='model_channel'` the prior is `p_i = s_i` (`prior_bank.py:550,555`), never r; the
decode readout likewise never reads r (`prior_bank.py:557-568`). An exhaustive word-boundary grep
for `r_mu`/`r_sigma_log` over `vfe3/` bounds the loss-graph consumers to exactly two mutually
exclusive, `s_e_step`-gated sites: `_hyper_prior_kl` (`model.py:1619-1620`, the `s_e_step=False`
regime) and `_refine_s` (`model.py:764-765`, the `s_e_step=True` regime). Everything else is
diagnostics or optimizer bookkeeping.

Numerical confirmation (real `VFEModel`, K=4, V=8, N=4, B=2, 1 layer/head, 2 e-steps, CPU): with
`s_e_step=False` the autograd gradient of the full training loss w.r.t. `(r_mu, r_sigma_log)` is
**bitwise identical** (max diff exactly 0.0) between `lambda_alpha_mode='constant'` and
`'state_dependent_per_coord'` at identical parameters, while the losses themselves differ
(2.65840721 vs 2.65861797 — alpha genuinely live in the belief channel). The r-dependent subgraph
shares no intermediate tensor with any alpha-touched op, so the invariance is structural, not
merely numerical.

The closed form itself is correct: `barycenter_r_` (`prior_bank.py:535-540`) computes
`r_mu = mean_v s_mu_v`, `r_var = mean_v[s_sigma_v + (s_mu_v - r_mu)^2]`, the unique minimizer of
the unclamped uniform-over-vocab `sum_v KL(s_v||r)` for diagonal Gaussians (derivative computation
re-done by the theory agent; second derivative positive, unique minimum). Numerically the written
r is an exact stationary point (gradient inf-norms 8.9e-8 / 2.4e-7 fp32; fp64 confirms), and it
matches the hand moment-match to 0.0 through the real `bounded_variance_from_log` round-trip.

The `config.py:1687-1689` exactness-warning predicate — which checks `renyi_order`,
`divergence_family`, `lambda_h_mode` and omits `lambda_alpha_mode` — is therefore correct as
written, not an omission. The two pre-existing caveats stand and are both alpha-independent:

- **Frequency weighting**: the scored term reduces with `mean()` over (B,N) token occurrences,
  the barycenter is uniform over vocab types. Measured: at the barycenter, the scored term's
  r-gradient is ~2.2e-2 for a non-uniform batch and ~2.8e-9 for a batch covering each vocab id
  exactly once.
- **kl_max clamp**: the scored KL is clamped, the barycenter is unclamped. Sharper than
  previously documented: a vocab row with `KL(s_v||r) > kl_max` contributes *zero gradient* to
  the scored term but *full weight* to the barycenter, so for far-drifted outlier rows the two
  updates can disagree in direction, not just magnitude.

## Verdict, leg 2: barycenter + s_e_step=True is inexact as documented, but mechanistically the stronger update

Confirmed inexactness: under `s_e_step=True` the assembled loss equals the CE **exactly**
(measured `loss == ce` to the last bit; scored hyper-prior and gamma blocks are gated off at
`model.py:1542,1554`). r reaches the loss only as the frozen prior of the unrolled `_refine_s`
E-step (`model.py:764-776`), whose refined s1 replaces both q0 and the belief prior
(`model.py:975-983,1015`) — note this rebinding is unconditional under `s_e_step`, independent of
`prior_source`. At the barycenter point the full-loss r-gradient is nonzero (inf-norm ~2.7e-2 in
the tiny model): the barycenter is not the argmin of what the model minimizes, exactly as the
`config.py:1669-1678` warning says. Under this regime the r-gradient also acquires a *small*
`lambda_alpha_mode` dependence (max diff ~7.6e-5, ~0.3% of the gradient magnitude) because s1
feeds the alpha-weighted belief self-coupling — real but second-order; it does not change the
character of the inexactness.

Why the "inexact" combination can still win. The consistent alternative (`r_update_mode='gradient'`
under `s_e_step=True`) trains r on the weakest gradient path in the model:

1. **No anchoring objective.** With the scored `KL(s||r)` gated off, nothing in the loss says r
   should be a centroid; r is a free 2K-dim steering parameter for wherever the CE momentarily
   points.
2. **Weak, batch-limited signal.** One Jacobian-vector product through the (n_e_steps) unrolled
   s-refine; the r prior arm is the minority precision in the mm_exact fusion
   (`lambda_h=0.25` vs `lambda_gamma=0.75` pair arm); only batch tokens contribute, so rare
   vocab rows never inform r.
3. **AdamW scale-free noise amplification.** `g/sqrt(v)` is magnitude-invariant, so a tiny
   high-variance gradient still produces near-full-LR steps — with no restoring force, a
   batch-noise random walk.
4. **kl_max deadband.** The kernel self-mask `1[D(s_i||r) < kl_max]` zeroes both the prior arm
   and the r-gradient for saturated rows (the footgun already recorded in the user's own
   `train_vfe3.py` comment), and the dropout is self-reinforcing.
5. **Estimator fragility.** The path is severed outright under `straight_through`/`detach`
   (`config.py:2226-2240`); the codebase enumerates gradient-mode-r-under-s_e_step in two
   separate freeze-warning predicates.
6. **Biased.** r also shapes the forward through the detached `gamma_as_beta_prior` fold
   (`model.py:991-1001`), invisible to the attached gradient.

The barycenter replaces all of that with a zero-variance, full-vocab, closed-form population
statistic applied after every successful optimizer step (`train.py:669-670`): the exact M-step of
the population empirical-Bayes objective `sum_v KL(s_v||r)`. Its `r_var` absorbs the between-type
spread, which softens the shrinkage pull on outlier rows (the prior-arm precision is
`a_i/sigma_p`) and keeps the kl_max deadband from binding — variance-calibrated, James-Stein-style
shrinkage. It also resolves a weight-decay drift mismatch: the s tables inherit global
`weight_decay=0.02` while gradient-mode r has wd=0.0 (`train.py:274-275` — the "WD shrinks r"
hypothesis is FALSE) and no tether; the barycenter re-tethers r to the decayed population every
step. Under barycenter mode r is `requires_grad=False` (`model.py:226`), excluded from AdamW
(`train.py:269`), so the combo is a bilevel scheme with r defined as a moment-match closure
`r := r(s)` — a two-timescale, target-network-style update. One honest framing caveat from the
skeptic: "generalized-EM" is charitable — no single objective is monotonically descended by the
alternation (the barycenter's objective is not a term of the s_e_step loss), so "consistent
population target with far better signal properties" is the defensible statement. "Inexact
M-step" does not mean "descends nothing."

## Cheap falsification ablations (mechanism ranking)

The mechanisms above all exist in the code; *ranking* their contribution to the observed win
requires ablations. Cheapest discriminators: (E1) gradient mode with r's groups at lr/10 (tests
the Adam noise-walk); (E2) log the self-mask saturation rate, rerun with kl_max huge (tests the
deadband); (E3) barycenter computed from refined s1 / occurrence-weighted (tests whether full-vocab
batch-independence is load-bearing); (E4) ablate the between-type term in `r_var` (tests the
softened-shrinkage mechanism); (E5) EMA-smoothed barycenter (tests two-timescale smoothing per
se); (E7) frozen r control (tests whether tracking the population adds value over mere stability).

## Incidental findings

- **Live config drift (flagged, not touched):** at investigation time `train_vfe3.py:287-290` on
  disk read `r_update_mode='gradient'`, `learnable_r=False`, `s_e_step=True`. Under that exact
  file the barycenter never fires (`train.py:669` requires `learnable_r=True` and
  `r_update_mode='barycenter'`) and r is not in the optimizer, so r stays frozen at init
  (`r_mu=0`), serving only as a fixed shrinkage target inside `_refine_s`. If a run intended to
  use the barycenter combo, both toggles need to be set.
- **Test gap (per-coord alpha kernel):** the suite pins kernel==oracle for
  `state_dependent_per_coord` only at a saturation edge case (N=1, K=2, identity omega, pair term
  inert). No generic-random-point kernel-vs-oracle test ships for the per-coord form (the generic
  test at `tests/test_gradients_kernels.py:46-53` covers only per-position `state_dependent`). An
  ad-hoc probe (N=3, K=3, random glk transport, per-coord (K,) b0/c0) agreed to 2.4e-7, so the
  implementation is correct today, but the suite would not catch a per-coord-only regression in
  the pair/self interaction at a generic point.
- **Per-coord envelope theorem verified:** F_self is separable in k, alpha_k* = c0_k/(b0_k+D_k)
  is the unique coordinate-wise stationary point, and the envelope cancellation reduces the total
  derivative to sum_k alpha_k* dD_k — implemented identically on the kernel route
  (`alpha_gradient_coefficient`), the oracle (undetached alpha*D + R), and the mm_exact fusion;
  autograd-vs-detached-coefficient identity measured exact (0.0). The D_k feeding alpha_k* is the
  per-coordinate *clamped* divergence on every route (consistent across kernel/oracle/mm/F-value),
  so the envelope coefficient floors at c0/(b0+kl_max) for saturated coordinates.
- Test runs during verification: 53/53 passed (`test_alpha_i.py`, `test_gradients_kernels.py`,
  `test_mstep_self_coupling.py`, `test_fullcov_alpha_roadmap_2026_06_13.py`) and 92 passed / 1
  skipped (`test_free_energy.py`, `test_p3_pairwise_stats_reuse_20260711.py`), read from junitxml.
