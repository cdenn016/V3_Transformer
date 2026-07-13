# Sigma-Validation Gate — Pre-Registration (2026-06-28)

The binding pre-registration for the sigma-validation gate of the active-inference EFE policy scorer,
referenced by `docs/superpowers/specs/2026-06-28-active-inference-efe-policy-scorer-spec.md`
(Sections 2.7, 4.5, 4.7, Guard 4). The measurement is implemented in `vfe3/inference/sigma_gate.py` and
run by `sigma_gate_measure.py`; the result is written to
`vfe3_policy_results/sigma_gate/<checkpoint_id>.json`.

## Why the gate is binding, not advisory

At a sigma-free point belief the model's belief over the latent is a delta, `q(s|pi) = delta(mu)`, so
the mutual-information bridge that defines the epistemic value of a policy collapses,
`I = H[q(o|pi)] - E_{q(s|pi)} H[p(o|s)] = H[p(o|mu)] - H[p(o|mu)] = 0`, and this holds at every horizon,
not only at one step. A live epistemic term therefore requires belief covariance `sigma` that genuinely
carries outcome uncertainty. Before any sigma-derived quantity may be named an epistemic or ambiguity
value, before the Monte Carlo ambiguity estimator `sigma_mc` is unlocked, and before the epistemic-only
and shuffled-sigma arms may be claimed rather than merely reported, `sigma` must demonstrably predict
realized outcomes on the checkpoint hosting the arm. The gate measures exactly that. The structural lock
is already in place: `policy_sigma_ambiguity_validated` defaults False and config validation raises
unless it is paired with a `policy_sigma_gate_artifact` reference to a PASS record (config Guard 4), and
`sigma_mc` raises until then.

## Protocol

On the checkpoint hosting the arm, replay the belief path on a held-out split and join, position by
position, the per-token belief-covariance trace `tr(Sigma_q)`, the realized decode cross-entropy on the
gold token, the predicted confidence (max softmax probability), and the correctness indicator
(argmax equals gold). This is the `belief_ce_bank` join, which reinstates the s-refine anchor and the
precision-bias fold so the traced covariance is the one whose mean produced the logits. From the aligned
per-token tensors compute the four gate statistics below.

## Sealed thresholds

The numeric thresholds are sealed by spec Section 4.7 and may not be retuned to rescue a result:
`sigma_ce_spearman >= 0.2`, sigma-binned expected calibration error `< 0.05`, and the bootstrap
confidence level `alpha = 0.05` (a 95 percent interval). The seed list recorded in the artifact is
`(6, 23, 64)`.

## Operationalizations (recorded here so they are auditable)

The spec states the gate in prose; the following concrete operationalizations are fixed before any run.

The Spearman rho between `tr(Sigma_q)` and realized cross-entropy is the headline statistic, with a
paired-token bootstrap percentile confidence interval (resample tokens with replacement, recompute the
rank correlation, take the 2.5 and 97.5 percentiles). The measured floor is a permutation null: shuffle
`sigma` against cross-entropy, recompute the rank correlation many times, and take the 95th percentile of
that null distribution as the noise band the real correlation must clear. The bootstrap CI lower bound
must exceed both zero and this floor.

Sigma-stratified cross-entropy uses equal-count sigma-quantile strata (ten by default). The sealed
`monotone` flag is True iff the stratum-mean cross-entropy is non-decreasing across strata ordered by
sigma; the rank correlation of stratum index against stratum-mean cross-entropy (`mono_spearman`) is
reported alongside as the robust trend diagnostic, so a single noisy stratum that breaks strict
monotonicity is visible rather than silently failing the gate.

The sigma-binned expected calibration error uses equal-count sigma-quantile bins (ten by default) and
averages the absolute gap between mean predicted confidence and mean realized accuracy within each bin,
weighted by bin size. This measures whether the model stays calibrated within each uncertainty stratum,
the second clause of the spec's third condition.

## Pass rule

PASS holds iff all of the following are true: the Spearman rho is at least 0.2; its bootstrap CI lower
bound exceeds zero and the permutation floor; the sigma-stratified cross-entropy is monotone; and the
sigma-binned expected calibration error is below 0.05. Any failure yields FAIL. The record carries every
statistic plus a single PASS or FAIL stamp, the checkpoint id, the spec commit hash, and the seed list.

## Status and expected outcome

The gate has not been measured at the v1 operating point (`embed_dim = 20`, 15k steps, linear decode,
`use_head_mixer=False`). The only sigma calibration numbers in hand come from a disjoint 60k-step
head-mixer regime, where `sigma_ce_spearman` sat near 0.176 to 0.19, which is below the 0.2 threshold and
does not transfer to the operating point. The gate may therefore fail. This measurement is run first in
Phase 3 precisely because it is the cheap, decisive test of whether the epistemic half of Phase 3 is
achievable at all: a forward-pass eval over a few held-out batches, no new task or cache required.

If the gate PASSES, the epistemic arms (`sigma_mc`, epistemic-only, shuffled-sigma as a meaningful
contrast) may be claimed and the masked-retrieval epistemic task is worth building, since the
information-gain term can then carry signal. If the gate FAILS, the information-gain term stays at its
inert value, all sigma-derived arms remain reported-only, and no epistemic active-inference claim is
available at this operating point; the honest response is to report the negative result and reconsider
whether a different regime produces a validated sigma rather than to build an epistemic task that cannot
show an epistemic effect.

## Result (2026-06-29) — FAIL

The gate was measured at the v1 operating point on a prior-bank (KL-to-prior) decode checkpoint
(`embed_dim = 20`, 15k steps, `use_head_mixer=False`, `use_prior_bank=True`). This is the decode path on
which the belief covariance enters the readout, and so the strongest available test rather than a
favorable one. The outcome is FAIL, and not by a narrow margin: the headline statistic is negative. Over
40,960 held-out tokens the Spearman correlation between `tr(Sigma_q)` and realized cross-entropy is
`-0.137` with a bootstrap 95 percent interval of `[-0.147, -0.127]`, against a permutation floor of
`0.008`. The correlation is significantly different from zero and clears the noise band, but with the
wrong sign: higher belief uncertainty predicts lower realized surprise. The sigma-stratified
cross-entropy confirms this, trending downward across strata (`mono_spearman = -0.43`), with the
highest-sigma decile carrying the lowest mean cross-entropy (3.30 nats) and the lowest-sigma decile the
highest (5.50 nats). The artifact is `vfe3_policy_results/sigma_gate/wikitext103_ed20_15k.json`, bound to
spec commit `c05b927`.

Two facts explain the failure. First, the trace barely varies: its coefficient of variation is `0.044`,
so `tr(Sigma_q)` occupies a tight band (roughly 33.9 to 39.9) across all tokens. The belief covariance
has effectively collapsed to a near-constant scale and carries little token-discriminative information
regardless of sign. Second, what little variation exists is anti-aligned with predictive difficulty
rather than aligned with it. A near-constant, sign-inverted sigma is precisely the condition under which
the mutual-information bridge `I = H[q(o|pi)] - E_q H[p(o|s)]` has nothing to act on; the structural
argument that `I` collapses at a sigma-free point belief is realized empirically here as a sigma that is
informative-free in practice. One clause of the gate does hold in isolation: the sigma-binned expected
calibration error is `0.019`, below the `0.05` threshold, so the model stays calibrated within each
uncertainty stratum. Calibration within strata is not predictiveness across them, and the gate as a whole
fails.

The pre-registered consequence follows without amendment. The information-gain term stays at its inert
value; the `sigma_mc` ambiguity estimator, the epistemic-only arm, and shuffled-sigma remain
reported-only; `policy_sigma_ambiguity_validated` cannot be set, and config Guard 4 correctly refuses
this FAIL artifact. The EFE policy scorer is therefore a validated pragmatic preference reranker built on
the risk term and the likelihood-entropy ambiguity term, not an exploratory agent with a live epistemic
value. The conclusion holds across both decode paths examined: the linear decode gave a weakly positive
but sub-threshold `0.105`, and the prior-bank decode gives the negative result recorded here. No
epistemic active-inference claim is available at this operating point. In the language of the
pre-registration, the honest reading is to report the negative result rather than build a masked-retrieval
epistemic task that could not show an epistemic effect, and to treat whether a different regime (a larger
latent dimension, longer training, or an alternative uncertainty parameterization) yields a validated
sigma as an open question for a future, separately pre-registered measurement.
