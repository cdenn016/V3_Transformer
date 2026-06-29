# V3 Active-Inference Buildout — Closeout (2026-06-29)

This note closes the active-inference / Expected-Free-Energy (EFE) thread in the V3 transformer
(`vfe3`). It consolidates the spec, the experiments, and three investigations into a single program-
level conclusion. The short version: V3 now carries a validated, default-off, no-grad **pragmatic**
EFE policy scorer, and the **epistemic** side of active inference is structurally blocked in this
architecture and cannot be restored by any configuration toggle. The viable route to an epistemic
covariance is an architectural change that belongs to the multi-agent model, not V3.

## Conclusion

The deployed V3 EFE scorer is a preference-matching reranker: given an explicit preference over
outcomes it ranks candidate continuations by `G = risk + ambiguity` and selects with a policy
posterior, entirely under `@torch.no_grad`, with `policy_mode='none'` the byte-identical default. It is
honest and it works as a pragmatic reranker. What it does not have, and provably cannot acquire by a
toggle in this repository, is a live epistemic (information-seeking) term: the belief covariance does
not carry per-token uncertainty, the information-gain term is identically zero at the operating point,
and no no-leakage observation channel changes that. Active inference, as a negative empirical result on
its distinguishing feature, does not work in V3; its pragmatic shadow does.

## What was built and validated

The buildout followed the sealed spec
`docs/superpowers/specs/2026-06-28-active-inference-efe-policy-scorer-spec.md` and its phased plan.
Phase 0 added the shared belief-production seam `forward_beliefs` / `rollout_beliefs` and three
config-selected registries (`policy_mode`, preference, ambiguity), all default-off. Phase 1 added the
one-step scorer `efe_one_step` and the matched-tuning `logprob_control`, plus the controlled closed-loop
ring goal-steering task. Phase 3a added the belief-prefix cache (`vfe3/inference/belief_cache.py`) and
unlocked the horizon scorer `efe_rollout` (H > 1); under a causal attention prior the converged context
beliefs are invariant to appended tokens, so the rollout recomputes only the appended positions and the
cache reproduces the full recompute to float tolerance (golden-tested). All of this is merged to main
and the full suite is green (1347 passed, 1 skipped, 1 xpassed, 0 failures at the Phase-3a merge).

## The pragmatic result (ring Phase 1 to 2)

The ring experiment returns a clean **GO**: full EFE with the goal preference reaches success rate
1.000 across three seeds while every goal-blind arm (temperature-tuned log-prob, the data-distribution
control, nucleus, locally-typical, random) collapses to roughly chance (0.03 to 0.06). This confirms the
pre-registered pragmatic claim that an explicit goal preference steers the closed loop better than the
controls and baselines. The honest reading, recorded in the Phase-2 scope note, is that this GO is near-
tautological: the ring trains the action token independently of the goal, so only the goal-conditioned
arm can steer, and the matched-compute beam / best-of-N baselines are degenerate at the one-step horizon
the ring supports. The ring validates pragmatic preference-steering and is, by construction, no evidence
about epistemic active inference.

## The epistemic wall (three investigations)

Three independent investigations converge on the same structural limit.

The pre-registered sigma-validation gate **failed**. On the operating-point checkpoint
`tr(Sigma_q)` does not predict realized cross-entropy: the Spearman correlation is `-0.137` (wrong
sign), the trace is near-constant (coefficient of variation `0.044`), and the stratified cross-entropy
trends the wrong way. A failed gate keeps the `sigma_mc` ambiguity, the epistemic-only arm, and every
sigma-derived claim reported-only, and the information-gain term inert
(`2026-06-28-sigma-gate-prereg.md`, `Result` section).

The why-sigma-collapses investigation explained the cause. The live belief-covariance gradient has no
data or likelihood precision channel; it is self-coupling plus belief-coupling and nothing else. The
covariance is therefore pinned near a per-token prior table that is itself shrunk toward a single shared
centroid, by a single low-rate E-step, with the Fisher and SPD geometry contracting rather than
dispersing. The retraction-clamp hypothesis was refuted. The residual variation is anti-correlated with
cross-entropy only as a mean-borne frequency confound, not a covariance-readout effect
(`2026-06-29-why-sigma-collapses.md`).

The observation-term no-leakage investigation and its adversarial verification then closed the obvious
remedy. Wiring the canonical `-E_q[log p(o|x)]` term into the E-step cannot make the covariance
epistemic for next-token language modeling, for three separable reasons. First, definitionally, a belief
used to predict `x_{i+1}` may not acquire precision from `x_{i+1}`; that is the forbidden target leak
that produces the train-perplexity-collapse, validation-perplexity-explode pattern. Second, the only
trivially legal observation, a current-token reconstruction, targets the same per-token table that
`encode` already used to initialize the belief (`q = p`), so it is functionally a rescaling of the alpha
self-coupling, the same fixed point made more confident. Third, and most sharply, the Gaussian
covariance gradient `0.5 H^T R^{-1} H` is data-independent, so for a fixed observation model it injects a
near-constant uniform shrinkage that deepens the collapse rather than making it epistemic
(`2026-06-29-observation-term-e-step-no-leakage-investigation.md`,
`2026-06-29-observation-term-verification.md`).

## Why it is structural, not a bug

The three reasons above are properties of the design, not omissions. The E-step is target-blind by
construction, which is exactly what keeps causal language modeling honest; the belief is initialized
from the token's own identity and self-couples back to it; and the geometry contracts toward the prior.
Per-token epistemic covariance from legal data would require both a heteroscedastic or context-dependent
observation model and a prior that genuinely disagrees with the datum. This matches the manuscript's own
framing that the observation-free E-step is a sanctioned structural-EM reduction rather than a defect,
and the prior reconciliation that reached the same conclusion. The sigma-collapse should be accepted as
a structural property of the target-blind, encode-from-token-identity next-token model.

## The only viable route

The one principled way to obtain a covariance the likelihood can genuinely correct is a top-down or
cross-scale prior with a causal likelihood on already-observed data: a prior that depends on a higher
scale rather than on `x_i`, so the datum carries non-redundant information. That cross-scale tower is
explicitly out of scope in this single-scale transformer (`prior_bank.py:213-221` assigns it to the
multi-agent model). The epistemic active-inference thread therefore continues, if at all, in
`MAgent_Model`, not in V3.

## State of the code

Everything is default-off and the pure path is intact. `policy_mode='none'` leaves `forward` and
`generate` byte-identical to the pre-buildout baseline. The shipped surface is the belief seam, the
three registries, the `efe_one_step` / `logprob_control` / `efe_rollout` scorers, the belief-prefix
cache, the ring task, and the sigma-gate measurement and artifact. The masked key-value retrieval task
foundation (`vfe3/inference/masked_retrieval.py`) is committed as a standalone module but its closed-
loop runner and arms were deliberately not built and the experiment not run, because the predicted
outcome is null for the same structural reason. The `free_energy(log_likelihood=...)` hook remains a
documented inert stub; wiring it is not recommended on the basis of the verification above.

## What remains optional in V3

The pragmatic scorer is complete. The only remaining scheduled experiment with any signal is the
language-modeling matched-compute comparison (`efe_rollout` H >= 2 versus beam / best-of-N on WikiText),
where the model's sequence log-probability carries information unlike the ring; it is likely marginal,
because language modeling supplies no per-episode goal preference, so EFE reduces to roughly negative-
log-likelihood plus an ambiguity confidence term and the comparison mainly tests whether that term beats
beam. Phases 4 and 5 (the pure prior-bank repeat and the train-time EFE regularizer) are gated on a
working epistemic term and are therefore blocked. No further epistemic experiment in V3 is warranted by
the evidence.

## Pointers

- Spec: `docs/superpowers/specs/2026-06-28-active-inference-efe-policy-scorer-spec.md`
- Buildout assessment: `docs/research/2026-06-28-active-inference-buildout-plan-investigation.md`
- Binding debate: `docs/debates/2026-06-27-active-inference-lm-efficacy/`
- Phase-2 scope note: `docs/research/active-inference/2026-06-28-phase2-scope-note.md`
- Sigma gate (pre-registration + FAIL result): `docs/research/active-inference/2026-06-28-sigma-gate-prereg.md`
- Why sigma collapses: `docs/research/active-inference/2026-06-29-why-sigma-collapses.md`
- Observation-term investigation: `docs/research/active-inference/2026-06-29-observation-term-e-step-no-leakage-investigation.md`
- Observation-term verification: `docs/research/active-inference/2026-06-29-observation-term-verification.md`
- Masked-retrieval design (foundation only): `docs/research/active-inference/2026-06-29-masked-retrieval-task-design.md`
