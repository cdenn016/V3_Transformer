# Active-Inference EFE Policy Scorer — Pre-Registration Amendments (2026-06-28)

Amends `docs/superpowers/specs/2026-06-28-active-inference-efe-policy-scorer-spec.md` (Section 4).

These three amendments were made during Phase 1 implementation, **before the official sealed run**,
after smoke runs on seed 6 showed the pre-registration as written was mechanically broken: the EFE
score could not drive policy selection. No empirical conclusion was drawn from the broken runs; they
were debugging diagnostics. Each amendment fixes a flaw in operationalizing the spec's own stated
intent, not a post-hoc change to rescue a result. Discovered and applied on branch
`feat/active-inference-efe-policy-scorer`.

## A1. Candidate generator: top-Kp tokens to the three control actions

Section 4.2 sealed the candidate generator as "the top-Kp next-token set under the current model
logits, Kp=8." On the ring control task this admits non-action tokens (for example `CUR`) into the
menu. Appending such a token is out of distribution; the model's diffuse prediction there has a
moderate cross-entropy to the goal-peaked preference (roughly uniform over states, about 17 nats),
which beats a genuine one-step move toward a far goal (whose predicted next state is still far, risk
about `beta_C*(d-1)`, around 30 for `d~7`). The policy then selects a non-action that maps to a wasted
STAY and the agent freezes. The control-task policy space is the action set, so the candidate
generator is amended to the three action tokens `{DEC, STAY, INC}` (config `candidate_mode="actions"`).
The top-Kp generator is retained as `candidate_mode="top_k"` for the language-modeling continuation
setting. Verified on seed 6: restricted to the action set, the argmin-G candidate is the
goal-reducing action on every sampled state/goal pair.

## A2. Preference floor: exactly zero to finite "approximately zero"

Section 4.1 specifies `p_task` with "approximately zero mass on non-state tokens." The implementation
used exactly zero mass (utility `-inf` on non-state tokens), which makes the forward
`KL[q(o|pi) || p_task]` diverge to `+inf` wherever the model's `q` carries any off-state mass. That
collapses every candidate's score to `+inf` and the policy posterior to `nan`; `argmax(nan)` returns a
fixed index, so all arms tie. Amended to a finite floor utility one ring-step below the farthest state,
`-beta_C*(M/2 + 1)`, giving non-state tokens negligible-but-nonzero mass, consistent with the spec's
"approximately zero." A defensive `clamp_min(-60)` on the preference log-probabilities is also added in
the scorer so any zero-support preference stays finite. The same finite-floor fix is applied to the
`held_out_predictive` control's state-marginal preference.

## A3. Random lesion gate: strict global-argmin to "EFE clearly beats random"

Sections 4.6/4.7 require "random-score must be clearly worst" (falsified if "random-score is not
clearly worst"). The implementation tested random equal to the strict global-minimum success rate. At
v1 every goalless arm (`ambiguity_only`, `flat`, `p_data`, `logprob`) legitimately collapses to about
random, so they cluster within sampling noise and any one can edge random by a single episode without
weakening the lesion. The gate is amended to operationalize the falsifier's intent directly: the lesion
passes if and only if full EFE beats random by more than `delta_min`. On the seed-6 smoke full EFE was
1.000 versus random 0.059, so the lesion passes by a wide margin; the strict-argmin form spuriously
failed only because `ambiguity_only` (0.058) edged random (0.059) by one episode in 5000.

## Unchanged

All other sealed constants (Section 4.7) stand: `m=16`, `T_ep=10`, `V=32`, `N_ep=5000`, `S=16`,
`beta_C=5.0`, the `gamma` grid `{0.5,1,2,4,8}`, `delta_min=0.05`, `alpha=0.05`, FDR `q=0.05`, the seed
list `(6,23,64)`, the 15k-step budget, the predictive-adequacy precondition `>= 0.98`, and the
sigma-gate thresholds.

## Status

Smoke validation (seed 6, 3k steps, 5000 test episodes) passed decisively: full EFE 1.000 versus all
goalless arms about 0.06, with the pragmatic decomposition behaving as predicted (`risk_only` 1.000;
`ambiguity_only` about random). The official verdict awaits the sealed run (seeds 6/23/64, 15k steps,
5000 paired test episodes). The result remains a v1 pragmatic-steering validation only: epistemic
active inference (`I` identically zero at v1), the gauge generative decode (`use_prior_bank=True`,
Phase 4), and any language-modeling benefit are out of scope for this gate.
