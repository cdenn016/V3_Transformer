# Verdict (scope) - active-inference-lm-efficacy

## Verdict

Outcome: BLUE_WINS.

The claim is well-formed enough to adjudicate as a theory-and-design claim, not as a claim that the present V3 code already implements active-inference policy selection. It is a single but compound sentence: it asserts that an opt-in inference-time EFE policy scorer is legitimate, that the scorer is worth empirical testing, and that a train-time EFE replacement is premature. The debate answered that scoped proposition. Red successfully forced the missing contract into view, but Blue accepted those constraints and defended the narrow claim as a finite, explicit, no-grad policy-evaluation layer whose failure modes can be tested.

| Check | Result |
|-------|--------|
| Single declarative sentence? | Yes: "For V3-style language modeling, an opt-in inference-time expected-free-energy policy scorer over explicit candidate continuations or agent sets is theoretically legitimate and empirically worth testing, while a train-time EFE replacement is premature." |
| Falsifiable? What observation would refute? | Yes, under the narrow design reading. Refuters include failure to specify A/B/C/D/E analogs; unchanged rankings when C changes; no degradation under sigma shuffling or ambiguity removal; matched-compute log-probability, beam, truncation, or shallow sigma baselines erasing the gain; or post-hoc rescue by changing preferences, horizon, candidates, or proxies. |
| Domain (theory / code / both)? | Theory plus empirical test design, with code feasibility as supporting evidence. It is not a present-implementation claim. |
| Key terms anchored? | Anchored: expected free energy, finite policy set, risk/ambiguity, active-inference A/B/C/D/E, no-grad inference-time scorer, train-time replacement. Still needs implementation-time anchoring: "V3-style," "agent sets," the outcome space, preferences C, horizon, policy prior/precision, and target metrics. |

## Decisive scope evidence

The frame-breaking risk was visible in the evidence pack: current evidence shows only that "the theory supplies a policy-selection functional" and that V3 has plausible hooks; "efficacy remains an empirical question requiring matched-compute controls" [01_evidence.md:57]. The same section says that without explicit preferences and transition/outcome definitions, "EFE is only a label for reranking" [01_evidence.md:59].

Blue answered that risk rather than bypassing it. Its rebuttal concedes that "a scalar reranker with active-inference vocabulary is not active inference" and lists the canonical requirements [03_blue_rebuttal.md:5]. It also concedes that V3 lacks a current public policy layer [03_blue_rebuttal.md:7]. The decisive line is Blue's scoped defense: the defended claim is that "a finite, explicit, no-grad policy scorer over candidate continuations or agent sets is a legitimate active-inference experiment and worth a falsifiable V3 test" [03_blue_rebuttal.md:11]. That is the claim in 00_claim.md read against the evidence context, so the debate answered the actual proposition rather than an out-of-scope stronger one.

## Reasoning

### Claim drift across rounds

| Side | Round | What was actually argued | Drift from 00_claim.md? |
|------|-------|--------------------------|--------------------------|
| Red | Opening | The scorer is currently under-specified and risks being a costly reranker unless policies, transitions, outcome likelihoods, preferences, and ambiguity are explicit. | Mild narrowing toward present implementation and missing design details. Useful pressure, not fatal drift. |
| Blue | Opening | A no-grad finite candidate scorer is legitimate if it names the active-inference generative-model pieces, while train-time replacement remains premature. | No material drift. Blue adds conditions already demanded by the evidence context. |
| Red | Rebuttal | Finite candidates can be policies only after future observations and outcome likelihoods are declared. | Frame-correct attack. Red rejects sufficiency of naming components, but does not disprove the conditional design claim. |
| Blue | Rebuttal | The claim is not proven efficacy; it is a testable finite-policy EFE layer, subject to A/B/C/D/E and ablation constraints. | Slight narrowing, but it resolves the ambiguity rather than changing the target. |

### False dichotomies and equivocations detected

- "Policy" could mean an LM decoding rule or an active-inference policy under a generative model. The debate resolved this by requiring explicit predicted outcomes, transitions, likelihoods, preferences, and policy priors.
- "The code can host a scorer" could be confused with "the code already implements active inference." Blue conceded the present implementation boundary, so this did not force REMAND.
- Candidate continuations and agent sets are both finite policy menus, but they need separate outcome spaces and baselines. They can share the abstract scorer contract; they should not share empirical conclusions.
- "Empirically worth testing" is weak unless it names falsifiers. The final blue frame names them, so the claim becomes testable rather than purely promissory.

### Scope leakage detected

- Adjacent active-inference LLM papers motivate testing but do not establish V3 efficacy.
- Current V3 code paths support feasibility and blockers, not theoretical legitimacy by themselves.
- The gauge-transport ablation and seed noise floor are baseline controls, not evidence that EFE will help.
- A theoretical claim cannot be won by a future implementation plan alone; the win comes only because the external active-inference canon supplies finite policy scoring and because the proposed test has explicit refuters.

## Strength of each side

Red was strongest on vocabulary anchoring. It made the debate scientifically useful by forcing the scorer to declare candidate policies, rollout or transition model, outcome likelihood, preference distribution, ambiguity or epistemic term, horizon, candidate generator, and matched-compute baselines. Red also correctly blocked any inference from "V3 has sigma" to "V3 has epistemic value."

Blue was stronger on the actual claim. The claim does not assert that the current V3 checkout already contains an active-inference POMDP, nor does it claim EFE will improve perplexity. Blue defended the narrow proposition: a finite, explicit, no-grad EFE scorer is a legitimate active-inference policy-evaluation experiment, and a train-time replacement is premature. The canon-cop reports record no source-of-truth strikes for Blue's opening or rebuttal [02_canoncop_blue.md:23], [03_canoncop_blue.md:51].

## Action

Proceed with the narrow research action, but write the implementation spec before code: define A/B/C/D/E analogs, outcome space, preference distribution C, horizon, policy prior or precision, risk and ambiguity diagnostics, and fixed falsifiers. Treat candidate continuations and agent sets as two sub-experiments with separate metrics and baselines. Do not build or claim a train-time EFE replacement until a policy-conditioned generative-model layer exists and the no-grad scorer survives matched-compute, sigma, preference, and ambiguity ablations.

## Outcome (this judge)

BLUE_WINS
