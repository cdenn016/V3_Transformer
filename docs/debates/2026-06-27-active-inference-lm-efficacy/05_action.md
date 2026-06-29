# Action - active-inference-lm-efficacy

**From verdict:** BLUE_WINS
**Reconciliation rule (panel=full):** Rule 3

## Recommended action

Accept the claim only in its narrow defended form. A default-off, no-grad expected-free-energy policy scorer over explicit candidate continuations or agent sets is theoretically legitimate and empirically worth testing in V3 if it declares A/B/C/D/E analogs, outcome space, preference distribution, horizon, policy prior or precision, risk and ambiguity diagnostics, fixed falsifiers, and matched-compute baselines.

Do not treat the verdict as evidence that V3 already implements such a scorer, that EFE improves perplexity or generation quality, or that replacing the training objective with EFE is justified. The first follow-up should be an implementation spec, not code: define the active-inference contract, candidate pool, rollout API, scoring terms, ablations, baselines, and pass/fail criteria.

## Follow-up debates

None required before writing the implementation spec. A useful later debate would evaluate a concrete scorer spec after A/B/C/D/E, preferences, outcome likelihood, horizon, and metrics are fixed.
