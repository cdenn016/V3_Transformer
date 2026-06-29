# Simulated Blue Expert Memo - philosophy-of-science

## Position

The blue defense is acceptable only because the claim is modest. It says "theoretically legitimate and empirically worth testing," not "already proven effective." The defense becomes circular if it cites the V3 report or wiki as proof that the EFE mapping is correct.

## Analysis

The external active-inference canon gives enough structure to make the narrow claim meaningful: policies, predicted outcomes, preferences, and expected free energy are part of the standard process theory [Friston et al. 2017; Smith et al. 2022; Parr, Pezzulo, and Friston 2022]. The V3 project evidence can show that the code has Gaussian belief states and a generation hook, but it cannot certify that an LLM continuation is an active-inference policy in the canonical sense. That mapping is a proposed interpretation and must earn its keep by explicit generative-model commitments.

Popper's standard is useful here: an empirical claim must forbid something [Popper 1959]. "EFE is a nice vocabulary for generation" forbids nothing. "EFE improves candidate selection when its risk and ambiguity terms are computed from calibrated V3 belief rollouts, under matched compute, and fails under sigma shuffling" does forbid observations. The latter is a scientific claim; the former is branding.

Lakatos also matters. A new EFE scorer can be a progressive protective-belt addition if it predicts a new pattern before the fact, such as improved uncertainty-aware routing or calibration after sigma-sensitive rollout. It becomes degenerating if every failed test is rescued by changing the preference distribution, rollout horizon, policy temperature, or endpoint after seeing the result [Lakatos 1978]. Cartwright's warning about domain-bound regularities also applies: a scorer that helps agent routing or selective prediction need not help perplexity or open-ended storytelling [Cartwright 1999].

The blue side should state that train-time EFE replacement is premature. That concession is not weakness; it keeps the claim inside the domain where the canon and the code overlap. The current code exposes logits from `forward()`, not a full active-inference generative model with actions and outcomes, and the active-inference canon itself requires those components.

## Newly-discovered canon

- [Popper 1959] A scientific claim must have potential falsifiers.
- [Lakatos 1978] A research-program extension is progressive only if it yields novel corroborated content rather than post-hoc patching.
- [Cartwright 1999] Model regularities are domain-bound; success in one decision domain does not export automatically to all LM behavior.
- [Parr, Pezzulo, and Friston 2022] Active inference is a generative-model-based framework for perception, learning, planning, and action; the book does not validate the V3-specific mapping.

## Recommendation

Defend the narrow opt-in scorer and list falsifiers. Treat the Research wiki as context and source map, not as evidence that the proposed extension is correct.
