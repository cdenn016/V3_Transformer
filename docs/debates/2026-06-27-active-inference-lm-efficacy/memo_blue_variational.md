# Simulated Blue Expert Memo - variational

## Position

The narrow claim is defensible. A finite set of candidate continuations or candidate agents can be treated as a policy set if the implementation supplies predicted outcomes, prior preferences, and a transition or rollout model. Under that reading, scoring candidates by expected free energy is a standard active-inference move, not a new training principle. The claim would fail if the proposed score lacked a real outcome model or collapsed to the same ordering as next-token log probability.

## Analysis

Canonical active inference selects policies by expected free energy over future outcomes. Friston et al. define a policy prior of the form `P(pi) = softmax(-gamma * G(pi))`, where `G(pi)` decomposes into risk plus ambiguity or, equivalently, epistemic plus extrinsic value [Friston et al. 2017]. Smith, Friston, and Whyte give the implementation-facing discrete POMDP template: likelihood `A`, transitions `B`, preferences `C`, initial-state prior `D`, and policy prior `E`; policy posteriors use softmax scores containing expected free energy [Smith et al. 2022]. Friston et al. 2016 also keep state inference, policy selection, and parameter learning on different timescales, which supports the claim's split between inference-time scoring and caution about training-time replacement [Friston et al. 2016].

For V3-style language modeling, the clean defense is not that a continuation is literally a biological motor plan. It is that an explicit candidate continuation is a finite policy hypothesis inside a predictive generative model. If V3 can roll a belief state forward under candidate continuations, decode predicted outcome distributions, and compare those outcomes to an explicit preference distribution, then a score of the form risk plus ambiguity is a legitimate active-inference policy score. The epistemic term is also meaningful only if V3's covariance or posterior-update proxy tracks uncertainty reduction rather than acting as decoration.

The training-loss replacement is premature because expected free energy is not the same object as present-observation variational free energy or next-token cross entropy. A training objective would need a differentiable policy distribution over future trajectories, an explicit preference model, an outcome likelihood, and a learning rule that does not erase the E-step/M-step separation. The present evidence says V3's scalar free-energy assembly has an optional observation likelihood term but no production caller, and `forward()` returns logits rather than a reusable final belief object. That is enough for a decoding experiment after a rollout helper exists, but not enough for a principled train-through-policy objective.

## Newly-discovered canon

- [Friston et al. 2017] Active-inference process theory: policy prior `P(pi) = softmax(-gamma G(pi))`; expected free energy decomposes into risk plus ambiguity and epistemic plus extrinsic value.
- [Smith et al. 2022] Operational POMDP template with `A`, `B`, `C`, `D`, and `E`; expected free energy supplies policy selection, not a generic logit decoration.
- [Friston et al. 2016] State inference, policy selection, and parameter learning operate on distinct timescales, supporting inference-time experimentation before training-loss replacement.
- [Buckley et al. 2017] Boundary source: predictive-coding and perception-side VFE do not by themselves supply EFE policy selection.

## Recommendation

Defend the inference-time candidate scorer as a finite-policy active-inference layer. Do not defend train-time EFE replacement on current evidence.
