# Simulated Blue Expert Rebuttal Memo - numerical-analyst

## Charge

Specify the experiment that makes the no-grad EFE scorer worth testing and protects against false positives.

## Concession

Red is correct that a candidate scorer can absorb gains from compute, candidate count, horizon, or decoding choice. EFE planning can be computationally heavy, and Malekzadeh and Plataniotis flag the computational burden of expected-free-energy planning as a real limitation [Malekzadeh and Plataniotis 2022]. In V3, generation already calls a full `forward(context)` per new token at `vfe3/model/model.py:1201`; scoring M candidates over horizon H multiplies that unless a cached rollout helper exists.

## Defense

The experiment is still worth running because it can be made finite, controlled, and causal. The score should be evaluated on a predeclared candidate set under no-grad. The matched baselines should receive the same candidate pool and the same forward-call budget: length-normalized log probability, beam or diverse beam, top-p/top-k candidate reranking, a shallow sigma/entropy reranker, and an ablated EFE score with risk only or ambiguity only. Report bootstrap intervals, seed sensitivity, and the known V3 noise floor from the 2026-06-27 ablation suite, where the three-seed coefficient of variation was about 0.6-1.1 percent. A single-seed gain below that scale should be treated as noise.

Nontriviality requires causal ablations, not just metric movement. Preference sensitivity: change C while holding candidates and logits fixed; rankings should move in the predicted direction. Ambiguity sensitivity: remove or shuffle sigma; if the claimed epistemic term matters, rankings and downstream metrics should degrade. Epistemic validation: the epistemic term should correlate with realized uncertainty reduction or calibration improvement on held-out continuation outcomes. Compute fairness: if the EFE scorer uses extra rollouts, a baseline with the same rollouts and a learned or hand-coded reranker should be allowed.

This design does not prove efficacy in advance. It makes the test capable of refuting the claim and of separating active-inference structure from ordinary reranking [Popper 1959; Lakatos 1978; Holtzman et al. 2019].

## Newly Discovered Canon

- Popper 1959 supplies the falsification standard for the empirical half of the claim.
- Lakatos 1978 warns against endless auxiliary repairs, so the first EFE scorer must have predeclared failure criteria.
- The V3 ablation run supplies a concrete noise floor and already-measured gauge-transport effect that any new scorer must not re-credit to itself.
