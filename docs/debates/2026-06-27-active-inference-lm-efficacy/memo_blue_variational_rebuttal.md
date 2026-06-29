# Simulated Blue Expert Rebuttal Memo - variational

## Charge

Assess whether Red is right that the proposed inference-time scorer is merely a reranker, and whether a narrow no-grad expected-free-energy scorer over explicit candidate continuations can still be theoretically legitimate.

## Concession

Red is correct that an arbitrary scalar attached to candidate strings is not active inference. Canonical active inference requires a policy-conditioned generative model with predicted outcomes, hidden states, likelihoods, transitions, prior preferences, and policy priors. Smith, Friston, and Whyte give the implementation-facing template with A, B, C, D, and E objects; Friston et al. define policy beliefs through a softmax over negative expected free energy; Sajid et al. show that removing preferences changes the object into Bayesian experimental design rather than the full EFE objective [Smith et al. 2022; Friston et al. 2017; Sajid et al. 2021].

## Defense

The narrow claim survives Red's attack if the V3 scorer is framed as finite policy evaluation, not as a new training objective. A candidate continuation is a policy only after the experiment declares a candidate generator, a transition rule for rolling the current belief state through each candidate, an outcome model, a preference distribution, and an ambiguity or epistemic term. Under those declarations, scoring a finite candidate set by risk plus ambiguity is a valid special case of active-inference policy selection, even if the candidate set was produced by an ordinary language model. The external canon does not require that the policy set be continuous, differentiable, or train-through; discrete POMDP active inference already scores enumerated policies and then selects with a softmax over negative G [Friston et al. 2017; Smith et al. 2022; Heins et al. 2022].

The train-time replacement remains premature for a variational reason. Friston et al. separate state inference, policy selection, and learning timescales, while the V3 code path currently exposes logits and loss rather than a policy-conditioned outcome model. The evidence that `free_energy()` has an optional `log_likelihood` argument with no production caller under `vfe3` supports keeping the training objective unchanged until the policy layer has a clean generative-model interface [Friston et al. 2016; Neal 1998; `vfe3/free_energy.py:327-342`; `vfe3/free_energy.py:401-402`].

## Nontriviality Conditions

The scorer is nontrivial only if changing C changes rankings at fixed logits and candidates, if shuffling or replacing sigma damages the epistemic or ambiguity term, and if removing the ambiguity term changes behavior in the predicted direction. If those tests fail, Red wins: the scorer is a log-probability reranker with active-inference names.

## Newly Discovered Canon

- Friston et al. 2016 separates policy selection from learning in active-inference models, which supports an inference-time policy scorer before train-time objective replacement.
- Heins et al. 2022 implements active inference over finite POMDP policy sets, which supports finite candidate policy scoring when A/B/C/D/E analogs are explicit.
- Neal 1998 frames EM as coordinate ascent on a free-energy objective, which supports preserving V3's current training semantics while testing a separate policy layer.
