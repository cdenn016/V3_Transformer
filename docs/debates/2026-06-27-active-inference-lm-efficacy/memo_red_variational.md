# Simulated Expert Memo - Red Variational

Side: red. Round: opening. Memo path: memo_red_variational.md.

## Source-of-truth rule

The V3 report, wiki pages, manuscripts, and project docs are the claim under evaluation. Canonical active-inference and variational-inference sources set the standard.

## Newly-discovered canon

- Friston et al. (2017), "Active Inference: A Process Theory," Neural Computation, DOI 10.1162/NECO_a_00912: the discrete process theory uses a generative model with outcomes, hidden states, policies, and parameters; policy priors are a softmax over negative expected free energy.
- Smith, Friston, and Whyte (2022), "A step-by-step tutorial on active inference and its application to empirical data," Journal of Mathematical Psychology, DOI 10.1016/j.jmp.2021.102632: the operational POMDP template names likelihood A, transition B, preference C, initial-state prior D, and policy prior E.
- Sajid et al. (2021), "Active inference, Bayesian optimal design, and expected utility," arXiv:2110.04074: "When removing prior outcomes preferences from expected free energy, active inference reduces to optimal Bayesian design"; without ambiguity and relative risk, it reduces to expected utility.
- Heins et al. (2022), "pymdp: A Python library for active inference in discrete state spaces," arXiv:2201.03904: active inference is implemented through POMDP agents rather than an arbitrary score attached to candidates.

## Memo

The narrow recommendation is defensible only after the policy scorer is written as a generative-model calculation. In active inference, a policy is not just a candidate string; it is a counterfactual sequence evaluated through predicted future hidden states and outcomes under a likelihood and transition model, with prior preferences over outcomes. The canonical risk-plus-ambiguity form,

```text
G(pi) = KL[q(o|pi) || p(o|C)] + E_q H[p(o|s)],
```

requires q(o|pi), p(o|C), and p(o|s). Without those terms, "EFE over continuations" is a label placed on a reranking rule [Friston et al. 2017; Smith et al. 2022].

The strongest red objection is that the claim treats "theoretically legitimate" as satisfied by structural resemblance. A V3 belief tuple supplies mu, sigma, and phi, but it does not by itself define A, B, C, D, E, a policy posterior, or an epistemic value term. If C is next-token probability, future cross-entropy, repetition aversion, or task reward, those are different models with different semantics. If the ambiguity term is a sigma trace, that must be derived as the entropy of an outcome likelihood, not merely borrowed from the belief representation.

The train-time caution in the claim is sound, but it strengthens red. Canonical active inference separates state inference, policy selection, and parameter learning across variables and timescales [Friston et al. 2016; Smith et al. 2022]. A train-time EFE replacement is premature because the policy layer is not even specified. The same missing specification weakens the inference-time claim: no-grad scoring avoids gradient damage, but it does not make the scorer active inference.

Direct falsifier from this lens: if an EFE scorer ranks candidates identically, or near-identically, to a log-probability plus uncertainty reranker after preferences and transitions are made explicit, then the active-inference content is absent. A second falsifier is preference sensitivity: changing C should change policy rankings in the predicted direction; if rankings do not move, the claimed EFE semantics are not live.
