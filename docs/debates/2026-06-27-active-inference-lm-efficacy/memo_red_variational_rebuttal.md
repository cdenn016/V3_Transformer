# Red Rebuttal Memo - variational

## Newly-discovered canon

- Guo, Pleiss, Sun, and Weinberger (2017), "On Calibration of Modern Neural Networks," arXiv:1706.04599, https://arxiv.org/abs/1706.04599. Use: predictive confidence and uncertainty surrogates require calibration checks; a model-internal variance cannot be treated as a reliable correctness probability by name alone.
- Kuleshov, Fenner, and Ermon (2018), "Accurate Uncertainties for Deep Learning Using Calibrated Regression," arXiv:1807.00263, https://arxiv.org/abs/1807.00263. Use: approximate Bayesian uncertainty estimates can be inaccurate under misspecification and approximate inference, so calibrated intervals require empirical correction or validation.
- Rainforth, Foster, Ivanova, and Bickford Smith (2023), "Modern Bayesian Experimental Design," arXiv:2302.14545, https://arxiv.org/abs/2302.14545. Use: expected-information-gain objectives are defined over possible experimental outcomes and can be computationally hard; they are not supplied by merely listing choices.

## Expert memo

Blue's strongest point is correct: a finite candidate set can serve as a policy set in active inference if each candidate induces a predictive distribution over future outcomes and if risk and ambiguity are computed from an explicit generative model [Friston et al. 2017; Smith et al. 2022]. That condition defeats a blanket red claim that finite candidate scoring is never active inference.

The rebuttal is that Blue's defense shifts from "if V3 defines the rollout, outcome distribution, preference distribution, and ambiguity proxy" to "the scorer is theoretically legitimate once those components are named." Canonical active inference does not use labels as sufficient structure. In the POMDP template, `A` is the outcome likelihood, `B` is the policy-conditioned transition model, `C` is the preference distribution over outcomes, `D` is the initial-state prior, and `E` is the policy prior [Smith et al. 2022]. Expected free energy is an expectation over future outcomes under those pieces, not a retrospective score on a fixed string. The risk term is `KL[q(o | pi) || p(o | C)]`, and the ambiguity term is the expected entropy of `p(o | s)`, not an arbitrary entropy of hidden beliefs [Friston et al. 2017; Smith et al. 2022].

The finite-continuation version therefore needs to specify what counts as an outcome after the continuation. If the outcome is the continuation text itself, then the "future observation" has already been fixed by the candidate and the epistemic term loses its active-inference meaning. If the outcome is future cross-entropy, task success, user reward, route success, or community utility, then V3 needs an outcome likelihood connecting hidden belief states to that outcome. The current evidence says this likelihood is not live: `free_energy.py:341` defines `log_likelihood` as optional, `free_energy.py:401-402` subtracts it only when provided, and the code search found no production caller passing `log_likelihood=`.

My recommendation for synthesis is to concede finite policies in principle, then deny sufficiency. Red should say that V3 may implement a useful reranker, but the active-inference claim becomes legitimate only after it supplies causal future observations or a modeled outcome distribution, a live likelihood, and an ambiguity term tied to `H[p(o | s)]`.
