# Extended Evidence - active-inference-lm-efficacy

## Phase 2 Red Harvested Canon

The red panel did not have access to nested agent dispatch, so these are deduplicated excerpts from the simulated expert memos and the coordinator's external canon search.

### Active-inference policy machinery

- Friston, FitzGerald, Rigoli, Schwartenbeck, and Pezzulo (2017), "Active Inference: A Process Theory," Neural Computation, DOI 10.1162/NECO_a_00912. Canonical use: discrete active-inference process theory scores policies by expected free energy and updates policy beliefs with a softmax over negative free-energy terms.
- Smith, Friston, and Whyte (2022), "A step-by-step tutorial on active inference and its application to empirical data," Journal of Mathematical Psychology, DOI 10.1016/j.jmp.2021.102632. Canonical use: the implementation template names A as likelihood, B as transitions, C as prior preferences, D as initial-state priors, and E as policy priors.
- Heins et al. (2022), "pymdp: A Python library for active inference in discrete state spaces," arXiv:2201.03904, https://arxiv.org/abs/2201.03904. Excerpt: the package is for "simulating active inference with partially-observable Markov Decision Processes or POMDPs."
- Sajid, Da Costa, Parr, and Friston (2021), "Active inference, Bayesian optimal design, and expected utility," arXiv:2110.04074, https://arxiv.org/abs/2110.04074. Excerpt: "When removing prior outcomes preferences from expected free energy, active inference reduces to optimal Bayesian design." The abstract also states that in another limiting case it reduces to Bayesian decision theory.

### VFE/FEP boundary

- Buckley, Kim, McGregor, and Seth (2017), "The free energy principle for action and perception: A mathematical review," arXiv:1705.09156, https://arxiv.org/abs/1705.09156. Excerpt: the paper aims to "disclose the assumption structure" of a widely used FEP implementation. Red use: the V3 proposal must disclose its assumptions rather than rely on broad FEP language.

### LM decoding and empirical baselines

- Holtzman, Buys, Du, Forbes, and Choi (2019), "The Curious Case of Neural Text Degeneration," arXiv:1904.09751, https://arxiv.org/abs/1904.09751. Excerpt: decoding choices can strongly change machine text quality while holding the neural LM fixed. Red use: an EFE scorer must beat strong decoding and reranking baselines under matched candidates and compute.
- Finlayson, Hewitt, Koller, Swayamdipta, and Sabharwal (2023), "Closing the Curious Case of Neural Text Degeneration," arXiv:2310.01693, https://arxiv.org/abs/2310.01693. Canonical use: truncation sampling is a serious baseline family, not a straw target.
- Malekzadeh and Plataniotis (2022), "Active Inference and Reinforcement Learning," arXiv:2212.07946, https://arxiv.org/abs/2212.07946. Excerpt: active inference is limited in part by "the computational challenges associated with EFE."

### Philosophy of science

- Popper (1959), The Logic of Scientific Discovery. Canonical use: an empirical claim must identify potential falsifiers.
- Lakatos (1978), The Methodology of Scientific Research Programmes. Canonical use: a research program can protect itself with auxiliary hypotheses; a fair test must state which repairs count as failures of the current claim.

### V3 code evidence added by red

- `vfe3/belief.py:22-28`: `BeliefState` carries `mu`, `sigma`, `phi`, and optional `s` and `r`.
- `vfe3/model/model.py:660-661`: `forward()` encodes token ids to beliefs and applies positional gauge frames.
- `vfe3/model/model.py:725-734`: `forward()` calls `vfe_stack(...)` for the E-step.
- `vfe3/model/model.py:790-794`: inference decodes final beliefs to logits and returns logits when `targets is None`.
- `vfe3/model/model.py:1198-1221`: generation loops over full `forward(context)`, then selects tokens by greedy, temperature, top-k, top-p, or sampling.
- `vfe3/free_energy.py:327-342` and `vfe3/free_energy.py:401-402`: `log_likelihood` exists only as an optional argument and is subtracted only when non-None; `rg "log_likelihood\\s*=" vfe3` returned no production caller.
- `vfe3/model/model.py:1045-1049`: model-channel gamma energy builds transport with `transport_mode="flat"`.

## Phase 3 Red Harvested Canon

Nested Agent dispatch was unavailable, so these entries are deduplicated from the five simulated red rebuttal memos and the coordinator's live canon search.

### Calibration and uncertainty validation

- Guo, Pleiss, Sun, and Weinberger (2017), "On Calibration of Modern Neural Networks," arXiv:1706.04599, https://arxiv.org/abs/1706.04599. Canonical use: confidence-like outputs from neural predictors can be miscalibrated; calibration must be measured rather than inferred from model form.
- Kuleshov, Fenner, and Ermon (2018), "Accurate Uncertainties for Deep Learning Using Calibrated Regression," arXiv:1807.00263, https://arxiv.org/abs/1807.00263. Canonical use: approximate Bayesian uncertainty estimates can be inaccurate under misspecification and approximate inference; calibrated uncertainty needs empirical validation.

### Bayesian design and active-inference interpretation

- Rainforth, Foster, Ivanova, and Bickford Smith (2023), "Modern Bayesian Experimental Design," arXiv:2302.14545, https://arxiv.org/abs/2302.14545. Canonical use: expected-information-gain objectives are defined over possible experimental outcomes and can be computationally hard; listing choices does not supply an outcome model.
- Sajid, Da Costa, Parr, and Friston (2021), "Active inference, Bayesian optimal design, and expected utility," arXiv:2110.04074, https://arxiv.org/abs/2110.04074. Canonical use: EFE bridges Bayesian decision theory and Bayesian optimal design in limiting cases, so the epistemic term has to correspond to expected information gain about hidden states or parameters.

### Language-generation baselines

- Meister, Pimentel, Wiher, and Cotterell (2022), "Locally Typical Sampling," arXiv:2202.00666, https://arxiv.org/abs/2202.00666. Canonical use: information-theoretic decoding methods can improve generation behavior without constituting active inference.
- Meister, Wiher, Pimentel, and Cotterell (2022), "On the probability-quality paradox in language generation," arXiv:2203.17217, https://arxiv.org/abs/2203.17217. Canonical use: high probability and human-perceived quality can diverge, so an EFE reranker must beat decoding controls designed for that divergence.
- Meister, Pimentel, Malagutti, Wilcox, and Cotterell (2023), "On the Efficacy of Sampling Adapters," arXiv:2307.03749, https://arxiv.org/abs/2307.03749. Canonical use: sampling adapters trade precision and recall in generated text; EFE needs matched baselines from this family.

### Code evidence rechecked by red in Phase 3

- `vfe3/belief.py:22-30`: `BeliefState` carries `mu`, `sigma`, `phi`, and optional `s` and `r`; this shows available belief fields, not calibrated decision value.
- `vfe3/model/model.py:660-661`: `forward()` encodes token ids to beliefs and applies positional gauge frames.
- `vfe3/model/model.py:725-734`: `forward()` calls `vfe_stack(...)` for the E-step.
- `vfe3/model/model.py:791-794`: inference decodes `mu_final` and `sigma_final` to logits and returns logits directly when `targets is None`; the public path does not return final beliefs.
- `vfe3/model/model.py:1164-1221`: `generate()` is no-grad, calls `forward(context)`, takes last-token logits, and selects by greedy, temperature, top-k, top-p, or sampling.
- `vfe3/free_energy.py:341` and `vfe3/free_energy.py:401-402`: `log_likelihood` is optional and subtracted only when non-None; no production caller passing `log_likelihood=` was found in `vfe3`.
- `vfe3/model/model.py:1045-1049`: the model-channel gamma energy uses flat belief transport for `s_mu` and `s_sigma`.

## Phase 3 Blue Harvested Canon

The blue rebuttal panel did not have access to nested agent dispatch, so these are deduplicated excerpts from the simulated expert memos and the coordinator's code/wiki pass.

### Active-inference policy legitimacy

- Friston et al. (2016), "Active Inference and Learning," supports a separation between state inference, policy selection, and learning timescales. Blue use: a no-grad inference-time policy scorer can be tested before any train-time EFE replacement.
- Friston et al. (2017), "Active Inference: A Process Theory," establishes policy selection by softmax over negative expected free energy. Blue use: a finite candidate set can be treated as a policy set when outcomes, transitions, preferences, and ambiguity terms are explicit.
- Smith, Friston, and Whyte (2022), "A step-by-step tutorial on active inference and its application to empirical data," supplies the A/B/C/D/E POMDP implementation checklist and the risk-plus-ambiguity EFE form. Blue use: this checklist defines what the V3 scorer must instantiate to avoid becoming a loose reranker.
- Heins et al. (2022), "pymdp: A Python library for active inference in discrete state spaces," implements active inference in finite POMDP policy spaces. Blue use: enumerated candidate continuations are not disqualified merely because they are finite, provided the generative-model objects are declared.

### Boundaries and baselines

- Sajid, Da Costa, Parr, and Friston (2021), "Active inference, Bayesian optimal design, and expected utility," clarifies limiting cases of expected free energy. Blue use: removing prior preferences or epistemic terms changes the semantics, so the V3 scorer must include preference sensitivity and epistemic or ambiguity ablations.
- Holtzman et al. (2019), "The Curious Case of Neural Text Degeneration," and Finlayson et al. (2023), "Closing the Curious Case of Neural Text Degeneration," make decoding strategy a serious baseline. Blue use: the EFE scorer must beat matched-candidate and matched-compute decoding/reranking controls.
- Popper (1959) and Lakatos (1978) set the philosophy-of-science constraint: the first V3 EFE scorer must predeclare falsifiers and cannot preserve the claim by changing C, horizon, candidate generator, or ambiguity proxy after failure.

### V3 code evidence checked by blue

- `vfe3/belief.py:22-28`: `BeliefState` exposes `mu`, `sigma`, and `phi`, the raw material for candidate belief rollouts.
- `vfe3/model/model.py:792-794`: inference decodes final beliefs to logits and returns logits, not a public final-belief policy object.
- `vfe3/model/model.py:1201-1221`: `generate()` calls `forward(context)` and then performs token selection by greedy or sampling rules, giving a natural no-grad policy-scoring hook.
- `vfe3/model/prior_bank.py:223` and `vfe3/model/prior_bank.py:312`: token-to-belief and belief-to-logit interfaces exist, which can support an explicit candidate-rollout helper.
- `vfe3/model/model.py:1045-1062`: model-channel gamma scoring uses flat transport, so agent-set EFE claims must be limited to that semantics unless a separate non-flat covariant path is built and tested.
