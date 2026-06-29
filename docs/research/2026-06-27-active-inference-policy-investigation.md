# Active Inference Policy Investigation for V3

Date: 2026-06-27. Scope: investigate whether a theoretically principled active-inference feature, based on expected free energy over future token or agent policies, could, should, and would improve the V3 Transformer relative to the current codebase. This report used three parallel sub-agents: a variational-theory reviewer, a codebase-integration auditor, and an experiment analyst. It also consulted the Research wiki pages `[[Active Inference]]`, `[[Expected Free Energy]]`, `[[Free-energy principle active inference]]`, `[[Collective active inference]]`, `[[Multi-agent variational free energy]]`, `[[Meta-agents and hierarchical emergence]]`, `[[VFE Transformer Program]]`, and the run note `[[2026-06-27-gauge-transport-ablation-suite]]`.

## Executive Judgment

An active-inference feature can be made theoretically principled in V3, but only if it is implemented as policy inference over explicit counterfactual futures, with a defined action space, transition model, likelihood, and prior preferences. It should not be introduced first as a new training loss. The lowest-risk and most honest first implementation is an opt-in, no-grad, registry-selected generation or candidate-selection policy scorer that reranks a pruned set of policies by expected free energy, then compares against compute-matched log-probability, beam, random-score, and shuffled-uncertainty controls.

The answer to "could" is yes. V3 already has the necessary primitives: Gaussian belief states, iterative E-step inference, gauge transport, attention softmaxes arising from free-energy envelopes, uncertainty carried by `sigma`, model-channel `s` tables, and a generation loop where policy selection could attach. The answer to "should" is yes, but staged: fix or quarantine the known covariance-dependent oracle issue first for non-flat/covariant rollouts, expose a clean belief-rollout API, and begin with `policy_mode="none"` as the default pure path plus an opt-in `efe_rollout` mode. The answer to "would produce better results" is not established. The most plausible wins are calibration, uncertainty-aware continuation choice, long-horizon coherence, and candidate-community selection. A validation perplexity improvement is possible only if the EFE scorer changes the evaluated predictive distribution, and it must clear the current seed-noise floor.

## Theoretical Standard

In active inference, policy selection is not ordinary future likelihood maximization. A policy is a sequence, or structured batch, of actions. It is selected by minimizing expected free energy over future outcomes generated under that policy. In the discrete active-inference canon of Friston et al. 2016/2017, Parr and Friston 2019, and Smith, Friston, and Whyte 2022, the policy score has the form

```text
G_tau(pi) =
E_{q(o_tau,s_tau | pi)}
[ log q(s_tau | pi) - log p(o_tau,s_tau | C) ].
```

One operational decomposition is risk plus ambiguity:

```text
G_tau(pi) =
KL[q(o_tau | pi) || p(o_tau | C)]
+ E_{q(s_tau | pi)} H[p(o_tau | s_tau)].
```

This is the implementation-facing form. The first term penalizes predicted outcomes that disagree with prior preferences. The second penalizes policies that lead to ambiguous observations even when the hidden state is known.

The same policy score is often read as pragmatic value plus epistemic value:

```text
G_tau(pi) =
- E_{q(o_tau | pi)} log p(o_tau | C)
- I_q(s_tau ; o_tau | pi),
```

up to the standard assumptions used in the active-inference derivation. This is the interpretation-facing form. Minimizing `G` favors preferred outcomes while also favoring expected information gain. A V3 feature that only scores the most likely future tokens would be a lookahead decoder, not active inference in the strong sense. The required extra ingredient is a preference distribution `p(o | C)` and an epistemic or ambiguity term computed from the model's predicted hidden beliefs and observations.

The user's community example is a natural fit, but it needs one correction. "Add N applicants" is not automatically a horizon. It is usually a batch action:

```text
A subset C_candidates, |A| = N,
```

or an ordered admission policy:

```text
pi = (a_1, ..., a_N), a_k in C_candidates.
```

The planning horizon is the future period over which the resulting community is evaluated. A principled community EFE score would be

```text
G_H(A) =
sum_{tau=1}^H rho_tau [
  KL(q(o_tau | A) || p_C(o_tau))
  + E_{q(s_tau | A)} H[p(o_tau | s_tau)]
],
```

where hidden state `s_tau` includes community belief state, role coverage, compatibility, conflict risk, model uncertainty, and possibly latent applicant parameters. If learning about applicants is itself valuable, add parameter information gain by including `theta` in the hidden state or by adding a term equivalent to `-I_q(theta ; o_tau | A)`.

For tokens, the analogous policy is a candidate continuation:

```text
pi = (a_t, ..., a_{t+H-1}), a_tau in vocab.
```

This is principled only in generation or closed-loop interaction, where emitted tokens affect later observations such as user response, tool result, task success, or the model's own later belief state. In teacher-forced next-token prediction on a fixed corpus, the model does not act on the world. In that setting EFE can still be evaluated as a reranking or calibration objective, but the "active" part is weak unless the policy changes future observations.

## V3 Integration Map

The live V3 primitives are already close to what an opt-in active-inference scorer needs. `BeliefState` carries the Gaussian tuple `mu`, `sigma`, and `phi`, with optional future channels `s` and `r` in `vfe3/belief.py:22-30`. `PriorBank.encode` maps token ids into initial Gaussian beliefs in `vfe3/model/prior_bank.py:223-228`, while `encode_s` exposes a separate model-channel belief in `vfe3/model/prior_bank.py:230-247`. The E-step path is `VFEModel.forward -> vfe_stack -> vfe_block -> e_step`: the model encodes beliefs in `vfe3/model/model.py:660-661`, builds priors and transports in `vfe3/model/model.py:662-734`, and returns either chunked CE or dense logits in `vfe3/model/model.py:764-805`. The stack iterates blocks in `vfe3/model/stack.py:68-79`, and each block dispatches to `e_step` in `vfe3/model/block.py:63-80`.

The free-energy machinery is usable as a policy scorer. `attention_weights` computes the softmax over negative energy in `vfe3/free_energy.py:268-279`; `reduced_free_energy` gives the envelope `-tau log Z` in `vfe3/free_energy.py:307-324`; and `free_energy` assembles self-coupling, belief coupling, attention entropy, and the optional observation term in `vfe3/free_energy.py:327-403`. These functions are the natural place to define the risk and ambiguity components, or to reuse the existing envelope score as a future-belief stability term.

The natural attachment point is generation, not training. `VFEModel.generate` currently reruns `forward(context)` for each generated token and selects by greedy, temperature, top-k, or top-p sampling in `vfe3/model/model.py:1165-1222`. A policy layer should attach at that selection point as a registered `policy_mode`, with default `none` preserving current behavior. A candidate implementation would take the top-k next-token set, construct candidate continuations for horizon `H`, run them through `forward` or a new belief-rollout helper under `torch.no_grad`, score each continuation by `G(pi)`, and select by argmin or by `softmax(-gamma_policy * G)`.

This route respects the no-neural-network constraint. It needs no value network, no MLP, and no learned controller. It is just tensor scoring over beliefs and priors already present in the model. If preferences are token-level, they can be an explicit log-prior over outcomes or a registry-selected functional of candidate continuation diagnostics. If preferences are community-level, they can be a hand-defined or data-derived distribution over future community observables, again registered as a preference model rather than inserted as a neural head.

## Required API Shape

The current `forward` API returns logits/loss, not the final belief tuple. A serious policy scorer needs a small internal helper, for example `infer_beliefs(token_ids, *, return_logits=False)`, that returns the converged `BeliefState` and optionally logits. This should not duplicate the forward path. It should factor the existing encode, prior, E-step, final norm, and decode sequence so generation, diagnostics, and EFE rollouts all read the same belief trajectory.

The policy scorer should be registered separately from decode. A minimal registry might expose:

```text
policy_mode = "none" | "logprob" | "efe_one_step" | "efe_rollout"
policy_horizon = H
policy_top_k = Kp
policy_precision = gamma_policy
policy_preference = registry key
policy_score_terms = risk, ambiguity, epistemic, current_logprob
```

The default `policy_mode="none"` should leave `generate` unchanged. `efe_one_step` should rerank top-k next tokens with a one-step score. `efe_rollout` should build short continuations, initially with horizon 2 or 4, because full recompute generation is expensive. Longer horizons need caching before they are computationally honest.

## Current Blockers

The first theoretical blocker is that V3's observation likelihood term is not live in the E-step. `free_energy(..., log_likelihood=...)` is explicitly marked as a gated stub in `vfe3/free_energy.py:341` and explained in `vfe3/free_energy.py:358-368`; wiring it into this one function would not affect the analytic gradient kernels or oracle. A real active-inference policy objective can avoid this at first by using an inference-time no-grad score. A train-time EFE loss would need a proper observation/preference path through both kernels and oracle.

The second theoretical blocker is preferences. Active inference replaces reward with prior preferences over outcomes. V3's current next-token training target is cross-entropy on fixed data. That supplies a likelihood objective, not a policy preference model. The clean first preference choices are explicit and measurable: lower predicted future CE, lower ambiguity from `sigma`, higher expected information gain, lower repetition, better calibration, or a domain-specific community utility distribution. Without this, "EFE" becomes a decorated beam-search score.

The first code blocker is the known covariant oracle gap. The 2026-06-27 multi-agent audit found that `_omega_builder(mu_q, mu_k)` in `vfe3/inference/e_step.py:426-435` closes over `belief.sigma` instead of passing live sigma leaves, while `regime_ii_covariant` consumes covariance features in `vfe3/geometry/transport.py:472-483`. Any policy rollout that depends on covariance-sensitive non-flat transport gradients should wait for that fix or stay no-grad and flat. Since the recommended first policy scorer is no-grad, this is not fatal, but it matters before train-time or covariant policy-gradient work.

The second code blocker is model-channel geometry. `_gamma_energy` builds the model-channel transport with `transport_mode="flat"` in `vfe3/model/model.py:1021-1062`, even when the belief channel uses configured transport. That may be intentional, but an active-inference community policy over model beliefs should initially either accept flat model-channel semantics or wait for an explicit design decision on whether `s/gamma` should share configured transport.

The third code blocker is cost. Generation currently recomputes the whole model for every token in `vfe3/model/model.py:1198-1221`. A top-k, horizon-H policy scorer multiplies that cost by roughly `Kp * H` before pruning. The first ablation should therefore use tiny horizons and log wall time. If the result only improves because it spends far more compute, it is not an architectural win.

## Empirical Evidence and Where EFE Might Help

The strongest measured V3 effect is still learned gauge transport. The 2026-06-27 run note reports learned transport around 154 PPL, frozen random transport around 279 PPL, and identity transport around 268 PPL at the ablation operating point. This is a large causal effect and should remain the main training-core story. An EFE layer should be measured against that background, not sold as the missing central mechanism.

The most plausible near-term EFE target is calibration and uncertainty-aware candidate choice. Existing artifacts already record ECE, sigma-to-CE diagnostics, and sigma trace statistics. The experiment sub-agent observed that a K=60 scaling run has `sigma_ce_spearman=0.176` and `sigma_trace_cv=0.110`, which means covariance carries a modest but real uncertainty signal. A successful EFE policy should improve ECE, sigma-stratified CE, or uncertainty-sensitive sample selection, not merely shift top-1 accuracy by an unmeasured amount.

The second target is long-horizon or out-of-distribution generation. The 2026-06-27 position-extrapolation results show learned absolute position is best at train length but degrades strongly by length 512, while ALiBi, T5, and RoPE remain much flatter. EFE cannot fix a positional mechanism by itself, but a rollout scorer can prefer continuations that keep future beliefs less ambiguous and less unstable. This is the right kind of metric for an active policy layer: not just "next token more likely," but "future belief trajectory remains coherent under the model."

The third target is policy precision and attention entropy. The canonical attention-entropy ablation shows a real gap at low kappa and little gap at kappa 1.0. That is the closest existing V3 evidence to the active-inference softmax over negative expected free energy. It suggests the current code already has the mathematical substrate for policy precision, but the action side has not yet been built.

## Recommended Ablation Sequence

Phase 0 should fix or quarantine known blockers. For a flat, no-grad inference-time scorer, the high oracle issue can be noted but does not block the first test. For any non-flat covariant or train-through-policy scorer, fix the sigma-leaf oracle rebuild first. Also factor out a belief-rollout helper so generation and diagnostics do not fork the inference path.

Phase 1 should be a post-hoc one-step EFE reranker on a frozen checkpoint. For each context, take the top-k next-token candidates under the current logits, append each candidate, run one no-grad belief inference pass, and score

```text
score(a) =
risk_weight * risk(a)
+ ambiguity_weight * ambiguity(a)
- epistemic_weight * information_gain_proxy(a)
- logprob_weight * log p(a | context).
```

The first ambiguity proxy can be a function of future `sigma` trace or entropy. The first risk term can be negative preference log-probability, teacher-forced CE when evaluating held-out continuations, or an explicit domain preference for generation tasks. The first epistemic proxy should be predicted reduction in belief uncertainty across the rollout, not a diversity bonus.

Phase 2 should run decomposition controls: risk-only, ambiguity-only, epistemic-only, logprob-only, full EFE, shuffled sigma, random score, and sign-flipped epistemic term. The full score must beat the best component and a temperature-tuned logprob baseline to count as more than heuristic reranking.

Phase 3 should test short horizons, initially `H=2` and `H=4`, with the same top-k and compute budgets. Evaluate validation CE if the policy changes logits, ECE/reliability, sigma-stratified CE, repetition and distinct-n sample metrics, long-context CE/PPL at the existing N=192..512 extrapolation points, and wall-clock. Require at least three seeds or a matched checkpoint set before claiming a small effect.

Phase 4 should repeat the promising arms on the pure KL prior-bank path with `use_prior_bank=True`. The current ablation operating point uses `use_prior_bank=False`, the learned linear readout path. A theoretically pure EFE claim should eventually use the KL-to-prior readout because prior preferences belong naturally in that generative space.

Only after those phases should V3 try a train-time auxiliary EFE regularizer. The train-time version needs a live preference/observation term through gradients, not only a no-grad generation score. It should have a lambda sweep, compute-matched controls, and at least three seeds.

## Falsification Criteria

The active-inference feature should be demoted to a costly reranker if matched-compute experiments fail to improve either a primary metric or two secondary metrics. Primary metrics are held-out CE/PPL under a policy-adjusted predictive distribution and task success in closed-loop generation. Secondary metrics are ECE, sigma-stratified CE, `sigma_ce_spearman`, attention entropy behavior, long-context CE/PPL, repetition, distinct-n, and sample preference scores.

The idea is also falsified if predicted epistemic value does not correlate with realized posterior uncertainty reduction; if gains disappear under compute-matched beam or best-of-N baselines; if shuffled sigma performs as well as true sigma; if a sign-flipped epistemic term performs equally well; if the policy does not causally change future observations in a closed-loop setting; or if the numerical EFE score reduces to current free energy plus a diversity heuristic.

For the community setting, the strongest falsifier is semantic: if there is no explicit future outcome model for what a better community means, then admission policies cannot be scored by EFE. The model needs observable outcomes such as role coverage, future disagreement, task throughput, calibration, conflict rate, or collective free-energy reduction. Without that, "which N applicants are best" is not active inference; it is an ungrounded compatibility score.

## Answer to the User's Proposal

The tentative idea that a policy is the best expected free energy over the next predicted N tokens or agents is close, but it needs two refinements. For tokens, `N` can be the planning horizon only when generated tokens are actions that affect later observations. For agents, `N` is usually the size of the batch action, while the horizon is the future interval over which the admitted group is evaluated. The community analogy is therefore more principled than plain next-token prediction, because adding applicants really is an action that changes future observations of the system.

A V3 implementation that is wholly theoretically principled should define actions as token continuations or applicant sets, define hidden states as Gaussian belief tuples plus any community latent variables, define outcomes as future tokens or community observables, define preferences explicitly, and select policies by `softmax(-gamma_policy * G)`. This is a genuine extension of the current VFE transformer rather than a rebranding of beam search.

## References Used

Friston et al. 2016, "Active Inference and Learning"; Friston et al. 2017, "Active Inference: A Process Theory"; Parr and Friston 2019, "Generalised free energy and active inference"; Smith, Friston, and Whyte 2022, "A step-by-step tutorial on active inference"; Buckley et al. 2017, "The free energy principle for action and perception"; Friston et al. 2024, "Federated inference and belief sharing"; Heins et al. 2024, "Collective behavior from surprise minimization"; Albarracin et al. 2022, "Epistemic Communities under Active Inference"; Waade et al. 2025, "As One and Many"; Research wiki pages named in the scope section; V3 source files and audit artifacts at commit `5e88afc`.

Recent arXiv adjacency was also checked on 2026-06-27. The closest papers were ODAR, which uses active inference for adaptive LLM reasoning-route selection, and "Free Energy Heuristics," which uses EFE under uncertain precision to explain when chain-of-thought can hurt. These support the plausibility of active-inference decision rules around LLMs, but they do not provide a direct template for V3's internal Gaussian-belief transformer. They should be treated as adjacent, not canonical.
