# Evidence Pack - active-inference-lm-efficacy

## Investigation Object

The immediate object is `docs/research/2026-06-27-active-inference-policy-investigation.md`, copied from the live checkout into this isolated worktree. Its central recommendation is narrow: if active inference is added to V3, the first serious implementation should be an opt-in, no-grad policy scorer at generation or candidate-selection time, not a replacement training loss. It states that a principled feature must define an action space, transition model, likelihood, prior preferences, and an epistemic or ambiguity term, and that the plausible wins are calibration, uncertainty-aware continuation choice, long-horizon coherence, and community/applicant selection rather than guaranteed perplexity reduction.

The report also identifies candidate implementation points and blockers. The existing V3 model carries Gaussian belief tuples, iterative E-step inference, gauge transport, sigma uncertainty, and generation-time selection, but `forward()` returns logits/loss rather than a reusable final belief object. A belief-rollout helper would be needed before a clean EFE policy layer. The report also warns that a current covariance-sensitive oracle gap and the fixed-flat model-channel transport path matter before any train-through-policy or non-flat covariant rollout claim.

## Research Wiki Evidence

The Research wiki page `[[Active Inference]]` distinguishes active inference from bare FEP and predictive coding. It says V3 currently imports the perceptual half, namely precision-weighted belief updating as attention, but generally not expected-free-energy planning over actions. The page explicitly flags the mapping from active inference policy selection to a sequence model as loose, which is a major red-side pressure point.

The page `[[Expected Free Energy]]` gives the canonical active-inference object: policies are scored by expected free energy over future outcomes, with a risk-plus-ambiguity form

```text
G(pi) = KL[q(o | pi) || p(o | C)] + E_{q(s | pi)} H[p(o | s)].
```

It also gives the epistemic/pragmatic reading and the softmax-over-negative-EFE policy rule, including a policy precision parameter. This is the blue side's main theoretical foothold, but the page marks the mapping from EFE to GL(K) attention and multi-agent generalization as program synthesis rather than an external-source claim.

The method page `[[Free-energy principle active inference]]` says V3 takes the perceptual/free-energy inference semantics and supplies richer geometry, but it does not adopt active inference's action/policy machinery. Its limitation paragraph is relevant: the FEP is broad, hard to falsify, and leaves representation and metric choices under-specified.

The project page `[[VFE Transformer Program]]` records the current project as a language-model instantiation of iterative variational inference over Gaussian token beliefs. It emphasizes learned GL(K) transport, information geometry, SPD covariance, and attention as precision weighting. The run note `[[2026-06-27-gauge-transport-ablation-suite]]` records the strongest current empirical effect: learned transport around 154 PPL versus frozen random around 279 and identity around 268 at the ablation operating point, with a three-seed scaling noise floor of roughly 0.6-1.1 percent CV. This matters because any proposed EFE gain must beat compute-matched and seed-aware baselines rather than absorbing credit from the already-measured gauge-transport effect.

## External Canon

Friston et al. 2017, "Active Inference: A Process Theory" defines the discrete MDP active-inference process theory. In the wiki source note `friston-2017-active-inference-process-theory.md`, the policy prior is summarized as `P(pi) = softmax(-gamma * G(pi))`, and expected free energy decomposes into risk plus ambiguity or epistemic plus extrinsic value. This source establishes the canonical policy-selection pattern, not an LLM benchmark.

Friston et al. 2016, "Active Inference and Learning" separates fast state inference, fast policy selection, and slower parameter learning. The source note states that policies are selected by minimizing EFE over future time steps and that learning proceeds through parameter updates on a slower timescale. This is evidence against replacing V3's training loss with a naive EFE objective before a proper policy/generative-model layer exists.

Smith, Friston, and Whyte 2022 gives the operational POMDP template with `A`, `B`, `C`, `D`, and `E` components: likelihood, transitions, preferences, initial-state priors, and policy priors. Its source note states the risk-plus-ambiguity equation and softmax policy posterior. This is the strongest implementation-facing canon, because it names exactly what a V3 EFE implementation would have to instantiate.

Parr and Friston 2019 supplies the generalised-free-energy / active-inference machinery for policy selection. The local source note is a stub but records the relevance: expected/free-energy machinery clarifies exploratory and exploitative behavior under one variational objective.

Buckley et al. 2017 is useful as a boundary source. Its note derives the continuous-state FEP and predictive-coding dynamics, but explicitly stops short of expected-free-energy policy selection. This supports the distinction between V3's existing perceptual-free-energy machinery and the proposed action/policy layer.

Popper 1959 is relevant to the philosophy-of-science lens. The source note says empirical claims must forbid something. An active-inference LM proposal is scientifically useful only if it names falsifiers such as shuffled sigma matching true sigma, matched-compute beam search erasing the gain, epistemic-value proxies failing to correlate with realized uncertainty reduction, or EFE collapsing into log-probability reranking.

## Current Literature Adjacency

A live web/arXiv check on 2026-06-27 found recent adjacent work using active-inference or free-energy ideas around LLMs. ODAR, an active-inference route-selection preprint at `https://arxiv.org/abs/2602.23681`, reports adaptive fast/slow LLM routing and free-energy/risk-sensitive answer fusion under compute-matched evaluation. "Free Energy Heuristics," an arXiv preprint at `https://arxiv.org/abs/2606.15877`, argues that EFE under uncertain precision predicts when longer chain-of-thought should stop helping and start hurting. "Active Inference for Self-Organizing Multi-LLM Systems" at `https://arxiv.org/abs/2412.10425` implements active inference as a cognitive layer above multi-LLM prompt/search systems. Sajid et al., "Active inference, Bayesian optimal design, and expected utility" at `https://arxiv.org/abs/2110.04074`, is not LLM-specific but clarifies that EFE bridges expected utility and Bayesian optimal design under limiting cases.

These papers are relevant because they show active-inference decision rules are being explored around LLM routing, reasoning length, and multi-agent prompting. They are adjacent evidence only. They do not establish that V3's internal Gaussian-belief transformer will benefit from EFE policy scoring.

## Code References

- `vfe3/belief.py:22` defines `BeliefState` with `mu`, `sigma`, `phi`, and optional `s` and `r` channels. This is the state object a future EFE rollout would need to expose cleanly.
- `vfe3/model/model.py:644` starts `VFEModel.forward()`. The code encodes token ids to beliefs at `vfe3/model/model.py:660`, applies positional gauge frames at `vfe3/model/model.py:661`, calls the E-step stack at `vfe3/model/model.py:725`, and returns logits directly at `vfe3/model/model.py:793-794` when `targets is None`. It does not currently return final beliefs as a public inference result.
- `vfe3/model/model.py:1165` starts `generate()`. The generation loop calls `forward(context)` at `vfe3/model/model.py:1201`, reads last-token logits at `vfe3/model/model.py:1202`, then chooses by greedy, top-k, top-p, temperature, and multinomial sampling at `vfe3/model/model.py:1203-1220`. This is the natural hook for an inference-time policy scorer.
- `vfe3/free_energy.py:268` defines `attention_weights()` as a softmax over negative energy with optional prior. `vfe3/free_energy.py:307` defines `reduced_free_energy()`. `vfe3/free_energy.py:327` defines the scalar free-energy assembly with attention entropy, but `vfe3/free_energy.py:341` exposes the observation likelihood term only as an optional argument and `vfe3/free_energy.py:401` subtracts it only if provided. No production caller makes this a live training objective.
- `vfe3/model/prior_bank.py:223` and `vfe3/model/prior_bank.py:230` encode token and model-channel beliefs; `vfe3/model/prior_bank.py:312` decodes beliefs to logits. These are relevant to preference/outcome modeling.
- `vfe3/model/model.py:1021-1062` builds model-channel gamma energy using flat transport from `phi`. This limits claims about a full active-inference community policy over `s` unless the design explicitly accepts flat model-channel semantics.
- `vfe3/inference/e_step.py:423-435` builds covariance-sensitive transport using live `mu` leaves but currently closes over `belief.sigma` and detached `belief.sigma` for sigma. This is the known issue from the earlier active-inference report and matters for any covariant non-flat train-through-policy story.

## What This Evidence Does Not Settle

The evidence does not show that EFE improves perplexity, calibration, long-horizon coherence, or candidate selection in V3. It establishes only that the theory supplies a policy-selection functional, the code has plausible belief and generation hooks, and the current project already distinguishes perceptual inference from active policy selection. Efficacy remains an empirical question requiring matched-compute controls.

The evidence also does not settle the right preference distribution. Token-level preferences, task success, lower ambiguity, lower future CE, community utility, repetition penalties, and calibration all imply different active-inference semantics. Without an explicit preference model and transition/outcome definition, EFE is only a label for reranking.

Finally, the evidence does not settle whether an "active inference language model" should mean a decoder-only LM with an EFE reranker, a closed-loop tool/user agent, a community-selection model, or a train-time active-inference architecture. This debate evaluates the narrow V3-first claim in `00_claim.md`, not all possible active-inference LMs.
