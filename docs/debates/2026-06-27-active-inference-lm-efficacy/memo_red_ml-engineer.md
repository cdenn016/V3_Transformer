# Simulated Expert Memo - Red ML-Engineer

Side: red. Round: opening. Memo path: memo_red_ml-engineer.md.

## Source-of-truth rule

Project reports and run notes can motivate hypotheses, but empirical worth is decided by controlled benchmarks. External ML and active-inference literature set the burden for cost and baseline comparison.

## Newly-discovered canon

- Malekzadeh and Plataniotis (2022), "Active Inference and Reinforcement Learning," arXiv:2212.07946: active inference in POMDPs selects actions by minimizing EFE, but its use is limited by the computational challenges associated with EFE.
- Holtzman et al. (2019), arXiv:1904.09751: decoding choices alone can change generation quality for a fixed LM, so decoding baselines are not optional.
- Kaplan et al. (2020), "Scaling Laws for Neural Language Models," arXiv:2001.08361, and Hoffmann et al. (2022), "Training Compute-Optimal Large Language Models," arXiv:2203.15556: LM comparisons must account for compute and scale, not just raw metric movement.

## Memo

The phrase "empirically worth testing" is weak unless it names a minimum detectable effect and a compute-matched baseline. The evidence pack says the current V3 gauge-transport ablation has a measured seed noise floor around 0.6 to 1.1 percent CV, and that learned transport already explains a large measured perplexity effect. A new EFE scorer must be evaluated above that noise floor and must not absorb credit from the existing transport mechanism.

Inference-time EFE is costly in the V3 code shape. `generate()` performs a full `forward(context)` call per generated token at `vfe3/model/model.py:1198-1202`. A candidate-continuation scorer multiplies that by candidate count and rollout horizon unless it has a belief-cache or analytic rollout. Since active-inference planning is already known to face EFE computational challenges [Malekzadeh and Plataniotis 2022], red should demand a cost curve: wall time, forward calls, memory, candidate count, horizon, and quality per unit compute.

The baseline set must include at least these arms: greedy, temperature, top-k, top-p, beam or length-normalized beam, log-probability reranking, uncertainty reranking using the same sigma features, and a learned shallow reranker trained on the same candidate features. Holtzman et al. show why decoding baselines can dominate perceived quality independently of the underlying model [Holtzman et al. 2019]. A red judge should reject any "EFE helped" claim that compares only against greedy decoding or samples a different candidate set.

Direct falsifier from this lens: under fixed candidates and equal forward-call budget, EFE fails to improve perplexity, calibration error, long-horizon repetition, or task success beyond the measured seed and bootstrap intervals. A stronger falsifier is sigma shuffling: if permuting sigma among candidates leaves the EFE score's performance unchanged, the epistemic term is not using V3 uncertainty in a meaningful way.
