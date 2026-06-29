# Simulated Blue Expert Rebuttal Memo - transformer-ml

## Charge

Evaluate Red's reranker objection from language-model decoding practice and identify what an EFE scorer must beat to be worth testing.

## Concession

Red is right that an inference-time scorer can be weak science if it only improves over greedy decoding. Holtzman et al. and later truncation-sampling work show that decoding rules alone can change text quality while the base model is fixed, so a comparison against greedy or naive sampling would not isolate an active-inference mechanism [Holtzman et al. 2019; Finlayson et al. 2023].

## Defense

The narrow claim does not need to prove that EFE will beat the best decoder in advance. It only needs to show that the proposed scorer is worth a falsifiable V3 test because it uses information unavailable to ordinary logit-only decoding. V3 carries belief variances and gauge-frame state in `BeliefState` at `vfe3/belief.py:22-28`, and `generate()` currently selects the next token after `forward(context)` returns logits at `vfe3/model/model.py:1201-1221`. That is exactly the type of inference-time boundary where a no-grad candidate scorer can be inserted without replacing the LM training objective.

The test becomes meaningful only under strong baselines. Candidates should be held fixed across log-probability reranking, beam search, top-p or top-k sampling, and the EFE scorer. Compute should be matched by forward calls or wall-clock under the same model. Length normalization, repetition controls, and calibration metrics must be included because EFE could otherwise win by exploiting length or surface probability artifacts. If the scorer's only live term is the base log probability, it is not an active-inference contribution.

The blue-positive case is preference sensitivity plus epistemic sensitivity. A preference vector C over outcomes can target task success, calibration, uncertainty reduction, or coherence, but the declared target must move rankings at fixed candidate pool. The ambiguity or epistemic term should have a causal ablation: remove it, shuffle sigma across candidates, or replace it with entropy from logits. If quality does not change as predicted, Red's reranker criticism holds [Smith et al. 2022; Sajid et al. 2021; Holtzman et al. 2019].

## Newly Discovered Canon

- Holtzman et al. 2019 establishes that decoding choice is an active baseline class for fixed LMs.
- Finlayson et al. 2023 strengthens truncation sampling as a nontrivial baseline family.
- Sajid et al. 2021 clarifies that EFE has limiting cases that resemble expected utility or Bayesian design, so the baseline must match the declared preference and epistemic terms.
