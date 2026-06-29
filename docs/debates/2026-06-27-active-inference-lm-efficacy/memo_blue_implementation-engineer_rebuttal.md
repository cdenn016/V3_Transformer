# Simulated Blue Expert Rebuttal Memo - implementation-engineer

## Charge

Check whether V3 has an actual implementation path for the defended narrow claim and whether Red's code objections should force concession.

## Concession

Red is correct that the current public inference interface is not already an active-inference policy model. `VFEModel.forward()` encodes token ids, applies positional gauge frames, runs `vfe_stack`, decodes final beliefs to logits, and returns logits when no targets are supplied at `vfe3/model/model.py:644`, `vfe3/model/model.py:725`, and `vfe3/model/model.py:792-794`. It does not return a public final-belief object suitable for reusable policy rollouts. The model-channel path also uses flat transport in `_gamma_energy()` at `vfe3/model/model.py:1021-1062`, so agent-set policy claims must not imply non-flat covariant community rollout.

## Defense

The code still supports the narrow feasibility claim. The belief tuple exists as `BeliefState(mu, sigma, phi)` at `vfe3/belief.py:22-28`; `PriorBank.encode()` and `PriorBank.decode()` provide token-to-belief and belief-to-logit interfaces at `vfe3/model/prior_bank.py:223` and `vfe3/model/prior_bank.py:312`; generation calls `forward(context)` and then performs all selection at `vfe3/model/model.py:1201-1221`. A no-grad scorer can therefore be designed as a wrapper around explicit candidate continuations: build candidate sequences, roll them through the existing model, expose or reconstruct terminal beliefs, compute declared risk and ambiguity terms, and select among candidates. That is finite policy scoring, not train-time replacement.

The implementation boundary should be written as a blocker list. First, add a public no-grad inference helper that returns final beliefs and logits for a supplied candidate continuation. Second, define the transition semantics for appending tokens under teacher-forced candidate rollout. Third, define the outcome and preference objects outside the base training loss. Fourth, disallow gradients through the scorer in the first experiment. Fifth, restrict model-channel or agent-set claims to flat transport unless a separate non-flat covariant path is implemented and tested. These restrictions align the code with the active-inference canon's separation of state inference, policy scoring, and learning [Friston et al. 2016; Smith et al. 2022].

## Newly Discovered Canon

- Smith et al. 2022 supplies the A/B/C/D/E checklist that can be mapped onto V3 code objects for an implementation spec.
- Friston et al. 2016 supports separating fast policy selection from slower learning.
- The verified V3 code path supplies a real generation hook but no public belief-rollout API: `vfe3/model/model.py:1201-1221` and `vfe3/model/model.py:792-794`.
