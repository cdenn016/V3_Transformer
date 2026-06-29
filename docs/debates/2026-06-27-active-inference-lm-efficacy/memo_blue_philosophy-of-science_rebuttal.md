# Simulated Blue Expert Rebuttal Memo - philosophy-of-science

## Charge

Police whether the blue defense is circular, whether Red's strongest point should be conceded, and whether the remaining claim is scientific.

## Concession

Red's strongest true point is that "active inference for language modeling" is too elastic unless the proposal names the policy space, transition model, outcome model, preference distribution, ambiguity or epistemic term, and failure conditions. The Research wiki itself marks the mapping from active-inference policy selection to a sequence model as loose. That page may guide the local program, but it is not external authority. External canon supplies the policy machinery; the project must supply a concrete instantiation [Friston et al. 2017; Smith et al. 2022].

## Defense

The narrow claim is not "EFE efficacy is proven." It is "a finite, explicit, no-grad EFE policy scorer is legitimate enough to test, while train-time replacement is premature." That claim is defensible because it forbids outcomes. It fails if the scorer cannot be written with explicit A/B/C/D/E analogs; if changing C does not change rankings; if ambiguity or epistemic ablations do not matter; if matched-compute log-probability, beam, truncation, or shallow sigma baselines erase the gain; or if the experiment keeps changing preferences, horizons, or candidate generators after failure.

This is a Lakatosian auxiliary hypothesis inside a broader VFE research program, not a confirmed theorem about language modeling. That status is acceptable if the auxiliary is stated in advance and exposed to risky tests. The mature blue position should therefore defend only a testable bridge from canonical EFE to V3 candidate scoring, not a broad active-inference LM rebrand. It should also separate train-time claims from inference-time scoring, because replacing the training objective before the policy/generative-model layer exists would conflate learning with action selection [Popper 1959; Lakatos 1978; Friston et al. 2016].

## Newly Discovered Canon

- Popper 1959: the claim must name observations that would count against it.
- Lakatos 1978: auxiliary hypotheses are acceptable only when they increase empirical content rather than absorb failures.
- Friston et al. 2016: active inference distinguishes learning from policy selection, supporting the no-grad inference-time-first stance.
