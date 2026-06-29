# Simulated Expert Memo - Red Philosophy-of-Science

Side: red. Round: opening. Memo path: memo_red_philosophy-of-science.md.

## Source-of-truth rule

The research report can state a proposal, but it cannot certify that the proposal is active inference or worth testing. The scientific status of the claim depends on risky predictions and possible refutation.

## Newly-discovered canon

- Popper (1959), The Logic of Scientific Discovery: empirical claims require potential falsifiers; a claim that forbids no observation has low empirical content.
- Lakatos (1978), The Methodology of Scientific Research Programmes: auxiliary hypotheses can protect a research program, so a test must say which adjustment counts as rescue and which counts as failure.
- Buckley et al. (2017), "The free energy principle for action and perception," arXiv:1705.09156: FEP implementations need their assumption structure disclosed, which is a useful norm for preventing broad FEP language from becoming non-refutable.

## Memo

The claim is better than a grand "active inference LMs work" thesis because it is narrow and it rejects a train-time EFE replacement. Red should still attack the phrase "theoretically legitimate and empirically worth testing" as too permissive. Many interventions are testable in the weak sense. Scientific content comes from saying what would make the active-inference interpretation fail [Popper 1959].

The danger is elasticity. If the scorer helps, the success may be called active inference. If it fails, defenders can say the preferences, horizon, transition model, candidate set, or ambiguity proxy were wrong. That is a Lakatosian protective belt unless the debate fixes which failures count against the claim [Lakatos 1978]. The opening should demand direct falsifiers before accepting "worth testing" as more than exploratory engineering.

The conceptual equivocation is also clear. "Policy" in active inference has a technical role in a generative model; "policy" in LM decoding can mean any next-token selection rule. "Preference" in active inference is a prior over outcomes; in LM practice it can mean user taste, likelihood, anti-repetition, calibration, or task reward. "Epistemic value" is expected information gain under the model; a sigma penalty is not epistemic value unless it is tied to an outcome likelihood [Friston et al. 2017; Smith et al. 2022].

Direct falsifier from this lens: pre-register a scorer where C, transition, outcome likelihood, ambiguity, horizon, and candidate generator are fixed. Then state that the active-inference claim fails if matched-compute decoding/reranking erases the gain, if shuffling sigma leaves performance unchanged, or if changing C does not predictably change rankings. Without those falsifiers, the claim is too loose to win.
