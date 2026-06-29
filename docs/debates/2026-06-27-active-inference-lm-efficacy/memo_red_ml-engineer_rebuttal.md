# Red Rebuttal Memo - ml-engineer

## Newly-discovered canon

- Guo, Pleiss, Sun, and Weinberger (2017), "On Calibration of Modern Neural Networks," arXiv:1706.04599, https://arxiv.org/abs/1706.04599. Use: confidence calibration must be measured; raw confidence-like values can be miscalibrated.
- Kuleshov, Fenner, and Ermon (2018), "Accurate Uncertainties for Deep Learning Using Calibrated Regression," arXiv:1807.00263, https://arxiv.org/abs/1807.00263. Use: approximate Bayesian uncertainty can be inaccurate under misspecification and approximate inference.
- Holtzman et al. (2019), "The Curious Case of Neural Text Degeneration," arXiv:1904.09751, https://arxiv.org/abs/1904.09751. Use: decoding policy can strongly change text quality with a fixed LM, so EFE must clear strong decoding controls.

## Expert memo

The claim's empirical half should be conceded in a narrow form: an opt-in, no-grad scorer is cheap enough and scientifically testable enough to run, as long as it does not replace the training loss and as long as it is evaluated against serious baselines.

The red attack is that "V3 carries Gaussian belief uncertainty" is not evidence that sigma is a useful decision variable. `BeliefState` carries `sigma` at `vfe3/belief.py:26`, and `forward()` decodes with `sigma_final` at `vfe3/model/model.py:791-792`. That establishes availability. It does not establish calibration, monotonic relation to future error, relation to ambiguity `H[p(o | s)]`, or independence from log-probability. Modern neural predictors can be poorly calibrated even when they output confidence-like probabilities [Guo et al. 2017], and approximate Bayesian uncertainties can miss empirical coverage unless calibrated [Kuleshov et al. 2018].

A strong test needs predeclared endpoints. Examples: sigma-bin calibration for next-token correctness, selective prediction curves, correlation between ambiguity proxy and realized future cross-entropy, shuffled-sigma ablation, frozen-sigma ablation, scalarized-sigma ablation, and matched-candidate reranking against length-normalized log-probability, top-p, typical sampling, and self-consistency. The three-seed noise floor recorded in the evidence pack means the observed gain must clear ordinary run variance rather than rely on a single favorable seed.

The synthesis should not attack testing itself. It should attack the inference from "worth testing" to "theoretically legitimate active inference." Testing is justified exactly because the decision value of sigma and ambiguity is unknown.
