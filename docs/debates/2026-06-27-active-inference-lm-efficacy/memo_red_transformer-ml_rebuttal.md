# Red Rebuttal Memo - transformer-ml

## Newly-discovered canon

- Meister, Pimentel, Wiher, and Cotterell (2022), "Locally Typical Sampling," arXiv:2202.00666, https://arxiv.org/abs/2202.00666. Use: language generation already has information-theoretic decoding methods that alter candidate choice without becoming active inference.
- Meister, Wiher, Pimentel, and Cotterell (2022), "On the probability-quality paradox in language generation," arXiv:2203.17217, https://arxiv.org/abs/2203.17217. Use: high model probability and perceived generation quality can diverge, so EFE must beat decoding baselines designed for this divergence.
- Meister, Pimentel, Malagutti, Wilcox, and Cotterell (2023), "On the Efficacy of Sampling Adapters," arXiv:2307.03749, https://arxiv.org/abs/2307.03749. Use: sampling adapters trade precision and recall in generated text; any EFE reranker needs a matched baseline from this family.

## Expert memo

Blue is right that the proposal should not replace training first. The code path for generation is a no-grad loop, and the natural engineering surface is post-forward candidate selection. That point supports the weaker claim that EFE-style features deserve a decoding experiment.

The active-inference label is still doing too much work. Autoregressive generation already treats next-token choice as an action-like decision, and many decoding methods score finite candidate sets. Nucleus sampling, locally typical sampling, and other sampling adapters change continuation quality under a fixed language model without claiming an active-inference generative model [Holtzman et al. 2019; Meister et al. 2022; Meister et al. 2023]. The fact that V3 can assign additional scores to candidate continuations does not separate it from a broad class of information-theoretic rerankers.

The code evidence reinforces this. `VFEModel.generate()` calls `self.forward(context)` at `vfe3/model/model.py:1201`, takes last-position logits at `vfe3/model/model.py:1202`, then chooses tokens by greedy, temperature, top-k, top-p, and sampling at `vfe3/model/model.py:1203-1220`. That is a standard generation surface. `forward()` decodes final beliefs to logits at `vfe3/model/model.py:791-792` and returns logits when `targets is None` at `vfe3/model/model.py:793-794`. The current public inference path does not expose the final belief trajectory or candidate-conditioned future observation distribution.

The right red synthesis is that a candidate-continuation scorer may be a useful decoding adapter, but it is not active inference unless it adds an actual policy-conditioned predictive model over future outcomes. If the score's ordering matches log-probability, typicality, risk-only preferences, or uncertainty penalties under the same candidate pool, then the EFE vocabulary has added no independent decision content.
