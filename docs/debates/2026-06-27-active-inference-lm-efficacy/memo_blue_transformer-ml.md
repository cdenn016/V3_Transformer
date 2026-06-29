# Simulated Blue Expert Memo - transformer-ml

## Position

The claim is plausible as a decoding-time or routing-time experiment and weak as a training-objective claim. The best transformer-ML framing is: keep the autoregressive model trained in the ordinary way, then test whether a no-grad policy scorer improves candidate choice under matched compute.

## Analysis

The external LM canon does not say that language models should be trained by expected free energy. The Transformer paper establishes attention as a softmax-weighted sequence operator and evaluates sequence modeling tasks with standard supervised objectives [Vaswani et al. 2017]. GPT-2 frames decoder-only language modeling as autoregressive estimation of `p(x_i | x_{<i})` over token sequences and shows that scale and data diversity make that objective useful for zero-shot task transfer [Radford et al. 2019]. Kaplan et al. and Hoffmann et al. show that pretraining loss follows compute, data, and parameter scaling laws under autoregressive transformer training, with Hoffmann et al. emphasizing compute-optimal allocation between parameters and tokens [Kaplan et al. 2020; Hoffmann et al. 2022].

Those facts support blue only because the claim is narrow. An inference-time scorer over explicit candidates does not ask V3 to abandon next-token modeling. It is a decision layer applied after the model has produced, sampled, or enumerated candidate continuations. That layer can optimize criteria that raw token likelihood does not directly optimize: preference satisfaction, lower ambiguity, uncertainty reduction, answer routing, or agent/community selection. The evidence pack's recent adjacent LLM work on active-inference routing, reasoning-length control, and multi-LLM self-organization is relevant as motivation, but it is not proof of V3 benefit.

The test must beat ordinary decoding baselines. If EFE reranking is compared only against a weak sampler, the result will be uninterpretable. The correct control set includes greedy, temperature, top-k, top-p, beam or candidate reranking with the same candidate pool, length-normalized log probability, uncertainty-only penalties, and task-specific preference-only scores. The compute budget must include all candidate rollouts and belief updates. If the EFE scorer wins only by spending more forward passes, the claim becomes an engineering trade, not evidence for active inference.

## Newly-discovered canon

- [Vaswani et al. 2017] Standard transformer attention is a softmax-normalized sequence operator; this anchors the baseline being extended.
- [Radford et al. 2019] GPT-style language modeling uses autoregressive next-token factorization and demonstrates why that training path should not be displaced without strong evidence.
- [Kaplan et al. 2020] LM loss scales predictably with compute, data, and model size, making matched-compute controls mandatory.
- [Hoffmann et al. 2022] Compute-optimal LM training depends on correct allocation of model size and data, reinforcing that new training objectives need stronger evidence than a decoding-layer experiment.

## Recommendation

Defend inference-time EFE as a candidate-selection experiment. Require compute-matched baselines and do not claim a perplexity win in advance.
