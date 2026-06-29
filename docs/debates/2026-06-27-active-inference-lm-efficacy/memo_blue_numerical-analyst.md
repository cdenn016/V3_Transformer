# Simulated Blue Expert Memo - numerical-analyst

## Position

The narrow claim is worth testing only if the experiment is designed to prevent EFE from winning by extra compute, proxy leakage, or post-hoc metric choice. The defense should state these constraints rather than hide them.

## Analysis

Expected free energy has two measurable pieces in the discrete POMDP form: a risk term comparing predicted outcomes to prior preferences and an ambiguity term measuring residual outcome uncertainty under the likelihood [Smith et al. 2022; Friston et al. 2017]. For V3, a candidate scorer can approximate those pieces only after defining what the outcome is. For token continuations, outcomes could be next-token distributions, task-success labels, continuation-level coherence diagnostics, or future cross-entropy under held-out text. For agent sets, outcomes could be predicted answer quality or uncertainty reduction after a route. These choices are not interchangeable.

The empirical test should be pre-registered around deltas and controls. A minimal test matrix should hold the candidate pool fixed and compare: raw length-normalized log probability, uncertainty penalty alone, preference/risk term alone, ambiguity or information-gain proxy alone, and full EFE. It should also include ordinary generation baselines under the same forward-pass budget. Kaplan et al. and Hoffmann et al. show how strongly LM loss and performance respond to compute, data, and scale; an EFE experiment that ignores compute matching cannot isolate the policy scorer [Kaplan et al. 2020; Hoffmann et al. 2022].

The strongest available V3-specific reason to test is the presence of `sigma`: V3 carries an uncertainty-bearing Gaussian belief state, so the model has more than a single logit vector. If `sigma` is calibrated and candidate rollouts alter future belief uncertainty in meaningful ways, EFE can exploit information that standard log-probability reranking discards. This is also the easiest way to falsify the mechanism. If shuffling, freezing, or replacing `sigma` with a scalar uncertainty proxy preserves the gain, the active-inference interpretation loses force.

The scorer should report diagnostics rather than only final text metrics. It should log candidate length, raw log probability, risk, ambiguity, score temperature, rollout depth, and compute. Success should require improvement on a predeclared endpoint such as calibration error, Brier score, selective prediction, task success under fixed candidate pool, long-horizon repetition/coherence, or route-selection accuracy. Perplexity improvement is neither expected nor needed for the narrow claim, because the scorer acts after training and may select lower-likelihood continuations that better satisfy preferences.

## Newly-discovered canon

- [Smith et al. 2022] Risk plus ambiguity supplies the measurable decomposition an experiment must log.
- [Friston et al. 2017] The epistemic term is expected information gain, so uncertainty proxies must predict realized uncertainty reduction.
- [Kaplan et al. 2020; Hoffmann et al. 2022] Compute and scale are major determinants of LM loss; policy-scorer gains need compute-matched controls.
- [Popper 1959] The empirical claim must rule out possible observations, such as a shuffled-uncertainty ablation matching full EFE.

## Recommendation

Defend "empirically worth testing" as a falsifiable, seed-aware, compute-matched experiment. Do not defend any unconditional efficacy claim.
