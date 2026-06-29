# Observation Term in the E-step Without Target Leakage (2026-06-29)

## Scope

This note investigates how an observation or likelihood term could be added to the VFE transformer's
belief E-step without target leakage. The concrete failure pattern under review is: training
perplexity drops sharply, while validation perplexity or generation quality degrades. The investigation
used five read-only specialist agents, then re-checked their claims against live source, the existing
active-inference reports, and the Research wiki pages on Variational EM, Precision weighting, Expected
Free Energy, and the 2026-06-29 sigma-gate failure.

No production source was edited. The result is a design and validation plan, not an implementation.

## Executive conclusion

The current executable path is target-blind in the E-step, and that is exactly what keeps causal
language modeling honest. `forward()` produces beliefs from `token_ids` before the training branch uses
`targets`; the shifted next-token labels enter only in decode and cross-entropy
(`vfe3/model/model.py:812-823`, `vfe3/model/model.py:853-883`). The E-step path
`forward_beliefs -> vfe_stack -> vfe_block -> e_step -> e_step_iteration -> belief_gradients` has no
target or observation payload (`vfe3/model/model.py:678-743`, `vfe3/model/stack.py:69-78`,
`vfe3/model/block.py:65-85`, `vfe3/inference/e_step.py:471-485`,
`vfe3/gradients/kernels.py:205-234`). The scalar `free_energy(log_likelihood=...)` hook exists, but
the live analytic kernel and the autograd oracle both differentiate a likelihood-free functional
(`vfe3/free_energy.py:341-402`, `vfe3/gradients/oracle.py:137-143`).

The safe rule is therefore simple: an E-step observation may depend on already available prefix data,
including the current input token `x_i` or an already committed generated token, but it may not depend
on `targets[:, i] = x_{i+1}` while forming the belief used to predict that target. Once `x_{i+1}` has
arrived, it can become evidence for the next belief state. It cannot be retroactive evidence for the
belief that was scored against it.

If an observation term makes train PPL collapse while validation PPL explodes, the first suspects are
target-conditioned inner inference, a train-only likelihood path that evaluation or generation cannot
run, or overconfident covariance collapse hidden by clamps. In this codebase, causal attention masks do
not protect against a separate target side channel: they mask attention weights, not arbitrary tensors
threaded into the E-step.

## Current executable state

The data loader constructs standard causal language-model windows. `TokenWindows.__getitem__` returns
`tokens[start:end]` as input and `tokens[start + 1:end + 1]` as the target sequence
(`vfe3/data/datasets.py:162-165`). Training moves both tensors to device and calls
`model(tokens, targets)` (`vfe3/train.py:718-734`). Evaluation does the same under `torch.no_grad()`
(`vfe3/train.py:426-460`). Generation, in contrast, calls `self.forward(context)` with no targets and
reads only last-position logits before appending the sampled token (`vfe3/model/model.py:1285-1314`).

Inside `forward()`, the invariant is currently clean. With `targets is None`, the model returns logits
from `forward_beliefs(token_ids, return_logits=True)` (`vfe3/model/model.py:812-815`). With targets
present, the training branch first calls `forward_beliefs(token_ids, return_logits=False)` and only
then computes either fused chunked CE or dense decode CE from `targets`
(`vfe3/model/model.py:820-883`). This means the posterior belief is a function of `token_ids`, model
parameters, causal priors, and E-step hyperparameters, not a function of the answer tensor.

The E-step gradient itself confirms the same fact. `e_step_iteration()` calls `belief_gradients()` with
`belief.mu`, `belief.sigma`, prior means and covariances, transport, attention prior, and config knobs,
but no target, observation, or likelihood argument (`vfe3/inference/e_step.py:471-485`). The default
kernel returns

```text
grad_mu = self_mu + lambda_beta * pair_mu
grad_sigma = self_sig + lambda_beta * pair_sig
```

where the covariance gradient is only self-coupling plus belief-coupling terms
(`vfe3/gradients/kernels.py:130-136`). The oracle path builds the same likelihood-free free energy and
calls `torch.autograd.grad(F, [mu_q, sigma_q])` (`vfe3/gradients/oracle.py:137-143`). As the
sigma-collapse report already recorded, there is no `1/sigma_data` or `Lambda_o` channel in the live
belief-covariance gradient (`docs/research/active-inference/2026-06-29-why-sigma-collapses.md:44-57`).

## What counts as an observation

Canonical variational EM allows observed data in the E-step because the E-step optimizes the
approximate posterior over latent variables under fixed parameters. Neal and Hinton's functional view
is coordinate ascent on one variational free-energy or ELBO objective, and partial E-steps are licensed
when they improve that same functional (Research source note
`Research/sources/papers/neal-1998-variational-em.md:30-55`). Bogacz's Gaussian free-energy tutorial
derives the corresponding precision-weighted prediction-error updates: prior error and likelihood
error both enter the belief update, weighted by their inverse variances
(`Research/sources/papers/bogacz-2017-free-energy-tutorial.md:34-50`).

Causal language modeling adds a time-indexed constraint. For the belief used to predict position
`i + 1`, the observed datum is the prefix state `D_i = x_{\le i}`. The label `y_i = x_{i+1}` is not
observed yet by the predictor. A canonical likelihood term is legal only if its observation variable
belongs to `D_i`: the current input token, previous tokens, already committed generated tokens, causal
position features, or an external sensory channel revealed before the prediction. The shifted CE target
does not belong to `D_i`.

The active-inference rollout spec already uses the same causal boundary. It rejects appending the
environment response into a scored rollout because that would rank candidate actions using an answer not
yet earned (`docs/superpowers/specs/2026-06-28-active-inference-efe-policy-scorer-spec.md:51-55`).
The same logic applies to train-time observations: a future response or future token can advance the
real loop after commitment, but it cannot be inserted into the belief used to decide or predict.

## Leakage modes

The direct leak is to add `targets[:, i]` to the inner objective for `q_i`. This includes using it as a
categorical NLL, one-hot embedding, Gaussian pseudo-observation, target-conditioned precision, target
mask, or gathered target logit before belief inference is complete. Since `targets[:, i]` is
`token_ids[:, i + 1]`, this conditions the posterior on the answer the decoder is about to be scored
against.

The second leak is future-token access through a non-causal attention prior or any observation payload
derived from future positions. The active config's causal prior can mask attention rows, but it cannot
stop a separate observation tensor from carrying future labels into the E-step. Any implementation
therefore needs both attention-causality tests and target-side-channel tests.

The third leak is a train/eval mismatch. Evaluation runs under `torch.no_grad()` and still calls
`model(tokens, targets)` (`vfe3/train.py:426-460`). If the observation term is implemented as a
train-only autograd gradient and is absent in no-grad evaluation or generation, training and validation
will run different inference rules. Conversely, if evaluation also passes future labels into the E-step,
validation PPL becomes leaked and no longer measures causal predictive performance. Either case can
produce the reported train/validation split.

The fourth failure is covariance overconfidence rather than logical leakage. The SPD retraction keeps
covariances positive by exponentiating a whitened update and clamping to `[eps, sigma_max]`
(`vfe3/geometry/retraction.py:126-134`), and the Gaussian natural-gradient preconditioner scales the
covariance tangent by `2 Sigma grad_sigma Sigma` in the full case or `2 sigma^2 grad_sigma` in the
diagonal case (`vfe3/geometry/retraction.py:373-381`). This preserves the manifold constraint, but it
does not guarantee good conditioning. If the observation precision is too strong, sigma can hit floors
or become effectively overconfident while losses remain finite. Existing guard metrics already expose
this class through `sigma_floor_frac`, `sigma_ceil_frac`, KL saturation fractions, Fisher trace, and
causal sanity (`vfe3/metrics.py:245-265`, `vfe3/metrics.py:426-490`).

## Principled observation objective

For a Gaussian observation that is already available at the current time index, the clean term is

```text
L_obs_i = 0.5 (H mu_i - z_i)^T R^{-1} (H mu_i - z_i)
        + 0.5 tr(H^T R^{-1} H Sigma_i).
```

Its Euclidean belief gradients are

```text
grad_mu_obs = H^T R^{-1} (H mu_i - z_i)
grad_Sigma_obs = 0.5 H^T R^{-1} H.
```

The precision contribution `Lambda_o = H^T R^{-1} H` is positive semidefinite. It should be added to
the raw E-step gradient before family natural-gradient preconditioning, then passed through the existing
SPD retraction. For diagonal Gaussian beliefs this yields a first-order precision update of the expected
form: posterior precision increases by the observation precision times the step size. This matches the
inverse-variance logic summarized in the Research wiki's Precision weighting page
(`Research/wiki/concepts/Precision weighting.md:28-48`).

For a categorical token observation, the no-leakage version would reconstruct an already visible token,
for example `token_ids[:, i]`, not `targets[:, i]`. However, in the current architecture this is not
automatically a useful next-token learning signal. `prior_bank.encode(token_ids)` initializes the belief
from the same token identity (`vfe3/model/model.py:678-680`), so a current-token reconstruction
likelihood can become a double count or identity-shrinkage term unless the prior is top-down,
cross-scale, corrupted, or otherwise not already the same token table. It may still be useful for
sigma-calibration experiments, but it should not be sold as a direct next-token PPL improvement until
the held-out and generation gates pass.

## Safe implementation plan

The default path should remain `observation_mode="none"` with zero weight. The theoretically pure path
must continue to exist under default toggles.

The first implementation step should be tests, not a feature branch that hopes the tests are obvious
later. Add target-independence tests that hold `token_ids` fixed, replace `targets` with random or
shifted labels, and require `forward_beliefs(...).mu`, `forward_beliefs(...).sigma`, E-step diagnostics,
and logits to stay identical. CE is allowed to change only after the belief has been formed. Add a
future-token sentinel that mutates `token_ids[:, j]` and verifies beliefs and logits for positions
`< j` are unchanged under the causal prior. Add a generation spy proving that `generate()` and policy
rollout never pass targets.

The feature seam should be inside `forward_beliefs()`, not `forward()`. Build an observation payload
from `token_ids` only, then thread it through `vfe_stack`, `vfe_block`, `e_step`, and
`e_step_iteration`. Do not add `targets` to `forward_beliefs()` or any lower E-step signature. Do not
use `targets == -100` as an observation mask, because that mask is metadata about future labels, not
metadata about prefix observations.

The live gradient insertion point is after `belief_gradients(...)` and before
`get_family(...).natural_gradient(...)` in `e_step_iteration()` (`vfe3/inference/e_step.py:471-498`).
At that point the observation gradient can be added to `grad_mu` and `grad_sigma`, so the existing
family Fisher metric and SPD retraction own the geometry. Scalar diagnostics and oracle parity must be
updated in the same change: `free_energy_value()` and `belief_gradients_autograd()` must see the same
likelihood term, or the scalar logged objective and the actual descent direction will diverge again.

The observation term itself should live behind a registry rather than call-site branches. A minimal
surface would be an E-step observation registry with `none`, `prefix_gaussian`, and possibly a later
`prefix_categorical_reconstruct` mode. The registry returns both a scalar term for diagnostics and raw
belief gradients for the live kernel path. For any autograd-based categorical term, the code must use a
small `torch.enable_grad()` island over temporary belief leaves while detaching model parameters if the
intent is recognition-only inference; otherwise train and eval may run different rules.

## Validation and kill switches

Every candidate observation mode should pass the following gates before it is allowed into long runs.

1. Target invariance: changing `targets` with fixed `token_ids` changes CE but not E-step beliefs,
   attention maps, logits, or E-step gradient diagnostics.

2. Causal prefix invariance: changing a future input token cannot change past-position beliefs or
   logits under a causal prior.

3. Kernel/oracle parity: the analytic observation gradient and autograd-of-scalar objective agree on a
   tiny Gaussian case, including the covariance gradient.

4. Train/eval parity: the same observation rule runs in training, validation, and generation whenever
   the same prefix is available. There is no train-only target path.

5. Numeric health: sigma floor and ceiling fractions, KL clamp fractions, Fisher trace, nonfinite
   fractions, and E-step sigma-gradient norms stay within predeclared bounds.

6. Generalization: train PPL may improve only if held-out target-blind PPL and free-running generation
   do not degrade. If the improvement disappears when the observation channel is target-blinded, it is a
   shortcut.

7. Sigma gate: any claim that the observation term makes `Sigma_q` epistemic requires rerunning the
   existing sigma-validation gate. The current gate failed with negative Spearman correlation and
   near-constant covariance, so no sigma-derived EFE claim is restored by design alone
   (`Research/sources/runs/2026-06-29-sigma-gate-fail-and-collapse.md:33-51`,
   `Research/sources/runs/2026-06-29-sigma-gate-fail-and-collapse.md:117-129`).

## Experiment plan

Run four arms on the same small seeds before any expensive training. The baseline is the current
target-blind E-step. The legal observation arm is prefix/current-token only, with small observation
precision and with the same behavior in train, validation, and generation. The pseudo-observation arm
uses detached model predictions or an EMA teacher from the same prefix; this is safe from target leakage
but is self-confirming and should be labeled as calibration, not sensory grounding. The positive-control
leak arm intentionally feeds shifted targets into the E-step only inside a quarantined test harness; it
should reproduce the suspicious train-PPL collapse and is never a candidate model.

For each arm, log train CE/PPL, validation CE/PPL, generation samples, generalization gap, sigma guard
fractions, Fisher trace, E-step raw gradient norms, causal sanity, sigma-gate statistics, and
target-blinded validation. The reported success condition is not "train PPL fell." The success condition
is "target-blind held-out CE improves, generation does not degrade, no guard is load-bearing, and sigma
claims pass the sigma gate."

## Recommended decision

Do not wire next-token CE or `targets` into the E-step. That is the acausal path that explains the
train-PPL-plummets/val-PPL-explodes pattern.

If the project wants a canonical observation channel, build a default-off prefix-observation registry,
threaded from `forward_beliefs()` down to the E-step gradient, with scalar/oracle/kernel parity and
target-invariance tests in the first patch. Treat current-token reconstruction as a cautious
calibration experiment, because the current prior-bank encode path already exposes the same token
identity. The cleaner long-term theoretical form is a top-down or cross-scale prior plus a causal
likelihood on already observed data; that gives the likelihood something real to correct without
turning the E-step into an answer key.

