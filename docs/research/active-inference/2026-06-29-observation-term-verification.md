# Verification of the observation-term no-leakage investigation (2026-06-29)

Adversarial verification of `docs/research/active-inference/2026-06-29-observation-term-e-step-no-leakage-investigation.md`
against live source. Method: four parallel cluster verifiers (target-blindness and leakage modes; the
encode double-count; the observation math and insertion seam; whether it fixes the sigma-collapse), each
told to steelman then refute with path:line evidence, followed by an adversarial reconciler over the
four verdicts (workflow `wf_da680afe-48f`, 5 agents). The doc is **sound, with corrections**, and its
pessimistic conclusion is not only confirmed but sharpened: no no-leakage observation toggle makes the
belief covariance epistemic for next-token language modeling, and the cleanest proof is one the doc
under-uses.

## Verdict by cluster

The E-step is fully target-blind, as the doc claims. The chain
`forward -> forward_beliefs -> vfe_stack -> vfe_block -> e_step -> e_step_iteration -> belief_gradients`
carries no target, observation, or likelihood argument at any level; `forward()` forms the belief from
`token_ids` before `targets` enter, and `targets` are read only in the decode and cross-entropy
(`model.py:812-823`, `:844-883`). The covariance gradient is self-coupling plus belief-coupling and
nothing else (`kernels.py:130-136`); the `free_energy(log_likelihood=...)` hook is a gated stub the
oracle never supplies and the analytic kernel never calls. The four leakage modes are real, with one
framing caveat: the "causal prior masks attention weights, not a target side channel" point is correct
but currently hypothetical, because no observation tensor is threaded at all, so the exposure is
forward-looking rather than live. The doc also omits that the s-channel refinement (`_refine_s`,
`model.py:730`) and the precision-bias fold (`model.py:738`) are additional belief-shaping paths that
run before decode and are likewise target-blind.

The current-token double-count is confirmed and is sharper than the doc states. `prior_bank.encode`
initializes the belief `q0 = p0` from the per-token table entry for `x_i` (`_encode_per_token`,
`prior_bank.py:652-669`), and the E-step self-couples that belief back to the same entry through the
`alpha KL(q||p)` term (`kernels.py:130`). A Gaussian reconstruction likelihood on `x_i` with mean equal
to that table entry therefore reproduces the self-coupling direction `R^{-1}(mu_i - mu_p)` exactly: it
is functionally a rescaling of `alpha` and a reshaping of `sigma_p`, the same fixed point made more
confident, not new next-token evidence. The doc's escape clause, an observation against a table that is
"not already the same token table," is necessary but not sufficient: any table keyed by `x_i` (including
the `model_channel` s-tables or a separate observation embedding) is token-deterministic and adds no
conditioning information about the next token. Genuine non-redundancy requires the prior or observation
to depend on something other than `x_i`, that is context or a higher scale, or a corrupted encode input.

The Gaussian calculus and the insertion seam are correct. `grad_mu = H^T R^{-1}(H mu - z)`,
`grad_Sigma = 0.5 H^T R^{-1} H`, and `Lambda_o = H^T R^{-1} H` positive semidefinite are standard
expected-Gaussian-NLL algebra. `belief_gradients()` returns raw Euclidean gradients, the family
natural gradient is applied immediately after, and the SPD retraction follows
(`e_step.py:471-498`, `:530-536`), so the named seam (add the observation gradient to
`grad_mu`/`grad_sigma` in that gap) does let the existing Fisher metric and SPD retraction own the
geometry. The parity requirement is accurate: the analytic kernel never calls `free_energy`, the oracle
calls it without `log_likelihood`, and the scalar `free_energy_value` diagnostic is a third path, so all
three need the term wired explicitly or the logged objective and the descent direction diverge. One
implementation detail to specify: the `grad_record` raw-norm diagnostic sits between `belief_gradients`
and the natural gradient, so an observation gradient must be added after that snapshot if the logged
E-step norms are meant to include it.

Whether it fixes the sigma-collapse: it does not, and the decisive reason is data-independence. For a
fixed linear-Gaussian observation model the covariance gradient `0.5 H^T R^{-1} H` contains no `z_i`, so
it is both data- and token-independent. A legal current-token Gaussian observation therefore injects a
near-constant uniform precision bump applied to every token, which tightens the covariance roughly
uniformly and deepens the collapse rather than making it epistemic. This is the cleanest disproof of the
originating hypothesis, and the doc under-uses it, leaning instead on the double-count and
target-blindness arguments. The doc's first-order "posterior precision increases by the observation
precision times the step size" is correct only in the small-step, unsaturated regime: under the live
natural gradient `nat_sigma = 2 sigma^2 grad_sigma` and the multiplicative SPD-affine retraction with
`[eps, sigma_max]` and trust-region clamps (`retraction.py:126-134`, `:373-381`), the update is
geodesic and clamp-bounded, not additive.

## The crux

The sigma-collapse is structural for next-token language modeling under this architecture, not curable
by any no-leakage toggle, for three separable reasons in increasing depth. First, definitionally, a
belief used to predict `x_{i+1}` may not acquire precision from `x_{i+1}`; that is exactly the forbidden
target leak the validation gates exist to catch, so no prefix-only term can calibrate `Sigma_i` against
the token it predicts. Second, the only trivially legal observation, a current-token reconstruction
against the table `encode` already used, is functionally an `alpha` rescale. Third, even granting a
legal Gaussian observation, its covariance gradient is data-independent, so it is uniform shrinkage, not
per-token epistemic precision. Per-token epistemic covariance from legal data would require both a
heteroscedastic or context-dependent observation model (`R_i` or `H_i` depending on the prefix) and a
prior that genuinely disagrees with the datum (top-down or cross-scale). The latter is unimplemented and
out of scope in this single-scale model, where `prior_bank.py:213-221` assigns the cross-scale tower to
the multi-agent model; and even that combination would calibrate prefix self-consistency rather than
next-token predictive uncertainty, so it would not by itself restore the failed sigma gate (negative
Spearman, near-constant covariance). This is consistent with the earlier reconciliation that the
observation-free E-step is a manuscript-sanctioned structural-EM reduction, not a bug.

## Corrections to fold into the doc

The principled-objective section should state plainly that `grad_Sigma_obs = 0.5 H^T R^{-1} H` is data-
and token-independent for fixed `H, R`, so a legal current-token Gaussian observation is uniform
shrinkage that worsens the collapse; this is the sharpest proof of the doc's own verdict. The precision
update should be described as multiplicative and clamp-bounded, not additive. The recommended top-down
or cross-scale prior fix should be labeled an unimplemented cross-repo design proposal (assigned to the
multi-agent model), not a seam that exists here. The double-count escape clause should be tightened from
"not the same token table" to "depends on something other than `x_i`." Minor citation fixes: the direct
`model(tokens, targets)` call the doc attributes to `train.py:718-734` actually lives inside `train_step`
and literally at the eval path `train.py:460`; the `q = p` initialization is in `_encode_per_token`
(`prior_bank.py:652-669`), not the dispatch line; and `prior_handoff_rho` defaults to `1.0` but
`n_layers` defaults to `1`, so the handoff is applied after the single block and never re-consumed, which
means the default-depth E-step prior is exactly the encode-time token entry and the prior only drifts at
`n_layers > 1`.

## Gate additions

The seven validation gates are well-targeted and collectively catch the train-collapse, validation-
explode leakage, but four additions are warranted. A positive-control gate should assert that the suite
actually fires on a planted `targets[:, i]` leak, since a gate suite that never fails on a known leak is
untrustworthy; this is the most important omission. A batched-equals-per-sample (and shuffle-batch)
equivalence gate should cover the new observation payload, because `forward_beliefs` relies on sequence
independence and a payload threaded through `vfe_stack` could couple across the batch. Gate four
(train/eval parity) should explicitly assert the term is numerically active under `torch.no_grad` eval,
so a categorical autograd-island term cannot silently become a train-only path. And an M-step gate should
require that `param.grad` is invariant to target substitution at fixed `token_ids`, because belief-value
invariance alone does not guarantee that no target-correlated leaf reaches the M-step.

## Recommendation

Treat the doc as a sound design and validation plan with the corrections folded in, but treat the
originating question as answered in the negative: do not implement a current-token or any fixed-model
Gaussian prefix observation expecting an epistemic covariance, because it reduces to an `alpha` rescale
plus uniform shrinkage. Keep the default `observation_mode = "none"` and the pure path intact. The only
theoretically promising route to an epistemic covariance, a top-down or cross-scale prior with a causal
likelihood on already observed data, is a genuine architectural change that lives in the multi-agent
model, not this single-scale repository. For V3 the realistic conclusion is that the sigma-collapse is a
structural property of the target-blind, encode-from-token-identity next-token model and should be
accepted as such rather than chased with an observation toggle.

## Open questions

What concrete heteroscedastic or context-dependent observation model would make the covariance carry
per-token signal within legal data is named as the only legal route but is designed and tested nowhere.
Whether even a top-down cross-scale prior with a causal likelihood would calibrate next-token predictive
uncertainty, as opposed to prefix self-consistency, is unproven, and design alone does not address the
failed sigma gate.

*Provenance: workflow wf_da680afe-48f (4 cluster verifiers + adversarial reconciler), 5 agents,
~354k tokens, verified against HEAD of feat/efe-belief-cache-phase3a. No production source edited.*
