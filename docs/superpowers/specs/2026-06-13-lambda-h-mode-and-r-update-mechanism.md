# The hyper-prior `(r, lambda_h)`: is `learnable_r` a neural network, how should `r` evolve, and should `lambda_h` mirror `alpha_mode`

Date: 2026-06-13
Status: investigation complete; full implementation delivered (see "Implementation" below)
Branch: `vfe3-learnable-r-lambda-h-mechanism-2026-06-13`
Method: 6-expert workflow panel (variational, info-geometry, implementation, ML/optimizer,
philosophy-of-science, gauge) plus 3 adversarial red-team verifications; every load-bearing
formula and manuscript line re-verified directly against source.

## The three questions

The hyper-prior channel adds `lambda_h * mean_i KL(s_i || r)` to the free energy, pulling the
model-channel beliefs `s_i` toward a global centroid `r = (r_mu, r_sigma_log)`. The toggle
`learnable_r=True` un-freezes `r` as a trainable `nn.Parameter` (`prior_bank.py:193-194`)
optimized by the AdamW M-step (`train.py:116-122`). This document answers: (Q1) is
`learnable_r` a neural network in the project's banned sense; (Q2) does it evolve the way the
embedding/prior tables evolve, and is that the right mechanism; and (Q3) should `lambda_h`
receive an `alpha_mode`-style registry.

## The architecture the questions sit in

The model is variational EM. The E-step refines beliefs `q_i = (mu, sigma, phi)` by iterative
free-energy minimization inside each forward pass; beliefs are computed states, not parameters.
The M-step trains the PriorBank prior tables by AdamW, with the E-step unrolled into the graph
(`train.py:1-9`). The embedding/prior tables `mu_embed`, `sigma_log_embed`, `s_mu_embed` are
`nn.Parameter`s the project explicitly calls "PRIORS (nn.Parameter), not a neural map"
(`prior_bank.py:172`).

The free energy carries three coupling coefficients on KL terms, and they are treated with very
uneven sophistication. The self-coupling weight `alpha` on `KL(q_i||p_i)` has a full registry
(`vfe3/alpha_i.py`): a constant form, a closed-form state-dependent envelope
`alpha* = c0/(b0 + D)`, a per-coordinate envelope, and a learnable NN-exception. The
belief-coupling block weight `lambda_beta` is a scalar with a `learnable_lambda_beta`
NN-exception. The model-coupling weights `gamma_ij` are softmax-determined per pair. The
hyper-prior weight `lambda_h` is a bare scalar (`config.py:197`): no registry, no state-dependent
form, no learnable form. It is the only coupling coefficient without a mode registry, which is
already a deviation from the project's "registry behind every seam" mandate.

The hyper-prior is naturally a `(mean, precision)` pair: `r` is the centroid (the mean the `s_i`
are pulled toward) and `lambda_h` is the strength of that pull (the precision of the
hyper-prior). The two questions about `r` and the question about `lambda_h` are therefore one
question about how the object `(r, lambda_h)` should be treated.

## Q1 — `learnable_r` is not a neural network

All six experts and the adversarial verifier agree, and the verdict survives a direct
falsification attempt (severity-if-wrong: low). The demarcation the project actually draws is
not "is it an `nn.Parameter` trained by AdamW" — the sanctioned embedding tables are exactly
that. The banned object is a neural map: an `nn.Linear`/MLP/activation that transforms an input
vector into an output vector through learned weights. `r` is a single `(K,)` Gaussian centroid
with no input argument, no weight matrix, and no nonlinearity. Its only two consumers read it as
a `DiagonalGaussian(mean, variance)`: the second slot of `KL(s_i||r)` (`model.py:719-722`) and
the E-step prior target in `_refine_s` (`model.py:419-420`). Training it by AdamW is the M-step
of variational EM applied to a prior leaf, which the constraint explicitly permits.

`r` is also not a fourth NN-exception belonging on the CLAUDE.md list. The three listed
exceptions (linear decode `mu @ W^T`, head mixer/CG coupling, Regime-II `connection_W`) share two
properties `r` lacks: each multiplies a data vector to produce a transformed output, and each
breaks strict gauge equivariance when nonzero. The gauge analysis confirms `r` carries no frame
and no transport edge, sits only inside the gauge-invariant divergence `KL(s_i||r)`, and so
cannot violate the cocycle the way `connection_W` does (whose edge factor is invariant only at
`W=0`). `r` is strictly less map-like than even the embedding tables, which at least index on
token id. The manuscript names this object directly: at the top scale `r` is "a fixed boundary
condition encoding evolutionary or training-time structural defaults"
(`Participatory_it_from_bit.tex:554`), and `learnable_r` fills precisely the "training-time"
slot — an empirical-Bayes stand-in for the deferred top-down meta-agent `r_i = Ω̃[s_I^{(s+1)}]`
(`Participatory:629`), which needs the unbuilt scale-`(s+1)` agent.

The verifier's sharpest attack was that under the active config (`s_e_step=True`) the forward
`lambda_h*KL(s||r)` term is gated off (`model.py:660`), so if that were `r`'s only entry point
the toggle would be inert and the "trains like the embeddings" claim vacuous. It executed a
forward/backward under the live toggles and measured `r_mu.grad` and `r_sigma_log.grad` both
nonzero: `r` trains through the unrolled `_refine_s` trajectory instead. The toggle is live, and
the verdict holds.

## Q2 — it evolves exactly like the embeddings, and a closed-form alternative exists but is regime-dependent

Mechanically `r` evolves identically to the embeddings: the same AdamW optimizer, the same
per-table learning-rate split (mean at `m_mu_lr`, log-scale at `m_sigma_lr`), the same
backprop-through-unrolled-E-step. The one difference is `weight_decay=0`, which matches the
no-decay exemption the unigram-bias prior and the gauge frame already carry, not the decaying
`mu_embed` group. So the literal answer to "is it implemented the way the embeddings/priors
evolve" is yes.

`r` does, however, admit a purer update the embeddings cannot. The embedding tables feed the
coupled decode cross-entropy through a non-conjugate softmax and have no closed-form M-step,
which forces gradient descent. `r` appears in a single term, `lambda_h * sum_i KL(s_i||r)`, and
the argmin of that term over a diagonal-Gaussian `r` is the closed-form forward-KL barycenter of
`{s_i}`: `r_mu* = mean_i s_mu_i` and `r_sigma* = mean_i [s_sigma_i + (s_mu_i - r_mu*)^2]`, the
moment-matched centroid. I verified this against numerical argmin to roughly `1e-6`, and it is
the diagonal unit-weight specialization of the manuscript's own meta-agent barycenter
(`Participatory:2189-2193`). This closed form is the same kind of object the codebase already
prefers everywhere it can: `alpha*`, `beta*`, and `gamma*` are all closed-form stationary points
of `F`, not learned by backprop. By that standard, AdamW on `r` is the less pure of two available
M-steps, and the optimizer lens adds that AdamW's diagonal preconditioner and momentum converge
to a variance-warped, lagged centroid rather than the exact barycenter.

The adversarial verifier landed a genuine and important qualification here. The closed-form
barycenter is the exact M-step only when `KL(s_i||r)` is `r`'s sole objective, which holds in the
scored regime (`s_e_step=False`, `lambda_h>0`). Under the user's active config
(`s_e_step=True`, `prior_source='model_channel'`), the standalone `lambda_h*KL(s||r)` loss term is
gated off; `r` enters only through `_refine_s`, the refined `s` then replaces the belief and flows
into the decode cross-entropy, so `r` is coupled to the CE through `s`. In that regime the
barycenter of `{s_i}` is not the argmin of what the model actually minimizes, and
AdamW-through-unroll is the update consistent with the active objective. The recommendation is
therefore regime-scoped: offer the closed-form barycenter as the pure path for the scored
`s_e_step=False` regime, and keep AdamW-through-unroll for the coupled `s_e_step=True` regime,
where it is the correct tool rather than a less-pure stand-in.

## Q3 — `lambda_h` should get an `alpha_mode`-style registry; the manuscript names this exact extension

The core recommendation is correct and survives adversarial attack. The manuscript writes the
self-coupling `alpha_i KL(q_i||p_i)` and the hyper-prior `lambda_h KL(s_i||r_i)` side by side in
the same functional (`Participatory:~3766`) and then states plainly: "On the model fiber the
fixed hyper-prior weight λ_h plays the analogous role, and a state-dependent model precision under
the substitution `(β,Ω,q,p) → (γ,Ω̃,s,r)` is a parallel extension not developed here." The
reverse framing appears at line 1279, where the entire state-dependent-`alpha` development is
introduced as "mirroring the explicit λ_h on the model channel." The two coefficients are the
same role on parallel fibers — `(q,p,alpha)` on the state fiber and `(s,r,lambda_h)` on the model
fiber — and the manuscript already anticipates promoting `lambda_h` to a state-dependent
precision but never carries it out, because its simulations ran `lambda_h ∈ {0, 1}`
(`gamma_ij = 0`, line 1296). Giving `lambda_h` a registry completes the manuscript's own
unfinished symmetry, and `(r, lambda_h)` then forms a complete empirical-Bayes hyper-prior: the
barycenter supplies the mean, the envelope supplies the precision.

The state-dependent form is `lambda_h*_i = c0_h / (b0_h + KL(s_i||r))`, the stationary point of
`lambda_h * KL(s_i||r) + R_h(lambda_h)` with `R_h(lambda_h) = b0_h*lambda_h - c0_h*log(lambda_h)`.
I verified symbolically that `argmin_λ [λ·D + b0·λ - c0·log λ] = c0/(b0+D)` and that the
envelope cancellation `d/dD[envelope value] = λ*` holds, identical to the `alpha` envelope. This
is the same Gamma-prior-on-precision construction the manuscript gives for `alpha`
(`GL(K)_attention.tex:956-964`).

Two honesty qualifications, both load-bearing, are what the adversarial verifier correctly forced.
First, this is a principled extension, not a derived manuscript result: no equation in any
manuscript carries a state-dependent `lambda_h`, and the correspondence should not be sold as
"EXACT Gamma conjugacy," because the manuscript declines that label even for `alpha` itself —
`alpha` "weights an entire divergence rather than parameterizing a Gaussian precision, so the
correspondence is structural rather than the exact Normal-Gamma conjugacy" (line 1344). The
`lambda_h` analogue inherits exactly that structural-not-exact status, no weaker and no stronger.
Second, a state-dependent `lambda_h` is not a free drop-in. The envelope cancellation holds only
when the regularizer is present in the differentiated free energy. The `s` E-step currently passes
`alpha_mode='constant'`, `value=lambda_h`, with zero regularizer (`model.py:429`), matching the
manuscript's bare-`lambda_h` envelope gradient (line 1414, no `R_h`). A state-dependent
`lambda_h` must therefore add `R_h(lambda_h)` both to the scored forward term and to the `s`
E-step objective; without it the product-rule term does not cancel, the kernel coefficient stops
equaling `lambda_h*`, and `lambda_h*_i → 0` as `KL(s||r)` grows would silently kill the
hyper-prior with no penalty. The `constant` and `learnable` modes are free registry adds; only
`state_dependent` carries the `R_h` plumbing cost.

## Verified mathematics

The forward-KL barycenter `r_mu* = mean_i s_mu_i`, `r_sigma* = mean_i [s_sigma_i + (s_mu_i -
r_mu*)^2]` matches numerical argmin of `sum_i KL(s_i||r)` to roughly `1e-6` (torch). The
state-dependent envelope `argmin_λ [λ·KL(s||r) + b0_h·λ - c0_h·log λ] = c0_h/(b0_h + KL(s||r))`
with envelope cancellation `d/dD[envelope] = λ*` (sympy). Both are the model-fiber transcription
of the manuscript's belief-fiber results, parameterization-invariant in the `(mu, log-sigma)`
coordinates the code uses.

## Recommendation

Treat `(r, lambda_h)` as a single empirical-Bayes hyper-prior and finish it symmetrically with
the `alpha` machinery, while preserving the existing frozen-`r`, constant-`lambda_h` pure path as
the default. Concretely: add a `lambda_h_mode` registry in a new `vfe3/lambda_h_i.py` mirroring
`vfe3/alpha_i.py`, with `constant` (today's behavior, the default), `state_dependent` returning
`(c0_h/(b0_h+KL(s||r)), R_h)`, and `learnable` returning `(exp(log_lambda_h), 0)` as a
documented NN-exception sibling of `log_alpha`/`log_lambda_beta`. Thread the chosen weight and its
`R_h` through both `_hyper_prior_term` (the `s_e_step=False` scored path) and the `_refine_s` `s`
E-step (the `s_e_step=True` path), replacing the hardcoded `alpha_mode='constant'`,
`value=cfg.lambda_h` at `model.py:429`. Add `b0_h`/`c0_h` config fields and the
`learnable_r`-style optimizer group for `log_lambda_h`. Separately, offer the closed-form
forward-KL barycenter as the pure-path update for `r` in the scored `s_e_step=False` regime, and
keep AdamW `learnable_r` for the coupled `s_e_step=True` regime where the barycenter is not the
exact argmin.

Do not classify `learnable_r` as a neural network or add it to the CLAUDE.md NN-exception list. A
one-line positive note in CLAUDE.md is warranted: `learnable_r` un-freezes the hyper-prior centroid
as an empirical-Bayes M-step prior (like the embedding tables), the top-scale boundary-condition
stand-in for the deferred meta-agent transport.

## Scope and honest caveats

The state-dependent `lambda_h` and the closed-form `r` barycenter are principled extensions
transcribed from the manuscript's `alpha` derivation and its meta-agent barycenter, not results
the manuscript derives for the model fiber; documentation must say so and must not cite the
manuscript as deriving them. The closed-form barycenter is the exact M-step only in the scored
`s_e_step=False` regime; under the active coupled config it is not. The state-dependent
`lambda_h` requires the `R_h` regularizer in `F` and the `s` E-step or the envelope is wrong. The
manuscript's truly pure `r` remains the deferred top-down `r_i = Ω̃[s_I^{(s+1)}]`; both
`learnable_r` and a closed-form barycenter are same-scale empirical-Bayes stand-ins for it, which
the manuscript sanctions as the training-time boundary condition.

## Implementation

Delivered on this branch, all defaults byte-identical to the prior behavior. New registry
`vfe3/lambda_h_i.py` (`hyper_prior_lambda_h`, modes `constant` / `state_dependent` / `learnable`)
delegates to `vfe3/alpha_i.py` so the envelope `lambda_h* = c0_h/(b0_h+KL)` and `R_h` share alpha's
single verified implementation. Config gains `lambda_h_mode` (default `constant`), `b0_h`/`c0_h`
(default `1.0`), and `r_update_mode` (default `gradient`), each registry-validated, with inert-toggle
warnings and `log_lambda_h` added to the two E-step gradient-flow freeze warnings. In `model.py`,
`lambda_h_mode='learnable'` builds `log_lambda_h` (init `log(cfg.lambda_h)`, so step-0 byte-identical
to constant) with a detach footgun; `_refine_s` routes the `s` E-step self-coupling through
`cfg.lambda_h_mode` with `b0_h`/`c0_h`/`log_lambda_h` (the `e_step` already adds `alpha_reg = R_h` for
non-constant modes, so the envelope cancellation transfers automatically); a new
`_hyper_prior_weighted` applies the registry weight plus `R_h` to the raw per-token KL, the scored
caller drops its external `lambda_h` factor, and diagnostics fold the registry-weighted contribution
into `total` (raw KL preserved for observability). The closed-form M-step lives in
`PriorBank.barycenter_r_()` (moment-matched centroid of the `s` tables, under `no_grad`); under
`r_update_mode='barycenter'` `r` is constructed `requires_grad=False` (out of the optimizer) and set
each M-step from `train_step` after `optimizer.step()`. `build_optimizer` groups `log_lambda_h` at
`m_mu_lr`.

Verified: smoke checks (constant linear oracle, `state_dependent` term equal to the envelope
oracle, `learnable` init/grad/grouping, exact barycenter) plus the test suites — `test_hyperprior.py`
28 passed (12 new), and the config/alpha/gamma/model/train/live-s batches green. The pure default
path (`lambda_h_mode='constant'`, `r_update_mode='gradient'`, `learnable_r=False`) is unchanged.
