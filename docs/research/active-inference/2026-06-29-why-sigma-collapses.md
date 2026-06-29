# Why the Belief Covariance Collapses (2026-06-29)

Investigation into why the belief-covariance trace `tr(Sigma_q)` is near-constant (coefficient of
variation 0.0445) and anti-correlated with realized cross-entropy (Spearman -0.137) at the
sigma-validation gate's operating point. Companion to the gate result in
`docs/research/active-inference/2026-06-28-sigma-gate-prereg.md`. Produced by a multi-agent workflow
(run `wf_b7859da6-2bd`): an operating-point ground-truth pass, five domain specialists each testing
one mechanism against the live code, then an adversarial skeptic on every survivor. Findings below are
the verified synthesis; every claim carries a path:line from the live code, and several were checked
with executed forward probes on the actual checkpoint.

## Operating point (corrected)

The gate's `wikitext103_ed20_15k` artifact was measured on
`vfe3_runs/162.08_wikitext-103_K20_block_glk_s54/checkpoints/step_15000.pt`, whose stored config is
`use_prior_bank=True` (the KL-to-prior diagonal decode, so `sigma_q` does enter the logits),
`use_head_mixer=False`, `n_layers=1`, `embed_dim=20`. The `153.89...linear` path still sitting in the
`sigma_gate_measure.py` WIP config is the linear-decode sibling (`use_prior_bank=False`, which discards
`sigma_q` at readout) and is not the measured run. The relevant E-step covariance knobs in the stored
config are `e_q_sigma_lr=0.001` (config default 0.015), `n_e_steps=1`, `e_sigma_q_trust=10.0` (default
5.0), `sigma_max=100.0` (default 10.0), `e_phi_lr=0.0`, `spd_retract_mode='spd_affine'`,
`family='gaussian_diagonal'`, `divergence_family='renyi'` at `renyi_order=1.0` (exact KL),
`gradient_mode='filtering'`, `prior_source='model_channel'`, `s_e_step=True`.

## Why tr(Sigma_q) is near-constant

Three reinforcing causes, all on the live path; a fourth candidate (the retraction throttle) is refuted.

The dominant cause is that the belief covariance is anchored to a per-token prior table that is itself
near-constant. With `prior_source='model_channel'` and `s_e_step=True`, the belief is initialized to a
per-vocabulary log-variance table `s_sigma_log_embed` of shape `(V=50257, K=20)`
(`prior_bank.py:666`, table init `log(sigma_init)` with `sigma_init=3` at `prior_bank.py:153`), then
the s-channel E-step refines it to `s_sigma1` and installs that as both the belief's starting `sigma`
and its self-coupling anchor `sigma_p` (`model.py:730-731`, forwarded frozen across the single block at
`model.py:743`, `stack.py:70,78`). The s-refine pulls every token toward a single shared `(K,)`
hyper-prior centroid `r_sigma` broadcast over all tokens (`prior_bank.py:211-212`, `model.py:570-571`).
A forward probe on the checkpoint measured the raw table trace at mean 35.4, cv 0.081 across the vocab,
compressed by the shared-`r` shrinkage to the level seen in the belief; on the held-out (frequency-
realistic) token stream the trace compresses further to the gate's cv 0.0445, because common tokens
dominate and cluster. The belief tracks this per-token prior table at Pearson 0.99 and sits within 2.6
percent of it. So both the level and the small token-to-token spread of `tr(Sigma_q)` are inherited from
a learned prior table that is near-constant by construction.

The belief E-step has no force capable of re-dispersing it, because the live belief-covariance gradient
carries no per-token data precision. At this config `uses_kernel_route` is true, so `belief_gradients`
dispatches to the hand kernel (`kernels.py:193-202,291`), whose `grad_sigma` is exactly the
self-coupling term plus the belief-coupling pair term and nothing else,
`grad_sigma = self_mask*alpha*0.5*(1/sp - 1/sq) + lambda_beta*sum_j beta_ij*0.5*(1/st_ij - 1/sq)`
(`kernels.py:134-136`). There is no `1/sigma_data` channel anywhere in it. The canonical observation
term `-E_q[log p(o|x)]` exists only as `free_energy()`'s optional `log_likelihood` argument, subtracted
under an `if log_likelihood is not None` guard (`free_energy.py:341,401-402`), and `free_energy()` is
never on the descent path (the kernel does not call it, the autograd oracle has no likelihood term, and
no caller ever passes a non-`None` `log_likelihood`). It is a gated stub. Consequently the only forces
on `Sigma_q` are the KL-to-prior self-coupling, which is identically zero at the start of the step
because the belief is initialized to its own anchor (`sq == sp`, so `1/sp - 1/sq = 0`), and the small
belief-coupling pair term. With a single E-step iteration at `e_q_sigma_lr=0.001` the belief never
leaves the immediate neighborhood of the prior table.

The Fisher/SPD geometry reinforces the same direction. The diagonal-Gaussian natural gradient is
`nat_sigma = 2 sigma^2 grad_sigma` (`retraction.py:373-376`, `gaussian.py:241-246`) and the SPD-affine
retraction is multiplicative (`retraction.py:128-133`), so the stationary point is `grad_sigma = 0`,
whose root is a precision-weighted convex average
`1/sigma_q* = [alpha/sp + lambda_beta sum_j beta_ij/st_ij] / [alpha + lambda_beta sum_j beta_ij]`. This
attractor is token-dependent (it moves with the per-token prior `sp` and the transported neighbors
`st`), but it contracts dispersion toward the shared prior/neighbor precision pool rather than expanding
it. The self-coupling weight is the state-dependent `alpha_i = 1/(1 + KL(q_i||p_i))` (`alpha_i.py:81-96`,
`b0=c0=1`), which weakens self-pull exactly where a belief drifts, so it does not re-disperse `sigma`
either.

The retraction throttle is not the cause. The diagonal SPD retraction clamps the tangent in whitened
(log) units after the learning rate, so the per-step log-change is bounded by `trust_region * step_size
= 10`, a multiplicative factor of `exp(+/-10) ~ 22026`, further bounded only by `[eps, sigma_max=100]`.
A single step can move `tr(Sigma)` from the `eps` floor (~1.5e-3) up to `K*sigma_max = 2000` — about six
orders of magnitude of permitted range, against the observed 1.18x end-to-end decile spread. The
constancy is upstream in the gradient and the anchor, not in the clamp. (The H4 retraction-pinning
hypothesis was refuted on these grounds.) One internal numerical disagreement is worth recording: a
specialist's toy probe reported the single step moving `sigma` by a relative 1.3e-4, while a retraction
probe on the checkpoint reported per-step moves of 0.11 to 6.25 nats depending on `grad_sigma`
magnitude. The reconciliation is that the per-forward move scales with the (small, pair-term-only)
gradient and is not by itself the load-bearing fact; the robust, measured fact is that the belief ends
within 2.6 percent of the per-token prior table, so the near-constancy is governed by that near-constant
anchor regardless of the exact step size.

## Why the residual variation is anti-correlated with cross-entropy

The sign is a confound carried by the belief mean `mu_q`, not a covariance-readout effect. In the
diagonal KL decode the logit is `-KL(q||pi_v)/tau`, and `sigma_q` enters a per-vocabulary logit only
through the precision-weighted trace `sum_k sigma_q_k/sigma_v_k` (`prior_bank.py:724-726`) plus a
`v`-independent log-determinant that cancels under softmax (`prior_bank.py:730`). The decisive test is a
counterfactual run on the checkpoint: replacing `sigma_q` with a per-dimension constant in the decode
(leaving `mu_q` intact) leaves the anti-correlation essentially unchanged, Spearman -0.141 to -0.133,
with the decile cross-entropy still falling monotonically. Since the relationship survives removing
`sigma` from the logits, the sign cannot be a `sigma`-readout artifact; roughly 94 percent of it is
borne by the Mahalanobis (mean) term, and `tr(Sigma_q)` merely co-varies with the same token/context
property that sets difficulty. The gate also reads the raw Euclidean trace `sum_k sigma_q_k`
(`metrics.py:283`, `extract.py:217`), a different SPD functional than the precision-weighted contraction
the decode actually responds to, so the gate is not even measuring the quantity the readout uses.

The co-varying property is token frequency and context redundancy: the residual `tr(Sigma_q)` weakly
tracks input-token frequency (Spearman +0.077) while cross-entropy tracks target predictability
strongly (Spearman -0.64), and the learned `s_sigma` table gives frequent or redundant-context tokens
slightly larger `sigma` while they also tend to precede easy, low-cross-entropy targets. Two honest
caveats attach. The specific frequency label is quantitatively under-powered (`0.077 * -0.64 ~ -0.05`
does not by itself close the observed -0.137), so the exact confounder is not fully pinned, though the
counterfactual establishes "confound, not readout artifact" independent of the label. And `sigma` is not
absolutely inert: holding it constant raises mean cross-entropy from 4.79 to 5.30 nats, and rescaling it
by 0.5x or 2x explodes cross-entropy to ~14 nats because `tau_eff = 0.073` makes the trace term
scale-sensitive. The covariance carries real absolute calibration; only its natural 4.6 percent spread
is too small to move the cross-entropy rank.

## Synthesis

The covariance cannot be an epistemic signal at this operating point because the architecture
structurally prevents it from being one. Two design facts conspire. First, there is no data or
likelihood term in the live belief-covariance gradient, so `Sigma_q` is never updated by per-token
evidence; it remains a near-copy of a learned prior table that the belief is initialized to and pinned
near by a single low-rate E-step. Second, that prior table is shrunk toward a single shared centroid, so
its token-to-token spread is already small. The net is a covariance that is near-constant by
construction and whose residual variation tracks token frequency, a difficulty proxy with the wrong sign
for epistemics. The model still produces sensible cross-entropy because the predictive content lives in
the belief mean `mu_q`, which is exactly why the anti-correlation is mean-borne. This is a perceptual
active-inference model whose mean beliefs do the work and whose covariance is, at this configuration, an
almost static learned prior rather than a live uncertainty estimate.

## What would be required to make sigma epistemic

These are the levers the investigation implies, listed because the user asked whether `sigma` could be
made informative; they are options to evaluate, not a prescription, and the gate is reusable to re-test
after each.

The single largest lever is to activate a data/likelihood precision term in the belief-covariance
E-step, wiring the canonical `-E_q[log p(o|x)]` term (currently a dead stub at `free_energy.py:341,401`)
into the live `grad_sigma` so the posterior precision contracts where the observation is informative and
stays wide where it is not. This is the canonical free-energy term, not a neural-network addition, but it
is a genuine design change and is absent from the current kernel. Without a channel through which
evidence enters `Sigma_q`, no amount of tuning will make the covariance encode per-token uncertainty.

Secondary levers all concern letting the covariance move off its prior anchor: raising `e_q_sigma_lr`
(0.001 is roughly fifteen times below the config default) and/or `n_e_steps` so the belief is not pinned
to its initialization; loosening the shared-centroid shrinkage on the s-channel prior table, which
currently compresses a large fraction of the per-token spread; and not initializing the belief to its
own self-coupling anchor, so the self-coupling term is nonzero from the first step. Each of these widens
the achievable spread but, on its own, still leaves `Sigma_q` driven by prior and coupling rather than
by data, so they are complements to the likelihood term, not substitutes for it.

## Per-hypothesis verdicts

- H1 (no live data-precision term in `grad_sigma`): confirmed as a structural fact; its causal weight on
  the near-constancy is through "nothing re-disperses sigma," not through an unreached fixed point.
- H2 (prior anchor): confirmed and dominant for the near-constancy; the anchor is token-dependent, not a
  single constant, but is itself near-constant and the belief tracks it at Pearson 0.99.
- H3 (Fisher/SPD attractor): confirmed as a contributing, dispersion-contracting force; its internal
  "frozen at init" sub-claim is partly overstated (the E-step is materially active), but the
  near-constancy survives via the contracting attractor and the near-constant anchor.
- H4 (retraction/clamp pinning): refuted; the retraction permits ~six orders of magnitude of range and
  is far from binding.
- H5 (the anti-correlation sign): confirmed as a mean-borne confound, not a covariance-readout artifact,
  by a constant-sigma counterfactual; the exact confounding variable (frequency/redundancy) is not fully
  pinned quantitatively.
