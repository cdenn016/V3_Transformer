# Tier 3 investigation and the Tier 1 / Tier 2 status ledger

Date: 2026-07-26
Branch: `investigate/2026-07-26-tier3-whitepaper-ideas` off `main` @ `5b15e0a`
Source under investigation: `docs/2026-07-25-vfe4-performance-hypotheses.md`, section 5
Manuscripts read: the vault WIPs at `Research/manuscripts/` (vfe4 chapters 04-08, magent 06-08 and
10-12, `PIFB2.tex`, `verified-ledger.md`)

Eight agents worked this in parallel: one expert per Tier 3 item, two status auditors over Tier 1 and
Tier 2, and one reading the whitepapers fresh for what the hypothesis document missed. Nothing was
edited outside `docs/`. No model was run at production scale. Every claim below carries a `path:line`
or an artifact path; where an agent could not produce one, the claim is marked unverified rather
than repeated.

A note on the numbering, since it has caused confusion. The source document has no "Tier 0". Its
section 7 defines **Stage 0** as the free checkpoint-only measurement set, and those measurements
were run and are recorded as M1 through M6 in
`docs/2026-07-25-phi-bound-calibration-and-stage0-report.md`. What the document labels Tier 1 is
section 3 (items 1.1 through 1.5), Tier 2 is section 4 (items 2.1 through 2.5), and Tier 3 is
section 5. The internal numbering is inconsistent in the source and has been normalized here.

## 1. The headline

Four of the five Tier 3 items are dead, and two of them died for the same reason: the hypothesis
document attributed to the VFE 4.0 whitepapers a prescription the whitepapers did not make. In one
case it implemented a lower bound that the manuscript introduced as a warning; in another it quoted
a coordinate update while omitting the terms that make it a coordinate, then proposed a fix carrying
the identical target leak the same document rejects three paragraphs later. This is the most
important structural finding of the sweep, because it means the source document's whitepaper
grounding cannot be taken on trust; every remaining citation in it deserves the same check.

Tier 1 and Tier 2 are substantially closed. Of ten items, four are measured-refuted, two are already
set in the working configuration, and one is confirmed but unaddressed. Only two Tier 2 items remain
genuinely live, and only one Tier 1 item still needs code.

The compensation is that reading the whitepapers fresh produced four candidates that were never in
the hypothesis document at all, three of them aimed at channels the existing measurements price
higher than anything Tier 3 targeted.

## 2. Tier 3 verdicts

| item | verdict | basis |
|---|---|---|
| 5.1 additive predictive-kernel reference covariance | KILL as specified | cited equation is a lower bound, not a prescription |
| 5.2 recalibrate `b0`/`c0` for state-dependent alpha | KILL | bitwise no-op; proven by execution |
| 5.3 CG solve of the joint precision | KILL | `J` is nonsymmetric and nilpotent; the experiment is already done |
| 5.4 metric-aware gauge M-step | PARK the switch, PURSUE the pieces | mode is config-rejected at production scale |
| 5.5 schedule and batch hygiene | PARTIALLY DONE, remainder inside noise | three of seven already set; each survivor under the floor |

### 5.1 Additive predictive-kernel reference covariance

`edge_reference_floor` was never built; the string occurs exactly once in the repository, in the
hypothesis text itself at `docs/2026-07-25-vfe4-performance-hypotheses.md:321`. The algebra in the
proposal is correct, and sympy confirmed all three derivative claims against the energy the code
actually computes at `vfe3/gradients/pairwise_stats.py:99-118`: `dE/dS = (S - sigma_q - Delta^2)/(2S^2)`,
`dA/dS = 1/(2R)`, and monotonicity iff `R >= sigma_q + Delta^2`.

What fails is the attribution. `vfe4_whitepaper/06_elbo_coordinate_updates.tex:702-707` derives the
predictive-kernel form as a lower bound by convexity, `:699` states that mean-field "does not replace
that kernel by the transported recognition marginal," and `:721` warns that "Even this arithmetic
predictive mixture is not the noiseless transported sender law in general." Since the term enters F
with a positive sign, substituting it yields `F' <= F`, which is no longer an upper bound on
`-log p(o)`. The `dA/dS = 1/(2R)` result the proposal leans on comes from a different equation at
`:664-681`.

Two further blockers. A constant `R` breaks even the residual diagonal-gauge invariance that survives
under `block_glk`: executed at `g = diag(d)` per `vfe3/geometry/groups.py:404-410`, the deviation is
2.4e-7 at `R=0` against 0.336 at `R=1.0`, so `R = c*I` is no rescue. And the whitepaper's own
compatibility proposition at `:799-813` requires the sender covariance fixed rather than variationally
updated, which `skip_belief_sigma_update=True` (`train_vfe3.py:438`,
`vfe3/inference/e_step.py:1130-1132`) already delivers, at which point `A = D_V3 + d/2` and the
correction is a row constant the softmax cancels.

One correction propagates out of this item and matters elsewhere. The widely repeated claim that
`estep_grad_norm_sigma` is exactly 0.0 in all 180 rows is a tautology, not a measurement:
`vfe3/gradients/kernels.py:184-187` returns `None` and `vfe3/inference/e_step.py:1076-1077` writes a
literal zero. A K=4 probe measured `s_sigma_log_embed` gradient norm at 4.94e-4, live through beta.
Sigma is less inert than the corpus asserts, which raises rather than lowers the value of item 2.5.

### 5.2 Recalibrating `b0` and `c0`

Not built; `git log --since=2026-07-24 -- vfe3/alpha_i.py` is empty and both constants remain 1.0 at
`vfe3/config.py:288-289`. The proposal is a bitwise no-op, and this was established by execution
rather than argument. `vfe3/model/model.py:1120` calls `vfe_stack(beliefs, beliefs.mu, beliefs.sigma, ...)`,
so the initial belief is the prior and `D = 0` exactly where both kernels evaluate alpha
(`vfe3/gradients/kernels.py:444-445` on the gradient route, `:622-623` under `mm_exact`). At `D = 0`
the formula reduces to `alpha* = c0/b0`, which `b0 = c0 = 1e-3` preserves exactly. A K=3 probe
returned `max|mu* diff| = 0.0`, `max|sigma* diff| = 0.0`, and `max|grad_mu diff| = 0.0`. With
`n_e_steps = 1`, which all fifteen run configurations and `train_vfe3.py:92` set, there is no second
iteration in which `D` could become nonzero.

The theory came back against the proposal as well. `belief_inertia.tex:239` states that the canonical
objective uses unit prior weight and that `c0/(b0 + D)` is an explicit extension at
`eq:state_dependent_alpha`; `vfe4_whitepaper/06_elbo_coordinate_updates.tex:312` states that
reweighting a KL complexity term "is not the ELBO." Grepping `alpha` across the four whitepaper
sections the hypothesis document pointed at returns zero hits, and the "per-coordinate anchor" the
document invokes is an invented phrase, the manuscript object being the per-position
`D_i = KL(q_i||p_i)`.

Sympy additionally recovered the profiled self-coupling sector as the log barrier
`c0[1 + log((b0 + D)/c0)]`, predicting 300.096 at `b0 = c0 = 1, K = 300` against the measured
`[300.044, 300.609]`. That closes the question of why 63.8% of the logged free energy is constant: it
is the barrier evaluated at `D = 0`, not a defect. Setting `b0 = 1e-3` would also cut the mean-sector
convexity margin from 3115x to 3.1x, since convexity fails at `D = b0` independent of `c0`.

A side finding: the calibration warning at `vfe3/config.py:1743-1756` is unreachable for the shipped
form, because `config.py:1728-1735` rejects the only divergence family satisfying its gate.

### 5.3 Solving the joint precision

Not built. The implicit system is real and was reconstructed from `vfe3/gradients/kernels.py:618-620,673,680-690`,
with the replica reproducing `mm_exact_update`'s `mu_star` bit-identically and the assembled `J`, `b`
reproducing one sweep to 4.44e-16. Conjugate gradients cannot be applied to it. `Omega_ji = Omega_ij^{-1}`
rather than `Omega_ij^T` (`vfe3/geometry/transport.py:2182-2183`) and beta is a row softmax
(`vfe3/free_energy.py:329-332`), so `J` is nonsymmetric with measured relative asymmetry between 0.16
and 0.46 across a four-seed grid, while CG requires a Hermitian positive-definite operator.

The decisive fact is structural. The causal mask at `vfe3/attention_prior.py:212-213` makes `J`
strictly block-lower-triangular, with measured strict-upper mass 1.6e-15, so the Jacobi iteration
matrix is nilpotent: `rho(M) = 5.4e-15` and `||M^N|| = 1.4e-73`. Convergence is finite rather than
geometric, and with `cond_2(J) = 6.03` and `cond_2(D^{-1}J) = 1.78` there is nothing for a Krylov
method to accelerate. One sweep already solves the system to 1.3% relative error and four sweeps to
9.1e-6.

The proposal is also directionally wrong. The exact frozen-beta solve is a worse point of the
canonical free energy than the depth-8 iterate at every pair share tested, capturing 4.5% of the F
that the live-beta iteration captures at the shipped 0.196. The whitepaper predicts precisely this at
`06_elbo_coordinate_updates.tex:855`, where V3's frozen-template row is described as "a plug-in best
response or majorization surrogate, not coordinate ascent on one fixed global ELBO"; the manuscript's
`J` is symmetric only because it carries the recoil blocks at `:726-747` that `gradient_mode='filtering'`
structurally omits.

The experiment has in any case already been run. Running the E-step to depth 8 is an eight-sweep
Jacobi solve, and `docs/2026-07-26-b01-remeasurement.json` shows the K=300 displacement saturating by
depth 5 at a cost of +0.013 nats of cross-entropy, a cost with the wrong sign. The one free lever the
item surfaces is `mm_damping`, which is 0.75 in both baselines; undamping to 1.0 closes roughly half
the gap to the exact solve at zero compute.

### 5.4 Metric-aware gauge M-step

The live-versus-dead contradiction between the hypothesis document's two accounts resolves against
the Tier 3 bullet. Under `m_phi_update_mode='adamw'`, `phi_mstep_max_matrix_norm` is live:
`vfe3/train.py:746-750` gates `project_phi_parameter_rows_` on the mode being `adamw` and the field
being non-None, and 150 rows of projection telemetry in `vfe3_runs/138.40_.../metrics.csv` confirm it
fires, with `phi_chart_projected_rows` averaging 364 of 50385 and `phi_chart_preproject_max` at 5.018
against a bound of 5.0, costing 0.43 ms of an 85.7 ms step. The bound-13-at-K=300 arm therefore needs
no mode switch. `phi_precond_mode` and `m_phi_group_trust_radius` genuinely are dead
(`vfe3/train.py:365,380`), because `e_phi_lr = 0`. Under `pullback_group` the field changes meaning
entirely: projection is off and the value becomes a fail-closed `FloatingPointError` abort
(`vfe3/gauge_optim.py:297-302`, uncaught at `vfe3/train.py:730`).

Three new defects fell out. The mode is config-rejected at production scale, since
`vfe3/config.py:1976-1980` requires `d_head <= 12` and the 452M-parameter run has 30, so the
proposal's payoff sits on a configuration the mode refuses to build. The trust radius, the chart
bound, and the currently live projection all reduce the Frobenius norm jointly over the `GL(d)^H`
factors (`vfe3/gauge_optim.py:112,264,288,414`), which is the same defect the source document
identified in the transport clamp, and which means a bound of 13 at K=300 is really 4.11 per factor.
Measured cost on the 5090 at K=20 is 8.2 ms per active phi row, implying roughly 20 to 35 seconds per
step against the current 85.7 ms.

The gauge result is the substantive one and is worth recording independently of the proposal.
Executed at K=4 in fp64, the "natural direction" is identically the ambient Frobenius gradient on
`U = exp(phi)`, with residual 1.6e-6 at the finite-difference floor, which
`Research/manuscripts/verified-ledger.md:45` already states in as many words. It transforms by
congruence rather than by the adjoint, so it is exactly `O(d)^H`-equivariant and GL(K)-breaking,
while AdamW is non-equivariant even under `O(K)`. Sympy then showed that the Ad-invariant forms on
`gl(2,R)` are two-dimensional and none is positive definite, so no exactly GL(K)-equivariant metric
M-step exists at all. The manuscripts are silent on the gauge M-step.

The weight-decay half of the proposal should be decoupled. `phi_weight_decay` is live and independent
at `vfe3/train.py:217,259,269`, and `vfe3/config.py:2974-2986` exists specifically to warn that
`pullback_group` overrides it. It is not currently zero: `train_vfe3.py:355` and all seventeen run
configurations read 0.03.

### 5.5 Schedule and batch hygiene

All seven config fields exist and are live; there are no phantoms. But the bullet is not seven
proposals. Three are already done in the working configuration: `batch_size` is at 64, past the
proposed 32 (`train_vfe3.py:88`); the EMA lines are deleted so `vfe3/config.py:695`'s 0.999 default
already applies; and `grad_clip_per_role=True` is set at `train_vfe3.py:436` and in every run config.
A fourth, the `grad_clip` bullet, contains no proposal at all, merely the observation that a knob is
inert, which was confirmed and strengthened: the maximum `grad_norm` is 0.818455 at step 71000, and
across all seventeen K=20 runs the global maximum is 0.5805 with zero of 2576 logged rows above 1.0.
That is a 0.1% sample, since `vfe3/train.py:663` computes the norm only on logged steps.

`decode_unigram_prior` is the best-evidenced survivor and was reproduced rather than trusted:
regressing `output_proj_bias` on the corpus log-unigram over 116,840,318 tokens gives Pearson
`r = +0.9115`, `R^2 = 0.831`, slope 1.23 across 47,938 seen tokens. The bias did spend training
rediscovering log-unigram. The document's "max 15.47" is misstated; that is the minimum
(-15.4711), and the true maximum is +6.4361.

Two corrections to the supporting numbers. Peak memory is 17.94 GiB of 32, not 19.26 of 32, because
`vfe3/train.py:1587` divides by 1e6 and the logged figure is decimal megabytes. And the floored tail
was not wasted budget: validation perplexity got worse by 0.147 across it, so `min_lr_frac=0` is not
recovering a lost 6.4%.

Every surviving item is individually inside the noise floor, at roughly 0.15 PPL for `min_lr_frac`,
0.5 for the unigram prior, 0.1 for the z-loss and 0.3 for warmup. Only the bundle is testable, and
its summed 1.05 PPL clears the 1.5-wide band marginally at best. `use_ema` was excluded on the
grounds that its 2.116 GB shadow conflicts with a larger batch and that it substitutes against
`min_lr_frac=0`; `batch_size` was excluded because it doubles wall clock from 18.7 to 37 hours, with
`grad_accum_steps=2` the cheaper route.

## 3. Tier 1 status ledger

| item | status | basis |
|---|---|---|
| 1.1 transport clamp | MEASURED-REFUTED | M2 sweep is a minimum at 20 |
| 1.1b per-block versus joint reduction | code fact CONFIRMED, lever REFUTED, one number unsupported | `transport.py:1930`, `:1942` |
| 1.2 `matrix_exp` in fp64 | mechanism CONFIRMED-UNADDRESSED, payoff UNTESTED | `transport.py:1967-1968`, `:1746` |
| 1.3 BCH composition | SET-IN-CONFIG, performance premise REFUTED | `train_vfe3.py:249`, commit `f279ceb` |
| 1.4 weight decay on rare rows | MEASURED-CONFIRMED-BUT-UNADDRESSED | `train_vfe3.py:355-356`, `train.py:275` |
| 1.5 diagonal covariance | MEASURED-REFUTED as a defect, framing SUPERSEDED | `docs/2026-07-25-edits.md:161-171` |

The clamp sweep on trained weights is a clean minimum at 20: `max_norm` 10 gives PPL 99.393, 20 gives
55.275, 40 gives 67.000, 60 gives 78.431, and 1e6 raises `FloatingPointError`
(`docs/2026-07-25-phi-bound-calibration-and-stage0-report.md:153-159`). Both accounts agree the clamp
substitutes a different operator; they differ on the consequence, and the measurement settles it,
because the trained weights are a fixed point of the surrogate.

The most useful new fact in this section is operational. At the live configuration the clamp never
fires at all: `phi_exp_clamp_frac` has maximum 0.0000 and `phi_matrix_norm_max` is exactly 5.000
across all seventeen runs in `vfe3_runs/`, because `phi_mstep_max_matrix_norm=5` binds first. That
retires the whole 1.1 thread including the per-block sub-finding, which remains a genuine code fact
at `vfe3/geometry/transport.py:1930` (deliberate, for blocked and unblocked bit-equivalence, and
pinned by `test_per_block_exp_is_bit_equivalent_to_full_exp`) but is unreachable under the bound
actually in use. The sub-finding's claim that a per-block check "would have fired on 0 of 10 blocks"
is unsupported: `phi_matrix_norm_*` is the joint full-K norm (`vfe3/model/model.py:3025-3027`), so a
p95 of 83.92 is about 26.5 per block, which is above 20, not below it.

On 1.5, the arithmetic sub-claim is confirmed and amplified, with an exact-to-truncated median ratio
of 7.2x, p99 of 426x, and a maximum of 10912 nats against `kl_max=160`. But both proposed remedies
were built and trained and are much worse: exact congruence was abandoned at train PPL 432 against
139, and frame-intrinsic went 139.3 to 308.1 with `phi_embed.grad is None`. The truncation is
load-bearing regularization for a non-compact gauge, which is the superseding framing.

What remains actionable in Tier 1 is thin. The `s_mu_embed` zero-decay group at `vfe3/train.py:275`
is the only item with no config route and needs a one-line parameter-group exemption. The
`exp_fp64_mode='norm'` A/B is config-only but inert at the working configuration's `d_head=10`, since
the fp64 trigger is `exp_dim = max(block_dims)` at `vfe3/geometry/transport.py:1746`. Training under
the per-block reduction would first need a per-block norm metric that does not exist.

## 4. Tier 2 status ledger

| item | status | basis |
|---|---|---|
| 2.1 frame-intrinsic covariance | MEASURED-REFUTED | 139.3 to 308.1 val PPL, phi frozen |
| 2.2 attention temperature | PARTIALLY DONE | M1 sweep refuted; `learnable_kappa_beta` built, 0 of 19 runs |
| 2.3 anchored depth | PARTIALLY DONE | the one 2-layer arm ran at `prior_handoff_rho=1` |
| 2.4 gauge table re-budget | PARTIALLY DONE | premise refuted by M3; `tied_block_glk` registered, 0 of 19 runs |
| 2.5 sigma | BUILT-NOT-MEASURED | every toggle wired, 0 of 19 runs set either half |

Item 2.1 was the source document's "strongest single proposal" and it is dead, with a mechanism worth
stating because it generalizes. In Regime I the transport is a pure coboundary `Omega_ij = U_i U_j^{-1}`.
The frame-intrinsic family stores `a_i = U_i^{-1} mu_i` and `Sigma_i = U_i diag(sigma_i) U_i^T`, so
reading the transported key in the receiver frame annihilates both frames: `U_i^{-1} Omega_ij mu_j = a_j`
and `Omega_ij Sigma_j Omega_ij^T = U_i diag(sigma_j) U_i^T`. By invariance under a common invertible
pushforward the edge energy collapses to a diagonal divergence containing no `U` at all. The code
does this literally, both seams at `vfe3/families/frame_gaussian.py:136-169` being `expand` views that
never read `omega`. Since `phi_embed` reaches the loss only through `U` and `e_phi_lr=0` supplies no
second path, the graph holds no node depending on it, so `.grad` is `None` rather than zero and the
452M-parameter table freezes at initialization (`vertex_cond_median` 1.65 against 47.3). The
proposal's 7.1e-15 accuracy and 17.1x speedup claims both stand; they describe the exact energy of a
different model, one with the gauge deleted from the coupling. The identical signature appears
independently at `docs/audit-results.md:84-93` for Regime II, where the energy is bit-identical
across all four modes and `d loss / d connection_W` is `None`.

The refutation was derivable in advance from the very equation the source document quoted. Applying
`G = U_i^{-1}` to both arguments of `KL(q_i || Omega_ij q_j)` under `eq:entropy-shift-kl-invariance`
(`vfe4_whitepaper/05_structured_information_form.tex:368`) is mathematically sound, but the same
common pushforward that closes the family also cancels the coboundary. That is worth recording
because the same equation will be reached for again.

Item 2.3's motivating evidence has been substantially withdrawn. The depth pathology was traced to a
shared config field, and once the belief loop is decoupled it contributes only +0.005 to +0.013 nats
across depths 2 through 8. The single 2-layer arm on disk, `139.97_2-layer`, is a genuine single-field
arm at +1.57 PPL but ran at `prior_handoff_rho=1`, so the anchored variant the proposal actually asks
for has never been tried. Item 2.4's premise is refuted by M3 and M5, which together show the gauge
table is load-bearing (+1.916 nats to zero it) but precision-redundant and rank-saturated, closing
the dimensional cuts; `tied_block_glk` is registered at `vfe3/geometry/groups.py:311-344` and gets
re-motivated on entirely different grounds, namely exact head-mixer equivariance, where the mixer is
worth a measured 9.85 PPL while breaking it.

That leaves 2.5 as the strongest surviving item across both tiers: fully built, pure path is the
default, and zero of nineteen runs have set either half of the pair.

## 5. New candidates from a fresh reading of the whitepapers

None of these appears in the source document's Tier 1, 2, 3, or section 6, and none is refuted by the
later measurements. Every magnitude is honestly unknown; given the documented base rate, the prior on
any of them clearing the noise floor should be low.

**The exact V4 edge cost uses a fixed receiver covariance plus an additive sender-uncertainty trace.**
`eq:mean-field-model-edge-kl-plus-uncertainty` at `vfe4_whitepaper/06_elbo_coordinate_updates.tex:697`
gives the exact cost as `KL(q_t || N(Omega mu_j + c, R_t)) + (1/2) tr(P_t Omega Sigma_j Omega^T)`,
with `R_t` the receiver covariance, declared SPD and fixed by the generative model at
`04_generative_model.tex:106-111`. V3 instead uses the live transported sender variance as the
reference precision, `1/diag(Omega Sigma Omega^T)` at `vfe3/gradients/kernels.py:682-684` via
`vfe3/geometry/transport.py:2584`, which scales like `cond(Omega)^2`. Under the whitepaper's exact
form the reference precision is constant across the row, the transported sender covariance enters
only linearly through the trace, and no inversion of a badly conditioned congruence occurs anywhere,
so the `cond^2` mechanism that `docs/2026-07-25-state-of-knowledge.md:29-35` identifies as the reason
diagonal truncation is load-bearing structurally cannot arise. This is a different object from Tier 3
item 5.1, which implements the lower bound. The seam is partial: `register_family` at
`vfe3/families/base.py:411` is the registered extension point, but `mm_exact_update` hard-gates on
`family="gaussian_diagonal"` at `vfe3/gradients/kernels.py:610-617` and raises rather than falling
through, so a new family runs on the autograd oracle unless a kernel is registered too. Effort M.

**The categorical emission belongs in the belief's Markov blanket, and it is prefix-safe.**
`eq:state-model-markov-blanket-potentials` at `06_elbo_coordinate_updates.tex:406` puts
`log L(x_s | z_s, m_s)` inside the belief potential, and `:422` states that "an observation-conditioned
update of either current continuous latent includes the emission." It uses the current token, not the
next, so it is prefix-safe by construction. PIFB2's canonical functional carries the same term at
`PIFB2.tex:690`. V3's `log_likelihood` is a gated stub with no production caller
(`vfe3/free_energy.py:399`, `:419-424`), independently confirmed at `docs/audit-results.md:864-866`.
The source document dismisses this at `:373-377` on the grounds that "the encode prior already is"
the `x_t` factor, which conflates a Gaussian pull in latent space with a categorical factor through
the shared output table; only the second ties the belief's inference objective to the decoder that
scores the prediction. That distinction is exactly the root cause the stage 0 report diagnoses for
the weak content channel at `docs/2026-07-25-phi-bound-calibration-and-stage0-report.md:399-404`. The
manuscripts anticipate the nonconjugacy and name the admissible routes at `:777`, and a declared
Böhning-type quadratic bound gives a constant curvature computable once per optimizer step. This is
the only candidate whose ceiling is not bounded by M6's 0.210 nats, because that number prices the
content channel under an objective with no data term. Effort M.

**The source prior may be content-dependent as long as it is prefix-measurable.**
`eq:prefix-measurable-content-source-prior` at `vfe4_whitepaper/07_transformer_crosswalk.tex:91`
explicitly licenses a learned, target-blind content-dependent prior, with the obligation at
`04_generative_model.tex:37` that it changes the generative joint and requires a fresh normalization
audit. Every registered V3 prior is a function of `(n_query, n_key)` only
(`vfe3/attention_prior.py:75-354`), so no token content can enter. M6 measured the positional prior
carrying 0.612 nats against the content energy's 0.210, meaning the dominant router is entirely
content-blind, and the whitepaper's answer is to make the prior the content-dependent object rather
than to strengthen the energy. The seam is better than it looks: `_effective_beta_log_prior` at
`vfe3/model/model.py:2386-2418` already receives token IDs and already produces batch-dependent
priors through two existing folds, so a new fold slots in without touching the ten call sites. A
learned content score would be a raw `nn.Parameter` in the same documented-exception class as
`t5_learnable_bias`, touching no gauge transport, and should be default OFF with the fixed positional
prior remaining the pure path. Effort M.

**The emission reads both channels; V3's decode reads only the belief mean.**
`eq:gauge-compatible-categorical-emission` at `04_generative_model.tex:213` gives
`l_t = W^z z_t + W^m m_t + d_t`, two dual readout maps. V3 computes `logits = mu_q @ W^T + b` at
`vfe3/model/prior_bank.py:1901-1903` with sigma marked discarded at `:1887`, so `W^m m_t` is absent
and the model channel reaches the logits only through the belief's prior, which is a measured 80/20
convex blend. This matters because the two channels contribute 1.343 and 1.854 nats individually but
only 1.947 jointly, and because the model channel takes context displacement from 0.000 to 0.812
while the belief E-step raises it only to 0.872. The channel carrying the context has no direct path
to the logits. This is the one candidate that cannot be added by registration alone, since the decode
seam signature `(pb, mu_q, sigma_q, tau_eff)` has no `s` argument. Cost is one `(V, K)` table, 15.1M
parameters at K=300. Effort M.

Three further gaps were recorded but are not recommended without more evidence: the untried third
branch of the nonclosure trichotomy (an orthogonal group with isotropic per-block covariance, blocked
on the `so_n`/`sp_n` validator bug and costing the whole non-compact richness that M3 prices at 1.9
nats); the absence of any return path from the belief residual to the model channel
(`vfe3/model/model.py:1085` has `_refine_s` output replace the belief, making the two channels a
feed-forward chain and supplying a mechanism for the measured redundancy); and the observation that
`mass_phi`, whose gradient reaches only the rows in the batch, is the whitepaper-compliant regulator
for the noncompact frame direction in a way that decoupled AdamW weight decay is not, so
`mass_phi > 0` paired with `phi_weight_decay = 0` is the correct form of the 1.4 fix rather than the
weight-decay exemption alone. That last one is blocked on D-08.

## 6. Where the source document misrepresents the whitepapers

All six equation labels cited in `docs/2026-07-25-vfe4-performance-hypotheses.md` exist, as does
`eq:free_energy_functional_final` cited in `CLAUDE.md`. Two citations are substantively misused.

`eq:gaussian-cavi-mean-update` exists at `magent_elbo_whitepaper/06_mean_field_theory.tex:224`, but
the claim at `:332-337` that `kernels.py:551-553` is "literally" that equation fails three ways. In
the manuscript, `J` is the exact posterior precision of a fixed normalized joint, whereas V3's fused
precision is assembled from live, detached, beta-weighted, mask-gated pair terms with no fixed joint
behind them, which `07_transformer_crosswalk.tex:147` states about V3 by name: the pair divergence
"is an engineered consensus energy with useful routing semantics, not a fixed generative transition."
Read to the end, the same chapter says a receiving-only update is not CAVI (`06_mean_field_theory.tex:135,168`),
so the equation cited says the opposite of what it is used to argue. And the proposed CG solve on a
symmetric `J` introduces future-to-past edges under causal masking, making `mu_i` depend on `x_{>i}`,
which is the identical leak class the same document rejects at `:373-377`. Two opposite verdicts on
one leak, in one document.

`eq:mean-field-predictive-kernel-covariances` exists at `06_elbo_coordinate_updates.tex:719` but is
presented as a prescription when the manuscript introduces it to state a lower bound. The exact
prescription is the fixed `R` plus additive trace described above, which is both cheaper and
structurally different.

Two further citations are accurate but under-read. The `eq:mean-field-source-row` gloss "only the
value of tau is off the unit scale" is too narrow, since the unit scale applies to the pair
`(A_j, tau)` and V4's `A_j` is the fixed-`R` transition cost rather than V3's moving-peer KL; the
`tau=1` direction is closed by measurement anyway, at +0.0266 nats, but for the reason the fuller
statement predicts. And `eq:diagonal-covariance-nonclosure-example` is correctly cited while omitting
the normative sentence two lines later at `:390` offering three branches, of which the document
presents branch 2 as the strongest proposal without recording that branches exist at all.

Separately, the document's supersession banner at `:13-23` remains wrong about 1.5 and 2.1, which
`docs/2026-07-26-findings-triage.md` item 11 already flagged and which two independent agents
confirmed here. Many line citations in sections 3 through 5 have drifted; that is ordinary drift from
three weeks of edits, not error.

## 7. New defects surfaced

The joint-versus-per-factor Frobenius reduction in `vfe3/gauge_optim.py:112,264,288,414` affects the
trust radius, the chart bound, and the currently live projection, so a stated bound of `R` is really
`R/sqrt(H)` per factor. This is the same defect class as the transport clamp and is live under the
default `adamw` mode, unlike the clamp version.

`m_phi_update_mode='pullback_group'` is unbuildable at production scale because
`vfe3/config.py:1976-1980` requires `d_head <= 12`.

`vfe3/config.py:1743-1756`'s alpha calibration warning is unreachable, because `config.py:1728-1735`
rejects the only divergence family satisfying its gate.

`estep_grad_norm_sigma = 0.0` is hardcoded rather than measured
(`vfe3/gradients/kernels.py:184-187`, `vfe3/inference/e_step.py:1076-1077`), so every downstream
inference that sigma receives no gradient rests on a constant.

`docs/2026-07-26-findings-triage.md:308` cites a seed-0 replicate run that does not exist on disk.

## 8. What is actually worth doing

Ranked by evidence quality rather than by hoped-for magnitude.

The `s_mu_embed` zero-decay parameter group at `vfe3/train.py:275` is a one-line change closing the
only Tier 1 item with no config route, and it should be paired with `mass_phi > 0` rather than shipped
as a bare `phi_weight_decay = 0`, for the normalization reason in section 5. D-08 needs fixing first
so the logged F carries the term the phi substep descends.

Item 2.5's paired sigma change is fully built, has the pure path as its default, and has never been
measured. The tautology finding on `estep_grad_norm_sigma` strengthens rather than weakens the case.

`mm_damping` at 1.0 instead of 0.75 is free and closes about half the gap to the exact frozen-beta
solve, which is the only surviving fragment of Tier 3 item 5.3.

The Stage 1 hygiene bundle (`min_lr_frac=0`, `decode_unigram_prior=True`, `z_loss_weight=1e-4`, and
`warmup_steps=1000` only when the step budget is at or above 100k) is the only testable unit among
the schedule items, at a summed 1.05 PPL that clears the floor marginally.

Among the new candidates, the categorical emission in the belief's Markov blanket is the one whose
ceiling is not already bounded by M6, and the content-dependent prefix-measurable source prior is
aimed at the 0.612-nat channel rather than the 0.210-nat one. Both are effort M and both are
unmeasured.

## 9. Open obligations

Every magnitude in section 5 is unknown, and the project's own base rate is that essentially all
highly ranked mechanism hypotheses have been refuted by cheap measurement. Any arm derived from
"make the objective more exactly the ELBO" must be judged on held-out perplexity and never on F,
because V3's measurements show the two dissociating in both directions.

`gradient_mode='smoothing'` cannot be scored at all until a separate target-blind prior-predictive
scorer exists, because under causal masking the recoil terms make position `i` depend on `x_{i+1}`.
The correct claim about V3's E-step is the manuscripts' own: it is a receiver-only local surrogate,
correctly so given the autoregressive protocol, and no coordinate-ascent or monotonicity language
should attach to it.

Ten manuscript chapters were not read in full, the appendices being the largest unread surface and
the likeliest place a further executable prescription would hide. The Böhning bound was not verified
numerically. No vault writes were made; the retracted B-01 residue on four wiki pages remains and
requires the user's confirmation.
