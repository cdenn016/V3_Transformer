# Performance brainstorm from a six-agent sweep of the VFE 4.0 manuscripts, the vault, and the code

Date: 2026-07-25. Baseline: `data/55.41_wikitext-103_K300_block_glk_linear_mix_s6` (K=300, 10 heads,
d_head=30, n_layers=1, context 128, 529M params, val PPL 54.18 / test 55.41).

Six expert agents swept the VFE 4.0 whitepapers (`vfe4_whitepaper` 9 chapters,
`magent_elbo_whitepaper` 13 chapters, `PIFB2.tex`, `GL(K)_attention.tex` and its supplementary),
the Obsidian research vault, and the executable. Two further agents (the content-channel
decomposition and the external literature baseline) were stopped before reporting; their questions
are recorded as open in section 8. Organized by mechanism, not by cost. Nothing here has been
implemented and no application code was changed.

Read `docs/2026-07-25-state-of-knowledge.md` first. Everything its section 2 refuted stays refuted.

## 1. The softmax logit budget is spent on something that is not content

This is the sweep's strongest result, and it arrived three times from three unrelated directions.

The gauge-theoretic agent found the theory's exact decomposition of the coupling energy
(`eq:isotropic_general_omega`, `GL(K)_attention.tex:1195`):

    KL(q_i || Omega_ij q_j) = S(Omega_ij) + (1 / 2 sigma^2) || Omega_ij^{-1} mu_i - mu_j ||^2

where the geometric bias `S(Omega) = 0.5 [ log det(Omega Omega^T) + Tr((Omega Omega^T)^{-1}) - d ]`
is non-negative and vanishes if and only if `Omega` is orthogonal (`eq:geometric_bias`, `:1190`).
V3 implements this literally: at `vfe3/families/gaussian.py:218,221,222` the energy is
`0.5 * (trace_term + mahal_term - K + logdet_term)`, and the combination
`trace_term + logdet_term - K` contains no `mu` at all. Because `sigma_log_embed` is frozen at
`log 4`, that content-free part is a pure function of the row norms of `Omega_ij = U_i U_j^{-1}`,
which is to say of frame conditioning. It competes for the same softmax budget as content.

The VFE 4.0 agent reached the same object from `eq:normalized-gaussian-transition-densities`
(`04:125-141`), which indexes the transition covariance by the receiver `t` alone and never by the
pair `tj`. V3 instead scores against `sigma_t`, the diagonal of the congruence
`U_i U_j^{-1} Sigma_j U_j^{-T} U_i^T` (`transport.py:2371-2407`), whose trace, Mahalanobis and
logdet terms are all j-dependent. Under the exact congruence the j-dependent logdet is
`log det Sigma_j - 2 tr(phi_j)`, so each row carries a content-free per-key bias of roughly
`tr(phi_j) / tau`. The per-block basis is full `gl(30)` (`groups.py:255-292`), so `tr(phi)` is
unconstrained.

The exact-ELBO agent found a third face of it. At `tau != 1` the tempered row needs a
per-key log-normalizer (`09:284`), which for a Gaussian factor contributes
`-0.5 (1 - 1/tau) log det S_ij` to the logit. V3 adds nothing:
`vfe3/free_energy.py:318-321` is `logits = -energy/tau + log_prior`. At `tau = 5.477` the missing
coefficient is 0.409.

Three fixes to one defect, and they are not the same fix:

**Orthogonalize the transport.** A per-token, per-head polar retraction `R_i = polar(exp phi_i)`
gives `Omega_ij = R_i R_j^T` and `S == 0` exactly (`eq:ok_transport`, `:1214`). It costs zero new
parameters, keeps the entire 9000-coordinate table (the polar of `exp phi` depends nonlinearly on
all 900 per-head coordinates, so this is emphatically not the 435-parameter `so_n` cut), and one
30x30 polar per token-head is cheaper than the `matrix_exp` already at `transport.py:1444`. Its
cost is the loss of the non-compact symmetric directions, which M5's rank saturation is a genuine
warning about, though a polar retraction is not a rank cut.

**Fix the reference covariance at the receiver.** Replacing the transported sender covariance with
a receiver-indexed `R_t` kills both nuisance terms (they cancel in the softmax) and makes
`pair_prec = sum_j w_ij P_t = P_t`, an explicitly learned pair/prior ratio. The VFE 4.0 theory
proves this is the only regime in which V3's edge is genuinely an ELBO term
(`eq:mean-field-fixed-covariance-v3-reduction`, `06:799-813`) and that it fails for a live sender
covariance (`06:815`).

**Add the missing tempered normalizer.** `log det S_ij` is already computed inside the pair KL
(`families/gaussian.py:221`), so this is an arithmetic change at the seam.

A connection worth recording rather than proposing. The exact congruence was refuted because its
KL grows like `cond(Omega)^2`, saturates `kl_max`, and the clamp's zero derivative then removes the
only force pulling `phi` back toward well-conditioned frames. Under the polar retraction
`cond(Omega) = 1` identically, so that specific failure mode is absent by construction. This does
not revive the exact-congruence family on its own terms, but it does mean the two questions are
not independent.

## 2. The two channels share their routing, so they are not two channels

Three agents independently landed on `vfe3/model/model.py:2342-2364`. Under
`gamma_as_beta_prior=True`, the belief channel's attention prior is

    pi_ij = (1 - w) softmax(ALiBi_ij) + w gamma_ij,   w = gamma_prior_weight = 0.5

with `gamma` computed under `no_grad` and entering the softmax at unit log-weight, while the
belief's own coupling energy enters at `1/tau = 1/5.477`. Both channels additionally run
`causal_alibi_noself` and share the identical transport object.

The belief channel therefore inherits half its routing from the model channel for free and pays
5.5x to use its own content. That is a single concrete mechanism for two of the four open
questions at once: why attention reads as mostly positional (open question 2) and why the two
channels overlap by 1.25 nats of their combined 3.20 (open question 3).

The theory reframes this from a wiring accident into a prescription. `eq:prefix-measurable-content-
source-prior` (`07:81-92`) puts content in the prior, `pi_t(j) ∝ exp s_theta,tj(Y_{<t}, x_{<t}, Gamma)`,
and `06:572` forbids `tau != 1` on that row. Read that way, the content path the theory wants is the
untempered prior, not the tempered energy, and `gamma_as_beta_prior` is the seam where it already
half-exists. The design question is whether to strengthen that path or to separate the two channels
so they do different jobs, for instance by giving the s-channel a windowed prior and the belief
channel ALiBi, rather than the same kernel twice.

## 3. The hyper-prior anchor is the zero vector

`vfe3/model/prior_bank.py:560-561` allocates `r_mu = torch.zeros(K)` with
`requires_grad=learnable_r`, and the baseline sets `learnable_r=False`. It is broadcast at
`model.py:851` and passed as `mu_p` to the s-channel E-step at `model.py:863`. In
`kernels.py:666` the update is `mu_star = (a mu_p / sp + pair_mean) / P` with `a = lambda_h = 0.25`
and `sp = 4`, so the anchor contributes exactly zero to the numerator and 0.0625 to the fused
precision.

`lambda_h` is therefore not a pull toward a learned centroid. It is pure shrinkage toward the
origin. On a tiny CPU build at the baseline toggles, `||s||_rms` runs 0.01702 at depth 0 to
0.00872, 0.00582, 0.00192 and 0.00008 by depth 8, with `cos(s_d, s_0)` falling from 1.00 to 0.300.
That sharpens the 3.5-nat depth cliff from "collapse toward a global centroid" into "collapse to
zero," which is a different and more tractable defect. Both fixes are already wired:
`learnable_r=True` (300 parameters) or `r_update_mode='barycenter'` (`prior_bank.py:795-809`).

The companion item is the layer handoff. `stack.py:147` is
`mu_p <- (1 - rho) mu_p + rho belief.mu`, a running blend in which the entry prior's weight decays
as `(1 - rho)^L`, reaching 1/16 by four layers even at `rho = 0.5`. Anchoring instead to the entry
prior captured before the loop keeps weight `(1 - rho)` at every depth. This is standard input
injection and is one line plus one config field.

Two depth hypotheses died here. The transport-free prior handoff is **not** a defect: the blend at
`stack.py:147-148` is at the same position, means live in the ambient basis, and `phi` enters only
through `Omega_ij` between distinct positions, so under `mu -> g_i mu` both terms carry the same
`g_i` and the blend is exactly equivariant. And `prebuilt_transport` is valid and already on;
transport builds stay at 2 for `n_layers` 1, 2 and 3.

## 4. The prior/pair balance is a directly controllable scalar, and the theory has no residual

`lambda_beta` multiplies `w` at `kernels.py:649` and enters only `pair_prec` and `pair_mean`;
`prior_prec = a/sp` at `:656` is untouched. The pair share is therefore
`lambda * 0.298 / (lambda * 0.298 + 0.702)`, giving 0.46 at `lambda = 2` and 0.56 at `lambda = 3`.
The measured 70/30 split is a tunable, not a fixed property of the architecture.

The theory is more pointed than that. VFE 4.0 has no residual term at all:
`eq:typed-gaussian-transition-locations` (`04:113-123`) sets the prior mean to
`Omega_tj z_j + B_t m_t + c_t`, and the entire aggregation/prior balance is carried by the learned
receiver precision `P_s^z` in `eq:mean-field-state-natural-update` (`06:726-747`). V3's 70%
residual is an artifact of minimizing a weighted sum of KLs, in which `-log sigma_q` is counted
`(alpha + lambda_beta)` times where the exact reverse-KL counts recognition entropy once
(`07_configuration_elbo.tex:188`). The same double-count makes V3's `sigma_star` exactly
`2 J_bb^{-1}` rather than the exact block optimum `J_bb^{-1}` (`kernels.py:675-676` versus
`eq:gaussian-block-reverse-kl-optimum`), which is worth precisely nothing today because
`skip_belief_sigma_update=True`, but becomes real the moment sigma acquires a consumer.

`lambda_twohop` (`kernels.py:650-652`, currently 0.0) is the only wired compositional path, and it
is honest about its ceiling: flat Regime-I composition telescopes to `U_{v_r} U_{v_0}^{-1}`
(`eq:regime-i-open-path-telescope`, `09:458-463`), so it adds weights, not geometry.

## 5. The decode is rank-limited, and the fix is parameter-negative

The linear decode is `logits = mu W^T + b` (`prior_bank.py:1900-1904`), so logit rank is bounded by
`K + 1 = 301` against a 50257 vocabulary. The prior-bank KL decode computes
`logits = -(L_i . R_v + c_v - p_i) / (2 tau)` with `L_i = [sigma_q + mc_q^2, -2 mc_q]` in `R^{2K}`
(`prior_bank.py:1648-1656`), and `p_i` is row-constant so it drops under the softmax. Rank is
bounded by `2K + 1 = 601`, exactly double, obtained from the elementwise square, and it drops
`output_proj_weight` for a net saving of 15.08M parameters.

The evidence that the bound binds is a measured invariance: `output_proj_weight` has entropy
effective rank 223.8 of 300 (74.6%) at K=300 and 178.1 of 240 (74.2%) at K=240. A utilization
fraction that holds fixed under a 25% width increase is the signature of a filled budget rather
than slack. Cost is a decode GEMM of width `2K` instead of `K`, so decode FLOPs exactly double,
which is 4 to 6% of wall clock. It pairs with `skip_belief_sigma_update=False`, since the linear
decode marks `sigma_q` discarded at `prior_bank.py:1888`.

A correction to the prior write-up: sigma is frozen only in the E-step. `s_sigma_log_embed` trained
hard in the M-step, ending at mean 0.0102 against `log(4) = 1.3863` at initialization.

The sibling proposal is a second readout, `logits = W^z mu + W^m s + d`
(`eq:gauge-compatible-categorical-emission`, `04:205-214`). Since the decode is linear and
`mu* ~ 0.70 s + 0.30 A`, adding `W^m` makes the prior/pair balance learnable per readout direction
with no E-step change at all, and zero-init leaves step 0 byte-identical. The theory's observation
kernel reads both channels (`o_i ~ N(H_i k_i + L_i m_i + d_i, R_i^o)`, `04:82-87`); V3 has `L = 0`,
which is the theory's own account of the channel redundancy, since the two channels are chained
through one K=300 bottleneck instead of being read jointly.

## 6. Capacity re-budgets are close to a wash, and the slope pricing them is confounded

The `-0.226` width slope is not a controlled pair. `data/58.28_..._K240/config.json` and
`data/55.41_..._K300/config.json` differ in `amp_dtype` (None versus bf16), `m_s_phi_lr` (0.016
versus 0.007), `prior_handoff_rho` (0 versus 1), and code revision, and the comparison moves `K` and
`n_gen` together since `n_gen = K * d_head`. It cannot say whether width or gauge capacity paid.

`tied_block_glk` looks free and is not. The head-average of `phi` retains 0.1025 of its Frobenius
energy at H=10 and 0.1277 at H=8, which is exactly `1/H`, with cross-head cosine 0.0026 and 0.0030.
The ten per-head gauge blocks are statistically independent, so the tie discards 90% of the table.
Priced on the log-slope, the tie costs +0.52 nats for `ln(9000/900)` and spending the freed 407M on
`K` returns -0.52 for the same `ln 10`. The prediction is identically zero. The freed budget is also
largely unspendable, because the `(B,N,N,K)` pair tensors at `transport.py:2408` scale linearly in
`K`.

The K=600 / H=60 / d_head=10 re-budget totals 392.83M parameters, which is 21% smaller than a
current-main build of the baseline config, so the slope predicts it is worse rather than better.
It also drops `tau = kappa sqrt(d_head)` from 5.477 to 3.162, off M1's measured optimum. One
sub-claim from the earlier write-up is wrong: the clamp does not become 2.45x tighter, because
`block_glk`'s generator Gram is exactly the identity (`groups.py:291-292`), so the joint reduction
bounds `||phi||_2 <= 20` regardless of `H`, and cutting `n_gen` from 9000 to 6000 makes
per-coordinate room 1.22x looser.

The vault is blunter still. Three independent width-scaling fits give extrapolated PPL floors of
52, 64 and 70, all at or above the 55.41 baseline. The width axis is near-exhausted. Across-seed CV
is 0.6 to 1.1%, so +/-0.6 PPL at this baseline, which is larger than several of the wins proposed
across this whole sweep.

## 7. Items that are now closed, with the reason

The joint-precision solve is dead, and not for the reason anyone expected. Because
`beta_attention_prior` is causal, `J` is block lower-triangular, so the mean-field evidence gap
`eq:gaussian-mean-field-evidence-gap` is exactly zero (verified residual -4.9e-14 at every
conditioning tested). Jacobi's iteration matrix is nilpotent, so there is no asymptotic gap to
close, and `J` is nonsymmetric so CG does not apply in any case. At the run's
`vertex_cond_median = 66`, `(J + J^T)/2` is not even positive definite (`lambda_min = -1.9` at
cond 37), so no Gaussian posterior has this precision and the exact-ELBO results have false
premises here.

The canonical-F versus entropy-suppressed-surrogate question is settled and worth zero nats. V3
computes the canonical form (`free_energy.py:437` plus `:458-459`, with the envelope at `:371`), and
the training loss is plain cross-entropy (`model.py:1585-1595`); the only F-in-loss path
(`model.py:1689`) is gated on `not s_e_step`, which the baseline does not satisfy.

The attention temperature is theory-correct, not a defect: `free_energy.py:71-72` returns
`kappa sqrt(30)`, exactly the canonical `tau = kappa sqrt(K_q)` specialized per head (`PIFB2:673`,
`attention.tex:908`). The untempered log-prior is likewise prescribed (`attention.tex:806`) and
correctly implemented. The vault adds that softmax temperatures cannot be learned by F-descent at
all, since `dF_red/dtau = KL(beta || pi) >= 0`, verified symbolically.

A documentation correction is outstanding. The supersession banner on
`docs/2026-07-25-vfe4-performance-hypotheses.md` preserves sections 1.5 and 2.1 (frame-intrinsic
covariance, described there as "the strongest single proposal"), but that family is measured-refuted:
the Regime-I coboundary cancels, `phi_embed.grad is None` under every estimator, and PPL goes
139.3 to 308.1. The banner should retract 1.5 and 2.1 rather than preserve them.

## 8. Open, including what this sweep failed to answer

Two agents were stopped before reporting, and their questions remain open.

The content-channel decomposition is the more important of the two. The specific question is whether
the pair energy's mean term, `mu_i - Omega_ij mu_j` with sigma effectively constant, is a
gauge-warped squared distance with the query/key asymmetry that dot-product attention has, or
something closer to symmetric and therefore a structurally weaker router. That would explain the
0.21-against-0.61 content-to-position ratio at a level none of the six returned memos reach. The
partial result before the stop was only that the measurement harness reproduced M6 exactly.

The external literature baseline was never established: what a standard transformer achieves on
wikitext-103 at 529M parameters, 368.6M training tokens (about 3.6 epochs of a 103M-token corpus)
and a 128-token context, and hence how much of 55.41 is architecture rather than budget. The related
anomaly stands unexplained, that this model gains 0.030 nats going from a 32-63 token context to
64-127 where standard transformers gain considerably more.

From the vault, the priority-1 five-seed variance floor and the priority-2 gauge ON/OFF/frozen study
at L=2 were both specified and never run, so every mechanism ablation except the width and block
sweeps is single-seed. Width-aware learning-rate and muP controls never ran either, so the K=300
baseline's tuning is unvalidated. Two theory levers carry concrete predictions and have never been
measured: the symplectic-momentum E-step with its `sqrt(M/K)` overshoot law, and the susceptibility
response law `chi = 1/lambda_alpha`.

Three magnitudes in this document are unmeasured and gate everything above them. The share of
row-centered logit variance carried by `S(Omega)` decides section 1. The row-wise agreement between
`beta` and `gamma` decides section 2. The s-channel's own prior/pair precision split, the analogue of
the belief channel's 0.702/0.298, decides how much of section 3 matters.

## 9. Query/key asymmetry and the geometric bias have the same source

This section is algebra done after the agents returned, and it revises the recommendation in
section 1.

Start from the KL as written, `KL(q_i || Omega_ij q_j)`, whose Mahalanobis term is

    (Omega^{-1} mu_i - mu_j)^T Sigma_j^{-1} (Omega^{-1} mu_i - mu_j)

with cross term `mu_i^T U_i^{-T} U_j^T Sigma_j^{-1} mu_j`. That is exactly the manuscript's
separable compatibility `M_ij = A_i B_j` with `A_i = U_i^{-T}` and `B_j = U_j^T Sigma_j^{-1}`
(`GL(K)_attention.tex:1267`). The query side is therefore `U_i^{-1} mu_i` and the key side is
`U_j^T Sigma_j^{-1} mu_j`.

Take the polar decomposition `U = R P` with `R` orthogonal and `P` symmetric positive definite, and
write `a = R^T mu` for the body-frame coordinate. Under isotropic `Sigma_j = sigma^2 I`,

    query side = P_i^{-1} a_i,   key side = P_j a_j / sigma^2,   score_ij ∝ <P_i^{-1} a_i, P_j a_j>.

The SPD factor enters the query as `P^{-1}` and the key as `P`. That is what makes the score
asymmetric in `i` and `j`. If the transport is orthogonal, `P = I`, the two sides become `a_i` and
`a_j`, and the score collapses to the **symmetric** inner product `<a_i, a_j>` plus a per-key bias.

An earlier draft of this section derived the same conclusion with the two sides inverted, by
starting from the code's `(mu_i - Omega mu_j)^T Sigma_t^{-1} (mu_i - Omega mu_j)` and treating
`Sigma_t^{-1}` as approximately the identity on the grounds that `sigma_trace_cv = 0.1155`. That
step is wrong: `Sigma_t = Omega Sigma_j Omega^T`, so `Sigma_t^{-1}` carries the `Omega` factors and
is precisely what makes the form separable. The structural conclusion is unaffected, but the
assignment of `P` versus `P^{-1}` to query and key was backwards.

Two refinements the manuscript forces. Because `Sigma_j` in V3 is diagonal and token-dependent
rather than isotropic, an orthogonal transport leaves a residual asymmetry through
`R_j^T Sigma_j^{-1} mu_j`, so orthogonalization removes the dominant source of asymmetry rather
than all of it. And the diagonal truncation of the transported covariance means V3's realized
compatibility does not factor as `A_i B_j` at all, since the truncated `Sigma_t` depends on the pair;
the separable form above is the exact-congruence idealization.

Meanwhile `S(Omega)` depends on `Omega Omega^T`, whose deviation from the identity is governed by
the same `P`. Asymmetry and geometric bias are generated by one object.

The consequence is a direct caution against section 1's polar-retraction proposal. Orthogonalizing
the transport does kill `S(Omega)` exactly, as claimed, but it simultaneously destroys query/key
asymmetry. A symmetric attention score cannot express "attend to the token that FOLLOWED a previous
occurrence of my current token," which is the induction-head pattern that two-layer attention-only
transformers are known to rely on (Elhage et al. 2021; Olsson et al. 2022). Since V3 is effectively a
two-layer attention-only network with K-composition already available (the belief channel's keys are
built from the s-channel's output), induction is exactly the mechanism most worth preserving. The
polar retraction should be demoted accordingly.

This also partially answers the question the stopped content-channel agent was assigned. The pair
energy is **not** symmetric, so V3 is not structurally barred from asymmetric routing. But the
asymmetry is carried entirely by the non-compact directions of the gauge, which are the same
directions that generate the nuisance logit variance and the `cond(Omega)^2` blowup that killed the
exact congruence. The three problems are one problem.

The fix that follows is narrower and better than orthogonalization: **keep the full `U` in the
bilinear term and drop the content-free trace and logdet terms from the routing logits only**,
leaving the exact KL intact wherever `F` is reported. Routing then sees content and asymmetry with
no frame-conditioning nuisance. The honest cost is that `beta` is no longer the exact softmax
stationary point of `F`, which is defensible here only because `F` never enters the loss (section 7),
and should be documented as an opt-in rather than presented as theory-pure.

One caution on that fix, which nobody raised. `pos_phi_free` is an absolute-position table that
composes into `phi`, so `tr(phi_j)` carries a positional contribution as well as a token one. The
content-free part of the energy may therefore be one of the channels through which absolute position
reaches the routing. Removing it might remove position rather than noise, which would show up as the
position share falling along with the nuisance. That makes the decomposition of row-centered logit
variance worth splitting three ways, into content, token-frame conditioning, and position-frame
conditioning, rather than two.

## 10. What V3 structurally lacks that a standard transformer has

An earlier draft of this section claimed V3 "has no position-wise nonlinearity at all." That is
wrong, and `GL(K)_attention.tex` had already scoped the question correctly at
`sec:ffn_nonlinearity` (`:2032-2102`). The E-step is nonlinear: the per-edge contribution is a
linear residual `e_ij = mu_i - Omega_ij mu_j` multiplied by a Boltzmann gate
`exp(-E_ij/tau)/Z_i` (`eq:vfe_glu`), the binary case reduces to a logistic sigmoid and has the
`x sigma(g(x))` form of SiLU (`eq:binary_silu`), the softmax response
`d beta_ij / d mu_i` supplies a further nonlinear channel on the entropy-suppressed objective
(`eq:softmax_gradient_nonlinearity`), and repeated inner iterations supply nonlinear recurrent
computation. The corrected absences are narrower and the manuscript states two of them itself.

**The nonlinearity is a source-normalized gate, not a learned channel transformation.** The
manuscript is explicit that this "is not a transformer FFN: it lacks the learned expansion and
contraction maps, its gate is normalized across sources, and its argument is a relative KL energy
rather than an independently learned channel projection" (`:2035`), and that "repetition supplies
nonlinear recurrent computation, but its iteration count is not FFN depth and each step is not
literally a GLU. A standard FFN has untied learned channel expansion, activation, and contraction
that are absent here" (`:2102`). Normalization across sources is the operative point for
performance: the gate mixes over neighbors `j`, so it is a routing nonlinearity in the same family
as ordinary attention, and nothing in the model applies a learned nonlinear map to a single
position's contextualized representation. The token tables are enormous per-position capacity, but
they are indexed by the raw token id, so they act before context arrives and are context-free by
construction. Everything downstream of the lookup is affine plus routing. That is the gap section 11
fills, and the manuscript names its two missing pieces exactly: learned expansion and learned
contraction.

**The compatibility is constrained to be separable and invertible.** In standard attention the score
is `x_i^T W_Q^T W_K x_j` with `W_QK` one arbitrary learned matrix shared across all pairs, and it
may be singular. V3's is pair-dependent but must factor as `M_ij = A_i B_j`, and "every ambient
`M_ij` here is invertible, whereas a transformer compatibility may be singular. The two mechanisms
share a factorized dot-product evaluation, not an identical representable function class"
(`:1278`; see also `:1267`). Invertibility is the binding restriction, because low-rank `W_QK` is
how a transformer implements selective matching on a subspace while ignoring the rest. A learned
per-head SPD metric `G_h` in the bilinear term (900 parameters per head, 9000 in total) adds a
shared learned component but does not lift the invertibility constraint; a learned low-rank or
singular shared factor would. A constant `G` does not transform correctly under a general gauge, so
this is an opt-in non-equivariant baseline with the same footprint as the existing head mixer, and
it is exactly equivariant under a tied or orthogonal gauge.

The manuscript also supports section 9's routing proposal from an independent direction. It shows
the key-norm bias term cancels under the row softmax only when key norms are exactly constant
(`:1483`), and that the residual key-dependent fluctuation is `O(sqrt(d_k))`, "the same order as the
dot-product signal, so increasing dimension does not make the bias row-constant" (`:1342`). A
non-content per-key bias of the same order as the content signal is precisely what the
Mahalanobis-only routing logits remove.

**The decode reads only the final belief.** A transformer's readout sees the whole residual stream,
which is the sum of every layer's contribution. V3 decodes `mu_q` alone (`prior_bank.py:1900`),
discarding both the token prior and the s-channel output that produced it. Reading `[mu; s_mu]`, or
more generally the concatenated stream, is the readout analogue of a residual connection and is
zero-risk under zero initialization.

## 11. A mixture hyper-prior supplies the missing learned expansion and contraction

The hyper-prior `h` at the top of the `h -> s -> p -> q` hierarchy is currently a single Gaussian
whose mean is the zero vector (section 3). Replace it with a mixture `{(r_c, Sigma_c)}` for
`c = 1..C`, with responsibilities `rho_ic ∝ pi_c exp(-KL(s_i || h_c))` and anchor
`sum_c rho_ic r_c`.

Every property wanted above falls out. The anchor becomes a data-dependent soft lookup over `C`
learned prototypes, which is a position-wise nonlinear map, and a key-value memory in precisely
Geva's sense. The parameter cost is `C * K`, so 4096 prototypes at `K = 300` is 1.2M parameters,
0.23% of the model. It fixes `r_mu = 0` at the root instead of patching it: iterating the s-channel
now moves each token toward its nearest prototypes rather than toward the origin, which removes the
depth-collapse mechanism rather than damping it. It gives depth something to do, since repeated
application is a soft clustering rather than repeated averaging. And it is theory-native, the
natural generalization of a single-Gaussian hyper-prior, and the same object the depth agent
identified as the in-theory alternative to a per-token `r_i` that the shadow-prior work already
rejected.

The risks are honest ones. Responsibilities over `C` components add a softmax over `C` KL
evaluations per position, which is a real cost at large `C`. Prototype collapse is the standard
failure mode of any mixture with learned components and needs the usual counter-pressure. And the
gauge question needs care, since `KL(s_i || h_c)` is gauge-invariant only if the prototypes live in
the same frame as `s_i`.

## 12. Two existing refutations are narrower than they have been read

**M5 does not refute a factored parameterization.** M5 established that truncating the *trained*
`phi_embed` to rank 4096 of 9000 still costs 7.1 PPL, so the trained solution is high-rank. That is
a statement about post-hoc truncation of a table trained dense. It says nothing about a table
*parameterized* as low-rank and trained from scratch, which learns different and better-conditioned
frames and shares statistical strength across tokens. Post-hoc pruning and constrained training are
not the same claim, and the capacity agent made exactly this point about the tied gauge ("the 1/H
independence is a property of the trained untied solution and does not bound a model trained tied
from scratch") without generalizing it. Sharing matters most for the 2319 vocabulary rows with zero
training count and the roughly 4000 more below random initialization, which a factored table would
give structure instead of decay.

**`kappa_beta` and `alibi_slope` are a two-parameter family, not one knob.** The logits are
`-E/tau + s * ALiBi`. Scaling `kappa` shrinks content; scaling `s` grows position. Both move the
content-to-position ratio, so it is tempting to call them the same knob, and if the softmax were
scale-invariant they would be. It is not: `softmax(c x) != softmax(x)`. The pair `(kappa, s)`
therefore spans a two-dimensional family whose second direction is the overall logit temperature,
and M1's eight-point `kappa` sweep explored one line through it. `alibi_slope` at fixed `kappa` moves
in a genuinely different direction and has never been swept. The magnitudes make this worth doing:
head 8's ALiBi span is `0.70711 * 127 = 89.8` nats untempered, against a content spread of roughly
`3.75 / 5.477 = 0.69` nats. A ratio near 130 to 1 is a sufficient explanation for "attention is
mostly positional" on its own.

## 13. The statistical efficiency problem underneath all of this

The run saw 368.6M tokens against 529M parameters, about 0.7 tokens per parameter, where
compute-optimal practice is nearer 20. Standard scaling intuition does not transfer cleanly, because
the parameters are overwhelmingly a per-token lookup table rather than shared weights, but that is
the problem rather than an excuse: a per-token 9000-dimensional gauge frame needs many observations
of that token to estimate, and wikitext-103 supplies 103M tokens across 50257 types. The measured
consequence is already recorded, with 2319 rows at exactly zero and rows at counts 1 to 15 sitting
below random initialization.

No proposal in this document addresses that directly except the factored parameterization of section
12 and the mixture prototypes of section 11, both of which work by sharing structure across tokens
instead of estimating each independently. The vault's finding that three width-scaling fits floor
between PPL 52 and 70 is the same problem seen from the other end: adding width adds per-token
parameters that the corpus cannot estimate.

## 14. Ranked plan

Ranked by expected value, which is magnitude times confidence, with the gating measurement named.
Tier 0 items are measurements rather than changes, and everything in tiers 1 and 2 is priced against
them. Effort is S, M or L.

**Tier 0, the four measurements that reprice everything else.** Split row-centered logit variance
three ways into content, token-frame conditioning and position-frame conditioning (section 9), which
gates items 2 and 3. Row-wise agreement between `beta` and `gamma` (section 2), which gates item 6.
The s-channel's own prior/pair precision split (section 3), which gates item 5. A two-dimensional
`(kappa_beta, alibi_slope)` grid rather than a line (section 12), which is the cheapest test of the
single largest ratio in the model. None requires retraining.

**Tier 1, structural, highest expected value.**

1. Mixture hyper-prior supplying a learned position-wise channel map (section 11). Effort M to L,
   roughly 1.2M parameters at C=4096. It adds the learned expansion and contraction that
   `GL(K)_attention.tex:2035,2102` names as the two pieces separating V3's gated linear message from
   a transformer FFN, and it subsumes the `r_mu` fix rather than competing with it. Magnitude
   unknown and potentially the largest on the list, because it adds a capability rather than tuning
   one.
2. Mahalanobis-only routing logits (section 9). Effort S. Directly targets the 0.21-against-0.61
   content-to-position ratio while preserving the query/key asymmetry that orthogonalization would
   destroy. Gated by tier 0's three-way split.
3. Learned shared per-head query-key metric `G_h` (section 10). Effort S, 9000 parameters. Supplies
   the one piece of standard attention V3 has never had. Opt-in non-equivariant, same footprint as
   the head mixer.

**Tier 2, structural, well-understood.**

4. Prior-bank KL decode with `skip_belief_sigma_update=False` (section 5). Effort S to M. Doubles
   logit rank from 301 to 601 and is parameter-negative by 15.08M. The 74.6%-versus-74.2% rank
   utilization invariance is the strongest direct evidence in this document that a bound binds.
5. Fix the hyper-prior anchor and the layer handoff (section 3): `learnable_r=True` or
   `r_update_mode='barycenter'`, plus a fixed entry anchor in place of the running blend at
   `stack.py:147`. Effort S. Small on its own; it is the prerequisite that makes depth available at
   all, and it is subsumed by item 1 if item 1 is taken.
6. Separate the two channels (section 2): retune or remove the `gamma_prior_weight = 0.5` fold, and
   give the two channels different attention priors so they do different jobs. Effort S.
7. Second readout from `[mu; s_mu]` (sections 5 and 10). Effort S, zero-init byte-identical.

**Tier 3, free, do alongside anything.** Weight-decay exemption for `phi_embed`, `sigma_log_embed`
and `s_mu_embed`, or better, frequency-aware decay so frequent rows still get regularized;
`exp_fp64_mode='norm'` with threshold 21, worth roughly 2x throughput on its own;
`min_lr_frac=0`; `warmup_steps` near 1500.

**Tier 4, speculative but cheap to specify.** A factored `phi_embed` trained from scratch rather than
truncated post hoc (section 12). Differentiated transports for the two channels, a cheap gauge for
the s-channel and the full one for the belief channel (section 10).

**Not recommended.** The polar retraction as a standalone fix, demoted by section 9 because it
destroys query/key asymmetry. `tied_block_glk`, whose cost and gain are the same measured slope with
opposite signs. K=600 with `d_head=10`, which is 21% smaller than the current build and predicted
worse. The joint-precision CG solve, dead on triangularity. Anything further on the exact congruence
or the frame-intrinsic family.

**A standing caveat on all of it.** Across-seed CV is 0.6 to 1.1%, which is plus or minus 0.6 PPL at
this baseline, and every mechanism ablation on record except the width and block sweeps is
single-seed. Several items above are individually smaller than the noise floor and cannot be
distinguished from it by a single arm.
