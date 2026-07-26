# Cross-scale shadow priors for V3: six-expert investigation, scope, and plan

Date: 2026-07-25

Question put to the panel: the author observed in the MAgent simulations that agents descend the
variational free energy onto a GAUGE ORBIT and that OBSERVATIONS break the symmetry, and asked
whether the manuscript's cross-scale shadow prior
(`p_i^{(s)} = Omega_{i,I}[q_I^{(s+1)}]`, PIFB2 `sec:cross_scale_shadows`) is the way to give the
transformer an analogous observation channel.

Six independent read-only investigations: gauge theory, variational inference, transformer
architecture, implementation scoping, literature, and falsification. Every load-bearing claim below
was re-verified against the artifacts before being recorded here.

## 0. Corrections to the framing this investigation was launched on

Three claims made in the conversation that motivated this work do not survive checking. They are
recorded first because two of the six briefs were written on top of them.

**"Free energy falls monotonically while cross-entropy rises" — WRONG, twice over.** The full
`estep_depth_sensitivity.json` from both runs:

| depth | K=20 CE | K=20 F/token | K=300 CE | K=300 F/token |
|---|---|---|---|---|
| 0 | 6.8321 | 34.4987 | 5.8679 | 433.1152 |
| 1 (trained) | **4.8831** | 31.9145 | **3.7744** | 386.4054 |
| 2 | 5.1809 | 31.2674 | 4.1369 | 376.4673 |
| 3 | 5.9814 | **31.0221** | 5.0272 | 372.6440 |
| 5 | 7.5561 | 31.0294 | 7.0092 | **370.3582** |
| 8 | 8.1485 | 31.0674 | 7.9681 | 370.9829 |

First, DEPTH 0 EXISTS and is the worst CE in both runs. The first E-step lowers F *and* CE by ~1.9
nats (K=20) / ~2.1 nats (K=300). Inference and prediction are ALIGNED on the step the model was
trained to take; the anti-alignment begins only at depth 2. Second, F is NOT monotone: it bottoms at
depth 3 (K=20) / depth 5 (K=300) and rises after. The largest CE damage (depth 3 to 8, +2.2 nats
K=20) occurs where F is FLAT OR RISING, so "F descent causes CE ascent" fails precisely where the
effect is biggest.

**"The token enters as an initial condition, not a force" — WRONG.** `gradients/kernels.py`
`mm_exact_update` fuses `mu_star = (a * mu_p / sp + pair_mean) / P` with
`prior_prec = a / sp`: the prior enters EVERY iteration as a precision-weighted anchor. Under the
active config `e_step_update='mm_exact'`, so this is the live path. The token is a persistent force.
What is true is narrower and more interesting: the anchor's WEIGHT DECAYS (Section 2).

**"`p_i` is a per-token embedding lookup" — incomplete.** The active config sets
`prior_source='model_channel'` with `s_e_step=True`, so `p_i` is already supplied by the s-tables and
refined by `_refine_s` before the belief E-step; `mu_embed`/`sigma_log_embed` are `None` (the base
tables are omitted, not merely unused).

## 1. Does the MAgent gauge-orbit argument transfer? No, and the reason is structural

V3 has exactly two flat directions, and both are parameterization redundancies invisible to every
observable.

The rigid right action `U_i -> U_i g` (one `g`, all `i`) is exact because `phi` enters the loss ONLY
through `Omega_ij = U_i U_j^{-1}`; nothing else reads it. It changes no energy, no beta, no
aggregated mean, no logit. The second is global conjugation of all tables
(`mu_v -> g mu_v`, `Sigma_v -> g Sigma_v g^T`, `U_v -> g U_v`, `W -> W g^{-1}`), invariant because a
divergence is invariant under a common invertible pushforward; its realizable subgroup is small here
(the diagonal family plus the diagonal-of-sandwich transport is equivariant only for block-preserving
monomial `g`, and `use_head_mixer=True` forces `g` tied across heads by Schur).

Both are already lifted, by AdamW weight decay (`phi_weight_decay=0.03`): `||phi||^2` is not
invariant under `U -> U g`. The gauge is fixed by regularization and by the exp-chart clamp. There is
no per-site freedom at all, because `alpha_i D(q_i || p_i)` with `p_i` a fixed ambient-basis table is
itself a gauge-fixing source.

**Therefore no likelihood built from `q` can break either direction.** Direction one leaves `q`
literally unchanged, so no functional of `q` can see it. Direction two: a bank-referenced likelihood
`-D(q_i || p_v)/tau` has BOTH arguments co-transforming, so it is frame-blind — it is the same source
`alpha_i D(q_i||p_i)` already supplies, applied twice. MAgent's mechanism requires a source in a
declared, NON-co-transforming frame; PIFB2 sets `Omega_{i,e_k} = I` for the sensor, and states that
this gauge fixing "is the implicit content of the explicit symmetry breaking." V3 already has one
such object per position: `p_i`. The only genuinely new external frame available in a language model
is the TARGET TOKEN, which `metrics.py` already books as the cross-entropy.

The panel's blunt formulation: prescribing a leaf likelihood to "break the orbit" conflates a flat
direction of the objective with a degeneracy that data can resolve. A true gauge flat direction is
unobservable, so only gauge fixing can lift it, and V3 has three gauge fixings already.

## 2. What the pathology actually is: consensus collapse from anchor decay

Four independent lenses converged on the same mechanism, which is NOT a missing data term.

`lambda_alpha_mode='state_dependent_per_coord'` sets `alpha* = c0/(b0 + D)` (`alpha_i.py`), so with
`b0=c0=1` the self-anchor coefficient is `alpha* = 1/(1+D)`. **The pinning force decays as 1/D while
the coupling weight `lambda_beta = 1` does not.** Deeper E-steps therefore mean progressively weaker
pinning against an undiminished consensus pull.

What it slides into is a genuine degeneracy, just not a gauge one. At fixed `Omega`, invariance of
`sum_j beta_ij D(q_i || Omega_ij q_j)` under `q_i -> g_i q_i` requires
`U_i^{-1} g_i U_i = U_j^{-1} g_j U_j`, and the ZERO SET of the coupling block is larger still: all
body-frame beliefs equal, i.e. consensus, of dimension `2K`. Positions become indistinguishable, so
CE rises while the coupling block falls.

Three corroborations from different directions. The architecture lens: in the `mm_exact` fusion,
`p_i` is the SOLE non-averaging term — every other contribution is an average over positions — so it
is the only thing preventing the fixed point from collapsing to a constant. The falsification lens:
the anchor is switched off entirely once `D >= kl_max` because the self-mask is `1[D < kl_max]`, giving
a second, sharper route to pure peer-averaging. The literature lens: the coupling term is a
smoothness functional and without a data term drives token uniformity — the over-smoothing / rank-collapse
phenomenon (Dong et al., ICML 2021, arXiv:2103.03404).

**The consequence for the proposal is decisive and inverts it.** The shadow prior replaces `p_i` — the
only non-averaging anchor — with a barycenter, which is another average. It makes every input to the
fixed point an average of beliefs: a doubly stochastic averaging operator with no anchor. As a
remedy for consensus collapse the shadow prior is not merely unnecessary, it is counterindicated. The
existing layer stack already demonstrates this at `prior_handoff_rho=1.0`, where
`stack.py:147` `mu_p = (1-rho)*mu_p + rho*belief.mu` makes the next block's prior the previous
block's converged belief, elementwise and with NO transport — the degenerate one-token-per-agent
shadow — and the CE rise with depth is the measurement of that collapse.

## 3. Variational status: the token cannot be both prior and observation

Two findings, both checkable in closed form.

**The shadow term is not the mean-field ELBO term of the augmented joint.** Expanding the MF edge
term of `eq:mf_free_energy` gives `KL(q_i || N(Omega mu_pi, sigma^2 I)) + c`, which DIVERGES as
`sigma^2 -> 0`. The implemented shadow term is `KL(q_i || N(Omega mu_pi, Omega Sigma_pi Omega^T + sigma^2 I))`.
These are different functionals of `(mu_i, Sigma_i)`. The manuscript's "exact ELBO at zero
within-scale coupling" holds for `eq:mf_free_energy`, not for the shadow-substituted objective the
code would run. `rem:shadow_marginal_vs_mf` is therefore load-bearing rather than cosmetic — the
shadow is the Gaussian belief-propagation message, so the object is a Bethe-family free energy:
exact on a tree, stationary-point-only once `beta != 0` adds within-scale loops (Yedidia, Freeman &
Weiss, IEEE TIT 51(7), 2005). With attention on it bounds nothing, and the finiteness of the
rigid-link objective is purchased by the substitution rather than derived.

**Double counting.** Under the proposal each token `o_t` enters the one joint through two likelihood
factors — as `p_A(o_t | k_t)` (the new leaf term) and as `p_B(o_t | k_{t-1})` (the existing CE). The
result is a valid ELBO for data counted with multiplicity two, a generalized/Gibbs posterior at
inverse temperature 2 (Bissiri, Holmes & Walker, JRSS-B 2016), not for
`p_LM(o_{1:N}) = prod_t p(o_t | o_{<t})`. At the default `untie_decode_bank=False` one emission
channel would be asked to place `q_i` simultaneously near `pi_{o_i}` and near `pi_{o_{i+1}}` — a
fight inside the objective, not a bookkeeping nuisance.

**The framing collapses on re-indexing.** A correct causal-LM per-step ELBO puts the likelihood on
the token the latent generates. Put it on `o_t` and it is not the LM objective; put it on `o_{t+1}`
and the current token must re-enter through the prior, which is exactly today's PriorBank. The token
cannot simultaneously set the prior and serve as the observation. "Free the token to be an
observation" is not available.

## 4. Literature: the architecture is known, the bug is real

Strip the gauge apparatus and the proposal is a hierarchical latent-variable LM with a top-down
prior over a token span plus a leaf likelihood: Ladder VAE (Sonderby et al., NeurIPS 2016,
arXiv:1602.02282) structure applied to text, with the coarse/fine split of MEGABYTE
(arXiv:2305.07185) and BLT (arXiv:2412.09871). `Omega` contributes a factorized, per-token,
equivariant special case of the learned linear top-down map everyone else uses — an inductive-bias
difference, not a capability difference. No published result says it buys perplexity.

Two facts should govern the appetite for building it. The record for TEXT is far worse than for
images: Bowman et al. (CoNLL 2016, arXiv:1511.06349) is the canonical demonstration that a strong
autoregressive decoder ignores the latent entirely, Optimus (arXiv:2004.04092) remains the largest
language VAE and uses a single sentence latent with annealing and free bits, and there is no widely
adopted hierarchical latent LM. And posterior collapse is not cured by going top-down: Kuzina &
Tomczak (arXiv:2302.09976) find it concentrated in the layers FURTHEST FROM THE DATA — precisely the
meta-agent.

Two useful positives. Semi-amortized and iterative-inference constructions (Marino et al., ICML 2018,
arXiv:1807.09356; Kim et al., arXiv:1802.02550) all put the likelihood INSIDE the inner loop, so its
absence from V3's E-step is the anomaly rather than the loop being exotic. And the pathology's genus
is named: loss-calibrated approximate inference (Lacoste-Julien, Huszar & Ghahramani, AISTATS 2011)
establishes that the ELBO-optimal `q` is not the utility-optimal `q`. Note also Belrose et al.
(arXiv:2303.08112): a plain transformer is already performing iterative inference, so the loop must
beat that rather than resemble it.

## 5. Implementation scope, if it were ever built

Recorded for completeness; Section 2 argues it should not be built as a fix for this pathology.

The prior seam is SMALL: `e_step`, `vfe_block` and `vfe_stack` take `mu_p`/`sigma_p` as plain
tensors, so a top-down prior needs zero signature changes below `forward_beliefs`. But the encode
registry cannot express it — `EncodeCallable = Callable[[PriorBank, Tensor], BeliefState]` sees only
`token_ids`, while a top-down prior needs other tokens' beliefs — so a new seam is required rather
than a new `encode_mode`.

The likelihood seam is LARGE. `free_energy` is not on the live descent path at all under the active
config (`uses_kernel_route()==True`), confirming the docstring: the term must be added to
`mm_exact_update`, `_diag_kl_filtering_kernel`, the oracle, `free_energy_value` and possibly
`phi_alignment_loss`, and the four must stay consistent or the logged F is not the F being minimized
— a silent, test-passing divergence, since no current test compares them with `log_likelihood` set.
That contract is the single biggest implementation risk. On cost, `decode_mode='family'` is inert
under `use_prior_bank=False` and OOMs at K=300 anyway (9.38 GiB single allocation); the affordable
substitute `decode_ce_diagonal_chunked` is mathematically identical for
`gaussian_diagonal`+`renyi`+alpha=1 and costs 15.6 ms at K=300, but inside the E-step it multiplies
by `n_e_steps x n_layers`.

No gauge-covariant N-to-M pooling exists anywhere in the repo. The s-channel is a second FIBER, not a
second SCALE: its index set is the same N tokens and its prior is a token-uniform `(K,)` broadcast.
Reusable pieces are `transport_mean`/`transport_covariance` (rank-agnostic, would serve `Omega_{I,j}`
unchanged) and `mix_dispersion`. The no-neural-network constraint survives (pooling weights are
softmax-of-energy, the emission is a table lookup) and the pure path survives iff the new seam
defaults to `per_token`.

One cheap trick worth recording if this is ever revisited: **gauge-fix the parent frame `U_I = I`**.
Then `Omega_{i,I} = U_i`, which is already computed, and the shadow costs ZERO extra `matrix_exp` —
the dominant per-step cost. Of the causal formation options, fixed non-overlapping chunks with the
parent visible only to later chunks is exact and cheapest (about +2.9% step time at span 8, and the
moment-pooled barycenter is exact because the transport factorizes); prefix-only EMA barycenters are
cheapest in attention but worst in `matrix_exp` (one parent per position, +23%) and provide no
coarse-graining at all.

## 6. Plan

The panel converts a proposed architecture change into a mechanism question with cheap answers. Do
these in order and stop as soon as one settles it.

**Stage A, free (checkpoint-only, ~one eval pass each).** All four measurements target Section 2's
mechanism and are jointly decisive about it.

1. Cross-position dispersion of the body-frame means `U_i^{-1} mu_i` at depths 0,1,2,3,5,8. Monotone
   collapse confirms consensus; flat dispersion refutes it.
2. `selfdiv_klmax_frac` (already implemented in `metrics.py`) at the same depths. Rising toward 1
   confirms the `1[D < kl_max]` anchor-dropout route.
3. Effective rank and mean pairwise cosine of `mu` across depths (over-smoothing signature).
4. The `alpha* = 1/(1+D)` trajectory across depths, to quantify anchor decay directly.

Re-run the depth sweep with `kl_max = 1e6` and full-batch F. Two known defects in the existing
artifact must be fixed while doing so: F is computed on sequence 0 only while CE is over all 16
sequences, and `estep_fp_kl = 8.2e-4` in the same run says one extra E-step moves the belief by
8e-4 nats, which is hard to reconcile with a +0.30-nat CE jump from depth 1 to 2. Resolve that
inconsistency before drawing any conclusion from the curve.

**Stage B, one training arm.** The off-distribution objection is prior to everything else: the sweep
varies inference depth at weights trained with `n_e_steps=1`. Retrain one K=20 arm at `n_e_steps=4`
(or randomized 1-4) and re-run the sweep on both. If argmin-CE tracks the trained depth, the
anti-alignment is an off-distribution artifact and the entire motivating story dissolves. If
argmin-CE stays at 1 in both arms, the objective is structurally anti-aligned and Stage A tells us
which mechanism.

**Stage C, only if A and B implicate the anchor.** The remedy is then an anchor-schedule question,
not an architecture: the `alpha` form (a floor on `alpha*`, or `b0`/`c0`), the `kl_max` self-gate,
and the `lambda_beta`/`alpha` ratio. All are existing config toggles. The panel's prediction is that
the anti-alignment slope scales monotonically with `lambda_beta/alpha`, which is directly testable.

**Not recommended:** building the shadow prior as a fix for this pathology. It replaces the only
non-averaging anchor with an average (Section 2), the token cannot be both prior and observation
(Section 3), the construction is a known architecture with a poor track record on text (Section 4),
and no gauge degeneracy exists for it to break (Section 1).

## 7. Agreed degeneration warning signs

Recorded in advance so they can be refused later. The programme would be degenerating if: the shadow
prior ships and is defended by "the whitepaper derives it" rather than a perplexity delta; after a
null result `n_layers`, `rho` and `kl_max` are retuned until the depth curve looks right and that is
re-declared confirmation; "data term" is never given a measurable signature distinct from a
re-parameterization; or the depth-0 point stays out of the plots.

The base rate is relevant: of the top-ranked mechanism hypotheses in the Stage 0 report, five of five
were refuted by cheap measurement. This one deserves the same treatment before any code is written.
