# GL(K) attention manuscript — deep review pass 8 (verification pass: the last open analytic items)

Date: 2026-06-20. Targets: the canonical `Research/manuscripts/GL(K)_attention.tex` and `GL(K)_supplementary.tex`.
Pass 8 closed out the three load-bearing equations that the pass-7 completeness critic had flagged as never
independently recomputed end-to-end, and ran a fresh adversarial sweep over still-under-examined material seeded with
the verified-ledger do-not-reflag list. The workflow dispatched nine agents (~0.53M tokens): three end-to-end
recomputations, five expert-lens sweeps, and a completeness critic; every load-bearing identity was recomputed in
python (numpy/sympy or finite difference) rather than eyeballed.

Headline: after eight passes the manuscript is mathematically exhausted at the level of its analytic derivations. The
three target equations are correct to machine precision, the sweep produced zero new high/critical findings, and the
completeness critic — after recomputing five further blocks fresh — reports no remaining un-recomputed load-bearing
identity. The residue is entirely expository, citation-accuracy, or already-flagged empirical/data-pending material.

## The three pending items — all confirmed correct

1. **Meta-agent moment matching `Sigma_A` (eq:meta_agent_beliefs / eq:meta_agent_beliefs_supp, supp ~950).**
   `Sigma_A = (1/|A|) sum_i Sigma_i + Var_A(mu)` is exactly the covariance of the equal-weight Gaussian mixture
   `(1/n) sum_i N(mu_i, Sigma_i)` — the Euclidean law of total covariance (mean within-token covariance plus dispersion
   of the means). Symbolic residual identically zero; Monte Carlo deviation is pure sampling noise decaying as
   `1/sqrt(N)` (rel error `1.17e-3` at `N = 8e6`). The population `1/|A|` normalization is the unique correct
   moment-matching choice — the Bessel-corrected `1/(|A|-1)` variant carries a systematic ~4% bias and a nonzero
   symbolic residual, so the manuscript's biased normalization is right, not a defect. `mu_A` is the exact mixture
   first moment, and `g_1^(emer) = ||Var_A(mu)||/sigma^2` is the anisotropic component of the same `Sigma_A`, normalized
   consistently with `g_1^(orig)`, same sign. Prose accurate.

2. **ALiBi / T5 / sliding-window / causal logit reductions into eq:mixture_softmax_general (attn ~758–840).**
   Substituting each prior `pi_j` reproduces the published mechanism exactly: the log-prior enters the logits
   additively as `+log pi_j` with NO `1/tau` factor (correct, since `beta* = pi_j exp(-E_ij/tau)/Z`; dividing the bias
   by tau is wrong and deviates `0.04–0.18` in tests). Signs correct in every case (ALiBi `-m|i-j|` penalizes distant
   keys; T5 `+b_{i-j}` adds a learned offset). ALiBi matches Press et al. 2022 (their causal `-m(i-j)` equals `-m|i-j|`
   on the causal support); T5 matches Raffel et al. 2020 (additive bucketed relative-position logit bias). Slope
   schedule `m = 2^{-8h/H}` re-confirmed (first term and common ratio both `2^{-8/H}`). Elementwise residuals
   `5.6e-17`. One low-severity expository note recorded below.

3. **Frobenius-pullback natural-gradient metric + `Psi(ad_phi)` (eq:pullback_metric, supp ~606–621).**
   The right-trivialised Frechet derivative `D_phi(exp)[T_a] = Psi(ad_phi)(T_a) . exp(phi)` matches a finite-difference
   Frechet derivative to `~1e-10` (K=3,4); the trace-cyclicity collapse to eq:pullback_metric is exact; all three
   limits hold to `~1e-14` (`phi=0` gives the Gram matrix; skew gives `I`; symmetric gives `exp(2phi)`); the
   left/right operator identity `Psi_R = e^z Psi_L` is symbolically exact; `Psi(z) = (e^z-1)/z = sum z^k/(k+1)!` and its
   inverse `z/(e^z-1)` is genuinely the Bernoulli generating function (the `B_1 = -1/2` classical convention; sympy's
   `+1/2` is the only differing term, a convention artifact). `G(phi)` is a true pullback Gram `J^T J`, hence symmetric
   PSD by construction. The manuscript's own caveat ("exact up to series truncation of `Psi`") is real, and the
   verifier confirmed the implied degeneracy: at `2*pi*i` integer eigenvalue resonances of `ad_phi`, `Psi(ad_phi)` is
   singular and `G` loses rank (`9 -> 3` at `theta = 2pi`), so `G^{-1} grad` is ill-posed exactly there. No false
   statement is made; an optional one-clause warning is recorded below.

## Additional blocks recomputed fresh by the completeness critic (none previously in the ledger; all correct)

- **phi-gradient / KL-through-transport chain** (App C, supp eqs:phi_grad_complete 408–419, reverse_beta_grad_phi,
  beta_grad_phi, mean/trace/logdet_term_phi 515–538, dtilde_mu/dtilde_Sigma). Finite-difference (K=3, full gl(K)
  basis): `eq:beta_grad_phi` max `|FD - analytic| = 6.8e-11`; `eq:reverse_beta_grad_phi = 1.7e-10`; combined
  mean+trace+logdet KL-through-transport gradient with `Q_a = D_phi(exp)[T_a] e^{-phi}`, rel residual `5.0e-10`.
  Includes the second-order `-Sigma^{-1}(dSigma)Sigma^{-1}` term and the right-trivialised `Q_a`. The ledger had the
  mu- and Sigma-gradients of `F`; the gauge-frame (phi) gradient through the right-trivialised dexp was a distinct,
  previously unverified derivation.
- **Cartan/Killing-form preconditioner block** (supp eqs:cartan_decomposition, P_sym projector, killing_metric). GL(K)
  Killing form `B(X,Y) = 2K tr(XY) - 2 tr(X)tr(Y)` matches the genuine `tr(ad_X ad_Y)` at K=2,3,4 (max diff
  `3.6e-15`); `tilde_g_ab` is positive-definite on `sl(K)` (min eig `4.0/6.0/4.69` for K=2/3/4) and degenerate on the
  center (exactly one zero eigenvalue); `P_sym = (1/2)G^{-1}(G+S)` is idempotent (`||P^2 - P|| = 0`). The Ad-invariance
  iff-condition (`tr((gXg^{-1})^T(gYg^{-1})) = tr(X^T Y)` for all X,Y iff `g^T g` scalar) holds in both directions
  numerically.
- **Geometric-mean stationarity 1/2 exponents** (supp eq:q_i_star_supp): re-derived `dL/dH = -1/2`, `dL/dg_j = b_j/2`
  symbolically, confirming `q_i* ∝ e^{-H/2} prod (Omega q_j)^{beta/2}`.
- **RoPE relative composition** (attn eq:rope_relative 1875, eq:rope_lie_algebra 1894): `R(theta_i)^T R(theta_j) =
  R(theta_{j-i})` (residual `2.2e-16`); `exp(phi_i - phi_j)` and `exp(phi_j - phi_i)` exact transposes/inverses on the
  abelian `SO(2)^{d/2}` subgroup.
- **Off-diagonal codimension arithmetic** (attn 1826): `512^2 - 8*64^2 = 229,376 = 87.5%`; sigmoid `n_s = 2` reduction
  exact (`5.6e-17`).

## Applied this pass

One edit, an internal-consistency fix in the canonical attention manuscript.

1. **Value-aggregation gloss (attn line 1043).** The forward-reference called `hat_mu_i = sum_j beta_ij Omega_ij mu_j`
   "the posterior mean of the mixture **variational model** under responsibilities `beta_ij`". The mean-field
   variational posterior `Q(k,z) = q_i(k) beta(z)` is a product, so its `k`-marginal is `q_i` and `E_Q[k] = mu_i`, not
   `hat_mu_i`. The object `hat_mu_i` is the responsibility-weighted mean of the **generative** mixture components, which
   is exactly how the careful derivation at `sec:value_aggregation` (line 1327) names it ("the posterior mean under
   mixture responsibilities... the standard mixture-of-Gaussians expectation... consistent with the generative
   model"). Changed `mixture variational model` -> `generative mixture` so the one-liner agrees with the section it
   cites. Minimal substitution; the equation and the downstream `hat_mu_i = sum_j beta_ij V_j` are untouched.
   - old: `... is the posterior mean of the mixture variational model under responsibilities $\beta_{ij}$ (derived formally in Section~\ref{sec:value_aggregation}):`

## Recommended items — applied on user request

These are real but minor; none is a defect, and several are contestable empirical framing. After the pass-8 review the
user directed that the recommended items be applied; all were applied to the canonical vault manuscripts, with two
qualifications: the Knapp reference was repointed at chapter granularity (`[Ch.~VI]`) because the exact
equation/corollary was not re-verified from the primary source here, and the RG `y3` fit-window item is data-pending,
so it was recorded in the in-source `DATA-PENDING` TODO rather than edited in the displayed numbers.

Applied (canonical vault manuscripts):

- supp:596 — Knapp bibref `[Prop.~1.93]` -> `[Ch.~VI]`.
- supp:878 — temp-dispersion "ranking anti-correlates... systematically" -> "negatively associated (Pearson `r=-0.87`, Spearman `rho=-0.6`)".
- supp:924 — key-norm `0.256` relabeled the between-head mean of absolute correlations, distinguished from the signed population magnitude `0.475`, with a Jensen note.
- supp:588 — Cartan projector `P_sym` noted to retain the center/dilation direction (`P_sym(I)=I`, full `Sym(K) (+) R.I`).
- supp:621 — pullback-metric `2*pi*i`-resonance degeneracy of `G` plus the `G + eps*I` regularization clause.
- attn:778 — ALiBi/T5/window reductions cross-referenced to `eq:F_align_canonical_tau` for the `/tau`.
- supp:1023 (TODO comment) — RG `y3` fit-window inconsistency recorded; displayed numbers unchanged (data-pending).

- **Knapp citation may be mis-pointed (supp line 596, medium).** `g(X,Y) = -B(X, theta(Y))` with `theta(X) = -X^T` is
  attributed to `\citep[Prop.~1.93]{Knapp2002}`. The Cartan-involution definition and the positive-definiteness of
  `B_theta(X,Y) = -B(X, theta Y)` live in Knapp Chapter VI (Structure Theory of Semisimple Groups), section 2, around
  eq. (6.13) and Cor. 6.18, not in the Chapter I numbering `1.93`. A web check corroborates the Chapter VI location of
  the Cartan-involution material; the sweep agent further reports (from a PDF extraction this review could not
  independently re-fetch) that Prop. 1.93 is a purely topological-groups statement. The algebra of eq:killing_metric is
  verified correct this pass — only the cross-reference is suspect. Recommend the author verify against their copy and
  repoint to Knapp Ch. VI sec. 2 (eq. 6.13 / Cor. 6.18). Not applied because the exact replacement number was not
  confirmed from a primary source here.

- **RG graph-based `y3` fit window (supp Table tab:rg_flow_supp / line 1041, medium, data-pending).** The printed
  `y3 = +0.17` reproduces only from a levels **1–5** log-log fit; the caption states the `y3` fit uses levels **0–5**,
  which gives ~`+0.6` to `+0.8` from the printed column. (`y2 = -0.66` is consistent with the full levels 0–6 window,
  not 0–5.) The two printed exponents therefore use mutually inconsistent fit windows, and `y3` contradicts its own
  caption. This table already carries an in-source `DATA-PENDING` TODO (the `g1_tot` column violates
  `g1_tot = g1_orig + g1_emer`), so the fix belongs to the same blocked data re-extraction: state the exact level
  window per exponent and make them consistent against the finalized CSV. Numbers not edited (blocked on data).

- **Temperature-dispersion "ranking anti-correlates" (supp line 878, low).** Recomputed: Spearman rank correlation
  between the temp-dispersion CV and `r@19` is `rho = -0.6` (Pearson `-0.87`). The claim "the ranking by temperature
  dispersion anti-correlates with the ranking by correlation... systematically worse" is directionally supported but
  mildly strong for `rho = -0.6` (not a strict inversion). The sweep's specific counterexample (DistilBERT) is in fact
  consistent with anti-correlation, so the claim is not false. Optional: report the actual Pearson `-0.87` and soften
  "ranking... systematically" to "negatively associated", or state the Spearman value. Author's framing call.

- **Key-norm `0.475` vs `0.256` phrasing (supp line 924, low).** Calling `|hat_mu_beta| = 0.475` "the larger
  between-head average" that the frequentist `|rho|_beta = 0.256` "shrinks toward" risks conflating two different
  between-head summaries (a partially pooled signed population-location parameter vs a raw mean of per-head absolute
  correlations). The text already hedges "not directly comparable". Confirming whether `0.256` is itself a between-head
  average requires reading `sec:keynorm_bias_supp`; raised as a question for the author rather than edited.

- **Cartan projector scope (supp lines 585–591 vs eq:cartan_decomposition 573/577, low, off released path).** `P_sym =
  (1/2)G^{-1}(G+S)` maps `X -> (1/2)(X + X^T)`, the full symmetric part including the trace/center direction
  (`P_sym(I) = I`, verified), whereas eq:cartan_decomposition defines `Sym(K)` as traceless and lists the center as a
  separate summand. The arm is explicitly not wired into the released path, so the impact is expository. Optional: state
  that `P_sym` dampens both the traceless shears and the center, or subtract the trace projector if only `Sym_0(K)` is
  intended.

- **ALiBi/T5 reduced-form cross-reference (attn ~791, ~819, ~836, ~845, low).** The displayed reduced softmaxes carry
  `E/tau`, while the equation they cite, eq:mixture_softmax_general (~757), is written tau-free; the `1/tau` enters via
  the WLOG rescaling at eq:F_align_canonical_tau (~768). Optional tidiness: cite eq:F_align_canonical_tau alongside so
  the `/tau` is attributed to the equation that introduces it. Zero effect on the math.

- **Pullback-metric resonance caveat (supp eq:pullback_metric, low, optional).** Add one clause noting `G(phi)`
  degenerates when `ad_phi` has a nonzero eigenvalue in `2*pi*i*Z` (so the `G^{-1}` solve should be regularized near
  such resonances), mirroring the `epsilon`-regularization already noted for the Killing form.

## Verification

Eight edits total were applied to the canonical vault manuscripts this pass: the value-aggregation gloss (attn:1043)
during review, and the seven recommended items on the user's request (attn:778; supp:596, 588, 621, 878, 924, and the
1023 TODO comment). All edits are expository, citation, or documentation changes: no equation, label, exponent, or
displayed numerical value was altered. The temperature-dispersion `r = -0.87` (Pearson) was independently recomputed
by hand from the printed table before insertion, and `0.256` was confirmed from supp:785 to be the between-head mean
of absolute correlations. No banned patterns (horizontal rules, the claudeism list) were introduced — the one `---`
initially placed in the supp:1023 TODO comment was changed to a colon; the `$\alpha$--$\beta$` compound matches the
manuscript's existing notation. All numerical residuals above are from the pass-8 workflow agents' python
recomputations.
