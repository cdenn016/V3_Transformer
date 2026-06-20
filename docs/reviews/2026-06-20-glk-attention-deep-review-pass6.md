# GL(K) attention manuscript — deep review pass 6 (mathematical/theoretical, recompute-driven)

Date: 2026-06-20. Targets: the canonical `Research/manuscripts/GL(K)_attention.tex` and `GL(K)_supplementary.tex`.
This pass answered the request for a genuinely DEEP referee review of the mathematics, after pass 5 leaned toward
notation and citation hygiene. Seven expert lenses (gauge theory, SPD/differential geometry, information geometry,
variational/free-energy, transformer-ML, numerical analysis, proof rigor) were each required to RECOMPUTE every
load-bearing identity by Bash + python rather than eyeball it, and every finding was put to an independent adversarial
verifier who re-derived it (three-voter majority on high/critical). 16 raw findings, 10 confirmed, 6 rejected.

## Big leads checked and CLEARED in GL(K)

A parallel deep review of the companion `PIFB.tex` had just confirmed three real errors there. The lenses checked
whether GL(K) shares them, and recomputed the five previously unexamined critic gaps. The good news is concrete:

- T1 (Gaussian / Fisher-Rao curvature). A grep across both files found GL(K) makes no "nonpositive / negative
  sectional curvature of the Gaussian manifold" claim, so the mixed-sign error that the companion carries (the full
  `N(mu,Sigma)` manifold has `+1/4` curvature on pure-mean planes at `Sigma=I` for `K>=2`) is absent here.
- T2 (Wilson action on noncompact GL+(K)). The action `S = beta sum (1 - W_ijk/K)` is unbounded below for the
  noncompact group, but GL(K)'s passage at line 665 already carries the compactness caveat; the only residual is that
  the minimum-at-`H=I` claim is not restated at the per-head-holonomy site, a low-severity note.
- T3 (ALiBi head-slope). The slope expression at line 823 was recomputed correct for general head count, not only
  `H=8`.
- G1 (untied-QK joint realizability). Recomputed: the gauge bilinear factors as `M_ij = A_i B_j` with `A_i` an
  i-only and `B_j` a j-only factor, so the family is genuinely rank-structured. The "expressive power identical to
  learned `W_Q W_K^T`" thesis is proven only per-pair; the gap to joint realizability of an arbitrary family `{M_ij}`
  is a minor imprecision in the closing analogy, not a defect (the verifier rejected the strong framing).
- G2 (per-head temperature factor of two). Recomputed: `eq:per_head_temperature` divides the full KL (which already
  carries the `1/2`) by `kappa_a sqrt(d_head)`, so `kappa_a = 1` gives an effective `2 sqrt(d_head)` on the squared
  distance, which equals the dot-product `sqrt(d_head)` softmax to `4.4e-16` under the constant-key-norm condition the
  text invokes. Internally consistent, no defect.

## Confirmed findings

### Applied this pass (settled factual corrections)

The key-norm-bias effect size Cohen's `d = 1.43` is restated in the "Standard attention as a degenerate limit"
summary at line 2270 as one of the "non-trivial quantitative predictions" without the non-convergence caveat it
carries at line 2041 and in the supplement (lines 906, 922). The supporting chain is poorly mixed: the population
scale `sigma_beta` has `R-hat = 1.09` and `ESS_bulk = 18`, propagating to a `d`-posterior with `ESS_bulk = 44`, far
below the few-hundred effective draws needed for a stable HDI. The caveat was restored at line 2270 with a pointer to
the BERT summary (medium).

The per-head holonomy paragraph at line 1788 stated that a single `||H_ijk - I||_F` evaluated on the full
block-diagonal holonomy "report[s] the sum of per-head deviations." For a block-diagonal matrix the squared Frobenius
norm is additive across blocks, so the aggregate is the root-sum-of-squares of the per-head deviations, not their
arithmetic sum (a numeric check gave `2.084` versus the naive sum `3.592`). Corrected; the additive Wilson-trace claim
`W_ijk = sum_a W_ijk^(a)` in the same paragraph was independently confirmed correct and left untouched (low).

The main text at line 1481 cited "Supplementary Appendix~D for derivation" of the inverse Fisher mean preconditioner
`G_mu^{-1} = Sigma`, but that appendix states the result without deriving it. The one-line score-covariance derivation
was added inline (the mean score is `Sigma^{-1}(x-mu)`, so `G_mu = Sigma^{-1} E[(x-mu)(x-mu)^T] Sigma^{-1} = Sigma^{-1}`),
verified symbolically (low).

The entropy ratio was written `H(beta)/H(alpha) = 1.076` at lines 2039 and 2270, which reads as the ratio of the two
displayed corpus-mean entropies, but `1.784/1.774 = 1.006`; the figure `1.076` is the per-head mean of the ratio. The
overline notation was restored (matching the supplement's displayed equation at line 814) and the ratio-of-means value
noted at line 2039 (low).

### Surfaced for author decision — subsequently APPLIED on user confirmation

The four items below were presented to the author, who directed that all four be applied with the causal mask left as
D. Applied: ALiBi, T5, and sliding-window Table 1 rows reclassified D to S (causal mask and RoPE kept D); the "L layers
as L natural-gradient steps" paragraph reframed as a structural correspondence with the tied-weight inner loop named as
the genuine single-functional iteration; the App-H reverse implication given its normalizability assumption and the
open-interval sweep derived from the geometric-mean form; and the line-2039 temperature relabeled from "theory-predicted
optimal temperature tau = 19.0" to the empirical optimum 19.0 with the theory value 2 sqrt(d_k) = 16 stated alongside.
The original wording of each is recorded below.

The four positional-bias rows in Table 1 (causal mask, ALiBi, T5 relative bias, sliding window) are stamped "D"
(derived), but the table's own legend reserves "S" for a correspondence whose specific form the framework "does not
uniquely predict," and the prose at line 840 concedes the framework supplies only the additive-log-prior form, not
which prior. Three lenses independently flagged this as the same choose-the-prior move that pass 4 reclassified from D
to S for layer normalization. It is held for author judgment because the relabeling is a deliberate status decision and
because the causal mask is the debatable case: line 883 argues it follows from the autoregressive generative
factorization (a genuine derivation), whereas ALiBi's slope `m`, T5's learned table, and the window width `w` are
clearly chosen. Recommended resolution: ALiBi, T5, and sliding window to "S"; causal mask to the author's call.

The "Layers as VFE iterations" paragraph (lines 1685-1686) states that a standard `L`-layer transformer "corresponds
to `L` natural-gradient steps" of free-energy minimization. With independently parameterized per-layer weights each
step descends a different functional `F^(l)`, so the trajectory is not iterated minimization of one `F` and need not
reduce any single functional; only the tied-weight inner loop is genuine iterated minimization. A reframing that
labels the cross-layer count a structural correspondence (matching the residual subsection's own candor and the Table 1
"S" status of the Euler/residual row) is ready.

The reverse implication of the Conditional Uniqueness Theorem (supplement lines 1287-1292) sets all transported
neighbors equal and lets `p_i` vary so that `q_i^*` "sweeps a range of values," concluding the likelihood ratios
contain an open subinterval of `(0, infinity)`. Two domain conditions are used silently: that `Omega_ij^{-1} qbar` is a
normalizable density in the admissible neighbor-belief family, and that the sweep is genuinely open. The surrounding
algebra was verified sound (`f(t) = t log t - t + 1`, `f'(1) = 0`, the WLOG shift `f -> f + c(t-1)` is divergence-
preserving); only the realizability/openness link is asserted. A clarifying clause is ready.

The BERT temperature at line 2039 is written "theory-predicted optimal temperature `tau = 19.0 ~ 2 sqrt(d_k)` for
`d_k = 64`," but `2 sqrt(64) = 16`, not 19; 19.0 is the empirical optimum (the supplement treats 16 as the theory value
and 19.0/21.2 as the empirical/posterior optimum). A framing fix distinguishing the empirical optimum from the theory
value is left to the author (this item was raised by the completeness critic and not put through adversarial
verification).

## Rejected on verification

GT-1 (per-head temperature off by a factor of two): rejected, the two conventions are the same distribution under the
key-norm condition (verified to `4.4e-16`). VI-1 (query-side update breaks an invoked Neal-Hinton monotonicity):
rejected, the manuscript never invokes monotone descent and already caveats that the local update is not the full
gradient of `F_red`. TML-3 / G1 (untied-QK family realizability as a defect): rejected, the per-pair claim is correct
and the family is rank-structured by construction; only a minor closing-analogy imprecision remains. GT-2 and
perhead-holonomy-vacuous (the "vacuous as written" subsection ships a derivation): rejected as presentation nits whose
mathematics is correct and whose scope the text already attributes to the companion. GT-3 ("irrep block" mislabels the
Levi subgroup): rejected, the `H` blocks are pairwise inequivalent irreps under the acting `GL(d_head)^H`, so the
decomposition is genuine.

## Completeness critic — still unexamined

The critic flagged, for a possible later pass, the per-pair key-side bias absorption in the central QK reduction (line
1186, `r_j = ||K_j||^2` absorbed into `log pi_ij` then assumed approximately constant under layer norm) as the same
choose-the-prior move applied to the headline derivation, and recorded the cleared T1/T2/T3/G1/G2 items so a later pass
does not re-flag them.

## Applied this pass

`GL(K)_attention.tex`: 1481 (App-D Fisher derivation inline), 1788 (Frobenius root-sum-of-squares), 2039 and 2270
(entropy-ratio overline plus ratio-of-means note), 2270 (Cohen's d non-convergence caveat).

Verification: braces balance (3979/3979); zero LaTeX spacing macros; no claudeisms; `\S\ref{sec:bert_summary}`
resolves; the overline is restored at both entropy sites and the old bare literal is gone; the IG-1 inline derivation
was sanitized of an agent-proposed `\,` spacing macro before applying.
