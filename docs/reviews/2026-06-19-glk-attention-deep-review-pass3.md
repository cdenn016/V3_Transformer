# GL(K) attention manuscript — deep review pass 3 (code-fidelity, secondary derivations, RG g2)

Date: 2026-06-19. Branch: `vfe3-per-layer-figures`. Targets: `Manuscripts-Theory/GL(K)_attention.tex`,
`Manuscripts-Theory/GL(K)_supplementary.tex`, `Manuscripts-Theory/references.bib`, cross-checked against the
released `vfe3/` tree and the research wiki.

## Method and scope

This is the third review pass. Pass 1 (`2026-06-19-glk-attention-wiki-peer-review.md`) covered literature and
citation gaps against the wiki and applied the five big connections (Sengupta-Friston, Dong, Geshkovski,
Mehta-Schwab, Beny-Osborne). Pass 2 (`2026-06-19-glk-attention-deep-derivation-addendum.md`) verified the seven
load-bearing proofs line by line and applied the proof fixes recorded in `DERIVATION_VERIFICATION.md`. This pass
deliberately targets the territory those two did not: manuscript-versus-code fidelity in the released
implementation, the secondary derivations outside the seven, internal consistency and cross-reference integrity,
claim-strength calibration in the framing sentences, a second wiki-completeness sweep, and a dedicated resolution
of the open RG `g2` scaling exponent.

The pass ran as a six-lens multi-agent workflow (implementation-engineer, geometer, internal-consistency,
philosophy-of-science, wiki-second-sweep, info-geometer), with every substantive finding handed to an independent
adversarial verifier instructed to refute it from primary sources before it could be reported. Tallies: 30 raw
findings, 20 substantive, of which 13 survived verification, 7 were rejected, and 10 minor items were logged
without adversarial challenge. The verification layer earned its place: it overturned three findings that had been
flagged "high" severity (an alleged abelian collapse of the holonomy, an alleged invalid coarse-grained transport,
and an alleged unfalsifiable holonomy-compositionality bridge), each on direct evidence that the original lens had
misread the notation, the code, or a section it had not read.

## Summary

The headline is that several of the supplementary's implementation citations name functions, files, and algorithms
that do not exist in the released tree, while the body's own derivations and exponents survive intact. The newly
confirmed defects are concentrated in documentation and reproducibility prose rather than in any theorem: four
code-fidelity catches (a phantom `sanitize_sigma`, a phantom `retract_to_principal_ball`, a misnamed `connection.py`
"MLP mode", and a stale `kappa ∝ sigma^2` glossary entry), four broken or under-defined cross-references and a
missing-citation pair, plus one genuinely load-bearing exposition gap in the RG section where the correct
`y2 = -1` exponent is left unmotivated in the main text. None of these overturns a result; all are actionable with
targeted edits. The adversarial verification cleared away several would-be high-severity claims, each of which
rested on a misreading of the manuscript or the code.

## Confirmed findings

### Code-fidelity

#### Phantom `sanitize_sigma` in the SPD-sanitization paragraph

Location: `GL(K)_supplementary.tex:649-654` against `vfe3/numerics.py` and `vfe3/geometry/retraction.py`.

The supplementary describes a post-retraction safeguard named `sanitize_sigma` that "symmetrizes the result, raises
on any floating-point anomalies (NaNs), and applies a spectral floor" with `epsilon_SPD = 1e-6` and
`sigma_max = 10`. No such function exists anywhere in the released tree; a whole-tree search for `sanitize` returns
nothing. The actual SPD clamp is inline in the retraction kernels, namely `sigma_new.clamp(min=eps, max=sigma_max)`
at `retraction.py:131` and `eig_new.clamp(min=eps, max=sigma_max)` at `retraction.py:187`, and the only
eigenvalue-floor utility that exists is `floor_eigenvalues` in `numerics.py`, which symmetrizes and clamps
eigenvalues up to a floor but applies no ceiling and does not raise on NaNs. The named NaN-raising guard therefore
has no counterpart in code, and the quoted `sigma_max = 10` matches neither retraction kernel default (both `5.0`);
the value `10` lives only in `config.py:349`. Recommendation: replace the `sanitize_sigma` prose with the real
path, the inline `clamp(min=eps, max=sigma_max)` inside the retraction together with `floor_eigenvalues` from
`numerics.py`, and delete the "raises on NaNs" claim. Either rename the prose to the existing functions or rename a
code function to match the paper, but do not cite a function absent from the release.

#### Phantom `retract_to_principal_ball` and a modulo-2π/antipodal algorithm the code never executes

Location: `GL(K)_supplementary.tex:672` (and the parallel SO paragraph at `:561`) against `retract_phi` at
`retraction.py:379` and `lie_ops.py:374,397`.

The supplementary states that angle retraction proceeds via `retract_to_principal_ball`, "which wraps angles modulo
2π, applies an antipodal flip for angles exceeding π, and clamps the result to ||phi|| < π - eps_margin". A
whole-tree search finds no `principal_ball`, `antipodal`, `fmod`, or `% (2*pi)` anywhere in `vfe3/`. The real entry
is `retract_phi`, which dispatches on `group.skew_symmetric`: the non-skew GL branch calls `retract_glk` with
`max_norm = 5.0`, the skew SO branch calls `retract_son` with `max_norm = pi`, and both route through
`_retract_core`, whose clamp is a continuous radial Frobenius rescale `phi_new * (max_norm / (n_norm + eps))` with
no modulo, no `fmod`, and no `phi -> -phi`. The fabricated algorithm appears twice, at `:561` (SO-scoped) and
`:672`, so both sites need correction. The default headline group `block_glk` is non-skew, so the reported
GL(15)/GL(10) language runs exercise the GL Frobenius-clamp branch, and the manuscript's own GL paragraph at line
563 already states correctly that "no retraction is required beyond standard gradient clipping" for GL.
Recommendation: rewrite both paragraphs to describe `retract_phi` as a group-aware Frobenius trust-region step plus
a max-norm clamp (GL `5.0`, SO `pi`) with optional `sl(K)` projection and trace clamp, and drop the modulo-2π,
antipodal-flip, and principal-ball language entirely.

#### Regime II misnamed as a `connection.py` "MLP mode"

Location: `GL(K)_attention.tex:2288` and `:2365` against `transport.py:200-328` and `model.py:216`.

The manuscript describes the Regime II edge-relaxed cocycle as "accessed in the codebase through the
`connection.py` MLP mode," a phrase that recurs in the "no neural networks" scope paragraph at `:2365`. There is no
`connection.py` in the tree, and the mechanism is not an MLP. The Regime II edge factor is assembled in
`_build_regime_ii` (`transport.py:200`) from a bilinear form `delta_ij^a = mu_i^T W^a mu_j`, where `connection_W`
is a raw `nn.Parameter` created only under `transport_mode == 'regime_ii'` (`model.py:216`); a search of
`transport.py` for `MLP`, `nn.Linear`, `nn.Module`, or `nn.Sequential` returns nothing, and no MLP class exists
anywhere in `vfe3`. Both occurrences misname a nonexistent file and mischaracterize a single bilinear parameter as
a multilayer perceptron, sending a reader hunting for an absent file and an absent architecture. Recommendation:
change both sites to something like "accessed through `transport_mode='regime_ii'`, which inserts a learned
bilinear edge connection `delta_ij^a = mu_i^T W^a mu_j` (`connection_W`, an `nn.Parameter`, default OFF),
gauge-invariant only at `W = 0`," removing the `connection.py` and MLP wording.

### Consistency

#### Glossary entry `kappa ∝ sigma^2` contradicts the implementation paragraph and the code

Location: `GL(K)_attention.tex:409` against `:1793-1800` and `free_energy.py:41-76`, `config.py:181`,
`divergence.py:12-14`.

The notation table writes "`kappa (tau)` ... Attention temperature; `kappa_a ∝ sigma_a^2` in isotropic limit," but
the implementation paragraph at line 1800 states the opposite in its own voice: `kappa_a` is "a dimensionless
sharpness handle independent of `sigma_a`," and "identifying `kappa_a` with `sigma_a^2` as well would double-count
the covariance scale, since the energy already carries the factor `sigma_a^{-2}` once." The code matches line 1800:
`attention_tau` returns `kappa * sqrt(d_energy)` with `kappa = cfg.kappa_beta` defaulting to `1.0`, a
sigma-independent scalar, while the diagonal-Gaussian KL in `divergence.py:12-14` already carries the entire sigma
dependence. Reading `kappa ∝ sigma^2` while the numerator KL also depends on sigma double-counts the covariance
scale, the exact failure mode the temperature handle is meant to avoid; line 409 is a stale relic of an older
derivation in which `kappa` absorbed `sigma^2`. The hedge "in isotropic limit" conflates the temperature `tau`
(which does scale with `sigma^2` via the energy) with the scalar handle `kappa`, which does not. Recommendation:
a one-line glossary edit deleting the `kappa_a ∝ sigma_a^2` clause and stating `kappa_a` as the sigma-independent
per-head sharpness scalar with `tau_a = kappa_a sqrt(d_head)`, so the table, the implementation paragraph, and the
code agree.

#### Broken cross-reference `\ref{sec:rope_gauge}`

Location: `GL(K)_attention.tex:2046`, with the intended target labelled `sec:rope` at `:1849`.

Line 2046 reads "(see Section~`\ref{sec:rope_gauge}`)," but no label `sec:rope_gauge` exists in either file; the
RoPE subsection at line 1849 is labelled `\label{sec:rope}`, and the equation label `eq:rope_gauge` at line 1885 is
a display-equation label, not a section. Two sibling references at lines 366 and 2024 already use `\ref{sec:rope}`,
establishing it as the canonical target. The reference renders as "??" with an undefined-reference warning but does
not break compilation, so this is a cosmetic copyedit. Recommendation: change `\ref{sec:rope_gauge}` at line 2046
to `\ref{sec:rope}`.

#### Dangling `\ref{fig:glk_pca_frames}`, cited twice, with no figure present

Location: `GL(K)_attention.tex:2251` and `:2255`; no matching `\label` in either file.

Both lines support the gauge-frame PCA and categorical-separation discussion with "Figure~`\ref{fig:glk_pca_frames}`,"
yet no `\label{fig:glk_pca_frames}` exists. The main file's complete figure-label set is exactly
`bundle_sections_surface`, `glk_attention`, and `glk_training`, the supplementary's nine figure labels do not
include it, no PCA `\includegraphics` appears in either file, and the main file has no `\input`/`\include`
directives, so the figure cannot arrive from an external source. This is a missing figure rather than a mislabel:
both references compile to "??" while the prose ("punctuation tokens separate sharply along PC1," "visible in
Figure~...") points at an absent plot. Recommendation: either insert the PCA gauge-frame figure with
`\label{fig:glk_pca_frames}` or, if the figure was cut, rewrite the two sentences to drop the figure reference,
since the surrounding ANOVA, silhouette, and Calinski-Harabasz numbers stand on their own.

#### Broken `\S\ref{sec:keynorm_bias_supp}` in the supplement

Location: `GL(K)_supplementary.tex:919`, with the intended target the subsection at `:764`.

Line 919 cites "the head-level frequentist estimate cited in `\S\ref{sec:keynorm_bias_supp}` as
`|rho|_beta = 0.256`," but no label `sec:keynorm_bias_supp` exists; the "Key-Norm Bias and Layer Normalization"
subsection at line 764, where the value `0.256` is actually reported at line 780, carries no `\label`, and the only
nearby labels are figure labels. The supplement is a standalone `article`, so the unresolved reference compiles to
"??". The fix matches the file's own convention, since the parallel "Bayesian Uncertainty Quantification"
subsection at line 900 is immediately followed by `\label{sec:bayesian_validation_supp}` at line 901. Recommendation:
add `\label{sec:keynorm_bias_supp}` immediately after the subsection heading at line 764.

#### Symbols `d_q`, `d_p` used in the boxed gauge-action equation but never defined

Location: `GL(K)_attention.tex:437-439` (`eq:representations`) and `:445-457` (`eq:gauge_action_gaussians`);
notation table `:381-419`.

Equation `eq:representations` introduces `rho_q: GL(K) -> GL(d_q)` and `rho_p: GL(K) -> GL(d_p)`, and the boxed
gauge-action equation applies `rho_q(Omega)` to `mu_q`, but the notation table defines `K`, `K_q`, `K_p`, `N`,
`d_k`, and `d_v` and never `d_q` or `d_p`. Since `mu_{q,i}` lives in `R^{K_q}` and the text equates `K = K_q`, the
boxed action only typechecks if `rho_q(Omega)` is `K_q x K_q`, which forces `d_q = K_q` (the defining
representation), an identification the manuscript never states. The intended reading is recoverable from the
surrounding `Sigma_q` congruence and `mu_q` multiplication and from the generic `rho: G -> GL(V)` table entry, so
the gap is expository rather than load-bearing. Recommendation: either replace `d_q`/`d_p` with `K_q`/`K_p`
throughout (cleanest), or add one sentence after `eq:representations` stating that `rho_q` and `rho_p` are the
defining representations so `d_q = K_q` and `d_p = K_p`, and add both symbols to the notation table.

#### Two conflicting "theory-predicted optimal temperatures" in the main text

Location: `GL(K)_attention.tex:1314` and `:1717` (`tau = sqrt(d_k)`) against `:2039` and `:2270`
(`tau ≈ 2 sqrt(d_k)`); reconciliation only in the supplement at `:710,756,762`.

The main text boxes `tau = sqrt(d_k)` at line 1314 and tabulates the `1/sqrt(d_k)` scaling as the prediction at
line 1717, while lines 2039 and 2270 call `tau = 19.0 ≈ 2 sqrt(d_k)` "the theory-predicted optimal temperature."
The bridging explanation, that the dot-product form yields `sqrt(d_k)` while the squared-distance form used for the
BERT comparison keeps the KL's factor of one-half and yields `2 sqrt(d_k)`, lives only in the supplement at lines
756 and 762, and the underlying algebra is elementary and sound. The main text does leave breadcrumbs (line 1717
tags its value "dot-product form" and line 2037 states the BERT comparison uses the squared-distance form), so this
is a presentation gap rather than a genuine contradiction. Recommendation: add a half-sentence at lines 2039 and
2270 clarifying that `2 sqrt(d_k)` is the prediction in the squared-distance form, equal to the dot-product-form
`sqrt(d_k)` up to the KL's one-half prefactor, with a cross-reference to the supplement so the two main-text values
are visibly reconciled.

### Wiki connections

#### Voita2019 head-specialization is in the bibliography but cited zero times where Section 5 makes exactly its claim

Location: `GL(K)_attention.tex` Section 5, figure caption near `:2187` (`\label{fig:glk_attention}`) and the
per-head paragraph near `:2282-2283`; `references.bib:197`.

`references.bib:197` defines `Voita2019` ("Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy
Lifting, the Rest Can Be Pruned"), but the entry is cited nowhere in the GL(K) pair. Meanwhile the figure caption
at line 2187 states that heads "develop qualitatively distinct strategies such as broad context, token-selective,
query-driven aggregation, and local recency," and line 2283 reports that "per-head optimal temperatures span a wide
range (RoBERTa: median = 25, std = 15.3) ... carries real information that a single global temperature discards"
and vary "by a factor of 2.4x across architectures." These are head-functional-heterogeneity claims that map
directly onto Voita's contributions, so the unused bib entry sits precisely where it would supply external
empirical grounding. The linkage is interpretive rather than like-for-like, since Voita studies standard
transformers with learned QK/V projections whereas the gauge model has no learned attention projections, so the
wording should hedge accordingly. Recommendation: add `\citep{Voita2019}` at the figure caption near line 2187 and
at line 2283, with a sentence such as "This per-head heterogeneity is the gauge-theoretic counterpart of the
empirically observed functional specialization of attention heads, where a minority of heads do the heavy lifting
and the remainder are prunable `\citep{Voita2019}`; here that specialization is read off the trace component
`kappa_a` of each head's gauge block."

### RG g2

#### `y2 = -1` is correct under the manuscript's own averaging definition, but the main text omits the `n^2`-edge justification

Location: `GL(K)_attention.tex:2304` (`eq:g2_def`) and `:2318-2323` (`eq:rg_scaling`); the resolving argument lives
only at `GL(K)_supplementary.tex:958` with the validation table at `:1004`.

The exponent itself is correct, not an error, and this resolves the issue both prior passes flagged but did not
settle. The supplementary at line 958 supplies the bookkeeping: the meta-agent transport
`Omega_AB = (|A||B|)^{-1} sum_{i in A, j in B} Omega_ij` averages `|A||B| = n^2` inter-cluster edges, so `g2` is the
norm of a mean of `n^2` independent mean-zero edge fluctuations and contracts as `(n^2)^{-1/2} = n^{-1}`, while
`g1` averages only `n` token covariances and contracts as `n^{-1/2}`. Two independent numerical checks (2000 trials
per level, `K=8`, `delta ~ N(0, 0.1)`) give coarse-to-single fluctuation-norm ratios matching `1/n` to three
significant figures and clearly not `1/sqrt(n)`, in agreement with the manuscript's own CLT validation table (`y2`
measured `-1.000`, predicted `-1.000`). The defect is purely expository: the main text defines
`g2 = ||Omega_ij - Omega|| / ||Omega||` at line 2304, structurally identical to the single-edge `g1` ratio, then
asserts `g2' = n^{-1} g2` at lines 2318-2323 with no edge count, so a main-text-only reader sees an unexplained
asymmetry with `g1`. Recommendation: change no exponent and instead port the supplementary's edge-count argument
into the main text by inserting, immediately before the displayed scaling equation, a sentence such as: "The
intrinsic anisotropy `g_1` averages `n` token-level covariance perturbations and so contracts as `n^{-1/2}`,
whereas the meta-agent transport `Omega_{AB} = (|A||B|)^{-1} sum_{i in A, j in B} Omega_{ij}` averages the
`|A||B| = n^2` inter-cluster edges, so its fluctuation norm `g_2` contracts as `(n^2)^{-1/2} = n^{-1}`; the
holonomy is a linear-order sum of three such transport deviations and inherits the same `n^{-1}` rate." This
removes the false symmetry with `g1` that makes `y2 = -1` look inconsistent with the linear-norm definition.

## Minor / unverified

The following are low-severity items noted but not adversarially checked in this pass.

- `sigma_max = 10` in the supplementary (`:653`) is true only because the active config (`config.py:349`) overrides
  the retraction defaults of `5.0`; either align the function defaults to `10.0` or note that `sigma_max` is a
  `VFEConfig` knob so the value is reproducible without reading signatures.
- The supplementary claims the `killing_per_block` preconditioner was "used in the reported runs" (`:670`), but
  `config.py:353` defaults `phi_precond_mode` to `'none'`; either cite the specific run config that sets it or
  soften to "available as an option."
- The manuscript cites entry point `transformer/vfe/train_vfe.py` (`:2221,2228`), but the released `vfe3` entry
  point is `train_vfe3.py` at the repo root; state whether the numbers come from the VFE_2.0 checkout (and give
  that path) or are reproducible from this release (and cite `train_vfe3.py`).
- POSITIVE: transport assembly, free-energy assembly, and the gauge-breaking-toggle disclosure faithfully match the
  code (`transport.py:679`, `free_energy.py:373-403`), so the documentation fixes above are reproducibility-relevant
  cosmetics, not substantive theory errors.
- Coarse-grained `Sigma_A` (`supplementary:946`) is the Euclidean law-of-total-covariance moment, not the
  affine-invariant SPD mean used in the retraction; add a sentence stating it is the moment-matching second-moment
  coarse-grained covariance, deliberately distinct from the affine-invariant mean, or motivate why moment matching
  is the right notion for RG coarse-graining of beliefs.
- The symbol `lambda` is overloaded across three meanings: alignment coupling (`:410`), Lagrange multiplier
  (`supplementary:751`), and key-norm bias coefficient (`supplementary:766,782`); rename the key-norm coefficient
  (e.g. `lambda_K`) or add a local disambiguating note.
- In-text K-sweep perplexities do not match the cited CSV artifact; resolve per the existing in-source TODO at
  `:2230` (state the checkpoint-selection policy and reconcile against the finalized CSV).
- "Substantive content rather than a coincidence" (`:1683`) over-claims a residual/skip-connection correspondence
  that Table 1 marks merely structural (status `S`, `:1734`); soften to a conjecture of a common computational
  principle pending the depth experiment, and cross-reference the Limitations at line 2357.
- "Learning is symmetry breaking" (`conclusion:2419`) is presented as a result but is a relabeling absent a
  vacuum-degeneracy prediction; either attach a concrete order-parameter prediction or downgrade the verb and keep
  the `S` hedge visible in the conclusion.
- The natural-gradient/mirror-descent reading (`supplementary:628-634`) is under-cited; add `\citep{amari1998natural}`
  alongside `Cencov1982` at line 629 and optionally a clause noting the forward-KL Gibbs update is a mirror-descent
  step in the entropic geometry. Partially handled already; not urgent.

## Rejected on verification

- Holonomy collapses to an abelian phase: REJECTED. The finding misread `delta_ij·G` as a scalar times one fixed
  generator; it is the standard `sum_a delta_ij^a G_a` basis contraction (per the notation table and
  `transport.py:227-228`), so the connection is matrix-valued and non-abelian, with a numerical commutator norm of
  `0.53`, not zero.
- Meta-agent coarse-grained transport leaves the structure group, invalidating the RG exponents: REJECTED. The
  arithmetic mean does leave the group, but this is a low-severity terminology nit; the stated exponents are
  unaffected (the intrinsic mean gives `y2 = -1.014`, if anything strengthening them), and the
  intrinsic-versus-extrinsic point duplicates a prior addendum.
- Holonomy-compositionality bridge is unfalsifiable and never operationalized: REJECTED. The finding missed
  `sec:flat_bundle` (`:2285-2292`), which already demotes the claim to an open question and sketches the exact
  Regime-II measurement on COGS/SCAN it asks for.
- Abstract/conclusion "attention IS variational inference" contradicts the D/S/I taxonomy: REJECTED. The copula
  attaches to the exactly-derived (D-tier) attention rule, not the S/I-tier architecture components, and the
  abstract switches to hedged verbs precisely where it reaches them.
- "First-principles" framing omits the approximations actually used: REJECTED. None of the four `first-principles`
  instances attaches to the trained runs; all modify the theory object, and every listed approximation is already
  disclosed in dedicated qualification paragraphs.
- von Oswald in-context-learning-as-gradient-descent absent from the layer-equals-gradient-step derivation:
  REJECTED as not-new. The gap is real but already logged as items 41 and 45 of prior review pass 1, and the
  finding even cites the wrong bibkey.
- `eq:g2_def` should be redefined at the coarse-grained level (inconsistency): REJECTED as stated. The single-edge
  definition is correct and parallel to `g1` and `g3`; the only real residual is the expository main-text gap
  already captured as the confirmed RG g2 finding above.

## Recommended actions

1. Correct the four code-fidelity defects in the supplementary and main text, since these misrepresent the released
   implementation to anyone auditing reproducibility: replace `sanitize_sigma` with the real inline clamp plus
   `floor_eigenvalues` and drop the false NaN-raising claim (`supplementary:649-654`); rewrite both the `:561` and
   `:672` paragraphs to describe `retract_phi` as a group-aware Frobenius trust-region step (GL `5.0`, SO `pi`),
   dropping the modulo-2π/antipodal/principal-ball language; and change both `connection.py` "MLP mode" sites
   (`:2288,:2365`) to the bilinear `connection_W` description.
2. Fix the `kappa ∝ sigma^2` glossary entry (`:409`) with the one-line edit so the notation, the implementation
   paragraph (`:1800`), and the code agree; this removes a real internal contradiction at near-zero cost.
3. Port the `n^2`-edge justification into the main text immediately before `eq:rg_scaling` (`:2318`), changing no
   exponent, so the `y2 = -1` result is self-contained and the `g1`/`g2` asymmetry is motivated.
4. Repair the three broken cross-references: `\ref{sec:rope_gauge}` to `\ref{sec:rope}` at line 2046, add
   `\label{sec:keynorm_bias_supp}` after `supplementary:764`, and resolve `fig:glk_pca_frames` (`:2251,:2255`) by
   inserting the PCA figure with that label or rewriting the two sentences to drop the figure clauses.
5. Add the missing `Voita2019` citations at the head-specialization claims (`:2187` and `:2283`) with the suggested
   interpretive-hedge sentence, strengthening the empirical grounding of Section 5.
6. Resolve the `d_q`/`d_p` definition gap (`:437-439`) by replacing them with `K_q`/`K_p` or adding the
   defining-representation sentence plus notation-table entries.
7. Add the half-sentence reconciling `sqrt(d_k)` and `2 sqrt(d_k)` at lines 2039 and 2270 with a cross-reference to
   the supplement, so the two "theory-predicted" temperatures are visibly the same prediction in two conventions.
8. Address the minor items opportunistically in the same copyedit pass: clarify the `sigma_max = 10` and
   `killing_per_block` config provenance, correct the `train_vfe.py` entry-point path, disambiguate the overloaded
   `lambda`, soften the residual-correspondence and symmetry-breaking claims, and add the `amari1998natural` and
   `Sigma_A` moment-matching notes.

## Lens-by-lens summaries

Manuscript-versus-code fidelity: the load-bearing math (transport `Omega_ij = exp(phi_i)exp(-phi_j)`, the
`tau = kappa sqrt(d_head)` temperature, the canonical-versus-surrogate free-energy assembly, the SPD spectral
floor/ceiling) is faithfully implemented and the gauge-breaking toggles are disclosed, but the supplementary names
three functions/algorithms absent from the code (`sanitize_sigma`, `retract_to_principal_ball`, the `connection.py`
MLP mode), plus the `kappa ∝ sigma^2` glossary contradiction and the config-override caveat on `sigma_max`.

Differential geometry: the holonomy 3-cycle factored form, the geometric-mean Boltzmann exponents, the `Sigma_A`
coarse-graining moment formula, and the dexp cross-reference all verified correct; the two would-be problems
(abelian holonomy collapse, group-leaving coarse transport) were both refuted on adversarial verification.

Internal consistency and bib integrity: the bibliography is fully sound (all 100 distinct cited keys resolve; the
App H keys the prior pass worried about are not the keys actually used); the real defects are three broken
cross-references, the undefined `d_q`/`d_p` pair, and the two-temperature presentation gap.

Philosophy of science: the manuscript is well-calibrated on the specific over-reaches the brief worried about (the
`r=0.804` demotion, the toggle-disclosure paragraphs, the D/D#/S/I taxonomy); all five candidate framing
over-claims were either refuted or reduced to one-clause hedges in the minor list.

Second wiki sweep: the strongest new connection is `Voita2019` (in the bib, cited zero times, exactly Section 5's
claim); the von Oswald and predictive-coding precision-weighting gaps duplicate prior-pass items, and Wang-2023
SPD attention is a neighboring-not-competing architecture worth one positioning sentence (logged minor).

Information geometry: the open `g2` exponent resolves cleanly to `y2 = -1` under the manuscript's own `n^2`-edge
averaging, confirmed numerically and against the supplementary validation table; the only fix needed is porting the
edge-count justification from the supplement into the main text.
