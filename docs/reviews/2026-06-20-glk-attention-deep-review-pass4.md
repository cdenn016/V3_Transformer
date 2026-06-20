# GL(K) attention manuscript — deep review pass 4 (dimensional/number consistency, companion cross-check, framing)

Date: 2026-06-20. Branch: `vfe3-per-layer-figures`. Targets: `Manuscripts-Theory/GL(K)_attention.tex`,
`GL(K)_supplementary.tex`, `references.bib`, the companion `Participatory_it_from_bit.tex`, cross-checked against the
released `vfe3/` tree. This pass ran autonomously.

## Method and scope

Fourth review pass, fresh lenses chosen to avoid the territory of passes 1 through 3 (wiki citations; the seven
load-bearing proofs; code-fidelity, secondary derivations, and the RG `g2` exponent). The five lenses were:
dimensional and shape consistency of every displayed equation; figure, table, caption, and number accuracy with all
arithmetic recomputed; cross-consistency between the companion paper and the GL(K) pair; a second round of
abstract/introduction/conclusion calibration; and an adversarial re-check of the five pass-3 minor items that had
never been verified. Each substantive finding was handed to an independent adversarial verifier instructed to refute
it from primary sources before it could be reported.

Tallies (from the workflow result): 20 raw findings, 7 substantive, of which 6 were confirmed, 1 was rejected, and
13 were logged as minor without adversarial challenge. The verification layer again earned its place: it rejected a
"high" finding (the `train_vfe.py` entry-point path) after establishing that the released artifact is a separate
repository, so the apparent defect was a category error.

All six confirmed fixes were applied autonomously this pass, per the standing instruction. The two framing items
were applied with lighter, prose-clean rewordings than the verifiers' draft edits, which had injected status-table
jargon into running prose against the project style rule; the lighter edits achieve the same calibration.

## Confirmed findings (all applied)

### Dimensional and shape consistency

#### GLU summary box dropped the gauge-transport sandwich metric

Location: `GL(K)_attention.tex` eq:vfe_glu (line 1948), against eq:glu_message (1933) and eq:dkl_quadratic (1941).

The boxed Gated-Linear-Unit summary wrote the Boltzmann-gate exponent with metric `Sigma_j^{-1}`, the plain query
precision, while the two equations it explicitly summarizes both carry the transported precision
`(Omega_ij Sigma_j Omega_ij^T)^{-1}`. The transported precision equals `Sigma_j^{-1}` only when `Omega` lies in
`O(K)`, so the box displayed a genuinely different bilinear form, and the isotropic limit `Sigma_j = sigma^2 I` is
introduced only later at line 1969, so no orthogonality restriction is in force at the box. A NumPy check on random
`GL(3)` `Omega` and SPD `Sigma` gave `e^T (Omega Sigma Omega^T)^{-1} e = 0.610` against `e^T Sigma^{-1} e = 1.911`,
confirming the forms differ. Applied edit: both gate-energy subscripts `Sigma_j^{-1}` and `Sigma_k^{-1}` in the boxed
equation were replaced by `(Omega_ij Sigma_j Omega_ij^T)^{-1}` and `(Omega_ik Sigma_k Omega_ik^T)^{-1}`, so the
summary box matches the two equations it summarizes. Severity low: the underlying derivation is correct and the right
transported metric is already displayed two lines above, so this was a presentational defect in a shorthand box.

### Figures, tables, and numbers

#### Ablation advantage mis-attributed to the wrong model

Location: `GL(K)_attention.tex:2272`, against the ablation rows of `tab:glk_results` (2213-2215) and the in-source
TODO at 2360.

The Discussion stated that "the gauge VFE (81.4M params) thus outperforms ... by 1.66x ... against parameter-equalized
ablation baselines at d_model=90 the gauge VFE still outperforms by 1.87 to 1.91x." The grammatical subject is the
81.4M `GL(15)` model whose test perplexity is 71.6, but the ablation test perplexities (RoPE 138.6, attention-only
142.8, parameter-equalized 145.8) divided by 71.6 give 1.94, 1.99, and 2.04, not 1.87 to 1.91. The claimed range maps
exactly onto the `GL(10)`/76.4 divisor, confirmed by the author's own TODO at line 2360, and line 2359 correctly
attributes 1.91x and 1.87x to the `GL(10)` 58.8M model. Applied edit: line 2272 now reads "the gauge VFE (test PPL
71.6) still outperforms by 1.94x (RoPE), 1.99x (attention-only), and 2.04x (parameter-equalized)," making the divisor
agree with the stated subject. Line 2359 and the TODO at 2360 were left untouched, since the 1.91x/1.87x figures there
correctly describe the `GL(10)` comparison the TODO flags for refresh. Severity high (a numerical attribution error in
the headline efficiency claim).

#### Training-curve caption disagreed with its own body text

Location: `GL(K)_attention.tex:2158` (caption of `fig:glk_training` panel b) versus body line 2145.

The body said perplexity drops "from 50,257 at initialization to ~76 at convergence; a ~658x improvement," while the
caption of the same panel said "perplexity drops from >900 to ~75 over 60,000 steps." The arithmetic
`50257/76.4 = 657.8 ~ 658` confirms the body uses the uniform-prior initialization (the vocabulary size) and the test
final 76.4; the caption mixed a first-logged-step value (>900) with the validation final (75.5). Applied edit: the
caption now reads "perplexity drops from its uniform-prior initialization (equal to the vocabulary size, 50,257) to
test PPL 76.4 over 60,000 steps (the curve enters the logged range below 900 within the first steps)," aligning the
initial and final values with the body and all tables while preserving the first-logged-step observation. Severity low
(expository, not a wrong computed number).

### Companion-paper cross-consistency

#### Two RG/metric deliverables deferred to a companion that supplies neither

Location: `GL(K)_attention.tex:2347` and `:2349`, against `Participatory_it_from_bit.tex:2551`.

The GL(K) paper deferred "a principled treatment of `g2`" and "the metric-level formulation" (Fisher-Bures), and the
trained-model RG validation, to "the companion development" and "a companion paper." The only companion cited anywhere
in GL(K) is `Dennis2025it`, that is `Participatory_it_from_bit.tex`. A grep of the companion for `g_1`/`g_2`/`g_3`,
the coupling definitions, `rg_scaling`, scaling-dimension, and Fisher-Bures returns nothing relevant: the couplings
are defined only in GL(K), and the companion's sole RG passage at line 2551 disclaims a derived flow ("an RG-flavored
cascade but is not a derived RG flow"). The companion thus delivers none of the three deferred items. Applied edit:
at line 2347 "to the companion development" became "to future work," and at line 2349 "We defer this analysis to a
companion paper where it can be developed with trained-model evidence" became "We leave this analysis to future work,
where it can be developed with trained-model evidence." The other `Dennis2025it` citations (lines 400, 667, 2292,
2355) were left untouched, since they reference Regime-II structure the companion genuinely develops. Severity medium
(an attribution defect, no result affected).

### Framing calibration

#### Residual correspondence claimed as substantive content while Table 1 rates it structural

Location: `GL(K)_attention.tex:1683`, against the Table 1 row at 1734 and its caption at 1743.

The paragraph closed by calling the residual/skip correspondence "the substantive content of the correspondence
rather than a coincidence of form," yet Table 1 rates that exact row status S, and the caption defines S as a
correspondence whose specific form the framework "does not uniquely predict," using this very row as its example. The
word "form" thus did opposite work in the two places. Applied edit (lighter than the verifier draft, no status-table
jargon): the sentence now reads that the shared anti-collapse role "reflects a shared computational role rather than a
coincidence of names; the framework explains why an identity path is needed without uniquely predicting its additive
form, and whether the variational restoring term suffices in place of the skip connection at depth is the untested
question deferred to Section (Limitations)." This keeps the role-sharing point, drops the form-coincidence phrasing
that collided with the caption, and folds in the depth caveat. Severity low.

#### Conclusion stated "learning is symmetry breaking" as a revealed result

Location: `GL(K)_attention.tex:2419`, against Table 1 rows 1729-1730 and the Elitzur caveat at line 1011.

The conclusion said the framework "reveals" a geometric interpretation in which "learning is symmetry breaking," while
Table 1 rates the relevant rows S and the body installs an Elitzur caveat that the gauge symmetry is reparameterization
redundancy rather than a physical symmetry, deferring the only falsifiable content (Hessian zero-modes) to future work.
The verified defect is that the verb "reveals" connotes a derived result that clashes with the S tier and the explicit
deferral, and the bare epigram drops the redundancy hedge. Applied edit (lighter than the verifier draft, no
status-table jargon, and removing two pre-existing em-dashes in the process): "reveals" became "suggests," the
explicit-rather-than-spontaneous distinction and the reparameterization-redundancy reading were made visible, a
concrete order-parameter test was flagged as future work, and the epigram "learning is symmetry breaking" was softened
to "learning lifts a gauge degeneracy." Severity low.

## Minor / unverified (not applied this pass)

These were recorded but not adversarially checked, and no edit was made. Several are good candidates for a future
copyedit pass.

- RG coupling `g3` (eq:g3_def) is the only one of the three left un-normalized relative to `g1`/`g2`; either normalize
  it or add one sentence noting it is an absolute rather than relative deviation.
- The prose "`sigma_a^2` is the per-head temperature" (lines 1791, 1800) omits the `kappa_a sqrt(d_head)` factor in
  eq:per_head_temperature; soften to "sets the per-head temperature scale up to the dimensionless normalization."
- The WikiText-103 token count appears as 102M (line 2046), ~119M (footnote 2240), and ~123M (line 2238); annotate
  line 2046 as word-level versus GPT-2-BPE so the figures are commensurable.
- Two supplementary tables, `tab:multi_model_supp` (849) and `tab:rg_flow_supp` (1020), are never bound to text via
  `\ref`; add the anchors in their paragraphs.
- The RoBERTa per-head temperature triple "median=25, std=15.3, CV=0.787" (line 2283) is not self-consistent with
  `tau_opt=29.0` at `tab:temp_dispersion_supp`; source the median and std and reconcile, or report the mean.
- The abstract/text K-sweep perplexities (64.9 at K=120, 194.5 at K=10) disagree with the cited CSV; duplicate of the
  in-source TODO at line 2229, deferred to the finalized CSV reconciliation.
- The three-limit attention reduction drops the learnable `kappa` in GL(K) (lines 1314, 1320, 1369) but retains it in
  the companion (lines 1868, 1870, 1897); make the GL(K) boxes match so the headline reduction is identical.
- Bare `lambda` is overloaded across alignment coupling (notation table 410), Lagrange multiplier (supplementary
  1177-1180), and key-norm bias coefficient (supplementary 772, 775, 785, 793); rename the key-norm coefficient to
  `lambda_K`.
- The natural-gradient subsection (supplementary 629) omits the canonical `amari1998natural` citation that is present
  in `references.bib`; add it.
- The coarse-grained `Sigma_A` (supplementary 947-952) is the Euclidean law-of-total-covariance moment, not the
  affine-invariant SPD mean used elsewhere; add one sentence distinguishing the two.
- The `sigma_max` config-default caveat (supplementary 652-654) is accurate against the code; pass-3 resolved, no edit.
- The `killing_per_block` claim that it is used in the reported runs is true under the released `train_vfe3.py`
  configuration; optional provenance citation only.

## Rejected on verification

- The manuscript cites a released entry point `transformer/vfe/train_vfe.py` that does not exist in this repository.
  REJECTED: the released artifact is the separate `epistemic-geometry` repository named at line 2423, not this
  clean-room V3 rebuild, so the path being absent here is a category error, and the recommended redirect to
  `train_vfe3.py` would be actively false. A narrower provenance-clarity point (anchoring "released" to the
  `epistemic-geometry` repo name at the citation site) may exist but was not what the finding stated.

## Applied this pass

1. `GL(K)_attention.tex:1948` (eq:vfe_glu) — both gate-energy metric subscripts changed from `Sigma_j^{-1}` /
   `Sigma_k^{-1}` to the transported precisions `(Omega_ij Sigma_j Omega_ij^T)^{-1}` / `(Omega_ik Sigma_k Omega_ik^T)^{-1}`.
2. `GL(K)_attention.tex:2272` — ablation advantage corrected to 1.94x / 1.99x / 2.04x against the GL(15) test PPL 71.6,
   matching the sentence subject.
3. `GL(K)_attention.tex:2158` — training-curve caption aligned to the uniform-prior initialization (50,257) and test
   PPL 76.4, with the first-logged-step value kept as a parenthetical.
4. `GL(K)_attention.tex:2347` and `:2349` — the two RG/metric deferrals retargeted from the named companion to "future
   work."
5. `GL(K)_attention.tex:1683` — residual correspondence reworded to a shared computational role, dropping the
   form-coincidence phrasing that collided with the status-S table row and folding in the depth caveat.
6. `GL(K)_attention.tex:2419` — conclusion downgraded "reveals" to "suggests," surfaced the explicit-versus-spontaneous
   and reparameterization-redundancy hedges, and softened the symmetry-breaking epigram.

Verification: braces balance (main 3981/3981); a word-diff scan of the pure insertions (1104 chars) shows zero new
horizontal rules, em-dashes, LaTeX spacing macros, or claudeisms; `sec:limitations` resolves; the GLU and ratio edits
were arithmetic and shape checked.
