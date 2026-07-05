# Deep Peer Review Pass 19: GL(K) Manuscript Pair

Review date: 2026-07-04.

Reviewed files (the authoritative **repo** copy, per the author's direction that the
`V3_Transformer` copy is the current WIP, not the vault):
`Manuscripts-Theory/GL(K)_attention.tex` (2488 lines) and
`Manuscripts-Theory/GL(K)_supplementary.tex` (1516 lines), plus
`Manuscripts-Theory/references.bib` and the scaling data under
`vfe3_scaling_results/grow_K_GL10/`.

Method: nine-lens expert panel (numerical/scaling-fits, provenance/reproducibility,
gauge-equivariance, transformer-scaling, claim-status/philosophy, variational/info-geometry,
citation, cross-reference/notation consistency), each seeded with the do-not-reflag list from
the vault verified-ledger (passes 3-12) and the repo pass 13-18 review reports, then
adversarial skeptic verification on every high finding. Every load-bearing scaling number was
recomputed from `scaling_points.csv` (36 runs = 12 widths x 3 seeds) and `scaling_summary.json`.

## Recommendation

**Major revision** before submission, the same verdict pass 18 reached on 2026-06-27. No new
equation-level defect was found in the ledger-settled analytic core, and it was not
re-litigated. The mathematics remains strong. The blockers are all empirical reporting,
claim-scope, and provenance, and several were introduced or exposed by the 2026-07-04 scaling
ingest (which no prior pass had seen). One item is a hard submission blocker (an all-`TBD`
hyperparameter table whose caption claims it is populated). The most substantive new cluster
concerns the learned head mixer that is active in the headline scaling sweep.

## What reproduced exactly (verified, now settled)

The scaling results are numerically sound. Recomputed from the CSV against the repo's own
fitting code:

- Table `tab:vfe3_scaling`: all twelve rows of mean test PPL +/- std reproduce to the printed
  precision (219.0 +/- 1.3 at K=10 through 74.1 +/- 0.3 at K=120, seeds 6/23/64), and the
  `Params` column (7.6M to 90.7M) matches `n_params`. The abstract (attn:48) and results
  (attn:2281) sequences match the table row for row and are monotone.
- Offset cross-entropy fit `L(N)=E+A N^{-alpha}`: alpha=0.5584, E=3.9506 nats (PPL floor
  exp(E)=51.97), R^2=0.99956, matching `scaling_summary.json` to all digits.
- Offset-free CE fit: alpha=0.08735 (manuscript 0.0873), R^2=0.96794 (0.9679); the 95% CI
  reproduces as [0.0633, 0.0956] (manuscript [0.0627, 0.0957]; the sub-0.001 differences are
  bootstrap RNG-stream ordering). The point estimate sits at the 49th percentile of a genuinely
  left-skewed bootstrap, so the value-distance asymmetry is real skew, correctly reported, not
  an error.
- Offset PPL fit `PPL(K)=aK^b+c`: b=-1.0489 (-1.05), c=63.96 (64.0), R^2=0.99901 (0.999).
- tokens/parameter at K=120 = 491.52M/90.67M = 5.42 (manuscript "about 5.4").
- Citations spot-checked against primary sources hold: HoffmannChinchilla2022 (the ~20:1
  compute-optimal ratio), grave2017improving (LSTM 48.7 ~ 49), dai2019transformerxl (18.3 ~ 18),
  chen1998empirical (modified KN with per-order discounts), Neal1998 and Dempster1977 both
  resolve in the repo bib.

## Claim-Status Table

| Claim | Status | Review judgment |
| --- | --- | --- |
| KL attention softmax from entropy-regularized source assignment | D/P | Settled (ledger); no issue. |
| Gaussian KL is GL(K)-invariant under transported pushforwards | P | Settled (ledger); no issue. |
| Standard QK attention recovered in the isotropic/flat/key-norm limit | D-sharp/S | Caveats preserved; sound. |
| GL+(10) sweep improves monotonically 219.0 to 74.1 at K=120 (3 seeds) | E | Numbers reproduce exactly; provenance and head-mixer caveats need surfacing (M2, M5). |
| The sweep "measures the gauge VFE parameter-scaling law itself" | E->S | Over-claimed: the curve is one head-mixer-enabled, 10-commit, single-layer, fixed-budget configuration (M2, M3). |
| E=3.95 nats is an "irreducible" loss floor | E | Not irreducible: cited baselines reach CE 2.89 nats; it is a fixed-budget/model-family artifact (M3). |
| Trained with "natural-gradient dynamics" (abstract/methods, blanket) | E/D | Belief channel only; parameters and swept phi are Adam-learned (M4). |
| Per-head temperature dispersion "carries real information" | E->S | n=5 association; verb inflated (pass-18 MC#5, still live) (M4). |
| Gauge VFE "outperforms the strongest classical statistical method" | E->S | Only a single KN-5 baseline was run; narrow the descriptor (minor). |
| Only learned linear map retained is the output projection | D | False for the sweep, which also carries the head mixer (M2). |

## Major Comments

### M1. The hyperparameter table is entirely `TBD` while its caption and Code Availability claim reproducibility (essential; submission blocker)

Every one of the fifteen value rows of Table `tab:glk_hyperparams` (attn:2094-2112) reads
`TBD`: optimizer, learning-rate schedule, batch size, token budget, gradient clipping, weight
decay, all four learning rates, kappa, inner iterations, seeds, checkpoint policy, and hardware.
The caption (attn:2115) states the table is "populated from the archived per-run configuration
dictionaries," and Code Availability (attn:2476) asserts that "all experiments reported in the
results section can be reproduced." These are incompatible with an empty table, and the
commented-out author note at attn:2085 confirms it is a known unfilled placeholder. At least six
of the `TBD` values are already stated elsewhere in the same manuscript: optimizer = Adam
(Algorithm 1, attn:2171); inner iterations T = 1 (attn:2129, CSV `n_e_steps=1`); batch size 16
headline / 64 sweep (attn:2289; 491.52M/60000/128 = 64); token budget 491.5M / 60000 steps
(supp:1461, attn:2211); seeds 6, 23, 64 (attn:2081, supp:1487); hardware RTX 5090 / 64GB /
Ryzen 9900x (attn:2050). This is pass-18 Major Comment #2, unresolved a week later, and it
verified as CONFIRMED / high under adversarial review. Fix: populate all rows from the archived
config dicts (per-run columns where they differ), or remove the table; until then the Code
Availability reproducibility sentence is unsupported by the paper itself. Also add the missing
`\ref{tab:glk_hyperparams}` (the table currently floats uncited).

### M2. The head mixer active in the headline scaling sweep is disclosed inconsistently, and its equivariance status is stated only far from the result (essential)

The reproducible sweep enables a learned per-irrep-block head mixer at all eleven multi-head
widths (attn:2063, 2065, 2078, 2081; supp:1461). Five distinct problems surround it, and they
should be fixed together:

1. **Abstract and intro overstate minimality.** "the only learned linear map retained is a
   single output projection" (attn:48, echoed attn:67) is false for the very sweep the same
   sentence cites: the mixer is a second learned linear map. Five body sites disclose it; only
   the abstract and intro make an unqualified exclusivity claim. Adversarial review DOWNGRADED
   this from high to medium because under the linear decode used in every sweep run the mixer is
   absorbable into the output projection, `logits = W(M mu) = (WM) mu` (attn:2428), so the
   trained function class is a single effective linear readout and capacity is unchanged. The
   fix is a one-clause abstract scoping, not a retraction. Do state the absorbability where the
   mixer is introduced.

2. **The dedicated "no neural networks" scope paragraph omits the mixer.** attn:2415 lists two
   qualifications, the output projection and the `connection_W` variant that is explicitly *not
   enabled*, and closes "this is the one neural component." It omits the head mixer that *is*
   enabled in the headline sweep. Add it as a third qualification.

3. **The equivariance break is stated only in Future Directions.** RoPE (attn:2063, 2417) and
   Regime II (attn:2343) both have their symmetry-forfeiture fenced at the point of use. The
   head mixer, the only one of the three actually on in a headline result, has its forfeiture
   stated only at attn:2428 ("does not commute with the gauge action and the map is not
   gauge-equivariant in this regime"), disconnected from the scaling section. Add a matching
   clause at attn:2065 / the Table caption / supp:1461.

4. **"Schur-commutant" is a misnomer for the untied gauge actually used.** On the untied gauge
   `GL(d_head)^H` the equivariant commutant is the per-head scalars `diag(c_1 I, ..., c_H I)`,
   not the full `A_t (x) I` family, and the paper itself says so at attn:2428. Calling the
   enabled mixer a "Schur-commutant coupling" at attn:2063/2065/2081 and supp:1461 (all
   describing the untied `block_glk` sweep) contradicts attn:2428: `A_t (x) I` is the Schur
   intertwiner of the *tied* gauge. Recomputation confirms `[(A (x) I) G] = [G (A (x) I)]` for
   independent per-head `g_a` forces `A` diagonal. Drop or qualify "Schur-commutant" where it
   describes the untied sweep.

5. **Future Directions treats the mixer as unevaluated while the headline depends on it.**
   attn:2428 says "whether the map improves language modeling ... is an open empirical question"
   requiring "a learning-rate-matched mixer-off baseline," yet the 219.0->74.1 curve quoted in
   the abstract, intro, results, and conclusion is produced with it on, with no such baseline,
   and the K=10 endpoint is mixer-off while K>=20 are mixer-on. Move that acknowledgment to the
   result, and note the mixer-off/on split across the curve.

### M3. The scaling curve is presented as a scaling law, but the fixed-data design and the "irreducible" floor need re-scoping (essential)

The numbers are exact; the interpretation overreaches in four ways.

1. **Fixed-data regime mixes over- and under-trained points.** Every run uses 491.5M tokens, so
   tokens/parameter falls from 64.7 at K=10 to 5.4 at K=120. The small models are heavily
   over-trained and the large ones under-trained relative to Chinchilla's ~20:1. The paper
   fences the large end ("the upper end is data-limited") but not the over-trained small end,
   and the resulting exponent is not comparable to a Kaplan/Chinchilla parameter exponent
   (which are measured near compute-optimal). Say so.

2. **E=3.95 nats is not an irreducible floor.** The paper's own cited baselines reach WikiText-103
   test PPL ~18 = CE 2.89 nats (attn:2291, dai2019transformerxl), below the claimed
   "irreducible" 3.95 nats / PPL 52. In a fixed-D sweep the fitted offset absorbs the growing
   data-starvation at large K, so it is a budget/model-family constant, not an irreducible
   entropy. Rename "irreducible term" and state the contamination.

3. **The two floors disagree by ~23%.** The CE offset fit gives exp(E) = 52; the PPL offset fit
   gives c = 64.0. Both estimate the same infinite-width irreducible PPL, and the paragraph
   presents them as one deceleration story with only the single word "apparent" on c. Note that
   a floor extrapolated from twelve widths whose smallest PPL is 74 is weakly identified, so 52
   and 64 are consistent within floor-estimation uncertainty, and neither should be quoted to
   two or three significant figures.

4. **Reporting asymmetry.** The headline exponent alpha=0.0873 is taken from the decisively
   worse-fitting offset-free convention (R^2=0.97 vs 0.9996), and its CI is reported while the
   better-fitting offset exponent alpha=0.56 is given with no CI (it is [0.39, 0.60] in
   `scaling_summary.json`, a ~+/-19% band). Report the offset CI too. Separately, the
   goodness-of-fit contrast "0.999 vs 0.958" mixes metrics: 0.999 is a linear-space R^2 and
   0.958 reproduces only as a log-space R^2 (linear-space is 0.960); put both in the same space.

### M4. Two pass-18 major comments remain unresolved in the current repo copy (essential)

1. **Killing preconditioner scope (pass-18 MC#3).** supp:560 and supp:578 still state "the
   reported runs use the block-diagonal Killing-form variant." For the released sweep, phi is
   held fixed in the E-step (`eta_phi^E = 0`, attn:2162) and trained only at the M-step by Adam
   (attn:2171), so an E-step phi preconditioner cannot be the one used. Scope the claim to the
   headline / English-comparison runs (E-step-learned phi); the sweep uses no E-step Killing
   preconditioner.

2. **Per-head temperature verb (pass-18 MC#5).** attn:2334 still reads "demonstrates that this
   covariance structure carries real information that a single global temperature discards" for
   an n=5 cross-architecture association. The neighboring sentences were softened in earlier
   passes; this one was missed. Recast as "indicates that per-head temperature is an exercised
   degree of freedom that a single global temperature would collapse."

Also scope the blanket "natural-gradient dynamics" (attn:48, attn:2063): the supplement already
scopes it correctly at supp:1582 (belief channel only; parameters and swept phi are Adam),
mirror that one clause in the abstract and methods.

### M5. Provenance disclosure covers the aggregate but not the per-width error bars (recommended)

The "36 runs span 10 source commits, development-provenance evidence" caveat appears in three
places (supp:1487, attn:2281, attn:2411). It discloses that the pooled bootstrap CI excludes
cross-commit variation, but not that four of the twelve per-width mean +/- std entries are
themselves cross-commit: parsing the `git_sha` column, K=70 ran its three seeds on three
distinct commits, and K=80/100/110 on two each (K=10-60, 90, 120 are single-commit). So bars
like "84.5 +/- 0.7" at K=70 conflate seed variance with code-version variance and are not pure
seed dispersion. Add one clause to the caption naming the four cross-commit widths.

### M6. The repo and vault copies disagree on whether the central ablation exists (recommended, and a housekeeping hazard)

The repo copy states the gauge on/off ablation "is not yet run" (attn:2287). A divergent vault
copy reports it done (learned frame PPL 154, frozen random 279, identity 267) and pass 18
reviewed that version. This is precisely the controlled test that would isolate the paper's
central causal claim, so the two authoritative-looking copies must not disagree on whether it
exists. Reconcile before submission: if the learned/frozen/identity result is trustworthy,
import it into the repo copy with its config and drop "not yet run" (and apply pass-18 MC#4:
the "locates the advantage in the gauge-transport geometry" conclusion is one step broader than
a frozen-random-vs-learned and confounded-identity comparison proves); otherwise remove it from
the vault. More generally, the two copies have drifted (this ablation paragraph, and the
`Neal1998` bib key present only in the repo), which raises the risk of applying fixes to the
wrong file. Settle on one source of truth.

## Minor Comments

1. **Appendix J is absent from both Supplementary-Material summaries.** supp:44 and attn:2483
   both stop at Appendix I, but the supplement now has a tenth appendix (`app:vfe3_ablations`,
   supp:1458) housing the scaling sweep, divergence-order, and positional-extrapolation
   results. Add it to both lists. (Pass-18 minor, still open after the fresh edit added the
   section.)

2. **The manuscript-embedded CE figure is stale.** `Manuscripts-Theory/attention/figs/vfe3_gl10_scaling_ce_vs_params.png`
   (md5 d11fe8f0..., Jul 3) differs from the regenerated
   `vfe3_scaling_results/grow_K_GL10/figures/scaling_ce_vs_params.png` (md5 4dfe5f9b..., Jul 4),
   so the paper renders an outdated left panel. The right panel
   (`vfe3_gl10_ppl_vs_embed_dim_offset.png`) is byte-identical to its source and is current.
   Recopy the CE figure (renaming to add the `vfe3_gl10_` prefix the source basenames lack).
   The build will not break either way, since both referenced files exist.

3. **Decode-scope temperature slip.** supp:1454 says the decode temperature mirrors "the
   learnable attention temperature tau_attn = kappa sqrt(K)." Everywhere else the attention
   temperature is per-head `kappa sqrt(d_head)` (attn:2102, 2151, 1810). For K=90 / 9 heads the
   two differ by a factor of 3. Fix to `sqrt(d_head)`, or state explicitly that the decode uses
   the full belief dimension `sqrt(K)`.

4. **KN-5 literature figure.** attn:2291 attaches the "~153-156" word-level KN-5 WikiText-103
   figure to `\citep{merity2017pointer}`, which reports no WikiText-103 model results (only PTB
   and WikiText-2). Pass 9 already scoped this cite to the corpus/vocabulary; the current
   phrasing still reads as sourcing the number. Either move the cite so it attaches only to "the
   WikiText-103 corpus," or add the actual source for the ~153-156 figure (e.g. Merity et al.
   2018, "An Analysis of Neural Language Modeling at Multiple Scales").

5. **Baseline-scope overreach.** attn:2291 and attn:2293 say the model "outperforms the
   strongest classical statistical method / the best classical statistical methods." Only a
   modified KN-5 baseline was trained. Narrow to "the strongest classical n-gram language model
   (modified Kneser-Ney 5-gram)"; the 71.6-vs-134.8 result itself is real.

6. **Two K=90 GL+(10) numbers.** The results table (attn:2257) reports the archived
   English-comparison GL+(10) K=90 as 58.8M params / PPL 76.4 (two seeds); the sweep table
   (supp:1481) reports 68.0M / 79.3 (three seeds). Both are labeled GL+(10), K=90, 9 heads. The
   9.2M-parameter and 2.9-PPL gaps (archived run vs mixer-enabled sweep, different generator
   count) are distinguished only across two footnotes in two files. Add a half-sentence where
   76.4 appears noting the sweep counterpart is 79.3 and why they differ.

7. **M-step objective presentation.** Algorithm 1 (attn:2170) prints the M-step loss with three
   terms (`L_CE + alpha_hat KL(q||p) + (alpha_phi/2)||phi||^2`), while the prose (attn:2121,
   supp:1454) calls the M-step "the cross-entropy learning loss." The released config zeroes the
   self-coupling term (`mstep_self_coupling_weight = 0.0`), reconciling the two, but the
   algorithm does not flag `alpha_hat = 0`. Annotate it.

8. **Decode-exactness condition overshoots.** supp:1454 scope 2 says the decode identity is
   "exact only when q and pi_v lie on the same statistical submanifold." The closed-form
   Gaussian KL is exact across submanifolds; the genuine caveat, which the same sentence then
   gives correctly, is the diagonal projection of a full-covariance belief at decode. Reword to
   match.

9. **Single-seed over-reading.** The divergence-order sweep (supp:1507) and positional-prior
   extrapolation (supp:1510) read directional effects out of sub-1% single-seed deltas that sit
   below the appendix's own ~2% noninterpretability floor (which was itself measured on the
   three-seed scaling sweep, a different regime). Report the near-unity divergence orders and
   the offset/relative priors as "within noise"; the robust effects (alpha >= 1.5 degrades
   sharply; the learned absolute prior degrades 11.4% out of distribution) are well above the
   floor and stand.

10. **Divergence optimum vs headline choice.** supp:1507 argues KL (alpha=1) is the empirical
    optimum on WikiText, while the Japanese headline uses Renyi-1/2 (attn:48). Not a
    contradiction (different corpora, honestly disclosed), but add a clause noting the optimum
    is corpus-specific so a referee does not read it as inconsistent.

11. **Stale bib venue.** `geshkovski2023mathematical` is listed as an arXiv preprint; it is now
    published in Bull. Amer. Math. Soc. 62(3):427-479 (2025). The cited claim is supported by
    the source; update for completeness only.

12. **American English.** "analogue" appears six times (attn:931, 1610, 1655, 2012, 2329;
    supp:130); the project style requires "analog."

13. **Comma splice.** attn:2291 "... rather than count-based smoothing suggesting that the
    geometric structure ..." runs on; split into two sentences or add "which suggests."

## Questions For The Author

1. Which copy is authoritative, the repo or the vault? They have diverged on the gauge on/off
   ablation (repo: "not yet run"; vault: learned 154 / frozen 279 / identity 267) and on the
   bibliography. All fixes should target one source.

2. For each headline empirical row, which optimizer updates phi: Adam at the M-step (as the
   released sweep does), or an E-step natural-gradient / Killing preconditioner? The supplement
   (560/578) and the algorithm (2162/2171) currently disagree.

3. Should the head mixer be in the headline scaling sweep at all? If it is capacity-neutral
   under the linear decode (absorbable into W_out), a mixer-off sweep would give a cleaner
   "gauge VFE scaling law" and remove the equivariance-forfeiture caveat from the headline
   result. If it stays, a learning-rate-matched mixer-off baseline is the control the paper
   itself asks for.

4. Is the fixed-491.5M-token curve intended as a scaling law (an exponent to be compared with
   Kaplan/Chinchilla) or as a monotone-improvement demonstration? The framing "measures the
   gauge VFE parameter-scaling law itself" implies the former; the fixed-budget design supports
   only the latter.

## Bottom Line

The analytic center is unchanged and remains heavily verified across eighteen prior passes; the
scaling numbers reproduce exactly from the data. The submission risk is entirely in the
empirical reporting introduced or exposed by the 2026-07-04 ingest: an all-`TBD` hyperparameter
table with a caption that claims it is populated (a hard blocker), a head mixer that is on in the
headline curve but disclosed inconsistently and mislabeled, a scaling-law framing that a
fixed-data sweep does not support, two unresolved pass-18 comments, provenance error bars that
mix code versions, and a repo/vault divergence on the central ablation. None of these touch the
gauge-theoretic derivations; all are addressable by populating one table, reconciling the
head-mixer disclosure across five sites, re-scoping the scaling prose, and syncing the two
copies and the stale CE figure.
