# Investigation: Six Criticisms of `GL(K)_attention.tex`

Date: 2026-06-22
Branch: feat/gauge-transport-toggle
Files investigated (repo copy and Research-vault copy are byte-identical, so line numbers are exact):
`Manuscripts-Theory/GL(K)_attention.tex`, `Manuscripts-Theory/GL(K)_supplementary.tex`, `Manuscripts-Theory/references.bib`.

Method: a 24-agent adversarial workflow. For each criticism, one domain-expert investigator (distinct lens per claim) produced a structured finding; a blue team defended the criticism and a red team defended the manuscript, each citing primary sources; a judge issued a binding verdict. Every load-bearing factual claim was checked against the actual `.tex`/`.bib`, and the two most decisive facts (C2's key-norm measurement, C3's naming slip) were re-verified by hand against the source.

## Verdict summary

| # | Criticism | Verdict | Priority |
|---|-----------|---------|----------|
| C1 | Abstract overclaims "single variational principle" coverage | Partially correct | should-fix |
| C2 | LayerNorm claim too strong + missing citation | Partially correct (both sub-points valid) | should-fix |
| C3 | Separate GL(K) theorem / GL⁺(K) flat implementation / future non-flat | Partially correct (terminological half right, substantive half wrong) | should-fix |
| C4 | Transformer recovery is conditional, not unconditional | Partially correct | should-fix |
| C5 | FFN / activation section says too much at line 1943 | Partially correct | should-fix |
| C6 | RG section should stay conjectural, not on par with the KL derivation | Moot / already addressed | no-change |

Overall pattern: none of the six is wholesale wrong, and none is wholesale right. Five identify a genuine but narrow defect that the manuscript's own body/tables already contradict, so each fix is surgical (one sentence or one bib entry) rather than structural. The sixth describes the manuscript's existing posture rather than a defect. The recurring theme is an abstract-versus-body register mismatch: the body and the D/D♯/S/I status taxonomy (Table `tab:fep_nn_correspondence`, the document's third table) are honest and well-hedged; a handful of headline sentences in the abstract and section openers still carry derivation-grade verbs for items the taxonomy codes S or I.

## C1 — Abstract overclaims "single variational principle" coverage

Verdict: PARTIALLY CORRECT, should-fix, high confidence.

What is right: the abstract's umbrella verb at line 47 ("standard transformer architectural choices are **recovered as special cases** of the variational geometry ... **Under specific limits of this principle**") spans a heterogeneous list that the manuscript's own taxonomy splits across tiers. ALiBi, T5, and sliding-window are coded **S** (lines 1726/1727/1729; caption at 1760: "the framework explains the component's role but does not uniquely predict its specific form"). In this paper "limit" is a derivation operation (the D-row mechanism column reads "concentration of measure," "Euler discretization"), so describing a hand-chosen free function `pi_j` as a "specific limit of this principle" presents a choice as a derivation. The body itself uses the opposite, correct verb at line 856 ("accommodated as a particular choice of `pi_j` ... rather than derived from first principles") and trichotomizes status at line 1938.

What is overstated in the criticism: of its three cited loci, two need no change. Line 700 ("engineered to make the softmax form of `beta` its **exact KKT stationary point**") is correctly pure-D and bound to the softmax; line 856 already contains the demanded hedge verbatim. The criticism's "Table 1" pointer is a mislabel: the D/S/I taxonomy is in the third table (`tab:fep_nn_correspondence`, ~line 1705), not Table 1 (the notation table at line 370).

Recommended edit (abstract only): at line 47 replace the tail "... and causal masking and positional biases from non-uniform attention priors `pi_j`." with wording that keeps the legitimate D claims (KL/softmax, temperature, causal mask) but marks the S items as accommodations, e.g. "... and causal masking from a hard prior constraint, while ALiBi, T5, and sliding-window positional biases are accommodated as choices of the attention prior `pi_j` — the framework fixing the form through which any such prior enters, not the choice itself (see the D/D♯/S/I status taxonomy of Table~\ref{tab:fep_nn_correspondence})." Do not edit line 700 or 856; do not rewrite the body. Introduction line 58 ("full suite ... emerges as consequences of a single variational principle") is a softer secondary candidate, framed there as the prior-literature gap, left to author discretion.

## C2 — LayerNorm claim too strong + missing citation

Verdict: PARTIALLY CORRECT (both sub-points valid), should-fix, high confidence.

What is right (citation): a grep over all references.bib entries for `layernorm|1607.06450|kiros|rmsnorm|qk.?norm|key.?norm|Ba.*Hinton|Layer Normalization` returns zero matches; the only Hinton hits are unrelated, and the only "Henry" is the book "The Education of Henry Adams." LayerNorm is invoked without a citation at `GL(K)_attention.tex:1257, 1272, 2041, 2058` and `GL(K)_supplementary.tex:756`.

What is right (too strong): lines 1272 and 2041 say LayerNorm "**eliminates** the source of key-norm bias at its algebraic origin" and 1257 says it makes `||mu_j||` "**constant across tokens**." This contradicts the manuscript's own Appendix E, which measures `CV(||K||^2) = 0.240` (24% spread, explicitly "not constant," `supp:764`) and reports the residual key-norm bias as a confirmed effect (`|rho| = 0.256` for beta at `supp:785`; Cohen's d = 1.43 at line 2058). A paper cannot both eliminate the bias and headline it as a confirmed large effect. Mechanism check: in a standard transformer the key is `K_j = W_K · LayerNorm(x_j)`; LN fixes only the standardized pre-gain norm to sqrt(d), and the learnable elementwise gain and `W_K` rescale per coordinate, so post-projection key norms are not constant. External corroboration: QK-Normalization (Henry et al. 2020, arXiv:2010.04245) and rogue/outlier dimensions from the LN gain (Timkey & van Schijndel 2021) exist precisely because LN does not norm-control the projected key.

What is overstated in the criticism: line 2295 is already correctly hedged ("layer normalization is one mechanism that achieves this condition"); the table row at 1734 is already coded S; lines 1321/1327/1339 already use "approximately." The criticism's third request — propose or report a direct key-norm measurement — is already satisfied by Appendix E (`supp:764`).

Recommended edit (two surgical fixes, no new experiment):
(A) Add to references.bib and `\citep` at first mention (line 1257):
`@article{ba2016layernorm, title={Layer Normalization}, author={Ba, Jimmy Lei and Kiros, Jamie Ryan and Hinton, Geoffrey E.}, journal={arXiv preprint arXiv:1607.06450}, year={2016}}`.
Optionally cite Henry et al. 2020 (QK-Norm) and Timkey & van Schijndel 2021 as evidence that post-projection key norms are not LN-controlled.
(B) Reconcile the overstated sentences to the manuscript's own line-2295 wording: at 1257, "Layer normalization is one approximate mechanism promoting key-norm control, regularizing `||mu_j||` toward a common scale across tokens (the learnable gain and `W_K` leave a residual spread; see Appendix E)"; at 1272 and 2041, soften "eliminates ... at its algebraic origin" to "suppresses ... up to the residual fluctuation measured in Appendix E (`CV(||K||^2) ≈ 0.24`)." Leave 2295, the S-coded table row, and the already-approximate lines as-is.

## C3 — Separate the GL(K) theorem, the GL⁺(K) flat implementation, and the future non-flat theory

Verdict: PARTIALLY CORRECT — terminological half right, substantive half wrong; should-fix, high confidence.

What is right (terminological): the trained artifact is built from `Omega_ij = exp(phi_i) exp(-phi_j) ∈ GL⁺(K)` (line 633), an identity-component, globally trivial, flat-by-construction object (line 637; supplementary line 56). Yet it is named with the full-group symbol `GL(15)`/`GL(10)` throughout: the abstract line 49 mixes both in one sentence-group ("training GL⁺(K) gauge transformers ... a GL(15) gauge transformer"), and the tables (2075/2076/2077/2078, 2220/2221) and body (2242/2245/2289/2386, headings 2060/2155) use GL(15)/GL(10) for the flat trained model. Reserving plain `GL(K)` for the Theorem-1 invariance group and using `GL⁺(K)` for the implementation would remove the slip.

What is wrong in the criticism: the three notions are not conceptually blurred. The det>0 restriction and the deferred reflection sector are stated explicitly at line 582 ("our exponential parameterization restricts to det>0 ... Disconnected transformations are reserved for future study") and 637; the invariance theorem's proof at 554 (the `(det Omega)^2` factors cancel) holds for any sign of det, so Theorem 1 genuinely needs the full `GL(K)` and is not secretly `GL⁺`. The blue team's "self-contradiction" at the "Full General Model" heading (2155) was a misreading: "general/unreduced" there contrasts the anisotropic non-trivial-transport experiments against the degenerate-limit BERT check, not the full disconnected group.

Recommended edit: apply the symbol fix both sides concede. Rename the trained artifact `GL⁺(15)`/`GL⁺(10)` at abstract line 49, intro line 68, body line 2063, table rows 2075/2076/2220, and line 2386; rename headings 2060 and 2155 to "GL⁺(K) Language Modeling" (keep the "Full General Model" subtitle, which is correct). Reserve plain `GL(K)` for Theorem 1 and the unrestricted framework. Optional one clause near line 49 or 637 distinguishing "the GL(K)-invariant score (Theorem 1)" from "the GL⁺(15) flat vertex-frame implementation." Do not adopt the framing that this is a conceptual blur or a missing-reflection overclaim — that half is incorrect.

## C4 — Transformer recovery is conditional, not an unconditional recovery

Verdict: PARTIALLY CORRECT, should-fix, high confidence.

What is overstated in the criticism: its "without qualification" premise fails at three of its cited loci. Line 1525 already reads "The natural gradient recovers the standard transformer update rule **under a chain of limits**" and closes "without imposing the additional projection-absorption limits"; abstract line 47 already states "recovered, **up to a head-space-factor equivalence class** ... `W_Q` and `W_K` are not themselves gauge transformations"; line 2043 says the projections "realize **one orbit representative**." The praised caveat block at line 1374 is intact.

What is right: line 1370 boxes "**identical** to the standard transformer attention update `z_i = sum_j alpha_ij V_j`" with no inline conditional, and that value route depends on the rectangular `U_V^a` embedding (line 1362) that the manuscript's own scope paragraph at line 1300 ("It does not derive the rectangular subspace embeddings `U_Q^a, U_K^a, U_V^a`") and the S-coded table row at 1721 explicitly exclude from the derivation. That is an internal modality mismatch. The section opener at line 1039 ("This **establishes** neural network training as a limit ...") is also stronger than its neighbors.

Recommended edit (two inline softenings, no table-tier change): at line 1370, qualify "identical ... on the head subspace, once the structural subspace embeddings of \S\ref{sec:value_aggregation} are fixed (cf. the structural `U_V^a` row of Table~\ref{tab:fep_nn_correspondence})"; at line 1039, soften "establishes" to "frames neural network training as a limiting case ... (under the limits enumerated below)." Lines 1525, 2043, 47, and 1374 need no change.

## C5 — FFN / activation section says too much at line 1943

Verdict: PARTIALLY CORRECT, should-fix, high confidence.

What is right: line 1943's topic sentence says "the activation functions used in practice (SiLU, GELU, ReLU) are **specific instantiations of this dynamics under successive limits**." Since the caption at 1760 defines D as "mathematically exact or follows from explicit limits" and "limit" is load-bearing in this paper, "under successive limits" asserts the D modality. But the same subsection codes the GELU/SiLU row **I** (line 1751) and states at line 1994 "the correspondence is one of **family membership rather than functional identity** ... does not uniquely predict GELU over SiLU" (Gaussian PDF for the VFE gate versus Gaussian CDF for GELU; energy proportional to x for SiLU versus x^2 for the VFE gate). That is a genuine D-versus-I contradiction for GELU and SiLU.

What is overstated in the criticism: the "section should be downgraded" framing is already met — the row is coded I (the weakest non-derived tier), and line 1938 pre-frames the subsection as accommodation. ReLU genuinely is the zero-temperature limit of the binary gate (line 1996) and the GLU skeleton (Eq. `vfe_glu`) is exact; both earn the limit-language. The three activation citations are present and correct.

Recommended edit (one clause, on line 1943 only): replace the final sentence with "The GLU skeleton is exact, and ReLU arises as the zero-temperature limit of the binary gate; the smooth gates used in practice (SiLU, GELU) belong to the same Gaussian-gated family without being functionally identical to it (Status I, Table~\ref{tab:fep_nn_correspondence})." Optionally soften the preceding "not an independent architectural choice but a natural consequence" so it predicts the gated-linear form, not the specific GELU-versus-SiLU choice. Leave 1994, the I-row, the ReLU limit, and the GLU skeleton untouched.

## C6 — RG section should stay conjectural

Verdict: MOOT / ALREADY ADDRESSED, no-change, high confidence.

Both of the criticism's demands are already met. (1) The RG result is kept conjectural and separated from the KL Theorem: the subsection title literally begins "Conjecture" (line 2311), the Proposition is fenced as pure CLT bookkeeping with no gauge content (lines 2314, 2352, 2366: "the gauge content of the framework does not enter the calculation ... remains an open conjecture"), the empirical claim is walled into a separate Conjecture stated open twice, and the deviating measured exponents (`y2 ≈ -0.6, y3 ≈ +0.2` versus the predicted `-1`) are reported as not discriminating the two hypotheses — self-criticism stronger than the criticism requests. (2) The result is not on the footing of the KL derivation: a grep confirms it is absent from the abstract, the two-item Summary of Contributions, and the Conclusion. The CLT scaling dimensions were verified as correct textbook averaging (`g1` averages n perturbations, `n^-1/2`; `g2`/`g3` average `n^2` edges, `n^-1`), matching both teams' independent numerical reproductions.

Recommended edit: none required. At most a zero-severity optional half-clause at line 2314 noting "this is not a contribution on the footing of Theorem~\ref{thm:glk_invariance}," but the existing "We separately conjecture — but do not claim to have validated on trained models" already conveys the subordinate status.

## Suggested action order

All five actionable items are surgical and independent. A reasonable order:
1. C2(A): add the Ba/Kiros/Hinton 2016 LayerNorm bib entry and cite at line 1257 (the only purely mechanical, unambiguous fix).
2. C2(B): soften "eliminates"/"constant" at 1257/1272/2041 to match the manuscript's own line-2295 wording and the Appendix E measurement.
3. C3: make the GL⁺(K) artifact naming consistent (abstract, intro, tables, body, headings).
4. C5: repair the line-1943 topic sentence.
5. C4: soften "identical" at 1370 and "establishes" at 1039.
6. C1: qualify the abstract umbrella verb at line 47 with the form/choice distinction.

C6 needs no change. Items 2, 4, 5, 6 are all the same underlying issue (a few headline verbs out of register with the D/S/I taxonomy that the body already uses correctly), so they could be applied in one consistency pass.

## Edits applied (2026-06-22)

All fixes were applied to `Manuscripts-Theory/GL(K)_attention.tex`, `Manuscripts-Theory/GL(K)_supplementary.tex`, and `Manuscripts-Theory/references.bib`, then synced byte-identically to the Research-vault `manuscripts/` copies. The LayerNorm bibliography entry was added to both `references.bib` files preserving each file's line endings (repo CRLF, vault LF). No working-tree WIP was touched (`train_vfe3.py`, `.obsidian/*`, `meta-entropy-manuscript.md`).

Verification: every `\ref`/`\eqref`/`\cite` target introduced resolves to an existing `\label`; brace counts balance in both files; no banned `\,`/`\;`/`\!` spacing macros were introduced; all `GL(15)`/`GL(10)` artifact names were converted to `GL^+`. The math behind B1 (envelope versus surrogate prefactor) and B3 (generalized geometric-mean exponents) was sympy-verified. A standalone LaTeX compile was not possible here because the journal class `jmlr2e.sty` is not installed in this environment (pre-existing, unrelated to these edits); verification therefore rests on the static checks above.

First batch (C1 through C5; C6 needed no change):

- C1 (abstract, line 47): the umbrella "positional biases from non-uniform attention priors" clause now states that ALiBi, T5, and sliding-window biases are accommodated as choices of the attention prior, with a pointer to the D/D-sharp/S/I status taxonomy.
- C2 (lines 1257, 1272, 2041, and references.bib): added `@article{ba2016layernorm}` (Ba, Kiros and Hinton 2016, arXiv:1607.06450) cited at line 1257; softened "enforce constant key-norms"/"eliminates" to "one approximate mechanism"/"suppresses ... up to the residual fluctuation (CV approx 0.24, Appendix E)".
- C3 (lines 49 and 637, plus all results, tables, and the trained-model labels): renamed the trained artifact from `GL(15)`/`GL(10)` to `GL^+(15)`/`GL^+(10)` throughout, and added a convention sentence at line 637 distinguishing the GL(K)-invariant score (Theorem 1) from the GL+(K) vertex-frame implementation.
- C4 (lines 1039, 1370): softened "establishes ... as a limit" to "frames ... as a limiting case (under the limits below)" and qualified "identical to the standard transformer attention update" with "on the head subspace, once the structural subspace embeddings are fixed".
- C5 (line 1943): replaced "SiLU, GELU, ReLU are specific instantiations under successive limits" with "the GLU skeleton is exact, ReLU is the zero-temperature limit of the binary gate, and SiLU/GELU belong to the same Gaussian-gated family without being functionally identical (status I)".

Second batch (B1 through B8):

- B1 (attention 1004, 1006): made explicit that `(alpha*)^2 b0/c0` is the gradient of `alpha* D_KL` taken in isolation with the regularizer omitted, whereas the canonical reduced free energy gives `alpha* dD_KL` by the envelope theorem; the two differ by `dR(alpha*)/dtheta`. Removed the sentence claiming the reference implementation evaluates the chain-rule form, which contradicted line 1582.
- B2 (attention 1011): relabeled the belief-dynamics equation as the entropy-suppressed (autograd) surrogate, naming the envelope/detached path the single canonical convention, consistent with lines 887, 1582, and 2106.
- B3 (supplementary, after the Appendix-H proof at 1308): added a remark giving the generalized geometric-mean exponents `alpha_i/(alpha_i+1)` and `beta_ij/(alpha_i+1)`, reducing to 1/2 and beta_ij/2 at alpha_i = 1.
- B4 (supplementary 1111): corrected the f-divergence local form from `int q_i f(q_i / Omega q_j)` to the Csiszar form `int Omega q_j f(q_i / Omega q_j)`, matching the theorem at 1214.
- B5 (supplementary 1331): corrected the real-group statement; SO(K) sits in GL(K,R), while U(N) and SU(N) enter only after complexification of the fiber, equivalently realification into GL(2N,R).
- B6 (supplementary 380, 284, 278): weakened the line-380 claim to beta-fixed convexity (not local attractivity, consistent with line 392); qualified the tau to 0 vanishing of the softmax-correction term as pointwise but non-uniform near attention ties (decay O(tau), non-monotone) at lines 284 and 278.
- B7 (attention 2340, supplementary 954): labeled the arithmetic transport average as a flat coordinate diagnostic, noting GL(K) is non-convex so the mean can be singular; a group-respecting coarse-graining would use a projected, log-Euclidean, or Karcher mean.
- B8 (supplementary 45, attention 2450): added Appendix I (prior-bank decode) to both appendix summaries.
