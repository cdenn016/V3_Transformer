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

## Batch 3 (D1-D10) edits applied (2026-06-22)

Verified by a 20-agent workflow (expert investigator + adversarial skeptic per claim; B1/B3-style math rechecked). All ten were CORRECT or PARTIALLY_CORRECT. Static checks after editing: refs resolve, braces balance, `\IF/\ENDIF` matched, no banned `\,` macros, no residual `SE(K)` or `GL(15/10)`.

- D1 (Algorithm 1, att): split the gauge-frame E-step into the reported-canonical schedule (phi held to the M-step, `eta_phi^E = 0`) and an optional in-E-step schedule via `\IF{eta_phi^E > 0}`, and clarified the caption. Matches runtime (`vfe3/config.py` `e_phi_lr` default 0.0, `m_phi_lr` default on).
- D2 (Algorithm 1, att): the covariance update now reads `Retract_SPD(Sigma, -2 eta Sigma sym(grad) Sigma)` with `sym` defined, matching the supplement's Fisher-Rao step (the printed gradient is already symmetric, so `sym` is the identity here, but the notation now matches).
- D3 (att): replaced "the transport operators Omega_ij between agents do [exhaust GL+(K)]" with the single-edge-expressivity statement plus the global-cocycle/DOF caveat (Lemma vanishing-holonomy).
- D4 (att): RoPE relaxes the constant-transport limit, not the flat-bundle condition (it stays a flat vertex-frame coboundary). Fixed the line-1905 and line-2041 prose and the interpolation-equation underbrace ("Standard attention (constant transport)").
- D5 (att): softened "unreduced theory/framework/forms" / "Full General Model" to "the full gauge model under a diagonal-covariance approximation" (and concise variants) at the abstract, intro, experimental design, section title, and conclusion; left the honest limitation at the diagonal-covariance-RoPE paragraph and the disclosures intact.
- D6 (att): narrowed "all gradient derivations carry over" (Renyi) to the softmax/attention-weight and product-rule gradients, and explicitly fenced the closed-form geometric-mean belief update, the forward-KL uniqueness theorem (Appendix H), and the covariance fixed-point algebra (Appendix B) as alpha=1 specific.
- D7 (supp): renamed the traceless symmetric summand `Sym(K) -> Sym_0(K)` in the Cartan decomposition (and its back-reference), reserving `Sym(K)` for all symmetric matrices; reconciled the line-565 "no retraction beyond clipping" claim with the implemented BCH/norm-ceiling/determinant-control retraction.
- D8 (supp): softened the gauge-frame validation claim (the matrix_exp autodiff check validates the differential convention, not an independent oracle for the assembled phi gradient) and recommended a central-finite-difference-in-phi check of the scalar free energy for parity with mu/Sigma.
- D9 (att): renamed the line-1582 "Killing-form Lie-group Riemannian gradient on gl(K)" to "implemented Cartan-involution-modified preconditioner," with the non-Ad-invariance note (Killing form degenerate/indefinite on gl(K); Ad-invariant only under O(K)).
- D10 (att): `SE(K) -> SO(K)` covariance at both the experimental-design (2063) and limitation (2384) occurrences (RoPE acts by rotations only, no translations); left the supplement's `SE` ("standard error") untouched.

## B1 fix checked against the codebase (`vfe3`)

The user asked whether the corrected B1 manuscript matches the implementation. It does, and the fix actually resolved a prior manuscript-vs-code mismatch.

- `vfe3/alpha_i.py` defines the manuscript objects verbatim: `alpha_regularizer` = `R(alpha) = b0*alpha - c0*log(alpha)`, and `alpha_state_dependent` = `alpha* = c0/(b0 + D_KL)`.
- The belief-gradient kernel uses the ENVELOPE coefficient `alpha*`, not the `(alpha*)^2 b0/c0` surrogate: `vfe3/gradients/kernels.py:285` calls `alpha_gradient_coefficient(...)` (which returns `alpha*`), and line 130 forms `self_mu = alpha_coef * (mu_q - mu_p)/sp`, i.e. `alpha* * dD_KL`. The docstring states "the coefficient is alpha* itself ... no product-rule correction is needed (R must be present in F)."
- The free energy includes `R(alpha)`: `vfe3/free_energy.py:377-378` adds `alpha_reg` to the self-term. So the envelope cancellation (chain terms through `alpha*` cancel against `dR(alpha*)`) is exactly the precondition the corrected manuscript states, and the autograd oracle differentiating the full F reproduces the same `alpha* dD_KL`.
- The `(alpha*)^2 b0/c0` surrogate is NOT used in the default path; it is the "differentiate `alpha* D_KL` alone, R omitted" path the manuscript presents as a distinct surrogate.

The old manuscript sentence I removed in the B1 fix ("Our reference implementation evaluates the corrected form (transformer/core/vfe_gradients.py)") had MISDESCRIBED the code: it claimed the implementation evaluates the product-rule `(alpha*)^2 b0/c0` form, whereas the code uses the envelope `alpha*` form ("no product-rule correction is needed"), and it cited a stale pre-rebuild path. The B1 edit therefore brought the manuscript into agreement with the implementation.

## Batch E (E1-E10) — verified via 20-agent investigator + adversarial-skeptic workflow

All ten verified CORRECT or PARTIALLY_CORRECT (none wrong); E8's body was already fixed by D4. Skeptics tightened several proposals to minimal-surgical scope. All edits in `GL(K)_attention.tex` except the SUPP `\editor` comment.

- E1 (CORRECT, att 474/503): the table column "Full Gauge Theory" overclaims — its own rows are the flat, 0D, vertex-frame case (connection "(0D: absent)", holonomy "Trivial"), with the genuinely non-flat construction reserved for edge-relaxed Regime II. Renamed header to "Flat Vertex-Frame Gauge Theory" and the caption to "the flat vertex-frame gauge-theoretic formulation". (Kept "Gauge Theory" framing per the skeptic; did NOT use "Regime I" — it is first defined ~1300 lines later, a forward reference.)
- E2 (CORRECT, att 480/418): (a) the table row labeled "Gauge transformation" for `Omega_ij = e^{phi_i}e^{-phi_j}` conflates transport with gauge transformation — relabeled "Edge transport `Omega_ij`" (the manuscript reserves "gauge transformation" for the per-agent `h_i`, footnote 429 / Cor. local-gauge-invariance). (b) the symbol-table parenthetical "`W_Q,W_K` (-> gauge transformations)" contradicts the abstract ("not themselves gauge transformations") and is wrong for rectangular matrices — changed to "the product `W_Q W_K^T = sigma^{-2} Omega^{-T}` is the transport; not individually gauge transformations".
- E3 (PARTIALLY_CORRECT, att 784): tau is not "without loss of generality" relative to the literal mixture KL (whose entropy coefficient is 1); introducing tau != 1 moves the minimizer from softmax(-E) to softmax(-E/tau). Replaced only the first sentence with explicit tempering language stating it is a genuine modeling choice recovering the temperature-1 KL only at tau=1; kept the already-accurate "This rescaling, together with multiplication of F by tau..." sentence and its hand-off into the canonical row-Lagrangian equation.
- E4 (CORRECT, att 411): the attention prior is row-conditional (causal `1/i` for `j<=i`, window, ALiBi, T5 all depend on query `i`) but the symbol table wrote `pi_j`. Changed the symbol-table entry to `pi_{ij}` with `sum_j pi_{ij}=1` and a clause licensing suppression of `i` where held fixed. (Skeptic: do NOT rewrite the body derivation, which is internally consistent with `i` fixed — minimal fix at the source only.)
- E5 (PARTIALLY_CORRECT, att 702/730): "generative model" is defensible as agent i's LOCAL/subjective model (standard mean-field message passing), so the critic's renames overcorrect; but the bare header/line-730 noun drops the local framing. Added "Local" to the subsubsection header and "agent $i$'s local" at line 730. (Dropped the investigator's parenthetical restating the held-fixed mechanism — line 715 already states it.)
- E6 (CORRECT, att 839): ALiBi (Press et al. 2022) is a one-sided causal bias `-m(i-j)` for `j<=i` with head-specific slopes, not the symmetric `-m|i-j|` written; "exactly" also conflicts with the table's "S" marking. Softened to "recovers the same additive linear-bias form ... on the causal support `j<=i` we have `|i-j|=i-j`, so the symmetric prior reduces there to ALiBi's one-sided penalty `-m(i-j)`."
- E7 (PARTIALLY_CORRECT, att 1718/1739/1743/1755): per-row remedy (two reword-keep-D, two D->S). (1) "Forward KL uniqueness -> Why dot-product attention | D": the App-H theorem forces the Gibbs/softmax FORM, not dot-product (that needs the separate isotropic+flat+key-norm reduction) — retargeted cell to "Gibbs/softmax attention form", kept D. (2) "F[{q_i}] -> Loss | D" -> S (whole-functional<->whole-loss, structural). (3) "Euler discretization -> Gradient descent (backprop) | D": Euler=GD step is genuinely D, but "(backprop)" double-counts the dedicated S row 1744 — dropped the parenthetical. (4) "p(o|mu)=Cat(softmax(W mu)) -> Output proj + softmax | D" -> S (positing the categorical-softmax model is a modeling choice; Gaussian->MSE equally available). Rows 1740/1741 correctly stay D.
- E8 (PARTIALLY_CORRECT, att 1920): the substantive claim (RoPE relaxes constant transport, not flatness) was ALREADY FIXED by D4 (body lines 1905/1927/2041 correct, bundle stays flat). One residual: the heading "Interpolation Between Flat and Full Gauge" implies the full GL(K) gauge is non-flat, but the flat-bundle theorem makes ALL three tiers flat — renamed to "Interpolation Between Constant and Full Gauge" (matching the body's "constant transport" endpoint).
- E9 (CORRECT, att 1272/2041): the LayerNorm-as-sl(K)-projection claim is an analogy, not a theorem (LN controls the embedding-vector radial norm; the R/trace direction of gl(K) is the gauge-frame determinant-scaling — distinct objects), and the table already marks it "S". Softened line 1272 ("quotients out ... projecting onto a sub-manifold" -> "plays a role analogous to removing this scaling degree of freedom, regularizing toward ...") and line 2041 ("is a projection onto the sl(K) subalgebra" -> "acts analogously to a projection toward ... regularizing the key-norm bias toward a token-independent constant that cancels under softmax"). Left the hedged line 1257 as-is.
- E10 (CORRECT, att+supp): commented out `\editor{TBD}` at line 39 of both files (safe under jmlr2e `[preprint]` — `\@editor` is only consumed in the non-preprint branch; not fabricating a name); removed the stray `\` line break inside the inline math at att line 557.

Verification: all old overclaiming strings gone; 7 new anchor strings present; zero banned spacing macros (`\,` `\;` `\!`); braces balance (4141/4141), begin/end 190/190; both `\editor` lines commented. Standalone compile still blocked by the pre-existing missing `jmlr2e.sty` (unrelated to edits).

## Batch F (F1-F9) — verified via 12-agent workflow (F4-F9) + direct checks (F1-F3); two user decisions

F4-F9 verified by investigator + adversarial-skeptic pairs (all CORRECT or PARTIALLY_CORRECT). F1/F2 were user decisions (asked, no fabrication); F3 was a user directive (build the table, fill values later).

- F1 (USER: point to this repo): Code Availability URL `github.com/cdenn016/epistemic-geometry` (reviewer reports 404) -> `\url{https://github.com/cdenn016/V3_Transformer}` (the actual push remote). Used `\url` so the underscore is handled. Did not fabricate a tag.
- F2 (USER: mark provisional): the RG flow table `tab:rg_flow_supp` `g1_tot` column violates its defining identity `g1_tot = g1_orig + g1_emer` (printed 42.47... vs the definitional sums 0.300, 2.471, 1.214, ...; already a DATA-PENDING TODO), and the y2/y3 fit windows are inconsistent. Corrected data not available, so NO fabricated numbers: added a provisional note to the caption (g1_tot and the fitted exponents pending re-extraction; identity stated; per-exponent windows to be stated on re-extraction), removed the broken-column `y1_tot ~= +0.01` figure from the prose while keeping the qualitative non-decay claim. Recompute/remove deferred to finalized data.
- F3 (USER: make the table): added `tab:glk_hyperparams` after `tab:glk_spec` with the full requested row structure (optimizer, LR schedule, batch size, token budget, gradient clipping, weight decay/alpha; temperature kappa, component LRs eta_q/eta_Sigma/eta_phi, prior precision, inner iterations T; seeds, checkpoint policy, hardware) and `TBD` placeholder cells to fill with finalized experimental values.
- F4 (PARTIALLY_CORRECT, att): largely already balanced (abstract, body 2251, TODO 2379 all carry the 48.5 caveat). One remaining unbalanced transformer-beating sentence in the Conclusion -> appended the param-matched caveat (118.6 embed-matched; 48.5 at d_model=1,280 "clearly lower"). Classical-baseline comparisons left intact.
- F5 (CORRECT, supp+att): demoted the n=5 BERT temperature-dispersion claims from causal/variable-importance ("dominant explanatory variable" x2, "strongest predictor") to exploratory associations scoped to five architectures, at supp 854/874(caption)/878 and att 2056 finding (iii). All correlation numbers (r-bar, r=-0.87, rho=-0.6, CVs) preserved.
- F6 (CORRECT, att abstract+intro): the body (att 2063) and tab:glk_spec caption already disclose "a linear vocabulary decode"; the abstract (49) and intro (68) omitted it. Added a clause disclosing the one retained learned linear map (logits = mu W^T for token decoding). Verified against code: vfe3/model/prior_bank.py:825 computes logits = x @ output_proj_weight^T (the linear decode reached when use_prior_bank=False). [FYI code note, not edited: vfe3/config.py:377 sets use_prior_bank=False as the dataclass default while in-source comments call True the "pure/default path" -- a comment-vs-default mismatch worth a separate look; does not affect the manuscript, since the table caption pins the reported runs to the linear decode.]
- F7 (CORRECT, att intro): added one sentence after the "not been fully unified" line citing the three FEP-critique papers already in the bib (bruineberg2022emperor, aguilera2022particular, biehl2021technical), scoping the paper to a concrete falsifiable derivation independent of the contested general-FEP claims.
- F8 (CORRECT, att 789): the main text bound the "consistent dual interpretation" inside the forward-KL uniqueness predicate (presenting it as a selection criterion); supp Dual Cost Identification (1327) correctly calls it a consistency property of the linearly coupled class. Rewrote att 789 to separate the genuine selection basis (exponential-family closure / App-H uniqueness, kept verbatim with the Appendix~H ref) from the dual interpretation, which now "follows as a consistency property ... rather than as a separate selection criterion". (App H itself, supp ~1107, is internally a touch looser; left to the author.)
- F9 (CORRECT, att fig:bundle_sections_surface caption): the bundle schematic draws GL(K) transport as a pure rotation (the transported ghost reuses q_j's ellipse shape, re-rotated only), with no disclaimer. Added a caption clause: the rotation is illustrative; a general GL(K) transport also includes shear and anisotropic scaling, not only rotation.

Verification: zero banned spacing macros in either file; braces balance (att 4178/4178, begin/end 192/192 after the +2 from tab:glk_hyperparams; supp 2466/2466, 116/116); old overclaim strings gone (strongest predictor, epistemic-geometry, the +0.01 exponent, "explained by"). Standalone compile still blocked by the pre-existing missing jmlr2e.sty.
