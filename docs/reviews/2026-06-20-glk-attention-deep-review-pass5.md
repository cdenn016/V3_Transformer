# GL(K) attention manuscript — deep review pass 5 (citation accuracy, logic/edge-cases, notation, claim-status, backlog)

Date: 2026-06-20. Targets: the migrated canonical copies at
`C:/Users/chris and christine/Desktop/Research/manuscripts/GL(K)_attention.tex` and `GL(K)_supplementary.tex`
(now the single source of truth across repos), cross-checked against `references.bib`, the companion `PIFB.tex`, and
the research wiki. All edits this pass were applied to the `Research/manuscripts/` copies. The V3 mirror
`Manuscripts-Theory/` was confirmed content-identical to the post-pass-4 canonical base before editing (only a CRLF/LF
line-ending difference on `references.bib`), so this pass began from the same text passes 1-4 left.

## Method and scope

Fifth review pass, five lenses chosen to avoid the territory of passes 1 through 4 (literature gaps; the seven
load-bearing proofs; code-fidelity and the RG g2 exponent; dimensional/figure/number consistency and framing). The
lenses were: citation *accuracy* (does each cited source support the sentence it is attached to, and do bibkeys point
to the right work — distinct from pass 1's gap hunt); logical rigor and edge cases on the secondary scaffold (lemmas
and corollaries outside the seven verified proofs, degenerate cases); global notation and definition consistency;
claim-status consistency against the paper's own D/D#/S/I taxonomy; and a triage of the un-applied minor backlog from
passes 1 and 3. Each substantive finding was handed to an independent adversarial verifier instructed to refute it
from the actual `.tex` and primary sources before it could be reported; high-severity findings were put to a
three-voter majority.

Tallies (from the workflow result): 41 raw findings, of which 12 were confirmed-open, 14 were verified
already-applied (valuable confirmation that prior-pass fixes are in place), 5 were deferred as needing data or new bib
entries, and 10 were rejected on verification. The verification layer again earned its place: it rejected the
"Cencov/Chentsov cited as two independent sources" escalation (it is the same duplicate-key pair already counted), the
"strict-convexity stated without the N>=2 condition" claim (the support condition is already stated at line 746), and
seven others on direct evidence.

## Confirmed findings — all applied this pass

### Notation consistency

The symbol `d_k` carried two incompatible meanings (full embedding dimension `= K` in the multi-head partition
`d_k = H d_head` and the ambient group `GL(d_k)`, versus the per-head dimension `d_head = K/H` in the dot-product
reduction and the temperature law `tau = sqrt(d_k)`), with the two readings colliding inside one correspondence. The
notation-conventions paragraph at line 571 now states both meanings and the `d_k = K` versus `d_k = K/H` reconciliation
explicitly and names `d_head` where the per-head value is operative; no equation was changed (high; the proposed
cross-reference to `sec:free_energy_section` was dropped because that section uses `K`/`K_q`, not `d_k`).

The glyph `\sigma` denoted both the logistic sigmoid and the belief standard deviation, colliding in a single
expression at eq:binary_silu (the outer sigmoid wrapping an inner `2\sigma^2`). The sigmoid was renamed to `\varsigma`
at all six function-call sites (lines 1956, 1959, 1961, 1965, 1972, 1977); every scale `\sigma` was preserved (high).

The standard dot-product attention scaling was written `1/sqrt(d)` at line 2037 but `1/sqrt(d_k)` (with `d_k = 64`)
one line later; the table reserves `d` for the ambient dimension and `d_k` for the head dimension, so line 2037 now
reads `1/sqrt(d_k)` (medium).

The bare `\lambda` was overloaded as the belief-alignment coupling (notation table) and the softmax-KKT Lagrange
multiplier (lines 743, 751); the multiplier was renamed to `\nu`, freeing `\lambda` for its table meaning (medium; the
"triple-purposed" reading was corrected by the verifier — the weight-decay and key-norm coefficients are already
subscripted `\lambda_p`/`\lambda_K`).

The glyph `\rho` was triple-overloaded (representation map, Pearson correlation, SPD trust-region clip radius); the
clip radius at supplementary line 640 was subscripted to `\rho_{tr}` (low; the correlation usage is conventional
function-call notation and was left).

### Claim-status taxonomy

The abstract equated layer normalization with "the geometric condition for frame-independent inference," while Table 1
rates the LN correspondence S and the Discussion says LN is only "one mechanism that achieves this condition." The
abstract now reads "one mechanism realizing the geometric condition," restoring the body's hedge (medium).

The introduction grouped LN (S) and training dynamics (S) with attention and temperature (D) under one verb,
"recovered as special cases or limits," flattening the distinction Table 1's caption draws. It now states attention and
temperature are "recovered as explicit limits" while LN and training dynamics "receive a structural account within the
same geometry" (medium).

The conclusion claimed the model "learns structured, head-specialized attention patterns," upgrading the abstract's
accurate frame-structure claim; the trained-GL(K) results measure categorical structure of the gauge frames `phi` and
per-head temperature dispersion, not head-specialized attention patterns. The phrase was removed (medium).

### Logic / edge cases

The geometric-bias positivity claim `S(Omega) >= 0` with equality iff `Omega in O(K)` was asserted without its
supporting inequality. A one-line eigenvalue justification was added at line 1120: writing `S` over the eigenvalues
`lambda_a` of `Omega Omega^T` as `(1/2) sum_a (lambda_a^{-1} - 1 + log lambda_a)`, each term has the form
`t - 1 - log t >= 0` with `t = lambda_a^{-1}`, vanishing only at `t = 1`, so equality holds precisely when
`Omega Omega^T = I` (low).

### Citation accuracy / bibliography

The shared `references.bib` carries roughly thirty duplicate entries (a CamelCase key and a lowercase key for the same
work, from merging two manuscripts' bib files). For six works **both** keys are cited within GL(K), so the GL(K)
reference list renders the identical work twice under inconsistent keys: `Bishop2006`/`bishop2006pattern`,
`Amari2016`/`amari2016information`, `Blei2017`/`blei2017variational`, `Friston2010`/`friston2010free`,
`Absil2008`/`absil2008optimization`, and `Cencov1982`/`Chentsov1982`. The safe, GL(K)-scoped fix was applied: every
GL(K) call site was canonicalized onto the ecosystem-canonical key (the one PIFB and belief_inertia already use), so
GL(K) cites each work once, **without deleting any bib entry** (the shared bib also serves PIFB, belief_inertia, and
meta_entropy; meta_entropy uses its own embedded `\bibitem` and is independent of `references.bib`). All six lowercase
keys now have zero GL(K) citations. Entry-level deduplication (deleting the duplicate entries, which would touch
PIFB/belief_inertia) is left for an ecosystem-wide pass (high, severity from the duplicate reference-list lines).

## Verified already-applied (prior-pass fixes confirmed in place)

The `d_q`/`d_p` overload (pass 3) and the `\lambda -> \lambda_K` key-norm rename (pass 4) are present. The Sengupta
gauge-theory citation points to the correct 2017 preprint, not the 2018 synchronization paper. All 101 keys cited in
GL(K) resolve in `references.bib` (no compile-breaking citations). The Pennec affine-invariant exp-map citation is
already correct (`Pennec2006`). The positional-prior "first-principles" wording is already handled by the
derived/explained/accommodated taxonomy. The Renyi alpha-divergence singular-at-`alpha=1` limit and the covariance
fixed-point local-stability caveat are already stated. Several claim-status rows (the `W_Q W_K^T` D# hedge, the
GELU/SiLU I-tier family-membership language, backpropagation as S, the multi-head block-diagonal restriction) are
consistent across abstract, body, and conclusion. The `geshkovski`, `dong2021attention`, `press2022train`,
`chen1998empirical`, `culver1966existence`, `chung2015recurrent`, and `xiao2024efficient` attributions were verified
accurate against the wiki source notes.

## Deferred — surfaced for user decision (require new bib entries and new prose)

These are pass-1 overlooked-connection recommendations, repeatedly downgraded and never applied, that add new
scholarly assertions and citations the author should vet. Each is ready with a target location and a draft sentence:
the Wang-2023 SMSA contrast at Value Aggregation; the von Oswald depth-as-gradient-descent citation at the "L
natural-gradient steps" site; the Karcher/Moakher/Jeuris affine-invariant-mean contrast at the covariance barycenter;
the RG H3 spectral-clustering resolution-limit hypothesis with Fortunato/Newman-Girvan; and the
QRF/Bayesian-mechanics/FEP-critique/philosophy-of-science scope sentences. All need entries added to `references.bib`
(Wang2023, von Oswald, Moakher, Jeuris, Fortunato, Newman-Girvan, and the philosophy-of-science set), which is why
they were not auto-applied as surgical copyedits.

## Rejected on verification

Cencov = Chentsov "two independent sources" (it is the single duplicate-key pair already counted in the bib finding);
App-H reverse-implication quantifier requiring non-normalizable neighbor densities (the construction is admissible);
strict-convexity stated without the `N >= 2`/nonempty-support condition (the active-support condition is already at
line 746); homogeneous-limit covariance uniqueness from a "scalar-looking" identity (the identity is correctly
matrix-valued); the O(K) reflection-extension holonomy-sign claim; the sliding-window prior boundary-normalizability
nit; the `d_k` overload "inside the Table 1 caption row set" (duplicate of the confirmed N1); the conclusion's
"surpassing classical baselines" superlative (supported, an asymmetric-framing nit not a taxonomy contradiction); the
SO(N)/SO(3) Rodrigues heading (the body already restricts the closed form to SO(3)); and the Bogacz/Rao
precision-weighted-prediction-error citation at the GLU gate (the verifier judged the existing intro-list placement
adequate).

## Completeness critic — gaps no pass (1-5) has examined

Five substantive questions surfaced for a future pass, none auto-actioned:

1. The untied-QK realizability claim (lines 1166-1186) that "as `(U_i, U_j)` range over `GL(d_k)^2` and `Sigma_j` over
   SPD, `M_ij` realizes any element of `GL(d_k)`" proves only per-pair reachability; whether the shared per-token
   factorization simultaneously realizes an arbitrary indexed *family* `{M_ij}` is asserted, not shown, and it
   underwrites the "expressive power identical to learned `W_Q W_K^T`" thesis (a D# row).

2. The per-head temperature denominator `kappa_a sqrt(d_head)` (eq:per_head_temperature) carries no factor of two,
   while the KL numerator it divides is the squared-distance form that the BERT section says predicts `2 sqrt(d_k)`;
   the `kappa_a = 1` reduction may land on `2 sqrt(d_head)`, not `sqrt(d_head)`. Whether the per-head equation is
   internally consistent with the pass-3/4 factor-of-two reconciliation was never checked.

3. The per-head holonomy subsection (lines 1782-1788) opens by calling itself "vacuous as written" (Regime I), then
   derives a direct-sum holonomy and an additive Wilson observable for a Regime-II construction deferred to the
   companion; the scope (does a self-labeled-vacuous derivation belong here) and the additive factorization
   `W_ijk = sum_a W_ijk^(a)` (which omits identity-block contributions) are unexamined.

4. The causal-mask / ALiBi / sliding-window / T5 rows are stamped D in Table 1, but the prior `pi_k` that produces
   them is imposed, not derived — the same choose-the-prior-that-reproduces-a-known-bias move that pass 4 reclassified
   from D to S for layer normalization. Whether these four positional-structure rows should be S deserves a check.

5. The BERT key-norm-bias effect size (Cohen's `d = 1.43`) comes from a sampler the manuscript itself flags as
   non-converged (`R-hat ~ 1.02`, limited ESS); the same number is re-stated at line 2270 *without* the caveat and
   used to support a "non-trivial quantitative prediction." Whether a non-converged estimate should be load-bearing,
   and whether the caveat must propagate to every restatement, was not audited.

## Applied this pass

`GL(K)_attention.tex`: 47 (abstract LN hedge), 62 (intro D/S split), 571 (`d_k` disambiguation), 1120
(`S(Omega) >= 0` eigenvalue justification), 1956-1977 (sigmoid `\sigma -> \varsigma`, six sites), 2037
(`sqrt(d) -> sqrt(d_k)`), 2417 (drop "head-specialized attention patterns"); plus citation canonicalization at 58, 60,
520, 672, 885, 947.

`GL(K)_supplementary.tex`: 743/751 (Lagrange `\lambda -> \nu`), 640 (`\rho -> \rho_{tr}`); plus citation
canonicalization at 52, 204, 237, 629, 1319.

Verification: braces balance (main 3966/3966, supplementary 2347/2347); `\varsigma` present at 7 sites with no
leftover sigmoid `\sigma(`; all six lowercase duplicate bib keys removed from GL(K) with canonical targets confirmed
present; `\eqref{eq:geometric_bias}` resolves; the pure insertions introduce no horizontal rules, em-dashes, LaTeX
spacing macros, or claudeisms.
