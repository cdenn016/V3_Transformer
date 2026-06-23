# GL(K) Attention — Pass-16 Deep Peer Review: Fixes Applied

Date: 2026-06-23
Base: `origin/main` @ `04897b1` (full B/C/D/E/F + pass-15 state). Branch: `manuscript/glk-peer-review-pass16`.
Verification: 32-agent investigator + adversarial-skeptic workflow (one pair per comment), each re-reading the live file before adjudicating. 10 edits applied, all in `GL(K)_attention.tex`; SUPP unchanged.

## Per-comment verdicts (10 major + 6 minor)

| # | Verdict | Action |
|---|---------|--------|
| M1 filtering objective | partial | The section already frames it as "one objective, two schemes"; scoped the lone overstatement ("standard coordinate-ascent belief update of mean-field VI") to **exact coordinate ascent on the per-position filtering factor** (position-i contribution, future-key couplings detached), reserving full F_red for the opt-in smoothing path. Author's framing preserved. |
| M2 temperature / eq-ref | fixed | Line 862 substitution now cites `F_align_canonical_tau` (tempered form) and the symbol `F_align^{(\tau)}`, not the τ=1 `mixture_energy_entropy`. Abstract already separated softmax-from-entropy-reg and 1/√d_k-from-concentration → no abstract change. |
| M3 key-side "prior" | fixed | `r_j` relabeled a **key-side logit bias** absorbed into the `log π_ij` slot as an algebraic rewrite, declared state-dependent (`r_j = μ_jᵀΣ_j⁻¹μ_j`), read as empirical-Bayes/source-salience — distinct from the pre-data positional priors (causal/ALiBi/T5). |
| M4 value row grade | fixed | Correspondence-table value-aggregation row **D → D♯**, matching its head-space siblings (QK and multi-head kernel rows) and the caption's D♯ definition; the absorption of Ω into the free W_V carries the non-uniqueness, but the aggregation form itself is derived. |
| M5 RoPE/DEQ rows | already resolved | Both already graded **S** (pass-15). No edit. |
| M6 FFN fence | fixed | Appended the disanalogy: a transformer FFN supplies learned per-token channel mixing via expansion/contraction matrices the bare Boltzmann gate lacks; the depth correspondence is **structural/interpretive pending deep-stack + rank-collapse diagnostics**. |
| M7 SU(N)/Lorentz (main text) | fixed | Line 2441 now qualifies: SO(N) and the Lorentz group SO(N−1,1) are **real** subgroups of GL(K,ℝ) directly; U(N)/SU(N) enter **only after complexification + realification** into GL(2N,ℝ). (Skeptic corrected the investigator draft, which wrongly lumped Lorentz with the complex groups.) GL(K)-vs-GL⁺(K) already explicit at 638–641; SUPP SU(N) already correct at 1337. |
| M8 "0-dim gauge theory" | already resolved | Line 2445 already reads "can be represented, under this reconstruction, as the 0-dimensional flat-gauge limit" (pass-15). No edit. |
| M9 App-H richness in theorem | already resolved | Theorem statement already carries the richness/normalizability hypothesis (pass-15 M4b). No edit. |
| M10 Rényi framing | already resolved | Abstract already flags Rényi-½ "rather than the exactly-derived KL"; in-text already lists geometric-mean / App-H uniqueness / App-B fixed-point as α=1-only. No edit. |
| min1 eq-ref 862 | subsumed | Same location as M2; M2's superset fix applied. |
| min2 "vanish under closure" | fixed | The uncertainty-correction terms `x_iᵀH_jx_i`, `tr(H_jP_i)` **become j-independent** under the closure (H_j→C⁻¹) and **cancel under the row softmax** — they do not vanish algebraically. Math re-verified by the skeptic. |
| min3 apriori | fixed | `apriori` → `a priori` (line 943; sole occurrence). |
| min4 rectangular W_V transition | already resolved | The rectangular factorization `W_V^a = U_V^a A_V^a` (W_O absorption) already precedes the boxed equality; scope para already disclaims deriving U_V. No edit. |
| min5 "two equivalent routes" | fixed | Softened: routes coincide on the strict dot-product form only after each route's closure (SPD-commutant for route 1, isotropic for route 2) plus shared key-norm control; before that they leave different residual key-side bias. (Skeptic fixed which closure belongs to which route.) |
| min6 line 588 generality | fixed | "does not restrict the generality of the derivations" → "is sufficient for the derivations shown" (product/subgroup reps make the equality a genuine restriction). |

## Integrity
Braces 4197/4197, begin/end 192/192, 0 banned spacing macros, 0 banned words introduced. Diff = 10 edits across 9 lines (M3 + min2 share line 1203). SUPP untouched.

## Notes
- Pre-existing "By leveraging" at line 2441 left untouched (outside the pass-16 criticisms; surgical discipline).
- Branch is off `origin/main`; the user's `feat/multiseed-digest` WIP (`train_vfe3.py`, `docs/edits/...`) was not touched.
