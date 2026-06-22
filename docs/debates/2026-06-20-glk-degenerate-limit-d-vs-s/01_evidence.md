# Evidence Pack — glk-degenerate-limit-d-vs-s

## Manuscript references (the claim under evaluation, NOT canon)

- `GL(K)_attention.tex:47` (abstract): "Two successive limits (isotropic covariances, flat gauge connection) recover the standard rule β_ij ∝ softmax(Q_i K_j^T / √d_k). Under full column rank, the invertible head-space factor of the bilinear form W_Q W_K^T can be identified with σ^{-2}Ω^{-T}, so standard transformer attention is recovered as a degenerate, isotropic, flat-bundle case of gauge-theoretic attention. The individual rectangular projections W_Q and W_K are not themselves gauge transformations."
- `GL(K)_attention.tex:568` (methods): "we shall demonstrate that the bilinear form W_Q W_K^T determining the attention logits in standard transformers can be identified with a gauge transformation. In the isotropic, flat-bundle limit, the constant gauge identification satisfies W_Q W_K^T = σ^{-2}Ω^{-T}. This identification operates within each head's projected subspace (the per-head matrices W_Q^a, W_K^a are rectangular, not individually elements of GL(d_k); see Section multihead)."
- `GL(K)_attention.tex:~1077-1186` (the untied-QK / Mahalanobis reduction): the isotropic-limit KL collapses to a quadratic form whose cross-bilinear coefficient is `−σ^{-2}Ω^{-T}`; `M_ij := Ω_ij^{-T}Σ_j^{-1} = U_i^{-T}U_j^T Σ_j^{-1}` (attn:1180); the three-limit chain (general untied → isotropic → flat) recovers `softmax(Q K^T/√d_k)`.
- The correspondence tables (Table 1 / Table 3) assign the reduction a status tier (D / S / I). The standard-attention recovery row and the W_Q W_K^T identification row are the rows at issue.

## Settled (verified ledger — do NOT re-derive; cited as background)

- Three-limit reduction: surjectivity of `M_ij` onto `GL(d)` and SVD-factorization existence — sympy + numpy. Single-head transport `Ω = (σ² W W^T)^{-1}` with square `W ∈ GL(d_k)`; multi-head head-space factor `A^a = (σ² A_K^a (A_Q^a)^T)^{-1} ∈ GL(d_head)` (thin-SVD) — numerical [residuals 3.3e-11] [pass7].
- `eq:per_head_temperature` (G2): the full KL (carrying the ½) divided by `κ_a √d_head` gives effective `2√d_head`, equal to the dot-product `√d_head` softmax under the constant-key-norm condition — numerical [4.4e-16].
- Ledger §3 adjudication "G1": the untied-QK "joint realizability / expressive power identical to learned W_Q W_K^T" strong framing — the thesis is proven PER-PAIR and the family is rank-structured by construction; only a minor closing-analogy imprecision remains [pass6, pass7].
- Ledger §2: "Table 1 positional-bias rows ALiBi/T5/window reclassified D→S; causal mask and RoPE kept D"; "L-layer transformer ↔ L natural-gradient steps reframed as a structural correspondence." (Precedent: the project HAS demoted reduction-style claims from D to S when they were choose-the-prior/structural rather than forced.)

## Canon (source of truth — external)

- `[Vaswani 2017 §3.2.1]` scaled dot-product attention: `Attention(Q,K,V) = softmax(QK^T/√d_k)V`. The bilinear `W_Q W_K^T` is a single learned matrix per head; standard attention places no positive-definiteness, invertibility, or congruence-transport structure on it.
- The standard distinction (philosophy of science; Lakatos, Cartwright): a DERIVATION shows the special case FOLLOWS NECESSARILY from the general theory under stated assumptions; a STRUCTURAL CORRESPONDENCE (analogy) shows the two share a functional form under a re-identification of symbols. Setting general parameters to special values to MATCH an existing object is a reduction; RE-LABELLING an existing learned object as an instance of a general construction is an identification.

## What this evidence does NOT settle

1. Whether "standard attention is RECOVERED as a degenerate limit" is a derivation (the general theory FORCES the dot-product form once isotropy + flatness are imposed) or an identification (the dot-product form is MATCHED by choosing Ω so that σ^{-2}Ω^{-T} equals the already-learned W_Q W_K^T).
2. Whether the direction of the identification matters: standard attention learns an arbitrary `W_Q W_K^T`; the framework reads it as `σ^{-2}Ω^{-T}`. Does requiring `Ω ∈ GL` (so `Ω^{-T}` is any invertible matrix) make the identification vacuous (any bilinear can be so written) or substantive (the gauge structure constrains it)?
3. Whether the per-pair proof (ledger-verified) supports the GENERAL/universal "standard attention IS a degenerate case" claim, or only the existential "for each learned W there EXISTS a gauge Ω matching it" claim — and whether the table's D tier should reflect the latter weaker form.
4. Whether the manuscript's own hedge ("the individual rectangular projections W_Q, W_K are not themselves gauge transformations"; "the invertible head-space FACTOR") narrows the claim enough that D is earned, or whether the narrowing is exactly what makes it S.
