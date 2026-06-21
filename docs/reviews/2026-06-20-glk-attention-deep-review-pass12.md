# GL(K) attention manuscript — deep review pass 12 (cross-manuscript consistency)

Date: 2026-06-20. Targets: `GL(K)_attention.tex`, `GL(K)_supplementary.tex`, and `PIFB.tex` (the full general theory), all canonical in the Research vault `manuscripts/`.

Pass 12 opens the inter-document axis. Passes 3 through 11 reviewed each GL(K) file along intra-document axes and against the V3 code; PIFB had its own rounds 9 through 12. None checked whether the three papers are mutually consistent — whether GL(K) is a correct specialization of PIFB's general theory, whether shared equations agree term-for-term, whether shared notation denotes the same objects, and whether a claim graded one way in one paper is graded the same way in another. That is this pass.

Method: five lenses dispatched in parallel — variational/free-energy, gauge theory, information geometry, notation/symbol/cross-reference, and philosophy/claim-status — each seeded with the verified-ledger do-not-reflag list and instructed to review consistency ACROSS the documents only (the internal correctness of each document is settled). Each recomputed shared identities where useful.

Headline: the GL(K) papers are a faithful specialization of PIFB's general theory, with four cross-document inconsistencies found — none in the mathematics itself (every shared transport, holonomy, free-energy, Fisher, and divergence identity agrees term-for-term and, where recomputable, to machine precision). The four are a dropped coefficient on an inert display equation, a stale framing sentence that PIFB had already corrected on its own side, an unbridged notation symbol, and a companion-citation bibkey tangle. Three are applied; the fourth is deferred to the author because it turns on which bibkey is canonical for each paper. No high/critical finding.

## Recommendation

Minor revisions, three applied. The applied items harmonize the supplement to PIFB's canonical/corrected statements and bridge one notation symbol; none changes a result. The deferred citation item needs an author decision.

## Findings applied

### F1 — Supplement's two-channel free energy dropped the `λ_h` weight and over-claimed completeness (MEDIUM, variational)

`GL(K)_supplementary.tex` eq:free_energy_full_supp called itself "the most general formulation / the complete free energy" but wrote the model-channel term `Σ_i KL(s_i‖r_i)` with implicit unit weight and omitted both attention-entropy terms, whereas PIFB's canonical general F (eq:free_energy_functional_final) carries `λ_h Σ_i χ_i KL(s_i‖r_i)` with `λ_h` an explicit free weight (PIFB defines the `λ_h∈{0,1}` regimes), and GL(K)_attention (~679) names `λ_h` plus both entropy terms as constituents of the complete two-channel form, routing the complete object to the companion paper. The supplement never mentioned `λ_h` (grep: 0 hits). The entropy omission was already adjudicated candid in the ledger (§3, pass7); the `λ_h` omission was new and not covered. Severity is bounded because the supplement operates entirely in the `γ=0`, models-fixed regime, so the model channel is dynamically inert in everything it computes — this is a labeling/specialization inconsistency, not a wrong result.

Applied: added `\lambda_h` to the model term in eq:free_energy_full_supp, and reworded the lead-in from "the most general formulation … the complete free energy" to "the two-channel free energy … with hyper-prior weight `λ_h` … and the attention and meta-attention entropy terms suppressed in this display (the complete formulation … is given in the companion paper)."

### F2 — Supplement framed the dual-cost as a co-selector of forward KL; PIFB had corrected this (MEDIUM, information geometry)

`GL(K)_supplementary.tex` Interpretations (~1321) stated "the forward KL is the only divergence that simultaneously yields a closed-form Boltzmann solution … and a consistent dual cost," and the Summary said the route "selects the forward KL from the requirement of a closed-form Gibbs update together with a consistent dual attention cost." But the envelope dual-cost identity `C_ij = ∂F_i/∂β_ij = D(q_i*, Ω_ij q_j)` holds for every divergence entering F linearly in `β_ij` (verified numerically: it holds for the χ²-divergence to 2e-7), so it cannot select the forward KL — the selection is done entirely by exponential-family closure of the stationary solution. PIFB had already corrected exactly this on its own side (PIFB:4698: "a consistency check … rather than an additional selection principle"; PIFB:4701: "the dual cost … equals the divergence itself, by the envelope theorem, for every member of the linearly coupled class; what singles out the forward KL is that it alone also yields the closed-form … Boltzmann solution"). The ledger's Round-10 adjudication examined only PIFB's abstract, not the supplement, so the two finalized documents disagreed on what does the selecting.

Applied: harmonized the supplement's two sentences to PIFB's corrected language — the dual cost equals the divergence by the envelope theorem for the whole linearly coupled class (a consistency property, not a selection criterion), and what singles out the forward KL is the closed-form Boltzmann solution; the Summary's "together with a consistent dual attention cost" demoted to "the dual attention cost then following as a consistency property."

### F3 — Model-fiber dimension `K_p` (GL(K)) vs `K_m` (PIFB) unbridged (LOW-MEDIUM, notation)

The model/generative-model fiber dimension is the same object in both papers but is written `K_p` throughout GL(K) (12 uses) and `K_m` throughout PIFB, and neither document stated the correspondence — even though the belief-fiber `K↔K_q` correspondence is explicitly bridged in GL(K)'s notation paragraph. (The PIFB-internal `K_p→K_m` refactor recorded in the ledger Round-10 never propagated to the GL(K) papers.)

Applied: added one clause to GL(K)_attention's notation conventions paragraph — "The model fiber dimension is written `K_p` (denoted `K_m` in the companion general theory)."

## Finding deferred to the author

### F4 — Companion-citation bibkey/title tangle; instantiation relationship asserted one-directionally (MEDIUM)

GL(K)_attention cites PIFB as the companion paper under `Dennis2025it`, whose bib title ("A Theoretical and Computational Implementation of a Participatory 'It From Bit' Universe") is stale relative to PIFB's current self-title ("A Gauge-Theoretic Framework Toward a Participatory 'It From Bit' Program …"). PIFB reciprocally cites a transformer companion under `Dennis2025trans` ("Implementing Attention and Transformers without Neural Networks: Validation of Gauge-Theoretic Transformers"), which matches neither this manuscript's self-title ("Attention as Gauge-Theoretic Variational Inference") nor the third, dead key `Dennis2025atten` ("Attention, Transformers, and Backpropagation are Degenerate Limits …"). So the two papers' reciprocal "companion" pointers do not resolve to each other by title, and GL(K) never explicitly states that it is the language-model instantiation/specialization of PIFB's general theory (PIFB makes the reciprocal Level-1 move; GL(K) does not). `_PROVENANCE.txt` carries a divergence warning, suggesting drifting repo copies as the root cause.

Not applied: this turns on author intent — whether there are genuinely three transformer-related papers (an attention paper, a validation paper, and a degenerate-limits paper) or one paper with stale bib metadata. Guessing which bibkey is canonical for each manuscript could be actively wrong. Recommended author actions: (a) reconcile the three `Dennis2025*` bib entries so each paper's "companion" citation resolves by title to the document it means; (b) remove or wire the dead `Dennis2025atten` key; (c) add one clause to GL(K) stating it is the language-model instantiation of the companion general framework. Recorded as open in the ledger (§4).

## Cross-document consistency confirmed (no finding)

- Gauge theory (no findings, all clean): covariance sandwich `ΩΣΩᵀ`, mean transport `Ωμ`, dual `Ω^{-⊤}`, the untied-QK bilinear `M_ij = Ω_ij^{-⊤}Σ_j^{-1}` (byte-identical across attn:1180 and PIFB:1781), the left-invariant connection law, cocycle/holonomy/Regime-I flatness (`F=0`), the Regime-II edge factor, GL(K) as the correct specialization of PIFB's general `G` (`GL⁺(K_q)→GL(K)`, multi-head `GL(d_head)^H`), RoPE-as-SO(2)-subgroup, and the irrep/block decomposition — all identical or correctly specialized, recomputed to 1e-16 where applicable.
- Variational: the reduced F `−τ log Z`, the envelope identity, the canonical-vs-surrogate `−τ⁻¹ Cov_β` gap, the accuracy+complexity decomposition, the structural-EM caveat (reconciled with PIFB's belief-flow monotonicity statement, which is about a different object), and the mean-field/target-blind-E-step are mutually consistent.
- Information geometry: the forward-KL selection proofs (the supplement's `B=1` specialization of PIFB's general-`B` derivation), the Fisher metric `G_μ=Σ⁻¹`, the negative-entropy-Bregman/Fisher-Rao-Hessian identification, and the Cencov/Petz/Amari classical-uniqueness framing agree across docs.
- Notation: `Ω, φ, τ, κ, α_i, β_ij, γ_ij, λ_h, σ², M=W_Q W_Kᵀ`, the q/p/s/r hierarchy, and `K↔K_q`, `d_k↔d_head` all denote the same objects across the three documents; the GL(K)_attention↔supplementary cross-references resolve in both directions.
- Claim-status: the copula ("attention is variational inference" → D-tier rule), holonomy↔compositional-language (open, Regime-II-conditional), the RG fixed point (labeled conjecture), symmetry-breaking↔training (S-tier, explicit-not-spontaneous), standard-attention recovery (limit, not derivation), and the participatory/it-from-bit reading (absent in GL(K), Level-3-fenced in PIFB) are graded at the same tier in every document that states them.

## Method

Five expert-lens agents in parallel, each given all three vault manuscripts and the verified-ledger do-not-reflag list, restricted to cross-document consistency (shared equations agreeing term-for-term, correct specialization of PIFB→GL(K), shared-notation agreement, and claim-status parity), recomputing shared identities in python. The gauge lens returned no finding; the variational, info-geometry, notation, and philosophy lenses each returned one cross-document item; the four were reconciled here (three applied, one deferred to the author). No high/critical finding, so the adversarial-skeptic stage did not fire; the applied items were each re-verified against the live text of both documents before editing.
