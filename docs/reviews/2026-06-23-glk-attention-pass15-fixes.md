# GL(K) Attention - Non-Empirical Peer Review Pass 15: fixes applied

Date: 2026-06-23. Branch: manuscript/glk-peer-review-pass15 (off origin/main, full B-F manuscript state).
Method: 22-agent investigator + adversarial-skeptic workflow (one pair per comment), then surgical edits. Several comments were already addressed by prior batches (B-F) or existing hedges; those were verified and left untouched (no churn).

## Major comments
- M1 (PARTIALLY): abstract (45/47) and intro line 62 already aligned with the proof surface; the lone residual overclaim at intro line 58 ("full suite ... single variational principle") reworded to "the attention rule follows from a single variational principle, with the remaining ... recovered as explicit limits or given a structural account."
- M2 (PARTIALLY): SUPP "Bundle triviality" -- "promotes the bundle to a non-trivial principal G-bundle" overclaims (no topological cocycle obstruction / characteristic class is constructed; ATT keeps the bundle globally trivial). Changed to "equips the bundle with a non-flat discrete connection ...". Downstream cocycle-failure / Wilson-observable clauses kept.
- M3 (ALREADY_ADDRESSED): the RG meta-agent average Omega_AB already carries the full hedge in both files ("a diagnostic ... not a group-valued mean, since GL(K) is non-convex ... log-Euclidean/Karcher"). No edit.
- M4 (PARTIALLY): (a) the f-divergence base-measure form is already correct (int (Omega q_j) f(q_i/(Omega q_j)) dc at the summary and the proof; the theorem uses the generic int p f(q/p) with p = Omega q_j). (b) promoted the load-bearing richness + post-inverse-transport normalizability hypothesis from the proof into the App-H theorem STATEMENT.
- M5 (ALREADY_ADDRESSED): the SU(N) statement already reads the precise version (U(N)/SU(N) enter only after complexification / realification into GL(2N,R); subgroups of GL(N,C), not real GL(K,R)). No bare "GL(K) contains SU(N)" remains. No edit.
- M6 (PARTIALLY): table grade fixes. RoPE row D->S and tied-recurrence/DEQ row D->S (both architectural accommodations, not derivations). Rows 1718 (retargeted to "Gibbs/softmax attention form") and 1743 ("(backprop)" removed) were already fixed by batch E7 -> not re-touched.
- M7 (PARTIALLY): symmetry-breaking sentence recast -- a non-gauge-invariant likelihood breaking a redundancy is a gauge-fixing / coordinate-readout choice, not physical symmetry breaking; added the "express specialization via gauge-invariant quantities or a declared gauge-fixed diagnostic" safeguard. (Skeptic dropped a proposed ||mu_i|| example, since the Euclidean norm is not GL(K)-invariant.)
- M8 (ALREADY_ADDRESSED): the FEP-critique sentence (bruineberg2022emperor, aguilera2022particular, biehl2021technical) is already in the intro (batch F7). No edit.
- M9 (CORRECT): RoPE section intro "Here we derive RoPE" -> "Here we recover RoPE's structure ... SO(2)^{d_k/2} subgroup embedding"; added a caveat that the frequency ladder theta_n = 10000^{-2n/d_k} is an architectural choice "accommodated by, but not fixed by, the variational principle."
- M10 (PARTIALLY): (b) ATT ~1199 already keeps the shared-per-token-factor constraint (batch D3). (a) SUPP GL(K)-covering claim strengthened with the per-edge-existence vs simultaneous-graphwise-representability caveat + the cocycle identity Omega_ij Omega_jk = Omega_ik.

## Minor comments
- min1: "Fischer curvature" -> "Fisher curvature"; removed informal "and much, much more."
- min2: "allows the FEP to be satisfied" -> stated the actual variational condition (stationarity / joint minimization of the coupled free-energy functional under the fast-belief, frozen-prior scheme).
- min3: "Standard attention transformers are a 0-dimensional gauge theory" -> fenced "... can be represented, under this reconstruction, as the 0-dimensional flat-gauge limit ...".
- min4: "noumenal" base manifold -> "latent base manifold".
- min5 (ALREADY_ADDRESSED): the state-dependent precision section already frames R(alpha) as an optional "augmented free energy" with the R=0/alpha=1 ablation; separate from the KL-softmax derivation. No edit.
- min6: subsumed by M6 (conservative D/D#/S/I grading); no standalone edit.

## Verification
Braces balance (att 4181/4181, begin/end 192/192; supp 2475/2475, 116/116); zero banned spacing macros; old overclaim strings gone. Standalone compile still blocked by the pre-existing missing jmlr2e.sty. Vault synced as CRLF (content-identical to this branch modulo line endings).
