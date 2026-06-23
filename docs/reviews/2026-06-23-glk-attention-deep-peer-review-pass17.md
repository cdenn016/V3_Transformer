# GL(K) Attention Manuscripts - Deep Peer Review Pass 17

Date: 2026-06-23

Files reviewed: `Manuscripts-Theory/GL(K)_attention.tex`, `Manuscripts-Theory/GL(K)_supplementary.tex`, and `Manuscripts-Theory/references.bib`.

Scope: this pass deliberately ignores live empirical metric values, including PPL, loss, CE, and any table entries that are being regenerated. The review focuses on non-empirical manuscript risk: internal consistency, source-submission hygiene, mathematical notation, and claim calibration.

Recommendation: revise before any public source release or submission. The cross-reference wiring is in good condition, but several prose and theorem-framing issues still expose avoidable reviewer targets.

## Machine-Checked Structure

A regex-based check over the two GL(K) manuscript files and the manuscript bibliography found 205 labels, 0 duplicate labels, 325 references with 0 missing targets, 168 citation keys with 0 missing bibliography entries, and 378 BibTeX keys. This is a real strength of the current draft. The structural LaTeX wiring appears clean.

## Findings

### 1. Source placeholders and unresolved review comments remain in the manuscript files.

Both manuscript roots still contain `\editor{TBD}` at `GL(K)_attention.tex:39` and `GL(K)_supplementary.tex:39`. The source also contains embedded review comments at `GL(K)_attention.tex:2248`, `GL(K)_attention.tex:2255`, `GL(K)_attention.tex:2379`, and `GL(K)_supplementary.tex:1023`.

Because the experiments are currently being rerun, I am not treating the numeric content of those comments as a live finding. The issue is narrower: if the TeX source is uploaded to arXiv, shared with reviewers, or committed as a submission artifact, these comments advertise unresolved internal audit notes. Move the data-pending notes into a private issue tracker or delete them after the regenerated tables are installed. Fill or remove the editor placeholder before any source distribution.

### 2. The supplement overstates the temperature-dispersion interpretation.

The main manuscript has been moving toward a cautious framing of the BERT-family validation as an exploratory flat-bundle comparison. The supplement is less cautious. At `GL(K)_supplementary.tex:854`, RoBERTa's lower correlation is said to be attributable to per-head temperature dispersion as the dominant explanatory variable. At `GL(K)_supplementary.tex:878`, the table is said to reveal that same dominant explanatory variable.

That wording is too strong for a five-architecture comparison, regardless of the final regenerated values. It reads as causal or variable-importance language, while the design supports at most an exploratory association unless the manuscript adds a proper model-comparison or ablation analysis. Replace "dominant explanatory variable" with language such as "suggestively associated with the cross-model variation" or "a plausible contributor to the observed variation."

### 3. The collapse-prevention story gives the entropy term two incompatible roles.

At `GL(K)_attention.tex:931`, the manuscript says the attention-entropy regularizer is introduced to forestall single-cluster collapse. The later limitations section is sharper: at `GL(K)_attention.tex:2376`, the draft says the entropy term prevents one-hot assignment but can still produce token averaging, and that the real anti-collapse mechanisms should be the prior-anchoring identity path and token-dependent transport.

The later version is the more defensible one. Entropy regularization controls assignment sharpness; it does not by itself inject rank or preserve token distinctions under deep mixing. The earlier paragraph should be revised so it does not credit the entropy term as the primary brake against consensus collapse. A safer formulation is that entropy smooths the mixing distribution, while alpha-prior anchoring and token-dependent transport are the candidate rank-preserving mechanisms.

### 4. Appendix H blurs pushforward notation with pointwise density multiplication.

The main text defines transport as a pushforward, for example `GL(K)_attention.tex:530`, `GL(K)_attention.tex:539`, and the local gauge-invariance proof at `GL(K)_attention.tex:564` to `GL(K)_attention.tex:573`. Appendix H then writes local density expressions such as `\Omega_{ij}(c) q_j(c)` and `\Omega_{ij}q_j`, including the uniqueness statement at `GL(K)_supplementary.tex:1111`, the functional derivative calculation at `GL(K)_supplementary.tex:1236`, and the richness step at `GL(K)_supplementary.tex:1289`.

This is not just typographic. A linear map acting on a density is a pushforward with the appropriate Jacobian, not pointwise multiplication by a matrix-valued function. If Appendix H is intended only as shorthand, state that explicitly before the theorem and write `(\Omega_{ij})_* q_j` in the theorem statement. If the proof is actually using a common reference measure after transport, define the transported density with respect to that measure and carry the Jacobian convention once. This will prevent a reviewer from reading the theorem as dimensionally or measure-theoretically ill-posed.

### 5. The supplement still overstates bundle nontriviality in the edge-relaxed extension.

At `GL(K)_supplementary.tex:56`, the supplement says the edge-relaxed extension promotes the bundle to a non-trivial principal `G`-bundle. That remains too strong unless the manuscript proves a topological obstruction or gives a nontrivial transition-cocycle class. The construction described there more directly gives a non-flat discrete connection or nontrivial holonomy observable on the graph.

Revise this sentence to say that the edge-relaxed extension permits non-flat connection data and loop holonomy. Reserve "non-trivial principal bundle" for a setting where the base, transition functions, and obstruction class are all specified.

### 6. "Epistemically dead" is vivid but reviewer-hostile terminology.

The supplementary appendix uses "epistemically dead" at `GL(K)_supplementary.tex:97` and `GL(K)_supplementary.tex:109` for agents that share beliefs and models after gauge alignment. The phrase is memorable, but it is nonstandard and sounds rhetorical rather than mathematical.

Replace it with neutral terminology. Suitable alternatives include "gauge-synchronized," "informationally redundant under the coarse-graining relation," or "dynamically quiescent in the absence of observations." The concept is useful; the name is the problem.

### 7. The project-style cleanup needs a prose-only pass.

The manuscript source still contains TeX triple-hyphen em-dash syntax in narrative prose, while project instructions ban that pattern in manuscripts. Some occurrences are TeX or TikZ syntax and must be left alone, but prose examples occur around `GL(K)_attention.tex:68`, `GL(K)_attention.tex:560`, `GL(K)_attention.tex:1199`, `GL(K)_attention.tex:1760`, `GL(K)_attention.tex:2302`, `GL(K)_attention.tex:2378`, and nearby lines. The line `GL(K)_attention.tex:2405` also uses "leveraging," which is close to the banned project diction.

Run a prose-only style sweep after content stabilization. Replace TeX em-dashes in sentences with commas, parentheses, or semicolons. Do not do a blind regex replacement, because TikZ paths and LaTeX syntax use hyphen sequences legitimately.

### 8. The abstract and introduction still compress several constructions into "one variational principle."

The abstract and opening framing, including `GL(K)_attention.tex:45`, `GL(K)_attention.tex:58`, and `GL(K)_attention.tex:62`, still present the full transformer correspondence as emerging from a single variational principle. The body has become more careful: it uses a canonical free energy, a reduced softmax objective, engineered consensus terms, flat-bundle comparisons, and correspondence claims of different strengths.

The safest revision is to keep the unified variational framing but avoid implying that every transformer component is derived with equal mathematical force. Say that the paper gives a variational reconstruction that recovers attention and temperature scaling exactly under stated limits, while other architectural correspondences are graded analogies or design-compatible extensions.

## Punch List

1. Fill or remove `\editor{TBD}` in both manuscript files before source release.
2. Remove or relocate the embedded data-pending review comments from the TeX source after the regenerated empirical tables are installed.
3. In the supplement, downgrade "dominant explanatory variable" for temperature dispersion to exploratory association language.
4. Align the collapse discussion so entropy regularization controls assignment sharpness, while alpha-prior anchoring and token-dependent transport carry the anti-collapse burden.
5. In Appendix H, replace ambiguous `\Omega q` density notation with pushforward notation or define the transported density convention explicitly.
6. Replace "non-trivial principal bundle" for the edge-relaxed graph construction with "non-flat discrete connection" or "nontrivial loop holonomy," unless a topological bundle class is actually proved.
7. Replace "epistemically dead" with neutral technical terminology.
8. Run a prose-only cleanup for TeX em-dash syntax and nearby banned diction, leaving TikZ and LaTeX syntax intact.
9. Soften the abstract/introduction's "single variational principle" language into a graded reconstruction claim.
