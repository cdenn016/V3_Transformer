# GL(K) attention manuscript — deep review pass 13 (whole-paper referee: reporting, figures, readiness)

Date: 2026-06-20. Targets: the canonical vault `GL(K)_attention.tex` and `GL(K)_supplementary.tex`.

Pass 13 is the synthesis pass. After twelve passes exhausted the analytic mathematics, citations, framing, notation, code-fidelity, and cross-document consistency, this pass reads the paper as a journal referee would for an acceptance decision, along three lenses that the micro-error passes did not cover: empirical-reporting completeness (does the paper give a reader enough to trust and reproduce the results, independent of the data-pending values?), figures and captions, and whole-paper coherence and submission-readiness.

Method: three referee lenses dispatched in parallel — an ML-engineer reporting-completeness referee, a figures/captions reviewer, and a philosophy-of-science holistic referee — each seeded with the verified-ledger do-not-reflag list and instructed not to re-verify the known data-pending values.

Headline: the paper is coherent and substantially submission-ready, with rare candor about its own limitations, and the recommendation is minor revision. Pass 13 found four surgically-fixable issues (applied) and a cluster of reporting/data items that need the author (data or a submission decision), the most important of which is that the data-pending K-sweep perplexity has propagated into the abstract. No analytic error; everything here is about how the empirical story is reported, not whether the math is right.

## Recommendation

Minor revision. Four fixes applied this pass. Before submission the author should additionally (a) reconcile the abstract's `64.9` K-sweep perplexity against the released CSV (the standing data-pending item, now also in the abstract), and (b) add an in-paper hyperparameter/optimizer table for the language-model runs. Neither requires an equation change or a new result; both need information the reviewer does not have (the finalized CSV and the run configs).

## Findings applied this pass

### F1 — Ablation causal claim asserted what its own TODO says is untested (attn:2236, HIGH)

The Ablation paragraph read "removing the MLP … is not sufficient to match the gauge VFE's performance — the VFE's advantage stems from its geometric structure, not merely from operating without an MLP," while the in-source TODO at 2237 states that the controlled gauge on/off ablation (`Ω=I` vs gauge-on vs frozen-random) which would establish this is commented out of the sweep and unrun. The attention-only baseline (142.8) rules out "no-MLP alone" but does not isolate gauge transport from the diagonal-covariance beliefs, the KL attention, or the parameter overhead. This is the same affirming-the-consequent over-reach that pass 9 corrected in the parallel Results sentence (2242, harmonized to "in part because … consistent with"); pass 9 did not catch the Ablation twin.

Applied: kept the supported clause (no-MLP ruled out) and hedged the attribution — "Whether it stems specifically from the gauge-transport geometry, rather than from the diagonal-covariance beliefs or the KL-divergence attention, awaits the controlled gauge on/off ablation, which is not yet run." Consistent with the pass-9 decision and the author's own TODO.

### F2 — Abstract stated single-seed headline perplexities without the caveat the body carries (attn:49, MEDIUM)

The abstract reported the GL(15) Japanese (24.2) and WikiText-103 (71.6) headline perplexities as flat numbers, while the body's seed-disclosure paragraph (2370) states both are single-seed (only the non-headline GL(10) row is n=2). An abstract-only reader would take them as robust estimates.

Applied: added "(single seed)" to each of the two headline perplexities in the abstract, matching the body's own disclosure. No number changed.

### F3 — Flagship bundle figure was never referenced (attn, figures, LOW-MEDIUM)

`fig:bundle_sections_surface` — the paper's only conceptual schematic of the entire gauge-transport / fiber-bundle construction (inline TikZ, ~attn 74-359) — carried a `\label` but was never `\ref`'d anywhere in the body, so the reader is never directed to it.

Applied: added "(Figure~\ref{fig:bundle_sections_surface})" at the body sentence that introduces the local-section / associated-bundle construction (attn:366). The other unreferenced floats (four supplementary empirical figures and the supplement's `fig:hierarchy`) are noted below as remaining low items.

## Items reported for the author (not applied — need data, a submission decision, or are data-pending)

- **Abstract leads with `64.9` (K-sweep), which the cited CSV does not contain (HIGH, data-pending).** The abstract and body sweep (194.5→64.9) report a K=120 perplexity of 64.9; the in-source DATA-PENDING TODO (2230) records that the released `aggregated_K_sweep.csv` gives 72.71 at K=120 and 222.70 at K=10, an unreconciled best-validation-vs-final-checkpoint selection mismatch. This is the standing ledger §4 item; pass 13's addition is that the unreconciled number now leads the abstract. The monotonicity and the sub-KN-5 conclusion survive either number set, so the scientific claim is unaffected — but the number's provenance must be reconciled (or the checkpoint policy stated) before submission. Not edited here: changing or hedging a headline result number is the author's call pending the CSV.
- **No in-paper optimizer/hyperparameter table for the LM runs (HIGH, needs configs).** Adam step size, `(β1,β2,ε)`, schedule/warmup, batch size, gradient clip, κ/τ, trust-region radius, and the per-component learning rates are not stated in the paper (only "archived with the code repository"); Algorithm 1 says "Adam" with no values. A referee cannot reproduce the runs from this. The fix is a methods hyperparameter table, which requires the run configs (in the `epistemic-geometry` repo), so it cannot be supplied from the manuscript.
- **Single-seed headline ratios carry no dispersion (MEDIUM).** The 1.66×/1.88×/1.94-2.04× advantages are built on single-seed numerators; the n=2 GL(10) std (1.05, ~1.4% CV) is available and could be propagated as a sensitivity band. Author framing call.
- **n=5 cross-model "strongest predictor" claim (MEDIUM).** The temperature-dispersion-vs-correlation claim rests on a correlation over five architectures (r=−0.87, ρ=−0.6) testing four hypotheses, with no power caveat; pass 8 already softened the adjacent wording, so this is left as a recommended further softening rather than re-litigated.
- **K-sweep checkpoint policy unstated (MEDIUM, data-pending).** Same root as the 64.9 item; state the selection policy and per-K dispersion when the CSV is finalized.
- **Remaining unreferenced floats (LOW).** Four supplementary empirical figures (`fig:temperature_sweep_supp`, `fig:correlation_distribution_supp`, `fig:key_norm_bias_supp`, `fig:attention_entropy_supp`) and the supplement's `fig:hierarchy` are defined but not `\ref`'d in their own document; add textual call-outs at the prose that already discusses each.
- **Bundle diagram depicts transport as a shape-preserving rotation (LOW).** The ghost ellipse `Ω_ij q_j` has the same semi-axes as `q_j` (a pure rotation), and the caption says "rotating," in mild tension with the body's "GL(K), not orthogonality" statement and the geometric-bias term — but the body itself uses "frame rotation" language (`eq:gauge_frame_rotation`), so changing only the figure would create a figure-vs-body mismatch. Left for the author to decide whether to depict the shear and reword both.
- **Abstract emphasis leads with "outperforming" (LOW).** Against the paper's settled explanatory-not-competitive framing; the parameter-matched counterweight (48.5) is already in the same sentence, so this is optional emphasis polish.

## What reads well (referee positives, do not re-flag)

The paper's candor surface is unusually strong: a dedicated seed-disclosure paragraph, a four-point limitations section (single-layer/no-depth, 17.7× parameter overhead, underperformance vs the parameter-matched transformer stated plainly, diagonal-RoPE covariance break, the "no neural networks" qualifications), in-source DATA-PENDING TODOs, a Bayesian-convergence caveat on the one weak chain, and the explicit "this is an explanatory theory, not state-of-the-art" framing. The abstract-to-body contribution mapping is otherwise faithful; the D/S/I claim-status taxonomy is defined and applied consistently; the RG section is correctly demoted to a textbook-CLT Proposition plus an open Conjecture whose own falsifying measurement (y2≈−0.6, y3≈+0.2) is reported rather than explained away; the holonomy-compositionality thread is an open question with its prerequisite named and the homomorphism-to-gauge mapping conceded as an underived novel construction; and the BERT correspondence is correctly classified as a consistency check (algebraic identity under LayerNorm) rather than independent confirmation. The empirical-reporting referee separately confirmed the BERT validation's n / SE / 95% CI / Bonferroni reporting, the compute-environment disclosure, the fair embedding-/parameter-matched baseline design, and the honest admission that the parameter-matched transformer wins.

## Method

Three referee-lens agents in parallel (reporting-completeness, figures/captions, holistic), each given the vault manuscripts and the verified-ledger do-not-reflag list, instructed to assess whole-paper reporting/figures/coherence and not to re-verify the data-pending values. The four applied items were re-confirmed against the live text before editing; the remaining items are reported for the author because they require data (the finalized CSV, the run configs) or a submission-level decision rather than a manuscript edit. No high/critical analytic finding was produced.
