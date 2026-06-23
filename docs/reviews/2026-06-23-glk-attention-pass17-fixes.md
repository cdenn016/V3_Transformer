# GL(K) Attention — Pass-17 Deep Peer Review: Fixes Applied

Date: 2026-06-23
Base: `origin/main` @ `eee9dae` (B/C/D/E/F + pass-15 + pass-16). Branch: `manuscript/glk-peer-review-pass17`.
Verification: investigator + adversarial-skeptic workflow for the content edits and the em-dash sweep, each re-reading the live file. Items selected by the user: punch-list #1/#2, #3, #4, #5, #6, #8, #9 (skipping #7, "epistemically dead" kept by user request).

## Per-item verdicts

| Punch # | Verdict | Action |
|---------|---------|--------|
| #4 collapse story | fixed | The ~931 sentence over-credited the attention-entropy term as the brake against single-cluster collapse. Rewritten so entropy "shapes the mixing distribution without itself arresting the collapse" while the **prior-anchoring identity path** `α_i D_KL(q_i‖p_i)` and **token-dependent transport** `Ω_ij` are named the candidate rank-preserving counterforces — consistent with the already-correct passages at lines 1700/2412. |
| #5 App-H pushforward | fixed | Added one convention sentence before the App-H theorem: `Ω_ij q_j` denotes the pushforward density `[(Ω_ij)_* q_j]` (Jacobian included, as defined in App-B), so `Ω_ij(c) q_j(c)` is shorthand for that density at `c`, not matrix×scalar. Resolves all App-H instances at once. |
| #6 non-trivial bundle | already resolved | SUPP:56 already reads "equips the bundle with a non-flat discrete connection… Wilson observable on closed loops is non-trivial" (pass-15 M2). No edit. |
| #8 em-dash + diction | fixed | Prose-only sweep: **19** narrative/caption `---` em-dashes converted to commas/parentheses (incl. the two introduced by pass-16 M7 at line 2441). Left untouched: TikZ `%`-comment dividers, algorithm-block `\STATE \textit{---}` markers, table missing-data cells, and `\eqref{..}--\eqref{..}` en-dash ranges. ("leveraging" at the start of the subgroup sentence is a separate clause outside the touched em-dash spans; left as-is.) |
| #1/#2 source hygiene | fixed | Removed the inert `% \editor{TBD}` placeholders (ATT, SUPP). **Relocated** (not deleted) the 5 DATA-PENDING `% TODO` comments to `docs/reviews/2026-06-23-data-pending-tracker.md` and removed them from the `.tex` sources — they flag real unresolved data-reconciliation tasks for experiments still being rerun, so the content is preserved for closure before release. |
| #3 temperature dispersion | already resolved | SUPP:854/874/878 already use "exploratory association," "suggestive," and literally "rather than a dominant explanatory variable" (batch F6). No edit. |
| #9 single principle | already resolved | Abstract/intro lines 45/47/58/62 already grade attention + temperature as exact-under-limits and layer norm / training dynamics as structural accounts (pass-15 M1). No edit. |
| #7 epistemically dead | skipped | Per user: kept as-is. |

## Integrity
ATT braces 4201/4201, begin/end 192/192; SUPP braces 2486/2486, begin/end 116/116. 0 banned spacing macros, 0 banned words. Remaining `---` in ATT are exactly the 10 TikZ/algorithm/table LEAVE lines.

## Notes
- 21 manuscript edits total (C4 + C5 + 19 em-dash) plus 7 comment-line removals; tracker doc added.
- Branch off `origin/main` (pass-16 merged via PR #131); the user's `feat/multiseed-digest` WIP was not touched.
