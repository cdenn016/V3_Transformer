# VERDICT — F9 (mojibake-and-refs)

## Outcome

**PARTIAL (narrowed edit).** The critique bundles three sub-claims; only sub-claim (b)
survives as an actionable defect. (a) is falsified on the file; (c) is a non-defect
stylistic preference that both teams ultimately decline. Editor action: fix the two
dangling cross-document refs; leave the file otherwise unchanged.

This maps to the red/blue frame as **mostly RED** on the bundle as stated (the headline
"mojibake" claim and the repointing recommendation both fail), with **BLUE's narrowed
core (b) upheld** — which is exactly the convergent position both openings reached.

## Decisive evidence

- **(b) UPHELD — verified.** `\ref{eq:beta_grad_phi}` (used `attention.tex:1993`,
  `(Eq.~\ref{eq:beta_grad_phi})`) and `\ref{sec:supp_decode_derivation}` (used
  `attention.tex:2071`, `(Section~\ref{sec:supp_decode_derivation})`) have NO matching
  `\label` in `attention.tex` (grep: "No matches found") and ARE defined in
  `GL(K)_supplementary.tex` at lines 438 and 1335 respectively. No
  `\usepackage{xr}` / `\externaldocument` exists anywhere in `attention.tex` (grep:
  "No matches found"; preamble lines 1-15 confirm jmlr2e with no xr). Standard LaTeX2e
  behavior: a `\ref` to a label outside the compilation unit and without `xr` wiring
  resolves to `??` (LaTeX2e `xr` package documentation; Mittelbach & Fischer, *The LaTeX
  Companion* 2e). Both refs therefore ship `??` into the PDF. The document already uses a
  plain-text "Supplementary Appendix~A/E" convention for the separately-compiled
  supplement (lines 400, 2137, 2264, 2434), which both fixes the bug and matches house
  style.

- **(a) FALSIFIED.** Preamble is `\usepackage[preprint]{jmlr2e}` with no `inputenc`
  (line 4); pdfLaTeX has defaulted to UTF-8 input since the 2018 format (LaTeX2e News
  Issue 28). Both teams independently confirmed clean UTF-8 decode and that the only
  non-comment specials are legitimate codepoints (é in "Rényi" :2047/2059; § :2366);
  Ω/∩/→/° appear only inside `%` TikZ comments. The "mojibake" is a cp1252-on-UTF-8
  viewer artifact, reproduced by both teams. No fix needed.

- **(c) NON-DEFECT — repointing declined.** The label `sec:glk_invariance` (518) anchors
  a real, on-topic section. Line 514 states the algebraic structure "already suffices to
  recover transformer attention (as shown below)," and the identity
  `W_Q W_K^\top = \sigma^{-2}\Omega^{-\top}` is announced at line 568 — both WITHIN the
  `sec:glk_invariance` section block. The three back-references (1762, 1882, 2289) all
  read "Recall from Section~\ref{sec:glk_invariance} that, in the isotropic flat-bundle
  limit ..." — they recall the licensing principle/announced identity, which genuinely
  live in that section. Repointing to the derivation sections (1051/1192) changes the
  author's chosen referent. Blue classified (c) as optional; red opposed it; neither
  defends it as a defect.

## Reasoning

Weighing by evidence: the file is the artifact under evaluation. On (a) both sides agree
and the file confirms clean UTF-8 — the critique's strongest-sounding claim is the
emptiest, an auditor-viewer artifact, not a file defect. On (b) the evidence is
incontrovertible and conceded by both: two `\ref` targets resolve to `??`. On (c) the
distinction is between "broken" and "could be more precise"; the refs are not broken, the
anchor is on-topic, and changing the referent overrides an authorial choice — the project
CLAUDE.md explicitly forbids re-deliberating the author's intentional choices and
demands surgical, traceable edits. There is no genuine evidentiary split across parts of
the claim that would require REMAND; both sides converged. The correct disposition is a
single narrowed edit (b).

## Action

Make exactly one edit set in
`C:\Users\chris and christine\Desktop\Research\manuscripts\GL(K)_attention.tex`,
following the document's existing plain-text supplement convention:

1. Line 1993 — replace `(Eq.~\ref{eq:beta_grad_phi})` with
   `(Supplementary Appendix~C, Eq.~12)` **if and only if** the editor pins the supplement
   number; the convention-safe replacement that needs no number lookup is
   `(see Supplementary Appendix~C)`. Recommended exact replacement of the parenthetical:
   `(Supplementary Appendix~C)`.
2. Line 2071 — replace `(Section~\ref{sec:supp_decode_derivation})` with
   `(Supplementary Appendix~F)` — matching the plain-text style of lines 2137/2434.
   The surrounding prose already names it "the Supplementary decode derivation," so the
   parenthetical should read: `(Supplementary Appendix~F)`.

Editor note: confirm the supplement's appendix letters before committing (the supplement
uses `\section`-level appendices; verify C = gauge-frame gradients and F = decode
derivation, or substitute the correct letters). The ONLY robust alternative is to add
`\usepackage{xr-hyper}` + `\externaldocument{GL(K)_supplementary}` to the preamble, but
that requires a two-pass build with the supplement's `.aux` present and is heavier than
the house plain-text convention — prefer the plain-text fix.

Do NOT touch any UTF-8 glyph (a) and do NOT repoint the `sec:glk_invariance` refs (c).
