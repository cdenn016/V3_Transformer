# Red opening — F9 (mojibake and refs)

## Steelman of the manuscript

The vault `GL(K)_attention.tex` is a clean UTF-8 source whose only non-ASCII bytes in
typeset text are legitimate, intentional glyphs (em/en dashes, curly quotes, "Rényi", the
"§" section glyph); its two supplement-targeted cross-references and its back-references to
the `sec:glk_invariance` anchor are deliberate authorial pointers to where each idea is
first established, not broken plumbing — so a "fix" risks degrading correct text and
mislabeling correct pointers.

## Where the critique stands after primary-source verification

The critique has three sub-claims. After reading the actual vault file, (a) is false,
(b) is true, and (c) is at best a defensible no-op. I attack (a) and (c) and concede (b)
because honest concession is worth more than a manufactured defense.

### (a) Mojibake — FALSIFIED. There is no mojibake; this is a viewer artifact.

The file decodes as clean UTF-8 with no decode error (verified:
`open(...,'rb').read().decode('utf-8')` succeeds; the byte stream raised
`UnicodeDecodeError` on *zero* bytes). The complete non-ASCII codepoint inventory is 11
distinct characters, all standard Unicode:

```
U+2014 EM DASH      x17     U+2019 RIGHT SINGLE QUOTE/APOSTROPHE  x3
U+2013 EN DASH      x5      U+00B0 DEGREE SIGN                    x2
U+03A9 GREEK OMEGA  x5      U+00A7 SECTION SIGN                   x2
U+2229 INTERSECTION x4      U+00E9 e ACUTE (Rényi)                x2
U+2192 RIGHTWARDS ARROW x1  U+201C/U+201D LEFT/RIGHT DOUBLE QUOTE x1 each
```

Of these, every Omega/intersection/arrow/degree codepoint sits inside a `%` TikZ comment
line (verified by line scan: L136, L155, L197, L227, L230, L231, L242, L249, L277, L279,
L284, L315 — all `comment=True`). LaTeX discards everything after `%` before it reaches the
typesetter, so those bytes can never produce output, mojibake or otherwise. The only
non-comment specials are the section sign and the e-acute:

- `GL(K)_attention.tex:2047` — "...the Kullback-Leibler divergence, or its **Rényi**-$\tfrac{1}{2}$
  generalization..." The é is the correct spelling of a proper noun.
- `GL(K)_attention.tex:2366` — "...stated in the abstract and **§**\ref{sec:glk_lm}." The §
  is the intended section glyph.

The lines the critique flags as mojibake (66 en-dash, 427 em-dash+apostrophe, 2047 e-acute,
2366 section-sign) are legitimate UTF-8 characters that render as garbage *only when a UTF-8
file is opened under a cp1252 viewer*. I reproduced exactly this failure mode mid-audit: a
Windows cp1252 console raised `UnicodeEncodeError: 'charmap' codec can't encode character
'Ω'` while *printing* the (correctly decoded) content — proving the corruption lives in
the viewer's egress encoding, not in the file. The preamble is `jmlr2e` with no `inputenc`
(verified: `inputenc present?: False`), relying on the pdfLaTeX/LaTeX UTF-8 default
(`\input{utf8.def}` since the 2018 format), under which all 11 codepoints typeset correctly.
**No mojibake exists to fix.** The most that is warranted is optional portability hardening
(macro-ize prose Unicode or add `inputenc`), which is a preference, not a defect.

### (b) Dangling refs — CONCEDED (see rebuttal section is N/A; this is the opening).

Two `\ref` targets in `attention.tex` are undefined locally and defined only in
`GL(K)_supplementary.tex`, with no `xr` wiring (verified: of 216 ref calls,
`eq:beta_grad_phi` and `sec:supp_decode_derivation` resolve to SUPPLEMENT;
`externaldocument present?: False`, `usepackage{xr} present?: False`). Both render `??` in a
standalone compile of `attention.tex`. This is a real defect. The correct fix is the one the
manuscript's own convention already prescribes — the document otherwise cites the supplement
in **plain text** ("Supplementary Appendix E"), so converting these two `\ref`s to the same
plain-text form is consistent and minimal; `xr` wiring is the heavier alternative.

### (c) Repointing `sec:glk_invariance` refs — the proposed edit is wrong / narrower than claimed.

The critique asserts several `sec:glk_invariance` refs "actually mean the transformer-limit
section." Read in context, they mean exactly what they point to. `sec:glk_invariance`
(`GL(K)_attention.tex:517-520`) is the section titled "GL(K) Gauge Invariance of KL
Divergence," and its own prose at L514 states the algebraic structure "already suffices to
recover transformer attention (as shown below)" — i.e., this section is the *anchor* of the
recovery, with the mechanics deferred downstream. Each contested back-reference is phrased as
a **recall of the licensing principle**, not a pointer to the derivation steps:

- `:1762` — "**Recall** from Section~\ref{sec:glk_invariance} that, in the isotropic
  flat-bundle limit, standard transformer attention can be interpreted as GL(d_k)
  gauge-theoretic attention..." This recalls the invariance result that makes the GL(d_k)
  interpretation legitimate before block-diagonalizing it.
- `:1882` — "**Recall** from Section~\ref{sec:glk_invariance} that
  $W_Q W_K^\top = \sigma^{-2}\Omega^{-\top}$." This identity is *previewed* under the
  invariance anchor; repointing it to `sec:dot_product_derivation` (L1192) would point a
  "recall" at a section the reader may not have read as primary.
- `:2289` — "the learned projection $W_Q W_K^\top = \sigma^{-2}\Omega^{-\top}$
  (Section~\ref{sec:glk_invariance}) is position-independent and therefore trivially flat."
  An attribution to the invariance origin, not a derivation cross-link.

Repointing these to `sec:transformer_limit` (L1051) or `sec:dot_product_derivation` (L1192)
would substitute "where it is derived" for "where it is established as invariant," changing
the rhetorical referent the author chose. The manuscript already cross-links the derivation
sections where it means them (e.g. L859 "the standard-transformer recovery in
Section~\ref{sec:glk_invariance}"; L1189 "the analysis of Section~\ref{sec:dot_product_derivation}
below applies"), demonstrating the author distinguishes the two and uses each deliberately.
This sub-claim is a stylistic preference dressed as an error; under the load-bearing test it
does not collapse the manuscript.

## Load-bearing assumption the critique depends on

The critique stands or falls on the assumption that **what the auditor's viewer displayed is
what the file contains.** That assumption is false for (a): the bytes are clean UTF-8 and the
"mojibake" is produced by the cp1252 rendering path, not by the file. Strip that assumption
and two of three sub-claims evaporate.

## Strongest objection in falsifiable form

If `attention.tex` is compiled with current pdfLaTeX (UTF-8 default, no `inputenc`) and the
PDF is inspected, then (a) every one of the 11 non-ASCII codepoints typesets correctly with
zero mojibake glyphs — the claim of "visible mojibake to fix" fails. The claim that survives
is narrower than stated: exactly **two** dangling supplement refs (b), fixable by the
manuscript's existing plain-text convention. The mojibake claim (a) is a no-op, and the
repoint claim (c) is a contestable style edit, not a defect.

## Citations

- Primary (vault): `GL(K)_attention.tex:4` (`\usepackage[preprint]{jmlr2e}`, no `inputenc`);
  `:514`, `:517-520` (sec:glk_invariance anchor + "suffices to recover... as shown below");
  `:859`, `:1189` (author distinguishes invariance anchor from derivation sections);
  `:1051`, `:1192` (derivation section labels); `:1762`, `:1882`, `:2289` ("Recall"
  back-references); `:2047` (Rényi é), `:2366` (§). Executed verification: clean UTF-8
  decode; 11-codepoint inventory; all Omega/intersect/arrow/degree in `%` comments; 216 ref
  calls with exactly two (`eq:beta_grad_phi`, `sec:supp_decode_derivation`) supplement-only
  and no `xr`.
- External canon: under the LaTeX2e UTF-8 default (`utf8` input encoding active by default
  since the 2018 LaTeX release), straight UTF-8 source compiles without `inputenc`; the
  flagged glyphs are valid input, not corruption. The KL invariance the anchor section
  states is the standard information-geometry result that f-divergences are invariant under a
  common invertible reparameterization (Amari 2016, *Information Geometry and Its
  Applications*, Ch. on invariant divergences) — which is why `sec:glk_invariance` is the
  correct conceptual referent for the GL(K) identification, supporting the author's choice in
  (c).
