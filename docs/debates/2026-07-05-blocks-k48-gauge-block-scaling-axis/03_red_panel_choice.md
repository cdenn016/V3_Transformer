# Red Panel Choice — Phase 3 (rebuttal)

Mode: implementation. Round: rebuttal. Blue's opening conceded the "publishable exponent" and
"parameter-efficient-as-dominance" conjuncts and defends only three propositions: (i) the real
monotone within-sweep CE effect at a single 245.76M-token budget, (ii) `phi_embed` as a legitimate
token-indexed embedding-table access pattern, (iii) the gauge-block axis as structurally new
(`n_gen = 48·g`). The rebuttal must press that "structurally new axis" is not "efficient scaling
axis," that blue's own concessions gut three of the four conjuncts so the FULL claim as stated
fails (a REMAND, not a BLUE_WINS), and that the missing non-gauge matched-size control leaves
"gauge structure did it" unidentified.

Panel (exactly 5; philosophy-of-science mandatory):

1. **philosophy-of-science** (mandatory) — frame-check that "structurally new" ≠ "efficient scaling
   axis"; underdetermination of a 3-knob co-varying manipulation (n_params, n_heads, g); the
   confound-vs-mechanism distinction; whether the surviving sub-claim is a REMAND rather than a win.
2. **gauge-theorist** — the identification attack: `n_gen = 48·g` (untied) adds a `V × n_gen` table
   whose gain is not separated from any matched-size non-gauge table; `tied_block_glk` gives `g²`;
   and blue's own equivariance canon (Cohen & Welling) raises accuracy at *fixed* parameter count,
   which this 3.62×-param sweep does not do — the analogy cuts against blue.
3. **ml-engineer** — the Chinchilla `L(N,D)` 2× token-budget confound on any cross-sweep
   "complementary to width" comparison; scaling-law framing that a distinct efficient *axis* needs a
   matched-budget, matched-control demonstration that does not yet exist.
4. **implementation-engineer** — the code-truth defense blue conceded: the `active/tok` algebraic
   identity (`run_artifacts.py:616`, `5·V·K + 2·K + n_gen`), the `g²` transport growth (64×), and
   the U-shaped empirical wall-time; confirm these stand.
5. **transformer-ml** — the head-count/head-width confound (`n_heads = 48/g` co-varies), and the
   turn that blue's embedding-table analogy is itself the confound: a plain `V × m` table reads one
   row per token too, so the access pattern does not identify gauge structure as the cause.

Swap vs Phase 2 red panel: numerical-analyst (power-law fit-conditioning, largely mooted by blue's
exponent concession) → gauge-theorist (the now-central mechanism-identification battleground).
