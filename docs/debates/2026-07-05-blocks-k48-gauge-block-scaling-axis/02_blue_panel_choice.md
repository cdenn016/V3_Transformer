# Blue Panel Choice — blocks-k48-gauge-block-scaling-axis (Phase 2, opening)

Side: BLUE (defend by steelmanning). Exactly 5 experts; `philosophy-of-science` mandatory.

Selected to maximize the strongest honest defense of the compound claim (genuine AND publishable AND parameter-efficient AND distinct scaling axis, NOT artifact+confound):

1. **ml-engineer** — Owns the scaling-law frame (Kaplan 2020, Hoffmann 2022) and the sparse/conditional-computation total-vs-active distinction (Shazeer 2017, Fedus 2021). Defends the monotonic, seed-robust improvement on half the tokens as an efficiency signal and the matched-active-compute / matched-wall-time frontier comparison.
2. **transformer-ml** — Defends `phi_embed (V, n_gen)` as the standard GPT embedding-table pattern (one active row per token = low per-token working set at high total capacity) and shows the head mixer shrinks (16x16 -> 2x2) as blocks enlarge, so the gain is not head-mixer capacity.
3. **implementation-engineer** — Establishes the code truth (weighted 3x by the code-truth judge): that the active working set genuinely stays ~12.06M at runtime (`run_artifacts.py:616`, `prior_bank.py:682`), that `phi_embed` is a token-indexed lookup, and that the toggles do not smuggle in the improvement.
4. **gauge-theorist** — Defends that the added capacity is *structured* gauge capacity (n_gen = 48*g, richer within-block connection), a principled axis orthogonal to width, rebutting the "just raw capacity in any V x m table" attack. Honest about the head-mixer/linear-decode equivariance breakage.
5. **philosophy-of-science** (mandatory) — Polices the frame: is "new complementary axis" a Lakatosian progressive problemshift (novel, previously-unpublished direction) or an artifact frame? Enforces falsifiability and guards against manuscript-as-authority circularity — blue's signature failure mode.

Discounted: geometer, info-geometer, variational, numerical-analyst, code-quality — weaker defenders for a claim that turns on scaling-law framing, parameter accounting, and runtime code truth rather than SPD/retraction/ELBO internals.
