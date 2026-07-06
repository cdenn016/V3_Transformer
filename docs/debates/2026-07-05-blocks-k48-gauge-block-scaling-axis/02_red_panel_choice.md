# Red Panel Choice — Phase 2 (opening)

Debate: blocks-k48-gauge-block-scaling-axis | Mode: implementation | Side: RED

The claim's load-bearing pillars are a code-defined proxy (`active_params_per_token`),
a scaling-law framing across a 2× token confound, a "distinct scaling axis" assertion,
and a causal attribution to "gauge structure." The panel is chosen to attack exactly those.

| Expert | One-line rationale |
|---|---|
| `philosophy-of-science` (MANDATORY) | Frame-checks the compound conjunction ("genuine AND publishable AND parameter-efficient AND distinct axis") and the missing non-gauge control — causal attribution to "gauge structure" is unestablished without a matched-size control. |
| `ml-engineer` | Owns the scaling-law attack: the exact-2× token confound under Chinchilla `L(N,D)`, Kaplan data-limited slices, and the axis-dependent fitted exponent (0.18 vs n_gen, 0.93 vs n_params) failing scaling-law well-posedness. |
| `implementation-engineer` | Establishes code-truth: `active = 5·V·K + 2·K + n_gen` (run_artifacts.py:616) is a closed-form proxy whose only sweep-varying term is `+n_gen (~1008)`, and the "extra" 50M params are a token-indexed `phi_embed (V, n_gen)` lookup (prior_bank.py:167,682) — the "flat working set" is definitional. |
| `transformer-ml` | Targets the 3-way co-variation, especially `n_heads` 16→2, and applies the standard MoE total-vs-active distinction (Shazeer 2017 / Fedus 2021): "active params" is meaningful only paired with a compute axis. |
| `numerical-analyst` | Real compute is NOT flat: transport FLOPs ∝ g² (64×), U-shaped wall-time; and the axis-dependent exponent is a conditioning artifact of fitting a power law over a compressed 3.62× x-range with CI [0.07, 1.73]. |

Discounted: `gauge-theorist`/`geometer` (would risk conceding the gauge structure is genuinely
expressive, aiding blue); `code-quality` (this is not a design-smell dispute); `info-geometer`
and `variational` (the divergence/ELBO internals are not contested here).
