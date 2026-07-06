# Claim — blocks-k48-gauge-block-scaling-axis

**Mode:** implementation
**Panel:** full
**Rounds:** 2
**Judging:** panel
**Experts override:** none
**Evidence scope:** auto
**Canon location:** embedded

## Claim

In `blocks_K48`, enlarging the GL gauge block at fixed `embed_dim=48` is a genuine, publishable parameter-efficient scaling axis for the VFE transformer — it lowers cross-entropy substantially (PPL 124.6 → 92.2 as GL3 → GL24) while the per-token active-parameter working set stays essentially constant (~12.06M), making it a distinct scaling knob complementary to width-scaling (`grow_K_GL10`), rather than a metric artifact of `active_params_per_token` plus a half-token-budget confound.

## User context

Two scaling sweeps in `vfe3_scaling_results/`:
- `grow_K_GL10`: fixes the GL(10) gauge block, grows width/heads K10→K120 (embed_dim = 10·n_heads), 491.52M tokens/run, 12 sizes × 3 seeds.
- `blocks_K48`: fixes `embed_dim=48`, varies the gauge partition GL(3)^16 → GL(24)^2 (n_heads = 48/g, n_gen = 48·g), 245.76M tokens/run, 5 sizes × 3 seeds.

The claim arose after the orchestrator initially presented a "CE vs active-params/token" panel in which `blocks_K48` appears as a near-vertical column (active/tok ~constant while total params grow 3.6×). The user challenged this ("vertical params/token? why? GL24 has MANY more parameters than GL3"), which exposed that `active_params_per_token` is a closed-form proxy dominated by the fixed decode term, and that the extra parameters live in a token-indexed `phi_embed` lookup table. This debate adjudicates whether the block-enlargement direction is a real, reportable efficient scaling axis or an artifact of that metric compounded by the 2× token-budget difference.

The load-bearing proposition is compound; it is treated as a single conjunction ("genuine AND publishable AND parameter-efficient AND a distinct scaling axis, NOT artifact+confound"). If the panel finds the empirical improvement real but the "parameter-efficient / publishable as-is" qualifier unsupported, that is a REMAND toward a narrower true sub-claim.
