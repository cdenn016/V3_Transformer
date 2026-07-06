# Action — blocks-k48-gauge-block-scaling-axis

**From verdict:** REMAND
**Reconciliation rule (panel=full):** Rule 2 — Scope override for REMAND on equivocation (unanimous REMAND across all three first-pass judges; disposition over-determined by Rule 3 majority).

## Recommended action

Report only the surviving sub-claim; strike three phrases from the original claim; run one paired experiment to convert the REMAND into a compound win.

### Report as-is (S1 — supported)

At fixed `embed_dim=48`, enlarging the GL gauge block `GL(3)^16 → GL(24)^2` lowers cross-entropy strictly monotonically (PPL 124.57 → 92.15) at a 245.76M-token budget, three-seed-robust (per-label std ≤ 1.08), along a previously-unpublished fixed-`embed_dim` block-partition design axis absent from both manuscripts. Present it as a within-sweep **structural ablation curve**, not a scaling law — causal and efficiency framing left explicitly open.

### Strike from the claim (refuted by verified external canon)

1. **"publishable [exponent]"** — the fitted exponent is axis-dependent on identical CE data (0.929 vs `n_params`, CI [0.07, 1.73] crossing α=1; 0.181 vs `n_gen`; degenerate vs analytic FLOPs), spanning < 1 decade — below the Stumpf & Porter two-decade credibility floor. Not a reportable exponent.
2. **"parameter-efficient"** — the flat ~12.06M `active_params_per_token` is a definitional identity (`5·V·K + 2·K + n_gen`, `run_artifacts.py:616-620`), not a measurement; on honest compute axes transport grows 64× (∝g²), wall-time is U-shaped (GL24 slowest), and blocks is +4.8 to +12.8 PPL worse than width at matched total params. The Shazeer/Fedus constant-compute precondition fails.
3. **"distinct scaling axis complementary to width-scaling"** — a cross-sweep comparison across a 2× token gap (blocks 245.76M vs grow 491.52M) on different Chinchilla D-slices; non-identified under `L(N,D) = E + A/N^α + B/D^β`.

### The one paired experiment that converts REMAND → win

Run a **matched 491.52M-token `blocks_K48` sweep** (`batch_size=64` at `max_steps=60000`, removing the exact 2× D-confound) **paired with a non-gauge matched-parameter `V × m` learned-table control** at `m = n_gen`, fixed head geometry (discharging the three-knob confound; isolating the `gl(g)` generator algebra from raw `phi_embed` table capacity). **Convert only if** the GL3→GL24 improvement survives at matched D **and** the plain `V × m` table fails to reproduce the GL24 gain. Adjudicate efficiency only afterward, against wall-clock or a transport-inclusive FLOP axis — never against the definitionally flat `active_params_per_token`.

Equivariance-clean add-on: a strictly gauge-equivariant **tied** run (`tied_block_glk`, `n_gen = g²`, or its exactly-equivariant `so_n`-tied sibling) to test whether per-block untied richness (`n_gen = 48·g`) is the mechanism — the current cells run `use_head_mixer=True` + `use_prior_bank=False`, which break strict gauge equivariance off identity-init.

## Follow-up debates (spawned sub-claims, each currently unsupported/ill-posed)

- **S2 (causal — unidentified):** improvement attributable to gauge-block structure per se vs raw table capacity / head-geometry recovery. Needs the `V × m` control, a non-gauge multi-head baseline (d_model=48, 16 heads × d_k=3 → 2 heads × d_k=24), and a tied-gauge run.
- **S3 (efficiency — adverse as a compute claim):** needs a calibrated wall-clock/FLOP frontier at matched tokens.
- **S4 (cross-sweep — non-identified):** needs the matched-token run AND an exponent that tightens to an axis-invariant value (CI width < 2×, agreeing across n_params / n_gen / a compute axis).
