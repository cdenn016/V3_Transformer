# WIKI INGEST DRAFT (review before writing to vault)

> Staged in the V3 repo for review. NOTHING is written to the vault until confirmed.


## Target file operations

CREATE (1 new file):
- C:\Users\chris and christine\Desktop\Research\sources\runs\2026-07-05-blocks-k48-gauge-block-scaling.md   (type: run; the drafted_source_note below)

UPDATE (existing files):
- C:\Users\chris and christine\Desktop\Research\wiki\projects\VFE Transformer Program.md   (add Experiments-table row after line 49 + a "> [!note] Finding (2026-07-05)" callout; see drafted_project_update)
- C:\Users\chris and christine\Desktop\Research\index.md   (append the new run under "## 🧪 Sources — experiment runs" after line 1061; bump "At a glance" count "**4** runs" → "**5** runs" at line 13)
- C:\Users\chris and christine\Desktop\Research\log.md   (append the drafted INGEST line)
- C:\Users\chris and christine\Desktop\Research\wiki\fields\CS-ML MOC.md   (Bases-driven, auto-lists field/cs-ml — no manual edit needed, but verify the new note surfaces)

OPTIONAL propagation (per ingest steps 2-3, orchestrator's call):
- wiki/concepts/Neural scaling laws.md — could add ONE caveat sentence that a second, block-partition capacity axis (blocks_K48) exists but was REMANDed as a curve, NOT a scaling law / exponent (do not present it as an exponent). Recommend a short caveat only, to avoid contradicting the S1 verdict.
- wiki/concepts/GL(K) gauge-equivariant attention.md and/or theme "Gauge equivariance and geometric deep learning" — optional backlink to the new run.

DO NOT create a new concept page — existing [[Neural scaling laws]], [[GL(K) gauge-equivariant attention]], [[Irreducible representation]] cover it.


## Tags

Source run note (mirrors the grow_K_GL10 sibling note's tag set so the two scaling axes co-tag and are found together):
cluster/vfe, cluster/gauge-theory, cluster/info-geometry, cluster/attention, project/transformer, field/cs-ml

Rationale: cluster/gauge-theory (the GL gauge-block partition is the swept object), cluster/vfe (VFE transformer), cluster/attention (heads = 48/g), cluster/info-geometry (the KL/gauge machinery + the scaling-fit framing it deliberately refuses). project/transformer (vfe3 language model; single-project — a cluster/vfe+gauge note is project/transformer per the schema rule of thumb). field/cs-ml is the sole discipline (origin-first; source notes only) → also add the note to wiki/fields/CS-ML MOC (Bases auto-lists field/cs-ml). NOTE: cluster/methodology is defined in the schema as "the wiki pattern itself, tooling" (NOT research/scaling methodology), so it does NOT apply here despite the finding's methodological content.


---

## (1) CREATE  sources/runs/2026-07-05-blocks-k48-gauge-block-scaling.md

---
type: run
title: "blocks_K48 gauge-block scaling sweep — fixed embed_dim, vary GL partition GL(3)^16 → GL(24)^2 (S1 survives red/blue REMAND)"
aliases: [blocks_K48, blocks-K48 gauge-block sweep, K48 gauge partition scaling, GL block-partition axis]
tags: [cluster/vfe, cluster/gauge-theory, cluster/info-geometry, cluster/attention, project/transformer, field/cs-ml]
created: 2026-07-05
updated: 2026-07-05
---

# blocks_K48 gauge-block scaling sweep — fixed embed_dim, vary the GL partition

> [!info] Provenance
> `V3_Transformer` repo. Sweep `blocks_K48` under `vfe3_scaling_results/` (config resolved from
> `vfe3_scaling_results/blocks_K48/K48_GL24/s6/config.json`). Five gauge partitions of a fixed
> `embed_dim=48`, **3 seeds {6, 23, 64}**, **245,760,000 tokens/run**. WikiText-103 (shared
> `data_sha256` with the sibling `grow_K_GL10` sweep — confirm dataset field against the run config).
> All empirical numbers independently recomputed and CONFIRMED (per-label PPL std ≤ 1.08). Full
> adjudication: `V3_Transformer/docs/debates/2026-07-05-blocks-k48-gauge-block-scaling-axis/`
> (`04_verdict.md`, `05_action.md`, `01_evidence.md`).

> [!warning] Config toggle/group names pending re-verification
> A concurrent codebase cleanup was removing unused toggles when this note was drafted. The names below
> (`block_glk`, `tied_block_glk`, `so_n`, `use_head_mixer`, `use_prior_bank`, `use_cg_coupling`,
> `transport_mode='flat'`, `n_gen`, `phi_embed`) MUST be re-verified against the post-cleanup code before
> the note is committed or the paired experiment is launched.

## Debate outcome — REMAND (surviving sub-claim S1)

A full-panel red/blue debate asked whether block-enlargement at fixed width is "a genuine, publishable,
parameter-efficient scaling axis complementary to width-scaling." All three first-pass judges
(canon-strict, code-truth, scope) returned **REMAND**; the chief reconciler applied Rule 2 (scope override
on equivocation) because the claim is a five-fold conjunction whose equivocating terms live in the claim
text. The disposition is over-determined (Rule 3 unanimous REMAND). The empirical descent is real; three of
the four qualifier conjuncts fail. Report **S1 only, as a within-sweep structural ablation curve — not a
scaling law, not an exponent; do not attach "parameter-efficient" or "complementary to width-scaling."**

## Configuration (frozen)

Single-layer `block_glk` transformer, identical across all five cells except `n_heads`. `vocab_size` V = 50257
(GPT-2 BPE), `embed_dim` K = **48 (fixed across the whole sweep)**, `max_seq_len` N = 128, `n_layers` = 1,
`n_e_steps` = 1. `n_heads` H = 48/g = {16, 8, 6, 4, 2} for block width g = {3, 6, 8, 12, 24}, so the gauge
partition runs GL(3)^16 → GL(24)^2 and the per-block generator count is `n_gen = 48·g`. `transport_mode='flat'`
(Regime-I, the pure transport path). Two opt-in ablation toggles are ON: `use_prior_bank=False` (decode is the
learned linear projection `logits = mu @ W^T`, not the KL-to-prior decode) and `use_head_mixer=True` (learned
per-irrep-block head mixer; per the CLAUDE.md exception it breaks strict gauge equivariance off identity-init).
Model channel active: `prior_source='model_channel'`, `s_e_step=True`, `lambda_h=0.25`, `lambda_gamma=0.75`.
`use_cg_coupling=False`. The strictly-pure gauge-equivariant KL-decode path is NOT the one exercised here (it
exists under other toggles); per CLAUDE.md audit policy the question is not toggle purity but whether the
improvement is attributable to the gauge-block axis.

## S1 — the surviving empirical result

At fixed `embed_dim=48`, enlarging the GL gauge block GL(3)^16 → GL(24)^2 lowers test cross-entropy **strictly
monotonically**, 3-seed-robust (per-label std ≤ 1.08, far below every reported gap), along a previously
unpublished fixed-`embed_dim` block-partition design axis absent from both manuscripts:

| label | g | H | n_gen | n_params | active/tok | transport g² | wall s | CE | PPL |
|-------|---|---|-------|----------|------------|--------------|--------|--------|--------|
| GL3  | 3  | 16 | 144  | 19.37M | 12,061,920 | 9   | 6366  | 4.8249 | 124.57 |
| GL6  | 6  | 8  | 288  | 26.62M | 12,062,064 | 36  | 5010  | 4.7222 | 112.41 |
| GL8  | 8  | 6  | 384  | 31.46M | 12,062,160 | 64  | 4657  | 4.6653 | 106.20 |
| GL12 | 12 | 4  | 576  | 41.13M | 12,062,352 | 144 | 4796  | 4.6029 |  99.77 |
| GL24 | 24 | 2  | 1152 | 70.16M | 12,062,928 | 576 | 11049 | 4.5234 |  92.15 |

Report this as a **curve / ablation**, with causal decomposition and efficiency framing explicitly OPEN.

## What the debate struck (refuted by verified external canon)

- **"publishable exponent" — struck.** The fitted exponent is axis-dependent on identical CE data: offset
  α = 0.929 vs `n_params` (CI [0.07, 1.73], crossing α=1), 0.181 vs `n_gen`, degenerate (R²=0.17) vs analytic
  FLOPs. The `n_params` axis spans only 3.62× (< 1 decade), below the Stumpf–Porter two-decade credibility
  floor. Not a reportable scaling exponent — a mathematical artifact of fitting over a compressed x-range.
- **"parameter-efficient" — struck.** The flat ~12.06M `active_params_per_token` is a definitional identity
  `active = 5·V·K + 2·K + n_gen` (`vfe3/run_artifacts.py:616-620`; base 5·50257·48 + 2·48 = 12,061,776, only
  the additive `+n_gen` varies, +1,008 total = +0.008%), NOT a measurement — a token reads its own single
  `phi_embed` row (width n_gen), not the `V × n_gen` bulk. `phi_embed` is shape (V, n_gen) and accounts for
  99.7% of the 3.62× total-param growth. On honest compute axes the claim is adverse: the transport sub-term
  `2·N·g²` grows **64×** (9 → 576), `est_flops_analytic` looks ~flat (1.03×) only because fixed decode `2·V·K`
  dominates, empirical `wall_time_s` is **U-shaped** (6366 → 4657 GL8 minimum → 11049 GL24 slowest), and at
  matched total n_params blocks_K48 is **+4.8 to +12.8 PPL worse than width-scaling**. The Shazeer/Fedus
  constant-compute precondition for an "active-params" efficiency axis fails.
- **"distinct scaling axis complementary to width-scaling" — struck.** Inherently a cross-sweep comparison
  against `grow_K_GL10` (blocks 245.76M vs grow 491.52M tokens = **exactly 2×**), sitting on different
  Chinchilla D-slices; non-identified under Hoffmann's `L(N,D) = E + A·N^-α + B·D^-β` additive-in-D floor.

## Sibling axis (context, already recorded)

`grow_K_GL10` (the OTHER, already-published axis): GL(10) block fixed, grow width/heads K10→K120,
**491.52M tokens/run** (exactly 2× blocks, same `data_sha256`), params 7.6M → 90.7M, PPL 219.0 → 74.1;
offset fit vs n_params α = 0.558 (E = 3.95, PPL floor ~52, R² = 0.9996), exponent robust across axes
(0.558 n_params / 0.555 n_gen / 0.569 FLOPs). Recorded in [[2026-06-27-gauge-transport-ablation-suite]],
the [[VFE Transformer Program]] Experiments table, and GL(K)_supplementary.tex Appendix J
(`app:vfe3_scaling`, `tab:vfe3_scaling`, `fig:vfe3_gl10_scaling`). blocks_K48 varies the *opposite* axis
(fixed width, vary block partition) and is filed here as the complementary new axis.

## The paired experiment that converts REMAND → a compound win

Run all three arms; convert only if the GL3→GL24 improvement survives at matched D AND the plain V×m table
fails to reproduce the GL24 gain:
1. **Matched-budget blocks_K48 run** — a 491.52M-token blocks_K48 sweep (mechanically `batch_size=64` at the
   same `max_steps=60000`), removing the exact 2× cross-sweep D-slice confound under the `B·D^-β` floor.
2. **Non-gauge matched-parameter control** — a `V × m` learned-table baseline at `m = n_gen`, fixed head
   geometry, isolating the `gl(g)` generator algebra from raw `phi_embed` table capacity (discharges the
   Duhem/Mill/Woodward three-knob confound: n_params 3.62×, n_heads 16→2, block width g all co-vary).
3. **Tied-gauge (equivariance-clean) control** — a strictly gauge-equivariant `tied_block_glk` run with
   `n_gen = g²` (or its exactly-equivariant `so_n`-tied sibling), testing whether per-block untied richness
   (`n_gen = 48·g`) is the mechanism by checking whether the tied variant matches the untied curve at far
   fewer parameters. The current cells run `use_head_mixer=True` + `use_prior_bank=False`, which break strict
   gauge equivariance off identity-init.

Adjudicate efficiency only afterward, against wall-clock or a transport-inclusive FLOP axis — never against
the definitionally flat `active_params_per_token`. Spawned follow-up debates: **S2** (causal — is the gain
gauge-block structure per se vs raw table capacity / head-geometry recovery?), **S3** (efficiency — adverse
as a compute claim, needs a calibrated wall-clock/FLOP frontier at matched tokens), **S4** (cross-sweep —
needs the matched-token run AND an axis-invariant exponent, CI width < 2×).

## Relevance to this research

Third results-level scaling finding for the [[VFE Transformer Program]], and the first probe of a *second*
capacity axis: the gauge-block partition at fixed embedding dimension, complementary to the width axis of
[[2026-06-27-gauge-transport-ablation-suite]]. It supplies a clean, seed-robust, D-confound-free within-sweep
structural-ablation curve for [[GL(K) gauge-equivariant attention]] while explicitly refusing the scaling-law
framing — a cautionary companion to [[Neural scaling laws]] (axis-dependent exponents over a compressed
x-range; the active-vs-total parameter distinction; the 2× Chinchilla D-slice confound). blocks_K48 is
currently absent from both manuscripts; this note is its sole research record pending the paired upgrade.
Touches [[GL(K) gauge-equivariant attention]], [[Irreducible representation]], [[Group equivariance]],
[[Holonomy]], [[Neural scaling laws]].

## Cross-links
- Project: [[VFE Transformer Program]] · roadmap [[VFE Transformer Research Directions (2026-06-21)]]
- Sibling scaling axis: [[2026-06-27-gauge-transport-ablation-suite]] (grow_K_GL10, width axis)
- Manuscript: [[gl-k-attention]] (blocks_K48 not yet included; grow_K_GL10 is in GL(K)_supplementary Appendix J)
- Concepts: [[Neural scaling laws]] · [[GL(K) gauge-equivariant attention]] · [[Irreducible representation]] · [[Group equivariance]] · [[Holonomy]]


---

## (2) UPDATE  wiki/projects/VFE Transformer Program.md  +  index.md

In C:\Users\chris and christine\Desktop\Research\wiki\projects\VFE Transformer Program.md:

(A) Add this row to the Experiments table, immediately AFTER the existing line 49 ([[2026-06-27-gauge-transport-ablation-suite]] row):

| [[2026-07-05-blocks-k48-gauge-block-scaling]] | 48 / 16→2 | wikitext-103 | 124.57 → 92.15 | fixed-`embed_dim` gauge partition GL(3)$^{16}$→GL(24)$^2$ lowers CE strictly monotonically (S1, 3-seed std ≤ 1.08); reported as a within-sweep **structural ablation curve, not a scaling law** — a red/blue REMAND struck "publishable exponent", "parameter-efficient", and "complementary scaling axis" |

(B) Add this callout immediately after the existing "> [!note] Finding (2026-06-21) …" block (i.e. as a new note in the Experiments section):

> [!note] Finding (2026-07-05) — a second capacity axis (gauge-block partition), REMANDed to a curve. A new sweep `blocks_K48` ([[2026-07-05-blocks-k48-gauge-block-scaling]]) holds `embed_dim=48` fixed and enlarges the GL gauge block GL(3)$^{16}$→GL(24)$^2$ (`n_heads=48/g`, `n_gen=48·g`), the axis complementary to the width axis of [[2026-06-27-gauge-transport-ablation-suite]]. Test PPL falls strictly monotonically 124.57 → 92.15 across g = 3,6,8,12,24 at a 245.76M-token budget, robust over 3 seeds {6,23,64} (per-label std ≤ 1.08). A full-panel red/blue debate **REMANDed** the compound claim (unanimous 3-judge, Rule 2 scope override): only the ablation curve (**S1**) survives; it must be reported as a curve, **not** a scaling law. Struck: "publishable exponent" (axis-dependent — 0.929 vs `n_params` with CI [0.07, 1.73], 0.181 vs `n_gen`, degenerate vs FLOPs — over < 1 decade of range); "parameter-efficient" (the flat ~12.06M `active_params_per_token` is the definitional identity `5·V·K + 2·K + n_gen`, not a measurement; real transport grows 64× ∝ g², wall-time is U-shaped with GL24 the slowest run, and blocks is +4.8 to +12.8 PPL worse than width at matched total params); and "complementary scaling axis" (a cross-sweep comparison across a 2× token gap vs `grow_K_GL10`, non-identified under Chinchilla's `E + A·N^{-α} + B·D^{-β}`). Upgrade path (converts REMAND → win): a matched 491.52M-token blocks run + a non-gauge `V × m` learned-table control at `m = n_gen` + a tied-gauge (`tied_block_glk`, `n_gen = g²`, or `so_n`-tied) control. Absent from both manuscripts. Full adjudication: `V3_Transformer/docs/debates/2026-07-05-blocks-k48-gauge-block-scaling-axis/`.

(C) index.md: append under "## 🧪 Sources — experiment runs" (after line 1061):
- [[2026-07-05-blocks-k48-gauge-block-scaling|blocks_K48 gauge-block scaling sweep — fixed embed_dim, vary the GL partition (S1 survives REMAND)]]
and bump the "At a glance" line 13 count "**4** runs" → "**5** runs".


---

## (3) APPEND  log.md

```
2026-07-05  INGEST   blocks_K48 gauge-block scaling sweep + red/blue debate REMAND (from V3_Transformer docs/debates/2026-07-05-blocks-k48-gauge-block-scaling-axis). Created sources/runs/[[2026-07-05-blocks-k48-gauge-block-scaling]] (type: run): at fixed embed_dim=48, enlarging the GL gauge partition GL(3)^16→GL(24)^2 (n_heads=48/g, n_gen=48·g) lowers test CE strictly monotonically (PPL 124.57→92.15 across g=3,6,8,12,24) at 245.76M tokens/run, 3-seed-robust (std≤1.08) — a previously-unpublished fixed-embed_dim block-partition axis, absent from both manuscripts and the whole vault (obs 30609). Reported as a within-sweep STRUCTURAL ABLATION CURVE, not a scaling law: a full-panel debate REMANDed the compound claim (unanimous canon-strict/code-truth/scope, Rule 2 scope override), striking "publishable exponent" (axis-dependent — 0.929 vs n_params CI[0.07,1.73], 0.181 vs n_gen, degenerate vs FLOPs, <1 decade range), "parameter-efficient" (flat ~12.06M active/tok is the identity 5·V·K+2·K+n_gen at run_artifacts.py:616-620, not a measurement; transport grows 64× ∝ g², wall-time U-shaped GL24 slowest, +4.8–12.8 PPL worse than width at matched params), and "complementary scaling axis" (cross-sweep across a 2× token gap vs grow_K_GL10, non-identified under Chinchilla E+A·N^-α+B·D^-β). Recorded the paired upgrade experiment (matched 491.52M-token blocks run + non-gauge V×m table control at m=n_gen + tied-gauge tied_block_glk/so_n control) and spawned S2/S3/S4. Updated [[VFE Transformer Program]] (Experiments-table row + Finding note). index.md runs 4→5. NOTE: config toggle/group names (block_glk, tied_block_glk, so_n, use_head_mixer, use_prior_bank, use_cg_coupling, transport_mode='flat', n_gen, phi_embed) pending re-verification against a concurrent codebase toggle-cleanup.
```
