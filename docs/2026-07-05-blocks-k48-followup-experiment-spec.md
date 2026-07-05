# blocks_K48 follow-up experiment spec — convert the REMAND to a win

Derived from the red/blue debate `docs/debates/2026-07-05-blocks-k48-gauge-block-scaling-axis/`
(verdict REMAND; action `05_action.md`). Surviving sub-claim **S1**: at fixed `embed_dim=48`,
enlarging the GL gauge block `GL(3)^16 → GL(24)^2` lowers CE strictly monotonically
(PPL 124.57 → 92.15) at 245.76M tokens, 3-seed-robust. Three qualifier conjuncts were struck
("publishable exponent", "parameter-efficient", "complementary scaling axis"). This spec is the
paired battery that would upgrade S1 from a curve to a compound win.

> [!warning] Concurrent cleanup — verify names/lines before running
> A separate agent was removing unused toggles when this was written. Every symbol below
> (`route_vary_block_fixed_k`, `block_glk`, `tied_block_glk`, `so_n`, `use_head_mixer`,
> `use_prior_bank`, `use_cg_coupling`, `phi_embed`, `n_gen`, the BASELINE fields) MUST be
> re-verified against the post-cleanup code before launch. Cite by SYMBOL, not line number —
> the harness line numbers already drifted during the cleanup. This doc edits nothing; apply it
> after the cleanup lands.

Entry point (click-to-run, no CLI): `scaling.py` → `main()`; select routes via `CONFIG['routes']`;
routes live in the `ROUTES` dict; shared training config is the local `BASELINE` dict; cells built
by `_cell_cfg_dict`, run by `run_cell` → `vfe3.train.train` → `finalize_run`; analyze with
`scaling_analysis.py`. Output layout: `vfe3_scaling_results/<registry_key>/<label>/s<seed>/`.

Token budget identity (`vfe3/run_artifacts.py`, `tokens_seen = max_steps * batch_size * max_seq_len`;
no grad-accum factor). Persisted `blocks_K48` used `batch_size=32` → `60000*32*128 = 245,760,000`.
`grow_K_GL10` used `batch_size=64` → `491,520,000` (exactly 2×). The only differing knob is
`batch_size`; `max_steps=60000` and `max_seq_len=128` are identical.

---

## Arm 1 — matched 491.52M-token blocks_K48 run  (ready as config)

Removes the exact 2× Chinchilla D-slice confound so the cross-sweep comparison to `grow_K_GL10`
becomes identified under `L(N,D) = E + A·N^-α + B·D^-β`.

**Apply:**
1. Add a NEW route (do not reuse `blocks_K48` — see clobber note):
   ```python
   # in the ROUTES dict, alongside the existing blocks_K48 entry:
   'blocks_K48_2x': route_vary_block_fixed_k(48, [3, 6, 8, 12, 24]),
   ```
   (The current registry lists `blocks_K48 = route_vary_block_fixed_k(48, [48])` — single block only;
   the S1 window `[3,6,8,12,24]` must be supplied explicitly.)
2. Add `'blocks_K48_2x'` to `CONFIG['routes']`.
3. Budget knob: `BASELINE['batch_size'] = 64` (already 64 in the current BASELINE) →
   `60000 * 64 * 128 = 491,520,000`. Leave `max_steps=60000`, `max_seq_len=128`.

**Two mandatory conditions (else it is silently NOT a matched arm):**
- **`use_head_mixer=True`.** The persisted `blocks_K48` and `grow_K_GL10` both ran `use_head_mixer=True`,
  but the current `BASELINE` drifted to `False`. Override it True (per-cell override on the new route,
  or flip the BASELINE field). All S1-window cells GL3..GL24 have `h = 48/g ≥ 2`, so the mixer applies
  (the builder auto-disables it only at `h < 2`, i.e. the single-block GL48 cell).
- **New registry key.** Reusing `blocks_K48` would clobber the 245.76M summaries: `batch_size` and
  `use_head_mixer` now differ from the persisted `config.json`, so `_cell_is_current` returns False and
  the cells re-run and overwrite. Analyzer gotcha: `route_vary_block_fixed_k` hardcodes the internal
  cell route tag `f'blocks_K{embed_dim}' = 'blocks_K48'`, and `scaling_analysis.py` reads the route from
  `scaling_cell.json`; so point `scaling_analysis CONFIG['input_dir'] = 'vfe3_scaling_results/blocks_K48_2x'`
  (or patch each cell's `route` tag to `blocks_K48_2x`) to keep the 491.52M points separate from the
  245.76M points.

**No LR/schedule change needed.** Warmup is step-based (`warmup_steps=100`), the half-cosine decays over
`max_steps` (unchanged at 60000), and `base_lrs` are the configured per-group LRs with no batch/token
rescaling. Doubling via `batch_size` (not via `max_steps`) leaves the schedule step-identical — the same
mechanism `grow_K_GL10` used. Doubling via `max_steps=120000` would instead stretch the cosine and double
eval/checkpoint counts, diverging from `grow_K_GL10`'s schedule — do NOT use that lever.

**Seeds:** {6, 23, 64} (match S1). **Read out:** re-run `scaling_analysis.py` on the new input_dir;
compare the 491.52M curve to the persisted 245.76M curve and to `grow_K_GL10`.

---

## Arm 2 — matched-parameter control  (NOT a gauge group — a design choice, deferred)

Isolates the `gl(g)` generator algebra from raw `phi_embed` table capacity. The GL3→GL24 improvement
co-varies THREE knobs (n_params 3.62×, n_heads 16→2, block width g); this control holds the added
`V × n_gen` table capacity fixed while removing the gauge-generator structure.

**Sharpened finding (read the code):** the debate's literal control — a non-gauge `V × m` table at
`m = n_gen` under FIXED dim-3 head geometry (`n_heads = 16`) — is **impossible as a gauge group**.
`n_gen` equals the group's generator count (`n_gen = build_group(cfg).generators.shape[0]`), and any
`(m, K, K)` basis that actually transports is a `gl(K)` subalgebra. Reaching `m = 1152` on `K = 48`
needs 1152 independent generators — most of `gl(48)` (dim 2304) — which necessarily couple ACROSS the
dim-3 blocks: there is no "16 heads / dim-3 blocks / n_gen = 1152" transport. Growing `n_gen` at fixed
block size either enlarges the blocks (the GL24 direction under test) or adds off-block generators (a
LARGER gauge). Zero generators leave the table inert (no gradient), so that is not a capacity control
either. A clean control therefore needs a **non-transport table path** — a genuine feature, not a
`register_group`.

**Three candidate controls (choose before building — a research-design decision):**
1. **(2a) Plain non-transport `V × m` table** — a new encode path where a learned `(V, m)` table feeds
   an additive, structure-free contribution (to the logit or the belief), bypassing the `gl(g)` sandwich.
   Matched per-token params, no gauge. The debate's intended control and the cleanest structure-vs-capacity
   isolation, but a real ~1-day TDD feature (`@register_encode`/decode wiring + a config dim field + a
   golden test). **Recommended.**
2. **(2b) Plain multi-head baseline** — a standard scaled-dot-product transformer at `d_model = 48`,
   heads 16→2 (`d_k` 3→24), no gauge. Tests "gauge-VFE vs vanilla attention at matched width/heads";
   largest scope; partly reachable via the existing gauge on/off toggles.
3. **(2c) Fixed-base + cross-couplings** — `block_glk` at `n_heads = 16` padded by `cross_couplings` to
   grow `n_gen` toward the targets. Route-only (existing kwarg, no new code), but the padding generators
   are off-block GAUGE, so it isolates "n_gen count at fixed base block" — a weaker question than
   "gauge vs non-gauge."

**Chosen and implemented: option (2a).** The control is `encode_mode='per_token_additive'` (`prior_bank.py`):
the SAME learned `(V, n_gen)` `phi_embed` table is read, mapped by a FROZEN seeded readout `additive_R`
`(K, n_gen)` (a buffer, not a parameter — learned params stay `V·n_gen`, matched to the gauge cell) to an
additive mean shift `mu += phi @ R^T`, and encode returns `phi = 0` so `Omega = exp(phi·G) = I` (no gl(g)
transport). `n_gen` stays the gauge cell's `48·b` (block_glk), so the table width matches automatically.
Route `blocks_K48_ctrl_2x` (block_glk + `encode_mode='per_token_additive'` + `pos_phi='none'`, S1 window,
491.52M budget). Pinned by `tests/test_additive_table_control.py` (6 tests) and verified end-to-end
(model builds, loss finite, `phi_embed` learns via the additive path, `additive_R` is a non-parameter
buffer). Deliberately NOT gauge-equivariant — that is the control. **Decision:** if this structure-free
table reproduces the GL24 gain, the improvement is raw capacity; if not, the gl(g) structure earns it.

**Decision use:** if the plain `V × m` table reproduces the GL24 gain, the improvement is raw table
capacity, not gauge structure (S1 stays a curve). If it does NOT, the `gl(g)` structure is doing the
work.

---

## Arm 3 — tied-gauge equivariance-clean control  (ready as config)

Tests whether per-block UNTIED richness (`n_gen = 48·g`) is the mechanism, by matching the curve at far
fewer parameters with the tied gauge (`n_gen = g²`).

**Apply:** a route identical to Arm 1's cells but with `gauge_group='tied_block_glk'`:
```python
'blocks_K48_tied_2x': <route builder emitting cells with overrides
    {embed_dim: 48, n_heads: 48//g, gauge_group: 'tied_block_glk'} for g in [3,6,8,12,24]>,
```
`tied_block_glk` is registered; its generators are `kron(I_{n_heads}, gl(d_head))` with
`n_gen = d_head² = g²` (vs untied `block_glk` `n_gen = n_heads · d_head² = 48·g`). So per cell the tied
run has `g²` generators against `48·g` untied — e.g. GL24: 576 vs 1152; GL3: 9 vs 144.

**Bonus — equivariance-clean:** under `tied_block_glk` the Schur-commutant head mixer is *exactly*
equivariant, so this arm may keep `use_head_mixer=True` without breaking strict gauge equivariance
(unlike `block_glk`, where the mixer breaks it off identity-init). For a fully-pure variant also set
`use_prior_bank=True` (KL-to-prior decode) and consider the exactly-equivariant `so_n` sibling
(`n_gen = N(N-1)/2`). Run at the matched 491.52M budget, seeds {6,23,64}.

**Decision use:** if tied matches untied at far fewer params, per-block untied richness is not the driver;
if untied clearly beats tied, the extra `48·g − g²` generators are earning their keep.

---

## Conversion rule (REMAND → compound win)

Convert **only if**: Arm 1 shows the GL3→GL24 improvement **survives at matched D (491.52M)**, **AND**
Arm 2's plain `V × m` table **fails to reproduce** the GL24 gain. Arm 3 then attributes the effect to
untied-vs-tied richness. Adjudicate any efficiency claim afterward on **wall-clock or a
transport-inclusive FLOP axis** — never on the definitionally flat `active_params_per_token`
(`active = 5·V·K + 2·K + n_gen`; only the `+n_gen` term varies).

Spawned follow-up debates (see `05_action.md`): **S2** causal (Arms 2+3), **S3** efficiency (needs the
compute frontier), **S4** cross-sweep exponent (needs Arm 1 + an axis-invariant exponent, CI width < 2×).

## Order of operations
All three arms are implemented on branch `arms/blocks-k48-followup` (routes `blocks_K48_2x`,
`blocks_K48_tied_2x`, `blocks_K48_ctrl_2x`) and pinned by tests (14 route/geometry + 6 encode). On
`origin/main` (284b6a6) `BASELINE.use_head_mixer=True` and `batch_size=64` already hold, so after merging
the branch each arm runs by setting `CONFIG['routes']` and running `scaling.py`, then `scaling_analysis.py`
on the matching `vfe3_scaling_results/<route>` dir. Re-verify the symbols against `main` at run time.
1. Run Arm 1 (`blocks_K48_2x`) — matched-budget curve vs the persisted 245.76M run and vs grow_K_GL10.
2. Run Arm 2a (`blocks_K48_ctrl_2x`) and Arm 3 (`blocks_K48_tied_2x`) to decompose capacity vs structure.
3. Adjudicate efficiency (if at all) on wall-clock / transport-inclusive FLOPs, never `active_params_per_token`.
