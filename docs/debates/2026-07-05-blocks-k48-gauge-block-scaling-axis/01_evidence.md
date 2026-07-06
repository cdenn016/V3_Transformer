# Evidence Pack — blocks-k48-gauge-block-scaling-axis

Neutral fact pack. Facts here cut in both directions; interpretation is for the coordinators. All empirical numbers were independently recomputed and CONFIRMED by an out-of-band verification pass (3 seeds {6,23,64} averaged per label; per-label PPL std ≤ 1.08, far below the reported gaps).

## Active config (resolved from `vfe3_scaling_results/blocks_K48/K48_GL24/s6/config.json`)

Identical across all `blocks_K48` cells except `n_heads`:

| key | value | note |
|---|---|---|
| `vocab_size` V | 50257 | GPT-2 BPE; fixed |
| `embed_dim` K | 48 | **fixed across the whole sweep** |
| `n_heads` H | 16 / 8 / 6 / 4 / 2 | = 48/g for g = 3/6/8/12/24 |
| `gauge_group` | `block_glk` | block-diagonal GL |
| `transport_mode` | `flat` | Regime-I (pure transport path) |
| `use_prior_bank` | **False** | decode is the **learned linear projection** `logits = mu @ W^T` (linear-decode ablation), not the KL-to-prior decode |
| `use_head_mixer` | **True** | learned per-irrep-block head mixer; per CLAUDE.md exception (2) it breaks strict gauge equivariance off identity-init |
| `prior_source` | `model_channel` | model channel active |
| `s_e_step` | True | model channel active |
| `lambda_h`, `lambda_gamma` | 0.25, 0.75 | both > 0 → `model_channel` branch on |
| `n_layers`, `n_e_steps` | 1, 1 | single layer, single E-step |
| `max_seq_len` N | 128 | |

So these runs are on **two opt-in ablation toggles** (`use_prior_bank=False` linear decode + `use_head_mixer=True`), plus the model channel. The strictly-pure gauge-equivariant KL-decode path is not the one exercised here (it exists under other toggles). Per CLAUDE.md audit policy the question is not whether these toggles are "pure" but whether the improvement is attributable to the gauge-block axis.

## Code references

- `vfe3/run_artifacts.py:616` — `active = (2 * V * K) + (2 * K + n_gen)`, then `+= V*K` when `not use_prior_bank` (line 617-618) and `+= 2*V*K` when `model_channel` (line 619-620). For this config that resolves to **`active_params_per_token = 5·V·K + 2·K + n_gen`**. With V, K fixed, the only sweep-varying term is the additive `+ n_gen`. Closed exactly: `5·50257·48 + 2·48 = 12,061,776` matches the CSV base (active − n_gen) to the integer.
- `vfe3/run_artifacts.py:597-603` — docstring states `n_params` is "dominated by the vocab-size gauge/prior tables (`phi_embed` is `V * n_gen`)"; `active_params_per_token` is "the honest working set (decode-bound, ~K, NOT phi/n_gen-bound)". (Docstring, not authority — the code above is authority.)
- `vfe3/run_artifacts.py:626` — `fpt_estep = L*T*(2*N*K + 2*N*d_head²)`, with `d_head = K/n_blocks` (line 611). For block_glk at fixed K=48, `d_head = g` (block width). So the transport sub-term `2*N*d_head² = 2*N*g²` grows **g²: 9 → 576 (64×)** across GL3→GL24.
- `vfe3/run_artifacts.py:625-627` — `est_flops_analytic = (fpt_decode + fpt_estep) * tokens_seen`, `fpt_decode = 2·V·K` (fixed). Because `fpt_decode` (≈12M/token) dominates `fpt_estep` (≈0.1–1.3M/token at these N,T,L), the reported `est_flops_analytic` varies only **1.03×** across the sweep even though the transport sub-term varies 64×.
- `vfe3/model/prior_bank.py:167` — `self.phi_embed = nn.Parameter(phi_scale * torch.randn(vocab_size, n_gen))` → shape **(V, n_gen)**.
- `vfe3/model/prior_bank.py:682` — `phi = pb.phi_embed[token_ids]` → **token-indexed row lookup** (one row of width n_gen per token), structurally an embedding table.
- Head mixer size scales with **H = n_heads = 48/g**, i.e. it is a per-block map on H heads. It **shrinks** as blocks enlarge (16×16 at GL3 → 2×2 at GL24), opposite to the loss improvement. Under `use_prior_bank=False` it composes with decode as `logits = W(M·mu) = (WM)·mu`.

## Empirical facts (seed-averaged; CONFIRMED independently)

### blocks_K48 — embed_dim=48 fixed, gauge partition GL3→GL24, **245,760,000 tokens/run**

| label | g | H | n_gen | n_params | phi=V·n_gen (% of params) | active/tok | transport g² | wall s | CE | PPL |
|---|--|--|------|---------|--------------------------|-----------|-------------|-------|-----|-----|
| GL3 | 3 | 16 | 144 | 19.37M | 7.24M (37%) | 12,061,920 | 9 | 6366 | 4.8249 | 124.57 |
| GL6 | 6 | 8 | 288 | 26.62M | 14.47M (54%) | 12,062,064 | 36 | 5010 | 4.7222 | 112.41 |
| GL8 | 8 | 6 | 384 | 31.46M | 19.30M (61%) | 12,062,160 | 64 | **4657** | 4.6653 | 106.20 |
| GL12 | 12 | 4 | 576 | 41.13M | 28.95M (70%) | 12,062,352 | 144 | 4796 | 4.6029 | 99.77 |
| GL24 | 24 | 2 | 1152 | 70.16M | 57.90M (83%) | 12,062,928 | 576 | 11049 | 4.5234 | 92.15 |

- PPL decreases **strictly monotonically** with block size g (124.6 → 92.2), robust across seeds.
- Total params grow **3.62×**; `phi_embed` accounts for **99.7%** of that growth.
- `active_params_per_token` grows by **+1,008** total (= Δn_gen) = **0.008%** — a token reads its own single phi row (n_gen wide), not the V×n_gen bulk.
- The design **co-varies three quantities at once** as g grows: total n_params (3.62×), n_heads (16→2), and block width g. "Richer within-block gauge" is not isolated from raw capacity or from head-count reduction.
- Wall-time is **non-monotonic (U-shaped)**: 6366 (GL3) → **4657 (GL8 minimum)** → 11049 (GL24). The headline "2.37×" is the GL8→GL24 min-to-max ratio, not a GL3→GL24 growth.

### grow_K_GL10 — GL(10) block fixed, width/heads K10→K120, **491,520,000 tokens/run (2× blocks, same data_sha256)**

- active/tok grows **2.5M → 30.2M (12×)** in lockstep with width; params 7.6M → 90.7M; PPL 219.0 → 74.1.
- Offset fit vs n_params: α = 0.558 (CI [0.39, 0.60]), E = 3.95 (PPL floor ~52), R² = 0.9996; **exponent robust across axes** (vs n_params 0.558 | vs n_gen 0.555 | vs analytic-FLOPs 0.569).

### Fit / cross-comparison facts (CONFIRMED)

- **blocks_K48 fitted exponent is axis-dependent on identical CE data:** offset α = **0.929** vs n_params (CI [0.07, 1.73]), **0.181** vs n_gen, degenerate (R²=0.17) vs analytic-FLOPs. Verifier note: this axis-dependence is a mathematical property of fitting over a compressed x-range (n_params spans only 3.62× because the fixed vocab/decode tables dominate), not a physical scaling exponent.
- **At matched total n_params, blocks_K48 is less parameter-efficient** than grow_K_GL10 by ≈ +4.8 to +12.8 PPL at all 5 matched points; survives seed noise. Caveat: "nearest grow point" is not always param-matched; an interpolated grow curve at GL3's 19.37M gives ~123 PPL, shrinking that particular gap.
- **At matched active-params/token (~12M) or matched wall-time,** blocks_K48's best point (GL24: CE 4.523, PPL 92.15) sits on / slightly below the grow frontier at the same active-compute (grow K50: active 12.56M, CE 4.544) — **but on half the tokens**. GL24 is also the slowest blocks run.

## Manuscript status (from freshest vault copies `Research/manuscripts/GL(K)_*.tex`)

- `grow_K_GL10` **is in the manuscript** (supplementary only): `\subsection{Embedding-Dimension Scaling}` `\label{app:vfe3_scaling}`, Fig `\label{fig:vfe3_gl10_scaling}`, Table `\label{tab:vfe3_scaling}`; summarized in the main abstract. Headline exponent reported under the **no-floor convention** α = 0.0873 (CI [0.0627, 0.0957], R²=0.9679); offset law E=3.95/α=0.56 also reported.
- The manuscript **already states the Chinchilla caveat**: α is "a parameter-count exponent at fixed data rather than an infinite-data exponent"; at K=120 the model sees "only about 5.4 tokens per parameter, far below the compute-optimal ratio of roughly twenty reported by Hoffmann … Chinchilla, so the upper end is data-limited and no compute-optimal or data-scaling exponent is claimed."
- `blocks_K48` (fixed-width, vary-block) is **NOT anywhere in either manuscript** — a genuinely new result (corroborated by memory obs 30609). The paper has only ever varied the opposite axis (fixed block, grow width).

## Canon excerpts (embedded; scaling-law + conditional-computation literature)

- **[Kaplan et al. 2020, "Scaling Laws for Neural Language Models"]** — test loss follows a power law in the axis that is actually varied; when a run is data-limited, the parameter-only power law is a fixed-data slice, not the infinite-data exponent. Comparing loss across models trained on different token counts confounds the N-scaling.
- **[Hoffmann et al. 2022, Chinchilla]** — `L(N,D) = E + A/N^α + B/D^β`; loss depends jointly on parameters N and data D. Compute-optimal ratio ≈ 20 tokens/param. A 2× difference in D between two sweeps is a first-order confound for any absolute-loss or floor (E) comparison.
- **[Shazeer et al. 2017; Fedus et al. 2021, Switch Transformer]** — the total-vs-active parameter distinction is standard for sparse / conditional-computation models. "Active params per token" is a legitimate efficiency axis **only when paired with a compute (FLOP or wall-clock) axis**; a large embedding/lookup table inflates total-param count while contributing one active row per token, exactly like `phi_embed`.

## What this evidence does NOT settle

1. Whether blocks_K48's improvement **survives at a matched token budget** (491.52M) — the matched-budget blocks runs were not done.
2. Whether the improvement comes from the **gauge-block structure per se** or from **raw added capacity** in a `V × n_gen` table that any non-gauge embedding of matched size might supply — no non-gauge control (a plain `V × m` learned table of matched size, or a matched-param dense head-mix) was trained.
3. Whether "parameter-efficient / publishable **as a scaling axis**" is the right frame at all, versus "a gauge-structure ablation at fixed width." The n_params axis spans only 3.62× and mixes three co-varying knobs (n_params, n_heads, g).
4. What the correct x-axis is for a headline efficiency claim: n_params (inflated by phi), active/tok (excludes phi bulk and transport FLOPs), analytic-FLOPs (decode-dominated, ~flat), or wall-time (U-shaped, empirical).
5. Whether the `use_prior_bank=False` linear decode + `use_head_mixer=True` toggles contribute learned-linear capacity that co-moves with the sweep (head mixer shrinks 16×16→2×2, so likely not the driver, but untested in isolation).
