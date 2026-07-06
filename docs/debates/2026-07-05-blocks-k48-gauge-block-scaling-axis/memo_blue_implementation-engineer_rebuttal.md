# Memo — debate-expert-implementation-engineer — BLUE — rebuttal — blocks-k48-gauge-block-scaling-axis

## Lens
Runtime behavior of the actual code — config trace, path:line reading, reachability proof. Two targets: (A) is `phi_embed` genuine embedding-table capacity or a metric trick, and (B) is the within-sweep token budget actually fixed, which decides whether red's D-slice attack can reach the effect.

## Active config used
Resolved from `vfe3_scaling_results/blocks_K48/{GL3..GL24}/s6/config.json['config']`, consumed by the proxy at `vfe3/run_artifacts.py:608-627`:
`vocab_size=50257`, `embed_dim=48` (identical in all five cells), `n_heads=16/8/6/4/2` (=48/g), `gauge_group=block_glk`, `transport=flat`, `use_prior_bank=False`, `use_head_mixer=True`, `prior_source=model_channel`, `s_e_step=True`, `lambda_h=0.25`, `lambda_gamma=0.75`, `encode_mode=per_token`, `decode_mode=diagonal_chunked`, `n_layers=1`, `n_e_steps=1`, `max_seq_len=128`, `batch_size=32`, `grad_accum_steps=1`, `max_steps=60000`. Derived token budget `60000*32*128*1 = 245,760,000` in every cell (verified by direct arithmetic on all five configs).

## Steelman of the opposing position
The flat active-parameter proxy is an algebraic identity at fixed V,K, the honest compute axis (wall-time) reverses, and the 2x token-budget gap makes every efficiency comparison unidentifiable — so no claim beyond a within-budget monotone trend survives.

## My position (in service of BLUE)
Red's two code-level pillars misfire against the effect itself: `phi_embed` is a genuine token-indexed embedding table (total-vs-active by construction, not a trick), and every blocks_K48 cell trains at the SAME fixed 245.76M-token budget, so the monotone CE decrease lives entirely at one D — placing it outside the reach of red's Kaplan/Hoffmann D-slice argument, which only bites cross-sweep.

## Evidence
- `vfe3/model/prior_bank.py:167` — `self.phi_embed = nn.Parameter(phi_scale * torch.randn(vocab_size, n_gen))`, shape `(V, n_gen)`. A learned parameter table, one row per vocabulary entry, structurally the GPT-2 `[50257 x 768]` token-embedding matrix with `n_gen` in place of `d_model` (Vaswani 2017 §3.4; Radford 2019). Total capacity is `V*n_gen`; at GL24 that is `50257*1152 = 57.9M`, 99.7% of the 3.62x growth — the total half of total-vs-active.
- `vfe3/model/prior_bank.py:682` — `phi = pb.phi_embed[token_ids]` returns `(B, N, n_gen)`: exactly ONE row of width `n_gen` per token. REACHED under the active config: `encode_mode='per_token'` selects `_encode_per_token` (`prior_bank.py:669-684`); the sibling `gauge_fixed` at `:687` is a `NotImplementedError` stub, not this path. The flat active/tok is the active half of the same identity, not a metric contrivance — the sparse read is real in the code.
- `vfe3/run_artifacts.py:616-620` — REACHED at `:757`; `use_prior_bank=False` fires `+= V*K` (`:617-618`), `model_channel=True` (`lambda_h=0.25>0`, resolved `:612-613`) fires `+= 2*V*K` (`:619-620`), giving `active = 5*V*K + 2*K + n_gen`. CONCEDED: with V,K fixed only `+n_gen` moves (`+1,008`, `+0.008%`); the flatness is an algebraic identity of the PROXY, and `fpt_decode = 2*V*K = 4,824,672` (`:625`) dominates `fpt_estep`, so the analytic-FLOP proxy is decode-bound. This concession is about the proxy's arithmetic, not the CE measurement.
- Fixed within-sweep budget (decisive against red section 2): all five configs carry identical `max_steps=60000`, `batch_size=32`, `max_seq_len=128`, `grad_accum_steps=1` → `245,760,000` tokens each, at `embed_dim=48` fixed; only `n_heads` varies (16->2), `n_params` 19.37M->70.16M. The 2x gap is CROSS-SWEEP only: `grow_K_GL10` differs solely by `batch_size=64` → `491,520,000` tokens (verified from `grow_K_GL10/K100_GL10/s23/config.json`). Therefore the monotone CE decrease GL3->GL24 is a single-D trend; Kaplan (2020) and Hoffmann (2022, `L(N,D)=E+A/N^α+B/D^β`) constrain comparisons across DIFFERENT D, and the `B/D^β` term is common to all five cells — it cannot generate a within-sweep ordering. Red's own falsifier #1 and section 2 conflate the cross-sweep confound (blocks vs grow) with the within-sweep effect.

## Newly-discovered context (for 01b_extended_evidence.md)
- The cross-sweep 2x token gap is mechanically a `batch_size` difference (32 vs 64) at identical `max_steps=60000`, `data_sha256`, not a design choice about D per se. The single missing artifact that removes the entire cross-sweep D-confound is a blocks_K48 run at `batch_size=64` (491.52M tokens); it was never run (`01_evidence.md:79`).
- `dataset=wikitext-103` in every blocks_K48 config, confirming shared data across the sweep.

## Falsification conditions
The BLUE code-truth spine is wrong if: line `prior_bank.py:682` is NOT reached (it is — `encode_mode='per_token'`); if any blocks_K48 cell's `max_steps*batch_size*max_seq_len*grad_accum_steps` differs from 245.76M (none does); or if the within-sweep CE ordering could be produced by the shared `B/D^β` floor (it cannot — that term is D-constant across the five cells). It does NOT rescue the compound efficiency/exponent claim, which I do not defend here.

## Confidence
HIGH — the fixed-budget fact is a direct arithmetic identity on all five configs and the phi_embed row-lookup is reached at runtime; I would shift only if a config file showed a per-cell budget difference, which it does not.
