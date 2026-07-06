# Memo — debate-expert-implementation-engineer — BLUE — round 1 — blocks-k48-gauge-block-scaling-axis

## Lens
Runtime behavior of the actual code — config trace, path:line reading, reachability proof.

## Active config used (resolved, `vfe3_scaling_results/blocks_K48/K48_GL24/s6/config.json`)
`vocab_size=50257`, `embed_dim=48` (fixed sweep-wide), `n_heads=2` (GL24; =48/g), `gauge_group=block_glk`,
`transport_mode=flat`, `use_prior_bank=false`, `use_head_mixer=true`, `prior_source=model_channel`,
`s_e_step=true`, `lambda_h=0.25`, `lambda_gamma=0.75`, `n_layers=1`, `n_e_steps=1`, `max_seq_len=128`,
`decode_bias=true`, `pos_phi=learned`, `learnable_r=true`. All `blocks_K48` cells are identical except
`n_heads` (16/8/6/4/2 for g=3/6/8/12/24), hence `n_gen = K*g = 48g` (144/288/384/576/1152).

## Steelman of the opposing position
The 3.62x total-parameter growth and the monotone PPL drop travel together; "active/tok ~constant" is a
metric convenience that hides where the added `V*n_gen` capacity and the 64x transport-compute growth
actually live, so the block axis is a capacity ablation on a data-halved budget, not an efficiency law.

## My position (in service of BLUE)
The two load-bearing runtime facts are code-true: at runtime a token touches exactly one width-`n_gen`
row of `phi_embed`, and the closed-form `active_params_per_token` varies by only `+n_gen` (+1,008 total,
0.008%) across the sweep. The CE improvement is real. The "parameter-efficient scaling axis" framing is
defensible only on the decode-bound working-set axis, and must be reported alongside the 64x per-token
transport-compute growth and the U-shaped wall-time — both of which the code confirms.

## Evidence (each line self-verified as reached under the active config)
- **`prior_bank.py:167`** — `self.phi_embed = nn.Parameter(phi_scale * torch.randn(vocab_size, n_gen))`. Shape is literally `(V, n_gen)`. Reached: constructed unconditionally in `PriorBank.__init__`.
- **`prior_bank.py:682`** — `phi = pb.phi_embed[token_ids]` → `(B, N, n_gen)`. A token indexes ONE row of width `n_gen`, never the `V*n_gen` bulk. This is the `encode_mode="per_token"` path (config confirms), reached every forward. CONFIRMED-RUNTIME-FACT: the "working set stays constant" claim is a real access-pattern property, structurally an embedding gather, not a metric artifact (cf. Shazeer 2017; Fedus 2021 — total vs. active param accounting).
- **`run_artifacts.py:616-620`** — `active = 2*V*K + (2*K + n_gen)`; `+= V*K` (use_prior_bank=False, line 617-618); `+= 2*V*K` (model_channel, line 619-620). Under this config both branches fire, so `active_params_per_token = 5*V*K + 2*K + n_gen`. I recomputed all five cells from the resolved config: 12,061,920 / 12,062,064 / 12,062,160 / 12,062,352 / 12,062,928 — matches the CSV to the integer. Base `5*50257*48 + 2*48 = 12,061,776`; the only sweep-varying term is `+n_gen`.
- **`head_mixer.py:105-107`** — `self.mixer_deltas = nn.ParameterList(nn.Parameter(torch.zeros(m, m)) ...)` with `m = n_heads` for `block_glk` (single equal-dims run, `irrep_labels=None`, line 73-80). Reached: `model.py:172` builds `HeadMixer` under `use_head_mixer=True`; `block.py:96-97` applies it after the E-step, before norm/decode. The learned mixer capacity SHRINKS 16x16=256 → 2x2=4 params as blocks enlarge — opposite to the loss gain, so it cannot be the driver.
- **`prior_bank.py:181, 625`** — `output_proj_weight = nn.Parameter(torch.empty(V,K))` (created only under `use_prior_bank=False`); `_decode_linear` reads `W = self.output_proj_weight`. So `logits = (M mu) @ W^T = mu @ (W M)^T` — the mixer composes with linear decode. Reached under this config.
- **`run_artifacts.py:611,626`** — `d_head = K/n_blocks = g`; `fpt_estep = L*T*(2*N*K + 2*N*d_head^2)`. The transport sub-term `2*N*g^2` grows 9→576 (64x) across GL3→GL24. But `fpt_decode = 2*V*K = 4,824,672` (line 625, fixed) dominates `fpt_estep` (14,592 → 159,744), so `est_flops_analytic` (line 627) moves only 4,839,264 → 4,984,416 = **1.030x**. Recomputed from config.

## Concessions / limits
- `active_params_per_token` is a CLOSED-FORM PROXY dominated by the fixed decode term `5*V*K`; it EXCLUDES the `V*n_gen` phi bulk AND the 64x transport growth. "Active/tok constant" is true only for the decode-bound working set and must be reported next to the transport-compute figure, not instead of it.
- `est_flops_analytic` is decode-saturated (1.03x) and near-useless as a headline efficiency exponent; the honest empirical compute signal is wall-time, which is U-shaped (6366 GL3 → 4657 GL8 → 11049 GL24). The "2.37x" headline is a GL8→GL24 min-to-max ratio, not monotone growth.
- Convention correction: evidence line 31 labels `fpt_decode ≈ 12M/token`; the code value is `2*V*K = 4.82M/token`. The ~12M figure is `active_params_per_token`, a different quantity. The decode-dominance conclusion is unaffected (4.82M >> 0.16M).
- Under `use_prior_bank=False` + `use_head_mixer=True`, decode carries a learned `(W M)` linear map co-moving with the sweep; the mixer shrinks (so likely not the driver) but is untested in isolation. The strictly-pure KL-to-prior decode path is NOT the one exercised here.
- The block axis co-varies THREE knobs at once (total n_params 3.62x, n_heads 16→2, block width g), and the runs use half the token budget of `grow_K_GL10` — a first-order confound for any absolute-CE comparison (Hoffmann 2022; Kaplan 2020).

## Falsification conditions
BLUE's runtime defense is wrong if: (1) `phi_embed` were read as a full `(V, n_gen)` contraction per token rather than the `[token_ids]` gather at `prior_bank.py:682` — it is not; (2) the `+= V*K` / `+= 2*V*K` branches at `run_artifacts.py:617-620` did NOT both fire under this config — they do (`use_prior_bank=false`, model_channel active); (3) the head mixer parameter GREW with g — it shrinks (`head_mixer.py:106`, `m=n_heads`); (4) `d_head` in `fpt_estep` were independent of g — it is `K/n_blocks=g` (line 611). The efficiency CLAIM (not the runtime facts) falls if a matched-token-budget (491.52M) blocks run, or a non-gauge `V x m` matched-capacity control, erases the CE gap.

## Newly-discovered canon (for 01b_extended_evidence.md)
- `pos_phi="learned"` is active in this config (config:30) — a second learned gauge table (positional phi) beyond `phi_embed`; add to the capacity accounting when isolating "gauge-block structure per se."
- `decode_bias=true` (config:85) allocates a learned per-vocab bias `(V,)` under the linear-decode path (`prior_bank.py:183-189`), read every decode — additional learned linear capacity co-moving with the sweep.
- `learnable_r=true`, `r_update_mode=gradient` (config:50-51): the hyper-prior centroid is a learned parameter here (memory 29498), another non-phi learned term.
- Reachability chain for the mixer, for other experts: `model.py:172` (construct) → `model.py:773/1258` (pass to block) → `block.py:96-97` (apply after E-step). `detach_e_step=false` here, so the `use_head_mixer`+detach freeze footgun (model.py:187-201) is not triggered.

## Confidence
HIGH on every path:line runtime fact (config-traced and recomputed); MEDIUM on the "efficient scaling axis" framing — a matched-token-budget blocks run or a matched-capacity non-gauge control would shift me.

## External citations (general principles only; code is canonical for behavior)
- Shazeer et al. 2017, "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer" — total-vs-active parameter distinction; a lookup table inflates total params while contributing one active row.
- Fedus, Zoph, Shazeer 2021, "Switch Transformers" — active-params-per-token is a legitimate efficiency axis only when paired with a compute (FLOP/wall-clock) axis.
- Williams, Waterman, Patterson 2009, "Roofline: An Insightful Visual Performance Model" — decode over all V is memory-bandwidth/arithmetic-intensity bound; explains why the analytic proxy saturates on `2VK` and why wall-time (not FLOP proxy) is the ground truth.
- Hoffmann et al. 2022 (Chinchilla) and Kaplan et al. 2020 — a 2x token-budget difference and a data-limited regime confound any absolute-CE or floor comparison across the two sweeps.
