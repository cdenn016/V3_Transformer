# Performance hypotheses for V3 from the VFE 4.0 whitepapers plus an eight-agent codebase sweep

Date: 2026-07-25
Baseline under investigation: `data/55.41_wikitext-103_K300_block_glk_linear_mix_s6`
(wikitext-103, test PPL 55.41, test CE 4.0148, 528,901,458 params, 180k steps at
batch 16 x seq 128 = 368.6M tokens, 18.7 h, RTX 5090, bf16, seed 6).

Sources read in full: `VFE_4.0/Manuscripts/vfe4_whitepaper/` (9 chapters) and
`VFE_4.0/Manuscripts/magent_elbo_whitepaper/` (13 chapters). Eight expert agents then swept
the executable against the run artifacts. Every claim below is tagged with its evidence.
Nothing here has been implemented; no application code was changed.

> **SUPERSEDED IN PART, 2026-07-25.** Sections 3.1 (transport clamp), 4.2 (attention
> temperature), 4.3 (position curve) and 4.4 (gauge table "decorative") were written before any
> of them had been measured. They were subsequently tested directly against the checkpoint and
> **four were refuted**: the clamp is load-bearing rather than defective, `kappa_beta=1.0` is the
> measured optimum, context is worth 1.68 nats rather than being flat, and zeroing the gauge
> table costs 320 PPL. Read
> `vfe3_runs/ablations_single_seed/138.40_mstep-phi-norm=5/phi-bound-calibration-and-stage0-report.md`
> first; it carries the measurements and the corrected recommendations. What survives from this
> file is the parameter accounting (section 2), the weight-decay finding (3.4), the diagonal
> covariance conditioning concern (1.5/2.1), and the depth question (2.3), none of which have
> been tested yet.

## 1. Two framing premises were wrong, and correcting them changes the priorities

**Decode is not the bottleneck.** The `flops_per_token_decode = 30,154,200` versus
`flops_per_token_estep = 614,400` in `summary.json` is not a measurement. It is the analytic
proxy `2*V*K` at `vfe3/run_artifacts.py:2694`, which the code itself labels "a proxy, not a
calibrated count" (`run_artifacts.py:2661,2692`) and which omits every matrix exponential,
the `(B,N,N,K)` covariance congruence, both float64 islands, and all backward passes.
Two agents measured the real split independently on the 5090 at production shapes:
the decode GEMM plus cross-entropy is roughly 6-21 ms of a **366.28 ms** step
(`train_step_ms_mean`, metrics.csv, std 1.43 ms over 180 rows), i.e. **4-6% of wall clock**.
The dominant cost is `matrix_exp` on `(2048, 10, 30, 30)`: **363.2 ms in fp64 versus 83.9 ms
in fp32**. The second cost is the transported-covariance einsum at
`vfe3/geometry/transport.py:2400`, 141.6 GFLOP forward and run twice per forward, versus the
decode's 61.8 GFLOP.

**`decode_tau=0.008` is inert.** Under `use_prior_bank=False` the linear decode at
`vfe3/model/prior_bank.py:1885-1904` never reads `tau_eff` (declared DISCARDED at :1889);
`vfe3/config.py:2049-2054` emits a warning saying exactly this. There is no 125x logit gain,
no bf16 softmax saturation, and no rare-token damage from this setting. Three agents
confirmed independently. Do not sweep it while the linear decode is selected.

A third correction: the run was **not** still improving at step 180k. `lr_mu` reaches the
`min_lr_frac=0.01` floor at step 169000; `val_ppl` went 57.175 @160k -> 54.034 @170k ->
54.181 @180k. `best_step=170000` is the cosine-anneal bottom. The final 11,500 steps (6.4%
of the budget) bought nothing.

## 2. Verified structural facts about the baseline

Parameter accounting was verified byte-exact against `best_model.pt`, not inferred:

| Tensor | Shape | Params | Share |
|---|---|---|---|
| `prior_bank.phi_embed` | (50257, 9000) | 452,313,000 | **85.52%** |
| 5 x `(V, K)` tables | (50257, 300) | 75,385,500 | 14.25% |
| `pos_phi_free` | (128, 9000) | 1,152,000 | 0.22% |
| `output_proj_bias`, `head_mixer`, scalars | | 50,958 | 0.01% |
| **Total** | | **528,901,458** | |

Two of those five `(V, K)` tables were **frozen at random initialization** for the entire run:
`mu_embed.std() = 6.4976e-2` against `mu_init_std = 0.065`, and
`sigma_log_embed.absmean() = 1.386295 = log(4)` with std `3.6e-7`. They received no gradient
(`can_omit_base_mean=True` on both the per-token encode at `prior_bank.py:1554` and the linear
decode at `prior_bank.py:1879-1883`), and no weight decay either, because
`zero_grad(set_to_none=True)` leaves `p.grad = None` and AdamW skips such parameters.
That is 30,154,200 parameters (5.7%) of dead weight. Commit `b362506` (2026-07-21,
`base_mean_consumed`, `prior_bank.py:459-476`) already drops them on `main`; the run started
2026-07-20 and predates the fix. A fresh build of this exact config now yields **498,747,258**
parameters.

Other confirmed facts: `n_layers` is a weight-tied recurrence, not a stack
(`vfe3/model/block.py:73-198` owns no parameters; `stack.py:100-148` loops the same
`cfg`/`group`/`prior_bank`), so additional depth costs **zero** parameters. The training loss
is plain cross-entropy — `train_loss == train_ce` in all 180 rows of all three runs checked;
the free energy is an E-step inner-loop diagnostic that never enters the M-step objective.
And 63.8% of the reported free energy is a constant: `self_coupling` sits in
`[300.044, 300.609]` for all 180k steps, pinned at exactly `K = 300`.

## 3. Tier 1 — defects that are silently capping the current model

These are the highest-confidence findings. Each is a case where the executable is computing
something other than what the configuration describes.

### 1.1 The transport clamp replaces `exp(M)` with a different operator for most tokens

`TRANSPORT_CLAMP_MAX_NORM = 20.0` is hard-coded at `vfe3/geometry/transport.py:1257` with no
config toggle, and is applied to the **full 300x300** matrix under `no_grad`
(`transport.py:1379-1418`), substituting `exp(20 M / ||M||_F)` whenever the norm exceeds 20.

It fires almost always. `phi_exp_clamp_frac` has mean **0.882** across the 180 logged steps,
was **1.0 for the first 125k of 180k steps**, and ends at 0.469 (val 0.539);
`phi_exp_scale_min` falls to **7.16e-4**. Meanwhile `phi_matrix_norm_p95 = 83.9` and
`max = 112.8`. Measured surrogate error against the true `exp(M)`: **25.9% at ||phi|| = 25,
72.0% at 40, 98.8% at 84, 99.9% at 113**. Because the rescale is detached and `mass_phi = 0.0`,
no gradient ever tells `phi` to shrink; `phi_norm_mean` wanders 37.7 -> 454.4 -> 31.6.

There is a second, subtler error: `block_glk` is a direct product `GL(30)^10`, so the chart
bound belongs **per factor**, but `_stable_compact_glk_exp_pair` reduces the norm jointly over
all ten blocks (`transport.py:1785`, `blocks.square().sum(dim=(-3,-2,-1))`) and applies one
shared scale (`:1797`). The effective per-factor bound is therefore `20/sqrt(10) = 6.32`,
roughly three times tighter than intended. A per-block check at the same threshold of 20 would
have fired on **0 of 10 blocks** for the observed norms.

Above the cap the radial coordinate of `phi` is exactly in the loss kernel (verified:
`exp(clamp(1.7 M))` versus `exp(clamp(M))` differ by 2.3e-16), so a large fraction of the
452M-parameter table has been receiving a spurious gradient in a direction the loss cannot see.

*Cheapest test, no retraining:* load `best_model.pt`, evaluate test CE with `max_norm = 20`
versus `max_norm = 60`. If CE moves, the clamp is load-bearing.
*Fix without code change:* `phi_mstep_max_matrix_norm = 5.0` (activates
`project_phi_parameter_rows_`, `train.py:747-753`), or `mass_phi = 1e-3` to bound `||phi||` by
a gradient rather than by truncation. *Structural fix (S):* per-block norm reduction in
`_stable_compact_glk_exp_pair`.

### 1.2 `matrix_exp` runs in float64 for no accuracy reason

`vfe3/geometry/transport.py:1444-1454`: under `exp_fp64_mode='dim'`, `d_eff = 30 >= 20`
forces `up_dtype = torch.float64` unconditionally on every forward, and the result is cast
straight back to fp32 at `:1465`/`:1472`. The `exp_fp64_norm_threshold=15.0` escape is read
only in the `large_skew` branch, which requires `skew_symmetric=True`; `block_glk` is not skew
(`transport.py:1616`), so that config value is **dead**.

Measured fp32-versus-fp64 relative error on 30x30 GL blocks: **2.6e-7 at the run's median
||phi||_F = 18.3**, 6.8e-7 at 20, 2.2e-6 at 113 — all below the fp32 rounding the output
already takes. Recovering the difference is up to **279 ms of a 366 ms step**.

*Change:* `exp_fp64_mode='norm'`, `exp_fp64_norm_threshold=21.0` (must exceed the 20.0
Frobenius clamp so the key never trips). Effort S, config only.
*Test:* 20 steps each way, compare `train_step_ms_mean` and step-0 logits (expect <= 1e-5
relative).

### 1.3 The BCH positional composition is 9-58% wrong

`pos_phi_compose='bch'` with `bch_pe_order=4`. The BCH series converges only for
`||A|| + ||B|| < log 2`; here `phi_matrix_norm_median = 17.44`. Measured in
`phi_numerics.json`: `bch_relative_error_median = 0.0940`, **p95 = 0.532, max = 0.580**.
The composed frame is simply not `exp(phi) exp(Y)`.

`pos_phi_compose='group_product'` is registered and validated
(`config.py:219`, `:863-869`; `transport.py:1905-1915`) and is exact. It costs one extra
`matrix_exp` over 128 positional matrices, about 6% of the 2048 token exponentials.
**Your uncommitted WIP at `train_vfe3.py:218` already sets this** — this finding confirms it
is the right call. Effort S, already done.

### 1.4 Weight decay annihilates the rare-token rows of the encode tables

A dead *tensor* escapes AdamW decay, but a dead *row* inside a live tensor does not:
`pb.phi_embed[token_ids]` (`prior_bank.py:1568`) produces a dense `(50257, 9000)` gradient, so
AdamW applies decoupled decay to all 50,257 rows on every step, whether or not the token
appeared. Measured against corpus counts (116,840,318 train tokens):

| train count | ntypes | % tokens | `norm(phi_v)` | `norm(W_v)` | `norm(s_mu_v)` |
|---|---|---|---|---|---|
| 0 | 2319 | 0.000 | **0.000** | 0.111 | **0.000** |
| 1-3 | 1528 | 0.002 | 2.047 | 0.674 | 0.637 |
| 4-15 | 2422 | 0.018 | 4.235 | 1.463 | 1.414 |
| 16-63 | 5144 | 0.169 | 7.302 | 2.158 | 2.345 |
| 64-255 | 12797 | 1.620 | 9.516 | 2.543 | 2.842 |
| 4096+ | 3270 | 78.185 | 12.568 | 2.790 | 3.240 |

Initialization is `0.06 * sqrt(9000) = 5.692` for `phi_v` and `0.065 * sqrt(300) = 1.126` for
`s_mu_v`. Zero-count rows are at 0.000 — `exp(-lr * wd * steps) = exp(-27) ~ 2e-12`. Rows for
counts 1-15 sit **below random initialization**. The predicted crush threshold
(half-life `ln2/(lr*wd) = 2310` steps versus inter-arrival `57050/c` steps, damage for
`c <~ 250`) matches the measured turnover at 64-256 exactly. `output_proj_weight` survives
because the softmax gives every row a dense gradient.

*Fix:* `phi_weight_decay = 0.0`, `sigma_weight_decay = 0.0`, and a `weight_decay = 0.0`
group for `s_mu_embed` (`train.py:274`) — the same exemption `output_proj_bias`, `r`, and
`t5_bias` already carry. Effort S, zero cost.

*Honest ceiling.* This explains the rare-token CE of 8.62, but the payoff is bounded. Two
agents independently bounded the frequent stratum's share of total CE at **91.7-92.8%** from
the mass constraint. Driving rare and mid all the way to the frequent stratum's 3.58 saves at
most **0.17 nats**, a floor near PPL 46.7. Worth doing (it is free), but it is not the path to
a large win.

### 1.5 Diagonal covariance discards more than it keeps at the observed conditioning

`_compact_factored_diagonal_covariance` (`transport.py:2371-2408`) computes the full congruence
`U_i (U_j^{-1} diag(sigma_j) U_j^{-T}) U_i^T` and returns only its diagonal — this is the
whitepaper's `eq:diagonal-covariance-nonclosure-example` realized in production. Measured
discarded off-diagonal Frobenius mass relative to the retained diagonal, at d = 30:
**0.51 at cond(U) = 2.1, 0.97 at 4.1, 1.67 at 16.1**. The run's `vertex_cond_median = 66.0`
and `p95 = 329`, well past the crossover where more is thrown away than kept. At cond 16 the
diagonal readout **understates the exact pair energy by 7.1x** and inflates attention entropy
from 0.03 to 1.00 nats.

## 4. Tier 2 — structural changes with genuine upside

### 2.1 Frame-intrinsic diagonal covariance (the strongest single proposal)

Declare `Sigma_i = U_i diag(sigma_i) U_i^T`, i.e. diagonal in the **fiber frame** rather than
the fixed basis. By KL gauge invariance (`eq:entropy-shift-kl-invariance`),
`KL(q_i || Omega_ij q_j)` reduces to `KL(N(a_i, sigma_i) || N(a_j, sigma_j))` with
`a = U^{-1} mu`. The family is closed under GL(K) by construction, so section 1.5 dissolves
without paying for full 30x30 blocks.

Verified: the resulting pair energy matches `FullGaussian.renyi_closed_form` to a maximum
error of **7.1e-15**. Measured cost at B=16, N=128, K=300, H=10, d=30, fp32: current mean plus
covariance transport plus KL is **29.67 ms / 5362 MB**; the intrinsic form is
**1.73 ms / 240 MB** — **17.1x faster and 22.4x less memory**. Every term factors into three
GEMMs per head.

This simultaneously restores the attention contrast the theory prescribes and frees enough
compute for roughly 2-3x more tokens or a substantially larger K. Register beside
`DiagonalGaussian` at `vfe3/families/gaussian.py:75` with a `transport_dispersion` that
whitens instead of sandwiching. Effort M-L. Falsify at K=64 over 2k steps against the diagonal
baseline, watching val PPL and `attn_entropy`.

### 2.2 Attention temperature: `tau = 5.477` against an ELBO-exact `tau = 1`

Confirmed in code: `free_energy.py:71-72` returns `kappa * irrep_dims[0]**0.5`, and
`block_glk` sets `irrep_dims = [30]*10` (`geometry/groups.py:275,283`), so **tau = 5.4772**.
The whitepaper's exact source row (`eq:mean-field-source-row-updates`,
`eq:mean-field-source-row-envelope`) has coefficient exactly one. V3's entropy *term* is
correct (`free_energy.py:440-459`); only the value of tau is off the unit scale.

The consequence is measurable. Closed form (SymPy, residual identically zero at order t^2):
`KL(softmax(log pi + t u) || pi) = (t^2/2) Var_pi(u) + O(t^3)`, so with `u = -E` and
`t = 1/tau`, `KL(beta || pi) = Var_pi(E) / (2 tau^2)`. From the run,
`val_attention_entropy = 12.891` gives `KL(beta_h || pi_h) = 0.235` nats and
`Var_pi(E) ~ 14.1` nats^2. Against that, head 8's ALiBi bias spans `0.707 * 127 = 90` nats in
a single row. **Content is a rounding error against position.** `kappa = 0.5` quadruples the
content KL; `kappa = 0.3` reaches 2.6 nats.

Note that ALiBi is *already* per-head Press schedule (`attention_prior.py:210-211`,
`_press_slopes` at `:39-48`); `alibi_slope=1.0` is a base multiplier, not a shared slope. The
"add per-head slopes" idea is already implemented.

*Free test, no retraining:* evaluate the existing checkpoint at
`kappa_beta in {1, 0.7, 0.5, 0.35, 0.25}`; tau enters only the softmax. An interior minimum
below test CE 4.015 confirms the lever. Then `learnable_kappa_beta=True` (10 scalars,
byte-identical at init; the gradient does reach it, `kernels.py:623`, and the
`detach_e_step` footgun does not apply since `detach_e_step=False` with `unroll`).

### 2.3 Depth, but only with the residual anchor restored

`pos_loss_first_q = 4.0161` versus `pos_loss_last_q = 3.9341`, ratio **0.9796**: predicting
token 120 with 119 tokens of context is 0.082 nats better than predicting token 10 with 9.
Worse, this degrades with capacity across runs — K=20: 0.8829, K=240: 0.9928, K=300: 0.9796.
Extra width buys better context-free prediction, not longer-range structure. With
`n_layers = 1` and `n_e_steps = 1`, no two-hop or induction composition is architecturally
possible.

The obstacle is that `estep_depth_sensitivity.json` shows free energy falling monotonically
(433.1 -> 386.4 -> 376.5 -> 372.6 -> 370.4) while CE **rises** past depth 1
(5.868 -> **3.774** -> 4.137 -> 5.027 -> 7.009). The same pattern holds in all three other
runs that measured it. The mechanism is now identified: `gradients/kernels.py:551-553` makes
the mean update a convex combination, so depth is repeated averaging, i.e. GNN-style
over-smoothing — and `stack.py:147` sets `mu_p <- (1-rho) mu_p + rho mu_q` with
`prior_handoff_rho = 1.0`, which **completely destroys** the token's own embedding prior after
layer 1. With the anchor gone, `n_layers > 1` is pure diffusion.

*Therefore:* `n_layers = 2-3` **together with** `prior_handoff_rho = 0.5` (a geometric
residual path back to the embedding at every depth), and only as a retrained arm, never as an
inference-time swap. Zero parameter cost; `prebuilt_transport` (`stack.py:142`, valid at
`e_phi_lr=0` plus flat) shares the expensive `matrix_exp` build across layers, so extra layers
pay only pair-energy. If depth lands, add `norm_type_block='mahalanobis'` — the diagonal
branch (`norms.py:58-61`) is invariant only under diagonal scalings, which costs nothing here
because `family='gaussian_diagonal'` is *already* declared non-GL(K)-invariant. That makes the
Fisher-metric norm free purity-wise in this configuration, unlike `layernorm`.

A complementary option is randomized-depth training (`randomize_e_steps=True`,
`e_steps_min=1`, `e_steps_max=4`, eval at `n_e_steps=4`), which addresses the train/test depth
mismatch directly. `estep_fp_kl = 0.593` shows one E-step is nowhere near the fixed point.

### 2.4 Re-budget the gauge table

Three independent signs it is not earning 85.5% of the parameters: the tail rows are decayed
to zero (1.4); the transport was a direction-only surrogate for most of training (1.1); and
`n_gen = K * d_head` is pure quadratic waste. The best controlled pair on disk — K=240 versus
K=300, identical 180k steps at batch 16 — spends **+105.77M parameters to buy 0.0505 nats**
(2.87 PPL), about 2.1M parameters per 0.001 nat.

Two concrete re-budgets:

- `gauge_group='tied_block_glk'` (`groups.py:296-329`): `n_gen = d_head^2 = 900`, phi table
  **45.2M**, freeing 407M. Under the tied gauge `Omega = kron(I, h)` the head mixer
  `M = kron(A, I_d)` commutes **exactly** (`head_mixer.py:38-42`), fixing a real defect: the
  current mixer is 78% non-equivariant (`val_builder_resid = 0.778`, `metrics.py:1666-1713`)
  and it is the *only* cross-head path, since Omega is block-diagonal and there is one
  application at `n_layers=1`.
- Keep `n_gen = K * d_head` but shift the ratio: K=600 with `n_heads=60`, `d_head=10` gives a
  phi table of 301.5M and a **total near 392M — smaller than today with double the logit rank**.
  Since the decode is `logits = mu @ W^T` (`prior_bank.py:1885-1891`), K is a hard softmax
  bottleneck (Yang et al. 2018), whereas `d_head` only sets gauge richness.

Measured width slope at matched tokens and steps: `d(CE)/d ln N = -0.226`. Rebudgeting to
K=500-600 predicts roughly `-0.115` nats, i.e. **PPL near 49**. Confidence low-medium (the
slope was measured at fixed `d_head=30`).

### 2.5 Make sigma load-bearing, or stop paying for it

`estep_grad_norm_sigma`, `estep_fp_sigma_rms`, `estep_r_sigma_last` are **exactly 0.0 in all
180 rows**; `e_step.py:1122-1124` passes sigma through unchanged; the linear decode marks
`sigma_q` DISCARDED (`prior_bank.py:1888`). `sigma_trace_cv = 0.1155` and
`sigma_ce_spearman = -0.101`: a nearly constant, nearly uninformative tag. The Gaussian belief
is operationally a point estimate.

Sigma is not entirely inert — `kernels.py:653-666` uses `1/sigma_p` and `1/sigma_t` as the mm
fusion's precision weights — so it is a *frozen learned per-token precision*. Reviving it is
the better bet than removing it, since the table is only 15.1M parameters (2.9%). Ship two
changes together (`config.py:2743` warns that either alone is incoherent):
`skip_belief_sigma_update=False` (mm_exact's `sigma* = (a + sum_j w)/P` at `kernels.py:676` is
exactly the block reverse-KL optimum `C_b* = J_bb^{-1}` from the MAgent paper) plus
`use_prior_bank=True` with `decode_mode='diagonal_chunked'`, which gives sigma a direct
likelihood gradient. Measured costs: +5.7 ms per E-step forward and about +1.2 ms of decode.

## 5. Tier 3 — whitepaper-derived ideas worth a look

**Additive predictive-kernel reference covariance.** V3's moving-peer edge has a verified sign
pathology: `dE/dS = (S - sigma_q - Delta^2)/(2 S^2)` is **negative** whenever
`S < sigma_q + Delta^2`, so a confident, well-matched key is *penalized for being confident*.
The whitepaper's fixed-R form gives `dA/dS = 1/(2R) > 0` always, and the additive predictive
covariance `S' = R + Omega Sigma_j Omega^T` (`eq:mean-field-predictive-kernel-covariances`) is
monotone for `R >= sigma_q + Delta^2`. Seam: `vfe3/families/base.py:258` plus
`gradients/kernels.py:596,659-663`; toggle `edge_reference_floor` with 0.0 reproducing the
current path byte-identically. Effort S/M.

**Recalibrate `b0`/`c0` so alpha is actually state-dependent.** `alpha^(k) = c0/(b0 + D^(k))`
(`alpha_i.py:126`) with `b0 = 1` and a measured `D^(k) = 3.21e-4` gives `alpha = 0.99968`,
a range of 1.0003x. The entire self-coupling row is the constant barrier `K * R(1)`.
`config.py:1610` warns about precisely this but gates on
`divergence_family == 'squared_hellinger'`, so it never fires. Setting `b0 = c0 ~ 1e-3` makes
the per-coordinate anchor live at zero parameter cost. Effort S, expected gain small.

**Solve the joint precision instead of one Jacobi sweep.** `kernels.py:551-553` is literally
`eq:gaussian-cavi-mean-update`: `mu*_i = (a mu_p/sp + sum_j w_ij mu_t/sigma_t)/P`, one Jacobi
step of an implicit joint precision with `J_ij = -w_ij Omega_ij / sigma_{t,ij}`. Replacing it
with 4-6 CG iterations at frozen beta reaches the exact block-CAVI mean without the beta
re-linearization that makes naive depth diverge. This is a direct test of whether the
depth pathology in 2.3 is Jacobi or the objective. Cost roughly +8 ms on top of 2.1;
effort M.

**Metric-aware gauge M-step.** `m_phi_update_mode='pullback_group'` routes phi through
`stage_pullback_group_direction` (`gauge_optim.py:190-264`), which supplies the float64 natural
direction, the group trust radius, `chart_max_norm` backtracking, **and** `weight_decay=0.0`
(`gauge_optim.py:81`) — addressing 1.1 and 1.4 at the source. Currently `adamw` registers empty
metadata (`gauge_optim.py:71-77`), so `phi_precond_mode`, `m_phi_group_trust_radius`, and
`phi_mstep_max_matrix_norm` are all dead config. Against it: forfeiting Adam's second moment on
a 452M-parameter embedding table is a real risk, and the K=20 ablation grid shows gauge and
optimizer knobs are inert at small scale. Effort S, confidence low.

**Schedule and batch hygiene.** `min_lr_frac = 0` (the floor wasted 6.4% of the budget);
`warmup_steps` 100 -> 1000-2000 (100 is below AdamW's second-moment horizon
`1/(1-beta2) = 1000`); `batch_size` 16 -> 32 (peak memory was 19.26 GB of 32 GB, constant all
run); `use_ema=True` with `ema_decay=0.999` (the current 0.95 is a 20-step horizon and inert
anyway); `z_loss_weight=1e-4`; `decode_unigram_prior=True` (the learned `output_proj_bias`
converged to absmean 4.82, max 15.47 — it spent training rediscovering log-unigram).
`grad_clip=1.0` never fired (max grad norm 0.818), so per-role clipping is currently inert.

## 6. Falsified or not worth doing

- **`decode_tau` sweep** — inert on the linear decode path. See section 1.
- **Per-head ALiBi slopes** — already implemented (`attention_prior.py:39-48`).
- **Longer context (128 -> 512)** — `pos_loss_ratio = 0.9796` says the position curve is flat;
  context is not the binding constraint until depth is fixed. Also `pos_phi_free` is an
  *absolute*-position table, so T is fixed at build time.
- **More E-steps at current weights** — `estep_depth_sensitivity.json` shows depth 2 already
  costs +0.36 nats CE. Only viable with 2.3's residual anchor and retraining.
- **An F-acceptance line search on the E-step** — `estep_f_nondecreasing_frac = 0.0` and
  `estep_f_drop in [-2.91, -0.62]` at every logged step: the step is *already* monotone in F.
  Licensing harder F-descent is measurably worse for the likelihood. Sweep `mm_damping` in
  {0.5, 0.75, 1.0} instead; it is the only live E-step knob (`e_q_mu_lr`,
  `e_step_mu_precond`, `mu_trust_mode`, `e_q_sigma_lr`, `e_sigma_q_trust`,
  `spd_retract_mode`, `phi_retract_mode`, `omega_reorth_every` are **all inert** under
  `e_step_update='mm_exact'` — see `e_step.py:1003-1039` versus the unreached `else` at
  `:1106-1131`).
- **Wiring `E_q[log L]` into the E-step** — position t's target is `x_{t+1}`, so a filtering
  posterior at t may only use `x_{<=t}`; the only non-leaky observation factor is `x_t`, which
  the encode prior already is (`model.py:971-973`). Conditioning `q_t` on `x_{t+1}` is a
  genuine leak because evaluation runs the same path. The affordable stand-in is
  `mstep_self_coupling_weight > 0` (`model.py:1598-1640`).
- **`cross_couplings`** — `geometry/groups.py:276-283` sets `irrep_dims=[K]`, collapsing 10
  heads to 1. That is single-head attention.
- **`transport_mode='regime_ii'` with `connection_W`** — `(n_gen, K, K) = 810M` parameters at
  K=300 *and* gauge-breaking. Reject. If curvature is wanted, the viable variant is a
  **relative** link `A_{i-j}`: `(2N-1, n_gen) = 2.3M` parameters, exactly covariant,
  nontrivial cycle holonomy, and length-extrapolating. Effort L.
- **`query_adaptive_tau` / `precision_weighted_attention`** — both modulate on `tr(Sigma)`,
  which is frozen with `sigma_trace_cv = 0.115`. Revisit only after 2.5.
- **Metrics that mean nothing:** `attn_entropy_collapsed_heads` = n_heads in every row of
  every run and `attn_entropy_min = 3.509e-09` identically — that is the causal-mask row-0
  entropy (one legal key), structurally saturated. `val_pos_content_r2` is corrupted by the
  127 masked diagonal entries clamped to `log(1e-12)` (`metrics.py:1151-1153`). Neither is
  evidence of anything.

## 7. Recommended sequence

**Stage 0 (free, no training).** Three checkpoint-only measurements that resolve the largest
uncertainties: (a) `kappa_beta` sweep for the attention-temperature curve (2.2); (b)
`max_norm` 20 versus 60 to size the transport clamp (1.1); (c) zero or mean-collapse
`phi_embed` and re-score, to test whether 85.5% of the parameters are decorative (2.4). Add
the full per-position CE curve and a context-truncation sweep — `pos_loss_ratio` is a
two-bucket summary and the full curve tells you whether this is effectively a bigram-plus-
position model.

**Stage 1 (config only, one short run).** `exp_fp64_mode='norm'` + threshold 21;
`pos_phi_compose='group_product'`; `phi_weight_decay=0`, `sigma_weight_decay=0`;
`min_lr_frac=0`; `warmup_steps=1500`; `phi_mstep_max_matrix_norm=5.0`; the winning
`kappa_beta` from Stage 0. Expect a large throughput gain and a modest PPL gain, with the
budget then spent on more steps or `batch_size=32`.

**Stage 2 (one code change each, 20k-step arms against the baseline's own 20k val PPL of
151.25).** Depth with the residual anchor (`n_layers=2`, `prior_handoff_rho=0.5`,
`norm_type_block='mahalanobis'`); then the sigma revival pair; then the gauge re-budget
(`tied_block_glk` at matched parameters, or K=600/d_head=10).

**Stage 3 (the big one).** Frame-intrinsic diagonal covariance (2.1). Prototype at K=64 first.

## 8. Bounds on what is reachable

Combining the honest ceilings: the rare/mid tail is worth at most 0.17 nats (PPL floor ~46.7);
the measured width slope suggests a re-budget to K=500-600 is worth roughly 0.115 nats
(PPL ~49); a log-linear fit of `val_ce` against `ln(step)` over 10k-100k gives -0.1891 nats per
e-fold with the anneal delivering ~0.55 nats beyond trend, so a retuned 2x schedule lands near
**PPL 48-50** on schedule alone. These overlap rather than add. The items that could plausibly
break out of the 45-50 band are the depth fix (2.3) and the frame-intrinsic covariance (2.1),
because both change what the model can represent rather than how well it is trained. Neither
has a measured magnitude yet.

One caution on training longer: on wiki-en at K=20, 1.87x more tokens made the model
**worse** (2.46B tokens -> 164.33 PPL versus 4.59B tokens -> 167.38 PPL). Treat "just train
longer" as an optimistic ceiling, not a safe bet, until the depth and over-smoothing behavior
is understood.
