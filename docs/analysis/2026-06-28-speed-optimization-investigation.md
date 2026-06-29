# Speed Optimization Investigation - 2026-06-28

## Scope and Method

This investigation used the clean worktree `C:\tmp\V3_Transformer-speed-audit-20260628`,
branch `perf/speed-audit-20260628`, at `origin/main` commit
`5e88afc941e4b08f96b2b057ad6c1cb41a0fdfa1`. The live checkout was dirty and was not used
for source claims. No production files were edited during the investigation, and no training or
pytest run was executed.

Five expert agents inspected independent surfaces: training/evaluation/reporting, model attention
and decode, numerical geometry, configuration and existing toggles, and an adversarial code-review
pass. I also verified the high-yield claims against executable source. Wiki context used:
`VFE Transformer Program`, `Symmetric spaces and the SPD cone`, `GL(K) gauge group`,
`Transformer Architecture`, `KV Cache`, and the `2026-06-27-gauge-transport-ablation-suite`
run note. The central constraints from that context are that learned GL(K) transport is
empirically load-bearing and that covariance matrices live on a precision-sensitive SPD geometry,
so speed work must preserve the exact learned-transport path and the numerically stable covariance
path unless an approximation is explicitly opt-in.

## Executive Verdict

Speed-up opportunities do exist. The safest immediate wins are not mathematical rewrites: they are
benchmarking, tuning already-wired chunking, reducing repeated validation/reporting replays, and
avoiding dense logits where only last-token or per-position statistics are needed. Mixed precision is
not a good general speed knob for covariance-bearing VFE paths: the SPD covariance operations,
transported KLs, Cholesky/eigendecomposition/log-det style numerics, and near-boundary conditioning
are precision-sensitive enough that `bf16` should be treated as an opt-in stress test, not a default
optimization. The highest-risk temptations are also clear: do not remove fp32/fp64 numerical islands,
do not silently approximate the decode KL ranking, do not present straight-through or detach as
transparent speed-ups, and do not generalize factored transport beyond the cases already proved exact.

## Highest-Priority Actions

### 1. Add a real CUDA benchmark/profiler harness

**Evidence.** `tests/test_perf_equivalence.py:1-11` explicitly states that current performance tests
are equivalence gates, not timing measurements. Searches found no `torch.profiler`,
`torch.cuda.Event`, CUDA synchronization harness, or `docs/perf` directory in the clean worktree.
Existing timing in `vfe3/train.py:726-735` and `vfe3/run_artifacts.py:524-583` records coarse
wall-clock summaries, not synchronized kernel or step timing.

**Why it matters.** Without a benchmark harness, changes to hot paths are hard to rank and easy to
misattribute. This is the gating task before deep kernel work.

**Fix.** Add a click-to-run benchmark script or module that measures `train_step`, `evaluate`,
one full eval event, `finalize_run` probes, decode chunk sizes, precision-mode stability checks,
natural-gradient modes, and generation. Use fixed synthetic or cached batches, warmup,
`torch.cuda.Event`, explicit synchronization, peak-memory reset, fixed seeds, and JSON output.
Keep it opt-in, for example behind `VFE3_BENCH=1`, so normal tests remain lightweight.

**Expected impact.** High decision value. Low correctness risk.

### 2. Do not promote AMP for covariance-bearing paths

**Evidence.** `vfe3/config.py:548-561` defines `amp_dtype`; `vfe3/model/model.py:501-524` can enter
autocast for `bf16` or `fp16`; `vfe3/model/model.py:745-805` keeps decode and cross-entropy in
fp32; `vfe3/train.py:644-648` enables `GradScaler` only for fp16. The wiki's SPD and GL(K) pages
identify the covariance as a first-class point on the SPD cone acted on by GL(K) congruence, which is
exactly the part of the model where low mantissa precision is most likely to perturb log-determinant,
inverse-variance, Cholesky, eigenspectrum, and transported-KL calculations. Tests in
`tests/test_amp.py:26-114` pin default-off behavior and fp32 decode/CE islands, but they do not prove
that autocasting the E-step is a safe training-speed optimization for covariance geometry.

**Why it matters.** The pure fp32 path is not just a conservative default; for this architecture it is
part of the mathematical contract. The model's capacity is carried by Gaussian beliefs
`(mu, Sigma, phi)`, not by ordinary dense neural layers where mixed precision is usually a free win.
Moving covariance-bearing inference into `bf16` risks changing attention rankings and belief
conditioning before any profiler result can be trusted.

**Fix.** Keep `amp_dtype=None` as the recommended training and audit path. If `bf16` is explored,
label it explicitly as an opt-in numerical-stability experiment, compare fixed-seed trajectories
against fp32, and require nonfinite checks, covariance-conditioning diagnostics, validation CE/PPL,
and final metric agreement before treating any timing result as meaningful.

**Expected impact.** High risk reduction by eliminating a false lead. Any speed gain from AMP should
be considered configuration-specific until the covariance diagnostics show that the training
trajectory remains equivalent enough for the research claim being tested.

### 3. Sweep `decode_chunk_size`

**Evidence.** `train_vfe3.py:120`, `ablation.py:133`, and `scaling.py:140` already select
`decode_mode='diagonal_chunked'`. `vfe3/model/model.py:753-789` avoids materializing `(B,N,V)`
logits when targets are present. `vfe3/config.py:391-395` exposes `decode_chunk_size`, currently
8192. `tests/test_chunked_decode.py` checks chunked value and gradient equivalence.

**Why it matters.** Chunked decode is already active; the next win is shape-specific tuning, not a
rewrite.

**Fix.** Benchmark chunk sizes such as 2048, 4096, 8192, 16384, and 32768 on the real
`B,N,K,V` shape. Record tokens/s and peak memory. Keep current equivalence tests.

**Expected impact.** High memory control and possible medium throughput gain. Low correctness risk.

### 4. Separate or fuse validation diagnostics and artifact replays

**Evidence.** One eval event can call `evaluate()` (`vfe3/train.py:751`), `_val_diagnostics()`
(`vfe3/train.py:782`), attention-map replay (`vfe3/train.py:789`), and gamma-map replay
(`vfe3/train.py:792`). `_val_diagnostics()` itself calls `model.diagnostics()` (`vfe3/train.py:519`),
`model.attention_maps()` (`vfe3/train.py:530`), `e_step_belief_trace()` (`vfe3/train.py:547`), and a
fresh dense logits pass (`vfe3/train.py:561-564`).

**Why it matters.** Validation cadence is replaying expensive E-step, transport, attention, and
decode work multiple times on the same batch.

**Fix.** Either introduce a validation snapshot that computes CE, converged state, attention maps,
and first-batch diagnostics once, or add separate cadences/toggles for heavy diagnostics and
attention PNG generation. The cheap train-loss/tokens-per-second log can remain at `log_interval`;
geometry diagnostics can move to `eval_interval` or an explicit `diagnostic_interval`.

**Expected impact.** High at eval time, medium to high for short sweeps. Risk is medium for fusion,
low for separate cadence.

### 5. Keep evaluation accumulation on device

**Evidence.** `evaluate()` syncs each validation batch by converting the valid-token count and CE to
Python at `vfe3/train.py:461-462`.

**Why it matters.** Each batch pays one or two host/device synchronization points.

**Fix.** Accumulate `total_nats` and `total_tok` as device tensors and convert once after the loop.

**Expected impact.** Low to medium, larger for many small eval batches. Low correctness risk.

### 6. Add last-token decode for generation

**Evidence.** In inference mode, `vfe3/model/model.py:790-793` materializes `(B,N,V)` logits, while
`generate()` immediately slices only `logits[:, -1, :]` at `vfe3/model/model.py:1200-1202`.

**Why it matters.** For `B=64,N=128,V=50257`, dense logits are about 1.53 GiB; last-row logits are
about 12.27 MiB. Generation does this every new token.

**Fix.** Add `forward_last_logits()` and `PriorBank.decode_last()` that return only the last row.
Route `generate()` through it. For greedy and top-k, consider streaming vocab chunks.

**Expected impact.** High memory reduction and a likely generation speedup. Low correctness risk if
guarded by `forward_last_logits(x) == model(x)[:, -1, :]` tests across decode modes.

### 7. Stream report/calibration statistics instead of dense logits

**Evidence.** `_val_diagnostics()` reintroduces dense validation logits at `vfe3/train.py:561-564`.
Report/calibration extractors also materialize logits and softmax statistics in `vfe3/viz/extract.py`
and `vfe3/run_artifacts.py`.

**Why it matters.** Training and evaluation already avoid `(B,N,V)` logits on target-supplied CE, but
diagnostic/report paths undo that memory win.

**Fix.** Add chunked per-position CE and chunked stats APIs for confidence, top-1, entropy, and
probability sums. Use them in validation diagnostics and report/calibration code.

**Expected impact.** High VRAM reduction during diagnostics and final reports. Medium implementation
risk because the dense statistics need careful equivalence tests.

### 8. Gate end-of-run probes and per-eval PNGs

**Evidence.** `finalize_run()` performs test eval (`vfe3/run_artifacts.py:624`), no-E-step eval
(`vfe3/run_artifacts.py:642`), an E-step trace (`vfe3/run_artifacts.py:662`), research probes
(`vfe3/run_artifacts.py:710`), and figure generation (`vfe3/run_artifacts.py:717`). Per-eval
attention PNGs detach maps to CPU and write per-head figures at `vfe3/run_artifacts.py:126-147`.

**Why it matters.** Finished training can still spend substantial time in probes and plotting. During
sweeps, the canonical numeric result usually needs test eval; the rest can be delayed or opt-in.

**Fix.** Add explicit `finalize_probes` or `research_artifacts` levels, and an independent
`attention_plot_interval` or off switch. Scaling already disables `generate_figures`; the remaining
probe and PNG surfaces need the same clarity.

**Expected impact.** High for sweeps and scaling runs. Low risk if canonical test eval remains default.

### 9. Optimize active natural-gradient gauge preconditioning

**Evidence.** The active click-run config enables `m_phi_natural_grad=True` in `train_vfe3.py:133-137`.
The custom gauge optimizer path is in `vfe3/train.py:183-200`. Agent inspection found repeated
pullback algebra metadata work and synchronization points in `vfe3/geometry/phi_preconditioner.py`,
plus `bool(active.any())` in `vfe3/gauge_optim.py:156-158`.

**Why it matters.** This path bypasses the fused AdamW path used for ordinary parameters, so optimizer
overhead may matter at current sizes.

**Fix.** Cache block membership, block generators, Gram pseudoinverses, and structure constants by
generator identity/device/dtype/shape. Solve pullback blocks independently instead of assembling a
block-diagonal full solve. Replace CUDA `.any()` Python branches with sync-free tensor-side handling
or tracked active-row metadata.

**Expected impact.** Medium to high for natural-gradient runs. Risk is medium because optimizer state
semantics must be preserved exactly.

### 10. Consider query-chunked or sparse causal/windowed E-step kernels

**Evidence.** `vfe3/geometry/transport.py:753-762` and `vfe3/geometry/transport.py:825-867` produce
full transported mean/covariance intermediates. `vfe3/gradients/kernels.py` consumes dense pairwise
energy before masking. Causal and windowed priors are available in `vfe3/attention_prior.py`.

**Why it matters.** The current pairwise energy path computes full `N x N` work even when a causal
prior discards future tokens, and windowed priors can keep only a small band.

**Fix.** Add dense-equivalent sparse row/window kernels for allowed prior structures, and add a
`query_chunk_size` route for the closed-form filtering kernel. Keep oracle and smoothing routes dense.

**Expected impact.** High memory reduction at long sequence length. Medium implementation risk, with
gradient and loss allclose tests required.

## Lower-Priority or Configuration-Specific Opportunities

- **Exact causal generation state cache.** `generate()` re-runs full forward for every token
  (`vfe3/model/model.py:1194-1202`). A cache over prefix beliefs, prior rows, and transport factors
  could be very high impact, but exactness conditions are architecture-specific and harder than a
  standard transformer KV cache.
- **Learnable T5 bucket/mask cache.** The learnable T5 path bypasses log-prior cache at
  `vfe3/model/model.py:451-458` and rebuilds bucket structures in `vfe3/attention_prior.py:260-270`.
  Cache bucket indices and future masks, not the live bias values.
- **Regime-II chunk budget.** `vfe3/geometry/transport.py` hard-codes `_REGIME_II_CHUNK_ELEMS`.
  Threading it through config would help large opt-in Regime-II experiments.
- **RoPE full-gauge factored transport.** `RopeTransport` can force dense materialization under
  full-gauge covariance transport. Exact factor algebra could preserve the factored path, but this is
  opt-in and should not precede benchmarks.
- **Full-covariance Renyi/SPD optimizations.** Some factorization and spectral projection work can be
  reduced on full-covariance paths. These are opt-in and must keep the current safe path as default.
- **DataLoader worker tuning.** Current data loading already uses cached tensors, CUDA-aware
  `pin_memory`, and non-blocking transfers. Do not add workers/persistent workers until profiling
  shows input stalls.

## Already Handled

- **Chunked training/eval CE is already active.** Target-supplied training/evaluation routes avoid
  dense logits when `decode_mode` is chunked.
- **Attention priors and RoPE are already cached for ordinary non-learnable cases.** See
  `vfe3/model/model.py:431-480` and `vfe3/model/model.py:483-499`.
- **Batch vectorization already exists.** `tests/test_perf_equivalence.py:64-78` pins batched forward
  against per-sample execution.
- **Factored flat equal-block transport already exists.** `vfe3/inference/e_step.py:85-150` selects
  factored transport only where exactness is established.

## Optimizations to Reject or Delay

- Do not remove decode/CE fp32 islands or full-covariance fp64 islands for speed. These protect
  cancellation-sensitive and ill-conditioned numerical paths.
- Do not treat AMP/bf16 as a transparent speed-up for covariance-bearing E-step or SPD paths. It can
  remain an opt-in experiment, but the default recommendation should stay precision-first.
- Do not silently clamp or approximate decode KL. `PriorBank.reference_decode()` disables `kl_max`
  for decode because vocab ranking must preserve full KL order.
- Do not present `straight_through` or `detach` as transparent speed-ups. They are gradient estimators
  with different training semantics.
- Do not widen factored transport beyond flat equal-block cases without exact algebra and tests.
- Do not force hand gradient kernels onto unsupported Regime-II, covariant, or value-decoupled paths.
- Do not spend first effort on generic `torch.compile` or CUDA graphs. They may be useful later, but
  the current code has Python-controlled E-step, chunking, optimizer, and scalar-control surfaces that
  need profiling and isolation first.

## Suggested Implementation Order

1. Add the benchmark/profiler harness.
2. Sweep `decode_chunk_size`; keep mixed precision only as an explicit covariance-stability audit.
3. Add on-device evaluation accumulation.
4. Add last-token decode for generation.
5. Gate or fuse validation diagnostics and per-eval artifact replays.
6. Stream per-position/report statistics through chunked decode APIs.
7. Optimize natural-gradient pullback/gauge optimizer internals.
8. Only then attempt sparse/query-chunked E-step kernels and exact generation-state caching.

## Verification Discipline

Every speed change should ship with both an equivalence guard and a timing/memory measurement. For
tests, reuse the existing style of `tests/test_perf_equivalence.py`, `tests/test_chunked_decode.py`,
`tests/test_amp.py`, and the transport/regime tests. For measurements, report synchronized CUDA
median/IQR timing, tokens/s, peak memory, shape/config, device name, and whether the fast route was
actually selected. No pass counts should be reported unless read from pytest output or JUnit XML.
