# Why `transport_mean_per_head=True` gives no speedup (2026-07-11)

## Question

`transport_mean_per_head=True` was expected to speed up training but the `vfe3_runs` timings show
no improvement. Investigated via a multi-agent workflow (code trace, static cost model, kernel/
profiler analysis, run-log forensics) with two adversarial verification passes; neither verifier
refuted the root cause.

## Measured evidence (vfe3_runs)

Two 15,000-step wikitext-103 runs whose `config.json` files differ in exactly two keys
(`transport_mean_per_head` and `generate_figures`, plus timestamp); same seed, data hashes,
torch 2.10.0.dev/cu128, git SHA `05070ec`:

| run | toggle | mean step (CUDA events) | steady peak mem |
|---|---|---|---|
| `141.11_...` | False | 80.010 ± 0.342 ms | 3672.7–3672.8 MB |
| `140.85_...` | True  | 81.451 ± 0.358 ms | 3685.2 MB |

The True run is *slower* in 140/140 steady-state windows, +1.507 ± 0.363 ms (+1.9%), and holds
+12.4 MB. This is systematic, not noise (per-window z ≈ 4–7 sigma; the distributions barely
overlap). Confounds excluded: `train_step_ms_mean` brackets only `train_step` with CUDA event
pairs (`vfe3/train.py:1099-1104`, `vfe3/timing.py:59-106`), so eval/figures/loading are outside
the bracket; the `generate_figures` asymmetry biases *against* this finding (the figure-burdened
False run is still faster per step and in total wall clock, 1234.6 s vs 1246.7 s); all runs were
strictly sequential on the GPU; the later probe run at 77.24 ms rules out thermal drift.

## Root cause

The toggle works exactly as coded — the wiring is intact end to end and the promised FLOP cut is
real — but it optimizes a quantity (einsum MACs) that is not what the step spends time on, and its
implementation adds exactly the costs (kernel launches, memory traffic) that the step is actually
bound by.

**What the toggle changes.** Under `block_glk` + flat, the transport is a `FactoredTransport`
(per-token `exp(±phi)` only); the dense `(B,N,N,K,K)` Omega is never built on either setting
(`transport.py:1469-1474`, `e_step.py:146-150, 288-292`). False path: two full-K einsums
(`transport.py:1472-1473`). True path: `_factored_per_head_mean` (`transport.py:1759-1786`) loops
over the 2 gauge blocks, runs the same two einsums per d=10 slice, then `torch.cat`. It executes
exactly twice per step (s-refine and belief-block `mm_exact_update`, both at
`vfe3/gradients/kernels.py:461`); `share_refine_s_transport` dedupes the matrix-exp build (one
shared build at `model.py:945-956` carrying the flag) but cannot dedupe the two contractions
(different mu inputs). The covariance sandwich is per-head regardless of the toggle
(`transport.py:1789-1831`), so the toggle cannot touch it.

**Why the FLOP cut is invisible.** Per call the pair einsum drops from B·N²·K² = 419.4 MMAC to
B·N²·(10²+10²) = 209.7 MMAC — exactly the H=2× reduction advertised — i.e. ~2.5 GFLOP/step saved
including backward. But the step budget (fwd+bwd, ~122 GFLOP) is dominated elsewhere: 50257-vocab
linear decode CE ≈ 65.9 GFLOP (54%; gradient-checkpointed chunked CE runs the forward twice,
`prior_bank.py:964, 985-989`), diagonal covariance sandwich ≈ 25.2 GFLOP (21%, identical in both
runs), order-4 BCH pos-phi compose with n_gen=200 ≈ 17.7 GFLOP (14%), transport build ≈ 6 GFLOP —
the toggled mean transport is only ~4.1%. And the step is not compute-bound at all: 122 GFLOP /
80 ms ≈ 1.5 TFLOP/s ≈ 1.5% of the 5090's fp32 peak; estimated DRAM traffic (~12–17 GB) accounts
for only ~7–10 ms at 1.79 TB/s, so most of the 80 ms is eager-mode kernel-launch and Python
dispatch overhead. The pair einsum itself is bandwidth-bound (~10 FLOP/byte vs the 5090's ~56
FLOP/byte balance point), and both settings must write the identical 83.9 MB `(B,N,N,K)` output —
so the realistic Amdahl ceiling for the toggle is ~0.4–0.5 ms (~0.5%), even before its overhead.

**Why True is actively slower.** Profiled at the live shapes (B=64, N=128, K=20, H=2): the False
path is ~9 profiler kernel events fwd+bwd; the True path is ~69 (12 bmm, 16 copy_, 10 zeros/zero_
from `slice_backward` of the strided `exp_phi[..., s:e, s:e]` views, 6 contiguity clones — einsum
must materialize the non-viewable block slices — 3 grad add_, 1 cat). The `torch.cat` re-reads and
re-writes the full 83.9 MB output; profiler-measured allocation is 1425.7 MB vs 505.3 MB per call
(+1.84 GB/step). Under `e_step_gradient=unroll` every extra op has a backward twin in the loss
graph. Arithmetic: +65–120 launches/step at 5–20 us eager/WDDM launch cost (+0.5–1.3 ms) plus the
extra traffic (~+1 ms ceiling, overlapping) against a ~25–60 us FLOP saving nets +1.0–1.5 ms —
matching the measured +1.44 ms. The +12.4 MB peak matches the ~7.2 MB/call of contiguous slice
copies saved for backward (measured by autograd-graph walk; ×2 calls ≈ 14 MB upper bound).

**The expectation came from a different regime.** The deleted 2026-07-05 improvement-ideas doc
premised the toggle on the pair einsum being "two orders of magnitude above everything else" and
promised a "~4x cut of the dominant GEMM at H=4" — the production d_head=25 (K=100, H=4) regime.
At K=20/H=2 the cut is 2× of a ~4%-of-FLOPs term. No doc or test ever claimed a speedup at these
shapes; `tests/test_tier12_transport.py` pins exactness only (allclose 1e-6). Whether it wins even
at K=100/H=4 is unverified — the cat re-read/re-write and per-block launches also grow with K/H;
the crossover needs roughly the per-pair K² GEMM to dominate the N²·K output write (large K/H²).

## Secondary findings

- `exp_fp64_mode='dim'` does NOT put the hot-path matrix exp in fp64 at this config: the factored
  builder keys the fp64 decision on the block dim d=10 < threshold 20, so exps run fp32
  (`transport.py:1117-1127, 1421-1424`). The `exp_fp64_mode='norm'` probe run (20260711-144721)
  was ~0.7–0.8 ms slower than `dim` over its 7 warmup windows.
- The gamma-as-beta-prior fold (`gamma_as_beta_prior=True`) builds a third, fresh
  `FactoredTransport` per forward WITHOUT the Tier-1 toggle and runs a full-K `transport_mean`
  under `no_grad` (`model.py:974-984, 1688-1691`) — identical in both runs, so it does not affect
  the comparison, but it is a real per-step cost: the user's own probe 20260711-144919
  (`gamma_as_beta_prior=False`) ran at 77.24 ms vs 81.45 — the fold costs ~4.2 ms/step, ~3× the
  toggle's entire Amdahl ceiling.
- The `compile_pair_kernel=True` probe (20260711-144835) showed no early-window win (81.81 ms).
- `metrics.csv` writes `pipeline_tokens_per_s` into BOTH `tokens_per_s` and
  `pipeline_tokens_per_s` (`train.py:1356-1357`), so that pair is not an independent check; the
  real signal (`pipeline` vs `train_step_tokens_per_s`, median ratio 0.9939) shows data loading is
  never the bottleneck.
- Minor numeric corrections applied from verification: the `slice_backward` zero-fill for a
  `(B,N,K,K)` gradient is 13.1 MB (not 26.2); the `(B,N,10,10)` block copy is 3.28 MB.

## Bottom line

At K=20/H=2 the toggle halves FLOPs that were never the bottleneck while adding kernel launches,
contiguity copies, slice-backward zero-fills, and a full extra read+write of the 84 MB output via
`torch.cat` — in a step that is launch/bandwidth-bound and decode-dominated. At this scale it is a
small net pessimization (+1.5 ms/step, +12 MB); `False` is the better setting. If a per-head win
is ever wanted at small K, the cat could be eliminated by writing blocks into a preallocated
output, but the launch overhead and slice backwards would remain — and the whole op caps out at
~0.5% of step time regardless. The measurable per-step savings at this config live elsewhere
(the gamma fold's ~4.2 ms, the decode-dominated budget, and the ~70 ms of eager dispatch
overhead).
