# Design: P3 Diagonal-KL Statistics Reuse

Date: 2026-07-11
Status: approved for implementation
Base: `origin/main` at `05070ecf4ed6b827c0ee9e70b0d49dcf48df0fa0`

## Problem

The P6 baseline run at K=20 measured 80.01 ms per clean training step and showed that 97.2 percent
of training-loop wall time is inside `train_step`. On the canonical diagonal-Gaussian, Renyi-order-1
filtering route, pairwise KL evaluation computes clamped transported variances and transported mean
differences before attention is formed. The hand filtering kernel and the `mm_exact` precision fusion
then reconstruct overlapping pairwise statistics after beta is available. At the measured
`B=64, N=128, K=20` shape, one `B x N x N x K` float32 tensor is 80 MiB.

## Selected design

Add an opt-in configuration field:

```python
reuse_pairwise_kl_stats: bool = False
```

The default is OFF. The OFF route must execute the existing implementation without constructing a
statistics object and must retain byte-identical outputs. Neither `train_vfe3.py` nor `ablation.py`
will be edited; experiments enable the field explicitly when they are ready to compare it.

The ON route is available only where the current hand-kernel predicate already proves the canonical
diagonal-KL filtering assumptions: diagonal Gaussian beliefs, Renyi order one, canonical attention
entropy, flat transport, and coupled score/value transport. Other covariance families,
divergences, oracle routes, and diagnostic uses of `pairwise_energy` remain unchanged.

A private, per-call statistics object will hold the clamped pair energy, its two-sided saturation
mask, the transported inverse variance, and the transported-minus-query mean difference. The object
is local to one E-step call and remains graph-live; it is never stored globally or on a model. The
same tensors feed attention and then either the registered filtering kernel or `mm_exact_update`.
The generic family/functional registry and `pairwise_energy` return contract will not change.

The helper preserves the current per-head layout for single-block, equal-block, and unequal-block
groups. It retains the three KL reductions as separate trace, Mahalanobis, and log-determinant sums,
then applies `safe_kl_clamp`. The pair mask is derived from the final clamped energy. The MM
self-divergence and its upper-only saturation mask remain separate because `D(q_i || p_i)=0` must
retain the prior anchor.

The optimized route is float32-only. If its inputs are not float32, the toggle falls back to the
legacy route so mixed-precision behavior does not change without separate evidence.

## Numerical contract

The shared reciprocal changes the final few float32 bits relative to repeated division. A direct
CPU probe measured maximum pair-energy drift of `7.63e-6` and maximum VJP drift of `3.05e-5` while
leaving clamp decisions unchanged. Acceptance therefore requires tight numerical equivalence, not
bitwise equality, on the ON route. Tests will use measured, explicit tolerances no looser than
`atol=5e-5, rtol=1e-5` for gradients and tighter tolerances for forward values. Pair masks,
degenerate pass-through, frozen sigma, exact zero, and exact `kl_max` boundaries remain exact.

The OFF route remains bitwise pinned. Existing oracle and noncanonical routes must not call the new
helper even when the toggle is set.

## Tests and verification

Implementation is test-first. RED tests will require a single shared-statistics construction and
prove that both the gradient and MM consumers use the supplied statistics. Characterization tests
will compare values and VJPs against a frozen legacy reference across single-head and per-head
layouts, two-hop coupling on and off, sigma update on and off, and every supported alpha-coefficient
shape. Boundary regressions will pin zero energy, exact `kl_max`, saturated rows, and the distinct
self/pair masks.

Focused CPU verification will cover the new tests plus the existing stationarity, monotonicity,
oracle fallback, compile fallback, and prior-anchor suites. The same focused correctness route will
run on CUDA. The complete default CPU suite will run before merge, with counts read from JUnit XML.

The user will run the long RTX 5090 performance benchmark after delivery. The handoff benchmark
compares OFF versus ON at identical configuration and random state, including the measured
`B=64, N=128, K=20, H=2` route. No timing threshold belongs in pytest.

## Scope exclusions

P3 does not change model settings, attention semantics, transport settings, optimizer behavior,
checkpoint schemas, reporting cadence, or finalization timing. It does not add a custom CUDA
operator and does not change `pairwise_energy` for generic callers. P1 packed transport, P2 lazy
vocabulary optimization, and P5 asynchronous best-state persistence remain separate tasks.
