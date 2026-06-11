# VRAM Investigation — flat / block_glk training path (2026-06-10 → 11)

**Status: all four actionable sinks fixed and committed on `vfe3-vram-fixes-2026-06-11`;
suite 843/0/0 (junitxml). Fix details in `docs/edits/2026-06-11-edits.md`.**

## Symptom

K=60, 6 heads, N=128, B=64 trains at ~5 it/s holding 20.4 GB of 32 GB (RTX 5090); K=80,
8 heads was reported feasible only at very small batch. VFE_2.0 handled K=120, B=16. The user
also reported `phi_precond_mode='pullback_per_block'` being worse for VRAM than
`'killing_per_block'`.

## Method

Five parallel code-tracing investigators (transport/Omega, phi machinery, E-step kernel and
decode, VFE_2.0 comparison, model/train loop), each computing byte counts at the active config
from source, followed by an adversarial verification wave that re-derived every dominant/major
claim against `path:line` evidence and CPU `saved_tensors_hooks` measurements. The
investigators' headline numbers were repeatedly wrong (a recurring bits-for-bytes 8× slip and
two 10× slips); every figure below is verifier-corrected.

## Verified accounting (retained to `loss.backward()` unless noted)

At the failing-side reference point (K=80, 8 heads, N=128, B=16) and scaled to the user's
live run (K=60, 6 heads, B=64) in parentheses:

| Sink | Mechanism | Size |
|---|---|---|
| BCH positional compose | 34 pinned `(B,N,K,K)` fp32 copies at order 4 — 12 bracket intermediates plus 22 broadcast-materialized duplicates of the one unbatched positional element (measured by saved-tensor census) | 1.78 GB (4.01 GB) |
| Linear-decode CE | `(B,N,V)` logits retained for the W-grad plus cross-entropy's log-softmax copy | ~0.8 GB (~3.3 GB) |
| Kernel pair family | ~10 saved `(B,N²,K)`-element tensors per call: mu_t, sigma_t (via its clamp), the beta_coord broadcast, and the einsum difference operands | 0.84 GB (2.5 GB) |
| f64 exp island | `build_factored_transport` passed no `exp_dim`, so K ≥ 20 upcast the full `(B,N,K,K)` to float64; ~0.42 GB transient per build + 26 MB retained (blocks), plus fp64-throughput `matrix_exp` | 0.4 GB tr. (0.9 GB tr.) |
| energy/beta/pair_mask | `(B,H,N,N)` ×3 | ~0.2 GB (~0.1 GB) |
| Optimizer/params | phi_embed `(V, n_gen)` + AdamW moments dominate | ~0.5 GB (~0.5 GB) |

With allocator caching and eval/diagnostic transients (the no_grad dense `(N,N,K,K)` Omega in
`diagnostics()`/`attention_maps` at 0.24–0.42 GB) this reproduces the observed ~20 GB at the
live operating point.

## Claims refuted by verification

The dense `(B,N,N,K,K)` Omega is NOT on the training path: `_can_fuse_flat` is verifiably true
for flat block_glk with `cross_couplings=None`, the kernel route is active, and every consumer
of the `FactoredTransport` stays factored — three independent investigator claims to the
contrary were refuted with reachability traces. `phi_precond_mode` is a dead knob at the
active config: both consumers are gated (`e_phi_lr > 0`; `m_phi_natural_grad=True`), so the
pullback-vs-killing difference the user observed was confounded (pullback_per_block would be a
≥15.7 GB sink only with `e_phi_lr > 0`). A config warning now discloses the dead knob.

## Fixes landed

The float64 island now keys on the per-block exp dimension at both flat builders (mirroring
regime_ii); `compose_bch` broadcasts once and runs its Dynkin brackets on the diagonal-block
stacks for multi-equal-block groups; the kernel contracts the compact per-head beta against a
head-shaped view instead of materializing `beta_coord`; and the linear decode gained the fused
chunked-vocab CE (`decode_ce_linear_chunked`), activated by `decode_mode='diagonal_chunked'`
with `use_prior_bank=False`. Expected combined effect at the live operating point: roughly
8 GB off the ~20 GB step plus the fp64→fp32 `matrix_exp` throughput recovery.

## Not done (candidates if more headroom is needed)

VFE_2.0-style per-head/row-tiled evaluation of the kernel pair family (the remaining ~2 GB at
B=64 sits in mu_t/sigma_t and the einsum difference operands — a deeper kernel restructuring
or checkpointing of the pair term); factored/no_grad transport for `diagnostics()` and
`attention_maps`; the `(B,N,d,d,d)` `ep2` outer product in `_factored_diagonal_covariance`
(small); bool storage for `pair_mask`.
