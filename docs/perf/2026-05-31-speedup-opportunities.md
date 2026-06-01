# VFE_3.0 training speedup opportunities (multi-agent investigation, 2026-05-31)

Investigation only — no code was changed. Five expert agents (implementation/runtime,
numerical analysis, ML training/autograd, and two performance engineers) audited the hot
path in parallel; this report reconciles their memos. Every claim below is anchored to
`path:line` in the actual code, not docstrings.

## Method and the load-bearing caveat

A throwaway profiler ran 8 training steps at the click-to-run config
(`B=16, N=128, K=embed_dim=20, V=50257, L=n_layers=1, T=n_e_steps=1, heads=2`,
`group=block_glk`, `family=gaussian_diagonal`, `divergence=renyi/alpha=1`,
`e_phi_lr=0.0`, `detach_e_step=False`). **This box is CPU-only, so the profile is
DIRECTIONAL ONLY — its cost *ranking* does not port to the user's RTX 5090.** The CPU
numbers (backward 65%, decode-over-V 13%, E-step 19%) over-weight the dense `V=50257`
decode matmul and `matrix_exp`, both of which are near-trivial on GPU. Recommendations
below are ranked by **expected GPU behaviour** (occupancy, kernel-launch overhead,
memory traffic) and each finding is labelled GPU-real vs CPU-artifact. One caveat on the
"CPU-artifact" label: the `V=50257` decode + cross-entropy (and their backward — a ~412 MB
logit tensor at default `B·N·V·4`) are *lighter* on GPU, but they are the prime suspect to
re-emerge as the top cost once the serial loop is gone. Treat that label as **provisional
pending a GPU re-profile** (see Sequencing).

Measured CPU baseline: **1541 ms/step, 0.65 it/s**. The `128 = 8 × 16` E-step call count
confirms the serial batch loop.

## Priority summary

| # | Opportunity | Tier | Equivalence | Effort/Risk | GPU-real? |
|---|---|---|---|---|---|
| 1 | Batch-vectorize the `for b in range(B)` E-step loop | **P0** | exact (kernel bit-identical; transport matmul 1 ULP < golden atol) | S–M / low–med | **YES — the headline** |
| 2 | Skip the dense `(B,N,N,K,K)` Omega; fuse exp into mean/cov | **P0** | exact (verified 3.6e-15) | M / M | **YES — long-N memory** |
| 3 | One batched `matrix_exp` over stacked blocks (not a Python loop) | **P1** | bit-identical (0.0) | S / S | YES |
| 4 | Cap the periodic validation pass (`eval_interval`) | **P1** | n/a (eval-only) | S / S | YES |
| 5 | AdamW `fused=True` (CUDA) | **P1** | n/a (outside graph) | S / S | YES (small, free) |
| 6 | DataLoader `pin_memory` + `non_blocking` H2D | **P1** | n/a (data path) | S / S | YES (small, free) |
| 7 | `_beta_to_coordinate`: `repeat_interleave` → `expand/reshape` | **P2** | bit-identical (verified) | S / S | YES (small) |
| 8 | φ-frozen Omega hoist out of the E-step inner loop | **P2** | exact | S / S | only at `T>1`/`L>1` |
| 9 | `torch.compile(mode="reduce-overhead")` on the batched subgraph | **P2** | close (eager fallback kept) | M / M | YES (after #1) |
| 10 | Gradient checkpointing per block / E-step iteration | **P3** | exact | M / M | only at `T>1`/`L>1` |
| 11 | Double-clamp / per-step sync tidy-ups | **P3** | exact | S / S–M | marginal |
| — | TF32 / `matmul_precision('high')` | **REJECT** | breaks decode atol-1e-3 pin | — | unsafe |

Two non-speed **correctness flags** surfaced during the audit are recorded at the end.

## P0 — the two device-portable structural wins

### 1. The serial batch loop (`vfe3/model/model.py:123`)

`forward` runs the entire belief stack inside `for b in range(B):`, building a fresh
single-sequence `BeliefState` (`model.py:124`) and `torch.stack`-ing the outputs
(`model.py:128-129, 154`). With `B=16` the GPU sees sixteen *serial* passes of tiny
`(N=128, K=20)` kernels at near-zero occupancy; launch latency and ramp-up dominate
over useful FLOPs, and the Python loop forbids the scheduler from overlapping the
sixteen independent replicas. Cost scales as launch-overhead ∝ `B·L·T` and effective
occupancy ∝ `1/B`. This is the single largest GPU cost even though a CPU profile
(no occupancy penalty) under-weights it. The decode and the dense-Omega memory are
orthogonal to it.

The lower layers are *already* batch-transparent: `compute_transport_operators` takes
`(B,N,n_gen)` (`transport.py:96`), `transport_mean/covariance` carry the `b` axis
(`transport.py:181-205`), `pairwise_energy`/`attention_weights`/`natural_gradient`/
`retract_spd_*` reduce on negative axes, and `retract_spd_full` already reshapes
`(B,N,K,K)→(B·N,K,K)` (`retraction.py:68-71`). The active-path blocker set is **four
mechanical sites**: the loop + per-`b` `BeliefState` + the `stack`s (`model.py:123-129,154`);
the `.unsqueeze(0)…[0]` batch-strip in `_transport` (`e_step.py:32`); the same strip on
the transport calls inside `belief_gradients` (`kernels.py:153-154`); and the kernel
reduction einsums `"ijk,ijk->ik"` → `"bijk,bijk->bik"` (`kernels.py:96,100`). Feeding a
4-D tensor into the 3-subscript einsum raises a hard `RuntimeError` rather than silently
miscomputing, which makes the change safe to apply mechanically. The same pass must also
batch the off-default oracle strip (`oracle.py:65-66`) and the φ-island `_transport`
(`e_step.py:136-138`) so non-default toggles keep working.

There is no true per-sequence data dependency: each sequence reads only its own
`(mu,sigma,phi)[b]` and the shared, sequence-independent `log_prior`; the stack handoff
(`stack.py:34`) is per-position elementwise and broadcasts over `B` unchanged.

**Equivalence.** Verified by execution: the kernel reduction and the matrix exponentials
are bit-identical (`0.0`) batched vs looped; the only difference is the Omega-assembly
matmul `"bikl,bjlm->bijkm"` at `≈1.4e-7` relative (one float32 ULP, a GEMM accumulation-
order effect, identical math). Golden tolerances are `atol≥1e-6`/`rtol=1e-3`, far above
that. The pure path and registry seams are untouched — this is a layout change, not a
formula change. **Expected GPU impact:** removes ~`(B-1)/B = 94%` of launch-bound overhead
on the E-step and raises occupancy ~`B×`; plausibly the largest single win, low-single-digits
to ~10× on the E-step segment, growing with `B`. Peak transport memory after batching is
`B·N²·K²·4 ≈ 419 MB` at default and scales `O(N²)` — fine at `N=128`, but couple this with
finding 2 (or chunk the batch) before long-context runs.

### 2. The dense `(B,N,N,K,K)` Omega (`vfe3/geometry/transport.py:134`)

`compute_transport_operators` materializes `Omega_ij = exp(phi_i)·exp(-phi_j)` as a dense
`(B,N,N,K,K)` tensor — **419 MB forward + 419 MB autograd-saved = 838 MB at default**,
scaling `O(B·N²·K²)`: 3.4 GB at `N=256`, ~13 GB at `N=512`. This is the long-context
scaling wall. Both downstream consumers can avoid forming it:

- **Mean (exact, verified 3.6e-15):** `mu_t[i,j] = exp(phi_i) @ (exp(-phi_j) @ mu_j)`. The
  inner `m_j = einsum("bjlm,bjm->bjl", exp_neg_phi, mu)` has no `i`-dependence — compute it
  once per `j` (`O(B·N·K²)`), then `mu_t = einsum("bikl,bjl->bijk", exp_phi, m_j)`
  (`O(B·N²·K)`). The `(N,N,K,K)` tensor is never formed for the mean path.
- **Diagonal sandwich covariance:** does **not** factor by naively squaring the
  exponentials (verified failure, error 28.9 — the square sits inside the `l`-sum). It
  **does** factor by block-diagonality: per head it is a `(d,d)` sandwich, so the
  materialized intermediate drops from `K²` to `H·d²` (2× at default, `d/K →` better as
  heads grow). Verified exact (3.6e-15). Guard on `len(irrep_dims)>1` so single-block /
  cross-coupled / full-covariance paths keep the unfused route.

**Equivalence gate (hard):** these are reassociations of *linear* contractions (additions
reordered), round-off-level, distinct from the *nonlinear* squared-exp trap. The fp64
probe must be re-pinned at fp32 against the VFE_2.0 golden before shipping. **Effort M /
Risk M** — it changes `transport_mean/covariance` signatures, and the raw Omega is also
passed into `belief_gradients`/oracle, so those consumers must be re-plumbed or kept on the
Omega path.

**Honest memory accounting (do not over-claim).** The mean path forms no `(N,N,K,K)`
tensor at all. The covariance path still materializes a per-head sandwich intermediate
`Ob = (B,N,N,d,d)` and the outputs `mu_t`/`sigma_t` are themselves `O(B·N²·K)` and
unavoidable, so peak memory **stays `O(B·N²)`** — the fusion removes the dominant `K²`
*factor* (and the 838 MB dense-Omega term), it does not lower the asymptotic order. At
`N=512` the realistic effect is roughly **13 GB → ~4 GB** (≈6.7 GB if the per-head
intermediates are stacked, less if looped one head at a time), i.e. a large constant-factor
cut that makes long context *feasible*, not a sub-gigabyte result. Budget an N=512 run
accordingly.

## P1 — high-value, mostly free

**3. Batched `matrix_exp` (`transport.py:76`).** `_blockwise_matrix_exp` loops over irrep
blocks calling `torch.linalg.matrix_exp` per block (+ a separate `exp(-M)`); for
`block_glk` that is `H` sequential small-matrix launches plus a `zeros_like` scatter —
exactly the latency-bound pattern a 5090 punishes. One `matrix_exp` over a stacked
`(H,…,d,d)` tensor is **bit-identical (0.0)** and device-portable (forward and backward).
A secondary, *conditional* win — getting `exp(-M)` as `inv(exp(M))` rather than a second
exponential — is cheaper and was *more* accurate at moderate `‖M‖`, but is conditioning-
gated (`κ(exp(M)) ≤ e^{30}` at the `‖M‖=15` clamp could lose ~13 fp32 digits), so it must
be opt-in, not default.

**4. Periodic eval is unbounded (`vfe3/train.py:205-206`).** `evaluate(...)` at
`eval_interval=500` runs over the *entire* validation split with **no `max_batches`**
(`train_vfe3.py:182` caps only the train split). On wikitext-103 one such pass can dwarf
500 train steps. Thread a `max_batches` (e.g. 50) into the *periodic* call; leave the final
post-training eval (`train_vfe3.py:212`) uncapped. Eval-only, no golden impact. S/S.

**5. AdamW `fused=True` (`vfe3/train.py:42`).** One fused CUDA kernel for the M-step instead
of many small per-tensor kernels; per-group LRs are preserved. Guard to CUDA (fall back to
`foreach=True` on CPU). Outside the inference graph, so no golden impact. Free.

**6. DataLoader transfer (`vfe3/data/datasets.py`, `train.py:183-184,121-122`).** Add
`pin_memory=True` and `tokens.to(device, non_blocking=True)` so the (small, ~256 KB/step)
host→device copy overlaps compute. **Keep `num_workers=0`** — the dataset is an in-memory
tensor slice, so workers would cost more than they save. Free.

## P2 / P3 — micro and depth-dependent

**7. `_beta_to_coordinate` (`vfe3/gradients/kernels.py:179`)** uses `torch.repeat_interleave`
(index build + gather) to expand per-head beta `(N,N,H)→(N,N,K)`, ~0.21 s on CPU and a real
gather on GPU. Because `block_glk` blocks are equal size, `…unsqueeze(-1).expand(N,N,H,d)
.reshape(N,N,K)` is **bit-identical** (verified) and skips the gather; guard on
`len(set(irrep_dims))==1`, else fall back.

**8. φ-frozen Omega hoist (`e_step.py:181`).** With `e_phi_lr=0.0` (the default — φ evolves
only as a slow M-step variable, confirmed by the user), `belief.phi` is invariant across the
`n_e_steps` inner iterations *and* across blocks, so `_transport(belief.phi)` rebuilds a
**bit-identical** Omega `L·T` times. Hoisting it (guarded on `e_phi_lr==0`) saves
`(L·T − 1)` full transport+`matrix_exp` builds. **Zero benefit at the current `T=L=1`;**
a clean `T×`/`L×` transport saving the moment the user deepens the E-step or the stack.

**9. `torch.compile(mode="reduce-overhead")`** on the batched stack, *after* finding 1 gives
it a static-shape graph. The default path is compile-friendly — `e_phi_lr=0` skips the
`enable_grad`/`autograd.grad` φ island, the hand kernel is pure tensor code, and the registry
dispatch + caches are loop-invariant and trace away. Keep an eager fallback for the paths
CLAUDE.md requires pure (`e_phi_lr>0`, non-KL/oracle divergences, `pullback`, full-covariance),
and preserve the explicit `autocast(enabled=False)` islands around `matrix_exp`/SPD math.

**10. Gradient checkpointing** over `e_step_iteration`/`vfe_block` — exact, but frees memory
only at `L>1`/`T>1`; nothing to checkpoint at `L=T=1`.

**11. Tidy-ups.** `_diag_kl_filtering_kernel` re-clamps `sigma_q/sigma_p` that `_raw_diag_kl`
clamps again (`kernels.py:90,52`) — pass the clamped tensors through. The per-step
`float(loss.detach())` (`train.py:88`) and `.any()` (`model.py:140`) each force a host sync
every step; deferring to the log boundary is opt-in and couples to the it/s timer, so handle
carefully.

## Rejected: global TF32 / `set_float32_matmul_precision('high')`

Tempting for the decode matmul, but **unsafe here.** `_decode_diagonal`
(`prior_bank.py:234-259`) reconstructs the Mahalanobis term via a catastrophically-
cancelling matmul, tamed only by the `c = mean_v(mu_v)` offset trick and pinned to
`reference_decode` at **atol-1e-3**. TF32's 10-bit mantissa worsens exactly that
cancellation and would threaten the golden pin. Transport/SPD/KL already opt out of autocast
(`transport.py:53`, `retraction.py:37/73/131`), so the decode matmul is the one TF32-exposed
op — and it is the most cancellation-sensitive in the model. No global low-precision flag is
free; keep the pure fp32 path.

## Correctness flags (not speed — surfaced incidentally, worth a look)

1. **The autograd oracle detaches its output** (`vfe3/gradients/oracle.py:77-78` returns
   `grad_mu.detach(), grad_sigma.detach()`). The default hand kernel keeps the gradient live
   (`kernels.py:152-162`), but any non-default family that falls back to the oracle
   (`gradient_mode=smoothing`, `family=gaussian_full`, `alpha_div≠1`, surrogate) trains with a
   **truncated/detached** E-step gradient — the unrolled-through-inference signal to the prior
   tables is lost on those paths. If those toggles are meant to be first-class, this needs a
   `create_graph`/non-detach mode.
2. **`detach_e_step=True` gives `phi_embed` zero gradient** (per the Phase-2b spec): it is a
   degraded path (φ never trains, `mass_phi` inert), *not* an implicit-function-theorem
   fixed-point regime. A true 1-step implicit gradient is not implemented and would only be
   meaningful at large `T`.

## Recommended sequencing

Land **1 → 2 → 3** (the structural trio) first; that is where the 5090 throughput lives.
**Then re-profile on the 5090 before ordering the rest.** Collapsing the serial loop is a
regime change, and the post-batch GPU ranking will not match this CPU-derived one — the same
"rankings don't port" logic that motivated this whole investigation applies to the regime the
fix itself creates. In particular, expect the `V=50257` decode + cross-entropy and their
backward to climb once the loop overhead is gone; let the GPU profile, not this list, order
P1+. Turn on **4, 5, 6** immediately regardless (free, independent of the refactor). Add **7**
as a cheap bit-identical tidy. Defer **8, 10** until the operating point is `T>1`/`L>1`, and
**9** until after 1–2 give a static-shape batched graph. Every change must re-pass the VFE_2.0
golden suite at its pinned tolerances and keep the theoretically-pure path available under a
toggle.

## Open question for the user

The profiled point is the shallow default (`L=1, T=1`). Findings 8 and 10 (and the relative
priority of the autograd-graph work) change substantially with depth. If real runs will use
`n_layers>1` or `n_e_steps>1`, the φ-frozen hoist and checkpointing rise in priority.
