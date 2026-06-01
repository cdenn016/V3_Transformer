# Post-edit log — training speedups (2026-05-31)

Branch `vfe3-perf-speedup-2026-05-31` (off the console-logging HEAD; a clean `checkout main`
was blocked by uncommitted `train_vfe3.py` console-logging changes). Companion to the
investigation report `docs/perf/2026-05-31-speedup-opportunities.md`. Every change is gated by
the pre-existing golden oracle in `tests/test_perf_equivalence.py` (captured before any perf
edit) plus the full suite. Baseline before edits: **212 tests pass** (junit). After: **214 pass**
(212 + 2 new tests), 0 failures.

## P1 — free / low-risk toggles (shipped)

- **`eval_max_batches` config knob** (`vfe3/config.py`): caps the PERIODIC validation pass that
  `train()` runs every `eval_interval` steps. Default `None` = the pure full-split eval (current
  behavior); set to a positive int for fast diagnostic eval. The final post-training eval at
  `train_vfe3.py` stays uncapped. Threaded into the periodic `evaluate(...)` call in
  `vfe3/train.py`. TDD: `tests/test_config.py::test_config_eval_max_batches_default_none_and_validated`
  and `tests/test_train.py::test_train_caps_periodic_eval_at_eval_max_batches` written RED first.
- **`evaluate()` loop draws exactly `max_batches`** (`vfe3/train.py`): was process-then-`enumerate`
  over-drawing one extra batch (yield-then-break); now breaks after processing `max_batches`.
  Computed CE is identical (same batches processed).
- **Device-conditional AdamW `fused`** (`vfe3/train.py` `build_optimizer`): `fused=True` when the
  priors live on CUDA (one fused M-step kernel), else standard AdamW on CPU. Per-group LRs honored.
- **DataLoader `pin_memory` + `non_blocking` H2D** (`vfe3/data/datasets.py`, `vfe3/train.py`):
  `pin_memory=torch.cuda.is_available()`; `tokens.to(device, non_blocking=True)` in the train and
  eval loops. `num_workers` stays 0 (in-memory tensor slice). No math change.

## P0 #1 — batch-vectorize the E-step (shipped) — the headline GPU win

Removed the serial `for b in range(B)` loop in `vfe3/model/model.py::forward`; the belief tuple
now carries a leading `B` axis through the whole E-step in one set of kernels. Mechanism: the
transport primitives `transport_mean` / `transport_covariance` (`vfe3/geometry/transport.py`) were
made **rank-agnostic** via leading-ellipsis einsums (`"...ijkl,...jl->...ijk"` etc.), so the
`unsqueeze(0)[0]` batch-of-one dances dropped out of `belief_gradients` (`vfe3/gradients/kernels.py`),
`phi_alignment_loss` (`vfe3/inference/e_step.py`), and the autograd oracle (`vfe3/gradients/oracle.py`).
`_transport` is now rank-aware (2-D diagnostics phi vs 3-D batched phi). The kernel reduction
einsums became `"...ijk,...ijk->...ik"`. `lie_ops`/`retraction`/`norms` were already rank-agnostic.

Correctness: sequences are independent (each reads only its own belief + the shared,
sequence-independent `log_prior`), so the batched result equals the per-sample result — pinned by
`test_batched_forward_equals_per_sample` (atol 1e-5) and the frozen forward checksums
`test_model_forward_matches_frozen_oracle` (loss/logits 1e-5). Both green; the only difference is
a ~1e-7 ULP from batched-vs-looped GEMM accumulation in the Omega assembly, far under tolerance.
The φ-island (`e_phi_lr>0`, exercised by the frozen oracle at `e_phi_lr=0.1, T=2, L=2`) and
`detach_e_step`, `killing_per_block`, full-covariance, and norm paths all stay green batched.

Measured (CPU, directional only): at the click-run config (B=16,N=128,K=20,V=50257,L=1,T=1)
**1541 → 1228 ms/step (~1.25x)**; the full test suite wall time dropped **193s → 42s**. CPU
understates the win (no occupancy/launch penalty); the real gain is collapsing B serial tiny-kernel
launches into one batched launch on the 5090.

## P0 #3 — batched per-block matrix_exp (shipped)

`_blockwise_matrix_exp` (`vfe3/geometry/transport.py`): when irrep blocks are equal size
(block_glk's GL(d_head)^H), the H diagonal blocks are stacked into ONE batched `matrix_exp` instead
of H sequential calls (the launch-bound pattern a GPU is starved by). Bit-identical to the loop
(Higham §10.3); pinned at 1e-12 by `test_per_block_exp_is_bit_equivalent_to_full_exp`. Unequal
blocks fall back to the per-block loop.

## P0 #2 — skip the dense (N,N,K,K) Omega — STATUS

Deferred pending a scope decision (see below). At the user's shallow N=128 operating point this is
a long-context MEMORY win (838 MB → O(B·N²) constant-factor cut), not a throughput win, and it is
the highest-risk item (block-structure-conditional covariance fusion). The verified algebra is in
the report; intended as an opt-in `transport_mode` toggle with the dense Omega as the default pure
path.
