# 2026-06-05 — Rényi divergence investigation: ablation confound + latent alpha>1 fix

Triggered by the user's report that `alpha_div < 1` trains ~2.5x faster than `alpha_div = 1`
and that the `vfe3_ablation_results/alpha_div` sweep shows Rényi at much worse validation
perplexity (~259-273 vs ~159 for KL). Four expert agents investigated (implementation,
information-geometry, numerical-analysis, variational). Full findings in
`docs/audits/audit-2026-06-05.md`; math verification logged in `docs/verified.md`.

## Diagnosis (no code defect in the divergence)

The Rényi math is correct (diagonal, full-cov, per-coord all verified against van Erven &
Harremoës 2014 / Gil 2013). The `alpha_div` ablation is confounded by a gradient-routing
asymmetry: `alpha_div = 1` uses the always-live analytic hand kernel
(`vfe3/gradients/kernels.py:179-194`), while `alpha_div != 1` falls back to the autograd
oracle, which under the default `oracle_unroll_grad=False` returns a detached belief
gradient (`vfe3/gradients/oracle.py:118`, gated by `vfe3/inference/e_step.py:357` +
`vfe3/config.py:247`). With the ablation's `e_phi_lr=0`, that detachment severs the prior
and gauge-frame tables from the loss, producing both the speedup (dropped backward graph)
and the degraded perplexity (untrained priors/frames). The cleanest proof is the sweep's
own `alpha_div=0.99` row, which jumps discontinuously to PPL 273 / 695s exactly at the
kernel-gate boundary — impossible for a continuous divergence-order effect. The pure
live-gradient path already exists via `oracle_unroll_grad=True`, so this is a
dangerous-default / confounded-ablation issue, not a missing-pure-path one.

## Changes

`ablation.py` — the `alpha_div` sweep entry now carries
`"requires": {"oracle_unroll_grad": True}` so every `alpha_div != 1` cell runs the live
oracle, making the sweep a clean single-variable divergence-order comparison rather than a
gradient-truncation comparison. This is a no-op at `alpha_div = 1` (the kernel ignores the
toggle), so the KL baseline is unchanged. (The file also carries the user's concurrent
sweep operating-point tuning.)

`vfe3/families/gaussian.py` — latent-bug fix (found by the numerical agent, not the user's
α<1 regime). The diagonal Rényi closed form and per-coordinate form previously did
`sigma_blend = (...).clamp(min=eps)`, which for `alpha > 1` silently turns a non-positive
(indefinite) blend into a tiny positive variance and emits a wrong finite divergence with
a nonzero gradient, escaping the `kl_max` sentinel that the full-covariance path correctly
uses via `safe_cholesky`. Both branches now build the raw blend, clamp only to guard
log/division on the in-bounds coordinates, then map non-positive-blend elements to
`NaN -> kl_max` (summed: any non-PD coordinate masks the whole pair; per-coord: only the
bad coordinate is masked). For `alpha in (0,1)` the blend is a convex combination and
always positive, so the mask is inert and the path is byte-identical (the user's regime is
unaffected).

`tests/test_divergence.py` — two new tests (TDD, written failing first):
`test_diagonal_renyi_alpha_gt_one_negative_blend_masks_to_kl_max` (summed) and
`test_diagonal_renyi_per_coord_alpha_gt_one_masks_only_bad_coord` (per-coordinate), mirroring
the existing full-cov `alpha > 1` mask tests.

`docs/verified.md`, `docs/audits/audit-2026-06-05.md` — verification log and full audit.

## Verification

Full suite: 575 tests, 574 passed + 1 xpassed, 0 failures, 0 errors (junit XML, CPU box).
The new tests fail against the old `clamp(min=eps)` code (returned 4.73 / 4.61 instead of
kl_max) and pass after the fix. The ablation override is wired:
`make_run_overrides('alpha_div')` yields `{'oracle_unroll_grad': True, 'alpha_div': 0.99}`.

## Not done (needs the GPU)

The corrected sweep must be RERUN on the user's RTX 5090 (this box is CPU-only) to get the
valid divergence-order numbers: `python ablation.py` with the `alpha_div` sweep. Expect
Rényi to be slower than KL (autograd vs analytic kernel) and the perplexity gap to shrink.
Two real non-bug effects mean exact parity is not guaranteed even after the fix: the
state-dependent self-coupling `alpha^(k) = c0/(b0 + D)` is larger for `alpha < 1`
(Rényi D <= KL), and attention softens at fixed `tau = kappa*sqrt(K)` (retune kappa). For
`alpha != 1`, F is a heuristic consensus functional, not an evidence bound (author-disclosed,
`GL(K)_attention.tex:771`).
