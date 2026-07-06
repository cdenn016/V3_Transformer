# Memo — debate-expert-ml-engineer — red — opening — blocks-k48-gauge-block-scaling-axis

## Lens
General ML / optimization — scaling laws (Kaplan, Chinchilla), the compute-vs-parameter
distinction (MoE active/total), fixed-token vs infinite-data exponents, power-law fit
well-posedness (CI, axis-invariance), and the wall-clock/FLOP compute axis. Not the gauge
math (geometer), not the ELBO internals (variational), not head-count representation
(transformer-ml).

## Steelman of the opposing position
The total-vs-active parameter decoupling is standard for conditional-computation models, so
lowering CE 124.6 -> 92.2 while the per-token active working set holds at ~12.06M is a genuine
efficiency axis complementary to width-scaling — and it achieves the grow-frontier CE on *half*
the data, if anything under-reporting its efficiency.

## My position (in service of red)
`blocks_K48` is not a well-posed parameter-efficient scaling axis in the Kaplan/Hoffmann sense,
on three independent ML-engineering grounds: (i) its fitted power-law exponent is not
axis-invariant and its n_params CI [0.07, 1.73] spans an order of magnitude, so it is a monotone
trend, not a scaling law; (ii) every cross-sweep comparison to `grow_K_GL10` is *non-identified*
because the two sweeps sit on different D-slices of `L(N,D)` (exactly 2x token difference), and the
one run that would remove the confound — `blocks_K48` at 491.52M tokens — was never done; (iii) on
the only honest compute axis (wall-time), the best-loss point GL24 is the *slowest* run, so the
axis is not compute-efficient. Each is individually sufficient to defeat "distinct scaling axis,
parameter-efficient, publishable as-is."

## Evidence

- **Hoffmann et al. 2022 (Chinchilla), "Training Compute-Optimal Large Language Models," Approach 3.**
  Fitted parametric loss: "**L̂(N,D) ≜ E + A/N^α + B/D^β**", with **E = 1.69, A = 406.4, B = 410.7,
  α = 0.34, β = 0.28**. Loss depends *jointly* on parameters N and tokens D. The data term `B/D^β`
  is strictly decreasing in D, so at fixed N, moving D from 245.76M to 491.52M lowers modeled loss
  by `B·D^{-β}(1 - 2^{-β})` — a strictly positive, first-order additive shift (with β=0.28,
  `1 - 2^{-0.28} ≈ 0.18`, i.e. ~18% of the data-term magnitude; the absolute constants are
  Chinchilla-specific and do not transfer to this V=50257, K=48, single-layer model, but the
  *form* guarantees the confound). Any cross-sweep comparison of absolute PPL, of the fitted floor
  E, or of "sits on the frontier" inherits this shift.

- **Hoffmann et al. 2022 (Chinchilla).** "model size and the number of training tokens should be
  increased in approximately equal proportions"; the compute-optimal ratio is ~20 tokens per
  parameter (Gopher-sized: 280B params optimal near ~5.9T tokens). `blocks_K48` GL24 sees
  245.76M / 70.16M ≈ **3.5 tokens per total-parameter**, roughly 6x below compute-optimal — deep in
  the data-limited corner where no compute-optimal or data-scaling exponent is claimable.

- **Kaplan et al. 2020, "Scaling Laws for Neural Language Models," §1.** `L(N) = (Nc/N)^{α_N}`,
  α_N ≈ 0.076, Nc ≈ 8.8×10^13 — stated "for models with a limited number of parameters, trained to
  convergence on **sufficiently large datasets**." The joint form `L(N,D) = [(Nc/N)^{α_N/α_D} + Dc/D]^{α_D}`
  makes D-dependence explicit. Direct caveat: "Performance improves predictably as long as we scale
  up N and D **in tandem, but enters a regime of diminishing returns if either N or D is held fixed**
  while the other increases." `blocks_K48` holds D fixed at 245.76M while N (via `phi_embed`, +n_gen)
  grows 3.62x — by Kaplan's own statement, the diminishing-returns / data-limited regime, so the
  fitted exponent is a fixed-data slice, not an infinite-data exponent.

- **Fedus, Zoph, Shazeer 2021 (Switch Transformer), §1.** They define a legitimate
  parameter-scaling axis as: "increase the *parameter count* while **keeping the floating point
  operations (FLOPs) per example constant**." A well-posed active-parameter axis requires FLOPs/token
  held constant *and* fixed data. `blocks_K48` violates both: the analytic FLOP proxy holds ~constant
  only because it is decode-dominated (1.03x), while the real transport sub-term `2·N·d_head² = 2·N·g²`
  grows 64x (g: 3->24) and empirical wall-time is U-shaped (6366 -> 4657 min -> 11049s), with the
  best-loss point GL24 the slowest run. On the honest compute axis the "efficiency" reverses.

## Newly-discovered canon (for 01b_extended_evidence.md)

- **Fedus, Zoph, Shazeer 2021, "Switch Transformers: Scaling to Trillion Parameter Models with
  Simple and Efficient Sparsity," JMLR 2022; arXiv:2101.03961, §1.**
  URL: https://arxiv.org/abs/2101.03961 — "Our hypothesis is that the parameter count, independent
  of total computation performed, is a separately important axis on which to scale... we investigate
  a fourth axis: increase the *parameter count* while keeping the floating point operations (FLOPs)
  per example constant." Establishes that a total-vs-active parameter efficiency claim is legitimate
  only when FLOPs/token are held constant — which the g²-growing transport cost and U-shaped
  wall-time of `blocks_K48` violate.

- **Kaplan et al. 2020, arXiv:2001.08361, §1.2 and §4 (finite-data / overfitting).**
  URL: https://arxiv.org/abs/2001.08361 — for limited-data early-stopped training the relevant law
  is `L(D) = (Dc/D)^{α_D}` (α_D ≈ 0.095, Dc ≈ 5.4×10^13 tokens), and the overfitting relation depends
  on the ratio `N^{0.74}/D`. Formalizes that a fixed-token sweep measures a D-conditioned slice, and
  that two sweeps at different D are not on the same law.

- **Hoffmann et al. 2022, arXiv:2203.15556, Approach 3 + Table A3.**
  URL: https://arxiv.org/abs/2203.15556 — the `E + A/N^α + B/D^β` decomposition with efficient-frontier
  exponents `N_opt ∝ C^{0.46}`, `D_opt ∝ C^{0.54}`. The additive-in-D floor term is what makes any
  cross-sweep floor-E comparison between the 245.76M-token and 491.52M-token sweeps non-identified.

## Falsification conditions
Red is wrong on ML-engineering grounds if: (X) a matched-D `blocks_K48` run at 491.52M tokens
preserves the GL3->GL24 CE improvement *and* the fitted exponent tightens to an axis-invariant value
with a CI that does not span an order of magnitude (say, width < 2x) across ≥2 resource axes — then
it is a genuine power law, not a monotone trend; or (Y) an iso-wall-clock (equal GPU-seconds)
comparison shows `blocks_K48` reaching strictly lower loss than `grow_K_GL10` at equal *empirical*
compute — then it is compute-efficient despite the g² transport cost; or (Z) the fixed-token
exponent is shown to coincide with the infinite-data exponent at these token counts (e.g. a
convergence check showing the models are near the data-independent regime), voiding the
data-limited-slice objection.

## Confidence
HIGH — the 2x-D non-identification, the CI [0.07, 1.73] exponent, and the U-shaped wall-time are all
in-pack, code-confirmed facts; only a matched-D 491.52M-token blocks run (which does not exist) or an
iso-wall-clock frontier crossing would move me.
