# Why the gauge-RoPE run collapsed: investigation, 2026-07-26

Four expert agents were dispatched against `pos_rotation='rope'` after the attention maps looked wrong
by eye. The run in question is `vfe3_runs/339.73_wikitext-103_K20_block_glk_linear_mix_s6`
(`pos_rotation='rope'`, `rope_base=100.0`, `rope_full_gauge=False`, `rope_on_value=False`,
`pos_phi='none'`, `e_step_update='gradient'`, `oracle_unroll_grad=False`), test PPL **339.73** against
138.40 / 139.97 / 141.38 for the three otherwise-comparable non-rope runs in the same directory.

The eyeball read was right, and the cause is not the rotary math. Two defects, both verified against
the saved checkpoint here rather than taken on report, account for the collapse. A third makes the
plotted map wrong independently of the model. A fourth is a genuine correctness bug in the rotary
construction that did not dominate this run only because the parameter it corrupts was frozen.

Verification status is marked per finding: **[verified here]** means reproduced against the run
artifacts in this session; **[agent-reported]** means the evidence came from a dispatched agent and was
not independently re-derived.

## R-1 (critical) The gauge frame table was frozen at random init for all 15000 steps

**[verified here]** `prior_bank.phi_embed` — the table that generates every transport operator in the
model — received no gradient for the entire run.

| run | `phi_embed` std | absmax | `gauge_invariant_spread` |
|---|---|---|---|
| rope, 339.73 | **0.05999918** | 0.324 | **0.326** |
| baseline, 138.40 | 0.23183507 | 1.937 | 6.961 |
| gradient, 141.38 | 0.22803190 | 1.793 | 7.458 |

The rope run's `phi_embed` standard deviation is `0.05999918` against a configured
`phi_scale = 0.06`, which is the initialization `phi_scale * torch.randn(...)` at
`vfe3/model/prior_bank.py:481`. It is the init draw, untouched. The two trained runs moved the same
table to std 0.23 and absmax ~1.9. The whole attention geometry ran on a frozen random frame.

The mechanism is a four-step chain, each link in executable code:

1. `rope_on_value=False` sets `decoupled_value_gauge = (pos_rotation == 'rope' and not rope_on_value)`
   (`vfe3/config.py:2481`).
2. That makes the config non-kernel-eligible, so the belief gradient routes to the autograd oracle
   rather than the closed-form kernel (`vfe3/gradients/kernels.py:307`).
3. The oracle returns a DETACHED tangent, because `create_graph = (oracle_unroll_grad and
   e_step_gradient == 'unroll')` (`vfe3/inference/e_step.py:1063`) and `oracle_unroll_grad=False`.
   `phi_embed` reaches the loss ONLY through the E-step tangent, so it receives nothing.
4. The rescue that exists for exactly this situation does not fire. `vfe3/config.py:2491` auto-enables
   `oracle_unroll_grad` when the config routes to the oracle — but it is gated on
   `_stateful_transport`, which is `False` for `transport_mode='flat'` because the flat registration
   has no `state_builder`.

The gate is the bug. Its own comment says "the belief gradient routes to the DETACHED oracle and
transport state would freeze; auto-enable the differentiable oracle there" — but it rescues only
*registry-owned transport state*, while `prior_bank.phi_embed` freezes by the identical mechanism on
the flat path and is not covered.

**[agent-reported]** corroboration from the checkpoint optimizer state:
`checkpoints/step_15000.pt` carries `optimizer_populated_slot_manifest.parameter_ids = [2,3,4,5,6]`;
the two comparable non-rope runs carry `[1,2,3,4,5,6,7]`. Group 1 is the `phi_group` built at
`vfe3/train.py:216,234` (`weight_decay=0.03 == cfg.phi_weight_decay`) and never received a gradient.
A direct probe on a tiny CPU model reproduced it:

```
pos_rotation=none                                     -> phi_embed.grad norm 2.366e-03  kernel route
rope + rope_on_value=True                             -> phi_embed.grad norm 3.554e-03  kernel route
rope + rope_on_value=False                            -> phi_embed.grad = None          oracle route
rope + rope_on_value=False + oracle_unroll_grad=True  -> phi_embed.grad norm 2.830e-03
```

**Fix.** Extend the `config.py:2491` predicate to fire whenever the config routes to the oracle AND
the gauge frame reaches the loss only through the E-step tangent, not only when the transport
registration is stateful.

**Workaround available today, no code change:** set `oracle_unroll_grad=True` whenever
`pos_rotation='rope'` with `rope_on_value=False`.

## R-2 (critical) 93.6% of pairwise energies pin at `kl_max`, so most of the attention map is the bare positional prior

**[verified here]** `pure_path_report.json`:

| run | `guard_energy_klmax_frac` |
|---|---|
| rope, 339.73 | **0.936248779296875** |
| baseline, 138.40 | 0.0 |
| 2-layer, 139.97 | 0.0 |
| gradient, 141.38 | 0.0 |

`safe_kl_clamp` is a hard clamp (`vfe3/families/base.py:26`, `kl.clamp(min=0.0, max=kl_max)`), so every
saturated pair carries the identical energy `160.0` and exactly zero gradient. Over that 94%,
`beta = softmax(log_prior - 160/tau)` reduces to `softmax(log_prior)` — the bare
`causal_alibi_noself` prior with NO content dependence whatsoever.

This is the direct answer to "the attention patterns look wrong": for 94% of pairs the plotted map is
not attention, it is the ALiBi distance prior. The remaining 6% is the only content-carrying part.

R-2 is best read as a consequence of R-1 rather than an independent defect. A frozen random gauge frame
produces transported means that bear no relation to the query, so the divergences blow past a `kl_max`
that the three trained runs never approach. Whether it fully resolves once the frame trains is an open
question and should be checked on the re-run, not assumed.

## R-3 (high) The plotted attention map is not the attention the model used

**[agent-reported, not independently verified]** `build_diagnostic_snapshot` computes `beta_maps` from
`diagnostic['layer_outputs']` (`vfe3/model/model.py:2591-2594`), which `vfe3/model/stack.py:146`
appends AFTER `vfe_block` has applied the head mixer and block norm. `attention_maps`
(`model.py:3193-3236`) scores at the same post-block point. At `n_e_steps=1` the model's single E-step
scored beta at the belief ENTERING the stack. Measured on this checkpoint:

```
entry belief (what the E-step scored) vs plotted:   max 1.0000   mean 0.0162
converged (post E-step, pre-mixer)    vs plotted:   max 0.2197   mean 0.0008
mu drift entry->converged 128.97, converged->out 83.48
```

A max difference of 1.0 means rows where the plotted map puts all mass on one key and the beta the
model actually used puts none there. Note the gap is dominated by the E-step displacement, not by the
head mixer. This is the same class of defect as B-01 and B-11 earlier today: a diagnostic reading the
right quantity at the wrong point in the pipeline.

**Fix.** Capture beta from inside `e_step_iteration` — the `state_record` seam already carries every
iterate — or plot both and label which is which.

## R-4 (high) The rotation is inserted on the wrong side of the learned frame

**[agent-reported by two independent experts, convergent]** The shipped operator is

```python
# vfe3/geometry/transport.py:323
rot = torch.einsum("...ikl,...ijlm,...jnm->...ijkn", rope, omega, rope)   # R_i Omega_ij R_j^T
```

which is the coboundary of the frame `W_i = R_i U_i`. The two rotations sit OUTSIDE the content
operator and therefore never meet. Writing `R_i Ω R_j^T = (R_i Ω R_i^T) R(θ_i - θ_j)`, the relative
factor is present but the transport is conjugated by the query's ABSOLUTE angle.

Two consequences, each measured by a different agent:

*Relative-position property lost.* With period-8 content, a truly relative scheme must satisfy
`E[i+8, j+8] == E[i, j]`. Measured contamination as a fraction of the energy's spread:

| `phi_scale` | mean abs dE / E-std | mean row TV(beta) |
|---|---|---|
| 0.0 | 0.000 | 0.000 |
| 0.02 | 0.072 | 0.035 |
| **0.06 (this run's frozen init)** | **0.212** | **0.101** |
| 0.2 | 0.563 | 0.257 |
| 0.5 | 0.605 | 0.277 |

*Gauge equivariance lost.* Under a structure-group gauge `g`, through the shipped `RopeTransport` and
`coupling_energy` seams:

```
no rope                              ||E - E^g||_inf = 1.907e-06   TV(beta) = 5.121e-08
SHIPPED   W_i = R_i U_i              ||E - E^g||_inf = 1.437e+01   TV(beta) = 2.802e-01
ALTERNATE W_i = U_i R_i              ||E - E^g||_inf = 2.861e-06   TV(beta) = 5.277e-08
```

Both experts propose the same structural correction from different directions: right-insert the
rotation so the operator is `Ω^rope_ij = U_i R(Δ) U_j^{-1}` instead of `R_i U_i U_j^{-1} R_j^T`. They
differ only on the sign of `Δ`, which is the same textbook-convention difference noted in R-6. The
right insertion is still a flat cocycle, still satisfies `Ω_ii = I`, still reduces to `R(θ_i - θ_j)` at
`U = I`, costs the same single einsum, and restores exact gauge equivariance.

It is also what the manuscript specifies: `Research/manuscripts/GL(K)_attention.tex:1425` says the
rotation is "inserted between query and key", which gives the cross term
`(U_i^T S^{-1} μ_i)^T R(Δ) (U_j^{-1} μ_j) = Q^T R(Δ) K`. The codebase's other positional mechanism
already right-inserts (`transport.py:1993-1994`, `exp_phi = exp_phi @ right_exp` under
`pos_phi_compose='group_product'`). Rope is the odd one out.

This did NOT dominate the 339.73 collapse, because R-1 froze `phi` at std 0.06 where the contamination
is 21% of energy spread rather than 60%. It will dominate once the frame actually trains.

**Consequence for the config warning.** `vfe3/config.py:1998-2015` asserts that gauge-RoPE "breaks
GLOBAL gauge equivariance for EVERY group" because `R(θ_i)` does not commute with a global gauge
element. The premise is true; the conclusion does not follow. Non-commutation matters only because `R`
was placed to the left of the gauge action — the right insertion is exactly equivariant with the same
non-commuting `R`. The warning records a composition-order artifact as a theorem and forecloses the
fix.

**Not tested.** Every rope composition test uses `Ω = I` (`tests/test_rope.py:36-62`), which is
precisely where left and right insertion coincide. The convention is pinned only against itself.

## R-5 (medium) Five frequency bands per head is too few, and `rope_base` cannot fix it

**[verified here]** `rope.py:84` computes `freq = base ** (-2.0 * k / d)` with `d` the irrep BLOCK
size, and the `for d in irrep_dims` loop restarts `k` at 0 in every block — so with
`irrep_dims = [10, 10]` **both heads receive an identical 5-band ladder**. This is what standard
multi-head RoPE does (LLaMA computes frequencies on `head_dim` and shares them across heads), so the
loop is conformant. The problem is that `d_head = 10` yields only 5 bands.

The content-averaged positional kernel `c(δ) = mean_k cos(δ ω_k)`:

| `rope_base` | c(2) | c(4) | c(8) | c(16) | c(32) | c(64) | range | mean |
|---|---|---|---|---|---|---|---|---|
| 100 (this run) | 0.64 | 0.42 | 0.20 | 0.13 | **0.49** | -0.02 | 1.51 | 0.03 |
| 1000 | 0.69 | 0.57 | 0.46 | 0.18 | 0.42 | 0.07 | 1.10 | 0.31 |
| 10000 | 0.71 | 0.63 | 0.63 | 0.23 | **0.77** | 0.31 | 1.05 | 0.40 |
| *real head, d=64, base=10000* | *0.88* | *0.75* | *0.70* | *0.61* | *0.61* | *0.43* | *0.63* | *0.50* |

At every base the kernel is non-monotone with `c(32) > c(16)`: a token 32 away reads as more
positionally similar than one 16 away, which is a bright off-diagonal band in the heatmap. Raising the
base improves the near field (c(8): 0.20 to 0.63) but makes the far-field revival WORSE (c(32): 0.49
to 0.77), because the `k=0` band is `base^0 = 1` at every base while the slower bands freeze out of
the context window and stop contributing to the averaging that damped it. A 32-band head decays
smoothly; a 5-band head cannot.

One agent recommended raising `rope_base` to 10000 as a cheap fix. **That recommendation is not
supported** — measured above, it trades one artifact for a larger one. The levers that actually add
bands are fewer heads (`n_heads=1` gives `d_head=20`, 10 bands), a larger `embed_dim`, or slicing one
full-`K` ladder across heads.

For contrast, VFE_2.0 normalizes by the full `K` (`transformer/core/transport_ops.py:79`,
`freqs = 1/(base**(arange(half_K)/half_K))` with `half_K = K//2 = 10`) and lays 10 contiguous bands
across the whole belief, so its head split hands head 0 the five fast bands and head 1 the five slow
ones (wavelengths 77 to 571). Head 1 has **no revival above 0.3 anywhere in 0-127** — a clean locality
kernel. That single smooth head is almost certainly what "the old one looked more sensible" refers to.
It is an accident of a non-standard normalization, not a correctness property.

## R-6 Cleanly verified as correct (recorded so they are not re-investigated)

- **Composition order and transpose placement.** `R_i Ω_ij R_j^T` is what the code builds, matching the
  docstring, to 0.000e+00; the inverted convention differs by 2.311. No transpose error.
- **Cocycle.** `Ω^rope_ii = I` to 5.96e-08 and `Ω^rope_ij Ω^rope_jk = Ω^rope_ik` over all `N^3` triples
  to 2.98e-07. The `R_j^T R_j` cancellation survives the factored, compact, and dense realizations.
- **Structure group.** For `block_glk` with `irrep_dims=[10,10]` the structure group is
  `GL(10) ⊕ GL(10)`, which contains `SO(10) ⊕ SO(10)` in full, so `R` is INSIDE it and `R Ω R^T` does
  not leave it. The absent `sp`/`so_n`/`sp_n` warning is correct behavior here.
- **Rope config wiring.** All nine `rope_on_value` default sites agree at `True`, all `rope_on_cov`
  sites at `False`, `rope_base` at `100.0`. E-07 is fixed with no third disagreeing site, and no
  un-forwarded hop exists across `vfe_block` / `vfe_stack` / `e_step` / `e_step_iteration` /
  `free_energy_value` / `build_belief_transport`.
- **The model channel is fully roped.** `_refine_s` forwards `rope`/`rope_on_cov`/`rope_on_value`
  (`model.py:931-933`) and `_gamma_energy` builds its transport with the same rotation
  (`model.py:1878-1880`), so the s-fiber and belief fiber transport identically.
  `share_refine_s_transport` is explicitly gated off when `rope is not None` (`model.py:1051`).
- **The figure path does not drop rope.** The diagnostic snapshot stores the same rope tensor object
  the model used (`model.py:1112`, `:2630`) and the plotted maps consume it (`model.py:2592`). The
  suspicion that the picture was un-roped while the model was roped is REFUTED. (The picture is still
  wrong, but for the different reason in R-3.)
- **`pos_phi='none'` does not degenerate the transport.** `_apply_pos_phi` early-returns
  (`model.py:725-726`) and `phi_i = phi_embed[token_id_i]` stays token-dependent, so
  `Ω^learned ≠ I`. Rope becomes the only POSITIONAL signal, not the only signal.
- **`rope_on_value=False` is the correct analogue of standard RoPE**, which rotates queries and keys
  but not values. `oracle.py:216-221` scores beta on the rotated energy while the belief descends the
  un-rotated base. Wired correctly; its only cost is forcing the oracle route — which is what triggers
  R-1.
- **Softmax axis, causal/no-self masking, per-block beta, temperature.** Row sums to 2.4e-07, zero
  future mass, zero self mass off row 0, `tau = kappa*sqrt(d_block) = 3.1623` dividing inside the
  exponent. No shape or broadcasting defect.

## What to do next

The experiment has not yet tested gauge-RoPE. It tested a frozen random gauge frame with a saturated
energy clamp. Three things in order:

1. **Re-run with `oracle_unroll_grad=True`.** One config field, no code change. This is the difference
   between "rope is bad" and "rope was never trained". Check `guard_energy_klmax_frac` in the new
   `pure_path_report.json` — if R-2 was downstream of R-1 it should fall back toward 0.
2. **Run the missing control.** There is no `pos_phi='none', pos_rotation='none'` run on the gradient
   route, so nothing currently separates "rope hurts" from "dropping the learned positional phi hurts".
   Without it the rope arm cannot be scored even after R-1 is fixed.
3. **Then decide on R-4.** The right-insertion change is small and restores exact gauge equivariance,
   but it changes the operator, so it belongs after the baseline is trustworthy — and it needs the
   `Ω != I` test that `tests/test_rope.py` currently lacks.

R-5 is a design question rather than a bug and should be settled last, on evidence: at `d_head = 10`
there is no `rope_base` that produces a monotone kernel.

---

## Follow-up, same day: R-1 confirmed by re-run, and it unmasked R-4

Re-running with `oracle_unroll_grad=True` and nothing else changed took the rope arm from **339.73 to
163.99** test PPL. Every predicted downstream effect resolved:

| `pos_rot` | `pos_phi` | `oracle_unroll_grad` | PPL | `phi_embed` std | `guard_energy_klmax_frac` | `gauge_invariant_spread` | wall (s) |
|---|---|---|---|---|---|---|---|
| none | learned | – | 138.40 | 0.232 | 0.000 | 6.96 | 1521 |
| none | learned | – | 141.38 | 0.228 | 0.000 | 7.46 | 1728 |
| rope | none | **True** | **163.99** | **0.236** | **0.000** | 1.95 | 1782 |
| rope | none | False | 339.73 | 0.060 | 0.936 | 0.33 | 1378 |

The gauge table trains (0.060 to 0.236, inside the 0.228-0.246 band of every healthy run) and R-2's
`kl_max` saturation went 0.936 to **0.000**, so it was entirely downstream of the freeze rather than an
independent defect. Cost of the flag is about **+3%** wall against the comparable gradient-route run
(1782 vs 1728 s); the frozen run's 1378 s was fast only because a whole parameter group was inert.

**R-4 measured on the trained checkpoint.** With the frame now at std 0.236 instead of pinned at 0.060,
the left-insertion contamination was re-measured directly rather than extrapolated. Periodic token
content with period 8, belief moments held constant across positions so every residual is positional; a
relative scheme must satisfy `E[i+P, j+P] == E[i, j]`:

| variant | mean abs dE | E std | ratio | max row TV(beta) |
|---|---|---|---|---|
| no rope (control) | 0.0000 | 1.7012 | 0.000 | 0.0000 |
| SHIPPED `R_i O R_j^T` | 2.1731 | 3.1337 | **0.693** | **0.3250** |
| PROPOSED `O R_i R_j^T` | 0.0000 | 3.2226 | 0.000 | 0.0000 |

The control reading exactly 0.0000 validates the harness; the proposed right insertion reaching exactly
0.0000 on the SAME trained weights proves the placement is the sole cause. 69% of the pair energy's
spread is absolute-position contamination, and up to a third of the attention mass in a row moves for
content that did not change — worse than the 56% the synthetic-phi probe projected at this frame size.

So the ordering is now established rather than conjectured: R-1 held the frame at 0.060 where R-4 was
mild (21%), and fixing R-1 scaled R-4 up to dominant (69%). The residual 163.99 vs 141.38 gap is R-4's
prime suspect, still confounded with `pos_phi='none'` until the control in step 2 below is run.
