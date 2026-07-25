# Merged fix order — audits of 2026-07-24 and 2026-07-25

Consolidates two independent audits into one deduplicated, ordered remediation list.

- **A1** = `docs/audits/audit-2026-07-24.md` (local, 13 investigators, 65 findings, verifier plus two
  adversarial duels). Audited `ad3a5ad` **plus** the uncommitted working-tree config edits.
- **A2** = `docs/audits/audit-2026-07-25-second-deeper.md` (PR #178, 7 lenses, 29 findings). Audited
  committed `ad3a5ad` **only**.

A2 ran blind to A1: A1's report was committed locally and never pushed, so the cloud checkout could
not read it. Where the two agree, that is independent convergence rather than one agent reading the
other, and those items are marked **CONVERGENT** and weighted up accordingly.

Provenance is given per item. Where the two audits assigned different severities, the higher is taken
and the reason noted. Nothing here has been fixed; this is the order, not the work.

## Quick win to do first

**F0. Delete the stale node ID at `check_audit_fixes.py:32`.**
Provenance: A1 H6, A2 M7. **CONVERGENT.** One line. `tests/test_belief_cache.py` was removed by commit
`53e72f1`, and because all 26 node IDs go to a single pytest invocation, that one missing file aborts
collection for the entire driver: `collected 0 items`, exit code 4, zero of the other 25 pinned
regression tests run, and the banner reads `FAILURES` rather than "stale reference". Fixing this
restores a verification driver that is currently 100% dead, so it should precede everything else that
wants to be checked by it.

## Tier 1 — corrupts the live training path or its correctness guarantees

**F1. Give the closed-form belief kernel the same float32 island the oracle already has.**
Provenance: A1 M7 (`audit-info-geometer`), A2 H1 (found independently by two of its lenses).
**CONVERGENT, and both audits' highest functional finding.** `vfe3/gradients/kernels.py:396-403`
calls the autocast-eligible transport contractions directly, while `vfe3/gradients/oracle.py:140-146`
guards itself with `torch.autocast(..., enabled=False)` and `.float()`. Under the active
`amp_dtype='bf16'` the two sides of a seam that golden tests pin to agree at float32 tolerance
diverge: A1 measured 3.82e-03 relative on `grad_mu`, A2 measured ~0.3% relative and, additionally,
that **the gradient-active pair mask flips on 22% of pairs**. Second-order consequences both audits
tie to this single fix: the structural self-pair energy stops being exactly zero (0.0 fp32 versus
6.542e-06 bf16), so `pair_mask` no longer gates the self-pair derivative; `reuse_pairwise_kl_stats`
silently self-disables because its gate requires all-float32 (A1 L2); and A2 additionally found an
fp16 reciprocal overflow at `kernels.py:153`/`:190` that the same island removes.
`mm_exact_update` at `kernels.py:528` needs the same treatment.
**`tests/test_amp.py` has no kernel-under-autocast test at all** — add one, or this regresses silently.

**F2. Keep the transport vertex exp/inverse pair at float32, and make its failure observable.**
Provenance: A1 H1 (`audit-numerical-analyst` and `audit-gauge-theorist`, convergent within A1),
A2 M3. Three distinct changes in one subsystem:

- *Storage.* `vfe3/geometry/transport.py:1766-1771` rounds the fp32 phi chart to the autocast dtype;
  `:1702` stores the exponential back at bf16; `:1572` rounds the float64 inverse back to bf16. The
  edge exponential already does this correctly (`_direct_link_edge_exp`, `:1084-1098`, coerces both
  input and output to fp32); match it. Note the precise mechanism: the vertex path *does* open an
  autocast-disabled island and upcast, so "it has no float32 island" is wrong — the defect is
  input/output rounding around a correct interior.
- *Detection.* `_checked_group_inverse`'s three guards check only finiteness, never
  `U U^{-1} ≈ I`. `group_element_inverse` already accepts a `residual_tol` parameter, validates it at
  `:1594-1596`, and then never uses it (A1 L23) — wire it up, and the one named guard against this
  becomes real.
- *Certificates.* `_freeze_tensor` (`vfe3/model/model.py:170-176`) upcasts bf16 to fp32 for every
  diagnostic snapshot, and `_val_diagnostics` (`vfe3/train.py:937-967`) carries no autocast wrapper,
  so `cocycle_residual`, `holonomy_deviation`, and `vertex_cond_*` all describe a transport the
  forward never used. A run logged `cocycle_residual = 1.30e-04` while the in-force value measures
  1.5e-02 to 7.9e-01. Compute the certificate inside the autocast context.
- *Clamp (A2 M3).* `transport.py:1342-1353` returns a surrogate that is not `exp(M)` with no warning
  and no error when both `transport_clamp_monitor` and `transport_chart_max_norm` are off, which is
  the committed default. A2 measured **98% relative error at `‖M‖_F = 30`**. Default
  `transport_chart_max_norm` to a finite bound below `TRANSPORT_CLAMP_MAX_NORM`, or warn
  unconditionally when `scale < 1`.

Severity note: A1's adversarial duel downgraded this from critical to high after the skeptic showed,
from real trained weights across 100,352 vertex factors, that >99% of blocks sit at cond < 130 where
the attention error is bf16-noise-sized with zero argmax changes. It stays high because the monitor is
blind, not because today's trajectory is broken.

**F3. Give the factored diagonal congruence a precision island and a nonnegativity guard.**
Provenance: A1 H2. `vfe3/geometry/transport.py:2259-2265` computes the compact diagonal congruence as
a two-stage quadratic form with mixed-sign intermediates, while the full-covariance sibling at
`:2082-2084` explicitly upcasts to float64. Measured minimum off-diagonal `sigma_t` reaches -1.51e+03
at cond 3.5e3 and -1.27e+07 at cond 7.1e4; **fp32 also breaks**, returning -4.52 at the clamp boundary
where the dense reference gives +4.00. A negative variance is detected nowhere and is silently mapped
to `clamp(min=eps)` = 1e-6 by `renyi_closed_form` (`families/gaussian.py:216`), inverting that key's
precision weight by about six orders of magnitude, pushing `E_ij` to `kl_max`, and zeroing the pair
gradient. Clamp at zero with a monitored violation count.
**Correction carried from A1's verifier:** the manifestly nonnegative form at `transport.py:2350-2352`
is **not** available as a drop-in fallback — it lives in the dense `_factored_diagonal_covariance`,
not in the compact function.

**F4. Bound `‖phi‖`, and bound it against the embedded norm.**
Provenance: A1 H3, A2 M4. Two complementary halves:
- *Nothing bounds it at all under the active M-step (A1).* `vfe3/train.py:745-749` gates the chart
  projection on `phi_mstep_max_matrix_norm is not None`, which is None; `gauge_optim.py:71-77`
  registers `"adamw"` with `requires_manifold_optimizer=False`, and all three phi passes skip groups
  without the pullback flag; the bounding retraction (`retract_phi`, `max_norm=5.0`) is reachable only
  from the E-step phi step and `e_phi_lr=0.00`. `TRANSPORT_CLAMP_MAX_NORM = 20.0` bounds the
  exponentiated operator, not the parameter.
- *The cap that exists measures the wrong thing (A2).* `retract_phi`'s cap bounds the **coordinate**
  norm rather than the embedded norm, so `tied_block_glk` at `n_heads >= 16` can silently reach F2's
  surrogate regime.
Fix both together: default `phi_mstep_max_matrix_norm` to the GL retraction's 5.0 under `adamw`, and
bound against the embedded norm. Empirical note from A1's duel: `phi_weight_decay=0.03` does produce
an equilibrium in archived runs (`weight_norm_phi` peaks at 1283 near step 3700, declining to 1053),
so this is an absent guard rather than observed divergence.

## Tier 2 — corrupts reported metrics, which invalidates analysis and figures

These do not change what the model learns. They change what you would conclude from it, which is why
they rank above registry hygiene.

**F5. Reconcile the gamma head reduction between diagnostics and the objective.**
Provenance: A1 L19 (rated low), A2 M1 (rated medium). **CONVERGENT — taking A2's medium**, because A2
established the consequence A1 missed: the error is written to the CSV. `_gamma_coupling_rows`
(`model.py:2007-2010`) reduces heads by `"sum"` for `diagnostics` (`:2870`) and by `"mean"` for the
scored objective (`:2050`), so the reported gamma block is `n_heads` times the scale that enters the
loss. The `s_e_step=True` path descends the sum form via the kernel contraction, so one
`lambda_gamma` means the canonical block at `s_e_step=True` and `1/H` of it at `s_e_step=False`; an
ablation toggling `s_e_step` at fixed `lambda_gamma` is not weight-matched.

**F6. Apply the `include_attention_entropy` gate to the reported gamma meta-entropy.**
Provenance: A1 M11, A2 M2. **CONVERGENT.** `model.py:2877-2885` folds `meta_entropy_rows` into the
reported `total` with no gate, while the scored objective gates it at `:2058-2061` and `_refine_s`
forwards the flag into the s E-step. Under the surrogate toggle the reported total, and every figure
fed by `gamma_meta_entropy` (`viz/figures.py:826-829`), carries a block neither the s E-step nor the
loss descends.

**F7. Fix the CSV free-energy columns so they mean what their names say.**
Provenance: A1 M12, A2 L2. Two related defects in the same output:
- `free_energy_total` is the inner alignment energy: `d["total"]` is assembled with
  `log_likelihood=None`, so the column omits `-E_q[log p(o|x)]` entirely
  (`train.py:973`, `:1691`; `metrics.py:1391-1407`), and `multiseed_analysis.py:587-588` reads it back
  under the other name. `free_energy_full_decomposition` warns and still returns an undercounting
  total at the active `lambda_h`/`lambda_gamma`.
- The per-term fields are raw and unweighted while `total` is weighted
  (`metrics.py:384-409`), so the logged columns do not sum to the logged total. A2 measured a naive
  column sum of 0.00828822 against a total of 0.00418519 at `lambda_beta=0.5` (98% relative), and 64%
  relative at `lambda_twohop=0.3`, with every other config agreeing to 7.2e-08. Emit the weights as
  columns or add `*_weighted` siblings, as was already done for `hyper_prior_weighted`.

**F8. Resolve the `log_diag` semantics in `covariance_from_packed`.**
Provenance: A2 M6 only — A1 did not find this. The field is a log Cholesky pivot, not a log-variance,
and the decode reference reads it as a variance, so **the encode and decode priors diverge as the
packed table trains**. Either have the decode table read the marginal diagonal of `L Lᵀ`, or rename
the field away from "log-variance".

## Tier 3 — pure-path and registry integrity, the repo's stated contract

Context: A1 enumerated all 26 registry seams and confirmed every seam's theoretically pure key is
reachable, with an all-pure config constructing under `-W error::UserWarning` with zero warnings and
running to the closed-form kernel. The items below are the exceptions to that result, plus the
call-site bypasses that violate "add a variant by writing-and-registering, never by editing call
sites."

**F9. `lambda_h_mode` bypasses the alpha registry its mirrored sibling uses.**
Provenance: A1 H4. `vfe3/lambda_h_i.py:35` holds a static `_LAMBDA_H_MODES` tuple and `:58-60` raises
`KeyError` from it, while `config.py:1532-1533` validates `lambda_alpha_mode` against
`tuple(sorted(_ALPHAS))`. Probed: one new form registered via `@register_alpha` is accepted by
`lambda_alpha_mode` and rejected by both the `lambda_h_mode` validator and the runtime dispatcher.
This is the only seam-level pure-path exception A1 found.

**F10. Family-key the `KL(s||r)` operands in the extractor.**
Provenance: A2 M5 only. `vfe3/viz/extract.py:1250-1254` hardcodes `DiagonalGaussian` and reads
`pb.r_sigma_log` directly, bypassing `r_parameters()`, while every sibling extraction
(`extract.py:830, 976, 1076, 1292`) plus `e_step.py:1414` and `model.py:1762` is family-keyed. Under
`family='gaussian_full'` with `s_e_step=True` this silently returns the wrong shape when `N == K`
(`(4,4)` where `(N,)` is expected) and raises otherwise. The config constructs without complaint.
**This does not contradict A1's pure-path result:** A1's table covers the training-path seams, and
this site is in the diagnostics/figures path. It does mean `gaussian_full` is reachable for training
but not for extraction.

**F11. Close the registry bypasses.** A cluster, cheapest first:
- `e_step_update` validation is a hardcoded literal desynced from `_E_STEP_UPDATE_ALIASES` and the
  public `canonical_e_step_update()` helper, which `e_step.py:879` and `extract.py:870` both use
  (A1 I-7 / A2 L16, **CONVERGENT**). `config.py:2477-2484`.
- The belief-gradient kernel registry is gated by a hardcoded family literal, conjoining
  `family == "gaussian_diagonal"` with `has_kernel(family)` and making the latter a tautology, so no
  other kernel is selectable and `register_kernel` is advertised but inert (A2 L14).
  `kernels.py:298-308`.
- The pair-stats reuse gate uses a hardcoded class whitelist with exact `type` rather than
  `isinstance`, 26 lines after the same module queries the registry correctly (A1 M19).
  `kernels.py:324`.
- Five sites hardcode `transport_mode == "flat"` where `TransportRegistration` carries no such flag,
  so a newly registered flat-equivalent connection silently loses the fused routes (A1 M19).
- `pos_phi_compose='group_product'` is a third mode of a registry-backed seam implemented purely by
  literal call-site branches, while `phi_retract_mode='group_product'` is rejected on the same shared
  registry (A1 M9).
- The `numerics.py` monitor registry has zero production consumers — decide whether the registry or
  the direct calls are the seam (A2 M8).
- Two metrics are registered under keys the only production selector never contains (A2 L13).
- `encode_mode="gauge_fixed"` passes registry validation and is then unconditionally rejected with
  `NotImplementedError` two lines later (A1 pure-path note, A2 L15, **CONVERGENT** — both audits
  independently identified it as the single unselectable key across eighteen registries).

**F12. Stop silently zeroing `phi_weight_decay`.**
Provenance: A1 H5. `train.py:216-218` sets `"weight_decay": cfg.phi_weight_decay` then calls
`phi_group.update(phi_group_metadata)`, and `gauge_optim.py:81` supplies `{"weight_decay": 0.0}`, so
`config.json` records 0.03 while the optimizer ran 0.0 and `phi_weight_decay` has no consumer in
`run_artifacts.py`. A1's verifier found this is **broader than first reported**, also hitting
`train.py:257-259` (`pos_phi_free`) and `:265-271` (`s_phi_embed`/`s_pos_phi_free`). Warn on the
override and record the effective per-group decay in artifacts.

## Tier 4 — latent traps and estimator gaps

Not live under the committed or current configuration, but they fire on a toggle flip, which
`CLAUDE.md` states is routine.

**F13. Name `phi_embed` in the freeze warnings, and patch the two unpatched ablation sweeps.**
Provenance: A1 M6, downgraded from high by adversarial duel. The oracle-route leg was **dropped**: it
is test-pinned at `tests/test_fullcov_alpha_roadmap_2026_06_13.py:123-145` and warned at
`e_step.py:957-967` with a one-toggle remedy. What survives: `e_step_gradient='straight_through'`
freezes `phi_embed` and `s_sigma_log_embed` with no warning under flat transport with
`learnable_r=False`, is not test-pinned, the enumerating warning at `config.py:2403-2409` omits
`phi_embed` while naming its sibling `s_phi_embed`, and the auto-enable coercion at
`config.py:2358-2374` skips it. Concretely: `ablation.py:883-903` (three `gaussian_full` arms) and
`:934-935` (the `unroll` versus `straight_through` sweep) carry no `oracle_unroll_grad` override while
`:845-848` and `:1063` do, so the hazard is patched per-arm from memory. Those two sweeps will run
with a 10,051,400-parameter frozen frame, and `ablation.py:1831` sums `numel()` — allocation, not
liveness — so the arms report as parameter-matched. Add a liveness check to the sweep harness;
`train.py:1983` already computes `dead_names` but `ablation.py` and `scaling.py` bypass the banner.

**F14. Add the inert-toggle warnings.**
Provenance: A1 M8, with specifics from A1 L12–L14 and M15. 24 of 26 probed dependent fields fire no
warning, including `mm_damping` under `e_step_update='gradient'`, `cocycle_relaxation` and
`link_alpha` under flat transport, `b0`/`c0` under `lambda_alpha_mode='constant'`, `rope_base` and
`rope_on_value` under `pos_rotation='none'`, `ema_decay` under `use_ema=False`, and
`share_refine_s_transport=True` under `e_phi_lr>0`. Named specifics worth their own guards:
`m_s_phi_lr` inert unless `s_frame_mode='phi_tilde'` yet printed in the run banner as a live LR;
`prior_handoff_rho`/`sigma` inert at `n_layers=1` because the blend is the last statement of the loop
body and nothing after it reads the result; `phi_precond_mode` inert under `e_phi_lr=0` plus `adamw`;
and `windowed`/`causal_windowed` **exactly** inert at `attention_window=128` with `max_seq_len=128`,
where `torch.equal(causal_windowed, causal)` is True.

**F15. Fix the phi-chart retraction axiom and the BCH accuracy bound.**
Provenance: A2 L9 and L8 only. These refine rather than contradict A1's clean negatives, which
verified the **SPD** retractions (`spd_affine`, `log_euclidean`) and the BCH **coefficients**:
- `mode="bch"` phi retraction does not satisfy `dR_phi(0) = id` in the coordinate chart. Measured
  relative `|dR(0)v − v|`: `euclidean` 1.9e-09 versus `bch` 7.1e-02 (glk K=3), 6.3e-02
  (`block_glk`), 5.1e-02 (`so_k`). Structural, since `BCH(X,Y) = X + dexp^{-1}_X(Y) + O(Y²)`.
- The order-4 BCH truncation is inaccurate over the frame domain the retraction itself permits, and
  the accuracy gate is default-OFF. In production `X` is the stored frame (bounded only by the
  retraction cap) and `Y` the trust step, so the governing term is `~‖X‖^{order+1}‖Y‖`, not the
  docstring's symmetric bound. At `‖X‖ = 5` the order sequence has stopped converging (o2 5.3e-3,
  o3 6.0e-3, o4 2.7e-3). A1 verified the coefficients correct by convergence slope at small `‖X‖`;
  both results are right, and A2 found the domain where the bound fails. Restate the docstring bound,
  or enable `bch_residual_max` by default when `phi_retract_mode="bch"`. Note this interacts with F4:
  bounding `‖phi‖` also bounds `‖X‖` here.

**F16. Decide the `lambda_twohop` contract.** Two distinct defects, both gated behind
`lambda_twohop > 0` (committed default 0.0, so neither is live):
- *Flatness (A1 M13).* `free_energy.py:460-470` reuses the direct edge energy as the composed two-step
  energy, an identity valid only for a flat connection, and `config.py:2553-2554` validates only
  nonnegativity with no `transport_mode` cross-check. Measured cocycle residual under `regime_ii` is
  3.19e+01 against 2.22e-16 for flat, with resulting energy error up to 126.2 on an `|E|` scale of
  42.96.
- *Potential (A2 L1).* `w2 = beta.detach() @ beta.detach()` drops the `dW2/dmu · E` term, so the
  descent direction is not the gradient of the reported `F`. Kernel and oracle agree with each other
  to 8.9e-16 (same convention) but both differ from central finite differences of the assembled scalar
  by 1.71e-1 on a gradient of scale 3.20 at `lambda_twohop = 0.3`, versus 1.9e-9 at 0.
Either reject `lambda_twohop > 0` for non-flat transport and drop the detach, or document both
limitations in the `free_energy` docstring.

**F17. Geometry guards on opt-in paths.** Provenance: A1 M4, M5, M14.
- EMA averaging moves a stored `omega_direct` gauge frame off the structure group (`ema.py:104`).
  Measured after 3000 steps: live `‖UᵀU − I‖` = 1.4e-14, shadow 2.5e-03, `det` 0.9975, trace form no
  longer preserved. The optimizer already carries this drift control for the same table
  (`gauge_optim.py:943`); the EMA path bypasses it, and `config.py` places no constraint on `use_ema`
  under `omega_direct`.
- `retract_log_euclidean`'s diagonal arm applies a sup-norm trust region before `step_size`, admitting
  a chart step `√K = 4.47×` larger than its own full arm and than `spd_affine`, 13.4× at
  `step_size = 3`, contradicting the validator's asserted equivalence at `config.py:1961-1968`.
- Polar re-orthogonalization gates on block **count**, so a single-block non-defining irrep tower is
  projected onto `O(K)` and off the structure group: polar restores orthogonality to 1.6e-15 but
  leaves the element 2.0e-3 off `rho(SO(3))`, then clears the dirty flag. The comment at
  `gauge_optim.py:918-921` states the correct intent; the guard does not implement it.

**F18. Remaining latent hazards.** Provenance: A2 L6, L7, L10, L11, L12; A1 L20.
- Below the `eps` floor the kernel keeps the sigma derivative while the oracle zeroes it
  (`kernels.py:152`), a second kernel/oracle divergence independent of F1.
- The gamma prior fold turns a zero-mixture-mass row into an exactly uniform row with no NaN and no
  warning (`model.py:2367-2370`), which is also A1 L20's `log(1e-12)` floor reintroducing what the m8
  fix removed. Build the mixture in log space with `logaddexp`.
- Under `causal_noself`, query 0 places all beta on `(0,0)`, exactly the `E_ii ≈ 0` structural sink
  the prior exists to remove, so the saturation mask kills its belief coupling entirely.
- `_apply_reflection` has no `CompactFactoredTransport` branch and falls through to `.clone()`, which
  that dataclass lacks. **Both audits independently confirmed it is unreachable today** because every
  call site gates compactness on `reflection is None`; raise an explicit `TypeError` instead of
  leaving the fallthrough.
- `_omega_retract_cayley` NaNs in double-backward at an exactly-zero algebra step because its
  Frobenius norm is not wrapped in `no_grad()`, unlike the structurally identical clamp in
  `stable_matrix_exp_pair` whose comment states the reason verbatim.

## Tier 5 — tests, tooling, and documentation that actively misleads

**F19. Stop tests pinning live config values, and bound their cost.** Provenance: A1.
`tests/test_ablation_sweep_route_compatibility_20260711.py:10` and
`tests/test_2026_07_15_driver_reliability_remediation.py:1515` assert literal toggle **values**, so
they go red on every flip and the second aborts before reaching the arm-construction coverage it
exists to provide. Assert instead that every arm constructs and that
`ablation.BASELINE_CONFIG[k] == train_vfe3.config[k]` for shared keys. Separately,
`tests/test_train.py:481` shrinks `max_seq_len`, `batch_size`, and `n_layers` but not `embed_dim` or
`vocab_size`, so its cost is unbounded in the live `embed_dim`; and four files build real models above
the mandatory `K < 6` limit (`test_fp16_gradscaler.py:87` at 10, `test_head_mixer_isotypic.py:88` at
8, `test_cg.py:261` and `test_audit_fixes_2026_06_10.py:21` at 9). None currently hangs, and the
`so_n`/`sp_n` cases are structurally forced because an `l0+l1+l2` tower has block dims `1+3+5 = 9` —
record that in a comment so a future reader does not "fix" them.

**F20. Make `record_property` evidence reach the XML.** Provenance: A1 L7. `junit_family` is unset, so
it defaults to `xunit2`, which is incompatible with `record_property`:
`PytestWarning: record_property is incompatible with junit_family 'xunit2'`. The recorded float64
residuals and chart norms therefore never reach the JUnit XML that the project's own verification
discipline treats as the only admissible source. Set `junit_family = "xunit1"`.

**F21. Fix the comments and docstrings that actively misdescribe the code.** Provenance: A1 L26, A2
L3–L5. These matter more than ordinary staleness because each would mislead someone deciding whether
a toggle is safe:
- `config.py:148` says the clamp is `max_norm=15`; `transport.py:1257` sets 20.0 and the emitted
  warning says 20.0.
- `groups.py:40` documents `skew_symmetric` as an `exp(-M) = exp(M)ᵀ` fast path. **No such path
  exists** — both audits confirmed independently that every inverse routes through
  `_checked_group_inverse`, and `group_element_inverse` explicitly discards the group before inverting.
  A2 verified behaviorally: `group_element_inverse(U, so_k)` returns bit-identical tensors at
  `residual_tol` 1e-12 and 1e2 while `|Uᵀ − U^{-1}| = 2.38e-07`.
- `gauge_optim.py:574-576` justifies `omega_reorth_every` by that nonexistent transpose path. The
  mechanism is legitimate (it keeps stored `U` inside `O(K)`); only the stated reason is wrong.
- `attention_prior.py:230-232`, `:256-257` claim `window` is not threaded from the model, while
  `model.py:641-643` threads it; the header lists 6 priors where 8 are registered.
- Signature argument-order violations at `retraction.py:708`, `run_artifacts.py:1423`,
  `model.py:2280`, `:2319`; about 14 UK spellings while the anti-UK test guards only
  `vfe3/viz/report.py`.

**F22. Retire the confirmed-dead surface.** Provenance: A1 L5/L6/L26, A2 L17/L18/L19/L20. Report only,
per `CLAUDE.md`'s instruction to mention rather than delete pre-existing dead code:
`config_from_serialized` and `consumed_retired_keys` (no production consumer);
`GaugeManifoldAdamW.load_state_dict`'s two legacy branches (unreachable — the validator rejects all 54
on-disk bundles first); `kernels.py:130`'s `pair_mask=None` path (both production callers always pass
a mask); `head_mixer.mixer_delta` (test-only, 0 of 54 bundles carry the old key);
`train.py:2028 run_training` (docstring says DEPRECATED, consumers are two tests);
`CONTROLLED_MAX_TOKENS` (declared cap, never enforced or persisted);
`tests/test_extract_forward_fidelity.py:46-58` (replays `connection_W/M/L=` while all three production
sites pass `transport_state=`, drifted at `1de67f8`); unused imports at `scaling.py:65`,
`check_gpu_tests.py:31`, `verification_gate.py:15`; duplicate re-imports at
`tests/test_free_energy.py:103`, `tests/test_phi_retraction.py:129`.

## One conflict between the two audits, unresolved

**`merge_legacy_transport_state` and the `connection_W/M/L` kwargs.** A1's deprecation lane
adjudicated `transport.py:485` as **live**, reporting that all nine internal call sites pass the named
kwargs, and explicitly listed it as "mislabeled but live — do not remove". A2's L17 ran an AST scan
over every call to `vfe_stack`/`vfe_block`/`e_step`/`e_step_iteration`/`merge_legacy_transport_state`
across `vfe3/`, the drivers, and `tests/`, found zero sites supplying both `transport_state=` and any
`connection_W/M/L=`, and concluded the duplicate-provision `ValueError` is unreachable and the legacy
kwargs are production-dead.

A2 is probably right: its L18 independently found a test that drifted at commit `1de67f8`
("refactor: register trainable transport state") precisely because production moved to
`transport_state=`, and that migration would explain A1 reading the older shape. But the two claims
are not strictly the same proposition — A1 asserts the kwargs *are passed*, A2 asserts *no site passes
both forms at once* — so they could both be true with the guard still dead. One grep settles it, and
it should be settled before anyone touches that function.

## What both audits independently cleared

Recorded so nobody re-spends effort here. A2's four prioritized hypotheses were all refuted against
committed main, and all four match A1's independently obtained clean negatives:

1. The Renyi kernel has an explicit `|alpha − 1| < 1e-6` KL branch in every divergence path plus a
   float64 cancellation band; it never divides by a near-zero. A1 additionally verified the closed form
   against 4M-point float64 quadrature to 8.9e-16 and confirmed limit continuity.
2. `Omega^{-1}` is a true float64 inverse everywhere it matters, never a transpose — except for RoPE,
   where the operator is a rotation and the transpose genuinely is the inverse. The covariance
   congruence correctly uses the transpose. A1 read every `transpose` in four geometry modules to reach
   the same conclusion.
3. `causal_alibi_noself` does not produce a fully masked row 0: the executable line retains `(0,0)`.
   A1 measured no NaN, no silent uniform renormalization, consistency across all four consumers, and
   bitwise causality under perturbation.
4. **The bf16 `eps` premise is arithmetically false** — bfloat16 carries float32's exponent range, so
   an absolute clamp floor binds correctly. This was an a-priori concern raised in the brief for both
   audits and independently refuted by both.

Also verified correct by both, at least in the overlapping region: every BCH coefficient and
truncation order (symbolically); cocycle composition and identity holonomy for every registered
transport; gauge equivariance for general-linear as well as orthogonal gauges; KL argument order at
roughly 30 call sites; the Fisher factor of two in both parameterizations; and that the closed-form
gradients are the exact gradient of the canonical `F` the oracle differentiates (A2:
`max |kernel − oracle| ≤ 9e-16` in float64, `≤ 4e-15` at second order; A1: 1.19e-07 in fp32 across
three alpha modes, a binding `kl_max`, unequal irrep dims, and both gradient routes).

A1 additionally cleared the security surface (deserialization, injection, path traversal, secrets,
error swallowing), the no-neural-network constraint (zero call sites; all grep hits are comments), the
absence of CLI parsing, determinism, and the mechanical deprecation layer (zero hits for NumPy 2.x
removals, removed torch APIs, and Python 3.12+ removals, which on numpy 2.4.4 and Python 3.14 would be
crashes rather than warnings).

## Test state

| Environment | tests | failures | errors | skipped | Notes |
|---|---|---|---|---|---|
| A1, pinned local env, clean `HEAD` worktree | 4217 | 0 | 0 | 28 | exit 0, green |
| A1, pinned local env, working tree with WIP edits | 4217 | 9 | 0 | 28 | all 9 traced to the uncommitted config edits |
| A2, cloud container | 4217 | 8 | 0 | 9 | container started with **no Python packages**; A2 installed torch 2.13.0+cu130, not the pinned build |

A2's eight failures are six environmental (three proxy-blocked GPT-2 BPE downloads, a missing CJK
glyph, a missing `powershell`, no NVIDIA driver) and two torch-version-sensitive goldens. **Your pinned
environment is green at 4217/4217.** Re-running A2's two golden failures on the pinned build is an open
obligation.

## Open obligations carried forward

1. **No bf16 training run has ever been logged.** Every archived run used `amp_dtype=None`, so the
   bf16-trained `‖phi‖` equilibrium and the in-force cocycle residual are unmeasured. Both of A1's duel
   agents named this independently. One bf16 run logging `vertex_cond_max` and an in-autocast cocycle
   residual would settle F2's severity either way.
2. **No CUDA path was exercised by either audit.** A1's environment is `torch 2.11.0+cpu`; A2's
   container had no driver. All measurements are CPU.
3. **A2's two torch-version goldens** need re-running on the pinned build.
4. **The `merge_legacy_transport_state` conflict** above.
