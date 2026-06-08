# VFE_3.0 lifecycle audit — multi-agent sweep (2026-06-07)

Full initialization-to-finish audit of the VFE_3.0 language model, run as a 16-expert
multi-agent workflow over the model lifecycle (init → encode → E-step → free-energy/attention
→ decode → M-step → geometry → diagnostics → future buildout), checked against
`Participatory_it_from_bit.tex` (PIFB, primary) and `GL(K)_attention.tex` /
`GL(K)_supplementary.tex` (0-D LLM-specific, secondary). Every high/critical finding was then
re-litigated by an adversarial skeptic + defender + judge.

**Method.** Each lens read the actual code (not comments) and the manuscripts, and returned
structured findings (title / location / severity / evidence / manuscript-ref / fix). Findings
were deduped/canonicalized by a triage pass, then the high/critical set was verified
adversarially. Audit rule applied throughout (per CLAUDE.md): a *theoretical-impurity* finding
counts only if **no toggle setting recovers the pure/canonical path** — a non-pure default is not
a finding.

**Coverage.** 16/16 lenses reported. 57 raw → **41 deduped findings**; **6 verified**.
`belief-gradient-kernel-oracle` returned a clean negative (kernel↔oracle envelope agreement holds).
The sweep was rate-limited by Anthropic-side throughput throttling and ultimately completed in
waves of 4 with automatic retries.

## Headline

**No confirmed high/critical correctness bug survives on the default / pure path.** After
adversarial verification, the four "high" candidates all collapsed: one **refuted**, three
**downgraded to medium**, plus one medium **confirmed** and one **downgraded to low**. Every
genuinely-wrong behavior is gated behind a *non-default opt-in toggle* (often two at once) or is a
*diagnostic-only misreport* — the trained default-config model is sound. The dominant theme is
**silent footguns on opt-in paths**: learnable parameters that freeze without warning, geometry
that truncates without error, and diagnostics that under-report the objective.

---

## 1. Verified findings (adversarial skeptic + defender + judge)

| # | Finding | Filed | **Verdict** | Final | Status (2026-06-08) |
|---|---------|:---:|:---:|:---:|:---:|
| V1 | ALiBi prior is symmetric and replaces the causal mask → "future-token leak" | high | **REFUTED** | low | n/a (refuted) |
| V2 | Cross-coupled gauge basis never closed under the bracket (`close_basis` dead) | high | **severity-adjusted** | medium | ✅ **fixed** |
| V3 | Oracle E-step severs `log_alpha`/`connection_W`/`log_lambda_beta` under `unroll`, un-warned | high | **severity-adjusted** | medium | ✅ **fixed** |
| V4 | `pos_phi='learned'` (default) silently freezes under `straight_through`/`detach` | high | **severity-adjusted** | medium | ✅ **fixed** |
| V5 | `killing`/`killing_per_block` are not no-ops in the gauge natural-grad M-step | medium | **severity-adjusted** | low | open (doc-only, low) |
| V6 | `free_energy_terms` omits `R(alpha)` and ignores `include_attention_entropy` | medium | **CONFIRMED** | medium | ✅ **fixed** |

**Status update (2026-06-08).** Re-verification against the live tree closes the actionable verified
set. **V2** — a `close_basis` config field (`config.py:100`, AUTO-True when `cross_couplings` is set)
forwards into `build_group` (`model.py:62-64`) and closes the basis via `geometry/closure.py`
(`tests/test_fix_gauge_audit.py`). **V3**/**V4** — the freeze-warning predicate now also covers the
`unroll`+oracle route and `pos_phi='learned'` (`config.py:736-766`), and the pos_phi warning fires on
the *effective* estimator (`model.py:204-214`). **V6** — `free_energy_terms` now threads `alpha_reg` and
`include_attention_entropy` (`metrics.py:104-150`), wired from `model.py:721`
(`tests/test_fix_metrics_audit.py`, `test_fix_model_audit.py`). **V5** remains open (low, doc-only);
**V1** needed no fix. Findings are annotated resolved rather than deleted to preserve the forensic record.

**V1 — ALiBi "leak" — REFUTED (→ low).** The mechanism is real (`prior_alibi` returns a symmetric
`-slope*|i-j|`, `attention_prior.py:81`, no `-inf` mask) but the headline is false for the operative
path: the wired default is `attention_prior="causal"` (`config.py:187`, `train_vfe3.py:131`), a correct
`-inf` causal mask (`attention_prior.py:49-64`), so the trained model never leaks. Non-causality is not
alibi-specific (`uniform` leaks identically), so selecting `alibi` is config misuse, not a path defect,
and the default ablation sweep does **not** auto-exercise it (`SWEEP_ORDER` omits `attention_prior`,
`ablation.py:575-615`). The pure causal path exists. *Surviving residual (low):* no `causal_alibi` prior
is registered, and the manuscript mislabels its symmetric bias as Press et al. 2022 (whose ALiBi rides
on causal masking with **per-head** slopes). Track as a missing-variant note.

**V2 — cross-coupled basis not bracket-closed — medium.** Confirmed from source: `build_group`
forwards only `cross_couplings`, never `close_basis` (`model.py:54`); `close_basis` defaults `False`
with no config field (`groups.py:93,111`); `_structure_constants` and `compose_bch` both silently
project out-of-span terms (`phi_preconditioner.py:237-238`, `lie_ops.py:158`). The judge's executed
probe scoped the blast radius: closure is **topological** — `[(0,1)]` and the symmetric pair
`[(0,1),(1,0)]` are *already* exact subalgebras (0 generators added, nothing truncated); the defect
only bites for **3+-head chains** like `[(0,1),(1,2)]` (off-span fraction 1.0, BCH rel-err 1.3e-2 raw
vs 3.4e-8 closed). Silent wrong geometry (pullback metric + BCH composition under the default
`pos_phi_compose='bch'`), but narrow opt-in × chain-topology gated, and forward energy/loss is
unaffected (Ω∈GL(K), KL gauge-invariant). **Fix:** add a `close_basis` config field auto-True when
`cross_couplings` is set (or always `close_under_brackets`), and assert span-closure residual in
`_structure_constants` / warn in `compose_bch`.

**V3 — oracle freezes learnable E-step params under `unroll` — medium.** Confirmed by probe: the
non-kernel families route to the autograd oracle, which returns *detached* gradients unless
`create_graph` is set, and `create_graph = oracle_unroll_grad and unroll` with `oracle_unroll_grad`
default-off (`oracle.py:87,118`, `e_step.py:357`). The freeze-warning (`config.py:631`) keys only on
`straight_through`/`detach`, so this `unroll`+oracle route is un-warned. **Scope correction:** the
default config trains all three params *live via the kernel* (the kernel gate `kernels.py:179-186` has
no `alpha_mode`/`transport_mode`/`learnable_lambda_beta` term), so the freeze needs a **second**
off-default deviation (`smoothing`, `alpha_div≠1`, or `gaussian_full`); `regime_ii` *alone* does **not**
freeze `connection_W`. Real silent-freeze + missing-warning on a sanctioned hand-edit, no default
impact. **Fix:** extend the `config.py:631` warning to the `unroll`+oracle+E-step-only-learnable case,
or auto-enable `create_graph` for the diagonal/smoothing oracle.

**V4 — `pos_phi='learned'` freezes under string estimators — medium.** Confirmed: the freeze-warning
predicate (`config.py:631-635`) omits `pos_phi`, which **defaults to `'learned'`** (`config.py:97`).
`pos_phi_free` reaches the loss only through the mu/sigma E-step tangent (the phi step is itself
detached; `out.phi` is loss-connected only when `mass_phi>0`, default 0), so `straight_through`/`detach`
severs its sole route. A pos_phi-*specific* freeze warning already exists (`model.py:193`) but is gated
on the `detach_e_step` **bool**, so the equivalent `e_step_gradient` string routes bypass it. Default-ON
parameter, uniquely unwarned; pure path (`pos_phi='none'`) exists and the default `unroll` estimator
trains it. **Fix:** add `pos_phi=='learned'` to the `config.py:631` predicate and/or fire `model.py:193`
on the *effective* estimator.

**V5 — `killing` not a no-op in the gauge natural-grad M-step — low.** The config doc
(`config.py:261`: "only the non-conformal pullback metric … changes the trajectory") is **false** in the
natural-grad branch: the gauge group is stepped raw (`gauge_optim.py:97-98`, then `p.grad=None` so AdamW
never normalizes it), and `precondition_phi_gradient` runs for every mode. But the judge's probe shows
the Killing metric is *exactly* `0.0625·I` (block_glk K=8) — a **direction-preserving global scalar**,
i.e. mode `none` with `m_phi_lr×16`. The code is mathematically correct (a conformal natural-grad step
*should* rescale by the conformal factor); only the doc is wrong, on a non-default opt-in path. **Fix:**
correct `config.py:261` / `gauge_optim.py:11-14` / `train.py:134`, or fold the conformal scalar into the
effective LR.

**V6 — `free_energy_terms` misreports the logged F — CONFIRMED medium.** `metrics.free_energy_terms`
(`metrics.py:124,132`) computes `self_coupling=(alpha*self_div).sum()` with **no `R(alpha)` term** and
**unconditionally** adds the attention entropy, with no `include_attention_entropy` parameter; `model.py:688`
discards the regularizer (`alpha, _ = ...`). Under the active `alpha_mode='state_dependent_per_coord'`
(`train_vfe3.py:109`), `R(alpha)=b0·alpha − c0·log(alpha)` is nonzero and *part of* the descended F
(`e_step.py:219,225` → `free_energy.py:289-291`). The reg term can exceed the `alpha·D` term it
accompanies and is nonlinear in `D`, so a monotone-decreasing logged total can mask a non-monotone true
self-coupling — defeating the console line, `metrics.csv free_energy_total`, and the descent figure.
Diagnostic-only (the objective itself is correct). **Fix:** thread `alpha_reg` into `free_energy_terms`
and add an `include_attention_entropy` flag, both passed from `model.diagnostics`.

---

## 2. Unverified findings by category (medium / low / info)

### Correctness / numerical (opt-in or latent)

- **diagnostics/attention_maps report β at the post-E-step belief, not the β the forward used** —
  `model.py:662-697, 765-785`. At `n_e_steps=1` one iteration is not a fixed point; logged
  `attn_entropy` and saved heatmaps correspond to a belief the decode never read (max|Δβ|≈0.17). The
  "attention at the fixed point" docstring is false at T=1. *(medium)*
- **`amp_dtype='fp16'` accepted but no `GradScaler`** — `config.py:664`, `train.py:253-271`. Selecting
  fp16 trains with unscaled fp16 gradients through the deep unrolled E-step → underflow. Pure path
  (fp32/bf16) intact. **Fix:** raise on `'fp16'` in `__post_init__` until a scaler is wired. *(medium)*
- **ALiBi slope identical across heads** — `attention_prior.py:81`, `free_energy.py:211`. A single
  `(N,N)` bias broadcasts over all heads; Press et al. use distinct per-head slopes. *(medium)*
- **`_amp_context` bare `else` → fp16** — `model.py:273`. Any non-`'bf16'` value falls through to fp16;
  a future/typo'd `amp_dtype` would silently autocast fp16. **Fix:** explicit dict/elif + raise. *(low)*
- **`stable_matrix_exp_pair` Frobenius clamp is silent** — `transport.py:230-232`. For ‖M‖_F>15 it
  returns `exp(15·M/‖M‖)`, not `exp(M)`, with no activation monitor; inert on the flat path, reachable
  under drifted φ / large `regime_ii` δ. *(low, numerical)*
- **`oracle_unroll_grad=True` → NaN `connection_W` grad on diagonal `regime_ii`+`smoothing`** —
  `config.py:241,246-249`. The double-backward NaN the caveat scopes to full-cov is reachable on a
  *diagonal* path; the opt-in fix for the V3 freeze yields NaN here. *(low, numerical)*
- **`grad_accum` equal-token assumption unenforced** — `train.py:255-268`. Bias under uneven counted
  tokens; not reachable on the current (unpadded) dataloader, latent for a masked corpus. *(low)*
- **`gauge_trace_spread` ≡ 0 on `so_k`/`sp`** — `model.py:710`. Traceless generators → the logged/plotted
  gauge-volume is a flat zero on 2/5 groups; the group-dispatched fix `group_gauge_invariant`
  (`metrics.py:352-376`) exists but is never wired in. *(low; active `block_glk` unaffected)*
- **`retract_logeuclidean_full` docstring says `[eps, sigma_max^2]`, code clamps `[eps, sigma_max]`** —
  `retraction.py:231,264`. Code is the correct/consistent one; stale docstring. *(low, other)*

### Dead toggles

- **ALiBi `slope` unreachable from config** — `attention_prior.py:73`, `model.py:233`. Pinned at 1.0 on
  the training path; the `**kwargs` seam has no config feeder. *(medium)*
- **`encode_mode`/`decode_mode` validated against stale hardcoded literals** — `config.py:20-21,559,574`.
  Every other seam validates against its live registry; a newly-registered decode kernel would be
  rejected at construction until the literal is also edited — breaking add-by-registering. *(medium,
  modularity)*
- **`gauge_parameterization` is inert** — `config.py:19,353,367`. Only `'phi'` is reachable and nothing
  dispatches on it; `'omega_direct'` is soundly rejected (no belief source for a non-exp GL(K) element).
  `transport.py:3-5` docstring overstates `omega_direct` as usable. *(low; pure `phi` path is the
  behavior)*
- **Metric registry (`register_metric`/`compute_metrics`) has zero live callers** — `metrics.py:838,861`.
  The dispatch layer is dead; its `holonomy_deviation` entry wraps the *biased* deterministic estimator
  the live path (`model.py:709`) deliberately avoids. *(low)*
- **`tau`/`tau_gamma` config properties have no live consumer** — `config.py:666-688`. Active temperature
  is always `attention_tau(kappa, irrep_dims)`; the `tau_gamma` docstring falsely claims it feeds the
  gamma block. *(low)*
- **`generate_glk(include_identity=False)` sl(K) branch unreachable** — `generators.py:49,72`. det-control
  is done by `project_phi_to_slk` on the full gl(K) basis instead; correct-if-reached. *(info)*

### Theoretical impurity (pure path exists unless noted)

- **Pullback preconditioner metric ≠ manuscript Eq 2714** — `phi_preconditioner.py:284-287`. Code computes
  the *embedding* pullback `⟨Ψ(ad_φ)T_a·exp(φ), …⟩_F` (trailing `exp(φ)`); PIFB Eq 2714 is the
  right-invariant form *without* it. They agree on compact/skew φ (4.4e-16) but diverge on non-compact
  gl(2) (max-abs 21.7 at ‖φ‖=2) — the exact regime the pullback is introduced for. The manuscript is
  internally ambiguous (prose "through d exp" = code; Eq 2714 formula ≠ code). **Author to reconcile.**
  *(medium)*
- **`m_phi_natural_grad=True` + default `phi_precond_mode='none'` = momentum-SGD, not a geometric step** —
  `train.py:138`, `config.py:198`. The advertised geometric M-step needs *both* toggles +
  `pullback_per_block`. Pure path exists; footgun. **Fix:** warn, or auto-select `pullback_per_block`.
  *(low)*
- **M-step self-coupling prior fold exact only at `n_layers=1`** — `model.py:443-445`. Folds one converged
  belief instead of each block's intermediate; exact at L=1 (default) or `rho=0`. *(low, documented)*
- **`lambda_h`/`gamma` use per-token `.mean()`, not the manuscript sum-over-agents** — `model.py:480-485,
  535-536`. Rescales the coupling by `1/(B·N)`; internally consistent with the mean CE, but no toggle
  recovers the canonical sum weight. *(low)*
- **`renyi_per_coord` per-coordinate clamp ≠ summed closed form when saturating** — `gaussian.py:108,146`.
  Per-coord clamp is the *correct* behavior for the per-coord alpha consumer; the docstring identity holds
  only unclamped. *(low, doc imprecision)*
- **Aggregate `holonomy_deviation` is a faithful flatness certificate but loses per-head attribution** —
  `metrics.py:574`. Manuscript-sanctioned (PIFB:886); only matters under `regime_ii`. *(info)*

### Clean negatives (confirmed correct)

- **EM separation is clean** — `e_step.py:282-451` vs `model.py:412-536`. The E-step is blind to targets,
  `s`, `gamma`, `lambda_h`; all extra channels are assembled post-E-step at the loss level. No label
  leakage. *(info)*
- **Forward gauge-fixing is intended** — `model.py:313-397`. The frame-anchored prior bank + decode fix
  the gauge; the live requirement (transport *covariance*) holds (cocycle err ≈6.7e-7, gauge law
  verified). A global-invariance test of `forward()` is engineered to fail by design (residual = global-
  diagonal stabilizer, PIFB:1227). *(info)*
- **kernel↔oracle envelope agreement** — `belief-gradient-kernel-oracle` lens returned no findings.

---

## 3. Future buildout roadmap (prioritized)

The user explicitly asked for features to plan/build. Ordered by leverage.

**Tier 1 — completes the manuscript's core theory (the hierarchy).**
1. **s-channel E-step + un-detached s→q coupling.** Today the model channel `s_i` is a detached
   training-loss regularizer (`model.py:518`, `tilde_Omega := out.phi.detach()`); it never enters the
   belief E-step (grep of `vfe3/inference` for `s`/`gamma`/`lambda_h` = 0). Build a slow-channel `s_i`
   update in `e_step.py`, an s-channel gauge frame (`s_phi` table + its own transport seam for a distinct
   `tilde_Omega`), and let `s` feed `p_i` grad-connected behind a new toggle (keep the inert default).
   PIFB eq:pointwise_free_energy (1241-1249); GL(K)_supp 1083-1088.
2. **Meta-agent / scale-(s+1) hierarchy → un-freeze `r`.** `r_mu`/`r_sigma_log` is a hardcoded frozen
   centroid (`prior_bank.py:165-172`). Build a scale-(s+1) `BeliefState` whose transported belief supplies
   `p_i` and `r_i` top-down (the cross-scale shadow prior), reusing the existing transport/energy
   machinery + a new *vertical* transport builder. This is the same gate as #1. PIFB eq:extended_free_energy
   (1949-1952), PIFB:545,1233.
3. **Canonical observation-likelihood term `−E_q[log p(o|x)]`.** `free_energy()` accepts `log_likelihood`
   but no caller ever supplies it (`free_energy.py:268,302`), so the canonical F-with-observations is
   unreachable. Either thread a per-token categorical-from-decode likelihood behind an opt-in toggle, or
   remove the half-wired argument. Roadmap already flags the decision. PIFB:1273,1949.

**Tier 2 — systems / scale.**
4. ~~**Checkpoint resume.**~~ ✅ **RESOLVED (2026-06-08).** `save_checkpoint`/`load_checkpoint`
   (`run_artifacts.py:145,174`) now round-trip model + optimizer + RNG + step, and
   `train(resume_from=...)` rebuilds the cosine `LambdaLR` at the saved step. Residual: exact
   dataloader-position restoration is not claimed (resume restarts the token stream). *(low)*
5. **fp16 `GradScaler`.** Wire `torch.amp.GradScaler` in `train.py` gated on `amp_dtype=='fp16'` (bf16/fp32
   unscaled) — see V-section; today fp16 silently mistrains. *(medium)*
6. **Incremental belief reuse in `generate()`** — `model.py:569-597`. Every token re-runs the full
   encode→E-step→decode; cache converged prefix beliefs and warm-start only the appended position. *(low)*
7. **Chunked full-covariance decode.** `_decode_full` is O(B·N·V·K³) per-pair Cholesky
   (`prior_bank.py:473-496`); no chunked counterpart. Register a `full_chunked` decode. *(low)*

**Tier 3 — expandability seams.**
8. **Multi-irrep SO(N)** (spin-1/spin-2 heads) — `generators.py:257` TODO; only the fundamental basis
   exists. Add higher-irrep generator builders + a block-irrep group. *(medium)*
9. **`gauge_fixed` encode** — `prior_bank.py:393-406` registered NotImplementedError stub; implement the
   gauge-orbit encode (every prior = `exp(φ_v)` on a shared base belief) and lift the config rejection.
   *(low)*
10. **Filtered (frozen-keys) free-energy form for `regime_ii`** — `e_step.py:204-212` raises for non-flat;
    the global-F branch is covered, only the filtered *diagnostic* is missing. Also thread RoPE into the
    logged F. *(low)*
11. **`pos_rotation='rope'` odd-`d_head` validation** — `rope.py:68`. An odd irrep block silently leaves
    one coordinate un-rotated; add a config warning. *(low)*
12. **Executable group-admissibility verifier.** `invariant_for` is a string-membership test; add a
    `check_admissible(group, family)` that samples `g=exp(φ·G)` and asserts divergence invariance — a
    guardrail before adding non-Gaussian families. *(low)*
13. **Route live SPD paths through `safe_spd_inverse`/`floor_eigenvalues`** — `numerics.py:63-99` defined
    + tested but unused; the inline `eps·I` ridges are safe, so no urgency. *(info, roadmap item 15)*

**Documentation hygiene (cheap, do alongside).** Stale comments that contradict the code: `transport.py:87`
calls Regime II "deferred" though `regime_ii` is registered; `prior_bank.py:13` calls `full` decode a
"named stub" though it is implemented. Update to match the code (CLAUDE.md: code is the truth).

---

## Appendix — finder yield

`config-dead-toggles` 3 · `registry-modularity` 3 · `group-generator-construction` 2 ·
`transport-equivariance-holonomy` 4 · `spd-retraction-natgrad` 2 · `positional-bch-norm` 2 ·
`divergence-families-alpha` 1 · `belief-gradient-kernel-oracle` 0 (clean) ·
`free-energy-functional-estep` 1 · `mstep-channels-hierarchy` 7 · `attention-decode-kernel` 3 ·
`numerics-stability` 8 · `training-mstep-wiring` 3 · `estep-backward-estimators` 3 ·
`future-buildout` 10 · `metrics-diagnostics-data` 5. Raw 57 → deduped 41 → verified 6.
