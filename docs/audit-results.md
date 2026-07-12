# Partial-Buildout Audit — V3_Transformer — 2026-07-12

## Outcome

This audit inspected the complete repository at `origin/main` commit
`2a988eb16e8907864bc4c01ddc04ef13e90001fc` for features that have executable artifacts but stop
before a required configuration, runtime, persistence, observation, or testing boundary. It did not
treat a comment, manuscript ambition, reserved registry name, default toggle choice, or disclosed
approximation as sufficient evidence. A candidate survived only when current executable source showed
both an implemented portion and a missing end-to-end seam.

Independent verification confirmed 16 partial candidates, which consolidate to 15 implementation
findings because the two-hop phi omission and the Metropolis objective omission share one root cause.
The exclusion register separately records five intentional complete limitations, three reserved or
unstarted ideas, and three refuted claims. After adversarial review, one finding remains High, twelve
are Medium, and two are Low. No default model-training or inference mathematical-correctness failure
was found; PB-01 is instead a default-on experiment-integrity defect. No production code was changed.

The full baseline command was:

```powershell
python -m pytest -x --junitxml="C:\tmp\vfe3-partial-build-baseline-20260712.xml"
```

It completed with `2346 passed, 17 skipped, 0 failed, 0 errors` in 349.32 seconds. The pass count came
from the pytest output and is independently checked against the JUnit XML before closeout.

## Scope and method

The live checkout was dirty and one commit behind the fetched remote, so it was left untouched. The
audit ran in `C:\tmp\V3_Transformer-partial-build-audit-20260712` on branch
`codex/partial-build-audit-20260712`, created directly from the fetched `origin/main`. The repository
contained 218 Python files, below the whole-repository audit threshold.

Three read-only investigators covered runtime and configuration reachability, mathematical and state
completeness, and experiment/reporting/data completeness. Their claims were deduplicated and sent to a
fresh verifier that re-read every cited path. The four candidates initially rated High received a
separate skeptic/defender challenge. Source behavior, rather than comments, determined every final
verdict. Comments and the Research wiki were used only to identify intended boundaries and avoid
misclassifying a disclosed scope choice as a defect.

## Confirmed partial buildouts

| ID | Severity | Partial boundary | Executable evidence | Completion plan |
|---|---:|---|---|---|
| PB-01 | **High** | Default-on ablation resume can reuse stale cells after source, tokenizer, or corpus changes. | `_cell_is_current()` compares reconstructed config, a dataset label, token cap, marker status, and requested diagnostics, but no code or data identity; the cache hit is then reused at `ablation.py:1851-1927,2025-2032,2050-2062`. | [`2026-07-12-artifact-resume-integrity.md`](superpowers/plans/2026-07-12-artifact-resume-integrity.md) |
| PB-02 | Medium | A default ablation cell does not produce the self-contained artifact set the runner advertises. | The remote baseline has `max_steps=15000`, `log_interval=100000`, and `eval_interval=150000`; `_cell_cfg_dict()` forces `checkpoint_interval=0`. The final validation computes the headline result but does not write `metrics.csv`, `best_model.pt`, or a resumable terminal bundle (`ablation.py:20-23,109,356-360,1584-1589,1758-1818`; `vfe3/train.py:1079-1084`). | [`2026-07-12-artifact-resume-integrity.md`](superpowers/plans/2026-07-12-artifact-resume-integrity.md) |
| PB-03 | Medium | Cross-run resume restores historical-best metadata without restoring the corresponding weights. | Checkpoints contain the current model state plus `best_val_ppl` and `best_step`, while load restores only those best scalars. A fresh run directory has no old `best_model.pt`, and finalization reloads only a best file found in the new directory (`vfe3/run_artifacts.py:344-358,452-459,915-946`; `train_vfe3.py:520-579`). `reloaded_best=False` exposes the mismatch but does not repair it. | [`2026-07-12-artifact-resume-integrity.md`](superpowers/plans/2026-07-12-artifact-resume-integrity.md) |
| PB-04 | Medium | The EFE ring experiment discards each trained model and cannot resume completed seeds. | The seed loop trains and evaluates a model, then retains only aggregate adequacy, arm metrics, and gates in one JSON record; no per-seed state, RNG, config fingerprint, or code identity is serialized (`efe_ring_experiment.py:224-270`). | [`2026-07-12-artifact-resume-integrity.md`](superpowers/plans/2026-07-12-artifact-resume-integrity.md) |
| PB-05 | Medium | `efe_rollout` is implemented and config-valid but unreachable through public generation. | Configuration accepts `policy_horizon>1`, and the scorer consumes `(B,Kp,H)` candidates, but `_policy_select()` raises because it constructs only `(B,Kp,1)` menus (`vfe3/config.py:1927-1930`; `vfe3/inference/policy.py:450-497`; `vfe3/model/model.py:2033-2043`). | [`2026-07-12-efe-policy-generation-completion.md`](superpowers/plans/2026-07-12-efe-policy-generation-completion.md) |
| PB-06 | Medium, gate-blocked | A validated sigma-gate artifact cannot activate `sigma_mc`. | Configuration can validate a PASS record, but the registered estimator always raises and the generation call supplies no ambiguity-mode dispatch (`vfe3/config.py:1951-1964`; `vfe3/inference/policy.py:234-250`; `vfe3/model/model.py:2067-2070`). The current empirical gate is FAIL, so implementation must remain fail-closed. | [`2026-07-12-efe-policy-generation-completion.md`](superpowers/plans/2026-07-12-efe-policy-generation-completion.md) |
| PB-07 | Low | Metric and figure registries do not drive production reports, and four completed sweep plots are orphaned. | Non-test call search finds `compute_metrics()` and `get_figure()` only at their definitions; production drivers call concrete functions. `capacity_scaling`, `pareto_frontier`, `ablation_forest`, and `lr_grid_heatmap` render in tests but have no persisted-artifact driver (`vfe3/metrics.py:1302-1409`; `vfe3/viz/figures.py:497-515,2295-2453`; `vfe3/viz/report.py:222-342`). | [`2026-07-12-reporting-registry-completion.md`](superpowers/plans/2026-07-12-reporting-registry-completion.md) |
| PB-08 | Medium, performance | The corpus “memmap” route materializes the requested stream as a full int64 tensor. | The `.bin` loader opens `np.memmap` and immediately converts the entire view to `torch.long`; the uncapped `.pt` branch disables mapped loading (`vfe3/data/datasets.py:86-101,220-230,247-283`). | [`2026-07-12-out-of-core-corpus-loading.md`](superpowers/plans/2026-07-12-out-of-core-corpus-loading.md) |
| PB-09 | Medium, architectural | The observation likelihood exists as a scalar and diagnostic seam but has no live inference value or gradient path. | `free_energy()` accepts and subtracts `log_likelihood`, while the production E-step scalar call omits it and the analytic kernels expose no observation input (`vfe3/free_energy.py:371-387,464-465`; `vfe3/inference/e_step.py:489-494`; `vfe3/gradients/kernels.py:86-104`). A literal target injection would be acausal at deployment, so this requires an observation-model design rather than one more argument. | Existing comprehensive plan: [`2026-07-11-backprop-free-vfe-lm-plan.md`](plans/2026-07-11-backprop-free-vfe-lm-plan.md) |
| PB-10 | Medium | The active q/p/s/h hierarchy has no single typed evaluator with explicit reduction and stop-gradient semantics. | The authoritative `free_energy()` evaluates the q self/beta blocks only. Hyper-prior and gamma terms are either added separately or realized through `_refine_s`, and diagnostics reconstruct them separately (`vfe3/free_energy.py:371-403`; `vfe3/model/model.py:1542-1583,2520-2548`). | [`2026-07-12-hierarchical-probabilistic-completeness.md`](superpowers/plans/2026-07-12-hierarchical-probabilistic-completeness.md) |
| PB-11 | Medium | The model channel remains a diagonal-Gaussian, flat-connection island while supported belief routes use full covariance or nonflat transport. | `s_e_step` rejects full covariance; `_refine_s()` fixes `family="gaussian_diagonal"` and `transport_mode="flat"`; `_gamma_energy()` also uses flat transport (`vfe3/config.py:1554-1590,2145-2158`; `vfe3/model/model.py:773-816,1699-1723`). | [`2026-07-12-hierarchical-probabilistic-completeness.md`](superpowers/plans/2026-07-12-hierarchical-probabilistic-completeness.md) |
| PB-12 | Medium | Nonzero two-hop coupling is absent from the phi coordinate objective and from Metropolis scoring. | Mean and covariance routes include `lambda_twohop`, but `phi_alignment_loss()` has no such input. The reflection scorer also omits two-hop and the folded precision/gamma priors, although compatible combinations construct (`vfe3/inference/e_step.py:505-540,935-970`; `vfe3/model/model.py:1130-1158`). | [`2026-07-12-phi-reflection-objective-parity.md`](superpowers/plans/2026-07-12-phi-reflection-objective-parity.md) |
| PB-13 | Medium | Clebsch-Gordan coupling transforms means but has no covariance pushforward or explicit probabilistic energy contribution. | `CGCoupling.forward()` computes a bilinear mean delta and returns sigma unchanged; the block applies that tuple directly (`vfe3/model/cg_coupling.py:11-13,139-162`; `vfe3/model/block.py:157-158`). | [`2026-07-12-hierarchical-probabilistic-completeness.md`](superpowers/plans/2026-07-12-hierarchical-probabilistic-completeness.md) |
| PB-14 | Medium | The prior-bank decode remains fixed Gaussian alpha-1 KL under supported non-KL or non-Gaussian E-step objectives. | Configuration accepts the objective but warns that decode ignores `renyi_order` and `divergence_family`; decoder kernels call the fixed KL path (`vfe3/config.py:1999-2028`; `vfe3/model/prior_bank.py:20-30`). | [`2026-07-12-hierarchical-probabilistic-completeness.md`](superpowers/plans/2026-07-12-hierarchical-probabilistic-completeness.md) |
| PB-15 | Low | Gradient clipping strategy is configurable, but its threshold is not on the authoritative click-run config surface. | `VFE3Config` exposes `grad_clip_per_role`, while `train()` owns an independent `grad_clip=1.0` argument and the click-run entry points do not pass it (`vfe3/config.py:795-799`; `vfe3/train.py:455,883-905,1538-1545`; `train_vfe3.py:601-611`; `ablation.py:1780-1791`; `scaling.py:765-768`; `check_gpu_tests.py:55-56`). | [`2026-07-12-gradient-clipping-config-wiring.md`](superpowers/plans/2026-07-12-gradient-clipping-config-wiring.md) |

## Adversarial rulings on proposed High findings

| Candidate | Skeptic | Defender | Final | Source-backed reason |
|---|---|---|---|---|
| Default ablation artifact contract | Medium | Medium | **DOWNGRADED to Medium** | The terminal validation number and aggregate marker survive, and checkpoint suppression is explicit; the defect is reproducibility and archival completeness, not corrupted training. |
| Stale ablation resume | High | High | **UPHELD High** | Resume is default-on and silently combines cells whose source or corpus content may differ while labels and config remain equal. |
| Cross-run best-weight continuity | Medium | High | **DOWNGRADED to Medium** | The selected weights are genuinely absent, but `reloaded_best=False` exposes the condition and resume is opt-in. The mixed best metadata still requires repair. |
| Observation-conditioned inference | Low | Medium | **DOWNGRADED to Medium architectural priority** | No supported configuration claims a live observation E-step, and the deployed target-blind path is correct for causal generation. The partial seam matters for the canonical research program, not current default execution. |

## Excluded candidates

The following boundaries are real but are not partially runnable features, so no implementation plan
is assigned by this audit.

| Disposition | Candidate | Reason |
|---|---|---|
| Intentional complete limitation | Flat-preference EFE generation | With the default risk-plus-ambiguity terms, flat preference cancels to a constant; the click-run script explicitly selects ambiguity-only confidence reranking. A typed goal-context API would be a separate future design, not a missing promise or part of the PB-05/PB-06 completion plan. |
| Intentional complete limitation | Ordinary generation without an incremental cache | Generation is correct and explicitly full-recompute. A generic cache would be a separate performance design, not a correctness defect or part of the PB-05/PB-06 completion plan. |
| Intentional complete limitation | Outer frame updates are additive | `phi_retract_mode` is an inner E-step setting. A manifold-aware outer optimizer would be a new feature. |
| Intentional complete limitation | Synchronous best-model writes | The writer is correct and atomic. Its cost should be measured before an asynchronous redesign. |
| Intentional complete limitation | Diagonal GL(K) covariance projection | The projection is disclosed and the exact full-covariance sibling already exists. |
| Reserved, not public | `gauge_fixed` encoder | The registry name and callable both fail closed before a valid model can run. Promote it through a design decision before planning implementation. |
| Reserved, not public | Row-lazy vocabulary optimizer | No runnable registry or partial optimizer exists; this is an unstarted performance idea. |
| Reserved, not public | Exact compact-subgroup Route A / Wilson action | Only a design note exists beside the implemented Route B. It is a research target, not a half-runnable route. |
| Refuted | Filtered-key nonflat/omega-direct global-F route | Production call sites use `keys=None`; the guards block optional direct diagnostic combinations only. |
| Refuted | Laplace natural-parameter exceptions | Varying-location Laplace is not a joint natural exponential family; its closed-form divergence and Fisher natural-gradient routes are implemented and live. |
| Refuted | Reflection STE as a promised mode | Both STE values are rejected at construction and are not presented as supported. The Metropolis objective mismatch survives separately as PB-12. |

## Plan order

The recommended execution order is based on evidence integrity and dependency structure, not raw line
count. First implement artifact/resume integrity, because stale cells and missing selected weights can
invalidate experiments produced by every later feature. Next implement the small gradient-clipping
config seam and the out-of-core loader. Then close phi/Metropolis objective parity before running any
two-hop/reflection experiment. Implement hierarchical probabilistic completeness before reporting
registry completion: reporting composes with the versioned ablation contract and the final typed
diagnostic decomposition. The policy-generation plan can follow artifact integrity but must retain its
empirical sigma FAIL gate. PB-09 already has a larger, falsifiable nudged two-phase plan and should not
be replaced by a cosmetic `log_likelihood` call.

## Research-wiki disposition

The audit consulted `[[VFE Transformer Program]]`, `[[Nudged two-phase EM]]`, the sigma-gate failure
record, and the EFE policy pre-registration. No Research-vault file was changed. The verified
partial-buildout inventory is worth ingesting into the project page after user confirmation because it
updates implementation status without changing the underlying theory.

# Archived audit — V3_Transformer (vfe3) — 2026-07-06

**Fix status (branch `fix/audit-2026-07-06-majors`)**: all four MAJOR findings (M1-M4) fixed with TDD and pinned by new tests (M1: `test_extractors_use_learned_kappa_in_iter_and_fe_kwargs`, `test_converged_state_beta_tracks_learned_kappa`; M2: `test_phi_clamp_monitor_threshold_matches_transport_clamp`; M3: `test_cache_supported_gates_result_changing_toggles`; M4: `test_gauge_transport_figure_aggregates_seeds`, `test_mu_precond_figure_aggregates_seeds`, `test_attention_entropy_figure_aggregates_seeds`). Full suite excluding the `test_viz.py` sklearn/llvmlite env crash: `tests=1539 failures=0 errors=0`. The MINOR (m1-m31), test-suite (t1-t8), and hygiene (h1) items below are NOT yet addressed. See `docs/2026-07-06-edits.md`.

Seven parallel expert passes over the working tree (core free-energy math; geometry/gauge; model assembly; inference/gradients; training/config/entry points; test suite; viz/data/reporting), followed by a full test-suite run and first-hand re-verification of every MAJOR finding by a second read of the cited lines. Per audit instructions, default toggle values were not judged; the audited question is whether theoretically pure paths exist and whether the code does what its seams promise. All claims were checked against actual code, not comments.

**Tree audited**: branch `audit/2026-07-06` cut from local `main` (1ee2be4) with the user's uncommitted WIP intact (`CLAUDE.md`, `scaling.py`, `train_vfe3.py`). Git state note: `origin/main` is one commit ahead (a8ed362, "learnable kappa history figures"), and that commit tracks a `docs/2026-07-06-edits.md` that collides with the local untracked file of the same name — a plain checkout/pull of `origin/main` will refuse until one of them is moved. Not resolved here; user's call.

**Test suite** (run on the Windows venv, counts read from junit XML, not memory): `tests=1576, failures=3, errors=0, skipped=8`, 193.5 s wall. All three failures are `ModuleNotFoundError: No module named 'sklearn'` in `tests/test_viz.py` — scikit-learn is missing from the `.venv`, an environment gap rather than a code defect (the three tests also lack the `importorskip` guard their umap siblings use). 1565 passed.

## Verdict

No CRITICAL math errors. The pure paths hold up: the canonical F with the attention-entropy term is assembled term-for-term against the manuscript form and softmax β is its stationary point (the envelope identity Σβ*E + τΣβ*log(β*/π) = −τ log Z was re-derived and holds because `log_partition` normalizes the prior); diagonal and full-covariance Gaussian KL, the Renyi closed forms, and the Laplace family were re-derived and are exact; Ω_ij = exp(φ_i)exp(−φ_j) transports from j to i consistently everywhere with covariance always moving by congruence ΩΣΩᵀ; the analytic kernels match the KL derivatives in sign and factor and share their building blocks with the autograd oracle structurally (they cannot drift apart silently); so(n)/sp(2m) generator constructions satisfy their defining relations, with irrep towers verified executably at build time; the no-NN constraint holds (the full nn.Parameter inventory reconciles against the exception family, with the documentation drift noted in m1/m4 below); no CLI parsing anywhere; no dead config key; checkpoint/resume, EMA, optimizer grouping, and the metrics math (BPC, token-weighted CE) are correct.

What the audit did find is a cluster of **observability failures**: several diagnostic, figure, and cached-inference paths recompute model quantities with stale or incomplete inputs and silently disagree with the forward pass they claim to describe (M1–M4). None corrupts training; all can corrupt conclusions drawn from runs.

## MAJOR (each independently re-verified)

**M1. Every viz extractor ignores the learned softmax temperature under `learnable_kappa_beta=True`.** `vfe3/viz/extract.py` never threads `kappa_beta_override`: a grep for `effective_kappa_beta|kappa_beta_override` over `vfe3/` hits model.py (forward at 817, diagnostics at 1305/1630/1673/1894/1998) and stack.py, and nothing in `viz/`. Every extractor builds τ from the static init value (`extract.py:61` and the sibling `_fe_kwargs`/`vfe_stack` call sites at 209-218, 278-287, 438, 487-497, 527, 595): `tau=attention_tau(_as_coeff(cfg.kappa_beta, ...))`. Under the just-shipped learnable-kappa toggle, every extracted figure — Σ–CE calibration join, belief banks, converged β/energy, E-step F traces, numerical health — describes a model running at the init temperature, not the trained one. Figures lie exactly when the toggle the figures were built to study is on. Fix: thread `model.effective_kappa_beta(device)` into the extractors' iteration kwargs.

**M2. Phi transport-clamp drift monitor checks 20 while the actual clamp fires at 15.** `vfe3/train.py:48`: `max_norm: float = 20.0,   # stable_matrix_exp_pair's default Frobenius clamp` — but the clamp is `vfe3/geometry/transport.py:742`: `max_norm: float = 15.0`, and both production transport builds call it without override. A φ row whose embedded Frobenius norm drifts into (15, 20] already receives the surrogate `exp(15·M/‖M‖)` transport while the M-step monitor stays silent — precisely the drift regime the monitor was added to catch (2026-07-05 m8), and the monitor warns only once per process so the miss is never recovered. train.py's own docstring says "(default 15)" two lines below the wrong constant, and `config.py:113` repeats "max_norm=20". Found independently by two audit passes. One-line fix: default 15.0, or import the constant from `transport` so the two cannot diverge.

**M3. `cache_supported` misses six result-changing toggles; the cached EFE rollout silently diverges from the full recompute it is pinned to equal.** `vfe3/inference/belief_cache.py:61-79` gates on layers/e-steps/filtering/family/divergence/entropy/transport/priors/pos_rotation/mixers/precond/trust — but not on `lambda_twohop` (the cached kernel call at 132-135 never forwards it), `e_step_update='mm_exact'` or `skip_belief_sigma_update` (137-141 always does the gradient step plus retraction), `query_adaptive_tau`, `gamma_as_beta_prior` (no prior fold), or `learnable_kappa_beta` (line 176 reads `model.cfg.kappa_beta`, the static init, same class as M1). Dispatch is silent (`policy.py:268-269` takes the cache whenever `cache_supported` passes), while `efe_rollout` elsewhere raises specifically to avoid a dishonest fallback. Fix: extend the conjunction with the six gates (or read `effective_kappa_beta` in the cache).

**M4. All three multiseed ablation headline figures are broken by the `__s<seed>` label suffix.** `run_sweep` renames every cell `{label}__s{seed}` when a sweep declares `seeds` (`ablation.py:1681`), and the three sweeps that declare `seeds: [6, 64, 23]` are exactly the ones whose figure parsers were never updated: (a) `_plot_gauge_transport` (A1/EXP-2, "the program's central causal claim") requires `len(label.split("_")) == 2` (`ablation.py:2070-2071`), so `on_L1__s6` → 4 parts → every cell skipped → the figure silently never renders; (b) `_plot_fisher_mu_precond` (B3/EXP-14) does `int("1__s6")` → ValueError → continue (`ablation.py:2231-2237`) → same silent no-op; (c) `_plot_attention_entropy` (C1/EXP-4) parses labels fine but `plot_entropy_ppl_gap._ppl` returns `v[0]` of the three seed cells (`vfe3/viz/figures.py:2849-2852`), so the bar heights and the annotated Δ=surr−canon gap come from one arbitrary seed — a gap that can flip sign by seed presented as the cell value. Fix: strip the seed suffix with the existing `_base_label` helper and aggregate seeds (mean ± SD) before plotting.

## MINOR

**Documentation drift on the exception list (same class as 2026-07-05 m6; CLAUDE.md is user WIP, not edited here)**

- **m1. `connection_M` and `connection_L` are learned transport parameters absent from CLAUDE.md's exception list.** `model.py:243` (`nn.Parameter(torch.zeros(n_gen, 3))`, regime_ii_covariant) and `model.py:261` (`nn.Parameter(torch.zeros(max_seq_len, max_seq_len, n_gen))`, regime_ii_link/_charted). Both are zero-init, default OFF, carry detach-freeze warnings, and are sanctioned by in-code comments — but CLAUDE.md exception (3) names only `connection_W`. The letter of the constraint is violated until the list is updated.
- **m2. `decode_log_scale` is an always-created, always-trainable learned scalar with no freeze toggle.** `prior_bank.py:182` creates it unconditionally; `prior_bank.py:432` applies it (`tau_eff = tau·exp(−clamp(s,−3,3))`); `train.py:149` trains it in the sigma group. It is a gauge-invariant per-model scalar (same equivariance class as the kappa exceptions), but it is matched to no documented exception and no toggle yields a fixed-temperature decode — the only freeze (`m_p_sigma_lr=0`) also freezes `sigma_log_embed`. The theoretically pure fixed-τ decode path does not exist under any toggle.
- **m3. `output_proj_bias` (`prior_bank.py:218-219`) and the `untie_decode_bank` tables (`prior_bank.py:279-280`) extend exception (1) beyond its letter** (opt-in, zero-init/cloned, step-0 byte-identical; documentation parity only).

**Silent divergence between forward pass and diagnostics/replays**

- **m4. `gamma_as_beta_prior` fold missing from every replay.** Forward folds it (`model.py:795-801`); `diagnostics` (1614-1615), `attention_maps` (1887-1888), `gamma_attention_maps` (1293-1294), `diagnostics_per_layer` (1994-1995), and all extract.py replays fold only the precision bias. Under the toggle, every replayed E-step converges a different belief than the forward, silently.
- **m5. `numerical_health` raises NameError under `pos_rotation='rope'`.** `extract.py:427` uses `RopeTransport`, which is imported only locally inside `converged_state` (479) and `attention_entropy_cov_gap` (571), never at module level (the line-30 import lists four other names). report.py's `_safe` swallows the NameError, so the health panel renders empty for exactly the rope configs the r2-id11 fix targeted. Verified by grep. Fix: add the local import.
- **m6. `converged_state` rebuilds the last-block prior from the final belief only** (`extract.py:498-502` reuses `out.mu` for every handoff step) where the real stack blends per-layer intermediates (`stack.py:96`). The self_div figure is quantitatively wrong for `n_layers>1` with `prior_handoff_rho>0`; exact at L=1 or ρ=0.
- **m7. One-step provenance skew inside a metrics row**: F-decomposition diagnostics are computed post-step, loss/CE pre-step (`train.py:884-894`). Cosmetic.

**Free energy / E-step**

- **m8. Entropy-term prior clamp distorts logged F and oracle gradients for strong finite priors.** `free_energy.py:440` clamps π at 1e-12 before the log; a finite prior below −27.6 nats relative to its row (ALiBi tail at large N) makes the logged F deviate from −τ log Z, and the autograd oracle inherits the distortion through the dβ·log π terms; the kernel route is immune. Same pattern at `metrics.py:155`. Fix: `log_softmax` with masked entries neutralized where β = 0.
- **m9. `lambda_twohop > 0` with `e_phi_lr > 0` descends a mixed objective**: `phi_alignment_loss` (`e_step.py:351-417`) has no two-hop block while the mu/sigma kernel (`kernels.py:146,153`) and both F bookkeepers carry it — φ descends a strictly different objective than μ/σ, unwarned. Also the hop-weight masking convention differs between kernel (pair-masked β) and F value (unmasked β) under saturation.
- **m10. `e_steps_backprop_last` boundary detach is defeated by the hoisted transport on the default flat path**: the last-k iterations consume `_hoisted_omega` built from the pre-detach φ (`e_step.py:803-820, 869-885`), so encode/pos-φ tables still receive transport gradients through the truncation boundary on the flat route but not on rebuilt routes — semantics differ by transport mode.
- **m11. `straight_through` plus the opt-in mean trust region leaks a live σ dependence where the clamp binds** (`e_step.py:679-683` passes live `belief.sigma` into the whitening; `numerics.py:47-55`). Detach σ in the whitening under ST (values unchanged).
- **m12. `mm_exact_update` on a fully saturated row jumps to (μ*, σ*) = (0, eps)** instead of staying put (`kernels.py:485-487`, a=0, w=0 corner); the gradient route moves nothing there. Blend toward the current belief when P floors.
- **m13. `free_energy_value` calls `transport_covariance` without `diagonal_out`** (`e_step.py:321-322`) — the one diagnostic site not hardened by the m7 (2026-07-05) fix; a link-regime Ω against a batched diagonal σ raises, or at B=N=K silently runs the full-cov sandwich on diagonal σ. Pass the explicit flag as kernels.py:362/oracle.py:122 do.
- **m14. Dead `gauge_mode` kwarg**: `e_step.py:815` reads it from `**kwargs` for the hoist but forwards the still-populated kwargs to callees that do not accept it — any caller actually passing it gets a TypeError one line later. Delete or plumb.
- **m15. Oracle truncation unwarned when only σ is grad-free** (`oracle.py:98` requires both; the runtime warning at `e_step.py:547-562` fires only when `not oracle_unroll_grad`).

**Geometry**

- **m16. The retraction's coordinate-norm cap under-bounds the embedded norm for so_n/sp_n towers, so the exp clamp surrogate is reachable on the pure tower path.** Retractions cap ‖φ‖ in coordinates (`lie_ops.py:374-376`; π for so_n, 5.0 for glk) but tower generators are not unit-norm: ‖embed(φ)‖²_F = φᵀ Gram φ, and per-generator norms grow with irrep rank (so(3) l3 block: π·√28 ≈ 16.6 > 15; sp(2) sym3: 5·√20 ≈ 22 > 15), while the clamp at `transport.py:796-808` measures the full-matrix norm with `clamp_monitor=False`. For skew inputs the clamp is not even needed (exp of skew is orthogonal at any norm) yet silently shortens the rotation. Cap on φᵀ Gram φ, or default the monitor on for tower groups.
- **m17. `_KILLING_INV_CACHE` keys on `data_ptr()` without retaining the tensor** (`phi_preconditioner.py:132-136`) — free-and-realloc can collide two same-shape bases with different values (two towers built sequentially in one ablation process). The sibling cache in `lie_ops.py:133-134` retains the tensor for exactly this reason. Opt-in killing modes only.
- **m18. `invariant_families` names a family that does not exist** (`groups.py:40` lists `"gaussian"`; the registry has `gaussian_diagonal`/`gaussian_full`/`laplace_diagonal`). Currently inert (no runtime consumer), but a future admissibility guard keyed on `cfg.family` would never match.
- **m19. Pullback Ψ series can exhaust `series_order=40` without meeting tolerance, silently** (`phi_preconditioner.py:312-317`). Warn on exhaustion.

**Model / decode**

- **m20. `z_loss_weight > 0` is silently inert on every dense (non-chunked) decode path.** The dense branch (`model.py:944-957`) adds no z-loss; only the four fused chunked kernels consume it (`model.py:920/929/934/941`). config.py validates `>= 0` and never warns. Add the z-loss to the dense branch or fail closed in config.
- **m21. Fused-CE dispatch is hardcoded on literal `decode_mode` names** (`model.py:915-935`): a future decoder registered with `chunked=True` would fall into the generic bank branch and get `decode_ce_diagonal_chunked` — a different kernel than its logits — silently. Register the fused-CE twin alongside the decode kernel.
- **m22. Dense decode floors KL at 0; the fused CE kernels do not** (`prior_bank.py:997` vs `559`), so fused CE ≠ CE(decode(...)) bit-wise in the near-zero-KL band (within the 1e-3 test pin; also a gradient asymmetry where the floor binds).
- **m23. `_log_prior_cache`/`_rope_cache` grow unboundedly during generation** (`model.py:289-292, 494`): one (n,n) entry per distinct length as `generate()` re-runs the forward — Σn² ≈ N³/3 floats retained. Bound or slice-from-largest.
- **m24. A single `t5_bias` table is shared when both β and γ channels select t5** (`model.py:335-336`) — plausibly intended, not documented.

**Training / artifacts / analysis**

- **m25. `RunArtifacts` pointed at an existing run dir truncates `metrics.csv`** (first `log_metrics` of a resumed process opens `"w"`, `run_artifacts.py:135-138`) and `__init__` rewrites `config.json`. Safe on the standard fresh-dir-per-invocation path; document or append-with-header-check.
- **m26. Cross-run-dir resume restores best-val scalars but not `best_model.pt`** (`run_artifacts.py:366-368` vs the `best_path.exists()` gate at 684): `summary.json` reports the old best while the test eval scores final weights (`reloaded_best=False` makes it visible). Copy the weights or null the scalars.
- **m27. `grad_clip` is not config-plumbed** — frozen at 1.0 for every entry point (`train.py:731-733`; no config field, no caller passes it). The one training hyperparameter outside the config surface.
- **m28. `scaling_analysis` explicit-null coalesce was fixed for `test_ce` only** (`scaling_analysis.py:74-76`); `test_ppl`/`test_bpc`/`n_params`/`wall_time_s` (83-104) keep the vulnerable `d.get(k, fallback)` pattern that drops a real value when the first source holds an explicit null.
- **m29. The offset power-law fit silently ignores the WLS weights** (`figures.py:2262-2268` passes no `sigma` to `curve_fit` on the `with_offset=True` headline path while the log-log branch honors them), and single-seed points get weight 1.0 against (mean/SEM)² ≫ 10³ for multi-seed points in `plot_scaling_law:2349` — effectively excluding multi-seed cells from the fit they anchor.
- **m30. `_plot_one_sweep` labels the categorical bar axis "validation PPL"** on the y (arm names) while the actual PPL axis (x) is unlabeled (`ablation.py:1917-1922`).
- **m31. `parameter_report`'s grad-enabled forward consumes global RNG under `randomize_e_steps=True`**, shifting the batch/init stream for the deprecated `run_training` entry only (`e_step.py:842-845`, `train.py:1187-1189, 1256-1262`); `train_vfe3._run_once` reseeds and is immune.

**Test suite** (counts and durations measured from the junit XML of this audit's run)

- **t1. `test_gram_pinv_is_cached_and_value_identical` builds `block_glk(140, 7)`** — K=140 with a double fp64 Gram-pinv over 2800 generators (`test_audit_fixes_2026_06_13.py:22-28`). Measured 3.0 s today, so a rule violation (K < 6 mandatory) rather than a hang, but it is the single largest-K object in the suite and its cost is O(n_gen²·K²). Pin the caching property at a golden shape; the K=140 orthonormality fact is analytic (E_ij basis ⇒ Gram = I).
- **t2. `test_train_vfe3_clickrun_importable_and_runs_one_step` trains one step of the LIVE `train_vfe3.py` config** (`test_train.py:383-395`). Measured 3.2 s under today's WIP config, but the cost is coupled to the user's live toggles — an innocent config edit (bigger embed_dim, gaussian_full, more e-steps) silently turns this into a multi-minute CPU job, against CLAUDE.md's "production configs belong on the GPU, not in the test suite." Keep the importability assertions; run the train step on a dim-shrunk copy.
- **t3. The sole positive learnability gate cannot fail**: `test_training_decreases_loss_on_structured_stream` is `@pytest.mark.xfail(strict=False)` (`test_train.py:188-198`) — it passes on XPASS and silently xfails on regression, so the suite's one "the model actually learns structure" cutover is unenforced (16.0 s of it runs every suite invocation regardless). The reason string documents the thin margin honestly; promote after the planned GPU re-validation or add a comfortably-cleared smoke bound as a hard assert.
- **t4. The three sklearn tests in `test_viz.py` hard-fail instead of skipping** when scikit-learn is absent (this run's only failures). Add `pytest.importorskip("sklearn")` like the umap tests — and/or install scikit-learn into the venv.
- **t5. K-norm drift**: K=8 is the de facto norm across newer tests (tier12, belief_cache, phase0, efe_scorer, exp*-buildouts, cell_diagnostics), with outliers K=12 (`test_ring_task.py`), K=20 inherited from ablation baselines (`test_cell_diagnostics.py:78-88`, construction-only), K=48 generators-only (`test_blocks_k48_followup_routes.py:44-47`), and GPT-2 vocab 50257 in three train tests. All terminate fast today; the mandate says K=2–4.
- **t6. CPU-skipped coverage**: the Laplace CPU/CUDA agreement test and the CUDA half of the efe_scorer device regression only run under `VFE3_TEST_DEVICE=cuda` — the GPU leg must actually be run periodically or those pins guard nothing.
- **t7. Hygiene**: unseeded global-RNG draws in `test_alpha_i.py`, `test_belief.py`, and scattered spots (input-independent assertions, low flake risk, but they inherit ambient seeds); one weak OR-assertion (`test_divergence.py:650` — either table's grad being non-None passes).
- **t8. The one genuine coverage gap**: no end-to-end full-model gauge-invariance property test on the pure path (transform all prior tables by a global group element; assert logits ranking unchanged). Equivariance is pinned thoroughly at the KL/transport/mixer/metric component level; the composite claim is never pinned directly.

**Hygiene**

- **h1. `tmp_probe_out.txt`** (contents "DONE") sits at repo root — leftover probe file, gitignored but against the "don't leave messes" rule. Delete when convenient.

## Verified sound (condensed; each item re-derived or traced by an audit pass)

Diagonal and full-covariance Gaussian KL, all Renyi closed forms (including the α>1 non-PD → kl_max policy and the fp64 cancellation bands), and the Laplace KL/Renyi integrals: exact. F assembly matches the canonical reference term-for-term; canonical-vs-surrogate routing is structural (`uses_kernel_route` requires entropy-ON, so the surrogate always reaches the oracle, which carries the −τ⁻¹ Cov_β correction exactly); τ = κ√(block dim) plumbed once, per-head and per-query broadcasts mutually consistent; α* = c0/(b0+D) is the exact envelope stationary point and the kernel reuses the registered form. Transport direction convention globally consistent; ΩΣΩᵀ congruence on every path; factored/fused fast paths are exact reassociations of the dense einsums; generators satisfy defining relations with executable build-time verification; matrix exp is exact `torch.linalg.matrix_exp` everywhere (the only approximations are the documented clamp, the order-4 BCH compose mode, and the Ψ series — m16/m19); retraction/natural-gradient/Fisher formulas correct; RoPE is the flat cocycle of the composite frame with exactly orthogonal blocks. Analytic kernels match KL derivatives in sign and factor; kernel and oracle share building blocks structurally; mm_exact fusion is the exact zero of the kernel gradients; straight-through detaches exactly what its warnings claim; every E-step-coupled parameter warns at construction under severing estimators. Full nn.Parameter inventory reconciled (16 parameter sites, all matched to exceptions or flagged above); no nn.Linear/MLP/activations anywhere; no CLI parsing (the only sys.argv is inside the isolated UMAP subprocess string); no dead or orphaned config key in a full plumbing sweep; typo'd config keys fail loud through the dataclass. Optimizer exact-coverage guard; decay-exempt grouping for the exception params correct (connection/sigma tables inherit global decay with documented escape hatches); gauge optimizer applies no Euclidean decay to manifold params; checkpoint bundle complete (model/optimizer/RNG/scaler/EMA/config) with atomic writes and fail-loud strict loading; EMA bracketing correct. Dataset target alignment is exactly input-shifted-by-one with no train/val leakage and seeded loaders; BPC/PPL/CE math correct; multiseed statistics (SEM, t-CI, Holm, BH, McNemar, paired bootstrap) correct. The test suite is oracle-triangulated (hand kernel ↔ autograd ↔ finite differences), golden values are pinned literals, no duplicated test names, no loose tolerances, no tautological pins found beyond t7.
