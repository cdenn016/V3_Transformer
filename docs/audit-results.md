# Deep Audit — 2026-06-21 — vfe3/

Multi-agent deep audit of the `vfe3/` package (clean-room gauge-theoretic VFE transformer). Base-five engineering sweep + the full seven-agent `audit-*` domain-expert tier + an independent source-cited verifier + an adversarial red/blue challenge on the one pure-path finding. Conducted under the project audit rule: a non-pure *default* toggle is not a defect — only an *absent* pure path is — and findings rest on executable code, not comments.

## Headline

The codebase is in strong shape. **No critical or high finding survived**, and — decisively — the four math/theory lenses (gauge, information-geometry, variational, and the transformer-architecture lens) returned **clean negatives backed by float64 recomputation**: the covariance transport is the exact congruence sandwich `Ω Σ Ωᵀ` (not a bare product), the flat Regime-I path is genuinely flat (holonomy ≈ 0), RoPE/irreps/Clebsch-Gordan are correct, the free-energy assembly matches the canonical `F` term-by-term with the load-bearing attention-entropy term present, the reduced `F = −τ log Z` and the envelope identity hold, EM target-blindness is respected, and the KL/Rényi/Fisher/natural-gradient machinery is correct (α=1 limit guarded, divergences non-negative, self-divergence zero). The config-wiring lens confirmed the **theoretically-pure path is selectable and wired end-to-end** with an exact optimizer coverage guard. The surviving items are engineering, performance, and config-hygiene, plus one bounded pure-path enhancement on the SPD-retraction ceiling.

## Scope

`vfe3/` — config, free_energy, divergence, attention_prior, alpha_i, lambda_h_i, numerics, ema, metrics, belief, gauge_optim, train; geometry/{transport,rope,groups,generators,irreps,cg,lie_ops,retraction,phi_preconditioner,closure,norms}; inference/e_step; gradients/{kernels,oracle}; families/{base,gaussian,laplace}; model/{model,block,stack,prior_bank,head_mixer,cg_coupling,positional_phi}.

## Investigators dispatched

- **Base five:** code-reviewer, debugger, refactoring-specialist, performance-engineer, python-pro.
- **Expert tier (theory gate met — full pool):** audit-gauge-theorist, audit-geometer, audit-info-geometer, audit-variational, audit-transformer-ml, audit-numerical-analyst, audit-implementation-engineer.
- **Verifier:** general-purpose (source-cited, all 12 medium+ findings re-read).
- **Challenge:** audit-skeptic vs audit-defender on F5 (the SPD-retraction pure-path claim).

## Clean negatives (recomputed; no findings in these lenses)

- **code-reviewer:** no security/quality/no-NN-purity defect. `weights_only=True` on all external loads; the one `weights_only=False` is a self-written resumable bundle; no `eval`/`exec`/`shell=True`; the linear decode is the sanctioned raw-`nn.Parameter` matmul; no `nn.Linear`/MLP/activation module anywhere.
- **audit-gauge-theorist:** no findings. Sandwich congruence exact; flat cocycle/holonomy ≈ 0; RoPE subgroup correct; irreps real in the algebra; in-group equivariance exact with a live out-of-group negative control; the pure equivariant path (flat, mixer/cg/connection off) verified on every group.
- **audit-info-geometer:** no findings. KL argument-order (forward) correct; α=1 Rényi limit guarded (no `/(α−1)` blow-up); f-divergences non-negative; Fisher metric = score covariance; natural gradient `δμ=Σ∇μ`, `δΣ=2Σ sym(∇)Σ`; α-envelope cancellation exact.
- **audit-variational:** no findings. `F` term-by-term canonical; attention-entropy term `τβ log(β/π)` present and load-bearing (without it β is a delta, not softmax); reduced `F=−τ log Z`; envelope kernel-vs-oracle agree to ~1e-4 FD; EM target-blindness intact; observation term a gated stub.
- **audit-implementation-engineer:** no config-wiring findings. End-to-end forward/backward shows every active toggle's parameter trains; the only null-grad params are the documented intentional-dead set; pure path constructs with zero warnings and routes to the closed-form kernel; every registry seam dispatches through the config value.

## Surviving punch list (post-verifier, post-challenge; none critical/high)

All twelve medium-band findings below were **CONFIRMED** by the independent verifier against source (no refutations, no contradictions). Ranked by actionability.

1. **[medium · perf] `_rel_gap_eps` host-sync on the full-covariance retraction** — `geometry/retraction.py:104` does `float(A.detach().abs().max())` (a CUDA→host sync), called 3× per `retract_spd_full` (166/179/186) and 3× per `retract_logeuclidean_full` (266/278/283). Serializes the GPU pipeline on the full-covariance path (the diagonal default avoids it). *Fix:* derive the relative gap from the eigenvalues already materialized by the first `eigh`, or accept the fixed `1e-8` for the retraction's internal calls.
2. **[medium · perf] Redundant per-iteration transport rebuild** — `inference/e_step.py:433-437` rebuilds `build_belief_transport` at the top of every `e_step_iteration`; on the flat path it depends only on `belief.phi`, which is unchanged across iterations when `e_phi_lr==0`, so the rebuild is fully redundant there. *Fix:* hoist the flat-path transport out of the iteration loop when `e_phi_lr==0`.
3. **[medium · correctness, diagnostic path] `gamma_attention_maps` omits `connection_M` and `_fold_precision_bias`** — `model/model.py:1136-1147` passes `connection_W` but not `connection_M` to `vfe_stack`, and skips `_fold_precision_bias`, unlike `forward` (711/722), `attention_maps` (1546/1562), and `diagnostics` (1284). Under `transport_mode='regime_ii_covariant'` and/or `precision_weighted_attention=True` the gamma-channel visualization replays a different geometry/prior than the model's actual forward, so the diagnostic figure diverges from real behaviour. No effect on training/inference (diagnostic, `no_grad`). *Fix:* add `connection_M=getattr(self,"connection_M",None)` and `log_prior=self._fold_precision_bias(log_prior, belief.sigma)` to match the siblings.
4. **[low-medium · pure-path enhancement] SPD-retraction `sigma_max` ceiling breaks gauge-equivariance in the saturated tail (challenged, UPHELD-downgraded)** — `geometry/retraction.py:187` (and `:284`) clamps output eigenvalues `clamp(min=eps, max=sigma_max)` unconditionally; the defender demonstrated that on a legitimate `block_glk`/`spd_affine`/`trust_region=5`/`sigma_max=10` config the congruence inflates eigenvalues past 10.0 and the clamped map is non-congruence-equivariant (rel err 0.91 vs 2.6e-6 un-clamped), with no toggle (`sigma_max` is `float`, not `Optional`) to recover the affine-exact map. The skeptic's offsetting points: the `eps` *floor* is a necessary positivity guard (defender conceded), the map is exactly affine in the operative unsaturated regime, and the ceiling is a ledger-documented numerical guard (pass-7). **Verdict: UPHELD, downgraded to low-medium, narrowed to the ceiling** (not the floor). *Fix:* expose `sigma_max: Optional[float]` (honor `None` to disable the ceiling while keeping the `eps` floor), or replace the per-eigenvalue ceiling with a congruence-equivariant trace/det bound, so the affine-exact retraction is selectable.
5. **[low] `sigma_max` default mismatch (mitigated)** — `VFE3Config.sigma_max=10.0` (config.py:349) vs retraction-function defaults `5.0` (retraction.py:116/142/207/237) and `free_energy_value` default `5.0` (e_step.py:179). The live call sites thread `cfg.sigma_max` (block.py:67, e_step.py:516), so the mismatch only bites a caller relying on a bare default. *Fix:* align the function defaults to the config value or share a module-level constant.
6. **[low] `gauge_parameterization` reserved-stub field (challenged down from high)** — config.py:64/668 validates and stores it but no runtime path reads `cfg.gauge_parameterization`; the only non-`'phi'` value `'omega_direct'` is rejected with `NotImplementedError` (config.py:682). The implementation-engineer's read (a degenerate single-value reserved stub, not a propagation failure) is correct; downgraded from the refactoring lens's "high". *Fix:* mark as a reserved stub (`# not yet consumed`) or remove until the `omega_direct` parameterization lands.
7. **[low] `self_coupling_alpha` swallows misspelled kwargs** — alpha_i.py:173 forwards `**kwargs` verbatim to `get_alpha(mode)(kl, **kwargs)`, and each form also has trailing `**kwargs`, so a typo (`b0→b00`, `c0→C0`) is silently dropped and the default coupling used. *Fix:* enumerate the accepted kwargs (reject unknowns) or assert no residual kwargs in each form.
8. **[low] `model.py` hardcodes `transport_mode` string branches** — model.py:215/237/1315-1316/1572-1573/1672-1673 pattern-match literal mode strings instead of querying the transport registry, so a new regime requires editing five sites (the modularity contract is otherwise honored — `_transport` itself goes through `get_transport`). *Fix:* have the transport builder declare its parameter needs (`needs_mu`/`needs_sigma`/`needs_connection`) so `model.py` queries the registry.
9. **[low] Duplicate diagonal-Gaussian KL** — `gradients/kernels.py:37-77` (`_raw_diag_kl`/`_per_coord`) reimplements the diagonal KL that `families/gaussian.py:99-104` already computes; the two can drift on a clamp/sign edit. *Fix:* extract one unclamped diagonal-KL utility in `gaussian.py` and import it.
10. **[low] `DiagonalLaplace.natural() -> NoReturn` vs ABC `-> Tuple[...]`** — laplace.py:91 narrows the `BeliefParams` (base.py:83) return contract to `NoReturn`; an LSP/`mypy` violation (runtime raise is correct). *Fix:* annotate `-> Tuple[torch.Tensor, ...]` and let the body raise.
11. **[low] `_SQRT_D_CACHE` keyed on raw `torch.device`** — free_energy.py:79/86; `torch.device("cuda") != torch.device("cuda:0")` → a duplicate cache entry on a device-string mismatch (benign re-alloc, defeats the cache). *Fix:* normalize the key to `str(device)`.
12. **[low] `attention_tau` recomputed per `vfe_block`** — block.py:62 inside the per-layer loop (stack.py:64); inputs are loop-invariant (the expensive `_sqrt_dims` is already cached, so the residual cost is the call/lookup). *Fix:* hoist `tau` to `vfe_stack` and pass it in.

### Expert-tier low items (recorded, not on the main list)
- `FullGaussian.entropy()` discards the `safe_cholesky` ok-mask → a finite wrong value on a non-PD input (gaussian.py:263; no live caller). `_logdet_chol` clamps a failed factor diagonal to `1e-12` instead of propagating NaN (base.py:41).
- The `pullback`/`killing` φ-preconditioner solve is under-ridged near the `Ψ(ad_φ)` resonance at `ad`-eigenvalue `2πi` (phi_preconditioner.py:271/336/405; opt-in path, reachable only by higher-spin towers). *Fix:* scale the ridge to the metric spectrum.
- Diagonal vs full SPD trust-region geometry differ (L∞ box vs Frobenius ball; retraction.py:128 vs 175), so the diagonal family is not the diagonal restriction of the full family once the trust region binds.
- `_refine_s` (model.py:610) forces flat transport and drops RoPE, so under the double opt-in `s_e_step=True`+`pos_rotation='rope'` the model channel refines under inconsistent positional geometry (self-documented; off the default path).

## Verifier verdicts

All 12 main findings: **CONFIRMED** against source (file:line cited per finding above). Zero REFUTED, zero INCONCLUSIVE on the factual claim; the perf findings are confirmed as static facts (runtime magnitude unmeasured). No two findings contradicted each other (F4 and F5 reinforce; the only tension was the F1 severity label, on which the facts are agreed).

## Adversarial challenge

| Finding | Red (attack) | Blue (defend) | Verdict | Reason |
|---|---|---|---|---|
| F5 SPD `sigma_max` ceiling / no affine-exact path | DROP: affine map exact below the ceiling; `eps` floor necessary; ceiling a documented guard reached only on opt-in extreme | UPHELD-MEDIUM: ceiling reached on legitimate `block_glk` default; non-equivariant (0.91 rel err); no `sigma_max=None` bypass | **UPHELD, downgraded to low-medium, scoped to the ceiling** | retraction.py:187 unconditional clamp; reachable via e_step.py:514/config.py:355; `eps` floor conceded necessary so finding narrows to the ceiling; ledger-documented guard bounds severity |

## Test suite

- Command: `python -m pytest tests` (system Python, torch 2.11 CPU, `VFE3_TEST_DEVICE=cpu`)
- Result: **1142 passed, 1 skipped, 1 xpassed, 0 failed, 0 errors** (1144 total) in 238s
- Failures: none

## Disposition

No code was modified — per the deep-audit contract, the verified, challenged punch list is presented for authorization first. None of the surviving items is critical/high; all are engineering/perf/hygiene plus the one bounded SPD-retraction pure-path enhancement (item 4). Recommended order if fixes are authorized: items 3 (diagnostic one-liner) and 1–2 (perf hot-path) first, then 5–12 (low hygiene), with item 4 (the `sigma_max` ceiling) as a deliberate design decision (optional ceiling vs trace/det control) rather than a quick patch.
