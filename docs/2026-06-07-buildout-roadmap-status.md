# VFE_3.0 Buildout Roadmap — Status Update (2026-06-07)

This document re-verifies every item of `docs/2026-06-01-buildout-roadmap.md` against the
live codebase six days later. The verification ran an eight-investigator read-only sweep
(one per item cluster) under a deliberately strict bar set to defeat the roadmap's own
recurring failure mode (a feature that exists in the geometry layer but is dead from the
entry point, e.g. the original cross-coupling finding). A verdict of **ADDRESSED** requires
*both* a config-selection path *and* a test that exercises that config-selected path
end-to-end, each cited at `file:line`. Symbol presence alone (a bare `@register_*` decorator,
or a function no config can reach) is downgraded to **PARTIAL**. Several agents additionally
ran the relevant targeted tests and recorded the pass counts inline.

The headline: the roadmap's entire prioritized top-ten punch list — the family seam, the
M-step self-coupling regularizer, the straight-through E-step toggle, the retraction registry,
autoregressive generation, the hyper-prior/model-coupling channel, the transport registry with
Regime II, checkpoint *saving*, chunked-vocab decode — has been built and wired since 2026-06-01,
along with mixed precision, gradient accumulation, RoPE, cross-coupling, the Sp group, learnable
alpha, the `squared_hellinger` second f-divergence, and the M1–M4 modularity items. What remains
is a short tail of genuine gaps, only one of which is high value.

## The one high-value gap: checkpoint resume (load side)

`save_checkpoint` (`run_artifacts.py:145`) writes a resumable
`{step, model_state, optimizer_state, config}` bundle, and `train()` calls it every
`cfg.checkpoint_interval` steps (`train.py:512`). Nothing reads it back. `train()`
(`train.py:330`) unconditionally builds a fresh optimizer (`train.py:369`) and a fresh
`LambdaLR`, then iterates `range(n_steps)` from zero; `finalize_run` reloads `best_model.pt`
for *test evaluation only* (`run_artifacts.py:190-197`), which is not a training resume. There
is no `load_checkpoint`, no `resume_from`/`start_step` thread into `train()`, and no
`LambdaLR.last_epoch` restore, so a killed multi-hour 5090 run cannot continue. The save side
is the only half built. (The uncommitted working-tree edit that deleted the "Checkpoint resume
(write-only today)" bullet from the 06-01 roadmap was premature: the save side is now wired,
but the load/resume side remains absent, confirmed by reading the actual `train()` loop.)

## Per-item status ledger

Identifiers match the 06-01 roadmap's punch-list ranks (PLn), modularity findings (Mn),
theory-fidelity findings (Tn), and Tier-D items.

| ID | Item | Status | Evidence (code / config / test) |
|---|---|---|---|
| PL1 | Family seam (`ExponentialFamily`) | ADDRESSED | `families/base.py:48-211`, `families/gaussian.py`; cfg `config.py:437`; tests `test_families.py:65-109`, `test_full_covariance.py:33` |
| PL2 | M-step self-coupling `alpha*KL(q*\|\|p)` | ADDRESSED | `model.py:441-482`; cfg `config.py:150`; test `test_mstep_self_coupling.py:56` |
| PL3 | Straight-through E-step toggle | ADDRESSED | `e_step.py:370-379`; cfg `config.py:249`; test `test_straight_through.py` (16 passed) |
| PL4 | SPD retraction registry | ADDRESSED | `retraction.py:24,177,274` (spd_affine, log_euclidean); cfg `config.py:210`; test `test_retraction.py:250,384` (5 passed) |
| PL5 | Autoregressive `generate()` + sampling | ADDRESSED | `model.py:562`; reached from `train.py:460`; test `test_generate.py:32-129`. *No `_SAMPLERS` registry (samplers inline); capability complete.* |
| PL6 | Hyper-prior `lambda_h` + gamma model-coupling | ADDRESSED | `model.py:483-558`, `prior_bank.py:162-198`; cfg `config.py:158,172`; tests `test_hyperprior.py`, `test_gamma_coupling.py` (18 passed). *As an M-step loss regularizer; the live E-step s-update is now the opt-in `s_e_step` channel (PL6-3, 2026-06-08).* |
| PL7 | Transport registry + Regime II | ADDRESSED | `transport.py:94-128`; cfg `config.py:71,76`; test `test_regime_ii.py` |
| PL9 | Chunked-vocab decode | ADDRESSED | `prior_bank.py:272,456`; cfg `config.py:215`; test `test_chunked_decode.py:114` (atol-1e-3 vs full) |
| PL11 | f-divergence beyond Rényi (`squared_hellinger`) | ADDRESSED | `families/base.py:254-282`; cfg `config.py:46`; test `test_divergence.py:556` |
| PL12 | Cross-coupling reachable from config | ADDRESSED | `model.py:55-56`, `groups.py:96`; cfg `config.py:82`; test `test_model.py:310`, `test_gauge_groups.py:124` |
| PL13-rope | RoPE-on-mu transport | ADDRESSED | `geometry/rope.py:47`; cfg `config.py:116`; test `test_rope.py:89` |
| PL14 | alpha>1 full-cov Cholesky hardening | ADDRESSED | `families/gaussian.py:253-279` via `numerics.safe_cholesky`; test `test_divergence.py:155` (2 passed) |
| PL16 | Mixed precision (`amp_dtype`) | ADDRESSED | `model.py:274-311`; cfg `config.py:320`; test `test_amp.py:52,106` |
| grad-accum | Gradient accumulation | ADDRESSED | `train.py:257-270`; cfg `config.py:289`; test `test_grad_accum.py:82` |
| PL19-Sp | Sp(2m,R) gauge group | ADDRESSED | `groups.py:168`, `generators.py:210`; cfg `config.py:359`; test `test_model.py:290` |
| PL19-alpha-learnable | Learnable alpha | ADDRESSED | `alpha_i.py:87`; cfg `config.py:133`; test `test_learnable_alpha.py` |
| PL19-batched | Batched `pairwise_energy` (equal blocks) | ADDRESSED | `free_energy.py:95-111`; default block_glk path; test `test_free_energy.py:200` |
| M1 | Covariance `cov_kind` tag (no name-sniffing) | ADDRESSED | `families/base.py:67,138`; `free_energy.py:48,162`; test `test_divergence.py:360` |
| M2 | Parameter-object divergence signature | ADDRESSED | `free_energy.py:61-135`, `families/base.py:201`; test `test_families.py:211` |
| M3 | `BeliefState` extensible (s,r fields) | ADDRESSED | `belief.py:22-30` (NamedTuple + trailing Optional); test `test_belief.py:51` |
| M4 | Config validates `cov_kind`, not name literal | ADDRESSED | `config.py:436-443`; test `test_config.py:291,361` |
| **PL8** | **Checkpoint resume (load side)** | **MISSING** | save-only `run_artifacts.py:145`; `train()` has no resume `train.py:330`; no `load_checkpoint` anywhere |
| PL13-priors | T5 relative-bias + windowed attention priors | MISSING | `attention_prior.py` registers only uniform/causal/alibi/causal_alibi |
| PL10 | Causal-packed transport (skip j>i triangle) | MISSING | `transport.py:341` always builds full `(B,N,N)`; perf-only, L-effort, golden-re-pin risk |
| PL17 | Admissibility verifier `check_admissible` | MISSING | `groups.py:49` only string-membership; inline test math at `test_gauge_groups.py:155` not factored into a callable |
| PL15 | Route live SPD paths through `safe_spd_inverse`/`floor_eigenvalues` | PARTIAL | `safe_cholesky` is live (`gaussian.py`); the other two stay orphaned — the numerics audit concluded no live site is routable without changing the numerical result |
| PL18 | Observation-likelihood seam | MISSING (decided) | `free_energy.py:268` is a documented dead gated-stub; commit 393e7a5 chose to keep it a stub (wiring would double-count CE) |
| PL19-surrogate | Surrogate end-to-end test | PARTIAL | wiring spied (`test_fix_model_audit.py:101`); no e_step/model test asserts F differs by exactly the entropy term |
| PL19-USU | U(K)/SU(K) groups | MISSING | needs an anti-Hermitian/complex basis; low value |
| PL19-alpha-bayesian | Fully-Bayesian alpha (Gamma posterior) | MISSING | only Gamma-MAP forms; threading a 2-param posterior through F is large, low value |
| M5 | Norm registry `(mu,sigma)` contract | MISSING | `norms.py:69-72` mean-only; blocks SPD/whitening norms |
| M6 | `b0`/`c0` accept `float\|Sequence[float]` | ADDRESSED (2026-06-08) | `cfg.b0`/`cfg.c0` accept `float\|list[float]` (len `K`, validated); converted to a `(K,)` tensor via `_as_coeff` at every consumption site (block/model/viz); `list[float]` is json-serializable. Test `test_cheap_ledger_wins.py` |
| M7 | Decode registry blind to `alpha_div` | MISSING | `prior_bank.py` hardcodes alpha=1 KL readout |
| M8 | `transport_covariance` cov_action seam | MISSING (Phase-5) | single hardcoded sandwich `transport.py:412` |
| T1 | Per-head `kappa_a` scalar | ADDRESSED (2026-06-08) | `cfg.kappa`/`cfg.kappa_gamma` accept `float\|list[float]` (len `n_heads`, block_glk only); `free_energy._broadcast_tau` reshapes the `(H,)` tau to `(H,1,1)` at the softmax sites; list converted at every `attention_tau` site. Default scalar byte-identical. Test `test_cheap_ledger_wins.py` |
| T2 | `kappa` learnable | PARTIAL (doc) | fixed scalar, faithful to Alg 1; rationale undocumented |
| T3 | ALiBi per-head slope schedule | ADDRESSED (2026-06-08) | `prior_alibi`/`prior_causal_alibi` return `(H,N,N)` with the Press geometric slope `2^(-8(h+1)/H)`; config `alibi_slope`; `_attention_log_prior` passes `n_heads`. Default `causal` unaffected. Test `test_cheap_ledger_wins.py` |
| TierD-sandwich | Diagonal-sandwich config/call warn | PARTIAL | docstring only; no config warn |
| TierD-kappa | `kappa_max` condition-number cap | MISSING | monitor-only; `[eps,sigma_max]` clamp bounds it incidentally |
| TierD-expM | exp(M) clamp activation monitor | MISSING (by choice) | hot-path host-sync deliberately avoided; cannot fire on default path |
| TierD-son | SO(N) principal-ball wrapping | MISSING | `lie_ops.py:303` radial rescale only |
| PL6-3 | Slow-channel s-update in E-step | ADDRESSED (2026-06-08) | opt-in `s_e_step` (default OFF, requires `prior_source='model_channel'`): `model._refine_s` runs a model-channel E-step (frozen `r` self-target, `gamma_coupling` consensus, frame fixed), anchored as the belief prior+init in `forward` so `s` reaches `mu_final` at `n_e_steps=1`; cfg `config.py` (`s_e_step`/`e_s_mu_lr`/`e_s_sigma_lr`); tests `test_live_s_model_channel.py` (default-off byte-identity, T=1 liveness, gradient-to-s-tables, `e_s_lr=0` reduction). Spec `docs/superpowers/specs/2026-06-08-live-s-model-channel-design.md` |
| PL19-estep-conv | E-step convergence diagnostic test | PARTIAL | `estep_residuals` exists; non-increasing-on-default-loop assertion absent |
| PL19-diag-reuse | `diagnostics()` forward reuse flag | MISSING | low impact (~4% of steps) |
| PL19-recognition | Recognition-factor registry | MISSING | two-branch dispatch; roadmap rates speculative |

## Prioritized remaining work

The numerical-hardening pair the 06-01 roadmap flagged as load-bearing turned out resolved
(PL14 done; PL15 intentionally not routed per the numerics audit), which leaves checkpoint
resume as the single high-value survivor. The rest is contained expandability and test-coverage
hygiene. Ranked by value over effort:

1. **PL8 checkpoint resume (load side)** — high. Multi-hour 5090 runs cannot currently survive
   an interruption.
2. **PL13-priors: T5 relative-bias and windowed attention priors** — medium. Each is a small
   `register_prior` addition completing the positional-prior axis the spec enumerates.
3. **PL17 admissibility verifier `check_admissible`** — medium. The pushforward-invariance math
   is already proven inline in a test; factoring it into a callable turns the family-to-group
   bridge from a maintained string convention into a verified invariant.
4. **PL19 surrogate end-to-end test** — medium (test only). Guards the standard-transformer
   training baseline through the real stack.
5. Cheap contained wins (S each): M6 (`b0`/`c0` sequence config, the kernel already supports
   it), T1 (per-head `kappa_a`), T3 (ALiBi per-head slope), TierD-sandwich config warn.

Explicitly *not* recommended for unsupervised work: PL10 causal-packed transport (an L-effort
GPU optimization whose correctness rests on re-pinning catastrophic-cancellation-sensitive
goldens), U(K)/SU(K) (needs a complex representation), and the fully-Bayesian alpha (needs a
two-parameter posterior threaded through F). All preserve no pure-path concern but carry high
overnight-failure risk for low marginal value.

## Implemented this session

Each feature below is an opt-in toggle, default OFF, with the pure path preserved, built TDD with
per-item commits. See `docs/edits/2026-06-07-deep-audit-fixes.md` for the running change record.

1. **PL8 checkpoint resume (load side)** — `run_artifacts.load_checkpoint` restores model + AdamW
   momentum + RNG; `train(resume_from=...)` (or `cfg.resume_from`) rebuilds the cosine `LambdaLR`
   at the saved step and continues `range(start_step, n_steps)`. The gold test pins that a straight
   run equals (train → checkpoint → resume) bit-for-bit under a constant stream. A follow-up test
   on the geometric M-step (`m_phi_natural_grad=True`) caught a real bug — `GaugeNaturalGradAdamW`'s
   `gauge_mom`-only state crashed `Adam.__setstate__`'s `KeyError: 'step'` on resume — fixed by a
   `__setstate__` override, so resume is verified for both AdamW and the gauge optimizer.
2. **PL13-priors T5 + windowed attention priors** — `windowed`, `causal_windowed`, and
   `t5_relative_bias` (faithful T5 bucketing, optional learnable per-bucket handle) registered;
   config-selectable via the live `_PRIORS` validation with no call-site edit.
3. **PL17 admissibility verifier** — `groups.check_admissible(group, family)` turns
   `invariant_for`'s string declaration into a verified invariant (full Gaussian invariant for every
   registered group; diagonal Gaussian correctly fails under a general GL(K) congruence).
4. **PL19 surrogate end-to-end test** — `include_attention_entropy=False` exercised through VFEModel
   (oracle-branch forward+backward, E-step descent change, exact F gate), closing the
   local-closure-only coverage gap. Test-only.

The remaining gaps are all low value (M5–M8, T1–T3, the Tier-D hygiene items, U/SU groups, the
fully-Bayesian alpha, the recognition-factor registry, M7's decode KL-pin) or carry high
unsupervised-failure risk for their value (PL10 causal-packed transport's golden re-pin). M6
(`b0`/`c0` sequence config) was deliberately deferred: the per-coordinate kernel already accepts a
`(K,)` tensor, but a tensor-valued config field is against the codebase grain (no precedent, breaks
`asdict`→`config.json` serialization), so it is left as documented-contained work rather than added
unsupervised. The pure path is preserved across every change; all 690 tests pass (664 baseline + 26 new).
