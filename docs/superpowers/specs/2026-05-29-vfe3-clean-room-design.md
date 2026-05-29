# VFE_3.0 (V3_Transformer) — Clean-Room Design Spec

Date: 2026-05-29
Status: DRAFT — awaiting user review before implementation planning
Source project: `C:\Users\chris and christine\Desktop\VFE_2.0` (the `/vfe` main path)
Target repo: `C:\Users\chris and christine\Desktop\V3_Transformer` (GitHub: cdenn016/V3_Transformer)

## 1. Motivation

VFE_2.0's `/vfe` core works but is structurally tangled, and the tangle is real rather than cosmetic. A three-agent investigation plus an independent verifier established the following facts about the current `transformer/vfe/` and `transformer/core/` code (file:line citations are against VFE_2.0 as of 2026-05-29):

The free energy functional F has no single authoritative definition. In the production hot path F exists only as gradients; the analytic kernels return `(grad_mu, grad_sigma)` and never form the scalar F. The only place a scalar F is materialized is the diagnostic `_f_monotone_step` (`vfe/e_step.py:131`), which is gated off by default (called only under `monitor_monotonicity` or `track_layer_diagnostics`, both default `False`).

Gaussian KL is reimplemented at least four independent times: `core/kl_computation.py` (`_kl_kernel_dense:122`, `_kl_kernel_diagonal:365`), `core/gauge_utils.py` fused kernels (`fused_block_diagonal_kl_diag:355`, `fused_block_diagonal_kl_full:567`), and two separate `vfe/prior_bank.py` decode implementations (diagonal fused-matmul `:824-836`, exact full-cov Cholesky `:700-737`). The function that bills itself as "the single parametric entry point" (`kl_computation.py:557`) is bypassed by both the decode path and the E-step gradient path.

The canonical-versus-surrogate free energy is not two expressions but one term's presence: the attention-distribution entropy `tau * beta * log(beta / pi)`. The toggle is a single config flag `include_attention_entropy` (`config.py:246`, default `True`). In the mean/covariance path the canonical form is realized through the envelope identity by passing `lambda_softmax = 0` (`e_step.py:1004, 1068, 1161, 1220`), so canonical and surrogate run the same kernel with a different lambda; only the phi-update path materializes the entropy term explicitly, where it is quintuplicated.

The hand-derived gradient kernels are already centralized in one file (`core/vfe_gradients.py`) as three distinct (mu, sigma) kernels — `_compute_vfe_gradients_block_diagonal` (full cov, `:168`), `_compute_vfe_gradients_block_diagonal_diag` (diagonal, `:746`), and `_fused_attention_and_vfe_gradients_block_diag` (fused diagonal, the primary live path, `:1115`) — but the diagonal and fused kernels duplicate identical math (self-coupling at `:837-838` is byte-identical to `:1240-1241`), and the fused kernel triplicates its own per-pair formulas inside the causal-triangle branch. The phi gradient is never hand-derived; it is always `torch.autograd.grad` on a scalar alignment loss, then preconditioned (`_apply_phi_retraction`, `e_step.py:1336-1369`).

There is no natural-parameter, log-partition, sufficient-statistic, or differential-entropy abstraction anywhere in the path; every divergence is written in Gaussian moment form (trace, Mahalanobis, log-det).

The belief representation is a thin transport container, not an object with behavior: `BeliefState` (`core/types.py:13`) is a `NamedTuple(mu, sigma, phi, omega?)` used only at function boundaries and immediately destructured into separate positional tensors that flow through the entire math layer.

The user's stated goal is a clean, production-quality implementation; 2.0 is described as "VERY messy, tangled, and confusing." Because the tangle is structural (no single F, divergence reimplemented many times, gauge and divergence concerns fused), it cannot be fully removed by editing in place without effectively rewriting the core. The chosen strategy is therefore a clean-room rebuild in a new repo, porting verified math kernels one at a time under golden numerical-equivalence tests, with 2.0 left running untouched until 3.0 reaches parity.

## 2. Goals and non-goals

Goals. A single authoritative scalar definition of F, canonical with a surrogate toggle (original idea 1). All belief-gradient math derived from or verified against that single F (original idea 2). An exponential-family abstraction housing the multivariate Gaussian first but structured to admit other families at the divergence layer (original idea 3). A clean separation between the family-agnostic divergence layer and the Gaussian-specific gauge layer. Numerical parity with VFE_2.0's `/vfe` path, proven kernel-by-kernel by golden tests. High modularity so features swap in and out by config without editing call sites (registry behind every seam; see section 4). Production code quality: typed signatures, single home per formula, the project's argument-ordering and alignment conventions, finite-difference gradient checks as first-class tests.

Non-goals for this effort. The `coupled_fep` and `pure_fep` subpackages are out of scope (not investigated, not ported). The hyper-prior `lambda_h * KL(s||h)` and model-coupling `gamma * KL(s_i || Omega s_j)` terms are absent from 2.0's main path and are not ported now; they appear in the design only as named extension points. Generalizing the gauge action to non-Gaussian families is explicitly deferred to a Phase 5 investigation, not assumed.

## 3. Honest scope boundary on the exponential-family idea

The divergence values generalize cleanly. For any exponential family with natural parameter theta and log-partition A(theta), KL is the Bregman divergence of A, and the Renyi/alpha-divergence has the closed form `R_alpha = 1/(alpha-1) * [A(alpha*theta1 + (1-alpha)*theta2) - alpha*A(theta1) - (1-alpha)*A(theta2)]`, valid whenever the interpolated parameter stays in the natural-parameter domain. The codebase already exhibits this domain boundary as a runtime artifact: the `sigma_blend = (1-alpha)*sigma_q + alpha*sigma_t` clamp guarding `alpha > 1` (`kl_computation.py:397, 441-443`) is exactly the Gaussian precision leaving the natural-parameter domain.

The gauge core does not generalize. GL(K) acts on Gaussians because the family is closed under affine maps on the sample space (a location-scale / elliptical structure). The action has a natural-parameter form for the Gaussian, but families such as the categorical, product-Poisson, or gamma admit no GL(K) action preserving the family. Therefore `Omega_ij = exp(phi_i) exp(-phi_j)`, the sandwich transport `Sigma -> Omega Sigma Omega^T`, the SPD retraction, RoPE-on-mu, and MahalanobisNorm are tied to the Gaussian's location-scale structure, not to exponential families in general.

The consequence for the design is the central architectural decision: the `families` layer is family-agnostic and is the seam an exp-family abstraction lives behind; the `geometry` layer is explicitly Gaussian-specialized and walled off. Whether any non-Gaussian family deserves a gauge hook is a research question answered in Phase 5, before the family/geometry boundary is frozen.

## 4. Architecture

The package is built strictly bottom-up; build order equals dependency order, and each layer is golden-tested against VFE_2.0 before the next is built. Proposed package root: `vfe3/`.

Modularity is a first-class requirement: features must swap in and out easily. Every seam exposes a small, stable interface plus a config-selected registry, so a new variant is added by writing it and registering it, never by editing call sites. The intended swap points are the divergence (`divergence.py`: KL, Renyi, future divergences), the self-coupling coefficient (`alpha_i.py`: constant, learnable, Bayesian per-dimension), the exponential family (`families/`: Gaussian first, others behind the same interface), the transport and gauge variant (`geometry/`: flat, non-flat connection, RoPE on or off), the retraction (`geometry/`: SPD exp-map, diagonal exponential, phi Lie-algebra), the attention coefficient and temperature, and the decode head (PriorBank versus linear projection). The free energy is canonical-versus-surrogate by a single toggle. Each registry entry is selected from `config.py` by name, and the golden and finite-difference tests run per registered variant, so adding a feature cannot silently break an existing one. This keeps the layers loosely coupled: a caller depends on a seam's interface, not on which concrete variant is active.

### 4.1 `divergence.py` — the divergence seam, and `families/` — the exponential-family parameter layer

`divergence.py` is the single file every caller imports its divergence from (for example `from divergence import kl, renyi`). The Renyi/alpha-divergence is the primitive and KL is its `alpha = 1` special case, matching 2.0 where every divergence is the Renyi closed form and KL is recovered at `alpha = 1`. This one file replaces the four independent Gaussian-KL implementations enumerated in section 1. In the Gaussian-only stage the Renyi closed forms (diagonal and full covariance, including the `alpha > 1` blended-covariance domain clamp) live here, ported and verified from 2.0's `kl_computation.py`. When the exponential-family abstraction lands (idea 3) the per-family closed forms move behind the `ExponentialFamily` interface in `families/` and `divergence.py` dispatches to them, but `divergence.py` remains the single public entry point its callers see at every stage.

`families/base.py` defines the `ExponentialFamily` interface distilled from the audit's section 4: conversion between natural and moment parameters, `log_partition(theta)`, and `entropy(theta)`. These are pure functions of distribution parameters with no gauge and no transport. `families/gaussian.py` provides the `DiagonalGaussian` and `FullGaussian` parameter representations. This layer is the dispatch target `divergence.py` uses for non-Gaussian families; the Gaussian moment forms are kept for exact numerical fidelity to 2.0, and the interface is structured so that a future A(theta) form can back the same divergences without changing callers. Gaussian is the only concrete family until Phase 5 decides whether to add others.

### 4.2 `geometry/` — gauge, transport, and manifold layer (Gaussian-specialized)

`geometry/gauge.py`: `Omega_ij = exp(phi_i) exp(-phi_j)`, the transport action `mu -> Omega mu` and `Sigma -> Omega Sigma Omega^T`, RoPE-on-mu, and the non-flat connection variant. Ported from 2.0's `core/transport_ops.py`.

`geometry/retraction.py`: the SPD exponential-map retraction (diagonal and full), the phi Lie-algebra retraction, and the Fisher / natural-gradient preconditioner. Ported from `core/vfe_utils.py`, `core/phi_evolution.py`, `core/gauge_preconditioner.py`. This layer is cleanly isolated so the Phase 5 investigation can decide whether it acquires a per-family hook or remains Gaussian-only.

### 4.3 `free_energy.py` — the single authoritative scalar F, and `alpha_i.py` — the self-coupling coefficient

The total free energy is the sum over positions of a per-position functional, `F = sum_i F_i`, with

```
F_i = alpha_i * KL(q_i || p_i)                               # self-coupling, per-position alpha_i
    + sum_j [ beta_ij * KL(q_i || Omega_ij q_j) ]           # belief coupling (surrogate core)
    + sum_j [ tau * beta_ij * log(beta_ij / pi_ij) ]        # attention entropy (canonical only)
    - E_q[log p(o_i | x)]                                    # observation likelihood
```

where every `KL` is the Renyi divergence supplied by `divergence.py` (KL is its `alpha = 1` case), and `tau = kappa * sqrt(K)`. `include_attention_entropy` toggles canonical (entropy term present) versus surrogate (absent); this is one function in which exactly one term differs between the two forms, replacing 2.0's scattering of the entropy term across five sites. The hyper-prior and model-coupling terms appear as clearly named, unimplemented extension points so they are never half-wired.

The self-coupling coefficient `alpha_i` is general (per-position, and per-dimension where applicable, generalizing 2.0's per-dimension Bayesian alpha). Its forms — constant, learnable, Bayesian per-dimension, and any future variant — live with their gradients in `alpha_i.py`. `free_energy.py` and the gradient layer obtain `alpha_i` (and its gradient contribution, including the learnable-alpha product-rule correction that 2.0 scattered across `_alpha_product_rule_diag_kl`/`_renyi` in `vfe_gradients.py`) from this single module. The active form is selected by config, so alpha variants are swapped without touching `free_energy.py` or the kernels.

### 4.4 `gradients/` — gradient oracle and optimized kernels

`gradients/oracle.py`: the reference gradient is `torch.autograd.grad(free_energy(...), params)`. This is the correctness source of truth.

`gradients/kernels.py`: the optimized hand-derived (mu, sigma) kernels ported from `core/vfe_gradients.py` (diagonal, fused diagonal, full covariance, RoPE full gauge). Each is pinned by both a finite-difference check against the oracle and a golden test against 2.0. The phi gradient remains autograd, as in 2.0. The diagonal-pair math is factored once, removing the `837-838 == 1240-1241` duplication and the causal-triangle triplication. The self-coupling alpha gradient (and its learnable-alpha product-rule correction) is supplied by `alpha_i.py`, not reimplemented here; the kernels consume it.

This realizes the "oracle plus dedup everything" decision: F is the specification, the oracle is the correctness reference, the kernels are the optimized implementation, and the three are pinned together by tests rather than drifting independently.

### 4.5 `inference/e_step.py` — the E-step

A clean iterative belief-update loop reading `gradients` and `geometry`, with decoupled learning rates (`e_mu_lr`, `e_sigma_lr`, `e_phi_lr`) and the trust-region clamp (`e_sigma_q_trust`). The `forward()` that 2.0 had to decompose after the fact is structured cleanly from the start.

### 4.6 `model/` — model, blocks, and decode

`model/prior_bank.py`: PriorBank decode as `logits = -KL(q || pi_v) / tau` using the `families` KL, removing the independent decode-KL reimplementations. `model/block.py`, `model/stack.py`, `model/model.py`: clean assembly with the belief handoff `mu_q -> mu_p` across blocks. MahalanobisNorm lives in a small `norms` module within `geometry` or alongside it.

### 4.7 `config.py`

A single dataclass with one `__post_init__` of validation, no override tangle, and no dead meta-fields mixed into the main config.

### 4.8 `tests/golden/` — the equivalence harness

For each ported kernel, a pinned VFE_2.0 snapshot and the 3.0 implementation are run on identical inputs and asserted equal within float32 tolerance. This is the safety net that makes clean-room porting trustworthy.

## 5. Data flow

embeddings -> `PriorBank.encode` -> `BeliefState(mu, Sigma, phi)` -> E-step iterating `gradients` plus `geometry` retraction -> belief handoff `mu_q -> mu_p` across blocks -> `PriorBank.decode` (`-KL / tau`) -> logits -> cross-entropy.

## 6. Numerics

SPD eigenvalue floors, safe symmetric-positive-definite inverses, and the `alpha > 1` blended-covariance clamp (the natural-parameter-domain boundary) are ported from 2.0's `_numerics.py` and `vfe_utils.py`, each given a single home in `families` or `geometry` rather than being scattered.

## 7. Testing strategy

Golden equivalence per kernel (3.0 versus a pinned 2.0 snapshot). Finite-difference gradient checks (kernel versus oracle versus numerical), now a first-class requirement per CLAUDE.md. Property tests: gauge equivariance (transport then divergence equals divergence then transport), canonical-F stationarity (softmax beta is a stationary point of the canonical F), and the canonical-minus-surrogate gradient gap equal to `-tau^{-1} Cov_beta(KL, grad KL)`.

## 8. Phase plan

Each phase is gated by passing its golden and property tests before the next begins.

Phase 0. Scaffold the V3_Transformer repo (`git init`, wire the GitHub remote, package skeleton, config dataclass) and build the golden-equivalence harness against a pinned VFE_2.0 snapshot. This is the first implementation step and requires user go-ahead.

Phase 1. `divergence.py` (Renyi primitive, KL as `alpha = 1`) plus the `families` Gaussian parameter representation (entropy, log-partition) and the divergence registry seam. Golden against 2.0's `kl_computation`, `gauge_utils` fused kernels, and the two `prior_bank` decode KLs.

Phase 2. `geometry`: gauge, transport, SPD and phi retraction, RoPE. Golden against 2.0's `transport_ops` and `vfe_utils`. Property: equivariance.

Phase 3. `free_energy.py` (the single scalar F, `F = sum_i F_i`) and `alpha_i.py` (the self-coupling coefficient forms). Golden against 2.0's `_f_monotone_step` and the implied F of the analytic kernels.

Phase 4. `gradients`: oracle plus ported kernels, together with the `alpha_i.py` gradients (including the learnable-alpha product-rule correction), all finite-difference and golden tested.

Phase 5. Gauge-generalization theory investigation: determine whether any non-Gaussian exponential family admits a useful gauge action (for example a natural-parameter affine action), then finalize the `families`/`geometry` boundary and the exp-family extension decision.

Phase 6. `inference` E-step. Golden against 2.0's `e_step` on fixed seeds.

Phase 7. `model`, decode, and config. Full-model parity: 3.0 versus 2.0 training curves on a fixed seed match within tolerance.

Phase 8. Cutover criteria and documentation.

## 9. Risks and open questions

The pinned-2.0-snapshot dependency for golden tests must import VFE_2.0 code; the harness needs a stable path to a 2.0 checkout (sibling directory) and should pin a specific commit so equivalence is reproducible. Full-covariance and RoPE-full-gauge paths in 2.0 are opt-in and less exercised; their golden tests may surface latent 2.0 behavior that must be matched deliberately or consciously corrected (any intentional divergence from 2.0 must be recorded). The Phase 5 investigation may conclude that no non-Gaussian family is worth a gauge hook, in which case the `geometry` layer stays Gaussian-only and the exp-family abstraction is confined to the divergence layer; the design already assumes this as the default.

## 10. Cutover criteria

VFE_3.0 replaces VFE_2.0's `/vfe` path when: all ported kernels pass golden equivalence within tolerance; finite-difference gradient checks pass for every kernel against the oracle; a full training run reproduces 2.0's loss curve on a fixed seed within tolerance; and the exp-family and gauge boundaries are finalized per Phase 5.
