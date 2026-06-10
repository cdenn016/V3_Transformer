# Deep Audit — 2026-06-09 (overnight run)

## Scope

Full repository on branch `vfe3-deep-audit-2026-06-09` as of commit ea8a996 plus the user's
uncommitted working-tree edits (`ablation.py`, `train_vfe3.py`, manuscript tex): all of `vfe3/`,
the entry points `train_vfe3.py`, `ablation.py`, `make_figures.py`, with `Manuscripts-Theory/`
as theory ground truth and `tests/` as a secondary surface. The user's charge: identify any and
all shortcomings, places for improvement, speedups, errors, and optimizations, judging purity
solely by whether theoretically pure paths exist under toggles (config defaults explicitly out
of scope), then fix what does not require an explicit user decision. This run complements the
same-day morning audit (`docs/audits/audit-2026-06-09.md`, 49 findings, punch list fixed in
commits 711af1a..ea8a996); investigators were instructed to read it first and target primarily
the code that landed after it — the Clebsch-Gordan solver (`geometry/cg.py`), the irrep towers
(`irreps.py`, `groups.py`, `generators.py`), the isotypic head mixer, and the `CGCoupling`
module — plus anything the morning sweep missed. Unlike the morning run, this run executed the
adversarial challenge tier and the test suite.

## Run record

The audit ran as a background workflow (12 investigators, then batched verification). Four of
the five base investigators were killed mid-flight by an account session-usage limit
("session limit, resets 2:40am America/Chicago"); the workflow stalled from roughly 22:06 to
02:40 and resumed when the limit reset. The four lost lenses (code-reviewer, debugger,
refactoring-specialist, python-pro) were re-dispatched as standalone agents running concurrently
with the expert wave, and their findings were verified by three dedicated fresh-context
verifiers under the same source-only rules as the in-workflow batch. The first replacement
refactoring agent exhausted its budget mid-run; its only in-flight claim (that the sp(2m)
symmetric towers are reducible) was refuted by an orchestrator-executed Schur-commutant probe
(see Adversarial Challenge) and the lens was re-run to completion. The claude-mem plugin worker
was down for the whole session (environmental; observations were persisted to disk instead).

## Investigators Dispatched

- Base five: `code-reviewer`, `debugger`, `refactoring-specialist`, `performance-engineer`,
  `python-pro` (performance-engineer completed inside the workflow; the other four as
  standalone replacements after the session-limit kill).
- Experts (all seven — the theory-invariant gate is met and the scope is the whole repo, so the
  entire `audit-*` pool is on-scope): `audit-gauge-theorist`, `audit-geometer`,
  `audit-info-geometer`, `audit-variational`, `audit-numerical-analyst`,
  `audit-transformer-ml`, `audit-implementation-engineer`.
- Verification: five fresh-context batched verifiers inside the workflow (8 findings each) plus
  three for the replacement wave — source-only evidence, mandatory `path:line` citations,
  reachability notes, and the out-of-scope rule for default-impurity claims.
- Challenge tier: `audit-skeptic` + `audit-defender` duels on the two escalated findings (no
  finding survived verification at critical or high, so both duels were discretionary
  escalations: one contested theory finding, one verifier-downgraded high).

Totals: 60 findings filed, 58 CONFIRMED, 2 REFUTED, 0 INCONCLUSIVE. Post-verification severity:
0 critical, 0 high, 10 medium, 48 low. Verifiers materially corrected eleven findings rather
than rubber-stamping (severity changes on F1, F2, F3, F6, F24, F31, DB1, PP1, PP2, PP3, RF1),
and the challenge tier resolved one direct contradiction between two experts (F19, below).

## Investigator Findings

Severities shown as `original→verified` where the verifier corrected them. Reach legend:
**default** = live under default config; **toggle** = needs non-default toggle(s); **latent** =
unreachable from any current config; **diag** = diagnostics/figures only.

### Workflow wave (performance-engineer + seven experts)

#### performance-engineer (base, ran inside workflow) — 9 finding(s)

| # | Sev (verified) | Verdict | Reach | Finding | Location |
|---|---|---|---|---|---|
| F1 | high→low | CONFIRMED | toggle | CGCoupling forward runs all P path iterations at zero init with no identity short-circuit | `vfe3/model/cg_coupling.py:109` |
| F2 | high→medium | CONFIRMED | toggle | CGCoupling forward: 40 sequential per-path einsums when all paths share one intertwiner | `vfe3/model/cg_coupling.py:110` |
| F3 | high→medium | CONFIRMED | default | _blockwise_matrix_exp: H Python-loop slice reads (torch.stack) + H slice writes around one batched exp | `vfe3/geometry/transport.py:286` |
| F4 | medium | CONFIRMED | default | _factored_diagonal_covariance materializes rank-5 (B,N,d,d,d) intermediate per head at large d | `vfe3/geometry/transport.py:488` |
| F5 | medium | CONFIRMED | toggle | HeadMixer._dense_m rematerializes the full (K,K) commutant matrix on every full-covariance forward | `vfe3/model/head_mixer.py:152` |
| F6 | medium→low | CONFIRMED | toggle | cg_selection enumerates all n_unique^2*(n_unique+1)/2 CG triples at CGCoupling init including zero-multiplicity ones | `vfe3/geometry/cg.py:112` |
| F7 | low | CONFIRMED | toggle | attention_tau allocates a new CPU tensor every vfe_block call for unequal irrep_dims (so_n/sp_n) | `vfe3/free_energy.py:60` |
| F8 | low | CONFIRMED | diag | free_energy() allocates torch.full_like(beta, 1/N) for the uniform pi on the log_prior=None branch | `vfe3/free_energy.py:351` |
| F9 | low | CONFIRMED | toggle | CG intertwiner clone on every cache hit, then immediate register_buffer (two copies at init) | `vfe3/geometry/cg.py:61` |

Clean areas: The morning audit's 12-point punch-list items are substantially addressed in current source. The E-step hot path (vfe3/inference/e_step.py) is clean on the performance lens: `build_belief_transport` correctly dispatches to `FactoredTransport` on the default flat + equal-block path (PE1/PE3 from the morning audit are fixed), the phi-step autograd island is correctly scoped under `enable_grad`, and the truncation-warning for the oracle non-kernel path is now emitted at runtime rather than only at config time. The `free_energy` computation is correcty vectorized: `pairwise_energy` uses a single stacked functional call for equal-block groups, and the attention entropy uses `reduced_free_energy` (logsumexp envelope) on the canonical path, avoiding a redundant beta materialization. The `_CG_CACHE` is process-global and invalidated correctly on device/dtype moves via `CGCoupling._apply`. The float64 CG intertwiner storage with lazy cast to runtime dtype (the `_cast_buffers` pattern) is sound: the `.to(dtype=mu.dtype)` on `path_weights` is a no-op identity return for float32 (confirmed by pointer equality). The `stable_matrix_exp_pair` float64-island correctly bypasses AMP with `torch.amp.autocast('cuda', enabled=False)`. The `transport_mean` and `transport_covariance` factored-path einsums are algebraically exact reassociations with no precision loss. The `HeadMixer` diagonal-covariance path (the default) is fast at 0.8 ms vs 36 ms full-cov, and it correctly avoids `_dense_m` when `sigma.dim() == mu.dim()`. The `is_identity` helper in both `CGCoupling` and `HeadMixer` is not called anywhere in the forward/E-step hot path — the `.item()` CPU sync is confined to diagnostic/logging callers. The `generate_son`/`generate_sp`/`direct_sum_generators` build-time verification (bracket homomorphism assert) fires only at construction and is correctly gated on a raise-not-warn policy. The `cg_selection` and `cg_intertwiners` caches are persistent for the process lifetime and survive repeated CGCoupling instantiations within one training run.

#### audit-gauge-theorist — 3 finding(s)

| # | Sev (verified) | Verdict | Reach | Finding | Location |
|---|---|---|---|---|---|
| F10 | low | CONFIRMED | toggle | CGCoupling builds dead zero-gradient antisymmetric self-pairing paths | `vfe3/model/cg_coupling.py:67-73` |
| F11 | low | CONFIRMED | toggle | CGCoupling/head_mixer modify mu but leave Sigma un-updated, so the post-coupling belief tuple is not an affine-coherent congruence image | `vfe3/model/cg_coupling.py:116` |
| F12 | low | CONFIRMED | toggle | CG multiplicity counting rests on an absolute atol with no runtime gap monitor | `vfe3/geometry/cg.py:88` |

Clean areas: I swept the post-morning-audit gauge code: cg.py, irreps.py, groups.py, generators.py, transport.py, model/head_mixer.py, model/cg_coupling.py, plus their config/model/block/stack/optimizer wiring, with executed probes throughout (manuscripts as ground truth, not docstrings).

CG intertwiner machinery is correct from first principles. The numerical solver reproduces the textbook angular-momentum series l1xl1 = l0+l1+l2 with correct symmetry types (l0,l2 symmetric, l1 antisymmetric), gives so(4)-correct selection rules (l2xl2->l1 = 0 mult, distinct from the so(3) tower), and the build-time equivariance assert holds. CGCoupling is exactly equivariant under the embedded SO(3)/Sp action with random nonzero path weights (||f(g.mu)-g.f(mu)|| = 3.3e-15), and its means-update gradient to path_weights is FD-correct (1.5e-10) even under multiple paths writing the same target slice; fp32 vs fp64 forward agree to 4.5e-8. The a<=b source canonicalization loses no expressivity (swapped distinct-label pairings are proportional; same-label swaps reachable by the free scalar weight sign).

Generator algebras close: all sp(2m) generators satisfy JA+A^TJ=0 exactly, so(N) embedded towers are exactly skew (residual 2e-16, exp(-M)=exp(M)^T fast path and det=1 verified), sp_n correctly non-skew, and the direct-sum tower bracket-homomorphism passes. The isotypic head-mixer's "full linear commutant for real-type irreps" claim is verified: every probed so(3) (l1,l2,l3) and sp(4) (sym1,sym2,sym3) irrep has commutant dimension exactly 1 (real type), so kron(A,I) is the full commutant; the full-cov mixer is exactly equivariant under the tied tower gauge (5e-15) and breaks under untied block_glk exactly as documented.

Transport for the new towers is correct: the phi-cocycle holds on the unequal-dim [1,3,5] so_n tower (Omega_ij Omega_jk = Omega_ik to 3e-7, Omega_ii=I, Omega_ij Omega_ji=I), and the factored diagonal-covariance/mean fast paths match the dense sandwich to machine epsilon even with unequal block dims.

Wiring is sound: config rejects use_cg_coupling off so_n/sp_n and rejects per-block phi preconditioners under the tied tower gauge; CGCoupling/head_mixer are applied per-block after the E-step (forward feature maps, not scored by F, with the M-step self-coupling reading the pre-mix belief); the optimizer exact-coverage guard groups path_weights and head_mixer.parameters(); zero-init gives byte-identical step-0 output; and the CG/mixer test suite covers nonzero-weight equivariance, selection rules, cost guard, cache immunity, sigma passthrough, and e2e training. The pure no-NN flat path remains the default with all three of these as opt-in default-off exceptions. (Note: PostToolUse claude-mem hook errors throughout were environmental and did not affect any probe; all numbers quoted are from executed tool output.)

#### audit-geometer — 3 finding(s)

| # | Sev (verified) | Verdict | Reach | Finding | Location |
|---|---|---|---|---|---|
| F13 | low | CONFIRMED | toggle | CGCoupling enumerates dead self-product paths for antisymmetric intertwiners (l1xl1->l1 etc.) | `vfe3/model/cg_coupling.py:67-73` |
| F14 | low | CONFIRMED | toggle | CGCoupling.forward runs the full per-path Python loop every block/forward with no is_identity short-circuit | `vfe3/model/cg_coupling.py:109-116` |
| F15 | low | CONFIRMED | toggle | No test asserts CGCoupling has no structurally dead path weights | `tests/test_cg.py:82-155` |

Clean areas: I re-verified the morning audit's CLEAN differential-geometry verdict against current source with executed probes and it holds. The affine-invariant SPD retraction (retraction.py) matches the canonical Sigma^{1/2} expm(Sigma^{-1/2} dSigma Sigma^{-1/2}) Sigma^{1/2} to rel err ~1e-6, satisfies R(S,0)=S to 1e-6 for eigenvalues within the documented [eps, sigma_max] ceiling, and keeps Sigma SPD (min eig clamped at eps). The affine-invariant geodesic distance is congruence-invariant to 2e-14 (Bhatia 2007 Ch.6), confirming the retraction implements the correct metric. Covariance transport (transport.py) is exactly the congruence/sandwich Omega Sigma Omega^T (residual 0.0 vs the dense einsum, and differs from similarity Omega Sigma Omega^-1 by ~21.7 -- it is NOT similarity); the flat phi-cocycle holds (Omega_ii=I to 3.6e-7, Omega_ij Omega_ji=I to 7.2e-7). The Fisher natural gradient (natural_gradient) gives nat_mu=Sigma grad_mu and nat_sigma=2 Sigma grad_sigma Sigma to ~1e-5. New irrep-tower code is sound: so_n embedded generators are exactly skew (|G+G^T|<3e-16), the build-time bracket-homomorphism and dimension-formula asserts (irreps.py) pass for the so/sp towers I exercised, the sp(2m) defining generators satisfy J A + A^T J = 0 (morning audit, re-read), and the numerical Clebsch-Gordan solver (cg.py) returns intertwiners with equivariance residual <1e-7 (its own assert) for the so and sp triples I built. CGCoupling is genuinely means-only (sigma returned as the same object, torch.equal True), exactly equivariant for nonzero weights (residual 9e-16 under a random SO(3) tower gauge), and autograd-safe through the in-place delta-slice accumulation (mu and weight grads finite). The isotypic HeadMixer diagonal closed form sigma'[m]=sum_n A[m,n]^2 sigma[n] equals the full M Sigma M^T sandwich diagonal to machine epsilon and preserves SPD-ness. The phi preconditioners (Killing/pullback in phi_preconditioner.py) and Lie-algebra retraction primitives (lie_ops.py, closure.py, norms.py MahalanobisNorm gauge-invariance) read clean and were not contradicted by any probe. 43 geometry tests (test_cg, test_head_mixer_isotypic, test_son_irreps) pass. NOTE: the claude-mem plugin hook failed on every tool call this session (worker unreachable) -- a harness nuisance, not a repo defect; Read was blocked but Bash cat/sed gave full file contents.

#### audit-info-geometer — 3 finding(s)

| # | Sev (verified) | Verdict | Reach | Finding | Location |
|---|---|---|---|---|---|
| F16 | low | CONFIRMED | toggle | CGCoupling runs the full bilinear path loop every E-step even at zero-init (no is_identity early-out) | `vfe3/model/cg_coupling.py:99-116` |
| F17 | low | CONFIRMED | latent | is_identity() on CGCoupling and HeadMixer is dead code (no source caller) | `vfe3/model/cg_coupling.py:99; vfe3/model/head_mixer.py:117` |
| F18 | low | CONFIRMED | diag | No gradcheck pins the correctness (not just finiteness) of CGCoupling's in-place overlapping-slice delta accumulation | `tests/test_cg.py:155-172` |

Clean areas: I swept the information-geometry surface and verified the core numerically rather than by reading prose. Divergence closed forms (vfe3/families/gaussian.py, base.py): the diagonal and full Gaussian KL/Renyi closed forms match the generic Bregman/Renyi-from-A path to ~5e-7 for alpha in {1, 0.3, 0.5, 0.8}, including the full Gaussian's matrix sufficient-statistic (t2) routed through the generic A-path; self-divergence D(q||q) is exactly 0.0 (full and diagonal); KL >= 0; Renyi -> KL as alpha -> 1. The squared_hellinger f-divergence matches a direct Bhattacharyya-coefficient computation to 4e-9, stays in [0,1), and gives 0 at q==q. The divergence_family seam (renyi, squared_hellinger), the family seam (gaussian_diagonal/gaussian_full), and the alpha_div (Renyi order) vs alpha/value (self-coupling weight) routing stay distinct: the oracle (vfe3/gradients/oracle.py:105,108) threads alpha_div into both self_divergence_for_alpha and pairwise_energy, while the closed-form kernel route is gated to alpha_div==1 only (uses_kernel_route). The Fisher natural-gradient sidedness (vfe3/geometry/retraction.py:323) is correct: nat_mu = Sigma grad_mu, nat_sigma = 2 Sigma grad_sigma Sigma, verified to ~1e-6 (full) and ~1e-8 (diagonal), matching Amari & Nagaoka 2000 section 3.5; it is applied (not at grad) so no second-order term leaks. The new numerical CG intertwiner solver (vfe3/geometry/cg.py) is sound: so(3) l1xl1 = l0+l1+l2, so(4)/sp(4) symmetric products give the right multiplicities, l0xl0->l0 (identically-zero Gram) resolves to n_mult=1, and the build-time equivariance assert is real. CGCoupling (vfe3/model/cg_coupling.py) is exactly equivariant under nonzero weights (residual 7e-16), its in-place delta accumulation passes gradcheck, it touches mu only and passes sigma through, and it is applied after the E-step exactly like the sanctioned head mixer (block.py:76), so it does not distort any F KL term beyond the documented exception-2 footprint. The morning-audit info-geometry items are fixed in current source: IG1 (fisher_trace full-cov inverse) now carries the eps ridge (metrics.py:255), and N1 (pair-term saturation mask) is now applied at kernels.py:218. The full-Gaussian generic-path raising-cholesky (morning N2) remains latent (renyi_closed_form shadows the generic path for both registered functionals).

#### audit-variational — 5 finding(s)

| # | Sev (verified) | Verdict | Reach | Finding | Location |
|---|---|---|---|---|---|
| F19 | medium | CONFIRMED | toggle | M-step self-coupling regularizer reads the POST-mixer/POST-norm belief, not the converged variational q* | `vfe3/model/model.py:556-573 (term), vfe3/model/block.py:73-80 (out is overwritten by mixer/cg/norm)` |
| F20 | low | CONFIRMED | toggle | CGCoupling builds structurally-dead path_weights for self-copy antisymmetric intertwiners (cross(x,x)=0) | `vfe3/model/cg_coupling.py:62-73` |
| F21 | low | CONFIRMED | toggle | CGCoupling forward is an O(n_paths) Python loop of per-path einsum + in-place slice writes, run per block per layer | `vfe3/model/cg_coupling.py:110-115` |
| F22 | low | REFUTED | latent | CG intertwiner buffers are persisted in state_dict while learned path_weights depend on the eigh multiplicity-slot basis | `vfe3/model/cg_coupling.py:56 (register_buffer); vfe3/geometry/cg.py:20-23 (docstring acknowledges basis is eigh-build-dependent)` |
| F23 | low | CONFIRMED | toggle | Head-mixer/CG coupling enter the inter-block belief handoff for all L layers, an unstated placement vs the manuscript's single readout slot | `vfe3/model/block.py:73-78 (applied per block inside vfe_stack), vfe3/model/stack.py:57 (mixed belief folds into next prior mu_p)` |

Clean areas: I verified the core variational structure of the post-audit code against the canonical F and the manuscripts. CGCoupling is exactly gauge-equivariant for nonzero random weights on an SO(3) l1,l1,l2 tower (means residual 4.9e-15 under a tied SO(3) gauge), sigma passes through untouched as documented, its in-place delta accumulation is autograd-safe with overlapping target slots, dtype handling is correct (fp32 out, per-(dtype,device) cast cache), and trivial CG triples (l0xl0->l0, l0xl1->l1) count multiplicity 1 with no over-counting. EM separation is clean for the two new learnable seams: head_mixer.parameters() and cg_coupling.path_weights both land in M-step optimizer groups at m_mu_lr and are covered by the exact-coverage guard in build_optimizer; the E-step itself sees no targets and the mixer/CG run after the E-step fixed point, so belief updates carry no label leakage. The manuscript is explicit (GL(K)_attention.tex:2362-2364) that the head mixer and CG cross-type coupling are NOT generated by the free energy and are tolerated readout augmentations, and both are default-off toggles with a pure no-mixer/no-CG path, so the pure-path-existence constraint holds. The CGCoupling/HeadMixer/son_irreps/mstep_self_coupling test suites pass (48 passed). I re-checked the three morning-audit high items in current source: alibi single-block (model.py:97 guard added, fixed), b0/c0 + state_dependent (config.py:632-655 guard added, fixed), and the V1 gamma-channel include_attention_entropy mismatch (model.py:687-695 now branches canonical-envelope vs surrogate on the toggle, fixed). The free_energy() functional still matches the canonical form term-by-term including the tau*beta*log(beta/pi) attention-entropy term, consistent with the morning audit's stationarity verification. The remaining findings are toggle-gated theory-fidelity and efficiency matters on the new opt-in paths, not pure-path or default-path defects.

#### audit-numerical-analyst — 5 finding(s)

| # | Sev (verified) | Verdict | Reach | Finding | Location |
|---|---|---|---|---|---|
| F24 | medium→low | CONFIRMED | toggle | CG coupling / head mixer have no fp32 autocast island; bf16 breaks the exact-equivariance invariant | `vfe3/model/cg_coupling.py:107-116, vfe3/model/head_mixer.py:130-159` |
| F25 | low | CONFIRMED | toggle | cg_selection raises on a cost-guarded triple during admissibility enumeration, blocking otherwise-valid towers | `vfe3/geometry/cg.py:66-71, 113-119` |
| F26 | low | CONFIRMED | toggle | Non-compact sp_n/sp/glk full-covariance transport reaches cond ~1e7 at the retraction's max_norm, amplifying the Sigma sandwich by ~1e15 | `vfe3/geometry/transport.py:341-349 (compute_transport_operators), vfe3/geometry/lie_ops.py:255-276` |
| F27 | low | REFUTED | latent | CG multiplicity-slot basis is fixed by torch's eigh ordering; persisted path_weights are not portable across eigh orderings | `vfe3/geometry/cg.py:87-89` |
| F28 | low | CONFIRMED | diag | CG build-time equivariance assert runs in float64 (1e-7) but the runtime forward casts to fp32 where the residual reaches ~1e-7 | `vfe3/geometry/cg.py:90-97` |

Clean areas: I swept the post-morning-audit modules (geometry/cg.py, irreps.py, generators.py, groups.py; model/head_mixer.py, cg_coupling.py) plus the priority numerics paths (retraction.py, transport.py matrix_exp, gradients/kernels.py, families/gaussian.py, numerics.py) with executed probes, and the pure paths are numerically solid. CG null-space extraction: the Gram spectral gap is 13+ orders of magnitude (zero eigenvalues at machine-eps, nonzero at ~2.0), so the absolute atol=1e-8 multiplicity cut is well-conditioned for all reachable SO(3/4/5)/Sp(4) towers; multiplicity counts and the build-time equivariance assert behave correctly. The traceless irrep construction (_invariant_basis) matches the closed-form dimension across N in {3..6}, p in {2..4} with an SVD rank gap of ~1e0 vs the 1e-10 cut. so_n tower generators are exactly skew (max|G+G^T|=2e-16) and the skew transport fast path is bit-exact (exp_neg=exp_pos^T to 0.0, det=1 to 4e-7); sp_n is correctly non-skew. Head-mixer diagonal closed form equals the full-sandwich diagonal to 1e-7 and is SPD-preserving; the ParameterList isotypic refactor and its checkpoint key remap are sound and captured by the optimizer's exact-coverage guard along with cg_coupling.path_weights. CGCoupling means-only coupling is exactly equivariant at fp32 (residual 2.4e-7) with a correct gradient (FD rel err 1.3e-4) even with multiple paths writing the same target slot. The SPD retraction is PD-preserving and finite-gradient at cond(Sigma)=5e6 and at the degenerate Sigma=I init (the _eigh_damped Lorentzian backward works), with correct clamp dead-zoning. The morning-audit pair-term saturation gap (P7/N1) is fixed (pair_mask present in belief_gradients), and safe_cholesky/safe_kl_clamp NaN->kl_max masking in families/gaussian.py is consistent across the alpha=1 and alpha>1 branches. The hooks failing throughout this session are an unrelated claude-mem plugin worker, not a code defect; all tool outputs came through intact.

#### audit-transformer-ml — 2 finding(s)

| # | Sev (verified) | Verdict | Reach | Finding | Location |
|---|---|---|---|---|---|
| F29 | medium | CONFIRMED | diag | Diagnostic belief replay omits head_mixer and cg_coupling, diverging from forward under those toggles | `vfe3/viz/extract.py:269-273, 355-360` |
| F30 | low | CONFIRMED | toggle | CGCoupling.forward runs the full bilinear path loop even when path weights are all zero (is_identity) | `vfe3/model/cg_coupling.py:109-116` |

Clean areas: I focused on the code that landed after the morning audit (cg.py, irreps.py, groups.py so_n/sp_n, head_mixer isotypic refactor, cg_coupling) plus the priority attention/decode surface, and ran executable probes throughout. The numerical Clebsch-Gordan solver in geometry/cg.py is correct: l1xl1->l0 gives one intertwiner (the dot product), l1xl1->l1 gives one with d_c=3 (the cross product), trivial l0xl0->l0 gives the scalar, and the sp tower selection enumerates the right triples; the build-time equivariance assert is real (raise, not warn). CGCoupling is exactly equivariant under an SO(3) tower gauge (residual 4.8e-7), its forward autograd matches a non-in-place reference to maxdiff 0.0 (the in-place delta slice assignment is autograd-safe and target slices do not alias source reads), and sigma genuinely passes through untouched (means-only as documented). The isotypic HeadMixer is correct on mults-one towers (per-block 1x1 scalar gains) and mult-3 towers (one 3x3 commutant), identity at init. Per-head tau on UNEQUAL tower dims [3,5] returns kappa*sqrt(d_h)=[1.73,2.24] and satisfies the per-head envelope identity sum beta E + tau sum beta log(beta/pi) = -tau log Z to 1.5e-7, with softmax rows summing to 1 and zero future leakage under the causal prior. The attention_prior registry masks with -inf BEFORE softmax on every causal variant (causal, causal_alibi, causal_windowed, t5 bidirectional=False), softmax is over the key axis (dim=-1), and the construction-time guard at model.py:99 rejects the alibi (H,N,N) head-axis vs single-block energy mismatch (morning P1, fixed). The diagonal decode is the exact -KL(q||pi_v)/tau closed form with the catastrophic-cancellation offset removed. The optimizer exact-coverage guard in train.py groups both head_mixer.parameters() and cg_coupling.path_weights and asserts no trainable parameter is left ungrouped. RoPE-on-tower does break gauge equivariance (measured commutator 0.81 against the so_n transport) but this is the already-documented morning G1/G2 finding, now warned explicitly at config.py:766 with the correct "R(theta) is generally not in the irrep image" reasoning, so it is handled rather than a new defect. The remaining gaps I found are diagnostic-fidelity (extract.py) and a minor zero-init perf waste, not pure-path or correctness defects on the training path.

#### audit-implementation-engineer — 4 finding(s)

| # | Sev (verified) | Verdict | Reach | Finding | Location |
|---|---|---|---|---|---|
| F31 | high→medium | CONFIRMED | toggle | HeadMixer and CGCoupling learnables silently freeze under detach_e_step=True (no warning) | `vfe3/model/model.py:465,474 (consuming vfe3/model/block.py:73-78)` |
| F32 | medium | CONFIRMED | diag | attention_maps() omits head_mixer, replaying CG on but mixer off | `vfe3/model/model.py:935-944` |
| F33 | medium | CONFIRMED | diag | viz/extract.py replay sites omit head_mixer and cg_coupling (publication figures replay both toggles OFF) | `vfe3/viz/extract.py:184,270,355` |
| F34 | low | CONFIRMED | diag | No test pins the detach_e_step mixer/CG gradient flow, and none pins the viz/attention-map replay parity | `tests/test_cg.py:155, tests/test_head_mixer_isotypic.py:83` |

Clean areas: I config-traced every toggle that landed after the morning audit from the entry-point dicts (train_vfe3.py/ablation.py) through config.py, model.py, stack.py, block.py to consumption. The pure path is intact: with use_head_mixer=False and use_cg_coupling=False the model sets self.head_mixer=None and self.cg_coupling=None, so no learned compute touches the forward (block.py guards both with `if ... is not None`). CGCoupling is mathematically correct and exactly equivariant: probed f(g.mu)=g.f(mu) residual 5.7e-9 for so(3) [l0,l1,l2] and 7.8e-9 for sp(4) [sym1,sym2] at random nonzero path_weights; the cg.py null-space intertwiner solver reproduces the exact so(3) CG series (l1xl1=l0+l1+l2, l2xl2=l0..l4), the build-time equivariance assert is real (1e-7), and overlapping target-block writes accumulate correctly with autograd to path_weights intact. float32 is preserved end-to-end (forward logits float32, no float64 param creep; CG intertwiner buffers are float64 storage re-cast to mu.dtype per the documented design). The optimizer (train.py:90-93) covers head_mixer.parameters() and cg_coupling.path_weights exactly at m_mu_lr and the exact-coverage guard passes with both on (probe: optimizer covers all params exactly True). Checkpoint round-trip is clean: state_dict carries head_mixer.mixer_deltas.{0,1,2}, cg_coupling.path_weights, and the cg_* buffers; save/load with weights_only=True restores all bit-exactly, and the _load_from_state_dict shim remaps the legacy mixer_delta key. The so_n/sp_n group builders populate irrep_labels correctly and HeadMixer/CGCoupling consume them through the registry, and config validation guards use_cg_coupling to so_n/sp_n with CGCoupling raising at construction on no admissible paths. The forward decode reads the post-mixer/post-CG belief, so under the default unroll path the mixer/CG do train (probed nonzero grads end-to-end). The HeadMixer diagonal-sigma closed form (A*A einsum) matches diag(M Sigma M^T), and diagnostics correctly applies both modules.


### Replacement base wave (standalone agents, separately verified)

#### code-reviewer — 5 findings

| # | Sev (verified) | Verdict | Reach | Finding | Location |
|---|---|---|---|---|---|
| CR1 | low | CONFIRMED | latent | CG intertwiner cache key omits atol; a later call with a different tolerance returns the first call's null-space result | `vfe3/geometry/cg.py:59` |
| CR2 | low | CONFIRMED | toggle | CGCoupling.forward never short-circuits at zero init; is_identity() has no production caller | `vfe3/model/cg_coupling.py:102` |
| CR3 | low | CONFIRMED | toggle | HeadMixer.is_identity() never called in production; full einsum cost paid at exact-zero deltas | `vfe3/model/head_mixer.py:118` |
| CR4 | low | CONFIRMED | toggle | HeadMixer full-cov path rebuilds dense (K,K) commutant M every forward | `vfe3/model/head_mixer.py:158` |
| CR5 | low | CONFIRMED | latent | Process-global _CG_CACHE has no eviction across model rebuilds | `vfe3/geometry/cg.py:37` |

Clean areas: the six post-audit modules and their full config/model/stack/optimizer wiring;
optimizer exact-coverage guard groups `path_weights`; CGCoupling autograd through in-place delta
accumulation probe-verified; build-time raise-not-warn guards (equivariance, bracket
homomorphism, dimension cross-check); float64 construction with explicit casts; eigh-basis
intertwiners persisted as buffers so resume is basis-consistent; `torch.load(weights_only=True)`
everywhere except the justified self-written bundle; no registry-seam or signature-convention
violations in the new modules; the uncommitted `ablation.py`/`train_vfe3.py` diffs are
config-dict tweaks only.

#### debugger — 6 findings

| # | Sev (verified) | Verdict | Reach | Finding | Location |
|---|---|---|---|---|---|
| DB1 | medium→low | CONFIRMED | latent | HeadMixer._A() lacks a dtype cast: loud crash on module/input dtype mismatch (both diagonal and full-cov arms; CGCoupling casts correctly) | `vfe3/model/head_mixer.py:114` |
| DB2 | medium | CONFIRMED | toggle | CGCoupling forward: O(n_paths) Python loop of per-path einsums (probe: 21 paths ≈ 4 ms/call CPU; ≥quadratic copy scaling) | `vfe3/model/cg_coupling.py:110` |
| DB3 | low | CONFIRMED | toggle | Non-contiguous same-label blocks yield per-copy 1×1 mixers — a proper subspace of the true commutant, silent (test-sanctioned as intended) | `vfe3/model/head_mixer.py:85` |
| DB4 | low | CONFIRMED | toggle | cg_selection warms the cache, then CGCoupling.__init__ re-calls cg_intertwiners per triple (cache-hit clones, construction-time only) | `vfe3/model/cg_coupling.py:47` |
| DB5 | low | CONFIRMED | diag | No test pins the mixer's dtype contract (mixed module/input dtypes crash, untested) | `tests/test_head_mixer_isotypic.py:1` |
| — | — | — | — | Positive cross-check: attention_maps (model.py:806) and diagnostics (model.py:941) thread cg_coupling identically to training; the legacy state_dict shim works; _dense_m isotypic block layout verified correct | — |

#### python-pro — 9 findings

| # | Sev (verified) | Verdict | Reach | Finding | Location |
|---|---|---|---|---|---|
| PP1 | high→low | CONFIRMED | toggle | No config-level guard for use_head_mixer on single-block groups — but probe shows a fast, clearly-worded ValueError at model construction (stage-consistency nit, not silent) | `vfe3/config.py:503` |
| PP2 | medium→low | CONFIRMED | latent | cg_intertwiners equivariance-verify gate hardcoded 1e-7, independent of atol | `vfe3/geometry/cg.py:93` |
| PP3 | medium→low | CONFIRMED | toggle | _dense_m rebuild is O(K²) against the O(K³·batch) sandwich it feeds — relative cost minor | `vfe3/model/head_mixer.py:152` |
| PP4 | low | CONFIRMED | toggle | Duplicate of DB4 (one clone per triple at construction, negligible) | `vfe3/model/cg_coupling.py:47` |
| PP5 | low | CONFIRMED | diag | _CG_CACHE never reset between tests — warm/cold path order dependence within one pytest process | `vfe3/geometry/cg.py:37` |
| PP6 | low | CONFIRMED | toggle | All-distinct-label towers silently degenerate the mixer to per-block scalar gains (Schur-forced; the gap is the missing warning — probe: zero warnings) | `vfe3/model/head_mixer.py:85` |
| PP7 | low | CONFIRMED | toggle | GaugeNaturalGradAdamW.__init__ places Tensor args after params — forced by the torch Optimizer contract; document the convention exception | `vfe3/gauge_optim.py:58` |
| PP8 | low | CONFIRMED | toggle | irrep_spec annotation omits the accepted pre-coercion list-of-lists form | `vfe3/config.py:111,459` |
| PP9 | low | CONFIRMED | toggle | No test exercises the sp algebra through cg_intertwiners/CGCoupling (all CG tests use so) | `tests/test_cg.py:1` |

#### refactoring-specialist — 6 findings (second attempt; first attempt truncated, see Adversarial Challenge)

| # | Sev (verified) | Verdict | Reach | Finding | Location |
|---|---|---|---|---|---|
| RF1 | medium→low | CONFIRMED | toggle | CGCoupling algebra string derived from cfg.gauge_group ternary, not GaugeGroup — mis-dispatch needs a second edit to the validator, so maintainability only | `vfe3/model/model.py:138` |
| RF2 | low | CONFIRMED | toggle | The same algebra ternary duplicated in config.__post_init__ (live at :470) | `vfe3/config.py:438` |
| RF3 | low | CONFIRMED | toggle | Duplicate of DB2 (per-path loop; accumulation is non-inplace assignment, not +=) | `vfe3/model/cg_coupling.py:110` |
| RF4 | low | CONFIRMED | diag | path_types built at construction; only consumer is one test | `vfe3/model/cg_coupling.py:61` |
| RF5 | low | CONFIRMED | latent | omega_direct still in _VALID_GAUGE_PARAM then unconditionally rejected (morning-audit R8, unchanged; now framed as a deliberate reservation stub) | `vfe3/config.py:522` |
| RF6 | low | CONFIRMED | latent | Duplicate of CR5 (entries capped ~40 KB by the 5000 cost guard) | `vfe3/geometry/cg.py:37` |

Clean areas (replacement wave, condensed): use_cg_coupling wiring traced end-to-end
(model.py:135-141 → stack.py → block.py:76-78) with correct BeliefState reassignment and
optimizer registration; path enumeration consistent with cg_selection's canonical ordering;
GaugeGroup.irrep_labels populated exactly by the tower builders with clear None-handling
elsewhere; CG solver internals (gram null-space via eigh, build-time equivariance assert,
clone-on-return) sound; SO(3) selection rules and the a≤b canonicalization verified
mathematically; path_weights gradients all finite and nonzero (18/18); _cast_cache correctly
keyed and invalidated; no mutable default arguments and no new orphaned imports in the new
modules.

## Verifier Verdicts

All 60 findings received source-cited verdicts; none rested on comments, and multiple verifiers
ran their own reproduction probes. The corrections that materially changed findings:

- **PP1 high→low.** The missing config guard for `use_head_mixer` on single-block groups fails
  fast at `VFEModel` construction with an actionable `HeadMixer` ValueError naming the fix
  (probe across all seven gauge groups); a stage-consistency nit rather than a silent failure.
- **F1 high→low / F2 high→medium.** A zero-init short-circuit in `CGCoupling.forward` would cut
  `path_weights` out of the autograd graph, so the zero-init weights could never train — the
  short-circuit is only sound under no-grad/eval (the same caveat applies to CR2/CR3 on the
  mixer). The per-path loop cost itself stands (probe: ≈4 ms/call CPU at 21 paths).
- **CR4/PP3.** Caching `_dense_m` keyed on (device, dtype) would serve stale gradients: M is
  differentiable through the trainable deltas, so the per-forward rebuild is required during
  training; only an eval-mode cache is sound, and the rebuild is O(K²) against the O(K³·batch)
  sandwich it feeds.
- **DB1 medium→low.** The mixer's missing dtype cast crashes loudly on both covariance arms (the
  mean einsum fires first), but only under a module/input dtype mixture no default flow
  produces.
- **F22/F27 REFUTED.** The claimed eigh-basis/path_weights persistence hazard is inverted by the
  code: the intertwiners are registered buffers, persisted in the state_dict, so resume restores
  the exact basis the weights were trained against.
- **Contradiction resolved (F19).** The gauge expert's clean-area summary asserted the M-step
  self-coupling reads the pre-mix belief; the variational expert's F19 claimed post-mix. The
  challenge tier (below) settled it for post-mix from executable source — the in-repo comment at
  `model.py:548` asserting "BEFORE head_mixer/norm" is false and is exactly what a reviewer
  trusting comments would repeat.

Per-finding verifier notes for the workflow wave are archived below in the appendix.

## Adversarial Challenge

No finding survived verification at critical or high, so the mandatory duel set was empty. Two
findings were escalated at orchestrator discretion, and one in-flight investigator claim was
refuted directly:

| # | Finding | Skeptic | Defender | Verdict | Reason |
|---|---|---|---|---|---|
| F19 | M-step self-coupling reads the post-mixer/post-norm belief, not the converged variational belief | Downgrade to low: pure path intact (all overwriting branches skip at defaults, probe maxdiff 0.0); post-mixer read is consistent with what the handoff (stack.py:60) and decode (model.py:488) consume; the term is dead in the active run (mstep_self_coupling_weight=0.0) | Upheld at medium: the manuscript pins the self-term to the variational belief q* (Participatory_it_from_bit.tex:1251, 1562); mixer/norm are post-inference transforms absent from F; probe quantifies the divergence (0.0 toggles-off; 1.6e-2 under mahalanobis norm; 5.4e-3 under mixer); comment at model.py:548 is false | **UPHELD (medium)** | block.py:73-80 overwrites `out` before return; model.py:574 reads it; the manuscript citation decides the theory question, the pure-path probe caps severity. The belief-source semantics under toggles is flagged as a user decision; the false comment is fixed tonight. |
| F31 | HeadMixer/CGCoupling learnables silently freeze under detach_e_step=True | Cannot refute: probe shows mixer grads None under detach while 3/6 params train; the five sibling freeze-warnings (model.py:157-247) establish the project's own standard, and mixer/CG are omitted from every one | Upheld at medium: full reachability trace config.py:1058 → model.py:465,474 → block.py:73-78 (sole application site, inside no_grad); optimizer still owns the frozen params; bonus defect — the config-level warning predicate (config.py:927) keys on the raw e_step_gradient literal, which __post_init__ forces to "unroll" under detach_e_step, so it can never fire on the detach path | **UPHELD (medium)** | Both duelists agree with concurring probes; fix = add the missing freeze-warnings and repair the config predicate. |
| — | (refactoring agent, in-flight, unfiled) sp(2m) Sym^p towers are reducible; "Casimir non-scalar on sp(4) sym2" | — | — | **REFUTED (orchestrator probe)** | Schur-commutant test with the repo's own builders: commutant dimension exactly 1 (absolutely irreducible) for sp(4) sym1/sym2/sym3 and so(3) l1/l2. The agent's naive Casimir Σ G_aG_a over a non-Killing-orthonormal basis is not basis-invariant; classical theory (Sym^p of the defining rep of sp(2m) is irreducible) and the code agree. |

## Consolidated Punch List (post-verification, post-challenge, deduped, ranked)

No critical or high item survived. Mediums first, then grouped lows.

1. **[medium / theory, toggle] F19 (+F23)** — M-step self-coupling reads the post-mixer/
   post-norm belief; the comment at `vfe3/model/model.py:548` claims the opposite; the
   mixer/CG placement in the inter-block handoff is undocumented. Tonight: fix the false
   comment, document the actual semantics. The belief-source choice is a **user decision**
   (see User Decisions).
2. **[medium / toggle] F31** — silent freeze of mixer/CG learnables under `detach_e_step=True`;
   add model-level freeze-warnings mirroring the five siblings, and fix the dead config-level
   predicate (`config.py:927` keys on the raw literal).
3. **[medium / performance, default] F3** — `_blockwise_matrix_exp` H-loop slice reads/writes
   around one batched exp — `vfe3/geometry/transport.py:286`.
4. **[medium / performance, default] F4** — `_factored_diagonal_covariance` materializes a
   rank-5 (…,N,d,d,d) intermediate per head block — `vfe3/geometry/transport.py:488`.
5. **[medium / diagnostics] F29+F32+F33** — viz replay sites (`viz/extract.py:184,270,355`) omit
   both mixer and CG; `attention_maps` (model.py:935-944) passes CG but omits the mixer —
   figures and replays describe a different model than the one training under those toggles.
6. **[medium / performance, toggle] F2+DB2+RF3+F21** — CGCoupling per-path Python loop; batch
   paths sharing an intertwiner. Companion correctness-of-parameterization item: **F10+F13+F20**
   dead zero-gradient antisymmetric self-pair paths should be pruned at enumeration (with a
   liveness test, F15).
7. **[low, config/validation]** PP1 config-level use_head_mixer guard; PP6+DB3 degenerate-tower
   construction warnings; PP8 irrep_spec annotation.
8. **[low, CG numerics hygiene]** CR1+PP2+F12 atol coherence (cache key + verify gate); F25
   cost-guard raise mid-enumeration in cg_selection; DB4+PP4+F6+F9 double-build/clone churn at
   construction; CR5+RF6+PP5 cache clear API.
9. **[low, eval-path speed]** F1+F14+F16+F30+CR2+CR3+F17 — grad-safe identity short-circuit
   (no-grad contexts only, per the verifier's autograd caveat), giving `is_identity()` its
   production caller.
10. **[low, perf trivia]** F7 attention_tau per-call CPU tensor; F8 uniform-pi allocation; F5
    `_dense_m` (document the training-rebuild requirement; optional eval cache).
11. **[low, maintainability]** RF1+RF2 algebra string centralised on GaugeGroup; RF4 path_types
    documented as introspection-only.
12. **[low, numerics docs]** F24 fp32 policy for mixer/CG under autocast (document or island);
    F26 transport conditioning at max_norm; F28 fp64-vs-fp32 equivariance gate note; F11
    means-only coupling is not a congruence image (document); PP7 optimizer signature exception
    note; DB1 mixer dtype cast.
13. **[low, tests]** PP9 sp-algebra CG test; DB5 mixer dtype-contract test; F18 CGCoupling
    overlapping-slice gradcheck; F34 detach+mixer/CG warning pin; F15 path liveness.
14. **[low, pre-existing]** RF5 omega_direct reservation stub (morning R8) — left as the
    documented deliberate rejection.

## User Decisions (flagged, not fixed autonomously)

1. **F19 — which belief should the M-step self-coupling anchor the priors to when post-E-step
   transforms are enabled?** The manuscript's F pins the self-term to the converged variational
   belief q*; the code anchors to the post-mixer/post-norm handoff belief (what the next layer
   and decode consume). Both semantics are defensible for the documented NN-exception
   extensions; they differ measurably (probe: 1.6e-2 under mahalanobis norm). Tonight's action
   was limited to correcting the false comment and documenting the current behavior. If you want
   the manuscript semantics, vfe_block would need to return (or stash) the pre-transform
   converged belief for the M-step term.
2. **CGCoupling dead-path pruning and checkpoint shape.** Pruning the structurally dead
   antisymmetric self-pair paths (punch 6) changes the length of `path_weights`; any existing
   checkpoint trained with `use_cg_coupling=True` under the old enumeration would fail a strict
   load. The feature is days old and default-off, so this was fixed tonight; flagged here in
   case such a checkpoint exists.

## Test Suite

- Baseline before any fix (read from JUnit XML): **808 tests, 0 failures, 0 errors** (~120 s).
- After the fix pass (read from JUnit XML): **825 tests, 0 failures, 0 errors** (~83 s) — all
  808 baseline tests still pass alongside the 17 new regression pins in
  `tests/test_audit_fixes_2026_06_10.py`. The fixes are logged in
  `docs/edits/2026-06-10-edits.md`.
- The active `train_vfe3.py` config pairs `use_head_mixer=True` with the default 'unroll'
  E-step estimator, so the F31 freeze is NOT live in the current click-to-run setup; the new
  warning protects future toggle combinations.

## Appendix — per-finding verifier notes (workflow wave)

- **F1** (CONFIRMED, src `vfe3/model/cg_coupling.py:102-116; vfe3/model/block.py:76-77; vfe3/train.py:92-93`): forward() iterates all paths with no zero-weight short-circuit; is_identity() (cg_coupling.py:99-100) exists but has no production caller, and block.py:76-77 invokes the coupling unconditionally whenever cfg.use_cg_coupling built it. Correction: path_weights are optimized (train.py:92-93), so the all-zero waste is transient (first optimizer step / untrained inference only) — the standing cost is F2's loop, not the missing guard. The 12 ms figure is investigator runtime data I could not verify.
- **F2** (CONFIRMED, src `vfe3/model/cg_coupling.py:110-115; vfe3/model/block.py:76-78`): Confirmed: P sequential per-path slice/outer-product/einsum operations in a Python loop, each launching separate kernels. Correction: 'all paths share one intertwiner' holds only for single-triple towers (e.g. 4-copy l1: 10 pairs x 4 targets x 1 mult = 40 paths on cg[0][0]); generally grouping is per (t,m). The coupling runs once per block after the E-step (block.py:76-78), not per inner iteration, so 'high' overstates the cost; the 2.2x probe number is unverified runtime data.
- **F3** (CONFIRMED, src `vfe3/geometry/transport.py:283-300, 246-252, 374-376; vfe3/inference/e_step.py:113-114, 285, 359; vfe3/config.py:66; vfe3/geometry/groups.py:128`): Code structure is exactly as claimed: H-iteration torch.stack read + .contiguous(), one batched matrix_exp, H-loop slice write-back. Default-reachable: block_glk default with flat transport takes build_factored_transport -> stable_matrix_exp_pair with block_dims, skew_symmetric=False so both exp(phi) and exp(-phi) run the blockwise path, rebuilt every E-step iteration (e_step.py:359) and again in the phi step (e_step.py:285). Correction on severity: the dominant op (matrix_exp) is already a single batched call; the remaining waste is ~2H small slice kernels plus one extra full-tensor copy, so medium rather than high.
- **F4** (CONFIRMED, src `vfe3/geometry/transport.py:480-491, 446-453; vfe3/config.py:66,146`): Confirmed: ep2 = ep.unsqueeze(-1) * ep.unsqueeze(-2) materializes the (..., N, d, d, d) rank-5 intermediate per head block (transport.py:488), reached on the default flat + block_glk + diagonal-covariance path via transport_covariance -> _factored_diagonal_covariance. The finding's own fix concedes rank-5 beats the rank-4 alternative when d < N (the typical regime), so the substance is a VRAM caveat at large d, not a wrong algorithm; GB figures are unverified estimates.
- **F5** (CONFIRMED, src `vfe3/model/head_mixer.py:146-159; vfe3/config.py:119,146`): Confirmed: the full-covariance branch of forward (head_mixer.py:148-150) calls _dense_m on every invocation, and _dense_m (152-159) rebuilds the (K,K) zeros + per-component kron each call with no cache. Requires use_head_mixer=True (default False) AND a full-covariance family (the default diagonal path at :141-147 never touches _dense_m), so doubly toggled. The 36 ms measurement is investigator runtime data I could not verify.
- **F6** (CONFIRMED, src `vfe3/geometry/cg.py:37, 111-119, 48-99; vfe3/config.py:503`): Confirmed: cg_selection's triple loop issues n_unique*(n_unique+1)/2 * n_unique cg_intertwiners calls, fully solving (Gram + eigh) zero-multiplicity triples before discarding them at :118, and _CG_CACHE (:37) is a process-local dict with no cross-process persistence. Correction: the finding itself concedes this is an acceptable one-time init cost, and the DDP/DataLoader-worker advice is speculative — the repo's entry points are single-process click-to-run — so low rather than medium; the 253 ms figure is unverified.
- **F7** (CONFIRMED, src `vfe3/free_energy.py:58-63, 30-32; vfe3/model/block.py:56`): Confirmed: on the unequal-irrep-dims branch attention_tau allocates a fresh CPU tensor (no device arg, free_energy.py:60) on every call, and block.py:56 calls it inside every vfe_block invocation; _broadcast_tau (free_energy.py:30-32) then re-sends the (H,) tau to the energy's device per attention computation. Equal-dims (default block_glk) returns a scalar with no allocation, so the issue is confined to so_n/sp_n unequal towers as claimed.
- **F8** (CONFIRMED, src `vfe3/free_energy.py:349-353; vfe3/gradients/oracle.py:110; vfe3/inference/e_step.py:235,499; vfe3/viz/extract.py:238,244; vfe3/attention_prior.py:60,286; docs/audits/audit-2026-06-09.md:132`): Confirmed: free_energy() allocates torch.full_like(beta, 1/N) on the log_prior=None branch (free_energy.py:350-351), and the algebraic simplification is valid. Reach 'diag' verified: every registered attention prior returns a tensor (even 'uniform' returns zeros, attention_prior.py:60), so model paths pass non-None log_prior; free_energy() itself is reached only via the autograd oracle and the diagnostic free_energy_value (trajectory logging / viz). This duplicates the still-OPEN audit item PE7 (audit-2026-06-09.md:132) — not fixed in current source, so the already-fixed refutation rule does not apply.
- **F9** (CONFIRMED, src `vfe3/geometry/cg.py:60-61,98-99; vfe3/model/cg_coupling.py:47-56`): cg_intertwiners returns _CG_CACHE[key].clone() on every cache hit (cg.py:60-61) and C.clone() on first build (cg.py:99); CGCoupling.__init__ is the only production caller and registers the result as a buffer (cg_coupling.py:50-56). Correction: register_buffer stores a reference, not a further copy — the two live copies per triple are the cache entry plus the clone held as the buffer; cg_selection (cg.py:116-117) additionally allocates-and-discards clones while probing multiplicities. One-time init cost, severity low is right.
- **F10** (CONFIRMED, src `vfe3/model/cg_coupling.py:67-73 (loop skips only jb<ia, keeping ia==jb); probe on [l0,l1,l2] tower`): Verified by an executed probe: the l1xl1->l1 intertwiner is antisymmetric under copy swap (residual 7.4e-16, |C(x,x)| ~ 1.7e-15), and with all path_weights=1.0 the diagonal self-pair paths ('l1','l1','l1') and ('l2','l2','l1') receive grads 3.2e-15/4.9e-15 while every live path gets O(1)-O(100) — structurally frozen learnables. I confirmed 2 dead paths in the standard mult-1 tower; the investigator's 4-path count for a [('l1',2),('l0',1)] tower is consistent (2 diagonal copy pairs x 2 targets) but not re-run.
- **F11** (CONFIRMED, src `vfe3/model/cg_coupling.py:109-116 (return mu + delta, sigma); tests/test_cg.py:124-134 (S2 is S)`): CGCoupling.forward updates mu and returns sigma untouched (same object, pinned by test_cg_coupling_full_cov_sigma_passes_through), so post-coupling (mu',Sigma) is indeed not a congruence image — but this is the CLAUDE.md-sanctioned 'means-only sigma' design for CG coupling. Correction: the head_mixer half of the claim is false — head_mixer.py:141-150 does update sigma (diagonal A*A closed form and full-cov M @ sigma @ M^T sandwich). Informational only; no code defect.
- **F12** (CONFIRMED, src `vfe3/geometry/cg.py:56 (atol=1e-8 default), cg.py:87-88 (evals < atol cut), cg.py:90-97 (1e-7 equivariance assert)`): Code-factual claim confirmed: multiplicity counting is an absolute eigenvalue cut at a fixed atol with no runtime emission of the spectral gap; the build-time assert (cg.py:90-97) raises on spurious slots (false positives) but nothing in executable code detects a dropped slot (false negative). Reach corrected from diag to toggle: cg.py:88 determines the live model's path structure under use_cg_coupling, it is not diagnostics-only code — the proposed FIX is a diagnostic, the affected path is not.
- **F13** (CONFIRMED, src `vfe3/model/cg_coupling.py:67-73; probe grads ('l1','l1','l1')=3.2e-15, ('l2','l2','l1')=4.9e-15`): Duplicate of F10, same code path: the enumeration keeps ia==jb diagonal copy pairs for a==b, and for antisymmetric intertwiners (l1xl1->l1, l2xl2->l1 in so(3)) C(x,x)=0 identically, leaving those path_weights with structurally zero gradient — verified by an executed probe on the standard [l0,l1,l2] tower (2 dead of 11 paths).
- **F14** (CONFIRMED, src `vfe3/model/cg_coupling.py:99-116; vfe3/model/block.py:76-78; repo-wide grep for is_identity()`): forward (cg_coupling.py:102-116) runs the full per-path Python einsum loop unconditionally; is_identity (99-100) has no caller anywhere in vfe3/ or tests/ (the grep hits are head_mixer tests and unrelated test names). It executes once per block per forward via block.py:76-78. Caveat on the proposed fix: a short-circuit must be gated to inference (eval/no_grad) as the finding says, because the zero-weight forward is what produces the nonzero dF/dw needed to train path_weights off zero init.
- **F15** (CONFIRMED, src `tests/test_cg.py:109-121 (single hand-picked path), tests/test_cg.py:155-172 (aggregate grad.abs().sum() > 0 only)`): No test asserts per-path liveness: test_cg_coupling_self_product_reaches_other_types activates only the ('l1','l1','l2') path, and test_cg_model_step0_byte_identical_and_trains checks only the summed gradient magnitude, which the dead antisymmetric self-pairs (F10/F13, probe-verified) cannot fail. Minor correction: the file extends to line 172, not 155.
- **F16** (CONFIRMED, src `vfe3/model/cg_coupling.py:99-116; vfe3/model/block.py:76-78`): Same substance as F14, confirmed: no is_identity early-out in forward and the predicate has zero callers. Correction to the title: the loop does NOT run every E-step — block.py:76-78 invokes cg_coupling once per block forward, after the e_step call that internally runs cfg.n_e_steps iterations; cost is O(n_paths) einsums per block per forward, not per E-step iteration.
- **F17** (CONFIRMED, src `vfe3/model/cg_coupling.py:99; vfe3/model/head_mixer.py:118; tests/test_head_mixer.py:27; tests/test_head_mixer_isotypic.py:13`): CGCoupling.is_identity has zero callers anywhere in the repo (grep over all *.py: only the def lines). Correction: HeadMixer.is_identity is NOT fully dead — it is asserted by two tests (test_head_mixer.py:27, test_head_mixer_isotypic.py:13), though it has no production caller in vfe3/; also the head_mixer def is at line 118, not 117.
- **F18** (CONFIRMED, src `tests/test_cg.py:155-172; tests/test_retraction.py:309 (only gradcheck in suite); vfe3/model/cg_coupling.py:110-115`): Quoted asserts at test_cg.py:169-172 verified verbatim — only isfinite and nonzero-grad checks. test_cg.py is the only test file referencing CGCoupling and contains no gradcheck or finite-difference comparison; the sole torch.autograd.gradcheck in the suite is test_retraction.py:309. The overlapping-slice accumulation (cg_coupling.py:114, delta[..., sc:sc+dc] = delta[...] + ...) is indeed unpinned by any numeric gradient test.
- **F19** (CONFIRMED, src `vfe3/model/block.py:73-80; vfe3/model/model.py:481, 548, 569-574; vfe3/config.py:175`): vfe_block overwrites out with head_mixer (block.py:73-75), cg_coupling (76-78), and block_norm mu (80) before returning; model.py:573-574 computes self_div from fam(out.mu, out.sigma) on that post-mixer/post-norm belief while the in-code comment at model.py:548 claims it is the converged belief BEFORE head_mixer/norm. Two corrections: block_norm replaces only mu, so out.sigma is post-mixer/cg but unaffected by the norm; and the prior-side rebuild (model.py:569-571) folding out.mu is consistent with the live stack handoff (stack.py:57), so the deviation from the manuscript term is on the q*-side only. Needs mstep_self_coupling_weight>0 (default 0.0) plus a mixer/cg/norm toggle to manifest.
- **F20** (CONFIRMED, src `vfe3/model/cg_coupling.py:62-74 (path construction includes sa==sb self-copy with ia<=jb); probe on (l0,l1,l2) SO(3) tower`): Read-only probe with random path_weights and float64 inputs: paths ('l1','l1','l1') and ('l2','l2','l1') (both sa==sb single-copy) have path_weights grads 2.1e-14 and 2.2e-15 versus ~11-47 for the other 9 paths — the antisymmetric intertwiner contracted with the symmetric diagonal mu(x)mu vanishes identically, so those nn.Parameter coordinates never train and never affect output, exactly as claimed.
- **F21** (CONFIRMED, src `vfe3/model/cg_coupling.py:110-115; vfe3/model/block.py:76-78; vfe3/model/stack.py:51-56`): Forward is a Python for-loop over self.paths with one einsum plus a slice read-modify-write per path; probe confirms 18 paths for a ['l1','l1','l2'] SO(3) tower (the e2e test tower l0,l1,l2 has 11), and the module runs once per block inside vfe_stack's n_layers loop, so kernel launches scale as O(n_paths * n_layers) per forward. Performance observation, not a correctness bug.
- **F22** (REFUTED, src `vfe3/model/cg_coupling.py:56, 89-97; vfe3/geometry/cg.py:87-88; vfe3/run_artifacts.py:217, 279; vfe3/viz/report.py:113; state_dict probe`): The facts are individually true (buffers persisted — probe shows cg_0..cg_10 in state_dict; eigh null-space basis at cg.py:87-88; no load-time re-verification), but the claimed hazard is inverted by those same facts: every checkpoint load in the repo is a strict load_state_dict (run_artifacts.py:217/279, report.py:113), which restores the cg_* buffers TOGETHER with path_weights, and forward consumes the restored buffers via _cast_buffers/getattr (cg_coupling.py:94). A torch/eigh change in a new build therefore cannot silently apply trained multiplicity weights to a rotated basis — the freshly built (possibly rotated) buffers are overwritten by the checkpoint's. Persistence is precisely the mechanism that keeps weights and basis consistent; the finding's risk scenario does not exist in current code (it would only arise if the buffers were registered persistent=False).
- **F23** (CONFIRMED, src `vfe3/model/block.py:73-78; vfe3/model/stack.py:51-57; Manuscripts-Theory/GL(K)_attention.tex:2364`): Code-factual content verified: head_mixer then cg_coupling are applied inside every vfe_block, and stack.py:57 folds the mixed belief.mu into the next block's prior (mu_p = (1-rho)mu_p + rho*belief.mu), so under n_layers>1 the maps iterate into every layer's prior handoff. GL(K)_attention.tex:2364 does place the mixer 'between the belief update and the readout' in the W_O slot. Correction: the same manuscript passage already concedes 'genuine cross-head capacity appears only under a prior-bank decode ..., a final normalization, or depth greater than one', so it is aware depth>1 exceeds the W_O-absorption argument — what is unstated is specifically the per-block prior-handoff recursion. Doc-alignment issue; note the file lives at Manuscripts-Theory/, not Attention/.
- **F24** (CONFIRMED, src `vfe3/model/cg_coupling.py:107-116; vfe3/model/head_mixer.py:130-159; vfe3/model/model.py:474-487; vfe3/geometry/retraction.py:104,141,241,313,345; vfe3/geometry/transport.py:242`): Verified: no autocast appears anywhere in cg_coupling.py or head_mixer.py (repo grep), while transport.py and retraction.py wrap their math in torch.amp.autocast(enabled=False) islands; vfe_stack — and hence the mixer/CG forwards (block.py:73-78) — executes inside the amp context (model.py:474 'with run, amp:'), and cg_coupling casts weights/buffers to mu.dtype (lines 107-108), so under amp_dtype='bf16' these einsums run in bf16. Severity correction: equivariance is algebraic and holds for any weights at the working precision — bf16 coarsens rounding (vs fp32's own rounding) rather than introducing training-drift-style breakage, and the einsums are not cancellation-prone like the matrix_exp/eigh the existing islands protect; doubly opt-in (amp_dtype AND use_head_mixer/use_cg_coupling non-default). Downgrade to low.
- **F25** (CONFIRMED, src `vfe3/geometry/cg.py:66-71,113-119; vfe3/model/cg_coupling.py:47`): cg_selection's triple loop (cg.py:113-117) calls cg_intertwiners with no exception handling, and the dc*da*db>5000 guard (cg.py:66-71) raises mid-enumeration; probe reproduced the exact ValueError for sp N=6 ['sym1','sym2','sym4'] (15876>5000). Since CGCoupling.__init__ (cg_coupling.py:47) calls cg_selection, a tower containing one large label fails model construction even though its small-product paths (e.g. sym1 x sym1 -> sym2) are buildable.
- **F26** (CONFIRMED, src `vfe3/geometry/transport.py:334-342; vfe3/geometry/lie_ops.py:248-276 (_retract_core/retract_glk, max_norm=5.0); vfe3/families/gaussian.py:153; vfe3/config.py:66,147`): compute_transport_operators builds Omega=exp(phi_i)exp(-phi_j) with no conditioning guard, and the retraction's max-norm clamp (executable code in _retract_core, default max_norm=5.0) permits ||phi||=5; my probe (sp_n, N=2, [sym1,sym2], ||phi||=5) measured cond(Omega_01)=4.4e10, cond^2=1.9e21 (draw-dependent; the investigator's 6.6e7 is a milder draw), so the fp32 full-covariance sandwich can lose all precision. Reach corrected from latent to toggle: gauge_group='sp_n' and family='gaussian_full' are both registered config options with no validator forbidding the combination.
- **F27** (REFUTED, src `vfe3/model/cg_coupling.py:56,80; vfe3/run_artifacts.py:96,172,217,279; vfe3/geometry/cg.py:87-89`): The eigh-basis arbitrariness within an n_mult>1 null space is real (cg.py:87-89), but the claimed consequence is contradicted by the code: each intertwiner stack is a PERSISTENT registered buffer (cg_coupling.py:56, register_buffer default persistent=True), saved inside model.state_dict() (run_artifacts.py:96,172) and restored by strict load_state_dict on resume/eval (run_artifacts.py:217,279), overwriting any rebuild's eigh basis. The forward reads those buffers via _cast_buffers, so resumed path_weights stay bound to the checkpoint's slots; the fix's alternative ('persist the intertwiner stack alongside path_weights') is already the implemented behavior.
- **F28** (CONFIRMED, src `vfe3/geometry/cg.py:90-97; vfe3/model/cg_coupling.py:107; tests/test_cg.py (all tensors dtype=torch.float64, no float32 occurrences; only test file referencing use_cg_coupling)`): Build-time verification runs on the float64 C against a 1e-7 threshold (cg.py:90-97) while CGCoupling.forward casts intertwiners to mu.dtype (cg_coupling.py:107), fp32 in this project; my fp32 probe reproduced the claimed residuals exactly (l1xl1->l2 2.980e-8, l2xl2->l2 4.470e-8, l2xl2->l4 1.192e-7) and grep confirms tests/test_cg.py is all-float64 with no fp32 equivariance pin. Note the 1.19e-7 residual is at fp32 eps scale, i.e. equivariance is as exact as the dtype permits -- this is a test-coverage gap, not a broken invariant.
- **F29** (CONFIRMED, src `vfe3/viz/extract.py:184,270-275,355-360 (grep: zero head_mixer/cg_coupling occurrences in extract.py); vfe3/model/block.py:34-35,73-78; vfe3/model/stack.py:28-29,53; vfe3/model/model.py:483,805-806`): The vfe_block call in across_layer_belief_trace (extract.py:270-275) and the vfe_stack call in converged_state (extract.py:355-360) omit head_mixer and cg_coupling, which the signatures accept and which forward (model.py:483) and diagnostics (model.py:805-806) pass; under use_head_mixer/use_cg_coupling the replayed beliefs diverge from the model that actually ran. Additionally (beyond the cited lines) the vfe_stack call at extract.py:184 has the same omission and numerical_health replays via raw e_step_iteration, also mixer-free.
- **F30** (CONFIRMED, src `vfe3/model/cg_coupling.py:99-100,109-116 (grep: is_identity defined at cg_coupling.py:99 and head_mixer.py:118, called nowhere in vfe3/)`): forward unconditionally loops every path computing the outer product and einsum even when all path_weights are zero (the init state, where delta is exactly zero); is_identity() exists but is never consulted by forward or any caller. Pure efficiency finding, no correctness impact; note the suggested short-circuit must remain training-aware since zero-init weights need gradient flow.
- **F31** (CONFIRMED, src `vfe3/model/model.py:465,474,481-487 (mixer/CG forwarded at :483 inside 'with run, amp:'); vfe3/model/block.py:73-78; vfe3/model/model.py:130-141 (no warning at creation) vs :157-168,172-183,193-203,205-214,238-252 (warnings for connection_W/log_alpha/log_lambda_beta/prior-tables/pos_phi); vfe3/model/head_mixer.py:103-104; vfe3/model/cg_coupling.py:80; vfe3/config.py:299`): Confirmed: under effective 'detach' the entire vfe_stack including the post-E-step head_mixer/cg_coupling applications (block.py:73-78) runs inside torch.no_grad, so mixer_deltas and path_weights build no graph and stay frozen at their zero-init identity, and grep shows construction warnings exist for the five parallel footguns but none for use_head_mixer/use_cg_coupling. Severity corrected high->medium: it requires combining two default-OFF toggles (detach_e_step default False, config.py:299), produces frozen-at-identity behavior (no corruption, the model just runs as if the toggles were off), and is the same already-precedented missing-warning class as connection_W.
- **F32** (CONFIRMED, src `vfe3/model/model.py:935-943 (cg_coupling=self.cg_coupling at :941, no head_mixer kwarg) vs :483 (forward passes both) and :805-806 (diagnostics passes both)`): Confirmed: attention_maps' per-layer vfe_block replay passes cg_coupling but omits head_mixer, so under use_head_mixer=True each layer's converged belief (and the attention computed from it at :947-960) is mixer-free while forward and diagnostics apply the mixer per block; the asymmetry (CG wired, mixer not) marks it as an oversight. Reach noted as diag rather than toggle since attention_maps is figure-generation only (off the training path), though manifesting also requires the use_head_mixer toggle.
- **F33** (CONFIRMED, src `vfe3/viz/extract.py:184-190,270-275,355-361; vfe3/model/block.py:34-35,73-78; vfe3/model/stack.py:28-29,53; vfe3/model/model.py:483,805-806`): All three viz replay call sites pass connection_W but omit head_mixer/cg_coupling (grep of vfe3/viz finds zero hits for either), while the live forward (model.py:483) and diagnostics (model.py:805-806) pass both; vfe_block applies them to the belief before norm/handoff (block.py:73-78), so replayed beliefs diverge from the trained model whenever use_head_mixer or use_cg_coupling is ON. Corrections: line 355 is converged_state (not 'the free-energy decomposition'), the per-layer function is across_layer_belief_trace, and numerical_health/e_step_belief_trace replay via raw e_step_iteration which operates below the block level and also never applies mixer/CG. Defect lives entirely in figure-extraction code but only manifests under the non-default toggles.
- **F34** (CONFIRMED, src `tests/test_cg.py:155-172; vfe3/model/model.py:130-141,157-214,464-487,889-941; grep of tests/ for detach_e_step and use_head_mixer/use_cg_coupling`): test_cg.py:155-172 trains only under default unroll (no detach_e_step in _e2e_cfg) and no test combines detach_e_step=True with use_cg_coupling/use_head_mixer (grep: detach_e_step appears only in test_config, test_fix_model_audit, test_lambda_beta, test_learnable_alpha, test_model, test_straight_through, test_use_prior_bank); the freeze is code-plausible since model.py:465 wraps vfe_stack (where mixer/CG apply) in torch.no_grad under 'detach', and model.py warns for connection_W/log_alpha/log_lambda_beta/prior-tables under detach (157-214) but not for mixer/CG. No replay-parity test exists either (attention_maps tests at test_model.py:253-286, test_rope.py:98, test_son_irreps.py:228 never enable the toggles); moreover attention_maps itself passes cg_coupling at model.py:941 but omits head_mixer entirely, a live replay discrepancy the missing parity test would have caught — arguably raising this above pure test-gap severity.
