# GL(K) attention manuscript — deep review pass 11 (manuscript-vs-implementation fidelity)

Date: 2026-06-20. Targets: the canonical vault `GL(K)_attention.tex` and `GL(K)_supplementary.tex`, checked against the V3 reference implementation in `C:\Users\chris and christine\Desktop\V3_Transformer\vfe3\`.

Pass 11 opens a new axis. Passes 3 through 10 reviewed the GL(K) manuscripts along intra-document axes: analytic mathematics (exhausted), citations, claim-status framing, notation, convention, and the LaTeX build. The one major lens never applied systematically to these two files is manuscript-versus-implementation fidelity — does the code actually implement the mathematics the papers claim, and do the theoretically-pure paths the papers describe exist in the code? This is the deeper, prose-versus-reality pass, and it directly serves the project's standing instruction to focus on the actual code and to ensure the pure paths exist.

Framing (to avoid false positives). The manuscript's named/released code is the separate `epistemic-geometry` repository (attn:2424), not this V3_Transformer clean-room; the ledger already adjudicates that a redirect to V3's entry point would be false. V3 is a clean-room continuation "numerically pinned to VFE_2.0 by golden tests" that implements the same mathematics. The pass therefore checked the manuscripts' mathematics and algorithms against the V3 reference implementation of that math, and verified pure-path existence, while treating config-default/toggle differences and repo-naming as out of scope per the project's standing audit rule (the concern is whether the pure path exists, not whether it is the default).

Method: five `audit-implementation-engineer` agents dispatched in parallel, one per cluster (attention kernel/softmax/temperature/priors; gauge transport/RoPE/groups/holonomy/toggles; SPD geometry/retraction/dexp/preconditioner/numerics; free energy/EM/envelope/mean-field/couplings; decode/prior-bank/families/divergence-uniqueness), each seeded with the verified-ledger do-not-reflag list (including the settled code adjudications), each tracing every load-bearing claim to its V3 code path, recomputing the mathematics in python against the actual modules, and running the relevant test suites.

Headline: full fidelity confirmed. All five clusters returned no finding — no mathematical or algorithmic divergence between the manuscripts and the V3 implementation, and no pure-path-existence gap. Every load-bearing GL(K) equation and algorithm is faithfully implemented (recomputation residuals from 1e-7 fp32 round-off down to machine precision), and every theoretically-pure path the manuscripts describe exists and is config-reachable, most as the active default. No high/critical finding was produced, so the adversarial-skeptic stage had nothing to verify.

## Recommendation

No action. The GL(K) manuscripts faithfully describe the V3 reference implementation of their mathematics, and the pure paths exist. This is an honest clean result for the fidelity axis, not an absence of scrutiny — each cluster was traced to executable code and recomputed.

## Fidelity confirmed, by cluster

### Attention kernel / softmax / temperature / positional priors
- `β = softmax(−E/τ)` with the log-prior added after the `1/τ` (no `1/τ` on the bias): `free_energy.py:276-279` (`logits = -energy/τ; logits = logits + log_prior`); recomputed `max|code − softmax(−E/τ + B)| = 0` vs the wrong `B/τ` form deviating 0.26. `τ = κ√d_head` (`attention_tau`, `free_energy.py:41-76`), the irrep tower giving per-head `κ_h√d_h`.
- Attention score `D_KL(q_i‖Ω_ij q_j)` on the transported key (`pairwise_energy`, `free_energy.py:120-182`, with `transport_mean`/`transport_covariance`), equal to the geometric bias `S(Ω) + (1/2σ²)‖Ω⁻¹μ_i − μ_j‖²` to fp32 tolerance. Value aggregation `Σ_j β_ij Ω_ij μ_j` realized as the E-step mean-update stationary target (`gradients/kernels.py:135`).
- Causal / sliding-window / ALiBi / T5 priors all enter as an additive untempered log-bias (`attention_prior.py`); ALiBi slope `m = 2^{−8h/H}` (`_press_slopes`); T5 bucketed bias. Pure uniform prior `π=1/N` and deterministic T5 default `−log1p(bucket)` both reachable without the learned table.

### Gauge transport / RoPE / groups / holonomy / toggles
- Covariance transport is the sandwich `ΩΣΩᵀ` (`transport.py:760`, recomputed to 3.8e-6); means transport `Ωμ`; the dual `Ω^{-⊤}` enters the scoring metric via the transported-key precision `(ΩΣΩᵀ)⁻¹`, and the full score is gauge-invariant under `Ω → g_iΩg_j⁻¹`. Cocycle `Ω_ijΩ_jk=Ω_ik`, `Ω_ii=I` (`compute_transport_operators`, residual 3.6e-7); flat Regime-I holonomy ≈ 0 (7e-7); curvature nonzero only under the Regime-II bilinear `connection_W` nn.Parameter (holonomy 1.1e-6 at W=0, 6.1e4 at W≠0). Flat transport is the registered default.
- RoPE as a gauge transport on the `SO(2)^{d/2}` subgroup, `R(θ_i)ᵀR(θ_j)=R(θ_{j−i})` (6e-8); the RoPE-folded operator stays a flat cocycle. Multi-head `GL(d_head)^H ⊂ GL(K)` (off-block entries exactly 0); generators/irreps carry build-time homomorphism asserts; Clebsch-Gordan intertwiner equivariance to 2.8e-16.
- Pure paths for the equivariance-breaking opt-ins all present and default-OFF: head_mixer (identity-init bit-identical to OFF; tied-block variant exactly equivariant), cg_coupling (zero-init, exactly equivariant), Regime-II connection_W (flat is the default), rope_full_gauge (bool).

### SPD geometry / retraction / dexp / φ-preconditioner / numerics
- SPD retraction = Pennec affine-invariant exp-map `Σ^{1/2}exp(Σ^{-1/2}(τdΣ)Σ^{-1/2})Σ^{1/2}` (`retraction.py:134-193`, residual 1.1e-7 fp32, SPD-preserving), the diagonal arm `σ·exp(τdσ/σ)`; the registered default `spd_retract_mode="spd_affine"`. dexp SO(3) Rodrigues coefficients reproduce the FD right-trivialized differential to 2.6e-10; the general `Ψ(ad_φ)=Σ z^k/(k+1)!` series matches the Rodrigues metric to 8.2e-14 (the manuscript correctly restricts the closed form to SO(3) and states the live GL(K) path flows through autodiff of `torch.matrix_exp`).
- Killing-form preconditioner `B = 2K tr(XY) − 2 tr X tr Y` (residual 0.0 vs both the formula and `tr(ad_X ad_Y)`); Fisher natural gradient `δμ=Σ∇_μ`, `δΣ=2Σ sym(∇)Σ`. Eigenvalue floor `eps=1e-6` + `σ_max` ceiling, no condition cap (matching the ledger); matrix-exp argument clamped to [−50,50]; float32 preserved end-to-end with an `autocast(False)` fp32 island.

### Free energy / E-step-M-step / envelope / mean-field / couplings
- Canonical `F_red = Σ_i α_i KL(q_i‖p_i) − τ Σ_i log Z_i − E_q[log p(o|k)]` assembled term-by-term (`free_energy.py:327-403`); the envelope identity `Σβ*E + τΣβ*log(β*/π) = −τ log Z` to 2.4e-7. The analytic kernel gradients equal the autograd-of-F oracle to 4.8e-7 (mean) / 3.7e-8 (sigma), with `tests/test_gradients_oracle.py` 7/7 passing (FD-anchored). Target-blind E-step (the observation term enters only the M-step cross-entropy); structural EM as the manuscript states.
- α_i registry with the pure `constant=1.0` path plus state-dependent and learnable variants (the envelope cancellation `d/dD[α*D + R(α*)] = α*` verified); λ_h hyper-prior coupling wired (`model.py:901`), γ_ij model-coupling channel present (off by default). The pure full-gradient (smoothing) path is config-reachable and FD-correct.

### Decode / prior bank / families / divergence uniqueness
- Decode logit = Bishop quadratic discriminant + variational plug-in = `−KL(q_i‖π_v)/τ_eff` with the full per-position constant carried (so the absolute logit, not just the softmax-relative one, equals `−KL/τ`; C=0 confirmed); `τ_eff = τ_cfg·e^{−s}` (`prior_bank.py:292-298`). Both decode paths exist and are config-selected through one `use_prior_bank` gate routing `get_decode(mode)`: the pure KL-to-prior decode (default; no linear weight created) and the learned-linear ablation `mu@Wᵀ` (exact). Full-covariance pure decode runs end-to-end.
- Exponential-family modularity behind the `BeliefParams` registry (Gaussian diagonal/full + the genuinely non-Gaussian Laplace), Gaussian KL matching `eq:supp_gaussian_kl` exactly; divergence registry with forward-KL as the default (Rényi α=1) and the negative-entropy Bregman identity holding numerically. 89/89 cluster tests pass.

## Observations (not findings)

- The supplement's SPD-retraction constants (`eps_SPD=1e-4`, `kappa_max=1e4`, supp ~654-659) still differ from the released `1e-6` floor / no-cap — the already-open documentation gap recorded in ledger §4, not re-raised.
- The `sigma_max` config default (10.0) versus the kernel default (5.0) is an internal config-default discrepancy that does not affect path correctness; out of scope per the standing audit instruction (config defaults/toggles are intentional working state), noted only for completeness.
- The `log_likelihood` argument in `free_energy()` is a gated stub with no live caller; this is fidelity, not a gap — the E-step is target-blind by the manuscript's own Algorithm 1, and the `−E_q[log p(o|k)]` term is realized as the M-step cross-entropy.

## Method

Five `audit-implementation-engineer` agents in parallel, one per cluster, each given the canonical vault manuscripts, the V3 `vfe3/` source tree, and the verified-ledger do-not-reflag list (including the settled code adjudications: the `epistemic-geometry`-vs-V3 distinction, `max_condition` belonging to VFE_2.0, the phantom-method fixes, the Regime-II bilinear connection). Each traced every load-bearing manuscript claim to its V3 code path, recomputed the mathematics in python against the actual modules (reporting residuals), and ran the relevant test suites. No finding survived in any cluster, so the adversarial-skeptic stage did not fire.
