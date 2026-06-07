# Verified — math/theory checks (consult before re-verifying)

Per the CLAUDE.md audit/verification policy: each entry states WHAT was checked, whether it was
INCORRECT, and whether it was RIGOROUSLY verified. Consult this before re-verifying the same thing.

## 2026-06-07 — Overnight multi-agent deep audit + fixes (see docs/audits/audit-2026-06-07.md)

- **Full-covariance SPD retraction backward at the isotropic spectrum.** CHECKED, was **INCORRECT
  (NaN gradients)**, now FIXED + test-pinned + RIGOROUS. `retract_spd_full` / `retract_logeuclidean_full`
  (`geometry/retraction.py`) called `torch.linalg.eigh`, whose adjoint carries `1/(lambda_i-lambda_j)`
  gap terms that diverge on a degenerate spectrum. `Sigma=I` (the default `gaussian_full` prior init)
  is fully degenerate, so the first unrolled-E-step backward was 100% NaN on the DEFAULT estimator —
  the full-covariance pure path was unusable from its own default init. The forward is smooth there
  (`V f(lambda) V^T` is basis-independent); only the eigh adjoint blows up. Fix: custom autograd
  `_EighDamped` with a Lorentzian-damped gap `Delta/(Delta^2+gap_eps)` at every eigendecomposition site;
  forward bit-identical, backward finite.
  **CORRECTION (commit 63368fd, found by the audit's own pass 2):** the FIRST fix (f069a8a) shipped the
  adjoint with the WRONG SIGN (`delta_ij = w_i - w_j` giving `F = +1/(w_i-w_j)`; the symmetric-eigh
  adjoint requires `1/(w_j-w_i)`). The forward / NaN-cure was correct, but the backward returned
  plausible-but-WRONG gradients on `gaussian_full` whenever eigenvectors rotate. It shipped because the
  original "agreement" test used `(sqrtA*sqrtA).sum() = tr(A) = sum(w)`, which is EIGENVALUE-ONLY
  (`gV=0`), so the F-term (where the sign lives) never participated and the test passed for either sign.
  Now corrected: `delta_ij = w_j - w_i`, the test downstream rebuilt to be eigenvector-dependent (a
  fixed asymmetric contraction), and the adjoint VERIFIED against stock `torch.linalg.eigh` backward to
  MACHINE PRECISION (6.7e-14; the wrong sign was 0.44 off) + FD (1.3e-8) on that downstream + the
  strengthened test confirmed RED on the wrong sign / GREEN on the corrected one. Finiteness at `Sigma=I`
  holds for either sign (`F=0` at exact degeneracy). `tests/test_retraction.py` (now 9 eigh/full-cov
  tests incl. a mutation-discriminating model-level guard); full suite 612 passed.
  LESSON: an eigh-adjoint validation test MUST use an eigenvector-dependent downstream, else it is blind
  to the F-term sign. The damping is a small bounded bias near genuine degeneracies (factor capped at
  `1/(2 sqrt(gap_eps))`, gap_eps=1e-8); `gaussian_diagonal` is untouched. Commit f069a8a.
  WHY THE DAMPED GRADIENT IS CORRECT (not merely finite) at EXACT degeneracy: a smooth matrix function
  `V f(lambda) V^T` cannot depend on the arbitrary eigenvector choice within a degenerate eigenspace, so
  the true eigenvector-gradient contribution there genuinely vanishes — exactly what
  `F = Delta/(Delta^2+gap_eps) -> 0` produces, leaving only the correct eigenvalue path. So the fix
  returns the RIGHT gradient at `Sigma=I`, not just a non-NaN one (corroborated by the stock-eigh
  agreement test on well-separated spectra).

- **`min_lr_frac` intended default is 0.0.** CHECKED. The 2026-06-06 edit-doc states the proportional
  LR floor was added with default 0.0 ("keeps current behavior"); the committed 0.01 contradicted that
  and reddened two pre-existing scheduler tests. Restored to 0.0 (the proportional floor stays opt-in;
  `min_lr=min_lr_frac=0` is the pure half-cosine-to-zero). NOT a correctness/math defect. Commit ac19d2b.

- **Pure gauge transport path.** CHECKED, CONFIRMED CORRECT (clean negative, model-assembly gauge lens,
  `geometry/transport.py:449-450`). The flat phi-cocycle transport on the pure path is correct; no defect.

- **`cross_couplings` JSON round-trip.** CHECKED, was **INCORRECT (cold-start crash)**, now FIXED. A
  config.json reloaded by `viz.report._load_config` rebuilds `VFE3Config` with list pairs (JSON has no
  tuples), which failed the `isinstance(pair, tuple)` gate. Now coerced list→tuple in `__post_init__`.
  Implementation robustness, not theory. Commit ac19d2b.

- **Means-only RoPE invariant (documentation).** CHECKED, CONFIRMED (gauge lens). Means-only RoPE
  transports the mean under `R_i Omega_ij R_j^T` but the covariance under the bare `Omega_ij`, so the
  transported (mu,Sigma) is not a single congruence image and the Mahalanobis/affine invariant is not
  preserved for that belief — the coherent pure path is `rope_full_gauge` (`on_cov=True`). This is a
  by-design opt-in (not a code bug); the `RopeTransport` docstring was tightened to disclose it. No fix
  to behavior. (Consistent with the means-only-default / full-gauge-opt-in design in
  vfe3-positional-encodings memory.)

- **Diagnostics/metrics LOW findings — NOT yet addressed (deferred follow-up).** CHECKED, several
  registered metrics are subtly off but all are diagnostics-only (off the differentiated F): registered
  `holonomy_deviation` uses the biased row-major estimator not the sampled one; `effective_rank` passes
  raw sigma not eigenvalues under full-cov; the `attention_entropy` registered metric returns Shannon
  H(beta) while the CSV column under the same key carries `tau*KL(beta||pi)`; `gauge_trace_spread` is
  identically 0 for unimodular SO/Sp; `fisher_trace` inverts an unfloored matrix; `condition_number`
  mis-ranks a non-PD input; several `FullGaussian` non-PD paths discard the `safe_cholesky` ok-mask
  (finite-but-wrong instead of NaN→kl_max, confined to the opt-in alpha>1 full-cov toggle). None affect
  training; recommended as a single diagnostics-hardening pass. Do NOT re-flag as new.

## 2026-06-05 — Codex deep-audit triage (see docs/audits/audit-2026-06-05-codex-triage.md)

- **Regime-II connection_W gauge-equivariance (audit Finding 6).** CHECKED, CONFIRMED, RIGOROUS.
  The opt-in regime_ii edge factor exp(delta_ij·G), delta_ij^a = mu_i^T W^a mu_j, is gauge-INVARIANT
  (NOT covariant — the vertex factors exp(phi_i), exp(-phi_j) already carry the full g_i(.)g_j^{-1}
  conjugation, so the middle factor must be unchanged) iff g_i^T W^a g_j = W^a for all group elements
  g; setting g_i=I gives W^a(g_j - I)=0, so the ONLY constant solution is W^a=0. Hence a trained
  nonzero connection_W breaks strict gauge equivariance — exact at zero init, deviates as W drifts
  (empirically monotone in ||W||: 0 at W=0, 62.3 at ||W||~1 vs ||E||~9.6), the same footprint as the
  head mixer. Codex's "covariance / W -> Ad(g) W / constrain to an invariant family" framing is wrong
  (the only invariant is W=0). Now disclosed in CLAUDE.md and pinned by
  tests/test_regime_ii.py::test_regime_ii_edge_factor_breaks_gauge_invariance_for_nonzero_W. The other
  audit findings are implementation/diagnostics/reporting (F1 loader split-semantics, F4 registry
  consistency, F7 holonomy estimator, F8 banner tau), not theory — see the triage doc.

## 2026-06-05 — Rényi/alpha-divergence + alpha_div ablation (see docs/audits/audit-2026-06-05.md)

- **Gaussian Rényi closed forms (diagonal, full-cov, per-coordinate).** CHECKED, CORRECT, RIGOROUS
  (CONFIRMED). `vfe3/families/gaussian.py:88-99,101-131,230-256` match the canonical multivariate
  Gaussian Rényi D_alpha(q||p) = (alpha/2)Delta^T Sigma_a^{-1} Delta - 1/(2(alpha-1)) ln(|Sigma_a| /
  (|Sigma_q|^{1-alpha}|Sigma_p|^alpha)), Sigma_a = (1-alpha)Sigma_q + alpha Sigma_p (van Erven &
  Harremoës 2014; Gil/Alajaji/Linder 2013), term-for-term, correct q||p orientation. Passes
  self-divergence-zero, non-negativity, monotonicity in alpha, alpha->1 KL limit, full-vs-diagonal
  agreement, per-coord summation. Configured alpha genuinely reaches `renyi_closed_form` for BOTH the
  attention energy and the belief gradient (alpha=1-only geometric-mean kernel correctly gated off for
  alpha!=1; Rényi is NOT silently running as KL). For alpha in (0,1) the blend is convex -> SPD;
  autograd gradient finite and FD-matched to fp32; saturates kl_max LESS than KL.

- **Softmax-beta stationarity + EM separation under Rényi.** CHECKED, CORRECT, RIGOROUS (CONFIRMED).
  The beta-block gradient equals the `-tau log Z` envelope gradient to ~1e-7 at alpha=0.5 and 1.0;
  the row-Lagrangian needs only linearity in beta + the entropy term, not KL (disclosed
  `GL(K)_supplementary.tex:1084`). E-step takes only priors (no labels) -> EM separation intact.

- **Variational-bound status for alpha!=1 — INTERPRETIVE, author-disclosed.** CHECKED. For alpha!=1
  F is an entropy-regularized consensus functional, NOT an evidence bound (the alignment block is no
  longer KL[q_i||Omega_ij q_j]). Disclosed: forward KL is the unique f-divergence preserving exp-family
  closure + dual attention (`GL(K)_attention.tex:771`); closed-form stationary belief is alpha=1-specific
  (`GL(K)_supplementary.tex:1096`). Heuristic generalization by design, not a defect.

- **alpha_div ablation is CONFOUNDED (the user's symptom).** CHECKED, CONFIRMED, reproduced. NOT a
  divergence-math bug. alpha_div=1 uses the live analytic kernel; alpha_div!=1 falls to the autograd
  oracle which, under the default `oracle_unroll_grad=False`, returns a DETACHED belief gradient
  (`oracle.py:118`, gated by `e_step.py:357` + `config.py:247`). With `e_phi_lr=0`, that detaches the
  prior + gauge-frame tables from the loss (prior grad 651->0; phi_embed/pos_phi_free grad -> None),
  giving both the 2.5x speedup (dropped backward graph; oracle is 1.9x slower PER CALL) and worse PPL.
  Falsifier = the sweep's own alpha_div=0.99 row: PPL 159->273, wall 1672->695s discontinuously at the
  kernel gate `abs(alpha-1)<1e-9`, impossible for continuous divergence-order. Pure path EXISTS
  (`oracle_unroll_grad=True`, test-pinned `tests/test_e_step.py:306`); fix = force it on alpha_div!=1
  sweep rows and rerun. Two real non-bug effects remain (self-coupling magnitude shift via larger
  alpha^(k) for alpha<1; softer attention at fixed tau).

- **Latent (not the user's regime):** diagonal alpha>1 `sigma_blend.clamp(min=eps)`
  (`gaussian.py:89,123`) masks a non-PD blend as a WRONG finite value instead of NaN->kl_max (full-cov
  handles it correctly) — MEDIUM, alpha>1 diagonal only. KL-branch threshold mismatch (kernel 1e-9 vs
  formula 1e-6) and float32 hard-switch near alpha=1 with no Taylor branch — LOW, no config sits there.

## 2026-06-03 — Deep-audit corrections (see docs/audits/audit-2026-06-03.md)

- **Per-block sl(K)/trace det-control under `tied_block_glk`.** CHECKED, was **INCORRECT**, now FIXED
  + test-pinned. `project_phi_to_slk` / `clamp_phi_trace` (`lie_ops.py`) used the per-block-independent
  orthogonal projection `coeffs = s/||V_h||^2`, valid only when the per-block trace functionals V_h are
  mutually orthogonal. Under the tied gauge the generators kron(I_n, E_ij) make all n_heads V_h rows
  identical, so the projection over-subtracted by a factor of n_heads (sign flip + n_heads x; det Omega
  -> 0.68 not 1.0). Replaced with the JOINT Gram solve `coeffs = s @ pinv(V V^T)`, which (i) drives each
  block's trace to 0 / clamps it correctly under tied, and (ii) reduces to `1/||V_h||^2` for an
  orthogonal (untied) basis (diagonal Gram) -- within golden tolerance, since `pinv` is SVD-based and
  not bit-exact (the untied tests pass at atol 1e-5), so untied `block_glk` is unchanged to tolerance.
  Pinned by
  `tests/test_phi_retraction.py::test_project_slk_zeros_block_trace_under_tied_gauge` and
  `..._clamp_phi_trace_bounds_block_trace_under_tied_gauge`; the pre-existing untied tests still pass.

- **`sigma_max` ceiling convention across the cov_kind retraction seam.** CHECKED, was **INCONSISTENT**,
  now FIXED + test-pinned. `retract_spd_diagonal` clamped the **variance** to `sigma_max`, but
  `retract_spd_full` (and `retract_logeuclidean_full`) clamped the **eigenvalues** to `sigma_max^2` --
  eigenvalues ARE variances, so the same physical quantity was bounded a factor `sigma_max` looser on
  the full family under one shared knob. Changed the full-cov eigenvalue ceiling to `sigma_max`. Pinned
  by `tests/test_retraction.py::test_sigma_max_caps_variance_consistently_across_diag_and_full`. (Note:
  the centering axiom R(Sigma,0)=Sigma now holds only within the variance box on BOTH arms, as the
  diagonal arm already did -- the full identity test was re-baselined to a non-binding sigma_max.)

- **phi E-step descends the active connection regime.** CHECKED, was **INCORRECT**, now FIXED +
  test-pinned. `phi_alignment_loss` built its Omega with `_transport(phi, group)` at defaults (flat),
  so under `transport_mode='regime_ii'` + `e_phi_lr>0` the phi update descended the flat objective while
  mu/sigma descended the regime_ii one. Threaded `transport_mode`/`connection_W`(detached)/`cocycle_relaxation`
  through `phi_alignment_loss`. Pinned by `tests/test_regime_ii.py::test_phi_estep_descends_regime_ii_not_flat`.
  (This was a real correctness gap confined to the opt-in regime_ii NN-exception toggle; the pure flat
  path was and is correct.)

## 2026-06-02 — Gamma model-coupling block (hyper-prior increment 2)

Code: `vfe3/model/model.py` (gamma block in `forward`), `vfe3/model/prior_bank.py` (s-table gate),
`vfe3/config.py` (gamma_coupling/kappa_gamma/gamma_attention_prior + tau_gamma). Tests:
`tests/test_gamma_coupling.py` (12). Verified by a 5-lens adversarial workflow (each verifier told to
falsify) plus the test oracles. Suite at completion: 438 tests, 0 failures, 0 errors.

- **Envelope assembly = canonical `-tau_g log Z^s_i`.** CHECKED, CORRECT, RIGOROUS. The gamma block
  assembles `gamma_coupling * mean_i [ -tau_gamma log Z^s_i ]`, the reduced/envelope form of
  `sum_ij [ gamma_ij KL(s_i||Omega_tilde_ij s_j) + tau_g gamma_ij log(gamma_ij/pi^s_ij) ]`
  (`Participatory_it_from_bit.tex` eq:pointwise_free_energy, 1241-1249; reduction eq:free_energy_reduced
  1383-1397). The envelope identity `sum_j beta E + tau sum_j beta log(beta/pi) = -tau log Z` was
  verified symbolically (sympy → 0). A from-scratch per-head reimplementation matched the model's loss
  delta to 5.96e-8. `tau_gamma = kappa_gamma*sqrt(d_head)` confirmed.

- **Energy orientation `E_s[i,j] = KL(s_i || Omega_ij s_j)`.** CHECKED, CORRECT, RIGOROUS. A
  primitive-free K=2/N=2 hand-matmul confirmed the transported key is `Omega_ij @ s_j` (left action,
  key index j) and FALSIFIED both `Omega_ij^T @ s_j` and `Omega_ji @ s_j`; the diagonal covariance is
  the sandwich `diag(Omega_ij diag(sigma_j) Omega_ij^T)`. Independently re-pinned in the suite by
  `test_gamma_energy_equals_analytic_kl_at_nonzero_phi` (analytic diagonal-Gaussian KL at nonzero phi,
  Omega != I, formula-independent of transport_mean/covariance and of the renyi kernel).

- **Detach / predictive inertness.** CHECKED, CORRECT, RIGOROUS. `Omega` built from
  `out.phi.detach()`, so `autograd.grad(gamma_term, [phi_embed, mu_embed, sigma_log_embed])` returns
  `None, None, None` (graph-level disconnection); grad reaches only the s tables. Forward logits/ce are
  byte-identical to the gamma=0 path (`torch.equal`). A no-detach mutation probe makes the
  phi-grad-equality test FAIL, proving the test discriminates.

- **Byte-identity of pre-existing paths.** CHECKED, CORRECT, RIGOROUS. The PriorBank s/r gate split
  (s on `lambda_h>0 OR gamma_coupling>0`, r on `lambda_h>0`) introduces no RNG reordering (s drawn
  before r; r tables are RNG-free zeros/full). Pure-path RNG state identical; existing `lambda_h>0`
  build draws byte-identical s and r. No `'s implies r'` dependency. The 3 new config fields are
  default-safe.

- **Manuscript-fidelity CONCERN (not a defect in what was built).** CHECKED. The implemented term is
  the CORRECT model-coupling term with its own `s_i`, `pi^s` (`gamma_attention_prior`), and `tau_gamma`,
  distinct from the belief beta block. The KNOWN, DOCUMENTED scope reductions are: (i) `s_i` is a static
  per-token table, not the inferred field the manuscript varies (eq:envelope_gradient_model 1410-1419);
  (ii) the detach severs the `phi<-gamma` coupling line 1420 keeps under a shared frame; (iii) `.mean()`
  over (B,H,N) vs the canonical sum-over-ij (a free coupling scale); (iv) diagonal sandwich (belief-
  family parity); (v) flat cocycle under regime_ii. The manuscript sets `gamma_ij=0` in its reported
  sims (line 1296), so this is a buildout target, not a fidelity fix. Items (i)-(ii) ARE the deferred
  s->q design — left for explicit design input, NOT built blind.

## 2026-06-02 — s->q coupling (prior_source="model_channel", Realization A)

Code: `vfe3/config.py` (prior_source field), `vfe3/model/prior_bank.py` (_prior_mu_table/
_prior_sigma_log_table accessors + 5 rerouted prior reads), `vfe3/model/model.py` (wiring),
`vfe3/train.py` (s/r optimizer groups). Tests: `tests/test_prior_source.py` (8). User-chosen design:
REPLACE the belief prior with the model channel, p_i = s_i. Verified by a 5-lens adversarial workflow
(4 returned; code-quality lens failed to emit structured output, self-assessed). Suite: 446 / 0 / 0.

- **Reroute consistency (p_i = s_i everywhere).** CHECKED, CORRECT, RIGOROUS (CONFIRMED). A whole-tree
  grep found exactly five prior-VALUE reads (encode + 4 decode kernels: diagonal/full/chunked/
  reference), ALL routed through the single accessor pair; the E-step self-coupling, the M-step
  self-coupling rebuild, the prior_handoff fold, and diagnostics() all consume the rerouted encode
  output (no direct mu_embed read). Empirical directional checks on every path: perturbing mu_embed
  leaves output invariant, perturbing s moves it.

- **Byte-identity of the default 'token' path.** CHECKED, CORRECT, RIGOROUS (CONFIRMED). The accessor
  returns the LITERAL same object (`_prior_mu_table() is mu_embed` -> True) on token, so the
  catastrophic-cancellation-sensitive decode is byte-for-byte unchanged. RNG order preserved (s drawn
  last, only when model channel active). Copy-equivalence (s := belief tables -> byte-identical) is a
  genuine non-vacuous torch.equal oracle, now also pinned through the M-step self-coupling rebuild.

- **Trainability.** CHECKED, CORRECT, RIGOROUS (CONFIRMED). The s tables (the LIVE prior under
  model_channel) are grouped into the optimizer (mean@m_mu_lr, log-scale@m_sigma_lr); build_optimizer's
  exact-coverage guard passes; loss decreases 2.996->2.418 over 20 steps. Dead mu_embed (grad None)
  is skipped by AdamW. The hyper-prior centroid r (lambda_h>0) is FROZEN (requires_grad=False; user
  decision): a fixed centroid per the manuscript's "higher, slower meta-level" (supp:1081); the coverage
  guard now exempts non-trainable params, so build_optimizer works for lambda_h>0 (s grouped, frozen r
  skipped), and the hyper-prior channel trains end-to-end (s trains, r fixed). Still ungrouped & RAISES
  (PRE-EXISTING, genuinely trainable): log_alpha (alpha_mode='learnable') and connection_W
  (transport_mode='regime_ii').

- **Manuscript-fidelity CONCERN — same-scale vs cross-scale s->p (IMPORTANT, user decision).** CHECKED,
  primary source READ. The realization p_i = s_i is the identity-conditional special case of the
  SAME-SCALE hierarchical-Bayes prior in GL(K)_supplementary.tex:1083-1085 (p_i(k_i) = integral
  p_i(k_i|m_i) s_i(m_i) dm_i) — mathematically faithful to THAT equation. BUT the main
  Participatory_it_from_bit.tex:1440 makes p_i a CROSS-SCALE shadow (the meta-agent's belief q^(s+1)
  transported down, eq:cross_scale_shadow) and states verbatim "s_i does not act through p_i at the
  same scale" (there s_i is regulated only by its own hyper-prior r_i). So the two manuscripts carry
  DIFFERENT s->p mechanisms; this increment realizes the supplementary's same-scale reading, which the
  main manuscript's text contradicts. INITIAL CITATION WAS WRONG (attributed p_i(k_i|m_i) to
  Participatory; it is in the supplementary, and Participatory says the opposite) — corrected in
  config.py / test docstring, and the tension is now disclosed there. The cross-scale realization would
  need a meta-agent/scale-(s+1) object that does not exist. NOT a code/math flaw; a documented
  design-choice-in-a-theoretical-tension that the user (manuscript author) should adjudicate.
