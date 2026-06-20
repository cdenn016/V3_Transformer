# GL(K) attention manuscript — deep review pass 7 (completion: recompute-driven, adversarially verified)

Date: 2026-06-20. Targets: the canonical `Research/manuscripts/GL(K)_attention.tex` and `GL(K)_supplementary.tex`.
This pass completes the partial pass-7 run that was interrupted. It confirmed the three actionable items left open,
then ran six fresh recomputation lenses over under-examined sections, with adversarial three-voter verification on
high/critical findings, plus a completeness critic. The completion workflow dispatched 17 agents (~1.0M tokens);
every load-bearing identity was independently recomputed in python (numpy/sympy) rather than eyeballed. The headline
result: after six prior passes the manuscript is mathematically robust — the sweep produced one genuine new defect
(the App I decode constant) and a short list of equation-level precision fixes; fifteen separately recomputed
identities returned correct or already-candid.

## Applied this pass

Originals recorded for each. All edits made in the canonical vault manuscripts.

### `GL(K)_supplementary.tex`

1. App I decode constant (eq:supp_kl_decomposition, line ~1407). The text stated `C = -(K/2) log 2pi - K/2`. Three
   independent voters plus the numerical-stats lens recomputed the constant symbolically and by Monte Carlo: the
   cross-entropy identity `E_q[log p] = -D_KL(q||pi_v) - H(q)` is exact, so `C = 0`. The stated value double-counts
   the `(K/2) log(2 pi e)` constant part of the differential entropy (residual with the stated C is `+5.6758` for
   K=4, identical across trials; residual with C=0 is machine zero). The defect is inert for every downstream claim
   because C is `v`-invariant and cancels in the softmax (eq:supp_softmax_cancellation), but it was a flatly wrong
   displayed identity in a rigorous appendix. Replaced with `C = 0` plus a one-clause explanation; the `+C` / `-C`
   appearances at the decomposition, the softmax cancellation, and the accuracy-term identification (line ~1442)
   remain correct with the constant equal to zero.
   - old: `... and $C = -\tfrac{K}{2}\log 2\pi - \tfrac{K}{2}$ is a $v$-invariant scalar constant. Both $H(q_i^{\star})$ and $C$ depend on the row index $i$ but not on the column index $v$.`

2. Connection transformation law on bundle overlaps (line ~176). The displayed law
   `A^(i) = Omega A^(j) Omega^-1 + Omega d Omega^-1` is wrong under the manuscript's own left-invariant
   (left Maurer-Cartan) convention `A^(i) = U_i^-1 d U_i` (line 152, consistent with the `+[A,A]` curvature at line
   158). Finite-difference recomputation (K=3, smooth non-abelian paths): the displayed form gives residual `44.5`;
   the "obvious" textbook fix `Omega A^(j) Omega^-1 + (d Omega) Omega^-1` ALSO fails here (residual `29.1`), because
   that adjoint form belongs to the right-invariant convention `A = (dU) U^-1`. The adversarial verifier flagged
   exactly this trap. The correct left-invariant law, re-derived by product rule and confirmed numerically (residual
   `3.4e-10`), is
   `A^(i)_mu = A^(j)_mu + U_j^-1 (Omega_ij^-1 d_mu Omega_ij) U_j`.
   Applied that form, plus a one-sentence note giving the familiar adjoint form and stating it is recovered under the
   right-invariant convention (with the field strength then carrying `-[A,A]`). This is a foundational-scaffold
   equation that is not load-bearing downstream (line 179 routes downstream derivations through the GL(K)
   KL-invariance theorem); the cocycle, holonomy, and curvature identities in the same section were independently
   re-confirmed correct (residuals `9.6e-16`, `1.5e-15`, `5.7e-10`). Note: switching the whole section to the
   right-invariant convention (the textbook Yang-Mills look) is the alternative; it is a larger coupled change to the
   connection definition and curvature sign and was left to author preference.
   - old: `A^{(i)}_\mu = \Omega_{ij} A^{(j)}_\mu \Omega_{ij}^{-1} + \Omega_{ij} \partial_\mu \Omega_{ij}^{-1}.`

3. SPD trust region (sec:retraction lead-in, line ~640). The prose described only the full-covariance Frobenius
   clip. Code check (retraction.py): the full arm scales `||R||_F` (lines 173-177) but the diagonal arm clamps the
   per-coordinate whitened ratio `delta_sigma/sigma` componentwise to `[-rho, rho]` (line 128), an L-infinity bound;
   defaults also differ (diagonal 5.0, full 2.0). For an in-box tangent the two arms can take steps differing by up
   to `sqrt(K)`. Added a parenthetical documenting the diagonal L-infinity clip. (Already logged in code memory ID
   21765; the manuscript prose still omitted it.)
   - old: `... by clipping its Frobenius norm to a maximum relative step size $\rho_{\mathrm{tr}}$.`

4. App G "Vacuum State" prose and figure caption (lines ~1048, ~1053). The figure (`Symmetric Vacuum.png`) shows the
   six agents converging to a common nonzero norm (mean `||mu|| = 0.2857`, `Var(||mu||) = 2.8e-14`) at six DISTINCT
   directions in the 9-dimensional `ell=4` representation space — a shared gauge ORBIT, not a common point.
   Representation theory (recomputed two ways: stacked-generator kernel and Haar projector) confirms the `ell=4`
   SO(3) irrep has no nonzero invariant vector, so "rotationally invariant vacuum state ... collapsing to a common
   point" is imprecise; the invariant object is the orbit / the common norm. Reworded the prose and the bottom-left /
   bottom-right caption clauses to say "common gauge orbit" and "distinct directions of a common norm sphere".
   - prose old: `Without observations, all agent beliefs converge to a shared rotationally invariant vacuum state with identical norms ..., occupying the gauge orbit predicted by the symmetric vacuum theory.`
   - caption old: `... showing agents collapsing to a common point. ... confirming zero variance across agents and preserved gauge symmetry.`

### `GL(K)_attention.tex`

5. Multi-head block transport (eq, line ~1769). The per-head transport was written
   `Omega^a = (sigma^2 W_K^a (W_Q^a)^T)^-1 in GL(d_head)`, but line 1772 declares `W_Q^a` rectangular
   (`d_k x d_head`), so `W_K^a (W_Q^a)^T` is `d_k x d_k` of rank at most `d_head` — singular, not invertible, not in
   GL(d_head). The invertible head-space object is the thin-SVD factor `A^a in GL(d_head)` (already used at
   eq:head_space_kernel line 1278 and the value path line 1346). Replaced `W -> A`:
   `Omega^a = (sigma^2 A_K^a (A_Q^a)^T)^-1 in GL(d_head)` (verified `3.3e-11` against the identity-derived transport).
   The single-head form at line 1761 uses square `W in GL(d_k)` and is correct as written.
   - old: `\Omega^a = (\sigma^2 W_K^a (W_Q^a)^\top)^{-1} \in \mathrm{GL}(d_{\text{head}}),`

6. Residual temperature mislabel (Discussion, line ~2270). The grand mean `r_bar = 0.804` and posterior `0.867` were
   attributed to "the theory-predicted optimal temperature `tau ~ 2 sqrt(d_k)`", but those statistics were measured
   at the empirical optimum `tau = 19.0`, not at the theory value `16` (`(19-16)/16 = 18.8% ~ 19%`). The parallel
   one-line summary at line 2039 already phrases this correctly. Relabeled to the empirical optimum with the theory
   value `2 sqrt(d_k) = 16` stated alongside, mirroring line 2039. (The supplementary caption the partial run
   expected to fix, line ~737, was already correct; this Discussion line was the surviving instance.)
   - old: `... at the theory-predicted optimal temperature $\tau \approx 2\sqrt{d_k}$ (squared-distance form; ...)`

7. Diagonal-covariance RoPE config token (line ~2367). The Limitations paragraph wrote
   `rope_full_gauge='off'` (string), but config.py:151 declares `rope_full_gauge: bool = False` (all call sites use
   the boolean; the means-only path is gated by `not self.rope_full_gauge`). The SE(K)-breakage math the sentence
   asserts is correct; only the config type was misquoted. Changed to `rope_full_gauge=False` (means-only).
   - old: `... under the \texttt{rope\_full\_gauge=\textquoteleft off\textquoteright} configuration,`

8. Central-QK key-norm reduction (line ~1186, optional tightening). The text correctly absorbs the `j`-only key-side
   terms into `log pi_ij`, but a reader could think a `log det Sigma_j` bias survives. Under the stated closure
   `Sigma_j = U_j C U_j^T` the log-determinant terms collapse: `log det Sigma_j - 2 log|det U_j| = log det C`
   (`j`-independent). Added one sentence making this explicit so the absorbed bias is unambiguously
   `b_ij = -r_j/(2 tau) + const` with no surviving `j`-dependent log-det term. (Flagged by the completeness critic in
   passes 6 and 7.)

## Recomputed and cleared (no change — robustness confirmation)

Each independently recomputed by the assigned lens and, where flagged, an adversarial verifier; all returned correct
or already-candid:

- Softmax-`beta` stationarity WITH the entropy term, and the delta degeneracy WITHOUT it (row-Lagrangian, sympy).
- Surrogate-vs-envelope gradient gap `= -tau^-1 Cov_beta(E, dE/dx)` (exact symbolically) — this answers the
  completeness critic's "eq:autograd_envelope_gap unverified" note: it is now verified.
- Geometric-mean Boltzmann belief `q_i^* propto e^{-H/2} prod (Omega q_j)^{beta/2}` (correct 1/2 exponents).
- EM separation / mean-field factorization `Q = q_i beta` (target-blind E-step; parameters move only in M-step).
- App H reverse implication (Step 3) is rigorous after the pass-6 realizability fix (generator stationarity,
  open-interval sweep, envelope theorem all reproduce; proof is non-circular).
- Fisher mean preconditioner `G_mu^-1 = Sigma` and covariance natural-gradient `delta_Sigma = -2 Sigma sym(grad) Sigma`
  (exact inverse-Fisher operator, residuals `~2e-15`).
- SPD retraction = Pennec affine-invariant exponential map (eq 645, residual `6.3e-14`); Rodrigues SO(3) dexp closed
  form and Taylor coefficients (eq:dexp_so3, residual `1.6e-16`).
- All reported arithmetic: PPL ratios, random-chance factors, head-count percentages, Bonferroni threshold,
  parameter overheads, temperature-dispersion / CV stats, the `within 19%` claim, and the grand-mean 95% CI
  (consistent with a bootstrap/per-passage interval, not the naive head SE) — every printed value reproduces.
- Two-channel "complete free energy" display (supp line 1073) omits the entropy terms but is candid-already (the
  entropy lives in the per-row selection objective eq:J_i_supp and the companion paper; `gamma_ij = 0` throughout).

## Completeness critic — residual items for a future pass

Not defects; equations or steps no pass 1-7 has independently recomputed end-to-end:

- Meta-agent moment matching (eq:meta_agent_beliefs 2313 / supp 950): `Sigma_A = mean(Sigma_i) + Var_A(mu)` as the
  exact second moment of the within-cluster Gaussian mixture (bears on the RG `g1_emer` column and the supp 1021
  table TODO).
- ALiBi / T5 / sliding-window logit reductions (eq:alibi_attention 820, eq:relative_bias_attention 837,
  eq:window_prior 802): substitute the stated priors into eq:mixture_softmax_general and confirm they reduce to the
  published Press et al. / Raffel et al. forms (sign and `1/tau` scaling).
- Frobenius-pullback natural-gradient metric (supp eq:pullback_metric 616) and the right-trivialised `Psi(ad_phi)`
  Gram assembly in App D-pullback.

Candid threads adjudicated: G1 untied-QK realizability (attn ~1183) and the RG y3 deviation (attn ~2349 / supp ~1039)
are adequate as written and were re-confirmed numerically; the central-QK thread (attn ~1186) received the one-line
log-det tightening above.

## Verification

Both files: braces balanced (attention 4024/4024, supplementary 2407/2407); zero LaTeX spacing macros introduced;
zero claudeisms; no new horizontal-rule or en-dash characters added (existing author em-dashes untouched). Every
applied edit confirmed present by fixed-string grep and both stale strings (the old `C` formula, the `'off'` config
literal) confirmed gone. Code-fidelity claims checked against source: config.py:151 (`bool = False`),
retraction.py:126-128 (L-infinity diagonal clamp) and 173-177 (Frobenius full clip).
