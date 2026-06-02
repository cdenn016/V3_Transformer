# SPD Retraction Variants — Log-Euclidean and Bures-Wasserstein

**Design spec for the `register_retraction` seam — 2026-06-01**
**Status: buildable once decided (no open research). Default path unchanged.**
**Author note: judged from executable code (file:line), not docstrings.**

## Motivation

Punch-list item 4 of the buildout roadmap (`docs/2026-06-01-buildout-roadmap.md:111`, Tier B "SPD retraction registry") calls for `register_retraction`/`get_retraction` mirroring `_DECODERS` and `_PRECOND`, naming the SPD retraction as "the one geometry seam the spec names as registry-backed yet it is dispatched by a hardcoded `if belief.sigma.dim()==belief.mu.dim()+1` branch in `e_step.py`." That branch is live at `vfe3/inference/e_step.py:202`, choosing `retract_spd_full` (`vfe3/geometry/retraction.py:48`) versus `retract_spd_diagonal` (`retraction.py:22`) by tensor rank. Both are the affine-invariant exponential map on the SPD cone, the manuscript-canonical retraction pinned at `Manuscripts-Theory/GL(K)_supplementary.tex:640-645` (`Sigma_{k+1} = V Lambda^{1/2} U exp(tau Lambda_B) U^T Lambda^{1/2} V^T`, citing Pennec2006, Bhatia2007, Absil2008). The registry itself is assumed built tonight with the current affine-invariant retraction registered as the default entry `spd_affine`. This document designs the two new variants that register behind it: log-Euclidean (Arsigny et al.) and Bures-Wasserstein, each a write-and-register addition that the E-step selects by config without editing the call site (the spec constraint at clean-room design sec 4.2, "added by writing-and-registering, never by editing call sites").

## The geometry the seam must own (the central design decision)

The current covariance update is a **matched metric pair**, not a bare retraction. The E-step applies a Fisher preconditioner then retracts (`e_step.py:199-209`): `natural_gradient` (`retraction.py:109-143`) converts the Euclidean `grad_sigma` to the natural gradient `nat_sigma = 2 Sigma grad_sigma Sigma` (the affine-invariant Fisher metric, `retraction.py:141`), and `retract_spd_full` steps along the affine-invariant geodesic with that tangent. Both halves are the same affine-invariant geometry. The diagonal branch is the same statement reduced: `nat_sigma = 2 sigma^2 grad_sigma` (`retraction.py:138`).

This forces the answer to the user's explicit question — **should the natural-gradient preconditioner gain a per-retraction hook?** Yes, but not as a second parallel registry that must be kept in sync with the first. If a variant swaps only the retraction (log-Euclidean geodesic) while keeping `2 Sigma G Sigma`, it converts the gradient under one metric and steps along a different metric's geodesic: an incoherent natural-gradient scheme. The clean resolution is to draw the retraction seam so that **each registered mode consumes the Euclidean `grad_sigma` plus the current `sigma` and returns the new `sigma`, owning its own metric conversion internally.** Then:

- `spd_affine` is a thin wrapper that calls the existing `natural_gradient` then `retract_spd_{diagonal,full}` in sequence, bit-identical to today's two-line E-step block, which is exactly what the "affine path unchanged" oracle requires.
- Log-Euclidean and Bures-Wasserstein each bundle their own gradient-to-tangent conversion. There is no mismatched-pair footgun and no separate hook to keep synchronized.

The mean update stays shared and untouched. For a Gaussian the Fisher metric on `mu` is `Sigma^{-1}` regardless of the covariance parameterization, so `nat_mu = Sigma grad_mu` (`retraction.py:137,140`) is correct for all three modes; the `mu` branch of the E-step (`e_step.py:201`) does not move into the seam.

## Interface and architecture

### New registry (built tonight; specified here for completeness)

In `vfe3/geometry/retraction.py`, mirroring `_PRECOND` (`phi_preconditioner.py:33-48`):

```python
_RETRACTIONS: Dict[str, Callable[..., Tuple[torch.Tensor, torch.Tensor]]] = {}

def register_retraction(name: str) -> Callable: ...   # decorator
def get_retraction(name: str) -> Callable: ...        # KeyError-with-available-list, as get_precond
```

Each registered retraction has the **uniform signature** (the gradient-swallowing boundary):

```python
def retract(
    sigma:        torch.Tensor,             # (..., K) diagonal OR (..., K, K) full covariance
    grad_sigma:   torch.Tensor,             # (..., K) or (..., K, K) Euclidean grad wrt sigma
    grad_mu:      torch.Tensor,             # (..., K) Euclidean grad wrt mu
    sigma_q:      torch.Tensor,             # (..., K) or (..., K, K) current cov (== sigma; kept explicit for the Fisher mu metric)

    *,
    step_size:    float = 1.0,
    trust_region: float = 5.0,
    eps:          float = 1e-6,
    sigma_max:    float = 5.0,
) -> Tuple[torch.Tensor, torch.Tensor]:     # (nat_mu, sigma_new): preconditioned mean step + retracted covariance
```

The mode returns the **mean natural gradient** (shared `Sigma grad_mu`, computed once at the top via the existing `natural_gradient` mu branch) alongside the new covariance, so the E-step never recomputes a metric the mode might disagree with. Rank dispatch (diagonal vs full) moves inside each mode, sub-dispatched on `sigma.dim() == grad_mu.dim()` exactly as `natural_gradient` already does at `retraction.py:129`. This keeps `spd_retract_mode` a single config field parallel to `decode_mode`, rather than splitting diagonal/full into separate keys.

### The default entry (registered tonight, pinned by this doc's oracle)

```python
@register_retraction("spd_affine")
def _retract_spd_affine(sigma, grad_sigma, grad_mu, sigma_q, *, step_size, trust_region, eps, sigma_max):
    nat_mu, nat_sigma = natural_gradient(grad_mu, grad_sigma, sigma_q, eps=eps)
    if sigma.dim() == grad_mu.dim() + 1:
        sigma_new = retract_spd_full(sigma, -step_size * nat_sigma, trust_region=trust_region, eps=eps, sigma_max=sigma_max)
    else:
        sigma_new = retract_spd_diagonal(sigma, -step_size * nat_sigma, trust_region=trust_region, eps=eps, sigma_max=sigma_max)
    return nat_mu, sigma_new
```

`retract_spd_diagonal` and `retract_spd_full` remain bare module functions exactly as written; `spd_affine` composes them with `natural_gradient` in the precise order the E-step uses today. This is the verbatim-unchanged path.

### E-step call-site change (built tonight)

The `e_step_iteration` block at `e_step.py:199-209` collapses to a single dispatch:

```python
nat_mu, sigma = get_retraction(spd_retract_mode)(
    belief.sigma, grad_sigma, grad_mu, belief.sigma,
    step_size=e_sigma_lr, trust_region=e_sigma_q_trust, eps=eps, sigma_max=sigma_max,
)
mu = belief.mu - e_mu_lr * nat_mu
```

Note `e_mu_lr` and `e_sigma_lr` enter at the call site (mean step scaled by `e_mu_lr`, covariance step `step_size=e_sigma_lr`), preserving the decoupled learning rates the docstring at `e_step.py:8` describes. The rank branch is gone; `spd_retract_mode` is threaded through the `**kwargs` bag the E-step already forwards (`e_step.py:250,268`).

### New config field

In `vfe3/config.py`, beside the E-step block (`config.py:80-88`):

```python
_VALID_SPD_RETRACT_MODES = ("spd_affine", "log_euclidean", "bures_wasserstein")
...
spd_retract_mode:          str   = "spd_affine"     # SPD covariance retraction geometry
```

validated with `_require(self.spd_retract_mode, _VALID_SPD_RETRACT_MODES, "spd_retract_mode")` in `__post_init__`. Default `spd_affine` keeps the pure, manuscript-pinned path the default, satisfying the CLAUDE.md "theoretically pure path must always exist" constraint and the audit policy (a non-default mode is opt-in).

### New variant files

The two variants register in `vfe3/geometry/retraction.py` (single file, beside the affine entry) or in a sibling `vfe3/geometry/retraction_spd_variants.py` imported for its registration side effect. Recommend keeping them in `retraction.py` since they reuse `retract_spd_full`'s eigh machinery; a 60-line addition does not warrant a new module.

## Math

Notation: `Sigma` is the current SPD covariance, `G = grad_sigma` the Euclidean gradient `partial F / partial Sigma` (symmetrized), `tau` the step size, and the descent direction is `-tau * (tangent)`. The eigendecomposition `Sigma = V Lambda V^T` (via `torch.linalg.eigh`, already used at `retraction.py:79`) is available to all modes.

### 1. Affine-invariant (default, unchanged)

Metric `g_Sigma(A, B) = (1/2) tr(Sigma^{-1} A Sigma^{-1} B)`; natural gradient `nat_Sigma = 2 Sigma G Sigma`. Geodesic retraction (Pennec2006; `GL(K)_supplementary.tex:641-645`):

    Sigma_new = Sigma^{1/2} exp( Sigma^{-1/2} (-tau * nat_Sigma) Sigma^{-1/2} ) Sigma^{1/2}.

SPD-preserving unconditionally for any symmetric tangent, because `exp` of a symmetric matrix is SPD and the congruence by `Sigma^{1/2}` preserves SPD. Diagonal reduction: `sigma_new = sigma exp(-tau * (2 sigma^2 G) / sigma^2) = sigma exp(-2 tau sigma G)`, the `retraction.py:44` form with the Fisher factor absorbed.

### 2. Log-Euclidean (Arsigny–Fillard–Pennec–Ayache)

The log-Euclidean metric (Arsigny et al. 2006, 2007) pulls the flat Euclidean metric back through the matrix logarithm: SPD matrices form a vector space under `Sigma_1 (.) Sigma_2 = exp(log Sigma_1 + log Sigma_2)`, and the metric is `g_Sigma(A,B) = <D log_Sigma(A), D log_Sigma(B)>_F`. The retraction is the closed form

    Sigma_new = expm( logm(Sigma) + Delta_log ),

where `Delta_log` is the tangent expressed in the log-domain (the matrix-log chart). Because `logm(Sigma)` is symmetric and `expm` of a symmetric matrix is SPD, **this is SPD-preserving unconditionally, for any magnitude of `Delta_log`** — no trust region is needed for positivity (the trust region remains available as a stability knob only). `logm`/`expm` are computed in the `Sigma = V Lambda V^T` eigenbasis: `logm(Sigma) = V diag(log lambda_j) V^T`, `expm(M) = U exp(mu_j) U^T` from `M`'s own eigendecomposition — the same two-eigh structure as `retract_spd_full` (`retraction.py:79,92`).

Two readings of `Delta_log`, which the user chooses between (DECISION 3):

- **(2a) Pure retraction, lower fidelity.** Take the symmetric Euclidean `Delta_log = -tau * G` directly as the log-domain tangent. This is a valid SPD retraction (a first-order approximation to the LE geodesic) but is not the LE natural gradient: it steps with the Euclidean gradient interpreted in the wrong chart. Cheapest to build; SPD-exact.
- **(2b) Coherent LE natural gradient.** Convert `partial F / partial Sigma` to the log-domain gradient through the Fréchet derivative of `logm`, then step. The conversion is *not* a sandwich: by the Daleckii–Krein theorem the Fréchet derivative `D logm(Sigma)[H]` in `Sigma`'s eigenbasis is the Hadamard product of `V^T H V` with the divided-difference (Loewner) matrix `L_{ij} = (log lambda_i - log lambda_j)/(lambda_i - lambda_j)` (and `1/lambda_i` on the diagonal) (Higham 2008; Bhatia 1997). This is buildable from the eigh already in hand (the repo already sums a `Psi` divided-difference series for the phi pullback metric at `phi_preconditioner.py:276-281`), so it is engineering, not research.

**Scope finding the user must see:** on the **diagonal** family, `logm` is the elementwise `log` and LE coincides with the affine-invariant retraction up to the Fisher `1/2` constant — both reduce to `sigma_k exp(-c tau sigma_k G_k)`, with the constant foldable into the learning rate. **Log-Euclidean is therefore only a genuinely new variant for full covariance** (`family='gaussian_full'`), where `logm(Sigma) != elementwise log`. Under the default `gaussian_diagonal` family (`config.py:66`) LE adds nothing over affine. This must be stated honestly in the registered docstring and the morning summary.

### 3. Bures-Wasserstein

The Bures-Wasserstein metric is the 2-Wasserstein (optimal-transport) metric restricted to centered Gaussians (Bhatia–Jain–Lim 2019; Malagò–Montrucchio–Pistone 2018): `g_Sigma(A,B) = (1/2) tr(L_A Sigma L_B)` where `L_A` solves the Lyapunov equation `A = L_A Sigma + Sigma L_A`. The exponential-map retraction is (Chewi et al. 2020, the BW gradient-descent congruence form)

    exp_Sigma(V) = (I + L) Sigma (I + L),    where  V = L Sigma + Sigma L  (Lyapunov solve for L given V).

The tangent `V` here is the descent direction `-tau * (BW natural gradient)`; the BW natural gradient of `F` is `nat_Sigma^{BW} = 4 (G Sigma + Sigma G)/...` in the standard form, but the cleanest implementable route takes the Euclidean `G` as the cotangent and solves the single Lyapunov system once. Verify the exp-map form by the geodesic conditions `gamma(0) = Sigma` and `gamma'(0) = L Sigma + Sigma L = V`.

SPD preservation is **conditional**: `(I + L) Sigma (I + L)` is SPD iff `(I + L)` is nonsingular, which fails when `L` has an eigenvalue `<= -1`, i.e. for large steps. The trust region is therefore the **SPD safeguard** for this mode, not merely a stability knob: clamp `L`'s spectrum (or the Frobenius norm of `V` in the whitened metric) so `eigmin(I+L) > floor`. The Lyapunov solve `V = L Sigma + Sigma L` has the eigenbasis closed form `(V tilde)_{ij} = (L tilde)_{ij} (lambda_i + lambda_j)`, so `L tilde_{ij} = (V tilde)_{ij}/(lambda_i + lambda_j)` with `V tilde = V^T V V` ... computed as `L = V ( (V^T V V) ./ (lambda_i + lambda_j) ) V^T` using the same eigh — no separate Sylvester solver needed, and no division-by-zero since `lambda_i + lambda_j >= 2 eps > 0`.

Diagonal reduction: `sigma_new = sigma (1 + X/(2 sigma))^2` with `X` the diagonal tangent — a **distinct quadratic** form, not the affine/LE exponential. BW changes behavior even on the diagonal family, so it is a genuinely new variant for both `gaussian_diagonal` and `gaussian_full`.

**Caveat the user must see:** the BW/OT metric on Gaussians is **not a product metric over (mu, Sigma)** — it couples mean and covariance. A covariance-only BW retraction paired with the Fisher mean update `Sigma grad_mu` is a valid SPD-manifold retraction but a **hybrid**, not the true Gaussian Bures-Wasserstein natural gradient. It must not be sold as "the BW Gaussian geometry." It is also an optimal-transport metric, sitting outside the Fisher-information family the VFE free-energy story is built on (the manuscript's self-coupling and Renyi/KL structure is Fisher-Rao, `GL(K)_supplementary.tex:627-645`); the manuscript explicitly notes Wasserstein lies outside the convex f-divergence class assumed there (`GL(K)_supplementary.tex:1187`). BW is therefore an exploratory variant, not a path toward the canonical theory.

## Numerical concerns

- **Matrix log/exp conditioning (LE).** `logm` amplifies error for small eigenvalues (`log` near 0 has unbounded derivative); floor `lambda_j >= eps` before `log`, exactly as `retract_spd_full` floors at `retraction.py:80`. `expm` of the log-domain sum is bounded by the same `clamp(-50, 50)` overflow guard the affine path uses (`retraction.py:93`). All in the fp32 island `with torch.amp.autocast('cuda', enabled=False)` (`retraction.py:73`), reused verbatim, because both eigh and `expm` are precision-sensitive (the same reason transport.py:53 and retraction.py:37,73,131 are islands).
- **Eigenvalue floor and ceiling.** The post-retraction projection to `[eps, sigma_max^2]` (`retraction.py:99-101`) applies identically to LE and BW outputs; reuse that final block. For LE this only guards against fp drift (the map is SPD-exact); for BW it is part of the safeguard alongside the `(I+L)` nonsingularity clamp.
- **Lyapunov conditioning (BW).** The divisor `lambda_i + lambda_j >= 2 eps` is bounded away from zero by the same eigenvalue floor, so the eigenbasis Lyapunov solve is well-posed without a separate ridge.
- **Fréchet derivative (LE 2b).** The Loewner divided-difference `(log lambda_i - log lambda_j)/(lambda_i - lambda_j)` is removably singular on the diagonal `i = j` (limit `1/lambda_i`); implement with the standard `torch.where(|lambda_i - lambda_j| < tol, 1/lambda_i, ...)` guard. This is the only kernel with any subtlety and it has a known closed form.
- **No `alpha > 1` interaction.** These retractions act on the E-step covariance update only; they are independent of the divergence `alpha_div` and the families seam, so the `alpha > 1` Cholesky hardening item (roadmap Tier D) is orthogonal.

## Phased TDD implementation outline

Each phase names the task, the key test, and the oracle that proves it. Tests are device-agnostic (`VFE3_TEST_DEVICE`), float32, extending `tests/test_retraction.py`.

**Phase 0 — Registry + affine default (built tonight; this doc pins its oracle).**
Task: add `_RETRACTIONS`/`register_retraction`/`get_retraction`, register `spd_affine`, route `e_step.py:199-209` through it.
Key test: `test_spd_affine_bit_identical_to_legacy` — call `get_retraction("spd_affine")` and assert its `(nat_mu, sigma_new)` equals direct `natural_gradient` + `retract_spd_{diagonal,full}` calls at **atol=0** on both diagonal `(B,N,K)` and full `(B,N,K,K)` random inputs. Plus re-run all of `tests/test_retraction.py` verbatim (it imports the bare functions, which are untouched). Oracle: bit-identity to the existing composition is the literal meaning of "the existing affine path unchanged."

**Phase 1 — Log-Euclidean (recommended first variant).**
Task: register `log_euclidean` with reading 2a (pure retraction) as the minimum, optionally 2b (Fréchet natural gradient) behind a sub-flag if the user chooses fidelity (DECISION 3).
Key tests:
- `test_log_euclidean_stays_spd_unconditionally` — random SPD `Sigma`, large-magnitude symmetric tangent (no trust region), assert `eigvalsh(out) > 0`. Oracle: SPD-preservation property; LE's exp-of-symmetric guarantees it for any step, which distinguishes it from BW.
- `test_log_euclidean_full_hand_computed_K2` — a fixed `2x2` SPD `Sigma` and fixed symmetric `Delta_log`, assert `out == expm(logm(Sigma) + Delta_log)` computed by hand (eigen-decompose, log eigenvalues, add, re-exponentiate) to atol=1e-5. Oracle: agreement with a hand-computed small case.
- `test_log_euclidean_diagonal_equals_affine_up_to_lr` — on a diagonal `Sigma`, assert `log_euclidean` matches `spd_affine` after folding the Fisher `1/2` constant into the step size, to atol=1e-5. Oracle: the documented diagonal-coincidence scope finding, turned into a guard so a future change that breaks it is caught.

**Phase 2 — Bures-Wasserstein (second variant).**
Task: register `bures_wasserstein` with the eigenbasis Lyapunov solve and the `(I+L)` nonsingularity safeguard.
Key tests:
- `test_bures_wasserstein_stays_spd_within_trust_region` — random SPD `Sigma`, tangent inside the trust region, assert `eigvalsh(out) > 0`; and a paired `test_..._trust_region_is_the_safeguard` showing that with the trust region disabled and a deliberately large negative tangent the unguarded `(I+L)` becomes singular (documenting *why* the clamp is the SPD safeguard, not optional). Oracle: conditional SPD-preservation property, the discriminating behavior versus LE.
- `test_bures_wasserstein_K1_hand_computed` — `Sigma=[[2.0]]`, fixed tangent `X`, assert `out == sigma*(1 + X/(2 sigma))^2` to atol=1e-5 (the distinct diagonal quadratic). Oracle: hand-computed small case; note this is a *different* expected value than `test_full_retraction_K1_matches_diagonal_formula` (`test_retraction.py:49`), confirming BW is not the affine map even at K=1.
- `test_bures_wasserstein_geodesic_conditions` — finite-difference check that `gamma(t) = exp_Sigma(t V)` satisfies `gamma'(0) = L Sigma + Sigma L = V` to atol=1e-4. Oracle: the exp-map definition itself.

**Phase 3 — E-step integration + config.**
Task: add `spd_retract_mode` to config + validation; add an E-step-level test.
Key test: `test_e_step_runs_each_retract_mode` — one `e_step_iteration` under each of the three modes on the default `gaussian_diagonal` and on `gaussian_full`, asserting finite SPD output and that `spd_affine` reproduces the current E-step output bit-identically (atol=0). Oracle: the registry swap changes nothing on the default and produces valid beliefs on the variants, end-to-end through the call site.

## Risks

- **Hybrid-metric incoherence (BW).** Pairing the BW covariance step with the Fisher mean step is a hybrid; documented above. Risk is interpretive (someone reads it as the true Gaussian-BW gradient), mitigated by the docstring and morning note. No correctness risk to the default path.
- **Diagonal LE is a no-op masquerading as a variant.** If the user runs LE on the default diagonal family expecting new behavior, it silently equals affine. Mitigated by the `test_log_euclidean_diagonal_equals_affine_up_to_lr` guard and an explicit docstring warning; arguably the config validator should warn (not error) when `spd_retract_mode='log_euclidean'` is paired with a diagonal family.
- **Fréchet-derivative correctness (LE 2b).** The Loewner divided-difference is the one kernel that can be got subtly wrong; the diagonal singularity guard and the finite-difference `D logm` oracle pin it. If the user chooses 2a (pure retraction) this risk disappears entirely.
- **fp32-island regression.** Forgetting the `autocast(enabled=False)` wrapper would silently degrade eigh/expm under a future AMP toggle (roadmap Tier C). Mitigated by reusing the existing island block verbatim and a test that asserts SPD output under an autocast context.
- **Trust-region semantics drift (BW).** The trust region means something different for BW (SPD safeguard) than for affine/LE (stability). Risk that a shared default value is wrong for BW; mitigated by the paired safeguard test and a BW-specific default.

## DECISION NEEDED FROM USER

1. **Which variant(s) to build first.** Recommendation: **log-Euclidean first, Bures-Wasserstein second.** LE is unconditionally SPD, reuses the existing eigh, has a clean closed form, and coincides with affine on the diagonal (a clean pin). BW is conditional-SPD (needs the `(I+L)` safeguard), requires a Lyapunov solve, couples mean and covariance, and is an optimal-transport metric outside the Fisher-Rao family the VFE theory is built on (`GL(K)_supplementary.tex:1187`) — it is the more exploratory, higher-surface-area option. Build LE tonight; build BW only if the user wants the OT geometry for comparison.

2. **Per-retraction gradient hook: how.** Recommendation: **yes, fold the gradient-to-tangent conversion into the retraction seam** (each mode consumes Euclidean `grad_sigma` + `sigma`, returns `sigma_new`, and also returns the shared Fisher `nat_mu`). Do **not** build a second parallel preconditioner registry over `natural_gradient` that has to be kept in metric-sync with the retraction registry — that reintroduces the mismatched-pair footgun. The mean update stays the shared `Sigma grad_mu` for all modes.

3. **Log-Euclidean fidelity: pure retraction (2a) or coherent natural gradient (2b).** Recommendation: **build 2a first** (Euclidean symmetric tangent taken directly in the log chart) since it is SPD-exact and trivial, then add 2b (Daleckii–Krein Fréchet derivative of `logm`) behind a sub-flag if the user wants the geometrically coherent LE natural gradient. 2b is buildable from the in-repo eigh and divided-difference machinery (not research) but is the only nontrivial kernel.

4. **`spd_retract_mode` as a single field vs split diagonal/full.** Recommendation: **single field**, sub-dispatching on tensor rank inside each registered mode, keeping config parallel to `decode_mode`.

5. **Should the config validator warn when `log_euclidean` is paired with a diagonal family** (since it equals affine there)? Recommendation: warn, do not error — the pairing is harmless, just redundant.

**Verdict: buildable once decided, not research.** Every closed form (affine, LE retraction, the LE Fréchet derivative, the BW exp map, the eigenbasis Lyapunov solve) is known and citable; the only nontrivial kernel is the LE `logm` Fréchet derivative, which has a standard divided-difference form and reuses the eigh and series machinery already in the repo.

**Citations.** Affine-invariant: Pennec et al. 2006, Bhatia 2007 (*Positive Definite Matrices*), Absil et al. 2008 (*Optimization Algorithms on Matrix Manifolds*) — the manuscript's own citations at `GL(K)_supplementary.tex:641`. Log-Euclidean: Arsigny, Fillard, Pennec, Ayache 2006/2007. Fréchet derivative of `logm`/`expm`: Higham 2008 (*Functions of Matrices*), Bhatia 1997 (*Matrix Analysis*, Loewner/divided-difference). Bures-Wasserstein: Bhatia, Jain, Lim 2019 ("On the Bures-Wasserstein distance"), Malagò, Montrucchio, Pistone 2018 (Wasserstein geometry of Gaussians), Chewi et al. 2020 (BW gradient descent / congruence form). Manuscript pin: `Manuscripts-Theory/GL(K)_supplementary.tex:640-645,627-638,1187`; spec `docs/superpowers/specs/2026-05-29-vfe3-clean-room-design.md` sec 4.2.