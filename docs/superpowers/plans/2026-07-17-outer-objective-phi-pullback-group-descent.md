# Outer-Objective Phi Pullback Group Descent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task by task.

**Goal:** Replace the optional stateful phi M-step with a stateless, certified Frobenius-pullback group descent driven by the existing outer supervised objective, while preserving ordinary AdamW and `omega_direct` behavior and removing the retired heavy-ball/Adam phi implementations.

**Architecture:** A strict M-step-only geometry registry computes a ridge-regularized chart direction and its left-trivialized velocity in float64. `GaugeManifoldAdamW` stages right-product BCH candidates for every active stored phi factor, validates them atomically, and commits them without phi optimizer state. A separate update-policy registry selects ordinary AdamW or this opt-in route. Typed serialized-config migration preserves historical fingerprints and rejects exact resume from removed stateful-phi checkpoints before optimizer topology validation.

**Tech Stack:** Python 3, PyTorch autograd and linear algebra, pytest, JUnit XML, CUDA events on the RTX 5090, and JSON benchmark records.

## Global Constraints

- Treat `docs/plans/2026-07-17-phi-pullback-group-gradient-design.md` as the approved source of truth. Do not widen the objective to the canonical fixed-returned-state VFE M-step in this build.
- After this approved design and plan have been merged, start implementation as a new file-changing task. Fetch `origin`, inspect `origin/main`, and create `codex/phi-pullback-group-descent-20260717` in `C:\tmp\vfe3-phi-pullback-group-descent-20260717` from that fetched ref. Do not reuse the design/plan branch. Preserve the user's live checkout and all pre-existing worktrees.
- Keep the default `m_phi_update_mode="adamw"` topology, one-step values, state layout, fused-CUDA selection, and `phi_weight_decay` behavior unchanged. Keep the `omega_direct` mixed optimizer, `optimizer_extra`, dirty-row masks, and retraction cadence unchanged apart from the class rename.
- Use the term `ridge-regularized Frobenius-pullback direction`; do not label the executable update Fisher, natural, affine-invariant, bi-invariant, gauge-invariant, or canonical VFE.
- Do not modify the existing E-step `_PRECOND` registry, `pullback_metric`, `pullback_metric_per_block`, or `precondition_phi_gradient` behavior. Add a separate strict M-step registry and helper. If shared internals are factored despite this instruction, first pin every touched E-step route with byte-identity tests.
- Keep the complete strict geometry path in an autocast-disabled float64 island. Cast only a fully validated committed phi candidate back to parameter dtype.
- Use the focused K10 `block_glk` regime with two GL(5) blocks. Do not use the discarded K20 run as justification or the primary benchmark.
- Add tests before implementation in each task, run them red, implement the smallest change, and rerun them green. Never add another `-q` to pytest because `pyproject.toml` already supplies it. Derive pass/failure/error counts only from pytest's final summary or JUnit XML.
- Update `docs/2026-07-17-edits.md` throughout; do not create another dated edit record for the same day. Record only verification and timing values read from actual artifacts.
- Use American English in source, comments, test names, documentation, and commit messages.

Before Task 1, run:

```powershell
git fetch origin
git log --oneline -5 origin/main
git worktree add -b codex/phi-pullback-group-descent-20260717 C:\tmp\vfe3-phi-pullback-group-descent-20260717 origin/main
```

Run every remaining command from `C:\tmp\vfe3-phi-pullback-group-descent-20260717`.

---

## Task 1: Add the strict M-step pullback-direction kernel

**Files:**

- Modify: `vfe3/geometry/phi_preconditioner.py:28-38, 316-523`
- Modify: `tests/test_phi_preconditioner.py:1-320`
- Modify only if an existing public oracle is reused: `tests/test_curated_geometry_math_20260709.py:683-730`

### Step 1: Write the failing registry and result-contract tests

- [ ] Before adding any new geometry code, add `test_estep_preconditioner_modes_are_byte_identical_to_legacy_goldens`. Use a deterministic float32 K4/two-GL(2) phi/gradient fixture and a frozen test-local copy of the current arithmetic for `none`, `clip`, `killing`, `killing_per_block`, `pullback`, and `pullback_per_block`; require `torch.equal` for every mode. Run `python -m pytest tests/test_phi_preconditioner.py -k "estep_preconditioner_modes_are_byte_identical" --junitxml=C:\tmp\vfe3-estep-phi-characterization-20260717.xml` against untouched source and require exit code zero before proceeding.
- [ ] Add imports for `PullbackGroupDirectionResult`, `_PHI_GROUP_DIRECTIONS`, and `pullback_group_direction` to `tests/test_phi_preconditioner.py`.
- [ ] Add `test_pullback_group_registry_is_separate_from_estep_registry`. Assert that the new registry contains exactly `{"pullback", "pullback_per_block"}` and is a different object from `_PRECOND`.
- [ ] Add `test_pullback_group_direction_returns_float64_gram_relative_zero_chart_solution`. At `phi=0`, build the generator Gram `B`, assert `v_phi == solve((1 + 1e-6) B, grad)` within `1e-10`, `xi == v_phi`, all result tensors are float64, and the input dtypes are unchanged.
- [ ] Run the focused red test and preserve its JUnit evidence:

```powershell
python -m pytest tests/test_phi_preconditioner.py -k "pullback_group_registry or gram_relative_zero" --junitxml=C:\tmp\vfe3-phi-kernel-red-20260717.xml
```

Expected: collection fails because the new public API does not exist. Confirm a nonzero exit code and `errors` or `failures` greater than zero in the XML; do not report a guessed count.

### Step 2: Define the strict public result and registry without touching the E-step path

- [ ] In `vfe3/geometry/phi_preconditioner.py`, add a frozen result dataclass and a separate registry after `_PRECOND`:

```python
@dataclass(frozen=True)
class PullbackGroupDirectionResult:
    v_phi:                                  torch.Tensor
    xi:                                     torch.Tensor
    min_undamped_generalized_eigenvalue:    torch.Tensor
    undamped_generalized_condition:         torch.Tensor
    damped_generalized_condition:           torch.Tensor
    scaled_solve_residual:                  torch.Tensor
    series_order:                           int


_PHI_GROUP_DIRECTIONS: Dict[str, Callable[..., PullbackGroupDirectionResult]] = {}
```

- [ ] Add `register_phi_group_direction` and `get_phi_group_direction` with duplicate-registration rejection matching the existing registry style.
- [ ] Add the public dispatcher with tensors first and optional values last:

```python
def pullback_group_direction(
    grad_phi:   torch.Tensor,             # (..., n_gen) processed outer-objective covector
    phi:        torch.Tensor,             # (..., n_gen) current chart coordinates
    generators: torch.Tensor,             # (n_gen, K, K) registered basis

    *,
    mode:       str,
    irrep_dims: Optional[List[int]] = None,
) -> PullbackGroupDirectionResult:
    return get_phi_group_direction(mode)(
        grad_phi,
        phi,
        generators,
        irrep_dims=irrep_dims,
    )
```

- [ ] Define module-private fixed semantics, not new config fields: minimum order 40, maximum order 128, order increment 8, tail tolerance `1e-12`, Gram-relative ridge `1e-6`, minimum undamped generalized eigenvalue `1e-8`, maximum damped generalized condition `1e6`, and scaled solve residual `1e-10`.

### Step 3: Write the failing differential-oracle and tail-certificate tests

- [ ] In the test file, implement the float64 augmented-matrix exponential oracle for `D exp_X[H]` using the upper-right block of

```python
torch.linalg.matrix_exp(torch.cat((
    torch.cat((X, H), dim=-1),
    torch.cat((torch.zeros_like(X), X), dim=-1),
), dim=-2))
```

- [ ] Parameterize `test_pullback_group_differentials_match_augmented_exp_oracle` over symmetric, random nonnormal, Jordan-like, and traceless diagonal GL blocks at embedded chart norms 0, 1, 3, and 5. Construct `J`, `J.T @ J`, the right differential, and the left differential from the oracle. Require relative differential and metric error at most `1e-9`.
- [ ] Add `test_pullback_group_adaptive_series_accepts_traceless_gl5_norm_five`; assert a certified order from `range(40, 129, 8)` and finite outputs.
- [ ] Add a fixture whose certificate cannot pass by order 128 and assert a fail-closed exception rather than a warning or truncated result.
- [ ] Run only these tests and confirm they fail before implementing the adaptive series:

```powershell
python -m pytest tests/test_phi_preconditioner.py -k "differentials_match or adaptive_series" --junitxml=C:\tmp\vfe3-phi-series-red-20260717.xml
```

### Step 4: Implement the shared adaptive right/left differential series

- [ ] Add a strict M-step structure-constant builder rather than calling the warning-only legacy `_structure_constants`. Solve bracket coordinates against the generator Gram through its Cholesky factor, reconstruct every bracket, and reject an out-of-span relative residual above the existing `1e-4` closure tolerance. Compute it only on a full GL block of dimension at most 12. For `block_glk`, reuse `_generator_block_index` and slice the same local sub-bases as `pullback_metric_per_block`; never construct a full direct-sum structure-constant tensor.
- [ ] Use the recurrence `term = term @ ad / (k + 2)` rather than integer factorials. Select the smaller of the induced one-norm and infinity-norm per row once, and use that same selected norm for the accumulated-series norm and the tail bound.
- [ ] At candidate orders from `range(40, 129, 8)`, calculate

```python
t_m = alpha.pow(order) / float(math.factorial(order + 1))
r_m = alpha / float(order + 2)
tail = torch.where(r_m < 1.0, t_m / (1.0 - r_m), torch.full_like(r_m, torch.inf))
```

The expression above is the mathematical contract, not the production evaluation. Initialize the scalar term bound at one for series index zero; update `bound_k = bound_k * alpha / float(k + 1)` for each accumulated index `k >= 1`; after accumulating through `k=order-1`, calculate the first omitted bound as `t_m = bound_k * alpha / float(order + 1)`. Do not evaluate a large power or factorial. Accept only when the tail is no larger than `1e-12 * max(1, accumulated_operator_norm)` for every row and for both sign series.
- [ ] Build `Psi_R(ad)` with positive recurrence and `Psi_L(ad)` with alternating signs. The pullback metric uses `D exp_phi[G_a] = Psi_R(ad_phi)[G_a] exp(X)`. The group velocity uses `xi = Psi_L(ad_phi) v_phi`.
- [ ] Keep inputs detached only where optimizer semantics require no higher-order graph; the pure helper itself must remain a deterministic tensor operation and must not mutate inputs.

### Step 5: Write the failing regularity, solve, sign, and per-block tests

- [ ] Add `test_pullback_group_direction_matches_cholesky_oracle_and_residual`. Compare against a float64 oracle and require direction error at most `1e-8` and scaled residual at most `1e-10`.
- [ ] Add `test_pullback_group_direction_rejects_damped_condition_above_limit`; construct a basis/point above `1e6` and assert rejection even if the backward residual is small.
- [ ] Add `test_pullback_group_rejects_gl2_rotation_pi_singularity_before_ridge`. Use the GL(2) rotation generator at angle pi and assert the undamped `dexp` regularity gate rejects before damping can hide the singularity.
- [ ] Use stable analytic threshold fixtures: `X=(pi-delta)J` with `delta=3e-4` must fail the `1e-8` undamped gate; `delta=4.5e-4` must pass as a damping-dominated near-threshold point with damped condition below `1e6`; and `X=diag(5/sqrt(2), -5/sqrt(2))` must pass undamped regularity but fail the damped generalized-condition gate.
- [ ] Add `test_pullback_group_left_trivialization_satisfies_u_xi_equals_dexp_v`. Require relative Frobenius residual at most `1e-9`, and explicitly show the opposite-sign or opposite-trivialization control fails.
- [ ] Add `test_pullback_group_per_block_matches_two_independent_gl5_solves_without_full_structure_constants`. Monkeypatch the full K10 `_structure_constants` call to raise, then verify two local GL(5) calls succeed and match independent local oracles.
- [ ] Add `test_pullback_group_rejects_nonclosed_generator_span`. The strict M-step helper must measure the bracket reconstruction residual and reject, rather than inheriting the E-step helper's warning-and-projection behavior.

### Step 6: Implement the generalized regularity gates and Cholesky solve

- [ ] Compute the generator Gram `B = einsum("aij,bij->ab", G64, G64)` in float64.
- [ ] Use a Cholesky factor of `B` to whiten `sym(metric)` and calculate undamped generalized eigenvalues. Reject a minimum below `1e-8` before adding the ridge.
- [ ] Form `A = sym(metric) + 1e-6 * B`, calculate its generalized condition in the same whitened coordinates, and reject values above `1e6`.
- [ ] Solve only through `torch.linalg.cholesky_ex` plus `torch.cholesky_solve`; never call `inverse` or use an identity-coordinate ridge.
- [ ] Calculate and return the scaled residual

```python
residual = torch.linalg.vector_norm(A @ v - grad, dim=-1)
scale = (
    torch.linalg.matrix_norm(A, ord=2, dim=(-2, -1))
    * torch.linalg.vector_norm(v, dim=-1)
    + torch.linalg.vector_norm(grad, dim=-1)
)
scaled_residual = residual / scale.clamp_min(torch.finfo(torch.float64).tiny)
```

Reject nonfinite values, Cholesky failures, residuals above `1e-10`, and every failed certificate. For per-block results, aggregate the minimum and maximum generalized eigenvalues across every local block, calculate each global direct-sum condition as global maximum divided by global minimum, and return the maximum residual and maximum series order. Taking only the maximum local condition can underestimate the full direct-sum condition and is not acceptable.

### Step 7: Verify the new helper and the untouched E-step route

- [ ] Run the complete geometry-focused test set:

```powershell
python -m pytest tests/test_phi_preconditioner.py tests/test_curated_geometry_math_20260709.py tests/test_fix_gauge_audit.py --junitxml=C:\tmp\vfe3-phi-kernel-green-20260717.xml
```

Expected: exit code zero, with `failures="0"` and `errors="0"` in the XML.
- [ ] Inspect `git diff -- vfe3/geometry/phi_preconditioner.py` and confirm the legacy E-step functions from `_PRECOND` through `precondition_phi_gradient` are behaviorally unchanged. If any were factored, add a seeded literal/reference regression for every touched mode and require `torch.equal`.
- [ ] Commit this task:

```powershell
git add vfe3/geometry/phi_preconditioner.py tests/test_phi_preconditioner.py tests/test_curated_geometry_math_20260709.py
git commit -m "feat: add certified phi pullback direction"
```

---

## Task 2: Replace the stateful phi optimizer with the policy-selected group route

**Files:**

- Modify: `vfe3/gauge_optim.py:1-732`
- Modify: `vfe3/geometry/groups.py:33-47, 154-204, 216-283`
- Modify: `vfe3/config.py:9-33, 147-166, 582-631, 846-910, 1001-1443, 1823-1843, 1960-1975`
- Modify: `vfe3/train.py:164-405, 605-769, 1458-1467, 1736-1759`
- Modify: `vfe3/model/model.py:1611-1744`
- Modify: `train_vfe3.py:150-165`
- Modify: `ablation.py:267-291, 618-638, 921-949`
- Modify: `scaling.py:180-195`
- Modify: `tests/test_gauge_optim.py:1-end`
- Modify: `tests/test_config.py:1-end`
- Modify: `tests/test_fix_config_audit.py:1-end`
- Modify: `tests/test_b5_finite_config_controls_20260716.py:115-145`
- Modify: `tests/test_2026_07_15_driver_reliability_remediation.py:1510-1580`
- Modify class imports/references now, with behavioral rewrites left to their later tasks: `tests/test_checkpoint_resume.py`, `tests/test_exp8_buildout.py`, `tests/test_fp16_gradscaler.py`, `tests/test_hyperprior.py`, `tests/test_omega_direct.py`, `tests/test_2026_07_15_performance_remediation.py`, and `tests/test_final_audit_integrity_20260716.py`.

### Step 1: Write failing policy, validation, and direct-optimizer tests

- [ ] Replace the old public-class import in `tests/test_gauge_optim.py` with `GaugeManifoldAdamW`, `PhiUpdatePolicy`, `_PHI_UPDATE_POLICIES`, and `get_phi_update_policy`.
- [ ] Add tests that the registry contains exactly `adamw` and `pullback_group`, that the AdamW policy contributes an empty metadata mapping, and that the pullback policy contributes `{"pullback_group": True, "weight_decay": 0.0}`, requires the manifold optimizer, and requires strict pullback geometry.
- [ ] Add a true byte-level default-AdamW golden. On CPU, call `torch.manual_seed(0)` and construct `VFE3Config(vocab_size=8, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=1, n_e_steps=1, pos_phi="none")`. Use `tokens=torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=torch.long)`, `targets=torch.tensor([[1, 2, 3, 4], [2, 1, 0, 7]], dtype=torch.long)`, `optimizer=build_optimizer(model, cfg)`, `scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)`, and `train_step(model, optimizer, scheduler, tokens, targets, grad_clip=1.0)`. Define each digest as `sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()`. Pin optimizer type `torch.optim.AdamW`, group roles `['mu', 'sigma', 'phi', 'mu']`, populated state IDs `[0, 1, 3, 4]` with exactly `{"step", "exp_avg", "exp_avg_sq"}`, loss `2.0770487785339355`, and post-step SHA256 values: `prior_bank.mu_embed` `ffc101ec6c9b0fc34e1089dda4b5b28cafa6e2d53678c257ced04723e6e2a66a`, `prior_bank.sigma_log_embed` `b632d4a844923ebb4e8af9e1158ae9eb20ad37a64677bc53186235915d2790fd`, `prior_bank.phi_embed` `e31bfece5ed86861d7d32f3e16214c17ddcb780dfda1116c3cffbf0e27a674b9`, `prior_bank.decode_log_scale` `df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119`, and `prior_bank.output_proj_weight` `1bc018feb7e7c10fcd644d14bbf24a3ac2fe50073b9a355e2bcca69b907618de`. Assert no phi optimizer slot is populated.
- [ ] In `tests/test_config.py`, assert the default is `m_phi_update_mode="adamw"`, the retired dataclass fields are absent, and `m_phi_group_trust_radius == 0.1`.
- [ ] Parameterize fail-closed configuration tests for: unknown mode; nonpositive/nonfinite trust radius; `omega_direct`; groups other than built-in `glk`/untied `block_glk`; GL(K) dimension above 12; block dimension above 12; cross-coupled `block_glk`; wrong full/per-block preconditioner pairing; `pos_phi_project_slk=True`; missing transport chart bound; transport bound at or below the factor chart radius; and transport bound at or above the hard transport clamp 20.
- [ ] Register a temporary custom override under a phase-one group name without the new pullback capability metadata and assert it is rejected. Restore the original registry entry in `finally` so test order cannot leak the override. This pins “built-in registrations only” rather than trusting a string key that a custom builder can shadow.
- [ ] Add valid controls for K10 `glk` plus `pullback`, and K10 two-GL(5) `block_glk` plus `pullback_per_block`, each with `transport_chart_max_norm=6.0` and the default factor radius 5.0.
- [ ] Add direct optimizer tests for active-row-only movement, zero movement of inactive rows, no phi state after the step, right-product order, trust scaling, mandatory BCH residual backtracking, chart-bound rejection, and atomic rejection across two parameter groups.
- [ ] Pin the noncommuting GL(2) backtracking fixtures exactly. At chart norm 3, use `phi=[0.4461370124222645, -0.37866876575372294, 1.1203006449654396, -2.7207532407183694]` and `delta=[0.0075751275839514645, -0.002453059541722277, -0.003516904099976695, -0.002529014357447788]`; the first accepted right-product residual occurs after six halvings and is approximately `5.48315e-7`, while the reversed-product residual remains about `1.083e-4`.
- [ ] At chart norm 5, use `phi=[0.19176642751290957, -1.411275433700944, 3.477796870171524, -3.297947273280199]` and `delta=[0.008133453045635446, 0.0015903479064029997, -0.0025065446684178387, -0.001781968201968022]`; the first accepted right-product residual occurs after seven halvings and is approximately `9.92428e-7`, while the reversed-product residual remains about `8.28e-5`.
- [ ] Run the focused red set:

```powershell
python -m pytest tests/test_gauge_optim.py tests/test_config.py tests/test_fix_config_audit.py tests/test_b5_finite_config_controls_20260716.py -k "phi or pullback or gauge" --junitxml=C:\tmp\vfe3-phi-runtime-red-20260717.xml
```

Expected: nonzero exit because the new class, fields, and policy registry are absent.

### Step 2: Add the update-policy registry and rename the mixed optimizer

- [ ] In `vfe3/gauge_optim.py`, add the frozen metadata type and duplicate-safe registry:

```python
@dataclass(frozen=True)
class PhiUpdatePolicy:
    optimizer_group_metadata: Mapping[str, object]
    requires_manifold_optimizer: bool
    requires_pullback_geometry: bool = False


_PHI_UPDATE_POLICIES: Dict[str, PhiUpdatePolicy] = {}
```

- [ ] Register `adamw` with an empty metadata mapping and both requirements false. Register `pullback_group` with `{"pullback_group": True, "weight_decay": 0.0}`, `requires_manifold_optimizer=True`, and `requires_pullback_geometry=True`. Return immutable/copy-safe metadata so a caller cannot mutate the registry.
- [ ] Extend `GaugeGroup` and `register_group` with optional pullback capability metadata. Set it to `"pullback"` only on the built-in full elementary `glk` registration and `"pullback_per_block"` only on the built-in untied elementary `block_glk` registration; all other and custom registrations default to `None`. Copy the capability onto both the builder and built `GaugeGroup`, matching existing registry metadata propagation.
- [ ] Rename `GaugeNaturalGradAdamW` to `GaugeManifoldAdamW` everywhere. Do not leave a compatibility alias; tests and runtime imports must use the new name.
- [ ] In this same task, update every executable and test import/reference returned by `rg -l "GaugeNaturalGradAdamW" vfe3 tests train_vfe3.py ablation.py scaling.py`. Do not defer any old-class import to Task 5, because removing the class and alias must not leave the branch unimportable.
- [ ] Update every direct constructor call at the same time: pass the `GaugeGroup` object and the new required phi bounds/mode, remove generator/irrep positional arguments and retired moment arguments, and keep each omega-only test's existing math/state assertions unchanged.
- [ ] Retain the custom `__setstate__` structure that calls base `Optimizer.__setstate__` and performs Adam step migration only when a `step` slot exists. Remove only the gauge-moment wording and branches; `omega_dirty` has no Adam `step`, so inherited Adam migration remains unsafe.
- [ ] Delete constructor parameters/attributes for `gauge_momentum` and `gauge_update_rule`. Use this exact replacement constructor surface, retaining the mandatory optimizer-params positional exception and the existing `**kwargs` pass-through:

```python
def __init__(
    self,
    params,
    group: GaugeGroup,

    *,
    phi_group_trust_radius: float,
    phi_chart_max_norm:     float,
    phi_bch_residual_max:   float,
    phi_precond_mode:       str,
    omega_retract_mode:     str           = "lie_exp",
    omega_reorth_every:     int           = 0,
    **kwargs,
) -> None:
```

Store `group`; derive generators, irrep dimensions, skew-symmetry, and group name from it; validate all required positive finite bounds; and keep omega state intact.

### Step 3: Replace additive/moment phi mutation with atomic candidate staging

- [ ] Delete all `precondition_phi_gradient`, `gauge_mom`, `gauge_m`, `gauge_v`, `gauge_step`, heavy-ball, and coordinatewise-Adam phi code.
- [ ] Add this frozen result and pure production staging seam in `vfe3/gauge_optim.py`:

```python
@dataclass(frozen=True)
class PullbackGroupCandidate:
    candidate_phi:            torch.Tensor
    trust_scale:              torch.Tensor
    backtracking_reductions:  torch.Tensor
    candidate_chart_norm:     torch.Tensor
    group_product_residual:   torch.Tensor
    direction:                PullbackGroupDirectionResult


@torch.no_grad()
def stage_pullback_group_candidate(
    grad_phi:   torch.Tensor,             # (active, n_gen) processed covector
    phi:        torch.Tensor,             # (active, n_gen) current chart
    group:      GaugeGroup,

    *,
    learning_rate:          float,
    trust_radius:           float,
    chart_max_norm:         float,
    bch_residual_max:       float,
    phi_precond_mode:       str,
    max_backtracks:         int = 10,
) -> PullbackGroupCandidate:
```

The helper owns the complete float64 direction, trust scaling, BCH4, rowwise residual backtracking, chart bound, and validation path. It returns float64 candidates/certificates without mutating the parameter or consuming its gradient. Both `GaugeManifoldAdamW.step()` and the benchmark in Task 6 must call this same function; do not duplicate a benchmark-only approximation.
- [ ] Before mutating any phi table, flatten each `pullback_group=True` parameter to rows, identify rows whose current processed gradient has nonzero magnitude, and call `stage_pullback_group_candidate` only for those rows.
- [ ] Scale the right factor after multiplying by the group learning rate:

```python
right_factor = -learning_rate * result.xi
right_norm = embedded_phi_frobenius_norm(right_factor, group, warn_fallback=False)
trust_scale = (trust_radius / right_norm.clamp_min(tiny)).clamp(max=1.0)
delta = trust_scale.unsqueeze(-1) * right_factor
```

- [ ] Form the stored chart candidate with `compose_bch(phi64, delta64, generators64, order=4, block_dims=irrep_dims, compact_blocks=(group.name == "block_glk"))`. Do not reverse operands.
- [ ] Compute the rowwise float64 exact-product residual between `exp(candidate.G)` and `exp(phi.G) @ exp(delta.G)`. Require at most `min(1e-6, cfg.bch_residual_max)` when the optional bound is set, otherwise `1e-6`.
- [ ] For `block_glk`, evaluate the exact product and residual on the two local GL blocks and aggregate their squared Frobenius errors/reference norms before taking the ratio. Do not densify every active row to K10 matrices when the same exact block-product certificate is available locally.
- [ ] For residual failures only, halve that row's factor scale and retry up to ten reductions. Reject the complete phi manifold step if any row still fails. Treat nonfiniteness, geometry rejection, Cholesky failure, chart-radius violation, or invalid candidate as an immediate complete-step rejection.
- [ ] Check the embedded candidate chart norm against `phi_mstep_max_matrix_norm` when set, otherwise 5.0. Do not invoke the generic post-step radial projector on this route.
- [ ] Stage candidates across token phi, learned positional phi, independent model token phi, and independent model positional phi groups. Validate every plan before committing any. Cast each accepted candidate once to parameter dtype, commit active rows, then set the consumed parameter gradient to `None` so base AdamW cannot double-step it.
- [ ] Never access `self.state[p]` for a pullback-group parameter. Preserve omega staging and state byte-for-byte except renamed messages.

### Step 4: Replace the public config fields and add fail-closed validation

- [ ] In `VFE3Config`, replace the three retired fields with:

```python
m_phi_update_mode:         str   = "adamw"
m_phi_group_trust_radius:  float = 0.1
```

Keep `phi_mstep_max_matrix_norm` and `phi_weight_decay` in place, with comments that distinguish the AdamW projector from the pullback-group in-optimizer bound.
- [ ] Validate `m_phi_update_mode` against the live policy registry and require a finite positive trust radius.
- [ ] When the selected policy declares `requires_pullback_geometry`, require `gauge_parameterization="phi"`; a group builder whose registered pullback capability matches `phi_precond_mode`; built-in `glk` with `embed_dim <= 12`, or built-in untied/non-cross-coupled `block_glk` with `d_head <= 12`; `pos_phi_project_slk=False`; and a certified bracket-closed basis. Do not key this validation on a hard-coded update-mode literal.
- [ ] Define `factor_radius = phi_mstep_max_matrix_norm if not None else 5.0`. Require an explicit `transport_chart_max_norm` satisfying `factor_radius < transport_chart_max_norm < TRANSPORT_CLAMP_MAX_NORM`, importing the hard constant from `vfe3.geometry.transport` rather than duplicating 20.
- [ ] Keep the AdamW mode free of every new geometric compatibility constraint except the globally positive trust field, so current configurations retain their behavior.

### Step 5: Route optimizer groups through policy metadata

- [ ] In `build_optimizer`, call `get_phi_update_policy(cfg.m_phi_update_mode)` once. Apply its metadata only to stored phi-factor groups: `phi_embed`, `pos_phi_free`, `s_phi_embed`, and `s_pos_phi_free`. Do not tag connection parameters or other parameters whose role happens to be `phi`.
- [ ] For AdamW, preserve the exact existing group dictionaries, `phi_weight_decay`, and plain/fused optimizer selection. For pullback group, set weight decay to zero via registry metadata and instantiate `GaugeManifoldAdamW` with `model.group` and the validated geometry controls. Instantiate the same mixed class for `omega_direct` with its existing metadata.
- [ ] Gate the post-step `project_phi_parameter_rows_` call to `m_phi_update_mode == "adamw"`. Preserve its current behavior on that route.
- [ ] Keep the established order: GradScaler unscale, finite-gradient check, clipping, then `scaler.step(optimizer)`. The geometry helper must receive `p.grad` exactly as processed at that seam.

### Step 6: Add the canonical objective TODO at the exact completed-loss seam

- [ ] Insert this exact comment immediately before `return logits, loss, ce.detach()` in `vfe3/model/model.py`, after the CG and every other optional outer addition:

```python
# TODO(canonical-vfe-phi-mstep): `loss` is the outer supervised objective (CE plus
# enabled outer regularizers), so the pullback/group phi step consumes its covector.
# Add a separately selected fixed-returned-state VFE frame objective, declare whether
# beta/gamma are frozen or envelope-eliminated, and keep the optimizer objective-agnostic.
```

- [ ] Do not add objective assembly, belief-state, beta, gamma, or CE knowledge to the optimizer.

### Step 7: Update click-run controls and the m-phi ablation

- [ ] Replace the retired controls in `train_vfe3.py`, `ablation.py`, and `scaling.py` with `m_phi_update_mode="adamw"` and `m_phi_group_trust_radius=0.1` while preserving every unrelated current click-run value.
- [ ] In the m-phi ablation, compare `adamw` with `pullback_group`; delete the old killing and stateful-natural-gradient arms because they do not implement the approved route. Require `pullback_per_block`, K10/two GL(5) blocks, `e_phi_lr=0.0`, and a valid explicit transport bound for the new arm.
- [ ] Update driver reliability tests to pin the new keys and prove no retired override survives.

### Step 8: Verify runtime routing and commit

- [ ] Run the focused runtime set:

```powershell
python -m pytest tests/test_gauge_optim.py tests/test_config.py tests/test_fix_config_audit.py tests/test_b5_finite_config_controls_20260716.py tests/test_2026_07_15_driver_reliability_remediation.py --junitxml=C:\tmp\vfe3-phi-runtime-green-20260717.xml
```

Expected: exit code zero, XML failures and errors both zero.
- [ ] Confirm the default builds plain `torch.optim.AdamW`, the selected route builds `GaugeManifoldAdamW`, and `omega_direct` still builds the mixed optimizer.
- [ ] Commit this coherent runtime replacement:

```powershell
git add vfe3/gauge_optim.py vfe3/geometry/groups.py vfe3/config.py vfe3/train.py vfe3/model/model.py train_vfe3.py ablation.py scaling.py tests/test_gauge_optim.py tests/test_config.py tests/test_fix_config_audit.py tests/test_b5_finite_config_controls_20260716.py tests/test_2026_07_15_driver_reliability_remediation.py tests/test_checkpoint_resume.py tests/test_exp8_buildout.py tests/test_fp16_gradscaler.py tests/test_hyperprior.py tests/test_omega_direct.py tests/test_2026_07_15_performance_remediation.py tests/test_final_audit_integrity_20260716.py
git commit -m "feat: replace stateful phi updates with group descent"
```

---

## Task 3: Add typed serialized-config migration and resume boundaries

**Files:**

- Modify: `vfe3/config.py:9-13, 2806-2842`
- Modify: `vfe3/run_artifacts.py:758-848, 954-1092, 1667-1710, 1838-1903`
- Modify: `generate_efe.py:175-200, 291-305`
- Modify: `vfe3/viz/run_loading.py:1-49`
- Modify: `tests/test_config.py:130-end`
- Modify: `tests/test_checkpoint_resume.py:500-560, 1317-1390`
- Modify: `tests/test_run_artifacts.py:300-370`
- Modify: `tests/test_final_audit_integrity_20260716.py:440-470`
- Modify: `tests/test_2026_07_15_cache_serialization_remediation.py:80-115`

### Step 1: Write failing migration-matrix tests

- [ ] Add tests for a frozen `SerializedConfigMigration` result carrying `config`, a copied `raw_config`, `consumed_retired_keys: frozenset[str]`, and `legacy_stateful_phi_optimizer: bool`.
- [ ] Test legacy mappings independently: old `False` plus phi storage maps to `adamw` without provenance; old `False` plus `omega_direct` maps to `adamw` without losing omega controls; old `True` maps to `adamw` with provenance true; retired momentum/update-rule keys are consumed and absent from the effective dataclass.
- [ ] Test old/new coexistence: historical `False` is compatible only with explicit new `adamw`; historical `False` plus `pullback_group` and historical `True` plus any explicit new mode raise a conflict. If the old boolean is absent, validated retired momentum/rule keys are inert provenance and may be consumed with either new mode.
- [ ] Test strict mode admits only current fields plus the exact three retired keys and rejects a genuinely unknown/newer field. Test permissive `config_from_serialized` retains its warning behavior for unrelated unknown fields and returns `.config` only.
- [ ] Run the red migration subset:

```powershell
python -m pytest tests/test_config.py tests/test_checkpoint_resume.py tests/test_run_artifacts.py -k "serialized or migration or legacy_stateful_phi or selection_config" --junitxml=C:\tmp\vfe3-phi-migration-red-20260717.xml
```

### Step 2: Implement the typed lower-level migration API

- [ ] Add:

```python
_RETIRED_PHI_CONFIG_FIELDS = frozenset({
    "m_phi_natural_grad",
    "m_gauge_momentum",
    "m_gauge_update_rule",
})


@dataclass(frozen=True)
class SerializedConfigMigration:
    config:                         VFE3Config
    raw_config:                     Mapping[str, Any]
    consumed_retired_keys:          frozenset[str]
    legacy_stateful_phi_optimizer:  bool
```

- [ ] Implement `migrate_serialized_config(payload, *, source, strict_unknown=False)`. Copy and validate the raw mapping before mutation, normalize the legacy boolean with the same exact bool/string rules as current serialized booleans, require a finite numeric retired momentum and a retired rule in `{"heavy_ball", "adam"}` when those keys are present, enforce the coexistence matrix, set effective mode to `adamw` when no new field exists, remove retired keys, normalize all current booleans and `policy_score_terms`, and construct `VFE3Config`.
- [ ] Make `config_from_serialized` a convenience wrapper returning `migrate_serialized_config(payload, source=source).config` so visualization and weight-only consumers remain source-compatible.

### Step 3: Preserve raw fingerprint order in artifact consumers

- [ ] Keep `_validate_best_model_mapping`'s raw `config_fingerprint` comparison before semantic migration. Change `_selection_semantic_config` to call strict typed migration so only the retired fields bypass the current-schema guard.
- [ ] In `generate_efe._bound_config` and `vfe3/viz/run_loading`, verify any stored fingerprint against the raw historical mapping first, then call migration. Do not recompute an old fingerprint from the migrated mapping.
- [ ] Add corruption tests proving a bad raw fingerprint still fails even when the historical config would migrate successfully.

### Step 4: Move resume migration ahead of optimizer topology validation

- [ ] Change `_preflight_resume_config` to return the typed migration plus deterministic drift against `asdict(active_cfg)`. Compute drift from the effective migrated config, excluding only the established resume policy fields.
- [ ] In `load_checkpoint`, migrate/preflight `ckpt["config"]` before `_validate_optimizer_state`. If an optimizer is supplied and `legacy_stateful_phi_optimizer=True`, raise a clear error containing both `stateful phi optimizer is incompatible` and `restart from model weights/current config` before any model, optimizer, scaler, RNG, or cursor mutation.
- [ ] Allow weight-only load with `optimizer=None` to proceed. Preserve exact resume for migrated legacy `False` plus phi AdamW and legacy `False` plus `omega_direct` with `optimizer_extra`/dirty masks.
- [ ] Delete `_validate_optimizer_state` support for `gauge_mom`, `gauge_m`, `gauge_v`, and `gauge_step`. A nonempty pullback-group parameter slot is unsupported; only ordinary AdamW and omega-dirty state remain valid.

### Step 5: Add exact checkpoint tests

- [ ] Replace the old heavy-ball resume test at `tests/test_checkpoint_resume.py:516-544` with: new pullback-group step-exact resume with no phi slots; legacy `False` phi AdamW exact resume; legacy `False` omega-direct exact resume with omega extras; legacy `True` exact-resume rejection before mutation; and legacy `True` weight-only success.
- [ ] Assert that a newly saved pullback-group checkpoint has no `gauge_*` keys and no populated state entry for any phi-manifold parameter.
- [ ] Run the focused green set:

```powershell
python -m pytest tests/test_config.py tests/test_checkpoint_resume.py tests/test_run_artifacts.py tests/test_final_audit_integrity_20260716.py tests/test_2026_07_15_cache_serialization_remediation.py --junitxml=C:\tmp\vfe3-phi-migration-green-20260717.xml
```

Expected: exit code zero, XML failures and errors both zero.
- [ ] Commit:

```powershell
git add vfe3/config.py vfe3/run_artifacts.py generate_efe.py vfe3/viz/run_loading.py tests/test_config.py tests/test_checkpoint_resume.py tests/test_run_artifacts.py tests/test_final_audit_integrity_20260716.py tests/test_2026_07_15_cache_serialization_remediation.py
git commit -m "feat: migrate retired phi optimizer configs"
```

---

## Task 4: Prove the outer-objective covector and training-pipeline composition

**Files:**

- Create: `tests/test_phi_outer_objective_gradient.py`
- Modify: `tests/test_fp16_gradscaler.py:75-105`
- Modify: `tests/test_exp8_buildout.py:1-130`
- Modify: `tests/test_phi_weight_decay.py:1-80`
- Modify: `tests/test_hyperprior.py:270-300`
- Modify: `tests/test_omega_tilde_model_frame.py:375-410`
- Modify: `tests/test_gauge_optim.py:1-end`
- Modify: `tests/test_2026_07_15_geometry_remediation.py:150-260`

### Step 1: Add the raw outer-CE finite-difference test

- [ ] Build a deterministic tiny phi-coordinate model with stochastic behavior, autocast, optional outer regularizers, and gradient clipping disabled. Use fixed token/target tensors and a normalized seeded direction supported only on active `phi_embed` rows.
- [ ] Compute the returned scalar loss and its raw autograd phi covector. Evaluate central differences at `h in {1e-2, 3e-3, 1e-3}`, restoring the parameter exactly after every perturbation.
- [ ] Require a stable error plateau and at most `5e-3` relative disagreement between the directional derivative of the returned CE scalar and the autograd contraction. Assert that `loss == ce` in this all-regularizers-off fixture.
- [ ] Add a second fixture with one enabled outer regularizer and assert the consumed covector is the derivative of the completed returned `loss`, not a separately reconstructed CE.

### Step 2: Prove unscale and clipping precede geometry unchanged

- [ ] Wrap `pullback_group_direction` with a test spy that clones its incoming `grad_phi`, delegates to the real helper, and does not change outputs.
- [ ] Run one training step with deterministic gradients and clipping enabled. Independently compute the expected post-unscale/post-clip phi covector from an identical model state, then assert the spy received it unchanged. Prove no second normalization occurs inside the optimizer.
- [ ] Add `test_pullback_group_gradient_accumulation_uses_accumulated_covector`. Compare a full-batch update with `grad_accum_steps=2` on equal-token microbatches, spy on the production staging seam, and require the same accumulated processed covector and accepted candidate to roundoff.
- [ ] Update `tests/test_fp16_gradscaler.py` to select `pullback_group`, a valid K10/two-GL(5) config, and an explicit transport bound. Test finite autocast/GradScaler composition, overflow-skipped mutation, and one cast back to float32 on commit.
- [ ] Add/retain the disabled-scaler nonfinite-gradient test proving a rejected step leaves all phi tables unchanged.
- [ ] Add an independent nonfinite-scalar-loss/skipped-step test. Snapshot token, positional, model-token, and model-positional phi tables and assert every byte is unchanged, the staging helper is not called, and the scheduler/accepted-update clock does not advance.

### Step 3: Cover every stored factor and projector/transport boundary

- [ ] Update model-frame tests so token phi, learned positional phi, independent model token phi, and independent model positional phi all enter manifold-tagged groups and move only on nonzero current gradients.
- [ ] In `tests/test_phi_weight_decay.py`, assert pullback-group factor groups have zero weight decay and no phi slots, while default AdamW still uses `phi_weight_decay` exactly as before.
- [ ] Monkeypatch `project_phi_parameter_rows_` to raise and prove a successful pullback-group train step does not call it. Keep the existing AdamW projector test green.
- [ ] Add a composed token-position transport test: an in-bound effective chart remains at scale one, while an out-of-bound composed chart raises at `transport_chart_max_norm` before the detached hard clamp. Do not accept a scale below one as success.
- [ ] Add float32 committed right-product residual checks at current chart norms 3 and 5, requiring at most `5e-6` after the float64 staging path met `1e-6`.

### Step 4: Run the training-integration set and commit

```powershell
python -m pytest tests/test_phi_outer_objective_gradient.py tests/test_fp16_gradscaler.py tests/test_exp8_buildout.py tests/test_phi_weight_decay.py tests/test_hyperprior.py tests/test_omega_tilde_model_frame.py tests/test_gauge_optim.py tests/test_2026_07_15_geometry_remediation.py --junitxml=C:\tmp\vfe3-phi-integration-green-20260717.xml
```

Expected: exit code zero, XML failures and errors both zero.

- [ ] Commit:

```powershell
git add tests/test_phi_outer_objective_gradient.py tests/test_fp16_gradscaler.py tests/test_exp8_buildout.py tests/test_phi_weight_decay.py tests/test_hyperprior.py tests/test_omega_tilde_model_frame.py tests/test_gauge_optim.py tests/test_2026_07_15_geometry_remediation.py
git commit -m "test: verify outer-objective phi group descent"
```

---

## Task 5: Replace misleading diagnostics and remove every retired implementation reference

**Files:**

- Modify: `vfe3/gauge_optim.py:350-365, 512-523, 717-732`
- Modify: `vfe3/train.py:1458-1467, 1736-1759`
- Modify: `vfe3/run_artifacts.py:345-353, 3706-3714`
- Modify: `vfe3/viz/figures.py:1475-1503, 4186-4188`
- Modify: `tests/test_exp8_buildout.py:1-end`
- Modify: `tests/test_omega_direct.py:1-end`
- Modify: `tests/test_2026_07_15_performance_remediation.py:1-end`
- Modify: `tests/test_final_audit_integrity_20260716.py:1-end`
- Modify if the removal search confirms a live reference: `tests/test_b5_finite_config_controls_20260716.py`, `tests/test_checkpoint_resume.py`, `tests/test_config.py`, `tests/test_fix_config_audit.py`, `tests/test_fp16_gradscaler.py`, `tests/test_gauge_optim.py`, `tests/test_hyperprior.py`, `tests/test_omega_tilde_model_frame.py`, and `tests/test_phi_weight_decay.py`.

### Step 1: Add diagnostic-contract tests

- [ ] On logging steps only, assert the optimizer exposes `phi_ridge_direction_cosine_mean`, `phi_pullback_damped_gen_cond_median`, `phi_pullback_damped_gen_cond_max`, `phi_group_trust_scale_mean`, `phi_group_trust_scale_min`, `phi_group_active_rows`, and `phi_group_chart_norm_max`.
- [ ] Assert silent steps perform no tensor-to-host conversions for these diagnostics. Reuse the geometry result's certificate tensors; do not recompute the legacy fixed-series/identity-damped metric.
- [ ] Assert the first metrics row remains rectangular and visualization/artifact labels use `pullback`/`ridge direction`, not `natural gradient`.

### Step 2: Implement diagnostic collection and labels

- [ ] Accumulate the new result fields and trust/chart data during staging only when `_collect_gauge_diag=True`. Host-reduce only inside that gate.
- [ ] Update train CSV extraction, artifact chart-route metadata, and visualization labels. Keep any historical CSV compatibility in a loader/migration layer; do not emit old aliases from the new runtime.
- [ ] Make `_phi_chart_norm_route` report the effective pullback-group factor radius even when `phi_mstep_max_matrix_norm is None`, because the route then uses the fixed radius 5.0.

### Step 3: Remove retired names and moment code repository-wide

- [ ] Require zero old-class and executable moment-branch hits:

```powershell
rg -n "GaugeNaturalGradAdamW" vfe3 tests train_vfe3.py ablation.py scaling.py
rg -n "gauge_mom|gauge_m\b|gauge_v\b|gauge_step|heavy.ball.*phi|phi.*heavy.ball" vfe3/gauge_optim.py vfe3/train.py vfe3/run_artifacts.py
```

- [ ] Check retired serialized field names against this exact migration/fixture allowlist. Any hit outside it is stale live wiring and fails the task:

```powershell
$allowedLegacyFiles = @(
    'vfe3\config.py',
    'tests\test_config.py',
    'tests\test_checkpoint_resume.py',
    'tests\test_run_artifacts.py',
    'tests\test_final_audit_integrity_20260716.py',
    'tests\test_2026_07_15_cache_serialization_remediation.py'
)
$legacyHits = @(rg -l "m_phi_natural_grad|m_gauge_momentum|m_gauge_update_rule" vfe3 tests train_vfe3.py ablation.py scaling.py)
$unexpectedLegacyHits = @($legacyHits | Where-Object { $allowedLegacyFiles -notcontains $_ })
if ($unexpectedLegacyHits.Count -ne 0) { throw "unexpected retired phi controls: $unexpectedLegacyHits" }
```

- [ ] Within the allowlisted files, retain the old names only in `_RETIRED_PHI_CONFIG_FIELDS`, typed migration logic, raw historical payload fixtures, and explicit removal assertions. `gauge_mom`, `gauge_m`, `gauge_v`, and `gauge_step` may remain only in `tests/test_checkpoint_resume.py` fixtures that prove legacy state rejection. Historical dated audit/design evidence may retain quoted old names because the scans above deliberately exclude it.
- [ ] Verify the Task 2 class rename did not alter omega candidate math, state, cadence, or assertions.
- [ ] Run a complementary search for overclaims:

```powershell
rg -n -i "natural.gradient.*phi|phi.*natural.gradient|fisher.*pullback|pullback.*fisher" vfe3 tests train_vfe3.py ablation.py scaling.py
```

Replace executable-route terminology with the approved phrasing.

### Step 4: Run cleanup regressions and commit

```powershell
python -m pytest tests/test_exp8_buildout.py tests/test_omega_direct.py tests/test_2026_07_15_performance_remediation.py tests/test_final_audit_integrity_20260716.py --junitxml=C:\tmp\vfe3-phi-cleanup-green-20260717.xml
```

Expected: exit code zero, XML failures and errors both zero; old class/executable moment scans return no hits, and every retired config-name hit is inside the exact migration/fixture allowlist.

- [ ] Commit all exact cleanup files shown by `git status --short`:

```powershell
git add vfe3/gauge_optim.py vfe3/train.py vfe3/run_artifacts.py vfe3/viz/figures.py tests/test_exp8_buildout.py tests/test_omega_direct.py tests/test_2026_07_15_performance_remediation.py tests/test_final_audit_integrity_20260716.py tests/test_b5_finite_config_controls_20260716.py tests/test_checkpoint_resume.py tests/test_config.py tests/test_fix_config_audit.py tests/test_fp16_gradscaler.py tests/test_gauge_optim.py tests/test_hyperprior.py tests/test_omega_tilde_model_frame.py tests/test_phi_weight_decay.py
git commit -m "refactor: remove retired phi optimizer modes"
```

---

## Task 6: Add and run the RTX 5090 paired performance benchmark

**Files:**

- Create: `benchmarks/benchmark_phi_pullback_group.py`
- Create: `tests/test_phi_pullback_group_benchmark.py`
- Create after a successful real run: `docs/testing/2026-07-17-phi-pullback-group-rtx5090.json`
- Modify: `docs/2026-07-17-edits.md`

### Step 1: Write failing benchmark-contract tests

- [ ] Add a direct repo-root import smoke using `runpy.run_path` with a non-`__main__` run name; importing the click-run script must not execute the CUDA benchmark or parse CLI arguments.
- [ ] Add a small CPU schema/pairing test with two GL(2) rows and two repeats. Require paired metric/full samples, alternating call order, all environment/tolerance fields, and no CPU timing threshold.
- [ ] Monkeypatch `vfe3.gauge_optim.stage_pullback_group_candidate`, run the CPU helper, and assert the full-path measurement calls that production staging function. Structure the benchmark as `import vfe3.gauge_optim as gauge_optim` so the monkeypatch reaches the call; do not copy staging logic into the script.
- [ ] Pin percentile interpolation and the deterministic paired-bootstrap calculation against literal small arrays and a fixed bootstrap seed. Pin the JSON schema and the upper-confidence-bound gate independently of wall-clock values.
- [ ] Run red:

```powershell
python -m pytest tests/test_phi_pullback_group_benchmark.py --junitxml=C:\tmp\vfe3-phi-benchmark-red-20260717.xml
```

Expected: collection fails because the benchmark module does not exist.

### Step 2: Implement the direct-run benchmark

- [ ] Follow `benchmarks/benchmark_phi_projection.py` for repo-root bootstrap and the no-CLI click-to-run style, but do not reuse its K20 cases or historical timings.
- [ ] Configure K10 `block_glk`, two GL(5) blocks, active-row counts 128, 512, and 2,048, 20 untimed warmups, and at least 100 paired alternating measurements.
- [ ] Compare identical seeded phi rows/covectors through (a) the existing metric-only `pullback_metric_per_block` kernel and (b) `gauge_optim.stage_pullback_group_candidate`, which is the complete production strict direction plus trust scaling, BCH4 candidate, and exact-product residual acceptance path.
- [ ] Use CUDA events around only the measured regions. Synchronize outside those regions. Reset and record peak allocated memory separately for each case.
- [ ] Implement a deterministic bootstrap over paired differences. Write raw paired samples, medians, p95 values, confidence intervals for median and p95 differences, active rows, geometry-kernel time, peak memory, device name, CUDA/PyTorch versions, seed, warmups, repeats, and every fixed tolerance to the tracked JSON path.

### Step 3: Add the performance review gate

- [ ] For each paired bootstrap resample of common indices, calculate `p95(full_resample) / p95(metric_resample) - 1`. Use a fixed seed and at least 10,000 resamples to form the 2.5th/97.5th percentile interval. Require the upper 95 percent confidence bound to be at most `0.20`; an interval that straddles 20 percent is inconclusive and fails the gate just like a higher interval.
- [ ] If the upper bound exceeds 20 percent, stop publication, profile, revise the implementation, and rerun. Do not weaken numerical tolerances or acceptance gates to meet performance. Do not add a brittle CPU pytest timing assertion.
- [ ] Run the green benchmark-contract test before using the GPU:

```powershell
python -m pytest tests/test_phi_pullback_group_benchmark.py --junitxml=C:\tmp\vfe3-phi-benchmark-green-20260717.xml
```

Expected: exit code zero, XML failures and errors both zero.

### Step 4: Run on CUDA and inspect the actual JSON

```powershell
python benchmarks/benchmark_phi_pullback_group.py
```

Expected: CUDA identifies the RTX 5090, all three active-row cases complete, the JSON contains at least 100 paired samples per case, and the review gate passes. Record actual values in the dated edit log only after inspecting the file.

- [ ] Commit the benchmark, machine-readable evidence, and dated record:

```powershell
git add benchmarks/benchmark_phi_pullback_group.py tests/test_phi_pullback_group_benchmark.py docs/testing/2026-07-17-phi-pullback-group-rtx5090.json docs/2026-07-17-edits.md
git commit -m "perf: benchmark phi pullback group descent"
```

---

## Task 7: Complete focused/full verification, documentation, and the mandatory Git lifecycle

**Files:**

- Modify: `docs/2026-07-17-edits.md`
- Verify: every file changed by Tasks 1-6

### Step 1: Run the complete focused acceptance matrix

```powershell
python -m pytest tests/test_phi_preconditioner.py tests/test_gauge_optim.py tests/test_config.py tests/test_checkpoint_resume.py tests/test_run_artifacts.py tests/test_phi_outer_objective_gradient.py tests/test_phi_pullback_group_benchmark.py tests/test_fp16_gradscaler.py tests/test_exp8_buildout.py tests/test_fix_config_audit.py tests/test_phi_weight_decay.py tests/test_hyperprior.py tests/test_omega_tilde_model_frame.py tests/test_b5_finite_config_controls_20260716.py tests/test_2026_07_15_geometry_remediation.py tests/test_2026_07_15_driver_reliability_remediation.py tests/test_2026_07_15_cache_serialization_remediation.py tests/test_final_audit_integrity_20260716.py --junitxml=C:\tmp\vfe3-phi-focused-final-20260717.xml
```

- [ ] Confirm exit code zero. Read `tests`, `failures`, `errors`, and `skipped` from the XML; do not infer them from progress dots.

### Step 2: Run the full suite with machine-readable evidence

```powershell
python -m pytest --junitxml=C:\tmp\vfe3-full-phi-final-20260717.xml
```

- [ ] Confirm exit code zero and parse the JUnit attributes. If any test fails, diagnose and repair the root cause, rerun the focused file red/green, then rerun the full suite. Do not claim completion from an earlier artifact.

### Step 3: Perform static and diff verification

```powershell
git diff --check
rg -n "GaugeNaturalGradAdamW" vfe3 tests train_vfe3.py ablation.py scaling.py
rg -n "gauge_mom|gauge_m\b|gauge_v\b|gauge_step" vfe3/gauge_optim.py vfe3/train.py vfe3/run_artifacts.py
rg -n "TODO\(canonical-vfe-phi-mstep\)" vfe3/model/model.py
git status --short
git diff --stat
```

- [ ] Rerun Task 5's exact retired-field allowlist check. Require `git diff --check` to succeed, no old-class or executable moment-branch hits, no retired-field hits outside the migration/fixture allowlist, exactly one canonical TODO at the completed-loss seam, and no task-owned temporary artifacts inside the worktree.
- [ ] Review the entire staged diff. Confirm every changed line traces to the approved spec and no unrelated config value changed.

### Step 4: Update the dated edit record with observed evidence

- [ ] Append one implementation section to `docs/2026-07-17-edits.md`. Describe the objective contract, geometry, state removal, migration boundary, default preservation, focused/full JUnit counts, and RTX 5090 benchmark results using only values read from the final XML/JSON.
- [ ] Do not claim the canonical VFE phi M-step was implemented. State that the existing completed outer supervised scalar supplies the covector and that the code TODO marks the later fixed-returned-state objective.

### Step 5: Commit the final verification record

```powershell
git add docs/2026-07-17-edits.md
git status --short
git diff --cached --check
git commit -m "docs: record phi group descent verification"
```

### Step 6: Push, merge to main, and clean up safely

- [ ] Fetch and inspect the actual remote before publication:

```powershell
git fetch origin
git log --oneline -5 origin/main
```

- [ ] If `origin/main` moved, integrate it in the isolated worktree and rerun affected verification. Push the task branch, merge it into `main`, push `main`, then fetch and inspect the resulting `origin/main` SHA.
- [ ] Fast-forward the user's local `main` checkout only if `git status --short` proves doing so cannot overwrite user WIP. Otherwise leave it untouched and report the exact dirty paths and their owner.
- [ ] Remove only the task-owned temporary worktree and local task branch after confirming the remote merge. Do not remove the pre-existing `C:\tmp\vfe3-mphi-ng-investigation-20260717` worktree or its branch.
- [ ] Show final `git status --short` for the live checkout and any retained worktree. Report the task branch, commit SHA, pushed branch, resulting `origin/main` SHA, focused/full JUnit result, benchmark artifact, worktree removal, and any remaining dirty files.

---

## Plan Self-Review

- The plan implements the approved outer-objective-first route and places the canonical VFE TODO at the final scalar seam.
- The strict M-step geometry is separate from the existing E-step registry and preserves the full/per-block memory boundary.
- The plan includes every fixed numerical threshold, the left/right differential sign convention, right-product operand order, trust scaling, BCH backtracking, chart/transport guards, float64 island, and one-time commit cast.
- The default AdamW and `omega_direct` routes have explicit regression obligations.
- Heavy-ball, coordinatewise Adam-on-pullback, their public controls, and all phi moment slots are removed rather than deprecated.
- Serialized migration covers raw fingerprint order, the exact three retired keys, old/new conflicts, weight-only loading, all three legacy resume cases, and pre-mutation rejection.
- Token, positional, model-token, and model-positional factors; clipping; GradScaler; overflow; atomicity; checkpointing; diagnostics; and the RTX 5090 performance gate are all covered.
- No task contains an unresolved planning placeholder, guessed pass count, or K20 performance justification. The one explicit code `TODO` is the approved canonical-VFE objective marker required by the specification.
