# P3 Diagonal-KL Statistics Reuse Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an OFF-by-default A/B toggle that reuses canonical diagonal-KL pair statistics across attention and the filtering-gradient or `mm_exact` consumer.

**Architecture:** A focused `vfe3.gradients.pairwise_stats` module computes one graph-live float32 statistics bundle without changing the generic divergence registry. `belief_gradients` and `mm_exact_update` opt into it only on their existing canonical hand-kernel route; every other route and the disabled toggle execute the legacy code.

**Tech Stack:** Python 3.12, PyTorch 2.10/CUDA 12.8, pytest, JUnit XML.

## Global Constraints

- `reuse_pairwise_kl_stats` defaults to `False` and the disabled route is byte-identical.
- Do not edit `train_vfe3.py` or `ablation.py`.
- The optimized path is float32-only; other dtypes use the legacy route.
- Do not change `pairwise_energy`, the family/functional registry ABI, attention semantics, or clamp masks.
- Keep the MM upper-only self mask distinct from the two-sided pair mask.
- Use test-first RED/GREEN cycles and read pass counts from JUnit XML without adding `-q`.

---

### Task 1: Configuration and end-to-end toggle routing

**Files:**
- Modify: `vfe3/config.py`
- Modify: `vfe3/model/block.py`
- Modify: `vfe3/model/model.py`
- Modify: `vfe3/inference/e_step.py`
- Create: `tests/test_p3_pairwise_stats_reuse_20260711.py`

**Interfaces:**
- Produces: `VFE3Config.reuse_pairwise_kl_stats: bool = False`
- Produces: `e_step_iteration(..., reuse_pairwise_kl_stats: bool = False)`
- Consumes: existing `e_step` keyword forwarding for iteration-only controls.

- [ ] **Step 1: Write the failing toggle test**

Add a test that asserts the default is false, an explicit true value survives construction, and a
monkeypatched P3 helper is never reached by a one-iteration model forward while the toggle is false.

```python
def test_p3_toggle_defaults_off_and_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    assert VFE3Config().reuse_pairwise_kl_stats is False
    assert VFE3Config(reuse_pairwise_kl_stats=True).reuse_pairwise_kl_stats is True
```

- [ ] **Step 2: Run RED**

Run:

```powershell
python -m pytest tests/test_p3_pairwise_stats_reuse_20260711.py --junitxml=C:\tmp\vfe3-p3-red-toggle.xml
```

Expected: failure because `VFE3Config` has no `reuse_pairwise_kl_stats` field.

- [ ] **Step 3: Add and route the field**

Add the field beside the exactness-preserving performance toggles:

```python
reuse_pairwise_kl_stats: bool = False
```

Pass it from `vfe_block` and `_refine_s` into `e_step`; accept it in `free_energy_value` as an
iteration-only ignored control and in `e_step_iteration` as an executable control. Forward it to
both `belief_gradients` and `mm_exact_update`.

- [ ] **Step 4: Run GREEN and commit**

```powershell
python -m pytest tests/test_p3_pairwise_stats_reuse_20260711.py tests/test_curated_inference_math_20260709.py --junitxml=C:\tmp\vfe3-p3-toggle.xml
git add vfe3/config.py vfe3/model/block.py vfe3/model/model.py vfe3/inference/e_step.py tests/test_p3_pairwise_stats_reuse_20260711.py
git commit -m "feat(perf): add opt-in P3 statistics toggle"
```

### Task 2: Canonical diagonal-KL statistics builder

**Files:**
- Create: `vfe3/gradients/pairwise_stats.py`
- Modify: `tests/test_p3_pairwise_stats_reuse_20260711.py`

**Interfaces:**
- Produces: `DiagonalKLPairStats(energy, pair_mask, inv_sigma_t, delta_tq)`.
- Produces: `diagonal_kl_pair_stats(mu_q, sigma_q, mu_t, sigma_t, *, kl_max, eps, irrep_dims)`.

- [ ] **Step 1: Write the failing value and mask tests**

Cover `irrep_dims=None`, `[K]`, equal `[2, 2]`, and unequal `[1, 3]`. Compare energy with generic
`pairwise_energy`, require exact mask equality, and compare the two sufficient statistics to their
direct definitions.

```python
torch.testing.assert_close(stats.energy, reference_energy, atol=1e-5, rtol=1e-6)
assert torch.equal(stats.pair_mask, reference_mask)
torch.testing.assert_close(stats.inv_sigma_t, 1.0 / sigma_t.clamp(min=eps))
torch.testing.assert_close(stats.delta_tq, mu_t - mu_q.unsqueeze(-2))
```

- [ ] **Step 2: Run RED**

Expected: import failure for `vfe3.gradients.pairwise_stats`.

- [ ] **Step 3: Implement the focused module**

Use a frozen dataclass and a private coordinate reducer. Compute one clamped transported variance,
one reciprocal, and one transported-minus-query difference. Keep trace, Mahalanobis, and logdet
reductions separate, then call `safe_kl_clamp` and derive the mask from the clamped energy.

```python
@dataclass(frozen=True)
class DiagonalKLPairStats:
    energy:        torch.Tensor
    pair_mask:     torch.Tensor
    inv_sigma_t:   torch.Tensor
    delta_tq:      torch.Tensor
```

- [ ] **Step 4: Run GREEN and commit**

```powershell
python -m pytest tests/test_p3_pairwise_stats_reuse_20260711.py --junitxml=C:\tmp\vfe3-p3-stats.xml
git add vfe3/gradients/pairwise_stats.py tests/test_p3_pairwise_stats_reuse_20260711.py
git commit -m "feat(perf): build reusable diagonal KL statistics"
```

### Task 3: Consume one statistics bundle in gradient and MM routes

**Files:**
- Modify: `vfe3/gradients/kernels.py`
- Modify: `tests/test_p3_pairwise_stats_reuse_20260711.py`

**Interfaces:**
- Consumes: `DiagonalKLPairStats` from Task 2.
- Extends: `belief_gradients(..., reuse_pairwise_kl_stats: bool = False)`.
- Extends: `mm_exact_update(..., reuse_pairwise_kl_stats: bool = False)`.
- Extends the private registered kernel with optional `pair_inv_sigma_t` and `pair_delta_tq` tensors.

- [ ] **Step 1: Write RED consumption tests**

Monkeypatch `diagonal_kl_pair_stats` to count calls and return deliberately modified inverse and
difference tensors. Require exactly one construction per consumer and require the poisoned values
to affect its result. Also monkeypatch the helper to raise while the toggle is false.

- [ ] **Step 2: Add a frozen legacy characterization**

Copy the pre-P3 `belief_gradients`/`mm_exact_update` arithmetic into test-only reference helpers.
Compare forward values and VJPs for single-head and `[2, 2]` layouts, `lambda_twohop` in `{0, 0.7}`,
sigma update enabled/disabled, and the supported alpha modes. Use `atol=1e-5, rtol=1e-6` for forward
values and `atol=5e-5, rtol=1e-5` for VJPs.

- [ ] **Step 3: Run RED**

Expected: the helper is not called and poisoned values are not consumed.

- [ ] **Step 4: Implement the consumers**

Construct statistics only when the toggle is true, the hand-kernel route is selected, and all pair
inputs are float32. Reuse `energy` and `pair_mask` for beta, `-delta_tq * inv_sigma_t` for the mean
gradient, `inv_sigma_t` for the sigma gradient and MM precision, and `mu_t * inv_sigma_t` for the MM
precision-weighted mean. Leave the existing code in a separate false/fallback branch.

- [ ] **Step 5: Run GREEN and commit**

```powershell
python -m pytest tests/test_p3_pairwise_stats_reuse_20260711.py tests/test_tier12_estep.py tests/test_mm_exact_prior_anchor.py tests/test_gradients_kernels.py tests/test_perf_equivalence.py --junitxml=C:\tmp\vfe3-p3-consumers.xml
git add vfe3/gradients/kernels.py tests/test_p3_pairwise_stats_reuse_20260711.py
git commit -m "feat(perf): reuse pair statistics in filtering updates"
```

### Task 4: Boundary, route-isolation, and CUDA correctness gates

**Files:**
- Modify: `tests/test_p3_pairwise_stats_reuse_20260711.py`
- Modify: `check_gpu_tests.py`

**Interfaces:**
- Verifies the completed Task 1-3 interfaces without adding production behavior.

- [ ] **Step 1: Add exact boundary tests**

Pin exact zero pair energy, exact `kl_max` pair energy, exact `kl_max` self energy, causal no-self
priors, saturated-row pass-through, and frozen-sigma equality with `torch.equal`.

- [ ] **Step 2: Add route-isolation tests**

Make the helper raise and prove it is not called for smoothing, Renyi order 0.5, full covariance,
entropy-suppressed attention, nonflat transport, non-float32 inputs, or when the toggle is false.

- [ ] **Step 3: Add CUDA smoke routing**

Add one real-CUDA node that executes both gradient and MM consumers with the toggle on and checks
finite outputs, finite VJPs, and equality of masks with the generic reference. Register the node in
`check_gpu_tests.py` without adding a timing assertion.

- [ ] **Step 4: Run CPU and CUDA gates and commit**

```powershell
python -m pytest tests/test_p3_pairwise_stats_reuse_20260711.py tests/test_tier12_estep.py tests/test_mm_exact_prior_anchor.py tests/test_gradients_kernels.py tests/test_perf_equivalence.py --junitxml=C:\tmp\vfe3-p3-focused.xml
$env:VFE3_TEST_DEVICE='cuda'; python -m pytest tests/test_p3_pairwise_stats_reuse_20260711.py --junitxml=C:\tmp\vfe3-p3-cuda.xml
git add tests/test_p3_pairwise_stats_reuse_20260711.py check_gpu_tests.py
git commit -m "test(perf): pin P3 equivalence and route isolation"
```

### Task 5: Documentation, complete verification, review, and delivery

**Files:**
- Modify: `docs/2026-07-11-edits.md`
- Verify only: `train_vfe3.py`, `ablation.py`

- [ ] **Step 1: Record implementation and verification**

Append the exact files, toggle semantics, RED evidence, JUnit-derived counts, CUDA device, and the
user-owned benchmark handoff to the existing dated document.

- [ ] **Step 2: Prove entry-point configs were untouched**

```powershell
git diff origin/main -- train_vfe3.py ablation.py
```

Expected: no output.

- [ ] **Step 3: Run the complete CPU suite**

```powershell
python -m pytest --junitxml=C:\tmp\vfe3-p3-full.xml
```

Read `tests`, `failures`, `errors`, `skipped`, and elapsed time from the XML.

- [ ] **Step 4: Run review and diff gates**

Require `git diff --check`, inspect the staged diff, and obtain independent correctness and
numerical reviews. Resolve every finding and rerun affected tests.

- [ ] **Step 5: Commit, push, merge, and clean**

```powershell
git add vfe3 tests check_gpu_tests.py docs/2026-07-11-edits.md
git commit -m "docs: record P3 verification"
git push -u origin codex/p3-diag-kl-stats-20260711
```

Fast-forward or merge the verified branch into `main`, push `main`, fetch, confirm `origin/main`,
remove task-owned XML files and the temporary worktree, and leave the user's live config WIP
untouched. Report the benchmark command/config needed for the user's OFF/ON comparison.
