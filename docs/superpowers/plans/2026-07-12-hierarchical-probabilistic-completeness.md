# Hierarchical Probabilistic Completeness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the probabilistic `q/p/s/h` hierarchy with one typed differentiable hierarchical/reporting evaluator, a bitwise-pure legacy q-only scalar path, family-correct model states, full-SPD and nonflat model-channel parity, a covariance-bearing CG route, and a prior-bank decoder that uses the configured family and divergence.

**Architecture:** A tensor-only `HierarchicalFreeEnergyTerms` record becomes the authoritative hierarchical and reporting decomposition. Existing scalar `free_energy()` retains its original one-shot reduction order as the bitwise-pure q-only training compatibility path and is pinned numerically, rather than bitwise after regrouping, to the row decomposition. `PriorBank` stores packed strict-lower Cholesky coordinates for the model-channel s/r tables only under `gaussian_full`, while every model-channel calculation dispatches through the configured family and the active transport registry. CG obtains an opt-in first-order full-covariance pushforward, and noncanonical decodes use a generic family/divergence registrant while the current fused Gaussian-KL kernels remain the canonical fast path.

**Tech Stack:** Python 3, PyTorch, typed `NamedTuple` contracts, existing VFE3 family/divergence/transport/decode registries, pytest with JUnit XML.

## Global Constraints

The deployed target-blind E-step remains the default and receives no target or future-token tensor in this plan. Observation-conditioned inference is a prerequisite owned by `docs/plans/2026-07-11-backprop-free-vfe-lm-plan.md`; reference that implementation when it lands instead of adding a second observation design here. `lambda_h=0`, `lambda_gamma=0`, `use_cg_coupling=False`, `family="gaussian_diagonal"`, `transport_mode="flat"`, `renyi_order=1.0`, `divergence_family="renyi"`, and the existing Gaussian-KL decode must preserve exact state-dict keys, RNG draws, tensor values, and gradients. Full-SPD model-channel s/r storage is created only for `family="gaussian_full"`; its packed strict-lower storage is an opt-in computationally extreme path. Vocabulary-prior and decode variance tables remain diagonal in every family. Nonflat model transport shares connection parameters with the belief channel but evaluates registry state features from the model-channel `s` state. No neural layer, MLP, activation, or new dependency is allowed. Production tensors remain float32 unless the existing numerical policy enters a documented float64 island. Every new focused test uses CPU with `K < 6`; CUDA is reserved for the final smoke. Every implementation task starts from a failing test, reads pass counts from JUnit XML, updates `docs/2026-07-12-edits.md`, and uses the repository's isolated-worktree git lifecycle.

---

### Task 1: Introduce the authoritative typed hierarchical evaluator

**Files:**

- Create: `tests/test_hierarchical_probabilistic_completeness_20260712.py`
- Modify: `vfe3/free_energy.py:346-466`
- Modify: `vfe3/metrics.py:115-199`
- Modify: `vfe3/model/model.py:1542-1795,2503-2548`
- Modify: `docs/2026-07-12-edits.md`

**Interfaces:**

- Add `BeliefFreeEnergyRows` in `vfe3.free_energy`. Its five fields are per-query rows with shape `(..., N)`: `self_coupling`, `belief_coupling`, `attention_entropy`, `twohop_coupling`, and `observation_nll`. A private `_belief_free_energy_rows(...)` accepts the current `free_energy()` tensor inputs but is a new hierarchical/diagnostic decomposition path: it collapses coordinate, key, and optional head axes while preserving the batch/query axes. It accepts `beta_override: Optional[torch.Tensor] = None`; the compatibility metrics wrapper supplies its captured beta argument so diagnostic weights are not recomputed. The five rows are already signed and weighted contributions: `lambda_beta` is included in both beta rows, `lambda_twohop` is included in the two-hop row, and `observation_nll=-log_likelihood` when the legacy argument is present. The legacy scalar `free_energy()` body does not delegate to this row path, because float32 one-shot reductions are not bitwise associative with row-then-query reductions.
- Add the following exact scalar evaluator. Every input has shape `(..., N)`; absent blocks are represented by `torch.zeros_like(belief_rows.self_coupling)`, never by a hidden branch. The first four rows and `observation_nll_rows` use `q_reduction`; the three s-channel rows use `model_reduction`. Each reduction accepts only `"sum"` or `"mean"` and applies directly to all batch/query entries after structural axes have already been collapsed.

```python
class BeliefFreeEnergyRows(NamedTuple):
    self_coupling:     torch.Tensor
    belief_coupling:   torch.Tensor
    attention_entropy: torch.Tensor
    twohop_coupling:   torch.Tensor
    observation_nll:   torch.Tensor

class HierarchicalFreeEnergyTerms(NamedTuple):
    self_coupling:     torch.Tensor
    belief_coupling:   torch.Tensor
    attention_entropy: torch.Tensor
    twohop_coupling:   torch.Tensor
    hyper_prior:       torch.Tensor
    model_coupling:    torch.Tensor
    meta_entropy:      torch.Tensor
    observation_nll:   torch.Tensor
    total:             torch.Tensor


def hierarchical_free_energy_terms(
    self_coupling_rows:     torch.Tensor,  # (..., N), already alpha-weighted and regularized
    belief_coupling_rows:   torch.Tensor,  # (..., N), lambda_beta * sum_j beta_ij E_ij
    attention_entropy_rows: torch.Tensor,  # (..., N), lambda_beta * tau * sum_j beta log(beta/pi)
    twohop_coupling_rows:   torch.Tensor,  # (..., N), lambda_twohop * sum_k (beta beta)_ik E_ik
    hyper_prior_rows:       torch.Tensor,  # (..., N), lambda_h_i D(s_i||h) + R_h
    model_coupling_rows:    torch.Tensor,  # (..., N), lambda_gamma * sum_j gamma_ij E^s_ij
    meta_entropy_rows:      torch.Tensor,  # (..., N), lambda_gamma * tau_g * sum_j gamma log(gamma/pi_s)
    observation_nll_rows:   torch.Tensor,  # (..., N), -E_q[log p(o|x)]

    *,
    q_reduction:     str = "sum",
    model_reduction: str = "mean",
) -> HierarchicalFreeEnergyTerms:
```

The evaluator validates that all inputs are tensors with matching `(..., N)` shapes and devices, reduces each row exactly once, and assembles `total` in the field order shown. It never detaches or reweights an input. The only detach remains inside `_belief_free_energy_rows` when it forms the already documented fixed two-hop weights. Call sites choose whether model-frame construction is passive or attached before producing model-coupling rows; that choice is no longer hidden inside the scalar evaluator. This plan supplies an exact zero observation row, which remains the only production value until the separate observation-model plan is implemented.

Call profiles are explicit. Legacy `free_energy()` keeps its existing scalar expressions and reduction order byte-for-byte; it is the pure q-only training route and does not call the row evaluator. `VFEModel.diagnostics()` uses `q_reduction="sum", model_reduction="sum"` so its persisted component scales remain unchanged within numerical tolerance. When the opt-in scored h/s blocks are active, the outer loss passes zero q rows plus live h/s rows through the evaluator with `model_reduction="mean"`; this preserves their existing per-token scale without regrouping the legacy q scalar.

- [ ] **Step 1: Write the red decomposition and gradient tests.** Add `test_hierarchical_terms_sum_exact_components`, `test_q_row_specialization_matches_free_energy_numerically`, `test_legacy_free_energy_reduction_order_is_bitwise_unchanged`, `test_model_mean_vs_sum_reduction_is_explicit`, `test_gamma_energy_gradient_is_not_hiddenly_detached`, and `test_observation_slot_defaults_to_zero` using `K=3`, `N=3`. Use float64 for tight row-oracle checks. For the legacy-order test, copy the pre-change scalar expressions into a local test oracle, use ordinary float32 inputs chosen so regrouped sums differ, and require `torch.equal(free_energy(...), legacy_scalar_oracle(...))` for values and gradients. Add compatibility-wrapper cases with `lambda_beta=0.25`, nonzero `lambda_twohop`, nonzero `log_likelihood`, and both attention-entropy settings. Require `belief_coupling`, `attention_entropy`, `twohop_coupling`, and `observation_likelihood` to retain their current raw public values while `total` alone applies weights, entropy gating, and the negative likelihood sign; compare that total numerically with `free_energy()`.
- [ ] **Step 2: Run the focused red tests.**

```powershell
$env:VFE3_TEST_DEVICE = "cpu"
python -m pytest tests/test_hierarchical_probabilistic_completeness_20260712.py --junitxml=C:\tmp\vfe3-hierarchy-task1-red.xml
```

Expected: the JUnit file records failures because `HierarchicalFreeEnergyTerms` and `hierarchical_free_energy_terms` do not exist.

- [ ] **Step 3: Implement the row record and evaluator without touching scalar reduction order.** Add `_belief_free_energy_rows(...)` beside `free_energy()` and reduce q/model rows through separate validated helpers. Form the hierarchical `total` in this fixed order: self, belief coupling, attention entropy, two-hop, hyper-prior, model coupling, meta-entropy, observation NLL. Leave every expression and one-shot `.sum()` in `free_energy()` unchanged; do not make it delegate to row reductions.
- [ ] **Step 4: Replace duplicate diagnostic and opt-in outer-loss assembly without breaking the registry.** Preserve the complete public raw-input signature and raw-field contract of `metrics.free_energy_terms(self_div, energy, beta, alpha, *, tau, ...)`. The wrapper computes the same raw one-hop, entropy, two-hop, and observation-likelihood values it exposes today. Separately, it builds weighted signed `BeliefFreeEnergyRows`: apply `lambda_beta` to one-hop and to entropy only when entropy participates, apply `lambda_twohop` to the two-hop row, and negate the observation-likelihood row. Supply explicit zero model rows, invoke `hierarchical_free_energy_terms`, and use only its `total`; return the legacy raw float fields under their existing names. Never divide a weighted row to reconstruct a raw field, because zero weights must remain valid. This keeps `_m_free_energy_terms` and `compute_metrics` callable exactly as the reporting-registry plan requires and preserves all public diagnostic semantics. Make `VFEModel.diagnostics()` supply its live hyper-prior/gamma rows to the typed evaluator instead of incrementing `d["total"]` independently. Its q rows are `(N,)`; therefore compute model rows from `token_ids[:1]` and explicitly index their singleton batch as `[0]` before strict shape validation. Add `_gamma_coupling_rows(..., head_reduction: str) -> tuple[torch.Tensor, torch.Tensor]`. After summing the key axis it accepts only `"mean"` or `"sum"`: training uses `mean(dim=head)` to turn `(B,H,N)` into `(B,N)` before `model_reduction="mean"`, exactly matching the live `_gamma_coupling_term` average over B/H/N; diagnostics uses `sum(dim=head)` before selecting batch zero, preserving its existing sum-over-heads scale. The single-head path inserts/removes a singleton H axis through the same helper. Add diagnostics integration tests for H=1 and H=2 first-sequence shapes and finite totals. On `not s_e_step`, retain the full `(B,N)` axis, construct `hyper_prior_rows` from `_hyper_prior_weighted`, construct the two mean-over-head gamma rows from the same active model frame, use `zeros_like` for absent model blocks and every q slot, and add `hierarchical_free_energy_terms(..., q_reduction="sum", model_reduction="mean").total` once to the unchanged legacy q loss. When `include_attention_entropy=False`, pass `zeros_like(model_coupling_rows)` as the evaluator's `meta_entropy_rows`; diagnostics may still expose the raw counterfactual meta measurement, but it must not enter `total`. This opt-in migration changes fp32 association on the canonical gamma branch from the reduced envelope to the mathematically equal coupling-plus-entropy decomposition; it is allowed only when `lambda_h` or `lambda_gamma` activates the model channel and cannot affect the all-off pure path. Regressions parameterize `include_attention_entropy` true/false and H=1/H=2, compute the corresponding pre-change `_hyper_prior_weighted(...).mean() + cfg.lambda_gamma * _gamma_coupling_term(...)` expression, and use tight `torch.testing.assert_close` checks for values and s/r/gamma gradients under constant lambda modes; no bitwise claim is made for the active model channel.
- [ ] **Step 5: Run the task tests and existing objective tests.**

```powershell
python -m pytest tests/test_hierarchical_probabilistic_completeness_20260712.py tests/test_free_energy.py tests/test_model_channel_diagnostics_2026_06_13.py --junitxml=C:\tmp\vfe3-hierarchy-task1-green.xml
```

Expected: `failures="0"` and `errors="0"`; the row decomposition is numerically equal under the float64 tolerance, while the legacy float32 scalar value and gradients use `torch.equal` against the pre-change reduction-order oracle.

- [ ] **Step 6: Commit the evaluator boundary.**

```powershell
git add vfe3/free_energy.py vfe3/metrics.py vfe3/model/model.py tests/test_hierarchical_probabilistic_completeness_20260712.py docs/2026-07-12-edits.md
git commit -m "feat: unify hierarchical free-energy evaluation"
```

### Task 2: Add family-owned model-channel and full-SPD prior storage

**Files:**

- Create: `vfe3/families/covariance_tables.py`
- Modify: `vfe3/model/prior_bank.py:166-371,474-568`
- Modify: `vfe3/model/model.py:1596-1676`
- Modify: `vfe3/config.py:1319-1329,1550-1591,1669-1680`
- Modify: `vfe3/train.py:202-275`
- Test: `tests/test_hierarchical_probabilistic_completeness_20260712.py`
- Test: `tests/test_checkpoint_resume.py`
- Modify: `docs/2026-07-12-edits.md`

**Interfaces:**

- Produces: `packed_strict_lower_size(K: int) -> int`, `covariance_from_packed(log_diag, packed_lower, *, eps) -> torch.Tensor`, and `packed_from_covariance(covariance, *, eps) -> tuple[torch.Tensor, torch.Tensor]`.
- Produces: `PriorBank.encode_s(token_ids) -> tuple[torch.Tensor, torch.Tensor]` and `PriorBank.r_parameters() -> tuple[torch.Tensor, torch.Tensor]`, with the s/r covariance rank following `family_cov_kind(cfg.family)`.

```python
def covariance_from_packed(
    log_diag:    torch.Tensor,
    packed_lower: torch.Tensor,
    *,
    eps:         float = 1e-6,
) -> torch.Tensor:
    k = log_diag.shape[-1]
    row, col = torch.tril_indices(k, k, offset=-1, device=log_diag.device)
    chol = log_diag.new_zeros(*log_diag.shape[:-1], k, k)
    chol[..., row, col] = packed_lower
    diagonal_variance = bounded_variance_from_log(log_diag, eps=eps)
    chol.diagonal(dim1=-2, dim2=-1).copy_(torch.sqrt(diagonal_variance))
    return chol @ chol.transpose(-1, -2)
```

Import and reuse `vfe3.numerics.bounded_variance_from_log`; do not introduce a second max-log policy. The bounded variance is formed before the square root, so `log_diag=100` remains finite in float32 and follows the same warning/clamp behavior as existing diagonal tables.

Only the full family creates `s_sigma_lower_embed` and `r_sigma_lower`. They are packed `(..., K*(K-1)//2)` tensors initialized to zero, so initial model-channel covariances are exactly diagonal. This task deliberately does not add off-diagonal vocabulary-prior or decode tables: `sigma_log_embed` and an untied decode variance table remain diagonal priors, which the existing full-belief decode embeds as diagonal full matrices. The diagonal and Laplace model channels create no packed keys. `barycenter_r_()` uses full Gaussian moment matching for `gaussian_full`, keeps the existing formula for `gaussian_diagonal`, and configuration rejects `r_update_mode="barycenter"` for families without a registered barycenter.

- [ ] **Step 1: Write red storage, family, optimizer, and checkpoint tests.** Cover diagonal state-dict identity; full s/r packed shapes; SPD reconstruction; nonzero off-diagonal s/r gradients; `encode_s`/`r_parameters` rank parity; the absence of vocabulary/decode lower-triangle keys; full-Gaussian barycenter moment matching; Laplace model-channel construction with `r_update_mode="gradient"`; exact optimizer coverage; and checkpoint round-trip of both packed model-channel tables. Add `log_diag=torch.full((3,), 100.0)` with zero packed lower coordinates, require a finite covariance, and require its diagonal to equal `bounded_variance_from_log(log_diag)` exactly.
- [ ] **Step 2: Run the red tests.**

```powershell
python -m pytest tests/test_hierarchical_probabilistic_completeness_20260712.py tests/test_checkpoint_resume.py --junitxml=C:\tmp\vfe3-hierarchy-task2-red.xml
```

Expected: failures identify the absent packed covariance helpers and the current full-family `s_e_step` rejection.

- [ ] **Step 3: Implement packed model-channel storage and accessors.** Add the helper module, pass `family` into `PriorBank`, create packed parameters only for full s/r covariance, and replace direct s/r log-variance reads with `encode_s` and `r_parameters`. Leave vocabulary encode/decode variance storage unchanged.
- [ ] **Step 4: Dispatch hyper-prior family construction.** Replace hardcoded `DiagonalGaussian` in `_hyper_prior_kl` with `get_family(cfg.family)` and use `PriorBank.r_parameters()`.
- [ ] **Step 5: Remove only the obsolete full-s rejection.** Retain all per-coordinate functional guards and add the non-Gaussian barycenter rejection before model construction.
- [ ] **Step 6: Group packed covariance parameters.** Put `s_sigma_lower_embed` and learnable `r_sigma_lower` in the sigma optimizer role with `m_p_sigma_lr` and the configured sigma weight decay; keep the exact-coverage assertion load-bearing.
- [ ] **Step 7: Run focused family, optimizer, and resume tests.**

```powershell
python -m pytest tests/test_hierarchical_probabilistic_completeness_20260712.py tests/test_full_covariance.py tests/test_laplace_family.py tests/test_hyperprior.py tests/test_checkpoint_resume.py tests/test_train.py --junitxml=C:\tmp\vfe3-hierarchy-task2-green.xml
```

Expected: `failures="0"`, `errors="0"`, and every new model uses `K <= 5`.

- [ ] **Step 8: Commit the family storage layer.**

```powershell
git add vfe3/families/covariance_tables.py vfe3/model/prior_bank.py vfe3/model/model.py vfe3/config.py vfe3/train.py tests/test_hierarchical_probabilistic_completeness_20260712.py tests/test_checkpoint_resume.py docs/2026-07-12-edits.md
git commit -m "feat: complete model-channel covariance families"
```

### Task 3: Give the model channel family and nonflat transport parity

**Files:**

- Modify: `vfe3/model/model.py:734-830,1678-1795`
- Modify: `vfe3/model/block.py:45-68`
- Modify: `vfe3/config.py:2110-2159`
- Test: `tests/test_hierarchical_probabilistic_completeness_20260712.py`
- Test: `tests/test_live_s_model_channel.py`
- Test: `tests/test_regime_ii_covariant.py`
- Test: `tests/test_regime_ii_link.py`
- Modify: `docs/2026-07-12-edits.md`

**Interfaces:**

- `_refine_s(...)` consumes the model's active `connection_W`, `connection_M`, or `connection_L` and passes `family=cfg.family`, `transport_mode=cfg.transport_mode`.
- `_gamma_energy(...)` builds transport through the same registry and returns `(energy, tau, log_prior)` for the configured family.
- Shared nonflat parameter tensors follow registry metadata. Stateful `regime_ii`/`regime_ii_covariant` transport evaluates channel-local q versus s means/covariances. `regime_ii_link` and `regime_ii_link_charted` remain deliberately belief-state-independent and share `connection_L`; the charted variant may additionally depend on its supplied frame coordinates.

- [ ] **Step 1: Write red family-parity tests.** Construct diagonal Gaussian, full Gaussian, and Laplace model channels at `K=4`; assert `_refine_s`, `_hyper_prior_kl`, and `_gamma_energy` use the selected family and return the expected covariance ranks.
- [ ] **Step 2: Write red nonflat parity tests.** For all four nonflat modes, perturb only the active shared connection parameter and require both q and s energies to change. For `regime_ii` and `regime_ii_covariant`, perturb q or s state separately and require only that channel's state-derived connection features to change. For `regime_ii_link`, require the built transport to remain invariant to q/s mean and covariance perturbations but change with `connection_L`. For `regime_ii_link_charted`, require the same q/s invariance, sensitivity to `connection_L`, and sensitivity to the appropriate supplied frame coordinates. These expectations come from registration metadata, not a blanket statefulness assumption.
- [ ] **Step 3: Run the red tests.**

```powershell
python -m pytest tests/test_hierarchical_probabilistic_completeness_20260712.py tests/test_live_s_model_channel.py tests/test_regime_ii_covariant.py tests/test_regime_ii_link.py --junitxml=C:\tmp\vfe3-hierarchy-task3-red.xml
```

Expected: failures show `_refine_s` and `_gamma_energy` still select diagonal/flat behavior.

- [ ] **Step 4: Thread the active family and connection.** Use transport-registration metadata rather than mode-name conditionals when deciding whether to pass `s_mu` and `s_sigma`; forward all connection parameters and link controls through the shared kwargs builder.
- [ ] **Step 5: Attach connection gradients deliberately.** The s-table and shared-connection gradients remain live; the existing passive model-frame policy remains explicit at its caller and is not widened by this task.
- [ ] **Step 6: Replace the obsolete nonflat warning with validation.** Configuration now rejects only combinations unsupported by the registered family's covariance action; valid nonflat model channels construct without the flat-island warning.
- [ ] **Step 7: Run parity and finite-gradient tests.**

```powershell
python -m pytest tests/test_hierarchical_probabilistic_completeness_20260712.py tests/test_live_s_model_channel.py tests/test_gamma_coupling.py tests/test_regime_ii.py tests/test_regime_ii_covariant.py tests/test_regime_ii_link.py --junitxml=C:\tmp\vfe3-hierarchy-task3-green.xml
```

Expected: `failures="0"` and `errors="0"`.

- [ ] **Step 8: Commit transport parity.**

```powershell
git add vfe3/model/model.py vfe3/model/block.py vfe3/config.py tests/test_hierarchical_probabilistic_completeness_20260712.py tests/test_live_s_model_channel.py tests/test_regime_ii_covariant.py tests/test_regime_ii_link.py docs/2026-07-12-edits.md
git commit -m "feat: share connection laws with the model channel"
```

### Task 4: Complete CG covariance and probabilistic-energy participation

**Files:**

- Modify: `vfe3/config.py:178-191,980-1000`
- Modify: `vfe3/contracts.py:10-16`
- Modify: `vfe3/model/cg_coupling.py`
- Modify: `vfe3/model/block.py:147-162`
- Modify: `vfe3/model/model.py:1466-1516,2835-3012`
- Test: `tests/test_cg.py`
- Test: `tests/test_hierarchical_probabilistic_completeness_20260712.py`
- Modify: `docs/2026-07-12-edits.md`

**Interfaces:**

- Produces: `CGMomentResult(mu, sigma, jacobian)` and `CGCoupling.forward_moments(mu, sigma) -> CGMomentResult`.
- Extends typed `MStepCapture` with `cg_moment_energy_rows: List[torch.Tensor]` and `cg_pre_moments: List[Tuple[torch.Tensor, torch.Tensor]]`; these keys are present only when `cg_energy_weight>0`.
- Adds: `cg_covariance_mode: str = "passthrough"` with values `"passthrough" | "delta_full"`; `delta_full` requires `family="gaussian_full"`.
- Produces the exact q-only regularizer below. It is the post-CG moment divergence `D(q_post || q_pre)`, not a second hierarchical total and not a post-minus-pre copy of unrelated q/p/s/h terms.

```python
def cg_moment_energy_rows(
    pre_mu:     torch.Tensor,  # (..., N, K)
    pre_sigma:  torch.Tensor,  # (..., N, K) or (..., N, K, K)
    post_mu:    torch.Tensor,  # same shape as pre_mu
    post_sigma: torch.Tensor,  # same shape as pre_sigma

    *,
    renyi_order:      float = 1.0,
    kl_max:           float = 100.0,
    eps:              float = 1e-6,
    family:           str   = "gaussian_diagonal",
    divergence_family: str  = "renyi",
) -> torch.Tensor:            # (..., N), D(q_post || q_pre)
```

- Adds: `cg_energy_weight: float = 0.0`. Allocate the M-step capture when `cfg.mstep_self_coupling_weight > 0.0 or cfg.cg_energy_weight > 0.0`, but initialize CG lists only when `cg_energy_weight>0`. Under attached E-step estimators, each `vfe_block` appends its attached `cg_moment_energy_rows(...)` tensor. Under `effective_e_step_gradient="detach"`, the no-grad block appends detached pre-CG `(mu, sigma)` moments instead; after the stack, `torch.enable_grad()` re-evaluates the shared `CGCoupling.forward_moments` from each fixed pair and computes rows, so the regularizer stays detached from belief inference but attached to `path_weights`. Only when `cg_energy_weight>0`, require exactly `n_layers` nonempty row tensors and define `cg_moment_energy = torch.stack([rows.mean() for rows in per_layer_rows]).mean()`. A capture allocated solely for M-step self-coupling never reads or stacks a CG list. The outer objective adds `cg_energy_weight * cg_moment_energy` exactly once. Diagnostics report `cg_moment_energy`, the ordered `cg_moment_energy_layers` values, and `objective_total_with_cg` separately; `HierarchicalFreeEnergyTerms.total` remains the canonical q/p/s/h total and is never reweighted or double-counted.

For each bilinear path `C(x tensor y)`, construct the exact Jacobian at the current mean from the two contractions `C(· tensor y)` and `C(x tensor ·)`. The delta-method covariance is `J @ Sigma @ J.T`, symmetrized before return. This is explicitly a first-order Gaussian moment closure, not an exact distributional pushforward. With zero path weights, `J=I`, `mu_out=mu`, and `sigma_out=sigma` exactly.

- [ ] **Step 1: Write red Jacobian and covariance tests.** At `K=3` or `K=5`, compare the analytic Jacobian to `torch.autograd.functional.jacobian`, compare `J Sigma J.T` to the returned covariance, test SPD/equivariance, and require exact zero-weight identity.
- [ ] **Step 2: Write red energy-participation tests.** Set the shared CG module's path weights nonzero and `cg_energy_weight>0`; compare `cg_moment_energy_rows` to a direct active-family divergence oracle and require the outer loss and `path_weights.grad` to change under both `unroll` and `detach`. In detach mode, require fixed pre-CG inputs with finite nonzero `path_weights.grad`. With `cg_energy_weight=0`, require the pre-change loss exactly. Add `mstep_self_coupling_weight>0`, `use_cg_coupling=False`, `cg_energy_weight=0` and require no CG list access, no empty stack, and exact pre-change output. Under `prior_source="token", s_e_step=False, lambda_h>0`, compare s/r parameter gradients at weight zero and positive weight and require exact equality, proving the CG regularizer cannot reweight the independent h/s block. Add an `n_layers=2` model using the live shared `self.cg_coupling`. A layer-indexed spy on `cg_moment_energy_rows` returns attached constant rows `1` and `3` on its two calls; assert two ordered captures and a reported layer mean of `2`. This proves neither application is omitted without inventing per-block weights.
- [ ] **Step 3: Run the red CG tests.**

```powershell
python -m pytest tests/test_cg.py tests/test_hierarchical_probabilistic_completeness_20260712.py --junitxml=C:\tmp\vfe3-hierarchy-task4-red.xml
```

Expected: failures identify the missing Jacobian/covariance result and energy weight.

- [ ] **Step 4: Implement the grouped analytic Jacobian.** Reuse `_groups`, cached CG buffers, and path weights; do not call autograd in production. Preserve `forward(mu, sigma) -> tuple[Tensor, Tensor]` as the compatibility wrapper selected by `cg_covariance_mode`.
- [ ] **Step 5: Route only the moment divergence into the outer objective.** At every attached block, capture tensors immediately before and after CG and append the active-family/divergence rows. For detach, capture detached pre-CG moments and perform the explicit `torch.enable_grad()` CG re-evaluation after the no-grad stack. Gate re-evaluation, list validation, stacking, diagnostics, and weighted addition on `cg_energy_weight>0`; otherwise do none of them even if the shared capture exists for M-step self-coupling. Apply the explicit token-then-layer mean and add its weighted scalar once. Do not call `hierarchical_free_energy_terms` from this path. Preserve the canonical hierarchy record; emit the per-layer values, their unweighted mean, and the separately assembled `objective_total_with_cg` diagnostic.
- [ ] **Step 6: Add construction guards.** Reject `delta_full` on diagonal/non-Gaussian families, reject nonfinite or negative `cg_energy_weight`, and require `use_cg_coupling=True` whenever `cg_energy_weight>0`. With `cg_energy_weight=0`, leave `use_cg_coupling=False` and `passthrough` untouched. Add a construction test for the positive-weight/coupling-off rejection so an empty capture can never reach `torch.stack`.
- [ ] **Step 7: Run focused CG and hierarchy tests.**

```powershell
python -m pytest tests/test_cg.py tests/test_son_irreps.py tests/test_hierarchical_probabilistic_completeness_20260712.py tests/test_model.py --junitxml=C:\tmp\vfe3-hierarchy-task4-green.xml
```

Expected: `failures="0"`, `errors="0"`, and the identity tests use `torch.equal`.

- [ ] **Step 8: Commit CG completion.**

```powershell
git add vfe3/config.py vfe3/contracts.py vfe3/model/cg_coupling.py vfe3/model/block.py vfe3/model/model.py tests/test_cg.py tests/test_hierarchical_probabilistic_completeness_20260712.py docs/2026-07-12-edits.md
git commit -m "feat: add probabilistic CG moment closure"
```

### Task 5: Make prior-bank decode family and divergence consistent

**Files:**

- Modify: `vfe3/model/prior_bank.py:50-150,628-900,1204-1415`
- Modify: `vfe3/model/model.py:205-240`
- Modify: `vfe3/config.py:1850-1880,1999-2031`
- Modify: `vfe3/train.py:420-455`
- Test: `tests/test_prior_bank.py`
- Test: `tests/test_laplace_family.py`
- Test: `tests/test_full_covariance.py`
- Test: `tests/test_hierarchical_probabilistic_completeness_20260712.py`
- Modify: `docs/2026-07-12-edits.md`

**Interfaces:**

- `PriorBank` stores `family`, `divergence_family`, and `renyi_order`. Add them to the existing constructor's defined keyword groups with compatibility defaults `family="gaussian_diagonal"`, `divergence_family="renyi"`, and `renyi_order=1.0`; every current direct constructor therefore remains source-compatible and creates exactly the old state dict. `VFEModel` passes its configured values explicitly.
- `DecodeRegistration` gains `family_consistent: bool` and resolved `covariance_kinds: FrozenSet[str]` while retaining the public `supports_full` field, `supports_chunked`, and `fused_ce` coherence. Extend `register_decode` with optional `covariance_kinds` while continuing to accept every existing `register_decode(..., supports_full=True|False)` call. When kinds are omitted, derive the old singleton rank from `supports_full`; when kinds are supplied, derive `supports_full` from membership and reject an explicitly contradictory legacy value. Keep direct `DecodeRegistration(callable, supports_full, supports_chunked, fused_ce)` construction compatible by giving its new metadata defaults and resolving them in `__post_init__`. Configuration validates membership in the resolved set rather than treating the bool as an exclusive rank bit.
- Register `family` and `family_chunked` with `covariance_kinds=frozenset({"diagonal", "full"})`; both compute logits `-D_configured(q || p_v) / tau_eff` with no `kl_max` ranking clamp. Add diagonal-only metadata to expected-likelihood modes and the appropriate singleton set to each canonical mode.
- Existing `diagonal`, `diagonal_chunked`, `full`, and `full_chunked` remain the optimized `gaussian_* + renyi(alpha=1)` implementations. The full kernels continue to score a full q against the intentionally diagonal vocabulary-prior table; Task 2 adds full SPD storage only to s/r and does not change this decode contract.

```python
def _decode_family(
    pb:      PriorBank,
    mu_q:    torch.Tensor,
    sigma_q: torch.Tensor,
    tau_eff: torch.Tensor,
) -> torch.Tensor:
    family_cls = get_family(pb.family)
    q_sigma = sigma_q.unsqueeze(-3 if family_cls.cov_kind == "full" else -2)
    q = family_cls(mu_q.unsqueeze(-2), q_sigma)
    p_sigma_diag = bounded_variance_from_log(
        pb._decode_sigma_log_table(), eps=pb.eps
    )
    p_sigma = (
        torch.diag_embed(p_sigma_diag)
        if family_cls.cov_kind == "full"
        else p_sigma_diag
    )
    p = family_cls(pb._decode_mu_table(), p_sigma)
    functional = get_functional(pb.divergence_family)
    energy = functional(q, p, alpha=pb.renyi_order,
                        kl_max=float("inf"), eps=pb.eps)
    return -energy / tau_eff
```

`family_chunked` streams vocabulary chunks through this same registered functional and the existing fused log-sum-exp/gather reduction. For a full family it materializes only `(B, N, Vc, K, K)` functional workspace inside the checkpointed chunk, using diagonal-embedded prior chunks; it never assumes a full SPD vocabulary table.

- [ ] **Step 1: Write red dense-reference tests.** For diagonal Gaussian alpha `0.5/1.0/1.5`, full Gaussian alpha `0.5/1.0/1.5`, and diagonal Laplace alpha `0.5/1.0`, compare registered logits to direct family-functional calls and show at least one noncanonical ranking differs from Gaussian KL. Construct both diagonal and full configs with the generic registration, assert covariance-kind membership accepts each, and assert the full path has no vocabulary/decode lower-triangle state key. Instantiate `PriorBank` through every existing test-style constructor without the three new arguments and require unchanged defaults/state-dict keys. Register temporary decoders with omitted, explicit-false, and explicit-true `supports_full` arguments and require the old capability values plus the corresponding resolved singleton sets; also exercise a two-rank registration and reject contradictory dual metadata.
- [ ] **Step 2: Write red chunked parity tests.** Compare dense and chunked logits/CE values and gradients for every supported family at `V=7`, `K<=4`, including gradients to the diagonal prior variances, untied decode tables, and unigram bias.
- [ ] **Step 3: Run the red decode tests.**

```powershell
python -m pytest tests/test_prior_bank.py tests/test_laplace_family.py tests/test_full_covariance.py tests/test_hierarchical_probabilistic_completeness_20260712.py --junitxml=C:\tmp\vfe3-hierarchy-task5-red.xml
```

Expected: failures show the registry lacks family-consistent decoders and `PriorBank` lacks family metadata.

- [ ] **Step 4: Implement dense and chunked registrants.** Use the configured family and functional in both paths, promote the existing diagonal vocabulary prior with `diag_embed` only when the family is full, and preserve the existing semantic temperature/unigram seams.
- [ ] **Step 5: Replace warnings with capability validation.** Canonical Gaussian/KL configs keep their existing fast modes. Any non-Gaussian or noncanonical divergence with `use_prior_bank=True` must select a registration with `family_consistent=True`; require `family_cov_kind(cfg.family) in registration.covariance_kinds` at config construction and test both generic ranks.
- [ ] **Step 6: Make `reference_decode` authoritative.** It must dispatch through configured family/divergence and remain the oracle for all fast canonical kernels.
- [ ] **Step 7: Run decode and fused-CE tests.**

```powershell
python -m pytest tests/test_prior_bank.py tests/test_use_prior_bank.py tests/test_chunked_decode.py tests/test_laplace_family.py tests/test_full_covariance.py tests/test_hierarchical_probabilistic_completeness_20260712.py --junitxml=C:\tmp\vfe3-hierarchy-task5-green.xml
```

Expected: `failures="0"` and `errors="0"`.

- [ ] **Step 8: Commit decode parity.**

```powershell
git add vfe3/model/prior_bank.py vfe3/model/model.py vfe3/config.py vfe3/train.py tests/test_prior_bank.py tests/test_laplace_family.py tests/test_full_covariance.py tests/test_hierarchical_probabilistic_completeness_20260712.py docs/2026-07-12-edits.md
git commit -m "feat: align prior-bank decode with belief geometry"
```

### Task 6: Integrate, document, and verify the completed hierarchy

**Files:**

- Modify: `README.md`
- Modify: `docs/2026-07-12-edits.md`
- Modify: `train_vfe3.py`
- Modify: `ablation.py`
- Create: `tests/hierarchy_identity_probe.py`
- Test: `tests/test_hierarchical_probabilistic_completeness_20260712.py`

**Interfaces:**

- Produces one click-run surface for the new CG covariance/energy and family-consistent decode settings, all defaulted to the old route.
- Produces a README scope statement that distinguishes the target-blind deployed hierarchy from the separately planned observation-conditioned trainer.

- [ ] **Step 1: Add an end-to-end CPU matrix.** Cover canonical diagonal/flat, full-SPD model channel, Laplace model channel with family decode, nonflat full model channel, and full-CG moment closure. Each cell uses `V<=9`, `K<6`, one forward/backward, finite gradients, optimizer coverage, save/load, and typed-term equality.
- [ ] **Step 2: Run the consolidated focused matrix.**

```powershell
$env:VFE3_TEST_DEVICE = "cpu"
python -m pytest tests/test_hierarchical_probabilistic_completeness_20260712.py tests/test_free_energy.py tests/test_hyperprior.py tests/test_gamma_coupling.py tests/test_live_s_model_channel.py tests/test_cg.py tests/test_prior_bank.py tests/test_checkpoint_resume.py --junitxml=C:\tmp\vfe3-hierarchy-focused-20260712.xml
```

Expected: `failures="0"`, `errors="0"`.

- [ ] **Step 3: Prove the pure-route identity against executable branch-base code.** Create `tests/hierarchy_identity_probe.py` as an environment-driven helper, not a production entry point. Before importing `vfe3`, it prepends `VFE3_PROBE_REPO` to `sys.path`; it writes to `VFE3_PROBE_OUT`. With deterministic CPU algorithms and one fixed `V=9, K=4, N=4, L=2, n_e_steps=2` diagonal/flat config whose new toggles are all defaulted, it records encoded belief tensors, every layer's `state_record["beliefs"]` iterate tensors, logits, loss, named gradients, model state after one optimizer step, optimizer state, and sorted state-dict keys. Patch `vfe3.model.block.e_step` inside the probe only to inject one `state_record` per layer; do not change production APIs for this test.

Run the same probe script against a detached worktree at the feature branch's merge base and against the feature worktree, then compare recursively with `torch.equal`:

```powershell
$baseSha = git merge-base HEAD origin/main
$baselinePath = "C:\tmp\vfe3-hierarchy-baseline-$($baseSha.Substring(0, 8))"
$probe = (Resolve-Path -LiteralPath 'tests/hierarchy_identity_probe.py').Path
git worktree add --detach $baselinePath $baseSha

$env:VFE3_PROBE_REPO = $baselinePath
$env:VFE3_PROBE_OUT = 'C:\tmp\vfe3-hierarchy-baseline.pt'
python $probe

$env:VFE3_PROBE_REPO = (Get-Location).Path
$env:VFE3_PROBE_OUT = 'C:\tmp\vfe3-hierarchy-feature.pt'
python $probe

$env:VFE3_BASELINE_BUNDLE = 'C:\tmp\vfe3-hierarchy-baseline.pt'
$env:VFE3_FEATURE_BUNDLE = 'C:\tmp\vfe3-hierarchy-feature.pt'
python -m pytest tests/test_hierarchical_probabilistic_completeness_20260712.py -k pure_route_bundle --junitxml=C:\tmp\vfe3-hierarchy-identity.xml

git worktree remove $baselinePath
Remove-Item -LiteralPath 'C:\tmp\vfe3-hierarchy-baseline.pt'
Remove-Item -LiteralPath 'C:\tmp\vfe3-hierarchy-feature.pt'
```

Expected: the JUnit file records zero failures/errors and every recursive tensor comparison is exact. The cleanup removes only the detached comparison worktree and the two probe-owned bundles.
- [ ] **Step 4: Run the full suite once.**

```powershell
python -m pytest --junitxml=C:\tmp\vfe3-hierarchy-full-20260712.xml
```

Expected: no new failures or errors relative to the branch baseline; report exact JUnit counts.

- [ ] **Step 5: Add and run one RTX 5090 CUDA smoke.** Add `test_hierarchy_full_covariant_cuda_smoke` to `tests/test_hierarchical_probabilistic_completeness_20260712.py`, guarded by the repository's `VFE3_TEST_DEVICE` convention. Use `family="gaussian_full"`, `s_e_step=True`, `transport_mode="regime_ii_covariant"`, `K=4`, and one optimizer step; require finite loss, SPD model covariances, finite gradients, and CUDA residency. Run it explicitly:

```powershell
$env:VFE3_TEST_DEVICE = "cuda"
python -m pytest tests/test_hierarchical_probabilistic_completeness_20260712.py -k "hierarchy_full_covariant_cuda_smoke" --junitxml=C:\tmp\vfe3-hierarchy-cuda-20260712.xml
```

Read the XML and require `tests="1"`, `skipped="0"`, `failures="0"`, and `errors="0"` on the RTX 5090; a skipped smoke is not verification.
- [ ] **Step 6: Update documentation.** Record exact supported family/transport/decode combinations, packed-storage cost, CG first-order scope, and the observation-plan prerequisite; use American English and no LaTeX spacing macros.
- [ ] **Step 7: Stage and complete git closeout.** Run `git add vfe3/free_energy.py vfe3/metrics.py vfe3/families/covariance_tables.py vfe3/model/prior_bank.py vfe3/model/model.py vfe3/model/block.py vfe3/model/cg_coupling.py vfe3/config.py vfe3/train.py train_vfe3.py ablation.py tests/test_hierarchical_probabilistic_completeness_20260712.py tests/hierarchy_identity_probe.py tests/test_checkpoint_resume.py tests/test_live_s_model_channel.py tests/test_regime_ii_covariant.py tests/test_regime_ii_link.py tests/test_cg.py tests/test_prior_bank.py tests/test_laplace_family.py tests/test_full_covariance.py README.md docs/2026-07-12-edits.md`, then inspect `git diff --cached --check`, `git status --short`, and the staged diff. Commit every intended file, push the task branch, merge it into `main`, push `main`, safely fast-forward the user's checkout only if WIP is untouched, then remove the temporary worktree and report SHAs and JUnit counts.
