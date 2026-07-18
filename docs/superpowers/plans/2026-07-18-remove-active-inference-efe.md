# Active-Inference and EFE Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the complete live active-inference and Expected-Free-Energy policy subsystem while preserving ordinary VFE behavior and load compatibility for historical non-policy checkpoints.

**Architecture:** Delete the dedicated policy modules, drivers, artifacts, and tests; collapse `VFEModel.generate()` to its ordinary sampler; and remove policy configuration and policy-only contracts. Preserve old checkpoint loading with a one-way retired-key migration tombstone that cannot restore policy behavior.

**Tech Stack:** Python 3, PyTorch, pytest, JUnit XML, Git worktrees.

## Global Constraints

Work only in `C:\tmp\V3_Transformer-remove-active-inference-efe-20260718` on `codex/remove-active-inference-efe-20260718`.

Do not modify the user's live checkout or its changes to `scaling.py`, `train_vfe3.py`, `vfe3/config.py`, or `vfe3_scaling_results/grow_K_GL10`.

Retain core VFE/free-energy code, Gaussian belief inference, ordinary autoregressive generation, and dated historical audits.

Treat JUnit as the only source for counts. The accepted baseline is 3,919 tests, 14 failures, zero errors, and 17 skips. Final verification may retain only the eleven unrelated baseline failure node IDs after EFE tests are removed.

Keep task-owned XML outside the repository and update only `docs/2026-07-18-edits.md` for the dated post-edit record.

---

### Task 1: Establish the removal contract in RED

**Files:**

- Create: `tests/test_removed_policy_surface.py`

**Interfaces:**

- Consumes: `VFE3Config`, `VFEModel`, and `migrate_serialized_config`.
- Produces: a permanent architectural guard for deleted runtime surfaces and retired checkpoint keys.

- [ ] **Step 1: Write the failing removal tests**

```python
from dataclasses import asdict, fields
from importlib.util import find_spec
from pathlib import Path

import pytest

from vfe3.config import VFE3Config, migrate_serialized_config
from vfe3.model.model import VFEModel


FORMER_POLICY_FIELDS = frozenset({
    "policy_mode",
    "policy_horizon",
    "policy_top_k",
    "policy_precision",
    "policy_preference",
    "policy_score_terms",
    "policy_sigma_ambiguity_validated",
    "policy_sigma_gate_artifact",
    "policy_ambiguity_mode",
    "policy_sigma_mc_samples",
})


def test_public_runtime_has_no_policy_surface() -> None:
    assert FORMER_POLICY_FIELDS.isdisjoint(field.name for field in fields(VFE3Config))
    assert not hasattr(VFEModel, "_policy_select")
    assert not hasattr(VFEModel, "rollout_beliefs")


@pytest.mark.parametrize("module", [
    "vfe3.inference.policy",
    "vfe3.inference.ring_task",
    "vfe3.inference.belief_cache",
    "vfe3.inference.candidate_menu",
    "vfe3.inference.sigma_gate",
])
def test_policy_modules_are_absent(module: str) -> None:
    assert find_spec(module) is None


def test_policy_drivers_and_artifacts_are_absent() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "efe_ring_experiment.py",
        "generate_efe.py",
        "sigma_gate_measure.py",
        "vfe3/inference/sigma_gate_preregistry.json",
        "vfe3_policy_results/sigma_gate/wikitext103_ed20_15k.json",
    ):
        assert not (root / relative).exists(), relative


def test_historical_policy_keys_migrate_only_as_retired_fields() -> None:
    payload = asdict(VFE3Config())
    payload.update({
        "policy_mode": "none",
        "policy_horizon": 1,
        "policy_top_k": 8,
        "policy_precision": 1.0,
        "policy_preference": "task",
        "policy_score_terms": ["risk", "ambiguity"],
        "policy_sigma_ambiguity_validated": False,
        "policy_sigma_gate_artifact": None,
        "policy_ambiguity_mode": "likelihood_entropy",
        "policy_sigma_mc_samples": 16,
    })
    with pytest.warns(UserWarning, match="retired active-inference"):
        migration = migrate_serialized_config(
            payload,
            source="historical checkpoint",
            strict_unknown=True,
        )
    assert migration.consumed_retired_keys == FORMER_POLICY_FIELDS
    assert asdict(migration.config) == asdict(VFE3Config())
```

- [ ] **Step 2: Run the test and verify RED**

Run: `python -m pytest tests/test_removed_policy_surface.py --junitxml=C:\tmp\vfe3-remove-efe-red-20260718.xml`

Expected: exit 1 because the public fields, modules, drivers, artifacts, and model methods still exist and retired-key migration does not yet warn.

- [ ] **Step 3: Inspect the XML**

Run: `[xml]$xml = Get-Content -Raw C:\tmp\vfe3-remove-efe-red-20260718.xml; $xml.testsuites.testsuite | Select-Object tests,failures,errors,skipped`

Expected: at least one failure and zero collection errors.

### Task 2: Remove the production runtime and add one-way checkpoint migration

**Files:**

- Modify: `vfe3/config.py:470-510,2039-2158,2836-2956`
- Modify: `vfe3/contracts.py:3,60-89`
- Modify: `vfe3/model/model.py:975-985,1487-1514,2175-2438`
- Modify: `vfe3/run_artifacts.py:133-145,366-413`
- Modify: `vfe3/viz/extract.py:424-432`
- Modify: `README.md:1157,1175,1210`
- Delete: `efe_ring_experiment.py`
- Delete: `generate_efe.py`
- Delete: `sigma_gate_measure.py`
- Delete: `vfe3/inference/belief_cache.py`
- Delete: `vfe3/inference/candidate_menu.py`
- Delete: `vfe3/inference/policy.py`
- Delete: `vfe3/inference/ring_task.py`
- Delete: `vfe3/inference/sigma_gate.py`
- Delete: `vfe3/inference/sigma_gate_preregistry.json`
- Delete: `vfe3_policy_results/sigma_gate/wikitext103_ed20_15k.json`

**Interfaces:**

- Consumes: current ordinary `generate()` sampler and serialized-config migration.
- Produces: one generation path and strict migration of the ten retired keys.

- [ ] **Step 1: Remove public config and validation**

Delete the ten `policy_*` dataclass fields and the complete policy validation block. Replace the phi-only retired set with the union below, use it for unknown-key filtering and `consumed_retired_keys`, and remove the obsolete `policy_score_terms` list conversion.

```python
_RETIRED_POLICY_CONFIG_FIELDS = frozenset({
    "policy_mode",
    "policy_horizon",
    "policy_top_k",
    "policy_precision",
    "policy_preference",
    "policy_score_terms",
    "policy_sigma_ambiguity_validated",
    "policy_sigma_gate_artifact",
    "policy_ambiguity_mode",
    "policy_sigma_mc_samples",
})
_RETIRED_CONFIG_FIELDS = _RETIRED_PHI_CONFIG_FIELDS | _RETIRED_POLICY_CONFIG_FIELDS
```

After `consumed_retired_keys` is computed, add:

```python
retired_policy_keys = frozenset(raw_config.keys() & _RETIRED_POLICY_CONFIG_FIELDS)
if retired_policy_keys:
    warnings.warn(
        f"serialized config from {source} contains retired active-inference field(s) "
        f"{sorted(retired_policy_keys)}; ignoring them",
        UserWarning,
        stacklevel=2,
    )
```

- [ ] **Step 2: Collapse model generation to the ordinary path**

Delete `rollout_beliefs()` and `_policy_select()`. In `generate()`, retain unconditional sampler validation and the ordinary loop body only:

```python
if not greedy:
    if not (temperature > 0.0):
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if top_k is not None and not (1 <= top_k <= self.cfg.vocab_size):
        raise ValueError(f"top_k must be in [1, vocab_size={self.cfg.vocab_size}], got {top_k}")
    if top_p is not None and not (0.0 < top_p <= 1.0):
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")

seq = token_ids
for _ in range(max_new_tokens):
    context = seq[:, -self.cfg.max_seq_len:]
    _belief, decoded = self.forward_beliefs(
        context, return_logits=True, decode_last=True, training=False)
    # retain the existing finite-logit, greedy, top-k, top-p, and multinomial body verbatim
    seq = torch.cat([seq, next_token], dim=-1)
return seq
```

- [ ] **Step 3: Remove policy-only contracts and artifact helpers**

Delete `AmbiguityEstimate`, `PolicyRollout`, `sigma_behavior_config`, and `model_behavior_fingerprint`. Remove `sigma_gate_measure.py` from `_package_code_identity()` and remove the sigma-gate paragraph from visualization extraction guidance.

- [ ] **Step 4: Delete the dedicated files and current README rows**

Use `apply_patch` delete directives for every dedicated file listed above. Remove only the three current README rows; retain dated documentation.

- [ ] **Step 5: Run the removal guard in GREEN**

Run: `python -m pytest tests/test_removed_policy_surface.py --junitxml=C:\tmp\vfe3-remove-efe-green-20260718.xml`

Expected: JUnit reports eight tests, zero failures, zero errors, and zero skips.

- [ ] **Step 6: Commit the production removal and guard**

```powershell
git add vfe3 README.md tests/test_removed_policy_surface.py efe_ring_experiment.py generate_efe.py sigma_gate_measure.py vfe3_policy_results
git diff --cached --check
git commit -m "refactor: remove active-inference EFE runtime"
```

### Task 3: Remove dedicated tests and EFE sections from mixed suites

**Files:**

- Delete: `tests/test_belief_cache.py`
- Delete: `tests/test_candidate_menu.py`
- Delete: `tests/test_efe_ring_experiment.py`
- Delete: `tests/test_efe_scorer.py`
- Delete: `tests/test_policy_registry.py`
- Delete: `tests/test_ring_task.py`
- Delete: `tests/test_sigma_gate.py`
- Modify: `tests/test_2026_07_15_cache_serialization_remediation.py`
- Modify: `tests/test_2026_07_15_driver_reliability_remediation.py`
- Modify: `tests/test_audit_runtime_state_20260713.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_final_audit_integrity_20260716.py`
- Modify: `tests/test_fixes_20260709_data.py`
- Modify: `tests/test_fixes_20260709_scripts.py`
- Modify: `tests/test_generate.py`
- Modify: `tests/test_july13_root_fixes.py`
- Modify: `tests/test_omega_direct.py`
- Modify: `tests/test_phase0_forward_beliefs.py`
- Modify: `tests/test_regime_ii_link.py`
- Modify: `tests/test_round3_artifacts.py`
- Modify: `tests/test_round3_registry_guards.py`
- Modify: `tests/test_run_artifacts.py`

**Interfaces:**

- Consumes: the post-removal production tree.
- Produces: collectable retained tests with no import of deleted subjects.

- [ ] **Step 1: Delete dedicated test modules**

Delete the seven dedicated files listed above with `apply_patch`.

- [ ] **Step 2: Remove feature-owned blocks from mixed tests**

Remove the sigma/cache tests from the July 15 cache file; ring and generation-driver tests from the July 15 driver file; `_efe_score` import/test from the runtime audit; policy list-conversion tests from `test_config.py`; `generate_efe` migration tests from the final-integrity and July 9 script files; EFE branches from `test_generate.py`; sigma tests from the July 13 and round-3 artifact files; belief-cache tests from omega-direct and Regime-II files; policy registry entries from round-3 registry guards; and sigma fingerprint tests from `test_run_artifacts.py`. Remove orphaned imports and helpers in the same patches.

Retain the ordinary generation tests in `test_generate.py`. Retain `forward_beliefs` tests in `test_phase0_forward_beliefs.py`, but remove `rollout_beliefs` assertions and rewrite the module description as an ordinary inference-seam regression.

- [ ] **Step 3: Search for deleted imports and test subjects**

Run:

```powershell
rg -n "generate_efe|efe_ring_experiment|sigma_gate_measure|vfe3\.inference\.(policy|ring_task|belief_cache|candidate_menu|sigma_gate)|_efe_score|PolicyRollout|AmbiguityEstimate" tests vfe3 --glob '*.py'
```

Expected: matches only in `tests/test_removed_policy_surface.py` and the explicit retired-key migration text; no import statement resolves a deleted module.

- [ ] **Step 4: Run affected retained suites**

Run: `python -m pytest tests/test_removed_policy_surface.py tests/test_config.py tests/test_generate.py tests/test_phase0_forward_beliefs.py tests/test_2026_07_15_cache_serialization_remediation.py tests/test_2026_07_15_driver_reliability_remediation.py tests/test_audit_runtime_state_20260713.py tests/test_final_audit_integrity_20260716.py tests/test_fixes_20260709_data.py tests/test_fixes_20260709_scripts.py tests/test_july13_root_fixes.py tests/test_omega_direct.py tests/test_regime_ii_link.py tests/test_round3_artifacts.py tests/test_round3_registry_guards.py tests/test_run_artifacts.py --junitxml=C:\tmp\vfe3-remove-efe-focused-20260718.xml`

Expected: no new failures. The pre-existing scaling stale-marker failure may remain.

- [ ] **Step 5: Commit test cleanup**

```powershell
git add tests
git diff --cached --check
git commit -m "test: remove active-inference EFE coverage"
```

### Task 4: Complete documentation, static sweeps, and full verification

**Files:**

- Modify: `docs/2026-07-18-edits.md`
- Modify as discovered only when a nonhistorical live reference remains: `README.md`, Python source, or tests.

**Interfaces:**

- Consumes: baseline XML and final XML.
- Produces: exact differential evidence and the final dated edit record.

- [ ] **Step 1: Run live-surface searches**

```powershell
rg -n -i --glob '*.py' --glob 'README.md' "active[- ]inference|expected[- ]free[- ]energy|\bEFE\b|policy_mode|policy_ambiguity|sigma_gate"
rg -n "from vfe3\.inference\.(policy|ring_task|belief_cache|candidate_menu|sigma_gate)|import generate_efe|import efe_ring_experiment|import sigma_gate_measure"
```

Expected: only `tests/test_removed_policy_surface.py`, the explicit `vfe3/config.py` compatibility tombstone, and the dated removal documentation contain former names. No live import or dispatch remains.

- [ ] **Step 2: Run the full suite**

Run: `python -m pytest --junitxml=C:\tmp\vfe3-remove-efe-final-20260718.xml`

Expected: no error and no failure node ID outside the eleven unrelated baseline failures.

- [ ] **Step 3: Compare JUnit failure node IDs programmatically**

Parse `C:\tmp\vfe3-remove-efe-baseline-20260718.xml` and `C:\tmp\vfe3-remove-efe-final-20260718.xml`; assert that the final failure set is a subset of the baseline failure set and excludes the three deleted EFE nodes.

```powershell
$efeFailures = @(
  'tests.test_generate::test_generate_sigma_mc_calls_consumer_gate_and_fails_closed',
  'tests.test_policy_registry::test_generate_efe_driver_rejects_sigma_mc_override_before_generation',
  'tests.test_sigma_gate::test_sigma_gate_spec_identity_is_known_on_tracked_tree'
)
```

- [ ] **Step 4: Update the dated edit record with exact evidence**

Record final JUnit totals, remaining failure node IDs, focused totals, static-search boundaries, and `git diff --check` status in `docs/2026-07-18-edits.md`. Do not claim a green full suite if accepted baseline failures remain.

- [ ] **Step 5: Run final static checks and commit**

```powershell
git diff --check
git status --short
git add docs/2026-07-18-edits.md README.md vfe3 tests
git diff --cached --check
git commit -m "docs: record active-inference EFE removal"
```

### Task 5: Publish, merge, and clean up

**Files:** None beyond Git metadata and task-owned temporary XML.

**Interfaces:**

- Consumes: verified branch commits.
- Produces: updated `origin/main`, safe local state, and removed temporary worktree.

- [ ] **Step 1: Invoke the finishing-a-development-branch skill and reverify**

Fetch `origin`, inspect `origin/main`, and rerun any verification invalidated by incoming changes.

- [ ] **Step 2: Push and merge**

Push `codex/remove-active-inference-efe-20260718`, merge it into `main`, and push `main` only after the differential gate remains satisfied.

- [ ] **Step 3: Handle the live checkout conservatively**

Fast-forward the user's local `main` only if Git can do so without touching its existing modified and deleted files. Otherwise leave it untouched and report the exact WIP blocker.

- [ ] **Step 4: Remove task-owned artifacts and worktree**

Delete the task-owned JUnit XML files from `C:\tmp`, remove `C:\tmp\V3_Transformer-remove-active-inference-efe-20260718`, delete the local task branch, fetch, and inspect final `origin/main` and live `git status --short`.
