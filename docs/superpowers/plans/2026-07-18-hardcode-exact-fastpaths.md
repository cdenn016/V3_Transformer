# Hardcode Exact Fast Paths Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove three exactness-preserving performance choices from public configuration and make production request compact phi transport, pairwise diagonal-KL statistics reuse, and per-head mean transport whenever their existing internal eligibility guards permit them.

**Architecture:** `VFE3Config` and click-to-run dictionaries lose the three keys. Production call sites forward literal `True` values, while compact phi selection continues to pass through the existing route/layout predicate. Low-level Boolean parameters remain available only for dense-versus-optimized numerical tests and unsupported-route fallbacks.

**Tech Stack:** Python 3.11+, PyTorch, pytest, JUnit XML, Git worktrees.

## Global Constraints

Work only in `C:\tmp\V3_Transformer-hardcode-fastpaths-20260718` on `codex/hardcode-fastpaths-20260718`.

Do not modify the user's live checkout or its changes to `scaling.py`, `train_vfe3.py`, and `vfe3_scaling_results/grow_K_GL10`.

Do not add migration tombstones for `compact_phi_block_transport`, `reuse_pairwise_kl_stats`, or `transport_mean_per_head`; historical serialized configurations are outside the compatibility boundary approved for this change.

Retain the internal Boolean parameters in `vfe3/inference/e_step.py`, `vfe3/gradients/kernels.py`, and transport helpers. Their false-valued paths remain differential test oracles and automatic fallbacks, not public configuration.

Do not broaden compact transport eligibility. Phi parameterization, flat transport, reflections off, and `block_head_row_major` layout remain mandatory. Preserve dtype, family, divergence, and route guards for pairwise-statistics reuse and preserve all existing transport fallbacks.

Use machine-readable JUnit XML for counts. The accepted untouched baseline is 3,659 tests, 10 failures, zero errors, and 16 skips. Final failures must be a subset of the ten node IDs in the approved design, with zero new failures and zero errors.

Keep task-owned XML under `C:\tmp`, remove it after final verification, and update only `docs/2026-07-18-edits.md` for the dated post-edit record.

---

### Task 1: Establish the hardcoded production contract in RED

**Files:**

- Create: `tests/test_hardcoded_exact_fastpaths_20260718.py`

**Interfaces:**

- Consumes: `VFE3Config`, `e_step_shared_kwargs`, `VFEModel`, and production source modules.
- Produces: an architectural guard that separates removed public controls from retained internal oracle parameters.

- [ ] **Step 1: Write the failing architectural guard**

Create the file with the following complete test module:

```python
"""Architectural contract for exact fast paths that are mandatory in production."""

import ast
from dataclasses import fields
from pathlib import Path

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.model.block import e_step_shared_kwargs
from vfe3.model.model import VFEModel


ROOT = Path(__file__).resolve().parents[1]
FORMER_CONFIG_FIELDS = frozenset({
    "compact_phi_block_transport",
    "reuse_pairwise_kl_stats",
    "transport_mean_per_head",
})
PRODUCTION_MODULES = (
    "vfe3/model/block.py",
    "vfe3/model/model.py",
    "vfe3/viz/extract.py",
)


def _keyword_values(relative: str, keyword_name: str) -> list[ast.expr]:
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    return [
        keyword.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == keyword_name
    ]


def test_public_configuration_and_drivers_have_no_fastpath_controls() -> None:
    assert FORMER_CONFIG_FIELDS.isdisjoint(field.name for field in fields(VFE3Config))
    for relative in ("vfe3/config.py", "train_vfe3.py", "scaling.py", "ablation.py"):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert all(name not in source for name in FORMER_CONFIG_FIELDS), relative


def test_production_does_not_read_removed_fastpath_attributes() -> None:
    for relative in PRODUCTION_MODULES:
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        reads = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr in FORMER_CONFIG_FIELDS
        }
        assert reads == set(), relative


@pytest.mark.parametrize(
    ("relative", "keyword_name"),
    [
        ("vfe3/model/block.py", "reuse_pairwise_kl_stats"),
        ("vfe3/model/block.py", "transport_mean_per_head"),
        ("vfe3/model/model.py", "reuse_pairwise_kl_stats"),
        ("vfe3/model/model.py", "transport_mean_per_head"),
        ("vfe3/viz/extract.py", "transport_mean_per_head"),
    ],
)
def test_production_fastpath_requests_are_literal_true(
    relative:     str,
    keyword_name: str,
) -> None:
    values = _keyword_values(relative, keyword_name)
    assert values, (relative, keyword_name)
    assert all(isinstance(value, ast.Constant) and value.value is True for value in values)


def test_shared_e_step_kwargs_always_request_pairwise_reuse() -> None:
    kwargs = e_step_shared_kwargs(VFE3Config(), torch.device("cpu"))
    assert kwargs["reuse_pairwise_kl_stats"] is True


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, True),
        ({"gauge_parameterization": "omega_direct", "pos_phi": "none"}, False),
        ({"transport_mode": "regime_ii"}, False),
        ({"phi_reflection": "init_seed"}, False),
        ({"n_heads": 1}, False),
    ],
)
def test_compact_phi_route_is_automatic(
    overrides: dict[str, object],
    expected:  bool,
) -> None:
    values: dict[str, object] = {
        "vocab_size": 9,
        "embed_dim": 4,
        "n_heads": 2,
        "max_seq_len": 3,
        "n_layers": 1,
        "n_e_steps": 1,
    }
    values.update(overrides)
    model = VFEModel(VFE3Config(**values)).eval()
    assert model._compact_phi_blocks_enabled() is expected
```

- [ ] **Step 2: Run the guard and verify RED**

Run:

`python -m pytest tests/test_hardcoded_exact_fastpaths_20260718.py --junitxml=C:\tmp\vfe3-hardcoded-fastpaths-red-20260718.xml`

Expected: exit 1 because the dataclass fields and driver keys still exist, production reads configuration attributes, shared E-step arguments inherit the false default, and the compact predicate is disabled by default.

- [ ] **Step 3: Inspect the RED XML**

Run:

`[xml]$xml = Get-Content -Raw C:\tmp\vfe3-hardcoded-fastpaths-red-20260718.xml; $xml.testsuites.testsuite | Select-Object tests,failures,errors,skipped`

Expected: one or more failures and zero collection errors.

### Task 2: Remove the public configuration and driver surface

**Files:**

- Modify: `vfe3/config.py:681-695`
- Modify: `train_vfe3.py:402-404`
- Modify: `scaling.py:433-435`
- Modify: `ablation.py:521-523`

**Interfaces:**

- Removes: the three `VFE3Config` constructor fields and executable dictionary keys.
- Preserves: all unrelated performance controls, strict unknown-key behavior, and the current click-to-run configuration style.

- [ ] **Step 1: Delete the three dataclass fields and their obsolete opt-in comments**

Remove the complete `transport_mean_per_head`, `compact_phi_block_transport`, and `reuse_pairwise_kl_stats` declarations. The next config section should begin directly with the still-configurable fp64 island controls:

```python
    # fp64 island keying for stable_matrix_exp_pair. 'dim' (default): upcast when the block dim
    # >= its dim_threshold (the long-standing rule). 'norm': upcast only when the clamped block
    # Frobenius norm exceeds exp_fp64_norm_threshold -- the conditioning argument is a norm
    # argument, not a dimension argument; small-norm blocks are fp32-accurate at any dim.
    exp_fp64_mode:             str   = "dim"          # "dim" | "norm"
    exp_fp64_norm_threshold:   float = 5.0            # 'norm' mode: upcast when max ||M||_F >= this
```

Do not add the names to any retired-field set or migration function.

- [ ] **Step 2: Delete the three keys from every click-to-run dictionary**

In `train_vfe3.py`, `scaling.py`, and `ablation.py`, remove only these entries:

```python
    compact_phi_block_transport = True,
    reuse_pairwise_kl_stats     = True,
    transport_mean_per_head     = True,
```

Preserve the neighboring configuration values and the user's established alignment style.

- [ ] **Step 3: Run the public-surface portion of the guard**

Run:

`python -m pytest tests/test_hardcoded_exact_fastpaths_20260718.py -k "public_configuration" --junitxml=C:\tmp\vfe3-hardcoded-fastpaths-surface-20260718.xml`

Expected: one test passes. The complete guard remains RED until production forwarding is changed.

### Task 3: Make production request the exact fast paths unconditionally

**Files:**

- Modify: `vfe3/model/block.py:32-47,67,114-133`
- Modify: `vfe3/model/model.py:714-723,902-906,1065-1068,1246-1248,1287,1879-1880,2483-2489`
- Modify: `vfe3/viz/extract.py:304-315`
- Modify: `vfe3/geometry/transport.py:2029`

**Interfaces:**

- Produces: literal production requests for pairwise reuse and per-head mean transport.
- Preserves: `_compact_phi_blocks_enabled()` as the authoritative automatic eligibility predicate.

- [ ] **Step 1: Hardcode shared block forwarding**

In `e_step_shared_kwargs`, make reuse unconditional:

```python
        reuse_pairwise_kl_stats=True,
```

In `vfe_block`, replace the compact predicate with:

```python
    compact_phi_blocks = (
        gauge_parameterization == "phi"
        and cfg.transport_mode == "flat"
        and cfg.phi_reflection == "off"
        and group.phi_coordinate_layout == "block_head_row_major"
    )

```

Then replace the existing `e_step` keyword `transport_mean_per_head=cfg.transport_mean_per_head`
with `transport_mean_per_head=True`; retain the adjacent
`compact_phi_block_transport=compact_phi_blocks` keyword verbatim.

Rewrite the nearby docstring so it describes `transport_mean_per_head` as an explicit internal E-step parameter and `reuse_pairwise_kl_stats` as a mandatory shared argument, without claiming these behaviors default off.

- [ ] **Step 2: Make the model eligibility predicate automatic**

Replace `_compact_phi_blocks_enabled()` with:

```python
    def _compact_phi_blocks_enabled(self) -> bool:
        r"""Whether this model is on the canonical route supported by packed phi transport."""
        cfg = self.cfg
        return (
            cfg.gauge_parameterization == "phi"
            and cfg.transport_mode == "flat"
            and cfg.phi_reflection == "off"
            and self.group.phi_coordinate_layout == "block_head_row_major"
        )
```

- [ ] **Step 3: Replace every production config read with a literal request**

In `vfe3/model/model.py`, preserve each surrounding call and replace only the removed-field reads:

```python
            transport_mean_per_head=True,
            compact_phi_block_transport=self._compact_phi_blocks_enabled(),
            reuse_pairwise_kl_stats=True,
```

Use `transport_mean_per_head=True` in the shared prebuilt transport, Metropolis objective replay, gamma transport, and compact diagnostic transport calls. Keep `compact_phi_block_transport=self._compact_phi_blocks_enabled()` wherever route gating is required.

In `vfe3/viz/extract.py`, retain the model predicate for compact layout and use the literal per-head request:

```python
        compact_phi_block_transport=model._compact_phi_blocks_enabled(),
        transport_mean_per_head=True,
```

Update the objective and transport docstrings to describe per-head contraction as active production numerics rather than a configuration toggle.

- [ ] **Step 4: Run the complete architectural guard in GREEN**

Run:

`python -m pytest tests/test_hardcoded_exact_fastpaths_20260718.py --junitxml=C:\tmp\vfe3-hardcoded-fastpaths-green-20260718.xml`

Expected: all tests in the new module pass with zero failures and zero errors.

- [ ] **Step 5: Commit the architectural contract and production change**

Run:

`git add vfe3/config.py vfe3/model/block.py vfe3/model/model.py vfe3/viz/extract.py vfe3/geometry/transport.py train_vfe3.py scaling.py ablation.py tests/test_hardcoded_exact_fastpaths_20260718.py`

`git diff --cached --check`

`git commit -m "refactor: hardcode exact production fast paths"`

### Task 4: Retarget existing tests from config toggles to internal numerical oracles

**Files:**

- Modify: `tests/test_p1_compact_phi_block_transport_20260711.py:39-41,523-556,605-643,702-776`
- Modify: `tests/test_p3_pairwise_stats_reuse_20260711.py:26-128`
- Modify: `tests/test_2026_07_15_performance_remediation.py:90-165`
- Modify: `tests/test_amp.py:112-126`
- Modify: `tests/test_objective_state_transitions_20260715.py:114`
- Modify: `tests/test_phi_reflection_objective_parity_20260712.py:492`
- Modify: `tests/test_tier12_transport.py:1-3`

**Interfaces:**

- Preserves: dense/optimized output, gradient, VJP, dtype, fallback, diagnostics, and objective-parity checks.
- Removes: tests whose only contract was that production could disable an exact fast path through configuration.

- [ ] **Step 1: Rewrite compact-phi model tests around automatic eligibility**

Delete `test_compact_phi_block_transport_defaults_off_and_accepts_opt_in`.

Remove the `compact_phi_block_transport` entries from the model-construction dictionaries in `test_model_threads_compact_toggle_into_live_phi_bch_retraction` and `test_model_gates_packed_bch_to_eligible_transport_route`; rename the first test to use `automatic_route` rather than `toggle` language. Its eligible/nonflat/reflection expectations remain unchanged.

Replace `_tiny_two_channel_config(compact: bool)` with a no-argument production config:

```python
def _tiny_two_channel_config() -> VFE3Config:
    return VFE3Config(
        vocab_size=9,
        embed_dim=4,
        n_heads=2,
        max_seq_len=5,
        n_layers=1,
        n_e_steps=1,
        e_phi_lr=0.0,
        use_prior_bank=True,
        prior_source="model_channel",
        s_e_step=True,
        lambda_h=1.0,
        lambda_gamma=0.75,
        share_refine_s_transport=True,
    )
```

The gamma test now constructs one production model and asserts its observed factored transport has `mean_per_head is True`. For the end-to-end dense comparison, construct two identical models, force only the reference model's route predicate off, then compare it with the unmodified production model:

```python
    dense_model = VFEModel(_tiny_two_channel_config()).eval()
    compact_model = VFEModel(_tiny_two_channel_config()).eval()
    compact_model.load_state_dict(dense_model.state_dict())
    monkeypatch.setattr(dense_model, "_compact_phi_blocks_enabled", lambda: False)
```

Keep all direct calls that pass `compact_phi_block_transport=False` or `True` into `build_belief_transport`, `e_step`, or other low-level helpers. Those are the approved numerical oracles, not public model configuration.

- [ ] **Step 2: Rewrite pairwise-reuse routing tests**

Remove `reuse_pairwise_kl_stats` from `_tiny_two_channel_config` and delete these obsolete tests:

```text
test_p3_toggle_defaults_off_and_is_opt_in
test_p3_default_and_explicit_false_are_bit_identical
test_p3_disabled_route_cannot_reach_future_helper
```

Rename `test_p3_enabled_forwards_true_to_q_and_s_consumers` to `test_p3_production_forwards_true_to_q_and_s_consumers`, construct the model with only `e_step_update`, and retain the exact assertion:

```python
    model = _build_model(_tiny_two_channel_config(e_step_update=e_step_update))
    # _refine_s runs first and the belief q iteration follows.
    assert seen == [(True, 0.75), (True, 1.0)]
```

Keep the later direct kernel tests that parametrically pass false and true values; they prove fallback and parity below the public configuration boundary.

- [ ] **Step 3: Preserve dense diagnostics oracles without restoring config**

In each of the three compact diagnostics/report tests, replace the two toggle-bearing constructors with:

```python
    dense_model = VFEModel(_tiny_config()).eval()
    compact_model = VFEModel(_tiny_config()).eval()
    compact_model.load_state_dict(dense_model.state_dict())
    monkeypatch.setattr(dense_model, "_compact_phi_blocks_enabled", lambda: False)
```

Do not patch the production model. Existing output, mapping, memory, and factored-transport assertions remain unchanged.

- [ ] **Step 4: Remove residual config reads from supporting tests**

Remove `compact_phi_block_transport=True` and `transport_mean_per_head=True` from the `VFE3Config` constructor in `tests/test_amp.py`; the hardcoded production route now supplies both behaviors.

Replace objective-replay arguments that read `cfg.transport_mean_per_head` with `transport_mean_per_head=True` in `tests/test_objective_state_transitions_20260715.py` and `tests/test_phi_reflection_objective_parity_20260712.py` so their reference calculations match production.

Rewrite the `tests/test_tier12_transport.py` module docstring to say that production requests per-head mean contraction while the low-level false-valued path remains a parity oracle. Keep its direct true/false helper calls unchanged. Keep `tests/test_omega_direct.py` and `tests/test_fixes_20260709_scripts.py` unchanged because they exercise retained low-level interfaces.

- [ ] **Step 5: Run the focused exactness matrix**

Run:

```powershell
python -m pytest `
  tests/test_hardcoded_exact_fastpaths_20260718.py `
  tests/test_p1_compact_phi_block_transport_20260711.py `
  tests/test_p3_pairwise_stats_reuse_20260711.py `
  tests/test_2026_07_15_performance_remediation.py `
  tests/test_amp.py `
  tests/test_omega_direct.py `
  tests/test_objective_state_transitions_20260715.py `
  tests/test_phi_reflection_objective_parity_20260712.py `
  tests/test_tier12_transport.py `
  --junitxml=C:\tmp\vfe3-hardcoded-fastpaths-focused-20260718.xml
```

Expected: exit 0, zero failures, and zero errors. Read the exact test/pass/skip counts from the XML.

- [ ] **Step 6: Commit the retargeted numerical tests**

Run:

`git add tests/test_p1_compact_phi_block_transport_20260711.py tests/test_p3_pairwise_stats_reuse_20260711.py tests/test_2026_07_15_performance_remediation.py tests/test_amp.py tests/test_objective_state_transitions_20260715.py tests/test_phi_reflection_objective_parity_20260712.py tests/test_tier12_transport.py`

`git diff --cached --check`

`git commit -m "test: pin hardcoded exact fast paths"`

### Task 5: Verify the boundary, run the full suite, and record the edit

**Files:**

- Modify: `docs/2026-07-18-edits.md`

**Interfaces:**

- Produces: static proof of the public/internal boundary, focused JUnit evidence, full-suite differential evidence, and the required dated record.

- [ ] **Step 1: Prove the public controls and configuration reads are absent**

Run:

```powershell
rg -n "compact_phi_block_transport|reuse_pairwise_kl_stats|transport_mean_per_head" `
  vfe3/config.py train_vfe3.py scaling.py ablation.py
```

Expected: exit 1 with no matches.

Run:

```powershell
rg -n "(?:self\.)?cfg\.(compact_phi_block_transport|reuse_pairwise_kl_stats|transport_mean_per_head)" `
  vfe3/model vfe3/viz
```

Expected: exit 1 with no matches.

Run:

```powershell
rg -n "compact_phi_block_transport|reuse_pairwise_kl_stats|transport_mean_per_head" `
  vfe3/inference/e_step.py vfe3/gradients/kernels.py tests
```

Expected: matches remain only in the new architectural guard, direct low-level forwarding, and numerical oracle tests. Inspect every match; no `VFE3Config(...)` construction or removed-field config-attribute read may remain.

- [ ] **Step 2: Run syntax and diff checks**

Run:

`python -m compileall -q vfe3 train_vfe3.py scaling.py ablation.py`

Expected: exit 0.

Run:

`git diff --check`

Expected: exit 0.

- [ ] **Step 3: Run the full suite to JUnit**

Run:

`python -m pytest --junitxml=C:\tmp\vfe3-hardcoded-fastpaths-final-20260718.xml`

Expected: pytest may exit 1 only for accepted baseline failures; JUnit must contain zero errors and no failure node outside the approved ten-node set.

- [ ] **Step 4: Compare exact failure identities with the accepted baseline**

Use PowerShell to extract `$($_.classname)::$($_.name)` for every JUnit testcase with a failure or error. Compare that set with the ten node IDs recorded in `docs/superpowers/specs/2026-07-18-hardcode-exact-fastpaths-design.md`.

Acceptance:

```text
errors == 0
final_failure_ids - accepted_baseline_ids == empty set
```

Record exact tests, failures, errors, skips, passes, and elapsed time from the final XML. Do not infer counts from console dots or the earlier baseline.

- [ ] **Step 5: Update the dated post-edit document**

Append one section titled `Exact fast paths hardcoded in production`. State
that the three names are no longer public configuration, production requests
the exact optimizations unconditionally, the established route/layout/dtype/
family guards retain the fallbacks, and low-level false-valued parameters
remain only as numerical parity oracles.

In the same section, transcribe the focused and full tests, failures, errors,
skips, passes, and elapsed times from their JUnit files. State the exact number
of new failure node IDs from the accepted-baseline comparison and the exit
statuses of the static source, compile, and diff checks.

- [ ] **Step 6: Commit the verified edit record**

Run:

`git add docs/2026-07-18-edits.md`

`git diff --cached --check`

`git commit -m "docs: record hardcoded fastpath verification"`

### Task 6: Publish, fast-forward remote main, safely fast-forward locally, and clean up

**Files:**

- No source edits expected.

**Interfaces:**

- Produces: pushed task branch, fast-forwarded `origin/main`, safely updated live checkout, and removed temporary worktree/branch.

- [ ] **Step 1: Inspect final branch state**

Run:

`git status --short`

`git log --oneline --decorate origin/main..HEAD`

Expected: clean task worktree and only the intended design, plan, implementation, test, and edit-record commits.

- [ ] **Step 2: Push the task branch**

Run:

`git push -u origin codex/hardcode-fastpaths-20260718`

- [ ] **Step 3: Refresh and prove fast-forward mergeability**

Run:

`git fetch origin`

`git log -3 --oneline origin/main`

`git merge-base --is-ancestor origin/main codex/hardcode-fastpaths-20260718`

Expected: the ancestry check exits 0. If `origin/main` advanced incompatibly, stop and rebase or merge only after inspecting the exact remote commits and conflicts.

- [ ] **Step 4: Fast-forward remote main from the verified task branch**

Run:

`git push origin codex/hardcode-fastpaths-20260718:main`

Expected: the remote accepts a fast-forward update. Fetch immediately afterward and verify that `origin/main` equals the task branch tip.

- [ ] **Step 5: Safely fast-forward the user's live checkout**

In `C:\Users\chris and christine\Desktop\V3_Transformer`, run `git status --short` and inspect overlap with the task diff. The known user changes in `scaling.py` and `train_vfe3.py` are at unrelated lines, but Git is authoritative. Fast-forward local `main` only if Git can preserve those modifications without overwrite or conflict. Otherwise leave the live checkout untouched and report the blocker.

- [ ] **Step 6: Verify remote state and remove task-owned artifacts**

Fetch and inspect `origin/main`, record its SHA, remove the task-owned XML files under `C:\tmp`, remove `C:\tmp\V3_Transformer-hardcode-fastpaths-20260718`, and delete the local `codex/hardcode-fastpaths-20260718` branch after confirming it is merged.

- [ ] **Step 7: Report the completed lifecycle**

Report the task branch, implementation commit SHA, pushed remote branch, resulting `origin/main` SHA, focused and full JUnit results, failure-set comparison, live-checkout fast-forward result, worktree removal, local branch deletion, and final `git status --short` for the live checkout with the remaining files identified as user-owned.
