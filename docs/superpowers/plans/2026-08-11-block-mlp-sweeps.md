# Block MLP Sweeps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add five independent, inactive block-MLP ablation sweeps while preserving every current local setting.

**Architecture:** Extend the existing declarative `SWEEPS` registry only. Each hyperparameter sweep forces `use_block_mlp=True`, emits literal one-field overrides, and stays outside `SWEEP_ORDER`.

**Tech Stack:** Python, pytest, the existing `ablation.make_run_overrides` and `ablation.validate_sweeps` APIs.

## Global Constraints

- Preserve current `ablation.py`, `train_vfe3.py`, and `vfe3/config.py` contents except for the approved sweep declarations.
- Preserve `zzzzz.py` byte-for-byte and leave it untracked.
- Do not add any MLP sweep to `SWEEP_ORDER`.
- Use CPU-only verification; do not interact with active GPU processes.

---

### Task 1: Declare and verify the MLP sweep family

**Files:**
- Modify: `ablation.py`
- Modify: `tests/test_block_mlp_launchers.py`

**Interfaces:**
- Consumes: `ablation.SWEEPS`, `ablation.SWEEP_ORDER`, `ablation.make_run_overrides(name)`, and `ablation.validate_sweeps(names)`.
- Produces: the sweep names `block_mlp`, `block_mlp_expansion`, `block_mlp_activation`, `block_mlp_dropout`, and `m_block_mlp_lr`.

- [ ] **Step 1: Write the failing behavior test**

Add a test that calls `make_run_overrides` for the four missing hyperparameter sweeps and compares their emitted labels and override dictionaries against literal expected values. Validate all five sweep names and assert none appears in `SWEEP_ORDER`.

- [ ] **Step 2: Run the test to verify RED**

Run: `C:/Python314/python.exe -m pytest tests/test_block_mlp_launchers.py::test_block_mlp_hyperparameter_sweeps_are_complete_and_inactive -q`

Expected: failure because `block_mlp_expansion` is absent from `SWEEPS`.

- [ ] **Step 3: Add the minimal declarations**

Add the four `param`-style entries with the exact approved value lists and `requires={"use_block_mlp": True}`. Leave `SWEEP_ORDER` unchanged.

- [ ] **Step 4: Run focused GREEN verification**

Run: `C:/Python314/python.exe -m pytest tests/test_block_mlp_launchers.py::test_block_mlp_hyperparameter_sweeps_are_complete_and_inactive tests/test_block_mlp_launchers.py::test_block_mlp_ablation_is_opt_in_and_arms_differ_only_by_enable_toggle -q`

Expected: two passing tests.

- [ ] **Step 5: Verify syntax and live configuration construction**

Parse `ablation.py`, `train_vfe3.py`, and `vfe3/config.py`; instantiate `VFE3Config` from both launcher dictionaries; validate the five sweep names.

- [ ] **Step 6: Commit intentionally**

Stage the approved docs, test, `ablation.py`, `train_vfe3.py`, and `vfe3/config.py`; exclude `zzzzz.py`. Commit with a message describing the MLP sweep family and preserved local settings.
