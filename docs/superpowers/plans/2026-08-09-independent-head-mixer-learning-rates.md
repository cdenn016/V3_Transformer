# Independent Head-Mixer Learning Rates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the PriorBank head-evidence mixer and legacy post-belief HeadMixer independent,
backward-compatible M-step learning rates, expose both in the click-to-run configuration
dictionaries, and report their scheduled values.

**Architecture:** Add two optional absolute LR fields to `VFE3Config`; `None` resolves to
`m_p_mu_lr` only when the corresponding optimizer group is built. Keep the current optimizer group
order, roles, and weight-decay policies, label the two groups with auxiliary reporting identities,
and read those labels into stable CSV columns without changing the existing role-LR helper.

**Tech Stack:** Python 3, dataclasses, PyTorch AdamW/LambdaLR, pytest, JUnit XML, Ruff.

## Global Constraints

- `m_head_evidence_lr` controls only `PriorBank.head_evidence_logits`.
- `m_head_mixer_lr` controls only parameters owned by the legacy post-belief `HeadMixer`.
- Each field defaults to `None`; `None` inherits `m_p_mu_lr` and preserves old serialized configs.
- Explicit values are finite and nonnegative; zero remains frozen under the existing scheduler.
- PriorBank evidence retains `weight_decay=0.0`; legacy HeadMixer retains inherited
  `cfg.weight_decay`.
- Keep optimizer parameter-group count and order unchanged.
- Set both launcher dictionaries to explicit `0.001`; do not change mixer enablement or add sweeps.
- Use `C:/anaconda/python.exe` for every test that imports Torch; set `VFE3_TEST_DEVICE=cuda` for
  the final CUDA-relevant lane.
- Preserve the user's dirty live checkout; modify only the isolated worktree.

---

### Task 1: Configuration fields, validation, and serialization

**Files:**
- Modify: `vfe3/config.py:723-727,3171-3174`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: existing `VFE3Config.m_p_mu_lr: float` and serialized-config migration.
- Produces: `VFE3Config.m_head_evidence_lr: Optional[float]` and
  `VFE3Config.m_head_mixer_lr: Optional[float]`.

- [ ] **Step 1: Write failing config tests**

Add tests with the wished-for public API:

```python
def test_head_mixer_learning_rates_default_to_inheritance():
    cfg = VFE3Config()
    assert cfg.m_head_evidence_lr is None
    assert cfg.m_head_mixer_lr is None


@pytest.mark.parametrize("field", ["m_head_evidence_lr", "m_head_mixer_lr"])
@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_head_mixer_learning_rates_require_finite_nonnegative_or_none(field, value):
    with pytest.raises(ValueError, match=field):
        VFE3Config(**{field: value})


def test_head_mixer_learning_rates_roundtrip_serialized_config():
    cfg = VFE3Config(m_head_evidence_lr=0.0011, m_head_mixer_lr=0.0022)
    restored = migrate_serialized_config(asdict(cfg), source="test").config
    assert restored.m_head_evidence_lr == pytest.approx(0.0011)
    assert restored.m_head_mixer_lr == pytest.approx(0.0022)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
& 'C:/anaconda/python.exe' -m pytest -q -p no:cacheprovider `
  tests/test_config.py -k 'head_mixer_learning_rates'
```

Expected: failures because `VFE3Config` does not accept the two new keyword fields.

- [ ] **Step 3: Implement the minimal configuration contract**

Add beside the existing M-step LRs:

```python
m_head_evidence_lr: Optional[float] = None
m_head_mixer_lr: Optional[float] = None
```

Validate without coercion:

```python
for name in ("m_head_evidence_lr", "m_head_mixer_lr"):
    value = getattr(self, name)
    if value is not None and (not math.isfinite(value) or value < 0.0):
        raise ValueError(f"{name} must be finite and >= 0 or None, got {value}")
```

- [ ] **Step 4: Run focused config tests and adjacent serialization tests**

Run:

```powershell
& 'C:/anaconda/python.exe' -m pytest -q -p no:cacheprovider `
  tests/test_config.py -k 'head_mixer_learning_rates or serialized_config or migrate'
```

Expected: selected tests pass, excluding the separately documented stale default assertion if it is
selected by a broad expression.

- [ ] **Step 5: Commit Task 1**

```powershell
git add vfe3/config.py tests/test_config.py
git commit -m "feat: add independent head mixer learning rates"
```

---

### Task 2: Optimizer routing and independent scheduling

**Files:**
- Modify: `vfe3/train.py:181-207,274-278`
- Modify: `tests/test_priorbank_head_evidence.py:389-420`
- Modify: `tests/test_head_mixer.py:121-128`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: Task 1's optional config fields.
- Produces: optimizer groups labeled `lr_aux_role="head_evidence"` and
  `lr_aux_role="head_mixer"`, each with its independently resolved base LR.

- [ ] **Step 1: Write failing optimizer-routing tests**

Update/add real-model assertions:

```python
def _group_owning(optimizer, parameter):
    return next(
        group for group in optimizer.param_groups
        if any(candidate is parameter for candidate in group["params"])
    )


def test_both_mixers_receive_independent_explicit_learning_rates():
    cfg = _enabled_cfg(
        use_head_mixer=True,
        m_p_mu_lr=0.0123,
        m_head_evidence_lr=0.0011,
        m_head_mixer_lr=0.0022,
    )
    model = VFEModel(cfg)
    optimizer = build_optimizer(model, cfg)
    evidence = _group_owning(optimizer, model.prior_bank.head_evidence_logits)
    legacy = _group_owning(optimizer, model.head_mixer.mixer_delta)
    assert evidence["lr"] == pytest.approx(0.0011)
    assert legacy["lr"] == pytest.approx(0.0022)
    assert evidence["weight_decay"] == 0.0
    assert legacy["weight_decay"] == pytest.approx(cfg.weight_decay)
    assert evidence["lr_aux_role"] == "head_evidence"
    assert legacy["lr_aux_role"] == "head_mixer"


def test_both_mixer_learning_rates_inherit_mean_lr_when_none():
    cfg = _enabled_cfg(
        use_head_mixer=True,
        m_p_mu_lr=0.0123,
        m_head_evidence_lr=None,
        m_head_mixer_lr=None,
    )
    model = VFEModel(cfg)
    optimizer = build_optimizer(model, cfg)
    assert _group_owning(
        optimizer, model.prior_bank.head_evidence_logits)["lr"] == cfg.m_p_mu_lr
    assert _group_owning(
        optimizer, model.head_mixer.mixer_delta)["lr"] == cfg.m_p_mu_lr
```

Add a scheduler test constructing the real optimizer with one explicit mixer LR equal to zero and
asserting its `LambdaLR` output remains exactly zero while the other mixer follows its own base.

- [ ] **Step 2: Run optimizer/scheduler tests and verify RED**

Run:

```powershell
& 'C:/anaconda/python.exe' -m pytest -q -p no:cacheprovider `
  tests/test_priorbank_head_evidence.py tests/test_head_mixer.py tests/test_train.py `
  -k 'independent_explicit_learning_rates or inherit_mean_lr or mixer_learning_rate_scheduler'
```

Expected: explicit-LR assertions fail because both groups still use `m_p_mu_lr`; auxiliary labels are
absent.

- [ ] **Step 3: Implement minimal optimizer routing**

Resolve each LR only at its owning group:

```python
head_evidence_lr = (
    cfg.m_p_mu_lr if cfg.m_head_evidence_lr is None else cfg.m_head_evidence_lr
)
head_mixer_lr = (
    cfg.m_p_mu_lr if cfg.m_head_mixer_lr is None else cfg.m_head_mixer_lr
)
```

Use them without reordering groups:

```python
groups.append({
    "params": [pb.head_evidence_logits],
    "lr": head_evidence_lr,
    "weight_decay": 0.0,
    "role": "mu",
    "lr_aux_role": "head_evidence",
})

groups.append({
    "params": list(model.head_mixer.parameters()),
    "lr": head_mixer_lr,
    "role": "mu",
    "lr_aux_role": "head_mixer",
})
```

Update the optimizer docstring to state the separate controls and inheritance behavior.

- [ ] **Step 4: Run focused GREEN and full mixer regressions**

Run:

```powershell
$env:VFE3_TEST_DEVICE='cuda'
& 'C:/anaconda/python.exe' -m pytest -q -p no:cacheprovider `
  tests/test_priorbank_head_evidence.py tests/test_head_mixer.py tests/test_train.py
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 2**

```powershell
git add vfe3/train.py tests/test_priorbank_head_evidence.py tests/test_head_mixer.py tests/test_train.py
git commit -m "feat: route independent head mixer learning rates"
```

---

### Task 3: Scheduled-LR metrics, launchers, and documentation

**Files:**
- Modify: `vfe3/train.py:440-458,1959-1971,2428-2444`
- Modify: `train_vfe3.py:350-365,525-540`
- Modify: `ablation.py:495-510`
- Modify: `README.md:1054-1075`
- Modify: `tests/test_train.py`
- Modify: `tests/test_fixes_20260709_scripts.py`
- Modify: `tests/test_run_artifacts.py`

**Interfaces:**
- Consumes: Task 2's `lr_aux_role` optimizer metadata.
- Produces: `_learning_rates_by_aux_role(...) -> Dict[str, float]` and stable CSV keys
  `lr_head_evidence`, `lr_head_mixer`.

- [ ] **Step 1: Write failing metrics and launcher tests**

Add a helper contract test:

```python
def test_learning_rates_by_aux_role_reports_active_and_nan_for_inactive():
    groups = [
        {"lr": 0.3, "lr_report_role": "mu"},
        {"lr": 0.01, "lr_aux_role": "head_evidence"},
    ]
    result = _learning_rates_by_aux_role(groups, [0.15, 0.005])
    assert result["head_evidence"] == pytest.approx(0.005)
    assert math.isnan(result["head_mixer"])
```

Extend an artifact-backed one-step training test to assert both CSV columns always exist and that
inactive mixers emit blank/NaN values. Add script-source assertions that both launcher dictionaries
contain:

```python
m_head_evidence_lr = 0.001
m_head_mixer_lr = 0.001
```

- [ ] **Step 2: Run the new reporting/launcher tests and verify RED**

Run:

```powershell
& 'C:/anaconda/python.exe' -m pytest -q -p no:cacheprovider `
  tests/test_train.py tests/test_run_artifacts.py tests/test_fixes_20260709_scripts.py `
  -k 'aux_role or head_mixer_lr_columns or launcher_head_mixer_learning_rates'
```

Expected: helper import/column/source assertions fail because reporting and launcher fields do not
exist.

- [ ] **Step 3: Implement auxiliary LR reporting**

Add a helper without changing `_learning_rates_by_role`:

```python
def _learning_rates_by_aux_role(param_groups, lrs):
    if len(param_groups) != len(lrs):
        raise RuntimeError("optimizer parameter groups and scheduler learning rates differ in length")
    resolved = {"head_evidence": float("nan"), "head_mixer": float("nan")}
    for group, lr in zip(param_groups, lrs):
        name = group.get("lr_aux_role")
        if name is not None:
            if name not in resolved or not math.isnan(resolved[name]):
                raise RuntimeError(f"invalid or duplicate auxiliary learning-rate role: {name!r}")
            resolved[name] = float(lr)
    return resolved
```

At every metrics row, add:

```python
aux_lrs = _learning_rates_by_aux_role(optimizer.param_groups, raw_lrs)
"lr_head_evidence": aux_lrs["head_evidence"],
"lr_head_mixer": aux_lrs["head_mixer"],
```

Update both training banners to render `inherit(mu)` for `None` or the explicit value.

- [ ] **Step 4: Add launcher values and README documentation**

Place both explicit fields beside the existing M-step rates in both editable dictionaries:

```python
m_head_evidence_lr = 0.001,
m_head_mixer_lr = 0.001,
```

Document ownership, `None` inheritance, current weight-decay policies, and scheduled CSV fields in
the canonical PriorBank/HeadMixer README section.

- [ ] **Step 5: Run reporting and script GREEN tests**

Run:

```powershell
& 'C:/anaconda/python.exe' -m pytest -q -p no:cacheprovider `
  tests/test_train.py tests/test_run_artifacts.py tests/test_fixes_20260709_scripts.py `
  -k 'aux_role or head_mixer_lr_columns or launcher_head_mixer_learning_rates'
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit Task 3**

```powershell
git add vfe3/train.py train_vfe3.py ablation.py README.md `
  tests/test_train.py tests/test_run_artifacts.py tests/test_fixes_20260709_scripts.py
git commit -m "feat: report independent mixer learning rates"
```

---

### Task 4: Final verification and review

**Files:**
- Verify: all files changed by Tasks 1-3
- Create: JUnit XML under the external visualization artifact directory only

**Interfaces:**
- Consumes: the complete implementation.
- Produces: machine-readable focused/regression evidence and a review-ready branch.

- [ ] **Step 1: Run static checks**

Run:

```powershell
git diff --check origin/main...HEAD
& 'C:/anaconda/python.exe' -m ruff check `
  vfe3/config.py vfe3/train.py tests/test_config.py tests/test_priorbank_head_evidence.py `
  tests/test_head_mixer.py tests/test_train.py tests/test_run_artifacts.py `
  tests/test_fixes_20260709_scripts.py
```

Expected: exit 0.

- [ ] **Step 2: Run focused CUDA-capable JUnit verification**

Run with a fresh external basetemp/JUnit path:

```powershell
$env:VFE3_TEST_DEVICE='cuda'
$env:MKL_THREADING_LAYER='SEQUENTIAL'
& 'C:/anaconda/python.exe' -m pytest -q -p no:cacheprovider `
  --basetemp='C:/Users/chris and christine/.codex/visualizations/2026/08/09/019fe412-7ed2-7980-813f-64c384b363e6/pytest-independent-head-mixer-lrs-focused' `
  --junitxml='C:/Users/chris and christine/.codex/visualizations/2026/08/09/019fe412-7ed2-7980-813f-64c384b363e6/independent-head-mixer-lrs-focused.xml' `
  tests/test_config.py tests/test_priorbank_head_evidence.py tests/test_head_mixer.py `
  tests/test_train.py tests/test_run_artifacts.py tests/test_fixes_20260709_scripts.py `
  -k 'not test_config_model_defaults'
```

Expected: zero failures/errors; parse exact totals from JUnit rather than terminal progress.

- [ ] **Step 3: Run affected checkpoint/scheduler/reporting regressions**

Run with a second fresh JUnit path:

```powershell
$env:MKL_THREADING_LAYER='SEQUENTIAL'
& 'C:/anaconda/python.exe' -m pytest -q -p no:cacheprovider `
  --basetemp='C:/Users/chris and christine/.codex/visualizations/2026/08/09/019fe412-7ed2-7980-813f-64c384b363e6/pytest-independent-head-mixer-lrs-regression' `
  --junitxml='C:/Users/chris and christine/.codex/visualizations/2026/08/09/019fe412-7ed2-7980-813f-64c384b363e6/independent-head-mixer-lrs-regression.xml' `
  tests/test_checkpoint_resume.py tests/test_grad_accum.py tests/test_reporting_additions.py `
  tests/test_experiment_metrics.py
```

Expected: zero failures/errors; any established environment skip is reported from JUnit.

- [ ] **Step 4: Self-review requirements and diff**

Confirm mechanically:

```powershell
git status --short
git diff --stat origin/main...HEAD
git diff origin/main...HEAD -- vfe3/config.py vfe3/train.py train_vfe3.py ablation.py README.md
```

Check every design requirement, parameter ownership, weight-decay preservation, fallback behavior,
launcher value, CSV column, and test result. Correct any defect with a new RED/GREEN cycle.

- [ ] **Step 5: Commit any verification-only documentation correction**

If Task 4 changes tracked documentation, stage only that exact file and commit it. Otherwise leave
the tested implementation commits unchanged.
