# Reporting Registry Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make metric and figure registration affect production reporting, and route the four implemented but orphaned sweep plots from persisted artifacts to emitted files.

**Architecture:** Keep numerical kernels and plot functions in their current modules. Add small declarative report specifications that name a registered callable, its persisted-artifact adapter, and its output filename; one shared dispatcher resolves the registry and owns skip/error/figure-close behavior. Existing report drivers migrate to the dispatcher in bounded groups so default outputs remain identical.

**Tech Stack:** Python, dataclasses, pathlib, JSON/CSV adapters, matplotlib, pytest.

## Global Constraints

- A new metric or figure becomes selectable by registration plus a report specification, without editing a callable switch statement.
- Preserve every current filename and default report output during migration.
- Adapters consume persisted artifacts; they do not rerun model inference.
- Plot failures remain isolated and must not leak matplotlib figures.
- Keep figure-adapter tests data-only. The metric-dispatch regression may construct one `K=2`, one-layer CPU model; no production-size model or inference replay is required.
- Update `docs/2026-07-12-edits.md` as the single dated edit note.

---

### Task 1: Define declarative registered-report specifications

**Files:**

- Create: `vfe3/viz/specs.py`
- Modify: `vfe3/viz/report.py:220-240`
- Test: `tests/test_report.py`

**Interfaces:**

- Produces: `FigureSpec(name: str, output_name: str, adapter: Callable[[Mapping[str, object]], Optional[Mapping[str, object]]])`.
- Produces: `emit_registered_figures(specs, context, output_dir) -> List[Path]`.
- Consumes: `vfe3.viz.figures.get_figure(name)`.

- [ ] **Step 1: Write failing dispatch and cleanup tests**

```python
def test_emit_registered_figure_uses_registry(tmp_path, monkeypatch):
    seen = {}

    @register_figure("report_probe", override=True)
    def probe(*, value, path=None):
        seen["value"] = value
        return _one_axis_figure(path)

    spec = FigureSpec("report_probe", "probe.png", lambda ctx: {"value": ctx["value"]})
    written = emit_registered_figures([spec], {"value": 7}, tmp_path)
    assert [p.name for p in written] == ["probe.png"]
    assert seen == {"value": 7}
```

Add sibling tests for an adapter returning `None`, a builder raising after creating a figure, a builder returning without writing, a missing registry key, and duplicate output names. Precreate every failure/skip target with sentinel bytes and require those bytes to remain unchanged; a successful dispatch must atomically replace the sentinel. Require no same-directory temporary file after either success or failure.

- [ ] **Step 2: Run the tests and confirm the API is absent**

Run: `python -m pytest tests/test_report.py -k registered --junitxml=C:\tmp\vfe3-report-registry-red.xml`

Expected: import or attribute failure for `FigureSpec` and `emit_registered_figures`.

- [ ] **Step 3: Implement the minimal immutable spec and dispatcher**

```python
@dataclass(frozen=True)
class FigureSpec:
    name:        str
    output_name: str
    adapter:     Callable[[Mapping[str, object]], Optional[Mapping[str, object]]]
```

The dispatcher implementation is bounded and exact:

```python
import logging
import os
from uuid import uuid4

import matplotlib.pyplot as plt

from vfe3.viz.figures import get_figure

logger = logging.getLogger(__name__)


def emit_registered_figures(
    specs:      Sequence[FigureSpec],
    context:    Mapping[str, object],
    output_dir: Path,
) -> List[Path]:
    names = [spec.output_name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("registered figure output names must be unique")
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for spec in specs:
        before = set(plt.get_fignums())
        fig = None
        tmp_target = None
        try:
            kwargs = spec.adapter(context)
            if kwargs is None:
                continue
            target = output_dir / spec.output_name
            tmp_target = output_dir / (
                f".{target.stem}.{uuid4().hex}.tmp{target.suffix}"
            )
            fig = get_figure(spec.name)(**dict(kwargs), path=str(tmp_target))
            if not tmp_target.is_file() or tmp_target.stat().st_size == 0:
                raise RuntimeError(
                    f"figure {spec.name!r} did not write its temporary output"
                )
            os.replace(tmp_target, target)
            written.append(target)
        except Exception as exc:
            logger.warning("registered figure %s skipped: %s", spec.name, exc)
        finally:
            if tmp_target is not None:
                tmp_target.unlink(missing_ok=True)
            if fig is not None:
                plt.close(fig)
            for number in set(plt.get_fignums()) - before:
                plt.close(number)
    return written
```

The missing-key, ImportError, TypeError, builder-failure, and builder-no-write tests assert one warning, no returned path, unchanged preexisting target bytes, no temporary file, and `set(plt.get_fignums())` equal to its pre-call value. Adapter-`None` is an intentional skip with the same target-preservation rule and no warning. Duplicate output names raise before any adapter is invoked. Each builder writes only to a unique same-directory temporary path; `os.replace` publishes it after the write check, so no old target can masquerade as new output and failures cannot corrupt an existing figure. The `finally` block removes task-owned temporary output and closes every figure opened by the builder.

- [ ] **Step 4: Run the focused tests**

Run: `python -m pytest tests/test_report.py -k registered --junitxml=C:\tmp\vfe3-report-registry.xml`

Expected XML: zero failures and zero errors.

- [ ] **Step 5: Commit the dispatch seam**

```powershell
git add vfe3/viz/specs.py vfe3/viz/report.py tests/test_report.py
git commit -m "feat: dispatch reports through figure registry"
```

### Task 2: Route production numeric diagnostics through the metric registry

**Files:**

- Modify: `vfe3/metrics.py:1302-1409`
- Modify: `vfe3/model/model.py:2340-2705`
- Test: `tests/test_metrics.py`
- Test: `tests/test_diagnostics.py`

**Interfaces:**

- Consumes: `compute_metrics(names, **context)`.
- Produces: `DIAGNOSTIC_METRIC_NAMES = ("attention_entropy", "free_energy_terms", "effective_rank")` and the declarative output mapping `(("attention_entropy", "attn_entropy", False), ("free_energy_terms", None, True), ("effective_rank", "effective_rank", False))`, where the final flag means flatten a returned mapping.

```python
DIAGNOSTIC_METRIC_NAMES = (
    "attention_entropy",
    "free_energy_terms",
    "effective_rank",
)
DIAGNOSTIC_METRIC_OUTPUTS = (
    ("attention_entropy", "attn_entropy", False),
    ("free_energy_terms", None, True),
    ("effective_rank", "effective_rank", False),
)
```

- [ ] **Step 1: Add a registry-substitution regression**

Register an override for `attention_entropy`, run `model.diagnostics()` on a `K=2`, one-layer model, and assert the override value appears under `attn_entropy` while the free-energy component remains under `attention_entropy`. Restore the original callable in `finally` so the test cannot pollute later cases. Add a diagonal `N == K` case proving the registered effective-rank wrapper receives an explicit `diagonal` flag instead of misclassifying the variance table as a full covariance.

- [ ] **Step 2: Verify the test fails because diagnostics calls the concrete function**

Run: `python -m pytest tests/test_diagnostics.py -k registry_dispatch --junitxml=C:\tmp\vfe3-metric-registry-red.xml`

Expected: the output retains the concrete metric value instead of the override.

- [ ] **Step 3: Build one context dictionary and dispatch the migrated group**

```python
_diag = out.sigma.dim() == out.mu.dim()
metric_context = {
    "sigma":                    out.sigma,
    "diagonal":                 _diag,
    "self_div":                 self_div,
    "energy":                   energy,
    "beta":                     beta,
    "alpha":                    alpha,
    "tau":                      _tau_b,
    "lambda_beta":              cfg.lambda_beta,
    "lambda_twohop":            cfg.lambda_twohop,
    "include_attention_entropy": cfg.include_attention_entropy,
    "log_prior":                log_prior,
    "alpha_reg":                 (alpha_reg if cfg.lambda_alpha_mode != "constant" else None),
    "coupling_energy":           coupling_energy,
    "log_likelihood":            log_likelihood,
}
registered = metrics.compute_metrics(list(metrics.DIAGNOSTIC_METRIC_NAMES), **metric_context)
d: Dict[str, float] = {}
for metric_name, output_name, flatten in metrics.DIAGNOSTIC_METRIC_OUTPUTS:
    value = registered[metric_name]
    if flatten:
        overlap = set(value) & set(d)
        if overlap:
            raise KeyError(f"diagnostic metric {metric_name!r} would overwrite {sorted(overlap)}")
        d.update({key: float(item) for key, item in value.items()})
    else:
        d[output_name] = float(value)
```

Place this block exactly where the live method currently constructs `d`, before any later diagnostic writes. Remove the old direct `attention_entropy`, `free_energy_terms`, and `effective_rank` calls and remove the later duplicate `_diag` assignment; every subsequent diagnostic continues writing into `d`. Update `_m_eff_rank` to require `diagonal` and pass it to `_spectrum`. Migrate only the three names above. Leave holonomy and gauge metrics in their bespoke paths because diagnostics consumes sibling confidence bounds and active-frame branches that their scalar registry wrappers do not return. The alias prevents the registered row entropy from overwriting the existing free-energy component named `attention_entropy`; parity tests must pin both values. Do not add accepted-but-unused adapter arguments.

- [ ] **Step 4: Run diagnostic parity tests**

Run: `python -m pytest tests/test_metrics.py tests/test_diagnostics.py tests/test_model_channel_diagnostics_2026_06_13.py --junitxml=C:\tmp\vfe3-metric-registry.xml`

Expected XML: zero failures and zero errors; existing default diagnostic dictionaries remain equal.

- [ ] **Step 5: Commit production metric dispatch**

```powershell
git add vfe3/metrics.py vfe3/model/model.py tests/test_metrics.py tests/test_diagnostics.py
git commit -m "refactor: route diagnostics through metric registry"
```

### Task 3: Add persisted-artifact adapters for scaling plots

**Files:**

- Create: `vfe3/viz/sweep_adapters.py`
- Modify: `scaling_analysis.py`
- Modify: `vfe3/viz/figures.py:2295-2397`
- Test: `tests/test_scaling_mup.py`
- Test: `tests/test_figures_tail.py`
- Test: `tests/test_viz.py`

**Interfaces:**

- Produces a separate `aggregate_validation_points(rows) -> list[dict]`. It groups by `(route, label)`, retains a group when it has finite positive persisted `best_val_ppl` even if every test metric is absent, computes `val_bits_per_token_mean` as the seed mean of `log2(best_val_ppl)`, and computes `wall_time_mean` from persisted `wall_time_s`. This is validation bits per token, not BPC: persisted scaling headlines do not carry the validation split's tokens-per-character factor. Existing `aggregate_points`, `test_bpc`, and `bpc_mean` remain character-corrected test-set paths and are never relabeled as validation metrics.
- Produces: `capacity_scaling_kwargs(points, axis_routes) -> Optional[dict]`, where `axis_routes` maps `embed_dim`, `n_heads`, and `n_layers` to one explicit scaling route each. A point enters an axis only when both `point["route"] == axis_routes[axis]` and `point["scale_knob"] == axis`; route equality alone is insufficient because the `inference` route carries both `n_e_steps` and `n_layers` cells.
- Produces: `pareto_frontier_kwargs(points) -> Optional[dict]` from aggregated parameter points carrying finite `val_bits_per_token_mean` and `n_params`.
- Consumes: registered figures `capacity_scaling` and `pareto_frontier`.

- [ ] **Step 1: Write adapter golden tests from synthetic persisted rows**

Build raw harvested-row fixtures with explicit `best_val_ppl`, deliberately different `test_bpc`, `n_params`, `wall_time_s`, route, scale_knob, and structural fields. Pass them through `aggregate_validation_points()` before either adapter. Assert `val_bits_per_token_mean == mean(log2(best_val_ppl))`, assert no adapter reads `test_bpc` or `bpc_mean`, and pin exact sorted arrays and labels. Add a validation-only fixture with `test_ce=None`, `test_ppl=None`, and `test_bpc=None`; it must survive validation aggregation and reach both adapters. Put both `infer_T` rows (`route="inference", scale_knob="n_e_steps"`) and `infer_L` rows (`route="inference", scale_knob="n_layers"`) in the same fixture and assert only the latter enter the `n_layers` panel. Use `axis_routes={"embed_dim": "grow_K", "n_heads": "blocksize", "n_layers": "inference"}`; an unavailable route is omitted, while a selected route with fewer than two finite points makes the capacity adapter return `None` with one reason. Include one explicit-null `best_val_ppl` in a selected complete route and require a clear validation error rather than silent test-metric substitution.

- [ ] **Step 2: Run the adapter tests and confirm the module is absent**

Run: `python -m pytest tests/test_scaling_mup.py tests/test_figures_tail.py -k "capacity_scaling or pareto_frontier" --junitxml=C:\tmp\vfe3-scaling-adapter-red.xml`

- [ ] **Step 3: Implement adapters and register two production specs**

The adapters and plotting contract are exact:

```python
capacity_kwargs = {
    "scaling": {
        axis: {
            "x": x_values,
            "bits_per_token": validation_bits,
            "wall_time": wall_times,
        }
        for axis in selected_axes
    }
}
pareto_kwargs = {
    "points": {
        "bits_per_token": validation_bits,
        "n_params": parameter_counts,
        "wall_time": wall_times,
        "label": labels,
    }
}
```

Update `plot_capacity_scaling` and `plot_pareto_frontier` to read `bits_per_token`, label their quality axes `validation bits/token`, and describe the quantity that way in docstrings. Update their direct tests in `tests/test_viz.py`; output filenames stay unchanged. `scaling_analysis.py` computes `validation_points = aggregate_validation_points(rows)` beside its unchanged test-based `points`, defines the explicit route map once, and dispatches these exact specs:

```python
SCALING_FIGURE_SPECS = (
    FigureSpec("capacity_scaling", "capacity_scaling.png",
               lambda ctx: capacity_scaling_kwargs(ctx["validation_points"], ctx["axis_routes"])),
    FigureSpec("pareto_frontier", "pareto_frontier.png",
               lambda ctx: pareto_frontier_kwargs(ctx["validation_points"])),
)
```

Held-out test BPC is not consumed by either figure, and no validation value is called BPC without a tokens-per-character factor.

- [ ] **Step 4: Run end-to-end file-emission tests**

Run: `python -m pytest tests/test_scaling_mup.py tests/test_figures_tail.py -k "capacity_scaling or pareto_frontier" --junitxml=C:\tmp\vfe3-scaling-adapter.xml`

Expected: both PNG paths exist and are nonempty; zero XML failures/errors.

- [ ] **Step 5: Commit scaling report adapters**

```powershell
git add vfe3/viz/sweep_adapters.py vfe3/viz/figures.py scaling_analysis.py tests/test_scaling_mup.py tests/test_figures_tail.py tests/test_viz.py
git commit -m "feat: emit registered scaling figures"
```

### Task 4: Add persisted-artifact adapters for ablation plots

**Files:**

- Modify: `vfe3/viz/sweep_adapters.py`
- Modify: `ablation.py:462-1300,1729-1818,1985-2084,2770-2797`
- Modify: `vfe3/viz/figures.py:2400-2425`
- Test: `tests/test_ablation_tackon.py`
- Test: `tests/test_figures_tail.py`
- Test: `tests/test_viz.py`
- Create: `tests/test_ablation_reporting.py`

**Interfaces:**

- Adds an opt-in, single-seed `component_ablation_forest` multi-arm sweep with exact `baseline`, `head_mixer_off`, and `precision_attention_off` arms. It sets `paired_token_bootstrap=True` and `forest_baseline_label="baseline"`, and stays out of `SWEEP_ORDER`.
- Adds an opt-in `e_q_mu_sigma_lr_grid` multi-arm sweep containing the Cartesian product of the existing `e_q_mu_lr` and `e_q_sigma_lr` value lists plus their current `BASELINE_CONFIG` values. It records `grid_x="e_q_mu_lr"`, `grid_y="e_q_sigma_lr"`, and stays out of `SWEEP_ORDER`.
- Produces: `ablation_forest_kwargs(sweep_dir, baseline_label) -> Optional[dict]` from persisted, aligned `val_token_nats.pt` files.
- Produces: `lr_grid_heatmap_kwargs(rows, x_key, y_key, x_values, y_values, baseline) -> Optional[dict]` from one completed two-dimensional sweep.
- Consumes: registered figures `ablation_forest` and `lr_grid_heatmap`.

- [ ] **Step 1: Write failing adapters and accumulated-sweep emission tests**

Construct one tiny forest sweep directory with three `ablation_result.json` files and aligned `val_token_nats.pt` tensors, and a separate 2-by-2 grid directory whose result rows carry both learning rates in `overrides`. For the forest, compute the oracle with `bootstrap_token_ce_band`, divide `delta`, `lo`, and `hi` by `log(2)` to produce bits/token, and assert exact adapter rows. For the grid, assert Cartesian completeness, x/y sort order, z placement from `primary_val_ppl`, the baseline marker, and rejection of a duplicate or missing cell. Dispatch both specs and assert `ablation_forest.png` and `lr_grid_heatmap.png` are nonempty.

- [ ] **Step 2: Confirm no production output is emitted yet**

Run: `python -m pytest tests/test_ablation_tackon.py tests/test_figures_tail.py -k "ablation_forest or lr_grid_heatmap" --junitxml=C:\tmp\vfe3-ablation-adapter-red.xml`

- [ ] **Step 3: Implement adapters and dispatch from the ablation tail**

Define the two dedicated sweeps after the `SWEEPS` literal so their values are derived from the live one-dimensional entries rather than copied:

```python
_GRID_MU_LRS = sorted(set([
    *SWEEPS["e_q_mu_lr"]["values"],
    BASELINE_CONFIG["e_q_mu_lr"],
]))
_GRID_SIGMA_LRS = sorted(set([
    *SWEEPS["e_q_sigma_lr"]["values"],
    BASELINE_CONFIG["e_q_sigma_lr"],
]))

SWEEPS["component_ablation_forest"] = {
    "description": "paired-token component ablation forest",
    "configs": [
        {"label": "baseline"},
        {"label": "head_mixer_off", "use_head_mixer": False},
        {"label": "precision_attention_off", "precision_weighted_attention": False},
    ],
    "paired_token_bootstrap": True,
    "forest_baseline_label": "baseline",
}
SWEEPS["e_q_mu_sigma_lr_grid"] = {
    "description": "joint q-mean and q-covariance E-step learning-rate grid",
    "configs": [
        {"label": f"mu={mu:g},sigma={sigma:g}",
         "e_q_mu_lr": mu, "e_q_sigma_lr": sigma}
        for sigma in _GRID_SIGMA_LRS
        for mu in _GRID_MU_LRS
    ],
    "grid_x": "e_q_mu_lr",
    "grid_y": "e_q_sigma_lr",
    "grid_x_values": _GRID_MU_LRS,
    "grid_y_values": _GRID_SIGMA_LRS,
    "grid_baseline": (BASELINE_CONFIG["e_q_mu_lr"], BASELINE_CONFIG["e_q_sigma_lr"]),
}
```

The grid contains every pair exactly once and `sweep_n_runs()` pins `len(_GRID_MU_LRS) * len(_GRID_SIGMA_LRS)`; neither sweep is added to `SWEEP_ORDER`. Persist `paired_token_bootstrap`, `forest_baseline_label`, `grid_x`, `grid_y`, both grid value lists, and `grid_baseline` in `sweep_meta.json` so adapters can operate after process restart.

Thread the opt-in through the only scope where the trained model and validation loader exist:

```python
def run_single(
    label:       str,
    overrides:   Dict[str, Any],
    run_dir:     Path,

    *,
    dataset:               str,
    device:                torch.device,
    seed:                  int,
    collect_diagnostics:   bool          = False,
    collect_extrapolation: bool          = False,
    paired_token_bootstrap: bool         = False,
    max_tokens:            Optional[int] = None,
    max_steps:             Optional[int] = None,
) -> Dict[str, Any]:
    # train(..., terminal_callback=...) populated terminal_result and returned the EMA-selected
    # live model. Preserve cell identity, then merge validation fields; do not restore evaluate().
    result = {
        "label": label,
        "error_kind": None,
        "n_params": n_params,
        "seed": int(cfg.seed),
        "overrides": _jsonable(overrides),
        "max_tokens": int(max_tokens) if max_tokens is not None else None,
    }
    result.update(terminal_result)
    token_identity = None
    if paired_token_bootstrap:
        per_token = per_unit_eval_nats(model, val_loader, device=device)["per_token_nats"]
        final_path = run_dir / "val_token_nats.pt"
        tmp_path = run_dir / "val_token_nats.pt.tmp"
        torch.save(per_token.detach().cpu(), tmp_path)
        _atomic_replace(final_path, tmp_path)
        token_identity = {
            "path": final_path.name,
            "sha256": _sha256_file(final_path),
            "size_bytes": final_path.stat().st_size,
            "numel": int(per_token.numel()),
            "dtype": str(per_token.dtype),
        }
    result["paired_token_bootstrap"] = bool(paired_token_bootstrap)
    result["val_token_nats_path"] = (
        token_identity["path"] if token_identity is not None else None
    )
    result["val_token_nats_sha256"] = (
        token_identity["sha256"] if token_identity is not None else None
    )
    result["val_token_nats_size_bytes"] = (
        token_identity["size_bytes"] if token_identity is not None else None
    )
    result["val_token_nats_numel"] = (
        token_identity["numel"] if token_identity is not None else None
    )
    result["val_token_nats_dtype"] = (
        token_identity["dtype"] if token_identity is not None else None
    )
```

Import `hashlib`, import `per_unit_eval_nats` from `vfe3.viz.extract`, and reuse `_atomic_replace` from `vfe3.run_artifacts`. This plan is implemented after the artifact-integrity plan, so it extends that plan's versioned cell contract instead of restoring the superseded `_cell_is_current` signature. In `run_sweep`, derive the flag, include it in `diagnostic_flags`, and apply a separate artifact-presence validator after the two-argument contract check:

```python
paired_token_bootstrap = bool(sweep.get("paired_token_bootstrap", False))
diagnostic_flags = {
    "collect_diagnostics": collect_diagnostics,
    "collect_extrapolation": collect_extrapolation,
    "paired_token_bootstrap": paired_token_bootstrap,
}
expected_contract = _expected_cell_contract_or_none(
    overrides, dataset, diagnostic_flags,
    seed=cell_seed, max_steps=max_steps, max_tokens=max_tokens,
)

if (expected_contract is not None
        and _cell_is_current(run_dir, expected_contract)
        and _paired_token_artifact_is_current(run_dir, required=paired_token_bootstrap)):
    # existing cached branch

result = run_single(
    label, overrides, run_dir,
    dataset=dataset, device=device, seed=cell_seed,
    collect_diagnostics=collect_diagnostics,
    collect_extrapolation=collect_extrapolation,
    paired_token_bootstrap=paired_token_bootstrap,
    max_tokens=max_tokens, max_steps=max_steps,
)
result["paired_token_bootstrap"] = paired_token_bootstrap
result.setdefault("val_token_nats_path", None)
result.setdefault("val_token_nats_sha256", None)
result.setdefault("val_token_nats_size_bytes", None)
result.setdefault("val_token_nats_numel", None)
result.setdefault("val_token_nats_dtype", None)
```

Define a local streaming `_sha256_file(path)` with the artifact plan's fixed 1 MiB block loop, then define `_paired_token_artifact_is_current(run_dir, *, required)`. It returns true immediately when `required` is false. When true, it requires the completion marker to contain a real true `paired_token_bootstrap`, `val_token_nats_path == "val_token_nats.pt"`, the exact SHA-256, file byte size, tensor length, and dtype fields shown above, and an existing file in `run_dir`. Recompute the file hash and size before safe-loading; then require a finite, nonempty, one-dimensional tensor whose `numel` and string dtype equal the marker. Missing, malformed, or nonexact metadata fails closed. The contract's sorted `diagnostic_flags` binds the request itself; this post-contract validator binds the requested artifact's exact bytes and tensor schema. Publish the tensor and compute its identity before the artifact-integrity workflow publishes `cell_contract.json` and its completion marker. Add all five paired-token identity fields plus `paired_token_bootstrap` to `_CSV_COLUMNS`. The stable, unshuffled validation loader and shared seed make rows position-aligned. This per-unit extraction is the only added validation replay; aggregate final validation remains owned by the artifact plan's terminal callback. Add a regression requiring `label`, `seed`, `overrides`, `n_params`, `max_tokens`, and `primary_val_ppl` to survive the terminal-field merge, with both grid learning rates retained in `overrides`. Add a same-shape finite replacement regression: overwrite a valid tensor with different finite values and require both cache reuse and the forest adapter to reject it because the file digest changed. Do not restore the old post-train `evaluate()` or collect or save token losses for any existing sweep.

Use the accumulated sweep view after all cells finish. The adapters return only arguments accepted by their plotters:

```python
def ablation_forest_kwargs(sweep_dir: Path, baseline_label: str) -> Optional[dict]:
    markers = _successful_markers_by_label(sweep_dir)
    baseline_marker = markers.get(baseline_label)
    if baseline_marker is None:
        return None
    baseline = _load_token_vector(sweep_dir, baseline_marker)
    rows = []
    for label in sorted(markers):
        arm = _load_token_vector(sweep_dir, markers[label])
        if arm.shape != baseline.shape:
            return None
        band = bootstrap_token_ce_band(arm, baseline, seed=0)
        rows.append({"label": label, **{key: band[key] / math.log(2.0)
                                        for key in ("delta", "lo", "hi")}})
    return {"rows": rows}


def lr_grid_heatmap_kwargs(
    rows:     List[Dict[str, Any]],
    x_key:    str,
    y_key:    str,
    x_values: Sequence[float],
    y_values: Sequence[float],
    baseline: Tuple[float, float],
) -> Optional[dict]:
    xs = tuple(float(value) for value in x_values)
    ys = tuple(float(value) for value in y_values)
    expected = {(x, y) for y in ys for x in xs}
    cells: Dict[Tuple[float, float], float] = {}
    for row in rows:
        overrides = row.get("overrides")
        if not isinstance(overrides, Mapping):
            return None
        try:
            pair = (float(overrides[x_key]), float(overrides[y_key]))
            value = float(row["primary_val_ppl"])
        except (KeyError, TypeError, ValueError):
            return None
        if pair not in expected or pair in cells or not math.isfinite(value):
            return None
        cells[pair] = value
    if set(cells) != expected:
        return None
    z = np.asarray([[cells[(x, y)] for x in xs] for y in ys], dtype=float)
    return {"grid": {"x": np.asarray(xs), "y": np.asarray(ys), "z": z,
                     "xlabel": x_key, "ylabel": y_key, "baseline": baseline}}
```

`_load_token_vector` resolves only the marker's exact child filename, recomputes and verifies the same hash/size/numel/dtype identity as the resume validator, calls `torch.load(..., map_location="cpu", weights_only=True)`, and requires a finite nonempty 1-D tensor. `plot_ablation_forest` changes its docstring and x-axis label from delta-BPC to `delta bits/token`, and `tests/test_viz.py` pins that label. The grid adapter requires the exact Cartesian product declared by persisted sweep metadata. Skip with one logged reason when a baseline artifact or complete grid is absent; never fabricate confidence bounds or cells. In `main()`, retain `union = run_sweep(...)`; immediately after the existing `sweep_dir`/`fig_dir` setup, define the active registry entry and complete context before dispatch:

```python
sweep = SWEEPS[name]
sweep_dir = output_dir / name
meta = json.loads((sweep_dir / "sweep_meta.json").read_text(encoding="utf-8"))
report_context = {
    "sweep_dir": sweep_dir,
    "rows": union,
    "baseline_label": meta.get("forest_baseline_label"),
    "grid_x": meta.get("grid_x"),
    "grid_y": meta.get("grid_y"),
    "grid_x_values": meta.get("grid_x_values"),
    "grid_y_values": meta.get("grid_y_values"),
    "baseline": tuple(meta["grid_baseline"]) if meta.get("grid_baseline") else None,
}
specs = []
if sweep.get("paired_token_bootstrap"):
    specs.append(FigureSpec("ablation_forest", "ablation_forest.png",
                 lambda ctx: ablation_forest_kwargs(ctx["sweep_dir"], ctx["baseline_label"])))
if sweep.get("grid_x") and sweep.get("grid_y"):
    specs.append(FigureSpec("lr_grid_heatmap", "lr_grid_heatmap.png",
                 lambda ctx: lr_grid_heatmap_kwargs(
                     ctx["rows"], ctx["grid_x"], ctx["grid_y"],
                     ctx["grid_x_values"], ctx["grid_y_values"], ctx["baseline"])))
emit_registered_figures(specs, report_context, fig_dir)
```

- [ ] **Step 4: Run end-to-end ablation emission tests**

Run: `python -m pytest tests/test_ablation_tackon.py tests/test_figures_tail.py tests/test_ablation_reporting.py -k "ablation_forest or lr_grid_heatmap or component_ablation_forest or e_q_mu_sigma_lr_grid" --junitxml=C:\tmp\vfe3-ablation-adapter.xml`

Expected: both registered figures are emitted for complete synthetic inputs and explicitly skipped for incomplete inputs.

- [ ] **Step 5: Commit ablation report adapters**

```powershell
git add vfe3/viz/sweep_adapters.py vfe3/viz/figures.py ablation.py tests/test_ablation_tackon.py tests/test_figures_tail.py tests/test_viz.py tests/test_ablation_reporting.py
git commit -m "feat: emit registered ablation figures"
```

### Task 5: Verify compatibility and document the registry contract

**Files:**

- Modify: `README.md:1057-1080`
- Modify: `docs/2026-07-12-edits.md`
- Modify: `tests/test_reporting_additions.py`

- [ ] **Step 1: Add a complete report-registry integration test**

Run one tiny persisted scaling fixture and one tiny ablation fixture through their production drivers. Assert the legacy files remain and the four new files appear; monkeypatch every registered builder once to prove the drivers resolve by name.

- [ ] **Step 2: Run the reporting slice**

Run: `python -m pytest tests/test_report.py tests/test_metrics.py tests/test_diagnostics.py tests/test_figures_tail.py tests/test_ablation_tackon.py tests/test_ablation_reporting.py tests/test_scaling_mup.py tests/test_reporting_additions.py --junitxml=C:\tmp\vfe3-reporting-registry-focused.xml`

Expected XML: zero failures and zero errors.

- [ ] **Step 3: Update documentation**

Document that callable registration and report selection are separate: a callable becomes production-reachable when a `FigureSpec` or metric-name list selects it. Record exact output names and artifact requirements.

- [ ] **Step 4: Run full verification**

Run: `python -m pytest -x --junitxml=C:\tmp\vfe3-reporting-registry-full.xml`

Expected XML: zero failures and zero errors; report exact tests/skips from XML.

- [ ] **Step 5: Commit documentation**

```powershell
git add README.md docs/2026-07-12-edits.md tests/test_reporting_additions.py
git commit -m "docs: define production report registration"
```

- [ ] **Step 6: Complete repository closeout.** Fetch and inspect `origin/main`, integrate any remote advance only inside the task worktree, and rerun affected verification. Push the task branch, fast-forward it into `main`, push `main`, and fetch again to verify the remote SHA. Fast-forward the user's live checkout only when doing so cannot alter its WIP; otherwise leave it untouched and report the blocker. Remove the temporary worktree, delete the local task branch, remove task-owned XML/figure fixtures, and show the final `git worktree list`, live-checkout `git status --short`, task commit SHA, pushed branch, and resulting `origin/main` SHA.

## Self-review

The plan makes both existing registries production-relevant, preserves current report outputs, and gives every orphan plot a persisted-artifact route. It avoids model replay, keeps failure isolation, defines exact adapters and filenames, and includes substitution tests that would fail if a driver bypasses the registry.
