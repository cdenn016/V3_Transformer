r"""Across-seed digest for an identical-config multi-seed run.

``train_vfe3.py`` with ``NUM_RUNS``/``SEEDS`` writes one seed-labelled run dir per seed, each holding a
``summary.json`` / ``test_results.json`` (headline scalars), ``provenance.json`` (the ``seed``),
``research.json`` (ECE, frequency-stratified CE, sigma diagnostics), ``metrics.csv`` (the per-step
training curves) and ``metrics_per_layer.csv`` (final per-layer diagnostics). This module aggregates
EVERY one of those across seeds and emits the full figure + data set in one pass:

- scalar headline + research metrics -> across-seed mean / SD / CV table, ``scalar_cv_summary`` figure,
  and a per-metric noise band (the floor every ablation 'win' must clear);
- per-step curves -> across-seed mean +/-1 SD ribbons (one per curated metric + an overview grid);
- per-layer diagnostics -> across-seed per-layer bars with SD error bars;
- machine-readable ``multiseed_summary.json`` / ``.csv`` and a human-readable ``MULTISEED_ANALYSIS.md``.

Click-to-run (project policy: no argparse): edit ``CONFIG`` below, then ``python multiseed_analysis.py``.
``run_root`` resolves a bare run-folder name under ``vfe3_runs/`` (so ``"K=20_GL(10)"`` just works).

Caveat carried into every output: the per-run reseed (train_vfe3) currently shares the data-shuffle
order across seeds, so this SD is the init+optimization spread only -- a LOWER BOUND on deployment
variance. A fixed data-order generator (model-init RNG varies, batch order held) is the companion fix;
see docs/experiments/2026-06-21-experiment-readiness.md (S6).
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _strip_seed_keys(value: Any) -> Any:
    """Recursively remove exact ``seed`` keys while preserving every other serialized value."""
    if isinstance(value, dict):
        return {key: _strip_seed_keys(item) for key, item in value.items() if key != "seed"}
    if isinstance(value, list):
        return [_strip_seed_keys(item) for item in value]
    return value


def _semantic_config(config_path: Path) -> Dict[str, Any]:
    """Load one current nested or legacy flat config artifact and remove only exact seed keys."""
    try:
        serialized = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"cannot aggregate {config_path.parent}: expected a readable config.json at {config_path}"
        ) from exc
    if not isinstance(serialized, dict):
        raise ValueError(f"cannot aggregate {config_path.parent}: config.json must contain an object")
    semantic = serialized.get("config", serialized)
    if not isinstance(semantic, dict):
        raise ValueError(
            f"cannot aggregate {config_path.parent}: config.json['config'] must contain an object"
        )
    return _strip_seed_keys(semantic)


def _config_fingerprint(config_path: Path) -> str:
    """Deterministic SHA-256 identity of one seed-normalized semantic config."""
    normalized = json.dumps(
        _semantic_config(config_path),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def _assert_homogeneous_configs(run_dirs: List[Path]) -> Optional[str]:
    """Return the shared semantic fingerprint or abort with each distinct fingerprint and path."""
    groups: Dict[str, List[Path]] = {}
    for run_dir in run_dirs:
        config_path = run_dir / "config.json"
        fingerprint = _config_fingerprint(config_path)
        groups.setdefault(fingerprint, []).append(config_path)
    if len(groups) > 1:
        detail = ["mixed semantic config fingerprints; refusing across-seed aggregation:"]
        for fingerprint, paths in sorted(groups.items()):
            detail.extend(f"  {fingerprint}  {path}" for path in paths)
        raise ValueError("\n".join(detail))
    return next(iter(groups), None)


def _as_finite_float(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def aggregate_seed_metric(
    run_root:  Any,
    key:       str = "test_ppl",

    *,
    filename:  str = "summary.json",
    seed_file: str = "config.json",
) -> Dict[str, Any]:
    r"""Scan ``run_root`` for per-seed ``filename``, gather ``key``, return the across-seed summary.

    Returns ``{n, mean, sd, two_sd, cv, values, seeds}`` with the UNBIASED (ddof=1) SD; ``sd``/``cv``
    are NaN for n<2. ``seeds`` reads :func:`_seed_for` (provenance.json -> config.json -> dir name).
    Non-finite or unreadable points are skipped (never crash the aggregation).
    """
    root = Path(run_root)
    run_dirs = sorted({path.parent for path in root.rglob(filename)})
    _assert_homogeneous_configs(run_dirs)
    values: List[float] = []
    seeds: List[Optional[int]] = []
    for run_dir in run_dirs:
        f = run_dir / filename
        v = _as_finite_float(_read_json(f).get(key))
        if v is None:
            continue
        values.append(v)
        seeds.append(_seed_for(run_dir, config_name=seed_file))

    out = _summarize(values)
    out["seeds"] = seeds
    return out


def _summarize(values: List[float]) -> Dict[str, Any]:
    r"""Across-seed ``{n, mean, sd, two_sd, cv, values}`` with the UNBIASED (ddof=1) SD (NaN for n<2)."""
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": math.nan, "sd": math.nan, "two_sd": math.nan,
                "cv": math.nan, "values": []}
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) if n >= 2 else math.nan
    cv = (sd / abs(mean)) if (n >= 2 and mean != 0.0) else math.nan
    return {"n": n, "mean": mean, "sd": sd, "two_sd": 2.0 * sd, "cv": cv, "values": values}


def _resolve_run_root(
    run_root: Any,

    *,
    search_dirs: tuple = ("vfe3_runs",),
) -> Path:
    r"""Resolve ``run_root`` to a directory: the literal if it exists, else ``<search_dir>/run_root``
    for the first ``search_dir`` that contains it, else the literal (the caller reports "nothing
    found"). Lets the bare run name ``"K=20_GL(10)"`` resolve to ``vfe3_runs/K=20_GL(10)``.
    """
    p = Path(run_root)
    if p.exists():
        return p
    for s in search_dirs:
        cand = Path(s) / run_root
        if cand.exists():
            return cand
    return p


def _seed_for(
    run_dir: Any,

    *,
    config_name: str = "config.json",
    prov_name:   str = "provenance.json",
) -> Optional[int]:
    r"""The seed for one run dir: ``provenance.json["seed"]`` -> ``config.json["seed"]`` -> the
    ``_s<NN>`` suffix of the dir name. None if every source is absent. (``config.json`` stores
    ``null`` in current runs; the real seed lives in ``provenance.json``.)
    """
    run_dir = Path(run_dir)
    for fname in (prov_name, config_name):
        s = _read_json(run_dir / fname).get("seed")
        if isinstance(s, (int, float)) and not isinstance(s, bool):
            return int(s)
    m = re.search(r"_s(\d+)$", run_dir.name)
    return int(m.group(1)) if m else None


def _seed_dirs(root: Path) -> List[Path]:
    r"""The per-seed run dirs directly under ``root`` (those carrying a run artifact), sorted.
    Skips siblings like ``figures/`` that hold no run JSON."""
    root = Path(root)
    if not root.exists():
        return []
    out = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and any((p / f).exists()
                              for f in ("summary.json", "provenance.json", "config.json")):
            out.append(p)
    _assert_homogeneous_configs(out)
    return out


def _dig(d: Dict[str, Any], dotted: str) -> Any:
    r"""Nested lookup by dotted key (``"corpus_freq_strata_ce.rare"``); None if missing."""
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def aggregate_scalar(
    run_root: Any,
    key:      str,

    *,
    sources: tuple = ("summary.json", "test_results.json", "research.json"),
) -> Dict[str, Any]:
    r"""Across-seed summary of one (possibly dotted) scalar ``key``, searching ``sources`` in order
    per seed dir. Same return shape as :func:`aggregate_seed_metric`. Handles ``research.json`` keys
    (ECE, ``corpus_freq_strata_ce.rare``, ...) that never appear in ``summary.json``.
    """
    root = _resolve_run_root(run_root)
    values: List[float] = []
    seeds: List[Optional[int]] = []
    for run_dir in _seed_dirs(root):
        val = None
        for src in sources:
            v = _as_finite_float(_dig(_read_json(run_dir / src), key))
            if v is not None:
                val = v
                break
        if val is None:
            continue
        values.append(val)
        seeds.append(_seed_for(run_dir))
    out = _summarize(values)
    out["seeds"] = seeds
    return out


def _read_csv_columns(path: Path) -> Dict[str, List[Optional[float]]]:
    r"""Read a CSV into ``{column: [float | None]}``; empty / non-numeric / non-finite cells -> None."""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        cols = list(reader.fieldnames or [])
        data: Dict[str, List[Optional[float]]] = {c: [] for c in cols}
        for row in reader:
            for c in cols:
                data[c].append(_as_finite_float(row.get(c)))
    return data


def _nan_mean_sd(M: np.ndarray) -> tuple:
    r"""NaN-aware across-row (axis 0) mean, UNBIASED (ddof=1) SD, and finite count per column.
    SD is NaN where fewer than 2 seeds reported; mean is NaN where none did."""
    n = np.sum(np.isfinite(M), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, np.nansum(M, axis=0) / n, np.nan)
        ss = np.nansum((M - mean) ** 2, axis=0)
        sd = np.where(n >= 2, np.sqrt(ss / np.where(n >= 2, n - 1, 1)), np.nan)
    return mean, sd, n


def aggregate_seed_curves(
    run_root: Any,

    *,
    columns:  Optional[List[str]] = None,
    filename: str = "metrics.csv",
    x:        str = "step",
) -> Dict[str, Dict[str, np.ndarray]]:
    r"""Across-seed per-step curves from each seed's ``metrics.csv``.

    Returns ``{column: {steps, mean, sd, n}}`` on the union ``x`` (step) grid, with the NaN-aware
    across-seed mean / ddof=1 SD / finite-seed-count per step (sparse columns like ``val_*`` average
    only the seeds that reported). ``columns=None`` -> every numeric column except ``x``.
    """
    root = _resolve_run_root(run_root)
    per_seed = []                                                # list of (steps[np], {col: vals[np]})
    all_cols: set = set()
    for run_dir in _seed_dirs(root):
        f = run_dir / filename
        if not f.exists():
            continue
        data = _read_csv_columns(f)
        if x not in data:
            continue
        steps = np.array([np.nan if v is None else v for v in data[x]], float)
        keep = np.isfinite(steps)
        steps = steps[keep]
        cols = {}
        for c, vals in data.items():
            if c == x:
                continue
            cols[c] = np.array([np.nan if v is None else v for v in vals], float)[keep]
            all_cols.add(c)
        per_seed.append((steps, cols))
    if not per_seed:
        return {}
    sel = list(columns) if columns is not None else sorted(all_cols)
    grid = np.unique(np.concatenate([s for s, _ in per_seed]))
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for c in sel:
        stack = []
        for steps, cols in per_seed:
            row = np.full(grid.shape, np.nan)
            if c in cols:
                row[np.searchsorted(grid, steps)] = cols[c]
            stack.append(row)
        mean, sd, n = _nan_mean_sd(np.vstack(stack))
        out[c] = {"steps": grid, "mean": mean, "sd": sd, "n": n}
    return out


def aggregate_per_layer(
    run_root: Any,

    *,
    filename: str = "metrics_per_layer.csv",
    layer_col: str = "layer",
) -> Dict[int, Dict[str, Dict[str, Any]]]:
    r"""Across-seed per-layer diagnostics from each seed's ``metrics_per_layer.csv``.
    Returns ``{layer: {column: {mean, sd, n, values}}}`` with the ddof=1 SD across seeds.
    """
    root = _resolve_run_root(run_root)
    acc: Dict[int, Dict[str, List[float]]] = {}
    for run_dir in _seed_dirs(root):
        f = run_dir / filename
        if not f.exists():
            continue
        data = _read_csv_columns(f)
        if layer_col not in data:
            continue
        for i, lay in enumerate(data[layer_col]):
            if lay is None:
                continue
            L = int(lay)
            row = acc.setdefault(L, {})
            for c, vals in data.items():
                if c == layer_col or vals[i] is None:
                    continue
                row.setdefault(c, []).append(vals[i])
    out: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for L, cols in acc.items():
        out[L] = {}
        for c, vals in cols.items():
            s = _summarize(vals)
            out[L][c] = {"mean": s["mean"], "sd": s["sd"], "n": s["n"], "values": vals}
    return out


def flag_noise_dominated(
    cell_metric: Dict[str, Optional[float]],
    sd:          float,

    *,
    k:           float = 2.0,
) -> List[str]:
    r"""Labels of single-seed ablation cells within ``k`` SD of the best cell -- i.e. plausibly seed
    noise rather than a real win. ``cell_metric`` maps label -> the cell's (single-seed) metric.
    """
    finite = {lab: v for lab, v in cell_metric.items() if _as_finite_float(v) is not None}
    if not finite or not math.isfinite(sd):
        return []
    best = min(finite.values())
    return [lab for lab, v in finite.items() if (v - best) < k * sd]


# =============================================================================
# CLICK-TO-RUN  -- edit, then `python multiseed_analysis.py`.
# =============================================================================
CONFIG: Dict[str, Any] = {
    "run_root": "K=60_GL(10)",   # run folder (bare name resolves under vfe3_runs/) OR a path
    "key":      "test_ppl",      # headline metric for the per-seed noise band
}

# Headline scalars to aggregate (searched in summary.json -> test_results.json -> research.json).
# Dotted keys dig into nested research blocks.
SCALAR_KEYS: List[str] = [
    "test_ppl", "best_val_ppl", "test_ce", "test_bpc", "test_ce_no_estep", "estep_capacity_gain",
    "wall_time_s", "ece", "overall_ce", "sigma_trace_cv", "sigma_ce_spearman",
    "fd_gradient_worst_rel_error",
    "corpus_freq_strata_ce.rare", "corpus_freq_strata_ce.mid", "corpus_freq_strata_ce.frequent",
]

# Per-step curves to draw an across-seed band for: (metrics.csv column, log-y?). Axis/title labels
# are publication-quality math, resolved centrally by vfe3.viz.figures.pub_label (PUB_LABELS).
CURVE_SPECS: List[tuple] = [
    ("train_ce",           False),
    ("val_ppl",            False),
    ("free_energy_total",  False),
    ("self_coupling",      False),
    ("belief_coupling",    False),
    ("attention_entropy",  False),
    ("self_divergence",    False),
    ("hyper_prior",        False),
    ("gamma_coupling",     False),
    ("grad_norm",          True),
    ("holonomy_deviation", True),
    ("gauge_trace_spread", False),
    ("effective_rank",     False),
    ("attn_entropy",       False),
    ("belief_cond_median", True),
    ("fisher_trace_mean",  False),
    ("generalization_gap", False),
]

GRID_COLS: List[str] = [                                          # overview-grid subset
    "train_ce", "val_ppl", "free_energy_total", "grad_norm", "holonomy_deviation",
    "gauge_trace_spread", "effective_rank", "attn_entropy", "belief_cond_median",
]
_LOGY_COLS = {c for c, logy in CURVE_SPECS if logy}

PER_LAYER_METRICS: List[str] = [                                  # per-layer bars to draw
    "self_coupling", "belief_coupling", "attention_entropy", "effective_rank",
    "holonomy_deviation", "gauge_trace_spread", "belief_cond_median",
]

CAVEAT = ("Per-run reseed shares the data-shuffle order across seeds, so every SD here is the "
          "init+optimization spread only -- a LOWER BOUND on deployment variance "
          "(companion fix: docs/experiments/2026-06-21-experiment-readiness.md S6).")


def _slug(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", s)


def _fnum(x: Any, fmt: str = ".4f") -> str:
    r"""Format a float for a table, or 'n/a' if None / non-finite."""
    return format(x, fmt) if isinstance(x, (int, float)) and math.isfinite(x) else "n/a"


def _json_clean(obj: Any) -> Any:
    r"""Recursively make ``obj`` JSON-valid: numpy scalars/arrays -> python; non-finite floats -> None."""
    if isinstance(obj, dict):
        return {k: _json_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_json_clean(v) for v in obj.tolist()]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    return obj


def _final_finite(d: Dict[str, np.ndarray]) -> Dict[str, Any]:
    r"""Last-finite-step {step, mean, sd, n} of one aggregated curve (the converged value)."""
    finite = np.where(np.isfinite(d["mean"]))[0]
    if finite.size == 0:
        return {"step": None, "mean": None, "sd": None, "n": 0}
    i = int(finite[-1])
    sd = float(d["sd"][i])
    return {"step": float(d["steps"][i]), "mean": float(d["mean"][i]),
            "sd": (sd if math.isfinite(sd) else None), "n": int(d["n"][i])}


def _write_csv(path: Path, scalars: Dict[str, Any]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "n", "mean", "sd", "two_sd", "cv"])
        for k, v in scalars.items():
            row = [k, v["n"]]
            for key in ("mean", "sd", "two_sd", "cv"):
                x = v[key]
                row.append("" if not (isinstance(x, (int, float)) and math.isfinite(x)) else x)
            w.writerow(row)


def _write_md(path: Path, root: Path, seeds, scalars, per_layer) -> None:
    L: List[str] = [f"# Multi-seed digest: {root.name}", "",
                    f"Across {len(seeds)} seeds: {seeds}.", "",
                    "## Headline scalars (across-seed)", "",
                    "| metric | n | mean | SD (ddof=1) | 2*SD | CV% |",
                    "|---|---|---|---|---|---|"]
    for k, v in scalars.items():
        cvpct = 100 * v["cv"] if math.isfinite(v["cv"]) else float("nan")
        L.append(f"| {k} | {v['n']} | {_fnum(v['mean'])} | {_fnum(v['sd'])} | "
                 f"{_fnum(v['two_sd'])} | {_fnum(cvpct, '.3f')} |")
    finite = {k: v for k, v in scalars.items() if math.isfinite(v["cv"])}
    if finite:
        most = min(finite.items(), key=lambda kv: kv[1]["cv"])
        least = max(finite.items(), key=lambda kv: kv[1]["cv"])
        L += ["", "## Key findings", ""]
        if "test_ppl" in scalars:
            t = scalars["test_ppl"]
            vals = [round(x, 3) for x in t["values"]]
            L.append(f"- test PPL = {_fnum(t['mean'], '.3f')} +/- {_fnum(t['sd'], '.3f')} "
                     f"(2*SD = {_fnum(t['two_sd'], '.3f')}); per-seed {vals}.")
        L.append(f"- Most seed-stable: `{most[0]}` (CV {100 * most[1]['cv']:.3f}%).")
        L.append(f"- Least seed-stable: `{least[0]}` (CV {100 * least[1]['cv']:.3f}%).")
    if per_layer:
        L += ["", f"## Per-layer ({len(per_layer)} layer(s))", "",
              "See `figures/per_layer__*.png` for across-seed per-layer bars."]
    L += ["", "## Caveat", "", CAVEAT, "",
          "## Figures", "",
          "In `figures/`: per-metric `curve_band__*.png` + `curve_band_grid.png` (per-step bands), "
          "`scalar_cv_summary.png` (seed stability), `per_layer__*.png`, and "
          f"`noise_band__{_slug(CONFIG['key'])}.png`.", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def _emit_figures(root: Path, scalars, curves, per_layer) -> None:
    r"""Write the full figure set into ``root/figures``; each figure is best-effort (a plotting /
    dependency error is logged and skipped), mirroring the rest of the figure suite."""
    try:
        from vfe3.viz import figures as figs
    except Exception as exc:                                      # matplotlib optional / headless
        print(f"  (figures skipped, plotting unavailable: {exc})")
        return
    figs.set_publication_style()
    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    made = 0

    def _try(make, out: Path) -> None:
        nonlocal made
        try:
            figs.plt.close(make())
            made += 1
        except Exception as exc:                                  # one bad figure never aborts the rest
            print(f"  (figure {out.name} skipped: {exc})")

    key = CONFIG["key"]
    if key in scalars:
        out = fig_dir / f"noise_band__{_slug(key)}.png"
        _try(lambda: figs.plot_ppl_noise_band(scalars[key], label=key, path=str(out)), out)

    out = fig_dir / "scalar_cv_summary.png"
    _try(lambda: figs.plot_scalar_cv_summary(scalars, path=str(out)), out)

    for col, logy in CURVE_SPECS:
        d = curves.get(col)
        if d is None or not np.any(np.isfinite(d["mean"])):
            continue
        out = fig_dir / f"curve_band__{_slug(col)}.png"
        _try(lambda d=d, logy=logy, col=col, out=out:
             figs.plot_curve_band(d["steps"], d["mean"], d["sd"], label=col,
                                  n=d["n"], logy=logy, path=str(out)), out)

    grid = [{"steps": curves[c]["steps"], "mean": curves[c]["mean"], "sd": curves[c]["sd"],
             "title": c, "logy": c in _LOGY_COLS}
            for c in GRID_COLS if c in curves and np.any(np.isfinite(curves[c]["mean"]))]
    if grid:
        out = fig_dir / "curve_band_grid.png"
        _try(lambda: figs.plot_curve_band_grid(grid, path=str(out)), out)

    for metric in PER_LAYER_METRICS:
        if per_layer and any(metric in cols for cols in per_layer.values()):
            out = fig_dir / f"per_layer__{_slug(metric)}.png"
            _try(lambda metric=metric, out=out:
                 figs.plot_per_layer_band(per_layer, metric, path=str(out)), out)

    print(f"  figures -> {fig_dir}  ({made} written)")


def main() -> None:
    root = _resolve_run_root(CONFIG["run_root"])
    seed_dirs = _seed_dirs(root)
    if not seed_dirs:
        print(f"no per-seed run dirs under {str(root)!r} "
              f"(looked for summary.json / provenance.json / config.json)")
        return
    seeds = [_seed_for(d) for d in seed_dirs]
    print(f"\n=== Multi-seed digest: {root}  ({len(seed_dirs)} seeds: {seeds}) ===\n")

    scalars: Dict[str, Any] = {}
    print(f"{'metric':<28}{'n':>3}{'mean':>14}{'sd':>13}{'2sd':>12}{'cv%':>9}")
    print("-" * 79)
    for k in SCALAR_KEYS:
        a = aggregate_scalar(root, k)
        if a["n"] == 0:
            continue
        scalars[k] = a
        cvpct = 100 * a["cv"] if math.isfinite(a["cv"]) else float("nan")
        print(f"{k:<28}{a['n']:>3}{a['mean']:>14.4f}{_fnum(a['sd']):>13}"
              f"{_fnum(a['two_sd']):>12}{_fnum(cvpct, '.3f'):>9}")
    print(f"\nNOTE: {CAVEAT}\n")

    curves = aggregate_seed_curves(root)
    per_layer = aggregate_per_layer(root)

    manifest = {
        "run_root": str(root),
        "config_fingerprint": _config_fingerprint(seed_dirs[0] / "config.json"),
        "n_seeds": len(seed_dirs),
        "seeds": seeds,
        "caveat": CAVEAT,
        "scalars": scalars,
        "curves_final_step": {c: _final_finite(d) for c, d in curves.items()},
        "per_layer": {str(L): {c: {"mean": s["mean"], "sd": s["sd"], "n": s["n"]}
                               for c, s in cols.items()}
                      for L, cols in per_layer.items()},
    }
    (root / "multiseed_summary.json").write_text(
        json.dumps(_json_clean(manifest), indent=2), encoding="utf-8")
    _write_csv(root / "multiseed_summary.csv", scalars)
    _write_md(root / "MULTISEED_ANALYSIS.md", root, seeds, scalars, per_layer)
    for name in ("multiseed_summary.json", "multiseed_summary.csv", "MULTISEED_ANALYSIS.md"):
        print(f"  data    -> {root / name}")

    _emit_figures(root, scalars, curves, per_layer)


if __name__ == "__main__":
    main()
