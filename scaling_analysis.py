r"""Click-to-run analysis + figures for the VFE_3.0 PARAMETER-scaling runs written by ``scaling.py``.

Harvests every run directory under ``input_dir`` (each carries ``summary.json`` with the enriched
``scaling_point`` block, plus ``config.json`` / ``provenance.json`` / ``scaling_cell.json``), writes a
tidy ``scaling_points.csv`` (one row per run), aggregates seeds per (route, size) point, fits the loss
power law ``test_ce = A * N^{-alpha}`` (with bootstrap confidence on the exponent), runs the multi-route
frontier-collapse F-test (do different ways of growing N share one ``L(N)`` curve?), prints the tables,
and writes the scaling figures via ``vfe3.viz.figures``. There is no model re-run and no CLI parsing:
edit ``CONFIG`` and run ``python scaling_analysis.py``.

The y-axis is ``test_ce`` (nats/token), the canonical additive scaling quantity; perplexity is a
nonlinear re-read. The x-axis is the recorded ``n_params`` (never a derived ``d^2`` proxy). Runs whose
``test_ce`` is null (no test split) or non-positive are dropped before the log fit; the analysis warns
if the harvested points span more than one ``data_sha256`` (mixed corpus) or ``git_sha`` (code drift),
either of which confounds a frontier.

Two further figures, ``capacity_scaling.png`` and ``pareto_frontier.png`` (audit finding PB-07), are
dispatched separately from persisted validation metrics: ``vfe3.viz.sweep_adapters.aggregate_validation_points``
collapses seeds keyed on the persisted ``best_val_ppl`` (never a test metric) into ``validation bits/token``
(``log2(best_val_ppl)``), which is NOT the character-corrected test BPC that ``aggregate_points`` /
``bpc_mean`` carry above.
"""

import os
if os.environ.get("VFE3_ALLOW_DUPLICATE_OPENMP") == "1":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from vfe3.viz.sweep_adapters import aggregate_validation_points, capacity_scaling_kwargs, pareto_frontier_kwargs
from vfe3.run_artifacts import _write_json_atomic

logger = logging.getLogger("scaling_analysis")

INFERENCE_ROUTE = "inference"                                # flat-N routes, plotted separately from L(N)

# PB-07: validation-metric figures (capacity_scaling.png, pareto_frontier.png) read validation_points
# (aggregate_validation_points, keyed on the persisted best_val_ppl) -- never the test-metric
# points/param_points/infer_points used above. The route each axis reads is defined once here; the
# FigureSpec pair itself is built lazily inside _make_figures (behind the same guarded viz import the
# figure pass already uses) so `import scaling_analysis` never requires the matplotlib stack.
AXIS_ROUTES: Dict[str, str] = {"embed_dim": "grow_K", "n_heads": "blocksize", "n_layers": INFERENCE_ROUTE}

CONFIG: Dict[str, Any] = {
    "input_dir":   "vfe3_scaling_results/blocks_K48",      # where scaling.py wrote the run dirs
    "with_offset": True,                                   # headline fit: False -> A*N^-alpha; True -> E + A*N^-alpha
    "n_bootstrap": 2000,                                    # nested (points x seeds) bootstrap for the exponent CI
    "min_points":  2,                                       # a route needs this many sizes to get its own fit
}

_CSV_COLUMNS = [
    "route", "scale_knob", "label", "seed", "n_params", "n_learnable_params", "embed_dim", "n_heads",
    "n_gen", "gauge_group", "n_layers", "n_e_steps", "family", "tokens_seen", "est_flops_6ND",
    "est_flops_analytic", "active_params_per_token", "test_ce", "test_ppl",
    "test_bits_per_token", "test_bpc",
    "estep_final_f_per_token", "best_val_ppl",
    "wall_time_s", "data_sha256", "train_data_sha256", "val_data_sha256", "test_data_sha256",
    "git_sha",
]

_ROUTE_NOTES = {
    "blocks_K48_tied_2x": "tied structural ablation; not a strict full-covariance pure control",
}


# =============================================================================
# HARVEST
# =============================================================================

def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def harvest(input_dir: Path) -> List[Dict[str, Any]]:
    r"""One row per run directory (a dir containing ``summary.json``). Pulls the scaling point, the
    held-out test numbers, the structural config, the cell's route/scale_knob, and provenance."""
    rows: List[Dict[str, Any]] = []
    for summ_path in sorted(input_dir.glob("**/summary.json")):
        run = summ_path.parent
        summ = _read_json(summ_path)
        sp = summ.get("scaling_point", {})
        cfgj = _read_json(run / "config.json")
        cfg = cfgj.get("config", {})
        prov = _read_json(run / "provenance.json")
        cell = _read_json(run / "scaling_cell.json")
        test = _read_json(run / "test_results.json")
        test_ce = sp.get("test_ce")                          # coalesce: dict.get's default fires only on
        if test_ce is None:                                  # a MISSING key, not an explicit null, so a
            test_ce = test.get("test_ce")                    # run with test_ce=null would drop a real value (r2 id14)
        rows.append({
            "run_dir":     str(run),
            "route":       cell.get("route", "(unlabeled)"),
            "scale_knob":  cell.get("scale_knob", "(unknown)"),
            "label":       cell.get("label", run.name),
            "seed":        prov.get("seed", cfg.get("seed")),
            "n_params":    summ.get("n_params", sp.get("n_params")),
            "n_learnable_params":      sp.get("n_learnable_params"),
            "embed_dim":   sp.get("embed_dim", cfg.get("embed_dim")),
            "n_heads":     sp.get("n_heads", cfg.get("n_heads")),
            "n_gen":       sp.get("n_gen", cell.get("n_gen")),
            "gauge_group": sp.get("gauge_group", cfg.get("gauge_group")),
            "n_layers":    sp.get("n_layers", cfg.get("n_layers")),
            "n_e_steps":   sp.get("n_e_steps", cfg.get("n_e_steps")),
            "family":      cfg.get("family"),
            "tokens_seen": sp.get("tokens_seen"),
            "est_flops_6ND":           sp.get("est_flops_6ND"),
            "est_flops_analytic":      sp.get("est_flops_analytic"),
            "active_params_per_token": sp.get("active_params_per_token"),
            "test_ce":     test_ce,
            "test_ppl":    test.get("test_ppl", sp.get("test_ppl")),
            "test_bits_per_token": test.get(
                "test_bits_per_token",
                sp.get("test_bits_per_token", summ.get("test_bits_per_token")),
            ),
            "test_bpc": test.get(
                "test_bpc", sp.get("test_bpc", summ.get("test_bpc"))),
            "estep_final_f_per_token": (summ.get("estep_final_f_per_token")          # C2/EXP-5 join
                                        if summ.get("estep_final_f_per_token") is not None
                                        else test.get("estep_final_f_per_token")),
            "best_val_ppl": (summ.get("best_val_ppl") if summ.get("best_val_ppl") is not None
                             else test.get("best_val_ppl")),
            "wall_time_s": summ.get("wall_time_s", sp.get("wall_time_s")),
            "data_sha256": prov.get("data_sha256") or prov.get("test_data_sha256"),
            "train_data_sha256": prov.get("train_data_sha256"),
            "val_data_sha256": prov.get("val_data_sha256"),
            "test_data_sha256": prov.get("test_data_sha256") or prov.get("data_sha256"),
            "git_sha":     prov.get("git_sha"),
        })
    return rows


def _requested_design(input_dir: Path, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Left-join the declared scaling cells to harvested results and classify every request."""
    path = input_dir / "scaling_design.json"
    raw = _read_json(path)
    requested = raw.get("cells")
    if raw.get("schema_version") != 1 or not isinstance(requested, list):
        return {
            "available": False,
            "complete": None,
            "status": "unverifiable_design",
            "cells": [],
            "counts": {},
        }

    observed: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}
    for row in rows:
        try:
            key = (str(row["route"]), str(row["label"]), int(row["seed"]))
        except (KeyError, TypeError, ValueError):
            continue
        observed.setdefault(key, []).append(row)

    cells: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    for item in requested:
        if not isinstance(item, Mapping):
            cell = {"status": "unreadable", "error": "design cell is not a mapping"}
        else:
            cell = dict(item)
            try:
                key = (str(cell["route"]), str(cell["label"]), int(cell["seed"]))
            except (KeyError, TypeError, ValueError):
                cell["status"] = "unreadable"
                cell["error"] = "design cell lacks a valid route, label, or seed"
            else:
                matches = observed.get(key, [])
                declared = str(cell.get("status", "pending"))
                if declared in {"failed", "nonfinite", "unreadable"}:
                    cell["status"] = declared
                elif len(matches) > 1:
                    cell["status"] = "duplicate"
                elif not matches:
                    cell["status"] = "missing"
                else:
                    value = _as_float(matches[0].get("test_ce"))
                    cell["status"] = (
                        "complete" if np.isfinite(value) and value > 0.0 else "nonfinite"
                    )
        status = str(cell.get("status", "unreadable"))
        counts[status] = counts.get(status, 0) + 1
        cells.append(cell)
    complete = bool(cells) and counts.get("complete", 0) == len(cells)
    return {
        "available": True,
        "complete": complete,
        "status": "complete" if complete else "incomplete",
        "cells": cells,
        "counts": counts,
    }


def write_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    r"""Tidy ``scaling_points.csv`` (fixed columns; missing keys blank). Python file I/O, never
    PowerShell ``>`` redirection (which writes UTF-16LE+BOM that numpy/csv misread)."""
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in _CSV_COLUMNS})


# =============================================================================
# AGGREGATION  (seeds -> one point per (route, label))
# =============================================================================

def _as_float(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _sorted_strings(rows: List[Dict[str, Any]], key: str) -> List[str]:
    return sorted({str(r[key]) for r in rows if r.get(key)})


def _analysis_provenance(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    r"""Summarize the code, data, and token-budget identities of the fitted parameter rows."""
    sha_keys = ("data_sha256", "train_data_sha256", "val_data_sha256", "test_data_sha256")
    sha_sets = {key: _sorted_strings(rows, key) for key in sha_keys}
    git_shas = _sorted_strings(rows, "git_sha")
    token_values = sorted({
        value for value in (_as_float(r.get("tokens_seen")) for r in rows)
        if np.isfinite(value)
    })
    token_budgets: List[int | float] = [int(v) if float(v).is_integer() else float(v) for v in token_values]
    missing = {
        "git_sha": sum(not r.get("git_sha") for r in rows),
        "train_data_sha256": sum(not r.get("train_data_sha256") for r in rows),
        "val_data_sha256": sum(not r.get("val_data_sha256") for r in rows),
        "test_data_sha256": sum(not (r.get("test_data_sha256") or r.get("data_sha256")) for r in rows),
    }
    mixed_corpus = any(len(sha_sets[key]) > 1 for key in sha_keys)
    return {
        **sha_sets,
        "git_sha": git_shas,
        "n_distinct_data_sha256": len(sha_sets["data_sha256"]),
        "n_distinct_train_data_sha256": len(sha_sets["train_data_sha256"]),
        "n_distinct_val_data_sha256": len(sha_sets["val_data_sha256"]),
        "n_distinct_test_data_sha256": len(sha_sets["test_data_sha256"]),
        "n_distinct_git_sha": len(git_shas),
        "mixed_corpus": mixed_corpus,
        "code_drift": len(git_shas) > 1,
        "token_budgets": token_budgets,
        "n_distinct_token_budgets": len(token_budgets),
        "token_budget_varies": len(token_budgets) > 1,
        "n_missing_token_budgets": sum(not np.isfinite(_as_float(r.get("tokens_seen"))) for r in rows),
        "missing": missing,
    }


def _pooled_status(
    provenance: Dict[str, Any],
    frontier:   Dict[str, Any],

    *,
    has_fit:    bool,
    n_routes:   int,
) -> Tuple[str, List[str]]:
    if not has_fit:
        return "not_fitted", []
    reasons: List[str] = []
    if provenance.get("code_drift"):
        reasons.append("code_drift")
    if provenance.get("mixed_corpus"):
        reasons.append("mixed_corpus")
    if provenance.get("token_budget_varies"):
        reasons.append("token_budget_varies")
    if any(int(v) > 0 for v in (provenance.get("missing") or {}).values()):
        reasons.append("incomplete_provenance")
    if int(provenance.get("n_missing_token_budgets", 0)) > 0:
        reasons.append("incomplete_token_budget")
    if frontier.get("testable") and frontier.get("collapses") is False:
        reasons.append("routes_diverge")
    if reasons:
        return "confounded", reasons
    if frontier.get("testable") and frontier.get("collapses") is None:
        return "unassessed", []
    if n_routes > 1 and not frontier.get("testable"):
        return "unassessed", []
    return "clean", []


def _frontier_verdict(frontier: Dict[str, Any]) -> str:
    collapses = frontier.get("collapses")
    if collapses is True:
        return "one shared frontier (routes collapse)"
    if collapses is False:
        return "routes diverge (route-specific slope/intercept)"
    return "indeterminate (route collapse could not be decided)"


def aggregate_points(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    r"""Collapse seeds: one point per (route, label) carrying ``ce_seeds`` (the per-seed test CE) and
    the shared structural fields. n_params is averaged across seeds (identical by construction; a
    spread is warned about). Points with no finite positive CE are dropped."""
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((r["route"], r["label"]), []).append(r)
    points: List[Dict[str, Any]] = []
    for (route, label), grp in sorted(groups.items()):
        ce = [_as_float(g["test_ce"]) for g in grp]
        ce = [c for c in ce if np.isfinite(c) and c > 0]
        if not ce:
            continue
        npars = [_as_float(g["n_params"]) for g in grp if np.isfinite(_as_float(g["n_params"]))]
        if npars and (max(npars) - min(npars)) > 0:
            logger.warning("n_params varies across seeds for %s/%s (%s); using the mean", route, label, npars)
        # C2/EXP-5: per-seed converged final E-step F/token and named bit metrics.
        f_seeds = [c for c in (_as_float(g.get("estep_final_f_per_token")) for g in grp) if np.isfinite(c)]
        bits_per_token = [
            value for value in (_as_float(g.get("test_bits_per_token")) for g in grp)
            if np.isfinite(value)
        ]
        bpc = [c for c in (_as_float(g.get("test_bpc")) for g in grp) if np.isfinite(c)]
        # F1/EXP-6: per-seed test PPL for the muP K-stability figure (CE is the L(N) fit quantity above).
        ppl = [c for c in (_as_float(g.get("test_ppl")) for g in grp) if np.isfinite(c) and c > 0]
        points.append({
            "route": route, "label": label, "scale_knob": grp[0]["scale_knob"],
            "n_params": float(np.mean(npars)) if npars else float("nan"),
            "embed_dim": _as_float(grp[0].get("embed_dim")),         # F1/EXP-6 K axis
            "n_gen": _as_float(grp[0].get("n_gen")),
            "tokens_seen": _as_float(grp[0].get("tokens_seen")),
            "est_flops_6ND": _as_float(grp[0].get("est_flops_6ND")),
            "est_flops_analytic": _as_float(grp[0].get("est_flops_analytic")),
            "n_e_steps": _as_float(grp[0].get("n_e_steps")),
            "n_layers": _as_float(grp[0].get("n_layers")),
            "ce_seeds": ce, "n_seeds": len(ce),
            "ce_mean": float(np.mean(ce)),
            "ce_sem": float(np.std(ce, ddof=1) / np.sqrt(len(ce))) if len(ce) > 1 else 0.0,
            "f_seeds": f_seeds,
            "f_mean": float(np.mean(f_seeds)) if f_seeds else float("nan"),
            "bits_per_token_seeds": bits_per_token,
            "bits_per_token_mean": (
                float(np.mean(bits_per_token)) if bits_per_token else float("nan")),
            "bpc_seeds": bpc,
            "bpc_mean": float(np.mean(bpc)) if bpc else float("nan"),
            "ppl_mean": float(np.mean(ppl)) if ppl else float("nan"),
            "ppl_sem": float(np.std(ppl, ddof=1) / np.sqrt(len(ppl))) if len(ppl) > 1 else 0.0,
        })
    return points


def _kmup_series(param_points: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    r"""Group the grow_K / grow_K_mup points into the F1/EXP-6 ``kmup_stability`` series.

    SPLITS the muP route's matched ``K{k}_fixed`` / ``K{k}_mup`` arms -- they share
    ``route='grow_K_mup'`` (so a route-only grouping would collapse both into one duplicated-K curve
    and destroy the |b_fixed - b_muP| width-stability contrast). Keys: ``grow_K`` (the standalone
    width-fixed control), ``grow_K_mup/fixed``, ``grow_K_mup/mup`` -- one point per K each."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for p in param_points:
        if p["route"] not in ("grow_K", "grow_K_mup"):
            continue
        if not (np.isfinite(_as_float(p.get("embed_dim"))) and np.isfinite(_as_float(p.get("ppl_mean")))):
            continue
        key = p["route"]
        if p["route"] == "grow_K_mup":
            lab = str(p.get("label", ""))
            key = ("grow_K_mup/fixed" if lab.endswith("_fixed")
                   else "grow_K_mup/mup" if lab.endswith("_mup") else "grow_K_mup")
        out.setdefault(key, []).append({
            "embed_dim": _as_float(p["embed_dim"]), "ppl_mean": _as_float(p["ppl_mean"]),
            "ppl_sem": _as_float(p.get("ppl_sem", 0.0)), "n": int(p.get("n_seeds", 1)),
        })
    return out


# =============================================================================
# FITTING  (power law + bootstrap exponent CI + multi-route ANCOVA)
# =============================================================================

def bootstrap_exponent_ci(
    points:      List[Dict[str, Any]],

    *,
    weights:     Optional[np.ndarray] = None,
    n_boot:      int = 2000,
    with_offset: bool = False,
) -> Tuple[float, float, float]:
    r"""Bootstrap the power-law exponent with the same estimator form as the point estimate.

    Ordinary power-law replicates resample size clusters and their seed CEs. A successful offset-law
    fit keeps the distinct-size design fixed and resamples only seed CEs because cluster resampling
    usually makes that three-parameter fit underdetermined. Replicates whose realized fit form differs
    from the point estimate are rejected. Returns (alpha_hat, lo2.5, hi97.5). Needs >= 2 points;
    degenerate or no same-form replicates -> NaNs.

    ``weights`` (per-point, same order as ``points``) MUST match the headline fit's weighting so the
    point estimate and the CI come from ONE estimator -- otherwise a weighted headline alpha can sit
    outside its own unweighted CI (audit r2 id10). Resampled per replicate as ``weights[idx]``."""
    from vfe3.viz.figures import _fit_power_law
    xs = np.array([p["n_params"] for p in points], dtype=float)
    if xs.size < 2:
        return float("nan"), float("nan"), float("nan")
    seed_arrays = [np.asarray(p["ce_seeds"], dtype=float) for p in points]
    means = np.array([a.mean() for a in seed_arrays])
    point_fit = _fit_power_law(xs, means, weights=weights, with_offset=with_offset)
    alpha_hat = point_fit["alpha"]
    point_form = point_fit.get("form")
    rng = np.random.default_rng(0)                           # fixed -> reproducible CI
    boot: List[float] = []
    n = len(points)
    for _ in range(n_boot):
        # The offset estimator needs four distinct sizes. Resampling size clusters silently turns
        # most replicates into the no-offset fallback, so keep the fitted design fixed and resample
        # only seed observations. The ordinary power-law fit retains the cluster bootstrap.
        idx = np.arange(n) if point_form == "offset_power_law" else rng.integers(0, n, n)
        bx = xs[idx]
        by = np.array([float(rng.choice(seed_arrays[i], size=seed_arrays[i].size).mean()) for i in idx])
        bw = weights[idx] if weights is not None else None
        fit = _fit_power_law(bx, by, weights=bw, with_offset=with_offset)
        a = fit["alpha"]
        if np.isfinite(a) and fit.get("form") == point_form:
            boot.append(a)
    if not boot:
        return alpha_hat, float("nan"), float("nan")
    return alpha_hat, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _loglog_rss(x: np.ndarray, y: np.ndarray) -> float:
    r"""Residual sum of squares of a 1-D log-log least-squares line (NaN if < 2 points)."""
    if x.size < 2:
        return float("nan")
    b, a = np.polyfit(x, y, 1)
    return float(np.sum((y - (b * x + a)) ** 2))


def ancova_frontier_collapse(points: List[Dict[str, Any]], min_points: int = 2) -> Dict[str, Any]:
    r"""F-test for whether the routes share ONE log-log L(N) line. Pooled model (1 line, 2 params) vs
    full model (a separate slope+intercept per route, 2R params); a significant F means the routes do
    NOT collapse (route-specific slope and/or intercept). Uses only routes with >= ``min_points`` sizes
    and a total dof margin (n > 2R). Returns the F statistic, dof, p-value, and per-route point counts."""
    by_route: Dict[str, List[Dict[str, Any]]] = {}
    for p in points:
        n_params = _as_float(p.get("n_params"))
        ce_mean = _as_float(p.get("ce_mean"))
        if np.isfinite(n_params) and n_params > 0.0 and np.isfinite(ce_mean) and ce_mean > 0.0:
            by_route.setdefault(p["route"], []).append(p)
    route_sizes = {r: {float(p["n_params"]) for p in ps} for r, ps in by_route.items()}
    routes = {r: ps for r, ps in by_route.items() if len(route_sizes[r]) >= min_points}
    out: Dict[str, Any] = {
        "routes": {r: len(ps) for r, ps in routes.items()},
        "distinct_sizes": {r: len(route_sizes[r]) for r in routes},
        "testable": False,
    }
    if len(routes) < 2:
        out["reason"] = "need >= 2 routes with enough sizes"
        return out
    x_all, y_all = [], []
    rss_full, n_total, R = 0.0, 0, len(routes)
    for r, ps in routes.items():
        x = np.log(np.array([p["n_params"] for p in ps], dtype=float))
        y = np.log(np.array([p["ce_mean"] for p in ps], dtype=float))
        x_all.append(x); y_all.append(y)
        rss_full += _loglog_rss(x, y)
        n_total += x.size
    rss_pooled = _loglog_rss(np.concatenate(x_all), np.concatenate(y_all))
    df1, df2 = 2 * R - 2, n_total - 2 * R
    if df2 <= 0 or not np.isfinite(rss_full) or rss_full <= 0:
        out["reason"] = "insufficient dof for the F-test"
        return out
    F = ((rss_pooled - rss_full) / df1) / (rss_full / df2)
    try:
        from scipy.stats import f as f_dist
        p_value = float(f_dist.sf(F, df1, df2))
    except Exception:
        p_value = float("nan")
    out.update({"testable": True, "F": float(F), "df1": int(df1), "df2": int(df2),
                "p_value": p_value, "rss_pooled": rss_pooled, "rss_full": rss_full,
                "collapses": (p_value > 0.05) if np.isfinite(p_value) else None})
    return out


# =============================================================================
# REPORT + FIGURES
# =============================================================================

def _print_points_table(points: List[Dict[str, Any]]) -> None:
    print(f"\n{'route':<14}{'label':<18}{'N params':>14}{'n_gen':>8}{'seeds':>7}"
          f"{'test_ce':>11}{'+/-SEM':>10}")
    print("-" * 82)
    for p in sorted(points, key=lambda q: (q["route"], q["n_params"])):
        print(f"{p['route']:<14}{p['label']:<18}{int(p['n_params']):>14,}{int(p['n_gen']):>8}"
              f"{p['n_seeds']:>7}{p['ce_mean']:>11.4f}{p['ce_sem']:>10.4f}")


def analyze() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    input_dir = Path(CONFIG["input_dir"])
    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir {input_dir} does not exist; run scaling.py first")

    rows = harvest(input_dir)
    design = _requested_design(input_dir, rows)
    if not rows and not design["available"]:
        print(f"No runs (with summary.json) under {input_dir}; run scaling.py first.")
        return
    fig_dir = input_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    write_csv(rows, input_dir / "scaling_points.csv")
    print(f"\nVFE_3.0 scaling analysis\n  input:   {input_dir}\n  runs:    {len(rows)}"
          f"\n  csv:     {input_dir / 'scaling_points.csv'}")

    points = aggregate_points(rows)
    allow_parameter_fit = design["complete"] is not False
    # PB-07: best_val_ppl-keyed validation points for the two registered supplementary figures --
    # best-effort: aggregate_validation_points keeps its fail-loud ValueError (an explicit-null
    # best_val_ppl beside finite sibling seeds is a data-integrity fault, never silently dropped or
    # substituted with a test metric), but that fault withholds ONLY the validation figures; it must
    # never abort the legacy test-metric analysis below.
    if allow_parameter_fit:
        try:
            validation_points: Optional[List[Dict[str, Any]]] = aggregate_validation_points(rows)
        except Exception as exc:
            logger.warning("validation-point aggregation failed (%s); capacity_scaling/pareto_frontier withheld", exc)
            validation_points = None
    else:
        validation_points = None
    param_points = [p for p in points if p["route"] != INFERENCE_ROUTE]
    infer_points = [p for p in points if p["route"] == INFERENCE_ROUTE]
    fit_infer_points = infer_points if allow_parameter_fit else []
    fit_param_mask = np.array([
        allow_parameter_fit
        and np.isfinite(_as_float(p.get("n_params"))) and _as_float(p.get("n_params")) > 0.0
        for p in param_points
    ], dtype=bool)
    fit_param_points = [p for p, keep in zip(param_points, fit_param_mask) if keep]
    retained_param_cells = {
        (p["route"], p["label"])
        for p in fit_param_points
    }
    param_rows = [
        r for r in rows
        if (r["route"], r["label"]) in retained_param_cells
        and np.isfinite(_as_float(r.get("test_ce"))) and _as_float(r.get("test_ce")) > 0.0
    ]
    _print_points_table(points)

    # Persist every identity used to judge whether the parameter frontier is comparable.  Scope the
    # sets to the rows that actually enter L(N), not flat-N inference arms that are plotted separately.
    provenance = _analysis_provenance(param_rows)
    if provenance["mixed_corpus"]:
        logger.warning("harvested parameter runs span multiple data SHA-256 identities -- mixed corpus confounds the fit")
    if provenance["code_drift"]:
        logger.warning("harvested parameter runs span %d distinct git_sha -- code drift across the sweep",
                       provenance["n_distinct_git_sha"])
    if provenance["token_budget_varies"]:
        logger.warning("harvested parameter runs span multiple token budgets -- data budget confounds the fit")

    routes = sorted({p["route"] for p in fit_param_points})
    frontier = (
        ancova_frontier_collapse(fit_param_points, min_points=CONFIG["min_points"])
        if allow_parameter_fit
        else {"testable": False, "collapses": None, "reason": "incomplete_design"}
    )
    route_notes = {route: _ROUTE_NOTES[route] for route in routes if route in _ROUTE_NOTES}

    # Accumulate the printed fits / tests into a persisted summary (scaling_summary.json plus the
    # SCALING_ANALYSIS.md report): the pooled fit, bootstrap CI, per-route exponents, the
    # frontier-collapse F-test, and the E-step structural-EM correlations were console-only.
    summary: Dict[str, Any] = {
        "input_dir": str(input_dir), "n_runs": len(rows), "with_offset": CONFIG["with_offset"],
        "n_param_points": len(param_points), "n_inference_points": len(infer_points),
        "n_fitted_param_points": len(fit_param_points),
        "design": design,
        "provenance": provenance, "pooled_fit": None, "pooled_fit_status": "not_fitted",
        "pooled_fit_confounds": [], "per_route": {}, "frontier_collapse": frontier,
        "route_notes": route_notes, "estep_structural": None,
    }

    weights_by_route: Dict[str, np.ndarray] = {}
    if param_points:
        from vfe3.viz.figures import _scaling_sem_weights
        all_means = np.array([p["ce_mean"] for p in param_points], dtype=float)
        all_sem = np.array([p["ce_sem"] for p in param_points], dtype=float)
        all_weights = _scaling_sem_weights(all_means, all_sem)
        for route in sorted({p["route"] for p in param_points}):
            route_mask = np.array([p["route"] == route for p in param_points], dtype=bool)
            positive_x = np.array([
                np.isfinite(_as_float(p.get("n_params"))) and _as_float(p.get("n_params")) > 0.0
                for p in param_points
            ], dtype=bool)
            weights_by_route[route] = all_weights[route_mask & positive_x]
        fit_weights = all_weights[fit_param_mask]
    else:
        all_weights = np.array([], dtype=float)
        fit_weights = np.array([], dtype=float)

    # ---- L(N) power law + bootstrap exponent CI (pooled over the parameter routes) ----
    n_distinct_param_sizes = len({float(p["n_params"]) for p in fit_param_points})
    summary["n_distinct_param_sizes"] = n_distinct_param_sizes
    if n_distinct_param_sizes >= 2:
        from vfe3.viz.figures import _fit_power_law
        xs = np.array([p["n_params"] for p in fit_param_points], dtype=float)
        ys = np.array([p["ce_mean"] for p in fit_param_points], dtype=float)
        fit = _fit_power_law(xs, ys, weights=fit_weights, with_offset=CONFIG["with_offset"])
        _a_hat, lo, hi = bootstrap_exponent_ci(
            fit_param_points,
            weights=fit_weights,
            n_boot=CONFIG["n_bootstrap"],
            with_offset=CONFIG["with_offset"],
        )
        if np.isfinite(_as_float(fit.get("alpha"))):
            print(f"\nPOOLED L(N) fit ({fit['form']}):  alpha = {fit['alpha']:.4f}  "
                  f"[95% CI {lo:.4f}, {hi:.4f}]   A = {fit['A']:.4g}   "
                  + (f"E = {fit['E']:.4f}   " if fit["form"] == "offset_power_law" else "")
                  + f"R^2 = {fit['r2']:.4f}   over {fit['n_distinct_sizes']} distinct sizes")
            summary["pooled_fit"] = {
                "form": fit["form"], "alpha": fit["alpha"], "alpha_ci": [lo, hi],
                "A": fit["A"], "E": fit.get("E"), "r2": fit["r2"], "n_points": fit["n_points"],
                "n_distinct_sizes": fit["n_distinct_sizes"],
            }
        else:
            print("\n(pooled L(N) estimator returned no finite exponent; no estimate persisted)")
        # ---- per-route fits use exact slices of the one pooled SEM-weight vector ----
        if routes:
            print("\nper-route exponents:")
            for r in routes:
                pr = [p for p in fit_param_points if p["route"] == r]
                route_sizes = {float(p["n_params"]) for p in pr}
                if len(route_sizes) >= 2:
                    route_mask = np.array([p["route"] == r for p in fit_param_points], dtype=bool)
                    route_weights = fit_weights[route_mask]
                    fr = _fit_power_law(np.array([p["n_params"] for p in pr], dtype=float),
                                        np.array([p["ce_mean"] for p in pr], dtype=float),
                                        weights=route_weights, with_offset=CONFIG["with_offset"])
                    if not np.isfinite(_as_float(fr.get("alpha"))):
                        print(f"  {r:<14} (estimator returned no finite exponent; no estimate persisted)")
                        continue
                    display = f"{r} [structural ablation]" if r in route_notes else r
                    print(f"  {display:<36} alpha={fr['alpha']:.4f}  R^2={fr['r2']:.4f}  "
                          f"({fr['n_distinct_sizes']} distinct sizes; {fr['form']})")
                    summary["per_route"][r] = {
                        "form": fr["form"], "alpha": fr["alpha"], "A": fr["A"], "E": fr.get("E"),
                        "r2": fr["r2"], "n_points": fr["n_points"],
                        "n_distinct_sizes": fr["n_distinct_sizes"], "n_sizes": fr["n_distinct_sizes"],
                    }
                else:
                    print(f"  {r:<14} (only {len(route_sizes)} distinct size; need >= 2 to fit)")
    else:
        print(f"\n(only {n_distinct_param_sizes} distinct parameter size present; add more sizes to fit L(N))")

    if frontier.get("testable"):
        verdict = _frontier_verdict(frontier)
        print(f"\nfrontier-collapse F-test:  F({frontier['df1']},{frontier['df2']}) = {frontier['F']:.3f}  "
              f"p = {frontier['p_value']:.4g}  ->  {verdict}")
    else:
        print(f"\nfrontier-collapse F-test: not testable ({frontier.get('reason')})")

    if allow_parameter_fit:
        status, confounds = _pooled_status(
            provenance,
            frontier,
            has_fit=summary["pooled_fit"] is not None,
            n_routes=len(routes),
        )
    else:
        status, confounds = "incomplete_design", ["incomplete_design"]
    summary["pooled_fit_status"] = status
    summary["pooled_fit_confounds"] = confounds
    if summary["pooled_fit"] is not None:
        detail = f" ({', '.join(confounds)})" if confounds else ""
        print(f"pooled fit status: {status.upper()}{detail}")

    # ---- C2/EXP-5: structural non-Neal-Hinton EM -- F-vs-CE decorrelation across the n_e_steps arms ----
    estep_pts = sorted([p for p in fit_infer_points if p["scale_knob"] == "n_e_steps"
                        and np.isfinite(p.get("f_mean", float("nan")))],
                       key=lambda q: q["n_e_steps"])
    if len(estep_pts) >= 2:
        ne = np.array([p["n_e_steps"] for p in estep_pts])
        ff = np.array([p["f_mean"] for p in estep_pts])
        ce = np.array([p["ce_mean"] for p in estep_pts])

        def _pear(a, b):
            return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
        print(f"\nE-step structural-EM check ({len(estep_pts)} n_e_steps arms):"
              f"\n  Pearson(n_e_steps, final F/token) = {_pear(ne, ff):+.4f}  (expect strongly negative)"
              f"\n  Pearson(final F/token, test CE)   = {_pear(ff, ce):+.4f}  (expect ~0 / >=0 if F is target-blind)")
        summary["estep_structural"] = {
            "n_arms": len(estep_pts), "pearson_ne_final_f": _pear(ne, ff),
            "pearson_final_f_test_ce": _pear(ff, ce),
        }

    # ---- persist the analysis summary (json + a human-readable markdown report); best-effort so a
    # serialization/render error never suppresses the figure pass below ----
    try:
        _write_json_atomic(input_dir / "scaling_summary.json", summary)
        _write_scaling_md(input_dir / "SCALING_ANALYSIS.md", summary)
        print(f"\nsummary -> {input_dir / 'scaling_summary.json'}"
              f"\n           {input_dir / 'SCALING_ANALYSIS.md'}")
    except Exception as exc:
        logger.warning("scaling summary write failed (%s); skipped", exc)

    # ---- figures (best-effort, never fatal) ----
    figure_param_points = param_points if allow_parameter_fit else []
    _make_figures(figure_param_points, fit_infer_points, fig_dir, weights_by_route=weights_by_route,
                  validation_points=validation_points, axis_routes=AXIS_ROUTES)
    print(f"\nfigures -> {fig_dir}")


def _write_scaling_md(path: Path, summary: Dict[str, Any]) -> None:
    r"""Render ``summary`` as a readable SCALING_ANALYSIS.md (the pooled fit, per-route exponents,
    persisted provenance/confounds, frontier test, and E-step structural-EM correlations)."""
    L = ["# VFE_3.0 scaling analysis", "",
         f"- input: `{summary.get('input_dir')}`",
         f"- runs: {summary.get('n_runs')}  (parameter points: {summary.get('n_param_points')}, "
         f"inference points: {summary.get('n_inference_points')})",
         f"- requested fit: {'E + A N^-alpha (offset)' if summary.get('with_offset') else 'A N^-alpha'}", ""]

    provenance = summary.get("provenance") or {}
    if provenance:
        def _values(key: str) -> str:
            values = provenance.get(key) or []
            return ", ".join(f"`{value}`" for value in values) if values else "(missing)"

        L += ["## Provenance and confounds", "",
              f"- Git SHA set: {_values('git_sha')}",
              f"- train-data SHA-256 set: {_values('train_data_sha256')}",
              f"- validation-data SHA-256 set: {_values('val_data_sha256')}",
              f"- test-data SHA-256 set: {_values('test_data_sha256')}",
              f"- legacy held-out data SHA-256 set: {_values('data_sha256')}",
              f"- token budgets: {_values('token_budgets')}",
              f"- code drift: {bool(provenance.get('code_drift'))}",
              f"- mixed corpus: {bool(provenance.get('mixed_corpus'))}",
              f"- token-budget variation: {bool(provenance.get('token_budget_varies'))}",
              f"- missing identities: `{json.dumps(provenance.get('missing') or {}, sort_keys=True)}`",
              f"- missing token budgets: {int(provenance.get('n_missing_token_budgets', 0))}", ""]

    pf = summary.get("pooled_fit")
    status = str(summary.get("pooled_fit_status", "unassessed"))
    confounds = summary.get("pooled_fit_confounds") or []
    status_line = f"- status: **{status}**"
    if confounds:
        status_line += f" ({', '.join(str(reason) for reason in confounds)})"
    if pf:
        n_sizes = pf.get("n_distinct_sizes", pf.get("n_points"))
        L += ["## Pooled L(N) power law", "",
              status_line,
              f"- realized fit form: `{pf.get('form', 'unknown')}`",
              f"- exponent alpha = {pf['alpha']:.4f}  (95% CI [{pf['alpha_ci'][0]:.4f}, {pf['alpha_ci'][1]:.4f}])",
              f"- A = {pf['A']:.4g}" + (f", E = {pf['E']:.4f}"
                                        if pf.get("form") == "offset_power_law" and pf.get("E") is not None else ""),
              f"- R^2 = {pf['r2']:.4f} over {n_sizes} distinct sizes ({pf['n_points']} points)", ""]
    else:
        L += ["## Pooled L(N) power law", "", status_line,
              "- no pooled estimate is available", ""]
    pr = summary.get("per_route") or {}
    if pr:
        L += ["## Per-route exponents", "",
              "| route | realized form | alpha | R^2 | distinct sizes |",
              "|---|---|---|---|---|"]
        L += [
            f"| {r} | {d.get('form', 'unknown')} | {d['alpha']:.4f} | {d['r2']:.4f} | "
            f"{d.get('n_distinct_sizes', d.get('n_sizes'))} |"
            for r, d in pr.items()
        ]
        L.append("")
    route_notes = summary.get("route_notes") or {}
    if route_notes:
        L += ["## Route interpretation", ""]
        L += [f"- `{route}`: {note}." for route, note in route_notes.items()]
        L.append("")
    anc = summary.get("frontier_collapse")
    if anc and anc.get("testable"):
        verdict = _frontier_verdict(anc)
        L += ["## Frontier-collapse F-test", "",
              f"- F({anc['df1']},{anc['df2']}) = {anc['F']:.3f}, p = {anc['p_value']:.4g}",
              f"- verdict: {verdict}", ""]
    elif anc:
        L += ["## Frontier-collapse F-test", "", f"- not testable ({anc.get('reason')})", ""]
    es = summary.get("estep_structural")
    if es:
        L += ["## E-step structural-EM check", "",
              f"- arms: {es['n_arms']}",
              f"- Pearson(n_e_steps, final F/token) = {es['pearson_ne_final_f']:+.4f}  (expect strongly negative)",
              f"- Pearson(final F/token, test CE) = {es['pearson_final_f_test_ce']:+.4f}  (expect ~0 / >=0)", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def _make_figures(
    param_points: List[Dict[str, Any]],
    infer_points: List[Dict[str, Any]],
    fig_dir:      Path,

    *,
    weights_by_route:  Optional[Mapping[str, np.ndarray]]       = None,
    validation_points: Optional[List[Dict[str, Any]]]           = None,
    axis_routes:       Optional[Mapping[str, str]]              = None,
) -> None:
    try:
        from vfe3.viz import figures as figs
        figs.set_publication_style()
    except Exception as exc:
        logger.warning("figures unavailable (%s); skipping the figure pass", exc)
        return

    if validation_points is not None and axis_routes is not None:
        # PB-07: dispatch the two registered validation-metric figures through the declarative
        # FigureSpec seam. vfe3.viz.specs is imported lazily behind this try (the same graceful
        # degradation the figs import above gets), and the whole dispatch is best-effort: a failure
        # here skips only these two figures, never the legacy figure pass below.
        try:
            from vfe3.viz.specs import FigureSpec, emit_registered_figures
            SCALING_FIGURE_SPECS = (
                FigureSpec("capacity_scaling", "capacity_scaling.png",
                           lambda ctx: capacity_scaling_kwargs(ctx["validation_points"], ctx["axis_routes"])),
                FigureSpec("pareto_frontier", "pareto_frontier.png",
                           lambda ctx: pareto_frontier_kwargs(ctx["validation_points"])),
            )
            written = emit_registered_figures(
                SCALING_FIGURE_SPECS,
                {"validation_points": validation_points, "axis_routes": axis_routes},
                fig_dir,
            )
            for out_path in written:
                print(f"  figure -> {out_path}")
        except Exception as exc:
            logger.warning("registered validation figures skipped (%s)", exc)

    def _try(name: str, fn) -> None:
        try:
            fig = fn()
            figs.plt.close(fig)
            print(f"  figure -> {fig_dir / name}")
        except Exception as exc:
            logger.warning("figure %s failed (%s); skipped", name, exc)

    if len(param_points) >= 2:
        _try("scaling_ce_vs_params.png", lambda: figs.plot_scaling_law(
            param_points, x_key="n_params", xlabel="parameters N",
            with_offset=CONFIG["with_offset"], path=str(fig_dir / "scaling_ce_vs_params.png")))
        if len({p["route"] for p in param_points}) > 1:
            _try("scaling_routes_overlay.png", lambda: figs.plot_scaling_routes(
                param_points, x_key="n_params", xlabel="parameters N",
                with_offset=CONFIG["with_offset"], weights_by_route=weights_by_route,
                path=str(fig_dir / "scaling_routes_overlay.png")))
        # compute axis (a proxy): CE vs estimated FLOPs across the size grid (tokens fixed -> compute
        # grows with N). Labeled a proxy; uses the 6ND column.
        if any(np.isfinite(p["est_flops_6ND"]) for p in param_points):
            _try("scaling_ce_vs_flops.png", lambda: figs.plot_scaling_law(
                param_points, x_key="est_flops_6ND", xlabel="est FLOPs (6ND proxy)",
                with_offset=CONFIG["with_offset"], title="Scaling vs compute (6ND proxy)",
                path=str(fig_dir / "scaling_ce_vs_flops.png")))
        # data axis: only meaningful if tokens_seen actually varies (add a data route to populate it).
        toks = {p["tokens_seen"] for p in param_points if np.isfinite(p["tokens_seen"])}
        if len(toks) > 1:
            _try("scaling_ce_vs_tokens.png", lambda: figs.plot_scaling_law(
                param_points, x_key="tokens_seen", xlabel="tokens seen (data)",
                with_offset=CONFIG["with_offset"], title="Scaling vs data",
                path=str(fig_dir / "scaling_ce_vs_tokens.png")))
        # Pooled PPL offset law vs width (the June-27 headline result): PPL = E + A K^{-b} over ALL
        # parameter points, distinct from the per-arm kmup_stability split below. Uses the per-point
        # ppl_mean from aggregate_points; gated on >= 2 distinct widths.
        if len({_as_float(p.get("embed_dim")) for p in param_points
                if np.isfinite(_as_float(p.get("embed_dim")))}) > 1:
            _try("ppl_vs_embed_dim_offset.png", lambda: figs.plot_ppl_offset(
                param_points, path=str(fig_dir / "ppl_vs_embed_dim_offset.png")))

    # F1/EXP-6: muP width-stability -- grow_K (width-fixed) vs grow_K_mup's matched _fixed/_mup arms on
    # the shared K=embed_dim axis (test PPL per K + offset power-law fit, b annotated per arm). The
    # split is in _kmup_series so the |b_fixed - b_muP| contrast survives. Auto-emits when any arm has
    # >= 2 K cells.
    kmup = _kmup_series(param_points)
    if any(len(v) >= 2 for v in kmup.values()):
        _try("kmup_stability.png", lambda: figs.plot_kmup_stability(
            kmup, path=str(fig_dir / "kmup_stability.png")))

    if infer_points:
        # group the flat-N inference points by the knob they swept (n_e_steps / n_layers).
        series: Dict[str, List[Dict[str, Any]]] = {}
        for p in infer_points:
            knob = p["scale_knob"]
            x = p.get(knob, float("nan"))
            series.setdefault(knob, []).append({"x": _as_float(x), "ce_seeds": p["ce_seeds"]})
        n_flat = int(infer_points[0]["n_params"]) if np.isfinite(infer_points[0]["n_params"]) else None
        _try("inference_capacity.png", lambda: figs.plot_inference_capacity(
            series, n_params=n_flat, path=str(fig_dir / "inference_capacity.png")))

        # C2/EXP-5: the n_e_steps arms -> the F-vs-CE decorrelation scatter (needs f_mean + ce_mean)
        # and E-step-as-capacity (BPC + converged F vs T; additionally needs bpc_mean -- gated
        # separately so a heterogeneous run set missing test_bpc on some cells still gets the
        # decorrelation figure rather than a NaN BPC point).
        estep = sorted([p for p in infer_points if p["scale_knob"] == "n_e_steps"
                        and np.isfinite(p.get("f_mean", float("nan")))],
                       key=lambda q: q["n_e_steps"])
        if len(estep) >= 2:
            arms = [{"n_e_steps": p["n_e_steps"], "final_f": p["f_mean"], "ce": p["ce_mean"]}
                    for p in estep]
            _try("f_ce_decorrelation.png", lambda: figs.plot_f_ce_decorrelation(
                arms, path=str(fig_dir / "f_ce_decorrelation.png")))
            cap = [p for p in estep if np.isfinite(p.get("bpc_mean", float("nan")))]
            if len(cap) >= 2:
                _try("estep_capacity.png", lambda: figs.plot_estep_capacity(
                    [p["n_e_steps"] for p in cap], [p["bpc_mean"] for p in cap],
                    [p["f_mean"] for p in cap], n_params=n_flat,
                    path=str(fig_dir / "estep_capacity.png")))


if __name__ == "__main__":
    analyze()
