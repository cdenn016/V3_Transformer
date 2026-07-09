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
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # keep parity with the rest of the suite; numpy
#   here pulls no torch, but the figures import vfe3.viz which may, so set it before any heavy import.

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("scaling_analysis")

INFERENCE_ROUTE = "inference"                                # flat-N routes, plotted separately from L(N)

CONFIG: Dict[str, Any] = {
    "input_dir":   "vfe3_scaling_results/blocks_K48",      # where scaling.py wrote the run dirs
    "with_offset": True,                                   # headline fit: False -> A*N^-alpha; True -> E + A*N^-alpha
    "n_bootstrap": 2000,                                    # nested (points x seeds) bootstrap for the exponent CI
    "min_points":  2,                                       # a route needs this many sizes to get its own fit
}

_CSV_COLUMNS = [
    "route", "scale_knob", "label", "seed", "n_params", "n_learnable_params", "embed_dim", "n_heads",
    "n_gen", "gauge_group", "n_layers", "n_e_steps", "family", "tokens_seen", "est_flops_6ND",
    "est_flops_analytic", "active_params_per_token", "test_ce", "test_ppl", "test_bpc",
    "estep_final_f_per_token", "best_val_ppl",
    "wall_time_s", "data_sha256", "git_sha",
]


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
            "test_bpc":    test.get("test_bpc", summ.get("test_bpc")),
            "estep_final_f_per_token": (summ.get("estep_final_f_per_token")          # C2/EXP-5 join
                                        if summ.get("estep_final_f_per_token") is not None
                                        else test.get("estep_final_f_per_token")),
            "best_val_ppl": (summ.get("best_val_ppl") if summ.get("best_val_ppl") is not None
                             else test.get("best_val_ppl")),
            "wall_time_s": summ.get("wall_time_s", sp.get("wall_time_s")),
            "data_sha256": prov.get("data_sha256"),
            "git_sha":     prov.get("git_sha"),
        })
    return rows


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
        # C2/EXP-5: per-seed converged final E-step F/token and test BPC (NaN when absent on older runs).
        f_seeds = [c for c in (_as_float(g.get("estep_final_f_per_token")) for g in grp) if np.isfinite(c)]
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
    r"""Nested cluster bootstrap of the power-law exponent: each replicate resamples the SIZE points
    with replacement (the lever-arm effect) and, within each kept point, resamples its seed CEs, then
    refits. Returns (alpha_hat, lo2.5, hi97.5). Needs >= 2 points; degenerate -> NaNs.

    ``weights`` (per-point, same order as ``points``) MUST match the headline fit's weighting so the
    point estimate and the CI come from ONE estimator -- otherwise a weighted headline alpha can sit
    outside its own unweighted CI (audit r2 id10). Resampled per replicate as ``weights[idx]``."""
    from vfe3.viz.figures import _fit_power_law
    xs = np.array([p["n_params"] for p in points], dtype=float)
    if xs.size < 2:
        return float("nan"), float("nan"), float("nan")
    seed_arrays = [np.asarray(p["ce_seeds"], dtype=float) for p in points]
    means = np.array([a.mean() for a in seed_arrays])
    alpha_hat = _fit_power_law(xs, means, weights=weights, with_offset=with_offset)["alpha"]
    rng = np.random.default_rng(0)                           # fixed -> reproducible CI
    boot: List[float] = []
    n = len(points)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        bx = xs[idx]
        by = np.array([float(rng.choice(seed_arrays[i], size=seed_arrays[i].size).mean()) for i in idx])
        bw = weights[idx] if weights is not None else None
        a = _fit_power_law(bx, by, weights=bw, with_offset=with_offset)["alpha"]
        if np.isfinite(a):
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
        by_route.setdefault(p["route"], []).append(p)
    routes = {r: ps for r, ps in by_route.items() if len(ps) >= min_points}
    out: Dict[str, Any] = {"routes": {r: len(ps) for r, ps in routes.items()}, "testable": False}
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
    if not rows:
        print(f"No runs (with summary.json) under {input_dir}; run scaling.py first.")
        return
    fig_dir = input_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    write_csv(rows, input_dir / "scaling_points.csv")
    print(f"\nVFE_3.0 scaling analysis\n  input:   {input_dir}\n  runs:    {len(rows)}"
          f"\n  csv:     {input_dir / 'scaling_points.csv'}")

    # provenance guards: a frontier must be one corpus + (ideally) one code state.
    shas = {r["data_sha256"] for r in rows if r.get("data_sha256")}
    gits = {r["git_sha"] for r in rows if r.get("git_sha")}
    if len(shas) > 1:
        logger.warning("harvested runs span %d distinct data_sha256 -- mixed corpus confounds the fit", len(shas))
    if len(gits) > 1:
        logger.warning("harvested runs span %d distinct git_sha -- code drift across the sweep", len(gits))

    points = aggregate_points(rows)
    param_points = [p for p in points if p["route"] != INFERENCE_ROUTE]
    infer_points = [p for p in points if p["route"] == INFERENCE_ROUTE]
    _print_points_table(points)

    # Accumulate the printed fits / tests into a persisted summary (scaling_summary.json plus the
    # SCALING_ANALYSIS.md report): the pooled fit, bootstrap CI, per-route exponents, the
    # frontier-collapse F-test, and the E-step structural-EM correlations were console-only.
    summary: Dict[str, Any] = {
        "input_dir": str(input_dir), "n_runs": len(rows), "with_offset": CONFIG["with_offset"],
        "n_param_points": len(param_points), "n_inference_points": len(infer_points),
        "pooled_fit": None, "per_route": {}, "frontier_collapse": None, "estep_structural": None,
    }

    # ---- L(N) power law + bootstrap exponent CI (pooled over the parameter routes) ----
    if len(param_points) >= 2:
        from vfe3.viz.figures import _fit_power_law
        xs = np.array([p["n_params"] for p in param_points], dtype=float)
        ys = np.array([p["ce_mean"] for p in param_points], dtype=float)
        sem = np.array([p["ce_sem"] for p in param_points], dtype=float)
        w = np.where(sem > 0, (ys / np.where(sem > 0, sem, 1.0)) ** 2, 1.0)
        fit = _fit_power_law(xs, ys, weights=w, with_offset=CONFIG["with_offset"])
        a_hat, lo, hi = bootstrap_exponent_ci(param_points, weights=w, n_boot=CONFIG["n_bootstrap"],
                                              with_offset=CONFIG["with_offset"])
        print(f"\nPOOLED L(N) fit ({fit['form']}):  alpha = {fit['alpha']:.4f}  "
              f"[95% CI {lo:.4f}, {hi:.4f}]   A = {fit['A']:.4g}   "
              + (f"E = {fit['E']:.4f}   " if CONFIG["with_offset"] else "")
              + f"R^2 = {fit['r2']:.4f}   over {fit['n_points']} sizes")
        summary["pooled_fit"] = {
            "form": fit["form"], "alpha": fit["alpha"], "alpha_ci": [lo, hi],
            "A": fit["A"], "E": fit.get("E"), "r2": fit["r2"], "n_points": fit["n_points"],
        }
        # ---- per-route fits + frontier-collapse F-test ----
        routes = sorted({p["route"] for p in param_points})
        if len(routes) > 1:
            print("\nper-route exponents:")
            for r in routes:
                pr = [p for p in param_points if p["route"] == r]
                if len(pr) >= 2:
                    fr = _fit_power_law(np.array([p["n_params"] for p in pr], dtype=float),
                                        np.array([p["ce_mean"] for p in pr], dtype=float))
                    print(f"  {r:<14} alpha={fr['alpha']:.4f}  R^2={fr['r2']:.4f}  ({len(pr)} sizes)")
                    summary["per_route"][r] = {"alpha": fr["alpha"], "r2": fr["r2"], "n_sizes": len(pr)}
                else:
                    print(f"  {r:<14} (only {len(pr)} size; need >= 2 to fit)")
            anc = ancova_frontier_collapse(param_points, min_points=CONFIG["min_points"])
            summary["frontier_collapse"] = anc
            if anc.get("testable"):
                verdict = ("ONE shared frontier (routes collapse)" if anc["collapses"]
                           else "routes DIVERGE (route-specific slope/intercept)")
                print(f"\nfrontier-collapse F-test:  F({anc['df1']},{anc['df2']}) = {anc['F']:.3f}  "
                      f"p = {anc['p_value']:.4g}  ->  {verdict}")
            else:
                print(f"\nfrontier-collapse F-test: not testable ({anc.get('reason')})")
    else:
        print("\n(only one parameter size present; add more sizes to fit L(N))")

    # ---- C2/EXP-5: structural non-Neal-Hinton EM -- F-vs-CE decorrelation across the n_e_steps arms ----
    estep_pts = sorted([p for p in infer_points if p["scale_knob"] == "n_e_steps"
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
        (input_dir / "scaling_summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8")
        _write_scaling_md(input_dir / "SCALING_ANALYSIS.md", summary)
        print(f"\nsummary -> {input_dir / 'scaling_summary.json'}"
              f"\n           {input_dir / 'SCALING_ANALYSIS.md'}")
    except Exception as exc:
        logger.warning("scaling summary write failed (%s); skipped", exc)

    # ---- figures (best-effort, never fatal) ----
    _make_figures(param_points, infer_points, fig_dir)
    print(f"\nfigures -> {fig_dir}")


def _write_scaling_md(path: Path, summary: Dict[str, Any]) -> None:
    r"""Render ``summary`` as a readable SCALING_ANALYSIS.md (the pooled fit, per-route exponents,
    frontier-collapse F-test, and E-step structural-EM correlations that were console-only)."""
    L = ["# VFE_3.0 scaling analysis", "",
         f"- input: `{summary.get('input_dir')}`",
         f"- runs: {summary.get('n_runs')}  (parameter points: {summary.get('n_param_points')}, "
         f"inference points: {summary.get('n_inference_points')})",
         f"- fit form: {'E + A N^-alpha (offset)' if summary.get('with_offset') else 'A N^-alpha'}", ""]
    pf = summary.get("pooled_fit")
    if pf:
        L += ["## Pooled L(N) power law", "",
              f"- exponent alpha = {pf['alpha']:.4f}  (95% CI [{pf['alpha_ci'][0]:.4f}, {pf['alpha_ci'][1]:.4f}])",
              f"- A = {pf['A']:.4g}" + (f", E = {pf['E']:.4f}"
                                        if summary.get("with_offset") and pf.get("E") is not None else ""),
              f"- R^2 = {pf['r2']:.4f} over {pf['n_points']} sizes", ""]
    pr = summary.get("per_route") or {}
    if pr:
        L += ["## Per-route exponents", "", "| route | alpha | R^2 | sizes |", "|---|---|---|---|"]
        L += [f"| {r} | {d['alpha']:.4f} | {d['r2']:.4f} | {d['n_sizes']} |" for r, d in pr.items()]
        L.append("")
    anc = summary.get("frontier_collapse")
    if anc and anc.get("testable"):
        verdict = ("one shared frontier (routes collapse)" if anc.get("collapses")
                   else "routes diverge (route-specific slope/intercept)")
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


def _make_figures(param_points: List[Dict[str, Any]], infer_points: List[Dict[str, Any]],
                  fig_dir: Path) -> None:
    try:
        from vfe3.viz import figures as figs
        figs.set_publication_style()
    except Exception as exc:
        logger.warning("figures unavailable (%s); skipping the figure pass", exc)
        return

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
                path=str(fig_dir / "scaling_routes_overlay.png")))
        # compute axis (a proxy): CE vs estimated FLOPs across the size grid (tokens fixed -> compute
        # grows with N). Labeled a proxy; uses the 6ND column.
        if any(np.isfinite(p["est_flops_6ND"]) for p in param_points):
            _try("scaling_ce_vs_flops.png", lambda: figs.plot_scaling_law(
                param_points, x_key="est_flops_6ND", xlabel="est FLOPs (6ND proxy)",
                title="Scaling vs compute (6ND proxy)", path=str(fig_dir / "scaling_ce_vs_flops.png")))
        # data axis: only meaningful if tokens_seen actually varies (add a data route to populate it).
        toks = {p["tokens_seen"] for p in param_points if np.isfinite(p["tokens_seen"])}
        if len(toks) > 1:
            _try("scaling_ce_vs_tokens.png", lambda: figs.plot_scaling_law(
                param_points, x_key="tokens_seen", xlabel="tokens seen (data)",
                title="Scaling vs data", path=str(fig_dir / "scaling_ce_vs_tokens.png")))
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
