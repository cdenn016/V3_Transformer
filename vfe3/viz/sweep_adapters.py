r"""Validation-metric adapters for the registered scaling figures (audit finding PB-07).

``scaling_analysis.aggregate_points`` (and the ``test_ce`` / ``test_ppl`` / ``test_bpc`` /
``bpc_mean`` fields it carries downstream) is the character-corrected TEST-set path: the L(N)
power-law fit, the frontier-collapse F-test, and every figure built from it read the held-out test
split, and none of that is touched here or ever relabeled as a validation quantity.
``plot_capacity_scaling`` and ``plot_pareto_frontier`` (``vfe3/viz/figures.py``) are about the
VALIDATION split instead, and previously rendered only from hand-built dicts in tests -- no
production driver fed them. This module is that seam: ``aggregate_validation_points`` collapses
harvested rows into one point per ``(route, label)`` keyed on the persisted ``best_val_ppl`` (never
on a test metric), and ``capacity_scaling_kwargs`` / ``pareto_frontier_kwargs`` turn those points
into the two plotters' exact kwargs.

The quantity plotted is ``log2(best_val_ppl)`` -- validation BITS PER TOKEN, not BPC: the persisted
scaling headline carries no tokens-per-character factor for the validation split, so labeling it
"bits per character" would be wrong.
"""

import logging
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_MIN_AXIS_POINTS   = 2                                        # a capacity-scaling axis needs >= 2 sizes
_MIN_PARETO_POINTS = 2                                        # a Pareto frontier needs >= 2 points
_INFERENCE_ROUTE   = "inference"                              # flat-N runs are not a parameter frontier


def _as_float(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def aggregate_validation_points(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    r"""Collapse seeds into one point per ``(route, label)``, keyed on persisted ``best_val_ppl``.

    A group is retained iff it carries at least one finite, positive ``best_val_ppl`` across its
    seeds -- test metrics (``test_ce`` / ``test_ppl`` / ``test_bpc``) are never consulted, so a
    validation-only row (every test metric null) still survives. ``val_bits_per_token_mean`` is the
    seed mean of ``log2(best_val_ppl)`` (validation bits/token; NOT the character-corrected BPC that
    ``scaling_analysis.aggregate_points`` computes from the test split). ``wall_time_mean`` is the
    seed mean of the persisted ``wall_time_s``.

    A row with an EXPLICIT ``best_val_ppl: None`` (the key is present, just null -- not merely
    absent) alongside a sibling seed carrying a finite value in the SAME group is a data-integrity
    fault, not an absent metric: this raises ``ValueError`` rather than silently averaging over the
    surviving seeds or falling back to a test metric.
    """
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((r["route"], r["label"]), []).append(r)

    points: List[Dict[str, Any]] = []
    for (route, label), grp in sorted(groups.items()):
        has_explicit_null = any("best_val_ppl" in g and g["best_val_ppl"] is None for g in grp)
        val_ppl = [v for v in (_as_float(g.get("best_val_ppl")) for g in grp) if np.isfinite(v) and v > 0.0]
        if has_explicit_null and val_ppl:
            raise ValueError(
                f"{route}/{label}: explicit-null best_val_ppl alongside {len(val_ppl)} finite "
                "seed(s) -- refusing to silently drop it or substitute a test metric"
            )
        if not val_ppl:
            continue
        bits_seeds = [float(np.log2(v)) for v in val_ppl]
        wall_seeds = [w for w in (_as_float(g.get("wall_time_s")) for g in grp) if np.isfinite(w)]
        npars      = [w for w in (_as_float(g.get("n_params")) for g in grp) if np.isfinite(w)]
        points.append({
            "route": route, "label": label, "scale_knob": grp[0]["scale_knob"],
            "n_params":   float(np.mean(npars)) if npars else float("nan"),
            "embed_dim":  _as_float(grp[0].get("embed_dim")),
            "n_heads":    _as_float(grp[0].get("n_heads")),
            "n_layers":   _as_float(grp[0].get("n_layers")),
            "n_e_steps":  _as_float(grp[0].get("n_e_steps")),
            "val_bits_per_token_seeds": bits_seeds,
            "val_bits_per_token_mean":  float(np.mean(bits_seeds)),
            "n_val_seeds": len(bits_seeds),
            "wall_time_mean": float(np.mean(wall_seeds)) if wall_seeds else float("nan"),
        })
    return points


def capacity_scaling_kwargs(
    points:      List[Dict[str, Any]],
    axis_routes: Mapping[str, str],
) -> Optional[Dict[str, Any]]:
    r"""Build ``plot_capacity_scaling``'s ``scaling`` kwarg from validation points, one axis at a
    time. A point enters ``axis`` only when BOTH ``point["route"] == axis_routes[axis]`` AND
    ``point["scale_knob"] == axis`` -- route equality alone is not enough because the ``inference``
    route carries both ``n_e_steps`` and ``n_layers`` cells, and only the latter belongs on the
    ``n_layers`` panel.

    An axis whose route never appears in ``points`` is simply omitted (that sweep was not run). An
    axis whose route DOES appear but yields fewer than ``_MIN_AXIS_POINTS`` finite points is a gap
    in a sweep this figure otherwise relies on: the whole figure is withheld (returns ``None``)
    rather than silently rendering a partial capacity-scaling figure, and the gap is logged.
    """
    reasons: List[str] = []
    scaling: Dict[str, Dict[str, np.ndarray]] = {}
    for axis, route in axis_routes.items():
        if not any(p["route"] == route for p in points):
            continue                                          # route never ran: omit, not an error
        axis_points = [
            p for p in points
            if p["route"] == route and p["scale_knob"] == axis
            and np.isfinite(p.get("val_bits_per_token_mean", float("nan")))
        ]
        if len(axis_points) < _MIN_AXIS_POINTS:
            reasons.append(f"{axis} (route={route!r}): only {len(axis_points)} finite point(s), "
                            f"need >= {_MIN_AXIS_POINTS}")
            continue
        ordered = sorted(axis_points, key=lambda p: p[axis])
        scaling[axis] = {
            "x":              np.array([p[axis] for p in ordered], dtype=float),
            "bits_per_token": np.array([p["val_bits_per_token_mean"] for p in ordered], dtype=float),
            "wall_time":      np.array([p["wall_time_mean"] for p in ordered], dtype=float),
        }
    if reasons:
        logger.warning("capacity_scaling withheld: %s", "; ".join(reasons))
        return None
    if not scaling:
        return None
    return {"scaling": scaling}


def pareto_frontier_kwargs(points: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    r"""Build ``plot_pareto_frontier``'s ``points`` kwarg from aggregated PARAMETER points -- the
    flat-N ``inference`` route is excluded, mirroring ``scaling_analysis``'s own
    ``param_points`` / ``INFERENCE_ROUTE`` split: a Pareto frontier over parameter count is not
    meaningful for cells whose parameter count never varies. Needs >= ``_MIN_PARETO_POINTS`` points
    carrying both a finite ``val_bits_per_token_mean`` and a finite ``n_params``; otherwise ``None``.
    """
    eligible = [
        p for p in points
        if p["route"] != _INFERENCE_ROUTE
        and np.isfinite(p.get("val_bits_per_token_mean", float("nan")))
        and np.isfinite(p.get("n_params", float("nan")))
    ]
    if len(eligible) < _MIN_PARETO_POINTS:
        return None
    ordered = sorted(eligible, key=lambda p: p["n_params"])
    return {
        "points": {
            "bits_per_token": np.array([p["val_bits_per_token_mean"] for p in ordered], dtype=float),
            "n_params":       np.array([p["n_params"] for p in ordered], dtype=float),
            "wall_time":      np.array([p["wall_time_mean"] for p in ordered], dtype=float),
            "label":          [p["label"] for p in ordered],
        }
    }
