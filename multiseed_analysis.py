r"""Across-seed aggregation for the EXP-1 multi-seed variance floor (I1).

``train_vfe3.py`` with ``NUM_RUNS``/``SEEDS`` writes one seed-labelled run dir per seed, each with a
``summary.json`` (headline ``test_ppl``/``best_val_ppl``) and a ``config.json`` (the ``seed``).
Nothing aggregated the seeds into a mean +/- SD error bar -- this does, and flags ablation cells whose
between-cell spread is within the seed-noise band (so a "win" is not read off seed noise).

Click-to-run (project policy: no argparse): edit ``CONFIG`` below, then ``python multiseed_analysis.py``.

Caveat the printout repeats: the per-run reseed (train_vfe3) currently shares the data-shuffle order
across seeds, so this SD is the init+optimization spread only -- a LOWER BOUND on deployment variance.
A fixed data-order generator (so model-init RNG varies while the batch order is held) is the companion
fix; see docs/experiments/2026-06-21-experiment-readiness.md (S6).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


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
    are NaN for n<2. ``seeds`` reads the sibling ``config.json`` ``seed`` (None if absent). Non-finite
    or unreadable points are skipped (never crash the aggregation).
    """
    root = Path(run_root)
    values: List[float] = []
    seeds: List[Optional[int]] = []
    for f in sorted(root.rglob(filename)):
        v = _as_finite_float(_read_json(f).get(key))
        if v is None:
            continue
        values.append(v)
        s = _read_json(f.parent / seed_file).get("seed")
        seeds.append(int(s) if isinstance(s, (int, float)) else None)

    n = len(values)
    if n == 0:
        return {"n": 0, "mean": math.nan, "sd": math.nan, "two_sd": math.nan,
                "cv": math.nan, "values": [], "seeds": []}
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) if n >= 2 else math.nan
    cv = (sd / abs(mean)) if (n >= 2 and mean != 0.0) else math.nan
    return {"n": n, "mean": mean, "sd": sd, "two_sd": 2.0 * sd, "cv": cv,
            "values": values, "seeds": seeds}


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
    "run_root": "vfe3_runs",   # dir holding the per-seed run folders (train_vfe3 NUM_RUNS/SEEDS output)
    "key":      "test_ppl",    # headline metric in summary.json (test_ppl | best_val_ppl | test_ce ...)
}


def main() -> None:
    out = aggregate_seed_metric(CONFIG["run_root"], CONFIG["key"])
    if out["n"] == 0:
        print(f"no '{CONFIG['key']}' found under {CONFIG['run_root']!r} (looked for summary.json)")
        return
    print(f"\nAcross-seed {CONFIG['key']} over {out['n']} run(s)  (seeds: {out['seeds']})")
    print(f"  mean    = {out['mean']:.4f}")
    print(f"  SD      = {out['sd']:.4f}  (ddof=1)")
    print(f"  +/-1 SD = [{out['mean'] - out['sd']:.4f}, {out['mean'] + out['sd']:.4f}]")
    print(f"  2*SD    = {out['two_sd']:.4f}")
    print(f"  CV      = {out['cv']:.4f}  ({100 * out['cv']:.2f}% of mean)")
    print("  NOTE: data order is shared across seeds (per-run reseed), so this SD is the "
          "init+optimization spread only -- a LOWER BOUND on deployment variance.")


if __name__ == "__main__":
    main()
