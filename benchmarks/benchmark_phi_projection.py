r"""CUDA acceptance benchmark for the projected phi M-step.

Edit the constants below and run this file directly. The benchmark intentionally has no CLI parser.
It measures the complete global projection over the listed phi tables after warmup and writes a
machine-readable JSON record outside the repository.
"""

import json
import math
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vfe3.gauge_optim import (
    embedded_phi_frobenius_norm,
    phi_projection_chunk_rows,
    project_phi_parameter_rows_,
)
from vfe3.geometry.groups import get_group


OUTPUT_PATH = Path(r"C:\tmp\vfe3-phi-projection-benchmark-20260715.json")
WARMUPS = 5
REPEATS = 20
RADIUS = 4.0
HISTORICAL_K20_BOUNDED_STEP_MS = 340.7574
HISTORICAL_K20_UNBOUNDED_STEP_MS = 83.5599
BENCHMARK_CASES = [
    {"name": "k20_block_glk", "K": 20, "n_heads": 2, "table_rows": [50_257, 128]},
    {"name": "k240_block_glk", "K": 240, "n_heads": 8, "table_rows": [50_257, 128]},
]


def _percentile(values: Sequence[float], quantile: float) -> float:
    """Linearly interpolated percentile for a nonempty timing sample."""
    ordered = sorted(float(value) for value in values)
    position = quantile * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _benchmark_model(
    table_rows: Sequence[int],
    group,

    *,
    radius:    float,
    device:    torch.device,
) -> SimpleNamespace:
    """Build only the phi tables and group required by the projection contract."""
    coordinate_width = int(group.generators.shape[0])
    uniform = group.gram_diagonal_uniform()
    if uniform is None:
        raise ValueError("benchmark cases require a certified uniform diagonal generator Gram")
    value = 2.0 * radius / math.sqrt(coordinate_width * uniform)
    tables = [
        torch.full(
            (int(rows), coordinate_width),
            value,
            device=device,
            dtype=torch.float32,
        )
        for rows in table_rows
    ]
    padded: List[torch.Tensor | None] = tables + [None] * (4 - len(tables))
    return SimpleNamespace(
        group=group,
        prior_bank=SimpleNamespace(phi_embed=padded[0], s_phi_embed=padded[2]),
        pos_phi_free=padded[1],
        s_pos_phi_free=padded[3],
    )


def _timed_projection(
    model:  SimpleNamespace,
    radius: float,
    device: torch.device,
) -> float:
    """Time one silent projection without introducing diagnostic synchronization inside it."""
    if device.type == "cuda":
        with torch.cuda.device(device):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            project_phi_parameter_rows_(model, radius, collect_stats=False)
            end.record()
            end.synchronize()
            return float(start.elapsed_time(end))
    start_cpu = time.perf_counter()
    project_phi_parameter_rows_(model, radius, collect_stats=False)
    return (time.perf_counter() - start_cpu) * 1000.0


def _maximum_embedded_norm(model: SimpleNamespace) -> float:
    """Reduce the post-projection maximum without materializing embedded matrices."""
    tables = [
        model.prior_bank.phi_embed,
        model.prior_bank.s_phi_embed,
        model.pos_phi_free,
        model.s_pos_phi_free,
    ]
    maximum = torch.zeros(
        (),
        device=model.group.generators.device,
        dtype=model.group.generators.dtype,
    )
    for table in tables:
        if table is None:
            continue
        rows = table.reshape(-1, table.shape[-1])
        chunk_rows = phi_projection_chunk_rows(
            rows.shape[-1],
            model.group.generators.shape[-1],
            rows.element_size(),
        )
        for start in range(0, rows.shape[0], chunk_rows):
            norms = embedded_phi_frobenius_norm(
                rows[start:start + chunk_rows],
                model.group,
            )
            maximum.copy_(torch.maximum(maximum, norms.max()))
    return float(maximum)


def benchmark_projection_case(
    *,
    name:       str,
    K:          int,
    n_heads:    int,
    table_rows: Sequence[int],
    radius:     float,
    device:     torch.device,
    warmups:    int,
    repeats:    int,
) -> Dict[str, object]:
    """Benchmark one full-table projection case and return a JSON-serializable record."""
    if warmups < 0 or repeats < 1:
        raise ValueError("warmups must be nonnegative and repeats must be positive")
    group = get_group("block_glk")(
        K=K,
        n_heads=n_heads,
        dtype=torch.float32,
        device=device,
    )
    model = _benchmark_model(table_rows, group, radius=radius, device=device)
    for _ in range(warmups):
        _timed_projection(model, radius, device)
    samples = [_timed_projection(model, radius, device) for _ in range(repeats)]
    maximum_post = _maximum_embedded_norm(model)
    projection_median = float(statistics.median(samples))
    result: Dict[str, object] = {
        "name": name,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
        "torch_version": torch.__version__,
        "K": K,
        "n_heads": n_heads,
        "coordinate_width": int(group.generators.shape[0]),
        "table_rows": [int(rows) for rows in table_rows],
        "total_rows": int(sum(table_rows)),
        "route": group.phi_norm_route(),
        "radius": radius,
        "warmups": warmups,
        "repeats": repeats,
        "projection_median_ms": projection_median,
        "projection_p95_ms": _percentile(samples, 0.95),
        "projection_samples_ms": samples,
        "maximum_post_projection_norm": maximum_post,
        "disabled_control_median_ms": None,
        "disabled_control_p95_ms": None,
    }
    if K == 20:
        result.update({
            "historical_bounded_step_ms": HISTORICAL_K20_BOUNDED_STEP_MS,
            "historical_unbounded_step_ms": HISTORICAL_K20_UNBOUNDED_STEP_MS,
            "historical_slowdown_ratio": (
                HISTORICAL_K20_BOUNDED_STEP_MS / HISTORICAL_K20_UNBOUNDED_STEP_MS
            ),
            "estimated_step_overhead_ratio": (
                (HISTORICAL_K20_UNBOUNDED_STEP_MS + projection_median)
                / HISTORICAL_K20_UNBOUNDED_STEP_MS
            ),
        })
    return result


def main() -> None:
    """Run the configured CUDA cases and write one machine-readable benchmark artifact."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the configured K20/K240 acceptance benchmark")
    device = torch.device("cuda")
    results = []
    for case in BENCHMARK_CASES:
        results.append(benchmark_projection_case(
            **case,
            radius=RADIUS,
            device=device,
            warmups=WARMUPS,
            repeats=REPEATS,
        ))
        torch.cuda.empty_cache()
    payload = {
        "benchmark": "phi_projection_hot_path",
        "historical_evidence": {
            "k20_bounded_step_ms": HISTORICAL_K20_BOUNDED_STEP_MS,
            "k20_unbounded_step_ms": HISTORICAL_K20_UNBOUNDED_STEP_MS,
        },
        "cases": results,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
