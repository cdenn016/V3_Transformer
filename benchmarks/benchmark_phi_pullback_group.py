r"""Paired RTX 5090 benchmark for strict phi pullback-group staging.

Edit the constants below and run this repository-root script directly. The
benchmark intentionally has no CLI parser. Importing it does not inspect CUDA
or execute measurements.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence

import torch


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import vfe3.gauge_optim as gauge_optim
import vfe3.geometry.phi_preconditioner as phi_preconditioner
from vfe3.geometry.groups import get_group


OUTPUT_PATH = (
    _REPO_ROOT / "docs" / "testing" / "2026-07-17-phi-pullback-group-rtx5090.json"
)
ACTIVE_ROW_COUNTS = [128, 512, 2_048]
K = 10
N_HEADS = 2
SEED = 20_260_717
BOOTSTRAP_SEED = 17_071_726
WARMUPS = 20
REPEATS = 100
BOOTSTRAP_RESAMPLES = 10_000

LEARNING_RATE = 0.015
TRUST_RADIUS = 0.1
CHART_MAX_NORM = 5.0
BCH_RESIDUAL_MAX = 1e-6
MAX_BACKTRACKS = 10
MAXIMUM_P95_SCOPE_RATIO_MINUS_ONE = 0.20

FIXED_TOLERANCES: Dict[str, float | int] = {
    "adaptive_series_minimum_order": 40,
    "adaptive_series_maximum_order": 128,
    "adaptive_series_order_increment": 8,
    "adaptive_series_relative_tail": 1e-12,
    "gram_relative_ridge": 1e-6,
    "minimum_undamped_generalized_eigenvalue": 1e-8,
    "maximum_damped_generalized_condition": 1e6,
    "maximum_scaled_solve_residual": 1e-10,
    "learning_rate": LEARNING_RATE,
    "trust_radius": TRUST_RADIUS,
    "chart_max_norm": CHART_MAX_NORM,
    "bch_order": 4,
    "bch_residual_max": BCH_RESIDUAL_MAX,
    "effective_group_product_residual_max": 1e-6,
    "max_backtracks": MAX_BACKTRACKS,
    "maximum_p95_scope_ratio_minus_one": MAXIMUM_P95_SCOPE_RATIO_MINUS_ONE,
}


def _percentile(values: Sequence[float], quantile: float) -> float:
    """Return the type-7 linearly interpolated percentile of finite values."""
    if not values:
        raise ValueError("percentile requires a nonempty sample")
    if not math.isfinite(quantile) or quantile < 0.0 or quantile > 1.0:
        raise ValueError(f"quantile must be finite and in [0, 1], got {quantile!r}")
    ordered = sorted(float(value) for value in values)
    if not all(math.isfinite(value) for value in ordered):
        raise ValueError("percentile samples must be finite")
    position = quantile * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _paired_bootstrap(
    metric_samples_ms: Sequence[float],
    full_samples_ms:   Sequence[float],

    *,
    seed:      int,
    resamples: int,
) -> Dict[str, Any]:
    """Bootstrap paired timing differences and the mandated p95 scope ratio."""
    metric = [float(value) for value in metric_samples_ms]
    full = [float(value) for value in full_samples_ms]
    if not metric or len(metric) != len(full):
        raise ValueError("paired bootstrap requires equal nonempty samples")
    if not all(math.isfinite(value) and value > 0.0 for value in metric):
        raise ValueError("metric-only timings must be finite and positive")
    if not all(math.isfinite(value) and value > 0.0 for value in full):
        raise ValueError("full-path timings must be finite and positive")
    if type(resamples) is not int or resamples < 1:
        raise ValueError("resamples must be a positive int")

    pair_differences = [full_value - metric_value for metric_value, full_value in zip(metric, full)]
    metric_p95 = _percentile(metric, 0.95)
    full_p95 = _percentile(full, 0.95)
    if not math.isfinite(metric_p95) or metric_p95 <= 0.0:
        raise ValueError("metric-only p95 must be finite and positive")

    generator = random.Random(seed)
    median_differences: List[float] = []
    p95_differences: List[float] = []
    p95_scope_ratios_minus_one: List[float] = []
    for _ in range(resamples):
        indices = [generator.randrange(len(metric)) for _ in range(len(metric))]
        metric_resample = [metric[index] for index in indices]
        full_resample = [full[index] for index in indices]
        difference_resample = [
            full[index] - metric[index]
            for index in indices
        ]
        metric_resample_p95 = _percentile(metric_resample, 0.95)
        full_resample_p95 = _percentile(full_resample, 0.95)
        if not math.isfinite(metric_resample_p95) or metric_resample_p95 <= 0.0:
            raise ValueError("bootstrapped metric-only p95 must be finite and positive")
        median_differences.append(_percentile(difference_resample, 0.50))
        p95_differences.append(full_resample_p95 - metric_resample_p95)
        p95_scope_ratios_minus_one.append(
            full_resample_p95 / metric_resample_p95 - 1.0
        )

    return {
        "seed": int(seed),
        "resamples": int(resamples),
        "confidence_level": 0.95,
        "percentile_interpolation": "type_7_linear",
        "common_paired_indices": True,
        "median_difference_ms": _percentile(pair_differences, 0.50),
        "median_difference_ci95_ms": [
            _percentile(median_differences, 0.025),
            _percentile(median_differences, 0.975),
        ],
        "p95_difference_ms": full_p95 - metric_p95,
        "p95_difference_ci95_ms": [
            _percentile(p95_differences, 0.025),
            _percentile(p95_differences, 0.975),
        ],
        "p95_full_to_metric_ratio_minus_one": full_p95 / metric_p95 - 1.0,
        "p95_full_to_metric_ratio_minus_one_ci95": [
            _percentile(p95_scope_ratios_minus_one, 0.025),
            _percentile(p95_scope_ratios_minus_one, 0.975),
        ],
    }


def _passes_review_gate(
    confidence_interval: Sequence[float],

    *,
    maximum_overhead: float,
) -> bool:
    """Accept only a finite upper confidence bound at or below the fixed limit."""
    if len(confidence_interval) != 2:
        raise ValueError("review-gate confidence interval must have two endpoints")
    lower, upper = (float(value) for value in confidence_interval)
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        return False
    return upper <= float(maximum_overhead)


def _measure_operation(
    operation: Callable[[], object],
    device:    torch.device,
) -> float:
    """Measure only one callable region using CUDA events or a CPU clock."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = operation()
        end.record()
        end.synchronize()
        elapsed_ms = float(start.elapsed_time(end))
        del result
        return elapsed_ms
    start_cpu = time.perf_counter()
    result = operation()
    elapsed_ms = (time.perf_counter() - start_cpu) * 1_000.0
    del result
    return elapsed_ms


def _warmup_operation(
    operation: Callable[[], object],
    device:    torch.device,
) -> None:
    """Run one untimed operation and complete it before the next arm."""
    result = operation()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    del result


def _peak_allocated_memory(
    operation: Callable[[], object],
    device:    torch.device,
) -> Dict[str, int | None]:
    """Measure isolated CUDA baseline, peak, and incremental allocated bytes."""
    if device.type != "cuda":
        return {"baseline": None, "peak": None, "delta": None}
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline = int(torch.cuda.memory_allocated(device))
    result = operation()
    torch.cuda.synchronize(device)
    peak = int(torch.cuda.max_memory_allocated(device))
    del result
    torch.cuda.empty_cache()
    return {
        "baseline": baseline,
        "peak": peak,
        "delta": max(0, peak - baseline),
    }


def _seeded_inputs(
    active_rows: int,
    group:       object,
    device:      torch.device,
    seed:        int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build immutable, seeded phi rows and outer-objective covectors."""
    coordinate_width = int(group.generators.shape[0])
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    phi = torch.randn(
        (active_rows, coordinate_width),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    grad_phi = torch.randn(
        (active_rows, coordinate_width),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    tiny = torch.finfo(phi.dtype).tiny
    phi = 0.25 * phi / torch.linalg.vector_norm(
        phi,
        dim=-1,
        keepdim=True,
    ).clamp_min(tiny)
    grad_phi = 0.10 * grad_phi / torch.linalg.vector_norm(
        grad_phi,
        dim=-1,
        keepdim=True,
    ).clamp_min(tiny)
    return phi.contiguous(), grad_phi.contiguous()


def _candidate_certificates(candidate: object) -> Dict[str, float | int]:
    """Host-reduce one untimed production candidate's numerical certificates."""
    direction = candidate.direction
    return {
        "series_order": int(direction.series_order),
        "minimum_undamped_generalized_eigenvalue": float(
            direction.min_undamped_generalized_eigenvalue.min()
        ),
        "maximum_undamped_generalized_condition": float(
            direction.undamped_generalized_condition.max()
        ),
        "maximum_damped_generalized_condition": float(
            direction.damped_generalized_condition.max()
        ),
        "maximum_scaled_solve_residual": float(direction.scaled_solve_residual.max()),
        "minimum_initial_trust_scale": float(candidate.trust_scale.min()),
        "maximum_backtracking_reductions": int(candidate.backtracking_reductions.max()),
        "maximum_candidate_chart_norm": float(candidate.candidate_chart_norm.max()),
        "maximum_group_product_residual": float(candidate.group_product_residual.max()),
    }


def _benchmark_case(
    active_rows: int,
    group:       object,
    device:      torch.device,

    *,
    seed:                int,
    warmups:             int,
    repeats:             int,
    bootstrap_seed:      int,
    bootstrap_resamples: int,
) -> Dict[str, Any]:
    """Measure one paired metric-only/full-staging scope comparison."""
    if active_rows < 1 or warmups < 0 or repeats < 1:
        raise ValueError("active_rows/repeats must be positive and warmups nonnegative")
    phi, grad_phi = _seeded_inputs(active_rows, group, device, seed)
    irrep_dims = list(group.irrep_dims)

    def metric_only() -> object:
        return phi_preconditioner.pullback_metric_per_block(
            phi,
            group.generators,
            irrep_dims,
            series_tol=1e-12,
            series_order=40,
        )

    def full_path() -> object:
        return gauge_optim.stage_pullback_group_candidate(
            grad_phi,
            phi,
            group,
            learning_rate=LEARNING_RATE,
            trust_radius=TRUST_RADIUS,
            chart_max_norm=CHART_MAX_NORM,
            bch_residual_max=BCH_RESIDUAL_MAX,
            phi_precond_mode="pullback_per_block",
            max_backtracks=MAX_BACKTRACKS,
        )

    operations = {
        "metric_only": metric_only,
        "full_path": full_path,
    }
    for pair_index in range(warmups):
        order = (
            ("metric_only", "full_path")
            if pair_index % 2 == 0
            else ("full_path", "metric_only")
        )
        for arm in order:
            _warmup_operation(operations[arm], device)

    metric_samples: List[float] = []
    full_samples: List[float] = []
    pairs: List[Dict[str, Any]] = []
    for pair_index in range(repeats):
        order = (
            ("metric_only", "full_path")
            if pair_index % 2 == 0
            else ("full_path", "metric_only")
        )
        observations: Dict[str, float] = {}
        for arm in order:
            observations[arm] = _measure_operation(operations[arm], device)
        metric_samples.append(observations["metric_only"])
        full_samples.append(observations["full_path"])
        pairs.append({
            "repeat": pair_index,
            "order": list(order),
            "metric_only_ms": observations["metric_only"],
            "full_path_ms": observations["full_path"],
        })

    bootstrap = _paired_bootstrap(
        metric_samples,
        full_samples,
        seed=bootstrap_seed,
        resamples=bootstrap_resamples,
    )
    confidence_interval = bootstrap["p95_full_to_metric_ratio_minus_one_ci95"]
    passed = _passes_review_gate(
        confidence_interval,
        maximum_overhead=MAXIMUM_P95_SCOPE_RATIO_MINUS_ONE,
    )
    metric_memory = _peak_allocated_memory(metric_only, device)
    full_memory = _peak_allocated_memory(full_path, device)
    if device.type == "cuda":
        candidate = full_path()
        torch.cuda.synchronize(device)
        full_path_certificates: Dict[str, float | int] | None = _candidate_certificates(
            candidate
        )
        del candidate
    else:
        full_path_certificates = None

    return {
        "active_rows": int(active_rows),
        "scope_comparison": {
            "interpretation": "comparison of distinct callable scopes, not additive overhead",
            "metric_only_callable": (
                "vfe3.geometry.phi_preconditioner.pullback_metric_per_block"
            ),
            "full_path_callable": "vfe3.gauge_optim.stage_pullback_group_candidate",
        },
        "samples": {
            "metric_only_ms": metric_samples,
            "full_path_ms": full_samples,
            "call_order": [pair["order"] for pair in pairs],
            "pairs": pairs,
        },
        "summary": {
            "metric_only_median_ms": float(statistics.median(metric_samples)),
            "metric_only_p95_ms": _percentile(metric_samples, 0.95),
            "full_path_median_ms": float(statistics.median(full_samples)),
            "full_path_p95_ms": _percentile(full_samples, 0.95),
        },
        "geometry_kernel_time_ms": {
            "median": float(statistics.median(metric_samples)),
            "p95": _percentile(metric_samples, 0.95),
        },
        "bootstrap": bootstrap,
        "peak_allocated_bytes": {
            "metric_only": metric_memory,
            "full_path": full_memory,
        },
        "full_path_certificates": full_path_certificates,
        "performance_gate": {
            "statistic": "p95(full_path_resample)/p95(metric_only_resample)-1",
            "confidence_bound": "upper_97.5_percentile",
            "maximum_p95_scope_ratio_minus_one": MAXIMUM_P95_SCOPE_RATIO_MINUS_ONE,
            "observed_ci95": confidence_interval,
            "passed": passed,
        },
    }


def _git_provenance() -> Dict[str, Any]:
    """Capture the exact source and local tracked-worktree state used for the run."""
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {
        "git_head": head,
        "worktree_dirty": bool(status),
        "status_porcelain": status,
        "script_sha256": script_hash,
    }


def _environment(device: torch.device) -> Dict[str, Any]:
    """Return JSON-safe runtime and accelerator identity fields."""
    if device.type == "cuda":
        index = torch.cuda.current_device() if device.index is None else device.index
        properties = torch.cuda.get_device_properties(index)
        device_name = torch.cuda.get_device_name(index)
        total_memory = int(properties.total_memory)
    else:
        index = None
        device_name = "cpu"
        total_memory = None
    return {
        "device_type": device.type,
        "device_index": index,
        "device_name": device_name,
        "device_total_memory_bytes": total_memory,
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


def run_benchmark(
    device: torch.device,

    *,
    active_row_counts:    Sequence[int],
    K:                    int,
    n_heads:              int,
    warmups:              int,
    repeats:              int,
    bootstrap_seed:       int,
    bootstrap_resamples:  int,
    seed:                 int = SEED,
) -> Dict[str, Any]:
    """Run paired cases and return the complete machine-readable payload."""
    group = get_group("block_glk")(
        K=K,
        n_heads=n_heads,
        dtype=torch.float32,
        device=device,
    )
    cases = [
        _benchmark_case(
            int(active_rows),
            group,
            device,
            seed=seed,
            warmups=warmups,
            repeats=repeats,
            bootstrap_seed=bootstrap_seed,
            bootstrap_resamples=bootstrap_resamples,
        )
        for active_rows in active_row_counts
    ]
    payload: Dict[str, Any] = {
        "schema_version": 1,
        "benchmark": "phi_pullback_group_scope_comparison",
        "environment": _environment(device),
        "provenance": _git_provenance(),
        "configuration": {
            "gauge_group": "block_glk",
            "K": int(K),
            "n_heads": int(n_heads),
            "irrep_dims": list(group.irrep_dims),
            "coordinate_width": int(group.generators.shape[0]),
            "active_row_counts": [int(value) for value in active_row_counts],
            "seed": int(seed),
            "warmup_pairs": int(warmups),
            "paired_repeats": int(repeats),
            "paired_alternation": True,
            "seeded_phi_embedded_frobenius_norm": 0.25,
            "seeded_covector_coordinate_norm": 0.10,
            "bootstrap_seed": int(bootstrap_seed),
            "bootstrap_resamples": int(bootstrap_resamples),
        },
        "tolerances": dict(FIXED_TOLERANCES),
        "cases": cases,
        "all_cases_passed": all(case["performance_gate"]["passed"] for case in cases),
    }
    validate_payload(payload)
    return payload


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    """Return a mapping or raise a schema-specific error."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def validate_payload(payload: Mapping[str, Any]) -> None:
    """Validate the fixed JSON schema and paired-sample invariants."""
    required_top = {
        "schema_version",
        "benchmark",
        "environment",
        "provenance",
        "configuration",
        "tolerances",
        "cases",
        "all_cases_passed",
    }
    missing_top = required_top.difference(payload)
    if missing_top:
        raise ValueError(f"payload missing fields: {sorted(missing_top)}")
    environment = _require_mapping(payload["environment"], "environment")
    for field in (
        "device_type",
        "device_index",
        "device_name",
        "device_total_memory_bytes",
        "torch_version",
        "cuda_version",
        "python_version",
        "platform",
    ):
        if field not in environment:
            raise ValueError(f"environment missing {field}")
    configuration = _require_mapping(payload["configuration"], "configuration")
    repeats = int(configuration.get("paired_repeats", 0))
    cases = payload["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a nonempty list")
    for case in cases:
        case_mapping = _require_mapping(case, "case")
        for field in (
            "active_rows",
            "scope_comparison",
            "samples",
            "summary",
            "geometry_kernel_time_ms",
            "bootstrap",
            "peak_allocated_bytes",
            "full_path_certificates",
            "performance_gate",
        ):
            if field not in case_mapping:
                raise ValueError(f"case missing {field}")
        samples = _require_mapping(case_mapping["samples"], "samples")
        metric = samples.get("metric_only_ms")
        full = samples.get("full_path_ms")
        orders = samples.get("call_order")
        pairs = samples.get("pairs")
        if not all(isinstance(values, list) for values in (metric, full, orders, pairs)):
            raise ValueError("paired sample arrays must be lists")
        if not (len(metric) == len(full) == len(orders) == len(pairs) == repeats):
            raise ValueError("paired sample arrays must match paired_repeats")
    try:
        json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("payload is not finite JSON") from error


def _validate_acceptance_regime(payload: Mapping[str, Any]) -> None:
    """Reject any real run that weakens the approved RTX 5090 regime."""
    environment = _require_mapping(payload["environment"], "environment")
    configuration = _require_mapping(payload["configuration"], "configuration")
    if environment["device_type"] != "cuda" or "RTX 5090" not in str(
        environment["device_name"]
    ):
        raise RuntimeError(
            "Task 6 requires an RTX 5090; observed "
            f"{environment['device_type']} {environment['device_name']}"
        )
    expected = {
        "K": K,
        "n_heads": N_HEADS,
        "active_row_counts": ACTIVE_ROW_COUNTS,
        "warmup_pairs": WARMUPS,
        "paired_repeats": REPEATS,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
    }
    for field, expected_value in expected.items():
        if configuration.get(field) != expected_value:
            raise RuntimeError(
                f"acceptance regime changed {field}: expected {expected_value!r}, "
                f"got {configuration.get(field)!r}"
            )
    if WARMUPS < 20 or REPEATS < 100 or BOOTSTRAP_RESAMPLES < 10_000:
        raise RuntimeError("acceptance constants weaken the approved benchmark")


def _temporary_output_path() -> Path:
    """Return the same-directory path used for atomic evidence publication."""
    return OUTPUT_PATH.with_suffix(OUTPUT_PATH.suffix + ".tmp")


def _remove_output_artifacts() -> None:
    """Remove only this benchmark's final and temporary evidence artifacts."""
    OUTPUT_PATH.unlink(missing_ok=True)
    _temporary_output_path().unlink(missing_ok=True)


def main() -> None:
    """Run all configured RTX 5090 cases and publish only a passing payload."""
    _remove_output_artifacts()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; Task 6 requires an RTX 5090")
        device = torch.device("cuda", torch.cuda.current_device())
        device_name = torch.cuda.get_device_name(device)
        if "RTX 5090" not in device_name:
            raise RuntimeError(f"Task 6 requires an RTX 5090; observed {device_name!r}")
        payload = run_benchmark(
            device,
            active_row_counts=ACTIVE_ROW_COUNTS,
            K=K,
            n_heads=N_HEADS,
            warmups=WARMUPS,
            repeats=REPEATS,
            bootstrap_seed=BOOTSTRAP_SEED,
            bootstrap_resamples=BOOTSTRAP_RESAMPLES,
            seed=SEED,
        )
        _validate_acceptance_regime(payload)
        serialized = json.dumps(payload, indent=2, allow_nan=False)
        if not payload["all_cases_passed"]:
            print(serialized)
            raise RuntimeError(
                "Task 6 performance gate failed; success JSON was not written"
            )
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = _temporary_output_path()
        temporary_path.write_text(serialized + "\n", encoding="utf-8")
        temporary_path.replace(OUTPUT_PATH)
        print(serialized)
    except BaseException:
        _remove_output_artifacts()
        raise


if __name__ == "__main__":
    main()
