"""Contract tests for the paired phi pullback-group benchmark."""

from __future__ import annotations

import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from benchmarks import benchmark_phi_pullback_group as benchmark


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "benchmarks" / "benchmark_phi_pullback_group.py"


def test_benchmark_import_is_click_run_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing from the repository root neither runs CUDA nor parses arguments."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: (_ for _ in ()).throw(
        AssertionError("import attempted to inspect CUDA")
    ))

    namespace = runpy.run_path(str(_SCRIPT), run_name="phi_pullback_import_smoke")

    assert callable(namespace["main"])


def test_percentile_uses_pinned_linear_interpolation() -> None:
    assert benchmark._percentile([0.0, 10.0, 20.0, 30.0], 0.25) == 7.5
    assert benchmark._percentile([0.0, 10.0, 20.0, 30.0], 0.95) == pytest.approx(28.5)


def test_paired_bootstrap_is_deterministic_and_matches_literal_fixture() -> None:
    metric = [1.0, 2.0, 4.0]
    full = [2.0, 2.5, 5.0]

    first = benchmark._paired_bootstrap(
        metric,
        full,
        seed=7,
        resamples=4,
    )
    second = benchmark._paired_bootstrap(
        metric,
        full,
        seed=7,
        resamples=4,
    )

    assert first == second
    assert first["median_difference_ms"] == pytest.approx(1.0)
    assert first["median_difference_ci95_ms"] == pytest.approx([0.5375, 1.0])
    assert first["p95_difference_ms"] == pytest.approx(0.95)
    assert first["p95_difference_ci95_ms"] == pytest.approx([0.53375, 1.0])
    assert first["p95_full_to_metric_ratio_minus_one"] == pytest.approx(0.25)
    assert first["p95_full_to_metric_ratio_minus_one_ci95"] == pytest.approx(
        [0.25, 0.26875]
    )


def test_review_gate_uses_upper_confidence_bound_inclusively() -> None:
    assert benchmark._passes_review_gate([0.05, 0.20], maximum_overhead=0.20)
    assert not benchmark._passes_review_gate([0.05, 0.2000001], maximum_overhead=0.20)
    assert not benchmark._passes_review_gate([0.19, 0.21], maximum_overhead=0.20)


def test_cuda_timer_brackets_only_the_callable(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class FakeEvent:
        def __init__(self, name: str) -> None:
            self.name = name

        def record(self) -> None:
            events.append(f"{self.name}.record")

        def synchronize(self) -> None:
            events.append(f"{self.name}.synchronize")

        def elapsed_time(self, other: object) -> float:
            events.append(f"{self.name}.elapsed_time")
            return 3.25

    fake_events = iter((FakeEvent("start"), FakeEvent("end")))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: events.append("pre_sync"))
    monkeypatch.setattr(torch.cuda, "Event", lambda **kwargs: next(fake_events))

    elapsed = benchmark._measure_operation(
        lambda: events.append("call"),
        torch.device("cuda"),
    )

    assert elapsed == 3.25
    assert events == [
        "pre_sync",
        "start.record",
        "call",
        "end.record",
        "end.synchronize",
        "start.elapsed_time",
    ]


def test_cpu_helper_pairs_samples_alternates_order_and_calls_live_staging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    phi_pointers: list[int] = []

    def fake_metric(*args: object, **kwargs: object) -> torch.Tensor:
        calls.append("metric_only")
        assert isinstance(args[0], torch.Tensor)
        assert isinstance(args[1], torch.Tensor)
        assert args[0].dtype == torch.float32
        assert args[1].dtype == torch.float32
        phi_pointers.append(args[0].data_ptr())
        return torch.zeros((2, 8, 8), dtype=torch.float64)

    def fake_stage(*args: object, **kwargs: object) -> SimpleNamespace:
        calls.append("full_path")
        assert isinstance(args[1], torch.Tensor)
        assert args[1].dtype == torch.float32
        phi_pointers.append(args[1].data_ptr())
        return SimpleNamespace()

    monkeypatch.setattr(
        benchmark.phi_preconditioner,
        "pullback_metric_per_block",
        fake_metric,
    )
    monkeypatch.setattr(
        benchmark.gauge_optim,
        "stage_pullback_group_candidate",
        fake_stage,
    )

    payload = benchmark.run_benchmark(
        device=torch.device("cpu"),
        active_row_counts=[2],
        K=4,
        n_heads=2,
        warmups=2,
        repeats=2,
        bootstrap_seed=7,
        bootstrap_resamples=4,
    )

    case = payload["cases"][0]
    assert calls == [
        "metric_only", "full_path",
        "full_path", "metric_only",
        "metric_only", "full_path",
        "full_path", "metric_only",
    ]
    assert len(set(phi_pointers)) == 1
    assert len(case["samples"]["metric_only_ms"]) == 2
    assert len(case["samples"]["full_path_ms"]) == 2
    assert case["samples"]["call_order"] == [
        ["metric_only", "full_path"],
        ["full_path", "metric_only"],
    ]
    assert case["performance_gate"]["maximum_p95_scope_ratio_minus_one"] == 0.20
    assert payload["environment"]["device_type"] == "cpu"
    assert payload["environment"]["device_name"] == "cpu"
    assert payload["configuration"]["K"] == 4
    assert payload["configuration"]["irrep_dims"] == [2, 2]
    assert payload["configuration"]["warmup_pairs"] == 2
    assert payload["configuration"]["paired_repeats"] == 2
    assert payload["configuration"]["bootstrap_seed"] == 7
    assert payload["configuration"]["bootstrap_resamples"] == 4
    assert payload["tolerances"] == {
        "adaptive_series_minimum_order": 40,
        "adaptive_series_maximum_order": 128,
        "adaptive_series_order_increment": 8,
        "adaptive_series_relative_tail": 1e-12,
        "gram_relative_ridge": 1e-6,
        "minimum_undamped_generalized_eigenvalue": 1e-8,
        "maximum_damped_generalized_condition": 1e6,
        "maximum_scaled_solve_residual": 1e-10,
        "learning_rate": 0.015,
        "trust_radius": 0.1,
        "chart_max_norm": 5.0,
        "bch_order": 4,
        "bch_residual_max": 1e-6,
        "effective_group_product_residual_max": 1e-6,
        "max_backtracks": 10,
        "maximum_p95_scope_ratio_minus_one": 0.20,
    }
    assert case["scope_comparison"] == {
        "interpretation": "comparison of distinct callable scopes, not additive overhead",
        "metric_only_callable": (
            "vfe3.geometry.phi_preconditioner.pullback_metric_per_block"
        ),
        "full_path_callable": "vfe3.gauge_optim.stage_pullback_group_candidate",
    }
    assert set(payload["provenance"]) == {
        "git_head", "worktree_dirty", "status_porcelain", "script_sha256",
    }
    assert len(payload["provenance"]["git_head"]) == 40
    assert len(payload["provenance"]["script_sha256"]) == 64
    assert set(payload["environment"]) == {
        "device_type", "device_index", "device_name", "device_total_memory_bytes",
        "torch_version", "cuda_version", "python_version", "platform",
    }
    assert set(case["peak_allocated_bytes"]["metric_only"]) == {
        "baseline", "peak", "delta",
    }
    assert set(case["peak_allocated_bytes"]["full_path"]) == {
        "baseline", "peak", "delta",
    }
    assert case["full_path_certificates"] is None
    assert len(case["samples"]["pairs"]) == 2
    assert case["bootstrap"]["percentile_interpolation"] == "type_7_linear"
    assert case["bootstrap"]["common_paired_indices"] is True
    assert set(case["performance_gate"]) == {
        "statistic", "confidence_bound", "maximum_p95_scope_ratio_minus_one",
        "observed_ci95", "passed",
    }
    assert payload["all_cases_passed"] is case["performance_gate"]["passed"]
    assert "timing_threshold" not in case


def test_json_schema_accepts_complete_payload_and_rejects_missing_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        benchmark.phi_preconditioner,
        "pullback_metric_per_block",
        lambda *args, **kwargs: torch.zeros((2, 8, 8), dtype=torch.float64),
    )
    monkeypatch.setattr(
        benchmark.gauge_optim,
        "stage_pullback_group_candidate",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    payload = benchmark.run_benchmark(
        device=torch.device("cpu"),
        active_row_counts=[2],
        K=4,
        n_heads=2,
        warmups=0,
        repeats=2,
        bootstrap_seed=7,
        bootstrap_resamples=4,
    )

    benchmark.validate_payload(payload)
    del payload["cases"][0]["peak_allocated_bytes"]
    with pytest.raises(ValueError, match="peak_allocated_bytes"):
        benchmark.validate_payload(payload)


def test_failed_main_removes_stale_final_and_temporary_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "phi-pullback-success.json"
    temporary = output.with_suffix(output.suffix + ".tmp")
    output.write_text("stale final", encoding="utf-8")
    temporary.write_text("stale temporary", encoding="utf-8")
    monkeypatch.setattr(benchmark, "OUTPUT_PATH", output)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        benchmark.main()

    assert not output.exists()
    assert not temporary.exists()
