import ast
from pathlib import Path
from typing import Any

from tests.pytest_policy import (
    CUDA_TESTS,
    EXTERNAL_TESTS,
    RESOURCE_GROUPS,
    SLOW_TESTS,
    UMAP_TESTS,
    node_key,
    pytest_collection_modifyitems,
)


class _Config:
    def __init__(self, *, runslow: bool) -> None:
        self.runslow = runslow

    def getoption(self, name: str) -> bool:
        assert name == "--runslow"
        return self.runslow


class _Item:
    def __init__(self, nodeid: str) -> None:
        self.nodeid = nodeid
        self.markers: list[Any] = []

    def add_marker(self, marker: Any) -> None:
        self.markers.append(marker)

    @property
    def marker_names(self) -> list[str]:
        return [marker.mark.name for marker in self.markers]

    def marker_kwargs(self, name: str) -> dict[str, object]:
        return next(marker.mark.kwargs for marker in self.markers if marker.mark.name == name)


def test_node_key_normalizes_paths_and_parametrization() -> None:
    assert node_key(r"tests\test_example.py::test_case[value]") == "test_example.py::test_case"


def test_runslow_preserves_slow_marker_without_skip() -> None:
    item = _Item("tests/test_report.py::test_generate_figures_drives_live_model")
    pytest_collection_modifyitems(_Config(runslow=True), [item])
    assert item.marker_names == ["slow", "umap", "xdist_group"]
    assert item.marker_kwargs("xdist_group") == {"name": "umap"}


def test_default_lane_marks_semantics_before_slow_skip() -> None:
    item = _Item("tests/test_report.py::test_generate_figures_drives_live_model")
    pytest_collection_modifyitems(_Config(runslow=False), [item])
    assert item.marker_names == ["slow", "umap", "xdist_group", "skip"]
    assert item.marker_kwargs("skip") == {
        "reason": "slow integration test; pass --runslow to run it",
    }


def test_cuda_nodes_are_classified_and_resource_grouped() -> None:
    item = _Item("tests/test_laplace_family.py::test_laplace_cuda_matches_cpu")
    pytest_collection_modifyitems(_Config(runslow=False), [item])
    assert item.marker_names == ["cuda", "xdist_group"]
    assert item.marker_kwargs("xdist_group") == {"name": "cuda"}


def test_external_bundle_probe_is_classified_without_a_parallel_group() -> None:
    item = _Item(
        "tests/test_hierarchical_probabilistic_completeness_20260712.py"
        "::test_pure_route_bundle_is_byte_identical_to_branch_base"
    )
    pytest_collection_modifyitems(_Config(runslow=False), [item])
    assert item.marker_names == ["external"]


def test_policy_sets_define_disjoint_execution_lanes() -> None:
    assert (
        "test_round3_artifacts.py::test_emit_closes_figure_registered_by_raising_thunk"
        in UMAP_TESTS
    )
    assert UMAP_TESTS < SLOW_TESTS
    assert not (SLOW_TESTS & CUDA_TESTS)
    assert not (SLOW_TESTS & EXTERNAL_TESTS)
    assert not (CUDA_TESTS & EXTERNAL_TESTS)
    assert set(RESOURCE_GROUPS) == UMAP_TESTS | CUDA_TESTS
    assert {RESOURCE_GROUPS[key] for key in UMAP_TESTS} == {"umap"}
    assert {RESOURCE_GROUPS[key] for key in CUDA_TESTS} == {"cuda"}


def test_cuda_and_external_tables_cover_the_dedicated_prerequisite_lanes() -> None:
    assert CUDA_TESTS == {
        "test_generate.py::test_efe_rollout_sigma_mc_cuda_synthetic_pass",
        (
            "test_hierarchical_probabilistic_completeness_20260712.py"
            "::test_hierarchy_full_covariant_cuda_smoke"
        ),
        "test_laplace_family.py::test_laplace_cuda_matches_cpu",
        (
            "test_p3_pairwise_stats_reuse_20260711.py"
            "::test_p3_cuda_filtering_and_mm_reuse_smoke"
        ),
        (
            "test_phi_reflection_objective_parity_20260712.py"
            "::test_phi_reflection_objective_parity_cuda_smoke"
        ),
        "test_training_timing_20260711.py::test_real_cuda_training_timing_smoke",
    }
    assert EXTERNAL_TESTS == {
        (
            "test_hierarchical_probabilistic_completeness_20260712.py"
            "::test_pure_route_bundle_is_byte_identical_to_branch_base"
        ),
    }


def test_every_policy_node_resolves_to_a_top_level_test_without_importing_modules() -> None:
    tests_dir = Path(__file__).parent
    policy_nodes = SLOW_TESTS | UMAP_TESTS | CUDA_TESTS | EXTERNAL_TESTS | set(RESOURCE_GROUPS)
    functions_by_file: dict[str, set[str]] = {}

    for node in sorted(policy_nodes):
        filename, separator, function_name = node.partition("::")
        assert separator and function_name.startswith("test_"), node
        if filename not in functions_by_file:
            tree = ast.parse((tests_dir / filename).read_text(encoding="utf-8"))
            functions_by_file[filename] = {
                child.name
                for child in tree.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
        assert function_name in functions_by_file[filename], node
