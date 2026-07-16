import ast
from contextlib import contextmanager
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import pytest

from tests import conftest as test_conftest
from tests import pytest_policy
from tests.pytest_policy import (
    CUDA_TESTS,
    EXTERNAL_TESTS,
    RESOURCE_GROUPS,
    SLOW_TESTS,
    node_key,
    pytest_collection_modifyitems,
)


CUDA_MIRROR_TESTS: frozenset[str] = getattr(
    pytest_policy,
    "CUDA_MIRROR_TESTS",
    frozenset(),
)

EXPECTED_CUDA_MIRROR_TESTS: frozenset[str] = frozenset({
    "test_tier12_transport.py::test_per_head_transport_mean_matches_dense",
    "test_tier12_transport.py::test_per_head_transport_mean_rope_wrapped_matches_dense",
    "test_tier12_transport.py::test_stable_exp_norm_mode_small_norm_takes_fp32_path_exactly",
    "test_tier12_transport.py::test_stable_exp_norm_mode_large_norm_reenters_fp64_island",
    "test_tier12_estep.py::test_mm_exact_stationarity_folds_twohop",
    "test_tier12_estep.py::test_mm_exact_monotone_filtered_f_descent",
    "test_tier12_estep.py::test_twohop_zero_is_byte_identical",
    "test_tier12_estep.py::test_backprop_last_truncates_transport_gradient_to_phi",
    "test_omega_tilde_model_frame.py::test_phi_tilde_mm_exact_device_smoke",
    "test_tier12_attention.py::test_query_adaptive_tau_monotone_detached_and_c0_inert",
    "test_tier12_attention.py::test_twohop_term_matches_hand_computation",
    "test_tier12_decode.py::test_expected_likelihood_decode_matches_naive_dense",
    "test_tier12_decode.py::test_z_loss_full_chunked_matches_dense_lse",
    "test_divergence.py::test_safe_kl_clamp_bounds_and_nan",
    "test_free_energy.py::test_free_energy_entropy_exact_for_deep_finite_prior",
    "test_retraction.py::test_full_retraction_stays_spd",
})


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


class _TF32Backend:
    def __init__(self, *, modern: bool) -> None:
        self.modern = modern
        self._fp32_precision = "tf32"
        self._allow_tf32 = True
        self.fp32_sets: list[str] = []
        self.allow_sets: list[bool] = []
        self.allow_gets = 0

    @property
    def fp32_precision(self) -> str:
        if not self.modern:
            raise AttributeError("legacy TF32 backend")
        return self._fp32_precision

    @fp32_precision.setter
    def fp32_precision(self, value: str) -> None:
        if not self.modern:
            raise AssertionError("modern TF32 API used on a legacy backend")
        self.fp32_sets.append(value)
        self._fp32_precision = value

    @property
    def allow_tf32(self) -> bool:
        self.allow_gets += 1
        return self._allow_tf32

    @allow_tf32.setter
    def allow_tf32(self, value: bool) -> None:
        self.allow_sets.append(value)
        self._allow_tf32 = value


class _CudnnBackend(_TF32Backend):
    def __init__(self, *, modern: bool) -> None:
        super().__init__(modern=modern)
        self.deterministic = False
        self.benchmark = True


class _FakeTorch:
    def __init__(self, *, modern_tf32: bool) -> None:
        self.algorithms_enabled = False
        self.warn_only_enabled = True
        self.use_deterministic_calls: list[tuple[bool, bool]] = []
        self.matmul = _TF32Backend(modern=modern_tf32)
        self.cudnn = _CudnnBackend(modern=modern_tf32)
        self.backends = SimpleNamespace(
            cuda=SimpleNamespace(matmul=self.matmul),
            cudnn=self.cudnn,
        )

    @staticmethod
    def device(name: str) -> SimpleNamespace:
        return SimpleNamespace(type=name.partition(":")[0])

    def are_deterministic_algorithms_enabled(self) -> bool:
        return self.algorithms_enabled

    def is_deterministic_algorithms_warn_only_enabled(self) -> bool:
        return self.warn_only_enabled

    def use_deterministic_algorithms(self, enabled: bool, *, warn_only: bool) -> None:
        self.use_deterministic_calls.append((enabled, warn_only))
        self.algorithms_enabled = enabled
        self.warn_only_enabled = warn_only


def _source_tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _torch_import_line(tree: ast.Module) -> int:
    return next(
        node.lineno
        for node in tree.body
        if isinstance(node, ast.Import)
        and any(alias.name == "torch" for alias in node.names)
    )


def _environment_assignment(tree: ast.Module, name: str, value: str) -> ast.Assign:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not (
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Attribute)
            and isinstance(target.value.value, ast.Name)
            and target.value.value.id == "os"
            and target.value.attr == "environ"
            and isinstance(target.slice, ast.Constant)
            and target.slice.value == name
            and isinstance(node.value, ast.Constant)
            and node.value.value == value
        ):
            continue
        return node
    raise AssertionError(f"missing os.environ assignment for {name}")


def test_node_key_normalizes_paths_and_parametrization() -> None:
    assert node_key(r"tests\test_example.py::test_case[value]") == "test_example.py::test_case"


def test_runslow_preserves_slow_marker_without_skip() -> None:
    item = _Item("tests/test_report.py::test_finalize_skips_figures_when_disabled")
    pytest_collection_modifyitems(_Config(runslow=True), [item])
    assert item.marker_names == ["slow"]


def test_default_lane_marks_semantics_before_slow_skip() -> None:
    item = _Item("tests/test_report.py::test_finalize_skips_figures_when_disabled")
    pytest_collection_modifyitems(_Config(runslow=False), [item])
    assert item.marker_names == ["slow", "skip"]
    assert item.marker_kwargs("skip") == {
        "reason": "slow integration test; pass --runslow to run it",
    }


@pytest.mark.parametrize("requested_device", ["cpu", "cuda:0"])
def test_cuda_nodes_are_classified_and_resource_grouped(
    monkeypatch: pytest.MonkeyPatch,
    requested_device: str,
) -> None:
    monkeypatch.setenv("VFE3_TEST_DEVICE", requested_device)
    item = _Item("tests/test_laplace_family.py::test_laplace_cuda_matches_cpu")
    pytest_collection_modifyitems(_Config(runslow=False), [item])
    assert item.marker_names == ["cuda", "xdist_group"]
    assert item.marker_kwargs("xdist_group") == {"name": "cuda"}


def test_cuda_mirror_node_is_an_ordinary_cpu_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VFE3_TEST_DEVICE", "cpu")
    item = _Item(
        "tests/test_tier12_transport.py::test_per_head_transport_mean_matches_dense"
    )
    pytest_collection_modifyitems(_Config(runslow=False), [item])
    assert item.marker_names == []


def test_cuda_mirror_node_is_marked_and_grouped_for_indexed_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VFE3_TEST_DEVICE", "cuda:0")
    item = _Item(
        "tests/test_tier12_transport.py::test_per_head_transport_mean_matches_dense"
    )
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
    assert SLOW_TESTS == {
        "test_report.py::test_finalize_skips_figures_when_disabled",
        "test_run_artifacts.py::test_train_with_artifacts_writes_attention_pngs",
        (
            "test_run_diagnostics_2026_06_13.py"
            "::test_finalize_writes_tier3_research_and_provenance"
        ),
    }
    policy_sets = (SLOW_TESTS, CUDA_TESTS, CUDA_MIRROR_TESTS, EXTERNAL_TESTS)
    for left, right in combinations(policy_sets, 2):
        assert not (left & right)
    assert RESOURCE_GROUPS == {key: "cuda" for key in CUDA_TESTS}


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


def test_cuda_mirror_table_is_the_exact_curated_cohort() -> None:
    assert CUDA_MIRROR_TESTS == EXPECTED_CUDA_MIRROR_TESTS


def test_every_policy_node_resolves_to_a_top_level_test_without_importing_modules() -> None:
    tests_dir = Path(__file__).parent
    policy_nodes = (
        SLOW_TESTS
        | CUDA_TESTS
        | CUDA_MIRROR_TESTS
        | EXTERNAL_TESTS
        | set(RESOURCE_GROUPS)
    )
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


def test_device_resolves_indexed_cuda_before_availability_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str | None]] = []
    resolved = SimpleNamespace(type="cuda")

    def fake_device(name: str) -> SimpleNamespace:
        calls.append(("device", name))
        return resolved

    def fake_is_available() -> bool:
        calls.append(("available", None))
        return False

    monkeypatch.setenv("VFE3_TEST_DEVICE", "cuda:0")
    monkeypatch.setattr(test_conftest.torch, "device", fake_device)
    monkeypatch.setattr(test_conftest.torch.cuda, "is_available", fake_is_available)

    with pytest.raises(pytest.skip.Exception, match="CUDA requested but not available"):
        test_conftest.device.__wrapped__()

    assert calls == [("device", "cuda:0"), ("available", None)]


def test_deterministic_policy_fixture_is_session_autouse() -> None:
    fixture = getattr(test_conftest, "deterministic_cuda_policy", None)
    metadata = getattr(fixture, "_pytestfixturefunction", None) or getattr(
        fixture,
        "_fixture_function_marker",
        None,
    )
    assert metadata is not None
    assert metadata.scope == "session"
    assert metadata.autouse is True


def test_conftest_starts_cuda_policy_during_initialization_before_plugins() -> None:
    conftest_path = Path(__file__).with_name("conftest.py")
    tree = _source_tree(conftest_path)
    start_assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "_DETERMINISTIC_CUDA_POLICY_CONTEXT"
            for target in node.targets
        )
    )
    plugin_assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "pytest_plugins"
            for target in node.targets
        )
    )
    fixture_definition = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "deterministic_cuda_policy"
    )

    assert isinstance(start_assignment.value, ast.Call)
    assert isinstance(start_assignment.value.func, ast.Name)
    assert start_assignment.value.func.id == "_start_deterministic_cuda_policy"
    assert _torch_import_line(tree) < start_assignment.lineno
    assert start_assignment.lineno < plugin_assignment.lineno
    assert start_assignment.lineno < fixture_definition.lineno


def test_session_fixture_only_closes_already_started_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    @contextmanager
    def _recording_policy(
        _torch_module:          Any,
        _requested_device_name: str,
    ) -> Iterator[None]:
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    monkeypatch.setattr(
        test_conftest,
        "_deterministic_cuda_policy",
        _recording_policy,
    )
    starter = getattr(test_conftest, "_start_deterministic_cuda_policy", None)
    assert callable(starter)

    started_policy = starter(object(), "cuda:0")
    assert events == ["enter"]
    monkeypatch.setattr(
        test_conftest,
        "_DETERMINISTIC_CUDA_POLICY_CONTEXT",
        started_policy,
    )

    fixture_lifetime = test_conftest.deterministic_cuda_policy.__wrapped__()
    next(fixture_lifetime)
    assert events == ["enter"]
    with pytest.raises(StopIteration):
        next(fixture_lifetime)
    assert events == ["enter", "exit"]


def test_cpu_deterministic_policy_is_inert(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = getattr(test_conftest, "_deterministic_cuda_policy", None)
    assert callable(policy)
    fake_torch = _FakeTorch(modern_tf32=True)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    with policy(fake_torch, "cpu"):
        assert fake_torch.algorithms_enabled is False
        assert fake_torch.warn_only_enabled is True
        assert fake_torch.cudnn.deterministic is False
        assert fake_torch.cudnn.benchmark is True

    assert fake_torch.use_deterministic_calls == []
    assert fake_torch.matmul.fp32_sets == []
    assert fake_torch.matmul.allow_sets == []
    assert fake_torch.cudnn.fp32_sets == []
    assert fake_torch.cudnn.allow_sets == []


def test_cuda_deterministic_policy_requires_preimport_cublas_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = getattr(test_conftest, "_deterministic_cuda_policy", None)
    assert callable(policy)
    fake_torch = _FakeTorch(modern_tf32=True)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    with pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG.*before importing torch"):
        with policy(fake_torch, "cuda:0"):
            raise AssertionError("policy must reject the missing CUDA process setting")

    assert fake_torch.use_deterministic_calls == []


@pytest.mark.parametrize("modern_tf32", [True, False], ids=["modern", "legacy"])
@pytest.mark.parametrize("raises_inside", [False, True], ids=["normal", "exception"])
def test_cuda_deterministic_policy_applies_and_restores_exact_state(
    monkeypatch: pytest.MonkeyPatch,
    modern_tf32: bool,
    raises_inside: bool,
) -> None:
    policy = getattr(test_conftest, "_deterministic_cuda_policy", None)
    assert callable(policy)
    fake_torch = _FakeTorch(modern_tf32=modern_tf32)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    def assert_applied() -> None:
        assert fake_torch.algorithms_enabled is True
        assert fake_torch.warn_only_enabled is False
        assert fake_torch.cudnn.deterministic is True
        assert fake_torch.cudnn.benchmark is False
        if modern_tf32:
            assert fake_torch.matmul._fp32_precision == "ieee"
            assert fake_torch.cudnn._fp32_precision == "ieee"
        else:
            assert fake_torch.matmul._allow_tf32 is False
            assert fake_torch.cudnn._allow_tf32 is False

    if raises_inside:
        with pytest.raises(LookupError, match="sentinel"):
            with policy(fake_torch, "cuda:0"):
                assert_applied()
                raise LookupError("sentinel")
    else:
        with policy(fake_torch, "cuda:0"):
            assert_applied()

    assert fake_torch.algorithms_enabled is False
    assert fake_torch.warn_only_enabled is True
    assert fake_torch.cudnn.deterministic is False
    assert fake_torch.cudnn.benchmark is True
    assert fake_torch.use_deterministic_calls == [(True, False), (False, True)]
    if modern_tf32:
        assert fake_torch.matmul._fp32_precision == "tf32"
        assert fake_torch.cudnn._fp32_precision == "tf32"
        assert fake_torch.matmul.fp32_sets == ["ieee", "tf32"]
        assert fake_torch.cudnn.fp32_sets == ["ieee", "tf32"]
        assert fake_torch.matmul.allow_gets == 0
        assert fake_torch.cudnn.allow_gets == 0
        assert fake_torch.matmul.allow_sets == []
        assert fake_torch.cudnn.allow_sets == []
    else:
        assert fake_torch.matmul._allow_tf32 is True
        assert fake_torch.cudnn._allow_tf32 is True
        assert fake_torch.matmul.fp32_sets == []
        assert fake_torch.cudnn.fp32_sets == []
        assert fake_torch.matmul.allow_sets == [False, True]
        assert fake_torch.cudnn.allow_sets == [False, True]


def test_cuda_request_establishes_cublas_config_before_torch_import() -> None:
    conftest_path = Path(__file__).with_name("conftest.py")
    tree = _source_tree(conftest_path)
    assignment = _environment_assignment(
        tree,
        "CUBLAS_WORKSPACE_CONFIG",
        ":4096:8",
    )
    parent_by_child = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    validation_call = next(
        node
        for node in tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "_validate_cuda_preimport"
    )

    assert validation_call.lineno < assignment.lineno
    assert assignment.lineno < _torch_import_line(tree)
    conditional = parent_by_child[assignment]
    assert isinstance(conditional, ast.If)
    assert isinstance(conditional.test, ast.Call)
    assert isinstance(conditional.test.func, ast.Name)
    assert conditional.test.func.id == "_requests_cuda"


@pytest.mark.parametrize("initial_value", [None, ":16:8"], ids=["missing", "wrong"])
def test_cuda_preimport_validation_rejects_preloaded_torch_without_required_cublas(
    initial_value: str | None,
) -> None:
    validate = getattr(test_conftest, "_validate_cuda_preimport", None)
    assert callable(validate)
    environment = {}
    if initial_value is not None:
        environment["CUBLAS_WORKSPACE_CONFIG"] = initial_value

    with pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG.*before importing torch"):
        validate("cuda:0", environment, {"torch": object()})

    assert environment.get("CUBLAS_WORKSPACE_CONFIG") == initial_value


def test_cuda_preimport_validation_accepts_safe_or_cpu_processes() -> None:
    validate = getattr(test_conftest, "_validate_cuda_preimport", None)
    assert callable(validate)
    expected = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}

    validate("cuda:0", expected, {"torch": object()})
    validate("cuda:0", {}, {})
    validate("cpu", {}, {"torch": object()})


def test_gpu_runner_uses_only_the_canonical_serial_cuda_marker_lane() -> None:
    runner_path = Path(__file__).parents[1] / "check_gpu_tests.py"
    tree = _source_tree(runner_path)
    runner_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_pytest_junit"
    ]
    assert len(runner_calls) == 1
    command = ast.literal_eval(runner_calls[0].args[0])
    assert command == ["-m", "cuda", "-v", "-p", "no:cacheprovider"]
    assert "-n" not in command

    string_literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert not {
        value
        for value in string_literals
        if value.startswith("tests/") and "::test_" in value
    }


def test_gpu_runner_sets_cuda_process_environment_before_torch_import() -> None:
    runner_path = Path(__file__).parents[1] / "check_gpu_tests.py"
    tree = _source_tree(runner_path)
    torch_import_line = _torch_import_line(tree)
    device_assignment = _environment_assignment(tree, "VFE3_TEST_DEVICE", "cuda")
    cublas_assignment = _environment_assignment(
        tree,
        "CUBLAS_WORKSPACE_CONFIG",
        ":4096:8",
    )
    assert device_assignment.lineno < torch_import_line
    assert cublas_assignment.lineno < torch_import_line
