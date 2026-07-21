import os

import pytest
import torch


SLOW_TESTS: frozenset[str] = frozenset({
    "test_report.py::test_finalize_skips_figures_when_disabled",
    "test_run_artifacts.py::test_train_with_artifacts_writes_attention_pngs",
    (
        "test_run_diagnostics_2026_06_13.py"
        "::test_finalize_writes_tier3_research_and_provenance"
    ),
})

CUDA_TESTS: frozenset[str] = frozenset({
    (
        "test_audit_diagnostic_memory_20260720.py"
        "::test_cuda_diagnostic_snapshot_peak_is_bounded_to_one_sequence"
    ),
    (
        "test_final_audit_integrity_20260716.py"
        "::test_cuda_custom_optimizer_resume_preserves_cpu_control_state"
    ),
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
})

CUDA_MIRROR_TESTS: frozenset[str] = frozenset({
    (
        "test_audit_full_gaussian_numerics_20260720.py"
        "::test_full_gaussian_self_kl_cuda_mirror"
    ),
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

EXTERNAL_TESTS: frozenset[str] = frozenset({
    (
        "test_hierarchical_probabilistic_completeness_20260712.py"
        "::test_pure_route_bundle_is_byte_identical_to_branch_base"
    ),
})

RESOURCE_GROUPS: dict[str, str] = {
    **{key: "cuda" for key in CUDA_TESTS},
}


def node_key(nodeid: str) -> str:
    """Return the file/function policy key for a collected pytest node ID."""
    tail = nodeid.replace("\\", "/").split("/")[-1]
    return tail.split("[")[0]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run the slow figure/artifact integration tests (skipped by default)",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    run_slow = config.getoption("--runslow")
    skip_slow = pytest.mark.skip(reason="slow integration test; pass --runslow to run it")
    mirror_cuda = torch.device(os.environ.get("VFE3_TEST_DEVICE", "cpu")).type == "cuda"

    for item in items:
        key = node_key(item.nodeid)
        if key in SLOW_TESTS:
            item.add_marker(pytest.mark.slow)
        if key in CUDA_TESTS or (mirror_cuda and key in CUDA_MIRROR_TESTS):
            item.add_marker(pytest.mark.cuda)
        if key in EXTERNAL_TESTS:
            item.add_marker(pytest.mark.external)
        if key in RESOURCE_GROUPS:
            item.add_marker(pytest.mark.xdist_group(name=RESOURCE_GROUPS[key]))
        elif mirror_cuda and key in CUDA_MIRROR_TESTS:
            item.add_marker(pytest.mark.xdist_group(name="cuda"))
        if key in SLOW_TESTS and not run_slow:
            item.add_marker(skip_slow)
