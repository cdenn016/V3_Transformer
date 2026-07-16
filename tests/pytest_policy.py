import pytest


SLOW_TESTS: frozenset[str] = frozenset({
    "test_report.py::test_generate_figures_emits_s_channel_under_s_e_step",
    "test_report.py::test_finalize_autoruns_figures",
    "test_report.py::test_generate_figures_drives_live_model",
    "test_report.py::test_generate_figures_reloads_from_run_dir",
    "test_report.py::test_finalize_skips_figures_when_disabled",
    "test_run_artifacts.py::test_finalize_run_writes_test_results_and_figures",
    "test_run_artifacts.py::test_finalize_writes_gauge_geometry_figure",
    "test_run_artifacts.py::test_finalize_reloads_best_checkpoint",
    "test_run_artifacts.py::test_train_with_artifacts_writes_attention_pngs",
    "test_round3_artifacts.py::test_emit_closes_figure_registered_by_raising_thunk",
    "test_viz.py::test_plot_belief_umap_fallback_no_decode",
    "test_viz.py::test_plot_belief_umap_per_channel_categories",
    "test_viz.py::test_umap_embed_shape",
    (
        "test_model_channel_diagnostics_2026_06_13.py"
        "::test_generate_figures_emits_model_channel_figures"
    ),
    (
        "test_model_channel_diagnostics_2026_06_13.py"
        "::test_finalize_emits_model_channel_terms_iff_active"
    ),
    (
        "test_model_channel_diagnostics_2026_06_13.py"
        "::test_model_channel_bank_gating_and_umap_render"
    ),
    "test_july13_root_fixes.py::test_umap_worker_reuses_one_process_for_two_embeddings",
    (
        "test_run_diagnostics_2026_06_13.py"
        "::test_finalize_writes_tier3_research_and_provenance"
    ),
})

UMAP_TESTS: frozenset[str] = frozenset({
    "test_report.py::test_generate_figures_emits_s_channel_under_s_e_step",
    "test_report.py::test_finalize_autoruns_figures",
    "test_report.py::test_generate_figures_drives_live_model",
    "test_report.py::test_generate_figures_reloads_from_run_dir",
    "test_run_artifacts.py::test_finalize_run_writes_test_results_and_figures",
    "test_run_artifacts.py::test_finalize_writes_gauge_geometry_figure",
    "test_run_artifacts.py::test_finalize_reloads_best_checkpoint",
    "test_viz.py::test_plot_belief_umap_fallback_no_decode",
    "test_viz.py::test_plot_belief_umap_per_channel_categories",
    "test_viz.py::test_umap_embed_shape",
    (
        "test_model_channel_diagnostics_2026_06_13.py"
        "::test_generate_figures_emits_model_channel_figures"
    ),
    (
        "test_model_channel_diagnostics_2026_06_13.py"
        "::test_finalize_emits_model_channel_terms_iff_active"
    ),
    (
        "test_model_channel_diagnostics_2026_06_13.py"
        "::test_model_channel_bank_gating_and_umap_render"
    ),
    "test_july13_root_fixes.py::test_umap_worker_reuses_one_process_for_two_embeddings",
})

CUDA_TESTS: frozenset[str] = frozenset({
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
})

EXTERNAL_TESTS: frozenset[str] = frozenset({
    (
        "test_hierarchical_probabilistic_completeness_20260712.py"
        "::test_pure_route_bundle_is_byte_identical_to_branch_base"
    ),
})

RESOURCE_GROUPS: dict[str, str] = {
    **{key: "umap" for key in UMAP_TESTS},
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
        help="run the slow figure/artifact/UMAP integration tests (skipped by default)",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    run_slow = config.getoption("--runslow")
    skip_slow = pytest.mark.skip(reason="slow integration test; pass --runslow to run it")

    for item in items:
        key = node_key(item.nodeid)
        if key in SLOW_TESTS:
            item.add_marker(pytest.mark.slow)
        if key in UMAP_TESTS:
            item.add_marker(pytest.mark.umap)
        if key in CUDA_TESTS:
            item.add_marker(pytest.mark.cuda)
        if key in EXTERNAL_TESTS:
            item.add_marker(pytest.mark.external)
        if key in RESOURCE_GROUPS:
            item.add_marker(pytest.mark.xdist_group(name=RESOURCE_GROUPS[key]))
        if key in SLOW_TESTS and not run_slow:
            item.add_marker(skip_slow)
