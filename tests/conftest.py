import os

import pytest
import torch


@pytest.fixture
def device():
    # Tests are device-agnostic; default CPU for portability.
    # Set VFE3_TEST_DEVICE=cuda to run on the GPU.
    name = os.environ.get("VFE3_TEST_DEVICE", "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA requested but not available")
    return torch.device(name)


# ---------------------------------------------------------------------------
# Slow-test gate (CPU amenability).
# These figure / artifact / UMAP INTEGRATION tests each render the full
# single-run figure set end-to-end (dozens of matplotlib PNGs) or spawn the
# UMAP subprocess, so they dominate CPU wall time -- a single figure-driver
# test is ~90 s and the group is ~6-7 min of a ~13 min suite, which is what
# makes a plain `pytest` CPU run time out. Their model CONFIGS are already
# CPU-tiny (embed_dim=4, n_heads=2 -> GL(2), vocab_size=6); the cost is
# matplotlib rendering, NOT model size, so shrinking configs cannot help --
# gating is the right lever. They are therefore SKIPPED BY DEFAULT so a CPU
# `pytest` completes quickly; run them (CI / pre-merge / GPU) with:
#     pytest --runslow
# Correctness tests (training learnability gates, the forward-KL theorem) are
# intentionally NOT gated and still run by default. Listed by
# "<file>::<test>" so no per-file edits are needed; a renamed test simply falls
# back to running by default. See docs/2026-07-10-edits.md.
_SLOW_TESTS = frozenset({
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
    "test_model_channel_diagnostics_2026_06_13.py::test_generate_figures_emits_model_channel_figures",
    "test_model_channel_diagnostics_2026_06_13.py::test_finalize_emits_model_channel_terms_iff_active",
    "test_model_channel_diagnostics_2026_06_13.py::test_model_channel_bank_gating_and_umap_render",
    "test_july13_root_fixes.py::test_umap_worker_reuses_one_process_for_two_embeddings",
    "test_run_diagnostics_2026_06_13.py::test_finalize_writes_tier3_research_and_provenance",
})


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False,
        help="run the slow figure/artifact/UMAP integration tests (skipped by default)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: heavy figure/artifact/UMAP integration test; skipped unless --runslow"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="slow integration test; pass --runslow to run it")
    for item in items:
        # nodeid: "tests/test_report.py::test_x" (or "...::test_x[param]") on any OS
        tail = item.nodeid.replace("\\", "/").split("/")[-1]
        key = tail.split("[")[0]                              # drop any parametrization id
        if key in _SLOW_TESTS:
            item.add_marker(pytest.mark.slow)
            item.add_marker(skip_slow)
