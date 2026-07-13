r"""Click-to-run verifier for the 2026-07-06 audit fixes (+ the kappa per-block figure feature).

Runs, in one shot, every new or changed test that pins an audit fix, and prints a clear GREEN/RED
banner. No CLI args -- edit the config below, then::

    python check_audit_fixes.py

The RTX 5090: set ``DEVICE = "cuda"`` (or export ``VFE3_TEST_DEVICE=cuda``) to run on the GPU;
the tests read ``VFE3_TEST_DEVICE`` at import, so it is set here BEFORE pytest imports them.

NOTE: ``tests/test_viz.py`` is deliberately NOT listed -- importing it triggers the umap/llvmlite
LLVM-JIT access violation on this Windows box (that crash IS audit finding t4/t5), which would abort
the whole run. The three fixes with no dedicated test are byte-identical / cosmetic / docs / hygiene
and are listed at the bottom for the record: m7, m14, m25, m30, h1 (and t4 above).
"""

import os

# ---- config (edit me) ------------------------------------------------------
DEVICE   = "cpu"          # "cpu" or "cuda" (RTX 5090)
VERBOSE  = True           # per-test PASS/FAIL lines
STOP_ON_FIRST_FAIL = False # True -> abort at the first failure
# ---------------------------------------------------------------------------

os.environ["VFE3_TEST_DEVICE"] = DEVICE     # tests read this at import time

import sys

from check_junit import run_pytest_junit

# Each entry: (finding, "path::test"). Parametrized tests expand to all their cases.
AUDIT_TESTS = [
    # ---- MAJOR M1-M4 ----
    ("M1", "tests/test_learnable_kappa.py::test_extractors_use_learned_kappa_in_iter_and_fe_kwargs"),
    ("M1", "tests/test_learnable_kappa.py::test_converged_state_beta_tracks_learned_kappa"),
    ("M2", "tests/test_train.py::test_phi_clamp_monitor_threshold_matches_transport_clamp"),
    ("M3", "tests/test_belief_cache.py::test_cache_supported_gates_result_changing_toggles"),
    ("M4", "tests/test_reporting_additions.py::test_gauge_transport_figure_aggregates_seeds"),
    ("M4", "tests/test_reporting_additions.py::test_mu_precond_figure_aggregates_seeds"),
    ("M4", "tests/test_reporting_additions.py::test_attention_entropy_figure_aggregates_seeds"),
    # ---- MINOR (m*) ----
    ("m4",  "tests/test_tier12_attention.py::test_gamma_prior_folded_in_diagnostic_replays"),
    ("m5",  "tests/test_extract.py::test_numerical_health_under_rope_does_not_raise"),
    ("m8",  "tests/test_free_energy.py::test_free_energy_entropy_exact_for_deep_finite_prior"),
    ("m10", "tests/test_tier12_estep.py::test_backprop_last_truncates_transport_gradient_to_phi"),
    ("m11", "tests/test_tier12_estep.py::test_straight_through_mean_trust_region_no_sigma_leak"),
    ("m12", "tests/test_tier12_estep.py::test_mm_exact_update_stays_put_on_saturated_row"),
    ("m16", "tests/test_transport.py::test_skew_transport_exp_not_clamped_for_large_phi"),
    ("m17", "tests/test_phi_preconditioner.py::test_killing_per_block_caches_parent_without_strong_retention"),
    ("m19", "tests/test_phi_preconditioner.py::test_pullback_series_warns_on_non_convergence"),
    ("m20", "tests/test_tier12_decode.py::test_z_loss_applied_on_dense_decode"),
    ("m26", "tests/test_run_artifacts.py::test_finalize_run_writes_test_results_and_figures"),
    ("m29", "tests/test_reporting_additions.py::test_offset_power_law_honors_weights"),  # SKIPS without scipy
    ("m31", "tests/test_train.py::test_parameter_report_leaves_global_rng_untouched"),
    # ---- TEST-SUITE (t*) ----
    ("t1",  "tests/test_audit_fixes_2026_06_13.py::test_gram_pinv_is_cached_and_value_identical"),
    ("t2",  "tests/test_train.py::test_train_vfe3_clickrun_importable_and_runs_one_step"),
    ("t7",  "tests/test_divergence.py::test_model_forward_under_new_divergence"),
    ("t8",  "tests/test_gauge_groups.py::test_full_model_logits_invariant_under_global_gauge"),
    # ---- kappa per-block + tau figure FEATURE ----
    ("feat", "tests/test_learnable_kappa.py::test_training_logs_per_block_kappa_and_tau"),
    ("feat", "tests/test_reporting_additions.py::test_kappa_block_trajectory_renders_per_block_kappa_and_tau"),
    ("feat", "tests/test_reporting_additions.py::test_save_figures_emits_kappa_block_trajectory"),
]

# Fixes with no dedicated test (verified by byte-identity / cosmetic / docs / hygiene, and t4 which
# lives in the crash-on-Windows test_viz.py): m7, m14, m25, m30, h1, t4.

if __name__ == "__main__":
    node_ids = [nid for _, nid in AUDIT_TESTS]
    args = node_ids + ["-p", "no:cacheprovider"]
    args += ["-v"] if VERBOSE else []
    args += ["-x"] if STOP_ON_FIRST_FAIL else []
    print(f"Running {len(node_ids)} requested audit-fix nodes on device={DEVICE!r} ...\n")
    code, counts = run_pytest_junit(args, prefix="vfe3-audit-fixes-")
    bar = "=" * 64
    print("\n" + bar)
    if code == 0:
        print(
            "AUDIT-FIX VERIFICATION: ALL GREEN  "
            f"({counts['passes']} passed, {counts['skipped']} skipped, "
            f"{counts['tests']} collected, device={DEVICE})"
        )
    else:
        print(
            "AUDIT-FIX VERIFICATION: FAILURES  "
            f"({counts['passes']} passed, {counts['failures']} failed, {counts['errors']} errors, "
            f"{counts['skipped']} skipped, pytest exit code {code}, device={DEVICE})"
        )
    print(bar)
    sys.exit(code)
