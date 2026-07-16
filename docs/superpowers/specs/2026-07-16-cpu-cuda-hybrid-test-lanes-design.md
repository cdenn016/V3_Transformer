# CPU/CUDA Hybrid Test-Lanes Design

## Status and scope

The user approved this design after the source-only CUDA suitability investigation. The work has two explicit goals: delete every test in the policy-defined slow UMAP cohort, accepting the resulting loss of native UMAP and incidental report/finalization coverage, and implement a hybrid execution model that uses all twelve physical cores of the AMD Ryzen 9 9900X for the broad CPU lane while exercising selected numerical contracts on the RTX 5090.

The implementation does not remove production UMAP support, the `umap-learn` visualization dependency, lightweight mocked worker tests, pure report-planning tests, CPU goldens, finite-difference or analytic oracles, exact same-device identities, or host-side artifact and serialization tests outside the eleven approved nodes. It does not convert the entire suite to CUDA.

## Approved UMAP deletion boundary

The deleted cohort is the complete eleven-node `UMAP_TESTS` table at the approved source commit. Four nodes directly test native UMAP embedding, plotting, or subprocess reuse. Seven nodes test broader figure, report, finalization, checkpoint-reload, and cleanup behavior while executing native UMAP incidentally. The user explicitly accepts losing all eleven contracts.

The deleted nodes are:

- `test_viz.py::test_umap_embed_shape`;
- `test_viz.py::test_plot_belief_umap_per_channel_categories`;
- `test_viz.py::test_plot_belief_umap_fallback_no_decode`;
- `test_july13_root_fixes.py::test_umap_worker_reuses_one_process_for_two_embeddings`;
- `test_round3_artifacts.py::test_emit_closes_figure_registered_by_raising_thunk`;
- `test_report.py::test_generate_figures_drives_live_model`;
- `test_report.py::test_generate_figures_reloads_from_run_dir`;
- `test_report.py::test_finalize_autoruns_figures`;
- `test_run_artifacts.py::test_finalize_run_writes_test_results_and_figures`;
- `test_run_artifacts.py::test_finalize_writes_gauge_geometry_figure`;
- `test_run_artifacts.py::test_finalize_reloads_best_checkpoint`.

The `UMAP_TESTS` policy, `umap` pytest marker, UMAP resource group, and orphaned finalized-artifact fixture/evidence type are removed. The remaining slow lane contains only three non-UMAP integrations. Historical audit and design records remain unchanged because they describe the repository state at their recorded commits.

## CPU execution architecture

A click-to-run `run_cpu_tests.py` owns the ordinary CPU verification union. Its editable configuration uses twelve xdist workers for the fast lane, matching the Ryzen 9 9900X's twelve physical cores, and at most three workers for the three-node slow lane. It invokes each lane in a fresh child process, passes an explicit integer rather than `-n auto`, writes JUnit XML to the operating-system temporary directory, stops after the first failed lane, and reports counts only by parsing JUnit.

The child environment pins `VFE3_TEST_DEVICE=cpu` and `CUDA_VISIBLE_DEVICES=-1`. It sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `NUMBA_NUM_THREADS`, `BLIS_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS` to one before workers import numerical libraries. This produces one independently scheduled test process per physical core instead of multiplying native thread pools inside every worker. The runner validates its configured worker counts against `os.cpu_count()` and fails before pytest when the configuration is invalid.

The fast lane uses `--dist loadscope` and excludes `slow`, `cuda`, and `external`. The slow lane uses `--runslow`, selects only `slow and not cuda and not external`, and uses no more workers than selected nodes. CUDA and external prerequisites remain separate serial processes. Parallel branch coverage uses the same twelve-worker fast lane plus the three-worker slow complement.

## Hybrid CUDA architecture

One canonical pytest policy supplies the six CUDA-only node IDs and a curated CUDA-mirror node table. `tests/pytest_policy.py`, `check_gpu_tests.py`, lane documentation, and policy meta-tests consume that table instead of maintaining duplicate CUDA lists.

Two CUDA classes remain distinct in the policy:

1. `CUDA_TESTS` contains hardware-only tests that always receive the `cuda` marker and are excluded from CPU execution.
2. `CUDA_MIRROR_TESTS` contains ordinary CPU contracts that receive the `cuda` marker and CUDA resource group only when `VFE3_TEST_DEVICE` resolves to a CUDA device. They remain unmarked and normally scheduled in CPU runs.

The initial mirror is deliberately curated rather than inferred from every fixture user. It contains four transport and matrix-exponential contracts, four E-step/MM/two-hop/gradient contracts, one phi-tilde model-frame residency contract, two attention contracts, two decode contracts, and representative divergence, canonical free-energy, and full-SPD retraction contracts. The final three are converted to the shared device fixture using CPU-seeded inputs moved to the selected device. CPU-created inputs preserve the mathematical sample across devices; CUDA results use existing same-device identities or operation-specific tolerances rather than global tolerance relaxation.

The sixteen mirror node IDs are:

- `test_tier12_transport.py::test_per_head_transport_mean_matches_dense`;
- `test_tier12_transport.py::test_per_head_transport_mean_rope_wrapped_matches_dense`;
- `test_tier12_transport.py::test_stable_exp_norm_mode_small_norm_takes_fp32_path_exactly`;
- `test_tier12_transport.py::test_stable_exp_norm_mode_large_norm_reenters_fp64_island`;
- `test_tier12_estep.py::test_mm_exact_stationarity_folds_twohop`;
- `test_tier12_estep.py::test_mm_exact_monotone_filtered_f_descent`;
- `test_tier12_estep.py::test_twohop_zero_is_byte_identical`;
- `test_tier12_estep.py::test_backprop_last_truncates_transport_gradient_to_phi`;
- `test_omega_tilde_model_frame.py::test_phi_tilde_mm_exact_device_smoke`;
- `test_tier12_attention.py::test_query_adaptive_tau_monotone_detached_and_c0_inert`;
- `test_tier12_attention.py::test_twohop_term_matches_hand_computation`;
- `test_tier12_decode.py::test_expected_likelihood_decode_matches_naive_dense`;
- `test_tier12_decode.py::test_z_loss_full_chunked_matches_dense_lse`;
- `test_divergence.py::test_safe_kl_clamp_bounds_and_nan`;
- `test_free_energy.py::test_free_energy_entropy_exact_for_deep_finite_prior`;
- `test_retraction.py::test_full_retraction_stays_spd`.

The CUDA command is serial and selects `-m cuda`. Before CUDA collection, the test harness fixes `CUBLAS_WORKSPACE_CONFIG=:4096:8`, enables deterministic algorithms, disables cuDNN benchmarking, and disables TF32 for CUDA matmul and cuDNN. The shared fixture resolves `cuda:0` and other indexed CUDA requests by device type rather than exact string equality. This policy protects float32 golden comparisons on the RTX 5090. It does not promise bit identity between CPU and CUDA. CUDA verification must not run while another training process owns the GPU.

## Coverage and failure behavior

The retained semantic suite is the union of the twelve-worker CPU fast lane, the three-node CPU slow lane, the serial CUDA-only/mirror lane, and the external-bundle lane when its prerequisites exist. The eleven deleted UMAP nodes are not replaced and are not described as covered. Line and branch coverage may decrease only where those deleted nodes were the sole callers; the final report records the measured delta without disguising it.

CPU lane failures stop the click-to-run driver immediately. CUDA absence or a busy GPU prevents a CUDA result from being claimed, but does not invalidate a completed CPU result. Machine-readable JUnit is authoritative for counts. Task-owned XML, coverage, cache, and temporary artifacts remain outside the commit and are deleted during closeout.

## Verification

Implementation follows red-green cycles for policy semantics, CPU command construction, canonical CUDA selection, and the newly device-routed numerical contracts. Verification then runs policy/runner tests, affected test modules, the complete twelve-worker fast CPU lane, the retained slow lane, branch coverage, and the serial CUDA lane only after the RTX 5090 becomes idle. An independent reviewer checks the complete branch before merge. The dated post-edit record reports baseline and final JUnit counts, elapsed times, worker counts, coverage deltas, CUDA availability, and any external prerequisite skip.
