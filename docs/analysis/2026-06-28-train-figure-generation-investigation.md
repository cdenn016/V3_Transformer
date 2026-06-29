# Train Figure Generation Investigation

Date: 2026-06-28

Question: why the figures and plots are not visible at the end of the current `train_vfe3.py` run.

## Finding

The current K160 run has not reached the end of training. The run directory
`vfe3_runs/20260628-205936_wikitext-103_K160_block_glk_linear_mix_s6` has `config.json`,
`metrics.csv`, `best_model.pt`, `attention/`, and `checkpoints/`, but it does not yet have
`summary.json`, `test_results.json`, root PNGs, or `figures/`. That is the expected state before
`finalize_run` executes.

At 2026-06-28 22:05 local time, `metrics.csv` showed step 5200 of the configured 60000 steps.
The observed pace was about 79 steps per minute, leaving roughly 11.5 hours before finalization.
Since the end-of-run figure hook lives inside `finalize_run`, no final figures should exist until
after the run writes `summary.json` and `test_results.json`.

## Source Trace

`train_vfe3.py` constructs `RunArtifacts`, trains, then calls `finalize_run` only after
`train(...)` returns. `finalize_run` writes `test_results.json`, `summary.json`, research
artifacts, the history-only root figures, and then calls `vfe3.viz.report.generate_figures(...)`
when `cfg.generate_figures` is true. The saved config for the live K160 run has
`generate_figures=True`, so the hook is enabled for the run once it reaches finalization.

The completed control run `vfe3_runs/153.89_wikitext-103_K20_block_glk_linear_s54` has the expected
final artifact shape: 12 root PNGs and 31 files under `figures/`, including attention, belief UMAP,
vocabulary, sigma, holonomy, per-layer, and numerical-trust figures. That proves the current source
can produce the report-driver figures from a completed run.

## Verification

Focused finalize autorun checks were rerun on CPU, without touching the live CUDA process:

```
python -m pytest tests/test_report.py --junitxml=C:\tmp\vfe3-report-full-20260628.xml
```

Machine-readable result: `tests=8 failures=0 errors=0 skipped=0`. Console result:
`8 passed, 17 warnings in 49.36s`.

```
python -m pytest tests/test_run_artifacts.py --junitxml=C:\tmp\vfe3-run-artifacts-full-20260628.xml
```

Machine-readable result: `tests=13 failures=0 errors=0 skipped=0`. Console result:
`13 passed, 9 warnings in 39.49s`.

## Conclusion

No production-code bug was reproduced. The live run is still training, so its missing final figures
are expected. If the run later has `summary.json` and `test_results.json` but still lacks root PNGs
or `figures/*.png`, then the next step is to inspect the finalize log warnings and rerun
`vfe3.viz.report.generate_figures(run_dir)` on the completed run directory.
