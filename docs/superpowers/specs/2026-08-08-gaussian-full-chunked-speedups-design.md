# Gaussian-Full Chunked Decoder Speedups Design

Date: 2026-08-08
Approved basis: `docs/audits/gaussian-full-chunked-performance-2026-08-08.md` and the user's instruction to begin fixes.

## Goal

Remove the measured canonical `family_chunked` performance cliff and close the registered-logits workspace safety gap without changing noncanonical family/divergence semantics or the user's training configuration.

## Scope

This first implementation wave contains two independently reviewable changes:

1. Route the exact built-in `gaussian_full` + built-in Rényi + `alpha == 1.0` + active fp32 precision seam from `family_chunked` to the existing analytic `full_chunked` scorer for fused CE and registered logits.
2. Apply the full-family workspace ceiling to registered `family_chunked` logits and size full-family workspace/checkpoint estimates for the largest dtype the functional may use.

The first wave does not implement noncanonical sufficient-statistic kernels, a custom fused CUDA CE kernel, shape autotuning, full-path caching, or configuration changes.

## Canonical fast-path boundary

A private predicate in `vfe3/model/prior_bank.py` compares the resolved registry objects with the original built-in `FullGaussian` class and `renyi` callable identities. The fast path is enabled only when:

- the currently registered family object is the captured built-in `gaussian_full` class;
- the currently registered divergence object is the captured built-in `renyi` callable;
- `renyi_order == 1.0` exactly;
- `full_cov_kl_precision == "fp32_escalate"`;
- `decode_av_precision == "fp32"`.
- query mean, query covariance, decode tables, and decode scale all have one homogeneous public dtype in `{torch.float32, torch.float64}`.

The precision conditions deliberately limit this first speed path to the benchmarked active configuration. They prevent a caller that selected unconditional fp64 or condition-triggered fp64 recomputation from silently receiving the analytic scorer's different numerical policy. The homogeneous-dtype condition preserves a currently supported generic mixed-dtype call that the analytic scorer rejects. Later work may extend the analytic scorer with explicit precision and mixed-dtype parity.

For fused CE, `decode_ce_family_chunked` delegates before it constructs generic family state. It forwards `targets`, `z_loss_weight`, `tau`, `chunk_size`, and `ignore_index` unchanged. For registered logits, only `_decode_family_chunked` delegates to `_decode_full_chunked`; dense `family` and `reference_decode` remain generic oracles. The shared outer `PriorBank.decode` seam continues to add unigram bias exactly once.

## Workspace safety

`_decode_ce_family_effective_chunk` remains the shared sizing helper. For the built-in `FullGaussian`, callers pass a scalar-byte cost derived at call time from the family-owned full-covariance precision policy, conservatively including any fp64 route. Other families retain the reference tensor's element size because the functional registry currently exposes no trustworthy workspace metadata. Diagonal families remain byte-identical and retain the requested chunk.

The fused family CE and registered `family_chunked` logits both derive their effective width from this helper. The whole-vocabulary checkpoint estimate uses the same conservative element size. Dense `family` remains intentionally unchunked.

This closes the confirmed dtype undercount for the built-in `FullGaussian`; it does not claim a universal hard bound for arbitrary custom functionals. A later registration-metadata change is required to encode function-specific workset counts, especially for composed functionals such as Jeffreys.

## Preserved behavior

- Noncanonical Rényi orders and all non-Rényi functionals remain on the generic family route.
- Runtime registry overrides under the names `gaussian_full` or `renyi` remain generic.
- Mixed-dtype inputs and non-active precision-policy combinations remain generic.
- Temperature, unigram bias, z-loss, ignore-index handling, degenerate-SPD exclusion/uniform logits, tied/untied decode banks, and remainder chunks keep their established seams.
- `reference_decode` remains the family/divergence oracle.
- No configuration values are changed.

## Tests

Tests must be written and observed failing before production edits.

- A behavior test wraps the real `torch.linalg.solve_triangular` operation and requires zero pair-grid solves for canonical fused CE and registered logits, while a noncanonical control requires a positive count.
- Paired `family_chunked` and `full_chunked` tests compare CE, logits, and gradients with explicit fp32 tolerances under z-loss, unigram bias, learned temperature, and a remainder chunk.
- Negative tests prove `alpha != 1`, non-Rényi functionals, and registry overrides still use the generic result.
- A registered-logits test forces a one-entry workspace budget and verifies the real functional receives a complete width-one tiling; current HEAD forwards the raw requested width.
- Workspace unit tests derive the full-family width using eight-byte elements and prove diagonal-family behavior remains unchanged.
- Existing family workspace tests are moved to a noncanonical Rényi order so they continue to exercise the generic route after canonical dispatch.
- Baseline and post-change CUDA-enabled targeted lanes write JUnit XML, followed by the relevant full-covariance roadmap tests and a synchronized live-shape decoder benchmark.

## Success criteria

- Canonical operation-count tests pass with zero pair-grid triangular solves and the noncanonical control records a positive count.
- Canonical value/gradient parity stays within declared fp32 tolerances.
- Generic-route negative tests and all pre-existing targeted tests pass.
- The logits workspace test observes no slice wider than the effective conservative ceiling.
- A synchronized RTX 5090 benchmark demonstrates the canonical `family_chunked` call now falls onto the analytic performance class.
