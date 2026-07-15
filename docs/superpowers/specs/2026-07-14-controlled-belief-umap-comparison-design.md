# Controlled Belief-Geometry Comparison Design

## Status and decision

This design repairs the belief/model UMAP reporting path so that figures generated for different sequence lengths can support a controlled comparison. The existing adaptive plot remains available as an explicitly exploratory view. Production run finalization instead uses a fixed-token comparison contract and writes machine-readable diagnostics beside every UMAP image.

The user approved the full repair after the N=128, N=256, and N=512 artifacts showed visually cleaner UMAP islands at larger N while test perplexity worsened slightly. No claim that longer context improves semantic representation quality is retained unless it survives the controlled comparison metrics defined below.

## Root cause

`generate_figures` currently caps the bank at 64 sequences. Consequently, the default bank contains 8,192 tokens at N=128, 16,384 at N=256, and 32,768 at N=512. `plot_belief_umap` then chooses display `n_neighbors` from the square root of that changing token count, while `_cluster_embedding` chooses `min_cluster_size` from the same quantity. For channels wider than ten dimensions, cluster labels are produced by HDBSCAN after a second, independently fitted ten-dimensional UMAP. The sample, display neighborhood, density threshold, and stochastic clustering coordinates therefore all change with N.

The result is valid for exploratory inspection of one run but does not identify a cross-run change in native belief geometry. The PNG footer discloses the settings, but the title still presents cluster counts and unclustered fractions without distinguishing exploratory from comparison-safe results.

## Approaches considered

The smallest approach would replace the current defaults with one fixed token count and fixed UMAP parameters. This would make finalized runs more comparable, but it would remove the adaptive behavior that remains useful for direct calls on small or unusually large banks.

A second approach would keep every current figure unchanged and add an external analysis script for this one sequence-length sweep. That preserves compatibility but duplicates feature geometry, sampling, clustering, and report conventions outside the registered visualization path. Future sweeps could silently return to the same comparison error.

The selected approach separates two contracts inside the existing reporting system. Exploratory mode retains adaptive rendering for arbitrary banks. Comparison mode fixes the sample and parameters, moves cluster discovery out of stochastic UMAP coordinates, persists diagnostics, and validates comparability across sidecars. This adds one reusable comparison seam without changing the model or training path.

## Sampling contract

Run finalization will default to a deterministic bank of exactly 16,384 validation tokens rather than 64 sequences. The unshuffled validation stream will be consumed from its beginning, with whole batches collected and every bank tensor sliced to the exact token cap. N=128 will therefore use 128 sequences, N=256 will use 64, and N=512 will use 32, while the selected flattened input-token stream remains identical when the underlying dataset and tokenizer are identical.

`belief_bank` and `model_channel_bank` will accept an optional `max_tokens` cap in addition to the existing `max_sequences` compatibility argument. They will record the within-sequence position of every state. `generate_figures` will compute the necessary number of batches from `batch_size * max_seq_len` when a token cap is active. A SHA-256 fingerprint of the selected token IDs will be written to each sidecar. The cross-run comparator will refuse to call two arms comparable if token count, token fingerprint, channel geometry, display settings, or clustering settings differ.

## Display and clustering

Comparison-mode display UMAP uses `n_neighbors=32`, `min_dist=0.1`, PCA initialization, and seeds 0 through 4. Seed zero produces the displayed coordinates. All seeds contribute quantitative trustworthiness and neighborhood-stability summaries. Exploratory mode retains the adaptive display-neighborhood rule and one requested seed.

HDBSCAN will no longer discover clusters in a stochastic UMAP embedding when comparison mode is active. Channels of at most ten dimensions will be clustered in their native feature chart. Wider channels will use deterministic PCA to ten dimensions, fitted only on the selected bank. The feature chart remains channel-specific: Euclidean means, log-Euclidean covariance coordinates, and stored gauge coordinates. HDBSCAN parameters are explicit and fixed under the equal-token comparison contract. The 2-D UMAP remains a display, not the source of cluster membership.

The plot title will identify either `exploratory clusters` or `controlled clusters`. Its footer will state the sampling mode, exact token count, token fingerprint prefix, display parameters, clustering space, and gauge-coordinate caveat.

## Quantitative diagnostics

Each `belief_umap_<channel>.png` and `model_umap_<channel>.png` will have a same-stem JSON sidecar. The record will contain the comparison contract and the following measurements:

* native-space silhouette for BPE and function/content taxonomies;
* UMAP trustworthiness for every requested seed, plus mean and standard deviation;
* mean k-nearest-neighbor overlap between the seed-zero embedding and the other embeddings;
* HDBSCAN cluster count and noise fraction;
* adjusted mutual information between cluster membership and token category, position bin, and sequence identity;
* sample size, sequence count, token fingerprint, channel feature dimension, feature chart, display parameters, clustering parameters, and random seeds.

Noise labels remain part of the cluster assignment for the reported noise fraction but are excluded from adjusted-mutual-information calculations that would otherwise let one large noise class dominate the score. Metrics that cannot be computed because a label is constant or a bank is too small are stored as null with a reason rather than silently omitted.

The sequence-identity score is diagnostic only. Sequence identifiers differ when N changes, but a high within-run value exposes a representation organized around individual sampled sequences. Position bins are normalized to sequence-relative quartiles so that their labels remain comparable across N.

## Cross-run comparison artifact

A report helper will read two or more UMAP sidecars, validate their comparison contract, and write a compact comparison JSON plus a multi-panel metric figure. It will not overlay independently fitted UMAP coordinates or imply that their axes are shared. The figure will compare native silhouettes, trustworthiness, cluster noise, and adjusted mutual information with token category, relative position, and sequence identity.

Raw mean, covariance, and frame coordinates are gauge-fixed diagnostics, not gauge-invariant observables. The sidecar and figure will say so. The comparator will compare within-arm metrics but will not pool raw coordinates across independently trained models. A future pooled-coordinate view would require an explicit frame-alignment or gauge-invariant feature contract and is outside this repair.

## Compatibility and failure behavior

Direct callers of `plot_belief_umap` retain exploratory behavior unless they request the controlled contract. Existing `max_sequences` callers remain valid. Supplying both `max_tokens` and `max_sequences` is an error because the selected population would otherwise be ambiguous.

Run finalization treats a failed UMAP image as best-effort, as it does today. A comparison sidecar is atomic with its image: if required diagnostics fail, neither artifact is presented as controlled. The cross-run comparator fails closed with a message naming every mismatched contract field.

## Tests and acceptance criteria

Tests will first reproduce the defect by showing that the current sequence cap yields different token counts and adaptive parameters at N=128 and N=512. The implementation will then be accepted when deterministic synthetic banks demonstrate all of the following behavior.

The bank extractors return the exact requested token count and aligned `mu`, `sigma`, `phi`, `token_ids`, `seq_idx`, and position tensors. The N=128 and N=512 report paths request the same 16,384-token comparison population. Controlled plots use fixed display and HDBSCAN parameters; exploratory plots retain adaptive parameters. Clustering coordinates are native or PCA, never UMAP, in controlled mode. Five seeded embeddings produce finite trustworthiness and neighborhood-stability summaries. Every controlled PNG has a schema-valid JSON sidecar. Cross-run comparison accepts matching contracts, rejects token-fingerprint and parameter mismatches, and never overlays independent coordinates. Existing visualization, model-channel, report, and pure-path tests remain green.

No training configuration, objective, checkpoint, or run result will be modified. Historical PNGs remain historical; controlled figures can be regenerated from their saved checkpoints after the code lands.
