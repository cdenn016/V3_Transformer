# Controlled Semantic-Probe Evaluation Design

## Status and decision

This design extends the controlled belief-geometry reporting path with preregistered English semantic probes. It answers whether expected lexical relations appear in the learned belief charts without treating a two-dimensional UMAP layout or one HDBSCAN partition as semantic ground truth.

The user approved the direction after reviewing the current V3 reporting contract and the older VFE_2.0 semantic analysis. VFE_2.0 established the useful idea of known semantic fields and word-pair checks, but its aggregate related-versus-unrelated calculation was invalid because the test `"related" in relation` also matches `"unrelated"`. Its bare-word token lookup also missed common GPT-2 leading-space tokens and split words such as bare `queen`. The V3 implementation therefore reuses the idea, not that code.

## Scope

The feature is diagnostic-only. It changes no model, training objective, checkpoint, dataset, tokenizer, clustering parameters, or UMAP coordinates. It operates on the same exact-token controlled bank and channel features already passed to `controlled_embedding_record`.

The initial probe manifest is English and GPT-2-oriented. It is active only when the existing `english_linguistic_diagnostics` policy is active and a decoder is available. Japanese, Arabic, synthetic-token, and decoder-unavailable paths retain their current figures and receive an explicit unavailable record with a reason. Multi-token composition is outside the initial scope; every accepted surface form must resolve to one decoded token ID present in the bank.

The initial channels are the existing belief and model-channel charts: Euclidean means, log-Euclidean covariance coordinates, and stored gauge coordinates. Their semantic results are within-run, gauge-fixed coordinate diagnostics. Raw coordinates and relation vectors are not pooled across independently trained models.

## Approaches considered

A plot-only approach would label a small list of known words directly on each UMAP. This is visually immediate but gives a stochastic projection too much evidential weight, creates label clutter, and cannot distinguish semantic distance from projection artifacts.

A separate analysis script would reproduce the VFE_2.0 organization. It could evolve independently, but it would duplicate token selection, feature-chart construction, HDBSCAN labels, multilingual policy, and sidecar comparison rules. Run finalization could then publish a controlled UMAP without the semantic check.

The selected approach adds one focused semantic-probe module behind the existing controlled sidecar. Detailed concept, group, pair, and cluster-agreement results live in each same-stem JSON file. The existing cross-run comparison JSON and figure gain aggregate semantic panels. The UMAP itself remains an uncluttered display and is never the source of a semantic distance.

## Probe manifest

The manifest is immutable module data with an explicit schema version. A concept contains a stable name and one or more exact decoded single-token surface forms, including leading-space and sentence-initial variants when both exist. For example, `king` can accept `" king"` and `"king"`, while `queen` accepts `" queen"` because bare GPT-2 `queen` is not one token.

Semantic fields cover a small balanced set with clear lexical interpretations: royalty, kinship, animals, colors, spatial terms, emotions, motion verbs, and cognition verbs. Each field has at least four concepts in the manifest, but a field contributes to aggregate metrics only when at least two concepts occur in the selected bank.

Pair probes use an exact expectation enum, never substring logic:

* `close` for preregistered co-hyponyms, gender-role parallels, morphological or functional neighbors, and distributionally related contrasts;
* `control` for unrelated cross-field comparisons;
* `descriptive` for relationships that are plausible only under a broad feature, such as `king` and `father`, and therefore must be reported without entering the positive or negative aggregate.

Every pair also names its relation type. `king` and `queen` are a close royalty/gender-role pair; `father` and `mother` are the parallel kinship pair; `king` and `father` are a descriptive broad-role association rather than a required same-cluster pair.

## Resolution and type aggregation

The resolver decodes each unique bank token ID once and matches exact strings against the manifest. A concept may resolve to several token IDs when sentence-initial and leading-space forms both occur. The sidecar records every accepted form, matched token ID, decoded string, and occurrence count. Missing concepts remain in the record with a null reason.

The bank contains contextual token occurrences, not one static vector per word type. Semantic distance therefore uses one centroid per resolved concept in the active native feature chart. All resolved concepts receive equal weight in field and pair summaries regardless of corpus frequency. Occurrence-level HDBSCAN labels remain available for cluster co-membership and semantic-label agreement.

## Native-space metrics

For each resolved pair, the record contains the Euclidean distance between concept centroids in the active feature chart, its percentile among all resolved concept-centroid pairs, the partner's nearest-neighbor rank from each endpoint, reciprocal-rank mean, hit-at-five, and HDBSCAN co-membership probability. Noise never counts as shared cluster membership.

Co-membership is computed from each concept's empirical non-noise cluster distribution. The probability is the dot product of the two distributions multiplied by each concept's non-noise coverage, so unresolved occurrences and HDBSCAN noise cannot inflate agreement.

For each resolved semantic field, the record contains concept count, occurrence count, mean within-field centroid distance, mean distance from the field concepts to concepts in other fields, and the between-to-within ratio. The aggregate record contains semantic-field silhouette on concept centroids, adjusted mutual information between occurrence-level semantic-field labels and non-noise HDBSCAN labels, and separate summaries for `close`, `control`, and `descriptive` pairs. A close-versus-control separation ratio is reported only when both aggregates contain at least one resolved pair.

Pair distance percentiles, field silhouette, and field coherence use native chart coordinates. HDBSCAN co-membership and adjusted mutual information use the controlled native-or-PCA cluster labels. UMAP coordinates are absent from every semantic metric.

## Nulls and failure behavior

Expected data scarcity is not an exception. Unresolved concepts, fields with fewer than two resolved concepts, pairs with a missing endpoint, constant labels, and insufficient cluster coverage produce `{value: null, reason: ...}` records. JSON serialization remains strict and contains no NaN or infinity.

Manifest errors are programmer errors and fail tests: duplicate concept names, duplicate field membership, unknown pair endpoints, identical pair endpoints, or an expectation outside the exact enum raise `ValueError`. A semantic diagnostic failure during controlled rendering is atomic with the controlled sidecar. The UMAP image is not presented as controlled if its required sidecar cannot be completed.

When English diagnostics are disabled or the decoder is unavailable, the sidecar still contains a stable `semantic_probes` schema with `available=false` and the policy reason. This preserves cross-run schema shape without inventing English semantics for other languages.

## Sidecar and comparison contract

Each controlled sidecar gains a `semantic_probes` block containing the manifest identity, resolution audit, native-space metrics, cluster agreement, and aggregate pair summaries. The semantic manifest name and schema version become comparison-contract fields so cross-run synthesis fails closed if two arms used different preregistrations.

The compact cross-run comparison JSON adds semantic-field silhouette, semantic AMI, close-pair mean distance percentile, close-pair mean reciprocal rank, close-pair hit-at-five, close-pair co-membership, and close-versus-control distance separation. Unavailable values remain null.

The comparison figure expands from four to six panels. The two new panels show semantic field structure and pair-probe retrieval/co-membership. Independently fitted UMAP axes remain absent. Detailed per-concept and per-pair results remain in the sidecars rather than overloading the figure.

## Components and interfaces

`vfe3/viz/semantic_probes.py` owns the manifest, validation, token resolution, concept centroids, native-space pair and field metrics, cluster agreement, and unavailable-record construction. It accepts NumPy-compatible features, token IDs, controlled cluster labels, and a decoder callback; it returns a strict-JSON-compatible dictionary.

`vfe3/viz/embedding_comparison.py` remains responsible for the complete controlled sidecar and cross-run scalar extraction. Its `controlled_embedding_record` receives an optional semantic-probe record produced by the focused module and stores it without duplicating semantic calculations.

`vfe3/viz/figures.py` resolves semantic probes from the already available bank, decoder, features, and controlled HDBSCAN labels. It passes the result into the controlled record and expands the cross-run metric figure. Exploratory UMAP behavior is unchanged.

`vfe3/viz/report.py` continues to route the dataset-derived English diagnostics policy and decoder. No new configuration knob is added.

## Testing and acceptance criteria

Tests use deterministic synthetic feature banks with known tight fields, controls, repeated contextual occurrences, explicit HDBSCAN noise, and GPT-2-style decoded strings. The implementation is accepted when the following are demonstrated through red-green test cycles.

The manifest validates exact expectation values and rejects malformed references. Leading-space and sentence-initial forms resolve without merging unrelated decoded strings. Missing `queen` forms are reported rather than composed from two tokens. Contextual occurrences aggregate to one equal-weight concept centroid. Native pair ranks, distance percentiles, field coherence, silhouette, semantic AMI, and noise-aware co-membership match hand-computed fixtures. Descriptive `king`-`father` results are emitted but excluded from close/control aggregates. Non-English and decoder-unavailable records are explicit null-bearing schemas. Controlled sidecars contain the semantic block, comparison validation detects manifest drift, and the six-panel comparison figure contains metrics but no UMAP-coordinate overlay.

Focused semantic and controlled-UMAP tests, visualization/report regressions, `git diff --check`, and the full pytest suite must pass. Exact test totals come from JUnit XML. The required `docs/2026-07-15-edits.md` record is updated throughout implementation.
