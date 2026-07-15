# Controlled Semantic-Probe Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add preregistered, decoder-resolved semantic concept probes to controlled belief-geometry reports so expected semantic relationships can be evaluated quantitatively without treating a UMAP picture or a single HDBSCAN label as ground truth.

**Architecture:** A focused `vfe3.viz.semantic_probes` module owns the versioned English GPT-2 manifest, validation, token resolution, native-chart metrics, cluster co-membership, and explicit unavailable records. `plot_belief_umap` computes one semantic record from the same native features and controlled HDBSCAN labels already used by the sidecar, `embedding_comparison` serializes and compares it, and the comparison figure adds semantic metric panels without plotting independent UMAP coordinates.

**Tech Stack:** Python 3.10+, NumPy, scikit-learn, Matplotlib, pytest, JUnit XML.

## Global Constraints

Do not change training, objectives, model state, data caches, tokenizer selection, UMAP coordinates, HDBSCAN assignments, or user configuration dictionaries.

Score semantic structure only in the active native feature chart. UMAP remains display-only. HDBSCAN contributes only a noise-aware co-membership diagnostic.

Keep the probe manifest English and GPT-2-oriented. For non-English policy or a missing decoder, emit a stable unavailable record with a reason rather than a misleading score.

Resolve concepts by exact decoded single-token forms. Record accepted forms, resolved token IDs, decoded strings, and occurrence counts. Treat unresolved concepts and underdetermined metrics as JSON null values with explicit reasons; never serialize NaN or infinity.

Treat `close`, `control`, and `descriptive` as distinct expectation classes. The `descriptive` class, including `king` to `father`, must not contribute to close-versus-control claims.

Every test run used as verification must write JUnit XML, and reported counts must be read from that XML. Update `docs/2026-07-15-edits.md`; do not create another dated edit file.

## Task 1: Implement manifest validation and exact token resolution

**Files:**

- Create: `vfe3/viz/semantic_probes.py`
- Create: `tests/test_semantic_probes_20260715.py`

**Interfaces:**

- Produces: `DEFAULT_SEMANTIC_MANIFEST`, `validate_manifest(manifest)`, `resolve_concepts(token_ids, decode, manifest=...)`, and `unavailable_record(reason, manifest=...)`.
- Manifest contract: unique concepts, exactly one field per concept, known and distinct pair endpoints, and expectation values restricted to `close`, `control`, or `descriptive`.

- [ ] **Step 1: Write failing manifest and resolution tests**

Add tests that import the absent module, then pin the default manifest identity and schema version. Add malformed-manifest cases for duplicate concept names, duplicate field membership, unknown pair endpoints, identical endpoints, and invalid expectations. Add a fake decoder map that resolves both `" king"` and `"king"` to separate token IDs, resolves only the leading-space form of `queen`, and leaves one concept absent. Assert deterministic token resolution and explicit missing-concept reasons.

- [ ] **Step 2: Run the RED test and inspect JUnit**

```powershell
python -m pytest tests/test_semantic_probes_20260715.py --junitxml=C:\tmp\vfe3-semantic-probes-task1-red-20260715.xml
```

Expected: collection fails because `vfe3.viz.semantic_probes` does not yet exist.

- [ ] **Step 3: Implement the minimal manifest layer**

Define one immutable-by-convention JSON-compatible manifest containing the semantic fields royalty, kinship, animals, colors, spatial terms, emotions, motion verbs, and cognition verbs. Include close pairs such as `king` to `queen` and `father` to `mother`, unrelated controls, and descriptive pairs such as `king` to `father`. Validate all manifest references before resolution.

Decode each unique token ID once, compare its decoded text to the exact accepted forms, and aggregate every matching contextual occurrence by concept. Return deterministic concept order, resolved IDs, decoded forms, occurrence row indices, and null-valued missing records.

- [ ] **Step 4: Run the Task 1 GREEN test and inspect JUnit**

```powershell
python -m pytest tests/test_semantic_probes_20260715.py -k "manifest or resolve or unavailable" --junitxml=C:\tmp\vfe3-semantic-probes-task1-green-20260715.xml
```

Expected: zero failures and zero errors.

- [ ] **Step 5: Commit the manifest and resolution layer**

```powershell
git add vfe3/viz/semantic_probes.py tests/test_semantic_probes_20260715.py
git commit -m "feat(viz): define semantic probe manifest"
```

## Task 2: Compute native-space semantic and cluster diagnostics

**Files:**

- Modify: `vfe3/viz/semantic_probes.py`
- Modify: `tests/test_semantic_probes_20260715.py`

**Interfaces:**

- Produces: `evaluate_semantic_probes(features, token_ids, cluster_labels, decode, manifest=..., k=5) -> dict`.
- Pair metrics: centroid distance, empirical distance percentile, directional partner ranks, mean reciprocal rank, hit at five, and noise-aware HDBSCAN co-membership.
- Field metrics: resolved concept count, occurrence count, within-field distance, between-field distance, and between-to-within ratio.
- Aggregate metrics: semantic-field silhouette, semantic-field adjusted mutual information against non-noise clusters, expectation-specific summaries, and control-to-close distance separation.

- [ ] **Step 1: Write failing deterministic metric tests**

Use a small hand-constructed feature matrix with two semantic fields, repeated contextual occurrences, two non-noise clusters, and noise. Assert that concept centroids average occurrences rather than token IDs, closer partners rank ahead of controls, hit at five and reciprocal rank follow the resolved concept population, and shared noise never counts as co-membership. Add a second case with a missing `queen` concept and assert null pair metrics with the exact resolution reason.

- [ ] **Step 2: Run the metric RED test and inspect JUnit**

```powershell
python -m pytest tests/test_semantic_probes_20260715.py -k "centroid or pair or field or aggregate or missing" --junitxml=C:\tmp\vfe3-semantic-probes-task2-red-20260715.xml
```

Expected: failures identify the missing evaluator and metric schema.

- [ ] **Step 3: Implement native-chart metrics**

Validate that features are finite and two-dimensional and that token and cluster arrays align. Compute one equal-weight concept centroid per resolved concept. Use all resolved concept pairs as the empirical distance reference, rank each endpoint against the other resolved concepts, and summarize only pairs from the requested expectation class.

For co-membership, divide each concept's count in every non-noise cluster by its total occurrence count and take the dot product of the two distributions. This makes noise reduce coverage but never become a shared cluster. Compute semantic-field silhouette over concept centroids and occurrence-level adjusted mutual information after excluding HDBSCAN noise. Return null metrics with reasons whenever sample or label structure is insufficient.

- [ ] **Step 4: Prove descriptive pairs are excluded from confirmatory summaries**

Add a regression test that moves the descriptive `king` to `father` pair arbitrarily close and asserts that close-pair and close-versus-control aggregates do not change.

- [ ] **Step 5: Run the Task 2 GREEN test and inspect JUnit**

```powershell
python -m pytest tests/test_semantic_probes_20260715.py --junitxml=C:\tmp\vfe3-semantic-probes-task2-green-20260715.xml
```

Expected: zero failures and zero errors.

- [ ] **Step 6: Commit the semantic evaluator**

```powershell
git add vfe3/viz/semantic_probes.py tests/test_semantic_probes_20260715.py
git commit -m "feat(viz): score native semantic probes"
```

## Task 3: Add semantic records to controlled sidecars

**Files:**

- Modify: `vfe3/viz/embedding_comparison.py`
- Modify: `vfe3/viz/figures.py`
- Modify: `tests/test_controlled_umap_comparison_20260714.py`

**Interfaces:**

- Extends: `controlled_embedding_record(..., semantic_probes: Optional[Mapping[str, object]] = None) -> dict`.
- Extends behavior: `plot_belief_umap` evaluates semantics from native `features`, bank token IDs, controlled HDBSCAN labels, and the existing decoder.
- Sidecar contract: `semantic_probes.manifest.name` and `semantic_probes.manifest.schema_version` participate in comparison compatibility.

- [ ] **Step 1: Write failing sidecar integration tests**

Add a direct controlled-record test that supplies a semantic record and asserts exact round-trip serialization. Extend the controlled plotting test with a fake semantic decoder and assert that the sidecar contains a resolved concept, native pair metrics, and cluster co-membership. Add cases showing that `english_linguistic_diagnostics=False` and `decode=None` produce explicit unavailable semantic records.

- [ ] **Step 2: Run the sidecar RED test and inspect JUnit**

```powershell
python -m pytest tests/test_controlled_umap_comparison_20260714.py -k "semantic" --junitxml=C:\tmp\vfe3-semantic-probes-task3-red-20260715.xml
```

Expected: failures identify the absent controlled-record argument and sidecar section.

- [ ] **Step 3: Wire the evaluator into controlled plotting**

In the controlled path, evaluate semantics only when English linguistic diagnostics are enabled and a decoder is present. Otherwise create an unavailable record with the dataset-policy or decoder reason. Pass the record into `controlled_embedding_record`; do not use display coordinates and do not alter cluster assignment.

When callers omit the optional semantic record, serialize a stable unavailable section rather than omitting the contract. Preserve the existing atomic image-plus-sidecar behavior and strict JSON validation.

- [ ] **Step 4: Run controlled sidecar and visualization GREEN tests**

```powershell
python -m pytest tests/test_semantic_probes_20260715.py tests/test_controlled_umap_comparison_20260714.py tests/test_viz.py --junitxml=C:\tmp\vfe3-semantic-probes-task3-green-20260715.xml
```

Expected: zero failures and zero errors.

- [ ] **Step 5: Commit controlled sidecar integration**

```powershell
git add vfe3/viz/embedding_comparison.py vfe3/viz/figures.py tests/test_controlled_umap_comparison_20260714.py
git commit -m "feat(viz): record controlled semantic probes"
```

## Task 4: Compare semantic metrics across controlled runs

**Files:**

- Modify: `vfe3/viz/embedding_comparison.py`
- Modify: `vfe3/viz/figures.py`
- Modify: `tests/test_controlled_umap_comparison_20260714.py`

**Interfaces:**

- Comparison JSON adds semantic-field silhouette, semantic adjusted mutual information, close-pair mean distance percentile, mean reciprocal rank, hit-at-five rate, mean co-membership, and control-to-close distance separation.
- Comparison validation rejects manifest name or schema-version drift.
- `plot_controlled_embedding_comparison` expands from four to six metric panels and continues to contain no UMAP axes.

- [ ] **Step 1: Write failing comparison-contract and six-panel tests**

Add a manifest-drift test that changes only the semantic manifest version and asserts that validation names the mismatched contract field. Extend the comparison JSON assertion to pin all semantic summary keys. Update the comparison figure assertion to require six axes and titles describing metrics rather than UMAP coordinates.

- [ ] **Step 2: Run the comparison RED test and inspect JUnit**

```powershell
python -m pytest tests/test_controlled_umap_comparison_20260714.py -k "comparison or manifest or semantic" --junitxml=C:\tmp\vfe3-semantic-probes-task4-red-20260715.xml
```

Expected: failures show missing semantic fields, missing manifest validation, and the four-panel layout.

- [ ] **Step 3: Extend controlled comparison extraction and rendering**

Add semantic manifest identity to the comparison contract paths. Extract every semantic aggregate through the existing nullable metric mechanism. Expand the figure to a three-by-two grid: preserve the four current panels, add semantic-field silhouette and semantic adjusted mutual information in one panel, and add close-pair reciprocal rank, hit at five, and co-membership in the other. Keep distance percentile and close-versus-control separation in JSON even when their scales make them unsuitable for that shared axis.

- [ ] **Step 4: Run comparison and report GREEN tests**

```powershell
python -m pytest tests/test_controlled_umap_comparison_20260714.py tests/test_report.py tests/test_reporting_additions.py --junitxml=C:\tmp\vfe3-semantic-probes-task4-green-20260715.xml
```

Expected: zero failures and zero errors.

- [ ] **Step 5: Commit cross-run semantic comparison**

```powershell
git add vfe3/viz/embedding_comparison.py vfe3/viz/figures.py tests/test_controlled_umap_comparison_20260714.py
git commit -m "feat(viz): compare semantic probe metrics"
```

## Task 5: Verify, document, review, and complete the repository lifecycle

**Files:**

- Modify: `docs/2026-07-15-edits.md`
- Test artifacts: task-owned JUnit XML files under `C:\tmp`, removed after their counts are recorded.

- [ ] **Step 1: Run the focused semantic and controlled-report suite**

```powershell
python -m pytest tests/test_semantic_probes_20260715.py tests/test_controlled_umap_comparison_20260714.py tests/test_viz.py tests/test_report.py tests/test_reporting_additions.py --junitxml=C:\tmp\vfe3-semantic-probes-focused-20260715.xml
```

Expected: zero failures and zero errors in the XML.

- [ ] **Step 2: Run static and diff checks**

```powershell
python -m ruff check vfe3/viz/semantic_probes.py vfe3/viz/embedding_comparison.py vfe3/viz/figures.py tests/test_semantic_probes_20260715.py tests/test_controlled_umap_comparison_20260714.py
git diff --check origin/main...HEAD
```

Expected: both commands exit zero.

- [ ] **Step 3: Run the full repository suite**

```powershell
python -m pytest --junitxml=C:\tmp\vfe3-semantic-probes-full-20260715.xml
```

Expected: zero failures and zero errors in the XML.

- [ ] **Step 4: Update the dated edit record from machine-readable results**

Record the semantic manifest, exact-resolution behavior, native pair and field metrics, HDBSCAN co-membership definition, unavailable-policy behavior, controlled comparison fields, figure expansion, and exact focused/full JUnit counts in `docs/2026-07-15-edits.md`.

- [ ] **Step 5: Inspect and commit final documentation**

Run `git status --short`, `git diff --check`, and inspect the staged diff before committing every intended file. Remove task-owned temporary artifacts, then commit the dated record with `git commit -m "docs: record semantic probe verification"`.

- [ ] **Step 6: Complete the mandatory git lifecycle**

Fetch `origin`, compare against current `origin/main`, address any integration conflict without touching live WIP, push `codex/semantic-probes-20260715`, fast-forward `origin/main` from the verified task branch, and confirm the remote SHA. Fast-forward the live `main` checkout only if Git can preserve all pre-existing WIP. Remove the temporary worktree and local task branch, then report the task commits, pushed branch, resulting `origin/main` SHA, JUnit counts, cleanup result, and the live checkout's exact `git status --short` with ownership of remaining changes.
